#!/usr/bin/env python3
"""
z1102: GPU Telemetry as Physical Reservoir

Key insight from literature: Physical Reservoir Computing (PRC) uses the natural
dynamics of physical systems as neural network computation. The GPU's thermal/power
dynamics have ~1-10 second time constants - these ARE a reservoir.

Instead of:
    hidden_state = transformer(x) + energy_prediction(hidden_state)  # FAILS

We try:
    reservoir_state = GPU_telemetry_history  # 64 timesteps × 8 sensors = 512 dim
    output = readout(concat(transformer(x), reservoir_state))

The reservoir is "free" - it's already happening! We just need to read it.

Claims to validate:
1. Reservoir state carries temporal information (autocorrelation > 0)
2. Reservoir helps prediction (accuracy with reservoir > without)
3. Reservoir state changes with workload (intervention test)
"""

import json
import time
import sys
import math
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import deque
import threading

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.telemetry.sysfs_hwmon import SysfsHwmonTelemetry

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class Config:
    # Model
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 256
    vocab_size: int = 256
    max_seq_len: int = 64

    # Reservoir
    reservoir_window: int = 32  # timesteps of telemetry history
    telemetry_dim: int = 6      # power, temp_edge, temp_junction, freq, gpu_busy, mem_used
    reservoir_dim: int = 192    # reservoir_window * telemetry_dim

    # Training
    n_epochs: int = 8
    batch_size: int = 32
    lr: float = 1e-3

    # Validation
    n_seeds: int = 3
    train_split: float = 0.8

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================================
# Physical Reservoir
# ============================================================================

class GPUReservoir:
    """
    Physical reservoir using GPU telemetry dynamics.

    The GPU's thermal mass, power delivery, and clock management create
    natural temporal dynamics. We treat these as a reservoir that transforms
    inputs over time.
    """

    def __init__(self, telemetry: SysfsHwmonTelemetry, window: int = 32, dim: int = 6):
        self.telemetry = telemetry
        self.window = window
        self.dim = dim
        self.history = deque(maxlen=window)

        # Running statistics for normalization
        self.ema_mean = torch.zeros(dim)
        self.ema_var = torch.ones(dim)
        self.ema_alpha = 0.1

        # Fill initial history
        for _ in range(window):
            self._sample()
            time.sleep(0.02)  # 50 Hz

    def _sample(self):
        """Sample current GPU state."""
        sample = self.telemetry.read_sample()
        vec = torch.tensor([
            sample.power_w / 100.0,           # Normalize power
            sample.temp_edge_c / 100.0,       # Normalize temp
            sample.temp_junction_c / 100.0,
            sample.freq_sclk_mhz / 2000.0,    # Normalize freq
            sample.gpu_busy_pct / 100.0,
            sample.vram_used_gb / 16.0        # Normalize VRAM
        ], dtype=torch.float32)

        # Update EMA statistics
        self.ema_mean = self.ema_alpha * vec + (1 - self.ema_alpha) * self.ema_mean
        self.ema_var = self.ema_alpha * (vec - self.ema_mean)**2 + (1 - self.ema_alpha) * self.ema_var

        # Normalize
        vec_norm = (vec - self.ema_mean) / (self.ema_var.sqrt() + 1e-6)

        self.history.append(vec_norm)

    def get_state(self) -> torch.Tensor:
        """Get flattened reservoir state."""
        self._sample()
        state = torch.stack(list(self.history), dim=0)  # [window, dim]
        return state.flatten()  # [window * dim]

    def get_state_2d(self) -> torch.Tensor:
        """Get reservoir state as 2D tensor for attention."""
        self._sample()
        return torch.stack(list(self.history), dim=0)  # [window, dim]


# ============================================================================
# Models
# ============================================================================

class SmallTransformer(nn.Module):
    """Baseline transformer without reservoir."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Parameter(torch.randn(1, cfg.max_seq_len, cfg.d_model) * 0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_ff,
            batch_first=True,
            activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)
        self.out = nn.Linear(cfg.d_model, cfg.vocab_size)

    def forward(self, x):
        B, T = x.shape
        h = self.embed(x) + self.pos[:, :T]
        h = self.encoder(h)
        return self.out(h)


class ReservoirTransformer(nn.Module):
    """
    Transformer with GPU telemetry reservoir.

    The reservoir state is concatenated with the transformer output
    before the final projection. Only the readout layer learns to
    use the reservoir - the reservoir dynamics are "free".
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Parameter(torch.randn(1, cfg.max_seq_len, cfg.d_model) * 0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_ff,
            batch_first=True,
            activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)

        # Reservoir readout: project reservoir to model dim
        self.reservoir_proj = nn.Sequential(
            nn.Linear(cfg.reservoir_dim, cfg.d_model),
            nn.ReLU(),
            nn.Linear(cfg.d_model, cfg.d_model)
        )

        # Gate to control reservoir influence
        self.reservoir_gate = nn.Linear(cfg.d_model * 2, 1)

        self.out = nn.Linear(cfg.d_model, cfg.vocab_size)

    def forward(self, x, reservoir_state: torch.Tensor = None):
        B, T = x.shape

        h = self.embed(x) + self.pos[:, :T]
        h = self.encoder(h)  # [B, T, D]

        if reservoir_state is not None:
            # Project reservoir to model dim
            r = self.reservoir_proj(reservoir_state)  # [D]
            r = r.unsqueeze(0).unsqueeze(0).expand(B, T, -1)  # [B, T, D]

            # Gated combination
            gate_input = torch.cat([h, r], dim=-1)
            gate = torch.sigmoid(self.reservoir_gate(gate_input))

            h = h + gate * r  # Residual with gate

        return self.out(h)


# ============================================================================
# Data
# ============================================================================

def load_data(cfg: Config):
    """Load TinyShakespeare."""
    data_path = Path(__file__).parent.parent / "data" / "input.txt"

    if not data_path.exists():
        import urllib.request
        data_path.parent.mkdir(exist_ok=True)
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        urllib.request.urlretrieve(url, data_path)

    text = data_path.read_text()
    data = torch.tensor([ord(c) % cfg.vocab_size for c in text], dtype=torch.long)

    n_seqs = len(data) // cfg.max_seq_len
    data = data[:n_seqs * cfg.max_seq_len].view(n_seqs, cfg.max_seq_len)

    n_train = int(len(data) * cfg.train_split)
    train_data, test_data = data[:n_train], data[n_train:]

    train_loader = DataLoader(
        TensorDataset(train_data[:, :-1], train_data[:, 1:]),
        batch_size=cfg.batch_size, shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(test_data[:, :-1], test_data[:, 1:]),
        batch_size=cfg.batch_size
    )

    return train_loader, test_loader


# ============================================================================
# Training
# ============================================================================

def train_baseline(model, loader, optimizer, cfg):
    """Train baseline model (no reservoir)."""
    model.train()
    total_loss, total_tokens = 0, 0

    for x, y in loader:
        x, y = x.to(cfg.device), y.to(cfg.device)

        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * x.numel()
        total_tokens += x.numel()

    return {'loss': total_loss / total_tokens, 'ppl': math.exp(min(total_loss / total_tokens, 10))}


def train_reservoir(model, loader, optimizer, cfg, reservoir: GPUReservoir):
    """Train reservoir model with live GPU state."""
    model.train()
    total_loss, total_tokens = 0, 0

    for x, y in loader:
        x, y = x.to(cfg.device), y.to(cfg.device)

        # Get LIVE reservoir state from GPU
        r_state = reservoir.get_state().to(cfg.device)

        optimizer.zero_grad()
        logits = model(x, r_state)
        loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * x.numel()
        total_tokens += x.numel()

    return {'loss': total_loss / total_tokens, 'ppl': math.exp(min(total_loss / total_tokens, 10))}


@torch.no_grad()
def evaluate(model, loader, cfg, reservoir=None):
    """Evaluate model."""
    model.eval()
    total_loss, total_tokens = 0, 0

    for x, y in loader:
        x, y = x.to(cfg.device), y.to(cfg.device)

        if reservoir:
            r_state = reservoir.get_state().to(cfg.device)
            logits = model(x, r_state)
        else:
            logits = model(x)

        loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), y.view(-1))
        total_loss += loss.item() * x.numel()
        total_tokens += x.numel()

    return {'loss': total_loss / total_tokens, 'ppl': math.exp(min(total_loss / total_tokens, 10))}


# ============================================================================
# Reservoir Analysis
# ============================================================================

def analyze_reservoir(reservoir: GPUReservoir, n_samples: int = 100):
    """Analyze reservoir dynamics."""
    print("\nAnalyzing reservoir dynamics...")

    states = []
    for i in range(n_samples):
        state = reservoir.get_state()
        states.append(state)
        time.sleep(0.05)

    states = torch.stack(states)  # [n_samples, reservoir_dim]

    # Temporal autocorrelation
    autocorr = []
    for lag in [1, 5, 10, 20]:
        if lag < n_samples:
            corr = torch.corrcoef(torch.stack([
                states[:-lag].flatten(),
                states[lag:].flatten()
            ]))[0, 1].item()
            autocorr.append((lag, corr))
            print(f"  Autocorr lag={lag}: {corr:.3f}")

    # Variance analysis
    var_per_dim = states.var(dim=0)
    print(f"  Variance: mean={var_per_dim.mean():.4f}, max={var_per_dim.max():.4f}")

    # Entropy estimate (discretized)
    n_bins = 10
    entropy_est = 0
    for i in range(states.shape[1]):
        hist = torch.histc(states[:, i], bins=n_bins, min=-3, max=3)
        probs = hist / hist.sum()
        probs = probs[probs > 0]
        entropy_est += -(probs * probs.log()).sum().item()
    entropy_est /= states.shape[1]
    print(f"  Entropy estimate: {entropy_est:.3f}")

    return {
        'autocorrelation': autocorr,
        'variance_mean': var_per_dim.mean().item(),
        'variance_max': var_per_dim.max().item(),
        'entropy_estimate': entropy_est
    }


def intervention_test(reservoir: GPUReservoir, cfg: Config):
    """Test if reservoir responds to workload changes."""
    print("\nRunning intervention test...")

    # Idle state
    time.sleep(1.0)
    idle_states = [reservoir.get_state() for _ in range(20)]
    idle_mean = torch.stack(idle_states).mean(dim=0)

    # Heavy workload
    x = torch.randn(64, 1024, 1024, device=cfg.device)
    for _ in range(50):
        x = torch.matmul(x, x.transpose(-1, -2))
        x = x / x.norm()

    # Measure under load
    load_states = [reservoir.get_state() for _ in range(20)]
    load_mean = torch.stack(load_states).mean(dim=0)

    # Difference
    diff = (load_mean - idle_mean).abs().mean().item()
    print(f"  Idle→Load state diff: {diff:.4f}")

    return {'idle_load_diff': diff}


# ============================================================================
# Main Experiment
# ============================================================================

def run_condition(condition: str, cfg: Config, train_loader, test_loader,
                  telemetry, seed: int):
    """Run one experimental condition."""
    torch.manual_seed(seed)

    if condition == "A_baseline":
        model = SmallTransformer(cfg).to(cfg.device)
        reservoir = None
    elif condition == "B_reservoir":
        model = ReservoirTransformer(cfg).to(cfg.device)
        reservoir = GPUReservoir(telemetry, cfg.reservoir_window, cfg.telemetry_dim)
    elif condition == "C_frozen_reservoir":
        # Reservoir but frozen (control for temporal information)
        model = ReservoirTransformer(cfg).to(cfg.device)
        reservoir = GPUReservoir(telemetry, cfg.reservoir_window, cfg.telemetry_dim)
        # Freeze to initial state
        frozen_state = reservoir.get_state()
        reservoir.get_state = lambda: frozen_state  # Always return same state
    else:
        raise ValueError(f"Unknown condition: {condition}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    # Training
    for epoch in range(cfg.n_epochs):
        if reservoir:
            train_reservoir(model, train_loader, optimizer, cfg, reservoir)
        else:
            train_baseline(model, train_loader, optimizer, cfg)

    # Evaluation
    test_metrics = evaluate(model, test_loader, cfg, reservoir)

    return {
        'test_ppl': test_metrics['ppl'],
        'test_loss': test_metrics['loss']
    }


def main():
    print("=" * 70)
    print("z1102: GPU TELEMETRY AS PHYSICAL RESERVOIR")
    print("=" * 70)
    print()
    print("Key insight: GPU thermal/power dynamics ARE a reservoir.")
    print("Instead of predicting energy, we USE energy dynamics AS computation.")
    print()
    print("Claims to validate:")
    print("  1. Reservoir has temporal structure (autocorrelation > 0)")
    print("  2. Reservoir helps prediction (PPL with < without)")
    print("  3. Reservoir responds to workload (intervention test)")
    print()

    cfg = Config()
    print(f"Config: {cfg.n_seeds} seeds, {cfg.n_epochs} epochs")
    print(f"Reservoir: {cfg.reservoir_window} timesteps × {cfg.telemetry_dim} dims = {cfg.reservoir_dim} dim")
    print()

    # Initialize telemetry
    try:
        telemetry = SysfsHwmonTelemetry()
        telemetry.start_continuous_sampling()
        time.sleep(1.0)
        print("✓ Telemetry initialized")
    except Exception as e:
        print(f"✗ Telemetry failed: {e}")
        return

    # Load data
    train_loader, test_loader = load_data(cfg)
    print(f"Data: {len(train_loader.dataset)} train, {len(test_loader.dataset)} test")

    # Analyze reservoir first
    reservoir_analysis = GPUReservoir(telemetry, cfg.reservoir_window, cfg.telemetry_dim)
    dynamics = analyze_reservoir(reservoir_analysis)
    intervention = intervention_test(reservoir_analysis, cfg)

    # Run experiments
    conditions = ["A_baseline", "B_reservoir", "C_frozen_reservoir"]
    results = {c: {'ppl': []} for c in conditions}

    for seed in range(cfg.n_seeds):
        print(f"\n{'='*70}")
        print(f"Seed {seed + 1}/{cfg.n_seeds}")
        print(f"{'='*70}")

        for condition in conditions:
            print(f"\n  Running {condition}...")
            start = time.time()

            res = run_condition(condition, cfg, train_loader, test_loader, telemetry, seed + 42)

            elapsed = time.time() - start
            print(f"    PPL: {res['test_ppl']:.3f}, Time: {elapsed:.1f}s")

            results[condition]['ppl'].append(res['test_ppl'])

    telemetry.stop_continuous_sampling()

    # Analysis
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)

    def mean_std(arr):
        m = sum(arr) / len(arr)
        s = (sum((x-m)**2 for x in arr) / len(arr)) ** 0.5
        return m, s

    for cond in conditions:
        m, s = mean_std(results[cond]['ppl'])
        print(f"\n{cond}: PPL = {m:.3f} ± {s:.3f}")

    # Claims validation
    print()
    print("=" * 70)
    print("CLAIMS VALIDATION")
    print("=" * 70)

    claims = []

    # Claim 1: Temporal structure
    autocorr_lag1 = dynamics['autocorrelation'][0][1] if dynamics['autocorrelation'] else 0
    c1_valid = autocorr_lag1 > 0.1
    claims.append({
        'claim': 'Reservoir has temporal structure (autocorr > 0.1)',
        'validated': bool(c1_valid),
        'evidence': f'autocorr_lag1 = {autocorr_lag1:.3f}'
    })
    print(f"\n1. Temporal structure: {'✓' if c1_valid else '✗'} (autocorr={autocorr_lag1:.3f})")

    # Claim 2: Reservoir helps
    base_ppl, _ = mean_std(results['A_baseline']['ppl'])
    res_ppl, _ = mean_std(results['B_reservoir']['ppl'])
    ppl_improvement = (base_ppl - res_ppl) / base_ppl * 100
    c2_valid = res_ppl < base_ppl
    claims.append({
        'claim': 'Reservoir improves prediction',
        'validated': bool(c2_valid),
        'evidence': f'PPL: {res_ppl:.3f} vs {base_ppl:.3f} ({ppl_improvement:+.1f}%)'
    })
    print(f"2. Reservoir helps: {'✓' if c2_valid else '✗'} ({ppl_improvement:+.1f}%)")

    # Claim 3: Responds to workload
    c3_valid = intervention['idle_load_diff'] > 0.1
    claims.append({
        'claim': 'Reservoir responds to workload',
        'validated': bool(c3_valid),
        'evidence': f'idle_load_diff = {intervention["idle_load_diff"]:.3f}'
    })
    print(f"3. Workload response: {'✓' if c3_valid else '✗'} (diff={intervention['idle_load_diff']:.3f})")

    # Claim 4: Live beats frozen (temporal info matters)
    frozen_ppl, _ = mean_std(results['C_frozen_reservoir']['ppl'])
    c4_valid = res_ppl < frozen_ppl
    claims.append({
        'claim': 'Live reservoir beats frozen (temporal info matters)',
        'validated': bool(c4_valid),
        'evidence': f'Live {res_ppl:.3f} vs Frozen {frozen_ppl:.3f}'
    })
    print(f"4. Live > Frozen: {'✓' if c4_valid else '✗'} (live={res_ppl:.3f}, frozen={frozen_ppl:.3f})")

    n_valid = sum(1 for c in claims if c['validated'])
    print(f"\n{n_valid}/{len(claims)} claims validated")

    # Save
    output = {
        'experiment': 'z1102_gpu_reservoir',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'config': asdict(cfg),
        'reservoir_analysis': dynamics,
        'intervention_test': intervention,
        'results': results,
        'claims': claims,
        'n_validated': n_valid
    }

    out_path = Path(__file__).parent.parent / "results" / "z1102_gpu_reservoir.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, default=str))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
