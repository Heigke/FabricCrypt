#!/usr/bin/env python3
"""
Z907: Kill Shot - Definitive Causal Embodiment Proof
=====================================================

Hypothesis: The best embodied model (from z905/z906) exhibits TRUE causal
embodiment that is provably different from any non-embodied baseline.

Protocol (5-way ablation matrix):
  LIVE     - Embodied model + Real telemetry        -> expect HIGH Q
  FROZEN   - Embodied model + Constant telemetry     -> expect LOW Q
  SHUFFLED - Embodied model + Time-shuffled telemetry -> expect LOW Q
  FIXED    - Standard model (no gates, all layers)   -> expect ZERO Q
  RANDOM   - Embodied model + Random uniform [0,1]   -> expect LOW Q

Q metric candidates (all computed):
  1. Mutual information: I(layer_firing_pattern ; gpu_temperature)
  2. Granger causality: does telemetry predict future firing patterns?
  3. Intervention test: artificial temp extremes -> measure firing response
  4. Energy efficiency gap: J/token across conditions

Success criterion: Q(LIVE) significantly exceeds all controls
  (p < 0.01 Bonferroni-corrected across 4 comparisons)

Author: FEEL Research Team
Date: 2026-01-28
"""

import os
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '11.0.0'

import sys
import json
import time
import math
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.telemetry.sysfs_hwmon import SysfsHwmonTelemetry, EnergyMeter

# Try scipy; fall back to bootstrap if unavailable
try:
    from scipy import stats as sp_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("[z907] scipy not available, using bootstrap statistical tests")

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SENSOR_DIM = 32
NUM_GATED_LAYERS = 7
SEQ_LEN = 128
VOCAB_SIZE = 50257  # GPT-2


# ===========================================================================
# Lightweight Gated Transformer (self-contained, no HuggingFace dependency)
# ===========================================================================

class SensorGateMini(nn.Module):
    """Lightweight sensor-driven gate: hidden + sensor -> probability."""

    def __init__(self, hidden_dim: int, sensor_dim: int = SENSOR_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim + sensor_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        nn.init.constant_(self.net[-1].bias, 0.7)
        self.last_gate_prob = 0.5

    def forward(self, hidden: torch.Tensor, sensors: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden: [B, D] pooled hidden state
            sensors: [B, SENSOR_DIM]
        Returns:
            gate: [B, 1] in (0, 1)
        """
        fused = torch.cat([hidden, sensors], dim=-1)
        logit = self.net(fused)
        prob = torch.sigmoid(logit)
        self.last_gate_prob = prob.mean().item()
        return prob


class GatedTransformerBlock(nn.Module):
    """Single transformer block with sensor-driven gating."""

    def __init__(self, hidden_dim: int, n_heads: int, sensor_dim: int = SENSOR_DIM):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.gate = SensorGateMini(hidden_dim, sensor_dim)
        self.has_gate = True

    def forward(self, x: torch.Tensor, sensors: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Returns:
            (output, gate_prob_float)
        """
        B = x.size(0)

        # Pool for gate decision
        pooled = x.mean(dim=1)  # [B, D]
        gate_prob = self.gate(pooled, sensors)  # [B, 1]
        gate_val = gate_prob.mean().item()

        # Attention + MLP (always compute in training for gradients)
        normed = self.ln1(x)
        attn_out, _ = self.attn(normed, normed, normed, need_weights=False)
        x_attn = x + attn_out
        x_out = x_attn + self.mlp(self.ln2(x_attn))

        # Soft gating during training, hard gating during eval
        if self.training:
            g = gate_prob.unsqueeze(-1)  # [B, 1, 1]
            output = g * x_out + (1 - g) * x
        else:
            if gate_val > 0.5:
                output = x_out
            else:
                output = x  # Hard skip

        return output, gate_val


class FixedTransformerBlock(nn.Module):
    """Standard transformer block without gating (baseline)."""

    def __init__(self, hidden_dim: int, n_heads: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.has_gate = False

    def forward(self, x: torch.Tensor, sensors: torch.Tensor) -> Tuple[torch.Tensor, float]:
        normed = self.ln1(x)
        attn_out, _ = self.attn(normed, normed, normed, need_weights=False)
        x_attn = x + attn_out
        x_out = x_attn + self.mlp(self.ln2(x_attn))
        return x_out, 1.0  # Always fires


class GatedTransformerLM(nn.Module):
    """
    Full gated transformer LM for embodiment experiments.
    Lightweight (fits on any GPU) but has real gating logic.
    """

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        hidden_dim: int = 256,
        n_layers: int = 8,
        n_heads: int = 4,
        sensor_dim: int = SENSOR_DIM,
        use_gates: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.sensor_dim = sensor_dim
        self.use_gates = use_gates

        self.token_emb = nn.Embedding(vocab_size, hidden_dim)
        self.pos_emb = nn.Embedding(1024, hidden_dim)
        self.drop = nn.Dropout(0.1)

        if use_gates:
            self.blocks = nn.ModuleList([
                GatedTransformerBlock(hidden_dim, n_heads, sensor_dim)
                for _ in range(n_layers)
            ])
        else:
            self.blocks = nn.ModuleList([
                FixedTransformerBlock(hidden_dim, n_heads)
                for _ in range(n_layers)
            ])

        self.ln_f = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        sensors: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[float]]:
        """
        Args:
            input_ids: [B, T]
            sensors: [B, SENSOR_DIM]
        Returns:
            (logits [B, T, V], gate_probs list)
        """
        B, T = input_ids.shape
        device = input_ids.device

        tok_emb = self.token_emb(input_ids)
        pos_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        pos_emb = self.pos_emb(pos_ids)
        x = self.drop(tok_emb + pos_emb)

        gate_probs = []
        for block in self.blocks:
            x, gp = block(x, sensors)
            gate_probs.append(gp)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits, gate_probs


# ===========================================================================
# Data Loading (TinyShakespeare)
# ===========================================================================

def download_tiny_shakespeare(data_dir: Path) -> str:
    """Download TinyShakespeare if not present, return text content."""
    data_dir.mkdir(parents=True, exist_ok=True)
    filepath = data_dir / "tiny_shakespeare.txt"

    if filepath.exists():
        return filepath.read_text()

    print("[z907] Downloading TinyShakespeare...")
    try:
        import urllib.request
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        urllib.request.urlretrieve(url, str(filepath))
        return filepath.read_text()
    except Exception as e:
        print(f"[z907] Download failed ({e}), generating synthetic data")
        # Fallback: generate synthetic character-level data
        import string
        rng = np.random.RandomState(42)
        chars = list(string.ascii_lowercase + " \n.,;:!?'-")
        text = "".join(rng.choice(chars, size=100000))
        filepath.write_text(text)
        return text


def encode_text(text: str, vocab_size: int = VOCAB_SIZE) -> torch.Tensor:
    """Encode text as token IDs (character-level mod vocab_size)."""
    ids = [ord(c) % vocab_size for c in text]
    return torch.tensor(ids, dtype=torch.long)


def make_batches(
    token_ids: torch.Tensor,
    batch_size: int,
    seq_len: int,
    n_batches: int,
    device: torch.device,
) -> List[torch.Tensor]:
    """Create list of input_ids batches from linear token array."""
    total = token_ids.numel()
    batches = []
    for i in range(n_batches):
        start = (i * batch_size * seq_len) % max(total - seq_len, 1)
        batch_ids = []
        for b in range(batch_size):
            offset = (start + b * seq_len) % max(total - seq_len, 1)
            batch_ids.append(token_ids[offset : offset + seq_len])
        batches.append(torch.stack(batch_ids).to(device))
    return batches


# ===========================================================================
# Model Loading / Training
# ===========================================================================

def load_or_train_embodied_model(
    checkpoint_dir: Path,
    device: torch.device,
    token_ids: torch.Tensor,
) -> GatedTransformerLM:
    """Load z906 checkpoint or train a fresh embodied model."""
    ckpt_path = checkpoint_dir / "z906_full.pt"

    model = GatedTransformerLM(use_gates=True).to(device)

    if ckpt_path.exists():
        print(f"[z907] Loading embodied checkpoint from {ckpt_path}")
        try:
            state = torch.load(ckpt_path, map_location=device, weights_only=False)
            if isinstance(state, dict) and "model_state_dict" in state:
                model.load_state_dict(state["model_state_dict"], strict=False)
            elif isinstance(state, dict) and "embodied" in state:
                model.load_state_dict(state["embodied"], strict=False)
            else:
                model.load_state_dict(state, strict=False)
            print("[z907] Checkpoint loaded successfully")
            return model
        except Exception as e:
            print(f"[z907] Checkpoint load failed ({e}), training fresh")

    # Train a fresh embodied model (5 epochs, simplified)
    print("[z907] Training fresh embodied model (5 epochs)...")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

    batches = make_batches(token_ids, batch_size=8, seq_len=SEQ_LEN, n_batches=200, device=device)

    # Create a simulated sensor stream for training
    n_epochs = 5
    step = 0
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        n_steps = 0
        for batch in batches:
            # Simulate varying stress during training
            stress = 0.3 + 0.4 * math.sin(step * 0.05)
            sensor_vec = _make_synthetic_sensors(stress, device)
            sensor_batch = sensor_vec.unsqueeze(0).expand(batch.size(0), -1)

            logits, gate_probs = model(batch[:, :-1], sensor_batch)
            targets = batch[:, 1:]

            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )

            # Metabolic penalty: high stress * high gate usage
            if gate_probs:
                avg_gate = sum(gate_probs) / len(gate_probs)
                metabolic = stress * avg_gate * 0.3
                loss = loss + metabolic

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_steps += 1
            step += 1

        avg = epoch_loss / max(n_steps, 1)
        print(f"  Epoch {epoch+1}/{n_epochs}  loss={avg:.4f}")

    # Save checkpoint
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_path = checkpoint_dir / "z907_embodied.pt"
    torch.save({"model_state_dict": model.state_dict()}, save_path)
    print(f"[z907] Saved embodied model to {save_path}")

    return model


def create_fixed_baseline(device: torch.device) -> GatedTransformerLM:
    """Create a standard (non-gated) transformer for FIXED condition."""
    model = GatedTransformerLM(use_gates=False).to(device)
    return model


def _make_synthetic_sensors(stress: float, device: torch.device) -> torch.Tensor:
    """Create a synthetic sensor vector for a given stress level."""
    s = np.zeros(SENSOR_DIM, dtype=np.float32)
    s[0:4] = 0.3 + stress * 0.5   # Thermal
    s[4:9] = 0.2 + stress * 0.6   # Power
    s[9:14] = 0.5 + stress * 0.3  # Clocks
    s[14:19] = 0.3 + stress * 0.6 # Utilization
    s[19:23] = 1.0 if stress > 0.8 else 0.0  # Throttle
    s[23:28] = 0.3 + stress * 0.4 # System
    s[28] = stress                 # temp_delta
    s[29] = 1.0 - stress           # power_efficiency
    s[30] = 1.0 - stress           # thermal_headroom
    s[31] = stress                 # stress_composite
    return torch.from_numpy(np.clip(s, 0, 1)).float().to(device)


# ===========================================================================
# Telemetry Helpers
# ===========================================================================

def get_real_sensor_vector(telem: SysfsHwmonTelemetry, device: torch.device) -> torch.Tensor:
    """Read real GPU telemetry and convert to SENSOR_DIM vector."""
    sample = telem.read_sample()
    s = np.zeros(SENSOR_DIM, dtype=np.float32)

    # Normalize to 0-1 using standard ranges
    s[0] = np.clip(sample.temp_edge_c / 100.0, 0, 1)
    s[1] = np.clip(sample.temp_junction_c / 110.0, 0, 1)
    s[2] = np.clip(sample.temp_mem_c / 95.0, 0, 1)
    s[3] = s[1]  # hotspot approx junction
    s[4] = np.clip(sample.power_w / 300.0, 0, 1)
    s[5] = s[4] * 0.85
    s[6] = s[4] * 0.15
    cap = 300.0
    s[7] = np.clip(sample.power_w / cap, 0, 1)
    s[8] = np.clip(1.0 - s[7], 0, 1)
    s[9] = np.clip(sample.freq_sclk_mhz / 3000.0, 0, 1)
    s[10] = np.clip(sample.freq_mclk_mhz / 2500.0, 0, 1)
    s[11] = s[9]
    s[12] = s[10]
    s[13] = 0.5
    s[14] = np.clip(sample.gpu_busy_pct / 100.0, 0, 1)
    s[15] = s[14]
    s[16] = 0.5
    s[17] = np.clip(sample.vram_used_gb / 16.0, 0, 1)
    s[18] = 0.5
    # Throttle / system / derived filled with reasonable defaults
    s[19:23] = 0.0
    s[23:28] = 0.3
    baseline_temp = 40.0
    s[28] = np.clip((sample.temp_edge_c - baseline_temp) / 60.0, 0, 1)
    busy = sample.gpu_busy_pct
    s[29] = np.clip(busy / (sample.power_w + 1.0), 0, 1) if busy > 10 else 0.5
    s[30] = np.clip(1.0 - sample.temp_edge_c / 90.0, 0, 1)
    s[31] = np.clip(
        (1 - s[30]) * 0.3 + s[7] * 0.3 + s[14] * 0.2,
        0, 1,
    )

    return torch.from_numpy(s).float().to(device)


# ===========================================================================
# Q-Metric Computations
# ===========================================================================

def compute_mutual_information(
    firing_patterns: np.ndarray,
    temperatures: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Compute mutual information I(firing_pattern ; temperature) in bits.

    firing_patterns: [N, num_layers] floats (gate probabilities)
    temperatures: [N] float
    """
    N = firing_patterns.shape[0]
    if N < 10:
        return 0.0

    # Discretize temperature
    t_min, t_max = temperatures.min(), temperatures.max()
    if t_max - t_min < 1e-6:
        return 0.0
    temp_bins = np.digitize(
        temperatures,
        np.linspace(t_min, t_max, n_bins + 1)[1:-1],
    )

    # Discretize firing patterns: threshold at 0.5, pack to integer
    fire_binary = (firing_patterns > 0.5).astype(np.uint8)
    # Hash each row into an integer (for up to 64 layers)
    n_layers = fire_binary.shape[1]
    fire_ints = np.zeros(N, dtype=np.int64)
    for col in range(min(n_layers, 63)):
        fire_ints += fire_binary[:, col].astype(np.int64) << col

    # Joint distribution
    joint = {}
    margin_t = {}
    margin_f = {}
    for i in range(N):
        t = int(temp_bins[i])
        f = int(fire_ints[i])
        joint[(t, f)] = joint.get((t, f), 0) + 1
        margin_t[t] = margin_t.get(t, 0) + 1
        margin_f[f] = margin_f.get(f, 0) + 1

    mi = 0.0
    for (t, f), count in joint.items():
        p_joint = count / N
        p_t = margin_t[t] / N
        p_f = margin_f[f] / N
        if p_joint > 0 and p_t > 0 and p_f > 0:
            mi += p_joint * math.log2(p_joint / (p_t * p_f))

    return max(mi, 0.0)


def granger_causality_test(
    telemetry_history: np.ndarray,
    firing_history: np.ndarray,
    max_lag: int = 5,
) -> float:
    """
    Simplified Granger causality test.
    Tests if telemetry history helps predict firing patterns.

    telemetry_history: [N] float (e.g. temperature)
    firing_history: [N] float (e.g. avg gate prob)

    Returns: p-value (low = telemetry Granger-causes firing)
    """
    N = len(telemetry_history)
    if N < max_lag + 10:
        return 1.0  # Not enough data

    # Construct design matrices
    Y = firing_history[max_lag:]  # target
    n = len(Y)

    # Restricted model: Y[t] ~ Y[t-1:t-lag]
    X_restricted = np.column_stack([
        firing_history[max_lag - k - 1 : N - k - 1]
        for k in range(max_lag)
    ])

    # Unrestricted model: Y[t] ~ Y[t-1:t-lag] + telemetry[t-1:t-lag]
    X_unrestricted = np.column_stack([
        X_restricted,
        *[
            telemetry_history[max_lag - k - 1 : N - k - 1].reshape(-1, 1)
            for k in range(max_lag)
        ],
    ])

    # Add intercept
    X_r = np.column_stack([np.ones(n), X_restricted])
    X_u = np.column_stack([np.ones(n), X_unrestricted])

    # OLS fits
    try:
        beta_r = np.linalg.lstsq(X_r, Y, rcond=None)[0]
        resid_r = Y - X_r @ beta_r
        rss_r = np.sum(resid_r ** 2)

        beta_u = np.linalg.lstsq(X_u, Y, rcond=None)[0]
        resid_u = Y - X_u @ beta_u
        rss_u = np.sum(resid_u ** 2)

        # F-test
        p_r = X_r.shape[1]
        p_u = X_u.shape[1]
        df1 = p_u - p_r
        df2 = n - p_u

        if rss_u <= 0 or df1 <= 0 or df2 <= 0:
            return 1.0

        f_stat = ((rss_r - rss_u) / df1) / (rss_u / df2)

        if HAS_SCIPY:
            p_value = 1.0 - sp_stats.f.cdf(f_stat, df1, df2)
        else:
            # Approximate p-value using bootstrap
            p_value = _bootstrap_f_pvalue(f_stat, df1, df2)

        return float(np.clip(p_value, 0, 1))

    except Exception:
        return 1.0


def _bootstrap_f_pvalue(f_stat: float, df1: int, df2: int, n_samples: int = 10000) -> float:
    """Approximate F-distribution p-value via simulation."""
    rng = np.random.RandomState(42)
    samples = rng.f(df1, df2, size=n_samples)
    return float(np.mean(samples >= f_stat))


def intervention_test(
    model: GatedTransformerLM,
    input_ids: torch.Tensor,
    device: torch.device,
) -> Dict[str, float]:
    """
    Test model response to artificial telemetry extremes.
    Measures firing rate under cold/mid/hot conditions.
    """
    model.eval()
    results = {}
    B = input_ids.size(0)

    for label, stress in [("cold", 0.1), ("mid", 0.5), ("hot", 0.9)]:
        sensors = _make_synthetic_sensors(stress, device)
        sensor_batch = sensors.unsqueeze(0).expand(B, -1)

        with torch.no_grad():
            _, gate_probs = model(input_ids, sensor_batch)

        avg_fire = sum(gate_probs) / max(len(gate_probs), 1)
        results[label] = avg_fire

    # Intervention response = |cold - hot|
    results["response_magnitude"] = abs(results.get("cold", 0.5) - results.get("hot", 0.5))
    return results


# ===========================================================================
# Condition Runners
# ===========================================================================

@dataclass
class ConditionResult:
    """Results from running one experimental condition."""
    name: str
    mutual_information: float = 0.0
    granger_pvalue: float = 1.0
    intervention_response: float = 0.0
    intervention_detail: Dict[str, float] = field(default_factory=dict)
    energy_j_per_token: float = 0.0
    perplexity: float = 0.0
    avg_gate_prob: float = 0.0
    avg_temperature: float = 0.0
    firing_patterns: Optional[np.ndarray] = None
    temperatures: Optional[np.ndarray] = None
    n_steps: int = 0


def run_condition(
    name: str,
    model: GatedTransformerLM,
    batches: List[torch.Tensor],
    device: torch.device,
    n_steps: int,
    telem: Optional[SysfsHwmonTelemetry],
    sensor_mode: str = "live",
    frozen_sensor_cache: Optional[np.ndarray] = None,
    shuffled_temps: Optional[np.ndarray] = None,
) -> ConditionResult:
    """
    Run a single experimental condition.

    sensor_mode:
      "live"     - real GPU telemetry
      "frozen"   - constant (mean of pre-recorded values)
      "shuffled" - real values in random temporal order
      "fixed"    - doesn't matter (no gates)
      "random"   - uniform random [0, 1]
    """
    model.eval()
    result = ConditionResult(name=name)

    all_gate_probs = []    # [N, n_layers]
    all_temperatures = []  # [N]
    all_losses = []
    total_tokens = 0
    total_energy_j = 0.0

    # Precompute shuffled sensor cache if provided
    shuffled_idx = 0

    print(f"\n{'='*60}")
    print(f"  Condition: {name}  ({sensor_mode})  [n_steps={n_steps}]")
    print(f"{'='*60}")

    for step_i in range(n_steps):
        batch_idx = step_i % len(batches)
        input_ids = batches[batch_idx]
        B, T = input_ids.shape
        tokens_in_batch = B * (T - 1)

        # Determine sensor vector
        if sensor_mode == "live" and telem is not None:
            sensor_vec = get_real_sensor_vector(telem, device)
            sample = telem.read_sample()
            temp_c = sample.temp_edge_c
        elif sensor_mode == "frozen":
            if frozen_sensor_cache is not None:
                sensor_vec = torch.from_numpy(frozen_sensor_cache).float().to(device)
            else:
                sensor_vec = _make_synthetic_sensors(0.5, device)
            temp_c = frozen_sensor_cache[0] * 100.0 if frozen_sensor_cache is not None else 50.0
        elif sensor_mode == "shuffled":
            if shuffled_temps is not None and shuffled_idx < len(shuffled_temps):
                stress = shuffled_temps[shuffled_idx]
                shuffled_idx += 1
            else:
                stress = np.random.uniform(0.2, 0.8)
            sensor_vec = _make_synthetic_sensors(stress, device)
            temp_c = stress * 100.0
        elif sensor_mode == "random":
            rand_vals = np.random.uniform(0, 1, SENSOR_DIM).astype(np.float32)
            sensor_vec = torch.from_numpy(rand_vals).float().to(device)
            temp_c = rand_vals[0] * 100.0
        else:  # "fixed" or fallback
            sensor_vec = _make_synthetic_sensors(0.5, device)
            temp_c = 50.0

        sensor_batch = sensor_vec.unsqueeze(0).expand(B, -1)

        # Measure energy
        start_ns = time.time_ns()
        with torch.no_grad():
            logits, gate_probs = model(input_ids[:, :-1], sensor_batch)
            targets = input_ids[:, 1:]
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )
        end_ns = time.time_ns()

        # Energy estimate: read power * duration
        if telem is not None:
            try:
                power_sample = telem.read_sample()
                duration_s = (end_ns - start_ns) / 1e9
                step_energy = power_sample.power_w * duration_s
            except Exception:
                step_energy = 0.0
        else:
            step_energy = 0.0

        total_energy_j += step_energy
        total_tokens += tokens_in_batch
        all_losses.append(loss.item())
        all_gate_probs.append(gate_probs)
        all_temperatures.append(temp_c)

        if (step_i + 1) % 100 == 0:
            avg_loss = sum(all_losses[-100:]) / 100
            print(f"    step {step_i+1}/{n_steps}  loss={avg_loss:.4f}  temp={temp_c:.1f}C")

    # Aggregate results
    n_layers = len(all_gate_probs[0]) if all_gate_probs else 1
    firing_mat = np.array([[gp for gp in gps] for gps in all_gate_probs])  # [N, n_layers]
    temp_arr = np.array(all_temperatures)  # [N]

    result.firing_patterns = firing_mat
    result.temperatures = temp_arr
    result.n_steps = n_steps

    # Perplexity
    avg_loss = sum(all_losses) / max(len(all_losses), 1)
    result.perplexity = math.exp(min(avg_loss, 20.0))

    # Average gate probability
    result.avg_gate_prob = float(firing_mat.mean())
    result.avg_temperature = float(temp_arr.mean())

    # Energy per token
    if total_tokens > 0:
        result.energy_j_per_token = total_energy_j / total_tokens
    else:
        result.energy_j_per_token = 0.0

    # === Q-Metric 1: Mutual Information ===
    result.mutual_information = compute_mutual_information(firing_mat, temp_arr)

    # === Q-Metric 2: Granger Causality ===
    avg_firing = firing_mat.mean(axis=1)  # [N]
    result.granger_pvalue = granger_causality_test(temp_arr, avg_firing)

    # === Q-Metric 3: Intervention Test ===
    test_batch = batches[0][:4]  # Use first 4 samples
    intv = intervention_test(model, test_batch[:, :-1], device)
    result.intervention_response = intv.get("response_magnitude", 0.0)
    result.intervention_detail = intv

    print(f"  Results for {name}:")
    print(f"    MI      = {result.mutual_information:.4f} bits")
    print(f"    Granger = {result.granger_pvalue:.6f} (p-value)")
    print(f"    Interv  = {result.intervention_response:.4f}")
    print(f"    Energy  = {result.energy_j_per_token:.6f} J/tok")
    print(f"    PPL     = {result.perplexity:.2f}")
    print(f"    AvgGate = {result.avg_gate_prob:.4f}")

    return result


# ===========================================================================
# Statistical Testing
# ===========================================================================

def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Cohen's d effect size."""
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return 0.0
    mx, my = np.mean(x), np.mean(y)
    sx, sy = np.std(x, ddof=1), np.std(y, ddof=1)
    pooled_std = math.sqrt(((nx - 1) * sx**2 + (ny - 1) * sy**2) / (nx + ny - 2))
    if pooled_std < 1e-12:
        return 0.0
    return (mx - my) / pooled_std


def paired_comparison(
    live_values: np.ndarray,
    control_values: np.ndarray,
    label: str,
) -> Dict[str, float]:
    """
    Paired t-test (or bootstrap) between LIVE and a control condition.
    Returns t-stat, raw p-value, and Cohen's d.
    """
    n = min(len(live_values), len(control_values))
    if n < 3:
        return {"t_stat": 0.0, "p_value": 1.0, "cohens_d": 0.0, "label": label}

    live = live_values[:n]
    ctrl = control_values[:n]

    if HAS_SCIPY:
        t_stat, p_value = sp_stats.ttest_rel(live, ctrl)
    else:
        # Bootstrap paired difference test
        diffs = live - ctrl
        mean_diff = np.mean(diffs)
        if np.std(diffs) < 1e-12:
            t_stat = 0.0
            p_value = 1.0
        else:
            t_stat = mean_diff / (np.std(diffs, ddof=1) / math.sqrt(n))
            # Two-sided p approximation
            rng = np.random.RandomState(42)
            boot_means = []
            for _ in range(5000):
                idx = rng.choice(n, size=n, replace=True)
                boot_means.append(np.mean(diffs[idx]))
            boot_means = np.array(boot_means)
            p_value = float(np.mean(np.abs(boot_means - np.mean(boot_means)) >= abs(mean_diff))) * 2
            p_value = min(p_value, 1.0)

    d = cohens_d(live, ctrl)

    return {
        "t_stat": float(t_stat) if not np.isnan(t_stat) else 0.0,
        "p_value": float(p_value) if not np.isnan(p_value) else 1.0,
        "cohens_d": float(d),
        "label": label,
    }


# ===========================================================================
# Composite Q-Score
# ===========================================================================

def compute_q_score(result: ConditionResult) -> float:
    """
    Composite embodiment quality score.
    Combines MI, Granger significance, and intervention response.
    Higher = more embodied.
    """
    # MI contribution (0 to ~2 bits typically)
    mi_score = min(result.mutual_information / 0.5, 1.0)  # Normalize

    # Granger contribution (low p-value = high score)
    granger_score = max(1.0 - result.granger_pvalue, 0.0)

    # Intervention contribution
    intv_score = min(result.intervention_response / 0.3, 1.0)

    # Weighted combination
    q = 0.4 * mi_score + 0.3 * granger_score + 0.3 * intv_score
    return float(q)


# ===========================================================================
# Main Experiment
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="Z907: Kill Shot - Causal Embodiment Proof")
    parser.add_argument("--n-eval-steps", type=int, default=500,
                        help="Number of forward passes per condition")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: auto, cuda, cpu")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/",
                        help="Directory for model checkpoints")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size for evaluation")
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN,
                        help="Sequence length")
    args = parser.parse_args()

    print("=" * 70)
    print("  Z907: KILL SHOT - Definitive Causal Embodiment Proof")
    print("=" * 70)
    print(f"  n_eval_steps = {args.n_eval_steps}")
    print(f"  device       = {args.device}")
    print(f"  checkpoint   = {args.checkpoint_dir}")

    # --- Device ---
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"\n  Using device: {device}")

    project_root = Path(__file__).parent.parent

    # --- Telemetry ---
    telem = None
    try:
        telem = SysfsHwmonTelemetry(sample_rate_hz=50)
        telem.start_continuous_sampling()
        time.sleep(0.5)
        sample = telem.read_sample()
        print(f"\n  GPU Telemetry active: {sample.temp_edge_c:.1f}C, {sample.power_w:.1f}W")
    except Exception as e:
        print(f"\n  [WARNING] GPU telemetry unavailable ({e}), using synthetic sensors")
        telem = None

    # --- Data ---
    print("\n  Loading TinyShakespeare...")
    data_dir = project_root / "data" / "tinyshakespeare"
    text = download_tiny_shakespeare(data_dir)
    token_ids = encode_text(text)
    print(f"  Tokens: {token_ids.numel():,}")

    batches = make_batches(
        token_ids,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        n_batches=max(args.n_eval_steps, 200),
        device=device,
    )

    # --- Models ---
    checkpoint_dir = project_root / args.checkpoint_dir
    print("\n  Loading/training embodied model...")
    embodied_model = load_or_train_embodied_model(checkpoint_dir, device, token_ids)
    embodied_model.eval()

    print("\n  Creating fixed baseline model...")
    fixed_model = create_fixed_baseline(device)
    fixed_model.eval()

    # --- Pre-record live sensor data for FROZEN and SHUFFLED conditions ---
    print("\n  Pre-recording sensor data for FROZEN/SHUFFLED conditions...")
    prerecord_sensors = []
    prerecord_temps = []
    if telem is not None:
        for _ in range(min(args.n_eval_steps, 200)):
            svec = get_real_sensor_vector(telem, device)
            prerecord_sensors.append(svec.cpu().numpy())
            sample = telem.read_sample()
            prerecord_temps.append(sample.temp_edge_c)
            time.sleep(0.01)
    else:
        # Synthetic pre-recording with varying stress
        for i in range(min(args.n_eval_steps, 200)):
            stress = 0.3 + 0.4 * math.sin(i * 0.03)
            svec = _make_synthetic_sensors(stress, device)
            prerecord_sensors.append(svec.cpu().numpy())
            prerecord_temps.append(stress * 100.0)

    prerecord_arr = np.array(prerecord_sensors)
    frozen_mean = prerecord_arr.mean(axis=0)  # Mean sensor vector

    # Time-shuffled: random permutation of recorded stress values
    shuffled_stresses = np.array(prerecord_temps) / 100.0
    rng = np.random.RandomState(42)
    rng.shuffle(shuffled_stresses)

    print(f"  Pre-recorded {len(prerecord_sensors)} sensor samples")
    print(f"  Frozen mean temp sensor: {frozen_mean[0]:.4f}")

    # ==================================================================
    # Run 5-way ablation matrix
    # ==================================================================

    print("\n" + "=" * 70)
    print("  RUNNING 5-WAY ABLATION MATRIX")
    print("=" * 70)

    results: Dict[str, ConditionResult] = {}

    # 1. LIVE: Embodied model + Real telemetry
    results["LIVE"] = run_condition(
        name="LIVE",
        model=embodied_model,
        batches=batches,
        device=device,
        n_steps=args.n_eval_steps,
        telem=telem,
        sensor_mode="live",
    )

    # 2. FROZEN: Embodied model + Constant (mean) telemetry
    results["FROZEN"] = run_condition(
        name="FROZEN",
        model=embodied_model,
        batches=batches,
        device=device,
        n_steps=args.n_eval_steps,
        telem=telem,
        sensor_mode="frozen",
        frozen_sensor_cache=frozen_mean,
    )

    # 3. SHUFFLED: Embodied model + Time-shuffled telemetry
    results["SHUFFLED"] = run_condition(
        name="SHUFFLED",
        model=embodied_model,
        batches=batches,
        device=device,
        n_steps=args.n_eval_steps,
        telem=telem,
        sensor_mode="shuffled",
        shuffled_temps=shuffled_stresses,
    )

    # 4. FIXED: Standard model (no gates) + Real telemetry
    results["FIXED"] = run_condition(
        name="FIXED",
        model=fixed_model,
        batches=batches,
        device=device,
        n_steps=args.n_eval_steps,
        telem=telem,
        sensor_mode="fixed",
    )

    # 5. RANDOM: Embodied model + Random uniform sensors
    results["RANDOM"] = run_condition(
        name="RANDOM",
        model=embodied_model,
        batches=batches,
        device=device,
        n_steps=args.n_eval_steps,
        telem=telem,
        sensor_mode="random",
    )

    # ==================================================================
    # Compute Q-scores
    # ==================================================================

    print("\n" + "=" * 70)
    print("  Q-SCORE SUMMARY")
    print("=" * 70)

    q_scores = {}
    for name, res in results.items():
        q = compute_q_score(res)
        q_scores[name] = q
        print(f"  {name:10s}  Q={q:.4f}  (MI={res.mutual_information:.4f}, "
              f"Granger_p={res.granger_pvalue:.4e}, Interv={res.intervention_response:.4f})")

    # ==================================================================
    # Statistical Tests (LIVE vs each control)
    # ==================================================================

    print("\n" + "=" * 70)
    print("  STATISTICAL TESTS (LIVE vs Controls)")
    print("=" * 70)

    live_result = results["LIVE"]
    controls = ["FROZEN", "SHUFFLED", "FIXED", "RANDOM"]
    n_comparisons = len(controls)
    stat_tests = {}

    for ctrl_name in controls:
        ctrl_result = results[ctrl_name]

        # Compare MI distributions using per-step sliding window
        n = min(live_result.n_steps, ctrl_result.n_steps)
        window = 50
        live_mi_windows = []
        ctrl_mi_windows = []

        for start in range(0, n - window, window // 2):
            end = start + window
            live_chunk = live_result.firing_patterns[start:end]
            live_temp_chunk = live_result.temperatures[start:end]
            ctrl_chunk = ctrl_result.firing_patterns[start:end]
            ctrl_temp_chunk = ctrl_result.temperatures[start:end]

            live_mi_windows.append(compute_mutual_information(live_chunk, live_temp_chunk))
            ctrl_mi_windows.append(compute_mutual_information(ctrl_chunk, ctrl_temp_chunk))

        live_mi_arr = np.array(live_mi_windows) if live_mi_windows else np.array([0.0])
        ctrl_mi_arr = np.array(ctrl_mi_windows) if ctrl_mi_windows else np.array([0.0])

        comparison = paired_comparison(live_mi_arr, ctrl_mi_arr, f"LIVE_vs_{ctrl_name}")

        # Bonferroni correction
        comparison["p_bonferroni"] = min(comparison["p_value"] * n_comparisons, 1.0)
        comparison["significant_001"] = comparison["p_bonferroni"] < 0.01

        stat_tests[ctrl_name] = comparison

        sig_marker = "***" if comparison["significant_001"] else "n.s."
        print(f"  LIVE vs {ctrl_name:10s}:  t={comparison['t_stat']:7.3f}  "
              f"p_raw={comparison['p_value']:.4e}  "
              f"p_bonf={comparison['p_bonferroni']:.4e}  "
              f"d={comparison['cohens_d']:.3f}  {sig_marker}")

    # ==================================================================
    # Verdict
    # ==================================================================

    print("\n" + "=" * 70)
    print("  VERDICT")
    print("=" * 70)

    live_q = q_scores["LIVE"]
    all_control_qs = [q_scores[c] for c in controls]
    max_control_q = max(all_control_qs) if all_control_qs else 0.0

    # Check: Q(LIVE) > 2x max control Q
    q_ratio = live_q / max_control_q if max_control_q > 1e-6 else float('inf')

    # Check: all statistical tests significant
    all_significant = all(
        stat_tests[c]["significant_001"] for c in controls
    )

    # Determine verdict
    tests_passed = []
    tests_failed = []

    # Test 1: Q ratio
    if q_ratio > 2.0:
        tests_passed.append(f"Q_ratio={q_ratio:.2f} > 2.0")
    else:
        tests_failed.append(f"Q_ratio={q_ratio:.2f} <= 2.0")

    # Test 2: Statistical significance
    for c in controls:
        if stat_tests[c]["significant_001"]:
            tests_passed.append(f"LIVE > {c} (p_bonf={stat_tests[c]['p_bonferroni']:.4e})")
        else:
            tests_failed.append(f"LIVE vs {c} not significant (p_bonf={stat_tests[c]['p_bonferroni']:.4e})")

    # Test 3: Intervention response
    if live_result.intervention_response > 0.05:
        tests_passed.append(f"Intervention_response={live_result.intervention_response:.4f} > 0.05")
    else:
        tests_failed.append(f"Intervention_response={live_result.intervention_response:.4f} <= 0.05")

    # Test 4: MI > 0
    if live_result.mutual_information > 0.01:
        tests_passed.append(f"MI={live_result.mutual_information:.4f} > 0.01")
    else:
        tests_failed.append(f"MI={live_result.mutual_information:.4f} <= 0.01")

    # Overall verdict
    if len(tests_failed) == 0 and q_ratio > 2.0 and all_significant:
        verdict = "EMBODIMENT CONFIRMED"
    else:
        verdict = "EMBODIMENT NOT PROVEN"

    print(f"\n  {'*' * 50}")
    print(f"  *  VERDICT: {verdict}")
    print(f"  {'*' * 50}")
    print(f"\n  Q(LIVE)      = {live_q:.4f}")
    print(f"  max(Q_ctrl)  = {max_control_q:.4f}")
    print(f"  Q ratio      = {q_ratio:.2f}")
    print(f"  All sig?     = {all_significant}")
    print(f"\n  Tests passed ({len(tests_passed)}):")
    for t in tests_passed:
        print(f"    [PASS] {t}")
    if tests_failed:
        print(f"\n  Tests failed ({len(tests_failed)}):")
        for t in tests_failed:
            print(f"    [FAIL] {t}")

    # ==================================================================
    # Save Results
    # ==================================================================

    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / "z907_kill_shot.json"

    output = {
        "experiment": "z907_kill_shot",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {
            "n_eval_steps": args.n_eval_steps,
            "device": str(device),
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "has_real_telemetry": telem is not None,
        },
        "conditions": {},
        "q_scores": q_scores,
        "statistical_tests": stat_tests,
        "verdict": verdict,
        "q_ratio": q_ratio,
        "all_significant": all_significant,
        "tests_passed": tests_passed,
        "tests_failed": tests_failed,
    }

    for name, res in results.items():
        output["conditions"][name] = {
            "mutual_information": res.mutual_information,
            "granger_pvalue": res.granger_pvalue,
            "intervention_response": res.intervention_response,
            "intervention_detail": res.intervention_detail,
            "energy_j_per_token": res.energy_j_per_token,
            "perplexity": res.perplexity,
            "avg_gate_prob": res.avg_gate_prob,
            "avg_temperature": res.avg_temperature,
            "n_steps": res.n_steps,
        }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n  Results saved to {output_path}")

    # Cleanup
    if telem is not None:
        telem.stop_continuous_sampling()

    print("\n" + "=" * 70)
    print("  Z907 COMPLETE")
    print("=" * 70)

    return output


if __name__ == "__main__":
    main()
