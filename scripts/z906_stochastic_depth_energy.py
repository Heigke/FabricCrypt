#!/usr/bin/env python3
"""
Z906: Stochastic Depth with Energy-Aware Bernoulli Gates

Hypothesis: Per-layer Bernoulli gates conditioned on BOTH hidden state AND
real-time GPU telemetry achieve better energy-quality Pareto curves than
fixed stochastic depth or either signal alone.

Architecture: 12-layer transformer (256 hidden, 4 heads, ff_dim=1024, ~1M params)
  - Per-layer gate: skip_prob_i = sigmoid(gate_MLP(concat(h_pooled, telemetry_2d)))
  - Each layer fires with probability (1 - skip_prob_i) via Straight-Through Estimator
  - gate_MLP: Linear(hidden_dim + 2, 32) -> ReLU -> Linear(32, 1)
  - telemetry_2d: [normalized_power, normalized_temp]

Key difference from early exit: Each layer independently fires or skips.
Layer 3 can fire while layer 5 skips. This is genuinely variable computation.

Training: REINFORCE with reward = -L_task - lambda * E_measured
  Gate parameters trained with policy gradient, transformer with standard backprop.

Controls:
  A: Fixed stochastic depth (p=0.1 per layer, no gate awareness)
  B: Gate depends only on hidden state (no telemetry)
  C: Gate depends only on telemetry (no semantic input)
  D: Full model -- gate depends on hidden state AND telemetry

Metrics: % layers fired vs GPU temp, perplexity vs compute, Pareto frontier,
         layer firing heatmaps.
"""

import os
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"

import sys
import time
import json
import math
import argparse
import urllib.request
from pathlib import Path
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass, field, asdict

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Telemetry imports with robust fallback
# ---------------------------------------------------------------------------
TELEMETRY_AVAILABLE = False
try:
    from src.telemetry.sysfs_hwmon import SysfsHwmonTelemetry, EnergyMeter
    TELEMETRY_AVAILABLE = True
except Exception as e:
    print(f"[WARN] sysfs_hwmon telemetry unavailable: {e}")
    print("[WARN] Falling back to synthetic telemetry")


class FallbackTelemetry:
    """Synthetic telemetry when real hardware is unavailable."""

    def __init__(self):
        self._power = 30.0
        self._temp = 55.0

    def read_sample(self):
        """Return a duck-typed sample object."""
        self._power = 30.0 + np.random.normal(0, 2)
        self._temp = 55.0 + np.random.normal(0, 3)
        return type("Sample", (), {
            "power_w": self._power,
            "temp_edge_c": self._temp,
        })()

    def start_continuous_sampling(self):
        pass

    def stop_continuous_sampling(self):
        pass

    def reset_accumulator(self):
        pass


class FallbackEnergyMeter:
    """Synthetic energy meter for non-GPU environments."""

    def __init__(self, telemetry, sync_cuda=True):
        self.telemetry = telemetry
        self.energy_j = 0.0
        self._start_time = 0.0

    def __enter__(self):
        self._start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        dt = time.perf_counter() - self._start_time
        sample = self.telemetry.read_sample()
        self.energy_j = sample.power_w * dt


# ---------------------------------------------------------------------------
# Data: TinyShakespeare (byte-level)
# ---------------------------------------------------------------------------
DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_DIR = Path(__file__).parent.parent / "data"


def get_shakespeare_data(seq_len: int = 256) -> Tuple[torch.Tensor, torch.Tensor]:
    """Download TinyShakespeare and return byte-level train/val tensors."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    data_path = DATA_DIR / "tinyshakespeare.txt"

    if not data_path.exists():
        print(f"[DATA] Downloading TinyShakespeare to {data_path} ...")
        try:
            urllib.request.urlretrieve(DATA_URL, str(data_path))
            print(f"[DATA] Downloaded {data_path.stat().st_size / 1024:.0f} KB")
        except Exception as e:
            print(f"[WARN] Download failed: {e}")
            print("[DATA] Generating synthetic corpus instead")
            text = ("To be or not to be that is the question " * 500 +
                    "Now is the winter of our discontent " * 500 +
                    "All that glitters is not gold " * 500 +
                    "The quick brown fox jumps over the lazy dog " * 500)
            data_path.write_text(text)

    raw = data_path.read_text(encoding="utf-8", errors="replace")
    data = torch.tensor(list(raw.encode("utf-8", errors="replace")), dtype=torch.long)

    # Truncate to multiple of seq_len
    n = (len(data) // seq_len) * seq_len
    data = data[:n]

    split = int(0.9 * n)
    train_data = data[:split]
    val_data = data[split:]

    print(f"[DATA] Total: {n} bytes, Train: {len(train_data)}, Val: {len(val_data)}")
    return train_data, val_data


def make_batches(data: torch.Tensor, batch_size: int, seq_len: int,
                 device: torch.device) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Create (input, target) batches from byte sequence."""
    n_seq = len(data) // (seq_len + 1)
    if n_seq < batch_size:
        batch_size = max(1, n_seq)

    n_batches = n_seq // batch_size
    batches = []
    for i in range(n_batches):
        batch_inputs = []
        batch_targets = []
        for j in range(batch_size):
            idx = (i * batch_size + j) * (seq_len + 1)
            chunk = data[idx: idx + seq_len + 1]
            batch_inputs.append(chunk[:seq_len])
            batch_targets.append(chunk[1: seq_len + 1])
        inp = torch.stack(batch_inputs).to(device)
        tgt = torch.stack(batch_targets).to(device)
        batches.append((inp, tgt))

    return batches


# ===========================================================================
# Model Architecture
# ===========================================================================

class LayerGate(nn.Module):
    """Per-layer firing gate conditioned on hidden state and/or telemetry."""

    def __init__(self, hidden_dim: int, use_hidden: bool = True,
                 use_telemetry: bool = True):
        super().__init__()
        input_dim = 0
        if use_hidden:
            input_dim += hidden_dim
        if use_telemetry:
            input_dim += 2  # [power_norm, temp_norm]
        if input_dim == 0:
            input_dim = 1  # dummy for fixed mode

        self.gate = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.use_hidden = use_hidden
        self.use_telemetry = use_telemetry

        # Initialize bias so layers fire most of the time (~88% fire rate)
        nn.init.constant_(self.gate[-1].bias, 2.0)

    def forward(self, h_pooled: torch.Tensor,
                telemetry_2d: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            h_pooled: [batch, hidden_dim] -- mean-pooled hidden state
            telemetry_2d: [2] or [batch, 2] -- [norm_power, norm_temp]
        Returns:
            skip_prob: [batch] -- probability of SKIPPING this layer
        """
        parts = []
        if self.use_hidden:
            parts.append(h_pooled)
        if self.use_telemetry and telemetry_2d is not None:
            if telemetry_2d.dim() == 1:
                telemetry_2d = telemetry_2d.unsqueeze(0)
            parts.append(telemetry_2d.expand(h_pooled.size(0), -1))
        if not parts:
            parts.append(torch.zeros(h_pooled.size(0), 1, device=h_pooled.device))

        x = torch.cat(parts, dim=-1)
        skip_logit = self.gate(x).squeeze(-1)  # [batch]
        skip_prob = torch.sigmoid(skip_logit)
        return skip_prob


class GatedTransformerBlock(nn.Module):
    """Pre-norm transformer block with multi-head attention + FFN."""

    def __init__(self, hidden_dim: int, num_heads: int, ff_dim: int,
                 dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-norm self-attention
        h = self.ln1(x)
        h, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + h
        # Pre-norm FFN
        h = self.ln2(x)
        h = self.ffn(h)
        x = x + h
        return x


@dataclass
class GateInfo:
    """Records gate decision for one layer in one forward pass."""
    layer_idx: int
    skip_prob: torch.Tensor  # [batch] -- differentiable
    fired: torch.Tensor      # [batch] -- bool, whether the layer executed


class StochasticDepthTransformer(nn.Module):
    """
    12-layer transformer with per-layer Bernoulli skip gates.

    Each layer independently fires or skips based on:
      - Hidden state (semantic signal)
      - GPU telemetry (energy signal)
      - Both (full model)
      - Neither (fixed stochastic depth baseline)
    """

    def __init__(self, vocab_size: int = 256, hidden_dim: int = 256,
                 num_layers: int = 12, num_heads: int = 4, ff_dim: int = 1024,
                 max_seq_len: int = 256, dropout: float = 0.1,
                 use_hidden_gate: bool = True, use_telemetry_gate: bool = True):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_hidden_gate = use_hidden_gate
        self.use_telemetry_gate = use_telemetry_gate

        # Embeddings
        self.token_emb = nn.Embedding(vocab_size, hidden_dim)
        self.pos_emb = nn.Embedding(max_seq_len, hidden_dim)
        self.emb_drop = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            GatedTransformerBlock(hidden_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

        # Per-layer gates
        self.gates = nn.ModuleList([
            LayerGate(hidden_dim, use_hidden=use_hidden_gate,
                      use_telemetry=use_telemetry_gate)
            for _ in range(num_layers)
        ])

        # Output
        self.ln_out = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size, bias=False)

        # Causal mask cache
        self._causal_mask = None
        self._causal_mask_size = 0

    def _get_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        if self._causal_mask is None or self._causal_mask_size < seq_len:
            mask = torch.triu(
                torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
                diagonal=1
            )
            self._causal_mask = mask  # True = masked out
            self._causal_mask_size = seq_len
        return self._causal_mask[:seq_len, :seq_len]

    def forward(self, input_ids: torch.Tensor,
                telemetry_2d: Optional[torch.Tensor] = None,
                return_gate_info: bool = True,
                fixed_skip_prob: Optional[float] = None):
        """
        Args:
            input_ids: [batch, seq_len] long tensor of byte values
            telemetry_2d: [2] tensor of [norm_power, norm_temp]
            return_gate_info: whether to return gate decisions
            fixed_skip_prob: if set, use fixed skip probability (condition A)
        Returns:
            logits: [batch, seq_len, vocab_size]
            gate_infos: list of GateInfo (if return_gate_info)
        """
        B, S = input_ids.shape
        device = input_ids.device

        # Embeddings
        positions = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        x = self.emb_drop(x)

        # Causal mask
        causal_mask = self._get_causal_mask(S, device)

        gate_infos = []

        for i, (block, gate) in enumerate(zip(self.blocks, self.gates)):
            if fixed_skip_prob is not None:
                # Condition A: fixed stochastic depth
                skip_prob = torch.full((B,), fixed_skip_prob, device=device)
            else:
                # Compute gate from pooled hidden state + telemetry
                h_pooled = x.mean(dim=1)  # [B, hidden_dim]
                skip_prob = gate(h_pooled, telemetry_2d)  # [B]

            if self.training:
                # Sample Bernoulli: 1 = skip, 0 = fire
                skip_sample = torch.bernoulli(skip_prob)  # [B]
                fired = (skip_sample == 0)  # [B] bool

                # Straight-Through Estimator: pass gradients through
                # We use (1 - skip_sample) as scaling, with STE for backprop
                fire_mask = (1.0 - skip_sample).detach() + (1.0 - skip_prob) - (1.0 - skip_prob).detach()
                # fire_mask: forward uses Bernoulli sample, backward uses skip_prob gradient

                # Apply block with masking
                block_out = block(x, attn_mask=causal_mask)
                # fire_mask is [B], need to broadcast to [B, S, hidden_dim]
                x = x + fire_mask.unsqueeze(-1).unsqueeze(-1) * (block_out - x)
            else:
                # Deterministic: fire if skip_prob < 0.5
                fired = (skip_prob < 0.5)  # [B] bool

                if fired.any():
                    block_out = block(x, attn_mask=causal_mask)
                    fire_float = fired.float().unsqueeze(-1).unsqueeze(-1)
                    x = x * (1 - fire_float) + block_out * fire_float

            if return_gate_info:
                gate_infos.append(GateInfo(
                    layer_idx=i,
                    skip_prob=skip_prob,
                    fired=fired,
                ))

        # Output head
        x = self.ln_out(x)
        logits = self.head(x)

        if return_gate_info:
            return logits, gate_infos
        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def gate_parameters(self):
        """Return only gate parameters (for REINFORCE optimizer)."""
        for gate in self.gates:
            yield from gate.parameters()

    def transformer_parameters(self):
        """Return non-gate parameters (for standard optimizer)."""
        gate_param_ids = {id(p) for p in self.gate_parameters()}
        for p in self.parameters():
            if id(p) not in gate_param_ids:
                yield p


# ===========================================================================
# Telemetry helpers
# ===========================================================================

def read_telemetry_2d(telemetry, device: torch.device,
                      power_cap_w: float = 150.0,
                      temp_max_c: float = 100.0) -> torch.Tensor:
    """Read current GPU telemetry and return normalized [power, temp] tensor."""
    try:
        sample = telemetry.read_sample()
        power_norm = min(sample.power_w / power_cap_w, 1.0)
        temp_norm = min(sample.temp_edge_c / temp_max_c, 1.0)
        return torch.tensor([power_norm, temp_norm], device=device, dtype=torch.float32)
    except Exception:
        return torch.tensor([0.5, 0.5], device=device, dtype=torch.float32)


# ===========================================================================
# Training
# ===========================================================================

@dataclass
class EpochStats:
    epoch: int = 0
    condition: str = ""
    train_loss: float = 0.0
    train_ppl: float = 0.0
    val_loss: float = 0.0
    val_ppl: float = 0.0
    avg_energy_j: float = 0.0
    avg_fire_rate: float = 0.0
    per_layer_fire_rates: List[float] = field(default_factory=list)
    avg_power_w: float = 0.0
    avg_temp_c: float = 0.0
    # Firing rates binned by temperature
    fire_rate_by_temp_bin: Dict[str, float] = field(default_factory=dict)


def evaluate(model: StochasticDepthTransformer, batches, vocab_size: int,
             telemetry, device: torch.device,
             fixed_skip_prob: Optional[float] = None) -> Tuple[float, float, List[float]]:
    """Evaluate model, return (loss, perplexity, per_layer_fire_rates)."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    layer_fire_counts = [0.0] * model.num_layers
    layer_total = [0] * model.num_layers

    telemetry_2d = read_telemetry_2d(telemetry, device)

    with torch.no_grad():
        for inp, tgt in batches:
            logits, gate_infos = model(inp, telemetry_2d=telemetry_2d,
                                        fixed_skip_prob=fixed_skip_prob)
            loss = F.cross_entropy(logits.view(-1, vocab_size), tgt.reshape(-1))
            total_loss += loss.item() * tgt.numel()
            total_tokens += tgt.numel()

            for gi in gate_infos:
                layer_fire_counts[gi.layer_idx] += gi.fired.float().sum().item()
                layer_total[gi.layer_idx] += gi.fired.numel()

    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(min(avg_loss, 20.0))
    fire_rates = [
        layer_fire_counts[i] / max(layer_total[i], 1)
        for i in range(model.num_layers)
    ]
    return avg_loss, ppl, fire_rates


def train_epoch(model: StochasticDepthTransformer, batches, vocab_size: int,
                opt_transformer, opt_gate, telemetry, device: torch.device,
                lambda_energy: float, gate_weight: float,
                fixed_skip_prob: Optional[float] = None,
                reward_baseline: float = 0.0) -> Tuple[float, float, float, List[float], List[dict]]:
    """
    Train one epoch.

    Returns: (avg_loss, avg_energy, new_reward_baseline, per_layer_fire_rates, temp_fire_records)
    """
    model.train()
    total_loss = 0.0
    total_tokens = 0
    energy_accum = []
    layer_fire_counts = [0.0] * model.num_layers
    layer_total = [0] * model.num_layers
    temp_fire_records = []  # [{temp, fire_rate}] for binning

    # Use real energy meter or fallback
    MeterClass = EnergyMeter if TELEMETRY_AVAILABLE else FallbackEnergyMeter

    num_batches = len(batches)
    for batch_idx, (inp, tgt) in enumerate(batches):
        telemetry_2d = read_telemetry_2d(telemetry, device)

        # ---- Forward pass (task loss) ----
        logits, gate_infos = model(inp, telemetry_2d=telemetry_2d,
                                    fixed_skip_prob=fixed_skip_prob)
        loss_task = F.cross_entropy(logits.view(-1, vocab_size), tgt.reshape(-1))

        # ---- Energy measurement (separate forward) ----
        with MeterClass(telemetry) as meter:
            with torch.no_grad():
                _ = model(inp, telemetry_2d=telemetry_2d,
                          fixed_skip_prob=fixed_skip_prob,
                          return_gate_info=False)
        energy_j = meter.energy_j
        energy_accum.append(energy_j)

        # Record temperature and firing for binning
        try:
            sample = telemetry.read_sample()
            cur_temp = sample.temp_edge_c
            cur_power = sample.power_w
        except Exception:
            cur_temp = 55.0
            cur_power = 30.0

        batch_fire_rate = 0.0
        for gi in gate_infos:
            fr = gi.fired.float().mean().item()
            batch_fire_rate += fr
            layer_fire_counts[gi.layer_idx] += gi.fired.float().sum().item()
            layer_total[gi.layer_idx] += gi.fired.numel()
        batch_fire_rate /= max(len(gate_infos), 1)
        temp_fire_records.append({"temp_c": cur_temp, "fire_rate": batch_fire_rate})

        # ---- REINFORCE for gates ----
        reward = -(loss_task.item() + lambda_energy * energy_j)
        advantage = reward - reward_baseline
        # Update baseline with exponential moving average
        reward_baseline = 0.95 * reward_baseline + 0.05 * reward

        if fixed_skip_prob is None:
            # Compute REINFORCE gate loss
            gate_loss = torch.tensor(0.0, device=device)
            for gi in gate_infos:
                fired_float = gi.fired.float()  # [B]
                # log_prob of action taken
                log_prob_fire = torch.log(1.0 - gi.skip_prob + 1e-8)
                log_prob_skip = torch.log(gi.skip_prob + 1e-8)
                log_prob = fired_float * log_prob_fire + (1.0 - fired_float) * log_prob_skip
                gate_loss = gate_loss - (log_prob * advantage).mean()

            total_backward_loss = loss_task + gate_weight * gate_loss
        else:
            total_backward_loss = loss_task

        # ---- Backward ----
        opt_transformer.zero_grad()
        if opt_gate is not None:
            opt_gate.zero_grad()
        total_backward_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        opt_transformer.step()
        if opt_gate is not None:
            opt_gate.step()

        total_loss += loss_task.item() * tgt.numel()
        total_tokens += tgt.numel()

        # Progress every 20% of batches
        if (batch_idx + 1) % max(1, num_batches // 5) == 0:
            cur_avg = total_loss / total_tokens
            cur_ppl = math.exp(min(cur_avg, 20.0))
            print(f"    Batch {batch_idx + 1}/{num_batches}  "
                  f"loss={cur_avg:.4f}  ppl={cur_ppl:.1f}  "
                  f"E={energy_j:.4f}J  temp={cur_temp:.1f}C  "
                  f"fire_rate={batch_fire_rate:.2%}")

    avg_loss = total_loss / max(total_tokens, 1)
    avg_energy = float(np.mean(energy_accum)) if energy_accum else 0.0
    fire_rates = [
        layer_fire_counts[i] / max(layer_total[i], 1)
        for i in range(model.num_layers)
    ]

    return avg_loss, avg_energy, reward_baseline, fire_rates, temp_fire_records


# ===========================================================================
# Conditions
# ===========================================================================

@dataclass
class ConditionConfig:
    name: str
    label: str
    use_hidden: bool
    use_telemetry: bool
    fixed_skip_prob: Optional[float]  # None = learned gates


CONDITIONS = [
    ConditionConfig("A_fixed", "Fixed p=0.1", False, False, 0.1),
    ConditionConfig("B_hidden_only", "Hidden-only gate", True, False, None),
    ConditionConfig("C_telemetry_only", "Telemetry-only gate", False, True, None),
    ConditionConfig("D_full", "Full (hidden+telemetry)", True, True, None),
]


# ===========================================================================
# Main experiment
# ===========================================================================

def run_experiment(args):
    print("=" * 80)
    print("Z906: Stochastic Depth with Energy-Aware Bernoulli Gates")
    print("=" * 80)

    # ---- Device ----
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[CONFIG] Device: {device}")
    print(f"[CONFIG] Epochs: {args.epochs}")
    print(f"[CONFIG] Batch size: {args.batch_size}")
    print(f"[CONFIG] Seq length: {args.seq_len}")
    print(f"[CONFIG] Lambda energy: {args.lambda_energy}")
    print(f"[CONFIG] Gate weight: {args.gate_weight}")

    # ---- Telemetry ----
    if TELEMETRY_AVAILABLE:
        try:
            telemetry = SysfsHwmonTelemetry(sample_rate_hz=50)
            sample = telemetry.read_sample()
            print(f"[TELEMETRY] Real AMD telemetry active: "
                  f"{sample.power_w:.1f}W, {sample.temp_edge_c:.1f}C")
        except Exception as e:
            print(f"[WARN] Real telemetry init failed: {e}")
            telemetry = FallbackTelemetry()
            print("[TELEMETRY] Using fallback telemetry")
    else:
        telemetry = FallbackTelemetry()
        print("[TELEMETRY] Using fallback telemetry")

    # ---- Data ----
    train_data, val_data = get_shakespeare_data(args.seq_len)
    train_batches = make_batches(train_data, args.batch_size, args.seq_len, device)
    val_batches = make_batches(val_data, args.batch_size, args.seq_len, device)
    print(f"[DATA] Train batches: {len(train_batches)}, Val batches: {len(val_batches)}")

    if not train_batches or not val_batches:
        print("[ERROR] No batches created. Check data size vs batch/seq params.")
        return

    # ---- Output directories ----
    results_dir = Path(__file__).parent.parent / "results"
    ckpt_dir = Path(__file__).parent.parent / "checkpoints"
    results_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ---- Run all conditions ----
    all_results = {}
    vocab_size = 256

    for cond in CONDITIONS:
        print()
        print("=" * 80)
        print(f"CONDITION {cond.name}: {cond.label}")
        print("=" * 80)

        # Build model
        model = StochasticDepthTransformer(
            vocab_size=vocab_size,
            hidden_dim=256,
            num_layers=12,
            num_heads=4,
            ff_dim=1024,
            max_seq_len=args.seq_len,
            dropout=0.1,
            use_hidden_gate=cond.use_hidden,
            use_telemetry_gate=cond.use_telemetry,
        ).to(device)

        n_params = model.count_parameters()
        n_gate_params = sum(p.numel() for p in model.gate_parameters())
        print(f"[MODEL] Total params: {n_params:,} "
              f"(gates: {n_gate_params:,})")

        # Optimizers
        opt_transformer = torch.optim.AdamW(
            model.transformer_parameters(), lr=3e-4, weight_decay=0.01
        )
        if cond.fixed_skip_prob is None:
            opt_gate = torch.optim.Adam(
                model.gate_parameters(), lr=1e-3
            )
        else:
            opt_gate = None

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt_transformer, T_max=args.epochs, eta_min=1e-5
        )

        reward_baseline = 0.0
        epoch_stats_list = []
        all_temp_fire_records = []

        for epoch in range(1, args.epochs + 1):
            print(f"\n--- Epoch {epoch}/{args.epochs} ---")
            t0 = time.perf_counter()

            # Train
            train_loss, avg_energy, reward_baseline, train_fire_rates, temp_fire_records = \
                train_epoch(
                    model, train_batches, vocab_size,
                    opt_transformer, opt_gate, telemetry, device,
                    args.lambda_energy, args.gate_weight,
                    fixed_skip_prob=cond.fixed_skip_prob,
                    reward_baseline=reward_baseline,
                )
            all_temp_fire_records.extend(temp_fire_records)

            # Validate
            val_loss, val_ppl, val_fire_rates = evaluate(
                model, val_batches, vocab_size, telemetry, device,
                fixed_skip_prob=cond.fixed_skip_prob,
            )

            train_ppl = math.exp(min(train_loss, 20.0))
            elapsed = time.perf_counter() - t0
            avg_fire = float(np.mean(val_fire_rates))

            # Read telemetry for stats
            try:
                sample = telemetry.read_sample()
                cur_power = sample.power_w
                cur_temp = sample.temp_edge_c
            except Exception:
                cur_power = 0.0
                cur_temp = 0.0

            # Bin fire rates by temperature
            temp_bins = {"<50C": [], "50-60C": [], "60-70C": [], "70-80C": [], ">80C": []}
            for rec in temp_fire_records:
                t = rec["temp_c"]
                if t < 50:
                    temp_bins["<50C"].append(rec["fire_rate"])
                elif t < 60:
                    temp_bins["50-60C"].append(rec["fire_rate"])
                elif t < 70:
                    temp_bins["60-70C"].append(rec["fire_rate"])
                elif t < 80:
                    temp_bins["70-80C"].append(rec["fire_rate"])
                else:
                    temp_bins[">80C"].append(rec["fire_rate"])
            fire_rate_by_temp = {
                k: float(np.mean(v)) if v else None
                for k, v in temp_bins.items()
            }

            stats = EpochStats(
                epoch=epoch,
                condition=cond.name,
                train_loss=train_loss,
                train_ppl=train_ppl,
                val_loss=val_loss,
                val_ppl=val_ppl,
                avg_energy_j=avg_energy,
                avg_fire_rate=avg_fire,
                per_layer_fire_rates=val_fire_rates,
                avg_power_w=cur_power,
                avg_temp_c=cur_temp,
                fire_rate_by_temp_bin=fire_rate_by_temp,
            )
            epoch_stats_list.append(stats)

            # Print summary
            layer_str = " ".join(f"{r:.0%}" for r in val_fire_rates)
            print(f"  Train PPL: {train_ppl:.2f}  |  Val PPL: {val_ppl:.2f}")
            print(f"  Avg Energy: {avg_energy:.4f} J  |  Avg Fire Rate: {avg_fire:.2%}")
            print(f"  Layer firing: [{layer_str}]")
            print(f"  Power: {cur_power:.1f}W  Temp: {cur_temp:.1f}C  Time: {elapsed:.1f}s")
            active_bins = {k: f"{v:.2%}" for k, v in fire_rate_by_temp.items() if v is not None}
            if active_bins:
                print(f"  Fire rate by temp: {active_bins}")

            scheduler.step()

        # Save checkpoint
        ckpt_path = ckpt_dir / f"z906_{cond.name}.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "condition": cond.name,
            "epoch": args.epochs,
            "val_ppl": epoch_stats_list[-1].val_ppl if epoch_stats_list else None,
        }, ckpt_path)
        print(f"\n  Checkpoint saved: {ckpt_path}")

        # Store results for this condition
        all_results[cond.name] = {
            "label": cond.label,
            "use_hidden": cond.use_hidden,
            "use_telemetry": cond.use_telemetry,
            "fixed_skip_prob": cond.fixed_skip_prob,
            "n_params": n_params,
            "n_gate_params": n_gate_params,
            "epochs": [asdict(s) for s in epoch_stats_list],
            "final_val_ppl": epoch_stats_list[-1].val_ppl if epoch_stats_list else None,
            "final_fire_rate": epoch_stats_list[-1].avg_fire_rate if epoch_stats_list else None,
            "final_energy_j": epoch_stats_list[-1].avg_energy_j if epoch_stats_list else None,
            "final_per_layer_fire_rates": epoch_stats_list[-1].per_layer_fire_rates if epoch_stats_list else None,
            "fire_rate_by_temp": epoch_stats_list[-1].fire_rate_by_temp_bin if epoch_stats_list else None,
        }

    # ==================================================================
    # Summary comparison
    # ==================================================================
    print()
    print("=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)

    header = f"{'Condition':<30} {'Val PPL':>10} {'Fire Rate':>10} {'Energy(J)':>10} {'Pareto':>8}"
    print(header)
    print("-" * len(header))

    # Compute Pareto dominance (lower PPL AND lower energy is better)
    condition_points = []
    for cname, cdata in all_results.items():
        ppl = cdata["final_val_ppl"] or float("inf")
        energy = cdata["final_energy_j"] or float("inf")
        condition_points.append((cname, ppl, energy))

    def is_pareto_optimal(idx, points):
        """Check if point at idx is not dominated by any other."""
        p_ppl, p_eng = points[idx][1], points[idx][2]
        for j, (_, o_ppl, o_eng) in enumerate(points):
            if j != idx and o_ppl <= p_ppl and o_eng <= p_eng:
                if o_ppl < p_ppl or o_eng < p_eng:
                    return False
        return True

    pareto_flags = {
        condition_points[i][0]: is_pareto_optimal(i, condition_points)
        for i in range(len(condition_points))
    }

    for cname, cdata in all_results.items():
        ppl_str = f"{cdata['final_val_ppl']:.2f}" if cdata["final_val_ppl"] else "N/A"
        fire_str = f"{cdata['final_fire_rate']:.2%}" if cdata["final_fire_rate"] is not None else "N/A"
        eng_str = f"{cdata['final_energy_j']:.4f}" if cdata["final_energy_j"] is not None else "N/A"
        pareto_str = "YES" if pareto_flags.get(cname) else "no"
        label = cdata["label"]
        print(f"{label:<30} {ppl_str:>10} {fire_str:>10} {eng_str:>10} {pareto_str:>8}")

    # Per-layer firing heatmap (text)
    print()
    print("Per-layer firing rates (final epoch):")
    print(f"{'Condition':<30} " + " ".join(f"L{i:02d}" for i in range(12)))
    print("-" * (30 + 12 * 5))
    for cname, cdata in all_results.items():
        rates = cdata.get("final_per_layer_fire_rates", [])
        if rates:
            rate_str = " ".join(f"{r:.0%}" for r in rates)
        else:
            rate_str = "N/A"
        print(f"{cdata['label']:<30} {rate_str}")

    # Temperature-dependent firing
    print()
    print("Fire rate by temperature bin (final epoch):")
    bins = ["<50C", "50-60C", "60-70C", "70-80C", ">80C"]
    print(f"{'Condition':<30} " + " ".join(f"{b:>8}" for b in bins))
    print("-" * (30 + len(bins) * 9))
    for cname, cdata in all_results.items():
        temp_data = cdata.get("fire_rate_by_temp", {})
        vals = []
        for b in bins:
            v = temp_data.get(b)
            vals.append(f"{v:.2%}" if v is not None else "---")
        print(f"{cdata['label']:<30} " + " ".join(f"{v:>8}" for v in vals))

    # Hypothesis evaluation
    print()
    print("=" * 80)
    print("HYPOTHESIS EVALUATION")
    print("=" * 80)

    d_ppl = all_results.get("D_full", {}).get("final_val_ppl")
    d_energy = all_results.get("D_full", {}).get("final_energy_j")

    for baseline in ["A_fixed", "B_hidden_only", "C_telemetry_only"]:
        b_ppl = all_results.get(baseline, {}).get("final_val_ppl")
        b_energy = all_results.get(baseline, {}).get("final_energy_j")
        if d_ppl and b_ppl and d_energy and b_energy:
            ppl_delta = ((d_ppl - b_ppl) / b_ppl) * 100
            eng_delta = ((d_energy - b_energy) / b_energy) * 100
            better_ppl = d_ppl < b_ppl
            better_eng = d_energy < b_energy
            verdict = "BETTER" if (better_ppl or better_eng) else "WORSE"
            if better_ppl and better_eng:
                verdict = "DOMINATES"
            print(f"  D vs {baseline}: PPL {ppl_delta:+.1f}%, Energy {eng_delta:+.1f}% -> {verdict}")

    d_pareto = pareto_flags.get("D_full", False)
    print()
    if d_pareto:
        print("  RESULT: Full model (D) is Pareto-optimal. Hypothesis SUPPORTED.")
    else:
        others_pareto = [k for k, v in pareto_flags.items() if v and k != "D_full"]
        print(f"  RESULT: Full model (D) is NOT Pareto-optimal.")
        print(f"  Pareto-optimal conditions: {others_pareto}")
        print("  Hypothesis NEEDS FURTHER INVESTIGATION.")

    # ---- Save results ----
    output = {
        "experiment": "z906_stochastic_depth_energy",
        "hypothesis": "Per-layer Bernoulli gates conditioned on both hidden state AND "
                      "telemetry achieve better energy-quality Pareto curves than "
                      "fixed stochastic depth or either signal alone.",
        "config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "lambda_energy": args.lambda_energy,
            "gate_weight": args.gate_weight,
            "device": str(device),
            "telemetry": "real" if TELEMETRY_AVAILABLE else "fallback",
        },
        "conditions": all_results,
        "pareto_optimal": {k: v for k, v in pareto_flags.items()},
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    results_path = results_dir / "z906_stochastic_depth_energy.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n[SAVED] Results: {results_path}")
    print("[DONE]")


# ===========================================================================
# Entry point
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Z906: Stochastic Depth with Energy-Aware Bernoulli Gates"
    )
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs per condition (default: 10)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size (default: 32)")
    parser.add_argument("--seq-len", type=int, default=256,
                        help="Sequence length in bytes (default: 256)")
    parser.add_argument("--lambda-energy", type=float, default=0.01,
                        help="Energy penalty weight in REINFORCE reward (default: 0.01)")
    parser.add_argument("--gate-weight", type=float, default=0.1,
                        help="Weight of REINFORCE gate loss in total loss (default: 0.1)")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: auto, cuda, cpu (default: auto)")
    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
