#!/usr/bin/env python3
"""
z1701: Active Inference with Real GPU Hardware-in-the-Loop
==========================================================

Implements Friston's Free Energy Principle for embodied AI:
- A world model predicts BOTH next tokens AND next body state (GPU telemetry)
- Actions are selected to minimize Expected Free Energy (EFE)
- The model maintains homeostatic setpoints via active inference

Three conditions:
  A: Active Inference — full world model, actions minimize EFE
  B: Reactive        — no world model, action = argmax(action_logits)
  C: No Action       — world model but random actions (tests prediction)

Metrics: task perplexity, body prediction MSE, free energy trajectory,
action entropy, preferred state deviation, J/token, tokens/sec.

Author: FEEL Research Team
Date: 2026-02-04
"""

import os
import sys
import json
import time
import math
import signal
import traceback
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, List, Tuple
from collections import deque

# HSA override for gfx1151 compatibility
os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '11.0.0')

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.metabolic.film_transformer import (
    MetabolicTransformer, MetabolicConfig, create_metabolic_transformer,
)
from src.actuation.gpu_actuator import GPUActuator, EmbodiedGPUController, PerformanceLevel
from src.telemetry.sysfs_hwmon import SysfsHwmonTelemetry, GpuSample, EnergyMeter

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PROJECT_ROOT = Path(__file__).parent.parent


# =============================================================================
# Homeostatic Setpoints
# =============================================================================

PREFERRED_STATE = torch.tensor([
    0.3,   # power: ~15W / 50W (moderate)
    0.4,   # temp: ~40C / 100C (cool)
    0.6,   # freq: ~1800MHz / 3000MHz (medium-high)
    0.5,   # busy: 50%
    0.6,   # power_cap: moderate headroom
    0.0,   # no throttling
    0.0, 0.0, 0.0, 0.0,  # stable derivatives (power, temp, freq, busy)
    0.0,   # no thermal deviation
    0.5,   # 50% power headroom
], dtype=torch.float32)

ACTION_NAMES = ['ECO', 'BALANCED', 'PERFORMANCE', 'MAX']


# =============================================================================
# World Model Head
# =============================================================================

class WorldModelHead(nn.Module):
    """
    Predicts next body state from current hidden state + action.

    Outputs a Gaussian distribution (mean, log-variance) over the
    12-dim telemetry vector, enabling uncertainty-aware planning.
    """

    def __init__(self, hidden_dim: int = 256, telemetry_dim: int = 12,
                 num_actions: int = 4):
        super().__init__()
        self.num_actions = num_actions
        self.telemetry_dim = telemetry_dim

        self.net = nn.Sequential(
            nn.Linear(hidden_dim + num_actions, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(64, telemetry_dim)
        self.logvar_head = nn.Linear(64, telemetry_dim)

        # Initialize logvar to small values for stable initial predictions
        nn.init.constant_(self.logvar_head.bias, -2.0)

    def forward(self, hidden: torch.Tensor,
                action_onehot: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden: [batch, hidden_dim] from transformer last-token
            action_onehot: [batch, num_actions] one-hot action

        Returns:
            mean: [batch, telemetry_dim]
            logvar: [batch, telemetry_dim]
        """
        x = torch.cat([hidden, action_onehot], dim=-1)
        h = self.net(x)
        mean = self.mean_head(h)
        logvar = self.logvar_head(h).clamp(-6.0, 2.0)
        return mean, logvar

    def predict_all_actions(self, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict next state for ALL actions simultaneously.

        Args:
            hidden: [batch, hidden_dim]

        Returns:
            means: [batch, num_actions, telemetry_dim]
            logvars: [batch, num_actions, telemetry_dim]
        """
        batch = hidden.size(0)
        device = hidden.device

        all_means = []
        all_logvars = []
        for a in range(self.num_actions):
            onehot = torch.zeros(batch, self.num_actions, device=device)
            onehot[:, a] = 1.0
            m, lv = self.forward(hidden, onehot)
            all_means.append(m)
            all_logvars.append(lv)

        means = torch.stack(all_means, dim=1)       # [B, A, T]
        logvars = torch.stack(all_logvars, dim=1)    # [B, A, T]
        return means, logvars


# =============================================================================
# Telemetry Helpers
# =============================================================================

def build_telemetry_vector(sample: GpuSample, prev_sample: Optional[GpuSample],
                           power_cap_max: float = 50.0) -> np.ndarray:
    """Build normalized 12-dim telemetry from GpuSample."""
    power_norm = sample.power_w / max(power_cap_max, 1.0)
    temp_norm = sample.temp_edge_c / 100.0
    freq_norm = sample.freq_sclk_mhz / 3000.0
    busy_norm = sample.gpu_busy_pct / 100.0
    pcap_norm = 0.6  # default
    throttle = 0.0

    # Derivatives (finite difference from previous sample)
    if prev_sample is not None:
        dt = max((sample.timestamp_ns - prev_sample.timestamp_ns) / 1e9, 0.001)
        d_power = (sample.power_w - prev_sample.power_w) / max(power_cap_max, 1.0) / dt
        d_temp = (sample.temp_edge_c - prev_sample.temp_edge_c) / 100.0 / dt
        d_freq = (sample.freq_sclk_mhz - prev_sample.freq_sclk_mhz) / 3000.0 / dt
        d_busy = (sample.gpu_busy_pct - prev_sample.gpu_busy_pct) / 100.0 / dt
    else:
        d_power = d_temp = d_freq = d_busy = 0.0

    thermal_dev = max(0, (sample.temp_edge_c - 80.0)) / 20.0
    power_headroom = max(0, 1.0 - power_norm)

    vec = np.array([
        power_norm, temp_norm, freq_norm, busy_norm,
        pcap_norm, throttle,
        np.clip(d_power, -1, 1), np.clip(d_temp, -1, 1),
        np.clip(d_freq, -1, 1), np.clip(d_busy, -1, 1),
        thermal_dev, power_headroom,
    ], dtype=np.float32)
    return vec


def apply_action_to_gpu(action_idx: int, controller: EmbodiedGPUController) -> bool:
    """Apply model-selected action to GPU. Returns True if successful."""
    perf_map = {
        0: ("low", 0.3),       # ECO
        1: ("balanced", 0.5),  # BALANCED
        2: ("high", 0.7),      # PERFORMANCE
        3: ("high", 1.0),      # MAX
    }
    level_str, power_frac = perf_map.get(action_idx, ("balanced", 0.5))
    try:
        controller.apply_action(power_fraction=power_frac, perf_level=level_str)
        return True
    except Exception:
        return False


# =============================================================================
# Data Loading
# =============================================================================

def load_tinyshakespeare(path: str = None, seq_len: int = 256) -> Tuple[torch.Tensor, int]:
    """Load TinyShakespeare as char-level data."""
    if path is None:
        path = str(PROJECT_ROOT / 'data' / 'tinyshakespeare.txt')
    with open(path, 'r') as f:
        text = f.read()
    data = torch.tensor([ord(c) % 256 for c in text], dtype=torch.long)
    return data, 256  # vocab_size


def make_batches(data: torch.Tensor, batch_size: int, seq_len: int) -> List[torch.Tensor]:
    """Slice data into non-overlapping [batch_size, seq_len+1] chunks."""
    total = data.numel()
    chunk = seq_len + 1  # +1 for target shift
    usable = (total // (batch_size * chunk)) * batch_size * chunk
    data = data[:usable].view(-1, chunk)
    # Shuffle rows
    perm = torch.randperm(data.size(0))
    data = data[perm]
    # Group into batches
    n_batches = data.size(0) // batch_size
    batches = []
    for i in range(n_batches):
        batches.append(data[i * batch_size : (i + 1) * batch_size])
    return batches


# =============================================================================
# Expected Free Energy
# =============================================================================

def compute_efe(means: torch.Tensor, logvars: torch.Tensor,
                preferred: torch.Tensor, task_logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
    """
    Compute Expected Free Energy for each action.

    EFE(a) = epistemic_value(a) + pragmatic_value(a)
           = E_q[entropy of predicted state]
             + E_q[KL from predicted to preferred]

    Args:
        means: [batch, num_actions, telemetry_dim]
        logvars: [batch, num_actions, telemetry_dim]
        preferred: [telemetry_dim] homeostatic setpoints
        task_logits: [batch, seq_len, vocab] (for task uncertainty)
        targets: [batch, seq_len] target tokens

    Returns:
        efe: [batch, num_actions] — lower is better
    """
    preferred = preferred.to(means.device)
    batch, num_actions, tdim = means.shape

    # Pragmatic value: squared distance from preferred state
    pref = preferred.unsqueeze(0).unsqueeze(0).expand_as(means)
    pragmatic = ((means - pref) ** 2).sum(dim=-1)  # [B, A]

    # Epistemic value: predicted uncertainty (high logvar = high info gain)
    # We want actions that REDUCE uncertainty, so penalize high variance
    epistemic = logvars.sum(dim=-1)  # [B, A]

    # Task uncertainty penalty: entropy of token predictions
    with torch.no_grad():
        token_probs = F.softmax(task_logits[:, -1, :], dim=-1)
        task_entropy = -(token_probs * (token_probs + 1e-8).log()).sum(dim=-1)  # [B]
    task_penalty = task_entropy.unsqueeze(1).expand(batch, num_actions)

    # EFE = pragmatic + 0.5 * epistemic + 0.1 * task_uncertainty
    efe = pragmatic + 0.5 * epistemic + 0.1 * task_penalty
    return efe


# =============================================================================
# Training One Condition
# =============================================================================

@dataclass
class ConditionResult:
    name: str
    task_perplexity: List[float] = field(default_factory=list)
    body_pred_mse: List[float] = field(default_factory=list)
    free_energy: List[float] = field(default_factory=list)
    action_entropy: List[float] = field(default_factory=list)
    pref_deviation: List[float] = field(default_factory=list)
    energy_j: float = 0.0
    total_tokens: int = 0
    total_time_s: float = 0.0
    epoch_perplexities: List[float] = field(default_factory=list)


def run_condition(
    name: str,
    mode: str,  # 'active_inference', 'reactive', 'no_action'
    data: torch.Tensor,
    telemetry: SysfsHwmonTelemetry,
    controller: EmbodiedGPUController,
    n_epochs: int = 5,
    batch_size: int = 4,
    seq_len: int = 256,
    lr_model: float = 3e-4,
    lr_world: float = 1e-3,
    alpha: float = 1.0,   # body prediction loss weight
    beta: float = 0.5,    # EFE loss weight
) -> ConditionResult:
    """Run one experimental condition."""
    print(f"\n{'='*70}")
    print(f"  CONDITION {name}: mode={mode}")
    print(f"{'='*70}")

    result = ConditionResult(name=name)
    use_world_model = mode in ('active_inference', 'no_action')
    use_efe_actions = mode == 'active_inference'

    # Build model
    config = MetabolicConfig(
        hidden_dim=256, num_layers=6, num_heads=4,
        ff_dim=1024, telemetry_dim=12, num_actions=4,
    )
    model = MetabolicTransformer(config).to(DEVICE)

    # World model head (only for A and C)
    world_model = None
    if use_world_model:
        world_model = WorldModelHead(
            hidden_dim=config.hidden_dim,
            telemetry_dim=config.telemetry_dim,
            num_actions=config.num_actions,
        ).to(DEVICE)

    # Optimizers
    model_params = list(model.parameters())
    optimizer_model = torch.optim.Adam(model_params, lr=lr_model)

    optimizer_world = None
    if world_model is not None:
        optimizer_world = torch.optim.Adam(world_model.parameters(), lr=lr_world)

    preferred = PREFERRED_STATE.to(DEVICE)

    # Telemetry state
    prev_sample = None
    prev_telemetry_vec = None
    actuation_success = True

    t_start = time.time()
    total_tokens = 0
    total_loss_accum = 0.0
    total_batches = 0

    # Energy tracking
    energy_meter_start = time.time_ns()
    telemetry.reset_accumulator()
    telemetry.start_continuous_sampling()

    try:
        for epoch in range(n_epochs):
            batches = make_batches(data, batch_size, seq_len)
            epoch_losses = []
            epoch_body_mse = []
            epoch_fe = []
            epoch_act_ent = []
            epoch_pref_dev = []

            model.train()
            if world_model is not None:
                world_model.train()

            for bi, batch_chunk in enumerate(batches):
                batch_chunk = batch_chunk.to(DEVICE)
                inputs = batch_chunk[:, :-1]   # [B, seq_len]
                targets = batch_chunk[:, 1:]   # [B, seq_len]

                # --- Step 1: Read GPU telemetry ---
                sample = telemetry.read_sample()
                telem_np = build_telemetry_vector(sample, prev_sample)
                telem_t = torch.from_numpy(telem_np).float().to(DEVICE)
                telem_batch = telem_t.unsqueeze(0).expand(batch_size, -1)

                # --- Step 2: Forward pass ---
                out = model(inputs, telemetry=telem_batch, return_hidden=True)
                logits = out['logits']           # [B, S, V]
                action_logits = out['action_logits']  # [B, 4]
                hidden = out['hidden'][:, -1, :]  # [B, H] — last token hidden

                # Task loss
                task_loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1),
                )
                ppl = math.exp(min(task_loss.item(), 10.0))

                # --- Step 3 & 4: World model + action selection ---
                body_loss = torch.tensor(0.0, device=DEVICE)
                efe_loss = torch.tensor(0.0, device=DEVICE)
                selected_action = 1  # default BALANCED

                if use_world_model and world_model is not None:
                    # Predict next state for all actions
                    means, logvars = world_model.predict_all_actions(hidden.detach())

                    if use_efe_actions:
                        # Active inference: select action minimizing EFE
                        efe = compute_efe(means, logvars, preferred, logits.detach(), targets)
                        selected_action = efe.mean(dim=0).argmin().item()
                        efe_loss = efe.mean()
                    else:
                        # No-action mode: random action
                        selected_action = torch.randint(0, 4, (1,)).item()

                    # Body prediction loss (trained on PREVIOUS step's prediction)
                    if prev_telemetry_vec is not None:
                        actual_next = torch.from_numpy(prev_telemetry_vec).float().to(DEVICE)
                        actual_next = actual_next.unsqueeze(0).expand(batch_size, -1)

                        # What we predicted last step for the action we took
                        prev_action_oh = torch.zeros(batch_size, 4, device=DEVICE)
                        prev_action_oh[:, selected_action] = 1.0
                        pred_mean, pred_logvar = world_model(hidden.detach(), prev_action_oh)
                        body_loss = F.mse_loss(pred_mean, actual_next)
                else:
                    # Reactive: just use action head
                    with torch.no_grad():
                        selected_action = action_logits.mean(dim=0).argmax().item()

                # --- Step 5: Execute action ---
                if use_efe_actions or mode == 'reactive':
                    actuation_success = apply_action_to_gpu(selected_action, controller)

                # --- Step 6: Observe next telemetry (for next iteration) ---
                prev_sample = sample
                prev_telemetry_vec = telem_np.copy()

                # --- Step 7: Combined loss ---
                total_loss = task_loss + alpha * body_loss + beta * efe_loss

                optimizer_model.zero_grad()
                if optimizer_world is not None:
                    optimizer_world.zero_grad()

                total_loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if world_model is not None:
                    torch.nn.utils.clip_grad_norm_(world_model.parameters(), 1.0)

                optimizer_model.step()
                if optimizer_world is not None:
                    optimizer_world.step()

                # --- Metrics ---
                total_tokens += inputs.numel()
                total_batches += 1

                # Action entropy
                act_probs = F.softmax(action_logits.detach().mean(dim=0), dim=-1)
                act_ent = -(act_probs * (act_probs + 1e-8).log()).sum().item()

                # Preferred state deviation
                pref_dev = ((telem_t - preferred) ** 2).mean().item()

                # Free energy = task_loss + complexity
                fe = task_loss.item() + alpha * body_loss.item()

                epoch_losses.append(task_loss.item())
                epoch_body_mse.append(body_loss.item())
                epoch_fe.append(fe)
                epoch_act_ent.append(act_ent)
                epoch_pref_dev.append(pref_dev)

                # Print progress every 50 batches
                if (bi + 1) % 50 == 0:
                    avg_loss = np.mean(epoch_losses[-50:])
                    avg_bm = np.mean(epoch_body_mse[-50:])
                    print(f"  [{name}] Epoch {epoch+1}/{n_epochs} "
                          f"Batch {bi+1}/{len(batches)} | "
                          f"Loss={avg_loss:.4f} PPL={math.exp(min(avg_loss, 10)):.1f} | "
                          f"BodyMSE={avg_bm:.4f} | "
                          f"Action={ACTION_NAMES[selected_action]} "
                          f"Ent={act_ent:.3f} | "
                          f"PrefDev={pref_dev:.4f}")

            # Epoch summary
            ep_ppl = math.exp(min(np.mean(epoch_losses), 10.0))
            result.epoch_perplexities.append(ep_ppl)
            result.task_perplexity.append(np.mean(epoch_losses))
            result.body_pred_mse.append(np.mean(epoch_body_mse))
            result.free_energy.append(np.mean(epoch_fe))
            result.action_entropy.append(np.mean(epoch_act_ent))
            result.pref_deviation.append(np.mean(epoch_pref_dev))

            print(f"\n  [{name}] === Epoch {epoch+1} Summary ===")
            print(f"    Task PPL:        {ep_ppl:.2f}")
            print(f"    Body pred MSE:   {np.mean(epoch_body_mse):.6f}")
            print(f"    Free energy:     {np.mean(epoch_fe):.4f}")
            print(f"    Action entropy:  {np.mean(epoch_act_ent):.4f}")
            print(f"    Pref deviation:  {np.mean(epoch_pref_dev):.4f}")

    finally:
        telemetry.stop_continuous_sampling()

    t_end = time.time()
    result.total_time_s = t_end - t_start
    result.total_tokens = total_tokens

    # Energy
    result.energy_j = telemetry.get_accumulated_energy_j()

    return result


# =============================================================================
# Verdict & Reporting
# =============================================================================

def compute_verdict(results: Dict[str, ConditionResult]) -> Dict[str, bool]:
    """Evaluate pass/fail criteria."""
    a = results.get('A')
    b = results.get('B')

    verdicts = {}

    # 1. Body prediction MSE < 0.1
    if a and a.body_pred_mse:
        final_mse = a.body_pred_mse[-1]
        verdicts['body_model_learned'] = final_mse < 0.1
        print(f"\n  V1: Body model MSE = {final_mse:.6f} (threshold < 0.1) "
              f"-> {'PASS' if verdicts['body_model_learned'] else 'FAIL'}")

    # 2. Active inference pref deviation < Reactive
    if a and b and a.pref_deviation and b.pref_deviation:
        a_dev = a.pref_deviation[-1]
        b_dev = b.pref_deviation[-1]
        verdicts['homeostasis_better'] = a_dev < b_dev
        print(f"  V2: A pref_dev={a_dev:.4f} vs B pref_dev={b_dev:.4f} "
              f"-> {'PASS' if verdicts['homeostasis_better'] else 'FAIL'}")

    # 3. Free energy decreases over training
    if a and len(a.free_energy) >= 2:
        fe_decreasing = a.free_energy[-1] < a.free_energy[0]
        verdicts['fe_decreasing'] = fe_decreasing
        print(f"  V3: FE start={a.free_energy[0]:.4f} end={a.free_energy[-1]:.4f} "
              f"-> {'PASS' if fe_decreasing else 'FAIL'}")

    # 4. Action entropy < 1.0 (convergent policy)
    if a and a.action_entropy:
        final_ent = a.action_entropy[-1]
        verdicts['convergent_policy'] = final_ent < 1.0
        print(f"  V4: Action entropy = {final_ent:.4f} (threshold < 1.0) "
              f"-> {'PASS' if verdicts['convergent_policy'] else 'FAIL'}")

    return verdicts


def print_comparison_table(results: Dict[str, ConditionResult]):
    """Print side-by-side comparison of conditions."""
    print(f"\n{'='*80}")
    print(f"  COMPARISON TABLE")
    print(f"{'='*80}")

    header = f"{'Metric':<30}"
    for name in ['A', 'B', 'C']:
        if name in results:
            header += f"  {results[name].name:>16}"
    print(header)
    print('-' * 80)

    def row(label, getter, fmt=".4f"):
        line = f"{label:<30}"
        for name in ['A', 'B', 'C']:
            if name in results:
                val = getter(results[name])
                if val is not None:
                    line += f"  {val:>16{fmt}}"
                else:
                    line += f"  {'N/A':>16}"
        print(line)

    row("Final Task PPL",
        lambda r: r.epoch_perplexities[-1] if r.epoch_perplexities else None, ".2f")
    row("Final Body MSE",
        lambda r: r.body_pred_mse[-1] if r.body_pred_mse else None, ".6f")
    row("Final Free Energy",
        lambda r: r.free_energy[-1] if r.free_energy else None, ".4f")
    row("Final Action Entropy",
        lambda r: r.action_entropy[-1] if r.action_entropy else None, ".4f")
    row("Final Pref Deviation",
        lambda r: r.pref_deviation[-1] if r.pref_deviation else None, ".4f")

    # Energy efficiency
    for name in ['A', 'B', 'C']:
        if name in results:
            r = results[name]
            j_tok = (r.energy_j / r.total_tokens * 1000) if r.total_tokens > 0 else 0
            tok_s = r.total_tokens / r.total_time_s if r.total_time_s > 0 else 0
            print(f"  {r.name}: {j_tok:.3f} mJ/token, {tok_s:.0f} tok/s, "
                  f"{r.energy_j:.1f} J total, {r.total_time_s:.1f}s")

    print()


# =============================================================================
# Main
# =============================================================================

def main():
    print("="*70)
    print("  z1701: Active Inference with Real GPU Hardware-in-the-Loop")
    print(f"  Device: {DEVICE}")
    print(f"  Time: {datetime.now().isoformat()}")
    print("="*70)

    # --- Init hardware ---
    try:
        telemetry = SysfsHwmonTelemetry()
        sample = telemetry.read_sample()
        print(f"\n  GPU telemetry OK: {sample.power_w:.1f}W, "
              f"{sample.temp_edge_c:.1f}C, {sample.freq_sclk_mhz}MHz")
    except Exception as e:
        print(f"\n  WARNING: Telemetry init failed: {e}")
        print("  Continuing with fallback zero telemetry")
        telemetry = None

    controller = EmbodiedGPUController(card_id=0)
    actuator = controller.actuator

    # Test actuation (non-destructive)
    state = actuator.get_current_state()
    print(f"  GPU state: power_cap={state.power_cap_w:.1f}W, "
          f"perf_level={state.performance_level}")

    # --- Load data ---
    print("\n  Loading TinyShakespeare...")
    data, vocab_size = load_tinyshakespeare()
    print(f"  Data: {data.numel():,} chars, vocab={vocab_size}")

    # --- Run conditions ---
    N_EPOCHS = 5
    BATCH_SIZE = 4
    SEQ_LEN = 256
    COOLDOWN = 30

    results: Dict[str, ConditionResult] = {}

    # Use actuator as context manager for safe restore
    with actuator:
        # Condition A: Active Inference
        if telemetry:
            res_a = run_condition(
                name="A:ActiveInference", mode="active_inference",
                data=data, telemetry=telemetry, controller=controller,
                n_epochs=N_EPOCHS, batch_size=BATCH_SIZE, seq_len=SEQ_LEN,
            )
            results['A'] = res_a
        else:
            # Fallback with mock telemetry
            print("\n  Skipping condition A (no telemetry)")

        # Cooldown
        print(f"\n  Cooldown {COOLDOWN}s between conditions...")
        time.sleep(COOLDOWN)

        # Condition B: Reactive
        if telemetry:
            res_b = run_condition(
                name="B:Reactive", mode="reactive",
                data=data, telemetry=telemetry, controller=controller,
                n_epochs=N_EPOCHS, batch_size=BATCH_SIZE, seq_len=SEQ_LEN,
            )
            results['B'] = res_b
        else:
            print("\n  Skipping condition B (no telemetry)")

        # Cooldown
        print(f"\n  Cooldown {COOLDOWN}s between conditions...")
        time.sleep(COOLDOWN)

        # Condition C: No Action (world model, random actions)
        if telemetry:
            res_c = run_condition(
                name="C:NoAction", mode="no_action",
                data=data, telemetry=telemetry, controller=controller,
                n_epochs=N_EPOCHS, batch_size=BATCH_SIZE, seq_len=SEQ_LEN,
            )
            results['C'] = res_c
        else:
            print("\n  Skipping condition C (no telemetry)")

    # --- Results ---
    if results:
        print_comparison_table(results)
        verdicts = compute_verdict(results)
    else:
        verdicts = {}
        print("\n  No conditions were run (telemetry unavailable)")

    # --- Save ---
    output = {
        'experiment': 'z1701_active_inference',
        'timestamp': datetime.now().isoformat(),
        'device': str(DEVICE),
        'config': {
            'n_epochs': N_EPOCHS,
            'batch_size': BATCH_SIZE,
            'seq_len': SEQ_LEN,
            'hidden_dim': 256,
            'num_layers': 6,
            'lr_model': 3e-4,
            'lr_world': 1e-3,
            'alpha': 1.0,
            'beta': 0.5,
        },
        'conditions': {},
        'verdicts': verdicts,
    }

    for key, res in results.items():
        j_tok = (res.energy_j / res.total_tokens * 1000) if res.total_tokens > 0 else 0
        tok_s = res.total_tokens / res.total_time_s if res.total_time_s > 0 else 0
        output['conditions'][key] = {
            'name': res.name,
            'epoch_perplexities': res.epoch_perplexities,
            'task_perplexity_per_epoch': res.task_perplexity,
            'body_pred_mse_per_epoch': res.body_pred_mse,
            'free_energy_per_epoch': res.free_energy,
            'action_entropy_per_epoch': res.action_entropy,
            'pref_deviation_per_epoch': res.pref_deviation,
            'energy_j': res.energy_j,
            'total_tokens': res.total_tokens,
            'total_time_s': res.total_time_s,
            'mj_per_token': j_tok,
            'tokens_per_sec': tok_s,
        }

    # Convert numpy types for JSON serialization
    def jsonify(obj):
        if isinstance(obj, dict):
            return {k: jsonify(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [jsonify(v) for v in obj]
        elif hasattr(obj, 'item'):  # numpy scalar
            return obj.item()
        elif isinstance(obj, (bool,)):
            return obj
        elif hasattr(obj, '__bool__'):
            return bool(obj)
        return obj

    results_path = PROJECT_ROOT / 'results' / 'z1701_active_inference.json'
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(jsonify(output), f, indent=2)
    print(f"\n  Results saved to {results_path}")

    # --- Final verdict ---
    print(f"\n{'='*70}")
    print(f"  FINAL VERDICT")
    print(f"{'='*70}")
    all_pass = all(verdicts.values()) if verdicts else False
    n_pass = sum(1 for v in verdicts.values() if v)
    n_total = len(verdicts)
    print(f"  {n_pass}/{n_total} criteria passed")
    for k, v in verdicts.items():
        status = "PASS" if v else "FAIL"
        print(f"    {status}: {k}")

    overall = "PASS" if all_pass else "FAIL"
    print(f"\n  OVERALL: {overall}")
    print(f"{'='*70}")

    return 0 if all_pass else 1


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n  Interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n  FATAL: {e}")
        traceback.print_exc()
        sys.exit(1)
