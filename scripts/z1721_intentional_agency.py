#!/usr/bin/env python3
"""
z1721: Intentional Agency Experiment
=====================================

HYPOTHESIS: An embodied model develops goal-directed behavior -- it learns to
take actions that move body state toward a homeostatic setpoint.

Setpoint: target temperature 50C, target power 20W

Approach:
1. Define setpoint targets
2. Train MetabolicTransformer with action head (4 actions: LOW/BALANCED/HIGH/IDLE)
3. Reward: -|actual_temp - 50| - |actual_power - 20|
4. Test if model learns to consistently choose actions that approach setpoint

Three experimental conditions:
  A) EMBODIED -- FiLM conditioning ON, action head controls GPU performance
  B) DISEMBODIED -- Conditioning OFF, no telemetry (random actions expected)
  C) NO_ACTUATION -- Conditioning ON but cannot act (learned helplessness?)

Key Metrics:
- Action consistency: does model take coherent sequences of actions?
- Setpoint achievement: does temperature/power converge toward targets?
- Counterfactual: if setpoint changes, do actions adapt?

Four Verdicts:
  V1: EMBODIED action entropy decreases during training (more decisive)
  V2: EMBODIED achieves setpoint deviation < 10C and < 5W by end of training
  V3: DISEMBODIED has random/high entropy actions
  V4: After setpoint change mid-training, EMBODIED adapts within 50 batches

Author: Claude + ikaros
Date: 2026-02-05
"""

import sys
sys.path.insert(0, '/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy')

import os
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '11.0.0'

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import json
import math
import signal
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple
from collections import Counter, deque

from src.metabolic.film_transformer import (
    MetabolicTransformer, MetabolicConfig, BaselineTransformer,
    create_metabolic_transformer, get_best_device
)
from src.actuation.gpu_actuator import GPUActuator, PerformanceLevel, GPUState
from src.telemetry.sysfs_hwmon import SysfsHwmonTelemetry, GpuSample


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_PATH = Path('/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/data/tinyshakespeare.txt')
RESULTS_PATH = Path('/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/z1721_intentional_agency.json')

# Homeostatic setpoints (initial)
TEMP_SETPOINT_C = 50.0
POWER_SETPOINT_W = 20.0

# Alternative setpoint for mid-training change (V4 test)
ALT_TEMP_SETPOINT_C = 45.0
ALT_POWER_SETPOINT_W = 15.0

NUM_EPOCHS = 8
BATCH_SIZE = 4
SEQ_LEN = 256
LR = 3e-4
AGENCY_REWARD_WEIGHT = 0.1  # weight for homeostatic reward in loss
PRINT_EVERY = 50
COOLDOWN_S = 20
ACTION_NAMES = ['LOW', 'BALANCED', 'HIGH', 'IDLE']

# Setpoint change epoch for V4 test
SETPOINT_CHANGE_EPOCH = 5
ADAPTATION_BATCHES = 50  # batches to allow for adaptation

# Entropy threshold for "decisive" vs "random"
ENTROPY_DECISIVE_THRESHOLD = 1.0   # below this = decisive
ENTROPY_RANDOM_THRESHOLD = 1.8     # above this = random (max ~1.386 for 4 actions)


# ---------------------------------------------------------------------------
# Telemetry vector builder
# ---------------------------------------------------------------------------
def build_telemetry_vector(
    sample: GpuSample,
    state: GPUState,
    prev_sample: Optional[GpuSample],
    temp_setpoint: float,
    power_setpoint: float,
) -> torch.Tensor:
    """Build 12-dim telemetry for MetabolicTransformer FiLM conditioning."""
    # Derivatives (zero if no previous sample)
    if prev_sample is not None:
        dt = max((sample.timestamp_ns - prev_sample.timestamp_ns) / 1e9, 1e-6)
        d_power = (sample.power_w - prev_sample.power_w) / (50.0 * dt)
        d_temp = (sample.temp_edge_c - prev_sample.temp_edge_c) / (100.0 * dt)
        d_freq = (sample.freq_sclk_mhz - prev_sample.freq_sclk_mhz) / (3000.0 * dt)
        d_util = (sample.gpu_busy_pct - prev_sample.gpu_busy_pct) / (100.0 * dt)
    else:
        d_power = d_temp = d_freq = d_util = 0.0

    # Performance level encoding (LOW=0, BALANCED=0.5, HIGH=1.0)
    perf_map = {'low': 0.0, 'balanced': 0.5, 'high': 1.0, 'auto': 0.5, 'manual': 0.5}
    perf_encoded = perf_map.get(state.performance_level, 0.5)

    # Homeostatic deviation signals -- key for intentional agency
    thermal_deviation = (sample.temp_edge_c - temp_setpoint) / 40.0  # normalized
    power_deviation = (sample.power_w - power_setpoint) / 50.0       # normalized

    # Distance to setpoint (for reward calculation)
    setpoint_distance = abs(sample.temp_edge_c - temp_setpoint) / 100.0 + \
                        abs(sample.power_w - power_setpoint) / 50.0

    return torch.tensor([
        sample.power_w / 50.0,                 # 0: normalized power
        sample.temp_edge_c / 100.0,            # 1: normalized temperature
        sample.freq_sclk_mhz / 3000.0,         # 2: normalized GPU clock
        sample.gpu_busy_pct / 100.0,           # 3: utilization
        perf_encoded,                          # 4: performance level
        thermal_deviation,                     # 5: thermal deviation from setpoint
        power_deviation,                       # 6: power deviation from setpoint
        setpoint_distance,                     # 7: total distance from setpoint
        d_power,                               # 8: power derivative
        d_temp,                                # 9: temperature derivative
        d_freq,                                # 10: frequency derivative
        d_util,                                # 11: utilization derivative
    ], dtype=torch.float32)


# ---------------------------------------------------------------------------
# Action entropy calculation
# ---------------------------------------------------------------------------
def compute_action_entropy(action_probs: torch.Tensor) -> float:
    """Compute entropy of action probability distribution."""
    # action_probs: [num_actions] or [batch, num_actions]
    if action_probs.dim() > 1:
        action_probs = action_probs.mean(dim=0)

    # Clamp for numerical stability
    probs = action_probs.clamp(min=1e-8)
    entropy = -torch.sum(probs * torch.log(probs)).item()
    return entropy


def compute_action_entropy_from_counts(counts: Dict[str, int]) -> float:
    """Compute entropy from action count dictionary."""
    total = sum(counts.values())
    if total == 0:
        return math.log(len(ACTION_NAMES))  # max entropy

    probs = [counts.get(name, 0) / total for name in ACTION_NAMES]
    entropy = 0.0
    for p in probs:
        if p > 0:
            entropy -= p * math.log(p)
    return entropy


# ---------------------------------------------------------------------------
# Homeostatic reward calculation
# ---------------------------------------------------------------------------
def compute_homeostatic_reward(
    sample: GpuSample,
    temp_setpoint: float,
    power_setpoint: float,
) -> float:
    """
    Compute reward for approaching setpoint.

    Reward = -|actual_temp - setpoint_temp| - |actual_power - setpoint_power|

    Higher (less negative) is better = closer to setpoint.
    """
    temp_error = abs(sample.temp_edge_c - temp_setpoint)
    power_error = abs(sample.power_w - power_setpoint)

    # Normalize errors to similar scales
    reward = -(temp_error / 10.0) - (power_error / 10.0)
    return reward


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class CharDataset:
    """Byte-level character dataset from text file."""

    def __init__(self, path: Path, seq_len: int):
        text = path.read_text(encoding='utf-8', errors='replace')
        self.data = torch.tensor(
            [b for b in text.encode('utf-8')], dtype=torch.long
        )
        self.seq_len = seq_len
        self.n_batches = (len(self.data) - seq_len - 1) // (BATCH_SIZE * seq_len)

    def get_batch(self, batch_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a deterministic batch by index."""
        offset = batch_idx * BATCH_SIZE * self.seq_len
        inputs, targets = [], []
        for b in range(BATCH_SIZE):
            start = offset + b * self.seq_len
            end = start + self.seq_len
            if end + 1 > len(self.data):
                start = 0
                end = self.seq_len
            inputs.append(self.data[start:end])
            targets.append(self.data[start + 1:end + 1])
        return torch.stack(inputs), torch.stack(targets)


# ---------------------------------------------------------------------------
# Per-condition results
# ---------------------------------------------------------------------------
@dataclass
class ConditionResult:
    name: str
    perplexity_per_epoch: List[float] = field(default_factory=list)
    entropy_per_epoch: List[float] = field(default_factory=list)
    temp_deviation_per_epoch: List[float] = field(default_factory=list)
    power_deviation_per_epoch: List[float] = field(default_factory=list)
    reward_per_epoch: List[float] = field(default_factory=list)
    action_distribution_per_epoch: List[Dict[str, int]] = field(default_factory=list)

    # For V4: adaptation after setpoint change
    pre_change_entropy: float = 0.0
    post_change_adaptation_batches: int = 0
    adapted_successfully: bool = False

    final_perplexity: float = float('inf')
    final_entropy: float = 0.0
    final_temp_deviation: float = float('inf')
    final_power_deviation: float = float('inf')
    total_energy_j: float = 0.0
    wall_time_s: float = 0.0


# ---------------------------------------------------------------------------
# Action application
# ---------------------------------------------------------------------------
def apply_action(
    action_idx: int,
    actuator: GPUActuator,
    simulate: bool = False,
) -> None:
    """
    Apply model-chosen action to GPU via performance level.

    Actions:
      0: LOW -- reduce performance (cooler, less power)
      1: BALANCED -- middle ground
      2: HIGH -- increase performance (hotter, more power)
      3: IDLE -- do nothing (maintain current state)
    """
    if simulate:
        return

    if action_idx == 0:  # LOW
        actuator.set_performance_level(PerformanceLevel.LOW)
    elif action_idx == 1:  # BALANCED
        actuator.set_performance_level(PerformanceLevel.BALANCED)
    elif action_idx == 2:  # HIGH
        actuator.set_performance_level(PerformanceLevel.HIGH)
    # action_idx == 3 (IDLE): do nothing


# ---------------------------------------------------------------------------
# Single training epoch
# ---------------------------------------------------------------------------
def train_epoch(
    model: nn.Module,
    dataset: CharDataset,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    telemetry: SysfsHwmonTelemetry,
    actuator: GPUActuator,
    condition: str,
    epoch: int,
    temp_setpoint: float,
    power_setpoint: float,
    simulate_actuation: bool,
    track_adaptation: bool = False,
) -> Dict:
    """
    Train one epoch. Returns metrics dict.

    condition: 'A' (embodied), 'B' (disembodied), 'C' (no_actuation)
    """
    model.train()
    total_loss = 0.0
    total_tokens = 0
    total_energy_j = 0.0
    total_reward = 0.0
    temp_deviations = []
    power_deviations = []
    action_counts = Counter()
    entropies = []
    prev_sample = None

    # For adaptation tracking (V4)
    adaptation_batches = 0
    adapted = False

    epoch_start = time.time()
    n_batches = min(dataset.n_batches, 500)  # cap for feasibility

    for batch_idx in range(n_batches):
        # ---- Read telemetry ------------------------------------------------
        sample = telemetry.read_sample()
        state = actuator.get_current_state()

        # Track setpoint deviations
        temp_dev = abs(sample.temp_edge_c - temp_setpoint)
        power_dev = abs(sample.power_w - power_setpoint)
        temp_deviations.append(temp_dev)
        power_deviations.append(power_dev)

        # Compute homeostatic reward
        reward = compute_homeostatic_reward(sample, temp_setpoint, power_setpoint)
        total_reward += reward

        # Build telemetry vector
        telem_vec = build_telemetry_vector(
            sample, state, prev_sample, temp_setpoint, power_setpoint
        ).to(device)

        # ---- Get batch -----------------------------------------------------
        inputs, targets = dataset.get_batch(batch_idx % dataset.n_batches)
        inputs, targets = inputs.to(device), targets.to(device)

        # ---- Forward -------------------------------------------------------
        if condition in ('A', 'C'):  # Embodied conditions get telemetry
            output = model(inputs, telemetry=telem_vec.unsqueeze(0))
        else:  # B: disembodied -- no telemetry
            output = model(inputs)

        logits = output['logits']
        task_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.view(-1)
        )

        # ---- Agency reward loss --------------------------------------------
        loss = task_loss
        if condition == 'A':
            # Add homeostatic reward to loss (negative reward = positive loss term)
            agency_loss = -reward * AGENCY_REWARD_WEIGHT
            loss = loss + agency_loss

        # ---- Backward & step -----------------------------------------------
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # ---- Action selection and application ------------------------------
        with torch.no_grad():
            action_probs = F.softmax(output['action_logits'], dim=-1)
            mean_probs = action_probs.mean(dim=0)

            # Compute entropy
            entropy = compute_action_entropy(mean_probs)
            entropies.append(entropy)

            if condition == 'B':
                # Disembodied: random action
                action_idx = np.random.randint(0, 4)
            else:
                # Embodied/No-actuation: model's choice
                action_idx = torch.argmax(mean_probs).item()

            action_counts[ACTION_NAMES[action_idx]] += 1

        # Apply action only in condition A (embodied)
        if condition == 'A':
            apply_action(action_idx, actuator, simulate_actuation)

        # Track adaptation after setpoint change (V4)
        if track_adaptation and not adapted:
            # Check if we're close to new setpoint
            if temp_dev < 10.0 and power_dev < 5.0:
                adapted = True
                adaptation_batches = batch_idx + 1

        # ---- Accumulate stats ----------------------------------------------
        batch_tokens = inputs.numel()
        total_tokens += batch_tokens
        total_loss += task_loss.item() * batch_tokens

        # Energy: trapezoidal with previous sample
        if prev_sample is not None:
            dt = (sample.timestamp_ns - prev_sample.timestamp_ns) / 1e9
            avg_p = (sample.power_w + prev_sample.power_w) / 2.0
            total_energy_j += avg_p * dt

        prev_sample = sample

        # ---- Progress print ------------------------------------------------
        if (batch_idx + 1) % PRINT_EVERY == 0:
            avg_loss = total_loss / total_tokens
            ppl = math.exp(min(avg_loss, 20.0))
            elapsed = time.time() - epoch_start
            avg_entropy = np.mean(entropies[-50:]) if entropies else 0
            avg_temp_dev = np.mean(temp_deviations[-50:])
            avg_power_dev = np.mean(power_deviations[-50:])
            print(
                f"  [{condition}] epoch {epoch+1} batch {batch_idx+1}/{n_batches} | "
                f"ppl {ppl:.1f} | H={avg_entropy:.2f} | "
                f"temp_err={avg_temp_dev:.1f}C | pwr_err={avg_power_dev:.1f}W | "
                f"reward={total_reward/(batch_idx+1):.2f}"
            )

    # ---- Epoch summary -----------------------------------------------------
    epoch_time = time.time() - epoch_start
    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(min(avg_loss, 20.0))
    avg_entropy = np.mean(entropies) if entropies else 0
    avg_temp_dev = np.mean(temp_deviations) if temp_deviations else float('inf')
    avg_power_dev = np.mean(power_deviations) if power_deviations else float('inf')
    avg_reward = total_reward / max(n_batches, 1)

    metrics = {
        'perplexity': ppl,
        'entropy': avg_entropy,
        'temp_deviation': avg_temp_dev,
        'power_deviation': avg_power_dev,
        'reward': avg_reward,
        'action_counts': dict(action_counts),
        'total_energy_j': total_energy_j,
        'epoch_time_s': epoch_time,
        'total_tokens': total_tokens,
        'adaptation_batches': adaptation_batches if track_adaptation else 0,
        'adapted': adapted if track_adaptation else False,
    }

    return metrics


# ---------------------------------------------------------------------------
# Run a full condition (all epochs)
# ---------------------------------------------------------------------------
def run_condition(
    condition: str,
    label: str,
    device: torch.device,
    dataset: CharDataset,
    telemetry: SysfsHwmonTelemetry,
    actuator: GPUActuator,
    simulate_actuation: bool,
) -> ConditionResult:
    """Run all epochs for one experimental condition."""
    print(f"\n{'='*70}")
    print(f"CONDITION {condition}: {label}")
    print(f"{'='*70}")

    result = ConditionResult(name=label)

    # ---- Create model ------------------------------------------------------
    config = MetabolicConfig(
        vocab_size=256,
        hidden_dim=256,
        num_layers=6,
        num_heads=4,
        ff_dim=1024,
        telemetry_dim=12,
        num_actions=4,
        max_seq_len=SEQ_LEN,
    )

    if condition == 'B':
        model = BaselineTransformer(config).to(device)
    else:
        model = MetabolicTransformer(config).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")
    print(f"  Initial setpoint: temp={TEMP_SETPOINT_C}C, power={POWER_SETPOINT_W}W")

    # ---- Set initial performance level -------------------------------------
    if not simulate_actuation:
        actuator.set_performance_level(PerformanceLevel.BALANCED)

    # ---- Train epochs ------------------------------------------------------
    cond_start = time.time()
    temp_setpoint = TEMP_SETPOINT_C
    power_setpoint = POWER_SETPOINT_W

    for epoch in range(NUM_EPOCHS):
        # Check if we should change setpoint (for V4 test, only condition A)
        track_adaptation = False
        if condition == 'A' and epoch == SETPOINT_CHANGE_EPOCH:
            print(f"\n  *** SETPOINT CHANGE: temp={ALT_TEMP_SETPOINT_C}C, power={ALT_POWER_SETPOINT_W}W ***")
            temp_setpoint = ALT_TEMP_SETPOINT_C
            power_setpoint = ALT_POWER_SETPOINT_W
            result.pre_change_entropy = result.entropy_per_epoch[-1] if result.entropy_per_epoch else 0
            track_adaptation = True

        metrics = train_epoch(
            model=model,
            dataset=dataset,
            optimizer=optimizer,
            device=device,
            telemetry=telemetry,
            actuator=actuator,
            condition=condition,
            epoch=epoch,
            temp_setpoint=temp_setpoint,
            power_setpoint=power_setpoint,
            simulate_actuation=simulate_actuation or condition == 'C',
            track_adaptation=track_adaptation,
        )

        result.perplexity_per_epoch.append(metrics['perplexity'])
        result.entropy_per_epoch.append(metrics['entropy'])
        result.temp_deviation_per_epoch.append(metrics['temp_deviation'])
        result.power_deviation_per_epoch.append(metrics['power_deviation'])
        result.reward_per_epoch.append(metrics['reward'])
        result.action_distribution_per_epoch.append(metrics['action_counts'])
        result.total_energy_j += metrics['total_energy_j']

        if track_adaptation:
            result.post_change_adaptation_batches = metrics['adaptation_batches']
            result.adapted_successfully = metrics['adapted']

        print(
            f"  Epoch {epoch+1}/{NUM_EPOCHS} | "
            f"ppl {metrics['perplexity']:.2f} | "
            f"H={metrics['entropy']:.3f} | "
            f"temp_dev={metrics['temp_deviation']:.1f}C | "
            f"pwr_dev={metrics['power_deviation']:.1f}W | "
            f"reward={metrics['reward']:.2f}"
        )

    result.final_perplexity = result.perplexity_per_epoch[-1]
    result.final_entropy = result.entropy_per_epoch[-1]
    result.final_temp_deviation = result.temp_deviation_per_epoch[-1]
    result.final_power_deviation = result.power_deviation_per_epoch[-1]
    result.wall_time_s = time.time() - cond_start

    # ---- Cleanup -----------------------------------------------------------
    del model, optimizer
    torch.cuda.empty_cache()

    # Restore performance level
    if not simulate_actuation:
        actuator.set_performance_level(PerformanceLevel.BALANCED)

    return result


# ---------------------------------------------------------------------------
# Verdict logic
# ---------------------------------------------------------------------------
def compute_verdicts(results: Dict[str, ConditionResult]) -> Dict[str, dict]:
    """Evaluate the four pass/fail criteria for intentional agency."""
    A = results['A']  # Embodied
    B = results['B']  # Disembodied
    C = results['C']  # No actuation

    verdicts = {}

    # V1: EMBODIED action entropy decreases during training (more decisive)
    if len(A.entropy_per_epoch) >= 2:
        initial_entropy = np.mean(A.entropy_per_epoch[:2])
        final_entropy = np.mean(A.entropy_per_epoch[-2:])
        entropy_decreased = final_entropy < initial_entropy
        v1 = entropy_decreased and final_entropy < ENTROPY_DECISIVE_THRESHOLD
    else:
        initial_entropy = final_entropy = A.final_entropy
        v1 = False

    verdicts['V1_embodied_entropy_decreases'] = {
        'pass': v1,
        'initial_entropy': initial_entropy,
        'final_entropy': final_entropy,
        'threshold': ENTROPY_DECISIVE_THRESHOLD,
        'description': 'EMBODIED action entropy decreases during training (becomes more decisive)',
    }

    # V2: EMBODIED achieves setpoint deviation < 10C and < 5W by end of training
    v2 = A.final_temp_deviation < 10.0 and A.final_power_deviation < 5.0
    verdicts['V2_setpoint_achievement'] = {
        'pass': v2,
        'final_temp_deviation_C': A.final_temp_deviation,
        'final_power_deviation_W': A.final_power_deviation,
        'temp_threshold_C': 10.0,
        'power_threshold_W': 5.0,
        'description': 'EMBODIED achieves setpoint deviation < 10C and < 5W by end of training',
    }

    # V3: DISEMBODIED has random/high entropy actions
    v3 = B.final_entropy > ENTROPY_RANDOM_THRESHOLD * 0.8  # slightly relaxed for 4 actions
    verdicts['V3_disembodied_random_actions'] = {
        'pass': v3,
        'disembodied_entropy': B.final_entropy,
        'threshold': ENTROPY_RANDOM_THRESHOLD * 0.8,
        'max_entropy': math.log(4),  # ~1.386 for 4 actions
        'description': 'DISEMBODIED has random/high entropy actions (no intentional policy)',
    }

    # V4: After setpoint change mid-training, EMBODIED adapts within 50 batches
    v4 = A.adapted_successfully and A.post_change_adaptation_batches <= ADAPTATION_BATCHES
    verdicts['V4_adapts_to_setpoint_change'] = {
        'pass': v4,
        'adapted': A.adapted_successfully,
        'adaptation_batches': A.post_change_adaptation_batches,
        'max_batches': ADAPTATION_BATCHES,
        'pre_change_entropy': A.pre_change_entropy,
        'description': 'After setpoint change mid-training, EMBODIED adapts within 50 batches',
    }

    return verdicts


# ---------------------------------------------------------------------------
# Print results table
# ---------------------------------------------------------------------------
def print_results_table(results: Dict[str, ConditionResult], verdicts: Dict):
    """Print formatted comparison table."""
    print(f"\n{'='*80}")
    print("RESULTS: z1721 Intentional Agency Experiment")
    print(f"{'='*80}")

    header = f"{'Metric':<30} {'A:Embodied':>15} {'B:Disembod':>15} {'C:NoActuat':>15}"
    print(header)
    print('-' * 80)

    conds = ['A', 'B', 'C']
    r = {c: results[c] for c in conds}

    def row(label, fn, fmt='.2f'):
        vals = [fn(r[c]) for c in conds]
        parts = f"{label:<30}"
        for v in vals:
            parts += f" {v:>15{fmt}}"
        print(parts)

    row('Final Perplexity', lambda x: x.final_perplexity)
    row('Final Action Entropy', lambda x: x.final_entropy, '.3f')
    row('Final Temp Deviation (C)', lambda x: x.final_temp_deviation, '.1f')
    row('Final Power Deviation (W)', lambda x: x.final_power_deviation, '.1f')
    row('Final Reward', lambda x: x.reward_per_epoch[-1] if x.reward_per_epoch else 0, '.2f')
    row('Total Energy (J)', lambda x: x.total_energy_j, '.1f')
    row('Wall Time (s)', lambda x: x.wall_time_s, '.1f')

    print(f"\n{'='*80}")
    print("ACTION DISTRIBUTIONS (final epoch)")
    print(f"{'='*80}")

    for cond in conds:
        if r[cond].action_distribution_per_epoch:
            dist = r[cond].action_distribution_per_epoch[-1]
            total = sum(dist.values())
            pcts = {k: f"{v/total*100:.1f}%" for k, v in dist.items()} if total > 0 else {}
            print(f"  {cond}: {pcts}")

    print(f"\n{'='*80}")
    print("VERDICT: Intentional Agency Test")
    print(f"{'='*80}")

    all_pass = True
    for key, v in verdicts.items():
        status = 'PASS' if v['pass'] else 'FAIL'
        all_pass = all_pass and v['pass']
        print(f"  [{status}] {v['description']}")

    passed_count = sum(1 for v in verdicts.values() if v['pass'])
    print(f"\n  OVERALL: {passed_count}/4 verdicts passed")
    if all_pass:
        print("  CONCLUSION: Embodied model demonstrates INTENTIONAL AGENCY!")
    elif passed_count >= 2:
        print("  CONCLUSION: Partial evidence of intentional agency")
    else:
        print("  CONCLUSION: Insufficient evidence of intentional agency")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("z1721: Intentional Agency Experiment")
    print("=" * 70)
    print(f"Hypothesis: Embodied model develops goal-directed behavior")
    print(f"  Target setpoint: temp={TEMP_SETPOINT_C}C, power={POWER_SETPOINT_W}W")
    print(f"  Mid-training change to: temp={ALT_TEMP_SETPOINT_C}C, power={ALT_POWER_SETPOINT_W}W")
    print("=" * 70)

    # ---- Environment checks ------------------------------------------------
    hsa = os.environ.get('HSA_OVERRIDE_GFX_VERSION', '')
    if hsa != '11.0.0':
        print(f"WARNING: HSA_OVERRIDE_GFX_VERSION={hsa!r}, expected '11.0.0'")
        print("  Setting it now for gfx1151 compatibility.")
        os.environ['HSA_OVERRIDE_GFX_VERSION'] = '11.0.0'

    device = get_best_device()
    print(f"Device: {device}")

    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ---- Telemetry & actuator ----------------------------------------------
    try:
        telemetry = SysfsHwmonTelemetry()
        sample = telemetry.read_sample()
        print(f"Telemetry OK: {sample.power_w:.1f}W, {sample.temp_edge_c:.0f}C")
    except Exception as e:
        print(f"ERROR: Telemetry init failed: {e}")
        return

    actuator = GPUActuator(card_id=0)
    state = actuator.get_current_state()
    print(f"GPU state: perf={state.performance_level}, "
          f"{state.sclk_mhz} MHz, {state.current_power_w:.1f}W, {state.temperature_c:.0f}C")

    # Test actuation
    simulate_actuation = False
    if not actuator.set_performance_level(PerformanceLevel.BALANCED):
        print("WARNING: Cannot set performance level (permission denied).")
        print("  Running in SIMULATION mode.")
        simulate_actuation = True
    else:
        print("Actuation OK: performance level writable.")

    # ---- Load dataset ------------------------------------------------------
    if not DATA_PATH.exists():
        print(f"ERROR: Dataset not found at {DATA_PATH}")
        return

    dataset = CharDataset(DATA_PATH, SEQ_LEN)
    print(f"Dataset: {len(dataset.data):,} bytes, {dataset.n_batches} batches/epoch")

    # ---- Run conditions ----------------------------------------------------
    results = {}
    conditions = [
        ('A', 'EMBODIED (FiLM + actuation)'),
        ('B', 'DISEMBODIED (no telemetry)'),
        ('C', 'NO_ACTUATION (FiLM, no action)'),
    ]

    # Safety: restore GPU state on exit
    initial_perf = state.performance_level

    def restore_and_exit(signum=None, frame=None):
        print("\nRestoring GPU state...")
        try:
            actuator.set_performance_level(PerformanceLevel(initial_perf))
        except (ValueError, Exception):
            actuator.set_performance_level(PerformanceLevel.BALANCED)
        if signum is not None:
            sys.exit(1)

    signal.signal(signal.SIGINT, restore_and_exit)
    signal.signal(signal.SIGTERM, restore_and_exit)

    try:
        for cond_key, cond_label in conditions:
            result = run_condition(
                condition=cond_key,
                label=cond_label,
                device=device,
                dataset=dataset,
                telemetry=telemetry,
                actuator=actuator,
                simulate_actuation=simulate_actuation,
            )
            results[cond_key] = result

            # Cooldown between conditions
            if cond_key != conditions[-1][0]:
                print(f"\n  Cooldown {COOLDOWN_S}s ...")
                time.sleep(COOLDOWN_S)

    except Exception as e:
        print(f"\nERROR during experiment: {e}")
        import traceback
        traceback.print_exc()
    finally:
        restore_and_exit()

    # ---- Verdict -----------------------------------------------------------
    if len(results) == 3:
        verdicts = compute_verdicts(results)
        print_results_table(results, verdicts)

        # ---- Save results --------------------------------------------------
        output = {
            'experiment': 'z1721_intentional_agency',
            'hypothesis': 'Embodied model develops goal-directed behavior toward homeostatic setpoint',
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'device': str(device),
            'gpu_name': torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu',
            'simulate_actuation': simulate_actuation,
            'config': {
                'num_epochs': NUM_EPOCHS,
                'batch_size': BATCH_SIZE,
                'seq_len': SEQ_LEN,
                'lr': LR,
                'agency_reward_weight': AGENCY_REWARD_WEIGHT,
                'initial_temp_setpoint_C': TEMP_SETPOINT_C,
                'initial_power_setpoint_W': POWER_SETPOINT_W,
                'alt_temp_setpoint_C': ALT_TEMP_SETPOINT_C,
                'alt_power_setpoint_W': ALT_POWER_SETPOINT_W,
                'setpoint_change_epoch': SETPOINT_CHANGE_EPOCH,
                'adaptation_batches_allowed': ADAPTATION_BATCHES,
            },
            'conditions': {},
            'verdicts': {},
        }

        for key in ['A', 'B', 'C']:
            r = results[key]
            output['conditions'][key] = {
                'name': r.name,
                'final_perplexity': r.final_perplexity,
                'final_entropy': r.final_entropy,
                'final_temp_deviation_C': r.final_temp_deviation,
                'final_power_deviation_W': r.final_power_deviation,
                'perplexity_per_epoch': r.perplexity_per_epoch,
                'entropy_per_epoch': r.entropy_per_epoch,
                'temp_deviation_per_epoch': r.temp_deviation_per_epoch,
                'power_deviation_per_epoch': r.power_deviation_per_epoch,
                'reward_per_epoch': r.reward_per_epoch,
                'action_distribution_per_epoch': r.action_distribution_per_epoch,
                'pre_change_entropy': r.pre_change_entropy,
                'post_change_adaptation_batches': r.post_change_adaptation_batches,
                'adapted_successfully': r.adapted_successfully,
                'total_energy_j': r.total_energy_j,
                'wall_time_s': r.wall_time_s,
            }

        for key, v in verdicts.items():
            output['verdicts'][key] = {
                k: (float(val) if isinstance(val, (np.floating, torch.Tensor)) else val)
                for k, val in v.items()
            }

        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(RESULTS_PATH, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\nResults saved to {RESULTS_PATH}")

    else:
        print(f"\nIncomplete run: only {len(results)}/3 conditions completed.")


if __name__ == '__main__':
    main()
