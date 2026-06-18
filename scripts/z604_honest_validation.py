#!/usr/bin/env python3
"""
Z604: HONEST Validation with Fixed Power Measurement

Key fixes from z603:
1. Uses hwmon power (not broken gpu_metrics)
2. FIXED TASK: Same compute work per iteration, measure J/work
3. Trial-level statistics (not sample-level)
"""

import os
import sys
import time
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List
import statistics

# AMD GPU setup
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class Config:
    n_trials: int = 3
    n_steps_per_trial: int = 30
    # FIXED TASK: every controller does same work
    fixed_batch: int = 64
    fixed_size: int = 1024
    fixed_iters: int = 10
    cooldown_s: float = 5.0


# ============================================================================
# Correct Power Reading
# ============================================================================

def read_hwmon_power() -> float:
    """Read power from hwmon (CORRECT source)."""
    hwmon_base = Path("/sys/class/drm/card1/device/hwmon")
    for d in hwmon_base.iterdir():
        if d.is_dir():
            power_file = d / "power1_average"
            if power_file.exists():
                return int(power_file.read_text().strip()) / 1_000_000
    return 0.0


def set_perf_level(level: str):
    """Set GPU performance level."""
    path = Path("/sys/class/drm/card1/device/power_dpm_force_performance_level")
    path.write_text(level)


# ============================================================================
# Controllers (same fixed task, different strategies)
# ============================================================================

def run_fixed_task(device) -> tuple:
    """Run FIXED compute task, return (time, energy)."""
    import torch

    config = Config()

    # Measure power before
    p0 = read_hwmon_power()

    # FIXED TASK
    a = torch.randn(config.fixed_batch, config.fixed_size, device=device)
    b = torch.randn(config.fixed_size, config.fixed_size, device=device)

    t0 = time.perf_counter()
    for _ in range(config.fixed_iters):
        c = torch.matmul(a, b)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    # Measure power after (average with before for crude energy estimate)
    p1 = read_hwmon_power()
    avg_power = (p0 + p1) / 2
    energy_j = avg_power * elapsed

    del a, b, c

    return elapsed, energy_j, avg_power


class ReactiveController:
    """Reactive: adjust perf_level based on observed power."""

    def __init__(self):
        self.current_level = 'auto'

    def step(self, device) -> Dict:
        # Read current state
        power = read_hwmon_power()

        # Reactive decision
        if power > 50:
            self.current_level = 'low'
        elif power < 30:
            self.current_level = 'high'
        else:
            self.current_level = 'auto'

        set_perf_level(self.current_level)
        time.sleep(0.1)

        # Run FIXED task
        elapsed, energy, power = run_fixed_task(device)

        return {
            'elapsed': elapsed,
            'energy_j': energy,
            'power_w': power,
            'level': self.current_level,
        }


class FixedHighController:
    """Always high performance (baseline for max throughput)."""

    def step(self, device) -> Dict:
        set_perf_level('high')
        time.sleep(0.1)
        elapsed, energy, power = run_fixed_task(device)
        return {'elapsed': elapsed, 'energy_j': energy, 'power_w': power, 'level': 'high'}


class FixedLowController:
    """Always low performance (baseline for min energy)."""

    def step(self, device) -> Dict:
        set_perf_level('low')
        time.sleep(0.1)
        elapsed, energy, power = run_fixed_task(device)
        return {'elapsed': elapsed, 'energy_j': energy, 'power_w': power, 'level': 'low'}


class SimpleEmbodiedController:
    """Embodied: predict and choose based on efficiency target."""

    def __init__(self):
        # Simple learned model: track which level gives best J/work
        self.level_stats = {'low': [], 'auto': [], 'high': []}
        self.current_level = 'auto'

    def step(self, device) -> Dict:
        # Choose level with best observed efficiency (or explore)
        if all(len(v) >= 2 for v in self.level_stats.values()):
            # Exploit: pick level with lowest energy per task
            best_level = min(
                self.level_stats.keys(),
                key=lambda k: statistics.mean(self.level_stats[k][-5:]) if self.level_stats[k] else float('inf')
            )
            self.current_level = best_level
        else:
            # Explore: cycle through levels
            levels = ['low', 'auto', 'high']
            min_samples = min(len(self.level_stats[l]) for l in levels)
            self.current_level = levels[min_samples % 3]

        set_perf_level(self.current_level)
        time.sleep(0.1)

        elapsed, energy, power = run_fixed_task(device)

        # Learn: track energy for this level
        self.level_stats[self.current_level].append(energy)

        return {
            'elapsed': elapsed,
            'energy_j': energy,
            'power_w': power,
            'level': self.current_level,
        }


# ============================================================================
# Main
# ============================================================================

def run_trial(controller, device, n_steps: int) -> List[Dict]:
    """Run one trial."""
    results = []
    for _ in range(n_steps):
        r = controller.step(device)
        results.append(r)
    return results


def main():
    print("=" * 70)
    print("Z604: HONEST VALIDATION (Fixed Power Measurement)")
    print("=" * 70)

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    config = Config()
    print(f"Trials: {config.n_trials}, Steps/trial: {config.n_steps_per_trial}")
    print(f"Fixed task: {config.fixed_batch}x{config.fixed_size} @ {config.fixed_iters} iters")

    # Verify power reading
    print(f"\nPower check: {read_hwmon_power():.1f}W (should be 20-40W at idle)")

    controllers = {
        'fixed_high': FixedHighController,
        'fixed_low': FixedLowController,
        'reactive': ReactiveController,
        'embodied': SimpleEmbodiedController,
    }

    results = {name: [] for name in controllers}

    for trial in range(config.n_trials):
        print(f"\n--- Trial {trial+1}/{config.n_trials} ---")

        for name, ControllerClass in controllers.items():
            # Cooldown
            set_perf_level('low')
            time.sleep(config.cooldown_s)

            # Run trial
            ctrl = ControllerClass()
            trial_results = run_trial(ctrl, device, config.n_steps_per_trial)

            # Aggregate trial
            energies = [r['energy_j'] for r in trial_results]
            times = [r['elapsed'] for r in trial_results]
            powers = [r['power_w'] for r in trial_results]

            trial_summary = {
                'total_energy_j': sum(energies),
                'total_time_s': sum(times),
                'mean_power_w': statistics.mean(powers),
                'j_per_task': statistics.mean(energies),
            }
            results[name].append(trial_summary)

            print(f"  {name:12s}: {trial_summary['j_per_task']:.3f} J/task, "
                  f"{trial_summary['mean_power_w']:.1f}W avg")

    # Reset
    set_perf_level('auto')

    # ========================================================================
    # TRIAL-LEVEL STATISTICS (proper)
    # ========================================================================
    print("\n" + "=" * 70)
    print("RESULTS (Trial-level statistics)")
    print("=" * 70)

    for name in controllers:
        j_per_task = [t['j_per_task'] for t in results[name]]
        mean = statistics.mean(j_per_task)
        std = statistics.stdev(j_per_task) if len(j_per_task) > 1 else 0

        print(f"\n{name.upper()}:")
        print(f"  J/task: {mean:.4f} ± {std:.4f}")
        print(f"  Per-trial: {j_per_task}")

    # ========================================================================
    # HONEST COMPARISON
    # ========================================================================
    print("\n--- Comparison (lower J/task = better) ---")

    baseline_mean = statistics.mean([t['j_per_task'] for t in results['fixed_high']])

    for name in controllers:
        mean = statistics.mean([t['j_per_task'] for t in results[name]])
        pct = (baseline_mean - mean) / baseline_mean * 100
        print(f"  {name:12s}: {mean:.4f} J/task  ({pct:+.1f}% vs fixed_high)")

    # ========================================================================
    # Save
    # ========================================================================
    output = {
        'config': {
            'n_trials': config.n_trials,
            'n_steps': config.n_steps_per_trial,
            'fixed_task': f"{config.fixed_batch}x{config.fixed_size}x{config.fixed_iters}",
        },
        'results': {
            name: {
                'trials': results[name],
                'mean_j_per_task': statistics.mean([t['j_per_task'] for t in results[name]]),
            }
            for name in controllers
        },
        'power_source': 'hwmon (CORRECT)',
    }

    output_path = Path("results/z604_honest_validation.json")
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
