#!/usr/bin/env python3
"""
z1945: GPU Rapid Signal Drift Task

INSIGHT FROM z1940:
- When FPGA was disconnected (random fallback): 42.2% embodied advantage
- When FPGA was connected (static values): 1.5% no advantage
- The UNPREDICTABLE SIGNAL was the key!

THIS EXPERIMENT:
Focus ONLY on GPU's rapidly-varying signals:
1. Power consumption (changes with each matmul)
2. GPU utilization (very fast changes)
3. Temperature derivative (slower but still important)

Target = f(power_deriv, util_deriv, temp_deriv) + noise
No momentum, no history prediction possible.

The target is a DIRECT function of hardware derivatives.
History cannot predict the NEXT derivative value.
"""

import os
import sys
import time
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, List
from scipy import stats as scipy_stats

sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '11.0.0')

import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class RapidGPUSensor:
    """GPU sensor optimized for tracking rapid changes"""

    def __init__(self):
        self.card = '/sys/class/drm/card1/device'

        # History for derivatives
        self.last_temp = None
        self.last_power = None
        self.last_util = None
        self.last_time = None

        # Track signal variability
        self.signal_history = {'temp_d': [], 'power_d': [], 'util_d': []}

    def _hwmon(self, p, d=0):
        try:
            for h in os.listdir(f'{self.card}/hwmon'):
                f = f'{self.card}/hwmon/{h}/{p}'
                if os.path.exists(f):
                    with open(f) as fp:
                        return float(fp.read().strip())
        except:
            pass
        return d

    def _read(self, f, d=0):
        try:
            with open(f'{self.card}/{f}') as fp:
                return float(fp.read().strip())
        except:
            return d

    def read(self) -> Dict:
        """Get hardware state with ALL derivatives"""
        now = time.time()

        # Raw values
        temp_raw = self._hwmon('temp1_input', 50000)
        temp = temp_raw / 1000  # Celsius
        power = self._hwmon('power1_average', 50e6) / 1e6  # Watts
        util = self._read('gpu_busy_percent', 50)  # Percent

        # Compute derivatives
        if self.last_time is not None:
            dt = now - self.last_time
            if dt > 0.001:
                temp_d = (temp - self.last_temp) / dt if self.last_temp else 0
                power_d = (power - self.last_power) / dt if self.last_power else 0
                util_d = (util - self.last_util) / dt if self.last_util else 0
            else:
                temp_d = power_d = util_d = 0
        else:
            temp_d = power_d = util_d = 0

        # Update history
        self.last_temp = temp
        self.last_power = power
        self.last_util = util
        self.last_time = now

        # Track for statistics
        self.signal_history['temp_d'].append(temp_d)
        self.signal_history['power_d'].append(power_d)
        self.signal_history['util_d'].append(util_d)

        return {
            'temp': temp,
            'temp_norm': temp / 100,
            'power': power,
            'power_norm': power / 100,
            'util': util,
            'util_norm': util / 100,
            'temp_deriv': np.clip(temp_d, -10, 10),
            'temp_deriv_norm': np.clip(temp_d / 10, -1, 1),
            'power_deriv': np.clip(power_d, -50, 50),
            'power_deriv_norm': np.clip(power_d / 50, -1, 1),
            'util_deriv': np.clip(util_d, -500, 500),
            'util_deriv_norm': np.clip(util_d / 500, -1, 1),
        }

    def get_signal_stats(self) -> Dict:
        """Get statistics on signal variability"""
        stats = {}
        for key, values in self.signal_history.items():
            if values:
                stats[f'{key}_std'] = np.std(values[-100:]) if len(values) > 1 else 0
                stats[f'{key}_range'] = max(values[-100:]) - min(values[-100:]) if values else 0
        return stats

    def reset_stats(self):
        """Reset statistics tracking"""
        self.signal_history = {'temp_d': [], 'power_d': [], 'util_d': []}


class DirectDerivativeTask:
    """
    Task where target is DIRECTLY a function of hardware derivatives.
    NO momentum, NO history dependence.

    Target(t) = w1*power_deriv + w2*util_deriv + w3*temp_deriv + noise

    This is IMPOSSIBLE to predict from target history alone because
    derivatives are unpredictable from past values.
    """

    def __init__(self, noise_scale=0.08):
        self.noise_scale = noise_scale

        # Weights for each derivative signal
        self.w_power = 0.4  # Power changes fast!
        self.w_util = 0.35  # Util changes very fast!
        self.w_temp = 0.15  # Temp changes slower

        self.y = 0.5  # Current target

    def compute_target(self, hw_state: Dict) -> Tuple[float, Dict]:
        """Compute target directly from hardware derivatives"""
        # Get derivatives (already normalized to [-1, 1])
        power_d = hw_state['power_deriv_norm']
        util_d = hw_state['util_deriv_norm']
        temp_d = hw_state['temp_deriv_norm']

        # Target is weighted sum of derivatives
        # This CANNOT be predicted from history!
        signal = (self.w_power * power_d +
                  self.w_util * util_d +
                  self.w_temp * temp_d)

        # Map to [0, 1] range with some baseline
        target = 0.5 + 0.4 * signal

        # Add small noise
        noise = np.random.normal(0, self.noise_scale)
        target = float(np.clip(target + noise, 0, 1))

        self.y = target

        return target, {
            'power_d': power_d,
            'util_d': util_d,
            'temp_d': temp_d,
            'signal': signal,
        }

    def reset(self):
        self.y = 0.5


class EmbodiedDerivModel(nn.Module):
    """Model that sees hardware state including derivatives"""

    def __init__(self, hidden_dim=64):
        super().__init__()

        # Input: temp_n, power_n, util_n, temp_d, power_d, util_d (6 dims)
        self.net = nn.Sequential(
            nn.Linear(6, 48),
            nn.ReLU(),
            nn.Linear(48, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, hw_state: torch.Tensor) -> torch.Tensor:
        return self.net(hw_state).squeeze(-1)


class BlindHistoryModel(nn.Module):
    """Model that only sees target history"""

    def __init__(self, hist_len=15, hidden_dim=64):
        super().__init__()

        # Same parameter count
        self.net = nn.Sequential(
            nn.Linear(hist_len, 48),
            nn.ReLU(),
            nn.Linear(48, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        return self.net(history).squeeze(-1)


def create_random_gpu_load():
    """Create random GPU load to vary power and utilization"""
    intensity = np.random.choice([0, 1, 2, 3, 4], p=[0.2, 0.2, 0.2, 0.2, 0.2])

    if intensity == 0:
        time.sleep(0.02)  # Idle
    elif intensity == 1:
        _ = torch.randn(500, 500, device=DEVICE) @ torch.randn(500, 500, device=DEVICE)
    elif intensity == 2:
        _ = torch.randn(1000, 1000, device=DEVICE) @ torch.randn(1000, 1000, device=DEVICE)
    elif intensity == 3:
        for _ in range(2):
            _ = torch.randn(1500, 1500, device=DEVICE) @ torch.randn(1500, 1500, device=DEVICE)
    else:
        for _ in range(3):
            _ = torch.randn(2000, 2000, device=DEVICE) @ torch.randn(2000, 2000, device=DEVICE)

    torch.cuda.synchronize()


def run_episode(embodied, blind, task, sensor, target_history,
                n_steps=60, train=True):
    """Run one episode"""

    if train:
        opt_e = torch.optim.Adam(embodied.parameters(), lr=1e-3)
        opt_b = torch.optim.Adam(blind.parameters(), lr=1e-3)

    task.reset()
    sensor.reset_stats()

    emb_errors = []
    blind_errors = []

    for step in range(n_steps):
        # Create varying GPU load
        create_random_gpu_load()

        # Read hardware (includes derivatives)
        hw = sensor.read()

        # Compute true target (directly from derivatives!)
        true_target, info = task.compute_target(hw)

        # Hardware tensor for embodied model
        hw_tensor = torch.tensor([
            hw['temp_norm'],
            hw['power_norm'],
            hw['util_norm'],
            hw['temp_deriv_norm'],
            hw['power_deriv_norm'],
            hw['util_deriv_norm'],
        ], dtype=torch.float32, device=DEVICE).unsqueeze(0)

        # History tensor for blind model
        hist_tensor = torch.tensor([target_history], dtype=torch.float32, device=DEVICE)

        target_tensor = torch.tensor([true_target], dtype=torch.float32, device=DEVICE)

        # Predictions
        emb_pred = embodied(hw_tensor)
        blind_pred = blind(hist_tensor)

        # Losses
        emb_loss = F.mse_loss(emb_pred, target_tensor)
        blind_loss = F.mse_loss(blind_pred, target_tensor)

        if train:
            opt_e.zero_grad()
            emb_loss.backward()
            opt_e.step()

            opt_b.zero_grad()
            blind_loss.backward()
            opt_b.step()

        emb_errors.append(emb_loss.item())
        blind_errors.append(blind_loss.item())

        # Update history
        target_history.append(true_target)
        target_history.pop(0)

    signal_stats = sensor.get_signal_stats()

    return {
        'emb_mse': np.mean(emb_errors),
        'blind_mse': np.mean(blind_errors),
        'signal_stats': signal_stats,
    }


def main():
    print("=" * 70)
    print("z1945: GPU Rapid Signal Drift Task")
    print("Target = f(power_deriv, util_deriv, temp_deriv)")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Timestamp: {datetime.now().isoformat()}")

    results = {
        'experiment': 'z1945_gpu_rapid_signals',
        'timestamp': datetime.now().isoformat(),
        'device': str(DEVICE),
    }

    # Initialize
    print("\n=== Initialization ===")
    sensor = RapidGPUSensor()
    task = DirectDerivativeTask(noise_scale=0.06)

    # Warm-up sensor
    for _ in range(10):
        create_random_gpu_load()
        sensor.read()

    # Models
    embodied = EmbodiedDerivModel(hidden_dim=64).to(DEVICE)
    blind = BlindHistoryModel(hist_len=15, hidden_dim=64).to(DEVICE)

    print(f"Embodied params: {sum(p.numel() for p in embodied.parameters()):,}")
    print(f"Blind params: {sum(p.numel() for p in blind.parameters()):,}")

    target_history = [0.5] * 15

    # Training
    print("\n=== Training (50 episodes) ===")
    training_log = []

    for ep in range(50):
        result = run_episode(embodied, blind, task, sensor, target_history,
                            n_steps=60, train=True)
        training_log.append(result)

        if (ep + 1) % 10 == 0:
            stats = result['signal_stats']
            print(f"  Ep {ep+1}: Emb={result['emb_mse']:.4f}, Blind={result['blind_mse']:.4f}")
            print(f"    Deriv STDs: power={stats.get('power_d_std',0):.2f}, "
                  f"util={stats.get('util_d_std',0):.2f}, "
                  f"temp={stats.get('temp_d_std',0):.2f}")

    results['training'] = training_log

    # Evaluation
    print("\n=== Evaluation (25 episodes) ===")
    embodied.eval()
    blind.eval()

    eval_emb = []
    eval_blind = []

    with torch.no_grad():
        for ep in range(25):
            result = run_episode(embodied, blind, task, sensor, target_history,
                                n_steps=80, train=False)
            eval_emb.append(result['emb_mse'])
            eval_blind.append(result['blind_mse'])

    emb_mean = np.mean(eval_emb)
    emb_std = np.std(eval_emb)
    blind_mean = np.mean(eval_blind)
    blind_std = np.std(eval_blind)

    improvement = (blind_mean - emb_mean) / blind_mean * 100 if blind_mean > 0 else 0
    t_stat, p_value = scipy_stats.ttest_ind(eval_emb, eval_blind)

    print(f"\n{'Model':<15} | {'MSE':>10} | {'Std':>10}")
    print("-" * 40)
    print(f"{'EMBODIED':<15} | {emb_mean:>10.4f} | {emb_std:>10.4f}")
    print(f"{'BLIND':<15} | {blind_mean:>10.4f} | {blind_std:>10.4f}")
    print(f"\nImprovement: {improvement:+.1f}%")
    print(f"p-value: {p_value:.2e}")

    # Get final signal statistics
    final_stats = sensor.get_signal_stats()
    print(f"\nSignal variability: power_d_std={final_stats.get('power_d_std',0):.2f}, "
          f"util_d_std={final_stats.get('util_d_std',0):.2f}")

    results['evaluation'] = {
        'embodied_mse': {'mean': emb_mean, 'std': emb_std, 'values': eval_emb},
        'blind_mse': {'mean': blind_mean, 'std': blind_std, 'values': eval_blind},
        'improvement_pct': improvement,
        't_statistic': t_stat,
        'p_value': p_value,
        'signal_stats': final_stats,
    }

    # Tests
    t1 = improvement > 30 and p_value < 0.001
    t2 = improvement > 10 and p_value < 0.05
    t3 = emb_mean < 0.1

    tests_passed = int(t1) + int(t2) + int(t3)

    results['tests'] = {
        'T1_strong': f'{t1} (>30%, p<0.001)',
        'T2_moderate': f'{t2} (>10%, p<0.05)',
        'T3_low_error': f'{t3} (MSE<0.1)',
    }
    results['tests_passed'] = tests_passed

    # Verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    if t1:
        verdict = "STRONG EMBODIMENT ADVANTAGE"
        print(f"\n✅ {verdict}")
        print(f"   Embodied beats blind by {improvement:.1f}% (p={p_value:.2e})")
    elif t2:
        verdict = "MODERATE EMBODIMENT ADVANTAGE"
        print(f"\n⚠️ {verdict}")
        print(f"   Embodied beats blind by {improvement:.1f}% (p={p_value:.4f})")
    else:
        verdict = "INSUFFICIENT EVIDENCE"
        print(f"\n❌ {verdict}")
        print(f"   Improvement: {improvement:.1f}% (p={p_value:.4f})")

    results['verdict'] = verdict

    # Save
    output_path = Path(__file__).parent.parent / 'results' / 'z1945_gpu_rapid_signals.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == '__main__':
    main()
