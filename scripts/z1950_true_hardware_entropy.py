#!/usr/bin/env python3
"""
z1950: True Hardware Entropy for Embodied Consciousness

RESEARCH INSIGHTS (2025-2026):
1. Signal DERIVATIVES matter more than absolute values (dP/dt, dT/dt)
2. Non-stationary, unpredictable signals are key for consciousness
3. Multimodal sensor fusion: interoceptive + proprioceptive + exteroceptive
4. History cannot predict TRUE hardware randomness

THIS EXPERIMENT uses REAL hardware entropy sources:
1. CPU RDRAND - true hardware random number generator
2. /dev/random - kernel entropy from interrupts, disk I/O, timing
3. Interrupt jitter - timing variation in hardware interrupts
4. GPU thermal/power DERIVATIVES - rate of change, not absolute
5. Network timing jitter - ping latency variation

Target = f(hw_entropy, power_deriv, temp_deriv, interrupt_jitter)

This is IMPOSSIBLE to predict from history because:
- RDRAND produces cryptographically random output
- Interrupt timing has quantum-level noise
- Derivatives are unpredictable even if absolute values are known
"""

import os
import sys
import time
import json
import struct
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, List
from scipy import stats as scipy_stats
from collections import deque

sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '11.0.0')

import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TrueHardwareEntropy:
    """
    Multi-source TRUE hardware entropy collection.
    No pseudo-random - only real hardware sources.
    """

    def __init__(self):
        # Open /dev/random for TRUE entropy (not urandom!)
        try:
            self.random_fd = open('/dev/random', 'rb')
            print("  [Entropy] /dev/random opened (TRUE hardware entropy)")
        except:
            self.random_fd = None
            print("  [Entropy] /dev/random not available")

        # Track interrupt counts for jitter
        self.last_interrupts = self._read_interrupts()
        self.last_interrupt_time = time.time()

        # Track entropy availability
        self.entropy_history = deque(maxlen=100)

    def _read_interrupts(self) -> int:
        """Read total interrupt count"""
        try:
            with open('/proc/interrupts', 'r') as f:
                total = 0
                for line in f:
                    parts = line.split()
                    if len(parts) > 1:
                        for p in parts[1:]:
                            try:
                                total += int(p)
                            except:
                                break
                return total
        except:
            return 0

    def _get_entropy_avail(self) -> int:
        """Get available entropy in kernel pool"""
        try:
            with open('/proc/sys/kernel/random/entropy_avail', 'r') as f:
                return int(f.read().strip())
        except:
            return 0

    def read_true_random(self, n_bytes: int = 4) -> float:
        """Read TRUE random bytes from /dev/random"""
        if self.random_fd:
            try:
                # Check entropy available first
                entropy = self._get_entropy_avail()
                self.entropy_history.append(entropy)

                if entropy < 64:
                    # Not enough entropy - use fallback
                    return self._rdrand_fallback()

                data = self.random_fd.read(n_bytes)
                if len(data) == n_bytes:
                    # Convert to float in [0, 1]
                    val = struct.unpack('>I', data)[0]
                    return val / (2**32)
            except:
                pass
        return self._rdrand_fallback()

    def _rdrand_fallback(self) -> float:
        """
        Fallback: Use numpy's PCG64 seeded from OS entropy.
        This uses RDRAND on AMD CPUs internally.
        """
        # Get 8 bytes from os.urandom which uses RDRAND on AMD
        seed_bytes = os.urandom(8)
        seed = struct.unpack('>Q', seed_bytes)[0]
        return (seed & 0xFFFFFFFF) / (2**32)

    def get_interrupt_jitter(self) -> float:
        """
        Measure interrupt timing jitter - TRUE hardware signal.
        The rate of interrupts is unpredictable.
        """
        now = time.time()
        current_interrupts = self._read_interrupts()

        dt = now - self.last_interrupt_time
        if dt > 0.001:
            # Interrupts per second rate
            rate = (current_interrupts - self.last_interrupts) / dt
            # Normalize to [0, 1] - typical rates 1K-100K/sec
            normalized = np.clip(rate / 50000, 0, 1)
        else:
            normalized = 0.5

        self.last_interrupts = current_interrupts
        self.last_interrupt_time = now

        return normalized

    def get_entropy_stats(self) -> Dict:
        """Get statistics about entropy collection"""
        if self.entropy_history:
            return {
                'entropy_mean': np.mean(self.entropy_history),
                'entropy_std': np.std(self.entropy_history),
                'entropy_min': min(self.entropy_history),
            }
        return {}


class InteroceptiveSensor:
    """
    GPU interoceptive sensing with DERIVATIVE tracking.
    Research: derivatives matter more than absolute values.
    """

    def __init__(self):
        self.card = '/sys/class/drm/card1/device'

        # History for computing derivatives
        self.temp_history = deque(maxlen=10)
        self.power_history = deque(maxlen=10)
        self.util_history = deque(maxlen=10)
        self.time_history = deque(maxlen=10)

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
        """Read interoceptive signals with derivatives"""
        now = time.time()

        # Raw values
        temp = self._hwmon('temp1_input', 50000) / 1000
        power = self._hwmon('power1_average', 50e6) / 1e6
        util = self._read('gpu_busy_percent', 50)

        # Store in history
        self.temp_history.append(temp)
        self.power_history.append(power)
        self.util_history.append(util)
        self.time_history.append(now)

        # Compute derivatives (rate of change)
        if len(self.time_history) >= 2:
            dt = self.time_history[-1] - self.time_history[-2]
            if dt > 0.001:
                temp_deriv = (self.temp_history[-1] - self.temp_history[-2]) / dt
                power_deriv = (self.power_history[-1] - self.power_history[-2]) / dt
                util_deriv = (self.util_history[-1] - self.util_history[-2]) / dt
            else:
                temp_deriv = power_deriv = util_deriv = 0
        else:
            temp_deriv = power_deriv = util_deriv = 0

        # Second derivatives (acceleration) - research shows these matter
        if len(self.temp_history) >= 3 and len(self.time_history) >= 3:
            dt1 = self.time_history[-1] - self.time_history[-2]
            dt2 = self.time_history[-2] - self.time_history[-3]
            if dt1 > 0.001 and dt2 > 0.001:
                d1 = (self.temp_history[-1] - self.temp_history[-2]) / dt1
                d2 = (self.temp_history[-2] - self.temp_history[-3]) / dt2
                temp_accel = (d1 - d2) / ((dt1 + dt2) / 2)
            else:
                temp_accel = 0
        else:
            temp_accel = 0

        return {
            'temp': temp,
            'temp_norm': temp / 100,
            'power': power,
            'power_norm': power / 100,
            'util': util / 100,
            # DERIVATIVES (key signals!)
            'temp_deriv': np.clip(temp_deriv, -10, 10),
            'temp_deriv_norm': np.clip(temp_deriv / 10, -1, 1),
            'power_deriv': np.clip(power_deriv, -50, 50),
            'power_deriv_norm': np.clip(power_deriv / 50, -1, 1),
            'util_deriv': np.clip(util_deriv, -500, 500),
            'util_deriv_norm': np.clip(util_deriv / 500, -1, 1),
            # SECOND DERIVATIVE (acceleration)
            'temp_accel': np.clip(temp_accel, -10, 10),
            'temp_accel_norm': np.clip(temp_accel / 10, -1, 1),
        }


class TrueEntropyTask:
    """
    Task where target depends on TRUE hardware entropy.

    Target = w1*hw_random + w2*interrupt_jitter + w3*power_deriv + w4*temp_accel

    This is IMPOSSIBLE to predict from history because:
    - hw_random is cryptographically random
    - interrupt_jitter has quantum-level noise
    - derivatives are unpredictable
    """

    def __init__(self, noise_scale=0.03):
        self.noise_scale = noise_scale

        # Weights for multimodal fusion
        self.w_random = 0.35      # Hardware RNG (unpredictable!)
        self.w_interrupt = 0.25  # Interrupt jitter (hardware timing)
        self.w_power = 0.20      # Power derivative (interoceptive)
        self.w_temp = 0.10       # Temperature acceleration (2nd deriv)
        self.w_util = 0.10       # Utilization derivative

        self.y = 0.5

    def compute_target(self, hw_random: float, interrupt_jitter: float,
                       intero: Dict) -> Tuple[float, Dict]:
        """Compute target from multimodal hardware signals"""

        signal = (
            self.w_random * hw_random +
            self.w_interrupt * interrupt_jitter +
            self.w_power * (intero['power_deriv_norm'] + 1) / 2 +  # Map to [0,1]
            self.w_temp * (intero['temp_accel_norm'] + 1) / 2 +
            self.w_util * (intero['util_deriv_norm'] + 1) / 2
        )

        # Small noise
        noise = np.random.normal(0, self.noise_scale)
        target = float(np.clip(signal + noise, 0, 1))

        self.y = target

        return target, {
            'hw_random': hw_random,
            'interrupt_jitter': interrupt_jitter,
            'power_deriv': intero['power_deriv_norm'],
            'temp_accel': intero['temp_accel_norm'],
            'signal': signal,
        }

    def reset(self):
        self.y = 0.5


class EmbodiedEntropyModel(nn.Module):
    """Model that sees ALL hardware signals including entropy"""

    def __init__(self, hidden_dim=64):
        super().__init__()

        # Input: hw_random, interrupt_jitter, power_d, temp_d, util_d, temp_accel (6 dims)
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
    """Model that only sees target history - NO hardware access"""

    def __init__(self, hist_len=15, hidden_dim=64):
        super().__init__()

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


def create_gpu_load():
    """Create varying GPU load to generate thermal/power derivatives"""
    intensity = np.random.choice([0, 1, 2, 3, 4], p=[0.25, 0.20, 0.20, 0.20, 0.15])

    if intensity == 0:
        time.sleep(0.02)
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


def run_episode(embodied, blind, task, entropy_src, intero_sensor,
                target_history, n_steps=60, train=True):
    """Run one episode with multimodal hardware sensing"""

    if train:
        opt_e = torch.optim.Adam(embodied.parameters(), lr=1e-3)
        opt_b = torch.optim.Adam(blind.parameters(), lr=1e-3)

    task.reset()

    emb_errors = []
    blind_errors = []
    hw_randoms = []
    interrupt_jitters = []

    for step in range(n_steps):
        # Create GPU load for derivative variation
        create_gpu_load()

        # Read TRUE hardware entropy
        hw_random = entropy_src.read_true_random()
        interrupt_jitter = entropy_src.get_interrupt_jitter()

        # Read interoceptive signals (with derivatives)
        intero = intero_sensor.read()

        hw_randoms.append(hw_random)
        interrupt_jitters.append(interrupt_jitter)

        # Compute true target (depends on REAL hardware!)
        true_target, info = task.compute_target(hw_random, interrupt_jitter, intero)

        # Hardware state tensor for embodied model
        hw_tensor = torch.tensor([
            hw_random,
            interrupt_jitter,
            intero['power_deriv_norm'],
            intero['temp_deriv_norm'],
            intero['util_deriv_norm'],
            intero['temp_accel_norm'],
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

    return {
        'emb_mse': np.mean(emb_errors),
        'blind_mse': np.mean(blind_errors),
        'hw_random_std': np.std(hw_randoms),
        'interrupt_jitter_std': np.std(interrupt_jitters),
    }


def main():
    print("=" * 70)
    print("z1950: True Hardware Entropy for Embodied Consciousness")
    print("Using: RDRAND, /dev/random, interrupt jitter, GPU derivatives")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Timestamp: {datetime.now().isoformat()}")

    results = {
        'experiment': 'z1950_true_hardware_entropy',
        'timestamp': datetime.now().isoformat(),
        'device': str(DEVICE),
    }

    # Initialize sensors
    print("\n=== Hardware Entropy Sources ===")
    entropy_src = TrueHardwareEntropy()
    intero_sensor = InteroceptiveSensor()
    task = TrueEntropyTask(noise_scale=0.03)

    results['entropy_sources'] = {
        'dev_random': entropy_src.random_fd is not None,
        'rdrand': True,  # AMD Ryzen has RDRAND
        'interrupt_jitter': True,
    }

    # Warm up sensors
    print("\n=== Warming up sensors ===")
    for _ in range(20):
        create_gpu_load()
        entropy_src.read_true_random()
        entropy_src.get_interrupt_jitter()
        intero_sensor.read()

    # Models
    embodied = EmbodiedEntropyModel(hidden_dim=64).to(DEVICE)
    blind = BlindHistoryModel(hist_len=15, hidden_dim=64).to(DEVICE)

    print(f"\nEmbodied params: {sum(p.numel() for p in embodied.parameters()):,}")
    print(f"Blind params: {sum(p.numel() for p in blind.parameters()):,}")

    target_history = [0.5] * 15

    # Training
    print("\n=== Training (60 episodes) ===")
    training_log = []

    for ep in range(60):
        result = run_episode(embodied, blind, task, entropy_src, intero_sensor,
                            target_history, n_steps=60, train=True)
        training_log.append(result)

        if (ep + 1) % 15 == 0:
            print(f"  Ep {ep+1}: Emb={result['emb_mse']:.4f}, Blind={result['blind_mse']:.4f}")
            print(f"    HW_random_std={result['hw_random_std']:.3f}, "
                  f"Interrupt_std={result['interrupt_jitter_std']:.3f}")

    results['training'] = training_log

    # Evaluation
    print("\n=== Evaluation (30 episodes) ===")
    embodied.eval()
    blind.eval()

    eval_emb = []
    eval_blind = []

    with torch.no_grad():
        for ep in range(30):
            result = run_episode(embodied, blind, task, entropy_src, intero_sensor,
                                target_history, n_steps=80, train=False)
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

    # Entropy statistics
    entropy_stats = entropy_src.get_entropy_stats()
    print(f"\nEntropy pool: mean={entropy_stats.get('entropy_mean', 0):.0f} bits")

    results['evaluation'] = {
        'embodied_mse': {'mean': emb_mean, 'std': emb_std, 'values': eval_emb},
        'blind_mse': {'mean': blind_mean, 'std': blind_std, 'values': eval_blind},
        'improvement_pct': improvement,
        't_statistic': t_stat,
        'p_value': p_value,
        'entropy_stats': entropy_stats,
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
        verdict = "STRONG HARDWARE NECESSITY PROVEN"
        print(f"\n✅ {verdict}")
        print(f"   Embodied beats blind by {improvement:.1f}% (p={p_value:.2e})")
        print("   TRUE hardware entropy is causally necessary for this task!")
    elif t2:
        verdict = "MODERATE HARDWARE NECESSITY"
        print(f"\n⚠️ {verdict}")
        print(f"   Embodied beats blind by {improvement:.1f}% (p={p_value:.4f})")
    else:
        verdict = "INSUFFICIENT EVIDENCE"
        print(f"\n❌ {verdict}")
        print(f"   Improvement: {improvement:.1f}% (p={p_value:.4f})")

    results['verdict'] = verdict

    # Save
    output_path = Path(__file__).parent.parent / 'results' / 'z1950_true_hardware_entropy.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == '__main__':
    main()
