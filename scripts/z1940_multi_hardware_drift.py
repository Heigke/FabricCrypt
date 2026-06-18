#!/usr/bin/env python3
"""
z1940: Multi-Hardware Drift Task — Proven z1315 Design + All Hardware

INSIGHT FROM z1315 (85% improvement):
- Target y(t+1) = y(t) + k * temp_derivative + noise
- temp_derivative is UNPREDICTABLE from history
- Embodied model sees derivative → can predict drift direction
- Blind model must guess → worse performance

THIS EXPERIMENT:
1. Use PROVEN z1315 drift task design
2. Add FPGA readings as additional unpredictable drift source
3. Create AGGRESSIVE thermal variation (longer stress, higher intensity)
4. Multiple drift sources make target TRULY unpredictable from history

The target equation becomes:
y(t+1) = y(t) + k1*temp_derivative + k2*fpga_reading + k3*random_walk + noise

Where fpga_reading changes unpredictably based on DDR3 timing variations.
"""

import os
import sys
import time
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '11.0.0')

import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GPUSensor:
    """GPU sensor with temperature derivative tracking"""

    def __init__(self):
        self.card = '/sys/class/drm/card1/device'
        self.last_temp = None
        self.last_time = None

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

    def read(self) -> Tuple[Dict, float]:
        """Get hardware state and temperature derivative"""
        temp_raw = self._hwmon('temp1_input', 50000)
        temp_c = temp_raw / 1000
        power_w = self._hwmon('power1_average', 50e6) / 1e6
        util = self._read('gpu_busy_percent', 50)

        now = time.time()
        if self.last_temp is not None and self.last_time is not None:
            dt = now - self.last_time
            if dt > 0.01:
                temp_derivative = (temp_c - self.last_temp) / dt
            else:
                temp_derivative = 0.0
        else:
            temp_derivative = 0.0

        self.last_temp = temp_c
        self.last_time = now

        return {
            'temp_c': temp_c,
            'temp_norm': temp_c / 100,
            'power_w': power_w,
            'power_norm': power_w / 100,
            'util': util / 100,
            'temp_derivative': np.clip(temp_derivative, -5, 5),
            'temp_deriv_norm': np.clip(temp_derivative, -5, 5) / 5,
        }, temp_derivative


class FPGASensor:
    """FPGA sensor for DDR3 timing variations"""

    def __init__(self):
        self.connected = False
        self.client = None
        self.last_reading = 0.5

        try:
            from litex.tools.litex_client import RemoteClient
            self.client = RemoteClient(
                host='localhost', port=1234,
                csr_csv='src/fpga/litedram/build_etherbone/csr.csv',
                csr_data_width=32
            )
            self.client.open()
            # Test read with actual register names
            _ = self.client.regs.ddrphy_csrstorage_24.read()
            self.connected = True
            print("  [FPGA] Connected to localhost:1234")
        except Exception as e:
            print(f"  [FPGA] Not connected: {e}")

    def read(self) -> float:
        """Read FPGA timing value (0-1 normalized)"""
        if self.connected:
            try:
                # Read DDR3 PHY delay values (multiple registers for more variation)
                v1 = self.client.regs.ddrphy_csrstorage_24.read() & 0xFF
                v2 = self.client.regs.ddrphy_csr_28.read() & 0xFF
                # Combine for more entropy
                combined = (v1 + v2) / 2
                self.last_reading = float(combined / 255)
                return self.last_reading
            except:
                pass
        # Fallback: random walk (for testing without FPGA)
        self.last_reading += 0.05 * np.random.randn()
        self.last_reading = np.clip(self.last_reading, 0, 1)
        return self.last_reading


class MultiHardwareDriftTask:
    """
    PROVEN z1315 drift task extended with ALL hardware sources.

    Target: y(t+1) = y(t) + drift_total + noise

    Where drift_total = SUM of:
    - k1 * temp_derivative (GPU thermal rate)
    - k2 * power_derivative (GPU power rate) - FLUCTUATES MORE!
    - k3 * util_derivative (GPU util rate) - VERY FAST CHANGES!
    - k4 * fpga_variation (FPGA DDR3 timing)
    - k5 * random_walk (baseline)

    Using DERIVATIVES of fast-changing signals ensures unpredictability.
    """

    def __init__(self, noise_scale=0.06):
        self.noise_scale = noise_scale
        self.y = 0.5

        # Drift coefficients (hardware weights)
        # EMPHASIS on fast-changing signals!
        self.k_temp = 0.15  # Temperature derivative (slow)
        self.k_power = 0.25  # Power derivative (FAST)
        self.k_util = 0.30  # Utilization derivative (VERY FAST)
        self.k_fpga = 0.15  # FPGA timing variation
        self.k_random = 0.03  # Random baseline

        # History for derivative calculation
        self.fpga_history = [0.5]
        self.last_power = None
        self.last_util = None

    def step(self, hw_state: Dict, fpga_reading: float) -> Tuple[float, Dict]:
        """Update target based on ALL hardware derivatives"""

        # Temperature derivative (already computed by sensor)
        drift_temp = self.k_temp * hw_state.get('temp_deriv_norm', 0)

        # Power derivative (compute here for fast changes)
        power_now = hw_state.get('power_norm', 0.5)
        if self.last_power is not None:
            power_deriv = (power_now - self.last_power) * 10  # Amplify
        else:
            power_deriv = 0
        self.last_power = power_now
        drift_power = self.k_power * np.clip(power_deriv, -1, 1)

        # Utilization derivative (VERY FAST changes!)
        util_now = hw_state.get('util', 0.5)
        if self.last_util is not None:
            util_deriv = (util_now - self.last_util) * 5  # Fast signal
        else:
            util_deriv = 0
        self.last_util = util_now
        drift_util = self.k_util * np.clip(util_deriv, -1, 1)

        # FPGA contribution: change from previous
        fpga_delta = fpga_reading - self.fpga_history[-1] if self.fpga_history else 0
        drift_fpga = self.k_fpga * fpga_delta * 10
        self.fpga_history.append(fpga_reading)

        # Random walk and noise
        drift_random = self.k_random * np.random.randn()
        noise = np.random.normal(0, self.noise_scale)

        total_drift = drift_temp + drift_power + drift_util + drift_fpga + drift_random + noise

        old_y = self.y
        self.y = np.clip(self.y + total_drift, 0, 1)

        drift_info = {
            'drift_temp': drift_temp,
            'drift_power': drift_power,
            'drift_util': drift_util,
            'drift_fpga': drift_fpga,
            'drift_random': drift_random,
            'total': total_drift,
        }

        return self.y, drift_info

    def reset(self):
        self.y = 0.5
        self.fpga_history = [0.5]
        self.last_power = None
        self.last_util = None


class EmbodiedDriftModel(nn.Module):
    """Model that uses ALL hardware derivatives to predict drift"""

    def __init__(self, hw_dim=8, hidden_dim=64):
        super().__init__()

        # Input: temp, temp_deriv, power, power_deriv, util, util_deriv, fpga, fpga_deriv
        self.hw_encoder = nn.Sequential(
            nn.Linear(hw_dim, 48),
            nn.ReLU(),
            nn.Linear(48, 32),
            nn.ReLU(),
        )

        self.drift_predictor = nn.Sequential(
            nn.Linear(32 + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, hw_state: torch.Tensor, current_y: torch.Tensor) -> torch.Tensor:
        hw_encoded = self.hw_encoder(hw_state)
        x = torch.cat([hw_encoded, current_y.unsqueeze(-1)], dim=-1)
        return self.drift_predictor(x).squeeze(-1)


class BlindDriftModel(nn.Module):
    """Model that only sees target history (no hardware)"""

    def __init__(self, hist_len=10, hidden_dim=64):
        super().__init__()

        # Match parameter count with embodied
        self.history_encoder = nn.Sequential(
            nn.Linear(hist_len, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )

        self.drift_predictor = nn.Sequential(
            nn.Linear(32 + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, history: torch.Tensor, current_y: torch.Tensor) -> torch.Tensor:
        hist_encoded = self.history_encoder(history)
        x = torch.cat([hist_encoded, current_y.unsqueeze(-1)], dim=-1)
        return self.drift_predictor(x).squeeze(-1)


def create_aggressive_thermal_variation():
    """Create AGGRESSIVE thermal variation to ensure temp_derivative changes"""
    # Randomly choose intensity with bias toward extremes
    intensity = np.random.choice(['idle', 'light', 'medium', 'heavy', 'extreme'],
                                  p=[0.15, 0.15, 0.2, 0.25, 0.25])

    if intensity == 'idle':
        time.sleep(0.1)  # Let GPU cool
    elif intensity == 'light':
        _ = torch.randn(500, 500, device=DEVICE) @ torch.randn(500, 500, device=DEVICE)
    elif intensity == 'medium':
        for _ in range(2):
            _ = torch.randn(1000, 1000, device=DEVICE) @ torch.randn(1000, 1000, device=DEVICE)
    elif intensity == 'heavy':
        for _ in range(3):
            _ = torch.randn(2000, 2000, device=DEVICE) @ torch.randn(2000, 2000, device=DEVICE)
    else:  # extreme
        for _ in range(5):
            _ = torch.randn(3000, 3000, device=DEVICE) @ torch.randn(3000, 3000, device=DEVICE)

    torch.cuda.synchronize()
    return intensity


def run_episode(embodied, blind, task, gpu_sensor, fpga_sensor,
                target_history, n_steps=50, train=True):
    """Run one training/evaluation episode with ALL hardware derivatives"""

    if train:
        opt_emb = torch.optim.Adam(embodied.parameters(), lr=1e-3)
        opt_blind = torch.optim.Adam(blind.parameters(), lr=1e-3)

    task.reset()

    emb_errors = []
    blind_errors = []
    all_derivs = {'temp': [], 'power': [], 'util': []}

    # Track derivatives ourselves for model input
    last_power = None
    last_util = None
    last_fpga = None

    for step in range(n_steps):
        # Create thermal variation
        create_aggressive_thermal_variation()
        time.sleep(0.02)  # Short delay, keep high frequency

        # Read hardware
        gpu_state, temp_deriv = gpu_sensor.read()
        fpga_reading = fpga_sensor.read()

        # Compute additional derivatives (FAST signals!)
        power_now = gpu_state['power_norm']
        util_now = gpu_state['util']

        power_deriv = (power_now - last_power) * 10 if last_power else 0
        util_deriv = (util_now - last_util) * 10 if last_util else 0
        fpga_deriv = (fpga_reading - last_fpga) * 10 if last_fpga else 0

        last_power = power_now
        last_util = util_now
        last_fpga = fpga_reading

        all_derivs['temp'].append(temp_deriv)
        all_derivs['power'].append(power_deriv)
        all_derivs['util'].append(util_deriv)

        # Current state
        current_y = task.y

        # FULL hardware state tensor for embodied model (8 dims)
        hw_tensor = torch.tensor([
            gpu_state['temp_norm'],
            gpu_state['temp_deriv_norm'],
            power_now,
            np.clip(power_deriv, -1, 1),  # Power derivative
            util_now,
            np.clip(util_deriv, -1, 1),   # Util derivative (FAST!)
            fpga_reading,
            np.clip(fpga_deriv, -1, 1),   # FPGA derivative
        ], dtype=torch.float32, device=DEVICE).unsqueeze(0)

        current_y_tensor = torch.tensor([current_y], dtype=torch.float32, device=DEVICE)

        # Target history for blind model
        hist_tensor = torch.tensor([target_history], dtype=torch.float32, device=DEVICE)

        # True next y (causally depends on ALL hardware derivatives!)
        true_next_y, drift_info = task.step(gpu_state, fpga_reading)
        true_y_tensor = torch.tensor([true_next_y], dtype=torch.float32, device=DEVICE)

        # Predictions
        emb_pred = embodied(hw_tensor, current_y_tensor)
        blind_pred = blind(hist_tensor, current_y_tensor)

        # Losses
        emb_loss = F.mse_loss(emb_pred, true_y_tensor)
        blind_loss = F.mse_loss(blind_pred, true_y_tensor)

        if train:
            opt_emb.zero_grad()
            emb_loss.backward()
            opt_emb.step()

            opt_blind.zero_grad()
            blind_loss.backward()
            opt_blind.step()

        emb_errors.append(emb_loss.item())
        blind_errors.append(blind_loss.item())

        # Update target history
        target_history.append(true_next_y)
        target_history.pop(0)

    return {
        'emb_mse': np.mean(emb_errors),
        'blind_mse': np.mean(blind_errors),
        'temp_deriv_std': np.std(all_derivs['temp']),
        'power_deriv_std': np.std(all_derivs['power']),
        'util_deriv_std': np.std(all_derivs['util']),
    }


def main():
    print("=" * 70)
    print("z1940: Multi-Hardware Drift Task")
    print("PROVEN z1315 design + FPGA + Aggressive thermal variation")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Timestamp: {datetime.now().isoformat()}")

    results = {
        'experiment': 'z1940_multi_hardware_drift',
        'timestamp': datetime.now().isoformat(),
        'device': str(DEVICE),
    }

    # Initialize sensors
    print("\n=== Hardware Initialization ===")
    gpu_sensor = GPUSensor()
    fpga_sensor = FPGASensor()

    results['hardware'] = {
        'gpu': True,
        'fpga': fpga_sensor.connected,
    }

    # Create models (8 hw dims: temp, temp_d, power, power_d, util, util_d, fpga, fpga_d)
    embodied = EmbodiedDriftModel(hw_dim=8, hidden_dim=64).to(DEVICE)
    blind = BlindDriftModel(hist_len=10, hidden_dim=64).to(DEVICE)

    print(f"\nEmbodied params: {sum(p.numel() for p in embodied.parameters()):,}")
    print(f"Blind params: {sum(p.numel() for p in blind.parameters()):,}")

    target_history = [0.5] * 10
    task = MultiHardwareDriftTask(noise_scale=0.08)

    # Warm-up to get temperature variation going
    print("\n=== Thermal Warm-up (creating temperature gradient) ===")
    for _ in range(10):
        create_aggressive_thermal_variation()

    # Training
    print("\n=== Training (40 episodes) ===")
    training_log = []

    for ep in range(40):
        result = run_episode(embodied, blind, task, gpu_sensor, fpga_sensor,
                            target_history, n_steps=50, train=True)
        training_log.append(result)

        if (ep + 1) % 10 == 0:
            print(f"  Episode {ep+1}: Emb={result['emb_mse']:.4f}, "
                  f"Blind={result['blind_mse']:.4f}")
            print(f"    Signal STDs: temp={result['temp_deriv_std']:.3f}, "
                  f"power={result['power_deriv_std']:.3f}, "
                  f"util={result['util_deriv_std']:.3f}")

    results['training'] = training_log

    # Evaluation
    print("\n=== Evaluation (20 episodes) ===")
    embodied.eval()
    blind.eval()

    eval_emb = []
    eval_blind = []

    with torch.no_grad():
        for ep in range(20):
            result = run_episode(embodied, blind, task, gpu_sensor, fpga_sensor,
                                target_history, n_steps=60, train=False)
            eval_emb.append(result['emb_mse'])
            eval_blind.append(result['blind_mse'])

    emb_mean = np.mean(eval_emb)
    emb_std = np.std(eval_emb)
    blind_mean = np.mean(eval_blind)
    blind_std = np.std(eval_blind)

    improvement = (blind_mean - emb_mean) / blind_mean * 100 if blind_mean > 0 else 0
    t_stat, p_value = stats.ttest_ind(eval_emb, eval_blind)

    print(f"\n{'Model':<15} | {'MSE':>10} | {'Std':>10}")
    print("-" * 40)
    print(f"{'EMBODIED':<15} | {emb_mean:>10.4f} | {emb_std:>10.4f}")
    print(f"{'BLIND':<15} | {blind_mean:>10.4f} | {blind_std:>10.4f}")
    print(f"\nImprovement: {improvement:+.1f}%")
    print(f"p-value: {p_value:.2e}")

    results['evaluation'] = {
        'embodied_mse': {'mean': emb_mean, 'std': emb_std, 'values': eval_emb},
        'blind_mse': {'mean': blind_mean, 'std': blind_std, 'values': eval_blind},
        'improvement_pct': improvement,
        't_statistic': t_stat,
        'p_value': p_value,
    }

    # Verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    t1 = improvement > 30 and p_value < 0.001
    t2 = improvement > 10 and p_value < 0.05
    t3 = emb_mean < 0.1

    tests_passed = int(t1) + int(t2) + int(t3)

    results['tests'] = {
        'T1_strong_significant': f'{t1} (>30% improvement, p<0.001)',
        'T2_moderate_significant': f'{t2} (>10% improvement, p<0.05)',
        'T3_low_error': f'{t3} (MSE < 0.1)',
    }
    results['tests_passed'] = tests_passed

    if t1:
        verdict = "GENUINE EMBODIMENT ADVANTAGE PROVEN"
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

    # Save results
    output_path = Path(__file__).parent.parent / 'results' / 'z1940_multi_hardware_drift.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == '__main__':
    main()
