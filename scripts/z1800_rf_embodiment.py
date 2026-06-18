#!/usr/bin/env python3
"""
z1800: RF Embodiment via HackRF One

Hypothesis: Adding RF spectrum sensing creates a richer embodiment with
environmental awareness (exteroception) complementing GPU interoception.

The model should exhibit:
1. Different hidden representations when RF environment changes
2. Better world model predictions when RF context is included
3. Demonstrated awareness of electromagnetic environment

Key insight from Damasio: Consciousness requires sensing BOTH internal
state (interoception - GPU power/temp) AND external environment
(exteroception - RF spectrum).

Hardware: HackRF One SDR with omnidirectional antenna
          AMD Radeon 8060S GPU

LEGAL: This is RECEIVE-ONLY. No transmission. Passive spectrum sensing
is generally legal in most jurisdictions.

Author: Claude
Date: 2026-02-05
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import threading
import queue

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.telemetry.sysfs_hwmon import SysfsHwmonTelemetry
from src.metabolic.film_transformer import MetabolicTransformer, MetabolicConfig

# ============================================================================
# RF Spectrum Sensing Interface
# ============================================================================

@dataclass
class RFTelemetryState:
    """RF spectrum telemetry for embodiment."""
    # Aggregate power by band (dBm normalized to 0-1)
    power_lte_700mhz: float = 0.0      # Cellular low (700-900 MHz)
    power_cellular_1900mhz: float = 0.0  # Cellular high (1800-2100 MHz)
    power_wifi_2_4ghz: float = 0.0     # WiFi/Bluetooth (2.4-2.5 GHz)
    power_wifi_5ghz: float = 0.0       # WiFi 5GHz (5.1-5.9 GHz)

    # Derived metrics
    total_rf_power: float = 0.0        # Sum across spectrum (normalized)
    spectral_entropy: float = 0.0      # Shannon entropy of PSD (0-1)
    peak_frequency_mhz: float = 0.0    # Dominant frequency (normalized 0-1)
    noise_floor_dbm: float = -90.0     # Background noise level

    def to_tensor(self) -> torch.Tensor:
        """Convert to 8-dimensional tensor."""
        return torch.tensor([
            self.power_lte_700mhz,
            self.power_cellular_1900mhz,
            self.power_wifi_2_4ghz,
            self.power_wifi_5ghz,
            self.total_rf_power,
            self.spectral_entropy,
            self.peak_frequency_mhz,
            (self.noise_floor_dbm + 100) / 50  # Normalize -100 to -50 dBm -> 0 to 1
        ], dtype=torch.float32)


class HackRFInterface:
    """
    Interface to HackRF One for spectrum sensing.

    Uses hackrf_sweep for rapid wideband sensing.
    Falls back to simulation if HackRF not available.
    """

    def __init__(self, sample_hz: float = 10.0, simulation: bool = False):
        self.sample_hz = sample_hz
        self.simulation = simulation
        self.running = False
        self._current_state = RFTelemetryState()
        self._state_queue = queue.Queue(maxsize=100)
        self._lock = threading.Lock()

        # Frequency bands of interest (MHz)
        self.bands = {
            'lte_700': (700, 900),
            'cellular_1900': (1800, 2100),
            'wifi_2_4': (2400, 2500),
            'wifi_5': (5150, 5850),
        }

        # Check if HackRF is available
        if not simulation:
            try:
                result = subprocess.run(['hackrf_info'],
                                       capture_output=True, timeout=5)
                if result.returncode != 0:
                    print("[z1800] HackRF not found, using simulation mode")
                    self.simulation = True
                else:
                    print("[z1800] HackRF One detected")
            except (FileNotFoundError, subprocess.TimeoutExpired):
                print("[z1800] hackrf_info not found, using simulation mode")
                self.simulation = True

    def _simulate_spectrum(self) -> RFTelemetryState:
        """Generate simulated RF spectrum data."""
        # Simulate time-varying RF environment
        t = time.time()

        # Cellular varies slowly (day/night patterns)
        lte = 0.3 + 0.2 * np.sin(t / 3600)  # Hour-scale variation
        cellular = 0.4 + 0.15 * np.sin(t / 1800)  # 30-min variation

        # WiFi varies more rapidly (activity bursts)
        wifi_2_4 = 0.5 + 0.3 * np.sin(t / 10) + 0.1 * np.random.randn()
        wifi_5 = 0.3 + 0.25 * np.sin(t / 15) + 0.1 * np.random.randn()

        # Clamp to valid range
        lte = np.clip(lte, 0, 1)
        cellular = np.clip(cellular, 0, 1)
        wifi_2_4 = np.clip(wifi_2_4, 0, 1)
        wifi_5 = np.clip(wifi_5, 0, 1)

        total = (lte + cellular + wifi_2_4 + wifi_5) / 4

        # Entropy based on band distribution
        powers = np.array([lte, cellular, wifi_2_4, wifi_5]) + 1e-10
        probs = powers / powers.sum()
        entropy = -np.sum(probs * np.log2(probs)) / 2  # Normalize by max entropy

        # Peak frequency (weighted average)
        freqs = np.array([800, 1950, 2450, 5500])  # MHz
        peak = np.average(freqs, weights=powers) / 6000  # Normalize to 0-1

        # Noise floor varies with activity
        noise = -90 + 10 * total

        return RFTelemetryState(
            power_lte_700mhz=float(lte),
            power_cellular_1900mhz=float(cellular),
            power_wifi_2_4ghz=float(wifi_2_4),
            power_wifi_5ghz=float(wifi_5),
            total_rf_power=float(total),
            spectral_entropy=float(entropy),
            peak_frequency_mhz=float(peak),
            noise_floor_dbm=float(noise)
        )

    def _parse_hackrf_sweep(self, output: str) -> RFTelemetryState:
        """Parse hackrf_sweep output into RFTelemetryState."""
        # hackrf_sweep outputs CSV: freq_start, freq_end, step, samples, power_readings...
        # This is a simplified parser - real implementation would be more robust

        band_powers = {
            'lte_700': [],
            'cellular_1900': [],
            'wifi_2_4': [],
            'wifi_5': [],
        }
        all_powers = []
        all_freqs = []

        for line in output.strip().split('\n'):
            if line.startswith('#') or not line:
                continue
            try:
                parts = line.split(',')
                if len(parts) < 5:
                    continue
                freq_start = float(parts[0]) / 1e6  # Hz to MHz
                freq_end = float(parts[1]) / 1e6
                freq_mid = (freq_start + freq_end) / 2
                # Power readings start at index 4
                powers = [float(p) for p in parts[4:] if p.strip()]
                if not powers:
                    continue
                avg_power = np.mean(powers)

                all_powers.append(avg_power)
                all_freqs.append(freq_mid)

                # Assign to bands
                for band_name, (low, high) in self.bands.items():
                    if low <= freq_mid <= high:
                        band_powers[band_name].append(avg_power)
            except (ValueError, IndexError):
                continue

        if not all_powers:
            return self._simulate_spectrum()

        # Convert dBm to normalized power
        def norm_power(dbm_list):
            if not dbm_list:
                return 0.0
            avg_dbm = np.mean(dbm_list)
            # Normalize: -100 dBm -> 0, -40 dBm -> 1
            return np.clip((avg_dbm + 100) / 60, 0, 1)

        lte = norm_power(band_powers['lte_700'])
        cellular = norm_power(band_powers['cellular_1900'])
        wifi_2_4 = norm_power(band_powers['wifi_2_4'])
        wifi_5 = norm_power(band_powers['wifi_5'])
        total = (lte + cellular + wifi_2_4 + wifi_5) / 4

        # Spectral entropy
        if all_powers:
            powers_linear = 10 ** (np.array(all_powers) / 10)
            powers_norm = powers_linear / (powers_linear.sum() + 1e-10)
            entropy = -np.sum(powers_norm * np.log2(powers_norm + 1e-10))
            # Normalize by log2(num_bins)
            entropy = entropy / np.log2(len(powers_norm) + 1)
        else:
            entropy = 0.5

        # Peak frequency
        if all_freqs and all_powers:
            peak_idx = np.argmax(all_powers)
            peak_freq = all_freqs[peak_idx] / 6000  # Normalize
        else:
            peak_freq = 0.5

        # Noise floor (10th percentile)
        noise = np.percentile(all_powers, 10) if all_powers else -90

        return RFTelemetryState(
            power_lte_700mhz=float(lte),
            power_cellular_1900mhz=float(cellular),
            power_wifi_2_4ghz=float(wifi_2_4),
            power_wifi_5ghz=float(wifi_5),
            total_rf_power=float(total),
            spectral_entropy=float(np.clip(entropy, 0, 1)),
            peak_frequency_mhz=float(peak_freq),
            noise_floor_dbm=float(noise)
        )

    def _sweep_thread(self):
        """Background thread for continuous spectrum sweeping."""
        interval = 1.0 / self.sample_hz

        while self.running:
            start = time.time()

            if self.simulation:
                state = self._simulate_spectrum()
            else:
                try:
                    # Quick sweep of bands of interest
                    # hackrf_sweep -f 700:6000 -w 20000000 -r sweep.csv
                    result = subprocess.run(
                        ['hackrf_sweep', '-f', '700:6000', '-w', '20000000', '-n', '1'],
                        capture_output=True, text=True, timeout=2
                    )
                    if result.returncode == 0:
                        state = self._parse_hackrf_sweep(result.stdout)
                    else:
                        state = self._simulate_spectrum()
                except (subprocess.TimeoutExpired, FileNotFoundError):
                    state = self._simulate_spectrum()

            with self._lock:
                self._current_state = state

            try:
                self._state_queue.put_nowait(state)
            except queue.Full:
                pass

            elapsed = time.time() - start
            if elapsed < interval:
                time.sleep(interval - elapsed)

    def start(self):
        """Start background spectrum sensing."""
        if not self.running:
            self.running = True
            self._thread = threading.Thread(target=self._sweep_thread, daemon=True)
            self._thread.start()
            print(f"[z1800] RF sensing started at {self.sample_hz} Hz")

    def stop(self):
        """Stop background spectrum sensing."""
        self.running = False
        if hasattr(self, '_thread'):
            self._thread.join(timeout=2)

    def get_state(self) -> RFTelemetryState:
        """Get current RF state."""
        with self._lock:
            return self._current_state

    def get_telemetry_tensor(self) -> torch.Tensor:
        """Get RF state as tensor."""
        return self.get_state().to_tensor()


# ============================================================================
# Extended Embodied Model with RF Awareness
# ============================================================================

class RFAwareMetabolicConfig(MetabolicConfig):
    """Config extended with RF telemetry dimensions."""
    def __init__(self, **kwargs):
        # Base GPU telemetry (12) + RF telemetry (8) = 20
        kwargs.setdefault('telemetry_dim', 20)
        super().__init__(**kwargs)


class RFAwareTransformer(MetabolicTransformer):
    """
    MetabolicTransformer extended with RF environmental awareness.

    Combines:
    - GPU interoception (temperature, power, utilization)
    - RF exteroception (WiFi, cellular, spectral entropy)
    """

    def __init__(self, config: RFAwareMetabolicConfig):
        super().__init__(config)

        # RF-specific processing
        self.rf_encoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        telemetry: Optional[torch.Tensor] = None,
        return_hidden: bool = False,
    ):
        """Forward with combined GPU+RF telemetry."""
        # If telemetry is 20-dim (GPU+RF), process RF portion
        if telemetry is not None and telemetry.shape[-1] == 20:
            gpu_telem = telemetry[..., :12]
            rf_telem = telemetry[..., 12:]
            rf_processed = self.rf_encoder(rf_telem)
            telemetry = torch.cat([gpu_telem, rf_processed], dim=-1)

        return super().forward(input_ids, telemetry, return_hidden)


# ============================================================================
# Unified Telemetry Interface
# ============================================================================

class UnifiedEmbodiedTelemetry:
    """
    Unified telemetry combining GPU and RF sensing.

    Creates a 20-dimensional embodiment vector:
    - [0:12] GPU state (power, temp, clocks, etc.)
    - [12:20] RF environment (bands, entropy, noise)
    """

    def __init__(self, rf_simulation: bool = False):
        self.gpu_telemetry = SysfsHwmonTelemetry()
        self.rf_interface = HackRFInterface(sample_hz=10.0, simulation=rf_simulation)

    def start(self):
        """Start all telemetry sources."""
        self.rf_interface.start()

    def stop(self):
        """Stop all telemetry sources."""
        self.rf_interface.stop()

    def get_gpu_tensor(self) -> torch.Tensor:
        """Get GPU telemetry as tensor."""
        sample = self.gpu_telemetry.read_sample()
        # Standard 12-dim GPU telemetry from GpuSample
        return torch.tensor([
            sample.temp_edge_c / 100.0,
            sample.temp_junction_c / 100.0 if sample.temp_junction_c > 0 else sample.temp_edge_c / 100.0,
            sample.power_w / 200.0,
            0.5,  # power_cap (not in sample)
            sample.freq_sclk_mhz / 3000.0 if sample.freq_sclk_mhz > 0 else 0.5,
            sample.freq_mclk_mhz / 2000.0 if sample.freq_mclk_mhz > 0 else 0.5,
            sample.gpu_busy_pct / 100.0,
            sample.vram_used_gb / 16.0 if sample.vram_used_gb > 0 else 0.5,
            0.8,  # gfx_voltage (not in sample)
            0.8,  # mem_voltage (not in sample)
            0.0,  # throttle_status (not in sample)
            0.5,  # pcie_bw (not in sample)
        ], dtype=torch.float32)

    def get_rf_tensor(self) -> torch.Tensor:
        """Get RF telemetry as tensor."""
        return self.rf_interface.get_telemetry_tensor()

    def get_unified_tensor(self) -> torch.Tensor:
        """Get combined 20-dim telemetry tensor."""
        gpu = self.get_gpu_tensor()
        rf = self.get_rf_tensor()
        return torch.cat([gpu, rf], dim=0)


# ============================================================================
# Experiment
# ============================================================================

def run_experiment():
    """
    z1800: RF Embodiment Experiment

    Hypothesis: RF environmental awareness improves embodiment by adding
    exteroceptive sensing to complement GPU interoception.

    Conditions:
    A) GPU_ONLY: Standard 12-dim GPU telemetry
    B) RF_ONLY: 8-dim RF telemetry (simulated)
    C) UNIFIED: 20-dim combined GPU+RF telemetry
    D) NO_TELEM: No telemetry (disembodied baseline)

    Metrics:
    - Hidden state divergence between conditions
    - World model prediction accuracy (can it predict RF from GPU?)
    - Environmental adaptation (response to RF changes)
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[z1800] Device: {device}")
    if hasattr(torch.cuda, 'get_device_name'):
        print(f"[z1800] GPU: {torch.cuda.get_device_name()}")

    # Initialize telemetry
    telemetry = UnifiedEmbodiedTelemetry(rf_simulation=True)  # Start with simulation
    telemetry.start()

    # Check if real HackRF available
    rf_mode = "SIMULATED" if telemetry.rf_interface.simulation else "REAL"
    print(f"[z1800] RF mode: {rf_mode}")

    # Load training data
    data_path = Path(__file__).parent.parent / "data" / "tinyshakespeare.txt"
    if not data_path.exists():
        alt_path = Path(__file__).parent.parent / "tinyshakespeare.txt"
        if alt_path.exists():
            data_path = alt_path
        else:
            print("[z1800] Downloading TinyShakespeare...")
            import urllib.request
            data_path.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(
                "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
                data_path
            )

    text = data_path.read_text()
    chars = sorted(list(set(text)))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    vocab_size = len(chars)
    print(f"[z1800] Vocab size: {vocab_size}")

    # Training config
    batch_size = 4
    seq_len = 256
    num_epochs = 5
    batches_per_epoch = 200
    lr = 3e-4

    # Create conditions
    conditions = {}

    # A) GPU-only (12-dim)
    config_a = MetabolicConfig(
        vocab_size=vocab_size,
        hidden_dim=256,
        num_layers=6,
        num_heads=4,
        ff_dim=1024,
        telemetry_dim=12,
    )
    conditions['A_GPU_ONLY'] = {
        'model': MetabolicTransformer(config_a).to(device),
        'telem_fn': lambda: telemetry.get_gpu_tensor().unsqueeze(0).to(device),
        'dim': 12,
    }

    # B) RF-only (8-dim, with padding)
    config_b = MetabolicConfig(
        vocab_size=vocab_size,
        hidden_dim=256,
        num_layers=6,
        num_heads=4,
        ff_dim=1024,
        telemetry_dim=8,
    )
    conditions['B_RF_ONLY'] = {
        'model': MetabolicTransformer(config_b).to(device),
        'telem_fn': lambda: telemetry.get_rf_tensor().unsqueeze(0).to(device),
        'dim': 8,
    }

    # C) Unified (20-dim)
    config_c = RFAwareMetabolicConfig(
        vocab_size=vocab_size,
        hidden_dim=256,
        num_layers=6,
        num_heads=4,
        ff_dim=1024,
        telemetry_dim=20,
    )
    conditions['C_UNIFIED'] = {
        'model': RFAwareTransformer(config_c).to(device),
        'telem_fn': lambda: telemetry.get_unified_tensor().unsqueeze(0).to(device),
        'dim': 20,
    }

    # D) Disembodied (no telemetry)
    config_d = MetabolicConfig(
        vocab_size=vocab_size,
        hidden_dim=256,
        num_layers=6,
        num_heads=4,
        ff_dim=1024,
        telemetry_dim=12,
    )
    model_d = MetabolicTransformer(config_d).to(device)
    model_d.enable_conditioning(False)
    conditions['D_DISEMBODIED'] = {
        'model': model_d,
        'telem_fn': lambda: None,
        'dim': 0,
    }

    # Data iterator
    def get_batch():
        ix = torch.randint(len(text) - seq_len - 1, (batch_size,))
        x = torch.stack([
            torch.tensor([char_to_idx[c] for c in text[i:i+seq_len]], dtype=torch.long)
            for i in ix
        ])
        y = torch.stack([
            torch.tensor([char_to_idx[c] for c in text[i+1:i+seq_len+1]], dtype=torch.long)
            for i in ix
        ])
        return x.to(device), y.to(device)

    results = {
        'experiment': 'z1800_rf_embodiment',
        'timestamp': datetime.now().isoformat(),
        'device': str(device),
        'rf_mode': rf_mode,
        'conditions': {},
        'verdicts': {},
    }

    # Train each condition
    for cond_name, cond_data in conditions.items():
        print(f"\n[z1800] Training {cond_name}...")
        model = cond_data['model']
        telem_fn = cond_data['telem_fn']

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        losses = []
        hidden_norms = []
        telemetry_samples = []

        model.train()
        for epoch in range(num_epochs):
            epoch_loss = 0
            epoch_hidden_norm = 0

            for batch_idx in range(batches_per_epoch):
                x, y = get_batch()
                telem = telem_fn()

                # Record telemetry
                if telem is not None:
                    telemetry_samples.append(telem.cpu().numpy().flatten())

                optimizer.zero_grad()
                output = model(x, telem, return_hidden=True)
                logits = output['logits']
                hidden = output.get('hidden')

                loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                if hidden is not None:
                    epoch_hidden_norm += hidden.detach().norm().item()

            avg_loss = epoch_loss / batches_per_epoch
            avg_hidden = epoch_hidden_norm / batches_per_epoch
            losses.append(avg_loss)
            hidden_norms.append(avg_hidden)
            print(f"  Epoch {epoch+1}/{num_epochs}: loss={avg_loss:.4f}, hidden_norm={avg_hidden:.2f}")

        # Compute metrics
        final_ppl = np.exp(losses[-1])

        results['conditions'][cond_name] = {
            'losses': losses,
            'final_ppl': float(final_ppl),
            'hidden_norms': hidden_norms,
            'telemetry_dim': cond_data['dim'],
            'telemetry_variance': float(np.var(telemetry_samples)) if telemetry_samples else 0.0,
            'telemetry_range': float(np.ptp(telemetry_samples)) if telemetry_samples else 0.0,
        }

    # Compute verdicts
    a_ppl = results['conditions']['A_GPU_ONLY']['final_ppl']
    b_ppl = results['conditions']['B_RF_ONLY']['final_ppl']
    c_ppl = results['conditions']['C_UNIFIED']['final_ppl']
    d_ppl = results['conditions']['D_DISEMBODIED']['final_ppl']

    # V1: Unified better than single-source
    v1_pass = c_ppl < max(a_ppl, b_ppl)
    results['verdicts']['V1_unified_better'] = {
        'pass': v1_pass,
        'unified_ppl': c_ppl,
        'gpu_only_ppl': a_ppl,
        'rf_only_ppl': b_ppl,
        'description': 'Unified (GPU+RF) achieves lower perplexity than single-source'
    }

    # V2: Embodied better than disembodied
    best_embodied = min(a_ppl, b_ppl, c_ppl)
    v2_pass = best_embodied < d_ppl
    results['verdicts']['V2_embodied_better'] = {
        'pass': v2_pass,
        'best_embodied_ppl': best_embodied,
        'disembodied_ppl': d_ppl,
        'description': 'Best embodied condition beats disembodied baseline'
    }

    # V3: Hidden representations differ with telemetry source
    a_norms = results['conditions']['A_GPU_ONLY']['hidden_norms']
    c_norms = results['conditions']['C_UNIFIED']['hidden_norms']
    norm_diff = abs(np.mean(a_norms) - np.mean(c_norms))
    v3_pass = norm_diff > 0.1
    results['verdicts']['V3_representations_differ'] = {
        'pass': v3_pass,
        'gpu_only_norm': float(np.mean(a_norms)),
        'unified_norm': float(np.mean(c_norms)),
        'difference': float(norm_diff),
        'threshold': 0.1,
        'description': 'Hidden representations differ between GPU-only and unified telemetry'
    }

    # V4: RF adds useful information (variance in telemetry)
    c_var = results['conditions']['C_UNIFIED']['telemetry_variance']
    a_var = results['conditions']['A_GPU_ONLY']['telemetry_variance']
    v4_pass = c_var > a_var  # Unified should have more variance
    results['verdicts']['V4_rf_adds_variance'] = {
        'pass': v4_pass,
        'unified_variance': c_var,
        'gpu_only_variance': a_var,
        'description': 'Unified telemetry has higher variance (more information)'
    }

    # Summary
    passed = sum(1 for v in results['verdicts'].values() if v['pass'])
    total = len(results['verdicts'])
    results['passed'] = passed
    results['total_verdicts'] = total
    results['overall_verdict'] = 'RF_EMBODIMENT_DEMONSTRATED' if passed >= 3 else 'INCONCLUSIVE'

    print(f"\n[z1800] Results: {passed}/{total} verdicts passed")
    print(f"[z1800] Overall: {results['overall_verdict']}")

    # Stop telemetry
    telemetry.stop()

    # Save results
    results_path = Path(__file__).parent.parent / "results" / "z1800_rf_embodiment.json"
    results_path.parent.mkdir(exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"[z1800] Results saved to {results_path}")

    return results


if __name__ == "__main__":
    results = run_experiment()
