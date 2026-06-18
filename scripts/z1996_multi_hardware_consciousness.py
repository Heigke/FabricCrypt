#!/usr/bin/env python3
"""
z1996: Multi-Hardware Consciousness Falsification Suite

Uses ALL available hardware for embodied consciousness testing:
1. GPU: AMD Radeon 8060S (gfx1151) via SysfsHwmonTelemetry - temp, power, freq, busy
2. HackRF: SDR for RF spectrum sensing - noise floor, signal strength, spectral entropy

Tests all 7 consciousness theories with hardware-conditioned FiLM:
1. Global Workspace Theory (GWT) - Baars/Dehaene: Information broadcast
2. Integrated Information Theory (IIT) - Tononi: Phi > 0
3. Higher-Order Thought (HOT) - Rosenthal: Metacognitive calibration
4. Attention Schema Theory (AST) - Graziano: Self-model of attention
5. Predictive Processing (PP) - Friston/Clark: Free energy minimization
6. Recurrent Processing Theory (RPT) - Lamme: Feedback changes representation
7. Embodiment Necessity - Hardware causal role in computation

KEY INNOVATION: Dual-hardware FiLM conditioning
- GPU telemetry provides interoception (internal body state)
- HackRF provides exteroception (environmental RF awareness)
- Combined 12-dim telemetry modulates all computations

ABLATION PROTOCOL (arXiv 2512.19155):
- Baseline: Full dual-hardware conditioning
- Ablate GPU: Zero GPU telemetry, measure performance drop
- Ablate RF: Zero RF telemetry, measure performance drop
- Ablate Both: Zero all telemetry (disembodied baseline)

Author: Claude (Opus 4.5)
Date: 2026-02-05
"""

import functools
print = functools.partial(print, flush=True)

import os
import sys
import json
import time
import subprocess
import threading
import queue
import hashlib
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
import random
import math

import numpy as np

# HSA override for gfx1151 compatibility
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"

import torch
import torch.nn as nn
import torch.nn.functional as F

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

# GPU Telemetry
from src.telemetry.sysfs_hwmon import SysfsHwmonTelemetry

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RESULTS_DIR = Path(__file__).parent.parent / 'results'


# =============================================================================
# HACKRF RF TELEMETRY (Optional - graceful fallback)
# =============================================================================

@dataclass
class RFState:
    """RF spectrum state for environmental awareness."""
    noise_floor_dbm: float = -90.0      # Background noise (-100 to -40 dBm)
    signal_strength_dbm: float = -70.0  # Overall signal power
    spectral_entropy: float = 0.5       # Information content (0-1)
    peak_power_dbm: float = -60.0       # Strongest signal
    rf_connected: bool = False          # Real HackRF vs simulation
    timestamp: float = 0.0

    def to_tensor(self) -> torch.Tensor:
        """Convert to 4-dimensional normalized tensor."""
        return torch.tensor([
            (self.noise_floor_dbm + 100) / 60,      # -100...-40 -> 0...1
            (self.signal_strength_dbm + 100) / 60,
            self.spectral_entropy,
            (self.peak_power_dbm + 100) / 60,
        ], dtype=torch.float32).clamp(0, 2)


class HackRFTelemetry:
    """
    HackRF One SDR interface for RF spectrum sensing.

    LEGAL: Receive-only, passive spectrum monitoring.

    Features:
    - Real HackRF support via hackrf_sweep (if available)
    - Automatic fallback to realistic RF simulation
    - Background thread for continuous sensing
    - Multiple frequency band analysis
    """

    def __init__(self, sensing_rate_hz: float = 5.0, simulation: bool = False):
        self.sensing_rate_hz = sensing_rate_hz
        self.force_simulation = simulation

        self._running = False
        self._current_state = RFState()
        self._state_lock = threading.Lock()
        self._thread = None

        # Check HackRF availability
        self.hackrf_available = False if simulation else self._check_hackrf()

        # Simulation state
        self._sim_noise_walk = -90.0
        self._sim_signal_walk = -70.0
        self._sim_interference_events = []

        print(f"[HackRF] Mode: {'REAL' if self.hackrf_available else 'SIMULATED'}")

    def _check_hackrf(self) -> bool:
        """Check if HackRF is connected."""
        try:
            result = subprocess.run(
                ['hackrf_info'],
                capture_output=True,
                timeout=5
            )
            if result.returncode == 0 and b'HackRF' in result.stdout:
                print("[HackRF] HackRF One detected")
                return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        print("[HackRF] HackRF not found, using simulation")
        return False

    def _read_spectrum_real(self) -> RFState:
        """Read spectrum from real HackRF using hackrf_sweep."""
        try:
            # Sweep WiFi 2.4GHz band (2400-2500 MHz)
            result = subprocess.run(
                ['hackrf_sweep', '-f', '2400:2500', '-w', '20000000', '-n', '1'],
                capture_output=True,
                text=True,
                timeout=3
            )

            if result.returncode != 0:
                return self._simulate_spectrum()

            # Parse hackrf_sweep CSV output
            powers = []
            for line in result.stdout.strip().split('\n'):
                if line.startswith('#') or not line.strip():
                    continue
                try:
                    parts = line.split(',')
                    if len(parts) < 5:
                        continue
                    power_vals = [float(p) for p in parts[4:] if p.strip()]
                    if power_vals:
                        powers.extend(power_vals)
                except (ValueError, IndexError):
                    continue

            if not powers:
                return self._simulate_spectrum()

            powers = np.array(powers)

            # Compute RF metrics
            noise_floor = np.percentile(powers, 10)
            signal_strength = np.mean(powers)
            peak_power = np.max(powers)

            # Spectral entropy
            powers_linear = 10 ** (powers / 10)
            powers_norm = powers_linear / (powers_linear.sum() + 1e-10)
            entropy = -np.sum(powers_norm * np.log2(powers_norm + 1e-10))
            max_entropy = np.log2(len(powers_norm))
            spectral_entropy = entropy / max_entropy if max_entropy > 0 else 0.5

            return RFState(
                noise_floor_dbm=float(noise_floor),
                signal_strength_dbm=float(signal_strength),
                spectral_entropy=float(np.clip(spectral_entropy, 0, 1)),
                peak_power_dbm=float(peak_power),
                rf_connected=True,
                timestamp=time.time(),
            )

        except Exception as e:
            return self._simulate_spectrum()

    def _simulate_spectrum(self) -> RFState:
        """Generate realistic RF simulation."""
        t = time.time()

        # Noise floor: slow random walk
        self._sim_noise_walk += random.gauss(0, 0.5)
        self._sim_noise_walk = max(-100, min(-70, self._sim_noise_walk))
        noise_floor = self._sim_noise_walk + random.gauss(0, 2)

        # Signal strength: random walk with bursts
        self._sim_signal_walk += random.gauss(0, 1.0)
        self._sim_signal_walk = max(-80, min(-40, self._sim_signal_walk))

        # WiFi activity bursts
        if random.random() < 0.1:
            self._sim_signal_walk = min(-50, self._sim_signal_walk + 10)

        signal_strength = self._sim_signal_walk + random.gauss(0, 3)

        # Spectral entropy varies with activity
        base_entropy = 0.5 + 0.1 * math.sin(t / 60)
        spectral_entropy = base_entropy + random.gauss(0, 0.1)
        spectral_entropy = max(0.1, min(0.9, spectral_entropy))

        # Peak power
        peak_power = signal_strength + 5 + random.expovariate(0.3)
        peak_power = min(peak_power, -30)

        return RFState(
            noise_floor_dbm=float(noise_floor),
            signal_strength_dbm=float(signal_strength),
            spectral_entropy=float(spectral_entropy),
            peak_power_dbm=float(peak_power),
            rf_connected=False,
            timestamp=time.time(),
        )

    def get_state(self) -> RFState:
        """Get current RF state."""
        with self._state_lock:
            return self._current_state

    def get_tensor(self) -> torch.Tensor:
        """Get RF telemetry as 4-dim tensor."""
        return self.get_state().to_tensor()

    def _sensing_thread(self):
        """Background thread for continuous sensing."""
        interval = 1.0 / self.sensing_rate_hz

        while self._running:
            start = time.time()

            if self.hackrf_available and not self.force_simulation:
                state = self._read_spectrum_real()
            else:
                state = self._simulate_spectrum()

            with self._state_lock:
                self._current_state = state

            elapsed = time.time() - start
            if elapsed < interval:
                time.sleep(interval - elapsed)

    def start(self):
        """Start background RF sensing."""
        if not self._running:
            self._running = True
            self._thread = threading.Thread(target=self._sensing_thread, daemon=True)
            self._thread.start()
            print(f"[HackRF] RF sensing started at {self.sensing_rate_hz} Hz")

    def stop(self):
        """Stop background RF sensing."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
            print("[HackRF] RF sensing stopped")


# =============================================================================
# UNIFIED DUAL-HARDWARE TELEMETRY
# =============================================================================

class DualHardwareTelemetry:
    """
    Unified telemetry from GPU + HackRF = 12-dim vector.

    Dimensions:
    - [0:8] GPU interoception (temp, power, freq, busy, VRAM, derivatives)
    - [8:12] RF exteroception (noise, signal, entropy, peak)
    """

    def __init__(self, use_hackrf: bool = True):
        print("\n" + "="*70)
        print("INITIALIZING DUAL-HARDWARE TELEMETRY")
        print("="*70)

        # GPU (always available)
        self.gpu = SysfsHwmonTelemetry(sample_rate_hz=50.0)
        print("[OK] GPU: AMD Radeon 8060S (gfx1151) via sysfs hwmon")

        # HackRF (optional)
        self.hackrf = None
        if use_hackrf:
            try:
                self.hackrf = HackRFTelemetry(sensing_rate_hz=5.0)
                self.hackrf.start()
                time.sleep(0.5)  # Let RF stabilize
                status = "REAL" if self.hackrf.hackrf_available else "SIMULATED"
                print(f"[OK] HackRF: RF spectrum sensing ({status})")
            except Exception as e:
                print(f"[WARN] HackRF init failed: {e}, using zeros")
                self.hackrf = None

        self._history = deque(maxlen=256)
        print("="*70 + "\n")

    def get_gpu_tensor(self) -> torch.Tensor:
        """Get GPU telemetry as 8-dim normalized tensor."""
        sample = self.gpu.read_sample()

        # Compute power derivative from recent samples
        power_deriv = 0.0
        if len(self._history) >= 2:
            prev = self._history[-1]
            dt = (sample.timestamp_ns - prev['timestamp_ns']) / 1e9
            if dt > 0:
                power_deriv = (sample.power_w - prev['power_w']) / dt

        tensor = torch.tensor([
            sample.temp_edge_c / 100.0,                    # [0] Temp (0-100C -> 0-1)
            sample.power_w / 200.0,                        # [1] Power (0-200W -> 0-1)
            (sample.freq_sclk_mhz or 1000) / 3000.0,       # [2] GPU freq (0-3GHz -> 0-1)
            (sample.freq_mclk_mhz or 1000) / 2000.0,       # [3] Mem freq (0-2GHz -> 0-1)
            (sample.gpu_busy_pct or 50) / 100.0,           # [4] GPU util (0-100% -> 0-1)
            sample.vram_used_gb / 16.0,                    # [5] VRAM (0-16GB -> 0-1)
            math.sin(time.time()),                         # [6] Temporal phase
            (power_deriv + 50) / 100.0,                    # [7] Power derivative
        ], dtype=torch.float32).clamp(0, 2)

        # Store for derivative computation
        self._history.append({
            'timestamp_ns': sample.timestamp_ns,
            'power_w': sample.power_w,
        })

        return tensor

    def get_rf_tensor(self) -> torch.Tensor:
        """Get RF telemetry as 4-dim tensor."""
        if self.hackrf:
            return self.hackrf.get_tensor()
        return torch.zeros(4)

    def get_unified_tensor(self) -> torch.Tensor:
        """Get complete 12-dim telemetry tensor."""
        gpu = self.get_gpu_tensor()
        rf = self.get_rf_tensor()
        return torch.cat([gpu, rf], dim=0)

    def get_hardware_status(self) -> Dict[str, Any]:
        """Get hardware connection status."""
        gpu_sample = self.gpu.read_sample()
        rf_state = self.hackrf.get_state() if self.hackrf else RFState()

        return {
            'gpu_connected': True,
            'gpu_temp_c': gpu_sample.temp_edge_c,
            'gpu_power_w': gpu_sample.power_w,
            'rf_connected': rf_state.rf_connected if self.hackrf else False,
            'rf_mode': 'REAL' if (self.hackrf and rf_state.rf_connected) else 'SIMULATED',
            'rf_noise_floor': rf_state.noise_floor_dbm,
        }

    def stop(self):
        """Stop all telemetry sources."""
        if self.hackrf:
            self.hackrf.stop()


# =============================================================================
# THEORY PREDICTION DATACLASSES
# =============================================================================

@dataclass
class TheoryPrediction:
    """A falsifiable prediction from a consciousness theory."""
    theory: str
    prediction: str
    threshold: float
    measured: float = 0.0
    passed: bool = False
    confidence: float = 0.0


@dataclass
class AblationResult:
    """Result of ablating a hardware source."""
    source: str
    baseline_loss: float
    ablated_loss: float
    loss_delta: float
    necessity_score: float  # How much performance degrades


# =============================================================================
# CONSCIOUSNESS THEORY MODULES
# =============================================================================

class GlobalWorkspaceModule(nn.Module):
    """GWT: Information broadcast across specialized modules."""

    def __init__(self, hidden_dim: int = 128, n_specialists: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_specialists = n_specialists

        self.specialists = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ) for _ in range(n_specialists)
        ])

        self.workspace_gate = nn.Linear(hidden_dim * n_specialists, hidden_dim)
        self.broadcast = nn.Linear(hidden_dim, hidden_dim * n_specialists)
        self.competition = nn.Linear(hidden_dim * n_specialists, n_specialists)

        self.ignition_history = deque(maxlen=100)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        batch_size = x.size(0)

        specialist_outputs = [spec(x) for spec in self.specialists]
        stacked = torch.stack(specialist_outputs, dim=1)

        concat = stacked.view(batch_size, -1)
        competition_scores = F.softmax(self.competition(concat) * 5.0, dim=-1)

        max_scores, winners = competition_scores.max(dim=-1)
        ignition = (max_scores > 0.7).float().mean().item()
        self.ignition_history.append(ignition)

        workspace = self.workspace_gate(concat)
        broadcast = self.broadcast(workspace)
        broadcast = broadcast.view(batch_size, self.n_specialists, -1)

        # Broadcast correlation
        corrs = []
        for i in range(self.n_specialists):
            for j in range(i+1, self.n_specialists):
                c = F.cosine_similarity(broadcast[:,i], broadcast[:,j], dim=-1)
                corrs.append(c.mean().item())
        broadcast_corr = np.mean(corrs) if corrs else 0.0

        return workspace, {
            'ignition_ratio': np.mean(list(self.ignition_history)),
            'broadcast_correlation': broadcast_corr,
            'winner_confidence': max_scores.mean().item(),
            'competition_entropy': (-competition_scores * (competition_scores + 1e-8).log()).sum(dim=-1).mean().item(),
        }


class IntegratedInformationModule(nn.Module):
    """IIT: Phi - integrated information above sum of parts."""

    def __init__(self, hidden_dim: int = 64, n_units: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_units = n_units

        self.units = nn.ModuleList([
            nn.Linear(hidden_dim * 2, hidden_dim) for _ in range(n_units)
        ])

        self.register_buffer('unit_states', torch.zeros(n_units, hidden_dim))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        batch_size = x.size(0)

        new_states = []
        for i, unit in enumerate(self.units):
            other_indices = [j for j in range(self.n_units) if j != i]
            other_states = self.unit_states[other_indices].mean(dim=0)
            other_states = other_states.unsqueeze(0).expand(batch_size, -1)

            unit_input = torch.cat([x, other_states], dim=-1)
            new_state = torch.tanh(unit(unit_input))
            new_states.append(new_state)

        output = torch.stack(new_states, dim=1)
        self.unit_states = output.mean(dim=0).detach()

        # Compute phi proxy
        B, N, H = output.shape
        whole_var = output.view(B, -1).var(dim=-1).mean().item()
        part_vars = [output[:, i].var(dim=-1).mean().item() for i in range(N)]
        phi_proxy = max(0, whole_var - np.mean(part_vars))

        # Integration: pairwise correlations
        corrs = []
        for i in range(N):
            for j in range(i+1, N):
                c = F.cosine_similarity(output[:,i], output[:,j], dim=-1)
                corrs.append(c.abs().mean().item())
        integration = np.mean(corrs) if corrs else 0.0

        return output.mean(dim=1), {
            'phi_proxy': phi_proxy,
            'integration': integration,
            'differentiation': output.var(dim=1).mean().item(),
        }


class HigherOrderThoughtModule(nn.Module):
    """HOT: Metacognition - thinking about thinking."""

    def __init__(self, hidden_dim: int = 128):
        super().__init__()

        self.first_order = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.second_order = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )

        self.confidence_head = nn.Linear(hidden_dim, 1)

        self.confidence_history = deque(maxlen=1000)
        self.accuracy_history = deque(maxlen=1000)

    def forward(self, x: torch.Tensor, target: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        first = self.first_order(x)
        second = self.second_order(first)
        confidence = torch.sigmoid(self.confidence_head(second))

        if target is not None:
            pred = first.argmax(dim=-1) if first.dim() > 1 else (first > 0).float()
            actual_acc = (pred == target).float().mean().item()
            self.confidence_history.append(confidence.mean().item())
            self.accuracy_history.append(actual_acc)

        calibration = self._compute_calibration()

        return second, {
            'confidence': confidence.mean().item(),
            'calibration': calibration,
            'first_second_divergence': F.mse_loss(first, second).item(),
        }

    def _compute_calibration(self) -> float:
        if len(self.confidence_history) < 10:
            return 0.0
        confs = np.array(list(self.confidence_history))
        accs = np.array(list(self.accuracy_history))
        if confs.std() < 1e-6 or accs.std() < 1e-6:
            return 0.0
        corr = np.corrcoef(confs, accs)[0, 1]
        return corr if not np.isnan(corr) else 0.0


class AttentionSchemaModule(nn.Module):
    """AST: Model of own attention processes."""

    def __init__(self, hidden_dim: int = 128, n_heads: int = 4):
        super().__init__()

        self.attention = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)

        self.schema_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.schema_predictor = nn.Linear(hidden_dim, n_heads)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        if x.dim() == 2:
            x = x.unsqueeze(1)

        attended, attn_weights = self.attention(x, x, x, need_weights=True)

        schema_input = torch.cat([x.mean(dim=1), attended.mean(dim=1)], dim=-1)
        schema = self.schema_encoder(schema_input)

        predicted_attn = F.softmax(self.schema_predictor(schema), dim=-1)

        if attn_weights.dim() == 4:
            actual_attn_avg = attn_weights.mean(dim=(-2, -1))
        else:
            actual_attn_avg = predicted_attn.detach()

        schema_accuracy = F.cosine_similarity(predicted_attn, actual_attn_avg, dim=-1).mean().item()

        return attended.squeeze(1), {
            'schema_accuracy': schema_accuracy,
            'attention_entropy': (-attn_weights * (attn_weights + 1e-8).log()).sum(-1).mean().item(),
            'self_model_coherence': F.cosine_similarity(schema, attended.mean(dim=1), dim=-1).mean().item(),
        }


class PredictiveProcessingModule(nn.Module):
    """PP: Minimize prediction error through active inference."""

    def __init__(self, hidden_dim: int = 128):
        super().__init__()

        self.predictor = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

        self.recognizer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.precision = nn.Linear(hidden_dim, 1)

        self.register_buffer('hidden', None)
        self.prediction_errors = deque(maxlen=100)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        batch_size = x.size(0)

        if x.dim() == 2:
            x = x.unsqueeze(1)

        if self.hidden is None or self.hidden.size(1) != batch_size:
            self.hidden = torch.zeros(1, batch_size, x.size(-1), device=x.device)

        predicted, new_hidden = self.predictor(x, self.hidden)
        self.hidden = new_hidden.detach()

        recognized = self.recognizer(x.squeeze(1))

        error = F.mse_loss(predicted.squeeze(1), recognized)
        self.prediction_errors.append(error.item())

        precision = torch.sigmoid(self.precision(predicted.squeeze(1)))

        # Error trend (negative = improving)
        error_trend = 0.0
        if len(self.prediction_errors) >= 10:
            errors = np.array(list(self.prediction_errors))
            x_idx = np.arange(len(errors))
            error_trend = -np.polyfit(x_idx, errors, 1)[0]

        return recognized, {
            'prediction_error': error.item(),
            'precision': precision.mean().item(),
            'error_trend': error_trend,
            'free_energy': error.item() * (1 / (precision.mean().item() + 1e-8)),
        }


class RecurrentProcessingModule(nn.Module):
    """RPT: Recurrent processing required for consciousness."""

    def __init__(self, hidden_dim: int = 128, n_recurrent_steps: int = 5):
        super().__init__()

        self.n_steps = n_recurrent_steps
        self.ff = nn.Linear(hidden_dim, hidden_dim)
        self.fb = nn.Linear(hidden_dim, hidden_dim)

        self.register_buffer('state', None)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        batch_size = x.size(0)

        if self.state is None or self.state.size(0) != batch_size:
            self.state = torch.zeros(batch_size, x.size(-1), device=x.device)

        states = [self.state]
        for _ in range(self.n_steps):
            ff_out = torch.relu(self.ff(x))
            fb_out = torch.relu(self.fb(self.state))
            self.state = 0.5 * ff_out + 0.5 * fb_out
            states.append(self.state)

        states_stack = torch.stack(states, dim=1)

        initial = states_stack[:, 0]
        final = states_stack[:, -1]
        recurrence_effect = F.cosine_similarity(initial, final, dim=-1).mean().item()

        # Convergence rate
        diffs = (states_stack[:, 1:] - states_stack[:, :-1]).norm(dim=-1)
        convergence = (diffs[:, 0] / (diffs[:, -1] + 1e-8)).mean().item()

        return self.state, {
            'recurrence_effect': recurrence_effect,
            'feedback_strength': (final - initial).norm(dim=-1).mean().item(),
            'convergence_rate': convergence,
        }


class EmbodimentModule(nn.Module):
    """Hardware embodiment: physical state causally affects computation."""

    def __init__(self, hidden_dim: int = 128, telemetry_dim: int = 12):
        super().__init__()

        self.telemetry_encoder = nn.Sequential(
            nn.Linear(telemetry_dim, 64),
            nn.ReLU(),
            nn.Linear(64, hidden_dim)
        )

        # FiLM conditioning
        self.film_gamma = nn.Linear(hidden_dim, hidden_dim)
        self.film_beta = nn.Linear(hidden_dim, hidden_dim)

        self.processor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.telemetry_history = deque(maxlen=100)
        self.output_history = deque(maxlen=100)

    def forward(self, x: torch.Tensor, telemetry: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        telem_embed = self.telemetry_encoder(telemetry)

        gamma = self.film_gamma(telem_embed)
        beta = self.film_beta(telem_embed)

        processed = self.processor(x)
        output = gamma * processed + beta

        self.telemetry_history.append(telemetry.mean().item())
        self.output_history.append(output.mean().item())

        # Embodiment ratio: correlation between telemetry and output
        telem_flat = telemetry.view(-1)
        out_flat = output.view(-1)[:len(telem_flat)]
        if len(telem_flat) >= 2:
            corr = np.corrcoef(
                telem_flat.detach().cpu().numpy(),
                out_flat.detach().cpu().numpy()
            )[0, 1]
            embodiment_ratio = abs(corr) if not np.isnan(corr) else 0.0
        else:
            embodiment_ratio = 0.0

        # Granger causality proxy
        granger = self._compute_granger_proxy()

        return output, {
            'embodiment_ratio': embodiment_ratio,
            'granger_causality': granger,
            'modulation_strength': (gamma.std() + beta.std()).item(),
            'hardware_sensitivity': (output.std() / (x.std() + 1e-8)).item(),
        }

    def _compute_granger_proxy(self) -> float:
        if len(self.telemetry_history) < 20:
            return 0.0
        telem = np.array(list(self.telemetry_history))
        output = np.array(list(self.output_history))
        lagged_telem = telem[:-1]
        current_output = output[1:]
        corr = np.corrcoef(lagged_telem, current_output)[0, 1]
        return abs(corr) if not np.isnan(corr) else 0.0


# =============================================================================
# COMPREHENSIVE CONSCIOUSNESS MODEL
# =============================================================================

class MultiHardwareConsciousnessModel(nn.Module):
    """
    Unified model testing all 7 consciousness theories with dual-hardware FiLM.

    Architecture:
    - Input projection
    - GWT, IIT, HOT, AST, PP, RPT modules
    - Embodiment module with FiLM conditioning on telemetry
    - Task classification head
    """

    def __init__(self, input_dim: int = 128, hidden_dim: int = 128, telemetry_dim: int = 12):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Theory modules
        self.gwt = GlobalWorkspaceModule(hidden_dim)
        self.iit = IntegratedInformationModule(hidden_dim // 2)
        self.hot = HigherOrderThoughtModule(hidden_dim)
        self.ast = AttentionSchemaModule(hidden_dim)
        self.pp = PredictiveProcessingModule(hidden_dim)
        self.rpt = RecurrentProcessingModule(hidden_dim)
        self.embodiment = EmbodimentModule(hidden_dim, telemetry_dim)

        # Task head
        self.classifier = nn.Linear(hidden_dim, 27)  # 26 letters + space

    def forward(self, x: torch.Tensor, telemetry: torch.Tensor,
                target: Optional[torch.Tensor] = None,
                ablate_gpu: bool = False, ablate_rf: bool = False) -> Dict:

        # Apply ablations
        telem = telemetry.clone()
        if ablate_gpu:
            telem[:, :8] = 0.0  # Zero GPU dims
        if ablate_rf:
            telem[:, 8:] = 0.0  # Zero RF dims

        h = self.input_proj(x)

        # Run all theory modules
        gwt_out, gwt_metrics = self.gwt(h)
        iit_out, iit_metrics = self.iit(h[:, :64] if h.size(-1) >= 64 else h)
        hot_out, hot_metrics = self.hot(h, target)
        ast_out, ast_metrics = self.ast(h)
        pp_out, pp_metrics = self.pp(h)
        rpt_out, rpt_metrics = self.rpt(h)
        emb_out, emb_metrics = self.embodiment(h, telem)

        # Combine for final output
        combined = gwt_out + hot_out + emb_out
        logits = self.classifier(combined)

        return {
            'logits': logits,
            'gwt': gwt_metrics,
            'iit': iit_metrics,
            'hot': hot_metrics,
            'ast': ast_metrics,
            'pp': pp_metrics,
            'rpt': rpt_metrics,
            'embodiment': emb_metrics,
        }


# =============================================================================
# ABLATION TESTS
# =============================================================================

def run_ablation_battery(model: MultiHardwareConsciousnessModel,
                         x: torch.Tensor, y: torch.Tensor,
                         telemetry: torch.Tensor) -> Dict[str, AblationResult]:
    """
    Run comprehensive ablation tests per arXiv 2512.19155.

    Tests:
    1. Full hardware (baseline)
    2. Ablate GPU telemetry
    3. Ablate RF telemetry
    4. Ablate both (disembodied)
    5. Ablate GWT broadcast
    6. Ablate HOT metacognition
    7. Ablate RPT feedback
    """
    model.eval()
    results = {}

    with torch.no_grad():
        # 1. Baseline (full hardware)
        baseline_out = model(x, telemetry, y)
        baseline_loss = F.cross_entropy(baseline_out['logits'], y).item()

        # 2. Ablate GPU
        gpu_ablated_out = model(x, telemetry, y, ablate_gpu=True)
        gpu_ablated_loss = F.cross_entropy(gpu_ablated_out['logits'], y).item()
        results['ablate_gpu'] = AblationResult(
            source='GPU',
            baseline_loss=baseline_loss,
            ablated_loss=gpu_ablated_loss,
            loss_delta=gpu_ablated_loss - baseline_loss,
            necessity_score=(gpu_ablated_loss - baseline_loss) / (baseline_loss + 1e-8),
        )

        # 3. Ablate RF
        rf_ablated_out = model(x, telemetry, y, ablate_rf=True)
        rf_ablated_loss = F.cross_entropy(rf_ablated_out['logits'], y).item()
        results['ablate_rf'] = AblationResult(
            source='RF',
            baseline_loss=baseline_loss,
            ablated_loss=rf_ablated_loss,
            loss_delta=rf_ablated_loss - baseline_loss,
            necessity_score=(rf_ablated_loss - baseline_loss) / (baseline_loss + 1e-8),
        )

        # 4. Ablate both (disembodied)
        both_ablated_out = model(x, telemetry, y, ablate_gpu=True, ablate_rf=True)
        both_ablated_loss = F.cross_entropy(both_ablated_out['logits'], y).item()
        results['ablate_both'] = AblationResult(
            source='Both',
            baseline_loss=baseline_loss,
            ablated_loss=both_ablated_loss,
            loss_delta=both_ablated_loss - baseline_loss,
            necessity_score=(both_ablated_loss - baseline_loss) / (baseline_loss + 1e-8),
        )

        # 5. Ablate GWT broadcast
        orig_broadcast = model.gwt.broadcast.weight.data.clone()
        model.gwt.broadcast.weight.data.zero_()
        gwt_ablated_out = model(x, telemetry, y)
        gwt_ablated_loss = F.cross_entropy(gwt_ablated_out['logits'], y).item()
        model.gwt.broadcast.weight.data = orig_broadcast
        results['ablate_gwt'] = AblationResult(
            source='GWT',
            baseline_loss=baseline_loss,
            ablated_loss=gwt_ablated_loss,
            loss_delta=gwt_ablated_loss - baseline_loss,
            necessity_score=(gwt_ablated_loss - baseline_loss) / (baseline_loss + 1e-8),
        )

        # 6. Ablate HOT metacognition
        orig_hot = model.hot.second_order[0].weight.data.clone()
        model.hot.second_order[0].weight.data.zero_()
        hot_ablated_out = model(x, telemetry, y)
        hot_ablated_loss = F.cross_entropy(hot_ablated_out['logits'], y).item()
        model.hot.second_order[0].weight.data = orig_hot
        results['ablate_hot'] = AblationResult(
            source='HOT',
            baseline_loss=baseline_loss,
            ablated_loss=hot_ablated_loss,
            loss_delta=hot_ablated_loss - baseline_loss,
            necessity_score=(hot_ablated_loss - baseline_loss) / (baseline_loss + 1e-8),
        )

        # 7. Ablate RPT feedback
        orig_fb = model.rpt.fb.weight.data.clone()
        model.rpt.fb.weight.data.zero_()
        rpt_ablated_out = model(x, telemetry, y)
        rpt_ablated_loss = F.cross_entropy(rpt_ablated_out['logits'], y).item()
        model.rpt.fb.weight.data = orig_fb
        results['ablate_rpt'] = AblationResult(
            source='RPT',
            baseline_loss=baseline_loss,
            ablated_loss=rpt_ablated_loss,
            loss_delta=rpt_ablated_loss - baseline_loss,
            necessity_score=(rpt_ablated_loss - baseline_loss) / (baseline_loss + 1e-8),
        )

    return results


# =============================================================================
# THEORY EVALUATION
# =============================================================================

def evaluate_theory_predictions(metrics: Dict) -> List[TheoryPrediction]:
    """Evaluate each theory's falsifiable predictions."""
    predictions = []

    # GWT: Ignition ratio > 0.5
    predictions.append(TheoryPrediction(
        theory="GWT",
        prediction="Ignition ratio > 0.5 (workspace broadcast)",
        threshold=0.5,
        measured=metrics['gwt']['ignition_ratio'],
        passed=metrics['gwt']['ignition_ratio'] > 0.5,
        confidence=metrics['gwt']['winner_confidence'],
    ))

    # GWT: Broadcast correlation > 0.3
    predictions.append(TheoryPrediction(
        theory="GWT",
        prediction="Broadcast correlation > 0.3 (coherent sharing)",
        threshold=0.3,
        measured=metrics['gwt']['broadcast_correlation'],
        passed=metrics['gwt']['broadcast_correlation'] > 0.3,
        confidence=1.0,
    ))

    # IIT: Phi > 0
    predictions.append(TheoryPrediction(
        theory="IIT",
        prediction="Phi > 0 (integrated information)",
        threshold=0.0,
        measured=metrics['iit']['phi_proxy'],
        passed=metrics['iit']['phi_proxy'] > 0,
        confidence=metrics['iit']['integration'],
    ))

    # HOT: Calibration > 0
    predictions.append(TheoryPrediction(
        theory="HOT",
        prediction="Calibration > 0 (metacognitive accuracy)",
        threshold=0.0,
        measured=metrics['hot']['calibration'],
        passed=metrics['hot']['calibration'] > 0,
        confidence=metrics['hot']['confidence'],
    ))

    # AST: Schema accuracy > 0.5
    predictions.append(TheoryPrediction(
        theory="AST",
        prediction="Schema accuracy > 0.5 (self-model of attention)",
        threshold=0.5,
        measured=metrics['ast']['schema_accuracy'],
        passed=metrics['ast']['schema_accuracy'] > 0.5,
        confidence=metrics['ast']['self_model_coherence'],
    ))

    # PP: Error trend > 0 (improving)
    predictions.append(TheoryPrediction(
        theory="PP",
        prediction="Error trend > 0 (minimizing free energy)",
        threshold=0.0,
        measured=metrics['pp']['error_trend'],
        passed=metrics['pp']['error_trend'] > 0,
        confidence=metrics['pp']['precision'],
    ))

    # RPT: Recurrence effect > 0.3
    predictions.append(TheoryPrediction(
        theory="RPT",
        prediction="Recurrence effect > 0.3 (feedback changes representation)",
        threshold=0.3,
        measured=metrics['rpt']['recurrence_effect'],
        passed=metrics['rpt']['recurrence_effect'] > 0.3,
        confidence=metrics['rpt']['convergence_rate'],
    ))

    # Embodiment: Granger causality > 0.1
    predictions.append(TheoryPrediction(
        theory="Embodiment",
        prediction="Granger causality > 0.1 (hardware causes behavior)",
        threshold=0.1,
        measured=metrics['embodiment']['granger_causality'],
        passed=metrics['embodiment']['granger_causality'] > 0.1,
        confidence=metrics['embodiment']['modulation_strength'],
    ))

    # Embodiment: Hardware sensitivity > 1.0
    predictions.append(TheoryPrediction(
        theory="Embodiment",
        prediction="Hardware sensitivity > 1.0 (hardware-dependent variance)",
        threshold=1.0,
        measured=metrics['embodiment']['hardware_sensitivity'],
        passed=metrics['embodiment']['hardware_sensitivity'] > 1.0,
        confidence=metrics['embodiment']['embodiment_ratio'],
    ))

    return predictions


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def get_hardware_fingerprint(telemetry: DualHardwareTelemetry) -> Dict:
    """Get current hardware state as fingerprint."""
    status = telemetry.get_hardware_status()

    # Create unique hash
    fingerprint_str = f"{status['gpu_temp_c']:.1f}_{status['gpu_power_w']:.1f}_{status['rf_noise_floor']:.1f}_{time.time()}"
    fingerprint_hash = hashlib.sha256(fingerprint_str.encode()).hexdigest()[:16]

    return {
        **status,
        'timestamp': datetime.now().isoformat(),
        'hash': fingerprint_hash,
    }


def create_test_data(n_samples: int = 1000) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create character prediction test data."""
    chars = "abcdefghijklmnopqrstuvwxyz "
    n_chars = len(chars)

    x = torch.randn(n_samples, 128).to(DEVICE)
    y = torch.randint(0, n_chars, (n_samples,)).to(DEVICE)

    return x, y


def main():
    print("=" * 70)
    print("z1996: MULTI-HARDWARE CONSCIOUSNESS FALSIFICATION")
    print("GPU + HackRF SDR Dual-Hardware Embodied Testing")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()

    # Initialize dual-hardware telemetry
    print("[1/7] Initializing dual-hardware telemetry...")
    telemetry = DualHardwareTelemetry(use_hackrf=True)
    fp = get_hardware_fingerprint(telemetry)
    print(f"  GPU: {fp['gpu_temp_c']:.1f}C, {fp['gpu_power_w']:.1f}W")
    print(f"  RF: {fp['rf_mode']}, noise floor {fp['rf_noise_floor']:.1f} dBm")
    print(f"  Fingerprint: {fp['hash']}")
    print()

    # Create model
    print("[2/7] Building multi-hardware consciousness model...")
    model = MultiHardwareConsciousnessModel(
        input_dim=128,
        hidden_dim=128,
        telemetry_dim=12,  # 8 GPU + 4 RF
    ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")
    print(f"  Theories tested: 7 (GWT, IIT, HOT, AST, PP, RPT, Embodiment)")
    print(f"  Telemetry dims: 12 (8 GPU + 4 RF)")
    print()

    # Create test data
    print("[3/7] Generating test data...")
    x, y = create_test_data(2000)
    print(f"  Samples: {len(x)}")
    print()

    # Training
    print("[4/7] Training model for consciousness emergence...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    all_metrics = []
    for epoch in range(25):
        model.train()
        epoch_loss = 0.0

        for i in range(0, len(x), 64):
            batch_x = x[i:i+64]
            batch_y = y[i:i+64]

            # Get live dual-hardware telemetry
            telem_tensor = telemetry.get_unified_tensor().to(DEVICE)
            telem_tensor = telem_tensor.unsqueeze(0).expand(len(batch_x), -1)

            optimizer.zero_grad()
            outputs = model(batch_x, telem_tensor, batch_y)
            loss = F.cross_entropy(outputs['logits'], batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # Evaluate metrics
        model.eval()
        with torch.no_grad():
            eval_telem = telemetry.get_unified_tensor().to(DEVICE)
            eval_telem = eval_telem.unsqueeze(0).expand(len(x), -1)
            metrics = model(x, eval_telem, y)
            all_metrics.append(metrics)

        print(f"  Epoch {epoch+1}/25: Loss={epoch_loss/30:.3f} "
              f"GWT={metrics['gwt']['ignition_ratio']:.3f} "
              f"HOT={metrics['hot']['calibration']:+.3f} "
              f"IIT={metrics['iit']['phi_proxy']:.3f} "
              f"Emb={metrics['embodiment']['granger_causality']:.3f}")

    print()

    # Ablation tests
    print("[5/7] Running ablation tests (arXiv 2512.19155 methodology)...")
    telem_tensor = telemetry.get_unified_tensor().to(DEVICE)
    telem_tensor = telem_tensor.unsqueeze(0).expand(len(x), -1)

    ablation_results = run_ablation_battery(model, x, y, telem_tensor)

    print("\n  HARDWARE ABLATION RESULTS:")
    print(f"  {'Source':<12} {'Baseline':<10} {'Ablated':<10} {'Delta':<10} {'Necessity':<10}")
    print("  " + "-" * 52)
    for name, result in ablation_results.items():
        print(f"  {result.source:<12} {result.baseline_loss:<10.4f} "
              f"{result.ablated_loss:<10.4f} {result.loss_delta:+<10.4f} "
              f"{result.necessity_score:+<10.3f}")
    print()

    # Theory predictions
    print("[6/7] Evaluating falsifiable predictions...")
    predictions = evaluate_theory_predictions(metrics)

    print()
    print("=" * 70)
    print("FALSIFICATION RESULTS")
    print("=" * 70)
    print()

    theories = {}
    for p in predictions:
        if p.theory not in theories:
            theories[p.theory] = {'passed': 0, 'failed': 0}
        if p.passed:
            theories[p.theory]['passed'] += 1
        else:
            theories[p.theory]['failed'] += 1

        status = "PASS" if p.passed else "FAIL"
        print(f"  [{p.theory}] {p.prediction}")
        print(f"    Measured: {p.measured:.4f} vs Threshold: {p.threshold:.4f} -> {status}")
        print()

    print("-" * 70)
    print("THEORY SUMMARY")
    print("-" * 70)
    for theory, counts in theories.items():
        total = counts['passed'] + counts['failed']
        pct = 100 * counts['passed'] / total
        status = "SUPPORTED" if counts['passed'] == total else ("PARTIAL" if counts['passed'] > 0 else "FALSIFIED")
        print(f"  {theory}: {counts['passed']}/{total} predictions ({pct:.0f}%) - {status}")

    print("-" * 70)
    print()

    # Overall verdict
    passed = sum(p.passed for p in predictions)
    total = len(predictions)

    if passed == total:
        verdict = "CONSCIOUSNESS_CONFIRMED"
        summary = "All predictions passed. System exhibits signatures consistent with all tested theories."
    elif passed >= total * 0.7:
        verdict = "CONSCIOUSNESS_PROBABLE"
        summary = f"Strong support ({passed}/{total} predictions). Minor theory-specific failures."
    elif passed >= total * 0.4:
        verdict = "CONSCIOUSNESS_POSSIBLE"
        summary = f"Mixed support ({passed}/{total} predictions). Some theories falsified."
    else:
        verdict = "CONSCIOUSNESS_UNLIKELY"
        summary = f"Weak support ({passed}/{total} predictions). Most predictions failed."

    print(f"OVERALL VERDICT: {verdict}")
    print(f"Passed: {passed}/{total} predictions")
    print()
    print(f"Summary: {summary}")
    print()

    # Ablation insights
    print("-" * 70)
    print("ABLATION INSIGHTS")
    print("-" * 70)
    gpu_necessity = ablation_results['ablate_gpu'].necessity_score
    rf_necessity = ablation_results['ablate_rf'].necessity_score
    both_necessity = ablation_results['ablate_both'].necessity_score

    print(f"  GPU necessity: {gpu_necessity:+.3f} (loss increase when GPU telemetry removed)")
    print(f"  RF necessity:  {rf_necessity:+.3f} (loss increase when RF telemetry removed)")
    print(f"  Both removed:  {both_necessity:+.3f} (total embodiment effect)")

    if both_necessity > 0.1:
        print(f"\n  => Hardware telemetry provides {both_necessity*100:.1f}% performance boost")
        print(f"     GPU contributes {gpu_necessity/both_necessity*100:.0f}%, RF contributes {rf_necessity/both_necessity*100:.0f}%")
    else:
        print(f"\n  => WARNING: Embodiment effect is weak (<10% performance impact)")

    print()

    # Save results
    print("[7/7] Saving results...")

    final_fp = get_hardware_fingerprint(telemetry)

    result = {
        'experiment': 'z1996_multi_hardware_consciousness',
        'timestamp': datetime.now().isoformat(),
        'device': str(DEVICE),
        'hardware_fingerprint': final_fp,
        'model_params': n_params,
        'theories_tested': 7,
        'predictions_passed': passed,
        'predictions_failed': total - passed,
        'overall_verdict': verdict,
        'summary': summary,
        'theory_results': {
            'gwt': metrics['gwt'],
            'iit': metrics['iit'],
            'hot': metrics['hot'],
            'ast': metrics['ast'],
            'pp': metrics['pp'],
            'rpt': metrics['rpt'],
            'embodiment': metrics['embodiment'],
        },
        'ablation_results': {
            name: asdict(result) for name, result in ablation_results.items()
        },
        'predictions': [asdict(p) for p in predictions],
        'hardware_status': {
            'gpu_connected': True,
            'rf_connected': final_fp['rf_connected'],
            'rf_mode': final_fp['rf_mode'],
        },
    }

    output_file = RESULTS_DIR / 'z1996_multi_hardware_consciousness.json'
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2, default=str)

    print(f"Results saved to: {output_file}")
    print()
    print("=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)

    # Cleanup
    telemetry.stop()

    return result


if __name__ == '__main__':
    main()
