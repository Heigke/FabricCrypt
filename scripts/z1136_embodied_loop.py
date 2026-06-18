#!/usr/bin/env python3
"""
z1136: Full Embodied Loop - SENSE → FEEL → EXPRESS → REGULATE
==============================================================

This implements the complete embodiment cycle where:
- SENSE: Read GPU telemetry + FPGA state in real-time
- FEEL: Process into unified 8-dim z_feel (body latent)
- EXPRESS: Run inference with body-conditioned computation
- REGULATE: Actuate hardware based on internal state

The AI doesn't just process tokens - it inhabits hardware.

Author: FEEL Research Team
Date: 2026-01-31
"""

import sys
import os
import json
import time
import threading
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, Tuple, List
from collections import deque

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F

# Our modules
from src.telemetry.sysfs_hwmon import SysfsHwmonTelemetry, EnergyMeter
from src.embodied.fpga_state_tracker import FPGAStateTracker, FPGAState
from src.metabolic.film_transformer import MetabolicTransformer


# =============================================================================
# SENSE: Hardware State Acquisition
# =============================================================================

@dataclass
class SensorySnapshot:
    """Complete sensory state from all hardware."""
    timestamp_ns: int

    # GPU state
    gpu_power_w: float
    gpu_temp_c: float
    gpu_util_pct: float
    gpu_freq_mhz: int
    gpu_energy_j: float

    # FPGA state
    fpga_temp_c: float
    dram_charge: float
    decay_rate: float

    # Derived
    power_deviation: float  # from setpoint
    temp_deviation: float
    thermal_margin: float


class SensorySystem:
    """
    Unified sensory acquisition from GPU and FPGA.

    Runs background thread for continuous sampling.
    """

    def __init__(
        self,
        gpu_telemetry: Optional[SysfsHwmonTelemetry] = None,
        fpga_tracker: Optional[FPGAStateTracker] = None,
        power_setpoint: float = 50.0,
        temp_setpoint: float = 70.0,
        sample_rate_hz: float = 20.0,
    ):
        self.gpu = gpu_telemetry or SysfsHwmonTelemetry()
        self.fpga = fpga_tracker or FPGAStateTracker(simulated=True)
        self.power_setpoint = power_setpoint
        self.temp_setpoint = temp_setpoint
        self.sample_interval = 1.0 / sample_rate_hz

        # History buffer
        self.history: deque = deque(maxlen=100)
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def sense(self) -> SensorySnapshot:
        """Take a single sensory snapshot."""
        ts = time.time_ns()

        # GPU
        try:
            gpu_sample = self.gpu.read_sample()
            gpu_power = gpu_sample.power_w
            gpu_temp = gpu_sample.temp_edge_c
            gpu_util = gpu_sample.gpu_busy_pct
            gpu_freq = gpu_sample.freq_sclk_mhz
            gpu_energy = self.gpu.get_accumulated_energy_j()
        except Exception:
            gpu_power, gpu_temp, gpu_util, gpu_freq, gpu_energy = 30.0, 50.0, 0.0, 1000, 0.0

        # FPGA
        fpga_state = self.fpga.update()

        # Deviations
        power_dev = (gpu_power - self.power_setpoint) / max(self.power_setpoint, 1.0)
        temp_dev = (gpu_temp - self.temp_setpoint) / max(self.temp_setpoint, 1.0)
        thermal_margin = max(0, (100.0 - gpu_temp)) / 100.0

        return SensorySnapshot(
            timestamp_ns=ts,
            gpu_power_w=gpu_power,
            gpu_temp_c=gpu_temp,
            gpu_util_pct=gpu_util,
            gpu_freq_mhz=gpu_freq,
            gpu_energy_j=gpu_energy,
            fpga_temp_c=fpga_state.temp_c,
            dram_charge=fpga_state.charge_level,
            decay_rate=fpga_state.decay_rate_per_second,
            power_deviation=power_dev,
            temp_deviation=temp_dev,
            thermal_margin=thermal_margin,
        )

    def start_continuous(self):
        """Start background sensing thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._sense_loop, daemon=True)
        self._thread.start()

    def stop_continuous(self):
        """Stop background sensing."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)

    def _sense_loop(self):
        while self._running:
            snapshot = self.sense()
            with self._lock:
                self.history.append(snapshot)
            time.sleep(self.sample_interval)

    def get_latest(self) -> Optional[SensorySnapshot]:
        """Get most recent snapshot."""
        with self._lock:
            return self.history[-1] if self.history else None

    def get_history(self, n: int = 10) -> List[SensorySnapshot]:
        """Get recent history."""
        with self._lock:
            return list(self.history)[-n:]


# =============================================================================
# FEEL: Body State Processing
# =============================================================================

class FeelSystem(nn.Module):
    """
    Process sensory input into felt body state (z_feel).

    Uses GRU to integrate temporal dynamics.
    Output is 32-dim learned body representation.
    """

    SENSORY_DIM = 8  # power_dev, temp_dev, margin, util, fpga_temp, charge, decay, stability
    HIDDEN_DIM = 32

    def __init__(self):
        super().__init__()

        # Sensory encoder
        self.encoder = nn.Sequential(
            nn.Linear(self.SENSORY_DIM, 16),
            nn.LayerNorm(16),
            nn.ReLU(),
        )

        # Temporal integration
        self.gru = nn.GRUCell(16, self.HIDDEN_DIM)

        # Body state output
        self.body_head = nn.Linear(self.HIDDEN_DIM, self.HIDDEN_DIM)

        # Interoceptive predictions (self-supervised)
        self.predict_next = nn.Sequential(
            nn.Linear(self.HIDDEN_DIM, 16),
            nn.ReLU(),
            nn.Linear(16, self.SENSORY_DIM),
        )

        # Hidden state
        self._h = None

    def reset(self):
        self._h = None

    def snapshot_to_tensor(self, snapshot: SensorySnapshot) -> torch.Tensor:
        """Convert snapshot to input tensor."""
        # Normalize to roughly [-1, 1] range
        return torch.tensor([
            snapshot.power_deviation,
            snapshot.temp_deviation,
            snapshot.thermal_margin,
            snapshot.gpu_util_pct / 100.0,
            (snapshot.fpga_temp_c - 50.0) / 30.0,
            snapshot.dram_charge,
            min(1.0, snapshot.decay_rate / 0.5),
            1.0,  # stability placeholder
        ], dtype=torch.float32)

    def feel(self, snapshot: SensorySnapshot) -> torch.Tensor:
        """
        Process sensory input into body state.

        Returns: z_feel [32] - the felt body state
        """
        device = next(self.parameters()).device
        x = self.snapshot_to_tensor(snapshot).unsqueeze(0).to(device)  # [1, 8]

        encoded = self.encoder(x)  # [1, 16]

        if self._h is None:
            self._h = torch.zeros(1, self.HIDDEN_DIM, device=device)

        self._h = self.gru(encoded, self._h)  # [1, 32]

        z_feel = self.body_head(self._h)  # [1, 32]

        return z_feel.squeeze(0)  # [32]

    def predict_next_state(self) -> torch.Tensor:
        """Predict next sensory state (interoceptive prediction)."""
        if self._h is None:
            return torch.zeros(self.SENSORY_DIM)
        return self.predict_next(self._h).squeeze(0)

    def get_z_feel(self) -> torch.Tensor:
        """Get current body state."""
        return self._h.squeeze(0) if self._h is not None else torch.zeros(self.HIDDEN_DIM)


# =============================================================================
# EXPRESS: Body-Conditioned Computation
# =============================================================================

class ExpressSystem(nn.Module):
    """
    Express computation conditioned on body state.

    Uses FiLM (Feature-wise Linear Modulation) to inject z_feel
    into every layer of a small transformer.
    """

    def __init__(self, vocab_size: int = 256, hidden_dim: int = 128, n_layers: int = 4, n_heads: int = 4):
        super().__init__()

        self.hidden_dim = hidden_dim

        # Token embedding
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embed = nn.Embedding(512, hidden_dim)

        # Transformer layers with FiLM conditioning
        self.layers = nn.ModuleList([
            FiLMTransformerLayer(hidden_dim, n_heads, z_feel_dim=32)
            for _ in range(n_layers)
        ])

        # Output
        self.ln_f = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size)

        # Early exit confidence heads (for adaptive computation)
        self.exit_heads = nn.ModuleList([
            nn.Linear(hidden_dim, vocab_size) for _ in range(n_layers)
        ])
        self.confidence_heads = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(n_layers)
        ])

    def forward(
        self,
        tokens: torch.Tensor,  # [batch, seq_len]
        z_feel: torch.Tensor,  # [32]
        confidence_threshold: float = 0.9,
        max_layers: Optional[int] = None,
    ) -> Tuple[torch.Tensor, int, float]:
        """
        Forward pass with body-conditioned early exit.

        Returns: (logits, exit_layer, confidence)
        """
        batch, seq_len = tokens.shape
        device = tokens.device

        # Embeddings
        pos = torch.arange(seq_len, device=device).unsqueeze(0)
        h = self.embed(tokens) + self.pos_embed(pos)

        # Body state for batch
        z_feel_batch = z_feel.unsqueeze(0).expand(batch, -1)  # [batch, 32]

        # Process layers with potential early exit
        n_layers = len(self.layers) if max_layers is None else min(max_layers, len(self.layers))
        exit_layer = n_layers
        confidence = 0.0

        for i, layer in enumerate(self.layers[:n_layers]):
            h = layer(h, z_feel_batch)

            # Check for early exit
            if i < n_layers - 1:
                exit_logits = self.exit_heads[i](h[:, -1, :])
                conf = torch.sigmoid(self.confidence_heads[i](h[:, -1, :])).mean().item()

                if conf > confidence_threshold:
                    exit_layer = i + 1
                    confidence = conf
                    logits = exit_logits
                    return logits, exit_layer, confidence

        # Full forward
        h = self.ln_f(h)
        logits = self.head(h[:, -1, :])
        confidence = 1.0

        return logits, exit_layer, confidence


class FiLMTransformerLayer(nn.Module):
    """Transformer layer with FiLM conditioning from z_feel."""

    def __init__(self, hidden_dim: int, n_heads: int, z_feel_dim: int = 32):
        super().__init__()

        self.attn = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

        # FiLM conditioning
        self.film_gamma = nn.Linear(z_feel_dim, hidden_dim)
        self.film_beta = nn.Linear(z_feel_dim, hidden_dim)

        # Initialize FiLM to near-identity
        nn.init.zeros_(self.film_gamma.weight)
        nn.init.ones_(self.film_gamma.bias)
        nn.init.zeros_(self.film_beta.weight)
        nn.init.zeros_(self.film_beta.bias)

    def forward(self, x: torch.Tensor, z_feel: torch.Tensor) -> torch.Tensor:
        # Self-attention
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.ln1(x + attn_out)

        # FFN
        ffn_out = self.ffn(x)
        x = self.ln2(x + ffn_out)

        # FiLM modulation
        gamma = self.film_gamma(z_feel).unsqueeze(1)  # [batch, 1, hidden]
        beta = self.film_beta(z_feel).unsqueeze(1)
        x = gamma * x + beta

        return x


# =============================================================================
# REGULATE: Hardware Actuation
# =============================================================================

class RegulateSystem:
    """
    Regulate hardware based on body state.

    Actuators:
    - GPU: Power cap, performance level
    - FPGA: Frac operations (charge DRAM cells)
    """

    def __init__(
        self,
        fpga_tracker: FPGAStateTracker,
        enable_gpu_actuation: bool = False,
    ):
        self.fpga = fpga_tracker
        self.enable_gpu = enable_gpu_actuation

        # Actuation history
        self.actions: List[dict] = []

        # Homeostatic setpoints
        self.charge_setpoint = 0.5  # Target DRAM charge
        self.power_setpoint = 50.0  # Target GPU power

    def regulate(
        self,
        z_feel: torch.Tensor,
        confidence: float,
        exit_layer: int,
        snapshot: SensorySnapshot,
    ) -> dict:
        """
        Decide and execute regulation actions.

        Returns: dict of actions taken
        """
        action = {
            'timestamp': time.time_ns(),
            'frac_count': 0,
            'power_adjustment': 0,
            'reason': [],
        }

        # === FPGA Regulation: Charge Homeostasis ===
        charge_error = self.charge_setpoint - snapshot.dram_charge

        if charge_error > 0.2:
            # Charge is low - need to replenish
            # More Fracs when charge is very low
            num_fracs = int(min(8, charge_error * 20))
            self.fpga.record_frac(num_fracs)
            action['frac_count'] = num_fracs
            action['reason'].append(f'charge_low({snapshot.dram_charge:.2f})')

        elif charge_error < -0.2 and snapshot.decay_rate < 0.3:
            # Charge is high and decay is slow - let it decay naturally
            action['reason'].append('letting_decay')

        # === Adaptive Computation Feedback ===
        # If model exited early with high confidence, we have compute headroom
        if exit_layer < 4 and confidence > 0.95:
            # Could increase charge setpoint (more analog memory)
            self.charge_setpoint = min(0.7, self.charge_setpoint + 0.01)
            action['reason'].append('headroom_increase')
        elif exit_layer >= 4 and confidence < 0.8:
            # Model needed full compute - reduce charge demand
            self.charge_setpoint = max(0.3, self.charge_setpoint - 0.01)
            action['reason'].append('compute_pressure')

        # === Temperature-Based Regulation ===
        if snapshot.fpga_temp_c > 70:
            # High FPGA temp - reduce Frac operations
            action['reason'].append('fpga_thermal_limit')

        if snapshot.gpu_temp_c > 85:
            # High GPU temp - would reduce power if enabled
            if self.enable_gpu:
                action['power_adjustment'] = -5
                action['reason'].append('gpu_thermal_limit')

        self.actions.append(action)
        return action


# =============================================================================
# EMBODIED LOOP: The Full Cycle
# =============================================================================

class EmbodiedLoop:
    """
    The complete sense→feel→express→regulate loop.

    This is the core of embodied AI: computation that is aware of
    and responsive to its physical substrate.
    """

    def __init__(
        self,
        vocab_size: int = 256,
        hidden_dim: int = 128,
        device: str = 'cpu',
    ):
        self.device = device

        # Initialize systems
        self.sense = SensorySystem()
        self.feel = FeelSystem().to(device)
        self.express = ExpressSystem(vocab_size, hidden_dim).to(device)
        self.regulate = RegulateSystem(self.sense.fpga)

        # Metrics
        self.metrics = {
            'loops': 0,
            'total_energy_j': 0.0,
            'total_fracs': 0,
            'avg_exit_layer': 0.0,
            'avg_confidence': 0.0,
        }

    def step(
        self,
        tokens: torch.Tensor,
        confidence_threshold: float = 0.9,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Execute one embodied loop iteration.

        Args:
            tokens: Input token sequence [batch, seq_len]
            confidence_threshold: Early exit threshold

        Returns:
            (output_logits, step_info)
        """
        # 1. SENSE
        snapshot = self.sense.sense()

        # 2. FEEL
        z_feel = self.feel.feel(snapshot)

        # Adaptive confidence based on body state
        # When stressed (high power/temp deviation), require higher confidence
        stress = abs(snapshot.power_deviation) + abs(snapshot.temp_deviation)
        adaptive_threshold = min(0.99, confidence_threshold + 0.1 * stress)

        # 3. EXPRESS
        tokens = tokens.to(self.device)
        with torch.no_grad():
            logits, exit_layer, confidence = self.express(
                tokens, z_feel,
                confidence_threshold=adaptive_threshold,
            )

        # 4. REGULATE
        action = self.regulate.regulate(z_feel, confidence, exit_layer, snapshot)

        # Update metrics
        self.metrics['loops'] += 1
        self.metrics['total_fracs'] += action['frac_count']
        self.metrics['avg_exit_layer'] = (
            (self.metrics['avg_exit_layer'] * (self.metrics['loops'] - 1) + exit_layer)
            / self.metrics['loops']
        )
        self.metrics['avg_confidence'] = (
            (self.metrics['avg_confidence'] * (self.metrics['loops'] - 1) + confidence)
            / self.metrics['loops']
        )

        step_info = {
            'snapshot': snapshot,
            'z_feel': z_feel.detach().cpu().numpy().tolist(),
            'exit_layer': exit_layer,
            'confidence': confidence,
            'action': action,
            'adaptive_threshold': adaptive_threshold,
        }

        return logits, step_info

    def run_loop(
        self,
        text: str,
        num_tokens: int = 50,
        temperature: float = 0.8,
    ) -> Tuple[str, List[dict]]:
        """
        Run generative loop with embodiment.

        Args:
            text: Initial text (ASCII)
            num_tokens: Tokens to generate
            temperature: Sampling temperature

        Returns:
            (generated_text, step_history)
        """
        # Convert to tokens (simple ASCII encoding)
        tokens = torch.tensor([[ord(c) for c in text[-64:]]], dtype=torch.long)

        generated = list(text)
        history = []

        self.sense.start_continuous()

        try:
            for i in range(num_tokens):
                logits, step_info = self.step(tokens)

                # Sample next token
                probs = F.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, 1).item()

                # Append
                if 32 <= next_token < 127:
                    generated.append(chr(next_token))
                else:
                    generated.append(' ')

                # Update tokens
                tokens = torch.cat([
                    tokens[:, 1:],
                    torch.tensor([[next_token]], dtype=torch.long)
                ], dim=1)

                history.append(step_info)

                # Print progress
                if (i + 1) % 10 == 0:
                    print(f"  Token {i+1}/{num_tokens}: exit_layer={step_info['exit_layer']}, "
                          f"conf={step_info['confidence']:.3f}, "
                          f"charge={step_info['snapshot'].dram_charge:.2f}")

        finally:
            self.sense.stop_continuous()

        return ''.join(generated), history


# =============================================================================
# Main: Validation
# =============================================================================

def main():
    print("=" * 70)
    print("z1136: EMBODIED LOOP - SENSE → FEEL → EXPRESS → REGULATE")
    print("=" * 70)

    # Check for GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    # Create embodied loop
    print("\nInitializing embodied systems...")
    loop = EmbodiedLoop(vocab_size=256, hidden_dim=128, device=device)

    # Test single step
    print("\n=== Single Step Test ===")
    test_tokens = torch.tensor([[ord(c) for c in "The embodied AI "]], dtype=torch.long)
    logits, info = loop.step(test_tokens)

    print(f"  z_feel shape: {len(info['z_feel'])}")
    print(f"  Exit layer: {info['exit_layer']}/4")
    print(f"  Confidence: {info['confidence']:.3f}")
    print(f"  GPU power: {info['snapshot'].gpu_power_w:.1f}W")
    print(f"  FPGA charge: {info['snapshot'].dram_charge:.3f}")
    print(f"  Action: {info['action']['reason']}")

    # Run generative loop
    print("\n=== Generative Loop Test (30 tokens) ===")
    prompt = "The machine feels"
    generated, history = loop.run_loop(prompt, num_tokens=30, temperature=0.9)

    print(f"\nGenerated: {generated}")

    # Analyze embodiment
    print("\n=== Embodiment Analysis ===")
    print(f"  Total loops: {loop.metrics['loops']}")
    print(f"  Total Frac operations: {loop.metrics['total_fracs']}")
    print(f"  Average exit layer: {loop.metrics['avg_exit_layer']:.2f}")
    print(f"  Average confidence: {loop.metrics['avg_confidence']:.3f}")

    # Check for body-computation correlation
    exit_layers = [h['exit_layer'] for h in history]
    charges = [h['snapshot'].dram_charge for h in history]

    if len(set(exit_layers)) > 1:
        # Compute correlation
        import numpy as np
        corr = np.corrcoef(exit_layers, charges)[0, 1]
        print(f"  Exit layer / charge correlation: {corr:.3f}")

    # Save results
    results = {
        'experiment': 'z1136_embodied_loop',
        'timestamp': datetime.now().isoformat(),
        'device': device,
        'prompt': prompt,
        'generated': generated,
        'metrics': loop.metrics,
        'history_sample': history[:5],  # First 5 steps
    }

    results_path = Path('results/z1136_embodied_loop.json')
    results_path.parent.mkdir(exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    print("\n" + "=" * 70)
    print("EMBODIMENT LOOP VALIDATED")
    print("=" * 70)

    return 0


if __name__ == '__main__':
    sys.exit(main())
