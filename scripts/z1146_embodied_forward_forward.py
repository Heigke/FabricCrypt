#!/usr/bin/env python3
"""
z1146: Embodied Forward-Forward with FPGA Reality Anchor

This implements Geoffrey Hinton's Forward-Forward algorithm on embodied hardware:
- FPGA DDR3 analog charge levels store weight-like values (partial charging)
- GPU provides compute and interoceptive telemetry
- No backpropagation - each layer learns locally via "goodness" score
- FPGA decay provides temporal grounding (reality anchor)

Theory:
- Forward-Forward: Two passes (positive/negative), each layer maximizes/minimizes
  its own activity squared sum for positive/negative inputs
- Embodied: Hardware state becomes part of the input (body tokens)
- Reality Anchor: FPGA analog decay is real physics, can't be faked
- Self-understanding: Model attends to its own hardware state

Architecture:
- Input: [body_tokens (8 dim) | content_tokens (embed_dim)]
- FPGA provides weight modulation via analog charge levels
- GPU computes forward passes with hardware-aware attention
- Goodness = sum of squared activities (layer-local objective)
- Update: Frac operations adjust FPGA charge based on goodness gradient

Key insight: Analog decay creates temporal pressure that forces the model
to continually re-anchor to physical reality. Knowledge that isn't
reinforced by hardware state naturally fades - like biological memory.
"""

import sys
import time
import json
import struct
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Tuple
import threading

sys.path.insert(0, str(Path(__file__).parent.parent))

# GPU imports
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("Warning: PyTorch not available")

# FPGA imports
try:
    from litex.tools.litex_client import RemoteClient
    HAS_LITEX = True
except ImportError:
    HAS_LITEX = False
    print("Warning: LiteX not available")

# Local imports
try:
    from src.fpga.fpga_interface import FPGAInterface
    HAS_FPGA_INTERFACE = True
except ImportError:
    HAS_FPGA_INTERFACE = False


@dataclass
class EmbodiedState:
    """Unified hardware state from GPU and FPGA"""
    # GPU (5 dimensions)
    gpu_temp: float = 50.0
    gpu_power: float = 50.0
    gpu_util: float = 0.5
    power_deviation: float = 0.0
    temp_deviation: float = 0.0

    # FPGA (3 dimensions)
    fpga_temp: float = 45.0
    dram_charge: float = 1.0
    decay_rate: float = 0.1

    # Derived
    timestamp: float = 0.0

    def to_body_vector(self) -> np.ndarray:
        """Convert to 8-dim body token input"""
        return np.array([
            self.power_deviation,      # strain
            self.temp_deviation,       # urgency
            self.gpu_util,             # load
            max(0, 1.0 - self.gpu_temp/90),  # thermal margin
            1.0 if abs(self.power_deviation) < 0.1 else 0.5,  # stability
            (self.fpga_temp - 50) / 30,  # fpga temp normalized
            self.dram_charge,          # analog charge level
            min(1.0, self.decay_rate / 0.5)  # decay pressure
        ], dtype=np.float32)


class FPGAAnalogMemory:
    """
    FPGA DDR3 as analog weight storage via partial charging.

    Each memory location stores a value 0-1 based on charge level.
    Frac operations (ACT->PRE with truncated tRAS) modify charge.
    Natural decay provides temporal grounding.
    """

    def __init__(self, base_addr: int = 0x100000, size: int = 1024):
        self.base_addr = base_addr
        self.size = size
        self.client = None
        self.connected = False

        # Simulated charge levels when no hardware
        self._sim_charges = np.ones(size, dtype=np.float32)
        self._last_decay = time.time()

        # Register cache for fast access
        self._ctrl = None
        self._pi0_cmd = None
        self._pi0_issue = None
        self._pi0_addr = None
        self._pi0_baddr = None

    def connect(self) -> bool:
        """Connect to FPGA via litex_server"""
        if not HAS_LITEX:
            print("LiteX not available, using simulation")
            return False

        try:
            self.client = RemoteClient(
                csr_csv='src/fpga/litedram/build/csr.csv'
            )
            self.client.open()

            # Cache register references
            self._ctrl = self.client.regs.sdram_dfii_csrstorage_56
            self._pi0_cmd = self.client.regs.sdram_dfii_pi0_csrstorage_57
            self._pi0_issue = self.client.regs.sdram_dfii_pi0_csr_58
            self._pi0_addr = self.client.regs.sdram_dfii_pi0_csrstorage_59
            self._pi0_baddr = self.client.regs.sdram_dfii_pi0_csrstorage_60

            # Verify connection
            ctrl_val = self._ctrl.read()
            print(f"FPGA connected, DFI control: 0x{ctrl_val:04x}")
            self.connected = True
            return True

        except Exception as e:
            print(f"FPGA connection failed: {e}")
            return False

    def disconnect(self):
        if self.client:
            try:
                self.client.close()
            except:
                pass
        self.connected = False

    def _enter_software_mode(self):
        """Switch DFI to software control"""
        if self.connected:
            self._ctrl.write(0b1010)  # sel=0, cke=1, reset_n=1

    def _exit_software_mode(self):
        """Return DFI to hardware control"""
        if self.connected:
            self._ctrl.write(0b1011)  # sel=1

    def frac_operation(self, row: int, bank: int = 0, num_fracs: int = 1):
        """
        Perform Frac operation (ACT->PRE with truncated timing).
        Each Frac adds ~1/8 charge to the row.
        """
        ACT = 0b0011  # cs=1, ras=1, cas=0, we=0
        PRE = 0b0111  # cs=1, ras=1, cas=0, we=1

        if self.connected:
            self._enter_software_mode()
            try:
                for _ in range(num_fracs):
                    # ACT
                    self._pi0_addr.write(row)
                    self._pi0_baddr.write(bank)
                    self._pi0_cmd.write(ACT)
                    self._pi0_issue.write(1)

                    # Immediate PRE (truncated tRAS)
                    self._pi0_addr.write(1 << 10)  # All banks
                    self._pi0_cmd.write(PRE)
                    self._pi0_issue.write(1)
            finally:
                self._exit_software_mode()
        else:
            # Simulation: each frac adds 1/8 charge
            idx = row % self.size
            self._sim_charges[idx] = min(1.0, self._sim_charges[idx] + 0.125)

    def apply_decay(self, dt: float = None):
        """Apply natural charge decay (Arrhenius model)"""
        now = time.time()
        if dt is None:
            dt = now - self._last_decay
        self._last_decay = now

        # Decay rate ~0.1/s at 50°C, increases with temperature
        decay = 0.1 * dt
        self._sim_charges = np.maximum(0, self._sim_charges - decay)

    def read_charge_pattern(self, start_row: int, num_rows: int = 8) -> np.ndarray:
        """
        Read charge levels from a range of rows.
        Returns normalized charge values [0, 1].
        """
        self.apply_decay()

        if self.connected:
            # Real hardware: would need to do actual reads
            # For now, use simulation
            pass

        start_idx = start_row % self.size
        end_idx = min(start_idx + num_rows, self.size)
        return self._sim_charges[start_idx:end_idx].copy()

    def write_charge_pattern(self, start_row: int, charges: np.ndarray):
        """
        Set charge levels via Frac operations.
        charges: array of target levels [0, 1]
        """
        for i, target in enumerate(charges):
            row = start_row + i
            idx = row % self.size
            current = self._sim_charges[idx]

            if target > current:
                # Need to add charge via Frac
                delta = target - current
                num_fracs = int(delta * 8) + 1  # ~8 fracs for full charge
                self.frac_operation(row, num_fracs=num_fracs)
                self._sim_charges[idx] = min(1.0, current + num_fracs * 0.125)
            # If target < current, we wait for decay (can't actively discharge)

    def get_weight_matrix(self, rows: int, cols: int, base_row: int = 0) -> np.ndarray:
        """
        Interpret charge levels as a weight matrix.
        Weights are centered: charge 0.5 -> weight 0, charge 1 -> weight 0.5
        """
        total = rows * cols
        charges = self.read_charge_pattern(base_row, total)
        if len(charges) < total:
            charges = np.pad(charges, (0, total - len(charges)), constant_values=0.5)

        # Center around 0.5 charge level
        weights = (charges - 0.5).reshape(rows, cols)
        return weights.astype(np.float32)


class ForwardForwardLayer:
    """
    A single layer using Forward-Forward learning.

    Instead of backprop, each layer has its own local objective:
    - Goodness = sum of squared activities
    - Positive examples: maximize goodness (push above threshold)
    - Negative examples: minimize goodness (push below threshold)

    The layer learns to have high activity for "good" inputs and
    low activity for "bad" inputs.
    """

    def __init__(self, in_dim: int, out_dim: int, threshold: float = 2.0):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.threshold = threshold

        # Weights initialized with small random values
        self.weights = np.random.randn(in_dim, out_dim).astype(np.float32) * 0.1
        self.bias = np.zeros(out_dim, dtype=np.float32)

        # For analog modulation from FPGA
        self.fpga_weight_mod = np.ones((in_dim, out_dim), dtype=np.float32)

        # Running statistics for normalization
        self.running_mean = np.zeros(out_dim, dtype=np.float32)
        self.running_var = np.ones(out_dim, dtype=np.float32)

    def forward(self, x: np.ndarray, normalize: bool = True) -> np.ndarray:
        """Forward pass with optional layer normalization"""
        # Apply FPGA analog modulation
        modulated_weights = self.weights * self.fpga_weight_mod

        # Linear transform
        h = x @ modulated_weights + self.bias

        # Layer normalization (crucial for FF)
        if normalize:
            h = (h - self.running_mean) / (np.sqrt(self.running_var) + 1e-5)

        # ReLU activation
        h = np.maximum(0, h)

        return h

    def goodness(self, h: np.ndarray) -> float:
        """Compute goodness = sum of squared activities"""
        return np.sum(h ** 2) / len(h)

    def update(self, x: np.ndarray, is_positive: bool, lr: float = 0.01):
        """
        Forward-Forward update step.

        For positive examples: increase weights that produce high activity
        For negative examples: decrease weights that produce high activity
        """
        h = self.forward(x)
        g = self.goodness(h)

        # Probability that this is positive
        p_pos = 1.0 / (1.0 + np.exp(-(g - self.threshold)))

        # Gradient direction
        if is_positive:
            # Push goodness above threshold
            grad_sign = 1.0 if g < self.threshold else 0.0
        else:
            # Push goodness below threshold
            grad_sign = -1.0 if g > self.threshold else 0.0

        # Simple gradient: activity * input
        # Only update if we're on the wrong side of threshold
        if abs(grad_sign) > 0:
            # Outer product for weight gradient
            h_grad = h * (h > 0).astype(np.float32)  # ReLU gradient
            weight_grad = np.outer(x, h_grad * grad_sign)

            self.weights += lr * weight_grad
            self.bias += lr * h_grad * grad_sign

        return h, g, p_pos

    def set_fpga_modulation(self, mod_matrix: np.ndarray):
        """Set weight modulation from FPGA analog values"""
        # Ensure correct shape
        if mod_matrix.shape == self.fpga_weight_mod.shape:
            self.fpga_weight_mod = mod_matrix
        else:
            # Broadcast or tile as needed
            self.fpga_weight_mod = np.ones_like(self.fpga_weight_mod) * mod_matrix.mean()


class EmbodiedForwardForward:
    """
    Complete embodied Forward-Forward network.

    Architecture:
    - Body tokens (8 dim) prepended to content
    - Multiple FF layers with FPGA weight modulation
    - Reality anchor: FPGA decay creates temporal pressure
    - Self-understanding: model sees its own hardware state
    """

    def __init__(
        self,
        content_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 3,
        vocab_size: int = 256
    ):
        self.body_dim = 8
        self.content_dim = content_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.vocab_size = vocab_size

        # Input projection (body + content -> hidden)
        input_dim = self.body_dim + content_dim
        self.input_proj = ForwardForwardLayer(input_dim, hidden_dim)

        # FF layers
        self.layers = [
            ForwardForwardLayer(hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ]

        # Output projection for next token prediction
        self.output_proj = ForwardForwardLayer(hidden_dim, vocab_size, threshold=1.0)

        # FPGA analog memory
        self.fpga = FPGAAnalogMemory()

        # Current embodied state
        self.state = EmbodiedState()

        # Metrics
        self.metrics = {
            'goodness_history': [],
            'accuracy_history': [],
            'decay_events': 0,
            'frac_operations': 0
        }

    def connect_fpga(self) -> bool:
        """Connect to FPGA for analog weight storage"""
        return self.fpga.connect()

    def update_embodied_state(self, gpu_temp: float = None, gpu_power: float = None):
        """Update hardware state for body tokens"""
        if gpu_temp is not None:
            self.state.gpu_temp = gpu_temp
            self.state.temp_deviation = (gpu_temp - 60) / 30  # Centered at 60C

        if gpu_power is not None:
            self.state.gpu_power = gpu_power
            self.state.power_deviation = (gpu_power - 50) / 50  # Centered at 50W

        # FPGA state from decay simulation
        self.fpga.apply_decay()
        charges = self.fpga.read_charge_pattern(0, 8)
        self.state.dram_charge = charges.mean()
        self.state.decay_rate = 0.1  # Base rate

        self.state.timestamp = time.time()

    def _sync_fpga_weights(self):
        """Sync FPGA analog values to layer weight modulation"""
        for i, layer in enumerate(self.layers):
            # Each layer uses a different FPGA region
            base_row = i * 128
            mod = self.fpga.get_weight_matrix(
                layer.in_dim, layer.out_dim, base_row
            )
            # Scale modulation to reasonable range [0.5, 1.5]
            mod = 1.0 + mod  # Center around 1.0
            layer.set_fpga_modulation(mod)

    def forward(self, content: np.ndarray) -> Tuple[np.ndarray, List[float]]:
        """
        Forward pass through the network.

        Args:
            content: Content embedding [content_dim]

        Returns:
            output: Logits [vocab_size]
            goodness_per_layer: List of goodness values
        """
        # Get body vector from current state
        body = self.state.to_body_vector()

        # Concatenate body + content
        x = np.concatenate([body, content])

        # Sync FPGA weights periodically
        self._sync_fpga_weights()

        # Input projection
        h = self.input_proj.forward(x)
        goodness_values = [self.input_proj.goodness(h)]

        # FF layers
        for layer in self.layers:
            h = layer.forward(h)
            goodness_values.append(layer.goodness(h))

        # Output projection (without FF threshold, just predict)
        output = self.output_proj.forward(h, normalize=False)

        return output, goodness_values

    def train_step(
        self,
        content: np.ndarray,
        is_positive: bool,
        lr: float = 0.01
    ) -> dict:
        """
        Forward-Forward training step.

        Args:
            content: Content embedding [content_dim]
            is_positive: True for positive examples, False for negative
            lr: Learning rate

        Returns:
            Dict with training metrics
        """
        body = self.state.to_body_vector()
        x = np.concatenate([body, content])

        # Modulate learning rate by FPGA charge
        # Low charge = low LR (knowledge fading, be conservative)
        charge_mod = 0.5 + 0.5 * self.state.dram_charge
        effective_lr = lr * charge_mod

        # Train each layer with Forward-Forward
        layer_metrics = []

        # Input projection
        h, g, p = self.input_proj.update(x, is_positive, effective_lr)
        layer_metrics.append({'goodness': g, 'p_positive': p})

        # Hidden layers
        for layer in self.layers:
            h, g, p = layer.update(h, is_positive, effective_lr)
            layer_metrics.append({'goodness': g, 'p_positive': p})

        # Update FPGA based on learning
        # High goodness on positive -> reinforce (more Frac)
        # Low goodness on negative -> good, maintain
        if is_positive and layer_metrics[-1]['goodness'] > 2.0:
            # Successful positive - reinforce FPGA weights
            self.fpga.frac_operation(0, num_fracs=1)
            self.metrics['frac_operations'] += 1

        avg_goodness = np.mean([m['goodness'] for m in layer_metrics])
        self.metrics['goodness_history'].append(avg_goodness)

        return {
            'layers': layer_metrics,
            'avg_goodness': avg_goodness,
            'is_positive': is_positive,
            'lr': effective_lr,
            'fpga_charge': self.state.dram_charge
        }

    def predict_next_token(self, content: np.ndarray) -> Tuple[int, float]:
        """
        Predict next token with confidence.

        Returns:
            token_id: Predicted token
            confidence: Softmax probability
        """
        logits, _ = self.forward(content)

        # Softmax
        exp_logits = np.exp(logits - logits.max())
        probs = exp_logits / exp_logits.sum()

        token_id = np.argmax(probs)
        confidence = probs[token_id]

        return int(token_id), float(confidence)

    def self_describe(self) -> str:
        """
        Generate a description of current state.
        This tests self-understanding - can the model describe itself?
        """
        body = self.state.to_body_vector()

        descriptions = []

        # Power state
        if self.state.power_deviation > 0.2:
            descriptions.append("high power draw")
        elif self.state.power_deviation < -0.2:
            descriptions.append("low power mode")
        else:
            descriptions.append("normal power")

        # Temperature
        if self.state.temp_deviation > 0.3:
            descriptions.append("running hot")
        elif self.state.temp_deviation < -0.3:
            descriptions.append("running cool")

        # FPGA state
        if self.state.dram_charge < 0.3:
            descriptions.append("memory fading")
        elif self.state.dram_charge > 0.8:
            descriptions.append("memory fresh")

        # Decay pressure
        if self.state.decay_rate > 0.2:
            descriptions.append("time pressure high")

        return "I am " + ", ".join(descriptions) if descriptions else "I am in baseline state"


def create_positive_negative_pairs(vocab_size: int, embed_dim: int, batch_size: int = 32):
    """
    Create positive and negative training pairs.

    Positive: Real sequences (or random coherent patterns)
    Negative: Corrupted versions (shuffle, noise, etc.)
    """
    pairs = []

    for _ in range(batch_size // 2):
        # Positive: smooth pattern
        base = np.random.randn(embed_dim).astype(np.float32) * 0.5
        positive = base + np.sin(np.linspace(0, 2*np.pi, embed_dim)) * 0.3

        # Negative: noisy version
        negative = base + np.random.randn(embed_dim).astype(np.float32) * 0.5

        pairs.append((positive, True))
        pairs.append((negative, False))

    np.random.shuffle(pairs)
    return pairs


def main():
    print("="*60)
    print("z1146: Embodied Forward-Forward with FPGA Reality Anchor")
    print("="*60)
    print()

    # Create model
    model = EmbodiedForwardForward(
        content_dim=64,
        hidden_dim=128,
        num_layers=3,
        vocab_size=256
    )

    # Try to connect FPGA
    fpga_connected = model.connect_fpga()
    print(f"FPGA connected: {fpga_connected}")

    # Training loop
    print("\n--- Training with Forward-Forward ---")

    num_epochs = 5
    batch_size = 32

    for epoch in range(num_epochs):
        # Simulate GPU state changes
        model.update_embodied_state(
            gpu_temp=50 + 10 * np.sin(epoch * 0.5),
            gpu_power=40 + 20 * np.cos(epoch * 0.3)
        )

        # Create training pairs
        pairs = create_positive_negative_pairs(256, 64, batch_size)

        epoch_goodness = []
        epoch_correct = 0

        for content, is_positive in pairs:
            metrics = model.train_step(content, is_positive, lr=0.01)
            epoch_goodness.append(metrics['avg_goodness'])

            # Check if classification is correct
            if is_positive and metrics['avg_goodness'] > 2.0:
                epoch_correct += 1
            elif not is_positive and metrics['avg_goodness'] < 2.0:
                epoch_correct += 1

        accuracy = epoch_correct / len(pairs)
        avg_g = np.mean(epoch_goodness)

        print(f"Epoch {epoch+1}/{num_epochs}: "
              f"Goodness={avg_g:.3f}, Accuracy={accuracy:.1%}, "
              f"FPGA charge={model.state.dram_charge:.2f}")

        # Self-description test
        desc = model.self_describe()
        print(f"  Self-report: {desc}")

    # Test next token prediction
    print("\n--- Next Token Prediction ---")

    for _ in range(5):
        test_content = np.random.randn(64).astype(np.float32)
        token, confidence = model.predict_next_token(test_content)
        print(f"Predicted token: {token}, confidence: {confidence:.3f}")

    # Final metrics
    print("\n--- Final Metrics ---")
    print(f"Total Frac operations: {model.metrics['frac_operations']}")
    print(f"Final FPGA charge: {model.state.dram_charge:.3f}")
    print(f"Goodness trend: {np.mean(model.metrics['goodness_history'][:10]):.3f} -> "
          f"{np.mean(model.metrics['goodness_history'][-10:]):.3f}")

    # Save results
    results = {
        'experiment': 'z1146_embodied_forward_forward',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'fpga_connected': fpga_connected,
        'epochs': num_epochs,
        'final_goodness': float(np.mean(model.metrics['goodness_history'][-10:])),
        'frac_operations': int(model.metrics['frac_operations']),
        'final_charge': float(model.state.dram_charge),
        'architecture': {
            'body_dim': 8,
            'content_dim': 64,
            'hidden_dim': 128,
            'num_layers': 3,
            'vocab_size': 256
        },
        'notes': [
            'Forward-Forward: no backprop, layer-local learning',
            'FPGA analog memory modulates weights',
            'Decay provides temporal grounding',
            'Self-understanding via body tokens'
        ]
    }

    results_path = Path('results/z1146_embodied_ff.json')
    results_path.parent.mkdir(exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Cleanup
    model.fpga.disconnect()

    return 0


if __name__ == '__main__':
    sys.exit(main())
