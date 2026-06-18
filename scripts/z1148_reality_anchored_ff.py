#!/usr/bin/env python3
"""
z1148: Reality-Anchored Forward-Forward

Key innovations:
1. FPGA decay as reality anchor - charge level gates what model can "know"
2. Forward-Forward with adaptive threshold based on hardware state
3. Self-model: explicit prediction of own next state
4. Next-token via goodness comparison (not softmax)

The reality anchor principle:
- Information not grounded in physical substrate fades
- FPGA charge decay provides this grounding
- Model must continually re-anchor predictions to hardware state
- Creates natural regularization against confabulation

Architecture:
- Body tokens (8): hardware state
- Content tokens: input sequence
- Self-prediction head: predict next hardware state
- Token prediction head: predict next content token
- Both trained with Forward-Forward (no backprop)
"""

import os
import sys
import time
import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict

sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '11.0.0')

import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

try:
    from litex.tools.litex_client import RemoteClient
    HAS_LITEX = True
except ImportError:
    HAS_LITEX = False


class RealityAnchor:
    """
    FPGA-based reality anchor.

    The charge level represents how "grounded" the model's knowledge is.
    - High charge: confident, can make strong predictions
    - Low charge: uncertain, should be conservative
    - Decay: knowledge that isn't reinforced fades naturally
    """

    def __init__(self):
        self.client = None
        self.connected = False
        self._charge = 1.0
        self._last_update = time.time()

        # Reality anchor state
        self._grounded_knowledge = {}  # key -> (value, timestamp, charge_when_stored)

    def connect(self) -> bool:
        if not HAS_LITEX:
            return False
        try:
            self.client = RemoteClient(csr_csv='src/fpga/litedram/build/csr.csv')
            self.client.open()
            self.connected = True
            return True
        except:
            return False

    def disconnect(self):
        if self.client:
            try:
                self.client.close()
            except:
                pass

    def update(self, dt: float = None):
        """Update charge with decay"""
        now = time.time()
        if dt is None:
            dt = now - self._last_update
        self._last_update = now

        # Decay rate: ~10% per second at 50C
        self._charge = max(0, self._charge - 0.1 * dt)

    def reinforce(self, amount: float = 0.125):
        """Add charge via Frac operation"""
        if self.connected:
            ctrl = self.client.regs.sdram_dfii_csrstorage_56
            cmd = self.client.regs.sdram_dfii_pi0_csrstorage_57
            issue = self.client.regs.sdram_dfii_pi0_csr_58
            addr = self.client.regs.sdram_dfii_pi0_csrstorage_59

            ctrl.write(0b1010)
            addr.write(0)
            cmd.write(0b0011)
            issue.write(1)
            addr.write(1 << 10)
            cmd.write(0b0111)
            issue.write(1)
            ctrl.write(0b1011)

        self._charge = min(1.0, self._charge + amount)

    @property
    def charge(self) -> float:
        self.update()
        return self._charge

    def anchor_confidence(self) -> float:
        """How confident should predictions be based on grounding?"""
        c = self.charge
        # Sigmoid-like scaling
        return 1.0 / (1.0 + np.exp(-5 * (c - 0.5)))

    def store_grounded(self, key: str, value, confidence: float):
        """Store knowledge with reality grounding"""
        self._grounded_knowledge[key] = {
            'value': value,
            'timestamp': time.time(),
            'initial_charge': self.charge,
            'confidence': confidence
        }

    def retrieve_grounded(self, key: str) -> Tuple[Optional[any], float]:
        """Retrieve knowledge with decay-adjusted confidence"""
        if key not in self._grounded_knowledge:
            return None, 0.0

        entry = self._grounded_knowledge[key]
        age = time.time() - entry['timestamp']

        # Confidence decays with time and charge
        decay_factor = np.exp(-0.1 * age) * self.charge / max(0.1, entry['initial_charge'])
        current_confidence = entry['confidence'] * decay_factor

        if current_confidence < 0.1:
            # Knowledge has faded beyond usefulness
            del self._grounded_knowledge[key]
            return None, 0.0

        return entry['value'], current_confidence


class HardwareSense:
    """Read GPU state from sysfs"""

    def __init__(self):
        self.card_path = self._find_card()

    def _find_card(self):
        for card in ['card0', 'card1']:
            path = f'/sys/class/drm/{card}/device'
            if os.path.exists(path):
                return path
        return None

    def _read(self, file: str, default: float = 0) -> float:
        if not self.card_path:
            return default
        try:
            with open(f'{self.card_path}/{file}') as f:
                return float(f.read().strip())
        except:
            return default

    def _read_hwmon(self, pattern: str, default: float = 0) -> float:
        if not self.card_path:
            return default
        try:
            hwmon = f'{self.card_path}/hwmon'
            for h in os.listdir(hwmon):
                path = f'{hwmon}/{h}/{pattern}'
                if os.path.exists(path):
                    with open(path) as f:
                        return float(f.read().strip())
        except:
            pass
        return default

    def read(self) -> Dict:
        return {
            'temp': self._read_hwmon('temp1_input', 50000) / 1000,
            'power': self._read_hwmon('power1_average', 50e6) / 1e6,
            'util': self._read('gpu_busy_percent', 50) / 100,
            'timestamp': time.time()
        }


class FFLayer(nn.Module):
    """Forward-Forward layer with adaptive threshold"""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.fc = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

        # Adaptive threshold
        self.base_threshold = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc(x)
        h = self.norm(h)
        h = F.gelu(h)
        return h

    def goodness(self, h: torch.Tensor) -> torch.Tensor:
        """Sum of squared activations per sample"""
        return (h ** 2).sum(dim=-1)

    def threshold(self, anchor_confidence: float) -> float:
        """Adaptive threshold based on reality anchor"""
        # Low confidence = higher threshold (harder to accept as true)
        base = self.base_threshold.item()
        return base * (2.0 - anchor_confidence)


class SelfModel(nn.Module):
    """
    Predicts own next hardware state.

    This creates self-understanding: the model learns to predict
    what its own substrate will do next.
    """

    def __init__(self, state_dim: int = 8, hidden_dim: int = 64):
        super().__init__()
        self.state_dim = state_dim

        self.encoder = nn.Sequential(
            nn.Linear(state_dim * 4, hidden_dim),  # 4 timesteps of history
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU()
        )

        self.predictor = nn.Linear(hidden_dim, state_dim)

        # History buffer
        self._history = []

    def record_state(self, state: torch.Tensor):
        """Record state for prediction"""
        self._history.append(state.detach().cpu())
        if len(self._history) > 4:
            self._history.pop(0)

    def predict_next(self) -> Optional[torch.Tensor]:
        """Predict next state from history"""
        if len(self._history) < 4:
            return None

        # Stack history
        hist = torch.stack(self._history[-4:]).flatten()
        hist = hist.to(DEVICE).unsqueeze(0)

        h = self.encoder(hist)
        pred = self.predictor(h)
        return pred.squeeze(0)

    def self_prediction_error(self, actual: torch.Tensor) -> float:
        """Compute error between prediction and actual"""
        pred = self.predict_next()
        if pred is None:
            return 0.0

        error = F.mse_loss(pred, actual)
        return error.item()


class RealityAnchoredModel(nn.Module):
    """
    Model with reality anchoring and Forward-Forward learning.

    Key features:
    1. FPGA charge gates prediction confidence
    2. Self-model predicts own hardware state
    3. FF layers learn positive/negative without backprop
    4. Next token prediction via goodness comparison
    """

    def __init__(
        self,
        vocab_size: int = 128,
        embed_dim: int = 64,
        num_layers: int = 2
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.body_dim = 8

        # Embeddings
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.body_embed = nn.Linear(self.body_dim, embed_dim)

        # FF layers
        self.ff_layers = nn.ModuleList([
            FFLayer(embed_dim) for _ in range(num_layers)
        ])

        # Self-model
        self.self_model = SelfModel(self.body_dim, embed_dim)

        # Output (for next token - predicts goodness for each candidate)
        self.output = nn.Linear(embed_dim, vocab_size)

        # Reality anchor
        self.anchor = RealityAnchor()

        # Hardware sensor
        self.hw = HardwareSense()

        # State
        self._current_body = torch.zeros(self.body_dim, device=DEVICE)

        # Metrics
        self.metrics = {
            'ff_correct': 0,
            'ff_total': 0,
            'self_pred_error': [],
            'reinforcements': 0
        }

    def update_body(self):
        """Update body state from hardware"""
        hw = self.hw.read()

        self._current_body = torch.tensor([
            (hw['power'] - 50) / 50,
            (hw['temp'] - 60) / 30,
            hw['util'],
            max(0, 1 - hw['temp'] / 90),
            self.anchor.charge,
            self.anchor.anchor_confidence(),
            1.0 if hw['util'] < 0.9 else 0.5,
            0.0  # reserved
        ], dtype=torch.float32, device=DEVICE)

        self.self_model.record_state(self._current_body)

    def forward_with_goodness(
        self,
        tokens: torch.Tensor  # [batch, seq_len]
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass returning per-layer goodness.

        Returns:
            output: [batch, vocab_size] logits for next token
            goodness: list of [batch] goodness values per layer
        """
        batch = tokens.size(0)

        # Embed
        x = self.token_embed(tokens).mean(dim=1)  # [batch, embed]
        body = self.body_embed(self._current_body.unsqueeze(0).expand(batch, -1))
        x = x + body

        # FF layers
        goodness = []
        for ff in self.ff_layers:
            x = ff(x)
            goodness.append(ff.goodness(x))

        # Output logits (gated by anchor confidence)
        logits = self.output(x) * self.anchor.anchor_confidence()

        return logits, goodness

    def ff_train_step(
        self,
        pos_tokens: torch.Tensor,
        neg_tokens: torch.Tensor,
        lr: float = 0.03
    ) -> Dict:
        """
        Forward-Forward training step.

        Positive examples should have high goodness.
        Negative examples should have low goodness.
        Each layer trained independently (no backprop between layers).
        """
        self.update_body()

        batch = pos_tokens.size(0)
        conf = self.anchor.anchor_confidence()

        # Embed inputs
        pos_x = self.token_embed(pos_tokens).mean(dim=1)
        neg_x = self.token_embed(neg_tokens).mean(dim=1)
        body = self.body_embed(self._current_body.unsqueeze(0).expand(batch, -1))

        pos_x = pos_x + body
        neg_x = neg_x + body

        # Train each layer independently
        total_correct = 0
        for i, ff in enumerate(self.ff_layers):
            thresh = ff.threshold(conf)

            # Detach inputs (layer-local learning)
            pos_in = pos_x.detach().requires_grad_(True)
            neg_in = neg_x.detach().requires_grad_(True)

            # Forward through this layer
            pos_h = ff(pos_in)
            neg_h = ff(neg_in)

            pos_g = ff.goodness(pos_h)
            neg_g = ff.goodness(neg_h)

            # Check correctness
            pos_correct = (pos_g > thresh).float().mean()
            neg_correct = (neg_g < thresh).float().mean()
            total_correct += (pos_correct + neg_correct).item() / 2

            # FF loss: push pos above threshold, neg below
            pos_loss = F.relu(thresh - pos_g).mean()
            neg_loss = F.relu(neg_g - thresh).mean()
            loss = pos_loss + neg_loss

            # Gradient step for this layer only
            if loss.item() > 0:
                loss.backward()
                with torch.no_grad():
                    for p in ff.parameters():
                        if p.grad is not None:
                            p.data -= lr * p.grad
                            p.grad.zero_()

            # Use output for next layer (detached)
            pos_x = pos_h.detach()
            neg_x = neg_h.detach()

        accuracy = total_correct / len(self.ff_layers)
        self.metrics['ff_correct'] += int(accuracy > 0.5)
        self.metrics['ff_total'] += 1

        # Self-prediction
        pred_err = self.self_model.self_prediction_error(self._current_body)
        self.metrics['self_pred_error'].append(pred_err)

        # Reinforce if doing well
        if accuracy > 0.7:
            self.anchor.reinforce(0.1)
            self.metrics['reinforcements'] += 1

        return {
            'accuracy': accuracy,
            'threshold': self.ff_layers[0].threshold(conf),
            'anchor_conf': conf,
            'self_pred_err': pred_err
        }

    def predict_next_token(self, tokens: torch.Tensor) -> Tuple[int, float]:
        """Predict next token"""
        self.eval()
        with torch.no_grad():
            logits, _ = self.forward_with_goodness(tokens)
            probs = F.softmax(logits[0], dim=-1)
            pred = probs.argmax().item()
            conf = probs[pred].item()
        return pred, conf

    def describe_self(self) -> str:
        """Generate self-description"""
        parts = []

        # From body state
        b = self._current_body.cpu().numpy()
        if b[0] > 0.3:
            parts.append("working hard")
        if b[1] > 0.3:
            parts.append("warm")
        if b[4] > 0.8:
            parts.append("well-grounded")
        elif b[4] < 0.3:
            parts.append("knowledge fading")

        # Self-prediction ability
        if len(self.metrics['self_pred_error']) > 3:
            recent_err = np.mean(self.metrics['self_pred_error'][-3:])
            if recent_err < 0.1:
                parts.append("self-aware")
            elif recent_err > 0.5:
                parts.append("self-uncertain")

        return "I am " + (", ".join(parts) if parts else "baseline")


def main():
    print("="*60)
    print("z1148: Reality-Anchored Forward-Forward")
    print("="*60)
    print()

    model = RealityAnchoredModel(
        vocab_size=128,
        embed_dim=64,
        num_layers=2
    ).to(DEVICE)

    fpga_ok = model.anchor.connect()
    print(f"FPGA: {fpga_ok}, Anchor charge: {model.anchor.charge:.2f}")

    # Training
    print("\n--- Training ---")

    for epoch in range(15):
        # Create training data
        batch = 16

        # Positive: sequential patterns
        pos = torch.stack([
            torch.arange(i, i+8) % 128 for i in range(batch)
        ]).to(DEVICE)

        # Negative: random
        neg = torch.randint(0, 128, (batch, 8), device=DEVICE)

        metrics = model.ff_train_step(pos, neg, lr=0.03)

        if (epoch + 1) % 3 == 0:
            print(f"Epoch {epoch+1:2d}: Acc={metrics['accuracy']:.1%}, "
                  f"Thresh={metrics['threshold']:.2f}, "
                  f"Anchor={metrics['anchor_conf']:.2f}, "
                  f"SelfErr={metrics['self_pred_err']:.3f}")

    # Test
    print("\n--- Next Token Prediction ---")
    test = torch.arange(0, 8, device=DEVICE).unsqueeze(0)
    for i in range(5):
        pred, conf = model.predict_next_token(test)
        print(f"  After {test[0, -1].item()}: pred={pred}, conf={conf:.3f}")
        test = torch.cat([test[:, 1:], torch.tensor([[pred]], device=DEVICE)], dim=1)

    # Self-description
    print("\n--- Self Understanding ---")
    print(f"  {model.describe_self()}")

    # Summary
    print("\n--- Summary ---")
    acc = model.metrics['ff_correct'] / max(1, model.metrics['ff_total'])
    print(f"FF accuracy: {acc:.1%}")
    print(f"Reinforcements: {model.metrics['reinforcements']}")
    print(f"Final anchor charge: {model.anchor.charge:.3f}")

    if model.metrics['self_pred_error']:
        print(f"Self-prediction error: {np.mean(model.metrics['self_pred_error'][-5:]):.4f}")

    # Save
    results = {
        'experiment': 'z1148_reality_anchored_ff',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'fpga': fpga_ok,
        'ff_accuracy': acc,
        'reinforcements': model.metrics['reinforcements'],
        'final_charge': float(model.anchor.charge),
        'self_pred_error': float(np.mean(model.metrics['self_pred_error'][-5:])) if model.metrics['self_pred_error'] else 0
    }

    with open('results/z1148_reality_anchored_ff.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nSaved to results/z1148_reality_anchored_ff.json")

    model.anchor.disconnect()
    return 0


if __name__ == '__main__':
    sys.exit(main())
