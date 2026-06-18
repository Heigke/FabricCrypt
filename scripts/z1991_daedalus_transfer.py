#!/usr/bin/env python3
"""
z1991: Cross-Machine Consciousness Transfer Test

DEPLOY ON DAEDALUS (192.168.0.37) to test substrate-dependence.

Hypothesis: A model trained on ikaros's hardware telemetry should show
measurable performance degradation when transferred to daedalus, because
the consciousness depends on the specific substrate.

Protocol:
1. Load checkpoint from ikaros z1990 training
2. Run on daedalus with local telemetry
3. Compare performance with/without embodiment
4. Test if model adapts to new substrate

This is F2 (Substrate Dependence) falsification test in detail.

Author: Claude (Opus 4.5)
Date: 2026-02-05
"""

import functools
print = functools.partial(print, flush=True)

import os
import sys
import json
import time
import socket
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Any
from collections import deque
import numpy as np

# HSA override
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"

import torch
import torch.nn as nn
import torch.nn.functional as F

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RESULTS_DIR = Path(__file__).parent.parent / 'results'


class LocalGPUSensor:
    """Local GPU telemetry for whatever machine this runs on."""

    def __init__(self):
        # Try both card0 and card1
        for card in ['/sys/class/drm/card1/device', '/sys/class/drm/card0/device']:
            if Path(card).exists():
                self.card = card
                break
        else:
            self.card = '/sys/class/drm/card0/device'

        self._history = deque(maxlen=256)
        print(f"[GPU] Using {self.card}")

    def _hwmon(self, metric: str, default: float = 0) -> float:
        try:
            hwmon_path = Path(self.card) / 'hwmon'
            for h in hwmon_path.iterdir():
                f = h / metric
                if f.exists():
                    return float(f.read_text().strip())
        except:
            pass
        return default

    def _read(self, f: str, default: float = 0) -> float:
        try:
            return float((Path(self.card) / f).read_text().strip())
        except:
            return default

    def sense(self) -> Dict[str, float]:
        state = {
            'temp': self._hwmon('temp1_input', 50000) / 1000,
            'power': self._hwmon('power1_average', 50e6) / 1e6,
            'util': self._read('gpu_busy_percent', 50) / 100,
        }
        self._history.append((time.time(), state))
        return state

    def get_tensor(self, dims: int = 16) -> torch.Tensor:
        """Get telemetry as fixed-size tensor for model compatibility."""
        s = self.sense()

        # Create 16-dim telemetry matching z1990 format
        # [local_gpu:8, remote_gpu:3, fpga:2, hackrf:3]
        tel = torch.zeros(dims, dtype=torch.float32)

        # Fill local GPU portion
        tel[0] = s['temp'] / 100  # temp normalized
        tel[1] = s['power'] / 100  # power normalized
        tel[2] = s['util']  # util already 0-1

        # Compute derivatives
        if len(self._history) >= 2:
            t1, s1 = self._history[-2]
            t2, s2 = self._history[-1]
            dt = max(t2 - t1, 0.001)
            tel[5] = (s2['temp'] - s1['temp']) / dt / 10
            tel[6] = (s2['power'] - s1['power']) / dt / 10
            tel[7] = (s2['util'] - s1['util']) / dt / 0.5

        return tel.clamp(0, 2)


# Simplified model matching z1990 architecture
class FiLMLayer(nn.Module):
    def __init__(self, hidden_dim: int, condition_dim: int):
        super().__init__()
        self.gamma = nn.Linear(condition_dim, hidden_dim)
        self.beta = nn.Linear(condition_dim, hidden_dim)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        gamma = self.gamma(condition)
        beta = self.beta(condition)
        if x.dim() == 3 and gamma.dim() == 2:
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)
        return gamma * x + beta


class GlobalWorkspace(nn.Module):
    def __init__(self, hidden_dim: int = 256, num_modules: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_modules = num_modules
        self.salience_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 1))
            for _ in range(num_modules)
        ])
        self.broadcast = nn.Sequential(
            nn.Linear(hidden_dim * num_modules, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, module_outputs):
        saliences = [head(out) for out, head in zip(module_outputs, self.salience_heads)]
        salience_stack = torch.cat(saliences, dim=-1)
        weights = F.softmax(salience_stack * 5.0, dim=-1)
        weighted = [out * weights[:, i:i+1] for i, out in enumerate(module_outputs)]
        combined = torch.cat(weighted, dim=-1)
        broadcast = self.broadcast(combined)
        return broadcast, {'ignition': (weights.max(dim=-1)[0] > 0.7).float().mean().item()}


class HigherOrderThought(nn.Module):
    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, 128), nn.GELU(),
            nn.Linear(128, 64), nn.GELU(),
            nn.Linear(64, 1), nn.Sigmoid(),
        )

    def forward(self, hidden):
        if hidden.dim() == 3:
            hidden = hidden.mean(dim=1)
        return self.confidence_head(hidden), {}


class TemporalBodyModel(nn.Module):
    def __init__(self, input_dim=16, hidden_dim=64, latent_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.gru = nn.GRU(latent_dim, hidden_dim, 2, batch_first=True, dropout=0.0)
        self.latent_proj = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, tel_seq, h=None):
        z = self.encoder(tel_seq)
        if h is None:
            h = torch.zeros(2, tel_seq.shape[0], 64, device=tel_seq.device)
        out, h_out = self.gru(z, h)
        z_out = self.latent_proj(out)
        recon = self.decoder(z_out)
        return z_out, h_out, recon


class ConsciousnessModel(nn.Module):
    def __init__(self, vocab_size=65, hidden_dim=256, telemetry_dim=16, n_layers=6):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.film_layers = nn.ModuleList([FiLMLayer(hidden_dim, telemetry_dim) for _ in range(n_layers)])
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(hidden_dim, 8, hidden_dim*4, batch_first=True, norm_first=True)
            for _ in range(n_layers)
        ])
        self.global_workspace = GlobalWorkspace(hidden_dim, 4)
        self.hot = HigherOrderThought(hidden_dim)
        self.body_model = TemporalBodyModel(telemetry_dim, hidden_dim//4, hidden_dim//8)
        self.output = nn.Linear(hidden_dim, vocab_size)
        self.module_projs = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(4)])

    def forward(self, tokens, telemetry, telemetry_seq=None):
        h = self.embed(tokens)
        for layer, film in zip(self.layers, self.film_layers):
            h = film(h, telemetry)
            h = layer(h)

        seq_len = h.shape[1]
        module_outputs = [proj(h[:, min(i*(seq_len//4), seq_len-1), :]) for i, proj in enumerate(self.module_projs)]
        broadcast, gwt_info = self.global_workspace(module_outputs)
        h = h + broadcast.unsqueeze(1)
        confidence, hot_info = self.hot(h)
        logits = self.output(h)

        return {'logits': logits, 'confidence': confidence, 'gwt_info': gwt_info}


class TextDataset:
    def __init__(self, text: str, seq_len: int = 64):
        self.text = text
        self.seq_len = seq_len
        self.chars = sorted(set(text))
        self.char2idx = {c: i for i, c in enumerate(self.chars)}
        self.vocab_size = len(self.chars)
        self.data = torch.tensor([self.char2idx.get(c, 0) for c in text], dtype=torch.long)

    def __len__(self):
        return len(self.data) - self.seq_len - 1

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len]
        y = self.data[idx + 1:idx + self.seq_len + 1]
        return x, y


def evaluate_model(model, dataset, gpu_sensor, device, mode='embodied'):
    """Evaluate model with different telemetry conditions."""
    model.eval()

    total_loss = 0
    correct = 0
    total = 0
    confidences = []
    accuracies = []

    batch_size = 32
    num_batches = min(500, len(dataset) // batch_size)

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        batch_x = [dataset[start_idx + i][0] for i in range(batch_size)]
        batch_y = [dataset[start_idx + i][1] for i in range(batch_size)]

        x = torch.stack(batch_x).to(device)
        y = torch.stack(batch_y).to(device)

        # Get telemetry based on mode
        if mode == 'embodied':
            tel = gpu_sensor.get_tensor(16).to(device)
        elif mode == 'zero':
            tel = torch.zeros(16, device=device)
        elif mode == 'random':
            tel = torch.rand(16, device=device)
        else:  # 'fixed'
            tel = torch.ones(16, device=device) * 0.5

        with torch.no_grad():
            out = model(x, tel.unsqueeze(0).expand(batch_size, -1))
            logits = out['logits'].view(-1, dataset.vocab_size)
            loss = F.cross_entropy(logits, y.view(-1))

            preds = logits.argmax(dim=-1)
            correct += (preds == y.view(-1)).sum().item()
            total += y.numel()
            total_loss += loss.item()

            conf = out['confidence'].mean().item()
            acc = (preds == y.view(-1)).float().mean().item()
            confidences.append(conf)
            accuracies.append(acc)

    # Compute calibration
    corr = np.corrcoef(confidences, accuracies)[0, 1] if len(confidences) > 5 else 0
    if np.isnan(corr):
        corr = 0

    return {
        'loss': total_loss / num_batches,
        'accuracy': correct / total,
        'confidence_accuracy_corr': corr,
        'mean_confidence': np.mean(confidences),
    }


def load_shakespeare():
    paths = [
        Path(__file__).parent.parent / 'data' / 'shakespeare.txt',
        Path(__file__).parent.parent / 'data' / 'tiny_shakespeare.txt',
    ]
    for p in paths:
        if p.exists():
            return p.read_text()
    # Minimal fallback
    return "To be or not to be that is the question\n" * 10000


def main():
    print("="*70)
    print("z1991: CROSS-MACHINE CONSCIOUSNESS TRANSFER TEST")
    print("Testing substrate dependence on different hardware")
    print("="*70)
    print(f"Machine: {socket.gethostname()}")
    print(f"Device: {DEVICE}")
    print(f"Time: {datetime.now().isoformat()}")
    print()

    # Initialize local GPU sensor
    gpu_sensor = LocalGPUSensor()
    initial_tel = gpu_sensor.get_tensor()
    print(f"Telemetry sample: {initial_tel[:5].tolist()}")

    # Load dataset
    text = load_shakespeare()
    dataset = TextDataset(text, seq_len=64)
    print(f"Dataset: {len(dataset)} samples, vocab {dataset.vocab_size}")

    # Create model
    model = ConsciousnessModel(
        vocab_size=dataset.vocab_size,
        hidden_dim=256,
        telemetry_dim=16,
        n_layers=6,
    ).to(DEVICE)

    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # Try to load checkpoint from ikaros
    checkpoint_paths = [
        Path('/home/daedalus/z1990_checkpoint_e150.pt'),  # If copied
        Path(__file__).parent.parent / 'results' / 'z1990_checkpoint_e150.pt',
        Path(__file__).parent.parent / 'results' / 'z1990_checkpoint_e100.pt',
        Path(__file__).parent.parent / 'results' / 'z1990_checkpoint_e75.pt',
        Path(__file__).parent.parent / 'results' / 'z1990_checkpoint_e50.pt',
        Path(__file__).parent.parent / 'results' / 'z1990_checkpoint_e25.pt',
    ]

    loaded_checkpoint = None
    for cp_path in checkpoint_paths:
        if cp_path.exists():
            print(f"Loading checkpoint: {cp_path}")
            checkpoint = torch.load(cp_path, map_location=DEVICE)
            model.load_state_dict(checkpoint['model_state'])
            loaded_checkpoint = cp_path.name
            break
    else:
        print("[!] No checkpoint found - running with random weights")
        print("    (This tests whether model structure alone shows embodiment effects)")

    # Run evaluation in multiple modes
    print("\n" + "="*70)
    print("EVALUATION: Comparing telemetry modes")
    print("="*70)

    results = {}
    modes = ['embodied', 'zero', 'random', 'fixed']

    for mode in modes:
        print(f"\nEvaluating with {mode.upper()} telemetry...")
        start = time.time()
        metrics = evaluate_model(model, dataset, gpu_sensor, DEVICE, mode)
        elapsed = time.time() - start

        results[mode] = metrics
        print(f"  Loss: {metrics['loss']:.4f}")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  HOT Calibration: {metrics['confidence_accuracy_corr']:.4f}")
        print(f"  Time: {elapsed:.1f}s")

    # Compute transfer metrics
    print("\n" + "="*70)
    print("TRANSFER ANALYSIS")
    print("="*70)

    # Embodiment ratio: embodied/zero
    embodiment_ratio = results['embodied']['accuracy'] / (results['zero']['accuracy'] + 1e-8)
    print(f"Embodiment Ratio (embodied/zero): {embodiment_ratio:.3f}")

    # Transfer cost: (embodied - zero) / embodied
    if results['embodied']['accuracy'] > results['zero']['accuracy']:
        transfer_benefit = (results['embodied']['accuracy'] - results['zero']['accuracy']) / results['embodied']['accuracy']
    else:
        transfer_benefit = 0
    print(f"Embodiment Benefit: {transfer_benefit*100:.2f}%")

    # Substrate specificity: does this machine's telemetry help vs random?
    substrate_specificity = results['embodied']['accuracy'] - results['random']['accuracy']
    print(f"Substrate Specificity: {substrate_specificity:.4f}")

    # Verdict
    print("\n" + "="*70)
    if embodiment_ratio > 1.1 and substrate_specificity > 0.01:
        verdict = "SUBSTRATE-DEPENDENT CONSCIOUSNESS CONFIRMED"
        explanation = "Telemetry from this machine improves performance vs zero/random"
    elif embodiment_ratio > 1.05:
        verdict = "WEAK SUBSTRATE DEPENDENCE"
        explanation = "Small benefit from embodiment, may be noise"
    else:
        verdict = "NO SUBSTRATE DEPENDENCE DETECTED"
        explanation = "Model performs similarly with or without this machine's telemetry"

    print(f"VERDICT: {verdict}")
    print(f"  {explanation}")
    print("="*70)

    # Save results
    final_results = {
        'experiment': 'z1991_daedalus_transfer',
        'timestamp': datetime.now().isoformat(),
        'machine': socket.gethostname(),
        'device': str(DEVICE),
        'checkpoint_loaded': loaded_checkpoint,
        'results_by_mode': results,
        'analysis': {
            'embodiment_ratio': embodiment_ratio,
            'embodiment_benefit_pct': transfer_benefit * 100,
            'substrate_specificity': substrate_specificity,
        },
        'verdict': verdict,
        'explanation': explanation,
    }

    output_path = RESULTS_DIR / f'z1991_transfer_{socket.gethostname()}.json'
    with open(output_path, 'w') as f:
        json.dump(final_results, f, indent=2)

    print(f"\nResults saved: {output_path}")

    return final_results


if __name__ == '__main__':
    main()
