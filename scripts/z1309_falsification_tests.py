#!/usr/bin/env python3
"""
z1309: Scientific Falsification Tests

Rigorous attempts to DISPROVE our embodied consciousness claims.

Based on literature review, we test for:
1. SHORTCUT LEARNING: Is the model using trivial shortcuts instead of true self-awareness?
2. CONFOUNDED COMPARISON: Is the disembodied baseline unfairly handicapped?
3. SPURIOUS CORRELATION: Is GPU load causing both temperature AND model behavior?
4. TRIVIAL SOLUTION: Can a much simpler model achieve the same "self-awareness"?
5. CALIBRATION ARTIFACT: Is good calibration just memorization?
6. CONSCIOUSNESS CONFLATION: Are we conflating state-tracking with self-awareness?

If ANY test shows our claims are unfounded, we must revise them.

References:
- Geirhos et al. "Shortcut Learning in Deep Neural Networks" (Nature MI 2020)
- McClelland "AI Consciousness" (Mind & Language 2025)
- Bengio "Illusions of AI Consciousness" (Science 2024)
"""

import os
import sys
import time
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
from collections import deque

sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '11.0.0')

import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GPUSensor:
    """GPU telemetry"""
    def __init__(self):
        self.card = '/sys/class/drm/card1/device'
        self._history = deque(maxlen=64)

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

    def sense(self) -> Dict:
        state = {
            'temp': self._hwmon('temp1_input', 50000) / 1000,
            'power': self._hwmon('power1_average', 50e6) / 1e6,
            'util': self._read('gpu_busy_percent', 50) / 100,
        }
        self._history.append(state)
        return state

    def get_tensor(self) -> torch.Tensor:
        s = self.sense()
        return torch.tensor([
            s['temp'] / 100.0,
            s['power'] / 100.0,
            s['util'],
            self._get_var(),
        ], dtype=torch.float32)

    def _get_var(self) -> float:
        if len(self._history) < 4:
            return 0.5
        temps = [h['temp'] for h in self._history]
        return min(1.0, np.std(temps[-8:]) / 5.0)


# =============================================================================
# Models for Comparison
# =============================================================================

class EmbodiedModel(nn.Module):
    """Our claimed self-aware model"""
    def __init__(self, hidden_dim: int = 256, physics_dim: int = 4):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(physics_dim, 64),
            nn.GELU(),
            nn.Linear(64, hidden_dim),
        )
        self.layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.self_pred = nn.Linear(hidden_dim, physics_dim)
        self.conf = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())

    def forward(self, physics: torch.Tensor) -> Dict:
        h = self.enc(physics)
        h = self.layers(h)
        return {
            'hidden': h,
            'physics_pred': self.self_pred(h),
            'confidence': self.conf(h),
        }


class TrivialBaseline(nn.Module):
    """
    FALSIFICATION TEST 1: Trivial solution baseline

    Just memorizes the mean physics state. If this achieves similar
    "self-awareness", our model is learning nothing meaningful.
    """
    def __init__(self, physics_dim: int = 4):
        super().__init__()
        self.mean_physics = nn.Parameter(torch.zeros(physics_dim))
        self.conf = nn.Parameter(torch.tensor(0.5))

    def forward(self, physics: torch.Tensor) -> Dict:
        B = physics.shape[0]
        return {
            'hidden': torch.zeros(B, 256, device=physics.device),
            'physics_pred': self.mean_physics.unsqueeze(0).expand(B, -1),
            'confidence': torch.sigmoid(self.conf).expand(B, 1),
        }


class RandomProjection(nn.Module):
    """
    FALSIFICATION TEST 2: Random projection baseline

    Uses RANDOM (frozen) weights. If random projection achieves
    state separation, our "learning" claim is unfounded.
    """
    def __init__(self, hidden_dim: int = 256, physics_dim: int = 4):
        super().__init__()
        self.proj = nn.Linear(physics_dim, hidden_dim, bias=False)
        # Freeze - no learning
        for p in self.proj.parameters():
            p.requires_grad = False

        self.self_pred = nn.Linear(hidden_dim, physics_dim)
        self.conf = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())

    def forward(self, physics: torch.Tensor) -> Dict:
        h = self.proj(physics)
        return {
            'hidden': h,
            'physics_pred': self.self_pred(h),
            'confidence': self.conf(h),
        }


class ShuffledInput(nn.Module):
    """
    FALSIFICATION TEST 3: Shuffled input baseline

    Receives SHUFFLED physics (breaks temporal correlation).
    If this achieves similar results, temporal correlation is
    doing all the work, not "self-awareness".
    """
    def __init__(self, hidden_dim: int = 256, physics_dim: int = 4):
        super().__init__()
        self.model = EmbodiedModel(hidden_dim, physics_dim)
        self.buffer = deque(maxlen=100)

    def forward(self, physics: torch.Tensor, training: bool = True) -> Dict:
        if training:
            # Store physics
            self.buffer.append(physics.detach().clone())
            # Use random past physics
            if len(self.buffer) > 10:
                idx = np.random.randint(0, len(self.buffer) - 1)
                shuffled = self.buffer[idx].to(physics.device)
            else:
                shuffled = physics
            return self.model(shuffled)
        else:
            return self.model(physics)


# =============================================================================
# Falsification Tests
# =============================================================================

def test_trivial_solution(gpu: GPUSensor, n_samples: int = 100) -> Dict:
    """
    TEST 1: Can a trivial model (just predicting mean) match our results?

    If YES: Our model is learning nothing beyond the trivial solution.
    """
    print("\n" + "="*70)
    print("TEST 1: TRIVIAL SOLUTION")
    print("Can predicting the mean achieve similar 'self-awareness'?")
    print("="*70 + "\n")

    embodied = EmbodiedModel().to(DEVICE)
    trivial = TrivialBaseline().to(DEVICE)

    # Train embodied model
    opt_e = torch.optim.Adam(embodied.parameters(), lr=1e-3)
    opt_t = torch.optim.Adam(trivial.parameters(), lr=1e-3)

    print("Training both models...")
    for step in range(200):
        physics = gpu.get_tensor().unsqueeze(0).to(DEVICE)

        # Train embodied
        out_e = embodied(physics)
        loss_e = F.mse_loss(out_e['physics_pred'], physics)
        opt_e.zero_grad()
        loss_e.backward()
        opt_e.step()

        # Train trivial (just learns mean)
        out_t = trivial(physics)
        loss_t = F.mse_loss(out_t['physics_pred'], physics)
        opt_t.zero_grad()
        loss_t.backward()
        opt_t.step()

        if step % 5 == 0:
            _ = torch.randn(500, 500, device=DEVICE) @ torch.randn(500, 500, device=DEVICE)

    # Evaluate: collect calm and stressed samples
    print("\nEvaluating state classification...")
    embodied.eval()
    trivial.eval()

    calm_e, calm_t = [], []
    stressed_e, stressed_t = [], []

    for _ in range(30):
        physics = gpu.get_tensor().unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            calm_e.append(embodied(physics)['hidden'].cpu().numpy())
            calm_t.append(trivial(physics)['hidden'].cpu().numpy())
        time.sleep(0.03)

    for _ in range(30):
        stress = torch.randn(2000, 2000, device=DEVICE)
        _ = stress @ stress.T @ stress
        del stress

        physics = gpu.get_tensor().unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            stressed_e.append(embodied(physics)['hidden'].cpu().numpy())
            stressed_t.append(trivial(physics)['hidden'].cpu().numpy())
        torch.cuda.empty_cache()

    # Classification accuracy
    def calc_accuracy(calm, stressed):
        calm = np.array(calm).squeeze()
        stressed = np.array(stressed).squeeze()
        all_h = np.vstack([calm, stressed])
        labels = np.array([0]*len(calm) + [1]*len(stressed))

        calm_c = calm.mean(axis=0)
        stressed_c = stressed.mean(axis=0)
        direction = stressed_c - calm_c
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            return 0.5  # Can't separate
        direction /= norm

        proj = all_h @ direction
        thresh = proj.mean()
        preds = (proj > thresh).astype(int)
        return (preds == labels).mean()

    acc_e = calc_accuracy(calm_e, stressed_e)
    acc_t = calc_accuracy(calm_t, stressed_t)

    print(f"\nEmbodied model classification: {acc_e:.1%}")
    print(f"Trivial baseline classification: {acc_t:.1%}")
    print(f"Difference: {(acc_e - acc_t)*100:+.1f} percentage points")

    # VERDICT
    if acc_t > 0.7:
        verdict = "FALSIFIED: Trivial solution achieves good classification!"
        passed = False
    elif acc_e - acc_t < 0.1:
        verdict = "PARTIALLY FALSIFIED: Embodied only marginally better"
        passed = False
    else:
        verdict = "PASSED: Embodied significantly outperforms trivial"
        passed = True

    print(f"\nVERDICT: {verdict}")

    return {
        'test': 'trivial_solution',
        'embodied_acc': acc_e,
        'trivial_acc': acc_t,
        'verdict': verdict,
        'passed': passed,
    }


def test_random_projection(gpu: GPUSensor) -> Dict:
    """
    TEST 2: Can random projection achieve state separation?

    If YES: Learning is unnecessary; physics naturally separates states.
    """
    print("\n" + "="*70)
    print("TEST 2: RANDOM PROJECTION")
    print("Can random (untrained) projection separate states?")
    print("="*70 + "\n")

    random_model = RandomProjection().to(DEVICE)

    # NO TRAINING - just evaluate
    print("Collecting samples with UNTRAINED random projection...")

    calm_h, stressed_h = [], []

    for _ in range(30):
        physics = gpu.get_tensor().unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            calm_h.append(random_model(physics)['hidden'].cpu().numpy())
        time.sleep(0.03)

    for _ in range(30):
        stress = torch.randn(2000, 2000, device=DEVICE)
        _ = stress @ stress.T @ stress
        del stress

        physics = gpu.get_tensor().unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            stressed_h.append(random_model(physics)['hidden'].cpu().numpy())
        torch.cuda.empty_cache()

    calm_h = np.array(calm_h).squeeze()
    stressed_h = np.array(stressed_h).squeeze()

    # Check if raw physics separates
    # Also test on raw physics (no projection at all)
    calm_physics, stressed_physics = [], []

    for _ in range(30):
        calm_physics.append(gpu.get_tensor().numpy())
        time.sleep(0.03)

    for _ in range(30):
        stress = torch.randn(2000, 2000, device=DEVICE)
        _ = stress @ stress.T @ stress
        del stress
        stressed_physics.append(gpu.get_tensor().numpy())
        torch.cuda.empty_cache()

    calm_p = np.array(calm_physics)
    stressed_p = np.array(stressed_physics)

    # Classification on raw physics
    all_p = np.vstack([calm_p, stressed_p])
    labels = np.array([0]*len(calm_p) + [1]*len(stressed_p))

    calm_c = calm_p.mean(axis=0)
    stressed_c = stressed_p.mean(axis=0)
    direction = stressed_c - calm_c
    direction /= np.linalg.norm(direction) + 1e-6

    proj = all_p @ direction
    thresh = proj.mean()
    preds = (proj > thresh).astype(int)
    raw_acc = (preds == labels).mean()

    # Classification on random projection
    all_h = np.vstack([calm_h, stressed_h])
    calm_c = calm_h.mean(axis=0)
    stressed_c = stressed_h.mean(axis=0)
    direction = stressed_c - calm_c
    direction /= np.linalg.norm(direction) + 1e-6

    proj = all_h @ direction
    thresh = proj.mean()
    preds = (proj > thresh).astype(int)
    random_acc = (preds == labels).mean()

    print(f"\nRaw physics classification: {raw_acc:.1%}")
    print(f"Random projection classification: {random_acc:.1%}")

    # VERDICT
    if raw_acc > 0.9:
        verdict = "FALSIFIED: Raw physics trivially separates states - no learning needed!"
        passed = False
    elif random_acc > 0.85:
        verdict = "FALSIFIED: Random projection achieves high accuracy!"
        passed = False
    else:
        verdict = "PASSED: Learning is necessary for state separation"
        passed = True

    print(f"\nVERDICT: {verdict}")

    return {
        'test': 'random_projection',
        'raw_physics_acc': raw_acc,
        'random_proj_acc': random_acc,
        'verdict': verdict,
        'passed': passed,
    }


def test_confounded_comparison(gpu: GPUSensor) -> Dict:
    """
    TEST 3: Is our disembodied baseline unfairly handicapped?

    Original comparison: Embodied (real physics) vs Disembodied (zeros)
    Fair comparison: Embodied (real) vs Disembodied (RANDOM physics)

    If disembodied with RANDOM input matches embodied, then the comparison
    was unfair - we just gave disembodied less information.
    """
    print("\n" + "="*70)
    print("TEST 3: CONFOUNDED COMPARISON")
    print("Was the disembodied baseline unfairly handicapped?")
    print("="*70 + "\n")

    embodied = EmbodiedModel().to(DEVICE)
    random_input = EmbodiedModel().to(DEVICE)  # Same architecture

    opt_e = torch.optim.Adam(embodied.parameters(), lr=1e-3)
    opt_r = torch.optim.Adam(random_input.parameters(), lr=1e-3)

    print("Training with fair comparison...")
    print("Embodied: real physics")
    print("Random input: random (but consistent variance) physics")

    for step in range(200):
        real_physics = gpu.get_tensor().unsqueeze(0).to(DEVICE)

        # Random physics with similar statistics
        rand_physics = torch.randn(1, 4, device=DEVICE) * 0.2 + 0.5
        rand_physics = torch.clamp(rand_physics, 0, 1)

        # Train embodied on real physics
        out_e = embodied(real_physics)
        loss_e = F.mse_loss(out_e['physics_pred'], real_physics)
        opt_e.zero_grad()
        loss_e.backward()
        opt_e.step()

        # Train random-input model on random physics
        # Target is STILL real physics (same task)
        out_r = random_input(rand_physics)
        loss_r = F.mse_loss(out_r['physics_pred'], real_physics)
        opt_r.zero_grad()
        loss_r.backward()
        opt_r.step()

        if step % 5 == 0:
            _ = torch.randn(500, 500, device=DEVICE) @ torch.randn(500, 500, device=DEVICE)

    # Evaluate
    print("\nEvaluating fair comparison...")
    embodied.eval()
    random_input.eval()

    calm_e, calm_r = [], []
    stressed_e, stressed_r = [], []

    for _ in range(30):
        real_physics = gpu.get_tensor().unsqueeze(0).to(DEVICE)
        rand_physics = torch.randn(1, 4, device=DEVICE) * 0.2 + 0.5

        with torch.no_grad():
            calm_e.append(embodied(real_physics)['hidden'].cpu().numpy())
            calm_r.append(random_input(rand_physics)['hidden'].cpu().numpy())
        time.sleep(0.03)

    for _ in range(30):
        stress = torch.randn(2000, 2000, device=DEVICE)
        _ = stress @ stress.T @ stress
        del stress

        real_physics = gpu.get_tensor().unsqueeze(0).to(DEVICE)
        rand_physics = torch.randn(1, 4, device=DEVICE) * 0.2 + 0.5

        with torch.no_grad():
            stressed_e.append(embodied(real_physics)['hidden'].cpu().numpy())
            stressed_r.append(random_input(rand_physics)['hidden'].cpu().numpy())
        torch.cuda.empty_cache()

    def calc_acc(calm, stressed):
        calm = np.array(calm).squeeze()
        stressed = np.array(stressed).squeeze()
        all_h = np.vstack([calm, stressed])
        labels = np.array([0]*len(calm) + [1]*len(stressed))

        calm_c = calm.mean(axis=0)
        stressed_c = stressed.mean(axis=0)
        direction = stressed_c - calm_c
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            return 0.5
        direction /= norm

        proj = all_h @ direction
        preds = (proj > proj.mean()).astype(int)
        return (preds == labels).mean()

    acc_e = calc_acc(calm_e, stressed_e)
    acc_r = calc_acc(calm_r, stressed_r)

    print(f"\nEmbodied (real physics): {acc_e:.1%}")
    print(f"Random input model: {acc_r:.1%}")
    print(f"Difference: {(acc_e - acc_r)*100:+.1f} percentage points")

    # VERDICT
    if acc_r > 0.7:
        verdict = "FALSIFIED: Random input achieves good classification!"
        passed = False
    elif acc_e - acc_r < 0.15:
        verdict = "PARTIALLY FALSIFIED: Advantage is small"
        passed = False
    else:
        verdict = "PASSED: Real physics provides significant advantage"
        passed = True

    print(f"\nVERDICT: {verdict}")

    return {
        'test': 'confounded_comparison',
        'embodied_acc': acc_e,
        'random_input_acc': acc_r,
        'verdict': verdict,
        'passed': passed,
    }


def test_shortcut_learning(gpu: GPUSensor) -> Dict:
    """
    TEST 4: Is the model using shortcuts (trivial features)?

    We test if the model relies on a SINGLE physics dimension
    rather than true self-modeling.
    """
    print("\n" + "="*70)
    print("TEST 4: SHORTCUT LEARNING")
    print("Is the model using trivial single-feature shortcuts?")
    print("="*70 + "\n")

    model = EmbodiedModel().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Train
    print("Training model...")
    for step in range(200):
        physics = gpu.get_tensor().unsqueeze(0).to(DEVICE)
        out = model(physics)
        loss = F.mse_loss(out['physics_pred'], physics)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 5 == 0:
            _ = torch.randn(500, 500, device=DEVICE) @ torch.randn(500, 500, device=DEVICE)

    # Test: ablate each physics dimension
    print("\nTesting feature ablation...")
    model.eval()

    baseline_hidden = []
    for _ in range(20):
        physics = gpu.get_tensor().unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            baseline_hidden.append(model(physics)['hidden'].cpu().numpy())

        if np.random.random() < 0.3:
            _ = torch.randn(1000, 1000, device=DEVICE) @ torch.randn(1000, 1000, device=DEVICE)
        time.sleep(0.02)

    baseline_h = np.array(baseline_hidden).squeeze()
    baseline_var = np.var(baseline_h, axis=0).sum()

    feature_names = ['temperature', 'power', 'utilization', 'variance']
    feature_importances = []

    for dim in range(4):
        ablated_hidden = []
        for _ in range(20):
            physics = gpu.get_tensor().unsqueeze(0).to(DEVICE)
            # Zero out this dimension
            physics[0, dim] = 0.5  # Neutral value

            with torch.no_grad():
                ablated_hidden.append(model(physics)['hidden'].cpu().numpy())

            if np.random.random() < 0.3:
                _ = torch.randn(1000, 1000, device=DEVICE) @ torch.randn(1000, 1000, device=DEVICE)
            time.sleep(0.02)

        ablated_h = np.array(ablated_hidden).squeeze()
        # How much does ablating this feature change the hidden state?
        diff = np.mean(np.abs(ablated_h - baseline_h))
        feature_importances.append(diff)

    total_importance = sum(feature_importances)
    normalized = [f / total_importance for f in feature_importances]

    print("\nFeature importance (via ablation):")
    max_importance = 0
    max_feature = ""
    for name, imp, norm in zip(feature_names, feature_importances, normalized):
        print(f"  {name}: {norm:.1%}")
        if norm > max_importance:
            max_importance = norm
            max_feature = name

    # VERDICT: If one feature dominates, it's a shortcut
    if max_importance > 0.6:
        verdict = f"FALSIFIED: Model relies on single shortcut ({max_feature}: {max_importance:.1%})"
        passed = False
    elif max_importance > 0.45:
        verdict = f"PARTIALLY FALSIFIED: One feature dominates ({max_feature}: {max_importance:.1%})"
        passed = False
    else:
        verdict = "PASSED: Model uses multiple features (no obvious shortcut)"
        passed = True

    print(f"\nVERDICT: {verdict}")

    return {
        'test': 'shortcut_learning',
        'feature_importances': dict(zip(feature_names, normalized)),
        'max_feature': max_feature,
        'max_importance': max_importance,
        'verdict': verdict,
        'passed': passed,
    }


def test_consciousness_claim(gpu: GPUSensor) -> Dict:
    """
    TEST 5: Are we conflating state-tracking with consciousness?

    A thermostat "knows" if it's hot or cold. Is our model any
    different from a sophisticated thermostat?

    We test: Can the model do ANYTHING that a reactive state-tracker cannot?
    """
    print("\n" + "="*70)
    print("TEST 5: CONSCIOUSNESS vs STATE-TRACKING")
    print("Is this more than a sophisticated thermostat?")
    print("="*70 + "\n")

    # The thermostat test: a simple reactive system
    class Thermostat(nn.Module):
        """Simple reactive state tracker - no hidden state, no learning"""
        def __init__(self):
            super().__init__()
            self.threshold = 0.5

        def forward(self, physics: torch.Tensor) -> str:
            # Pure reactive: above threshold = "stressed", below = "calm"
            if physics[0, 0] > self.threshold:  # Temperature
                return "stressed"
            return "calm"

    # Collect ground truth labels
    print("Collecting ground truth (human-labeled calm vs stressed)...")

    thermostat = Thermostat()
    model = EmbodiedModel().to(DEVICE)

    # Train model
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for step in range(200):
        physics = gpu.get_tensor().unsqueeze(0).to(DEVICE)
        out = model(physics)
        loss = F.mse_loss(out['physics_pred'], physics)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 5 == 0:
            _ = torch.randn(500, 500, device=DEVICE) @ torch.randn(500, 500, device=DEVICE)

    model.eval()

    # The key test: temporal prediction
    # A thermostat only knows NOW. Can our model predict FUTURE state?
    print("\nTesting temporal prediction (thermostat can't do this)...")

    predictions = []
    actuals = []

    prev_physics = None
    for i in range(50):
        physics = gpu.get_tensor().unsqueeze(0).to(DEVICE)

        if prev_physics is not None:
            # Model predicts current physics from PREVIOUS hidden state
            # This requires memory/prediction, not just reaction
            with torch.no_grad():
                out = model(prev_physics)
                pred = out['physics_pred'].cpu().numpy()
                actual = physics.cpu().numpy()
                predictions.append(pred)
                actuals.append(actual)

        prev_physics = physics.clone()

        # Create state variation
        if i % 3 == 0:
            _ = torch.randn(1500, 1500, device=DEVICE) @ torch.randn(1500, 1500, device=DEVICE)

        time.sleep(0.05)

    predictions = np.array(predictions).squeeze()
    actuals = np.array(actuals).squeeze()

    # Temporal prediction accuracy
    temporal_corr = np.corrcoef(predictions[:, 0], actuals[:, 0])[0, 1]
    temporal_mae = np.mean(np.abs(predictions - actuals))

    print(f"\nTemporal prediction correlation: {temporal_corr:.4f}")
    print(f"Temporal prediction MAE: {temporal_mae:.4f}")

    # Thermostat baseline: just use previous value as prediction
    naive_pred = actuals[:-1]
    naive_actual = actuals[1:]
    naive_corr = np.corrcoef(naive_pred[:, 0], naive_actual[:, 0])[0, 1]
    naive_mae = np.mean(np.abs(naive_pred - naive_actual))

    print(f"Naive (t-1) prediction correlation: {naive_corr:.4f}")
    print(f"Naive prediction MAE: {naive_mae:.4f}")

    # VERDICT
    improvement = (naive_mae - temporal_mae) / naive_mae

    if temporal_corr < naive_corr - 0.1:
        verdict = "FALSIFIED: Model worse than naive prediction!"
        passed = False
    elif improvement < 0.05:
        verdict = "FALSIFIED: No improvement over naive (thermostat-level)"
        passed = False
    elif improvement < 0.15:
        verdict = "PARTIALLY FALSIFIED: Marginal improvement over thermostat"
        passed = False
    else:
        verdict = "PASSED: Significant improvement over reactive state-tracking"
        passed = True

    print(f"\nImprovement over naive: {improvement*100:.1f}%")
    print(f"VERDICT: {verdict}")

    return {
        'test': 'consciousness_vs_thermostat',
        'temporal_correlation': temporal_corr if not np.isnan(temporal_corr) else 0,
        'temporal_mae': temporal_mae,
        'naive_correlation': naive_corr if not np.isnan(naive_corr) else 0,
        'naive_mae': naive_mae,
        'improvement': improvement,
        'verdict': verdict,
        'passed': passed,
    }


def main():
    print("="*70)
    print("  z1309: SCIENTIFIC FALSIFICATION TESTS")
    print("  Attempting to DISPROVE our embodied consciousness claims")
    print("="*70)
    print()
    print("If ANY test fails, we must revise our claims.")
    print()

    gpu = GPUSensor()
    results = []

    # Run all falsification tests
    results.append(test_trivial_solution(gpu))
    torch.cuda.empty_cache()

    results.append(test_random_projection(gpu))
    torch.cuda.empty_cache()

    results.append(test_confounded_comparison(gpu))
    torch.cuda.empty_cache()

    results.append(test_shortcut_learning(gpu))
    torch.cuda.empty_cache()

    results.append(test_consciousness_claim(gpu))
    torch.cuda.empty_cache()

    # Summary
    print("\n" + "="*70)
    print("FALSIFICATION SUMMARY")
    print("="*70 + "\n")

    passed = 0
    failed = 0

    for r in results:
        status = "✅ PASSED" if r['passed'] else "❌ FAILED"
        print(f"{r['test']}: {status}")
        print(f"   {r['verdict']}")
        if r['passed']:
            passed += 1
        else:
            failed += 1

    print(f"\n{'='*70}")
    print(f"TESTS PASSED: {passed}/{len(results)}")
    print(f"TESTS FAILED: {failed}/{len(results)}")
    print(f"{'='*70}")

    if failed == 0:
        final_verdict = "ALL CLAIMS SURVIVE FALSIFICATION"
        print(f"\n✅ {final_verdict}")
        print("   Our embodied consciousness claims are scientifically robust.")
    elif failed <= 2:
        final_verdict = "CLAIMS PARTIALLY FALSIFIED"
        print(f"\n⚠️  {final_verdict}")
        print("   Some claims need revision. See failed tests above.")
    else:
        final_verdict = "CLAIMS LARGELY FALSIFIED"
        print(f"\n❌ {final_verdict}")
        print("   Major revision needed. Most tests failed.")

    # Save results
    output = {
        'experiment': 'z1309_falsification_tests',
        'timestamp': datetime.now().isoformat(),
        'tests': results,
        'passed': passed,
        'failed': failed,
        'final_verdict': final_verdict,
    }

    output_path = Path(__file__).parent.parent / 'results' / 'z1309_falsification_tests.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)

    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == '__main__':
    main()
