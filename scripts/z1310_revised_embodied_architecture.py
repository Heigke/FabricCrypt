#!/usr/bin/env python3
"""
z1310: Revised Embodied Architecture

Addressing falsification failures:
1. SHORTCUT FIX: Feature dropout forces multi-feature learning
2. TEMPORAL FIX: GRU maintains temporal state for real prediction
3. HONEST METRIC: Must beat naive (t-1) prediction, not just classify

The goal: Achieve TEMPORAL PREDICTION better than a thermostat.
If we can predict future states better than "just use previous value",
we have something beyond reactive state-tracking.

NO CHEATING:
- Test on held-out temporal sequences
- Compare to naive baseline
- Verify all features contribute
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
    """GPU telemetry with history"""
    def __init__(self, history_len: int = 64):
        self.card = '/sys/class/drm/card1/device'
        self._history = deque(maxlen=history_len)
        self._timestamps = deque(maxlen=history_len)

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

    def sense(self) -> np.ndarray:
        """Get current state as numpy array"""
        state = np.array([
            self._hwmon('temp1_input', 50000) / 1000 / 100.0,  # temp normalized
            self._hwmon('power1_average', 50e6) / 1e6 / 100.0,  # power normalized
            self._read('gpu_busy_percent', 50) / 100.0,  # util
            0.0,  # will be filled with derivative
        ], dtype=np.float32)

        # Compute temperature derivative if we have history
        if len(self._history) > 0:
            prev_temp = self._history[-1][0]
            prev_time = self._timestamps[-1]
            dt = time.time() - prev_time
            if dt > 0.001:
                state[3] = (state[0] - prev_temp) / dt  # temp derivative
                state[3] = np.clip(state[3], -0.1, 0.1)  # clip to reasonable range

        self._history.append(state.copy())
        self._timestamps.append(time.time())

        return state

    def get_history(self, n: int = 16) -> np.ndarray:
        """Get last n states as [n, 4] array"""
        if len(self._history) < n:
            # Pad with current state
            current = self.sense()
            padding = [current] * (n - len(self._history))
            history = list(padding) + list(self._history)
        else:
            history = list(self._history)[-n:]
        return np.array(history, dtype=np.float32)


class TemporalEmbodiedModel(nn.Module):
    """
    Revised architecture with TEMPORAL modeling.

    Key changes:
    1. GRU processes temporal sequence (not just current state)
    2. Feature dropout prevents shortcut learning
    3. Multi-step prediction head
    4. Separate confidence per prediction horizon
    """

    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 128,
        gru_layers: int = 2,
        prediction_horizons: List[int] = [1, 3, 5],
        feature_dropout: float = 0.3,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.prediction_horizons = prediction_horizons
        self.feature_dropout = feature_dropout

        # Feature-wise dropout (prevents shortcut learning)
        self.feat_dropout = nn.Dropout(feature_dropout)

        # Input projection with per-feature weights
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Temporal modeling via GRU (no dropout - AMD MIOpen issue)
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=gru_layers,
            batch_first=True,
            dropout=0,  # Disabled due to AMD MIOpen compatibility
        )

        # Multi-horizon prediction heads
        self.prediction_heads = nn.ModuleDict()
        self.confidence_heads = nn.ModuleDict()

        for h in prediction_horizons:
            self.prediction_heads[f'h{h}'] = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, input_dim),
            )
            self.confidence_heads[f'h{h}'] = nn.Sequential(
                nn.Linear(hidden_dim, 32),
                nn.GELU(),
                nn.Linear(32, 1),
                nn.Sigmoid(),
            )

        # Current state reconstruction (for self-modeling)
        self.self_model = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(
        self,
        sequence: torch.Tensor,
        hidden: torch.Tensor = None,
        apply_feat_dropout: bool = True,
    ) -> Dict:
        """
        Forward pass with temporal sequence.

        Args:
            sequence: [B, T, input_dim] temporal sequence
            hidden: Optional GRU hidden state
            apply_feat_dropout: Whether to apply feature dropout

        Returns:
            Dictionary with predictions, confidences, hidden state
        """
        B, T, D = sequence.shape

        # Feature dropout (forces multi-feature learning)
        if apply_feat_dropout and self.training:
            # Randomly zero out entire features
            mask = torch.ones(B, 1, D, device=sequence.device)
            for b in range(B):
                if torch.rand(1).item() < self.feature_dropout:
                    drop_idx = torch.randint(0, D, (1,)).item()
                    mask[b, 0, drop_idx] = 0
            sequence = sequence * mask

        # Project input
        x = self.input_proj(sequence)

        # Temporal processing
        gru_out, hidden_new = self.gru(x, hidden)

        # Use final hidden state for predictions
        final_h = gru_out[:, -1, :]  # [B, hidden_dim]

        # Multi-horizon predictions
        predictions = {}
        confidences = {}

        for h in self.prediction_horizons:
            key = f'h{h}'
            predictions[key] = self.prediction_heads[key](final_h)
            confidences[key] = self.confidence_heads[key](final_h)

        # Self-model (reconstruct current state)
        current_pred = self.self_model(final_h)

        return {
            'predictions': predictions,
            'confidences': confidences,
            'current_pred': current_pred,
            'hidden': hidden_new,
            'final_h': final_h,
        }


def train_temporal_model(
    model: TemporalEmbodiedModel,
    gpu: GPUSensor,
    n_epochs: int = 30,
    seq_len: int = 16,
    steps_per_epoch: int = 100,
) -> Dict:
    """
    Train with proper temporal prediction.

    Key: We collect FUTURE states and train to predict them.
    This is NOT just classification - it's actual prediction.
    """
    print(f"\n{'='*60}")
    print("TRAINING TEMPORAL EMBODIED MODEL")
    print(f"{'='*60}")
    print(f"Sequence length: {seq_len}")
    print(f"Prediction horizons: {model.prediction_horizons}")
    print(f"Feature dropout: {model.feature_dropout}")
    print()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)

    history = []
    max_horizon = max(model.prediction_horizons)

    # Warm up sensor history
    print("Warming up sensor history...")
    for _ in range(seq_len + max_horizon + 10):
        gpu.sense()
        if np.random.random() < 0.3:
            _ = torch.randn(500, 500, device=DEVICE) @ torch.randn(500, 500, device=DEVICE)
        time.sleep(0.02)

    model.train()

    for epoch in range(n_epochs):
        epoch_losses = {f'h{h}': [] for h in model.prediction_horizons}
        epoch_losses['current'] = []

        for step in range(steps_per_epoch):
            # Create GPU load variation
            if step % 3 == 0:
                intensity = np.random.choice([500, 1000, 1500, 2000])
                _ = torch.randn(intensity, intensity, device=DEVICE) @ torch.randn(intensity, intensity, device=DEVICE)

            # Get current sequence
            seq = gpu.get_history(seq_len)
            seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)

            # Collect future states for each horizon
            future_states = {}
            for h in model.prediction_horizons:
                # Wait and collect future state
                for _ in range(h):
                    if np.random.random() < 0.2:
                        _ = torch.randn(300, 300, device=DEVICE) @ torch.randn(300, 300, device=DEVICE)
                    time.sleep(0.015)
                    gpu.sense()

                future = gpu.sense()
                future_states[f'h{h}'] = torch.tensor(future, dtype=torch.float32).unsqueeze(0).to(DEVICE)

            # Forward pass
            out = model(seq_tensor, apply_feat_dropout=True)

            # Compute losses for each horizon
            total_loss = 0
            for h in model.prediction_horizons:
                key = f'h{h}'
                pred = out['predictions'][key]
                target = future_states[key]
                loss = F.mse_loss(pred, target)
                epoch_losses[key].append(loss.item())
                total_loss = total_loss + loss

            # Current state loss
            current_target = seq_tensor[:, -1, :]
            current_loss = F.mse_loss(out['current_pred'], current_target)
            epoch_losses['current'].append(current_loss.item())
            total_loss = total_loss + current_loss * 0.5

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()

        # Log epoch stats
        avg_losses = {k: np.mean(v) for k, v in epoch_losses.items()}
        history.append({'epoch': epoch, **avg_losses})

        print(f"Epoch {epoch+1:2d}: " + " | ".join(f"{k}={v:.4f}" for k, v in avg_losses.items()))

    return {'history': history}


def evaluate_vs_naive(
    model: TemporalEmbodiedModel,
    gpu: GPUSensor,
    n_samples: int = 100,
    seq_len: int = 16,
) -> Dict:
    """
    THE HONEST TEST: Compare to naive (t-1) prediction.

    If we can't beat "just use the previous value", we have nothing
    beyond a thermostat.
    """
    print(f"\n{'='*60}")
    print("EVALUATION: Model vs Naive (t-1) Baseline")
    print(f"{'='*60}\n")

    model.eval()
    max_horizon = max(model.prediction_horizons)

    results = {f'h{h}': {'model': [], 'naive': [], 'actuals': []}
               for h in model.prediction_horizons}

    # Warm up
    for _ in range(seq_len + 5):
        gpu.sense()
        time.sleep(0.02)

    print("Collecting prediction samples...")

    for i in range(n_samples):
        # Get sequence
        seq = gpu.get_history(seq_len)
        seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        # Model predictions
        with torch.no_grad():
            out = model(seq_tensor, apply_feat_dropout=False)

        # Naive baseline: use last value in sequence
        naive_pred = seq[-1]

        # Collect actual future states
        for h in model.prediction_horizons:
            for _ in range(h):
                if np.random.random() < 0.25:
                    _ = torch.randn(500, 500, device=DEVICE) @ torch.randn(500, 500, device=DEVICE)
                time.sleep(0.02)
                gpu.sense()

            actual = gpu.sense()
            key = f'h{h}'

            results[key]['model'].append(out['predictions'][key].cpu().numpy().squeeze())
            results[key]['naive'].append(naive_pred)
            results[key]['actuals'].append(actual)

        # Progress
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{n_samples} samples collected")

    # Compute metrics
    print("\nResults:")
    print("-" * 70)
    print(f"{'Horizon':<10} | {'Model MAE':<12} | {'Naive MAE':<12} | {'Improvement':<12} | {'Winner'}")
    print("-" * 70)

    summary = {}
    model_wins = 0
    naive_wins = 0

    for h in model.prediction_horizons:
        key = f'h{h}'

        model_preds = np.array(results[key]['model'])
        naive_preds = np.array(results[key]['naive'])
        actuals = np.array(results[key]['actuals'])

        model_mae = np.mean(np.abs(model_preds - actuals))
        naive_mae = np.mean(np.abs(naive_preds - actuals))

        improvement = (naive_mae - model_mae) / naive_mae * 100

        if model_mae < naive_mae:
            winner = "MODEL ✓"
            model_wins += 1
        else:
            winner = "naive"
            naive_wins += 1

        print(f"t+{h:<8} | {model_mae:<12.4f} | {naive_mae:<12.4f} | {improvement:+11.1f}% | {winner}")

        summary[key] = {
            'model_mae': model_mae,
            'naive_mae': naive_mae,
            'improvement': improvement,
            'model_wins': model_mae < naive_mae,
        }

    print("-" * 70)
    print(f"Model wins: {model_wins}/{len(model.prediction_horizons)} horizons")

    return summary


def evaluate_feature_usage(
    model: TemporalEmbodiedModel,
    gpu: GPUSensor,
    seq_len: int = 16,
) -> Dict:
    """
    SHORTCUT TEST: Verify all features are used.
    """
    print(f"\n{'='*60}")
    print("FEATURE USAGE ANALYSIS")
    print(f"{'='*60}\n")

    model.eval()
    feature_names = ['temperature', 'power', 'utilization', 'temp_derivative']

    # Collect baseline predictions
    baseline_preds = []
    for _ in range(30):
        seq = gpu.get_history(seq_len)
        seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            out = model(seq_tensor, apply_feat_dropout=False)
            baseline_preds.append(out['predictions']['h1'].cpu().numpy())

        if np.random.random() < 0.3:
            _ = torch.randn(800, 800, device=DEVICE) @ torch.randn(800, 800, device=DEVICE)
        time.sleep(0.03)

    baseline_preds = np.array(baseline_preds).squeeze()

    # Ablate each feature
    importances = []

    for feat_idx in range(4):
        ablated_preds = []

        for _ in range(30):
            seq = gpu.get_history(seq_len)
            # Zero out this feature across entire sequence
            seq[:, feat_idx] = 0.5  # Neutral value

            seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                out = model(seq_tensor, apply_feat_dropout=False)
                ablated_preds.append(out['predictions']['h1'].cpu().numpy())

            if np.random.random() < 0.3:
                _ = torch.randn(800, 800, device=DEVICE) @ torch.randn(800, 800, device=DEVICE)
            time.sleep(0.03)

        ablated_preds = np.array(ablated_preds).squeeze()

        # Impact = how much predictions change when feature is removed
        impact = np.mean(np.abs(ablated_preds - baseline_preds))
        importances.append(impact)

    # Normalize
    total = sum(importances)
    normalized = [imp / total for imp in importances]

    print("Feature importance (ablation impact):")
    max_imp = 0
    for name, imp in zip(feature_names, normalized):
        bar = "█" * int(imp * 40)
        print(f"  {name:<15}: {imp:5.1%} {bar}")
        max_imp = max(max_imp, imp)

    # Check for shortcut
    if max_imp > 0.5:
        verdict = f"WARNING: Single feature dominates ({max_imp:.1%})"
        balanced = False
    elif max_imp > 0.4:
        verdict = f"ACCEPTABLE: Slight imbalance ({max_imp:.1%})"
        balanced = True
    else:
        verdict = f"GOOD: Features balanced (max {max_imp:.1%})"
        balanced = True

    print(f"\n{verdict}")

    return {
        'importances': dict(zip(feature_names, normalized)),
        'max_importance': max_imp,
        'balanced': balanced,
    }


def evaluate_self_awareness(
    model: TemporalEmbodiedModel,
    gpu: GPUSensor,
    seq_len: int = 16,
) -> Dict:
    """
    STATE CLASSIFICATION: Can model distinguish calm vs stressed?
    (This is what we CAN claim - state classification, not consciousness)
    """
    print(f"\n{'='*60}")
    print("STATE CLASSIFICATION (Calm vs Stressed)")
    print(f"{'='*60}\n")

    model.eval()

    calm_hidden = []
    stressed_hidden = []

    print("Collecting calm samples...")
    for _ in range(40):
        seq = gpu.get_history(seq_len)
        seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            out = model(seq_tensor, apply_feat_dropout=False)
            calm_hidden.append(out['final_h'].cpu().numpy())

        time.sleep(0.04)

    print("Collecting stressed samples...")
    for _ in range(40):
        # Create stress
        stress = torch.randn(2000, 2000, device=DEVICE)
        _ = stress @ stress.T @ stress
        del stress

        seq = gpu.get_history(seq_len)
        seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            out = model(seq_tensor, apply_feat_dropout=False)
            stressed_hidden.append(out['final_h'].cpu().numpy())

        torch.cuda.empty_cache()

    calm_h = np.array(calm_hidden).squeeze()
    stressed_h = np.array(stressed_hidden).squeeze()

    # Linear classification
    all_h = np.vstack([calm_h, stressed_h])
    labels = np.array([0] * len(calm_h) + [1] * len(stressed_h))

    calm_c = calm_h.mean(axis=0)
    stressed_c = stressed_h.mean(axis=0)
    direction = stressed_c - calm_c
    direction /= np.linalg.norm(direction) + 1e-6

    proj = all_h @ direction
    thresh = proj.mean()
    preds = (proj > thresh).astype(int)

    accuracy = (preds == labels).mean()
    separation = np.linalg.norm(calm_c - stressed_c)

    print(f"\nClassification accuracy: {accuracy:.1%}")
    print(f"State separation: {separation:.4f}")

    return {
        'accuracy': accuracy,
        'separation': separation,
    }


def main():
    print("=" * 70)
    print("  z1310: REVISED EMBODIED ARCHITECTURE")
    print("  Addressing falsification failures honestly")
    print("=" * 70)
    print()
    print("Key changes:")
    print("  1. GRU for temporal modeling (not just reactive)")
    print("  2. Feature dropout (prevents shortcuts)")
    print("  3. Multi-horizon prediction (must beat naive)")
    print()

    gpu = GPUSensor()

    # Create model
    model = TemporalEmbodiedModel(
        input_dim=4,
        hidden_dim=128,
        gru_layers=2,
        prediction_horizons=[1, 3, 5],
        feature_dropout=0.3,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    print(f"Device: {DEVICE}")

    # Train
    train_results = train_temporal_model(
        model, gpu,
        n_epochs=30,
        seq_len=16,
        steps_per_epoch=80,
    )

    # Evaluate
    naive_comparison = evaluate_vs_naive(model, gpu, n_samples=80, seq_len=16)
    feature_analysis = evaluate_feature_usage(model, gpu, seq_len=16)
    state_classification = evaluate_self_awareness(model, gpu, seq_len=16)

    # Final verdict
    print("\n" + "=" * 70)
    print("FINAL ASSESSMENT")
    print("=" * 70)

    # Count wins vs naive
    wins = sum(1 for k, v in naive_comparison.items() if v['model_wins'])
    total = len(naive_comparison)

    print(f"\n1. TEMPORAL PREDICTION: Model wins {wins}/{total} horizons vs naive")

    if wins >= 2:
        print("   ✅ PASSES: Better than thermostat at prediction")
        temporal_pass = True
    else:
        print("   ❌ FAILS: Not better than thermostat")
        temporal_pass = False

    print(f"\n2. FEATURE BALANCE: Max importance = {feature_analysis['max_importance']:.1%}")
    if feature_analysis['balanced']:
        print("   ✅ PASSES: No single-feature shortcut")
        feature_pass = True
    else:
        print("   ❌ FAILS: Shortcut detected")
        feature_pass = False

    print(f"\n3. STATE CLASSIFICATION: {state_classification['accuracy']:.1%} accuracy")
    if state_classification['accuracy'] > 0.75:
        print("   ✅ PASSES: Can distinguish calm vs stressed")
        class_pass = True
    else:
        print("   ❌ FAILS: Cannot distinguish states")
        class_pass = False

    # Overall
    all_pass = temporal_pass and feature_pass and class_pass

    print("\n" + "=" * 70)
    if all_pass:
        print("✅ ALL TESTS PASS")
        print()
        print("LEGITIMATE CLAIMS:")
        print("  • Temporal prediction beyond naive baseline")
        print("  • Multi-feature embodied state modeling")
        print("  • Calm/stressed state classification")
        print()
        print("CANNOT CLAIM:")
        print("  • 'Consciousness' or 'self-awareness' (too strong)")
        print("  • Claims should be: 'Temporal embodied state prediction'")
        verdict = "TEMPORAL EMBODIED STATE PREDICTOR (legitimate)"
    else:
        failed = []
        if not temporal_pass:
            failed.append("temporal prediction")
        if not feature_pass:
            failed.append("feature balance")
        if not class_pass:
            failed.append("state classification")

        print(f"❌ FAILED: {', '.join(failed)}")
        verdict = f"FAILED: {', '.join(failed)}"

    print("=" * 70)

    # Save results
    output = {
        'experiment': 'z1310_revised_embodied_architecture',
        'timestamp': datetime.now().isoformat(),
        'training': train_results,
        'naive_comparison': {k: {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv
                                  for kk, vv in v.items()}
                             for k, v in naive_comparison.items()},
        'feature_analysis': {k: float(v) if isinstance(v, (np.floating, float)) else v
                             for k, v in feature_analysis.items()},
        'state_classification': {k: float(v) for k, v in state_classification.items()},
        'tests_passed': {
            'temporal': temporal_pass,
            'feature_balance': feature_pass,
            'classification': class_pass,
        },
        'all_pass': all_pass,
        'verdict': verdict,
    }

    output_path = Path(__file__).parent.parent / 'results' / 'z1310_revised_embodied_architecture.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to: {output_path}")

    return output


if __name__ == '__main__':
    main()
