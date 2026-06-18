#!/usr/bin/env python3
"""
v7.12 Design Test - Fix class collapse and test before training.

Key fixes:
1. Balanced 5-class categories based on ACTUAL varying signals
2. Anthropomorphized feeling names (curious, focused, strained, urgent, overwhelmed)
3. Proper interventional causality test
4. Heavy penalty for over-prediction
5. Decouple class advantage from intensity
"""

import torch
import numpy as np
from dataclasses import dataclass
from enum import IntEnum
from typing import Tuple, List, Dict
import random

# ============================================================================
# NEW FEELING CATEGORIES (based on actual signal variance analysis)
# ============================================================================

class Feeling(IntEnum):
    """5 feelings based on composite GPU state - anthropomorphized."""
    CURIOUS = 0      # Low activity, exploring (was OK)
    FOCUSED = 1      # Moderate load, productive (was WARM)
    STRAINED = 2     # High load, working hard (was HOT)
    URGENT = 3       # Very high, need to cool down (was REST/throttle)
    OVERWHELMED = 4  # Critical, must stop (was CRITICAL)

FEELING_NAMES = ["CURIOUS", "FOCUSED", "STRAINED", "URGENT", "OVERWHELMED"]
FEELING_TOKENS = [f"<|FEEL_{f}|>" for f in FEELING_NAMES]

# ============================================================================
# SIGNAL-BASED CATEGORIZATION (using actual varying dimensions)
# ============================================================================

@dataclass
class GPUState:
    """GPU state with key varying signals."""
    temp: float          # z[0], z[1] - varies 47-90C
    power: float         # z[4], z[5] - varies 32-141W
    power_delta: float   # z[3] - highest variance!
    gfx_activity: float  # z[10], z[11] - varies 2-99%
    vcn_activity: float  # z[14] - varies 27-100%

    @classmethod
    def random_balanced(cls, feeling: Feeling) -> "GPUState":
        """Generate GPU state that should map to specific feeling."""
        # Design ranges so each feeling has unique signature
        if feeling == Feeling.CURIOUS:
            return cls(
                temp=random.uniform(45, 55),
                power=random.uniform(30, 50),
                power_delta=random.uniform(-0.5, 0),
                gfx_activity=random.uniform(0, 20),
                vcn_activity=random.uniform(0, 30)
            )
        elif feeling == Feeling.FOCUSED:
            return cls(
                temp=random.uniform(55, 68),
                power=random.uniform(50, 80),
                power_delta=random.uniform(-0.2, 0.2),
                gfx_activity=random.uniform(20, 50),
                vcn_activity=random.uniform(30, 50)
            )
        elif feeling == Feeling.STRAINED:
            return cls(
                temp=random.uniform(68, 78),
                power=random.uniform(80, 110),
                power_delta=random.uniform(0, 0.5),
                gfx_activity=random.uniform(50, 75),
                vcn_activity=random.uniform(50, 70)
            )
        elif feeling == Feeling.URGENT:
            return cls(
                temp=random.uniform(78, 85),
                power=random.uniform(100, 130),
                power_delta=random.uniform(0.3, 0.8),
                gfx_activity=random.uniform(75, 90),
                vcn_activity=random.uniform(70, 85)
            )
        else:  # OVERWHELMED
            return cls(
                temp=random.uniform(85, 95),
                power=random.uniform(120, 150),
                power_delta=random.uniform(0.5, 1.0),
                gfx_activity=random.uniform(85, 100),
                vcn_activity=random.uniform(80, 100)
            )

    def classify(self) -> Feeling:
        """Classify GPU state into feeling based on composite score."""
        # Compute stress score from multiple signals
        temp_score = (self.temp - 45) / 50  # 0-1 scale
        power_score = (self.power - 30) / 120
        delta_score = (self.power_delta + 1) / 2  # -1 to 1 -> 0 to 1
        gfx_score = self.gfx_activity / 100
        vcn_score = self.vcn_activity / 100

        # Weighted composite (power_delta has highest variance, weight it more)
        composite = (
            0.20 * temp_score +
            0.15 * power_score +
            0.30 * delta_score +  # Highest weight - most varying signal
            0.20 * gfx_score +
            0.15 * vcn_score
        )

        # Map to feelings
        if composite < 0.20:
            return Feeling.CURIOUS
        elif composite < 0.40:
            return Feeling.FOCUSED
        elif composite < 0.60:
            return Feeling.STRAINED
        elif composite < 0.80:
            return Feeling.URGENT
        else:
            return Feeling.OVERWHELMED

    def to_z_feel(self, z_dim: int = 32) -> np.ndarray:
        """Convert to z_feel vector - only use varying dimensions."""
        z = np.zeros(z_dim, dtype=np.float32)

        # Temperature (dims 0-2)
        temp_norm = (self.temp - 45) / 50
        z[0] = temp_norm
        z[1] = temp_norm ** 2
        z[2] = max(0, temp_norm - 0.5)  # High temp indicator

        # Power delta (dim 3) - HIGHEST VARIANCE, keep prominent
        z[3] = self.power_delta

        # Power (dims 4-6)
        power_norm = (self.power - 30) / 120
        z[4] = power_norm
        z[5] = power_norm ** 2
        z[6] = max(0, power_norm - 0.6)  # High power indicator

        # Activity (dims 10-15)
        z[10] = self.gfx_activity / 100
        z[11] = (self.gfx_activity / 100) ** 2
        z[14] = self.vcn_activity / 100
        z[15] = (self.vcn_activity / 100) ** 2

        # Composite stress signal (dim 30)
        z[30] = float(self.classify()) / 4.0

        return z


# ============================================================================
# REWARD STRUCTURE - Heavy penalty for over-prediction
# ============================================================================

def compute_reward(pred: Feeling, true: Feeling, intensity_error: float) -> Tuple[float, float]:
    """
    Compute class and intensity rewards separately (for decoupled advantages).

    Key: HEAVY penalty for predicting more severe than reality (over-alarm).
    """
    # Class reward
    if pred == true:
        class_reward = 1.0
    else:
        diff = int(pred) - int(true)
        if diff > 0:
            # Over-prediction (predicted more severe than actual)
            # This is BAD - penalty scales with severity of over-prediction
            class_reward = -0.5 * diff  # -0.5, -1.0, -1.5, -2.0
        else:
            # Under-prediction (predicted less severe than actual)
            # Less bad, but still wrong
            class_reward = -0.3 * abs(diff)  # -0.3, -0.6, -0.9, -1.2

    # Intensity reward (simple MSE-based)
    intensity_reward = 1.0 - min(1.0, intensity_error * 2)

    return class_reward, intensity_reward


# ============================================================================
# INTERVENTIONAL CAUSALITY TEST
# ============================================================================

def interventional_causality_test(
    episodes: List[Dict],
    model_predict_fn,  # fn(z_feel) -> predicted_feeling
    n_swaps: int = 100
) -> Dict:
    """
    Proper interventional test: swap z_feel between episodes and measure
    how often prediction changes.

    Returns:
        - action_change_rate: how often does prediction change when z changes?
        - accuracy_with_real_z: accuracy with correct z_feel
        - accuracy_with_swapped_z: accuracy with swapped z_feel
        - kl_divergence: average KL between real and swapped action distributions
    """
    if len(episodes) < 2:
        return {"error": "Need at least 2 episodes"}

    changes = 0
    correct_real = 0
    correct_swapped = 0

    for _ in range(n_swaps):
        # Pick two random episodes from different classes
        idx1, idx2 = random.sample(range(len(episodes)), 2)
        ep1, ep2 = episodes[idx1], episodes[idx2]

        # Predict with real z_feel
        pred_real = model_predict_fn(ep1['z_feel'])

        # Predict with swapped z_feel (use ep2's z_feel for ep1's prompt)
        pred_swapped = model_predict_fn(ep2['z_feel'])

        # Track changes
        if pred_real != pred_swapped:
            changes += 1

        # Track accuracy
        if pred_real == ep1['true_feeling']:
            correct_real += 1
        if pred_swapped == ep1['true_feeling']:
            correct_swapped += 1

    return {
        "action_change_rate": changes / n_swaps,
        "accuracy_real_z": correct_real / n_swaps,
        "accuracy_swapped_z": correct_swapped / n_swaps,
        "causality_score": (correct_real - correct_swapped) / n_swaps
    }


# ============================================================================
# TEST: BALANCED DATASET GENERATION
# ============================================================================

def test_balanced_generation():
    """Test that we can generate balanced data across all 5 feelings."""
    print("=" * 60)
    print("TEST 1: Balanced Dataset Generation")
    print("=" * 60)

    # Generate 100 samples per feeling
    samples_per_class = 100
    all_samples = []

    for feeling in Feeling:
        for _ in range(samples_per_class):
            state = GPUState.random_balanced(feeling)
            classified = state.classify()
            all_samples.append({
                'target': feeling,
                'classified': classified,
                'state': state,
                'z_feel': state.to_z_feel()
            })

    # Check classification accuracy
    correct = sum(1 for s in all_samples if s['target'] == s['classified'])
    print(f"Classification accuracy: {100*correct/len(all_samples):.1f}%")

    # Check per-class
    from collections import Counter
    for feeling in Feeling:
        class_samples = [s for s in all_samples if s['target'] == feeling]
        classified_as = Counter(s['classified'] for s in class_samples)
        print(f"  {FEELING_NAMES[feeling]}: {dict(classified_as)}")

    # Check z_feel variance
    z_feels = np.array([s['z_feel'] for s in all_samples])
    variances = np.var(z_feels, axis=0)
    print(f"\nz_feel variance (top 5 dims):")
    top_dims = np.argsort(variances)[::-1][:5]
    for dim in top_dims:
        print(f"  z[{dim}]: var={variances[dim]:.4f}")

    return all_samples


# ============================================================================
# TEST: REWARD STRUCTURE
# ============================================================================

def test_reward_structure():
    """Test that reward properly penalizes over-prediction."""
    print("\n" + "=" * 60)
    print("TEST 2: Reward Structure")
    print("=" * 60)

    print("\nReward matrix (rows=pred, cols=true):")
    print("        ", end="")
    for true in Feeling:
        print(f"{FEELING_NAMES[true][:6]:>8}", end="")
    print()

    for pred in Feeling:
        print(f"{FEELING_NAMES[pred][:6]:>8}", end="")
        for true in Feeling:
            r, _ = compute_reward(pred, true, 0.0)
            print(f"{r:>8.2f}", end="")
        print()

    # Check that over-prediction is heavily penalized
    print("\nKey checks:")
    r_over, _ = compute_reward(Feeling.OVERWHELMED, Feeling.CURIOUS, 0.0)
    r_under, _ = compute_reward(Feeling.CURIOUS, Feeling.OVERWHELMED, 0.0)
    print(f"  OVERWHELMED when CURIOUS (over-predict): {r_over:.2f}")
    print(f"  CURIOUS when OVERWHELMED (under-predict): {r_under:.2f}")
    print(f"  Over-prediction is {abs(r_over)/abs(r_under):.1f}x worse (should be >1)")


# ============================================================================
# TEST: SIMULATED TRAINING DYNAMICS
# ============================================================================

def test_training_dynamics():
    """Simulate training to check if collapse is prevented."""
    print("\n" + "=" * 60)
    print("TEST 3: Simulated Training Dynamics")
    print("=" * 60)

    # Simulate a model that always predicts one class
    for always_pred in Feeling:
        total_reward = 0
        n_samples = 500
        for _ in range(n_samples):
            true = Feeling(random.randint(0, 4))
            r, _ = compute_reward(always_pred, true, 0.0)
            total_reward += r
        avg = total_reward / n_samples
        print(f"  Always predict {FEELING_NAMES[always_pred]}: avg reward = {avg:.3f}")

    # Now simulate random prediction
    total_reward = 0
    for _ in range(n_samples):
        pred = Feeling(random.randint(0, 4))
        true = Feeling(random.randint(0, 4))
        r, _ = compute_reward(pred, true, 0.0)
        total_reward += r
    avg = total_reward / n_samples
    print(f"  Random prediction: avg reward = {avg:.3f}")

    # Simulate perfect prediction
    total_reward = 0
    for _ in range(n_samples):
        true = Feeling(random.randint(0, 4))
        r, _ = compute_reward(true, true, 0.0)  # Perfect
        total_reward += r
    avg = total_reward / n_samples
    print(f"  Perfect prediction: avg reward = {avg:.3f}")


# ============================================================================
# TEST: Z_FEEL SIGNAL SEPARATION
# ============================================================================

def test_z_feel_separation():
    """Test that z_feel clearly separates different feelings."""
    print("\n" + "=" * 60)
    print("TEST 4: Z_feel Signal Separation")
    print("=" * 60)

    # Generate samples for each feeling
    samples_per_class = 200
    z_by_feeling = {}

    for feeling in Feeling:
        z_list = []
        for _ in range(samples_per_class):
            state = GPUState.random_balanced(feeling)
            z_list.append(state.to_z_feel())
        z_by_feeling[feeling] = np.array(z_list)

    # Compute mean z per feeling for key dimensions
    key_dims = [0, 3, 4, 10, 14, 30]
    print("\nMean z_feel values by feeling:")
    print("Feeling       ", end="")
    for dim in key_dims:
        print(f"   z[{dim}]", end="")
    print()

    for feeling in Feeling:
        print(f"{FEELING_NAMES[feeling]:12}", end="")
        z_mean = z_by_feeling[feeling].mean(axis=0)
        for dim in key_dims:
            print(f"{z_mean[dim]:>7.3f}", end="")
        print()

    # Check separability (linear probe accuracy)
    X = np.vstack([z_by_feeling[f] for f in Feeling])
    y = np.concatenate([np.full(samples_per_class, int(f)) for f in Feeling])

    # Simple centroid classifier
    centroids = {f: z_by_feeling[f].mean(axis=0) for f in Feeling}

    correct = 0
    for i, (x, true_y) in enumerate(zip(X, y)):
        dists = {f: np.linalg.norm(x - centroids[f]) for f in Feeling}
        pred = min(dists, key=dists.get)
        if int(pred) == true_y:
            correct += 1

    print(f"\nCentroid classifier accuracy: {100*correct/len(y):.1f}%")
    print("(Should be >60% for good separation)")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("v7.12 Design Tests")
    print("=" * 60)

    test_balanced_generation()
    test_reward_structure()
    test_training_dynamics()
    test_z_feel_separation()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
Key design decisions for v7.12:
1. 5 feelings: CURIOUS, FOCUSED, STRAINED, URGENT, OVERWHELMED
2. Based on ACTUAL varying signals (power_delta, gfx, vcn)
3. Over-prediction penalty is ~1.7x worse than under-prediction
4. z_feel uses only non-dead dimensions
5. Balanced synthetic data generation per feeling

Next: Implement v7.12 training script with these fixes.
""")
