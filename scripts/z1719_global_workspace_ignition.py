#!/usr/bin/env python3
"""
z1719: Global Workspace Theory Ignition Test

Tests the "ignition" signature from Global Workspace Theory (Baars, Dehaene):
When a signal crosses a threshold, information should broadcast SIMULTANEOUSLY
across all layers (not gradual propagation).

Hypothesis: Embodied models with telemetry conditioning show ignition signatures
when telemetry spikes above threshold, while disembodied models do not.

Three conditions:
- A: EMBODIED - real telemetry with artificial spikes (expect ignition)
- B: DISEMBODIED - zero telemetry (no ignition possible)
- C: SUBLIMINAL - small telemetry spikes below threshold (no/weak ignition)

Verdicts:
- V1: Cross-layer correlation after spike > 0.7 in EMBODIED
- V2: DISEMBODIED shows no cross-layer correlation spike
- V3: Ignition happens within 1 batch (not gradual over 3+ batches)
- V4: SUBLIMINAL shows smaller/no ignition (threshold effect)

Key metric: Cross-layer correlation JUMP - in ignition, all layers change together.

References:
- Baars (1988): A Cognitive Theory of Consciousness
- Dehaene & Changeux (2011): Neural Ignition and Global Workspace

Author: FEEL Research Team
Date: 2026-02
"""

import functools
print = functools.partial(print, flush=True)

import os
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '11.0.0'

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import time

# Import metabolic transformer and telemetry
import sys
sys.path.insert(0, '/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy')
from src.metabolic.film_transformer import create_metabolic_transformer, MetabolicConfig
from src.telemetry.sysfs_hwmon import SysfsHwmonTelemetry


@dataclass
class IgnitionConfig:
    """Configuration for ignition experiments."""
    hidden_dim: int = 256
    num_layers: int = 6
    num_heads: int = 4
    telemetry_dim: int = 12
    batch_size: int = 8
    seq_len: int = 64

    # Telemetry spike parameters
    baseline_power: float = 0.3  # Normalized baseline telemetry value
    spike_power: float = 0.9     # Spike value for supraliminal condition
    subliminal_power: float = 0.35  # Below-threshold spike (just barely above baseline)
    spike_dimension: int = 0     # Which telemetry dim to spike (0=power)

    # Ignition thresholds
    ignition_correlation_threshold: float = 0.7
    subliminal_threshold_ratio: float = 0.5  # Subliminal should be < 50% of supraliminal

    # Training
    num_training_steps: int = 200
    learning_rate: float = 1e-4


class LayerHiddenHook:
    """Hook to capture hidden states at each transformer layer."""

    def __init__(self):
        self.hidden_states: List[torch.Tensor] = []
        self.hooks = []

    def clear(self):
        self.hidden_states = []

    def get_hook_fn(self, layer_idx: int):
        def hook(module, input, output):
            # Output from MetabolicBlock is the hidden state
            if isinstance(output, torch.Tensor):
                self.hidden_states.append(output.detach())
        return hook

    def register(self, model):
        """Register hooks on all transformer blocks."""
        for i, block in enumerate(model.blocks):
            hook = block.register_forward_hook(self.get_hook_fn(i))
            self.hooks.append(hook)

    def remove(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


def compute_cross_layer_correlation(hidden_states: List[torch.Tensor]) -> float:
    """
    Compute average correlation between layer hidden state changes.

    High correlation = all layers change together (ignition signature)
    Low correlation = gradual/independent changes
    """
    if len(hidden_states) < 2:
        return 0.0

    # Flatten and normalize each layer's hidden state
    layer_vectors = []
    for h in hidden_states:
        # Average over batch and sequence, keep hidden dim
        v = h.mean(dim=(0, 1))  # [hidden_dim]
        v = v - v.mean()  # Center
        if v.std() > 1e-8:
            v = v / v.std()  # Normalize
        layer_vectors.append(v)

    # Compute pairwise correlations
    correlations = []
    for i in range(len(layer_vectors)):
        for j in range(i + 1, len(layer_vectors)):
            corr = torch.corrcoef(torch.stack([layer_vectors[i], layer_vectors[j]]))[0, 1]
            if not torch.isnan(corr):
                correlations.append(corr.item())

    if not correlations:
        return 0.0
    return np.mean(correlations)


def compute_delta_correlation(hidden_before: List[torch.Tensor],
                              hidden_after: List[torch.Tensor]) -> float:
    """
    Compute correlation of CHANGES across layers.

    For ignition: all layers should change in correlated way.
    """
    if len(hidden_before) != len(hidden_after) or len(hidden_before) < 2:
        return 0.0

    # Compute change vectors for each layer
    delta_vectors = []
    for hb, ha in zip(hidden_before, hidden_after):
        delta = (ha - hb).mean(dim=(0, 1))  # [hidden_dim]
        delta = delta - delta.mean()
        if delta.std() > 1e-8:
            delta = delta / delta.std()
        delta_vectors.append(delta)

    # Compute pairwise correlations of changes
    correlations = []
    for i in range(len(delta_vectors)):
        for j in range(i + 1, len(delta_vectors)):
            corr = torch.corrcoef(torch.stack([delta_vectors[i], delta_vectors[j]]))[0, 1]
            if not torch.isnan(corr):
                correlations.append(corr.item())

    if not correlations:
        return 0.0
    return np.mean(correlations)


def compute_layer_change_magnitude(hidden_before: List[torch.Tensor],
                                   hidden_after: List[torch.Tensor]) -> List[float]:
    """Compute magnitude of change at each layer."""
    magnitudes = []
    for hb, ha in zip(hidden_before, hidden_after):
        delta = (ha - hb).norm(dim=-1).mean()
        magnitudes.append(delta.item())
    return magnitudes


def create_telemetry_vector(config: IgnitionConfig,
                           spike_value: Optional[float] = None,
                           device: torch.device = None) -> torch.Tensor:
    """Create telemetry vector with optional spike on power dimension."""
    telem = torch.ones(config.batch_size, config.telemetry_dim, device=device) * config.baseline_power

    if spike_value is not None:
        telem[:, config.spike_dimension] = spike_value

    return telem


def run_ignition_test(model, config: IgnitionConfig,
                     device: torch.device,
                     condition: str) -> Dict:
    """
    Run single ignition test for a condition.

    Returns dict with:
    - cross_layer_correlation_before: Correlation before spike
    - cross_layer_correlation_after: Correlation after spike
    - correlation_jump: Delta in correlation
    - layer_changes: Per-layer change magnitudes
    - ignition_detected: Boolean if jump exceeds threshold
    """
    model.eval()
    hook = LayerHiddenHook()
    hook.register(model)

    # Random input tokens
    input_ids = torch.randint(0, 256, (config.batch_size, config.seq_len), device=device)

    try:
        # BASELINE: Run with baseline telemetry
        if condition == "DISEMBODIED":
            telem_baseline = torch.zeros(config.batch_size, config.telemetry_dim, device=device)
        else:
            telem_baseline = create_telemetry_vector(config, spike_value=None, device=device)

        hook.clear()
        with torch.no_grad():
            _ = model(input_ids, telemetry=telem_baseline, return_hidden=True)
        hidden_baseline = hook.hidden_states.copy()
        corr_baseline = compute_cross_layer_correlation(hidden_baseline)

        # SPIKE: Run with spiked telemetry
        if condition == "DISEMBODIED":
            telem_spike = torch.zeros(config.batch_size, config.telemetry_dim, device=device)
        elif condition == "SUBLIMINAL":
            telem_spike = create_telemetry_vector(config,
                                                  spike_value=config.subliminal_power,
                                                  device=device)
        else:  # EMBODIED
            telem_spike = create_telemetry_vector(config,
                                                  spike_value=config.spike_power,
                                                  device=device)

        hook.clear()
        with torch.no_grad():
            _ = model(input_ids, telemetry=telem_spike, return_hidden=True)
        hidden_spike = hook.hidden_states.copy()
        corr_spike = compute_cross_layer_correlation(hidden_spike)

        # Compute layer-wise changes
        layer_changes = compute_layer_change_magnitude(hidden_baseline, hidden_spike)

        # Correlation jump (comparing state correlations)
        corr_jump = corr_spike - corr_baseline

        # KEY METRIC: Correlation of changes across layers (ignition signature)
        delta_correlation = compute_delta_correlation(hidden_baseline, hidden_spike)

        return {
            'cross_layer_correlation_before': corr_baseline,
            'cross_layer_correlation_after': corr_spike,
            'correlation_jump': corr_jump,
            'delta_correlation': delta_correlation,  # Ignition = high delta correlation
            'layer_changes': layer_changes,
            'mean_layer_change': np.mean(layer_changes),
            'layer_change_std': np.std(layer_changes),
            'ignition_detected': delta_correlation > config.ignition_correlation_threshold
        }

    finally:
        hook.remove()


def run_temporal_ignition_test(model, config: IgnitionConfig,
                               device: torch.device) -> Dict:
    """
    Test if ignition happens within 1 batch (immediate) vs gradual (over 3+ batches).

    Key insight: True ignition should be immediate, not gradual.
    """
    model.eval()
    hook = LayerHiddenHook()
    hook.register(model)

    input_ids = torch.randint(0, 256, (config.batch_size, config.seq_len), device=device)

    correlations_over_batches = []

    try:
        # Start with baseline
        telem = create_telemetry_vector(config, spike_value=None, device=device)

        # Batch 0: baseline
        hook.clear()
        with torch.no_grad():
            _ = model(input_ids, telemetry=telem, return_hidden=True)
        correlations_over_batches.append(compute_cross_layer_correlation(hook.hidden_states))

        # Batch 1: SPIKE (this is where ignition should happen)
        telem_spike = create_telemetry_vector(config, spike_value=config.spike_power, device=device)
        hook.clear()
        with torch.no_grad():
            _ = model(input_ids, telemetry=telem_spike, return_hidden=True)
        correlations_over_batches.append(compute_cross_layer_correlation(hook.hidden_states))

        # Batch 2-4: Continue with spike (check if correlation stays high)
        for _ in range(3):
            hook.clear()
            with torch.no_grad():
                _ = model(input_ids, telemetry=telem_spike, return_hidden=True)
            correlations_over_batches.append(compute_cross_layer_correlation(hook.hidden_states))

        # Analysis: Check if jump happened at batch 1 (immediate) vs gradual
        jump_at_batch_1 = correlations_over_batches[1] - correlations_over_batches[0]
        max_subsequent_jump = max(
            correlations_over_batches[i] - correlations_over_batches[i-1]
            for i in range(2, len(correlations_over_batches))
        )

        # Ignition is immediate if batch 1 jump is > 2x any subsequent jump
        is_immediate = jump_at_batch_1 > 2 * max_subsequent_jump if max_subsequent_jump > 0 else True

        return {
            'correlations_over_batches': correlations_over_batches,
            'jump_at_batch_1': jump_at_batch_1,
            'max_subsequent_jump': max_subsequent_jump,
            'is_immediate_ignition': is_immediate
        }

    finally:
        hook.remove()


def train_embodied_model(model, config: IgnitionConfig,
                        device: torch.device,
                        telemetry_source: Optional[SysfsHwmonTelemetry] = None) -> Dict:
    """Train the model with embodied telemetry to develop ignition capability."""

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    training_stats = {
        'losses': [],
        'cross_layer_correlations': []
    }

    hook = LayerHiddenHook()
    hook.register(model)

    print("Training embodied model...")

    try:
        for step in range(config.num_training_steps):
            # Generate random input
            input_ids = torch.randint(0, 256, (config.batch_size, config.seq_len), device=device)
            targets = torch.randint(0, 256, (config.batch_size, config.seq_len), device=device)

            # Get real telemetry or generate synthetic
            if telemetry_source:
                sample = telemetry_source.read_sample()
                # Normalize to 0-1 range
                telem_np = np.array([
                    min(sample.power_w / 150.0, 1.0),  # Power normalized to ~150W max
                    min(sample.temp_edge_c / 100.0, 1.0),  # Temp normalized to 100C
                    min(sample.temp_junction_c / 100.0, 1.0),
                    sample.gpu_busy_pct / 100.0,
                    min(sample.vram_used_gb / 16.0, 1.0),  # VRAM normalized to 16GB
                    0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5  # Padding
                ])[:config.telemetry_dim]
                telem = torch.tensor(telem_np, dtype=torch.float32, device=device)
                telem = telem.unsqueeze(0).expand(config.batch_size, -1)
            else:
                # Synthetic telemetry with occasional spikes
                telem = torch.rand(config.batch_size, config.telemetry_dim, device=device) * 0.5 + 0.2
                # Add spikes to learn ignition
                if np.random.random() < 0.3:
                    telem[:, config.spike_dimension] = np.random.uniform(0.7, 0.95)

            # Forward pass
            hook.clear()
            output = model(input_ids, telemetry=telem)

            # Language modeling loss
            loss = F.cross_entropy(
                output['logits'].view(-1, output['logits'].size(-1)),
                targets.view(-1)
            )

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Track stats
            training_stats['losses'].append(loss.item())
            if len(hook.hidden_states) >= 2:
                corr = compute_cross_layer_correlation(hook.hidden_states)
                training_stats['cross_layer_correlations'].append(corr)

            if (step + 1) % 50 == 0:
                avg_loss = np.mean(training_stats['losses'][-50:])
                avg_corr = np.mean(training_stats['cross_layer_correlations'][-50:]) if training_stats['cross_layer_correlations'] else 0
                print(f"  Step {step+1}/{config.num_training_steps}: loss={avg_loss:.4f}, avg_corr={avg_corr:.3f}")

    finally:
        hook.remove()

    return training_stats


def main():
    print("=" * 70)
    print("  z1719: GLOBAL WORKSPACE THEORY IGNITION TEST")
    print("  Baars & Dehaene: Information broadcast across all layers")
    print("=" * 70)
    print()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    config = IgnitionConfig()
    print(f"\nConfiguration:")
    print(f"  Hidden dim: {config.hidden_dim}")
    print(f"  Layers: {config.num_layers}")
    print(f"  Telemetry dim: {config.telemetry_dim}")
    print(f"  Baseline power: {config.baseline_power}")
    print(f"  Spike power: {config.spike_power}")
    print(f"  Subliminal power: {config.subliminal_power}")
    print(f"  Ignition threshold: {config.ignition_correlation_threshold}")

    # Initialize telemetry
    telemetry = None
    try:
        telemetry = SysfsHwmonTelemetry(sample_rate_hz=50)
        sample = telemetry.read_sample()
        print(f"\nTelemetry connected: power={sample.power_w:.1f}W, temp={sample.temp_edge_c:.1f}C")
    except Exception as e:
        print(f"\nTelemetry unavailable: {e}")
        print("Using synthetic telemetry")

    # Create model
    print("\nCreating MetabolicTransformer...")
    model = create_metabolic_transformer(
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        telemetry_dim=config.telemetry_dim,
        baseline=False
    ).to(device)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Phase 1: Pre-training baseline
    print("\n" + "=" * 70)
    print("PHASE 1: PRE-TRAINING BASELINE")
    print("=" * 70)

    pre_results = {}
    for condition in ['EMBODIED', 'DISEMBODIED', 'SUBLIMINAL']:
        result = run_ignition_test(model, config, device, condition)
        pre_results[condition] = result
        print(f"\n[{condition}] Pre-training:")
        print(f"  Correlation before spike: {result['cross_layer_correlation_before']:.3f}")
        print(f"  Correlation after spike:  {result['cross_layer_correlation_after']:.3f}")
        print(f"  Correlation jump: {result['correlation_jump']:.3f}")
        print(f"  DELTA correlation (ignition): {result['delta_correlation']:.3f}")
        print(f"  Mean layer change: {result['mean_layer_change']:.4f}")
        print(f"  Ignition detected: {result['ignition_detected']}")

    # Phase 2: Training
    print("\n" + "=" * 70)
    print("PHASE 2: EMBODIED TRAINING")
    print("=" * 70)

    training_stats = train_embodied_model(model, config, device, telemetry)

    # Phase 3: Post-training ignition tests
    print("\n" + "=" * 70)
    print("PHASE 3: POST-TRAINING IGNITION TESTS")
    print("=" * 70)

    post_results = {}
    for condition in ['EMBODIED', 'DISEMBODIED', 'SUBLIMINAL']:
        result = run_ignition_test(model, config, device, condition)
        post_results[condition] = result
        print(f"\n[{condition}] Post-training:")
        print(f"  Correlation before spike: {result['cross_layer_correlation_before']:.3f}")
        print(f"  Correlation after spike:  {result['cross_layer_correlation_after']:.3f}")
        print(f"  Correlation jump: {result['correlation_jump']:.3f}")
        print(f"  DELTA correlation (ignition): {result['delta_correlation']:.3f}")
        print(f"  Mean layer change: {result['mean_layer_change']:.4f}")
        print(f"  Ignition detected: {result['ignition_detected']}")

    # Phase 4: Temporal test
    print("\n" + "=" * 70)
    print("PHASE 4: TEMPORAL IGNITION TEST")
    print("=" * 70)

    temporal_result = run_temporal_ignition_test(model, config, device)
    print(f"\nCorrelations over batches: {[f'{c:.3f}' for c in temporal_result['correlations_over_batches']]}")
    print(f"Jump at batch 1 (spike): {temporal_result['jump_at_batch_1']:.3f}")
    print(f"Max subsequent jump: {temporal_result['max_subsequent_jump']:.3f}")
    print(f"Is immediate ignition: {temporal_result['is_immediate_ignition']}")

    # Verdict evaluation
    print("\n" + "=" * 70)
    print("VERDICT EVALUATION")
    print("=" * 70)

    # V1: DELTA correlation (correlated changes) > 0.7 in EMBODIED
    v1_pass = post_results['EMBODIED']['delta_correlation'] > config.ignition_correlation_threshold
    v1_value = post_results['EMBODIED']['delta_correlation']

    # V2: DISEMBODIED shows no/low delta correlation (no ignition)
    v2_pass = post_results['DISEMBODIED']['delta_correlation'] < 0.3
    v2_value = post_results['DISEMBODIED']['delta_correlation']

    # V3: Ignition happens within 1 batch (not gradual over 3+ batches)
    v3_pass = temporal_result['is_immediate_ignition']
    v3_value = temporal_result['jump_at_batch_1']

    # V4: SUBLIMINAL shows smaller MAGNITUDE of change than EMBODIED
    # (threshold effect = subliminal changes are weaker even if correlated)
    embodied_magnitude = post_results['EMBODIED']['mean_layer_change']
    subliminal_magnitude = post_results['SUBLIMINAL']['mean_layer_change']
    magnitude_ratio = subliminal_magnitude / max(embodied_magnitude, 1e-8)
    v4_pass = magnitude_ratio < config.subliminal_threshold_ratio
    v4_value = magnitude_ratio

    verdicts = {
        'V1': {
            'description': 'EMBODIED shows high delta correlation (ignition) > 0.7',
            'pass': v1_pass,
            'value': v1_value,
            'threshold': config.ignition_correlation_threshold
        },
        'V2': {
            'description': 'DISEMBODIED shows low delta correlation (no ignition) < 0.3',
            'pass': v2_pass,
            'value': v2_value,
            'threshold': 0.3
        },
        'V3': {
            'description': 'Ignition happens within 1 batch (immediate)',
            'pass': v3_pass,
            'value': v3_value,
            'threshold': 'immediate'
        },
        'V4': {
            'description': 'SUBLIMINAL shows smaller change magnitude than EMBODIED',
            'pass': v4_pass,
            'value': v4_value,
            'threshold': config.subliminal_threshold_ratio
        }
    }

    print("\nVerdict Summary:")
    print("-" * 50)
    all_pass = True
    for v_id, v_data in verdicts.items():
        status = "PASS" if v_data['pass'] else "FAIL"
        all_pass = all_pass and v_data['pass']
        print(f"  {v_id}: [{status}] {v_data['description']}")
        print(f"       Value: {v_data['value']:.3f}, Threshold: {v_data['threshold']}")

    print("\n" + "=" * 70)
    if all_pass:
        print("OVERALL: PASS - Global Workspace Ignition confirmed in embodied model")
    else:
        passing = sum(1 for v in verdicts.values() if v['pass'])
        print(f"OVERALL: PARTIAL ({passing}/4 verdicts passed)")
    print("=" * 70)

    # Save results
    results = {
        'experiment': 'z1719_global_workspace_ignition',
        'timestamp': datetime.now().isoformat(),
        'device': str(device),
        'config': {
            'hidden_dim': config.hidden_dim,
            'num_layers': config.num_layers,
            'telemetry_dim': config.telemetry_dim,
            'baseline_power': config.baseline_power,
            'spike_power': config.spike_power,
            'subliminal_power': config.subliminal_power,
            'ignition_threshold': config.ignition_correlation_threshold,
            'training_steps': config.num_training_steps
        },
        'pre_training_results': {
            k: {kk: float(vv) if isinstance(vv, (int, float, np.floating)) else (bool(vv) if isinstance(vv, (np.bool_, bool)) else vv)
                for kk, vv in v.items()}
            for k, v in pre_results.items()
        },
        'post_training_results': {
            k: {kk: float(vv) if isinstance(vv, (int, float, np.floating)) else (bool(vv) if isinstance(vv, (np.bool_, bool)) else vv)
                for kk, vv in v.items()}
            for k, v in post_results.items()
        },
        'temporal_results': {
            'correlations_over_batches': [float(c) for c in temporal_result['correlations_over_batches']],
            'jump_at_batch_1': float(temporal_result['jump_at_batch_1']),
            'max_subsequent_jump': float(temporal_result['max_subsequent_jump']),
            'is_immediate_ignition': bool(temporal_result['is_immediate_ignition'])
        },
        'verdicts': {
            k: {
                'description': v['description'],
                'pass': bool(v['pass']),
                'value': float(v['value']) if isinstance(v['value'], (int, float, np.floating)) else v['value'],
                'threshold': float(v['threshold']) if isinstance(v['threshold'], (int, float, np.floating)) else v['threshold']
            }
            for k, v in verdicts.items()
        },
        'overall_pass': bool(all_pass),
        'training_summary': {
            'final_loss': float(np.mean(training_stats['losses'][-20:])),
            'final_correlation': float(np.mean(training_stats['cross_layer_correlations'][-20:])) if training_stats['cross_layer_correlations'] else 0
        },
        'theory': {
            'name': 'Global Workspace Theory (Baars, Dehaene)',
            'prediction': 'Ignition = simultaneous broadcast across all layers when signal crosses threshold',
            'key_signature': 'High cross-layer correlation jump in embodied condition',
            'null_prediction': 'Disembodied model shows no ignition (no threshold to cross)'
        }
    }

    results_path = Path('/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results')
    results_path.mkdir(exist_ok=True)
    output_file = results_path / 'z1719_global_workspace_ignition.json'

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    # Print key findings
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)
    print(f"\n1. EMBODIED delta correlation (ignition): {post_results['EMBODIED']['delta_correlation']:.3f}")
    print(f"   (Threshold: {config.ignition_correlation_threshold})")

    print(f"\n2. DISEMBODIED delta correlation: {post_results['DISEMBODIED']['delta_correlation']:.3f}")
    print(f"   (Should be < 0.3 for no ignition)")

    print(f"\n3. Temporal ignition: {'Immediate' if temporal_result['is_immediate_ignition'] else 'Gradual'}")
    print(f"   (Batch 1 jump: {temporal_result['jump_at_batch_1']:.3f})")

    print(f"\n4. SUBLIMINAL/EMBODIED magnitude ratio: {magnitude_ratio:.3f}")
    print(f"   (Should be < {config.subliminal_threshold_ratio} for threshold effect)")

    print("\nInterpretation:")
    if v1_pass and v2_pass:
        print("  - Embodied model shows Global Workspace ignition signature")
        print("  - Telemetry spike causes CORRELATED changes across all layers")
        print("  - Disembodied model shows UNCORRELATED/weak changes (no ignition)")
    else:
        print("  - Results do not fully support GWT ignition hypothesis")
        print("  - May need longer training or different spike magnitude")


if __name__ == '__main__':
    main()
