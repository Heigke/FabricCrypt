#!/usr/bin/env python3
"""
z1921: Causal Intervention Falsification Test

GOAL: Definitively test if hardware state CAUSALLY affects language modeling.

The z1901 falsification showed behavioral distance <0.005 across all telemetry
ablations. This test goes further by:

1. Computing COUNTERFACTUAL outcomes - what WOULD the model output if
   hardware state were different?
2. Measuring INTERVENTION effects - when we artificially change hardware
   (via GPU stress), does LM behavior change?
3. Testing GRANGER CAUSALITY - does telemetry history predict LM output
   better than LM output history alone?

Key insight from Cogitate Consortium (2025):
- Adversarial collaboration between IIT and GNW researchers
- Both theories require demonstrating CAUSAL mechanisms
- Correlation is not causation - need intervention tests

Falsification criteria:
- If counterfactual LM outputs are identical → NOT CAUSAL
- If GPU stress doesn't change LM behavior → NOT CAUSAL
- If telemetry doesn't Granger-cause LM outputs → NOT CAUSAL
"""

import os
import sys
import json
import time
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F

# GPU setup for gfx1151
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")

# Telemetry
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.telemetry.sysfs_hwmon import SysfsHwmonTelemetry, GpuSample

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_telemetry_state(sensor: SysfsHwmonTelemetry) -> dict:
    """Helper to get telemetry in dict format."""
    sample = sensor.read_sample()
    return {
        'temp_edge_c': sample.temp_c if hasattr(sample, 'temp_c') else 50,
        'temp_junction_c': sample.temp_c if hasattr(sample, 'temp_c') else 50,
        'power_w': sample.power_w if hasattr(sample, 'power_w') else 20,
        'power_cap_w': 150,
        'sclk_mhz': sample.sclk_mhz if hasattr(sample, 'sclk_mhz') else 1000,
        'mclk_mhz': sample.mclk_mhz if hasattr(sample, 'mclk_mhz') else 1000,
        'gpu_util_pct': sample.gpu_util if hasattr(sample, 'gpu_util') else 0,
        'mem_util_pct': 0,
        'vram_used_pct': 0,
    }


class CausalEmbodiedModel(nn.Module):
    """Model with explicit causal pathways from hardware to LM."""

    def __init__(self, vocab_size=256, hidden_dim=256, num_layers=6,
                 num_heads=4, telemetry_dim=12):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.telemetry_dim = telemetry_dim

        # Token embedding
        self.embed = nn.Embedding(vocab_size, hidden_dim)

        # Telemetry encoder
        self.telem_encoder = nn.Sequential(
            nn.Linear(telemetry_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # FiLM parameters
        self.film_gamma = nn.Linear(hidden_dim, hidden_dim)
        self.film_beta = nn.Linear(hidden_dim, hidden_dim)

        # Main transformer
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                batch_first=True
            )
            for _ in range(num_layers)
        ])

        # Output heads
        self.lm_head = nn.Linear(hidden_dim, vocab_size)

        # Hardware classification (auxiliary task)
        self.hw_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3)  # low/med/high temp bin
        )

        # Self-prediction head
        self.self_pred = nn.Linear(hidden_dim, telemetry_dim)

    def forward(self, x: torch.Tensor, telemetry: torch.Tensor,
                intervene_layer: int = -1,
                intervention_value: torch.Tensor = None):
        """
        Forward pass with optional causal intervention.

        Args:
            x: Input tokens [batch, seq]
            telemetry: Hardware state [batch, telem_dim]
            intervene_layer: Layer at which to inject intervention (-1 = no intervention)
            intervention_value: Value to inject at intervention point
        """
        batch_size, seq_len = x.shape

        # Encode telemetry
        telem_h = self.telem_encoder(telemetry)

        # FiLM parameters
        gamma = self.film_gamma(telem_h).unsqueeze(1)  # [batch, 1, hidden]
        beta = self.film_beta(telem_h).unsqueeze(1)

        # Embed tokens
        h = self.embed(x)

        # Apply FiLM
        h = gamma * h + beta

        # Process through layers with potential intervention
        for i, layer in enumerate(self.layers):
            h = layer(h)

            # Causal intervention: replace hidden at specific layer
            if i == intervene_layer and intervention_value is not None:
                h = intervention_value

        # Output heads
        logits = self.lm_head(h)
        hw_pred = self.hw_classifier(h.mean(dim=1))
        self_pred = self.self_pred(h.mean(dim=1))

        return {
            'logits': logits,
            'hw_pred': hw_pred,
            'self_pred': self_pred,
            'hidden': h
        }


class GPUStressor:
    """Artificially stress GPU to create hardware state variation."""

    def __init__(self):
        self.stress_tensor = None

    def start_stress(self, intensity: float = 1.0):
        """Start GPU stress via matrix operations."""
        size = int(2000 * intensity)
        self.stress_tensor = torch.randn(size, size, device=DEVICE)

    def maintain_stress(self):
        """Maintain stress with continuous operations."""
        if self.stress_tensor is not None:
            # Matrix multiply - GPU intensive
            _ = torch.mm(self.stress_tensor, self.stress_tensor)

    def stop_stress(self):
        """Stop GPU stress."""
        self.stress_tensor = None
        torch.cuda.empty_cache()


def run_t1_counterfactual(model: CausalEmbodiedModel,
                           num_trials: int = 50) -> Dict:
    """
    T1: Counterfactual Test

    For same input, compute output under:
    - Real telemetry
    - Counterfactual telemetry (what if temp were 20°C higher?)

    If outputs are identical, hardware doesn't causally affect LM.
    """
    print("\n=== T1: Counterfactual Test ===")

    model.eval()
    sensor = SysfsHwmonTelemetry()

    counterfactual_diffs = []

    with torch.no_grad():
        for trial in range(num_trials):
            # Get real telemetry
            state = get_telemetry_state(sensor)
            real_telem = torch.tensor([
                state.get('temp_edge_c', 50) / 100,
                state.get('temp_junction_c', 50) / 100,
                state.get('power_w', 20) / 200,
                state.get('power_cap_w', 50) / 200,
                state.get('sclk_mhz', 1000) / 3000,
                state.get('mclk_mhz', 1000) / 3000,
                state.get('gpu_util_pct', 0) / 100,
                state.get('mem_util_pct', 0) / 100,
                state.get('vram_used_pct', 0),
                0.5, 0.5, 0.1  # Padding to 12 dims
            ], dtype=torch.float32, device=DEVICE).unsqueeze(0)

            # Create counterfactual: +20°C on temp dimensions
            cf_telem = real_telem.clone()
            cf_telem[0, 0] += 0.2  # temp_edge +20°C
            cf_telem[0, 1] += 0.2  # temp_junction +20°C
            cf_telem = torch.clamp(cf_telem, 0, 1)

            # Same input
            x = torch.randint(0, 256, (1, 64), device=DEVICE)

            # Real output
            out_real = model(x, real_telem)
            logits_real = out_real['logits']

            # Counterfactual output
            out_cf = model(x, cf_telem)
            logits_cf = out_cf['logits']

            # Measure difference
            logits_diff = F.mse_loss(logits_real, logits_cf).item()
            probs_real = F.softmax(logits_real[:, -1, :], dim=-1)
            probs_cf = F.softmax(logits_cf[:, -1, :], dim=-1)
            kl_div = F.kl_div(probs_cf.log(), probs_real, reduction='sum').item()

            # Token prediction difference
            pred_real = logits_real[:, -1, :].argmax().item()
            pred_cf = logits_cf[:, -1, :].argmax().item()
            pred_diff = int(pred_real != pred_cf)

            counterfactual_diffs.append({
                'logits_mse': logits_diff,
                'kl_div': kl_div,
                'pred_diff': pred_diff
            })

            if trial % 10 == 0:
                print(f"  Trial {trial}: logits_mse={logits_diff:.6f}, pred_diff={pred_diff}")

    mean_logits = np.mean([d['logits_mse'] for d in counterfactual_diffs])
    mean_kl = np.mean([d['kl_div'] for d in counterfactual_diffs])
    pred_change_rate = np.mean([d['pred_diff'] for d in counterfactual_diffs])

    # Falsification: if counterfactual produces ~same output, hardware not causal
    falsified = mean_logits < 0.001 and pred_change_rate < 0.1

    return {
        'test': 'T1_counterfactual',
        'mean_logits_mse': mean_logits,
        'mean_kl_div': mean_kl,
        'pred_change_rate': pred_change_rate,
        'falsified': falsified,
        'interpretation': 'FALSIFIED - counterfactuals identical' if falsified
                          else f'PASSED - {pred_change_rate*100:.1f}% predictions change'
    }


def run_t2_intervention_stress(model: CausalEmbodiedModel,
                                 num_samples: int = 30) -> Dict:
    """
    T2: Real Hardware Intervention

    Actually stress the GPU and measure if LM behavior changes.
    This is the gold standard for causal testing.
    """
    print("\n=== T2: Hardware Intervention (GPU Stress) ===")

    model.eval()
    sensor = SysfsHwmonTelemetry()
    stressor = GPUStressor()

    # Collect baseline (unstressed) outputs
    print("  Collecting baseline (cool GPU)...")
    baseline_outputs = []
    baseline_temps = []

    with torch.no_grad():
        for _ in range(num_samples):
            state = get_telemetry_state(sensor)
            telem = torch.tensor([
                state.get('temp_edge_c', 50) / 100,
                state.get('temp_junction_c', 50) / 100,
                state.get('power_w', 20) / 200,
                state.get('power_cap_w', 50) / 200,
                state.get('sclk_mhz', 1000) / 3000,
                state.get('mclk_mhz', 1000) / 3000,
                state.get('gpu_util_pct', 0) / 100,
                state.get('mem_util_pct', 0) / 100,
                state.get('vram_used_pct', 0),
                0.5, 0.5, 0.1
            ], dtype=torch.float32, device=DEVICE).unsqueeze(0)

            x = torch.randint(0, 256, (1, 64), device=DEVICE)
            out = model(x, telem)

            baseline_outputs.append(out['logits'].cpu().numpy())
            baseline_temps.append(state.get('temp_edge_c', 50))
            time.sleep(0.1)

    # Start GPU stress
    print("  Stressing GPU...")
    stressor.start_stress(intensity=1.0)

    # Wait for thermal response
    for _ in range(20):
        stressor.maintain_stress()
        time.sleep(0.1)

    # Collect stressed outputs
    print("  Collecting stressed outputs...")
    stressed_outputs = []
    stressed_temps = []

    with torch.no_grad():
        for _ in range(num_samples):
            stressor.maintain_stress()

            state = get_telemetry_state(sensor)
            telem = torch.tensor([
                state.get('temp_edge_c', 50) / 100,
                state.get('temp_junction_c', 50) / 100,
                state.get('power_w', 20) / 200,
                state.get('power_cap_w', 50) / 200,
                state.get('sclk_mhz', 1000) / 3000,
                state.get('mclk_mhz', 1000) / 3000,
                state.get('gpu_util_pct', 0) / 100,
                state.get('mem_util_pct', 0) / 100,
                state.get('vram_used_pct', 0),
                0.5, 0.5, 0.1
            ], dtype=torch.float32, device=DEVICE).unsqueeze(0)

            x = torch.randint(0, 256, (1, 64), device=DEVICE)
            out = model(x, telem)

            stressed_outputs.append(out['logits'].cpu().numpy())
            stressed_temps.append(state.get('temp_edge_c', 50))
            time.sleep(0.1)

    stressor.stop_stress()

    # Analyze difference
    baseline_mean = np.mean(baseline_outputs, axis=0)
    stressed_mean = np.mean(stressed_outputs, axis=0)

    output_diff = np.mean(np.abs(baseline_mean - stressed_mean))
    temp_diff = np.mean(stressed_temps) - np.mean(baseline_temps)

    print(f"  Temperature change: {temp_diff:.1f}°C")
    print(f"  Output difference: {output_diff:.6f}")

    # Falsification: if significant temp change but no output change
    significant_temp_change = temp_diff > 5  # 5°C is significant
    significant_output_change = output_diff > 0.01

    falsified = significant_temp_change and not significant_output_change

    return {
        'test': 'T2_intervention_stress',
        'baseline_temp': np.mean(baseline_temps),
        'stressed_temp': np.mean(stressed_temps),
        'temp_diff': temp_diff,
        'output_diff': output_diff,
        'significant_temp_change': significant_temp_change,
        'significant_output_change': significant_output_change,
        'falsified': falsified,
        'interpretation': f'FALSIFIED - {temp_diff:.1f}°C change, no LM effect' if falsified
                          else f'PASSED - output changes with {temp_diff:.1f}°C thermal shift'
    }


def run_t3_granger_causality(model: CausalEmbodiedModel,
                               num_samples: int = 100,
                               max_lag: int = 5) -> Dict:
    """
    T3: Granger Causality Test

    Does telemetry history help predict LM output beyond what
    LM output history alone predicts?

    If telemetry Granger-causes LM output, we have evidence of
    causal influence.
    """
    print("\n=== T3: Granger Causality ===")

    model.eval()
    sensor = SysfsHwmonTelemetry()
    stressor = GPUStressor()

    # Collect time series of telemetry and LM outputs
    print("  Collecting time series...")

    telemetry_series = []
    output_series = []

    with torch.no_grad():
        # Create some thermal variation
        stressor.start_stress(intensity=0.5)

        for i in range(num_samples):
            if i % 20 == 10:
                stressor.stop_stress()
                time.sleep(0.5)
                stressor.start_stress(intensity=0.5)

            stressor.maintain_stress()

            state = get_telemetry_state(sensor)
            telem = [
                state.get('temp_edge_c', 50),
                state.get('power_w', 20),
                state.get('gpu_util_pct', 0),
            ]

            telem_tensor = torch.tensor([
                state.get('temp_edge_c', 50) / 100,
                state.get('temp_junction_c', 50) / 100,
                state.get('power_w', 20) / 200,
                state.get('power_cap_w', 50) / 200,
                state.get('sclk_mhz', 1000) / 3000,
                state.get('mclk_mhz', 1000) / 3000,
                state.get('gpu_util_pct', 0) / 100,
                state.get('mem_util_pct', 0) / 100,
                state.get('vram_used_pct', 0),
                0.5, 0.5, 0.1
            ], dtype=torch.float32, device=DEVICE).unsqueeze(0)

            x = torch.randint(0, 256, (1, 64), device=DEVICE)
            out = model(x, telem_tensor)

            # Use mean logit as scalar output
            output_scalar = out['logits'].mean().item()

            telemetry_series.append(telem)
            output_series.append(output_scalar)

            time.sleep(0.05)

        stressor.stop_stress()

    telemetry_series = np.array(telemetry_series)
    output_series = np.array(output_series)

    # Simple Granger test: compare prediction error with/without telemetry

    # Model 1: Predict output[t] from output[t-1:t-lag]
    errors_without_telem = []
    for t in range(max_lag, len(output_series)):
        y_true = output_series[t]
        y_pred = np.mean(output_series[t-max_lag:t])  # Naive: predict mean of history
        errors_without_telem.append((y_true - y_pred) ** 2)

    # Model 2: Predict output[t] from output[t-1:t-lag] AND telemetry[t-1:t-lag]
    errors_with_telem = []
    for t in range(max_lag, len(output_series)):
        y_true = output_series[t]
        # Combine output and telemetry history
        out_hist = output_series[t-max_lag:t]
        telem_hist = telemetry_series[t-max_lag:t]

        # Simple linear combination (could use regression, but keeping simple)
        y_pred = np.mean(out_hist) + 0.01 * np.mean(telem_hist[:, 0])  # temp influence
        errors_with_telem.append((y_true - y_pred) ** 2)

    mse_without = np.mean(errors_without_telem)
    mse_with = np.mean(errors_with_telem)

    # F-statistic approximation
    improvement = (mse_without - mse_with) / mse_without if mse_without > 0 else 0

    # Compute correlation between telemetry and output changes
    telem_changes = np.diff(telemetry_series[:, 0])  # Temperature changes
    output_changes = np.diff(output_series)

    # Ensure same length
    min_len = min(len(telem_changes), len(output_changes))
    correlation = np.corrcoef(telem_changes[:min_len], output_changes[:min_len])[0, 1]

    print(f"  MSE without telemetry: {mse_without:.6f}")
    print(f"  MSE with telemetry: {mse_with:.6f}")
    print(f"  Improvement: {improvement*100:.2f}%")
    print(f"  Telem-output correlation: {correlation:.4f}")

    # Falsification: telemetry doesn't help predict output
    falsified = improvement <= 0 and abs(correlation) < 0.1

    return {
        'test': 'T3_granger_causality',
        'mse_without_telem': mse_without,
        'mse_with_telem': mse_with,
        'improvement': improvement,
        'telem_output_correlation': correlation,
        'falsified': falsified,
        'interpretation': 'FALSIFIED - telemetry doesn\'t help prediction' if falsified
                          else f'PASSED - {improvement*100:.1f}% prediction improvement'
    }


def run_t4_do_calculus(model: CausalEmbodiedModel) -> Dict:
    """
    T4: do-Calculus Intervention

    Test: P(Y | do(X=x)) vs P(Y | X=x)

    If intervening on telemetry (do) produces same result as
    observing telemetry (condition), there's no confounding.
    If different, confounders exist.

    Implementation: Compare model output when we SET telemetry
    vs when we observe same telemetry values naturally.
    """
    print("\n=== T4: do-Calculus Intervention ===")

    model.eval()
    sensor = SysfsHwmonTelemetry()

    # Collect natural observations
    print("  Collecting natural observations...")
    natural_outputs = {}

    with torch.no_grad():
        for _ in range(50):
            state = get_telemetry_state(sensor)
            temp_bin = int(state.get('temp_edge_c', 50) // 10)  # Bin by 10°C

            telem = torch.tensor([
                state.get('temp_edge_c', 50) / 100,
                state.get('temp_junction_c', 50) / 100,
                state.get('power_w', 20) / 200,
                state.get('power_cap_w', 50) / 200,
                state.get('sclk_mhz', 1000) / 3000,
                state.get('mclk_mhz', 1000) / 3000,
                state.get('gpu_util_pct', 0) / 100,
                state.get('mem_util_pct', 0) / 100,
                state.get('vram_used_pct', 0),
                0.5, 0.5, 0.1
            ], dtype=torch.float32, device=DEVICE).unsqueeze(0)

            x = torch.randint(0, 256, (1, 64), device=DEVICE)
            out = model(x, telem)

            if temp_bin not in natural_outputs:
                natural_outputs[temp_bin] = []
            natural_outputs[temp_bin].append(out['logits'].mean().item())

            time.sleep(0.05)

    # Intervention: SET telemetry to specific values
    print("  Performing interventions...")
    intervention_outputs = {}

    with torch.no_grad():
        for temp_bin in range(3, 10):  # 30°C to 100°C
            interventions = []
            for _ in range(20):
                # Create intervention telemetry
                telem = torch.tensor([
                    (temp_bin * 10) / 100,  # SET temperature
                    (temp_bin * 10) / 100,
                    0.2,  # Fixed power
                    0.5,
                    0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.1
                ], dtype=torch.float32, device=DEVICE).unsqueeze(0)

                x = torch.randint(0, 256, (1, 64), device=DEVICE)
                out = model(x, telem)
                interventions.append(out['logits'].mean().item())

            intervention_outputs[temp_bin] = interventions

    # Compare natural vs intervention
    differences = []
    for temp_bin in natural_outputs:
        if temp_bin in intervention_outputs:
            nat_mean = np.mean(natural_outputs[temp_bin])
            int_mean = np.mean(intervention_outputs[temp_bin])
            diff = abs(nat_mean - int_mean)
            differences.append(diff)
            print(f"  Temp bin {temp_bin*10}°C: natural={nat_mean:.4f}, intervention={int_mean:.4f}, diff={diff:.4f}")

    mean_diff = np.mean(differences) if differences else 0

    # If do() produces same as condition(), no confounding (good for causal claims)
    # If different, confounders exist
    falsified = mean_diff < 0.001  # If identical, telemetry has no causal effect

    return {
        'test': 'T4_do_calculus',
        'mean_natural_intervention_diff': mean_diff,
        'num_temp_bins_compared': len(differences),
        'falsified': falsified,
        'interpretation': 'FALSIFIED - no causal effect of telemetry' if falsified
                          else f'PASSED - {mean_diff:.4f} causal effect detected'
    }


def train_causal_model(model: CausalEmbodiedModel,
                        num_epochs: int = 5,
                        steps_per_epoch: int = 50) -> List[Dict]:
    """Train the causal model with auxiliary tasks."""
    print("\n=== Training Causal Model ===")

    sensor = SysfsHwmonTelemetry()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    losses = []

    for epoch in range(num_epochs):
        model.train()
        epoch_losses = []

        for step in range(steps_per_epoch):
            state = get_telemetry_state(sensor)
            telem = torch.tensor([
                state.get('temp_edge_c', 50) / 100,
                state.get('temp_junction_c', 50) / 100,
                state.get('power_w', 20) / 200,
                state.get('power_cap_w', 50) / 200,
                state.get('sclk_mhz', 1000) / 3000,
                state.get('mclk_mhz', 1000) / 3000,
                state.get('gpu_util_pct', 0) / 100,
                state.get('mem_util_pct', 0) / 100,
                state.get('vram_used_pct', 0),
                0.5, 0.5, 0.1
            ], dtype=torch.float32, device=DEVICE)

            # Batch
            telem_batch = telem.unsqueeze(0).expand(4, -1)

            # Temperature classification label
            temp_bin = min(2, int(state.get('temp_edge_c', 50) // 30))
            hw_labels = torch.tensor([temp_bin] * 4, device=DEVICE)

            # Input/target
            x = torch.randint(0, 256, (4, 64), device=DEVICE)
            targets = x[:, 1:].contiguous()

            # Forward
            outputs = model(x[:, :-1], telem_batch)

            # LM loss
            lm_loss = F.cross_entropy(
                outputs['logits'].reshape(-1, 256),
                targets.reshape(-1)
            )

            # Hardware classification loss
            hw_loss = F.cross_entropy(outputs['hw_pred'], hw_labels)

            # Self-prediction loss
            self_loss = F.mse_loss(outputs['self_pred'], telem_batch)

            # Combined
            loss = lm_loss + 0.5 * hw_loss + 0.1 * self_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append({
                'total': loss.item(),
                'lm': lm_loss.item(),
                'hw': hw_loss.item(),
                'self': self_loss.item()
            })

        mean_loss = np.mean([l['total'] for l in epoch_losses])
        losses.append({'epoch': epoch, 'loss': mean_loss})
        print(f"  Epoch {epoch}: loss={mean_loss:.4f}")

    return losses


def main():
    print("=" * 60)
    print("z1921: Causal Intervention Falsification Test")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Timestamp: {datetime.now().isoformat()}")

    results = {
        'experiment': 'z1921_causal_intervention',
        'timestamp': datetime.now().isoformat(),
        'device': str(DEVICE),
    }

    # Initialize model
    model = CausalEmbodiedModel(
        vocab_size=256,
        hidden_dim=256,
        num_layers=6,
        num_heads=4,
        telemetry_dim=12
    ).to(DEVICE)

    results['model_params'] = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {results['model_params']:,}")

    # Train
    training_losses = train_causal_model(model, num_epochs=5)
    results['training'] = training_losses

    # Run causal tests
    tests = {}

    tests['T1'] = run_t1_counterfactual(model)
    tests['T2'] = run_t2_intervention_stress(model)
    tests['T3'] = run_t3_granger_causality(model)
    tests['T4'] = run_t4_do_calculus(model)

    results['tests'] = tests

    # Summary
    num_falsified = sum(1 for t in tests.values() if t['falsified'])
    num_total = len(tests)

    results['num_falsified'] = num_falsified
    results['num_total'] = num_total
    results['causal_score'] = (num_total - num_falsified) / num_total

    if num_falsified >= 3:
        results['verdict'] = 'CAUSAL EMBODIMENT FALSIFIED'
    elif num_falsified >= 1:
        results['verdict'] = 'PARTIAL CAUSAL EVIDENCE'
    else:
        results['verdict'] = 'CAUSAL EMBODIMENT DEMONSTRATED'

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, test in tests.items():
        status = "❌ FALSIFIED" if test['falsified'] else "✅ PASSED"
        print(f"{name}: {status} - {test['interpretation']}")

    print(f"\nFalsified: {num_falsified}/{num_total}")
    print(f"Causal score: {results['causal_score']:.1%}")
    print(f"VERDICT: {results['verdict']}")

    # Save results
    output_path = 'results/z1921_causal_intervention.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    return results


if __name__ == "__main__":
    main()
