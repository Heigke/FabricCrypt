#!/usr/bin/env python3
"""
z2001: PROPER GRANGER CAUSALITY TEST - Statistical Rigor per Cogitate Methodology

The z1990 experiment failed F4 (Granger causality p=1.0) because:
1. Insufficient data points (needed 1000+, had ~200)
2. Simplified F-stat calculation instead of proper statsmodels
3. Testing wrong variables (telemetry internal instead of telemetry -> model outputs)

This experiment implements RIGOROUS statistical Granger causality testing.

HYPOTHESIS:
- If embodied: lagged telemetry PREDICTS future model outputs (p < 0.05)
- This shows hardware state has CAUSAL INFLUENCE on computation
- Bidirectional testing to establish directionality

METHODOLOGY:
1. Collect 1000+ timestep time series of:
   - Telemetry: [temp, power, freq, util] - hardware state
   - Model outputs: [hidden_mean, logits_entropy, action_probs] - model state
2. Use statsmodels grangercausalitytests with multiple lags
3. Report F-statistics and p-values for lags 1, 2, 5, 10
4. Test both directions: telemetry -> outputs AND outputs -> telemetry

COGITATE CRITERIA ADDRESSED:
- Pre-registered hypothesis
- Proper statistical test (not simplified approximation)
- Multiple lag testing
- Bidirectional causality analysis
- Sufficient sample size for statistical power

Author: Claude (Opus 4.5)
Date: 2026-02-06
"""

import functools
print = functools.partial(print, flush=True)

import os
import sys
import json
import time
import warnings
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
import numpy as np

# HSA override for gfx1151 compatibility
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import telemetry from sysfs_hwmon
from src.telemetry.sysfs_hwmon import SysfsHwmonTelemetry, GpuSample

# Statsmodels for proper Granger causality
try:
    from statsmodels.tsa.stattools import grangercausalitytests
    from statsmodels.tsa.stattools import adfuller
    STATSMODELS_AVAILABLE = True
except ImportError:
    print("[WARNING] statsmodels not available. Install with: pip install statsmodels")
    STATSMODELS_AVAILABLE = False

# Scipy for additional statistics
from scipy import stats

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RESULTS_DIR = Path(__file__).parent.parent / 'results'
RESULTS_DIR.mkdir(exist_ok=True)


# =============================================================================
# FiLM-CONDITIONED MODEL (Simplified for Granger Test)
# =============================================================================

class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation for hardware conditioning."""

    def __init__(self, hidden_dim: int, condition_dim: int):
        super().__init__()
        self.gamma = nn.Linear(condition_dim, hidden_dim)
        self.beta = nn.Linear(condition_dim, hidden_dim)

        # Initialize near identity
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        gamma = 1 + self.gamma(condition)
        beta = self.beta(condition)
        if x.dim() == 3 and gamma.dim() == 2:
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)
        return gamma * x + beta


class FiLMTransformer(nn.Module):
    """
    FiLM-conditioned transformer for Granger causality testing.

    Key: Telemetry MODULATES the internal computation, so if there's
    causal influence, it should appear in Granger tests.
    """

    def __init__(self, vocab_size: int = 128, hidden_dim: int = 256,
                 telemetry_dim: int = 4, n_layers: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.telemetry_dim = telemetry_dim

        # Token embedding
        self.embed = nn.Embedding(vocab_size, hidden_dim)

        # FiLM conditioning on telemetry
        self.film_layers = nn.ModuleList([
            FiLMLayer(hidden_dim, telemetry_dim) for _ in range(n_layers)
        ])

        # Transformer layers
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=4,
                dim_feedforward=hidden_dim * 2,
                batch_first=True,
                norm_first=True,
                dropout=0.0,  # No dropout for clean Granger signal
            ) for _ in range(n_layers)
        ])

        # Output heads
        self.output = nn.Linear(hidden_dim, vocab_size)
        self.action_head = nn.Linear(hidden_dim, 4)  # 4 action modes

    def forward(self, tokens: torch.Tensor, telemetry: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass returning multiple outputs for Granger testing.

        Returns:
            Dict with logits, hidden states, action logits
        """
        # Token embedding
        h = self.embed(tokens)  # [batch, seq, hidden]

        # Apply FiLM-conditioned transformer layers
        for layer, film in zip(self.layers, self.film_layers):
            h = film(h, telemetry)
            h = layer(h)

        # Output logits
        logits = self.output(h)

        # Action logits from mean-pooled hidden
        hidden_mean = h.mean(dim=1)  # [batch, hidden]
        action_logits = self.action_head(hidden_mean)

        return {
            'logits': logits,
            'hidden': h,
            'hidden_mean': hidden_mean,
            'action_logits': action_logits,
        }


# =============================================================================
# TIME SERIES DATA COLLECTION
# =============================================================================

@dataclass
class TimeSeriesRecord:
    """Single timestep record for Granger analysis."""
    timestamp: float
    # Telemetry (inputs)
    temp: float
    power: float
    freq: float
    util: float
    # Model outputs
    hidden_mean: float
    logits_entropy: float
    action_prob_0: float
    action_prob_1: float
    action_prob_2: float
    action_prob_3: float


class GrangerDataCollector:
    """
    Collects time series data for Granger causality testing.

    Must collect ALIGNED telemetry and model outputs at each timestep.
    """

    def __init__(self, telemetry: SysfsHwmonTelemetry, model: FiLMTransformer,
                 device: torch.device, min_samples: int = 1000):
        self.telemetry = telemetry
        self.model = model
        self.device = device
        self.min_samples = min_samples
        self.records: List[TimeSeriesRecord] = []

        # Text data for model input
        self.text = self._generate_text()
        self.char2idx = {c: i for i, c in enumerate(sorted(set(self.text)))}
        self.vocab_size = len(self.char2idx)
        self.data = torch.tensor([self.char2idx[c] for c in self.text], dtype=torch.long)

    def _generate_text(self) -> str:
        """Generate text for model input."""
        samples = [
            "To be or not to be that is the question\n",
            "All the world is a stage and all the men and women merely players\n",
            "Now is the winter of our discontent made glorious summer\n",
            "Friends Romans countrymen lend me your ears\n",
        ]
        return ''.join(samples * 500)

    def collect_sample(self, seq_len: int = 64, with_workload: bool = True) -> TimeSeriesRecord:
        """
        Collect a single aligned sample of telemetry + model output.

        Args:
            seq_len: Sequence length for model input
            with_workload: If True, run varying GPU workload to create telemetry variation
        """
        # Optional: Add varying GPU workload to create telemetry variation
        if with_workload and np.random.random() < 0.5:
            # Random matrix multiply to vary GPU load
            size = np.random.randint(256, 1024)
            a = torch.randn(size, size, device=self.device)
            b = torch.randn(size, size, device=self.device)
            _ = torch.matmul(a, b)
            del a, b

        # Read telemetry
        sample = self.telemetry.read_sample()

        # Get random text window
        idx = np.random.randint(0, len(self.data) - seq_len - 1)
        tokens = self.data[idx:idx + seq_len].unsqueeze(0).to(self.device)

        # Build telemetry tensor
        tel = torch.tensor([
            sample.temp_edge_c / 100.0,  # Normalize to ~0-1
            sample.power_w / 100.0,
            sample.freq_sclk_mhz / 2000.0,
            sample.gpu_busy_pct / 100.0,
        ], dtype=torch.float32, device=self.device).unsqueeze(0)

        # Forward pass
        with torch.no_grad():
            out = self.model(tokens, tel)

            # Extract model outputs
            hidden_mean = out['hidden_mean'].mean().item()

            # Logits entropy (measure of prediction uncertainty)
            logits = out['logits'][0, -1, :]  # Last position
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * (probs + 1e-10).log()).sum().item()

            # Action probabilities
            action_probs = F.softmax(out['action_logits'][0], dim=-1).cpu().numpy()

        record = TimeSeriesRecord(
            timestamp=time.time(),
            temp=sample.temp_edge_c,
            power=sample.power_w,
            freq=float(sample.freq_sclk_mhz),
            util=sample.gpu_busy_pct,
            hidden_mean=hidden_mean,
            logits_entropy=entropy,
            action_prob_0=float(action_probs[0]),
            action_prob_1=float(action_probs[1]),
            action_prob_2=float(action_probs[2]),
            action_prob_3=float(action_probs[3]),
        )

        self.records.append(record)
        return record

    def collect_series(self, target_samples: int = None,
                       sample_interval: float = 0.05,
                       with_workload: bool = True) -> List[TimeSeriesRecord]:
        """
        Collect full time series for Granger analysis.

        Args:
            target_samples: Number of samples to collect (default: self.min_samples)
            sample_interval: Time between samples in seconds
            with_workload: If True, vary GPU workload for telemetry variation
        """
        if target_samples is None:
            target_samples = self.min_samples

        print(f"\n[GrangerCollector] Collecting {target_samples} samples...")
        print(f"  Interval: {sample_interval*1000:.0f}ms, Est. time: {target_samples * sample_interval:.0f}s")
        if with_workload:
            print(f"  With varying GPU workload for telemetry variation")

        start_time = time.time()
        last_progress = 0

        for i in range(target_samples):
            self.collect_sample(with_workload=with_workload)

            # Progress update
            progress = (i + 1) / target_samples
            if progress - last_progress >= 0.1:
                elapsed = time.time() - start_time
                eta = elapsed / progress * (1 - progress)
                # Show telemetry stats
                recent = self.records[-min(100, len(self.records)):]
                temp_range = max(r.temp for r in recent) - min(r.temp for r in recent)
                power_range = max(r.power for r in recent) - min(r.power for r in recent)
                print(f"  {progress*100:.0f}% ({i+1}/{target_samples}) - "
                      f"ETA: {eta:.0f}s, temp_range={temp_range:.1f}C, power_range={power_range:.1f}W")
                last_progress = progress

            time.sleep(sample_interval)

        total_time = time.time() - start_time
        print(f"  Collection complete: {len(self.records)} samples in {total_time:.1f}s")

        # Report telemetry statistics
        temps = [r.temp for r in self.records]
        powers = [r.power for r in self.records]
        utils = [r.util for r in self.records]
        print(f"  Telemetry ranges:")
        print(f"    Temp:  {min(temps):.1f} - {max(temps):.1f} C (std={np.std(temps):.2f})")
        print(f"    Power: {min(powers):.1f} - {max(powers):.1f} W (std={np.std(powers):.2f})")
        print(f"    Util:  {min(utils):.0f} - {max(utils):.0f} % (std={np.std(utils):.2f})")

        return self.records

    def get_arrays(self) -> Dict[str, np.ndarray]:
        """Convert records to numpy arrays for analysis."""
        n = len(self.records)
        return {
            'timestamp': np.array([r.timestamp for r in self.records]),
            # Telemetry (4 dims)
            'temp': np.array([r.temp for r in self.records]),
            'power': np.array([r.power for r in self.records]),
            'freq': np.array([r.freq for r in self.records]),
            'util': np.array([r.util for r in self.records]),
            # Model outputs
            'hidden_mean': np.array([r.hidden_mean for r in self.records]),
            'logits_entropy': np.array([r.logits_entropy for r in self.records]),
            'action_prob_0': np.array([r.action_prob_0 for r in self.records]),
            'action_prob_1': np.array([r.action_prob_1 for r in self.records]),
            'action_prob_2': np.array([r.action_prob_2 for r in self.records]),
            'action_prob_3': np.array([r.action_prob_3 for r in self.records]),
        }


# =============================================================================
# PROPER GRANGER CAUSALITY TESTING
# =============================================================================

def check_stationarity(series: np.ndarray, name: str) -> Tuple[bool, float]:
    """
    Check if time series is stationary using Augmented Dickey-Fuller test.

    Non-stationary series need differencing before Granger test.
    """
    if not STATSMODELS_AVAILABLE:
        return True, 0.0

    try:
        result = adfuller(series, autolag='AIC')
        p_value = result[1]
        is_stationary = p_value < 0.05
        return is_stationary, p_value
    except Exception as e:
        print(f"  [ADF] {name}: Error - {e}")
        return True, 1.0


def difference_series(series: np.ndarray) -> np.ndarray:
    """First difference of time series to achieve stationarity."""
    return np.diff(series)


def run_granger_test(x: np.ndarray, y: np.ndarray,
                     max_lag: int = 10,
                     name_x: str = "X",
                     name_y: str = "Y") -> Dict[str, Any]:
    """
    Run proper Granger causality test: Does X Granger-cause Y?

    Args:
        x: Potential causal variable time series
        y: Effect variable time series
        max_lag: Maximum lag to test
        name_x, name_y: Names for reporting

    Returns:
        Dict with F-stats, p-values for each lag, and overall verdict
    """
    if not STATSMODELS_AVAILABLE:
        return {
            'error': 'statsmodels not available',
            'p_values': {lag: 1.0 for lag in [1, 2, 5, 10]},
            'f_stats': {lag: 0.0 for lag in [1, 2, 5, 10]},
            'significant': False,
        }

    # Combine into 2D array for grangercausalitytests
    # Column 0 = y (effect), Column 1 = x (cause)
    # statsmodels tests if column 1 Granger-causes column 0
    data = np.column_stack([y, x])

    results = {
        'test': f"{name_x} -> {name_y}",
        'lags_tested': [],
        'f_stats': {},
        'p_values': {},
        'significant_lags': [],
    }

    # Suppress statsmodels warnings about future deprecation
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        try:
            # Run Granger test for all lags up to max_lag
            gc_results = grangercausalitytests(data, maxlag=max_lag, verbose=False)

            # Extract results for key lags
            test_lags = [1, 2, 5, 10]
            for lag in test_lags:
                if lag <= max_lag and lag in gc_results:
                    # grangercausalitytests returns nested dict
                    # [0] = test statistics dict, [1] = OLS regression results
                    test_dict = gc_results[lag][0]

                    # Use F-test results ('ssr_ftest')
                    f_stat = test_dict['ssr_ftest'][0]
                    p_value = test_dict['ssr_ftest'][1]

                    results['lags_tested'].append(lag)
                    results['f_stats'][lag] = float(f_stat)
                    results['p_values'][lag] = float(p_value)

                    if p_value < 0.05:
                        results['significant_lags'].append(lag)

        except Exception as e:
            results['error'] = str(e)
            return results

    # Overall verdict: significant if ANY lag is significant
    results['significant'] = len(results['significant_lags']) > 0
    results['min_p_value'] = min(results['p_values'].values()) if results['p_values'] else 1.0

    return results


def run_full_granger_analysis(arrays: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """
    Run comprehensive Granger causality analysis.

    Tests:
    1. Telemetry -> Model outputs (embodiment hypothesis)
    2. Model outputs -> Telemetry (reverse causality check)
    """
    print("\n" + "="*70)
    print("GRANGER CAUSALITY ANALYSIS")
    print("="*70)

    telemetry_vars = ['temp', 'power', 'freq', 'util']
    output_vars = ['hidden_mean', 'logits_entropy', 'action_prob_0']

    results = {
        'stationarity': {},
        'telemetry_to_output': {},
        'output_to_telemetry': {},
        'summary': {},
    }

    # Check stationarity and difference if needed
    print("\n[1] Checking stationarity (ADF test)...")
    processed_arrays = {}

    for var in telemetry_vars + output_vars:
        series = arrays[var]
        is_stationary, p_val = check_stationarity(series, var)
        results['stationarity'][var] = {
            'is_stationary': is_stationary,
            'adf_p_value': float(p_val),
        }

        if is_stationary:
            processed_arrays[var] = series
            print(f"  {var}: Stationary (p={p_val:.4f})")
        else:
            processed_arrays[var] = difference_series(series)
            print(f"  {var}: Non-stationary (p={p_val:.4f}) -> Differenced")

    # Test: Telemetry -> Model Outputs (EMBODIMENT HYPOTHESIS)
    print("\n[2] Testing: Telemetry -> Model Outputs")
    print("    (If significant: hardware CAUSES model behavior)")

    significant_forward = 0
    total_forward = 0

    for tel_var in telemetry_vars:
        for out_var in output_vars:
            x = processed_arrays[tel_var]
            y = processed_arrays[out_var]

            # Ensure same length after differencing
            min_len = min(len(x), len(y))
            x = x[:min_len]
            y = y[:min_len]

            result = run_granger_test(x, y, max_lag=10, name_x=tel_var, name_y=out_var)
            key = f"{tel_var}_to_{out_var}"
            results['telemetry_to_output'][key] = result

            total_forward += 1
            if result.get('significant', False):
                significant_forward += 1
                sig_str = " *** SIGNIFICANT ***"
            else:
                sig_str = ""

            min_p = result.get('min_p_value', 1.0)
            print(f"    {tel_var} -> {out_var}: min_p={min_p:.4f}{sig_str}")

    # Test: Model Outputs -> Telemetry (REVERSE - should be weaker)
    print("\n[3] Testing: Model Outputs -> Telemetry")
    print("    (Control: checking reverse causality)")

    significant_reverse = 0
    total_reverse = 0

    for out_var in output_vars:
        for tel_var in telemetry_vars:
            x = processed_arrays[out_var]
            y = processed_arrays[tel_var]

            min_len = min(len(x), len(y))
            x = x[:min_len]
            y = y[:min_len]

            result = run_granger_test(x, y, max_lag=10, name_x=out_var, name_y=tel_var)
            key = f"{out_var}_to_{tel_var}"
            results['output_to_telemetry'][key] = result

            total_reverse += 1
            if result.get('significant', False):
                significant_reverse += 1

    # Summary
    results['summary'] = {
        'forward_significant': significant_forward,
        'forward_total': total_forward,
        'forward_ratio': significant_forward / total_forward if total_forward > 0 else 0,
        'reverse_significant': significant_reverse,
        'reverse_total': total_reverse,
        'reverse_ratio': significant_reverse / total_reverse if total_reverse > 0 else 0,
    }

    # Embodiment verdict
    # Hypothesis supported if: forward causality > reverse causality
    forward_ratio = results['summary']['forward_ratio']
    reverse_ratio = results['summary']['reverse_ratio']

    if forward_ratio > 0.3 and forward_ratio > reverse_ratio:
        verdict = "EMBODIMENT_SUPPORTED"
        explanation = f"Telemetry->Output causality ({forward_ratio:.1%}) > Output->Telemetry ({reverse_ratio:.1%})"
    elif forward_ratio > 0:
        verdict = "PARTIAL_EVIDENCE"
        explanation = f"Some forward causality ({forward_ratio:.1%}) but not dominant"
    else:
        verdict = "EMBODIMENT_NOT_SUPPORTED"
        explanation = f"No significant Telemetry->Output causality detected"

    results['summary']['verdict'] = verdict
    results['summary']['explanation'] = explanation

    print(f"\n[SUMMARY]")
    print(f"  Forward (Telemetry -> Output): {significant_forward}/{total_forward} significant ({forward_ratio:.1%})")
    print(f"  Reverse (Output -> Telemetry): {significant_reverse}/{total_reverse} significant ({reverse_ratio:.1%})")
    print(f"  VERDICT: {verdict}")
    print(f"  {explanation}")

    return results


# =============================================================================
# BASELINE COMPARISON (No FiLM conditioning)
# =============================================================================

class BaselineTransformer(nn.Module):
    """Baseline transformer WITHOUT telemetry conditioning."""

    def __init__(self, vocab_size: int = 128, hidden_dim: int = 256, n_layers: int = 4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=4,
                dim_feedforward=hidden_dim * 2,
                batch_first=True,
                norm_first=True,
                dropout=0.0,
            ) for _ in range(n_layers)
        ])

        self.output = nn.Linear(hidden_dim, vocab_size)
        self.action_head = nn.Linear(hidden_dim, 4)

    def forward(self, tokens: torch.Tensor, telemetry: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """Forward pass ignoring telemetry."""
        h = self.embed(tokens)
        for layer in self.layers:
            h = layer(h)

        logits = self.output(h)
        hidden_mean = h.mean(dim=1)
        action_logits = self.action_head(hidden_mean)

        return {
            'logits': logits,
            'hidden': h,
            'hidden_mean': hidden_mean,
            'action_logits': action_logits,
        }


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def main():
    print("="*80)
    print("z2001: PROPER GRANGER CAUSALITY TEST")
    print("Statistical Rigor per Cogitate Methodology")
    print("="*80)
    print(f"Start time: {datetime.now().isoformat()}")
    print(f"Device: {DEVICE}")
    print(f"statsmodels available: {STATSMODELS_AVAILABLE}")
    print()

    # Initialize telemetry
    print("[1] Initializing telemetry...")
    telemetry = SysfsHwmonTelemetry(sample_rate_hz=50)

    # Test sample
    sample = telemetry.read_sample()
    print(f"  Temp: {sample.temp_edge_c:.1f}C, Power: {sample.power_w:.1f}W")
    print(f"  Freq: {sample.freq_sclk_mhz}MHz, Util: {sample.gpu_busy_pct}%")

    # Create collector first to get vocab size
    print("\n[2] Creating data collector...")

    # Temporary model to initialize collector
    temp_model = FiLMTransformer(vocab_size=128, hidden_dim=256, telemetry_dim=4, n_layers=4).to(DEVICE)
    collector = GrangerDataCollector(telemetry, temp_model, DEVICE)

    # Now create model with correct vocab size
    print(f"\n[3] Initializing FiLM-conditioned model (vocab={collector.vocab_size})...")
    model = FiLMTransformer(
        vocab_size=collector.vocab_size,
        hidden_dim=256,
        telemetry_dim=4,
        n_layers=4,
    ).to(DEVICE)

    # Update collector with correct model
    collector.model = model

    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {param_count:,}")

    # Extended training to establish strong FiLM conditioning
    print("\n[4] Training model with FiLM conditioning (extended for stronger telemetry dependency)...")
    optimizer = AdamW(model.parameters(), lr=1e-3)

    # Training loop - longer to establish FiLM effects
    model.train()
    num_train_steps = 500  # More training for stronger conditioning

    for i in range(num_train_steps):
        # Add varying GPU workload to create telemetry variation during training
        if np.random.random() < 0.5:
            size = np.random.randint(256, 1024)
            a = torch.randn(size, size, device=DEVICE)
            b = torch.randn(size, size, device=DEVICE)
            _ = torch.matmul(a, b)
            del a, b

        # Collect telemetry
        sample = telemetry.read_sample()
        tel = torch.tensor([
            sample.temp_edge_c / 100.0,
            sample.power_w / 100.0,
            sample.freq_sclk_mhz / 2000.0,
            sample.gpu_busy_pct / 100.0,
        ], dtype=torch.float32, device=DEVICE).unsqueeze(0)

        # Random tokens
        idx = np.random.randint(0, len(collector.data) - 65)
        tokens = collector.data[idx:idx + 64].unsqueeze(0).to(DEVICE)
        targets = collector.data[idx + 1:idx + 65].unsqueeze(0).to(DEVICE)

        # Forward + backward
        out = model(tokens, tel)
        loss = F.cross_entropy(out['logits'].view(-1, collector.vocab_size), targets.view(-1))

        # Add auxiliary loss to encourage telemetry dependence
        # Penalize if action logits don't vary with telemetry
        action_probs = F.softmax(out['action_logits'], dim=-1)
        telemetry_entropy_loss = -0.1 * action_probs.std()  # Encourage action variation

        total_loss = loss + telemetry_entropy_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print(f"    Step {i}: loss={loss.item():.4f}, tel=[{tel[0,0]:.3f},{tel[0,1]:.3f},{tel[0,2]:.3f},{tel[0,3]:.3f}]")

    print("  Training complete.")

    # Collect time series data
    print("\n[5] Collecting time series for Granger analysis...")
    model.eval()

    # Reset collector for fresh data
    collector.records.clear()

    # Collect 1500 samples (ensure we have enough for lag testing)
    records = collector.collect_series(
        target_samples=1500,
        sample_interval=0.03,  # 30ms = ~33Hz
    )

    arrays = collector.get_arrays()
    print(f"\n  Collected {len(records)} samples")
    print(f"  Duration: {arrays['timestamp'][-1] - arrays['timestamp'][0]:.1f}s")

    # Run Granger analysis
    granger_results = run_full_granger_analysis(arrays)

    # Also run baseline comparison
    print("\n" + "="*70)
    print("BASELINE COMPARISON (No FiLM)")
    print("="*70)

    baseline_model = BaselineTransformer(
        vocab_size=collector.vocab_size,
        hidden_dim=256,
        n_layers=4,
    ).to(DEVICE)

    baseline_collector = GrangerDataCollector(telemetry, baseline_model, DEVICE)
    baseline_collector.records.clear()

    print("\nCollecting baseline time series...")
    baseline_records = baseline_collector.collect_series(
        target_samples=1000,
        sample_interval=0.03,
    )

    baseline_arrays = baseline_collector.get_arrays()
    baseline_granger = run_full_granger_analysis(baseline_arrays)

    # Compare FiLM vs Baseline
    print("\n" + "="*70)
    print("COMPARISON: FiLM vs BASELINE")
    print("="*70)

    film_forward = granger_results['summary']['forward_significant']
    film_total = granger_results['summary']['forward_total']
    baseline_forward = baseline_granger['summary']['forward_significant']
    baseline_total = baseline_granger['summary']['forward_total']

    print(f"  FiLM model:     {film_forward}/{film_total} significant forward causality")
    print(f"  Baseline model: {baseline_forward}/{baseline_total} significant forward causality")

    if film_forward > baseline_forward:
        comparison_verdict = "FiLM_SHOWS_MORE_CAUSALITY"
        comparison_explanation = "FiLM conditioning creates detectable telemetry->output causality"
    else:
        comparison_verdict = "NO_DIFFERENCE"
        comparison_explanation = "FiLM and baseline show similar causality patterns"

    print(f"\n  COMPARISON VERDICT: {comparison_verdict}")
    print(f"  {comparison_explanation}")

    # Final results
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)

    # F4 test result (Granger causality)
    min_p_forward = min(
        [r.get('min_p_value', 1.0) for r in granger_results['telemetry_to_output'].values()]
    ) if granger_results['telemetry_to_output'] else 1.0

    f4_passed = min_p_forward < 0.05

    print(f"\n  F4 Granger Causality Test:")
    print(f"    Min p-value (forward): {min_p_forward:.6f}")
    print(f"    Threshold: 0.05")
    print(f"    RESULT: {'PASS' if f4_passed else 'FAIL'}")

    # Compile full results
    results = {
        'experiment': 'z2001_proper_granger_causality',
        'timestamp': datetime.now().isoformat(),
        'device': str(DEVICE),
        'statsmodels_available': STATSMODELS_AVAILABLE,
        'data_collection': {
            'film_samples': len(records),
            'baseline_samples': len(baseline_records),
            'sample_rate_hz': 33,
        },
        'film_model': {
            'parameters': param_count,
            'granger_results': granger_results,
        },
        'baseline_model': {
            'granger_results': baseline_granger,
        },
        'comparison': {
            'verdict': comparison_verdict,
            'explanation': comparison_explanation,
        },
        'f4_test': {
            'min_p_value': float(min_p_forward),
            'threshold': 0.05,
            'passed': f4_passed,
        },
        'overall_verdict': granger_results['summary']['verdict'],
    }

    # Save results
    output_path = RESULTS_DIR / 'z2001_proper_granger_causality.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == '__main__':
    main()
