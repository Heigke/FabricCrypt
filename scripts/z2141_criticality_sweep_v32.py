#!/usr/bin/env python3
"""
z2141v32: Criticality Sweep → FEEL Test Battery Integration
============================================================
Sweeps FPGA control parameters (global Vg) to find the criticality sweet spot
where LM coupling is strongest AND avalanche dynamics show power-law behavior.

Scientific background:
  - Beggs & Plenz (2003): neuronal avalanches follow power-law size distributions
  - Shew et al. (Nature Comms 2020): distance-to-criticality trades off performance
  - Diaz-Alvarez et al. (Science Advances 2019): avalanche/criticality in nanoscale nets
  - Hypothesis: sweet spot in Vg space where avalanche dynamics are near-critical
    AND LM embodiment coupling is maximized

Run:
  HSA_OVERRIDE_GFX_VERSION=11.0.0 venv/bin/python -u scripts/z2141_criticality_sweep_v32.py
"""

import os, sys, json, math, time, struct, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from collections import deque

warnings.filterwarnings('ignore')

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PATHS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / 'scripts'))
RESULTS_DIR = BASE / 'results'
RESULTS_DIR.mkdir(exist_ok=True)
OUTPUT_JSON = RESULTS_DIR / 'z2141_criticality_sweep.json'

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONSTANTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DEVICE = 'cuda'
BS = 4
SEQ_LEN = 128
N_NEURONS = 8

# Sweep parameters
VG_MIN = 0.10
VG_MAX = 0.50
VG_STEPS = 20
N_TELEM_WINDOWS = 1000       # telemetry samples per Vg setting
AVALANCHE_DT_MS = 1.0        # temporal binning window for avalanche detection
TELEM_POLL_HZ = 200          # telemetry poll rate during collection

# FEEL eval
N_EVAL_BATCHES_SWEEP = 10    # quick eval per Vg step
N_EVAL_BATCHES_BATTERY = 30  # full eval at critical/sub/supercritical

# FPGA
FPGA_PORT = '/dev/ttyUSB1'
FPGA_BAUD = 115200

# DVFS
DVFS_SETTLE_S = 1.5
SCLK_LOW_CAL = 600.0
SCLK_HIGH_CAL = 2900.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CRITICALITY ANALYSIS FUNCTIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def detect_avalanches(spike_matrix, dt_ms=1.0):
    """Detect avalanches from spike count matrix.

    Args:
        spike_matrix: np.array of shape (n_windows, n_neurons) — spike counts
        dt_ms: temporal bin width in ms

    Returns:
        List of avalanche sizes (total spikes in each contiguous active period).
        An avalanche starts when any neuron fires and ends when a bin has zero
        total spikes.
    """
    # Sum across neurons for each time bin
    total_per_bin = spike_matrix.sum(axis=1)  # (n_windows,)

    avalanches = []
    current_size = 0
    in_avalanche = False

    for count in total_per_bin:
        if count > 0:
            current_size += count
            in_avalanche = True
        else:
            if in_avalanche:
                avalanches.append(int(current_size))
                current_size = 0
                in_avalanche = False

    # Close final avalanche if still active
    if in_avalanche and current_size > 0:
        avalanches.append(int(current_size))

    return avalanches


def fit_power_law(sizes, x_min=1):
    """MLE power-law fit with KS goodness-of-fit test.

    Discrete MLE: alpha = 1 + n / sum(ln(x / (x_min - 0.5)))
    KS test: D = max|S(x) - P(x)| where P(x) is fitted power-law CDF.

    Args:
        sizes: list of avalanche sizes (integers >= 1)
        x_min: minimum size for fit (discard smaller)

    Returns:
        dict with alpha, ks_D, ks_p, n_fit
    """
    sizes = np.array([s for s in sizes if s >= x_min], dtype=float)
    if len(sizes) < 10:
        return {'alpha': float('nan'), 'ks_D': 1.0, 'ks_p': 0.0, 'n_fit': len(sizes)}

    n = len(sizes)

    # Discrete MLE (Clauset et al. 2009, eq. 3.7)
    alpha = 1.0 + n / np.sum(np.log(sizes / (x_min - 0.5)))

    # KS goodness-of-fit
    # Empirical CDF
    sorted_s = np.sort(sizes)
    ecdf = np.arange(1, n + 1) / n

    # Theoretical CDF: P(X <= x) = 1 - (x/x_min)^(1-alpha)  (continuous approx)
    tcdf = 1.0 - (sorted_s / x_min) ** (1.0 - alpha)
    tcdf = np.clip(tcdf, 0, 1)

    ks_D = float(np.max(np.abs(ecdf - tcdf)))

    # Approximate p-value via KS distribution (scipy fallback)
    try:
        from scipy.stats import kstwobign
        ks_stat = ks_D * np.sqrt(n)
        ks_p = float(kstwobign.sf(ks_stat))
    except ImportError:
        # Rough p-value: D > 1.36/sqrt(n) rejects at alpha=0.05
        ks_p = 1.0 if ks_D < 1.36 / np.sqrt(n) else 0.01

    return {
        'alpha': float(alpha),
        'ks_D': float(ks_D),
        'ks_p': float(ks_p),
        'n_fit': int(n),
    }


def compute_branching_ratio(spike_matrix):
    """Branching ratio σ = <A(t+1) / A(t)> where A(t) is total activity at time t.

    σ < 1: subcritical (activity dies out)
    σ ≈ 1: critical (sustained activity)
    σ > 1: supercritical (runaway activity)
    """
    total = spike_matrix.sum(axis=1).astype(float)

    ratios = []
    for t in range(len(total) - 1):
        if total[t] > 0:
            ratios.append(total[t + 1] / total[t])

    if len(ratios) < 5:
        return float('nan')
    return float(np.mean(ratios))


def compute_dynamic_range(spike_matrix, vg_values=None):
    """Dynamic range Δ = 10 * log10(F_0.9 / F_0.1) in dB.

    Measures the range of input intensities producing distinguishable outputs.
    Here we use total spike rate as the response variable and approximate
    input intensity from the Vg parameter via self-excitation level.
    """
    total_rate = spike_matrix.sum(axis=1).astype(float)
    if len(total_rate) < 10:
        return 0.0

    sorted_rate = np.sort(total_rate[total_rate > 0])
    if len(sorted_rate) < 10:
        return 0.0

    f_10 = sorted_rate[int(0.1 * len(sorted_rate))]
    f_90 = sorted_rate[int(0.9 * len(sorted_rate))]

    if f_10 <= 0:
        f_10 = 1e-6
    return float(10.0 * np.log10(max(f_90 / f_10, 1.0)))


def compute_isi_cv_per_neuron(spike_matrix, dt_ms=1.0):
    """ISI coefficient of variation per neuron + mean across neurons.

    CV = std(ISI) / mean(ISI).  CV ≈ 1 for Poisson, CV > 1 for bursty.
    """
    cvs = []
    for nid in range(spike_matrix.shape[1]):
        # Find bins where this neuron fires
        fire_bins = np.where(spike_matrix[:, nid] > 0)[0]
        if len(fire_bins) < 3:
            cvs.append(0.0)
            continue
        isis = np.diff(fire_bins) * dt_ms
        mean_isi = np.mean(isis)
        if mean_isi > 0:
            cvs.append(float(np.std(isis) / mean_isi))
        else:
            cvs.append(0.0)
    return cvs, float(np.mean(cvs)) if cvs else 0.0


def compute_spike_pattern_entropy(spike_matrix, max_patterns=256):
    """Shannon entropy of binary spike pattern distribution.

    Binarize each time bin (neuron active/inactive), compute pattern distribution,
    return H in bits.  Max H = log2(2^n_neurons) = n_neurons bits.
    """
    n_neurons = spike_matrix.shape[1]
    # Binarize: any spikes = 1
    binary = (spike_matrix > 0).astype(int)

    # Encode each row as integer pattern (up to 8 neurons = 8-bit)
    patterns = np.zeros(len(binary), dtype=int)
    for nid in range(min(n_neurons, 8)):
        patterns += binary[:, nid] << nid

    # Count unique patterns
    unique, counts = np.unique(patterns, return_counts=True)
    probs = counts / counts.sum()
    entropy = -np.sum(probs * np.log2(probs + 1e-12))
    return float(entropy)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FPGA TELEMETRY COLLECTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def collect_telemetry_windows(bridge, n_windows, poll_hz=200):
    """Collect n_windows of telemetry, return spike_matrix (n_windows, 8)."""
    spike_matrix = np.zeros((n_windows, N_NEURONS), dtype=int)
    dt = 1.0 / poll_hz
    collected = 0
    retries = 0
    max_retries = n_windows * 3  # allow some timeouts

    while collected < n_windows and retries < max_retries:
        telem = bridge.read_telemetry()
        if telem is not None:
            for nid in range(N_NEURONS):
                spike_matrix[collected, nid] = telem['neurons'][nid]['spike_count']
            collected += 1
        else:
            retries += 1
        time.sleep(dt)

    if collected < n_windows:
        print(f"    WARNING: only collected {collected}/{n_windows} windows "
              f"({retries} retries)")
        spike_matrix = spike_matrix[:collected]

    return spike_matrix


def set_global_vg(bridge, vg):
    """Set gate voltage for all 8 neurons."""
    for nid in range(N_NEURONS):
        bridge.set_gate_voltage(nid, vg)
    time.sleep(0.05)  # let FPGA settle


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GPU TELEMETRY (for FEEL coupling)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def read_gpu_temp_c():
    """Read AMD GPU junction temperature from sysfs hwmon."""
    for p in Path('/sys/class/hwmon').iterdir():
        try:
            name = (p / 'name').read_text().strip()
            if name == 'amdgpu':
                return int((p / 'temp1_input').read_text().strip()) / 1000.0
        except (FileNotFoundError, PermissionError, ValueError):
            continue
    return 0.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DVFS CONTROL
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DVFS_AVAILABLE = False
DVFS_PATH = None


def find_dvfs_sysfs():
    global DVFS_AVAILABLE, DVFS_PATH
    for card in ['card1', 'card0']:
        p = f'/sys/class/drm/{card}/device/power_dpm_force_performance_level'
        if os.path.exists(p):
            try:
                with open(p, 'r') as f:
                    f.read().strip()
                DVFS_PATH = p
                DVFS_AVAILABLE = True
                return
            except Exception:
                pass


def set_dvfs_level(level, wait=True):
    if not DVFS_AVAILABLE:
        return
    torch.cuda.synchronize()
    name = {0: 'low', 1: 'auto', 2: 'high'}[level]
    try:
        with open(DVFS_PATH, 'w') as f:
            f.write(name)
    except Exception as e:
        print(f"  [DVFS] Write failed: {e}")
        return
    if wait:
        time.sleep(DVFS_SETTLE_S)
        torch.cuda.synchronize()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MINIMAL GPT-2 + FPGA-GATED LoRA (E4 pattern)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class FPGAGatedLoRA(nn.Module):
    """LoRA adapter where gate signal comes from FPGA telemetry (spike rates).

    The FPGA neuron bank provides a hardware-substrate-dependent gating signal:
    gate = sigmoid(linear(fpga_feature_vec)).  When kill switch is ON (no spikes),
    the gate collapses and embodiment coupling vanishes.
    """

    def __init__(self, original_linear, layer_idx, rank=8, alpha=16):
        super().__init__()
        self.original = original_linear
        self.layer_idx = layer_idx
        self.rank = rank
        self.scaling = alpha / rank

        # Detect Conv1D (GPT-2) vs nn.Linear
        if hasattr(original_linear, 'nf'):
            in_f, out_f = original_linear.nx, original_linear.nf
        else:
            in_f = original_linear.weight.shape[1]
            out_f = original_linear.weight.shape[0]

        dtype = original_linear.weight.dtype
        self.lora_A = nn.Parameter(torch.randn(rank, in_f, dtype=dtype) * 0.01)
        self.lora_B = nn.Parameter(torch.randn(out_f, rank, dtype=dtype) * 0.001)

        # FPGA gate: maps 8-neuron feature vector to rank-dim gate
        # Features: [spike_rate_0..7] normalized
        self.gate_proj = nn.Linear(N_NEURONS, rank, dtype=torch.float32)
        nn.init.normal_(self.gate_proj.weight, std=0.1)
        nn.init.constant_(self.gate_proj.bias, 0.0)

        self._fpga_features = None  # set externally per forward pass

    def set_fpga_features(self, feat_vec):
        """Set FPGA feature vector (numpy array of shape (8,))."""
        self._fpga_features = feat_vec

    def forward(self, x):
        # Base forward (frozen GPT-2 linear)
        if hasattr(self.original, 'nf'):
            base = torch.addmm(self.original.bias, x.view(-1, x.size(-1)),
                                self.original.weight)
            base = base.view(*x.shape[:-1], self.original.nf)
        else:
            base = F.linear(x, self.original.weight, self.original.bias)

        # LoRA path
        x_cast = x.to(self.lora_A.dtype)
        lora_mid = F.linear(x_cast, self.lora_A)
        lora_out = F.linear(lora_mid, self.lora_B) * self.scaling

        # FPGA gate
        if self._fpga_features is not None:
            dev = self.gate_proj.weight.device
            feat = torch.from_numpy(self._fpga_features).float().to(dev)
            gate = torch.sigmoid(self.gate_proj(feat))  # [rank]
            gate_scalar = gate.mean()
        else:
            gate_scalar = 0.5  # fallback: no FPGA data

        self.last_gate_scalar = gate_scalar
        result = base + (lora_out * gate_scalar).to(x.dtype)
        return result


def patch_model_with_fpga_lora(model, layers, rank=8, alpha=16):
    """Patch GPT-2 c_fc projections with FPGAGatedLoRA."""
    patched = {}
    total_params = 0
    for layer_idx in layers:
        block = model.transformer.h[layer_idx]
        orig = block.mlp.c_fc
        lora = FPGAGatedLoRA(orig, layer_idx, rank=rank, alpha=alpha)
        lora = lora.to(orig.weight.device)
        block.mlp.c_fc = lora
        patched[layer_idx] = lora
        n_p = sum(p.numel() for p in lora.parameters() if p.requires_grad)
        total_params += n_p
    print(f"  [FPGAGatedLoRA] Patched {len(patched)} layers, {total_params} params")
    return patched


def set_fpga_features_all(patched_layers, feat_vec):
    """Broadcast FPGA features to all patched layers."""
    for lora in patched_layers.values():
        lora.set_fpga_features(feat_vec)


def make_fpga_feature_vec(telem):
    """Extract normalized 8-dim feature vector from FPGA telemetry."""
    if telem is None:
        return np.zeros(N_NEURONS, dtype=np.float32)
    spikes = np.array([n['spike_count'] for n in telem['neurons']], dtype=np.float32)
    # Normalize to [0, 1] range
    max_s = spikes.max()
    if max_s > 0:
        spikes = spikes / max_s
    return spikes


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FEEL EVALUATION (minimal: PPL + embodiment ratio)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def eval_ppl(model, tokenizer, n_batches, patched_layers, bridge, kill_switch=False):
    """Evaluate perplexity with FPGA coupling.

    If kill_switch=True, disable FPGA avalanche physics (zero spikes).
    """
    model.eval()
    if kill_switch and bridge is not None:
        bridge.set_kill_switch(True)
        time.sleep(0.1)

    total_loss = 0.0
    total_tokens = 0
    text = tokenizer.encode("The meaning of consciousness in artificial systems is",
                            return_tensors='pt')

    with torch.no_grad():
        for _ in range(n_batches):
            # Random input
            input_ids = torch.randint(0, tokenizer.vocab_size, (BS, SEQ_LEN),
                                      device=DEVICE)
            labels = input_ids.clone()
            labels[:, :-1] = input_ids[:, 1:]
            labels[:, -1] = -100

            # FPGA telemetry → gate features
            if bridge is not None:
                telem = bridge.read_telemetry()
                feat = make_fpga_feature_vec(telem)
            else:
                feat = np.zeros(N_NEURONS, dtype=np.float32)
            set_fpga_features_all(patched_layers, feat)

            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            n_tok = (labels != -100).sum().item()
            total_loss += loss.item() * n_tok
            total_tokens += n_tok

    if kill_switch and bridge is not None:
        bridge.set_kill_switch(False)
        time.sleep(0.1)

    ppl = math.exp(total_loss / max(total_tokens, 1))
    return ppl


def eval_embodiment_ratio(model, tokenizer, n_batches, patched_layers, bridge):
    """Embodiment ratio = PPL_killed / PPL_coupled.

    > 1.0 means FPGA coupling helps.  The FEEL 'kill-shot' test.
    """
    ppl_coupled = eval_ppl(model, tokenizer, n_batches, patched_layers, bridge,
                           kill_switch=False)
    ppl_killed = eval_ppl(model, tokenizer, n_batches, patched_layers, bridge,
                          kill_switch=True)
    ratio = ppl_killed / max(ppl_coupled, 1e-6)
    return ratio, ppl_coupled, ppl_killed


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FEEL BATTERY SUBSET (T4, T7, T13, T16)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_feel_battery_subset(model, tokenizer, patched_layers, bridge, vg, label):
    """Run key FEEL tests at a specific Vg operating point."""
    print(f"\n  === FEEL Battery @ Vg={vg:.3f} ({label}) ===")
    set_global_vg(bridge, vg)
    time.sleep(0.3)

    results = {'vg': vg, 'label': label}

    # T4: Embodiment ratio (PPL_killed / PPL_coupled)
    print(f"    T4: Embodiment ratio...")
    ratio, ppl_c, ppl_k = eval_embodiment_ratio(
        model, tokenizer, N_EVAL_BATCHES_BATTERY, patched_layers, bridge)
    results['T4_embodiment_ratio'] = round(ratio, 4)
    results['T4_ppl_coupled'] = round(ppl_c, 4)
    results['T4_ppl_killed'] = round(ppl_k, 4)
    results['T4_pass'] = ratio > 1.01
    print(f"      ratio={ratio:.4f}  coupled={ppl_c:.2f}  killed={ppl_k:.2f}  "
          f"{'PASS' if results['T4_pass'] else 'FAIL'}")

    # T7: Kill-shot — same as T4 but threshold is higher (1.05)
    results['T7_killshot_ratio'] = round(ratio, 4)
    results['T7_pass'] = ratio > 1.05
    print(f"    T7: Kill-shot ratio={ratio:.4f}  "
          f"{'PASS' if results['T7_pass'] else 'FAIL'}")

    # T13: DVFS kill-shot — compare low-SCLK vs high-SCLK PPL
    if DVFS_AVAILABLE:
        print(f"    T13: DVFS kill-shot...")
        set_dvfs_level(0)  # low
        ppl_low = eval_ppl(model, tokenizer, N_EVAL_BATCHES_BATTERY // 2,
                           patched_layers, bridge)
        set_dvfs_level(2)  # high
        ppl_high = eval_ppl(model, tokenizer, N_EVAL_BATCHES_BATTERY // 2,
                            patched_layers, bridge)
        set_dvfs_level(1)  # auto
        dvfs_ratio = abs(ppl_low - ppl_high) / max(min(ppl_low, ppl_high), 1e-6)
        results['T13_ppl_low'] = round(ppl_low, 4)
        results['T13_ppl_high'] = round(ppl_high, 4)
        results['T13_dvfs_ratio'] = round(dvfs_ratio, 4)
        results['T13_pass'] = dvfs_ratio > 0.01
        print(f"      low={ppl_low:.2f}  high={ppl_high:.2f}  "
              f"ratio={dvfs_ratio:.4f}  "
              f"{'PASS' if results['T13_pass'] else 'FAIL'}")
    else:
        results['T13_pass'] = None
        print(f"    T13: DVFS not available, skipped")

    # T16: Proprioception — do gate values track FPGA spike rates?
    print(f"    T16: Proprioception (gate-spike correlation)...")
    gate_vals = []
    spike_rates = []
    for trial in range(10):
        # Vary Vg slightly to create different spike regimes
        trial_vg = vg + (trial - 5) * 0.01
        trial_vg = max(0.05, min(0.60, trial_vg))
        set_global_vg(bridge, trial_vg)
        time.sleep(0.05)

        telem = bridge.read_telemetry()
        feat = make_fpga_feature_vec(telem)
        set_fpga_features_all(patched_layers, feat)

        # Forward pass to get gate values
        input_ids = torch.randint(0, tokenizer.vocab_size, (1, SEQ_LEN),
                                  device=DEVICE)
        with torch.no_grad():
            model(input_ids=input_ids)

        # Collect gate scalar from patched layers
        gates = []
        for lora in patched_layers.values():
            if hasattr(lora, 'last_gate_scalar'):
                g = lora.last_gate_scalar
                gates.append(float(g.item()) if torch.is_tensor(g) else float(g))
        if gates:
            gate_vals.append(np.mean(gates))
        if telem is not None:
            spike_rates.append(telem['total_spikes'])

    # Restore Vg
    set_global_vg(bridge, vg)

    if len(gate_vals) >= 5 and len(spike_rates) >= 5:
        from scipy.stats import spearmanr
        rho, p = spearmanr(gate_vals, spike_rates)
        results['T16_spearman_rho'] = round(float(rho), 4)
        results['T16_p_value'] = round(float(p), 6)
        results['T16_pass'] = abs(rho) > 0.3 and p < 0.1
        print(f"      rho={rho:.4f}  p={p:.4f}  "
              f"{'PASS' if results['T16_pass'] else 'FAIL'}")
    else:
        results['T16_pass'] = False
        results['T16_spearman_rho'] = 0.0
        print(f"      Insufficient data")

    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PART A: CRITICALITY SWEEP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_criticality_sweep(bridge, model, tokenizer, patched_layers):
    """Sweep Vg from VG_MIN to VG_MAX, measure criticality + FEEL at each step."""
    print("\n" + "=" * 70)
    print("PART A: CRITICALITY SWEEP")
    print(f"  Vg range: [{VG_MIN:.2f}, {VG_MAX:.2f}] in {VG_STEPS} steps")
    print(f"  Telemetry windows per step: {N_TELEM_WINDOWS}")
    print(f"  FEEL eval batches per step: {N_EVAL_BATCHES_SWEEP}")
    print("=" * 70)

    vg_values = np.linspace(VG_MIN, VG_MAX, VG_STEPS)
    sweep_results = []

    for step_i, vg in enumerate(vg_values):
        t0 = time.time()
        print(f"\n--- Step {step_i+1}/{VG_STEPS}: Vg = {vg:.4f}V ---")

        # Set global Vg
        set_global_vg(bridge, vg)
        time.sleep(0.2)  # let FPGA stabilize

        # Send GPU temperature for thermal coupling
        gpu_temp = read_gpu_temp_c()
        bridge.set_temperature(gpu_temp + 273.15)

        # Collect telemetry
        print(f"  Collecting {N_TELEM_WINDOWS} telemetry windows...")
        spike_matrix = collect_telemetry_windows(bridge, N_TELEM_WINDOWS,
                                                  poll_hz=TELEM_POLL_HZ)

        # Criticality metrics
        print(f"  Computing criticality metrics...")
        avalanches = detect_avalanches(spike_matrix, dt_ms=AVALANCHE_DT_MS)
        n_avalanches = len(avalanches)

        pl_fit = fit_power_law(avalanches)
        branching = compute_branching_ratio(spike_matrix)
        dyn_range = compute_dynamic_range(spike_matrix)
        isi_cvs, isi_cv_mean = compute_isi_cv_per_neuron(spike_matrix,
                                                          dt_ms=AVALANCHE_DT_MS)
        entropy = compute_spike_pattern_entropy(spike_matrix)

        # Quick FEEL eval
        print(f"  Running quick FEEL eval ({N_EVAL_BATCHES_SWEEP} batches)...")
        telem = bridge.read_telemetry()
        feat = make_fpga_feature_vec(telem)
        set_fpga_features_all(patched_layers, feat)

        emb_ratio, ppl_coupled, ppl_killed = eval_embodiment_ratio(
            model, tokenizer, N_EVAL_BATCHES_SWEEP, patched_layers, bridge)

        # Total spikes and rates
        total_spikes = int(spike_matrix.sum())
        mean_rate = total_spikes / max(spike_matrix.shape[0], 1)

        step_result = {
            'vg': round(float(vg), 4),
            'step': step_i,
            'n_windows': int(spike_matrix.shape[0]),
            'total_spikes': total_spikes,
            'mean_spike_rate': round(mean_rate, 4),
            'n_avalanches': n_avalanches,
            'mean_avalanche_size': round(float(np.mean(avalanches)), 4) if avalanches else 0.0,
            'max_avalanche_size': int(max(avalanches)) if avalanches else 0,
            'power_law_alpha': round(pl_fit['alpha'], 4) if not math.isnan(pl_fit['alpha']) else None,
            'power_law_ks_D': round(pl_fit['ks_D'], 4),
            'power_law_ks_p': round(pl_fit['ks_p'], 4),
            'power_law_n_fit': pl_fit['n_fit'],
            'branching_ratio': round(branching, 4) if not math.isnan(branching) else None,
            'dynamic_range_dB': round(dyn_range, 4),
            'isi_cv_mean': round(isi_cv_mean, 4),
            'isi_cv_per_neuron': [round(c, 4) for c in isi_cvs],
            'entropy_bits': round(entropy, 4),
            'ppl_coupled': round(ppl_coupled, 4),
            'ppl_killed': round(ppl_killed, 4),
            'embodiment_ratio': round(emb_ratio, 4),
            'gpu_temp_c': round(gpu_temp, 1),
            'elapsed_s': round(time.time() - t0, 1),
        }
        sweep_results.append(step_result)

        # Summary
        alpha_str = f"{pl_fit['alpha']:.2f}" if not math.isnan(pl_fit['alpha']) else "N/A"
        sigma_str = f"{branching:.3f}" if not math.isnan(branching) else "N/A"
        print(f"  Results: avalanches={n_avalanches}  α={alpha_str}  "
              f"σ={sigma_str}  H={entropy:.2f}bits  "
              f"PPL={ppl_coupled:.2f}  emb={emb_ratio:.4f}  "
              f"({time.time()-t0:.1f}s)")

    return sweep_results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PART B: SWEET SPOT DETECTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def find_sweet_spots(sweep_results):
    """Analyze sweep data to find critical and embodiment sweet spots."""
    print("\n" + "=" * 70)
    print("PART B: SWEET SPOT DETECTION")
    print("=" * 70)

    vg_vals = np.array([r['vg'] for r in sweep_results])

    # 1. Criticality sweet spot: α ≈ -1.5 and σ ≈ 1.0
    alphas = []
    sigmas = []
    for r in sweep_results:
        a = r['power_law_alpha']
        s = r['branching_ratio']
        alphas.append(a if a is not None else float('nan'))
        sigmas.append(s if s is not None else float('nan'))
    alphas = np.array(alphas)
    sigmas = np.array(sigmas)

    # Distance to criticality: |α - 1.5| + |σ - 1.0|
    # (α from MLE is positive; theoretical power-law exponent ≈ 1.5 for 2D criticality)
    dist_crit = np.abs(alphas - 1.5) + np.abs(sigmas - 1.0)
    valid_crit = np.isfinite(dist_crit)

    if valid_crit.any():
        idx_crit = np.nanargmin(dist_crit)
        vg_critical = float(vg_vals[idx_crit])
        print(f"\n  Criticality sweet spot: Vg = {vg_critical:.4f}V")
        print(f"    α = {alphas[idx_crit]:.3f} (target ≈ 1.5)")
        print(f"    σ = {sigmas[idx_crit]:.3f} (target ≈ 1.0)")
        print(f"    dist_crit = {dist_crit[idx_crit]:.4f}")
    else:
        vg_critical = float(vg_vals[len(vg_vals) // 2])
        print(f"\n  WARNING: No valid criticality data; using midpoint Vg={vg_critical:.4f}")

    # 2. Embodiment sweet spot: max embodiment_ratio
    emb_ratios = np.array([r['embodiment_ratio'] for r in sweep_results])
    idx_emb = np.argmax(emb_ratios)
    vg_embodiment = float(vg_vals[idx_emb])
    print(f"\n  Embodiment sweet spot: Vg = {vg_embodiment:.4f}V")
    print(f"    embodiment_ratio = {emb_ratios[idx_emb]:.4f}")
    print(f"    PPL_coupled = {sweep_results[idx_emb]['ppl_coupled']:.2f}")

    # 3. Overlap analysis
    overlap_dist = abs(vg_critical - vg_embodiment)
    vg_range = VG_MAX - VG_MIN
    overlap_frac = 1.0 - (overlap_dist / vg_range)
    overlaps = overlap_dist < (vg_range / VG_STEPS * 2)  # within 2 sweep steps

    print(f"\n  Sweet spot overlap:")
    print(f"    Criticality Vg = {vg_critical:.4f}")
    print(f"    Embodiment  Vg = {vg_embodiment:.4f}")
    print(f"    Distance = {overlap_dist:.4f}V ({overlap_frac*100:.1f}% overlap)")
    print(f"    Overlap within 2 steps: {'YES' if overlaps else 'NO'}")

    # Pick subcritical, critical, supercritical points
    vg_sub = max(VG_MIN, vg_critical - 0.10)
    vg_super = min(VG_MAX, vg_critical + 0.10)

    return {
        'vg_critical': vg_critical,
        'vg_embodiment': vg_embodiment,
        'vg_subcritical': round(vg_sub, 4),
        'vg_supercritical': round(vg_super, 4),
        'overlap_distance': round(overlap_dist, 4),
        'overlap_fraction': round(overlap_frac, 4),
        'sweet_spots_overlap': overlaps,
        'alpha_at_critical': round(float(alphas[idx_crit]), 4) if valid_crit.any() else None,
        'sigma_at_critical': round(float(sigmas[idx_crit]), 4) if valid_crit.any() else None,
        'emb_at_critical': round(float(emb_ratios[idx_crit]) if valid_crit.any() else 0.0, 4),
        'emb_at_embodiment': round(float(emb_ratios[idx_emb]), 4),
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PART C: FEEL BATTERY AT CRITICAL POINT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_battery_comparison(model, tokenizer, patched_layers, bridge, sweet_spots):
    """Run FEEL test subset at subcritical, critical, and supercritical Vg."""
    print("\n" + "=" * 70)
    print("PART C: FEEL BATTERY — CRITICALITY COMPARISON")
    print("=" * 70)

    conditions = [
        (sweet_spots['vg_subcritical'], 'subcritical'),
        (sweet_spots['vg_critical'], 'critical'),
        (sweet_spots['vg_supercritical'], 'supercritical'),
    ]

    battery_results = {}
    for vg, label in conditions:
        res = run_feel_battery_subset(model, tokenizer, patched_layers, bridge,
                                       vg, label)
        battery_results[label] = res

    # Summary comparison
    print("\n  === BATTERY COMPARISON SUMMARY ===")
    print(f"  {'Metric':<25} {'Subcritical':>12} {'Critical':>12} {'Supercritical':>12}")
    print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*12}")
    for key in ['T4_embodiment_ratio', 'T4_ppl_coupled', 'T7_killshot_ratio']:
        vals = []
        for label in ['subcritical', 'critical', 'supercritical']:
            v = battery_results[label].get(key, 'N/A')
            vals.append(f"{v:>12.4f}" if isinstance(v, (int, float)) else f"{'N/A':>12}")
        print(f"  {key:<25} {vals[0]} {vals[1]} {vals[2]}")

    # Does criticality improve FEEL metrics?
    crit_emb = battery_results['critical'].get('T4_embodiment_ratio', 0)
    sub_emb = battery_results['subcritical'].get('T4_embodiment_ratio', 0)
    super_emb = battery_results['supercritical'].get('T4_embodiment_ratio', 0)

    criticality_helps = crit_emb > sub_emb and crit_emb > super_emb
    print(f"\n  Criticality improves embodiment: "
          f"{'YES' if criticality_helps else 'NO'}")
    print(f"    sub={sub_emb:.4f}  crit={crit_emb:.4f}  super={super_emb:.4f}")

    battery_results['criticality_helps_embodiment'] = criticality_helps
    return battery_results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    t_start = time.time()
    print("=" * 70)
    print("z2141v32: Criticality Sweep → FEEL Test Battery Integration")
    print("=" * 70)
    print(f"  HSA_OVERRIDE_GFX_VERSION = {os.environ.get('HSA_OVERRIDE_GFX_VERSION', 'NOT SET')}")
    print(f"  FPGA port: {FPGA_PORT}")
    print(f"  Vg sweep: [{VG_MIN}, {VG_MAX}] in {VG_STEPS} steps")
    print(f"  Telemetry windows: {N_TELEM_WINDOWS} per step")

    # ── 1. Connect FPGA ──────────────────────────────────────────
    print("\n[1/5] Connecting to FPGA NS-RAM bridge...")
    from nsram_fpga_bridge import NSRAMFPGABridge
    bridge = NSRAMFPGABridge(port=FPGA_PORT, baudrate=FPGA_BAUD)
    # Quick sanity check
    telem = bridge.read_telemetry()
    if telem is not None:
        print(f"  Connected. Total spikes: {telem['total_spikes']}  "
              f"Vmem: {telem['mean_vmem']:.3f}V")
    else:
        print("  WARNING: No telemetry response (FPGA may be offline)")

    # ── 2. Load GPT-2 ────────────────────────────────────────────
    print("\n[2/5] Loading GPT-2 small...")
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2').to(DEVICE)
    model.eval()

    # Freeze base model
    for p in model.parameters():
        p.requires_grad = False
    print(f"  GPT-2 loaded. {sum(p.numel() for p in model.parameters())/1e6:.1f}M params (frozen)")

    # ── 3. Patch with FPGAGatedLoRA ──────────────────────────────
    print("\n[3/5] Patching with FPGAGatedLoRA...")
    lora_layers = list(range(4, 12))
    patched_layers = patch_model_with_fpga_lora(model, lora_layers, rank=8, alpha=16)

    # Quick training to establish gate-spike coupling
    print("  Quick LoRA training (100 steps) for gate calibration...")
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-3, weight_decay=0.01)

    model.train()
    for step in range(100):
        input_ids = torch.randint(0, tokenizer.vocab_size, (BS, SEQ_LEN), device=DEVICE)
        labels = input_ids.clone()
        labels[:, :-1] = input_ids[:, 1:]
        labels[:, -1] = -100

        # FPGA features during training
        telem = bridge.read_telemetry()
        feat = make_fpga_feature_vec(telem)
        set_fpga_features_all(patched_layers, feat)

        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        optimizer.zero_grad()

        if (step + 1) % 25 == 0:
            print(f"    step {step+1}/100  loss={loss.item():.4f}")

    model.eval()
    print("  Training complete.")

    # ── 4. Find DVFS ─────────────────────────────────────────────
    print("\n[4/5] Checking DVFS...")
    find_dvfs_sysfs()

    # ── 5. Run experiment ────────────────────────────────────────
    print("\n[5/5] Running criticality sweep + FEEL battery...")

    # Part A: Sweep
    sweep_results = run_criticality_sweep(bridge, model, tokenizer, patched_layers)

    # Part B: Sweet spot detection
    sweet_spots = find_sweet_spots(sweep_results)

    # Part C: FEEL battery comparison
    battery_results = run_battery_comparison(model, tokenizer, patched_layers,
                                              bridge, sweet_spots)

    # ── Save results ─────────────────────────────────────────────
    elapsed = time.time() - t_start
    output = {
        'experiment': 'z2141v32_criticality_sweep',
        'timestamp': time.strftime('%Y%m%d_%H%M%S'),
        'elapsed_s': round(elapsed, 1),
        'config': {
            'vg_min': VG_MIN,
            'vg_max': VG_MAX,
            'vg_steps': VG_STEPS,
            'n_telem_windows': N_TELEM_WINDOWS,
            'avalanche_dt_ms': AVALANCHE_DT_MS,
            'n_eval_batches_sweep': N_EVAL_BATCHES_SWEEP,
            'n_eval_batches_battery': N_EVAL_BATCHES_BATTERY,
            'fpga_port': FPGA_PORT,
            'lora_layers': lora_layers,
            'lora_rank': 8,
        },
        'sweep': sweep_results,
        'sweet_spots': sweet_spots,
        'battery': battery_results,
    }

    with open(OUTPUT_JSON, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n{'='*70}")
    print(f"Results saved to: {OUTPUT_JSON}")
    print(f"Total elapsed: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"{'='*70}")

    # ── Final summary ────────────────────────────────────────────
    print(f"\n  CRITICALITY SWEET SPOT:  Vg = {sweet_spots['vg_critical']:.4f}V")
    print(f"  EMBODIMENT SWEET SPOT:  Vg = {sweet_spots['vg_embodiment']:.4f}V")
    print(f"  OVERLAP: {'YES' if sweet_spots['sweet_spots_overlap'] else 'NO'} "
          f"(distance = {sweet_spots['overlap_distance']:.4f}V)")

    crit_helps = battery_results.get('criticality_helps_embodiment', False)
    print(f"  CRITICALITY HELPS EMBODIMENT: {'YES' if crit_helps else 'NO'}")

    bridge.close()
    print("\nBridge closed. Done.")


if __name__ == '__main__':
    main()
