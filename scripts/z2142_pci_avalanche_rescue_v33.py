#!/usr/bin/env python3
"""
z2142v33: PCI Inversion Rescue via Avalanche Substrate
=======================================================
Tests whether coupling FEEL to an FPGA avalanche substrate can fix the
persistent PCI (Perturbational Complexity Index) inversion problem.

In biological brains, TARGETED perturbations produce HIGHER complexity
than random noise. In digital systems this is typically INVERTED (random >
targeted). Hypothesis: avalanche dynamics near criticality propagate
perturbations "biologically" — targeted perturbations to hub neurons
should produce higher-complexity cascades than random noise.

2x2 design:
  1. Digital-only PCI  — perturb GPU weights, no FPGA
  2. Hybrid PCI        — perturb FPGA neurons, propagate to GPU via loop
  3. FPGA-only PCI     — perturb FPGA, no GPU propagation
  4. Scrambled coupling — perturb FPGA, scramble GPU coupling

Hardware setup:
  sudo HSA_OVERRIDE_GFX_VERSION=11.0.0 PYTORCH_ROCM_ARCH=gfx1100 \\
    venv/bin/python -u scripts/z2142_pci_avalanche_rescue_v33.py
"""

import os, sys, json, math, time, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

# ---------------------------------------------------------------------------
# Add scripts dir to path for bridge import
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from nsram_fpga_bridge import NSRAMFPGABridge, read_gpu_telemetry, read_gpu_temp_c

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEVICE = 'cuda'
BS = 2
SEQ_LEN = 64
N_NEURONS = 8
N_TRIALS = 20          # trials per condition per perturbation type
VG_BASELINE = 0.35     # near-critical operating point
VG_PULSE_DV = 0.15     # perturbation amplitude (V)
PULSE_DURATION_S = 0.01  # 10 ms pulse
CASCADE_WINDOW_S = 0.10  # 100 ms post-perturbation recording window
TELEM_POLL_HZ = 100     # telemetry polling rate during cascade
LORA_RANK = 4
LORA_ALPHA = 8
ANALOG_LAYERS = list(range(4, 12))

RESULTS_PATH = SCRIPT_DIR.parent / 'results' / 'z2142_pci_avalanche_rescue.json'

# ---------------------------------------------------------------------------
# Lempel-Ziv Complexity (inline implementation)
# ---------------------------------------------------------------------------

def lempel_ziv_complexity(binary_string: str) -> float:
    """Compute normalized Lempel-Ziv complexity of a binary string.

    Returns LZc in [0, 1] where 1 = maximum complexity (random).
    Normalization: LZc = c(n) / (n / log2(n))
    """
    n = len(binary_string)
    if n <= 1:
        return 0.0

    s = binary_string
    c = 1  # complexity counter
    l = 1  # current prefix length
    i = 0  # scan start
    k = 1  # current position
    k_max = 1

    while k < n:
        # Check if substring s[i:i+l] exists in s[0:k]
        if s[i:i + l] in s[0:k]:
            l += 1
            k += 1
        else:
            c += 1
            i = k
            k += 1
            l = 1

    # Normalize
    norm = n / math.log2(n) if n > 1 else 1.0
    return c / norm


def spike_raster_to_binary(spike_counts: np.ndarray) -> str:
    """Convert spike count matrix [T, N_NEURONS] to binary string.

    Each neuron-timebin is 1 if spike_count > 0, else 0.
    String is row-major (time-first, then neuron).
    """
    binary = (spike_counts > 0).astype(int)
    return ''.join(str(b) for b in binary.flatten())


def spectral_complexity(spike_counts: np.ndarray) -> float:
    """Entropy of FFT power spectrum of total spike count time series."""
    total = spike_counts.sum(axis=1).astype(float)
    if total.std() < 1e-9:
        return 0.0
    # FFT
    fft_vals = np.abs(np.fft.rfft(total - total.mean()))
    power = fft_vals ** 2
    # Normalize to probability distribution
    total_power = power.sum()
    if total_power < 1e-12:
        return 0.0
    p = power / total_power
    p = p[p > 0]
    entropy = -np.sum(p * np.log2(p))
    # Normalize by log2(len) for [0,1] range
    max_entropy = math.log2(len(power)) if len(power) > 1 else 1.0
    return float(entropy / max_entropy) if max_entropy > 0 else 0.0


# ---------------------------------------------------------------------------
# Minimal FPGAGatedLoRA layer
# ---------------------------------------------------------------------------

class FPGAGatedLoRA(nn.Module):
    """LoRA adapter with gate modulated by FPGA spike signal."""

    def __init__(self, base_linear: nn.Linear, rank: int = LORA_RANK,
                 alpha: float = LORA_ALPHA):
        super().__init__()
        self.base = base_linear
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        d_out, d_in = base_linear.weight.shape
        self.lora_A = nn.Parameter(torch.randn(rank, d_in) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(d_out, rank))
        self.gate = nn.Parameter(torch.zeros(1))
        self.scale = alpha / rank
        self.fpga_signal = 0.0  # updated from bridge telemetry

    def forward(self, x):
        base_out = self.base(x)
        lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scale
        g = torch.sigmoid(self.gate + self.fpga_signal)
        return base_out + g * lora_out


# ---------------------------------------------------------------------------
# Build minimal GPT-2 + FPGAGatedLoRA
# ---------------------------------------------------------------------------

def build_model():
    """Load GPT-2 small and patch select layers with FPGAGatedLoRA."""
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    print("[model] Loading GPT-2 small ...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.eval()

    # Freeze all
    for p in model.parameters():
        p.requires_grad_(False)

    # Patch c_fc in selected layers
    lora_modules = []
    for idx in ANALOG_LAYERS:
        block = model.transformer.h[idx]
        orig = block.mlp.c_fc
        # GPT-2 uses Conv1D (transposed): convert to Linear
        linear = nn.Linear(orig.weight.shape[0], orig.weight.shape[1], bias=True)
        linear.weight.data = orig.weight.data.T.clone()
        linear.bias.data = orig.bias.data.clone()
        lora = FPGAGatedLoRA(linear, LORA_RANK, LORA_ALPHA)
        block.mlp.c_fc = lora
        lora_modules.append(lora)

    model = model.to(DEVICE)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] Trainable params: {n_train:,}")
    return model, tokenizer, lora_modules


# ---------------------------------------------------------------------------
# FPGA cascade recording
# ---------------------------------------------------------------------------

def record_cascade(bridge: NSRAMFPGABridge, duration_s: float = CASCADE_WINDOW_S
                   ) -> np.ndarray:
    """Record spike counts from all 8 neurons over a time window.

    Returns: array of shape [T, 8] where T = number of telemetry samples.
    """
    samples = []
    t0 = time.monotonic()
    poll_interval = 1.0 / TELEM_POLL_HZ
    prev_counts = None

    while time.monotonic() - t0 < duration_s:
        telem = bridge.read_telemetry()
        if telem is not None:
            counts = np.array([n['spike_count'] for n in telem['neurons']])
            if prev_counts is not None:
                # Delta spike counts (handles counter wrap at 65535)
                delta = counts.astype(np.int32) - prev_counts.astype(np.int32)
                delta = np.where(delta < 0, delta + 65536, delta)
                samples.append(delta)
            prev_counts = counts
        time.sleep(poll_interval)

    if len(samples) == 0:
        return np.zeros((1, N_NEURONS), dtype=np.int32)
    return np.array(samples, dtype=np.int32)


def identify_hub_neurons(bridge: NSRAMFPGABridge, n_samples: int = 50
                         ) -> list[int]:
    """Identify hub neurons = those with highest spike rate variance.

    Samples telemetry n_samples times, computes per-neuron spike count
    variance, returns neuron indices sorted by variance (descending).
    """
    all_counts = []
    for _ in range(n_samples):
        telem = bridge.read_telemetry()
        if telem is not None:
            counts = [n['spike_count'] for n in telem['neurons']]
            all_counts.append(counts)
        time.sleep(0.02)

    if len(all_counts) < 5:
        print("[hub] WARNING: insufficient telemetry, using default hub order")
        return list(range(N_NEURONS))

    arr = np.array(all_counts, dtype=np.float64)
    # Compute variance of delta spike counts
    deltas = np.diff(arr, axis=0)
    variance = np.var(deltas, axis=0)
    order = np.argsort(-variance)
    print(f"[hub] Neuron variance: {dict(enumerate(np.round(variance, 2)))}")
    print(f"[hub] Hub order: {order.tolist()}")
    return order.tolist()


# ---------------------------------------------------------------------------
# Perturbation functions
# ---------------------------------------------------------------------------

def apply_targeted_perturbation(bridge: NSRAMFPGABridge,
                                hub_neurons: list[int],
                                n_hubs: int = 2) -> None:
    """Apply brief Vg pulse to top hub neurons."""
    targets = hub_neurons[:n_hubs]
    for nid in targets:
        bridge.set_gate_voltage(nid, VG_BASELINE + VG_PULSE_DV)
    time.sleep(PULSE_DURATION_S)
    for nid in targets:
        bridge.set_gate_voltage(nid, VG_BASELINE)


def apply_random_perturbation(bridge: NSRAMFPGABridge,
                              n_hubs: int = 2) -> None:
    """Apply same total energy but distributed randomly across neurons."""
    total_dv = VG_PULSE_DV * n_hubs
    # Random subset and random amplitudes summing to total_dv
    n_target = random.randint(1, N_NEURONS)
    targets = random.sample(range(N_NEURONS), n_target)
    # Random partition of total_dv
    weights = np.random.dirichlet(np.ones(n_target))
    amplitudes = weights * total_dv

    for nid, amp in zip(targets, amplitudes):
        bridge.set_gate_voltage(nid, VG_BASELINE + amp)
    time.sleep(PULSE_DURATION_S)
    for nid in range(N_NEURONS):
        bridge.set_gate_voltage(nid, VG_BASELINE)


def apply_digital_targeted_perturbation(lora_modules: list, scale: float = 0.1):
    """Perturb GPU LoRA weights directly (targeted = highest-norm neurons)."""
    saved = []
    for lora in lora_modules:
        saved.append(lora.lora_B.data.clone())
        # Target highest-norm rows (= most influential output dims)
        norms = lora.lora_B.data.norm(dim=1)
        top_k = min(lora.lora_B.shape[0] // 4, 64)
        _, top_idx = torch.topk(norms, top_k)
        noise = torch.randn_like(lora.lora_B.data[top_idx]) * scale
        lora.lora_B.data[top_idx] += noise
    return saved


def apply_digital_random_perturbation(lora_modules: list, scale: float = 0.1):
    """Perturb GPU LoRA weights randomly with same total energy."""
    saved = []
    for lora in lora_modules:
        saved.append(lora.lora_B.data.clone())
        noise = torch.randn_like(lora.lora_B.data) * scale
        # Normalize to same total energy as targeted
        top_k = min(lora.lora_B.shape[0] // 4, 64)
        target_energy = (scale ** 2) * top_k * lora.lora_B.shape[1]
        actual_energy = (noise ** 2).sum().item()
        if actual_energy > 0:
            noise *= math.sqrt(target_energy / actual_energy)
        lora.lora_B.data += noise
    return saved


def restore_weights(lora_modules: list, saved: list):
    """Restore saved weights after perturbation."""
    for lora, w in zip(lora_modules, saved):
        lora.lora_B.data.copy_(w)


# ---------------------------------------------------------------------------
# GPU response complexity (residual norms after forward pass)
# ---------------------------------------------------------------------------

def gpu_response_complexity(model, tokenizer, n_tokens: int = 32) -> float:
    """Run a forward pass and return LZc of residual norm trace."""
    input_ids = torch.randint(0, tokenizer.vocab_size, (1, n_tokens),
                              device=DEVICE)
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)

    # Collect hidden state norms across layers
    norms = []
    for hs in outputs.hidden_states:
        norms.append(hs.float().norm(dim=-1).mean().item())
    norms = np.array(norms)

    # Binarize around median
    median = np.median(norms)
    binary = ''.join('1' if n > median else '0' for n in norms)
    return lempel_ziv_complexity(binary)


# ---------------------------------------------------------------------------
# Combined FPGA+GPU complexity
# ---------------------------------------------------------------------------

def combined_complexity(spike_raster: np.ndarray, gpu_lzc: float,
                        weight_fpga: float = 0.6) -> dict:
    """Compute combined complexity metrics from FPGA spike raster + GPU."""
    binary_str = spike_raster_to_binary(spike_raster)
    fpga_lzc = lempel_ziv_complexity(binary_str)
    fpga_spectral = spectral_complexity(spike_raster)
    combined_lzc = weight_fpga * fpga_lzc + (1 - weight_fpga) * gpu_lzc
    return {
        'fpga_lzc': fpga_lzc,
        'gpu_lzc': gpu_lzc,
        'combined_lzc': combined_lzc,
        'spectral': fpga_spectral,
        'raster_shape': list(spike_raster.shape),
        'total_spikes': int(spike_raster.sum()),
    }


# ---------------------------------------------------------------------------
# Condition runners
# ---------------------------------------------------------------------------

def set_all_neurons_vg(bridge: NSRAMFPGABridge, vg: float):
    """Set all 8 neurons to the same gate voltage."""
    for nid in range(N_NEURONS):
        bridge.set_gate_voltage(nid, vg)
    time.sleep(0.05)


def propagate_fpga_to_gpu(bridge: NSRAMFPGABridge, lora_modules: list):
    """Read FPGA telemetry and inject into LoRA gate signals."""
    telem = bridge.read_telemetry()
    if telem is None:
        return
    total = telem['total_spikes']
    mean_vmem = telem['mean_vmem']
    # Normalize to [-1, 1] range
    signal = np.tanh((total - 50) / 50.0) * 0.5 + np.tanh((mean_vmem - 0.5) / 0.3) * 0.5
    for lora in lora_modules:
        lora.fpga_signal = float(signal)


def run_condition_1_digital_only(model, tokenizer, lora_modules, n_trials):
    """Condition 1: Digital-only PCI — perturb GPU weights, no FPGA."""
    print("\n[C1] Digital-only PCI (no FPGA)")
    results = {'targeted': [], 'random': [], 'baseline': []}

    for trial in range(n_trials):
        # Baseline
        lzc_base = gpu_response_complexity(model, tokenizer)
        results['baseline'].append(lzc_base)

        # Targeted
        saved = apply_digital_targeted_perturbation(lora_modules)
        lzc_targ = gpu_response_complexity(model, tokenizer)
        restore_weights(lora_modules, saved)
        results['targeted'].append(lzc_targ)

        # Random
        saved = apply_digital_random_perturbation(lora_modules)
        lzc_rand = gpu_response_complexity(model, tokenizer)
        restore_weights(lora_modules, saved)
        results['random'].append(lzc_rand)

        if (trial + 1) % 5 == 0:
            print(f"  trial {trial+1}/{n_trials}: "
                  f"targ={np.mean(results['targeted'][-5:]):.4f}  "
                  f"rand={np.mean(results['random'][-5:]):.4f}  "
                  f"base={np.mean(results['baseline'][-5:]):.4f}")

    return results


def run_condition_2_hybrid(model, tokenizer, lora_modules, bridge, hub_neurons,
                           n_trials):
    """Condition 2: Hybrid PCI — perturb FPGA, propagate to GPU via loop."""
    print("\n[C2] Hybrid PCI (FPGA -> GPU closed loop)")
    results = {'targeted': [], 'random': [], 'baseline': []}

    for trial in range(n_trials):
        set_all_neurons_vg(bridge, VG_BASELINE)
        time.sleep(0.05)

        # Baseline: no perturbation
        raster_base = record_cascade(bridge)
        propagate_fpga_to_gpu(bridge, lora_modules)
        gpu_base = gpu_response_complexity(model, tokenizer)
        cx_base = combined_complexity(raster_base, gpu_base)
        results['baseline'].append(cx_base)

        # Targeted: pulse hub neurons
        set_all_neurons_vg(bridge, VG_BASELINE)
        time.sleep(0.05)
        apply_targeted_perturbation(bridge, hub_neurons)
        raster_targ = record_cascade(bridge)
        propagate_fpga_to_gpu(bridge, lora_modules)
        gpu_targ = gpu_response_complexity(model, tokenizer)
        cx_targ = combined_complexity(raster_targ, gpu_targ)
        results['targeted'].append(cx_targ)

        # Random: same energy, random distribution
        set_all_neurons_vg(bridge, VG_BASELINE)
        time.sleep(0.05)
        apply_random_perturbation(bridge)
        raster_rand = record_cascade(bridge)
        propagate_fpga_to_gpu(bridge, lora_modules)
        gpu_rand = gpu_response_complexity(model, tokenizer)
        cx_rand = combined_complexity(raster_rand, gpu_rand)
        results['random'].append(cx_rand)

        if (trial + 1) % 5 == 0:
            t_lzc = np.mean([r['combined_lzc'] for r in results['targeted'][-5:]])
            r_lzc = np.mean([r['combined_lzc'] for r in results['random'][-5:]])
            b_lzc = np.mean([r['combined_lzc'] for r in results['baseline'][-5:]])
            print(f"  trial {trial+1}/{n_trials}: "
                  f"targ={t_lzc:.4f}  rand={r_lzc:.4f}  base={b_lzc:.4f}  "
                  f"PCI={t_lzc/r_lzc:.3f}" if r_lzc > 0 else "")

    return results


def run_condition_3_fpga_only(bridge, hub_neurons, n_trials):
    """Condition 3: FPGA-only PCI — perturb FPGA, no GPU propagation."""
    print("\n[C3] FPGA-only PCI (no GPU coupling)")
    results = {'targeted': [], 'random': [], 'baseline': []}

    for trial in range(n_trials):
        set_all_neurons_vg(bridge, VG_BASELINE)
        time.sleep(0.05)

        # Baseline
        raster_base = record_cascade(bridge)
        cx_base = combined_complexity(raster_base, 0.0, weight_fpga=1.0)
        results['baseline'].append(cx_base)

        # Targeted
        set_all_neurons_vg(bridge, VG_BASELINE)
        time.sleep(0.05)
        apply_targeted_perturbation(bridge, hub_neurons)
        raster_targ = record_cascade(bridge)
        cx_targ = combined_complexity(raster_targ, 0.0, weight_fpga=1.0)
        results['targeted'].append(cx_targ)

        # Random
        set_all_neurons_vg(bridge, VG_BASELINE)
        time.sleep(0.05)
        apply_random_perturbation(bridge)
        raster_rand = record_cascade(bridge)
        cx_rand = combined_complexity(raster_rand, 0.0, weight_fpga=1.0)
        results['random'].append(cx_rand)

        if (trial + 1) % 5 == 0:
            t_lzc = np.mean([r['fpga_lzc'] for r in results['targeted'][-5:]])
            r_lzc = np.mean([r['fpga_lzc'] for r in results['random'][-5:]])
            print(f"  trial {trial+1}/{n_trials}: "
                  f"targ={t_lzc:.4f}  rand={r_lzc:.4f}  "
                  f"PCI={t_lzc/r_lzc:.3f}" if r_lzc > 0 else "")

    return results


def run_condition_4_scrambled(model, tokenizer, lora_modules, bridge,
                              hub_neurons, n_trials):
    """Condition 4: Scrambled coupling — perturb FPGA, scramble GPU coupling."""
    print("\n[C4] Scrambled coupling (FPGA perturbed, GPU coupling shuffled)")
    results = {'targeted': [], 'random': [], 'baseline': []}

    def scrambled_propagate():
        """Like propagate_fpga_to_gpu but randomly shuffles which LoRA gets which signal."""
        telem = bridge.read_telemetry()
        if telem is None:
            return
        total = telem['total_spikes']
        mean_vmem = telem['mean_vmem']
        signal = np.tanh((total - 50) / 50.0) * 0.5 + np.tanh((mean_vmem - 0.5) / 0.3) * 0.5
        # Scramble: each LoRA module gets a random signal unrelated to actual FPGA state
        for lora in lora_modules:
            lora.fpga_signal = float(random.gauss(0, abs(signal) + 0.01))

    for trial in range(n_trials):
        set_all_neurons_vg(bridge, VG_BASELINE)
        time.sleep(0.05)

        # Baseline
        raster_base = record_cascade(bridge)
        scrambled_propagate()
        gpu_base = gpu_response_complexity(model, tokenizer)
        cx_base = combined_complexity(raster_base, gpu_base)
        results['baseline'].append(cx_base)

        # Targeted
        set_all_neurons_vg(bridge, VG_BASELINE)
        time.sleep(0.05)
        apply_targeted_perturbation(bridge, hub_neurons)
        raster_targ = record_cascade(bridge)
        scrambled_propagate()
        gpu_targ = gpu_response_complexity(model, tokenizer)
        cx_targ = combined_complexity(raster_targ, gpu_targ)
        results['targeted'].append(cx_targ)

        # Random
        set_all_neurons_vg(bridge, VG_BASELINE)
        time.sleep(0.05)
        apply_random_perturbation(bridge)
        raster_rand = record_cascade(bridge)
        scrambled_propagate()
        gpu_rand = gpu_response_complexity(model, tokenizer)
        cx_rand = combined_complexity(raster_rand, gpu_rand)
        results['random'].append(cx_rand)

        if (trial + 1) % 5 == 0:
            t_lzc = np.mean([r['combined_lzc'] for r in results['targeted'][-5:]])
            r_lzc = np.mean([r['combined_lzc'] for r in results['random'][-5:]])
            print(f"  trial {trial+1}/{n_trials}: "
                  f"targ={t_lzc:.4f}  rand={r_lzc:.4f}  "
                  f"PCI={t_lzc/r_lzc:.3f}" if r_lzc > 0 else "")

    return results


# ---------------------------------------------------------------------------
# Analysis & summary
# ---------------------------------------------------------------------------

def summarize_condition(name: str, results: dict, is_digital: bool = False
                        ) -> dict:
    """Compute PCI ratio and statistics for one condition."""
    if is_digital:
        targ_vals = np.array(results['targeted'])
        rand_vals = np.array(results['random'])
        base_vals = np.array(results['baseline'])
    else:
        targ_vals = np.array([r['combined_lzc'] for r in results['targeted']])
        rand_vals = np.array([r['combined_lzc'] for r in results['random']])
        base_vals = np.array([r['combined_lzc'] for r in results['baseline']])

    targ_mean = float(np.mean(targ_vals))
    rand_mean = float(np.mean(rand_vals))
    base_mean = float(np.mean(base_vals))
    pci_ratio = targ_mean / rand_mean if rand_mean > 1e-9 else 0.0

    # Mann-Whitney U test: targeted vs random
    from scipy.stats import mannwhitneyu
    try:
        stat, pval = mannwhitneyu(targ_vals, rand_vals, alternative='greater')
    except Exception:
        stat, pval = 0.0, 1.0

    # Spectral complexity (only for FPGA conditions)
    spectral_targ = spectral_rand = 0.0
    if not is_digital and len(results['targeted']) > 0:
        spectral_targ = float(np.mean([r['spectral'] for r in results['targeted']]))
        spectral_rand = float(np.mean([r['spectral'] for r in results['random']]))

    summary = {
        'condition': name,
        'lzc_targeted': targ_mean,
        'lzc_random': rand_mean,
        'lzc_baseline': base_mean,
        'pci_ratio': pci_ratio,
        'pci_correct_direction': pci_ratio > 1.0,
        'mann_whitney_p': float(pval),
        'significant_p05': pval < 0.05,
        'spectral_targeted': spectral_targ,
        'spectral_random': spectral_rand,
        'n_trials': len(targ_vals),
    }

    direction = "CORRECT (targeted > random)" if pci_ratio > 1.0 else \
                "INVERTED (random > targeted)"
    sig = "*" if pval < 0.05 else "ns"
    print(f"\n  [{name}] PCI ratio = {pci_ratio:.4f}  ({direction})  "
          f"p={pval:.4f} {sig}")
    print(f"    LZc: targeted={targ_mean:.4f}  random={rand_mean:.4f}  "
          f"baseline={base_mean:.4f}")
    if spectral_targ > 0:
        print(f"    Spectral: targeted={spectral_targ:.4f}  "
              f"random={spectral_rand:.4f}")

    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 72)
    print("z2142v33: PCI Inversion Rescue via Avalanche Substrate")
    print("=" * 72)
    t_start = time.time()

    # GPU telemetry
    gpu_telem = read_gpu_telemetry()
    print(f"[gpu] temp={gpu_telem['temp_c']:.1f}C  "
          f"power={gpu_telem['power_w']:.1f}W  "
          f"vddgfx={gpu_telem['vddgfx_mv']}mV")

    # Build model
    model, tokenizer, lora_modules = build_model()

    # Connect FPGA
    print("\n[fpga] Connecting to FPGA bridge ...")
    bridge = NSRAMFPGABridge(port='/dev/ttyUSB1', baudrate=115200)

    try:
        # Set all neurons to near-critical Vg
        print(f"[fpga] Setting all neurons to Vg={VG_BASELINE}V (near-critical)")
        set_all_neurons_vg(bridge, VG_BASELINE)

        # Send GPU temperature to FPGA
        temp_c = read_gpu_temp_c()
        bridge.set_temperature(temp_c + 273.15)
        time.sleep(0.5)  # let FPGA stabilize

        # Identify hub neurons
        print("\n[fpga] Identifying hub neurons (high-variance spike rate) ...")
        hub_neurons = identify_hub_neurons(bridge, n_samples=50)

        # Verify FPGA is alive
        telem = bridge.read_telemetry()
        if telem is not None:
            print(f"[fpga] Alive: total_spikes={telem['total_spikes']}  "
                  f"mean_vmem={telem['mean_vmem']:.3f}V  "
                  f"mean_bvpar={telem['mean_bvpar']:.3f}V")
        else:
            print("[fpga] WARNING: no telemetry response, FPGA may be offline")

        # ---------------------------------------------------------------
        # Run all 4 conditions
        # ---------------------------------------------------------------
        all_results = {}

        # C1: Digital-only PCI
        c1 = run_condition_1_digital_only(model, tokenizer, lora_modules,
                                          N_TRIALS)
        all_results['C1_digital_only'] = c1

        # C2: Hybrid PCI
        c2 = run_condition_2_hybrid(model, tokenizer, lora_modules, bridge,
                                    hub_neurons, N_TRIALS)
        all_results['C2_hybrid'] = c2

        # C3: FPGA-only PCI
        c3 = run_condition_3_fpga_only(bridge, hub_neurons, N_TRIALS)
        all_results['C3_fpga_only'] = c3

        # C4: Scrambled coupling
        c4 = run_condition_4_scrambled(model, tokenizer, lora_modules, bridge,
                                       hub_neurons, N_TRIALS)
        all_results['C4_scrambled'] = c4

        # ---------------------------------------------------------------
        # Summarize
        # ---------------------------------------------------------------
        print("\n" + "=" * 72)
        print("SUMMARY: PCI Ratios (targeted/random LZc)")
        print("=" * 72)

        summaries = []
        summaries.append(summarize_condition('C1_digital_only', c1,
                                             is_digital=True))
        summaries.append(summarize_condition('C2_hybrid', c2))
        summaries.append(summarize_condition('C3_fpga_only', c3))
        summaries.append(summarize_condition('C4_scrambled', c4))

        # Key question
        c2_pci = summaries[1]['pci_ratio']
        c1_pci = summaries[0]['pci_ratio']
        rescue = c2_pci > 1.0 and c1_pci < 1.0

        print("\n" + "-" * 72)
        print(f"KEY RESULT: Hybrid PCI ratio = {c2_pci:.4f}")
        print(f"            Digital PCI ratio = {c1_pci:.4f}")
        if rescue:
            print("  >>> RESCUE SUCCESSFUL: Avalanche substrate corrects PCI direction!")
        elif c2_pci > 1.0:
            print("  >>> Hybrid PCI correct direction (but digital also correct)")
        else:
            print("  >>> Rescue NOT achieved: hybrid PCI still inverted")
        print("-" * 72)

        # ---------------------------------------------------------------
        # Save results
        # ---------------------------------------------------------------
        elapsed = time.time() - t_start

        # Serialize: convert numpy types for JSON
        def make_serializable(obj):
            if isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(v) for v in obj]
            elif isinstance(obj, (np.integer,)):
                return int(obj)
            elif isinstance(obj, (np.floating,)):
                return float(obj)
            elif isinstance(obj, (np.bool_,)):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        output = {
            'experiment': 'z2142v33_pci_avalanche_rescue',
            'timestamp': time.strftime('%Y%m%d_%H%M%S'),
            'elapsed_s': elapsed,
            'config': {
                'vg_baseline': VG_BASELINE,
                'vg_pulse_dv': VG_PULSE_DV,
                'pulse_duration_s': PULSE_DURATION_S,
                'cascade_window_s': CASCADE_WINDOW_S,
                'n_trials': N_TRIALS,
                'lora_rank': LORA_RANK,
                'analog_layers': ANALOG_LAYERS,
                'n_neurons': N_NEURONS,
            },
            'hub_neurons': hub_neurons,
            'gpu_telemetry': gpu_telem,
            'summaries': make_serializable(summaries),
            'rescue_achieved': rescue,
            'raw_results': {
                'C1_digital_only': make_serializable(c1),
                'C2_hybrid': make_serializable(c2),
                'C3_fpga_only': make_serializable(c3),
                'C4_scrambled': make_serializable(c4),
            },
        }

        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(RESULTS_PATH, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {RESULTS_PATH}")
        print(f"Total time: {elapsed:.1f}s")

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        # Restore safe Vg and close
        for nid in range(N_NEURONS):
            try:
                bridge.set_gate_voltage(nid, VG_BASELINE)
            except Exception:
                pass
        bridge.close()
        print("[fpga] Bridge closed.")


if __name__ == '__main__':
    main()
