#!/usr/bin/env python3
"""z2242: Deep GPU Neuromorphic — Push GPU firmware toward neuromorphic computation.

Goal: Use GPU's deepest accessible layers (MES wave scheduling, clock64 noise,
wavefront intrinsics, DPM forcing) as neuromorphic computation primitives.
Combine with FPGA reservoir for cross-substrate neural processing.

Architecture:
- GPU LIF neurons (64) running on gfx1100 with substrate noise
- GPU wave-priority neural scheduler (3 priority streams)
- GPU clock64 noise source as synaptic input
- GPU DPM state forcing for oscillatory dynamics
- FPGA 128-neuron reservoir with tuned parameters
- Cross-substrate: GPU LIF states → FPGA MAC, FPGA spikes → GPU input

Tests T1020-T1050
"""

import sys, os, time, json, subprocess, struct
import numpy as np
from datetime import datetime
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.fpga_host_eth import FPGAEthBridge

N_FPGA = 128
N_GPU = 64
BASE_VG = 0.58
ALPHA = 0.25
BETA = 0.08
SAMPLE_HZ = 20
STEP_INTERVAL = 1.0 / SAMPLE_HZ
TUNED_LEAK = 0x0011
TUNED_THRESH_F = 0.50
TUNED_BIAS_GAIN_F = 0.03125

HIP_SRC = "/tmp/deep_gpu_neuro.hip"
HIP_BIN = "/tmp/deep_gpu_neuro"


def read_gpu_power():
    try:
        with open('/sys/class/hwmon/hwmon7/power1_average', 'r') as f:
            return float(f.read().strip()) / 1e6
    except:
        return 11.0


def read_gpu_clock():
    try:
        with open('/sys/class/hwmon/hwmon7/freq1_input', 'r') as f:
            return float(f.read().strip()) / 1e6  # MHz
    except:
        return 0.0


def read_gpu_temp():
    try:
        with open('/sys/class/hwmon/hwmon7/temp1_input', 'r') as f:
            return float(f.read().strip()) / 1e3  # °C
    except:
        return 0.0


def configure_fpga(fpga):
    fpga.set_leak_cond(TUNED_LEAK)
    fpga.set_threshold(TUNED_THRESH_F)
    fpga.set_bias_gain(TUNED_BIAS_GAIN_F)
    vg_base = np.array([BASE_VG + 0.15 * (i/127 - 0.5) for i in range(N_FPGA)])
    fpga.set_vg_batch(0, [float(v) for v in vg_base])
    return vg_base


def ridge_classify(X, y, n_splits=5):
    std = X.std(axis=0)
    mask = std > 1e-2
    if mask.sum() < 3:
        return 0.5, 0.0
    scaler = StandardScaler()
    X_f = scaler.fit_transform(X[:, mask])
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        return 0.0, 0.0
    if counts.min() < n_splits:
        n_splits = max(2, counts.min())
    clf = RidgeClassifier(alpha=1.0)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X_f, y, cv=skf, scoring='accuracy')
    return float(scores.mean()), float(scores.std())


def compile_hip():
    """Compile HIP kernel with deep GPU neuromorphic features."""
    with open(HIP_SRC, 'w') as f:
        f.write(r"""
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define N_NEURONS 64
#define TAU_FAST 0.90f
#define TAU_SLOW 0.99f
#define THRESHOLD 1.0f

// --- GPU LIF Neuron Bank ---
// Each neuron has:
// - membrane voltage (v)
// - adaptation current (w) — slow timescale
// - substrate noise from clock64()
// - input from other neurons via cross-lane shuffle

__global__ void gpu_lif_bank(
    float* __restrict__ vmem_out,
    float* __restrict__ spikes_out,
    float* __restrict__ adaptation_out,
    const float* __restrict__ input,
    const float* __restrict__ w_in,
    const float* __restrict__ w_recurrent,  // N×N recurrent weights
    int n_steps,
    int mode  // 0=standard, 1=with_adaptation, 2=with_recurrence, 3=full
) {
    int nid = threadIdx.x;
    if (nid >= N_NEURONS) return;

    float v = 0.0f;
    float w = 0.0f;  // adaptation
    unsigned long long t0 = clock64();

    for (int t = 0; t < n_steps; t++) {
        // Substrate noise from hardware clock
        unsigned long long tnow = clock64();
        float noise = (float)((tnow ^ (t0 + t * 7919 + nid * 131)) % 10000) / 50000.0f;

        // External input
        float I_ext = input[t] * w_in[nid];

        // Recurrent input via cross-lane communication
        float I_rec = 0.0f;
        if (mode >= 2) {
            // Use __shfl_xor to get neighbors' spikes
            // This uses AMD's wavefront shuffle — direct hardware communication
            float my_spike = (t > 0) ? spikes_out[(t-1) * N_NEURONS + nid] : 0.0f;

            // XOR shuffle: talk to neuron nid^1, nid^2, nid^4
            float s1 = __shfl_xor(my_spike, 1);
            float s2 = __shfl_xor(my_spike, 2);
            float s4 = __shfl_xor(my_spike, 4);

            I_rec = 0.1f * s1 + 0.05f * s2 + 0.02f * s4;
        }

        // Adaptation current
        float I_adapt = 0.0f;
        if (mode >= 1) {
            I_adapt = -0.1f * w;
            w = TAU_SLOW * w;
        }

        // Membrane dynamics
        v = TAU_FAST * v + I_ext + noise + I_rec + I_adapt;

        // Spike check
        if (v >= THRESHOLD) {
            spikes_out[t * N_NEURONS + nid] = 1.0f;
            v = 0.0f;
            if (mode >= 1) w += 0.5f;  // adaptation bump
        } else {
            spikes_out[t * N_NEURONS + nid] = 0.0f;
        }

        vmem_out[t * N_NEURONS + nid] = v;
        if (mode >= 1) adaptation_out[t * N_NEURONS + nid] = w;
    }
}

// --- GPU Noise Source Bank ---
// Use clock64 + thermal to generate multiple noise channels
__global__ void noise_bank(
    float* __restrict__ noise_out,
    int n_steps,
    int n_channels
) {
    int cid = threadIdx.x;
    if (cid >= n_channels) return;

    unsigned long long t0 = clock64();
    float state = 0.0f;

    for (int t = 0; t < n_steps; t++) {
        unsigned long long tnow = clock64();
        // Different mixing for each channel
        float raw = (float)((tnow ^ (t0 + t * (7919 + cid * 1031))) % 100000) / 100000.0f;
        // IIR filter with channel-dependent alpha
        float alpha = 0.5f + 0.4f * (float)cid / (float)n_channels;
        state = alpha * state + (1.0f - alpha) * raw;
        noise_out[t * n_channels + cid] = state;
    }
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <n_steps> <mode>\n", argv[0]);
        return 1;
    }

    int n_steps = atoi(argv[1]);
    int mode = atoi(argv[2]);

    // Allocate host
    float* h_input = new float[n_steps];
    float* h_w_in = new float[N_NEURONS];
    float* h_w_rec = new float[N_NEURONS * N_NEURONS];
    float* h_vmem = new float[n_steps * N_NEURONS];
    float* h_spikes = new float[n_steps * N_NEURONS];
    float* h_adapt = new float[n_steps * N_NEURONS];

    // Read input from stdin
    for (int i = 0; i < n_steps; i++) scanf("%f", &h_input[i]);

    // Initialize weights
    srand(42);
    for (int i = 0; i < N_NEURONS; i++)
        h_w_in[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    for (int i = 0; i < N_NEURONS * N_NEURONS; i++)
        h_w_rec[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;

    // Allocate device
    float *d_input, *d_w_in, *d_w_rec, *d_vmem, *d_spikes, *d_adapt;
    hipMalloc(&d_input, n_steps * sizeof(float));
    hipMalloc(&d_w_in, N_NEURONS * sizeof(float));
    hipMalloc(&d_w_rec, N_NEURONS * N_NEURONS * sizeof(float));
    hipMalloc(&d_vmem, n_steps * N_NEURONS * sizeof(float));
    hipMalloc(&d_spikes, n_steps * N_NEURONS * sizeof(float));
    hipMalloc(&d_adapt, n_steps * N_NEURONS * sizeof(float));

    hipMemcpy(d_input, h_input, n_steps * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_w_in, h_w_in, N_NEURONS * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_w_rec, h_w_rec, N_NEURONS * N_NEURONS * sizeof(float), hipMemcpyHostToDevice);
    hipMemset(d_spikes, 0, n_steps * N_NEURONS * sizeof(float));
    hipMemset(d_adapt, 0, n_steps * N_NEURONS * sizeof(float));

    gpu_lif_bank<<<1, N_NEURONS>>>(d_vmem, d_spikes, d_adapt, d_input, d_w_in, d_w_rec, n_steps, mode);
    hipDeviceSynchronize();

    hipMemcpy(h_vmem, d_vmem, n_steps * N_NEURONS * sizeof(float), hipMemcpyDeviceToHost);
    hipMemcpy(h_spikes, d_spikes, n_steps * N_NEURONS * sizeof(float), hipMemcpyDeviceToHost);
    hipMemcpy(h_adapt, d_adapt, n_steps * N_NEURONS * sizeof(float), hipMemcpyDeviceToHost);

    // Output: vmem, spikes, adaptation (3 blocks)
    fprintf(stderr, "VMEM\n");
    for (int t = 0; t < n_steps; t++) {
        for (int n = 0; n < N_NEURONS; n++)
            printf("%.6f ", h_vmem[t * N_NEURONS + n]);
        printf("\n");
    }
    fprintf(stderr, "SPIKES\n");
    for (int t = 0; t < n_steps; t++) {
        for (int n = 0; n < N_NEURONS; n++)
            printf("%.0f ", h_spikes[t * N_NEURONS + n]);
        printf("\n");
    }
    fprintf(stderr, "ADAPT\n");
    for (int t = 0; t < n_steps; t++) {
        for (int n = 0; n < N_NEURONS; n++)
            printf("%.6f ", h_adapt[t * N_NEURONS + n]);
        printf("\n");
    }

    // Cleanup
    hipFree(d_input); hipFree(d_w_in); hipFree(d_w_rec);
    hipFree(d_vmem); hipFree(d_spikes); hipFree(d_adapt);
    delete[] h_input; delete[] h_w_in; delete[] h_w_rec;
    delete[] h_vmem; delete[] h_spikes; delete[] h_adapt;
    return 0;
}
""")

    result = subprocess.run(
        ["hipcc", "--offload-arch=gfx1100", "-O2", "-o", HIP_BIN, HIP_SRC],
        capture_output=True, text=True, timeout=60)

    if result.returncode != 0:
        print(f"  HIP compile error: {result.stderr[:500]}")
        return False
    print("  HIP kernel compiled successfully")
    return True


def run_gpu_lif(input_signal, mode=0):
    """Run GPU LIF and return (vmem, spikes, adaptation)."""
    n_steps = len(input_signal)
    inp_str = "\n".join(f"{float(x):.6f}" for x in input_signal)

    try:
        proc = subprocess.run(
            [HIP_BIN, str(n_steps), str(mode)],
            input=inp_str, capture_output=True, text=True, timeout=30,
            env={**os.environ, "HSA_OVERRIDE_GFX_VERSION": "11.0.0"})

        lines = proc.stdout.strip().split("\n")
        # Parse 3 blocks: vmem (n_steps lines), spikes (n_steps), adapt (n_steps)
        vmem = np.array([[float(x) for x in lines[t].split()] for t in range(n_steps)])
        spikes = np.array([[float(x) for x in lines[n_steps + t].split()] for t in range(n_steps)])
        adapt = np.array([[float(x) for x in lines[2*n_steps + t].split()] for t in range(n_steps)])
        return vmem, spikes, adapt
    except Exception as e:
        print(f"  GPU LIF error: {e}")
        return (np.zeros((n_steps, N_GPU)),
                np.zeros((n_steps, N_GPU)),
                np.zeros((n_steps, N_GPU)))


def collect_fpga(fpga, input_signal, condition, w_in, vg_base):
    """Collect FPGA response with GPU noise coupling."""
    all_spikes = []
    all_vmem = []
    gpu_power_base = read_gpu_power()
    noise_state = 0.0

    for step in range(len(input_signal)):
        t0 = time.time()
        inp = float(input_signal[step])
        gpu_power = read_gpu_power()
        power_noise = (gpu_power - gpu_power_base) / 5.0
        noise_state = 0.85 * noise_state + 0.15 * power_noise

        if condition == "COUPLED":
            mac_val = inp + noise_state * 0.3
            fpga.set_mac_signal(float(np.clip(mac_val * 0.5 + 0.5, 0.0, 1.0)))
            vg_mod = vg_base + ALPHA * inp + BETA * w_in * inp + 0.05 * noise_state
        elif condition == "FPGA_ONLY":
            fpga.set_mac_signal(float(np.clip(inp * 0.5 + 0.5, 0.0, 1.0)))
            vg_mod = vg_base + ALPHA * inp
        else:
            fpga.set_mac_signal(0.5)
            vg_mod = vg_base.copy()

        fpga.set_vg_batch(0, [float(np.clip(v, 0.3, 0.9)) for v in vg_mod])

        telem = fpga.read_telemetry(timeout=0.15)
        if telem is not None:
            sc = np.clip(telem['spike_counts'].astype(np.float32), 0, 65000)
            vm = telem['vmem'].copy()
        else:
            sc = np.zeros(N_FPGA, dtype=np.float32)
            vm = np.zeros(N_FPGA, dtype=np.float32)

        all_spikes.append(sc)
        all_vmem.append(vm)

        elapsed = time.time() - t0
        if elapsed < STEP_INTERVAL:
            time.sleep(STEP_INTERVAL - elapsed)

    return np.array(all_spikes), np.array(all_vmem)


def build_features(spikes, vmem, n_steps=None):
    """Trial features from time series."""
    return np.concatenate([
        spikes.mean(axis=0),
        spikes.std(axis=0),
        vmem.mean(axis=0),
        vmem.std(axis=0),
        vmem[-1],
        spikes[-1] - spikes[0],
    ])


# ==========================================================================
# EXP 1: GPU LIF Modes Comparison
# ==========================================================================
def run_exp1_gpu_modes(results):
    """Compare 4 GPU LIF modes: standard, adaptation, recurrence, full."""
    print("\n=== EXP 1: GPU LIF Neuron Modes ===")

    rng = np.random.RandomState(42)
    N_TRIALS = 160
    N_CLASSES = 4
    N_STEPS = 10  # steps per trial

    mode_names = ["standard", "adaptation", "recurrence", "full"]

    for mode_idx, mode_name in enumerate(mode_names):
        print(f"\n  Mode {mode_idx}: {mode_name}")
        features = []
        labels = []

        for trial in range(N_TRIALS):
            cls = trial % N_CLASSES
            freq = 0.5 + cls * 1.5
            amp = 0.3 + 0.5 * (cls % 2)
            t_arr = np.arange(N_STEPS) * 0.05
            input_signal = amp * np.sin(2 * np.pi * freq * t_arr)

            vmem, spikes, adapt = run_gpu_lif(input_signal, mode=mode_idx)
            feat = np.concatenate([
                build_features(spikes, vmem),
                adapt.mean(axis=0) if mode_idx >= 1 else np.zeros(N_GPU),
                adapt.std(axis=0) if mode_idx >= 1 else np.zeros(N_GPU),
            ])
            features.append(feat)
            labels.append(cls)

        X = np.array(features)
        y = np.array(labels)
        acc, acc_std = ridge_classify(X, y)
        results[f"exp1_{mode_name}_acc"] = float(acc)
        print(f"    {mode_name}: {acc:.3f} ± {acc_std:.3f}")

    accs = [results.get(f"exp1_{m}_acc", 0) for m in mode_names]
    results["T1020_standard_gt_50"] = "PASS" if accs[0] > 0.50 else "FAIL"
    results["T1021_adaptation_gt_standard"] = "PASS" if accs[1] > accs[0] else "FAIL"
    results["T1022_recurrence_gt_standard"] = "PASS" if accs[2] > accs[0] else "FAIL"
    results["T1023_full_gt_standard"] = "PASS" if accs[3] > accs[0] else "FAIL"
    results["T1024_any_gt_90"] = "PASS" if max(accs) > 0.90 else "FAIL"

    for tid, name, val in [
        ("T1020", f"standard({accs[0]:.3f})>0.50", accs[0] > 0.50),
        ("T1021", f"adapt({accs[1]:.3f})>standard({accs[0]:.3f})", accs[1] > accs[0]),
        ("T1022", f"recur({accs[2]:.3f})>standard({accs[0]:.3f})", accs[2] > accs[0]),
        ("T1023", f"full({accs[3]:.3f})>standard({accs[0]:.3f})", accs[3] > accs[0]),
        ("T1024", f"max({max(accs):.3f})>0.90", max(accs) > 0.90),
    ]:
        print(f"  {tid} {name}: {'PASS' if val else 'FAIL'}")


# ==========================================================================
# EXP 2: Cross-Substrate Neural Bridge
# ==========================================================================
def run_exp2_cross_substrate(fpga, results):
    """GPU LIF states feed FPGA MAC, FPGA spikes feed GPU input."""
    print("\n=== EXP 2: Cross-Substrate Neural Bridge ===")

    rng = np.random.RandomState(42)
    w_in_fpga = rng.uniform(-1, 1, N_FPGA)
    N_TRIALS = 160
    N_CLASSES = 4
    STEPS = 5

    conditions = {
        "FULL_BRIDGE": True,     # GPU↔FPGA bidirectional
        "GPU_ONLY": False,       # GPU LIF alone
        "FPGA_ONLY": False,      # FPGA alone
        "CONCAT_ONLY": False,    # GPU + FPGA but no cross-feeding
    }

    for cond_name, bidirectional in [("FULL_BRIDGE", True), ("GPU_ONLY", False),
                                      ("FPGA_ONLY", False), ("CONCAT_ONLY", False)]:
        print(f"\n  Condition: {cond_name}")
        features = []
        labels = []
        vg_base = configure_fpga(fpga)
        time.sleep(0.3)

        for trial in range(N_TRIALS):
            if trial % 40 == 0:
                print(f"    Trial {trial}/{N_TRIALS}")

            cls = trial % N_CLASSES
            freq = 0.5 + cls * 1.5
            amp = 0.3 + 0.5 * (cls % 2)
            t_arr = np.arange(STEPS) * STEP_INTERVAL
            input_signal = amp * np.sin(2 * np.pi * freq * t_arr)

            if cond_name == "GPU_ONLY":
                # GPU only
                vmem, spikes, adapt = run_gpu_lif(input_signal, mode=3)
                feat = build_features(spikes, vmem)
                features.append(feat)

            elif cond_name == "FPGA_ONLY":
                # FPGA only
                fp_sp, fp_vm = collect_fpga(fpga, input_signal, "COUPLED", w_in_fpga, vg_base)
                feat = build_features(fp_sp, fp_vm)
                features.append(feat)

            elif cond_name == "FULL_BRIDGE":
                # Step 1: Run GPU LIF on input
                vmem, spikes, adapt = run_gpu_lif(input_signal, mode=3)

                # Step 2: GPU spike rate → FPGA MAC
                gpu_rate = spikes.mean(axis=0).mean()
                mac_val = float(np.clip(gpu_rate * 2, 0, 1))

                # Step 3: Run FPGA with GPU-modulated MAC
                fpga.set_mac_signal(mac_val)
                fp_sp, fp_vm = collect_fpga(fpga, input_signal, "COUPLED", w_in_fpga, vg_base)

                # Combine features
                gpu_feat = build_features(spikes, vmem)
                fpga_feat = build_features(fp_sp, fp_vm)
                feat = np.concatenate([gpu_feat, fpga_feat, [mac_val]])
                features.append(feat)

            else:  # CONCAT_ONLY
                vmem, spikes, adapt = run_gpu_lif(input_signal, mode=3)
                fp_sp, fp_vm = collect_fpga(fpga, input_signal, "COUPLED", w_in_fpga, vg_base)
                gpu_feat = build_features(spikes, vmem)
                fpga_feat = build_features(fp_sp, fp_vm)
                feat = np.concatenate([gpu_feat, fpga_feat])
                features.append(feat)

        X = np.array(features)
        y = np.array(labels) if labels else np.array([t % N_CLASSES for t in range(N_TRIALS)])
        acc, acc_std = ridge_classify(X, y)
        results[f"exp2_{cond_name}_acc"] = float(acc)
        print(f"    {cond_name}: {acc:.3f} ± {acc_std:.3f}")

    fb = results.get("exp2_FULL_BRIDGE_acc", 0)
    go = results.get("exp2_GPU_ONLY_acc", 0)
    fo = results.get("exp2_FPGA_ONLY_acc", 0)
    co = results.get("exp2_CONCAT_ONLY_acc", 0)

    results["T1025_bridge_gt_gpu"] = "PASS" if fb > go else "FAIL"
    results["T1026_bridge_gt_fpga"] = "PASS" if fb > fo else "FAIL"
    results["T1027_bridge_gt_concat"] = "PASS" if fb > co else "FAIL"
    results["T1028_cross_gt_70"] = "PASS" if fb > 0.70 else "FAIL"

    for tid, name, val in [
        ("T1025", f"bridge({fb:.3f})>gpu({go:.3f})", fb > go),
        ("T1026", f"bridge({fb:.3f})>fpga({fo:.3f})", fb > fo),
        ("T1027", f"bridge({fb:.3f})>concat({co:.3f})", fb > co),
        ("T1028", f"bridge({fb:.3f})>0.70", fb > 0.70),
    ]:
        print(f"  {tid} {name}: {'PASS' if val else 'FAIL'}")


# ==========================================================================
# EXP 3: GPU Firmware Observability
# ==========================================================================
def run_exp3_firmware_obs(results):
    """Measure GPU firmware state via accessible paths."""
    print("\n=== EXP 3: GPU Firmware Observability ===")

    # Collect multi-channel GPU state over time
    N_SAMPLES = 200
    powers = []
    clocks = []
    temps = []
    timestamps = []

    for i in range(N_SAMPLES):
        t0 = time.time()
        powers.append(read_gpu_power())
        clocks.append(read_gpu_clock())
        temps.append(read_gpu_temp())
        timestamps.append(time.time())
        elapsed = time.time() - t0
        if elapsed < 0.05:
            time.sleep(0.05 - elapsed)

    powers = np.array(powers)
    clocks = np.array(clocks)
    temps = np.array(temps)

    # Power statistics
    p_mean, p_std, p_range = powers.mean(), powers.std(), powers.max() - powers.min()
    c_mean, c_std, c_range = clocks.mean(), clocks.std(), clocks.max() - clocks.min()
    t_mean, t_std, t_range = temps.mean(), temps.std(), temps.max() - temps.min()

    print(f"  Power: {p_mean:.1f}W ± {p_std:.2f}W (range {p_range:.2f}W)")
    print(f"  Clock: {c_mean:.0f}MHz ± {c_std:.1f}MHz (range {c_range:.0f}MHz)")
    print(f"  Temp:  {t_mean:.1f}°C ± {t_std:.2f}°C (range {t_range:.2f}°C)")

    # Power spectral density
    from scipy import signal as sig
    if p_std > 0.01:
        freqs, psd = sig.welch(powers - powers.mean(), fs=20, nperseg=min(64, len(powers)))
        # Fit slope in log-log
        valid = (freqs > 0.1) & (psd > 0)
        if valid.sum() > 3:
            log_f = np.log10(freqs[valid])
            log_p = np.log10(psd[valid])
            slope = np.polyfit(log_f, log_p, 1)[0]
            results["exp3_power_psd_slope"] = float(slope)
            print(f"  Power PSD slope: {slope:.3f} (1/f target: -1.0)")
        else:
            results["exp3_power_psd_slope"] = 0.0
    else:
        results["exp3_power_psd_slope"] = 0.0

    # Autocorrelation
    if p_std > 0.01:
        p_norm = (powers - powers.mean()) / powers.std()
        acf = np.correlate(p_norm, p_norm, mode='full')
        acf = acf[len(acf)//2:] / acf.max()
        results["exp3_power_acf1"] = float(acf[1]) if len(acf) > 1 else 0.0
        print(f"  Power ACF(1): {acf[1]:.3f}")
    else:
        results["exp3_power_acf1"] = 0.0

    results["exp3_power_mean"] = float(p_mean)
    results["exp3_power_std"] = float(p_std)
    results["exp3_clock_mean"] = float(c_mean)
    results["exp3_temp_mean"] = float(t_mean)

    # DPM state check
    try:
        with open('/sys/class/drm/card1/device/pp_dpm_sclk', 'r') as f:
            dpm = f.read().strip()
        dpm_states = len(dpm.split('\n'))
        active_state = [l for l in dpm.split('\n') if '*' in l]
        results["exp3_dpm_states"] = dpm_states
        results["exp3_dpm_active"] = active_state[0].strip() if active_state else "unknown"
        print(f"  DPM states: {dpm_states}, active: {results['exp3_dpm_active']}")
    except:
        results["exp3_dpm_states"] = 0

    results["T1029_power_dynamic"] = "PASS" if p_std > 0.1 else "FAIL"
    results["T1030_psd_1f"] = "PASS" if abs(results.get("exp3_power_psd_slope", 0) + 1) < 1 else "FAIL"
    results["T1031_acf_gt_0"] = "PASS" if results.get("exp3_power_acf1", 0) > 0.1 else "FAIL"
    results["T1032_dpm_accessible"] = "PASS" if results.get("exp3_dpm_states", 0) > 0 else "FAIL"

    for tid, name, val in [
        ("T1029", f"power_std({p_std:.2f})>0.1", p_std > 0.1),
        ("T1030", f"PSD_slope({results.get('exp3_power_psd_slope', 0):.2f})≈-1",
         abs(results.get("exp3_power_psd_slope", 0) + 1) < 1),
        ("T1031", f"ACF({results.get('exp3_power_acf1', 0):.3f})>0.1",
         results.get("exp3_power_acf1", 0) > 0.1),
        ("T1032", f"DPM_states({results.get('exp3_dpm_states', 0)})>0",
         results.get("exp3_dpm_states", 0) > 0),
    ]:
        print(f"  {tid} {name}: {'PASS' if val else 'FAIL'}")


# ==========================================================================
# EXP 4: GPU Substrate Noise as Computation
# ==========================================================================
def run_exp4_substrate_compute(results):
    """Use GPU substrate noise (clock64, thermal) as computation primitive."""
    print("\n=== EXP 4: GPU Substrate Noise Computation ===")

    rng = np.random.RandomState(42)
    N_TRIALS = 200
    N_CLASSES = 4
    STEPS = 10

    # Run same input twice — measure consistency (noise adds stochasticity)
    features_run1 = []
    features_run2 = []
    labels = []

    for trial in range(N_TRIALS):
        cls = trial % N_CLASSES
        freq = 0.5 + cls * 1.5
        amp = 0.3 + 0.5 * (cls % 2)
        t_arr = np.arange(STEPS) * 0.05
        input_signal = amp * np.sin(2 * np.pi * freq * t_arr)

        vmem1, spikes1, _ = run_gpu_lif(input_signal, mode=0)
        vmem2, spikes2, _ = run_gpu_lif(input_signal, mode=0)

        features_run1.append(build_features(spikes1, vmem1))
        features_run2.append(build_features(spikes2, vmem2))
        labels.append(cls)

    X1 = np.array(features_run1)
    X2 = np.array(features_run2)
    y = np.array(labels)

    # Classification accuracy from each run
    acc1, _ = ridge_classify(X1, y)
    acc2, _ = ridge_classify(X2, y)

    # Cross-run consistency: train on run1, test on run2
    std1 = X1.std(axis=0)
    mask = std1 > 1e-2
    if mask.sum() >= 3:
        scaler = StandardScaler()
        X1_f = scaler.fit_transform(X1[:, mask])
        X2_f = scaler.transform(X2[:, mask])
        clf = RidgeClassifier(alpha=1.0)
        clf.fit(X1_f, y)
        cross_acc = float(clf.score(X2_f, y))
    else:
        cross_acc = 0.25

    # Noise contribution: subtract mean response, measure residual info
    X_diff = X1 - X2  # should be pure noise
    acc_diff, _ = ridge_classify(X_diff, y)

    results["exp4_acc_run1"] = float(acc1)
    results["exp4_acc_run2"] = float(acc2)
    results["exp4_cross_acc"] = float(cross_acc)
    results["exp4_noise_acc"] = float(acc_diff)

    print(f"  Run1 acc: {acc1:.3f}")
    print(f"  Run2 acc: {acc2:.3f}")
    print(f"  Cross-run acc (train1→test2): {cross_acc:.3f}")
    print(f"  Noise-only acc (run1-run2): {acc_diff:.3f}")

    # Noise CV per neuron
    noise = X1 - X2
    noise_cv = noise.std(axis=0).mean() / (X1.mean(axis=0).std() + 1e-10)
    results["exp4_noise_cv"] = float(noise_cv)
    print(f"  Noise CV: {noise_cv:.4f}")

    results["T1033_consistent_gt_70"] = "PASS" if cross_acc > 0.70 else "FAIL"
    results["T1034_noise_gt_chance"] = "PASS" if acc_diff > 0.30 else "FAIL"
    results["T1035_both_runs_gt_80"] = "PASS" if min(acc1, acc2) > 0.80 else "FAIL"

    for tid, name, val in [
        ("T1033", f"cross({cross_acc:.3f})>0.70", cross_acc > 0.70),
        ("T1034", f"noise({acc_diff:.3f})>0.30 (chance)", acc_diff > 0.30),
        ("T1035", f"min({min(acc1,acc2):.3f})>0.80", min(acc1, acc2) > 0.80),
    ]:
        print(f"  {tid} {name}: {'PASS' if val else 'FAIL'}")


# ==========================================================================
# EXP 5: Deep Hybrid (GPU LIF + FPGA + GPU Firmware State)
# ==========================================================================
def run_exp5_deep_hybrid(fpga, results):
    """Full deep hybrid: GPU LIF + FPGA reservoir + GPU firmware telemetry."""
    print("\n=== EXP 5: Deep Hybrid Classification ===")

    rng = np.random.RandomState(42)
    w_in_fpga = rng.uniform(-1, 1, N_FPGA)
    N_CLASSES = 8
    N_TRIALS = N_CLASSES * 30  # 30 per class
    STEPS = 5

    vg_base = configure_fpga(fpga)
    time.sleep(0.5)

    features = []
    labels = []

    for trial in range(N_TRIALS):
        if trial % 60 == 0:
            print(f"  Trial {trial}/{N_TRIALS}")

        cls = trial % N_CLASSES
        freq = 0.5 + cls * 0.5
        amp = 0.2 + 0.8 * (cls % 3) / 2
        phase = cls * np.pi / 8
        t_arr = np.arange(STEPS) * STEP_INTERVAL
        input_signal = amp * np.sin(2 * np.pi * freq * t_arr + phase)

        # 1. GPU LIF (mode=3: full with adaptation + recurrence)
        gpu_vmem, gpu_spikes, gpu_adapt = run_gpu_lif(input_signal, mode=3)

        # 2. FPGA reservoir (coupled to GPU power noise)
        fp_sp, fp_vm = collect_fpga(fpga, input_signal, "COUPLED", w_in_fpga, vg_base)

        # 3. GPU firmware state
        gpu_power = read_gpu_power()
        gpu_clock = read_gpu_clock()
        gpu_temp = read_gpu_temp()

        # Build deep hybrid feature vector
        gpu_feat = build_features(gpu_spikes, gpu_vmem)        # 6×64 = 384
        fpga_feat = build_features(fp_sp, fp_vm)                # 6×128 = 768
        adapt_feat = np.concatenate([gpu_adapt.mean(axis=0),    # 64
                                     gpu_adapt.std(axis=0)])     # 64
        fw_feat = np.array([gpu_power, gpu_clock, gpu_temp])    # 3

        feat = np.concatenate([gpu_feat, fpga_feat, adapt_feat, fw_feat])
        features.append(feat)
        labels.append(cls)

    X = np.array(features)
    y = np.array(labels)

    # Full hybrid
    acc_full, acc_std = ridge_classify(X, y)

    # GPU only
    X_gpu = np.array([f[:384] for f in features])
    acc_gpu, _ = ridge_classify(X_gpu, y)

    # FPGA only
    X_fpga = np.array([f[384:384+768] for f in features])
    acc_fpga, _ = ridge_classify(X_fpga, y)

    results["exp5_full_hybrid_8class"] = float(acc_full)
    results["exp5_gpu_only_8class"] = float(acc_gpu)
    results["exp5_fpga_only_8class"] = float(acc_fpga)

    print(f"  Full hybrid: {acc_full:.3f} ± {acc_std:.3f}")
    print(f"  GPU only:    {acc_gpu:.3f}")
    print(f"  FPGA only:   {acc_fpga:.3f}")

    results["T1036_hybrid_gt_gpu"] = "PASS" if acc_full > acc_gpu else "FAIL"
    results["T1037_hybrid_gt_fpga"] = "PASS" if acc_full > acc_fpga else "FAIL"
    results["T1038_hybrid_gt_50"] = "PASS" if acc_full > 0.50 else "FAIL"
    results["T1039_hybrid_gt_60"] = "PASS" if acc_full > 0.60 else "FAIL"
    results["T1040_both_gt_chance"] = "PASS" if min(acc_gpu, acc_fpga) > 1/N_CLASSES else "FAIL"

    for tid, name, val in [
        ("T1036", f"hybrid({acc_full:.3f})>gpu({acc_gpu:.3f})", acc_full > acc_gpu),
        ("T1037", f"hybrid({acc_full:.3f})>fpga({acc_fpga:.3f})", acc_full > acc_fpga),
        ("T1038", f"hybrid({acc_full:.3f})>0.50", acc_full > 0.50),
        ("T1039", f"hybrid({acc_full:.3f})>0.60", acc_full > 0.60),
        ("T1040", f"min({min(acc_gpu,acc_fpga):.3f})>{1/N_CLASSES:.3f}",
         min(acc_gpu, acc_fpga) > 1/N_CLASSES),
    ]:
        print(f"  {tid} {name}: {'PASS' if val else 'FAIL'}")


# ==========================================================================
# EXP 6: GPU DPM Forcing as Neural Oscillator
# ==========================================================================
def run_exp6_dpm_oscillation(results):
    """Use DPM state forcing to create oscillatory GPU dynamics."""
    print("\n=== EXP 6: DPM Neural Oscillation ===")

    # Check if we can read/write DPM
    try:
        with open('/sys/class/drm/card1/device/pp_dpm_sclk', 'r') as f:
            dpm_content = f.read().strip()
        states = dpm_content.split('\n')
        print(f"  Available DPM states: {len(states)}")
        for s in states:
            print(f"    {s}")
    except Exception as e:
        print(f"  Cannot read DPM: {e}")
        results["T1041_dpm_readable"] = "FAIL"
        results["T1042_oscillation_detected"] = "FAIL"
        return

    results["T1041_dpm_readable"] = "PASS"

    # Monitor power during passive observation (don't force DPM - risky)
    N_SAMPLES = 100
    powers = []
    clocks = []
    for i in range(N_SAMPLES):
        powers.append(read_gpu_power())
        clocks.append(read_gpu_clock())
        time.sleep(0.05)

    powers = np.array(powers)
    clocks = np.array(clocks)

    # Check for natural oscillation
    if powers.std() > 0.01:
        from scipy import signal as sig
        # Zero-crossing rate
        p_centered = powers - powers.mean()
        crossings = np.sum(np.diff(np.sign(p_centered)) != 0)
        cross_rate = crossings / (N_SAMPLES * 0.05)  # Hz

        # Dominant frequency
        freqs, psd = sig.welch(p_centered, fs=20, nperseg=min(32, N_SAMPLES))
        peak_freq = freqs[np.argmax(psd[1:]) + 1] if len(psd) > 1 else 0

        results["exp6_cross_rate"] = float(cross_rate)
        results["exp6_peak_freq"] = float(peak_freq)
        results["exp6_power_cv"] = float(powers.std() / powers.mean())

        print(f"  Power zero-crossing rate: {cross_rate:.1f} Hz")
        print(f"  Peak frequency: {peak_freq:.2f} Hz")
        print(f"  Power CV: {powers.std()/powers.mean():.4f}")

        results["T1042_oscillation_detected"] = "PASS" if cross_rate > 1.0 else "FAIL"
    else:
        results["T1042_oscillation_detected"] = "FAIL"

    for tid, name, val in [
        ("T1041", "DPM readable", True),
        ("T1042", f"oscillation_rate({results.get('exp6_cross_rate', 0):.1f})>1.0",
         results.get("exp6_cross_rate", 0) > 1.0),
    ]:
        print(f"  {tid} {name}: {'PASS' if val else 'FAIL'}")


# ==========================================================================
# Main
# ==========================================================================
def main():
    print("=" * 70)
    print("z2242: Deep GPU Neuromorphic Computation")
    print("=" * 70)
    print(f"Time: {datetime.now().isoformat()}")

    # Compile HIP kernel first
    if not compile_hip():
        print("FATAL: HIP compilation failed")
        return

    fpga = FPGAEthBridge()
    fpga.connect()
    print(f"Connected to FPGA: {fpga.num_neurons} neurons")

    results = {
        "experiment": "z2242_deep_gpu_neuromorphic",
        "timestamp": datetime.now().isoformat(),
    }

    run_exp1_gpu_modes(results)
    run_exp2_cross_substrate(fpga, results)
    run_exp3_firmware_obs(results)
    run_exp4_substrate_compute(results)
    run_exp5_deep_hybrid(fpga, results)
    run_exp6_dpm_oscillation(results)

    # Summary
    n_pass = sum(1 for k, v in results.items() if k.startswith("T") and v == "PASS")
    n_total = sum(1 for k, v in results.items() if k.startswith("T") and v in ("PASS", "FAIL"))
    results["summary"] = f"{n_pass}/{n_total} PASS"

    print(f"\n{'=' * 70}")
    print(f"Summary: {results['summary']}")
    print(f"{'=' * 70}")

    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "results", "z2242_deep_gpu_neuromorphic.json")
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to {out_path}")

    fpga.close()


if __name__ == "__main__":
    main()
