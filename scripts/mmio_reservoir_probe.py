#!/usr/bin/env python3
"""
mmio_reservoir_probe.py — Sample GPU internal state registers via /dev/mem
WHILE HIP kernels execute. This is Layer 2 access: real-time observation
of GPU pipeline dynamics at ~100MHz from CPU side.

Registers sampled:
  GRBM_STATUS  (0x1410) — per-block busy flags (GFX, MEC, SQ, TA, TCC...)
  GRBM_STATUS2 (0x1408) — more block status bits
  CP_STAT      (0x3900) — Command Processor state
  SDMA0_STATUS (0x3614) — System DMA engine status

All via BAR5 MMIO at physical 0xB4400000.

Test plan:
  1. Sample registers with GPU IDLE — baseline
  2. Launch lightweight HIP kernel, sample during execution
  3. Launch heavy compute kernel, sample during execution
  4. Launch memory-bound kernel, sample during execution
  5. Analyze: entropy, variability, unique states, temporal dynamics
"""

import mmap
import struct
import os
import sys
import time
import subprocess
import threading
import numpy as np
import json

BAR5_PHYS = 0xB4400000
BAR5_SIZE = 0x100000  # 1MB

# Register indices (multiply by 4 for byte offset)
# These are the BYTE offsets we'll use directly
REGS = {
    'GRBM_STATUS':  0x0504 * 4,  # reg index 0x0504
    'GRBM_STATUS2': 0x0502 * 4,
    'GRBM_STATUS3': 0x0503 * 4,
    'CP_STAT':      0x0E40 * 4,
    'CP_CPC_STATUS': 0x0E54 * 4,
    'CP_CPF_STATUS': 0x0E53 * 4,
    'SDMA0_STATUS': 0x0D85 * 4,
    'RLC_STAT':     0x4C04 * 4,
}


def open_mmio():
    """Open /dev/mem and mmap BAR5"""
    fd = os.open('/dev/mem', os.O_RDONLY | os.O_SYNC)
    mm = mmap.mmap(fd, BAR5_SIZE, mmap.MAP_SHARED, mmap.PROT_READ,
                   offset=BAR5_PHYS)
    return fd, mm


def read_reg(mm, byte_offset):
    """Read 32-bit register at byte offset"""
    if byte_offset + 4 > BAR5_SIZE:
        return 0
    mm.seek(byte_offset)
    return struct.unpack('<I', mm.read(4))[0]


def sample_burst(mm, n_samples=10000, reg_offsets=None):
    """Sample registers as fast as possible, return arrays"""
    if reg_offsets is None:
        reg_offsets = [REGS['GRBM_STATUS'], REGS['CP_STAT'], REGS['SDMA0_STATUS']]

    n_regs = len(reg_offsets)
    data = np.zeros((n_samples, n_regs), dtype=np.uint32)
    timestamps = np.zeros(n_samples, dtype=np.float64)

    t0 = time.perf_counter()
    for i in range(n_samples):
        timestamps[i] = time.perf_counter() - t0
        for j, off in enumerate(reg_offsets):
            mm.seek(off)
            data[i, j] = struct.unpack('<I', mm.read(4))[0]

    return data, timestamps


def analyze_signal(data, timestamps, name="signal"):
    """Analyze a register time series for reservoir computing potential"""
    n = len(data)
    if n < 10:
        return {}

    # Basic stats
    unique = len(np.unique(data))
    transitions = np.sum(data[1:] != data[:-1])
    transition_rate = transitions / n

    # Bit-level analysis
    bits_active = 0
    for bit in range(32):
        bit_vals = (data >> bit) & 1
        if np.any(bit_vals == 1) and np.any(bit_vals == 0):
            bits_active += 1

    # Entropy (of unique values)
    vals, counts = np.unique(data, return_counts=True)
    probs = counts / n
    entropy = -np.sum(probs * np.log2(probs + 1e-30))

    # Autocorrelation at lag 1
    if np.std(data.astype(float)) > 0:
        d = data.astype(float)
        d = d - d.mean()
        acf1 = np.corrcoef(d[:-1], d[1:])[0, 1] if len(d) > 1 else 0
    else:
        acf1 = 1.0

    # Sampling rate
    dt = timestamps[-1] - timestamps[0]
    rate = n / dt if dt > 0 else 0

    result = {
        'name': name,
        'n_samples': n,
        'unique_values': unique,
        'transitions': int(transitions),
        'transition_rate': float(transition_rate),
        'bits_active': bits_active,
        'entropy_bits': float(entropy),
        'acf_lag1': float(acf1),
        'sampling_rate_hz': float(rate),
        'duration_s': float(dt),
        'min': int(np.min(data)),
        'max': int(np.max(data)),
        'example_values': [int(x) for x in data[:10]],
    }
    return result


def launch_hip_kernel(kernel_type, duration_s=2.0):
    """Launch a HIP kernel in background, return process"""
    # Write a tiny HIP program on the fly
    hip_code = {
        'idle': '',  # no kernel
        'light': '''
#include <hip/hip_runtime.h>
__global__ void light_kernel(float* out, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) out[i] = sinf((float)i * 0.001f);
}
int main() {
    float *d; hipMalloc(&d, 1024*sizeof(float));
    for (int rep = 0; rep < REPS; rep++)
        light_kernel<<<4, 256>>>(d, 1024);
    hipDeviceSynchronize();
    hipFree(d);
}
''',
        'heavy_compute': '''
#include <hip/hip_runtime.h>
__global__ void heavy_kernel(float* out, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    float x = (float)i;
    for (int j = 0; j < 10000; j++) {
        x = sinf(x) * cosf(x + 0.1f) + tanf(x * 0.01f);
    }
    if (i < n) out[i] = x;
}
int main() {
    float *d; hipMalloc(&d, 65536*sizeof(float));
    for (int rep = 0; rep < REPS; rep++)
        heavy_kernel<<<256, 256>>>(d, 65536);
    hipDeviceSynchronize();
    hipFree(d);
}
''',
        'heavy_memory': '''
#include <hip/hip_runtime.h>
__global__ void mem_kernel(float* data, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = 127; // prime stride for cache thrashing
    float sum = 0;
    for (int j = 0; j < 1000; j++) {
        int idx = (i * stride + j * 31) % n;
        sum += data[idx];
    }
    data[i % n] = sum;
}
int main() {
    int n = 4*1024*1024; // 16MB
    float *d; hipMalloc(&d, n*sizeof(float));
    hipMemset(d, 0, n*sizeof(float));
    for (int rep = 0; rep < REPS; rep++)
        mem_kernel<<<1024, 256>>>(d, n);
    hipDeviceSynchronize();
    hipFree(d);
}
''',
        'atomic_contention': '''
#include <hip/hip_runtime.h>
__global__ void atomic_kernel(int* counters, int n_counters) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int target = tid % n_counters;
    for (int j = 0; j < 10000; j++) {
        atomicAdd(&counters[target], 1);
        target = (target + 7) % n_counters;
    }
}
int main() {
    int *d; hipMalloc(&d, 64*sizeof(int));
    hipMemset(d, 0, 64*sizeof(int));
    for (int rep = 0; rep < REPS; rep++)
        atomic_kernel<<<64, 256>>>(d, 64);
    hipDeviceSynchronize();
    hipFree(d);
}
''',
    }

    if kernel_type == 'idle':
        return None

    code = hip_code[kernel_type]
    reps = max(1, int(duration_s * 50))  # rough scaling
    code = code.replace('REPS', str(reps))

    src = f'/tmp/mmio_test_{kernel_type}.hip'
    exe = f'/tmp/mmio_test_{kernel_type}'
    with open(src, 'w') as f:
        f.write(code)

    # Compile
    env = os.environ.copy()
    env['HSA_OVERRIDE_GFX_VERSION'] = '11.0.0'
    r = subprocess.run(['/opt/rocm/bin/hipcc', '-O2', '-o', exe, src],
                       env=env, capture_output=True, timeout=120)
    if r.returncode != 0:
        print(f"  Compile failed for {kernel_type}: {r.stderr.decode()[:200]}")
        return None

    # Launch in background
    proc = subprocess.Popen([exe], env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return proc


def main():
    print("=" * 70)
    print("MMIO RESERVOIR PROBE — GPU Internal State as Neuromorphic Signal")
    print("=" * 70)

    # Check thermal
    try:
        t = int(open('/sys/class/thermal/thermal_zone0/temp').read()) / 1000
        print(f"CPU temp: {t:.0f}°C")
        if t > 80:
            print("TOO HOT — waiting...")
            time.sleep(30)
    except:
        pass

    fd, mm = open_mmio()

    # Quick sanity check
    v = read_reg(mm, REGS['GRBM_STATUS'])
    print(f"GRBM_STATUS sanity: 0x{v:08x}")
    if v == 0 or v == 0xFFFFFFFF:
        print("WARNING: GRBM_STATUS looks wrong. GPU may not be active.")

    reg_offsets = [REGS['GRBM_STATUS'], REGS['CP_STAT'],
                   REGS['GRBM_STATUS2'], REGS['SDMA0_STATUS']]
    reg_names = ['GRBM_STATUS', 'CP_STAT', 'GRBM_STATUS2', 'SDMA0_STATUS']

    conditions = ['idle', 'light', 'heavy_compute', 'heavy_memory', 'atomic_contention']
    all_results = {}

    for cond in conditions:
        print(f"\n{'='*60}")
        print(f"CONDITION: {cond}")
        print(f"{'='*60}")

        # Check thermal
        try:
            t = int(open('/sys/class/thermal/thermal_zone0/temp').read()) / 1000
            if t > 75:
                print(f"  Cooling... ({t:.0f}°C)")
                while t > 55:
                    time.sleep(2)
                    t = int(open('/sys/class/thermal/thermal_zone0/temp').read()) / 1000
        except:
            pass

        # Launch kernel
        proc = launch_hip_kernel(cond, duration_s=3.0)
        if proc is not None:
            time.sleep(0.3)  # let kernel start

        # Sample registers
        n = 50000
        print(f"  Sampling {n} readings across {len(reg_offsets)} registers...")
        data, timestamps = sample_burst(mm, n_samples=n, reg_offsets=reg_offsets)

        # Wait for kernel to finish
        if proc is not None:
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()

        # Analyze each register
        cond_results = {}
        for j, rname in enumerate(reg_names):
            r = analyze_signal(data[:, j], timestamps, name=f"{rname}_{cond}")
            cond_results[rname] = r
            print(f"\n  {rname}:")
            print(f"    Unique values: {r['unique_values']}")
            print(f"    Transitions:   {r['transitions']}/{n} ({r['transition_rate']*100:.1f}%)")
            print(f"    Active bits:   {r['bits_active']}/32")
            print(f"    Entropy:       {r['entropy_bits']:.2f} bits")
            print(f"    ACF(1):        {r['acf_lag1']:.4f}")
            print(f"    Sample rate:   {r['sampling_rate_hz']/1000:.1f} kHz")
            print(f"    Range:         0x{r['min']:08x} — 0x{r['max']:08x}")
            # Show first few values
            vals = r['example_values'][:5]
            print(f"    First 5:       {' '.join(f'0x{v:08x}' for v in vals)}")

        all_results[cond] = cond_results

    # Cross-condition comparison
    print(f"\n{'='*60}")
    print("CROSS-CONDITION COMPARISON")
    print(f"{'='*60}")
    print(f"\n{'Register':<18} {'Condition':<20} {'Unique':>8} {'Trans%':>8} "
          f"{'Entropy':>8} {'ACF(1)':>8} {'Bits':>5}")
    print("-" * 75)

    for rname in reg_names:
        for cond in conditions:
            r = all_results[cond][rname]
            print(f"{rname:<18} {cond:<20} {r['unique_values']:>8} "
                  f"{r['transition_rate']*100:>7.1f}% "
                  f"{r['entropy_bits']:>7.2f} {r['acf_lag1']:>8.4f} "
                  f"{r['bits_active']:>5}")

    # Reservoir quality assessment
    print(f"\n{'='*60}")
    print("RESERVOIR QUALITY ASSESSMENT")
    print(f"{'='*60}")

    for rname in reg_names:
        idle = all_results['idle'][rname]
        best_cond = None
        best_entropy = 0
        for cond in conditions[1:]:
            r = all_results[cond][rname]
            # Reservoir quality = entropy * transition_rate * (1 - |acf|)
            # High entropy + many transitions + not just constant = good reservoir
            quality = r['entropy_bits'] * r['transition_rate'] * (1 - abs(r['acf_lag1']))
            if quality > best_entropy:
                best_entropy = quality
                best_cond = cond

        idle_ent = idle['entropy_bits']
        if best_cond:
            active = all_results[best_cond][rname]
            delta_ent = active['entropy_bits'] - idle_ent
            print(f"  {rname}: best under '{best_cond}' "
                  f"(entropy +{delta_ent:.2f} vs idle, "
                  f"quality={best_entropy:.4f})")
        else:
            print(f"  {rname}: no useful signal")

    # Save results
    out_path = '/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/mmio_reservoir_probe.json'

    # Convert numpy types for JSON
    def convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return obj

    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=convert)
    print(f"\nResults saved to {out_path}")

    mm.close()
    os.close(fd)
    print("\nDone.")


if __name__ == '__main__':
    main()
