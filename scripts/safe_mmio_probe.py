#!/usr/bin/env python3
"""
safe_mmio_probe.py — Sample GPU internal state via SAFE debugfs interfaces
while HIP kernels execute. NO /dev/mem writes, NO direct MMIO.

Sources:
  1. amdgpu_pm_info — SCLK, power, temp, load (text parse)
  2. amdgpu_fence_info — ring activity (fence counters)
  3. amdgpu_wave — active wavefront state (only during kernel exec)
  4. hwmon — power, temp, freq via sysfs
  5. SMN thermal via ryzen_smu pm_table
"""
import os
import sys
import time
import struct
import subprocess
import threading
import numpy as np
import json

DEBUGFS = '/sys/kernel/debug/dri/0'
RESULTS_DIR = '/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results'


def read_file(path):
    """Read file content safely"""
    try:
        with open(path, 'r') as f:
            return f.read()
    except:
        return ""


def parse_pm_info():
    """Parse amdgpu_pm_info for numeric values"""
    txt = read_file(f'{DEBUGFS}/amdgpu_pm_info')
    vals = {}
    for line in txt.split('\n'):
        line = line.strip()
        if 'MHz (SCLK)' in line:
            vals['sclk_mhz'] = int(line.split()[0])
        elif 'MHz (MCLK)' in line:
            vals['mclk_mhz'] = int(line.split()[0])
        elif 'W (average' in line:
            vals['power_avg_w'] = float(line.split()[0])
        elif 'W (current' in line:
            vals['power_cur_w'] = float(line.split()[0])
        elif 'GPU Temperature' in line:
            vals['temp_c'] = int(line.split(':')[1].strip().split()[0])
        elif 'GPU Load' in line:
            vals['load_pct'] = int(line.split(':')[1].strip().split()[0])
    return vals


def parse_fence_info():
    """Parse fence counters for ring activity"""
    txt = read_file(f'{DEBUGFS}/amdgpu_fence_info')
    fences = {}
    current_ring = None
    for line in txt.split('\n'):
        if '--- ring' in line:
            current_ring = line.split('(')[1].split(')')[0] if '(' in line else 'unknown'
        elif 'Last signaled fence' in line and current_ring:
            try:
                val = int(line.strip().split()[-1], 16)
                fences[f'{current_ring}_signaled'] = val
            except:
                pass
        elif 'Last emitted' in line and current_ring and 'trailing' not in line:
            try:
                val = int(line.strip().split()[-1], 16)
                fences[f'{current_ring}_emitted'] = val
            except:
                pass
    return fences


def read_hwmon():
    """Read hwmon values"""
    vals = {}
    try:
        # Find amdgpu hwmon
        for d in os.listdir('/sys/class/hwmon/'):
            name_path = f'/sys/class/hwmon/{d}/name'
            if os.path.exists(name_path):
                name = read_file(name_path).strip()
                if name == 'amdgpu':
                    base = f'/sys/class/hwmon/{d}'
                    for f in ['temp1_input', 'freq1_input', 'power1_average',
                              'in0_input', 'freq2_input']:
                        path = f'{base}/{f}'
                        if os.path.exists(path):
                            try:
                                vals[f] = int(read_file(path).strip())
                            except:
                                pass
                    break
    except:
        pass
    return vals


def read_pm_table_thermal():
    """Read hotspot thermal from ryzen_smu pm_table"""
    try:
        with open('/sys/kernel/ryzen_smu_drv/pm_table', 'rb') as f:
            data = f.read()
        temp = struct.unpack_from('<f', data, 0x004C)[0]
        return temp
    except:
        return None


def read_wave_info():
    """Read active wavefront info — only populated during kernel execution"""
    txt = read_file(f'{DEBUGFS}/amdgpu_wave')
    lines = [l for l in txt.strip().split('\n') if l.strip()]
    return len(lines)  # number of active waves


def sample_burst_safe(n_samples=500, delay_ms=2):
    """Sample all safe sources as fast as possible"""
    data = []
    t0 = time.perf_counter()

    for i in range(n_samples):
        t = time.perf_counter() - t0
        sample = {'t': t, 'i': i}

        # Fast sources (sysfs, no debugfs overhead)
        hwmon = read_hwmon()
        sample.update(hwmon)

        # PM table thermal (fast binary read)
        pm_temp = read_pm_table_thermal()
        if pm_temp is not None:
            sample['pm_hotspot_c'] = pm_temp

        # Fence counters (lightweight debugfs)
        if i % 5 == 0:  # every 5th sample to reduce overhead
            fences = parse_fence_info()
            sample.update(fences)

        # Wave count (lightweight)
        if i % 10 == 0:
            sample['wave_count'] = read_wave_info()

        # Full PM info (heavier, less frequent)
        if i % 20 == 0:
            pm = parse_pm_info()
            sample.update(pm)

        data.append(sample)
        if delay_ms > 0:
            time.sleep(delay_ms / 1000.0)

    return data


def launch_kernel(kernel_type, duration_s=5.0):
    """Launch a long-running HIP kernel"""
    codes = {
        'heavy_compute': '''
#include <hip/hip_runtime.h>
__global__ void k(float* o, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    float x = (float)i;
    for (int j = 0; j < 100000; j++)
        x = sinf(x) * cosf(x + 0.1f) + tanf(x * 0.01f);
    if (i < n) o[i] = x;
}
int main() {
    float *d; hipMalloc(&d, 65536*sizeof(float));
    for (int r = 0; r < REPS; r++)
        k<<<256, 256>>>(d, 65536);
    hipDeviceSynchronize();
    hipFree(d);
}
''',
        'atomic_storm': '''
#include <hip/hip_runtime.h>
__global__ void k(int* c, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int t = tid % n;
    for (int j = 0; j < 100000; j++) {
        atomicAdd(&c[t], 1);
        t = (t + 7) % n;
    }
}
int main() {
    int *d; hipMalloc(&d, 64*sizeof(int));
    hipMemset(d, 0, 64*sizeof(int));
    for (int r = 0; r < REPS; r++)
        k<<<64, 256>>>(d, 64);
    hipDeviceSynchronize();
    hipFree(d);
}
''',
        'memory_thrash': '''
#include <hip/hip_runtime.h>
__global__ void k(float* d, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    float sum = 0;
    for (int j = 0; j < 50000; j++) {
        int idx = (i * 127 + j * 31) % n;
        sum += d[idx];
    }
    d[i % n] = sum;
}
int main() {
    int n = 4*1024*1024;
    float *d; hipMalloc(&d, n*sizeof(float));
    hipMemset(d, 0, n*sizeof(float));
    for (int r = 0; r < REPS; r++)
        k<<<256, 256>>>(d, n);
    hipDeviceSynchronize();
    hipFree(d);
}
''',
    }

    if kernel_type not in codes:
        return None

    reps = max(1, int(duration_s * 10))
    code = codes[kernel_type].replace('REPS', str(reps))

    src = f'/tmp/safe_probe_{kernel_type}.hip'
    exe = f'/tmp/safe_probe_{kernel_type}'
    with open(src, 'w') as f:
        f.write(code)

    env = os.environ.copy()
    env['HSA_OVERRIDE_GFX_VERSION'] = '11.0.0'

    r = subprocess.run(['/opt/rocm/bin/hipcc', '-O2', '-o', exe, src],
                       env=env, capture_output=True, timeout=120)
    if r.returncode != 0:
        print(f"  Compile failed: {r.stderr.decode()[:200]}")
        return None

    proc = subprocess.Popen([exe], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return proc


def analyze_timeseries(data, key):
    """Analyze a single signal for reservoir potential"""
    vals = [d[key] for d in data if key in d]
    if len(vals) < 10:
        return None
    arr = np.array(vals, dtype=float)
    unique = len(np.unique(arr))
    transitions = np.sum(arr[1:] != arr[:-1])
    mean = np.mean(arr)
    std = np.std(arr)
    rng = np.max(arr) - np.min(arr)

    # ACF(1)
    if std > 0:
        d = arr - mean
        acf1 = np.corrcoef(d[:-1], d[1:])[0, 1] if len(d) > 1 else 1.0
    else:
        acf1 = 1.0

    return {
        'n': len(vals),
        'unique': unique,
        'transitions': int(transitions),
        'mean': float(mean),
        'std': float(std),
        'range': float(rng),
        'acf1': float(acf1),
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
    }


def main():
    print("=" * 60)
    print("SAFE MMIO PROBE — Debugfs + Hwmon + PM Table")
    print("=" * 60)

    # Thermal check
    try:
        t = int(open('/sys/class/thermal/thermal_zone0/temp').read()) / 1000
        print(f"CPU temp: {t:.0f}°C")
        if t > 75:
            print("Too hot, waiting...")
            time.sleep(30)
    except:
        pass

    conditions = ['idle', 'heavy_compute', 'atomic_storm', 'memory_thrash']
    all_results = {}

    for cond in conditions:
        print(f"\n{'='*50}")
        print(f"CONDITION: {cond}")
        print(f"{'='*50}")

        # Thermal check
        try:
            t = int(open('/sys/class/thermal/thermal_zone0/temp').read()) / 1000
            if t > 70:
                print(f"  Cooling ({t:.0f}°C)...")
                while t > 50:
                    time.sleep(3)
                    t = int(open('/sys/class/thermal/thermal_zone0/temp').read()) / 1000
        except:
            pass

        # Launch kernel
        proc = None
        if cond != 'idle':
            proc = launch_kernel(cond, duration_s=8.0)
            if proc:
                time.sleep(0.5)  # let kernel start
                print(f"  Kernel launched (pid={proc.pid})")

        # Sample
        n = 200
        print(f"  Sampling {n} readings...")
        data = sample_burst_safe(n_samples=n, delay_ms=5)

        # Wait for kernel
        if proc:
            try:
                proc.wait(timeout=60)
                print(f"  Kernel finished")
            except subprocess.TimeoutExpired:
                proc.kill()
                print(f"  Kernel killed (timeout)")

        # Analyze
        all_keys = set()
        for d in data:
            all_keys.update(d.keys())
        all_keys -= {'t', 'i'}

        cond_results = {}
        print()
        for key in sorted(all_keys):
            r = analyze_timeseries(data, key)
            if r and r['n'] > 5:
                cond_results[key] = r
                # Only print interesting signals
                if r['unique'] > 1:
                    print(f"  {key:30s}: unique={r['unique']:>4} "
                          f"range=[{r['min']:.1f}, {r['max']:.1f}] "
                          f"std={r['std']:.3f} acf1={r['acf1']:.3f}")

        all_results[cond] = cond_results

    # Cross-condition comparison
    print(f"\n{'='*60}")
    print("CROSS-CONDITION DYNAMIC SIGNALS")
    print(f"{'='*60}")

    # Find signals that CHANGE across conditions
    all_signals = set()
    for cond in conditions:
        all_signals.update(all_results[cond].keys())

    for sig in sorted(all_signals):
        vals_by_cond = {}
        for cond in conditions:
            if sig in all_results[cond]:
                r = all_results[cond][sig]
                vals_by_cond[cond] = r

        if len(vals_by_cond) < 2:
            continue

        # Check if signal differs across conditions
        means = [v['mean'] for v in vals_by_cond.values()]
        if max(means) - min(means) > 0.01 * (abs(max(means)) + 1e-10):
            print(f"\n  {sig}:")
            for cond, r in vals_by_cond.items():
                print(f"    {cond:20s}: mean={r['mean']:>10.2f} "
                      f"std={r['std']:>8.3f} unique={r['unique']:>4} "
                      f"acf1={r['acf1']:>6.3f}")

    # Save
    out = f'{RESULTS_DIR}/safe_mmio_probe.json'

    def convert(o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return o

    with open(out, 'w') as f:
        json.dump(all_results, f, indent=2, default=convert)
    print(f"\nSaved: {out}")

    # Also save raw timeseries for the most interesting condition
    print("\nDone.")


if __name__ == '__main__':
    main()
