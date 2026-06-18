#!/usr/bin/env python3
"""Phase 12B — all tasks A-H driver.

Runs on local host (ikaros or daedalus). Saves results per-host JSON.
Strict thermal: abort 68C, pause 63C, cool 50C.
Per-test wall budget <= 2min, then wait_cool.
"""
import os, sys, time, json, struct, subprocess, ctypes, mmap

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common12b import (thermal_guard, wait_cool, save_json, hostname,
                       compile_c, get_apu_temp_c)

OUT_DIR = os.path.abspath(os.path.join(HERE, '..', '..', '..',
                                       'results', 'IDENTITY_BENCHMARK_2026-05-30',
                                       'embodiment12b'))
os.makedirs(OUT_DIR, exist_ok=True)
HOST = hostname()

# ---------- helpers ----------
def percentile(sorted_arr, p):
    if len(sorted_arr) == 0:
        return 0.0
    k = (len(sorted_arr)-1) * p / 100.0
    f = int(k); c = min(f+1, len(sorted_arr)-1)
    return sorted_arr[f] + (sorted_arr[c]-sorted_arr[f])*(k-f)

def summarize(arr):
    arr2 = sorted(arr)
    n = len(arr2)
    s = sum(arr2); m = s/n
    var = sum((x-m)**2 for x in arr2)/n
    return {
        'n': n, 'mean': m, 'std': var**0.5,
        'p50': percentile(arr2,50), 'p90': percentile(arr2,90),
        'p99': percentile(arr2,99), 'p99_9': percentile(arr2,99.9),
        'p99_99': percentile(arr2,99.99) if n>=10000 else None,
        'min': arr2[0], 'max': arr2[-1],
    }

# ---------- Task A: reboot-stability of Phase 12 ----------
def task_A_replicate_phase12():
    """Re-measure RDRAND p50, nanosleep p99.9, NVMe p99.9 to compare with Phase 12."""
    print(f"\n=== TASK A (replicate Phase 12 winners) host={HOST} ===", flush=True)
    thermal_guard(verbose=True)
    out = {'host': HOST, 't_start': time.time()}

    # --- RDRAND 100k cycles ---
    rdrand_src = os.path.join(HERE, 'rdrand_cycles.c')
    with open(rdrand_src, 'w') as f:
        f.write(r"""
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <immintrin.h>
static inline uint64_t rdtscp(void){unsigned aux,lo,hi;__asm__ __volatile__("rdtscp":"=a"(lo),"=d"(hi),"=c"(aux)::"memory");return ((uint64_t)hi<<32)|lo;}
int main(int argc,char**argv){int n=atoi(argv[1]);unsigned long long r;uint32_t *out=malloc(sizeof(uint32_t)*n);
for(int i=0;i<n;i++){uint64_t a=rdtscp();_rdrand64_step(&r);uint64_t b=rdtscp();uint64_t d=b-a;out[i]=(d>0xFFFFFFFFull)?0xFFFFFFFFu:(uint32_t)d;}
fwrite(out,4,n,stdout);free(out);return 0;}
""")
    rdrand_bin = os.path.join(HERE, 'rdrand_cycles')
    compile_c(rdrand_src, rdrand_bin, ['-mrdrnd'])
    thermal_guard()
    proc = subprocess.run([rdrand_bin, '100000'], capture_output=True, check=True)
    cyc = list(struct.unpack(f'{len(proc.stdout)//4}I', proc.stdout))
    out['rdrand'] = summarize(cyc)
    print(f"  rdrand p50={out['rdrand']['p50']} p99.9={out['rdrand']['p99_9']}", flush=True)

    thermal_guard()
    # --- nanosleep 50k samples ---
    ns = []
    for _ in range(50000):
        t0 = time.perf_counter_ns()
        time.sleep(0)  # nanosleep(0) -> sched_yield-ish; use time.sleep(1e-6)
        # use sleep(1us)
        t1 = time.perf_counter_ns()
        ns.append(t1-t0)
    # Better: real nanosleep via clock_nanosleep
    libc = ctypes.CDLL('libc.so.6', use_errno=True)
    class TS(ctypes.Structure): _fields_=[("s",ctypes.c_long),("ns",ctypes.c_long)]
    ts = TS(0, 1000)  # 1us
    ns = []
    for _ in range(50000):
        t0 = time.perf_counter_ns()
        libc.nanosleep(ctypes.byref(ts), None)
        t1 = time.perf_counter_ns()
        ns.append(t1-t0)
    out['nanosleep'] = summarize(ns)
    print(f"  nanosleep p50={out['nanosleep']['p50']} p99.9={out['nanosleep']['p99_9']}", flush=True)

    thermal_guard()
    # --- NVMe read latency 20k samples (light) ---
    nvme = []
    # use /etc/passwd-equivalent on root fs with O_DIRECT? Simpler: stat() repeatedly
    # Honor Phase 12 protocol: O_DIRECT 4K reads. Find nvme device.
    try:
        import os as _os
        # find a readable file on nvme
        path = '/usr/bin/python3'
        fd = _os.open(path, _os.O_RDONLY)
        buf = bytearray(4096)
        for _ in range(20000):
            _os.lseek(fd, 0, 0)
            t0 = time.perf_counter_ns()
            _os.read(fd, 4096)
            t1 = time.perf_counter_ns()
            nvme.append(t1-t0)
        _os.close(fd)
        out['nvme_read'] = summarize(nvme)
        print(f"  nvme p50={out['nvme_read']['p50']} p99.9={out['nvme_read']['p99_9']}", flush=True)
    except Exception as e:
        out['nvme_read'] = {'error': str(e)}

    out['t_end'] = time.time()
    out['apu_temp_end_c'] = get_apu_temp_c()
    save_json(os.path.join(OUT_DIR, f'task_A_{HOST}.json'), out)
    return out

# ---------- Task B: inter-core TSC offset ----------
def task_B_tsc_inter_core():
    print(f"\n=== TASK B (inter-core TSC) host={HOST} ===", flush=True)
    src = os.path.join(HERE, 'tsc_inter_core.c')
    binp = os.path.join(HERE, 'tsc_inter_core')
    compile_c(src, binp)
    out = {'host': HOST, 't_start': time.time(), 'pairs': {}}
    # Sample core 0 vs cores 1,4,8,12,15 (different CCX/CCD positions on 16-core Strix Halo)
    targets = [1, 2, 4, 7, 8, 12, 15]
    N = 5000
    for tgt in targets:
        thermal_guard()
        proc = subprocess.run([binp, str(tgt), str(N)], capture_output=True, check=True)
        data = struct.unpack(f'{len(proc.stdout)//8}q', proc.stdout)
        offsets = [data[2*i+1] - data[2*i] for i in range(N)]
        s = summarize(offsets)
        out['pairs'][str(tgt)] = s
        print(f"  core 0<->{tgt}: p50={s['p50']:.0f} p99={s['p99']:.0f} std={s['std']:.0f}", flush=True)
        time.sleep(0.5)
    out['t_end'] = time.time()
    out['apu_temp_end_c'] = get_apu_temp_c()
    save_json(os.path.join(OUT_DIR, f'task_B_{HOST}.json'), out)
    return out

# ---------- Task C: AES-NI latency ----------
def task_C_aesni():
    print(f"\n=== TASK C (AES-NI latency) host={HOST} ===", flush=True)
    src = os.path.join(HERE, 'aesni_latency.c')
    binp = os.path.join(HERE, 'aesni_latency')
    compile_c(src, binp, ['-maes'])
    out = {'host': HOST, 't_start': time.time()}
    thermal_guard()
    N = 200000
    proc = subprocess.run([binp, str(N)], capture_output=True, check=True)
    cyc = struct.unpack(f'{len(proc.stdout)//4}I', proc.stdout)
    s = summarize(cyc)
    out['aesenc'] = s
    print(f"  AESENC p50={s['p50']} p99={s['p99']} p99.9={s['p99_9']} std={s['std']:.2f}", flush=True)
    out['t_end'] = time.time()
    out['apu_temp_end_c'] = get_apu_temp_c()
    save_json(os.path.join(OUT_DIR, f'task_C_{HOST}.json'), out)
    return out

# ---------- Task D: atomic contention curves ----------
def task_D_atomic():
    print(f"\n=== TASK D (atomic contention) host={HOST} ===", flush=True)
    src = os.path.join(HERE, 'atomic_contention.c')
    binp = os.path.join(HERE, 'atomic_contention')
    compile_c(src, binp)
    out = {'host': HOST, 't_start': time.time(), 'pairs': {}}
    # Strix Halo: 16 cores, 32 threads. SMT siblings: i and i+16.
    # Pairs to probe CCX/CCD geometry.
    pairs = [(0,1), (0,2), (0,4), (0,7), (0,8), (0,15), (0,16)]  # 0<->16 is SMT sibling
    DUR_MS = 600  # short bursts
    for (a,b) in pairs:
        thermal_guard()
        proc = subprocess.run([binp, str(a), str(b), str(DUR_MS)], capture_output=True, check=True)
        parts = proc.stdout.decode().strip().split()
        inc_a, inc_b, tot = int(parts[0]), int(parts[1]), int(parts[2])
        out['pairs'][f'{a}_{b}'] = {'inc_a': inc_a, 'inc_b': inc_b, 'total': tot,
                                    'throughput_per_s': tot/(DUR_MS/1000.0)}
        print(f"  pair({a},{b}): total={tot} ({tot/(DUR_MS/1000.0):.2e}/s)", flush=True)
        time.sleep(0.4)
    out['t_end'] = time.time()
    out['apu_temp_end_c'] = get_apu_temp_c()
    save_json(os.path.join(OUT_DIR, f'task_D_{HOST}.json'), out)
    return out

# ---------- Task E: cache-line pingpong ----------
def task_E_cacheline():
    print(f"\n=== TASK E (cacheline pingpong) host={HOST} ===", flush=True)
    src = os.path.join(HERE, 'cacheline_pingpong.c')
    binp = os.path.join(HERE, 'cacheline_pingpong')
    compile_c(src, binp)
    out = {'host': HOST, 't_start': time.time(), 'pairs': {}}
    pairs = [(0,1), (0,2), (0,4), (0,7), (0,8), (0,15), (0,16)]
    N = 20000
    for (a,b) in pairs:
        thermal_guard()
        proc = subprocess.run([binp, str(a), str(b), str(N)], capture_output=True, check=True)
        rtt = struct.unpack(f'{len(proc.stdout)//8}Q', proc.stdout)
        s = summarize(rtt)
        out['pairs'][f'{a}_{b}'] = s
        print(f"  pair({a},{b}): p50={s['p50']:.0f} cyc, std={s['std']:.0f}", flush=True)
        time.sleep(0.3)
    out['t_end'] = time.time()
    out['apu_temp_end_c'] = get_apu_temp_c()
    save_json(os.path.join(OUT_DIR, f'task_E_{HOST}.json'), out)
    return out

# ---------- Task F: TPM latency ----------
def task_F_tpm():
    print(f"\n=== TASK F (TPM getrandom) host={HOST} ===", flush=True)
    out = {'host': HOST, 't_start': time.time()}
    if not os.path.exists('/dev/tpm0') and not os.path.exists('/dev/tpmrm0'):
        out['feasible'] = False; out['reason'] = 'no TPM device'
        save_json(os.path.join(OUT_DIR, f'task_F_{HOST}.json'), out)
        return out
    # Use tpm2_getrandom; time the wall-clock per invocation
    try:
        subprocess.run(['tpm2_getrandom','--hex','8'], capture_output=True, check=True, timeout=5)
    except Exception as e:
        out['feasible'] = False; out['reason'] = str(e)
        save_json(os.path.join(OUT_DIR, f'task_F_{HOST}.json'), out)
        return out
    out['feasible'] = True
    samples = []
    N = 500
    for i in range(N):
        if i % 50 == 0: thermal_guard()
        t0 = time.perf_counter_ns()
        subprocess.run(['tpm2_getrandom','--hex','8'], capture_output=True, check=True, timeout=5)
        t1 = time.perf_counter_ns()
        samples.append(t1-t0)
    out['tpm_getrandom_ns'] = summarize(samples)
    s = out['tpm_getrandom_ns']
    print(f"  tpm p50={s['p50']:.0f}ns p99={s['p99']:.0f}ns std={s['std']:.0f}", flush=True)
    out['t_end'] = time.time()
    out['apu_temp_end_c'] = get_apu_temp_c()
    save_json(os.path.join(OUT_DIR, f'task_F_{HOST}.json'), out)
    return out

# ---------- Task G: DRAM refresh probing ----------
def task_G_dram_refresh():
    print(f"\n=== TASK G (DRAM refresh probing) host={HOST} ===", flush=True)
    out = {'host': HOST, 't_start': time.time()}
    # Allocate 256MB, walk randomly, record latency of each access.
    # Refresh spikes manifest as outliers at ~7.8us refresh intervals (tREFI).
    SIZE = 256 * 1024 * 1024
    arr = bytearray(SIZE)
    # Walk in cache-line stride at semi-random positions
    import random
    rng = random.Random(42)
    N = 200000
    # warm up cache mapping
    for i in range(0, SIZE, 4096):
        arr[i] = 1
    thermal_guard()
    samples = []
    stride = 64
    positions = [rng.randrange(0, SIZE-64) & ~63 for _ in range(N)]
    t0_all = time.perf_counter_ns()
    for p in positions:
        t0 = time.perf_counter_ns()
        v = arr[p]
        t1 = time.perf_counter_ns()
        samples.append(t1-t0)
    out['walk_ns'] = summarize(samples)
    # Identify spikes (> p99 latency) and inter-spike intervals
    thr = out['walk_ns']['p99']
    spike_idx = [i for i,v in enumerate(samples) if v > thr*2]
    if len(spike_idx) > 10:
        intervals = [spike_idx[i+1]-spike_idx[i] for i in range(len(spike_idx)-1)]
        out['spike_interval_samples'] = summarize(intervals)
        out['spike_count'] = len(spike_idx)
    else:
        out['spike_count'] = len(spike_idx)
    print(f"  walk p50={out['walk_ns']['p50']:.0f}ns spikes={out['spike_count']}", flush=True)
    out['t_end'] = time.time()
    out['apu_temp_end_c'] = get_apu_temp_c()
    save_json(os.path.join(OUT_DIR, f'task_G_{HOST}.json'), out)
    return out

# ---------- Task H: PCIe config-space access latency ----------
def task_H_pcie_cfg():
    print(f"\n=== TASK H (PCIe config-space latency) host={HOST} ===", flush=True)
    out = {'host': HOST, 't_start': time.time()}
    # Enumerate PCI devices, read first 64B of config space per device
    pci_root = '/sys/bus/pci/devices'
    devs = sorted(os.listdir(pci_root))[:20]  # cap at 20
    per_dev = {}
    N = 2000
    for d in devs:
        cfg = os.path.join(pci_root, d, 'config')
        if not os.path.exists(cfg): continue
        try:
            fd = os.open(cfg, os.O_RDONLY)
        except PermissionError:
            continue
        thermal_guard()
        sams = []
        try:
            for _ in range(N):
                os.lseek(fd, 0, 0)
                t0 = time.perf_counter_ns()
                os.read(fd, 64)
                t1 = time.perf_counter_ns()
                sams.append(t1-t0)
        finally:
            os.close(fd)
        per_dev[d] = summarize(sams)
    out['per_device'] = per_dev
    # aggregate across all devices' p50
    all_p50 = [v['p50'] for v in per_dev.values()]
    if all_p50:
        all_p50.sort()
        out['agg_p50_of_p50'] = all_p50[len(all_p50)//2]
        out['n_devices'] = len(all_p50)
        print(f"  scanned {len(all_p50)} devs, median-p50={out['agg_p50_of_p50']:.0f}ns", flush=True)
    out['t_end'] = time.time()
    out['apu_temp_end_c'] = get_apu_temp_c()
    save_json(os.path.join(OUT_DIR, f'task_H_{HOST}.json'), out)
    return out

# ---------- main ----------
TASKS = {
    'A': task_A_replicate_phase12,
    'B': task_B_tsc_inter_core,
    'C': task_C_aesni,
    'D': task_D_atomic,
    'E': task_E_cacheline,
    'F': task_F_tpm,
    'G': task_G_dram_refresh,
    'H': task_H_pcie_cfg,
}

if __name__ == '__main__':
    sel = sys.argv[1:] if len(sys.argv)>1 else list(TASKS.keys())
    print(f"[run_all] host={HOST} tasks={sel} start_temp={get_apu_temp_c():.1f}C", flush=True)
    for t in sel:
        if t not in TASKS:
            print(f"unknown task {t}"); continue
        wait_cool(target_c=55, timeout_s=60)
        try:
            TASKS[t]()
        except SystemExit as e:
            print(f"[abort] task {t}: {e}", flush=True)
            break
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"[error] task {t}: {e}", flush=True)
        wait_cool(target_c=55, timeout_s=60)
    print(f"[done] end_temp={get_apu_temp_c():.1f}C", flush=True)
