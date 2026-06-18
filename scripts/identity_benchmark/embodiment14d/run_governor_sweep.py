"""Phase 14D — Re-measure Phase 12B Task A/B/E on ikaros at BOTH governors.

This is the LOCAL half of the matched-governor confound test. SSH to daedalus
is currently refused (port 22 closed), so we cannot run on daedalus from this
session. We therefore measure ikaros at both governors and compare each to the
existing daedalus@<its-default-perf> data from Phase 12B.

Saves per-governor JSONs to:
  results/IDENTITY_BENCHMARK_2026-05-30/embodiment14d/task_<X>_ikaros_<gov>.json

Strict thermal: abort 68, pause 63, cool 50. Performance governor heats more,
so we cool to 48C between blocks.
"""
import os, sys, time, json, struct, subprocess, ctypes

HERE = os.path.dirname(os.path.abspath(__file__))
P12B = os.path.abspath(os.path.join(HERE, '..', 'embodiment12b'))
sys.path.insert(0, P12B)
sys.path.insert(0, HERE)

from common12b import thermal_guard, wait_cool, save_json, hostname, compile_c, get_apu_temp_c
from governor_ctl import set_governor, current_governor, all_agree, list_governors

OUT_DIR = os.path.abspath(os.path.join(HERE, '..', '..', '..',
                                       'results', 'IDENTITY_BENCHMARK_2026-05-30',
                                       'embodiment14d'))
os.makedirs(OUT_DIR, exist_ok=True)
HOST = hostname()


def percentile(sorted_arr, p):
    if not sorted_arr: return 0.0
    k = (len(sorted_arr)-1)*p/100.0
    f = int(k); c = min(f+1, len(sorted_arr)-1)
    return sorted_arr[f] + (sorted_arr[c]-sorted_arr[f])*(k-f)


def summarize(arr):
    arr2 = sorted(arr); n = len(arr2)
    s = sum(arr2); m = s/n
    var = sum((x-m)**2 for x in arr2)/n
    return {'n': n, 'mean': m, 'std': var**0.5,
            'p50': percentile(arr2, 50), 'p90': percentile(arr2, 90),
            'p99': percentile(arr2, 99), 'p99_9': percentile(arr2, 99.9),
            'min': arr2[0], 'max': arr2[-1]}


def task_A(gov):
    """RDRAND + nanosleep — Phase 12 winners revisited."""
    print(f"\n=== TASK A (rdrand + nanosleep) gov={gov} ===", flush=True)
    out = {'host': HOST, 'gov': gov, 't_start': time.time()}
    thermal_guard(verbose=True)

    rdrand_bin = os.path.join(P12B, 'rdrand_cycles')
    proc = subprocess.run([rdrand_bin, '100000'], capture_output=True, check=True)
    cyc = list(struct.unpack(f'{len(proc.stdout)//4}I', proc.stdout))
    out['rdrand'] = summarize(cyc)
    print(f"  rdrand p50={out['rdrand']['p50']} p99.9={out['rdrand']['p99_9']}", flush=True)

    thermal_guard()
    libc = ctypes.CDLL('libc.so.6', use_errno=True)
    class TS(ctypes.Structure): _fields_=[("s",ctypes.c_long),("ns",ctypes.c_long)]
    ts = TS(0, 1000)
    ns = []
    for _ in range(50000):
        t0 = time.perf_counter_ns()
        libc.nanosleep(ctypes.byref(ts), None)
        ns.append(time.perf_counter_ns() - t0)
    out['nanosleep'] = summarize(ns)
    print(f"  nanosleep p50={out['nanosleep']['p50']} p99.9={out['nanosleep']['p99_9']}", flush=True)

    out['t_end'] = time.time()
    out['apu_temp_end_c'] = get_apu_temp_c()
    save_json(os.path.join(OUT_DIR, f'task_A_{HOST}_{gov}.json'), out)
    return out


def task_B(gov):
    """Inter-core TSC offset."""
    print(f"\n=== TASK B (inter-core TSC) gov={gov} ===", flush=True)
    binp = os.path.join(P12B, 'tsc_inter_core')
    out = {'host': HOST, 'gov': gov, 't_start': time.time(), 'pairs': {}}
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
    save_json(os.path.join(OUT_DIR, f'task_B_{HOST}_{gov}.json'), out)
    return out


def task_E(gov):
    """Cacheline ping-pong RTT."""
    print(f"\n=== TASK E (cacheline pingpong) gov={gov} ===", flush=True)
    binp = os.path.join(P12B, 'cacheline_pingpong')
    out = {'host': HOST, 'gov': gov, 't_start': time.time(), 'pairs': {}}
    pairs = [(0,1), (0,2), (0,4), (0,7), (0,8), (0,15), (0,16)]
    N = 20000
    for (a,b) in pairs:
        thermal_guard()
        proc = subprocess.run([binp, str(a), str(b), str(N)], capture_output=True, check=True)
        rtt = struct.unpack(f'{len(proc.stdout)//8}Q', proc.stdout)
        s = summarize(rtt)
        out['pairs'][f'{a}_{b}'] = s
        print(f"  pair({a},{b}): p50={s['p50']:.0f} cyc std={s['std']:.0f}", flush=True)
        time.sleep(0.3)
    out['t_end'] = time.time()
    out['apu_temp_end_c'] = get_apu_temp_c()
    save_json(os.path.join(OUT_DIR, f'task_E_{HOST}_{gov}.json'), out)
    return out


def run_one_governor(gov):
    print(f"\n############# GOVERNOR={gov} #############", flush=True)
    ok, raw = set_governor(gov)
    print(f"[gov set] target={gov} ok={ok}", flush=True)
    if not ok:
        print(f"[gov set] WARN: not all cores at {gov}: {raw[:400]}", flush=True)
    time.sleep(30)  # let frequencies settle
    print(f"[gov verify] current first-core: {current_governor()}", flush=True)
    print(f"[gov verify] all_agree({gov})={all_agree(gov)}", flush=True)
    print(f"[cool] target=56C timeout=90s  cur={get_apu_temp_c():.1f}C", flush=True)
    wait_cool(target_c=56, timeout_s=90)
    snapshot = {'gov_target': gov, 'first_core_now': current_governor(),
                'all_agree': all_agree(gov), 'temp_c': get_apu_temp_c(),
                't': time.time()}
    save_json(os.path.join(OUT_DIR, f'gov_snapshot_{HOST}_{gov}.json'), snapshot)

    task_A(gov); print(f"[cool] cur={get_apu_temp_c():.1f}C", flush=True); wait_cool(56, 90)
    task_B(gov); print(f"[cool] cur={get_apu_temp_c():.1f}C", flush=True); wait_cool(56, 90)
    task_E(gov); print(f"[cool] cur={get_apu_temp_c():.1f}C", flush=True); wait_cool(56, 90)


def main():
    print(f"=== Phase 14D governor sweep on {HOST} ===", flush=True)
    print(f"start temp: {get_apu_temp_c():.1f}C", flush=True)
    initial_gov = current_governor()
    print(f"initial governor: {initial_gov}", flush=True)
    save_json(os.path.join(OUT_DIR, f'initial_state_{HOST}.json'),
              {'initial_gov': initial_gov, 'all_cores': list_governors(),
               'temp_start_c': get_apu_temp_c()})

    # measure powersave first (matches original ikaros), then performance
    for gov in ['powersave', 'performance']:
        run_one_governor(gov)

    # restore to original
    print(f"\n[restore] setting governor back to {initial_gov}", flush=True)
    set_governor(initial_gov)
    print(f"[restore] verified: {current_governor()}", flush=True)
    save_json(os.path.join(OUT_DIR, f'final_state_{HOST}.json'),
              {'restored_gov': current_governor(), 'all_agree': all_agree(initial_gov),
               'temp_end_c': get_apu_temp_c()})


if __name__ == "__main__":
    main()
