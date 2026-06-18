#!/usr/bin/env python3
"""Phase 19 orchestrator: 10 reps of each signal with strict cool-down.

Order chosen for thermal smoothness:
   s5 (idle, 20s)  -> s6 (idle, 11s) -> s7 (light, 0.5s) -> s4 (idle, 11s)
   -> s1 (burst, ~10ms) -> s2 (burst, ~40ms) -> s3 (4 pairs, ~few s)
   -> s9 (medium load, ~20s)
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common19 import get_apu_temp_c, wait_cool, hostname, save_json

ORDER = ['s7_rapl_precision', 's5_pcie_aer', 's6_thermal_spread',
         's4_gpu_clock_jitter', 's1_btb_warmup', 's2_tlb_miss',
         's3_ccx_wakeup', 's9_jacobian_dynamics']

def main(reps=10):
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),
        '..', '..', '..', 'results', 'IDENTITY_BENCHMARK_2026-05-30', 'embodiment19'))
    os.makedirs(out_dir, exist_ok=True)
    host = hostname()
    summary = {'host': host, 'reps': reps, 't_start': time.time(),
               'signals': {}, 'temp_start_c': get_apu_temp_c()}
    print(f"[run_all] host={host} reps={reps} temp_start={summary['temp_start_c']:.1f}C", flush=True)
    for name in ORDER:
        print(f"\n=== {name} ===", flush=True)
        wait_cool(target_c=60, timeout_s=60)
        t0 = time.time()
        try:
            mod = __import__(name)
            outpath = mod.run(reps=reps, out_dir=out_dir)
            summary['signals'][name] = {'status': 'ok', 'seconds': time.time()-t0,
                                        'path': outpath,
                                        'temp_end_c': get_apu_temp_c()}
        except SystemExit as e:
            summary['signals'][name] = {'status': f'thermal_abort:{e}',
                                        'seconds': time.time()-t0}
        except Exception as e:
            summary['signals'][name] = {'status': f'error:{e}',
                                        'seconds': time.time()-t0}
        save_json(os.path.join(out_dir, f'{host}_run_all.json'), summary)
    summary['t_end'] = time.time()
    summary['temp_end_c'] = get_apu_temp_c()
    save_json(os.path.join(out_dir, f'{host}_run_all.json'), summary)
    print(f"\n[run_all] DONE in {summary['t_end']-summary['t_start']:.0f}s", flush=True)

if __name__ == '__main__':
    main(reps=int(sys.argv[1]) if len(sys.argv) > 1 else 10)
