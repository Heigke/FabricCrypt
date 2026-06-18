#!/usr/bin/env python3
"""Phase 22 orchestrator: run s20..s27 — light read-only signals.

Order chosen for thermal smoothness; all are sysfs/subprocess reads,
none should heat the APU appreciably. s27 (60s sleep-paced) last so
the previous bursts cool out.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common22 import get_apu_temp_c, wait_cool, hostname, save_json

ORDER = ['s20_acpi_pci_topology',
         's21_pcie_link_degradation',
         's22_usb_descriptor',
         's25_ucsi_power',
         's26_umr_safe_reads',
         's23_dmi_smbios',
         's24_boot_timing',
         's27_hpet_rtc_drift']


def main(reps=10):
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),
        '..', '..', '..', 'results', 'IDENTITY_BENCHMARK_2026-05-30',
        'embodiment22'))
    os.makedirs(out_dir, exist_ok=True)
    host = hostname()
    summary = {'host': host, 'reps': reps, 't_start': time.time(),
               'signals': {}, 'temp_start_c': get_apu_temp_c(),
               'phase': 22}
    print(f"[run_all] host={host} reps={reps} "
          f"temp={summary['temp_start_c']:.1f}C", flush=True)
    for name in ORDER:
        print(f"\n=== {name} ===", flush=True)
        wait_cool(target_c=60, timeout_s=120)
        t0 = time.time()
        try:
            mod = __import__(name)
            outpath = mod.run(reps=reps, out_dir=out_dir)
            summary['signals'][name] = {
                'status': 'ok',
                'seconds': time.time()-t0,
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
    print(f"\n[run_all] DONE in {summary['t_end']-summary['t_start']:.0f}s "
          f"temp={summary['temp_end_c']:.1f}C", flush=True)


if __name__ == '__main__':
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    main(reps=reps)
