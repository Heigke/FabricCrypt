"""Phase 14D — orchestrate: for each governor, collect sigs + run T2/T3.

Run AFTER run_governor_sweep.py has completed.
Restores the governor at the end.
"""
import os, sys, time, subprocess

VENV_PY = "/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/venv/bin/python"
if not os.path.exists(VENV_PY):
    VENV_PY = VENV_PY

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
P12B = os.path.abspath(os.path.join(HERE, '..', 'embodiment12b'))
sys.path.insert(0, P12B)

from governor_ctl import set_governor, current_governor
from common12b import wait_cool, get_apu_temp_c

OUT = os.path.abspath(os.path.join(HERE, '..', '..', '..',
                                   'results', 'IDENTITY_BENCHMARK_2026-05-30',
                                   'embodiment14d'))


def run(cmd):
    print(f"\n$ {' '.join(cmd)}", flush=True)
    p = subprocess.run(cmd, capture_output=False)
    print(f"  rc={p.returncode}", flush=True)
    return p.returncode


def main():
    initial = current_governor()
    print(f"[orch] initial gov={initial}  temp={get_apu_temp_c():.1f}C", flush=True)
    for gov in ('powersave', 'performance'):
        print(f"\n############# T2/T3 @ gov={gov} #############", flush=True)
        ok, _ = set_governor(gov)
        print(f"[orch] set {gov} ok={ok}", flush=True)
        time.sleep(20)
        wait_cool(target_c=56, timeout_s=60)
        run([VENV_PY, os.path.join(HERE, 'collect_sigs.py'), gov])
        wait_cool(target_c=56, timeout_s=60)
        npz = os.path.join(OUT, f'ikaros_sigs_{gov}.npz')
        run([VENV_PY, os.path.join(HERE, 'eval_t2_t3.py'), npz, gov])
        wait_cool(target_c=56, timeout_s=60)
    print(f"\n[restore] -> {initial}", flush=True)
    set_governor(initial)


if __name__ == "__main__":
    main()
