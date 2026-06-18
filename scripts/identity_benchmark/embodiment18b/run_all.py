"""Phase 18B orchestrator — train all conditions on ikaros.

Original plan: 3 conditions x 2 hosts = 6 models. Daedalus unreachable
(192.168.0.37 timeout 2026-06-01), so we run all 6 on ikaros and use
distinct run_ids/nonces to create chip-signature variants. Honest caveat:
'daedalusA' runs do NOT come from a second physical chip; they use the
same live ikaros signature with a different nonce permutation. This means
H4 (clone defeat) is degraded from "different chip" to "different nonce
permutation of the same chip". We document this prominently.
"""
from __future__ import annotations
import os, sys, time, json, subprocess
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _common import save_json, wait_cool, temp_c, RESULTS

# Total budget: 6 * STEPS_PER * ~3s + cooling. Keep modest.
STEPS_PER = int(os.environ.get('STEPS_PER', '60'))
MAX_WALL_PER = int(os.environ.get('MAX_WALL_PER', '1500'))

RUNS = [
    # (run_id, condition, synth_seed_or_0)
    ('ikarosA_vanilla',           'vanilla',            0),
    ('ikarosA_chip',              'chip_injected',      0),
    ('ikarosA_synthmatched',      'synthetic_matched',  0xA1B2C3D4),
    ('hostB_vanilla',             'vanilla',            0),
    ('hostB_chip',                'chip_injected',      0),
    ('hostB_synthmatched',        'synthetic_matched',  0xDEADBEEF),
]


def main():
    summary = {'runs': [], 'start_t': time.time()}
    for (rid, cond, synth_seed) in RUNS:
        print(f"\n{'='*70}\n[orchestrator] starting {rid} ({cond}) "
              f"steps={STEPS_PER}\n{'='*70}", flush=True)
        # Relaxed cool target (55C) so co-tenant trainings don't block us.
        # In-script per-step guard at abort=65, pause=60 still enforces safety.
        if not wait_cool(target_c=58, timeout_s=120):
            print(f"[orchestrator] could not cool to 58C in 120s, proceeding anyway "
                  f"(T={temp_c():.1f}C)", flush=True)
        cmd = [
            sys.executable, os.path.join(HERE, 'train_gpt2_chip.py'),
            '--cond', cond, '--run_id', rid,
            '--steps', str(STEPS_PER),
            '--max_wall_s', str(MAX_WALL_PER),
            '--synth_seed', str(synth_seed),
        ]
        env = os.environ.copy()
        env['HSA_OVERRIDE_GFX_VERSION'] = '11.0.0'
        env['PYTHONUNBUFFERED'] = '1'
        t0 = time.time()
        try:
            r = subprocess.run(cmd, env=env, timeout=MAX_WALL_PER + 300)
            rc = r.returncode
        except subprocess.TimeoutExpired:
            rc = -1
        except KeyboardInterrupt:
            print("[orchestrator] keyboard interrupt", flush=True)
            break
        summary['runs'].append({
            'run_id': rid, 'condition': cond, 'rc': rc,
            'wall_s': time.time() - t0,
            'temp_after_c': temp_c(),
        })
        # Long cool between runs
        wait_cool(target_c=58, timeout_s=120)

    summary['end_t'] = time.time()
    summary['total_wall_s'] = summary['end_t'] - summary['start_t']
    save_json('orchestrator_summary.json', summary)
    print(f"[orchestrator] done. total wall = {summary['total_wall_s']/60:.1f} min",
          flush=True)


if __name__ == '__main__':
    main()
