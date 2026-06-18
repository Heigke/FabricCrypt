"""Sequentially train all remaining models with adaptive thermal handling.

We have parallel embodiment19 contention so we adopt:
  - small bursts (45s)
  - resume from any existing partial checkpoint
  - 200-step target if thermal contention severe; 500 if free
  - skip already-completed models (>= target steps in meta)
"""
import os, sys, json, subprocess, time

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = '/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/IDENTITY_BENCHMARK_2026-05-30/embodiment18'
THERMAL = '/sys/class/thermal/thermal_zone0/temp'

def temp_c():
    return int(open(THERMAL).read()) / 1000.0

def steps_done_for(tag):
    p = os.path.join(RESULTS, f"meta_{tag}.json")
    if not os.path.exists(p): return 0
    with open(p) as f: m = json.load(f)
    return int(m.get('steps_done', 0))


TARGETS = [
    ('ikaros', 'synthetic_matched'),
    ('daedalus', 'vanilla'),
    ('daedalus', 'chip_injected'),
    ('daedalus', 'synthetic_matched'),
]
GOAL = 500
BURST_S = 45  # small bursts

env = os.environ.copy()
env['HSA_OVERRIDE_GFX_VERSION'] = '11.0.0'

for host, cond in TARGETS:
    tag = f"{host}_{cond}"
    done = steps_done_for(tag)
    if done >= GOAL:
        print(f"[skip] {tag} already at {done} steps")
        continue
    # check if a .pt exists -> resume
    has_ckpt = os.path.exists(os.path.join(RESULTS, f"{tag}.pt"))
    remaining = GOAL - done
    cmd = ['/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/venv/bin/python', '-u',
           os.path.join(HERE, 'train_chip.py'),
           '--host', host, '--cond', cond,
           '--steps', str(GOAL), '--burst', str(BURST_S),
           '--bursts', '20']
    if has_ckpt and done > 0:
        cmd += ['--resume', '--resume_steps', str(done)]
        print(f"[run] {tag} RESUME from {done}, target {GOAL}")
    else:
        print(f"[run] {tag} FRESH, target {GOAL}")
    sys.stdout.flush()
    t0 = time.time()
    rc = subprocess.call(cmd, env=env, cwd='/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy')
    dt = time.time() - t0
    print(f"[done] {tag} rc={rc} elapsed={dt:.1f}s temp={temp_c():.1f}C")
print("[all_done]")
