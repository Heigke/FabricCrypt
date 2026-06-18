"""z312: HDC headline-curve extension — N in {8192, 16384} x sigma in {0.00, 0.05, 0.10}.

Reuses scripts/z293_hdc_envelope_sweep.py. 6 cell-jobs, 4 seeds each (24 seed-evals).
Submits to GPU queue (will land on zgx via node_pref=gpu).
"""
from __future__ import annotations
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SUBMIT = REPO / "scripts" / "queue" / "submit_job.py"
SURROGATE = "results/z278_mep2_surrogate_v3/surrogate_4d_v3.npz"
SCRIPT = "scripts/z293_hdc_envelope_sweep.py"
RESULTS_ROOT = "results/z312_hdc_n16k"
SEEDS = "0 1 2 3"


def submit(jid, args, out_dir, node_pref="gpu", wall_min=60):
    cmd = [
        sys.executable, str(SUBMIT),
        "--id", jid,
        "--title", jid,
        "--script", SCRIPT,
        "--args", args,
        "--node_pref", node_pref,
        "--wall_estimate_min", str(wall_min),
        "--needs_surrogate", SURROGATE,
        "--env", "HSA_OVERRIDE_GFX_VERSION=11.0.0",
        "--env", "PYTHONUNBUFFERED=1",
        "--out_dir", out_dir,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"FAIL: {jid}\nstderr={r.stderr}", file=sys.stderr)
        sys.exit(2)
    return jid


def tag(x):
    return f"{x:.2f}".replace(".", "p")


jobs = []
# N=16384 estimated ~10 min on zgx; give 60 min wall (4x safety)
# N=8192 estimated ~5 min; give 40 min wall
WALLS = {8192: 40, 16384: 60}
for N in (8192, 16384):
    for sigma in (0.00, 0.05, 0.10):
        s_tag = tag(sigma)
        jid = f"z312_N{N}_s{s_tag}"
        out_dir = f"{RESULTS_ROOT}/N{N}_s{s_tag}"
        args = (f"--N {N} --vd_high 2.00 --vd_low 0.50 --vg1 0.30 --vg2 0.30 "
                f"--sigma_noise {sigma} --seeds {SEEDS} --out_dir {out_dir}")
        jobs.append((jid, args, out_dir, "gpu", WALLS[N]))


def main():
    print(f"[z312] submitting {len(jobs)} jobs")
    for jid, args, out_dir, pref, wall in jobs:
        submit(jid, args, out_dir, node_pref=pref, wall_min=wall)
        print(f"  + {jid}  ({pref}, ~{wall}m)")
    print(f"[z312] all {len(jobs)} jobs queued.")


if __name__ == "__main__":
    main()
