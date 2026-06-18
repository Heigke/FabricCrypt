"""z293: submit all envelope sweep cells to the queue.

Phase 4B.1: N-scaling (5 jobs)
Phase 4B.2: noise tolerance (4 jobs)
Phase 4B.3: V_d 2D sweep (16 jobs)
Total: 25 jobs
"""
from __future__ import annotations
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SUBMIT = REPO / "scripts" / "queue" / "submit_job.py"
SURROGATE = "results/z278_mep2_surrogate_v3/surrogate_4d_v3.npz"
SCRIPT = "scripts/z293_hdc_envelope_sweep.py"
RESULTS_ROOT = "results/z293_envelope"
SEEDS = "0 1 2 3"


def submit(jid: str, args: str, out_dir: str,
           node_pref: str = "any", wall_min: int = 60) -> str:
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


jobs = []

# 4B.1 N-scaling: N in {64,128,256,512,1024}, V_d=2.00/0.50, V_G1=V_G2=0.30, sigma=0
for N in (64, 128, 256, 512, 1024):
    jid = f"z293_4B1_N{N}"
    out_dir = f"{RESULTS_ROOT}/4B1_Nscaling/N{N}"
    # Larger N => more compute. Estimate wall.
    wall = {64: 30, 128: 45, 256: 60, 512: 90, 1024: 120}[N]
    args = (f"--N {N} --vd_high 2.00 --vd_low 0.50 --vg1 0.30 --vg2 0.30 "
            f"--sigma_noise 0.0 --seeds {SEEDS} --out_dir {out_dir}")
    jobs.append((jid, args, out_dir, "gpu", wall))

# 4B.2 noise tolerance: sigma in {0,0.05,0.10,0.20} at N=128, V_d=2.00/0.50
for sigma in (0.0, 0.05, 0.10, 0.20):
    sig_tag = f"{sigma:.2f}".replace(".", "p")
    jid = f"z293_4B2_sigma{sig_tag}"
    out_dir = f"{RESULTS_ROOT}/4B2_noise/sigma{sig_tag}"
    args = (f"--N 128 --vd_high 2.00 --vd_low 0.50 --vg1 0.30 --vg2 0.30 "
            f"--sigma_noise {sigma} --seeds {SEEDS} --out_dir {out_dir}")
    jobs.append((jid, args, out_dir, "gpu", 45))

# 4B.3 V_d 2D sweep: 4x4=16 (HIGH > LOW always satisfied for our ranges)
for vd_high in (1.5, 2.0, 2.5, 3.0):
    for vd_low in (0.0, 0.2, 0.5, 1.0):
        # All HIGH > LOW for these grids; no skip
        hi_tag = f"{vd_high:.1f}".replace(".", "p")
        lo_tag = f"{vd_low:.1f}".replace(".", "p")
        jid = f"z293_4B3_H{hi_tag}_L{lo_tag}"
        out_dir = f"{RESULTS_ROOT}/4B3_vd_grid/H{hi_tag}_L{lo_tag}"
        args = (f"--N 128 --vd_high {vd_high} --vd_low {vd_low} "
                f"--vg1 0.30 --vg2 0.30 --sigma_noise 0.0 "
                f"--seeds {SEEDS} --out_dir {out_dir}")
        jobs.append((jid, args, out_dir, "gpu", 45))


def main():
    print(f"[z293] submitting {len(jobs)} jobs")
    for jid, args, out_dir, pref, wall in jobs:
        submit(jid, args, out_dir, node_pref=pref, wall_min=wall)
        print(f"  + {jid}  ({pref}, ~{wall}m)")
    print(f"[z293] all {len(jobs)} jobs queued.")


if __name__ == "__main__":
    main()
