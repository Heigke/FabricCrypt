"""z302: submit noise-robustness jobs (3 strategies) to the queue."""
from __future__ import annotations
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SUBMIT = REPO / "scripts" / "queue" / "submit_job.py"
SURROGATE = "results/z278_mep2_surrogate_v3/surrogate_4d_v3.npz"
SCRIPT = "scripts/z302_hdc_noise_robust.py"
RESULTS_ROOT = "results/z302_hdc_noise_robust"
SEEDS = "0 1 2 3"


def submit(jid: str, args: str, out_dir: str,
           node_pref: str = "gpu", wall_min: int = 60) -> str:
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
    return f"{x:.2f}".replace(".", "p").replace("-", "m")


jobs = []

# Strategy A: noise-injection during encoding.
# sigma_train in {0.00, 0.02, 0.05, 0.10}  x  sigma_test in {0.00, 0.05, 0.10, 0.20}
# N=1024, V_d=2.0/0.5 (headline cell).
for s_tr in (0.00, 0.02, 0.05, 0.10):
    for s_te in (0.00, 0.05, 0.10, 0.20):
        jid = f"z302_A_str{tag(s_tr)}_ste{tag(s_te)}"
        out_dir = f"{RESULTS_ROOT}/A_noisetrain/str{tag(s_tr)}_ste{tag(s_te)}"
        args = (f"--N 1024 --vd_high 2.00 --vd_low 0.50 --vg1 0.30 --vg2 0.30 "
                f"--sigma_train {s_tr} --sigma_test {s_te} "
                f"--strategy A_noisetrain --seeds {SEEDS} --out_dir {out_dir}")
        jobs.append((jid, args, out_dir, "gpu", 60))

# Strategy B: N-scaling for SNR at sigma_test=0.05 (no train noise).
# N in {1024, 2048, 4096}.
for N in (1024, 2048, 4096):
    jid = f"z302_B_N{N}"
    out_dir = f"{RESULTS_ROOT}/B_nscale/N{N}"
    wall = {1024: 60, 2048: 120, 4096: 240}[N]
    args = (f"--N {N} --vd_high 2.00 --vd_low 0.50 --vg1 0.30 --vg2 0.30 "
            f"--sigma_train 0.0 --sigma_test 0.05 "
            f"--strategy B_nscale --seeds {SEEDS} --out_dir {out_dir}")
    jobs.append((jid, args, out_dir, "gpu", wall))

# Strategy C: wider V_d separation at sigma_test=0.05 (no train noise).
# Baseline (Delta=1.5: 2.0/0.5) covered by A_str0p00_ste0p05. Add wider deltas.
vd_pairs = [
    (2.2, 0.4),  # Delta=1.8
    (2.4, 0.4),  # Delta=2.0
    (2.5, 0.5),  # Delta=2.0 alt (HIGH up, LOW unchanged)
    (3.0, 0.5),  # Delta=2.5 (push limit)
]
for vh, vl in vd_pairs:
    jid = f"z302_C_H{tag(vh)}_L{tag(vl)}"
    out_dir = f"{RESULTS_ROOT}/C_vdwide/H{tag(vh)}_L{tag(vl)}"
    args = (f"--N 1024 --vd_high {vh} --vd_low {vl} --vg1 0.30 --vg2 0.30 "
            f"--sigma_train 0.0 --sigma_test 0.05 "
            f"--strategy C_vdwide --seeds {SEEDS} --out_dir {out_dir}")
    jobs.append((jid, args, out_dir, "gpu", 60))


def main():
    print(f"[z302] submitting {len(jobs)} jobs")
    for jid, args, out_dir, pref, wall in jobs:
        submit(jid, args, out_dir, node_pref=pref, wall_min=wall)
        print(f"  + {jid}  ({pref}, ~{wall}m)")
    print(f"[z302] all {len(jobs)} jobs queued.")


if __name__ == "__main__":
    main()
