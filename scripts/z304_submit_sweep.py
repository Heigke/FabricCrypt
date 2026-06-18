"""z304_submit — submit 18 (V_G1, Bf) refit jobs to cluster queue.

Each job runs z304_sebas_three_branch_refit.py for one (V_G1, Bf) cell,
inner 4×4 sweep of (alpha0, Rs) → 16 evaluations per job.

Total: 3 branches × 6 Bf = 18 jobs. node_pref=gpu.

Run on master (ikaros). Workers on ikaros + daedalus pick up.
"""
from __future__ import annotations
import json, os, time, uuid, shlex
from pathlib import Path

REPO = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
QUEUE = REPO / "research_plan/job_queue"

VG1_LIST = [0.2, 0.4, 0.6]
BF_LIST  = [50, 500, 3000, 9000]   # reduced for wall-budget; matches z304 BF_GRID


def submit_job(vg1, bf):
    jid = f"z304_vg1_{vg1:.1f}_bf_{bf}".replace(".", "p")
    job = {
        "id": jid,
        "title": f"z304 refit V_G1={vg1} Bf={bf}",
        "script": "scripts/z304_sebas_three_branch_refit.py",
        "args": ["--vg1", str(vg1), "--bf", str(bf),
                  "--out", f"results/z304_sebas_refit/refit_vg1_{vg1:.1f}_bf_{bf}.json"],
        "node_pref": "gpu",
        "wall_estimate_min": 8,
        "needs_surrogate": None,
        "env_vars": {"HSA_OVERRIDE_GFX_VERSION": "11.0.0"},
        "out_dir": "results/z304_sebas_refit/",
        "submitted_at": time.time(),
        "submitted_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "retry_count": 0,
    }
    pending = QUEUE / "pending"
    pending.mkdir(parents=True, exist_ok=True)
    tmp = pending / f".{jid}.tmp"
    final = pending / f"{jid}.json"
    tmp.write_text(json.dumps(job, indent=2))
    os.rename(tmp, final)
    return jid


def main():
    submitted = []
    for vg1 in VG1_LIST:
        for bf in BF_LIST:
            jid = submit_job(vg1, bf)
            submitted.append(jid)
            print(f"submitted {jid}")
    print(f"\nTotal submitted: {len(submitted)} jobs")


if __name__ == "__main__":
    main()
