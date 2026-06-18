"""z305_submit — submit 12 (Bf, Rs) corrective jobs to the cluster queue.

Each job runs z305_sa4_corrective.py with one (Bf, Rs) pair and evaluates
all 3 V_G1 branches with SA1-canonical per-branch overrides.

Total: 4 Bf × 3 Rs = 12 jobs. node_pref=gpu.
"""
from __future__ import annotations
import json, os, time
from pathlib import Path

REPO = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
QUEUE = REPO / "research_plan/job_queue"

BF_LIST = [500, 1000, 3000, 9000]
RS_LIST = [0, 1.0e9, 1.0e10]


def submit_job(bf, rs):
    rs_tag = f"{rs:.0e}".replace("+", "").replace("e0", "e") if rs > 0 else "0"
    rs_tag = rs_tag.replace(".", "p")
    jid = f"z305_bf_{bf}_rs_{rs_tag}"
    out_file = f"results/z305_corrective/corrective_bf_{bf}_rs_{rs_tag}.json"
    job = {
        "id": jid,
        "title": f"z305 SA4 corrective Bf={bf} Rs={rs}",
        "script": "scripts/z305_sa4_corrective.py",
        "args": ["--bf", str(bf), "--rs", str(rs), "--out", out_file],
        "node_pref": "gpu",
        "wall_estimate_min": 6,
        "needs_surrogate": None,
        "env_vars": {
            "HSA_OVERRIDE_GFX_VERSION": "11.0.0",
            "PYTHONUNBUFFERED": "1",
        },
        "out_dir": "results/z305_corrective/",
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
    for bf in BF_LIST:
        for rs in RS_LIST:
            jid = submit_job(bf, rs)
            submitted.append(jid)
            print(f"submitted {jid}")
    print(f"\nTotal submitted: {len(submitted)} jobs")


if __name__ == "__main__":
    main()
