"""z305b_submit — submit 12 (Bf, Rs) jobs of z305b with per-branch ETAB fix."""
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
    jid = f"z305b_bf_{bf}_rs_{rs_tag}"
    out_file = f"results/z305b_etab_perbranch/corrective_bf_{bf}_rs_{rs_tag}.json"
    job = {
        "id": jid,
        "title": f"z305b ETAB per-branch Bf={bf} Rs={rs}",
        "script": "scripts/z305b_etab_perbranch.py",
        "args": ["--bf", str(bf), "--rs", str(rs), "--out", out_file],
        "node_pref": "gpu",
        "wall_estimate_min": 8,
        "needs_surrogate": None,
        "env_vars": {
            "HSA_OVERRIDE_GFX_VERSION": "11.0.0",
            "PYTHONUNBUFFERED": "1",
        },
        "out_dir": "results/z305b_etab_perbranch/",
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
