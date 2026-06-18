#!/usr/bin/env python3
"""Submit a job to the file-based queue.

Writes a JSON descriptor into research_plan/job_queue/pending/<id>.json on the
master (ikaros). The atomic mv-claim protocol guarantees a single worker picks
up each job.

Run locally on ikaros, or via ssh from a worker.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
import time
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
QUEUE_ROOT = REPO / "research_plan" / "job_queue"


def main() -> int:
    p = argparse.ArgumentParser(description="Submit a job to the NS-RAM cluster queue")
    p.add_argument("--id", default=None, help="Job id (default: uuid)")
    p.add_argument("--title", default=None, help="Human-readable title")
    p.add_argument("--script", required=True, help="Path to python script (relative to repo root)")
    p.add_argument("--args", default="", help="Argument string passed to the script (shell-split)")
    p.add_argument("--node_pref", default="any", choices=["gpu", "cpu", "any"], help="Node preference hint")
    p.add_argument("--wall_estimate_min", type=int, default=30)
    p.add_argument("--needs_surrogate", default=None, help="Optional path to surrogate file required by the job")
    p.add_argument("--env", action="append", default=[], help="VAR=VALUE (repeatable)")
    p.add_argument("--out_dir", default="results/", help="Job output dir (relative to repo root)")
    p.add_argument("--queue_root", default=str(QUEUE_ROOT))
    args = p.parse_args()

    job_id = args.id or f"job_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    if "/" in job_id or " " in job_id:
        print(f"FATAL: bad job id {job_id!r}", file=sys.stderr)
        return 2

    env_vars: dict[str, str] = {}
    for e in args.env:
        if "=" not in e:
            print(f"FATAL: --env must be VAR=VALUE, got {e!r}", file=sys.stderr)
            return 2
        k, v = e.split("=", 1)
        env_vars[k] = v

    job = {
        "id": job_id,
        "title": args.title or job_id,
        "script": args.script,
        "args": shlex.split(args.args) if args.args else [],
        "node_pref": args.node_pref,
        "wall_estimate_min": args.wall_estimate_min,
        "needs_surrogate": args.needs_surrogate,
        "env_vars": env_vars,
        "out_dir": args.out_dir,
        "submitted_at": time.time(),
        "submitted_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "retry_count": 0,
    }

    pending_dir = Path(args.queue_root) / "pending"
    pending_dir.mkdir(parents=True, exist_ok=True)
    # Write atomically: tmp + rename
    tmp = pending_dir / f".{job_id}.tmp"
    final = pending_dir / f"{job_id}.json"
    tmp.write_text(json.dumps(job, indent=2))
    os.rename(tmp, final)
    print(f"submitted: {final}")
    print(f"id: {job_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
