#!/usr/bin/env python3
"""Status of the NS-RAM cluster job queue.

Reports counts, per-node thermals (via SSH), active jobs. JSON output for
parsing or a human-friendly table.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
QUEUE_ROOT = REPO / "research_plan" / "job_queue"

# NOTE (2026-05-17): daedalus IP corrected to 192.168.0.40 (was 192.168.0.37 in CLAUDE.md/README).
NODES = {
    "ikaros":   {"host": "192.168.0.35", "user": "ikaros",   "pwd": None},
    "daedalus": {"host": "192.168.0.40", "user": "daedalus", "pwd": "daedalus"},
    "zgx":      {"host": "192.168.0.41", "user": "naorw",    "pwd": "kernel"},
}


def ssh_run(node: str, cmd: str, timeout: int = 8) -> str:
    cfg = NODES[node]
    base = ["ssh", "-o", "BatchMode=no", "-o", "StrictHostKeyChecking=no",
            "-o", f"ConnectTimeout={timeout}", f"{cfg['user']}@{cfg['host']}", cmd]
    if cfg["pwd"]:
        base = ["sshpass", "-p", cfg["pwd"]] + base
    try:
        out = subprocess.run(base, capture_output=True, text=True, timeout=timeout + 2)
        return out.stdout.strip()
    except Exception as e:
        return f"ERR:{e}"


def probe_thermal(node: str) -> dict:
    if node == "ikaros":
        try:
            apu = int(open("/sys/class/thermal/thermal_zone0/temp").read().strip()) / 1000
        except Exception:
            apu = None
    else:
        out = ssh_run(node, "cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null")
        try:
            apu = int(out) / 1000
        except Exception:
            apu = None
    gpu = None
    if node == "zgx":
        out = ssh_run(node, "nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -1")
        try:
            gpu = float(out.splitlines()[0])
        except Exception:
            gpu = None
    elif node in ("ikaros", "daedalus"):
        cmd = "cat /sys/class/drm/card?/device/hwmon/hwmon*/temp1_input 2>/dev/null | head -1"
        out = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True).stdout.strip() \
            if node == "ikaros" else ssh_run(node, cmd)
        try:
            gpu = int(out) / 1000
        except Exception:
            gpu = None
    return {"apu_c": apu, "gpu_c": gpu}


def worker_alive(node: str) -> bool:
    # Match either local repo path (ikaros) or ~/nsram_queue/worker.sh (remote)
    cmd = f"pgrep -af 'worker.sh {node}' | grep -v grep | wc -l"
    if node == "ikaros":
        out = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True).stdout.strip()
    else:
        out = ssh_run(node, cmd)
    try:
        return int(out.split()[0]) > 0
    except Exception:
        return False


def gather(queue_root: Path) -> dict:
    pending = sorted(queue_root.joinpath("pending").glob("*.json"))
    running = sorted(queue_root.joinpath("running").glob("*.json"))
    done    = sorted(queue_root.joinpath("done").glob("*.json"))
    failed  = sorted(queue_root.joinpath("failed").glob("*.json"))
    now = time.time()
    day_ago = now - 86400

    def load(p: Path) -> dict:
        try:
            return json.loads(p.read_text())
        except Exception:
            return {"id": p.stem, "_parse_error": True}

    running_per_node: dict[str, list] = {}
    for p in running:
        # filename: <id>.<node>.json
        parts = p.stem.split(".")
        node = parts[-1] if len(parts) >= 2 else "?"
        running_per_node.setdefault(node, []).append({"file": p.name, "id": ".".join(parts[:-1]) if len(parts) >= 2 else p.stem, "mtime": p.stat().st_mtime, "age_sec": int(now - p.stat().st_mtime)})

    def recent(files: list[Path]) -> list[dict]:
        out = []
        for p in files:
            mt = p.stat().st_mtime
            if mt < day_ago:
                continue
            d = load(p)
            out.append({"id": d.get("id", p.stem), "mtime_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(mt)), "worker": d.get("worker"), "duration_sec": d.get("duration_sec"), "error": d.get("error")})
        return out

    return {
        "queue_root": str(queue_root),
        "now_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "pending_count": len(pending),
        "running_count": len(running),
        "done_24h_count": sum(1 for p in done if p.stat().st_mtime >= day_ago),
        "failed_24h_count": sum(1 for p in failed if p.stat().st_mtime >= day_ago),
        "running_per_node": running_per_node,
        "done_recent": recent(done),
        "failed_recent": recent(failed),
        "pending_ids": [p.stem for p in pending][:20],
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--json", action="store_true", help="Emit JSON only")
    p.add_argument("--no-probe", action="store_true", help="Skip remote SSH thermal/worker probe")
    p.add_argument("--queue_root", default=str(QUEUE_ROOT))
    args = p.parse_args()

    state = gather(Path(args.queue_root))

    nodes_info = {}
    if not args.no_probe:
        for n in NODES:
            nodes_info[n] = {
                "thermal": probe_thermal(n),
                "worker_alive": worker_alive(n),
            }
    state["nodes"] = nodes_info

    if args.json:
        print(json.dumps(state, indent=2))
        return 0

    print(f"queue: {state['queue_root']}")
    print(f"now:   {state['now_iso']}")
    print(f"pending={state['pending_count']} running={state['running_count']} done24h={state['done_24h_count']} failed24h={state['failed_24h_count']}")
    print()
    print("running:")
    if not state["running_per_node"]:
        print("  (none)")
    for node, jobs in state["running_per_node"].items():
        for j in jobs:
            print(f"  {node:10s} {j['id']:40s} age={j['age_sec']}s")
    print()
    if nodes_info:
        print("nodes:")
        for n, info in nodes_info.items():
            t = info["thermal"]
            apu = f"{t['apu_c']:.1f}C" if t["apu_c"] is not None else "?"
            gpu = f"{t['gpu_c']:.1f}C" if t["gpu_c"] is not None else "?"
            alive = "ALIVE" if info["worker_alive"] else "DEAD"
            print(f"  {n:10s} worker={alive:5s} apu={apu:>6s} gpu={gpu:>6s}")
    print()
    if state["pending_ids"]:
        print("pending (oldest first):")
        for i in state["pending_ids"]:
            print(f"  {i}")
    if state["failed_recent"]:
        print()
        print("recent failures:")
        for f in state["failed_recent"][:5]:
            print(f"  {f['id']:40s} err={f.get('error','')[:80]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
