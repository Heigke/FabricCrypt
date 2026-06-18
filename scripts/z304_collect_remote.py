"""z304_collect_remote — pull refit_*.json from daedalus + zgx into master."""
from __future__ import annotations
import os, subprocess, sys
from pathlib import Path

REPO = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
LOCAL_DIR = REPO / "results/z304_sebas_refit"
LOCAL_DIR.mkdir(parents=True, exist_ok=True)

NODES = [
    ("daedalus", "192.168.0.40", "daedalus", "daedalus",
     "/home/daedalus/AMD_gfx1151_energy/results/z304_sebas_refit/"),
    ("zgx", "192.168.0.41", "naorw", "kernel",
     "/home/naorw/nsram_queue_sandbox/results/z304_sebas_refit/"),
]

for name, host, user, pwd, remote in NODES:
    cmd = ["sshpass", "-p", pwd, "rsync", "-az",
           "-e", "ssh -o BatchMode=no -o StrictHostKeyChecking=no",
           f"{user}@{host}:{remote}", str(LOCAL_DIR) + "/"]
    print(f"[collect] pulling from {name}: {remote}")
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        print(f"  rsync rc={r.returncode}")
        print(f"  stderr: {r.stderr[:300]}")
    else:
        print(f"  OK")

# List what we have
print("\n[collect] local files:")
for f in sorted(LOCAL_DIR.glob("refit_*.json")):
    print(f"  {f.name}  ({f.stat().st_size} bytes)")
