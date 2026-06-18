"""Collects local research status: log tail, latest verdicts, running procs, APU temp."""
from __future__ import annotations

import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any

REPO = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
LOG_FILE = REPO / "research_plan" / "01_LOG.md"
IDENT_DIR = REPO / "results" / "IDENTITY_BENCHMARK_2026-05-30"
THERMAL_ZONE = Path("/sys/class/thermal/thermal_zone0/temp")

PROC_PATTERNS = ("identity", "FEEL", "nsram", "ngspice", "phase1", "yggdrasil")


def apu_temp_c() -> float:
    try:
        return int(THERMAL_ZONE.read_text().strip()) / 1000.0
    except Exception:
        return -1.0


def log_tail(n: int = 10) -> list[str]:
    if not LOG_FILE.exists():
        return []
    lines = LOG_FILE.read_text(errors="ignore").splitlines()
    return [ln for ln in lines[-n:] if ln.strip()]


def latest_verdicts(limit: int = 3) -> list[dict[str, Any]]:
    if not IDENT_DIR.exists():
        return []
    files = sorted(IDENT_DIR.rglob("verdict*.md"),
                   key=lambda p: p.stat().st_mtime, reverse=True)[:limit]
    out = []
    for f in files:
        try:
            txt = f.read_text(errors="ignore")
            head = "\n".join(txt.splitlines()[:8])
            out.append({"path": str(f.relative_to(REPO)),
                        "mtime": f.stat().st_mtime, "head": head})
        except Exception:
            pass
    return out


def latest_jsons(limit: int = 3) -> list[dict[str, Any]]:
    if not IDENT_DIR.exists():
        return []
    files = sorted(IDENT_DIR.rglob("*.json"),
                   key=lambda p: p.stat().st_mtime, reverse=True)[:limit]
    out = []
    for f in files:
        try:
            d = json.loads(f.read_text())
            # keep only top-level scalars
            summary = {k: v for k, v in d.items()
                       if isinstance(v, (int, float, str, bool)) and len(str(v)) < 80}
            out.append({"path": str(f.relative_to(REPO)),
                        "mtime": f.stat().st_mtime, "summary": summary})
        except Exception:
            pass
    return out


def running_experiments() -> dict[str, int]:
    try:
        ps = subprocess.run(["ps", "-eo", "comm,args"], capture_output=True,
                            text=True, timeout=2).stdout
    except Exception:
        return {}
    counts: dict[str, int] = {}
    for line in ps.splitlines():
        low = line.lower()
        for pat in PROC_PATTERNS:
            if pat.lower() in low:
                counts[pat] = counts.get(pat, 0) + 1
                break
    return counts


def snapshot() -> dict[str, Any]:
    return {
        "ts": time.time(),
        "apu_c": apu_temp_c(),
        "log_tail": log_tail(8),
        "verdicts": latest_verdicts(3),
        "jsons": latest_jsons(3),
        "running": running_experiments(),
    }


def hashable_diff_key(snap: dict[str, Any]) -> str:
    """Stable string used to detect 'nothing new since last poll'."""
    parts = [
        "|".join(snap["log_tail"][-3:]),
        "|".join(v["path"] + str(int(v["mtime"])) for v in snap["verdicts"]),
        "|".join(j["path"] + str(int(j["mtime"])) for j in snap["jsons"]),
        ",".join(f"{k}={v}" for k, v in sorted(snap["running"].items())),
    ]
    return "::".join(parts)


if __name__ == "__main__":
    print(json.dumps(snapshot(), indent=2, default=str))
