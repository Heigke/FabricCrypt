#!/usr/bin/env python3
"""Angle ECC — DDR/GDDR ECC bad-block map via /sys EDAC.

Pre-registered DISCOVERY gate: >=10 unique cells with corrected errors AND
>=50% non-overlap between devices.

On Strix Halo APUs the GPU shares system DDR5 and EDAC may have no
controllers registered (unified memory, ECC not exposed). In that case
we record NULL with the device's full EDAC topology dump so the absence
is auditable.
"""
from __future__ import annotations
import json, os, socket, sys, time, glob
from pathlib import Path

_here = Path(__file__).resolve()
_repo_candidate = _here.parents[3] if len(_here.parents) > 3 else _here.parent
if (_repo_candidate / "results").exists():
    OUT_DIR = _repo_candidate / "results/IDENTITY_BENCHMARK_2026-05-30/novel_v2"
else:
    OUT_DIR = _here.parent / "results_novel_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EDAC = Path("/sys/devices/system/edac")
POLL_HZ = 1.0
DURATION_S = 60

def read_int(p: Path):
    try:
        return int(p.read_text().strip())
    except Exception:
        return None

def scan_edac():
    """Return dict: {mc_id: {ce_count, ue_count, csrow_channels: {row.ch: ce_count}}}."""
    out = {}
    for mc in sorted(EDAC.glob("mc/mc*")):
        rec = {"ce_count": read_int(mc/"ce_count"),
               "ue_count": read_int(mc/"ue_count"),
               "csrow_channels": {}}
        for csrow in sorted(mc.glob("csrow*")):
            for ch in sorted(csrow.glob("ch*_ce_count")):
                key = f"{csrow.name}.{ch.name}"
                rec["csrow_channels"][key] = read_int(ch)
        out[mc.name] = rec
    return out

def topology_dump():
    """Static EDAC tree topology for audit trail."""
    listing = []
    for root, dirs, files in os.walk(EDAC):
        for d in dirs:
            listing.append(os.path.join(root, d))
        for f in files:
            listing.append(os.path.join(root, f))
    return sorted(listing)[:500]

def main():
    host = socket.gethostname()
    samples = []
    t0 = time.time()
    step = 1.0/POLL_HZ
    n = int(DURATION_S * POLL_HZ)
    print(f"[ECC] host={host} polling EDAC {n} samples @ {POLL_HZ}Hz", file=sys.stderr)
    for i in range(n):
        samples.append({"t": time.time()-t0, "edac": scan_edac()})
        # gentle: cheap call, but sleep
        next_t = (i+1)*step
        sleep_for = max(0.0, t0+next_t - time.time())
        time.sleep(sleep_for)
    # Aggregate: unique (mc, csrow.ch) addresses with any ce>0 over the run
    cell_max_ce = {}
    for s in samples:
        for mc, rec in s["edac"].items():
            for k, v in rec.get("csrow_channels", {}).items():
                key = f"{mc}/{k}"
                if v is not None and v > 0:
                    cell_max_ce[key] = max(cell_max_ce.get(key, 0), v)
    out = {
        "host": host,
        "duration_s": DURATION_S,
        "poll_hz": POLL_HZ,
        "n_samples": len(samples),
        "n_controllers": len(samples[0]["edac"]) if samples else 0,
        "n_unique_error_cells": len(cell_max_ce),
        "cell_max_ce_counts": cell_max_ce,
        "topology_present": [p for p in topology_dump() if "/mc" in p][:200],
        "final_snapshot": samples[-1]["edac"] if samples else {},
        "discovery_gate": {
            "threshold_cells": 10,
            "passes_gate": len(cell_max_ce) >= 10,
        },
    }
    if out["n_controllers"] == 0:
        out["verdict"] = ("NULL — EDAC subsystem present but zero memory controllers "
                          "registered. Strix Halo APU unified DDR5 is not exposed via "
                          "EDAC; per-channel ECC counters unavailable. Angle is "
                          "cheaply falsified on this platform.")
    elif out["n_unique_error_cells"] == 0:
        out["verdict"] = ("NULL — controllers present but no corrected errors observed "
                          f"in {DURATION_S}s. Healthy RAM, identity angle gives no signal.")
    else:
        out["verdict"] = f"SIGNAL — {out['n_unique_error_cells']} unique error cells observed."
    fname = OUT_DIR / f"ECC_{host}.json"
    fname.write_text(json.dumps(out, indent=2))
    print(json.dumps({"host": host, "verdict": out["verdict"],
                      "n_controllers": out["n_controllers"],
                      "n_error_cells": out["n_unique_error_cells"]}, indent=2))
    print(f"[ECC] wrote {fname}", file=sys.stderr)

if __name__ == "__main__":
    main()
