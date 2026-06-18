#!/usr/bin/env python3
"""probeAB_runner.py — drive probe A (LDS+FMA) and B (RO-pair) binaries,
parse output, save JSON results.

Usage:
    python probeAB_runner.py --device-name ikaros --out-dir <results>
        [--reps-a 10000] [--races-b 10000]
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _thermal import (read_temp, wait_cool, inter_burst_sleep, ThermalMonitor,
                      MAX_BURST_S, COOL_TARGET_C, HOSTNAME)

PROBE_A = HERE / "probeA_lds_startup"
PROBE_B = HERE / "probeB_ro_pair"

# Hardened batch sizes
PROBE_A_REPS_PER_BURST = 200        # ~25 ms per burst (light) but enforce anyway
PROBE_B_RACES_PER_BURST = 500       # ~50 ms per burst
PROBE_B_MAX_TOTAL = 20 * 1000       # cap: 20 CU-pairs × 1000 races, harness clips externally

def run_probeA(reps: int, out_bin: Path) -> dict:
    """Run probeA in batches of PROBE_A_REPS_PER_BURST with cooling between bursts.
    The binary writes its own .bin per call; we concatenate payload sections.
    """
    if not PROBE_A.exists():
        return {"ok": False, "err": "binary missing"}
    env = os.environ.copy()
    env.pop("HSA_OVERRIDE_GFX_VERSION", None)
    tm = ThermalMonitor("probeA")
    payload_chunks = []
    header = None
    total_reps = 0
    remaining = reps
    with tm:
        batch_idx = 0
        while remaining > 0:
            n = min(PROBE_A_REPS_PER_BURST, remaining)
            tmp_bin = out_bin.with_suffix(f".part{batch_idx}.bin")
            t_pre = read_temp()
            if t_pre > COOL_TARGET_C:
                wait_cool()
            t_start = time.time()
            r = subprocess.run([str(PROBE_A), str(n), str(tmp_bin)],
                               env=env, capture_output=True, text=True,
                               timeout=int(MAX_BURST_S) + 30)
            dur = time.time() - t_start
            if r.returncode != 0:
                return {"ok": False, "err": r.stderr, "duration_s": dur,
                        "thermal": tm.as_dict()}
            if dur > MAX_BURST_S:
                print(f"[probeA] WARN batch {batch_idx} took {dur:.2f}s > {MAX_BURST_S}s cap")
            raw_batch = np.fromfile(tmp_bin, dtype=np.uint32)
            if header is None:
                header = raw_batch[:4].copy()
            payload_chunks.append(raw_batch[4:])
            total_reps += n
            tmp_bin.unlink()
            tm.sample()
            print(f"[probeA] batch {batch_idx} reps={n} dur={dur:.2f}s T={tm.max_c:.1f}C")
            remaining -= n
            batch_idx += 1
            if remaining > 0:
                inter_burst_sleep()
    # write combined bin
    header[1] = total_reps
    combined = np.concatenate(([header[0], total_reps, header[2], header[3]],
                               np.concatenate(payload_chunks)))
    combined.astype(np.uint32).tofile(out_bin)
    raw = np.fromfile(out_bin, dtype=np.uint32)
    magic, reps_, wpr, _ = raw[:4]
    pl = raw[4:].reshape(int(reps_), int(wpr))
    lds = pl[:, :256]
    fma = pl[:, 256:]
    # Per-lane statistics
    lds_unique_per_lane = (lds != lds[0:1, :]).any(axis=0).sum()
    fma_unique_per_lane = (fma != fma[0:1, :]).any(axis=0).sum()
    return {
        "ok": True, "duration_s": tm.dur_s, "reps": int(reps_),
        "thermal": tm.as_dict(),
        "lds_const_value": int(lds[0, 0]) if (lds == lds[0, 0]).all() else None,
        "lds_lanes_varying": int(lds_unique_per_lane),
        "fma_lanes_varying": int(fma_unique_per_lane),
        "fma_unique_values": int(len(np.unique(fma))),
        "lds_bit_prob1": float(np.unpackbits(lds.view(np.uint8)).mean()),
        "fma_bit_prob1": float(np.unpackbits(fma.view(np.uint8)).mean()),
        "fma_per_lane_mean":  fma.mean(axis=0).tolist(),
        "fma_per_lane_std":   fma.std(axis=0).tolist(),
        "fma_per_lane_mode_value": [int(np.bincount(fma[:, l].astype(np.int64)).argmax()) for l in range(min(fma.shape[1], 256))],
        "fma_per_lane_stable_frac": [float((fma[:, l] == np.bincount(fma[:, l].astype(np.int64)).argmax()).mean()) for l in range(min(fma.shape[1], 256))],
        "bin_file": str(out_bin),
    }

def run_probeB(races: int, out_bin: Path) -> dict:
    if not PROBE_B.exists():
        return {"ok": False, "err": "binary missing"}
    # Apply hard cap
    if races > PROBE_B_MAX_TOTAL:
        print(f"[probeB] capping races {races} -> {PROBE_B_MAX_TOTAL}")
        races = PROBE_B_MAX_TOTAL
    env = os.environ.copy()
    env.pop("HSA_OVERRIDE_GFX_VERSION", None)
    tm = ThermalMonitor("probeB")
    payload_chunks = []
    header = None
    total_races = 0
    remaining = races
    with tm:
        batch_idx = 0
        while remaining > 0:
            n = min(PROBE_B_RACES_PER_BURST, remaining)
            tmp_bin = out_bin.with_suffix(f".part{batch_idx}.bin")
            if read_temp() > COOL_TARGET_C:
                wait_cool()
            t_start = time.time()
            r = subprocess.run([str(PROBE_B), str(n), str(tmp_bin)],
                               env=env, capture_output=True, text=True,
                               timeout=int(MAX_BURST_S) + 30)
            dur = time.time() - t_start
            if r.returncode != 0:
                return {"ok": False, "err": r.stderr, "duration_s": dur,
                        "thermal": tm.as_dict()}
            if dur > MAX_BURST_S:
                print(f"[probeB] WARN batch {batch_idx} took {dur:.2f}s")
            rb = np.fromfile(tmp_bin, dtype=np.uint32)
            if header is None: header = rb[:4].copy()
            payload_chunks.append(rb[4:])
            total_races += n
            tmp_bin.unlink()
            tm.sample()
            print(f"[probeB] batch {batch_idx} races={n} dur={dur:.2f}s T={tm.max_c:.1f}C")
            remaining -= n
            batch_idx += 1
            if remaining > 0:
                inter_burst_sleep()
    header[1] = total_races
    combined = np.concatenate(([header[0], total_races, header[2], header[3]],
                               np.concatenate(payload_chunks)))
    combined.astype(np.uint32).tofile(out_bin)
    raw = np.fromfile(out_bin, dtype=np.uint32)
    magic, races_, fields, _ = raw[:4]
    pl = raw[4:].reshape(int(races_), int(fields))
    winners = pl[:, 0]
    hwid0 = pl[:, 1]
    hwid1 = pl[:, 2]
    dcyc0 = pl[:, 3]
    win_block0 = int((winners == 1).sum())
    win_block1 = int((winners == 2).sum())
    win_none = int((winners == 0).sum())
    # Per-CU-pair win frequency
    # Pair key = (min(hwid0, hwid1), max(hwid0, hwid1))
    pairs = {}
    for i in range(int(races_)):
        a, b = int(hwid0[i]), int(hwid1[i])
        key = (min(a, b), max(a, b))
        win = int(winners[i])  # 1 = block 0 won, 2 = block 1 won
        # Convert to "did min-id-CU win?": if hwid0 < hwid1 and win==1 -> min won
        if win == 0:
            continue
        winner_hwid = a if win == 1 else b
        min_won = (winner_hwid == key[0])
        d = pairs.setdefault(key, {"n": 0, "min_wins": 0})
        d["n"] += 1
        if min_won:
            d["min_wins"] += 1
    pair_stats = []
    for k, v in pairs.items():
        if v["n"] >= 5:
            p_min = v["min_wins"] / v["n"]
            pair_stats.append({"pair": [int(k[0]), int(k[1])], "n": v["n"],
                               "p_min_wins": p_min, "bias": abs(p_min - 0.5)})
    pair_stats.sort(key=lambda d: -d["bias"])
    return {
        "ok": True, "duration_s": tm.dur_s, "races": int(races_),
        "thermal": tm.as_dict(),
        "win_block0": win_block0, "win_block1": win_block1, "win_none": win_none,
        "unique_hwid0": len(np.unique(hwid0).tolist()),
        "unique_hwid1": len(np.unique(hwid1).tolist()),
        "n_distinct_pairs": len(pairs),
        "pair_stats_top20": pair_stats[:20],
        "dcyc0_mean": float(dcyc0.mean()),
        "dcyc0_std": float(dcyc0.std()),
        "bin_file": str(out_bin),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device-name", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--reps-a", type=int, default=10000)
    ap.add_argument("--races-b", type=int, default=10000)
    ap.add_argument("--skip-a", action="store_true")
    ap.add_argument("--skip-b", action="store_true")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    results = {"device": args.device_name, "ts": time.time(),
               "start_temp_c": read_temp()}
    if not args.skip_a:
        wait_cool(60.0)
        print(f"[runner] probeA reps={args.reps_a}")
        results["probeA"] = run_probeA(args.reps_a, args.out_dir / "probeA.bin")
        print(f"[runner] probeA done: ok={results['probeA'].get('ok')} dur={results['probeA'].get('duration_s')}")
    if not args.skip_b:
        wait_cool(60.0)
        print(f"[runner] probeB races={args.races_b}")
        results["probeB"] = run_probeB(args.races_b, args.out_dir / "probeB.bin")
        print(f"[runner] probeB done: ok={results['probeB'].get('ok')} dur={results['probeB'].get('duration_s')}")
    results["end_temp_c"] = read_temp()
    out = args.out_dir / "probeAB_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"[runner] wrote {out}")

if __name__ == "__main__":
    main()
