#!/usr/bin/env python3
"""
01_puf_signature.py — Phase 1 PUF signature extraction.

Launches puf_kernel binary across N reps under up to 3 thermal regimes,
aggregates per-CU stable bits + process-stat descriptors, writes
signature.json + raw.npz to results dir.

Usage:
    python 01_puf_signature.py --device-name ikaros [--reps 500] [--thermal cold,idle,warm]
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
REPO_ROOT = HERE.parent.parent
PUF_BIN = HERE / "puf_kernel"
THERMAL_ZONE = Path("/sys/class/thermal/thermal_zone0/temp")
DVFS_PATH = Path("/sys/class/drm/card0/device/power_dpm_force_performance_level")

PAUSE_AT_C = 75.0
ABORT_AT_C = 85.0


def read_apu_temp() -> float:
    try:
        return int(THERMAL_ZONE.read_text().strip()) / 1000.0
    except Exception:
        return -1.0


def set_dvfs(level: str) -> bool:
    """Try to set DVFS level. Return True on success, False if r/o."""
    if not DVFS_PATH.exists():
        return False
    try:
        DVFS_PATH.write_text(level)
        return True
    except PermissionError:
        return False


def thermal_wait(threshold_c: float = 60.0, timeout_s: float = 120.0) -> float:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        t = read_apu_temp()
        if t < threshold_c:
            return t
        time.sleep(3.0)
    return read_apu_temp()


def run_kernel(reps: int, out_bin: Path) -> None:
    env = os.environ.copy()
    # gfx1151 native — do NOT use HSA_OVERRIDE
    env.pop("HSA_OVERRIDE_GFX_VERSION", None)
    cmd = [str(PUF_BIN), str(reps), str(out_bin)]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=180)
    if r.returncode != 0:
        raise RuntimeError(f"puf_kernel failed: {r.stderr}\n{r.stdout}")


def parse_bin(path: Path) -> np.ndarray:
    """Return array shape (n_reps, n_waves, 8) uint32."""
    raw = np.fromfile(path, dtype=np.uint32)
    n_reps, n_waves, fields, magic = raw[0], raw[1], raw[2], raw[3]
    assert magic == 0x50554631, f"bad magic 0x{magic:08x}"
    payload = raw[4:].reshape(int(n_reps), int(n_waves), int(fields))
    return payload


def compute_signature(samples: np.ndarray, label: str) -> dict:
    """
    samples: (n_reps, n_waves, 8) uint32.
    Fields: 0=dot_bits, 1=hw_id, 2=cyc_delta, 3=perf, 4=old0, 5=xor_all,
            6=cu_global, 7=cyc_start
    """
    n_reps, n_waves, _ = samples.shape
    dot_bits = samples[:, :, 0]
    hw_id = samples[:, :, 1]
    cyc_delta = samples[:, :, 2].astype(np.int64)
    perf = samples[:, :, 3]
    old0 = samples[:, :, 4]
    cu_global = samples[:, :, 6]

    # --- Aggregate per-CU ---
    cu_ids = np.unique(cu_global)
    cu_ids = cu_ids[cu_ids != 0xFFFFFFFF]
    n_cu = len(cu_ids)

    dot_per_cu_med = np.zeros(n_cu, dtype=np.float64)
    cyc_per_cu_med = np.zeros(n_cu, dtype=np.float64)
    cyc_per_cu_std = np.zeros(n_cu, dtype=np.float64)
    perf_per_cu_med = np.zeros(n_cu, dtype=np.float64)

    # per-CU time series (median across waves of that CU, per rep)
    cu_ts_cyc = np.zeros((n_cu, n_reps), dtype=np.float64)
    cu_ts_perf = np.zeros((n_cu, n_reps), dtype=np.float64)
    cu_ts_dot = np.zeros((n_cu, n_reps), dtype=np.float64)

    for ci, cu in enumerate(cu_ids):
        # which (rep, wave) cells belong to this CU
        mask = (cu_global == cu)
        # dot as float
        dot_f = dot_bits.view(np.float32)
        d_vals = dot_f[mask]
        dot_per_cu_med[ci] = np.median(d_vals)
        cyc_per_cu_med[ci] = np.median(cyc_delta[mask])
        cyc_per_cu_std[ci] = np.std(cyc_delta[mask])
        perf_per_cu_med[ci] = np.median(perf[mask])

        # per-rep summaries (need to handle missing reps for a CU)
        for r in range(n_reps):
            row_mask = mask[r]
            if row_mask.any():
                cu_ts_cyc[ci, r] = np.median(cyc_delta[r][row_mask])
                cu_ts_perf[ci, r] = np.median(perf[r][row_mask])
                cu_ts_dot[ci, r] = np.median(dot_f[r][row_mask])
            else:
                cu_ts_cyc[ci, r] = np.nan
                cu_ts_perf[ci, r] = np.nan
                cu_ts_dot[ci, r] = np.nan

    # --- Stable bit signature ---
    # For each CU: compare each rep's median cycle vs its overall median.
    # Bit = 1 if above-median, 0 if below. Stability = fraction of reps that
    # agree with the modal bit.
    n_bits_per_cu = 8  # we'll build 8 bits per CU from different statistics
    # Bits 0-3: cycle quartiles per CU
    # Bits 4-5: dot mantissa low bits (median dot)
    # Bit 6:    perf parity (population count parity of perf median)
    # Bit 7:    old0 ordering parity
    sig_bits = np.zeros(n_cu * n_bits_per_cu, dtype=np.uint8)
    bit_stability = np.zeros(n_cu * n_bits_per_cu, dtype=np.float64)

    for ci, cu in enumerate(cu_ids):
        # per-rep bit vector for this CU then take majority + stability
        bit_votes = np.zeros((n_reps, n_bits_per_cu), dtype=np.uint8)
        ref_cyc = np.nanmedian(cu_ts_cyc[ci])
        ref_dot = np.nanmedian(cu_ts_dot[ci])
        for r in range(n_reps):
            c = cu_ts_cyc[ci, r]
            d = cu_ts_dot[ci, r]
            p = cu_ts_perf[ci, r]
            o = old0[r][cu_global[r] == cu]
            if not np.isfinite(c):
                continue
            bit_votes[r, 0] = int(c > ref_cyc)
            bit_votes[r, 1] = int((int(c) >> 1) & 1)
            bit_votes[r, 2] = int((int(c) >> 3) & 1)
            bit_votes[r, 3] = int((int(c) >> 5) & 1)
            # dot mantissa low bits
            db = int(np.float32(d).view(np.uint32))
            bit_votes[r, 4] = (db >> 0) & 1
            bit_votes[r, 5] = (db >> 1) & 1
            # perf parity
            pv = int(p) & 0xFFFFFFFF
            bit_votes[r, 6] = bin(pv).count("1") & 1
            # ordering parity
            if o.size > 0:
                ov = int(o[0])
                bit_votes[r, 7] = bin(ov & 0xFF).count("1") & 1

        for b in range(n_bits_per_cu):
            v = bit_votes[:, b]
            ones = int(v.sum())
            zeros = n_reps - ones
            sig_bits[ci * n_bits_per_cu + b] = 1 if ones >= zeros else 0
            bit_stability[ci * n_bits_per_cu + b] = max(ones, zeros) / max(n_reps, 1)

    # --- Process-stat channel ---
    # 1/f knee estimate on cycle time-series per CU
    knees = []
    for ci in range(n_cu):
        ts = cu_ts_cyc[ci]
        ts = ts[np.isfinite(ts)]
        if ts.size < 16:
            knees.append(np.nan)
            continue
        ts = ts - np.mean(ts)
        fft = np.abs(np.fft.rfft(ts))
        freqs = np.fft.rfftfreq(ts.size)
        # log-log slope
        keep = (freqs > 1e-3) & (fft > 0)
        if keep.sum() < 4:
            knees.append(np.nan)
            continue
        slope, _ = np.polyfit(np.log(freqs[keep] + 1e-12),
                              np.log(fft[keep] + 1e-12), 1)
        knees.append(float(slope))
    knees = np.array(knees, dtype=np.float64)

    # spatial correlation between CUs from cycle time series
    valid = np.all(np.isfinite(cu_ts_cyc), axis=1)
    if valid.sum() > 1:
        corr = np.corrcoef(cu_ts_cyc[valid])
    else:
        corr = np.zeros((n_cu, n_cu))

    # RTN telegraph detection on perf time series: count two-state transitions
    rtn_counts = []
    for ci in range(n_cu):
        ts = cu_ts_perf[ci]
        ts = ts[np.isfinite(ts)]
        if ts.size < 8:
            rtn_counts.append(0)
            continue
        med = np.median(ts)
        binar = (ts > med).astype(np.int8)
        transitions = int(np.abs(np.diff(binar)).sum())
        rtn_counts.append(transitions / max(ts.size - 1, 1))
    rtn_counts = np.array(rtn_counts, dtype=np.float64)

    # Noise control: raw PERF distribution (platform-wide)
    perf_flat = perf.flatten()
    perf_hist, perf_edges = np.histogram(perf_flat, bins=32)

    out = {
        "label": label,
        "n_reps": int(n_reps),
        "n_waves": int(n_waves),
        "n_cu": int(n_cu),
        "cu_ids": cu_ids.astype(int).tolist(),
        "sig_bits": sig_bits.tolist(),
        "bit_stability_mean": float(np.mean(bit_stability)),
        "bit_stability_min": float(np.min(bit_stability)),
        "process_stat": {
            "knee_slopes": [None if not np.isfinite(x) else float(x) for x in knees],
            "spatial_corr_mean_abs": float(np.mean(np.abs(corr - np.eye(corr.shape[0])))),
            "rtn_transition_rate_per_cu": rtn_counts.tolist(),
        },
        "noise_control": {
            "perf_hist": perf_hist.tolist(),
            "perf_edges": perf_edges.tolist(),
            "perf_mean": float(np.mean(perf_flat)),
            "perf_std": float(np.std(perf_flat)),
        },
        "per_cu_summary": {
            "dot_median": dot_per_cu_med.tolist(),
            "cyc_median": cyc_per_cu_med.tolist(),
            "cyc_std": cyc_per_cu_std.tolist(),
            "perf_median": perf_per_cu_med.tolist(),
        },
    }
    return out, {
        "samples": samples,
        "cu_ts_cyc": cu_ts_cyc,
        "cu_ts_perf": cu_ts_perf,
        "cu_ts_dot": cu_ts_dot,
        "sig_bits": sig_bits,
        "bit_stability": bit_stability,
        "spatial_corr": corr,
        "knees": knees,
        "rtn": rtn_counts,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device-name", required=True)
    ap.add_argument("--reps", type=int, default=500)
    ap.add_argument("--batch", type=int, default=10,
                    help="reps per kernel invocation (thermal break between)")
    ap.add_argument("--thermal", default="idle",
                    help="comma list from {cold,idle,warm}")
    ap.add_argument("--results-root", default=None)
    args = ap.parse_args()

    results_root = Path(args.results_root) if args.results_root else (
        REPO_ROOT / "results" / "IDENTITY_BENCHMARK_2026-05-30" / args.device_name
    )
    results_root.mkdir(parents=True, exist_ok=True)

    regimes = [r.strip() for r in args.thermal.split(",") if r.strip()]
    print(f"[puf] device={args.device_name} reps={args.reps} regimes={regimes}")
    print(f"[puf] results -> {results_root}")

    all_signatures = {}
    for regime in regimes:
        print(f"\n=== regime={regime} ===")
        if regime == "cold":
            set_dvfs("low")
            t = thermal_wait(threshold_c=45.0, timeout_s=60.0)
            print(f"[puf] cold pre-temp={t:.1f}C")
        elif regime == "warm":
            set_dvfs("high")
            print(f"[puf] warm pre-temp={read_apu_temp():.1f}C")
        else:
            set_dvfs("auto")

        # collect samples in batches with thermal breaks
        all_payloads = []
        done = 0
        bin_idx = 0
        while done < args.reps:
            t = read_apu_temp()
            if t >= ABORT_AT_C:
                print(f"[puf] ABORT temp={t:.1f}C >= {ABORT_AT_C}")
                break
            if t >= PAUSE_AT_C:
                print(f"[puf] pause at {t:.1f}C, waiting to cool...")
                cooled = thermal_wait(threshold_c=60.0, timeout_s=120.0)
                print(f"[puf]   cooled to {cooled:.1f}C")
            this_batch = min(args.batch, args.reps - done)
            tmp_bin = results_root / f"_raw_{regime}_b{bin_idx:04d}.bin"
            run_kernel(this_batch, tmp_bin)
            payload = parse_bin(tmp_bin)
            all_payloads.append(payload)
            tmp_bin.unlink(missing_ok=True)
            done += this_batch
            bin_idx += 1
            if bin_idx % 5 == 0:
                print(f"  reps={done}/{args.reps} temp={read_apu_temp():.1f}C")
            # small thermal break between batches
            time.sleep(0.2)

        if not all_payloads:
            print(f"[puf] no payloads for regime={regime}")
            continue
        samples = np.concatenate(all_payloads, axis=0)
        sig, raw = compute_signature(samples, f"{args.device_name}_{regime}")

        # save raw + sig per regime
        npz_path = results_root / f"raw_{regime}.npz"
        np.savez_compressed(npz_path, **{k: np.asarray(v) for k, v in raw.items()
                                         if isinstance(v, np.ndarray)})
        sig["raw_npz"] = str(npz_path)
        sig["regime"] = regime
        sig["mean_temp_post_C"] = read_apu_temp()
        all_signatures[regime] = sig
        print(f"[puf] regime={regime} sig stability mean={sig['bit_stability_mean']:.3f} "
              f"min={sig['bit_stability_min']:.3f} n_cu={sig['n_cu']}")

    set_dvfs("auto")

    # combined signature file
    out = {
        "device_name": args.device_name,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "reps_per_regime": args.reps,
        "regimes": all_signatures,
    }
    sig_path = results_root / "signature.json"
    with open(sig_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[puf] wrote {sig_path}")
    print(f"[puf] DONE")


if __name__ == "__main__":
    main()
