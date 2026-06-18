#!/usr/bin/env python3
"""Pre-step: are RTN and spatial-corr the same signal?

Phase 1b claimed 2 surviving channels (RTN scalar per CU and the 80x80
spatial-correlation matrix). All four oracles flagged that these may be
collinear. Test: Pearson r between per-CU RTN and the per-CU summary of
spatial_corr (mean off-diagonal corr per CU). If |r| > 0.7, treat them
as one channel, weakening Phase 1b's "two channels" claim.
"""
from __future__ import annotations
import json, os, sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "results/IDENTITY_BENCHMARK_2026-05-30"
OUT = RAW / "novel_v2" / "rtn_spatial_correlation.json"
OUT.parent.mkdir(parents=True, exist_ok=True)

def per_cu_spatial_summary(C: np.ndarray) -> np.ndarray:
    """Mean absolute off-diagonal correlation per CU (row)."""
    N = C.shape[0]
    M = np.abs(C).copy()
    np.fill_diagonal(M, np.nan)
    return np.nanmean(M, axis=1)

def pearson(a, b):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if a.size < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float('nan')
    return float(np.corrcoef(a, b)[0,1])

def main():
    out = {"description": "Pre-step: collinearity test RTN vs spatial-corr summary",
           "threshold_r": 0.7,
           "devices": {}}
    rtn_pool, spc_pool = [], []
    for dev in ("ikaros", "daedalus"):
        npz = np.load(RAW / dev / "raw_idle.npz")
        rtn = npz["rtn"].astype(float)
        spc_sum = per_cu_spatial_summary(npz["spatial_corr"].astype(float))
        r = pearson(rtn, spc_sum)
        out["devices"][dev] = {
            "n_cu": int(rtn.size),
            "rtn_mean": float(np.nanmean(rtn)),
            "rtn_std": float(np.nanstd(rtn)),
            "spc_summary_mean": float(np.nanmean(spc_sum)),
            "spc_summary_std": float(np.nanstd(spc_sum)),
            "pearson_r": r,
            "collinear": (abs(r) > 0.7) if np.isfinite(r) else None,
        }
        rtn_pool.append(rtn); spc_pool.append(spc_sum)
    # Naive pool (suffers Simpson's paradox due to device offset)
    rtn_all = np.concatenate(rtn_pool); spc_all = np.concatenate(spc_pool)
    r_pool_naive = pearson(rtn_all, spc_all)
    # Within-device z-scored pool (the right metric)
    def z(x):
        x = np.asarray(x, float); s = np.nanstd(x)
        return (x - np.nanmean(x))/s if s > 0 else x*np.nan
    rtn_z = np.concatenate([z(a) for a in rtn_pool])
    spc_z = np.concatenate([z(a) for a in spc_pool])
    r_pool_within = pearson(rtn_z, spc_z)
    out["pooled"] = {
        "pearson_r_naive_simpsons_paradox": r_pool_naive,
        "pearson_r_within_device_zscored": r_pool_within,
        "collinear": (abs(r_pool_within) > 0.7) if np.isfinite(r_pool_within) else None,
        "note": "Within-device z-score pool is the correct test; the naive pool only reflects device mean offset.",
    }
    # Per-device collinearity (ignoring NaN devices where RTN is degenerate)
    finite_devs = [d for d,v in out["devices"].items()
                   if v["collinear"] is not None]
    per_dev_any = any(out["devices"][d]["collinear"] for d in finite_devs)
    pooled_coll = out["pooled"]["collinear"]
    # Flag the broken-RTN device
    broken = [d for d,v in out["devices"].items() if v["rtn_std"] == 0.0]
    if broken:
        out["data_quality_flag"] = (
            f"RTN signal is degenerate (all zeros) on {broken}. "
            "Cannot test collinearity on that device. "
            "This itself weakens Phase 1b: RTN may not be measurable on every device."
        )
    if pooled_coll or per_dev_any:
        out["verdict"] = "COLLINEAR — Phase 1b two-channel claim is WEAKER; treat as one channel."
    elif broken:
        out["verdict"] = (
            "MIXED — Where RTN exists (daedalus), it is INDEPENDENT of spatial-corr "
            f"(r={out['devices']['daedalus']['pearson_r']:.3f}). "
            "But RTN is degenerate on ikaros, so the 'two channels' claim is "
            "not universally supported across devices."
        )
    else:
        out["verdict"] = "INDEPENDENT — Phase 1b two-channel claim survives this test."
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
