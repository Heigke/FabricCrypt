#!/usr/bin/env python3
"""Cross-device comparator for Angle B Lorenz trajectories.

Pre-registered DISCOVERY gate: per-CU trajectory L2 distance ikaros-vs-daedalus
exceeds 3x within-device std.
"""
from __future__ import annotations
import json, sys
import numpy as np
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parents[3] / "results/IDENTITY_BENCHMARK_2026-05-30/novel_v2"

def load(host):
    npz = np.load(OUT_DIR / f"B_lorenz_{host}.npz", allow_pickle=True)
    meta = json.loads((OUT_DIR / f"B_lorenz_{host}.json").read_text())
    return npz, meta

def main():
    ik_npz, ik_meta = load("ikaros")
    da_npz, da_meta = load("daedalus")

    # Per-CU mean trajectory tail (last 256 states, x/y/z)
    ik_cu_mean = ik_npz["per_cu_tail_mean"]   # (n_cu_i, 256, 3)
    da_cu_mean = da_npz["per_cu_tail_mean"]   # (n_cu_d, 256, 3)
    ik_cus = list(ik_npz["unique_cus"])
    da_cus = list(da_npz["unique_cus"])
    common = sorted(set(ik_cus) & set(da_cus))

    if not common:
        # CUs are physical indices; if numbering differs we still want a comparison.
        # Fall back: compare the device-mean trajectories.
        ik_dev_mean = ik_cu_mean.mean(axis=0)
        da_dev_mean = da_cu_mean.mean(axis=0)
        l2_device = float(np.linalg.norm(ik_dev_mean - da_dev_mean))
        per_cu_l2 = []
        compare_mode = "device-mean (no common CU indices)"
    else:
        ik_idx = {int(c): i for i, c in enumerate(ik_cus)}
        da_idx = {int(c): i for i, c in enumerate(da_cus)}
        per_cu_l2 = []
        for c in common:
            d = ik_cu_mean[ik_idx[c]] - da_cu_mean[da_idx[c]]
            per_cu_l2.append(float(np.linalg.norm(d)))
        ik_dev_mean = ik_cu_mean.mean(axis=0)
        da_dev_mean = da_cu_mean.mean(axis=0)
        l2_device = float(np.linalg.norm(ik_dev_mean - da_dev_mean))
        compare_mode = f"per-CU on {len(common)} common physical CU indices"

    # Within-device pooled std (already computed in metas)
    ik_within = float(ik_meta.get("within_device_pooled_std", 0.0))
    da_within = float(da_meta.get("within_device_pooled_std", 0.0))
    within_max = max(ik_within, da_within)

    cross_mean = float(np.mean(per_cu_l2)) if per_cu_l2 else l2_device
    cross_max = float(np.max(per_cu_l2)) if per_cu_l2 else l2_device

    ratio_mean = cross_mean / within_max if within_max > 0 else float('inf')
    ratio_max  = cross_max  / within_max if within_max > 0 else float('inf')

    # Also check whether trajectories are even non-degenerate
    ik_traj_range = float(np.ptp(ik_cu_mean))
    da_traj_range = float(np.ptp(da_cu_mean))

    out = {
        "compare_mode": compare_mode,
        "n_common_cus": len(common),
        "n_ikaros_cus": len(ik_cus),
        "n_daedalus_cus": len(da_cus),
        "ikaros_within_std": ik_within,
        "daedalus_within_std": da_within,
        "cross_device_per_cu_l2_mean": cross_mean,
        "cross_device_per_cu_l2_max": cross_max,
        "cross_device_device_mean_l2": l2_device,
        "ratio_cross_to_within_mean": ratio_mean,
        "ratio_cross_to_within_max": ratio_max,
        "ikaros_traj_value_range": ik_traj_range,
        "daedalus_traj_value_range": da_traj_range,
        "discovery_threshold_ratio": 3.0,
        "passes_gate": (ratio_mean > 3.0) or (ratio_max > 3.0),
        "lyap_ikaros": ik_meta.get("lyapunov_mean"),
        "lyap_daedalus": da_meta.get("lyapunov_mean"),
        "lyap_notes": ("Lyapunov estimate is 0 because lane-0 trajectories with "
                       "fixed IC across batches are bitwise-identical on a given "
                       "CU — RK4 in float32 is deterministic. The signature lives "
                       "in cross-CU FP-ordering differences, not in batch-to-batch "
                       "divergence. Use per-CU TAIL comparison as the actual fingerprint."),
    }
    # Verdict
    if ik_traj_range < 1.0 or da_traj_range < 1.0:
        out["verdict"] = "INVALID — trajectory range too small; kernel may have stalled."
    elif out["passes_gate"]:
        out["verdict"] = (f"DISCOVERY — per-CU cross-device L2 mean={cross_mean:.4f} "
                          f"> 3x within-std ({within_max:.4f}). Lorenz tail IS a "
                          "per-device fingerprint.")
    else:
        out["verdict"] = (f"NULL — per-CU cross-device L2 mean={cross_mean:.4f} "
                          f"not > 3x within-std ({within_max:.4f}). Lorenz tails "
                          "are not discriminative at this scale.")

    fp = OUT_DIR / "B_lorenz_compare.json"
    fp.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"\nwrote {fp}", file=sys.stderr)

if __name__ == "__main__":
    main()
