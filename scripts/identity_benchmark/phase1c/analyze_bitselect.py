#!/usr/bin/env python3
"""analyze_bitselect.py — bit-selection re-analysis for Phase 1c PUFs.

For Probe A (FMA-LSB stream) and Probe B (RO-pair winner stream):
  1. Identify stable bits per device (intra-Hamming < 1%).
  2. Compute uniqueness (inter-Hamming on retained bits).
  3. Compute reliability (1 - intra-HD on retained).
  4. Compute uniformity (P(bit=1) ≈ 0.5?).

For Probe C (per-DPM flip vectors) and Probe D (transient feature vectors):
  Mahalanobis distance between devices.

Usage:
    python analyze_bitselect.py --ikaros-dir <...> --daedalus-dir <...> --out <report.json>
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np


def load_probeA_bin(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.uint32)
    if raw.size < 4 or raw[0] != 0x4C445331:
        return None
    reps, wpr = int(raw[1]), int(raw[2])
    pl = raw[4:].reshape(reps, wpr)
    fma = pl[:, 256:]  # (reps, 256) uint32 — only low 16 bits populated
    # Convert each value to 16-bit array.
    bits = np.unpackbits(fma.astype('<u2').view(np.uint8).reshape(reps, -1, 2)[..., ::-1].reshape(reps, -1),
                         axis=1, bitorder='big')
    # Above is a bit fiddly; simpler:
    bits = np.zeros((reps, fma.shape[1] * 16), dtype=np.uint8)
    for b in range(16):
        bits[:, b::16] = (fma >> b) & 1
    return bits  # (reps, 256*16) = (reps, 4096)


def load_probeB_bin(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.uint32)
    if raw.size < 4 or raw[0] != 0x524F5031:
        return None
    races, fields = int(raw[1]), int(raw[2])
    pl = raw[4:].reshape(races, fields)
    winners = pl[:, 0]  # 1 or 2 (or 0 if tie/none)
    # Treat as 1-bit/race: 1 if block0 won, 0 if block1 won, drop ties.
    mask = (winners == 1) | (winners == 2)
    bits = (winners[mask] == 1).astype(np.uint8)
    # Reshape to (n_chunks, chunk_size) for HD analysis: chunks of 256 races
    cs = 256
    n = (bits.size // cs) * cs
    bits = bits[:n].reshape(-1, cs)
    return bits  # (n_chunks, cs)


def stable_bits(bits: np.ndarray, intra_threshold: float = 0.01):
    """Return (mask, prob1). bits = (reps, L). mask True for bits with
    intra-HD (P(bit flips) < intra_threshold)."""
    p1 = bits.mean(axis=0)
    # bit is stable-1 if p1 > 1 - intra_threshold; stable-0 if p1 < intra_threshold
    mask = (p1 > 1 - intra_threshold) | (p1 < intra_threshold)
    return mask, p1


def analyze_bit_stream(ika: np.ndarray, dae: np.ndarray, name: str) -> dict:
    """Return per-stream PUF metrics."""
    if ika is None or dae is None:
        return {"name": name, "ok": False, "err": "missing data"}
    # Align reps
    n = min(ika.shape[0], dae.shape[0])
    ika = ika[:n]; dae = dae[:n]
    L = ika.shape[1]
    mask_i, p1_i = stable_bits(ika)
    mask_d, p1_d = stable_bits(dae)
    intersection = mask_i & mask_d
    nbits_kept = int(intersection.sum())
    if nbits_kept == 0:
        return {
            "name": name, "ok": True, "n_reps": int(n), "L": int(L),
            "stable_bits_ikaros": int(mask_i.sum()),
            "stable_bits_daedalus": int(mask_d.sum()),
            "stable_bits_intersection": 0,
            "verdict": "NULL — no jointly stable bits",
        }
    # Reference values per device (majority bit) on retained
    ref_i = (p1_i[intersection] > 0.5).astype(np.uint8)
    ref_d = (p1_d[intersection] > 0.5).astype(np.uint8)
    inter_hd = int((ref_i != ref_d).sum())
    uniqueness = inter_hd / nbits_kept
    # Intra HD on retained: avg fraction of reps that disagree with reference
    sub_i = ika[:, intersection]
    sub_d = dae[:, intersection]
    intra_i = float((sub_i != ref_i[None, :]).mean())
    intra_d = float((sub_d != ref_d[None, :]).mean())
    reliability = 1.0 - 0.5 * (intra_i + intra_d)
    uniformity_i = float(ref_i.mean())
    uniformity_d = float(ref_d.mean())
    # Bit-aliasing — fraction of bits where both devices land on the same value
    bit_aliasing = float((ref_i == ref_d).mean())
    return {
        "name": name, "ok": True,
        "n_reps": int(n), "L": int(L),
        "stable_bits_ikaros":   int(mask_i.sum()),
        "stable_bits_daedalus": int(mask_d.sum()),
        "stable_bits_intersection": nbits_kept,
        "inter_hd": inter_hd,
        "uniqueness": uniqueness,
        "reliability": reliability,
        "uniformity_ikaros":   uniformity_i,
        "uniformity_daedalus": uniformity_d,
        "bit_aliasing": bit_aliasing,
        "verdict": _verdict(uniqueness, reliability, nbits_kept),
    }


def _verdict(uniq, rel, n):
    if n == 0:
        return "NULL"
    if uniq >= 0.40 and rel >= 0.95:
        return f"DISCOVERY (uniq={uniq:.3f} >= 0.40, rel={rel:.3f} >= 0.95, n={n})"
    if uniq >= 0.25 and rel >= 0.90:
        return f"PROMISING (uniq={uniq:.3f}, rel={rel:.3f}, n={n})"
    if uniq < 0.05:
        return f"NULL (uniq={uniq:.3f} < 0.05)"
    return f"WEAK (uniq={uniq:.3f}, rel={rel:.3f}, n={n})"


def mahalanobis(x: np.ndarray, y_mean: np.ndarray, y_cov_inv: np.ndarray) -> float:
    d = x - y_mean
    return float(np.sqrt(d @ y_cov_inv @ d))


def analyze_continuous(features_i: list, features_d: list, name: str,
                       keys: list) -> dict:
    if not features_i or not features_d:
        return {"name": name, "ok": False, "err": "missing features"}
    def to_mat(lst):
        rows = []
        for f in lst:
            row = []
            for k in keys:
                v = f.get(k, np.nan)
                row.append(v if v is not None and np.isfinite(v) else np.nan)
            rows.append(row)
        return np.array(rows)
    X_i = to_mat(features_i); X_d = to_mat(features_d)
    # drop rows with nan
    X_i = X_i[~np.isnan(X_i).any(axis=1)]
    X_d = X_d[~np.isnan(X_d).any(axis=1)]
    if X_i.shape[0] < 3 or X_d.shape[0] < 3:
        return {"name": name, "ok": False, "err": f"too few clean rows i={X_i.shape[0]} d={X_d.shape[0]}"}
    mu_i, mu_d = X_i.mean(0), X_d.mean(0)
    # Pool covariance
    cov = 0.5 * (np.cov(X_i.T) + np.cov(X_d.T))
    # Ridge
    cov += 1e-6 * np.eye(cov.shape[0])
    try:
        cov_inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        return {"name": name, "ok": False, "err": "singular cov"}
    md_centroids = float(np.sqrt((mu_i - mu_d) @ cov_inv @ (mu_i - mu_d)))
    # Intra MD: mean MD of each device's points to its own centroid
    intra_i = float(np.mean([mahalanobis(x, mu_i, cov_inv) for x in X_i]))
    intra_d = float(np.mean([mahalanobis(x, mu_d, cov_inv) for x in X_d]))
    ratio = md_centroids / (0.5 * (intra_i + intra_d) + 1e-9)
    return {
        "name": name, "ok": True,
        "keys": keys, "n_i": int(X_i.shape[0]), "n_d": int(X_d.shape[0]),
        "mu_ikaros": mu_i.tolist(), "mu_daedalus": mu_d.tolist(),
        "md_centroids": md_centroids,
        "intra_md_ikaros": intra_i, "intra_md_daedalus": intra_d,
        "separation_ratio": ratio,
        "verdict": ("DISCOVERY (ratio>=3)" if ratio >= 3 else
                    "PROMISING (ratio>=2)" if ratio >= 2 else
                    "WEAK (ratio>=1)" if ratio >= 1 else "NULL"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ikaros-dir", type=Path, required=True)
    ap.add_argument("--daedalus-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    out = {"ikaros_dir": str(args.ikaros_dir),
           "daedalus_dir": str(args.daedalus_dir), "results": {}}

    # Probe A
    A_i = load_probeA_bin(args.ikaros_dir / "probeA.bin")
    A_d = load_probeA_bin(args.daedalus_dir / "probeA.bin")
    out["results"]["probeA_fma_bits"] = analyze_bit_stream(A_i, A_d, "probeA_FMA")

    # Probe B
    B_i = load_probeB_bin(args.ikaros_dir / "probeB.bin")
    B_d = load_probeB_bin(args.daedalus_dir / "probeB.bin")
    out["results"]["probeB_ro_winner_bits"] = analyze_bit_stream(B_i, B_d, "probeB_RO")

    # Probe C (continuous: flip counts per DPM)
    try:
        cI = json.loads((args.ikaros_dir / "probeC_results.json").read_text())
        cD = json.loads((args.daedalus_dir / "probeC_results.json").read_text())
        # Build "feature vector" per device = (low_mean, auto_mean, high_mean) on FMA lanes
        keys = []
        feat_i = []; feat_d = []
        for lvl in ("low", "auto", "high"):
            if cI.get("per_level", {}).get(lvl, {}).get("ok"):
                feat_i.append({"x": cI["per_level"][lvl]["fma_mean"][:32]})
                feat_d.append({"x": cD["per_level"][lvl]["fma_mean"][:32]})
        # Flatten to per-DPM rows treated as N samples
        # (Actually only 1 sample per DPM here; use flip-count comparison instead.)
        out["results"]["probeC"] = {
            "name": "probeC",
            "flip_count_ikaros":   cI.get("flip_count_low_vs_high"),
            "flip_count_daedalus": cD.get("flip_count_low_vs_high"),
            "flip_lanes_ikaros":   cI.get("flip_lanes_low_vs_high"),
            "flip_lanes_daedalus": cD.get("flip_lanes_low_vs_high"),
            "flip_pattern_jaccard": (
                _jaccard(cI.get("flip_lanes_low_vs_high"),
                         cD.get("flip_lanes_low_vs_high"))
                if cI.get("flip_lanes_low_vs_high") and cD.get("flip_lanes_low_vs_high")
                else None
            ),
        }
    except FileNotFoundError as e:
        out["results"]["probeC"] = {"ok": False, "err": str(e)}

    # Probe D (continuous transient features)
    try:
        dI = json.loads((args.ikaros_dir / "probeD_results.json").read_text())
        dD = json.loads((args.daedalus_dir / "probeD_results.json").read_text())
        feature_keys = ["overshoot", "settle_time", "ring_freq", "apu_rise", "tail_mean"]
        out["results"]["probeD"] = analyze_continuous(
            dI.get("features", []), dD.get("features", []),
            "probeD", feature_keys)
    except FileNotFoundError as e:
        out["results"]["probeD"] = {"ok": False, "err": str(e)}

    # Top-level verdict
    verdicts = []
    for k, v in out["results"].items():
        verdicts.append({"probe": k, "verdict": v.get("verdict") if isinstance(v, dict) else str(v)})
    out["summary"] = verdicts
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, default=lambda x: float(x) if hasattr(x, '__float__') else None))
    print(f"[analyze] wrote {args.out}")
    for v in verdicts:
        print(f"  {v['probe']}: {v['verdict']}")


def _jaccard(a, b):
    if a is None or b is None:
        return None
    a = np.array(a); b = np.array(b)
    n = min(a.size, b.size)
    a = a[:n].astype(bool); b = b[:n].astype(bool)
    union = (a | b).sum()
    if union == 0:
        return 0.0
    return float((a & b).sum()) / float(union)


if __name__ == "__main__":
    main()
