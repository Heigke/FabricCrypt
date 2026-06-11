#!/usr/bin/env python3
"""02_compare_signatures.py — Phase 1 PUF gate evaluator.

Loads ikaros + daedalus signatures (+ raw .npz), computes:
  - intra-Hamming distance per device (across reps, via per-rep bit votes)
  - inter-Hamming distance (device-vs-device on stable signature bits)
  - process-stat channel divergence (KL on knee distrib, spatial-corr MSE,
    RTN-rate KL)
  - noise-control channel KL (PERF histogram, must be small)
Writes verdict markdown to research_plan/IDENTITY_BENCHMARK_2026-05-30_PHASE1.md.

Usage: python 02_compare_signatures.py [--results-root RES]
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent


def hamming(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a).astype(np.uint8).ravel()
    b = np.asarray(b).astype(np.uint8).ravel()
    n = min(a.size, b.size)
    if n == 0:
        return float("nan")
    return float(np.mean(a[:n] != b[:n]))


def kl_div_hist(p_counts, q_counts, eps=1e-9) -> float:
    p = np.asarray(p_counts, dtype=np.float64) + eps
    q = np.asarray(q_counts, dtype=np.float64) + eps
    p = p / p.sum()
    q = q / q.sum()
    n = min(len(p), len(q))
    return float(np.sum(p[:n] * np.log(p[:n] / q[:n])))


def per_rep_bit_votes_from_raw(raw: np.lib.npyio.NpzFile):
    """Rebuild per-rep bit vector matching compute_signature() logic so we
    can compute true intra-Hamming distance (mean per-rep distance from
    final majority signature)."""
    samples = raw["samples"]            # (R, W, 8)
    cu_ts_cyc = raw["cu_ts_cyc"]        # (n_cu, R)
    cu_ts_dot = raw["cu_ts_dot"]
    cu_ts_perf = raw["cu_ts_perf"]
    sig_bits = raw["sig_bits"]          # (n_cu*8,)
    n_cu, n_reps = cu_ts_cyc.shape
    n_bits_per_cu = 8
    old0 = samples[:, :, 4]
    cu_global = samples[:, :, 6]
    cu_ids = []
    seen = set()
    # match compute_signature order: np.unique on cu_global excluding 0xFFFFFFFF
    uq = np.unique(cu_global)
    cu_ids = uq[uq != 0xFFFFFFFF]
    assert len(cu_ids) == n_cu, f"cu_ids mismatch {len(cu_ids)} vs {n_cu}"

    bit_votes = np.full((n_reps, n_cu * n_bits_per_cu), -1, dtype=np.int8)
    for ci, cu in enumerate(cu_ids):
        ref_cyc = np.nanmedian(cu_ts_cyc[ci])
        for r in range(n_reps):
            c = cu_ts_cyc[ci, r]
            d = cu_ts_dot[ci, r]
            p = cu_ts_perf[ci, r]
            o_arr = old0[r][cu_global[r] == cu]
            if not np.isfinite(c):
                continue
            b0 = int(c > ref_cyc)
            b1 = (int(c) >> 1) & 1
            b2 = (int(c) >> 3) & 1
            b3 = (int(c) >> 5) & 1
            db = int(np.float32(d).view(np.uint32))
            b4 = (db >> 0) & 1
            b5 = (db >> 1) & 1
            b6 = bin(int(p) & 0xFFFFFFFF).count("1") & 1
            b7 = bin(int(o_arr[0]) & 0xFF).count("1") & 1 if o_arr.size > 0 else 0
            base = ci * n_bits_per_cu
            bit_votes[r, base+0:base+8] = [b0, b1, b2, b3, b4, b5, b6, b7]
    return bit_votes, sig_bits.astype(np.uint8)


def intra_hd(bit_votes: np.ndarray, sig: np.ndarray) -> tuple:
    """Mean per-rep Hamming distance to final majority signature."""
    dists = []
    for r in range(bit_votes.shape[0]):
        v = bit_votes[r]
        keep = v >= 0
        if not keep.any():
            continue
        dists.append(float(np.mean(v[keep] != sig[keep])))
    if not dists:
        return float("nan"), float("nan"), float("nan")
    return float(np.mean(dists)), float(np.min(dists)), float(np.max(dists))


def load_device(results_root: Path, name: str):
    sig_path = results_root / name / "signature.json"
    sig = json.load(open(sig_path))
    # pick first regime (we ran 'idle')
    regime_key = list(sig["regimes"].keys())[0]
    reg = sig["regimes"][regime_key]
    npz_path = Path(reg["raw_npz"])
    if not npz_path.exists():
        # fix path when copied from remote
        npz_path = results_root / name / npz_path.name
    raw = np.load(npz_path)
    return sig, reg, raw, npz_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", default=str(
        REPO_ROOT / "results" / "IDENTITY_BENCHMARK_2026-05-30"))
    ap.add_argument("--report-out", default=str(
        REPO_ROOT / "research_plan" / "IDENTITY_BENCHMARK_2026-05-30_PHASE1.md"))
    args = ap.parse_args()

    rroot = Path(args.results_root)
    sig_a, reg_a, raw_a, npz_a = load_device(rroot, "ikaros")
    sig_b, reg_b, raw_b, npz_b = load_device(rroot, "daedalus")

    bv_a, S_a = per_rep_bit_votes_from_raw(raw_a)
    bv_b, S_b = per_rep_bit_votes_from_raw(raw_b)

    intra_a_mean, intra_a_min, intra_a_max = intra_hd(bv_a, S_a)
    intra_b_mean, intra_b_min, intra_b_max = intra_hd(bv_b, S_b)
    intra_avg = 0.5 * (intra_a_mean + intra_b_mean)

    # Inter-HD on stable bits — align by min length
    inter = hamming(S_a, S_b)

    # ---- process-stat channel ----
    knees_a = np.array(reg_a["process_stat"]["knee_slopes"], dtype=np.float64)
    knees_b = np.array(reg_b["process_stat"]["knee_slopes"], dtype=np.float64)
    knees_a = knees_a[np.isfinite(knees_a)]
    knees_b = knees_b[np.isfinite(knees_b)]
    rtn_a = np.array(reg_a["process_stat"]["rtn_transition_rate_per_cu"])
    rtn_b = np.array(reg_b["process_stat"]["rtn_transition_rate_per_cu"])
    sc_a = reg_a["process_stat"]["spatial_corr_mean_abs"]
    sc_b = reg_b["process_stat"]["spatial_corr_mean_abs"]

    # histogram-based KL on knee distributions (shared bin edges)
    knee_lo = min(knees_a.min() if knees_a.size else 0,
                  knees_b.min() if knees_b.size else 0)
    knee_hi = max(knees_a.max() if knees_a.size else 1,
                  knees_b.max() if knees_b.size else 1)
    knee_bins = np.linspace(knee_lo, knee_hi, 16)
    ha, _ = np.histogram(knees_a, bins=knee_bins)
    hb, _ = np.histogram(knees_b, bins=knee_bins)
    kl_knee = kl_div_hist(ha, hb)

    rtn_bins = np.linspace(0, max(rtn_a.max(), rtn_b.max(), 1e-3), 16)
    ra, _ = np.histogram(rtn_a, bins=rtn_bins)
    rb, _ = np.histogram(rtn_b, bins=rtn_bins)
    kl_rtn = kl_div_hist(ra, rb)

    spatial_mse = float((sc_a - sc_b) ** 2)

    # ---- noise control ----
    perf_a = np.array(reg_a["noise_control"]["perf_hist"])
    perf_b = np.array(reg_b["noise_control"]["perf_hist"])
    kl_perf = kl_div_hist(perf_a, perf_b)

    # ---- gate evaluation ----
    DISCOVERY = (intra_avg <= 0.10) and (inter >= 0.40)
    KILL = inter <= intra_avg
    # AMBITIOUS: process-stat also separates strongly (heuristic threshold)
    AMBITIOUS = DISCOVERY and (kl_knee > 0.5 or kl_rtn > 0.5 or spatial_mse > 0.01)

    verdict = (
        "DISCOVERY+AMBITIOUS" if AMBITIOUS else
        "DISCOVERY" if DISCOVERY else
        "KILL" if KILL else "NULL"
    )

    # ---- write report ----
    lines = []
    L = lines.append
    L(f"# Identity Benchmark — Phase 1 Verdict")
    L(f"")
    L(f"Date: 2026-05-30 · Devices: ikaros vs daedalus (twin HP Z2 G1a, gfx1151)")
    L(f"")
    L(f"## Verdict: **{verdict}**")
    L(f"")
    L(f"Gates:")
    L(f"- DISCOVERY (intra ≤ 0.10 AND inter ≥ 0.40): **{DISCOVERY}**")
    L(f"- AMBITIOUS (process-stat also separates): **{AMBITIOUS}**")
    L(f"- KILL (inter ≤ intra): **{KILL}**")
    L(f"")
    L(f"## Stable-bit channel")
    L(f"")
    L(f"| metric | ikaros | daedalus |")
    L(f"|---|---|---|")
    L(f"| n_cu | {reg_a['n_cu']} | {reg_b['n_cu']} |")
    L(f"| signature length (bits) | {len(S_a)} | {len(S_b)} |")
    L(f"| intra-HD mean | {intra_a_mean:.4f} | {intra_b_mean:.4f} |")
    L(f"| intra-HD min  | {intra_a_min:.4f} | {intra_b_min:.4f} |")
    L(f"| intra-HD max  | {intra_a_max:.4f} | {intra_b_max:.4f} |")
    L(f"| bit_stability_mean (sig.json) | {reg_a['bit_stability_mean']:.4f} | {reg_b['bit_stability_mean']:.4f} |")
    L(f"")
    L(f"**Cross-device:**")
    L(f"- Inter-HD (stable channel) = **{inter:.4f}** (compared against intra={intra_avg:.4f})")
    L(f"")
    L(f"## Process-stat channel")
    L(f"")
    L(f"| metric | ikaros | daedalus |")
    L(f"|---|---|---|")
    L(f"| knee_slope mean | {knees_a.mean() if knees_a.size else float('nan'):.4f} | {knees_b.mean() if knees_b.size else float('nan'):.4f} |")
    L(f"| knee_slope std  | {knees_a.std() if knees_a.size else float('nan'):.4f} | {knees_b.std() if knees_b.size else float('nan'):.4f} |")
    L(f"| RTN rate mean   | {rtn_a.mean():.4f} | {rtn_b.mean():.4f} |")
    L(f"| spatial_corr_mean_abs | {sc_a:.4f} | {sc_b:.4f} |")
    L(f"")
    L(f"- KL(knee_slope distribution) = {kl_knee:.4f}")
    L(f"- KL(RTN rate distribution)   = {kl_rtn:.4f}")
    L(f"- spatial-corr MSE            = {spatial_mse:.4f}")
    L(f"")
    L(f"## Noise control (PERF_SNAPSHOT)")
    L(f"")
    L(f"| metric | ikaros | daedalus |")
    L(f"|---|---|---|")
    L(f"| perf mean | {reg_a['noise_control']['perf_mean']:.2f} | {reg_b['noise_control']['perf_mean']:.2f} |")
    L(f"| perf std  | {reg_a['noise_control']['perf_std']:.2f} | {reg_b['noise_control']['perf_std']:.2f} |")
    L(f"")
    L(f"- KL(PERF hist) = **{kl_perf:.4f}** (expected small: pure-noise control)")
    L(f"")
    L(f"## Raw data paths")
    L(f"- ikaros   raw: `{npz_a}`")
    L(f"- daedalus raw: `{npz_b}`")
    L(f"- ikaros   sig: `{rroot / 'ikaros' / 'signature.json'}`")
    L(f"- daedalus sig: `{rroot / 'daedalus' / 'signature.json'}`")
    L(f"")
    L(f"## Honest interpretation")
    L(f"")
    if DISCOVERY:
        L(f"Inter-HD ({inter:.3f}) exceeds intra-HD ({intra_avg:.3f}) on the stable-bit "
          f"channel — twin devices distinguishable at PUF level. ")
    elif KILL:
        L(f"KILL: inter-HD ({inter:.3f}) ≤ intra-HD ({intra_avg:.3f}). Stable channel "
          f"failed to separate twins. ")
    else:
        L(f"NULL: inter-HD ({inter:.3f}) > intra-HD ({intra_avg:.3f}) but DISCOVERY gate "
          f"(intra ≤ 0.10 AND inter ≥ 0.40) not met. ")
    L(f"")
    L(f"Confounds to acknowledge:")
    L(f"- Both runs were single 'idle' regime; cross-temperature stability NOT tested.")
    L(f"- Devices in different rooms / chassis at different ambient — inter-HD may "
      f"include temperature/PCIe drift, not pure silicon variance.")
    L(f"- PERF_SNAPSHOT KL ({kl_perf:.3f}) is the null: large value = platform drift, "
      f"small value = pure-noise truly fungible.")

    report_path = Path(args.report_out)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n")
    print(f"[compare] verdict: {verdict}")
    print(f"[compare] intra_a={intra_a_mean:.4f} intra_b={intra_b_mean:.4f} inter={inter:.4f}")
    print(f"[compare] kl_knee={kl_knee:.4f} kl_rtn={kl_rtn:.4f} kl_perf={kl_perf:.4f}")
    print(f"[compare] wrote {report_path}")


if __name__ == "__main__":
    main()
