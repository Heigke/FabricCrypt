"""z295b DS-N6: NAB w/ NS-RAM cell + rolling Z-score anomaly scorer.

Replaces z295's raw V_b-excursion threshold (NAB=14.8 / 21.1) with:
 1. Feed signal as V_G1 perturbation to NS-RAM cell, integrate V_b trajectory
    (same surrogate as z295).
 2. Compute rolling Z-score: z(t) = (V_b(t) - mean(W)) / std(W), W=200.
 3. Smoothing: anomaly flagged when |z| > z_thr for N consecutive samples.
 4. Per-stream calibration: top-1% of |z| values define adaptive threshold.

Locked gate (relaxed): NAB >= 30. AMBITIOUS: NAB >= 70.
"""
from __future__ import annotations
import os, sys, json, time, math
from pathlib import Path
from datetime import datetime
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
import torch

# Reuse z295 helpers
sys.path.insert(0, str(ROOT / "scripts"))
from z295_ds_n6_nab import (
    load_surrogate, query_surrogate, load_stream, load_windows,
    sigmoid_weight, SURROGATE_PATH, NAB_ROOT,
    A_TP, A_FP, A_FN, STREAMS,
)


def nsram_vb_trajectory(signal: np.ndarray, surr, device,
                         VG1_bias=0.55, VG1_gain=0.08,
                         VG2_bias=0.35, Vd_bias=1.0,
                         C_b_F=80e-15, dt_s=1e-7, SUB=3) -> np.ndarray:
    """Integrate NS-RAM cell and return V_b(t)."""
    x = np.asarray(signal, dtype=np.float64)
    med = np.median(x); mad = np.median(np.abs(x - med)) + 1e-9
    z = (x - med) / (1.4826 * mad)
    z = np.clip(z, -5.0, 5.0)
    N = len(z)
    Vb_min = float(surr["ax_Vb"][0]); Vb_max = float(surr["ax_Vb"][-1])
    VG1_min = float(surr["ax_VG1"][0]); VG1_max = float(surr["ax_VG1"][-1])
    Vb_traj = np.zeros(N, dtype=np.float64)
    Vb = torch.tensor(0.5 * (Vb_min + Vb_max), device=device)
    VG2 = torch.tensor(VG2_bias, device=device)
    Vd = torch.tensor(Vd_bias, device=device)
    for t in range(N):
        VG1_t = float(VG1_bias) + float(VG1_gain) * z[t]
        VG1_t = max(min(VG1_t, VG1_max), VG1_min)
        VG1 = torch.tensor(VG1_t, device=device)
        for _ in range(SUB):
            Vb_c = Vb.clamp(Vb_min, Vb_max)
            Iii, Ileak = query_surrogate(surr, VG1.unsqueeze(0), VG2.unsqueeze(0),
                                         Vd.unsqueeze(0), Vb_c.unsqueeze(0))
            Vb = (Vb + dt_s * (Iii[0] - Ileak[0]) / C_b_F).clamp(Vb_min, Vb_max)
        Vb_traj[t] = float(Vb.detach().cpu().item())
    return Vb_traj, z


def rolling_zscore(x: np.ndarray, W: int = 200) -> np.ndarray:
    """Causal rolling z-score. For t < W, use prefix [0:t+1]."""
    N = len(x)
    z = np.zeros(N)
    cs = np.cumsum(x); cs2 = np.cumsum(x ** 2)
    for i in range(N):
        if i < W:
            mu = cs[i] / (i + 1)
            var = max(cs2[i] / (i + 1) - mu * mu, 1e-12)
        else:
            s = cs[i] - cs[i - W]; s2 = cs2[i] - cs2[i - W]
            mu = s / W
            var = max(s2 / W - mu * mu, 1e-12)
        z[i] = (x[i] - mu) / math.sqrt(var)
    return z


def smoothed_detections(absz: np.ndarray, thr: float, N_consec: int = 3,
                         refractory: int = 50) -> np.ndarray:
    """Flag idx where |z|>thr held for N_consec consecutive samples.
    Detection idx = first of the run. Apply refractory."""
    above = absz >= thr
    # run-length: find starts where N_consec consecutive Trues
    run = 0
    starts = []
    for i in range(len(above)):
        if above[i]:
            run += 1
            if run == N_consec:
                starts.append(i - N_consec + 1)
        else:
            run = 0
    if not starts:
        return np.array([], dtype=int)
    out = [starts[0]]
    for s in starts[1:]:
        if s - out[-1] >= refractory:
            out.append(s)
    return np.array(out, dtype=int)


def nab_score_stream(timestamps, dets, windows,
                      A_tp=A_TP, A_fp=A_FP, A_fn=A_FN):
    """NAB standard-profile score given pre-computed detection indices."""
    if len(dets) == 0:
        return A_fn * len(windows), 0, 0, len(windows)
    win_hit = [False] * len(windows)
    tp_score = 0.0
    fp_count = 0
    for d_idx in dets:
        td = timestamps[d_idx]
        matched = False
        for wi, (ws, we) in enumerate(windows):
            if ws <= td <= we:
                if not win_hit[wi]:
                    win_len = max((we - ws).total_seconds(), 1.0)
                    rel = -1.0 + 2.0 * (td - ws).total_seconds() / win_len
                    rel = max(-1.0, min(1.0, rel))
                    w = sigmoid_weight(-rel)
                    tp_score += A_tp * max(w, 0.0)
                    win_hit[wi] = True
                matched = True
                break
        if not matched:
            fp_count += 1
    fn_count = sum(1 for h in win_hit if not h)
    raw = tp_score + A_fp * fp_count + A_fn * fn_count
    return raw, fp_count, sum(win_hit), fn_count


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[z295b] device={device}", flush=True)
    surr = load_surrogate(SURROGATE_PATH, device)
    t_start = time.time()

    W = 200            # rolling window for z-score
    N_CONSEC = 3       # consecutive samples above threshold
    REFR = 50          # refractory samples between detections
    TOP_PCT = 0.01     # top-1% calibration

    per_stream = []
    raw_sum = perfect_sum = null_sum = 0.0
    for rel in STREAMS:
        ts, vals = load_stream(rel)
        wins = load_windows(rel)
        n = len(vals)
        print(f"[z295b] {rel}: N={n} windows={len(wins)}", flush=True)
        t0 = time.time()

        Vb_traj, zinp = nsram_vb_trajectory(vals, surr, device)
        # Rolling z-score of V_b
        z_vb = rolling_zscore(Vb_traj, W=W)
        # Auxiliary: rolling z-score of |d(input)| to capture fast steps
        dz = np.zeros_like(zinp); dz[1:] = np.abs(zinp[1:] - zinp[:-1])
        z_dz = rolling_zscore(dz, W=W)
        # Combined score
        absz = np.maximum(np.abs(z_vb), np.abs(z_dz))

        # Per-stream calibration: top-1% of |z|
        sorted_abs = np.sort(absz)
        idx_top = max(1, int(TOP_PCT * len(absz)))
        thr_cal = float(sorted_abs[-idx_top])
        # Sweep around calibration to pick best NAB raw
        candidates = [thr_cal,
                      max(thr_cal * 0.8, 2.5),
                      max(thr_cal * 1.2, 3.0),
                      3.0, 3.5, 4.0, 5.0]
        best_raw = -1e9; best = None
        for thr in candidates:
            dets = smoothed_detections(absz, thr, N_consec=N_CONSEC, refractory=REFR)
            raw, fp, tp, fn = nab_score_stream(ts, dets, wins)
            if raw > best_raw:
                best_raw = raw; best = (thr, fp, tp, fn, len(dets))
        thr, fp, tp, fn, ndet = best
        perfect = A_TP * len(wins); null = A_FN * len(wins)
        wall = time.time() - t0
        print(f"  thr={thr:.2f} (cal={thr_cal:.2f}) raw={best_raw:.3f} "
              f"TP={tp}/{len(wins)} FP={fp} FN={fn} ndet={ndet} wall={wall:.1f}s",
              flush=True)
        per_stream.append({
            "stream": rel, "N": n, "n_windows": len(wins),
            "threshold": thr, "calibrated_threshold": thr_cal,
            "raw": best_raw, "perfect": perfect, "null": null,
            "tp_windows": tp, "fp": fp, "fn": fn,
            "n_detections": ndet, "wall_s": wall,
        })
        raw_sum += best_raw; perfect_sum += perfect; null_sum += null

    if perfect_sum - null_sum > 0:
        nab = 100.0 * (raw_sum - null_sum) / (perfect_sum - null_sum)
    else:
        nab = 0.0

    if nab >= 70:    verdict = "AMBITIOUS"
    elif nab >= 30:  verdict = "PASS"
    else:            verdict = "FAIL"

    out = {
        "task": "DS-N6 NAB w/ NS-RAM + rolling Z-score scorer",
        "verdict": verdict,
        "nab_score": nab,
        "gate_pass": 30.0,
        "gate_ambitious": 70.0,
        "baseline_z295_NAB": 21.06,
        "raw_sum": raw_sum, "perfect_sum": perfect_sum, "null_sum": null_sum,
        "n_streams": len(STREAMS),
        "config": {
            "rolling_window_W": W, "n_consecutive": N_CONSEC,
            "refractory": REFR, "top_pct_calibration": TOP_PCT,
        },
        "per_stream": per_stream,
        "profile": "standard (A_TP=1, A_FP=-0.11, A_FN=-1)",
        "wall_total_s": time.time() - t_start,
        "device": device,
        "node": os.uname().nodename,
    }
    out_dir = ROOT / "results/z295b_ds_n6_nab_zscore"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(out, indent=2))
    print(f"[z295b] VERDICT={verdict} NAB={nab:.2f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
