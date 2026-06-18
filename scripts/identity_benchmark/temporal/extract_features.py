#!/usr/bin/env python3
"""
temporal/extract_features.py — extract Groups T1..T7 temporal features
from a single device's temporal.npz, write features.json.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]


def safe_stat(x, fn, default=float("nan")):
    try:
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        if x.size == 0:
            return default
        return float(fn(x))
    except Exception:
        return default


def derivative(x, t):
    """Central difference dx/dt."""
    x = np.asarray(x, dtype=float)
    t = np.asarray(t, dtype=float)
    if len(x) < 3:
        return np.zeros_like(x)
    dx = np.zeros_like(x)
    dx[1:-1] = (x[2:] - x[:-2]) / np.maximum(t[2:] - t[:-2], 1e-9)
    dx[0] = (x[1] - x[0]) / max(t[1] - t[0], 1e-9)
    dx[-1] = (x[-1] - x[-2]) / max(t[-1] - t[-2], 1e-9)
    return dx


def hysteresis_area(T, P):
    """Signed area of (T,P) loop via shoelace on the closed polyline.

    Returns abs area in K·W units."""
    T = np.asarray(T, dtype=float)
    P = np.asarray(P, dtype=float)
    n = len(T)
    if n < 4:
        return 0.0
    # shoelace
    area = 0.5 * abs(np.sum(T[:-1] * P[1:] - T[1:] * P[:-1]) + T[-1]*P[0] - T[0]*P[-1])
    return float(area)


def damped_sinusoid_fit(t, y):
    """Rough damped-sinusoid metrics on a step-response window.

    Returns dict(ringback_hz, damping, overshoot)."""
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)
    if len(y) < 8:
        return dict(ringback_hz=float("nan"), damping=float("nan"), overshoot=float("nan"))
    y0 = y[0]
    yinf = np.median(y[-max(3, len(y) // 5):])
    span = yinf - y0
    if abs(span) < 1e-9:
        return dict(ringback_hz=float("nan"), damping=float("nan"), overshoot=0.0)
    # overshoot: max deviation above target / span
    over = (np.max(y) - yinf) / abs(span) if span > 0 else (yinf - np.min(y)) / abs(span)
    # ringback: dominant FFT freq of residual
    resid = y - np.interp(t, [t[0], t[-1]], [y0, yinf])
    fs = 1.0 / np.median(np.diff(t))
    if len(resid) >= 16:
        sp = np.abs(np.fft.rfft(resid - resid.mean()))
        fr = np.fft.rfftfreq(len(resid), 1.0 / fs)
        k = int(np.argmax(sp[1:])) + 1
        ringback = float(fr[k])
    else:
        ringback = float("nan")
    # damping: log-decrement
    pos_peaks = np.where((y[1:-1] > y[:-2]) & (y[1:-1] > y[2:]))[0] + 1
    if len(pos_peaks) >= 2:
        dec = np.log(max(abs(y[pos_peaks[0]] - yinf), 1e-9)
                     / max(abs(y[pos_peaks[-1]] - yinf), 1e-9))
        damping = float(dec / (len(pos_peaks) - 1))
    else:
        damping = float("nan")
    return dict(ringback_hz=ringback, damping=damping, overshoot=float(over))


def spectral_features(x, fs):
    x = np.asarray(x, dtype=float) - np.mean(x)
    if len(x) < 64:
        return dict(knee_hz=float("nan"), slope=float("nan"), bandpower=float("nan"))
    sp = np.abs(np.fft.rfft(x)) ** 2
    fr = np.fft.rfftfreq(len(x), 1.0 / fs)
    sp[0] = sp[1] if len(sp) > 1 else sp[0]
    # fit slope on log-log between 0.1 Hz and fs/4
    mask = (fr > 0.1) & (fr < fs / 4.0) & (sp > 0)
    if mask.sum() < 5:
        return dict(knee_hz=float("nan"), slope=float("nan"),
                    bandpower=float(np.sum(sp)))
    lf, ls = np.log10(fr[mask]), np.log10(sp[mask])
    slope, intercept = np.polyfit(lf, ls, 1)
    # crude knee: freq where slope-fit crosses constant-tail (plateau)
    tail = np.median(ls[-max(3, len(ls)//5):])
    line = intercept + slope * lf
    cross = np.where(line <= tail)[0]
    knee = float(10 ** lf[cross[0]]) if len(cross) else float("nan")
    return dict(knee_hz=knee, slope=float(slope),
                bandpower=float(np.sum(sp[mask])))


def coherence(x, y, fs, nperseg=128):
    """Simple Welch-style magnitude-squared coherence at the cross-spectrum peak."""
    x = np.asarray(x, dtype=float) - np.mean(x)
    y = np.asarray(y, dtype=float) - np.mean(y)
    n = min(len(x), len(y))
    if n < 4 * nperseg:
        nperseg = max(32, n // 4)
    if nperseg < 8:
        return dict(coh_peak=float("nan"), coh_freq=float("nan"))
    nseg = n // nperseg
    Sxx = Syy = Sxy = 0.0
    for k in range(nseg):
        xs = x[k*nperseg:(k+1)*nperseg]
        ys = y[k*nperseg:(k+1)*nperseg]
        w = np.hanning(nperseg)
        X = np.fft.rfft(w * xs)
        Y = np.fft.rfft(w * ys)
        Sxx = Sxx + np.abs(X) ** 2
        Syy = Syy + np.abs(Y) ** 2
        Sxy = Sxy + X * np.conj(Y)
    denom = (Sxx * Syy)
    denom[denom == 0] = 1e-30
    coh = np.abs(Sxy) ** 2 / denom
    fr = np.fft.rfftfreq(nperseg, 1.0 / fs)
    k = int(np.argmax(coh[1:])) + 1
    return dict(coh_peak=float(coh[k]), coh_freq=float(fr[k]))


def recurrence_density(x, eps=None):
    """Threshold-density of recurrence plot for embedded signal.

    Embedding dim=2, tau=2."""
    x = np.asarray(x, dtype=float)
    if len(x) < 200:
        return float("nan")
    x = (x - np.mean(x)) / (np.std(x) + 1e-9)
    tau = 2
    Y = np.column_stack([x[:-tau], x[tau:]])
    # subsample for tractability
    if len(Y) > 1000:
        idx = np.random.RandomState(42).choice(len(Y), 1000, replace=False)
        Y = Y[idx]
    if eps is None:
        eps = 0.3
    d = np.sqrt(((Y[:, None, :] - Y[None, :, :]) ** 2).sum(-1))
    return float((d < eps).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    d = np.load(args.npz)
    t = d["t"]; T_apu = d["T_apu_c"]; T_gpu = d["T_gpu_c"]
    P = d["P_gpu_w"]; F_gpu = d["F_gpu_hz"]; F_cpu = d["F_cpu_khz"]
    label = d["load_label"]
    fs = 1.0 / max(np.median(np.diff(t)), 1e-6)

    feats = {"meta": {"n": int(len(t)), "fs_hz": float(fs),
                       "duration_s": float(t[-1] - t[0])}}

    # T1: static baselines
    g1 = {}
    for name, x in [("T_apu", T_apu), ("T_gpu", T_gpu), ("P_gpu", P),
                    ("F_gpu", F_gpu), ("F_cpu", F_cpu)]:
        g1[f"{name}_mean"] = safe_stat(x, np.mean)
        g1[f"{name}_std"] = safe_stat(x, np.std)
        g1[f"{name}_p10"] = safe_stat(x, lambda v: np.percentile(v, 10))
        g1[f"{name}_p90"] = safe_stat(x, lambda v: np.percentile(v, 90))
    feats["T1_static"] = g1

    # T2: derivatives — heating vs cooling rate, accel
    dT_apu = derivative(T_apu, t)
    dT_gpu = derivative(T_gpu, t)
    dP = derivative(P, t)
    ddT_apu = derivative(dT_apu, t)
    g2 = {
        "dT_apu_mean_heat": safe_stat(dT_apu[dT_apu > 0], np.mean),
        "dT_apu_mean_cool": safe_stat(dT_apu[dT_apu < 0], np.mean),
        "dT_apu_asym": (safe_stat(dT_apu[dT_apu > 0], np.mean)
                        + safe_stat(dT_apu[dT_apu < 0], np.mean)),
        "dT_gpu_mean_heat": safe_stat(dT_gpu[dT_gpu > 0], np.mean),
        "dT_gpu_mean_cool": safe_stat(dT_gpu[dT_gpu < 0], np.mean),
        "dP_std": safe_stat(dP, np.std),
        "dP_p95": safe_stat(dP, lambda v: np.percentile(np.abs(v), 95)),
        "d2T_apu_std": safe_stat(ddT_apu, np.std),
        "d2T_apu_max": safe_stat(np.abs(ddT_apu), np.max),
    }
    feats["T2_derivatives"] = g2

    # T3: cross-channel coupling
    # dP/dT during ramp: regress dP on dT where |dT|>noise
    noise_dT = np.std(dT_apu) * 0.5
    mask = np.abs(dT_apu) > noise_dT
    if mask.sum() > 10:
        dPdT_apu = float(np.polyfit(dT_apu[mask], dP[mask], 1)[0])
    else:
        dPdT_apu = float("nan")
    noise_dTg = np.std(dT_gpu) * 0.5
    mask = np.abs(dT_gpu) > noise_dTg
    if mask.sum() > 10:
        dPdT_gpu = float(np.polyfit(dT_gpu[mask], dP[mask], 1)[0])
        dFdP = float(np.polyfit(dP[mask], derivative(F_gpu, t)[mask], 1)[0])
    else:
        dPdT_gpu = float("nan"); dFdP = float("nan")
    # T vs P lag via cross-correlation
    n = min(len(T_apu), len(P))
    a = (T_apu[:n] - np.mean(T_apu[:n])) / (np.std(T_apu[:n]) + 1e-9)
    b = (P[:n] - np.mean(P[:n])) / (np.std(P[:n]) + 1e-9)
    xcorr = np.correlate(a, b, mode="full")
    lags = np.arange(-n + 1, n)
    lag_idx = int(np.argmax(xcorr))
    lag_samples = int(lags[lag_idx])
    lag_s = float(lag_samples / fs)
    g3 = {
        "dP_dT_apu_slope_WperK": dPdT_apu,
        "dP_dT_gpu_slope_WperK": dPdT_gpu,
        "dF_dP_HzperW": dFdP,
        "T_P_xcorr_lag_s": lag_s,
        "T_P_xcorr_peak": float(xcorr[lag_idx] / n),
    }
    feats["T3_cross_coupling"] = g3

    # T4: hysteresis loop area in (T_apu, P)
    area = hysteresis_area(T_apu, P)
    # Per-segment heat-up area vs cool-down area asymmetry
    heat_mask = dT_apu > 0
    cool_mask = dT_apu < 0
    g4 = {
        "loop_area_KW": area,
        "heat_frac": float(heat_mask.mean()),
        "cool_frac": float(cool_mask.mean()),
        "P_at_heat_mean": safe_stat(P[heat_mask], np.mean),
        "P_at_cool_mean": safe_stat(P[cool_mask], np.mean),
        "asymmetry": (safe_stat(P[heat_mask], np.mean)
                      - safe_stat(P[cool_mask], np.mean)),
    }
    feats["T4_hysteresis"] = g4

    # T5: step response on each load transition (label change)
    transitions = np.where(np.diff(label) != 0)[0]
    step_metrics = []
    for ti in transitions:
        win = slice(ti, min(ti + int(fs * 8), len(t)))
        m = damped_sinusoid_fit(t[win], T_apu[win])
        m["label_to"] = int(label[ti + 1]) if ti + 1 < len(label) else -1
        step_metrics.append(m)
    if step_metrics:
        rb = [m["ringback_hz"] for m in step_metrics if np.isfinite(m["ringback_hz"])]
        ov = [m["overshoot"] for m in step_metrics if np.isfinite(m["overshoot"])]
        dp = [m["damping"] for m in step_metrics if np.isfinite(m["damping"])]
        g5 = {
            "ringback_hz_mean": float(np.mean(rb)) if rb else float("nan"),
            "ringback_hz_std": float(np.std(rb)) if rb else float("nan"),
            "overshoot_mean": float(np.mean(ov)) if ov else float("nan"),
            "damping_mean": float(np.mean(dp)) if dp else float("nan"),
            "n_steps": len(step_metrics),
        }
    else:
        g5 = {"n_steps": 0}
    feats["T5_step_response"] = g5

    # T6: spectral
    g6 = {}
    for name, x in [("T_apu", T_apu), ("T_gpu", T_gpu), ("P_gpu", P)]:
        s = spectral_features(x, fs)
        g6[f"{name}_knee_hz"] = s["knee_hz"]
        g6[f"{name}_slope"] = s["slope"]
        g6[f"{name}_bandpower"] = s["bandpower"]
    coh = coherence(T_apu, P, fs)
    g6["coh_TP_peak"] = coh["coh_peak"]
    g6["coh_TP_freq"] = coh["coh_freq"]
    feats["T6_spectral"] = g6

    # T7: phase-space (T_apu, P_gpu)
    # convex hull area
    try:
        from scipy.spatial import ConvexHull
        pts = np.column_stack([T_apu, P])
        # subsample
        if len(pts) > 2000:
            idx = np.random.RandomState(0).choice(len(pts), 2000, replace=False)
            pts = pts[idx]
        hull_area = float(ConvexHull(pts).volume)  # 2D: .volume = area
    except Exception:
        hull_area = float("nan")
    rec = recurrence_density(T_apu)
    # lyapunov (rosenstein toy estimate)
    try:
        x = (T_apu - np.mean(T_apu)) / (np.std(T_apu) + 1e-9)
        m = 3; tau = 2
        emb = np.column_stack([x[i:len(x) - (m - 1 - i) * tau] for i in range(m)])
        from scipy.spatial import cKDTree
        tree = cKDTree(emb)
        d, idx = tree.query(emb, k=2)
        nearest = idx[:, 1]
        # divergence over k steps
        K = 10
        div = []
        for k in range(1, K):
            valid = (np.arange(len(emb)) + k < len(emb)) & (nearest + k < len(emb))
            if valid.sum() < 50:
                continue
            d_t = np.linalg.norm(emb[np.where(valid)[0] + k]
                                 - emb[nearest[valid] + k], axis=1)
            d_t = d_t[d_t > 0]
            if len(d_t):
                div.append(np.mean(np.log(d_t)))
        lyap = float(np.polyfit(np.arange(len(div)), div, 1)[0]) if len(div) > 3 else float("nan")
    except Exception:
        lyap = float("nan")
    feats["T7_phase_space"] = {
        "hull_area": hull_area,
        "recurrence_density": rec,
        "lyapunov_proxy": lyap,
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(feats, indent=2))
    print(f"[OK] {args.out}")


if __name__ == "__main__":
    main()
