"""Embodiment Phase 8 — Dynamic Feature Extractor.

Loads a *_rich.npz produced by rich_substrate.py and produces a *_features.npz
containing a much richer descriptor of substrate dynamics.

Per channel (~120 raw channels) we compute the following SCALAR summaries
(over the whole 5-min window):

  Time-domain stats        : mean, std, min, max, IQR, skew, kurt
  Derivatives              : |d1|.mean, |d1|.std, |d2|.mean (jerk-1), |d3|.mean
  Multi-scale stats        : std at windows [0.1s, 1s, 10s] of the
                             rolling average of the signal
  Spectral                 : 5 bandpower bins, 1/f slope, spectral entropy
  Burst/RTN                : Fano factor (var/mean of binned counts above 1-sigma),
                             #large-jumps per second
  Phase-space (T, dT/dt)   : trajectory area (convex-hull proxy via PCA box)

Plus the following CROSS-CHANNEL pair features for a curated set of "physics"
pairs (thermal vs power, freq vs power, freq vs temp, etc.):
  - Pearson r at lag 0
  - Lag of max |r| (proxy for thermal RC time)
  - Peak |r|
  - Coherent power at LF band

Pair-set is auto-built: for every (temp_chan, power_chan), (temp_chan,
freq_chan), (power_chan, freq_chan) (capped at 60 pairs) we emit 4 features.

The output also includes a per-channel *time series of mid-resolution features*
suitable for use as input to the A/B/C/D ablation:

  features_ts : (T', F') float32   — F' channels of rolling derivative+spectral
                                     features at ~10 Hz resolution
  feature_names_ts : list[str]

  scalar_features : (S,) float32   — S scalar descriptors for whole-window stats
  scalar_names    : list[str]

Usage:
    python dynamic_features.py --in ikaros_rich.npz --out ikaros_features.npz
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

REPO = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
OUT_DIR = REPO / "results/IDENTITY_BENCHMARK_2026-05-30/embodiment8"


# ---------------------------------------------------------------------------
def _safe_z(x):
    s = x.std()
    return (x - x.mean()) / s if s > 1e-12 else np.zeros_like(x)


def per_channel_scalars(x, hz):
    """Return list of (name_suffix, value) for one channel."""
    out = []
    mean = float(x.mean()); std = float(x.std())
    out += [("mean", mean), ("std", std),
            ("min", float(x.min())), ("max", float(x.max())),
            ("iqr", float(np.percentile(x, 75) - np.percentile(x, 25))),
            ("range", float(x.max() - x.min()))]
    # Higher moments (on zscored)
    z = _safe_z(x)
    out += [("skew", float((z**3).mean())),
            ("kurt", float((z**4).mean()))]
    # Derivatives
    d1 = np.diff(x, n=1)
    d2 = np.diff(x, n=2)
    d3 = np.diff(x, n=3)
    out += [("absd1_mean", float(np.abs(d1).mean())),
            ("absd1_std",  float(np.abs(d1).std())),
            ("absd2_mean", float(np.abs(d2).mean())),
            ("absd3_mean", float(np.abs(d3).mean()))]
    # Multi-scale (rolling avg windows in samples)
    for tag, sec in (("ms100", 0.1), ("s1", 1.0), ("s10", 10.0)):
        w = max(2, int(sec * hz))
        if w < len(x):
            k = np.ones(w, dtype=np.float32) / w
            roll = np.convolve(x, k, mode="valid")
            out.append((f"std_{tag}", float(roll.std())))
            d1_roll = np.diff(roll)
            out.append((f"absd1_{tag}_mean", float(np.abs(d1_roll).mean())))
        else:
            out.append((f"std_{tag}", 0.0))
            out.append((f"absd1_{tag}_mean", 0.0))
    # Spectral: 5 logarithmic bands of |FFT|^2
    if std > 1e-12 and len(x) >= 16:
        n = len(x)
        f = np.fft.rfftfreq(n, d=1.0/hz)
        S = np.abs(np.fft.rfft(z))**2
        # Bands: [0, 0.1), [0.1, 1), [1, 5), [5, 15), [15, hz/2]
        edges = [0.0, 0.1, 1.0, 5.0, 15.0, hz/2 + 1e-6]
        for k in range(5):
            sel = (f >= edges[k]) & (f < edges[k+1])
            out.append((f"bp{k}", float(S[sel].sum())))
        # 1/f slope: log-log linear fit on f in [0.1, hz/4]
        sel = (f > 0.05) & (f < hz/4)
        if sel.sum() > 8:
            lf = np.log(f[sel]); ls = np.log(S[sel] + 1e-12)
            slope = float(np.polyfit(lf, ls, 1)[0])
        else:
            slope = 0.0
        out.append(("oneoverf_slope", slope))
        # spectral entropy
        p = S / (S.sum() + 1e-12)
        out.append(("spec_entropy", float(-(p * np.log(p + 1e-12)).sum())))
    else:
        for k in range(5):
            out.append((f"bp{k}", 0.0))
        out += [("oneoverf_slope", 0.0), ("spec_entropy", 0.0)]
    # Burst / Fano
    if std > 1e-12:
        thr = mean + std
        binned = (x > thr).astype(np.int32)
        bsec = max(2, int(hz))  # 1-sec bins
        nbins = len(binned) // bsec
        if nbins > 1:
            counts = binned[:nbins*bsec].reshape(nbins, bsec).sum(axis=1).astype(np.float64)
            m = counts.mean(); v = counts.var()
            out.append(("fano", float(v / m) if m > 1e-9 else 0.0))
            out.append(("burst_rate_per_s", float(counts.mean())))
        else:
            out += [("fano", 0.0), ("burst_rate_per_s", 0.0)]
        # large-jump rate: |dx| > 3*std(dx)
        if len(d1) > 0:
            jt = 3.0 * d1.std()
            out.append(("large_jumps_per_s", float((np.abs(d1) > jt).sum() / (len(x)/hz))))
        else:
            out.append(("large_jumps_per_s", 0.0))
    else:
        out += [("fano", 0.0), ("burst_rate_per_s", 0.0), ("large_jumps_per_s", 0.0)]
    # Phase-space area: bounding box of (x, dx/dt) after zscore
    if std > 1e-12 and len(d1) > 1:
        a = z[:-1]; b = _safe_z(d1)
        out.append(("ps_area", float((a.max()-a.min()) * (b.max()-b.min()))))
    else:
        out.append(("ps_area", 0.0))
    return out


# ---------------------------------------------------------------------------
def classify_channel(name):
    n = name.lower()
    if "temp" in n or "thermal" in n: return "temp"
    if "power" in n: return "power"
    if "cpufreq" in n or "freq" in n: return "freq"
    if "in0" in n or "in1" in n: return "voltage"
    if "curr" in n: return "current"
    return "other"


def cross_pair_features(x, y, hz, max_lag_s=5.0):
    """Compute Pearson r, peak |r| within ±max_lag, lag of peak, LF coherence."""
    out = {}
    zx = _safe_z(x); zy = _safe_z(y)
    out["r0"] = float((zx * zy).mean())
    n = len(zx)
    max_lag = min(int(max_lag_s * hz), n // 4)
    if max_lag < 2:
        out.update({"peak_r": out["r0"], "lag_peak_s": 0.0, "lf_coh": 0.0})
        return out
    # cross-correlation via FFT
    fx = np.fft.rfft(zx, n=2*n); fy = np.fft.rfft(zy, n=2*n)
    xcorr = np.fft.irfft(fx * np.conj(fy))[: n] / n
    xcorr_neg = np.fft.irfft(fx * np.conj(fy))[-(n-1):][::-1] / n
    full = np.concatenate([xcorr_neg, xcorr])  # length 2n-1
    lags = np.arange(-(n-1), n)
    sel = (np.abs(lags) <= max_lag)
    seg = full[sel]; ls = lags[sel]
    idx = int(np.argmax(np.abs(seg)))
    out["peak_r"] = float(seg[idx])
    out["lag_peak_s"] = float(ls[idx] / hz)
    # LF coherence proxy: correlation of 1-Hz lowpassed signals
    w = max(2, int(hz))
    k = np.ones(w, dtype=np.float32) / w
    lx = np.convolve(zx, k, mode="valid")
    ly = np.convolve(zy, k, mode="valid")
    if lx.std() > 1e-9 and ly.std() > 1e-9:
        out["lf_coh"] = float(np.corrcoef(lx, ly)[0, 1])
    else:
        out["lf_coh"] = 0.0
    return out


def build_pair_set(channels, max_pairs=60):
    types = [classify_channel(c) for c in channels]
    pairs = []
    # (temp, power), (temp, freq), (power, freq)
    for i, ti in enumerate(types):
        for j, tj in enumerate(types):
            if j <= i: continue
            kind = None
            if ti == "temp" and tj == "power": kind = "T_P"
            elif ti == "power" and tj == "temp": kind = "T_P"
            elif ti == "temp" and tj == "freq": kind = "T_F"
            elif ti == "freq" and tj == "temp": kind = "T_F"
            elif ti == "power" and tj == "freq": kind = "P_F"
            elif ti == "freq" and tj == "power": kind = "P_F"
            else: continue
            pairs.append((i, j, kind))
    # prefer most-distinct channel pairs; cap
    return pairs[:max_pairs]


# ---------------------------------------------------------------------------
def feature_timeseries(data, hz, out_hz=10.0, win_s=2.0):
    """Build a per-channel-derivative feature timeseries.

    For each raw channel produce 4 features:
      - rolling mean over win_s
      - rolling std over win_s
      - rolling mean of |d1| over win_s
      - rolling LF/HF energy ratio over win_s (proxy for spectral regime)
    Sampled at out_hz.

    Returns (data_out (T', 4*C), names list).
    """
    T, C = data.shape
    win = max(4, int(win_s * hz))
    step = max(1, int(hz / out_hz))
    starts = np.arange(0, T - win, step)
    T2 = len(starts)
    feats = np.zeros((T2, 4 * C), dtype=np.float32)
    for ci in range(C):
        col = data[:, ci]
        d1 = np.diff(col, prepend=col[0])
        for ti, s in enumerate(starts):
            seg = col[s:s+win]
            dseg = d1[s:s+win]
            m = seg.mean()
            sd = seg.std()
            ad = np.abs(dseg).mean()
            # LF/HF: low half vs high half of energy
            if sd > 1e-9:
                z = (seg - m) / sd
                S = np.abs(np.fft.rfft(z))**2
                half = len(S) // 2
                lf = S[:half].sum(); hf = S[half:].sum()
                ratio = float(lf / (hf + 1e-9))
            else:
                ratio = 0.0
            base = 4 * ci
            feats[ti, base+0] = m
            feats[ti, base+1] = sd
            feats[ti, base+2] = ad
            feats[ti, base+3] = ratio
    return feats


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--max_pairs", type=int, default=60)
    args = ap.parse_args()

    z = np.load(args.inp, allow_pickle=True)
    chs = list(z["channels"])
    data = z["data"].astype(np.float32)
    ts = z["ts"]
    T, C = data.shape
    hz = float(1.0 / np.median(np.diff(ts))) if len(ts) > 1 else 50.0
    print(f"[features] T={T} C={C} hz~{hz:.1f}")

    # ---- per-channel scalars ----
    scalar_names = []
    scalar_vals = []
    nonconst_mask = data.std(axis=0) > 1e-9
    for ci, name in enumerate(chs):
        if not nonconst_mask[ci]:
            continue
        for suf, val in per_channel_scalars(data[:, ci], hz):
            scalar_names.append(f"{name}__{suf}")
            scalar_vals.append(val)
    print(f"[features] scalar per-channel features: {len(scalar_vals)}")

    # ---- cross-channel ----
    pairs = build_pair_set([c for c in chs], max_pairs=args.max_pairs)
    # filter to non-constant
    pairs = [(i, j, k) for (i, j, k) in pairs if nonconst_mask[i] and nonconst_mask[j]]
    print(f"[features] pair features: {len(pairs)} pairs × 4 = {len(pairs)*4}")
    for (i, j, kind) in pairs:
        f = cross_pair_features(data[:, i], data[:, j], hz)
        for suf, val in f.items():
            scalar_names.append(f"PAIR_{kind}_{chs[i]}__{chs[j]}__{suf}")
            scalar_vals.append(val)
    print(f"[features] total scalar features: {len(scalar_vals)}")

    # ---- time-series features ----
    feats_ts = feature_timeseries(data, hz, out_hz=10.0, win_s=2.0)
    ts_names = []
    for ci, name in enumerate(chs):
        for s in ("mean", "std", "absd1", "lfhf"):
            ts_names.append(f"{name}__{s}")
    # drop ts features for constant channels (zero out so model doesn't waste capacity)
    for ci in range(C):
        if not nonconst_mask[ci]:
            feats_ts[:, 4*ci:4*ci+4] = 0.0
    print(f"[features] time-series features: shape={feats_ts.shape} names={len(ts_names)}")

    out = args.out or args.inp.replace("_rich.npz", "_features.npz")
    np.savez_compressed(
        out,
        scalar_names=np.array(scalar_names),
        scalar_values=np.array(scalar_vals, dtype=np.float32),
        features_ts=feats_ts,
        feature_names_ts=np.array(ts_names),
        nonconst_mask=nonconst_mask,
        channels=np.array(chs),
        hz=hz,
    )
    print(f"[features] saved: {out}")


if __name__ == "__main__":
    main()
