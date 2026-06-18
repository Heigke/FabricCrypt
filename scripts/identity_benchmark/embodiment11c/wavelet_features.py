"""Phase 11C — Task B: Continuous wavelet decomposition.

For each per-channel time series from <host>_max.npz, compute Morlet CWT at 32
log-spaced scales. Extract per-(channel, scale) features:
  - scale_power   : log mean(|W|^2)
  - scale_entropy : Shannon entropy of normalised |W|^2 envelope (proxy for
                    burstiness vs continuous power)
  - scale_kurt    : kurtosis of |W| (extreme-event signature)

Then cross-scale:
  - ratio_low_high : log scale_power[low] / scale_power[high]   (1/f slope)
  - peak_scale_idx : argmax(scale_power)

Output: <host>_wavelet.npz with a single (n_channels, n_features) array plus
feature_names.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pywt

REPO = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
OUT_DIR = REPO / "results/IDENTITY_BENCHMARK_2026-05-30/embodiment11c"

N_SCALES = 32
WAVELET = "cmor1.5-1.0"  # Complex Morlet


def cwt_features(x: np.ndarray, fs: float):
    """Return (3 * n_scales + 2,) feature vector for a single channel."""
    if x.size < 64 or fs <= 0:
        return np.zeros(3 * N_SCALES + 2, dtype=np.float32)
    x = (x - x.mean()) / (x.std() + 1e-12)
    # log-spaced scales spanning ~2 samples up to N/4
    scales = np.geomspace(2.0, max(4.0, x.size / 4), N_SCALES)
    try:
        coef, _ = pywt.cwt(x, scales, WAVELET, sampling_period=1.0 / fs)
    except Exception:
        return np.zeros(3 * N_SCALES + 2, dtype=np.float32)
    amp = np.abs(coef)                        # (n_scales, N)
    power = amp ** 2

    sp = np.log10(power.mean(axis=1) + 1e-20)             # (n_scales,)
    # entropy of normalised power envelope per scale
    p = power / (power.sum(axis=1, keepdims=True) + 1e-20)
    ent = -(p * np.log(p + 1e-20)).sum(axis=1)
    ent /= np.log(p.shape[1])                              # normalised
    # kurtosis of |W|
    m = amp.mean(axis=1)
    s = amp.std(axis=1) + 1e-20
    z = (amp - m[:, None]) / s[:, None]
    kurt = (z ** 4).mean(axis=1) - 3.0

    low_idx = slice(0, N_SCALES // 4)
    hi_idx = slice(3 * N_SCALES // 4, N_SCALES)
    ratio = sp[low_idx].mean() - sp[hi_idx].mean()
    peak = float(sp.argmax())

    feats = np.concatenate([sp, ent, kurt, [ratio, peak]]).astype(np.float32)
    return feats


def feature_names():
    names = []
    for i in range(N_SCALES): names.append(f"sp_{i:02d}")
    for i in range(N_SCALES): names.append(f"ent_{i:02d}")
    for i in range(N_SCALES): names.append(f"kurt_{i:02d}")
    names += ["ratio_lo_hi", "peak_scale_idx"]
    return names


def process_host(host):
    inp = OUT_DIR / f"{host}_max.npz"
    if not inp.exists():
        print(f"[wavelet] MISSING {inp}")
        return None
    d = np.load(inp, allow_pickle=True)
    # discover channel pairs
    pairs = []
    for k in d.files:
        if k.endswith("_ts"):
            base = k[:-3]
            if base + "_val" in d.files:
                pairs.append(base)
    print(f"[wavelet] {host}: {len(pairs)} channels")

    fnames_per_chan = feature_names()
    all_rows = []
    chan_names = []
    for base in pairs:
        ts = d[base + "_ts"]
        val = d[base + "_val"]
        if ts.size < 64:
            continue
        # achieved rate
        dur = ts[-1] - ts[0]
        fs = ts.size / dur if dur > 0 else 0
        # convert RAPL energy to power (diff)
        if base == "rapl_uj":
            v = np.diff(val) * fs * 1e-6
            v = np.concatenate([[v[0]], v])
        elif base == "cpu_busy":
            v = np.diff(val.astype(np.float64))
            v = np.concatenate([[v[0]], v])
        elif val.ndim == 2:
            # thermal: process each zone separately
            for zi in range(val.shape[1]):
                col = val[:, zi].astype(np.float64)
                feats = cwt_features(col, fs)
                all_rows.append(feats)
                chan_names.append(f"{base}_zone{zi}")
            continue
        else:
            v = val.astype(np.float64)
        feats = cwt_features(v, fs)
        all_rows.append(feats)
        chan_names.append(base)

    X = np.stack(all_rows, axis=0)  # (n_channels, n_features)
    print(f"[wavelet] {host}: feature matrix {X.shape}")
    out = OUT_DIR / f"{host}_wavelet.npz"
    np.savez_compressed(out,
                        X=X,
                        channel_names=np.array(chan_names),
                        feature_names=np.array(fnames_per_chan))
    print(f"[wavelet] saved {out}")
    return X.shape


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hosts", nargs="+", default=["ikaros", "daedalus"])
    args = ap.parse_args()
    for h in args.hosts:
        process_host(h)


if __name__ == "__main__":
    main()
