"""Phase 11C — Task D: Product-of-experts fusion.

Take ALL scalar features extracted in Phase 11C:
  - max_sample features per channel (basic stats + 1st/2nd derivs)
  - wavelet features (3*N_SCALES+2 per channel)
  - bias-grid features (5*8*10 flattened)
Plus prior Phase 8 features if available (3430-dim per host).

For each feature we estimate P(machine = ikaros | feature) via Gaussian KDE
fit on training samples and evaluate log-likelihood ratio on held-out test.
Fuse via sum of log-likelihood ratios (product-of-experts on independence
assumption).

Because we only have N=1 capture per host in Phase 11C, we build a *sample*
distribution by **bootstrap-windowing**: from each host's RAPL+aux trace we
extract M overlapping 5 s windows -> M feature vectors per host. We then
split train/test, fit per-feature KDE on train, score test.

We sweep K in {4, 8, 16, 32, 64, 128} top-features (selected by absolute
class-mean separation on train).

Output: poe_fusion.json with curve K -> accuracy, plus saved arrays.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from scipy.stats import gaussian_kde

REPO = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
OUT_DIR = REPO / "results/IDENTITY_BENCHMARK_2026-05-30/embodiment11c"

WIN_S = 5.0
STRIDE_S = 0.5
N_BOOTSTRAP = 100
K_VALUES = [4, 8, 16, 32, 64, 128]


def load_max(host):
    p = OUT_DIR / f"{host}_max.npz"
    if not p.exists():
        return None
    return np.load(p, allow_pickle=True)


def extract_windowed_features(host_data, host_label):
    """Return (n_windows, n_features) feature matrix.

    Features per window per channel: mean, std, log_std, abs_diff_mean,
    p95, p05, range, fano = var/mean.
    """
    if host_data is None:
        return None, []
    chan_names = []
    # gather (ts, val) pairs
    pairs = []
    for k in host_data.files:
        if k.endswith("_ts"):
            base = k[:-3]
            if base + "_val" in host_data.files:
                ts = host_data[base + "_ts"]
                val = host_data[base + "_val"]
                if ts.size < 50:
                    continue
                pairs.append((base, ts, val))
    if not pairs:
        return None, []

    # window definition by global time
    t0 = min(p[1][0] for p in pairs)
    t1 = max(p[1][-1] for p in pairs)
    starts = np.arange(t0, t1 - WIN_S, STRIDE_S)
    if len(starts) < 4:
        # use as many windows as we can
        starts = np.linspace(t0, max(t0, t1 - WIN_S * 0.5), 10)

    rows = []
    for s in starts:
        e = s + WIN_S
        row = []
        for base, ts, val in pairs:
            mask = (ts >= s) & (ts < e)
            if mask.sum() < 5:
                row += [0.0] * 8
                continue
            if val.ndim == 2:
                # thermal: aggregate per zone, but for windows just use mean across zones
                x = val[mask].mean(axis=1).astype(np.float64)
            else:
                x = val[mask].astype(np.float64)
                if base == "rapl_uj":
                    # convert energy to power
                    if x.size > 1:
                        dt = np.diff(ts[mask])
                        dx = np.diff(x)
                        valid = dt > 0
                        if valid.any():
                            x = (dx[valid] / dt[valid]) * 1e-6
                        else:
                            x = np.array([0.0])
                elif base == "cpu_busy":
                    if x.size > 1:
                        x = np.diff(x)
            if x.size < 2:
                row += [0.0] * 8
                continue
            mean = float(x.mean())
            std = float(x.std())
            row += [mean,
                    std,
                    float(np.log(std + 1e-12)),
                    float(np.abs(np.diff(x)).mean()),
                    float(np.percentile(x, 95)),
                    float(np.percentile(x, 5)),
                    float(x.max() - x.min()),
                    float(std * std / (abs(mean) + 1e-12))]
        rows.append(row)
    # build channel-feature names once
    feat_names = []
    for base, _, val in pairs:
        for s in ["mean", "std", "logstd", "absdiff", "p95", "p05", "range", "fano"]:
            feat_names.append(f"{base}_{s}")
    X = np.array(rows, dtype=np.float32)
    print(f"[poe] {host_label}: extracted {X.shape}")
    return X, feat_names


def add_wavelet(host, X_existing, names_existing):
    p = OUT_DIR / f"{host}_wavelet.npz"
    if not p.exists():
        return X_existing, names_existing
    d = np.load(p, allow_pickle=True)
    Xw = d["X"]  # (n_channels, n_wfeats)
    cn = d["channel_names"]
    fn = d["feature_names"]
    # flatten to single global feature vector
    flat = Xw.flatten().reshape(1, -1)
    flat_names = [f"wav_{c}_{f}" for c in cn for f in fn]
    # broadcast wavelet feats across all windows (acts as constant per-host bias)
    Xw_rep = np.repeat(flat, X_existing.shape[0], axis=0)
    return np.concatenate([X_existing, Xw_rep], axis=1), names_existing + flat_names


def add_bias_grid(host, X_existing, names_existing):
    p = OUT_DIR / f"{host}_bias_grid.npz"
    if not p.exists():
        return X_existing, names_existing
    d = np.load(p, allow_pickle=True)
    X = d["X"]  # (5, 8, 10)
    flat = X.flatten().reshape(1, -1)
    flat_names = [f"bg_s{si}_w{wi}_c{ci}"
                  for si in range(X.shape[0])
                  for wi in range(X.shape[1])
                  for ci in range(X.shape[2])]
    rep = np.repeat(flat, X_existing.shape[0], axis=0)
    return np.concatenate([X_existing, rep], axis=1), names_existing + flat_names


def fuse_and_eval(X_ik, X_da, feat_names, k_values, n_bootstrap, seed=0):
    rng = np.random.default_rng(seed)
    n_ik, n_da = X_ik.shape[0], X_da.shape[0]
    print(f"[poe] ikaros windows={n_ik}, daedalus windows={n_da}, features={X_ik.shape[1]}")
    if X_ik.shape[1] != X_da.shape[1]:
        # align by min feature count
        F = min(X_ik.shape[1], X_da.shape[1])
        X_ik = X_ik[:, :F]; X_da = X_da[:, :F]
        feat_names = feat_names[:F]
    # standardise per feature (jointly)
    X_all = np.concatenate([X_ik, X_da], axis=0)
    mu = X_all.mean(axis=0)
    sd = X_all.std(axis=0) + 1e-9
    X_ik = (X_ik - mu) / sd
    X_da = (X_da - mu) / sd

    F = X_ik.shape[1]
    accuracy = {k: [] for k in k_values}
    rocs = {k: [] for k in k_values}
    best_k_acc = {}

    for b in range(n_bootstrap):
        # train/test split per host
        ik_idx = rng.permutation(n_ik)
        da_idx = rng.permutation(n_da)
        if n_ik < 4 or n_da < 4:
            return None
        ik_tr = ik_idx[: n_ik // 2]; ik_te = ik_idx[n_ik // 2:]
        da_tr = da_idx[: n_da // 2]; da_te = da_idx[n_da // 2:]

        Xtr_ik = X_ik[ik_tr]; Xtr_da = X_da[da_tr]
        Xte_ik = X_ik[ik_te]; Xte_da = X_da[da_te]

        # class-mean separation on train -> rank features
        sep = np.abs(Xtr_ik.mean(0) - Xtr_da.mean(0))
        # variance penalty so we prefer features with discriminative AND low spread
        pooled_var = 0.5 * (Xtr_ik.var(0) + Xtr_da.var(0)) + 1e-6
        score = sep / np.sqrt(pooled_var)
        order = np.argsort(-score)

        # precompute KDE for each feature (top K_max)
        K_max = max(k_values)
        order_top = order[:K_max]
        kdes_ik, kdes_da = {}, {}
        for idx in order_top:
            xi = Xtr_ik[:, idx]; xd = Xtr_da[:, idx]
            try:
                k_i = gaussian_kde(xi, bw_method=0.3)
                k_d = gaussian_kde(xd, bw_method=0.3)
            except Exception:
                continue
            kdes_ik[idx] = k_i
            kdes_da[idx] = k_d

        for K in k_values:
            sel = [i for i in order[:K] if i in kdes_ik]
            if not sel:
                accuracy[K].append(0.5); continue
            # log-likelihood ratios for test
            def llr(X):
                ll = np.zeros(X.shape[0])
                for i in sel:
                    pi = kdes_ik[i].evaluate(X[:, i]) + 1e-30
                    pd = kdes_da[i].evaluate(X[:, i]) + 1e-30
                    ll += np.log(pi) - np.log(pd)
                return ll
            ll_ik = llr(Xte_ik)  # should be >0 for ikaros
            ll_da = llr(Xte_da)
            correct = (ll_ik > 0).sum() + (ll_da < 0).sum()
            total = ll_ik.size + ll_da.size
            accuracy[K].append(correct / total)

    summary = {}
    for K in k_values:
        a = np.array(accuracy[K])
        summary[K] = {
            "mean": float(a.mean()),
            "std": float(a.std()),
            "p5": float(np.percentile(a, 5)),
            "p95": float(np.percentile(a, 95)),
            "n": int(a.size),
        }
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hosts", nargs=2, default=["ikaros", "daedalus"])
    args = ap.parse_args()
    A, B = args.hosts

    d_a = load_max(A); d_b = load_max(B)
    if d_a is None or d_b is None:
        # fall back to Phase 8 rich data if Phase 11C capture missing
        print(f"[poe] Phase 11C max data missing for one host; checking Phase 8 fallback...")
        ph8 = REPO / "results/IDENTITY_BENCHMARK_2026-05-30/embodiment8"
        if d_a is None and (ph8 / f"{A}_features.npz").exists():
            print(f"  using Phase 8 features for {A}")
        if d_b is None and (ph8 / f"{B}_features.npz").exists():
            print(f"  using Phase 8 features for {B}")
    X_a, names_a = extract_windowed_features(d_a, A)
    X_b, names_b = extract_windowed_features(d_b, B)

    # If a host lacks Phase 11C data, build windows from Phase 8 rich data
    ph8 = REPO / "results/IDENTITY_BENCHMARK_2026-05-30/embodiment8"
    if (X_a is None or X_a.shape[0] < 4) and (ph8 / f"{A}_rich.npz").exists():
        print(f"[poe] {A}: falling back to Phase 8 rich (50 Hz) windows")
        d_rich = np.load(ph8 / f"{A}_rich.npz", allow_pickle=True)
        X_a, names_a = build_windows_from_rich(d_rich, A)
    if (X_b is None or X_b.shape[0] < 4) and (ph8 / f"{B}_rich.npz").exists():
        print(f"[poe] {B}: falling back to Phase 8 rich (50 Hz) windows")
        d_rich = np.load(ph8 / f"{B}_rich.npz", allow_pickle=True)
        X_b, names_b = build_windows_from_rich(d_rich, B)

    if X_a is None or X_b is None:
        # Phase 8 fallback path: use stored feature vectors (1 sample per host)
        ph8 = REPO / "results/IDENTITY_BENCHMARK_2026-05-30/embodiment8"
        f_a = np.load(ph8 / f"{A}_features.npz", allow_pickle=True) if (ph8 / f"{A}_features.npz").exists() else None
        f_b = np.load(ph8 / f"{B}_features.npz", allow_pickle=True) if (ph8 / f"{B}_features.npz").exists() else None
        if f_a is None or f_b is None:
            print("[poe] CRITICAL: cannot proceed without both hosts")
            return
        # use only the raw rich data and build pseudo-windows by sliding through 50Hz data
        d_a = np.load(ph8 / f"{A}_rich.npz", allow_pickle=True)
        d_b = np.load(ph8 / f"{B}_rich.npz", allow_pickle=True)
        X_a, names_a = build_windows_from_rich(d_a, A)
        X_b, names_b = build_windows_from_rich(d_b, B)

    # align feature names if differing channel sets
    common = [n for n in names_a if n in set(names_b)]
    idx_a = [names_a.index(n) for n in common]
    idx_b = [names_b.index(n) for n in common]
    X_a = X_a[:, idx_a]; X_b = X_b[:, idx_b]
    names = common

    # add wavelet + bias-grid (if available)
    if (OUT_DIR / f"{A}_wavelet.npz").exists() and (OUT_DIR / f"{B}_wavelet.npz").exists():
        X_a, names = add_wavelet(A, X_a, names)
        X_b, names_b2 = add_wavelet(B, X_b, list(names) if isinstance(names, list) else names)
        # crop to min
        m = min(X_a.shape[1], X_b.shape[1])
        X_a = X_a[:, :m]; X_b = X_b[:, :m]
        names = names[:m]
    if (OUT_DIR / f"{A}_bias_grid.npz").exists() and (OUT_DIR / f"{B}_bias_grid.npz").exists():
        X_a, names = add_bias_grid(A, X_a, names)
        X_b, names_b2 = add_bias_grid(B, X_b, list(names) if isinstance(names, list) else names)
        m = min(X_a.shape[1], X_b.shape[1])
        X_a = X_a[:, :m]; X_b = X_b[:, :m]
        names = names[:m]

    print(f"[poe] fused features: {X_a.shape[1]}, windows ikaros={X_a.shape[0]}, daedalus={X_b.shape[0]}")
    summary = fuse_and_eval(X_a, X_b, names, K_VALUES, N_BOOTSTRAP)

    # also compute inter vs intra Hamming distance for the bias-grid
    hamming = compute_bias_hamming(A, B)

    out = {
        "n_features": int(X_a.shape[1]),
        "n_windows_ikaros": int(X_a.shape[0]),
        "n_windows_daedalus": int(X_b.shape[0]),
        "k_curve": summary,
        "bias_grid_hamming": hamming,
    }
    out_path = OUT_DIR / "poe_fusion.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[poe] saved {out_path}")
    for K in K_VALUES:
        s = summary[K]
        print(f"  K={K:4d}: acc = {s['mean']:.3f} +/- {s['std']:.3f}  [p5={s['p5']:.3f}, p95={s['p95']:.3f}]")


def build_windows_from_rich(rich_npz, host_label):
    """Phase 8 rich data: (N, C) array at ~50 Hz. Build pseudo-windows."""
    if "data" in rich_npz.files and "ts" in rich_npz.files and "channels" in rich_npz.files:
        data = rich_npz["data"]
        ts = rich_npz["ts"]
        chans = list(rich_npz["channels"])
    else:
        # alt schema
        chans = list(rich_npz.files)
        return None, []
    # window = 5 s, stride 0.5 s
    fs = data.shape[0] / (ts[-1] - ts[0])
    w = max(1, int(WIN_S * fs))
    s = max(1, int(STRIDE_S * fs))
    rows = []
    feat_names = []
    for c, name in enumerate(chans):
        for st in ["mean", "std", "logstd", "absdiff", "p95", "p05", "range", "fano"]:
            feat_names.append(f"{name}_{st}")
    for start in range(0, data.shape[0] - w, s):
        win = data[start:start + w]
        row = []
        for c in range(data.shape[1]):
            x = win[:, c].astype(np.float64)
            mean = float(x.mean()); std = float(x.std())
            row += [mean, std,
                    float(np.log(std + 1e-12)),
                    float(np.abs(np.diff(x)).mean()) if x.size > 1 else 0.0,
                    float(np.percentile(x, 95)),
                    float(np.percentile(x, 5)),
                    float(x.max() - x.min()),
                    float(std * std / (abs(mean) + 1e-12))]
        rows.append(row)
    X = np.array(rows, dtype=np.float32)
    print(f"[poe] {host_label}: rich-window features {X.shape}")
    return X, feat_names


def compute_bias_hamming(A, B):
    a_p = OUT_DIR / f"{A}_bias_grid.npz"
    b_p = OUT_DIR / f"{B}_bias_grid.npz"
    if not a_p.exists() or not b_p.exists():
        return None
    A_X = np.load(a_p)["X"]; B_X = np.load(b_p)["X"]
    # quantise each cell to 4-bit signature using zscore against grand mean
    all_vals = np.concatenate([A_X.flatten(), B_X.flatten()])
    mu, sd = all_vals.mean(), all_vals.std() + 1e-9
    qa = np.clip(((A_X - mu) / sd * 2 + 8).astype(int), 0, 15)
    qb = np.clip(((B_X - mu) / sd * 2 + 8).astype(int), 0, 15)
    inter = float((qa != qb).mean())
    # intra: shuffle cells within host
    rng = np.random.default_rng(0)
    intra_vals = []
    for X in (qa, qb):
        flat = X.flatten()
        for _ in range(50):
            perm = rng.permutation(flat)
            intra_vals.append((flat != perm).mean())
    intra = float(np.mean(intra_vals))
    return {"inter_hamming_frac": inter,
            "intra_hamming_frac": intra,
            "ratio": inter / max(intra, 1e-9)}


if __name__ == "__main__":
    main()
