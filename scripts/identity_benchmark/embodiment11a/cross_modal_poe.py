"""
TASK A — Cross-modal weak-signal aggregation (Product-of-Experts).

For each common channel, fit a Gaussian KDE per machine on TRAIN windows,
compute log-likelihood-ratio LLR(channel) on TEST windows, sum LLRs across
the K weakest single-channel classifiers, and check whether the fused
classifier reaches >= 85% on held-out test.

Pre-reg: 16 weakest channels (single-channel acc < 60%) fused -> >= 85%
held-out, bootstrap 100 channel-subset draws, 95% CI.

Data: results/IDENTITY_BENCHMARK_2026-05-30/embodiment8/{ikaros,daedalus}_rich.npz
Output: results/.../embodiment11a/task_a_poe.json
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path
import numpy as np
from scipy.stats import gaussian_kde

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "results/IDENTITY_BENCHMARK_2026-05-30/embodiment8"
OUT = ROOT / "results/IDENTITY_BENCHMARK_2026-05-30/embodiment11a"
OUT.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(20260601)

# Window the raw 50Hz streams into 5-second windows (=250 samples),
# extract simple summary stats per window per channel -> per-window feature.
# Use mean+std as the channel feature -> 2 features per channel? Keep it 1D
# (std/iqr) for clean PoE; identity is mostly in variance/noise not absolute level.
WINDOW = 250  # 5s at 50Hz
FEATURE = "std"  # 'std' or 'iqr' — variance signature
TRAIN_FRAC = 0.6
MIN_K = 4
MAX_K = 24
N_BOOT = 100


def load_common():
    i = np.load(DATA / "ikaros_rich.npz")
    d = np.load(DATA / "daedalus_rich.npz")
    ic = list(map(str, i["channels"]))
    dc = list(map(str, d["channels"]))
    common = [c for c in ic if c in dc]
    iidx = [ic.index(c) for c in common]
    didx = [dc.index(c) for c in common]
    return i["data"][:, iidx], d["data"][:, didx], common


def windowize(arr, w=WINDOW, feature=FEATURE):
    # arr: (T, C). Returns (N_win, C) with per-window feature.
    T, C = arr.shape
    n = T // w
    arr = arr[: n * w].reshape(n, w, C)
    if feature == "std":
        return arr.std(axis=1)
    elif feature == "iqr":
        return np.percentile(arr, 75, axis=1) - np.percentile(arr, 25, axis=1)
    else:
        raise ValueError(feature)


def kde_loglik(x, train):
    # robust gaussian KDE log-pdf with small jitter; constant channels => -inf
    if np.std(train) < 1e-12:
        # treat as point mass -> log-pdf = 0 if equal, large neg if not
        diff = np.abs(x - train.mean())
        return -1e3 * (diff > 1e-9).astype(float)
    try:
        k = gaussian_kde(train + RNG.normal(0, 1e-9, train.shape))
        return np.log(np.clip(k.evaluate(x), 1e-300, None))
    except Exception:
        # fallback: gaussian fit
        mu, sd = train.mean(), train.std() + 1e-9
        return -0.5 * ((x - mu) / sd) ** 2 - np.log(sd)


def per_channel_acc(Ftr_i, Ftr_d, Fte_i, Fte_d):
    """Return (acc_per_channel, llr_test_ikaros, llr_test_daedalus)
    where llr = log p(x|ikaros) - log p(x|daedalus) on test windows."""
    C = Ftr_i.shape[1]
    n_te = Fte_i.shape[0]
    llr_i = np.zeros((n_te, C))
    llr_d = np.zeros((n_te, C))
    accs = np.zeros(C)
    for c in range(C):
        li_i = kde_loglik(Fte_i[:, c], Ftr_i[:, c])
        ld_i = kde_loglik(Fte_i[:, c], Ftr_d[:, c])
        llr_i[:, c] = li_i - ld_i  # positive => ikaros
        li_d = kde_loglik(Fte_d[:, c], Ftr_i[:, c])
        ld_d = kde_loglik(Fte_d[:, c], Ftr_d[:, c])
        llr_d[:, c] = li_d - ld_d  # ikaros windows should have higher llr
        # acc: ikaros windows have llr>0; daedalus windows have llr<0
        correct = (llr_i[:, c] > 0).sum() + (llr_d[:, c] < 0).sum()
        accs[c] = correct / (2 * n_te)
    return accs, llr_i, llr_d


def fused_acc(llr_i, llr_d, channel_idx):
    s_i = llr_i[:, channel_idx].sum(axis=1)
    s_d = llr_d[:, channel_idx].sum(axis=1)
    correct = (s_i > 0).sum() + (s_d < 0).sum()
    return correct / (len(s_i) + len(s_d))


def main():
    t0 = time.time()
    print("[A] Loading data ...")
    Xi, Xd, channels = load_common()
    print(f"[A] ikaros shape {Xi.shape}  daedalus shape {Xd.shape}  common channels {len(channels)}")

    Wi = windowize(Xi)
    Wd = windowize(Xd)
    n_i, n_d, C = Wi.shape[0], Wd.shape[0], Wi.shape[1]
    print(f"[A] windows  ikaros={n_i}  daedalus={n_d}  channels={C}")

    n_tr_i = int(TRAIN_FRAC * n_i)
    n_tr_d = int(TRAIN_FRAC * n_d)
    # interleaved split: chronological, train=first 60%, test=last 40%
    Ftr_i, Fte_i = Wi[:n_tr_i], Wi[n_tr_i:]
    Ftr_d, Fte_d = Wd[:n_tr_d], Wd[n_tr_d:]
    # equalize test set sizes
    n_te = min(Fte_i.shape[0], Fte_d.shape[0])
    Fte_i, Fte_d = Fte_i[:n_te], Fte_d[:n_te]
    print(f"[A] train ikaros={Ftr_i.shape} daedalus={Ftr_d.shape} test={Fte_i.shape}")

    print("[A] Per-channel KDE LLR ...")
    accs, llr_i, llr_d = per_channel_acc(Ftr_i, Ftr_d, Fte_i, Fte_d)
    order = np.argsort(accs)
    print(f"[A] worst5={accs[order[:5]].round(3).tolist()}  best5={accs[order[-5:]].round(3).tolist()}")

    # Pre-reg cohort: channels with acc < 0.60 ("weak")
    weak_mask = accs < 0.60
    weak_idx = np.where(weak_mask)[0]
    print(f"[A] {weak_mask.sum()} weak channels (<60% single-channel acc)")
    K = min(16, len(weak_idx))
    if K < MIN_K:
        # fall back: take the 16 worst above chance (>0.51)
        cand = order[(accs[order] > 0.51) & (accs[order] < 0.60)]
        weak_idx = cand[:16]
        K = len(weak_idx)
        print(f"[A] fallback weak set size={K}")

    # Headline: weakest-16 fused
    sel = order[: max(K, 16)][: 16] if K >= 16 else weak_idx
    if len(sel) < 16:
        # pad with next-worst channels
        extra = [i for i in order if i not in sel][: 16 - len(sel)]
        sel = np.concatenate([sel, extra])
    headline_acc = fused_acc(llr_i, llr_d, sel)
    print(f"[A] headline (16 weakest) fused acc = {headline_acc:.4f}")

    # Bootstrap: 100 random subsets of size 16 drawn from "weak" pool
    pool = weak_idx if len(weak_idx) >= 16 else order[:32]
    boot_accs = []
    for b in range(N_BOOT):
        if len(pool) <= 16:
            pick = pool
        else:
            pick = RNG.choice(pool, size=16, replace=False)
        boot_accs.append(float(fused_acc(llr_i, llr_d, pick)))
    boot_accs = np.array(boot_accs)
    ci_lo, ci_hi = np.percentile(boot_accs, [2.5, 97.5])
    print(f"[A] bootstrap mean={boot_accs.mean():.4f}  95%CI=[{ci_lo:.4f},{ci_hi:.4f}]")

    # Scaling curve: K from 1..min(32,C) using weakest-first
    curve = []
    for k in range(1, min(32, C) + 1):
        idx = order[:k]
        a = fused_acc(llr_i, llr_d, idx)
        curve.append({"k": k, "acc": float(a)})
    # also strongest-first for reference
    curve_top = []
    order_desc = order[::-1]
    for k in range(1, min(32, C) + 1):
        idx = order_desc[:k]
        a = fused_acc(llr_i, llr_d, idx)
        curve_top.append({"k": k, "acc": float(a)})

    out = {
        "task": "A_cross_modal_poe",
        "window_samples": WINDOW,
        "feature": FEATURE,
        "n_channels_common": int(C),
        "n_train_ikaros": int(Ftr_i.shape[0]),
        "n_train_daedalus": int(Ftr_d.shape[0]),
        "n_test_per_machine": int(n_te),
        "per_channel_acc_min": float(accs.min()),
        "per_channel_acc_median": float(np.median(accs)),
        "per_channel_acc_max": float(accs.max()),
        "n_weak_channels_lt60": int(weak_mask.sum()),
        "headline_16_weakest_fused_acc": float(headline_acc),
        "bootstrap_mean_acc": float(boot_accs.mean()),
        "bootstrap_ci95": [float(ci_lo), float(ci_hi)],
        "bootstrap_n_draws": int(N_BOOT),
        "prereg_threshold": 0.85,
        "prereg_PASS": bool(ci_lo >= 0.85),
        "scaling_curve_weakest_first": curve,
        "scaling_curve_strongest_first": curve_top,
        "weak_channel_names": [channels[i] for i in weak_idx[:32]],
        "elapsed_s": round(time.time() - t0, 2),
    }
    with open(OUT / "task_a_poe.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"[A] wrote {OUT/'task_a_poe.json'}  prereg_PASS={out['prereg_PASS']}")


if __name__ == "__main__":
    main()
