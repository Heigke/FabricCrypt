"""z374: Publication-grade UCI-HAR HDC benchmark.

Compares:
  - NS-RAM HDC with V_d-as-bit encoding (z292/DS_N5f-style)
  - NS-RAM HDC ablation (no V_d modulation, i.e. V_d_HIGH == V_d_LOW)
  - Binary HDC (Imani 2018-style, sign-bundled bipolar)
  - Bipolar HDC (Kanerva, sign-bundled bipolar)
  - Random projection + ridge
  - Linear ridge on raw 561 features (sanity)

Scales N in {1024, 4096, 16384, 65536, 262144, 1048576} for digital methods.
NS-RAM is capped at N <= NSRAM_MAX (default 16384) for compute feasibility;
the surrogate-driven inner loop is O(C * N_test * N * T_steps) which is
intractable beyond ~16K with T_steps=100 inside a 3-hour budget.

n_seeds = 10. Per (method, N): mean accuracy, 95% bootstrap CI (1000 resamples),
train time (s), test latency (ms/sample), peak GPU memory (MB), FLOPs proxy.

Paired t-tests NS-RAM vs each digital baseline at each N. Bonferroni correction.

Output: results/z374_uci_har_pubgrade/{summary.json, all_results.csv,
        comparison_plot.png, run.log}
"""
from __future__ import annotations
import argparse, json, time, gc, csv, sys, traceback
from pathlib import Path
import numpy as np
import torch

# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def load_uci_har(root: Path):
    root = Path(root)
    Xtr = np.loadtxt(root / "train" / "X_train.txt", dtype=np.float32)
    ytr = np.loadtxt(root / "train" / "y_train.txt", dtype=np.int64) - 1
    Xte = np.loadtxt(root / "test" / "X_test.txt", dtype=np.float32)
    yte = np.loadtxt(root / "test" / "y_test.txt", dtype=np.int64) - 1
    return Xtr, ytr, Xte, yte


def quantize(X, mins, maxs, Q):
    span = (maxs - mins)
    span = np.where(span < 1e-9, 1.0, span)
    Xn = np.clip((X - mins) / span, 0.0, 1.0)
    return np.clip(np.floor(Xn * (Q - 1) + 0.5).astype(np.int32), 0, Q - 1)


# --------------------------------------------------------------------------
# Digital HDC helpers (all on GPU, int8/int16)
# --------------------------------------------------------------------------
def build_level_codebook_one(D, Q, rng_gpu, device, kind="thermometer"):
    """Returns (Q, D) int8 codebook for one feature.
    kind: 'thermometer' (binary HDC, Imani) or 'random_bipolar' (Kanerva).
    """
    if kind == "thermometer":
        base = (torch.randint(0, 2, (D,), device=device, generator=rng_gpu,
                              dtype=torch.int8) * 2 - 1)
        flips_per_step = max(1, D // Q)
        order = torch.randperm(D, device=device, generator=rng_gpu)
        L = torch.empty((Q, D), dtype=torch.int8, device=device)
        L[0] = base
        cur = base.clone()
        for q in range(1, Q):
            idx = order[(q - 1) * flips_per_step: q * flips_per_step]
            cur[idx] = -cur[idx]
            L[q] = cur
        return L
    else:
        # Independent random bipolar per level (Kanerva)
        L = (torch.randint(0, 2, (Q, D), device=device, generator=rng_gpu,
                           dtype=torch.int8) * 2 - 1)
        return L


def encode_and_bundle(Xq_tr_gpu, Xq_te_gpu, F, D, Q, device, seed,
                      codebook_kind="thermometer"):
    Ntr = Xq_tr_gpu.shape[0]
    Nte = Xq_te_gpu.shape[0]
    Htr = torch.zeros((Ntr, D), dtype=torch.int16, device=device)
    Hte = torch.zeros((Nte, D), dtype=torch.int16, device=device)
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)
    for f in range(F):
        P = (torch.randint(0, 2, (D,), device=device, generator=rng,
                           dtype=torch.int8) * 2 - 1)
        L = build_level_codebook_one(D, Q, rng, device, kind=codebook_kind)
        idx_tr = Xq_tr_gpu[:, f].to(torch.long)
        idx_te = Xq_te_gpu[:, f].to(torch.long)
        bound_tr = (L[idx_tr] * P)
        Htr.add_(bound_tr.to(torch.int16))
        del bound_tr
        bound_te = (L[idx_te] * P)
        Hte.add_(bound_te.to(torch.int16))
        del bound_te, L, P
    return Htr, Hte


def class_prototypes_gpu(Hsum, y, n_classes, device, mode="cosine"):
    """mode='cosine' -> L2-normalized sum (float).
       mode='binary' -> sign() of per-class sum (Imani majority vote).
    """
    D = Hsum.shape[1]
    protos = torch.zeros((n_classes, D), dtype=torch.float32, device=device)
    Hf = Hsum.to(torch.float32)
    for c in range(n_classes):
        m = (y == c)
        if m.any():
            s = Hf[m].sum(dim=0)
            if mode == "binary":
                s = torch.sign(s)
                s[s == 0] = 1.0
            protos[c] = s
    if mode == "cosine":
        norms = protos.norm(dim=1, keepdim=True).clamp_min(1e-9)
        protos = protos / norms
    return protos


def predict_gpu(Hsum, protos, mode="cosine"):
    H = Hsum.to(torch.float32)
    if mode == "cosine":
        norms = H.norm(dim=1, keepdim=True).clamp_min(1e-9)
        Hn = H / norms
        sims = Hn @ protos.T
    else:
        # Hamming-like / dot product on sign(H) vs sign(protos)
        Hn = torch.sign(H)
        sims = Hn @ protos.T
    return sims.argmax(dim=1)


# --------------------------------------------------------------------------
# NS-RAM (V_d-as-bit) helpers
# --------------------------------------------------------------------------
Q_ELEM = 1.602176634e-19

def load_surrogate(path, device):
    z = np.load(path)
    return {
        "I_d":   torch.tensor(z["Id"],    dtype=torch.float32, device=device),
        "I_ii":  torch.tensor(z["Iii"],   dtype=torch.float32, device=device),
        "I_leak":torch.tensor(z["Ileak"], dtype=torch.float32, device=device),
        "ax_VG1":torch.tensor(z["vg1_axis"], dtype=torch.float32, device=device),
        "ax_VG2":torch.tensor(z["vg2_axis"], dtype=torch.float32, device=device),
        "ax_Vd": torch.tensor(z["vd_axis"],  dtype=torch.float32, device=device),
        "ax_Vb": torch.tensor(z["vb_axis"],  dtype=torch.float32, device=device),
    }


def bucketize_index(values, axis):
    n = axis.shape[0]
    return (torch.bucketize(values, axis) - 1).clamp(0, n - 2)


def query_surrogate(surr, VG1, VG2, Vd, Vb):
    iVG1 = bucketize_index(VG1, surr["ax_VG1"])
    iVG2 = bucketize_index(VG2, surr["ax_VG2"])
    iVd  = bucketize_index(Vd,  surr["ax_Vd"])
    iVb  = bucketize_index(Vb,  surr["ax_Vb"])
    return (surr["I_d"][iVG1, iVG2, iVd, iVb],
            surr["I_ii"][iVG1, iVG2, iVd, iVb],
            surr["I_leak"][iVG1, iVG2, iVd, iVb])


def nsram_rates_vd(VG1, VG2, Vd, surr, C_b_F, dt_s, T_steps):
    device = VG1.device
    B, N = VG1.shape
    Vb_min = surr["ax_Vb"][0]
    Vb_max = surr["ax_Vb"][-1]
    rate_accum = torch.zeros(B, N, device=device)
    spike_events = torch.zeros(B, device=device)
    Vb = torch.zeros(B, N, device=device)
    for _ in range(T_steps):
        Vb_c = Vb.clamp(Vb_min, Vb_max)
        I_d, I_ii, I_leak = query_surrogate(surr, VG1, VG2, Vd, Vb_c)
        Vb = (Vb + dt_s * (I_ii - I_leak) / C_b_F).clamp(Vb_min, Vb_max)
        rate_accum = rate_accum + I_d.abs() / T_steps
        spike_events = spike_events + (
            (I_d.abs() * dt_s) > Q_ELEM).float().sum(dim=1)
    return rate_accum, spike_events


def run_nsram_seed(Xtr, ytr, Xte, yte, surr, device, N, Q, seed, n_classes,
                  V_G1_BIAS=0.30, V_G2_BIAS=0.30,
                  V_d_HIGH=2.00, V_d_LOW=0.50,
                  g_in=0.25, C_b_F=8e-15, dt_s=1e-7, T_steps=100,
                  batch_size=128, ablation_no_vd=False):
    """NS-RAM HDC with V_d-as-bit encoding (z292-style). Returns metrics dict."""
    rng = np.random.default_rng(seed)
    F = Xtr.shape[1]
    mins = Xtr.min(axis=0); maxs = Xtr.max(axis=0)
    Xtrq = quantize(Xtr, mins, maxs, Q)
    Xteq = quantize(Xte, mins, maxs, Q)

    t0 = time.time()
    # Build digital HDC encoding (record-based, thermometer)
    Xq_tr_gpu = torch.from_numpy(Xtrq).to(device)
    Xq_te_gpu = torch.from_numpy(Xteq).to(device)
    Htr, Hte = encode_and_bundle(Xq_tr_gpu, Xq_te_gpu, F, N, Q, device, seed,
                                 codebook_kind="thermometer")
    # Bipolar class prototypes via sign
    ytr_t = torch.from_numpy(ytr).to(device)
    protos_int = class_prototypes_gpu(Htr, ytr_t, n_classes, device,
                                       mode="binary")  # (C, N) ±1
    protos_np = protos_int.detach().cpu().numpy().astype(np.float32)

    # Query normalize (sample inf-norm to [-1, 1])
    def normalize(H):
        Hf = H.to(torch.float32)
        m = Hf.abs().max(dim=1, keepdim=True).values.clamp_min(1e-9)
        return Hf / m
    Hte_n = normalize(Hte)
    Htr_n = normalize(Htr)
    t_enc = time.time() - t0

    VG1_min = float(surr["ax_VG1"][0].item())
    VG1_max = float(surr["ax_VG1"][-1].item())

    # In ablation mode, fix V_d to a single value (HIGH) for both arms
    if ablation_no_vd:
        _Vh, _Vl = V_d_HIGH, V_d_HIGH
    else:
        _Vh, _Vl = V_d_HIGH, V_d_LOW

    def vd_pair(p):
        vd_pos = np.where(p > 0, _Vh, _Vl).astype(np.float32)
        vd_neg = np.where(p > 0, _Vl, _Vh).astype(np.float32)
        return vd_pos, vd_neg

    VG2_full = torch.full((1, N), V_G2_BIAS, dtype=torch.float32, device=device)

    def score_set(H_norm, y_true):
        Nset = H_norm.shape[0]
        scores = torch.zeros((Nset, n_classes), device=device)
        total_spikes = 0.0
        for c in range(n_classes):
            vd_pos_np, vd_neg_np = vd_pair(protos_np[c])
            Vd_pos_t = torch.tensor(vd_pos_np, dtype=torch.float32, device=device)
            Vd_neg_t = torch.tensor(vd_neg_np, dtype=torch.float32, device=device)
            for b0 in range(0, Nset, batch_size):
                b1 = min(b0 + batch_size, Nset)
                H_b = H_norm[b0:b1]
                B = b1 - b0
                VG1 = (V_G1_BIAS + g_in * H_b).clamp(VG1_min, VG1_max)
                VG2 = VG2_full.expand(B, N)
                Vd_p = Vd_pos_t.expand(B, N)
                Vd_n = Vd_neg_t.expand(B, N)
                rates_p, spikes_p = nsram_rates_vd(VG1, VG2, Vd_p, surr,
                                                   C_b_F, dt_s, T_steps)
                rates_n, spikes_n = nsram_rates_vd(VG1, VG2, Vd_n, surr,
                                                   C_b_F, dt_s, T_steps)
                rates_eff = rates_p - rates_n
                s = (rates_eff * H_b).sum(dim=1)
                scores[b0:b1, c] = s
                total_spikes += float(spikes_p.sum().item()
                                      + spikes_n.sum().item())
        preds = scores.argmax(dim=1).cpu().numpy()
        acc = float((preds == y_true).mean())
        return acc, total_spikes

    t_infer0 = time.time()
    test_acc, test_spikes = score_set(Hte_n, yte)
    t_infer = time.time() - t_infer0
    wall = time.time() - t0
    energy_J = test_spikes * 6.4e-15
    Nte = Xte.shape[0]

    del Htr, Hte, Htr_n, Hte_n, Xq_tr_gpu, Xq_te_gpu, protos_int
    gc.collect(); torch.cuda.empty_cache()

    return {
        "test_acc": test_acc, "wall_s": wall, "encode_s": t_enc,
        "infer_s": t_infer, "latency_ms_per_sample": (t_infer / Nte) * 1000.0,
        "energy_J_total": energy_J,
        "spike_events_total": float(test_spikes),
    }


# --------------------------------------------------------------------------
# Digital baselines
# --------------------------------------------------------------------------
def run_digital_hdc_seed(Xtr, ytr, Xte, yte, device, D, Q, seed, n_classes,
                         kind="binary"):
    """kind: 'binary' (thermometer codebook + sign protos),
             'bipolar' (random bipolar codebook + cosine protos)."""
    mins = Xtr.min(axis=0); maxs = Xtr.max(axis=0)
    Xtrq = quantize(Xtr, mins, maxs, Q)
    Xteq = quantize(Xte, mins, maxs, Q)
    Xq_tr_gpu = torch.from_numpy(Xtrq).to(device)
    Xq_te_gpu = torch.from_numpy(Xteq).to(device)
    ytr_gpu = torch.from_numpy(ytr).to(device)
    yte_gpu = torch.from_numpy(yte).to(device)
    F = Xtr.shape[1]

    codebook_kind = "thermometer" if kind == "binary" else "random_bipolar"
    proto_mode = "binary" if kind == "binary" else "cosine"
    pred_mode = "binary" if kind == "binary" else "cosine"

    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time()
    Htr, Hte = encode_and_bundle(Xq_tr_gpu, Xq_te_gpu, F, D, Q, device, seed,
                                 codebook_kind=codebook_kind)
    torch.cuda.synchronize()
    t_enc = time.time() - t0

    t_cls0 = time.time()
    protos = class_prototypes_gpu(Htr, ytr_gpu, n_classes, device,
                                   mode=proto_mode)
    yhat_tr = predict_gpu(Htr, protos, mode=pred_mode)
    torch.cuda.synchronize()
    t_train = time.time() - t0
    t_inf0 = time.time()
    yhat_te = predict_gpu(Hte, protos, mode=pred_mode)
    torch.cuda.synchronize()
    t_infer = time.time() - t_inf0
    train_acc = float((yhat_tr == ytr_gpu).float().mean().item())
    test_acc = float((yhat_te == yte_gpu).float().mean().item())
    peak_mb = torch.cuda.max_memory_allocated(device) / 1e6
    Nte = Xte.shape[0]
    flops = 2.0 * Nte * n_classes * D  # cosine/dot product proxy

    del Htr, Hte, protos, Xq_tr_gpu, Xq_te_gpu, ytr_gpu, yte_gpu
    del yhat_tr, yhat_te
    gc.collect(); torch.cuda.empty_cache()
    return {
        "test_acc": test_acc, "train_acc": train_acc,
        "wall_s": t_train + t_infer, "encode_s": t_enc, "train_s": t_train,
        "infer_s": t_infer,
        "latency_ms_per_sample": (t_infer / Nte) * 1000.0,
        "peak_mem_mb": peak_mb, "flops_proxy": flops,
    }


def run_rp_ridge_seed(Xtr, ytr, Xte, yte, device, D, seed, n_classes,
                      ridge_alpha=1.0):
    """Gaussian random projection -> ridge classifier (one-vs-rest)."""
    rng = torch.Generator(device=device); rng.manual_seed(seed)
    F = Xtr.shape[1]
    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time()
    # Project in chunks if D large to limit peak memory of R
    # Standardize on CPU then move
    mu = Xtr.mean(axis=0); sd = Xtr.std(axis=0); sd[sd < 1e-9] = 1.0
    Xtr_z = ((Xtr - mu) / sd).astype(np.float32)
    Xte_z = ((Xte - mu) / sd).astype(np.float32)
    Xtr_t = torch.from_numpy(Xtr_z).to(device)
    Xte_t = torch.from_numpy(Xte_z).to(device)

    # Random projection matrix R (F, D). For D=1M, F=561 -> 1.1 GB float32.
    # Project in column-chunks to keep peak memory bounded.
    chunk = min(D, 65536)
    Htr = torch.empty((Xtr_t.shape[0], D), dtype=torch.float32, device=device)
    Hte = torch.empty((Xte_t.shape[0], D), dtype=torch.float32, device=device)
    for c0 in range(0, D, chunk):
        c1 = min(c0 + chunk, D)
        R = torch.randn((F, c1 - c0), generator=rng, device=device,
                        dtype=torch.float32) / (F ** 0.5)
        Htr[:, c0:c1] = Xtr_t @ R
        Hte[:, c0:c1] = Xte_t @ R
        del R
    torch.cuda.synchronize()
    t_enc = time.time() - t0

    # Ridge in primal if D <= 4096 else dual via kernel trick
    Y = torch.zeros((Xtr.shape[0], n_classes), dtype=torch.float32, device=device)
    Y[torch.arange(Xtr.shape[0], device=device),
      torch.from_numpy(ytr).to(device)] = 1.0

    t_train0 = time.time()
    if D <= 4096:
        # Primal: W = (Ht^T Ht + alpha I)^-1 Ht^T Y
        A = Htr.T @ Htr
        A += ridge_alpha * torch.eye(D, device=device, dtype=torch.float32)
        B = Htr.T @ Y
        W = torch.linalg.solve(A, B)
        scores_te = Hte @ W
        scores_tr = Htr @ W
        del A, B, W
    else:
        # Dual: alpha = (K + alpha I)^-1 Y;  scores_te = K_te @ alpha
        Ntr = Xtr.shape[0]
        K = Htr @ Htr.T
        K += ridge_alpha * torch.eye(Ntr, device=device, dtype=torch.float32)
        alpha = torch.linalg.solve(K, Y)
        scores_te = Hte @ (Htr.T @ alpha)
        scores_tr = Htr @ (Htr.T @ alpha)
        del K, alpha
    torch.cuda.synchronize()
    t_train = t_enc + (time.time() - t_train0)

    t_inf0 = time.time()
    yhat_te = scores_te.argmax(dim=1).cpu().numpy()
    yhat_tr = scores_tr.argmax(dim=1).cpu().numpy()
    torch.cuda.synchronize()
    t_infer = time.time() - t_inf0
    test_acc = float((yhat_te == yte).mean())
    train_acc = float((yhat_tr == ytr).mean())
    peak_mb = torch.cuda.max_memory_allocated(device) / 1e6
    Nte = Xte.shape[0]
    flops = 2.0 * Nte * D * n_classes

    del Htr, Hte, Xtr_t, Xte_t, Y, scores_te, scores_tr
    gc.collect(); torch.cuda.empty_cache()
    return {
        "test_acc": test_acc, "train_acc": train_acc,
        "wall_s": t_train + t_infer, "encode_s": t_enc, "train_s": t_train,
        "infer_s": t_infer,
        "latency_ms_per_sample": (t_infer / Nte) * 1000.0,
        "peak_mem_mb": peak_mb, "flops_proxy": flops,
    }


def run_linear_ridge_raw(Xtr, ytr, Xte, yte, device, seed, n_classes,
                         ridge_alpha=1.0):
    """Linear ridge on raw 561 features (sanity baseline)."""
    F = Xtr.shape[1]
    mu = Xtr.mean(axis=0); sd = Xtr.std(axis=0); sd[sd < 1e-9] = 1.0
    Xtr_z = ((Xtr - mu) / sd).astype(np.float32)
    Xte_z = ((Xte - mu) / sd).astype(np.float32)
    Xt = torch.from_numpy(Xtr_z).to(device)
    Xe = torch.from_numpy(Xte_z).to(device)
    Y = torch.zeros((Xtr.shape[0], n_classes), dtype=torch.float32, device=device)
    Y[torch.arange(Xtr.shape[0], device=device),
      torch.from_numpy(ytr).to(device)] = 1.0
    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time()
    A = Xt.T @ Xt + ridge_alpha * torch.eye(F, device=device)
    B = Xt.T @ Y
    W = torch.linalg.solve(A, B)
    torch.cuda.synchronize()
    t_train = time.time() - t0
    t_inf0 = time.time()
    yhat = (Xe @ W).argmax(dim=1).cpu().numpy()
    yhat_tr = (Xt @ W).argmax(dim=1).cpu().numpy()
    torch.cuda.synchronize()
    t_infer = time.time() - t_inf0
    test_acc = float((yhat == yte).mean())
    train_acc = float((yhat_tr == ytr).mean())
    peak_mb = torch.cuda.max_memory_allocated(device) / 1e6
    Nte = Xte.shape[0]
    del Xt, Xe, Y, W, A, B
    gc.collect(); torch.cuda.empty_cache()
    return {
        "test_acc": test_acc, "train_acc": train_acc,
        "wall_s": t_train + t_infer, "train_s": t_train, "infer_s": t_infer,
        "latency_ms_per_sample": (t_infer / Nte) * 1000.0,
        "peak_mem_mb": peak_mb, "flops_proxy": 2.0 * Nte * F * n_classes,
    }


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------
def bootstrap_ci(values, n_boot=1000, alpha=0.05, seed=0):
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n = len(values)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_means[i] = values[idx].mean()
    lo = np.percentile(boot_means, 100 * alpha / 2)
    hi = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return float(values.mean()), float(lo), float(hi)


def paired_t_test(a, b):
    """Returns (t, p two-sided) for paired samples."""
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    if len(a) != len(b) or len(a) < 2:
        return float("nan"), float("nan")
    d = a - b
    n = len(d)
    md = d.mean(); sd = d.std(ddof=1)
    if sd < 1e-12:
        return float("inf") if md != 0 else 0.0, 0.0 if md != 0 else 1.0
    t = md / (sd / np.sqrt(n))
    # two-sided p via survival of Student's t — approximate w/ normal for n>=10
    # Use exact via scipy if available, else normal approx
    try:
        from scipy.stats import t as t_dist
        p = 2.0 * (1.0 - t_dist.cdf(abs(t), df=n - 1))
    except Exception:
        from math import erf, sqrt
        p = 2.0 * (1.0 - 0.5 * (1 + erf(abs(t) / sqrt(2))))
    return float(t), float(p)


def cohens_d(a, b):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    s = np.sqrt(((a.std(ddof=1) ** 2) + (b.std(ddof=1) ** 2)) / 2.0)
    if s < 1e-12:
        return float("inf") if (a.mean() - b.mean()) != 0 else 0.0
    return float((a.mean() - b.mean()) / s)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",
                   default="/home/daedalus/AMD_gfx1151_energy/data/uci_har/UCI HAR Dataset")
    p.add_argument("--surrogate",
                   default="/home/daedalus/AMD_gfx1151_energy/results/z278_mep2_surrogate_v3/surrogate_4d_v3.npz")
    p.add_argument("--out_dir", default="results/z374_uci_har_pubgrade")
    p.add_argument("--scales", type=int, nargs="+",
                   default=[1024, 4096, 16384, 65536, 262144, 1048576])
    p.add_argument("--nsram_max", type=int, default=16384,
                   help="cap NS-RAM scale (surrogate cost prohibitive beyond)")
    p.add_argument("--n_seeds", type=int, default=10)
    p.add_argument("--Q", type=int, default=32)
    p.add_argument("--T_steps", type=int, default=100)
    p.add_argument("--nsram_batch", type=int, default=128)
    p.add_argument("--n_boot", type=int, default=1000)
    args = p.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"

    def log(msg):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        with open(log_path, "a") as f:
            f.write(line + "\n")

    log(f"z374 start. scales={args.scales} n_seeds={args.n_seeds} "
        f"nsram_max={args.nsram_max}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device={device} torch={torch.__version__}")
    if device.type == "cuda":
        log(f"gpu={torch.cuda.get_device_name(0)}")

    # Load data
    log("loading UCI-HAR ...")
    Xtr, ytr, Xte, yte = load_uci_har(args.data_root)
    n_classes = int(max(ytr.max(), yte.max())) + 1
    log(f"loaded: Xtr={Xtr.shape} Xte={Xte.shape} n_classes={n_classes}")

    # Load surrogate (NS-RAM)
    log("loading NS-RAM surrogate ...")
    surr = load_surrogate(args.surrogate, device)

    all_results = []  # list of dicts (method, N, seed, ...)

    methods_per_scale = {
        "nsram_vdbit":    run_nsram_seed,
        "nsram_ablation": run_nsram_seed,  # ablation_no_vd=True
        "binary_hdc":     run_digital_hdc_seed,
        "bipolar_hdc":    run_digital_hdc_seed,
        "rp_ridge":       run_rp_ridge_seed,
    }

    # Linear ridge raw is N-independent; run once with 10 seeds (deterministic
    # except shuffle order — included for sanity).
    log("== Linear ridge on raw 561 features (N-independent baseline) ==")
    for seed in range(args.n_seeds):
        try:
            r = run_linear_ridge_raw(Xtr, ytr, Xte, yte, device, seed, n_classes)
            r.update({"method": "linear_ridge_raw", "N": 561, "seed": seed})
            all_results.append(r)
            log(f"  linear_ridge_raw seed={seed} acc={r['test_acc']:.4f} "
                f"wall={r['wall_s']:.2f}s peak={r['peak_mem_mb']:.0f}MB")
        except Exception as e:
            log(f"  linear_ridge_raw seed={seed} ERROR: {e}")
            traceback.print_exc()

    # Per-scale loop
    for N in args.scales:
        log(f"== N = {N} ==")
        for method, fn in methods_per_scale.items():
            if method.startswith("nsram") and N > args.nsram_max:
                log(f"  skip {method} N={N} (> nsram_max {args.nsram_max})")
                continue
            for seed in range(args.n_seeds):
                try:
                    if method == "nsram_vdbit":
                        r = run_nsram_seed(
                            Xtr, ytr, Xte, yte, surr, device, N, args.Q,
                            seed, n_classes, T_steps=args.T_steps,
                            batch_size=args.nsram_batch,
                            ablation_no_vd=False)
                    elif method == "nsram_ablation":
                        r = run_nsram_seed(
                            Xtr, ytr, Xte, yte, surr, device, N, args.Q,
                            seed, n_classes, T_steps=args.T_steps,
                            batch_size=args.nsram_batch,
                            ablation_no_vd=True)
                    elif method == "binary_hdc":
                        r = run_digital_hdc_seed(
                            Xtr, ytr, Xte, yte, device, N, args.Q, seed,
                            n_classes, kind="binary")
                    elif method == "bipolar_hdc":
                        r = run_digital_hdc_seed(
                            Xtr, ytr, Xte, yte, device, N, args.Q, seed,
                            n_classes, kind="bipolar")
                    elif method == "rp_ridge":
                        r = run_rp_ridge_seed(
                            Xtr, ytr, Xte, yte, device, N, seed, n_classes)
                    else:
                        continue
                    r.update({"method": method, "N": N, "seed": seed})
                    all_results.append(r)
                    log(f"  {method:18s} N={N:>7d} seed={seed} "
                        f"acc={r['test_acc']:.4f} wall={r['wall_s']:.2f}s")
                except torch.cuda.OutOfMemoryError as e:
                    log(f"  {method} N={N} seed={seed} OOM: {e}")
                    torch.cuda.empty_cache(); gc.collect()
                except Exception as e:
                    log(f"  {method} N={N} seed={seed} ERROR: {e}")
                    traceback.print_exc(); gc.collect()
                    torch.cuda.empty_cache()
            # checkpoint save
            with open(out_dir / "all_results.csv", "w", newline="") as f:
                if all_results:
                    w = csv.DictWriter(f, fieldnames=sorted(
                        {k for r in all_results for k in r.keys()}))
                    w.writeheader()
                    for r in all_results:
                        w.writerow(r)

    # ----------------------------------------------------------------
    # Aggregate
    # ----------------------------------------------------------------
    log("aggregating ...")
    summary = {
        "experiment": "z374_uci_har_pubgrade",
        "n_seeds": args.n_seeds, "Q": args.Q, "T_steps": args.T_steps,
        "scales": args.scales, "nsram_max": args.nsram_max,
        "n_classes": n_classes, "n_train": int(Xtr.shape[0]),
        "n_test": int(Xte.shape[0]),
        "device": str(device),
        "per_method_per_N": {},
        "stat_tests": {},
        "gates": {},
    }

    def collect(method, N):
        return [r for r in all_results
                if r.get("method") == method and r.get("N") == N]

    methods = ["nsram_vdbit", "nsram_ablation", "binary_hdc",
               "bipolar_hdc", "rp_ridge", "linear_ridge_raw"]
    for m in methods:
        summary["per_method_per_N"][m] = {}
        if m == "linear_ridge_raw":
            Ns = [561]
        else:
            Ns = args.scales
        for N in Ns:
            rs = collect(m, N)
            if not rs:
                continue
            accs = [r["test_acc"] for r in rs]
            walls = [r["wall_s"] for r in rs]
            lats = [r["latency_ms_per_sample"] for r in rs]
            peaks = [r.get("peak_mem_mb", float("nan")) for r in rs]
            flops = [r.get("flops_proxy", float("nan")) for r in rs]
            mean_a, lo_a, hi_a = bootstrap_ci(accs, args.n_boot, seed=42)
            summary["per_method_per_N"][m][str(N)] = {
                "n_seeds": len(rs),
                "acc_mean": mean_a, "acc_ci95_lo": lo_a, "acc_ci95_hi": hi_a,
                "acc_std": float(np.std(accs, ddof=1) if len(accs) > 1 else 0.0),
                "wall_s_mean": float(np.mean(walls)),
                "latency_ms_per_sample_mean": float(np.mean(lats)),
                "peak_mem_mb_mean": float(np.nanmean(peaks)),
                "flops_proxy_mean": float(np.nanmean(flops)),
            }

    # Paired t-tests at each N, NS-RAM vs each digital baseline, Bonferroni
    digital_baselines = ["binary_hdc", "bipolar_hdc", "rp_ridge"]
    n_tests = 0
    test_records = []
    for N in args.scales:
        nsram_rs = collect("nsram_vdbit", N)
        if not nsram_rs:
            continue
        nsram_by_seed = {r["seed"]: r["test_acc"] for r in nsram_rs}
        for b in digital_baselines:
            base_rs = collect(b, N)
            if not base_rs:
                continue
            base_by_seed = {r["seed"]: r["test_acc"] for r in base_rs}
            common = sorted(set(nsram_by_seed) & set(base_by_seed))
            if len(common) < 2:
                continue
            a = [nsram_by_seed[s] for s in common]
            c = [base_by_seed[s] for s in common]
            t, p = paired_t_test(a, c)
            d = cohens_d(a, c)
            test_records.append({
                "N": N, "baseline": b, "n": len(common),
                "nsram_mean": float(np.mean(a)),
                "baseline_mean": float(np.mean(c)),
                "delta": float(np.mean(a) - np.mean(c)),
                "t": t, "p_raw": p, "cohens_d": d,
            })
            n_tests += 1
    # Bonferroni
    for r in test_records:
        r["p_bonferroni"] = min(1.0, r["p_raw"] * max(1, n_tests))
        r["significant_p05"] = bool(r["p_bonferroni"] < 0.05
                                    and r["delta"] > 0)
    summary["stat_tests"] = {
        "tests": test_records,
        "n_tests_bonferroni": n_tests,
    }

    # Gates
    digital_max_N = max(args.scales)
    infra_pass = all(
        len(collect(m, digital_max_N)) >= 1
        for m in ["binary_hdc", "bipolar_hdc", "rp_ridge"]
    )
    discovery_pass = any(
        r["significant_p05"] and r["baseline"] == "binary_hdc"
        for r in test_records
    )
    ambitious_pass = any(
        r["significant_p05"] and r["baseline"] == "binary_hdc"
        and r["N"] >= 65536 and r["cohens_d"] > 0.5
        for r in test_records
    )
    summary["gates"] = {
        "INFRA_all_baselines_run_at_max_N": bool(infra_pass),
        "DISCOVERY_nsram_beats_binary_p05_bonferroni": bool(discovery_pass),
        "AMBITIOUS_nsram_beats_binary_N_ge_65536_d_gt_0.5": bool(ambitious_pass),
        "HONEST_report_regardless": True,
    }

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # CSV (final)
    with open(out_dir / "all_results.csv", "w", newline="") as f:
        if all_results:
            fields = sorted({k for r in all_results for k in r.keys()})
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in all_results:
                w.writerow(r)

    # ----------------------------------------------------------------
    # Plot
    # ----------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        method_styles = {
            "nsram_vdbit":     ("NS-RAM HDC (V_d-as-bit)", "tab:red",    "o"),
            "nsram_ablation":  ("NS-RAM (no V_d mod)",     "tab:orange", "s"),
            "binary_hdc":      ("Binary HDC (Imani)",      "tab:blue",   "^"),
            "bipolar_hdc":     ("Bipolar HDC (Kanerva)",   "tab:green",  "v"),
            "rp_ridge":        ("RP + Ridge",              "tab:purple", "D"),
        }

        # (a) Accuracy vs N
        ax = axes[0, 0]
        for m, (label, color, mk) in method_styles.items():
            Ns = []; means = []; los = []; his = []
            for N in args.scales:
                s = summary["per_method_per_N"][m].get(str(N))
                if s is None: continue
                Ns.append(N); means.append(s["acc_mean"])
                los.append(s["acc_ci95_lo"]); his.append(s["acc_ci95_hi"])
            if not Ns: continue
            yerr = [[m_ - lo for m_, lo in zip(means, los)],
                    [hi - m_ for m_, hi in zip(means, his)]]
            ax.errorbar(Ns, means, yerr=yerr, fmt=mk + "-",
                        color=color, label=label, capsize=3)
        # add linear baseline horizontal
        lr = summary["per_method_per_N"].get("linear_ridge_raw", {}).get("561")
        if lr:
            ax.axhline(lr["acc_mean"], color="gray", linestyle="--",
                       label=f"Linear ridge (raw 561) {lr['acc_mean']:.3f}")
        ax.set_xscale("log"); ax.set_xlabel("Hypervector dim N")
        ax.set_ylabel("Test accuracy"); ax.set_title("(a) Accuracy vs N")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

        # (b) Training time vs N
        ax = axes[0, 1]
        for m, (label, color, mk) in method_styles.items():
            Ns = []; ts = []
            for N in args.scales:
                s = summary["per_method_per_N"][m].get(str(N))
                if s is None: continue
                Ns.append(N); ts.append(s["wall_s_mean"])
            if Ns:
                ax.plot(Ns, ts, mk + "-", color=color, label=label)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("N"); ax.set_ylabel("Wall time (s)")
        ax.set_title("(b) Wall time vs N"); ax.legend(fontsize=8)
        ax.grid(alpha=0.3, which="both")

        # (c) FLOPs proxy vs N
        ax = axes[1, 0]
        for m, (label, color, mk) in method_styles.items():
            if m.startswith("nsram"):
                # FLOPs proxy: 2 * Nte * N_classes * N * T_steps (analog ops)
                Ns = []; fs = []
                for N in args.scales:
                    s = summary["per_method_per_N"][m].get(str(N))
                    if s is None: continue
                    Ns.append(N)
                    fs.append(2.0 * summary["n_test"] * n_classes * N
                              * args.T_steps)
                if Ns: ax.plot(Ns, fs, mk + "-", color=color, label=label)
            else:
                Ns = []; fs = []
                for N in args.scales:
                    s = summary["per_method_per_N"][m].get(str(N))
                    if s is None: continue
                    Ns.append(N); fs.append(s["flops_proxy_mean"])
                if Ns: ax.plot(Ns, fs, mk + "-", color=color, label=label)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("N"); ax.set_ylabel("FLOPs proxy (inference)")
        ax.set_title("(c) Energy/compute proxy vs N")
        ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")

        # (d) Ablation: V_d-as-bit ON vs OFF
        ax = axes[1, 1]
        Ns = []; on = []; off = []; on_lo = []; on_hi = []; off_lo = []; off_hi = []
        for N in args.scales:
            s_on = summary["per_method_per_N"]["nsram_vdbit"].get(str(N))
            s_off = summary["per_method_per_N"]["nsram_ablation"].get(str(N))
            if s_on is None or s_off is None: continue
            Ns.append(N)
            on.append(s_on["acc_mean"])
            on_lo.append(s_on["acc_ci95_lo"]); on_hi.append(s_on["acc_ci95_hi"])
            off.append(s_off["acc_mean"])
            off_lo.append(s_off["acc_ci95_lo"]); off_hi.append(s_off["acc_ci95_hi"])
        if Ns:
            ax.errorbar(Ns, on,
                        yerr=[[m_-lo for m_, lo in zip(on, on_lo)],
                              [hi-m_ for m_, hi in zip(on, on_hi)]],
                        fmt="o-", color="tab:red", label="V_d-as-bit ON",
                        capsize=3)
            ax.errorbar(Ns, off,
                        yerr=[[m_-lo for m_, lo in zip(off, off_lo)],
                              [hi-m_ for m_, hi in zip(off, off_hi)]],
                        fmt="s-", color="tab:orange",
                        label="V_d-as-bit OFF (ablation)", capsize=3)
        ax.set_xscale("log"); ax.set_xlabel("N")
        ax.set_ylabel("Test accuracy")
        ax.set_title("(d) Ablation: V_d-as-bit modulation")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(out_dir / "comparison_plot.png", dpi=150)
        plt.close()
        log("plot saved")
    except Exception as e:
        log(f"plot ERROR: {e}")
        traceback.print_exc()

    log(f"DONE. gates: {summary['gates']}")
    log(f"summary -> {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
