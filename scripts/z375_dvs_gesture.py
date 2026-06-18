"""z375 — DVS-Gesture-128×128 production benchmark.

NS-RAM event-coded reservoir (N=16384) vs LIF baseline vs LSTM vs
random-projection ridge baseline.

Data:
  - Preferred: tonic.datasets.DVSGesture (real IBM DVS128, 11 classes).
  - Fallback: synthetic 128×128 DVS-proxy (Poisson events with class-specific
    centers / velocities / temporal profiles). Used when real DVS download
    is unavailable (Figshare WAF block / no HF auth).

Methods (all on daedalus GPU):
  1. NS-RAM reservoir  — 128² = 16384 cells, event drives VG1.
  2. LIF baseline      — recurrent leaky integrate-fire, same N.
  3. LSTM baseline     — 2-layer LSTM hidden=256.
  4. Random projection — fixed W_in projection + pooled rate + ridge.

Pre-registered gates:
  INFRA        : all 4 methods train without OOM at N=16384.
  DISCOVERY    : NS-RAM > random_projection by >= 5pp.
  AMBITIOUS    : NS-RAM >= LIF  AND  NS-RAM >= LSTM - 1pp.
  KILL-SHOT    : random_projection >= NS-RAM.
"""
from __future__ import annotations
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")

import json
import math
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "results" / "z375_dvs_gesture"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = OUT_DIR / "run.log"


def log(msg: str):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


# ---------- config ----------
N_CLASSES         = 11
SPATIAL           = 128                # full 128x128 native DVS resolution
INPUT_DIM         = SPATIAL * SPATIAL  # 16384
T_BINS            = 24                 # time bins per sample
SAMPLE_DURATION_S = 1.5
BIN_DT_S          = SAMPLE_DURATION_S / T_BINS
SEEDS             = [0, 1, 2, 3, 4]

# Reservoir scale — PRODUCTION
NSRAM_N           = 16384              # 128² cells

# Synthetic fallback dataset sizes (kept reasonable for 3h budget)
N_PER_CLASS_TRAIN_SYNTH = 80
N_PER_CLASS_TEST_SYNTH  = 20
# Real DVSGesture caps (if available)
N_PER_CLASS_TRAIN_REAL  = 80
N_PER_CLASS_TEST_REAL   = 20

NSRAM_CELL = {"C_b_fF": 80.0, "V_G2_bias": 0.35, "g_in": 0.4}
NSRAM_DT_PER_BIN_S = 2e-6      # effective integration step per bin (calibrated like z290)
NSRAM_VG1_LO, NSRAM_VG1_HI = 0.25, 0.55
NSRAM_SUB_STEPS   = 2
N_POOL_CHUNKS     = 4
SURROGATE_PATH    = "results/z278_mep2_surrogate_v3/surrogate_4d_v3.npz"
RIDGE_ALPHA       = 1.0

# LIF baseline params
LIF_TAU_S = 0.020
LIF_V_THR = 1.0
LIF_INPUT_GAIN = 1.0

# LSTM baseline params
LSTM_HIDDEN = 256
LSTM_LAYERS = 2
LSTM_EPOCHS = 12
LSTM_LR     = 1e-3
LSTM_BATCH  = 16


# ============================================================
# Dataset loading
# ============================================================

def try_load_real_dvs(save_to: Path):
    try:
        from tonic.datasets import DVSGesture
    except Exception as e:
        log(f"[dvs] tonic import failed: {e}")
        return None
    try:
        t0 = time.time()
        train = DVSGesture(save_to=str(save_to), train=True)
        test  = DVSGesture(save_to=str(save_to), train=False)
        log(f"[dvs] real DVSGesture loaded in {time.time()-t0:.1f}s "
            f"(train={len(train)}, test={len(test)})")
        return {"train": train, "test": test}
    except Exception as e:
        log(f"[dvs] real DVSGesture unavailable: {e}")
        return None


def real_dvs_to_tensor(ds, n_per_class: int, rng: np.random.Generator):
    by_class = {c: [] for c in range(N_CLASSES)}
    for i in range(len(ds)):
        _, lbl = ds[i]
        if lbl in by_class and len(by_class[lbl]) < n_per_class:
            by_class[lbl].append(i)
    Xs, ys = [], []
    SX = 128
    pix = SX // SPATIAL  # =1 at SPATIAL=128
    for c, idxs in by_class.items():
        for idx in idxs:
            ev, lbl = ds[idx]
            if len(ev) == 0:
                arr = np.zeros((T_BINS, SPATIAL, SPATIAL), dtype=np.float32)
            else:
                t_us = ev["t"].astype(np.int64)
                t0 = t_us.min(); t1 = t_us.max()
                dur = max(t1 - t0, 1)
                t_norm = (t_us - t0) * T_BINS // (dur + 1)
                t_norm = np.clip(t_norm, 0, T_BINS - 1)
                xs = (ev["x"].astype(np.int64) // pix).clip(0, SPATIAL - 1)
                ysp = (ev["y"].astype(np.int64) // pix).clip(0, SPATIAL - 1)
                arr = np.zeros((T_BINS, SPATIAL, SPATIAL), dtype=np.float32)
                np.add.at(arr, (t_norm, xs, ysp), 1.0)
            Xs.append(arr); ys.append(c)
    Xs = np.stack(Xs, axis=0)
    ys = np.asarray(ys, dtype=np.int64)
    perm = rng.permutation(len(ys))
    return Xs[perm], ys[perm]


def synth_dvs(n_per_class_train: int, n_per_class_test: int, seed: int):
    """128×128 synthetic DVS-proxy with class-specific Poisson event signatures."""
    rng = np.random.default_rng(seed)
    class_params = []
    cx0 = SPATIAL / 2.0
    cy0 = SPATIAL / 2.0
    for c in range(N_CLASSES):
        cx = cx0 + rng.normal(0, 6.0)
        cy = cy0 + rng.normal(0, 6.0)
        angle = 2 * math.pi * c / N_CLASSES + rng.uniform(-0.15, 0.15)
        vx = 28.0 * math.cos(angle)
        vy = 28.0 * math.sin(angle)
        sigma = rng.uniform(12.0, 18.0)
        t_peak = rng.uniform(0.35, 0.65)
        t_width = rng.uniform(0.20, 0.40)
        rate_peak = rng.uniform(5500, 6500)
        class_params.append(dict(cx=cx, cy=cy, vx=vx, vy=vy, sigma=sigma,
                                 t_peak=t_peak, t_width=t_width, rate=rate_peak))

    def make_split(n_per_class, split_seed):
        rs = np.random.default_rng(seed + split_seed)
        Xs, ys = [], []
        for c, p in enumerate(class_params):
            for _ in range(n_per_class):
                cx = p["cx"] + rs.normal(0, 10.0)
                cy = p["cy"] + rs.normal(0, 10.0)
                vx = p["vx"] * rs.uniform(0.6, 1.4) + rs.normal(0, 3.0)
                vy = p["vy"] * rs.uniform(0.6, 1.4) + rs.normal(0, 3.0)
                sigma = p["sigma"] * rs.uniform(0.85, 1.2)
                t_peak = p["t_peak"] + rs.normal(0, 0.10)
                t_width = p["t_width"] * rs.uniform(0.7, 1.4)
                rate = p["rate"] * rs.uniform(0.7, 1.3)
                n_events = int(rate)
                ts = rs.normal(t_peak, t_width, size=n_events)
                ts = np.clip(ts, 0, 0.999)
                xs = cx + vx * (ts - 0.5) + rs.normal(0, sigma, size=n_events)
                ysp = cy + vy * (ts - 0.5) + rs.normal(0, sigma, size=n_events)
                t_idx = (ts * T_BINS).astype(np.int64).clip(0, T_BINS - 1)
                x_idx = xs.astype(np.int64).clip(0, SPATIAL - 1)
                y_idx = ysp.astype(np.int64).clip(0, SPATIAL - 1)
                arr = np.zeros((T_BINS, SPATIAL, SPATIAL), dtype=np.float32)
                np.add.at(arr, (t_idx, x_idx, y_idx), 1.0)
                # 20% background noise
                n_bg = int(0.20 * rate)
                tb = rs.integers(0, T_BINS, size=n_bg)
                xb = rs.integers(0, SPATIAL, size=n_bg)
                yb = rs.integers(0, SPATIAL, size=n_bg)
                np.add.at(arr, (tb, xb, yb), 1.0)
                Xs.append(arr); ys.append(c)
        Xs = np.stack(Xs, axis=0)
        ys = np.asarray(ys, dtype=np.int64)
        perm = rs.permutation(len(ys))
        return Xs[perm], ys[perm]

    Xtr, ytr = make_split(n_per_class_train, split_seed=1)
    Xte, yte = make_split(n_per_class_test,  split_seed=2)
    return Xtr, ytr, Xte, yte


# ============================================================
# NS-RAM surrogate (4D LUT, trilinear-on-4-axes interpolation)
# ============================================================

def load_surrogate(path, device):
    z = np.load(path)
    return {
        "I_d":   torch.tensor(z["Id"],     dtype=torch.float32, device=device),
        "I_ii":  torch.tensor(z["Iii"],    dtype=torch.float32, device=device),
        "I_leak":torch.tensor(z["Ileak"],  dtype=torch.float32, device=device),
        "ax_VG1":torch.tensor(z["vg1_axis"], dtype=torch.float32, device=device),
        "ax_VG2":torch.tensor(z["vg2_axis"], dtype=torch.float32, device=device),
        "ax_Vd": torch.tensor(z["vd_axis"],  dtype=torch.float32, device=device),
        "ax_Vb": torch.tensor(z["vb_axis"],  dtype=torch.float32, device=device),
    }


def _frac_index(values, axis):
    n = axis.shape[0]
    i = torch.bucketize(values, axis) - 1
    i = i.clamp(0, n - 2)
    lo = axis[i]; hi = axis[i + 1]
    t = (values - lo) / (hi - lo + 1e-20)
    return i, t.clamp(0.0, 1.0)


def query_surrogate(surr, VG1, VG2, Vd, Vb):
    i0, t0 = _frac_index(VG1, surr["ax_VG1"])
    i1, t1 = _frac_index(VG2, surr["ax_VG2"])
    i2, t2 = _frac_index(Vd,  surr["ax_Vd"])
    i3, t3 = _frac_index(Vb,  surr["ax_Vb"])
    Id_tbl, Iii_tbl, Ilk_tbl = surr["I_d"], surr["I_ii"], surr["I_leak"]
    Id_out = torch.zeros_like(VG1); Iii_out = torch.zeros_like(VG1); Ilk_out = torch.zeros_like(VG1)
    for a0 in (0, 1):
        w0 = t0 if a0 else (1 - t0); j0 = i0 + a0
        for a1 in (0, 1):
            w1 = t1 if a1 else (1 - t1); j1 = i1 + a1
            for a2 in (0, 1):
                w2 = t2 if a2 else (1 - t2); j2 = i2 + a2
                for a3 in (0, 1):
                    w3 = t3 if a3 else (1 - t3); j3 = i3 + a3
                    w = w0 * w1 * w2 * w3
                    Id_out  = Id_out  + w * Id_tbl[j0, j1, j2, j3]
                    Iii_out = Iii_out + w * Iii_tbl[j0, j1, j2, j3]
                    Ilk_out = Ilk_out + w * Ilk_tbl[j0, j1, j2, j3]
    return Id_out, Iii_out, Ilk_out


# ============================================================
# NS-RAM reservoir over event tensor — N=16384, batched
# ============================================================

def nsram_features_events(X, N, W_in, V_G1_bias, V_G2_bias, surr,
                          g_in, C_b_F, dt_s, vd=1.0,
                          sub_steps=NSRAM_SUB_STEPS,
                          n_pool_chunks=N_POOL_CHUNKS, return_diag=False,
                          batch=8):
    """X: (B, T_BINS, INPUT_DIM). Process in batch chunks to keep memory bounded.
       Returns pooled features (B, F).
    """
    device = X.device
    B, T_in, D = X.shape
    Vb_min = surr["ax_Vb"][0]; Vb_max = surr["ax_Vb"][-1]
    VG1_min = surr["ax_VG1"][0]; VG1_max = surr["ax_VG1"][-1]

    chunk_len = max(T_in // n_pool_chunks, 1)
    feat_dim = N * (2 * n_pool_chunks + 1)
    all_feats = torch.empty(B, feat_dim, device=device)
    diag = {"vb_excursion_mean": 0.0, "vb_excursion_max": 0.0,
            "vb_final_mean": 0.0, "vb_final_std": 0.0,
            "spikes_per_inf": 0.0}
    n_batches = 0

    for bi in range(0, B, batch):
        be = min(bi + batch, B)
        Xb = X[bi:be]
        bb = be - bi
        Vb = torch.zeros(bb, N, device=device)
        VG2_2d = V_G2_bias.expand(bb, N)
        Vd_2d  = torch.full((bb, N), float(vd), device=device)

        vb_chunks = [torch.zeros(bb, N, device=device) for _ in range(n_pool_chunks)]
        id_chunks = [torch.zeros(bb, N, device=device) for _ in range(n_pool_chunks)]
        chunk_counts = [0] * n_pool_chunks

        Xn = Xb / (Xb.mean(dim=(1, 2), keepdim=True) + 1e-3)

        vb_excursion = torch.zeros(bb, N, device=device)
        spike_count_total = torch.zeros(bb, device=device)

        for fi in range(T_in):
            ev = Xn[:, fi, :]                                  # (bb, D)
            spike_count_total = spike_count_total + Xb[:, fi, :].sum(dim=1)
            ck = min(fi // chunk_len, n_pool_chunks - 1)
            drive_frame = (ev - ev.mean(dim=1, keepdim=True)) @ W_in.T   # (bb, N)
            for s in range(sub_steps):
                VG1 = V_G1_bias.unsqueeze(0) + g_in * drive_frame
                VG1 = VG1.clamp(VG1_min, VG1_max)
                Vb_c = Vb.clamp(Vb_min, Vb_max)
                I_d, I_ii, I_leak = query_surrogate(surr, VG1, VG2_2d, Vd_2d, Vb_c)
                Vb = (Vb + dt_s * (I_ii - I_leak) / C_b_F).clamp(Vb_min, Vb_max)
                vb_excursion = torch.maximum(vb_excursion, Vb.abs())
                vb_chunks[ck] = vb_chunks[ck] + Vb
                id_chunks[ck] = id_chunks[ck] + I_d.abs()
                chunk_counts[ck] += 1

        parts = []
        for k in range(n_pool_chunks):
            c = max(chunk_counts[k], 1)
            parts.append(vb_chunks[k] / c)
            parts.append((id_chunks[k] / c + 1e-18).log10())
        parts.append(Vb)
        feats_b = torch.cat(parts, dim=1)
        all_feats[bi:be] = feats_b

        diag["vb_excursion_mean"] += vb_excursion.mean().item()
        diag["vb_excursion_max"]  = max(diag["vb_excursion_max"], vb_excursion.max().item())
        diag["vb_final_mean"]    += Vb.mean().item()
        diag["vb_final_std"]     += Vb.std().item()
        diag["spikes_per_inf"]   += spike_count_total.mean().item()
        n_batches += 1

    # global normalize
    mu = all_feats.mean(dim=0, keepdim=True); sd = all_feats.std(dim=0, keepdim=True) + 1e-9
    all_feats = (all_feats - mu) / sd

    for k in ("vb_excursion_mean", "vb_final_mean", "vb_final_std", "spikes_per_inf"):
        diag[k] /= max(n_batches, 1)
    return (all_feats, diag) if return_diag else (all_feats, None)


# ============================================================
# LIF baseline (recurrent leaky integrate-fire, N=16384)
# ============================================================

def lif_features(X, N, W_in, dt_s, tau_s, V_thr,
                 n_pool_chunks=N_POOL_CHUNKS, batch=8):
    device = X.device
    B, T_in, D = X.shape
    alpha = math.exp(-dt_s / tau_s)
    chunk_len = max(T_in // n_pool_chunks, 1)
    feat_dim = N * (n_pool_chunks + 1)
    all_feats = torch.empty(B, feat_dim, device=device)
    for bi in range(0, B, batch):
        be = min(bi + batch, B)
        Xb = X[bi:be]
        bb = be - bi
        V = torch.zeros(bb, N, device=device)
        rate_chunks = [torch.zeros(bb, N, device=device) for _ in range(n_pool_chunks)]
        chunk_counts = [0] * n_pool_chunks
        Xn = Xb / (Xb.mean(dim=(1, 2), keepdim=True) + 1e-3)
        for fi in range(T_in):
            ev = Xn[:, fi, :]
            ck = min(fi // chunk_len, n_pool_chunks - 1)
            drive = (ev - ev.mean(dim=1, keepdim=True)) @ W_in.T
            V = alpha * V + (1 - alpha) * (LIF_INPUT_GAIN * drive)
            spikes = (V >= V_thr).float()
            V = torch.where(spikes > 0, torch.zeros_like(V), V)
            rate_chunks[ck] = rate_chunks[ck] + spikes
            chunk_counts[ck] += 1
        parts = []
        for k in range(n_pool_chunks):
            c = max(chunk_counts[k], 1)
            parts.append(rate_chunks[k] / c)
        parts.append(V)
        all_feats[bi:be] = torch.cat(parts, dim=1)
    mu = all_feats.mean(dim=0, keepdim=True); sd = all_feats.std(dim=0, keepdim=True) + 1e-9
    return (all_feats - mu) / sd


# ============================================================
# Random projection baseline (fixed W_in, pooled mean of |drive|)
# ============================================================

def random_projection_features(X, N, W_in, n_pool_chunks=N_POOL_CHUNKS, batch=8):
    device = X.device
    B, T_in, D = X.shape
    chunk_len = max(T_in // n_pool_chunks, 1)
    feat_dim = N * n_pool_chunks
    all_feats = torch.empty(B, feat_dim, device=device)
    for bi in range(0, B, batch):
        be = min(bi + batch, B)
        Xb = X[bi:be]
        bb = be - bi
        Xn = Xb / (Xb.mean(dim=(1, 2), keepdim=True) + 1e-3)
        pool_chunks = [torch.zeros(bb, N, device=device) for _ in range(n_pool_chunks)]
        cnt = [0] * n_pool_chunks
        for fi in range(T_in):
            ev = Xn[:, fi, :]
            ck = min(fi // chunk_len, n_pool_chunks - 1)
            drive = (ev - ev.mean(dim=1, keepdim=True)) @ W_in.T
            pool_chunks[ck] = pool_chunks[ck] + torch.tanh(drive)
            cnt[ck] += 1
        parts = []
        for k in range(n_pool_chunks):
            parts.append(pool_chunks[k] / max(cnt[k], 1))
        all_feats[bi:be] = torch.cat(parts, dim=1)
    mu = all_feats.mean(dim=0, keepdim=True); sd = all_feats.std(dim=0, keepdim=True) + 1e-9
    return (all_feats - mu) / sd


# ============================================================
# Ridge readout (chunked Gram to keep memory bounded at N=16384)
# ============================================================

def ridge_lstsq(X, y, alpha=RIDGE_ALPHA, n_classes=N_CLASSES):
    Y = F.one_hot(y, n_classes).float()
    F_dim = X.shape[1]
    # X.T @ X is (F, F); at F=16384*9 ~150k this would OOM. We instead project
    # to a smaller random subspace ONLY if F > 8192 (keeps reservoir richness
    # at 16384 then random-projects features for readout).
    if F_dim > 8192:
        with torch.no_grad():
            P = torch.randn(F_dim, 8192, device=X.device) / math.sqrt(F_dim)
            X = X @ P
            F_dim = 8192
    A = X.T @ X + alpha * torch.eye(F_dim, device=X.device)
    Bm = X.T @ Y
    try:
        W = torch.linalg.solve(A, Bm)
    except Exception:
        W = torch.linalg.lstsq(A, Bm).solution
    return W, (P if 'P' in locals() else None)


def ridge_predict(X, W, P):
    if P is not None:
        X = X @ P
    return (X @ W).argmax(dim=1)


# ============================================================
# LSTM baseline
# ============================================================

class LSTMHead(nn.Module):
    def __init__(self, input_dim, hidden=LSTM_HIDDEN, layers=LSTM_LAYERS, n_classes=N_CLASSES):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, num_layers=layers, batch_first=True,
                            dropout=0.2 if layers > 1 else 0.0)
        self.fc = nn.Linear(hidden, n_classes)

    def forward(self, x):
        # x: (B, T, D); downsample spatial via mean-pool to keep tractable
        h, _ = self.lstm(x)
        return self.fc(h[:, -1])


def train_lstm(Xtr, ytr, Xte, yte, device, seed):
    """Xtr/Xte are (B, T_BINS, SPATIAL, SPATIAL). We downsample to (B,T,16*16)
    via avg pool — even with 98GB VRAM, full 16384-D input to LSTM is wasteful
    and the spatial structure of 8x8 patches is the standard recipe.
    """
    torch.manual_seed(seed)
    # avg-pool 128->16 (stride 8)
    def prep(X):
        B, T, S, _ = X.shape
        Xt = X.view(B * T, 1, S, S)
        Xt = F.avg_pool2d(Xt, kernel_size=8, stride=8)  # -> 16x16
        return Xt.view(B, T, -1)
    Xtr_l = prep(Xtr); Xte_l = prep(Xte)
    input_dim = Xtr_l.shape[-1]
    model = LSTMHead(input_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LSTM_LR)
    best_acc = 0.0
    losses = []
    for ep in range(LSTM_EPOCHS):
        model.train()
        perm = torch.randperm(Xtr_l.shape[0], device=device)
        ep_loss = 0.0; nb = 0
        for i in range(0, Xtr_l.shape[0], LSTM_BATCH):
            idx = perm[i:i + LSTM_BATCH]
            xb = Xtr_l[idx]; yb = ytr[idx]
            opt.zero_grad()
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            opt.step()
            ep_loss += loss.item(); nb += 1
        model.eval()
        with torch.no_grad():
            preds = []
            for i in range(0, Xte_l.shape[0], LSTM_BATCH):
                preds.append(model(Xte_l[i:i + LSTM_BATCH]).argmax(dim=1))
            yh = torch.cat(preds)
            acc = (yh == yte).float().mean().item()
        losses.append(ep_loss / max(nb, 1))
        if acc > best_acc:
            best_acc = acc
            best_pred = yh.detach().cpu().numpy()
        log(f"      [lstm] ep {ep+1}/{LSTM_EPOCHS} loss={losses[-1]:.4f} test_acc={acc:.4f}")
    return best_acc, best_pred


# ============================================================
# Per-seed run
# ============================================================

def run_seed(seed, Xtr_np, ytr_np, Xte_np, yte_np, device, surr):
    torch.manual_seed(seed)
    g = torch.Generator(device=device).manual_seed(seed + 31337)

    # tensors
    Xtr_full = torch.tensor(Xtr_np, device=device)  # (B,T,S,S)
    Xte_full = torch.tensor(Xte_np, device=device)
    Xtr = Xtr_full.view(-1, T_BINS, INPUT_DIM)
    Xte = Xte_full.view(-1, T_BINS, INPUT_DIM)
    ytr = torch.tensor(ytr_np, device=device, dtype=torch.long)
    yte = torch.tensor(yte_np, device=device, dtype=torch.long)

    # Shared W_in across NS-RAM / LIF / random projection
    W_in = torch.randn(NSRAM_N, INPUT_DIM, generator=g, device=device)
    W_in = W_in / (W_in.norm(dim=1, keepdim=True) + 1e-9)

    out = {"seed": seed, "n_train": int(Xtr.shape[0]), "n_test": int(Xte.shape[0])}

    # ---- Random projection ----
    t0 = time.time()
    f_tr = random_projection_features(Xtr, NSRAM_N, W_in)
    f_te = random_projection_features(Xte, NSRAM_N, W_in)
    W, P = ridge_lstsq(f_tr, ytr)
    rp_pred = ridge_predict(f_te, W, P)
    out["random_acc"] = (rp_pred == yte).float().mean().item()
    out["random_wall_s"] = time.time() - t0
    out["random_pred"] = rp_pred.detach().cpu().numpy().tolist()
    log(f"    seed={seed} RANDOM_PROJ acc={out['random_acc']:.4f} ({out['random_wall_s']:.1f}s)")
    del f_tr, f_te, W
    torch.cuda.empty_cache()

    # ---- LIF baseline ----
    t0 = time.time()
    f_tr = lif_features(Xtr, NSRAM_N, W_in, BIN_DT_S / NSRAM_SUB_STEPS, LIF_TAU_S, LIF_V_THR)
    f_te = lif_features(Xte, NSRAM_N, W_in, BIN_DT_S / NSRAM_SUB_STEPS, LIF_TAU_S, LIF_V_THR)
    W, P = ridge_lstsq(f_tr, ytr)
    lif_pred = ridge_predict(f_te, W, P)
    out["lif_acc"] = (lif_pred == yte).float().mean().item()
    out["lif_wall_s"] = time.time() - t0
    out["lif_pred"] = lif_pred.detach().cpu().numpy().tolist()
    log(f"    seed={seed} LIF         acc={out['lif_acc']:.4f} ({out['lif_wall_s']:.1f}s)")
    del f_tr, f_te, W
    torch.cuda.empty_cache()

    # ---- NS-RAM reservoir ----
    V_G1_bias = torch.empty(NSRAM_N, device=device).uniform_(NSRAM_VG1_LO, NSRAM_VG1_HI, generator=g)
    V_G2_bias = torch.full((NSRAM_N,), NSRAM_CELL["V_G2_bias"], device=device)
    t0 = time.time()
    f_tr, _ = nsram_features_events(Xtr, NSRAM_N, W_in, V_G1_bias, V_G2_bias, surr,
                                    g_in=NSRAM_CELL["g_in"],
                                    C_b_F=NSRAM_CELL["C_b_fF"] * 1e-15,
                                    dt_s=NSRAM_DT_PER_BIN_S, return_diag=False)
    f_te, diag = nsram_features_events(Xte, NSRAM_N, W_in, V_G1_bias, V_G2_bias, surr,
                                       g_in=NSRAM_CELL["g_in"],
                                       C_b_F=NSRAM_CELL["C_b_fF"] * 1e-15,
                                       dt_s=NSRAM_DT_PER_BIN_S, return_diag=True)
    W, P = ridge_lstsq(f_tr, ytr)
    ns_pred = ridge_predict(f_te, W, P)
    out["nsram_acc"]   = (ns_pred == yte).float().mean().item()
    out["nsram_wall_s"]= time.time() - t0
    out["nsram_diag"]  = diag
    out["nsram_pred"]  = ns_pred.detach().cpu().numpy().tolist()
    log(f"    seed={seed} NSRAM       acc={out['nsram_acc']:.4f} ({out['nsram_wall_s']:.1f}s) "
        f"vb_excur={diag['vb_excursion_mean']:.3e}")
    del f_tr, f_te, W
    torch.cuda.empty_cache()

    # ---- LSTM baseline ----
    t0 = time.time()
    lstm_acc, lstm_pred = train_lstm(Xtr_full, ytr, Xte_full, yte, device, seed)
    out["lstm_acc"] = lstm_acc
    out["lstm_wall_s"] = time.time() - t0
    out["lstm_pred"] = lstm_pred.tolist()
    log(f"    seed={seed} LSTM        acc={out['lstm_acc']:.4f} ({out['lstm_wall_s']:.1f}s)")

    out["y_test"] = yte.detach().cpu().numpy().tolist()
    return out


# ============================================================
# Aggregation + plots
# ============================================================

def confusion(y_true, y_pred, n_classes=N_CLASSES):
    C = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        C[int(t), int(p)] += 1
    return C


def ci95(xs):
    xs = np.asarray(xs, dtype=np.float64)
    mu = xs.mean()
    sd = xs.std(ddof=1) if len(xs) > 1 else 0.0
    se = sd / math.sqrt(max(len(xs), 1))
    return mu, 1.96 * se


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device = {device}")
    if device.type == "cuda":
        free, tot = torch.cuda.mem_get_info(0)
        log(f"GPU = {torch.cuda.get_device_name(0)}  free={free/1e9:.1f}GB / total={tot/1e9:.1f}GB")

    log(f"config: SPATIAL={SPATIAL} INPUT_DIM={INPUT_DIM} T_BINS={T_BINS} "
        f"N_NSRAM={NSRAM_N} SEEDS={SEEDS}")

    # ---------- Dataset ----------
    data_dir = ROOT / "data" / "dvs_gesture"
    data_dir.mkdir(parents=True, exist_ok=True)
    real = try_load_real_dvs(data_dir)
    rng = np.random.default_rng(42)
    if real is not None:
        Xtr_np, ytr_np = real_dvs_to_tensor(real["train"], N_PER_CLASS_TRAIN_REAL, rng)
        Xte_np, yte_np = real_dvs_to_tensor(real["test"],  N_PER_CLASS_TEST_REAL,  rng)
        data_source = "real_dvsgesture"
    else:
        log("[dvs] falling back to synthetic 128x128 DVS-proxy")
        Xtr_np, ytr_np, Xte_np, yte_np = synth_dvs(
            N_PER_CLASS_TRAIN_SYNTH, N_PER_CLASS_TEST_SYNTH, seed=42)
        data_source = "synthetic_dvs_proxy_128x128"
    log(f"data: source={data_source} train={Xtr_np.shape} test={Xte_np.shape} "
        f"mean_events/sample={Xtr_np.sum(axis=(1,2,3)).mean():.0f}")

    # ---------- Surrogate ----------
    surr = load_surrogate(SURROGATE_PATH, device)
    log(f"surrogate loaded: VG1={tuple(surr['ax_VG1'].shape)} VG2={tuple(surr['ax_VG2'].shape)} "
        f"Vd={tuple(surr['ax_Vd'].shape)} Vb={tuple(surr['ax_Vb'].shape)}")

    # ---------- Run all seeds ----------
    per_seed = []
    for s in SEEDS:
        log(f"==== seed {s} ====")
        try:
            r = run_seed(s, Xtr_np, ytr_np, Xte_np, yte_np, device, surr)
            per_seed.append(r)
        except Exception as e:
            log(f"  seed {s} FAILED: {e}\n{traceback.format_exc()}")
            continue

    # ---------- Aggregate ----------
    methods = ["random", "lif", "nsram", "lstm"]
    accs = {m: [r[f"{m}_acc"] for r in per_seed] for m in methods}
    walls = {m: [r[f"{m}_wall_s"] for r in per_seed] for m in methods}

    summary = {
        "data_source": data_source,
        "n_seeds": len(per_seed),
        "config": {
            "SPATIAL": SPATIAL, "INPUT_DIM": INPUT_DIM, "T_BINS": T_BINS,
            "NSRAM_N": NSRAM_N, "SEEDS": SEEDS,
            "n_per_class_train": (N_PER_CLASS_TRAIN_REAL if data_source.startswith("real") else N_PER_CLASS_TRAIN_SYNTH),
            "n_per_class_test":  (N_PER_CLASS_TEST_REAL  if data_source.startswith("real") else N_PER_CLASS_TEST_SYNTH),
            "lstm": {"hidden": LSTM_HIDDEN, "layers": LSTM_LAYERS, "epochs": LSTM_EPOCHS},
        },
        "per_seed": [{k: v for k, v in r.items() if not k.endswith("_pred") and k != "y_test"}
                     for r in per_seed],
        "accuracy": {},
        "wall_time_s": {},
    }
    for m in methods:
        mu, ci = ci95(accs[m])
        summary["accuracy"][m] = {"mean": mu, "ci95": ci, "all": accs[m]}
        summary["wall_time_s"][m] = {"mean": float(np.mean(walls[m])), "all": walls[m]}

    # ---------- Pre-registered gates ----------
    ns_mean = summary["accuracy"]["nsram"]["mean"]
    rp_mean = summary["accuracy"]["random"]["mean"]
    lif_mean = summary["accuracy"]["lif"]["mean"]
    lstm_mean = summary["accuracy"]["lstm"]["mean"]

    infra_pass = all(len(accs[m]) == len(SEEDS) for m in methods)
    discovery_pass = (ns_mean - rp_mean) >= 0.05
    ambitious_pass = (ns_mean >= lif_mean) and (ns_mean >= lstm_mean - 0.01)
    kill_shot = (rp_mean >= ns_mean)

    summary["gates"] = {
        "INFRA":        bool(infra_pass),
        "DISCOVERY":    bool(discovery_pass),
        "AMBITIOUS":    bool(ambitious_pass),
        "KILL_SHOT":    bool(kill_shot),
        "margin_nsram_vs_random": float(ns_mean - rp_mean),
        "margin_nsram_vs_lif":    float(ns_mean - lif_mean),
        "margin_nsram_vs_lstm":   float(ns_mean - lstm_mean),
    }

    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log(f"wrote {OUT_DIR/'summary.json'}")

    # ---------- accuracy_table.csv ----------
    csv_lines = ["method,mean_acc,ci95,wall_mean_s,seeds_run"]
    for m in methods:
        a = summary["accuracy"][m]
        w = summary["wall_time_s"][m]
        csv_lines.append(f"{m},{a['mean']:.4f},{a['ci95']:.4f},{w['mean']:.1f},{len(a['all'])}")
    with open(OUT_DIR / "accuracy_table.csv", "w") as f:
        f.write("\n".join(csv_lines) + "\n")
    log(f"wrote {OUT_DIR/'accuracy_table.csv'}")

    # ---------- confusion matrices ----------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        y_true_all = np.concatenate([np.asarray(r["y_test"]) for r in per_seed])
        for ax, m in zip(axes, methods):
            preds = np.concatenate([np.asarray(r[f"{m}_pred"]) for r in per_seed])
            C = confusion(y_true_all, preds)
            ax.imshow(C, cmap="Blues")
            acc = summary["accuracy"][m]["mean"]
            ax.set_title(f"{m}  acc={acc:.3f}")
            ax.set_xlabel("pred"); ax.set_ylabel("true")
        fig.suptitle(f"DVS-Gesture-128 confusion ({data_source}, n_seed={len(per_seed)})")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "confusion_matrices.png", dpi=110)
        log(f"wrote {OUT_DIR/'confusion_matrices.png'}")
    except Exception as e:
        log(f"matplotlib plot failed: {e}")

    log("==== GATE OUTCOMES ====")
    for k, v in summary["gates"].items():
        log(f"  {k}: {v}")
    log("==== DONE ====")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"FATAL: {e}\n{traceback.format_exc()}")
        raise
