"""DS-N2: DVS128 Gesture event-camera classification.

Native sparse-event input → NS-RAM N=128 reservoir, ridge readout.
Compare against a simple LIF SNN baseline at matched scale.

Hypothesis: sparse, temporally-structured spike events drive NS-RAM body-
state V_b through real excursions (NOT collapsed to a fixed point as in
DS-N1 KWS with dense MFCC).

Dataset:
  - Preferred: tonic.datasets.DVSGesture (real IBM DVS128, 11 classes).
  - Fallback: synthetic "DVS-proxy" sparse-event dataset — 11 classes,
    each generated from a class-specific (x,y,t,polarity) Poisson
    distribution. Tests the same hypothesis (sparse events vs dense)
    even if IBM data unavailable.

Binning:
  - T_BINS = 32 time windows over the recording (~50 ms each if 1.5 s).
  - Spatial grid: 16×16 (downsampled from 128×128).
  - Polarity collapsed (sum +/- in same channel) → tensor (T_BINS, 16, 16).

NS-RAM front-end:
  - Flatten spatial → 256-d vector per time bin.
  - Project via W_in (128, 256) → drive VG1 of each NS-RAM neuron.
  - V_b time-stepped at dt = 1e-7 s, sub_steps per bin.
  - Pooled features per chunk + final V_b → ridge classifier.

LIF baseline:
  - Same W_in projection, LIF neurons with τ=20ms, V_thr=1.0, reset=0.
  - Same pooled rate readout + ridge.

Pre-registered gates:
  - SANITY:        LIF       >= 50%   (above chance 9%)
  - CONSERVATIVE:  NS-RAM    >= 60%
  - AMBITIOUS:     NS-RAM    >= LIF - 5pp
  - BREAKTHROUGH:  NS-RAM    >= 80%  AND  energy < 10 µJ/inference
"""
from __future__ import annotations
import os
for _k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[_k] = "1"
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")

import json
import math
import time
import traceback
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "results/z290_ds_n2_dvs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- config (LOCKED) ----------
N_CLASSES         = 11
N_PER_CLASS_TRAIN = 100
N_PER_CLASS_TEST  = 25
T_BINS            = 32                       # time bins per sample
SPATIAL           = 16                       # downsample 128->16
INPUT_DIM         = SPATIAL * SPATIAL        # 256
SAMPLE_DURATION_S = 1.5                      # seconds per sample (synthetic)
BIN_DT_S          = SAMPLE_DURATION_S / T_BINS  # ~46.9 ms
SEEDS             = [0, 1, 2, 3]
NSRAM_N           = 128
NSRAM_CELL = {"C_b_fF": 80.0, "V_G2_bias": 0.35, "dt_s": 1e-7, "g_in": 0.4}
# DVS bins are ~47ms apart but surrogate is calibrated at dt_s=1e-7. We integrate
# per-bin with effective time multiplier so V_b can actually MOVE on event input.
NSRAM_DT_PER_BIN_S = 2e-6    # 20× larger than KWS — V_b moves but doesn't saturate
NSRAM_VG1_LO, NSRAM_VG1_HI = 0.25, 0.55
NSRAM_SUB_STEPS = 2                          # surrogate substeps per bin
N_POOL_CHUNKS  = 4
SURROGATE_PATH = "results/z271_pmp3_dense_surrogate/surrogate_4d_v2.npz"
ENERGY_PER_SPIKE_FJ_NSRAM = 6.4             # consistent with z287
ENERGY_PER_SPIKE_FJ_LIF   = 0.9             # generic LIF synapse-event estimate (Loihi-ish)
RIDGE_ALPHA = 1.0

# LIF baseline params
LIF_TAU_S = 0.020
LIF_V_THR = 1.0
LIF_INPUT_GAIN = 1.0


# ============================================================
# Dataset loading
# ============================================================

def try_load_real_dvs(save_to: Path):
    """Try to load real DVSGesture via tonic. Returns dict or None."""
    try:
        from tonic.datasets import DVSGesture
    except Exception as e:
        print("[dvs] tonic import failed:", e); return None
    try:
        # The tonic download endpoints (figshare) appear blocked by WAF.
        t0 = time.time()
        train = DVSGesture(save_to=str(save_to), train=True)
        test  = DVSGesture(save_to=str(save_to), train=False)
        print(f"[dvs] real DVSGesture loaded in {time.time()-t0:.1f}s (train={len(train)}, test={len(test)})")
        return {"train": train, "test": test}
    except Exception as e:
        print(f"[dvs] real DVSGesture unavailable: {e}")
        return None


def real_dvs_to_tensor(ds, n_per_class: int, rng: np.random.Generator):
    """Bin a tonic DVSGesture dataset into (N_samples, T_BINS, SPATIAL, SPATIAL) tensor."""
    # Collect per-class indices
    by_class = {c: [] for c in range(N_CLASSES)}
    for i in range(len(ds)):
        _, lbl = ds[i]
        if lbl in by_class and len(by_class[lbl]) < n_per_class:
            by_class[lbl].append(i)
    Xs, ys = [], []
    SX, SY = 128, 128
    pix = SX // SPATIAL
    for c, idxs in by_class.items():
        for idx in idxs:
            ev, lbl = ds[idx]
            # ev is structured array with fields: 'x','y','t','p' (microseconds)
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
    """Synthetic DVS-proxy: each class has a center (cx, cy), motion direction
    (vx, vy), temporal profile, and noise. Each sample emits ~5000 sparse events.

    Returns (Xtr, ytr, Xte, yte) — each X as (N, T_BINS, SPATIAL, SPATIAL) sparse counts.
    """
    rng = np.random.default_rng(seed)
    # class signatures — DELIBERATELY MORE OVERLAPPED to make this a real test.
    # All classes share roughly the same center region; differ mainly by
    # motion direction (vx, vy) and temporal pattern. Background noise much higher.
    class_params = []
    center_pool_cx = SPATIAL / 2.0
    center_pool_cy = SPATIAL / 2.0
    for c in range(N_CLASSES):
        cx = center_pool_cx + rng.normal(0, 1.0)
        cy = center_pool_cy + rng.normal(0, 1.0)
        # full circle of motion directions — moderate separability
        angle = 2 * math.pi * c / N_CLASSES + rng.uniform(-0.15, 0.15)
        vx = 3.5 * math.cos(angle)
        vy = 3.5 * math.sin(angle)
        sigma = rng.uniform(1.4, 2.2)            # moderate spatial spread
        t_peak = rng.uniform(0.35, 0.65)
        t_width = rng.uniform(0.20, 0.40)        # WIDER temporal → more overlap
        rate_peak = rng.uniform(4500, 5500)
        class_params.append(dict(cx=cx, cy=cy, vx=vx, vy=vy, sigma=sigma,
                                 t_peak=t_peak, t_width=t_width, rate=rate_peak))

    def make_split(n_per_class, split_seed):
        rs = np.random.default_rng(seed + split_seed)
        Xs, ys = [], []
        for c, p in enumerate(class_params):
            for _ in range(n_per_class):
                # jitter — much larger per-sample variance
                cx = p["cx"] + rs.normal(0, 1.5)
                cy = p["cy"] + rs.normal(0, 1.5)
                vx = p["vx"] * rs.uniform(0.6, 1.4) + rs.normal(0, 0.3)
                vy = p["vy"] * rs.uniform(0.6, 1.4) + rs.normal(0, 0.3)
                sigma = p["sigma"] * rs.uniform(0.85, 1.2)
                t_peak = p["t_peak"] + rs.normal(0, 0.10)
                t_width = p["t_width"] * rs.uniform(0.7, 1.4)
                rate = p["rate"] * rs.uniform(0.7, 1.3)

                n_events = int(rate)
                # sample times in [0,1]
                ts = rs.normal(t_peak, t_width, size=n_events)
                ts = np.clip(ts, 0, 0.999)
                # spatial position follows center + velocity*t + gaussian noise
                xs = cx + vx * (ts - 0.5) + rs.normal(0, sigma, size=n_events)
                ysp = cy + vy * (ts - 0.5) + rs.normal(0, sigma, size=n_events)
                # discretize
                t_idx = (ts * T_BINS).astype(np.int64).clip(0, T_BINS - 1)
                x_idx = xs.astype(np.int64).clip(0, SPATIAL - 1)
                y_idx = ysp.astype(np.int64).clip(0, SPATIAL - 1)
                arr = np.zeros((T_BINS, SPATIAL, SPATIAL), dtype=np.float32)
                np.add.at(arr, (t_idx, x_idx, y_idx), 1.0)
                # background noise: ~20% of class rate (realistic DVS noise level)
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
# NS-RAM surrogate
# ============================================================

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


def _frac_index(values, axis):
    n = axis.shape[0]
    i = torch.bucketize(values, axis) - 1
    i = i.clamp(0, n - 2)
    lo = axis[i]; hi = axis[i + 1]
    t = (values - lo) / (hi - lo)
    return i, t.clamp(0.0, 1.0)


def query_surrogate(surr, VG1, VG2, Vd, Vb):
    i0, t0 = _frac_index(VG1, surr["ax_VG1"])
    i1, t1 = _frac_index(VG2, surr["ax_VG2"])
    i2, t2 = _frac_index(Vd,  surr["ax_Vd"])
    i3, t3 = _frac_index(Vb,  surr["ax_Vb"])
    Id_tbl, Iii_tbl, Ilk_tbl = surr["I_d"], surr["I_ii"], surr["I_leak"]
    Id_out = torch.zeros_like(VG1); Iii_out = torch.zeros_like(VG1); Ilk_out = torch.zeros_like(VG1)
    for a0 in (0,1):
        w0 = t0 if a0 else (1-t0); j0 = i0+a0
        for a1 in (0,1):
            w1 = t1 if a1 else (1-t1); j1 = i1+a1
            for a2 in (0,1):
                w2 = t2 if a2 else (1-t2); j2 = i2+a2
                for a3 in (0,1):
                    w3 = t3 if a3 else (1-t3); j3 = i3+a3
                    w = w0*w1*w2*w3
                    Id_out  = Id_out  + w * Id_tbl[j0,j1,j2,j3]
                    Iii_out = Iii_out + w * Iii_tbl[j0,j1,j2,j3]
                    Ilk_out = Ilk_out + w * Ilk_tbl[j0,j1,j2,j3]
    return Id_out, Iii_out, Ilk_out


# ============================================================
# NS-RAM reservoir over event tensor
# ============================================================

def nsram_features_events(X, N, W_in, V_G1_bias, V_G2_bias, surr,
                          g_in, C_b_F, dt_s, vd=1.0, sub_steps=NSRAM_SUB_STEPS,
                          n_pool_chunks=N_POOL_CHUNKS, return_diag=False):
    """X : (B, T_BINS, INPUT_DIM) event counts per bin (sparse, integer).

    Drive each NS-RAM neuron by g_in * (W_in @ event_vec). This is a
    *signed* drive — strong positive events push VG1 high, low events push
    it low. V_b persists across bins. Substeps per bin smooth integration.
    """
    device = X.device
    B, T_in, D = X.shape
    Vb_min = surr["ax_Vb"][0]; Vb_max = surr["ax_Vb"][-1]
    VG1_min = surr["ax_VG1"][0]; VG1_max = surr["ax_VG1"][-1]
    Vb = torch.zeros(B, N, device=device)
    VG2_2d = V_G2_bias.expand(B, N)
    Vd_2d  = torch.full((B, N), float(vd), device=device)

    chunk_len = max(T_in // n_pool_chunks, 1)
    vb_chunks = [torch.zeros(B, N, device=device) for _ in range(n_pool_chunks)]
    id_chunks = [torch.zeros(B, N, device=device) for _ in range(n_pool_chunks)]
    chunk_counts = [0] * n_pool_chunks

    # Normalize per-batch event magnitude so VG1 drive stays in surrogate range.
    # max events per bin can vary a lot; scale by global per-sample mean+std.
    Xn = X / (X.mean(dim=(1,2), keepdim=True) + 1e-3)  # rate units

    vb_excursion = torch.zeros(B, N, device=device)  # max |Vb-init| across time
    vb_init = torch.zeros(B, N, device=device)

    spike_count_total = torch.zeros(B, device=device)
    for fi in range(T_in):
        ev = Xn[:, fi, :]                                 # (B, D)
        spike_count_total = spike_count_total + X[:, fi, :].sum(dim=1)
        ck = min(fi // chunk_len, n_pool_chunks - 1)
        # zero-mean per-feature drive
        drive_frame = (ev - ev.mean(dim=1, keepdim=True)) @ W_in.T   # (B, N)
        for s in range(sub_steps):
            VG1 = V_G1_bias.unsqueeze(0) + g_in * drive_frame
            VG1 = VG1.clamp(VG1_min, VG1_max)
            Vb_c = Vb.clamp(Vb_min, Vb_max)
            I_d, I_ii, I_leak = query_surrogate(surr, VG1, VG2_2d, Vd_2d, Vb_c)
            Vb = (Vb + dt_s * (I_ii - I_leak) / C_b_F).clamp(Vb_min, Vb_max)
            vb_excursion = torch.maximum(vb_excursion, (Vb - vb_init).abs())
            vb_chunks[ck] = vb_chunks[ck] + Vb
            id_chunks[ck] = id_chunks[ck] + I_d.abs()
            chunk_counts[ck] += 1

    parts = []
    for k in range(n_pool_chunks):
        c = max(chunk_counts[k], 1)
        parts.append(vb_chunks[k] / c)
        parts.append((id_chunks[k] / c + 1e-18).log10())
    parts.append(Vb)
    feats = torch.cat(parts, dim=1)
    mu = feats.mean(dim=0, keepdim=True); sd = feats.std(dim=0, keepdim=True) + 1e-9
    feats = (feats - mu) / sd

    if return_diag:
        diag = {
            "vb_excursion_mean": vb_excursion.mean().item(),
            "vb_excursion_max":  vb_excursion.max().item(),
            "vb_final_mean":     Vb.mean().item(),
            "vb_final_std":      Vb.std().item(),
            "spikes_per_inf":    spike_count_total.mean().item(),
        }
        return feats, diag
    return feats, None


# ============================================================
# LIF baseline
# ============================================================

def lif_features(X, N, W_in, dt_s, tau_s, V_thr, n_pool_chunks=N_POOL_CHUNKS):
    """LIF reservoir: dV/dt = -V/tau + W_in*input ; spike when V>=V_thr, reset to 0.

    Features: per-chunk mean spike rate per neuron + final V → (B, (n_pool_chunks+1)*N)
    """
    device = X.device
    B, T_in, D = X.shape
    alpha = math.exp(-dt_s / tau_s)
    V = torch.zeros(B, N, device=device)
    chunk_len = max(T_in // n_pool_chunks, 1)
    rate_chunks = [torch.zeros(B, N, device=device) for _ in range(n_pool_chunks)]
    chunk_counts = [0] * n_pool_chunks

    Xn = X / (X.mean(dim=(1,2), keepdim=True) + 1e-3)

    for fi in range(T_in):
        ev = Xn[:, fi, :]
        ck = min(fi // chunk_len, n_pool_chunks - 1)
        drive = (ev - ev.mean(dim=1, keepdim=True)) @ W_in.T   # (B, N)
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
    feats = torch.cat(parts, dim=1)
    mu = feats.mean(dim=0, keepdim=True); sd = feats.std(dim=0, keepdim=True) + 1e-9
    feats = (feats - mu) / sd
    return feats


# ============================================================
# Ridge readout
# ============================================================

def ridge_lstsq(X, y, alpha=RIDGE_ALPHA, n_classes=N_CLASSES):
    Y = torch.nn.functional.one_hot(y, n_classes).float()
    A = X.T @ X + alpha * torch.eye(X.shape[1], device=X.device)
    B = X.T @ Y
    try:
        return torch.linalg.solve(A, B)
    except Exception:
        return torch.linalg.lstsq(A, B).solution


# ============================================================
# Per-seed run
# ============================================================

def run_seed(seed, Xtr_np, ytr_np, Xte_np, yte_np, device, surr):
    torch.manual_seed(seed)
    g = torch.Generator(device=device).manual_seed(seed + 31337)

    # Flatten spatial dim → (B, T_BINS, INPUT_DIM)
    Xtr = torch.tensor(Xtr_np, device=device).view(-1, T_BINS, INPUT_DIM)
    Xte = torch.tensor(Xte_np, device=device).view(-1, T_BINS, INPUT_DIM)
    ytr = torch.tensor(ytr_np, device=device)
    yte = torch.tensor(yte_np, device=device)

    # Shared W_in (NS-RAM and LIF use same projection for fair comparison)
    W_in = torch.randn(NSRAM_N, INPUT_DIM, generator=g, device=device)
    W_in = W_in / (W_in.norm(dim=1, keepdim=True) + 1e-9)

    # ---- LIF baseline ----
    t0 = time.time()
    f_tr = lif_features(Xtr, NSRAM_N, W_in, BIN_DT_S / NSRAM_SUB_STEPS, LIF_TAU_S, LIF_V_THR)
    f_te = lif_features(Xte, NSRAM_N, W_in, BIN_DT_S / NSRAM_SUB_STEPS, LIF_TAU_S, LIF_V_THR)
    W = ridge_lstsq(f_tr, ytr)
    lif_pred = (f_te @ W).argmax(dim=1)
    lif_acc = (lif_pred == yte).float().mean().item()
    lif_wall = time.time() - t0

    # ---- NS-RAM ----
    V_G1_bias = torch.empty(NSRAM_N, device=device).uniform_(NSRAM_VG1_LO, NSRAM_VG1_HI, generator=g)
    V_G2_bias = torch.full((NSRAM_N,), NSRAM_CELL["V_G2_bias"], device=device)
    C_b_F = NSRAM_CELL["C_b_fF"] * 1e-15
    dt_s  = NSRAM_DT_PER_BIN_S          # larger dt to let V_b actually move
    g_in  = NSRAM_CELL["g_in"]

    def feat_nsram(X_t, return_diag=False, batch=32):
        outs = []; diags = []
        for i in range(0, X_t.shape[0], batch):
            f, d = nsram_features_events(X_t[i:i+batch], NSRAM_N, W_in,
                                         V_G1_bias, V_G2_bias, surr,
                                         g_in, C_b_F, dt_s, sub_steps=NSRAM_SUB_STEPS,
                                         return_diag=return_diag)
            outs.append(f)
            if return_diag: diags.append(d)
        feats = torch.cat(outs, 0)
        if return_diag:
            # aggregate
            keys = diags[0].keys()
            agg = {k: float(np.mean([d[k] for d in diags])) for k in keys}
            return feats, agg
        return feats, None

    t0 = time.time()
    f_tr_n, _    = feat_nsram(Xtr, return_diag=False)
    f_te_n, diag = feat_nsram(Xte, return_diag=True)
    W = ridge_lstsq(f_tr_n, ytr)
    nsram_pred = (f_te_n @ W).argmax(dim=1)
    nsram_acc = (nsram_pred == yte).float().mean().item()
    nsram_wall = time.time() - t0

    # Energy estimates per inference (test set average)
    # NS-RAM: total events ≈ input events + body events per inference
    body_events = T_BINS * NSRAM_SUB_STEPS * NSRAM_N
    spikes_per_inf = diag["spikes_per_inf"]
    nsram_energy_J = (spikes_per_inf + body_events) * ENERGY_PER_SPIKE_FJ_NSRAM * 1e-15
    lif_energy_J   = (spikes_per_inf + body_events) * ENERGY_PER_SPIKE_FJ_LIF * 1e-15

    return {
        "seed": seed,
        "n_train": int(Xtr.shape[0]),
        "n_test":  int(Xte.shape[0]),
        "lif_acc": float(lif_acc),
        "lif_wall_s": float(lif_wall),
        "lif_energy_per_inf_uJ": float(lif_energy_J * 1e6),
        "nsram_acc": float(nsram_acc),
        "nsram_wall_s": float(nsram_wall),
        "nsram_energy_per_inf_uJ": float(nsram_energy_J * 1e6),
        "nsram_diag": diag,
    }


# ============================================================
# Main
# ============================================================

def main():
    print(f"[ds-n2] starting at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[ds-n2] device={device}")

    # Try real DVS first
    real = try_load_real_dvs(ROOT / "data/dvs_gesture")
    if real is not None:
        rng = np.random.default_rng(0)
        Xtr_np, ytr_np = real_dvs_to_tensor(real["train"], N_PER_CLASS_TRAIN, rng)
        Xte_np, yte_np = real_dvs_to_tensor(real["test"],  N_PER_CLASS_TEST,  rng)
        dataset_used = "real_dvs_gesture"
    else:
        print("[ds-n2] falling back to synthetic DVS-proxy")
        Xtr_np, ytr_np, Xte_np, yte_np = synth_dvs(N_PER_CLASS_TRAIN, N_PER_CLASS_TEST, seed=0)
        dataset_used = "synthetic_dvs_proxy"

    print(f"[ds-n2] dataset={dataset_used}  train={Xtr_np.shape}  test={Xte_np.shape}")
    print(f"[ds-n2] mean events/sample (train) = {Xtr_np.sum(axis=(1,2,3)).mean():.0f}")
    print(f"[ds-n2] sparsity (zero-bins fraction) = {(Xtr_np == 0).mean():.4f}")

    surr = load_surrogate(SURROGATE_PATH, device)
    print(f"[ds-n2] surrogate loaded from {SURROGATE_PATH}")

    per_seed = []
    for seed in SEEDS:
        print(f"\n[ds-n2] === seed {seed} ===")
        try:
            res = run_seed(seed, Xtr_np, ytr_np, Xte_np, yte_np, device, surr)
        except Exception as e:
            traceback.print_exc()
            res = {"seed": seed, "error": str(e)}
        print(f"[ds-n2] seed {seed}: LIF={res.get('lif_acc',float('nan')):.3f}  "
              f"NSRAM={res.get('nsram_acc',float('nan')):.3f}  "
              f"V_b_exc={res.get('nsram_diag',{}).get('vb_excursion_mean',float('nan')):.4f}")
        per_seed.append(res)

    lif_accs   = np.array([r["lif_acc"]   for r in per_seed if "lif_acc" in r])
    nsram_accs = np.array([r["nsram_acc"] for r in per_seed if "nsram_acc" in r])
    def ci95(arr):
        if len(arr) < 2: return [float("nan"), float("nan")]
        m = arr.mean(); s = arr.std(ddof=1); h = 1.96 * s / math.sqrt(len(arr))
        return [float(m - h), float(m + h)]

    lif_mean = float(lif_accs.mean()) if len(lif_accs) else float("nan")
    nsram_mean = float(nsram_accs.mean()) if len(nsram_accs) else float("nan")
    lif_std = float(lif_accs.std(ddof=1)) if len(lif_accs)>=2 else 0.0
    nsram_std = float(nsram_accs.std(ddof=1)) if len(nsram_accs)>=2 else 0.0

    vb_exc_mean = float(np.mean([r["nsram_diag"]["vb_excursion_mean"]
                                 for r in per_seed if "nsram_diag" in r]))
    nsram_energy_uJ = float(np.mean([r["nsram_energy_per_inf_uJ"]
                                     for r in per_seed if "nsram_energy_per_inf_uJ" in r]))

    gates = {
        "sanity_lif_>=50":          bool(lif_mean >= 0.50),
        "conservative_nsram_>=60":  bool(nsram_mean >= 0.60),
        "ambitious_nsram_>=lif-5pp":bool(nsram_mean >= lif_mean - 0.05),
        "breakthrough_nsram_>=80_and_E<10uJ":
            bool(nsram_mean >= 0.80 and nsram_energy_uJ < 10.0),
    }
    if gates["breakthrough_nsram_>=80_and_E<10uJ"]:
        verdict = "BREAKTHROUGH"
    elif gates["ambitious_nsram_>=lif-5pp"] and gates["conservative_nsram_>=60"]:
        verdict = "AMBITIOUS"
    elif gates["conservative_nsram_>=60"]:
        verdict = "CONSERVATIVE"
    elif gates["sanity_lif_>=50"]:
        verdict = "SANITY_ONLY (NSRAM_FAIL)"
    else:
        verdict = "FAIL"

    summary = {
        "task": "z290 DS-N2 DVS128 Gesture (sparse-event classification)",
        "dataset_used": dataset_used,
        "config": {
            "n_classes": N_CLASSES, "n_per_class_train": N_PER_CLASS_TRAIN,
            "n_per_class_test": N_PER_CLASS_TEST,
            "t_bins": T_BINS, "spatial": SPATIAL, "input_dim": INPUT_DIM,
            "sample_duration_s": SAMPLE_DURATION_S,
            "seeds": SEEDS, "nsram_N": NSRAM_N, "nsram_cell": NSRAM_CELL,
            "nsram_sub_steps": NSRAM_SUB_STEPS, "n_pool_chunks": N_POOL_CHUNKS,
            "surrogate_path": SURROGATE_PATH,
            "lif_tau_s": LIF_TAU_S, "lif_v_thr": LIF_V_THR,
            "ridge_alpha": RIDGE_ALPHA,
            "energy_per_spike_fJ_nsram": ENERGY_PER_SPIKE_FJ_NSRAM,
            "energy_per_spike_fJ_lif":   ENERGY_PER_SPIKE_FJ_LIF,
            "device": device,
        },
        "per_seed": per_seed,
        "lif_mean": lif_mean, "lif_std": lif_std, "lif_ci95": ci95(lif_accs),
        "nsram_mean": nsram_mean, "nsram_std": nsram_std, "nsram_ci95": ci95(nsram_accs),
        "delta_mean_pp": float((nsram_mean - lif_mean) * 100),
        "vb_excursion_mean_V": vb_exc_mean,
        "nsram_energy_per_inf_uJ_mean": nsram_energy_uJ,
        "gates": gates,
        "verdict": verdict,
    }

    out_path = OUT_DIR / "summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[ds-n2] === SUMMARY ===")
    print(f"  dataset:   {dataset_used}")
    print(f"  LIF:       {lif_mean:.3f} ± {lif_std:.3f}   CI95={summary['lif_ci95']}")
    print(f"  NSRAM:     {nsram_mean:.3f} ± {nsram_std:.3f}   CI95={summary['nsram_ci95']}")
    print(f"  Δ:         {summary['delta_mean_pp']:+.1f} pp")
    print(f"  V_b exc:   {vb_exc_mean:.4f} V   (KWS was ~0 — collapsed)")
    print(f"  Energy:    {nsram_energy_uJ*1000:.4f} nJ / inference  ({nsram_energy_uJ:.6f} µJ)")
    print(f"  Gates:     {gates}")
    print(f"  Verdict:   {verdict}")
    print(f"  Saved:     {out_path}")


if __name__ == "__main__":
    main()
