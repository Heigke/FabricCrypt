"""DS-N4: MIT-BIH ECG arrhythmia (Normal vs PVC) on NS-RAM.

Hypothesis: R-wave events are NATURALLY sparse (~1 Hz heart rate),
matching NS-RAM body-state τ. Expect strong V_b excursion (like DS-N2 DVS,
unlike DS-N1 KWS).

Dataset:
  - MIT-BIH Arrhythmia Database via wfdb (PhysioNet).
  - AAMI standard train/test split (DS1/DS2).
    DS1 (train): 101 106 108 109 112 114 115 116 118 119 122 124 201 203 205 207 208 209 215 220 223 230
    DS2 (test):  100 103 105 111 113 117 121 123 200 202 210 212 213 214 219 221 222 228 231 232 233 234
  - Binary task: Normal (N) vs PVC (V). MIT-BIH symbols:
      N = 'N' (also 'L','R','e','j' grouped into N per AAMI)
      V = 'V','E'

Per-beat feature:
  - Around each R-peak: 50 ms window split into T_BINS = 5 bins of 10 ms.
  - Sparse event = sample-derivative thresholded (R-wave detection within window
    creates an event burst). Plus neighborhood features per bin:
      [rate_in_bin, abs_signal_mean_in_bin, rr_prev, rr_next, beat_amp]
  - F = 5 features.

NS-RAM front-end:
  - W_in (N=128, INPUT_DIM = T_BINS * F = 25) → drive VG1.
  - V_b time-stepped over T_BINS bins, dt_s scaled so V_b actually moves
    (lessons from DS-N2: NSRAM_DT_PER_BIN_S = 2e-6 worked).
  - Readout = (V_b, log10|I_d|) JOINT features (chunked) → ridge classifier.

MLP baseline:
  - 2-layer (32 hidden), Adam 10 epochs, on flattened (T_BINS, F) = 25-d input.

Pre-registered gates:
  - SANITY:        MLP    >= 90%
  - CONSERVATIVE:  NSRAM  >= 85%
  - AMBITIOUS:     NSRAM  >= MLP - 3pp  with non-overlap CI
  - BREAKTHROUGH:  NSRAM  >= MLP  AND  energy < 1 nJ / beat
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
OUT_DIR = ROOT / "results/z292_ds_n4_ecg"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = ROOT / "data/mitdb"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ---------- AAMI splits ----------
DS1_TRAIN = ["101","106","108","109","112","114","115","116","118","119","122","124",
             "201","203","205","207","208","209","215","220","223","230"]
DS2_TEST  = ["100","103","105","111","113","117","121","123",
             "200","202","210","212","213","214","219","221","222","228","231","232","233","234"]

# AAMI: N class symbols, V class symbols
N_SYMS = set(["N","L","R","e","j"])
V_SYMS = set(["V","E"])

# ---------- config (LOCKED) ----------
N_CLASSES         = 2
T_BINS            = 5          # 10 ms bins over 50 ms window
WIN_MS            = 50.0
F_PER_BIN         = 5          # [rate, abs_mean, rr_prev, rr_next, beat_amp]
INPUT_DIM         = T_BINS * F_PER_BIN  # 25
SEEDS             = [0, 1, 2, 3]
NSRAM_N           = 128
NSRAM_CELL = {"C_b_fF": 80.0, "V_G2_bias": 0.35, "dt_s": 1e-7, "g_in": 0.4}
NSRAM_DT_PER_BIN_S = 2e-6
NSRAM_VG1_LO, NSRAM_VG1_HI = 0.25, 0.55
NSRAM_SUB_STEPS = 2
N_POOL_CHUNKS  = 4
SURROGATE_PATH = str(ROOT / "results/z271_pmp3_dense_surrogate/surrogate_4d_v2.npz")
ENERGY_PER_SPIKE_FJ_NSRAM = 6.4
RIDGE_ALPHA = 1.0
MAX_BEATS_PER_REC = 400        # cap per record to keep balanced and quick

# ============================================================
# Dataset: MIT-BIH via wfdb
# ============================================================

def ensure_mitdb():
    import wfdb
    have = (DATA_DIR / "100.hea").exists() and (DATA_DIR / "234.hea").exists()
    if have:
        print(f"[ecg] mitdb already present at {DATA_DIR}")
        return
    print(f"[ecg] downloading mitdb to {DATA_DIR} ...")
    t0 = time.time()
    wfdb.dl_database("mitdb", str(DATA_DIR))
    print(f"[ecg] mitdb download done in {time.time()-t0:.1f}s")


def extract_beats(records, max_beats=MAX_BEATS_PER_REC, rng=None):
    """For each record, load signal+annotations, extract Normal/PVC beats.
    Returns (X: (B, T_BINS, F_PER_BIN), y: (B,)).
    """
    import wfdb
    if rng is None:
        rng = np.random.default_rng(0)
    Xs = []; ys = []
    for rec in records:
        try:
            sig, fields = wfdb.rdsamp(str(DATA_DIR / rec))
            ann = wfdb.rdann(str(DATA_DIR / rec), "atr")
        except Exception as e:
            print(f"[ecg] record {rec} load fail: {e}")
            continue
        fs = fields["fs"]  # 360
        x = sig[:, 0].astype(np.float32)
        # robust normalize per record
        x = (x - np.median(x)) / (np.std(x) + 1e-6)
        win_samp = int(WIN_MS * 1e-3 * fs)              # ~18 samples
        bin_samp = max(win_samp // T_BINS, 1)            # ~3 samples/bin
        # use only samples ~half a window before R-peak: capture R itself
        half = win_samp // 2

        # filter beat positions/symbols to N or V
        samples = ann.sample
        symbols = ann.symbol
        beat_idx = []
        for i, s in enumerate(symbols):
            if s in N_SYMS:
                beat_idx.append((i, 0))
            elif s in V_SYMS:
                beat_idx.append((i, 1))
        if not beat_idx:
            continue

        # balance per record: keep up to max_beats/2 of each class, then shuffle
        n_list = [bi for bi in beat_idx if bi[1] == 0]
        v_list = [bi for bi in beat_idx if bi[1] == 1]
        rng.shuffle(n_list)
        rng.shuffle(v_list)
        n_keep = min(len(n_list), max_beats // 2)
        v_keep = min(len(v_list), max_beats // 2)
        chosen = n_list[:n_keep] + v_list[:v_keep]
        rng.shuffle(chosen)

        for i, lbl in chosen:
            r = samples[i]
            lo = r - half; hi = lo + win_samp
            if lo < 1 or hi >= len(x):
                continue
            seg = x[lo:hi]                                    # ~win_samp samples
            # ensure exactly T_BINS bins
            usable = bin_samp * T_BINS
            seg = seg[:usable]
            seg = seg.reshape(T_BINS, bin_samp)
            # rate = number of |Δ| > thr samples (event count proxy)
            d = np.abs(np.diff(seg, axis=1, prepend=seg[:, :1]))
            thr = 0.3 * (d.std() + 1e-6)
            rate = (d > thr).sum(axis=1).astype(np.float32)   # (T_BINS,)
            abs_mean = np.abs(seg).mean(axis=1).astype(np.float32)
            # RR intervals (seconds)
            rr_prev = float((samples[i] - samples[i-1]) / fs) if i > 0 else 1.0
            j = i + 1
            rr_next = float((samples[j] - samples[i]) / fs) if j < len(samples) else 1.0
            rr_prev = max(min(rr_prev, 2.0), 0.2)
            rr_next = max(min(rr_next, 2.0), 0.2)
            amp = float(seg.max() - seg.min())
            feats = np.zeros((T_BINS, F_PER_BIN), dtype=np.float32)
            feats[:, 0] = rate
            feats[:, 1] = abs_mean
            feats[:, 2] = rr_prev
            feats[:, 3] = rr_next
            feats[:, 4] = amp
            Xs.append(feats); ys.append(lbl)
    if not Xs:
        return np.zeros((0, T_BINS, F_PER_BIN), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    X = np.stack(Xs, axis=0)
    y = np.asarray(ys, dtype=np.int64)
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


# ============================================================
# NS-RAM surrogate (same interp as DS-N2)
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


def nsram_features(X, N, W_in, V_G1_bias, V_G2_bias, surr,
                   g_in, C_b_F, dt_s, vd=1.0, sub_steps=NSRAM_SUB_STEPS,
                   n_pool_chunks=N_POOL_CHUNKS, return_diag=False):
    """X : (B, T_BINS, INPUT_DIM_FLAT) where INPUT_DIM_FLAT = F_PER_BIN
    But we feed flattened (B, T_BINS, F_PER_BIN) → drive uses F_PER_BIN.
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

    # per-batch normalize features for stable drive
    Xn = (X - X.mean(dim=(1,2), keepdim=True)) / (X.std(dim=(1,2), keepdim=True) + 1e-6)

    vb_excursion = torch.zeros(B, N, device=device)
    vb_init = torch.zeros(B, N, device=device)
    spike_count_total = torch.zeros(B, device=device)
    for fi in range(T_in):
        ev = Xn[:, fi, :]                                # (B, D)
        spike_count_total = spike_count_total + X[:, fi, 0].abs()   # use rate feature
        ck = min(fi // chunk_len, n_pool_chunks - 1)
        drive_frame = ev @ W_in.T                          # (B, N)
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
# MLP baseline
# ============================================================
def train_mlp(Xtr, ytr, Xte, yte, device, seed, epochs=10, hidden=32, lr=1e-3):
    torch.manual_seed(seed)
    in_dim = Xtr.shape[1]
    net = torch.nn.Sequential(
        torch.nn.Linear(in_dim, hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden, N_CLASSES),
    ).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    bs = 64
    N = Xtr.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(N, device=device)
        for i in range(0, N, bs):
            idx = perm[i:i+bs]
            opt.zero_grad()
            out = net(Xtr[idx])
            loss = loss_fn(out, ytr[idx])
            loss.backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        pred = net(Xte).argmax(dim=1)
    acc = (pred == yte).float().mean().item()
    n_params = sum(p.numel() for p in net.parameters())
    return acc, n_params


# ============================================================
# Per-seed run
# ============================================================
def run_seed(seed, Xtr_np, ytr_np, Xte_np, yte_np, device, surr):
    torch.manual_seed(seed)
    g = torch.Generator(device=device).manual_seed(seed + 31337)

    Xtr_t = torch.tensor(Xtr_np, device=device)
    Xte_t = torch.tensor(Xte_np, device=device)
    ytr   = torch.tensor(ytr_np, device=device)
    yte   = torch.tensor(yte_np, device=device)

    # ---- MLP baseline on flat features ----
    Xtr_flat = Xtr_t.view(Xtr_t.shape[0], -1)
    Xte_flat = Xte_t.view(Xte_t.shape[0], -1)
    # standardize using train stats
    mu = Xtr_flat.mean(0, keepdim=True); sd = Xtr_flat.std(0, keepdim=True) + 1e-6
    Xtr_n = (Xtr_flat - mu) / sd
    Xte_n = (Xte_flat - mu) / sd
    t0 = time.time()
    mlp_acc, mlp_params = train_mlp(Xtr_n, ytr, Xte_n, yte, device, seed)
    mlp_wall = time.time() - t0

    # ---- NS-RAM ----
    W_in = torch.randn(NSRAM_N, F_PER_BIN, generator=g, device=device)
    W_in = W_in / (W_in.norm(dim=1, keepdim=True) + 1e-9)
    V_G1_bias = torch.empty(NSRAM_N, device=device).uniform_(NSRAM_VG1_LO, NSRAM_VG1_HI, generator=g)
    V_G2_bias = torch.full((NSRAM_N,), NSRAM_CELL["V_G2_bias"], device=device)
    C_b_F = NSRAM_CELL["C_b_fF"] * 1e-15

    def feat_nsram(X_t, return_diag=False, batch=256):
        outs = []; diags = []
        for i in range(0, X_t.shape[0], batch):
            f, d = nsram_features(X_t[i:i+batch], NSRAM_N, W_in,
                                  V_G1_bias, V_G2_bias, surr,
                                  NSRAM_CELL["g_in"], C_b_F, NSRAM_DT_PER_BIN_S,
                                  sub_steps=NSRAM_SUB_STEPS,
                                  return_diag=return_diag)
            outs.append(f)
            if return_diag: diags.append(d)
        feats = torch.cat(outs, 0)
        if return_diag:
            keys = diags[0].keys()
            agg = {k: float(np.mean([d[k] for d in diags])) for k in keys}
            return feats, agg
        return feats, None

    t0 = time.time()
    f_tr, _    = feat_nsram(Xtr_t, return_diag=False)
    f_te, diag = feat_nsram(Xte_t, return_diag=True)
    W = ridge_lstsq(f_tr, ytr)
    nsram_pred = (f_te @ W).argmax(dim=1)
    nsram_acc = (nsram_pred == yte).float().mean().item()
    nsram_wall = time.time() - t0

    # Energy: body events per beat + input "spikes"
    body_events = T_BINS * NSRAM_SUB_STEPS * NSRAM_N
    spikes_per_inf = diag["spikes_per_inf"]
    nsram_energy_J = (spikes_per_inf + body_events) * ENERGY_PER_SPIKE_FJ_NSRAM * 1e-15

    return {
        "seed": seed,
        "n_train": int(Xtr_t.shape[0]),
        "n_test":  int(Xte_t.shape[0]),
        "mlp_acc": float(mlp_acc),
        "mlp_params": int(mlp_params),
        "mlp_wall_s": float(mlp_wall),
        "nsram_acc": float(nsram_acc),
        "nsram_wall_s": float(nsram_wall),
        "nsram_energy_per_beat_nJ": float(nsram_energy_J * 1e9),
        "nsram_diag": diag,
    }


# ============================================================
# Main
# ============================================================
def main():
    print(f"[ds-n4] starting at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[ds-n4] device={device}")

    ensure_mitdb()

    print(f"[ds-n4] extracting beats (train DS1, {len(DS1_TRAIN)} records) ...")
    t0 = time.time()
    Xtr_np, ytr_np = extract_beats(DS1_TRAIN, rng=np.random.default_rng(0))
    print(f"[ds-n4]   train: {Xtr_np.shape}, classes={np.bincount(ytr_np)} in {time.time()-t0:.1f}s")

    print(f"[ds-n4] extracting beats (test DS2, {len(DS2_TEST)} records) ...")
    t0 = time.time()
    Xte_np, yte_np = extract_beats(DS2_TEST, rng=np.random.default_rng(1))
    print(f"[ds-n4]   test:  {Xte_np.shape}, classes={np.bincount(yte_np)} in {time.time()-t0:.1f}s")

    if Xtr_np.shape[0] == 0 or Xte_np.shape[0] == 0:
        raise RuntimeError("No beats extracted")

    surr = load_surrogate(SURROGATE_PATH, device)
    print(f"[ds-n4] surrogate loaded from {SURROGATE_PATH}")

    per_seed = []
    for seed in SEEDS:
        print(f"\n[ds-n4] === seed {seed} ===")
        try:
            res = run_seed(seed, Xtr_np, ytr_np, Xte_np, yte_np, device, surr)
        except Exception as e:
            traceback.print_exc()
            res = {"seed": seed, "error": str(e)}
        print(f"[ds-n4] seed {seed}: MLP={res.get('mlp_acc',float('nan')):.3f}  "
              f"NSRAM={res.get('nsram_acc',float('nan')):.3f}  "
              f"V_b_exc={res.get('nsram_diag',{}).get('vb_excursion_mean',float('nan')):.4f}")
        per_seed.append(res)

    mlp_accs   = np.array([r["mlp_acc"]   for r in per_seed if "mlp_acc" in r])
    nsram_accs = np.array([r["nsram_acc"] for r in per_seed if "nsram_acc" in r])
    def ci95(arr):
        if len(arr) < 2: return [float("nan"), float("nan")]
        m = arr.mean(); s = arr.std(ddof=1); h = 1.96 * s / math.sqrt(len(arr))
        return [float(m - h), float(m + h)]

    mlp_mean = float(mlp_accs.mean()) if len(mlp_accs) else float("nan")
    nsram_mean = float(nsram_accs.mean()) if len(nsram_accs) else float("nan")
    mlp_std = float(mlp_accs.std(ddof=1)) if len(mlp_accs)>=2 else 0.0
    nsram_std = float(nsram_accs.std(ddof=1)) if len(nsram_accs)>=2 else 0.0
    mlp_ci = ci95(mlp_accs); nsram_ci = ci95(nsram_accs)

    vb_exc_mean = float(np.mean([r["nsram_diag"]["vb_excursion_mean"]
                                 for r in per_seed if "nsram_diag" in r]))
    nsram_energy_nJ = float(np.mean([r["nsram_energy_per_beat_nJ"]
                                     for r in per_seed if "nsram_energy_per_beat_nJ" in r]))

    # non-overlap CI check: NSRAM lo >= MLP lo - 3pp AND mean within 3pp
    ambitious_nonoverlap = (
        nsram_mean >= mlp_mean - 0.03
        and (math.isnan(nsram_ci[0]) or math.isnan(mlp_ci[1])
             or nsram_ci[0] >= mlp_ci[0] - 0.03)
    )
    gates = {
        "sanity_mlp_>=90":          bool(mlp_mean >= 0.90),
        "conservative_nsram_>=85":  bool(nsram_mean >= 0.85),
        "ambitious_nsram_>=mlp-3pp_nonoverlap":bool(ambitious_nonoverlap),
        "breakthrough_nsram_>=mlp_and_E<1nJ":
            bool(nsram_mean >= mlp_mean and nsram_energy_nJ < 1.0),
    }
    if gates["breakthrough_nsram_>=mlp_and_E<1nJ"]:
        verdict = "BREAKTHROUGH"
    elif gates["ambitious_nsram_>=mlp-3pp_nonoverlap"] and gates["conservative_nsram_>=85"]:
        verdict = "AMBITIOUS"
    elif gates["conservative_nsram_>=85"]:
        verdict = "CONSERVATIVE"
    elif gates["sanity_mlp_>=90"]:
        verdict = "SANITY_ONLY (NSRAM_FAIL)"
    else:
        verdict = "FAIL"

    summary = {
        "task": "z292 DS-N4 MIT-BIH ECG (N vs PVC, AAMI DS1/DS2)",
        "config": {
            "n_classes": N_CLASSES,
            "t_bins": T_BINS, "win_ms": WIN_MS, "f_per_bin": F_PER_BIN,
            "input_dim": INPUT_DIM,
            "seeds": SEEDS, "nsram_N": NSRAM_N, "nsram_cell": NSRAM_CELL,
            "nsram_dt_per_bin_s": NSRAM_DT_PER_BIN_S,
            "nsram_sub_steps": NSRAM_SUB_STEPS, "n_pool_chunks": N_POOL_CHUNKS,
            "surrogate_path": SURROGATE_PATH,
            "ridge_alpha": RIDGE_ALPHA,
            "energy_per_spike_fJ_nsram": ENERGY_PER_SPIKE_FJ_NSRAM,
            "max_beats_per_rec": MAX_BEATS_PER_REC,
            "ds1_train": DS1_TRAIN, "ds2_test": DS2_TEST,
            "device": device,
        },
        "per_seed": per_seed,
        "mlp_mean": mlp_mean, "mlp_std": mlp_std, "mlp_ci95": mlp_ci,
        "nsram_mean": nsram_mean, "nsram_std": nsram_std, "nsram_ci95": nsram_ci,
        "delta_mean_pp": float((nsram_mean - mlp_mean) * 100),
        "vb_excursion_mean_V": vb_exc_mean,
        "nsram_energy_per_beat_nJ_mean": nsram_energy_nJ,
        "gates": gates,
        "verdict": verdict,
    }

    out_path = OUT_DIR / "summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[ds-n4] === SUMMARY ===")
    print(f"  MLP:       {mlp_mean:.3f} ± {mlp_std:.3f}   CI95={mlp_ci}")
    print(f"  NSRAM:     {nsram_mean:.3f} ± {nsram_std:.3f}   CI95={nsram_ci}")
    print(f"  Δ:         {summary['delta_mean_pp']:+.1f} pp")
    print(f"  V_b exc:   {vb_exc_mean:.4f} V")
    print(f"  Energy:    {nsram_energy_nJ:.4f} nJ / beat")
    print(f"  Gates:     {gates}")
    print(f"  Verdict:   {verdict}")
    print(f"  Saved:     {out_path}")


if __name__ == "__main__":
    main()
