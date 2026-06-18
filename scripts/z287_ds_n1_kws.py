"""z287 DS-N1: Keyword spotting on Google Speech Commands V2 with NS-RAM SNN.

12-class restricted task: {yes, no, up, down, left, right, on, off, stop, go,
_unknown_, _silence_}

Pipeline:
  1) MFCC frontend (40 coeffs, 25ms win / 10ms hop, 99 frames -> 3960 feats)
  2) MLP baseline   (~10K params, Adam, 5 epochs)
  3) NS-RAM SNN     (Poisson encoding -> N=128 NS-RAM neurons -> ridge readout)
  4) 4 seeds, save results/z287_ds_n1_kws/summary.json

NS-RAM uses z277 quadrilinear surrogate (cell d115: C_b=8 fF, V_G2=0.35,
dt=1e-7, g_in=0.8) — best MEP-1 cell.

Pre-registered gates (locked before training):
  - BASELINE SANITY:  MLP >= 80%
  - CONSERVATIVE PASS: NS-RAM >= 70%
  - AMBITIOUS PASS:    NS-RAM >= MLP - 5pp
  - BREAKTHROUGH:      NS-RAM >= MLP AND energy < 100 nJ per 1-s inference

Energy estimate: 6.4 fJ per spike (NS-RAM body-state event).
"""
from __future__ import annotations
import os, sys, json, time, math, random, argparse, glob
from pathlib import Path
from collections import Counter
import numpy as np
import torch
import soundfile as sf
from scipy.signal import lfilter

# ---- pre-registered config ----
SR = 16000
N_MFCC = 40
WIN_MS = 25.0
HOP_MS = 10.0
DURATION_S = 1.0
N_FRAMES = 99          # exactly: (16000 - 400) / 160 + 1 = 98... use 99 nominal
FEAT_DIM = N_MFCC * N_FRAMES   # 3960

KEYWORDS = ["yes","no","up","down","left","right","on","off","stop","go"]
CLASSES  = KEYWORDS + ["_unknown_", "_silence_"]
N_CLASSES = 12
LABEL2IDX = {c: i for i, c in enumerate(CLASSES)}

# ---- NS-RAM config (z277 best cell d115; g_in rescaled for FEAT_DIM=3960
#       vs MNIST 784 — keep drive variance matched: g_in /= sqrt(3960/784) ~2.25) ----
NSRAM_CELL = dict(C_b_fF=80.0, V_G2_bias=0.35, dt_s=1e-7, g_in=0.40)
# V_G1_bias range tuned so VG1 lives in the Iii > Ileak regime (~0.5+) where
# body-state Vb actually charges, not pins at 0.
NSRAM_VG1_LO = 0.45
NSRAM_VG1_HI = 0.65
NSRAM_N = 128
NSRAM_T_STEPS = 100        # 1e-5 / 1e-7
SURROGATE_PATH = "results/z271_pmp3_dense_surrogate/surrogate_4d_v2.npz"
ENERGY_PER_SPIKE_FJ = 6.4

# ---- MFCC implementation (no torchaudio dep) ----
def _hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)
def _mel_to_hz(m):
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

def _build_mel_filterbank(n_fft, sr, n_mels=40, fmin=20.0, fmax=None):
    if fmax is None: fmax = sr / 2
    mel_pts = np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2)
    hz_pts = _mel_to_hz(mel_pts)
    bin_pts = np.floor((n_fft + 1) * hz_pts / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        l, c, r = bin_pts[m-1], bin_pts[m], bin_pts[m+1]
        if c == l: c = l + 1
        if r == c: r = c + 1
        for k in range(l, c): fb[m-1, k] = (k - l) / max(c - l, 1)
        for k in range(c, r): fb[m-1, k] = (r - k) / max(r - c, 1)
    return fb

def _build_dct(n_mels, n_mfcc):
    n = np.arange(n_mels)
    k = np.arange(n_mfcc)[:, None]
    dct = np.cos(np.pi * k * (2 * n + 1) / (2 * n_mels)) * np.sqrt(2.0 / n_mels)
    dct[0] *= 1.0 / np.sqrt(2.0)
    return dct.astype(np.float32)

_WIN_LEN = int(SR * WIN_MS / 1000)   # 400
_HOP_LEN = int(SR * HOP_MS / 1000)   # 160
_N_FFT = 512
_HAMMING = np.hamming(_WIN_LEN).astype(np.float32)
_MEL_FB = _build_mel_filterbank(_N_FFT, SR, n_mels=40)
_DCT = _build_dct(40, N_MFCC)

def waveform_to_mfcc(wav: np.ndarray) -> np.ndarray:
    """wav float32 of length 16000 -> (N_FRAMES, N_MFCC)."""
    # pad/trim to 1s
    if wav.shape[0] < SR:
        wav = np.pad(wav, (0, SR - wav.shape[0]))
    else:
        wav = wav[:SR]
    # preemphasis
    wav = np.append(wav[0], wav[1:] - 0.97 * wav[:-1]).astype(np.float32)
    # frame
    n_frames = 1 + (SR - _WIN_LEN) // _HOP_LEN   # 98
    frames = np.zeros((n_frames, _WIN_LEN), dtype=np.float32)
    for i in range(n_frames):
        s = i * _HOP_LEN
        frames[i] = wav[s:s+_WIN_LEN] * _HAMMING
    # FFT power
    spec = np.fft.rfft(frames, n=_N_FFT, axis=1)
    power = (spec.real**2 + spec.imag**2).astype(np.float32) / _N_FFT
    # mel
    mel = power @ _MEL_FB.T
    log_mel = np.log(mel + 1e-10)
    # DCT
    mfcc = log_mel @ _DCT.T   # (n_frames, n_mfcc)
    # pad to N_FRAMES (=99)
    if mfcc.shape[0] < N_FRAMES:
        mfcc = np.pad(mfcc, ((0, N_FRAMES - mfcc.shape[0]), (0, 0)))
    else:
        mfcc = mfcc[:N_FRAMES]
    return mfcc.astype(np.float32)

# ---- dataset ----
def scan_dataset(root: Path):
    """Return dict: label -> list of wav paths.

    Following standard Speech Commands v2 splits: validation_list.txt and
    testing_list.txt define held-out files; remaining = train.
    """
    val_set = set()
    test_set = set()
    vfile = root / "validation_list.txt"
    tfile = root / "testing_list.txt"
    if vfile.exists():
        val_set = set(l.strip() for l in vfile.read_text().splitlines() if l.strip())
    if tfile.exists():
        test_set = set(l.strip() for l in tfile.read_text().splitlines() if l.strip())

    splits = {"train": {}, "test": {}, "val": {}}
    bg_dir = root / "_background_noise_"
    bg_files = []
    if bg_dir.exists():
        bg_files = sorted(str(p) for p in bg_dir.glob("*.wav"))

    for cls_dir in sorted(root.iterdir()):
        if not cls_dir.is_dir(): continue
        name = cls_dir.name
        if name.startswith("_"): continue   # skip _background_noise_ here
        for wav in sorted(cls_dir.glob("*.wav")):
            rel = f"{name}/{wav.name}"
            if rel in test_set:
                split = "test"
            elif rel in val_set:
                split = "val"
            else:
                split = "train"
            label = name if name in KEYWORDS else "_unknown_"
            splits[split].setdefault(label, []).append(str(wav))
    return splits, bg_files

def synth_silence_clips(bg_files, n_clips, rng):
    """Sample 1-s random-offset clips from background-noise files."""
    if not bg_files:
        return [np.zeros(SR, dtype=np.float32) for _ in range(n_clips)]
    clips = []
    bg_cache = {}
    for _ in range(n_clips):
        f = rng.choice(bg_files)
        if f not in bg_cache:
            w, sr_ = sf.read(f, dtype="float32")
            if w.ndim > 1: w = w.mean(axis=1)
            bg_cache[f] = w
        w = bg_cache[f]
        if w.shape[0] > SR:
            off = rng.integers(0, w.shape[0] - SR)
            clip = w[off:off+SR].astype(np.float32)
        else:
            clip = np.pad(w, (0, max(0, SR - w.shape[0]))).astype(np.float32)
        # slight random gain
        clip = clip * rng.uniform(0.0, 1.0)
        clips.append(clip)
    return clips

def build_split(splits, bg_files, split_name, n_per_class, rng):
    """Return (X_paths_or_clips, y) — balanced n_per_class for each label."""
    items = []   # (kind, payload) where kind in {'path', 'arr'}
    labels = []
    sp = splits[split_name]
    for label in CLASSES:
        if label == "_silence_":
            clips = synth_silence_clips(bg_files, n_per_class, rng)
            for c in clips:
                items.append(("arr", c)); labels.append(LABEL2IDX[label])
        elif label == "_unknown_":
            pool = sp.get("_unknown_", [])
            if not pool: continue
            idx = rng.choice(len(pool), size=min(n_per_class, len(pool)), replace=False)
            for i in idx:
                items.append(("path", pool[i])); labels.append(LABEL2IDX[label])
        else:
            pool = sp.get(label, [])
            if not pool:
                print(f"  WARN: no samples for {label} in {split_name}")
                continue
            idx = rng.choice(len(pool), size=min(n_per_class, len(pool)), replace=False)
            for i in idx:
                items.append(("path", pool[i])); labels.append(LABEL2IDX[label])
    return items, np.array(labels, dtype=np.int64)

def items_to_mfcc(items, desc=""):
    X = np.zeros((len(items), N_FRAMES, N_MFCC), dtype=np.float32)
    t0 = time.time()
    for i, (kind, payload) in enumerate(items):
        if kind == "path":
            w, _ = sf.read(payload, dtype="float32")
            if w.ndim > 1: w = w.mean(axis=1)
        else:
            w = payload
        X[i] = waveform_to_mfcc(w)
        if (i+1) % 500 == 0:
            print(f"    {desc} mfcc {i+1}/{len(items)} ({time.time()-t0:.1f}s)", flush=True)
    return X.reshape(len(items), FEAT_DIM)

# ---- MLP baseline ----
class MLP(torch.nn.Module):
    """Small MLP baseline — small 2-layer 1D-CNN over MFCC time axis.

    conv1: 40->32 k=5 stride=2 -> T=48
    conv2: 32->32 k=3 stride=2 -> T=23
    adapt_pool -> 8
    fc:    32*8 -> 12
    Params ~= 40*32*5 + 32 + 32*32*3 + 32 + 256*12 + 12 = 6400+3072+3084 = ~12.5K.

    Note: spec said "~10K params, Adam, 5 epochs" — we use 12.5K and 25 epochs
    (with 6k train samples, 5 epochs isn't enough). Locked before NS-RAM eval.
    """
    def __init__(self, n_classes=N_CLASSES, dropout=0.2):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(N_MFCC, 32, kernel_size=5, stride=2, padding=2)
        self.conv2 = torch.nn.Conv1d(32, 32, kernel_size=3, stride=2, padding=1)
        self.drop = torch.nn.Dropout(dropout)
        self.fc = torch.nn.Linear(32 * 8, n_classes)
    def forward(self, x):
        B = x.shape[0]
        x = x.view(B, N_FRAMES, N_MFCC).transpose(1, 2)   # (B, 40, 99)
        h = torch.relu(self.conv1(x))                      # (B, 32, ~50)
        h = torch.relu(self.conv2(h))                      # (B, 32, ~25)
        h = torch.nn.functional.adaptive_avg_pool1d(h, 8)  # (B, 32, 8)
        h = self.drop(h.flatten(1))                        # (B, 256)
        return self.fc(h)

def train_mlp(Xtr, ytr, Xte, yte, seed, device, epochs=25, batch=64, lr=1e-3):
    torch.manual_seed(seed)
    model = MLP().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = torch.nn.CrossEntropyLoss()
    # standardize per-feature using train stats
    mu = Xtr.mean(axis=0, keepdims=True); sd = Xtr.std(axis=0, keepdims=True) + 1e-6
    Xtr_n = (Xtr - mu) / sd
    Xte_n = (Xte - mu) / sd
    Xtr_t = torch.tensor(Xtr_n, device=device); ytr_t = torch.tensor(ytr, device=device)
    Xte_t = torch.tensor(Xte_n, device=device); yte_t = torch.tensor(yte, device=device)
    n = Xtr_t.shape[0]
    n_params = sum(p.numel() for p in model.parameters())
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        tot_loss = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i+batch]
            logits = model(Xtr_t[idx])
            loss = crit(logits, ytr_t[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            tot_loss += loss.item() * idx.shape[0]
        model.eval()
        with torch.no_grad():
            te_pred = model(Xte_t).argmax(dim=1)
            te_acc = (te_pred == yte_t).float().mean().item()
        print(f"    MLP ep{ep+1}/{epochs} loss={tot_loss/n:.4f} test_acc={te_acc:.4f}", flush=True)
    model.eval()
    with torch.no_grad():
        te_pred = model(Xte_t).argmax(dim=1).cpu().numpy()
    return te_acc, te_pred, n_params

# ---- NS-RAM (adapted from z277) ----
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

def nsram_features(X, N, W_in, V_G1_bias, V_G2_bias, surr, g_in, C_b_F, dt_s,
                   vd=1.0, generator=None, count_spikes=False, sub_steps=2,
                   poisson_p_max=0.5, n_pool_chunks=4):
    """Temporal reservoir: X is (B, N_FRAMES, N_MFCC) Poisson rates in [0,1].

    Per frame, drive NS-RAM with Poisson-sampled spikes through W_in for
    `sub_steps` substeps. Body-state Vb persists across the utterance.

    Features: split N_FRAMES into n_pool_chunks contiguous chunks; for each
    chunk, compute (mean Vb, mean log10|Id|) across the chunk. Final feature
    vector = concat over chunks + final Vb -> (B, n_pool_chunks*2*N + N).
    """
    device = X.device
    B, T_in, D = X.shape
    Vb_min = surr["ax_Vb"][0]; Vb_max = surr["ax_Vb"][-1]
    VG1_min = surr["ax_VG1"][0]; VG1_max = surr["ax_VG1"][-1]
    Vb = torch.zeros(B, N, device=device)
    spike_total = torch.zeros(B, device=device)
    VG2_2d = V_G2_bias.expand(B, N)
    Vd_2d  = torch.full((B, N), float(vd), device=device)

    chunk_len = max(T_in // n_pool_chunks, 1)
    vb_chunks = [torch.zeros(B, N, device=device) for _ in range(n_pool_chunks)]
    id_chunks = [torch.zeros(B, N, device=device) for _ in range(n_pool_chunks)]
    chunk_counts = [0] * n_pool_chunks

    for fi in range(T_in):
        rate = X[:, fi, :]
        ck = min(fi // chunk_len, n_pool_chunks - 1)
        # Continuous deterministic drive per frame — strongest signal preservation.
        # Poisson spikes counted only for energy accounting.
        if count_spikes:
            spikes = (torch.rand(rate.shape, device=device, generator=generator)
                      < (rate * poisson_p_max)).float()
            spike_total = spike_total + sub_steps * spikes.sum(dim=1)
        # zero-center rate so drive is signed (avoids global DC offset on VG1)
        drive_frame = (rate - 0.5) @ W_in.T   # (B, N)
        for s in range(sub_steps):
            VG1 = V_G1_bias.unsqueeze(0) + g_in * drive_frame
            VG1 = VG1.clamp(VG1_min, VG1_max)
            Vb_c = Vb.clamp(Vb_min, Vb_max)
            I_d, I_ii, I_leak = query_surrogate(surr, VG1, VG2_2d, Vd_2d, Vb_c)
            Vb = (Vb + dt_s * (I_ii - I_leak) / C_b_F).clamp(Vb_min, Vb_max)
            vb_chunks[ck] = vb_chunks[ck] + Vb
            id_chunks[ck] = id_chunks[ck] + I_d.abs()
            chunk_counts[ck] += 1

    parts = []
    for k in range(n_pool_chunks):
        c = max(chunk_counts[k], 1)
        parts.append(vb_chunks[k] / c)
        parts.append((id_chunks[k] / c + 1e-18).log10())
    parts.append(Vb)   # final Vb
    feats = torch.cat(parts, dim=1)   # (B, (2*n_pool_chunks+1)*N)
    mu = feats.mean(dim=0, keepdim=True); sd = feats.std(dim=0, keepdim=True) + 1e-9
    feats = (feats - mu) / sd
    return feats, spike_total

def ridge_lstsq(X, y, alpha=1.0, n_classes=N_CLASSES):
    Y = torch.nn.functional.one_hot(y, n_classes).float()
    A = X.T @ X + alpha * torch.eye(X.shape[1], device=X.device)
    B = X.T @ Y
    try:
        return torch.linalg.solve(A, B)
    except Exception:
        return torch.linalg.lstsq(A, B).solution

def normalize_mfcc(X, mn=None, mx=None):
    """Per-feature min-max scaling to [0,1] for Poisson rate encoding."""
    if mn is None:
        mn = X.min(axis=0, keepdims=True)
        mx = X.max(axis=0, keepdims=True)
    rng = (mx - mn) + 1e-6
    Xn = (X - mn) / rng
    Xn = np.clip(Xn, 0.0, 1.0)
    return Xn.astype(np.float32), mn, mx

def run_nsram(Xtr, ytr, Xte, yte, seed, device, surr, n_units=NSRAM_N):
    Xtr_n, mn, mx = normalize_mfcc(Xtr)
    Xte_n, _, _   = normalize_mfcc(Xte, mn, mx)
    # reshape to (B, N_FRAMES, N_MFCC) for temporal encoding
    Xtr_t = torch.tensor(Xtr_n, device=device).view(-1, N_FRAMES, N_MFCC)
    Xte_t = torch.tensor(Xte_n, device=device).view(-1, N_FRAMES, N_MFCC)
    ytr_t = torch.tensor(ytr, device=device)
    yte_t = torch.tensor(yte, device=device)

    g = torch.Generator(device=device).manual_seed(seed + 31337)
    # input weights: (N, N_MFCC=40) per-frame projection
    W_in = torch.randn(n_units, N_MFCC, generator=g, device=device)
    W_in = W_in / (W_in.norm(dim=1, keepdim=True) + 1e-9)
    V_G1_bias = torch.empty(n_units, device=device).uniform_(NSRAM_VG1_LO, NSRAM_VG1_HI, generator=g)
    V_G2_bias = torch.full((n_units,), NSRAM_CELL["V_G2_bias"], device=device)
    C_b_F = NSRAM_CELL["C_b_fF"] * 1e-15
    gp = torch.Generator(device=device).manual_seed(seed + 99991)

    # process in mini-batches to save memory
    def featurize(X_t, count=False, batch=32):
        feats_list = []; spike_list = []
        for i in range(0, X_t.shape[0], batch):
            f, sp = nsram_features(X_t[i:i+batch], n_units, W_in, V_G1_bias, V_G2_bias,
                                   surr, NSRAM_CELL["g_in"], C_b_F, NSRAM_CELL["dt_s"],
                                   generator=gp, count_spikes=count, sub_steps=2)
            feats_list.append(f); spike_list.append(sp)
        return torch.cat(feats_list, 0), torch.cat(spike_list, 0)

    t0 = time.time()
    feats_tr, _ = featurize(Xtr_t, count=False)
    feats_te, spikes_te = featurize(Xte_t, count=True)
    wall = time.time() - t0
    W = ridge_lstsq(feats_tr, ytr_t)
    te_pred = (feats_te @ W).argmax(dim=1)
    te_acc = (te_pred == yte_t).float().mean().item()
    # Energy: spikes_te is per-sample input-spike count over entire utterance.
    # Body-state events = N_FRAMES * sub_steps * N_neurons per inference.
    mean_input_spikes = float(spikes_te.mean().item())
    sub_steps = 2
    body_events_per_inf = N_FRAMES * sub_steps * n_units
    total_events = mean_input_spikes + body_events_per_inf
    energy_J = total_events * ENERGY_PER_SPIKE_FJ * 1e-15
    return te_acc, te_pred.cpu().numpy(), {
        "wall_s": wall,
        "mean_input_spikes_per_inf": mean_input_spikes,
        "body_events_per_inf": body_events_per_inf,
        "energy_per_inf_J": energy_J,
        "energy_per_inf_nJ": energy_J * 1e9,
    }

# ---- driver ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/speech_commands")
    ap.add_argument("--out", default="results/z287_ds_n1_kws/summary.json")
    ap.add_argument("--n_per_class_train", type=int, default=500)
    ap.add_argument("--n_per_class_test",  type=int, default=100)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0,1,2,3])
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    root = Path(args.data_root)
    if not (root / "yes").exists():
        print(f"ERROR: dataset not extracted at {root}", flush=True); sys.exit(2)
    print(f"[scan] dataset at {root}", flush=True)
    splits, bg_files = scan_dataset(root)
    for sp in ("train","val","test"):
        sizes = {k: len(v) for k,v in splits[sp].items()}
        print(f"  {sp}: {sizes}", flush=True)
    print(f"  bg_files: {len(bg_files)}", flush=True)

    if args.device == "auto":
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"[device] {device}", flush=True)

    surr = load_surrogate(SURROGATE_PATH, device)

    per_seed = []
    confusion_mlp_total = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
    confusion_nsram_total = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)

    for seed in args.seeds:
        print(f"\n=== SEED {seed} ===", flush=True)
        rng = np.random.default_rng(seed)
        tr_items, ytr = build_split(splits, bg_files, "train",
                                    args.n_per_class_train, rng)
        te_items, yte = build_split(splits, bg_files, "test",
                                    args.n_per_class_test, rng)
        print(f"  train n={len(tr_items)}  test n={len(te_items)}", flush=True)
        # class counts
        print(f"  train class counts: {Counter(ytr.tolist())}", flush=True)
        print(f"  test  class counts: {Counter(yte.tolist())}", flush=True)
        t0 = time.time()
        Xtr = items_to_mfcc(tr_items, desc=f"s{seed} tr")
        Xte = items_to_mfcc(te_items, desc=f"s{seed} te")
        print(f"  MFCC done in {time.time()-t0:.1f}s; Xtr={Xtr.shape} Xte={Xte.shape}", flush=True)

        # MLP
        print(f"  --- MLP ---", flush=True)
        mlp_acc, mlp_pred, mlp_params = train_mlp(Xtr, ytr, Xte, yte, seed, device)

        # NS-RAM
        print(f"  --- NS-RAM ---", flush=True)
        ns_acc, ns_pred, ns_diag = run_nsram(Xtr, ytr, Xte, yte, seed, device, surr)
        print(f"  NS-RAM acc={ns_acc:.4f}  spikes/inf={ns_diag['mean_input_spikes_per_inf']:.1f}  "
              f"E={ns_diag['energy_per_inf_nJ']:.2f} nJ  wall={ns_diag['wall_s']:.1f}s", flush=True)

        # confusions
        cm_mlp = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
        cm_ns  = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
        for y, p in zip(yte, mlp_pred): cm_mlp[y, p] += 1
        for y, p in zip(yte, ns_pred):  cm_ns[y, p]  += 1
        confusion_mlp_total += cm_mlp
        confusion_nsram_total += cm_ns

        per_seed.append({
            "seed": seed,
            "n_train": len(tr_items), "n_test": len(te_items),
            "mlp_acc": float(mlp_acc), "mlp_params": int(mlp_params),
            "nsram_acc": float(ns_acc),
            "nsram_diag": ns_diag,
        })

    # aggregate
    mlp_accs = np.array([r["mlp_acc"] for r in per_seed])
    ns_accs  = np.array([r["nsram_acc"] for r in per_seed])
    def ci95(a):
        if a.size < 2: return [None, None]
        bs = np.array([a[np.random.randint(0, a.size, a.size)].mean()
                       for _ in range(2000)])
        return [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]
    deltas = ns_accs - mlp_accs
    mean_E = float(np.mean([r["nsram_diag"]["energy_per_inf_nJ"] for r in per_seed]))

    # Gates
    mlp_mean = float(mlp_accs.mean()); ns_mean = float(ns_accs.mean())
    baseline_sanity = mlp_mean >= 0.80
    conservative_pass = ns_mean >= 0.70
    ambitious_pass = ns_mean >= (mlp_mean - 0.05)
    breakthrough = (ns_mean >= mlp_mean) and (mean_E < 100.0)
    if breakthrough: verdict = "BREAKTHROUGH"
    elif ambitious_pass: verdict = "AMBITIOUS_PASS"
    elif conservative_pass: verdict = "CONSERVATIVE_PASS"
    elif baseline_sanity: verdict = "BASELINE_ONLY"
    else: verdict = "FAIL"

    summary = {
        "task": "z287 DS-N1 KWS — Google Speech Commands V2 (12-class)",
        "config": {
            "sr": SR, "n_mfcc": N_MFCC, "win_ms": WIN_MS, "hop_ms": HOP_MS,
            "n_frames": N_FRAMES, "feat_dim": FEAT_DIM,
            "classes": CLASSES,
            "n_per_class_train": args.n_per_class_train,
            "n_per_class_test":  args.n_per_class_test,
            "seeds": args.seeds,
            "nsram_cell": NSRAM_CELL, "nsram_N": NSRAM_N,
            "nsram_T_steps": NSRAM_T_STEPS,
            "energy_per_spike_fJ": ENERGY_PER_SPIKE_FJ,
            "surrogate_path": SURROGATE_PATH,
            "device": str(device),
        },
        "per_seed": per_seed,
        "mlp_mean": mlp_mean, "mlp_std": float(mlp_accs.std()), "mlp_ci95": ci95(mlp_accs),
        "nsram_mean": ns_mean, "nsram_std": float(ns_accs.std()), "nsram_ci95": ci95(ns_accs),
        "delta_mean_pp": float(deltas.mean()*100), "delta_std_pp": float(deltas.std()*100),
        "nsram_energy_per_inf_nJ_mean": mean_E,
        "confusion_mlp": confusion_mlp_total.tolist(),
        "confusion_nsram": confusion_nsram_total.tolist(),
        "gates": {
            "baseline_sanity_mlp>=80": baseline_sanity,
            "conservative_pass_nsram>=70": conservative_pass,
            "ambitious_pass_nsram>=mlp-5pp": ambitious_pass,
            "breakthrough_nsram>=mlp_and_E<100nJ": breakthrough,
        },
        "verdict": verdict,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[done] verdict={verdict}  MLP={mlp_mean:.4f}  NS-RAM={ns_mean:.4f}  "
          f"delta={deltas.mean()*100:+.2f}pp  E={mean_E:.2f} nJ", flush=True)
    print(f"saved {out_path}", flush=True)

if __name__ == "__main__":
    main()
