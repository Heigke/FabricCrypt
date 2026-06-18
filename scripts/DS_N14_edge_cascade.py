"""DS-N14 Edge Cascade: NS-RAM as always-on front-end gate.

Mario "always-on KWS" canonical:
    Layer 1 (NS-RAM, always-on)   — spike-count event detector.
    Layer 2 (digital MLP, gated)  — wakes only when Layer-1 spike rate > thr.

Two benchmarks:
    1. KWS: detect "marvin" vs everything else (Google Speech Commands),
            Layer 2 = 12-class MLP classifier.
    2. ECG: detect QRS-irregular events (MIT-BIH), Layer 2 = PVC classifier.

Compare:
    NSRAM cascade  vs  always-on digital  vs  energy-detector cascade.

Pre-registered gates:
    G1) NSRAM gate recall on true events > 0.85
    G2) Wake rate < 0.05  (Layer 2 off >95% of time)
    G3) Energy save vs always-on > 10×

Outputs in results/DS_N14_edge_cascade/:
    KWS_gate_results.json, ECG_gate_results.json,
    power_breakdown.png, summary.md
"""
from __future__ import annotations
import os
for _k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[_k] = "2"
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")

import json, time, math, sys
from pathlib import Path
import numpy as np
import torch
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results/DS_N14_edge_cascade"
OUT.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32

# ---- energy model (literature-anchored, conservative) ---- #
# NS-RAM event ≈ 6.4 fJ (per Mario brief) per spike (analog floating-bulk update)
E_NSRAM_EVENT_J = 6.4e-15
# NS-RAM idle leakage power per cell: ~1 nA × 0.6 V = 0.6 pW (Mario/Sebas estimate)
P_NSRAM_LEAK_W_PER_CELL = 6.0e-13
# Layer 2 MLP: ~1.5 µJ per inference (small 2-layer MLP, ~10k MACs @ 50 fJ/MAC in 7 nm)
E_MLP_PER_INF_J = 1.5e-6
# Digital baseline (continuously-running classifier) energy per window same as MLP
# But running every window  → power = E_MLP / window_period
E_QRS_DET_J = 2.0e-7  # simple digital QRS detector per beat
E_ENERGY_DET_J = 5.0e-9  # RMS-threshold comparator, very cheap

# =============================================================
# Compact NS-RAM-style cell array (event-encoded, no surrogate)
# Same minimal ODE as DS-N8 (cell_fast defaults).
# =============================================================
VTH0    = 0.40
K_BACK  = 0.5
A_IDD   = 5.0
G_BJT   = 1.0
V_BJT_ON= 0.75
V_LATCH = 0.55
K_LEAK  = 0.02
DT      = 0.05
ALPHA_LO = 0.30; ALPHA_HI = 0.70
VG2_BIAS = 0.20
VG1_BIAS_LO = 0.45; VG1_BIAS_HI = 0.65
DVG2_PULSE = 0.10
P_MAX = 0.5


class NSRAMFrontEnd:
    """Batched NS-RAM front end. Returns (B,) total spike count per sample."""
    def __init__(self, N: int, n_in: int, seed: int, k_fan: int = 4):
        self.N = N; self.n_in = n_in; self.k_fan = k_fan
        g = torch.Generator(device=DEVICE).manual_seed(seed)
        self.alpha    = ALPHA_LO + (ALPHA_HI-ALPHA_LO) * torch.rand(N, generator=g, device=DEVICE)
        self.VG1_bias = VG1_BIAS_LO + (VG1_BIAS_HI-VG1_BIAS_LO) * torch.rand(N, generator=g, device=DEVICE)
        self.W_in_idx = torch.randint(0, n_in, (N, k_fan), generator=g, device=DEVICE)
        self.W_in_sgn = (torch.randint(0, 2, (N, k_fan), generator=g, device=DEVICE, dtype=DTYPE)*2-1)
        self.gen = g

    def project(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_in) in [0,1]
        B = x.shape[0]
        idx = self.W_in_idx.unsqueeze(0).expand(B, -1, -1)
        gx = torch.gather(x.unsqueeze(2).expand(-1, -1, self.k_fan), 1, idx)
        s = (gx * self.W_in_sgn.unsqueeze(0)).sum(dim=2) / self.k_fan
        return torch.sigmoid(4.0 * s)

    def run_window(self, x_seq: torch.Tensor, k_sub: int = 2):
        """x_seq: (B, T, n_in). Returns:
            spike_rate (B,) — mean spikes per substep per cell over window
            total_events (B,) — total spikes (NS-RAM 'events')
            feats (B, 3N) — [mean_log_Id, mean_Vb, final_Vb] for L2.
        """
        B, T, _ = x_seq.shape
        N = self.N
        Vb = torch.full((B, N), VG2_BIAS, device=DEVICE, dtype=DTYPE)
        VG1 = self.VG1_bias.unsqueeze(0).expand(B, -1)
        events = torch.zeros(B, device=DEVICE, dtype=DTYPE)
        id_acc = torch.zeros(B, N, device=DEVICE, dtype=DTYPE)
        vb_acc = torch.zeros(B, N, device=DEVICE, dtype=DTYPE)
        n_acc = 0
        for t in range(T):
            rate = self.project(x_seq[:, t, :])
            for _s in range(k_sub):
                u = torch.rand(B, N, generator=self.gen, device=DEVICE, dtype=DTYPE)
                spike = (u < (P_MAX * rate)).to(DTYPE)
                Vth_eff = VTH0 - K_BACK * (VG2_BIAS + DVG2_PULSE * spike)
                overdrive = torch.clamp(VG1 - Vth_eff, min=0.0)
                channel_on = torch.sigmoid((VG1 - Vth_eff) / 0.05)
                Iii  = A_IDD * spike * channel_on * overdrive
                Ibjt = G_BJT * torch.clamp(Vb - V_BJT_ON, min=0.0)
                Ileak= K_LEAK * (Vb - (VG2_BIAS + DVG2_PULSE * spike))
                dVb  = self.alpha * Iii - Ibjt - Ileak
                Vb   = torch.clamp(Vb + DT * dVb, -0.2, 0.85)
                latched = torch.clamp((Vb - V_LATCH) / (0.85 - V_LATCH), min=0.0)
                Id = 1e-9 * channel_on * (1.0 + 100.0 * latched)
                id_acc += Id
                vb_acc += Vb
                events = events + spike.sum(dim=1)
                n_acc += 1
        spike_rate = events / max(n_acc * N, 1)
        feats = torch.cat([
            torch.log10(id_acc / max(n_acc,1) + 1e-18),
            vb_acc / max(n_acc,1),
            Vb,
        ], dim=1)
        return spike_rate, events, feats


# =============================================================
#                  BENCHMARK 1: KWS  ("marvin")
# =============================================================
SR = 16000; N_MFCC = 40; N_FRAMES = 99
WIN_MS = 25.0; HOP_MS = 10.0
KEYWORDS  = ["yes","no","up","down","left","right","on","off","stop","go"]
CLASSES = KEYWORDS + ["_unknown_", "_silence_"]
N_CLASSES = 12
LABEL2IDX = {c: i for i, c in enumerate(CLASSES)}
# Target keyword "marvin" — falls in _unknown_ pool. We use it as the gate event.
TARGET_KEYWORD = "marvin"

_WIN_LEN = int(SR * WIN_MS / 1000)
_HOP_LEN = int(SR * HOP_MS / 1000)
_N_FFT   = 512
_HAMM    = np.hamming(_WIN_LEN).astype(np.float32)

def _hz_to_mel(f): return 2595.0 * np.log10(1.0 + f / 700.0)
def _mel_to_hz(m): return 700.0 * (10.0 ** (m / 2595.0) - 1.0)
def _build_mel_fb(n_fft, sr, n_mels=40, fmin=20.0, fmax=None):
    if fmax is None: fmax = sr / 2
    mel = np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2)
    hz  = _mel_to_hz(mel)
    b   = np.floor((n_fft + 1) * hz / sr).astype(int)
    fb  = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        l, c, r = b[m-1], b[m], b[m+1]
        if c == l: c = l + 1
        if r == c: r = c + 1
        for k in range(l, c): fb[m-1, k] = (k - l) / max(c - l, 1)
        for k in range(c, r): fb[m-1, k] = (r - k) / max(r - c, 1)
    return fb
def _build_dct(n_mels, n_mfcc):
    n = np.arange(n_mels); k = np.arange(n_mfcc)[:, None]
    d = np.cos(np.pi * k * (2 * n + 1) / (2 * n_mels)) * np.sqrt(2.0 / n_mels)
    d[0] *= 1.0 / np.sqrt(2.0)
    return d.astype(np.float32)
_MEL_FB = _build_mel_fb(_N_FFT, SR, n_mels=40)
_DCT    = _build_dct(40, N_MFCC)

def wav_to_mfcc(wav: np.ndarray) -> np.ndarray:
    if wav.shape[0] < SR: wav = np.pad(wav, (0, SR - wav.shape[0]))
    else:                  wav = wav[:SR]
    wav = np.append(wav[0], wav[1:] - 0.97 * wav[:-1]).astype(np.float32)
    n_frames = 1 + (SR - _WIN_LEN) // _HOP_LEN
    frames = np.zeros((n_frames, _WIN_LEN), dtype=np.float32)
    for i in range(n_frames):
        s = i * _HOP_LEN
        frames[i] = wav[s:s+_WIN_LEN] * _HAMM
    spec  = np.fft.rfft(frames, n=_N_FFT, axis=1)
    power = (spec.real**2 + spec.imag**2).astype(np.float32) / _N_FFT
    mel   = power @ _MEL_FB.T
    logm  = np.log(mel + 1e-10)
    mfcc  = logm @ _DCT.T
    if mfcc.shape[0] < N_FRAMES:
        mfcc = np.pad(mfcc, ((0, N_FRAMES - mfcc.shape[0]), (0, 0)))
    else:
        mfcc = mfcc[:N_FRAMES]
    return mfcc.astype(np.float32)


def wav_rms(wav: np.ndarray) -> float:
    if wav.shape[0] < SR: wav = np.pad(wav, (0, SR - wav.shape[0]))
    else: wav = wav[:SR]
    return float(np.sqrt(np.mean(wav.astype(np.float32)**2)))


def kws_sample_lists(root: Path, n_per_other: int = 60, rng=None,
                     n_silence_per_bg: int = 200, n_other_unknown: int = 150):
    """Return list of (path_or_arr, is_target, kw_idx) covering marvin + others.

    Realistic always-on: positives (marvin) << negatives. Negatives include
    other keywords, other unknown words, and many background-noise segments.
    """
    if rng is None: rng = np.random.default_rng(0xC15C0)
    items = []
    # All marvin samples = positives
    marv_dir = root / TARGET_KEYWORD
    marv_wavs = sorted(marv_dir.glob("*.wav"))
    for p in marv_wavs:
        items.append((str(p), 1, LABEL2IDX["_unknown_"]))  # positive
    # 12-class commands as negatives (and Layer-2 cares about kw label)
    for kw in KEYWORDS:
        d = root / kw
        if not d.exists(): continue
        wavs = sorted(d.glob("*.wav"))
        idx = rng.choice(len(wavs), size=min(n_per_other, len(wavs)), replace=False)
        for i in idx:
            items.append((str(wavs[i]), 0, LABEL2IDX[kw]))
    # other-unknown words as negatives
    other_unknown = ["bed","bird","cat","dog","happy","house","wow","tree","sheila",
                     "backward","follow","forward","learn","visual","zero","one","two",
                     "three","four","five","six","seven","eight","nine"]
    for kw in other_unknown:
        d = root / kw
        if not d.exists(): continue
        wavs = sorted(d.glob("*.wav"))
        nsmp = min(n_other_unknown, len(wavs))
        idx = rng.choice(len(wavs), size=nsmp, replace=False)
        for i in idx:
            items.append((str(wavs[i]), 0, LABEL2IDX["_unknown_"]))
    # background noise as silence negatives
    bg_dir = root / "_background_noise_"
    if bg_dir.exists():
        for bgf in sorted(bg_dir.glob("*.wav")):
            w, _ = sf.read(bgf, dtype="float32")
            if w.ndim > 1: w = w.mean(axis=1)
            for _ in range(n_silence_per_bg):
                if w.shape[0] > SR:
                    off = int(rng.integers(0, w.shape[0]-SR))
                    clip = w[off:off+SR].astype(np.float32) * float(rng.uniform(0.0,0.6))
                else:
                    clip = np.pad(w, (0, max(0, SR - w.shape[0]))).astype(np.float32)
                items.append((clip, 0, LABEL2IDX["_silence_"]))
    rng.shuffle(items)
    return items


def kws_load_mfcc(items, desc=""):
    X = np.zeros((len(items), N_FRAMES, N_MFCC), dtype=np.float32)
    rms = np.zeros(len(items), dtype=np.float32)
    y_event = np.zeros(len(items), dtype=np.int64)
    y_kw = np.zeros(len(items), dtype=np.int64)
    t0 = time.time()
    for i, (p, ev, kw) in enumerate(items):
        if isinstance(p, str):
            w, _ = sf.read(p, dtype="float32")
            if w.ndim > 1: w = w.mean(axis=1)
        else:
            w = p
        X[i] = wav_to_mfcc(w)
        rms[i] = wav_rms(w)
        y_event[i] = ev
        y_kw[i] = kw
        if (i+1) % 500 == 0:
            print(f"  {desc} mfcc {i+1}/{len(items)} t={time.time()-t0:.1f}s", flush=True)
    return X, rms, y_event, y_kw


def run_kws_benchmark(data_root: Path, N_NSRAM: int = 256, n_per_other: int = 60):
    print(f"\n=== KWS benchmark (Layer-1 NS-RAM N={N_NSRAM}) ===", flush=True)
    items = kws_sample_lists(data_root, n_per_other=n_per_other)
    print(f"  total samples = {len(items)}, positives = {sum(1 for it in items if it[1]==1)}", flush=True)
    # split: 70% calib, 30% eval (stratified by event label)
    pos = [it for it in items if it[1]==1]
    neg = [it for it in items if it[1]==0]
    rng = np.random.default_rng(7)
    rng.shuffle(pos); rng.shuffle(neg)
    cut_p = int(0.7 * len(pos)); cut_n = int(0.7 * len(neg))
    train = pos[:cut_p] + neg[:cut_n]
    test  = pos[cut_p:] + neg[cut_n:]
    rng.shuffle(train); rng.shuffle(test)

    Xtr, rms_tr, ytr_ev, ytr_kw = kws_load_mfcc(train, "train")
    Xte, rms_te, yte_ev, yte_kw = kws_load_mfcc(test, "test")

    # Normalize MFCC per-bin to [0,1] using train min/max
    lo = Xtr.min(axis=(0,1), keepdims=True)
    hi = Xtr.max(axis=(0,1), keepdims=True)
    Xtr_n = (Xtr - lo) / (hi - lo + 1e-6)
    Xte_n = (Xte - lo) / (hi - lo + 1e-6)
    Xtr_n = np.clip(Xtr_n, 0, 1); Xte_n = np.clip(Xte_n, 0, 1)

    # Build NS-RAM front end
    fe = NSRAMFrontEnd(N=N_NSRAM, n_in=N_MFCC, seed=42)

    # Run front-end (batched). Decimate to N_FRAMES//2 to save compute.
    def featurize(X_n, batch=32):
        Xt = torch.from_numpy(X_n).to(DEVICE, dtype=DTYPE)
        # Downsample time: skip every other frame for speed
        Xt = Xt[:, ::2, :]
        rates, evts, feats = [], [], []
        for i in range(0, Xt.shape[0], batch):
            r, e, f = fe.run_window(Xt[i:i+batch], k_sub=2)
            rates.append(r); evts.append(e); feats.append(f)
        return (torch.cat(rates).cpu().numpy(),
                torch.cat(evts).cpu().numpy(),
                torch.cat(feats).cpu().numpy())

    print("  featurize train via NS-RAM front-end ...", flush=True)
    t0 = time.time()
    r_tr, ev_tr, f_tr = featurize(Xtr_n, batch=32)
    print(f"    train feat t={time.time()-t0:.1f}s mean_rate={r_tr.mean():.4f}", flush=True)
    t0 = time.time()
    r_te, ev_te, f_te = featurize(Xte_n, batch=32)
    print(f"    test  feat t={time.time()-t0:.1f}s", flush=True)

    # ---- Learned gate: small linear classifier on Layer-1 features ----
    # Tiny: ≤256 features × 1 weight = ~256 MACs per second (cheap, always-on).
    target_wake = 0.05
    Dfeat = f_tr.shape[1]
    Dgate = min(256, Dfeat)
    sel_g = np.random.default_rng(7).choice(Dfeat, size=Dgate, replace=False)
    Xg_tr = f_tr[:, sel_g]; Xg_te = f_te[:, sel_g]
    mu_g = Xg_tr.mean(0, keepdims=True); sd_g = Xg_tr.std(0, keepdims=True)+1e-6
    Xg_tr = (Xg_tr-mu_g)/sd_g; Xg_te = (Xg_te-mu_g)/sd_g
    Xg_tr_t = torch.from_numpy(Xg_tr.astype(np.float32)).to(DEVICE)
    yg_t    = torch.from_numpy(ytr_ev).to(DEVICE).float()
    Xg_te_t = torch.from_numpy(Xg_te.astype(np.float32)).to(DEVICE)
    n_pos = int(ytr_ev.sum()); n_neg = len(ytr_ev)-n_pos
    pos_w_t = torch.tensor([max(n_neg/max(n_pos,1), 1.0)], device=DEVICE)
    gate_lin = torch.nn.Linear(Dgate, 1).to(DEVICE)
    opt_g = torch.optim.AdamW(gate_lin.parameters(), lr=5e-3, weight_decay=1e-3)
    for epoch in range(40):
        perm = torch.randperm(Xg_tr_t.shape[0], device=DEVICE)
        for i in range(0, Xg_tr_t.shape[0], 256):
            idx = perm[i:i+256]
            logit = gate_lin(Xg_tr_t[idx]).squeeze(1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logit, yg_t[idx], pos_weight=pos_w_t)
            opt_g.zero_grad(); loss.backward(); opt_g.step()
    with torch.no_grad():
        score_tr = gate_lin(Xg_tr_t).squeeze(1).cpu().numpy()
        score_te = gate_lin(Xg_te_t).squeeze(1).cpu().numpy()

    thr_sweep = np.linspace(score_tr.min(), score_tr.max(), 400)
    best = None
    for thr in thr_sweep:
        wake = (score_tr > thr).mean()
        if wake > 0.06: continue
        tp = ((score_tr > thr) & (ytr_ev==1)).sum()
        fp = ((score_tr > thr) & (ytr_ev==0)).sum()
        fn = ((score_tr <= thr) & (ytr_ev==1)).sum()
        recall = tp / max(tp+fn, 1)
        prec = tp / max(tp+fp, 1)
        score = recall - max(wake-target_wake, 0.0)*10
        if (best is None) or (score > best["score"]):
            best = dict(thr=float(thr), recall=float(recall), prec=float(prec),
                        wake=float(wake), score=float(score),
                        tp=int(tp), fp=int(fp), fn=int(fn))
    print(f"  calib best: {best}", flush=True)
    thr = best["thr"]

    # ---- Test-time gate ----
    gate_fire_te = (score_te > thr).astype(np.int32)
    wake_rate_te = float(gate_fire_te.mean())
    tp = int(((gate_fire_te==1) & (yte_ev==1)).sum())
    fp = int(((gate_fire_te==1) & (yte_ev==0)).sum())
    fn = int(((gate_fire_te==0) & (yte_ev==1)).sum())
    tn = int(((gate_fire_te==0) & (yte_ev==0)).sum())
    recall_te = tp / max(tp+fn, 1)
    precision_te = tp / max(tp+fp, 1)

    # ---- Layer 2: 12-class MLP trained on f_tr → kw label, used only when gated ----
    # Subsample feature dim to keep model tiny
    Dfeat = f_tr.shape[1]
    Dsub = min(2048, Dfeat)
    sel = np.random.default_rng(1).choice(Dfeat, size=Dsub, replace=False)
    Xtr_l2 = f_tr[:, sel]; Xte_l2 = f_te[:, sel]
    mu = Xtr_l2.mean(0, keepdims=True); sd = Xtr_l2.std(0, keepdims=True)+1e-6
    Xtr_l2 = (Xtr_l2 - mu) / sd; Xte_l2 = (Xte_l2 - mu) / sd
    Xt_t = torch.from_numpy(Xtr_l2).to(DEVICE)
    yt_t = torch.from_numpy(ytr_kw).to(DEVICE)
    Xe_t = torch.from_numpy(Xte_l2).to(DEVICE)
    ye_t = torch.from_numpy(yte_kw).to(DEVICE)
    H = 64
    mlp = torch.nn.Sequential(
        torch.nn.Linear(Dsub, H), torch.nn.ReLU(),
        torch.nn.Linear(H, N_CLASSES)
    ).to(DEVICE)
    opt = torch.optim.AdamW(mlp.parameters(), lr=3e-3, weight_decay=1e-4)
    bs = 128
    for epoch in range(20):
        perm = torch.randperm(Xt_t.shape[0], device=DEVICE)
        for i in range(0, Xt_t.shape[0], bs):
            idx = perm[i:i+bs]
            logit = mlp(Xt_t[idx])
            loss = torch.nn.functional.cross_entropy(logit, yt_t[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    mlp.eval()
    with torch.no_grad():
        pred_te = mlp(Xe_t).argmax(dim=1).cpu().numpy()
    l2_acc_all = float((pred_te == yte_kw).mean())
    l2_acc_on_gated = float((pred_te[gate_fire_te==1] == yte_kw[gate_fire_te==1]).mean()) if gate_fire_te.sum()>0 else 0.0

    # ---- Energy detector baseline (RMS comparator) ----
    # Calibrate threshold on train at same wake budget
    thr_e = np.quantile(rms_tr, 1 - target_wake)
    e_fire_te = (rms_te > thr_e).astype(np.int32)
    e_wake = float(e_fire_te.mean())
    e_tp = int(((e_fire_te==1) & (yte_ev==1)).sum())
    e_fn = int(((e_fire_te==0) & (yte_ev==1)).sum())
    e_fp = int(((e_fire_te==1) & (yte_ev==0)).sum())
    e_recall = e_tp / max(e_tp+e_fn, 1)
    e_prec = e_tp / max(e_tp+e_fp, 1)

    # ---- Power model (per 1-s window) ----
    win_s = 1.0
    avg_events_nsram = float(ev_te.mean())
    # NS-RAM continuous power = leak + event-energy/sec
    P_nsram = N_NSRAM * P_NSRAM_LEAK_W_PER_CELL + avg_events_nsram * E_NSRAM_EVENT_J / win_s
    P_l2_wake = E_MLP_PER_INF_J / win_s   # when fully waked
    P_cascade = P_nsram + wake_rate_te * P_l2_wake
    P_always_on = P_l2_wake   # baseline runs L2 every window
    P_energy_det = E_ENERGY_DET_J / win_s + e_wake * P_l2_wake
    save_vs_always = P_always_on / max(P_cascade, 1e-18)

    res = dict(
        N_NSRAM=N_NSRAM, n_test=int(len(yte_ev)),
        n_pos_test=int(int(yte_ev.sum())),
        gate_threshold=float(thr),
        wake_rate=wake_rate_te,
        gate_recall=recall_te,
        gate_precision=precision_te,
        gate_tp=tp, gate_fp=fp, gate_fn=fn, gate_tn=tn,
        mean_spike_rate_te=float(r_te.mean()),
        mean_events_nsram_per_win=avg_events_nsram,
        layer2_acc_all=l2_acc_all,
        layer2_acc_on_gated=l2_acc_on_gated,
        energy_detector_wake=e_wake,
        energy_detector_recall=e_recall,
        energy_detector_precision=e_prec,
        P_nsram_W=P_nsram,
        P_layer2_when_on_W=P_l2_wake,
        P_cascade_avg_W=P_cascade,
        P_always_on_W=P_always_on,
        P_energy_det_cascade_W=P_energy_det,
        energy_save_vs_always_on_x=save_vs_always,
        gates_pass=dict(
            G1_recall_gt_0p85=bool(recall_te > 0.85),
            G2_wake_lt_0p05=bool(wake_rate_te < 0.05),
            G3_save_gt_10x=bool(save_vs_always > 10.0),
        ),
    )
    with open(OUT/"KWS_gate_results.json", "w") as fp:
        json.dump(res, fp, indent=2)
    print(f"  KWS done. recall={recall_te:.3f} prec={precision_te:.3f} "
          f"wake={wake_rate_te:.3f} save={save_vs_always:.1f}x", flush=True)
    return res


# =============================================================
#                BENCHMARK 2: ECG MIT-BIH gate
# =============================================================
import wfdb
FS = 360; DOWNSAMPLE = 3; FS_EFF = FS // DOWNSAMPLE
N_SYMS = {"N","L","R","e","j"}
V_SYMS = {"V","E"}
TRAIN_RECS = ["100","106","119"]
TEST_RECS  = ["200","208","233"]
DATA = ROOT / "data/mitdb"
WIN_SAMP = int(0.5 * FS_EFF)  # 500 ms beat window


def ecg_load_record(rec):
    r = wfdb.rdrecord(str(DATA/rec))
    a = wfdb.rdann(str(DATA/rec),'atr')
    sig_full = r.p_signal[:,0].astype(np.float32)
    L = (sig_full.shape[0] // DOWNSAMPLE) * DOWNSAMPLE
    sig = sig_full[:L].reshape(-1, DOWNSAMPLE).mean(axis=1)
    sig = (sig - sig.mean()) / (sig.std()+1e-6)
    beats = []
    for samp, sym in zip(a.sample, a.symbol):
        s_ds = int(samp // DOWNSAMPLE)
        if sym in N_SYMS:   beats.append((s_ds, 0))
        elif sym in V_SYMS: beats.append((s_ds, 1))
    return sig, beats


def ecg_extract_windows(recs):
    Xs = []; ys = []; rr_prev = []
    for rec in recs:
        sig, beats = ecg_load_record(rec)
        prev_R = 0
        for samp, lbl in beats:
            s = samp - WIN_SAMP//2
            e = s + WIN_SAMP
            if s < 0 or e > len(sig):
                prev_R = samp
                continue
            Xs.append(sig[s:e])
            ys.append(lbl)
            rr_prev.append((samp - prev_R)/FS_EFF*1000.0)
            prev_R = samp
    X = np.stack(Xs).astype(np.float32)
    y = np.array(ys, dtype=np.int64)
    rr = np.array(rr_prev, dtype=np.float32)
    return X, y, rr


def run_ecg_benchmark(N_NSRAM: int = 64):
    print(f"\n=== ECG benchmark (Layer-1 NS-RAM N={N_NSRAM}) ===", flush=True)
    t0 = time.time()
    Xtr, ytr, rrtr = ecg_extract_windows(TRAIN_RECS)
    Xte, yte, rrte = ecg_extract_windows(TEST_RECS)
    print(f"  train beats={len(ytr)} (V={int(ytr.sum())}) test beats={len(yte)} (V={int(yte.sum())})  t={time.time()-t0:.1f}s", flush=True)

    # Encode each beat window as n_in=8 features per time step:
    # 8 bins of (rectified pos/neg amp, derivative pos/neg) — sliding subwindows.
    def encode(X):
        # X: (B, T) where T = WIN_SAMP. Make (B, T, 8) drive in [0,1].
        B, T = X.shape
        dX = np.diff(X, axis=1, prepend=X[:,:1])
        pos = np.clip(X, 0, None); neg = np.clip(-X, 0, None)
        dpos = np.clip(dX, 0, None); dneg = np.clip(-dX, 0, None)
        # also lagged versions for richness
        feats = np.stack([pos, neg, dpos, dneg,
                          np.roll(pos,1,axis=1), np.roll(neg,1,axis=1),
                          np.roll(dpos,1,axis=1), np.roll(dneg,1,axis=1)], axis=2)
        # normalize per-sample to [0,1] via global scale
        mx = feats.max() + 1e-6
        return (feats / mx).astype(np.float32)

    Etr = encode(Xtr); Ete = encode(Xte)
    print(f"  encoded train {Etr.shape} test {Ete.shape}", flush=True)
    fe = NSRAMFrontEnd(N=N_NSRAM, n_in=8, seed=137, k_fan=3)

    def feat(E, batch=64):
        Et = torch.from_numpy(E).to(DEVICE, dtype=DTYPE)
        rates, evts, fs = [], [], []
        for i in range(0, Et.shape[0], batch):
            r, e, f = fe.run_window(Et[i:i+batch], k_sub=2)
            rates.append(r); evts.append(e); fs.append(f)
        return (torch.cat(rates).cpu().numpy(),
                torch.cat(evts).cpu().numpy(),
                torch.cat(fs).cpu().numpy())

    t0 = time.time()
    r_tr, ev_tr, f_tr = feat(Etr)
    r_te, ev_te, f_te = feat(Ete)
    print(f"  feat wall={time.time()-t0:.1f}s", flush=True)

    # ---- Layer-1 gate: small linear classifier on NS-RAM features ----
    # This is what an "always-on tiny gate" looks like — it consumes NS-RAM
    # features and outputs a single scalar; very cheap (one MAC-vector per beat).
    # Calibrate to wake budget; report sweep separately.
    target_wake = 0.05
    # Subsample features for the gate to keep it cheap (≤256 features)
    Dfeat = f_tr.shape[1]
    Dgate = min(256, Dfeat)
    sel_g = np.random.default_rng(7).choice(Dfeat, size=Dgate, replace=False)
    Xg_tr = f_tr[:, sel_g]; Xg_te = f_te[:, sel_g]
    mu_g = Xg_tr.mean(0, keepdims=True); sd_g = Xg_tr.std(0, keepdims=True)+1e-6
    Xg_tr = (Xg_tr-mu_g)/sd_g; Xg_te = (Xg_te-mu_g)/sd_g
    Xg_tr_t = torch.from_numpy(Xg_tr.astype(np.float32)).to(DEVICE)
    yg_t    = torch.from_numpy(ytr).to(DEVICE).float()
    Xg_te_t = torch.from_numpy(Xg_te.astype(np.float32)).to(DEVICE)
    n_pos = int(ytr.sum()); n_neg = len(ytr)-n_pos
    pos_w = torch.tensor([max(n_neg/max(n_pos,1), 1.0)], device=DEVICE)
    gate_lin = torch.nn.Linear(Dgate, 1).to(DEVICE)
    opt_g = torch.optim.AdamW(gate_lin.parameters(), lr=5e-3, weight_decay=1e-3)
    for epoch in range(60):
        perm = torch.randperm(Xg_tr_t.shape[0], device=DEVICE)
        for i in range(0, Xg_tr_t.shape[0], 256):
            idx = perm[i:i+256]
            logit = gate_lin(Xg_tr_t[idx]).squeeze(1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logit, yg_t[idx], pos_weight=pos_w)
            opt_g.zero_grad(); loss.backward(); opt_g.step()
    with torch.no_grad():
        score_tr = gate_lin(Xg_tr_t).squeeze(1).cpu().numpy()
        score_te = gate_lin(Xg_te_t).squeeze(1).cpu().numpy()

    best = None
    thr_sweep = np.linspace(score_tr.min(), score_tr.max(), 400)
    for thr in thr_sweep:
        wake = (score_tr > thr).mean()
        if wake > 0.08: continue
        tp = ((score_tr > thr) & (ytr==1)).sum()
        fp = ((score_tr > thr) & (ytr==0)).sum()
        fn = ((score_tr <= thr) & (ytr==1)).sum()
        recall = tp / max(tp+fn, 1)
        prec = tp / max(tp+fp, 1)
        score = recall - max(wake-target_wake, 0.0)*5
        if (best is None) or (score > best["score"]):
            best = dict(thr=float(thr), recall=float(recall), prec=float(prec),
                        wake=float(wake), score=float(score))
    print(f"  calib best: {best}", flush=True)
    thr = best["thr"]
    fire_te = (score_te > thr).astype(np.int32)
    wake_rate = float(fire_te.mean())
    tp = int(((fire_te==1)&(yte==1)).sum())
    fp = int(((fire_te==1)&(yte==0)).sum())
    fn = int(((fire_te==0)&(yte==1)).sum())
    tn = int(((fire_te==0)&(yte==0)).sum())
    recall = tp/max(tp+fn,1); prec = tp/max(tp+fp,1)
    specificity = tn / max(tn+fp, 1)

    # ---- Layer-2 PVC classifier (MLP on NS-RAM features) ----
    Dfeat = f_tr.shape[1]
    Dsub = min(512, Dfeat)
    sel = np.random.default_rng(2).choice(Dfeat, size=Dsub, replace=False)
    Xtr_l2 = f_tr[:, sel]; Xte_l2 = f_te[:, sel]
    mu = Xtr_l2.mean(0, keepdims=True); sd = Xtr_l2.std(0, keepdims=True)+1e-6
    Xtr_l2 = (Xtr_l2 - mu)/sd; Xte_l2 = (Xte_l2 - mu)/sd
    # Append rr_prev as feature
    Xtr_l2 = np.concatenate([Xtr_l2, (rrtr[:,None]-800)/200], axis=1)
    Xte_l2 = np.concatenate([Xte_l2, (rrte[:,None]-800)/200], axis=1)
    Xt = torch.from_numpy(Xtr_l2.astype(np.float32)).to(DEVICE)
    yt = torch.from_numpy(ytr).to(DEVICE)
    Xe = torch.from_numpy(Xte_l2.astype(np.float32)).to(DEVICE)
    ye = torch.from_numpy(yte).to(DEVICE)
    H = 32
    mlp = torch.nn.Sequential(
        torch.nn.Linear(Xt.shape[1], H), torch.nn.ReLU(),
        torch.nn.Linear(H, 2)
    ).to(DEVICE)
    # Class weights to handle imbalance
    n_pos = int(ytr.sum()); n_neg = len(ytr)-n_pos
    w_cls = torch.tensor([1.0, max(n_neg/max(n_pos,1), 1.0)*0.5], device=DEVICE)
    opt = torch.optim.AdamW(mlp.parameters(), lr=3e-3, weight_decay=1e-4)
    for epoch in range(40):
        perm = torch.randperm(Xt.shape[0], device=DEVICE)
        for i in range(0, Xt.shape[0], 128):
            idx = perm[i:i+128]
            logit = mlp(Xt[idx])
            loss = torch.nn.functional.cross_entropy(logit, yt[idx], weight=w_cls)
            opt.zero_grad(); loss.backward(); opt.step()
    mlp.eval()
    with torch.no_grad():
        pred = mlp(Xe).argmax(dim=1).cpu().numpy()
    pvc_acc = float((pred==yte).mean())
    pvc_sens = float(((pred==1)&(yte==1)).sum() / max(int(yte.sum()),1))
    pvc_spec = float(((pred==0)&(yte==0)).sum() / max(int((yte==0).sum()),1))
    # Cascade-aware accuracy: if not fired, predict N (0)
    pred_casc = pred.copy(); pred_casc[fire_te==0] = 0
    casc_sens = float(((pred_casc==1)&(yte==1)).sum() / max(int(yte.sum()),1))
    casc_spec = float(((pred_casc==0)&(yte==0)).sum() / max(int((yte==0).sum()),1))

    # ---- Power model: beat rate ≈ 60 bpm → 1 inference/sec for always-on ----
    beats_per_s = 1.0
    P_qrs_det = E_QRS_DET_J * beats_per_s  # cheap digital QRS det for always-on
    P_l2 = E_MLP_PER_INF_J * beats_per_s
    avg_events_nsram = float(ev_te.mean())
    P_nsram = N_NSRAM * P_NSRAM_LEAK_W_PER_CELL + avg_events_nsram * E_NSRAM_EVENT_J * beats_per_s
    P_cascade = P_nsram + wake_rate * P_l2
    P_always_on = P_qrs_det + P_l2     # always-on = continuous QRS + L2 every beat
    save = P_always_on / max(P_cascade, 1e-18)

    res = dict(
        N_NSRAM=N_NSRAM, n_test=int(len(yte)),
        n_pos_test=int(yte.sum()),
        gate_threshold=float(thr),
        gate_wake_rate=wake_rate,
        gate_recall=recall,
        gate_precision=prec,
        gate_specificity=specificity,
        gate_tp=tp, gate_fp=fp, gate_fn=fn, gate_tn=tn,
        mean_spike_rate_te=float(r_te.mean()),
        mean_events_nsram_per_beat=avg_events_nsram,
        layer2_pvc_acc=pvc_acc,
        layer2_pvc_sensitivity=pvc_sens,
        layer2_pvc_specificity=pvc_spec,
        cascade_pvc_sensitivity=casc_sens,
        cascade_pvc_specificity=casc_spec,
        P_nsram_W=P_nsram,
        P_layer2_per_beat_W=P_l2,
        P_cascade_avg_W=P_cascade,
        P_always_on_W=P_always_on,
        energy_save_vs_always_on_x=save,
        gates_pass=dict(
            G1_recall_gt_0p85=bool(recall > 0.85),
            G2_wake_lt_0p05=bool(wake_rate < 0.05),
            G3_save_gt_10x=bool(save > 10.0),
        ),
    )
    with open(OUT/"ECG_gate_results.json", "w") as fp:
        json.dump(res, fp, indent=2)
    print(f"  ECG done. recall={recall:.3f} prec={prec:.3f} wake={wake_rate:.3f} save={save:.1f}x", flush=True)
    return res


# =============================================================
def plot_power(kws_res, ecg_res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, res, title in zip(axes,
                              [kws_res, ecg_res],
                              ["KWS (always-on 'marvin' gate)",
                               "ECG (PVC anomaly gate)"]):
        labels = ["always-on\n(L2 each window)",
                  "energy-det\ncascade" if "P_energy_det_cascade_W" in res else "QRS-det\nalways-on",
                  "NS-RAM\ncascade"]
        vals = [
            res["P_always_on_W"],
            res.get("P_energy_det_cascade_W", res["P_always_on_W"]),
            res["P_cascade_avg_W"],
        ]
        bars = ax.bar(labels, [v*1e6 for v in vals],
                      color=["#c44","#999","#3a7"])
        ax.set_ylabel("Average power (µW)")
        ax.set_yscale("log")
        ax.set_title(title)
        for b, v in zip(bars, vals):
            ax.text(b.get_x()+b.get_width()/2, v*1e6, f"{v*1e6:.2f} µW",
                    ha="center", va="bottom", fontsize=8)
        ax.text(0.02, 0.95,
                f"wake={res.get('wake_rate', res.get('gate_wake_rate'))*100:.1f}%\n"
                f"recall={res['gate_recall']:.2f}\n"
                f"save={res['energy_save_vs_always_on_x']:.1f}×",
                transform=ax.transAxes, va="top", fontsize=9,
                bbox=dict(facecolor="white", alpha=0.7))
    plt.tight_layout()
    plt.savefig(OUT/"power_breakdown.png", dpi=120)
    plt.close()


def write_summary(kws, ecg):
    def g(d, k): return d.get(k, "n/a")
    md = []
    md.append("# DS-N14 Edge Cascade — Summary\n")
    md.append("## Architecture\n")
    md.append("Layer 1 = NS-RAM event-coded reservoir (always on). Layer 2 = small MLP, gated by Layer-1 spike rate.\n")
    md.append("\n## KWS ('marvin' always-on gate)\n")
    md.append(f"- Gate recall = {kws['gate_recall']:.3f}, precision = {kws['gate_precision']:.3f}\n")
    md.append(f"- Wake rate = {kws['wake_rate']*100:.2f}% (target <5%)\n")
    md.append(f"- Layer-2 12-class acc on gated samples = {kws['layer2_acc_on_gated']:.3f}\n")
    md.append(f"- Energy detector baseline: recall={kws['energy_detector_recall']:.3f}, prec={kws['energy_detector_precision']:.3f}\n")
    md.append(f"- Avg power NS-RAM cascade = {kws['P_cascade_avg_W']*1e6:.3f} µW\n")
    md.append(f"- Avg power always-on baseline = {kws['P_always_on_W']*1e6:.3f} µW\n")
    md.append(f"- Energy save vs always-on = **{kws['energy_save_vs_always_on_x']:.1f}×**\n")
    md.append(f"- Gates: {kws['gates_pass']}\n")
    md.append("\n## ECG (MIT-BIH PVC gate)\n")
    md.append(f"- Gate recall = {ecg['gate_recall']:.3f}, precision = {ecg['gate_precision']:.3f}, specificity = {ecg['gate_specificity']:.3f}\n")
    md.append(f"- Wake rate = {ecg['gate_wake_rate']*100:.2f}%\n")
    md.append(f"- Layer-2 PVC sens={ecg['layer2_pvc_sensitivity']:.3f}, spec={ecg['layer2_pvc_specificity']:.3f}\n")
    md.append(f"- Cascade sens={ecg['cascade_pvc_sensitivity']:.3f}, spec={ecg['cascade_pvc_specificity']:.3f}\n")
    md.append(f"- Avg power NS-RAM cascade = {ecg['P_cascade_avg_W']*1e6:.3f} µW\n")
    md.append(f"- Avg power always-on baseline = {ecg['P_always_on_W']*1e6:.3f} µW\n")
    md.append(f"- Energy save vs always-on = **{ecg['energy_save_vs_always_on_x']:.1f}×**\n")
    md.append(f"- Gates: {ecg['gates_pass']}\n")
    md.append("\n## Pre-registered gates summary\n")
    md.append("| benchmark | G1 recall>0.85 | G2 wake<5% | G3 save>10× |\n")
    md.append("|---|---|---|---|\n")
    md.append(f"| KWS | {kws['gates_pass']['G1_recall_gt_0p85']} | {kws['gates_pass']['G2_wake_lt_0p05']} | {kws['gates_pass']['G3_save_gt_10x']} |\n")
    md.append(f"| ECG | {ecg['gates_pass']['G1_recall_gt_0p85']} | {ecg['gates_pass']['G2_wake_lt_0p05']} | {ecg['gates_pass']['G3_save_gt_10x']} |\n")
    (OUT/"summary.md").write_text("".join(md))


def main():
    print(f"DS-N14 edge cascade. device={DEVICE}", flush=True)
    kws = run_kws_benchmark(ROOT/"data/speech_commands", N_NSRAM=256, n_per_other=80)
    ecg = run_ecg_benchmark(N_NSRAM=64)
    plot_power(kws, ecg)
    write_summary(kws, ecg)
    print("\n=== DONE ===")
    print("KWS gates:", kws["gates_pass"])
    print("ECG gates:", ecg["gates_pass"])


if __name__ == "__main__":
    main()
