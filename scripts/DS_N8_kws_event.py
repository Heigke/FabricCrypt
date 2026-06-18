"""DS-N8 KWS — Event-coded NS-RAM on Google Speech Commands.

Retry of DS-N1 (z287, chance ~8%) with TRUE event-coded input
(Poisson ΔV_G2 pulse trains driven by MFCC) and N scaled to 10K, 100K
cells via GPU-batched transient framework (per-utterance Vb tensor).

Architecture per utterance (1s, 16kHz):
  1) MFCC 40 x 99 frames, normalize per-feature to [0,1] over train.
  2) Sparse projection W_in (K_FAN=4 active bins per cell, signed random
     weights) maps each frame's 40-dim MFCC -> N-dim drive intensity.
  3) Each frame is split into K_SUB substeps. At each substep, draw
     Poisson spikes per cell with prob = sigmoid(W_in @ mfcc[frame]) *
     P_MAX. A spike sets that cell's VG2 := VG2_BIAS + dVG2 for that
     substep AND injects drive=1 into the cell ODE.
  4) cell.step is run with batched state (B, N) — Vb has a batch dim, all
     other cell params shared across batch.
  5) Time-pool into N_CHUNK chunks; for each chunk compute mean Id and
     mean Vb. Final Vb is appended. Feature dim = (2*N_CHUNK + 1) * N.
  6) Ridge readout (one-hot regression) for 12 classes.

10 keywords + _unknown_ + _silence_; 8000 train + 1000 test.
Scales: N in {10_000, 100_000}, Seeds: 5.
Gates: INFRA>=0.20, PASS>=0.50, AMBITIOUS>=0.80.
"""
from __future__ import annotations
import os, sys, json, time, argparse, math
from pathlib import Path
import numpy as np
import torch
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ---- pre-registered config ---- #
SR        = 16000
N_MFCC    = 40
N_FRAMES  = 99
WIN_MS    = 25.0
HOP_MS    = 10.0
KEYWORDS  = ["yes","no","up","down","left","right","on","off","stop","go"]
CLASSES   = KEYWORDS + ["_unknown_", "_silence_"]
N_CLASSES = 12
LABEL2IDX = {c: i for i, c in enumerate(CLASSES)}

# event encoding
K_SUB         = 3
K_FAN         = 4         # nonzero connections per cell
P_MAX         = 0.5
DVG2_PULSE    = 0.10
VG2_BIAS      = 0.20
VG1_BIAS_LO   = 0.45
VG1_BIAS_HI   = 0.65
N_CHUNK       = 4
E_PER_EVENT_J = 6.4e-15

# ODE constants (matches cell_fast defaults)
VTH0    = 0.40
K_BACK  = 0.5
A_IDD   = 5.0
G_BJT   = 1.0
V_BJT_ON= 0.75
V_LATCH = 0.55
K_LEAK  = 0.02
DT      = 0.05
ALPHA_LO = 0.30
ALPHA_HI = 0.70

# ---- MFCC (numpy only) ---- #
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

# ---- dataset ---- #
def scan_dataset(root: Path):
    val_set, test_set = set(), set()
    for fn, S in [("validation_list.txt", val_set), ("testing_list.txt", test_set)]:
        p = root / fn
        if p.exists(): S.update(l.strip() for l in p.read_text().splitlines() if l.strip())
    splits = {"train": {}, "test": {}, "val": {}}
    bg_dir = root / "_background_noise_"
    bg = sorted(str(p) for p in bg_dir.glob("*.wav")) if bg_dir.exists() else []
    for d in sorted(root.iterdir()):
        if not d.is_dir() or d.name.startswith("_"): continue
        name = d.name
        for wav in sorted(d.glob("*.wav")):
            rel = f"{name}/{wav.name}"
            if   rel in test_set: split = "test"
            elif rel in val_set:  split = "val"
            else:                 split = "train"
            label = name if name in KEYWORDS else "_unknown_"
            splits[split].setdefault(label, []).append(str(wav))
    return splits, bg

def synth_silence(bg_files, n, rng):
    if not bg_files: return [np.zeros(SR, dtype=np.float32) for _ in range(n)]
    cache, out = {}, []
    for _ in range(n):
        f = rng.choice(bg_files)
        if f not in cache:
            w, _ = sf.read(f, dtype="float32")
            if w.ndim > 1: w = w.mean(axis=1)
            cache[f] = w
        w = cache[f]
        if w.shape[0] > SR:
            off = rng.integers(0, w.shape[0] - SR)
            clip = w[off:off+SR].astype(np.float32)
        else:
            clip = np.pad(w, (0, max(0, SR - w.shape[0]))).astype(np.float32)
        out.append(clip * rng.uniform(0.0, 1.0))
    return out

def build_split(splits, bg, split_name, n_per_class, rng):
    items, labels = [], []
    sp = splits[split_name]
    for label in CLASSES:
        if label == "_silence_":
            for c in synth_silence(bg, n_per_class, rng):
                items.append(("arr", c)); labels.append(LABEL2IDX[label])
        elif label == "_unknown_":
            pool = sp.get("_unknown_", [])
            if not pool: continue
            idx = rng.choice(len(pool), size=min(n_per_class, len(pool)), replace=False)
            for i in idx: items.append(("path", pool[i])); labels.append(LABEL2IDX[label])
        else:
            pool = sp.get(label, [])
            if not pool: continue
            idx = rng.choice(len(pool), size=min(n_per_class, len(pool)), replace=False)
            for i in idx: items.append(("path", pool[i])); labels.append(LABEL2IDX[label])
    return items, np.array(labels, dtype=np.int64)

def items_to_mfcc(items, desc=""):
    X = np.zeros((len(items), N_FRAMES, N_MFCC), dtype=np.float32)
    t0 = time.time()
    for i, (k, p) in enumerate(items):
        if k == "path":
            w, _ = sf.read(p, dtype="float32")
            if w.ndim > 1: w = w.mean(axis=1)
        else:
            w = p
        X[i] = wav_to_mfcc(w)
        if (i+1) % 2000 == 0:
            print(f"    {desc} mfcc {i+1}/{len(items)} ({time.time()-t0:.1f}s)", flush=True)
    return X

# ---- NS-RAM event-coded reservoir, fully batched (B, N) ---- #
class BatchedCellReservoir:
    """Batched CellArray with per-utterance Vb (B, N).

    All cell parameters are (N,) and broadcast across batch. Drive,
    spikes, and per-substep VG2 modulation are (B, N).
    """
    def __init__(self, N: int, device: str, dtype: torch.dtype, seed: int):
        self.N = N
        self.device = device
        self.dtype = dtype
        g = torch.Generator(device=device).manual_seed(seed)
        # heterogeneous alpha for richer dynamics
        self.alpha    = ALPHA_LO + (ALPHA_HI-ALPHA_LO) * torch.rand(N, generator=g, device=device, dtype=dtype)
        # heterogeneous front-gate bias
        self.VG1_bias = VG1_BIAS_LO + (VG1_BIAS_HI-VG1_BIAS_LO) * torch.rand(N, generator=g, device=device, dtype=dtype)
        # sparse signed W_in: each cell connects to K_FAN MFCC bins
        bins   = torch.randint(0, N_MFCC, (N, K_FAN), generator=g, device=device)
        signs  = (torch.randint(0, 2, (N, K_FAN), generator=g, device=device, dtype=dtype) * 2 - 1)
        self.W_in_idx = bins              # (N, K_FAN)
        self.W_in_sgn = signs             # (N, K_FAN)
        # scalar shared parameters
        self.gen = g

    def project(self, mfcc_frame: torch.Tensor) -> torch.Tensor:
        """mfcc_frame: (B, N_MFCC) in [0,1] -> drive intensity (B, N) in [0,1]."""
        # gather: (B, N, K_FAN)
        B = mfcc_frame.shape[0]
        idx = self.W_in_idx.unsqueeze(0).expand(B, -1, -1)
        g = torch.gather(mfcc_frame.unsqueeze(2).expand(-1, -1, K_FAN),
                         1, idx)
        # weighted sum (signed) per cell, normalized by K_FAN, then sigmoid
        s = (g * self.W_in_sgn.unsqueeze(0)).sum(dim=2) / K_FAN
        # map to [0,1]
        return torch.sigmoid(4.0 * s)

    def run(self, mfcc_batch: torch.Tensor, count_events: bool):
        """mfcc_batch: (B, T, N_MFCC) normalized in [0,1].

        Returns features (B, (2*N_CHUNK+1)*N), event counts (B,).
        """
        B, T, _ = mfcc_batch.shape
        N = self.N
        device = self.device
        dtype = self.dtype
        Vb = torch.zeros(B, N, device=device, dtype=dtype) + VG2_BIAS  # init to VG2_bias
        Vth_base = VTH0 - K_BACK * VG2_BIAS  # (scalar)
        # Vth shift from pulse: -K_BACK * dVG2 when pulse ON
        chunk_len = max(T // N_CHUNK, 1)
        id_chunks = torch.zeros(N_CHUNK, B, N, device=device, dtype=dtype)
        vb_chunks = torch.zeros(N_CHUNK, B, N, device=device, dtype=dtype)
        cnt = [0] * N_CHUNK
        events = torch.zeros(B, device=device, dtype=dtype) if count_events else None
        VG1 = self.VG1_bias.unsqueeze(0).expand(B, -1)   # (B, N) shared param

        for t in range(T):
            ck = min(t // chunk_len, N_CHUNK - 1)
            rate = self.project(mfcc_batch[:, t, :])     # (B, N) in [0,1]
            for _s in range(K_SUB):
                # Poisson spike per (B, N)
                u = torch.rand(B, N, generator=self.gen, device=device, dtype=dtype)
                spike = (u < (P_MAX * rate)).to(dtype)
                # VG2 pulse on spiking cells -> Vth_eff lowered
                Vth_eff = VTH0 - K_BACK * (VG2_BIAS + DVG2_PULSE * spike)
                overdrive = torch.clamp(VG1 - Vth_eff, min=0.0)
                channel_on = torch.sigmoid((VG1 - Vth_eff) / 0.05)
                Iii  = A_IDD * spike * channel_on * overdrive          # drive=spike
                Ibjt = G_BJT * torch.clamp(Vb - V_BJT_ON, min=0.0)
                Ileak= K_LEAK * (Vb - (VG2_BIAS + DVG2_PULSE * spike))
                dVb  = self.alpha * Iii - Ibjt - Ileak
                Vb   = torch.clamp(Vb + DT * dVb, -0.2, 0.85)
                # Read Id
                latched = torch.clamp((Vb - V_LATCH) / (0.85 - V_LATCH), min=0.0)
                Id = 1e-9 * channel_on * (1.0 + 100.0 * latched)
                id_chunks[ck] = id_chunks[ck] + Id
                vb_chunks[ck] = vb_chunks[ck] + Vb
                cnt[ck] += 1
                if count_events:
                    events = events + spike.sum(dim=1)
        # normalize chunks
        parts = []
        for k in range(N_CHUNK):
            c = max(cnt[k], 1)
            # log-Id is more useful (spans many decades)
            parts.append(torch.log10(id_chunks[k] / c + 1e-18))
            parts.append(vb_chunks[k] / c)
        parts.append(Vb)
        feats = torch.cat(parts, dim=1)  # (B, (2*N_CHUNK+1) * N)
        return feats, events

def ridge_fit(X, y, alpha=1.0, n_classes=N_CLASSES):
    Y = torch.nn.functional.one_hot(y, n_classes).to(X.dtype)
    mu = X.mean(dim=0, keepdim=True)
    sd = X.std(dim=0, keepdim=True) + 1e-9
    Xn = (X - mu) / sd
    d = Xn.shape[1]
    A = Xn.T @ Xn + alpha * torch.eye(d, device=X.device, dtype=X.dtype)
    B = Xn.T @ Y
    W = torch.linalg.solve(A, B)
    return W, mu, sd

def ridge_predict(X, W, mu, sd):
    Xn = (X - mu) / sd
    return (Xn @ W).argmax(dim=1)

# ---- run one config ---- #
def featurize_in_minibatches(res: BatchedCellReservoir, X_t: torch.Tensor,
                              mb: int, count_events: bool, label: str):
    B = X_t.shape[0]
    feats = []
    ev_tot = []
    t0 = time.time()
    for i in range(0, B, mb):
        f, e = res.run(X_t[i:i+mb], count_events=count_events)
        feats.append(f)
        if count_events:
            ev_tot.append(e)
        if (i // mb) % 5 == 0:
            print(f"      {label} feat {i}/{B} t={time.time()-t0:.1f}s", flush=True)
    F = torch.cat(feats, dim=0)
    E = torch.cat(ev_tot, dim=0) if count_events else None
    return F, E

def run_one(N: int, seed: int, Xtr_n: np.ndarray, ytr: np.ndarray,
            Xte_n: np.ndarray, yte: np.ndarray, device: str, dtype):
    print(f"  [seed={seed} N={N}] start", flush=True)
    t0 = time.time()
    res = BatchedCellReservoir(N, device=device, dtype=dtype, seed=seed*7919 + 11)
    Xtr_t = torch.from_numpy(Xtr_n).to(device=device, dtype=dtype)
    Xte_t = torch.from_numpy(Xte_n).to(device=device, dtype=dtype)
    ytr_t = torch.from_numpy(ytr).to(device)
    yte_t = torch.from_numpy(yte).to(device)
    # mini-batch sized to memory: feature dim is (2*N_CHUNK+1)*N -> ~9N float32 per sample
    # For N=10K: 360K feats * 4B = 1.4MB per sample. With B=128 that's 184MB feat block, fine.
    # For N=100K: 3.6M feats * 4B = 14.4MB per sample. B=32 -> 460MB. Fine for GB10.
    mb = 128 if N <= 10_000 else (32 if N <= 100_000 else 16)

    feats_tr, _      = featurize_in_minibatches(res, Xtr_t, mb, False, "tr")
    feats_te, ev_te  = featurize_in_minibatches(res, Xte_t, mb, True,  "te")
    t_feat = time.time() - t0
    # ridge -> may be heavy at d=900K; use mini-batch normal-equations trick.
    print(f"    [seed={seed} N={N}] fitting ridge  d={feats_tr.shape[1]}", flush=True)
    t_r = time.time()
    # If d is huge, fit on a random subset of features.
    d = feats_tr.shape[1]
    if d > 80_000:
        # subsample features
        gsub = torch.Generator(device=device).manual_seed(seed + 99991)
        sel = torch.randperm(d, generator=gsub, device=device)[:80_000]
        Ftr = feats_tr[:, sel]
        Fte = feats_te[:, sel]
    else:
        Ftr, Fte = feats_tr, feats_te
    W, mu, sd = ridge_fit(Ftr, ytr_t, alpha=1.0)
    pred = ridge_predict(Fte, W, mu, sd)
    acc = (pred == yte_t).float().mean().item()
    wall = time.time() - t0
    mean_events = float(ev_te.mean().item())
    body_events_per_inf = N_FRAMES * K_SUB * N
    energy_J = (mean_events + body_events_per_inf) * E_PER_EVENT_J
    print(f"  [seed={seed} N={N}] acc={acc:.4f}  feat_wall={t_feat:.1f}s  "
          f"ridge_wall={time.time()-t_r:.1f}s  total={wall:.1f}s  "
          f"events={mean_events:.0f}  energy={energy_J*1e9:.2f}nJ/inf", flush=True)
    return {
        "N": N, "seed": seed, "acc": acc,
        "pred_test": pred.cpu().numpy().tolist(),
        "yte":       yte_t.cpu().numpy().tolist(),
        "wall_s": wall, "feat_wall_s": t_feat,
        "mean_input_events": mean_events,
        "body_events_per_inf": body_events_per_inf,
        "energy_per_inf_J":  energy_J,
        "energy_per_inf_nJ": energy_J * 1e9,
        "feat_dim_used": int(Ftr.shape[1]),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=str(ROOT / "data/speech_commands"))
    ap.add_argument("--out_dir",   default=str(ROOT / "results/DS_N8_kws_event"))
    ap.add_argument("--n_train_per_class", type=int, default=800)
    ap.add_argument("--n_test_per_class",  type=int, default=100)
    ap.add_argument("--Ns",    type=str, default="10000,100000")
    ap.add_argument("--seeds", type=str, default="0,1,2,3,4")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype",  default="float32")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"
    dtype  = torch.float32 if args.dtype == "float32" else torch.float64
    print(f"device={device} dtype={dtype}", flush=True)
    Ns    = [int(x) for x in args.Ns.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]

    cache = out / "mfcc_cache.npz"
    if cache.exists():
        z = np.load(cache)
        Xtr = z["Xtr"]; ytr = z["ytr"]; Xte = z["Xte"]; yte = z["yte"]
        print(f"loaded MFCC cache: tr={Xtr.shape} te={Xte.shape}", flush=True)
    else:
        rng = np.random.default_rng(0xDEFA)
        splits, bg = scan_dataset(Path(args.data_root))
        print(f"  classes/train: { {k: len(v) for k,v in splits['train'].items()} }", flush=True)
        tr_items, ytr = build_split(splits, bg, "train", args.n_train_per_class, rng)
        te_items, yte = build_split(splits, bg, "test",  args.n_test_per_class,  rng)
        print(f"  train={len(tr_items)} test={len(te_items)}", flush=True)
        Xtr = items_to_mfcc(tr_items, "train")
        Xte = items_to_mfcc(te_items, "test")
        np.savez_compressed(cache, Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte)

    # Normalize per-feature using train.
    flat_tr = Xtr.reshape(Xtr.shape[0], -1)
    flat_te = Xte.reshape(Xte.shape[0], -1)
    mn = flat_tr.min(axis=0, keepdims=True)
    mx = flat_tr.max(axis=0, keepdims=True)
    rng_d = (mx - mn) + 1e-6
    Xtr_n = np.clip((flat_tr - mn) / rng_d, 0.0, 1.0).astype(np.float32).reshape(-1, N_FRAMES, N_MFCC)
    Xte_n = np.clip((flat_te - mn) / rng_d, 0.0, 1.0).astype(np.float32).reshape(-1, N_FRAMES, N_MFCC)

    results = []
    for N in Ns:
        for s in seeds:
            r = run_one(N, s, Xtr_n, ytr, Xte_n, yte, device, dtype)
            results.append(r)
            with open(out / "accuracy_vs_N_seeds.json", "w") as fp:
                json.dump({"results": results}, fp, indent=2)
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

    # summary
    summary = {}
    for N in Ns:
        accs = [r["acc"] for r in results if r["N"] == N]
        walls = [r["wall_s"] for r in results if r["N"] == N]
        ens = [r["energy_per_inf_nJ"] for r in results if r["N"] == N]
        summary[str(N)] = {
            "mean_acc": float(np.mean(accs)),
            "std_acc":  float(np.std(accs)),
            "min_acc":  float(np.min(accs)),
            "max_acc":  float(np.max(accs)),
            "mean_wall_s": float(np.mean(walls)),
            "mean_energy_nJ": float(np.mean(ens)),
            "n_seeds": len(accs),
        }
    out_summary = {
        "config": dict(K_SUB=K_SUB, K_FAN=K_FAN, P_MAX=P_MAX,
                       DVG2_PULSE=DVG2_PULSE, VG2_BIAS=VG2_BIAS,
                       VG1_BIAS_LO=VG1_BIAS_LO, VG1_BIAS_HI=VG1_BIAS_HI,
                       N_CHUNK=N_CHUNK,
                       n_train_per_class=args.n_train_per_class,
                       n_test_per_class=args.n_test_per_class,
                       n_classes=N_CLASSES, K_SUB_per_frame=K_SUB),
        "summary": summary,
    }
    with open(out / "accuracy_vs_N_seeds.json", "w") as fp:
        json.dump({"results": results, "summary": out_summary}, fp, indent=2)

    # confusion matrix
    Nstar = max(Ns)
    best = max((r for r in results if r["N"] == Nstar), key=lambda r: r["acc"])
    try:
        from sklearn.metrics import confusion_matrix
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        cm = confusion_matrix(best["yte"], best["pred_test"], labels=list(range(N_CLASSES)))
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(cm, cmap="viridis")
        ax.set_xticks(range(N_CLASSES)); ax.set_yticks(range(N_CLASSES))
        ax.set_xticklabels(CLASSES, rotation=45, ha="right")
        ax.set_yticklabels(CLASSES)
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        ax.set_title(f"DS-N8 Event NS-RAM KWS  N={Nstar} seed={best['seed']} "
                     f"acc={best['acc']:.3f}")
        for i in range(N_CLASSES):
            for j in range(N_CLASSES):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="w" if cm[i, j] < cm.max()/2 else "k", fontsize=7)
        fig.colorbar(im); fig.tight_layout()
        fig.savefig(out / "confusion_matrix.png", dpi=120)
        plt.close(fig)
    except Exception as e:
        print(f"  confusion matrix plot skipped: {e}", flush=True)

    md = ["# DS-N8 KWS Event-Coded NS-RAM Results", ""]
    md.append(f"- Device: {device}")
    md.append(f"- Classes: {N_CLASSES} ({', '.join(CLASSES)})")
    md.append(f"- Train per class: {args.n_train_per_class}; Test per class: "
              f"{args.n_test_per_class}")
    md.append(f"- Encoding: event-coded sparse ΔV_G2 pulses; K_FAN={K_FAN} "
              f"bins/cell; K_SUB={K_SUB} substeps/frame; P_MAX={P_MAX}; "
              f"dVG2={DVG2_PULSE}; N_CHUNK={N_CHUNK}")
    md.append("")
    md.append("## Accuracy")
    for N in Ns:
        s = summary[str(N)]
        md.append(f"- N={N:>7d}: acc = {s['mean_acc']:.4f} ± {s['std_acc']:.4f} "
                  f"(min={s['min_acc']:.4f} max={s['max_acc']:.4f}, "
                  f"n_seeds={s['n_seeds']}, mean_wall={s['mean_wall_s']:.1f}s, "
                  f"energy={s['mean_energy_nJ']:.2f} nJ/inf)")
    md.append("")
    md.append("## Gates")
    Nstar_s = summary[str(Nstar)]
    md.append(f"- INFRA (≥0.20):      {'PASS' if Nstar_s['mean_acc'] >= 0.20 else 'FAIL'}")
    md.append(f"- PASS (≥0.50):       {'PASS' if Nstar_s['mean_acc'] >= 0.50 else 'FAIL'}")
    md.append(f"- AMBITIOUS (≥0.80):  {'PASS' if Nstar_s['mean_acc'] >= 0.80 else 'FAIL'}")
    (out / "summary.md").write_text("\n".join(md))
    print("\n".join(md))
    print(f"\nALL DONE  out={out}", flush=True)

if __name__ == "__main__":
    main()
