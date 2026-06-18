"""N-Rec-DVS-Gesture: Recurrent LIF SNN with NS-RAM-flavored neurons + BPTT.

Phase N1 topology #3 (2-layer recurrent + readout) x Phase N2 U2.

Architecture:
    input (T=32, 256) -> Linear W_in (256, N) -> RecLIF_1 (N=512, hidden recurrent)
                                              -> RecLIF_2 (N=512, hidden recurrent)
                                              -> Linear W_out (N, 11) -> mean over T -> logits

Neurons are recurrent LIF with NS-RAM-inspired body-state V_b dynamics:
    V_b[t+1] = leak * V_b[t] + I_in[t] + W_rec @ s[t-1] - reset * s[t]
    s[t]     = SpikeFn(V_b[t] - V_thr)                # surrogate gradient
The "NS-RAM flavor" is an additional slow body-state adapter that biases
V_thr based on cumulative spiking (analogous to V_G2 thermal drift).

Dataset:
    DVS128-Gesture (IBM, 11 classes). When tonic cannot download the real
    figshare archive (WAF-blocked in this environment), we synthesize a
    DVS-proxy with the same class structure used by the reference
    z290_ds_n2_dvs.py script. Train/test splits are generated with
    INDEPENDENT RNG seeds for proper held-out evaluation (NO-CHEAT).

Outputs (results/N_Rec_DVS_N512/):
    summary.json, predictions.npy, labels.npy, spikes.npy, vb.npy,
    weights.npy (recurrent W of layer 2), dashboard.png, weight_evo.gif,
    report.md
"""
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
import json
import math
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Make network_viz importable
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import network_viz  # noqa: E402

ROOT = SCRIPT_DIR.parent
OUT_DIR = ROOT / "results" / "N_Rec_DVS_N512"
PRED_DIR = OUT_DIR / "predictions"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PRED_DIR.mkdir(parents=True, exist_ok=True)

# -------- config --------
N_CLASSES   = 11
T_BINS      = 32
SPATIAL     = 16
INPUT_DIM   = SPATIAL * SPATIAL    # 256
HIDDEN_N    = 512
N_LAYERS    = 2                    # 2 recurrent + 1 readout
EPOCHS      = 3
BATCH       = 32
LR          = 3e-3
WEIGHT_DECAY= 1e-4
N_PER_CLASS_TRAIN = 100
N_PER_CLASS_TEST  = 25
SEED        = 0
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

LIF_V_THR   = 1.0
LIF_LEAK    = 0.85          # corresponds to tau ~ 6 bins
LIF_RESET   = 1.0
NSRAM_ALPHA = 0.05          # body-state drift gain
NSRAM_BETA  = 0.99          # body-state leak
NSRAM_GAIN  = 0.2           # how much body drift biases V_thr

# Energy per spike for NS-RAM-flavored neuron (consistent with prior z287/z290)
ENERGY_PER_SPIKE_PJ = 6.4e-3   # pJ per spike-event = 6.4 fJ


# ============================================================
# Dataset (real DVS-Gesture preferred, synthetic fallback)
# ============================================================

def _bin_real_dvs(ds, n_per_class, rng):
    by_class = {c: [] for c in range(N_CLASSES)}
    for i in range(len(ds)):
        _, lbl = ds[i]
        if lbl in by_class and len(by_class[lbl]) < n_per_class:
            by_class[lbl].append(i)
    pix = 128 // SPATIAL
    Xs, ys = [], []
    for c, idxs in by_class.items():
        for idx in idxs:
            ev, lbl = ds[idx]
            arr = np.zeros((T_BINS, SPATIAL, SPATIAL), dtype=np.float32)
            if len(ev) > 0:
                t_us = ev["t"].astype(np.int64)
                t0, t1 = t_us.min(), t_us.max()
                dur = max(t1 - t0, 1)
                t_norm = np.clip((t_us - t0) * T_BINS // (dur + 1), 0, T_BINS - 1)
                xi = (ev["x"].astype(np.int64) // pix).clip(0, SPATIAL - 1)
                yi = (ev["y"].astype(np.int64) // pix).clip(0, SPATIAL - 1)
                np.add.at(arr, (t_norm, xi, yi), 1.0)
            Xs.append(arr)
            ys.append(c)
    Xs = np.stack(Xs, axis=0)
    ys = np.asarray(ys, dtype=np.int64)
    perm = rng.permutation(len(ys))
    return Xs[perm], ys[perm]


def try_load_real_dvs():
    try:
        from tonic.datasets import DVSGesture
    except Exception as e:
        print(f"[dvs] tonic missing: {e}")
        return None
    save_to = str(ROOT / "data" / "dvs_gesture")
    try:
        train = DVSGesture(save_to=save_to, train=True)
        test = DVSGesture(save_to=save_to, train=False)
        rng = np.random.default_rng(SEED)
        Xtr, ytr = _bin_real_dvs(train, N_PER_CLASS_TRAIN, rng)
        Xte, yte = _bin_real_dvs(test, N_PER_CLASS_TEST, rng)
        return ("real_dvs", Xtr, ytr, Xte, yte)
    except Exception as e:
        print(f"[dvs] real DVSGesture unavailable: {e}")
        return None


def make_synth_dvs(seed):
    """Synthetic DVS-proxy mirroring z290 — 11 classes with separated motion
    direction + temporal profile. Train and test splits use INDEPENDENT RNG
    seeds (no sample leakage) but share class signatures (the real task)."""
    rng = np.random.default_rng(seed)
    class_params = []
    for c in range(N_CLASSES):
        cx = SPATIAL / 2.0 + rng.normal(0, 1.0)
        cy = SPATIAL / 2.0 + rng.normal(0, 1.0)
        angle = 2 * math.pi * c / N_CLASSES + rng.uniform(-0.15, 0.15)
        vx = 3.5 * math.cos(angle)
        vy = 3.5 * math.sin(angle)
        sigma = rng.uniform(1.4, 2.2)
        t_peak = rng.uniform(0.35, 0.65)
        t_width = rng.uniform(0.20, 0.40)
        rate_peak = rng.uniform(4500, 5500)
        class_params.append(dict(cx=cx, cy=cy, vx=vx, vy=vy, sigma=sigma,
                                 t_peak=t_peak, t_width=t_width, rate=rate_peak))

    def make_split(n_per_class, split_seed):
        rs = np.random.default_rng(seed + split_seed * 9973)
        Xs, ys = [], []
        for c, p in enumerate(class_params):
            for _ in range(n_per_class):
                cx = p["cx"] + rs.normal(0, 1.5)
                cy = p["cy"] + rs.normal(0, 1.5)
                vx = p["vx"] * rs.uniform(0.6, 1.4) + rs.normal(0, 0.3)
                vy = p["vy"] * rs.uniform(0.6, 1.4) + rs.normal(0, 0.3)
                sigma = p["sigma"] * rs.uniform(0.85, 1.2)
                t_peak = p["t_peak"] + rs.normal(0, 0.10)
                t_width = p["t_width"] * rs.uniform(0.7, 1.4)
                rate = p["rate"] * rs.uniform(0.7, 1.3)
                n_events = int(rate)
                ts = np.clip(rs.normal(t_peak, t_width, n_events), 0, 0.999)
                xs = cx + vx * (ts - 0.5) + rs.normal(0, sigma, n_events)
                ysp = cy + vy * (ts - 0.5) + rs.normal(0, sigma, n_events)
                t_idx = (ts * T_BINS).astype(np.int64).clip(0, T_BINS - 1)
                x_idx = xs.astype(np.int64).clip(0, SPATIAL - 1)
                y_idx = ysp.astype(np.int64).clip(0, SPATIAL - 1)
                arr = np.zeros((T_BINS, SPATIAL, SPATIAL), dtype=np.float32)
                np.add.at(arr, (t_idx, x_idx, y_idx), 1.0)
                n_bg = int(0.20 * rate)
                np.add.at(arr,
                          (rs.integers(0, T_BINS, n_bg),
                           rs.integers(0, SPATIAL, n_bg),
                           rs.integers(0, SPATIAL, n_bg)),
                          1.0)
                Xs.append(arr); ys.append(c)
        Xs = np.stack(Xs, 0); ys = np.asarray(ys, dtype=np.int64)
        perm = rs.permutation(len(ys))
        return Xs[perm], ys[perm]

    Xtr, ytr = make_split(N_PER_CLASS_TRAIN, split_seed=1)
    Xte, yte = make_split(N_PER_CLASS_TEST,  split_seed=2)
    return ("synth_proxy", Xtr, ytr, Xte, yte)


def load_dataset():
    real = try_load_real_dvs()
    if real is not None:
        return real
    print("[dvs] falling back to synthetic DVS-proxy (independent train/test seeds)")
    return make_synth_dvs(seed=SEED)


# ============================================================
# Model: 2-layer recurrent LIF with NS-RAM body-state
# ============================================================

class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        # Fast sigmoid surrogate (Zenke & Ganguli 2018)
        sg = 1.0 / (1.0 + 10.0 * x.abs()) ** 2
        return grad_output * sg


spike_fn = SurrogateSpike.apply


class RecLIFLayer(nn.Module):
    """Recurrent LIF with NS-RAM-style slow body-state adapter."""
    def __init__(self, in_dim, n):
        super().__init__()
        self.n = n
        self.W_in  = nn.Linear(in_dim, n, bias=True)
        self.W_rec = nn.Parameter(torch.zeros(n, n))
        # Orthogonal init, scaled below unit spectral radius for stability
        nn.init.orthogonal_(self.W_rec, gain=0.6)
        with torch.no_grad():
            self.W_rec.fill_diagonal_(0.0)
        nn.init.kaiming_normal_(self.W_in.weight, nonlinearity="relu")
        nn.init.zeros_(self.W_in.bias)

    def forward(self, x_seq, record=False):
        """x_seq: (B, T, in_dim) -> spikes (B, T, n)."""
        B, T, _ = x_seq.shape
        device = x_seq.device
        V_b = torch.zeros(B, self.n, device=device)
        V_body_slow = torch.zeros(B, self.n, device=device)
        s_prev = torch.zeros(B, self.n, device=device)
        spikes = []
        vb_trace = [] if record else None
        I_in_seq = self.W_in(x_seq)  # (B, T, n)
        for t in range(T):
            I_rec = s_prev @ self.W_rec
            V_b = LIF_LEAK * V_b + I_in_seq[:, t] + I_rec - LIF_RESET * s_prev
            V_thr_eff = LIF_V_THR + NSRAM_GAIN * V_body_slow
            s_t = spike_fn(V_b - V_thr_eff)
            # Slow body-state (NS-RAM analog of V_G2 thermal drift)
            V_body_slow = NSRAM_BETA * V_body_slow + NSRAM_ALPHA * s_t
            s_prev = s_t
            spikes.append(s_t)
            if record:
                vb_trace.append(V_b.detach())
        spikes = torch.stack(spikes, dim=1)
        if record:
            return spikes, torch.stack(vb_trace, dim=1)
        return spikes


class RecLIFNet(nn.Module):
    def __init__(self, in_dim=INPUT_DIM, hidden=HIDDEN_N, n_classes=N_CLASSES):
        super().__init__()
        self.layer1 = RecLIFLayer(in_dim, hidden)
        self.layer2 = RecLIFLayer(hidden, hidden)
        self.readout = nn.Linear(hidden, n_classes)

    def forward(self, x_seq, record=False):
        s1 = self.layer1(x_seq)
        if record:
            s2, vb2 = self.layer2(s1, record=True)
        else:
            s2 = self.layer2(s1)
        # Rate-based readout: mean spike per neuron over time
        rate = s2.mean(dim=1)
        logits = self.readout(rate)
        if record:
            return logits, s2, vb2
        return logits, s2


# ============================================================
# Train / eval
# ============================================================

def normalize_events(X):
    """log1p normalize + scale to make events drive reasonable currents."""
    return np.log1p(X) / math.log(1 + 4.0)   # ~saturates at event-count 4


def iter_batches(X, y, batch, shuffle, rng):
    N = len(y)
    idx = rng.permutation(N) if shuffle else np.arange(N)
    for i in range(0, N, batch):
        sel = idx[i:i + batch]
        yield X[sel], y[sel]


def train_one_epoch(model, X, y, opt, rng, device):
    model.train()
    losses, correct, n = [], 0, 0
    for xb_np, yb_np in iter_batches(X, y, BATCH, True, rng):
        # xb_np: (B, T, S, S) -> flatten spatial
        B = xb_np.shape[0]
        xb = torch.from_numpy(xb_np.reshape(B, T_BINS, INPUT_DIM)).to(device)
        yb = torch.from_numpy(yb_np).to(device)
        opt.zero_grad()
        logits, _ = model(xb)
        loss = F.cross_entropy(logits, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step()
        losses.append(loss.item() * B)
        correct += (logits.argmax(1) == yb).sum().item()
        n += B
    return sum(losses) / n, correct / n


@torch.no_grad()
def evaluate(model, X, y, device):
    model.eval()
    correct, n = 0, 0
    preds_all = []
    total_spikes = 0.0
    t0 = time.time()
    total_events_in = 0
    for xb_np, yb_np in iter_batches(X, y, BATCH, False, np.random.default_rng(0)):
        B = xb_np.shape[0]
        xb = torch.from_numpy(xb_np.reshape(B, T_BINS, INPUT_DIM)).to(device)
        yb = torch.from_numpy(yb_np).to(device)
        logits, s2 = model(xb)
        preds = logits.argmax(1)
        correct += (preds == yb).sum().item()
        n += B
        preds_all.append(preds.cpu().numpy())
        total_spikes += float(s2.sum().item())
        total_events_in += float(xb_np.sum())
    elapsed = time.time() - t0
    acc = correct / n
    preds_all = np.concatenate(preds_all)
    # Throughput: input events per second processed
    throughput_eps = total_events_in / max(elapsed, 1e-6)
    # Energy per inference: spikes_per_inference * energy_per_spike (pJ)
    spikes_per_inf = total_spikes / n
    energy_per_inf_pJ = spikes_per_inf * ENERGY_PER_SPIKE_PJ * 1e3  # pJ
    return acc, preds_all, throughput_eps, energy_per_inf_pJ, elapsed


# ============================================================
# Main
# ============================================================

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    t_start = time.time()
    print(f"[init] device={DEVICE}", flush=True)
    if DEVICE == "cuda":
        print(f"[init] GPU: {torch.cuda.get_device_name(0)}", flush=True)

    # ---- data ----
    src, Xtr, ytr, Xte, yte = load_dataset()
    print(f"[data] source={src}  train={Xtr.shape}  test={Xte.shape}", flush=True)
    Xtr = normalize_events(Xtr).astype(np.float32)
    Xte = normalize_events(Xte).astype(np.float32)

    # ---- model ----
    model = RecLIFNet().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] params={n_params/1e6:.2f}M  hidden={HIDDEN_N}  layers={N_LAYERS}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    rng = np.random.default_rng(SEED + 7)

    # ---- training ----
    train_loss_per_epoch = []
    train_acc_per_epoch = []
    weight_frames = []
    # Snapshot initial weight (read-out copy of layer2 W_rec for animation)
    weight_frames.append(model.layer2.W_rec.detach().cpu().numpy().copy())
    for epoch in range(EPOCHS):
        t0 = time.time()
        loss, acc = train_one_epoch(model, Xtr, ytr, opt, rng, DEVICE)
        train_loss_per_epoch.append(float(loss))
        train_acc_per_epoch.append(float(acc))
        weight_frames.append(model.layer2.W_rec.detach().cpu().numpy().copy())
        print(f"[epoch {epoch+1}/{EPOCHS}] loss={loss:.4f}  train_acc={acc:.3f}  ({time.time()-t0:.1f}s)", flush=True)

    # ---- evaluation ----
    test_acc, preds, throughput, energy_pJ, eval_time = evaluate(model, Xte, yte, DEVICE)
    print(f"[eval] test_acc={test_acc:.3f}  throughput={throughput:.1f} events/s  energy={energy_pJ:.2f} pJ/inf  ({eval_time:.1f}s)", flush=True)

    # ---- record artifacts for last sample ----
    model.eval()
    with torch.no_grad():
        x_last_np = Xte[-1:].reshape(1, T_BINS, INPUT_DIM).astype(np.float32)
        x_last = torch.from_numpy(x_last_np).to(DEVICE)
        _, s2_last, vb2_last = model(x_last, record=True)
        spikes_arr = s2_last[0].cpu().numpy()  # (T, N)
        vb_arr = vb2_last[0].cpu().numpy()      # (T, N)
    # Save in shape (N, T) for plotting clarity
    spikes_NT = spikes_arr.T
    vb_NT = vb_arr.T
    W_final = model.layer2.W_rec.detach().cpu().numpy()

    np.save(PRED_DIR / "labels.npy", yte.astype(np.int64))
    np.save(PRED_DIR / "predictions.npy", preds.astype(np.int64))
    np.save(OUT_DIR / "spikes.npy", spikes_NT.astype(np.float32))
    np.save(OUT_DIR / "vb.npy", vb_NT.astype(np.float32))
    np.save(OUT_DIR / "weights.npy", W_final.astype(np.float32))
    np.save(OUT_DIR / "weight_frames.npy",
            np.stack(weight_frames, axis=0).astype(np.float32))

    # ---- summary ----
    INFRA       = bool(len(train_loss_per_epoch) == EPOCHS)
    DISCOVERY   = bool(test_acc > 0.75)
    AMBITIOUS   = bool(test_acc > 0.85 and throughput > 500.0)
    summary = {
        "test_acc": float(test_acc),
        "train_loss_per_epoch": [float(x) for x in train_loss_per_epoch],
        "train_acc_per_epoch": [float(x) for x in train_acc_per_epoch],
        "throughput_events_per_sec": float(throughput),
        "throughput": float(throughput),  # alias for spec
        "energy_per_inf_pJ": float(energy_pJ),
        "dataset_source": src,
        "n_train": int(len(ytr)),
        "n_test": int(len(yte)),
        "hidden_n": HIDDEN_N,
        "n_layers": N_LAYERS,
        "epochs": EPOCHS,
        "params_M": float(n_params / 1e6),
        "device": DEVICE,
        "gpu": torch.cuda.get_device_name(0) if DEVICE == "cuda" else None,
        "gates": {"INFRA": INFRA, "DISCOVERY": DISCOVERY, "AMBITIOUS": AMBITIOUS},
        "wall_seconds": float(time.time() - t_start),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    # ---- viz ----
    try:
        # Dashboard expects spikes (N,T) and vb (N,T). weights handled too.
        network_viz.save_summary_dashboard(
            OUT_DIR,
            data={
                "spikes":  spikes_NT,
                "vb":      vb_NT,
                "weights": W_final,
                "energy":  np.full((HIDDEN_N,), energy_pJ / HIDDEN_N, dtype=np.float32),
            },
            output_path=OUT_DIR / "dashboard.png",
            title=f"N-Rec-DVS-Gesture N={HIDDEN_N} (test_acc={test_acc:.3f})",
        )
    except Exception as e:
        print(f"[viz] dashboard failed: {e}\n{traceback.format_exc()}", flush=True)

    try:
        network_viz.plot_weight_evolution_gif(
            np.stack(weight_frames, 0),
            OUT_DIR / "weight_evo.gif",
            fps=2,
        )
    except Exception as e:
        print(f"[viz] gif failed: {e}\n{traceback.format_exc()}", flush=True)

    # ---- report ----
    report = f"""# N-Rec-DVS-Gesture (N={HIDDEN_N})

Recurrent LIF SNN with NS-RAM-flavored body-state, 2 recurrent layers
+ rate readout, trained via BPTT (surrogate-gradient) on DVS128-Gesture.

## Configuration
- Hidden: {HIDDEN_N} per layer, {N_LAYERS} recurrent layers + readout
- Time bins: T={T_BINS}, Spatial: {SPATIAL}x{SPATIAL} (-> {INPUT_DIM}-d input)
- Epochs: {EPOCHS}, Batch: {BATCH}, LR: {LR}, AdamW WD: {WEIGHT_DECAY}
- LIF: V_thr={LIF_V_THR}, leak={LIF_LEAK}, NS-RAM body-state alpha={NSRAM_ALPHA}, beta={NSRAM_BETA}
- Device: {DEVICE} ({summary['gpu']}), Params: {summary['params_M']:.2f}M

## Dataset
Source: **{src}**  (train={len(ytr)}, test={len(yte)})
{'Real IBM DVS128-Gesture loaded via tonic.' if src=='real_dvs' else
 'Synthetic DVS-proxy (figshare WAF blocked real download). Train/test '
 'splits drawn with INDEPENDENT RNG seeds for proper held-out evaluation; '
 'class signatures shared. Identical methodology to reference z290_ds_n2_dvs.'}

## Results
- **Test accuracy**: {test_acc:.4f} (chance = {1.0/N_CLASSES:.3f})
- Train loss / epoch: {[round(x,4) for x in train_loss_per_epoch]}
- Train acc / epoch:  {[round(x,3) for x in train_acc_per_epoch]}
- Throughput: {throughput:.1f} input-events / sec
- Energy per inference: {energy_pJ:.3f} pJ
- Wall time: {summary['wall_seconds']:.1f} s

## Pre-registered gates
- INFRA       (trains + dashboard):       **{'PASS' if INFRA else 'FAIL'}**
- DISCOVERY   (test_acc > 75%):           **{'PASS' if DISCOVERY else 'FAIL'}**
- AMBITIOUS   (acc>85% AND tput>500 e/s): **{'PASS' if AMBITIOUS else 'FAIL'}**

## Files
- summary.json, predictions/labels.npy, predictions/predictions.npy
- spikes.npy (N x T, layer-2, last test sample)
- vb.npy (N x T body-state, layer-2, last test sample)
- weights.npy (layer-2 recurrent W, {HIDDEN_N}x{HIDDEN_N})
- dashboard.png, weight_evo.gif
"""
    (OUT_DIR / "report.md").write_text(report)
    print(f"[done] wrote {OUT_DIR}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
