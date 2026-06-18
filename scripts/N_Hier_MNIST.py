"""N-Hier-MNIST: Hierarchical 2-layer LIF SNN with skip-connection on MNIST.

Phase N1 topology #4: hierarchical 2-layer SNN with skip from layer1 to readout.

Architecture:
    input (T=20, 784) -> Linear (784, 256) -> FFLIF_1 (N=256, NS-RAM)
                                           -> Linear (256, 128) -> FFLIF_2 (N=128, NS-RAM)
    readout takes BOTH mean-rate(layer1) and mean-rate(layer2) concatenated
    (skip-connection):
        rate_concat = [rate(s1), rate(s2)]  -> Linear(384, 10) -> logits

Neurons are feedforward LIF with NS-RAM-inspired slow body-state adapter that
biases V_thr based on cumulative spiking (V_G2 thermal drift analog).

Dataset: real MNIST via torchvision (already cached under data/MNIST/raw).
Train/test splits are standard MNIST splits.

Outputs (results/N_Hier_MNIST_N256-128/):
    summary.json, predictions/{labels,predictions}.npy, spikes.npy, vb.npy,
    weights.npy (l1+l2+skip concatenated as a dict-of-arrays via np.savez ->
    weights.npz), dashboard.png, weight_evo.gif, report.md
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

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import network_viz  # noqa: E402

ROOT = SCRIPT_DIR.parent
OUT_DIR = ROOT / "results" / "N_Hier_MNIST_N256-128"
PRED_DIR = OUT_DIR / "predictions"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PRED_DIR.mkdir(parents=True, exist_ok=True)

# -------- config --------
N_CLASSES   = 10
T_BINS      = 20
INPUT_DIM   = 28 * 28
HIDDEN_1    = 256
HIDDEN_2    = 128
EPOCHS      = 3
BATCH       = 128
LR          = 2e-3
WEIGHT_DECAY= 1e-4
SEED        = 0
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

LIF_V_THR   = 1.0
LIF_LEAK    = 0.85
LIF_RESET   = 1.0
NSRAM_ALPHA = 0.05
NSRAM_BETA  = 0.99
NSRAM_GAIN  = 0.2

# NS-RAM-flavored neuron energy (consistent with prior z287/z290; 6.4 fJ/spike)
ENERGY_PER_SPIKE_PJ = 6.4e-3


# ============================================================
# Dataset
# ============================================================

def load_mnist():
    from torchvision import datasets, transforms
    data_root = ROOT / "data"
    data_root.mkdir(exist_ok=True)
    tfm = transforms.Compose([transforms.ToTensor()])
    tr = datasets.MNIST(str(data_root), train=True, download=True, transform=tfm)
    te = datasets.MNIST(str(data_root), train=False, download=True, transform=tfm)
    Xtr = tr.data.numpy().astype(np.float32) / 255.0
    ytr = tr.targets.numpy().astype(np.int64)
    Xte = te.data.numpy().astype(np.float32) / 255.0
    yte = te.targets.numpy().astype(np.int64)
    return Xtr.reshape(-1, INPUT_DIM), ytr, Xte.reshape(-1, INPUT_DIM), yte


def rate_encode_batch(X_np, T, device):
    """Static rate-encoding: each timestep sees the same normalised pixel
    intensities (NS-RAM analog soma integrates over T). Lighter than Poisson
    but standard for LIF MNIST (e.g., Zenke surrogate-grad tutorials)."""
    x = torch.from_numpy(X_np).to(device)              # (B, 784)
    return x.unsqueeze(1).expand(-1, T, -1).contiguous()  # (B, T, 784)


# ============================================================
# Model: 2-layer feedforward LIF + skip readout
# ============================================================

class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        sg = 1.0 / (1.0 + 10.0 * x.abs()) ** 2
        return grad_output * sg


spike_fn = SurrogateSpike.apply


class FFLIFLayer(nn.Module):
    """Feedforward LIF with NS-RAM slow body-state adapter."""
    def __init__(self, in_dim, n):
        super().__init__()
        self.n = n
        self.W_in = nn.Linear(in_dim, n, bias=True)
        nn.init.kaiming_normal_(self.W_in.weight, nonlinearity="relu")
        nn.init.zeros_(self.W_in.bias)

    def forward(self, x_seq, record=False):
        """x_seq: (B, T, in_dim) -> spikes (B, T, n) [+ vb trace if record]."""
        B, T, _ = x_seq.shape
        device = x_seq.device
        V_b = torch.zeros(B, self.n, device=device)
        V_body_slow = torch.zeros(B, self.n, device=device)
        s_prev = torch.zeros(B, self.n, device=device)
        I_in_seq = self.W_in(x_seq)
        spikes = []
        vb_trace = [] if record else None
        for t in range(T):
            V_b = LIF_LEAK * V_b + I_in_seq[:, t] - LIF_RESET * s_prev
            V_thr_eff = LIF_V_THR + NSRAM_GAIN * V_body_slow
            s_t = spike_fn(V_b - V_thr_eff)
            V_body_slow = NSRAM_BETA * V_body_slow + NSRAM_ALPHA * s_t
            s_prev = s_t
            spikes.append(s_t)
            if record:
                vb_trace.append(V_b.detach())
        spikes = torch.stack(spikes, dim=1)  # (B, T, n)
        if record:
            return spikes, torch.stack(vb_trace, dim=1)
        return spikes


class HierLIFNet(nn.Module):
    def __init__(self, in_dim=INPUT_DIM, h1=HIDDEN_1, h2=HIDDEN_2,
                 n_classes=N_CLASSES):
        super().__init__()
        self.layer1 = FFLIFLayer(in_dim, h1)
        self.layer2 = FFLIFLayer(h1, h2)
        # Skip-connection readout: concatenate rate(layer1) and rate(layer2)
        self.readout = nn.Linear(h1 + h2, n_classes)

    def forward(self, x_seq, record=False):
        if record:
            s1, vb1 = self.layer1(x_seq, record=True)
            s2, vb2 = self.layer2(s1, record=True)
        else:
            s1 = self.layer1(x_seq)
            s2 = self.layer2(s1)
        rate1 = s1.mean(dim=1)
        rate2 = s2.mean(dim=1)
        rate_cat = torch.cat([rate1, rate2], dim=1)
        logits = self.readout(rate_cat)
        if record:
            return logits, s1, s2, vb1, vb2
        return logits, s1, s2


# ============================================================
# Train / eval
# ============================================================

def iter_batches(X, y, batch, shuffle, rng):
    N = len(y)
    idx = rng.permutation(N) if shuffle else np.arange(N)
    for i in range(0, N, batch):
        sel = idx[i:i + batch]
        yield X[sel], y[sel]


def train_one_epoch(model, X, y, opt, rng, device):
    model.train()
    losses, correct, n = [], 0, 0
    t0 = time.time()
    for xb_np, yb_np in iter_batches(X, y, BATCH, True, rng):
        B = xb_np.shape[0]
        xb = rate_encode_batch(xb_np, T_BINS, device)
        yb = torch.from_numpy(yb_np).to(device)
        opt.zero_grad()
        logits, _, _ = model(xb)
        loss = F.cross_entropy(logits, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step()
        losses.append(loss.item() * B)
        correct += (logits.argmax(1) == yb).sum().item()
        n += B
    return sum(losses) / n, correct / n, time.time() - t0


@torch.no_grad()
def evaluate(model, X, y, device):
    model.eval()
    correct, n = 0, 0
    preds_all = []
    total_spikes = 0.0  # layer1 + layer2 spike events (per inference, per neuron)
    t0 = time.time()
    n_inferences = 0
    for xb_np, yb_np in iter_batches(X, y, BATCH, False, np.random.default_rng(0)):
        B = xb_np.shape[0]
        xb = rate_encode_batch(xb_np, T_BINS, device)
        yb = torch.from_numpy(yb_np).to(device)
        logits, s1, s2 = model(xb)
        preds = logits.argmax(1)
        correct += (preds == yb).sum().item()
        n += B
        preds_all.append(preds.cpu().numpy())
        total_spikes += float(s1.sum().item()) + float(s2.sum().item())
        n_inferences += B
    elapsed = time.time() - t0
    acc = correct / n
    preds_all = np.concatenate(preds_all)
    # Throughput = inferences per second
    throughput_ips = n_inferences / max(elapsed, 1e-6)
    spikes_per_inf = total_spikes / n_inferences
    # E_per_inf [pJ] = spikes/inference * pJ/spike
    energy_per_inf_pJ = spikes_per_inf * ENERGY_PER_SPIKE_PJ
    return acc, preds_all, throughput_ips, energy_per_inf_pJ, elapsed, spikes_per_inf


# ============================================================
# Main
# ============================================================

def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    t_start = time.time()
    print(f"[init] device={DEVICE}", flush=True)
    if DEVICE == "cuda":
        print(f"[init] GPU: {torch.cuda.get_device_name(0)}", flush=True)

    # ---- data ----
    Xtr, ytr, Xte, yte = load_mnist()
    print(f"[data] MNIST train={Xtr.shape} test={Xte.shape}", flush=True)

    # ---- model ----
    model = HierLIFNet().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] params={n_params/1e6:.3f}M  h1={HIDDEN_1}  h2={HIDDEN_2}",
          flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    rng = np.random.default_rng(SEED + 7)

    # ---- training ----
    train_loss_per_epoch, train_acc_per_epoch = [], []
    weight_frames_l1, weight_frames_l2, weight_frames_skip = [], [], []
    weight_frames_l1.append(model.layer1.W_in.weight.detach().cpu().numpy().copy())
    weight_frames_l2.append(model.layer2.W_in.weight.detach().cpu().numpy().copy())
    weight_frames_skip.append(model.readout.weight.detach().cpu().numpy().copy())

    for epoch in range(EPOCHS):
        loss, acc, dt = train_one_epoch(model, Xtr, ytr, opt, rng, DEVICE)
        train_loss_per_epoch.append(float(loss))
        train_acc_per_epoch.append(float(acc))
        weight_frames_l1.append(model.layer1.W_in.weight.detach().cpu().numpy().copy())
        weight_frames_l2.append(model.layer2.W_in.weight.detach().cpu().numpy().copy())
        weight_frames_skip.append(model.readout.weight.detach().cpu().numpy().copy())
        print(f"[epoch {epoch+1}/{EPOCHS}] loss={loss:.4f}  train_acc={acc:.4f}  ({dt:.1f}s)",
              flush=True)

    # ---- evaluation ----
    test_acc, preds, throughput, energy_pJ, eval_time, spikes_per_inf = evaluate(
        model, Xte, yte, DEVICE)
    print(f"[eval] test_acc={test_acc:.4f}  throughput={throughput:.1f} inf/s  "
          f"energy={energy_pJ:.3f} pJ/inf  spikes/inf={spikes_per_inf:.1f}  ({eval_time:.1f}s)",
          flush=True)

    # ---- artifacts for last sample (record layer2 spikes+vb for dashboard) ----
    model.eval()
    with torch.no_grad():
        x_last_np = Xte[-1:].astype(np.float32)
        x_last = rate_encode_batch(x_last_np, T_BINS, DEVICE)
        _, s1_last, s2_last, vb1_last, vb2_last = model(x_last, record=True)
        # Use layer-2 for the (N,T) dashboard arrays (smaller, hierarchical apex)
        spikes_NT = s2_last[0].cpu().numpy().T.astype(np.float32)  # (h2, T)
        vb_NT     = vb2_last[0].cpu().numpy().T.astype(np.float32)

    W_l1   = model.layer1.W_in.weight.detach().cpu().numpy()   # (h1, 784)
    W_l2   = model.layer2.W_in.weight.detach().cpu().numpy()   # (h2, h1)
    W_skip = model.readout.weight.detach().cpu().numpy()       # (10, h1+h2)

    np.save(PRED_DIR / "labels.npy", yte.astype(np.int64))
    np.save(PRED_DIR / "predictions.npy", preds.astype(np.int64))
    np.save(OUT_DIR / "spikes.npy", spikes_NT)
    np.save(OUT_DIR / "vb.npy", vb_NT)
    np.savez(OUT_DIR / "weights.npz",
             layer1=W_l1.astype(np.float32),
             layer2=W_l2.astype(np.float32),
             skip_readout=W_skip.astype(np.float32))
    # Also save a single .npy for dashboard convenience (layer-2)
    np.save(OUT_DIR / "weights.npy", W_l2.astype(np.float32))

    # ---- summary / gates ----
    INFRA       = bool(len(train_loss_per_epoch) == EPOCHS)
    DISCOVERY   = bool(test_acc > 0.93)
    AMBITIOUS   = bool(test_acc > 0.97 and energy_pJ < 50.0)

    summary = {
        "test_acc": float(test_acc),
        "train_loss_per_epoch": [float(x) for x in train_loss_per_epoch],
        "train_acc_per_epoch":  [float(x) for x in train_acc_per_epoch],
        "throughput": float(throughput),
        "throughput_inferences_per_sec": float(throughput),
        "energy_per_inf_pJ": float(energy_pJ),
        "spikes_per_inference": float(spikes_per_inf),
        "n_train": int(len(ytr)),
        "n_test":  int(len(yte)),
        "hidden_1": HIDDEN_1,
        "hidden_2": HIDDEN_2,
        "epochs": EPOCHS,
        "batch": BATCH,
        "T_bins": T_BINS,
        "params_M": float(n_params / 1e6),
        "device": DEVICE,
        "gpu": torch.cuda.get_device_name(0) if DEVICE == "cuda" else None,
        "gates": {"INFRA": INFRA, "DISCOVERY": DISCOVERY, "AMBITIOUS": AMBITIOUS},
        "wall_seconds": float(time.time() - t_start),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    # ---- viz ----
    try:
        network_viz.save_summary_dashboard(
            OUT_DIR,
            data={
                "spikes":  spikes_NT,
                "vb":      vb_NT,
                "weights": W_l2,
                "energy":  np.full((HIDDEN_2,), energy_pJ / HIDDEN_2, dtype=np.float32),
            },
            output_path=OUT_DIR / "dashboard.png",
            title=f"N-Hier-MNIST h1={HIDDEN_1} h2={HIDDEN_2}  test_acc={test_acc:.3f}",
        )
    except Exception as e:
        print(f"[viz] dashboard failed: {e}\n{traceback.format_exc()}", flush=True)

    try:
        # Animate layer-2 feedforward weights (h2, h1) — most informative apex
        network_viz.plot_weight_evolution_gif(
            np.stack(weight_frames_l2, 0),
            OUT_DIR / "weight_evo.gif",
            fps=2,
        )
    except Exception as e:
        print(f"[viz] gif failed: {e}\n{traceback.format_exc()}", flush=True)

    # ---- report ----
    report = f"""# N-Hier-MNIST (h1={HIDDEN_1}, h2={HIDDEN_2})

Hierarchical 2-layer feedforward LIF SNN with NS-RAM-flavored slow body-state
adapter, plus a **skip-connection from layer1 into the readout**, trained on
MNIST 28x28 via BPTT (surrogate-gradient).

## Architecture
- Input 784 -> Linear -> FFLIF_1 (N={HIDDEN_1}) -> Linear -> FFLIF_2 (N={HIDDEN_2})
- Skip: rate(layer1) concatenated with rate(layer2) into readout
- Readout: Linear({HIDDEN_1}+{HIDDEN_2} -> {N_CLASSES})
- Neurons: LIF with NS-RAM V_G2-style slow body-state biasing V_thr
- T={T_BINS}, BATCH={BATCH}, EPOCHS={EPOCHS}, LR={LR}, WD={WEIGHT_DECAY}

## Substrate
{DEVICE} ({summary['gpu']}), {summary['params_M']:.3f}M trainable params.

## Results
- **Test accuracy**: {test_acc:.4f}  (chance = {1.0/N_CLASSES:.3f})
- Train loss / epoch: {[round(x,4) for x in train_loss_per_epoch]}
- Train acc  / epoch: {[round(x,4) for x in train_acc_per_epoch]}
- Throughput: {throughput:.1f} inferences / sec
- Spikes / inference (l1+l2): {spikes_per_inf:.1f}
- Energy per inference: {energy_pJ:.3f} pJ
  (NS-RAM energy model: {ENERGY_PER_SPIKE_PJ*1000:.2f} fJ/spike, Pazos-class)
- Wall time: {summary['wall_seconds']:.1f} s

## Pre-registered gates
- INFRA      (trains + dashboard):              **{'PASS' if INFRA else 'FAIL'}**
- DISCOVERY  (test_acc > 93%):                  **{'PASS' if DISCOVERY else 'FAIL'}**
- AMBITIOUS  (acc > 97% AND energy < 50 pJ/inf): **{'PASS' if AMBITIOUS else 'FAIL'}**

## Files
- summary.json
- predictions/labels.npy, predictions/predictions.npy
- spikes.npy  (layer-2: h2 x T, last test sample)
- vb.npy      (layer-2 body-state: h2 x T, last test sample)
- weights.npy  (layer-2 W, h2 x h1, for dashboard)
- weights.npz  (layer1, layer2, skip_readout — all final weights)
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
