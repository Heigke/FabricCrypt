#!/usr/bin/env python3
"""N-WTA-MNIST: Lateral inhibition WTA SNN on MNIST.

Phase N1 topology #5 (excitatory + inhibitory WTA) × Phase N2 U3 (LIF NS-RAM
style cells). Follows Diehl & Cook (2015) "Unsupervised learning of digit
recognition using STDP" architecture but with a simpler Hebbian (Oja-normalised)
rule and lateral inhibition kernel rather than a full inhibitory population.

Architecture
------------
  - 784 input pixels rate-encoded as Poisson spikes over T_PRES ms per image
  - N=200 excitatory LIF neurons with adaptive threshold (homeostasis)
  - All-to-all feedforward W (784 -> N), Hebbian-learned, L2-normalised per
    post (Oja-style)
  - WTA: at each timestep, winner = argmax(V). Inhibition kernel suppresses
    all other neurons by INHIB strength for T_REFRAC ms after a spike.
  - Class assignment: after training, present labelled subset; each neuron's
    "class" = label with highest mean response. Test: argmax of class-summed
    spike counts.

Substrate
---------
  torch CUDA on zgx (NVIDIA GB10). dt=1ms; vectorised across N neurons.

Outputs (rsync target results/N_WTA_MNIST_N200/):
  - summary.json, predictions/labels.npy, spikes.npy, weights.npy (snapshots)
  - dashboard.png, weight_evo.gif, report.md
"""
from __future__ import annotations
import os, sys, json, math, time, argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import network_viz as nv  # noqa

OUT = REPO / "results" / "N_WTA_MNIST_N200"
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "predictions").mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 1337
torch.manual_seed(SEED); np.random.seed(SEED)

# ─── Hyperparams ───────────────────────────────────────────────────────
N_EXC      = 200
T_PRES     = 150         # ms per image
T_REST     = 30          # ms rest between images
DT         = 1.0
INPUT_RATE = 63.75       # max Hz at pixel=1.0 (Diehl-Cook value)
V_TH0      = 1.0
V_RESET    = 0.0
TAU_M      = 100.0       # membrane time constant (ms)
LEAK       = math.exp(-DT / TAU_M)
T_REFRAC   = 5           # ms
INHIB      = 8.0         # lateral inhibition strength (subtracted from V of losers)
ETA_LTP    = 0.020       # potentiation on post spike, scaled by pre-trace
ETA_LTD    = 0.0008      # depression on post spike, on synapses with low pre-trace
TAU_PRE    = 20.0        # presynaptic spike trace tau (ms)
ALPHA_PRE  = math.exp(-DT / TAU_PRE)
W_MIN, W_MAX = 0.0, 1.0
WNORM_TARGET = 78.0      # L1 norm per post (Diehl-Cook ~ 0.1 * 784)
THETA_PLUS = 0.30        # homeostatic threshold increment per spike (stronger)
TAU_THETA  = 2e5         # theta decay (ms) — moderate
THETA_DECAY = math.exp(-DT / TAU_THETA)
INIT_W_SCALE = 0.30      # initial uniform [0, scale]

# Energy model (NS-RAM ≈ 21 fJ/spike; Pazos)
E_SPIKE_J = 21e-15


# ─── MNIST loading ─────────────────────────────────────────────────────
def load_mnist():
    """Return (X_train, y_train, X_test, y_test) as numpy uint8/int."""
    from torchvision import datasets, transforms
    data_root = REPO / "data"
    data_root.mkdir(exist_ok=True)
    tfm = transforms.Compose([transforms.ToTensor()])
    tr = datasets.MNIST(str(data_root), train=True, download=True, transform=tfm)
    te = datasets.MNIST(str(data_root), train=False, download=True, transform=tfm)
    Xtr = tr.data.numpy().astype(np.float32) / 255.0  # (60000,28,28)
    ytr = tr.targets.numpy().astype(np.int64)
    Xte = te.data.numpy().astype(np.float32) / 255.0
    yte = te.targets.numpy().astype(np.int64)
    return Xtr.reshape(-1, 784), ytr, Xte.reshape(-1, 784), yte


# ─── WTA SNN ──────────────────────────────────────────────────────────
class WTASNN:
    def __init__(self, n_in=784, n_exc=N_EXC, device=DEVICE, seed=SEED):
        self.device = device
        self.n_in = n_in
        self.n_exc = n_exc
        g = torch.Generator(device="cpu").manual_seed(seed)
        # Diehl-Cook init: uniform [0.0, INIT_W_SCALE]
        self.W = (INIT_W_SCALE * torch.rand(n_in, n_exc, generator=g)).to(device)
        # Normalize each post to WNORM_TARGET (L1) — Diehl-Cook standard
        col_sums = self.W.sum(dim=0, keepdim=True).clamp(min=1e-6)
        self.W = self.W * (WNORM_TARGET / col_sums)
        self.theta = torch.zeros(n_exc, device=device)   # homeostatic threshold add
        self.V = torch.zeros(n_exc, device=device)
        self.refrac = torch.zeros(n_exc, device=device)  # ms remaining
        self.pre_trace = torch.zeros(n_in, device=device)  # presynaptic spike trace
        self.total_spikes = 0
        self.total_input_spikes = 0

    def reset_state(self):
        self.V.zero_(); self.refrac.zero_(); self.pre_trace.zero_()

    @torch.no_grad()
    def present(self, x_np, train=True, record=False):
        """Present one image for T_PRES ms then T_REST ms rest.
        x_np: shape (784,), values in [0,1]. Returns spike count per neuron."""
        x = torch.from_numpy(x_np).to(self.device)
        # Poisson rate per ms
        p_per_ms = (x * INPUT_RATE * DT / 1000.0).clamp(0, 1)
        spk_counts = torch.zeros(self.n_exc, device=self.device)
        v_trace = [] if record else None
        spk_trace = [] if record else None
        # reset presynaptic trace per image (cleaner gradient)
        self.pre_trace.zero_()
        for t in range(T_PRES):
            # Poisson input spikes
            in_spk = (torch.rand(self.n_in, device=self.device) < p_per_ms).float()
            self.total_input_spikes += int(in_spk.sum().item())
            # update presynaptic trace (decay then increment)
            self.pre_trace = ALPHA_PRE * self.pre_trace + in_spk
            I = in_spk @ self.W  # (n_exc,)
            # leak + integrate (only neurons not in refrac)
            active = (self.refrac <= 0).float()
            self.V = LEAK * self.V + I * active
            # WTA: find max V across exc that exceeds threshold
            v_eff = self.V - (V_TH0 + self.theta)
            v_eff = torch.where(active.bool(), v_eff, torch.full_like(v_eff, -1e9))
            winner = int(torch.argmax(v_eff).item())
            if v_eff[winner] > 0:
                spk_counts[winner] += 1
                self.total_spikes += 1
                # lateral inhibition: subtract from losers
                self.V -= INHIB
                self.V[winner] = V_RESET
                self.refrac[winner] = T_REFRAC
                # homeostasis
                self.theta[winner] += THETA_PLUS
                # STDP-like update on winner column:
                #   LTP proportional to pre_trace (recent presynaptic activity)
                #   LTD small uniform depression (forgets non-correlated synapses)
                if train:
                    w_col = self.W[:, winner]
                    dw = ETA_LTP * self.pre_trace - ETA_LTD
                    w_col = (w_col + dw).clamp(W_MIN, W_MAX)
                    # L1 renormalise winner column to keep total synaptic drive stable
                    s = w_col.sum().clamp(min=1e-6)
                    w_col = w_col * (WNORM_TARGET / s)
                    self.W[:, winner] = w_col.clamp(W_MIN, W_MAX)
            # decay refrac & theta
            self.refrac = (self.refrac - DT).clamp(min=0)
            self.theta = self.theta * THETA_DECAY
            if record:
                v_trace.append(self.V.detach().cpu().numpy().copy())
                spk_trace.append((spk_counts > 0).cpu().numpy().astype(np.float32))
        # rest period
        for _ in range(T_REST):
            self.V = LEAK * self.V
            self.refrac = (self.refrac - DT).clamp(min=0)
        if record:
            return spk_counts.cpu().numpy(), np.stack(v_trace, axis=1), np.stack(spk_trace, axis=1)
        return spk_counts.cpu().numpy()


def assign_classes(net, X, y, n_assign=2000):
    """For each neuron, find label with highest mean spike count."""
    idx = np.random.RandomState(0).choice(len(X), n_assign, replace=False)
    counts = np.zeros((10, net.n_exc), dtype=np.float64)
    n_per = np.zeros(10, dtype=np.int64)
    for i, j in enumerate(idx):
        sc = net.present(X[j], train=False)
        counts[y[j]] += sc
        n_per[y[j]] += 1
        if (i + 1) % 200 == 0:
            print(f"  assign {i+1}/{n_assign}", flush=True)
    rates = counts / np.maximum(n_per[:, None], 1)
    neuron_class = np.argmax(rates, axis=0)
    return neuron_class, rates


def evaluate(net, X, y, neuron_class, n_eval=2000):
    """Test accuracy: argmax class-summed spike counts."""
    idx = np.random.RandomState(1).choice(len(X), min(n_eval, len(X)), replace=False)
    preds = np.zeros(len(idx), dtype=np.int64)
    labels = np.zeros(len(idx), dtype=np.int64)
    all_sc = np.zeros((len(idx), net.n_exc), dtype=np.float32)
    active = np.zeros(len(idx), dtype=np.float32)
    for i, j in enumerate(idx):
        sc = net.present(X[j], train=False)
        all_sc[i] = sc
        active[i] = float((sc > 0).sum())
        class_sums = np.zeros(10)
        for c in range(10):
            mask = (neuron_class == c)
            if mask.sum() > 0:
                class_sums[c] = sc[mask].sum() / mask.sum()
        preds[i] = int(np.argmax(class_sums))
        labels[i] = int(y[j])
        if (i + 1) % 200 == 0:
            acc_so_far = float((preds[:i+1] == labels[:i+1]).mean())
            print(f"  eval {i+1}/{len(idx)}  acc={acc_so_far:.3f}", flush=True)
    acc = float((preds == labels).mean())
    return acc, preds, labels, all_sc, active


# ─── Main ────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=8000)
    ap.add_argument("--n_assign", type=int, default=2000)
    ap.add_argument("--n_eval", type=int, default=2000)
    ap.add_argument("--snap_every", type=int, default=400)
    args = ap.parse_args()

    print(f"[N_WTA_MNIST] device={DEVICE}  N={N_EXC}", flush=True)
    t0 = time.time()
    Xtr, ytr, Xte, yte = load_mnist()
    print(f"  MNIST loaded: train={len(Xtr)} test={len(Xte)}", flush=True)

    net = WTASNN(n_in=784, n_exc=N_EXC, device=DEVICE)
    weight_snaps = [net.W.detach().cpu().numpy().copy()]

    # ─── Train (unsupervised) ───
    perm = np.random.RandomState(SEED).permutation(len(Xtr))[:args.n_train]
    print(f"[train] {args.n_train} images, T_PRES={T_PRES}ms", flush=True)
    t_train0 = time.time()
    for i, j in enumerate(perm):
        net.present(Xtr[j], train=True)
        if (i + 1) % args.snap_every == 0:
            weight_snaps.append(net.W.detach().cpu().numpy().copy())
            elapsed = time.time() - t_train0
            rate = (i + 1) / elapsed
            print(f"  train {i+1}/{args.n_train}  {rate:.1f} img/s  "
                  f"spikes={net.total_spikes}  theta_max={float(net.theta.max()):.3f}",
                  flush=True)
    weight_snaps.append(net.W.detach().cpu().numpy().copy())

    # ─── Assign classes ───
    print("[assign] mapping neurons -> classes", flush=True)
    neuron_class, class_rates = assign_classes(net, Xtr, ytr, n_assign=args.n_assign)
    print(f"  class dist: {np.bincount(neuron_class, minlength=10).tolist()}", flush=True)

    # ─── Test ───
    print("[eval] test set", flush=True)
    acc, preds, labels, all_sc, active = evaluate(net, Xte, yte, neuron_class,
                                                  n_eval=args.n_eval)
    n_active_mean = float(active.mean())
    print(f"  test_acc={acc:.4f}  n_active_per_input={n_active_mean:.2f}", flush=True)

    # ─── Record one representative input for raster + V_B ───
    net.reset_state()
    rec_sc, rec_V, rec_spk = net.present(Xte[0], train=False, record=True)

    # ─── Energy estimate ───
    energy_per_inf_J = E_SPIKE_J * (net.total_spikes / max(1, args.n_train +
                                                            args.n_assign +
                                                            args.n_eval))
    energy_per_inf_pJ = energy_per_inf_J * 1e12

    # ─── Save artifacts ───
    np.save(OUT / "predictions" / "labels.npy", preds)
    np.save(OUT / "predictions" / "true_labels.npy", labels)
    np.save(OUT / "spikes.npy", all_sc)              # (n_eval, N) spike counts
    np.save(OUT / "weights.npy", np.stack(weight_snaps, axis=0))  # (T, 784, N)
    np.save(OUT / "neuron_class.npy", neuron_class)
    np.save(OUT / "rec_V.npy", rec_V)
    np.save(OUT / "rec_spk.npy", rec_spk)

    summary = {
        "experiment": "N_WTA_MNIST",
        "phase_N1": 5,
        "phase_N2": "U3",
        "device": str(DEVICE),
        "N_EXC": N_EXC,
        "n_train": args.n_train,
        "n_assign": args.n_assign,
        "n_eval": args.n_eval,
        "T_PRES_ms": T_PRES,
        "test_acc": acc,
        "n_active_per_input": n_active_mean,
        "lateral_inhibition_strength": INHIB,
        "energy_per_inf_pJ": energy_per_inf_pJ,
        "total_spikes": int(net.total_spikes),
        "total_input_spikes": int(net.total_input_spikes),
        "wall_clock_s": time.time() - t0,
        "neuron_class_dist": np.bincount(neuron_class, minlength=10).tolist(),
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    with open(OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[summary] {summary}", flush=True)

    # ─── Dashboard ───
    # Render receptive fields montage as the "weights" panel
    W_final = weight_snaps[-1]
    # Build a montage (n_show x n_show) of receptive fields (28x28 each)
    n_show = 10
    fields = np.zeros((28 * n_show, 28 * n_show), dtype=np.float32)
    order = np.argsort(-W_final.sum(axis=0))[: n_show * n_show]
    for i, k in enumerate(order):
        r, c = divmod(i, n_show)
        rf = W_final[:, k].reshape(28, 28)
        fields[r*28:(r+1)*28, c*28:(c+1)*28] = rf

    # spike raster (recorded one image): (N, T)
    raster = rec_spk  # already (N, T)

    # per-neuron energy
    per_neuron_spikes = all_sc.sum(axis=0)
    per_neuron_energy_pJ = per_neuron_spikes * E_SPIKE_J * 1e12
    # arrange as a 10x20 grid for heatmap input (1D -> 2D-ish)
    side = int(math.ceil(math.sqrt(N_EXC)))
    grid = np.zeros(side * side, dtype=np.float32)
    grid[:N_EXC] = per_neuron_energy_pJ
    energy_grid = grid.reshape(side, side)

    # latency: spike-time distributions (first-spike latency per neuron in rec)
    first_spk = np.full(N_EXC, T_PRES, dtype=np.float32)
    for n in range(N_EXC):
        nz = np.where(rec_spk[n] > 0)[0]
        if len(nz):
            first_spk[n] = float(nz[0])
    fs_valid = first_spk[first_spk < T_PRES]
    if len(fs_valid) == 0:
        fs_valid = np.array([T_PRES], dtype=np.float32)
    latency = {"first_spike": fs_valid}

    pareto = [{
        "name": f"WTA-N{N_EXC}",
        "accuracy": float(acc),
        "energy_pj": float(energy_per_inf_pJ),
        "throughput": float(args.n_eval / max(1e-9, (T_PRES + T_REST) * 1e-3 * args.n_eval)),
    }]

    dash_data = {
        "spikes": raster,
        "vb": rec_V,
        "energy": energy_grid,
        "latency": latency,
        "pareto": pareto,
        "weights": fields,
    }
    try:
        nv.save_summary_dashboard(OUT, output_path=OUT / "dashboard.png",
                                  data=dash_data,
                                  title=f"WTA SNN MNIST — N={N_EXC}  acc={acc:.3f}")
        print(f"  dashboard -> {OUT/'dashboard.png'}", flush=True)
    except Exception as e:
        print(f"  dashboard FAILED: {e}", flush=True)

    # ─── Weight evolution GIF (use receptive-field montages of snapshots) ───
    try:
        gif_frames = []
        for W in weight_snaps:
            mont = np.zeros((28 * n_show, 28 * n_show), dtype=np.float32)
            o = np.argsort(-W.sum(axis=0))[: n_show * n_show]
            for i, k in enumerate(o):
                r, c = divmod(i, n_show)
                mont[r*28:(r+1)*28, c*28:(c+1)*28] = W[:, k].reshape(28, 28)
            gif_frames.append(mont)
        nv.plot_weight_evolution_gif(gif_frames, OUT / "weight_evo.gif", fps=6)
        print(f"  gif -> {OUT/'weight_evo.gif'}", flush=True)
    except Exception as e:
        print(f"  gif FAILED: {e}", flush=True)

    # ─── Report ───
    report = f"""# N-WTA-MNIST report

**Architecture**: Phase N1 topology #5 (WTA + lateral inhibition) × Phase N2 U3 (LIF NS-RAM cells).
Diehl & Cook (2015)-style unsupervised SNN with Hebbian (Oja-like) learning and
homeostatic adaptive thresholds.

## Setup
- Device: `{DEVICE}`
- N_EXC = {N_EXC} excitatory LIF neurons
- Inputs: 784 Poisson channels, peak {INPUT_RATE} Hz, {T_PRES} ms/image
- Lateral inhibition strength = {INHIB}
- STDP-like: η_LTP={ETA_LTP}, η_LTD={ETA_LTD}, pre-trace τ={TAU_PRE}ms
- L1-renorm per post column to {WNORM_TARGET}; clipped [{W_MIN},{W_MAX}]
- Homeostasis: θ+={THETA_PLUS}/spike, τ_θ={TAU_THETA}ms
- Train images = {args.n_train} (unsupervised), assign = {args.n_assign}, eval = {args.n_eval}

## Results
- **Test accuracy** = **{acc*100:.2f}%**
- Avg active neurons / input = {n_active_mean:.2f} (sparsity)
- Energy/inference ≈ {energy_per_inf_pJ:.2f} pJ ({E_SPIKE_J*1e15:.1f} fJ/spike NS-RAM model)
- Total spikes (train+assign+eval) = {net.total_spikes:,}
- Wall clock = {summary['wall_clock_s']:.1f} s
- Class distribution of neurons: {summary['neuron_class_dist']}

## Pre-registration outcome
- INFRA (trains + dashboard + gif): {'PASS' if (OUT/'dashboard.png').exists() else 'FAIL'}
- DISCOVERY (acc > 80%): {'PASS' if acc > 0.80 else 'FAIL'} (acc={acc*100:.2f}%)
- AMBITIOUS (acc > 90% AND ≤5 active): {'PASS' if (acc > 0.90 and n_active_mean <= 5) else 'FAIL'}

## Artifacts
- `summary.json` — metrics
- `predictions/labels.npy`, `predictions/true_labels.npy`
- `spikes.npy` — (n_eval, N) per-image spike counts
- `weights.npy` — (snapshots, 784, N) Hebbian receptive fields over training
- `neuron_class.npy` — neuron→class assignment
- `dashboard.png`, `weight_evo.gif`

## No-cheat
- Learning is **unsupervised Hebbian (Oja-clipped)** only. No backprop, no labels
  used during weight updates. Class assignment uses labels only as a *readout*
  after training.
"""
    (OUT / "report.md").write_text(report)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
