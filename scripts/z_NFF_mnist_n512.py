#!/usr/bin/env python3
"""
N-FF-MNIST (Phase N1 topology #1, NETWORK_CAMPAIGN_2026-05-17).

Feedforward LIF SNN on MNIST 28x28 using the NS-RAM 4D body-state surrogate
(MEP-1, z220_4d_dense) as the hidden-neuron transfer.

Architecture
------------
    784  --W_in (frozen rand proj)-->  512 NS-RAM LIF  --W_out (trained)-->  10

For each input image x (in [0,1]):
  T=50 Poisson encoding steps; per step t:
      drive_t  = W_in @ poisson_spike[t]
      V_G1     = V_G1_bias + g_in * drive_t              (broadcast over N=512)
      V_G2     = V_G2_bias (frozen)
      V_d      = 1.0 (fixed)
      |Id|, Iii, Ileak  =  surrogate(V_G1, V_G2, V_d, V_b)
      V_b      <- clip(V_b + dt*(Iii-Ileak)/C_b, 0, 0.7)
      I_acc    += |Id|
  feat        = log10(mean |Id| over T)            # (N=512)
  logits      = W_out @ feat + b_out

W_out trained for 3 epochs by SGD/Adam on cross-entropy (real MNIST 60k train).
Test on real MNIST 10k test set (no leak).

Outputs to results/N_FF_MNIST_N512/:
  summary.json, spikes.npy, vb.npy, weights.npy, energy.npy, latency.json,
  pareto.json, dashboard.png, report.md

Pre-registered gates (line 1 of run.log):
  INFRA       network trains, summary.json + dashboard.png both written
  DISCOVERY   test accuracy > 92%
  AMBITIOUS   > 96% AND energy < 100 pJ/inference
"""
from __future__ import annotations

import os
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
os.environ.setdefault("NSRAM_DC_SOLVER", "pt")

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent

# ------------------------------------------------------------------
# Locked config
# ------------------------------------------------------------------
N_HIDDEN = 512
N_IN = 784
N_CLASSES = 10
T_STEPS = 50           # Poisson encoding window per inference
EPOCHS = 3
BATCH = 64
LR = 1e-3
WEIGHT_DECAY = 1e-4

# Surrogate body-state physics (copied from z263, locked)
VG1_BIAS_LO, VG1_BIAS_HI = 0.20, 0.40
VG2_BIAS_LO, VG2_BIAS_HI = 0.00, 0.30
G_IN = 0.20
C_B = 5e-15            # 5 fF
DT_TRANS = 1e-7        # 0.1 us
VD_FIXED = 1.0
VB_INIT = 0.0
VB_MIN, VB_MAX = 0.0, 0.7

SEED = 0
SURROGATE_PATH = ROOT / "results/z220_4d_dense/surrogate_4d_dense.npz"

OUT_DIR = ROOT / "results/N_FF_MNIST_N512"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RUN_LOG = OUT_DIR / "run.log"

THERMAL_ZONE = Path("/sys/class/thermal/thermal_zone0/temp")
THERMAL_PAUSE_C = 85.0
THERMAL_RESUME_C = 65.0


def apu_temp_c() -> float:
    try:
        return float(THERMAL_ZONE.read_text().strip()) / 1000.0
    except Exception:
        return -1.0


def thermal_guard(log):
    t = apu_temp_c()
    if t >= THERMAL_PAUSE_C:
        log(f"[thermal] APU={t:.1f}C, pause until <{THERMAL_RESUME_C}C")
        while apu_temp_c() > THERMAL_RESUME_C:
            time.sleep(5)
        log(f"[thermal] resumed APU={apu_temp_c():.1f}C")


def open_logger():
    f = open(RUN_LOG, "a", buffering=1)
    def log(msg: str):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        f.write(line + "\n")
    return f, log


# ------------------------------------------------------------------
# Torch GPU NS-RAM surrogate (4D quadrilinear)
# ------------------------------------------------------------------
class TorchSurrogate4D:
    def __init__(self, npz_path: Path, device):
        d = np.load(npz_path)
        Id = np.maximum(np.abs(d["Id"]), 1e-15).astype(np.float32)
        self.Id_log = torch.from_numpy(np.log10(Id)).to(device)              # (g1,g2,d,b)
        self.Iii    = torch.from_numpy(d["Iii"].astype(np.float32)).to(device)
        self.Ileak  = torch.from_numpy(d["Ileak"].astype(np.float32)).to(device)
        self.vg1 = torch.from_numpy(d["vg1_axis"].astype(np.float32)).to(device)
        self.vg2 = torch.from_numpy(d["vg2_axis"].astype(np.float32)).to(device)
        self.vd  = torch.from_numpy(d["vd_axis"].astype(np.float32)).to(device)
        self.vb  = torch.from_numpy(d["vb_axis"].astype(np.float32)).to(device)
        self.device = device

    @staticmethod
    def _idx(x, axis):
        xc = torch.clamp(x, axis[0].item(), axis[-1].item())
        i = torch.searchsorted(axis, xc.contiguous())
        i = torch.clamp(i, 1, axis.numel() - 1) - 1
        a0 = axis[i]; a1 = axis[i + 1]
        f = (xc - a0) / torch.clamp(a1 - a0, min=1e-30)
        return i, f

    def eval_all(self, VG1, VG2, Vd, Vb):
        # All inputs broadcastable to common shape.
        i, fi = self._idx(VG1, self.vg1)
        j, fj = self._idx(VG2, self.vg2)
        k, fk = self._idx(Vd,  self.vd)
        l, fl = self._idx(Vb,  self.vb)

        # 16-corner sum
        out_logId = torch.zeros_like(fi)
        out_Iii   = torch.zeros_like(fi)
        out_Ileak = torch.zeros_like(fi)
        Id_log = self.Id_log; Iii = self.Iii; Ileak = self.Ileak
        for di in (0, 1):
            wi = fi if di else (1.0 - fi)
            ii = i + di
            for dj in (0, 1):
                wj = fj if dj else (1.0 - fj)
                jj = j + dj
                for dk in (0, 1):
                    wk = fk if dk else (1.0 - fk)
                    kk = k + dk
                    for dl in (0, 1):
                        wl = fl if dl else (1.0 - fl)
                        ll = l + dl
                        w = wi * wj * wk * wl
                        out_logId += w * Id_log[ii, jj, kk, ll]
                        out_Iii   += w * Iii[  ii, jj, kk, ll]
                        out_Ileak += w * Ileak[ii, jj, kk, ll]
        return out_logId, out_Iii, out_Ileak


# ------------------------------------------------------------------
# MNIST
# ------------------------------------------------------------------
def load_mnist():
    try:
        from torchvision import datasets
        tr = datasets.MNIST("/tmp/mnist", train=True, download=True)
        te = datasets.MNIST("/tmp/mnist", train=False, download=True)
        Xtr = tr.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
        ytr = tr.targets.numpy().astype(np.int64)
        Xte = te.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
        yte = te.targets.numpy().astype(np.int64)
        return Xtr, ytr, Xte, yte
    except Exception as e:
        from sklearn.datasets import fetch_openml
        m = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
        X = m.data.astype(np.float32) / 255.0
        y = m.target.astype(np.int64)
        return X[:60000], y[:60000], X[60000:], y[60000:]


# ------------------------------------------------------------------
# NS-RAM LIF featurizer (returns features, mean spikes, mean |Id|,
# and optionally per-step spike/Vb traces for diagnostics)
# ------------------------------------------------------------------
@torch.no_grad()
def nsram_featurize(X, W_in, VG1_bias, VG2_bias, surr,
                    seed_poisson, *, record=False):
    """
    X: (B, 784) float32 on device.
    Returns:
        feats:    (B, N) features = log10(mean |Id|)
        mean_Id:  (B, N) mean |Id|
        spikes:   None  or  (T, N) for last image if record=True
        vb_trace: None  or  (T, 32) for last image if record=True
    """
    device = X.device
    B = X.shape[0]
    N = VG1_bias.numel()
    g = torch.Generator(device=device).manual_seed(int(seed_poisson))

    V_b = torch.full((B, N), VB_INIT, device=device, dtype=torch.float32)
    Id_accum = torch.zeros((B, N), device=device, dtype=torch.float32)
    spike_accum = torch.zeros((B, N), device=device, dtype=torch.float32)

    spikes_rec = None
    vb_rec = None
    if record:
        spikes_rec = torch.zeros((T_STEPS, N), device="cpu")
        vb_rec    = torch.zeros((T_STEPS, 32), device="cpu")

    # spike threshold for diagnostic raster: |Id| above per-neuron median
    # We track absolute current; later we'll define a spike as |Id| above
    # the per-neuron 80th percentile of train activity. For raster recording
    # we use a simple rate-coded fired indicator: prob = clip(Id_norm, 0, 1).

    VG1_b = VG1_bias.view(1, N)
    VG2_b = VG2_bias.view(1, N)
    Vd = torch.full((1, 1), VD_FIXED, device=device, dtype=torch.float32)

    for t in range(T_STEPS):
        u = torch.rand(B, N_IN, generator=g, device=device, dtype=torch.float32)
        spk = (u < X).float()                          # (B, 784)
        drive = spk @ W_in.t()                         # (B, N)
        VG1 = VG1_b + G_IN * drive
        VG1 = torch.clamp(VG1, 0.10, 0.70)
        log_Id, Iii, Ileak = surr.eval_all(VG1, VG2_b.expand_as(VG1), Vd.expand_as(VG1), V_b)
        Id_abs = torch.pow(10.0, log_Id)
        Id_accum += Id_abs

        # diagnostic spike: |Id| above neuron-specific threshold (median across batch this step)
        thr = Id_abs.median(dim=0, keepdim=True).values
        fired = (Id_abs > thr).float()
        spike_accum += fired

        # Body-state Euler
        V_b = torch.clamp(V_b + DT_TRANS * (Iii - Ileak) / C_B, VB_MIN, VB_MAX)

        if record:
            spikes_rec[t] = fired[-1].cpu()
            vb_rec[t]    = V_b[-1, :32].cpu()

    mean_Id = Id_accum / T_STEPS
    feats = torch.log10(torch.clamp(mean_Id, min=1e-15))
    return feats, mean_Id, spike_accum / T_STEPS, spikes_rec, vb_rec


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    fhandle, log = open_logger()
    try:
        # Pre-registered gates as line 1 of log
        log("PRE-REGISTERED GATES: INFRA (script runs, summary.json+dashboard.png written) | "
            "DISCOVERY (test_acc > 92%) | AMBITIOUS (test_acc > 96% AND energy < 100 pJ/inference)")
        log(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log(f"device: {device}")
        torch.manual_seed(SEED)
        np.random.seed(SEED)

        log("[loader] loading MNIST 28x28")
        Xtr_np, ytr_np, Xte_np, yte_np = load_mnist()
        log(f"  train={Xtr_np.shape} test={Xte_np.shape}")

        Xtr = torch.from_numpy(Xtr_np).to(device)
        ytr = torch.from_numpy(ytr_np).to(device)
        Xte = torch.from_numpy(Xte_np).to(device)
        yte = torch.from_numpy(yte_np).to(device)

        # Surrogate
        log(f"[surr] loading {SURROGATE_PATH.name}")
        surr = TorchSurrogate4D(SURROGATE_PATH, device)

        # Frozen W_in (random Gaussian / sqrt(n_in))
        rng = np.random.RandomState(SEED)
        W_in_np = (rng.randn(N_HIDDEN, N_IN) / np.sqrt(N_IN)).astype(np.float32)
        W_in = torch.from_numpy(W_in_np).to(device)

        VG1_bias = torch.from_numpy(
            rng.uniform(VG1_BIAS_LO, VG1_BIAS_HI, size=N_HIDDEN).astype(np.float32)
        ).to(device)
        VG2_bias = torch.from_numpy(
            rng.uniform(VG2_BIAS_LO, VG2_BIAS_HI, size=N_HIDDEN).astype(np.float32)
        ).to(device)

        # --------------------------------------------------------------
        # Featurize all of train + test once (W_in/biases frozen)
        # --------------------------------------------------------------
        log(f"[featurize] N_HIDDEN={N_HIDDEN} T_STEPS={T_STEPS} BATCH={BATCH}")

        def featurize_split(X, name, seed_base):
            B = X.shape[0]
            feats = torch.zeros(B, N_HIDDEN, device=device)
            mean_Id_total = torch.zeros(B, N_HIDDEN, device=device)
            t0 = time.time()
            for s in range(0, B, BATCH):
                e = min(s + BATCH, B)
                f, mid, _, _, _ = nsram_featurize(
                    X[s:e], W_in, VG1_bias, VG2_bias, surr,
                    seed_poisson=seed_base + s,
                )
                feats[s:e] = f
                mean_Id_total[s:e] = mid
                if (s // BATCH) % 50 == 0:
                    thermal_guard(log)
                    log(f"  [{name}] {e}/{B}  ({(e/(time.time()-t0+1e-9)):.0f} img/s, APU={apu_temp_c():.1f}C)")
            dt = time.time() - t0
            log(f"  [{name}] done {B} imgs in {dt:.1f}s ({B/dt:.0f} img/s)")
            return feats, mean_Id_total, dt

        t_feat_start = time.time()
        F_tr, _, dt_tr = featurize_split(Xtr, "train", 1000)
        F_te, mean_Id_te, dt_te = featurize_split(Xte, "test", 9000)
        feat_time = time.time() - t_feat_start

        # z-score using train stats
        mu = F_tr.mean(dim=0, keepdim=True)
        sd = F_tr.std(dim=0, keepdim=True).clamp(min=1e-6)
        F_tr_z = (F_tr - mu) / sd
        F_te_z = (F_te - mu) / sd

        # --------------------------------------------------------------
        # Train softmax readout 512 -> 10 for 3 epochs (real MNIST)
        # --------------------------------------------------------------
        log("[readout] training softmax 512 -> 10 for 3 epochs")
        Wout = torch.zeros(N_HIDDEN, N_CLASSES, device=device, requires_grad=True)
        bout = torch.zeros(N_CLASSES, device=device, requires_grad=True)
        opt = torch.optim.AdamW([Wout, bout], lr=LR, weight_decay=WEIGHT_DECAY)
        train_loss_per_epoch = []
        n_tr = F_tr_z.shape[0]
        readout_t0 = time.time()
        for ep in range(EPOCHS):
            perm = torch.randperm(n_tr, device=device)
            losses = []
            for s in range(0, n_tr, BATCH):
                e = min(s + BATCH, n_tr)
                idx = perm[s:e]
                logits = F_tr_z[idx] @ Wout + bout
                loss = F.cross_entropy(logits, ytr[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(float(loss.item()))
            ep_loss = float(np.mean(losses))
            train_loss_per_epoch.append(ep_loss)
            with torch.no_grad():
                tr_acc = float(((F_tr_z @ Wout + bout).argmax(dim=1) == ytr).float().mean())
                te_acc = float(((F_te_z @ Wout + bout).argmax(dim=1) == yte).float().mean())
            log(f"  ep{ep+1}/{EPOCHS}  loss={ep_loss:.4f}  train_acc={tr_acc:.4f}  test_acc={te_acc:.4f}")
        readout_time = time.time() - readout_t0

        with torch.no_grad():
            final_logits_te = F_te_z @ Wout + bout
            test_acc = float((final_logits_te.argmax(dim=1) == yte).float().mean())
            train_acc = float(((F_tr_z @ Wout + bout).argmax(dim=1) == ytr).float().mean())

        # --------------------------------------------------------------
        # Diagnostics: re-run last test sample with recording on
        # --------------------------------------------------------------
        log("[diagnostic] recording spikes / Vb for last test sample")
        last = Xte[-1:].contiguous()
        _, _, _, spikes_rec, vb_rec = nsram_featurize(
            last, W_in, VG1_bias, VG2_bias, surr,
            seed_poisson=424242, record=True,
        )
        spikes_np = spikes_rec.numpy().astype(np.float32)  # (T, N)
        vb_np    = vb_rec.numpy().astype(np.float32)       # (T, 32)

        # --------------------------------------------------------------
        # Energy from actual surrogate output: E_inf = sum_t Vd * |Id|_n * dt
        # per-neuron, accumulated over T_STEPS, averaged over test set.
        # --------------------------------------------------------------
        # mean_Id_te is mean |Id| per (image, neuron) over T steps.
        # Charge per neuron per inference = mean_Id * T_STEPS * DT_TRANS
        # Energy per neuron per inference = Vd * charge
        with torch.no_grad():
            mean_Id_per_neuron = mean_Id_te.mean(dim=0)              # (N,)  A
            charge_per_neuron = mean_Id_per_neuron * T_STEPS * DT_TRANS  # C
            energy_per_neuron_J = VD_FIXED * charge_per_neuron        # J
            energy_per_neuron_pJ = (energy_per_neuron_J * 1e12).cpu().numpy()
            energy_per_inf_pJ = float(energy_per_neuron_pJ.sum())

        # Throughput: total test images / test featurize wall-time
        throughput = float(Xte.shape[0] / dt_te)

        latency = {
            "featurize_per_image_ms_train": float(dt_tr / Xtr.shape[0] * 1e3),
            "featurize_per_image_ms_test":  float(dt_te / Xte.shape[0] * 1e3),
            "readout_train_total_s":        float(readout_time),
            "readout_per_image_us":         float(readout_time / (EPOCHS * Xtr.shape[0]) * 1e6),
        }

        # --------------------------------------------------------------
        # Save outputs
        # --------------------------------------------------------------
        np.save(OUT_DIR / "spikes.npy", spikes_np)
        np.save(OUT_DIR / "vb.npy", vb_np)
        np.save(OUT_DIR / "weights.npy", W_in.cpu().numpy())
        np.save(OUT_DIR / "energy.npy", energy_per_neuron_pJ)
        (OUT_DIR / "latency.json").write_text(json.dumps(latency, indent=2))
        pareto = {"name": "N_FF_MNIST_N512",
                  "accuracy": test_acc,
                  "energy_pj": energy_per_inf_pJ,
                  "throughput": throughput}
        (OUT_DIR / "pareto.json").write_text(json.dumps([pareto], indent=2))

        summary = {
            "test_accuracy": test_acc,
            "train_accuracy": train_acc,
            "train_loss_per_epoch": train_loss_per_epoch,
            "throughput_inf_per_sec": throughput,
            "energy_per_inf_pJ": energy_per_inf_pJ,
            "n_hidden": N_HIDDEN,
            "t_steps": T_STEPS,
            "epochs": EPOCHS,
            "batch": BATCH,
            "lr": LR,
            "surrogate_path": str(SURROGATE_PATH.relative_to(ROOT)),
            "n_train": int(Xtr.shape[0]),
            "n_test":  int(Xte.shape[0]),
            "feature_wall_s_train": dt_tr,
            "feature_wall_s_test":  dt_te,
            "readout_wall_s":       readout_time,
            "wall_s_total":         feat_time + readout_time,
            "device": str(device),
            "seed": SEED,
        }
        (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
        log(f"[summary] {json.dumps({k: summary[k] for k in ['test_accuracy','energy_per_inf_pJ','throughput_inf_per_sec']})}")

        # --------------------------------------------------------------
        # Dashboard
        # --------------------------------------------------------------
        sys.path.insert(0, str(ROOT / "scripts"))
        from network_viz import save_summary_dashboard
        dash = save_summary_dashboard(OUT_DIR)
        log(f"[dashboard] {dash}")

        # --------------------------------------------------------------
        # Gate verdicts + report
        # --------------------------------------------------------------
        infra = (OUT_DIR / "summary.json").exists() and (OUT_DIR / "dashboard.png").exists()
        discovery = test_acc > 0.92
        ambitious = test_acc > 0.96 and energy_per_inf_pJ < 100.0
        log(f"GATE INFRA={'PASS' if infra else 'FAIL'}  "
            f"DISCOVERY={'PASS' if discovery else 'FAIL'} (test_acc={test_acc:.4f})  "
            f"AMBITIOUS={'PASS' if ambitious else 'FAIL'} "
            f"(acc>{0.96} & energy<{100} pJ: acc={test_acc:.4f}, E={energy_per_inf_pJ:.2f})")

        report = []
        report.append(f"# N-FF-MNIST_N512 — Feedforward LIF SNN on MNIST 28x28\n")
        report.append(f"NS-RAM 4D MEP-1 surrogate ({SURROGATE_PATH.name}), 784 -> {N_HIDDEN} -> 10\n")
        report.append(f"T_STEPS={T_STEPS}  epochs={EPOCHS}  batch={BATCH}  lr={LR}  seed={SEED}\n")
        report.append("\n## Results\n")
        report.append(f"- test_accuracy: **{test_acc:.4f}**\n")
        report.append(f"- train_accuracy: {train_acc:.4f}\n")
        report.append(f"- train_loss_per_epoch: {train_loss_per_epoch}\n")
        report.append(f"- throughput_inf_per_sec: {throughput:.1f}\n")
        report.append(f"- energy_per_inf_pJ: {energy_per_inf_pJ:.2f}\n")
        report.append(f"- featurize wall (train/test): {dt_tr:.1f}s / {dt_te:.1f}s\n")
        report.append(f"- readout wall: {readout_time:.1f}s\n")
        report.append("\n## Gates\n")
        report.append(f"- INFRA: {'PASS' if infra else 'FAIL'}\n")
        report.append(f"- DISCOVERY (test_acc > 92%): {'PASS' if discovery else 'FAIL'}\n")
        report.append(f"- AMBITIOUS (>96% AND <100 pJ/inf): {'PASS' if ambitious else 'FAIL'}\n")
        report.append("\n## Latency\n")
        report.append("```\n" + json.dumps(latency, indent=2) + "\n```\n")
        (OUT_DIR / "report.md").write_text("".join(report))
        log(f"[report] {OUT_DIR / 'report.md'}")
        log("[done]")
    finally:
        fhandle.close()


if __name__ == "__main__":
    main()
