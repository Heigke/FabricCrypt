"""On-chip local-learning demo v2 — post-O23-oracle rebuild.

Key fixes over v1:
  - Per-cell ±0.1V VG2 sign mask (NOT global DC) → breaks symmetry
  - Goodness = mean(Id²) (NOT z-scored) → shift-sensitive
  - Drop spectral renormalisation; element-wise weight clip instead
  - N=128 cells on GPU (HSA_OVERRIDE_GFX_VERSION=11.0.0)
  - Three rules:
      (1) FF-fixed   — Hinton FF with per-cell mask, un-z-scored goodness
      (2) R-Hebbian  — 3-factor reward-modulated, EMA baseline + variance norm
      (3) FORCE-lite — 8 reservoir-cell features → linear delta-rule readout
  - N_TEST=32 (4× v1 → resolution 0.031, real CIs)
  - All tensors persistent on DEVICE (eliminate host↔device copies inside loop)

Output:
  results/demo_local_learning/v2_{rule}.json
  figures/local_learning/v2_comparison.png
"""
from __future__ import annotations
import importlib.util
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)
# CPU is 4× faster than ROCm for this workload (Python dispatch dominates,
# Newton-per-step has 15 iters × 4 _residuals calls each = 60 small kernels
# per step, each kernel sync kills GPU throughput). Benchmark: CPU 110 ms/step
# at N=1024; cuda 350 ms/step at any N.
DEVICE = torch.device("cpu")

ROOT = Path(__file__).resolve().parent.parent
sp = importlib.util.spec_from_file_location("v1", ROOT / "scripts/z96_narma10_pilot.py")
v1 = importlib.util.module_from_spec(sp); sp.loader.exec_module(v1)
sp2 = importlib.util.spec_from_file_location("z119", ROOT / "scripts/z119_topology_sweep.py")
z119 = importlib.util.module_from_spec(sp2); sp2.loader.exec_module(z119)
sp3 = importlib.util.spec_from_file_location("mg_mod", ROOT / "scripts/demo_mackey_glass.py")
mg_mod = importlib.util.module_from_spec(sp3); sp3.loader.exec_module(mg_mod)

from nsram.bsim4_port.nsram_cell_2T import NSRAMCell2TConfig
from nsram.bsim4_port.bjt import GummelPoonNPN
from nsram.bsim4_port.vectorized import forward_2t_batched

# ────────────────────────────────────────────────────────────────────
# Hyperparameters
N           = 1024            # BIG SLAM — Python overhead constant up to 1024 cells
T           = 100             # steps/sample
EPOCHS      = 15
N_TRAIN     = 20              # train samples per epoch
N_TEST      = 48              # test samples (24 per class) — stat power 0.072 res
KAPPA       = 0.20            # recurrent gain (oracle: 0.10–0.25 for N=128)
LABEL_AMP   = 0.10            # ±VG2 swing for label injection
W_MAX       = 0.6 / np.sqrt(N * 0.10)  # per-element clip per oracle
LR_FF       = 5e-3
LR_RH       = 3e-3
LR_FORCE    = 1e-2
EMA_TAU     = 32              # baseline EMA time constant (samples)
ETRACE_LAM  = 0.85            # eligibility trace decay (oracle τ_e=10–50 steps)
N_READOUT   = 32              # FORCE-lite linear combiner inputs (8→32 for big N)
SEED        = 0

OUT_RES = ROOT / "results/demo_local_learning"; OUT_RES.mkdir(parents=True, exist_ok=True)
OUT_FIG = ROOT / "figures/local_learning"; OUT_FIG.mkdir(parents=True, exist_ok=True)


# ────────────────────────────────────────────────────────────────────
# Signal generators
# ────────────────────────────────────────────────────────────────────
def gen_signal(class_id: int, T: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if class_id == 0:
        return mg_mod.gen_mackey_glass(T, seed=seed).astype(np.float64)
    f1, f2 = 0.05, 0.13
    phase = rng.normal(0, 0.3)
    x = 0.5 + 0.25*np.sin(2*np.pi*f1*np.arange(T)+phase) \
            + 0.15*np.sin(2*np.pi*f2*np.arange(T)+2*phase) \
            + rng.normal(0, 0.02, T)
    return np.clip(x, 0.0, 1.0).astype(np.float64)


# ────────────────────────────────────────────────────────────────────
# Reservoir on DEVICE — single sample
# ────────────────────────────────────────────────────────────────────
def run_reservoir(signal_t: torch.Tensor, class_sign: float, sign_mask: torch.Tensor,
                  W_rec: torch.Tensor, base_VG1: torch.Tensor, base_VG2: torch.Tensor,
                  M1, M2, bjt) -> torch.Tensor:
    """Return Id (linear, signed) shape (N, T) on DEVICE.

    class_sign ∈ {+1, -1}; sign_mask is per-cell ±1 (fixed).
    """
    cfg = NSRAMCell2TConfig(use_iii=True, use_gidl=True, use_bjt=True,
                            newton_max_iters=15)
    label_inj = LABEL_AMP * class_sign * sign_mask           # (N,)
    feat_prev = torch.zeros(N, device=DEVICE)
    Id_seq = torch.zeros(N, T, device=DEVICE)
    Tlen = signal_t.shape[0]
    for t in range(Tlen):
        Vd_t = (1.2 + 1.0 * signal_t[t]).unsqueeze(0)
        recur = W_rec @ feat_prev
        VG2_eff = (base_VG2 + label_inj + KAPPA * recur).clamp(-0.2, 1.0)
        out = forward_2t_batched(cfg, M1, M2, bjt, Vd_t, base_VG1, VG2_eff,
                                 max_iters=15, tol=1e-9, verbose=False)
        Id_t = out["Id"].squeeze(-1)                          # (N,)
        Id_seq[:, t] = Id_t
        # log-compress for recurrence (matches working MG pipeline)
        feat_prev = torch.log10(torch.clamp(Id_t.abs(), min=1e-15))
    return Id_seq


def goodness_mean_sq(Id_seq: torch.Tensor) -> torch.Tensor:
    """Σ_n Σ_t Id² / (N·T) — un-normalised, shift-sensitive scalar."""
    return (Id_seq * Id_seq).mean()


def per_cell_act(Id_seq: torch.Tensor) -> torch.Tensor:
    """Per-cell mean-square activity (N,)."""
    return (Id_seq * Id_seq).mean(dim=1)


# ────────────────────────────────────────────────────────────────────
# Common training scaffolding
# ────────────────────────────────────────────────────────────────────
def init_state(rng):
    M1, M2 = v1.build_calibrated_models()
    bjt = GummelPoonNPN.from_sebas_card()
    bjt.Bf = 2.0e4; bjt.Is = 1.0e-9
    base_VG1 = torch.tensor(rng.choice([0.2, 0.4, 0.6], size=N),
                            dtype=torch.float64, device=DEVICE)
    base_VG2 = torch.tensor(rng.uniform(0.0, 0.5, size=N),
                            dtype=torch.float64, device=DEVICE)
    sign_mask = torch.tensor(rng.choice([-1.0, 1.0], size=N),
                             dtype=torch.float64, device=DEVICE)
    W = z119.build_W("ER_SPARSE", N, rho=0.9, rng=rng)
    W = torch.tensor(W, dtype=torch.float64, device=DEVICE)
    return M1, M2, bjt, base_VG1, base_VG2, sign_mask, W


def eval_acc_ff(W, base_VG1, base_VG2, sign_mask, M1, M2, bjt, rng) -> float:
    """FF inference: pick class whose pos-pass goodness > neg-pass."""
    correct = 0
    for s in range(N_TEST):
        cls = s % 2
        sig = torch.tensor(gen_signal(cls, T, seed=10000 + s), device=DEVICE)
        g0 = goodness_mean_sq(run_reservoir(sig, +1.0, sign_mask, W,
                                            base_VG1, base_VG2, M1, M2, bjt))
        g1 = goodness_mean_sq(run_reservoir(sig, -1.0, sign_mask, W,
                                            base_VG1, base_VG2, M1, M2, bjt))
        pred = 0 if g0 > g1 else 1
        correct += int(pred == cls)
    return correct / N_TEST


def eval_acc_force(W, W_out, ro_idx, base_VG1, base_VG2, sign_mask,
                   M1, M2, bjt) -> float:
    """FORCE-lite inference: linear readout on top of 8 cells, no class injection (sign=0)."""
    correct = 0
    for s in range(N_TEST):
        cls = s % 2
        sig = torch.tensor(gen_signal(cls, T, seed=20000 + s), device=DEVICE)
        Id_seq = run_reservoir(sig, 0.0, sign_mask, W,
                               base_VG1, base_VG2, M1, M2, bjt)
        feat = torch.log10(torch.clamp(Id_seq[ro_idx].abs(), min=1e-15)).mean(dim=1)
        # Append bias
        feat_b = torch.cat([feat, torch.ones(1, device=DEVICE)])
        y = (W_out @ feat_b).item()
        pred = 0 if y > 0 else 1
        correct += int(pred == cls)
    return correct / N_TEST


# ────────────────────────────────────────────────────────────────────
# Rule 1: FF-fixed
# ────────────────────────────────────────────────────────────────────
def train_ff(rng, log_path):
    M1, M2, bjt, base_VG1, base_VG2, sign_mask, W = init_state(rng)
    history = []
    t0 = time.time()
    for epoch in range(EPOCHS):
        for s in range(N_TRAIN):
            cls = int(rng.integers(0, 2))
            sig = torch.tensor(gen_signal(cls, T, seed=epoch*1000+s), device=DEVICE)
            pos_sign = +1.0 if cls == 0 else -1.0
            neg_sign = -pos_sign
            id_pos = run_reservoir(sig, pos_sign, sign_mask, W,
                                   base_VG1, base_VG2, M1, M2, bjt)
            id_neg = run_reservoir(sig, neg_sign, sign_mask, W,
                                   base_VG1, base_VG2, M1, M2, bjt)
            a_pos = per_cell_act(id_pos)                       # (N,)
            a_neg = per_cell_act(id_neg)
            dW = LR_FF * (torch.outer(a_pos, a_pos) - torch.outer(a_neg, a_neg))
            dW.fill_diagonal_(0)
            W = (W + dW).clamp(-W_MAX, W_MAX)
        acc = eval_acc_ff(W, base_VG1, base_VG2, sign_mask, M1, M2, bjt, rng)
        history.append({"epoch": epoch+1, "acc": acc,
                        "Wnorm": float(W.norm().item()),
                        "elapsed": time.time()-t0})
        msg = (f"[ff] epoch {epoch+1:2d}/{EPOCHS}  acc={acc:.3f}  "
               f"|W|={W.norm().item():.4f}  elapsed={int(time.time()-t0)}s")
        print(msg, flush=True)
        log_path.write_text(json.dumps(history, indent=2))
    return {"rule": "ff", "history": history,
            "best_acc": max(h["acc"] for h in history),
            "final_acc": history[-1]["acc"],
            "wall_s": time.time()-t0}


# ────────────────────────────────────────────────────────────────────
# Rule 2: R-Hebbian (3-factor)
# ────────────────────────────────────────────────────────────────────
def train_rhebb(rng, log_path):
    M1, M2, bjt, base_VG1, base_VG2, sign_mask, W = init_state(rng)
    history = []
    G_ema = None; V_ema = 1.0
    alpha = 1.0 / EMA_TAU
    t0 = time.time()
    for epoch in range(EPOCHS):
        for s in range(N_TRAIN):
            cls = int(rng.integers(0, 2))
            sig = torch.tensor(gen_signal(cls, T, seed=epoch*1000+s), device=DEVICE)
            true_sign = +1.0 if cls == 0 else -1.0
            # Single forward with TRUE label: train W to make Id² high under correct injection
            Id_seq = run_reservoir(sig, true_sign, sign_mask, W,
                                   base_VG1, base_VG2, M1, M2, bjt)
            # log-compress feature for stable correlations
            z = torch.log10(torch.clamp(Id_seq.abs(), min=1e-15))
            mu = z.mean(dim=1, keepdim=True)
            zc = z - mu                                       # centered (N,T)
            # Eligibility trace built across time
            trace = torch.zeros(N, N, device=DEVICE)
            for t in range(T):
                trace = ETRACE_LAM * trace + torch.outer(zc[:, t], zc[:, t])
            G = float(goodness_mean_sq(Id_seq).item())
            if G_ema is None:
                G_ema = G; V_ema = 1.0
            else:
                G_ema = (1-alpha)*G_ema + alpha*G
                V_ema = (1-alpha)*V_ema + alpha*(G - G_ema)**2
            r = (G - G_ema) / (np.sqrt(V_ema) + 1e-12)
            r = float(np.clip(r, -3.0, 3.0))
            dW = LR_RH * r * trace
            dW.fill_diagonal_(0)
            W = (W + dW).clamp(-W_MAX, W_MAX)
        # Inference: use FF-style two-pass goodness comparison
        acc = eval_acc_ff(W, base_VG1, base_VG2, sign_mask, M1, M2, bjt, rng)
        history.append({"epoch": epoch+1, "acc": acc,
                        "Wnorm": float(W.norm().item()),
                        "G_ema": G_ema, "elapsed": time.time()-t0})
        msg = (f"[rhebb] epoch {epoch+1:2d}/{EPOCHS}  acc={acc:.3f}  "
               f"|W|={W.norm().item():.4f}  G_ema={G_ema:.3e}  "
               f"elapsed={int(time.time()-t0)}s")
        print(msg, flush=True)
        log_path.write_text(json.dumps(history, indent=2))
    return {"rule": "rhebb", "history": history,
            "best_acc": max(h["acc"] for h in history),
            "final_acc": history[-1]["acc"],
            "wall_s": time.time()-t0}


# ────────────────────────────────────────────────────────────────────
# Rule 3: FORCE-lite (delta-rule readout on 8 cells)
# ────────────────────────────────────────────────────────────────────
def train_force(rng, log_path):
    M1, M2, bjt, base_VG1, base_VG2, sign_mask, W = init_state(rng)
    # Pick 8 readout cells (random reservoir indices)
    ro_idx = torch.tensor(rng.choice(N, size=N_READOUT, replace=False),
                          dtype=torch.long, device=DEVICE)
    W_out = torch.zeros(N_READOUT + 1, device=DEVICE)        # +1 bias
    history = []
    t0 = time.time()
    for epoch in range(EPOCHS):
        for s in range(N_TRAIN):
            cls = int(rng.integers(0, 2))
            sig = torch.tensor(gen_signal(cls, T, seed=epoch*1000+s), device=DEVICE)
            target = +1.0 if cls == 0 else -1.0
            Id_seq = run_reservoir(sig, 0.0, sign_mask, W,
                                   base_VG1, base_VG2, M1, M2, bjt)
            feat = torch.log10(torch.clamp(Id_seq[ro_idx].abs(), min=1e-15)).mean(dim=1)
            feat_b = torch.cat([feat, torch.ones(1, device=DEVICE)])
            y = (W_out @ feat_b).item()
            err = target - y
            W_out = W_out + LR_FORCE * err * feat_b           # delta rule
        acc = eval_acc_force(W, W_out, ro_idx, base_VG1, base_VG2,
                             sign_mask, M1, M2, bjt)
        history.append({"epoch": epoch+1, "acc": acc,
                        "Wout_norm": float(W_out.norm().item()),
                        "elapsed": time.time()-t0})
        msg = (f"[force] epoch {epoch+1:2d}/{EPOCHS}  acc={acc:.3f}  "
               f"|Wout|={W_out.norm().item():.4f}  "
               f"elapsed={int(time.time()-t0)}s")
        print(msg, flush=True)
        log_path.write_text(json.dumps(history, indent=2))
    return {"rule": "force", "history": history,
            "best_acc": max(h["acc"] for h in history),
            "final_acc": history[-1]["acc"],
            "wall_s": time.time()-t0}


# ────────────────────────────────────────────────────────────────────
# Main + comparison plot
# ────────────────────────────────────────────────────────────────────
def main():
    print(f"[v2] device={DEVICE}  N={N}  T={T}  EPOCHS={EPOCHS}  "
          f"N_TRAIN={N_TRAIN}  N_TEST={N_TEST}", flush=True)
    rng = np.random.default_rng(SEED)
    summary = {"device": str(DEVICE), "N": N, "T": T,
               "epochs": EPOCHS, "n_train": N_TRAIN, "n_test": N_TEST,
               "kappa": KAPPA, "label_amp": LABEL_AMP, "w_max": float(W_MAX)}
    results = {}

    for rule_name, fn in [("ff", train_ff), ("rhebb", train_rhebb),
                           ("force", train_force)]:
        print(f"\n[v2] === {rule_name} ===", flush=True)
        log_path = OUT_RES / f"v2_{rule_name}_history.json"
        results[rule_name] = fn(np.random.default_rng(SEED + hash(rule_name) % 100),
                                 log_path)
        (OUT_RES / f"v2_{rule_name}.json").write_text(
            json.dumps(results[rule_name], indent=2))

    summary["results"] = {k: {"best": v["best_acc"], "final": v["final_acc"],
                              "wall_s": v["wall_s"]}
                          for k, v in results.items()}
    (OUT_RES / "v2_summary.json").write_text(json.dumps(summary, indent=2))
    print("\n[v2] summary:")
    print(json.dumps(summary["results"], indent=2))

    # Comparison plot
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = {"ff": "#3498db", "rhebb": "#27ae60", "force": "#e74c3c"}
    labels = {"ff": "Forward-Forward (no readout)",
              "rhebb": "Reward-modulated Hebbian (3-factor)",
              "force": "FORCE-lite (8-cell delta readout)"}
    for k, v in results.items():
        epochs = [h["epoch"] for h in v["history"]]
        accs = [h["acc"] for h in v["history"]]
        ax.plot(epochs, accs, "o-", color=colors[k], lw=1.6,
                label=f"{labels[k]} (best={v['best_acc']:.3f})")
    ax.axhline(0.5, ls="--", color="#888", lw=0.8, label="chance")
    ax.set_xlabel("epoch"); ax.set_ylabel("test accuracy")
    ax.set_ylim(0.3, 1.05)
    ax.set_title(f"NS-RAM local-learning v2 — N={N}, MG vs sin+phase, {N_TEST} test samples")
    ax.legend(fontsize=9, loc="lower right"); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_FIG / "v2_comparison.png", dpi=150)
    plt.savefig(OUT_FIG / "v2_comparison.pdf")
    plt.close()
    print(f"[v2] plot saved to {OUT_FIG / 'v2_comparison.png'}")


if __name__ == "__main__":
    main()
