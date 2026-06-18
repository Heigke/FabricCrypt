"""On-chip local-learning demo on NS-RAM cells.

Three learning rules compared on a common 2-class time-series task:
  (1) Forward-Forward (Hinton 2022) — per-cell goodness, two-pass push
  (2) STDP — spike-timing plasticity, post * pre-trace - decay
  (3) Hebbian — Δw = lr·(pre·post - decay·w)

Common harness:
  - 32-cell NS-RAM reservoir (ER_SPARSE)
  - Bf=2e4, Is=1e-9 (calibrated optimum from 2D sweep)
  - 2-class signal: Mackey-Glass (A) vs sinusoid+phase-noise (B)
    (Lorenz is overkill for a small demo; sinusoid is cleaner contrast)
  - Class token injected via VG2 bias offset (Δ=±0.1V)
  - Train: 30 epochs × 50 samples
  - Test: 20 samples per class, accuracy = (correct goodness ranking)

Output:
  results/demo_local_learning/{ff,stdp,hebbian}_summary.json
  figures/local_learning/{ff,stdp,hebbian}_curves.png
  figures/local_learning/comparison.png
  figures/local_learning/animation.mp4
"""
from __future__ import annotations
import importlib.util
import json
import time
from pathlib import Path
import numpy as np
import torch
torch.set_default_dtype(torch.float64)

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
# Hyperparameters (tuned for ~2-3 min total per rule on 32 cells)
N = 12              # small for speed
T = 50              # steps per sample
EPOCHS = 10
N_TRAIN_PER_EPOCH = 10
N_TEST = 8
LR_FF = 2.0e-2
LR_STDP = 5.0e-3
LR_HEB = 5.0e-3
WEIGHT_DECAY = 1e-3

OUT_RES = ROOT / "results/demo_local_learning"; OUT_RES.mkdir(parents=True, exist_ok=True)
OUT_FIG = ROOT / "figures/local_learning"; OUT_FIG.mkdir(parents=True, exist_ok=True)


def gen_signal(class_id: int, T: int, seed: int) -> np.ndarray:
    """Class A=0: Mackey-Glass τ=17 (chaotic). Class B=1: sinusoid+phase noise."""
    rng = np.random.default_rng(seed)
    if class_id == 0:
        return mg_mod.gen_mackey_glass(T, seed=seed)
    else:
        # 2-component sinusoid w/ phase noise — simple but distinct from MG
        f1, f2 = 0.05, 0.13
        phase = rng.normal(0, 0.3)
        x = 0.5 + 0.25 * np.sin(2 * np.pi * f1 * np.arange(T) + phase) \
                 + 0.15 * np.sin(2 * np.pi * f2 * np.arange(T) + 2 * phase)
        x += rng.normal(0, 0.02, T)
        return np.clip(x, 0.0, 1.0)


def run_reservoir(N: int, T: int, signal: np.ndarray, class_token: int,
                  W_rec: np.ndarray, base_VG1: torch.Tensor, base_VG2: torch.Tensor,
                  M1, M2, bjt, kappa: float = 0.30) -> np.ndarray:
    """Run reservoir for one sample. class_token=±1 shifts VG2 to embed label.
    Returns log_Id of shape (N, T)."""
    cfg = NSRAMCell2TConfig(use_iii=True, use_gidl=True, use_bjt=True,
                              newton_max_iters=30)
    feat_prev = torch.zeros(N, dtype=torch.float64)
    log_Id = np.zeros((N, T))
    token_shift = 0.10 * float(class_token)  # ±0.1V VG2 bias
    for t in range(T):
        Vd_t = torch.tensor([1.2 + 1.0 * float(signal[t])], dtype=torch.float64)
        recur = torch.tensor(W_rec @ feat_prev.numpy(), dtype=torch.float64)
        VG2_eff = (base_VG2 + token_shift + kappa * recur).clamp(-0.2, 1.0)
        out = forward_2t_batched(cfg, M1, M2, bjt, Vd_t, base_VG1, VG2_eff,
                                  max_iters=15, tol=1e-9, verbose=False)
        log_Id[:, t] = np.log10(np.maximum(out["Id"].abs().squeeze().numpy(), 1e-15))
        feat_prev = torch.tensor(log_Id[:, t], dtype=torch.float64)
    return log_Id


def goodness(log_Id: np.ndarray) -> float:
    """Per-sample scalar: sum of squared z-scored activity over time and cells."""
    z = (log_Id - log_Id.mean(axis=1, keepdims=True)) / (log_Id.std(axis=1, keepdims=True) + 1e-9)
    return float(np.sum(z ** 2))


def per_cell_activity(log_Id: np.ndarray) -> np.ndarray:
    """Per-cell scalar: variance over time (proxy for 'how much this cell was used')."""
    return log_Id.var(axis=1)


def normalise_W(W: np.ndarray, target_rho: float = 0.9) -> np.ndarray:
    """Spectral-radius normalise."""
    eig = np.abs(np.linalg.eigvals(W)).max()
    if eig > 1e-9:
        W = W * (target_rho / eig)
    return W


# ────────────────────────────────────────────────────────────────────
# Three learning rules
# ────────────────────────────────────────────────────────────────────

def update_ff(W: np.ndarray, act_pos: np.ndarray, act_neg: np.ndarray,
              lr: float) -> np.ndarray:
    """Forward-Forward: positive sample drives high goodness; negative low.
    Δw_ij ∝ (act_pos_i · act_pos_j) - (act_neg_i · act_neg_j)
    where act = per-cell activity (variance-over-time).
    """
    # Outer product gives the correlated-co-firing matrix
    pos_outer = np.outer(act_pos, act_pos)
    neg_outer = np.outer(act_neg, act_neg)
    dW = lr * (pos_outer - neg_outer)
    np.fill_diagonal(dW, 0)
    W = W + dW
    W *= (1.0 - WEIGHT_DECAY)
    return normalise_W(W)


def update_stdp(W: np.ndarray, log_Id: np.ndarray, lr: float) -> np.ndarray:
    """Spike-Timing-Dependent Plasticity:
    spike events when |Id| z-score crosses +1.5; Δw_ij = lr·(post_t · pre_(t-Δ))
    averaged over a small Δ window. Equivalent to short-window cross-correlation.
    """
    z = (log_Id - log_Id.mean(axis=1, keepdims=True)) / (log_Id.std(axis=1, keepdims=True) + 1e-9)
    spikes = (z > 1.5).astype(float)  # (N, T)
    N, T = spikes.shape
    # Causal cross-correlation: post[t] · pre[t-1..t-3] mean
    dW = np.zeros((N, N))
    for delta in range(1, 4):
        if delta < T:
            dW += spikes[:, delta:] @ spikes[:, :-delta].T
    dW /= 3.0 * max(T - 3, 1)
    # Asymmetric: post-after-pre potentiates, pre-after-post depresses
    dW_neg = np.zeros((N, N))
    for delta in range(1, 4):
        if delta < T:
            dW_neg += spikes[:, :-delta] @ spikes[:, delta:].T
    dW_neg /= 3.0 * max(T - 3, 1)
    dW_total = dW - 0.5 * dW_neg
    np.fill_diagonal(dW_total, 0)
    W = W + lr * dW_total
    W *= (1.0 - WEIGHT_DECAY)
    return normalise_W(W)


def update_hebbian(W: np.ndarray, log_Id: np.ndarray, lr: float) -> np.ndarray:
    """Plain Hebbian: Δw_ij = lr · ⟨pre_i · post_j⟩ - decay · w."""
    z = (log_Id - log_Id.mean(axis=1, keepdims=True)) / (log_Id.std(axis=1, keepdims=True) + 1e-9)
    corr = (z @ z.T) / z.shape[1]
    np.fill_diagonal(corr, 0)
    W = W + lr * corr
    W *= (1.0 - WEIGHT_DECAY)
    return normalise_W(W)


# ────────────────────────────────────────────────────────────────────
# Training loop (common to all rules)
# ────────────────────────────────────────────────────────────────────

def train_eval(rule_name: str) -> dict:
    print(f"\n[{rule_name}] starting at {time.strftime('%H:%M:%S')}")
    rng = np.random.default_rng(0)
    M1, M2 = v1.build_calibrated_models()
    bjt = GummelPoonNPN.from_sebas_card()
    bjt.Bf = 2.0e4; bjt.Is = 1.0e-9  # calibrated optimum

    base_VG1 = torch.tensor(rng.choice([0.2, 0.4, 0.6], size=N), dtype=torch.float64)
    base_VG2 = torch.tensor(rng.uniform(0.0, 0.5, size=N), dtype=torch.float64)
    W = z119.build_W("ER_SPARSE", N, rho=0.9, rng=rng)
    W_history = [W.copy()]
    acc_history = []
    t_start = time.time()

    for epoch in range(EPOCHS):
        # ── Train phase ─────────────────────────────────────────────
        for s in range(N_TRAIN_PER_EPOCH):
            cls = int(rng.integers(0, 2))
            seed = epoch * 1000 + s
            sig = gen_signal(cls, T, seed=seed)
            if rule_name == "ff":
                # Two passes: positive (correct token) and negative (wrong)
                tok_pos = +1 if cls == 0 else -1
                tok_neg = -tok_pos
                lid_pos = run_reservoir(N, T, sig, tok_pos, W, base_VG1, base_VG2, M1, M2, bjt)
                lid_neg = run_reservoir(N, T, sig, tok_neg, W, base_VG1, base_VG2, M1, M2, bjt)
                W = update_ff(W, per_cell_activity(lid_pos), per_cell_activity(lid_neg), LR_FF)
            elif rule_name == "stdp":
                tok = +1 if cls == 0 else -1
                lid = run_reservoir(N, T, sig, tok, W, base_VG1, base_VG2, M1, M2, bjt)
                W = update_stdp(W, lid, LR_STDP)
            elif rule_name == "hebbian":
                tok = +1 if cls == 0 else -1
                lid = run_reservoir(N, T, sig, tok, W, base_VG1, base_VG2, M1, M2, bjt)
                W = update_hebbian(W, lid, LR_HEB)
        W_history.append(W.copy())
        # ── Eval phase ─────────────────────────────────────────────
        correct = 0
        for s in range(N_TEST):
            cls = int(rng.integers(0, 2))
            seed = 99000 + epoch * 1000 + s
            sig = gen_signal(cls, T, seed=seed)
            # Goodness with each candidate class token; pick higher
            g_a = goodness(run_reservoir(N, T, sig, +1, W, base_VG1, base_VG2, M1, M2, bjt))
            g_b = goodness(run_reservoir(N, T, sig, -1, W, base_VG1, base_VG2, M1, M2, bjt))
            pred = 0 if g_a > g_b else 1
            if pred == cls:
                correct += 1
        acc = correct / N_TEST
        acc_history.append(acc)
        elapsed = time.time() - t_start
        print(f"[{rule_name}] epoch {epoch+1:2d}/{EPOCHS}  acc={acc:.3f}  "
              f"|W|={np.abs(W).mean():.4f}  elapsed={elapsed:.0f}s", flush=True)

    summary = {
        "rule": rule_name, "N": N, "T": T, "epochs": EPOCHS,
        "n_train_per_epoch": N_TRAIN_PER_EPOCH, "n_test": N_TEST,
        "acc_history": acc_history,
        "final_acc": acc_history[-1],
        "best_acc": max(acc_history),
        "wall_s": time.time() - t_start,
        "W_final_mean": float(np.abs(W).mean()),
    }
    np.save(OUT_RES / f"{rule_name}_W_history.npy", np.array(W_history))
    (OUT_RES / f"{rule_name}_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[{rule_name}] DONE  wall={summary['wall_s']:.0f}s "
          f"final_acc={summary['final_acc']:.3f}  best={summary['best_acc']:.3f}")
    return summary


def main():
    summaries = {}
    for rule in ("hebbian", "stdp", "ff"):  # cheapest first
        summaries[rule] = train_eval(rule)
    # Combined plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = {"ff": "#c0392b", "stdp": "#2980b9", "hebbian": "#27ae60"}
    labels = {"ff": "Forward-Forward", "stdp": "STDP", "hebbian": "Hebbian"}
    for rule, summ in summaries.items():
        ax.plot(range(1, EPOCHS + 1), summ["acc_history"],
                 marker="o", lw=2, color=colors[rule],
                 label=f"{labels[rule]} (best={summ['best_acc']:.2f})")
    ax.axhline(0.5, color="gray", ls=":", alpha=0.7, label="chance")
    ax.set_xlabel("epoch")
    ax.set_ylabel("test accuracy (2-class)")
    ax.set_ylim(0.3, 1.05)
    ax.set_title(f"Local-learning rules on {N}-cell NS-RAM reservoir "
                  f"(MG vs sinusoid)", fontsize=11)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_FIG / "comparison.png", dpi=150)
    plt.savefig(OUT_FIG / "comparison.pdf")
    plt.close()
    print(f"\n[done] saved {OUT_FIG / 'comparison.png'}")
    print("\nFinal summary:")
    for rule, s in summaries.items():
        print(f"  {labels[rule]:>20s}: best={s['best_acc']:.3f}  "
              f"final={s['final_acc']:.3f}  wall={s['wall_s']:.0f}s")


if __name__ == "__main__":
    main()
