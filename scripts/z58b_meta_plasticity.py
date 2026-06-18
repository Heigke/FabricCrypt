"""z58b_meta_plasticity.py — proof-of-concept: differentiable per-cell
VG2 trained as a meta-plasticity knob.

Setup: small recurrent network of N=24 cells, fixed sparse W. Each cell
has its own learnable VG2 ∈ [-0.4, +0.4]. Train VG2 to maximize
performance on a multi-task objective:

  T1: pattern recall — init Vb to a noisy version of one of K stored
      patterns, want Vb at end to be close to the clean pattern
  T2: reservoir delayed-recall — feed random input, predict u[t-2]
      from current Vb via fixed-output ridge regression (computed once
      per minibatch using the current Vb history)

Loss = α·T1_mse + β·T2_mse.

Compare against a fixed uniform VG2 = -0.10 baseline.

z58a found cell_fast spans bistable + soft regimes via VG2 but NOT
neuron/integrator (Vth_eff goes too high at +VG2). Expected: the
training will converge in the bistable<->soft range and produce a
meaningful loss reduction, but won't show the full 3-mode allocation
that Mario's Nature paper describes — which is exactly the argument
for switching to Robert's emulator.
"""
from __future__ import annotations
import json, time
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from nsram.cell_fast import CellArray
from nsram.plasticity_net import topo_random

OUT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/"
           "results/z58b_meta_plasticity")
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[device] {DEVICE}")

N = 24
K_PATTERNS = 4
T_RECALL = 15
T_RESERVOIR = 60
N_ITERS = 150
LR = 0.05
P_LO, P_HI = -0.10, 0.40
ALPHA_LOSS = 1.0    # weight for recall task
BETA_LOSS = 0.5     # weight for reservoir task

CAL = dict(VTH0=0.43, K_back=-0.98, A_iii=4.71, G_bjt=1.00,
           V_bjt_on=0.74, V_latch=0.58, K_leak=0.021)


def make_cells_with_vg2(vg2_param):
    """CellArray with VG2 wired as an nn.Parameter (gradient flows back)."""
    # Initialize with a placeholder VG2 then replace.
    placeholder = vg2_param.detach()
    cells = CellArray(N, alpha=1.5, VG2=placeholder, device=DEVICE, **CAL)
    cells.VG2 = vg2_param          # autograd-tracked
    cells.Vb = vg2_param.detach().clone()
    return cells


def hopfield_W_torch(patterns):
    """patterns: (K, N) ±1 tensor."""
    W = (patterns.T @ patterns) / N
    W = W - torch.diag(torch.diag(W))
    return W


def run_recall(cells, W, target_pattern, noisy_pattern, fb=0.5):
    """Init Vb to noisy_pattern (mapped to ±0.4/-0.1), run T_RECALL no-input."""
    Vb_init = torch.where(noisy_pattern > 0,
                          torch.tensor(P_HI, device=DEVICE),
                          torch.tensor(P_LO, device=DEVICE))
    cells.Vb = Vb_init
    VG1 = torch.tensor(0.6, device=DEVICE)
    for _ in range(T_RECALL):
        recurrent = W @ cells.Vb
        drive = fb * recurrent
        cells.step(VG1, drive)
    Vb_final = cells.Vb
    target_v = torch.where(target_pattern > 0,
                           torch.tensor(P_HI, device=DEVICE),
                           torch.tensor(P_LO, device=DEVICE))
    return ((Vb_final - target_v) ** 2).mean()


def run_reservoir(cells, W, u_seq, fb=0.5):
    """Feed random input, collect Vb history, ridge-predict u[t-2]."""
    cells.Vb = cells.VG2.detach().clone()
    States = []
    VG1 = torch.tensor(0.6, device=DEVICE)
    for t in range(T_RESERVOIR):
        recurrent = W @ cells.Vb
        drive = fb * recurrent + 0.5 * u_seq[t]
        cells.step(VG1, drive)
        States.append(cells.Vb)
    States = torch.stack(States, dim=0)        # (T, N)
    # target = u_seq[t-2]
    target = torch.zeros(T_RESERVOIR, device=DEVICE)
    target[2:] = u_seq[:-2]
    # closed-form ridge from first half, eval on second
    half = T_RESERVOIR // 2
    Xtr = States[5:half]; ytr = target[5:half]
    Xte = States[half:];  yte = target[half:]
    XtX = Xtr.T @ Xtr + 1e-2 * torch.eye(N, device=DEVICE)
    w = torch.linalg.solve(XtX, Xtr.T @ ytr)
    pred = Xte @ w
    return ((pred - yte) ** 2).mean()


def evaluate(vg2_param, W, patterns, seed):
    cells = make_cells_with_vg2(vg2_param)
    rng = np.random.default_rng(seed)
    # Recall: average MSE over patterns with random noise
    recall_loss = 0.0
    for k in range(K_PATTERNS):
        p = patterns[k]
        noise_mask = torch.tensor(rng.choice([-1.0, 1.0],
                                              size=N,
                                              p=[0.20, 0.80]),
                                   dtype=torch.float32, device=DEVICE)
        noisy = p * noise_mask
        recall_loss = recall_loss + run_recall(cells, W, p, noisy)
    recall_loss = recall_loss / K_PATTERNS
    # Reservoir: random input
    u_seq = torch.tensor(rng.choice([-1.0, 1.0], size=T_RESERVOIR),
                          dtype=torch.float32, device=DEVICE)
    res_loss = run_reservoir(cells, W, u_seq)
    return recall_loss, res_loss


def main():
    t0 = time.time()
    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    patterns_np = rng.choice([-1.0, 1.0], size=(K_PATTERNS, N)).astype(np.float32)
    patterns = torch.from_numpy(patterns_np).to(DEVICE)
    W = hopfield_W_torch(patterns)

    # Random sparse perturbation (so it's not pure Hopfield)
    g = torch.Generator(device=DEVICE).manual_seed(0)
    W_pert = topo_random(N, p=0.1, seed=0, device=DEVICE) * 0.3
    W = (W + W_pert).detach()

    # ── (A) Baseline: uniform VG2 = -0.10 ──────────────────────────────
    print("\n── BASELINE (uniform VG2=-0.10) ──")
    vg2_baseline = nn.Parameter(torch.full((N,), -0.10, device=DEVICE))
    with torch.no_grad():
        rl, sl = evaluate(vg2_baseline, W, patterns, seed=42)
        baseline_total = ALPHA_LOSS * rl + BETA_LOSS * sl
    print(f"  recall_mse={rl.item():.4f}  reservoir_mse={sl.item():.4f}  "
          f"total={baseline_total.item():.4f}")

    # ── (B) Learned per-cell VG2 ───────────────────────────────────────
    print("\n── TRAINED (per-cell VG2 learnable) ──")
    vg2_learn = nn.Parameter(torch.full((N,), -0.10, device=DEVICE) +
                              0.05 * torch.randn(N, device=DEVICE))
    vg2_init = vg2_learn.detach().cpu().numpy().copy()
    opt = torch.optim.Adam([vg2_learn], lr=LR)

    history = {"iter": [], "recall": [], "reservoir": [], "total": []}
    for it in range(N_ITERS):
        opt.zero_grad()
        # clamp VG2 to physical range ([-0.4, 0.4])
        rl, sl = evaluate(vg2_learn, W, patterns, seed=42 + it)
        loss = ALPHA_LOSS * rl + BETA_LOSS * sl
        loss.backward()
        opt.step()
        with torch.no_grad():
            vg2_learn.clamp_(-0.40, 0.40)
        if it % 10 == 0 or it == N_ITERS - 1:
            history["iter"].append(it)
            history["recall"].append(rl.item())
            history["reservoir"].append(sl.item())
            history["total"].append(loss.item())
            print(f"  it={it:3d}  recall={rl.item():.4f}  res={sl.item():.4f}  "
                  f"total={loss.item():.4f}  "
                  f"VG2[mean,std]=[{vg2_learn.mean().item():+.3f},{vg2_learn.std().item():.3f}]",
                  flush=True)

    vg2_final = vg2_learn.detach().cpu().numpy()

    # Final eval (no grad, fresh seed)
    with torch.no_grad():
        rl, sl = evaluate(vg2_learn, W, patterns, seed=999)
        learned_total = ALPHA_LOSS * rl + BETA_LOSS * sl
    print(f"\nFinal eval (unseen seed):  recall={rl.item():.4f}  "
          f"reservoir={sl.item():.4f}  total={learned_total.item():.4f}")
    print(f"Baseline total:  {baseline_total.item():.4f}")
    improvement = (1 - learned_total.item() / baseline_total.item()) * 100
    print(f"\nImprovement: {improvement:+.1f}%")

    elapsed = time.time() - t0
    summary = {
        "baseline_total": float(baseline_total),
        "learned_total": float(learned_total),
        "improvement_pct": float(improvement),
        "vg2_init": vg2_init.tolist(),
        "vg2_final": vg2_final.tolist(),
        "vg2_final_mean": float(vg2_final.mean()),
        "vg2_final_std": float(vg2_final.std()),
        "history": history,
        "elapsed_s": elapsed,
        "config": {"N": N, "K": K_PATTERNS, "T_recall": T_RECALL,
                    "T_reservoir": T_RESERVOIR, "n_iters": N_ITERS,
                    "lr": LR, "alpha": ALPHA_LOSS, "beta": BETA_LOSS}
    }
    with open(OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ── plots ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.plot(history["iter"], history["total"], "o-", lw=2,
            label="trained total", color="#3498db")
    ax.plot(history["iter"], history["recall"], "s-", lw=1.5, alpha=0.7,
            label="recall MSE", color="#e67e22")
    ax.plot(history["iter"], history["reservoir"], "^-", lw=1.5, alpha=0.7,
            label="reservoir MSE", color="#27ae60")
    ax.axhline(baseline_total.item(), color="black", ls="--", lw=2,
               label=f"baseline (uniform VG2)={baseline_total.item():.3f}")
    ax.set_xlabel("training iteration"); ax.set_ylabel("loss")
    ax.set_title("Training curve")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.hist(vg2_init, bins=15, alpha=0.6, color="#95a5a6", label="initial")
    ax.hist(vg2_final, bins=15, alpha=0.6, color="#3498db", label="learned")
    ax.axvline(-0.10, color="red", ls=":", label="uniform baseline (-0.10)")
    ax.set_xlabel("VG2 [V]"); ax.set_ylabel("# cells")
    ax.set_title(f"Per-cell VG2 distribution\nlearned: {vg2_final.mean():+.3f} ± {vg2_final.std():.3f}")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    sorted_idx = np.argsort(vg2_final)
    ax.plot(np.arange(N), vg2_final[sorted_idx], "o-", color="#3498db",
            label="learned (sorted)")
    ax.plot(np.arange(N), vg2_init[sorted_idx], "s--", color="#95a5a6",
            alpha=0.5, label="initial (same order)")
    ax.axhline(-0.10, color="red", ls=":", label="uniform baseline")
    ax.set_xlabel("cell index (sorted by learned VG2)")
    ax.set_ylabel("VG2 [V]")
    ax.set_title("Per-cell allocation after training")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    bars = ax.bar(["baseline\n(uniform)", "learned\n(per-cell)"],
                   [baseline_total.item(), learned_total.item()],
                   color=["#95a5a6", "#3498db"])
    for b, v in zip(bars, [baseline_total.item(), learned_total.item()]):
        ax.text(b.get_x() + b.get_width()/2, v, f"{v:.3f}",
                ha="center", va="bottom", fontsize=12, weight="bold")
    ax.set_ylabel("multi-task loss (lower is better)")
    ax.set_title(f"Improvement: {improvement:+.1f}%")
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle(f"z58b — Differentiable per-cell VG2 meta-plasticity  "
                 f"({elapsed:.0f}s, N={N}, {N_ITERS} iters)",
                 fontsize=13, weight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "meta_plasticity.png", dpi=140)
    plt.close(fig)
    print(f"\nWrote {OUT/'meta_plasticity.png'}")


if __name__ == "__main__":
    main()
