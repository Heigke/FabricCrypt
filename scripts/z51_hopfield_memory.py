"""z51_hopfield_memory.py — Hopfield-style associative memory test.

Hebbian (z48) and BCM (z49) failed because they pushed the network
toward edge of chaos, which is wrong for bistable cells.  But there's
ONE plasticity rule that's specifically designed for bistable
substrates: the classical Hopfield rule.  Each stored pattern p_k is
imprinted as

    W += (1/N) · p_k · p_k^T  (diagonal zeroed)

Once K patterns are stored, the bistable network should have K
attractor basins.  Presented with a noisy version of pattern m, it
should converge to attractor m.

Three experiments:
  Q1: capacity sweep — store K patterns, measure recall accuracy
  Q2: noise tolerance — fixed K, sweep input noise level
  Q3: bistable vs linear-cell baseline (does bistability still help?)

Pattern format: p ∈ {-0.10, +0.40}^N.  These map to cell_fast's two
stable states (low Vb when VG2 = -0.10, latched when biased above
V_latch = 0.58 by drive). Recall = correlation of final Vb with stored
patterns; max correlation pattern = "recalled".
"""
from __future__ import annotations
import json, time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from nsram.cell_fast import CellArray
from nsram.plasticity_net import NetSim

OUT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/"
           "results/z51_hopfield_memory")
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[device] {DEVICE}")
N_CELLS = 96
N_SEEDS = 5

CAL = dict(VTH0=0.43, K_back=-0.98, A_iii=4.71, G_bjt=1.00,
            V_bjt_on=0.74, V_latch=0.58, K_leak=0.021)
P_LO, P_HI = -0.10, 0.40    # binary pattern values


def hopfield_W(patterns: np.ndarray):
    """W_ij = (1/N) * sum_k p_k_i * p_k_j, diagonal zero.
    patterns: (K, N) ∈ {-1, +1}."""
    K, N = patterns.shape
    W = (patterns.T @ patterns) / N
    np.fill_diagonal(W, 0.0)
    return W


def make_net(W_np, alpha=1.5, fb=0.5, seed=0, linear_cells=False):
    if linear_cells:
        from scripts.z44_robustness import LinearCellArray
        VG2 = torch.full((N_CELLS,), -0.10, device=DEVICE)
        cells = LinearCellArray(N_CELLS, alpha=alpha, VG2=VG2, device=DEVICE)
    else:
        VG2 = torch.full((N_CELLS,), -0.10, device=DEVICE)
        cells = CellArray(N_CELLS, alpha=alpha, VG2=VG2,
                              **CAL, device=DEVICE)
    W = torch.from_numpy(W_np.astype(np.float32)).to(DEVICE)
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    W_in = torch.zeros(N_CELLS, 1, device=DEVICE)   # no external input
    return NetSim(cells=cells, W=W, W_in=W_in, feedback_gain=fb)


def make_patterns(K, N, seed):
    rng = np.random.default_rng(seed)
    return rng.choice([-1, 1], size=(K, N)).astype(np.float32)


def pattern_to_Vb(p):
    """Map ±1 pattern to (V_LO, V_HI)."""
    return np.where(p > 0, P_HI, P_LO).astype(np.float32)


def add_noise(p, noise_frac, rng):
    """Flip noise_frac of bits randomly."""
    p_n = p.copy()
    n_flip = int(noise_frac * len(p))
    if n_flip > 0:
        idx = rng.choice(len(p), n_flip, replace=False)
        p_n[idx] *= -1
    return p_n


def recall_test(net, pattern_init: np.ndarray, T_relax: int = 50):
    """Initialize Vb to pattern_init, run T_relax steps with no input,
    return final Vb."""
    Vb_init = torch.from_numpy(pattern_to_Vb(pattern_init)).to(DEVICE)
    net.cells.Vb = Vb_init.clone()
    u = torch.zeros(1, device=DEVICE)
    VG1 = torch.tensor(0.6, device=DEVICE)
    for _ in range(T_relax):
        net.step(u, VG1)
    return net.cells.Vb.cpu().numpy()


def closest_pattern(final_Vb, patterns):
    """Return index of pattern with highest correlation."""
    # Map final_Vb back to ±1 estimate (above midpoint = +1)
    midpoint = 0.5 * (P_LO + P_HI)
    final_sign = np.where(final_Vb > midpoint, 1, -1).astype(np.float32)
    overlaps = patterns @ final_sign / len(final_sign)
    return int(np.argmax(overlaps)), float(overlaps.max())


def run_q1():
    """Capacity sweep: K patterns, recall accuracy."""
    print("\n=== Q1: capacity sweep ===")
    Ks = [1, 2, 4, 6, 8, 12, 16, 20, 25, 30]
    rs = {"K": [], "accuracy": [], "accuracy_std": []}
    for K in Ks:
        accs = []
        for s in range(N_SEEDS):
            patterns = make_patterns(K, N_CELLS, seed=s)
            W = hopfield_W(patterns)
            net = make_net(W, seed=s)
            correct = 0
            for k in range(K):
                final = recall_test(net, patterns[k])
                guess, _ = closest_pattern(final, patterns)
                if guess == k: correct += 1
            accs.append(correct / K)
        rs["K"].append(K)
        rs["accuracy"].append(float(np.mean(accs)))
        rs["accuracy_std"].append(float(np.std(accs)))
        print(f"  K={K:3d}  accuracy={rs['accuracy'][-1]:.2f}±{rs['accuracy_std'][-1]:.2f}",
              flush=True)
    return rs


def run_q2():
    """Noise tolerance: fixed K=8, sweep noise level on input."""
    print("\n=== Q2: noise tolerance (K=8) ===")
    K = 8
    noise_levels = np.linspace(0.0, 0.4, 9)
    rs = {"noise": [], "accuracy": [], "accuracy_std": []}
    for noise in noise_levels:
        accs = []
        for s in range(N_SEEDS):
            patterns = make_patterns(K, N_CELLS, seed=s)
            W = hopfield_W(patterns)
            net = make_net(W, seed=s)
            rng = np.random.default_rng(2000 + s)
            correct = 0
            for k in range(K):
                noisy = add_noise(patterns[k], noise, rng)
                final = recall_test(net, noisy)
                guess, _ = closest_pattern(final, patterns)
                if guess == k: correct += 1
            accs.append(correct / K)
        rs["noise"].append(float(noise))
        rs["accuracy"].append(float(np.mean(accs)))
        rs["accuracy_std"].append(float(np.std(accs)))
        print(f"  noise={noise:.2f}  acc={rs['accuracy'][-1]:.2f}±{rs['accuracy_std'][-1]:.2f}",
              flush=True)
    return rs


def run_q3():
    """Bistable vs linear cells on K=8 recall, noise=0.2."""
    print("\n=== Q3: bistable vs linear-cell on Hopfield task (K=8, noise=0.2) ===")
    K = 8
    noise = 0.2
    rs = {"bistable": [], "linear": []}
    for s in range(N_SEEDS):
        patterns = make_patterns(K, N_CELLS, seed=s)
        W = hopfield_W(patterns)
        rng = np.random.default_rng(3000 + s)

        # Bistable
        net = make_net(W, seed=s, linear_cells=False)
        correct_b = 0
        for k in range(K):
            noisy = add_noise(patterns[k], noise, rng)
            final = recall_test(net, noisy)
            g, _ = closest_pattern(final, patterns)
            if g == k: correct_b += 1
        rs["bistable"].append(correct_b / K)

        # Linear
        net = make_net(W, seed=s, linear_cells=True)
        rng = np.random.default_rng(3000 + s)   # same noise
        correct_l = 0
        for k in range(K):
            noisy = add_noise(patterns[k], noise, rng)
            final = recall_test(net, noisy)
            g, _ = closest_pattern(final, patterns)
            if g == k: correct_l += 1
        rs["linear"].append(correct_l / K)

        print(f"  seed={s}  bistable={correct_b}/{K}  linear={correct_l}/{K}",
              flush=True)
    return {
        "bistable_mean": float(np.mean(rs["bistable"])),
        "bistable_std": float(np.std(rs["bistable"])),
        "linear_mean": float(np.mean(rs["linear"])),
        "linear_std": float(np.std(rs["linear"])),
    }


def main():
    t0 = time.time()
    q1 = run_q1()
    q2 = run_q2()
    q3 = run_q3()
    elapsed = time.time() - t0

    with open(OUT / "summary.json", "w") as f:
        json.dump({"Q1": q1, "Q2": q2, "Q3": q3, "elapsed_s": elapsed},
                    f, indent=2)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Q1
    ax = axes[0]
    r = q1
    ax.errorbar(r["K"], r["accuracy"], yerr=r["accuracy_std"],
                  fmt="o-", color="#3498db", lw=2, capsize=4)
    ax.axhline(1.0, color="green", ls="--", alpha=0.5, label="perfect")
    ax.axhline(1.0 / np.array(r["K"]).max(), color="red", ls="--", alpha=0.5,
                 label="random chance ~1/K")
    ax.set_xlabel("# stored patterns K"); ax.set_ylabel("recall accuracy")
    ax.set_title(f"Q1: capacity sweep — N={N_CELLS} cells\n"
                  f"Hopfield theory: K_max ≈ 0.14·N = {0.14*N_CELLS:.0f}")
    ax.legend(); ax.grid(alpha=0.3)

    # Q2
    ax = axes[1]
    r = q2
    ax.errorbar([n*100 for n in r["noise"]], r["accuracy"],
                  yerr=r["accuracy_std"], fmt="o-", color="#e67e22",
                  lw=2, capsize=4)
    ax.axhline(1.0, color="green", ls="--", alpha=0.5)
    ax.set_xlabel("input noise [%]"); ax.set_ylabel("recall accuracy")
    ax.set_title("Q2: noise tolerance (K=8)")
    ax.grid(alpha=0.3)

    # Q3
    ax = axes[2]
    bars = ax.bar(["bistable", "linear"],
                    [q3["bistable_mean"], q3["linear_mean"]],
                    yerr=[q3["bistable_std"], q3["linear_std"]],
                    color=["#3498db", "#95a5a6"], capsize=8)
    for b, v in zip(bars, [q3["bistable_mean"], q3["linear_mean"]]):
        ax.text(b.get_x() + b.get_width()/2, v, f"{v:.2f}",
                  ha="center", va="bottom", fontsize=11, weight="bold")
    ax.axhline(1.0, color="green", ls="--", alpha=0.5)
    ax.set_ylabel("recall accuracy")
    ax.set_title("Q3: bistable vs linear (K=8, noise=20%)")
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle(f"z51 — Hopfield associative memory  total {elapsed/60:.1f} min, "
                  f"N={N_CELLS}, {N_SEEDS} seeds",
                  fontsize=13, weight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "hopfield.png", dpi=140)
    plt.close(fig)
    print(f"\nWrote {OUT/'hopfield.png'}")
    print(f"Total: {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
