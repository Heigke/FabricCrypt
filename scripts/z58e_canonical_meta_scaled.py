"""z58e_canonical_meta_scaled.py — meta-plasticity on canonical, scaled.

Uses three speedups added to diff_canonical:
  (a) kcl_body_torch_batched   — vectorised KCL across N cells
  (b) torch.compile            — JIT the inner loop
  (c) soft_clamp               — keep gradient alive in latched regime

Compared to z58d (N=4, T=15): now N=32, T=25, 8 task roles.

Per-cell roles (8 categories, 4 cells each):
  0  'last bit'           bistable
  1  'first bit'          bistable + retention
  2  'count of 1s'        integrator
  3  'mean of last 5'     integrator + leak
  4  'majority overall'   thresholded integrator
  5  '1 if any flips'     bistable trigger
  6  'XOR last two'       not learnable in single cell — sanity
  7  'always 0.0'         control: should learn near linear/integrator
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

from nsram.diff_canonical import (kcl_body_torch_batched, soft_clamp)
from nsram.nsram_canonical import NSRAMParams

OUT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/"
           "results/z58e_canonical_meta_scaled")
OUT.mkdir(parents=True, exist_ok=True)
DTYPE = torch.float64

N = 32
T = 25
ROLES_PER_CELL = N // 8        # 4 cells per role
N_ITERS = 80
LR = 0.04
DT_NORM = 1e7

P = NSRAMParams(gamma_VG2=0.3, AGIDL=0.0,
                ALPHA0_mult=0.3, BJT_BF=200.0, BJT_IKF=1e-3, Rb=5e8)

USE_COMPILE = False  # disabled: breaks grad chain through autograd.Function

def kcl(Vb, VG1, VG2, Vd):
    return kcl_body_torch_batched(Vb, VG1, VG2, Vd, P)


def transient_batched(VG2, Vd_seq, Vb0, soft=True):
    """Batched transient. Vd_seq is shape (T,), broadcast to all cells.
    VG2, Vb0 shape (N,). Returns Vb_history (T, N)."""
    N_local = VG2.shape[0]
    VG1 = torch.full((N_local,), 0.6, dtype=DTYPE)
    Vb = Vb0
    history = []
    for t in range(Vd_seq.shape[0]):
        Vd_t = Vd_seq[t].expand(N_local)
        f = kcl(Vb, VG1, VG2, Vd_t)
        Vb = Vb + DT_NORM * f
        if soft:
            Vb = soft_clamp(Vb, -0.20, P.Vb_max, sharpness=25.0)
        else:
            Vb = torch.clamp(Vb, -0.20, P.Vb_max)
        history.append(Vb)
    return torch.stack(history)


def make_role_targets(Vd_pattern_np):
    """8 roles → expected final Vb per cell. Returns (8,) array, then we
    repeat to (N,) by broadcasting (4 cells per role).
    Vd_pattern_np in {0.5, 1.5}, bits = (>1.0)."""
    bits = (Vd_pattern_np > 1.0).astype(int)
    Tn = len(bits)
    last     = bits[-1]
    first    = bits[0]
    count    = bits.sum() / Tn
    last5    = bits[-5:].mean()
    majority = int(bits.sum() > Tn / 2)
    flips    = int(any(bits[i] != bits[i-1] for i in range(1, Tn)))
    xor_last = int(bits[-1] != bits[-2]) if Tn >= 2 else 0
    role_targets = np.array([
        0.55 if last else -0.05,
        0.55 if first else -0.05,
        -0.10 + count * 0.65,
        -0.10 + last5 * 0.65,
        0.55 if majority else -0.05,
        0.55 if flips else -0.05,
        0.55 if xor_last else -0.05,
        0.0,                                # control
    ])
    # Repeat to N cells (4 per role)
    full = np.repeat(role_targets, ROLES_PER_CELL)
    return torch.tensor(full, dtype=DTYPE)


def evaluate(VG2_param, n_trials=4, seed=0):
    rng = np.random.default_rng(seed)
    total_loss = torch.zeros((), dtype=DTYPE)
    finals_all = []; targets_all = []
    for trial in range(n_trials):
        Vd_pat = rng.choice([0.5, 1.5], T)
        Vd_seq = torch.tensor(Vd_pat, dtype=DTYPE)
        targets = make_role_targets(Vd_pat)
        Vb0 = VG2_param.detach()
        hist = transient_batched(VG2_param, Vd_seq, Vb0, soft=True)
        Vb_final = hist[-1]
        loss = ((Vb_final - targets) ** 2).mean()
        total_loss = total_loss + loss
        finals_all.append(Vb_final.detach().cpu().numpy())
        targets_all.append(targets.cpu().numpy())
    return total_loss / n_trials, np.array(finals_all), np.array(targets_all)


def main():
    t0 = time.time()
    torch.manual_seed(0); np.random.seed(0)

    # ── (1) Baseline: uniform VG2 = -0.10 ────────────────────────────
    print("\n── BASELINE (uniform VG2 = -0.10) ──")
    vg2_uniform = nn.Parameter(torch.full((N,), -0.10, dtype=DTYPE))
    with torch.no_grad():
        baseline_loss, _, _ = evaluate(vg2_uniform, n_trials=8, seed=999)
    print(f"  baseline loss = {baseline_loss.item():.4f}")

    # ── (2) Trained per-cell VG2 ─────────────────────────────────────
    print(f"\n── TRAINED (N={N}, T={T}, {N_ITERS} iters, "
          f"compile={USE_COMPILE}) ──")
    vg2 = nn.Parameter(torch.full((N,), -0.10, dtype=DTYPE) +
                        0.05 * torch.randn(N, dtype=DTYPE))
    vg2_init = vg2.detach().cpu().numpy().copy()
    opt = torch.optim.Adam([vg2], lr=LR)
    history = {"iter": [], "loss": []}

    for it in range(N_ITERS):
        opt.zero_grad()
        loss, _, _ = evaluate(vg2, n_trials=4, seed=it)
        loss.backward()
        opt.step()
        with torch.no_grad():
            vg2.clamp_(-0.40, 0.40)
        if it % 5 == 0 or it == N_ITERS - 1:
            history["iter"].append(it)
            history["loss"].append(loss.item())
            elapsed_now = time.time() - t0
            print(f"  it={it:3d}  loss={loss.item():.4f}  "
                  f"elapsed={elapsed_now:.1f}s  "
                  f"VG2[mean,std]=[{vg2.mean().item():+.3f},{vg2.std().item():.3f}]",
                  flush=True)

    vg2_final = vg2.detach().cpu().numpy()
    with torch.no_grad():
        learned_loss, finals, targets = evaluate(vg2, n_trials=8, seed=12345)
    improvement = (1 - learned_loss.item() / baseline_loss.item()) * 100
    print(f"\nFinal eval:  learned={learned_loss.item():.4f}  "
          f"baseline={baseline_loss.item():.4f}")
    print(f"Improvement: {improvement:+.1f}%")

    elapsed = time.time() - t0
    summary = {
        "baseline_loss": float(baseline_loss),
        "learned_loss":  float(learned_loss),
        "improvement_pct": float(improvement),
        "vg2_init":  vg2_init.tolist(),
        "vg2_final": vg2_final.tolist(),
        "history": history, "elapsed_s": elapsed,
        "config": {"N": N, "T": T, "n_iters": N_ITERS, "lr": LR,
                    "DT_NORM": DT_NORM, "compile": USE_COMPILE,
                    "roles": ["last", "first", "count", "last5",
                              "majority", "flips", "xor_last", "control"]}
    }
    with open(OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ── Plots ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    ax = axes[0, 0]
    ax.plot(history["iter"], history["loss"], "o-", lw=2, color="#3498db",
            label="trained")
    ax.axhline(baseline_loss.item(), color="black", ls="--",
               label=f"baseline={baseline_loss.item():.3f}")
    ax.set_xlabel("iter"); ax.set_ylabel("loss")
    ax.set_title("Training curve (canonical, batched)")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    role_names = ["last", "first", "count", "last5", "majority",
                  "flips", "xor", "ctrl"]
    role_centers = np.arange(8) * ROLES_PER_CELL + ROLES_PER_CELL / 2 - 0.5
    cells_idx = np.arange(N)
    ax.bar(cells_idx, vg2_final, color=[plt.get_cmap("tab10")(i // ROLES_PER_CELL) for i in cells_idx])
    for c, name in zip(role_centers, role_names):
        ax.text(c, ax.get_ylim()[1] * 1.02, name,
                ha="center", va="bottom", fontsize=10, weight="bold")
    ax.axhline(-0.10, color="red", ls=":", label="uniform baseline")
    for k in range(1, 8):
        ax.axvline(k * ROLES_PER_CELL - 0.5, color="black", ls=":", alpha=0.3)
    ax.set_xlabel("cell index (grouped by role)")
    ax.set_ylabel("learned VG2 [V]")
    ax.set_title(f"Per-cell VG2 allocation (mean±std per role)")
    ax.legend(loc="lower right", fontsize=9); ax.grid(alpha=0.3, axis="y")

    ax = axes[1, 0]
    bars = ax.bar(["baseline\n(uniform)", "learned\n(per-cell)"],
                   [baseline_loss.item(), learned_loss.item()],
                   color=["#95a5a6", "#3498db"])
    for b, v in zip(bars, [baseline_loss.item(), learned_loss.item()]):
        ax.text(b.get_x() + b.get_width()/2, v, f"{v:.4f}",
                ha="center", va="bottom", fontsize=12, weight="bold")
    ax.set_ylabel("loss")
    ax.set_title(f"Improvement: {improvement:+.1f}%")
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1, 1]
    # Per-role mean predicted vs target
    finals_mean = finals.mean(axis=0).reshape(8, ROLES_PER_CELL).mean(axis=1)
    targets_mean = targets.mean(axis=0).reshape(8, ROLES_PER_CELL).mean(axis=1)
    width = 0.35
    x = np.arange(8)
    ax.bar(x - width/2, targets_mean, width, label="target", color="#e67e22")
    ax.bar(x + width/2, finals_mean,  width, label="learned", color="#3498db")
    ax.set_xticks(x); ax.set_xticklabels(role_names, rotation=20)
    ax.set_ylabel("Vb")
    ax.set_title("Per-role: target vs learned (mean over trials & cells)")
    ax.legend(); ax.grid(alpha=0.3, axis="y")

    fig.suptitle(f"z58e — Scaled meta-plasticity on canonical NS-RAM  "
                 f"(N={N} cells, T={T}, {N_ITERS} iters, {elapsed:.0f}s, compile={USE_COMPILE})",
                 fontsize=13, weight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "meta_scaled.png", dpi=140)
    plt.close(fig)
    print(f"\nWrote {OUT/'meta_scaled.png'}")


if __name__ == "__main__":
    main()
