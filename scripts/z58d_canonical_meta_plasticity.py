"""z58d_canonical_meta_plasticity.py — meta-plasticity training on the
canonical NS-RAM model.

z58b's training on cell_fast diverged because cell_fast can't access the
3 regimes that meta-plasticity needs.  z58c just confirmed canonical
DOES span the regimes.  Now: train per-cell VG2 on canonical.

Strategy: forward-Euler transient on the body KCL using the
torch-differentiable kcl_body_torch.  We don't need brentq for transient
(only for steady-state) — the in-time integration is naturally
differentiable end-to-end.

Task (small + interpretable):
  N=4 cells, each gets the same drive sequence Vd(t).
  Random binary sequence of T=15 pulses.
  Target = a per-cell role:
    - cell 0: 'remember last pulse'      (needs bistable / synapse)
    - cell 1: 'count of pulses up'       (needs integrator)
    - cell 2: 'remember 2nd-last pulse'  (needs bistable + leak)
    - cell 3: 'instant follower'         (needs integrator with low VG2)
  Per-cell loss after T steps. Train per-cell VG2.

If meta-plasticity works on real physics, learned VG2 should self-organize
into 'binary'/'synapse'/'neuron' roles.
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

from nsram.diff_canonical import kcl_body_torch
from nsram.nsram_canonical import NSRAMParams

OUT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/"
           "results/z58d_canonical_meta_plasticity")
OUT.mkdir(parents=True, exist_ok=True)

DTYPE = torch.float64

N = 4
T = 15
N_ITERS = 60
LR = 0.05

# Canonical params — same regime-friendly settings as z58c
P = NSRAMParams(gamma_VG2=0.3, AGIDL=0.0,
                ALPHA0_mult=0.3, BJT_BF=200.0, BJT_IKF=1e-3,
                Rb=5e8)   # tighter leak so Euler steps are stable
DT_NORM = 1e7   # scale Iii / Ibjt / Ileak (~1e-9 A) to dVb ~ 0.01 per step


def transient(VG1, VG2_per_cell, Vd_seq, Vb0, dt_norm=DT_NORM):
    """Run the canonical body-KCL transient for one cell across T steps.
    All inputs are torch tensors (scalar VG1, scalar VG2, T-vec Vd, scalar Vb0).
    Returns Vb_history (T,)."""
    Vb = Vb0
    history = []
    Tlen = Vd_seq.shape[0]
    for t in range(Tlen):
        f = kcl_body_torch(Vb, VG1, VG2_per_cell, Vd_seq[t], P)
        Vb = Vb + dt_norm * f
        # soft clamp via tanh-style — keeps gradient alive at boundaries
        Vb = torch.clamp(Vb, -0.20, P.Vb_max)
        history.append(Vb)
    return torch.stack(history)


def make_targets(Vd_pattern):
    """Per-cell target Vb after T steps, given the binary Vd pattern.

    We construct human-interpretable targets:
      cell 0: high Vb if last bit = 1
      cell 1: Vb proportional to count of 1s (integrator)
      cell 2: high Vb if 2nd-last bit = 1
      cell 3: high Vb if first half had more 1s than second half
    Returns shape (N,).
    """
    pat = Vd_pattern.detach().cpu().numpy()  # 0/1 entries
    bits = (pat > 1.0).astype(int)          # >1.0V → bit=1
    Tn = len(bits)
    half = Tn // 2
    last = bits[-1]
    second_last = bits[-2] if Tn >= 2 else 0
    count = bits.sum() / Tn
    first_half_dom = (bits[:half].sum() > bits[half:].sum()).astype(int)
    targets = np.array([
        0.6 if last else -0.05,
        -0.05 + count * 0.7,
        0.6 if second_last else -0.05,
        0.5 if first_half_dom else 0.0,
    ])
    return torch.tensor(targets, dtype=DTYPE)


def evaluate(VG2_param, n_trials=4, seed=0):
    """Run n_trials random patterns, return total loss (scalar tensor)."""
    rng = np.random.default_rng(seed)
    VG1 = torch.tensor(0.6, dtype=DTYPE)
    total_loss = torch.zeros((), dtype=DTYPE)
    final_Vb_all = []
    targets_all = []
    for trial in range(n_trials):
        Vd_seq_np = rng.choice([0.5, 1.5], T)
        Vd_seq = torch.tensor(Vd_seq_np, dtype=DTYPE)
        targets = make_targets(Vd_seq)
        # Per-cell forward (independent — each cell sees same Vd but has own VG2)
        Vb_finals = []
        for i in range(N):
            Vb0 = VG2_param[i].detach()
            hist = transient(VG1, VG2_param[i], Vd_seq, Vb0)
            Vb_finals.append(hist[-1])
        Vb_final = torch.stack(Vb_finals)
        loss = ((Vb_final - targets) ** 2).sum()
        total_loss = total_loss + loss
        final_Vb_all.append(Vb_final.detach().cpu().numpy())
        targets_all.append(targets.cpu().numpy())
    return total_loss / n_trials, np.array(final_Vb_all), np.array(targets_all)


def main():
    t0 = time.time()
    torch.manual_seed(42)
    np.random.seed(42)

    # ── Baseline: uniform VG2 = -0.10 ─────────────────────────────────
    print("\n── BASELINE (uniform VG2 = -0.10) ──")
    vg2_baseline = nn.Parameter(torch.full((N,), -0.10, dtype=DTYPE))
    with torch.no_grad():
        baseline_loss, _, _ = evaluate(vg2_baseline, seed=999)
    print(f"  baseline loss = {baseline_loss.item():.4f}")

    # ── Trained per-cell VG2 ──────────────────────────────────────────
    print("\n── TRAINED (per-cell learnable VG2) ──")
    vg2 = nn.Parameter(torch.full((N,), -0.10, dtype=DTYPE) +
                        0.05 * torch.randn(N, dtype=DTYPE))
    vg2_init = vg2.detach().cpu().numpy().copy()
    opt = torch.optim.Adam([vg2], lr=LR)
    history = {"iter": [], "loss": [], "vg2_mean": [], "vg2_std": []}

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
            history["vg2_mean"].append(float(vg2.mean()))
            history["vg2_std"].append(float(vg2.std()))
            print(f"  it={it:3d}  loss={loss.item():.4f}  "
                  f"VG2=[{vg2[0].item():+.2f},{vg2[1].item():+.2f},"
                  f"{vg2[2].item():+.2f},{vg2[3].item():+.2f}]",
                  flush=True)

    vg2_final = vg2.detach().cpu().numpy()

    with torch.no_grad():
        learned_loss, finals, targets = evaluate(vg2, n_trials=8, seed=12345)
    print(f"\nFinal eval (unseen seeds):  learned loss = {learned_loss.item():.4f}  "
          f"baseline = {baseline_loss.item():.4f}")
    improvement = (1 - learned_loss.item() / baseline_loss.item()) * 100
    print(f"Improvement: {improvement:+.1f}%")

    elapsed = time.time() - t0
    summary = {
        "baseline_loss": float(baseline_loss),
        "learned_loss": float(learned_loss),
        "improvement_pct": float(improvement),
        "vg2_init": vg2_init.tolist(),
        "vg2_final": vg2_final.tolist(),
        "history": history,
        "elapsed_s": elapsed,
        "config": {"N": N, "T": T, "n_iters": N_ITERS, "lr": LR,
                    "DT_NORM": DT_NORM, "params": "ALPHA0_mult=0.3, BJT_BF=200, Rb=5e8"}
    }
    with open(OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ── Plots ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.plot(history["iter"], history["loss"], "o-", lw=2, color="#3498db", label="trained")
    ax.axhline(baseline_loss.item(), color="black", ls="--",
               label=f"baseline={baseline_loss.item():.3f}")
    ax.set_xlabel("iter"); ax.set_ylabel("loss")
    ax.set_title("Training curve (canonical model)")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    cells_idx = np.arange(N)
    width = 0.35
    ax.bar(cells_idx - width/2, vg2_init, width, label="initial", color="#95a5a6")
    ax.bar(cells_idx + width/2, vg2_final, width, label="learned", color="#3498db")
    ax.axhline(0, color="black", lw=0.5)
    ax.axhline(-0.10, color="red", ls=":", label="uniform baseline")
    ax.set_xlabel("cell index"); ax.set_ylabel("VG2 [V]")
    ax.set_xticks(cells_idx)
    ax.set_xticklabels([f"c{i}\n{role}" for i, role in enumerate(
        ["last", "count", "2nd-last", "first-half"])])
    ax.set_title("Per-cell VG2 allocation")
    ax.legend(); ax.grid(alpha=0.3, axis="y")

    ax = axes[1, 0]
    bars = ax.bar(["baseline\n(uniform)", "learned\n(per-cell)"],
                   [baseline_loss.item(), learned_loss.item()],
                   color=["#95a5a6", "#3498db"])
    for b, v in zip(bars, [baseline_loss.item(), learned_loss.item()]):
        ax.text(b.get_x() + b.get_width()/2, v, f"{v:.3f}",
                ha="center", va="bottom", fontsize=12, weight="bold")
    ax.set_ylabel("loss")
    ax.set_title(f"Improvement: {improvement:+.1f}%")
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1, 1]
    # Show predicted vs target on last eval set
    n_show = finals.shape[0]
    for trial in range(n_show):
        for cell in range(N):
            ax.scatter(targets[trial, cell], finals[trial, cell],
                       color=plt.get_cmap("tab10")(cell), alpha=0.6, s=40)
    lims = [-0.2, 0.85]
    ax.plot(lims, lims, "k--", alpha=0.3, label="ideal y=x")
    ax.set_xlabel("target Vb"); ax.set_ylabel("predicted Vb (after T steps)")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_title("Per-cell predictions vs targets (final eval)")
    ax.grid(alpha=0.3); ax.legend()

    fig.suptitle(f"z58d — Meta-plasticity on canonical NS-RAM model  "
                 f"({elapsed:.0f}s, N={N} cells, {N_ITERS} iters)",
                 fontsize=13, weight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "meta_canonical.png", dpi=140)
    plt.close(fig)
    print(f"\nWrote {OUT/'meta_canonical.png'}")


if __name__ == "__main__":
    main()
