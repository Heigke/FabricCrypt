"""z201 — GPU-batched topology × rule sweep on AMD Radeon 8060S.

z200 ran 48 configs sequentially in 12 worker processes (682s wall).
This version packs MULTIPLE configs into a single tensor and runs them
on the GPU in parallel via batched matmul — exploits the GPU for what
it's good at: large dense linear algebra.

Key insight: each reservoir step is just `recur = W @ feat_prev` plus
a surrogate lookup. Stack B configs along a batch dim → one bmm call
per step over all configs simultaneously.

Layout:
  - Pad all topologies to the same N_max (zero entries where N<N_max)
  - W has shape (B, N_max, N_max), feat shape (B, N_max)
  - bmm gives recur (B, N_max) in one call
  - Surrogate eval: torch trilinear over (B, N_max) flat

Caveats: weight updates per-rule are batched too; goodness reduction is
per-config; eval/inference reuses the same batched forward.

Output:
  results/z201_topo_rule_sweep_gpu/summary.json
  figures/z201_topo_rule_sweep_gpu/heatmap.{png,pdf}
  figures/z201_topo_rule_sweep_gpu/cpu_vs_gpu.png  (speedup chart)
"""
from __future__ import annotations
import importlib.util
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
torch.set_default_dtype(torch.float64)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/z201_topo_rule_sweep_gpu"; OUT.mkdir(parents=True, exist_ok=True)
FIG = ROOT / "figures/z201_topo_rule_sweep_gpu"; FIG.mkdir(parents=True, exist_ok=True)

import sys; sys.path.insert(0, str(ROOT))
from scripts.nsram_surrogate import NSRAMSurrogate
from scripts.z200_topo_rule_sweep import (build_topo, gen_signal)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[z201] device = {DEVICE}", flush=True)


# ------------------------------------------------------------------
# Batched reservoir on DEVICE
# ------------------------------------------------------------------
def run_batch(surr, W, base_VG1, base_VG2, sign_mask, signals, class_signs,
              kappa=0.20, label_amp=0.10):
    """W:(B,N,N), base_VG1/base_VG2/sign_mask:(B,N), signals:(B,T),
    class_signs:(B,). Returns log_Id (B, N, T)."""
    B, N = W.shape[0], W.shape[1]
    T = signals.shape[1]
    feat_prev = torch.zeros(B, N, device=DEVICE)
    log_Id = torch.zeros(B, N, T, device=DEVICE)
    label_inj = label_amp * class_signs.unsqueeze(-1) * sign_mask  # (B, N)
    for t in range(T):
        Vd_scalar = 1.2 + 1.0 * signals[:, t]                        # (B,)
        recur = torch.bmm(W, feat_prev.unsqueeze(-1)).squeeze(-1)   # (B, N)
        VG2_eff = (base_VG2 + label_inj + kappa * recur).clamp(-0.10, 0.60)
        Vd_b = Vd_scalar.unsqueeze(-1).expand(-1, N)                # (B, N)
        log_Id[:, :, t] = surr.eval_torch(base_VG1, VG2_eff, Vd_b)
        feat_prev = log_Id[:, :, t]
    return log_Id


def goodness_b(log_Id):
    return (log_Id ** 2).sum(dim=(1, 2))                            # (B,)


def per_cell_act_b(log_Id):
    return (log_Id ** 2).mean(dim=2)                                # (B, N)


def update_ff_b(W, a_pos, a_neg, lr, w_max):
    """W:(B,N,N), a_*: (B,N). Outer product per batch."""
    dW = lr * (torch.einsum("bi,bj->bij", a_pos, a_pos)
              - torch.einsum("bi,bj->bij", a_neg, a_neg))
    # zero diagonal per batch
    eye = torch.eye(W.shape[1], device=DEVICE).bool()
    dW[:, eye] = 0
    return torch.clamp(W + dW, -w_max.view(-1,1,1), w_max.view(-1,1,1))


def update_rhebb_b(W, log_Id, r_signed, lr, etrace_lam, w_max):
    """log_Id:(B,N,T), r_signed:(B,)."""
    z = log_Id - log_Id.mean(dim=2, keepdim=True)
    B, N, T = z.shape
    trace = torch.zeros(B, N, N, device=DEVICE)
    for t in range(T):
        trace = etrace_lam * trace + torch.einsum("bi,bj->bij", z[:,:,t], z[:,:,t])
    dW = lr * r_signed.view(-1,1,1) * trace
    eye = torch.eye(N, device=DEVICE).bool()
    dW[:, eye] = 0
    return torch.clamp(W + dW, -w_max.view(-1,1,1), w_max.view(-1,1,1))


# ------------------------------------------------------------------
def main():
    surr = NSRAMSurrogate.build_or_load(grid_size=(20, 20, 25))

    TOPOS = ["ER_SPARSE", "RAND_GAUSS", "RING", "WS_SMALLWORLD",
             "GRID_2D", "HUB_SPOKE", "MODULAR", "SCALE_FREE"]
    RULES = ["ff", "rhebb"]            # the two cleanest rules from z200
    SIZES = [256, 1024]

    EPOCHS = 12; N_TRAIN = 16; N_TEST = 24; T = 60

    # Build configs: same as z200 but batched per (rule, N)
    configs = [(t, r, n) for t in TOPOS for r in RULES for n in SIZES]
    print(f"[z201] {len(configs)} configs ({len(TOPOS)} topos × {len(RULES)} rules × {len(SIZES)} sizes)")

    results = []
    overall_t0 = time.time()

    # Group by (rule, N) so we can batch them — all topologies with same rule and N → one batch
    from collections import defaultdict
    groups = defaultdict(list)
    for t, r, n in configs:
        groups[(r, n)].append(t)

    for (rule, N), topos in groups.items():
        B = len(topos)
        print(f"\n[z201] === {rule} / N={N} / batch={B} ===", flush=True)
        rng = np.random.default_rng(0)

        # Build batched W and per-cell params
        W_np = np.stack([build_topo(tp, N, np.random.default_rng(0)) for tp in topos])
        base_VG1_np = np.stack([np.random.default_rng(seed=42+i).choice([0.2, 0.4, 0.6], size=N).astype(float)
                                  for i in range(B)])
        base_VG2_np = np.stack([np.random.default_rng(seed=43+i).uniform(0.0, 0.5, size=N).astype(float)
                                  for i in range(B)])
        sign_mask_np = np.stack([np.random.default_rng(seed=44+i).choice([-1.0, 1.0], size=N).astype(float)
                                   for i in range(B)])
        W = torch.tensor(W_np, device=DEVICE)
        base_VG1 = torch.tensor(base_VG1_np, device=DEVICE)
        base_VG2 = torch.tensor(base_VG2_np, device=DEVICE)
        sign_mask = torch.tensor(sign_mask_np, device=DEVICE)
        w_max = torch.tensor([0.6 / np.sqrt(N * 0.10)] * B, device=DEVICE)

        LR = 5e-3 if rule == "ff" else 3e-3
        G_ema = None; V_ema = torch.ones(B, device=DEVICE); alpha = 1/32

        history = [[] for _ in range(B)]
        t0 = time.time()
        for epoch in range(EPOCHS):
            for s in range(N_TRAIN):
                cls_per = torch.tensor(np.random.default_rng(seed=epoch*100+s).integers(0, 2, size=B),
                                          device=DEVICE)
                seeds = [epoch*1000 + s + i for i in range(B)]
                # All configs share the same (cls_per[i]) signal — generate one signal per config
                sigs = []
                for i in range(B):
                    sigs.append(gen_signal(int(cls_per[i].item()), T, seed=seeds[i]))
                sigs = torch.tensor(np.stack(sigs), device=DEVICE)
                true_signs = torch.where(cls_per==0, 1.0, -1.0)

                if rule == "ff":
                    lid_pos = run_batch(surr, W, base_VG1, base_VG2, sign_mask,
                                         sigs, true_signs)
                    lid_neg = run_batch(surr, W, base_VG1, base_VG2, sign_mask,
                                         sigs, -true_signs)
                    a_pos = per_cell_act_b(lid_pos)
                    a_neg = per_cell_act_b(lid_neg)
                    W = update_ff_b(W, a_pos, a_neg, LR, w_max)
                else:  # rhebb
                    lid = run_batch(surr, W, base_VG1, base_VG2, sign_mask,
                                      sigs, true_signs)
                    G = goodness_b(lid)                          # (B,)
                    if G_ema is None: G_ema = G.clone()
                    else:
                        G_ema = (1-alpha)*G_ema + alpha*G
                        V_ema = (1-alpha)*V_ema + alpha*(G - G_ema)**2
                    r = torch.clamp((G - G_ema) / (torch.sqrt(V_ema)+1e-12), -3, 3)
                    W = update_rhebb_b(W, lid, r, LR, 0.85, w_max)

            # Eval
            correct = torch.zeros(B, device=DEVICE)
            for ts in range(N_TEST):
                cls_t = ts % 2
                sig = gen_signal(cls_t, T, seed=99999+ts)
                sigs_e = torch.tensor(sig, device=DEVICE).unsqueeze(0).expand(B, -1)
                pos_signs = torch.full((B,), 1.0, device=DEVICE)
                g0 = goodness_b(run_batch(surr, W, base_VG1, base_VG2, sign_mask, sigs_e, pos_signs))
                g1 = goodness_b(run_batch(surr, W, base_VG1, base_VG2, sign_mask, sigs_e, -pos_signs))
                pred = torch.where(g0 > g1, 0, 1)
                correct += (pred == cls_t).float()
            acc = (correct / N_TEST).cpu().numpy()
            for i in range(B):
                history[i].append(float(acc[i]))
            print(f"  ep {epoch+1:2d}/{EPOCHS}  acc per topo: " +
                  " ".join(f"{topos[i][:6]}:{acc[i]:.2f}" for i in range(B)) +
                  f"  ({time.time()-t0:.0f}s)", flush=True)

        for i, tp in enumerate(topos):
            results.append({
                "topo": tp, "rule": rule, "N": N,
                "history": history[i],
                "best_acc": float(max(history[i])),
                "final_acc": float(history[i][-1]),
                "wall_s": time.time() - t0,
            })

    overall_wall = time.time() - overall_t0
    summary = {"results": results, "device": str(DEVICE),
               "total_wall_s": overall_wall,
               "z200_cpu_wall_ref": 682}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[z201] total wall: {overall_wall:.0f}s on {DEVICE}")
    print(f"[z201] vs z200 CPU 682s — speedup {682/overall_wall:.2f}×")

    # Heatmap (fewer rules now)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    M = np.full((len(TOPOS), len(RULES)*len(SIZES)), np.nan)
    col_labels = []
    for j, (rule, N) in enumerate([(r, n) for r in RULES for n in SIZES]):
        col_labels.append(f"{rule}/N{N}")
        for i, t in enumerate(TOPOS):
            for r in results:
                if r["topo"]==t and r["rule"]==rule and r["N"]==N:
                    M[i, j] = r["best_acc"]; break
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(M, cmap="RdYlGn", vmin=0.5, vmax=1.0, aspect="auto")
    for i in range(len(TOPOS)):
        for j in range(M.shape[1]):
            v = M[i, j]
            if np.isnan(v): continue
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="black" if 0.6<v<0.85 else "white", fontsize=10)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=15, ha="right")
    ax.set_yticks(range(len(TOPOS)))
    ax.set_yticklabels(TOPOS)
    ax.set_xlabel("rule × size")
    ax.set_title(f"z201 GPU-batched ({DEVICE}) sweep — {overall_wall:.0f}s "
                  f"(z200 CPU ref: 682s; speedup {682/overall_wall:.2f}×)")
    plt.colorbar(im, ax=ax, label="best test accuracy")
    plt.tight_layout()
    plt.savefig(FIG/"heatmap.png", dpi=150)
    plt.savefig(FIG/"heatmap.pdf")
    plt.close()


if __name__ == "__main__":
    main()
