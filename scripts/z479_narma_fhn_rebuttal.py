"""z479 — Rebuttal to ERvMESH NARMA-10 killshot using FHN-trap cells.

Replaces NSRAM_LUT scalar tanh cell with discrete FitzHugh-Nagumo-coupled
reservoir cell carrying an intrinsic slow recovery variable n. Per cell:

    V_{t+1} = clip( (1-a)·V_t + a·tanh(α·(pre - k_n·n_t)),  V_min, V_max )
    n_{t+1} = (1 - dt/τ_slow)·n_t + (dt/τ_slow)·α_n·(V_t - V_n0)

This is the discrete-time abstraction of z477c's SPICE-level FHN trap
(τ_slow=800ns, k_n=1e-4, V_b clamp, ~12 oscillation cycles per cell).
We set dt/τ_slow = 0.125 so each reservoir step ≈ τ_slow/8, giving
the slow variable several time-constants of memory and inducing
intrinsic relaxation oscillations under sustained drive.

Compared cells (4):
    NSRAM_LUT     — original yesterday's killshot cell (tanh, no slow var)
    Linear        — identity, leak-only (sanity)
    ESN_canon     — Jaeger 2001 tanh (N=200, p=0.05, sr=0.95)
    FHN_trap      — NEW: two-state FHN-coupled cell

Topologies: ER_SPARSE (p=0.02), MESH_4N (32x32 toroidal).
Tasks: Mackey-Glass τ=17 (NRMSE), NARMA-10 (NMSE), 3 seeds, N=1024.

Pre-registered gates (rebuttal restored if all PASS w/ FHN; partial if
NARMA-10 sanity passes; STILL_DEAD otherwise).
"""
from __future__ import annotations
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
OUT = REPO / "results" / "z479_narma_fhn_rebuttal"
OUT.mkdir(parents=True, exist_ok=True)

# Reuse benchmarks + ridge from er_vs_mesh_killshot
sys.path.insert(0, str(REPO / "scripts"))
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("ervmesh", REPO / "scripts" / "er_vs_mesh_killshot.py")
_ervmesh = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_ervmesh)

mackey_glass = _ervmesh.mackey_glass
narma10 = _ervmesh.narma10
build_er_sparse = _ervmesh.build_er_sparse
build_mesh_4n = _ervmesh.build_mesh_4n
ridge_train_cv = _ervmesh.ridge_train_cv
ridge_predict = _ervmesh.ridge_predict
nrmse = _ervmesh.nrmse
nmse = _ervmesh.nmse


class FHNReservoir:
    """Reservoir with per-cell FHN-coupled slow trap n.

    States per cell: V (fast/body charge), n (slow recovery).
        pre  = input_scale·u·W_in + W·V + bias
        V'   = clip( (1-a)·V + a·tanh(α·(pre - k_n·n)), V_min, V_max )
        n'   = (1 - dt_over_tau)·n + dt_over_tau·α_n·(V - V_n0)
        feat = tanh(α·V) concatenated with n  (2·N features)
    """

    def __init__(self, N, W, leak, alpha, input_scale,
                 dt_over_tau, k_n, V_n0, alpha_n,
                 V_clip, device, seed):
        self.N = N
        self.W = W
        self.leak = leak
        self.alpha = alpha
        self.input_scale = input_scale
        self.dt_over_tau = dt_over_tau
        self.k_n = k_n
        self.V_n0 = V_n0
        self.alpha_n = alpha_n
        self.V_min, self.V_max = V_clip
        self.device = device
        g = torch.Generator(device="cpu").manual_seed(seed + 7919)
        self.W_in = (torch.rand(N, generator=g) * 2 - 1).to(device)
        self.bias = (torch.rand(N, generator=g) * 0.2 - 0.1).to(device)
        self.V = torch.zeros(N, device=device)
        self.n = torch.zeros(N, device=device)

    def reset(self):
        self.V.zero_(); self.n.zero_()

    @torch.no_grad()
    def run(self, u):
        T = u.shape[0]
        a = self.leak
        dot = self.dt_over_tau
        feats = torch.empty((T, 2 * self.N), device=self.device)
        for t in range(T):
            pre = self.input_scale * u[t] * self.W_in + self.W @ self.V + self.bias
            drive = pre - self.k_n * self.n
            Vn = (1.0 - a) * self.V + a * torch.tanh(self.alpha * drive)
            Vn.clamp_(self.V_min, self.V_max)
            nn = (1.0 - dot) * self.n + dot * self.alpha_n * (self.V - self.V_n0)
            feats[t, :self.N] = torch.tanh(self.alpha * Vn)
            feats[t, self.N:] = nn
            self.V = Vn
            self.n = nn
        return feats


def build_W(topology, N, sr, seed, device):
    if topology == "ER_SPARSE":
        return build_er_sparse(N, p=0.02, sr=sr, seed=seed, device=device)
    elif topology == "MESH_4N":
        return build_mesh_4n(N, sr=sr, seed=seed, device=device)
    raise ValueError(topology)


def make_task_arrays(task, seed, device):
    WASHOUT = 500
    TRAIN, TEST = 4000, 2000
    if task == "MG":
        total = WASHOUT + TRAIN + TEST + 1
        x = mackey_glass(total, tau=17, seed=seed)
        x_norm = (x - x.mean()) / (x.std() + 1e-12)
        u_np = x_norm[:-1]; y_np = x_norm[1:]
    else:
        total = WASHOUT + TRAIN + TEST
        u_np, y_np = narma10(total, seed=seed)
    u = torch.tensor(u_np, dtype=torch.float32, device=device)
    y = torch.tensor(y_np, dtype=torch.float32, device=device)
    return u, y, WASHOUT, TRAIN, TEST


def err_for(task, yte, yhat):
    return (nrmse(yte, yhat), "NRMSE") if task == "MG" else (nmse(yte, yhat), "NMSE")


def run_legacy(task, topology, nonlinearity, seed, N, sr, device):
    """Original NSRAM_LUT / Linear from er_vs_mesh_killshot.Reservoir."""
    u, y, WASHOUT, TRAIN, TEST = make_task_arrays(task, seed, device)
    W = build_W(topology, N, sr, seed, device)
    in_scale = 1.0 if task == "MG" else 2.0
    res = _ervmesh.Reservoir(N=N, W=W, nonlinearity=nonlinearity,
                             leak=0.30, alpha=2.0, input_scale=in_scale,
                             device=device, seed=seed)
    res.reset(); _ = res.run(u[:32]); res.reset()
    t0 = time.time()
    feats = res.run(u)
    wall = time.time() - t0
    feats = feats[WASHOUT:]; y_use = y[WASHOUT:]
    Xtr = feats[:TRAIN]; ytr = y_use[:TRAIN]
    Xte = feats[TRAIN:TRAIN + TEST]; yte = y_use[TRAIN:TRAIN + TEST]
    W_read, alpha = ridge_train_cv(Xtr, ytr)
    yhat = ridge_predict(Xte, W_read)
    err, metric = err_for(task, yte, yhat)
    return {"task": task, "topology": topology, "cell": nonlinearity,
            "seed": seed, "N": N, "err_metric": metric, "err_test": err,
            "ridge_alpha": float(alpha) if alpha is not None else None,
            "wall_s": wall}


def run_esn_canon(task, topology, seed, device):
    """Jaeger 2001 canonical: N=200, p=0.05, sr=0.95, tanh, leak=0.3.
    Topology arg is ignored at fixed canonical config (kept for grid uniformity).
    Reports the topology label of the parent grid for accounting purposes."""
    N = 200
    u, y, WASHOUT, TRAIN, TEST = make_task_arrays(task, seed, device)
    W = build_er_sparse(N, p=0.05, sr=0.95, seed=seed, device=device)
    in_scale = 1.0 if task == "MG" else 2.0
    res = _ervmesh.Reservoir(N=N, W=W, nonlinearity="NSRAM_LUT",
                             leak=0.30, alpha=1.0, input_scale=in_scale,
                             device=device, seed=seed)
    res.reset(); _ = res.run(u[:32]); res.reset()
    t0 = time.time()
    feats = res.run(u)
    wall = time.time() - t0
    feats = feats[WASHOUT:]; y_use = y[WASHOUT:]
    Xtr = feats[:TRAIN]; ytr = y_use[:TRAIN]
    Xte = feats[TRAIN:TRAIN + TEST]; yte = y_use[TRAIN:TRAIN + TEST]
    W_read, alpha = ridge_train_cv(Xtr, ytr)
    yhat = ridge_predict(Xte, W_read)
    err, metric = err_for(task, yte, yhat)
    return {"task": task, "topology": topology, "cell": "ESN_canon",
            "seed": seed, "N": N, "err_metric": metric, "err_test": err,
            "ridge_alpha": float(alpha) if alpha is not None else None,
            "wall_s": wall}


def run_fhn(task, topology, seed, N, sr, device):
    u, y, WASHOUT, TRAIN, TEST = make_task_arrays(task, seed, device)
    W = build_W(topology, N, sr, seed, device)
    in_scale = 1.0 if task == "MG" else 2.0
    # FHN params abstracted from z477c (τ_slow=800ns, k_n=1e-4, V_b clamp ~0.6V).
    # dt_over_tau=0.125 → ~8 steps per τ_slow, ~800 τ_slow over a 6500-step run.
    # k_n is scaled up because we removed unit scales (analog→dimensionless).
    res = FHNReservoir(N=N, W=W, leak=0.30, alpha=2.0, input_scale=in_scale,
                       dt_over_tau=0.125, k_n=0.5, V_n0=0.0, alpha_n=1.0,
                       V_clip=(-0.7, 0.7), device=device, seed=seed)
    res.reset(); _ = res.run(u[:32]); res.reset()
    t0 = time.time()
    feats = res.run(u)
    wall = time.time() - t0
    feats = feats[WASHOUT:]; y_use = y[WASHOUT:]
    Xtr = feats[:TRAIN]; ytr = y_use[:TRAIN]
    Xte = feats[TRAIN:TRAIN + TEST]; yte = y_use[TRAIN:TRAIN + TEST]
    W_read, alpha = ridge_train_cv(Xtr, ytr)
    yhat = ridge_predict(Xte, W_read)
    err, metric = err_for(task, yte, yhat)
    return {"task": task, "topology": topology, "cell": "FHN_trap",
            "seed": seed, "N": N, "err_metric": metric, "err_test": err,
            "ridge_alpha": float(alpha) if alpha is not None else None,
            "wall_s": wall}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[z479] device={device}  torch={torch.__version__}", flush=True)

    N = 1024
    SR = 0.9
    SEEDS = [0, 1, 2]
    TOPOS = ["ER_SPARSE", "MESH_4N"]
    CELLS = ["NSRAM_LUT", "Linear", "ESN_canon", "FHN_trap"]
    TASKS = ["MG", "NARMA10"]

    results = []
    t_start = time.time()
    for task in TASKS:
        for topo in TOPOS:
            for cell in CELLS:
                for seed in SEEDS:
                    print(f"[z479] {task} {topo} {cell} seed={seed} ...", flush=True)
                    if cell in ("NSRAM_LUT", "Linear"):
                        r = run_legacy(task, topo, cell, seed, N, SR, device)
                    elif cell == "ESN_canon":
                        r = run_esn_canon(task, topo, seed, device)
                    elif cell == "FHN_trap":
                        r = run_fhn(task, topo, seed, N, SR, device)
                    else:
                        raise ValueError(cell)
                    print(f"        {r['err_metric']}={r['err_test']:.5f}  "
                          f"wall={r['wall_s']:.1f}s", flush=True)
                    results.append(r)

    t_total = time.time() - t_start
    print(f"[z479] all done in {t_total:.1f}s", flush=True)

    # ───────── analysis ─────────
    def med_set(task, **filt):
        vals = [r["err_test"] for r in results if r["task"] == task
                and all(r.get(k) == v for k, v in filt.items())]
        return (float(np.median(vals)) if vals else float("nan")), vals

    # Per-cell medians per (task, topology)
    summary = {}
    for task in TASKS:
        summary[task] = {}
        for topo in TOPOS:
            for cell in CELLS:
                m, vs = med_set(task, topology=topo, cell=cell)
                summary[task][f"{topo}|{cell}"] = {"median": m, "seeds": vs}

    # Pre-registered gates restricted to FHN_trap cell
    gates = {}
    # (1) ER_SPARSE advantage with FHN_trap on both tasks
    g1 = {}
    for task in TASKS:
        er, _ = med_set(task, topology="ER_SPARSE", cell="FHN_trap")
        ms, _ = med_set(task, topology="MESH_4N", cell="FHN_trap")
        g1[task] = {"median_ER_FHN": er, "median_MESH_FHN": ms,
                    "ratio": er / (ms + 1e-12), "threshold": 0.95,
                    "pass": (er <= 0.95 * ms)}
    gates["topology_FHN"] = {"per_task": g1,
                             "pass": all(g1[t]["pass"] for t in TASKS)}

    # (2) Nonlinearity benefit: median_Linear ≥ 1.10 × median_FHN_trap on both tasks
    g2 = {}
    for task in TASKS:
        lin_vals = [r["err_test"] for r in results
                    if r["task"] == task and r["cell"] == "Linear"]
        fhn_vals = [r["err_test"] for r in results
                    if r["task"] == task and r["cell"] == "FHN_trap"]
        med_lin = float(np.median(lin_vals)); med_fhn = float(np.median(fhn_vals))
        g2[task] = {"median_Linear": med_lin, "median_FHN": med_fhn,
                    "ratio": med_lin / (med_fhn + 1e-12), "threshold": 1.10,
                    "pass": (med_lin >= 1.10 * med_fhn)}
    gates["nonlinearity_FHN"] = {"per_task": g2,
                                 "pass": all(g2[t]["pass"] for t in TASKS)}

    # (3) Absolute sanity: ER_SPARSE+FHN_trap MG NRMSE ≤ 0.05 AND NARMA-10 NMSE ≤ 0.15
    mg_er_fhn, _ = med_set("MG", topology="ER_SPARSE", cell="FHN_trap")
    na_er_fhn, _ = med_set("NARMA10", topology="ER_SPARSE", cell="FHN_trap")
    g3 = {"MG_NRMSE": mg_er_fhn, "MG_pass": mg_er_fhn <= 0.05,
          "NARMA10_NMSE": na_er_fhn, "NARMA10_pass": na_er_fhn <= 0.15,
          "pass": (mg_er_fhn <= 0.05) and (na_er_fhn <= 0.15)}
    gates["absolute_sanity_FHN"] = g3

    fail_count = sum(0 if gates[k]["pass"] else 1
                     for k in ("topology_FHN", "nonlinearity_FHN", "absolute_sanity_FHN"))

    if fail_count == 0:
        verdict = "FULL_REBUTTAL"
    elif g3["NARMA10_pass"]:
        verdict = "PARTIAL_REBUTTAL"
    else:
        verdict = "STILL_DEAD"

    out = {
        "config": {"N": N, "sr": SR, "seeds": SEEDS, "topos": TOPOS,
                   "cells": CELLS, "tasks": TASKS, "wall_total_s": t_total,
                   "device": str(device)},
        "results": results,
        "summary_medians": summary,
        "gates": gates,
        "fail_count": fail_count,
        "verdict": verdict,
    }
    (OUT / "grid_results.json").write_text(json.dumps(out, indent=2))

    # comparison table
    lines = ["# z479 — NARMA-10 FHN-trap rebuttal — comparison table\n",
             f"\nWall: {t_total:.1f}s on {device}. N={N}, sr={SR}, seeds={SEEDS}.\n",
             "\n## Medians (lower better)\n",
             "| Task | Topology | Cell | Median | Seeds |",
             "|------|----------|------|--------|-------|"]
    for task in TASKS:
        for topo in TOPOS:
            for cell in CELLS:
                k = f"{topo}|{cell}"
                e = summary[task][k]
                seeds_str = ", ".join(f"{v:.4f}" for v in e["seeds"])
                lines.append(f"| {task} | {topo} | {cell} | {e['median']:.5f} | {seeds_str} |")
    lines += ["\n## Gates (FHN_trap)\n"]
    for task in TASKS:
        g = gates["topology_FHN"]["per_task"][task]
        lines.append(f"- (1) Topology {task}: ER={g['median_ER_FHN']:.5f} "
                     f"vs MESH={g['median_MESH_FHN']:.5f} ratio={g['ratio']:.3f} "
                     f"(need≤0.95) → {'PASS' if g['pass'] else 'FAIL'}")
    for task in TASKS:
        g = gates["nonlinearity_FHN"]["per_task"][task]
        lines.append(f"- (2) Nonlinearity {task}: Linear={g['median_Linear']:.5f} "
                     f"vs FHN={g['median_FHN']:.5f} ratio={g['ratio']:.3f} "
                     f"(need≥1.10) → {'PASS' if g['pass'] else 'FAIL'}")
    lines.append(f"- (3) Sanity MG NRMSE={g3['MG_NRMSE']:.5f} (≤0.05) → "
                 f"{'PASS' if g3['MG_pass'] else 'FAIL'}")
    lines.append(f"- (3) Sanity NARMA10 NMSE={g3['NARMA10_NMSE']:.5f} (≤0.15) → "
                 f"{'PASS' if g3['NARMA10_pass'] else 'FAIL'}")
    lines.append(f"\n**fail_count={fail_count}/3 → VERDICT: {verdict}**\n")
    (OUT / "comparison_table.md").write_text("\n".join(lines))

    # honest verdict
    vlines = [f"# z479 — Honest Verdict: {verdict}\n",
              f"\nFAIL count: {fail_count}/3\n",
              "\n## Pre-registered rebuttal criteria",
              "- FULL_REBUTTAL: all 3 gates PASS with FHN_trap on both tasks",
              "- PARTIAL_REBUTTAL: NARMA-10 sanity (NMSE≤0.15) passes",
              "- STILL_DEAD: NARMA-10 sanity fails → FHN does not rescue\n",
              "\n## Key numbers (FHN_trap cell)\n",
              f"- ER_SPARSE + FHN: MG NRMSE = {mg_er_fhn:.5f}, NARMA-10 NMSE = {na_er_fhn:.5f}",
              f"- MESH_4N + FHN: see table.",
              f"- Topology gate: {gates['topology_FHN']['pass']}",
              f"- Nonlinearity gate: {gates['nonlinearity_FHN']['pass']}",
              f"- Absolute sanity gate: {g3['pass']}",
              "\n## Compared to yesterday's killshot (NSRAM_LUT cell)\n"]
    for task in TASKS:
        nl_er, _ = med_set(task, topology="ER_SPARSE", cell="NSRAM_LUT")
        fhn_er, _ = med_set(task, topology="ER_SPARSE", cell="FHN_trap")
        delta = fhn_er - nl_er
        vlines.append(f"- {task} ER_SPARSE: NSRAM_LUT={nl_er:.5f}  "
                      f"FHN_trap={fhn_er:.5f}  Δ={delta:+.5f}")
    vlines.append("\n## ESN_canon reference (Jaeger 2001 N=200 sr=0.95)\n")
    for task in TASKS:
        esn, _ = med_set(task, cell="ESN_canon")
        vlines.append(f"- {task} ESN_canon median: {esn:.5f}")
    (OUT / "honest_verdict.md").write_text("\n".join(vlines))

    # pareto plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        colors = {"NSRAM_LUT": "tab:gray", "Linear": "tab:olive",
                  "ESN_canon": "tab:blue", "FHN_trap": "tab:red"}
        markers = {"ER_SPARSE": "o", "MESH_4N": "s"}
        for ax, task in zip(axes, TASKS):
            for cell in CELLS:
                for topo in TOPOS:
                    vs = summary[task][f"{topo}|{cell}"]["seeds"]
                    xs = [cell] * len(vs)
                    ax.scatter(xs, vs, c=colors[cell], marker=markers[topo],
                               edgecolors="k", s=70, alpha=0.85,
                               label=f"{topo}" if cell == CELLS[0] else None)
                med = summary[task][f"{TOPOS[0]}|{cell}"]["median"]
            if task == "NARMA10":
                ax.axhline(0.15, ls="--", c="r", alpha=0.6, label="sanity ≤0.15")
            else:
                ax.axhline(0.05, ls="--", c="r", alpha=0.6, label="sanity ≤0.05")
            ax.set_title(f"{task}  ({'NRMSE' if task=='MG' else 'NMSE'})")
            ax.set_ylabel("error")
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis="x", rotation=20)
        # dedupe legend
        h, l = axes[0].get_legend_handles_labels()
        seen = {}
        for hi, li in zip(h, l):
            if li not in seen: seen[li] = hi
        fig.legend(seen.values(), seen.keys(), loc="upper center", ncol=4,
                   bbox_to_anchor=(0.5, 1.02))
        fig.suptitle(f"z479 FHN-trap rebuttal — verdict={verdict}", y=1.08)
        fig.tight_layout()
        fig.savefig(OUT / "pareto_per_cell.png", dpi=130, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print(f"[z479] plot failed: {e}", flush=True)

    print(f"[z479] VERDICT: {verdict}  (fail_count={fail_count}/3)", flush=True)
    print(f"[z479] wrote {OUT}/", flush=True)


if __name__ == "__main__":
    main()
