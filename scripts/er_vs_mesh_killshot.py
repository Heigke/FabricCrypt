"""ERvMESH_killshot_1024_LUT — O81 oracle pre-registered falsifier.

Grid: topology ∈ {ER_SPARSE p=0.02, MESH_4N},
      nonlinearity ∈ {NSRAM_LUT (volatile R_body=1e7 surrogate), Linear},
      seeds ∈ {0,1,2}, N=1024, α=0.9.
Tasks: Mackey-Glass τ=17 (NRMSE), NARMA-10 (NMSE).

Pre-registered PASS (ALL three required):
  (1) ER_SPARSE advantage: median_ER ≤ 0.95 × median_MESH on both tasks.
  (2) Nonlinearity benefit: median_Linear ≥ 1.10 × median_NSRAM_LUT on both tasks.
  (3) Absolute sanity: ER_SPARSE+NSRAM_LUT MG NRMSE ≤ 0.05 AND NARMA-10 NMSE ≤ 0.15.

FAIL_count = number of {1,2,3} that fail. 0 = pass, 1 = partial, 2 = killshot, 3 = catastrophic.
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

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "ER_vs_MESH_killshot"
OUT.mkdir(parents=True, exist_ok=True)


# ───────────────────── benchmarks ─────────────────────
def mackey_glass(n_steps: int, tau: int = 17, beta: float = 0.2,
                 gamma: float = 0.1, n: float = 10.0,
                 dt: float = 1.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    burnin = 1000
    total = n_steps + burnin
    hist_len = max(tau + 1, 30)
    x = 1.2 + 0.05 * rng.standard_normal(total + hist_len)
    for t in range(hist_len, total + hist_len - 1):
        x[t + 1] = x[t] + dt * (
            beta * x[t - tau] / (1.0 + x[t - tau] ** n) - gamma * x[t]
        )
    return x[hist_len + burnin: hist_len + burnin + n_steps].astype(np.float64)


def narma10(n_steps: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Standard NARMA-10. Returns (u, y) where u ∈ U[0,0.5], y is the system."""
    rng = np.random.default_rng(seed)
    burnin = 200
    total = n_steps + burnin
    u = rng.uniform(0.0, 0.5, size=total)
    y = np.zeros(total)
    for t in range(10, total - 1):
        y[t + 1] = (0.3 * y[t]
                    + 0.05 * y[t] * np.sum(y[t - 9:t + 1])
                    + 1.5 * u[t - 9] * u[t]
                    + 0.1)
    return u[burnin:].astype(np.float64), y[burnin:].astype(np.float64)


# ───────────────────── topologies ─────────────────────
def _rescale_to_sr(Wd: torch.Tensor, sr: float, g) -> torch.Tensor:
    N = Wd.shape[0]
    v = torch.randn(N, generator=g)
    v /= v.norm() + 1e-12
    for _ in range(60):
        v = Wd @ v
        nrm = v.norm() + 1e-12
        v = v / nrm
    cur = float(nrm)
    if cur > 0:
        Wd = Wd * (sr / cur)
    return Wd


def build_er_sparse(N: int, p: float, sr: float, seed: int,
                    device, dtype=torch.float32) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    nnz = max(int(p * N * N), N)
    rows = torch.randint(0, N, (nnz,), generator=g)
    cols = torch.randint(0, N, (nnz,), generator=g)
    mask = rows != cols
    rows, cols = rows[mask], cols[mask]
    vals = torch.randn(rows.numel(), generator=g)
    Wd = torch.zeros((N, N))
    Wd[rows, cols] = vals
    Wd = _rescale_to_sr(Wd, sr, g)
    return Wd.to(device=device, dtype=dtype)


def build_mesh_4n(N: int, sr: float, seed: int,
                  device, dtype=torch.float32) -> torch.Tensor:
    """4-nearest-neighbor mesh on sqrt(N)×sqrt(N) toroidal grid."""
    side = int(round(math.sqrt(N)))
    assert side * side == N, f"N={N} must be perfect square for MESH_4N"
    g = torch.Generator(device="cpu").manual_seed(seed)
    Wd = torch.zeros((N, N))
    for i in range(side):
        for j in range(side):
            src = i * side + j
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni = (i + di) % side
                nj = (j + dj) % side
                dst = ni * side + nj
                # random sign/weight per edge (no self-loops)
                Wd[src, dst] = torch.randn(1, generator=g).item()
    Wd = _rescale_to_sr(Wd, sr, g)
    return Wd.to(device=device, dtype=dtype)


# ───────────────────── reservoir cell ─────────────────────
class Reservoir:
    """N cells, leaky-integrator dynamics.

    NSRAM_LUT mode: NS-RAM body charge surrogate calibrated to R_body=1e7
      volatile regime (z473). Membrane V is body charge analogue; sat
      nonlinearity is tanh with regime-tuned slope alpha=2.0 (saturates at
      ~Vb≈0.6 like z473 mario two-pulse Vb_peak=0.62), slow leak (=0.05)
      from t_reset≈40ns→dt mapping. Output = tanh(α·V) so volatile decay
      AND saturating I-V both visible to readout.

    Linear mode: V <- (1-a)·V + a·pre, output = V. Same leak so the only
      ablated factor is the nonlinearity.
    """

    def __init__(self, N: int, W: torch.Tensor, nonlinearity: str,
                 leak: float, alpha: float, input_scale: float,
                 device, seed: int):
        self.N = N
        self.W = W
        self.nonlinearity = nonlinearity
        self.leak = leak
        self.alpha = alpha
        self.input_scale = input_scale
        self.device = device
        g = torch.Generator(device="cpu").manual_seed(seed + 7919)
        self.W_in = (torch.rand(N, generator=g) * 2 - 1).to(device)
        self.bias = (torch.rand(N, generator=g) * 0.2 - 0.1).to(device)
        self.V = torch.zeros(N, device=device)

    def reset(self):
        self.V = torch.zeros(self.N, device=self.device)

    @torch.no_grad()
    def run(self, u: torch.Tensor) -> torch.Tensor:
        T = u.shape[0]
        a = self.leak  # update rate; leak retention = 1-a
        feats = torch.empty((T, self.N), device=self.device)
        for t in range(T):
            pre = self.input_scale * u[t] * self.W_in + self.W @ self.V + self.bias
            if self.nonlinearity == "NSRAM_LUT":
                Vn = (1.0 - a) * self.V + a * torch.tanh(self.alpha * pre)
                feats[t] = torch.tanh(self.alpha * Vn)
            elif self.nonlinearity == "Linear":
                Vn = (1.0 - a) * self.V + a * pre
                feats[t] = Vn
            else:
                raise ValueError(self.nonlinearity)
            self.V = Vn
        return feats


# ───────────────────── ridge ─────────────────────
def ridge_train_cv(X: torch.Tensor, y: torch.Tensor,
                   alphas=(1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
                   ) -> tuple[torch.Tensor, float]:
    n = X.shape[0]
    n_val = max(int(0.2 * n), 50)
    Xtr, Xv = X[:-n_val], X[-n_val:]
    ytr, yv = y[:-n_val], y[-n_val:]
    Xb = torch.cat([Xtr, torch.ones(Xtr.shape[0], 1, device=X.device)], dim=1)
    A0 = Xb.T @ Xb
    b = Xb.T @ ytr
    I = torch.eye(A0.shape[0], device=X.device)
    best = (float("inf"), None, None)
    for a in alphas:
        A = A0 + a * I
        A[-1, -1] = a * 1e-3
        try:
            W = torch.linalg.solve(A, b)
        except Exception:
            continue
        Xvb = torch.cat([Xv, torch.ones(Xv.shape[0], 1, device=X.device)], dim=1)
        yhat = Xvb @ W
        err = float(((yhat - yv) ** 2).mean())
        if err < best[0]:
            best = (err, W, a)
    return best[1], best[2]


def ridge_predict(X: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    Xb = torch.cat([X, torch.ones(X.shape[0], 1, device=X.device)], dim=1)
    return Xb @ W


def nrmse(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    rmse = float(torch.sqrt(((y_true - y_pred) ** 2).mean()))
    return rmse / (float(y_true.std()) + 1e-12)


def nmse(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    mse = float(((y_true - y_pred) ** 2).mean())
    var = float(y_true.var()) + 1e-12
    return mse / var


# ───────────────────── single run ─────────────────────
def build_W(topology: str, N: int, sr: float, seed: int, device):
    if topology == "ER_SPARSE":
        return build_er_sparse(N, p=0.02, sr=sr, seed=seed, device=device)
    elif topology == "MESH_4N":
        return build_mesh_4n(N, sr=sr, seed=seed, device=device)
    else:
        raise ValueError(topology)


def run_condition(task: str, topology: str, nonlinearity: str, seed: int,
                  N: int, sr: float, device) -> dict:
    WASHOUT = 500
    if task == "MG":
        TRAIN, TEST = 4000, 2000
        total = WASHOUT + TRAIN + TEST + 1
        x = mackey_glass(total, tau=17, seed=seed)
        x_norm = (x - x.mean()) / (x.std() + 1e-12)
        u_np = x_norm[:-1]
        y_np = x_norm[1:]
    elif task == "NARMA10":
        TRAIN, TEST = 4000, 2000
        total = WASHOUT + TRAIN + TEST
        u_np, y_np = narma10(total, seed=seed)
    else:
        raise ValueError(task)

    u = torch.tensor(u_np, dtype=torch.float32, device=device)
    y = torch.tensor(y_np, dtype=torch.float32, device=device)

    W = build_W(topology, N, sr, seed, device)
    # NS-RAM volatile regime mapping (z473 R_body=1e7): slow leak.
    leak_n = 0.30  # NSRAM_LUT: moderate volatile leak (per-step retention 0.70)
    leak_l = 0.30  # same leak — only nonlinearity differs
    alpha_n = 2.0
    in_scale = 1.0 if task == "MG" else 2.0

    res = Reservoir(N=N, W=W, nonlinearity=nonlinearity,
                    leak=(leak_n if nonlinearity == "NSRAM_LUT" else leak_l),
                    alpha=alpha_n, input_scale=in_scale,
                    device=device, seed=seed)
    res.reset()
    # warmup
    _ = res.run(u[:32])
    res.reset()

    t0 = time.time()
    if device.type == "cuda":
        torch.cuda.synchronize()
    feats = res.run(u)
    if device.type == "cuda":
        torch.cuda.synchronize()
    wall = time.time() - t0

    feats = feats[WASHOUT:]
    y_use = y[WASHOUT:]
    Xtr = feats[:TRAIN]; ytr = y_use[:TRAIN]
    Xte = feats[TRAIN:TRAIN + TEST]; yte = y_use[TRAIN:TRAIN + TEST]
    W_read, alpha_chosen = ridge_train_cv(Xtr, ytr)
    yhat = ridge_predict(Xte, W_read)

    if task == "MG":
        err_test = nrmse(yte, yhat)
        err_metric = "NRMSE"
    else:
        err_test = nmse(yte, yhat)
        err_metric = "NMSE"

    return {
        "task": task, "topology": topology, "nonlinearity": nonlinearity,
        "seed": seed, "N": N, "sr": sr,
        "err_metric": err_metric, "err_test": err_test,
        "ridge_alpha": float(alpha_chosen) if alpha_chosen is not None else None,
        "wall_s": wall,
        "throughput_steps_per_sec": float(u.shape[0] / wall),
    }


# ───────────────────── ESN sanity baseline ─────────────────────
def esn_sanity(task: str, seed: int, device) -> dict:
    """Jaeger 2001 ESN: tanh, N=200, sparse 0.05, sr=0.95, leak=0.3."""
    N = 200
    WASHOUT = 500
    if task == "MG":
        TRAIN, TEST = 4000, 2000
        total = WASHOUT + TRAIN + TEST + 1
        x = mackey_glass(total, tau=17, seed=seed)
        x_norm = (x - x.mean()) / (x.std() + 1e-12)
        u_np = x_norm[:-1]; y_np = x_norm[1:]
    else:
        TRAIN, TEST = 4000, 2000
        total = WASHOUT + TRAIN + TEST
        u_np, y_np = narma10(total, seed=seed)
    u = torch.tensor(u_np, dtype=torch.float32, device=device)
    y = torch.tensor(y_np, dtype=torch.float32, device=device)
    W = build_er_sparse(N, p=0.05, sr=0.95, seed=seed, device=device)
    res = Reservoir(N=N, W=W, nonlinearity="NSRAM_LUT", leak=0.30, alpha=1.0,
                    input_scale=(1.0 if task == "MG" else 2.0),
                    device=device, seed=seed)
    res.reset(); _ = res.run(u[:32]); res.reset()
    feats = res.run(u)
    feats = feats[WASHOUT:]; y_use = y[WASHOUT:]
    Xtr = feats[:TRAIN]; ytr = y_use[:TRAIN]
    Xte = feats[TRAIN:TRAIN + TEST]; yte = y_use[TRAIN:TRAIN + TEST]
    W_read, _ = ridge_train_cv(Xtr, ytr)
    yhat = ridge_predict(Xte, W_read)
    err = nrmse(yte, yhat) if task == "MG" else nmse(yte, yhat)
    return {"task": task, "seed": seed, "err": err}


# ───────────────────── main ─────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[killshot] device={device}  torch={torch.__version__}")

    N = 1024
    SR = 0.9
    SEEDS = [0, 1, 2]
    TOPOS = ["ER_SPARSE", "MESH_4N"]
    NONLINS = ["NSRAM_LUT", "Linear"]
    TASKS = ["MG", "NARMA10"]

    results = []
    t_start = time.time()
    for task in TASKS:
        for topo in TOPOS:
            for nonlin in NONLINS:
                for seed in SEEDS:
                    print(f"[killshot] {task} {topo} {nonlin} seed={seed} ...",
                          flush=True)
                    r = run_condition(task, topo, nonlin, seed, N, SR, device)
                    print(f"           {r['err_metric']}={r['err_test']:.5f}  "
                          f"thr={r['throughput_steps_per_sec']:.0f} st/s  "
                          f"wall={r['wall_s']:.1f}s", flush=True)
                    results.append(r)

    # ESN sanity baselines
    print("[killshot] running ESN sanity baselines (Jaeger 2001) ...")
    esn = []
    for task in TASKS:
        for seed in SEEDS:
            e = esn_sanity(task, seed, device)
            print(f"  ESN {task} seed={seed}: {e['err']:.5f}")
            esn.append(e)

    t_total = time.time() - t_start
    print(f"[killshot] all conditions done in {t_total:.1f}s")

    # ───────── analysis ─────────
    def median_over(task, topo, nonlin):
        vals = [r["err_test"] for r in results
                if r["task"] == task and r["topology"] == topo
                and r["nonlinearity"] == nonlin]
        return float(np.median(vals)), vals

    summary = {}
    for task in TASKS:
        summary[task] = {}
        for topo in TOPOS:
            for nonlin in NONLINS:
                m, vs = median_over(task, topo, nonlin)
                summary[task][f"{topo}|{nonlin}"] = {"median": m, "seeds": vs}

    # (1) topology advantage: ER ≤ 0.95 × MESH on both tasks (median across BOTH nonlins,
    #     per-nonlinearity, and combined). We pre-register "median across BOTH nonlinearities"
    #     to be the most general statement (oracle wording: "median_ER ≤ 0.95 × median_MESH on both tasks").
    topology_adv = {}
    for task in TASKS:
        er_all = [r["err_test"] for r in results
                  if r["task"] == task and r["topology"] == "ER_SPARSE"]
        mesh_all = [r["err_test"] for r in results
                    if r["task"] == task and r["topology"] == "MESH_4N"]
        med_er = float(np.median(er_all))
        med_mesh = float(np.median(mesh_all))
        topology_adv[task] = {
            "median_ER": med_er,
            "median_MESH": med_mesh,
            "ratio_ER_over_MESH": med_er / (med_mesh + 1e-12),
            "threshold": 0.95,
            "pass": med_er <= 0.95 * med_mesh,
        }
    topology_pass = all(topology_adv[t]["pass"] for t in TASKS)

    # (2) nonlinearity benefit: median_Linear ≥ 1.10 × median_NSRAM_LUT on both tasks
    #     (median across both topologies)
    nonlin_ben = {}
    for task in TASKS:
        lin_all = [r["err_test"] for r in results
                   if r["task"] == task and r["nonlinearity"] == "Linear"]
        nsr_all = [r["err_test"] for r in results
                   if r["task"] == task and r["nonlinearity"] == "NSRAM_LUT"]
        med_lin = float(np.median(lin_all))
        med_nsr = float(np.median(nsr_all))
        nonlin_ben[task] = {
            "median_Linear": med_lin,
            "median_NSRAM_LUT": med_nsr,
            "ratio_Linear_over_NSRAM": med_lin / (med_nsr + 1e-12),
            "threshold": 1.10,
            "pass": med_lin >= 1.10 * med_nsr,
        }
    nonlin_pass = all(nonlin_ben[t]["pass"] for t in TASKS)

    # (3) absolute sanity: ER_SPARSE+NSRAM_LUT MG NRMSE ≤ 0.05 AND NARMA-10 NMSE ≤ 0.15
    er_nsr_mg, _ = median_over("MG", "ER_SPARSE", "NSRAM_LUT")
    er_nsr_n10, _ = median_over("NARMA10", "ER_SPARSE", "NSRAM_LUT")
    sanity = {
        "median_MG_NRMSE_ER_NSRAM": er_nsr_mg,
        "median_NARMA10_NMSE_ER_NSRAM": er_nsr_n10,
        "threshold_MG": 0.05,
        "threshold_NARMA10": 0.15,
        "MG_pass": er_nsr_mg <= 0.05,
        "NARMA10_pass": er_nsr_n10 <= 0.15,
        "pass": (er_nsr_mg <= 0.05) and (er_nsr_n10 <= 0.15),
    }
    sanity_pass = sanity["pass"]

    fail_count = int(not topology_pass) + int(not nonlin_pass) + int(not sanity_pass)
    if fail_count == 0:
        verdict = "ALL_PASS (claim survives O81 killshot)"
    elif fail_count == 1:
        verdict = "PARTIAL_FAIL (1/3 thresholds failed)"
    elif fail_count == 2:
        verdict = "KILLSHOT_TRIGGERED (2/3 failed — O81 falsifier hit)"
    else:
        verdict = "CATASTROPHIC (3/3 failed)"

    # write artifacts
    (OUT / "grid_results.json").write_text(json.dumps({
        "grid": results,
        "summary_medians": summary,
        "esn_sanity": esn,
        "wall_total_s": t_total,
        "config": {"N": N, "SR": SR, "seeds": SEEDS, "tasks": TASKS,
                   "topologies": TOPOS, "nonlinearities": NONLINS,
                   "er_density_p": 0.02, "mesh_kind": "4N_toroidal",
                   "nsram_lut_mapping": "R_body=1e7 z473 volatile (leak=0.30, alpha=2.0)"},
    }, indent=2))
    (OUT / "topology_advantage.json").write_text(json.dumps({
        "tasks": topology_adv, "overall_pass": topology_pass,
    }, indent=2))
    (OUT / "nonlinearity_benefit.json").write_text(json.dumps({
        "tasks": nonlin_ben, "overall_pass": nonlin_pass,
    }, indent=2))
    (OUT / "absolute_sanity.json").write_text(json.dumps(sanity, indent=2))

    # honest_analysis.md
    lines = [
        "# ERvMESH_killshot_1024_LUT — O81 oracle pre-registered falsifier",
        "",
        "## Config",
        f"- N={N}, spectral radius α={SR}, seeds={SEEDS}",
        "- Topologies: ER_SPARSE (p=0.02), MESH_4N (32×32 toroidal 4-nearest)",
        "- Nonlinearities: NSRAM_LUT (z473 R_body=1e7 volatile surrogate, tanh α=2.0, leak=0.30),"
        " Linear (identity, leak=0.30)",
        "- Tasks: Mackey-Glass τ=17 (NRMSE), NARMA-10 (NMSE)",
        "- Schedule: washout=500, train=4000, test=2000 (real held-out test split, last 33% after washout)",
        f"- Wall time: {t_total:.1f}s on {device}",
        "",
        "## Headline medians (lower = better)",
        "| Task | Topology | Nonlinearity | Median | Seeds |",
        "|------|----------|--------------|--------|-------|",
    ]
    for task in TASKS:
        for topo in TOPOS:
            for nonlin in NONLINS:
                m, vs = median_over(task, topo, nonlin)
                vs_str = ", ".join(f"{v:.4f}" for v in vs)
                lines.append(f"| {task} | {topo} | {nonlin} | {m:.5f} | {vs_str} |")

    lines += [
        "",
        "## ESN sanity (Jaeger 2001: N=200, p=0.05, sr=0.95, tanh, leak=0.3)",
    ]
    for e in esn:
        lines.append(f"- {e['task']} seed={e['seed']}: err={e['err']:.5f}")

    lines += [
        "",
        "## Pre-registered thresholds",
        "",
        "### (1) Topology advantage: median_ER ≤ 0.95 × median_MESH on BOTH tasks",
    ]
    for task in TASKS:
        a = topology_adv[task]
        lines.append(
            f"- {task}: median_ER={a['median_ER']:.5f}, median_MESH={a['median_MESH']:.5f},"
            f" ratio={a['ratio_ER_over_MESH']:.3f} (need ≤0.95) — "
            f"{'PASS' if a['pass'] else 'FAIL'}"
        )
    lines.append(f"- **Overall topology gate: {'PASS' if topology_pass else 'FAIL'}**")

    lines += [
        "",
        "### (2) Nonlinearity benefit: median_Linear ≥ 1.10 × median_NSRAM_LUT on BOTH tasks",
    ]
    for task in TASKS:
        a = nonlin_ben[task]
        lines.append(
            f"- {task}: median_Linear={a['median_Linear']:.5f},"
            f" median_NSRAM_LUT={a['median_NSRAM_LUT']:.5f},"
            f" ratio={a['ratio_Linear_over_NSRAM']:.3f} (need ≥1.10) — "
            f"{'PASS' if a['pass'] else 'FAIL'}"
        )
    lines.append(f"- **Overall nonlinearity gate: {'PASS' if nonlin_pass else 'FAIL'}**")

    lines += [
        "",
        "### (3) Absolute sanity: ER_SPARSE+NSRAM_LUT must hit headline error budgets",
        f"- MG NRMSE = {sanity['median_MG_NRMSE_ER_NSRAM']:.5f} (need ≤0.05) — "
        f"{'PASS' if sanity['MG_pass'] else 'FAIL'}",
        f"- NARMA-10 NMSE = {sanity['median_NARMA10_NMSE_ER_NSRAM']:.5f} (need ≤0.15) — "
        f"{'PASS' if sanity['NARMA10_pass'] else 'FAIL'}",
        f"- **Overall sanity gate: {'PASS' if sanity_pass else 'FAIL'}**",
        "",
        "## VERDICT",
        f"- FAIL_count = {fail_count}/3",
        f"- **{verdict}**",
        "",
        "## Honest interpretation",
    ]
    if fail_count == 0:
        lines.append(
            "Claim survives. ER_SPARSE beats MESH_4N AND NS-RAM nonlinearity helps over linear "
            "AND absolute sanity holds. The O81 falsifier did NOT fire."
        )
    else:
        flags = []
        if not topology_pass:
            flags.append("ER_SPARSE does NOT beat MESH_4N at the 5% margin — the topology-win "
                         "claim is not supported by this 1024-cell, α=0.9, ridge-readout grid.")
        if not nonlin_pass:
            flags.append("NS-RAM nonlinearity does NOT outperform Linear by ≥10% — the "
                         "'reservoir nonlinearity is constitutive' claim is not supported here.")
        if not sanity_pass:
            flags.append("Absolute headline budgets MG NRMSE≤0.05 / NARMA-10 NMSE≤0.15 not met "
                         "by ER_SPARSE+NSRAM_LUT — the BEST cell of the matrix did not clear the bar.")
        for f in flags:
            lines.append(f"- {f}")
        if fail_count >= 2:
            lines.append(
                "\nO81 oracle Q2 falsifier is TRIGGERED. The 'multi-function primitive + "
                "topology-advantage' claim must be retracted or substantially restated."
            )

    (OUT / "honest_analysis.md").write_text("\n".join(lines))

    # pareto plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        for ax, task in zip(axes, TASKS):
            xs, ys, labels, colors = [], [], [], []
            for topo in TOPOS:
                for nonlin in NONLINS:
                    vs = [r["err_test"] for r in results
                          if r["task"] == task and r["topology"] == topo
                          and r["nonlinearity"] == nonlin]
                    for v in vs:
                        xs.append(f"{topo}\n{nonlin}")
                        ys.append(v)
            # boxplot grouping
            groups = {}
            for x, y in zip(xs, ys):
                groups.setdefault(x, []).append(y)
            keys = list(groups.keys())
            data = [groups[k] for k in keys]
            ax.boxplot(data, labels=keys, showmeans=True)
            ax.set_title(f"{task} — error (lower = better)")
            ax.set_ylabel("NRMSE" if task == "MG" else "NMSE")
            ax.grid(True, alpha=0.3)
            # threshold line
            if task == "MG":
                ax.axhline(0.05, color="red", linestyle="--", label="sanity 0.05")
            else:
                ax.axhline(0.15, color="red", linestyle="--", label="sanity 0.15")
            ax.legend()
        fig.suptitle(f"ERvMESH killshot — {verdict}")
        fig.tight_layout()
        fig.savefig(OUT / "pareto_per_condition.png", dpi=130)
        plt.close(fig)
    except Exception as e:
        print(f"[killshot] plot skipped: {e}")

    print("\n========================================")
    print(f"VERDICT: {verdict}")
    print(f"  topology_pass = {topology_pass}")
    print(f"  nonlinearity_pass = {nonlin_pass}")
    print(f"  sanity_pass = {sanity_pass}")
    print(f"  FAIL_count = {fail_count}/3")
    print(f"Artifacts written to: {OUT}")
    print("========================================")


if __name__ == "__main__":
    main()
