#!/usr/bin/env python3
"""J3 v5.5 topology zoo — replication-quality wrapper around DS_N13.

Plan (research_plan/LARGE_SCALE_CAMPAIGN_2026-05-21.md):
  topologies = ER_SPARSE, MESH_4N, SMALL_WORLD, RING
  N = 10000
  seeds = 0..9 (10)
  rho-norm conventions = {"none", "spectral_op", "row_l1"}
  v5.5 surrogate: results/SURROGATE_v55_zgx/surrogate_v55.npz

Acceptance: ER_SPARSE wins in >= 2/3 rho-norm conventions => CONFIRMS brief.

Reuses DS_N13 building blocks (run_nsram_network, ridge_predict, mackey_glass,
build_topology) so the physics is identical; we only change the W normalization
and aggregate over more seeds × topologies × rho conventions.

NOTE: The v5.5 surrogate is consumed *inside* NSRAMArray when present; we do
not need to load the npz here. If users want strict v5.5 path, ensure
DS_N11_predictive_coding.NSRAMArray points to v5.5 (already does on zgx where
SURROGATE_v55_zgx/ exists).
"""
from __future__ import annotations
import os, sys, json, time
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

# Import DS_N13 primitives by file (it's a script, not a module)
import importlib.util
_n13_spec = importlib.util.spec_from_file_location(
    "ds_n13", REPO / "scripts" / "DS_N13_topology_zoo.py")
ds_n13 = importlib.util.module_from_spec(_n13_spec)
_n13_spec.loader.exec_module(ds_n13)

OUT = REPO / "results" / "TOPOLOGY_ZOO_v55_2026-05-21"
OUT.mkdir(parents=True, exist_ok=True)

# Map plan names → DS_N13 builder names
TOPO_MAP = {
    "ER_SPARSE":    "er",
    "MESH_4N":      "lattice2d",   # 2D toroidal lattice, 8-connected (best DS_N13 proxy for MESH)
    "SMALL_WORLD":  "small_world",
    "RING":         "ring",
}


def rho_normalize(W: sp.csr_matrix, mode: str, target: float = 0.9) -> sp.csr_matrix:
    """Rescale W so its spectral structure matches `mode`.
    - "none":         no rescale (legacy DS_N13 behaviour)
    - "spectral_op":  rescale so largest |eigenvalue| (via ARPACK) == target
    - "row_l1":       rescale so max row L1 norm == target
    """
    if mode == "none":
        return W
    if mode == "row_l1":
        absW = abs(W)
        row_sum = np.asarray(absW.sum(axis=1)).ravel()
        m = float(row_sum.max())
        if m <= 0:
            return W
        return W.multiply(target / m).tocsr()
    if mode == "spectral_op":
        try:
            ev = spla.eigs(W.astype(np.float64), k=1, which="LM",
                           maxiter=200, tol=1e-3, return_eigenvectors=False)
            rho = float(abs(ev[0]))
        except Exception:
            # fallback: power iteration on W^T W eig sqrt (operator norm)
            rho = float(spla.norm(W, ord=2)) if hasattr(spla, "norm") else 1.0
        if rho <= 0:
            return W
        return W.multiply(target / rho).tocsr()
    raise ValueError(mode)


def run_one(topo_name: str, builder: str, N: int, seed: int, rho_mode: str,
            signal: np.ndarray, density: float = 0.01):
    t0 = time.time()
    W0 = ds_n13.build_topology(builder, N=N, seed=seed, density=density)
    W = rho_normalize(W0, rho_mode, target=0.9)
    feats, n_spk, _ = ds_n13.run_nsram_network(
        signal, W, N=N, dt=1e-6, sub_per_sample=1, seed=seed)
    div = bool(np.any(~np.isfinite(feats))) or float(np.max(np.abs(feats))) > 1e6
    if div:
        nrmse = float("nan")
    else:
        nrmse, _, _ = ds_n13.ridge_predict(feats, signal)
    return {
        "topo": topo_name, "builder": builder, "rho_mode": rho_mode,
        "seed": int(seed), "N": int(N), "nrmse": float(nrmse),
        "spikes": int(n_spk), "nnz": int(W.nnz),
        "diverged": bool(div), "wall_s": float(time.time() - t0),
    }


def main():
    N = 10_000
    T = 10_000
    SEEDS = list(range(10))
    RHO_MODES = ["none", "spectral_op", "row_l1"]
    DENSITY = 0.01

    signal = ds_n13.mackey_glass(T, tau=17, seed=42)
    print(f"[J3v55] N={N} T={T} seeds={SEEDS} rho_modes={RHO_MODES}", flush=True)
    print(f"[J3v55] topologies={list(TOPO_MAP)}", flush=True)

    rows = []
    progress_path = OUT / "progress.json"
    for rho_mode in RHO_MODES:
        for topo_name, builder in TOPO_MAP.items():
            seed_nrmses = []
            for seed in SEEDS:
                rec = run_one(topo_name, builder, N, seed, rho_mode, signal, DENSITY)
                rows.append(rec)
                seed_nrmses.append(rec["nrmse"])
                print(f"  [{rho_mode:11s} {topo_name:11s} s={seed}] "
                      f"nrmse={rec['nrmse']:.4f} spk={rec['spikes']} "
                      f"nnz={rec['nnz']} t={rec['wall_s']:.1f}s "
                      f"div={rec['diverged']}", flush=True)
                # incremental save
                progress_path.write_text(json.dumps({"rows": rows}, indent=2))
            arr = np.array([r for r in seed_nrmses if np.isfinite(r)])
            print(f"  → {rho_mode:11s} {topo_name:11s} "
                  f"mean={arr.mean():.4f} std={arr.std():.4f} "
                  f"(n_finite={len(arr)}/{len(SEEDS)})", flush=True)

    # aggregate
    summary = {}
    for rho_mode in RHO_MODES:
        summary[rho_mode] = {}
        for topo_name in TOPO_MAP:
            vals = [r["nrmse"] for r in rows
                    if r["rho_mode"] == rho_mode and r["topo"] == topo_name
                    and np.isfinite(r["nrmse"])]
            if not vals:
                summary[rho_mode][topo_name] = {"mean": None, "std": None, "n": 0}
                continue
            v = np.array(vals)
            summary[rho_mode][topo_name] = {"mean": float(v.mean()),
                                             "std": float(v.std()), "n": int(v.size)}

    # winner per rho_mode (lowest mean NRMSE)
    winners = {}
    for rho_mode in RHO_MODES:
        finite = {k: v for k, v in summary[rho_mode].items() if v["mean"] is not None}
        if finite:
            w = min(finite, key=lambda k: finite[k]["mean"])
            winners[rho_mode] = w

    er_wins = sum(1 for w in winners.values() if w == "ER_SPARSE")
    gate_pass = er_wins >= 2

    final = {
        "config": {"N": N, "T": T, "seeds": SEEDS, "rho_modes": RHO_MODES,
                   "density": DENSITY, "topologies": list(TOPO_MAP)},
        "rows": rows,
        "summary": summary,
        "winners": winners,
        "ER_SPARSE_wins": er_wins,
        "gate_ER_SPARSE_>=2of3_rho": bool(gate_pass),
    }
    (OUT / "results_final.json").write_text(json.dumps(final, indent=2))
    print(f"[J3v55] saved {OUT/'results_final.json'}", flush=True)
    print(f"[J3v55] winners: {winners}  ER wins {er_wins}/3  "
          f"GATE={'PASS' if gate_pass else 'FAIL'}", flush=True)


if __name__ == "__main__":
    main()
