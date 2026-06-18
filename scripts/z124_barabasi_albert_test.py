"""z124 — scale-free Barabási-Albert topology test (Gemini O13 follow-up).

Gemini 2.5-pro's O13 oracle review flagged scale-free networks
(Barabási-Albert) as an interesting topology we had not tested.
At the time we deferred it as "future work". This script closes
that gap.

Setup:
  - Topology: BA scale-free with attachment parameter m=4 (each
    new node connects to 4 existing nodes via preferential
    attachment). Average degree ≈ 2m = 8 ≪ ER's pN ≈ 20 at p=0.1.
  - Baseline comparators: RAND_GAUSS, ER_SPARSE (z121 reference).
  - Canonical 1/√k coupling normalization (k = average degree
    per topology, so spectral mass matches).
  - N = 200, T = 600, κ = 0.03, 5 seeds.
  - Per-task ridge sweep (z121 protocol).

Three possible outcomes:
  1. **BA wins:** scale-free hubs help MC/XOR more than ER's
     uniform sparse. C.3 v2.3 should consider scale-free as a
     primary alternative.
  2. **BA ties ER_SPARSE:** both expander-class graphs work
     equally well; the recommendation is robust to the specific
     sparse topology.
  3. **BA loses:** hub-driven dynamics introduce too much
     coupling-strength heterogeneity; ER's uniform-sparse is
     the intended target.

Wall budget: 3 topologies × 5 seeds × ~50s = ~12-13 min.
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import torch
torch.set_default_dtype(torch.float64)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/z124_barabasi_albert_test"
OUT.mkdir(parents=True, exist_ok=True)

import importlib.util
sp_v1 = importlib.util.spec_from_file_location("v1", ROOT / "scripts/z96_narma10_pilot.py")
v1 = importlib.util.module_from_spec(sp_v1); sp_v1.loader.exec_module(v1)
sp_z121 = importlib.util.spec_from_file_location("z121", ROOT / "scripts/z121_mc_xor_canonical_wrec.py")
z121 = importlib.util.module_from_spec(sp_z121); sp_z121.loader.exec_module(z121)

from nsram.bsim4_port.nsram_cell_2T import NSRAMCell2TConfig
from nsram.bsim4_port.bjt import GummelPoonNPN
from nsram.bsim4_port.vectorized import forward_2t_batched


def build_W_canonical(topology, N, rng):
    """Build W_rec normalized so total spectral mass matches across topologies."""
    if topology == "RAND_GAUSS":
        return rng.normal(0.0, 1.0/np.sqrt(N), size=(N, N))
    elif topology == "ER_SPARSE":
        p = 0.1
        mask = rng.random((N, N)) < p
        return np.where(mask, rng.normal(0.0, 1.0/np.sqrt(N*p), size=(N, N)), 0.0)
    elif topology == "BA_SCALEFREE":
        # Barabási-Albert preferential attachment, m=4.
        m = 4
        adj = np.zeros((N, N), dtype=bool)
        # Initial fully-connected core of m+1 nodes
        for i in range(m+1):
            for j in range(m+1):
                if i != j:
                    adj[i, j] = True
        # Preferential attachment for nodes m+1 .. N-1
        for i in range(m+1, N):
            degrees = adj.sum(axis=1)
            # Probability ∝ degree
            probs = degrees[:i] / max(degrees[:i].sum(), 1)
            targets = rng.choice(i, size=m, replace=False, p=probs)
            for t in targets:
                adj[i, t] = True
                adj[t, i] = True
        # Average degree ≈ 2m = 8; per-edge std = 1/√(<k>·N) for spectral
        # mass match to ER and RAND
        avg_deg = float(adj.sum() / N)
        sigma = 1.0 / np.sqrt(avg_deg * N)
        W = np.where(adj, rng.normal(0.0, sigma, size=(N, N)), 0.0)
        return W
    raise ValueError(topology)


def run_cell_sim(topology, N, T, kappa, seed):
    rng = np.random.default_rng(seed)
    u_int = rng.integers(0, 2, size=T)
    u = 2.0 * u_int - 1.0
    cfg = NSRAMCell2TConfig(use_iii=True, use_gidl=True, use_bjt=True, newton_max_iters=50)
    M1, M2 = v1.build_calibrated_models()
    bjt = GummelPoonNPN.from_sebas_card(); bjt.Bf = 5e4
    base_VG1 = torch.tensor(rng.choice([0.2, 0.4, 0.6], size=N), dtype=torch.float64)
    base_VG2 = torch.tensor(rng.uniform(0.0, 0.5, size=N), dtype=torch.float64)
    W_rec_np = build_W_canonical(topology, N, rng)
    W_rec = torch.tensor(W_rec_np, dtype=torch.float64)
    feat_prev = torch.zeros(N, dtype=torch.float64)
    log_Id = np.zeros((N, T))
    for t in range(T):
        Vd_t = torch.tensor([1.0 + 0.5*float(u[t])], dtype=torch.float64)
        recur = (W_rec @ feat_prev) if kappa > 0 else torch.zeros(N, dtype=torch.float64)
        VG2_eff = (base_VG2 + kappa*recur).clamp(-0.2, 1.0)
        out = forward_2t_batched(cfg, M1, M2, bjt, Vd_t, base_VG1, VG2_eff,
                                   max_iters=15, tol=1e-9, verbose=False)
        log_Id[:, t] = np.log10(np.maximum(out["Id"].abs().squeeze().numpy(), 1e-15))
        feat_prev = torch.tensor(log_Id[:, t], dtype=torch.float64)
    # Compute spectral and degree metadata
    deg_mean = float((W_rec_np != 0).sum() / N)
    eigs = np.linalg.eigvals(W_rec_np)
    rho = float(np.max(np.abs(eigs)))
    return log_Id, u_int, u, {"avg_degree": deg_mean, "spectral_radius": rho}


def main():
    t0 = time.time()
    topologies = ["RAND_GAUSS", "ER_SPARSE", "BA_SCALEFREE"]
    seeds = [42, 43, 44, 45, 46]
    ridges = [1e-3, 1e-1, 1e+1, 1e+3]
    N = 200; T = 600; kappa = 0.03
    print(f"[z124] scale-free Barabási-Albert vs ER_SPARSE vs RAND_GAUSS")
    print(f"  N={N}, T={T}, κ={kappa}, canonical 1/√(<k>·N) scaling")
    print(f"  Reference (z121): RAND MC=0.90, ER_SPARSE MC=2.90, XOR 0.59 vs 0.82\n")

    grid_mc = np.zeros((len(topologies), len(seeds)))
    grid_xor = np.zeros((len(topologies), len(seeds)))
    spectral = {topo: [] for topo in topologies}
    avg_deg = {topo: [] for topo in topologies}

    for i, topo in enumerate(topologies):
        for j, seed in enumerate(seeds):
            ti = time.time()
            log_Id, u_int, u, meta = run_cell_sim(topo, N, T, kappa, seed)
            sim_t = time.time() - ti
            MC, _ = z121.eval_MC(log_Id, u, T, ridges)
            XOR, _ = z121.eval_XOR(log_Id, u_int, tau=2, T=T, ridges=ridges)
            grid_mc[i, j] = MC
            grid_xor[i, j] = XOR
            spectral[topo].append(meta["spectral_radius"])
            avg_deg[topo].append(meta["avg_degree"])
            print(f"  {topo:13s} s={seed}  sim={sim_t:.0f}s  MC={MC:5.2f}  "
                   f"XOR={XOR:.2f}  ⟨k⟩={meta['avg_degree']:.1f}  ρ={meta['spectral_radius']:.3f}",
                   flush=True)

    print(f"\n[z124] === Aggregated ({len(seeds)} seeds, canonical 1/√(<k>·N)) ===")
    print(f"  {'topo':13s} {'⟨k⟩':>5s} {'ρ':>6s}  {'MC mean':>9s} {'± std':>7s}  "
           f"{'XOR mean':>9s} {'± std':>7s}")
    means_mc = grid_mc.mean(axis=1); stds_mc = grid_mc.std(axis=1, ddof=1)
    means_xor = grid_xor.mean(axis=1); stds_xor = grid_xor.std(axis=1, ddof=1)
    for i, topo in enumerate(topologies):
        ak = float(np.mean(avg_deg[topo]))
        ar = float(np.mean(spectral[topo]))
        print(f"  {topo:13s} {ak:>5.1f} {ar:>6.3f}  {means_mc[i]:>9.3f} {stds_mc[i]:>7.3f}  "
               f"{means_xor[i]:>9.3f} {stds_xor[i]:>7.3f}")

    # Paired-t per (topo - RAND_GAUSS)
    print(f"\n[z124] === Paired-t vs RAND_GAUSS baseline ===")
    for name, grid in [("MC", grid_mc), ("XOR", grid_xor)]:
        for i in range(1, len(topologies)):
            diffs = grid[i] - grid[0]
            d_mean = diffs.mean()
            d_sem = diffs.std(ddof=1) / np.sqrt(len(seeds))
            t_stat = d_mean / d_sem if d_sem > 1e-9 else float("inf")
            print(f"  {name} {topologies[i]:14s} - RAND_GAUSS: "
                   f"Δ={d_mean:+.3f}  SEM={d_sem:.3f}  t={t_stat:+.2f}")

    # Verdict
    print(f"\n[z124] === Verdict ===")
    er_mc, ba_mc = means_mc[1], means_mc[2]
    er_xor, ba_xor = means_xor[1], means_xor[2]
    print(f"  ER_SPARSE: MC={er_mc:.2f}, XOR={er_xor:.2f}")
    print(f"  BA_SCALEFREE: MC={ba_mc:.2f}, XOR={ba_xor:.2f}")
    if abs(ba_mc - er_mc) < 0.5 and abs(ba_xor - er_xor) < 0.05:
        print(f"  → BA ≈ ER_SPARSE; recommendation robust to specific sparse topology.")
    elif ba_mc > er_mc + 0.5:
        print(f"  → BA WINS; scale-free hubs benefit MC/XOR more than uniform sparse.")
    elif ba_mc < er_mc - 0.5:
        print(f"  → BA LOSES; hub-driven heterogeneity hurts; ER uniform sparse confirmed.")
    else:
        print(f"  → mixed result; see paired-t for significance.")

    json.dump({"topologies": topologies, "seeds": seeds, "ridges": ridges,
                "N": N, "T": T, "kappa": kappa,
                "MC_grid": grid_mc.tolist(), "XOR_grid": grid_xor.tolist(),
                "MC_means": means_mc.tolist(), "MC_stds": stds_mc.tolist(),
                "XOR_means": means_xor.tolist(), "XOR_stds": stds_xor.tolist(),
                "avg_degree": avg_deg, "spectral_radius": spectral},
               (OUT / "summary.json").open("w"), indent=2)
    print(f"\n[z124] saved {OUT}/summary.json")
    print(f"[z124] wall: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
