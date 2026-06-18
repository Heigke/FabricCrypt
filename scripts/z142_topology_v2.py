"""z142 — Topology v2: η-bounded cell + fair ρ + 5 seeds (M3b F3).

Builds on z139, addresses three O18+O19 critiques:

  1. Uses the M3b F1.v2 η-bounded cell (η ∈ [0, 1] sigmoid Iii→Vb
     injection, Bf=100 physical) — the validated honest model.
  2. Adds two fair ρ-normalization variants for hub-dominated graphs:
     rho_lambda  (current — λmax scaling, used by z139)
     rho_p95_sv  (95th-percentile singular-value scaling)
     rho_deg_norm (degree-normalized D⁻¹ᐟ²·W·D⁻¹ᐟ² then λmax)
  3. Bumps seeds to 5 (z139 had 3, only 2 valid after ridge bug).

Total: 6 topos × 3 N values × 3 ρ-norms × 5 seeds × 4 tasks = 270
sims. At ~90 s/sim → ~7 h wall. Designed to run overnight.
"""
from __future__ import annotations
import json, os, time
from pathlib import Path
import numpy as np
import torch
torch.set_default_dtype(torch.float64)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/z142_topology_v2"
OUT.mkdir(parents=True, exist_ok=True)

import importlib.util
sp = importlib.util.spec_from_file_location("v1", ROOT / "scripts/z96_narma10_pilot.py")
v1 = importlib.util.module_from_spec(sp); sp.loader.exec_module(v1)
sp2 = importlib.util.spec_from_file_location("z119", ROOT / "scripts/z119_topology_sweep.py")
z119 = importlib.util.module_from_spec(sp2); sp2.loader.exec_module(z119)
sp3 = importlib.util.spec_from_file_location("z139", ROOT / "scripts/z139_largescale_topology.py")
z139 = importlib.util.module_from_spec(sp3); sp3.loader.exec_module(z139)

from nsram.bsim4_port.nsram_cell_2T import NSRAMCell2TConfig
from nsram.bsim4_port.bjt import GummelPoonNPN
from nsram.bsim4_port.vectorized import forward_2t_batched

BF_PHYSICAL = 100.0   # M3b F1.v2 honest physical bound


# ---------------------------------------------------------------- ρ-norm variants

def rho_normalize(W: np.ndarray, rho: float, variant: str) -> np.ndarray:
    """Three ρ-normalization strategies for fair cross-topology comparison."""
    if variant == "rho_lambda":
        eig = np.linalg.eigvals(W)
        scale = float(np.max(np.abs(eig)))
    elif variant == "rho_p95_sv":
        # 95th percentile singular value — robust to hub-dominated spectra
        sv = np.linalg.svd(W, compute_uv=False)
        scale = float(np.percentile(sv, 95))
    elif variant == "rho_deg_norm":
        # Degree-normalized D^(-1/2) W D^(-1/2) then λmax
        deg = np.abs(W).sum(axis=1)
        deg_safe = np.where(deg > 1e-9, deg, 1.0)
        D_inv_sqrt = 1.0 / np.sqrt(deg_safe)
        W_norm = W * D_inv_sqrt[:, None] * D_inv_sqrt[None, :]
        eig = np.linalg.eigvals(W_norm)
        scale = float(np.max(np.abs(eig)))
    else:
        raise ValueError(f"unknown ρ variant: {variant}")
    return W * (rho / max(scale, 1e-9))


def build_W_v2(topology: str, N: int, rho: float, rng, rho_variant: str):
    """Builds W using z139's topology generators, applies fair-ρ normalization."""
    # Build raw W without z139's λ-normalize (we re-normalize)
    if topology == "RAND_GAUSS":
        W = rng.normal(0.0, 1.0/np.sqrt(N), size=(N, N))
    elif topology == "MESH_4N":
        W = np.zeros((N, N))
        side = int(np.ceil(np.sqrt(N)))
        coords = [(i, j) for i in range(side) for j in range(side)][:N]
        idx = {c: k for k, c in enumerate(coords)}
        for k, (i, j) in enumerate(coords):
            for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                ni, nj = (i+di) % side, (j+dj) % side
                if (ni, nj) in idx:
                    W[k, idx[(ni, nj)]] = rng.normal(0.0, 1.0)
    elif topology == "ER_SPARSE":
        p = 0.1
        mask = rng.random((N, N)) < p
        W = np.where(mask, rng.normal(0.0, 1.0, size=(N, N)), 0.0)
    elif topology == "WS_SMALLWORLD":
        k, beta = 4, 0.1
        W = np.zeros((N, N))
        for i in range(N):
            for off in range(1, k//2 + 1):
                for sign in (-1, +1):
                    j = (i + sign*off) % N
                    if rng.random() < beta:
                        j = int(rng.integers(0, N))
                        if j == i: continue
                    W[i, j] = rng.normal(0.0, 1.0)
    elif topology == "HUB_SPOKE":
        W = np.zeros((N, N))
        leaves = np.arange(1, N)
        W[0, leaves] = rng.normal(0, 1.0, size=N-1)
        W[leaves, 0] = rng.normal(0, 1.0, size=N-1)
        mask = rng.random((N-1, N-1)) < 0.02
        np.fill_diagonal(mask, False)
        W[1:, 1:] = np.where(mask, rng.normal(0, 0.5, size=(N-1, N-1)), 0.0)
    elif topology == "LAYERED":
        N1 = N // 2; N2 = N - N1
        W = np.zeros((N, N))
        W[N1:, :N1] = rng.normal(0, 1.0/np.sqrt(N1), size=(N2, N1))
        skip_mask = rng.random((N1, N2)) < 0.10
        W[:N1, N1:] = np.where(skip_mask, rng.normal(0, 1.0/np.sqrt(N2), size=(N1, N2)), 0.0)
        for i0, i1, n in [(0, N1, N1), (N1, N, N2)]:
            mask = rng.random((n, n)) < 0.10
            np.fill_diagonal(mask, False)
            W[i0:i1, i0:i1] = np.where(mask, rng.normal(0, 1.0/np.sqrt(n), size=(n, n)), 0.0)
    else:
        raise ValueError(topology)
    return rho_normalize(W, rho, rho_variant)


def run_cell_sim_v2(topology, N, T, kappa, drive_fn, seed, rho_variant):
    rng = np.random.default_rng(seed)
    cfg = NSRAMCell2TConfig(use_iii=True, use_gidl=True, use_bjt=True,
                            newton_max_iters=50)
    # M3b F1.v2 η-bounded cell defaults (no iii_body_gain set ⇒ η path)
    M1, M2 = v1.build_calibrated_models()
    bjt = GummelPoonNPN.from_sebas_card()
    bjt.Bf = BF_PHYSICAL
    base_VG1 = torch.tensor(rng.choice([0.2, 0.4, 0.6], size=N), dtype=torch.float64)
    base_VG2 = torch.tensor(rng.uniform(0.0, 0.5, size=N), dtype=torch.float64)
    W_rec = torch.tensor(build_W_v2(topology, N, rho=0.9, rng=rng,
                                     rho_variant=rho_variant),
                         dtype=torch.float64)
    feat_prev = torch.zeros(N, dtype=torch.float64)
    log_Id = np.zeros((N, T))
    for t in range(T):
        Vd_t = torch.tensor([float(drive_fn(t))], dtype=torch.float64)
        recur = (W_rec @ feat_prev) if kappa > 0 else torch.zeros(N, dtype=torch.float64)
        VG2_eff = (base_VG2 + kappa * recur).clamp(-0.2, 1.0)
        out = forward_2t_batched(cfg, M1, M2, bjt, Vd_t, base_VG1, VG2_eff,
                                  max_iters=15, tol=1e-9, verbose=False)
        log_Id[:, t] = np.log10(np.maximum(out["Id"].abs().squeeze().numpy(), 1e-15))
        feat_prev = torch.tensor(log_Id[:, t], dtype=torch.float64)
    return log_Id


def main():
    t0 = time.time()
    topologies = ["RAND_GAUSS", "MESH_4N", "ER_SPARSE", "WS_SMALLWORLD",
                  "HUB_SPOKE", "LAYERED"]
    Ns = [int(x) for x in os.environ.get("Z142_NS", "100,300,800").split(",")]
    seeds = list(range(42, 42 + int(os.environ.get("Z142_SEEDS", "5"))))
    rho_variants = ["rho_lambda", "rho_p95_sv", "rho_deg_norm"]
    T = int(os.environ.get("Z142_T", "500"))
    kappa = 0.03
    print(f"[z142] η-bounded topology v2 (M3b F3)")
    print(f"  topologies: {topologies}")
    print(f"  Ns: {Ns}, seeds: {seeds}, ρ-norms: {rho_variants}")
    print(f"  T={T}, κ={kappa}, Bf={BF_PHYSICAL} (physical)")
    print(f"  total sims: {len(topologies)*len(Ns)*len(rho_variants)*len(seeds)}")
    print()

    results = {}
    n_done = 0
    n_total = len(topologies)*len(Ns)*len(rho_variants)*len(seeds)
    for rho_v in rho_variants:
        for topo in topologies:
            for N in Ns:
                for seed in seeds:
                    ti = time.time()
                    rng = np.random.default_rng(seed)
                    u_bin_int = rng.integers(0, 2, size=T)
                    u_bin = 2.0 * u_bin_int - 1.0
                    drive = lambda t: 1.0 + 0.5 * float(u_bin[t])
                    try:
                        log_Id1 = run_cell_sim_v2(topo, N, T, kappa, drive,
                                                   seed, rho_v)
                        MC, NARMA_NRMSE = z119.eval_MC_NARMA(log_Id1, u_bin, T)
                        XOR_acc = z119.eval_XOR(log_Id1, u_bin_int, tau=2, T=T)
                        u_wave, cls = z119.waveform_inputs(
                            T, n_classes=4, rng=np.random.default_rng(seed+1000))
                        drive_w = lambda t: 1.0 + 0.5 * float(u_wave[t])
                        log_Id2 = run_cell_sim_v2(topo, N, T, kappa, drive_w,
                                                   seed, rho_v)
                        WAVE_acc = z119.eval_waveform(log_Id2, cls, T, n_classes=4)
                    except Exception as e:
                        MC = float("nan"); NARMA_NRMSE = float("nan")
                        XOR_acc = float("nan"); WAVE_acc = float("nan")
                        print(f"  ERROR {topo}/N{N}/{rho_v}/s{seed}: {e}", flush=True)

                    key = f"{topo}_N{N}_{rho_v}_s{seed}"
                    results[key] = {"topo": topo, "N": N,
                                    "rho_variant": rho_v, "seed": seed,
                                    "MC": MC, "NARMA_NRMSE": NARMA_NRMSE,
                                    "XOR_acc": XOR_acc, "WAVE_acc": WAVE_acc,
                                    "wall_s": float(time.time() - ti)}
                    n_done += 1
                    print(f"  [{n_done:3d}/{n_total}] {topo:14s} N={N:>4d} "
                          f"{rho_v:13s} s={seed}  MC={MC:5.2f} NARMA={NARMA_NRMSE:5.2f} "
                          f"XOR={XOR_acc:.2f} WAVE={WAVE_acc:.2f}  "
                          f"({time.time()-ti:.0f}s, total {time.time()-t0:.0f}s)",
                          flush=True)
                    json.dump({"results": results, "Bf": BF_PHYSICAL,
                                "topologies": topologies, "Ns": Ns,
                                "seeds": seeds, "rho_variants": rho_variants,
                                "T": T, "kappa": kappa,
                                "n_done": n_done, "n_total": n_total},
                               (OUT / "summary_partial.json").open("w"), indent=2)

    # Aggregate by (topo, N, rho_variant) over seeds
    print(f"\n[z142] === Aggregated (mean ± sd over {len(seeds)} seeds) ===")
    agg = {}
    for rho_v in rho_variants:
        print(f"\n  ρ-norm variant: {rho_v}")
        print(f"  {'topo':14s} {'N':>4s}  {'MC mean':>8s} {'MC sd':>6s} {'NARMA':>6s} {'XOR':>5s} {'WAVE':>5s}")
        for topo in topologies:
            for N in Ns:
                keys = [f"{topo}_N{N}_{rho_v}_s{s}" for s in seeds]
                valid = [results[k] for k in keys if k in results
                          and results[k]["MC"] == results[k]["MC"]]
                if not valid:
                    continue
                mcs = [r["MC"] for r in valid]
                nas = [r["NARMA_NRMSE"] for r in valid if np.isfinite(r["NARMA_NRMSE"])]
                xos = [r["XOR_acc"] for r in valid]
                was = [r["WAVE_acc"] for r in valid]
                agg[f"{topo}_N{N}_{rho_v}"] = {
                    "MC_mean": float(np.mean(mcs)), "MC_sd": float(np.std(mcs)),
                    "NARMA_NRMSE_mean": float(np.mean(nas)) if nas else float('nan'),
                    "XOR_acc_mean": float(np.mean(xos)),
                    "WAVE_acc_mean": float(np.mean(was)),
                    "n_valid": len(valid),
                }
                print(f"  {topo:14s} {N:>4d}  {np.mean(mcs):>7.2f}  {np.std(mcs):>5.2f}  "
                      f"{np.mean(nas) if nas else float('nan'):>5.2f}  "
                      f"{np.mean(xos):>5.2f}  {np.mean(was):>5.2f}")

    json.dump({"results": results, "agg": agg, "Bf": BF_PHYSICAL,
                "topologies": topologies, "Ns": Ns, "seeds": seeds,
                "rho_variants": rho_variants, "T": T, "kappa": kappa},
               (OUT / "summary.json").open("w"), indent=2)
    (OUT / "summary_partial.json").unlink(missing_ok=True)
    print(f"\n[z142] saved {OUT}/summary.json")
    print(f"[z142] wall: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
