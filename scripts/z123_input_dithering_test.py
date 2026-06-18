"""z123 — input-dithering test (option C of C.3 v2.3 sign-inverter alternatives).

z122 confirmed positive-only coupling collapses MC from 2.90 to 0.43.
The C.3 v2.3 addendum lists three implementation options for the
required sign-inverter sub-fabric:
  (A) source-follower inverter cell — preferred, +30% area.
  (B) per-cell signed-readout pair.
  (C) input-side dithering — cheapest, software-defined.

This script validates option (C). We keep ER_SPARSE coupling
**positive-only** (silicon-realisable) and add a per-cell signed
input gain e_i ∈ {-1, +1} that modulates how the input reaches
each cell:

    VG2_eff[i,t] = VG2_0[i] + κ · Σ_j |W_rec[i,j]| · log_Id_j[t-1]
                            + ν · e_i · u[t]

The first sum is the silicon-style positive-only coupling.
The new ν · e_i · u[t] term is the input dithering — half the
cells see +input, half see −input. This breaks the common-mode
that B_positive demonstrated.

If MC and XOR recover toward A_signed levels, option (C) is a
viable cheap alternative to source-follower inverters. If not,
options (A)/(B) are required.

Sweep:
  topology = ER_SPARSE p=0.1, positive-only, canonical 1/√(Np)
  N = 200, T = 600, κ = 0.03, 5 seeds
  ν ∈ {0.0, 0.05, 0.10, 0.20}  (dithering strength sweep)
  ridges = {1e-3, 1e-1, 1e+1, 1e+3} per task

Wall budget: 4 ν × 5 seeds × ~50s = ~17 min.
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import torch
torch.set_default_dtype(torch.float64)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/z123_input_dithering_test"
OUT.mkdir(parents=True, exist_ok=True)

import importlib.util
sp_v1 = importlib.util.spec_from_file_location("v1", ROOT / "scripts/z96_narma10_pilot.py")
v1 = importlib.util.module_from_spec(sp_v1); sp_v1.loader.exec_module(v1)
sp_z121 = importlib.util.spec_from_file_location("z121", ROOT / "scripts/z121_mc_xor_canonical_wrec.py")
z121 = importlib.util.module_from_spec(sp_z121); sp_z121.loader.exec_module(z121)

from nsram.bsim4_port.nsram_cell_2T import NSRAMCell2TConfig
from nsram.bsim4_port.bjt import GummelPoonNPN
from nsram.bsim4_port.vectorized import forward_2t_batched


def build_W_positive_canonical(N, rng):
    """ER sparse, positive-only, canonical 1/√(Np)."""
    p = 0.1
    mask = rng.random((N, N)) < p
    g = rng.normal(0.0, 1.0/np.sqrt(N*p), size=(N, N))
    return np.where(mask, np.abs(g), 0.0)


def run_cell_sim(N, T, kappa, nu, seed):
    rng = np.random.default_rng(seed)
    u_int = rng.integers(0, 2, size=T)
    u = 2.0 * u_int - 1.0
    cfg = NSRAMCell2TConfig(use_iii=True, use_gidl=True, use_bjt=True, newton_max_iters=50)
    M1, M2 = v1.build_calibrated_models()
    bjt = GummelPoonNPN.from_sebas_card(); bjt.Bf = 5e4
    base_VG1 = torch.tensor(rng.choice([0.2, 0.4, 0.6], size=N), dtype=torch.float64)
    base_VG2 = torch.tensor(rng.uniform(0.0, 0.5, size=N), dtype=torch.float64)
    W_rec = torch.tensor(build_W_positive_canonical(N, rng), dtype=torch.float64)
    # Per-cell signed input gain e_i ∈ {-1, +1}
    e = torch.tensor(2 * rng.integers(0, 2, size=N) - 1, dtype=torch.float64)
    feat_prev = torch.zeros(N, dtype=torch.float64)
    log_Id = np.zeros((N, T))
    for t in range(T):
        Vd_t = torch.tensor([1.0 + 0.5*float(u[t])], dtype=torch.float64)
        recur = (W_rec @ feat_prev) if kappa > 0 else torch.zeros(N, dtype=torch.float64)
        # Per-cell input dithering on VG2:
        dith = nu * e * float(u[t])
        VG2_eff = (base_VG2 + kappa*recur + dith).clamp(-0.2, 1.0)
        out = forward_2t_batched(cfg, M1, M2, bjt, Vd_t, base_VG1, VG2_eff,
                                   max_iters=15, tol=1e-9, verbose=False)
        log_Id[:, t] = np.log10(np.maximum(out["Id"].abs().squeeze().numpy(), 1e-15))
        feat_prev = torch.tensor(log_Id[:, t], dtype=torch.float64)
    return log_Id, u_int, u


def main():
    t0 = time.time()
    seeds = [42, 43, 44, 45, 46]
    ridges = [1e-3, 1e-1, 1e+1, 1e+3]
    nus = [0.00, 0.05, 0.10, 0.20]
    N = 200; T = 600; kappa = 0.03
    print(f"[z123] input-dithering on ER_SPARSE positive-only, N={N}, canonical 1/√(Np)")
    print(f"  ν sweep: {nus}, seeds: {seeds}")
    print(f"  Reference points (from z121/z122 at canonical 1/√(Np)):")
    print(f"    A_signed   (z121): MC = 2.90 ± 0.30, XOR = 0.821")
    print(f"    B_positive (z122): MC = 0.43 ± 0.54, XOR = 0.562")
    print()

    grid_mc = np.zeros((len(nus), len(seeds)))
    grid_xor = np.zeros((len(nus), len(seeds)))
    for i, nu in enumerate(nus):
        for j, seed in enumerate(seeds):
            ti = time.time()
            log_Id, u_int, u = run_cell_sim(N, T, kappa, nu, seed)
            sim_t = time.time() - ti
            MC, _ = z121.eval_MC(log_Id, u, T, ridges)
            XOR, _ = z121.eval_XOR(log_Id, u_int, tau=2, T=T, ridges=ridges)
            grid_mc[i, j] = MC
            grid_xor[i, j] = XOR
            print(f"  ν={nu:.2f}  s={seed}  sim={sim_t:.0f}s  MC={MC:5.2f}  XOR={XOR:.2f}",
                   flush=True)

    print(f"\n[z123] === Aggregated (5 seeds, ER_SPARSE positive-only, canonical 1/√(Np)) ===")
    print(f"  {'ν':>5s}  {'MC mean':>9s} {'± std':>7s}  {'XOR mean':>9s} {'± std':>7s}")
    means_mc = grid_mc.mean(axis=1)
    stds_mc = grid_mc.std(axis=1, ddof=1)
    means_xor = grid_xor.mean(axis=1)
    stds_xor = grid_xor.std(axis=1, ddof=1)
    for i, nu in enumerate(nus):
        print(f"  {nu:>5.2f}  {means_mc[i]:>9.3f} {stds_mc[i]:>7.3f}  "
               f"{means_xor[i]:>9.3f} {stds_xor[i]:>7.3f}")

    # Recovery ratio: how much of the A_signed advantage does ν=ν* recover?
    print(f"\n[z123] === Recovery vs A_signed (z121: MC=2.90, XOR=0.82) ===")
    A_MC, A_XOR = 2.90, 0.821
    B_MC, B_XOR = 0.43, 0.562
    for i, nu in enumerate(nus):
        rec_mc = (means_mc[i] - B_MC) / (A_MC - B_MC) if A_MC != B_MC else 0
        rec_xor = (means_xor[i] - B_XOR) / (A_XOR - B_XOR) if A_XOR != B_XOR else 0
        print(f"  ν={nu:.2f}  MC recovery = {rec_mc*100:5.1f}%  "
               f"XOR recovery = {rec_xor*100:5.1f}%")

    best_i = int(np.argmax(means_mc))
    print(f"\n[z123] Best ν: {nus[best_i]:.2f} → MC={means_mc[best_i]:.3f}, "
           f"XOR={means_xor[best_i]:.3f}")
    if means_mc[best_i] > 2.0:
        print(f"  → input dithering RECOVERS most of the sparse advantage; option (C) viable.")
    elif means_mc[best_i] > 1.0:
        print(f"  → partial recovery; option (C) may be viable as fallback.")
    else:
        print(f"  → option (C) ineffective; need source-follower inverter (option A).")

    json.dump({"seeds": seeds, "nus": nus, "ridges": ridges, "N": N,
                "T": T, "kappa": kappa,
                "MC_grid": grid_mc.tolist(), "XOR_grid": grid_xor.tolist(),
                "MC_means": means_mc.tolist(), "MC_stds": stds_mc.tolist(),
                "XOR_means": means_xor.tolist(), "XOR_stds": stds_xor.tolist(),
                "A_signed_ref": {"MC": A_MC, "XOR": A_XOR},
                "B_positive_ref": {"MC": B_MC, "XOR": B_XOR}},
               (OUT / "summary.json").open("w"), indent=2)
    print(f"\n[z123] saved {OUT}/summary.json")
    print(f"[z123] wall: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
