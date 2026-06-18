"""z122 — test whether sign-asymmetric W_rec helps MC + XOR.

OpenAI gpt-5 oracle (O13 review) flagged that purely-resistive
silicon coupling (NS-RAM shared body charge) is intrinsically
**symmetric and non-negative**. ER_SPARSE in z119/z121 used Gaussian
weights → ~50% positive, ~50% negative. If MC/XOR sparse advantage
depends on having signed weights, silicon implementation would
fall short.

This script tests three sign-conditions on the ER_SPARSE topology
at canonical 1/√N (z121 setting):
  A — SIGNED  (z121 baseline): Gaussian weights, both signs.
  B — POSITIVE-ONLY: |Gaussian| (all weights ≥ 0, mimics resistive).
  C — RANDOMLY SIGN-FLIPPED: ±|Gaussian| with 50/50 sign per edge,
                              same magnitude distribution as A.

If A ≈ C >> B, the advantage is **sign-dependent** → silicon
risk. If A ≈ B, the advantage is **magnitude-only** → silicon
implementable. If A ≈ C ≈ B, sign doesn't matter and silicon-
realisable resistive coupling is fine.

Sweep:
  topology = ER_SPARSE (p=0.1)
  N = 200, T = 600, κ = 0.03, 5 seeds
  ridge per task GCV-style sweep over {1e-3, 1e-1, 1e+1, 1e+3}
  tasks: MC (15 lags), XOR (τ=2)

Wall budget: 3 conditions × 5 seeds × ~50s = ~13 min.
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import torch
torch.set_default_dtype(torch.float64)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/z122_sign_asymmetry_test"
OUT.mkdir(parents=True, exist_ok=True)

import importlib.util
sp = importlib.util.spec_from_file_location("z121", ROOT / "scripts/z121_mc_xor_canonical_wrec.py")
z121 = importlib.util.module_from_spec(sp); sp.loader.exec_module(z121)


def build_W_signed(N, rng):
    """A: signed Gaussian (z121 ER_SPARSE baseline)."""
    p = 0.1
    mask = rng.random((N, N)) < p
    return np.where(mask, rng.normal(0.0, 1.0/np.sqrt(N*p), size=(N, N)), 0.0)


def build_W_positive(N, rng):
    """B: |Gaussian| — purely-resistive analog."""
    p = 0.1
    mask = rng.random((N, N)) < p
    g = rng.normal(0.0, 1.0/np.sqrt(N*p), size=(N, N))
    return np.where(mask, np.abs(g), 0.0)


def build_W_flipped(N, rng):
    """C: ±|Gaussian| with random ±1 per edge — same |w| distribution as A."""
    p = 0.1
    mask = rng.random((N, N)) < p
    g = rng.normal(0.0, 1.0/np.sqrt(N*p), size=(N, N))
    signs = np.where(rng.random((N, N)) < 0.5, -1.0, 1.0)
    return np.where(mask, signs * np.abs(g), 0.0)


def run_cell_sim(N, T, kappa, seed, W_builder):
    """Like z121.run_cell_sim but with custom W builder."""
    rng = np.random.default_rng(seed)
    u_int = rng.integers(0, 2, size=T)
    u = 2.0 * u_int - 1.0
    from nsram.bsim4_port.nsram_cell_2T import NSRAMCell2TConfig
    from nsram.bsim4_port.bjt import GummelPoonNPN
    from nsram.bsim4_port.vectorized import forward_2t_batched

    sp_v1 = importlib.util.spec_from_file_location("v1", ROOT / "scripts/z96_narma10_pilot.py")
    v1 = importlib.util.module_from_spec(sp_v1); sp_v1.loader.exec_module(v1)
    cfg = NSRAMCell2TConfig(use_iii=True, use_gidl=True, use_bjt=True, newton_max_iters=50)
    M1, M2 = v1.build_calibrated_models()
    bjt = GummelPoonNPN.from_sebas_card(); bjt.Bf = 5e4
    base_VG1 = torch.tensor(rng.choice([0.2, 0.4, 0.6], size=N), dtype=torch.float64)
    base_VG2 = torch.tensor(rng.uniform(0.0, 0.5, size=N), dtype=torch.float64)
    W_rec = torch.tensor(W_builder(N, rng), dtype=torch.float64)
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
    return log_Id, u_int, u


def main():
    t0 = time.time()
    seeds = [42, 43, 44, 45, 46]
    ridges = [1e-3, 1e-1, 1e+1, 1e+3]
    N = 200; T = 600; kappa = 0.03
    conditions = [("A_signed",    build_W_signed),
                    ("B_positive",  build_W_positive),
                    ("C_flipped",   build_W_flipped)]
    print(f"[z122] sign-asymmetry test on ER_SPARSE p=0.1, N={N}, canonical 1/√(Np)")
    print(f"  conditions: {[c[0] for c in conditions]}")
    print(f"  z121 reference (signed): MC=2.90, XOR=0.82\n")

    grid_mc = np.zeros((len(conditions), len(seeds)))
    grid_xor = np.zeros((len(conditions), len(seeds)))
    for i, (cname, builder) in enumerate(conditions):
        for j, seed in enumerate(seeds):
            ti = time.time()
            log_Id, u_int, u = run_cell_sim(N, T, kappa, seed, builder)
            sim_t = time.time() - ti
            MC, _ = z121.eval_MC(log_Id, u, T, ridges)
            XOR, _ = z121.eval_XOR(log_Id, u_int, tau=2, T=T, ridges=ridges)
            grid_mc[i, j] = MC
            grid_xor[i, j] = XOR
            print(f"  {cname:11s} s={seed}  sim={sim_t:.0f}s  MC={MC:5.2f}  "
                   f"XOR={XOR:.2f}", flush=True)

    print(f"\n[z122] === Aggregated (5 seeds, ER_SPARSE canonical 1/√(Np)) ===")
    print(f"  {'cond':12s} {'MC mean':>9s} {'± std':>7s}  {'XOR mean':>9s} {'± std':>7s}")
    means_mc, stds_mc = grid_mc.mean(axis=1), grid_mc.std(axis=1, ddof=1)
    means_xor, stds_xor = grid_xor.mean(axis=1), grid_xor.std(axis=1, ddof=1)
    for i, (cname, _) in enumerate(conditions):
        print(f"  {cname:12s} {means_mc[i]:>9.3f} {stds_mc[i]:>7.3f}  "
               f"{means_xor[i]:>9.3f} {stds_xor[i]:>7.3f}")

    # Paired-t per pair vs A_signed
    print(f"\n[z122] === Paired-t vs A_signed baseline ===")
    for name, grid in [("MC", grid_mc), ("XOR", grid_xor)]:
        for i in range(1, len(conditions)):
            diffs = grid[i] - grid[0]
            d_mean = diffs.mean()
            d_sem = diffs.std(ddof=1) / np.sqrt(len(seeds))
            t_stat = d_mean / d_sem if d_sem > 1e-9 else float("inf")
            print(f"  {name} {conditions[i][0]:11s} - {conditions[0][0]:11s}: "
                   f"Δ={d_mean:+.3f}  SEM={d_sem:.3f}  t={t_stat:+.2f}")

    # Verdict
    print(f"\n[z122] === Verdict ===")
    deltaB_mc = means_mc[1] - means_mc[0]
    deltaB_xor = means_xor[1] - means_xor[0]
    deltaC_mc = means_mc[2] - means_mc[0]
    deltaC_xor = means_xor[2] - means_xor[0]
    if abs(deltaB_mc) < 0.3 and abs(deltaB_xor) < 0.05:
        print(f"  B (positive-only) ≈ A (signed) → SILICON-REALISABLE.")
    else:
        print(f"  B vs A: ΔMC={deltaB_mc:+.3f}, ΔXOR={deltaB_xor:+.3f} — sign matters.")
    if abs(deltaC_mc) < 0.3 and abs(deltaC_xor) < 0.05:
        print(f"  C (flipped) ≈ A → magnitude alone drives advantage.")

    json.dump({"conditions": [c[0] for c in conditions], "seeds": seeds,
                "MC_grid": grid_mc.tolist(), "XOR_grid": grid_xor.tolist(),
                "MC_means": means_mc.tolist(), "MC_stds": stds_mc.tolist(),
                "XOR_means": means_xor.tolist(), "XOR_stds": stds_xor.tolist(),
                "N": N, "T": T, "kappa": kappa},
               (OUT / "summary.json").open("w"), indent=2)
    print(f"\n[z122] saved {OUT}/summary.json")
    print(f"[z122] wall: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
