"""z232 — B.5: bootstrap CI compare lumped vs quasi-2D on 25 biases.

Extends last wake-up's 3-bias finding (branch-protect reduces but
doesn't recover lumped's branch) to 25 biases with bootstrap 95% CI.

Configs:
  L : lumped solve_2t_steady_state (production BJT)
  A : q2d (no protect, no leak) — finds alt-root
  B : q2d + branch-protect (max ΔVb=50mV)
  C : q2d + body-leak 50 GΩ

Metric: log10|Id_X| - log10|Id_L| per bias. Bootstrap CI on median.

If |median dec(B)| > 1.0 → branch-protect hypothesis falsified at 25-N
statistical power.

CPU-only. ~30s wall.
"""
from __future__ import annotations
import os, sys, json, time
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
from pathlib import Path
import numpy as np
import torch
torch.set_default_dtype(torch.float64)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "nsram"))
OUT = ROOT / "results/z232_b5_lumped_vs_q2d"; OUT.mkdir(parents=True, exist_ok=True)


def main():
    from nsram.bsim4_port.bjt import GummelPoonNPN
    from nsram.bsim4_port.model_card import BSIM4Model
    from nsram.bsim4_port.nsram_cell_2T import (
        NSRAMCell2TConfig, solve_2t_steady_state, solve_2t_quasi2d_steady_state,
    )

    DATA = ROOT / "data/sebas_2026_04_22"
    m1 = BSIM4Model.from_spice((DATA / "M1_130DNWFB.txt").read_text(), model_type="nmos")
    m2 = BSIM4Model.from_spice((DATA / "M2_130bulkNSRAM.txt").read_text(), model_type="nmos")
    bjt = GummelPoonNPN.from_sebas_card()
    bjt.Bf = 9000.0; bjt.Va = 0.55; bjt.Is = 1e-9

    rng = np.random.default_rng(42)
    N = 25
    Vd_arr  = rng.uniform(0.5, 1.5, N)
    VG1_arr = rng.uniform(0.2, 0.7, N)
    VG2_arr = rng.uniform(0.0, 0.4, N)

    def t(x): return torch.tensor(float(x), dtype=torch.float64)

    cfg_L = NSRAMCell2TConfig()
    cfg_A = NSRAMCell2TConfig(quasi2d_body=True, Rb_SD=1e6, iii_split_alpha=0.7)
    cfg_B = NSRAMCell2TConfig(quasi2d_body=True, Rb_SD=1e6, iii_split_alpha=0.7,
                                q2d_branch_protect=True, q2d_branch_max_dvb=0.05)
    cfg_C = NSRAMCell2TConfig(quasi2d_body=True, Rb_SD=1e6, iii_split_alpha=0.7,
                                q2d_body_leak_R=5e10)

    def runcell(cfg, Vd, VG1, VG2, q2d):
        try:
            if q2d:
                out = solve_2t_quasi2d_steady_state(cfg, m1, bjt, t(Vd), t(VG1), t(VG2),
                                                       model_M2=m2)
            else:
                out = solve_2t_steady_state(cfg, m1, bjt, t(Vd), t(VG1), t(VG2),
                                              model_M2=m2)
            Id = float(out["Id"])
            conv = bool(out.get("converged", True))
            return Id, conv
        except Exception as e:
            return float("nan"), False

    print(f"=== z232 B.5: 25-bias lumped vs q2d (4 configs) ===")
    L = np.zeros(N); A = np.zeros(N); B = np.zeros(N); C = np.zeros(N)
    convs = {k: 0 for k in "LABC"}
    t0 = time.time()
    for i in range(N):
        L[i], cL = runcell(cfg_L, Vd_arr[i], VG1_arr[i], VG2_arr[i], q2d=False)
        A[i], cA = runcell(cfg_A, Vd_arr[i], VG1_arr[i], VG2_arr[i], q2d=True)
        B[i], cB = runcell(cfg_B, Vd_arr[i], VG1_arr[i], VG2_arr[i], q2d=True)
        C[i], cC = runcell(cfg_C, Vd_arr[i], VG1_arr[i], VG2_arr[i], q2d=True)
        convs["L"] += int(cL); convs["A"] += int(cA)
        convs["B"] += int(cB); convs["C"] += int(cC)
    wall = time.time() - t0
    print(f"wall {wall:.1f}s; conv L={convs['L']} A={convs['A']} B={convs['B']} C={convs['C']} (of {N})")

    def log10abs(x):
        return np.log10(np.maximum(np.abs(x), 1e-15))
    ddA = log10abs(A) - log10abs(L)
    ddB = log10abs(B) - log10abs(L)
    ddC = log10abs(C) - log10abs(L)

    def boot_ci(x, n=2000):
        rng = np.random.default_rng(0)
        m = []
        for _ in range(n):
            idx = rng.integers(0, len(x), len(x))
            m.append(np.median(x[idx]))
        m = np.array(m)
        return float(np.median(x)), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))

    medA, lA, hA = boot_ci(ddA)
    medB, lB, hB = boot_ci(ddB)
    medC, lC, hC = boot_ci(ddC)

    print(f"\n          Δlog10|Id|  vs  lumped (median, 95% bootstrap CI):")
    print(f"  q2d (A)         : {medA:+.2f}  CI [{lA:+.2f}, {hA:+.2f}]")
    print(f"  q2d+protect (B) : {medB:+.2f}  CI [{lB:+.2f}, {hB:+.2f}]")
    print(f"  q2d+leak (C)    : {medC:+.2f}  CI [{lC:+.2f}, {hC:+.2f}]")

    # Hypothesis test: does branch-protect (B) recover lumped (median ≈ 0)?
    hypo_pass = (lB <= 0.5 and hB >= -0.5)
    print(f"\nBranch-protect rescues lumped (CI contains 0): "
          f"{'✅ YES' if hypo_pass else '❌ FALSIFIED'}")
    print(f"  (CI for B is [{lB:+.2f}, {hB:+.2f}] dec; needs to bracket 0±0.5)")

    out = {
        "N": N, "wall_s": wall, "converged": convs,
        "median_dec": {"A": medA, "B": medB, "C": medC},
        "ci95_dec":   {"A": [lA, hA], "B": [lB, hB], "C": [lC, hC]},
        "branch_protect_rescues_lumped": bool(hypo_pass),
    }
    (OUT / "summary.json").write_text(json.dumps(out, indent=2))
    print(f"\nsaved {OUT / 'summary.json'}")


if __name__ == "__main__":
    main()
