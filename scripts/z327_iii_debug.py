"""z327 — R-11 BSIM4 IIMOD deep-trace.

R-9 (z325) reported Iii_M1 = 7.5e-48 A at V_G1=0.6, V_d=2.0 — physically
impossible if M1 is in strong saturation. The task asks us to:

1. Forcibly evaluate compute_iimpact at V_G1=0.6, V_d=2.0, Vsint=0 (M1 ON,
   strong saturation, Vds=2.0V) — the *isolated* IIMOD calc, decoupled
   from the solver basin that R-9 was stuck in.
2. Trace every intermediate (alpha0, beta0, leff, Vdseff, Idsa, T0..T2,
   exp arg, T1, Iii).
3. Hand-recompute via the BSIM4 v4.8.3 §6.1 formula and compare.
4. Identify whether the bug is in the formula or in the solver-state input.
5. If a code bug exists, fix it; otherwise document that the 1e-48 number
   is a *solver-state* artifact (Vsint pinned high → M1 OFF → Ids→0).

NOTE on the task's hand-calc (V_G1=0.6, V_d=2.0, Vsint=0):
  Vds_M1 = 2.0 V, Vgs_M1 = 0.6 V, Vth ~0.5 V → Vov ≈ 0.1 V → Vdseff ≈ 0.1 V
  diff = 1.90 V → exp(-20/1.90) = exp(-10.53) ≈ 2.7e-5
  alpha0/L · diff · Idsa·Vdseff
  Ids (saturation, weak Vov=0.1V) ≈ µCox·W/L·Vov²/2  — very small
"""
from __future__ import annotations
import os
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "4")

import importlib.util
import json
import math
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

torch.set_default_dtype(torch.float64)

ROOT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
SCRIPTS = ROOT / "scripts"
OUT_DIR = ROOT / "results/z327_iii_debug"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PARTIAL = OUT_DIR / "partial.json"
SUMMARY = OUT_DIR / "summary.json"

sys.path.insert(0, str(ROOT / "nsram"))

DEVICE = torch.device("cpu")  # tiny calc; CPU is enough and avoids ROCm noise
DTYPE = torch.float64

ALPHA0_CONST = 7.842e-5
BF_CARD = 10000.0


def _load_module(name, path):
    sp = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(mod)
    return mod


@contextmanager
def patch_sd_scaled(sd, overrides):
    if not overrides:
        yield
        return
    saved = {}
    try:
        for k, v in overrides.items():
            saved[k] = sd.scaled.get(k, None)
            sd.scaled[k] = v
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                sd.scaled.pop(k, None)
            else:
                sd.scaled[k] = v


def save(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def main():
    t0 = time.time()
    trace = {"script": "z327_iii_debug", "device": str(DEVICE), "steps": []}

    # ---------- Build models (re-use z304 plumbing) ----------
    z326 = _load_module("z326_solver_fix", SCRIPTS / "z326_solver_fix.py")
    z304 = _load_module("z304_sebas_three_branch_refit",
                        SCRIPTS / "z304_sebas_three_branch_refit.py")
    z91f, cfg, M1, M2, sd_M1, sd_M2, forward_2t = z304.build_models_once()
    sebas_rows = z304.load_sebas_params()
    print(f"[z327] models built ({time.time()-t0:.1f}s)", flush=True)

    from nsram.bsim4_port.dc import compute_dc
    from nsram.bsim4_port.leak import compute_iimpact

    VG1 = 0.6; VG2 = 0.20; Vd_val = 2.0
    z326.configure_v5b_postfix(cfg, VG1)

    sebas_row = z304.find_params(sebas_rows, VG1, VG2)
    P_M1, P_M2 = z304.make_row_overrides(
        sebas_row, ALPHA0_CONST, z91f.M2_STATIC_OVERRIDES)

    print(f"[z327] P_M1 overrides: {list(P_M1.keys())}", flush=True)

    # ---------------------------------------------------------------
    # STEP A: Forced strong-saturation OP for M1 (Vsint=0, Vb=0)
    #   Vds_M1 = 2.0 V, Vgs_M1 = 0.6 V, Vbs_M1 = 0 V
    # ---------------------------------------------------------------
    Vsint = 0.0
    Vb = 0.0
    Vds_M1 = Vd_val - Vsint           # 2.0
    Vgs_M1 = VG1 - Vsint              # 0.6
    Vbs_M1 = Vb - Vsint               # 0.0

    print(f"\n[z327] === STEP A: forced M1 strong-saturation OP ===", flush=True)
    print(f"   Vgs={Vgs_M1}, Vds={Vds_M1}, Vbs={Vbs_M1}", flush=True)

    with torch.no_grad(), patch_sd_scaled(sd_M1, P_M1):
        # Dump scaled params used.
        alpha0_used = float(sd_M1.scaled.get("alpha0", 0.0))
        alpha1_used = float(sd_M1.scaled.get("alpha1", 0.0))
        beta0_used = float(sd_M1.scaled.get("beta0", 0.0))
        leff = float(sd_M1.geom.leff)
        weff = float(sd_M1.geom.weff)
        print(f"   alpha0={alpha0_used}, alpha1={alpha1_used}, beta0={beta0_used}", flush=True)
        print(f"   leff={leff:.3e} m, weff={weff:.3e} m", flush=True)

        dc = compute_dc(
            M1, sd_M1,
            Vgs=torch.tensor(Vgs_M1, dtype=DTYPE),
            Vds=torch.tensor(Vds_M1, dtype=DTYPE),
            Vbs=torch.tensor(Vbs_M1, dtype=DTYPE),
        )
        Ids = float(dc.Ids)
        Vth = float(dc.Vth)
        Vgsteff = float(dc.Vgsteff)
        Vdseff = float(dc.Vdseff)
        Vdsat = float(dc.Vdsat)
        n_sub = float(dc.n)
        Idsa_x_Vdseff = float(dc.Idsa) if dc.Idsa is not None else None

        print(f"\n   DC result:", flush=True)
        print(f"     Vth     = {Vth:.4f} V", flush=True)
        print(f"     Vgsteff = {Vgsteff:.4e} V", flush=True)
        print(f"     Vdsat   = {Vdsat:.4f} V", flush=True)
        print(f"     Vdseff  = {Vdseff:.4f} V", flush=True)
        print(f"     n       = {n_sub:.3f}", flush=True)
        print(f"     Ids     = {Ids:.6e} A", flush=True)
        print(f"     Idsa·Vdseff = {Idsa_x_Vdseff:.6e} (A·V)", flush=True)

        # Now compute Iii via the port:
        Iii_port = float(compute_iimpact(
            M1, sd_M1, dc, Vds=torch.tensor(Vds_M1, dtype=DTYPE)))
        print(f"\n   compute_iimpact(M1) = {Iii_port:.6e} A", flush=True)

        # Hand-recompute step by step
        diff = Vds_M1 - Vdseff
        T2 = (alpha0_used + alpha1_used * leff) / leff
        if diff > 0:
            exp_arg = -beta0_used / max(diff, 1e-30)
            T1_strong = T2 * diff * math.exp(max(exp_arg, -34.0))
        else:
            T1_strong = 0.0
        Iii_hand = T1_strong * (Idsa_x_Vdseff if Idsa_x_Vdseff is not None else Ids)
        print(f"\n   HAND-CALC:", flush=True)
        print(f"     diff = Vds - Vdseff = {diff:.6f} V", flush=True)
        print(f"     T2   = (alpha0+alpha1·L)/L = {T2:.4e} /m", flush=True)
        print(f"     exp_arg = -beta0/diff = {exp_arg:.4f}", flush=True)
        print(f"     exp(exp_arg) = {math.exp(max(exp_arg,-34.0)):.4e}", flush=True)
        print(f"     T1   = T2·diff·exp() = {T1_strong:.4e}", flush=True)
        print(f"     Iii  = T1·(Idsa·Vdseff) = {Iii_hand:.6e} A", flush=True)

        rel_err = abs(Iii_port - Iii_hand) / max(abs(Iii_hand), 1e-300)
        print(f"\n   port-vs-hand rel err = {rel_err:.3e}", flush=True)

    step_A = {
        "name": "forced_strong_sat_Vsint0",
        "Vgs": Vgs_M1, "Vds": Vds_M1, "Vbs": Vbs_M1,
        "alpha0": alpha0_used, "alpha1": alpha1_used, "beta0": beta0_used,
        "leff": leff, "weff": weff,
        "Vth": Vth, "Vgsteff": Vgsteff, "Vdsat": Vdsat, "Vdseff": Vdseff,
        "n": n_sub, "Ids": Ids, "Idsa_Vdseff": Idsa_x_Vdseff,
        "diff_Vds_Vdseff": diff, "T2": T2, "exp_arg": exp_arg,
        "exp_value": math.exp(max(exp_arg, -34.0)),
        "T1_strong": T1_strong,
        "Iii_port": Iii_port, "Iii_hand": Iii_hand,
        "port_vs_hand_rel_err": rel_err,
    }
    trace["steps"].append(step_A)
    save(PARTIAL, trace)

    # ---------------------------------------------------------------
    # STEP B: Sweep Vsint from 0 → 1.9 V to see how Iii_M1 collapses
    #         as the solver moves Vsint toward Vd (M1 toward OFF).
    # ---------------------------------------------------------------
    print(f"\n[z327] === STEP B: Iii vs Vsint sweep (V_d=2, V_G1=0.6) ===", flush=True)
    print(f"   {'Vsint':>8} {'Vds_M1':>8} {'Vgs_M1':>8} {'Ids':>14} {'Vdseff':>10} {'Iii_port':>14}", flush=True)
    sweep_B = []
    with torch.no_grad(), patch_sd_scaled(sd_M1, P_M1):
        for vsint in [0.0, 0.1, 0.2, 0.5, 0.8, 1.0, 1.3, 1.5, 1.7, 1.85, 1.867, 1.9]:
            vds = Vd_val - vsint
            vgs = VG1 - vsint
            vbs = 0.0 - vsint
            dc = compute_dc(M1, sd_M1,
                            Vgs=torch.tensor(vgs, dtype=DTYPE),
                            Vds=torch.tensor(vds, dtype=DTYPE),
                            Vbs=torch.tensor(vbs, dtype=DTYPE))
            iii = float(compute_iimpact(M1, sd_M1, dc, Vds=torch.tensor(vds, dtype=DTYPE)))
            row = {"Vsint": vsint, "Vds": vds, "Vgs": vgs,
                   "Ids": float(dc.Ids), "Vdseff": float(dc.Vdseff),
                   "Idsa_Vdseff": float(dc.Idsa) if dc.Idsa is not None else None,
                   "Iii_port": iii}
            sweep_B.append(row)
            print(f"   {vsint:>8.4f} {vds:>8.4f} {vgs:>8.4f} {float(dc.Ids):>14.3e} {float(dc.Vdseff):>10.4f} {iii:>14.3e}", flush=True)

    trace["steps"].append({"name": "Vsint_sweep", "rows": sweep_B})
    save(PARTIAL, trace)

    # ---------------------------------------------------------------
    # STEP C: Verdict
    # ---------------------------------------------------------------
    # Bug check: if port-vs-hand agrees AND Iii at Vsint=0 is >1e-25, then
    # the IIMOD formula is correct and z325's 1e-48 is a *solver-state*
    # artifact (Vsint pinned high → M1 deeply OFF).
    Iii_at_0 = step_A["Iii_port"]
    Iii_at_z325 = sweep_B[-2]["Iii_port"]  # Vsint=1.867

    verdict = {
        "Iii_M1_at_Vsint_0": Iii_at_0,
        "Iii_M1_at_Vsint_1p867": Iii_at_z325,
        "infra_gate_Iii_gt_1e25": Iii_at_0 > 1e-25,
        "port_matches_hand_calc": step_A["port_vs_hand_rel_err"] < 1e-3,
        "diagnosis": (
            "IIMOD formula CORRECT. z325's 1e-48 is a solver-state artifact: "
            "Vsint converges to 1.867V → M1 has Vgs=-1.27V (deeply OFF), "
            "Ids→1e-13 A, Iii → near-zero. The fix is in the SOLVER (z326 task), "
            "not in compute_iimpact. No code change to leak.py warranted."
            if (step_A["port_vs_hand_rel_err"] < 1e-3 and Iii_at_0 > 1e-25)
            else "IIMOD formula MISMATCH — investigate further."
        ),
    }
    trace["verdict"] = verdict
    trace["elapsed_s"] = time.time() - t0

    print(f"\n[z327] === VERDICT ===", flush=True)
    for k, v in verdict.items():
        print(f"   {k}: {v}", flush=True)

    save(SUMMARY, trace)
    print(f"\n[z327] saved {SUMMARY}", flush=True)


if __name__ == "__main__":
    main()
