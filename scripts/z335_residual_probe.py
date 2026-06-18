"""z335 — R-19 Sub-task A: Discriminating residual probe.

Evaluate `_residuals` at EXACTLY the ngspice OP for VG1=0.6, VG2=0.20, Vd=2.0:
    Vsint = 0.382 V
    Vb    = 0.267 V

with the SAME cfg (z332 / configure_v5b_postfix, Sebas row, BJT card).

Print every term contributing to R_Sint and R_B, plus |R_Sint|, |R_B|, ||R||_inf.

Decision gate:
  ||R||_inf < 1e-9  → basin-lock alone (R-13 sufficient).
  ||R||_inf > 1e-6  → structural KCL bug (R-13 INSUFFICIENT, need M2.B fix).

Output: stdout + results/z335_residual_probe/summary.json
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
from pathlib import Path

import torch

torch.set_default_dtype(torch.float64)

ROOT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
SCRIPTS = ROOT / "scripts"
OUT_DIR = ROOT / "results/z335_residual_probe"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY = OUT_DIR / "summary.json"

sys.path.insert(0, str(ROOT / "nsram"))
DTYPE = torch.float64


def _load_module(name, path):
    sp = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(mod)
    return mod


def main():
    t0 = time.time()
    print("[z335] residual probe at ngspice OP", flush=True)

    z304 = _load_module("z304", SCRIPTS / "z304_sebas_three_branch_refit.py")
    z332 = _load_module("z332", SCRIPTS / "z332_vb_free_solver.py")
    z91f, cfg, M1, M2, sd_M1, sd_M2, _ = z304.build_models_once()
    sebas_rows = z304.load_sebas_params()

    VG1, VG2, Vd = 0.6, 0.20, 2.0
    Vsint_ng, Vb_ng = 0.382, 0.267

    # Apply z332's v5b configuration (same as z332 test_single_bias)
    z332.configure_v5b_postfix(cfg, VG1, vd_for_vnwell=Vd)

    row = next((r for r in sebas_rows
                if abs(r["VG1"] - VG1) < 1e-3 and abs(r["VG2"] - VG2) < 1e-3), None)
    if row is None:
        raise SystemExit("Sebas row VG1=0.6 VG2=0.20 not found")

    P_M1, P_M2, bjt = z332._build_row_setup(row, z91f)

    from nsram.bsim4_port.nsram_cell_2T import _residuals

    Vd_t    = torch.tensor(float(Vd), dtype=DTYPE)
    VG1_t   = torch.tensor(float(VG1), dtype=DTYPE)
    VG2_t   = torch.tensor(float(VG2), dtype=DTYPE)
    Vsint_t = torch.tensor(float(Vsint_ng), dtype=DTYPE)
    Vb_t    = torch.tensor(float(Vb_ng), dtype=DTYPE)

    with torch.no_grad(), \
         z332.patch_sd_scaled(sd_M1, P_M1), \
         z332.patch_sd_scaled(sd_M2, P_M2):
        R_S, R_B, comp = _residuals(
            cfg, M1, bjt, Vd_t, VG1_t, VG2_t, Vsint_t, Vb_t,
            P_M1=None, P_M2=None, model_M2=M2,
        )

    def f(x):
        try:
            return float(x.detach().cpu().item())
        except Exception:
            return float(x)

    # Pull component tensors as floats
    c = {k: f(v) for k, v in comp.items()}
    R_S_v = f(R_S)
    R_B_v = f(R_B)
    norm_inf = max(abs(R_S_v), abs(R_B_v))

    # Reconstruct individual contributions for printing.
    # R_Sint = Ids_M1 - Ids_M2 + Ibs_M1 + Ibd_M2 + bjt_sint_term
    bjt_emitter_to_gnd = bool(getattr(cfg, "bjt_emitter_to_gnd", False))
    bjt_sint_term = -c["Ic_Q1"] if bjt_emitter_to_gnd else -c["Ie_Q1"]
    sint_terms = {
        "+Ids_M1":      c["Ids_M1"],
        "-Ids_M2":      -c["Ids_M2"],
        "+Ibs_M1":      c["Ibs_M1"],
        "+Ibd_M2":      c["Ibd_M2"],
        "bjt_sint":     bjt_sint_term,
        "(Ic_Q1)":      c["Ic_Q1"],
        "(Ie_Q1)":      c["Ie_Q1"],
    }

    m1_d = float(cfg.m1_diode_scale)
    m2_body_gnd = bool(cfg.m2_body_gnd)
    use_local_base = bool(getattr(cfg, "use_local_base", False))
    iii_to_body_factor = 1.0  # eta_lat default 0 → factor=1
    # iii_gain at ngspice OP (Vds_eff≈Vd=2.0)
    eta_max = float(getattr(cfg, "eta_max", 1.0))
    eta_slope = float(getattr(cfg, "eta_slope", 10.0))
    eta_vds_th = float(getattr(cfg, "eta_vds_th", 1.0))
    import math as _math
    iii_gain = eta_max / (1.0 + _math.exp(-eta_slope * (Vd - eta_vds_th)))

    # R_B reconstruction (m2_body_gnd branch, no use_local_base):
    if m2_body_gnd and not use_local_base:
        body_terms = {
            "+iii*Iii_M1":  iii_to_body_factor * iii_gain * c["Iii_M1"],
            "+Igidl_M1":    c["Igidl_M1"],
            "+Igisl_M1":    c["Igisl_M1"],
            "+Igb_M1":      c["Igb_M1"],
            "-m1d*Ibs_M1":  -m1_d * c["Ibs_M1"],
            "-m1d*Ibd_M1":  -m1_d * c["Ibd_M1"],
            "-Ib_Q1":       -c["Ib_Q1"],
            "-Ib_lat_pair": -c["Ib_lat_pair"],
            "+I_well_body": c["I_well_body"],
            "-I_body_pdiode": -c["I_body_pdiode"],
            "(I_tat in pdiode)": c["I_tat"],
        }
    else:
        body_terms = {"NOTE": "non-default branch; consult code"}

    print("\n=== ngspice OP ===")
    print(f"  VG1={VG1}  VG2={VG2}  Vd={Vd}  Vsint={Vsint_ng}  Vb={Vb_ng}")
    print(f"  cfg.m2_body_gnd={m2_body_gnd}  bjt_emitter_to_gnd={bjt_emitter_to_gnd}  "
          f"use_local_base={use_local_base}")
    print(f"  iii_gain(Vds={Vd})={iii_gain:.4e}")
    print(f"  m1_diode_scale={m1_d}")

    print("\n=== R_Sint terms (A) ===")
    for k, v in sint_terms.items():
        print(f"  {k:<14s} = {v:+.6e}")
    print(f"  ----- R_Sint  = {R_S_v:+.6e}")

    print("\n=== R_B terms (A) ===")
    for k, v in body_terms.items():
        if isinstance(v, str):
            print(f"  {k}: {v}")
        else:
            print(f"  {k:<18s} = {v:+.6e}")
    print(f"  ----- R_B     = {R_B_v:+.6e}")

    print("\n=== Norms ===")
    print(f"  |R_Sint|        = {abs(R_S_v):.6e}")
    print(f"  |R_B|           = {abs(R_B_v):.6e}")
    print(f"  ||R||_inf       = {norm_inf:.6e}")

    if norm_inf < 1e-9:
        verdict = "BASIN_LOCK_ALONE (R-13 sufficient)"
    elif norm_inf > 1e-6:
        verdict = "STRUCTURAL_KCL_BUG (R-13 INSUFFICIENT, need M2.B fix)"
    else:
        verdict = "AMBIGUOUS (1e-9 <= ||R|| <= 1e-6)"
    print(f"\n=== VERDICT ===\n  {verdict}")

    summary = {
        "script": "z335_residual_probe",
        "ngspice_op": {"VG1": VG1, "VG2": VG2, "Vd": Vd,
                        "Vsint": Vsint_ng, "Vb": Vb_ng},
        "cfg_flags": {
            "m2_body_gnd": m2_body_gnd,
            "bjt_emitter_to_gnd": bjt_emitter_to_gnd,
            "use_local_base": use_local_base,
            "m1_diode_scale": m1_d,
            "iii_gain": iii_gain,
        },
        "R_Sint_terms": sint_terms,
        "R_B_terms": body_terms,
        "R_Sint": R_S_v,
        "R_B": R_B_v,
        "norm_inf": norm_inf,
        "verdict": verdict,
        "elapsed_s": time.time() - t0,
    }
    with open(SUMMARY, "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    print(f"\n[z335] wrote {SUMMARY}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
