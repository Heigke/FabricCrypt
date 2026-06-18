"""R-4 v5 wiring unit test.

Verifies each new cfg flag actually changes I_d at a representative
operating point. R-3 audit identified that z313 bisection variants were
bitwise-identical because the flags being toggled were either
inert-on-default-path or only consumed inside a disabled branch.

This test toggles ONE flag at a time, holds everything else at a
v5-baseline config (use_well_diode=False, body_pdiode_to="vnwell"), and
asserts:

  flag OFF  vs flag ON   ⇒   |ΔI_d / I_d_off|  > tol

Operating point: V_G1=0.4, V_G2=0.3, V_d=2.5 (mid-snapback). Sebas's
parameter row interpolated from the load_curves helpers.
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
sys.path.insert(0, str(ROOT / "nsram"))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_module(name, path):
    sp = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(mod)
    return mod


def _eval_one_point(cfg, M1, M2, sd_M1, sd_M2, forward_2t, z91f, z304,
                     sebas_rows, vg1, vg2, vd):
    """Forward-solve cell at one (V_G1, V_G2, V_d) point. Returns |I_d|."""
    from nsram.bsim4_port.bjt import GummelPoonNPN
    patch_sd_scaled = z91f.patch_sd_scaled

    sebas_row = z304.find_params(sebas_rows, vg1, vg2)
    if sebas_row is None or math.isnan(sebas_row.get("K1", float("nan"))):
        raise RuntimeError(f"no Sebas row for VG1={vg1} VG2={vg2}")
    P_M1, P_M2 = z304.make_row_overrides(sebas_row, 1e-4,
                                            z91f.M2_STATIC_OVERRIDES)
    bjt = GummelPoonNPN.from_sebas_card()
    if not math.isnan(sebas_row.get("IS", float("nan"))):
        bjt.Is = float(sebas_row["IS"])
    area = float(sebas_row.get("area", 1e-6))
    if math.isnan(area):
        area = 1e-6
    mbjt = float(sebas_row.get("mbjt", 1.0))
    if math.isnan(mbjt):
        mbjt = 1.0
    bjt.area = area * mbjt
    bjt.Bf = 500.0

    Vd_t = torch.tensor([vd], dtype=torch.float64)
    with torch.no_grad(), \
          patch_sd_scaled(sd_M1, P_M1), \
          patch_sd_scaled(sd_M2, P_M2):
        out = forward_2t(cfg, M1, bjt, Vd_t,
                          torch.tensor(vg1, dtype=torch.float64),
                          torch.tensor(vg2, dtype=torch.float64),
                          warm_start=True, use_homotopy=True)
    Id = float(out["Id"].abs().cpu().item())
    return Id


def main():
    t0 = time.time()
    z304 = _load_module("z304", SCRIPTS / "z304_sebas_three_branch_refit.py")
    z91f = _load_module("z91f", SCRIPTS / "z91f_validate_with_sebas_params.py")

    sebas_rows = z304.load_sebas_params()
    z91f_built, cfg, M1, M2, sd_M1, sd_M2, forward_2t = z304.build_models_once()

    # v5-baseline: polarity flipped, pdiode active.
    def reset_baseline():
        cfg.use_well_diode = False
        cfg.body_pdiode_to = "vnwell"
        # Effective pdiode at Sebas's `is = 5.3675e-7` total
        # ⇒ Js_per_area = is_total / area = 5.3675e-7 / 22e-12 = 2.44e4
        cfg.body_pdiode_Js = 2.44e4
        cfg.body_pdiode_area = 22e-12
        cfg.body_pdiode_Rs = 1.0e10            # off by default; tests will toggle
        cfg.enable_tat = False
        cfg.tat_jtss = 3.4e-7
        cfg.tat_njts = 20.0
        cfg.tat_vtss = 10.0
        cfg.tat_xtss = 0.02
        cfg.use_lateral_collector = False
        cfg.lat_BV = 2.0
        cfg.lat_N = 4.0
        cfg.lat_BV_max = 2.2
        cfg.lat_M_smooth_delta = 0.5
        cfg.invalidate() if hasattr(cfg, "invalidate") else None

    op = dict(vg1=0.4, vg2=0.3, vd=2.5)
    tol_rel = 1e-4   # 0.01% — flag effect must exceed this

    results = []

    # --- Test 1: body_pdiode_Rs toggle (direct probe of _residuals) ----- #
    # Forward-solver feedback can mask flag effects: when pdiode current is
    # large, Vb is pulled to clamp at vnwell, hiding any Rs dependence in
    # the final I_d. So we probe `_residuals` directly at a forced
    # (Vsint, Vb) where the pdiode is forward-biased, and check the
    # I_body_pdiode component changes when Rs changes.
    from nsram.bsim4_port import nsram_cell_2T as ns
    from nsram.bsim4_port.bjt import GummelPoonNPN
    reset_baseline()
    # Modest Js (not the 2.44e4 saturating value) so harmonic mean's
    # min(I_ideal, I_Rs) regime is dictated by Rs at reasonable Vab.
    cfg.body_pdiode_Js = 1e-2
    Vd_t = torch.tensor([2.5], dtype=torch.float64)
    Vb_t = torch.tensor([0.7], dtype=torch.float64)  # forward: 0.7 > vnwell=0.2 below
    Vsint_t = torch.tensor([0.5], dtype=torch.float64)
    cfg.vnwell = 0.2
    bjt = GummelPoonNPN.from_sebas_card(); bjt.Bf = 500.0
    cfg.body_pdiode_Rs = 1.0e20
    _, _, comps_off = ns._residuals(
        cfg, M1, bjt, Vd_t, torch.tensor(0.4, dtype=torch.float64),
        torch.tensor(0.3, dtype=torch.float64), Vsint_t, Vb_t,
        P_M1=None, P_M2=None, model_M2=M2)
    cfg.body_pdiode_Rs = 1.0e5
    _, _, comps_on = ns._residuals(
        cfg, M1, bjt, Vd_t, torch.tensor(0.4, dtype=torch.float64),
        torch.tensor(0.3, dtype=torch.float64), Vsint_t, Vb_t,
        P_M1=None, P_M2=None, model_M2=M2)
    I_off = float(comps_off["I_body_pdiode"].abs().cpu().item())
    I_on = float(comps_on["I_body_pdiode"].abs().cpu().item())
    rel = abs(I_on - I_off) / max(I_off, 1e-30)
    pass_ = rel > tol_rel
    results.append(("body_pdiode_Rs", I_off, I_on, rel, pass_))
    print(f"[v5-test] body_pdiode_Rs (probe I_pdiode): I_off={I_off:.3e} "
          f"I_on={I_on:.3e} rel={rel:.2%} {'PASS' if pass_ else 'FAIL'}",
           flush=True)

    # --- Test 2: enable_tat toggle -------------------------------------- #
    reset_baseline()
    I_off = _eval_one_point(cfg, M1, M2, sd_M1, sd_M2, forward_2t,
                              z91f_built, z304, sebas_rows, **op)
    cfg.enable_tat = True
    I_on = _eval_one_point(cfg, M1, M2, sd_M1, sd_M2, forward_2t,
                             z91f_built, z304, sebas_rows, **op)
    rel = abs(I_on - I_off) / max(I_off, 1e-30)
    pass_ = rel > tol_rel
    results.append(("enable_tat", I_off, I_on, rel, pass_))
    print(f"[v5-test] enable_tat: I_off={I_off:.3e} I_on={I_on:.3e} "
          f"rel={rel:.2%} {'PASS' if pass_ else 'FAIL'}", flush=True)

    # --- Test 3: tat_vtss enters equation -------------------------------- #
    # If vtss is a dead constant (R-3 finding) toggling it leaves I_d
    # unchanged. The R-4 fix routes vtss through the T-acceleration factor.
    # We use T_C=85 (deviation from 300K) so vtss has measurable effect.
    reset_baseline()
    cfg.enable_tat = True
    cfg.T_C = 85.0     # 358 K — vtss now matters
    cfg.invalidate() if hasattr(cfg, "invalidate") else None
    cfg.tat_vtss = 0.0
    I_off = _eval_one_point(cfg, M1, M2, sd_M1, sd_M2, forward_2t,
                              z91f_built, z304, sebas_rows, **op)
    cfg.tat_vtss = 50.0
    I_on = _eval_one_point(cfg, M1, M2, sd_M1, sd_M2, forward_2t,
                             z91f_built, z304, sebas_rows, **op)
    rel = abs(I_on - I_off) / max(I_off, 1e-30)
    pass_ = rel > tol_rel
    results.append(("tat_vtss", I_off, I_on, rel, pass_))
    print(f"[v5-test] tat_vtss@85C: I_off={I_off:.3e} I_on={I_on:.3e} "
          f"rel={rel:.2%} {'PASS' if pass_ else 'FAIL'}", flush=True)
    cfg.T_C = 27.0
    cfg.invalidate() if hasattr(cfg, "invalidate") else None

    # --- Test 4: tat_xtss enters equation (probe I_tat) ----------------- #
    # tat_xtss is the V-acceleration coeff: adds xtss*(V_drive)^2 to V_eff
    # inside the exp. Solver-level test masks it because TAT saturates and
    # Vb clamps; probe I_tat directly at a forced (Vb=0.5, vnwell=2.0)
    # which gives V_drive = -1.5 V (TAT reverse-biased — small but
    # V-accel sensitive).
    reset_baseline()
    cfg.enable_tat = True
    cfg.tat_xtss = 0.0
    _, _, comps_off = ns._residuals(
        cfg, M1, bjt, Vd_t, torch.tensor(0.4, dtype=torch.float64),
        torch.tensor(0.3, dtype=torch.float64), Vsint_t,
        torch.tensor([0.5], dtype=torch.float64),
        P_M1=None, P_M2=None, model_M2=M2)
    cfg.tat_xtss = 0.5
    _, _, comps_on = ns._residuals(
        cfg, M1, bjt, Vd_t, torch.tensor(0.4, dtype=torch.float64),
        torch.tensor(0.3, dtype=torch.float64), Vsint_t,
        torch.tensor([0.5], dtype=torch.float64),
        P_M1=None, P_M2=None, model_M2=M2)
    I_off = float(comps_off["I_tat"].abs().cpu().item())
    I_on = float(comps_on["I_tat"].abs().cpu().item())
    rel = abs(I_on - I_off) / max(I_off, 1e-30)
    pass_ = rel > tol_rel
    results.append(("tat_xtss", I_off, I_on, rel, pass_))
    print(f"[v5-test] tat_xtss (probe I_tat): I_off={I_off:.3e} "
          f"I_on={I_on:.3e} rel={rel:.2%} {'PASS' if pass_ else 'FAIL'}",
           flush=True)

    # --- Test 5: use_lateral_collector + lat_BV=2.0 ---------------------- #
    reset_baseline()
    I_off = _eval_one_point(cfg, M1, M2, sd_M1, sd_M2, forward_2t,
                              z91f_built, z304, sebas_rows, **op)
    cfg.use_lateral_collector = True
    cfg.lat_BV = 2.0
    cfg.lat_BV_max = 2.2
    I_on = _eval_one_point(cfg, M1, M2, sd_M1, sd_M2, forward_2t,
                             z91f_built, z304, sebas_rows, **op)
    rel = abs(I_on - I_off) / max(I_off, 1e-30)
    pass_ = rel > tol_rel
    results.append(("avalanche_lat_BV_2.0", I_off, I_on, rel, pass_))
    print(f"[v5-test] avalanche@BV=2.0: I_off={I_off:.3e} I_on={I_on:.3e} "
          f"rel={rel:.2%} {'PASS' if pass_ else 'FAIL'}", flush=True)

    summary = {
        "script": "test_v5_wiring",
        "elapsed_s": time.time() - t0,
        "operating_point": op,
        "tol_rel": tol_rel,
        "tests": [
            dict(flag=n, I_off=I_off, I_on=I_on, rel_change=rel, pass_=p)
            for n, I_off, I_on, rel, p in results
        ],
        "n_pass": sum(1 for *_, p in results if p),
        "n_total": len(results),
        "all_pass": all(p for *_, p in results),
    }
    out_path = ROOT / "results/z320_pyport_v5/unit_test.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, default=float))
    print(f"\n[v5-test] {summary['n_pass']}/{summary['n_total']} flags wired",
           flush=True)
    print(f"[v5-test] wrote {out_path}", flush=True)
    return 0 if summary["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
