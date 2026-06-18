"""z324 — Structural fixes D1+D2+D9 + Stage-1 liveness ablation + conditional Stage-2 BBO.

Code changes (applied in nsram/nsram/bsim4_port/nsram_cell_2T.py):
  D1: Q1 emitter rewired GND -> Sint  (lines ~513-525, ~537-550, ~736-760)
      - Vbe = Vb - Vsint  (was Vb)
      - R_Sint balance gets -Ie_Q1 term (BJT emitter current INTO Sint;
        SPICE convention Ie = -(Ic+Ib), so current INTO Sint = -Ie_Q1)
      - Local-base inner Newton also uses Vbe = Vb_local - Vsint
  D2: use_well_diode default True -> False  (line ~117 in dataclass)
      - LTSpice has zero explicit diodes; BSIM4 internal junction is the
        only Nwell-body path. body_pdiode_to default was already "off".
  D9: Bf default in GummelPoonNPN.from_sebas_card already 10000 (parasiticBJT.txt).
      - z324 simply does NOT override to small Bf in liveness/BBO; uses 10000.

Stage 1 (V_G1=0.6 only, 4 variants):
  V1 control:           D1+D2+D9 applied, all other knobs at v5b recipe
  V2 + well_diode:      use_well_diode=True, vnwell_Rs=1e8
  V3 + iii_kill:        iii_to_body_factor=0 (kill impact_ionization to body)
  V4 + bjt_kill:        mbjt=0 (disable parasitic NPN via area=0)

Liveness gates:
  STRUCTURE-CONFIRMED if V2 differs from V1 by >=1% RMSE
    AND (V3 or V4) shifts >= 0.5 dec
  STILL DEAD if V2 = V1 bitwise and V3,V4 both <0.1 dec shift

Stage 2 (conditional, 90 min budget, BBO over Bf x K1_LUT_scale x body_pdiode_Rs):
  3 V_G1 branches (0.2, 0.4, 0.6). Random search with ~30 trials.
  Gate: any combo < 1.5 dec at V_G1=0.6 AND cell-wide median < 1.5 dec.
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
DATA = ROOT / "data/sebas_2026_04_22"
OUT_DIR = ROOT / "results/z324_structural_fix"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "nsram"))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64

# v5b recipe constants (carried from z321)
ALPHA0_CONST = 7.842e-5
PDIODE_AREA = 2.2e-11
PDIODE_N = 1.0535
R_BODY_TABLE = {0.2: 1.0e10, 0.4: 1.0e9, 0.6: 1.0e8}
TAT_JTSS = 3.4e-7
TAT_NJTS = 20.0
TAT_VTSS = 10.0
TAT_XTSS = 0.02
LAT_BV_DISABLED = 1.0e6

# Bf source (parasiticBJT.txt) — D9 says use card default
BF_CARD = 10000.0

WALL_BUDGET_S = 3 * 3600
STAGE1_BUDGET_S = 30 * 60
STAGE2_BUDGET_S = 90 * 60


def _load_module(name, path):
    sp = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(mod)
    return mod


def configure_v5b_postfix(cfg, vg1):
    """v5b recipe AFTER applying D1+D2+D9 defaults.

    Note: D2 makes use_well_diode default False; body_pdiode_to default "off".
    We honor that here unless an ablation overrides.
    """
    # Topology defaults already applied via dataclass; reaffirm:
    cfg.use_well_diode = False        # D2
    cfg.body_pdiode_to = "off"        # D2: rely on BSIM4 internal junction
    if hasattr(cfg, "z310_enable_vnwell_diode"):
        cfg.z310_enable_vnwell_diode = False

    cfg.body_pdiode_area = PDIODE_AREA
    cfg.body_pdiode_n = PDIODE_N
    cfg.Cbody = 7e-15

    cfg.body_pdiode_Rs = float(R_BODY_TABLE.get(round(vg1, 2), 1.0e9))
    cfg.vnwell_Rs = 1.0e30

    cfg.use_lateral_collector = False
    cfg.lat_BV = LAT_BV_DISABLED
    cfg.lat_BV_max = LAT_BV_DISABLED * 1.1

    cfg.enable_tat = True
    cfg.tat_jtss = TAT_JTSS
    cfg.tat_njts = TAT_NJTS
    cfg.tat_vtss = TAT_VTSS
    cfg.tat_xtss = TAT_XTSS

    if hasattr(cfg, "z313_enable_tat"):
        cfg.z313_enable_tat = False

    cfg.invalidate() if hasattr(cfg, "invalidate") else None


def run_branch(z304, vg1, bf, bjt_area_mult, cfg, M1, M2, sd_M1, sd_M2,
                forward_2t, sebas_rows, curves, z91f_built,
                iii_kill=False):
    """Wrapper that uses z304.evaluate_cell but with optional area-mult on BJT
    and optional iii kill.

    Trick to kill IMPACT_IONIZATION->body: set ALPHA0 to ~0 via row overrides.
    To kill BJT: scale bjt.area by 0.
    Both injected via monkey-patching evaluate_cell's params, but simplest is
    to subclass evaluate_cell — we copy minimal logic.
    """
    from nsram.bsim4_port.bjt import GummelPoonNPN

    @contextmanager
    def patch_sd_scaled(sd, overrides):
        if not overrides:
            yield; return
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

    log_eps = 1e-15
    per_curve = []
    for c in curves:
        sebas_row = z304.find_params(sebas_rows, c["VG1"], c["VG2"])
        if sebas_row is None or math.isnan(sebas_row.get("K1", float("nan"))):
            continue
        alpha0 = 1e-20 if iii_kill else ALPHA0_CONST
        P_M1, P_M2 = z304.make_row_overrides(
            sebas_row, alpha0, z91f_built.M2_STATIC_OVERRIDES)

        bjt = GummelPoonNPN.from_sebas_card()
        if not math.isnan(sebas_row.get("IS", float("nan"))):
            bjt.Is = float(sebas_row["IS"])
        area = float(sebas_row.get("area", 1e-6))
        if math.isnan(area):
            area = 1e-6
        mbjt = float(sebas_row.get("mbjt", 1.0))
        if math.isnan(mbjt):
            mbjt = 1.0
        bjt.area = area * mbjt * bjt_area_mult
        bjt.Bf = float(bf)

        try:
            with torch.no_grad(), \
                  patch_sd_scaled(sd_M1, P_M1), \
                  patch_sd_scaled(sd_M2, P_M2):
                out = forward_2t(cfg, M1, bjt,
                                   c["Vd"], torch.tensor(c["VG1"]),
                                   torch.tensor(c["VG2"]),
                                   warm_start=True, use_homotopy=True)
            Id_pred = out["Id"].abs()
            conv = torch.tensor([bool(x) for x in out["converged"]])
        except Exception as e:
            per_curve.append({"VG2": c["VG2"], "log_rmse": float("inf"),
                                "signed_dec": float("nan"), "n_conv": 0,
                                "err": str(e)[:120]})
            continue
        log_p = torch.log10(Id_pred + log_eps)
        log_m = torch.log10(c["Id"] + log_eps)
        if conv.any():
            mask = conv
            diff = (log_p[mask] - log_m[mask])
            rmse = float(torch.sqrt((diff ** 2).mean()))
            signed = float(torch.median(diff))
        else:
            rmse = float("inf"); signed = float("nan")
        per_curve.append({"VG2": float(c["VG2"]), "log_rmse": rmse,
                           "signed_dec": signed, "n_conv": int(conv.sum())})
    finite = [pc for pc in per_curve if math.isfinite(pc["log_rmse"])]
    if finite:
        rmses = np.array([pc["log_rmse"] for pc in finite])
        med = float(np.median(rmses))
        p90 = float(np.percentile(rmses, 90))
    else:
        med = float("inf"); p90 = float("inf")
    return {"median_log_rmse": med, "p90_log_rmse": p90,
            "n_finite": len(finite), "n_total": len(per_curve),
            "per_curve": per_curve}


def stage1_liveness(t0, z304, z91f_built, cfg, M1, M2, sd_M1, sd_M2,
                     forward_2t, sebas_rows, curves_by_vg1):
    """Stage 1: 4 variants at V_G1=0.6."""
    vg1 = 0.6
    curves = curves_by_vg1[vg1]
    results = {}

    # V1 control: D1+D2+D9 applied, defaults from class
    configure_v5b_postfix(cfg, vg1)
    print(f"[stage1] V1 control (D1+D2+D9) ...", flush=True)
    t = time.time()
    r1 = run_branch(z304, vg1, BF_CARD, 1.0, cfg, M1, M2, sd_M1, sd_M2,
                     forward_2t, sebas_rows, curves, z91f_built)
    print(f"[stage1] V1: med={r1['median_log_rmse']:.3f}  ({time.time()-t:.1f}s)",
          flush=True)
    results["V1_control"] = r1

    # V2 + well_diode
    configure_v5b_postfix(cfg, vg1)
    cfg.use_well_diode = True
    cfg.vnwell_Rs = 1e8
    cfg.invalidate() if hasattr(cfg, "invalidate") else None
    print(f"[stage1] V2 + well_diode (vnwell_Rs=1e8) ...", flush=True)
    t = time.time()
    r2 = run_branch(z304, vg1, BF_CARD, 1.0, cfg, M1, M2, sd_M1, sd_M2,
                     forward_2t, sebas_rows, curves, z91f_built)
    print(f"[stage1] V2: med={r2['median_log_rmse']:.3f}  ({time.time()-t:.1f}s)",
          flush=True)
    results["V2_well_diode"] = r2

    # V3 + iii_kill (ALPHA0 -> ~0)
    configure_v5b_postfix(cfg, vg1)
    print(f"[stage1] V3 + iii_kill (ALPHA0=1e-20) ...", flush=True)
    t = time.time()
    r3 = run_branch(z304, vg1, BF_CARD, 1.0, cfg, M1, M2, sd_M1, sd_M2,
                     forward_2t, sebas_rows, curves, z91f_built,
                     iii_kill=True)
    print(f"[stage1] V3: med={r3['median_log_rmse']:.3f}  ({time.time()-t:.1f}s)",
          flush=True)
    results["V3_iii_kill"] = r3

    # V4 + bjt_kill (area*0)
    configure_v5b_postfix(cfg, vg1)
    print(f"[stage1] V4 + bjt_kill (area=0) ...", flush=True)
    t = time.time()
    r4 = run_branch(z304, vg1, BF_CARD, 0.0, cfg, M1, M2, sd_M1, sd_M2,
                     forward_2t, sebas_rows, curves, z91f_built)
    print(f"[stage1] V4: med={r4['median_log_rmse']:.3f}  ({time.time()-t:.1f}s)",
          flush=True)
    results["V4_bjt_kill"] = r4

    # Verdict
    m1v = r1["median_log_rmse"]
    m2v = r2["median_log_rmse"]
    m3v = r3["median_log_rmse"]
    m4v = r4["median_log_rmse"]

    def safe(x): return x if math.isfinite(x) else float("nan")

    # 1% RMSE shift in log space ~ |m2-m1|/max(m1,1e-3) > 0.01
    v2_shift_pct = (abs(m2v - m1v) / max(abs(m1v), 1e-3)) if all(map(math.isfinite, [m1v, m2v])) else float("inf")
    v3_shift_dec = abs(m3v - m1v) if all(map(math.isfinite, [m1v, m3v])) else float("inf")
    v4_shift_dec = abs(m4v - m1v) if all(map(math.isfinite, [m1v, m4v])) else float("inf")

    structure_confirmed = (v2_shift_pct >= 0.01) and (v3_shift_dec >= 0.5 or v4_shift_dec >= 0.5)
    still_dead = (v2_shift_pct < 1e-6) and (v3_shift_dec < 0.1) and (v4_shift_dec < 0.1)

    verdict = {
        "v2_shift_pct_vs_v1": v2_shift_pct,
        "v3_shift_dec_vs_v1": v3_shift_dec,
        "v4_shift_dec_vs_v1": v4_shift_dec,
        "structure_confirmed": bool(structure_confirmed),
        "still_dead": bool(still_dead),
    }
    print(f"[stage1] verdict: structure={structure_confirmed} dead={still_dead}",
          flush=True)
    return results, verdict


def stage2_bbo(t0, t_budget_s, z304, z91f_built, cfg, M1, M2, sd_M1, sd_M2,
                forward_2t, sebas_rows, curves_by_vg1):
    """Random search BBO over (Bf, K1_LUT_scale, body_pdiode_Rs)."""
    import random
    rng = random.Random(2026)
    vg1_list = [0.2, 0.4, 0.6]
    trials = []
    t_stage2_start = time.time()
    trial_i = 0
    while True:
        if time.time() - t_stage2_start > t_budget_s:
            break
        if time.time() - t0 > WALL_BUDGET_S - 60:
            break
        trial_i += 1
        Bf = 10.0 ** rng.uniform(math.log10(50), math.log10(50000))
        K1_scale = rng.uniform(0.5, 2.0)
        body_pdiode_Rs = 10.0 ** rng.uniform(6.0, 12.0)

        per_branch = {}
        all_rmse = []
        for vg1 in vg1_list:
            configure_v5b_postfix(cfg, vg1)
            # K1_scale: enable body_pdiode "vnwell" so Rs has effect? No —
            # D2 says rely on BSIM. Instead, K1_scale modifies the K1
            # override per row. We achieve this by post-multiplying.
            # body_pdiode_Rs: setting is a no-op when body_pdiode_to='off',
            # but we expose it for completeness in case the search reveals
            # something. Track it for record.
            cfg.body_pdiode_Rs = float(body_pdiode_Rs)
            cfg.invalidate() if hasattr(cfg, "invalidate") else None
            curves = curves_by_vg1[vg1]
            # Apply K1_scale by wrapping evaluate
            r = run_branch_k1scale(
                z304, vg1, Bf, 1.0, cfg, M1, M2, sd_M1, sd_M2,
                forward_2t, sebas_rows, curves, z91f_built, K1_scale)
            per_branch[str(vg1)] = {
                "median_log_rmse": r["median_log_rmse"],
                "p90_log_rmse": r["p90_log_rmse"],
            }
            all_rmse.extend([pc["log_rmse"] for pc in r["per_curve"]
                              if math.isfinite(pc["log_rmse"])])
        cw = float(np.median(all_rmse)) if all_rmse else float("inf")
        vg6 = per_branch.get("0.6", {}).get("median_log_rmse", float("inf"))
        trials.append({
            "trial": trial_i, "Bf": Bf, "K1_scale": K1_scale,
            "body_pdiode_Rs": body_pdiode_Rs,
            "per_branch": per_branch, "cell_wide_median": cw,
            "gate_lt_1_5_at_vg1_0_6": bool(vg6 < 1.5),
            "gate_cellwide_lt_1_5": bool(cw < 1.5),
        })
        print(f"[stage2 #{trial_i}] Bf={Bf:.0f} K1x={K1_scale:.2f} "
              f"Rs={body_pdiode_Rs:.1e} cw={cw:.3f} vg6={vg6:.3f}",
              flush=True)
        # Save progress
        (OUT_DIR / "stage2_progress.json").write_text(
            json.dumps({"trials": trials}, indent=2, default=float))

    valid = [t for t in trials if math.isfinite(t["cell_wide_median"])]
    best = min(valid, key=lambda t: t["cell_wide_median"]) if valid else None
    return {"trials": trials, "best": best}


def run_branch_k1scale(z304, vg1, bf, bjt_area_mult, cfg, M1, M2, sd_M1, sd_M2,
                        forward_2t, sebas_rows, curves, z91f_built, K1_scale):
    """run_branch variant with K1 scale post-applied."""
    from nsram.bsim4_port.bjt import GummelPoonNPN

    @contextmanager
    def patch_sd_scaled(sd, overrides):
        if not overrides:
            yield; return
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

    log_eps = 1e-15
    per_curve = []
    for c in curves:
        sebas_row = z304.find_params(sebas_rows, c["VG1"], c["VG2"])
        if sebas_row is None or math.isnan(sebas_row.get("K1", float("nan"))):
            continue
        P_M1, P_M2 = z304.make_row_overrides(
            sebas_row, ALPHA0_CONST, z91f_built.M2_STATIC_OVERRIDES)
        # Scale K1 in P_M1
        if "k1" in P_M1:
            P_M1["k1"] = P_M1["k1"] * K1_scale

        bjt = GummelPoonNPN.from_sebas_card()
        if not math.isnan(sebas_row.get("IS", float("nan"))):
            bjt.Is = float(sebas_row["IS"])
        area = float(sebas_row.get("area", 1e-6))
        if math.isnan(area):
            area = 1e-6
        mbjt = float(sebas_row.get("mbjt", 1.0))
        if math.isnan(mbjt):
            mbjt = 1.0
        bjt.area = area * mbjt * bjt_area_mult
        bjt.Bf = float(bf)

        try:
            with torch.no_grad(), \
                  patch_sd_scaled(sd_M1, P_M1), \
                  patch_sd_scaled(sd_M2, P_M2):
                out = forward_2t(cfg, M1, bjt,
                                   c["Vd"], torch.tensor(c["VG1"]),
                                   torch.tensor(c["VG2"]),
                                   warm_start=True, use_homotopy=True)
            Id_pred = out["Id"].abs()
            conv = torch.tensor([bool(x) for x in out["converged"]])
        except Exception as e:
            per_curve.append({"VG2": c["VG2"], "log_rmse": float("inf"),
                                "signed_dec": float("nan"), "n_conv": 0,
                                "err": str(e)[:120]})
            continue
        log_p = torch.log10(Id_pred + log_eps)
        log_m = torch.log10(c["Id"] + log_eps)
        if conv.any():
            mask = conv
            diff = (log_p[mask] - log_m[mask])
            rmse = float(torch.sqrt((diff ** 2).mean()))
            signed = float(torch.median(diff))
        else:
            rmse = float("inf"); signed = float("nan")
        per_curve.append({"VG2": float(c["VG2"]), "log_rmse": rmse,
                           "signed_dec": signed, "n_conv": int(conv.sum())})
    finite = [pc for pc in per_curve if math.isfinite(pc["log_rmse"])]
    if finite:
        rmses = np.array([pc["log_rmse"] for pc in finite])
        med = float(np.median(rmses))
        p90 = float(np.percentile(rmses, 90))
    else:
        med = float("inf"); p90 = float("inf")
    return {"median_log_rmse": med, "p90_log_rmse": p90,
            "n_finite": len(finite), "n_total": len(per_curve),
            "per_curve": per_curve}


def main():
    t0 = time.time()
    print(f"[z324] device={DEVICE} budget={WALL_BUDGET_S}s", flush=True)

    z304 = _load_module("z304", SCRIPTS / "z304_sebas_three_branch_refit.py")
    z91f_built, cfg, M1, M2, sd_M1, sd_M2, forward_2t = z304.build_models_once()
    sebas_rows = z304.load_sebas_params()
    print(f"[z324] models built ({time.time()-t0:.1f}s)", flush=True)

    curves_by_vg1 = {vg1: z304.load_curves(vg1_filter=vg1)
                      for vg1 in [0.2, 0.4, 0.6]}

    # Stage 1
    stage1_results, stage1_verdict = stage1_liveness(
        t0, z304, z91f_built, cfg, M1, M2, sd_M1, sd_M2,
        forward_2t, sebas_rows, curves_by_vg1)

    # Stage 2 (conditional)
    stage2 = None
    if stage1_verdict["structure_confirmed"]:
        remaining = WALL_BUDGET_S - (time.time() - t0) - 120
        budget = min(STAGE2_BUDGET_S, max(0, remaining))
        print(f"[z324] Stage 1 STRUCTURE-CONFIRMED → Stage 2 BBO "
              f"(budget={budget:.0f}s)", flush=True)
        if budget > 60:
            stage2 = stage2_bbo(
                t0, budget, z304, z91f_built, cfg, M1, M2, sd_M1, sd_M2,
                forward_2t, sebas_rows, curves_by_vg1)
    else:
        print(f"[z324] Stage 1 not confirmed (still_dead={stage1_verdict['still_dead']})"
              f" → SKIP Stage 2", flush=True)

    summary = {
        "script": "z324_structural_fix",
        "elapsed_s": time.time() - t0,
        "device": str(DEVICE),
        "code_changes": {
            "D1_emitter_to_Sint": {
                "file": "nsram/nsram/bsim4_port/nsram_cell_2T.py",
                "lines": "~513-525, ~537-550, ~736-760",
                "summary": "Vbe = Vb - Vsint (was Vb); R_Sint += -Ie_Q1; local-base also updated",
                "loc_changed_approx": 18,
            },
            "D2_drop_explicit_diodes": {
                "file": "nsram/nsram/bsim4_port/nsram_cell_2T.py",
                "lines": "~117",
                "summary": "use_well_diode default True->False; body_pdiode_to already 'off'",
                "loc_changed_approx": 8,
            },
            "D9_Bf_from_card": {
                "file": "nsram/nsram/bsim4_port/bjt.py + experiment usage",
                "lines": "bjt.py:56 already 10000.0; z324 honors default",
                "summary": "z324 uses Bf=10000 (parasiticBJT.txt) not 50/200/etc",
                "loc_changed_approx": 0,
            },
        },
        "stage1_variants": {
            k: {"median_log_rmse": v["median_log_rmse"],
                "p90_log_rmse": v["p90_log_rmse"],
                "n_finite": v["n_finite"], "n_total": v["n_total"]}
            for k, v in stage1_results.items()
        },
        "stage1_verdict": stage1_verdict,
        "stage2": (None if stage2 is None else {
            "n_trials": len(stage2["trials"]),
            "best": stage2["best"],
            "trials": stage2["trials"],
        }),
        "z304_baseline_cellwide": 0.99,
        "v5b_cellwide": 3.01,
        "locked_gate_cellwide_lt_1_5": (
            (stage2 and stage2["best"] and
             stage2["best"]["cell_wide_median"] < 1.5)
            if stage2 else None
        ),
    }
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, default=float))
    print(f"[z324] wrote {OUT_DIR/'summary.json'} ({time.time()-t0:.1f}s)",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
