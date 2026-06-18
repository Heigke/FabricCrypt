"""z326 — R-10 solver fix: escape Vb=Vd trivial fixed-point.

R-9 (z325) root-caused: at strong-bias OPs (V_d=2.0, V_G1=0.6) pyport
Newton converges to Vb~1.99999, Vsint~1.866 — a trivial basin where:
  * Iii=7.5e-48 (M1 off, Vds-Vdseff tiny)
  * NPN sub-threshold (Vbe=0.13, sourcing 9e-17 A)
  * Diodes reverse-biased
  * Cell-wide median = 3.01 dec (v5b) vs 0.99 (z304 with avalanche)

Fix strategies (try in order, stop at first that works):

  S1) Initial guess: Vb_init = max(Vd-0.7, 0.0). Biases first Newton
      iterate into physical basin where NPN base is fwd-biased
      (Vbe = Vb_init - Vsint_init ~ Vd/2 - 0.7 < 0; hmm — actually
      we need Vbe POSITIVE for NPN to draw current OUT of Vb.
      Vbe = Vb - Vsint. Vsint_init = Vd/2 = 1.0 at Vd=2.
      Vb_init = Vd - 0.7 = 1.3 → Vbe_init = 0.3 (forward, NPN ON,
      base sinks current out of body → keeps Vb<Vd). Per spec.)

  S2) Homotopy / continuation: solve at Vd=0, ramp up Vd in steps,
      carry Vb forward.

  S3) Tiny vnwell→Vb leak (Is=1e-18) to break symmetry.

Test bias: V_G1=0.6, V_G2=0.20, V_d=2.0. INFRA gate: Vb < Vd post-fix.

Gates (pre-registered 22:32):
  - INFRA: Vb < Vd at test bias
  - PASS-conservative: cell-wide median < 0.95 (beats z304's 0.99)
  - AMBITIOUS: < 0.5 dec
  - DIAGNOSTIC: V_G1=0.2 < 2.5 AND V_G1=0.6 < 0.7
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
OUT_DIR = ROOT / "results/z326_solver_fix"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PARTIAL = OUT_DIR / "partial.json"
SUMMARY = OUT_DIR / "summary.json"

sys.path.insert(0, str(ROOT / "nsram"))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64

ALPHA0_CONST = 7.842e-5
PDIODE_AREA = 2.2e-11
PDIODE_N = 1.0535
R_BODY_TABLE = {0.2: 1.0e10, 0.4: 1.0e9, 0.6: 1.0e8}
TAT_JTSS = 3.4e-7
TAT_NJTS = 20.0
TAT_VTSS = 10.0
TAT_XTSS = 0.02
LAT_BV_DISABLED = 1.0e6
BF_CARD = 10000.0


def _load_module(name, path):
    sp = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(mod)
    return mod


def configure_v5b_postfix(cfg, vg1):
    """v5b recipe (same as z325)."""
    cfg.use_well_diode = False
    cfg.body_pdiode_to = "off"
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
    if hasattr(cfg, "invalidate"):
        cfg.invalidate()


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


# ---------------------------------------------------------------------------
# Custom forward replacing forward_2t — bias-aware Vb_init per-point, plus
# optional strategy S2 (Vd-homotopy/continuation) and S3 (tiny well leak).
# ---------------------------------------------------------------------------

def forward_2t_S1(cfg, model_M1, model_M2, bjt, Vd_seq, VG1, VG2,
                   P_M1=None, P_M2=None, vb_offset=0.7):
    # NOTE: P_M1/P_M2 are passed to solve_2t_with_homotopy as None — the
    # caller wraps in `patch_sd_scaled` (sd.scaled[k] = v) which is the
    # correct override path; passing P_M1 to _residuals would route through
    # `_override_sd` which expects field-style attrs (e.g. sd.etab) and
    # crashes. Same idiom as z325.
    P_M1 = None; P_M2 = None
    """S1: per-point Vb_init = max(Vd - vb_offset, 0.0). Warm-start across
    Vd points DISABLED for Vb (keep Vsint warm).
    """
    from nsram.bsim4_port.nsram_cell_2T import solve_2t_with_homotopy
    Vd_seq = Vd_seq.to(DTYPE)
    VG1 = torch.as_tensor(VG1, dtype=DTYPE)
    VG2 = torch.as_tensor(VG2, dtype=DTYPE)
    T = int(Vd_seq.shape[0])
    Ids, Vss, Vbs, convs = [], [], [], []
    Vsint_warm = None
    for i in range(T):
        Vd_i = Vd_seq[i].unsqueeze(0)
        Vd_val = float(Vd_i.item())
        Vb_init = torch.tensor([max(Vd_val - vb_offset, 0.0)], dtype=DTYPE)
        if Vsint_warm is None:
            Vsint_init = (Vd_i * 0.5)
        else:
            Vsint_init = Vsint_warm
        out = solve_2t_with_homotopy(
            cfg, model_M1, bjt,
            Vd=Vd_i, VG1=VG1, VG2=VG2,
            P_M1=P_M1, P_M2=P_M2,
            Vsint_init=Vsint_init.expand_as(Vd_i),
            Vb_init=Vb_init.expand_as(Vd_i),
            model_M2=model_M2,
        )
        Ids.append(out["Id"].squeeze(0))
        Vss.append(out["Vsint"].squeeze(0))
        Vbs.append(out["Vb"].squeeze(0))
        convs.append(bool(torch.as_tensor(out["converged"]).all()))
        Vsint_warm = out["Vsint"].detach().squeeze(0)
    return {
        "Id": torch.stack(Ids),
        "Vsint": torch.stack(Vss),
        "Vb": torch.stack(Vbs),
        "converged": convs,
    }


def forward_2t_S2(cfg, model_M1, model_M2, bjt, Vd_seq, VG1, VG2,
                   P_M1=None, P_M2=None, n_homotopy_steps=6):
    P_M1 = None; P_M2 = None  # see forward_2t_S1 note
    """S2: Vd continuation. For each target Vd point, internally ramp Vd
    from 0 → target in n_homotopy_steps, carrying Vb forward. This
    guarantees we approach high-Vd OPs from a physical Vb<Vd trajectory.
    """
    from nsram.bsim4_port.nsram_cell_2T import solve_2t_with_homotopy
    Vd_seq = Vd_seq.to(DTYPE)
    VG1 = torch.as_tensor(VG1, dtype=DTYPE)
    VG2 = torch.as_tensor(VG2, dtype=DTYPE)
    T = int(Vd_seq.shape[0])
    Ids, Vss, Vbs, convs = [], [], [], []
    # Build a global Vd ramp from 0 to Vd_max, sampling target points.
    Vsint_warm = torch.tensor([0.0], dtype=DTYPE)
    Vb_warm = torch.tensor([0.0], dtype=DTYPE)
    Vd_min = 0.05
    for i in range(T):
        Vd_target = float(Vd_seq[i].item())
        # n_homotopy_steps intermediate climbs from current Vd to target.
        Vd_cur = Vd_min if i == 0 else float(Vd_seq[i-1].item())
        ramp = np.linspace(Vd_cur, Vd_target, n_homotopy_steps + 1)[1:]
        for j, vd_step in enumerate(ramp):
            Vd_t = torch.tensor([vd_step], dtype=DTYPE)
            out = solve_2t_with_homotopy(
                cfg, model_M1, bjt,
                Vd=Vd_t, VG1=VG1, VG2=VG2,
                P_M1=P_M1, P_M2=P_M2,
                Vsint_init=Vsint_warm.expand_as(Vd_t),
                Vb_init=Vb_warm.expand_as(Vd_t),
                model_M2=model_M2,
            )
            Vsint_warm = out["Vsint"].detach().squeeze(0).unsqueeze(0)
            Vb_warm = out["Vb"].detach().squeeze(0).unsqueeze(0)
        # Last `out` is at Vd_target.
        Ids.append(out["Id"].squeeze(0))
        Vss.append(out["Vsint"].squeeze(0))
        Vbs.append(out["Vb"].squeeze(0))
        convs.append(bool(torch.as_tensor(out["converged"]).all()))
    return {
        "Id": torch.stack(Ids),
        "Vsint": torch.stack(Vss),
        "Vb": torch.stack(Vbs),
        "converged": convs,
    }


# ---------------------------------------------------------------------------
# Single-bias test at V_G1=0.6, V_G2=0.20, V_d=2.0
# ---------------------------------------------------------------------------

def test_single_bias(strategy, cfg, M1, M2, sd_M1, sd_M2, sebas_rows, z91f,
                      vb_offset=0.7, cfg_overrides=None):
    """Verify Vb < Vd at the canonical test OP.

    Returns dict {Vb, Vsint, Id, converged, infra_pass}.
    """
    from nsram.bsim4_port.bjt import GummelPoonNPN
    VG1 = 0.6; VG2 = 0.20; Vd_val = 2.0
    configure_v5b_postfix(cfg, VG1)
    if cfg_overrides:
        for k, v in cfg_overrides.items():
            setattr(cfg, k, v)
        if hasattr(cfg, "invalidate"):
            cfg.invalidate()

    sebas_row = None
    for r in sebas_rows:
        if abs(r["VG1"] - VG1) < 1e-3 and abs(r["VG2"] - VG2) < 1e-3:
            sebas_row = r; break
    P_M1 = {}
    if not math.isnan(sebas_row.get("ETAB", float("nan"))):
        P_M1["etab"] = torch.tensor(sebas_row["ETAB"], dtype=DTYPE)
    if not math.isnan(sebas_row.get("K1", float("nan"))):
        P_M1["k1"] = torch.tensor(sebas_row["K1"], dtype=DTYPE)
    P_M1["alpha0"] = torch.tensor(ALPHA0_CONST, dtype=DTYPE)
    if not math.isnan(sebas_row.get("BETA0", float("nan"))):
        P_M1["beta0"] = torch.tensor(sebas_row["BETA0"], dtype=DTYPE)
    P_M2 = {}
    if not math.isnan(sebas_row.get("NFACTOR", float("nan"))):
        P_M2["nfactor"] = torch.tensor(sebas_row["NFACTOR"], dtype=DTYPE)
    for k, v in z91f.M2_STATIC_OVERRIDES.items():
        if k not in P_M2:
            P_M2[k] = torch.tensor(float(v), dtype=DTYPE)

    bjt = GummelPoonNPN.from_sebas_card()
    if not math.isnan(sebas_row.get("IS", float("nan"))):
        bjt.Is = float(sebas_row["IS"])
    area = float(sebas_row.get("area", 1e-6))
    if math.isnan(area): area = 1e-6
    mbjt = float(sebas_row.get("mbjt", 1.0))
    if math.isnan(mbjt): mbjt = 1.0
    bjt.area = area * mbjt
    bjt.Bf = BF_CARD

    Vd_seq = torch.tensor([Vd_val], dtype=DTYPE)
    with torch.no_grad(), patch_sd_scaled(sd_M1, P_M1), patch_sd_scaled(sd_M2, P_M2):
        if strategy == "S1":
            out = forward_2t_S1(cfg, M1, M2, bjt, Vd_seq, VG1, VG2,
                                 P_M1=P_M1, P_M2=P_M2, vb_offset=vb_offset)
        elif strategy == "S2":
            out = forward_2t_S2(cfg, M1, M2, bjt, Vd_seq, VG1, VG2,
                                 P_M1=P_M1, P_M2=P_M2)
        else:
            raise ValueError(strategy)

    Vb = float(out["Vb"].squeeze())
    Vsint = float(out["Vsint"].squeeze())
    Id = float(out["Id"].abs().squeeze())
    conv = bool(out["converged"][0])
    infra_pass = (Vb < Vd_val - 1e-4) and conv
    return {"strategy": strategy, "Vb": Vb, "Vsint": Vsint, "Id_A": Id,
             "Vd": Vd_val, "converged": conv, "infra_pass": infra_pass,
             "vb_offset": vb_offset}


# ---------------------------------------------------------------------------
# Full Sebas IV evaluation (cell-wide median + per-V_G1 breakdown)
# ---------------------------------------------------------------------------

def evaluate_full_iv(strategy, cfg, M1, M2, sd_M1, sd_M2, sebas_rows, curves,
                      z91f, z304, vb_offset=0.7, cfg_overrides=None):
    """Run all 33 IV curves with the selected strategy."""
    from nsram.bsim4_port.bjt import GummelPoonNPN
    log_eps = 1e-15
    per_curve = []
    for c in curves:
        VG1 = float(c["VG1"]); VG2 = float(c["VG2"])
        sebas_row = None
        for r in sebas_rows:
            if abs(r["VG1"] - VG1) < 1e-3 and abs(r["VG2"] - VG2) < 1e-3:
                sebas_row = r; break
        if sebas_row is None or math.isnan(sebas_row.get("K1", float("nan"))):
            continue
        configure_v5b_postfix(cfg, VG1)
        if cfg_overrides:
            for k, v in cfg_overrides.items():
                setattr(cfg, k, v)
            if hasattr(cfg, "invalidate"):
                cfg.invalidate()
        P_M1, P_M2 = z304.make_row_overrides(sebas_row, ALPHA0_CONST,
                                                z91f.M2_STATIC_OVERRIDES)
        bjt = GummelPoonNPN.from_sebas_card()
        if not math.isnan(sebas_row.get("IS", float("nan"))):
            bjt.Is = float(sebas_row["IS"])
        area = float(sebas_row.get("area", 1e-6))
        if math.isnan(area): area = 1e-6
        mbjt = float(sebas_row.get("mbjt", 1.0))
        if math.isnan(mbjt): mbjt = 1.0
        bjt.area = area * mbjt
        bjt.Bf = BF_CARD

        try:
            with torch.no_grad(), patch_sd_scaled(sd_M1, P_M1), \
                  patch_sd_scaled(sd_M2, P_M2):
                if strategy == "S1":
                    out = forward_2t_S1(cfg, M1, M2, bjt, c["Vd"], VG1, VG2,
                                         P_M1=P_M1, P_M2=P_M2,
                                         vb_offset=vb_offset)
                elif strategy == "S2":
                    out = forward_2t_S2(cfg, M1, M2, bjt, c["Vd"], VG1, VG2,
                                         P_M1=P_M1, P_M2=P_M2)
                else:
                    raise ValueError(strategy)
            Id_pred = out["Id"].abs()
            conv = torch.tensor([bool(x) for x in out["converged"]])
        except Exception as e:
            per_curve.append({"VG1": VG1, "VG2": VG2,
                                "log_rmse": float("inf"),
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
            # Also capture max(Vb) and frac with Vb>=Vd-1e-3 as diagnostic
            Vb_max = float(out["Vb"].max())
            Vb_ge_Vd_frac = float(((out["Vb"] >= (c["Vd"] - 1e-3)).float()).mean())
        else:
            rmse = float("inf"); signed = float("nan")
            Vb_max = float("nan"); Vb_ge_Vd_frac = float("nan")
        per_curve.append({"VG1": VG1, "VG2": VG2,
                            "log_rmse": rmse, "signed_dec": signed,
                            "n_conv": int(conv.sum()),
                            "Vb_max": Vb_max,
                            "Vb_ge_Vd_frac": Vb_ge_Vd_frac})

    # Aggregate
    finite = [pc for pc in per_curve if math.isfinite(pc["log_rmse"])]
    rmses_all = np.array([pc["log_rmse"] for pc in finite])
    cell_median = float(np.median(rmses_all)) if len(rmses_all) else float("inf")

    per_vg1 = {}
    for vg1 in (0.2, 0.4, 0.6):
        rms = [pc["log_rmse"] for pc in finite
                if abs(pc["VG1"] - vg1) < 1e-3]
        per_vg1[f"{vg1:.1f}"] = (float(np.median(rms)) if rms
                                   else float("inf"))

    return {
        "strategy": strategy,
        "cell_median_log_rmse": cell_median,
        "per_vg1_median": per_vg1,
        "n_finite": len(finite),
        "n_total": len(per_curve),
        "per_curve": per_curve,
    }


def save_partial(data):
    with open(PARTIAL, "w") as f:
        json.dump(data, f, indent=2, default=str)


def main():
    t0 = time.time()
    print(f"[z326] device={DEVICE}", flush=True)

    z304 = _load_module("z304", SCRIPTS / "z304_sebas_three_branch_refit.py")
    z91f, cfg, M1, M2, sd_M1, sd_M2, _ = z304.build_models_once()
    sebas_rows = z304.load_sebas_params()
    curves = z304.load_curves()
    print(f"[z326] models built; {len(curves)} curves ({time.time()-t0:.1f}s)",
          flush=True)

    summary = {
        "script": "z326_solver_fix",
        "start_time": time.time(),
        "z304_baseline_cell_median": 0.99,
        "v5b_baseline_cell_median": 3.01,
        "single_bias_tests": {},
        "full_iv_tests": {},
        "gates": {},
        "loc_changes": {
            "file": "scripts/z326_solver_fix.py (NEW)",
            "summary": ("No edits to nsram/bsim4_port/nsram_cell_2T.py — "
                          "fix implemented as a custom forward_2t wrapper "
                          "(forward_2t_S1, forward_2t_S2) that calls existing "
                          "solve_2t_with_homotopy with bias-aware Vb_init."),
            "key_code": ("forward_2t_S1: Vb_init = max(Vd - 0.7, 0.0) "
                          "per Vd point (~10 LOC at lines 130-165)"),
        },
    }
    save_partial(summary)

    # --- Strategy 1: bias-aware initial guess --------------------------------
    print("\n[z326] === Strategy 1: Vb_init = max(Vd - 0.7, 0.0) ===",
          flush=True)
    s1 = test_single_bias("S1", cfg, M1, M2, sd_M1, sd_M2, sebas_rows, z91f)
    print(f"  S1 single bias: Vb={s1['Vb']:.4f}, Vsint={s1['Vsint']:.4f}, "
          f"Id={s1['Id_A']:.3e}, conv={s1['converged']}, "
          f"infra_pass={s1['infra_pass']}", flush=True)
    summary["single_bias_tests"]["S1"] = s1
    save_partial(summary)

    working_strategy = None
    if s1["infra_pass"]:
        working_strategy = "S1"
        print("[z326] S1 INFRA PASS — using S1 for full IV", flush=True)
    else:
        # Try a couple of offsets before falling back to S2.
        for off in (0.5, 0.3, 1.0):
            s1b = test_single_bias("S1", cfg, M1, M2, sd_M1, sd_M2, sebas_rows,
                                     z91f, vb_offset=off)
            print(f"  S1(offset={off}): Vb={s1b['Vb']:.4f}, "
                  f"infra={s1b['infra_pass']}", flush=True)
            summary["single_bias_tests"][f"S1_off={off}"] = s1b
            save_partial(summary)
            if s1b["infra_pass"]:
                working_strategy = "S1"
                s1 = s1b
                summary["single_bias_tests"]["S1"] = s1
                print(f"[z326] S1 with offset={off} INFRA PASS", flush=True)
                break

    if working_strategy is None:
        # --- Strategy 2: Vd continuation ------------------------------------
        print("\n[z326] === Strategy 2: Vd continuation (0 → target) ===",
              flush=True)
        s2 = test_single_bias("S2", cfg, M1, M2, sd_M1, sd_M2, sebas_rows, z91f)
        print(f"  S2 single bias: Vb={s2['Vb']:.4f}, Vsint={s2['Vsint']:.4f}, "
              f"Id={s2['Id_A']:.3e}, conv={s2['converged']}, "
              f"infra_pass={s2['infra_pass']}", flush=True)
        summary["single_bias_tests"]["S2"] = s2
        save_partial(summary)
        if s2["infra_pass"]:
            working_strategy = "S2"
            print("[z326] S2 INFRA PASS — using S2 for full IV", flush=True)

    if working_strategy is None:
        # --- Strategy 3: physical symmetry-breaker. Re-enable body_pdiode
        # to GROUND (Vab = Vb, forward-bias when Vb > 0 → leaks body to GND).
        # R-9 root cause is body node is FLOATING at the trivial root with
        # no current path. With body_pdiode_to='gnd', current LEAVES body
        # when Vb is high, pinning Vb low enough for NPN/MOSFET to engage.
        print("\n[z326] === Strategy 3: body_pdiode_to='gnd' (symmetry break) ===",
              flush=True)
        s3_overrides = {"body_pdiode_to": "gnd"}
        s3 = test_single_bias("S1", cfg, M1, M2, sd_M1, sd_M2, sebas_rows,
                                z91f, vb_offset=0.7,
                                cfg_overrides=s3_overrides)
        print(f"  S3 single bias: Vb={s3['Vb']:.4f}, "
              f"Vsint={s3['Vsint']:.4f}, Id={s3['Id_A']:.3e}, "
              f"infra={s3['infra_pass']}", flush=True)
        s3["strategy"] = "S3_body_pdiode_gnd"
        summary["single_bias_tests"]["S3"] = s3
        save_partial(summary)
        if s3["infra_pass"]:
            working_strategy = "S3"
            print("[z326] S3 INFRA PASS", flush=True)
        else:
            # Last resort: body_pdiode to vnwell with vnwell=0 (= GND)
            print("\n[z326] === Strategy 3b: vnwell=0, body_pdiode_to=vnwell ===",
                  flush=True)
            s3b_overrides = {"vnwell": 0.0, "body_pdiode_to": "vnwell"}
            s3b = test_single_bias("S1", cfg, M1, M2, sd_M1, sd_M2, sebas_rows,
                                     z91f, vb_offset=0.7,
                                     cfg_overrides=s3b_overrides)
            print(f"  S3b: Vb={s3b['Vb']:.4f}, infra={s3b['infra_pass']}",
                  flush=True)
            s3b["strategy"] = "S3b_vnwell_0"
            summary["single_bias_tests"]["S3b"] = s3b
            save_partial(summary)
            if s3b["infra_pass"]:
                working_strategy = "S3b"
                s3 = s3b
            else:
                print("[z326] ALL STRATEGIES FAILED INFRA gate", flush=True)
                summary["status"] = "infra_fail"
                summary["elapsed_s"] = time.time() - t0
                with open(SUMMARY, "w") as f:
                    json.dump(summary, f, indent=2, default=str)
                return

    summary["working_strategy"] = working_strategy
    save_partial(summary)

    # --- Sweep over Rs values for the working strategy ---------------------
    # Cell-wide median is dominated by per-bias Id error. Stronger Vb pull
    # (smaller body_pdiode_Rs) → more device current → less overshoot.
    print(f"\n[z326] === Rs sweep with {working_strategy} ===", flush=True)
    rs_grid = [1.0e8, 1.0e7, 1.0e6, 1.0e5, 1.0e4]
    rs_results = {}
    for rs in rs_grid:
        if working_strategy == "S3":
            ov = {"body_pdiode_to": "gnd", "body_pdiode_Rs": rs}
        elif working_strategy == "S3b":
            ov = {"vnwell": 0.0, "body_pdiode_to": "vnwell",
                  "body_pdiode_Rs": rs}
        else:
            ov = {}
        sb = test_single_bias("S1", cfg, M1, M2, sd_M1, sd_M2, sebas_rows,
                                z91f, vb_offset=0.7, cfg_overrides=ov)
        print(f"  Rs={rs:.1e}: Vb={sb['Vb']:.4f}, Vsint={sb['Vsint']:.4f}, "
              f"Id={sb['Id_A']:.3e}", flush=True)
        rs_results[f"Rs_{rs:.0e}"] = sb
    summary["rs_sweep_single_bias"] = rs_results
    save_partial(summary)
    # Single-bias Id doesn't distinguish strategies well (at VG2=0.2 the
    # device is sub-Vth, Id ~1e-13 regardless of Vb). The discriminative
    # gate is full IV. We pick Rs by full-IV cell median, evaluating a
    # short grid. Rs values where Vb collapses near 0.65V are physically
    # cleanest (BJT/Mosfet body actually pinned).
    print(f"\n[z326] === Full-IV sweep over Rs ===", flush=True)
    rs_iv_results = {}
    iv_strategy = "S1"
    for rs in [1.0e8, 1.0e7, 1.0e6, 1.0e5]:
        if working_strategy == "S3":
            ov = {"body_pdiode_to": "gnd", "body_pdiode_Rs": rs}
        else:
            ov = {"vnwell": 0.0, "body_pdiode_to": "vnwell",
                  "body_pdiode_Rs": rs}
        t_iv = time.time()
        f_iv = evaluate_full_iv(iv_strategy, cfg, M1, M2, sd_M1, sd_M2,
                                  sebas_rows, curves, z91f, z304,
                                  vb_offset=0.7, cfg_overrides=ov)
        cm = f_iv["cell_median_log_rmse"]
        vg1m = f_iv["per_vg1_median"]
        print(f"  Rs={rs:.1e}: cell_med={cm:.3f}, "
              f"per_vg1={vg1m}  ({time.time()-t_iv:.0f}s)", flush=True)
        rs_iv_results[f"Rs_{rs:.0e}"] = {
            "cell_median": cm,
            "per_vg1": vg1m,
            "n_finite": f_iv["n_finite"],
            "n_total": f_iv["n_total"],
        }
        summary["rs_sweep_full_iv"] = rs_iv_results
        save_partial(summary)
    # Pick best by cell_median
    best_key = min(rs_iv_results.keys(),
                     key=lambda k: rs_iv_results[k]["cell_median"])
    best_rs = float(best_key.replace("Rs_", ""))
    summary["best_rs_full_iv"] = best_rs
    print(f"[z326] best_rs_full_iv={best_rs:.1e}, "
          f"cell_med={rs_iv_results[best_key]['cell_median']:.3f}", flush=True)

    # (Default-Rs full IV is the first entry in rs_iv_results above; no
    # need to repeat it here. The best_rs is selected by full-IV cell_med.)
    full = None  # placeholder
    save_partial(summary)

    # --- Final eval at best Rs ---------------------------------------------
    if working_strategy in ("S3", "S3b"):
        if working_strategy == "S3":
            ov_best = {"body_pdiode_to": "gnd", "body_pdiode_Rs": best_rs}
        else:
            ov_best = {"vnwell": 0.0, "body_pdiode_to": "vnwell",
                       "body_pdiode_Rs": best_rs}
        print(f"\n[z326] === Final IV at best_rs={best_rs:.1e} ===",
              flush=True)
        final_iv = evaluate_full_iv("S1", cfg, M1, M2, sd_M1, sd_M2,
                                      sebas_rows, curves, z91f, z304,
                                      vb_offset=0.7, cfg_overrides=ov_best)
        summary["final_iv"] = final_iv
        save_partial(summary)
        # Use final_iv for gate decision
        cm = final_iv["cell_median_log_rmse"]
        vg1_diag = final_iv["per_vg1_median"]
    else:
        cm = float("inf"); vg1_diag = {}
    summary["final_cell_median"] = cm
    summary["final_per_vg1"] = vg1_diag
    # The gate-relevant numbers:
    _ = vg1_diag  # placeholder for original code below
    gates = {
        "INFRA_Vb_lt_Vd_at_test_bias": bool(
            summary["single_bias_tests"][working_strategy]["infra_pass"]),
        "PASS_conservative_lt_0_95": bool(cm < 0.95),
        "AMBITIOUS_lt_0_5": bool(cm < 0.5),
        "DIAGNOSTIC_VG1_0_2_lt_2_5": bool(vg1_diag.get("0.2", math.inf) < 2.5),
        "DIAGNOSTIC_VG1_0_6_lt_0_7": bool(vg1_diag.get("0.6", math.inf) < 0.7),
    }
    summary["gates"] = gates
    summary["status"] = "complete"
    summary["elapsed_s"] = time.time() - t0
    print(f"\n[z326] GATES: {json.dumps(gates, indent=2)}", flush=True)

    with open(SUMMARY, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[z326] Wrote {SUMMARY}", flush=True)


if __name__ == "__main__":
    main()
