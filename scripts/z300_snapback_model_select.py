"""z300 — brute-force model selection over candidate physics terms for the
snapback shape-gap.

Hypothesis: prior parameter sweeps (F6 Bf×VAF, M3a.4 Bf-fine, etc.) left
the snapback shape-gap intact. The remaining lever is MISSING-PHYSICS, not
missing-parameter.

Candidate maskable terms (each ON/OFF):
  1. Rs(V_d)           variable source resistance: Rs = Rs0 + Rs1·V_d
  2. (skipped: C_well-sub — transient only; this run is DC)
  3. Self-heating      Δθ = R_th·I·V_d → effective T-dependent currents
  4. RaCBE             finite recombination on BJT: α_F_eff = α_F /(1+RaCBE)
  5. Body-source 2nd-term   second saturation-current path on body diode
  (We expose 4 binary masks since C_well-sub is DC-irrelevant.)

Objective:
  DC log-RMSE on the existing 33-row Sebas validation set (z91f baseline),
  + soft snapback-shape penalty (no NaN, monotone-then-fold, peak ≤ 4 V).
Optimiser: scipy.optimize.differential_evolution (CMA-ES equivalent budget).

Output: results/z300_snapback/summary.json
"""
from __future__ import annotations
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "4")

import json, math, time, re, csv
from pathlib import Path
from contextlib import contextmanager
import numpy as np
import torch

torch.set_default_dtype(torch.float64)
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data/sebas_2026_04_22"
OUT = ROOT / "results/z300_snapback"
OUT.mkdir(parents=True, exist_ok=True)

from nsram.bsim4_port.bjt import GummelPoonNPN
from nsram.bsim4_port.model_card import BSIM4Model
from nsram.bsim4_port.nsram_cell_2T import NSRAMCell2TConfig, forward_2t
from nsram.bsim4_port.temp import compute_size_dep
from nsram.bsim4_port.geometry import Geometry

# Re-use z91f loader helpers by exec-loading the file (avoids duplication).
import importlib.util as _ilu
_sp = _ilu.spec_from_file_location("z91f", ROOT / "scripts/z91f_validate_with_sebas_params.py")
z91f = _ilu.module_from_spec(_sp); _sp.loader.exec_module(z91f)

# ─────────────────────────────────────────────────────────────────────── #
# Build base infra (M1, M2, cfg, sd_M1, sd_M2)                            #
# ─────────────────────────────────────────────────────────────────────── #
def build_infra(use_bjt=True):
    text_M1 = (DATA / "M1_130DNWFB.txt").read_text()
    M1 = BSIM4Model.from_spice(text_M1, model_type="nmos")
    z91f.patch_model_values(M1, type_n=True)
    text_M2 = (DATA / "M2_130bulkNSRAM.txt").read_text()
    M2 = BSIM4Model.from_spice(text_M2, model_type="nmos")
    z91f.patch_model_values(M2, type_n=True)
    cfg = NSRAMCell2TConfig(use_iii=True, use_gidl=True, use_bjt=use_bjt,
                            newton_max_iters=25)
    sd_M1 = compute_size_dep(M1, Geometry(L=cfg.Ln, W=cfg.Wn), T_C=cfg.T_C)
    sd_M2 = compute_size_dep(M2, Geometry(L=cfg.Ln * cfg.M2_length_factor,
                                          W=cfg.Wn), T_C=cfg.T_C)
    cfg._sd_M1 = sd_M1
    cfg._sd_M2 = sd_M2
    return cfg, M1, M2, sd_M1, sd_M2


# ─────────────────────────────────────────────────────────────────────── #
# Candidate-term application — wraps a single forward eval.               #
# We modify (Vd, T_C, body diode Js, BJT alpha) externally per term.      #
# ─────────────────────────────────────────────────────────────────────── #
def predict_curve_with_terms(cfg, M1, M2, sd_M1, sd_M2, bjt_proto, sebas_row,
                              Vd_arr, terms, params, use_homotopy=False,
                              max_outer=2):
    """Evaluate Id along Vd_arr with given candidate terms.

    terms: dict of booleans {Rs, self_heat, RaCBE, body_2nd}
    params: continuous params {Rs0, Rs1, Rth, RaCBE, BS2_scale}
    """
    P_M1, P_M2 = z91f.make_overrides(sebas_row)

    # Clone BJT and apply RaCBE: α_F_eff = α_F /(1+RaCBE).
    # In Gummel-Poon, Bf = α_F/(1-α_F). So if α_F_eff = α_F/(1+r),
    # Bf_eff ≈ Bf / (1 + r·(1+Bf)) for small r·Bf. Simpler: scale Bf:
    bjt = z91f.make_bjt(sebas_row)
    if terms.get("RaCBE", False):
        r = float(params["RaCBE"])
        # honest map: alpha_F = Bf/(1+Bf); alpha_eff = alpha/(1+r); Bf_eff = alpha_eff/(1-alpha_eff)
        Bf0 = float(bjt.Bf)
        alpha = Bf0 / (1.0 + Bf0)
        alpha_eff = alpha / (1.0 + max(r, 0.0))
        Bf_eff = alpha_eff / max(1.0 - alpha_eff, 1e-6)
        bjt.Bf = float(Bf_eff)

    # Body-source 2nd-term: scale body p-diode Js by (1 + BS2_scale).
    # Implemented by toggling cfg.body_pdiode_to ON and scaling Js.
    saved_pd_to = cfg.body_pdiode_to
    saved_pd_Js = cfg.body_pdiode_Js
    if terms.get("body_2nd", False):
        cfg.body_pdiode_to = "vnwell"
        cfg.body_pdiode_Js = float(cfg.body_pdiode_Js) * (1.0 + float(params["BS2_scale"]))

    # Self-heating: shift cfg.T_C by ΔT = R_th · <P>. P = mean over Vd of Vd*Id_prev.
    # Simpler proxy: ΔT ∝ R_th · Vd · I_typ. Apply per-curve mean as constant T shift.
    # We do a 2-pass: first cold solve to estimate I, then warm-shift.
    saved_T_C = cfg.T_C
    saved_sd_M1 = cfg._sd_M1
    saved_sd_M2 = cfg._sd_M2

    # Effective drain after Rs(V_d) drop. We apply it pre-solve as Vd_eff:
    # The current depends on Id which depends on Vd_eff — iterate ~3 times.
    Vd_t = torch.tensor(Vd_arr, dtype=torch.float64)
    Vd_eff = Vd_t.clone()
    try:
        with torch.no_grad(), \
             z91f.patch_sd_scaled(sd_M1, P_M1), \
             z91f.patch_sd_scaled(sd_M2, P_M2):

            # Outer iterations: 1 cold pass + (Rs and/or SH) corrections.
            Id_prev = torch.zeros_like(Vd_t)
            for outer in range(max_outer):
                # ---- self-heating: ΔT_K from mean dissipation
                if terms.get("self_heat", False) and outer > 0:
                    R_th = float(params["Rth"])  # K/W
                    # dissipation per point P = Vd * Id
                    P_diss = (Vd_t * Id_prev.abs()).clamp(max=1e-2)  # cap 10 mW
                    dT = (R_th * P_diss).mean().item()
                    dT = max(min(dT, 200.0), -50.0)  # safety
                    new_T = float(saved_T_C) + dT
                    cfg.T_C = new_T
                    cfg._sd_M1 = compute_size_dep(M1, Geometry(L=cfg.Ln, W=cfg.Wn), T_C=new_T)
                    cfg._sd_M2 = compute_size_dep(M2, Geometry(
                        L=cfg.Ln * cfg.M2_length_factor, W=cfg.Wn), T_C=new_T)

                # ---- Rs(Vd) drop
                if terms.get("Rs", False) and outer > 0:
                    Rs0 = float(params["Rs0"]); Rs1 = float(params["Rs1"])
                    Rs = Rs0 + Rs1 * Vd_t
                    Vd_eff = (Vd_t - Rs * Id_prev.abs()).clamp(min=0.01)
                else:
                    Vd_eff = Vd_t

                out = forward_2t(cfg, M1, bjt,
                                  Vd_eff,
                                  torch.tensor(sebas_row["VG1"]),
                                  torch.tensor(sebas_row["VG2"]),
                                  warm_start=True, use_homotopy=use_homotopy,
                                  model_M2=M2)
                Id_new = out["Id"].abs()
                conv = torch.tensor([bool(x) for x in out["converged"]])
                if not (terms.get("Rs", False) or terms.get("self_heat", False)):
                    Id_prev = Id_new
                    break
                if torch.allclose(Id_new, Id_prev, rtol=1e-3, atol=1e-12):
                    Id_prev = Id_new
                    break
                Id_prev = Id_new
            return Id_prev.numpy(), conv.numpy()
    finally:
        cfg.T_C = saved_T_C
        cfg._sd_M1 = saved_sd_M1
        cfg._sd_M2 = saved_sd_M2
        cfg.body_pdiode_to = saved_pd_to
        cfg.body_pdiode_Js = saved_pd_Js


# ─────────────────────────────────────────────────────────────────────── #
# Validation set: 33-row from Sebas                                        #
# ─────────────────────────────────────────────────────────────────────── #
def load_validation():
    curves = z91f.load_curves()
    sebas_rows = z91f.load_sebas_params()
    valid = []
    for c in curves:
        row = z91f.find_params(sebas_rows, c["VG1"], c["VG2"])
        if row is None or math.isnan(row.get("K1", float("nan"))):
            continue
        valid.append((c, row))
    return valid


# ─────────────────────────────────────────────────────────────────────── #
# Snapback shape check: extend Vd from 0.05 to 4.5V at fixed bias.         #
# Use VG1=0.6, VG2=0.45 (high-VG1 snapback regime).                        #
# ─────────────────────────────────────────────────────────────────────── #
SNAPBACK_VD = np.linspace(0.1, 4.5, 14)

def snapback_check(cfg, M1, M2, sd_M1, sd_M2, valid_rows, terms, params):
    """Return (penalty, peak_Vd, peak_Id, n_nan)."""
    # Pick a representative high-VG1 row from validation
    target = None
    for c, row in valid_rows:
        if abs(row["VG1"] - 0.6) < 1e-3 and abs(row["VG2"] - 0.45) < 0.05:
            target = row; break
    if target is None:
        for c, row in valid_rows:
            if abs(row["VG1"] - 0.6) < 1e-3:
                target = row; break
    if target is None:
        target = valid_rows[-1][1]

    try:
        Id_pred, conv = predict_curve_with_terms(cfg, M1, M2, sd_M1, sd_M2,
                                                  None, target,
                                                  SNAPBACK_VD, terms, params,
                                                  use_homotopy=True, max_outer=2)
    except Exception:
        return 10.0, float("nan"), float("nan"), len(SNAPBACK_VD)

    n_nan = int((~np.isfinite(Id_pred)).sum())
    if n_nan > 0:
        return 5.0 + n_nan * 0.1, float("nan"), float("nan"), n_nan

    # Peak detection
    if not np.all(np.isfinite(Id_pred)):
        return 5.0, float("nan"), float("nan"), n_nan
    pk_idx = int(np.argmax(Id_pred))
    pk_V = float(SNAPBACK_VD[pk_idx])
    pk_I = float(Id_pred[pk_idx])

    penalty = 0.0
    # Want peak in [2.0, 4.0] V
    if pk_V < 2.0:
        penalty += 2.0 * (2.0 - pk_V)
    if pk_V > 4.0:
        penalty += 2.0 * (pk_V - 4.0)
    # Want monotone-then-fold: rise before peak, fall after
    if pk_idx > 1:
        rising = np.all(np.diff(Id_pred[:pk_idx+1]) >= -1e-15)
        if not rising:
            penalty += 0.5
    if pk_idx < len(SNAPBACK_VD) - 2:
        falling = Id_pred[-1] < pk_I  # weak: final < peak
        if not falling:
            penalty += 1.0
    return penalty, pk_V, pk_I, n_nan


# ─────────────────────────────────────────────────────────────────────── #
# Objective                                                                #
# ─────────────────────────────────────────────────────────────────────── #
class Evaluator:
    def __init__(self, cfg, M1, M2, sd_M1, sd_M2, valid):
        self.cfg, self.M1, self.M2 = cfg, M1, M2
        self.sd_M1, self.sd_M2 = sd_M1, sd_M2
        self.valid = valid
        self.cache = {}
        self.calls = 0

    def evaluate(self, mask, params, full=False):
        """mask: 4-tuple of bools; params: 5-dict of floats."""
        key = (tuple(mask), tuple(round(v, 6) for v in params.values()))
        if key in self.cache and not full:
            return self.cache[key]
        terms = {"Rs": bool(mask[0]),
                 "self_heat": bool(mask[1]),
                 "RaCBE": bool(mask[2]),
                 "body_2nd": bool(mask[3])}
        rmses = []
        for c, row in self.valid:
            try:
                Id_pred, conv = predict_curve_with_terms(
                    self.cfg, self.M1, self.M2, self.sd_M1, self.sd_M2,
                    None, row, c["Vd"].numpy(), terms, params,
                    use_homotopy=False, max_outer=2)
                Id_meas = c["Id"].numpy()
                conv = conv.astype(bool)
                if conv.sum() < 5:
                    rmses.append(10.0); continue
                log_p = np.log10(np.abs(Id_pred[conv]) + 1e-15)
                log_m = np.log10(np.abs(Id_meas[conv]) + 1e-15)
                rmse = float(np.sqrt(np.mean((log_p - log_m) ** 2)))
                if not np.isfinite(rmse):
                    rmse = 10.0
                rmses.append(rmse)
            except Exception:
                rmses.append(10.0)
        rmses = [r for r in rmses if np.isfinite(r)]
        med = float(np.median(rmses)) if rmses else 10.0

        penalty, pk_V, pk_I, n_nan = snapback_check(
            self.cfg, self.M1, self.M2, self.sd_M1, self.sd_M2,
            self.valid, terms, params)
        obj = med + 0.3 * penalty
        result = {"obj": obj, "dc_rmse_med": med, "snap_penalty": penalty,
                  "snap_peak_V": pk_V, "snap_peak_I": pk_I,
                  "snap_nan": n_nan, "mask": list(mask),
                  "params": params}
        self.cache[key] = result
        self.calls += 1
        return result


# ─────────────────────────────────────────────────────────────────────── #
# Optimizer driver                                                         #
# ─────────────────────────────────────────────────────────────────────── #
def vec_to_mask_params(x):
    """x = [m0,m1,m2,m3, Rs0,Rs1,Rth,RaCBE,BS2] ∈ R^9
       masks via sigmoid > 0.5."""
    mask = tuple(int(v > 0.5) for v in x[:4])
    params = {
        "Rs0": float(np.clip(x[4], 1.0, 1e5)),    # ohms
        "Rs1": float(np.clip(x[5], 0.0, 1e5)),    # ohms/V
        "Rth": float(np.clip(x[6], 1e2, 5e5)),    # K/W
        "RaCBE": float(np.clip(x[7], 0.0, 50.0)),  # dimensionless
        "BS2_scale": float(np.clip(x[8], 0.0, 100.0)),
    }
    return mask, params


def main():
    t0 = time.time()
    print(f"[z300] start at {time.strftime('%H:%M:%S')}", flush=True)

    cfg, M1, M2, sd_M1, sd_M2 = build_infra(use_bjt=True)
    valid_all = load_validation()
    # Sub-sample 6 diverse rows spanning (VG1, VG2) for the search; full set
    # used only on the final top-5 re-evaluation.
    valid_all_sorted = sorted(valid_all, key=lambda cr: (cr[1]["VG1"], cr[1]["VG2"]))
    if len(valid_all_sorted) > 6:
        idx = np.linspace(0, len(valid_all_sorted) - 1, 6).astype(int)
        valid_search = [valid_all_sorted[i] for i in idx]
    else:
        valid_search = valid_all_sorted
    print(f"[z300] loaded {len(valid_all)} validation curves, "
          f"search subset = {len(valid_search)}", flush=True)

    ev = Evaluator(cfg, M1, M2, sd_M1, sd_M2, valid_search)
    ev_full = Evaluator(cfg, M1, M2, sd_M1, sd_M2, valid_all)

    # Baseline (on search subset): all-OFF
    base_params = {"Rs0": 100.0, "Rs1": 0.0, "Rth": 1e4,
                   "RaCBE": 0.0, "BS2_scale": 0.0}
    base_search = ev.evaluate((0, 0, 0, 0), base_params)
    print(f"[z300] BASELINE-subset  DC log-RMSE={base_search['dc_rmse_med']:.3f}  "
          f"snap_pk_V={base_search['snap_peak_V']}  n_nan={base_search['snap_nan']}",
          flush=True)
    print(f"[z300] baseline-subset elapsed: {time.time()-t0:.1f}s", flush=True)

    # Brute-force enumerate all 16 mask combos with reasonable param init,
    # then refine the top-5 with a small differential_evolution (budget-aware).
    from itertools import product
    init_params = {"Rs0": 1000.0, "Rs1": 500.0, "Rth": 5e4,
                   "RaCBE": 1.0, "BS2_scale": 5.0}

    enum_results = []
    for mask in product([0, 1], repeat=4):
        r = ev.evaluate(mask, init_params)
        enum_results.append(r)
        flags = "".join(str(b) for b in mask)
        print(f"  [enum] mask={flags}  obj={r['obj']:.3f}  "
              f"dc={r['dc_rmse_med']:.3f}  snap_pen={r['snap_penalty']:.2f}  "
              f"pk_V={r['snap_peak_V']}", flush=True)

    enum_results.sort(key=lambda r: r["obj"])
    print(f"[z300] enum done at {time.time()-t0:.0f}s, refining top-3",
          flush=True)

    # Refine top-3 with DE on continuous params
    from scipy.optimize import differential_evolution
    bounds_cont = [(1.0, 1e5), (0.0, 1e5), (1e2, 5e5), (0.0, 20.0), (0.0, 50.0)]

    refined = []
    refine_budget_per_mask = 10  # very small — wall budget
    for top in enum_results[:2]:
        mask = tuple(top["mask"])
        def obj_fn(xc, mask=mask):
            params = {
                "Rs0": float(np.clip(xc[0], 1.0, 1e5)),
                "Rs1": float(np.clip(xc[1], 0.0, 1e5)),
                "Rth": float(np.clip(xc[2], 1e2, 5e5)),
                "RaCBE": float(np.clip(xc[3], 0.0, 50.0)),
                "BS2_scale": float(np.clip(xc[4], 0.0, 100.0)),
            }
            r = ev.evaluate(mask, params)
            return r["obj"]

        elapsed = time.time() - t0
        if elapsed > 2400:  # 40 min cap before refine
            print(f"[z300] budget exhausted at refine; skipping mask={mask}",
                  flush=True)
            refined.append(top); continue
        try:
            res = differential_evolution(
                obj_fn, bounds_cont, maxiter=refine_budget_per_mask,
                popsize=6, tol=1e-3, seed=42, polish=False,
                workers=1, updating="immediate")
            # Re-evaluate to get the full record
            params = {
                "Rs0": float(np.clip(res.x[0], 1.0, 1e5)),
                "Rs1": float(np.clip(res.x[1], 0.0, 1e5)),
                "Rth": float(np.clip(res.x[2], 1e2, 5e5)),
                "RaCBE": float(np.clip(res.x[3], 0.0, 50.0)),
                "BS2_scale": float(np.clip(res.x[4], 0.0, 100.0)),
            }
            best = ev.evaluate(mask, params)
            refined.append(best)
            flags = "".join(str(b) for b in mask)
            print(f"  [refine] mask={flags}  obj={best['obj']:.3f}  "
                  f"dc={best['dc_rmse_med']:.3f}  snap={best['snap_penalty']:.2f}  "
                  f"pk_V={best['snap_peak_V']}", flush=True)
        except Exception as e:
            print(f"  [refine] mask={mask}  FAILED: {e}", flush=True)
            refined.append(top)

    # Collect top-5 across enum+refine
    pool = enum_results + refined
    pool_unique = {}
    for r in pool:
        k = (tuple(r["mask"]), tuple(round(v, 4) for v in r["params"].values()))
        if k not in pool_unique or r["obj"] < pool_unique[k]["obj"]:
            pool_unique[k] = r
    top5 = sorted(pool_unique.values(), key=lambda r: r["obj"])[:5]

    # Re-evaluate top-5 on the FULL 25-curve validation set for honest reporting
    print(f"[z300] re-eval top-5 on full {len(valid_all)} curves", flush=True)
    top5_full = []
    for r in top5:
        try:
            full_r = ev_full.evaluate(tuple(r["mask"]), r["params"])
            top5_full.append(full_r)
            flags = "".join(str(b) for b in r["mask"])
            print(f"  [full] mask={flags}  dc_full={full_r['dc_rmse_med']:.3f}  "
                  f"pk_V={full_r['snap_peak_V']}", flush=True)
        except Exception as e:
            top5_full.append(r)
            print(f"  [full] mask={r['mask']} FAILED: {e}", flush=True)

    # Re-eval baseline on full set
    base_full = ev_full.evaluate((0, 0, 0, 0), base_params)
    print(f"[z300] baseline full DC RMSE = {base_full['dc_rmse_med']:.3f}",
          flush=True)

    top5 = top5_full
    # Verdict
    best = sorted(top5, key=lambda r: r["dc_rmse_med"])[0]
    base = base_full
    pk_V = best["snap_peak_V"]
    snap_ok = (pk_V is not None and np.isfinite(pk_V or float("nan"))
               and 2.0 <= (pk_V or -1) <= 4.0 and best["snap_nan"] == 0)
    pass_conservative = (snap_ok and
                        best["dc_rmse_med"] <= base["dc_rmse_med"] + 0.1)
    pass_ambitious = (snap_ok and
                     best["dc_rmse_med"] <= base["dc_rmse_med"] - 0.1)
    if pass_ambitious:
        verdict = "AMBITIOUS"
    elif pass_conservative:
        verdict = "PASS-CONSERVATIVE"
    else:
        verdict = "FAIL"

    term_names = ["Rs", "self_heat", "RaCBE", "body_2nd"]
    summary = {
        "elapsed_s": time.time() - t0,
        "baseline": {"dc_rmse_med": base["dc_rmse_med"],
                     "snap_peak_V": base["snap_peak_V"],
                     "snap_nan": base["snap_nan"]},
        "term_names": term_names,
        "top5": [{"mask_flags": dict(zip(term_names, r["mask"])),
                  "params": r["params"],
                  "dc_rmse_med": r["dc_rmse_med"],
                  "snap_peak_V": r["snap_peak_V"],
                  "snap_nan": r["snap_nan"],
                  "snap_penalty": r["snap_penalty"],
                  "obj": r["obj"]} for r in top5],
        "verdict": verdict,
        "n_evals": ev.calls,
        "n_valid_curves": len(valid),
        "gates": {
            "PASS-CONSERVATIVE": pass_conservative,
            "AMBITIOUS": pass_ambitious,
        },
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2,
                                                  default=str))
    print(f"\n[z300] DONE  elapsed={time.time()-t0:.0f}s  verdict={verdict}",
          flush=True)
    print(f"       baseline DC RMSE = {base['dc_rmse_med']:.3f}",
          flush=True)
    print(f"       best     DC RMSE = {best['dc_rmse_med']:.3f}  "
          f"mask={dict(zip(term_names, best['mask']))}", flush=True)
    print(f"       wrote {OUT}/summary.json", flush=True)


if __name__ == "__main__":
    main()
