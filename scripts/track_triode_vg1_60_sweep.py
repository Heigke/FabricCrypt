#!/usr/bin/env python3
"""Track Triode — BSIM4 + BJT parameter sensitivity audit for VG1=0.60.

Goal: explain the 46x current shortfall at VG1=0.60 (strong-inversion, high Vds).
Baseline: pic3.build_pyport_base() (median 1.163 dec, VG1=0.6 triode RMSE ~1.93).

For each parameter in the strong-inversion-saturation knob set, sweep multipliers
{0.5, 0.8, 1.0, 1.2, 2.0} (additive shifts {-0.05, -0.02, 0, +0.02, +0.05} V for
threshold-like params). All other parameters locked at v5.3 baseline.

Re-run just the 5 VG1=0.60 biases worst at baseline (fwd+bwd, n=10 traces).
Identify whether a single knob (or 2-3 combination) brings VG1=0.6 RMSE
from 1.93 dec to <=0.5 dec WITHOUT regressing VG1=0.2 / VG1=0.4.

Out: results/track_triode_vg1_60/{param_sensitivity.json, verdict.md}.
"""
from __future__ import annotations
import os, sys, math, json, time, traceback
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")

from pathlib import Path
import numpy as np
import torch
torch.set_default_dtype(torch.float64)

ROOT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "nsram"))
OUT = ROOT / "results/track_triode_vg1_60"; OUT.mkdir(parents=True, exist_ok=True)

import importlib.util
sp = importlib.util.spec_from_file_location("pic3",
        ROOT / "scripts/pillar_I_C3_jts_tat.py")
pic3 = importlib.util.module_from_spec(sp); sp.loader.exec_module(pic3)
from nsram.bsim4_port.nsram_cell_2T import forward_2t


def run_one_pass(cfg, M1, M2, bjt, curves, sebas_rows,
                 sd_overrides_M1=None, bjt_overrides=None, vmin=0.3):
    """Run fwd+bwd over given curves; apply EXTRA sd_M1.scaled override on top
    of csv per-bias overrides. Returns rows list-of-dict.
    sd_overrides_M1: dict {param_name: scalar value} (replaces csv if same key).
    bjt_overrides: dict patched onto bjt and restored.
    """
    rows = []
    sd_M1 = cfg.size_dep_M1(M1); sd_M2 = cfg.size_dep_M2(M2)

    # apply bjt overrides
    bjt_saved = {}
    if bjt_overrides:
        for k, v in bjt_overrides.items():
            bjt_saved[k] = getattr(bjt, k)
            setattr(bjt, k, float(v))
    try:
        for c in curves:
            row_sebas, _ = pic3.find_or_impute_row(sebas_rows, c["VG1"], c["VG2"])
            P_M1, P_M2 = pic3.make_overrides(row_sebas)
            # layer extra sweep override LAST so it wins over csv etab/k1/...
            if sd_overrides_M1:
                if P_M1 is None: P_M1 = {}
                for k, v in sd_overrides_M1.items():
                    P_M1[k] = float(v)
            for branch, vdk, idk in (("fwd","fwd_Vd","fwd_Id"),
                                     ("bwd","bwd_Vd","bwd_Id")):
                Vd_np = c[vdk]; Id_np = c[idk]
                Vd = torch.tensor(Vd_np, dtype=torch.float64)
                try:
                    with pic3.patch_sd_scaled(sd_M1, P_M1), pic3.patch_sd_scaled(sd_M2, P_M2):
                        out = forward_2t(cfg, model_M1=M1, model_M2=M2, bjt=bjt,
                                         Vd_seq=Vd,
                                         VG1=torch.tensor(c["VG1"], dtype=torch.float64),
                                         VG2=torch.tensor(c["VG2"], dtype=torch.float64),
                                         warm_start=True)
                    I_pred = np.abs(out["Id"].detach().cpu().numpy()).astype(np.float64)
                    if not np.all(np.isfinite(I_pred)):
                        I_pred = np.where(np.isfinite(I_pred), I_pred, 0.0)
                except Exception:
                    I_pred = np.zeros_like(Vd_np)
                res = pic3.log_residuals(Id_np, I_pred, Vd_np, vmin=vmin)
                med_dec = float(np.median(res)) if res.size else float("nan")
                mt = (Vd_np > 0.05) & (Vd_np <= 0.5) & (np.abs(Id_np) > pic3.DEC_FLOOR_MEAS) & (I_pred > 0)
                if mt.sum() >= 3:
                    lm = np.log10(np.clip(np.abs(Id_np[mt]), pic3.DEC_FLOOR_MEAS, None))
                    lp = np.log10(np.clip(I_pred[mt], pic3.DEC_FLOOR_PRED, None))
                    triode_rmse = float(np.sqrt(np.mean((lm - lp)**2)))
                else:
                    triode_rmse = float("nan")
                rows.append({
                    "VG1": c["VG1"], "VG2": c["VG2"], "branch": branch,
                    "file": c["f"], "n": int(res.size),
                    "med_dec": med_dec, "triode_rmse_dec": triode_rmse,
                    "Imeas_peak": float(np.max(np.abs(Id_np))),
                    "Ipred_peak": float(np.max(I_pred)) if I_pred.size else float("nan"),
                })
    finally:
        for k, v in bjt_saved.items():
            setattr(bjt, k, v)
    return rows


def agg(rows):
    rms = np.array([r["triode_rmse_dec"] for r in rows if np.isfinite(r["triode_rmse_dec"])])
    med = np.array([r["med_dec"] for r in rows if np.isfinite(r["med_dec"])])
    return {
        "n": len(rows),
        "triode_rmse_median": float(np.median(rms)) if rms.size else float("nan"),
        "triode_rmse_mean":   float(np.mean(rms))   if rms.size else float("nan"),
        "med_dec_median":     float(np.median(med)) if med.size else float("nan"),
    }


def main():
    t0 = time.time()
    print(f"[triode] start {time.strftime('%H:%M:%S')}", flush=True)

    cfg, M1, M2, bjt = pic3.build_pyport_base()
    sebas_rows = pic3.load_sebas_params()
    all_curves = pic3.load_curves()
    vg06 = [c for c in all_curves if abs(c["VG1"] - 0.6) < 1e-6]
    vg04 = [c for c in all_curves if abs(c["VG1"] - 0.4) < 1e-6]
    vg02 = [c for c in all_curves if abs(c["VG1"] - 0.2) < 1e-6]
    print(f"[triode] curves loaded: VG1=0.6 n={len(vg06)}, VG1=0.4 n={len(vg04)}, VG1=0.2 n={len(vg02)}", flush=True)

    # === Step 1: baseline pass on ALL VG1=0.6 to find top-5 worst ===
    print(f"[triode] baseline pass over VG1=0.6 ({len(vg06)} curves)...", flush=True)
    base_rows_06 = run_one_pass(cfg, M1, M2, bjt, vg06, sebas_rows)
    # rank by max(fwd,bwd) triode rmse per (VG1,VG2,file)
    from collections import defaultdict
    by_file = defaultdict(list)
    for r in base_rows_06:
        by_file[r["file"]].append(r)
    file_rmse = []
    for f, rs in by_file.items():
        rmses = [r["triode_rmse_dec"] for r in rs if np.isfinite(r["triode_rmse_dec"])]
        if rmses:
            file_rmse.append((f, float(np.mean(rmses)), rs[0]["VG2"]))
    file_rmse.sort(key=lambda t: -t[1])  # worst first
    top5 = file_rmse[:5]
    print(f"[triode] top-5 worst VG1=0.6 traces (mean triode RMSE):", flush=True)
    for f, r, vg2 in top5:
        print(f"   VG2={vg2:+.2f} RMSE={r:.3f}  {f}", flush=True)
    top5_files = {t[0] for t in top5}
    top5_curves = [c for c in vg06 if c["f"] in top5_files]

    base_agg_top5 = agg([r for r in base_rows_06 if r["file"] in top5_files])
    print(f"[triode] baseline top-5 VG1=0.6 triode RMSE median = {base_agg_top5['triode_rmse_median']:.3f}", flush=True)

    # baseline on VG1=0.2 and VG1=0.4 for regression check (only at end of candidates, save time)

    # === Step 2: parameter sweeps ===
    # Baseline values from sd_M1.scaled (post-temp/binning). We capture them.
    sd_M1 = cfg.size_dep_M1(M1)
    baseline_scaled = {k: float(sd_M1.scaled.get(k, float("nan")))
                       for k in ("u0","vth0","k1","k2","vsat","rdsw","rsw","rdw",
                                 "eta0","etab","pclm","a0","ags","delta","voff",
                                 "nfactor")}
    print(f"[triode] baseline sd_M1.scaled values:", flush=True)
    for k, v in baseline_scaled.items():
        print(f"   {k:>8s} = {v:.6g}", flush=True)
    baseline_bjt = {k: getattr(bjt, k) for k in ("Is","Bf","Nf","Va")}
    print(f"[triode] baseline bjt: {baseline_bjt}", flush=True)

    # Multiplicative sweep params (positive-only)
    MULT_SWEEPS = {
        # mobility & velocity
        "u0":    [0.5, 0.8, 1.0, 1.2, 2.0],
        "vsat":  [0.5, 0.8, 1.0, 1.2, 2.0],
        # threshold-curvature (k1 already CSV-overridden but we test mult anyway)
        "k1":    [0.5, 0.8, 1.0, 1.2, 2.0],
        # k2 baseline can be small/negative; mult is fine
        # DIBL
        "eta0":  [0.5, 0.8, 1.0, 1.2, 2.0],
        "etab":  [0.5, 0.8, 1.0, 1.2, 2.0],
        # series R
        "rdsw":  [0.0, 0.5, 0.8, 1.0, 1.2, 2.0],   # try =0 too
        # channel-length modulation
        "pclm":  [0.5, 0.8, 1.0, 1.2, 2.0],
        # bulk-charge
        "a0":    [0.5, 0.8, 1.0, 1.2, 2.0],
        "ags":   [0.0, 0.5, 0.8, 1.0, 1.2, 2.0],
        # smoothing
        "delta": [0.5, 0.8, 1.0, 1.2, 2.0],
        # nfactor (subthreshold but listed for completeness)
        "nfactor": [0.5, 0.8, 1.0, 1.2, 2.0],
    }
    # Additive sweep params (V)
    ADD_SWEEPS = {
        "vth0": [-0.10, -0.05, -0.02, 0.0, +0.02, +0.05, +0.10],  # subtractive helps current
        "voff": [-0.10, -0.05, -0.02, 0.0, +0.02, +0.05, +0.10],
    }
    # BJT mult sweeps
    BJT_SWEEPS = {
        "Is": [0.5, 0.8, 1.0, 1.2, 2.0, 10.0, 100.0],
        "Bf": [0.1, 0.5, 1.0, 2.0, 10.0],
        "Nf": [0.9, 1.0, 1.1],
        "Va": [0.1, 0.5, 1.0, 2.0, 10.0],
    }

    sensitivity = {
        "baseline_scaled": baseline_scaled,
        "baseline_bjt": baseline_bjt,
        "baseline_top5_VG1_06": base_agg_top5,
        "top5_files": [{"file": f, "VG2": v, "rmse_base": r} for f,r,v in top5],
        "sweeps": {},
    }

    def sweep_param(name, kind, values):
        results = []
        baseval = baseline_scaled.get(name) if kind == "sd_mult" or kind == "sd_add" \
                    else baseline_bjt.get(name)
        print(f"\n[triode] === {kind} sweep {name} (base={baseval:.6g}) ===", flush=True)
        for v in values:
            if kind == "sd_mult":
                newval = baseval * v; ov = {name: newval}; bjto = None
            elif kind == "sd_add":
                newval = baseval + v; ov = {name: newval}; bjto = None
            elif kind == "bjt_mult":
                newval = baseval * v; ov = None; bjto = {name: newval}
            else:
                continue
            try:
                rows = run_one_pass(cfg, M1, M2, bjt, top5_curves, sebas_rows,
                                    sd_overrides_M1=ov, bjt_overrides=bjto)
                a = agg(rows)
            except Exception as e:
                a = {"error": str(e)}
            tag = (f"x{v}" if "mult" in kind else f"{v:+.3f}")
            print(f"   {name} {tag:>8s} -> newval={newval:.4g}  "
                  f"triode_rmse_med={a.get('triode_rmse_median', float('nan')):.3f}  "
                  f"med_dec={a.get('med_dec_median', float('nan')):.3f}", flush=True)
            results.append({"factor": v, "newval": newval, "kind": kind, **a})
        return results

    for name, vals in MULT_SWEEPS.items():
        sensitivity["sweeps"][name] = sweep_param(name, "sd_mult", vals)
    for name, vals in ADD_SWEEPS.items():
        sensitivity["sweeps"][name] = sweep_param(name, "sd_add", vals)
    for name, vals in BJT_SWEEPS.items():
        sensitivity["sweeps"]["BJT_"+name] = sweep_param(name, "bjt_mult", vals)

    # === Step 3: identify candidate single-knob fixes ===
    print(f"\n[triode] === candidate analysis ===", flush=True)
    candidates = []
    for name, runs in sensitivity["sweeps"].items():
        for r in runs:
            if "triode_rmse_median" in r and np.isfinite(r["triode_rmse_median"]):
                if r["triode_rmse_median"] <= 0.5:
                    candidates.append((name, r))
    print(f"[triode] {len(candidates)} single-knob settings with triode RMSE <= 0.5:", flush=True)
    for n, r in candidates[:20]:
        print(f"   {n} factor={r['factor']} -> {r['triode_rmse_median']:.3f}", flush=True)

    # === Step 4: cross-check best candidates against VG1=0.2 and 0.4 ===
    if candidates:
        # take best per param
        best_per_param = {}
        for n, r in candidates:
            cur = best_per_param.get(n)
            if cur is None or r["triode_rmse_median"] < cur["triode_rmse_median"]:
                best_per_param[n] = r
        print(f"[triode] cross-checking {len(best_per_param)} param candidates on VG1=0.2 and 0.4 ...", flush=True)
        # also baseline VG1=0.2 / 0.4 once
        base_02 = agg(run_one_pass(cfg, M1, M2, bjt, vg02, sebas_rows))
        base_04 = agg(run_one_pass(cfg, M1, M2, bjt, vg04, sebas_rows))
        sensitivity["baseline_VG1_02"] = base_02
        sensitivity["baseline_VG1_04"] = base_04
        print(f"[triode] baseline VG1=0.2 triode_rmse_med={base_02['triode_rmse_median']:.3f}", flush=True)
        print(f"[triode] baseline VG1=0.4 triode_rmse_med={base_04['triode_rmse_median']:.3f}", flush=True)
        cross = {}
        for name, r in best_per_param.items():
            kind = r["kind"]
            v = r["factor"]
            # rebuild overrides
            if name.startswith("BJT_"):
                param = name[4:]
                bjto = {param: baseline_bjt[param] * v}; ov = None
            elif kind == "sd_mult":
                ov = {name: baseline_scaled[name] * v}; bjto = None
            elif kind == "sd_add":
                ov = {name: baseline_scaled[name] + v}; bjto = None
            else:
                continue
            try:
                rows_02 = run_one_pass(cfg, M1, M2, bjt, vg02, sebas_rows,
                                       sd_overrides_M1=ov, bjt_overrides=bjto)
                rows_04 = run_one_pass(cfg, M1, M2, bjt, vg04, sebas_rows,
                                       sd_overrides_M1=ov, bjt_overrides=bjto)
                a02 = agg(rows_02); a04 = agg(rows_04)
            except Exception as e:
                a02 = {"error": str(e)}; a04 = {"error": str(e)}
            d02 = a02.get("triode_rmse_median", float("nan")) - base_02["triode_rmse_median"]
            d04 = a04.get("triode_rmse_median", float("nan")) - base_04["triode_rmse_median"]
            cross[name] = {"VG1_06_rmse": r["triode_rmse_median"],
                           "VG1_04_rmse": a04.get("triode_rmse_median", float("nan")),
                           "VG1_04_delta": d04,
                           "VG1_02_rmse": a02.get("triode_rmse_median", float("nan")),
                           "VG1_02_delta": d02,
                           "factor": v, "kind": kind}
            print(f"   {name:>12s} fac={v}  "
                  f"VG06={r['triode_rmse_median']:.3f}  "
                  f"VG04={a04.get('triode_rmse_median', float('nan')):.3f} (Δ{d04:+.3f})  "
                  f"VG02={a02.get('triode_rmse_median', float('nan')):.3f} (Δ{d02:+.3f})",
                  flush=True)
        sensitivity["cross_check"] = cross
    else:
        sensitivity["cross_check"] = {}
        print(f"[triode] NO single-knob candidate hits triode RMSE <= 0.5 — combination needed.", flush=True)

    # === Step 5: dump JSON ===
    out_json = OUT / "param_sensitivity.json"
    with open(out_json, "w") as f:
        json.dump(sensitivity, f, indent=2, default=str)
    print(f"\n[triode] wrote {out_json}", flush=True)
    print(f"[triode] done in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
