"""C1b — Torch pyport NPN diagnostic.

Re-runs the z430 V_SINT_PIN pipeline (which produced the 1.62 dec fwd baseline
in P1a_honest_baseline) with the torch pyport's parasitic NPN turned OFF via
cfg.use_bjt=False, and compares against the NPN-ON baseline.

Goal: determine whether the GummelPoonNPN parallel branch wired into the torch
pyport drain residuals (nsram/nsram/bsim4_port/nsram_cell_2T.py:976-1018)
actually contributes to the 1.62 dec fwd / 2.82 dec bwd achievement.

Method:
  - Same curves loader as P1a (z91f.load_curves) — keeps result comparable to
    P1a_honest_baseline.summary.json
  - Both fwd and bwd direction (warm-start, same as P1a)
  - Per-region log-RMSE: subth (Vd<0.3), triode (0.3<=Vd<0.8),
    sat (0.8<=Vd<1.5), snap (Vd>=1.5)
  - NPN-ON: cfg.use_bjt=True (default, matches P1a)
  - NPN-OFF: cfg.use_bjt=False
  - Bootstrap 95% CI (n_boot=1000) on each scalar
  - NaN count tracked per condition

Output: results/C1b_torch_npn_off_control/{summary.json, verdict.md}
"""
from __future__ import annotations
import importlib.util as _ilu
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

torch.set_default_dtype(torch.float64)

ROOT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
OUT = ROOT / "results/C1b_torch_npn_off_control"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "nsram"))
sys.path.insert(0, str(ROOT / "scripts"))

LOG_FH = open(OUT / "run.log", "w")
def log(msg: str):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG_FH.write(line + "\n"); LOG_FH.flush()


def _load(name, path):
    spec = _ilu.spec_from_file_location(name, path)
    m = _ilu.module_from_spec(spec); spec.loader.exec_module(m); return m


z427 = _load("z427", ROOT / "scripts/z427_vsint_fix.py")
z429 = _load("z429", ROOT / "scripts/z429_multisolver_debug.py")

LOG_EPS = 1e-15
RNG = np.random.default_rng(20260519)

REGION_BOUNDS = [
    ("subth",  0.0, 0.3),
    ("triode", 0.3, 0.8),
    ("sat",    0.8, 1.5),
    ("snap",   1.5, 10.0),
]


def bootstrap_ci(values, alpha=0.05, n_boot=1000):
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=np.float64)
    if v.size == 0:
        return float("nan"), float("nan"), float("nan"), 0
    med = float(np.median(v))
    idx = RNG.integers(0, len(v), size=(n_boot, len(v)))
    boots = np.median(v[idx], axis=1)
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return med, lo, hi, int(v.size)


def run_pipeline(model_M1, model_M2, curves, sebas_rows, direction, use_bjt):
    """Run z430 V_SINT_PIN with NPN flag toggled.

    Returns list of per-bias dicts: VG1, VG2, log_rmse_full, log_rmse_subth,
    log_rmse_triode, log_rmse_sat, log_rmse_snap, n_conv, n_pts, n_nan.
    """
    extra = {"use_bjt": bool(use_bjt)}
    cfg, sd_M1, sd_M2 = z427.make_cfg(model_M1, model_M2, extra)
    per_bias = []
    total_nan = 0
    t0 = time.time()
    for c in curves:
        sebas_row = z427.find_params(sebas_rows, c["VG1"], c["VG2"])
        if sebas_row is None or math.isnan(sebas_row.get("K1", float("nan"))):
            continue
        P_M1, P_M2 = z427.make_overrides(sebas_row)
        bjt = z427.make_bjt(sebas_row)
        Vd_arr = c["Vd"].numpy()
        Id_meas = c["Id"].numpy()
        n = len(Vd_arr)
        order = list(range(n)) if direction == "forward" else list(range(n - 1, -1, -1))
        Id_pred = [float("nan")] * n
        conv_list = [False] * n
        try:
            with torch.no_grad(), z427.patch_sd_scaled(sd_M1, P_M1), z427.patch_sd_scaled(sd_M2, P_M2):
                Vb_warm = 0.0
                for idx in order:
                    Vd_f = float(Vd_arr[idx])
                    try:
                        r = z429.run_vsint_pinned(
                            cfg, model_M1, model_M2, bjt,
                            Vd_f, float(c["VG1"]), float(c["VG2"]),
                            Vsint_pin=0.0, Vb_init=Vb_warm)
                        Id_pred[idx] = abs(r["Id"])
                        conv_list[idx] = bool(r["converged"])
                        Vb_warm = r["Vb"] if r["converged"] else 0.0
                    except Exception:
                        Id_pred[idx] = float("nan")
                        conv_list[idx] = False
        except Exception as e:
            log(f"  fail VG1={c['VG1']} VG2={c['VG2']}: {e}")
            continue
        Id_pred_arr = np.asarray(Id_pred, dtype=np.float64)
        conv_arr = np.asarray(conv_list, dtype=bool)
        finite = np.isfinite(Id_pred_arr)
        n_nan = int(np.sum(~finite))
        total_nan += n_nan
        valid = conv_arr & finite & (Id_pred_arr > 0)
        if not valid.any():
            continue
        log_p = np.log10(Id_pred_arr[valid] + LOG_EPS)
        log_m = np.log10(Id_meas[valid] + LOG_EPS)
        sq = (log_p - log_m) ** 2
        rmse_full = float(math.sqrt(np.mean(sq)))
        per_region = {}
        Vd_v = Vd_arr[valid]
        for name, lo, hi in REGION_BOUNDS:
            m = (Vd_v >= lo) & (Vd_v < hi)
            per_region[name] = float(math.sqrt(np.mean(sq[m]))) if m.any() else float("nan")
        per_bias.append({
            "VG1": float(c["VG1"]), "VG2": float(c["VG2"]),
            "log_rmse_full": rmse_full,
            "log_rmse_subth": per_region["subth"],
            "log_rmse_triode": per_region["triode"],
            "log_rmse_sat": per_region["sat"],
            "log_rmse_snap": per_region["snap"],
            "n_conv": int(conv_arr.sum()),
            "n_pts": n,
            "n_nan": n_nan,
        })
    wall = time.time() - t0
    return {
        "direction": direction,
        "use_bjt": bool(use_bjt),
        "n_biases_evaluated": len(per_bias),
        "total_nan": total_nan,
        "wall_sec": round(wall, 1),
        "per_bias": per_bias,
    }


def summarize(run, label):
    """Compute aggregate stats with bootstrap CI."""
    pb = run["per_bias"]
    out = {"label": label, "direction": run["direction"], "use_bjt": run["use_bjt"],
           "n_biases": run["n_biases_evaluated"], "total_nan": run["total_nan"],
           "wall_sec": run["wall_sec"]}
    # Cell-wide (quadratic mean of per-bias log_rmse_full)
    rmses = np.array([r["log_rmse_full"] for r in pb if np.isfinite(r["log_rmse_full"])])
    cell = float(math.sqrt(np.mean(rmses ** 2))) if rmses.size else float("nan")
    out["cell_rmse_dec"] = cell
    # Bootstrap on median log_rmse_full
    med, lo, hi, n = bootstrap_ci([r["log_rmse_full"] for r in pb])
    out["median_log_rmse_full"] = {"median": med, "ci95_lo": lo, "ci95_hi": hi, "n": n}
    # Per-region medians
    for region in ("subth", "triode", "sat", "snap"):
        key = f"log_rmse_{region}"
        med, lo, hi, n = bootstrap_ci([r[key] for r in pb])
        out[f"median_{region}_rmse"] = {"median": med, "ci95_lo": lo, "ci95_hi": hi, "n": n}
    # Per-VG1 medians
    for vg1 in (0.2, 0.4, 0.6):
        vals = [r["log_rmse_full"] for r in pb if abs(r["VG1"] - vg1) < 1e-6]
        med, lo, hi, n = bootstrap_ci(vals)
        out[f"median_VG1={vg1}"] = {"median": med, "ci95_lo": lo, "ci95_hi": hi, "n": n}
    return out


def main():
    log("C1b starting — torch pyport NPN-off control on z430 V_SINT_PIN")
    log("Loading models and data...")
    model_M1, model_M2 = z427.build_models()
    z91f = _load("z91f", ROOT / "scripts/z91f_validate_with_sebas_params.py")
    curves = z91f.load_curves()
    sebas_rows = z91f.load_sebas_params()
    valid_curves = [c for c in curves
                    if z427.find_params(sebas_rows, c["VG1"], c["VG2"]) is not None
                    and not math.isnan(z427.find_params(sebas_rows, c["VG1"], c["VG2"]).get("K1", float("nan")))]
    log(f"Loaded {len(curves)} curves, {len(valid_curves)} with valid K1, {len(sebas_rows)} sebas rows")

    summary = {
        "script": "C1b_torch_npn_off_control",
        "date_utc": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "n_curves_total": len(curves),
        "n_curves_with_valid_K1": len(valid_curves),
        "pipeline": "z430_V_SINT_PIN (torch pyport)",
        "regions": [{"name": n, "Vd_lo": lo, "Vd_hi": hi} for n, lo, hi in REGION_BOUNDS],
        "n_boot": 1000,
        "rng_seed": 20260519,
        "conditions": {},
    }

    for label, use_bjt in [("NPN_ON", True), ("NPN_OFF", False)]:
        for direction in ("forward", "backward"):
            log(f"=== {label} {direction} ===")
            run = run_pipeline(model_M1, model_M2, curves, sebas_rows, direction, use_bjt)
            log(f"  {label} {direction}: n_biases={run['n_biases_evaluated']} "
                f"nan={run['total_nan']} wall={run['wall_sec']:.1f}s")
            s = summarize(run, f"{label}_{direction}")
            log(f"  cell_rmse={s['cell_rmse_dec']:.3f}  median_full={s['median_log_rmse_full']['median']:.3f}  "
                f"subth={s['median_subth_rmse']['median']:.3f}  triode={s['median_triode_rmse']['median']:.3f}  "
                f"sat={s['median_sat_rmse']['median']:.3f}  snap={s['median_snap_rmse']['median']:.3f}")
            key = f"{label}_{direction}"
            summary["conditions"][key] = {**s, "per_bias": run["per_bias"]}
            # incremental write
            with open(OUT / "summary.json", "w") as f:
                json.dump(summary, f, indent=2)

    # Verdict: NPN-OFF − NPN-ON cell RMSE delta (positive = NPN was useful)
    on_fwd = summary["conditions"]["NPN_ON_forward"]["cell_rmse_dec"]
    on_bwd = summary["conditions"]["NPN_ON_backward"]["cell_rmse_dec"]
    off_fwd = summary["conditions"]["NPN_OFF_forward"]["cell_rmse_dec"]
    off_bwd = summary["conditions"]["NPN_OFF_backward"]["cell_rmse_dec"]
    delta_fwd = off_fwd - on_fwd
    delta_bwd = off_bwd - on_bwd
    summary["verdict_metrics"] = {
        "cell_NPN_ON_fwd": on_fwd, "cell_NPN_ON_bwd": on_bwd,
        "cell_NPN_OFF_fwd": off_fwd, "cell_NPN_OFF_bwd": off_bwd,
        "delta_fwd_off_minus_on": delta_fwd,
        "delta_bwd_off_minus_on": delta_bwd,
        "killshot_threshold_dec": 0.5,
        "fwd_killshot": bool(delta_fwd < 0.5),   # True ⇒ NPN-OFF not worse by ≥0.5 dec
        "bwd_killshot": bool(delta_bwd < 0.5),
    }
    log(f"Δ_fwd = {delta_fwd:+.3f}, Δ_bwd = {delta_bwd:+.3f}")

    # Write verdict.md
    md = []
    md.append("# C1b — Torch pyport NPN-off control — VERDICT\n")
    md.append(f"Date: {summary['date_utc']}\n")
    md.append(f"Pipeline: {summary['pipeline']}  |  Curves used: {summary['n_curves_with_valid_K1']}/{summary['n_curves_total']}\n")
    md.append("\n## Cell-wide RMSE (decades)\n")
    md.append("| cond | fwd | bwd |")
    md.append("|---|---|---|")
    md.append(f"| NPN_ON  | {on_fwd:.3f} | {on_bwd:.3f} |")
    md.append(f"| NPN_OFF | {off_fwd:.3f} | {off_bwd:.3f} |")
    md.append(f"| Δ (off-on) | {delta_fwd:+.3f} | {delta_bwd:+.3f} |\n")
    md.append("## Median per-region log-RMSE (decades)\n")
    md.append("| cond/dir | subth | triode | sat | snap |")
    md.append("|---|---|---|---|---|")
    for k in ("NPN_ON_forward", "NPN_ON_backward", "NPN_OFF_forward", "NPN_OFF_backward"):
        c = summary["conditions"][k]
        md.append(f"| {k} | {c['median_subth_rmse']['median']:.3f} | "
                  f"{c['median_triode_rmse']['median']:.3f} | "
                  f"{c['median_sat_rmse']['median']:.3f} | "
                  f"{c['median_snap_rmse']['median']:.3f} |")
    md.append("\n## Median full log-RMSE with 95% CI\n")
    md.append("| cond/dir | median | CI95_lo | CI95_hi | n |")
    md.append("|---|---|---|---|---|")
    for k in ("NPN_ON_forward", "NPN_ON_backward", "NPN_OFF_forward", "NPN_OFF_backward"):
        c = summary["conditions"][k]["median_log_rmse_full"]
        md.append(f"| {k} | {c['median']:.3f} | {c['ci95_lo']:.3f} | {c['ci95_hi']:.3f} | {c['n']} |")
    md.append("\n## NaN counts\n")
    for k in ("NPN_ON_forward", "NPN_ON_backward", "NPN_OFF_forward", "NPN_OFF_backward"):
        md.append(f"- {k}: {summary['conditions'][k]['total_nan']} NaN points")
    md.append("\n## Verdict\n")
    p1a_fwd_baseline = 1.6187
    fwd_reproduces = abs(on_fwd - p1a_fwd_baseline) < 0.3
    md.append(f"- P1a baseline fwd = {p1a_fwd_baseline:.3f} dec; this run NPN_ON_fwd = {on_fwd:.3f} dec → "
              f"{'REPRODUCES' if fwd_reproduces else 'DOES NOT REPRODUCE'} baseline\n")
    if delta_fwd >= 0.5 or delta_bwd >= 0.5:
        md.append(f"- **NPN IS A REAL CONTRIBUTOR**: turning off NPN worsens fit by Δfwd={delta_fwd:+.3f}, Δbwd={delta_bwd:+.3f} dec\n")
        md.append("  → The GummelPoonNPN parallel branch was doing useful work in the torch pipeline.\n")
        md.append("  → The remaining gap to 1.0 dec must come from other sources (snapback subcircuit, IIMOD,\n")
        md.append("    Mario IPOS, or fundamental model-structure issues), not absence of bipolar coupling.\n")
    else:
        md.append(f"- **NPN IS DEAD WEIGHT (KILLSHOT)**: Δfwd={delta_fwd:+.3f}, Δbwd={delta_bwd:+.3f} dec — both below 0.5 dec gate\n")
        md.append("  → The GummelPoonNPN parallel branch in the torch pyport contributes nothing to the\n")
        md.append("    1.62 dec baseline. The achievement came from elsewhere: candidates include the\n")
        md.append("    snapback subcircuit, z474b IFT fix, Mario IPOS PWL, V_SINT pinning topology,\n")
        md.append("    or the torch loss landscape itself.\n")
        md.append("  → Consistent with the numpy C1 KILLSHOT result (results/Pillar_I_C1_npn_parallel/verdict.md).\n")

    with open(OUT / "verdict.md", "w") as f:
        f.write("\n".join(md))
    log("DONE — wrote summary.json and verdict.md")


if __name__ == "__main__":
    main()
