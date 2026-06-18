"""z275 — PMP-6b: rerun production forward DC eval with N-well pdiode ON.

Compares log10-residual of z91g's two-model validation between:
  - baseline:   NSRAM_PDI_TO=off     (cfg.body_pdiode_to default)
  - treatment:  NSRAM_PDI_TO=vnwell  (Sebas's slide-21 diode active)

Production BJT params: Bf=9000, Va=0.55, Is=1e-9. vnwell stays at 2.0V
default; diode card stays at cfg defaults (Js/n/Vj/M as in nsram_cell_2T.py).
NO parameter retuning between runs — direction of change is informative.

Pre-registered gate (research_plan/01_LOG.md 2026-05-11):
  PASS  if |median_log_rmse_on − median_log_rmse_off| ≥ 0.10 dec
  NULL  if |Δ| < 0.10 dec
"""
from __future__ import annotations
import json, math, os, subprocess, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/z275_pmp6b_diode_on"
OUT.mkdir(parents=True, exist_ok=True)

Z91G = ROOT / "scripts/z91g_two_model_validation.py"
PY = ROOT / "venv/bin/python"

# Production BJT params (PMP-6b spec)
BASE_ENV = {
    "NSRAM_BJT_BF": "9000",
    "NSRAM_BJT_VA": "0.55",
    "NSRAM_BJT_IS": "1e-9",
    # Re-use NSRAM_A0_MULT default in z91g (10.0). Don't touch any other knob.
    "PYTHONUNBUFFERED": "1",
}


def run_eval(label: str, pdi_to: str) -> dict:
    """Run z91g with NSRAM_PDI_TO=<pdi_to>. Returns parsed predictions list."""
    suffix = f"_pmp6b_{label}"
    env = os.environ.copy()
    env.update(BASE_ENV)
    env["NSRAM_PDI_TO"] = pdi_to
    env["NSRAM_OUT_SUFFIX"] = suffix
    z91g_out = ROOT / f"results/z91g_two_model_validation{suffix}"
    z91g_out.mkdir(parents=True, exist_ok=True)

    log_path = OUT / f"z91g_{label}.log"
    print(f"[z275] running z91g with NSRAM_PDI_TO={pdi_to} → {z91g_out.name}",
          flush=True)
    t0 = time.time()
    with open(log_path, "w") as fh:
        proc = subprocess.run(
            [str(PY), str(Z91G)],
            cwd=str(ROOT), env=env, stdout=fh, stderr=subprocess.STDOUT,
        )
    dt = time.time() - t0
    print(f"[z275] {label}: returncode={proc.returncode}, {dt:.0f}s, "
          f"log={log_path.name}", flush=True)
    if proc.returncode != 0:
        tail = log_path.read_text().splitlines()[-30:]
        raise RuntimeError(f"z91g[{label}] failed:\n" + "\n".join(tail))

    preds = json.loads((z91g_out / "predictions.json").read_text())
    summ = json.loads((z91g_out / "summary.json").read_text())
    return {"preds": preds, "summary": summ, "elapsed_s": dt,
            "out_dir": str(z91g_out)}


def per_branch_stats(preds: list, vg1_target: float) -> dict:
    rmses = [r["log_rmse"] for r in preds
             if not r.get("skipped")
             and math.isfinite(r["log_rmse"])
             and abs(r["VG1"] - vg1_target) < 1e-3]
    if not rmses:
        return {"median": float("nan"), "p90": float("nan"),
                "n": 0, "max": float("nan")}
    return {
        "median": float(np.median(rmses)),
        "p90": float(np.percentile(rmses, 90)),
        "max": float(np.max(rmses)),
        "n": len(rmses),
    }


def all_stats(preds: list) -> dict:
    rmses = [r["log_rmse"] for r in preds
             if not r.get("skipped") and math.isfinite(r["log_rmse"])]
    n_skip = sum(1 for r in preds if r.get("skipped"))
    n_conv_total = sum(r.get("n_converged", 0) for r in preds
                       if not r.get("skipped"))
    n_total = sum(r.get("n_total", 0) for r in preds
                  if not r.get("skipped"))
    return {
        "n_curves_evaluated": len(rmses),
        "n_skipped": n_skip,
        "median_log_rmse": float(np.median(rmses)) if rmses else float("inf"),
        "p90_log_rmse": float(np.percentile(rmses, 90)) if rmses else float("inf"),
        "mean_log_rmse": float(np.mean(rmses)) if rmses else float("inf"),
        "max_log_rmse": float(np.max(rmses)) if rmses else float("inf"),
        "newton_conv_rate": (n_conv_total / n_total) if n_total else 0.0,
        "n_conv": n_conv_total,
        "n_pts_total": n_total,
    }


def per_bias_delta(off_preds: list, on_preds: list) -> list:
    """Per (VG1, VG2) bias cell — list sorted by |Δ log_rmse| descending."""
    off_idx = {(round(r["VG1"], 3), round(r["VG2"], 3)): r for r in off_preds}
    on_idx = {(round(r["VG1"], 3), round(r["VG2"], 3)): r for r in on_preds}
    keys = sorted(set(off_idx) & set(on_idx))
    rows = []
    for k in keys:
        ro = off_idx[k]; rn = on_idx[k]
        if ro.get("skipped") or rn.get("skipped"):
            continue
        if not (math.isfinite(ro.get("log_rmse", float("inf"))) and
                math.isfinite(rn.get("log_rmse", float("inf")))):
            continue
        d = rn["log_rmse"] - ro["log_rmse"]
        rows.append({
            "VG1": k[0], "VG2": k[1],
            "log_rmse_off": ro["log_rmse"],
            "log_rmse_on": rn["log_rmse"],
            "delta": d,
            "abs_delta": abs(d),
            "conv_off": f"{ro['n_converged']}/{ro['n_total']}",
            "conv_on":  f"{rn['n_converged']}/{rn['n_total']}",
        })
    rows.sort(key=lambda r: r["abs_delta"], reverse=True)
    return rows


def main():
    t0 = time.time()
    print(f"[z275] PMP-6b diode-ON re-eval starting {time.strftime('%H:%M:%S')}",
          flush=True)
    print(f"[z275] base env: {BASE_ENV}", flush=True)

    off = run_eval("off", "off")
    on = run_eval("on", "vnwell")

    off_stats = all_stats(off["preds"])
    on_stats = all_stats(on["preds"])

    off_branch = {f"VG1={vg1}": per_branch_stats(off["preds"], vg1)
                  for vg1 in (0.2, 0.4, 0.6)}
    on_branch = {f"VG1={vg1}": per_branch_stats(on["preds"], vg1)
                 for vg1 in (0.2, 0.4, 0.6)}

    delta = on_stats["median_log_rmse"] - off_stats["median_log_rmse"]
    gate_threshold = 0.10
    if not (math.isfinite(delta)):
        verdict = "ERROR_NONFINITE"
    elif abs(delta) >= gate_threshold:
        verdict = "PASS_INFORMATIVE_SIGNIFICANT"
    else:
        verdict = "INFORMATIVE_NULL"

    per_bias = per_bias_delta(off["preds"], on["preds"])

    summary = {
        "experiment": "z275_pmp6b_diode_on",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "base_env": BASE_ENV,
        "gate": {
            "threshold_dec": gate_threshold,
            "delta_median_log_rmse": delta,
            "verdict": verdict,
        },
        "diode_off": {
            "overall": off_stats,
            "per_branch": off_branch,
            "elapsed_s": off["elapsed_s"],
        },
        "diode_on": {
            "overall": on_stats,
            "per_branch": on_branch,
            "elapsed_s": on["elapsed_s"],
        },
        "per_branch_delta_median": {
            k: (on_branch[k]["median"] - off_branch[k]["median"])
            for k in off_branch
        },
        "top10_bias_cell_deltas": per_bias[:10],
        "total_elapsed_s": time.time() - t0,
        "z91g_out_dirs": {"off": off["out_dir"], "on": on["out_dir"]},
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    (OUT / "per_bias_all.json").write_text(json.dumps(per_bias, indent=2))

    # Console report
    print("\n" + "=" * 72, flush=True)
    print("z275 PMP-6b — N-well pdiode ON vs OFF", flush=True)
    print("=" * 72, flush=True)
    print(f"DIODE OFF: median={off_stats['median_log_rmse']:.4f} dec  "
          f"p90={off_stats['p90_log_rmse']:.4f}  "
          f"conv={off_stats['newton_conv_rate']*100:.1f}%  "
          f"({off_stats['n_curves_evaluated']} curves)", flush=True)
    print(f"DIODE ON : median={on_stats['median_log_rmse']:.4f} dec  "
          f"p90={on_stats['p90_log_rmse']:.4f}  "
          f"conv={on_stats['newton_conv_rate']*100:.1f}%  "
          f"({on_stats['n_curves_evaluated']} curves)", flush=True)
    print(f"\nΔ median log-RMSE = {delta:+.4f} dec  "
          f"(threshold = ±{gate_threshold:.2f})", flush=True)
    print(f"VERDICT: {verdict}", flush=True)
    print(f"\nPer-branch (median log-RMSE):", flush=True)
    for k in off_branch:
        do = off_branch[k]["median"]; dn = on_branch[k]["median"]
        print(f"  {k}: off={do:.4f}  on={dn:.4f}  Δ={dn-do:+.4f}  "
              f"(n_off={off_branch[k]['n']}, n_on={on_branch[k]['n']})",
              flush=True)
    print(f"\nTop-5 bias cells by |Δ log-RMSE|:", flush=True)
    for r in per_bias[:5]:
        print(f"  VG1={r['VG1']:+.2f} VG2={r['VG2']:+.2f}: "
              f"off={r['log_rmse_off']:.3f} on={r['log_rmse_on']:.3f} "
              f"Δ={r['delta']:+.3f}  "
              f"conv off={r['conv_off']} on={r['conv_on']}", flush=True)
    print(f"\nSaved {OUT}/summary.json  ({time.time()-t0:.0f}s total)", flush=True)


if __name__ == "__main__":
    main()
