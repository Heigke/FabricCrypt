"""z386 — Test C: combined clamp-off + extreme etab.

Hypothesis 3 ∧ Hypothesis 4: maximally regenerative config.
  - use_well_diode=False, body_pdiode_to='off', etab_override=20.0
  - Targets: (VG1,VG2) ∈ {(0.2,0.10), (0.4,0.20), (0.6,0.20)}

Gates:
  AMBITIOUS  : combined gives cell-wide RMSE < 0.5 dec
  KILL-SHOT  : even this combo gives VG1=0.6 model_jump < 0.3 dec
               → these aren't the missing physics
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _z384_shared import (ROOT, TARGETS, build_base, load_sebas_params, run_one)

OUT = ROOT / "results/z386_combined"; OUT.mkdir(parents=True, exist_ok=True)
LOG = OUT / "run.log"


def _log(msg: str):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f: f.write(line + "\n")


def main():
    if LOG.exists(): LOG.unlink()
    rows = load_sebas_params()
    t_start = time.time()

    cfg, M1, M2, bjt = build_base()
    cfg.use_well_diode = False
    cfg.body_pdiode_to = "off"
    _log("=== combined: use_well_diode=False, body_pdiode_to='off', etab=20.0 ===")

    per_t = []
    for (vg1, vg2) in TARGETS:
        r = run_one(cfg, M1, M2, bjt, rows, vg1, vg2, etab_override=20.0, log=_log)
        _log(f"  VG1={vg1} VG2={vg2}: rmse={r.get('rmse_dec',float('nan')):.3f} dec  "
             f"jump(meas/model)={r.get('meas_jump_dec',0) or 0:.2f}/"
             f"{r.get('model_jump_dec',float('nan')):.2f}  "
             f"nan={r.get('has_nan')}  {r.get('elapsed_s',0):.1f}s")
        per_t.append(r)

    rmse_list = [r["rmse_dec"] for r in per_t if r.get("rmse_dec") == r.get("rmse_dec")]
    cellwide = sum(rmse_list)/len(rmse_list) if rmse_list else float("nan")
    mj_vg06 = next((r["model_jump_dec"] for r in per_t if r["VG1"] == 0.6), float("nan"))
    ambitious = cellwide == cellwide and cellwide < 0.5
    kill_shot = (mj_vg06 is not None and mj_vg06 == mj_vg06 and mj_vg06 < 0.3)
    discovery = mj_vg06 is not None and mj_vg06 == mj_vg06 and mj_vg06 > 0.5

    elapsed = time.time() - t_start
    summary = {
        "results": per_t,
        "gates": {
            "infra_ok": elapsed < 45*60,
            "elapsed_s": elapsed,
            "cellwide_rmse_dec": cellwide,
            "model_jump_vg06": mj_vg06,
            "discovery_fold_gt_0p5": bool(discovery),
            "ambitious_cellwide_lt_0p5": bool(ambitious),
            "kill_shot_combined_lt_0p3": bool(kill_shot),
        },
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    _log(f"DONE in {elapsed:.1f}s")
    _log(f"  cell-wide RMSE = {cellwide:.3f} dec")
    _log(f"  VG1=0.6 model_jump = {mj_vg06:.3f} dec")
    _log(f"  DISCOVERY (>0.5): {discovery}  |  AMBITIOUS (<0.5 dec): {ambitious}")
    _log(f"  KILL-SHOT (<0.3): {kill_shot}")


if __name__ == "__main__":
    main()
