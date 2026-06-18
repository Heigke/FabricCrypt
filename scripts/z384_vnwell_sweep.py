"""z384 — Test A: vnwell sweep + well-clamp removal.

Hypothesis 3: relaxing the well/body-pdiode clamp unlocks the VG1=0.6 snapback fold.

Sweep:
  - vnwell ∈ {2.0, 5.0, 10.0, 20.0, 50.0} V (with body_pdiode_to='vnwell' clamp ON)
  - "clamp_off" condition: body_pdiode_to='off' AND use_well_diode=False

Targets: (VG1,VG2) ∈ {(0.2,0.10), (0.4,0.20), (0.6,0.20)}

Gates:
  DISCOVERY  : any condition gives VG1=0.6 model_jump > 0.5 dec
  KILL-SHOT  : even clamp_off gives < 0.3 dec at VG1=0.6
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _z384_shared import (ROOT, TARGETS, build_base, load_sebas_params, run_one)

OUT = ROOT / "results/z384_vnwell"; OUT.mkdir(parents=True, exist_ok=True)
LOG = OUT / "run.log"


def _log(msg: str):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f: f.write(line + "\n")


def main():
    if LOG.exists(): LOG.unlink()
    rows = load_sebas_params()
    t_start = time.time()

    CONDITIONS = [
        ("vnwell_2.0",   {"vnwell": 2.0,  "use_well_diode": True,  "body_pdiode_to": "vnwell"}),
        ("vnwell_5.0",   {"vnwell": 5.0,  "use_well_diode": True,  "body_pdiode_to": "vnwell"}),
        ("vnwell_10.0",  {"vnwell": 10.0, "use_well_diode": True,  "body_pdiode_to": "vnwell"}),
        ("vnwell_20.0",  {"vnwell": 20.0, "use_well_diode": True,  "body_pdiode_to": "vnwell"}),
        ("vnwell_50.0",  {"vnwell": 50.0, "use_well_diode": True,  "body_pdiode_to": "vnwell"}),
        ("clamp_off",    {"vnwell": 2.0,  "use_well_diode": False, "body_pdiode_to": "off"}),
    ]

    results = {}
    for label, cfg_kw in CONDITIONS:
        _log(f"=== {label}  ({cfg_kw}) ===")
        cfg, M1, M2, bjt = build_base()
        for k, v in cfg_kw.items(): setattr(cfg, k, v)
        per_t = []
        for (vg1, vg2) in TARGETS:
            r = run_one(cfg, M1, M2, bjt, rows, vg1, vg2, log=_log)
            _log(f"  VG1={vg1} VG2={vg2}: rmse={r.get('rmse_dec',float('nan')):.3f} dec  "
                 f"jump(meas/model)={r.get('meas_jump_dec',0) or 0:.2f}/"
                 f"{r.get('model_jump_dec',float('nan')):.2f}  "
                 f"nan={r.get('has_nan')}  {r.get('elapsed_s',0):.1f}s")
            per_t.append(r)
        results[label] = per_t

    # Gate evaluation: at VG1=0.6
    def best_06(condmap):
        best = -1e9; best_label = None
        for lbl, lst in condmap.items():
            for r in lst:
                if r["VG1"] == 0.6:
                    mj = r.get("model_jump_dec", float("nan"))
                    if mj is not None and mj == mj and mj > best:
                        best = mj; best_label = lbl
        return best, best_label

    best_mj, best_lbl = best_06(results)
    discovery = best_mj > 0.5
    clamp_off_mj = next((r["model_jump_dec"] for r in results.get("clamp_off", [])
                        if r["VG1"] == 0.6), float("nan"))
    kill_shot = (clamp_off_mj is not None and clamp_off_mj == clamp_off_mj
                 and clamp_off_mj < 0.3)

    elapsed = time.time() - t_start
    summary = {
        "conditions": list(results.keys()),
        "results": results,
        "gates": {
            "infra_ok": elapsed < 45*60,
            "elapsed_s": elapsed,
            "best_model_jump_at_vg06": best_mj,
            "best_condition": best_lbl,
            "discovery_fold_gt_0p5": discovery,
            "clamp_off_model_jump_vg06": clamp_off_mj,
            "kill_shot_clamp_off_lt_0p3": bool(kill_shot),
        },
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    _log(f"DONE in {elapsed:.1f}s")
    _log(f"  best VG1=0.6 model_jump = {best_mj:.3f} dec  ({best_lbl})")
    _log(f"  DISCOVERY (fold>0.5): {discovery}")
    _log(f"  clamp_off VG1=0.6 jump = {clamp_off_mj:.3f} dec")
    _log(f"  KILL-SHOT (clamp_off<0.3): {kill_shot}")


if __name__ == "__main__":
    main()
