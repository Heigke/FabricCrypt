"""z390 — Test C: BSIM4 beta0 (IIMOD scale) sweep at VG1=0.6.

Iii ~ T2 · (Vds-Vdseff) · exp(-beta0/(Vds-Vdseff)) · Idsa·Vdseff
If beta0 is too high, exp(-beta0/diff) ≈ 0 and impact-ion is suppressed.
But z388 showed diffVds≈1.27 V at VG1=0.6, so for the R-46 default beta0=20
the exp argument is exp(-15.7)≈1.5e-7 — small but non-zero.

We sweep beta0 ∈ {default=20, 10, 5, 2, 1} at VG1=0.6, clamp-off + etab=20.

Lowering beta0 should re-energize Iii AT FIXED Ids_M1. But if the
fundamental problem is Ids_M1 itself (because Vsint pumps M1 subthreshold),
lowering beta0 won't help much.

Gates:
  DISCOVERY: VG1=0.6 model_jump > 0.5 dec at any beta0.
"""
from __future__ import annotations
import sys, json, time, math
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _z384_shared import (ROOT, build_base, load_sebas_params, find_or_impute_row,
                          make_overrides, load_measured, patch_sd_scaled,
                          metrics_one, PER_VG1)

OUT = ROOT / "results/z390_beta0"; OUT.mkdir(parents=True, exist_ok=True)
LOG = OUT / "run.log"
ETAB = 20.0


def _log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f: f.write(line + "\n")


def run_with_beta0(cfg, M1, M2, bjt, rows, vg1, vg2, beta0_val):
    from nsram.bsim4_port.nsram_cell_2T import forward_2t
    _, iii, Rs = PER_VG1[vg1]
    cfg.iii_body_gain = iii
    cfg.vnwell_Rs = Rs
    sd_M1 = cfg.size_dep_M1(M1); sd_M2 = cfg.size_dep_M2(M2)
    Vd_m, Id_m, _ = load_measured(vg1, vg2)
    row = find_or_impute_row(rows, vg1, vg2)
    P_M1, P_M2 = make_overrides(row, etab_override=ETAB)
    if P_M1 is None: P_M1 = {}
    if beta0_val is not None:
        P_M1["beta0"] = float(beta0_val)
    Vd_t = torch.tensor(Vd_m, dtype=torch.float64)
    t0 = time.time()
    try:
        with patch_sd_scaled(sd_M1, P_M1), patch_sd_scaled(sd_M2, P_M2):
            out = forward_2t(cfg, model_M1=M1, model_M2=M2, bjt=bjt, Vd_seq=Vd_t,
                             VG1=torch.tensor(vg1, dtype=torch.float64),
                             VG2=torch.tensor(vg2, dtype=torch.float64),
                             warm_start=True)
        Id_p = np.abs(out["Id"].detach().cpu().numpy())
        rmse, mj, mdlj, npts = metrics_one(Vd_m, Id_m, Id_p)
        return dict(VG1=vg1, VG2=vg2, beta0=beta0_val,
                    rmse_dec=rmse, meas_jump_dec=mj, model_jump_dec=mdlj,
                    elapsed_s=time.time()-t0,
                    Vd=Vd_m.tolist(), Id_m=Id_m.tolist(), Id_p=Id_p.tolist())
    except Exception as e:
        return dict(VG1=vg1, VG2=vg2, beta0=beta0_val, error=str(e),
                    rmse_dec=float("nan"), model_jump_dec=float("nan"),
                    elapsed_s=time.time()-t0)


def main():
    if LOG.exists(): LOG.unlink()
    rows = load_sebas_params()
    t0 = time.time()

    # Probe at VG1=0.6 (the failure case) AND VG1=0.2 (control).
    TGTS = [(0.6, 0.20), (0.2, 0.10)]
    BETA0_VALUES = [None, 10.0, 5.0, 2.0, 1.0, 0.5, 0.2]  # None = card default

    results = {}
    for b0 in BETA0_VALUES:
        lbl = f"beta0_default" if b0 is None else f"beta0_{b0}"
        _log(f"=== {lbl}  clamp-off + etab={ETAB} ===")
        cfg, M1, M2, bjt = build_base()
        cfg.use_well_diode = False
        cfg.body_pdiode_to = "off"
        per_t = []
        for vg1, vg2 in TGTS:
            r = run_with_beta0(cfg, M1, M2, bjt, rows, vg1, vg2, b0)
            _log(f"  VG1={vg1} VG2={vg2}: rmse={r.get('rmse_dec',float('nan')):.3f} "
                 f"jump(meas/model)={r.get('meas_jump_dec',0) or 0:.2f}/"
                 f"{r.get('model_jump_dec',float('nan')):.2f}  "
                 f"{r.get('elapsed_s',0):.1f}s")
            per_t.append(r)
        results[lbl] = per_t

    best = -1e9; best_lbl = None
    for lbl, lst in results.items():
        for r in lst:
            if r["VG1"] == 0.6:
                mj = r.get("model_jump_dec", float("nan"))
                if mj is not None and mj == mj and mj > best:
                    best = mj; best_lbl = lbl
    discovery = best > 0.5

    elapsed = time.time() - t0
    summary = {
        "etab": ETAB, "results": results,
        "gates": {
            "best_model_jump_at_vg06": best, "best_condition": best_lbl,
            "discovery_fold_gt_0p5": discovery, "elapsed_s": elapsed,
        }
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    _log(f"DONE {elapsed:.1f}s best VG1=0.6 mj={best:.3f} ({best_lbl}) "
         f"DISCOVERY={discovery}")


if __name__ == "__main__":
    main()
