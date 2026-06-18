"""z389 — Test B: vnwell_Rs → 1 Ω (effectively zero) + clamp-off + etab=20.

z388 confirmed Iii(VG1=0.6) is 3.4e-4 of Iii(VG1=0.2) — but ROOT cause is NOT
Vdsat saturation (diffVds=1.27V, plenty of room). It's that Vsint is being
pumped up to 0.19-0.23 V at high VG1, which gates M1 OFF (Vgs_M1 = VG1 - Vsint
becomes ~0.4 V, in subthreshold) — and Iii is proportional to Ids_M1.

If we force vnwell_Rs → 1 Ω, the well/body coupling can no longer drive Vsint
high.  Combined with clamp-off + etab=20, does VG1=0.6 fold appear?

Compare: PER_VG1 R values (1889 / 1092 / 417 Ω) vs uniform R_S=1.

Targets: (VG1,VG2) ∈ {(0.2,0.10), (0.4,0.20), (0.6,0.20)}.

Gates:
  DISCOVERY: VG1=0.6 model_jump > 0.5 dec.
  AMBITIOUS: rmse_dec < 0.5 across all 3 targets.
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _z384_shared import (ROOT, TARGETS, build_base, load_sebas_params, run_one,
                          PER_VG1)

OUT = ROOT / "results/z389_rs_zero"; OUT.mkdir(parents=True, exist_ok=True)
LOG = OUT / "run.log"

ETAB = 20.0


def _log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f: f.write(line + "\n")


def run_one_fixed_rs(cfg, M1, M2, bjt, rows, vg1, vg2, rs_val, etab_override):
    """Like _z384_shared.run_one but FORCE cfg.vnwell_Rs = rs_val (override the
    PER_VG1 mapping that run_one applies internally)."""
    from nsram.bsim4_port.nsram_cell_2T import forward_2t
    from _z384_shared import (find_or_impute_row, make_overrides, load_measured,
                              patch_sd_scaled, metrics_one)
    _, iii, _Rs_skipped = PER_VG1[vg1]
    cfg.iii_body_gain = iii
    cfg.vnwell_Rs = float(rs_val)              # FORCE override
    sd_M1 = cfg.size_dep_M1(M1); sd_M2 = cfg.size_dep_M2(M2)
    Vd_m, Id_m, fname = load_measured(vg1, vg2)
    row = find_or_impute_row(rows, vg1, vg2)
    P_M1, P_M2 = make_overrides(row, etab_override=etab_override)
    Vd_t = torch.tensor(Vd_m, dtype=torch.float64)
    t0 = time.time()
    try:
        with patch_sd_scaled(sd_M1, P_M1), patch_sd_scaled(sd_M2, P_M2):
            out = forward_2t(cfg, model_M1=M1, model_M2=M2, bjt=bjt, Vd_seq=Vd_t,
                             VG1=torch.tensor(vg1, dtype=torch.float64),
                             VG2=torch.tensor(vg2, dtype=torch.float64),
                             warm_start=True)
        dt = time.time() - t0
        Id_p = np.abs(out["Id"].detach().cpu().numpy())
        rmse, mj, mdlj, npts = metrics_one(Vd_m, Id_m, Id_p)
        return dict(VG1=vg1, VG2=vg2, rmse_dec=rmse, meas_jump_dec=mj,
                    model_jump_dec=mdlj, n_pts=npts, elapsed_s=dt,
                    Vd=Vd_m.tolist(), Id_m=Id_m.tolist(), Id_p=Id_p.tolist())
    except Exception as e:
        return dict(VG1=vg1, VG2=vg2, rmse_dec=float("nan"),
                    model_jump_dec=float("nan"), error=str(e),
                    elapsed_s=time.time()-t0)


def main():
    if LOG.exists(): LOG.unlink()
    rows = load_sebas_params()
    t0 = time.time()

    # Configs: (label, vnwell_Rs override or "default" to use PER_VG1)
    CONDS = [
        ("default_Rs",   None),      # PER_VG1 R-46 values
        ("Rs_1e6",       1.0e6),
        ("Rs_1e4",       1.0e4),
        ("Rs_1e2",       1.0e2),
        ("Rs_1",         1.0),
    ]

    results = {}
    for label, rs_val in CONDS:
        _log(f"=== {label}  vnwell_Rs={rs_val}  clamp-off  etab={ETAB} ===")
        cfg, M1, M2, bjt = build_base()
        cfg.use_well_diode = False
        cfg.body_pdiode_to = "off"
        per_t = []
        for vg1, vg2 in TARGETS:
            if rs_val is None:
                r = run_one(cfg, M1, M2, bjt, rows, vg1, vg2,
                            etab_override=ETAB, log=_log)
            else:
                r = run_one_fixed_rs(cfg, M1, M2, bjt, rows, vg1, vg2,
                                     rs_val, etab_override=ETAB)
            _log(f"  VG1={vg1} VG2={vg2}: rmse={r.get('rmse_dec',float('nan')):.3f} "
                 f"jump(meas/model)={r.get('meas_jump_dec',0) or 0:.2f}/"
                 f"{r.get('model_jump_dec',float('nan')):.2f}  "
                 f"{r.get('elapsed_s',0):.1f}s")
            per_t.append(r)
        results[label] = per_t

    # Discovery: best VG1=0.6 model_jump.
    best = -1e9; best_lbl = None
    for lbl, lst in results.items():
        for r in lst:
            if r["VG1"] == 0.6:
                mj = r.get("model_jump_dec", float("nan"))
                if mj is not None and mj == mj and mj > best:
                    best = mj; best_lbl = lbl
    discovery = best > 0.5

    # Ambitious: any condition with all 3 rmse < 0.5
    ambitious = False; ambi_lbl = None
    for lbl, lst in results.items():
        rmses = [r.get("rmse_dec", float("nan")) for r in lst]
        if all((rm == rm and rm < 0.5) for rm in rmses):
            ambitious = True; ambi_lbl = lbl; break

    elapsed = time.time() - t0
    summary = {
        "etab": ETAB, "results": results,
        "gates": {
            "best_model_jump_at_vg06": best, "best_condition": best_lbl,
            "discovery_fold_gt_0p5": discovery,
            "ambitious_rmse_lt_0p5_all_3": ambitious,
            "ambitious_label": ambi_lbl,
            "elapsed_s": elapsed,
        },
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    _log(f"DONE {elapsed:.1f}s best VG1=0.6 mj={best:.3f} ({best_lbl}) "
         f"DISCOVERY={discovery}  AMBITIOUS={ambitious}({ambi_lbl})")


if __name__ == "__main__":
    main()
