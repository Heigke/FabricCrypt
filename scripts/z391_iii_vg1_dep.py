"""z391 — Test D: VG1-dependent iii gain probe.

S3-C agent hypothesized a "VG1-rising regenerative path" — BJT β·M product
or VG1-gated avalanche. The simplest test: scale iii_body_gain by
(1 + k·(VG1 - 0.2)/0.4), so that VG1=0.6 has (1+k) times the impact-ion
drive vs VG1=0.2.

Targets: (VG1,VG2) ∈ {(0.2,0.10), (0.4,0.20), (0.6,0.20)}, clamp-off + etab=20.
k ∈ {0, 2, 5, 10}.

The PER_VG1 mapping already sets iii_body_gain = {6.78, 9.90, 10**9.10} =
{6.78, 9.90, 1.26e9} per VG1 (BIG values for VG1=0.4/0.6) but R-46 had
fold inverted. So just multiplying further may not help unless coupled
with a Vb-rise mechanism. Still — this is the clean k-sweep test.

Gates:
  DISCOVERY: VG1=0.6 model_jump > 0.5 dec for some k.
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _z384_shared import (ROOT, TARGETS, build_base, load_sebas_params, PER_VG1,
                          find_or_impute_row, make_overrides, load_measured,
                          patch_sd_scaled, metrics_one)

OUT = ROOT / "results/z391_iii_vg1_dep"; OUT.mkdir(parents=True, exist_ok=True)
LOG = OUT / "run.log"
ETAB = 20.0


def _log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f: f.write(line + "\n")


def vg1_scale(k, vg1):
    """Multiplier: 1 + k·(VG1-0.2)/0.4 → at VG1=0.2 → 1, at VG1=0.6 → 1+k."""
    return 1.0 + k * (vg1 - 0.2) / 0.4


def run_one_k(cfg, M1, M2, bjt, rows, vg1, vg2, k):
    from nsram.bsim4_port.nsram_cell_2T import forward_2t
    _, iii_base, Rs = PER_VG1[vg1]
    cfg.iii_body_gain = iii_base * vg1_scale(k, vg1)
    cfg.vnwell_Rs = Rs
    sd_M1 = cfg.size_dep_M1(M1); sd_M2 = cfg.size_dep_M2(M2)
    Vd_m, Id_m, _ = load_measured(vg1, vg2)
    row = find_or_impute_row(rows, vg1, vg2)
    P_M1, P_M2 = make_overrides(row, etab_override=ETAB)
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
        return dict(VG1=vg1, VG2=vg2, k=k, iii_gain=cfg.iii_body_gain,
                    rmse_dec=rmse, meas_jump_dec=mj, model_jump_dec=mdlj,
                    elapsed_s=time.time()-t0)
    except Exception as e:
        return dict(VG1=vg1, VG2=vg2, k=k, error=str(e),
                    rmse_dec=float("nan"), model_jump_dec=float("nan"),
                    elapsed_s=time.time()-t0)


def main():
    if LOG.exists(): LOG.unlink()
    rows = load_sebas_params()
    t0 = time.time()
    K_VALUES = [0, 2, 5, 10]
    results = {}
    for k in K_VALUES:
        lbl = f"k_{k}"
        _log(f"=== {lbl}  scale@VG1=0.6 = {vg1_scale(k, 0.6):.2f}× ===")
        cfg, M1, M2, bjt = build_base()
        cfg.use_well_diode = False
        cfg.body_pdiode_to = "off"
        per_t = []
        for vg1, vg2 in TARGETS:
            r = run_one_k(cfg, M1, M2, bjt, rows, vg1, vg2, k)
            _log(f"  VG1={vg1} VG2={vg2}: iii_gain={r.get('iii_gain',0):.3e}  "
                 f"rmse={r.get('rmse_dec',float('nan')):.3f} "
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
    summary = {"etab": ETAB, "results": results,
               "gates": {"best_model_jump_at_vg06": best,
                         "best_condition": best_lbl,
                         "discovery_fold_gt_0p5": discovery,
                         "elapsed_s": elapsed}}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    _log(f"DONE {elapsed:.1f}s best VG1=0.6 mj={best:.3f} ({best_lbl}) "
         f"DISCOVERY={discovery}")


if __name__ == "__main__":
    main()
