"""z339 — cfg-diff audit: identify which cfg flag controls the gap between
z338 (3.43 dec floor) and z304 (0.99 dec).

z304 cfg was DEFAULTS: NSRAMCell2TConfig(use_iii, use_gidl, use_bjt, newton_max_iters=50).
z337/z338 added bjt_emitter_to_gnd=True (BJT Vbe fix from R-15/R-20).

Hypothesis: the BJT "fix" is actually a regression at z304's tuned operating
point. Test 6 variants holding z338 best params constant, varying ONE cfg
flag at a time, on the full 33-curve set.

Variants:
  A) baseline_z338   bjt_emitter_to_gnd=True (everything else default)
  B) emitter_default bjt_emitter_to_gnd=False (i.e. z304 cfg)
  C) eta_sigmoid_on  bjt_emitter_to_gnd=True + eta_sigmoid=True
  D) m2_body_vb      bjt_emitter_to_gnd=True + m2_body_gnd=False
  E) pdiode_gnd      bjt_emitter_to_gnd=True + body_pdiode_to="gnd"
  F) all_revert      bjt_emitter_to_gnd=False + DEFAULT lat_BV, default Rs (true z304)

All use z338 eval-21 best params for (alpha0, Bf, Va, Is, lat_BV, body_pdiode_Rs)
except variant F which lets these be the defaults z304 used.
"""
from __future__ import annotations
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
import sys, json, re, time, importlib.util
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "nsram"))

OUT = ROOT / "results/z339_cfg_diff"
OUT.mkdir(parents=True, exist_ok=True)
DATA = ROOT / "data/sebas_2026_04_22"

# z338 eval-21 best params
BEST = {
    "alpha0": 1.63357328192734e-05,
    "Bf":     2605.2882016162002,
    "Va":     0.3567358318716285,
    "Is":     3.2906845928467974e-10,
    "lat_BV": 4.018266147002578,
    "body_pdiode_Rs": 5480383.486345367,
}


def build_models():
    sp = importlib.util.spec_from_file_location("v1", ROOT / "scripts/z96_narma10_pilot.py")
    v1 = importlib.util.module_from_spec(sp); sp.loader.exec_module(v1)
    from nsram.bsim4_port.bjt import GummelPoonNPN
    M1, M2 = v1.build_calibrated_models()
    bjt = GummelPoonNPN.from_sebas_card()
    return M1, M2, bjt


def make_cfg(variant: str):
    from nsram.bsim4_port.nsram_cell_2T import NSRAMCell2TConfig
    cfg = NSRAMCell2TConfig(use_iii=True, use_gidl=True, use_bjt=True,
                             newton_max_iters=40)
    if variant == "A_baseline_z338":
        cfg.bjt_emitter_to_gnd = True
    elif variant == "B_emitter_default":
        cfg.bjt_emitter_to_gnd = False
    elif variant == "C_eta_sigmoid_on":
        cfg.bjt_emitter_to_gnd = True
        cfg.eta_sigmoid = True
    elif variant == "D_m2_body_vb":
        cfg.bjt_emitter_to_gnd = True
        cfg.m2_body_gnd = False
    elif variant == "E_pdiode_gnd":
        cfg.bjt_emitter_to_gnd = True
        cfg.body_pdiode_to = "gnd"
    elif variant == "F_all_revert":
        # z304 plain defaults; do NOT set bjt_emitter_to_gnd
        pass
    else:
        raise ValueError(variant)
    return cfg


def apply_best(cfg, M1, M2, bjt, use_best_pkg=True):
    sd_M1 = cfg.size_dep_M1(M1)
    sd_M2 = cfg.size_dep_M2(M2)
    if use_best_pkg:
        sd_M1.scaled["alpha0"] = float(BEST["alpha0"])
        sd_M2.scaled["alpha0"] = float(BEST["alpha0"])
        bjt.Bf = float(BEST["Bf"])
        bjt.Va = float(BEST["Va"])
        bjt.Is = float(BEST["Is"])
        cfg.lat_BV = float(BEST["lat_BV"])
        cfg.body_pdiode_Rs = float(BEST["body_pdiode_Rs"])
    else:
        # For F_all_revert: use bjt defaults from card
        bjt.Bf = 9000.0; bjt.Va = 0.55; bjt.Is = 1e-9


def load_curves():
    curves = []
    for f in sorted(DATA.glob("VG1*VG2*.csv")):
        m = re.search(r"VG1=([\d.\-]+)[_ ]*VG2=([\d.\-]+)", f.name)
        if not m: continue
        vg1 = float(m.group(1)); vg2 = float(m.group(2))
        d = np.loadtxt(f, delimiter=",", skiprows=1)
        curves.append({"VG1": vg1, "VG2": vg2, "Vd": d[:,0], "Id": np.abs(d[:,1]), "f": f.name})
    if not curves:
        for sub in DATA.iterdir():
            if not sub.is_dir(): continue
            m_vg1 = re.search(r"VG1=([\d.\-]+)", sub.name)
            if not m_vg1: continue
            vg1 = float(m_vg1.group(1))
            for f in sorted(sub.glob("*.csv")):
                m = re.search(r"VG2=([\-\d.]+)", f.name)
                if not m: continue
                vg2 = float(m.group(1))
                d = np.loadtxt(f, delimiter=",", skiprows=1)
                if d.ndim != 2 or d.shape[1] < 2: continue
                curves.append({"VG1": vg1, "VG2": vg2, "Vd": d[:,0], "Id": np.abs(d[:,1]), "f": f.name})
    return curves


def eval_curve(cfg, M1, M2, bjt, c, forward_2t):
    Vd = torch.tensor(c["Vd"], dtype=torch.float64)
    try:
        out = forward_2t(cfg, model_M1=M1, model_M2=M2, bjt=bjt,
                         Vd_seq=Vd,
                         VG1=torch.tensor(c["VG1"], dtype=torch.float64),
                         VG2=torch.tensor(c["VG2"], dtype=torch.float64),
                         warm_start=True)
        Id_pred = np.abs(out["Id"].detach().cpu().numpy())
        mask = (c["Id"] > 1e-15) & (Id_pred > 1e-15) & np.isfinite(Id_pred)
        if mask.sum() < 3:
            return float("nan")
        logr = np.log10(Id_pred[mask]) - np.log10(c["Id"][mask])
        return float(np.sqrt(np.mean(logr ** 2)))
    except Exception:
        return float("nan")


def run_variant(variant, M1, M2, bjt, curves, forward_2t):
    cfg = make_cfg(variant)
    apply_best(cfg, M1, M2, bjt, use_best_pkg=(variant != "F_all_revert"))
    t0 = time.time()
    decs = []
    nans = 0
    for c in curves:
        r = eval_curve(cfg, M1, M2, bjt, c, forward_2t)
        if np.isnan(r):
            nans += 1
        else:
            decs.append(r)
    dt = time.time() - t0
    med = float(np.median(decs)) if decs else float("nan")
    p25 = float(np.percentile(decs, 25)) if decs else float("nan")
    p75 = float(np.percentile(decs, 75)) if decs else float("nan")
    return {
        "variant": variant,
        "median_dec": med,
        "p25": p25,
        "p75": p75,
        "n_valid": len(decs),
        "n_nan": nans,
        "dt_s": dt,
        "per_curve_dec": decs,
    }


def main():
    M1, M2, bjt = build_models()
    from nsram.bsim4_port.nsram_cell_2T import forward_2t
    curves = load_curves()
    print(f"[z339] loaded {len(curves)} curves", flush=True)

    variants = [
        "A_baseline_z338",
        "B_emitter_default",
        "C_eta_sigmoid_on",
        "D_m2_body_vb",
        "E_pdiode_gnd",
        "F_all_revert",
    ]
    results = []
    for v in variants:
        # rebuild bjt fresh so previous variant's settings don't carry
        from nsram.bsim4_port.bjt import GummelPoonNPN
        bjt_v = GummelPoonNPN.from_sebas_card()
        print(f"[z339] running {v} ...", flush=True)
        r = run_variant(v, M1, M2, bjt_v, curves, forward_2t)
        print(f"[z339]   {v}: median={r['median_dec']:.3f} dec  "
              f"p25={r['p25']:.3f} p75={r['p75']:.3f}  "
              f"valid={r['n_valid']}/{r['n_valid']+r['n_nan']}  "
              f"dt={r['dt_s']:.1f}s", flush=True)
        results.append(r)

    summary = {
        "best_params_z338_eval21": BEST,
        "n_curves": len(curves),
        "variants": results,
        "ranking": sorted([(r["variant"], r["median_dec"]) for r in results],
                          key=lambda t: (np.inf if np.isnan(t[1]) else t[1])),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[z339] === RANKING (best to worst) ===", flush=True)
    for name, med in summary["ranking"]:
        print(f"    {name:25s}  {med:.3f} dec", flush=True)
    print(f"[z339] saved {OUT/'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
