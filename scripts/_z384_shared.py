"""Shared helpers for z384/z385/z386 hypothesis 3+4 tests.

Reuses z376_R55a's loader pattern (build_base, load_sebas_params,
find_or_impute_row, make_overrides, load_measured, metrics_one).
"""
from __future__ import annotations
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")

import sys, json, math, csv, re, importlib.util, time
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "nsram"))
DATA = ROOT / "data/sebas_2026_04_22"

TARGETS = [(0.2, 0.10), (0.4, 0.20), (0.6, 0.20)]

BRANCH_FLAT = {
    0.4: {"ETAB": 1.9,  "K1": 0.53825, "ALPHA0": 7.842e-05, "BETA0": 19.0, "NFACTOR": 6.0, "trise": 10.59},
    0.6: {"ETAB": 2.5,  "K1": 0.41825, "ALPHA0": 7.842e-05, "BETA0": 20.0, "NFACTOR": 6.0, "trise": 9.04},
}
M2_STATIC = {"k1": 0.63825, "k2": -0.070435, "etab": -0.086777, "beta0": 18.0}
X_BEST = [1889.88, 1.8447, 9.1722, 1092.27, 1.5152, 9.8983, 417.63, 0.9036, 6.7846]
PER_VG1 = {0.2: (X_BEST[0], X_BEST[1], 10**X_BEST[2]),
           0.4: (X_BEST[3], X_BEST[4], 10**X_BEST[5]),
           0.6: (X_BEST[6], X_BEST[7], 10**X_BEST[8])}


@contextmanager
def patch_sd_scaled(sd, overrides):
    if not overrides: yield; return
    saved = {}
    try:
        for k, v in overrides.items():
            saved[k] = sd.scaled.get(k, None)
            sd.scaled[k] = float(v)
        yield
    finally:
        for k, v in saved.items():
            if v is None: sd.scaled.pop(k, None)
            else: sd.scaled[k] = v


def load_sebas_params():
    path = DATA / "2Tcell_BSIM_param_DC.csv"
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            row = {}
            for k, v in r.items():
                try: row[k] = float(v)
                except ValueError: row[k] = float("nan")
            rows.append(row)
    return rows


def find_or_impute_row(rows, VG1, VG2, atol=1e-3):
    target = None
    for r in rows:
        if abs(r["VG1"] - VG1) < atol and abs(r["VG2"] - VG2) < atol:
            target = dict(r); break
    if target is None: return None
    if math.isnan(target.get("K1", float("nan"))):
        branch = BRANCH_FLAT.get(round(VG1, 2))
        if branch is None: return target
        for k, v in branch.items():
            target[k] = float(v)
    return target


def make_overrides(row, etab_override=None):
    if row is None: return None, None
    P_M1 = {}
    for ck, pk in (("ETAB","etab"),("K1","k1"),("ALPHA0","alpha0"),("BETA0","beta0")):
        if not math.isnan(row.get(ck, float("nan"))):
            P_M1[pk] = float(row[ck])
    if etab_override is not None:
        P_M1["etab"] = float(etab_override)
    P_M2 = {}
    if not math.isnan(row.get("NFACTOR", float("nan"))):
        P_M2["nfactor"] = float(row["NFACTOR"])
    for k, v in M2_STATIC.items():
        P_M2.setdefault(k, float(v))
    return (P_M1 or None), (P_M2 or None)


def load_measured(vg1, vg2):
    sub = DATA / f"2vHCa-2 I-Vs@VG2 VG1={vg1} vnwell=2"
    pat = re.compile(rf"VG2={vg2:.2f}_VG={vg1}")
    for f in sorted(sub.glob("*.csv")):
        if pat.search(f.name):
            d = np.loadtxt(f, delimiter=",", skiprows=1)
            return d[:, 0], np.abs(d[:, 1]), f.name
    raise FileNotFoundError(f"no csv for VG1={vg1} VG2={vg2}")


def metrics_one(Vd_m, Id_m, Id_p):
    mask = (Id_m > 1e-15) & (Id_p > 1e-15) & np.isfinite(Id_p)
    rmse_dec = (float(np.sqrt(np.mean((np.log10(Id_p[mask]) - np.log10(Id_m[mask]))**2)))
                if mask.sum() >= 3 else float("nan"))
    dlog = np.diff(np.log10(np.maximum(Id_m, 1e-15)))
    Vmid = 0.5 * (Vd_m[1:] + Vd_m[:-1])
    valid_knee = Vmid >= 0.5
    if valid_knee.any() and len(dlog) > 0:
        dlog_masked = np.where(valid_knee, dlog, -np.inf)
        knee_idx = int(np.argmax(dlog_masked)) + 1
    else:
        knee_idx = None
    jump_dec = float(dlog.max()) if len(dlog) > 0 else None
    if knee_idx is not None and knee_idx < len(Id_p):
        lo = max(0, knee_idx-3); hi = min(len(Id_p), knee_idx+3)
        dlog_p_window = np.diff(np.log10(np.maximum(Id_p[lo:hi], 1e-15)))
        model_jump = float(dlog_p_window.max()) if len(dlog_p_window) else 0.0
    else:
        model_jump = float("nan")
    return rmse_dec, jump_dec, model_jump, int(mask.sum())


def build_base():
    sp = importlib.util.spec_from_file_location("v1", ROOT / "scripts/z96_narma10_pilot.py")
    v1 = importlib.util.module_from_spec(sp); sp.loader.exec_module(v1)
    from nsram.bsim4_port.nsram_cell_2T import NSRAMCell2TConfig
    from nsram.bsim4_port.bjt import GummelPoonNPN
    cfg = NSRAMCell2TConfig(use_iii=True, use_gidl=True, use_bjt=True, newton_max_iters=40)
    cfg.bjt_emitter_to_gnd = True
    cfg.body_pdiode_to = "vnwell"
    cfg.use_well_diode = True
    cfg.vnwell = 2.0
    cfg.body_pdiode_Js = 5.3675e-7 / 22e-12
    cfg.body_pdiode_n = 1.0535
    cfg.body_pdiode_Rs = 1.0e6
    M1, M2 = v1.build_calibrated_models()
    bjt = GummelPoonNPN.from_sebas_card()
    bjt.Va = 100.0; bjt.Is = 5e-9; bjt.Bf = 10000.0
    return cfg, M1, M2, bjt


def run_one(cfg, M1, M2, bjt, sebas_rows, vg1, vg2, etab_override=None, log=print):
    from nsram.bsim4_port.nsram_cell_2T import forward_2t
    _, iii, Rs = PER_VG1[vg1]
    cfg.iii_body_gain = iii
    cfg.vnwell_Rs = Rs
    sd_M1 = cfg.size_dep_M1(M1); sd_M2 = cfg.size_dep_M2(M2)
    Vd_m, Id_m, fname = load_measured(vg1, vg2)
    row = find_or_impute_row(sebas_rows, vg1, vg2)
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
        has_nan = bool(np.any(~np.isfinite(Id_p)))
        rmse, mj, mdlj, npts = metrics_one(Vd_m, Id_m, Id_p)
        return dict(VG1=vg1, VG2=vg2, rmse_dec=rmse, meas_jump_dec=mj,
                    model_jump_dec=mdlj, n_pts=npts, has_nan=has_nan,
                    elapsed_s=dt, Id_p=Id_p.tolist(), Vd=Vd_m.tolist(),
                    Id_m=Id_m.tolist())
    except Exception as e:
        return dict(VG1=vg1, VG2=vg2, rmse_dec=float("nan"), model_jump_dec=float("nan"),
                    has_nan=True, error=str(e), elapsed_s=time.time()-t0)
