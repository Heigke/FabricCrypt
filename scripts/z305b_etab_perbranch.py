"""z305b — ETAB_M1 per-branch bug fix of z305.

z305 applied ETAB_M1 = +1.8 as a GLOBAL constant across all V_G1 branches.
SA1 canonical (three_branch_params_extracted.json, ETAB_M1_vs_VG2) shows
ETAB_M1 is strongly V_G1-dependent:
    V_G1 = 0.2  →  ETAB_M1 ≈ 0.95   (red branch, V_G2 near 0 V)
    V_G1 = 0.4  →  ETAB_M1 ≈ 1.70   (blue branch)
    V_G1 = 0.6  →  ETAB_M1 ≈ 2.50   (black branch)

Hypothesis: the V_G1=0.2 regression (+2.5 dec, 2.06 → 4.56) in z305 is
the ETAB bug, not physics. Per-branch ETAB should restore V_G1=0.2.

Pre-registered gates (locked, per oracle O50):
  PASS-bug-confirmed : V_G1=0.2 log-RMSE ≤ 2.30 dec (within 0.3 of
                       z304 baseline 2.06)
  BONUS PASS         : cell-wide median < 0.5 dec
  FAIL               : V_G1=0.2 stays > 3.0 dec (real physics, not bug)
"""
from __future__ import annotations
import os
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "4")

import argparse
import csv
import importlib.util
import json
import math
import re
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

torch.set_default_dtype(torch.float64)

_ENV_ROOT = os.environ.get("NSRAM_REPO_ROOT")
if _ENV_ROOT:
    ROOT = Path(_ENV_ROOT)
elif Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy").exists():
    ROOT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
elif Path("/home/daedalus/AMD_gfx1151_energy").exists():
    ROOT = Path("/home/daedalus/AMD_gfx1151_energy")
else:
    ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data/sebas_2026_04_22"
OUT_DIR = ROOT / "results/z305b_etab_perbranch"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64

BF_GRID    = [500, 1000, 3000, 9000]
RS_GRID    = [0, 1.0e9, 1.0e10]
ALPHA0_FIX = 7.84e-5
RS_FALLBACK = 1.0e30

# ---- SA1 canonical per-V_G1 (per-branch ETAB_M1 — BUG FIX vs z305) ----
SA1_PER_VG1 = {
    0.2: {"K1": 0.558, "mbjt": 0.001, "BETA0": 10.75, "ETAB": 0.95},
    0.4: {"K1": 0.538, "mbjt": 1.0,   "BETA0": 19.0,  "ETAB": 1.70},
    0.6: {"K1": 0.418, "mbjt": 1.0,   "BETA0": 20.0,  "ETAB": 2.50},
}

VG1_DIRS = {
    0.2: "2vHCa-2 I-Vs@VG2 VG1=0.2 vnwell=2",
    0.4: "2vHCa-2 I-Vs@VG2 VG1=0.4 vnwell=2",
    0.6: "2vHCa-2 I-Vs@VG2 VG1=0.6 vnwell=2",
}
VG2_RE = re.compile(r"VG2=(-?\d+\.\d+)")


def _load_module(name, path):
    sp = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(mod)
    return mod


@contextmanager
def patch_sd_scaled(sd, overrides):
    if not overrides:
        yield; return
    saved = {}
    try:
        for k, v in overrides.items():
            saved[k] = sd.scaled.get(k, None)
            sd.scaled[k] = v
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                sd.scaled.pop(k, None)
            else:
                sd.scaled[k] = v


def load_sebas_params():
    path = DATA / "2Tcell_BSIM_param_DC.csv"
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            row = {}
            for k, v in r.items():
                try:
                    row[k] = float(v)
                except (ValueError, TypeError):
                    row[k] = float("nan")
            rows.append(row)
    return rows


def find_params(rows, VG1, VG2, atol=1e-3):
    for r in rows:
        if abs(r["VG1"] - VG1) < atol and abs(r["VG2"] - VG2) < atol:
            return r
    return None


def load_curves(vg1_filter=None):
    curves = []
    for vg1, subdir in VG1_DIRS.items():
        if vg1_filter is not None and abs(vg1 - vg1_filter) > 1e-3:
            continue
        d = DATA / subdir
        for csv_path in sorted(d.glob("StandardIV*.csv")):
            m = VG2_RE.search(csv_path.name)
            if not m:
                continue
            vg2 = float(m.group(1))
            try:
                arr = np.loadtxt(csv_path, delimiter=",", skiprows=1, usecols=(0, 1))
            except Exception as e:
                print(f"[z305b] load fail {csv_path.name}: {e}", flush=True)
                continue
            if arr.ndim != 2:
                continue
            half = len(arr) // 2
            Vd = arr[:half, 0]
            Id = np.abs(arr[:half, 1])
            mask = (Vd >= 0.05) & (Vd <= 2.0)
            Vd, Id = Vd[mask], Id[mask]
            if len(Vd) < 10:
                continue
            idx = np.linspace(0, len(Vd) - 1, 30).astype(int)
            Vd, Id = Vd[idx], Id[idx]
            curves.append({
                "VG1": vg1, "VG2": vg2, "file": csv_path.name,
                "Vd": torch.tensor(Vd, dtype=DTYPE),
                "Id": torch.tensor(Id, dtype=DTYPE),
            })
    return curves


def make_row_overrides_sa1(sebas_row, vg1, M2_STATIC):
    """SA1-canonical overrides with PER-BRANCH ETAB_M1 (the z305 bug fix)."""
    if sebas_row is None:
        return None, None
    sa1 = SA1_PER_VG1.get(round(vg1, 2))
    if sa1 is None:
        return None, None

    P_M1 = {
        "etab":   torch.tensor(sa1["ETAB"],  dtype=DTYPE),  # ← PER-BRANCH (fix)
        "k1":     torch.tensor(sa1["K1"],    dtype=DTYPE),
        "alpha0": torch.tensor(ALPHA0_FIX,   dtype=DTYPE),
        "beta0":  torch.tensor(sa1["BETA0"], dtype=DTYPE),
    }
    P_M2 = {}
    nf = sebas_row.get("NFACTOR", float("nan"))
    if not math.isnan(nf):
        P_M2["nfactor"] = torch.tensor(float(nf), dtype=DTYPE)
    for k, v in M2_STATIC.items():
        if k not in P_M2:
            P_M2[k] = torch.tensor(float(v), dtype=DTYPE)
    return P_M1, P_M2


def evaluate_cell(*, vg1, bf, rs, curves, sebas_rows,
                   z91f_mod, cfg, M1, M2, sd_M1, sd_M2, forward_2t):
    from nsram.bsim4_port.bjt import GummelPoonNPN

    cfg.vnwell_Rs = float(rs) if rs > 0 else RS_FALLBACK
    if hasattr(cfg, "invalidate"):
        cfg.invalidate()

    sa1 = SA1_PER_VG1[round(vg1, 2)]

    log_eps = 1e-15
    per_curve = []
    for c in curves:
        sebas_row = find_params(sebas_rows, c["VG1"], c["VG2"])
        if sebas_row is None:
            continue
        P_M1, P_M2 = make_row_overrides_sa1(sebas_row, vg1,
                                              z91f_mod.M2_STATIC_OVERRIDES)
        if P_M1 is None:
            continue

        bjt = GummelPoonNPN.from_sebas_card()
        if not math.isnan(sebas_row.get("IS", float("nan"))):
            bjt.Is = float(sebas_row["IS"])
        area = float(sebas_row.get("area", 1e-6))
        if math.isnan(area):
            area = 1e-6
        mbjt = sa1["mbjt"]
        bjt.area = area * mbjt
        bjt.Bf = float(bf)

        try:
            with torch.no_grad(), \
                  patch_sd_scaled(sd_M1, P_M1), \
                  patch_sd_scaled(sd_M2, P_M2):
                out = forward_2t(cfg, M1, bjt,
                                   c["Vd"], torch.tensor(c["VG1"]),
                                   torch.tensor(c["VG2"]),
                                   warm_start=True, use_homotopy=True)
            Id_pred = out["Id"].abs()
            conv = torch.tensor([bool(x) for x in out["converged"]])
        except Exception as e:
            per_curve.append({"VG2": float(c["VG2"]), "log_rmse": float("inf"),
                                "signed_dec": float("nan"), "n_conv": 0,
                                "err": str(e)[:120]})
            continue

        log_p = torch.log10(Id_pred + log_eps)
        log_m = torch.log10(c["Id"] + log_eps)
        if conv.any():
            mask = conv
            diff = (log_p[mask] - log_m[mask])
            rmse = float(torch.sqrt((diff ** 2).mean()))
            signed = float(torch.median(diff))
        else:
            rmse = float("inf")
            signed = float("nan")
        per_curve.append({"VG2": float(c["VG2"]), "log_rmse": rmse,
                           "signed_dec": signed, "n_conv": int(conv.sum())})

    finite = [pc for pc in per_curve if math.isfinite(pc["log_rmse"])]
    if finite:
        rmses = np.array([pc["log_rmse"] for pc in finite])
        signs = np.array([pc["signed_dec"] for pc in finite
                          if math.isfinite(pc["signed_dec"])])
        med = float(np.median(rmses))
        signed_med = float(np.median(signs)) if signs.size else float("nan")
        p90 = float(np.percentile(rmses, 90))
    else:
        med = float("inf"); signed_med = float("nan"); p90 = float("inf")
    return {
        "vg1": vg1, "bf": bf, "alpha0": ALPHA0_FIX, "rs": rs,
        "etab_used": sa1["ETAB"],
        "median_log_rmse": med, "signed_dec_median": signed_med,
        "p90_log_rmse": p90, "n_finite": len(finite), "n_total": len(per_curve),
        "per_curve": per_curve,
    }


def build_models_once():
    z91f = _load_module("z91f", ROOT / "scripts/z91f_validate_with_sebas_params.py")
    from nsram.bsim4_port.model_card import BSIM4Model
    from nsram.bsim4_port.nsram_cell_2T import NSRAMCell2TConfig, forward_2t
    from nsram.bsim4_port.temp import compute_size_dep
    from nsram.bsim4_port.geometry import Geometry

    text_M1 = (DATA / "M1_130DNWFB.txt").read_text()
    M1 = BSIM4Model.from_spice(text_M1, model_type="nmos")
    z91f.patch_model_values(M1, type_n=True)

    text_M2 = (DATA / "M2_130bulkNSRAM.txt").read_text()
    M2 = BSIM4Model.from_spice(text_M2, model_type="nmos")
    z91f.patch_model_values(M2, type_n=True)

    cfg = NSRAMCell2TConfig(use_iii=True, use_gidl=True, use_bjt=True,
                              newton_max_iters=50)
    sd_M1 = compute_size_dep(M1, Geometry(L=cfg.Ln, W=cfg.Wn), T_C=cfg.T_C)
    sd_M2 = compute_size_dep(M2,
                              Geometry(L=cfg.Ln * cfg.M2_length_factor,
                                       W=cfg.Wn), T_C=cfg.T_C)
    cfg._sd_M1 = sd_M1
    cfg._sd_M2 = sd_M2
    return z91f, cfg, M1, M2, sd_M1, sd_M2, forward_2t


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bf", type=int, required=True)
    p.add_argument("--rs", type=float, required=True)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    t0 = time.time()
    print(f"[z305b] device={DEVICE} start {time.strftime('%H:%M:%S')}", flush=True)
    print(f"[z305b] Bf={args.bf}  Rs={args.rs}  alpha0_fix={ALPHA0_FIX}", flush=True)
    print(f"[z305b] per-branch ETAB: 0.2→0.95  0.4→1.70  0.6→2.50", flush=True)

    if args.out:
        out_path = Path(args.out)
    else:
        rs_tag = f"{args.rs:.0e}" if args.rs > 0 else "0"
        out_path = OUT_DIR / f"corrective_bf_{args.bf}_rs_{rs_tag}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sebas_rows = load_sebas_params()
    z91f, cfg, M1, M2, sd_M1, sd_M2, forward_2t = build_models_once()
    print(f"[z305b] models built  ({time.time() - t0:.1f}s)", flush=True)

    curves_per_branch = {vg1: load_curves(vg1_filter=vg1) for vg1 in [0.2, 0.4, 0.6]}
    for vg1, cs in curves_per_branch.items():
        print(f"[z305b] V_G1={vg1}: {len(cs)} curves", flush=True)

    rows = []
    for vg1 in [0.2, 0.4, 0.6]:
        t_cell = time.time()
        r = evaluate_cell(
            vg1=vg1, bf=args.bf, rs=args.rs,
            curves=curves_per_branch[vg1], sebas_rows=sebas_rows,
            z91f_mod=z91f, cfg=cfg, M1=M1, M2=M2,
            sd_M1=sd_M1, sd_M2=sd_M2, forward_2t=forward_2t,
        )
        rows.append(r)
        print(f"[z305b] V_G1={vg1} Bf={args.bf} Rs={args.rs} ETAB={r['etab_used']} → "
              f"med={r['median_log_rmse']:.3f} signed={r['signed_dec_median']:+.3f} "
              f"n_finite={r['n_finite']}/{r['n_total']}  "
              f"({time.time()-t_cell:.1f}s)", flush=True)

    finite = [r for r in rows if math.isfinite(r["median_log_rmse"])]
    if finite:
        all_curve_rmses = []
        for r in rows:
            for pc in r["per_curve"]:
                if math.isfinite(pc["log_rmse"]):
                    all_curve_rmses.append(pc["log_rmse"])
        cellwide_med = float(np.median(all_curve_rmses)) if all_curve_rmses else float("inf")
        all_curve_signs = [pc["signed_dec"] for r in rows for pc in r["per_curve"]
                            if math.isfinite(pc.get("signed_dec", float("nan")))]
        cellwide_signed = float(np.median(all_curve_signs)) if all_curve_signs else float("nan")
        worst_branch = max(r["median_log_rmse"] for r in finite)
    else:
        cellwide_med = float("inf"); cellwide_signed = float("nan")
        worst_branch = float("inf")

    summary = {
        "script": "z305b_etab_perbranch",
        "bf": args.bf, "rs": args.rs, "alpha0": ALPHA0_FIX,
        "sa1_per_vg1": SA1_PER_VG1,
        "etab_m1_per_branch": {k: v["ETAB"] for k, v in SA1_PER_VG1.items()},
        "cellwide_median_log_rmse": cellwide_med,
        "cellwide_signed_dec_median": cellwide_signed,
        "worst_branch_median": worst_branch,
        "elapsed_s": time.time() - t0,
        "device": str(DEVICE),
        "rows": rows,
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\n[z305b] cell-wide median={cellwide_med:.3f}  signed={cellwide_signed:+.3f}  "
          f"worst-branch={worst_branch:.3f}", flush=True)
    print(f"[z305b] wrote {out_path}  ({time.time()-t0:.0f}s total)", flush=True)


if __name__ == "__main__":
    sys.exit(main() or 0)
