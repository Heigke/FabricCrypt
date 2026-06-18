"""z304 — Sebas three-branch refit (full per-branch sweep).

Abandons Mario-zenodo BJT card as ground truth. Uses Sebas's canonical
2Tcell_BSIM_param_DC.csv (per-row NFACTOR/ETAB/BETA0/K1/ALPHA0/IS/area/
mbjt overrides) reconciled with three_branch_params_extracted.json.

Sweep (per V_G1 branch):
  Bf      ∈ {50, 200, 500, 1000, 3000, 9000}
  alpha0  ∈ {1e-5, 1e-4, 1e-3, 1e-2}     (replaces canonical 7.842e-5)
  Rs      ∈ {0 ↔ vnwell_Rs=1e30, 100, 500, 2000}    (M1/M2 source resistance via vnwell_Rs)

Per V_G1, 6×4×4 = 96 cells. Total 3×96 = 288.

Args
----
--vg1 {0.2, 0.4, 0.6}            single branch to sweep
--bf N                            single Bf value to sweep (16 inner cells)
--out PATH                         output JSON path
--all                              run full 288-cell sweep locally on this node

Output
------
JSON dict { 'rows': [{vg1, bf, alpha0, Rs, median_log_rmse, signed_dec,
                       per_curve: [{vg2, log_rmse, signed_dec}], n_curves},
                      ...]}

Gates evaluated by z304_aggregate.py:
  PASS-conservative : per-branch median log-RMSE < 0.7 dec
  AMBITIOUS         : per-branch median < 0.3 AND |signed| < 0.1
  SAFETY            : per-branch < 1.5 dec
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
OUT_DIR = ROOT / "results/z304_sebas_refit"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64

# ---- Sweep grids ----
BF_GRID     = [50, 500, 3000, 9000]   # reduced from [50,200,500,1000,3000,9000] for wall-budget
ALPHA0_GRID = [1e-5, 1e-4, 1e-3, 1e-2]
RS_GRID     = [0, 1e8, 1e9, 1e10]   # vnwell_Rs (well↔body shunt); 0 → disabled (1e30)
                                     # spec asked {0,100,500,2000} but those are off-scale
                                     # for vnwell_Rs (existing code uses 1e9..1e10).
RS_FALLBACK = 1.0e30                # disables the well-body resistive shunt

VG1_DIRS = {
    0.2: "2vHCa-2 I-Vs@VG2 VG1=0.2 vnwell=2",
    0.4: "2vHCa-2 I-Vs@VG2 VG1=0.4 vnwell=2",
    0.6: "2vHCa-2 I-Vs@VG2 VG1=0.6 vnwell=2",
}
VG2_RE = re.compile(r"VG2=(-?\d+\.\d+)")


# Reuse z91f helpers (patch_sd_scaled, patch_model_values, etc.) via import.
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
    """Sebas IV curves (33 total). vg1_filter selects one branch if given."""
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
                arr = np.loadtxt(csv_path, delimiter=",", skiprows=1,
                                  usecols=(0, 1))
            except Exception as e:
                print(f"[z304] load fail {csv_path.name}: {e}", flush=True)
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


def make_row_overrides(sebas_row, alpha0_override, M2_STATIC):
    """Build (P_M1, P_M2) override dicts for one CSV row.

    alpha0_override replaces sebas_row['ALPHA0'] globally (sweep knob).
    """
    if sebas_row is None:
        return None, None
    P_M1 = {}
    if not math.isnan(sebas_row.get("ETAB", float("nan"))):
        P_M1["etab"] = torch.tensor(sebas_row["ETAB"], dtype=DTYPE)
    if not math.isnan(sebas_row.get("K1", float("nan"))):
        P_M1["k1"] = torch.tensor(sebas_row["K1"], dtype=DTYPE)
    # alpha0: sweep value, NOT CSV (CSV is constant 7.842e-5)
    P_M1["alpha0"] = torch.tensor(float(alpha0_override), dtype=DTYPE)
    if not math.isnan(sebas_row.get("BETA0", float("nan"))):
        P_M1["beta0"] = torch.tensor(sebas_row["BETA0"], dtype=DTYPE)
    P_M2 = {}
    if not math.isnan(sebas_row.get("NFACTOR", float("nan"))):
        P_M2["nfactor"] = torch.tensor(sebas_row["NFACTOR"], dtype=DTYPE)
    for k, v in M2_STATIC.items():
        if k not in P_M2:
            P_M2[k] = torch.tensor(float(v), dtype=DTYPE)
    return P_M1 or None, P_M2 or None


def evaluate_cell(*, vg1, bf, alpha0, rs, curves, sebas_rows,
                   z91f_mod, cfg, M1, M2, sd_M1, sd_M2, forward_2t):
    """Evaluate one (V_G1, Bf, alpha0, Rs) cell over all curves of that branch."""
    from nsram.bsim4_port.bjt import GummelPoonNPN

    # Apply Rs (override vnwell_Rs in cfg)
    cfg.vnwell_Rs = float(rs) if rs > 0 else RS_FALLBACK
    cfg.invalidate() if hasattr(cfg, "invalidate") else None

    log_eps = 1e-15
    per_curve = []
    for c in curves:
        sebas_row = find_params(sebas_rows, c["VG1"], c["VG2"])
        if sebas_row is None or math.isnan(sebas_row.get("K1", float("nan"))):
            continue
        P_M1, P_M2 = make_row_overrides(sebas_row, alpha0,
                                          z91f_mod.M2_STATIC_OVERRIDES)
        # Build per-row BJT instance using sweep Bf
        bjt = GummelPoonNPN.from_sebas_card()
        if not math.isnan(sebas_row.get("IS", float("nan"))):
            bjt.Is = float(sebas_row["IS"])
        area = float(sebas_row.get("area", 1e-6))
        if math.isnan(area):
            area = 1e-6
        mbjt = float(sebas_row.get("mbjt", 1.0))
        if math.isnan(mbjt):
            mbjt = 1.0
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
            per_curve.append({"VG2": c["VG2"], "log_rmse": float("inf"),
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
        "vg1": vg1, "bf": bf, "alpha0": alpha0, "rs": rs,
        "median_log_rmse": med, "signed_dec_median": signed_med,
        "p90_log_rmse": p90, "n_finite": len(finite), "n_total": len(per_curve),
        "per_curve": per_curve,
    }


def build_models_once():
    """Load M1/M2/cfg/sd_M1/sd_M2 (singleton; cell-sweep mutates cfg.vnwell_Rs)."""
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
    p.add_argument("--vg1", type=float, default=None,
                    help="Single V_G1 branch (0.2/0.4/0.6). If absent and --all not set, all branches.")
    p.add_argument("--bf", type=int, default=None,
                    help="Single Bf value. If absent, sweep full BF_GRID.")
    p.add_argument("--out", type=str, default=None, help="Output JSON path")
    p.add_argument("--all", action="store_true",
                    help="Run full 288-cell sweep locally on this node.")
    args = p.parse_args()

    t0 = time.time()
    print(f"[z304] device={DEVICE} starting at {time.strftime('%H:%M:%S')}", flush=True)

    vg1_list = [args.vg1] if args.vg1 is not None else [0.2, 0.4, 0.6]
    bf_list  = [args.bf]  if args.bf  is not None else BF_GRID

    if args.out:
        out_path = Path(args.out)
    else:
        tag = []
        if args.vg1 is not None: tag.append(f"vg1_{args.vg1:.1f}")
        if args.bf  is not None: tag.append(f"bf_{args.bf}")
        if not tag: tag = ["all"]
        out_path = OUT_DIR / f"refit_{'_'.join(tag)}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[z304] vg1={vg1_list} bf={bf_list} alpha0={ALPHA0_GRID} rs={RS_GRID}", flush=True)
    print(f"[z304] out={out_path}", flush=True)

    sebas_rows = load_sebas_params()
    z91f, cfg, M1, M2, sd_M1, sd_M2, forward_2t = build_models_once()
    print(f"[z304] models built  ({time.time() - t0:.1f}s)", flush=True)

    # Cache curves per branch
    curves_per_branch = {vg1: load_curves(vg1_filter=vg1) for vg1 in vg1_list}
    for vg1, cs in curves_per_branch.items():
        print(f"[z304] branch V_G1={vg1}: {len(cs)} curves", flush=True)

    rows = []
    n_cells = len(vg1_list) * len(bf_list) * len(ALPHA0_GRID) * len(RS_GRID)
    cell_i = 0
    for vg1 in vg1_list:
        curves = curves_per_branch[vg1]
        for bf in bf_list:
            for alpha0 in ALPHA0_GRID:
                for rs in RS_GRID:
                    cell_i += 1
                    t_cell = time.time()
                    r = evaluate_cell(
                        vg1=vg1, bf=bf, alpha0=alpha0, rs=rs,
                        curves=curves, sebas_rows=sebas_rows,
                        z91f_mod=z91f, cfg=cfg, M1=M1, M2=M2,
                        sd_M1=sd_M1, sd_M2=sd_M2, forward_2t=forward_2t,
                    )
                    rows.append(r)
                    elapsed = time.time() - t_cell
                    print(f"[z304] cell {cell_i}/{n_cells}: vg1={vg1} bf={bf} "
                          f"a0={alpha0:.0e} Rs={rs} → med={r['median_log_rmse']:.3f} "
                          f"signed={r['signed_dec_median']:+.3f} "
                          f"({elapsed:.1f}s, total {time.time()-t0:.0f}s)",
                          flush=True)

    summary = {
        "script": "z304_sebas_three_branch_refit",
        "vg1_list": vg1_list, "bf_grid": bf_list,
        "alpha0_grid": ALPHA0_GRID, "rs_grid": RS_GRID,
        "elapsed_s": time.time() - t0,
        "device": str(DEVICE),
        "rows": rows,
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\n[z304] wrote {out_path}  ({time.time()-t0:.0f}s total)", flush=True)


if __name__ == "__main__":
    sys.exit(main() or 0)
