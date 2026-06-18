"""z317 — Snapback peak law sweep on CURRENT pyport (z304 baseline).

Tests V_peak(V_G2) = 2.73 - 0.625*V_G2 law (P2-extracted from slide 15) at
V_G1=0.4. Sweeps V_G2 ∈ {0.05, 0.10, 0.15, 0.20, 0.30, 0.45}, with V_d in
[0.5, 3.5] @ 0.05V resolution (=61 pts). Per-cell V_peak = argmax(I_d).

Also runs a rate-probe: at V_G2=0.20, three V_d-step resolutions to proxy
slew-rate; check ~+0.15 V/dec V_peak shift (P2 finding).

Output: results/z317_snapback_law/summary.json

Gates:
  PASS-conservative : ≥4 of 6 V_G2 points within 0.3 V of law (allow shift)
  AMBITIOUS         : ≥5 of 6 within 0.2 V
  BONUS             : linear fit slope ∈ [-0.7, -0.55]
"""
from __future__ import annotations
import os
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "4")

import csv
import importlib.util
import json
import math
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

torch.set_default_dtype(torch.float64)

ROOT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
DATA = ROOT / "data/sebas_2026_04_22"
OUT_DIR = ROOT / "results/z317_snapback_law"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64

# ---- baseline (z304 canonical) ----
VG1 = 0.4
BF = 500
ALPHA0 = 1e-4
VG2_LIST = [0.05, 0.10, 0.15, 0.20, 0.30, 0.45]

# Snapback V_d sweep
VD_MIN, VD_MAX = 0.5, 3.5
VD_STEP = 0.05  # 61 points

# Rate-probe (proxy via V_d step size; coarser step ~ faster slew)
RATE_STEPS = [0.10, 0.05, 0.025]  # 31, 61, 121 pts → ~0.5 dec coverage

# Law (P2, at V_G1=0.3, trise=200µs)
LAW_INTERCEPT = 2.73
LAW_SLOPE = -0.625


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


def _load_module(name, path):
    sp = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(mod)
    return mod


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


def find_params(rows, vg1, vg2, atol=1e-3):
    best = None
    bestd = 1e9
    for r in rows:
        if abs(r["VG1"] - vg1) < atol:
            d = abs(r["VG2"] - vg2)
            if d < bestd:
                bestd = d; best = r
    return best, bestd


def make_row_overrides(sebas_row, alpha0_override, M2_STATIC):
    if sebas_row is None:
        return None, None
    P_M1 = {}
    if not math.isnan(sebas_row.get("ETAB", float("nan"))):
        P_M1["etab"] = torch.tensor(sebas_row["ETAB"], dtype=DTYPE)
    if not math.isnan(sebas_row.get("K1", float("nan"))):
        P_M1["k1"] = torch.tensor(sebas_row["K1"], dtype=DTYPE)
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


def build_models_once():
    z91f = _load_module("z91f",
                         ROOT / "scripts/z91f_validate_with_sebas_params.py")
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


def sweep_vpeak(*, vg1, vg2, vd_grid, sebas_rows, z91f, cfg, M1, sd_M1, sd_M2,
                 forward_2t):
    """Run forward_2t over V_d grid and return V_peak, I_peak, full curve."""
    from nsram.bsim4_port.bjt import GummelPoonNPN

    # use closest Sebas row (V_G2=0.45 may not exist → use closest)
    row, dist = find_params(sebas_rows, vg1, vg2)
    if row is None or math.isnan(row.get("K1", float("nan"))):
        return {"err": "no_row", "vg2_requested": vg2}

    P_M1, P_M2 = make_row_overrides(row, ALPHA0, z91f.M2_STATIC_OVERRIDES)
    bjt = GummelPoonNPN.from_sebas_card()
    if not math.isnan(row.get("IS", float("nan"))):
        bjt.Is = float(row["IS"])
    area = float(row.get("area", 1e-6))
    if math.isnan(area): area = 1e-6
    mbjt = float(row.get("mbjt", 1.0))
    if math.isnan(mbjt): mbjt = 1.0
    bjt.area = area * mbjt
    bjt.Bf = float(BF)

    Vd = torch.tensor(vd_grid, dtype=DTYPE)
    try:
        with torch.no_grad(), \
              patch_sd_scaled(sd_M1, P_M1), \
              patch_sd_scaled(sd_M2, P_M2):
            out = forward_2t(cfg, M1, bjt,
                              Vd, torch.tensor(vg1), torch.tensor(vg2),
                              warm_start=True, use_homotopy=True,
                              dense_vd_in_snapback=True,
                              snapback_vd_threshold=1.4,
                              snapback_vd_step=0.025)
    except Exception as e:
        return {"err": str(e)[:200], "vg2_requested": vg2,
                "row_vg2": row["VG2"], "row_dist": dist}

    Id = out["Id"].abs().cpu().numpy()
    conv = np.array([bool(x) for x in out["converged"]])
    vd_np = Vd.cpu().numpy()
    if conv.any():
        idx_conv = np.where(conv)[0]
        ipk = int(idx_conv[np.argmax(Id[idx_conv])])
    else:
        ipk = int(np.argmax(Id))
    v_peak = float(vd_np[ipk])
    i_peak = float(Id[ipk])
    return {
        "vg2_requested": vg2, "row_vg2": float(row["VG2"]),
        "row_dist": float(dist),
        "v_peak": v_peak, "i_peak": i_peak,
        "n_conv": int(conv.sum()), "n_total": int(len(vd_np)),
        "Vd": vd_np.tolist(), "Id": Id.tolist(),
        "converged": conv.tolist(),
    }


def main():
    t0 = time.time()
    print(f"[z317] device={DEVICE} start", flush=True)
    sebas_rows = load_sebas_params()
    z91f, cfg, M1, M2, sd_M1, sd_M2, forward_2t = build_models_once()
    print(f"[z317] models built ({time.time()-t0:.1f}s)", flush=True)

    # --- Main sweep ---
    vd_grid = np.round(np.arange(VD_MIN, VD_MAX + 1e-9, VD_STEP), 4).tolist()
    print(f"[z317] V_d grid: {len(vd_grid)} pts [{vd_grid[0]:.2f},{vd_grid[-1]:.2f}]",
          flush=True)

    results = []
    for vg2 in VG2_LIST:
        t_cell = time.time()
        r = sweep_vpeak(vg1=VG1, vg2=vg2, vd_grid=vd_grid,
                         sebas_rows=sebas_rows, z91f=z91f, cfg=cfg,
                         M1=M1, sd_M1=sd_M1, sd_M2=sd_M2,
                         forward_2t=forward_2t)
        results.append(r)
        if "err" in r:
            print(f"[z317] V_G2={vg2}: ERR {r['err']}", flush=True)
        else:
            vp_law = LAW_INTERCEPT + LAW_SLOPE * vg2
            dist = r["v_peak"] - vp_law
            print(f"[z317] V_G2={vg2}: V_peak={r['v_peak']:.3f} "
                  f"law={vp_law:.3f} Δ={dist:+.3f} "
                  f"(row V_G2={r['row_vg2']:.2f} d={r['row_dist']:.3f}, "
                  f"{time.time()-t_cell:.1f}s)", flush=True)

    # --- Rate-probe ---
    print(f"\n[z317] rate-probe @ V_G2=0.20:", flush=True)
    rate_probe = []
    for step in RATE_STEPS:
        grid = np.round(np.arange(VD_MIN, VD_MAX + 1e-9, step), 5).tolist()
        r = sweep_vpeak(vg1=VG1, vg2=0.20, vd_grid=grid,
                         sebas_rows=sebas_rows, z91f=z91f, cfg=cfg,
                         M1=M1, sd_M1=sd_M1, sd_M2=sd_M2,
                         forward_2t=forward_2t)
        # don't store full Vd/Id for rate probe — too big
        keep = {k: v for k, v in r.items() if k not in ("Vd", "Id", "converged")}
        keep["vd_step"] = step
        keep["n_pts"] = len(grid)
        rate_probe.append(keep)
        if "err" not in r:
            print(f"[z317] step={step}: V_peak={r['v_peak']:.3f} "
                  f"({len(grid)} pts)", flush=True)

    # --- Gate evaluation ---
    valid = [(r["vg2_requested"], r["v_peak"]) for r in results if "err" not in r]
    vp_arr = np.array([v for _, v in valid])
    vg2_arr = np.array([g for g, _ in valid])
    law_arr = LAW_INTERCEPT + LAW_SLOPE * vg2_arr
    raw_delta = vp_arr - law_arr
    # V_G1-shifted: allow constant offset (best-fit intercept shift)
    shift = float(np.median(raw_delta))
    shifted_delta = raw_delta - shift

    n_within_03 = int(np.sum(np.abs(shifted_delta) <= 0.3))
    n_within_02 = int(np.sum(np.abs(shifted_delta) <= 0.2))
    pass_conservative = n_within_03 >= 4
    pass_ambitious = n_within_02 >= 5

    # Linear fit slope
    if len(vp_arr) >= 2:
        slope, intercept = np.polyfit(vg2_arr, vp_arr, 1)
        slope = float(slope); intercept = float(intercept)
    else:
        slope = float("nan"); intercept = float("nan")
    bonus_slope = (slope >= -0.7) and (slope <= -0.55)

    # Rate-probe V_peak shifts (should be +0.15V per decade slewer)
    rp_valid = [r for r in rate_probe if "err" not in r]
    rp_shifts = []
    if len(rp_valid) >= 2:
        for i in range(1, len(rp_valid)):
            dv = rp_valid[i]["v_peak"] - rp_valid[0]["v_peak"]
            # decade ratio: coarser step is faster (~higher slew). Take ratio of steps.
            ratio = rp_valid[0]["vd_step"] / rp_valid[i]["vd_step"]
            decades = math.log10(ratio) if ratio > 0 else 0
            per_dec = dv / decades if decades != 0 else 0
            rp_shifts.append({"step": rp_valid[i]["vd_step"],
                               "v_peak": rp_valid[i]["v_peak"],
                               "dv_vs_first": dv,
                               "decades_vs_first": decades,
                               "shift_per_decade": per_dec})

    summary = {
        "script": "z317_snapback_peak_law",
        "elapsed_s": time.time() - t0,
        "device": str(DEVICE),
        "config": {"VG1": VG1, "BF": BF, "ALPHA0": ALPHA0,
                    "VG2_LIST": VG2_LIST,
                    "VD_MIN": VD_MIN, "VD_MAX": VD_MAX, "VD_STEP": VD_STEP,
                    "law_intercept": LAW_INTERCEPT, "law_slope": LAW_SLOPE},
        "results": results,
        "rate_probe": rate_probe,
        "rate_probe_shifts": rp_shifts,
        "analysis": {
            "v_peak_per_vg2": [{"vg2": float(g), "v_peak": float(v),
                                  "v_peak_law": float(LAW_INTERCEPT + LAW_SLOPE*g),
                                  "raw_delta": float(v - (LAW_INTERCEPT + LAW_SLOPE*g))}
                                 for g, v in valid],
            "vg1_shift_constant": shift,
            "shifted_deltas": shifted_delta.tolist(),
            "n_within_0.3_after_shift": n_within_03,
            "n_within_0.2_after_shift": n_within_02,
            "fit_slope": slope, "fit_intercept": intercept,
        },
        "gates": {
            "pass_conservative": bool(pass_conservative),
            "ambitious": bool(pass_ambitious),
            "bonus_slope": bool(bonus_slope),
            "verdict": ("AMBITIOUS" if pass_ambitious
                         else ("PASS" if pass_conservative else "FAIL")),
        },
    }
    out_path = OUT_DIR / "summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\n[z317] wrote {out_path} ({time.time()-t0:.0f}s)", flush=True)

    print("\n=== SUMMARY ===")
    print(f"V_G1=0.4, BF={BF}, ALPHA0={ALPHA0}")
    print(f"V_G1 shift constant (vs law @V_G1=0.3): {shift:+.3f} V")
    print(f"Fit slope: {slope:.3f} (law: {LAW_SLOPE})")
    print(f"Within 0.3 V (after shift): {n_within_03}/6")
    print(f"Within 0.2 V (after shift): {n_within_02}/6")
    print(f"Verdict: {summary['gates']['verdict']}, "
          f"bonus_slope={summary['gates']['bonus_slope']}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
