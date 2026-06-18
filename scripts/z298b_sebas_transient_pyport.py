"""z298b — Sebas quasi-static replay using MEP-7 GPU pyport directly.

Replaces z298's z271 surrogate (V_d ∈ [0.5,2.5], V_G2 ∈ [0,0.45]) with the
MEP-7 GPU pyport solver, which handles arbitrary (V_G1, V_G2, V_d, V_b)
natively — no edge-extrap clamping.

Quasi-static approximation: ramp rate ≈ 0.17 V/s is slow, so at each
measurement (V_d, t) we solve for V_b equilibrium (where Iii_in = Ileak_out)
via Vb-grid sweep + bracket + linear interp, then evaluate I_d there.

Gates (unchanged from z298):
    PASS-conservative : median forward log-RMSE < 0.5 dec
    AMBITIOUS         : same on reverse + hysteresis spread within 2x
"""
from __future__ import annotations
import os
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "4")

import importlib.util
import json
import re
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
DATA_ROOT = ROOT / "data/sebas_2026_04_22"
OUT = ROOT / "results/z298b_sebas_pyport"
OUT.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64
print(f"[z298b] device={DEVICE} dtype={DTYPE}")


# -------- load z294 MEP-7 module ----------------------------------------------
def _load_mep7():
    sp = importlib.util.spec_from_file_location(
        "z294", ROOT / "scripts/z294_mep7_gpu_pyport.py")
    mod = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(mod)
    return mod


# -------- Sebas CSV loading ---------------------------------------------------
VG1_DIRS = {
    0.2: "2vHCa-2 I-Vs@VG2 VG1=0.2 vnwell=2",
    0.4: "2vHCa-2 I-Vs@VG2 VG1=0.4 vnwell=2",
    0.6: "2vHCa-2 I-Vs@VG2 VG1=0.6 vnwell=2",
}
VG2_RE = re.compile(r"VG2=(-?\d+\.\d+)")


def load_one(csv_path: Path):
    arr = np.loadtxt(csv_path, delimiter=",", skiprows=1,
                     usecols=(0, 1, 2))
    return arr[:, 0], arr[:, 1], arr[:, 2]  # vd, id, t


# -------- quasi-static Vb equilibrium solver (batched GPU) --------------------
# Vb grid for bracket search
VB_GRID = np.linspace(0.0, 0.80, 25)  # 25 pts


def solve_vb_equilibrium_batched(mep7, cfg, M1, M2, bjt,
                                  VG1_np, VG2_np, Vd_np,
                                  vb_grid=VB_GRID):
    """For each point, find Vb such that Iii_in - Ileak_out = 0.

    Returns I_d at that equilibrium. All arrays length N.
    """
    N = len(VG1_np)
    G = len(vb_grid)
    # Broadcast to (N, G)
    VG1 = np.broadcast_to(VG1_np[:, None], (N, G)).reshape(-1)
    VG2 = np.broadcast_to(VG2_np[:, None], (N, G)).reshape(-1)
    Vd  = np.broadcast_to(Vd_np[:, None],  (N, G)).reshape(-1)
    Vb  = np.broadcast_to(vb_grid[None, :], (N, G)).reshape(-1)

    out = mep7.solve_batched_gpu(
        cfg, M1, M2, bjt,
        torch.tensor(Vd, dtype=DTYPE),
        torch.tensor(VG1, dtype=DTYPE),
        torch.tensor(VG2, dtype=DTYPE),
        torch.tensor(Vb, dtype=DTYPE),
        max_iters=cfg.newton_max_iters,
        device=str(DEVICE),
        dtype=DTYPE,
    )
    Iii = out["Iii_in"].cpu().numpy().reshape(N, G)
    Ile = out["Ileak_out"].cpu().numpy().reshape(N, G)
    Id  = out["Id"].cpu().numpy().reshape(N, G)
    Inet = Iii - Ile  # (N,G): positive => Vb rises, negative => Vb falls
    # Find Vb where Inet=0 via sign change, else clamp.
    Vb_eq = np.zeros(N)
    Id_eq = np.zeros(N)
    for n in range(N):
        s = Inet[n]
        # find first index where sign changes from + to - (rising→falling)
        sign = np.sign(s)
        # locate any sign change
        idx_change = np.where(np.diff(sign) != 0)[0]
        if idx_change.size == 0:
            # No crossing: pick boundary with smaller |Inet|
            if s[0] < 0:
                # already negative at Vb=0: equilibrium is Vb=0 (clamped)
                Vb_eq[n] = vb_grid[0]
                Id_eq[n] = Id[n, 0]
            else:
                # always positive: equilibrium beyond grid → clamp at top
                Vb_eq[n] = vb_grid[-1]
                Id_eq[n] = Id[n, -1]
        else:
            i = idx_change[0]
            x0, x1 = vb_grid[i], vb_grid[i + 1]
            y0, y1 = s[i], s[i + 1]
            # linear interp for Vb_eq
            if y1 != y0:
                t = -y0 / (y1 - y0)
            else:
                t = 0.5
            Vb_eq[n] = x0 + t * (x1 - x0)
            # linear interp Id between bracket
            Id_eq[n] = Id[n, i] + t * (Id[n, i + 1] - Id[n, i])
    return Id_eq, Vb_eq


# -------- scoring -------------------------------------------------------------
SAFE = 1e-15


def log10_safe(x):
    return np.log10(np.maximum(np.abs(x), SAFE))


def midpoint_current(vd, idd, target=1.0):
    if vd[0] > vd[-1]:
        vd = vd[::-1]; idd = idd[::-1]
    if target < vd.min() or target > vd.max():
        return None
    return float(np.interp(target, vd, np.abs(idd)))


def score_curve(meas_vd, meas_id, sim_id):
    apex = int(np.argmax(meas_vd))
    fwd_meas_vd = meas_vd[:apex + 1]
    fwd_meas_id = meas_id[:apex + 1]
    fwd_sim_id  = sim_id[:apex + 1]
    rev_meas_vd = meas_vd[apex:]
    rev_meas_id = meas_id[apex:]
    rev_sim_id  = sim_id[apex:]

    fwd_log_err = np.abs(log10_safe(fwd_sim_id) - log10_safe(fwd_meas_id))
    rev_log_err = np.abs(log10_safe(rev_sim_id) - log10_safe(rev_meas_id))

    fwd_med = float(np.median(fwd_log_err))
    rev_med = float(np.median(rev_log_err))

    fwd_mid_m = midpoint_current(fwd_meas_vd, fwd_meas_id, 1.0)
    rev_mid_m = midpoint_current(rev_meas_vd, rev_meas_id, 1.0)
    fwd_mid_s = midpoint_current(fwd_meas_vd, fwd_sim_id, 1.0)
    rev_mid_s = midpoint_current(rev_meas_vd, rev_sim_id, 1.0)
    hyst_meas = (abs(rev_mid_m - fwd_mid_m) / fwd_mid_m
                 if fwd_mid_m and rev_mid_m and fwd_mid_m > 0 else None)
    hyst_sim  = (abs(rev_mid_s - fwd_mid_s) / fwd_mid_s
                 if fwd_mid_s and rev_mid_s and fwd_mid_s > 0 else None)
    return {
        "apex_idx": apex,
        "n_points": int(meas_vd.shape[0]),
        "forward_log_rmse": fwd_med,
        "reverse_log_rmse": rev_med,
        "hysteresis_ratio_meas": hyst_meas,
        "hysteresis_ratio_sim":  hyst_sim,
    }


# -------- driver --------------------------------------------------------------
def main():
    t0 = time.time()
    mep7 = _load_mep7()
    ns4d = mep7._load_cpu_ref()
    cfg, M1, M2, bjt = ns4d._build_pyport_models()
    print(f"[z298b] loaded MEP-7 pyport. newton_max_iters={cfg.newton_max_iters}")

    # Step 1: enumerate all curves and concatenate per-point arrays
    curve_meta = []  # one per curve
    all_vg1 = []
    all_vg2 = []
    all_vd = []
    seg_idx = []  # (start, end) per curve

    for vg1, subdir in VG1_DIRS.items():
        d = DATA_ROOT / subdir
        for csv_path in sorted(d.glob("StandardIV*.csv")):
            m = VG2_RE.search(csv_path.name)
            if not m:
                continue
            vg2 = float(m.group(1))
            try:
                meas_vd, meas_id, meas_t = load_one(csv_path)
            except Exception as e:
                print(f"[z298b] load fail {csv_path.name}: {e}")
                continue
            start = len(all_vd)
            all_vd.append(meas_vd)
            all_vg1.append(np.full_like(meas_vd, vg1))
            all_vg2.append(np.full_like(meas_vd, vg2))
            end = start + len(meas_vd)
            seg_idx.append((start, end))
            curve_meta.append({
                "vg1": vg1, "vg2": vg2, "file": csv_path.name,
                "meas_vd": meas_vd, "meas_id": meas_id, "meas_t": meas_t,
                "t_total": float(meas_t[-1] - meas_t[0]),
            })

    all_vd  = np.concatenate(all_vd)
    all_vg1 = np.concatenate(all_vg1)
    all_vg2 = np.concatenate(all_vg2)
    print(f"[z298b] {len(curve_meta)} curves, {len(all_vd)} total points")
    print(f"[z298b]   Vd range:  [{all_vd.min():.3f}, {all_vd.max():.3f}]")
    print(f"[z298b]   VG1 range: [{all_vg1.min():.3f}, {all_vg1.max():.3f}]")
    print(f"[z298b]   VG2 range: [{all_vg2.min():.3f}, {all_vg2.max():.3f}]")

    # Step 2: batched Vb-equilibrium solve
    t_solve = time.time()
    Id_sim, Vb_eq = solve_vb_equilibrium_batched(
        mep7, cfg, M1, M2, bjt, all_vg1, all_vg2, all_vd
    )
    print(f"[z298b] solve wall: {time.time() - t_solve:.1f}s "
          f"({1000 * (time.time() - t_solve) / len(all_vd):.1f} ms/pt)")

    # Step 3: score each curve
    curves = []
    raw_traces = []
    for cm, (s, e) in zip(curve_meta, seg_idx):
        sim_id = Id_sim[s:e]
        res = score_curve(cm["meas_vd"], cm["meas_id"], sim_id)
        res["vg1"] = cm["vg1"]; res["vg2"] = cm["vg2"]
        res["file"] = cm["file"]
        res["t_total"] = cm["t_total"]
        res["ramp_rate_Vps"] = float(
            2.0 / max(cm["meas_t"][int(np.argmax(cm["meas_vd"]))]
                      - cm["meas_t"][0], 1e-9))
        curves.append(res)
        raw_traces.append({
            "vg1": cm["vg1"], "vg2": cm["vg2"], "file": cm["file"],
            "t":  cm["meas_t"].tolist(),
            "vd": cm["meas_vd"].tolist(),
            "id_meas": cm["meas_id"].tolist(),
            "id_sim":  sim_id.tolist(),
            "vb_sim":  Vb_eq[s:e].tolist(),
            "fwd_rmse": res["forward_log_rmse"],
            "rev_rmse": res["reverse_log_rmse"],
        })

    # aggregate
    fwd = np.array([c["forward_log_rmse"] for c in curves])
    rev = np.array([c["reverse_log_rmse"] for c in curves])
    hyst_m = np.array([c["hysteresis_ratio_meas"] for c in curves
                       if c["hysteresis_ratio_meas"] is not None])
    hyst_s = np.array([c["hysteresis_ratio_sim"]  for c in curves
                       if c["hysteresis_ratio_sim"]  is not None])

    med_fwd = float(np.median(fwd))
    med_rev = float(np.median(rev))
    med_hyst_m = float(np.median(hyst_m)) if hyst_m.size else None
    med_hyst_s = float(np.median(hyst_s)) if hyst_s.size else None

    hyst_within_2x = None
    if med_hyst_m and med_hyst_s and med_hyst_m > 0 and med_hyst_s > 0:
        r = med_hyst_s / med_hyst_m
        hyst_within_2x = bool(0.5 <= r <= 2.0)

    pass_conservative = med_fwd < 0.5
    pass_ambitious = pass_conservative and (med_rev < 0.5) and bool(hyst_within_2x)

    raw_traces_sorted = sorted(raw_traces, key=lambda r: r["fwd_rmse"])
    top3_best = [{"vg1": r["vg1"], "vg2": r["vg2"],
                  "fwd_rmse": r["fwd_rmse"], "rev_rmse": r["rev_rmse"]}
                 for r in raw_traces_sorted[:3]]
    top3_worst = [{"vg1": r["vg1"], "vg2": r["vg2"],
                   "fwd_rmse": r["fwd_rmse"], "rev_rmse": r["rev_rmse"]}
                  for r in raw_traces_sorted[-3:]]

    sign_dec = []
    for tr in raw_traces:
        meas_id = np.asarray(tr["id_meas"])
        sim_id  = np.asarray(tr["id_sim"])
        apex = int(np.argmax(tr["vd"]))
        d = log10_safe(sim_id[:apex + 1]) - log10_safe(meas_id[:apex + 1])
        sign_dec.append(float(np.median(d)))
    median_signed_bias_dec = float(np.median(sign_dec))

    summary = {
        "script": "scripts/z298b_sebas_transient_pyport.py",
        "device": str(DEVICE),
        "n_curves": len(curves),
        "vg1_set": sorted({c["vg1"] for c in curves}),
        "vg2_min": float(min(c["vg2"] for c in curves)),
        "vg2_max": float(max(c["vg2"] for c in curves)),
        "vb_grid_n": len(VB_GRID),
        "vb_grid_range": [float(VB_GRID[0]), float(VB_GRID[-1])],
        "aggregate": {
            "median_forward_log_rmse_dec": med_fwd,
            "median_reverse_log_rmse_dec": med_rev,
            "median_hysteresis_meas":      med_hyst_m,
            "median_hysteresis_sim":       med_hyst_s,
            "hysteresis_spread_within_2x": hyst_within_2x,
            "median_signed_bias_dec_forward": median_signed_bias_dec,
        },
        "gate_conservative_pass": bool(pass_conservative),
        "gate_ambitious_pass":    bool(pass_ambitious),
        "per_curve": curves,
        "top3_best": top3_best,
        "top3_worst": top3_worst,
        "runtime_sec": time.time() - t0,
    }

    with open(OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[z298b] wrote {OUT/'summary.json'}")
    print(f"[z298b] verdict conservative={pass_conservative} "
          f"ambitious={pass_ambitious}")
    print(f"[z298b] med_fwd={med_fwd:.3f} dec, med_rev={med_rev:.3f} dec, "
          f"bias={median_signed_bias_dec:+.3f} dec")
    print(f"[z298b] hyst meas={med_hyst_m}  sim={med_hyst_s}  "
          f"within2x={hyst_within_2x}")
    print(f"[z298b] top3 best:  {top3_best}")
    print(f"[z298b] top3 worst: {top3_worst}")
    print(f"[z298b] runtime {summary['runtime_sec']:.1f} s")


if __name__ == "__main__":
    main()
