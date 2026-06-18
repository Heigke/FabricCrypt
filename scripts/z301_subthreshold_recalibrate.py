"""z301 — Recalibrate Bf and voff to fix z298b's +1.67 dec subthreshold over-prediction.

z298b ran MEP-7 pyport against all 33 Sebas IV curves and found:
  - median forward log-RMSE = 1.668 dec
  - signed bias            = +1.668 dec (sim is 46x HIGH systematically)
  - worst at VG1=0.2 subthreshold (5+ dec)

Two knobs (the real-physics handles available in pyport):
  - Bf (BJT forward gain)              — directly scales BJT collector current
                                          which dominates when MOSFET M1 is sub-threshold
  - voff_shift (BSIM4 sub-threshold offset, M1+M2)
                                        — shifts MOSFET sub-threshold curve;
                                          MORE NEGATIVE => more leakage, MORE POSITIVE => less

The user phrased the 2nd knob as "Ioff_scale" but the BSIM4 card has no Ioff;
the physical analog is voff. We sweep voff_shift (positive => reduce leakage)
on a grid that conservatively maps {0.01,0.1,0.3,1.0} -> {0.30,0.15,0.05,0.0}
                                  (smaller = stronger pull-up = more attenuation).

Gates:
  PASS-conservative : median forward log-RMSE < 1.0 dec across 33 curves
  AMBITIOUS         : median forward log-RMSE < 0.5 dec
  SAFETY            : VG1=0.6 supra-threshold curves must not regress by > 0.3 dec
                      (relative to z298b baseline of ~0.65 dec on VG1=0.6)
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
OUT = ROOT / "results/z301_subthreshold_recal"
OUT.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64
print(f"[z301] device={DEVICE} dtype={DTYPE}")

# ─── Sweep grid ─────────────────────────────────────────────────────────────
BF_SWEEP        = [1.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 9000.0]
VOFF_SHIFT_SWEEP = [0.0, 0.05, 0.10, 0.15, 0.20]               # 0 = baseline
# Mapping rationale: each +0.05 V on voff (less negative) suppresses sub-Vt
# current by ~one decade (nfactor*kT/q ~ 0.05 V at room T for typical n).

VG1_DIRS = {
    0.2: "2vHCa-2 I-Vs@VG2 VG1=0.2 vnwell=2",
    0.4: "2vHCa-2 I-Vs@VG2 VG1=0.4 vnwell=2",
    0.6: "2vHCa-2 I-Vs@VG2 VG1=0.6 vnwell=2",
}
VG2_RE = re.compile(r"VG2=(-?\d+\.\d+)")

VB_GRID = np.linspace(0.0, 0.80, 25)
SAFE = 1e-15


# ─── load helpers ───────────────────────────────────────────────────────────
def _load_module(name, path):
    sp = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(mod)
    return mod


def load_one(csv_path: Path):
    arr = np.loadtxt(csv_path, delimiter=",", skiprows=1, usecols=(0, 1, 2))
    return arr[:, 0], arr[:, 1], arr[:, 2]


def log10_safe(x):
    return np.log10(np.maximum(np.abs(x), SAFE))


# ─── Vb equilibrium (vectorized over Bf/voff variants) ──────────────────────
def solve_vb_equilibrium(mep7, cfg, M1, M2, bjt,
                          VG1_np, VG2_np, Vd_np, vb_grid=VB_GRID):
    N = len(VG1_np); G = len(vb_grid)
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
        device=str(DEVICE), dtype=DTYPE,
    )
    Iii = out["Iii_in"].cpu().numpy().reshape(N, G)
    Ile = out["Ileak_out"].cpu().numpy().reshape(N, G)
    Id  = out["Id"].cpu().numpy().reshape(N, G)
    Inet = Iii - Ile
    Id_eq = np.zeros(N)
    for n in range(N):
        s = Inet[n]
        sign = np.sign(s)
        idx_change = np.where(np.diff(sign) != 0)[0]
        if idx_change.size == 0:
            Id_eq[n] = Id[n, 0] if s[0] < 0 else Id[n, -1]
        else:
            i = idx_change[0]
            y0, y1 = s[i], s[i + 1]
            t = (-y0 / (y1 - y0)) if y1 != y0 else 0.5
            Id_eq[n] = Id[n, i] + t * (Id[n, i + 1] - Id[n, i])
    return Id_eq


def score_curve(meas_vd, meas_id, sim_id):
    apex = int(np.argmax(meas_vd))
    fwd_meas = meas_id[:apex + 1]
    fwd_sim  = sim_id[:apex + 1]
    fwd_log_err = np.abs(log10_safe(fwd_sim) - log10_safe(fwd_meas))
    signed = float(np.median(log10_safe(fwd_sim) - log10_safe(fwd_meas)))
    return {
        "forward_log_rmse": float(np.median(fwd_log_err)),
        "forward_signed_dec": signed,
    }


# ─── per-config evaluation ──────────────────────────────────────────────────
def evaluate_point(mep7, ns4d, Bf, voff_shift, all_vg1, all_vg2, all_vd,
                   curve_meta, seg_idx):
    """Build models with given (Bf, voff_shift) and score 33 curves."""
    # Patch build_calibrated_models with voff_shift
    from nsram.bsim4_port.nsram_cell_2T import NSRAMCell2TConfig
    from nsram.bsim4_port.bjt import GummelPoonNPN
    v1 = _load_module("v1_z301", ROOT / "scripts/z96_narma10_pilot.py")
    cfg = NSRAMCell2TConfig(use_iii=True, use_gidl=True, use_bjt=True,
                             newton_max_iters=20)
    M1, M2 = v1.build_calibrated_models(voff_M1_shift=voff_shift,
                                          voff_M2_shift=voff_shift)
    bjt = GummelPoonNPN.from_sebas_card()
    bjt.Bf = Bf
    bjt.Va = ns4d.OPT_VA
    bjt.Is = ns4d.OPT_IS

    Id_sim = solve_vb_equilibrium(mep7, cfg, M1, M2, bjt,
                                    all_vg1, all_vg2, all_vd)
    per_curve = []
    for cm, (s, e) in zip(curve_meta, seg_idx):
        sim_id = Id_sim[s:e]
        sc = score_curve(cm["meas_vd"], cm["meas_id"], sim_id)
        sc["vg1"] = cm["vg1"]; sc["vg2"] = cm["vg2"]; sc["file"] = cm["file"]
        per_curve.append(sc)
    fwd = np.array([c["forward_log_rmse"] for c in per_curve])
    signed = np.array([c["forward_signed_dec"] for c in per_curve])
    # Stratify by VG1
    by_vg1 = {}
    for vg1 in (0.2, 0.4, 0.6):
        mask = np.array([c["vg1"] == vg1 for c in per_curve])
        if mask.any():
            by_vg1[vg1] = {
                "median_fwd_log_rmse": float(np.median(fwd[mask])),
                "median_signed_dec":   float(np.median(signed[mask])),
                "n":                    int(mask.sum()),
            }
    return {
        "Bf": Bf, "voff_shift": voff_shift,
        "median_fwd_log_rmse_all": float(np.median(fwd)),
        "median_signed_dec_all":   float(np.median(signed)),
        "by_vg1": by_vg1,
        "n_curves": len(per_curve),
    }


# ─── driver ─────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    mep7 = _load_module("z294", ROOT / "scripts/z294_mep7_gpu_pyport.py")
    ns4d = mep7._load_cpu_ref()
    print(f"[z301] loaded pyport. baseline Bf={ns4d.OPT_BF}, voff_shift=0.0")

    # Enumerate curves once
    curve_meta = []
    all_vg1 = []; all_vg2 = []; all_vd = []; seg_idx = []
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
                print(f"[z301] load fail {csv_path.name}: {e}")
                continue
            start = len(all_vd) and sum(len(a) for a in all_vd) or 0
            # easier: just rebuild flat arrays after
            curve_meta.append({
                "vg1": vg1, "vg2": vg2, "file": csv_path.name,
                "meas_vd": meas_vd, "meas_id": meas_id, "meas_t": meas_t,
            })
    # build flat arrays + seg idx
    seg_idx = []
    flat_vd = []; flat_vg1 = []; flat_vg2 = []
    for cm in curve_meta:
        s = len(flat_vd)
        flat_vd.extend(cm["meas_vd"].tolist())
        flat_vg1.extend([cm["vg1"]] * len(cm["meas_vd"]))
        flat_vg2.extend([cm["vg2"]] * len(cm["meas_vd"]))
        e = len(flat_vd)
        seg_idx.append((s, e))
    all_vd  = np.asarray(flat_vd)
    all_vg1 = np.asarray(flat_vg1)
    all_vg2 = np.asarray(flat_vg2)
    print(f"[z301] {len(curve_meta)} curves, {len(all_vd)} total points")

    # ── 1D sweeps ─────────────────────────────────────────────────────────
    print(f"\n[z301] === 1D sweep: Bf (voff_shift=0) ===")
    sweep_bf = []
    for Bf in BF_SWEEP:
        r = evaluate_point(mep7, ns4d, Bf, 0.0,
                           all_vg1, all_vg2, all_vd, curve_meta, seg_idx)
        sweep_bf.append(r)
        b = r["by_vg1"]
        print(f"  Bf={Bf:7.1f}  all={r['median_fwd_log_rmse_all']:.3f}  "
              f"signed={r['median_signed_dec_all']:+.3f}  "
              f"VG1=0.2:{b.get(0.2,{}).get('median_fwd_log_rmse',float('nan')):.3f} "
              f"VG1=0.6:{b.get(0.6,{}).get('median_fwd_log_rmse',float('nan')):.3f}")

    print(f"\n[z301] === 1D sweep: voff_shift (Bf=9000 baseline) ===")
    sweep_voff = []
    for vs in VOFF_SHIFT_SWEEP:
        r = evaluate_point(mep7, ns4d, 9000.0, vs,
                           all_vg1, all_vg2, all_vd, curve_meta, seg_idx)
        sweep_voff.append(r)
        b = r["by_vg1"]
        print(f"  voff_shift={vs:+.2f}  all={r['median_fwd_log_rmse_all']:.3f}  "
              f"signed={r['median_signed_dec_all']:+.3f}  "
              f"VG1=0.2:{b.get(0.2,{}).get('median_fwd_log_rmse',float('nan')):.3f} "
              f"VG1=0.6:{b.get(0.6,{}).get('median_fwd_log_rmse',float('nan')):.3f}")

    # ── 2D grid ───────────────────────────────────────────────────────────
    print(f"\n[z301] === 2D grid: Bf x voff_shift ({len(BF_SWEEP)*len(VOFF_SHIFT_SWEEP)} pts) ===")
    grid = []
    for Bf in BF_SWEEP:
        for vs in VOFF_SHIFT_SWEEP:
            r = evaluate_point(mep7, ns4d, Bf, vs,
                                all_vg1, all_vg2, all_vd, curve_meta, seg_idx)
            grid.append(r)
            b = r["by_vg1"]
            print(f"  Bf={Bf:7.1f} voff={vs:+.2f}  "
                  f"all={r['median_fwd_log_rmse_all']:.3f}  "
                  f"signed={r['median_signed_dec_all']:+.3f}  "
                  f"VG1=0.2:{b.get(0.2,{}).get('median_fwd_log_rmse',float('nan')):.3f} "
                  f"VG1=0.6:{b.get(0.6,{}).get('median_fwd_log_rmse',float('nan')):.3f}")

    # Baseline (Bf=9000, voff=0.0) is in grid; find it
    baseline = next(r for r in grid
                    if r["Bf"] == 9000.0 and r["voff_shift"] == 0.0)
    baseline_vg06 = baseline["by_vg1"].get(0.6, {}).get("median_fwd_log_rmse",
                                                          float("nan"))

    # Best point: min median_fwd_log_rmse_all subject to SAFETY
    SAFETY_THRESH = baseline_vg06 + 0.3
    candidates = []
    for r in grid:
        vg06 = r["by_vg1"].get(0.6, {}).get("median_fwd_log_rmse", float("inf"))
        r["vg06_regression_vs_baseline_dec"] = float(vg06 - baseline_vg06)
        r["safety_pass"] = bool(vg06 <= SAFETY_THRESH)
        if r["safety_pass"]:
            candidates.append(r)
    if not candidates:
        print("[z301] WARNING no points pass safety; falling back to global min.")
        candidates = list(grid)
    best = min(candidates, key=lambda r: r["median_fwd_log_rmse_all"])

    # Gates
    gate_conservative = bool(best["median_fwd_log_rmse_all"] < 1.0)
    gate_ambitious    = bool(best["median_fwd_log_rmse_all"] < 0.5)

    summary = {
        "script": "scripts/z301_subthreshold_recalibrate.py",
        "device": str(DEVICE),
        "n_curves": len(curve_meta),
        "vg1_set": sorted({c["vg1"] for c in curve_meta}),
        "bf_sweep": BF_SWEEP,
        "voff_shift_sweep": VOFF_SHIFT_SWEEP,
        "baseline_point": {
            "Bf": 9000.0, "voff_shift": 0.0,
            "median_fwd_log_rmse_all": baseline["median_fwd_log_rmse_all"],
            "median_signed_dec_all":   baseline["median_signed_dec_all"],
            "VG1=0.6_median_fwd_log_rmse": baseline_vg06,
            "VG1=0.2_median_fwd_log_rmse":
                baseline["by_vg1"].get(0.2, {}).get("median_fwd_log_rmse"),
        },
        "safety_threshold_vg06_dec": float(SAFETY_THRESH),
        "best_point": best,
        "sweep_bf_1d": sweep_bf,
        "sweep_voff_1d": sweep_voff,
        "grid_2d": grid,
        "gates": {
            "pass_conservative": gate_conservative,
            "pass_ambitious":    gate_ambitious,
            "safety_pass":       bool(best["safety_pass"]),
        },
        "runtime_sec": time.time() - t0,
    }
    with open(OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[z301] === RESULT ===")
    print(f"[z301] baseline (Bf=9000,voff=0):  all={baseline['median_fwd_log_rmse_all']:.3f}  "
          f"VG1=0.2={baseline['by_vg1'].get(0.2,{}).get('median_fwd_log_rmse',float('nan')):.3f} "
          f"VG1=0.6={baseline_vg06:.3f}")
    bp = best
    print(f"[z301] BEST point: Bf={bp['Bf']}, voff_shift={bp['voff_shift']:+.2f}")
    print(f"[z301]   median_fwd_log_rmse_all = {bp['median_fwd_log_rmse_all']:.3f} dec  "
          f"(was {baseline['median_fwd_log_rmse_all']:.3f})")
    print(f"[z301]   signed bias              = {bp['median_signed_dec_all']:+.3f} dec  "
          f"(was {baseline['median_signed_dec_all']:+.3f})")
    print(f"[z301]   VG1=0.2 subthreshold     = "
          f"{bp['by_vg1'].get(0.2,{}).get('median_fwd_log_rmse',float('nan')):.3f} dec  "
          f"(was {baseline['by_vg1'].get(0.2,{}).get('median_fwd_log_rmse',float('nan')):.3f})")
    print(f"[z301]   VG1=0.6 supra-threshold  = "
          f"{bp['by_vg1'].get(0.6,{}).get('median_fwd_log_rmse',float('nan')):.3f} dec  "
          f"(was {baseline_vg06:.3f})  regression={bp['vg06_regression_vs_baseline_dec']:+.3f}")
    print(f"[z301]   GATES: conservative<1.0={gate_conservative}  "
          f"ambitious<0.5={gate_ambitious}  safety={bp['safety_pass']}")
    print(f"[z301] wrote {OUT/'summary.json'}  runtime={summary['runtime_sec']:.1f}s")


if __name__ == "__main__":
    main()
