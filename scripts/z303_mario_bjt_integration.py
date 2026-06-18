"""z303 — Mario's published BJT params + V_G1-dependent BVPar.

Source of truth: data/nsram_zenodo/SimulationFiles/SPICE/dev/BJTparams.txt

  .param IsPar  = 1e-16
  .param BfPar  = 50
  .param VafPar = 40         (forward Early)
  .param NfPar  = 0.9
  .param NePar  = 1.5
  .param VarPar = 10         (reverse Early)
  .param BVPar  = 3.5 - 1.5*V_G1     ← V_G1-dependent BV
  .param nbvPar = 0  (Tsinghua var: 9 - 0.55/V_G — large)

We compare 3 configurations against the 33-row Sebas IV val set:

  baseline : Bf=9000, Va=0.55, no avalanche (current production default)
  da3      : Bf=3000, Va=0.55, no avalanche (z301 best, conservative gate already PASS)
  mario_only      : Mario BJT params (Bf=50, Vaf=40, Nf=0.9, Ne=1.5, Var=10, Is=1e-16),
                    NO avalanche
  mario_plus_bv   : Mario BJT params + BVPar(V_G1) = 3.5 - 1.5*V_G1 avalanche

Avalanche is applied as an EXTRA additive current to Id_eq AFTER the
Newton fixed-point converges (lumped-Vb), because the in-Newton path is
gated on `cfg.use_lateral_collector`, and the existing implementation
takes a SCALAR BV; we want a V_G1-dependent one.

Formula (mirrors nsram_cell_2T.py path C, lines 700-723):
   Vbc      = Vb_eq − Vd
   rev_mag  = clamp(−Vbc, 0, ∞)
   BV(V_G1) = max(3.5 − 1.5 * V_G1, 0.5)        # floor to avoid divide-by-0
   M_raw    = 1 + (rev_mag / BV) ** N_av
   sat      = sigmoid((BV_max − rev_mag) / δ)
   M_safe   = 1 + (M_raw − 1) * sat
   Ic_av    = (M_safe − 1) * Ids_M1            # ≈ Id_eq dominantly for sat region

We use N_av = 4 (default lat_N) and BV_max = 1.1 * BV.

Gates
-----
PASS-conservative : median fwd log-RMSE < 0.8 dec  (better than DA3's 0.99)
AMBITIOUS         : median fwd log-RMSE < 0.5 dec
SAFETY            : VG1=0.2 subthreshold improves OR stays within 0.3 dec of DA3
BONUS             : with BVPar(V_G1) active, snapback peak at V_G1=0.6 lands at
                    V_d = 2.6 ± 0.3 V
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
OUT = ROOT / "results/z303_mario_bjt"
OUT.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64
print(f"[z303] device={DEVICE} dtype={DTYPE}")

VG1_DIRS = {
    0.2: "2vHCa-2 I-Vs@VG2 VG1=0.2 vnwell=2",
    0.4: "2vHCa-2 I-Vs@VG2 VG1=0.4 vnwell=2",
    0.6: "2vHCa-2 I-Vs@VG2 VG1=0.6 vnwell=2",
}
VG2_RE = re.compile(r"VG2=(-?\d+\.\d+)")

VB_GRID = np.linspace(0.0, 0.80, 25)
SAFE = 1e-15

# Avalanche shaping constants (match path C defaults in nsram_cell_2T.py)
N_AV = 4.0
DELTA = 0.5
BV_FLOOR = 0.5


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


# --------------------------------------------------------------------------- #
# Vb equilibrium solver — returns Id_eq AND Vb_eq AND Ids_M1_eq per point.    #
# Mirror of z301 solve_vb_equilibrium but ALSO returns Vb_eq for avalanche.   #
# --------------------------------------------------------------------------- #
def solve_vb_equilibrium(mep7, cfg, M1, M2, bjt, VG1_np, VG2_np, Vd_np,
                          vb_grid=VB_GRID):
    N = len(VG1_np); G = len(vb_grid)
    VG1 = np.broadcast_to(VG1_np[:, None], (N, G)).reshape(-1)
    VG2 = np.broadcast_to(VG2_np[:, None], (N, G)).reshape(-1)
    Vd = np.broadcast_to(Vd_np[:, None], (N, G)).reshape(-1)
    Vb = np.broadcast_to(vb_grid[None, :], (N, G)).reshape(-1)
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
    Id = out["Id"].cpu().numpy().reshape(N, G)
    Inet = Iii - Ile
    Id_eq = np.zeros(N)
    Vb_eq = np.zeros(N)
    for n in range(N):
        s = Inet[n]
        sign = np.sign(s)
        idx_change = np.where(np.diff(sign) != 0)[0]
        if idx_change.size == 0:
            if s[0] < 0:
                Id_eq[n] = Id[n, 0]; Vb_eq[n] = vb_grid[0]
            else:
                Id_eq[n] = Id[n, -1]; Vb_eq[n] = vb_grid[-1]
        else:
            i = idx_change[0]
            y0, y1 = s[i], s[i + 1]
            t = (-y0 / (y1 - y0)) if y1 != y0 else 0.5
            Id_eq[n] = Id[n, i] + t * (Id[n, i + 1] - Id[n, i])
            Vb_eq[n] = vb_grid[i] + t * (vb_grid[i + 1] - vb_grid[i])
    return Id_eq, Vb_eq


# --------------------------------------------------------------------------- #
# Avalanche extra current (post-hoc on top of Id_eq).                         #
# --------------------------------------------------------------------------- #
def avalanche_extra_current(Vd_np, VG1_np, Vb_eq_np, Id_eq_np):
    """M(Vbc) shaping with BV = max(3.5 − 1.5*V_G1, BV_FLOOR).

    Returns Ic_av extra current (added to Id), magnitude-preserving sign of Id.
    """
    Vbc = Vb_eq_np - Vd_np
    rev_mag = np.clip(-Vbc, 0.0, None)
    BV = np.maximum(3.5 - 1.5 * VG1_np, BV_FLOOR)
    BV_max = 1.1 * BV
    # smooth saturation (numpy sigmoid)
    sat = 1.0 / (1.0 + np.exp(-(BV_max - rev_mag) / DELTA))
    # safe pow
    ratio = rev_mag / BV
    M_raw = 1.0 + np.power(np.clip(ratio, 0.0, 10.0), N_AV)
    M_safe = 1.0 + (M_raw - 1.0) * sat
    Ic_av = (M_safe - 1.0) * np.abs(Id_eq_np)
    return Ic_av, BV


def score_curve(meas_vd, meas_id, sim_id):
    apex = int(np.argmax(meas_vd))
    fwd_meas = meas_id[:apex + 1]
    fwd_sim = sim_id[:apex + 1]
    fwd_log_err = np.abs(log10_safe(fwd_sim) - log10_safe(fwd_meas))
    signed = float(np.median(log10_safe(fwd_sim) - log10_safe(fwd_meas)))
    return {
        "forward_log_rmse": float(np.median(fwd_log_err)),
        "forward_signed_dec": signed,
    }


def build_models(mep7, ns4d, *, Bf, Va, Is, Nf=None, Ne=None, Var=None):
    """Build cfg/M1/M2/bjt for a particular BJT parametrization."""
    from nsram.bsim4_port.nsram_cell_2T import NSRAMCell2TConfig
    from nsram.bsim4_port.bjt import GummelPoonNPN
    v1 = _load_module("v1_z303", ROOT / "scripts/z96_narma10_pilot.py")
    cfg = NSRAMCell2TConfig(use_iii=True, use_gidl=True, use_bjt=True,
                              newton_max_iters=20)
    M1, M2 = v1.build_calibrated_models()
    bjt = GummelPoonNPN.from_sebas_card()
    bjt.Bf = Bf
    bjt.Va = Va
    bjt.Is = Is
    if Nf is not None:
        bjt.Nf = Nf
    if Ne is not None:
        bjt.Ne = Ne
    if Var is not None:
        bjt.Vb = Var  # Vb attr stores Var (reverse Early)
    return cfg, M1, M2, bjt


def evaluate_config(mep7, ns4d, label, *,
                     Bf, Va, Is, Nf=None, Ne=None, Var=None,
                     use_bv_avalanche=False,
                     all_vg1, all_vg2, all_vd, curve_meta, seg_idx):
    cfg, M1, M2, bjt = build_models(mep7, ns4d, Bf=Bf, Va=Va, Is=Is,
                                       Nf=Nf, Ne=Ne, Var=Var)
    Id_sim, Vb_eq = solve_vb_equilibrium(mep7, cfg, M1, M2, bjt,
                                            all_vg1, all_vg2, all_vd)
    if use_bv_avalanche:
        Ic_av, BV_arr = avalanche_extra_current(all_vd, all_vg1, Vb_eq, Id_sim)
        Id_total = Id_sim + Ic_av * np.sign(np.where(Id_sim == 0, 1, Id_sim))
    else:
        Id_total = Id_sim
        BV_arr = np.full_like(all_vd, np.nan)

    per_curve = []
    for cm, (s, e) in zip(curve_meta, seg_idx):
        sim_id = Id_total[s:e]
        sc = score_curve(cm["meas_vd"], cm["meas_id"], sim_id)
        sc["vg1"] = cm["vg1"]; sc["vg2"] = cm["vg2"]; sc["file"] = cm["file"]
        # Find sim snapback peak (max Id along forward sweep)
        meas_vd = cm["meas_vd"]; apex = int(np.argmax(meas_vd))
        fwd_vd = meas_vd[:apex + 1]
        fwd_sim = sim_id[:apex + 1]
        if len(fwd_sim) > 2:
            i_peak = int(np.argmax(np.abs(fwd_sim)))
            sc["sim_peak_vd"] = float(fwd_vd[i_peak])
            sc["sim_peak_id"] = float(fwd_sim[i_peak])
        per_curve.append(sc)
    fwd = np.array([c["forward_log_rmse"] for c in per_curve])
    signed = np.array([c["forward_signed_dec"] for c in per_curve])
    by_vg1 = {}
    for vg1 in (0.2, 0.4, 0.6):
        mask = np.array([c["vg1"] == vg1 for c in per_curve])
        if mask.any():
            # median sim peak Vd for this VG1 (BONUS check)
            peak_vds = [c.get("sim_peak_vd") for c in per_curve
                         if c["vg1"] == vg1 and c.get("sim_peak_vd") is not None]
            by_vg1[vg1] = {
                "median_fwd_log_rmse": float(np.median(fwd[mask])),
                "median_signed_dec":   float(np.median(signed[mask])),
                "n":                    int(mask.sum()),
                "median_sim_peak_vd":   float(np.median(peak_vds)) if peak_vds else None,
            }
    BV_for_vg06 = float(np.max(3.5 - 1.5 * 0.6))  # = 2.6 V
    return {
        "label": label,
        "Bf": Bf, "Va": Va, "Is": Is,
        "Nf": Nf, "Ne": Ne, "Var": Var,
        "use_bv_avalanche": use_bv_avalanche,
        "BV_at_VG1_0.6_V": BV_for_vg06,
        "median_fwd_log_rmse_all": float(np.median(fwd)),
        "median_signed_dec_all":   float(np.median(signed)),
        "by_vg1": by_vg1,
        "n_curves": len(per_curve),
        "per_curve": per_curve,
    }


def main():
    t0 = time.time()
    mep7 = _load_module("z294_z303", ROOT / "scripts/z294_mep7_gpu_pyport.py")
    ns4d = mep7._load_cpu_ref()
    print(f"[z303] loaded pyport. ns4d.OPT_BF={ns4d.OPT_BF} OPT_VA={ns4d.OPT_VA} OPT_IS={ns4d.OPT_IS}")

    # Load Sebas curves once
    curve_meta = []
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
                print(f"[z303] load fail {csv_path.name}: {e}")
                continue
            curve_meta.append({
                "vg1": vg1, "vg2": vg2, "file": csv_path.name,
                "meas_vd": meas_vd, "meas_id": meas_id, "meas_t": meas_t,
            })
    seg_idx = []; flat_vd = []; flat_vg1 = []; flat_vg2 = []
    for cm in curve_meta:
        s = len(flat_vd)
        flat_vd.extend(cm["meas_vd"].tolist())
        flat_vg1.extend([cm["vg1"]] * len(cm["meas_vd"]))
        flat_vg2.extend([cm["vg2"]] * len(cm["meas_vd"]))
        seg_idx.append((s, len(flat_vd)))
    all_vd  = np.asarray(flat_vd)
    all_vg1 = np.asarray(flat_vg1)
    all_vg2 = np.asarray(flat_vg2)
    print(f"[z303] {len(curve_meta)} curves, {len(all_vd)} total points")

    configs = []
    # Baseline (Bf=9000, Va=0.55, no avalanche)
    print("\n[z303] === baseline (Bf=9000) ===")
    r = evaluate_config(mep7, ns4d, "baseline",
                         Bf=9000.0, Va=0.55, Is=1e-9,
                         all_vg1=all_vg1, all_vg2=all_vg2, all_vd=all_vd,
                         curve_meta=curve_meta, seg_idx=seg_idx)
    configs.append(r); print_brief(r)

    print("\n[z303] === da3 (Bf=3000) ===")
    r = evaluate_config(mep7, ns4d, "da3",
                         Bf=3000.0, Va=0.55, Is=1e-9,
                         all_vg1=all_vg1, all_vg2=all_vg2, all_vd=all_vd,
                         curve_meta=curve_meta, seg_idx=seg_idx)
    configs.append(r); print_brief(r)

    print("\n[z303] === mario_only (Bf=50, Vaf=40, Nf=0.9, Ne=1.5, Var=10, Is=1e-16) ===")
    r = evaluate_config(mep7, ns4d, "mario_only",
                         Bf=50.0, Va=40.0, Is=1e-16,
                         Nf=0.9, Ne=1.5, Var=10.0,
                         all_vg1=all_vg1, all_vg2=all_vg2, all_vd=all_vd,
                         curve_meta=curve_meta, seg_idx=seg_idx)
    configs.append(r); print_brief(r)

    print("\n[z303] === mario_plus_bv (Mario params + BVPar(V_G1)) ===")
    r = evaluate_config(mep7, ns4d, "mario_plus_bv",
                         Bf=50.0, Va=40.0, Is=1e-16,
                         Nf=0.9, Ne=1.5, Var=10.0,
                         use_bv_avalanche=True,
                         all_vg1=all_vg1, all_vg2=all_vg2, all_vd=all_vd,
                         curve_meta=curve_meta, seg_idx=seg_idx)
    configs.append(r); print_brief(r)

    # Hybrid: Mario topology + DA3 Bf (keep Bf=3000 but Vaf/Nf/Ne/Var from Mario)
    print("\n[z303] === mario_plus_da3_bf (Bf=3000, Vaf=40, Nf=0.9, Ne=1.5, Var=10) ===")
    r = evaluate_config(mep7, ns4d, "mario_plus_da3_bf",
                         Bf=3000.0, Va=40.0, Is=1e-16,
                         Nf=0.9, Ne=1.5, Var=10.0,
                         use_bv_avalanche=True,
                         all_vg1=all_vg1, all_vg2=all_vg2, all_vd=all_vd,
                         curve_meta=curve_meta, seg_idx=seg_idx)
    configs.append(r); print_brief(r)

    # Determine gates relative to mario_only (the headline)
    mario_only = next(c for c in configs if c["label"] == "mario_only")
    mario_bv   = next(c for c in configs if c["label"] == "mario_plus_bv")
    da3        = next(c for c in configs if c["label"] == "da3")
    baseline   = next(c for c in configs if c["label"] == "baseline")

    da3_vg02 = da3["by_vg1"].get(0.2, {}).get("median_fwd_log_rmse", float("inf"))

    def make_gates(cfg_):
        m = cfg_["median_fwd_log_rmse_all"]
        vg02 = cfg_["by_vg1"].get(0.2, {}).get("median_fwd_log_rmse", float("inf"))
        return {
            "pass_conservative": bool(m < 0.8),
            "pass_ambitious":    bool(m < 0.5),
            "safety_pass_vg02":  bool((vg02 <= da3_vg02) or (vg02 <= da3_vg02 + 0.3)),
            "vg02_delta_vs_da3_dec": float(vg02 - da3_vg02),
        }

    # BONUS check: at V_G1=0.6 with mario_plus_bv, sim_peak_vd ≈ 2.6 ± 0.3 V
    peak_vd_vg06 = mario_bv["by_vg1"].get(0.6, {}).get("median_sim_peak_vd")
    bonus_pass = (peak_vd_vg06 is not None) and (2.3 <= peak_vd_vg06 <= 2.9)

    summary = {
        "script": "scripts/z303_mario_bjt_integration.py",
        "device": str(DEVICE),
        "n_curves": len(curve_meta),
        "mario_params": {
            "IsPar": 1e-16, "BfPar": 50, "VafPar": 40,
            "NfPar": 0.9, "NePar": 1.5, "VarPar": 10,
            "BVPar_VG1": "3.5 - 1.5*V_G1",
        },
        "comparison": {
            "baseline":          {"med": baseline["median_fwd_log_rmse_all"],
                                  "vg02": baseline["by_vg1"].get(0.2, {}).get("median_fwd_log_rmse"),
                                  "vg06": baseline["by_vg1"].get(0.6, {}).get("median_fwd_log_rmse")},
            "da3":               {"med": da3["median_fwd_log_rmse_all"],
                                  "vg02": da3["by_vg1"].get(0.2, {}).get("median_fwd_log_rmse"),
                                  "vg06": da3["by_vg1"].get(0.6, {}).get("median_fwd_log_rmse")},
            "mario_only":        {"med": mario_only["median_fwd_log_rmse_all"],
                                  "vg02": mario_only["by_vg1"].get(0.2, {}).get("median_fwd_log_rmse"),
                                  "vg06": mario_only["by_vg1"].get(0.6, {}).get("median_fwd_log_rmse")},
            "mario_plus_bv":     {"med": mario_bv["median_fwd_log_rmse_all"],
                                  "vg02": mario_bv["by_vg1"].get(0.2, {}).get("median_fwd_log_rmse"),
                                  "vg06": mario_bv["by_vg1"].get(0.6, {}).get("median_fwd_log_rmse"),
                                  "peak_vd_vg06": peak_vd_vg06},
        },
        "gates": {
            "mario_only":     make_gates(mario_only),
            "mario_plus_bv":  make_gates(mario_bv),
            "BV_at_VG1_0.6":  2.6,
            "BONUS_snapback_peak_vg06": {
                "expected_V": 2.6,
                "tolerance":  0.3,
                "observed_V": peak_vd_vg06,
                "pass":       bool(bonus_pass),
            },
        },
        "configs": configs,
        "runtime_sec": time.time() - t0,
    }
    with open(OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=lambda x: float(x) if hasattr(x, '__float__') else None)
    print(f"\n[z303] === RESULT ===")
    for c in configs:
        print(f"  {c['label']:22s}  med={c['median_fwd_log_rmse_all']:.3f}  "
              f"VG1=0.2={c['by_vg1'].get(0.2,{}).get('median_fwd_log_rmse',float('nan')):.3f}  "
              f"VG1=0.6={c['by_vg1'].get(0.6,{}).get('median_fwd_log_rmse',float('nan')):.3f}")
    print(f"\n[z303] BONUS  V_G1=0.6 snapback peak (mario_plus_bv): "
          f"{peak_vd_vg06}  expected 2.6 ± 0.3 V  pass={bonus_pass}")
    print(f"[z303] mario_only gates:    {summary['gates']['mario_only']}")
    print(f"[z303] mario_plus_bv gates: {summary['gates']['mario_plus_bv']}")
    print(f"[z303] wrote {OUT/'summary.json'}  runtime={summary['runtime_sec']:.1f}s")


def print_brief(r):
    b = r["by_vg1"]
    print(f"  {r['label']:22s} med={r['median_fwd_log_rmse_all']:.3f}  "
          f"signed={r['median_signed_dec_all']:+.3f}  "
          f"VG1=0.2={b.get(0.2,{}).get('median_fwd_log_rmse',float('nan')):.3f} "
          f"VG1=0.4={b.get(0.4,{}).get('median_fwd_log_rmse',float('nan')):.3f} "
          f"VG1=0.6={b.get(0.6,{}).get('median_fwd_log_rmse',float('nan')):.3f}  "
          f"peak_vd_vg06={b.get(0.6,{}).get('median_sim_peak_vd')}")


if __name__ == "__main__":
    main()
