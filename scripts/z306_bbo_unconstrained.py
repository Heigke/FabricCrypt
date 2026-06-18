"""z306 — Unconstrained BBO co-optimization on current pyport topology.

O51 falsification test: if a wide BBO can drive cell-wide median forward
log-RMSE < 0.5 dec across the 33 Sebas IV curves WITHOUT changing topology,
then the "topology rebuild mandatory" claim is overturned and v4.4 path
re-opens.

Pre-registered gates (locked from 01:22 cron O51 corrective):
  FALSIFY    : cell-wide median log-RMSE < 0.5 dec
                AND |signed median| <= 0.10 dec
                → topology rebuild NOT mandatory
  AMBITIOUS  : same AND worst-branch median < 0.7 dec
  FAIL       : cell-wide > 0.7 dec → topology-rebuild hypothesis stands

Free parameters (~18): Bf, per-branch K1/BETA0/NFACTOR/ETAB, alpha0,
and Rs(V_G1) as logistic transition (Rlo, Rhi, V0, k).

Budget: scipy DE pop=64 maxiter=80 -> ~5000 evals. Each eval evaluates
all 33 curves under SA1-like injection, with the free parameters varied.
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
from scipy.optimize import differential_evolution

torch.set_default_dtype(torch.float64)

_ENV_ROOT = os.environ.get("NSRAM_REPO_ROOT")
if _ENV_ROOT:
    ROOT = Path(_ENV_ROOT)
elif Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy").exists():
    ROOT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
else:
    ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data/sebas_2026_04_22"
OUT_DIR = ROOT / "results/z306_bbo_unconstrained"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64

RS_FALLBACK = 1.0e30
THERMAL_TRIP_C = 85.0  # pause if APU >= this
THERMAL_RESUME_C = 60.0


VG1_DIRS = {
    0.2: "2vHCa-2 I-Vs@VG2 VG1=0.2 vnwell=2",
    0.4: "2vHCa-2 I-Vs@VG2 VG1=0.4 vnwell=2",
    0.6: "2vHCa-2 I-Vs@VG2 VG1=0.6 vnwell=2",
}
VG2_RE = re.compile(r"VG2=(-?\d+\.\d+)")


# ---- Parameter vector layout ----
# Index : name                 : transform        : bounds (in DE space)
PARAM_SPEC = [
    ("log10_Bf",         (math.log10(50),     math.log10(10000))),  # log10
    ("K1_v02",           (0.3, 0.8)),
    ("K1_v04",           (0.3, 0.8)),
    ("K1_v06",           (0.2, 0.6)),
    ("BETA0_v02",        (5.0, 25.0)),
    ("BETA0_v04",        (10.0, 30.0)),
    ("BETA0_v06",        (10.0, 30.0)),
    ("NFACTOR_v02",      (1.0, 15.0)),
    ("NFACTOR_v04",      (1.0, 15.0)),
    ("NFACTOR_v06",      (1.0, 15.0)),
    ("ETAB_v02",         (0.5, 3.0)),
    ("ETAB_v04",         (0.5, 3.0)),
    ("ETAB_v06",         (0.5, 3.0)),
    ("log10_alpha0",     (-6.0, -2.0)),                # log10
    ("Rlo",              (0.0, 1.0e3)),
    ("log10_Rhi",        (7.0, 12.0)),                 # log10
    ("V0",               (0.1, 0.5)),
    ("k_logistic",       (0.02, 0.2)),
]
PARAM_NAMES = [s[0] for s in PARAM_SPEC]
BOUNDS = [s[1] for s in PARAM_SPEC]


def unpack_x(x):
    d = dict(zip(PARAM_NAMES, x))
    Bf = 10.0 ** d["log10_Bf"]
    alpha0 = 10.0 ** d["log10_alpha0"]
    Rhi = 10.0 ** d["log10_Rhi"]
    Rlo = d["Rlo"]
    V0 = d["V0"]
    k = d["k_logistic"]
    per_vg1 = {
        0.2: {"K1": d["K1_v02"], "BETA0": d["BETA0_v02"],
              "NFACTOR": d["NFACTOR_v02"], "ETAB": d["ETAB_v02"]},
        0.4: {"K1": d["K1_v04"], "BETA0": d["BETA0_v04"],
              "NFACTOR": d["NFACTOR_v04"], "ETAB": d["ETAB_v04"]},
        0.6: {"K1": d["K1_v06"], "BETA0": d["BETA0_v06"],
              "NFACTOR": d["NFACTOR_v06"], "ETAB": d["ETAB_v06"]},
    }
    return Bf, alpha0, Rlo, Rhi, V0, k, per_vg1


def rs_of_vg1(vg1, Rlo, Rhi, V0, k):
    """Logistic: Rs(V_G1) interpolates between Rlo (low V_G1) and Rhi (high V_G1)."""
    # sigma(z) where z = (V_G1 - V0) / k
    z = (vg1 - V0) / max(k, 1e-6)
    s = 1.0 / (1.0 + math.exp(-z))
    return Rlo + (Rhi - Rlo) * s


# Per-branch mbjt step (V_G1=0.2 has BJT effectively off; 0.4/0.6 on)
MBJT_PER_VG1 = {0.2: 0.001, 0.4: 1.0, 0.6: 1.0}


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


def load_curves(subsample_per_branch=None):
    curves = []
    for vg1, subdir in VG1_DIRS.items():
        d = DATA / subdir
        files = sorted(d.glob("StandardIV*.csv"))
        if subsample_per_branch is not None and len(files) > subsample_per_branch:
            idx_sub = np.linspace(0, len(files) - 1, subsample_per_branch).astype(int)
            files = [files[i] for i in idx_sub]
        for csv_path in files:
            m = VG2_RE.search(csv_path.name)
            if not m:
                continue
            vg2 = float(m.group(1))
            try:
                arr = np.loadtxt(csv_path, delimiter=",", skiprows=1, usecols=(0, 1))
            except Exception:
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
            idx = np.linspace(0, len(Vd) - 1, 15).astype(int)
            Vd, Id = Vd[idx], Id[idx]
            curves.append({
                "VG1": vg1, "VG2": vg2, "file": csv_path.name,
                "Vd": torch.tensor(Vd, dtype=DTYPE),
                "Id": torch.tensor(Id, dtype=DTYPE),
            })
    return curves


def thermal_wait():
    try:
        t = int(open("/sys/class/thermal/thermal_zone0/temp").read().strip()) / 1000.0
    except Exception:
        return
    if t < THERMAL_TRIP_C:
        return
    print(f"[z306] thermal pause: APU={t:.1f}C, waiting until {THERMAL_RESUME_C}C", flush=True)
    t_start = time.time()
    while time.time() - t_start < 120:
        time.sleep(2)
        try:
            t = int(open("/sys/class/thermal/thermal_zone0/temp").read().strip()) / 1000.0
        except Exception:
            return
        if t < THERMAL_RESUME_C:
            return


class Evaluator:
    """Stateful evaluator: builds models once, then evaluates param vectors."""

    def __init__(self, subsample_per_branch=None):
        from nsram.bsim4_port.model_card import BSIM4Model
        from nsram.bsim4_port.nsram_cell_2T import NSRAMCell2TConfig, forward_2t
        from nsram.bsim4_port.temp import compute_size_dep
        from nsram.bsim4_port.geometry import Geometry
        from nsram.bsim4_port.bjt import GummelPoonNPN

        z91f = _load_module("z91f", ROOT / "scripts/z91f_validate_with_sebas_params.py")
        self.z91f = z91f
        self.GummelPoonNPN = GummelPoonNPN

        text_M1 = (DATA / "M1_130DNWFB.txt").read_text()
        M1 = BSIM4Model.from_spice(text_M1, model_type="nmos")
        z91f.patch_model_values(M1, type_n=True)

        text_M2 = (DATA / "M2_130bulkNSRAM.txt").read_text()
        M2 = BSIM4Model.from_spice(text_M2, model_type="nmos")
        z91f.patch_model_values(M2, type_n=True)

        cfg = NSRAMCell2TConfig(use_iii=True, use_gidl=True, use_bjt=True,
                                newton_max_iters=25)
        sd_M1 = compute_size_dep(M1, Geometry(L=cfg.Ln, W=cfg.Wn), T_C=cfg.T_C)
        sd_M2 = compute_size_dep(M2,
                                  Geometry(L=cfg.Ln * cfg.M2_length_factor,
                                           W=cfg.Wn), T_C=cfg.T_C)
        cfg._sd_M1 = sd_M1
        cfg._sd_M2 = sd_M2

        self.cfg = cfg
        self.M1 = M1
        self.M2 = M2
        self.sd_M1 = sd_M1
        self.sd_M2 = sd_M2
        self.forward_2t = forward_2t

        self.sebas_rows = load_sebas_params()
        self.curves_full = load_curves(subsample_per_branch=None)
        self.curves = load_curves(subsample_per_branch=subsample_per_branch)
        print(f"[z306] loaded {len(self.curves)} curves for DE  "
              f"(full set has {len(self.curves_full)})", flush=True)

        # group curves by VG1 for per-branch reporting
        self.curves_by_vg1 = {0.2: [], 0.4: [], 0.6: []}
        for c in self.curves:
            self.curves_by_vg1[round(c["VG1"], 2)].append(c)

        self.n_evals = 0
        self.history = []  # list of (n_evals, best_so_far)
        self.best = (float("inf"), None)
        self.t0 = time.time()

    def evaluate_curve(self, c, Bf, alpha0, Rlo, Rhi, V0, k_log, per_vg1):
        vg1 = round(c["VG1"], 2)
        sa1 = per_vg1[vg1]
        sebas_row = find_params(self.sebas_rows, c["VG1"], c["VG2"])
        if sebas_row is None:
            return None

        rs = rs_of_vg1(vg1, Rlo, Rhi, V0, k_log)
        self.cfg.vnwell_Rs = rs if rs > 0 else RS_FALLBACK
        if hasattr(self.cfg, "invalidate"):
            self.cfg.invalidate()

        P_M1 = {
            "etab":   torch.tensor(sa1["ETAB"],   dtype=DTYPE),
            "k1":     torch.tensor(sa1["K1"],     dtype=DTYPE),
            "alpha0": torch.tensor(alpha0,        dtype=DTYPE),
            "beta0":  torch.tensor(sa1["BETA0"],  dtype=DTYPE),
        }
        P_M2 = {"nfactor": torch.tensor(sa1["NFACTOR"], dtype=DTYPE)}
        for kk, vv in self.z91f.M2_STATIC_OVERRIDES.items():
            if kk not in P_M2:
                P_M2[kk] = torch.tensor(float(vv), dtype=DTYPE)

        bjt = self.GummelPoonNPN.from_sebas_card()
        if not math.isnan(sebas_row.get("IS", float("nan"))):
            bjt.Is = float(sebas_row["IS"])
        area = float(sebas_row.get("area", 1e-6))
        if math.isnan(area):
            area = 1e-6
        bjt.area = area * MBJT_PER_VG1[vg1]
        bjt.Bf = float(Bf)

        try:
            with torch.no_grad(), \
                  patch_sd_scaled(self.sd_M1, P_M1), \
                  patch_sd_scaled(self.sd_M2, P_M2):
                out = self.forward_2t(self.cfg, self.M1, bjt,
                                       c["Vd"], torch.tensor(c["VG1"]),
                                       torch.tensor(c["VG2"]),
                                       warm_start=True, use_homotopy=True)
            Id_pred = out["Id"].abs()
            conv = torch.tensor([bool(x) for x in out["converged"]])
        except Exception:
            return float("inf"), float("nan")

        log_eps = 1e-15
        log_p = torch.log10(Id_pred + log_eps)
        log_m = torch.log10(c["Id"] + log_eps)
        if conv.any():
            mask = conv
            diff = (log_p[mask] - log_m[mask])
            rmse = float(torch.sqrt((diff ** 2).mean()))
            signed = float(torch.median(diff))
            return rmse, signed
        return float("inf"), float("nan")

    def __call__(self, x):
        self.n_evals += 1
        if self.n_evals % 100 == 0:
            thermal_wait()
        Bf, alpha0, Rlo, Rhi, V0, k_log, per_vg1 = unpack_x(x)
        per_branch_rmses = {0.2: [], 0.4: [], 0.6: []}
        all_rmses = []
        all_signs = []
        for c in self.curves:
            res = self.evaluate_curve(c, Bf, alpha0, Rlo, Rhi, V0, k_log, per_vg1)
            if res is None:
                continue
            rmse, signed = res
            if math.isfinite(rmse):
                per_branch_rmses[round(c["VG1"], 2)].append(rmse)
                all_rmses.append(rmse)
                if math.isfinite(signed):
                    all_signs.append(signed)

        if not all_rmses:
            return 100.0  # heavy penalty

        cellwide_med = float(np.median(all_rmses))
        cellwide_signed = float(np.median(all_signs)) if all_signs else float("nan")
        # Worst branch median
        branch_meds = []
        for vg1, lst in per_branch_rmses.items():
            if lst:
                branch_meds.append(float(np.median(lst)))
        worst = max(branch_meds) if branch_meds else float("inf")

        # Objective: cell-wide median + soft penalty for worst > 1.5 dec
        obj = cellwide_med + max(0.0, worst - 1.5) * 0.5
        # Also light signed-bias penalty (so FALSIFY |signed|<=0.1 is incentivised)
        if math.isfinite(cellwide_signed):
            obj += max(0.0, abs(cellwide_signed) - 0.10) * 0.3

        if obj < self.best[0]:
            self.best = (obj, {
                "x": list(x),
                "cellwide_median_log_rmse": cellwide_med,
                "cellwide_signed_dec_median": cellwide_signed,
                "worst_branch_median": worst,
                "branch_medians": {str(k): v for k, v in zip(per_branch_rmses.keys(), branch_meds)},
                "n_evals": self.n_evals,
            })
            elapsed = time.time() - self.t0
            print(f"[z306] eval={self.n_evals:5d} t={elapsed:6.1f}s  "
                  f"obj={obj:.4f}  med={cellwide_med:.3f}  signed={cellwide_signed:+.3f}  "
                  f"worst={worst:.3f}", flush=True)

        if self.n_evals % 50 == 0:
            self.history.append({
                "n_evals": self.n_evals,
                "best_obj": self.best[0],
                "best_med": self.best[1]["cellwide_median_log_rmse"] if self.best[1] else None,
            })

        return obj


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--popsize", type=int, default=64)
    p.add_argument("--maxiter", type=int, default=80)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tol", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=1)  # GPU contention -> 1
    p.add_argument("--subsample", type=int, default=4,
                    help="Curves per V_G1 branch used during DE (default 4 = 12 total). "
                         "Final eval always uses full 33-curve set.")
    args = p.parse_args()

    t0 = time.time()
    print(f"[z306] device={DEVICE}  start {time.strftime('%H:%M:%S')}", flush=True)
    print(f"[z306] DE pop={args.popsize} maxiter={args.maxiter} seed={args.seed}", flush=True)
    print(f"[z306] {len(PARAM_NAMES)} free params: {PARAM_NAMES}", flush=True)

    ev = Evaluator(subsample_per_branch=args.subsample)

    result = differential_evolution(
        ev,
        bounds=BOUNDS,
        popsize=args.popsize,
        maxiter=args.maxiter,
        tol=args.tol,
        seed=args.seed,
        workers=args.workers,
        polish=False,
        updating="deferred",
        mutation=(0.5, 1.0),
        recombination=0.7,
        init="sobol",
        disp=True,
    )

    elapsed = time.time() - t0
    best_x = result.x
    Bf, alpha0, Rlo, Rhi, V0, k_log, per_vg1 = unpack_x(best_x)

    # Re-evaluate cleanly to get full per-branch report
    final_per_branch = {0.2: [], 0.4: [], 0.6: []}
    final_per_branch_signs = {0.2: [], 0.4: [], 0.6: []}
    all_rmses = []
    all_signs = []
    per_curve_details = []
    for c in ev.curves_full:
        res = ev.evaluate_curve(c, Bf, alpha0, Rlo, Rhi, V0, k_log, per_vg1)
        if res is None:
            continue
        rmse, signed = res
        vg1 = round(c["VG1"], 2)
        per_curve_details.append({"VG1": vg1, "VG2": c["VG2"], "file": c["file"],
                                    "log_rmse": rmse, "signed_dec": signed})
        if math.isfinite(rmse):
            final_per_branch[vg1].append(rmse)
            all_rmses.append(rmse)
        if math.isfinite(signed):
            final_per_branch_signs[vg1].append(signed)
            all_signs.append(signed)

    cellwide_med = float(np.median(all_rmses)) if all_rmses else float("inf")
    cellwide_signed = float(np.median(all_signs)) if all_signs else float("nan")
    per_branch_summary = {}
    for vg1, lst in final_per_branch.items():
        signs = final_per_branch_signs[vg1]
        per_branch_summary[str(vg1)] = {
            "n_curves": len(lst),
            "median_log_rmse": float(np.median(lst)) if lst else float("inf"),
            "p90_log_rmse": float(np.percentile(lst, 90)) if lst else float("inf"),
            "signed_dec_median": float(np.median(signs)) if signs else float("nan"),
        }
    worst_branch = max(v["median_log_rmse"] for v in per_branch_summary.values())

    # Verdict
    falsify = (cellwide_med < 0.5) and (abs(cellwide_signed) <= 0.10
                                         if math.isfinite(cellwide_signed) else False)
    ambitious = falsify and (worst_branch < 0.7)
    fail = cellwide_med > 0.7
    verdict = ("AMBITIOUS_PASS" if ambitious
               else "FALSIFY_PASS" if falsify
               else "FAIL" if fail
               else "INCONCLUSIVE")

    # Rs(V_G1) shape
    rs_curve = {f"V_G1={v}": rs_of_vg1(v, Rlo, Rhi, V0, k_log)
                for v in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8]}

    summary = {
        "script": "z306_bbo_unconstrained",
        "elapsed_s": elapsed,
        "n_evals_total": ev.n_evals,
        "param_names": PARAM_NAMES,
        "best_x_raw": list(best_x),
        "best_params": {
            "Bf": Bf,
            "alpha0": alpha0,
            "Rlo": Rlo, "Rhi": Rhi, "V0": V0, "k_logistic": k_log,
            "per_vg1": {str(k): v for k, v in per_vg1.items()},
        },
        "rs_logistic_curve": rs_curve,
        "cellwide_median_log_rmse": cellwide_med,
        "cellwide_signed_dec_median": cellwide_signed,
        "worst_branch_median": worst_branch,
        "per_branch": per_branch_summary,
        "per_curve": per_curve_details,
        "gates": {
            "FALSIFY_threshold_med": 0.5,
            "FALSIFY_threshold_signed": 0.10,
            "AMBITIOUS_threshold_worst": 0.7,
            "FAIL_threshold_med": 0.7,
        },
        "verdict": verdict,
        "de_result": {
            "fun": float(result.fun),
            "nit": int(result.nit),
            "nfev": int(result.nfev),
            "success": bool(result.success),
            "message": str(result.message),
        },
        "convergence_history": ev.history,
    }
    out_path = OUT_DIR / "summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=float)

    print(f"\n[z306] === FINAL ({elapsed:.0f}s, {ev.n_evals} evals) ===", flush=True)
    print(f"[z306] cellwide median log-RMSE = {cellwide_med:.3f} dec", flush=True)
    print(f"[z306] cellwide signed median  = {cellwide_signed:+.3f} dec", flush=True)
    print(f"[z306] worst-branch median     = {worst_branch:.3f} dec", flush=True)
    for vg1, s in per_branch_summary.items():
        print(f"[z306]   V_G1={vg1}: med={s['median_log_rmse']:.3f}  "
              f"signed={s['signed_dec_median']:+.3f}  n={s['n_curves']}", flush=True)
    print(f"[z306] Rs logistic: Rlo={Rlo:.2e}  Rhi={Rhi:.2e}  V0={V0:.3f}  k={k_log:.3f}", flush=True)
    print(f"[z306] Rs(0.2)={rs_curve['V_G1=0.2']:.2e}  Rs(0.4)={rs_curve['V_G1=0.4']:.2e}  "
          f"Rs(0.6)={rs_curve['V_G1=0.6']:.2e}", flush=True)
    print(f"[z306] VERDICT: {verdict}", flush=True)
    print(f"[z306] wrote {out_path}", flush=True)


if __name__ == "__main__":
    sys.exit(main() or 0)
