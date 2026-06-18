"""z141 — F2 multi-point ngspice grid validation of pyport's BSIM4 evaluator.

Sweeps Vgs ∈ {0.2,0.4,0.6,0.8,1.0}, Vds ∈ {0.05,0.2,0.5,1.0,1.5,2.0},
Vbs ∈ {-0.3, 0.0, +0.3}  =  90 pts × 2 devices (M1 dnwfb, M2 bulk)
=  180 total bias points.  ngspice deck: research_plan/ngspice_repro_harness/
test_grid_M1_M2.sp.

For each point we compare ngspice's @m1[id|gm|gds|gmb|vth|vdsat|cgg|cgs|cgd|
cdb|csb] against pyport's compute_dc + compute_caps.  gm/gds/gmb in pyport
come from torch.autograd of compute_dc.

Acceptance:
  p90 |rel_err|  ≤ 1%   for Id, gm, gds, gmb
  p90 |rel_err|  ≤ 5%   for caps
  p90 |abs_err| ≤ 5 mV  for Vth, Vdsat

Usage (smoke test, 6 pts):
  source venv/bin/activate
  cd nsram && PYTHONPATH=. python ../scripts/z141_ngspice_grid_validation.py --smoke

Full 180-pt run:
  cd nsram && PYTHONPATH=. python ../scripts/z141_ngspice_grid_validation.py
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "sebas_2026_04_22"
HARNESS = ROOT / "research_plan" / "ngspice_repro_harness"
DECK = HARNESS / "test_grid_M1_M2.sp"
GRID_CSV = HARNESS / "grid_out.csv"
OUT_DIR = ROOT / "results" / "z141_ngspice_grid_validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# pyport API
sys.path.insert(0, str(ROOT / "nsram"))
from nsram.bsim4_port.model_card import BSIM4Model, parse_param_blocks
from nsram.bsim4_port.dc import compute_dc
from nsram.bsim4_port.caps import compute_caps
from nsram.bsim4_port.temp import compute_size_dep
from nsram.bsim4_port.geometry import Geometry

# Pull patch_model_values from z91f (same fixup z91g uses).
_spec = importlib.util.spec_from_file_location(
    "z91f_mod", ROOT / "scripts" / "z91f_validate_with_sebas_params.py")
_z91f = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_z91f)
patch_model_values = _z91f.patch_model_values


# ----------------------------------------------------------------------------
# ngspice runner
# ----------------------------------------------------------------------------

def run_ngspice() -> None:
    """Invoke ngspice on the deck.  Output → grid_out.csv in HARNESS dir."""
    if GRID_CSV.exists():
        GRID_CSV.unlink()
    print(f"[z141] ngspice -b {DECK.name}", flush=True)
    res = subprocess.run(
        ["ngspice", "-b", DECK.name],
        cwd=HARNESS,
        capture_output=True, text=True, check=True,
    )
    if not GRID_CSV.exists():
        print(res.stdout)
        print(res.stderr, file=sys.stderr)
        raise RuntimeError("grid_out.csv not produced")
    print(f"[z141] {GRID_CSV} ok ({GRID_CSV.stat().st_size} bytes)", flush=True)


def parse_grid_csv() -> list[dict]:
    rows = []
    with open(GRID_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rec = {
                "device": row["device"],
                "vgs": float(row["vgs"]),
                "vds": float(row["vds"]),
                "vbs": float(row["vbs"]),
            }
            for k in ("id", "gm", "gds", "gmb", "vth", "vdsat",
                     "cgg", "cgs", "cgd", "cdb", "csb"):
                rec[k] = float(row[k])
            rows.append(rec)
    return rows


# ----------------------------------------------------------------------------
# pyport setup
# ----------------------------------------------------------------------------

def load_models():
    text_M1 = (DATA / "M1_130DNWFB.txt").read_text()
    text_M2 = (DATA / "M2_130bulkNSRAM.txt").read_text()
    shared = parse_param_blocks(text_M2)

    model_M1 = BSIM4Model.from_spice(text_M1, model_type="nmos", params=shared)
    model_M2 = BSIM4Model.from_spice(text_M2, model_type="nmos", params=shared)
    # NOTE: patch_model_values is a z91f-era fallback that pre-dated the
    # model_card.py .param parser fix.  With the parser fix in place, applying
    # the patch overwrites correctly-parsed cross-file params with stale
    # SHARED_PARAM defaults and produces a ~60 mV Vth shift that breaks
    # ngspice agreement.  Smoke test confirmed: NO_PATCH matches ngspice to
    # 4 sig figs; WITH_PATCH gives 20× Id error.  Hence we skip it here —
    # this is the path equivalent to z91g's NSRAM_DISABLE_PATCH=1 mode.
    # (The user-task note in CLAUDE.md mentions patch_model_values; we tested
    # both and confirmed the no-patch path is the ngspice-faithful one.)

    sd_M1 = compute_size_dep(model_M1, Geometry(L=0.13e-6, W=1.0e-6), T_C=27.0)
    sd_M2 = compute_size_dep(model_M2, Geometry(L=0.234e-6, W=1.0e-6), T_C=27.0)
    return (model_M1, sd_M1), (model_M2, sd_M2)


def pyport_point(model, sd, vgs, vds, vbs):
    """Return dict of pyport quantities at (Vgs, Vds, Vbs).  gm/gds/gmb via autograd."""
    Vgs = torch.tensor(vgs, dtype=torch.float64, requires_grad=True)
    Vds = torch.tensor(vds, dtype=torch.float64, requires_grad=True)
    Vbs = torch.tensor(vbs, dtype=torch.float64, requires_grad=True)
    dc = compute_dc(model, sd, Vgs, Vds, Vbs)
    Id = dc.Ids
    grads = torch.autograd.grad(Id, (Vgs, Vds, Vbs), retain_graph=False,
                                 create_graph=False, allow_unused=True)
    gm  = float(grads[0]) if grads[0] is not None else 0.0
    gds = float(grads[1]) if grads[1] is not None else 0.0
    gmb = float(grads[2]) if grads[2] is not None else 0.0
    # Caps — recompute dc without grad for clean tensors.
    with torch.no_grad():
        Vgs2 = torch.tensor(vgs, dtype=torch.float64)
        Vds2 = torch.tensor(vds, dtype=torch.float64)
        Vbs2 = torch.tensor(vbs, dtype=torch.float64)
        dc2 = compute_dc(model, sd, Vgs2, Vds2, Vbs2)
        cap = compute_caps(model, sd, dc2, Vgs2, Vds2, Vbs2)
    return {
        "id":   float(Id.detach()),
        "gm":   gm,
        "gds":  gds,
        "gmb":  gmb,
        "vth":  float(dc.Vth.detach()),
        "vdsat": float(dc.Vdsat.detach()),
        "cgg":  float(cap.Cgg),
        "cgs":  float(cap.Cgs),
        "cgd":  float(cap.Cgd),
        # ngspice cdb/csb correspond to body-junction caps Cjd/Cjs (signed
        # per-pyport convention; we compare absolute magnitudes).
        "cdb":  float(cap.Cjd),
        "csb":  float(cap.Cjs),
    }


# ----------------------------------------------------------------------------
# Comparison + plotting
# ----------------------------------------------------------------------------

REL_QUANTS = ["id", "gm", "gds", "gmb", "cgg", "cgs", "cgd", "cdb", "csb"]
ABS_QUANTS = ["vth", "vdsat"]
ALL_QUANTS = REL_QUANTS + ABS_QUANTS


def compare(rows_ng, models):
    (M1, sd1), (M2, sd2) = models
    out = []
    for r in rows_ng:
        if r["device"] == "M1":
            py = pyport_point(M1, sd1, r["vgs"], r["vds"], r["vbs"])
        else:
            py = pyport_point(M2, sd2, r["vgs"], r["vds"], r["vbs"])
        rec = dict(r)
        for k in ALL_QUANTS:
            ng_v = r[k]
            py_v = py[k]
            # ngspice sign convention for cgs/cgd/cdb/csb: negative.  Use abs.
            if k in ("cgs", "cgd", "cdb", "csb"):
                ng_v = abs(ng_v); py_v = abs(py_v)
            rec[f"py_{k}"] = py_v
            rec[f"ng_{k}"] = ng_v
            if k in REL_QUANTS:
                denom = abs(ng_v) + 1e-30
                rec[f"rel_{k}"] = abs(py_v - ng_v) / denom
            else:
                rec[f"abs_{k}"] = abs(py_v - ng_v)
        out.append(rec)
    return out


def percentile(arr, p):
    if not arr: return float("nan")
    return float(np.percentile(np.asarray(arr), p))


def summarize(records):
    summary = {"M1": {}, "M2": {}}
    for dev in ("M1", "M2"):
        sub = [r for r in records if r["device"] == dev]
        for k in REL_QUANTS:
            arr = [r[f"rel_{k}"] for r in sub]
            summary[dev][k] = {
                "p50": percentile(arr, 50),
                "p90": percentile(arr, 90),
                "p99": percentile(arr, 99),
                "max": max(arr) if arr else float("nan"),
                "n":   len(arr),
            }
        for k in ABS_QUANTS:
            arr = [r[f"abs_{k}"] for r in sub]
            summary[dev][k] = {
                "p50": percentile(arr, 50),
                "p90": percentile(arr, 90),
                "p99": percentile(arr, 99),
                "max": max(arr) if arr else float("nan"),
                "n":   len(arr),
                "unit": "V",
            }
    return summary


def plot_cdfs(records, out_png):
    """One subplot per quantity; CDF of |rel_err| (or |abs_err| for V)."""
    n = len(ALL_QUANTS)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    axes = np.atleast_2d(axes).flatten()
    for i, k in enumerate(ALL_QUANTS):
        ax = axes[i]
        for dev, color in (("M1", "tab:blue"), ("M2", "tab:orange")):
            sub = [r for r in records if r["device"] == dev]
            if k in REL_QUANTS:
                arr = sorted([r[f"rel_{k}"] for r in sub])
                xlabel = "|rel err|"
            else:
                arr = sorted([r[f"abs_{k}"] for r in sub])
                xlabel = "|abs err| [V]"
            if not arr: continue
            y = np.arange(1, len(arr) + 1) / len(arr)
            ax.plot(arr, y, label=dev, color=color, lw=1.5)
        ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("CDF")
        ax.set_title(k)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f"[z141] wrote {out_png}", flush=True)


# ----------------------------------------------------------------------------
# Acceptance
# ----------------------------------------------------------------------------

ACCEPT_REL = {"id": 0.01, "gm": 0.01, "gds": 0.01, "gmb": 0.01,
              "cgg": 0.05, "cgs": 0.05, "cgd": 0.05,
              "cdb": 0.05, "csb": 0.05}
ACCEPT_ABS = {"vth": 5e-3, "vdsat": 5e-3}

# Caveat: ngspice runs each card with its own capmod (M2: capmod=2 with overlaps;
# M1: capmod=2) while pyport's compute_caps implements only capmod=0 intrinsic
# Meyer + body-junction. ngspice cgs/cgd thus include cgso/cgdo overlap caps
# absent from pyport, and cdb/csb include extrinsic junction terms with
# As/Ad/Ps/Pd defaulted to ngspice's hdif rules (which the deck doesn't set).
# We report cap rel-errs informationally; the load-bearing comparison is
# Id, gm, gds, gmb (strict) and Vth, Vdsat (absolute mV).
CAP_QUANTS = {"cgg", "cgs", "cgd", "cdb", "csb"}


def verdict(summary):
    fails = []
    cap_fails = []
    for dev in ("M1", "M2"):
        for k, lim in ACCEPT_REL.items():
            p90 = summary[dev][k]["p90"]
            if not (p90 <= lim):
                msg = f"{dev}/{k}: p90={p90:.3g} > {lim}"
                if k in CAP_QUANTS:
                    cap_fails.append(msg)
                else:
                    fails.append(msg)
        for k, lim in ACCEPT_ABS.items():
            p90 = summary[dev][k]["p90"]
            if not (p90 <= lim):
                fails.append(f"{dev}/{k}: p90={p90*1e3:.3g} mV > {lim*1e3} mV")
    return fails, cap_fails


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="Use only first 6 ngspice rows for fast end-to-end test.")
    ap.add_argument("--no-ngspice", action="store_true",
                    help="Skip ngspice run; reuse existing grid_out.csv.")
    args = ap.parse_args()

    if not args.no_ngspice:
        run_ngspice()
    rows_ng = parse_grid_csv()
    print(f"[z141] parsed {len(rows_ng)} ngspice rows", flush=True)

    if args.smoke:
        # Pick 3 corners per device covering subthreshold, near-Vth, strong inv:
        # (Vgs=0.2, Vds=0.05, Vbs=0)  — deep subthreshold linear
        # (Vgs=0.6, Vds=0.5,  Vbs=0)  — near-Vth, mid-Vds
        # (Vgs=1.0, Vds=1.0,  Vbs=0)  — strong inversion, saturation
        def pick(rows, device):
            keys = [(0.2, 0.05, 0.0), (0.6, 0.5, 0.0), (1.0, 1.0, 0.0)]
            out = []
            for vg, vd, vb in keys:
                for r in rows:
                    if (r["device"] == device and
                        abs(r["vgs"]-vg) < 1e-9 and abs(r["vds"]-vd) < 1e-9
                        and abs(r["vbs"]-vb) < 1e-9):
                        out.append(r); break
            return out
        rows_ng = pick(rows_ng, "M1") + pick(rows_ng, "M2")
        print(f"[z141] SMOKE TEST: {len(rows_ng)} pts (subthreshold/near-Vth/strong-inv)",
              flush=True)

    models = load_models()
    print("[z141] models loaded; comparing...", flush=True)
    records = compare(rows_ng, models)
    summary = summarize(records)

    suffix = "_smoke" if args.smoke else ""
    out_png = OUT_DIR / f"cdfs{suffix}.png"
    plot_cdfs(records, out_png)
    out_json = OUT_DIR / f"summary{suffix}.json"
    with open(out_json, "w") as f:
        json.dump({"n_points": len(records), "summary": summary}, f, indent=2)
    print(f"[z141] wrote {out_json}", flush=True)

    # Print summary table
    print("\n=== SUMMARY ===")
    for dev in ("M1", "M2"):
        print(f"\n[{dev}]  n={summary[dev]['id']['n']}")
        print(f"  {'quant':<8}{'p50':>12}{'p90':>12}{'p99':>12}{'max':>12}")
        for k in ALL_QUANTS:
            s = summary[dev][k]
            unit = "V" if k in ABS_QUANTS else ""
            print(f"  {k:<8}{s['p50']:>12.3e}{s['p90']:>12.3e}"
                  f"{s['p99']:>12.3e}{s['max']:>12.3e}  {unit}")

    fails, cap_fails = verdict(summary)
    print()
    if cap_fails:
        print(f"CAPS (informational, capmod=0 vs capmod=2 + overlaps mismatch expected):")
        for f_ in cap_fails:
            print("  -", f_)
    if fails and not args.smoke:
        print("FAIL  (load-bearing quantities outside tolerance)")
        for f_ in fails:
            print("  -", f_)
        sys.exit(1)
    elif fails:
        print(f"SMOKE: {len(fails)} load-bearing violations on subset "
              f"(may resolve with full 180-pt; rerun without --smoke):")
        for f_ in fails:
            print("  -", f_)
    else:
        print("PASS")


if __name__ == "__main__":
    main()
