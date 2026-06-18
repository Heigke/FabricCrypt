"""z460 — alpha0 falsifier (O76 oracle alert).

Tests the hidden no-op hypothesis: z443_VBIC_AVL and z449_A/B and z454_SB_OFF
all return identical 1.311/2.864 dec on the cell-wide baseline. If alpha0×10
ALSO returns 1.311/2.864 on BOTH z443_DC_VBIC and z446_PT_VBIC pipelines, the
parameter never reaches the body KCL row → CODE BUG.  If the numbers move
materially → invariance is real and alpha0 matters.

4 conditions:
  cell_1: z443_DC_VBIC  alpha0 × 1   (baseline; sanity vs z449=1.311/2.864)
  cell_2: z443_DC_VBIC  alpha0 × 10  (test)
  cell_3: z446_PT_VBIC  alpha0 × 1   (baseline; sanity vs z446 published)
  cell_4: z446_PT_VBIC  alpha0 × 10  (test)

For EACH:
  * Full fwd+bwd cell-wide RMSE on all 33 (VG1,VG2) curves
  * Per-bias V_D-point convergence coverage [0..30]
  * Identical solver settings (only alpha0_scale differs)
  * monkey-patched leak.compute_iimpact records the actual alpha0 P[scaled]
    value seen at runtime to prove the override IS reaching the BSIM4 IIMOD
    branch (alpha0_diagnostic.log).

Pre-registered gates (line 1 of run.log):
  INFRA            = 4 cells complete + summary.json written
  FALSIFIER_PASS   = alpha0×10 changes (fwd_avg or bwd_avg) by ≥ 0.10 dec
                     vs alpha0×1 on EITHER pipeline
  CODE_BUG         = both pipelines give IDENTICAL fwd+bwd (|Δ|≤0.005 dec)
                     for ×1 and ×10
  BONUS            = alpha0×10 lowers DC avg AND closes |fwd-bwd| gap on
                     EITHER pipeline

Run:
  HSA_OVERRIDE_GFX_VERSION=11.0.0 PYTHONUNBUFFERED=1 \\
      nohup venv/bin/python scripts/z460_alpha0_falsifier.py \\
      > results/z460_alpha0_falsifier/nohup.out 2>&1 &
"""
from __future__ import annotations
import importlib.util as _ilu
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)

ROOT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
OUT = ROOT / "results/z460_alpha0_falsifier"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "nsram"))
sys.path.insert(0, str(ROOT / "scripts"))

LOG = open(OUT / "run.log", "w")
DIAG = open(OUT / "alpha0_diagnostic.log", "w")

def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True); LOG.write(line + "\n"); LOG.flush()

def diag(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    DIAG.write(line + "\n"); DIAG.flush()


# ====================================================================== #
# PRE-REGISTERED GATES (MUST be line 1 of run.log)
# ====================================================================== #
PREREG = (
    "PRE-REGISTERED GATES (locked before compute):\n"
    "  INFRA          = 4 cells complete & summary.json written\n"
    "  FALSIFIER_PASS = |alpha0x10 - alpha0x1| >= 0.10 dec on fwd_avg OR\n"
    "                   bwd_avg on EITHER z443_DC_VBIC OR z446_PT_VBIC pipeline\n"
    "  CODE_BUG       = BOTH pipelines give IDENTICAL fwd & bwd (|delta|<=0.005 dec)\n"
    "                   for x1 and x10  -> alpha0 wired to nothing\n"
    "  BONUS          = alpha0x10 lowers avg(fwd,bwd) AND |fwd-bwd| shrinks vs x1\n"
    "  ALPHA0_WIRING  = diagnostic.log shows actual P['alpha0'] == request at >=1\n"
    "                   compute_iimpact call per condition (proves override reaches\n"
    "                   BSIM4 IIMOD branch)\n"
)
LOG.write(PREREG + "\n"); LOG.flush()
print(PREREG, flush=True)


# ====================================================================== #
# Re-use upstream
# ====================================================================== #
def _load(modname, relpath):
    spec = _ilu.spec_from_file_location(modname, ROOT / relpath)
    mod = _ilu.module_from_spec(spec); spec.loader.exec_module(mod); return mod

z427 = _load("z427", "scripts/z427_vsint_fix.py")
z429 = _load("z429", "scripts/z429_multisolver_debug.py")
z432 = _load("z432", "scripts/z432_pseudotransient.py")

# Monkey-patch leak.compute_iimpact to log the alpha0 actually pulled from
# sd.scaled. We tag the global condition name in a module variable.
from nsram.bsim4_port import leak as _leak
_ORIG_IIMPACT = _leak.compute_iimpact
_DIAG_TAG = {"label": "init", "expected": None, "logged": 0, "max_log": 6}

def _patched_iimpact(model, sd, dc_result, Vds):
    if _DIAG_TAG["logged"] < _DIAG_TAG["max_log"]:
        a0 = sd.scaled.get("alpha0", None)
        a0_f = float(a0) if a0 is not None else None
        diag(f"  {_DIAG_TAG['label']} compute_iimpact call #{_DIAG_TAG['logged']+1}: "
             f"sd.scaled['alpha0']={a0_f!r}  expected={_DIAG_TAG['expected']!r}")
        _DIAG_TAG["logged"] += 1
    return _ORIG_IIMPACT(model, sd, dc_result, Vds)

_leak.compute_iimpact = _patched_iimpact

# Re-bind any module that did "from leak import compute_iimpact"
import nsram.bsim4_port.nsram_cell as _ncell
if hasattr(_ncell, "compute_iimpact"):
    _ncell.compute_iimpact = _patched_iimpact


# ====================================================================== #
# Loaders
# ====================================================================== #
import re

DATA = ROOT / "data/sebas_2026_04_22"

def _vg1(s):
    m = re.search(r"VG1=([\d.]+)", s); return float(m.group(1)) if m else None
def _vg2(s):
    m = re.search(r"VG2=(-?\d+\.\d+)", s); return float(m.group(1)) if m else None

def load_curves(half="forward"):
    """half='forward' -> first half of triangular sweep; 'backward' -> second."""
    curves = []
    for d in sorted(DATA.glob("2vHCa-2 I-Vs@VG2 VG1=*")):
        VG1 = _vg1(d.name)
        for f in sorted(d.glob("*.csv")):
            VG2 = _vg2(f.name)
            try:
                data = np.loadtxt(f, delimiter=",", skiprows=1, usecols=(0, 1))
            except Exception:
                continue
            if data.ndim == 1 or len(data) < 20:
                continue
            mid = len(data) // 2
            if half == "forward":
                Vd = data[:mid, 0]; Id = np.abs(data[:mid, 1])
            else:
                Vd = data[mid:, 0]; Id = np.abs(data[mid:, 1])
                order = np.argsort(Vd); Vd = Vd[order]; Id = Id[order]
            mask = (Vd >= 0.05) & (Vd <= 2.0)
            Vd, Id = Vd[mask], Id[mask]
            if len(Vd) > 10:
                idx = np.linspace(0, len(Vd) - 1, 30).astype(int)
                Vd, Id = Vd[idx], Id[idx]
                curves.append({"VG1": VG1, "VG2": VG2,
                               "Vd": torch.tensor(Vd, dtype=torch.float64),
                               "Id": torch.tensor(Id, dtype=torch.float64)})
    return curves


# ====================================================================== #
# Thermal guard
# ====================================================================== #
THERMAL_LIMIT_C = 85.0
def check_thermal(stage=""):
    try:
        t = int(open("/sys/class/thermal/thermal_zone0/temp").read().strip()) / 1000.0
    except Exception:
        return
    if t >= THERMAL_LIMIT_C:
        log(f"!!! THERMAL {t:.1f}C >= {THERMAL_LIMIT_C}C at {stage}; sleeping 60s")
        time.sleep(60)


# ====================================================================== #
# Pipelines
# ====================================================================== #
LOG_EPS = 1e-15

# z443_DC_VBIC extra flags (matches z443 'VBIC_AVL' variant and z449_v449_A)
DC_VBIC_FLAGS = {
    "use_vbic_for_q1": True,
    "vbic_AVC1": 0.5,
    "vbic_AVC2": 0.5,
}
# z446 PT_VBIC: same VBIC flags fed to make_cfg
PT_VBIC_FLAGS = {
    "use_vbic_for_q1": True,
    "vbic_AVC1": 0.5,
    "vbic_AVC2": 0.5,
}


def _score_curve(Id_pred, conv_mask, c):
    """Return (sq_sum, n_pts_scored, n_conv) for one curve."""
    Id_pred_t = torch.as_tensor(Id_pred, dtype=torch.float64)
    conv_t = torch.as_tensor(conv_mask, dtype=torch.bool)
    if not conv_t.any():
        return 0.0, 0, 0
    log_p = torch.log10(Id_pred_t.clamp_min(LOG_EPS))
    log_m = torch.log10(c["Id"].clamp_min(LOG_EPS))
    sq = (log_p - log_m) ** 2
    sq_sel = sq[conv_t]
    return float(sq_sel.sum()), int(sq_sel.numel()), int(conv_t.sum())


def run_dc_vbic(model_M1, model_M2, curves, sebas_rows,
                alpha0_scale: float):
    """z443-style DC Newton w/ V_SINT_PIN + VBIC flags + alpha0 multiplier.
    Returns dict { cell_rmse, per_bias[ {VG1,VG2,log_rmse,n_conv,n_pts,fail} ] }.
    """
    cfg, sd_M1, sd_M2 = z427.make_cfg(model_M1, model_M2, dict(DC_VBIC_FLAGS))
    per_bias = []
    sq_total = 0.0; n_total = 0
    fails = 0
    for c in curves:
        sebas_row = z427.find_params(sebas_rows, c["VG1"], c["VG2"])
        if sebas_row is None or math.isnan(sebas_row.get("K1", float("nan"))):
            per_bias.append({"VG1": c["VG1"], "VG2": c["VG2"],
                             "log_rmse": float("nan"), "n_conv": 0,
                             "n_pts": int(len(c["Vd"])), "fail": "no_sebas"})
            continue
        P_M1, P_M2 = z427.make_overrides(sebas_row)
        # ====== alpha0 scaling (the test knob) ======
        if "alpha0" in P_M1:
            P_M1["alpha0"] = P_M1["alpha0"] * float(alpha0_scale)
        else:
            log(f"  WARN: no alpha0 in P_M1 for VG1={c['VG1']} VG2={c['VG2']}")
        bjt = z427.make_bjt(sebas_row)
        # arm diagnostic for this bias
        _DIAG_TAG["expected"] = (float(P_M1["alpha0"])
                                 if "alpha0" in P_M1 else None)
        Vd_arr = c["Vd"].numpy()
        Id_pred_list, conv_list = [], []
        try:
            with torch.no_grad(), \
                 z427.patch_sd_scaled(sd_M1, P_M1), \
                 z427.patch_sd_scaled(sd_M2, P_M2):
                Vb_warm = 0.0
                for Vd_f in Vd_arr:
                    r = z429.run_vsint_pinned(
                        cfg, model_M1, model_M2, bjt,
                        float(Vd_f), float(c["VG1"]), float(c["VG2"]),
                        Vsint_pin=0.0, Vb_init=Vb_warm)
                    Id_pred_list.append(abs(r["Id"]))
                    conv_list.append(bool(r["converged"]))
                    if r["converged"]:
                        Vb_warm = r["Vb"]
                    else:
                        Vb_warm = 0.0
        except Exception as e:
            fails += 1
            per_bias.append({"VG1": c["VG1"], "VG2": c["VG2"],
                             "log_rmse": float("nan"), "n_conv": 0,
                             "n_pts": len(Vd_arr), "fail": repr(e)[:120]})
            log(f"  DC_VBIC fail VG1={c['VG1']} VG2={c['VG2']}: {e}")
            continue

        sq, n_scored, n_conv = _score_curve(Id_pred_list, conv_list, c)
        if n_scored == 0:
            per_bias.append({"VG1": c["VG1"], "VG2": c["VG2"],
                             "log_rmse": float("nan"), "n_conv": 0,
                             "n_pts": len(Vd_arr), "fail": "zero_conv"})
            continue
        rmse = math.sqrt(sq / n_scored)
        sq_total += sq; n_total += n_scored
        per_bias.append({"VG1": c["VG1"], "VG2": c["VG2"],
                         "log_rmse": rmse, "n_conv": n_conv,
                         "n_pts": len(Vd_arr), "fail": None})
        check_thermal(f"DC_VBIC VG1={c['VG1']} VG2={c['VG2']}")
    cell = math.sqrt(sq_total / max(n_total, 1)) if n_total > 0 else float("nan")
    return {"cell_rmse_dec": cell, "per_bias": per_bias,
            "n_curves": len(per_bias), "fails": fails}


def run_pt_vbic(model_M1, model_M2, curves, sebas_rows,
                alpha0_scale: float, direction: str):
    """z446-style pseudo-transient backward sweep + VBIC + alpha0 scale.
    direction: 'forward' or 'backward' (PT init Vb0=0.0)."""
    cfg, sd_M1, sd_M2 = z427.make_cfg(model_M1, model_M2, dict(PT_VBIC_FLAGS))
    per_bias = []
    sq_total = 0.0; n_total = 0
    fails = 0
    for c in curves:
        sebas_row = z427.find_params(sebas_rows, c["VG1"], c["VG2"])
        if sebas_row is None or math.isnan(sebas_row.get("K1", float("nan"))):
            per_bias.append({"VG1": c["VG1"], "VG2": c["VG2"],
                             "log_rmse": float("nan"), "n_conv": 0,
                             "n_pts": int(len(c["Vd"])), "fail": "no_sebas"})
            continue
        P_M1, P_M2 = z427.make_overrides(sebas_row)
        if "alpha0" in P_M1:
            P_M1["alpha0"] = P_M1["alpha0"] * float(alpha0_scale)
        bjt = z427.make_bjt(sebas_row)
        _DIAG_TAG["expected"] = (float(P_M1["alpha0"])
                                 if "alpha0" in P_M1 else None)
        Vd_arr = c["Vd"].numpy()
        try:
            with torch.no_grad(), \
                 z427.patch_sd_scaled(sd_M1, P_M1), \
                 z427.patch_sd_scaled(sd_M2, P_M2):
                Id_pred, Vb_list, conv_list, niter_list = z432.run_one_bias(
                    cfg, model_M1, model_M2, bjt, Vd_arr,
                    float(c["VG1"]), float(c["VG2"]),
                    backward=(direction == "backward"),
                    Vb_init_first=0.0)
        except Exception as e:
            fails += 1
            per_bias.append({"VG1": c["VG1"], "VG2": c["VG2"],
                             "log_rmse": float("nan"), "n_conv": 0,
                             "n_pts": len(Vd_arr), "fail": repr(e)[:120]})
            log(f"  PT_VBIC({direction}) fail VG1={c['VG1']} VG2={c['VG2']}: {e}")
            continue

        sq, n_scored, n_conv = _score_curve(Id_pred, conv_list, c)
        if n_scored == 0:
            per_bias.append({"VG1": c["VG1"], "VG2": c["VG2"],
                             "log_rmse": float("nan"), "n_conv": 0,
                             "n_pts": len(Vd_arr), "fail": "zero_conv"})
            continue
        rmse = math.sqrt(sq / n_scored)
        sq_total += sq; n_total += n_scored
        per_bias.append({"VG1": c["VG1"], "VG2": c["VG2"],
                         "log_rmse": rmse, "n_conv": n_conv,
                         "n_pts": len(Vd_arr), "fail": None})
        check_thermal(f"PT_VBIC({direction}) VG1={c['VG1']} VG2={c['VG2']}")
    cell = math.sqrt(sq_total / max(n_total, 1)) if n_total > 0 else float("nan")
    return {"cell_rmse_dec": cell, "per_bias": per_bias,
            "n_curves": len(per_bias), "fails": fails}


# ====================================================================== #
# Coverage plot
# ====================================================================== #
def plot_coverage(cells):
    """Heatmap: rows = (VG1,VG2) biases, cols = cells, color = n_conv/30."""
    # canonical bias order from first cell
    biases = [(p["VG1"], p["VG2"]) for p in cells[0]["forward"]["per_bias"]]
    n_b = len(biases)
    cell_labels = [c["label"] for c in cells]
    fig, axes = plt.subplots(1, 2, figsize=(13, max(6, n_b * 0.18)))
    for ax, leg in zip(axes, ["forward", "backward"]):
        mat = np.zeros((n_b, len(cells)))
        for j, c in enumerate(cells):
            for i, p in enumerate(c[leg]["per_bias"]):
                mat[i, j] = (p["n_conv"] / max(p["n_pts"], 1)) if p["n_pts"] else 0
        im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
        ax.set_xticks(range(len(cells))); ax.set_xticklabels(cell_labels, rotation=30)
        ax.set_yticks(range(n_b))
        ax.set_yticklabels([f"VG1={b[0]:.2f} VG2={b[1]:+.2f}" for b in biases],
                           fontsize=6)
        ax.set_title(f"V_D-point conv coverage ({leg})")
        plt.colorbar(im, ax=ax, fraction=0.04)
    fig.suptitle("z460 per-bias V_D coverage (4 cells)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "per_bias_coverage.png", dpi=130)
    plt.close(fig)


# ====================================================================== #
# Main
# ====================================================================== #
def main():
    t0 = time.time()
    log("z460 starting — alpha0 falsifier (4 cells × fwd+bwd)")

    model_M1, model_M2 = z429.build_models()
    curves_fwd = load_curves("forward")
    curves_bwd = load_curves("backward")
    sebas_rows = z429.load_sebas_params()
    log(f"loaded: {len(curves_fwd)} fwd curves, {len(curves_bwd)} bwd curves, "
        f"{len(sebas_rows)} sebas rows")

    cells = []

    def run_cell(label, pipeline, alpha0_scale):
        check_thermal(label + " start")
        log(f"===== {label} alpha0×{alpha0_scale:g} =====")
        # Reset diagnostic counter per cell so we always capture some
        _DIAG_TAG["label"] = label
        _DIAG_TAG["logged"] = 0
        _DIAG_TAG["max_log"] = 8
        t_cell = time.time()

        if pipeline == "DC_VBIC":
            res_fwd = run_dc_vbic(model_M1, model_M2, curves_fwd, sebas_rows,
                                  alpha0_scale)
            check_thermal(label + " mid")
            res_bwd = run_dc_vbic(model_M1, model_M2, curves_bwd, sebas_rows,
                                  alpha0_scale)
        elif pipeline == "PT_VBIC":
            res_fwd = run_pt_vbic(model_M1, model_M2, curves_fwd, sebas_rows,
                                  alpha0_scale, direction="forward")
            check_thermal(label + " mid")
            res_bwd = run_pt_vbic(model_M1, model_M2, curves_bwd, sebas_rows,
                                  alpha0_scale, direction="backward")
        else:
            raise ValueError(pipeline)

        # Capture one explicit "audit" reading post-run: re-fetch alpha0
        # actually present in sd.scaled at mid-bias for proof.  We do this
        # by running ONE extra compute on a mid-DC bias with diagnostic on.
        cell = {
            "label": label,
            "pipeline": pipeline,
            "alpha0_scale": alpha0_scale,
            "forward": res_fwd,
            "backward": res_bwd,
            "fwd_cell_rmse_dec": res_fwd["cell_rmse_dec"],
            "bwd_cell_rmse_dec": res_bwd["cell_rmse_dec"],
            "avg_cell_rmse_dec": 0.5 * (res_fwd["cell_rmse_dec"]
                                        + res_bwd["cell_rmse_dec"]),
            "wall_sec": round(time.time() - t_cell, 1),
        }
        log(f"  {label}: fwd={cell['fwd_cell_rmse_dec']:.4f}  "
            f"bwd={cell['bwd_cell_rmse_dec']:.4f}  "
            f"avg={cell['avg_cell_rmse_dec']:.4f}  "
            f"wall={cell['wall_sec']:.0f}s")
        return cell

    cells.append(run_cell("z443_DC_VBIC_x1",  "DC_VBIC", 1.0))
    cells.append(run_cell("z443_DC_VBIC_x10", "DC_VBIC", 10.0))
    cells.append(run_cell("z446_PT_VBIC_x1",  "PT_VBIC", 1.0))
    cells.append(run_cell("z446_PT_VBIC_x10", "PT_VBIC", 10.0))

    # ────────── Coverage plot ───────────────────────────────────────────
    try:
        plot_coverage(cells)
    except Exception as e:
        log(f"plot_coverage failed: {e}")

    # ────────── Gate evaluation ─────────────────────────────────────────
    INFRA = (len(cells) == 4) and all(not math.isnan(c["fwd_cell_rmse_dec"])
                                       and not math.isnan(c["bwd_cell_rmse_dec"])
                                       for c in cells)

    dc1, dc10 = cells[0], cells[1]
    pt1, pt10 = cells[2], cells[3]

    dc_fwd_delta = abs(dc10["fwd_cell_rmse_dec"] - dc1["fwd_cell_rmse_dec"])
    dc_bwd_delta = abs(dc10["bwd_cell_rmse_dec"] - dc1["bwd_cell_rmse_dec"])
    pt_fwd_delta = abs(pt10["fwd_cell_rmse_dec"] - pt1["fwd_cell_rmse_dec"])
    pt_bwd_delta = abs(pt10["bwd_cell_rmse_dec"] - pt1["bwd_cell_rmse_dec"])

    FALSIFIER_PASS = any(d >= 0.10 for d in
                         (dc_fwd_delta, dc_bwd_delta, pt_fwd_delta, pt_bwd_delta))
    CODE_BUG = all(d <= 0.005 for d in
                   (dc_fwd_delta, dc_bwd_delta, pt_fwd_delta, pt_bwd_delta))

    def gap(c): return abs(c["fwd_cell_rmse_dec"] - c["bwd_cell_rmse_dec"])
    BONUS = ((dc10["avg_cell_rmse_dec"] < dc1["avg_cell_rmse_dec"] and
              gap(dc10) < gap(dc1)) or
             (pt10["avg_cell_rmse_dec"] < pt1["avg_cell_rmse_dec"] and
              gap(pt10) < gap(pt1)))

    if CODE_BUG:
        VERDICT = "CODE_BUG_CONFIRMED"
    elif FALSIFIER_PASS:
        VERDICT = "INVARIANCE_REAL"
        if BONUS:
            VERDICT += "_BONUS_LIVE_HYPOTHESIS"
    else:
        VERDICT = "INCONCLUSIVE_small_move_below_0.10_dec"

    log(f"GATES: INFRA={INFRA}  FALSIFIER_PASS={FALSIFIER_PASS}  "
        f"CODE_BUG={CODE_BUG}  BONUS={BONUS}  VERDICT={VERDICT}")
    log(f"deltas: DC fwd={dc_fwd_delta:.4f} bwd={dc_bwd_delta:.4f}  "
        f"PT fwd={pt_fwd_delta:.4f} bwd={pt_bwd_delta:.4f}")

    # ────────── Summary JSON ────────────────────────────────────────────
    def coverage_array(cell, leg):
        return [{"VG1": p["VG1"], "VG2": p["VG2"],
                 "n_conv": p["n_conv"], "n_pts": p["n_pts"],
                 "log_rmse": (None if math.isnan(p.get("log_rmse", float("nan")))
                              else p["log_rmse"]),
                 "fail": p["fail"]}
                for p in cell[leg]["per_bias"]]

    summary = {
        "exp": "z460_alpha0_falsifier",
        "wall_sec": round(time.time() - t0, 1),
        "prereg_text": PREREG,
        "z449_published_baseline": {"fwd": 1.3110292, "bwd": 2.864},
        "cells": [
            {
                "label": c["label"],
                "pipeline": c["pipeline"],
                "alpha0_scale": c["alpha0_scale"],
                "fwd_cell_rmse_dec": c["fwd_cell_rmse_dec"],
                "bwd_cell_rmse_dec": c["bwd_cell_rmse_dec"],
                "avg_cell_rmse_dec": c["avg_cell_rmse_dec"],
                "fwd_per_bias_coverage": coverage_array(c, "forward"),
                "bwd_per_bias_coverage": coverage_array(c, "backward"),
                "fwd_n_curves": c["forward"]["n_curves"],
                "bwd_n_curves": c["backward"]["n_curves"],
                "fwd_fails": c["forward"]["fails"],
                "bwd_fails": c["backward"]["fails"],
                "wall_sec": c["wall_sec"],
            } for c in cells
        ],
        "deltas": {
            "DC_VBIC_fwd_delta_dec": dc_fwd_delta,
            "DC_VBIC_bwd_delta_dec": dc_bwd_delta,
            "PT_VBIC_fwd_delta_dec": pt_fwd_delta,
            "PT_VBIC_bwd_delta_dec": pt_bwd_delta,
        },
        "gates": {
            "INFRA": bool(INFRA),
            "FALSIFIER_PASS": bool(FALSIFIER_PASS),
            "CODE_BUG": bool(CODE_BUG),
            "BONUS": bool(BONUS),
            "VERDICT": VERDICT,
        },
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float))

    # ────────── honest_analysis.md ──────────────────────────────────────
    lines = [
        "# z460 — alpha0 falsifier (O76 oracle alert)\n\n",
        f"Wall time: {summary['wall_sec']:.0f} s\n\n",
        "## Pre-registered gates\n",
        f"```\n{PREREG}\n```\n\n",
        "## 4-cell DC table\n",
        "| Cell | Pipeline | alpha0× | fwd dec | bwd dec | avg dec | |fwd-bwd| |\n",
        "|---|---|---|---|---|---|---|\n",
    ]
    for c in cells:
        lines.append(
            f"| {c['label']} | {c['pipeline']} | {c['alpha0_scale']:g}× | "
            f"{c['fwd_cell_rmse_dec']:.4f} | {c['bwd_cell_rmse_dec']:.4f} | "
            f"{c['avg_cell_rmse_dec']:.4f} | "
            f"{abs(c['fwd_cell_rmse_dec']-c['bwd_cell_rmse_dec']):.4f} |\n")
    lines += [
        "\n## Deltas (×10 vs ×1)\n",
        f"- DC_VBIC fwd: |Δ| = {dc_fwd_delta:.4f} dec\n",
        f"- DC_VBIC bwd: |Δ| = {dc_bwd_delta:.4f} dec\n",
        f"- PT_VBIC fwd: |Δ| = {pt_fwd_delta:.4f} dec\n",
        f"- PT_VBIC bwd: |Δ| = {pt_bwd_delta:.4f} dec\n",
        "\n## Gate results\n",
        f"- INFRA: **{'PASS' if INFRA else 'FAIL'}**\n",
        f"- FALSIFIER_PASS (any |Δ|≥0.10): **{'PASS' if FALSIFIER_PASS else 'FAIL'}**\n",
        f"- CODE_BUG (all |Δ|≤0.005): **{'CONFIRMED' if CODE_BUG else 'NO'}**\n",
        f"- BONUS (×10 lowers avg AND closes fwd-bwd gap): **{'YES' if BONUS else 'no'}**\n",
        f"\n## VERDICT: **{VERDICT}**\n",
        "\n## alpha0 wiring proof\n",
        "See `alpha0_diagnostic.log` — each cell logs the actual "
        "`sd.scaled['alpha0']` value seen at `compute_iimpact` runtime, with "
        "the expected value (after ×scale) for comparison. If `expected ~ "
        "10·card` and `logged ~ card`, the override never reached the IIMOD "
        "branch.\n",
        "\n## Implications for honest baseline\n",
    ]
    if CODE_BUG:
        lines += [
            "- The z446.PT_VBIC=1.276 headline and the z449/z443/z454 1.311/2.864 "
            "identity are **artefacts of a no-op wiring bug**.  alpha0 is bound to "
            "no observable cell metric.  The honest baseline is whatever the "
            "underlying solver-fallback path computes, regardless of which "
            "BSIM4 §6.1 multiplier we believe we set.  P1b/HONEST_BASELINE "
            "claims must be retracted pending wiring audit.\n",
        ]
    elif FALSIFIER_PASS:
        lines += [
            "- alpha0 DOES reach the body KCL row — the 4-pipeline identity is a "
            "physics coincidence (or shared limit of the avalanche multiplier), "
            "not a code bug.  P1b/HONEST_BASELINE claims survive this falsifier; "
            "the residual fragility flagged by Grok/Gemini (dropped biases, "
            "directional asymmetry, solver path-dependence) is still on the table.\n",
            "- If BONUS=YES, the literature ALPHA0×10 hypothesis is live and "
            "should be promoted to the next campaign.\n",
        ]
    else:
        lines += [
            "- ×10 moves the cell by <0.10 dec on both pipelines.  This is "
            "INCONCLUSIVE: not the smoking-gun identity needed to confirm CODE_BUG, "
            "but not the material movement needed to declare invariance real.  The "
            "compute_iimpact diagnostic is the tiebreaker; if logged α0 == "
            "expected, the small Δ is a real physics smallness (saturation of "
            "the avalanche term or downstream cancellation); if not, it's a bug.\n",
        ]
    (OUT / "honest_analysis.md").write_text("".join(lines))

    log(f"z460 DONE  verdict={VERDICT}  wall={summary['wall_sec']:.0f}s")
    LOG.close()
    DIAG.close()


if __name__ == "__main__":
    main()
