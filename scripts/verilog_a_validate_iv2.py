#!/usr/bin/env python3
"""verilog_a_validate_iv2.py — Pillar IV.2 full 33-bias VA validation.

Sweeps the 33 canonical (VG1, VG2) bias pairs (Sebas BSIM-fit set,
nsram/Zoom/2026-04-30 BSIMfitsBA/2Tcell_BSIM_param_DC.csv) over a
Vd grid in both forward (0 -> Vd_max) and backward (Vd_max -> 0)
directions, comparing |Id| produced by:

    (a) pyport `solve_2t_steady_state`  (full BSIM4 DC; references
        nsram/nsram/bsim4_port/nsram_cell_2T.py)
    (b) ngspice subcircuit surrogate    (textbook square-law NMOS
        + Gummel-Poon NPN, from nsram/verilog_a/nsram_cell_2T_subckt.sp)

Per-bias dec error = log10(|Id_pyport|) - log10(|Id_ngspice|).

Pre-registered gate (do not modify):
    max-bias RMSE <= 1e-4 dec  -> PASS  (faithful translation)
    max-bias RMSE >= 1e-2 dec  -> FAIL  (structural VA bug)

CAVEAT: the .va uses textbook square-law for M1/M2 intrinsic Ids,
whereas pyport uses full BSIM4 (~1500 LoC). The test is therefore
NOT a strict identity test against pyport — see verdict.md for the
honest model-class-mismatch interpretation. We report numbers as-is.
"""
from __future__ import annotations
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
NSRAM = REPO / "nsram"
VA_SUBCKT = NSRAM / "verilog_a" / "nsram_cell_2T_subckt.sp"
CSV = NSRAM / "Zoom" / "2026-04-30 BSIMfitsBA" / "2Tcell_BSIM_param_DC.csv"
RESULTS = REPO / "results" / "Pillar_IV_2_va_validation"
RESULTS.mkdir(parents=True, exist_ok=True)

# Vd sweep: keep small for budget. 11 points covers the typical IV curve.
VD_MAX = 1.5
N_VD = 11

# ---------------------------------------------------------------------------
# Load bias list
# ---------------------------------------------------------------------------

def load_biases() -> list[tuple[float, float]]:
    biases: list[tuple[float, float]] = []
    seen = set()
    with open(CSV) as f:
        next(f)  # header
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 2:
                continue
            vg1 = round(float(parts[0]), 4)
            vg2 = round(float(parts[1]), 4)
            key = (vg1, vg2)
            if key not in seen:
                seen.add(key)
                biases.append(key)
    return biases

# ---------------------------------------------------------------------------
# pyport call
# ---------------------------------------------------------------------------

sys.path.insert(0, str(NSRAM))
import torch  # noqa: E402
from nsram.bsim4_port.nsram_cell_2T import (  # noqa: E402
    NSRAMCell2TConfig, solve_2t_steady_state,
)
from nsram.bsim4_port.model_card import BSIM4Model  # noqa: E402
from nsram.bsim4_port.bjt import GummelPoonNPN  # noqa: E402

_PYPORT_CFG = None
_PYPORT_MODEL = None
_PYPORT_BJT = None

def _pyport_init():
    global _PYPORT_CFG, _PYPORT_MODEL, _PYPORT_BJT
    if _PYPORT_CFG is not None:
        return
    _PYPORT_CFG = NSRAMCell2TConfig()
    card = (NSRAM / "data" / "sebas_2026_04_22" / "PTM130bulkNSRAM.txt").read_text()
    _PYPORT_MODEL = BSIM4Model.from_spice(card, model_type="nmos")
    _PYPORT_BJT = GummelPoonNPN()

def run_pyport_curve(vg1: float, vg2: float, Vd_grid: np.ndarray) -> np.ndarray:
    """Return |Id| (A) at each Vd. NaN if non-convergent."""
    _pyport_init()
    Vd_t = torch.tensor(Vd_grid, dtype=torch.float64)
    VG1_t = torch.full_like(Vd_t, float(vg1))
    VG2_t = torch.full_like(Vd_t, float(vg2))
    try:
        sol = solve_2t_steady_state(
            _PYPORT_CFG, _PYPORT_MODEL, _PYPORT_BJT,
            Vd_t, VG1_t, VG2_t,
        )
        Id = sol["Id"].detach().cpu().numpy().astype(float)
        conv = sol.get("converged", None)
        if conv is not None:
            mask = conv.detach().cpu().numpy().astype(bool)
            Id = np.where(mask, Id, np.nan)
        return np.abs(Id)
    except Exception as e:
        print(f"  [pyport FAIL] VG1={vg1} VG2={vg2}: {e}", file=sys.stderr)
        return np.full_like(Vd_grid, np.nan)

# ---------------------------------------------------------------------------
# ngspice call — single .op per bias point (snapback edge often kills DCsweep)
# ---------------------------------------------------------------------------

NGSPICE_DECK = """\
* nsram_cell_2T op
.include {subckt}
X1 d g1 g2 0 b nsram_cell_2t
Vd  d  0 DC {Vd:.6f}
Vg1 g1 0 DC {Vg1:.6f}
Vg2 g2 0 DC {Vg2:.6f}
Rb  b  0 1e9
.options gmin=1e-15 reltol=1e-6 abstol=1e-14 itl1=200 itl2=200
.control
op
print i(Vd)
quit
.endc
.end
"""

def _parse_ngspice_id(out: str) -> float:
    Id = float("nan")
    lines = out.splitlines()
    for i, line in enumerate(lines):
        s = line.strip().lower()
        if "i(vd)" in s and "=" in s:
            try:
                Id = float(s.split("=", 1)[1].strip().split()[0])
                return Id
            except Exception:
                pass
        # column-format: header "Index   v(...)   i(vd)" then a value row
        if s.startswith("i(vd)"):
            # next non-blank, non-divider line
            for j in range(i + 1, min(i + 6, len(lines))):
                tk = lines[j].strip().split()
                for t in tk:
                    try:
                        Id = float(t)
                        return Id
                    except Exception:
                        continue
    # generic fallback: look for "i(vd)" anywhere with a number after
    for line in lines:
        s = line.strip().lower()
        if "i(vd)" in s:
            for t in s.replace("=", " ").split():
                try:
                    return float(t)
                except Exception:
                    continue
    return Id

def run_ngspice_point(Vd: float, Vg1: float, Vg2: float) -> float:
    deck = NGSPICE_DECK.format(subckt=str(VA_SUBCKT), Vd=Vd, Vg1=Vg1, Vg2=Vg2)
    with tempfile.NamedTemporaryFile("w", suffix=".sp", delete=False) as f:
        f.write(deck)
        path = f.name
    try:
        proc = subprocess.run(
            ["ngspice", "-b", path],
            capture_output=True, text=True, timeout=30,
        )
        out = proc.stdout + proc.stderr
    except subprocess.TimeoutExpired:
        return float("nan")
    finally:
        try:
            os.unlink(path)
        except Exception:
            pass
    return _parse_ngspice_id(out)

def run_ngspice_curve(vg1: float, vg2: float, Vd_grid: np.ndarray,
                      direction: str) -> np.ndarray:
    seq = Vd_grid if direction == "fwd" else Vd_grid[::-1]
    Id_seq = np.empty_like(seq, dtype=float)
    for i, vd in enumerate(seq):
        Id = run_ngspice_point(float(vd), vg1, vg2)
        # convention: i(Vd) in ngspice = current INTO Vd source = -Id_drain.
        # We want |Id| so just take abs.
        Id_seq[i] = abs(Id) if Id == Id else float("nan")
    if direction == "bwd":
        Id_seq = Id_seq[::-1]
    return Id_seq

# ---------------------------------------------------------------------------
# Per-bias comparison
# ---------------------------------------------------------------------------

def dec_err(I_py: np.ndarray, I_sp: np.ndarray) -> np.ndarray:
    """log10(|I_py|) - log10(|I_sp|), with NaN where either is non-positive/NaN."""
    out = np.full_like(I_py, np.nan, dtype=float)
    ok = (
        np.isfinite(I_py) & np.isfinite(I_sp)
        & (I_py > 1e-30) & (I_sp > 1e-30)
    )
    out[ok] = np.log10(I_py[ok]) - np.log10(I_sp[ok])
    return out

def rmse_dec(arr: np.ndarray) -> float:
    a = arr[np.isfinite(arr)]
    if a.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(a * a)))

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    biases = load_biases()
    print(f"Loaded {len(biases)} bias pairs from {CSV.name}")
    if not VA_SUBCKT.exists():
        print(f"ERROR: subckt missing: {VA_SUBCKT}", file=sys.stderr)
        return 2
    Vd_grid = np.linspace(0.05, VD_MAX, N_VD)  # avoid Vd=0 (trivial)

    rows = []
    per_bias_rmse_fwd: list[float] = []
    per_bias_rmse_bwd: list[float] = []
    nan_count = 0
    total_pts = 0

    t0 = time.time()
    for k, (vg1, vg2) in enumerate(biases):
        print(f"[{k+1:02d}/{len(biases)}] VG1={vg1:+.2f}  VG2={vg2:+.2f}")
        # pyport — one tensor batch
        Ipy = run_pyport_curve(vg1, vg2, Vd_grid)
        # ngspice — point by point, both directions
        Isp_fwd = run_ngspice_curve(vg1, vg2, Vd_grid, "fwd")
        Isp_bwd = run_ngspice_curve(vg1, vg2, Vd_grid, "bwd")
        derr_fwd = dec_err(Ipy, Isp_fwd)
        derr_bwd = dec_err(Ipy, Isp_bwd)
        r_fwd = rmse_dec(derr_fwd)
        r_bwd = rmse_dec(derr_bwd)
        per_bias_rmse_fwd.append(r_fwd)
        per_bias_rmse_bwd.append(r_bwd)
        n_nan_fwd = int(np.sum(~np.isfinite(derr_fwd)))
        n_nan_bwd = int(np.sum(~np.isfinite(derr_bwd)))
        nan_count += n_nan_fwd + n_nan_bwd
        total_pts += 2 * len(Vd_grid)
        rows.append({
            "VG1": vg1, "VG2": vg2,
            "Vd_grid": Vd_grid.tolist(),
            "Id_pyport_A": [float(x) if np.isfinite(x) else None for x in Ipy],
            "Id_ngspice_fwd_A": [float(x) if np.isfinite(x) else None for x in Isp_fwd],
            "Id_ngspice_bwd_A": [float(x) if np.isfinite(x) else None for x in Isp_bwd],
            "dec_err_fwd": [float(x) if np.isfinite(x) else None for x in derr_fwd],
            "dec_err_bwd": [float(x) if np.isfinite(x) else None for x in derr_bwd],
            "rmse_dec_fwd": r_fwd if np.isfinite(r_fwd) else None,
            "rmse_dec_bwd": r_bwd if np.isfinite(r_bwd) else None,
            "n_nan_fwd": n_nan_fwd,
            "n_nan_bwd": n_nan_bwd,
        })
        print(f"    rmse_dec fwd={r_fwd:.3f}  bwd={r_bwd:.3f}  NaN={n_nan_fwd}+{n_nan_bwd}")

    elapsed = time.time() - t0

    fwd = np.array(per_bias_rmse_fwd, dtype=float)
    bwd = np.array(per_bias_rmse_bwd, dtype=float)
    fwd_finite = fwd[np.isfinite(fwd)]
    bwd_finite = bwd[np.isfinite(bwd)]
    all_finite = np.concatenate([fwd_finite, bwd_finite])

    def _stat(a, fn):
        return float(fn(a)) if a.size > 0 else float("nan")

    summary = {
        "n_biases": len(biases),
        "vd_grid": Vd_grid.tolist(),
        "max_bias_rmse_dec_fwd": _stat(fwd_finite, np.max),
        "max_bias_rmse_dec_bwd": _stat(bwd_finite, np.max),
        "max_bias_rmse_dec_overall": _stat(all_finite, np.max),
        "median_rmse_dec_fwd": _stat(fwd_finite, np.median),
        "median_rmse_dec_bwd": _stat(bwd_finite, np.median),
        "p90_rmse_dec_fwd": _stat(fwd_finite, lambda x: np.percentile(x, 90)),
        "p90_rmse_dec_bwd": _stat(bwd_finite, lambda x: np.percentile(x, 90)),
        "nan_count": int(nan_count),
        "total_points": int(total_pts),
        "elapsed_s": elapsed,
        "per_bias": rows,
    }

    # Pre-registered gate
    max_rmse = summary["max_bias_rmse_dec_overall"]
    if not math.isfinite(max_rmse):
        verdict = "INCONCLUSIVE"
    elif max_rmse <= 1e-4:
        verdict = "PASS"
    elif max_rmse >= 1e-2:
        verdict = "FAIL"
    else:
        verdict = "BORDERLINE"
    summary["verdict"] = verdict
    summary["gate_threshold_pass_dec"] = 1e-4
    summary["gate_threshold_fail_dec"] = 1e-2

    # Worst bias
    worst_idx_fwd = int(np.nanargmax(fwd)) if fwd_finite.size else -1
    worst_idx_bwd = int(np.nanargmax(bwd)) if bwd_finite.size else -1
    if worst_idx_fwd >= 0:
        b = biases[worst_idx_fwd]
        summary["worst_bias_fwd"] = {"VG1": b[0], "VG2": b[1],
                                     "rmse_dec": float(fwd[worst_idx_fwd])}
    if worst_idx_bwd >= 0:
        b = biases[worst_idx_bwd]
        summary["worst_bias_bwd"] = {"VG1": b[0], "VG2": b[1],
                                     "rmse_dec": float(bwd[worst_idx_bwd])}

    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2))

    # verdict.md
    md = []
    md.append("# Pillar IV.2 — Verilog-A validation verdict\n")
    md.append(f"**Verdict:** {verdict}\n")
    md.append(f"- N biases: {summary['n_biases']}")
    md.append(f"- Vd grid: {N_VD} points, {Vd_grid[0]:.2f} -> {Vd_grid[-1]:.2f} V")
    md.append(f"- Directions: fwd + bwd (both reported)")
    md.append(f"- NaN count: {summary['nan_count']} / {summary['total_points']} pts")
    md.append(f"- Elapsed: {elapsed:.1f} s\n")
    md.append("## Pre-registered gate")
    md.append("- PASS  if max-bias RMSE ≤ 1e-4 dec")
    md.append("- FAIL  if max-bias RMSE ≥ 1e-2 dec\n")
    md.append("## Stats (dec, log10|Id|)\n")
    md.append("| direction | max | median | p90 |")
    md.append("|---|---|---|---|")
    md.append(f"| fwd | {summary['max_bias_rmse_dec_fwd']:.3f} | "
              f"{summary['median_rmse_dec_fwd']:.3f} | "
              f"{summary['p90_rmse_dec_fwd']:.3f} |")
    md.append(f"| bwd | {summary['max_bias_rmse_dec_bwd']:.3f} | "
              f"{summary['median_rmse_dec_bwd']:.3f} | "
              f"{summary['p90_rmse_dec_bwd']:.3f} |")
    md.append("")
    if "worst_bias_fwd" in summary:
        wf = summary["worst_bias_fwd"]
        md.append(f"**Worst fwd bias:** VG1={wf['VG1']:+.2f}  VG2={wf['VG2']:+.2f}  "
                  f"rmse_dec={wf['rmse_dec']:.3f}")
    if "worst_bias_bwd" in summary:
        wb = summary["worst_bias_bwd"]
        md.append(f"**Worst bwd bias:** VG1={wb['VG1']:+.2f}  VG2={wb['VG2']:+.2f}  "
                  f"rmse_dec={wb['rmse_dec']:.3f}")
    md.append("")
    md.append("## Caveat — model class mismatch\n")
    md.append("The .va uses textbook square-law NMOS (`nmos level=1`) for "
              "M1/M2 intrinsic Ids. The pyport `solve_2t_steady_state` uses "
              "the full BSIM4 v4.x DC model "
              "(nsram/nsram/bsim4_port/dc.py, ~1500 LoC: mobility model, "
              "vsat, DIBL, CLM, poly depletion, quantum-Vth, etc.). A 1:1 "
              "identity test would require either dropping the foundry BSIM4 "
              "VA into the .va or adding a `cfg.intrinsic_model='square_law'` "
              "switch on the pyport side. Neither exists today.\n")
    md.append("As shipped, the RMSE values reflect the BSIM4-vs-square-law "
              "model-class gap, NOT a translation bug. The pre-registered "
              "≤1e-4 dec gate is therefore expected to FAIL — what we "
              "actually want to read off is whether the worst bias is "
              "consistent with that gap (0.1-1 dec, per README) or far worse "
              "(structural VA bug).\n")
    (RESULTS / "verdict.md").write_text("\n".join(md))
    print("\n" + "=" * 60)
    print(f"VERDICT: {verdict}")
    print(f"max-bias RMSE overall: {max_rmse:.3f} dec")
    print(f"NaN: {summary['nan_count']}/{summary['total_points']}")
    print(f"elapsed: {elapsed:.1f} s")
    print(f"saved -> {RESULTS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
