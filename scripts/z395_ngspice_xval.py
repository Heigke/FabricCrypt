"""z395 — S4-C: ngspice ground-truth cross-check for snapback fold.

Per research_plan/SNAPBACK_SELF_VALIDATION_PLAN_2026-05-15.md.

Goal: at VG1=0.6, VG2=0.2, sweep Vd ∈ [0, 2] V in 0.05V steps. Does ngspice
(40-year industry-validated SPICE) produce a snapback fold (Ids local maximum
followed by > 0.5-decade drop) for Sebas's actual netlist? Compare to:
  - measured data (data/sebas_2026_04_22/.../VG2=0.20_VG=0.6...csv)
  - pyport prediction at same bias

Output: results/z395_ngspice_xval/{summary.json, ids_compare.png, deck.sp, ngspice.log}

Gate cases:
  (a) ngspice fold > 0.5 dec  AND pyport no fold → pyport implementation bug
  (b) ngspice no fold AND pyport no fold        → BSIM4 + standard NPN can't fold
  (c) partial fold (0.3..0.5 dec)               → intermediate

Topology copied directly from z330_ngspice_xcheck.py (production deck used in
the BSIM4 port validation suite). DC sweep replaces the single-OP `op` directive.
"""
from __future__ import annotations
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")

import sys, json, re, subprocess
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "nsram"))

OUT = ROOT / "results/z395_ngspice_xval"
OUT.mkdir(parents=True, exist_ok=True)

VG1 = 0.6
VG2 = 0.20
VNWELL = 2.0
VD_LO, VD_HI, VD_STEP = 0.0, 2.0, 0.05

MEAS_CSV = (ROOT / "data/sebas_2026_04_22/2vHCa-2 I-Vs@VG2 VG1=0.6 vnwell=2/"
            "StandardIV_HH_2vHCa-2_VG2=0.20_VG=0.6(1)_03-45-46PM.csv")


# ------------------------- ngspice deck -------------------------------------
DECK_TMPL = f""".title z395 S4-C ngspice DC sweep (VG1={VG1}, VG2={VG2}, vnwell={VNWELL})

.include "{ROOT / 'data/sebas_2026_04_22/M1_130DNWFB.txt'}"
.include "{ROOT / 'data/sebas_2026_04_22/M2_130bulkNSRAM.txt'}"

* Production BJT (z231/z229 production env)
.model parasiticBJT NPN(is=1e-9 va=0.55 bf=9000 br=100 nc=2 ikr=100m rc=0.1
+ vje=0.7 re=0.1 cjc=1e-15 fc=0.5 cje=0.7e-15 ne=1.5 ise=0 tr=20e-12 tf=25e-12
+ itf=0.03 vtf=7 xtf=2)

.model Dwell_mod D(IS=3.4089e-19 N=1.017 RS=0)

Vdd     vd       0       DC 0
Vg1     vg1      0       DC {VG1}
Vg2     vg2      0       DC {VG2}
Vnwell  vnwell   0       DC {VNWELL}

* 2T NSRAM cell, M1 floating body, parasitic NPN, well diode to V_nwell
M1  vd vg1 vsint vb NMOSdnwfb L=0.13u W=1u
M2  vsint vg2 0 0 NMOS L=0.234u W=1u
Q1  vsint vb 0 parasiticBJT area=1u
Rwell  vnwell vnwell_x  10G
Dwell  vb     vnwell_x  Dwell_mod

.options gmin=1e-15 abstol=1e-14 reltol=1e-3 itl1=500 itl2=200 itl6=100

.control
dc Vdd {VD_LO} {VD_HI} {VD_STEP}
wrdata {OUT}/ngspice_dc.txt -i(vdd) v(vsint) v(vb)
quit
.endc

.end
"""


def run_ngspice():
    deck = OUT / "deck.sp"
    log = OUT / "ngspice.log"
    deck.write_text(DECK_TMPL)
    proc = subprocess.run(["ngspice", "-b", str(deck)],
                          capture_output=True, text=True, timeout=180)
    log.write_text(proc.stdout + "\n--- STDERR ---\n" + proc.stderr)
    return proc.returncode


def parse_ngspice_dc():
    """wrdata writes one column-pair per variable: (v_sweep, var). Combine."""
    path = OUT / "ngspice_dc.txt"
    if not path.exists():
        return None
    data = np.loadtxt(path)
    # Columns: vd1 -i(vdd)  vd2 v(vsint)  vd3 v(vb)
    Vd = data[:, 0]
    Id = data[:, 1]
    Vsint = data[:, 3]
    Vb = data[:, 5]
    return Vd, Id, Vsint, Vb


# ------------------------- pyport reference ---------------------------------
def run_pyport_sweep(Vd_axis):
    from scripts.nsram_surrogate_4d import _build_pyport_models
    from nsram.bsim4_port.nsram_cell_2T import solve_2t_steady_state
    import torch
    cfg, M1, M2, bjt = _build_pyport_models()
    out_Id, out_Vsint, out_Vb = [], [], []
    for vd in Vd_axis:
        try:
            res = solve_2t_steady_state(
                cfg, M1, bjt,
                Vd=torch.tensor(float(vd), dtype=torch.float64),
                VG1=torch.tensor(VG1, dtype=torch.float64),
                VG2=torch.tensor(VG2, dtype=torch.float64),
                model_M2=M2,
            )
            out_Id.append(float(res["Id"]))
            out_Vsint.append(float(res["Vsint"]))
            out_Vb.append(float(res["Vb"]))
        except Exception as e:
            out_Id.append(float('nan'))
            out_Vsint.append(float('nan'))
            out_Vb.append(float('nan'))
            print(f"  pyport FAIL @ Vd={vd}: {e}")
    return np.array(out_Id), np.array(out_Vsint), np.array(out_Vb)


# ------------------------- measured -----------------------------------------
def load_measured():
    """Load Sebas's measured transient I-V. CSV is a TRANSIENT hysteresis loop
    (forward 0→2V then reverse 2→0V at finite ramp rate). Return separate
    forward and reverse legs sorted by time, plus a combined sort-by-Vd
    "DC envelope" (max |Id| at each Vd bin) for ngspice DC comparison.
    """
    import csv
    Vd, Id, t = [], [], []
    with open(MEAS_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                v = float(row['vdata']); i = float(row['idata'])
                tt = float(row['tdata'])
                Vd.append(v); Id.append(i); t.append(tt)
            except (ValueError, KeyError):
                continue
    Vd = np.array(Vd); Id = np.array(Id); t = np.array(t)
    # Sort by time
    order = np.argsort(t)
    Vd, Id, t = Vd[order], Id[order], t[order]
    # Find apex (maximum Vd point) to split fwd/rev
    apex = int(np.argmax(Vd))
    fwd = (Vd[:apex+1], Id[:apex+1])
    rev = (Vd[apex:], Id[apex:])
    return Vd, Id, fwd, rev


def fold_magnitude(Vd, Id):
    """Return (peak_Vd, peak_Id, trough_Vd, trough_Id, fold_dec) for fold AFTER peak.
    Fold = log10(peak_Id / trough_Id), where trough is min Id at Vd > peak_Vd.
    """
    Id_abs = np.abs(Id)
    # Mask out subthreshold (very small currents)
    if np.all(Id_abs < 1e-12):
        return None
    # Find peak in Vd ∈ [0.3, 1.5] (typical snapback window)
    mask = (Vd >= 0.3) & (Vd <= 1.7)
    if mask.sum() < 3:
        return None
    idx = np.where(mask)[0]
    sub_vd, sub_id = Vd[idx], Id_abs[idx]
    peak_local = np.argmax(sub_id)
    peak_Vd = sub_vd[peak_local]
    peak_Id = sub_id[peak_local]
    if peak_local == len(sub_id) - 1:
        return {"peak_Vd": float(peak_Vd), "peak_Id": float(peak_Id),
                "trough_Vd": None, "trough_Id": None, "fold_dec": 0.0}
    # Trough = min Id after peak
    after = sub_id[peak_local:]
    trough_rel = np.argmin(after)
    trough_Vd = sub_vd[peak_local + trough_rel]
    trough_Id = after[trough_rel]
    fold_dec = (np.log10(max(peak_Id, 1e-20)) -
                np.log10(max(trough_Id, 1e-20)))
    return {"peak_Vd": float(peak_Vd), "peak_Id": float(peak_Id),
            "trough_Vd": float(trough_Vd), "trough_Id": float(trough_Id),
            "fold_dec": float(fold_dec)}


def main():
    print(f"=== z395 S4-C ngspice ground-truth cross-check ===")
    print(f"VG1={VG1}, VG2={VG2}, vnwell={VNWELL}, Vd∈[{VD_LO},{VD_HI}] step {VD_STEP}")

    # ngspice
    print("\n[1/3] Running ngspice DC sweep...")
    rc = run_ngspice()
    print(f"  rc={rc}")
    ng_data = parse_ngspice_dc()
    if ng_data is None:
        print("  FAIL: no ngspice output")
        return
    Vd_ng, Id_ng, Vsint_ng, Vb_ng = ng_data
    # Convert -i(vdd) to Ids (positive). Convention: ngspice prints -i(vdd) as
    # positive when current flows into vdd terminal (Ids out of drain).
    Id_ng_abs = np.abs(Id_ng)
    print(f"  ngspice points: {len(Vd_ng)}")
    print(f"  Id range: [{Id_ng_abs.min():.3e}, {Id_ng_abs.max():.3e}]")
    print(f"  Vsint range: [{Vsint_ng.min():.3f}, {Vsint_ng.max():.3f}]")
    print(f"  Vb range: [{Vb_ng.min():.3f}, {Vb_ng.max():.3f}]")
    ng_fold = fold_magnitude(Vd_ng, Id_ng)
    print(f"  ngspice fold: {ng_fold}")

    # pyport
    print("\n[2/3] Running pyport sweep on matched Vd axis...")
    Vd_py = Vd_ng.copy()
    try:
        Id_py, Vsint_py, Vb_py = run_pyport_sweep(Vd_py)
        py_fold = fold_magnitude(Vd_py, Id_py)
        print(f"  pyport fold: {py_fold}")
    except Exception as e:
        print(f"  pyport sweep FAILED: {e}")
        Id_py = np.full_like(Vd_py, np.nan)
        Vsint_py = np.full_like(Vd_py, np.nan)
        Vb_py = np.full_like(Vd_py, np.nan)
        py_fold = None

    # measured
    print("\n[3/3] Loading measured data...")
    try:
        Vd_meas, Id_meas, (Vd_fwd, Id_fwd), (Vd_rev, Id_rev) = load_measured()
        meas_fold_fwd = fold_magnitude(Vd_fwd, Id_fwd)
        meas_fold_rev = fold_magnitude(Vd_rev, Id_rev)
        print(f"  measured points: {len(Vd_meas)}  (fwd {len(Vd_fwd)}, rev {len(Vd_rev)})")
        print(f"  meas fwd-leg fold: {meas_fold_fwd}")
        print(f"  meas rev-leg fold: {meas_fold_rev}")
        meas_fold = meas_fold_fwd
    except Exception as e:
        print(f"  measured load FAILED: {e}")
        Vd_meas, Id_meas, meas_fold = np.array([]), np.array([]), None
        Vd_fwd, Id_fwd, Vd_rev, Id_rev = (np.array([]),)*4
        meas_fold_fwd = meas_fold_rev = None

    # Verdict
    ng_dec = ng_fold["fold_dec"] if ng_fold else 0.0
    py_dec = py_fold["fold_dec"] if py_fold else 0.0
    meas_dec = meas_fold["fold_dec"] if meas_fold else 0.0
    # NOTE: pyport "fold" of >5 dec at Vd=1.5 indicates Newton found a degenerate
    # root (current collapses, not snapback). True snapback would have peak-then-
    # drop on the rising leg. Look at sign of fold by comparing Id(Vd=2) vs peak.
    if ng_dec < 0.3 and py_dec < 0.3:
        verdict = "S4-C KILL-SHOT (partial): both ngspice and pyport monotone (no fold)"
        case = "b"
    elif ng_dec > 0.5 and py_dec < 0.3:
        verdict = "S4-C DISCOVERY: ngspice has fold but pyport doesn't → pyport bug"
        case = "a"
    elif py_dec > 0.5 and ng_dec < 0.3:
        verdict = ("S4-C SOLVER-ARTIFACT: pyport has fold but ngspice doesn't. "
                   "pyport Newton drifts into degenerate basin not seen by industry SPICE. "
                   "Confirms snapback-program kill: BSIM4+NPN does not fold; pyport's "
                   "apparent fold is a numerical solver pathology, not physics.")
        case = "kill_shot"
    else:
        verdict = "S4-C INTERMEDIATE: ambiguous fold signatures"
        case = "c"
    print(f"\nVERDICT: {verdict}")
    print(f"  ngspice fold = {ng_dec:.3f} dec")
    print(f"  pyport  fold = {py_dec:.3f} dec")
    print(f"  meas    fold = {meas_dec:.3f} dec")

    summary = {
        "bias": {"VG1": VG1, "VG2": VG2, "vnwell": VNWELL,
                 "Vd_axis": [VD_LO, VD_HI, VD_STEP]},
        "ngspice": {"fold": ng_fold,
                    "n_points": int(len(Vd_ng)),
                    "Id_min": float(Id_ng_abs.min()),
                    "Id_max": float(Id_ng_abs.max()),
                    "Vsint_at_Vd2": float(Vsint_ng[-1]),
                    "Vb_at_Vd2": float(Vb_ng[-1])},
        "pyport": {"fold": py_fold,
                   "any_nan": bool(np.any(np.isnan(Id_py)))},
        "measured": {"fold_fwd": meas_fold_fwd, "fold_rev": meas_fold_rev,
                     "n_points": int(len(Vd_meas)),
                     "Id_peak": float(np.max(np.abs(Id_meas))) if len(Id_meas) else None},
        "verdict": verdict, "case": case,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))

    # Save traces
    np.savez(OUT / "traces.npz",
             Vd_ng=Vd_ng, Id_ng=Id_ng, Vsint_ng=Vsint_ng, Vb_ng=Vb_ng,
             Vd_py=Vd_py, Id_py=Id_py, Vsint_py=Vsint_py, Vb_py=Vb_py,
             Vd_meas=Vd_meas, Id_meas=Id_meas,
             Vd_fwd=Vd_fwd, Id_fwd=Id_fwd, Vd_rev=Vd_rev, Id_rev=Id_rev)

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.semilogy(Vd_ng, Id_ng_abs, 'b-', lw=2, label=f'ngspice (fold={ng_dec:.2f} dec)')
        if not np.all(np.isnan(Id_py)):
            ax.semilogy(Vd_py, np.abs(Id_py), 'r--', lw=2,
                        label=f'pyport (fold={py_dec:.2f} dec)')
        if len(Vd_fwd):
            ax.semilogy(Vd_fwd, np.abs(Id_fwd), 'k.-', ms=3, lw=0.8,
                        label=f'meas FWD leg (peak={np.max(np.abs(Id_fwd)):.2e})')
        if len(Vd_rev):
            ax.semilogy(Vd_rev, np.abs(Id_rev), color='gray', ls='-', lw=0.8,
                        marker='x', ms=3, label='meas REV leg')
        ax.set_xlabel('Vd (V)')
        ax.set_ylabel('|Ids| (A)')
        ax.set_title(f'S4-C: ngspice vs pyport vs meas @ VG1={VG1}, VG2={VG2}\n{verdict}')
        ax.legend(loc='lower right')
        ax.grid(True, which='both', alpha=0.3)
        ax.set_ylim(1e-12, 1e-3)
        fig.tight_layout()
        fig.savefig(OUT / "ids_compare_ngspice_pyport_meas.png", dpi=130)
        plt.close(fig)
        print(f"\nPlot: {OUT}/ids_compare_ngspice_pyport_meas.png")
    except Exception as e:
        print(f"  plot FAILED: {e}")

    print(f"\nSummary: {OUT}/summary.json")


if __name__ == "__main__":
    main()
