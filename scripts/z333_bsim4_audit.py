"""z333 — R-15: BSIM4 port term-by-term audit vs ngspice ground truth.

Goal: at the ngspice-converged op-point of the 2T cell
  (VG1=0.6, VG2=0.20, Vd=2.0, Vsint=0.382, Vb=0.267),
compute every per-device BSIM4 current term in BOTH ngspice and pyport
and diff term-by-term. Any structural mismatch in a non-IIMOD term
should appear here.

Method (no solver involvement):
  1. Build ngspice deck identical to z330 (real Sebas BSIM4 cards,
     production BJT, Dwell). Use OP analysis but query *internal device
     variables* @m1[id]/@m1[ibd]/@m1[isub]/@m1[igidl]/@m1[igisl]/@m1[igb]
     for both M1 and M2, plus BJT branch currents.
  2. Read converged node voltages V(vsint), V(vb).
  3. Call pyport's `_residuals` at the EXACT same (Vsint, Vb) — no
     Newton, no basin issues — and read out the components dict.
  4. For each named term, log
        term, ngspice_value, pyport_value, ratio = py/ng, |Δ|
     Sort by |Δ| descending.
  5. Save term_table.json and term_diff_plot.png (log-scale bar chart
     of |ratio − 1| per term).

Verdict gate: any non-IIMOD term with |ratio − 1| > 0.05 is flagged.
"""
from __future__ import annotations
import os, sys, json, re, math, subprocess
from pathlib import Path

# Suppress BLAS oversubscription
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "nsram"))

OUT = ROOT / "results/z333_bsim4_audit"
OUT.mkdir(parents=True, exist_ok=True)

VG1 = 0.6
VG2 = 0.20
VD  = 2.0


# ---------------------------------------------------------------- ngspice deck
DECK = f""".title z333 BSIM4 term audit at VG1={VG1} VG2={VG2} Vd={VD}

.include "{ROOT / 'data/sebas_2026_04_22/M1_130DNWFB.txt'}"
.include "{ROOT / 'data/sebas_2026_04_22/M2_130bulkNSRAM.txt'}"

.model parasiticBJT NPN(is=1e-9 va=0.55 bf=9000 br=100 nc=2 ikr=100m rc=0.1
+ vje=0.7 re=0.1 cjc=1e-15 fc=0.5 cje=0.7e-15 ne=1.5 ise=0 tr=20e-12 tf=25e-12
+ itf=0.03 vtf=7 xtf=2)

.model Dwell_mod D(IS=3.4089e-19 N=1.017 RS=0)

Vdd     vd       0       DC {VD}
Vg1     vg1      0       DC {VG1}
Vg2     vg2      0       DC {VG2}
Vnwell  vnwell   0       DC 2.0

M1  vd vg1 vsint vb NMOSdnwfb L=0.13u W=1u
M2  vsint vg2 0 0 NMOS L=0.234u W=1u
Q1  vsint vb 0 parasiticBJT area=1u
Rwell  vnwell vnwell_x  10G
Dwell  vb     vnwell_x  Dwell_mod

.options gmin=1e-15 abstol=1e-12 reltol=1e-3 itl1=500

.control
op
print v(vsint) v(vb) v(vd)
* M1 internal device variables
print @m1[id] @m1[ibd] @m1[ibs] @m1[isub] @m1[igidl] @m1[igisl] @m1[igs] @m1[igd] @m1[igb]
* M2 internal device variables
print @m2[id] @m2[ibd] @m2[ibs] @m2[isub] @m2[igidl] @m2[igisl] @m2[igs] @m2[igd] @m2[igb]
* BJT branch currents
print @q1[ic] @q1[ib] @q1[ie]
* Well diode current (Vb side, i.e. anode current into vb)
print i(vnwell)
quit
.endc

.end
"""


_re_eq = re.compile(r"(@?\w+(?:\[\w+\])?|v\(\w+\)|i\(\w+\))\s*=\s*([-+]?\d+\.?\d*[eE]?[-+]?\d*)")

def parse_log(text: str) -> dict:
    d = {}
    for m in _re_eq.finditer(text):
        key = m.group(1).lower()
        try:
            d[key] = float(m.group(2))
        except ValueError:
            pass
    return d


# ---------------------------------------------------------------- pyport call
def pyport_components(Vsint_val: float, Vb_val: float) -> dict:
    """Call _residuals at exact (Vsint, Vb) — no solver."""
    import torch
    from scripts.nsram_surrogate_4d import _build_pyport_models
    from nsram.bsim4_port.nsram_cell_2T import _residuals

    cfg, M1, M2, bjt = _build_pyport_models()
    Vd_t   = torch.tensor(VD,  dtype=torch.float64)
    VG1_t  = torch.tensor(VG1, dtype=torch.float64)
    VG2_t  = torch.tensor(VG2, dtype=torch.float64)
    Vs_t   = torch.tensor(Vsint_val, dtype=torch.float64)
    Vb_t   = torch.tensor(Vb_val,    dtype=torch.float64)

    R_S, R_B, comp = _residuals(cfg, M1, bjt, Vd_t, VG1_t, VG2_t,
                                 Vs_t, Vb_t, model_M2=M2)
    out = {"R_Sint": float(R_S), "R_B": float(R_B)}
    for k, v in comp.items():
        try:
            out[k] = float(v)
        except Exception:
            pass
    return out


# ---------------------------------------------------------------- term map
# Map pyport-term-name → ngspice-key (lowercased, as parse_log emits it).
# A pyport term with multiple ngspice contributions sums them.
TERM_MAP = {
    # name                  : (pyport_key,    [list of ngspice keys to sum])
    "Ids_M1":               ("Ids_M1",          ["@m1[id]"]),
    "Ids_M2":               ("Ids_M2",          ["@m2[id]"]),
    "Iii_M1":               ("Iii_M1",          ["@m1[isub]"]),
    "Iii_M2":               ("Iii_M2",          ["@m2[isub]"]),
    "Igidl_M1":             ("Igidl_M1",        ["@m1[igidl]"]),
    "Igisl_M1":             ("Igisl_M1",        ["@m1[igisl]"]),
    "Igidl_M2":             ("Igidl_M2",        ["@m2[igidl]"]),
    "Igisl_M2":             ("Igisl_M2",        ["@m2[igisl]"]),
    "Igb_M1":               ("Igb_M1",          ["@m1[igb]"]),
    "Igb_M2":               ("Igb_M2",          ["@m2[igb]"]),
    "Ibd_M1":               ("Ibd_M1",          ["@m1[ibd]"]),
    "Ibs_M1":               ("Ibs_M1",          ["@m1[ibs]"]),
    "Ibd_M2":               ("Ibd_M2",          ["@m2[ibd]"]),
    "Ibs_M2":               ("Ibs_M2",          ["@m2[ibs]"]),
    "Ic_Q1":                ("Ic_Q1",           ["@q1[ic]"]),
    "Ib_Q1":                ("Ib_Q1",           ["@q1[ib]"]),
    "Ie_Q1":                ("Ie_Q1",           ["@q1[ie]"]),
}


def safe_ratio(py: float, ng: float) -> float | None:
    """Return py/ng, or None when both are essentially zero."""
    if abs(ng) < 1e-30 and abs(py) < 1e-30:
        return 1.0
    if abs(ng) < 1e-30:
        return float("inf") if py != 0.0 else 1.0
    return py / ng


def main():
    print(f"=== z333 BSIM4 term audit ({VG1}, {VG2}, {VD}) ===")
    deck_path = OUT / "deck.sp"
    log_path  = OUT / "ngspice.log"
    deck_path.write_text(DECK)

    proc = subprocess.run(
        ["ngspice", "-b", str(deck_path)],
        capture_output=True, text=True, timeout=120,
    )
    log = proc.stdout + "\n--- STDERR ---\n" + proc.stderr
    log_path.write_text(log)
    print(f"ngspice rc={proc.returncode}")

    parsed = parse_log(log)
    Vsint_ng = parsed.get("v(vsint)")
    Vb_ng    = parsed.get("v(vb)")
    if Vsint_ng is None or Vb_ng is None:
        print("FATAL: could not parse Vsint/Vb from ngspice log"); sys.exit(2)
    print(f"ngspice converged: Vsint={Vsint_ng:.6g}, Vb={Vb_ng:.6g}")

    # Run pyport at exact same node voltages
    py = pyport_components(Vsint_ng, Vb_ng)
    print(f"pyport R_Sint={py['R_Sint']:.3e}   R_B={py['R_B']:.3e}")

    # Build term-by-term table
    rows = []
    for term, (pkey, ng_keys) in TERM_MAP.items():
        py_val = py.get(pkey, math.nan)
        ng_val = sum(parsed.get(k, 0.0) for k in ng_keys)
        ratio  = safe_ratio(py_val, ng_val)
        absdiff = abs(py_val - ng_val)
        # |ratio - 1|; if ratio inf, set a large sentinel for sort
        if ratio is None or math.isinf(ratio) or math.isnan(ratio):
            rdev = float("inf")
        else:
            rdev = abs(ratio - 1.0)
        rows.append({
            "term": term,
            "ngspice_keys": ng_keys,
            "ngspice": ng_val,
            "pyport":  py_val,
            "ratio_py_over_ng": ratio,
            "rel_dev_abs": rdev,
            "abs_diff": absdiff,
        })

    rows.sort(key=lambda r: r["rel_dev_abs"], reverse=True)

    # Print summary
    print("\n--- term table (sorted by |ratio-1| desc) ---")
    print(f"{'term':<14s}{'ngspice':>14s}{'pyport':>14s}{'ratio':>14s}{'|r-1|':>12s}")
    for r in rows:
        rstr = f"{r['ratio_py_over_ng']:.4g}" if r['ratio_py_over_ng'] is not None else "n/a"
        rdev = r['rel_dev_abs']
        rdev_str = "INF" if math.isinf(rdev) else f"{rdev:.3e}"
        print(f"{r['term']:<14s}{r['ngspice']:>14.4e}{r['pyport']:>14.4e}"
              f"{rstr:>14s}{rdev_str:>12s}")

    # Flag non-IIMOD structural terms outside ±5 %
    NON_IIMOD = {"Ids_M1","Ids_M2","Igidl_M1","Igisl_M1","Igidl_M2","Igisl_M2",
                 "Igb_M1","Igb_M2","Ibd_M1","Ibs_M1","Ibd_M2","Ibs_M2",
                 "Ic_Q1","Ib_Q1","Ie_Q1"}
    flagged = [r for r in rows if r["term"] in NON_IIMOD
               and (math.isinf(r["rel_dev_abs"]) or r["rel_dev_abs"] > 0.05)
               # Suppress trivially-small terms (both <1e-20 A)
               and max(abs(r["ngspice"]), abs(r["pyport"])) > 1e-20]
    structural_bug = bool(flagged)

    summary = {
        "bias": {"VG1": VG1, "VG2": VG2, "Vd": VD,
                 "Vsint_ngspice": Vsint_ng, "Vb_ngspice": Vb_ng},
        "pyport_residuals": {"R_Sint": py["R_Sint"], "R_B": py["R_B"]},
        "term_table": rows,
        "non_iimod_flagged_terms": [r["term"] for r in flagged],
        "structural_bug_in_non_iimod_term": structural_bug,
        "verdict": ("STRUCTURAL_BUG_PRESENT" if structural_bug
                    else "NO_STRUCTURAL_BUG_OUTSIDE_IIMOD"),
    }
    (OUT / "term_table.json").write_text(json.dumps(summary, indent=2,
                                                     default=str))
    print(f"\nflagged non-IIMOD terms: {[r['term'] for r in flagged]}")
    print(f"VERDICT: {summary['verdict']}")

    # --- bar plot ------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        names = [r["term"] for r in rows]
        rdevs = [min(r["rel_dev_abs"], 100.0) if not math.isinf(r["rel_dev_abs"])
                  else 100.0 for r in rows]
        colors = ["red" if (r["term"] in NON_IIMOD and r in flagged)
                  else ("orange" if r["term"] in NON_IIMOD
                        else "gray") for r in rows]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(names, rdevs, color=colors)
        ax.axvline(0.05, color="k", ls="--", lw=0.8, label="5% tol")
        ax.set_xscale("symlog", linthresh=1e-3)
        ax.set_xlabel("|pyport/ngspice − 1|  (clipped at 100)")
        ax.set_title(f"z333 BSIM4 term audit  "
                     f"VG1={VG1},VG2={VG2},Vd={VD}  Vs={Vsint_ng:.3f},Vb={Vb_ng:.3f}")
        ax.legend(loc="lower right")
        ax.invert_yaxis()
        fig.tight_layout()
        fig.savefig(OUT / "term_diff_plot.png", dpi=120)
        plt.close(fig)
        print(f"plot: {OUT / 'term_diff_plot.png'}")
    except Exception as e:
        print(f"plot skipped: {e!r}")

    print(f"\nsaved: {OUT / 'term_table.json'}")


if __name__ == "__main__":
    main()
