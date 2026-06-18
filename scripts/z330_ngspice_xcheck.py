"""z330 — R-12c ngspice cross-check at VG1=0.6, VG2=0.20, Vd=2.0.

Goal: determine which Vsint the real SPICE solver finds and compare against
pyport's reported (Vsint=1.867 V, Vb=2.0 V). If ngspice disagrees by >0.5V
the pyport solver's basin/init is wrong; if both agree, Vsint≈Vd is the
real physical attractor for this bias and we need a physics-level fix.

Approach:
  1. Generate an ngspice deck mirroring research_plan/ngspice_repro_harness/
     test_2t_cell_prod.sp (production BJT, M1/M2 Sebas BSIM4 cards) but at
     the target bias (V_G1=0.6, V_G2=0.20, V_d=2.0).
  2. Run ngspice in batch mode (-b), capture stdout, parse v(vsint) v(vb).
  3. Also run pyport _solve_at_fixed_vb at Vb=2.0 for direct comparison.
  4. Apply locked gate (>0.5V diff = pyport bug confirmed).

Output: results/z330_ngspice_xcheck/{deck.sp,ngspice.log,summary.json}
"""
from __future__ import annotations
import os, sys, json, re, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "nsram"))
OUT = ROOT / "results/z330_ngspice_xcheck"
OUT.mkdir(parents=True, exist_ok=True)

VG1 = 0.6
VG2 = 0.20
VD  = 2.0
PYPORT_VSINT_REF = 1.867
PYPORT_VB_REF    = 2.0


DECK = f""".title z330 2T cell ngspice cross-check (VG1={VG1}, VG2={VG2}, Vd={VD})

.include "{ROOT / 'data/sebas_2026_04_22/M1_130DNWFB.txt'}"
.include "{ROOT / 'data/sebas_2026_04_22/M2_130bulkNSRAM.txt'}"

* Production BJT (matches z231/z229 production env: Bf=9000, Va=0.55, Is=1e-9)
.model parasiticBJT NPN(is=1e-9 va=0.55 bf=9000 br=100 nc=2 ikr=100m rc=0.1
+ vje=0.7 re=0.1 cjc=1e-15 fc=0.5 cje=0.7e-15 ne=1.5 ise=0 tr=20e-12 tf=25e-12
+ itf=0.03 vtf=7 xtf=2)

.model Dwell_mod D(IS=3.4089e-19 N=1.017 RS=0)

Vdd     vd       0       DC {VD}
Vg1     vg1      0       DC {VG1}
Vg2     vg2      0       DC {VG2}
Vnwell  vnwell   0       DC 2.0

* Topology: M1 drain=vd, gate=vg1, source=vsint, body=vb (floating)
*           M2 drain=vsint, gate=vg2, source=0, body=0 (m2_body_gnd=True)
*           Q1 NPN: collector=vsint, base=vb, emitter=0  (parasitic floating-body BJT)
*           Dwell: well-to-body diode pulls Vb up via well at 2V
M1  vd vg1 vsint vb NMOSdnwfb L=0.13u W=1u
M2  vsint vg2 0 0 NMOS L=0.234u W=1u
Q1  vsint vb 0 parasiticBJT area=1u
Rwell  vnwell vnwell_x  10G
Dwell  vb     vnwell_x  Dwell_mod

.options gmin=1e-15 abstol=1e-12 reltol=1e-3 itl1=500

.control
op
print v(vsint)
print v(vb)
print v(vd)
print -i(vdd)
quit
.endc

.end
"""


def parse_node(text: str, node: str) -> float | None:
    # ngspice .control print emits e.g. "v(vsint) = 1.234567e+00"
    pat = re.compile(rf"v\({re.escape(node)}\)\s*=\s*([-+]?\d+\.?\d*[eE]?[-+]?\d*)")
    m = pat.search(text)
    if m:
        return float(m.group(1))
    return None


def parse_current(text: str) -> float | None:
    pat = re.compile(r"-i\(vdd\)\s*=\s*([-+]?\d+\.?\d*[eE]?[-+]?\d*)")
    m = pat.search(text)
    if m:
        return float(m.group(1))
    return None


def run_pyport_ref(Vd, VG1, VG2, Vb_fixed):
    try:
        from scripts.nsram_surrogate_4d import _solve_at_fixed_vb, _build_pyport_models
        cfg, M1, M2, bjt = _build_pyport_models()
        return _solve_at_fixed_vb(cfg, M1, M2, bjt, Vd, VG1, VG2, Vb_fixed)
    except Exception as e:
        return {"error": repr(e)}


def main():
    deck_path = OUT / "deck.sp"
    log_path = OUT / "ngspice.log"
    deck_path.write_text(DECK)
    print(f"=== z330 ngspice cross-check (VG1={VG1}, VG2={VG2}, Vd={VD}) ===")
    print(f"deck written: {deck_path}")

    # Run ngspice
    try:
        proc = subprocess.run(
            ["ngspice", "-b", str(deck_path)],
            capture_output=True, text=True, timeout=120,
        )
        log = proc.stdout + "\n--- STDERR ---\n" + proc.stderr
        log_path.write_text(log)
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        print("ngspice TIMEOUT (>120s)")
        return
    except FileNotFoundError:
        print("ngspice NOT INSTALLED"); return

    print(f"ngspice rc={rc}")

    ng_vsint = parse_node(log, "vsint")
    ng_vb    = parse_node(log, "vb")
    ng_vd    = parse_node(log, "vd")
    ng_id    = parse_current(log)

    print(f"\nngspice converged values:")
    print(f"  V(vsint) = {ng_vsint}")
    print(f"  V(vb)    = {ng_vb}")
    print(f"  V(vd)    = {ng_vd}")
    print(f"  Id       = {ng_id}")

    # Run pyport at same bias with Vb pinned at PYPORT_VB_REF
    print(f"\nrunning pyport _solve_at_fixed_vb (Vb pinned at {PYPORT_VB_REF}) ...")
    py = run_pyport_ref(VD, VG1, VG2, PYPORT_VB_REF)
    print(f"  pyport result: {py}")

    # Compare
    diff_vsint = None
    if ng_vsint is not None:
        diff_vsint = abs(ng_vsint - PYPORT_VSINT_REF)

    gate_pass = (diff_vsint is not None) and (diff_vsint > 0.5)
    if gate_pass:
        verdict = "PASS — pyport solver bug CONFIRMED (>0.5V disagreement)"
    elif diff_vsint is not None:
        verdict = "FAIL/INTERESTING — ngspice agrees; Vsint≈Vd may be REAL physics"
    else:
        verdict = "ERROR — could not parse ngspice output"

    summary = {
        "bias": {"VG1": VG1, "VG2": VG2, "Vd": VD},
        "pyport_ref": {"Vsint": PYPORT_VSINT_REF, "Vb": PYPORT_VB_REF},
        "pyport_recomputed": py,
        "ngspice": {
            "rc": rc,
            "Vsint": ng_vsint, "Vb": ng_vb, "Vd": ng_vd, "Id": ng_id,
        },
        "diff_vsint_abs_V": diff_vsint,
        "gate_pass_pyport_bug_confirmed": bool(gate_pass),
        "verdict": verdict,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nΔVsint = {diff_vsint}")
    print(f"VERDICT: {verdict}")
    print(f"saved {OUT / 'summary.json'}")


if __name__ == "__main__":
    main()
