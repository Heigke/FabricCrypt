"""R-44: BJT standalone Ib comparison — pyport vs ngspice.

Probes Gummel-Poon NPN base current Ib(Vd) at flagship operating point
(Vb=0.484, Vsint=0.182), sweeping Vd ∈ [0.5, 3.0]. ngspice uses Sebas card
parameters (is=5E-9 va=100 bf=10000 ...). pyport uses the SAME parameters
via GummelPoonNPN.from_sebas_card().

This isolates whether Ib differs between implementations BEFORE the R-43
refit muddied the comparison (refit gave Bf=991, Va=0.903, Is=5.95e-12).

Output: results/R44_bjt_leakage/bjt_comparison.json
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import torch

import sys
ROOT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "nsram"))

from nsram.bsim4_port.bjt import GummelPoonNPN, compute_bjt  # noqa: E402

OUT = ROOT / "results/R44_bjt_leakage"
OUT.mkdir(parents=True, exist_ok=True)

# Flagship OP (from z361_pdiode_fix/summary.json)
VB = 0.4842638464549771
VSINT = 0.18220333301291128

# Sweep Vd (= Vc, collector). emitter=GND so Vbe=Vb (fixed)
VD_SWEEP = [0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.2, 2.5, 2.8, 3.0]

# Also probe the REFIT BJT parameters used in R-43 (the actual cell uses these)
REFIT_PARAMS = dict(Bf=991.0, Va=0.903, Is=5.95e-12)


def ngspice_deck(params: dict, vb: float, vsint: float, vd: float) -> str:
    """Build deck for standalone BJT with collector=Sint, base=Vb, emitter=0.

    NOTE: cell uses collector=Sint (z330 deck). Vsint is held by an ideal
    voltage source for this isolated probe. Vd separately drives a body diode
    in the full cell but here we only need Q1 alone.
    """
    is_ = params.get("Is", 5e-9)
    va = params.get("Va", 100.0)
    bf = params.get("Bf", 10000.0)
    br = params.get("Br", 100.0)
    nc = params.get("Nc", 2.0)
    ne = params.get("Ne", 1.5)
    ikr = params.get("Ikr", 0.1)
    ise = params.get("Ise", 0.0)
    isc = params.get("Isc", 0.0)
    # match Sebas card extras (don't matter at DC OP but include for parity)
    return f""".title R44 BJT-only probe vb={vb} vsint={vsint} vd={vd}
.model parasiticBJT NPN(is={is_:g} va={va:g} bf={bf:g} br={br:g} nc={nc:g} ne={ne:g}
+ ikr={ikr:g} ise={ise:g} isc={isc:g} rc=0.1 re=0.1 vje=0.7 cjc=1e-15 fc=0.5
+ cje=0.7e-15 tr=20e-12 tf=25e-12 itf=0.03 vtf=7 xtf=2)

Vb_src   vb     0   DC {vb}
Vsint_src vsint 0   DC {vsint}
* collector=vsint, base=vb, emitter=0  (matches z330 topology)
Q1       vsint  vb  0  parasiticBJT area=1u

.options gmin=1e-18 abstol=1e-18 reltol=1e-6 itl1=1000

.control
op
print -i(Vb_src)
print -i(Vsint_src)
print v(vb)
print v(vsint)
quit
.endc
.end
"""


def parse_ngspice_current(text: str, name: str) -> float | None:
    pat = re.compile(rf"-i\({re.escape(name.lower())}\)\s*=\s*([-+]?\d+\.?\d*[eE]?[-+]?\d*)", re.IGNORECASE)
    m = pat.search(text)
    return float(m.group(1)) if m else None


def run_ngspice(params: dict, vb: float, vsint: float, vd: float):
    deck = ngspice_deck(params, vb, vsint, vd)
    deck_path = OUT / "_tmp.sp"
    deck_path.write_text(deck)
    try:
        proc = subprocess.run(
            ["ngspice", "-b", str(deck_path)],
            capture_output=True, text=True, timeout=30,
        )
        out = proc.stdout + proc.stderr
        # Note: -i(Vb_src) is current INTO + terminal of Vb_src.
        # Vb_src has + at vb node; if base draws current from supply, -i = base current FROM supply INTO base
        # i.e. positive when base current sinks into BJT base (forward bias).
        ib = parse_ngspice_current(out, "Vb_src")
        ic = parse_ngspice_current(out, "Vsint_src")
        return {"Ib": ib, "Ic": ic, "raw": out[-2000:]}
    except subprocess.TimeoutExpired:
        return {"error": "timeout"}


def run_pyport(params: dict, vb: float, vsint: float, vd: float):
    """pyport: emitter=GND ⇒ Vbe=vb; collector=Sint ⇒ Vbc=vb-vsint (Vd not used directly).

    To match the ngspice deck, we sweep COLLECTOR voltage as `vsint` in the
    pyport call (collector = vsint in topology). In the full cell, Vsint is
    pinned by KCL, but in standalone probe we vary collector explicitly. We
    pass collector voltage as the `vsint` argument so Vbc = vb - vsint_eff.

    But the user asked to sweep Vd ∈ [0.5,3.0]. In the standalone probe with
    emitter=0, collector=Sint, the "Vd" axis IS the collector voltage. So we
    interpret vd_sweep as collector voltage and set vsint=vd in the deck.
    """
    bjt = GummelPoonNPN(
        Is=params.get("Is", 5e-9),
        Va=params.get("Va", 100.0),
        Vb=params.get("Vb_early", 1e30),
        Bf=params.get("Bf", 10000.0),
        Br=params.get("Br", 100.0),
        Nf=1.0, Nr=1.0,
        Nc=params.get("Nc", 2.0),
        Ne=params.get("Ne", 1.5),
        Ikf=1e30,
        Ikr=params.get("Ikr", 0.1),
        Ise=params.get("Ise", 0.0),
        Isc=params.get("Isc", 0.0),
        Re=0.1, Rc=0.1, Rb=0.0,
        area=1e-6,
    )
    Vbe_t = torch.tensor(vb, dtype=torch.float64)
    Vbc_t = torch.tensor(vb - vsint, dtype=torch.float64)
    out = compute_bjt(bjt, Vbe=Vbe_t, Vbc=Vbc_t, T_K=300.15)
    return {k: float(v) for k, v in out.items()}


def main():
    print("=== R-44 BJT-only probe: pyport vs ngspice ===")

    # Two param sets to test: SEBAS (factory card) and REFIT (R-43 fitted)
    sebas = dict(Is=5e-9, Va=100.0, Bf=10000.0, Br=100.0, Nc=2.0, Ne=1.5,
                 Ikr=0.1, Ise=0.0, Isc=0.0)
    refit = dict(Is=REFIT_PARAMS["Is"], Va=REFIT_PARAMS["Va"],
                 Bf=REFIT_PARAMS["Bf"], Br=100.0, Nc=2.0, Ne=1.5,
                 Ikr=0.1, Ise=0.0, Isc=0.0)

    results = {"flagship_op": {"Vb": VB, "Vsint": VSINT}, "param_sets": {}}

    for name, params in [("SEBAS", sebas), ("REFIT", refit)]:
        print(f"\n--- {name} params: {params}")
        per_vd = []
        for vd in VD_SWEEP:
            # In standalone probe, collector node is Vsint=vd (we sweep it).
            # For the cell's actual operation, Vsint is pinned ~0.182; vary
            # collector to see what HAPPENS if it varies.
            ng = run_ngspice(params, vb=VB, vsint=vd, vd=vd)
            py = run_pyport(params, vb=VB, vsint=vd, vd=vd)
            row = {"Vd": vd, "ngspice_Ib": ng.get("Ib"),
                   "ngspice_Ic": ng.get("Ic"),
                   "pyport_Ib": py["Ib"], "pyport_Ic": py["Ic"]}
            # Ib ratio (ngspice / pyport)
            if py["Ib"] != 0 and row["ngspice_Ib"] is not None:
                row["Ib_ratio_ng_over_py"] = row["ngspice_Ib"] / py["Ib"]
            per_vd.append(row)
            print(f"  Vd={vd:.2f}  ng_Ib={row['ngspice_Ib']!r}  py_Ib={py['Ib']:+.3e}  "
                  f"ratio={row.get('Ib_ratio_ng_over_py')!r}")
        results["param_sets"][name] = {"params": params, "sweep": per_vd}

    # Also probe at exact flagship: vsint=0.182, vd=2.0 (but cell uses collector=vsint
    # so Vbc=Vb-Vsint=0.302, Vd doesn't enter BJT directly).
    print("\n--- FLAGSHIP exact (Vb=0.484, Vsint=0.182, Vbc=0.302):")
    for name, params in [("SEBAS", sebas), ("REFIT", refit)]:
        ng = run_ngspice(params, vb=VB, vsint=VSINT, vd=2.0)
        py = run_pyport(params, vb=VB, vsint=VSINT, vd=2.0)
        print(f"  {name}: ng_Ib={ng.get('Ib')!r}  py_Ib={py['Ib']:+.3e}  "
              f"ng_Ic={ng.get('Ic')!r}  py_Ic={py['Ic']:+.3e}")
        results["param_sets"][name]["flagship"] = {
            "ngspice": {"Ib": ng.get("Ib"), "Ic": ng.get("Ic")},
            "pyport":  {"Ib": py["Ib"], "Ic": py["Ic"], "kqb": py["kqb"]},
        }

    out_json = OUT / "bjt_comparison.json"
    out_json.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nwrote {out_json}")


if __name__ == "__main__":
    main()
