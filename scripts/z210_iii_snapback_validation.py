"""z210: Compare Iii (impact-ionization current) at the snapback knee
between ngspice and our PyTorch BSIM4 port on Sebas's 2T NS-RAM cell.

Bias: VG1=0.4, VG2=0.2, Vd=1.8 V (snapback regime).

Outputs all internal currents needed to diagnose: Vsint, Vb, Id_M1,
Iii_M1, Iii_M2, Igidl_M1, Ic_Q1, Ib_Q1.
"""
from __future__ import annotations
import os, re, subprocess, tempfile
from pathlib import Path

import numpy as np
import torch

REPO = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
NGSPICE = "/usr/bin/ngspice"
PTM = REPO / "data/sebas_2026_04_22/PTM130bulkNSRAM.txt"
BJT = REPO / "data/sebas_2026_04_22/parasiticBJT.txt"

VG1 = float(os.environ.get("VG1", "0.4"))
VG2 = float(os.environ.get("VG2", "0.2"))
VD  = float(os.environ.get("VD",  "1.7"))


# ---------------- ngspice card normalization (same as z81) ----------------
def _normalize(src: Path, dst: Path) -> None:
    txt = src.read_text()
    txt = re.sub(r"^(\s*\.param\s+\w+)\s+(?!=)(\S)",
                 r"\1 = \2", txt, flags=re.MULTILINE | re.IGNORECASE)
    txt = "\n.param vsatn = 80000\n" + txt
    dst.write_text(txt)


PTM_NG = Path("/tmp/PTM130bulkNSRAM_ng.txt")
BJT_NG = Path("/tmp/parasiticBJT_ng.txt")
_normalize(PTM, PTM_NG)
_normalize(BJT, BJT_NG)


# ---------------- ngspice run ----------------
NETLIST = f"""* NS-RAM 2T snapback Iii probe
.include {PTM_NG}
.include {BJT_NG}

M1 D G1 Sint B NMOS l=0.18u w=0.36u
M2 Sint G2 0 B NMOS l=1.8u w=0.36u
Q1 D B Sint parasiticBJT area=1u
C1 B 0 1f

VD  D  0 DC {VD}
VG1 G1 0 DC {VG1}
VG2 G2 0 DC {VG2}

.options gmin=1e-15 abstol=1e-14 reltol=1e-4 itl1=500 itl2=200

.control
op
print v(d) v(sint) v(b)
print -i(vd) -i(vg1) -i(vg2)
* BSIM4 internal device currents
print @m1[id] @m1[isub] @m1[igidl] @m1[igisl] @m1[igb]
print @m2[id] @m2[isub] @m2[igidl] @m2[igisl] @m2[igb]
* BJT
print @q1[ic] @q1[ib] @q1[ie]
quit
.endc
.end
"""


def run_ngspice():
    with tempfile.NamedTemporaryFile("w", suffix=".cir", delete=False) as f:
        f.write(NETLIST); cir = f.name
    try:
        res = subprocess.run([NGSPICE, "-b", cir],
                             capture_output=True, text=True, timeout=60)
    finally:
        os.unlink(cir)
    return res.stdout + "\n" + res.stderr


# ---------------- Parse ngspice ASCII ----------------
def parse_value(text: str, key: str) -> float | None:
    """Parse pattern like 'v(sint)              = 1.234e-01' or
    '@m1[isub]           = 5.67e-09' from ngspice 'print' output."""
    pat = re.compile(rf"^\s*{re.escape(key)}\s*=\s*([\-\+\deE\.\s]+)$",
                     re.MULTILINE | re.IGNORECASE)
    m = pat.search(text)
    if not m:
        return None
    return float(m.group(1).strip().split()[0])


# ---------------- Python port run ----------------
def run_python():
    from nsram.bsim4_port.model_card import BSIM4Model
    from nsram.bsim4_port.bjt import GummelPoonNPN
    from nsram.bsim4_port.nsram_cell_2T import (
        NSRAMCell2TConfig, solve_2t_steady_state,
    )
    model = BSIM4Model.from_spice(PTM.read_text(), model_type="nmos")
    bjt = GummelPoonNPN.from_sebas_card()
    cfg = NSRAMCell2TConfig()
    Vd_t = torch.tensor([VD], dtype=torch.float64)
    VG1_t = torch.tensor([VG1], dtype=torch.float64)
    VG2_t = torch.tensor([VG2], dtype=torch.float64)
    out = solve_2t_steady_state(cfg, model, bjt,
                                Vd=Vd_t, VG1=VG1_t, VG2=VG2_t,
                                Vb_init=torch.tensor([0.5], dtype=torch.float64),
                                verbose=False)
    return out


def f(x):
    if x is None: return float("nan")
    return float(x)


def main():
    print(f"=== z210 Iii snapback validation @ VG1={VG1}, VG2={VG2}, Vd={VD} ===\n")

    # ngspice
    print("[ngspice]")
    out = run_ngspice()
    debug_path = "/tmp/z210_ngspice.log"
    Path(debug_path).write_text(out)
    print(f"  raw log → {debug_path}")
    ng = {
        "Vd":      parse_value(out, "v(d)"),
        "Vsint":   parse_value(out, "v(sint)"),
        "Vb":      parse_value(out, "v(b)"),
        "Id_total": -1 * (parse_value(out, "v(vd)#branch") or 0.0),  # may not be in print
        "Id_M1":   parse_value(out, "@m1[id]"),
        "Iii_M1":  parse_value(out, "@m1[isub]"),
        "Igidl_M1":parse_value(out, "@m1[igidl]"),
        "Igisl_M1":parse_value(out, "@m1[igisl]"),
        "Igb_M1":  parse_value(out, "@m1[igb]"),
        "Id_M2":   parse_value(out, "@m2[id]"),
        "Iii_M2":  parse_value(out, "@m2[isub]"),
        "Igidl_M2":parse_value(out, "@m2[igidl]"),
        "Ic_Q1":   parse_value(out, "@q1[ic]"),
        "Ib_Q1":   parse_value(out, "@q1[ib]"),
        "Ie_Q1":   parse_value(out, "@q1[ie]"),
    }
    for k, v in ng.items():
        print(f"  {k:>10s} = {f(v):+.6e}" if v is not None else f"  {k:>10s} = ---")

    # Python port
    print("\n[python port]")
    py_out = run_python()
    comp = py_out["components"]
    py = {
        "Vsint":  py_out["Vsint"].item(),
        "Vb":     py_out["Vb"].item(),
        "Id":     py_out["Id"].item(),
        "Id_M1":  comp["Ids_M1"].item(),
        "Id_M2":  comp["Ids_M2"].item(),
        "Iii_M1": comp["Iii_M1"].item(),
        "Iii_M2": comp["Iii_M2"].item(),
        "Igidl_M1": comp["Igidl_M1"].item(),
        "Igidl_M2": comp["Igidl_M2"].item(),
        "Igb_M1": comp["Igb_M1"].item(),
        "Ic_Q1":  comp["Ic_Q1"].item(),
        "Ib_Q1":  comp["Ib_Q1"].item(),
        "niter":  py_out["niter"],
        "conv":   bool(py_out["converged"].all()),
    }
    for k, v in py.items():
        if isinstance(v, (bool, int)):
            print(f"  {k:>10s} = {v}")
        else:
            print(f"  {k:>10s} = {v:+.6e}")

    # Compare
    print("\n[comparison]")
    rows = [
        ("Vsint",   ng["Vsint"], py["Vsint"], "abs"),
        ("Vb",      ng["Vb"],    py["Vb"],    "abs"),
        ("Id_M1",   ng["Id_M1"], py["Id_M1"], "ratio"),
        ("Iii_M1",  ng["Iii_M1"], py["Iii_M1"], "ratio"),
        ("Iii_M2",  ng["Iii_M2"], py["Iii_M2"], "ratio"),
        ("Igidl_M1",ng["Igidl_M1"],py["Igidl_M1"],"ratio"),
        ("Ic_Q1",   ng["Ic_Q1"], py["Ic_Q1"], "ratio"),
        ("Ib_Q1",   ng["Ib_Q1"], py["Ib_Q1"], "ratio"),
    ]
    print(f"  {'quantity':>10s}  {'ngspice':>14s}  {'python':>14s}  {'metric':>14s}")
    for name, a, b, kind in rows:
        a = f(a); b = f(b)
        if kind == "abs":
            d = b - a
            print(f"  {name:>10s}  {a:+.6e}  {b:+.6e}  Δ={d:+.3e} V")
        else:
            r = b/a if a not in (0.0,) and not np.isnan(a) else float("nan")
            print(f"  {name:>10s}  {a:+.6e}  {b:+.6e}  py/ng={r:+.3e}")


if __name__ == "__main__":
    main()
