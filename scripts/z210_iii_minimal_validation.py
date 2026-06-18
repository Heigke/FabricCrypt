"""z210b: Validate Iii equation between ngspice and our port using a
MINIMAL BSIM4 card with non-zero alpha0/beta0. The full Sebas card has
alpha0 effectively zeroed (the second `.model` block in BSIM4 dump shows
alpha0=0 — Iii is OFF in that card by design). To validate the EQUATION
we use an explicit minimal card. Single-MOSFET geometry of M1 (180nm/360nm).

Bias: VGS=1.2, VDS=2.5 (deep saturation, Iii fully on).
"""
from __future__ import annotations
import os, re, subprocess, tempfile
from pathlib import Path
import numpy as np
import torch

NGSPICE = "/usr/bin/ngspice"

# Minimal model — deliberately simple so both engines see identical params
MODEL = dict(
    LEVEL=14, TOXE=4e-9, VTH0=0.5,
    ALPHA0=1e-4, BETA0=18.0,
    VSAT=1.35e5,
)
L_NM = 180e-9; W_NM = 360e-9
VGS = 1.2; VDS = 2.5; VBS = 0.0


# ---------------- ngspice ----------------
NETLIST = f"""* minimal Iii probe
.model NM NMOS Level=14 toxe=4n vth0=0.5 alpha0=1e-4 beta0=18 vsat=1.35e5
M1 D G S B NM l={L_NM} w={W_NM}
VD D 0 DC {VDS}
VG G 0 DC {VGS}
VS S 0 DC 0
VB B 0 DC {VBS}
.control
op
print @m1[id] @m1[isub] @m1[vdsat] @m1[vth] @m1[gm] @m1[gds]
quit
.endc
.end
"""


def run_ngspice():
    with tempfile.NamedTemporaryFile("w", suffix=".cir", delete=False) as f:
        f.write(NETLIST); cir = f.name
    try:
        res = subprocess.run([NGSPICE, "-b", cir],
                             capture_output=True, text=True, timeout=30)
    finally:
        os.unlink(cir)
    return res.stdout


def parse(text, key):
    m = re.search(rf"{re.escape(key)}\s*=\s*([\-\+\deE\.\s]+)$",
                  text, re.MULTILINE | re.IGNORECASE)
    return float(m.group(1).strip().split()[0]) if m else None


# ---------------- python port ----------------
def run_python():
    """Build a minimal BSIM4Model with the same params and call compute_dc + compute_iimpact."""
    from nsram.bsim4_port.model_card import BSIM4Model
    from nsram.bsim4_port.geometry import Geometry
    from nsram.bsim4_port.temp import compute_size_dep
    from nsram.bsim4_port.dc import compute_dc
    from nsram.bsim4_port.leak import compute_iimpact

    spice = """.model NM NMOS Level=14
+toxe = 4e-9
+vth0 = 0.5
+alpha0 = 1e-4
+beta0  = 18
+vsat   = 1.35e5
"""
    model = BSIM4Model.from_spice(spice, model_type="nmos")
    geom = Geometry(L=L_NM, W=W_NM)
    sd = compute_size_dep(model, geom, T_C=27.0)

    Vgs = torch.tensor([VGS], dtype=torch.float64)
    Vds = torch.tensor([VDS], dtype=torch.float64)
    Vbs = torch.tensor([VBS], dtype=torch.float64)
    dc = compute_dc(model, sd, Vgs=Vgs, Vds=Vds, Vbs=Vbs)
    Iii = compute_iimpact(model, sd, dc, Vds=Vds)
    return dc, Iii


def main():
    print(f"=== z210b minimal Iii validation @ VGS={VGS}, VDS={VDS}, VBS={VBS} ===\n")

    print("[ngspice]")
    out = run_ngspice()
    Path("/tmp/z210b_ng.log").write_text(out)
    ng = {k: parse(out, f"@m1[{k.lower()}]") for k in ["id","isub","vdsat","vth","gm","gds"]}
    for k,v in ng.items(): print(f"  {k:>6s} = {v:+.6e}" if v is not None else f"  {k}=None")

    print("\n[python port]")
    dc, Iii = run_python()
    py = {
        "Ids":     dc.Ids.item(),
        "Idsa":    dc.Idsa.item() if dc.Idsa is not None else None,  # Idsa·Vdseff
        "Vdseff":  dc.Vdseff.item(),
        "Vdsat":   dc.Vdsat.item() if hasattr(dc,"Vdsat") and dc.Vdsat is not None else None,
        "Iii":     Iii.item(),
    }
    for k,v in py.items(): print(f"  {k:>6s} = {v:+.6e}" if v is not None else f"  {k}=None")

    print("\n[comparison]")
    a = ng["isub"]; b = py["Iii"]
    print(f"  Iii ngspice = {a:+.6e}")
    print(f"  Iii python  = {b:+.6e}")
    if a and a != 0:
        print(f"  ratio py/ng = {b/a:+.4f}")
        print(f"  rel err     = {abs(b-a)/abs(a)*100:.2f}%")
    a = ng["id"]; b = py["Ids"]
    print(f"  Id  ngspice = {a:+.6e}")
    print(f"  Id  python  = {b:+.6e}")
    if a and a != 0:
        print(f"  ratio py/ng = {b/a:+.4f}")


if __name__ == "__main__":
    main()
