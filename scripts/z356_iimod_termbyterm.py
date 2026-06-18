"""z356 — R-37 Iii term-by-term decomposition.

Goal: at flagship M1 standalone bias (Vg=0.6, Vd=2.0, Vs=0, Vb=0),
compute every intermediate of pyport `compute_iimpact` and compare to
ngspice equivalents. Identify which term holds the ~3.43 dec gap.
"""
from __future__ import annotations
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")

import sys, json, math, subprocess, re, importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "nsram"))

OUT = ROOT / "results/z356_iimod_bug"
OUT.mkdir(parents=True, exist_ok=True)

import torch
torch.set_default_dtype(torch.float64)

from nsram.bsim4_port.model_card import BSIM4Model
from nsram.bsim4_port.geometry import Geometry
from nsram.bsim4_port.temp import compute_size_dep
from nsram.bsim4_port.dc import compute_dc
from nsram.bsim4_port.constants import EXP_THRESHOLD, MIN_EXP


def build_M1():
    sp = importlib.util.spec_from_file_location(
        "v1", ROOT / "scripts/z96_narma10_pilot.py")
    v1 = importlib.util.module_from_spec(sp); sp.loader.exec_module(v1)
    text_M1 = (ROOT / "data/sebas_2026_04_22/M1_130DNWFB_LALPHA0_FIX.txt").read_text()
    M1 = BSIM4Model.from_spice(text_M1, model_type="nmos")
    v1.f.patch_model_values(M1, type_n=True)
    M1._values["voff"] = M1._values.get("voff", -0.1368)
    geom = Geometry(L=0.13e-6, W=1e-6)
    sd = compute_size_dep(M1, geom, T_C=27.0)
    return M1, sd, geom


NG_DECK = """.title z356 M1 Iii detail Vg={Vg} Vd={Vd} Vs=0 Vb={Vb}
.include "{M2path}"
.include "{M1path}"
Vd  d 0 DC {Vd}
Vg  g 0 DC {Vg}
Vs  s 0 DC 0
Vb  b 0 DC {Vb}
M1  d g s b NMOSdnwfb L=0.13u W=1u
.options gmin=1e-15 abstol=1e-14 reltol=1e-3 itl1=300
.control
op
print @m1[id] @m1[isub] @m1[vdsat] @m1[vth] @m1[vgs] @m1[vds] @m1[vbs]
print @m1[gm] @m1[gds] @m1[cgg] @m1[cdd]
quit
.endc
.end
"""

_RE_EQ = re.compile(r"(@?\w+(?:\[\w+\])?)\s*=\s*([-+]?\d+\.?\d*[eE]?[-+]?\d*)")

def run_ngspice(Vg, Vd, Vb):
    deck = NG_DECK.format(
        Vg=Vg, Vd=Vd, Vb=Vb,
        M2path=str(ROOT / "data/sebas_2026_04_22/M2_130bulkNSRAM.txt"),
        M1path=str(ROOT / "data/sebas_2026_04_22/M1_130DNWFB_LALPHA0_FIX.txt"))
    p = OUT / "deck.sp"
    p.write_text(deck)
    proc = subprocess.run(["ngspice", "-b", str(p)],
                          capture_output=True, text=True, timeout=120)
    text = proc.stdout + "\n--- STDERR ---\n" + proc.stderr
    (OUT / "ng.log").write_text(text)
    d = {}
    for m in _RE_EQ.finditer(text):
        try:
            d[m.group(1).lower()] = float(m.group(2))
        except ValueError:
            pass
    return d


def pyport_iimod(M1, sd, Vg, Vd, Vb):
    Vgs = torch.tensor(Vg, dtype=torch.float64)
    Vds = torch.tensor(Vd, dtype=torch.float64)
    Vbs = torch.tensor(Vb, dtype=torch.float64)
    dc = compute_dc(M1, sd, Vgs=Vgs, Vds=Vds, Vbs=Vbs)

    P = sd.scaled
    leff = float(sd.geom.leff)
    alpha0 = P.get("alpha0", 0.0)
    alpha1 = P.get("alpha1", 0.0)
    beta0  = P.get("beta0",  0.0)
    inv_l = float(sd.geom.Inv_L)
    binunit = M1._values.get("binunit", 1)

    Vdseff = float(dc.Vdseff)
    diffVds = max(Vd - Vdseff, 0.0)
    tmp = alpha0 + alpha1 * leff
    T2 = tmp / leff if leff > 0 else 0.0
    threshold = beta0 / EXP_THRESHOLD if beta0 > 0 else float("inf")

    if diffVds > threshold:
        T0 = -beta0 / diffVds
        exp_arg = T0
        T1 = T2 * diffVds * math.exp(T0)
    else:
        exp_arg = -EXP_THRESHOLD
        T1 = T2 * MIN_EXP * diffVds

    # Idsa here is stored as Idsa*Vdseff (per dc.py line 845)
    Idsa_Vdseff_stored = float(dc.Idsa) if dc.Idsa is not None else float(dc.Ids)
    Idsa_recovered = Idsa_Vdseff_stored / max(Vdseff, 1e-30)

    Iii = T1 * Idsa_Vdseff_stored
    return {
        "binunit": int(binunit),
        "Inv_L": inv_l,
        "leff": leff,
        "alpha0_scaled": alpha0,
        "alpha1_scaled": alpha1,
        "beta0_scaled":  beta0,
        "alpha0_base":   M1._values.get("alpha0", 0.0),
        "lalpha0":       M1._values.get("lalpha0", 0.0),
        "beta0_base":    M1._values.get("beta0",  0.0),
        "lbeta0":        M1._values.get("lbeta0", 0.0),
        "Vdseff":  Vdseff,
        "Vdsat":   float(dc.Vdsat),
        "diffVds": diffVds,
        "tmp_alpha_num": tmp,
        "T2":      T2,
        "threshold": threshold,
        "exp_arg": exp_arg,
        "T1":      T1,
        "Idsa_Vdseff_stored": Idsa_Vdseff_stored,
        "Idsa_recovered":     Idsa_recovered,
        "Ids_chan":           float(dc.Ids),
        "Iii":                Iii,
    }


def main():
    M1, sd, geom = build_M1()
    Vg, Vd, Vb = 0.6, 2.0, 0.0
    ng = run_ngspice(Vg, Vd, Vb)
    py = pyport_iimod(M1, sd, Vg, Vd, Vb)
    isub_ng = ng.get("@m1[isub]")
    py["Iii_gap_log10"] = (math.log10(abs(py["Iii"]) / abs(isub_ng))
                            if (isub_ng and py["Iii"]) else None)
    py["ng_isub"] = isub_ng
    py["ng_id"]   = ng.get("@m1[id]")
    py["ng_vth"]  = ng.get("@m1[vth]")
    py["ng_vdsat"]= ng.get("@m1[vdsat]")
    out = {"Vg": Vg, "Vd": Vd, "Vb": Vb, **py}
    (OUT / "term_by_term.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
