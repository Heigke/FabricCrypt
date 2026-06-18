"""z213_vth_bisect — bisect Vth term-by-term between PyTorch BSIM4 port and ngspice.

Single NMOS, M1 of NS-RAM 2T cell:  L=180nm, W=360nm
Bias: VGS=1.2, VDS=2.5, VBS=0
Card: data/sebas_2026_04_22/PTM130bulkNSRAM.txt (cleaned)

Strategy: replicate the dc.py Vth assembly inline with intermediate prints.
Then run a matching ngspice .op deck and probe @m1[vth].
"""
from __future__ import annotations
import math
import os
import subprocess
import tempfile
from pathlib import Path

import torch

from nsram.bsim4_port.model_card import BSIM4Model
from nsram.bsim4_port.geometry import Geometry
from nsram.bsim4_port.temp import compute_size_dep
from nsram.bsim4_port.constants import EPS0, EPSSI

ROOT = Path(__file__).resolve().parents[1]
CARD = ROOT / "data" / "sebas_2026_04_22" / "PTM130bulkNSRAM.txt"
OUT_DIR = ROOT / "results" / "z213_vth_bisect"
OUT_DIR.mkdir(parents=True, exist_ok=True)

L_DRAWN = 180e-9
W_DRAWN = 360e-9
T_C = 27.0
VGS = 1.2
VDS = 2.5
VBS = 0.0


def safe_sqrt(x):
    return torch.sqrt(torch.clamp(torch.as_tensor(x, dtype=torch.float64), min=1e-30))


def exp_thr(x):
    """Faithful BSIM4 exp threshold form: exp(-T0/2)/(1+exp(-T0))*scale or rational."""
    # Replicate BSIM4 _exp_threshold_branch from b4ld.c §1063-1076 — use the
    # 'safe' form: theta = exp(-x/2) / (1 + 2 exp(-x))   (manual eq.2.4-9 form)
    # Actually BSIM4's standard: T1=exp(-T0/2); T4=1+2*T1; Theta0=T1/T4.
    x = torch.as_tensor(x, dtype=torch.float64)
    T1 = torch.exp(-x / 2.0)
    T4 = 1.0 + 2.0 * T1
    return T1 / T4


def py_bisect():
    text = CARD.read_text()
    model = BSIM4Model.from_spice(text, model_type="nmos")
    geom = Geometry(L=L_DRAWN, W=W_DRAWN, NF=1)
    sd = compute_size_dep(model, geom, T_C=T_C)

    P = sd.scaled
    ctx = sd.model_ctx

    Leff = sd.geom.leff
    Weff = sd.geom.weff
    toxe = ctx.toxe

    # Card raw
    vth0_raw = model.get("vth0", 0.0)
    vth0_T = sd.vth0_T  # after L/W scaling + temp

    k1 = P["k1"]
    k2 = P["k2"]
    k3 = P.get("k3", model.get("k3", 80.0))
    k3b = P.get("k3b", model.get("k3b", 0.0))
    w0 = P.get("w0", model.get("w0", 2.5e-6))
    dvt0 = P["dvt0"]
    dvt1 = P["dvt1"]
    dvt2 = P["dvt2"]
    dvt0w = P["dvt0w"]
    dvt1w = P["dvt1w"]
    dvt2w = P["dvt2w"]
    eta0 = P["eta0"]
    etab = P["etab"]
    kt1 = model.get("kt1", -0.11)
    kt1l = model.get("kt1l", 0.0)
    kt2 = model.get("kt2", 0.022)
    lpe0 = model.get("lpe0", 1.74e-7)
    lpeb = model.get("lpeb", 0.0)
    dsub = P.get("dsub", model.get("dsub", model.get("drout", 0.56)))

    sqrtPhi_pre = sd.sqrtPhi
    phi_pre = sd.phi
    Xdep0 = sd.Xdep0
    vbi = sd.vbi
    vbsc = sd.vbsc
    k1ox = sd.k1ox
    k2ox = sd.k2ox
    litl = sd.litl
    factor1 = ctx.factor1
    epssub = ctx.epssub
    epsrox = ctx.epsrox

    # Bias smoothing  (b4ld.c §1002-1019)
    Vbs = VBS
    T0 = Vbs - vbsc - 0.001
    T1 = math.sqrt(T0 * T0 - 0.004 * vbsc)
    if T0 >= 0:
        Vbseff = vbsc + 0.5 * (T0 + T1)
    else:
        Vbseff = vbsc * (1.0 + (-0.002 / (T1 - T0)))
    T9 = 0.95 * phi_pre
    T0 = T9 - Vbseff - 0.001
    T1 = math.sqrt(T0 * T0 + 0.004 * T9)
    Vbseff = T9 - 0.5 * (T0 + T1)

    Phis = phi_pre - Vbseff
    sqrtPhis = math.sqrt(Phis)
    Xdep = Xdep0 * sqrtPhis / sqrtPhi_pre
    T3_xdep = math.sqrt(Xdep)

    V0 = vbi - phi_pre

    # lt1
    T0 = dvt2 * Vbseff
    if T0 >= -0.5:
        T1 = 1.0 + T0
    else:
        T1 = (1.0 + 3.0 * T0) / (3.0 + 8.0 * T0)
    lt1 = factor1 * T3_xdep * T1

    # ltw
    T0w = dvt2w * Vbseff
    if T0w >= -0.5:
        T1w = 1.0 + T0w
    else:
        T1w = (1.0 + 3.0 * T0w) / (3.0 + 8.0 * T0w)
    ltw = factor1 * T3_xdep * T1w

    # Theta0 (DVT short-L)
    T0_th = dvt1 * Leff / max(lt1, 1e-30)
    # BSIM4 standard:  T1=exp(-T0/2); T4=1+2T1; Theta0 = T1/T4
    T1_th = math.exp(-T0_th / 2.0)
    T4_th = 1.0 + 2.0 * T1_th
    Theta0 = T1_th / T4_th
    Delt_vth = dvt0 * Theta0 * V0   # SUBTRACTED (Vth -= Delt_vth)

    # Narrow-W via dvt0w
    T0_w = dvt1w * Weff * Leff / max(ltw, 1e-30)
    T1_w = math.exp(-T0_w / 2.0)
    T4_w = 1.0 + 2.0 * T1_w
    T5 = T1_w / T4_w
    T2_narrow = dvt0w * T5 * V0  # SUBTRACTED

    # Lpe / RSCE
    T0_lpe = math.sqrt(1.0 + lpe0 / Leff)
    TempRatio = ctx.Temp / ctx.Tnom - 1.0
    Tlpe1 = (k1ox * (T0_lpe - 1.0) * sqrtPhi_pre
             + (kt1 + kt1l / Leff + kt2 * Vbseff) * TempRatio)

    Vth_NarrowW = toxe * phi_pre / (Weff + w0)
    NarrowW_term = (k3 + k3b * Vbseff) * Vth_NarrowW

    # DIBL
    T3_d = eta0 + etab * Vbseff
    if T3_d < 1.0e-4:
        T9_d = 1.0 / (3.0 - 2.0e4 * T3_d)
        T3_clamped = (2.0e-4 - T3_d) * T9_d
    else:
        T3_clamped = T3_d
    tmp_dsub = math.sqrt(epssub / (epsrox * EPS0) * toxe * Xdep0)
    T0_dsub = dsub * Leff / max(tmp_dsub, 1e-40)
    T1_dsub = math.exp(-T0_dsub / 2.0)
    T4_dsub = 1.0 + 2.0 * T1_dsub
    theta0vb0_recomp = T1_dsub / T4_dsub
    DIBL_Sft = T3_clamped * theta0vb0_recomp * VDS

    Lpe_Vb = math.sqrt(1.0 + lpeb / Leff)

    # Body-effect term (k1ox·sqrtPhis - k1·sqrtPhi)·Lpe_Vb
    body_eff = (k1ox * sqrtPhis - k1 * sqrtPhi_pre) * Lpe_Vb
    k2ox_term = -k2ox * Vbseff   # ADDED (so -k2ox*Vbs)

    type_n = 1.0
    Vth = (type_n * vth0_T
           + body_eff
           + k2ox_term
           - Delt_vth
           - T2_narrow
           + NarrowW_term
           + Tlpe1
           - DIBL_Sft)

    out = {
        "vth0_raw": vth0_raw,
        "vth0_T": vth0_T,
        "Leff": Leff, "Weff": Weff, "toxe": toxe,
        "phi": phi_pre, "sqrtPhi": sqrtPhi_pre, "Xdep0": Xdep0,
        "vbi": vbi, "vbsc": vbsc, "Vbseff": Vbseff,
        "k1": k1, "k1ox": k1ox, "k2": k2, "k2ox": k2ox,
        "factor1": factor1, "litl": litl,
        "V0": V0,
        "lt1": lt1, "Theta0": Theta0, "Delt_vth": Delt_vth,
        "ltw": ltw, "T5": T5, "T2_narrow": T2_narrow,
        "T0_lpe": T0_lpe, "Tlpe1": Tlpe1,
        "Vth_NarrowW": Vth_NarrowW, "NarrowW_term": NarrowW_term,
        "tmp_dsub_arg_sqrt": tmp_dsub,
        "T0_dsub": T0_dsub, "theta0vb0_recomp": theta0vb0_recomp,
        "theta0vb0_cached": sd.theta0vb0,
        "DIBL_Sft": DIBL_Sft, "T3_clamped": T3_clamped,
        "body_eff": body_eff, "k2ox_term": k2ox_term,
        "Lpe_Vb": Lpe_Vb,
        "dvt0": dvt0, "dvt1": dvt1, "dvt2": dvt2,
        "dsub": dsub, "lpe0": lpe0, "lpeb": lpeb,
        "kt1": kt1, "TempRatio": TempRatio,
        "Vth_final": Vth,
    }
    return out


def run_ngspice():
    """Build matching ngspice deck. Probe @m1[vth] and other internals."""
    deck = f"""* Vth bisect — sebas card, M1 NS-RAM, L=180n W=360n
.include {CARD}

M1 d g s b NMOS L={L_DRAWN} W={W_DRAWN}
Vg g 0 {VGS}
Vd d 0 {VDS}
Vs s 0 0
Vb b 0 {VBS}

.option temp=27
.control
op
echo "VTH_PROBE_BEGIN"
print @m1[vth]
print @m1[vthsat]
print @m1[vfb]
print @m1[vdsat]
print @m1[gm]
print @m1[gds]
print @m1[gmbs]
print @m1[id]
print @m1[vbs]
print @m1[vgs]
print @m1[vds]
echo "VTH_PROBE_END"
.endc
.end
"""
    with tempfile.NamedTemporaryFile("w", suffix=".cir", delete=False) as f:
        f.write(deck)
        deck_path = f.name
    try:
        proc = subprocess.run(
            ["ngspice", "-b", deck_path],
            capture_output=True, text=True, timeout=30,
        )
    finally:
        os.unlink(deck_path)
    return proc.stdout + "\n=====STDERR=====\n" + proc.stderr


def parse_ngspice_vth(out: str) -> dict[str, float]:
    """Parse `print @m1[vth]` output of form `@m1[vth] = 7.489...e-01`."""
    res = {}
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("@m1[") and "=" in line:
            key = line.split("[")[1].split("]")[0]
            val = line.split("=")[1].strip()
            try:
                res[key] = float(val)
            except ValueError:
                pass
    return res


def fmt(v):
    if isinstance(v, float):
        return f"{v:.6e}"
    return str(v)


def main():
    print("=" * 70)
    print("PyTorch port intermediates")
    print("=" * 70)
    py = py_bisect()
    for k, v in py.items():
        print(f"  {k:24s} = {fmt(v)}")

    print()
    print("=" * 70)
    print("ngspice probe")
    print("=" * 70)
    ng_raw = run_ngspice()
    print(ng_raw)
    ng = parse_ngspice_vth(ng_raw)

    print()
    print("=" * 70)
    print("COMPARISON")
    print("=" * 70)
    py_vth = py["Vth_final"]
    ng_vth = ng.get("vth", float("nan"))
    print(f"  python Vth_final  = {py_vth:.6f} V")
    print(f"  ngspice @m1[vth]  = {ng_vth:.6f} V")
    print(f"  gap (ng - py)     = {(ng_vth - py_vth)*1000:.2f} mV")

    # write artifacts
    with open(OUT_DIR / "ngspice_raw.txt", "w") as f:
        f.write(ng_raw)
    with open(OUT_DIR / "py_intermediates.txt", "w") as f:
        for k, v in py.items():
            f.write(f"{k:24s} = {fmt(v)}\n")
    with open(OUT_DIR / "term_table.md", "w") as f:
        f.write("# Vth term-by-term  (sebas card, M1 L=180n W=360n; VGS=1.2 VDS=2.5 VBS=0)\n\n")
        f.write("| Term | Python | Notes |\n|---|---|---|\n")
        for name in [
            "vth0_raw", "vth0_T",
            "Vbseff", "phi", "sqrtPhi",
            "body_eff", "k2ox_term",
            "Theta0", "Delt_vth",
            "T5", "T2_narrow",
            "T0_lpe", "Tlpe1",
            "Vth_NarrowW", "NarrowW_term",
            "T0_dsub", "theta0vb0_recomp", "theta0vb0_cached",
            "T3_clamped", "DIBL_Sft",
            "Vth_final",
        ]:
            f.write(f"| {name} | {fmt(py[name])} |  |\n")
        f.write(f"\nngspice @m1[vth] = {ng_vth} V\n")
        f.write(f"gap (ng - py) = {(ng_vth - py_vth)*1000:.2f} mV\n")

    return py, ng


if __name__ == "__main__":
    main()
