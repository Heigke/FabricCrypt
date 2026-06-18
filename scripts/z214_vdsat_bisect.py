"""z214_vdsat_bisect — bisect Vdsat term-by-term between PyTorch BSIM4 port and ngspice.

Single NMOS, M1 of NS-RAM 2T cell:  L=180nm, W=360nm
Bias: VGS=1.2, VDS=2.5, VBS=0
Card: data/sebas_2026_04_22/PTM130bulkNSRAM.txt (cleaned)

Vth matches exactly. Suspect Vdsat block:
  Esat = 2*vsattemp/mueff
  EsatL = Esat * Leff
  Vdsat (full quadratic, b4ld.c §1636-1679)
  + Abulk computation upstream (§1338-1395)

Strategy:
  1. Call our compute_dc() and inspect DCResult fields
  2. Re-derive Esat, EsatL, simple Vdsat, full Vdsat from {mueff, Abulk, Vgsteff}
  3. Run ngspice with @m1[X] probes for: id, vdsat, gm, gds, vgs_eff, etc.
  4. Compare side by side; identify the differing intermediate.
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
from nsram.bsim4_port.dc import compute_dc

ROOT = Path(__file__).resolve().parents[1]
CARD = ROOT / "data" / "sebas_2026_04_22" / "PTM130bulkNSRAM.txt"
OUT_DIR = ROOT / "results" / "z214_vdsat_bisect"
OUT_DIR.mkdir(parents=True, exist_ok=True)

L_DRAWN = 180e-9
W_DRAWN = 360e-9
T_C = 27.0
VGS = 1.2
VDS = 2.5
VBS = 0.0


def py_compute():
    text = CARD.read_text()
    # Sebas card has `vsat = vsatn` with vsatn referenced but not defined; the
    # ngspice deck supplies vsatn = 80000 (z210 convention).  Pass the same
    # value to from_spice so the port and ngspice agree on vsat.
    model = BSIM4Model.from_spice(text, model_type="nmos",
                                  params={"vsatn": 80000.0})
    geom = Geometry(L=L_DRAWN, W=W_DRAWN, NF=1)
    sd = compute_size_dep(model, geom, T_C=T_C)
    res = compute_dc(model, sd,
                    Vgs=torch.tensor(VGS, dtype=torch.float64),
                    Vds=torch.tensor(VDS, dtype=torch.float64),
                    Vbs=torch.tensor(VBS, dtype=torch.float64))

    # Derived quantities
    Leff = float(sd.geom.leff)
    Weff = float(sd.geom.weff)
    vsattemp = float(sd.vsattemp)
    u0temp = float(sd.u0temp)
    coxe = float(sd.model_ctx.coxe)
    Vtm = float(sd.model_ctx.vtm)
    mueff = float(res.mueff)
    Abulk = float(res.Abulk)
    Vgsteff = float(res.Vgsteff)
    Vdsat = float(res.Vdsat)
    Vdseff = float(res.Vdseff)
    Vth = float(res.Vth)
    Ids = float(res.Ids)
    Rds = float(res.Rds) if res.Rds is not None else 0.0

    P = sd.scaled
    Esat = 2.0 * vsattemp / mueff
    EsatL = Esat * Leff
    Vgst2Vtm = Vgsteff + 2.0 * Vtm
    Vdsat_simple = EsatL * Vgst2Vtm / (Abulk * EsatL + Vgst2Vtm)

    # ---- Abulk components (re-derive faithfully to compare with C source) ----
    # T1 = 0.5 k1ox Lpe_Vb / sqrtPhis + k2ox - k3b·Vth_NarrowW
    # T5 = Leff / (Leff + 2 sqrt(xj·Xdep));  T7 = T5³
    # Abulk0 = 1 + T1·(a0·T5 + b0/(Weff+b1))
    # dAbulk_dVg = -T1 · ags·a0·T7
    # Abulk = Abulk0 + dAbulk_dVg · Vgsteff
    # then keta scaling (Vbseff=0 → factor=1)
    k1ox = float(sd.k1ox)
    k2ox = float(sd.k2ox)
    sqrtPhi = float(sd.sqrtPhi)
    Xdep0 = float(sd.Xdep0)
    Lpe_Vb = math.sqrt(1.0 + float(P.get("lpeb", model.get("lpeb", 0.0))) / Leff)
    # at Vbs=0: Xdep = Xdep0
    Xdep = Xdep0
    sqrtPhis = sqrtPhi
    xj_v = float(P.get("xj", model.get("xj", 1.5e-7)))
    T9_xj = math.sqrt(xj_v * Xdep)
    tmp1 = Leff + 2.0 * T9_xj
    T5_a = Leff / tmp1
    T6_a = T5_a * T5_a
    T7_a = T5_a * T6_a
    a0_v = float(P.get("a0", model.get("a0", 1.0)))
    ags_v = float(P.get("ags", model.get("ags", 0.0)))
    b0_v = float(P.get("b0", model.get("b0", 0.0)))
    b1_v = float(P.get("b1", model.get("b1", 0.0)))
    k3b_v = float(P.get("k3b", model.get("k3b", 0.0)))
    w0_v = float(P.get("w0", model.get("w0", 2.5e-6)))
    Vth_NarrowW = float(sd.model_ctx.toxe) * float(sd.phi) / (Weff + w0_v)
    T9_a_pre = 0.5 * k1ox * Lpe_Vb / sqrtPhis
    T1_a = T9_a_pre + k2ox - k3b_v * Vth_NarrowW
    tmp2_a = a0_v * T5_a
    tmp3_a = Weff + b1_v
    tmp4_a = b0_v / tmp3_a if tmp3_a != 0.0 else 0.0
    T2_a = tmp2_a + tmp4_a
    Abulk0_recomp = 1.0 + T1_a * T2_a
    T8_a = ags_v * a0_v * T7_a
    dAbulk_dVg_recomp = -T1_a * T8_a
    Abulk_recomp = Abulk0_recomp + dAbulk_dVg_recomp * Vgsteff

    # Reconstruct Lambda and full quadratic
    a1_v = float(P.get("a1", model.get("a1", 0.0)))
    a2_v = float(P.get("a2", model.get("a2", 1.0)))
    if a1_v == 0.0:
        Lambda = a2_v
    elif a1_v > 0.0:
        T0 = 1.0 - a2_v
        T1 = T0 - a1_v * Vgsteff - 1e-4
        T2 = math.sqrt(T1 * T1 + 4e-4 * T0)
        Lambda = a2_v + T0 - 0.5 * (T1 + T2)
    else:
        T1 = a2_v + a1_v * Vgsteff - 1e-4
        T2 = math.sqrt(T1 * T1 + 4e-4 * a2_v)
        Lambda = 0.5 * (T1 + T2)

    WVCox = Weff * vsattemp * coxe
    WVCoxRds = WVCox * Rds
    T9q = Abulk * WVCoxRds
    T7q = Vgst2Vtm * T9q
    T6q = Vgst2Vtm * WVCoxRds
    T0v = 2.0 * Abulk * (T9q - 1.0 + 1.0 / Lambda)
    T1v = Vgst2Vtm * (2.0 / Lambda - 1.0) + Abulk * EsatL + 3.0 * T7q
    T2v = Vgst2Vtm * (EsatL + 2.0 * T6q)
    disc = T1v * T1v - 2.0 * T0v * T2v
    if abs(T0v) < 1e-9:
        Vdsat_full_recomp = Vdsat_simple
    else:
        Vdsat_full_recomp = (T1v - math.sqrt(max(disc, 0.0))) / T0v

    return {
        "Leff": Leff, "Weff": Weff,
        "vsattemp": vsattemp, "u0temp": u0temp, "coxe": coxe, "Vtm": Vtm,
        "Vth": Vth, "Vgsteff": Vgsteff,
        "mueff": mueff, "Abulk": Abulk,
        "Esat": Esat, "EsatL": EsatL,
        "Vgst2Vtm": Vgst2Vtm,
        "Lambda": Lambda, "WVCoxRds": WVCoxRds,
        "Rds": Rds,
        "Vdsat_simple_formula": Vdsat_simple,
        "Vdsat_full_formula": Vdsat_full_recomp,
        "Vdsat_port_returned": Vdsat,
        "Vdseff": Vdseff, "Ids": Ids,
        # raw card params relevant to Abulk
        "a0": float(P.get("a0", 1.0)),
        "ags": float(P.get("ags", 0.0)),
        "b0": float(P.get("b0", 0.0)),
        "b1": float(P.get("b1", 0.0)),
        "keta": float(P.get("keta", 0.0)),
        "a1": a1_v, "a2": a2_v,
        "ua": float(P.get("ua", 0.0)),
        "ub": float(P.get("ub", 0.0)),
        "uc": float(P.get("uc", 0.0)),
        "ud": float(P.get("ud", 0.0)),
        "u0_scaled": float(P.get("u0", 0.0)),
        "vsat_scaled": float(P.get("vsat", 0.0)),
        "xj": float(P.get("xj", model.get("xj", 1.5e-7))),
        # Abulk decomposition
        "T9_xj": T9_xj, "T5_a": T5_a, "T7_a": T7_a, "T1_a": T1_a,
        "T2_a_(a0T5+b0/(W+b1))": T2_a,
        "T8_a_(ags·a0·T7)": T8_a,
        "Abulk0_recomp": Abulk0_recomp,
        "dAbulk_dVg_recomp": dAbulk_dVg_recomp,
        "Abulk_recomp": Abulk_recomp,
        "k1ox": k1ox, "k2ox": k2ox, "sqrtPhi": sqrtPhi, "Xdep0": Xdep0,
    }


def run_ngspice():
    deck = f"""* Vdsat bisect — sebas card, M1 NS-RAM
.param Nparam = 1.58
.param Citparam = 0
.param Voffparam = -0.1368
.param K2Par = -0.070435
.param toxn = 4e-009
.param vsatn = 80000
.include {CARD}

M1 d g s b NMOS L={L_DRAWN} W={W_DRAWN}
Vg g 0 {VGS}
Vd d 0 {VDS}
Vs s 0 0
Vb b 0 {VBS}

.option temp=27
.control
op
echo "PROBE_BEGIN"
print @m1[id]
print @m1[vth]
print @m1[vdsat]
print @m1[gm]
print @m1[gds]
print @m1[gmbs]
print @m1[vgs]
print @m1[vds]
print @m1[vbs]
print @m1[ueff]
print @m1[abulk]
print @m1[vgsteff]
print @m1[vdseff]
print @m1[esatl]
print @m1[rds]
echo "PROBE_END"
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


def parse_ngspice(out: str) -> dict[str, float]:
    res = {}
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("@m1[") and "=" in line:
            key = line.split("[")[1].split("]")[0]
            val_s = line.split("=", 1)[1].strip()
            try:
                res[key] = float(val_s)
            except ValueError:
                pass
    return res


def fmt(v):
    if isinstance(v, float):
        return f"{v:.6e}"
    return str(v)


def main():
    print("=" * 78)
    print("PyTorch port intermediates")
    print("=" * 78)
    py = py_compute()
    for k, v in py.items():
        print(f"  {k:28s} = {fmt(v)}")

    print()
    print("=" * 78)
    print("ngspice raw output")
    print("=" * 78)
    raw = run_ngspice()
    print(raw)
    ng = parse_ngspice(raw)

    print()
    print("=" * 78)
    print("Side-by-side")
    print("=" * 78)

    # Map ngspice probe -> port key
    pairs = [
        ("vth",      "Vth"),
        ("vgsteff",  "Vgsteff"),
        ("ueff",     "mueff"),
        ("abulk",    "Abulk"),
        ("esatl",    "EsatL"),
        ("vdsat",    "Vdsat_port_returned"),
        ("vdseff",   "Vdseff"),
        ("rds",      "Rds"),
        ("id",       "Ids"),
    ]
    rows = []
    print(f"  {'quantity':12s} {'ngspice':>16s} {'port':>16s} {'Δ':>14s}  {'rel%':>8s}")
    for ng_key, py_key in pairs:
        ng_v = ng.get(ng_key)
        py_v = py.get(py_key)
        if ng_v is None or py_v is None:
            print(f"  {ng_key:12s} ng={ng_v} port={py_v}")
            rows.append((ng_key, ng_v, py_v, None, None))
            continue
        diff = py_v - ng_v
        rel = (diff / ng_v) * 100.0 if ng_v != 0.0 else float("nan")
        print(f"  {ng_key:12s} {ng_v:16.6e} {py_v:16.6e} {diff:+14.4e}  {rel:+8.2f}")
        rows.append((ng_key, ng_v, py_v, diff, rel))

    # Save markdown table
    with open(OUT_DIR / "term_table.md", "w") as f:
        f.write("# z214 Vdsat term-by-term comparison\n\n")
        f.write(f"Card: {CARD}\nGeom: L={L_DRAWN}, W={W_DRAWN}\n")
        f.write(f"Bias: VGS={VGS}, VDS={VDS}, VBS={VBS}, T={T_C}C\n\n")
        f.write("| quantity | ngspice | port | Δ | rel% |\n")
        f.write("|---|---|---|---|---|\n")
        for q, n, p, d, r in rows:
            ns = f"{n:.6e}" if n is not None else "—"
            ps = f"{p:.6e}" if p is not None else "—"
            ds = f"{d:+.4e}" if d is not None else "—"
            rs = f"{r:+.2f}" if r is not None else "—"
            f.write(f"| {q} | {ns} | {ps} | {ds} | {rs} |\n")
        f.write("\n## Port-only intermediates\n\n")
        for k, v in py.items():
            f.write(f"- {k} = {fmt(v)}\n")

    return py, ng


if __name__ == "__main__":
    main()
