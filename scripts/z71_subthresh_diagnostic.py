"""z71 — Surgical block-level diagnostic for subthreshold + cold-T + reverse-Vbs error.

Goal: Find which equation block in `nsram/bsim4_port/dc.py` causes the 109% worst-case
relative-error vs ngspice at:
    Vgs=0.375, Vds=0.05, Vbs=-0.6, T=-20C, L=130n, W=10u (Sebas card)

Steps:
  S1: Block-level table at worst-case point (Python intermediates vs ngspice OP)
  S2: Vbs sweep at fixed Vgs/Vds/T (test body-bias machinery)
  S3: T sweep at fixed Vgs/Vds/Vbs (test temperature machinery)

Writes  results/bsim4_port_validation/subthresh_diagnostic.md
Budget: <=30 ngspice calls (S1=1, S2=5, S3=6 -> 12), no model edits.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import math
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "bsim4_port"))
sys.path.insert(0, str(ROOT / "nsram"))

from spice_oracle import Bias, Geometry as SpiceGeom, Sweep, run_op  # type: ignore
from nsram.bsim4_port.dc import compute_dc  # type: ignore
from nsram.bsim4_port.geometry import Geometry  # type: ignore
from nsram.bsim4_port.model_card import BSIM4Model  # type: ignore
from nsram.bsim4_port.temp import compute_size_dep  # type: ignore

GOLD = ROOT / "results/bsim4_port_validation/gold/sebas130.json"
OUT_MD = ROOT / "results/bsim4_port_validation/subthresh_diagnostic.md"

gold = json.loads(GOLD.read_text())
model_text = gold["model_text"]
model = BSIM4Model.from_spice(model_text, model_type="nmos")
model_name = "NMOS"


def worst_geom():
    return Geometry(L=1.3e-7, W=1.0e-5, NF=1)


def py_eval(Vgs, Vds, Vbs, T_C, L=1.3e-7, W=1.0e-5):
    g = Geometry(L=L, W=W, NF=1)
    sd = compute_size_dep(model, g, T_C=T_C)
    Vgs_t = torch.tensor([Vgs], dtype=torch.float64)
    Vds_t = torch.tensor([Vds], dtype=torch.float64)
    Vbs_t = torch.tensor([Vbs], dtype=torch.float64)
    res = compute_dc(model, sd, Vgs=Vgs_t, Vds=Vds_t, Vbs=Vbs_t)
    out = dict(
        Ids=float(res.Ids),
        Vth=float(res.Vth),
        Vgsteff=float(res.Vgsteff),
        Vdsat=float(res.Vdsat),
        n=float(res.n),
        Abulk=float(res.Abulk),
        mueff=float(res.mueff),
    )
    # Internals from sd / model_ctx
    out["sd_phi"] = float(sd.phi)
    out["sd_sqrtPhi"] = float(sd.sqrtPhi)
    out["sd_Xdep0"] = float(sd.Xdep0)
    out["sd_vbi"] = float(sd.vbi)
    out["sd_k1ox"] = float(sd.k1ox)
    out["sd_k2ox"] = float(sd.k2ox)
    out["sd_vth0_T"] = float(sd.vth0_T)
    out["ctx_Vtm"] = float(sd.model_ctx.vtm)
    out["ctx_Tnom"] = float(sd.model_ctx.Tnom)
    out["ctx_Temp"] = float(sd.model_ctx.Temp)
    out["ctx_TRatio"] = float(sd.model_ctx.TRatio)
    out["ctx_factor1"] = float(sd.model_ctx.factor1)
    out["ctx_coxe"] = float(sd.model_ctx.coxe)
    out["sd_voffcbn"] = float(sd.voffcbn)
    out["sd_mstar"] = float(sd.mstar)
    out["sd_cdep0"] = float(sd.cdep0)
    return out


def ng_eval(Vgs, Vds, Vbs, T_C):
    bias = Bias(Vd=Vds, Vg=Vgs, Vs=0.0, Vb=Vbs)
    sg = SpiceGeom(L=1.3e-7, W=1.0e-5, NF=1)
    return run_op(model_text, model_name, sg, bias, temp_C=T_C)


def relerr(p, n):
    if n is None or abs(n) < 1e-30:
        return float("nan")
    return (p - n) / n


def fmt(x):
    if x is None:
        return "  -  "
    if isinstance(x, float):
        if abs(x) > 0 and (abs(x) < 1e-3 or abs(x) > 1e3):
            return f"{x:.4e}"
        return f"{x:+.6f}"
    return str(x)


# ============================================================================
# Step 1: Block-level table at worst-case point
# ============================================================================
WC = dict(Vgs=0.375, Vds=0.05, Vbs=-0.6, T_C=-20.0)
print(f"=== STEP 1: worst-case point {WC} ===")
py = py_eval(**WC)
ng = ng_eval(**WC)
print("ngspice OP keys:", sorted(ng.keys()))

# Map ng keys (lowercase) to py keys we care about
compare = [
    ("Ids",      "Ids",  ng.get("ids",      ng.get("id"))),
    ("Vth",      "Vth",  ng.get("vth",      ng.get("von"))),
    ("Vdsat",    "Vdsat",ng.get("vdsat")),
    ("gm",       None,   ng.get("gm")),
    ("gds",      None,   ng.get("gds")),
    ("gmbs",     None,   ng.get("gmbs")),
]

# Step1 results
S1 = []
for label, py_key, ng_v in compare:
    py_v = py[py_key] if py_key else None
    err = relerr(py_v, ng_v) if (py_v is not None and ng_v is not None) else None
    S1.append((label, py_v, ng_v, err))
    print(f"  {label:8s}  py={fmt(py_v)}   ng={fmt(ng_v)}   relerr={fmt(err)}")

# Print Python intermediates that have no ng analog
print("--- Python-only intermediates ---")
for k in ("n", "Vgsteff", "Abulk", "mueff", "sd_phi", "sd_sqrtPhi", "sd_Xdep0",
          "sd_vbi", "sd_k1ox", "sd_k2ox", "sd_vth0_T", "ctx_Vtm",
          "ctx_TRatio", "ctx_factor1", "ctx_coxe", "sd_voffcbn", "sd_mstar"):
    print(f"  {k:16s} = {fmt(py[k])}")


# ============================================================================
# Step 2: Vbs sweep at fixed Vgs=0.375, Vds=0.05, T=-20C
# ============================================================================
print(f"\n=== STEP 2: Vbs sweep (Vgs=0.375, Vds=0.05, T=-20C) ===")
S2 = []
for Vbs in (-0.6, -0.4, -0.2, 0.0, 0.2):
    p = py_eval(Vgs=0.375, Vds=0.05, Vbs=Vbs, T_C=-20.0)
    n = ng_eval(Vgs=0.375, Vds=0.05, Vbs=Vbs, T_C=-20.0)
    e_ids = relerr(p["Ids"], n.get("ids"))
    e_vth = relerr(p["Vth"], n.get("vth"))
    S2.append((Vbs, p["Ids"], n.get("ids"), e_ids, p["Vth"], n.get("vth"), e_vth, p["n"]))
    print(f"  Vbs={Vbs:+.2f}  Ids_py={p['Ids']:.4e} Ids_ng={n.get('ids',0):.4e}  rerr_Ids={e_ids:+.4f}"
          f"  Vth_py={p['Vth']:+.4f} Vth_ng={n.get('vth',0):+.4f}  rerr_Vth={e_vth:+.4f}  n={p['n']:.3f}")


# ============================================================================
# Step 3: T sweep at fixed Vgs=0.375, Vds=0.05, Vbs=-0.6
# ============================================================================
print(f"\n=== STEP 3: T sweep (Vgs=0.375, Vds=0.05, Vbs=-0.6) ===")
S3 = []
for T_C in (-40.0, -20.0, 0.0, 27.0, 75.0, 125.0):
    p = py_eval(Vgs=0.375, Vds=0.05, Vbs=-0.6, T_C=T_C)
    n = ng_eval(Vgs=0.375, Vds=0.05, Vbs=-0.6, T_C=T_C)
    e_ids = relerr(p["Ids"], n.get("ids"))
    e_vth = relerr(p["Vth"], n.get("vth"))
    S3.append((T_C, p["Ids"], n.get("ids"), e_ids, p["Vth"], n.get("vth"), e_vth, p["n"], p["ctx_Vtm"]))
    print(f"  T={T_C:+.0f}C  Ids_py={p['Ids']:.4e} Ids_ng={n.get('ids',0):.4e}  rerr_Ids={e_ids:+.4f}"
          f"  Vth_py={p['Vth']:+.4f} Vth_ng={n.get('vth',0):+.4f}  rerr_Vth={e_vth:+.4f}  n={p['n']:.3f}  Vtm={p['ctx_Vtm']:.5f}")


# ============================================================================
# Step 3b: control sweep — same T sweep but Vbs=0 (isolates body-bias×T coupling)
# ============================================================================
print(f"\n=== STEP 3b CONTROL: T sweep (Vbs=0) ===")
S3b = []
for T_C in (-40.0, -20.0, 0.0, 27.0, 75.0, 125.0):
    p = py_eval(Vgs=0.375, Vds=0.05, Vbs=0.0, T_C=T_C)
    n = ng_eval(Vgs=0.375, Vds=0.05, Vbs=0.0, T_C=T_C)
    e_ids = relerr(p["Ids"], n.get("ids"))
    e_vth = relerr(p["Vth"], n.get("vth"))
    S3b.append((T_C, e_ids, e_vth, p["n"]))
    print(f"  T={T_C:+.0f}C  rerr_Ids={e_ids:+.4f}  rerr_Vth={e_vth:+.4f}  n={p['n']:.3f}")


# ============================================================================
# Persist markdown report
# ============================================================================
def md_table(headers, rows):
    out = "| " + " | ".join(headers) + " |\n"
    out += "| " + " | ".join("---" for _ in headers) + " |\n"
    for r in rows:
        out += "| " + " | ".join(fmt(v) for v in r) + " |\n"
    return out


lines = []
lines.append("# z71 — Subthreshold + Cold-T + Reverse-Vbs Diagnostic\n")
lines.append(f"**Bias point**: Vgs=0.375, Vds=0.05, Vbs=-0.6, T=-20°C, L=130n, W=10µ; Sebas130 card.\n")
lines.append("## Step 1 — Block-level comparison (worst-case point)\n")
lines.append(md_table(
    ["Quantity", "Python", "ngspice", "rel_err"],
    [(lbl, p, n, e) for (lbl, p, n, e) in S1],
))
lines.append("\n### Python-only intermediates\n")
intern_rows = []
for k in ("n", "Vgsteff", "Abulk", "mueff", "sd_phi", "sd_sqrtPhi", "sd_Xdep0",
         "sd_vbi", "sd_k1ox", "sd_k2ox", "sd_vth0_T", "ctx_Vtm",
         "ctx_TRatio", "ctx_factor1", "ctx_coxe", "sd_voffcbn", "sd_mstar"):
    intern_rows.append((k, py[k]))
lines.append(md_table(["Var", "Value"], intern_rows))

lines.append("\n## Step 2 — Vbs sweep (Vgs=0.375, Vds=0.05, T=-20°C)\n")
lines.append(md_table(
    ["Vbs", "Ids_py", "Ids_ng", "rel_err_Ids", "Vth_py", "Vth_ng", "rel_err_Vth", "n_py"],
    S2,
))

lines.append("\n## Step 3 — T sweep (Vgs=0.375, Vds=0.05, Vbs=-0.6)\n")
lines.append(md_table(
    ["T_C", "Ids_py", "Ids_ng", "rel_err_Ids", "Vth_py", "Vth_ng", "rel_err_Vth", "n_py", "Vtm"],
    S3,
))

lines.append("\n## Step 3b CONTROL — T sweep with Vbs=0\n")
lines.append(md_table(
    ["T_C", "rel_err_Ids", "rel_err_Vth", "n_py"],
    S3b,
))

OUT_MD.parent.mkdir(parents=True, exist_ok=True)
OUT_MD.write_text("\n".join(lines))
print(f"\nWrote {OUT_MD}")
