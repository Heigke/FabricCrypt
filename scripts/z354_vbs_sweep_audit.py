"""z354 — R-35: Vbs-dependence audit of pyport Vth vs ngspice Vth.

R-34 diagnostic at flagship (VG1=0.6,VG2=0.20,Vd=2.0):
  At ngspice OP (Vsint=0.41, Vbs=-0.14): pyport Ids = 6e-14
  but ngspice Ids ≈ 3.5e-11 — 3-decade gap. R-29 Vth/tox patch
  helped at the OLD (Vsint=0.38, Vbs=-0.115) probe but FAILED at
  the real ngspice basin (Vbs more negative ⇒ Vth higher ⇒ Ids back
  to subthreshold).

Hypothesis: pyport's Vbs-dependence of Vth is structurally wrong —
either body-effect K1eff·sqrtPhis(Vbs) term, or PHI value, or
Vbseff smoothing.

Method:
  Fix VG1=0.6, VD=2.0. Sweep Vbs ∈ {-0.20,-0.15,-0.10,-0.05,0.0,+0.05}
  (terminal — implemented by varying Vs while holding Vb=0 so that
   Vbs_M1 = Vb-Vs = -Vs; we directly drive Vs to set Vbs).
  Actually simplest: hold Vs=0, drive Vb in the netlist to obtain
  the target Vbs; for pyport call _eval_mosfet/compute_dc with the
  same Vgs,Vds,Vbs.

  For each Vbs:
    - pyport: compute_dc → Vth_py, plus log sqrtPhis, Vbseff, phi, k1, k2
    - ngspice: .op with `save @m1[vth]` → Vth_ng + vbs report

Outputs:
  results/z354_vbs_audit/vth_vs_vbs_compare.json
  results/z354_vbs_audit/verdict.md
"""
from __future__ import annotations
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")

import sys, json, re, math, subprocess, importlib.util, inspect
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "nsram"))

OUT = ROOT / "results/z354_vbs_audit"
OUT.mkdir(parents=True, exist_ok=True)

import torch
torch.set_default_dtype(torch.float64)

from nsram.bsim4_port.model_card import BSIM4Model
from nsram.bsim4_port.geometry import Geometry
from nsram.bsim4_port.temp import compute_size_dep
from nsram.bsim4_port.dc import compute_dc

# ============== bias config: at M1 OP, flagship (VG1=0.6, VD=2.0) =========
# At converged ngspice OP: Vsint≈0.41, Vb≈0.27
# M1 actually sees: Vgs = VG1-Vsint ≈ 0.19, Vds = Vd-Vsint ≈ 1.59, Vbs = Vb-Vsint ≈ -0.14.
# So to probe the subthreshold regime where the 3-dec Ids gap appears,
# we MUST run at Vgs≈0.19 (NOT Vgs=0.6). Use the netlist terminal mapping:
#   Vs=0.41, Vd=2.0, Vg=0.6  ⇒ Vgs=0.19, Vds=1.59
#   Sweep Vb to vary Vbs = Vb - Vs.
VG_TERM = 0.6      # cell VG1
VD_TERM = 2.0      # cell Vd
VS_TERM = 0.41     # cell Vsint
# Sweep Vbs ∈ {-0.20,...,+0.05}; Vb terminal = Vs + Vbs
VBS_LIST = [-0.20, -0.15, -0.10, -0.05, 0.0, +0.05]
# For pyport: Vgs=VG_TERM-VS_TERM, Vds=VD_TERM-VS_TERM, Vbs as swept
VG = VG_TERM - VS_TERM
VD = VD_TERM - VS_TERM

# In netlist we drive terminals Vd,Vg,Vs,Vb with Vs=0 so Vbs = Vb literally.
# Pyport call: Vgs=Vg-Vs=VG, Vds=Vd-Vs=VD, Vbs as swept.


# =============================== pyport ==================================
def build_M1():
    sp = importlib.util.spec_from_file_location("v1", ROOT / "scripts/z96_narma10_pilot.py")
    v1 = importlib.util.module_from_spec(sp); sp.loader.exec_module(v1)
    text_M1 = (ROOT / "data/sebas_2026_04_22/M1_130DNWFB_LALPHA0_FIX.txt").read_text()
    M1 = BSIM4Model.from_spice(text_M1, model_type="nmos")
    v1.f.patch_model_values(M1, type_n=True)
    M1._values["voff"] = M1._values.get("voff", -0.1368)
    geom = Geometry(L=0.13e-6, W=1e-6)
    sd = compute_size_dep(M1, geom, T_C=27.0)
    return M1, sd


def trace_compute_dc(M1, sd, Vgs, Vds, Vbs):
    """Run compute_dc + capture all local scalar tensors at last line."""
    from nsram.bsim4_port import dc as dc_mod
    snapshot = {}
    target_func_name = "compute_dc"
    target_filename  = inspect.getfile(dc_mod)

    def _capture(frame):
        for name, val in list(frame.f_locals.items()):
            try:
                if isinstance(val, torch.Tensor):
                    if val.numel() == 1:
                        snapshot[name] = float(val.detach().cpu().item())
                elif isinstance(val, (int, float)) and not isinstance(val, bool):
                    snapshot[name] = float(val)
            except Exception:
                pass

    def tracer(frame, event, arg):
        if frame.f_code.co_filename != target_filename:
            return None
        if frame.f_code.co_name != target_func_name:
            return None
        if event in ("call", "line", "return"):
            _capture(frame)
        return tracer

    Vgs_t = torch.tensor(Vgs, dtype=torch.float64)
    Vds_t = torch.tensor(Vds, dtype=torch.float64)
    Vbs_t = torch.tensor(Vbs, dtype=torch.float64)
    sys.settrace(tracer)
    try:
        res = compute_dc(M1, sd, Vgs=Vgs_t, Vds=Vds_t, Vbs=Vbs_t)
    finally:
        sys.settrace(None)

    out = {
        "Ids": float(res.Ids),
        "Vth": float(res.Vth),
        "Vgsteff": float(res.Vgsteff),
        "Vdsat": float(res.Vdsat),
        "n": float(res.n),
    }
    # Intermediates from compute_dc locals:
    for k in ("Vbseff", "Phis", "sqrtPhis", "Xdep", "Delt_vth",
              "DIBL_Sft", "Lpe_Vb", "Tlpe1", "Vth_NarrowW",
              "k1", "k2", "k1ox", "k2ox"):
        if k in snapshot:
            out[k] = snapshot[k]
    # sd intermediates
    out["sd_phi"] = float(sd.phi)
    out["sd_sqrtPhi"] = float(sd.sqrtPhi)
    out["sd_k1ox"] = float(sd.k1ox)
    out["sd_k2ox"] = float(sd.k2ox)
    out["sd_vth0_T"] = float(sd.vth0_T)
    out["sd_vbsc"] = float(sd.vbsc)
    out["sd_vbi"] = float(sd.vbi)
    # raw model params after patch
    out["model_k1"]  = float(M1._values.get("k1", float("nan")))
    out["model_k2"]  = float(M1._values.get("k2", float("nan")))
    out["model_phin"]= float(M1._values.get("phin", 0.0))
    out["model_ndep"]= float(M1._values.get("ndep", float("nan")))
    return out


# =============================== ngspice =================================
def build_deck(Vbs: float) -> str:
    Vb_term = VS_TERM + Vbs
    return f""".title z354 M1 Vth Vbs sweep, Vgs={VG} Vds={VD} Vbs={Vbs}
.include "{ROOT / 'data/sebas_2026_04_22/M2_130bulkNSRAM.txt'}"
.include "{ROOT / 'data/sebas_2026_04_22/M1_130DNWFB_LALPHA0_FIX.txt'}"
Vd  d 0 DC {VD_TERM}
Vg  g 0 DC {VG_TERM}
Vs  s 0 DC {VS_TERM}
Vb  b 0 DC {Vb_term}
M1  d g s b NMOSdnwfb L=0.13u W=1u
.options gmin=1e-15 abstol=1e-14 reltol=1e-3 itl1=300
.control
op
print @m1[id] @m1[vth] @m1[vdsat] @m1[vbs] @m1[vgs] @m1[vds]
print @m1[gm] @m1[gmbs] @m1[gds]
quit
.endc
.end
"""


_RE_EQ = re.compile(r"(@?\w+(?:\[\w+\])?)\s*=\s*([-+]?\d+\.?\d*[eE]?[-+]?\d*)")

def run_ngspice(Vbs: float) -> dict:
    deck = build_deck(Vbs)
    deck_path = OUT / f"deck_vbs_{Vbs:+.2f}.sp"
    log_path  = OUT / f"ngspice_vbs_{Vbs:+.2f}.log"
    deck_path.write_text(deck)
    proc = subprocess.run(["ngspice", "-b", str(deck_path)],
                          capture_output=True, text=True, timeout=120)
    text = proc.stdout + "\n--- STDERR ---\n" + proc.stderr
    log_path.write_text(text)
    d = {}
    for m in _RE_EQ.finditer(text):
        try:
            d[m.group(1).lower()] = float(m.group(2))
        except ValueError:
            pass
    return d


# ================================ main ===================================
def main():
    print(f"[z354] R-35 Vbs audit at VG={VG} VD={VD}", flush=True)
    print(f"[z354] Vbs grid: {VBS_LIST}", flush=True)

    M1, sd = build_M1()
    print(f"[z354] M1 built: k1={M1._values.get('k1'):.4f} "
          f"k2={M1._values.get('k2'):.4f} "
          f"phin={M1._values.get('phin',0.0):.4f} "
          f"ndep={M1._values.get('ndep'):.3e}", flush=True)
    print(f"[z354] sd.phi={sd.phi:.5f} sd.sqrtPhi={sd.sqrtPhi:.5f} "
          f"sd.k1ox={sd.k1ox:.4f} sd.k2ox={sd.k2ox:.4f}", flush=True)

    rows = []
    for Vbs in VBS_LIST:
        py = trace_compute_dc(M1, sd, Vgs=VG, Vds=VD, Vbs=Vbs)
        ng = run_ngspice(Vbs)
        ng_vth = ng.get("@m1[vth]", ng.get("vth"))
        ng_ids = ng.get("@m1[id]", ng.get("id"))
        ng_vbs = ng.get("@m1[vbs]", ng.get("vbs"))
        d_vth = (py["Vth"] - ng_vth) if ng_vth is not None else None
        row = {
            "Vbs_set": Vbs,
            "ng_vbs_actual": ng_vbs,
            "py_Vth": py["Vth"],
            "ng_Vth": ng_vth,
            "delta_Vth_mV": (d_vth * 1000.0) if d_vth is not None else None,
            "py_Ids": py["Ids"],
            "ng_Ids": ng_ids,
            "log10_ratio_Ids": (math.log10(abs(py["Ids"]/ng_ids))
                                if ng_ids and py["Ids"] and ng_ids != 0 else None),
            "py_Vbseff": py.get("Vbseff"),
            "py_sqrtPhis": py.get("sqrtPhis"),
            "py_Phis": py.get("Phis"),
            "py_Delt_vth": py.get("Delt_vth"),
            "py_DIBL_Sft": py.get("DIBL_Sft"),
            "py_Tlpe1": py.get("Tlpe1"),
            "py_Vgsteff": py["Vgsteff"],
        }
        rows.append(row)
        print(f"[z354] Vbs={Vbs:+.2f}  py_Vth={py['Vth']:.5f}  "
              f"ng_Vth={ng_vth if ng_vth is None else f'{ng_vth:.5f}'}  "
              f"ΔVth={row['delta_Vth_mV']:.1f}mV  "
              f"py_Vbseff={py.get('Vbseff'):.4f}  "
              f"py_sqrtPhis={py.get('sqrtPhis'):.4f}",
              flush=True)

    # Compute slope dVth/dVbs for both
    def _slope(xs, ys):
        xs = [x for x, y in zip(xs, ys) if y is not None]
        ys = [y for y in ys if y is not None]
        if len(xs) < 2: return None
        n = len(xs)
        mean_x = sum(xs)/n; mean_y = sum(ys)/n
        num = sum((x-mean_x)*(y-mean_y) for x,y in zip(xs,ys))
        den = sum((x-mean_x)**2 for x in xs)
        return num/den if den else None

    vbs_xs = [r["Vbs_set"] for r in rows]
    py_slope = _slope(vbs_xs, [r["py_Vth"] for r in rows])
    ng_slope = _slope(vbs_xs, [r["ng_Vth"] for r in rows])

    summary = {
        "bias": {"VG": VG, "VD": VD, "Vbs_grid": VBS_LIST},
        "M1": {
            "k1": float(M1._values.get("k1")),
            "k2": float(M1._values.get("k2")),
            "phin": float(M1._values.get("phin", 0.0)),
            "ndep": float(M1._values.get("ndep")),
            "lpe0": float(M1._values.get("lpe0", float("nan"))),
            "toxe": float(M1._values.get("toxe", float("nan"))),
        },
        "sd": {
            "phi": float(sd.phi),
            "sqrtPhi": float(sd.sqrtPhi),
            "k1ox": float(sd.k1ox),
            "k2ox": float(sd.k2ox),
            "vth0_T": float(sd.vth0_T),
            "vbsc": float(sd.vbsc),
            "vbi": float(sd.vbi),
        },
        "rows": rows,
        "py_dVth_dVbs": py_slope,
        "ng_dVth_dVbs": ng_slope,
        "delta_slope_pyport_minus_ngspice": (py_slope - ng_slope)
            if py_slope is not None and ng_slope is not None else None,
    }

    (OUT / "vth_vs_vbs_compare.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[z354] py dVth/dVbs slope: {py_slope}", flush=True)
    print(f"[z354] ng dVth/dVbs slope: {ng_slope}", flush=True)
    if py_slope is not None and ng_slope is not None:
        print(f"[z354] slope delta (py-ng) = {py_slope - ng_slope:+.4f} V/V", flush=True)
    print(f"[z354] wrote {OUT/'vth_vs_vbs_compare.json'}", flush=True)


if __name__ == "__main__":
    main()
