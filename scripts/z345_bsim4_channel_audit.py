"""z345 — R-29: Pyport BSIM4 channel intermediate audit at M1 OP.

Goal: pyport Ids_M1 reads ~4e-15 at the converged 2T OP while ngspice
reads ~1.5e-11 (3-decade gap, see R-25). Both stages were instrumented
externally; this script isolates the discrepancy by:

  1. Picking the M1 OP from R-25: Vg=0.6, Vd=2.0, Vs=0.382, Vb=0.267
     ⇒ Vgs=0.218, Vds=1.618, Vbs=-0.115
  2. Running pyport's compute_dc() under sys.settrace to capture *every*
     local intermediate at function return.
  3. Running ngspice on a single-transistor probe deck with hard-driven
     source/body voltages, reading id, vth, vdsat, vbs, vgs, vds, isub,
     gm, gmbs, gds, ueff, igidl, igisl, igb.
  4. Diffing matching variables and pin-pointing the upstream
     intermediate with the biggest fractional gap.

Outputs results/z345_bsim4_channel_audit/{term_by_term.json,verdict.md}.

Per CLAUDE.md: HSA override applied. ikaros venv. No subagents.
"""
from __future__ import annotations
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")

import sys, json, math, re, subprocess, importlib.util, inspect
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "nsram"))

OUT = ROOT / "results/z345_bsim4_channel_audit"
OUT.mkdir(parents=True, exist_ok=True)

# ---------- OP (R-25 converged @ VG1=0.6, VG2=0.20, Vd=2.0) ---------- #
VG1, VG2, VD = 0.6, 0.20, 2.0
VSINT = 0.382   # ngspice converged Vsint
VB    = 0.267   # ngspice converged Vb
# M1 terminal mapping: Vg=VG1, Vd=VD, Vs=Vsint, Vb=Vb
VGS = VG1 - VSINT   # 0.218
VDS = VD  - VSINT   # 1.618
VBS = VB  - VSINT   # -0.115


# ============================ ngspice probe ============================ #
DECK = f""".title z345 M1 single-transistor probe at Vgs={VGS:.4f} Vds={VDS:.4f} Vbs={VBS:.4f}

* Pull .param toxn / lintn / wintn / vth0n / etc. from the M2 multi-line .param block.
.include "{ROOT / 'data/sebas_2026_04_22/M2_130bulkNSRAM.txt'}"
.include "{ROOT / 'data/sebas_2026_04_22/M1_130DNWFB_LALPHA0_FIX.txt'}"

* Drive terminals directly with voltage sources at the R-25 OP.
Vd      d   0   DC {VD}
Vg      g   0   DC {VG1}
Vs      s   0   DC {VSINT}
Vb      b   0   DC {VB}

M1  d g s b NMOSdnwfb L=0.13u W=1u

.options gmin=1e-15 abstol=1e-14 reltol=1e-3 itl1=300

.control
op
print @m1[id]
print @m1[vds] @m1[vgs] @m1[vbs]
print @m1[vth] @m1[vdsat]
print @m1[gm]  @m1[gds] @m1[gmbs]
print @m1[isub] @m1[igidl] @m1[igisl] @m1[igb]
print @m1[ibd]  @m1[ibs]  @m1[igd]   @m1[igs]
quit
.endc
.end
"""

_RE_EQ = re.compile(r"(@?\w+(?:\[\w+\])?)\s*=\s*([-+]?\d+\.?\d*[eE]?[-+]?\d*)")

def run_ngspice() -> dict:
    deck_path = OUT / "deck.sp"
    log_path  = OUT / "ngspice.log"
    deck_path.write_text(DECK)
    proc = subprocess.run(["ngspice", "-b", str(deck_path)],
                          capture_output=True, text=True, timeout=120)
    text = proc.stdout + "\n--- STDERR ---\n" + proc.stderr
    log_path.write_text(text)
    print(f"ngspice rc={proc.returncode}")
    d = {}
    for m in _RE_EQ.finditer(text):
        try:
            d[m.group(1).lower()] = float(m.group(2))
        except ValueError:
            pass
    return d


# ============================ pyport probe ============================ #
def build_M1():
    """Build M1 BSIM4Model + SizeDependParam exactly like z343 does."""
    sp = importlib.util.spec_from_file_location("v1", ROOT / "scripts/z96_narma10_pilot.py")
    v1 = importlib.util.module_from_spec(sp); sp.loader.exec_module(v1)
    from nsram.bsim4_port.model_card import BSIM4Model
    from nsram.bsim4_port.geometry import Geometry
    from nsram.bsim4_port.temp import compute_size_dep
    text_M1 = (ROOT / "data/sebas_2026_04_22/M1_130DNWFB_LALPHA0_FIX.txt").read_text()
    M1 = BSIM4Model.from_spice(text_M1, model_type="nmos")
    v1.f.patch_model_values(M1, type_n=True)
    M1._values["voff"] = M1._values.get("voff", -0.1368)
    geom = Geometry(L=0.13e-6, W=1e-6)
    sd = compute_size_dep(M1, geom, T_C=27.0)
    return M1, sd


def trace_compute_dc(M1, sd, Vgs, Vds, Vbs) -> tuple[dict, dict]:
    """Run compute_dc with a sys.settrace recorder that snapshots every
    local fp64 tensor in the function's frame just BEFORE it returns."""
    import torch
    from nsram.bsim4_port import dc as dc_mod

    snapshot = {}
    target_func_name = "compute_dc"
    target_filename  = inspect.getfile(dc_mod)

    def _capture(frame):
        # Pull every local into a python scalar (if scalar tensor or float)
        for name, val in list(frame.f_locals.items()):
            try:
                if isinstance(val, torch.Tensor):
                    if val.numel() == 1:
                        snapshot[name] = float(val.detach().cpu().item())
                    elif val.numel() <= 4:
                        snapshot[name] = [float(x) for x in val.detach().cpu().flatten()]
                elif isinstance(val, (int, float)) and not isinstance(val, bool):
                    snapshot[name] = float(val)
            except Exception:
                pass

    def tracer(frame, event, arg):
        # only trace inside compute_dc
        if frame.f_code.co_filename != target_filename:
            return None
        if frame.f_code.co_name != target_func_name:
            return None
        # On every 'line' event keep updating snapshot so we capture the
        # state right before the final return.
        if event in ("call", "line", "return"):
            _capture(frame)
        return tracer

    Vgs_t = torch.tensor(Vgs, dtype=torch.float64)
    Vds_t = torch.tensor(Vds, dtype=torch.float64)
    Vbs_t = torch.tensor(Vbs, dtype=torch.float64)

    sys.settrace(tracer)
    try:
        res = dc_mod.compute_dc(M1, sd, Vgs=Vgs_t, Vds=Vds_t, Vbs=Vbs_t)
    finally:
        sys.settrace(None)

    result = {
        "Ids":     float(res.Ids),
        "Vth":     float(res.Vth),
        "Vgsteff": float(res.Vgsteff),
        "Vdsat":   float(res.Vdsat),
        "Vdseff":  float(res.Vdseff),
        "Abulk":   float(res.Abulk),
        "n":       float(res.n),
        "mueff":   float(res.mueff),
        "Idsa_Vdseff_preSCBE": float(res.Idsa) if res.Idsa is not None else None,
        "Vgs_eff": float(res.Vgs_eff) if res.Vgs_eff is not None else None,
        "Vbseff":  float(res.Vbseff)  if res.Vbseff  is not None else None,
    }
    return result, snapshot


# ============================ main ============================ #
def main():
    print(f"=== z345 BSIM4 channel audit ===")
    print(f"OP: Vg={VG1}, Vd={VD}, Vs={VSINT}, Vb={VB}")
    print(f"    Vgs={VGS:.4f}, Vds={VDS:.4f}, Vbs={VBS:.4f}")

    # --- ngspice
    ng = run_ngspice()
    ng_keys = [
        "@m1[id]", "@m1[vth]", "@m1[vdsat]", "@m1[vbs]", "@m1[vgs]", "@m1[vds]",
        "@m1[gm]", "@m1[gds]", "@m1[gmbs]",
        "@m1[isub]", "@m1[igidl]", "@m1[igisl]", "@m1[igb]",
        "@m1[ibd]", "@m1[ibs]",
    ]
    ng_vals = {k: ng.get(k, float("nan")) for k in ng_keys}
    print(f"ngspice id={ng_vals['@m1[id]']:.4e}  vth={ng_vals['@m1[vth]']:.5f}  "
          f"vdsat={ng_vals['@m1[vdsat]']:.5f}  isub={ng_vals['@m1[isub]']:.3e}")

    # --- pyport
    M1, sd = build_M1()
    print(f"pyport M1 alpha0={M1._values.get('alpha0')}  lalpha0={M1._values.get('lalpha0')}")
    py_top, py_locals = trace_compute_dc(M1, sd, VGS, VDS, VBS)
    print(f"pyport Ids={py_top['Ids']:.4e}  Vth={py_top['Vth']:.5f}  "
          f"Vdsat={py_top['Vdsat']:.5f}  Vgsteff={py_top['Vgsteff']:.5f}  "
          f"Vdseff={py_top['Vdseff']:.5f}  Abulk={py_top['Abulk']:.4f}")

    # --- map nominal corresponding variables ---
    # ngspice exposes:  id, vth (=Von), vdsat, vbs, vgs, vds.
    # pyport intermediates of interest:
    interesting = [
        "Vgs", "Vds", "Vbs", "Vbseff",
        "Vgs_eff", "Vgst", "Vgsteff", "Vth",
        "Vdsat", "Vdseff", "diffVds",
        "Abulk", "Abulk0", "mueff", "u0temp", "Esat", "EsatL",
        "Coxeff", "CoxeffWovL", "beta",
        "fgche1", "fgche2", "gche", "Idl", "Idsa", "Ids_chan", "Ids",
        "Rds", "WVCoxRds", "Vasat", "Va", "VACLM", "VADIBL", "VADITS", "VASCBE",
        "n", "ExpVgst", "T10v", "T9v",
    ]
    py_dump = {k: py_locals.get(k) for k in interesting}

    # Build side-by-side
    comparisons = {
        "Vgs":   (py_locals.get("Vgs"),   ng_vals["@m1[vgs]"]),
        "Vds":   (py_locals.get("Vds"),   ng_vals["@m1[vds]"]),
        "Vbs":   (py_locals.get("Vbseff", py_locals.get("Vbs")), ng_vals["@m1[vbs]"]),
        "Vth":   (py_top["Vth"],          ng_vals["@m1[vth]"]),
        "Vdsat": (py_top["Vdsat"],        ng_vals["@m1[vdsat]"]),
        "Ids":   (py_top["Ids"],          ng_vals["@m1[id]"]),
        "Isub":  (None,                    ng_vals["@m1[isub]"]),  # pyport stored elsewhere
        "gm":    (None,                    ng_vals["@m1[gm]"]),
        "gds":   (None,                    ng_vals["@m1[gds]"]),
        "Igidl": (None,                    ng_vals["@m1[igidl]"]),
        "Igisl": (None,                    ng_vals["@m1[igisl]"]),
        "Igb":   (None,                    ng_vals["@m1[igb]"]),
    }

    rows = []
    for name, (py_v, ng_v) in comparisons.items():
        if py_v is None or (isinstance(ng_v, float) and math.isnan(ng_v)):
            rows.append({"name": name, "pyport": py_v, "ngspice": ng_v,
                         "ratio_py_over_ng": None, "abs_log_ratio": None})
            continue
        if abs(ng_v) < 1e-40 and abs(py_v) < 1e-40:
            ratio, alr = 1.0, 0.0
        elif abs(ng_v) < 1e-40:
            ratio, alr = float("inf"), float("inf")
        else:
            ratio = py_v / ng_v
            alr = abs(math.log10(abs(ratio))) if ratio != 0 else float("inf")
        rows.append({"name": name, "pyport": py_v, "ngspice": ng_v,
                     "ratio_py_over_ng": ratio, "abs_log_ratio": alr})

    rows_sorted = sorted(rows, key=lambda r: (r["abs_log_ratio"] is None,
                                              -(r["abs_log_ratio"] or 0.0)))

    print("\n--- compare (pyport vs ngspice) ---")
    print(f"{'name':<10s}{'pyport':>16s}{'ngspice':>16s}{'ratio':>14s}{'|log10|':>10s}")
    for r in rows_sorted:
        py_s = f"{r['pyport']:.4e}" if isinstance(r['pyport'], float) else "n/a"
        ng_s = f"{r['ngspice']:.4e}" if isinstance(r['ngspice'], float) and not math.isnan(r['ngspice']) else "n/a"
        if r["ratio_py_over_ng"] is None:
            rr_s, al_s = "n/a", "n/a"
        elif math.isinf(r["ratio_py_over_ng"]):
            rr_s, al_s = "inf", "inf"
        else:
            rr_s = f"{r['ratio_py_over_ng']:.4g}"
            al_s = f"{r['abs_log_ratio']:.3f}"
        print(f"{r['name']:<10s}{py_s:>16s}{ng_s:>16s}{rr_s:>14s}{al_s:>10s}")

    print("\n--- pyport intermediates (selected) ---")
    for k in interesting:
        v = py_locals.get(k)
        if isinstance(v, float):
            print(f"  {k:<14s} = {v:.6e}")
        elif v is not None:
            print(f"  {k:<14s} = {v}")

    # ---- write outputs ---
    out = {
        "op": {"VG1": VG1, "Vd": VD, "Vsint": VSINT, "Vb": VB,
                "Vgs": VGS, "Vds": VDS, "Vbs": VBS},
        "ngspice": ng_vals,
        "pyport_top": py_top,
        "pyport_intermediates": {k: py_locals.get(k) for k in interesting},
        "comparisons": rows_sorted,
    }
    (OUT / "term_by_term.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\nsaved: {OUT / 'term_by_term.json'}")

    # ---- verdict.md (auto, top-3 divergent) ---
    top3 = [r for r in rows_sorted if r["abs_log_ratio"] not in (None, 0.0)][:3]
    md = ["# z345 BSIM4 channel audit — verdict\n",
          f"OP: Vg={VG1}, Vd={VD}, Vs={VSINT}, Vb={VB}",
          f"    Vgs={VGS:.4f}, Vds={VDS:.4f}, Vbs={VBS:.4f}\n",
          f"pyport Ids = {py_top['Ids']:.4e}",
          f"ngspice Id = {ng_vals['@m1[id]']:.4e}",
          f"ratio pyport/ngspice = "
          f"{py_top['Ids']/ng_vals['@m1[id]'] if ng_vals['@m1[id]'] else float('nan'):.3e}\n",
          "## Top-3 divergent intermediates"]
    for r in top3:
        md.append(f"- **{r['name']}**: pyport={r['pyport']}, ngspice={r['ngspice']}, ratio={r['ratio_py_over_ng']}")
    md.append("\n## Selected pyport intermediates")
    for k in interesting:
        v = py_locals.get(k)
        if isinstance(v, float):
            md.append(f"- {k} = {v:.6e}")
    (OUT / "verdict.md").write_text("\n".join(md))
    print(f"saved: {OUT / 'verdict.md'}")


if __name__ == "__main__":
    main()
