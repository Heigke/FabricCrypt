"""z355 — R-36: Apples-to-apples M1 component decomposition vs ngspice.

R-35 (z354) showed: pyport Ids vs ngspice @m1[id] gap is ~0.67 dec,
NOT 3 dec. But @m1[id] = Ids + Iii + Igidl - Igisl (terminal current),
so the comparison was apples-vs-oranges.

This script:
  * Forces M1 terminal voltages on a standalone NMOS deck.
  * Saves @m1[ids] (channel only), @m1[igidl], @m1[igisl],
    @m1[isub] (impact-ion / Iii), @m1[ibd], @m1[ibs], @m1[id] (total).
  * Calls pyport `_eval_mosfet`-style: dc.compute_dc → Ids,
    compute_iimpact → Iii, compute_igidl_gisl → Igidl, Igisl,
    compute_body_diodes → Ibs, Ibd.
  * Compares term-by-term, computes log10 gap statistics.

Bias grid: 27 = 3×3×3:
  VG1 ∈ {0.2, 0.4, 0.6}
  VG2 ∈ {-0.1, 0.1, 0.3}      (irrelevant for M1 standalone, but kept
                               as a label for analytical scan parallelism;
                               we use it to set Vbs grid below)
  Vd  ∈ {1.0, 1.5, 2.0}

For M1 standalone we need (Vgs, Vds, Vbs). Use:
  Vs = 0, Vg = VG1, Vd = Vd, Vb = Vbs
  Vbs ∈ {-0.20, -0.05, +0.05}  (we substitute VG2 axis with Vbs axis;
                                this isolates body-effect dependence.)

Output: results/z355_apples_compare/per_bias_components.json + verdict.md
"""
from __future__ import annotations
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")

import sys, json, re, math, subprocess, importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "nsram"))

OUT = ROOT / "results/z355_apples_compare"
OUT.mkdir(parents=True, exist_ok=True)

import torch
torch.set_default_dtype(torch.float64)

from nsram.bsim4_port.model_card import BSIM4Model
from nsram.bsim4_port.geometry import Geometry
from nsram.bsim4_port.temp import compute_size_dep
from nsram.bsim4_port.dc import compute_dc
from nsram.bsim4_port.leak import compute_iimpact, compute_igidl_gisl, compute_igb
from nsram.bsim4_port.diode import compute_body_diodes


# -------------------- Build M1 (matches z354 pattern) ------------------------
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


# -------------------- ngspice deck -------------------------------------------
NG_DECK = """.title z355 M1 apples standalone Vg={Vg} Vd={Vd} Vs=0 Vb={Vb}
.include "{M2path}"
.include "{M1path}"
Vd  d 0 DC {Vd}
Vg  g 0 DC {Vg}
Vs  s 0 DC 0
Vb  b 0 DC {Vb}
M1  d g s b NMOSdnwfb L=0.13u W=1u
.options gmin=1e-15 abstol=1e-14 reltol=1e-3 itl1=300
.save @m1[id] @m1[isub] @m1[igidl] @m1[igisl] @m1[ibd] @m1[ibs]
+ @m1[vth] @m1[vdsat] @m1[vgs] @m1[vds] @m1[vbs] @m1[gm] @m1[gds]
.control
op
print @m1[id] @m1[isub] @m1[igidl] @m1[igisl] @m1[ibd] @m1[ibs]
print @m1[vth] @m1[vdsat] @m1[gm] @m1[gds]
print @m1[vgs] @m1[vds] @m1[vbs]
quit
.endc
.end
"""

_RE_EQ = re.compile(r"(@?\w+(?:\[\w+\])?)\s*=\s*([-+]?\d+\.?\d*[eE]?[-+]?\d*)")

def run_ngspice(Vg, Vd, Vb, idx):
    deck = NG_DECK.format(
        Vg=Vg, Vd=Vd, Vb=Vb,
        M2path=str(ROOT / "data/sebas_2026_04_22/M2_130bulkNSRAM.txt"),
        M1path=str(ROOT / "data/sebas_2026_04_22/M1_130DNWFB_LALPHA0_FIX.txt"))
    deck_path = OUT / f"deck_{idx:02d}.sp"
    log_path  = OUT / f"ng_{idx:02d}.log"
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


# -------------------- pyport eval (M1 standalone) ----------------------------
def pyport_eval(M1, sd, Vg, Vd, Vb, geom):
    """Mirror _eval_mosfet for M1 standalone (Vs=0)."""
    Vgs = torch.tensor(Vg - 0.0, dtype=torch.float64)
    Vds = torch.tensor(Vd - 0.0, dtype=torch.float64)
    Vbs = torch.tensor(Vb - 0.0, dtype=torch.float64)
    Vbd = torch.tensor(Vb - Vd, dtype=torch.float64)

    dc = compute_dc(M1, sd, Vgs=Vgs, Vds=Vds, Vbs=Vbs)
    Iii = compute_iimpact(M1, sd, dc, Vds=Vds)
    Igidl, Igisl = compute_igidl_gisl(M1, sd, Vgs=Vgs, Vds=Vds, Vbs=Vbs)

    # junctions: As=Ad=W*L, Ps=Pd=2*(W+L) — matches _junctions_M1
    WL = float(geom.W * geom.L)
    WLp = float(2.0 * (geom.W + geom.L))
    Ibs, Ibd = compute_body_diodes(M1, sd, Vbs=Vbs, Vbd=Vbd,
                                   As=WL, Ad=WL, Ps=WLp, Pd=WLp)
    return {
        "Ids":   float(dc.Ids),
        "Iii":   float(Iii),
        "Igidl": float(Igidl),
        "Igisl": float(Igisl),
        "Ibs":   float(Ibs),
        "Ibd":   float(Ibd),
        "Vth":   float(dc.Vth),
        "Vdsat": float(dc.Vdsat),
        "Vgsteff": float(dc.Vgsteff),
    }


# -------------------- helpers ------------------------------------------------
def safe_log10_ratio(a, b):
    if a is None or b is None: return None
    try:
        aa, bb = abs(a), abs(b)
        if aa < 1e-30 or bb < 1e-30: return None
        return math.log10(aa / bb)
    except Exception:
        return None


# -------------------- main ---------------------------------------------------
def main():
    M1, sd, geom = build_M1()
    print(f"[z355] M1 built: k1={M1._values.get('k1'):.4f} "
          f"k2={M1._values.get('k2'):.4f} "
          f"phin={M1._values.get('phin',0.0):.4f}", flush=True)
    print(f"[z355] sd.phi={sd.phi:.5f} vth0_T={sd.vth0_T:.4f} "
          f"toxe={M1._values.get('toxe'):.3e}", flush=True)

    # Bias grid: 3 × 3 × 3 = 27 biases
    VG_LIST  = [0.2, 0.4, 0.6]
    VBS_LIST = [-0.10, 0.0, +0.05]   # body-effect axis (was VG2 in spec)
    VD_LIST  = [1.0, 1.5, 2.0]

    rows = []
    idx = 0
    for Vg in VG_LIST:
        for Vbs in VBS_LIST:
            for Vd in VD_LIST:
                idx += 1
                Vb = Vbs   # Vs=0, so Vbs == Vb
                ng = run_ngspice(Vg, Vd, Vb, idx)
                py = pyport_eval(M1, sd, Vg, Vd, Vb, geom)

                # Map ngspice components (sign convention: ngspice
                # returns currents with its own sign; we compare |·|).
                # ngspice BSIM4 does NOT expose channel-only `@m1[ids]`.
                # Its `@m1[id]` = cdrain + igidl - igisl (channel + GIDL).
                # Iii (isub) is routed separately to body.
                # So: channel_ng = id - igidl + igisl.
                ng_id    = ng.get("@m1[id]")
                ng_igidl = ng.get("@m1[igidl]")
                ng_igisl = ng.get("@m1[igisl]")
                ng_isub  = ng.get("@m1[isub]")
                ng_ibd   = ng.get("@m1[ibd]")
                ng_ibs   = ng.get("@m1[ibs]")
                ng_vth   = ng.get("@m1[vth]")
                if (ng_id is not None and ng_igidl is not None
                        and ng_igisl is not None):
                    ng_ids = ng_id - ng_igidl + ng_igisl  # channel only
                else:
                    ng_ids = None

                # Total ID reconstructed (pyport): Ids + Iii + Igidl - Igisl
                py_id_total = py["Ids"] + py["Iii"] + py["Igidl"] - py["Igisl"]

                row = {
                    "idx": idx,
                    "Vg": Vg, "Vd": Vd, "Vbs": Vbs,
                    # Currents
                    "py_Ids": py["Ids"], "ng_ids": ng_ids,
                    "py_Iii": py["Iii"], "ng_isub": ng_isub,
                    "py_Igidl": py["Igidl"], "ng_igidl": ng_igidl,
                    "py_Igisl": py["Igisl"], "ng_igisl": ng_igisl,
                    "py_Ibs": py["Ibs"], "ng_ibs": ng_ibs,
                    "py_Ibd": py["Ibd"], "ng_ibd": ng_ibd,
                    "py_id_total": py_id_total, "ng_id": ng_id,
                    # Internal vars
                    "py_Vth": py["Vth"], "ng_vth": ng_vth,
                    "py_Vgsteff": py["Vgsteff"], "py_Vdsat": py["Vdsat"],
                    # log10 ratios
                    "log_ratio_Ids":   safe_log10_ratio(py["Ids"], ng_ids),
                    "log_ratio_Iii":   safe_log10_ratio(py["Iii"], ng_isub),
                    "log_ratio_Igidl": safe_log10_ratio(py["Igidl"], ng_igidl),
                    "log_ratio_Igisl": safe_log10_ratio(py["Igisl"], ng_igisl),
                    "log_ratio_total": safe_log10_ratio(py_id_total, ng_id),
                    "delta_Vth_mV": (
                        (py["Vth"] - ng_vth) * 1000.0
                        if ng_vth is not None else None),
                }
                rows.append(row)
                print(
                    f"[{idx:02d}] Vg={Vg} Vd={Vd} Vbs={Vbs:+.2f} | "
                    f"Ids py={py['Ids']:.3e} ng={ng_ids if ng_ids is None else f'{ng_ids:.3e}'} "
                    f"r={row['log_ratio_Ids']} | "
                    f"Iii r={row['log_ratio_Iii']} | "
                    f"total r={row['log_ratio_total']}",
                    flush=True)

    # ----- summary stats -----
    def stats(key):
        xs = [r[key] for r in rows if r[key] is not None
              and not math.isnan(r[key]) and not math.isinf(r[key])]
        if not xs: return None
        n = len(xs)
        mean = sum(xs)/n
        var = sum((x-mean)**2 for x in xs)/max(n-1,1)
        return {"n": n, "mean": mean, "std": math.sqrt(var),
                "min": min(xs), "max": max(xs)}

    stat_Ids   = stats("log_ratio_Ids")
    stat_Iii   = stats("log_ratio_Iii")
    stat_Igidl = stats("log_ratio_Igidl")
    stat_total = stats("log_ratio_total")

    # Structural check: how does Ids ratio vary with each axis?
    def grouped(key, axis_key):
        groups = {}
        for r in rows:
            v = r[axis_key]
            if r[key] is None or math.isnan(r[key]): continue
            groups.setdefault(v, []).append(r[key])
        return {k: (sum(vs)/len(vs)) for k, vs in groups.items() if vs}

    by_Vg  = grouped("log_ratio_Ids", "Vg")
    by_Vd  = grouped("log_ratio_Ids", "Vd")
    by_Vbs = grouped("log_ratio_Ids", "Vbs")

    summary = {
        "bias_grid": {"VG_LIST": VG_LIST, "VBS_LIST": VBS_LIST,
                      "VD_LIST": VD_LIST, "N": len(rows)},
        "stats": {
            "log_ratio_Ids":   stat_Ids,
            "log_ratio_Iii":   stat_Iii,
            "log_ratio_Igidl": stat_Igidl,
            "log_ratio_total": stat_total,
        },
        "log_ratio_Ids_by_axis": {
            "Vg":  by_Vg,
            "Vd":  by_Vd,
            "Vbs": by_Vbs,
        },
        "rows": rows,
    }

    out_json = OUT / "per_bias_components.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"\n[z355] wrote {out_json}", flush=True)

    # ---------------- Heatmap ---------------------
    try:
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 4, figsize=(18, 4))
        for ax, key, title in zip(
            axes,
            ["log_ratio_Ids", "log_ratio_Iii", "log_ratio_Igidl", "log_ratio_total"],
            ["log10(py_Ids / ng_ids)", "log10(py_Iii / ng_isub)",
             "log10(py_Igidl / ng_igidl)", "log10(py_idTotal / ng_id)"],
        ):
            # Reshape rows back to (Vg, Vbs, Vd)
            cube = np.full((len(VG_LIST), len(VBS_LIST), len(VD_LIST)), np.nan)
            for r in rows:
                i = VG_LIST.index(r["Vg"])
                j = VBS_LIST.index(r["Vbs"])
                k = VD_LIST.index(r["Vd"])
                v = r[key]
                if v is not None and not math.isnan(v):
                    cube[i, j, k] = v
            # mean over Vd → 2D heatmap (Vg, Vbs)
            with np.errstate(all="ignore"):
                grid = np.nanmean(cube, axis=2)
            im = ax.imshow(grid, cmap="RdBu_r", vmin=-1.5, vmax=1.5,
                           aspect="auto", origin="lower")
            ax.set_xticks(range(len(VBS_LIST)))
            ax.set_xticklabels([f"{v:+.2f}" for v in VBS_LIST])
            ax.set_yticks(range(len(VG_LIST)))
            ax.set_yticklabels([f"{v:.2f}" for v in VG_LIST])
            ax.set_xlabel("Vbs (V)"); ax.set_ylabel("Vg (V)")
            ax.set_title(title, fontsize=10)
            plt.colorbar(im, ax=ax, fraction=0.046)
            # Annotate
            for i in range(grid.shape[0]):
                for j in range(grid.shape[1]):
                    if not np.isnan(grid[i, j]):
                        ax.text(j, i, f"{grid[i,j]:+.2f}",
                                ha="center", va="center",
                                fontsize=8,
                                color="white" if abs(grid[i,j])>0.7 else "black")
        plt.suptitle("z355 — log10(pyport / ngspice) per component "
                     "(mean over Vd)", fontsize=11)
        plt.tight_layout()
        png = OUT / "gap_heatmap.png"
        plt.savefig(png, dpi=120)
        plt.close()
        print(f"[z355] wrote {png}", flush=True)
    except Exception as e:
        print(f"[z355] plot failed: {e}", flush=True)

    # ---------------- Verdict -------------------
    def fmt(s):
        if s is None: return "n/a"
        return f"mean={s['mean']:+.3f}±{s['std']:.3f}  range=[{s['min']:+.3f}, {s['max']:+.3f}]  n={s['n']}"

    spread_Ids = (stat_Ids["max"] - stat_Ids["min"]) if stat_Ids else None
    structured = spread_Ids is not None and spread_Ids > 0.5
    by_axis_lines = []
    for axis, g in [("Vg", by_Vg), ("Vd", by_Vd), ("Vbs", by_Vbs)]:
        if g:
            items = " ".join(f"{k}→{v:+.2f}" for k, v in sorted(g.items()))
            by_axis_lines.append(f"  - {axis}: {items}")

    verdict_md = f"""# R-36 / z355 — Apples-to-apples component decomposition

**Date**: 2026-05-14
**Bias grid**: {len(rows)} biases
  VG ∈ {VG_LIST}, Vbs ∈ {VBS_LIST}, Vd ∈ {VD_LIST}  (Vs=0)
**Comparison**: pyport `_eval_mosfet` components vs ngspice `@m1[ids/isub/igidl/igisl/id]`

## Per-component log10(pyport / ngspice) statistics

| component  | stats |
|------------|------|
| Ids alone  | {fmt(stat_Ids)} |
| Iii alone  | {fmt(stat_Iii)} |
| Igidl alone| {fmt(stat_Igidl)} |
| Total Id (Ids+Iii+Igidl-Igisl) | {fmt(stat_total)} |

## Ids gap by bias axis (mean log10 ratio per slice)

{chr(10).join(by_axis_lines) if by_axis_lines else '  (no data)'}

## Diagnosis

- **Ids spread across grid**: {('%+.3f' % spread_Ids) if spread_Ids is not None else 'n/a'} dec
  → {"STRUCTURED (depends on bias)" if structured else "UNIFORM (multiplicative prefactor)"}
- If uniform: candidate causes = Weff/Leff scaling, Cox prefactor, mueff,
  or a single missing constant. Likely absorbable in W/L extraction.
- If structured: the term varying tells us which BSIM4 sub-block is off.
  See `log_ratio_Ids_by_axis` to identify dominant axis.

## Files

- Data: `results/z355_apples_compare/per_bias_components.json`
- Plot: `results/z355_apples_compare/gap_heatmap.png`
- Decks/logs: `results/z355_apples_compare/deck_*.sp`, `ng_*.log`
"""
    (OUT / "verdict.md").write_text(verdict_md)
    print(f"[z355] wrote {OUT / 'verdict.md'}", flush=True)

    # Stdout final summary
    print("\n=== R-36 final ===")
    print(f"  Ids alone   : {fmt(stat_Ids)}")
    print(f"  Iii alone   : {fmt(stat_Iii)}")
    print(f"  Igidl alone : {fmt(stat_Igidl)}")
    print(f"  Total Id    : {fmt(stat_total)}")
    print(f"  Ids spread  : {('%+.3f' % spread_Ids) if spread_Ids is not None else 'n/a'} dec  "
          f"→ {'STRUCTURED' if structured else 'UNIFORM'}")
    for line in by_axis_lines:
        print(line)


if __name__ == "__main__":
    main()
