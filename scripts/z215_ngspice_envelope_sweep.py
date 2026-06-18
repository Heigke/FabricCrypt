"""z215: Wide-envelope ngspice cross-validation of the BSIM4 PyTorch port.

Sweeps a full (VGS, VDS, VBS) × (geom) grid using Sebas's PTM130bulkNSRAM card.
For each bias point, runs ngspice .op (single-MOSFET deck) AND our PyTorch port
(`compute_size_dep` + `compute_dc`), extracts {Id, Vth, Vdsat, gm, gds}, and
computes per-quantity relative errors.

Aggregates statistics by region (subthreshold, linear, saturation, snapback)
and by geometry, plots a 4-panel scatter (port vs ngspice, y=x line), and
emits a verdict in findings.md.

Usage:
    source venv/bin/activate
    python scripts/z215_ngspice_envelope_sweep.py
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
OUT_DIR = REPO / "results" / "z215_ngspice_envelope"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NGSPICE = "/usr/bin/ngspice"
PTM = REPO / "data/sebas_2026_04_22/PTM130bulkNSRAM.txt"
PTM_NG = Path("/tmp/PTM130bulkNSRAM_ng.txt")


def _normalize_for_ngspice(src: Path, dst: Path) -> None:
    txt = src.read_text()
    txt = re.sub(r"^(\s*\.param\s+\w+)\s+(?!=)(\S)",
                 r"\1 = \2", txt, flags=re.MULTILINE | re.IGNORECASE)
    inject = "\n.param vsatn = 80000\n"
    txt = inject + txt
    dst.write_text(txt)


_normalize_for_ngspice(PTM, PTM_NG)

# ------------------------------------------------------------------ #
# Sweep grid
# ------------------------------------------------------------------ #
VGS_LIST = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
VDS_LIST = [0.05, 0.2, 0.5, 1.0, 1.5, 1.95]
VBS_LIST = [-0.4, -0.2, 0.0, 0.2, 0.4]
GEOMS = {
    "M1_short": dict(L=0.18e-6, W=0.36e-6, l_str="0.18u", w_str="0.36u"),
    "M2_long":  dict(L=1.8e-6,  W=0.36e-6, l_str="1.8u",  w_str="0.36u"),
}


# ------------------------------------------------------------------ #
# ngspice runner (one bias point → dict of probes)
# ------------------------------------------------------------------ #
def make_netlist(vgs: float, vds: float, vbs: float, l_str: str, w_str: str) -> str:
    """Single-MOSFET .op deck. Source/Bulk pinned via VS/VB voltage sources."""
    return f"""* z215 envelope probe
.include {PTM_NG}

M1 D G S B NMOS l={l_str} w={w_str}

VG G 0 DC {vgs}
VD D 0 DC {vds}
VS S 0 DC 0
VB B 0 DC {vbs}

.options gmin=1e-15 abstol=1e-15 reltol=1e-5 itl1=500 itl2=200

.control
op
print -i(vd) @m1[vth] @m1[vdsat] @m1[gm] @m1[gds]
quit
.endc
.end
"""


PROBE_RE = re.compile(r"^([\-@\w\[\]\(\)]+)\s*=\s*([\-+0-9.eE]+)\s*$")


def run_ngspice_one(vgs: float, vds: float, vbs: float, l_str: str, w_str: str) -> dict:
    nl = make_netlist(vgs, vds, vbs, l_str, w_str)
    with tempfile.NamedTemporaryFile("w", suffix=".cir", delete=False) as f:
        f.write(nl)
        cir = f.name
    try:
        res = subprocess.run([NGSPICE, "-b", cir],
                             capture_output=True, text=True, timeout=30)
        out = res.stdout + "\n" + res.stderr
        vals = {}
        for line in out.splitlines():
            m = PROBE_RE.match(line.strip())
            if m:
                key, val = m.group(1), m.group(2)
                try:
                    vals[key] = float(val)
                except ValueError:
                    pass
        # Rename to friendly keys
        out_dict = {
            "Id": vals.get("-i(vd)", float("nan")),
            "Vth": vals.get("@m1[vth]", float("nan")),
            "Vdsat": vals.get("@m1[vdsat]", float("nan")),
            "gm": vals.get("@m1[gm]", float("nan")),
            "gds": vals.get("@m1[gds]", float("nan")),
            "converged": "@m1[vth]" in vals,
        }
        return out_dict
    finally:
        try:
            os.unlink(cir)
        except OSError:
            pass


# ------------------------------------------------------------------ #
# PyTorch port runner
# ------------------------------------------------------------------ #
def get_port_model():
    from nsram.bsim4_port.model_card import BSIM4Model
    return BSIM4Model.from_spice(PTM.read_text(), model_type="nmos")


def run_port_one(model, sd, vgs, vds, vbs):
    from nsram.bsim4_port.dc import compute_dc
    Vgs = torch.tensor(vgs, dtype=torch.float64, requires_grad=True)
    Vds = torch.tensor(vds, dtype=torch.float64, requires_grad=True)
    Vbs = torch.tensor(vbs, dtype=torch.float64)
    res = compute_dc(model, sd, Vgs=Vgs, Vds=Vds, Vbs=Vbs)
    Id = res.Ids
    # gm = dId/dVgs, gds = dId/dVds (autograd)
    gm = torch.autograd.grad(Id, Vgs, retain_graph=True, create_graph=False)[0]
    gds = torch.autograd.grad(Id, Vds, retain_graph=False, create_graph=False)[0]
    return {
        "Id": float(Id.detach()),
        "Vth": float(res.Vth.detach()),
        "Vdsat": float(res.Vdsat.detach()),
        "gm": float(gm.detach()),
        "gds": float(gds.detach()),
    }


# ------------------------------------------------------------------ #
# Main sweep
# ------------------------------------------------------------------ #
def main():
    from nsram.bsim4_port.geometry import Geometry
    from nsram.bsim4_port.temp import compute_size_dep

    model = get_port_model()
    # Pre-compute SizeDependParam per geometry (cached)
    sd_cache = {}
    for name, g in GEOMS.items():
        geom = Geometry(L=g["L"], W=g["W"], NF=1)
        sd_cache[name] = compute_size_dep(model, geom, T_C=27.0)

    rows = []
    n_total = len(GEOMS) * len(VGS_LIST) * len(VDS_LIST) * len(VBS_LIST)
    n = 0
    t0 = time.time()
    for gname, g in GEOMS.items():
        sd = sd_cache[gname]
        for vgs in VGS_LIST:
            for vds in VDS_LIST:
                for vbs in VBS_LIST:
                    n += 1
                    ng = run_ngspice_one(vgs, vds, vbs, g["l_str"], g["w_str"])
                    try:
                        py = run_port_one(model, sd, vgs, vds, vbs)
                        py_ok = True
                        py_err = ""
                    except Exception as e:
                        py = {"Id": float("nan"), "Vth": float("nan"),
                              "Vdsat": float("nan"), "gm": float("nan"),
                              "gds": float("nan")}
                        py_ok = False
                        py_err = str(e)
                    rows.append({
                        "geom": gname,
                        "L": g["L"], "W": g["W"],
                        "VGS": vgs, "VDS": vds, "VBS": vbs,
                        "ng": ng, "py": py,
                        "py_ok": py_ok, "py_err": py_err,
                    })
                    if n % 20 == 0 or n == n_total:
                        dt = time.time() - t0
                        rate = n / dt if dt > 0 else 0
                        eta = (n_total - n) / rate if rate > 0 else 0
                        print(f"[{n:4d}/{n_total}] {gname} VGS={vgs:.2f} VDS={vds:.2f} "
                              f"VBS={vbs:+.2f}  Id_ng={ng['Id']:.3e}  Id_py={py['Id']:.3e}  "
                              f"({rate:.1f}/s, ETA {eta:.0f}s)")

    # ---- Aggregate stats -------------------------------------------- #
    def relerr(a, b, eps=1e-15):
        if not np.isfinite(a) or not np.isfinite(b):
            return float("nan")
        return abs(a - b) / max(abs(b), eps)

    def abserr(a, b):
        if not np.isfinite(a) or not np.isfinite(b):
            return float("nan")
        return abs(a - b)

    QUANTS = ["Id", "Vth", "Vdsat", "gm", "gds"]
    for r in rows:
        r["err"] = {}
        for q in QUANTS:
            r["err"][q + "_rel"] = relerr(r["py"][q], r["ng"][q])
            r["err"][q + "_abs"] = abserr(r["py"][q], r["ng"][q])

    def percentiles(arr):
        a = np.array([x for x in arr if np.isfinite(x)])
        if len(a) == 0:
            return dict(median=float("nan"), p95=float("nan"),
                        max=float("nan"), n=0)
        return dict(median=float(np.median(a)),
                    p95=float(np.percentile(a, 95)),
                    max=float(np.max(a)),
                    n=int(len(a)))

    # Region masks
    def region(r):
        Id = r["ng"]["Id"]
        VDS = r["VDS"]
        Vdsat = r["ng"]["Vdsat"]
        if not np.isfinite(Id):
            return "nan"
        aI = abs(Id)
        if aI < 1e-9:
            return "subthreshold"
        if aI <= 1e-5 and VDS < max(0.1, Vdsat):
            return "linear"
        if VDS > Vdsat and aI > 1e-9:
            # snapback / impact-ion regime: high VDS near supply
            if VDS >= 1.5 and aI > 1e-7:
                return "snapback_high_vds"
            return "saturation"
        return "other"

    for r in rows:
        r["region"] = region(r)

    # Aggregate per quantity globally
    stats = {"global": {}, "by_region": {}, "by_geom": {}}
    for q in QUANTS:
        stats["global"][q + "_rel"] = percentiles([r["err"][q + "_rel"] for r in rows])
        stats["global"][q + "_abs"] = percentiles([r["err"][q + "_abs"] for r in rows])

    regions = sorted(set(r["region"] for r in rows))
    for reg in regions:
        d = {}
        sub = [r for r in rows if r["region"] == reg]
        for q in QUANTS:
            d[q + "_rel"] = percentiles([r["err"][q + "_rel"] for r in sub])
            d[q + "_abs"] = percentiles([r["err"][q + "_abs"] for r in sub])
        d["count"] = len(sub)
        stats["by_region"][reg] = d

    for gname in GEOMS:
        d = {}
        sub = [r for r in rows if r["geom"] == gname]
        for q in QUANTS:
            d[q + "_rel"] = percentiles([r["err"][q + "_rel"] for r in sub])
            d[q + "_abs"] = percentiles([r["err"][q + "_abs"] for r in sub])
        d["count"] = len(sub)
        stats["by_geom"][gname] = d

    # ---- Save JSON --------------------------------------------------- #
    summary = {
        "grid": {
            "VGS": VGS_LIST, "VDS": VDS_LIST, "VBS": VBS_LIST,
            "geoms": {k: {kk: v for kk, v in vv.items() if kk != "l_str" and kk != "w_str"}
                      for k, vv in GEOMS.items()},
        },
        "n_points": len(rows),
        "stats": stats,
        "rows": rows,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"\nWrote {OUT_DIR / 'summary.json'}")

    # ---- Plot 4-panel scatter --------------------------------------- #
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 11))
    panels = [("Id", "log"), ("Vth", "linear"), ("Vdsat", "linear"), ("gm", "log")]
    colors = {"M1_short": "C0", "M2_long": "C3"}

    for ax, (q, scale) in zip(axes.flat, panels):
        for gname in GEOMS:
            x_ng = []; y_py = []
            for r in rows:
                if r["geom"] != gname:
                    continue
                xv = r["ng"][q]; yv = r["py"][q]
                if np.isfinite(xv) and np.isfinite(yv):
                    if scale == "log":
                        if abs(xv) <= 0 or abs(yv) <= 0:
                            continue
                        x_ng.append(abs(xv)); y_py.append(abs(yv))
                    else:
                        x_ng.append(xv); y_py.append(yv)
            ax.scatter(x_ng, y_py, s=10, alpha=0.6,
                       color=colors[gname], label=gname)
        # y=x line
        if scale == "log":
            ax.set_xscale("log"); ax.set_yscale("log")
            lo = 1e-15; hi = 1e-2
            ax.plot([lo, hi], [lo, hi], "k--", alpha=0.5, label="y=x")
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        else:
            allv = []
            for gname in GEOMS:
                for r in rows:
                    if r["geom"] == gname and np.isfinite(r["ng"][q]) and np.isfinite(r["py"][q]):
                        allv += [r["ng"][q], r["py"][q]]
            if allv:
                lo, hi = min(allv), max(allv)
                pad = 0.05 * max(abs(lo), abs(hi), 1e-3)
                ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad],
                        "k--", alpha=0.5, label="y=x")
                ax.set_xlim(lo - pad, hi + pad); ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlabel(f"{q} (ngspice)")
        ax.set_ylabel(f"{q} (port)")
        ax.set_title(f"{q}  port vs ngspice")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle("z215 BSIM4 port vs ngspice — full envelope sweep", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "scatter_id_vth_vdsat_gm.png", dpi=120)
    print(f"Wrote {OUT_DIR / 'scatter_id_vth_vdsat_gm.png'}")

    # ---- Findings ---------------------------------------------------- #
    def fmt_pct(x):
        if not np.isfinite(x):
            return "n/a"
        return f"{100*x:.2f}%"

    def fmt_v(x):
        if not np.isfinite(x):
            return "n/a"
        return f"{1000*x:.2f} mV"

    lines = []
    lines.append("# z215 ngspice envelope sweep — verdict\n")
    lines.append(f"- Grid: VGS={VGS_LIST}, VDS={VDS_LIST}, VBS={VBS_LIST}")
    lines.append(f"- Geoms: M1={GEOMS['M1_short']['L']*1e9:.0f}nm × {GEOMS['M1_short']['W']*1e9:.0f}nm; "
                 f"M2={GEOMS['M2_long']['L']*1e6:.1f}µm × {GEOMS['M2_long']['W']*1e9:.0f}nm")
    lines.append(f"- Total bias points: {len(rows)} (= {n_total})\n")

    lines.append("## Global relative-error stats (port vs ngspice)\n")
    lines.append("| Quantity | median | p95 | max | n |")
    lines.append("|---|---|---|---|---|")
    for q in QUANTS:
        s = stats["global"][q + "_rel"]
        lines.append(f"| {q} | {fmt_pct(s['median'])} | {fmt_pct(s['p95'])} | {fmt_pct(s['max'])} | {s['n']} |")
    lines.append("")

    lines.append("## Vth / Vdsat absolute-error stats (mV)\n")
    lines.append("| Quantity | median | p95 | max |")
    lines.append("|---|---|---|---|")
    for q in ["Vth", "Vdsat"]:
        s = stats["global"][q + "_abs"]
        lines.append(f"| {q} | {fmt_v(s['median'])} | {fmt_v(s['p95'])} | {fmt_v(s['max'])} |")
    lines.append("")

    lines.append("## By region (Id relative error)\n")
    lines.append("| Region | n | median | p95 | max |")
    lines.append("|---|---|---|---|---|")
    for reg in regions:
        s = stats["by_region"][reg]["Id_rel"]
        lines.append(f"| {reg} | {stats['by_region'][reg]['count']} | "
                     f"{fmt_pct(s['median'])} | {fmt_pct(s['p95'])} | {fmt_pct(s['max'])} |")
    lines.append("")

    lines.append("## By geometry (Id relative error)\n")
    lines.append("| Geom | n | median | p95 | max |")
    lines.append("|---|---|---|---|---|")
    for gname in GEOMS:
        s = stats["by_geom"][gname]["Id_rel"]
        lines.append(f"| {gname} | {stats['by_geom'][gname]['count']} | "
                     f"{fmt_pct(s['median'])} | {fmt_pct(s['p95'])} | {fmt_pct(s['max'])} |")
    lines.append("")

    # Verdict — failure criteria from task spec:
    #  - >10% rel error in Id in saturation regions
    #  - >5 mV abs error in Vth/Vdsat (anywhere)
    sat_id = stats["by_region"].get("saturation", {}).get("Id_rel", {"p95": 0, "max": 0})
    snap_id = stats["by_region"].get("snapback_high_vds", {}).get("Id_rel", {"p95": 0, "max": 0})
    vth_abs = stats["global"]["Vth_abs"]
    vdsat_abs = stats["global"]["Vdsat_abs"]

    fails = []
    if np.isfinite(sat_id.get("p95", float("nan"))) and sat_id["p95"] > 0.10:
        fails.append(f"Saturation Id p95 = {fmt_pct(sat_id['p95'])} > 10%")
    if np.isfinite(snap_id.get("p95", float("nan"))) and snap_id["p95"] > 0.10:
        fails.append(f"Snapback Id p95 = {fmt_pct(snap_id['p95'])} > 10%")
    if np.isfinite(vth_abs["p95"]) and vth_abs["p95"] > 5e-3:
        fails.append(f"Vth abs p95 = {fmt_v(vth_abs['p95'])} > 5 mV")
    if np.isfinite(vdsat_abs["p95"]) and vdsat_abs["p95"] > 5e-3:
        fails.append(f"Vdsat abs p95 = {fmt_v(vdsat_abs['p95'])} > 5 mV")

    lines.append("## Verdict\n")
    if not fails:
        lines.append("**PASS — port matches ngspice across the full envelope.**\n")
        lines.append("All regions pass: Id rel-err p95 ≤ 10% in saturation/snapback; "
                     "Vth & Vdsat absolute-error p95 ≤ 5 mV.")
    else:
        lines.append("**FAIL — residual port discrepancies detected:**\n")
        for f in fails:
            lines.append(f"- {f}")
    (OUT_DIR / "findings.md").write_text("\n".join(lines))
    print(f"Wrote {OUT_DIR / 'findings.md'}")
    print("\n=== VERDICT ===")
    if not fails:
        print("PASS — port matches ngspice across the full envelope.")
    else:
        for f in fails:
            print(" -", f)


if __name__ == "__main__":
    main()
