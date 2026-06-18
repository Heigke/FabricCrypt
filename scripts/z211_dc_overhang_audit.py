"""z211: Bisect the 2x Id overhang vs ngspice on a single-MOSFET DC sweep.

Sweep VDS 0..2.5 at three VGS (0.4, 0.8, 1.2). Capture from each engine:
  Id, Vdsat, Vth, gm, gds  (ngspice .save / show)
  Ids, Vdsat, Vdseff, Vgsteff, mueff, Abulk, Idsa  (python DCResult)

Compute ratio py/ng across the sweep, identify worst bias, and print the
per-block breakdown so we can see which quantity diverges first.

Outputs:
  results/z211_dc_overhang_audit/dc_overhang.png
  results/z211_dc_overhang_audit/findings.md
  results/z211_dc_overhang_audit/raw.json
"""
from __future__ import annotations
import os, re, subprocess, tempfile, json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NGSPICE = "/usr/bin/ngspice"
OUTDIR = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/z211_dc_overhang_audit")
OUTDIR.mkdir(parents=True, exist_ok=True)

# Same minimal model as z210
L_NM = 180e-9
W_NM = 360e-9
VGS_LIST = [0.4, 0.8, 1.2]
VDS_LIST = np.linspace(0.0, 2.5, 26).tolist()


def ngspice_one(vgs: float, vds: float):
    netlist = f"""* z211 single MOSFET probe
.model NM NMOS Level=14 toxe=4n vth0=0.5 alpha0=1e-4 beta0=18 vsat=1.35e5
M1 D G S B NM l={L_NM} w={W_NM}
VD D 0 DC {vds}
VG G 0 DC {vgs}
VS S 0 DC 0
VB B 0 DC 0
.control
op
print @m1[id] @m1[vdsat] @m1[vth] @m1[gm] @m1[gds] @m1[isub]
quit
.endc
.end
"""
    with tempfile.NamedTemporaryFile("w", suffix=".cir", delete=False) as f:
        f.write(netlist); cir = f.name
    try:
        res = subprocess.run([NGSPICE, "-b", cir],
                             capture_output=True, text=True, timeout=20)
    finally:
        os.unlink(cir)
    out = res.stdout
    def parse(key):
        m = re.search(rf"@m1\[{key}\]\s*=\s*([\-\+\deE\.\s]+)$",
                      out, re.MULTILINE | re.IGNORECASE)
        return float(m.group(1).strip().split()[0]) if m else float("nan")
    return dict(
        Id=parse("id"), Vdsat=parse("vdsat"), Vth=parse("vth"),
        gm=parse("gm"), gds=parse("gds"), Isub=parse("isub"),
    )


def python_one(vgs: float, vds: float):
    from nsram.bsim4_port.model_card import BSIM4Model
    from nsram.bsim4_port.geometry import Geometry
    from nsram.bsim4_port.temp import compute_size_dep
    from nsram.bsim4_port.dc import compute_dc

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

    Vgs = torch.tensor([vgs], dtype=torch.float64)
    Vds = torch.tensor([vds], dtype=torch.float64)
    Vbs = torch.tensor([0.0], dtype=torch.float64)
    dc = compute_dc(model, sd, Vgs=Vgs, Vds=Vds, Vbs=Vbs)
    return dict(
        Ids=float(dc.Ids.item()),
        Vth=float(dc.Vth.item()),
        Vgsteff=float(dc.Vgsteff.item()),
        Vdsat=float(dc.Vdsat.item()),
        Vdseff=float(dc.Vdseff.item()),
        Abulk=float(dc.Abulk.item()),
        mueff=float(dc.mueff.item()),
        Idsa=float(dc.Idsa.item()) if dc.Idsa is not None else float("nan"),
    )


def main():
    rows = []
    for vgs in VGS_LIST:
        for vds in VDS_LIST:
            ng = ngspice_one(vgs, vds)
            py = python_one(vgs, vds)
            rows.append(dict(VGS=vgs, VDS=vds, ng=ng, py=py))
            print(f"VGS={vgs:.2f} VDS={vds:.2f} | "
                  f"Id ng={ng['Id']:+.3e} py={py['Ids']:+.3e} "
                  f"ratio={py['Ids']/ng['Id'] if ng['Id'] else float('nan'):+.3f} | "
                  f"Vdsat ng={ng['Vdsat']:.4f} py={py['Vdsat']:.4f} "
                  f"d={py['Vdsat']-ng['Vdsat']:+.4f}")

    # Save raw
    (OUTDIR / "raw.json").write_text(json.dumps(rows, indent=2))

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    colors = {0.4: "C0", 0.8: "C1", 1.2: "C2"}
    for vgs in VGS_LIST:
        sub = [r for r in rows if r["VGS"] == vgs]
        vds = np.array([r["VDS"] for r in sub])
        ing = np.array([r["ng"]["Id"] for r in sub])
        ipy = np.array([r["py"]["Ids"] for r in sub])
        axes[0,0].plot(vds, ing, "-", color=colors[vgs], label=f"ng VGS={vgs}")
        axes[0,0].plot(vds, ipy, "--", color=colors[vgs], label=f"py VGS={vgs}")
        axes[0,1].semilogy(vds, np.maximum(ing,1e-15), "-", color=colors[vgs], label=f"ng VGS={vgs}")
        axes[0,1].semilogy(vds, np.maximum(ipy,1e-15), "--", color=colors[vgs], label=f"py VGS={vgs}")
        ratio = ipy / np.where(ing == 0, np.nan, ing)
        axes[1,0].plot(vds, ratio, "-o", color=colors[vgs], label=f"VGS={vgs}")
        vdsat_ng = np.array([r["ng"]["Vdsat"] for r in sub])
        vdsat_py = np.array([r["py"]["Vdsat"] for r in sub])
        axes[1,1].plot(vds, vdsat_py - vdsat_ng, "-o", color=colors[vgs], label=f"VGS={vgs}")

    axes[0,0].set_title("Id linear"); axes[0,0].set_xlabel("VDS"); axes[0,0].set_ylabel("Id [A]"); axes[0,0].legend(fontsize=8); axes[0,0].grid(True)
    axes[0,1].set_title("Id semilogy"); axes[0,1].set_xlabel("VDS"); axes[0,1].set_ylabel("Id [A]"); axes[0,1].legend(fontsize=8); axes[0,1].grid(True)
    axes[1,0].axhline(1.0, color="k", ls=":")
    axes[1,0].axhline(1.5, color="r", ls=":")
    axes[1,0].set_title("Id_python / Id_ngspice"); axes[1,0].set_xlabel("VDS"); axes[1,0].set_ylabel("ratio"); axes[1,0].legend(); axes[1,0].grid(True)
    axes[1,1].axhline(0.0, color="k", ls=":")
    axes[1,1].set_title("Vdsat_py - Vdsat_ng"); axes[1,1].set_xlabel("VDS"); axes[1,1].set_ylabel("ΔVdsat [V]"); axes[1,1].legend(); axes[1,1].grid(True)
    fig.tight_layout()
    fig.savefig(OUTDIR / "dc_overhang.png", dpi=110)
    print(f"\nSaved: {OUTDIR/'dc_overhang.png'}")

    # Findings table — at VGS=1.2, 5 Vds points
    print("\n=== ratio table @ VGS=1.2 ===")
    summary = []
    for vds_target in [0.5, 1.0, 1.5, 2.0, 2.5]:
        rec = min((r for r in rows if r["VGS"] == 1.2),
                  key=lambda r: abs(r["VDS"] - vds_target))
        ng, py = rec["ng"], rec["py"]
        ratio = py["Ids"]/ng["Id"] if ng["Id"] else float("nan")
        print(f"  VDS={rec['VDS']:.2f}  Id ng={ng['Id']:+.3e}  py={py['Ids']:+.3e}  "
              f"ratio={ratio:.3f}  Vdsat ng={ng['Vdsat']:.4f} py={py['Vdsat']:.4f}  "
              f"Vdseff_py={py['Vdseff']:.4f}  Vgsteff_py={py['Vgsteff']:.4f}  Abulk_py={py['Abulk']:.4f}")
        summary.append(dict(VDS=rec["VDS"], ratio=ratio, ng=ng, py=py))

    # Worst bias point
    worst = max(rows, key=lambda r: (r["py"]["Ids"]/r["ng"]["Id"]) if r["ng"]["Id"] else 0.0)
    print(f"\nWorst overhang: VGS={worst['VGS']} VDS={worst['VDS']} ratio={worst['py']['Ids']/worst['ng']['Id']:.3f}")
    print(f"  ng: {worst['ng']}")
    print(f"  py: {worst['py']}")

    # Build findings.md
    md = ["# z211 DC overhang audit — findings\n",
          f"Single NMOS L=180nm W=360nm, minimal model (TOXE=4n VTH0=0.5 VSAT=1.35e5)\n",
          f"\n## Worst-case bias\n",
          f"VGS={worst['VGS']} VDS={worst['VDS']} → ratio py/ng = {worst['py']['Ids']/worst['ng']['Id']:.3f}\n",
          f"\nngspice: Id={worst['ng']['Id']:.3e} Vdsat={worst['ng']['Vdsat']:.4f} Vth={worst['ng']['Vth']:.4f}\n",
          f"\npython:  Ids={worst['py']['Ids']:.3e} Vdsat={worst['py']['Vdsat']:.4f} Vdseff={worst['py']['Vdseff']:.4f} Vgsteff={worst['py']['Vgsteff']:.4f} Abulk={worst['py']['Abulk']:.4f} mueff={worst['py']['mueff']:.4f}\n",
          f"\n## Per-VGS ratio table (VGS=1.2)\n",
          "| VDS | Id_ng | Id_py | ratio | Vdsat_ng | Vdsat_py | ΔVdsat |\n",
          "|---|---|---|---|---|---|---|\n"]
    for s in summary:
        md.append(f"| {s['VDS']:.2f} | {s['ng']['Id']:.3e} | {s['py']['Ids']:.3e} | "
                  f"{s['ratio']:.3f} | {s['ng']['Vdsat']:.4f} | {s['py']['Vdsat']:.4f} | "
                  f"{s['py']['Vdsat']-s['ng']['Vdsat']:+.4f} |\n")
    (OUTDIR / "findings.md").write_text("".join(md))
    print(f"Saved: {OUTDIR/'findings.md'}")


if __name__ == "__main__":
    main()
