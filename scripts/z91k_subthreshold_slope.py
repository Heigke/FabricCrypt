"""z91k — subthreshold-slope diagnostic, isolated M2 BSIM4.

A.5.a: z91j showed pyport vs ngspice diverges by ~1 dec median on same
M2 card (subthreshold under, above-Vt over — polarity flip around Vth).
This tests whether the subthreshold-slope `n` is wrong.

Method: hold Vds=0.5V, sweep Vgs ∈ [-0.2, 0.8] at ΔVgs=0.025. Compute
log10|Id|(Vgs) for both engines. Subthreshold slope S = dVgs/d(log10 Id)
extracted by linear fit on the 1e-12 → 1e-9 decade.

S_theory at room T = n × ln(10) × kT/q ≈ n × 60 mV/dec.
S_perfect_MOS = 60 mV/dec at n=1.
Sebas's M2 card has nfactor=1.58 → expected S ≈ 95 mV/dec.

If our S << ngspice's S: our `n` is too small (subthreshold too steep
→ underpredicts Id at low Vg).
If our S >> ngspice's S: our `n` is too large (too lazy slope → over).
"""
from __future__ import annotations
import subprocess, tempfile, json
from pathlib import Path
import numpy as np
import torch
torch.set_default_dtype(torch.float64)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/z91k_subthreshold_slope"
OUT.mkdir(parents=True, exist_ok=True)

# Reuse z91j's helpers
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "z91j_mod", ROOT / "scripts/z91j_ngspice_isolated_m2.py")
z91j = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(z91j)

from nsram.bsim4_port.geometry import Geometry
from nsram.bsim4_port.nsram_cell_2T import NSRAMCell2TConfig
from nsram.bsim4_port.dc import compute_dc
from nsram.bsim4_port.temp import compute_size_dep
from nsram.bsim4_port.model_card import BSIM4Model

DATA = ROOT / "data/sebas_2026_04_22"


def run_ngspice_id_vgs(vd: float, geom: Geometry,
                        vgs_arr: np.ndarray) -> np.ndarray:
    card = z91j.make_ngspice_card_inline()
    vgs_lines = "\n".join(f"VG_{i} G_{i} 0 DC {v:g}" for i, v in enumerate(vgs_arr))
    # Single Vgs sweep via .dc on the gate voltage source
    cir_text = f"""* z91k Id-Vgs
{card}
VD D 0 DC {vd:g}
VG G 0 DC 0
VS S 0 DC 0
VB B 0 DC 0
M1 D G S B NMOSSEB L={geom.L:g} W={geom.W:g}
.options gmin=1e-15 reltol=1e-6 abstol=1e-16
.control
dc Vg {vgs_arr.min():g} {vgs_arr.max():g} {(vgs_arr[1]-vgs_arr[0]):g}
wrdata {{tmpfile}}.dat i(vd) v(g)
quit
.endc
.end
"""
    with tempfile.NamedTemporaryFile("w", suffix=".cir", delete=False) as f:
        cir_text = cir_text.replace("{tmpfile}", f.name)
        f.write(cir_text); cir = f.name
    res = subprocess.run(["ngspice", "-b", cir], capture_output=True,
                         text=True, timeout=60)
    if not Path(cir + ".dat").exists():
        print("[z91k] ngspice failed:", res.stderr[-300:])
        return np.array([])
    data = np.loadtxt(cir + ".dat")
    return -data[:, 1]   # i(vd) sign-flip


def run_pyport_id_vgs(vd: float, geom: Geometry, model: BSIM4Model,
                       vgs_arr: np.ndarray) -> np.ndarray:
    sd = compute_size_dep(model, geom, T_C=27.0)
    Vg = torch.tensor(vgs_arr, dtype=torch.float64)
    out = compute_dc(model=model, sd=sd,
                     Vgs=Vg, Vds=torch.full_like(Vg, vd),
                     Vbs=torch.zeros_like(Vg))
    return out.Ids.abs().numpy()


def extract_S(vgs: np.ndarray, Id: np.ndarray,
               id_lo=1e-12, id_hi=1e-9) -> float:
    """Subthreshold slope mV/dec by linear fit on log10(Id) range."""
    Id = np.maximum(np.abs(Id), 1e-30)
    mask = (Id > id_lo) & (Id < id_hi)
    if mask.sum() < 3:
        return float("nan")
    log_id = np.log10(Id[mask])
    v = vgs[mask]
    # Vgs vs log10(Id), slope = dVgs/d(log10 Id) → S in V/dec
    slope, _ = np.polyfit(log_id, v, 1)
    return float(slope * 1000.0)   # mV/dec


def main():
    text_M2 = (DATA / "M2_130bulkNSRAM.txt").read_text()
    model = BSIM4Model.from_spice(text_M2, model_type="nmos")
    z91j.z91f.patch_model_values(model, type_n=True)
    cfg = NSRAMCell2TConfig()
    geom = Geometry(L=cfg.Ln * cfg.M2_length_factor, W=cfg.Wn)
    nfactor = model.get("nfactor")
    print(f"[z91k] M2 card: nfactor={nfactor}, etab={model.get('etab')}, "
          f"cdsc={model.get('cdsc'):g}, cdscb={model.get('cdscb'):g}")
    print(f"[z91k] expected S ≈ {60 * (1 + nfactor):.1f} mV/dec  "
          f"(rough — ignores cdsc, cdscb)")

    vgs_arr = np.arange(-0.2, 0.81, 0.025)
    Vd = 0.5

    Id_ng = run_ngspice_id_vgs(Vd, geom, vgs_arr)
    if len(Id_ng) == 0:
        print("[z91k] ngspice failed, abort")
        return
    Id_py = run_pyport_id_vgs(Vd, geom, model, vgs_arr)

    S_ng = extract_S(vgs_arr, Id_ng)
    S_py = extract_S(vgs_arr, Id_py)
    print(f"[z91k] Vd={Vd}V")
    print(f"[z91k] ngspice S = {S_ng:.2f} mV/dec")
    print(f"[z91k] pyport  S = {S_py:.2f} mV/dec")
    print(f"[z91k] diff    = {S_py - S_ng:+.2f} mV/dec  ({(S_py-S_ng)/S_ng*100:+.1f}%)")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(vgs_arr, np.abs(Id_ng) + 1e-30, "k-", label=f"ngspice (S={S_ng:.1f} mV/dec)")
    ax.semilogy(vgs_arr, np.abs(Id_py) + 1e-30, "r--", label=f"pyport (S={S_py:.1f} mV/dec)")
    ax.set_xlabel("Vgs [V]")
    ax.set_ylabel("|Id| [A]")
    ax.set_title(f"z91k subthreshold slope — isolated M2, Vd={Vd}V, body=GND")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(1e-15, 1e-3)
    fig.tight_layout()
    fig.savefig(OUT / "id_vgs.png", dpi=140)

    summary = {
        "Vd": Vd,
        "S_ngspice_mV_per_dec": S_ng,
        "S_pyport_mV_per_dec": S_py,
        "S_diff": S_py - S_ng,
        "card_nfactor": nfactor,
        "card_etab": model.get("etab"),
        "card_cdsc": model.get("cdsc"),
        "card_cdscb": model.get("cdscb"),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    np.savetxt(OUT / "id_vgs.csv",
               np.column_stack([vgs_arr, Id_ng, Id_py]),
               header="Vgs,Id_ngspice,Id_pyport", delimiter=",", comments="")
    print(f"[z91k] saved {OUT}/id_vgs.png + summary.json")


if __name__ == "__main__":
    main()
