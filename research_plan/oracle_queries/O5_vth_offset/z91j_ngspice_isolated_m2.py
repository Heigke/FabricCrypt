"""z91j — ngspice cross-validation on isolated M2.

Tests whether our PyTorch BSIM4 port (compute_dc) reproduces ngspice's
Berkeley BSIM4 (level 14) on the SAME M2 card with body=GND.

If they match (log-RMSE < 0.3 dec): our compute_dc is faithful, residual
in z91g is cell-level wiring/BJT/body-coupling, not BSIM4 itself.
If they diverge: a compute_dc bug is part of the cell-level residual.

Single bias point: VG2 in {-0.1, 0.0, 0.2}, Vd ∈ [0, 2V], Vbs=0, geometry
matches z91g M2 (L = 10× Ln, W = Wn).
"""
from __future__ import annotations
import subprocess, tempfile, json, re
from pathlib import Path
import numpy as np
import torch
torch.set_default_dtype(torch.float64)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/z91j_ngspice_iso_m2"
OUT.mkdir(parents=True, exist_ok=True)
DATA = ROOT / "data/sebas_2026_04_22"

from nsram.bsim4_port.model_card import BSIM4Model
from nsram.bsim4_port.dc import compute_dc
from nsram.bsim4_port.geometry import Geometry
from nsram.bsim4_port.temp import compute_size_dep
from nsram.bsim4_port.nsram_cell_2T import NSRAMCell2TConfig

# Re-use z91f's post-load patcher (parser drops + continuation lines on .param)
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "z91f_mod", ROOT / "scripts/z91f_validate_with_sebas_params.py")
z91f = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(z91f)


def make_ngspice_card_inline() -> str:
    """Return self-contained ngspice .model NMOS body using Sebas's M2 params.

    We bake in the n-variant numeric values (parser already resolved them
    via .param continuation patch in z91f.patch_model_values).
    """
    # Use our parser to extract resolved values, then emit them as a flat
    # .model card ngspice can ingest. ngspice supports 'level=14' BSIM4.
    text_M2 = (DATA / "M2_130bulkNSRAM.txt").read_text()
    m = BSIM4Model.from_spice(text_M2, model_type="nmos")
    z91f.patch_model_values(m, type_n=True)
    # Pull out a curated set of params we know matter
    keys = ["vth0", "k1", "k2", "k3", "k3b", "w0", "nlx", "dvt0", "dvt1",
            "dvt2", "dvt0w", "dvt1w", "dvt2w", "u0", "ua", "ub", "uc",
            "vsat", "a0", "ags", "a1", "a2", "b0", "b1", "keta", "voff",
            "nfactor", "cdsc", "cdscb", "cdscd", "cit", "eta0", "etab",
            "dsub", "pclm", "pdiblc1", "pdiblc2", "pdiblcb", "drout",
            "pscbe1", "pscbe2", "pvag", "delta", "rdsw", "prwg", "prwb",
            "wr", "alpha0", "alpha1", "beta0", "agidl", "bgidl", "cgidl",
            "egidl", "agisl", "bgisl", "cgisl", "egisl", "tox", "xj",
            "nsub", "ngate", "ndep", "nsd", "lint", "wint", "xl", "xw",
            "rsh", "rdswmin", "rsw", "rdw"]
    pieces = []
    for k in keys:
        try:
            v = m[k]
            if isinstance(v, (int, float)):
                pieces.append(f"{k}={v:g}")
        except Exception:
            pass
    # Build .model card with proper ngspice continuations
    lines = [".model NMOSSEB NMOS (level=14"]
    for i in range(0, len(pieces), 6):
        lines.append("+ " + " ".join(pieces[i:i+6]))
    lines.append(")")
    return "\n".join(lines)


def run_ngspice_id_vd(vg2: float, geom: Geometry) -> tuple[np.ndarray, np.ndarray]:
    card = make_ngspice_card_inline()
    cir_text = f"""* z91j — isolated M2 BSIM4 cross-validation
{card}
VG G 0 DC {vg2:g}
VS S 0 DC 0
VB B 0 DC 0
VD D 0 DC 0
M1 D G S B NMOSSEB L={geom.L:g} W={geom.W:g}
.options gmin=1e-15 reltol=1e-6 abstol=1e-14
.control
dc Vd 0 2 0.05
wrdata {{tmpfile}}.dat i(vd) v(d)
quit
.endc
.end
"""
    with tempfile.NamedTemporaryFile("w", suffix=".cir", delete=False) as f:
        cir_text = cir_text.replace("{tmpfile}", f.name)
        f.write(cir_text)
        cir = f.name
    res = subprocess.run(["ngspice", "-b", cir], capture_output=True,
                         text=True, timeout=60)
    if not Path(cir + ".dat").exists():
        print("[z91j] ngspice stderr:", res.stderr[-500:])
        print("[z91j] ngspice stdout:", res.stdout[-500:])
        return np.array([]), np.array([])
    data = np.loadtxt(cir + ".dat")
    # ngspice wrdata cols: 0=sweep_x, 1=i(vd) real, 2=v(d) real
    Vd = data[:, 2]; Id = -data[:, 1]   # i(vd) negative-into-source
    return Vd, Id


def run_pyport_id_vd(vg2: float, geom: Geometry, model: BSIM4Model,
                      Vd_arr: np.ndarray) -> np.ndarray:
    sd = compute_size_dep(model, geom, T_C=27.0)
    Vd = torch.tensor(Vd_arr, dtype=torch.float64)
    out = compute_dc(model=model, sd=sd,
                     Vds=Vd, Vgs=torch.full_like(Vd, vg2),
                     Vbs=torch.zeros_like(Vd))
    return out.Ids.abs().numpy()


def main():
    text_M2 = (DATA / "M2_130bulkNSRAM.txt").read_text()
    model = BSIM4Model.from_spice(text_M2, model_type="nmos")
    z91f.patch_model_values(model, type_n=True)
    cfg = NSRAMCell2TConfig()
    geom = Geometry(L=cfg.Ln * cfg.M2_length_factor, W=cfg.Wn)
    print(f"[z91j] M2 geom: L={geom.L:g} W={geom.W:g}")

    results = {}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, vg2 in zip(axes, [-0.1, 0.0, 0.2]):
        Vd_ng, Id_ng = run_ngspice_id_vd(vg2, geom)
        if len(Vd_ng) == 0:
            print(f"[z91j] VG2={vg2}: ngspice failed")
            continue
        Id_py = run_pyport_id_vd(vg2, geom, model, Vd_ng)
        eps = 1e-15
        log_p = np.log10(np.abs(Id_py) + eps)
        log_n = np.log10(np.abs(Id_ng) + eps)
        rmse = float(np.sqrt(np.mean((log_p - log_n) ** 2)))
        print(f"[z91j] VG2={vg2:+.2f}  log-RMSE = {rmse:.3f}  "
              f"Id_py range [{Id_py.min():.2e}, {Id_py.max():.2e}]  "
              f"Id_ng range [{Id_ng.min():.2e}, {Id_ng.max():.2e}]")
        results[f"vg2={vg2}"] = {
            "log_rmse": rmse,
            "Vd": Vd_ng.tolist(),
            "Id_ngspice": Id_ng.tolist(),
            "Id_pyport": Id_py.tolist(),
        }
        ax.semilogy(Vd_ng, np.abs(Id_ng), "k-", label="ngspice")
        ax.semilogy(Vd_ng, Id_py, "r--", label="pyport")
        ax.set_title(f"VG2={vg2}  log-RMSE={rmse:.2f}")
        ax.set_xlabel("Vd"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
        ax.set_ylim(1e-13, 1e-3)
    axes[0].set_ylabel("|Id| [A]")
    fig.suptitle("z91j — isolated M2 BSIM4 cross-validation (ngspice vs pyport)")
    fig.tight_layout()
    fig.savefig(OUT / "iso_m2.png", dpi=140)
    plt.close(fig)
    rmses = [r["log_rmse"] for r in results.values()]
    summary = {"n": len(rmses), "median_log_rmse": float(np.median(rmses)) if rmses else float("nan"),
               "max_log_rmse": float(max(rmses)) if rmses else float("nan")}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    (OUT / "details.json").write_text(json.dumps(results, indent=2))
    print(f"[z91j] median log-RMSE = {summary['median_log_rmse']:.3f}, "
          f"max = {summary['max_log_rmse']:.3f}")


if __name__ == "__main__":
    main()
