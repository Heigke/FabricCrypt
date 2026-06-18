"""z428: generate publication plots for z427 H1+H2 best fix."""
import sys, json, pathlib, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nsram"))
from nsram.bsim4_port.nsram_cell_2T import build_2T_cell, forward_2t
import importlib
m1 = importlib.import_module("scripts.z427_vsint_fix")  # reuse loaders

OUT = ROOT / "results" / "z427_vsint_fix"
OUT.mkdir(exist_ok=True)

cards = m1.load_cards()
curves = m1.load_curves()
sebas = m1.load_sebas_per_bias()

def run_one(VG1, VG2, Vd_array, flags):
    cell, cfg = build_2T_cell(**cards)
    cfg = m1.apply_flags(cfg, flags, sebas, VG1, VG2)
    Vd_t = torch.tensor(Vd_array, dtype=torch.float64)
    res = forward_2t(cell, cfg,
        V_D=Vd_t, V_G1=torch.tensor([VG1]*len(Vd_t)),
        V_G2=torch.tensor([VG2]*len(Vd_t)))
    return res["I_D"].detach().cpu().numpy()

FLAGS = {
    "suppress_bulk_diode_forward": True,
    "q1_be_oneway": True,
    "use_mario_ipos": True,
    "mario_ipos_param": "VG1",
    "use_sebas_per_bias_fits": True,
    "m2_source_Rs": 1.0e6,
    "gidl_route_to_sint": True,
}

# Pick representative VG2 per VG1 branch
PICKS = {0.2: 0.30, 0.4: 0.30, 0.6: 0.00}

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, (vg1, vg2) in zip(axes, PICKS.items()):
    # find measured curve
    meas = curves.get((round(vg1,2), round(vg2,2)))
    if meas is None:
        # find closest
        keys = [(k, abs(k[0]-vg1)+abs(k[1]-vg2)) for k in curves]
        best = min(keys, key=lambda x: x[1])
        meas = curves[best[0]]
        print(f"VG1={vg1} VG2={vg2}: using nearest {best[0]}")
    Vd_meas, Id_meas = meas["Vd"], meas["Id"]
    # model
    Vd_grid = np.linspace(0.05, max(Vd_meas), 60)
    Id_model = run_one(vg1, vg2, Vd_grid, FLAGS)

    ax.semilogy(Vd_meas, np.abs(Id_meas), "ko", markersize=5, label="measured (Sebas)", alpha=0.7)
    ax.semilogy(Vd_grid, np.abs(Id_model), "r-", linewidth=2, label="pyport H1+H2")
    ax.set_xlabel("V_D [V]"); ax.set_ylabel("|I_D| [A]")
    ax.set_title(f"VG1={vg1}, VG2={vg2}")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_ylim([1e-12, 1e-3])

plt.suptitle("z427 H1+H2 — Sint shunt + GIDL→Sint (cell-wide RMSE = 1.733 dec, was 3.899)",
             fontsize=11, y=1.02)
plt.tight_layout()
plt.savefig(OUT / "overlay_3branches.png", dpi=140, bbox_inches="tight")
print(f"wrote {OUT}/overlay_3branches.png")
