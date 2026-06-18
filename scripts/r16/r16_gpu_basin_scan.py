"""R-16: GPU brute-force basin scan of pyport _residuals.

For each of Sebas's 33 biases (VG1, VG2) with V_d=2.0, build a 200x200 grid
over (Vsint, Vb) in [0, 2.5]^2, evaluate ||R(Vsint, Vb)||^2 in batch on CUDA,
find all local minima, and check whether the "physical basin" (low Vsint,
low Vb -- nominally near (0.38, 0.27) as observed by ngspice at the
flagship VG1=0.6/VG2=0.20 bias) shows up as a minimum.

Output:
  results/r16_gpu_basin_scan/
    landscape_bias_<i>.npy        (full 200x200 ||R||^2 per bias)
    minima_table.json             (per-bias minima list + physical-basin flag)
    heatmap_flagship.png          (heatmap for VG1=0.6, VG2=0.20)
    summary.json                  (aggregate counts)
"""
from __future__ import annotations
import os, sys, json, time, math
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

OUT = ROOT / "results/r16_gpu_basin_scan"
OUT.mkdir(parents=True, exist_ok=True)

# -- bias config --
V_D     = 2.0
N_GRID  = 200
VS_MIN, VS_MAX = 0.0, 2.5
VB_MIN, VB_MAX = 0.0, 2.5

# physical basin definition: low Vsint AND low Vb. From z330: ngspice gave
# (Vsint, Vb) ~ (0.38, 0.27) at the flagship bias. We accept any local min
# with Vsint < 1.0 and Vb < 1.0 as "physical" (vs the pyport "trivial" root
# at Vsint ~ Vd, Vb ~ Vd).
PHYS_VS_MAX = 1.0
PHYS_VB_MAX = 1.0
FLAGSHIP_VG1, FLAGSHIP_VG2 = 0.6, 0.20

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DT  = torch.float64


def load_biases():
    p = ROOT / "data/sebas_2026_04_22/2Tcell_BSIM_param_DC.csv"
    import csv
    biases = []
    with open(p) as f:
        r = csv.DictReader(f)
        for row in r:
            biases.append((float(row["VG1"]), float(row["VG2"])))
    return biases


def build_models():
    import importlib.util
    sp = importlib.util.spec_from_file_location("v1", ROOT / "scripts/z96_narma10_pilot.py")
    v1 = importlib.util.module_from_spec(sp); sp.loader.exec_module(v1)
    from nsram.bsim4_port.nsram_cell_2T import NSRAMCell2TConfig
    from nsram.bsim4_port.bjt import GummelPoonNPN
    cfg = NSRAMCell2TConfig(use_iii=True, use_gidl=True, use_bjt=True,
                            newton_max_iters=20)
    M1, M2 = v1.build_calibrated_models()
    bjt = GummelPoonNPN.from_sebas_card()
    bjt.Bf = 9000.0; bjt.Va = 0.55; bjt.Is = 1e-9
    return cfg, M1, M2, bjt


def residual_grid(cfg, M1, M2, bjt, VG1, VG2, Vd, vs_axis, vb_axis):
    """Compute ||R||^2 on a NxN grid. Returns numpy (N,N)."""
    from nsram.bsim4_port.nsram_cell_2T import _residuals

    N = len(vs_axis)
    # broadcast grids (N_vb, N_vs) -- rows=Vb, cols=Vsint
    Vsint = torch.tensor(vs_axis, dtype=DT, device=DEV).view(1, N).expand(N, N).contiguous()
    Vb    = torch.tensor(vb_axis, dtype=DT, device=DEV).view(N, 1).expand(N, N).contiguous()
    Vd_t  = torch.full_like(Vsint, Vd)
    VG1_t = torch.full_like(Vsint, VG1)
    VG2_t = torch.full_like(Vsint, VG2)
    with torch.no_grad():
        R_S, R_B, _ = _residuals(cfg, M1, bjt, Vd_t, VG1_t, VG2_t, Vsint, Vb, model_M2=M2)
    # Normalise: residuals can span many orders of magnitude. Use log10 scale on
    # ||R||^2 (sum of squares). Add tiny eps so log is finite.
    norm = (R_S * R_S + R_B * R_B)
    return norm.detach().cpu().numpy()


def find_local_minima(field, vs_axis, vb_axis):
    """Find local minima (8-neighbour) in the 2D field.
    field[i,j] indexed by (i=Vb idx, j=Vsint idx).
    Returns list of dicts with (Vsint, Vb, value)."""
    H, W = field.shape
    interior = field[1:-1, 1:-1]
    mins = []
    # vectorised 8-neighbour min check
    less = np.ones_like(interior, dtype=bool)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            shifted = field[1+di:H-1+di, 1+dj:W-1+dj]
            less &= (interior < shifted)
    ii, jj = np.where(less)
    for (i, j) in zip(ii + 1, jj + 1):
        mins.append({
            "Vsint": float(vs_axis[j]),
            "Vb":    float(vb_axis[i]),
            "Rnorm2": float(field[i, j]),
        })
    # Also include boundary minima if extremely low (just in case basin sits at edge)
    mins.sort(key=lambda d: d["Rnorm2"])
    return mins


def main():
    print(f"[r16] device={DEV} dtype={DT}")
    biases = load_biases()
    print(f"[r16] {len(biases)} Sebas biases loaded")

    vs_axis = np.linspace(VS_MIN, VS_MAX, N_GRID)
    vb_axis = np.linspace(VB_MIN, VB_MAX, N_GRID)

    cfg, M1, M2, bjt = build_models()
    print("[r16] models built; starting scan")

    per_bias = []
    flagship_field = None
    t0 = time.time()
    for idx, (vg1, vg2) in enumerate(biases):
        try:
            field = residual_grid(cfg, M1, M2, bjt, vg1, vg2, V_D, vs_axis, vb_axis)
        except Exception as e:
            print(f"  [{idx:02d}] VG1={vg1} VG2={vg2}  FAIL: {e}")
            per_bias.append({"idx": idx, "VG1": vg1, "VG2": vg2,
                             "error": repr(e),
                             "physical_basin_found": False})
            continue

        np.save(OUT / f"landscape_bias_{idx:02d}.npy", field.astype(np.float32))
        if abs(vg1 - FLAGSHIP_VG1) < 1e-6 and abs(vg2 - FLAGSHIP_VG2) < 1e-6:
            flagship_field = field.copy()

        mins = find_local_minima(field, vs_axis, vb_axis)
        phys = [m for m in mins if m["Vsint"] < PHYS_VS_MAX and m["Vb"] < PHYS_VB_MAX]
        glob = mins[0] if mins else None
        # also report global min argmin (even if at edge)
        gi, gj = np.unravel_index(np.argmin(field), field.shape)
        argmin = {"Vsint": float(vs_axis[gj]), "Vb": float(vb_axis[gi]),
                  "Rnorm2": float(field[gi, gj])}
        per_bias.append({
            "idx": idx, "VG1": vg1, "VG2": vg2,
            "n_local_minima": len(mins),
            "global_argmin": argmin,
            "best_local_min": glob,
            "physical_minima": phys[:5],
            "physical_basin_found": len(phys) > 0,
        })
        if idx % 5 == 0 or idx == len(biases) - 1:
            print(f"  [{idx:02d}] VG1={vg1:.2f} VG2={vg2:+.2f}  "
                  f"n_min={len(mins)} phys={len(phys)} "
                  f"argmin=(Vs={argmin['Vsint']:.2f},Vb={argmin['Vb']:.2f}) "
                  f"||R||^2={argmin['Rnorm2']:.3e}")

    wall = time.time() - t0
    n_phys = sum(1 for b in per_bias if b.get("physical_basin_found"))
    summary = {
        "n_biases": len(biases),
        "n_phys_basin_found": n_phys,
        "n_grid": N_GRID,
        "v_d": V_D,
        "wall_s": wall,
        "device": str(DEV),
        "phys_basin_criterion": f"Vsint<{PHYS_VS_MAX} AND Vb<{PHYS_VB_MAX}",
        "per_bias": per_bias,
    }
    (OUT / "minima_table.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[r16] wall={wall:.1f}s")
    print(f"[r16] biases with physical basin in residual landscape: {n_phys}/{len(biases)}")

    # heatmap for flagship
    if flagship_field is not None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            log_field = np.log10(flagship_field + 1e-30)
            fig, ax = plt.subplots(figsize=(7, 6))
            im = ax.imshow(log_field, origin="lower",
                           extent=[VS_MIN, VS_MAX, VB_MIN, VB_MAX],
                           aspect="auto", cmap="viridis")
            cb = plt.colorbar(im, ax=ax)
            cb.set_label("log10(||R||^2)")
            # mark ngspice basin
            ax.plot(0.38, 0.27, "rx", ms=14, mew=3, label="ngspice basin (0.38, 0.27)")
            # mark pyport trivial root ~ (1.87, 2.0)
            ax.plot(1.87, 2.00, "w+", ms=14, mew=3, label="pyport root (1.87, 2.0)")
            # mark argmin of this field
            gi, gj = np.unravel_index(np.argmin(flagship_field), flagship_field.shape)
            ax.plot(vs_axis[gj], vb_axis[gi], "o", mec="yellow",
                    mfc="none", ms=14, mew=2,
                    label=f"argmin ({vs_axis[gj]:.2f}, {vb_axis[gi]:.2f})")
            ax.set_xlabel("Vsint [V]")
            ax.set_ylabel("Vb [V]")
            ax.set_title(f"R-16 ||R||^2 landscape  VG1={FLAGSHIP_VG1}, "
                         f"VG2={FLAGSHIP_VG2}, Vd={V_D}")
            ax.legend(loc="upper right", fontsize=8)
            fig.tight_layout()
            png_path = OUT / "heatmap_flagship.png"
            fig.savefig(png_path, dpi=130)
            plt.close(fig)
            print(f"[r16] heatmap saved: {png_path}")
        except Exception as e:
            print(f"[r16] plot fail: {e}")
    else:
        print("[r16] WARNING: flagship bias not found in CSV?")


if __name__ == "__main__":
    main()
