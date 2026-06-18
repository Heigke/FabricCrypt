"""z293: aggregate envelope sweep results and produce plots + summary.json.

Inputs: results/z293_envelope/{4B1_Nscaling,4B2_noise,4B3_vd_grid}/<cell>/summary.json
Outputs:
  results/z293_envelope/plot_4B1_Nscaling.png
  results/z293_envelope/plot_4B2_noise.png
  results/z293_envelope/plot_4B3_vd_heatmap.png
  results/z293_envelope/summary.json
"""
from __future__ import annotations
import json, glob
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("results/z293_envelope")


def load_cell(d: Path) -> dict | None:
    p = d / "summary.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception as e:
        print(f"WARN: parse {p}: {e}")
        return None


def cells_in(sub: str) -> list[dict]:
    out = []
    for d in sorted((ROOT / sub).glob("*")):
        if not d.is_dir():
            continue
        s = load_cell(d)
        if s is not None:
            s["_dir"] = d.name
            out.append(s)
    return out


# -------- 4B.1 N-scaling --------
def plot_4B1(cells):
    if not cells:
        return None
    cells = sorted(cells, key=lambda c: c["cell"]["N"])
    Ns = [c["cell"]["N"] for c in cells]
    means = [c["mean_acc"] for c in cells]
    stds  = [c["std_acc"] or 0 for c in cells]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(Ns, means, yerr=stds, marker="o", capsize=4, color="C0")
    ax.axhline(0.6097, ls="--", color="gray", label="DS-N5c (0.610)")
    ax.axhline(0.76,   ls=":",  color="orange", label="Conservative (0.760)")
    ax.axhline(0.811,  ls=":",  color="red",    label="Fair baseline (0.811)")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("N (HDC bits)")
    ax.set_ylabel("Test accuracy (UCI-HAR)")
    ax.set_title("4B.1 N-scaling — V_d=2.00/0.50, V_G=0.30/0.30, σ=0")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fp = ROOT / "plot_4B1_Nscaling.png"
    fig.savefig(fp, dpi=130); plt.close(fig)
    return fp


# -------- 4B.2 noise tolerance --------
def plot_4B2(cells):
    if not cells:
        return None
    cells = sorted(cells, key=lambda c: c["cell"]["sigma_noise"])
    sigs = [c["cell"]["sigma_noise"] for c in cells]
    means = [c["mean_acc"] for c in cells]
    stds  = [c["std_acc"] or 0 for c in cells]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(sigs, means, yerr=stds, marker="s", capsize=4, color="C2")
    ax.axhline(0.6097, ls="--", color="gray", label="DS-N5c")
    ax.set_xlabel("sigma_noise (feature units)")
    ax.set_ylabel("Test accuracy")
    ax.set_title("4B.2 Noise tolerance — N=128, V_d=2.00/0.50")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    fp = ROOT / "plot_4B2_noise.png"
    fig.savefig(fp, dpi=130); plt.close(fig)
    return fp


# -------- 4B.3 V_d 2D heatmap --------
def plot_4B3(cells):
    if not cells:
        return None
    his = sorted({c["cell"]["vd_high"] for c in cells})
    los = sorted({c["cell"]["vd_low"]  for c in cells})
    H = np.full((len(his), len(los)), np.nan)
    for c in cells:
        i = his.index(c["cell"]["vd_high"])
        j = los.index(c["cell"]["vd_low"])
        if c["mean_acc"] is not None:
            H[i, j] = c["mean_acc"]
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(H, origin="lower", aspect="auto", cmap="viridis",
                   vmin=np.nanmin(H), vmax=np.nanmax(H))
    ax.set_xticks(range(len(los))); ax.set_xticklabels([f"{x:.1f}" for x in los])
    ax.set_yticks(range(len(his))); ax.set_yticklabels([f"{x:.1f}" for x in his])
    ax.set_xlabel("V_d_LOW (V)"); ax.set_ylabel("V_d_HIGH (V)")
    ax.set_title("4B.3 V_d 2D sweep — mean test acc, N=128, σ=0")
    for i in range(len(his)):
        for j in range(len(los)):
            v = H[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        color="white" if v < (np.nanmin(H) + np.nanmax(H)) / 2 else "black",
                        fontsize=8)
    fig.colorbar(im, ax=ax, label="mean test acc")
    fig.tight_layout()
    fp = ROOT / "plot_4B3_vd_heatmap.png"
    fig.savefig(fp, dpi=130); plt.close(fig)
    return fp


def best_cell(all_cells):
    cands = [c for c in all_cells if c.get("mean_acc") is not None]
    if not cands:
        return None
    return max(cands, key=lambda c: c["mean_acc"])


def is_local_max(cells_4B3, best):
    """Check whether best 4B.3 cell is a LOCAL MAX (not an edge-only monotone)."""
    if not cells_4B3 or best is None:
        return False
    his = sorted({c["cell"]["vd_high"] for c in cells_4B3})
    los = sorted({c["cell"]["vd_low"]  for c in cells_4B3})
    bhi = best["cell"]["vd_high"]; blo = best["cell"]["vd_low"]
    i = his.index(bhi); j = los.index(blo)
    interior_hi = 0 < i < len(his) - 1
    interior_lo = 0 < j < len(los) - 1
    return interior_hi and interior_lo


def main():
    c1 = cells_in("4B1_Nscaling")
    c2 = cells_in("4B2_noise")
    c3 = cells_in("4B3_vd_grid")
    print(f"loaded 4B1={len(c1)} 4B2={len(c2)} 4B3={len(c3)}")

    plot_4B1(c1); plot_4B2(c2); plot_4B3(c3)

    # Gates
    Nscaling = sorted(c1, key=lambda c: c["cell"]["N"])
    monotone_nondec = True
    for a, b in zip(Nscaling[:-1], Nscaling[1:]):
        if (a.get("mean_acc") is not None and b.get("mean_acc") is not None
                and b["mean_acc"] + 1e-9 < a["mean_acc"]):
            monotone_nondec = False
            break
    N1024 = next((c for c in Nscaling if c["cell"]["N"] == 1024), None)
    ambitious_4B1 = bool(N1024 and N1024.get("mean_acc") is not None
                         and N1024["mean_acc"] >= 0.76)

    s0 = next((c for c in c2 if c["cell"]["sigma_noise"] == 0.0), None)
    s05 = next((c for c in c2 if abs(c["cell"]["sigma_noise"] - 0.05) < 1e-9), None)
    s10 = next((c for c in c2 if abs(c["cell"]["sigma_noise"] - 0.10) < 1e-9), None)
    pass_4B2 = bool(s0 and s05 and s0.get("mean_acc") is not None
                    and s05.get("mean_acc") is not None
                    and abs(s05["mean_acc"] - s0["mean_acc"]) <= 0.01)
    ambitious_4B2 = bool(s0 and s10 and s0.get("mean_acc") is not None
                         and s10.get("mean_acc") is not None
                         and s10["mean_acc"] > s0["mean_acc"])

    best3 = best_cell(c3)
    pass_4B3_local_max = is_local_max(c3, best3)

    all_cells = c1 + c2 + c3
    best_overall = best_cell(all_cells)

    summary = {
        "experiment": "z293_envelope_sweep_aggregate",
        "counts": {"4B1": len(c1), "4B2": len(c2), "4B3": len(c3)},
        "4B1_Nscaling": [
            {"N": c["cell"]["N"], "mean_acc": c["mean_acc"],
             "std_acc": c["std_acc"], "ci95": c.get("ci95"),
             "energy_nJ": (c["mean_energy_J_per_inference"] or 0) * 1e9
                          if c.get("mean_energy_J_per_inference") else None,
             "verdict": c["gates"]["verdict"]}
            for c in Nscaling
        ],
        "4B2_noise": [
            {"sigma": c["cell"]["sigma_noise"], "mean_acc": c["mean_acc"],
             "std_acc": c["std_acc"], "verdict": c["gates"]["verdict"]}
            for c in sorted(c2, key=lambda c: c["cell"]["sigma_noise"])
        ],
        "4B3_vd_grid": [
            {"vd_high": c["cell"]["vd_high"], "vd_low": c["cell"]["vd_low"],
             "mean_acc": c["mean_acc"], "std_acc": c["std_acc"],
             "verdict": c["gates"]["verdict"]}
            for c in sorted(c3, key=lambda c: (c["cell"]["vd_high"], c["cell"]["vd_low"]))
        ],
        "gates_locked": {
            "4B1_monotone_nondecreasing": monotone_nondec,
            "4B1_ambitious_N1024_geq_0.76": ambitious_4B1,
            "4B2_sigma005_within_1pp": pass_4B2,
            "4B2_ambitious_sigma010_improves": ambitious_4B2,
            "4B3_local_max_interior": pass_4B3_local_max,
        },
        "best_4B3_cell": {
            "vd_high": best3["cell"]["vd_high"] if best3 else None,
            "vd_low":  best3["cell"]["vd_low"]  if best3 else None,
            "mean_acc": best3["mean_acc"] if best3 else None,
        } if best3 else None,
        "best_overall_cell": {
            "tag": best_overall["_dir"] if best_overall else None,
            "cell": best_overall["cell"] if best_overall else None,
            "mean_acc": best_overall["mean_acc"] if best_overall else None,
            "std_acc": best_overall["std_acc"] if best_overall else None,
            "verdict": best_overall["gates"]["verdict"] if best_overall else None,
        } if best_overall else None,
    }
    out = ROOT / "summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"wrote {out}")
    print(json.dumps(summary["gates_locked"], indent=2))
    print(f"best overall: {summary['best_overall_cell']}")


if __name__ == "__main__":
    main()
