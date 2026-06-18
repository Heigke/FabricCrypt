#!/usr/bin/env python3
"""Evaluate + plot z88 best fit (Stage 4) without re-running the fit."""
import json, sys, time, math
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)
torch.set_num_threads(2)

ROOT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "nsram"))
sys.path.insert(0, str(ROOT / "scripts"))

from z88_bsim4_port_fit_p7v10_skipnonconv import (
    OUT, load_curves, make_cfg_and_sd, evaluate_full,
    PARAM_SPEC, theta_to_value, fitted_dict, make_thetas,
)
from nsram.bsim4_port.model_card import BSIM4Model

DATA_DIR = ROOT / "data/sebas_2026_04_22"
SPICE_FILE = DATA_DIR / "PTM130bulkNSRAM.txt"


def value_to_theta(name: str, value: float) -> float:
    """Inverse of theta_to_value (linb / logb sigmoid reparam)."""
    spec = PARAM_SPEC[name]
    lo, hi = spec["bounds"]
    if spec["kind"] == "linb":
        s = (float(value) - lo) / (hi - lo)
    else:
        s = math.log(max(float(value), 1e-30) / lo) / math.log(hi / lo)
    s = min(max(s, 1e-9), 1.0 - 1e-9)
    return math.log(s / (1.0 - s))


def main():
    t0 = time.time()
    s4 = json.load(open(OUT / "stage4_summary.json"))
    P = s4["params"]
    print(f"Loaded Stage 4 params (loss={s4['loss']:.4f}):")
    for k, v in P.items():
        print(f"  {k:10s} = {v:+.4e}")

    thetas = make_thetas(seed=0)
    for name, val in P.items():
        if name in thetas:
            thetas[name].data = torch.tensor(value_to_theta(name, val), dtype=torch.float64)
    fitted = fitted_dict(thetas)
    print("\nRound-tripped through theta_to_value:")
    bad = 0
    for k in P:
        if abs(fitted[k] - P[k]) / max(abs(P[k]), 1e-30) > 1e-3:
            print(f"  ! mismatch {k}: target {P[k]:.4e} vs round-trip {fitted[k]:.4e}")
            bad += 1
    print(f"  ({len(P) - bad}/{len(P)} round-tripped OK)")

    spice_text = SPICE_FILE.read_text()
    model = BSIM4Model.from_spice(spice_text, model_type="nmos")
    assert len(model.given) > 50, (
        f"BSIM4Model loaded with only {len(model.given)} given params — "
        "card text was probably not read. Check SPICE_FILE.read_text() vs str(path)."
    )
    print(f"Loaded BSIM4 card with {len(model.given)} given params")
    cfg = make_cfg_and_sd(model, gates={"use_iii": True, "use_gidl": True, "use_bjt": True})
    curves = load_curves()
    print(f"\nLoaded {len(curves)} curves. Evaluating...")

    median_rmse, preds = evaluate_full(thetas, model, cfg, curves)
    print(f"\n=== Median per-curve log-RMSE = {median_rmse:.3f} ===")
    if preds:
        rmses = [p["log_rmse"] for p in preds]
        print(f"Per-curve: min={min(rmses):.3f}  max={max(rmses):.3f}  "
              f"mean={np.mean(rmses):.3f}  n={len(rmses)}")

    (OUT / "per_curve.json").write_text(json.dumps(preds, indent=1))

    by_vg1 = {}
    for p in preds:
        by_vg1.setdefault(p["VG1"], []).append(p)
    n_total = sum(len(np.asarray(p.get("converged", []))) for p in preds)
    n_conv = sum(int(np.asarray(p.get("converged", [])).sum()) for p in preds)
    print(f"Convergence: {n_conv}/{n_total} biases ({100 * n_conv / max(n_total, 1):.1f} %)")
    if by_vg1:
        fig, axes = plt.subplots(1, max(len(by_vg1), 1), figsize=(6 * len(by_vg1), 6),
                                  sharey=True, squeeze=False)
        axes = axes[0]
        cmap = plt.get_cmap("viridis")
        for ax, VG1 in zip(axes, sorted(by_vg1)):
            ps = sorted(by_vg1[VG1], key=lambda c: c["VG2"])
            n = len(ps)
            for i, p in enumerate(ps):
                color = cmap(i / max(n - 1, 1))
                Vd = np.asarray(p["Vd"])
                Id_meas = np.asarray(p["Id_meas"])
                Id_pred = np.asarray(p["Id_pred"])
                conv = np.asarray(p.get("converged", np.ones_like(Vd, dtype=bool)), dtype=bool)
                # Always show data
                ax.semilogy(Vd, np.abs(Id_meas), "o", color=color, ms=4, alpha=0.7,
                             label=f"VG2={p['VG2']:+.2f}  ({int(conv.sum())}/{len(conv)})")
                # Fit line on converged points only — markers, not connecting through gaps
                if conv.any():
                    ax.semilogy(Vd[conv], np.abs(Id_pred[conv]),
                                 "-", color=color, lw=1.4, alpha=0.95, marker=".", ms=4)
            ax.set_xlabel("Vd [V]"); ax.grid(alpha=0.3)
            ax.set_title(f"VG1 = {VG1} V    ({n} curves)")
            ax.legend(loc="lower right", fontsize=7, ncol=2)
        axes[0].set_ylabel("|Id| [A]")
        fig.suptitle(
            f"z88 best-fit: 2T BSIM4 port (Stage-4 masked loss = {s4['loss']:.3f})\n"
            f"Newton convergence {100 * n_conv / max(n_total, 1):.0f} %  "
            f"({n_conv}/{n_total} biases)  ·  "
            f"●=Sebas data, ─=fit (only on converged biases)",
            fontsize=11, weight="bold",
        )
        fig.tight_layout()
        fig.savefig(OUT / "fit_curves.png", dpi=140)
        plt.close(fig)
        print(f"\nWrote {OUT/'fit_curves.png'}  (eval took {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
