"""z70: Fit differentiable BSIM4 port to Sebas's 130nm I-V measurements.

Strategy:
  - Load 33 CSVs (VG1 in {0.2, 0.4, 0.6}; VG2 sweep) from data/sebas_2026_04_22/.
  - Forward sweep only (first 40 pts, dropping Vd~0 noise floor row).
  - Approximation: VG2 -> Vbs (body-bias) for the BSIM4 forward model.
  - Fit 8 params: vth0, voff, nfactor, u0, vsat, dvt0, dvt1, rdsw.
  - Loss: log-MSE + small linear term, normalized per curve.
  - Stage 1 Adam (lr=1e-2, 200 iter), Stage 2 L-BFGS (max_iter=50).
  - Cross-val: leave-one-VG1-out (3-fold).
  - Outputs: results/z70_bsim4_port_fit_sebas/{summary.json, fit_curves.png,
                                                 param_progress.png, cross_val.json}.
"""
from __future__ import annotations

import csv
import json
import math
import os
import re
import sys
import time
from copy import deepcopy
from glob import glob
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nsram"))

from nsram.bsim4_port.dc import compute_dc  # noqa: E402
from nsram.bsim4_port.geometry import Geometry  # noqa: E402
from nsram.bsim4_port.model_card import BSIM4Model  # noqa: E402
from nsram.bsim4_port.temp import compute_size_dep  # noqa: E402

DTYPE = torch.float64
DATA_ROOT = ROOT / "data" / "sebas_2026_04_22"
MODEL_CARD = DATA_ROOT / "PTM130bulkNSRAM.txt"
OUT_DIR = ROOT / "results" / "z70_bsim4_port_fit_sebas"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Number of forward-sweep points to keep per curve (drops first row near Vd=0
# and the back-sweep glitch starting around index 41).
N_FWD = 40

# Parameters we fit. log_space=True trains in log10-space (positive params).
# NOTE: rdsw is dropped because the BSIM4 port runs rdsmod=1 (Rds=0) and
# doesn't read rdsw in dc.py. We fit 7 params instead.
FIT_SPEC = [
    ("vth0", False),
    ("voff", False),
    ("nfactor", False),
    ("u0", True),
    ("vsat", True),
    ("dvt0", False),
    ("dvt1", False),
]


def parse_vg1(dir_name: str) -> float:
    m = re.search(r"VG1=([0-9.]+)", dir_name)
    if not m:
        raise ValueError(f"can't parse VG1 from {dir_name!r}")
    return float(m.group(1))


def parse_vg2(file_name: str) -> float:
    m = re.search(r"VG2=(-?[0-9.]+)", file_name)
    if not m:
        raise ValueError(f"can't parse VG2 from {file_name!r}")
    return float(m.group(1))


def load_curves():
    """Returns list of dicts with keys vg1, vg2, Vd (tensor), Id (tensor)."""
    curves = []
    for sub in sorted(os.listdir(DATA_ROOT)):
        full = DATA_ROOT / sub
        if not full.is_dir():
            continue
        if "VG1" not in sub:
            continue
        vg1 = parse_vg1(sub)
        for fp in sorted(glob(str(full / "*.csv"))):
            vg2 = parse_vg2(os.path.basename(fp))
            Vd_list, Id_list = [], []
            with open(fp) as f:
                r = csv.DictReader(f)
                for row in r:
                    Vd_list.append(float(row["vdata"]))
                    Id_list.append(float(row["idata"]))
            # Forward sweep, drop first noisy row near Vd=0
            Vd = Vd_list[1 : N_FWD + 1]
            Id = Id_list[1 : N_FWD + 1]
            Vd_t = torch.tensor(Vd, dtype=DTYPE)
            Id_t = torch.tensor(Id, dtype=DTYPE)
            # clip subthreshold negative-noise samples to 1e-13 floor
            Id_t = torch.clamp(Id_t, min=1e-13)
            curves.append({
                "vg1": vg1, "vg2": vg2,
                "Vd": Vd_t, "Id": Id_t,
                "src": fp,
            })
    return curves


# ---------------------------------------------------------------------------- #
# Fitting layer
# ---------------------------------------------------------------------------- #

class FitParams(torch.nn.Module):
    """Holds 8 fitable BSIM4 params as torch.nn.Parameters."""

    def __init__(self, init: dict):
        super().__init__()
        self.spec = FIT_SPEC
        self.log_space = {n: ls for n, ls in FIT_SPEC}
        for name, log_space in FIT_SPEC:
            v0 = float(init[name])
            if log_space:
                # log10 space; abs-ed (they're positive)
                v = math.log10(abs(v0))
            else:
                v = v0
            self.register_parameter(name, torch.nn.Parameter(torch.tensor(v, dtype=DTYPE)))

    def value(self, name: str) -> torch.Tensor:
        p = getattr(self, name)
        if self.log_space[name]:
            return 10.0 ** p
        return p

    def values_dict(self) -> dict:
        return {n: float(self.value(n).detach()) for n, _ in self.spec}


def make_sd_template(model: BSIM4Model, geom: Geometry, T_C: float):
    """Compute size-dep params once. Returns sd plus the temp-shift constants
    we need to re-apply when patching with parameter tensors."""
    sd = compute_size_dep(model, geom, T_C)
    # Temp-shift constants captured (snapshot from temp.py):
    #   vth0_T = scaled["vth0"] + vth0_shift
    #   u0temp = scaled["u0"] * (TRatio**ute)
    #   vsattemp = scaled["vsat"] - at*(TRatio-1)   (tempmod=0)
    vth0_shift = sd.vth0_T - sd.scaled["vth0"]
    u0_factor = sd.u0temp / max(sd.scaled["u0"], 1e-30)
    Tm1 = sd.model_ctx.TRatio - 1.0
    at = sd.scaled.get("at", 0.0)
    tempmod = int(model.get("tempmod", 0))
    if tempmod == 0:
        vsat_shift = -at * Tm1
        vsat_scale = 1.0
    else:
        vsat_shift = 0.0
        vsat_scale = 1.0 - at * sd.model_ctx.delTemp
    # voffcbn = voff + voffl/Leff; capture the additive piece so we can
    # rebuild voffcbn from a fitable voff parameter.
    voffl_term = sd.voffcbn - sd.scaled["voff"]
    return sd, {
        "vth0_shift": vth0_shift,
        "u0_factor": u0_factor,
        "vsat_scale": vsat_scale,
        "vsat_shift": vsat_shift,
        "voffl_term": voffl_term,
    }


def patch_sd(sd, fp: FitParams, shifts: dict):
    """Overwrite the 8 fitable params in `sd` with parameter tensors,
    propagating temp shifts. Mutates sd; safe between forward calls because
    nothing else reads stale floats for these names."""
    vth0 = fp.value("vth0")
    u0 = fp.value("u0")
    vsat = fp.value("vsat")
    sd.scaled["vth0"] = vth0
    sd.vth0_T = vth0 + shifts["vth0_shift"]
    sd.scaled["u0"] = u0
    sd.u0temp = u0 * shifts["u0_factor"]
    sd.scaled["vsat"] = vsat
    sd.vsattemp = vsat * shifts["vsat_scale"] + shifts["vsat_shift"]
    voff = fp.value("voff")
    sd.scaled["voff"] = voff
    sd.voffcbn = voff + shifts["voffl_term"]
    sd.scaled["nfactor"] = fp.value("nfactor")
    sd.scaled["dvt0"] = fp.value("dvt0")
    sd.scaled["dvt1"] = fp.value("dvt1")


def predict_curve(model, sd, vg1: float, vg2: float, Vd: torch.Tensor) -> torch.Tensor:
    Vgs = torch.full_like(Vd, vg1)
    Vbs = torch.full_like(Vd, vg2)  # VG2 ~ Vbs (back-gate / body bias)
    out = compute_dc(model, sd, Vgs, Vd, Vbs)
    return out.Ids


def loss_fn(I_pred: torch.Tensor, I_meas: torch.Tensor, lin_w: float = 0.1) -> torch.Tensor:
    eps = 1e-13
    log_resid = torch.log(torch.abs(I_pred) + eps) - torch.log(torch.abs(I_meas) + eps)
    log_mse = (log_resid ** 2).mean()
    Imax = I_meas.abs().max().clamp(min=1e-12)
    lin_resid = (I_pred - I_meas) / Imax
    lin_mse = (lin_resid ** 2).mean()
    return log_mse + lin_w * lin_mse


def total_loss(curves, model, sd, fp, shifts):
    patch_sd(sd, fp, shifts)
    losses = []
    for c in curves:
        I_pred = predict_curve(model, sd, c["vg1"], c["vg2"], c["Vd"])
        losses.append(loss_fn(I_pred, c["Id"]))
    return torch.stack(losses).mean()


# ---------------------------------------------------------------------------- #
# Train
# ---------------------------------------------------------------------------- #

def train(curves, model, sd, shifts, init_dict, n_adam=200, n_lbfgs=50, lr=1e-2,
          log_prefix=""):
    fp = FitParams(init_dict)
    history = {"iter": [], "loss": [], "params": []}

    opt = torch.optim.Adam(fp.parameters(), lr=lr)
    best_loss = float("inf")
    best_state = deepcopy(fp.state_dict())
    t0 = time.time()
    for it in range(n_adam):
        opt.zero_grad()
        L = total_loss(curves, model, sd, fp, shifts)
        if not torch.isfinite(L):
            print(f"  {log_prefix}adam iter {it}: NaN/Inf loss, stopping")
            break
        L.backward()
        torch.nn.utils.clip_grad_norm_(fp.parameters(), 5.0)
        opt.step()
        Lv = float(L.detach())
        if Lv < best_loss:
            best_loss = Lv
            best_state = deepcopy(fp.state_dict())
        if it % 10 == 0 or it == n_adam - 1:
            elapsed = time.time() - t0
            history["iter"].append(it)
            history["loss"].append(Lv)
            history["params"].append(fp.values_dict())
            print(f"  {log_prefix}adam {it:3d}: loss={Lv:.4f} best={best_loss:.4f} t={elapsed:.1f}s")

    # Restore best Adam state
    fp.load_state_dict(best_state)

    # L-BFGS polish
    lbfgs = torch.optim.LBFGS(fp.parameters(), lr=0.5, max_iter=n_lbfgs,
                               history_size=20, line_search_fn="strong_wolfe")

    lbfgs_iter = [0]

    def closure():
        lbfgs.zero_grad()
        L = total_loss(curves, model, sd, fp, shifts)
        L.backward()
        torch.nn.utils.clip_grad_norm_(fp.parameters(), 5.0)
        if lbfgs_iter[0] % 10 == 0:
            print(f"  {log_prefix}lbfgs {lbfgs_iter[0]:3d}: loss={float(L.detach()):.4f}")
        lbfgs_iter[0] += 1
        return L

    try:
        lbfgs.step(closure)
    except Exception as e:
        print(f"  {log_prefix}lbfgs aborted: {e}")

    final_L = float(total_loss(curves, model, sd, fp, shifts).detach())
    if final_L < best_loss:
        best_loss = final_L
        best_state = deepcopy(fp.state_dict())
    fp.load_state_dict(best_state)

    history["iter"].append(history["iter"][-1] + 1 if history["iter"] else 0)
    history["loss"].append(best_loss)
    history["params"].append(fp.values_dict())

    return fp, best_loss, history


# ---------------------------------------------------------------------------- #
# Eval / metrics
# ---------------------------------------------------------------------------- #

def per_curve_metrics(curves, model, sd, fp, shifts):
    patch_sd(sd, fp, shifts)
    rows = []
    with torch.no_grad():
        for c in curves:
            I_pred = predict_curve(model, sd, c["vg1"], c["vg2"], c["Vd"])
            I_meas = c["Id"]
            rmse = float(torch.sqrt(((I_pred - I_meas) ** 2).mean()))
            log_rmse = float(torch.sqrt(((torch.log(I_pred.abs() + 1e-13)
                                           - torch.log(I_meas.abs() + 1e-13)) ** 2).mean()))
            Imax = float(I_meas.abs().max())
            rows.append({
                "vg1": c["vg1"], "vg2": c["vg2"],
                "rmse_A": rmse, "rmse_rel": rmse / max(Imax, 1e-15),
                "log_rmse": log_rmse,
                "Imax_A": Imax,
            })
    return rows


# ---------------------------------------------------------------------------- #
# Plots
# ---------------------------------------------------------------------------- #

def plot_fits(curves, preds, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(curves)
    ncols = 6
    nrows = math.ceil(n / ncols)
    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * 2.5, nrows * 2.0),
                              sharex=True)
    axs = axs.flatten()
    for i, (c, I_pred) in enumerate(zip(curves, preds)):
        ax = axs[i]
        Vd = c["Vd"].cpu().numpy()
        Im = c["Id"].cpu().numpy()
        Ip = I_pred.cpu().numpy()
        ax.semilogy(Vd, Im, "k.", ms=2, label="meas")
        ax.semilogy(Vd, Ip, "r-", lw=1, label="fit")
        ax.set_title(f"VG1={c['vg1']:.1f} VG2={c['vg2']:+.2f}", fontsize=7)
        ax.tick_params(labelsize=6)
    for j in range(n, len(axs)):
        axs[j].axis("off")
    axs[0].legend(fontsize=6)
    fig.suptitle("z70 BSIM4 port fit to Sebas 130nm — measured vs fitted (log Id)",
                  fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_progress(history, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    its = history["iter"]
    losses = history["loss"]
    params_per_step = history["params"]
    fig, axs = plt.subplots(3, 3, figsize=(11, 8))
    axs = axs.flatten()
    axs[0].plot(its, losses, "b-")
    axs[0].set_title("loss")
    axs[0].set_yscale("log")
    axs[0].set_xlabel("iter")
    axs[0].grid(True, alpha=0.3)
    for i, (name, _) in enumerate(FIT_SPEC):
        ax = axs[i + 1]
        vals = [p[name] for p in params_per_step]
        ax.plot(its, vals, "k-")
        ax.set_title(name)
        ax.set_xlabel("iter")
        ax.grid(True, alpha=0.3)
    fig.suptitle("z70 fit param progression")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------- #
# Main
# ---------------------------------------------------------------------------- #

def main():
    print(f"[z70] loading data from {DATA_ROOT} ...")
    curves = load_curves()
    print(f"[z70] loaded {len(curves)} curves")
    by_vg1 = {}
    for c in curves:
        by_vg1.setdefault(c["vg1"], []).append(c)
    for k in sorted(by_vg1):
        print(f"  VG1={k}: {len(by_vg1[k])} curves")

    print("[z70] loading model card ...")
    with open(MODEL_CARD) as f:
        txt = f.read()
    model = BSIM4Model.from_spice(txt, model_type="nmos")
    geom = Geometry(L=130e-9, W=1e-6)

    init = {n: float(model.get(n)) for n, _ in FIT_SPEC}
    print(f"[z70] init params: {init}")

    sd, shifts = make_sd_template(model, geom, 27.0)

    # --- Full fit ---
    print("[z70] FULL FIT (all 33 curves) ...")
    fp, best_loss, history = train(curves, model, sd, shifts, init,
                                    n_adam=200, n_lbfgs=50, lr=1e-2,
                                    log_prefix="full|")
    final_params = fp.values_dict()
    print(f"[z70] FULL final loss = {best_loss:.4f}")
    print(f"[z70] FULL final params: {final_params}")

    # collect predictions for all curves
    patch_sd(sd, fp, shifts)
    preds = []
    with torch.no_grad():
        for c in curves:
            preds.append(predict_curve(model, sd, c["vg1"], c["vg2"], c["Vd"]).clone())

    per_curve = per_curve_metrics(curves, model, sd, fp, shifts)

    # --- Cross-val: leave one VG1 out ---
    print("[z70] CROSS-VAL (leave-one-VG1-out) ...")
    cv_results = {}
    vg1_groups = sorted(by_vg1.keys())
    for held in vg1_groups:
        train_curves = [c for c in curves if c["vg1"] != held]
        val_curves = [c for c in curves if c["vg1"] == held]
        sd_cv, shifts_cv = make_sd_template(model, geom, 27.0)
        print(f"  fold held={held}: train={len(train_curves)} val={len(val_curves)}")
        fp_cv, best_cv, _ = train(train_curves, model, sd_cv, shifts_cv, init,
                                    n_adam=120, n_lbfgs=30, lr=1e-2,
                                    log_prefix=f"cv{held}|")
        per_curve_val = per_curve_metrics(val_curves, model, sd_cv, fp_cv, shifts_cv)
        log_rmses = [r["log_rmse"] for r in per_curve_val]
        rel_rmses = [r["rmse_rel"] for r in per_curve_val]
        cv_results[f"VG1={held}"] = {
            "n_train": len(train_curves),
            "n_val": len(val_curves),
            "train_loss": best_cv,
            "median_log_rmse_held": float(torch.tensor(log_rmses).median()),
            "median_rel_rmse_held": float(torch.tensor(rel_rmses).median()),
            "params": fp_cv.values_dict(),
        }
        print(f"    median log_rmse held = {cv_results[f'VG1={held}']['median_log_rmse_held']:.3f}")

    # --- Save artifacts ---
    summary = {
        "n_curves": len(curves),
        "vg1_groups": {f"{k}": len(v) for k, v in by_vg1.items()},
        "init_params": init,
        "final_params": final_params,
        "final_train_loss": best_loss,
        "loss_history": [
            {"iter": i, "loss": l} for i, l in zip(history["iter"], history["loss"])
        ],
        "per_curve_metrics": per_curve,
        "median_log_rmse_train": float(torch.tensor([r["log_rmse"] for r in per_curve]).median()),
        "median_rel_rmse_train": float(torch.tensor([r["rmse_rel"] for r in per_curve]).median()),
        "fit_spec": [{"name": n, "log_space": ls} for n, ls in FIT_SPEC],
        "n_forward_pts_per_curve": N_FWD,
        "vg2_as_vbs_approx": True,
    }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[z70] wrote {OUT_DIR / 'summary.json'}")

    with open(OUT_DIR / "cross_val.json", "w") as f:
        json.dump(cv_results, f, indent=2)
    print(f"[z70] wrote {OUT_DIR / 'cross_val.json'}")

    plot_fits(curves, preds, OUT_DIR / "fit_curves.png")
    print(f"[z70] wrote {OUT_DIR / 'fit_curves.png'}")
    plot_progress(history, OUT_DIR / "param_progress.png")
    print(f"[z70] wrote {OUT_DIR / 'param_progress.png'}")

    # README
    readme = f"""# z70 BSIM4 port fit to Sebas 130nm I-V data

Curves loaded: {len(curves)} (VG1=0.2: {len(by_vg1[0.2])}, VG1=0.4: {len(by_vg1[0.4])}, VG1=0.6: {len(by_vg1[0.6])})
Forward sweep only: {N_FWD} points/curve (Vd: ~0.05 -> 2.0V)
VG2 mapped to Vbs (back-gate -> body-bias approximation).

Final training loss: {best_loss:.4f}
Median log-RMSE (train): {summary['median_log_rmse_train']:.3f}
Median rel-RMSE (train): {summary['median_rel_rmse_train']:.4f}

Cross-val held-out median log-RMSE:
""" + "\n".join(
        f"  {k}: {v['median_log_rmse_held']:.3f}"
        for k, v in cv_results.items()
    ) + "\n\nArtifacts: summary.json, cross_val.json, fit_curves.png, param_progress.png\n"
    with open(OUT_DIR / "README.md", "w") as f:
        f.write(readme)

    print("[z70] DONE")


if __name__ == "__main__":
    main()
