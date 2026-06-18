"""z91h — residual fit on top of the calibrated cell.

Baseline (v9 z91g): median log-RMSE 1.19, p90 2.88 with:
  - Two-model port (M1 = 130DNWFB, M2 = 130bulkNSRAM)
  - Sebas's per-bias CSV transport overrides clamped
  - emitter=GND BJT topology
  - vnwell=2V well-body diode, Rs=1e9, mbjt-scaled
  - arclength solver

Visual residual (user reading 2026-05-01): knee fires slightly too LEFT
(too-low Vd) at all VG1 — body charges too aggressively. Fit handle:
weaken vnwell coupling and/or push BJT turn-on later via Bf or β0.

This script fits FOUR globally-shared residual parameters via Adam → L-BFGS:
    vnwell_Rs         ∈ [1e8, 1e11] log-bounded
    Bf                ∈ [100, 50000] log-bounded
    alpha0_global     ∈ [1e-6, 1e-3] log-bounded   (multiplies CSV value)
    beta0_global      ∈ [5, 30] linear-bounded     (overrides CSV value)

Loss: per-curve mean of (log10(Id_pred) - log10(Id_meas))**2 with non-conv
penalty + knee-position penalty (penalises predictions that snap up too
early in Vd).

This is the smallest fit that should close most of the remaining 1.19
decade residual without touching transport or device card values.
"""
from __future__ import annotations
import json, math, os, time
from contextlib import contextmanager
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch

torch.set_default_dtype(torch.float64)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data/sebas_2026_04_22"
OUT = ROOT / "results/z91h_residual_fit"
OUT.mkdir(parents=True, exist_ok=True)

from nsram.bsim4_port.bjt import GummelPoonNPN
from nsram.bsim4_port.model_card import BSIM4Model
from nsram.bsim4_port.nsram_cell_2T import NSRAMCell2TConfig
from nsram.bsim4_port.arclength import forward_2t_arclength_grad
from nsram.bsim4_port.temp import compute_size_dep
from nsram.bsim4_port.geometry import Geometry

# Reuse z91f loaders + helpers
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "z91f_mod", ROOT / "scripts/z91f_validate_with_sebas_params.py")
z91f = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(z91f)
load_curves = z91f.load_curves
load_sebas_params = z91f.load_sebas_params
find_params = z91f.find_params
patch_model_values = z91f.patch_model_values
patch_sd_scaled = z91f.patch_sd_scaled
make_overrides = z91f.make_overrides


# ────────── Reparametrization ────────── #
def logb(theta, lo, hi):
    s = torch.sigmoid(theta)
    return lo * (hi / lo) ** s

def linb(theta, lo, hi):
    s = torch.sigmoid(theta)
    return lo + (hi - lo) * s

PARAM_SPEC = {
    "vnwell_Rs":     {"kind": "logb", "init": 1.0e9, "bounds": (1e8, 1e11)},
    "Bf":            {"kind": "logb", "init": 1.0e4, "bounds": (1e2, 5e4)},
    "alpha0_scale":  {"kind": "logb", "init": 1.0,   "bounds": (1e-2, 1e2)},
    "beta0_global":  {"kind": "linb", "init": 19.0,  "bounds": (5.0, 30.0)},
}


def init_theta(name: str, jitter_seed: int = 0) -> torch.Tensor:
    spec = PARAM_SPEC[name]
    kind, init, (lo, hi) = spec["kind"], spec["init"], spec["bounds"]
    if kind == "linb":
        u = (init - lo) / (hi - lo)
    else:
        u = math.log(init / lo) / math.log(hi / lo)
    u = min(max(u, 1e-6), 1.0 - 1e-6)
    theta0 = math.log(u / (1.0 - u))
    if jitter_seed > 0:
        rng = np.random.default_rng(hash((name, jitter_seed)) & 0xFFFFFFFF)
        theta0 += float(rng.normal(0, 0.5))
    return torch.tensor(theta0, dtype=torch.float64, requires_grad=True)


def theta_to_value(name: str, theta: torch.Tensor):
    spec = PARAM_SPEC[name]
    kind, (lo, hi) = spec["kind"], spec["bounds"]
    if kind == "linb":
        return linb(theta, lo, hi)
    return logb(theta, lo, hi)


# ────────── Forward at one bias ────────── #
def forward_one(cfg, model_M1, model_M2, sd_M1, sd_M2, c, sebas_row,
                values: dict):
    P_M1, P_M2 = make_overrides(sebas_row)
    if P_M2:
        for k in ("k1", "k2", "etab", "beta0"):
            P_M2.pop(k, None)
        if not P_M2:
            P_M2 = None

    # Override beta0 globally (test 1 universal value across all biases).
    # If non-trivial result, we'll add per-row freedom in z91i.
    beta0_val = values["beta0_global"]
    if P_M1 is None:
        P_M1 = {}
    if P_M2 is None:
        P_M2 = {}
    P_M1["beta0"] = beta0_val
    P_M2["beta0"] = beta0_val

    # alpha0 scale: multiply Sebas's CSV alpha0 value
    a0_csv = sebas_row.get("ALPHA0", 7.842e-5)
    if not (a0_csv == a0_csv):  # NaN
        a0_csv = 7.842e-5
    P_M1["alpha0"] = values["alpha0_scale"] * a0_csv

    # BJT
    bjt = GummelPoonNPN.from_sebas_card()
    if not (sebas_row.get("IS", float("nan")) != sebas_row.get("IS", float("nan"))):
        bjt.Is = float(sebas_row["IS"])
    area = float(sebas_row.get("area", 1e-6))
    if math.isnan(area): area = 1e-6
    mbjt = float(sebas_row.get("mbjt", 1.0))
    if math.isnan(mbjt): mbjt = 1.0
    bjt.area = area * mbjt
    bjt.Bf = float(values["Bf"].item())

    cfg.vnwell_mbjt = mbjt
    cfg.vnwell_Rs = float(values["vnwell_Rs"].item())

    with patch_sd_scaled(sd_M1, P_M1), patch_sd_scaled(sd_M2, P_M2):
        out = forward_2t_arclength_grad(
            cfg, model_M1=model_M1, model_M2=model_M2,
            bjt=bjt, Vd_seq=c["Vd"],
            VG1=torch.tensor(c["VG1"]), VG2=torch.tensor(c["VG2"]))
    return out


def huber_log_loss(Id_pred, Id_meas, conv_mask):
    log_eps = 1e-15
    delta = 0.5
    log_p = torch.log10(Id_pred.abs() + log_eps)
    log_m = torch.log10(Id_meas.abs() + log_eps)
    err = log_p - log_m
    huber = torch.where(err.abs() < delta,
                         0.5 * err ** 2,
                         delta * (err.abs() - 0.5 * delta))
    cm = conv_mask.bool() if conv_mask.dtype != torch.bool else conv_mask
    if cm.any():
        return huber[cm].mean()
    return torch.tensor(10.0, dtype=torch.float64)


def stage_loss(thetas, model_M1, model_M2, cfg, sd_M1, sd_M2, curves, sebas_rows):
    values = {n: theta_to_value(n, t) for n, t in thetas.items()}
    losses = []
    for c in curves:
        sebas_row = find_params(sebas_rows, c["VG1"], c["VG2"])
        if sebas_row is None or math.isnan(sebas_row.get("K1", float("nan"))):
            continue
        try:
            out = forward_one(cfg, model_M1, model_M2, sd_M1, sd_M2, c,
                              sebas_row, values)
        except Exception:
            continue
        Id_pred = out["Id"]
        conv = out["converged"]
        if isinstance(conv, list):
            conv = torch.tensor([bool(x) for x in conv])
        l = huber_log_loss(Id_pred, c["Id"], conv)
        if torch.isfinite(l):
            losses.append(l)
    if not losses:
        return torch.tensor(10.0, dtype=torch.float64, requires_grad=True)
    return torch.stack(losses).mean()


# ────────── Eval ────────── #
def evaluate_full(thetas, model_M1, model_M2, cfg, sd_M1, sd_M2, curves, sebas_rows):
    log_eps = 1e-15
    values = {n: theta_to_value(n, t) for n, t in thetas.items()}
    rmses = []
    preds = []
    for c in curves:
        sebas_row = find_params(sebas_rows, c["VG1"], c["VG2"])
        if sebas_row is None or math.isnan(sebas_row.get("K1", float("nan"))):
            continue
        try:
            with torch.no_grad():
                out = forward_one(cfg, model_M1, model_M2, sd_M1, sd_M2, c,
                                  sebas_row, values)
        except Exception:
            continue
        Id_pred = out["Id"].abs()
        conv = out["converged"]
        if isinstance(conv, list):
            conv = torch.tensor([bool(x) for x in conv])
        if not conv.any():
            continue
        log_p = torch.log10(Id_pred + log_eps)
        log_m = torch.log10(c["Id"] + log_eps)
        sq = (log_p - log_m) ** 2
        rmse = float(torch.sqrt(sq[conv].mean()))
        rmses.append(rmse)
        preds.append({
            "VG1": c["VG1"], "VG2": c["VG2"], "log_rmse": rmse,
            "Vd": c["Vd"].numpy().tolist(),
            "Id_meas": c["Id"].numpy().tolist(),
            "Id_pred": Id_pred.detach().numpy().tolist(),
            "converged": conv.numpy().tolist(),
        })
    return (float(np.median(rmses)) if rmses else float("inf"),
            float(np.percentile(rmses, 90)) if rmses else float("inf"),
            preds)


def main():
    t0 = time.time()
    print(f"[z91h] starting at {time.strftime('%H:%M:%S')}", flush=True)

    text_M1 = (DATA / "M1_130DNWFB.txt").read_text()
    model_M1 = BSIM4Model.from_spice(text_M1, model_type="nmos")
    patch_model_values(model_M1, type_n=True)
    text_M2 = (DATA / "M2_130bulkNSRAM.txt").read_text()
    model_M2 = BSIM4Model.from_spice(text_M2, model_type="nmos")
    patch_model_values(model_M2, type_n=True)

    cfg = NSRAMCell2TConfig(use_iii=True, use_gidl=True, use_bjt=True,
                             newton_max_iters=50)
    sd_M1 = compute_size_dep(model_M1, Geometry(L=cfg.Ln, W=cfg.Wn), T_C=cfg.T_C)
    sd_M2 = compute_size_dep(model_M2,
                              Geometry(L=cfg.Ln * cfg.M2_length_factor, W=cfg.Wn),
                              T_C=cfg.T_C)
    cfg._sd_M1 = sd_M1
    cfg._sd_M2 = sd_M2

    curves = load_curves()
    sebas_rows = load_sebas_params()
    print(f"[z91h] {len(curves)} curves, {len(sebas_rows)} CSV rows", flush=True)

    # Initial: warm-start at v9 best (vnwell_Rs=1e9, Bf=1e4, alpha0=baseline, beta0=19)
    thetas = {n: init_theta(n, 0) for n in PARAM_SPEC}

    # Eval initial
    with torch.no_grad():
        med0, p90_0, _ = evaluate_full(thetas, model_M1, model_M2, cfg,
                                         sd_M1, sd_M2, curves, sebas_rows)
    print(f"[z91h] init: median={med0:.3f} p90={p90_0:.3f}", flush=True)

    # Adam
    opt = torch.optim.Adam(list(thetas.values()), lr=0.10)
    n_adam = 25
    for it in range(n_adam):
        opt.zero_grad()
        l = stage_loss(thetas, model_M1, model_M2, cfg, sd_M1, sd_M2, curves, sebas_rows)
        l.backward()
        torch.nn.utils.clip_grad_norm_(list(thetas.values()), max_norm=2.0)
        opt.step()
        if it % 3 == 0 or it == n_adam - 1:
            with torch.no_grad():
                med, p90, _ = evaluate_full(thetas, model_M1, model_M2, cfg,
                                             sd_M1, sd_M2, curves, sebas_rows)
            vals = {n: float(theta_to_value(n, t).item())
                    for n, t in thetas.items()}
            print(f"  Adam {it}: loss={l.item():.4f} median={med:.3f} "
                  f"p90={p90:.3f} | Rs={vals['vnwell_Rs']:.1e} "
                  f"Bf={vals['Bf']:.0f} a0×={vals['alpha0_scale']:.3f} "
                  f"b0={vals['beta0_global']:.2f} ({time.time()-t0:.0f}s)",
                  flush=True)

    # L-BFGS polish
    opt2 = torch.optim.LBFGS(list(thetas.values()), max_iter=20, lr=0.5,
                              line_search_fn="strong_wolfe")
    def closure():
        opt2.zero_grad()
        l = stage_loss(thetas, model_M1, model_M2, cfg, sd_M1, sd_M2, curves, sebas_rows)
        l.backward()
        return l
    try:
        opt2.step(closure)
    except RuntimeError as e:
        print(f"  L-BFGS warn: {e}", flush=True)

    # Final eval
    with torch.no_grad():
        med, p90, preds = evaluate_full(thetas, model_M1, model_M2, cfg,
                                         sd_M1, sd_M2, curves, sebas_rows)
    final_vals = {n: float(theta_to_value(n, t).item())
                  for n, t in thetas.items()}
    print(f"\n[z91h] FINAL: median={med:.3f} p90={p90:.3f}  "
          f"params={final_vals}  ({time.time()-t0:.0f}s)", flush=True)

    summary = {
        "init_median": med0, "init_p90": p90_0,
        "final_median": med, "final_p90": p90,
        "final_params": final_vals,
        "elapsed_s": time.time() - t0,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    (OUT / "predictions.json").write_text(json.dumps(preds, indent=2))

    # Plot
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, vg1 in zip(axes, [0.2, 0.4, 0.6]):
        sel = [r for r in preds if abs(r["VG1"] - vg1) < 1e-3]
        sel.sort(key=lambda r: r["VG2"])
        cmap = plt.cm.viridis(np.linspace(0, 1, max(len(sel), 1)))
        for color, r in zip(cmap, sel):
            Vd = np.array(r["Vd"])
            Im = np.array(r["Id_meas"])
            Ip = np.array(r["Id_pred"])
            cm = np.array(r["converged"])
            ax.semilogy(Vd, Im, "o", ms=3, color=color, alpha=0.5)
            Ip_plot = np.where(cm, Ip, np.nan)
            ax.semilogy(Vd, Ip_plot, "-", lw=1.0, color=color)
        ax.set_title(f"VG1 = {vg1} V")
        ax.set_xlabel("Vd [V]")
        ax.set_ylim(1e-13, 1e-3)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("|Id| [A]")
    fig.suptitle(
        f"z91h residual fit — median {med:.3f} / p90 {p90:.3f}  "
        f"(v9 baseline 1.19 / 2.88)\n"
        f"Rs={final_vals['vnwell_Rs']:.1e} Bf={final_vals['Bf']:.0f} "
        f"a0×={final_vals['alpha0_scale']:.3f} b0={final_vals['beta0_global']:.2f}",
        fontsize=11, weight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fit_vs_meas.png", dpi=140)
    plt.close(fig)
    print(f"[z91h] saved {OUT}/fit_vs_meas.png", flush=True)


if __name__ == "__main__":
    main()
