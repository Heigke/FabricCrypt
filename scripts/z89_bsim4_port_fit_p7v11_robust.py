"""z89 — P7v11 robust fit: addresses two real issues from oracle deep RCA.

(1) Newton non-convergence at snapback biases (z88 had 3/20 fail). Fix:
    use ``solve_2t_with_homotopy`` (gmin homotopy) plus ``forward_2t``'s new
    ``dense_vd_in_snapback=True`` (4× denser warm-start grid for Vd>=1.4 V).

(2) Sebas's published fit makes alpha0/beta0/etc. POLYNOMIAL functions of
    (VG1, VG2). We're still fitting CONSTANT params to all 33 curves —
    fundamentally why the optimizer pushes params to bounds: it can't satisfy
    all 33 with one constant set. As a DIAGNOSTIC, this script fits THREE
    independent param sets, one per VG1 group (0.2, 0.4, 0.6 V), and writes
    a comparison table. If params differ >5× across groups, the polynomial
    bias-dependent form is required (z90 deliverable).

Usage::

    source venv/bin/activate
    python scripts/z89_bsim4_port_fit_p7v11_robust.py

This script is idempotent: results go to ``results/z89_bsim4_port_fit_p7v11_robust/``.
Per-group sub-folders ``vg1_0.2/``, ``vg1_0.4/``, ``vg1_0.6/``.
"""
from __future__ import annotations
import json
import math
import re
import time
from contextlib import contextmanager
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

torch.set_default_dtype(torch.float64)
DEVICE = torch.device("cpu")
torch.set_default_device(DEVICE)
print(f"[z89] Using device: {DEVICE}", flush=True)

from nsram.bsim4_port.bjt import GummelPoonNPN
from nsram.bsim4_port.model_card import BSIM4Model
from nsram.bsim4_port.nsram_cell_2T import (
    NSRAMCell2TConfig, forward_2t,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data/sebas_2026_04_22"
OUT = ROOT / "results/z89_bsim4_port_fit_p7v11_robust"
OUT.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Data loader                                                                 #
# --------------------------------------------------------------------------- #
def parse_vg2(s):
    m = re.search(r"VG2=(-?\d+\.\d+)", s)
    return float(m.group(1)) if m else None


def parse_vg1(s):
    m = re.search(r"VG1=([\d.]+)", s)
    return float(m.group(1)) if m else None


def load_curves():
    """Load Sebas's 33 I-V curves (VG1 ∈ {0.2, 0.4, 0.6} × VG2 sweep)."""
    curves = []
    for d in sorted(DATA_DIR.glob("2vHCa-2 I-Vs@VG2 VG1=*")):
        VG1 = parse_vg1(d.name)
        for f in sorted(d.glob("*.csv")):
            VG2 = parse_vg2(f.name)
            data = np.loadtxt(f, delimiter=",", skiprows=1, usecols=(0, 1))
            if data.ndim == 1:
                continue
            half = len(data) // 2
            Vd = data[:half, 0]
            Id = np.abs(data[:half, 1])
            mask = (Vd >= 0.05) & (Vd <= 2.0)
            Vd, Id = Vd[mask], Id[mask]
            if len(Vd) > 10:
                idx = np.linspace(0, len(Vd) - 1, 10).astype(int)
                Vd, Id = Vd[idx], Id[idx]
            if len(Vd) < 5:
                continue
            curves.append({
                "VG1": VG1, "VG2": VG2,
                "Vd": torch.tensor(Vd, dtype=torch.float64),
                "Id": torch.tensor(Id, dtype=torch.float64),
            })
    return curves


# --------------------------------------------------------------------------- #
# Param spec — same as z88. Each VG1 group fits its own copy.                 #
# --------------------------------------------------------------------------- #
PARAM_SPEC = {
    "vth0":  {"kind": "linb", "init": 0.4,   "bounds": (0.2, 0.6)},
    "u0":    {"kind": "logb", "init": 0.06,  "bounds": (0.02, 0.15)},
    "vsat":  {"kind": "logb", "init": 1e5,   "bounds": (5e4, 3e5)},
    "k1":    {"kind": "linb", "init": 0.5,   "bounds": (0.2, 0.9)},
    "k2":    {"kind": "linb", "init": 0.0,   "bounds": (-0.2, 0.2)},
    "agidl": {"kind": "logb", "init": 5e-7,  "bounds": (1e-7, 1e-5)},
    "bgidl": {"kind": "logb", "init": 8e8,   "bounds": (3e8, 1.2e9)},
    "cgidl": {"kind": "linb", "init": 0.5,   "bounds": (0.3, 0.7)},
    "egidl": {"kind": "linb", "init": 0.5,   "bounds": (0.3, 0.6)},
    "alpha0": {"kind": "logb", "init": 5e-3, "bounds": (1e-3, 5e-2)},
    "beta0":  {"kind": "linb", "init": 18.0, "bounds": (12.0, 30.0)},
    "Bf":     {"kind": "logb", "init": 100., "bounds": (50.0, 300.0)},
}

SCALED_KEYS = {"k1", "k2", "agidl", "bgidl", "cgidl", "egidl",
               "alpha0", "beta0"}
ATTR_KEYS = {"vth0": "vth0_T", "u0": "u0temp", "vsat": "vsattemp"}


# --------------------------------------------------------------------------- #
# Reparametrization (same as z88)                                             #
# --------------------------------------------------------------------------- #
def init_theta(name: str, jitter_seed: int = 0) -> torch.Tensor:
    spec = PARAM_SPEC[name]
    kind, init, bnd = spec["kind"], spec["init"], spec["bounds"]
    rng = np.random.default_rng(hash((name, jitter_seed)) & 0xFFFFFFFF)
    lo, hi = bnd
    if kind == "linb":
        u = (float(init) - lo) / (hi - lo)
    else:
        u = math.log(float(init) / lo) / math.log(hi / lo)
    u = min(max(u, 1e-6), 1.0 - 1e-6)
    theta0 = math.log(u / (1.0 - u))
    if jitter_seed != 0:
        theta0 += float(rng.normal(0, 0.3))
    return torch.tensor(theta0, dtype=torch.float64, requires_grad=True)


def theta_to_value(name: str, theta: torch.Tensor):
    spec = PARAM_SPEC[name]
    kind, bnd = spec["kind"], spec["bounds"]
    lo, hi = bnd
    s = torch.sigmoid(theta)
    if kind == "linb":
        return lo + (hi - lo) * s
    return lo * (hi / lo) ** s


def make_thetas(seed: int = 0) -> dict:
    return {n: init_theta(n, seed) for n in PARAM_SPEC}


def thetas_to_values(thetas: dict) -> dict:
    return {n: theta_to_value(n, t) for n, t in thetas.items()}


def clone_thetas(thetas: dict) -> dict:
    return {n: t.detach().clone().requires_grad_(True) for n, t in thetas.items()}


def fitted_dict(thetas: dict) -> dict:
    return {n: float(theta_to_value(n, thetas[n]).detach().item()) for n in thetas}


# --------------------------------------------------------------------------- #
# SizeDep override — patch sd in place (grad-aware)                           #
# --------------------------------------------------------------------------- #
@contextmanager
def patch_sd(sd, values: dict):
    saved_scaled, saved_attr = {}, {}
    try:
        for name, val in values.items():
            if name in SCALED_KEYS:
                saved_scaled[name] = sd.scaled.get(name, None)
                sd.scaled[name] = val
            elif name in ATTR_KEYS:
                attr = ATTR_KEYS[name]
                saved_attr[attr] = getattr(sd, attr)
                setattr(sd, attr, val)
        yield
    finally:
        for k, v in saved_scaled.items():
            if v is None:
                sd.scaled.pop(k, None)
            else:
                sd.scaled[k] = v
        for k, v in saved_attr.items():
            setattr(sd, k, v)


# --------------------------------------------------------------------------- #
# Build cfg + sd                                                              #
# --------------------------------------------------------------------------- #
def make_cfg_and_sd(model: BSIM4Model, gates: dict) -> NSRAMCell2TConfig:
    cfg = NSRAMCell2TConfig(
        Ln=180e-9, Wn=360e-9, M2_length_factor=10.0, T_C=27.0,
        use_iii=gates.get("use_iii", True),
        use_gidl=gates.get("use_gidl", True),
        use_bjt=gates.get("use_bjt", True),
    )
    cfg.size_dep_M1(model)
    cfg.size_dep_M2(model)
    return cfg


# --------------------------------------------------------------------------- #
# Forward — uses homotopy + dense Vd grid (z89 robustness improvements)       #
# --------------------------------------------------------------------------- #
def forward_curve(values: dict, model: BSIM4Model, cfg: NSRAMCell2TConfig,
                  VG1: float, VG2: float, Vd_seq: torch.Tensor,
                  use_homotopy: bool = True,
                  dense_vd: bool = True):
    """Solve the 2T cell across `Vd_seq` and return (|Id|, conv_mask).

    Robustness knobs (z89):
      use_homotopy=True → gmin homotopy at every Vd point (slower but
                          converges through snapback bistability).
      dense_vd=True     → 4× denser warm-start grid in Vd≥1.4V band.
    """
    sd_M1 = cfg._sd_M1
    sd_M2 = cfg._sd_M2
    bjt = GummelPoonNPN.from_sebas_card()
    if "Bf" in values:
        bjt.Bf = values["Bf"]
    VG1_t = torch.tensor(VG1, dtype=torch.float64)
    VG2_t = torch.tensor(VG2, dtype=torch.float64)
    with patch_sd(sd_M1, values), patch_sd(sd_M2, values):
        result = forward_2t(cfg, model, bjt, Vd_seq, VG1_t, VG2_t,
                            warm_start=True,
                            use_homotopy=use_homotopy,
                            dense_vd_in_snapback=dense_vd)
    converged_mask = torch.tensor([bool(c) for c in result["converged"]],
                                   dtype=torch.float64)
    return result["Id"].abs(), converged_mask


# --------------------------------------------------------------------------- #
# Loss                                                                        #
# --------------------------------------------------------------------------- #
def balanced_logmse(pred: torch.Tensor, meas: torch.Tensor,
                    mask: torch.Tensor = None) -> torch.Tensor:
    log_eps = 1e-15
    log_p = torch.log10(pred.abs() + log_eps)
    log_m = torch.log10(meas.abs() + log_eps)
    err = log_p - log_m
    dmeas = (log_m[1:] - log_m[:-1]).abs()
    weights = 1.0 + 5.0 * dmeas
    weights = torch.cat([weights, weights[-1:]])
    if mask is None:
        mask = torch.ones_like(err)
    sq = (err ** 2) * mask
    n = mask.sum().clamp_min(1.0)
    mse_uniform = sq.sum() / n
    knee_w = weights * mask
    mse_knee = (sq * weights).sum() / knee_w.sum().clamp_min(1.0)
    return 0.5 * mse_uniform + 0.5 * mse_knee


def stage_loss(thetas, model, cfg, curves, *, use_homotopy=True, dense_vd=True):
    values = thetas_to_values(thetas)
    losses = []
    for c in curves:
        try:
            Id_pred, conv_mask = forward_curve(values, model, cfg,
                                                c["VG1"], c["VG2"], c["Vd"],
                                                use_homotopy=use_homotopy,
                                                dense_vd=dense_vd)
        except RuntimeError as e:
            print(f"    skip VG1={c['VG1']} VG2={c['VG2']}: {e}", flush=True)
            continue
        l = balanced_logmse(Id_pred, c["Id"], mask=conv_mask)
        if torch.isfinite(l):
            losses.append(l)
    if not losses:
        return torch.tensor(1e3, dtype=torch.float64, requires_grad=True)
    return torch.stack(losses).mean()


# --------------------------------------------------------------------------- #
# Stage runner                                                                #
# --------------------------------------------------------------------------- #
def run_stage(stage_id, label, thetas, model, cfg, curves, fit_names,
              *, n_adam, n_lbfgs, lr_adam=0.05, lr_lbfgs=0.5, t0=0.0,
              use_homotopy=True, dense_vd=True):
    fit_thetas = [thetas[n] for n in fit_names]
    for n in fit_names:
        thetas[n].requires_grad_(True)

    with torch.no_grad():
        l0 = stage_loss(thetas, model, cfg, curves,
                        use_homotopy=use_homotopy, dense_vd=dense_vd)
    print(f"[stage {stage_id}{label}] init loss = {l0.item():.4f}  "
          f"(fitting {len(fit_names)}, {len(curves)} curves)  "
          f"({time.time()-t0:.0f}s)", flush=True)

    if n_adam > 0:
        opt = torch.optim.Adam(fit_thetas, lr=lr_adam)
        for it in range(n_adam):
            opt.zero_grad()
            l = stage_loss(thetas, model, cfg, curves,
                           use_homotopy=use_homotopy, dense_vd=dense_vd)
            l.backward()
            torch.nn.utils.clip_grad_norm_(fit_thetas, max_norm=2.0)
            opt.step()
            if it % 5 == 0 or it == n_adam - 1:
                print(f"  s{stage_id}{label} Adam {it}: loss={l.item():.4f}  "
                      f"({time.time()-t0:.0f}s)", flush=True)

    if n_lbfgs > 0:
        opt2 = torch.optim.LBFGS(fit_thetas, max_iter=n_lbfgs, lr=lr_lbfgs,
                                  line_search_fn="strong_wolfe")

        def closure():
            opt2.zero_grad()
            l = stage_loss(thetas, model, cfg, curves,
                           use_homotopy=use_homotopy, dense_vd=dense_vd)
            l.backward()
            return l

        try:
            opt2.step(closure)
        except RuntimeError as e:
            print(f"  s{stage_id}{label} L-BFGS warn: {e}", flush=True)

    with torch.no_grad():
        lf = stage_loss(thetas, model, cfg, curves,
                        use_homotopy=use_homotopy, dense_vd=dense_vd)
    print(f"[stage {stage_id}{label}] final loss = {lf.item():.4f}  "
          f"({time.time()-t0:.0f}s)", flush=True)
    return float(lf.item())


def multistart_stage(stage_id, base_thetas, model, cfg, curves, fit_names,
                     *, n_adam, n_lbfgs, n_seeds=3, t0,
                     lr_adam=0.05, lr_lbfgs=0.5,
                     use_homotopy=True, dense_vd=True):
    best_loss = float("inf")
    best_thetas = None
    for seed in range(n_seeds):
        thetas = clone_thetas(base_thetas)
        if seed > 0:
            for n in fit_names:
                thetas[n] = init_theta(n, jitter_seed=seed)
        loss = run_stage(stage_id, f".s{seed}", thetas, model, cfg, curves,
                         fit_names, n_adam=n_adam, n_lbfgs=n_lbfgs, t0=t0,
                         lr_adam=lr_adam, lr_lbfgs=lr_lbfgs,
                         use_homotopy=use_homotopy, dense_vd=dense_vd)
        if loss < best_loss:
            best_loss = loss
            best_thetas = clone_thetas(thetas)
            print(f"  ** stage {stage_id} new best @ seed {seed}: "
                  f"loss={loss:.4f}", flush=True)
    return best_thetas, best_loss


# --------------------------------------------------------------------------- #
# Per-group eval                                                              #
# --------------------------------------------------------------------------- #
def evaluate_full(thetas, model, cfg, curves):
    log_eps = 1e-15
    values = thetas_to_values(thetas)
    rmses, preds = [], []
    n_conv = 0
    n_total = 0
    for c in curves:
        try:
            with torch.no_grad():
                Id_pred, conv_mask = forward_curve(values, model, cfg,
                                                    c["VG1"], c["VG2"], c["Vd"],
                                                    use_homotopy=True,
                                                    dense_vd=True)
        except RuntimeError as e:
            print(f"  eval skip VG1={c['VG1']} VG2={c['VG2']}: {e}", flush=True)
            continue
        n_conv += int(conv_mask.sum().item())
        n_total += int(len(c["Vd"]))
        log_p = torch.log10(Id_pred.abs() + log_eps)
        log_m = torch.log10(c["Id"].abs() + log_eps)
        rmse = float(torch.sqrt(((log_p - log_m) ** 2).mean()).item())
        rmses.append(rmse)
        preds.append({"VG1": c["VG1"], "VG2": c["VG2"], "log_rmse": rmse,
                      "Vd": c["Vd"].numpy().tolist(),
                      "Id_meas": c["Id"].numpy().tolist(),
                      "Id_pred": Id_pred.detach().numpy().tolist(),
                      "converged": conv_mask.numpy().tolist()})
    if not rmses:
        return float("inf"), float("inf"), float("inf"), preds, (0, 0)
    median = float(np.median(rmses))
    p95 = float(np.percentile(rmses, 95))
    mx = float(np.max(rmses))
    return median, p95, mx, preds, (n_conv, n_total)


# --------------------------------------------------------------------------- #
# Per-group fit pipeline                                                      #
# --------------------------------------------------------------------------- #
def fit_one_group(vg1: float, model, all_curves, *, t0: float, group_out: Path) -> dict:
    """Fit a single VG1 group through 4 stages. Returns summary dict."""
    group_out.mkdir(parents=True, exist_ok=True)
    g_curves = [c for c in all_curves if abs(c["VG1"] - vg1) < 1e-3]
    print(f"\n{'='*70}\n[group VG1={vg1}] {len(g_curves)} curves\n{'='*70}",
          flush=True)

    # For VG1=0.2 we ARE the off-state set; for higher VG1 we still need an
    # "off-like" sub-fit for GIDL — use the lowest-VG2 curves as proxy.
    off_state = sorted(g_curves, key=lambda c: c["VG2"])[:max(2, len(g_curves) // 3)]
    low_vd = []
    for c in g_curves:
        mask = c["Vd"] < 0.8
        if int(mask.sum().item()) >= 4:
            low_vd.append({"VG1": c["VG1"], "VG2": c["VG2"],
                           "Vd": c["Vd"][mask], "Id": c["Id"][mask]})
    print(f"  off_state subset:    {len(off_state)}", flush=True)
    print(f"  low_vd (Vd<0.8):     {len(low_vd)}", flush=True)
    print(f"  full group:          {len(g_curves)}", flush=True)

    thetas = make_thetas(seed=0)

    # Stage 1: off-state, GIDL+vth0 (BJT off, Iii off)
    cfg1 = make_cfg_and_sd(model, gates={"use_iii": False, "use_gidl": True,
                                          "use_bjt": False})
    s1_names = ["agidl", "bgidl", "cgidl", "egidl", "vth0"]
    print(f"\n--- [VG1={vg1}] Stage 1: GIDL + vth0 ---", flush=True)
    thetas, l1 = multistart_stage(1, thetas, model, cfg1, off_state,
                                   s1_names, n_adam=20, n_lbfgs=10,
                                   n_seeds=2, t0=t0)
    (group_out / "stage1_summary.json").write_text(
        json.dumps({"loss": l1, "params": fitted_dict(thetas)}, indent=2))

    # Stage 2: core transport
    cfg2 = make_cfg_and_sd(model, gates={"use_iii": False, "use_gidl": True,
                                          "use_bjt": False})
    s2_names = ["u0", "vsat", "vth0", "k1", "k2"]
    print(f"\n--- [VG1={vg1}] Stage 2: Core transport ---", flush=True)
    thetas, l2 = multistart_stage(2, thetas, model, cfg2, low_vd,
                                   s2_names, n_adam=20, n_lbfgs=10,
                                   n_seeds=2, t0=t0)
    (group_out / "stage2_summary.json").write_text(
        json.dumps({"loss": l2, "params": fitted_dict(thetas)}, indent=2))

    # Stage 3: snapback (Iii + BJT)
    cfg3 = make_cfg_and_sd(model, gates={"use_iii": True, "use_gidl": True,
                                          "use_bjt": True})
    s3_names = ["alpha0", "beta0", "Bf"]
    print(f"\n--- [VG1={vg1}] Stage 3: Snapback (alpha0+beta0+Bf) ---", flush=True)
    thetas, l3 = multistart_stage(3, thetas, model, cfg3, g_curves,
                                   s3_names, n_adam=20, n_lbfgs=10,
                                   n_seeds=1, t0=t0,
                                   lr_adam=0.005, lr_lbfgs=0.1)
    (group_out / "stage3_summary.json").write_text(
        json.dumps({"loss": l3, "params": fitted_dict(thetas)}, indent=2))

    # Stage 4: final L-BFGS polish
    cfg4 = make_cfg_and_sd(model, gates={"use_iii": True, "use_gidl": True,
                                          "use_bjt": True})
    s4_names = list(PARAM_SPEC.keys())
    print(f"\n--- [VG1={vg1}] Stage 4: Final L-BFGS polish ---", flush=True)
    l4 = run_stage(4, "", thetas, model, cfg4, g_curves,
                   s4_names, n_adam=0, n_lbfgs=20, t0=t0)
    (group_out / "stage4_summary.json").write_text(
        json.dumps({"loss": l4, "params": fitted_dict(thetas)}, indent=2))

    median, p95, mx, preds, (n_conv, n_total) = evaluate_full(
        thetas, model, cfg4, g_curves)
    print(f"\n[group VG1={vg1}] log-RMSE median={median:.3f} "
          f"p95={p95:.3f} max={mx:.3f}  "
          f"converged={n_conv}/{n_total}", flush=True)

    summary = {
        "vg1_group": vg1,
        "n_curves": len(g_curves),
        "stage_losses": {"1": l1, "2": l2, "3": l3, "4": l4},
        "log_rmse": {"median": median, "p95": p95, "max": mx},
        "converged": {"n_conv": n_conv, "n_total": n_total},
        "fitted_params": fitted_dict(thetas),
    }
    (group_out / "summary.json").write_text(json.dumps(summary, indent=2))
    (group_out / "per_curve.json").write_text(json.dumps(preds, indent=1))
    return summary


# --------------------------------------------------------------------------- #
# Param comparison + plotting                                                 #
# --------------------------------------------------------------------------- #
def write_per_vg1_table(group_summaries: list, out_path: Path) -> None:
    """Write markdown comparison of fitted params across VG1 groups."""
    lines = ["# z89 — per-VG1 param comparison", ""]
    vg1s = [g["vg1_group"] for g in group_summaries]
    header = "| Param | " + " | ".join(f"VG1={v}" for v in vg1s) + " | Spread (max/min) | Note |"
    sep = "|" + "---|" * (len(vg1s) + 3)
    lines.append(header); lines.append(sep)

    rows = []
    for name in PARAM_SPEC:
        vals = [g["fitted_params"].get(name, float("nan")) for g in group_summaries]
        clean = [v for v in vals if v == v and v > 0]
        spread = max(clean) / min(clean) if (len(clean) == len(vals) and min(clean) > 0) else float("nan")
        if spread > 5:
            note = "STRONG VG1-dep — polynomial form needed"
        elif spread > 2:
            note = "moderate VG1-dep"
        else:
            note = "weak VG1-dep — constant OK"
        cells = " | ".join(f"{v:+.3e}" for v in vals)
        sp_cell = f"{spread:.2f}×" if spread == spread else "n/a"
        lines.append(f"| {name} | {cells} | {sp_cell} | {note} |")
        rows.append((name, vals, spread, note))

    lines.append("")
    lines.append("## Per-group fit quality")
    lines.append("")
    lines.append("| VG1 | Curves | log-RMSE median | p95 | max | Converged |")
    lines.append("|---|---|---|---|---|---|")
    for g in group_summaries:
        lines.append(
            f"| {g['vg1_group']} | {g['n_curves']} | "
            f"{g['log_rmse']['median']:.3f} | "
            f"{g['log_rmse']['p95']:.3f} | "
            f"{g['log_rmse']['max']:.3f} | "
            f"{g['converged']['n_conv']}/{g['converged']['n_total']} |"
        )

    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    n_strong = sum(1 for _, _, sp, _ in rows if sp == sp and sp > 5)
    n_mod = sum(1 for _, _, sp, _ in rows if sp == sp and 2 < sp <= 5)
    if n_strong >= 3:
        lines.append(
            f"**z90 needed**: {n_strong} params show >5× spread across VG1 groups, "
            "confirming Sebas's polynomial(VG1, VG2) bias-dependent param form is "
            "structurally necessary. Constant-param fits cannot satisfy all 33 curves "
            "simultaneously."
        )
    elif n_strong == 0 and n_mod <= 3:
        lines.append(
            "**Constant-param OK**: spreads <5× → polynomial form is NOT the "
            "bottleneck. Look elsewhere (forward-pass numerics, loss shape, etc.)."
        )
    else:
        lines.append(
            f"**Mixed signal**: {n_strong} strong, {n_mod} moderate. Some params do "
            "vary with VG1 but not all. Polynomial form may still help; consider "
            "promoting only the strongly-varying params (alpha0/beta0/Bf) to be "
            "polynomial in (VG1, VG2)."
        )
    out_path.write_text("\n".join(lines))


def plot_all_groups(group_summaries: list, out_path: Path) -> None:
    """3-column figure: each col = one VG1 group, all VG2 curves overlaid."""
    if not group_summaries:
        return
    fig, axes = plt.subplots(1, len(group_summaries),
                             figsize=(6 * len(group_summaries), 6),
                             sharey=True, squeeze=False)
    axes = axes[0]
    cmap = plt.get_cmap("viridis")
    for ax, g in zip(axes, group_summaries):
        per_curve_path = OUT / f"vg1_{g['vg1_group']}" / "per_curve.json"
        if not per_curve_path.exists():
            continue
        ps = sorted(json.loads(per_curve_path.read_text()),
                    key=lambda c: c["VG2"])
        n = len(ps)
        for i, p in enumerate(ps):
            color = cmap(i / max(n - 1, 1))
            Vd = np.asarray(p["Vd"])
            ax.semilogy(Vd, p["Id_meas"], "o", color=color, ms=4, alpha=0.7,
                         label=f"VG2={p['VG2']:+.2f}")
            ax.semilogy(Vd, p["Id_pred"], "-", color=color, lw=1.5)
        ax.set_xlabel("Vd [V]"); ax.grid(alpha=0.3)
        ax.set_title(
            f"VG1 = {g['vg1_group']} V    ({n} curves)\n"
            f"log-RMSE median={g['log_rmse']['median']:.2f}"
        )
        ax.legend(loc="lower right", fontsize=7, ncol=2)
    axes[0].set_ylabel("|Id| [A]")
    fig.suptitle(
        "z89 P7v11 — per-VG1 group fit (gmin homotopy + dense Vd snapback grid)",
        fontsize=12, weight="bold"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def main():
    t0 = time.time()
    base_card_text = (DATA_DIR / "PTM130bulkNSRAM.txt").read_text()
    model = BSIM4Model.from_spice(base_card_text)
    curves = load_curves()
    print(f"Loaded {len(curves)} curves at {time.time()-t0:.1f}s", flush=True)

    # Three groups (Sebas's data: VG1 ∈ {0.2, 0.4, 0.6})
    vg1_values = sorted({round(c["VG1"], 3) for c in curves})
    print(f"VG1 groups: {vg1_values}", flush=True)

    group_summaries = []
    for vg1 in vg1_values:
        group_out = OUT / f"vg1_{vg1}"
        summary = fit_one_group(vg1, model, curves, t0=t0, group_out=group_out)
        group_summaries.append(summary)

    # Comparison table + figure
    write_per_vg1_table(group_summaries, OUT / "per_vg1_params.md")
    plot_all_groups(group_summaries, OUT / "fit_curves.png")

    overall = {
        "groups": group_summaries,
        "elapsed_s": time.time() - t0,
        "config": "P7v11 — homotopy + dense Vd + per-VG1 group fit",
    }
    (OUT / "summary.json").write_text(json.dumps(overall, indent=2, default=str))

    print("\n" + "=" * 70)
    print("z89 fit complete. Per-VG1 param comparison:")
    print("  " + str(OUT / "per_vg1_params.md"))
    print("  " + str(OUT / "fit_curves.png"))
    print(f"Elapsed: {overall['elapsed_s']:.0f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
