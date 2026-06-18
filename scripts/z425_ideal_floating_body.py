#!/usr/bin/env python3
"""z425 / S16: Ideal Floating Body — pyport rebuild.

Implements the three physical fixes identified by audit:
  B) suppress_bulk_diode_forward — M1 bulk-S/D diode does NOT forward-conduct
     in deep-N-well isolated silicon (it is the *isolation* junction).
  C) q1_be_oneway — parasitic NPN's B-E junction only FIRES the BJT at
     V_BE > 0.7 V; it must NOT act as a passive forward diode draining
     the floating body.
  D) use_mario_ipos — inject Mario's Ipos = Iexp + Ipow at the body
     node B with PWL coefficients (a, b, d, e, f) digitized from
     slide 12.26 and parameterized by V_G1 (z422 Hyp B winner).

Fixes A (use_bsim4_eqns) and E (use_sebas_per_bias_fits) are already in
place in the pyport architecture (BSIM4Model is v4.8.3, P_M1/P_M2 path
already accepts Sebas's per-bias CSV overrides via z91f.make_overrides).

Driver runs 33-bias evaluation + 8-variant ablation, plots, gates.
"""
from __future__ import annotations
import json, math, time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)

ROOT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
DATA = ROOT / "data/sebas_2026_04_22"
PWLDIR = ROOT / "nsram/Zoom/ipos_pwl_digitized"
OUT = ROOT / "results/z425_ideal_floating_body"
OUT.mkdir(parents=True, exist_ok=True)

LOG = open(OUT / "run.log", "w")
def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.write(line + "\n"); LOG.flush()


# ──────────────────────────────────────────────────────────────────────
# Reuse z91f / z91g loaders to keep canonical paths identical
# ──────────────────────────────────────────────────────────────────────
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("z91f", ROOT / "scripts/z91f_validate_with_sebas_params.py")
z91f = _ilu.module_from_spec(_spec); _spec.loader.exec_module(z91f)

from nsram.bsim4_port.model_card import BSIM4Model, parse_param_blocks
from nsram.bsim4_port.nsram_cell_2T import NSRAMCell2TConfig, forward_2t
from nsram.bsim4_port.temp import compute_size_dep
from nsram.bsim4_port.geometry import Geometry

load_curves = z91f.load_curves
load_sebas_params = z91f.load_sebas_params
find_params = z91f.find_params
patch_model_values = z91f.patch_model_values
patch_sd_scaled = z91f.patch_sd_scaled
make_overrides = z91f.make_overrides
make_bjt = z91f.make_bjt


# ──────────────────────────────────────────────────────────────────────
# PWL loader — Mario Ipos coefficients
# ──────────────────────────────────────────────────────────────────────
PWL_FILES = {
    "a": "curve1_red_a_CALIBRATED.csv",
    "b": "curve3_red_b_middle_subplot_CALIBRATED.csv",
    "d": "curve2_blue_d_left_subplot.csv",
    "e": "curve4_blue_e_middle_subplot.csv",
    "f": "curve5_blue_f_right_subplot.csv",
}

def _load_pwl(p: Path):
    rows = []
    with open(p) as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln or ln.startswith("#") or ln.startswith("VG"):
                continue
            x, y = ln.split(",")
            rows.append((float(x), float(y)))
    arr = np.array(rows)
    idx = np.argsort(arr[:, 0])
    return arr[idx, 0], arr[idx, 1]

PWL = {nm: _load_pwl(PWLDIR / fn) for nm, fn in PWL_FILES.items()}


# ──────────────────────────────────────────────────────────────────────
# Build models (one-time, shared across variants)
# ──────────────────────────────────────────────────────────────────────
def build_models():
    text_M1 = (DATA / "M1_130DNWFB.txt").read_text()
    text_M2 = (DATA / "M2_130bulkNSRAM.txt").read_text()
    shared = parse_param_blocks(text_M2)
    m_M1 = BSIM4Model.from_spice(text_M1, model_type="nmos", params=shared)
    patch_model_values(m_M1, type_n=True)
    m_M2 = BSIM4Model.from_spice(text_M2, model_type="nmos", params=shared)
    patch_model_values(m_M2, type_n=True)
    return m_M1, m_M2


# ──────────────────────────────────────────────────────────────────────
# Run a single (variant flags) → cell-RMSE evaluation
# ──────────────────────────────────────────────────────────────────────
def run_variant(name: str, flags: dict, model_M1, model_M2, curves, sebas_rows,
                collect_traces: bool = False):
    cfg = NSRAMCell2TConfig(use_iii=True, use_gidl=True, use_bjt=True,
                            newton_max_iters=50)
    # Apply ablation flags
    cfg.suppress_bulk_diode_forward = flags.get("suppress_bulk_diode_forward", False)
    cfg.q1_be_oneway = flags.get("q1_be_oneway", False)
    cfg.use_mario_ipos = flags.get("use_mario_ipos", False)
    cfg.mario_ipos_param = flags.get("mario_ipos_param", "VG1")
    if cfg.use_mario_ipos:
        cfg.mario_ipos_pwl = PWL
    # Sebas per-bias fits (fix E) controlled by flag — when False, use card defaults
    use_sebas = flags.get("use_sebas_per_bias_fits", True)

    sd_M1 = compute_size_dep(model_M1, Geometry(L=cfg.Ln, W=cfg.Wn), T_C=cfg.T_C)
    sd_M2 = compute_size_dep(model_M2,
                             Geometry(L=cfg.Ln * cfg.M2_length_factor, W=cfg.Wn),
                             T_C=cfg.T_C)
    cfg._sd_M1 = sd_M1
    cfg._sd_M2 = sd_M2

    log_eps = 1e-15
    per_bias = []
    vb_max_overall = -1e30
    convergence_failures = 0
    traces = {}  # vg1 -> list of (vd, vb)
    t0 = time.time()
    for c in curves:
        sebas_row = find_params(sebas_rows, c["VG1"], c["VG2"])
        if sebas_row is None or math.isnan(sebas_row.get("K1", float("nan"))):
            continue
        if use_sebas:
            P_M1, P_M2 = make_overrides(sebas_row)
        else:
            P_M1, P_M2 = None, None
        bjt = make_bjt(sebas_row)
        try:
            with torch.no_grad(), \
                 patch_sd_scaled(sd_M1, P_M1), \
                 patch_sd_scaled(sd_M2, P_M2):
                out = forward_2t(cfg, model_M1, bjt,
                                 c["Vd"], torch.tensor(c["VG1"]),
                                 torch.tensor(c["VG2"]),
                                 model_M1=model_M1, model_M2=model_M2,
                                 warm_start=True, use_homotopy=True)
            Id_pred = out["Id"].abs()
            conv = torch.tensor([bool(x) for x in out["converged"]])
            Vb_arr = out["Vb"]
        except Exception as e:
            convergence_failures += 1
            log(f"  ERR {name} VG1={c['VG1']:.2f} VG2={c['VG2']:+.2f}: {e}")
            continue
        if not conv.any():
            convergence_failures += 1
            continue
        # cell-wide log RMSE
        log_p = torch.log10(Id_pred + log_eps)
        log_m = torch.log10(c["Id"] + log_eps)
        sq = (log_p - log_m) ** 2
        rmse = float(torch.sqrt(sq[conv].mean()))
        vb_max = float(Vb_arr.max())
        vb_max_overall = max(vb_max_overall, vb_max)
        per_bias.append({
            "VG1": c["VG1"], "VG2": c["VG2"],
            "log_rmse": rmse,
            "vb_max": vb_max,
            "Vd": c["Vd"].numpy().tolist(),
            "Id_meas": c["Id"].numpy().tolist(),
            "Id_pred": Id_pred.numpy().tolist(),
            "Vb": Vb_arr.numpy().tolist(),
            "converged": conv.numpy().tolist(),
        })
        if collect_traces:
            traces.setdefault(c["VG1"], []).append({
                "VG2": c["VG2"],
                "Vd": c["Vd"].numpy().tolist(),
                "Vb": Vb_arr.numpy().tolist(),
            })
    # Per-branch and cell-wide RMSE (squared, count-weighted)
    per_branch = {}
    cell_sq, cell_n = 0.0, 0
    for r in per_bias:
        b = f"VG1_{r['VG1']:.1f}"
        per_branch.setdefault(b, {"sq": 0.0, "n": 0})
        per_branch[b]["sq"] += (r["log_rmse"] ** 2)
        per_branch[b]["n"] += 1
        cell_sq += (r["log_rmse"] ** 2)
        cell_n += 1
    per_branch_rmse = {b: math.sqrt(v["sq"]/v["n"]) if v["n"] else float("inf")
                        for b, v in per_branch.items()}
    cell_rmse = math.sqrt(cell_sq / cell_n) if cell_n else float("inf")
    result = {
        "name": name,
        "flags": flags,
        "cell_rmse_dec": cell_rmse,
        "per_branch_rmse_dec": per_branch_rmse,
        "n_biases_evaluated": cell_n,
        "convergence_failures": convergence_failures,
        "vb_max_overall": vb_max_overall,
        "wall_sec": round(time.time() - t0, 1),
    }
    if collect_traces:
        result["traces"] = traces
        result["per_bias"] = per_bias  # for overlay plots
    log(f"  {name}: cell_rmse={cell_rmse:.3f}  per_branch={ {k:round(v,3) for k,v in per_branch_rmse.items()} }  Vb_max={vb_max_overall:.3f}  wall={result['wall_sec']}s")
    return result


# ──────────────────────────────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────────────────────────────
def plot_overlays(result: dict, out_dir: Path):
    """Overlay measured (dots) vs model (lines) on log scale per VG1 branch."""
    per_bias = result.get("per_bias", [])
    for vg1 in [0.2, 0.4, 0.6]:
        sel = [r for r in per_bias if abs(r["VG1"] - vg1) < 1e-3]
        if not sel:
            continue
        sel.sort(key=lambda r: r["VG2"])
        fig, ax = plt.subplots(figsize=(7, 5))
        cmap = plt.cm.viridis(np.linspace(0, 1, max(len(sel), 1)))
        for color, r in zip(cmap, sel):
            Vd = np.array(r["Vd"])
            Im = np.array(r["Id_meas"])
            Ip = np.array(r["Id_pred"])
            cm = np.array(r["converged"])
            ax.semilogy(Vd, Im, "o", ms=3, color=color, alpha=0.5,
                        label=f"VG2={r['VG2']:+.2f}")
            Ip_plot = np.where(cm, Ip, np.nan)
            ax.semilogy(Vd, Ip_plot, "-", lw=1.0, color=color)
        ax.set_title(f"z425 Ideal Floating Body — VG1={vg1} V  (cell RMSE={result['cell_rmse_dec']:.2f} dec)")
        ax.set_xlabel("V_D [V]")
        ax.set_ylabel("|I_D| [A]")
        ax.set_ylim(1e-13, 1e-3)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6, ncol=2, loc="lower right")
        fig.tight_layout()
        tag = f"{vg1:.1f}".replace(".", "p")
        fig.savefig(out_dir / f"overlay_VG1_{tag}.png", dpi=130)
        plt.close(fig)


def plot_vb_traces(result: dict, out_dir: Path):
    traces = result.get("traces", {})
    for vg1 in [0.2, 0.4, 0.6]:
        if vg1 not in traces:
            continue
        items = sorted(traces[vg1], key=lambda r: r["VG2"])
        fig, ax = plt.subplots(figsize=(7, 5))
        cmap = plt.cm.viridis(np.linspace(0, 1, max(len(items), 1)))
        for color, r in zip(cmap, items):
            ax.plot(r["Vd"], r["Vb"], "-", color=color, lw=1.0,
                    label=f"VG2={r['VG2']:+.2f}")
        ax.axhline(0.7, color="red", ls="--", alpha=0.5, lw=1, label="BJT threshold 0.7 V")
        ax.set_title(f"z425 V_B(V_D) trace — VG1={vg1} V")
        ax.set_xlabel("V_D [V]")
        ax.set_ylabel("V_B [V]")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6, ncol=2, loc="best")
        fig.tight_layout()
        tag = f"{vg1:.1f}".replace(".", "p")
        fig.savefig(out_dir / f"vb_trace_VG1_{tag}.png", dpi=130)
        plt.close(fig)


def plot_ablation_bar(ablation: dict, out_path: Path):
    names = list(ablation.keys())
    vals = [ablation[n]["cell_rmse_dec"] for n in names]
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["tab:blue" if "ALL_FLAGS" in n else
              ("tab:red" if "OFF_" in n else "tab:gray") for n in names]
    ax.bar(range(len(names)), vals, color=colors)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Cell-wide log-RMSE [decades]")
    ax.set_title("z425 ablation: each fix off vs ALL_FLAGS_ON")
    ax.axhline(2.0, color="green", ls="--", alpha=0.6, label="DISCOVERY gate 2.0 dec")
    ax.axhline(1.0, color="purple", ls="--", alpha=0.6, label="AMBITIOUS gate 1.0 dec")
    ax.axhline(3.5, color="red", ls="--", alpha=0.6, label="KILL_SHOT 3.5 dec")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.05, f"{v:.2f}", ha="center", fontsize=8)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def main():
    t_main = time.time()
    log(f"z425 starting — Ideal Floating Body pyport rebuild")
    log(f"PWL dir: {PWLDIR}")
    log(f"PWL coefficients loaded: {list(PWL.keys())}")

    model_M1, model_M2 = build_models()
    log(f"Models loaded: M1.vth0={model_M1.get('vth0')} M1.beta0={model_M1.get('beta0')}")

    curves = load_curves()
    sebas_rows = load_sebas_params()
    log(f"{len(curves)} measured curves, {len(sebas_rows)} CSV rows")

    # ── Variant set ──────────────────────────────────────────────────
    # ALL_FLAGS_ON is the headline result. Each ablation turns ONE fix off
    # to expose its contribution. BASELINE is no fixes (= existing pyport).
    ALL_ON = dict(
        suppress_bulk_diode_forward=True,
        q1_be_oneway=True,
        use_mario_ipos=True,
        mario_ipos_param="VG1",
        use_sebas_per_bias_fits=True,
    )
    variants = {
        "BASELINE":                 dict(use_sebas_per_bias_fits=True),
        "ALL_FLAGS_ON":             dict(ALL_ON),
        "OFF_suppress_bulk_diode":  {**ALL_ON, "suppress_bulk_diode_forward": False},
        "OFF_q1_be_oneway":         {**ALL_ON, "q1_be_oneway": False},
        "OFF_mario_ipos":           {**ALL_ON, "use_mario_ipos": False},
        "OFF_sebas_per_bias":       {**ALL_ON, "use_sebas_per_bias_fits": False},
        "ONLY_mario_ipos":          dict(use_mario_ipos=True, mario_ipos_param="VG1",
                                          use_sebas_per_bias_fits=True),
        "ALL_FLAGS_ON_HypA":        {**ALL_ON, "mario_ipos_param": "VG2"},
    }

    all_results = {}
    for name, flags in variants.items():
        collect = (name == "ALL_FLAGS_ON")
        log(f"Running variant: {name}  flags={flags}")
        all_results[name] = run_variant(name, flags, model_M1, model_M2,
                                        curves, sebas_rows, collect_traces=collect)

    # ── Plots ────────────────────────────────────────────────────────
    headline = all_results["ALL_FLAGS_ON"]
    plot_overlays(headline, OUT)
    plot_vb_traces(headline, OUT)

    # Strip per_bias / traces from saved JSON to keep file readable
    summary = {}
    for n, r in all_results.items():
        s = {k: v for k, v in r.items() if k not in ("per_bias", "traces")}
        summary[n] = s
    plot_ablation_bar(summary, OUT / "ablation_bar.png")

    # ── Gates ────────────────────────────────────────────────────────
    cell = headline["cell_rmse_dec"]
    vb_max = headline["vb_max_overall"]
    # Visible snapback check: detect a local Id maximum followed by a
    # ≥0.3-dec dip per VG1 branch in the headline.
    snapback_per_branch = {}
    for vg1 in [0.2, 0.4, 0.6]:
        sel = [r for r in headline["per_bias"] if abs(r["VG1"] - vg1) < 1e-3]
        found = False
        for r in sel:
            Id = np.array(r["Id_pred"])
            if Id.size < 5:
                continue
            i_pk = int(np.argmax(Id))
            if i_pk < Id.size - 2 and Id[i_pk] > 1e-9:
                tail_min = Id[i_pk:].min()
                if tail_min > 0 and Id[i_pk] / tail_min > 2.0:
                    found = True
                    break
        snapback_per_branch[f"VG1_{vg1}"] = bool(found)

    # Ablation deltas (each-flag-off rise vs ALL_ON)
    base_cell = headline["cell_rmse_dec"]
    abl_deltas = {}
    for n in ("OFF_suppress_bulk_diode", "OFF_q1_be_oneway",
              "OFF_mario_ipos", "OFF_sebas_per_bias"):
        abl_deltas[n] = round(all_results[n]["cell_rmse_dec"] - base_cell, 3)
    each_flag_helps_03 = all(d >= 0.3 for d in abl_deltas.values())

    gates = {
        "INFRA": (all_results["ALL_FLAGS_ON"]["convergence_failures"] == 0
                  and (time.time() - t_main) < 3 * 3600),
        "DISCOVERY": (cell < 2.0 and vb_max > 0.7 and all(snapback_per_branch.values())),
        "AMBITIOUS": (cell < 1.0 and each_flag_helps_03),
        "KILL_SHOT": (cell > 3.5),
    }

    summary_out = {
        "variants": summary,
        "headline_variant": "ALL_FLAGS_ON",
        "headline_cell_rmse_dec": cell,
        "headline_vb_max": vb_max,
        "snapback_per_branch": snapback_per_branch,
        "ablation_deltas_dec": abl_deltas,
        "gates": gates,
        "elapsed_total_s": round(time.time() - t_main, 1),
    }
    (OUT / "summary.json").write_text(json.dumps(summary_out, indent=2))
    (OUT / "ablation.json").write_text(json.dumps(abl_deltas, indent=2))
    log(f"Final cell RMSE (ALL_FLAGS_ON) = {cell:.3f} dec  Vb_max={vb_max:.3f} V")
    log(f"Ablation deltas: {abl_deltas}")
    log(f"Gates: {gates}")
    log(f"Wrote {OUT}/summary.json, ablation.json, plots.")
    return summary_out


if __name__ == "__main__":
    main()
