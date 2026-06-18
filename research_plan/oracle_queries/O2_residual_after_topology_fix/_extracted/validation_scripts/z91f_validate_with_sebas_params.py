"""z91f — End-to-end validation against Sebastian's extracted parameters.

Sebas (2026-04-30 email + BSIMfitsBA package) sent us:
  • 130DNWFB(M1).txt  — M1 device card (deep N-well floating body)
  • 130bulkNSRAM(M2).txt — M2 device card (bulk)
  • 2Tcell_BSIM_param_DC.csv — fitted BSIM4 + BJT params per (VG1, VG2)

Per his email: "some parameters change only for M1, while NFACTOR changes
only for M2 (I attribute this to LDE)". So the CSV columns split as:
  M1 overrides: ETAB, K1, ALPHA0, BETA0   (LDE-driven, vary with VG1)
  M2 overrides: NFACTOR                    (LDE on M2, varies with VG2)
  BJT/wrapper:  trise, mbjt, IS, area

This script does NO fitting. It runs our forward simulator with Sebas's
exact extracted parameters at each (VG1, VG2) and compares to measurement.

If we match Sebas's published SPICE fit → port is end-to-end validated and
all earlier "loss=X" numbers were noise from us using one card for both
devices and treating constants-across-bias.

If we don't match → the deviation tells us exactly which sub-block of the
port disagrees with industry SPICE.

Usage
-----
    python scripts/z91f_validate_with_sebas_params.py
    → results/z91f_validate_sebas/{summary.json, fit_vs_meas.png}
"""
from __future__ import annotations
import json, math, re, time, csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

torch.set_default_dtype(torch.float64)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data/sebas_2026_04_22"
OUT = ROOT / "results/z91f_validate_sebas"
OUT.mkdir(parents=True, exist_ok=True)

from contextlib import contextmanager
from nsram.bsim4_port.bjt import GummelPoonNPN
from nsram.bsim4_port.model_card import BSIM4Model
from nsram.bsim4_port.nsram_cell_2T import NSRAMCell2TConfig, forward_2t
from nsram.bsim4_port.temp import compute_size_dep
from nsram.bsim4_port.geometry import Geometry


@contextmanager
def patch_sd_scaled(sd, overrides):
    """Override sd.scaled[name] entries (NOT sd attributes). Mirrors the
    `patch_sd` helper in z91d. Use this for BSIM4 params that live in the
    SizeDep.scaled dict (k1, k2, etab, alpha0, beta0, nfactor, ...).
    """
    if not overrides:
        yield
        return
    saved = {}
    try:
        for k, v in overrides.items():
            saved[k] = sd.scaled.get(k, None)
            sd.scaled[k] = v
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                sd.scaled.pop(k, None)
            else:
                sd.scaled[k] = v


# --------------------------------------------------------------------------- #
# Load measurement curves (same loader as z91d)
# --------------------------------------------------------------------------- #
def parse_vg2(s):
    m = re.search(r"VG2=(-?\d+\.\d+)", s);  return float(m.group(1)) if m else None
def parse_vg1(s):
    m = re.search(r"VG1=([\d.]+)", s);      return float(m.group(1)) if m else None


def load_curves():
    curves = []
    for d in sorted(DATA.glob("2vHCa-2 I-Vs@VG2 VG1=*")):
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
                idx = np.linspace(0, len(Vd) - 1, 30).astype(int)
                Vd, Id = Vd[idx], Id[idx]
                curves.append({"VG1": VG1, "VG2": VG2,
                               "Vd": torch.tensor(Vd, dtype=torch.float64),
                               "Id": torch.tensor(Id, dtype=torch.float64)})
    return curves


# --------------------------------------------------------------------------- #
# Load Sebastian's per-bias parameter CSV
# --------------------------------------------------------------------------- #
def load_sebas_params():
    path = DATA / "2Tcell_BSIM_param_DC.csv"
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            row = {}
            for k, v in r.items():
                try:
                    row[k] = float(v)
                except ValueError:
                    row[k] = float("nan")
            rows.append(row)
    return rows


def find_params(rows, VG1, VG2, atol=1e-3):
    for r in rows:
        if abs(r["VG1"] - VG1) < atol and abs(r["VG2"] - VG2) < atol:
            return r
    return None


# --------------------------------------------------------------------------- #
# .param block from M2 card (continuation lines our parser drops)            #
# --------------------------------------------------------------------------- #
# Both M1 and M2 cards reference shared symbols (vth0n, vsatn, lpe0n, …).    #
# M2 defines them at top via `.param ... + continuation`; our parser only    #
# captures the first line (toxn=4e-009). The rest fall back to BSIM4         #
# defaults — that is the 5-decade error we saw in z91f run #1. We apply      #
# them post-load to BOTH model_M1 and model_M2.                              #
SHARED_PARAM = {
    "toxn":   4e-9,
    "toxp":   4e-9,
    "lintn":  1.219e-8,
    "lintp": -1.079e-8,
    "vth0n":  0.54153,
    "vth0p": -1.106133,
    "lpe0n":  1.2439e-7,
    "lpe0p": -7.833656e-8,
    "k3n":    65.28,
    "k3p":   -7.18419,
    "pvth0n":-1.45e-15,
    "pvth0p": 5.543149e-16,
    "vsatn":  102230.0,
    "vsatp":  8.07584e4,
    "wintn":  4.7689e-8,
    "wintp":  4.268414e-9,
}

# Direct attribute substitutions applied to each model after load. These
# correspond to body lines like `+vth0 = vth0n` which our parser failed to
# resolve, leaving BSIM4 defaults in place. We patch the resolved values.
def patch_model_values(model, type_n: bool = True):
    s = "n" if type_n else "p"
    pmap = {
        "vth0":  SHARED_PARAM[f"vth0{s}"],
        "vsat":  SHARED_PARAM[f"vsat{s}"],
        "lpe0":  SHARED_PARAM[f"lpe0{s}"],
        "lint":  SHARED_PARAM[f"lint{s}"],
        "wint":  SHARED_PARAM[f"wint{s}"],
        "k3":    SHARED_PARAM[f"k3{s}"],
        "pvth0": SHARED_PARAM[f"pvth0{s}"],
        "toxe":  SHARED_PARAM[f"tox{s}"],
        "toxp":  SHARED_PARAM[f"tox{s}"],
        "toxm":  SHARED_PARAM[f"tox{s}"],
    }
    for k, v in pmap.items():
        model._values[k] = float(v)


# Static deltas in the BODY of M2 vs M1 (k1, etab, beta0 — verified by diff).
# These get applied to sd_M2.scaled at forward time. CSV per-bias overrides
# on top.
M2_STATIC_OVERRIDES = {
    "k1":    0.63825,
    "k2":   -0.070435,
    "etab": -0.086777,
    "beta0": 18.0,
}


def make_overrides(sebas_row):
    """Map a CSV row → (P_M1, P_M2) override dicts for forward_2t."""
    if sebas_row is None:
        return None, None
    # M1 overrides — bias-dependent parameters that Sebas attributes to LDE
    P_M1 = {}
    if not math.isnan(sebas_row.get("ETAB", float("nan"))):
        P_M1["etab"] = torch.tensor(sebas_row["ETAB"], dtype=torch.float64)
    if not math.isnan(sebas_row.get("K1", float("nan"))):
        P_M1["k1"] = torch.tensor(sebas_row["K1"], dtype=torch.float64)
    if not math.isnan(sebas_row.get("ALPHA0", float("nan"))):
        P_M1["alpha0"] = torch.tensor(sebas_row["ALPHA0"], dtype=torch.float64)
    if not math.isnan(sebas_row.get("BETA0", float("nan"))):
        P_M1["beta0"] = torch.tensor(sebas_row["BETA0"], dtype=torch.float64)

    # M2 overrides — NFACTOR varies with VG2 due to LDE on M2
    P_M2 = {}
    if not math.isnan(sebas_row.get("NFACTOR", float("nan"))):
        P_M2["nfactor"] = torch.tensor(sebas_row["NFACTOR"], dtype=torch.float64)

    # Always apply the M2 card's static deltas (not in CSV)
    for k, v in M2_STATIC_OVERRIDES.items():
        if k not in P_M2:
            P_M2[k] = torch.tensor(float(v), dtype=torch.float64)

    return P_M1 or None, P_M2 or None


def make_bjt(sebas_row):
    """Build per-bias BJT instance from a Sebas-CSV row.

    mbjt is SPICE's device-multiplicity `m=` parameter — same effect as
    scaling `area`. Sebas uses it to switch the parasitic-NPN path on
    (VG1=0.4/0.6 → mbjt=1) or essentially off (VG1=0.2 → mbjt=0.001).
    Honour it via `area *= mbjt` per A1b finding.
    """
    bjt = GummelPoonNPN.from_sebas_card()
    if sebas_row is not None:
        if not math.isnan(sebas_row.get("IS", float("nan"))):
            bjt.Is = float(sebas_row["IS"])
        area = float(sebas_row.get("area", 1e-6))
        if math.isnan(area):
            area = 1e-6
        mbjt = float(sebas_row.get("mbjt", 1.0))
        if math.isnan(mbjt):
            mbjt = 1.0
        bjt.area = area * mbjt
    return bjt


# --------------------------------------------------------------------------- #
# Run validation
# --------------------------------------------------------------------------- #
def main():
    t0 = time.time()
    print(f"[z91f] starting at {time.strftime('%H:%M:%S')}", flush=True)

    # Load M1 card; patch resolved-from-.param values our parser dropped
    text_M1 = (DATA / "M1_130DNWFB.txt").read_text()
    model = BSIM4Model.from_spice(text_M1, model_type="nmos")
    patch_model_values(model, type_n=True)
    print(f"[z91f] loaded M1 card (130DNWFB), patched shared .param "
          f"values: vth0={model.get('vth0')} vsat={model.get('vsat')}",
          flush=True)

    # Load M2 card separately for sd_M2's transport baselines (vth0_T,
    # vsattemp etc.). Same .param patch — both cards share these globals.
    text_M2 = (DATA / "M2_130bulkNSRAM.txt").read_text()
    model_M2 = BSIM4Model.from_spice(text_M2, model_type="nmos")
    patch_model_values(model_M2, type_n=True)
    print(f"[z91f] loaded M2 card (130bulkNSRAM), patched: "
          f"vth0={model_M2.get('vth0')} vsat={model_M2.get('vsat')} "
          f"k1={model_M2.get('k1')} etab={model_M2.get('etab')}", flush=True)

    cfg = NSRAMCell2TConfig(use_iii=True, use_gidl=True, use_bjt=True,
                             newton_max_iters=50)
    sd_M1 = compute_size_dep(model, Geometry(L=cfg.Ln, W=cfg.Wn), T_C=cfg.T_C)
    sd_M2 = compute_size_dep(model_M2,
                              Geometry(L=cfg.Ln * cfg.M2_length_factor,
                                       W=cfg.Wn), T_C=cfg.T_C)
    cfg._sd_M1 = sd_M1
    cfg._sd_M2 = sd_M2

    curves = load_curves()
    sebas_rows = load_sebas_params()
    print(f"[z91f] loaded {len(curves)} measured curves and "
          f"{len(sebas_rows)} CSV rows", flush=True)

    log_eps = 1e-15
    results = []
    for c in curves:
        sebas_row = find_params(sebas_rows, c["VG1"], c["VG2"])
        if sebas_row is None or math.isnan(sebas_row.get("K1", float("nan"))):
            results.append({"VG1": c["VG1"], "VG2": c["VG2"],
                            "skipped": True,
                            "reason": "no Sebas params (NaN row)"})
            continue
        P_M1, P_M2 = make_overrides(sebas_row)
        bjt = make_bjt(sebas_row)
        try:
            with torch.no_grad(), \
                 patch_sd_scaled(sd_M1, P_M1), \
                 patch_sd_scaled(sd_M2, P_M2):
                out = forward_2t(cfg, model, bjt,
                                  c["Vd"], torch.tensor(c["VG1"]),
                                  torch.tensor(c["VG2"]),
                                  warm_start=True, use_homotopy=True)
            Id_pred = out["Id"].abs()
            conv = torch.tensor([bool(x) for x in out["converged"]])
        except Exception as e:
            results.append({"VG1": c["VG1"], "VG2": c["VG2"],
                            "skipped": True, "reason": f"forward error: {e}"})
            continue

        log_p = torch.log10(Id_pred + log_eps)
        log_m = torch.log10(c["Id"] + log_eps)
        if conv.any():
            sq = (log_p - log_m) ** 2
            rmse = float(torch.sqrt(sq[conv].mean()))
        else:
            rmse = float("inf")
        results.append({
            "VG1": c["VG1"], "VG2": c["VG2"], "skipped": False,
            "log_rmse": rmse,
            "n_converged": int(conv.sum()),
            "n_total": int(len(conv)),
            "Vd": c["Vd"].numpy().tolist(),
            "Id_meas": c["Id"].numpy().tolist(),
            "Id_pred": Id_pred.numpy().tolist(),
            "converged": conv.numpy().tolist(),
        })
        print(f"  VG1={c['VG1']:.2f} VG2={c['VG2']:+.2f}: "
              f"log_rmse={rmse:.3f}  conv={int(conv.sum())}/{len(conv)}  "
              f"({time.time()-t0:.0f}s)", flush=True)

    rmses = [r["log_rmse"] for r in results
             if not r.get("skipped") and math.isfinite(r["log_rmse"])]
    median_rmse = float(np.median(rmses)) if rmses else float("inf")
    p90_rmse = float(np.percentile(rmses, 90)) if rmses else float("inf")

    summary = {
        "n_curves": len(curves),
        "n_evaluated": len(rmses),
        "n_skipped": sum(1 for r in results if r.get("skipped")),
        "median_log_rmse": median_rmse,
        "p90_log_rmse": p90_rmse,
        "elapsed_s": time.time() - t0,
        "note": "forward-only validation with Sebastian's extracted "
                "BSIM4 + BJT parameters (CSV) and M2 card static deltas",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    (OUT / "predictions.json").write_text(json.dumps(results, indent=2))
    print(f"\n[z91f] median log-RMSE = {median_rmse:.3f}  "
          f"p90 = {p90_rmse:.3f}", flush=True)

    # Plot grid: 3 columns by VG1
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, vg1 in zip(axes, [0.2, 0.4, 0.6]):
        sel = [r for r in results
               if not r.get("skipped") and abs(r["VG1"] - vg1) < 1e-3]
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
        f"z91f forward-only validation — Sebas's parameters → our simulator\n"
        f"o = measurement, line = prediction · "
        f"median log-RMSE = {median_rmse:.3f}  p90 = {p90_rmse:.3f}",
        fontsize=11, weight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fit_vs_meas.png", dpi=140)
    plt.close(fig)
    print(f"[z91f] saved {OUT}/fit_vs_meas.png", flush=True)


if __name__ == "__main__":
    main()
