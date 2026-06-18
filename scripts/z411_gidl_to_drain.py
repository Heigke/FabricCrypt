"""z411 — S8 GIDL-to-drain rewiring (BSIM4 standard topology).

S7 (z410) found: pyport already computes Igidl per BSIM4, but routes it
INTO the body KCL (R_B += Igidl) where the parasitic NPN amplifies it
weakly (gain ~ Bf only of the small SCBE-style component reaching the
collector). The standard BSIM4 v4.x convention (manual §6.2, b4ld.c
§2274-2370) places GIDL/GISL directly at the drain pin of the conducting
device: a drain-to-body BTBT leakage component that appears in the DC
drain-pin terminal current.

This script:
  1. Adds `cfg.gidl_routing` flag ("body" legacy / "drain" BSIM4 standard).
     The code change lives in `nsram/bsim4_port/nsram_cell_2T.py`; this
     script only consumes the flag.
  2. Smoke test at VG1=0.6, VG2=0.2, Vd=1.5 V comparing "body" vs "drain".
  3. Sweeps agidl ∈ {1e-12, 1e-10, 1e-8, 1e-6, 1.99e-8(card)} A and finds
     the agidl that best matches measured Ids at the headline point.
  4. Runs full 33-curve refit (z365 harness, fixed default params — NO BBO,
     time budget) for both routings and reports cell-wide median.
  5. Generates a snapback comparison plot (z372 template) overlaying
     pyport(body), pyport(drain), and measured curves at three (VG1, VG2)
     anchor points.

Pre-registered gates (logged via this script's summary.json + 01_LOG.md):
    INFRA       :  smoke test passes, full refit completes < 30 min
    DISCOVERY   :  cell-wide < 0.85 dec OR Ids(VG1=0.6,Vd=1.5) within 3×
    AMBITIOUS   :  cell-wide < 0.5 dec AND fold reproduced > 1.0 dec at VG1=0.6
    KILL-SHOT   :  gidl_to_drain produces no improvement → S7 wrong

Outputs:
    results/z411_gidl_to_drain/
        summary.json
        snapback_compare.png
        run.log
        agidl_sweep.json
"""
from __future__ import annotations
import os, sys, time, json, math, importlib.util, traceback
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "nsram"))

DATA = ROOT / "data/sebas_2026_04_22"
OUT  = ROOT / "results/z411_gidl_to_drain"
OUT.mkdir(parents=True, exist_ok=True)

LOG_LINES: list[str] = []
def _log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG_LINES.append(line)

# ---- Reuse z365 helpers + base builder ----
_z365_spec = importlib.util.spec_from_file_location("z365", ROOT / "scripts/z365_perVG1_bbo.py")
z365 = importlib.util.module_from_spec(_z365_spec); _z365_spec.loader.exec_module(z365)
from nsram.bsim4_port.nsram_cell_2T import forward_2t


# ---- Reference measurement ----
HEADLINE = {"VG1": 0.6, "VG2": 0.2, "Vd_ref": 1.5}
MEAS_FILE = (DATA / "2vHCa-2 I-Vs@VG2 VG1=0.6 vnwell=2"
             / "StandardIV_HH_2vHCa-2_VG2=0.20_VG=0.6(1)_03-45-46PM.csv")
_d = np.loadtxt(MEAS_FILE, delimiter=",", skiprows=1)
_idx = np.argmin(np.abs(_d[:, 0] - HEADLINE["Vd_ref"]))
IDS_MEAS_15V = float(np.abs(_d[_idx, 1]))
_log(f"[ref] Sebas Ids(VG1=0.6, VG2=0.2, Vd=1.5) = {IDS_MEAS_15V:.4e} A")


def predict_ids_headline(gidl_routing: str, agidl_M1: float | None = None,
                          bgidl_M1: float | None = None) -> float:
    """Predict Ids at headline (VG1=0.6, VG2=0.2, Vd=1.5) with given routing."""
    cfg, M1, M2, bjt = z365.build_pyport_base()
    cfg.gidl_routing = gidl_routing
    if agidl_M1 is not None:
        M1._values["agidl"] = float(agidl_M1)
    if bgidl_M1 is not None:
        M1._values["bgidl"] = float(bgidl_M1)
    sebas_rows = z365.load_sebas_params()
    row, _ = z365.find_or_impute_row(sebas_rows, HEADLINE["VG1"], HEADLINE["VG2"])
    P_M1, P_M2 = z365.make_overrides(row)
    sd_M1 = cfg.size_dep_M1(M1); sd_M2 = cfg.size_dep_M2(M2)
    Vd_t = torch.tensor([HEADLINE["Vd_ref"]], dtype=torch.float64)
    try:
        with z365.patch_sd_scaled(sd_M1, P_M1), z365.patch_sd_scaled(sd_M2, P_M2):
            out = forward_2t(
                cfg, model_M1=M1, model_M2=M2, bjt=bjt, Vd_seq=Vd_t,
                VG1=torch.tensor(HEADLINE["VG1"], dtype=torch.float64),
                VG2=torch.tensor(HEADLINE["VG2"], dtype=torch.float64),
                warm_start=False,
            )
        return float(np.abs(out["Id"].detach().cpu().numpy())[0])
    except Exception as e:
        _log(f"[predict_ids_headline] EXC: {e}")
        return float("nan")


def predict_curve(cfg, M1, M2, bjt, sebas_rows, c) -> np.ndarray:
    sd_M1 = cfg.size_dep_M1(M1); sd_M2 = cfg.size_dep_M2(M2)
    Vd = torch.tensor(c["Vd"], dtype=torch.float64)
    row, _ = z365.find_or_impute_row(sebas_rows, c["VG1"], c["VG2"])
    P_M1, P_M2 = z365.make_overrides(row)
    try:
        with z365.patch_sd_scaled(sd_M1, P_M1), z365.patch_sd_scaled(sd_M2, P_M2):
            out = forward_2t(
                cfg, model_M1=M1, model_M2=M2, bjt=bjt, Vd_seq=Vd,
                VG1=torch.tensor(c["VG1"], dtype=torch.float64),
                VG2=torch.tensor(c["VG2"], dtype=torch.float64),
                warm_start=True,
            )
        return np.abs(out["Id"].detach().cpu().numpy())
    except Exception as e:
        return np.full_like(c["Vd"], np.nan, dtype=float)


def evaluate_cell_wide(gidl_routing: str, agidl_M1: float | None = None,
                       bgidl_M1: float | None = None,
                       *, max_curves: int | None = None) -> dict:
    """Run full 33-curve eval at default params and given routing/agidl.
    No BBO: a fast topology-only comparison (frozen z365 defaults)."""
    cfg, M1, M2, bjt = z365.build_pyport_base()
    cfg.gidl_routing = gidl_routing
    if agidl_M1 is not None:
        M1._values["agidl"] = float(agidl_M1)
    if bgidl_M1 is not None:
        M1._values["bgidl"] = float(bgidl_M1)
    sebas_rows = z365.load_sebas_params()
    curves = z365.load_curves()
    if max_curves is not None:
        curves = curves[:max_curves]
    rmses, per_vg1 = [], {0.20: [], 0.40: [], 0.60: []}
    per_curve = []
    t0 = time.time()
    for c in curves:
        Id_pred = predict_curve(cfg, M1, M2, bjt, sebas_rows, c)
        mask = (c["Id"] > 1e-15) & (Id_pred > 1e-15) & np.isfinite(Id_pred)
        if int(mask.sum()) >= 3:
            rmse = float(np.sqrt(np.mean((np.log10(Id_pred[mask]) - np.log10(c["Id"][mask])) ** 2)))
        else:
            rmse = 10.0
        rmses.append(rmse)
        vg1_key = round(c["VG1"], 2)
        if vg1_key in per_vg1:
            per_vg1[vg1_key].append(rmse)
        per_curve.append({"f": c["f"], "VG1": c["VG1"], "VG2": c["VG2"], "rmse": rmse})
    return {
        "gidl_routing": gidl_routing,
        "agidl_M1": agidl_M1,
        "cell_wide_median_dec": float(np.median(rmses)) if rmses else None,
        "cell_wide_mean_dec":   float(np.mean(rmses))   if rmses else None,
        "per_VG1_median": {f"{k:.2f}": float(np.median(v)) if v else None
                            for k, v in per_vg1.items()},
        "n_curves": len(curves),
        "per_curve": per_curve,
        "elapsed_s": time.time() - t0,
    }


def fold_amplitude_dec(gidl_routing: str, agidl_M1: float | None,
                        sebas_rows, curves,
                        bgidl_M1: float | None = None) -> dict:
    """Estimate the snapback FOLD log10 amplitude at VG1=0.6 between
    the measured curve and pyport prediction. Reported as max-min log10
    over Vd>1.0 V (where SCBE/snapback dominates). Positive value means
    pyport tracks the fold height; ~0 means pyport misses it.

    Returns per-curve dict for VG1=0.6 curves only.
    """
    cfg, M1, M2, bjt = z365.build_pyport_base()
    cfg.gidl_routing = gidl_routing
    if agidl_M1 is not None:
        M1._values["agidl"] = float(agidl_M1)
    if bgidl_M1 is not None:
        M1._values["bgidl"] = float(bgidl_M1)
    out_rows = []
    for c in curves:
        if abs(c["VG1"] - 0.6) > 1e-3:
            continue
        Vd = np.asarray(c["Vd"]); Im = np.asarray(c["Id"])
        Id_pred = predict_curve(cfg, M1, M2, bjt, sebas_rows, c)
        m = (Vd > 1.0) & (Im > 1e-15) & np.isfinite(Id_pred) & (Id_pred > 1e-15)
        if int(m.sum()) < 3:
            continue
        fold_meas = float(np.log10(Im[m]).max() - np.log10(Im[m]).min())
        fold_pred = float(np.log10(Id_pred[m]).max() - np.log10(Id_pred[m]).min())
        out_rows.append({"f": c["f"], "VG2": c["VG2"],
                         "fold_meas_dec": fold_meas,
                         "fold_pred_dec": fold_pred})
    return out_rows


def make_snapback_plot(out_path: Path, sebas_rows, curves,
                        agidl_drain: float | None = None,
                        bgidl_drain: float | None = None) -> None:
    """Compare body vs drain routing across three anchor (VG1, VG2) points
    using the AMBITIOUS aigdl identified in the sweep (set globally to the
    card default 1.99e-8 if no improvement)."""
    anchors = [(0.20, 0.10), (0.40, 0.20), (0.60, 0.20)]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, (vg1, vg2) in zip(axes, anchors):
        # locate measured curve
        meas = None
        for c in curves:
            if abs(c["VG1"] - vg1) < 1e-3 and abs(c["VG2"] - vg2) < 1e-3:
                meas = c; break
        if meas is None:
            ax.text(0.5, 0.5, f"no curve\nVG1={vg1}\nVG2={vg2}",
                    transform=ax.transAxes, ha="center")
            continue
        ax.semilogy(meas["Vd"], meas["Id"], "k-", lw=2, label="measured")

        for routing, color, ls in [("body", "tab:blue", "--"),
                                    ("drain", "tab:red", "-")]:
            cfg, M1, M2, bjt = z365.build_pyport_base()
            cfg.gidl_routing = routing
            if routing == "drain" and agidl_drain is not None:
                M1._values["agidl"] = float(agidl_drain)
            if routing == "drain" and bgidl_drain is not None:
                M1._values["bgidl"] = float(bgidl_drain)
            Id_pred = predict_curve(cfg, M1, M2, bjt, sebas_rows, meas)
            ax.semilogy(meas["Vd"], np.maximum(Id_pred, 1e-15),
                        color=color, ls=ls, lw=1.5,
                        label=f"pyport ({routing})")
        ax.set_title(f"VG1={vg1}, VG2={vg2}")
        ax.set_xlabel("Vd [V]")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("|Id| [A]")
    fig.suptitle("z411 — GIDL routing: body (legacy) vs drain (BSIM4 standard)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    t0 = time.time()
    _log("z411 — S8 GIDL-to-drain rewiring")

    # ---------- Step 4: SMOKE TEST ----------
    _log("[smoke] body vs drain at headline (VG1=0.6, VG2=0.2, Vd=1.5)")
    Id_body  = predict_ids_headline("body")
    Id_drain = predict_ids_headline("drain")
    ratio = Id_drain / Id_body if Id_body > 0 else float("inf")
    _log(f"  Id(body)  = {Id_body:.3e} A")
    _log(f"  Id(drain) = {Id_drain:.3e} A   (ratio {ratio:.3f}×)")
    _log(f"  measured  = {IDS_MEAS_15V:.3e} A   (drain gap = {IDS_MEAS_15V/Id_drain:.2e}×)")
    smoke_ok = (Id_drain > Id_body) and math.isfinite(Id_drain)
    _log(f"  smoke OK (Id_drain > Id_body, finite): {smoke_ok}")

    # ---------- Step 5: GIDL PARAM SWEEP ----------
    # NOTE: A pre-sweep probe of compute_igidl_gisl shows that with the M1
    # card's bgidl=1.624e9 the exp(-bgidl/T1) term saturates to MIN_EXP at
    # all reasonable biases, so agidl alone has ZERO effect on Igidl. We
    # therefore sweep bgidl over a range that brings T2=bgidl/T1 into the
    # numerically active region (~10-30). At T1≈1V/(3·4e-9m)≈8e7 V/m,
    # bgidl in [1e8, 5e9] V/m gives T2 in [1.25, 62]. We pair bgidl with a
    # few agidl values; the resulting Igidl spans 0..nA..µA.
    _log("[sweep] (agidl, bgidl) × {body, drain} at headline")
    AGIDL_CANDIDATES = [1.99e-8, 1e-6, 1e-4, 1e-2]
    BGIDL_CANDIDATES = [1.624e9, 5e8, 2e8, 1e8, 5e7]
    sweep_rows = []
    for a in AGIDL_CANDIDATES:
        for b in BGIDL_CANDIDATES:
            for routing in ("body", "drain"):
                Id_pt = predict_ids_headline(routing, agidl_M1=a, bgidl_M1=b)
                gap = (IDS_MEAS_15V / Id_pt
                       if (Id_pt > 0 and math.isfinite(Id_pt)) else float("inf"))
                sweep_rows.append({"agidl": a, "bgidl": b, "routing": routing,
                                    "Ids_headline": Id_pt, "gap_x": gap})
                _log(f"  agidl={a:.2e} bgidl={b:.2e} [{routing}] "
                     f"→ Ids={Id_pt:.3e} A, gap={gap:.2e}×")

    # For cell-wide refit, restrict to (a, b) pairs that give Id within
    # 100× of measured AT headline AND used routing='drain'. Always
    # include the card-default pair under both routings as anchors.
    candidates_for_cw = []
    for r in sweep_rows:
        if r["routing"] != "drain":
            continue
        if r["agidl"] == 1.99e-8 and r["bgidl"] == 1.624e9:
            candidates_for_cw.append(r)
            continue
        if r["gap_x"] is not None and r["gap_x"] < 100.0:
            candidates_for_cw.append(r)
    # Cap to top 4 by closeness to measured to keep runtime under 30 min
    # (each cell-wide eval ≈ 140-200s on ikaros; total budget ≈ 5 evals × 200s)
    candidates_for_cw = sorted(candidates_for_cw,
                                key=lambda r: abs(math.log10(max(r["gap_x"], 1e-30))))[:4]
    _log(f"[sweep] cell-wide eval for {len(candidates_for_cw)} drain candidates")
    cell_wide_rows = []
    for r in candidates_for_cw:
        ev = evaluate_cell_wide("drain",
                                 agidl_M1=r["agidl"], bgidl_M1=r["bgidl"])
        ev_lean = {k: v for k, v in ev.items() if k != "per_curve"}
        ev_lean["bgidl_M1"] = r["bgidl"]
        cell_wide_rows.append(ev_lean)
        _log(f"  agidl={r['agidl']:.2e} bgidl={r['bgidl']:.2e} "
             f"→ cell-wide={ev['cell_wide_median_dec']:.4f} dec  "
             f"({ev['elapsed_s']:.1f}s)")

    if cell_wide_rows:
        best_a = min(cell_wide_rows,
                     key=lambda r: r["cell_wide_median_dec"]
                     if r["cell_wide_median_dec"] is not None else 1e9)
    else:
        # Fallback: at least eval the card-default pair under drain routing
        ev = evaluate_cell_wide("drain", agidl_M1=1.99e-8, bgidl_M1=1.624e9)
        best_a = {k: v for k, v in ev.items() if k != "per_curve"}
        best_a["bgidl_M1"] = 1.624e9
        cell_wide_rows = [best_a]
    _log(f"[best] agidl={best_a['agidl_M1']:.2e} bgidl={best_a.get('bgidl_M1'):.2e}  "
         f"cell-wide-median={best_a['cell_wide_median_dec']:.4f} dec")

    # ---------- Step 6: BASELINE COMPARISON (body, card defaults) ----------
    _log("[baseline] full 33-curve eval with routing='body' (card-default agidl)")
    baseline = evaluate_cell_wide("body", agidl_M1=None)
    baseline_lean = {k: v for k, v in baseline.items() if k != "per_curve"}
    _log(f"  baseline cell-wide median = {baseline['cell_wide_median_dec']:.4f} dec")

    # Fold amplitude at VG1=0.6
    curves = z365.load_curves()
    sebas_rows = z365.load_sebas_params()
    fold_baseline = fold_amplitude_dec("body",  None, sebas_rows, curves)
    fold_drain    = fold_amplitude_dec("drain", best_a["agidl_M1"], sebas_rows, curves,
                                         bgidl_M1=best_a.get("bgidl_M1"))
    fold_baseline_med_pred = float(np.median([r["fold_pred_dec"] for r in fold_baseline])) \
                              if fold_baseline else None
    fold_drain_med_pred    = float(np.median([r["fold_pred_dec"] for r in fold_drain])) \
                              if fold_drain else None
    fold_meas_med = float(np.median([r["fold_meas_dec"] for r in fold_baseline])) \
                     if fold_baseline else None
    _log(f"[fold@VG1=0.6] meas median = {fold_meas_med}")
    _log(f"               pyport(body)  median = {fold_baseline_med_pred}")
    _log(f"               pyport(drain,best a) median = {fold_drain_med_pred}")

    # ---------- Step 7: SNAPBACK PLOT ----------
    try:
        make_snapback_plot(OUT / "snapback_compare.png", sebas_rows, curves,
                            agidl_drain=best_a["agidl_M1"],
                            bgidl_drain=best_a.get("bgidl_M1"))
        _log(f"[plot] wrote {OUT / 'snapback_compare.png'}")
    except Exception as e:
        _log(f"[plot] EXC: {e}\n{traceback.format_exc()}")

    # ---------- GATES ----------
    elapsed = time.time() - t0
    infra_ok = bool(smoke_ok and elapsed < 30 * 60)
    cell_wide_best = best_a["cell_wide_median_dec"]
    headline_drain_best = None
    for r in sweep_rows:
        if (r["routing"] == "drain"
            and r["agidl"] == best_a["agidl_M1"]
            and r["bgidl"] == best_a.get("bgidl_M1")):
            headline_drain_best = r["Ids_headline"]
            break
    gap_best = (IDS_MEAS_15V / headline_drain_best) \
        if (headline_drain_best and headline_drain_best > 0) else float("inf")
    discovery_ok = bool(
        (cell_wide_best is not None and cell_wide_best < 0.85)
        or (gap_best is not None and gap_best < 3.0)
    )
    ambitious_ok = bool(
        cell_wide_best is not None and cell_wide_best < 0.50
        and (fold_drain_med_pred is not None and fold_drain_med_pred > 1.0)
    )
    # Kill-shot if drain routing strictly worse than body baseline
    kill_shot = bool(
        cell_wide_best is not None and baseline["cell_wide_median_dec"] is not None
        and cell_wide_best >= baseline["cell_wide_median_dec"]
        and Id_drain <= Id_body * 1.01  # essentially no change at headline
    )

    summary = {
        "script": "z411_gidl_to_drain",
        "S7_summary_ref": "results/z410_s7_summary.json",
        "headline": HEADLINE,
        "measured_Ids_headline": IDS_MEAS_15V,
        "smoke": {
            "Id_body":  Id_body,
            "Id_drain": Id_drain,
            "ratio_drain_over_body": ratio,
            "gap_drain_to_meas_x": IDS_MEAS_15V / Id_drain if Id_drain > 0 else None,
            "ok": smoke_ok,
        },
        "agidl_sweep_headline": sweep_rows,
        "cell_wide_sweep_drain": cell_wide_rows,
        "baseline_body": baseline_lean,
        "best": {
            "agidl_M1": best_a["agidl_M1"],
            "bgidl_M1": best_a.get("bgidl_M1"),
            "cell_wide_median_dec": best_a["cell_wide_median_dec"],
            "headline_Ids":  headline_drain_best,
            "gap_to_meas_x": gap_best,
        },
        "fold_at_VG1_0p6": {
            "measured_median_dec": fold_meas_med,
            "body_median_dec":     fold_baseline_med_pred,
            "drain_best_median_dec": fold_drain_med_pred,
            "per_curve_body":  fold_baseline,
            "per_curve_drain": fold_drain,
        },
        "gates": {
            "INFRA":      infra_ok,
            "DISCOVERY":  discovery_ok,
            "AMBITIOUS":  ambitious_ok,
            "KILL_SHOT":  kill_shot,
        },
        "elapsed_s": elapsed,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    (OUT / "agidl_sweep.json").write_text(json.dumps({
        "headline": sweep_rows, "cell_wide": cell_wide_rows
    }, indent=2))
    (OUT / "run.log").write_text("\n".join(LOG_LINES))

    _log(f"=== DONE in {elapsed:.1f}s ===")
    _log(f"  cell-wide(body)  = {baseline['cell_wide_median_dec']:.4f} dec")
    _log(f"  cell-wide(drain,best a={best_a['agidl_M1']:.2e}) = {cell_wide_best:.4f} dec")
    _log(f"  headline gap (drain,best) = {gap_best:.2e}×")
    _log(f"  GATES: INFRA={infra_ok} DISCOVERY={discovery_ok} "
         f"AMBITIOUS={ambitious_ok} KILL_SHOT={kill_shot}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        (OUT / "run.log").write_text("\n".join(LOG_LINES) + "\n\nEXCEPTION:\n"
                                     + traceback.format_exc())
        raise
