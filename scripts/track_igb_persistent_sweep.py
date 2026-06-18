#!/usr/bin/env python3
"""Track persistent gate-body current Igb_persist sweep.

Hypothesis (oracle G1): Adding a PERSISTENT (steady-state) current source
between VG1 and the floating body Vb,

    Igb_persist = g_gate_body · (VG1 - Vb)    [+ INTO body → R_B += Igb]

shifts the model snapback knee from ~1.5V toward the data knee (0.85-1.15V)
at VG1=0.6 without degrading the full-33 K1+ALPHA0 0.665 dec fit.

Background:
  - Newton initial-guess Vb-nudges were ineffective (Newton relaxes away
    from the seed). A persistent term in the residual is required.
  - Mechanistic justification: Cgb-mediated tunneling / forward-biased
    gate-body coupling at high VG1 — additive residual current absent
    in stock BSIM4 (whose Igb is the oxide tunneling component already
    captured in m["Igb"]).

Setup:
  - K1+ALPHA0 card fix locked: K1@VG1=0.6 = 0.53825, ALPHA0 = 7.83756e-4.
  - cfg.g_gate_body ∈ {0, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5} S.
  - For each: measure
      (a) knee_vd at VG1=0.6 × VG2∈{-0.1, 0.0, 0.1, 0.2}  (fwd branch)
      (b) full 33-bias median_dec (fwd+bwd, n=66).

PASS gate (both):
  - Mean (model_knee − data_knee) at VG1=0.6 ≤ 0.2 V
  - Full-33 median_dec ≤ 0.766 dec (≤ 0.1 worse than 0.665 baseline)

Outputs: results/track_igb_persistent/{ablation.json, verdict.md, plot.png}
"""
from __future__ import annotations
import os, sys, json, time, traceback
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "nsram"))
sys.path.insert(0, str(ROOT / "scripts"))

import importlib.util
sp = importlib.util.spec_from_file_location("pillar_I", ROOT / "scripts/pillar_I_C3_jts_tat.py")
pillar = importlib.util.module_from_spec(sp); sp.loader.exec_module(pillar)

OUT = ROOT / "results/track_igb_persistent"
OUT.mkdir(parents=True, exist_ok=True)

# K1+ALPHA0 card fix (locked)
K1_CARD     = 0.53825
ALPHA0_CARD = 7.83756e-4
BASELINE_FULL33_DEC = 0.665   # K1+ALPHA0 combo card+card baseline (g_gb=0)

# Sweep grid
GGB_GRID = [0.0, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5]

# Knee subset
VG1_TARGET = 0.6
VG2_GRID   = [-0.1, 0.0, 0.1, 0.2]

# Pass thresholds
KNEE_PASS_GAP_V    = 0.20
DEC_PASS_THRESHOLD = 0.766


# ── Thermal monitor ───────────────────────────────────────────────
def cpu_temp_c() -> float:
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            return float(f.read().strip()) / 1000.0
    except Exception:
        return float("nan")


def wait_cool(threshold_c: float = 75.0, target_c: float = 50.0, timeout_s: float = 240.0):
    t0 = time.time()
    while True:
        t = cpu_temp_c()
        if not np.isfinite(t) or t <= threshold_c:
            return
        if time.time() - t0 > timeout_s:
            print(f"[igb] thermal timeout (T={t:.1f}°C)", flush=True)
            return
        print(f"[igb] T={t:.1f}°C > {threshold_c}°C — waiting", flush=True)
        time.sleep(5.0)
        if cpu_temp_c() < target_c:
            return


# ── Knee detector (matches track_bf_knee_sweep.py) ───────────────
def knee_vd(Vd: np.ndarray, Id: np.ndarray) -> float:
    Vd = np.asarray(Vd, dtype=np.float64)
    Id = np.abs(np.asarray(Id, dtype=np.float64))
    base_mask = (Vd >= 0.0) & (Vd <= 0.3) & np.isfinite(Id)
    if base_mask.sum() < 2:
        return float("nan")
    base = np.median(Id[base_mask])
    if not np.isfinite(base) or base <= 0:
        base = max(base, 1e-15)
    thresh = 10.0 * base
    post_mask = (Vd > 0.3) & np.isfinite(Id)
    if post_mask.sum() < 2:
        return float("nan")
    for v, c in zip(Vd, Id):
        if v > 0.3 and np.isfinite(c) and c > thresh:
            return float(v)
    return float("nan")


def pick_curves(curves):
    out = []
    for vg2 in VG2_GRID:
        match = None
        for c in curves:
            if abs(c["VG1"] - VG1_TARGET) < 1e-6 and abs(c["VG2"] - vg2) < 1e-6:
                match = c; break
        out.append((vg2, match))
    return out


def simulate_curve(cfg, M1, M2, bjt, c, sebas_rows):
    from nsram.bsim4_port.nsram_cell_2T import forward_2t
    sd_M1 = cfg.size_dep_M1(M1); sd_M2 = cfg.size_dep_M2(M2)
    Vd_np = c["fwd_Vd"]
    row_sebas, _ = pillar.find_or_impute_row(sebas_rows, c["VG1"], c["VG2"])
    P_M1, P_M2 = pillar.make_overrides(row_sebas)
    # apply K1+ALPHA0 locked card values for knee subset
    P_M1["alpha0"] = float(ALPHA0_CARD)
    P_M2["alpha0"] = float(ALPHA0_CARD)
    if abs(c["VG1"] - VG1_TARGET) < 1e-6:
        P_M1["k1"] = float(K1_CARD)
    Vd = torch.tensor(Vd_np, dtype=torch.float64)
    try:
        with pillar.patch_sd_scaled(sd_M1, P_M1), pillar.patch_sd_scaled(sd_M2, P_M2):
            out = forward_2t(cfg, model_M1=M1, model_M2=M2, bjt=bjt, Vd_seq=Vd,
                             VG1=torch.tensor(c["VG1"], dtype=torch.float64),
                             VG2=torch.tensor(c["VG2"], dtype=torch.float64),
                             warm_start=True)
        I_pred = np.abs(out["Id"].detach().cpu().numpy()).astype(np.float64)
        I_pred = np.where(np.isfinite(I_pred), I_pred, 0.0)
    except Exception as e:
        print(f"[igb] FAIL sim {c['f']}: {e}", flush=True)
        I_pred = np.zeros_like(Vd_np)
    return Vd_np, I_pred


def run_one(ggb, sel_curves, curves, sebas_rows):
    print(f"[igb] === g_gate_body = {ggb:.2e} S ===", flush=True)
    wait_cool()
    cfg, M1, M2, bjt = pillar.build_pyport_base()
    cfg.g_gate_body = float(ggb)

    # K1+ALPHA0 monkey patch for full-33
    saved_branch_k1 = pillar.BRANCH_FLAT[VG1_TARGET]["K1"]
    pillar.BRANCH_FLAT[VG1_TARGET]["K1"] = float(K1_CARD)
    orig_make = pillar.make_overrides

    def patched_make(sebas_row):
        P_M1, P_M2 = orig_make(sebas_row)
        if P_M1 is None: P_M1 = {}
        if P_M2 is None: P_M2 = {}
        P_M1["alpha0"] = float(ALPHA0_CARD)
        P_M2["alpha0"] = float(ALPHA0_CARD)
        if sebas_row is not None and abs(sebas_row.get("VG1", float("nan")) - VG1_TARGET) < 1e-6:
            P_M1["k1"] = float(K1_CARD)
        return P_M1, P_M2
    pillar.make_overrides = patched_make

    per_vg2 = []
    full33_summ = None
    try:
        # --- Knee subset (VG1=0.6 × 4 VG2) ---
        for vg2, c in sel_curves:
            if c is None:
                per_vg2.append({"VG2": vg2, "data_knee_V": float("nan"),
                                "model_knee_V": float("nan"),
                                "Vd": [], "Id_data": [], "Id_pred": []})
                continue
            Vd_np, I_pred = simulate_curve(cfg, M1, M2, bjt, c, sebas_rows)
            Id_data = np.abs(c["fwd_Id"])
            data_knee  = knee_vd(c["fwd_Vd"], Id_data)
            model_knee = knee_vd(Vd_np,        I_pred)
            print(f"[igb]   VG2={vg2:+.2f}  data_knee={data_knee:.3f}V  model_knee={model_knee:.3f}V", flush=True)
            per_vg2.append({
                "VG2": float(vg2), "file": c["f"],
                "data_knee_V":  float(data_knee),
                "model_knee_V": float(model_knee),
                "Vd":      Vd_np.tolist(),
                "Id_data": Id_data.tolist(),
                "Id_pred": I_pred.tolist(),
            })

        # --- Full 33-bias fwd+bwd ---
        wait_cool()
        t0 = time.time()
        rows, nan_count = pillar.run_grid(cfg, M1, M2, bjt, curves, sebas_rows,
                                          f"ggb={ggb:.2e}", do_bwd=True)
        dt = time.time() - t0
        full33_summ = pillar.summarize(rows, f"ggb={ggb:.2e}")
        full33_summ["nan_count"] = int(nan_count)
        full33_summ["runtime_s"] = float(dt)
        finite = sum(1 for r in rows if np.isfinite(r["med_dec"]) and r["med_dec"] > 0)
        full33_summ["n_rows"] = len(rows)
        full33_summ["n_finite"] = finite
        full33_summ["convergence_rate"] = finite / max(len(rows), 1)
        print(f"[igb]   full-33 median_dec = {full33_summ['median_dec_all']['median']:.3f} "
              f"(conv={full33_summ['convergence_rate']:.2f}, {dt:.1f}s)", flush=True)
    finally:
        pillar.make_overrides = orig_make
        pillar.BRANCH_FLAT[VG1_TARGET]["K1"] = saved_branch_k1

    data_knees  = np.array([d["data_knee_V"]  for d in per_vg2], dtype=np.float64)
    model_knees = np.array([d["model_knee_V"] for d in per_vg2], dtype=np.float64)
    valid = np.isfinite(data_knees) & np.isfinite(model_knees)
    mean_data  = float(np.nanmean(data_knees))
    mean_model = float(np.nanmean(model_knees))
    gap = float(np.nanmean(model_knees[valid] - data_knees[valid])) if valid.any() else float("nan")

    return {
        "g_gate_body": float(ggb),
        "per_vg2": per_vg2,
        "mean_data_knee_V":  mean_data,
        "mean_model_knee_V": mean_model,
        "knee_gap_V":        gap,
        "full33": full33_summ,
    }


def make_plot(results, sel_curves):
    fig, axes = plt.subplots(1, len(VG2_GRID), figsize=(4.4 * len(VG2_GRID), 4.2), sharey=True)
    if len(VG2_GRID) == 1:
        axes = [axes]
    colors = plt.cm.viridis(np.linspace(0, 0.92, len(GGB_GRID)))
    for ax, (vg2, c) in zip(axes, sel_curves):
        if c is not None:
            ax.semilogy(c["fwd_Vd"], np.clip(np.abs(c["fwd_Id"]), 1e-15, None),
                        "k-", lw=2.2, label="data")
        for ggb, col in zip(GGB_GRID, colors):
            tag = f"ggb={ggb:.0e}"
            r = results.get(tag)
            if r is None: continue
            for d in r["per_vg2"]:
                if abs(d["VG2"] - vg2) < 1e-6 and d.get("Vd"):
                    ax.semilogy(d["Vd"], np.clip(np.array(d["Id_pred"]), 1e-15, None),
                                color=col, lw=1.2, alpha=0.85, label=tag)
                    if np.isfinite(d["model_knee_V"]):
                        ax.axvline(d["model_knee_V"], color=col, lw=0.6, ls=":", alpha=0.5)
                    break
        if c is not None and np.isfinite(knee_vd(c["fwd_Vd"], np.abs(c["fwd_Id"]))):
            ax.axvline(knee_vd(c["fwd_Vd"], np.abs(c["fwd_Id"])), color="k", lw=0.8, ls="--", alpha=0.7)
        ax.set_xlabel("Vd [V]")
        ax.set_title(f"VG1=0.6, VG2={vg2:+.2f}")
        ax.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel("|Id| [A]")
    axes[0].legend(fontsize=7, loc="lower right")
    fig.suptitle("Persistent gate-body coupling sweep — K1+ALPHA0 card fix locked", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "plot.png", dpi=130)
    plt.close(fig)


def main():
    t0 = time.time()
    sebas_rows = pillar.load_sebas_params()
    curves = pillar.load_curves()
    print(f"[igb] loaded {len(curves)} curves, {len(sebas_rows)} sebas rows", flush=True)
    sel_curves = pick_curves(curves)
    n_have = sum(1 for _, c in sel_curves if c is not None)
    print(f"[igb] selected {n_have}/{len(VG2_GRID)} VG1=0.6 curves", flush=True)

    results = {}
    for ggb in GGB_GRID:
        tag = f"ggb={ggb:.0e}"
        try:
            results[tag] = run_one(ggb, sel_curves, curves, sebas_rows)
        except Exception as e:
            traceback.print_exc()
            results[tag] = {"g_gate_body": ggb, "error": str(e)}
        # incremental persist (drop big arrays in saved copy)
        snapshot = {}
        for k, v in results.items():
            if "error" in v:
                snapshot[k] = v; continue
            slim = dict(v)
            slim["per_vg2"] = [{kk: vv for kk, vv in d.items()
                                if kk not in ("Vd", "Id_data", "Id_pred")}
                               for d in v["per_vg2"]]
            snapshot[k] = slim
        with open(OUT / "ablation.json", "w") as f:
            json.dump(snapshot, f, indent=2, default=str)

    # Plot (uses full results in memory)
    try:
        make_plot(results, sel_curves)
    except Exception as e:
        print(f"[igb] plot failed: {e}", flush=True)

    # ── verdict.md ────────────────────────────────────────────────
    lines = []
    lines.append("# Track Igb persistent — gate→body current source sweep\n")
    lines.append(f"Hypothesis: a PERSISTENT current Igb = g_gate_body·(VG1−Vb) added to the")
    lines.append(f"body KCL (R_B residual) shifts the snapback knee left without degrading")
    lines.append(f"the 33-bias K1+ALPHA0 fit (baseline median_dec = {BASELINE_FULL33_DEC}).\n")
    lines.append(f"Locked card values: K1@VG1=0.6 = {K1_CARD}, ALPHA0 = {ALPHA0_CARD:.4e}.")
    lines.append(f"Knee subset: VG1=0.6, VG2 ∈ {VG2_GRID}. Knee = first Vd>0.3V where")
    lines.append(f"|Id| > 10× median(|Id|[Vd∈[0,0.3]]).\n")
    lines.append("## Sweep table\n")
    lines.append("| g_gate_body [S] | mean_model_knee [V] | mean_data_knee [V] | knee_gap (model−data) [V] | full-33 median_dec | Δdec vs 0.665 | conv | knee_pass | dec_pass |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|")

    best_tag = None
    best_score = float("inf")   # composite: prefer smaller gap, then smaller dec
    for ggb in GGB_GRID:
        tag = f"ggb={ggb:.0e}"
        r = results.get(tag)
        if r is None or "error" in r:
            err = r.get("error", "missing")[:40] if r else "missing"
            lines.append(f"| {ggb:.2e} | ERROR | | | | | | | | ({err}) |")
            continue
        gap = r["knee_gap_V"]
        full33 = r["full33"] or {}
        med = full33.get("median_dec_all", {}).get("median", float("nan"))
        conv = full33.get("convergence_rate", float("nan"))
        knee_pass = bool(np.isfinite(gap) and gap <= KNEE_PASS_GAP_V)
        dec_pass  = bool(np.isfinite(med) and med <= DEC_PASS_THRESHOLD)
        lines.append(
            f"| {ggb:.2e} | {r['mean_model_knee_V']:.3f} | {r['mean_data_knee_V']:.3f} | "
            f"{gap:+.3f} | {med:.3f} | {med - BASELINE_FULL33_DEC:+.3f} | {conv:.2f} | "
            f"{'YES' if knee_pass else 'NO'} | {'YES' if dec_pass else 'NO'} |"
        )
        # composite: knee_gap (clipped at 0) + dec excess (clipped at 0)
        score = max(gap, 0.0) + max(med - BASELINE_FULL33_DEC, 0.0)
        if knee_pass and dec_pass and score < best_score:
            best_score = score; best_tag = tag

    # Per-VG2 knee table for each ggb
    lines.append("\n## Per-VG2 model knee positions [V]\n")
    header = "| g_gate_body [S] | " + " | ".join(f"VG2={v:+.2f}" for v in VG2_GRID) + " |"
    sep    = "|---:|" + "|".join(["---:"] * len(VG2_GRID)) + "|"
    lines.append(header); lines.append(sep)
    for ggb in GGB_GRID:
        tag = f"ggb={ggb:.0e}"
        r = results.get(tag)
        if r is None or "error" in r:
            lines.append(f"| {ggb:.2e} | " + " | ".join(["ERR"] * len(VG2_GRID)) + " |")
            continue
        cells = []
        for vg2 in VG2_GRID:
            entry = next((d for d in r["per_vg2"] if abs(d["VG2"] - vg2) < 1e-6), None)
            cells.append(f"{entry['model_knee_V']:.3f}" if entry and np.isfinite(entry["model_knee_V"]) else "nan")
        lines.append(f"| {ggb:.2e} | " + " | ".join(cells) + " |")

    # data knees row (reference)
    base_r = next((results[f"ggb={g:.0e}"] for g in GGB_GRID
                   if f"ggb={g:.0e}" in results and "error" not in results[f"ggb={g:.0e}"]), None)
    if base_r is not None:
        drow = []
        for vg2 in VG2_GRID:
            entry = next((d for d in base_r["per_vg2"] if abs(d["VG2"] - vg2) < 1e-6), None)
            drow.append(f"{entry['data_knee_V']:.3f}" if entry and np.isfinite(entry["data_knee_V"]) else "nan")
        lines.append("\n### Data knees (reference)\n")
        lines.append("| " + " | ".join(f"VG2={v:+.2f}" for v in VG2_GRID) + " |")
        lines.append("|" + "|".join(["---:"] * len(VG2_GRID)) + "|")
        lines.append("| " + " | ".join(drow) + " |")

    # Verdict block
    lines.append("\n## Verdict\n")
    lines.append(f"- PASS gate: knee_gap (model−data, mean over 4 VG2 at VG1=0.6) ≤ {KNEE_PASS_GAP_V} V")
    lines.append(f"  AND full-33 median_dec ≤ {DEC_PASS_THRESHOLD} (≤ 0.1 worse than baseline 0.665).")

    if best_tag is not None:
        r = results[best_tag]
        med = r["full33"]["median_dec_all"]["median"]
        lines.append(f"- **PASS** — best condition: {best_tag}")
        lines.append(f"  - g_gate_body = {r['g_gate_body']:.2e} S")
        lines.append(f"  - mean knee_gap = {r['knee_gap_V']:+.3f} V "
                     f"(model {r['mean_model_knee_V']:.3f} vs data {r['mean_data_knee_V']:.3f})")
        lines.append(f"  - full-33 median_dec = {med:.3f}  (Δ vs baseline = {med-BASELINE_FULL33_DEC:+.3f})")
        lines.append(f"\nMechanistic framing: g_gate_body is a NEW physics knob (no card backing).")
        lines.append(f"It models a persistent Cgb-mediated gate→body charge pump / forward-biased")
        lines.append(f"gate-body coupling that injects current into the floating body in steady state,")
        lines.append(f"distinct from BSIM4's oxide-tunneling Igb (already in m['Igb']).")
    else:
        # find best by gap regardless of pass
        best_gap = float("inf"); bg_tag = None
        for ggb in GGB_GRID:
            tag = f"ggb={ggb:.0e}"
            r = results.get(tag)
            if r is None or "error" in r: continue
            gap = r["knee_gap_V"]
            if np.isfinite(gap) and gap < best_gap:
                best_gap = gap; bg_tag = tag
        # best by dec
        best_dec = float("inf"); bd_tag = None
        for ggb in GGB_GRID:
            tag = f"ggb={ggb:.0e}"
            r = results.get(tag)
            if r is None or "error" in r: continue
            full33 = r["full33"] or {}
            m = full33.get("median_dec_all", {}).get("median", float("nan"))
            if np.isfinite(m) and m < best_dec:
                best_dec = m; bd_tag = tag
        lines.append(f"- **FAIL** — no condition satisfies BOTH gates.")
        if bg_tag is not None:
            r = results[bg_tag]
            lines.append(f"  - Smallest knee_gap: {bg_tag} → gap={r['knee_gap_V']:+.3f}V "
                         f"(target ≤{KNEE_PASS_GAP_V}V), full-33 dec={r['full33']['median_dec_all']['median']:.3f}")
        if bd_tag is not None:
            r = results[bd_tag]
            lines.append(f"  - Smallest full-33 dec: {bd_tag} → {r['full33']['median_dec_all']['median']:.3f} "
                         f"(target ≤{DEC_PASS_THRESHOLD}), knee_gap={r['knee_gap_V']:+.3f}V")
        # diagnose
        lines.append("\n### Failure diagnosis\n")
        g_zero = results.get(f"ggb={0.0:.0e}")
        if g_zero and "error" not in g_zero:
            gap0 = g_zero["knee_gap_V"]
            lines.append(f"- At g_gate_body=0 (control): knee_gap = {gap0:+.3f}V (this is the inherited model-vs-data gap).")
            # check if knee moves with ggb at all
            knees_by_g = []
            for ggb in GGB_GRID:
                tag = f"ggb={ggb:.0e}"
                r = results.get(tag)
                if r and "error" not in r and np.isfinite(r["mean_model_knee_V"]):
                    knees_by_g.append((ggb, r["mean_model_knee_V"]))
            if len(knees_by_g) >= 2:
                spread = max(k for _,k in knees_by_g) - min(k for _,k in knees_by_g)
                lines.append(f"- Model knee range across sweep: {spread:.3f}V "
                             f"(min={min(k for _,k in knees_by_g):.3f}, max={max(k for _,k in knees_by_g):.3f}).")
                if spread < 0.05:
                    lines.append(f"- Knee essentially INSENSITIVE to g_gate_body — persistent injection")
                    lines.append(f"  does not propagate into Vb enough to advance the snapback trigger,")
                    lines.append(f"  likely because the body-pdiode / well-diode at Vb clamps Vb near 0V")
                    lines.append(f"  for the range of g_gb tested (currents absorbed by parasitic shunts).")
            # check dec degradation
            decs = [(ggb, results[f"ggb={ggb:.0e}"]["full33"]["median_dec_all"]["median"])
                    for ggb in GGB_GRID
                    if f"ggb={ggb:.0e}" in results and "error" not in results[f"ggb={ggb:.0e}"]]
            if decs:
                lines.append(f"- Full-33 dec range: "
                             f"[{min(d for _,d in decs):.3f}, {max(d for _,d in decs):.3f}].")

    lines.append("\n## Provenance")
    lines.append("- Cfg knob: `NSRAMCell2TConfig.g_gate_body` (added 2026-05-20, G1)")
    lines.append("- KCL term: `R_B += g_gate_body·(VG1−Vb)` in `nsram_cell_2T.py::_residuals`")
    lines.append("- Locked: K1@VG1=0.6 = 0.53825 (BSIM card), ALPHA0 = 7.83756e-4 (Mario LALPHA0_FIX)")
    lines.append("- Baseline reference: `track_combo_k1_alpha0/verdict.md` (full-33 median_dec = 0.665)")
    lines.append("- Knee defn matches `track_bf_knee_sweep.py`")
    lines.append(f"- Total runtime: {time.time() - t0:.1f}s")

    (OUT / "verdict.md").write_text("\n".join(lines) + "\n")
    print(f"[igb] wrote {OUT/'verdict.md'}", flush=True)
    print(f"[igb] wrote {OUT/'ablation.json'}", flush=True)
    print(f"[igb] DONE in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
