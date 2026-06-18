#!/usr/bin/env python3
"""Track Bf knee sweep — does raising NPN forward β shift snapback knee left?

Hypothesis (Grok #3): With K1+ALPHA0 card fix applied, the residual snapback-knee
gap (model fires at Vd≈1.5V, data fires at Vd≈0.75-1.15V) is set by the
parasitic NPN's forward gain. Higher Bf → less floating-body Vb needed for the
same collector current → snapback knee fires at lower Vd.

Plan:
  - Apply K1@VG1=0.6 = 0.53825 + ALPHA0 = 7.83756e-4 (the established combo).
  - Sweep bjt.Bf ∈ {100, 200, 300, 500, 1000}.
  - For each Bf, simulate the four VG1=0.6 forward sweeps with
    VG2 ∈ {-0.1, 0.0, 0.1, 0.2}, locate knee Vd in BOTH model and data.

Knee definition (per task spec):
  baseline = median(Id[Vd ∈ [0, 0.3]])
  knee_vd  = first Vd > 0.3 where Id > 10 × baseline (NaN if never crossed).

Outputs (results/track_bf_knee_sweep/):
  ablation.json, verdict.md, plot.png
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

OUT = ROOT / "results/track_bf_knee_sweep"
OUT.mkdir(parents=True, exist_ok=True)

# K1+ALPHA0 card fix values
K1_CARD     = 0.53825
ALPHA0_CARD = 7.83756e-4

# Bf grid (task spec)
BF_GRID = [100, 200, 300, 500, 1000]

# VG1=0.6 slice, VG2 grid (task spec)
VG1_TARGET = 0.6
VG2_GRID = [-0.1, 0.0, 0.1, 0.2]


# ── Thermal monitor ───────────────────────────────────────────────
def cpu_temp_c() -> float:
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            return float(f.read().strip()) / 1000.0
    except Exception:
        return float("nan")


def wait_cool(threshold_c: float = 75.0, target_c: float = 50.0, timeout_s: float = 180.0):
    t0 = time.time()
    while True:
        t = cpu_temp_c()
        if not np.isfinite(t) or t <= threshold_c:
            return
        if time.time() - t0 > timeout_s:
            print(f"[bf-knee] thermal timeout (T={t:.1f}°C)", flush=True)
            return
        print(f"[bf-knee] T={t:.1f}°C > {threshold_c}°C — waiting for cool-down to {target_c}°C", flush=True)
        time.sleep(5.0)
        if cpu_temp_c() < target_c:
            return


# ── Knee detector ────────────────────────────────────────────────
def knee_vd(Vd: np.ndarray, Id: np.ndarray) -> float:
    """Vd at which Id first exceeds 10× median(Id[Vd in [0, 0.3]]); NaN if never."""
    Vd = np.asarray(Vd, dtype=np.float64)
    Id = np.abs(np.asarray(Id, dtype=np.float64))
    base_mask = (Vd >= 0.0) & (Vd <= 0.3) & np.isfinite(Id)
    if base_mask.sum() < 2:
        return float("nan")
    base = np.median(Id[base_mask])
    if not np.isfinite(base) or base <= 0:
        # fall back to a tiny floor so knee still detectable in noisy log data
        base = max(base, 1e-15)
    thresh = 10.0 * base
    post_mask = (Vd > 0.3) & np.isfinite(Id)
    if post_mask.sum() < 2:
        return float("nan")
    # find first index in original order where Id > thresh and Vd > 0.3
    for i, (v, c) in enumerate(zip(Vd, Id)):
        if v > 0.3 and np.isfinite(c) and c > thresh:
            return float(v)
    return float("nan")


# ── Simulation ───────────────────────────────────────────────────
def simulate_curve(cfg, M1, M2, bjt, c, sebas_rows):
    """Run forward sim on the fwd branch of a curve dict; return (Vd, Id_pred)."""
    from nsram.bsim4_port.nsram_cell_2T import forward_2t
    sd_M1 = cfg.size_dep_M1(M1); sd_M2 = cfg.size_dep_M2(M2)
    Vd_np = c["fwd_Vd"]
    row_sebas, _ = pillar.find_or_impute_row(sebas_rows, c["VG1"], c["VG2"])
    P_M1, P_M2 = pillar.make_overrides(row_sebas)
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
        print(f"[bf-knee] FAIL sim {c['f']}: {e}", flush=True)
        I_pred = np.zeros_like(Vd_np)
    return Vd_np, I_pred


def pick_curves(curves):
    """Return list of curves matching VG1=0.6, VG2 in VG2_GRID, in VG2 order."""
    out = []
    for vg2 in VG2_GRID:
        match = None
        for c in curves:
            if abs(c["VG1"] - VG1_TARGET) < 1e-6 and abs(c["VG2"] - vg2) < 1e-6:
                match = c; break
        if match is None:
            print(f"[bf-knee] WARNING no curve for VG1=0.6 VG2={vg2:+.2f}", flush=True)
        out.append((vg2, match))
    return out


def run_one_bf(Bf, sel_curves, sebas_rows):
    print(f"[bf-knee] === Bf = {Bf} ===", flush=True)
    wait_cool()
    cfg, M1, M2, bjt = pillar.build_pyport_base()
    bjt.Bf = float(Bf)

    # K1+ALPHA0 monkey patch
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
    try:
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
            print(f"[bf-knee]   VG2={vg2:+.2f}  data_knee={data_knee:.3f}V  model_knee={model_knee:.3f}V", flush=True)
            per_vg2.append({
                "VG2": float(vg2),
                "file": c["f"],
                "data_knee_V":  float(data_knee),
                "model_knee_V": float(model_knee),
                "Vd":      Vd_np.tolist(),
                "Id_data": Id_data.tolist(),
                "Id_pred": I_pred.tolist(),
            })
    finally:
        pillar.make_overrides = orig_make
        pillar.BRANCH_FLAT[VG1_TARGET]["K1"] = saved_branch_k1

    data_knees  = np.array([d["data_knee_V"]  for d in per_vg2], dtype=np.float64)
    model_knees = np.array([d["model_knee_V"] for d in per_vg2], dtype=np.float64)
    valid = np.isfinite(data_knees) & np.isfinite(model_knees)
    mean_data  = float(np.nanmean(data_knees))
    mean_model = float(np.nanmean(model_knees))
    shift_left = float(np.nanmean(model_knees[valid]) - np.nanmean(data_knees[valid])) if valid.any() else float("nan")
    target_met = bool(np.isfinite(mean_model) and np.isfinite(mean_data) and (mean_model <= mean_data + 0.2))
    return {
        "Bf": float(Bf),
        "per_vg2": per_vg2,
        "mean_data_knee_V":  mean_data,
        "mean_model_knee_V": mean_model,
        "model_minus_data_V": float(mean_model - mean_data) if np.isfinite(mean_data) else float("nan"),
        "shift_left_vs_baseline_V": None,  # filled in main after baseline available
        "is_target_met": target_met,
    }


# ── Plotting ────────────────────────────────────────────────────
def make_plot(results, sel_curves):
    fig, axes = plt.subplots(1, len(VG2_GRID), figsize=(4.4 * len(VG2_GRID), 4.2), sharey=True)
    if len(VG2_GRID) == 1:
        axes = [axes]
    colors = plt.cm.viridis(np.linspace(0, 0.92, len(BF_GRID)))
    for ax, (vg2, c) in zip(axes, sel_curves):
        if c is not None:
            ax.semilogy(c["fwd_Vd"], np.clip(np.abs(c["fwd_Id"]), 1e-15, None),
                        "k-", lw=2.2, label="data")
        for bf, col in zip(BF_GRID, colors):
            tag = f"Bf={bf}"
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
    fig.suptitle("Bf knee sweep — K1+ALPHA0 card fix applied", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "plot.png", dpi=130)
    plt.close(fig)


# ── Main ────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    sebas_rows = pillar.load_sebas_params()
    curves = pillar.load_curves()
    print(f"[bf-knee] loaded {len(curves)} curves, {len(sebas_rows)} sebas rows", flush=True)
    sel_curves = pick_curves(curves)
    n_have = sum(1 for _, c in sel_curves if c is not None)
    print(f"[bf-knee] selected {n_have}/{len(VG2_GRID)} VG1=0.6 curves", flush=True)

    results = {}
    for bf in BF_GRID:
        tag = f"Bf={bf}"
        try:
            results[tag] = run_one_bf(bf, sel_curves, sebas_rows)
        except Exception as e:
            traceback.print_exc()
            results[tag] = {"Bf": bf, "error": str(e)}
        with open(OUT / "ablation.json", "w") as f:
            json.dump(results, f, indent=2, default=str)

    # shift_left = (mean_model_knee at Bf) - (mean_model_knee at Bf=100)
    base_tag = f"Bf={BF_GRID[0]}"
    base_model = results.get(base_tag, {}).get("mean_model_knee_V", float("nan"))
    for tag, r in results.items():
        if "error" in r: continue
        mm = r.get("mean_model_knee_V", float("nan"))
        r["shift_left_vs_baseline_V"] = float(base_model - mm) if (np.isfinite(mm) and np.isfinite(base_model)) else float("nan")

    with open(OUT / "ablation.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Plot
    try:
        make_plot(results, sel_curves)
    except Exception as e:
        print(f"[bf-knee] plot failed: {e}", flush=True)

    # verdict.md
    lines = []
    lines.append("# Track Bf knee sweep — does raising NPN Bf shift the snapback knee left?\n")
    lines.append(f"Applied combo fix: K1@VG1=0.6 = {K1_CARD}, ALPHA0 = {ALPHA0_CARD:.4e}.")
    lines.append(f"VG1 target = {VG1_TARGET}; VG2 set = {VG2_GRID}.")
    lines.append(f"Knee defined as first Vd > 0.3V where |Id| > 10× median(|Id|[Vd∈[0,0.3]]).\n")

    lines.append("## Per-VG2 knee positions (Volts)\n")
    header = "| Bf | " + " | ".join(f"VG2={v:+.2f}" for v in VG2_GRID) + " | mean_model_knee | mean_data_knee | model−data | shift_left_vs_Bf=100 | target_met |"
    sep    = "|---:|" + "|".join(["---:"] * len(VG2_GRID)) + "|---:|---:|---:|---:|:---:|"
    lines.append(header)
    lines.append(sep)
    for bf in BF_GRID:
        tag = f"Bf={bf}"
        r = results[tag]
        if "error" in r:
            lines.append(f"| {bf} | ERROR: {r['error'][:50]} |")
            continue
        cells = []
        for vg2 in VG2_GRID:
            entry = next((d for d in r["per_vg2"] if abs(d["VG2"] - vg2) < 1e-6), None)
            if entry is None or not np.isfinite(entry["model_knee_V"]):
                cells.append("nan")
            else:
                cells.append(f"{entry['model_knee_V']:.3f}")
        cells_str = " | ".join(cells)
        lines.append(
            f"| {bf} | {cells_str} | {r['mean_model_knee_V']:.3f} | "
            f"{r['mean_data_knee_V']:.3f} | {r['model_minus_data_V']:+.3f} | "
            f"{r['shift_left_vs_baseline_V']:+.3f} | "
            f"{'YES' if r['is_target_met'] else 'NO'} |"
        )

    lines.append("\n## Per-VG2 data knees (reference)\n")
    base_r = results[f"Bf={BF_GRID[0]}"]
    data_row = []
    for vg2 in VG2_GRID:
        entry = next((d for d in base_r["per_vg2"] if abs(d["VG2"] - vg2) < 1e-6), None)
        if entry is None or not np.isfinite(entry["data_knee_V"]):
            data_row.append("nan")
        else:
            data_row.append(f"{entry['data_knee_V']:.3f}")
    lines.append("| " + " | ".join(f"VG2={v:+.2f}" for v in VG2_GRID) + " |")
    lines.append("|" + "|".join(["---:"] * len(VG2_GRID)) + "|")
    lines.append("| " + " | ".join(data_row) + " |")

    # Pick best Bf (minimal mean model−data gap)
    best_bf, best_gap = None, float("inf")
    for bf in BF_GRID:
        r = results[f"Bf={bf}"]
        if "error" in r: continue
        gap = r["model_minus_data_V"]
        if np.isfinite(gap) and gap < best_gap:
            best_gap = gap; best_bf = bf
    lines.append("\n## Verdict\n")
    if best_bf is not None:
        r = results[f"Bf={best_bf}"]
        lines.append(f"- **Best Bf = {best_bf}** → mean_model_knee = {r['mean_model_knee_V']:.3f}V vs data {r['mean_data_knee_V']:.3f}V (gap = {best_gap:+.3f}V).")
        lines.append(f"- Shift-left vs Bf={BF_GRID[0]}: {r['shift_left_vs_baseline_V']:+.3f}V.")
        if r["is_target_met"]:
            lines.append("- **Target met** (model knee ≤ data knee + 0.2V).")
        else:
            lines.append(f"- Target NOT met (need model ≤ data+0.2 V; gap is {best_gap:+.3f}V).")
        # null-result diagnostic
        knees = [results[f"Bf={bf}"]["mean_model_knee_V"] for bf in BF_GRID if "error" not in results[f"Bf={bf}"]]
        knees = [k for k in knees if np.isfinite(k)]
        if len(knees) >= 2 and (max(knees) - min(knees)) < 0.02:
            lines.append(f"- **NULL RESULT**: model knee varies by < 20 mV across Bf grid (range = {min(knees):.3f}–{max(knees):.3f}V). Bf alone is not the lever for the snapback knee position.")
    else:
        lines.append("- All Bf cells failed.")

    lines.append("\n## Provenance")
    lines.append(f"- baseline: `pillar_I_C3_jts_tat.build_pyport_base()` (GummelPoonNPN.from_sebas_card())")
    lines.append(f"- monkey-patch: BRANCH_FLAT[0.6]['K1']={K1_CARD}; make_overrides forces alpha0={ALPHA0_CARD:.4e} on M1+M2 and k1 at VG1≈0.6")
    lines.append(f"- knee_vd: first Vd>0.3 where |Id|>10× median(|Id|[Vd∈[0,0.3]])")
    lines.append(f"- runtime: {time.time() - t0:.1f}s")

    (OUT / "verdict.md").write_text("\n".join(lines) + "\n")
    print(f"[bf-knee] wrote {OUT / 'verdict.md'}", flush=True)
    print(f"[bf-knee] wrote {OUT / 'ablation.json'}", flush=True)


if __name__ == "__main__":
    main()
