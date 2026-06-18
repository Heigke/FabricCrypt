#!/usr/bin/env python3
"""Track VG1→Vb capacitive coupling offset sweep — locked K1+ALPHA0 card fix.

Hypothesis (oracle 3-way unanimous, 2026-05-20):
  VG1 capacitively couples to the floating P-body via Cgb, producing a
  standing Vb offset ~0.3-0.5 V at VG1=0.6 BEFORE any drain voltage. This
  pre-charge brings V_BE close to the parasitic NPN turn-on (~0.7 V), so the
  additional impact-ionization current required for snapback is much
  smaller → knee fires at lower Vd (data: 0.85-1.15 V; model w/o coupling
  at the K1+ALPHA0 card baseline: ~1.5 V).

What this script does:
  Locks the K1+ALPHA0 card fix (K1@VG1=0.6 = 0.53825, alpha0 = 7.83756e-4)
  and sweeps `cfg.vb_gate_coupling ∈ {0.0, 0.2, 0.3, 0.5, 0.7}`. Per
  condition, measures:
    (a) Snapback knee Vd at VG1=0.6 × VG2 ∈ {-0.1, 0.0, 0.1, 0.2} on the
        forward branch.
    (b) Full 33-bias median_dec (fwd+bwd) so the knee-shift cost on the
        rest of the fit is visible.

PASS criterion:
  - model knee_vd ≤ data knee_vd + 0.2 V at VG1=0.6 (averaged across 4 VG2)
  - AND full-33 median_dec does NOT worsen by more than 0.1 dec relative
    to the locked K1+ALPHA0 baseline (= 0.665).

NO-CHEAT framing:
  vb_gate_coupling is a NEW PHYSICS KNOB with no card value. Any positive
  result means "knee position can be explained IF Cgb capacitive coupling
  exists with this magnitude" — NOT a card-derived fix.

Outputs:
  results/track_vg1_vb_offset_sweep/{ablation.json, verdict.md, plot.png, run.log}
"""
from __future__ import annotations
import os, sys, json, time, traceback
from pathlib import Path
import numpy as np
import torch

ROOT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "nsram"))
sys.path.insert(0, str(ROOT / "scripts"))

import importlib.util
sp = importlib.util.spec_from_file_location("pillar_I", ROOT / "scripts/pillar_I_C3_jts_tat.py")
pillar = importlib.util.module_from_spec(sp); sp.loader.exec_module(pillar)

from nsram.bsim4_port.nsram_cell_2T import forward_2t

OUT = ROOT / "results/track_vg1_vb_offset_sweep"
OUT.mkdir(parents=True, exist_ok=True)

# ── Locked K1+ALPHA0 card-fix baseline ────────────────────────────────
K1_CARD     = 0.53825      # BSIM card K1 at VG1=0.6
ALPHA0_CARD = 7.83756e-4   # Mario LALPHA0_FIX card

# ── Sweep grid ────────────────────────────────────────────────────────
VB_GATE_GRID = [0.0, 0.2, 0.3, 0.5, 0.7]
VG1_TARGET = 0.6
VG2_KNEE_LIST = [-0.1, 0.0, 0.1, 0.2]

# Reference: K1+ALPHA0 card baseline median_dec from track_combo_k1_alpha0
LOCKED_BASELINE_MEDIAN_DEC = 0.665

# Thermal monitor
THERMAL_PATH = Path("/sys/class/thermal/thermal_zone0/temp")


def cpu_temp_c():
    try:
        return float(THERMAL_PATH.read_text().strip()) / 1000.0
    except Exception:
        return float("nan")


def wait_cool(thresh_hi=75.0, thresh_lo=50.0, timeout=180.0):
    t0 = time.time()
    t = cpu_temp_c()
    if not np.isfinite(t) or t < thresh_hi:
        return
    print(f"[thermal] APU {t:.1f}°C > {thresh_hi}°C — pausing until ≤{thresh_lo}°C", flush=True)
    while True:
        t = cpu_temp_c()
        if not np.isfinite(t) or t <= thresh_lo:
            print(f"[thermal] APU {t:.1f}°C — resume", flush=True)
            return
        if time.time() - t0 > timeout:
            print(f"[thermal] timeout {timeout}s reached, resuming at {t:.1f}°C", flush=True)
            return
        time.sleep(2.0)


# ── Knee detection ────────────────────────────────────────────────────
def detect_knee(Vd, Id, frac=0.10, vmin=0.3):
    """Return Vd at which |Id| first exceeds `frac` × peak |Id|, restricted to Vd≥vmin.
    Returns NaN if curve too flat / peak unreliable.
    """
    Vd = np.asarray(Vd, dtype=np.float64)
    Id = np.abs(np.asarray(Id, dtype=np.float64))
    if Vd.size < 4 or not np.all(np.isfinite(Id)):
        return float("nan")
    m = Vd >= vmin
    if m.sum() < 3:
        return float("nan")
    Ipeak = float(np.max(Id[m]))
    if Ipeak < 1e-10:   # essentially no snapback
        return float("nan")
    thr = frac * Ipeak
    idx = np.where(m & (Id >= thr))[0]
    if idx.size == 0:
        return float("nan")
    return float(Vd[idx[0]])


def patch_make_overrides_card_fix():
    """Lock K1=card at VG1≈0.6 and ALPHA0=card everywhere. Mirrors
    track_combo_k1_alpha0's run_one() best-of-grid patch.
    """
    saved_branch_k1 = pillar.BRANCH_FLAT[0.6]["K1"]
    pillar.BRANCH_FLAT[0.6]["K1"] = float(K1_CARD)
    orig_make = pillar.make_overrides

    def patched_make(sebas_row):
        P_M1, P_M2 = orig_make(sebas_row)
        if P_M1 is None: P_M1 = {}
        if P_M2 is None: P_M2 = {}
        P_M1["alpha0"] = float(ALPHA0_CARD)
        P_M2["alpha0"] = float(ALPHA0_CARD)
        if sebas_row is not None and abs(sebas_row.get("VG1", float("nan")) - 0.6) < 1e-6:
            P_M1["k1"] = float(K1_CARD)
        return P_M1, P_M2

    pillar.make_overrides = patched_make
    return saved_branch_k1, orig_make


def unpatch(saved_branch_k1, orig_make):
    pillar.make_overrides = orig_make
    pillar.BRANCH_FLAT[0.6]["K1"] = saved_branch_k1


# ── Knee measurement at VG1=0.6 × 4 VG2 ──────────────────────────────
def measure_knees(cfg, M1, M2, bjt, curves, sebas_rows):
    """For each of the 4 target (VG1=0.6, VG2) curves, run forward_2t on the
    forward branch and return knee_vd_data, knee_vd_model.
    """
    sd_M1 = cfg.size_dep_M1(M1); sd_M2 = cfg.size_dep_M2(M2)
    out = {}
    for vg2_target in VG2_KNEE_LIST:
        match = None
        for c in curves:
            if abs(c["VG1"] - VG1_TARGET) < 1e-6 and abs(c["VG2"] - vg2_target) < 1e-6:
                match = c
                break
        if match is None:
            out[vg2_target] = {"knee_data": float("nan"), "knee_model": float("nan"),
                               "note": "no curve"}
            continue

        Vd_np = match["fwd_Vd"]; Id_np = match["fwd_Id"]
        knee_data = detect_knee(Vd_np, Id_np)

        row_sebas, _ = pillar.find_or_impute_row(sebas_rows, match["VG1"], match["VG2"])
        P_M1, P_M2 = pillar.make_overrides(row_sebas)
        Vd_t = torch.tensor(Vd_np, dtype=torch.float64)
        try:
            with pillar.patch_sd_scaled(sd_M1, P_M1), pillar.patch_sd_scaled(sd_M2, P_M2):
                fout = forward_2t(cfg, model_M1=M1, model_M2=M2, bjt=bjt,
                                  Vd_seq=Vd_t,
                                  VG1=torch.tensor(match["VG1"], dtype=torch.float64),
                                  VG2=torch.tensor(match["VG2"], dtype=torch.float64),
                                  warm_start=True)
            I_pred = np.abs(fout["Id"].detach().cpu().numpy()).astype(np.float64)
            knee_model = detect_knee(Vd_np, I_pred)
            # Capture Vb cold-start for diagnostics
            Vb_seq = fout["Vb"].detach().cpu().numpy()
            vb_at_low_vd = float(Vb_seq[0]) if Vb_seq.size else float("nan")
        except Exception as e:
            print(f"[knee] FAIL VG2={vg2_target}: {e}", flush=True)
            knee_model = float("nan")
            vb_at_low_vd = float("nan")

        out[vg2_target] = {
            "knee_data": knee_data,
            "knee_model": knee_model,
            "vb_at_first_vd": vb_at_low_vd,
            "Imeas_peak": float(np.max(np.abs(Id_np))) if Id_np.size else float("nan"),
        }
    return out


# ── One condition: vb_gate_coupling value ─────────────────────────────
def run_one(vb_coup, curves, sebas_rows):
    label = f"vb_gate_coupling={vb_coup:.2f}"
    print(f"[vbcoup] === {label} ===  APU={cpu_temp_c():.1f}°C", flush=True)
    cfg, M1, M2, bjt = pillar.build_pyport_base()
    cfg.vb_gate_coupling = float(vb_coup)

    saved_branch_k1, orig_make = patch_make_overrides_card_fix()
    try:
        # Knees first (cheap, ~4 curves)
        knees = measure_knees(cfg, M1, M2, bjt, curves, sebas_rows)
        wait_cool()
        # Full 33-bias fwd+bwd
        t0 = time.time()
        rows, nan_count = pillar.run_grid(cfg, M1, M2, bjt, curves, sebas_rows, label, do_bwd=True)
        dt = time.time() - t0
    finally:
        unpatch(saved_branch_k1, orig_make)

    summ = pillar.summarize(rows, label)
    summ["vb_gate_coupling"] = float(vb_coup)
    summ["nan_count"] = int(nan_count)
    summ["runtime_s"] = float(dt)
    summ["knees_at_VG1=0.6"] = knees

    # Aggregate knee metrics across the 4 VG2
    model_knees = np.array([k["knee_model"] for k in knees.values()
                            if np.isfinite(k["knee_model"])])
    data_knees = np.array([k["knee_data"] for k in knees.values()
                           if np.isfinite(k["knee_data"])])
    shifts = []
    for vg2 in VG2_KNEE_LIST:
        km = knees[vg2]["knee_model"]; kd = knees[vg2]["knee_data"]
        if np.isfinite(km) and np.isfinite(kd):
            shifts.append(km - kd)
    shifts = np.array(shifts)
    summ["knee_model_mean"] = float(np.mean(model_knees)) if model_knees.size else float("nan")
    summ["knee_data_mean"]  = float(np.mean(data_knees))  if data_knees.size  else float("nan")
    summ["knee_model_minus_data_mean"] = float(np.mean(shifts)) if shifts.size else float("nan")
    summ["knee_model_minus_data_max"]  = float(np.max(shifts))  if shifts.size else float("nan")
    return summ


def main():
    sebas_rows = pillar.load_sebas_params()
    curves = pillar.load_curves()
    print(f"[vbcoup] loaded {len(curves)} curves, {len(sebas_rows)} sebas rows", flush=True)
    print(f"[vbcoup] grid: vb_gate_coupling ∈ {VB_GATE_GRID}", flush=True)
    print(f"[vbcoup] knee targets: VG1={VG1_TARGET} × VG2 ∈ {VG2_KNEE_LIST}", flush=True)

    results = {}
    for vb in VB_GATE_GRID:
        tag = f"vb_coup={vb:.2f}"
        try:
            results[tag] = run_one(vb, curves, sebas_rows)
        except Exception as e:
            print(f"[vbcoup] FAIL {tag}: {e}", flush=True)
            traceback.print_exc()
            results[tag] = {"label": tag, "vb_gate_coupling": vb, "error": str(e)}
        with open(OUT / "ablation.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        wait_cool()

    # ── Pick best (PASS-aware) ────────────────────────────────────────
    # Tier 1: smallest |knee_model - knee_data_mean| s.t. median_dec_all
    # not worse than baseline+0.1.
    # Tier 2 (if none qualifies): smallest knee shift regardless of dec.
    PASS_DEC_BUDGET = 0.10
    best_pass = None
    best_pass_shift = float("inf")
    best_any = None
    best_any_shift = float("inf")
    for tag, s in results.items():
        if "error" in s: continue
        med = s["median_dec_all"]["median"]
        shift = s.get("knee_model_minus_data_mean", float("nan"))
        if not np.isfinite(shift):
            continue
        abs_shift = abs(shift)
        if abs_shift < best_any_shift:
            best_any_shift = abs_shift; best_any = s
        if np.isfinite(med) and (med - LOCKED_BASELINE_MEDIAN_DEC) <= PASS_DEC_BUDGET:
            if abs_shift < best_pass_shift:
                best_pass_shift = abs_shift; best_pass = s

    # ── PASS/FAIL check ───────────────────────────────────────────────
    KNEE_TOLERANCE = 0.20   # model knee may exceed data knee by up to 0.20 V
    pass_record = None
    for tag, s in results.items():
        if "error" in s: continue
        med = s["median_dec_all"]["median"]
        shift = s.get("knee_model_minus_data_mean", float("nan"))
        if not np.isfinite(shift) or not np.isfinite(med):
            continue
        knee_ok = shift <= KNEE_TOLERANCE
        dec_ok  = (med - LOCKED_BASELINE_MEDIAN_DEC) <= PASS_DEC_BUDGET
        if knee_ok and dec_ok:
            if pass_record is None or abs(shift) < abs(pass_record["knee_model_minus_data_mean"]):
                pass_record = s

    # ── verdict.md ────────────────────────────────────────────────────
    lines = []
    lines.append("# Track VG1→Vb capacitive coupling — sweep on locked K1+ALPHA0 card fix\n")
    lines.append("Hypothesis: VG1 capacitively couples to floating P-body via Cgb. Standing")
    lines.append("Vb offset ≈ vb_gate_coupling × VG1 pre-charges body close to parasitic-NPN")
    lines.append("turn-on, so snapback knee fires at lower Vd. **NEW PHYSICS KNOB** — no")
    lines.append("card value. Positive result means 'could explain knee IF such coupling")
    lines.append("exists at this magnitude', NOT a card-derived fix.\n")
    lines.append(f"- Locked baseline: K1@VG1=0.6 = {K1_CARD}, ALPHA0 = {ALPHA0_CARD:.4e}")
    lines.append(f"- Locked baseline median_dec (n=66): {LOCKED_BASELINE_MEDIAN_DEC:.3f}")
    lines.append(f"- vb_gate_coupling grid: {VB_GATE_GRID}")
    lines.append(f"- Knee target curves: VG1={VG1_TARGET} × VG2 ∈ {VG2_KNEE_LIST}\n")

    lines.append("## Sweep table\n")
    lines.append("| vb_coup | knee_model_mean [V] | knee_data_mean [V] | shift (model-data) [V] | median_dec (n=66) | Δ vs 0.665 | conv |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for tag, s in results.items():
        if "error" in s:
            lines.append(f"| {s['vb_gate_coupling']:.2f} | ERROR | | | | | |")
            continue
        med = s["median_dec_all"]["median"]
        n_finite = sum(1 for r in [s.get("median_dec_all", {})] if r.get("n", 0) > 0)
        conv = s["median_dec_all"]["n"] / 66.0
        lines.append(
            f"| {s['vb_gate_coupling']:.2f} | "
            f"{s.get('knee_model_mean', float('nan')):.3f} | "
            f"{s.get('knee_data_mean', float('nan')):.3f} | "
            f"{s.get('knee_model_minus_data_mean', float('nan')):+.3f} | "
            f"{med:.3f} | "
            f"{med - LOCKED_BASELINE_MEDIAN_DEC:+.3f} | "
            f"{conv:.2f} |"
        )

    lines.append("\n## Per-(VG2) knee detail (VG1=0.6, forward branch)\n")
    lines.append("| vb_coup | VG2=-0.1 model/data | VG2=0.0 model/data | VG2=+0.1 model/data | VG2=+0.2 model/data |")
    lines.append("|---:|---:|---:|---:|---:|")
    for tag, s in results.items():
        if "error" in s: continue
        knees = s.get("knees_at_VG1=0.6", {})
        cells = []
        for vg2 in VG2_KNEE_LIST:
            k = knees.get(vg2) or knees.get(str(vg2)) or {}
            km = k.get("knee_model", float("nan"))
            kd = k.get("knee_data",  float("nan"))
            cells.append(f"{km:.3f} / {kd:.3f}")
        lines.append(f"| {s['vb_gate_coupling']:.2f} | " + " | ".join(cells) + " |")

    lines.append("\n## PASS / FAIL\n")
    lines.append(f"PASS criteria:")
    lines.append(f"  - mean (knee_model − knee_data) at VG1=0.6 ≤ +{KNEE_TOLERANCE:.2f} V")
    lines.append(f"  - AND median_dec (n=66) ≤ baseline + {PASS_DEC_BUDGET:.2f} dec (≤ {LOCKED_BASELINE_MEDIAN_DEC+PASS_DEC_BUDGET:.3f})\n")
    if pass_record is not None:
        med = pass_record["median_dec_all"]["median"]
        lines.append(f"**RESULT: PASS** at vb_gate_coupling = {pass_record['vb_gate_coupling']:.2f}")
        lines.append(f"  - knee shift = {pass_record['knee_model_minus_data_mean']:+.3f} V (target ≤ +{KNEE_TOLERANCE:.2f})")
        lines.append(f"  - median_dec = {med:.3f}  (Δ vs baseline {med - LOCKED_BASELINE_MEDIAN_DEC:+.3f})")
    else:
        lines.append("**RESULT: FAIL** — no vb_gate_coupling value satisfies both knee-shift and dec-budget.")
        if best_any is not None:
            lines.append(f"  - closest knee match: vb={best_any['vb_gate_coupling']:.2f} → shift={best_any['knee_model_minus_data_mean']:+.3f} V, median_dec={best_any['median_dec_all']['median']:.3f}")

    lines.append("\n## Best in grid (most compatible with both targets)\n")
    if best_pass is not None:
        lines.append(f"- **vb_gate_coupling = {best_pass['vb_gate_coupling']:.2f}** "
                     f"(passes dec budget AND smallest |knee shift|)")
        lines.append(f"  - knee_model_mean = {best_pass['knee_model_mean']:.3f} V; "
                     f"knee_data_mean = {best_pass['knee_data_mean']:.3f} V; "
                     f"shift = {best_pass['knee_model_minus_data_mean']:+.3f} V")
        lines.append(f"  - median_dec (n=66) = {best_pass['median_dec_all']['median']:.3f}  "
                     f"(Δ vs baseline {best_pass['median_dec_all']['median'] - LOCKED_BASELINE_MEDIAN_DEC:+.3f})")
    else:
        lines.append("- No condition passed the dec budget. Knee-only best:")
        if best_any is not None:
            lines.append(f"  - **vb_gate_coupling = {best_any['vb_gate_coupling']:.2f}**  shift={best_any['knee_model_minus_data_mean']:+.3f} V, median_dec={best_any['median_dec_all']['median']:.3f}")

    lines.append("\n## Provenance / framing\n")
    lines.append("- Baseline builder: `scripts/pillar_I_C3_jts_tat.py::build_pyport_base()`")
    lines.append("- Config knob: `NSRAMCell2TConfig.vb_gate_coupling` (default 0.0; back-compat)")
    lines.append("- Cold-start seed: `Vb_init = vb_gate_coupling × VG1` (Newton relaxes from there)")
    lines.append("- K1+ALPHA0 card patch identical to `track_combo_k1_alpha0.py::run_one` best cell")
    lines.append("- **NO-CHEAT**: vb_gate_coupling is a new physics knob WITHOUT a card value.")
    lines.append("  A positive PASS means the snapback knee position is *consistent with* an")
    lines.append("  oxide-Cgb coupling of this magnitude; it does NOT prove such coupling exists.")

    (OUT / "verdict.md").write_text("\n".join(lines) + "\n")
    print(f"[vbcoup] wrote {OUT / 'verdict.md'}", flush=True)

    # ── Plot ──────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rows_ok = [(s["vb_gate_coupling"], s["knee_model_mean"], s["knee_data_mean"],
                    s["median_dec_all"]["median"],
                    s.get("knee_model_minus_data_mean", float("nan")))
                   for tag, s in results.items() if "error" not in s
                   and np.isfinite(s["median_dec_all"]["median"])]
        rows_ok.sort(key=lambda r: r[0])
        if rows_ok:
            xs = [r[0] for r in rows_ok]
            ymdl = [r[1] for r in rows_ok]
            ydat = [r[2] for r in rows_ok]
            ydec = [r[3] for r in rows_ok]
            yshift = [r[4] for r in rows_ok]

            fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
            ax = axes[0]
            ax.plot(xs, ymdl, "o-", label="model knee (mean over 4 VG2)")
            ax.plot(xs, ydat, "s--", label="data knee (mean over 4 VG2)")
            ax.set_xlabel("vb_gate_coupling")
            ax.set_ylabel("knee Vd [V]")
            ax.set_title("Snapback knee vs VG1→Vb coupling\n(VG1=0.6, K1+ALPHA0 card locked)")
            ax.axhline(np.nanmean(ydat), color="gray", ls=":", alpha=0.5)
            ax.legend(); ax.grid(alpha=0.3)

            ax2 = axes[1]
            ax2.plot(xs, ydec, "o-", color="C2", label="median_dec (n=66)")
            ax2.axhline(LOCKED_BASELINE_MEDIAN_DEC, color="gray", ls="--",
                        label=f"locked baseline {LOCKED_BASELINE_MEDIAN_DEC}")
            ax2.axhline(LOCKED_BASELINE_MEDIAN_DEC + 0.10, color="red", ls=":",
                        label="+0.10 budget")
            ax2.set_xlabel("vb_gate_coupling")
            ax2.set_ylabel("median_dec")
            ax2.set_title("Full 33-bias fit cost")
            ax2.legend(); ax2.grid(alpha=0.3)

            fig.tight_layout()
            fig.savefig(OUT / "plot.png", dpi=110)
            plt.close(fig)
            print(f"[vbcoup] wrote {OUT / 'plot.png'}", flush=True)
    except Exception as e:
        print(f"[vbcoup] plot FAIL: {e}", flush=True)

    print(f"[vbcoup] wrote {OUT / 'ablation.json'}", flush=True)
    print(f"[vbcoup] DONE", flush=True)


if __name__ == "__main__":
    main()
