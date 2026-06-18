#!/usr/bin/env python3
"""Track eta_vds_th — sweep the iii_gain sigmoid threshold to test whether
the snapback knee shifts left (toward Sebas data at 0.85-1.15V) when the
non-physical eta_vds_th fudge is lowered.

Honest framing
--------------
Per nsram_cell_2T.py:1476-1486 the iii_gain sigmoid is *itself* a
non-physical bounded fudge introduced post-O19 to replace an even more
unphysical unbounded gain. Defaults (eta_max=1.0, eta_slope=10/V,
eta_vds_th=1.0V) suppress Iii by 99.3% at Vd=0.5 and 50% only at Vd=1.0V.
The observed model snapback knee is therefore stuck at Vd≈1.5V (the
sigmoid mid-rise region) instead of the data's 0.85-1.15V.

This script sweeps eta_vds_th ∈ {0.3, 0.5, 0.7, 1.0 baseline, 1.5} plus
an iii_gain≡1 case (eta_max=1.0, eta_slope=0 → sigmoid(0)=0.5; we also
test eta_vds_th=-100 which forces sigmoid≈1 i.e. iii_gain≡eta_max). The
"remove the fudge" baseline = eta_vds_th=-100, eta_max=1.0, slope=10.

What we measure per condition:
  - knee_vd at VG1=0.6 for VG2 ∈ {-0.1, 0.0, 0.1, 0.2}, defined as the
    Vd ∈ [0.5, 1.6V] where the log-current derivative d(log10 Id)/dVd
    is maximal in the *forward* sweep (this is the snapback turn-on
    edge — see data inspection in commit message).
  - full 33-bias median_dec via pillar.run_grid (fwd+bwd) to confirm
    the global DC fit isn't destroyed.

LOCKS (card values from track_combo_k1_alpha0 / Mario):
  - K1@VG1=0.6 = 0.53825
  - ALPHA0     = 7.83756e-4
  - hurkx_bbt_A = 0.0 (Hurkx OFF)

PASS gate
---------
  PASS_knee  : max |knee_vd_model - knee_vd_data| ≤ 0.2V across 4 VG2
  PASS_dec   : median_dec_all ≤ baseline median_dec + 0.10 dec
  PASS_overall = PASS_knee AND PASS_dec

NO-CHEAT note
-------------
If a low eta_vds_th lands the knee at the right Vd without breaking the
33-bias fit, we are tuning one non-physical fudge against another. The
physically-honest action is then to drop the iii_gain sigmoid entirely
(case REMOVE_GAIN below) and revisit Bf / IS / NPN topology.

Outputs:
  results/track_eta_vds_th/{ablation.json, verdict.md, plot.png}
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

OUT = ROOT / "results/track_eta_vds_th"
OUT.mkdir(parents=True, exist_ok=True)

# Locked card values
K1_CARD     = 0.53825
ALPHA0_CARD = 7.83756e-4

# Sweep
ETA_VDS_TH_GRID = [0.3, 0.5, 0.7, 1.0, 1.5]   # baseline = 1.0
EXTRA_CASES = ["REMOVE_GAIN"]                 # iii_gain ≡ 1 (sigmoid forced ≈1)

# Knee detection
VG1_KNEE = 0.6
VG2_KNEE_LIST = [-0.1, 0.0, 0.1, 0.2]
KNEE_VD_LO, KNEE_VD_HI = 0.5, 1.6

# Baseline median_dec reference (canonical pillar, post-A.0)
BASELINE_MEDIAN_DEC = 1.39    # current honest baseline (Phase A, Bf=100 η≤1)
DEC_SLACK = 0.10              # PASS if median_dec ≤ baseline + 0.10


# ---- locked-card make_overrides patcher ----
def install_locks():
    orig_make = pillar.make_overrides
    saved_branch = pillar.BRANCH_FLAT[0.6]["K1"]
    pillar.BRANCH_FLAT[0.6]["K1"] = float(K1_CARD)
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
    return orig_make, saved_branch

def uninstall_locks(orig_make, saved_branch):
    pillar.make_overrides = orig_make
    pillar.BRANCH_FLAT[0.6]["K1"] = saved_branch


# ---- knee detection on forward sweep ----
def detect_knee(Vd, Id, lo=KNEE_VD_LO, hi=KNEE_VD_HI):
    Vd = np.asarray(Vd, dtype=np.float64)
    Id = np.abs(np.asarray(Id, dtype=np.float64))
    m = (Vd >= lo) & (Vd <= hi)
    if m.sum() < 3:
        return None, None
    sV = Vd[m]; sI = np.clip(Id[m], 1e-12, None)
    # uniformise: ensure monotone-increasing Vd (forward sweep already is)
    order = np.argsort(sV)
    sV = sV[order]; sI = sI[order]
    lg = np.log10(sI)
    dlg = np.diff(lg) / np.maximum(np.diff(sV), 1e-9)
    idx = int(np.argmax(dlg))
    return float(0.5 * (sV[idx] + sV[idx + 1])), float(dlg[idx])


def load_data_knees(curves):
    out = {}
    for c in curves:
        if abs(c["VG1"] - VG1_KNEE) > 1e-6: continue
        for vg2 in VG2_KNEE_LIST:
            if abs(c["VG2"] - vg2) > 1e-6: continue
            k, d = detect_knee(c["fwd_Vd"], c["fwd_Id"])
            out[round(vg2, 3)] = {"knee_vd": k, "dlog_dV": d}
    return out


def model_knee_for_vg2(cfg, M1, M2, bjt, curves, vg2_target):
    """Run forward_2t with the same Vd grid as the data curve for VG1=0.6 / vg2."""
    target = None
    for c in curves:
        if abs(c["VG1"] - VG1_KNEE) < 1e-6 and abs(c["VG2"] - vg2_target) < 1e-6:
            target = c; break
    if target is None:
        return None
    row_sebas, _ = pillar.find_or_impute_row(pillar.load_sebas_params(),
                                             VG1_KNEE, vg2_target)
    P_M1, P_M2 = pillar.make_overrides(row_sebas)
    sd_M1 = cfg.size_dep_M1(M1); sd_M2 = cfg.size_dep_M2(M2)
    Vd_np = target["fwd_Vd"]
    Vd_t = torch.tensor(Vd_np, dtype=torch.float64)
    try:
        with pillar.patch_sd_scaled(sd_M1, P_M1), pillar.patch_sd_scaled(sd_M2, P_M2):
            out = forward_2t(cfg, model_M1=M1, model_M2=M2, bjt=bjt, Vd_seq=Vd_t,
                             VG1=torch.tensor(VG1_KNEE, dtype=torch.float64),
                             VG2=torch.tensor(vg2_target, dtype=torch.float64),
                             warm_start=True)
        Id = np.abs(out["Id"].detach().cpu().numpy()).astype(np.float64)
    except Exception as e:
        print(f"    forward_2t failed VG2={vg2_target}: {e}", flush=True)
        return None
    k, d = detect_knee(Vd_np, Id)
    return {"vg2": float(vg2_target), "knee_vd_model": k, "dlog_dV_model": d,
            "Vd": Vd_np.tolist(), "Id_model": Id.tolist()}


# ---- per-condition runner ----
def configure_cfg(cfg, label, eta_vds_th):
    """Set cfg.eta_max / slope / vds_th for the given label.
       Also disables Hurkx unconditionally."""
    cfg.hurkx_bbt_A = 0.0
    # ensure legacy override is OFF (iii_body_gain must be None or 1.0)
    cfg.iii_body_gain = None
    if label == "REMOVE_GAIN":
        # iii_gain ≡ 1 : eta_max=1, slope large, vds_th=-100 → sigmoid≈1
        cfg.eta_max = 1.0
        cfg.eta_slope = 10.0
        cfg.eta_vds_th = -100.0
    else:
        cfg.eta_max = 1.0
        cfg.eta_slope = 10.0
        cfg.eta_vds_th = float(eta_vds_th)


def run_one(label, eta_vds_th, curves, sebas_rows, data_knees):
    print(f"\n[eta_sweep] === {label}  eta_vds_th={eta_vds_th} ===", flush=True)
    cfg, M1, M2, bjt = pillar.build_pyport_base()
    configure_cfg(cfg, label, eta_vds_th)

    orig_make, saved_branch = install_locks()
    try:
        # 1. Knee scan (4 VG2 values)
        knee_rows = []
        for vg2 in VG2_KNEE_LIST:
            kr = model_knee_for_vg2(cfg, M1, M2, bjt, curves, vg2)
            if kr is None:
                kr = {"vg2": vg2, "knee_vd_model": None, "dlog_dV_model": None}
            data_k = data_knees.get(round(vg2, 3), {}).get("knee_vd")
            kr["knee_vd_data"] = data_k
            if kr["knee_vd_model"] is not None and data_k is not None:
                kr["knee_shift"] = float(kr["knee_vd_model"] - data_k)
                kr["abs_knee_shift"] = abs(kr["knee_shift"])
            else:
                kr["knee_shift"] = None
                kr["abs_knee_shift"] = None
            knee_rows.append(kr)

        # 2. Full 33-bias fit
        t0 = time.time()
        rows, nan_count = pillar.run_grid(cfg, M1, M2, bjt, curves, sebas_rows,
                                          label, do_bwd=True)
        dt = time.time() - t0
    finally:
        uninstall_locks(orig_make, saved_branch)

    summ = pillar.summarize(rows, label)
    summ["label"] = label
    summ["eta_vds_th"] = float(eta_vds_th) if isinstance(eta_vds_th,(int,float)) else eta_vds_th
    summ["eta_max"] = float(cfg.eta_max)
    summ["eta_slope"] = float(cfg.eta_slope)
    summ["hurkx_bbt_A"] = float(cfg.hurkx_bbt_A)
    summ["nan_count"] = int(nan_count)
    summ["runtime_s"] = float(dt)
    finite = sum(1 for r in rows if np.isfinite(r["med_dec"]) and r["med_dec"] > 0)
    summ["n_rows"] = len(rows); summ["n_finite"] = finite
    summ["convergence_rate"] = finite / max(len(rows), 1)
    summ["knees"] = knee_rows
    valid_shifts = [k["abs_knee_shift"] for k in knee_rows if k["abs_knee_shift"] is not None]
    summ["max_abs_knee_shift"] = float(max(valid_shifts)) if valid_shifts else None
    summ["mean_abs_knee_shift"] = float(np.mean(valid_shifts)) if valid_shifts else None
    return summ


def main():
    sebas_rows = pillar.load_sebas_params()
    curves = pillar.load_curves()
    print(f"[eta_sweep] loaded {len(curves)} curves, {len(sebas_rows)} sebas rows", flush=True)

    data_knees = load_data_knees(curves)
    print(f"[eta_sweep] data knees @ VG1=0.6:", flush=True)
    for vg2 in VG2_KNEE_LIST:
        dk = data_knees.get(round(vg2, 3))
        print(f"   VG2={vg2:+.2f}: knee_vd={dk['knee_vd']:.3f}  d(log Id)/dV={dk['dlog_dV']:.2f}", flush=True)

    results = {}
    # numeric grid
    for th in ETA_VDS_TH_GRID:
        tag = f"eta_vds_th={th:.2f}"
        try:
            results[tag] = run_one(tag, th, curves, sebas_rows, data_knees)
        except Exception as e:
            print(f"[eta_sweep] FAIL {tag}: {e}", flush=True)
            traceback.print_exc()
            results[tag] = {"label": tag, "eta_vds_th": th, "error": str(e)}
        with open(OUT / "ablation.json", "w") as f:
            json.dump({"data_knees": data_knees, "results": results}, f, indent=2, default=str)

    # extra case: REMOVE_GAIN
    for tag_case in EXTRA_CASES:
        try:
            results[tag_case] = run_one(tag_case, -100.0, curves, sebas_rows, data_knees)
        except Exception as e:
            print(f"[eta_sweep] FAIL {tag_case}: {e}", flush=True)
            traceback.print_exc()
            results[tag_case] = {"label": tag_case, "eta_vds_th": -100.0, "error": str(e)}
        with open(OUT / "ablation.json", "w") as f:
            json.dump({"data_knees": data_knees, "results": results}, f, indent=2, default=str)

    # ---- verdict.md ----
    lines = []
    lines.append("# Track eta_vds_th — iii_gain Sigmoid Threshold Sweep")
    lines.append("")
    lines.append("**Hypothesis**: the snapback knee sits at Vd≈1.5V (vs data 0.85-1.15V) because the\n"
                 "iii_gain sigmoid `eta_max·σ(eta_slope·(Vds-eta_vds_th))` is centered at Vds_th=1.0V,\n"
                 "suppressing Iii by 99.3% at Vd=0.5V. Lowering eta_vds_th shifts the sigmoid left and\n"
                 "should let Iii charge the body earlier → knee moves left.")
    lines.append("")
    lines.append("**Locks (card values)**: K1@VG1=0.6 = 0.53825, ALPHA0 = 7.83756e-4, hurkx_bbt_A = 0.")
    lines.append("")
    lines.append(f"**PASS gate**: max|knee_shift| ≤ 0.2 V AND median_dec ≤ {BASELINE_MEDIAN_DEC}+{DEC_SLACK}.")
    lines.append("")
    lines.append("## Data knees (VG1=0.6, forward sweep, max d(log10 Id)/dVd in [0.5, 1.6] V)")
    lines.append("| VG2 | knee_vd_data (V) | d(log Id)/dV |")
    lines.append("|---:|---:|---:|")
    for vg2 in VG2_KNEE_LIST:
        dk = data_knees.get(round(vg2, 3))
        lines.append(f"| {vg2:+.2f} | {dk['knee_vd']:.3f} | {dk['dlog_dV']:.2f} |")
    lines.append("")
    lines.append("## Per-condition results")
    lines.append("| condition | eta_vds_th | knee VG2=-0.10 | knee VG2=0.00 | knee VG2=0.10 | knee VG2=0.20 | max|Δknee| | median_dec_all | conv | PASS_knee | PASS_dec |")
    lines.append("|:---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|")

    def knee_str(s, vg2):
        for k in s.get("knees", []):
            if abs(k["vg2"] - vg2) < 1e-6:
                if k["knee_vd_model"] is None: return "—"
                return f"{k['knee_vd_model']:.3f} (Δ={k['knee_shift']:+.3f})"
        return "—"

    for tag, s in results.items():
        if "error" in s:
            lines.append(f"| {tag} | — | ERROR: {s['error'][:40]} | | | | | | | | |")
            continue
        med = s["median_dec_all"]["median"]
        mks = s.get("max_abs_knee_shift")
        pass_knee = (mks is not None) and (mks <= 0.2)
        pass_dec  = np.isfinite(med) and (med <= BASELINE_MEDIAN_DEC + DEC_SLACK)
        lines.append(
            f"| {tag} | {s['eta_vds_th']} | "
            f"{knee_str(s, -0.1)} | {knee_str(s, 0.0)} | {knee_str(s, 0.1)} | {knee_str(s, 0.2)} | "
            f"{('%.3f' % mks) if mks is not None else '—'} | "
            f"{med:.3f} | {s['convergence_rate']:.2f} | "
            f"{'PASS' if pass_knee else 'FAIL'} | {'PASS' if pass_dec else 'FAIL'} |"
        )

    # Pick best (lowest max|Δknee| among PASS_dec; tie-break by median_dec)
    best_tag, best_score = None, (float("inf"), float("inf"))
    for tag, s in results.items():
        if "error" in s: continue
        mks = s.get("max_abs_knee_shift")
        med = s["median_dec_all"]["median"]
        if mks is None or not np.isfinite(med): continue
        score = (mks, med)
        if score < best_score:
            best_score = score; best_tag = tag

    lines.append("")
    if best_tag is not None:
        bs = results[best_tag]
        med = bs["median_dec_all"]["median"]
        mks = bs["max_abs_knee_shift"]
        pass_overall = (mks <= 0.2) and (med <= BASELINE_MEDIAN_DEC + DEC_SLACK)
        lines.append(f"## Best: **{best_tag}**  (max|Δknee|={mks:.3f} V, median_dec={med:.3f})")
        lines.append(f"- PASS_overall: **{'YES' if pass_overall else 'NO'}**")
    else:
        lines.append("## No usable condition (all errored or returned NaN knees)")

    # NO-CHEAT framing
    lines.append("")
    lines.append("## Honest framing (NO-CHEAT)")
    lines.append("`eta_vds_th` is itself a non-physical fudge (see `nsram_cell_2T.py:1476-1486`).\n"
                 "Lowering it to land the knee at the right Vd amounts to trading one knob for\n"
                 "another. The physically-honest alternative — removing the iii_gain sigmoid\n"
                 "entirely so that all impact-ionized holes reach the body — is tested as\n"
                 "**REMOVE_GAIN** (eta_vds_th=-100, sigmoid≈1). If REMOVE_GAIN beats or matches\n"
                 "the best tuned threshold, the iii_gain sigmoid should be retired.")

    lines.append("")
    lines.append("## Provenance")
    lines.append("- iii_gain formula: `nsram_cell_2T.py:1508`")
    lines.append("- Knee detector: max d(log10 Id)/dVd over Vd∈[0.5, 1.6] V (forward branch)")
    lines.append("- Baseline median_dec reference: 1.39 dec (Phase A honest fit, Bf=100 η≤1)")
    lines.append("- Lock pattern: `track_combo_k1_alpha0.py`")

    (OUT / "verdict.md").write_text("\n".join(lines) + "\n")
    print(f"[eta_sweep] wrote {OUT / 'verdict.md'}", flush=True)

    # ---- plot ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
        for ax, vg2 in zip(axes.flat, VG2_KNEE_LIST):
            # plot data
            for c in curves:
                if abs(c["VG1"] - 0.6) < 1e-6 and abs(c["VG2"] - vg2) < 1e-6:
                    ax.semilogy(c["fwd_Vd"], np.abs(c["fwd_Id"]), "k-", lw=2, label="data")
                    break
            dk = data_knees.get(round(vg2, 3), {}).get("knee_vd")
            if dk is not None:
                ax.axvline(dk, color="k", ls=":", lw=1, alpha=0.5)
            for tag, s in results.items():
                if "error" in s: continue
                for k in s.get("knees", []):
                    if abs(k["vg2"] - vg2) < 1e-6 and k.get("Id_model") is not None:
                        ax.semilogy(k["Vd"], k["Id_model"], lw=1.0, alpha=0.8, label=tag)
                        if k["knee_vd_model"] is not None:
                            ax.axvline(k["knee_vd_model"], ls="--", lw=0.5, alpha=0.4)
                        break
            ax.set_title(f"VG1=0.6  VG2={vg2:+.2f}")
            ax.set_xlim(0.0, 2.0)
            ax.set_ylim(1e-9, 1e-3)
            ax.grid(True, which="both", alpha=0.3)
        axes[1, 0].set_xlabel("Vd [V]"); axes[1, 1].set_xlabel("Vd [V]")
        axes[0, 0].set_ylabel("|Id| [A]"); axes[1, 0].set_ylabel("|Id| [A]")
        axes[0, 0].legend(fontsize=7, loc="lower right")
        fig.suptitle("iii_gain eta_vds_th sweep — snapback knee landing (VG1=0.6)")
        fig.tight_layout()
        fig.savefig(OUT / "plot.png", dpi=110)
        print(f"[eta_sweep] wrote {OUT / 'plot.png'}", flush=True)
    except Exception as e:
        print(f"[eta_sweep] plot failed: {e}", flush=True)


if __name__ == "__main__":
    main()
