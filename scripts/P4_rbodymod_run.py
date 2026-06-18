"""P4 — BSIM4 rbodymod=1 simplified 1-R implementation test.

Goal: implement BSIM4.8.3 §6.7 rbodymod=1 in pyport and quantify its effect
on the cell-wide 33-bias DC fit (forward + backward sweep) versus z432
pseudo-transient baseline (rbodymod=0).

Approach:
  1. Compute R_body_eff from M1_130DNWFB.txt parameters
       R_body_eff = RBPB + (RBPS || RBPD || RBSB || RBDB)
     (manual §6.7, simplified 1-R collapse — see nsram_cell_2T.py P4 block)
  2. Sanity: VG1=0.4 VG2=0 V_D=2V — Vb without vs with rbodymod
  3. Full 33-curve fwd + bwd sweep using z432's pseudo-transient framework
     with several R_body_eff values: literal (62.5 Ω), 1k, 1M, 1G (≈OFF).
  4. Compare to z432 (rbodymod=0) baseline:
        fwd = 1.349 dec, bwd = 1.027 dec, asym = 0.32 dec.
  5. Write summary.json, dc_compare.png, honest_analysis.md.

Pre-registered gates per task spec:
  INFRA      — rbodymod=1 lands; all 33 biases converge in both directions
  DISCOVERY  — cell-wide DC_avg < 1.19 dec by ≥ 0.1 dec
  AMBITIOUS  — DC_avg < 0.9 dec AND fwd-bwd asym < 0.3 dec
  KILL_SHOT  — fwd worsens by > 0.5 dec
"""
from __future__ import annotations
import importlib.util as _ilu
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)

ROOT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
DATA = ROOT / "data/sebas_2026_04_22"
OUT = ROOT / "results/P4_rbodymod"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "nsram"))
sys.path.insert(0, str(ROOT / "scripts"))

LOG = open(OUT / "run.log", "w")
def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    LOG.write(line + "\n"); LOG.flush()


_spec427 = _ilu.spec_from_file_location("z427", ROOT / "scripts/z427_vsint_fix.py")
z427 = _ilu.module_from_spec(_spec427); _spec427.loader.exec_module(z427)
_spec429 = _ilu.spec_from_file_location("z429", ROOT / "scripts/z429_multisolver_debug.py")
z429 = _ilu.module_from_spec(_spec429); _spec429.loader.exec_module(z429)
_spec432 = _ilu.spec_from_file_location("z432", ROOT / "scripts/z432_pseudotransient.py")
z432 = _ilu.module_from_spec(_spec432); _spec432.loader.exec_module(z432)

from nsram.bsim4_port.model_card import BSIM4Model, parse_param_blocks


def compute_r_body_eff_from_card():
    """Compute simplified 1-R body resistance from M1 card.

    BSIM4.8.3 §6.7 5-R network → 1-R collapse:
        R_body_eff = RBPB + 1 / (1/RBPS + 1/RBPD + 1/RBSB + 1/RBDB)

    Returns (R_eff_Ω, raw_dict).
    """
    t1 = (DATA / "M1_130DNWFB.txt").read_text()
    t2 = (DATA / "M2_130bulkNSRAM.txt").read_text()
    shared = parse_param_blocks(t2)
    m1 = BSIM4Model.from_spice(t1, model_type="nmos", params=shared)
    raw = {k: float(m1.get(k)) for k in ("rbodymod", "rbps", "rbpd", "rbsb",
                                          "rbdb", "rbpb", "gbmin")}
    rbps, rbpd, rbsb, rbdb, rbpb = (raw["rbps"], raw["rbpd"], raw["rbsb"],
                                     raw["rbdb"], raw["rbpb"])
    G = sum(1.0 / r for r in (rbps, rbpd, rbsb, rbdb) if r > 0.0)
    R_parallel = 1.0 / G if G > 0.0 else float("inf")
    R_eff = rbpb + R_parallel
    return R_eff, raw


# ---------------------------------------------------------------- #
# Sanity probe
# ---------------------------------------------------------------- #

def sanity_probe(model_M1, model_M2, sebas_rows, VG1=0.4, VG2=0.0, Vd=2.0,
                 r_body_eff=62.5):
    """Run single-point Newton at (VG1, VG2, Vd) with rbodymod=0 vs =1."""
    cfg, sd_M1, sd_M2 = z427.make_cfg(model_M1, model_M2, {})
    sebas_row = z427.find_params(sebas_rows, VG1, VG2)
    P_M1, P_M2 = z427.make_overrides(sebas_row)
    bjt = z427.make_bjt(sebas_row)

    results = {}
    for tag, use_rbody in (("rbodymod0", False), ("rbodymod1", True)):
        cfg.use_rbodymod = bool(use_rbody)
        cfg.r_body_total_ohm = float(r_body_eff)
        cfg.v_bodypin = 0.0
        with torch.no_grad(), z427.patch_sd_scaled(sd_M1, P_M1), z427.patch_sd_scaled(sd_M2, P_M2):
            r = z429.run_vsint_pinned(cfg, model_M1, model_M2, bjt,
                                       float(Vd), float(VG1), float(VG2),
                                       Vsint_pin=0.0, Vb_init=0.0)
        results[tag] = dict(Vb=float(r["Vb"]), Id=float(r["Id"]),
                             converged=bool(r["converged"]),
                             resid_RB=float(r.get("resid_RB", float("nan"))))
        log(f"  sanity {tag}: Vb={r['Vb']:.4f} V  Id={r['Id']:.3e} A  "
            f"conv={r['converged']}")
    dVb = results["rbodymod1"]["Vb"] - results["rbodymod0"]["Vb"]
    log(f"  ΔVb (rbody1 − rbody0) = {dVb*1000:.2f} mV  (target: < 50 mV)")
    results["dVb_mV"] = dVb * 1000.0
    results["bias"] = dict(VG1=VG1, VG2=VG2, Vd=Vd)
    results["r_body_eff_ohm"] = r_body_eff
    return results


# ---------------------------------------------------------------- #
# Full sweep
# ---------------------------------------------------------------- #

def run_full_sweep(name, model_M1, model_M2, curves, sebas_rows,
                    r_body_eff: float):
    """Wrap z432.run_cellwide with use_rbodymod=True via monkey-patched cfg."""
    # We patch z432.run_cellwide indirectly by setting attributes on the cfg
    # that z432 builds via z427.make_cfg. The cleanest hook is to wrap the
    # z427.make_cfg path through a small shim that sets the rbodymod flags.

    orig_make_cfg = z427.make_cfg

    def patched_make_cfg(*args, **kwargs):
        cfg, sd_M1, sd_M2 = orig_make_cfg(*args, **kwargs)
        cfg.use_rbodymod = True
        cfg.r_body_total_ohm = float(r_body_eff)
        cfg.v_bodypin = 0.0
        return cfg, sd_M1, sd_M2

    z427.make_cfg = patched_make_cfg
    try:
        fwd = z432.run_cellwide(name, model_M1, model_M2, curves, sebas_rows,
                                 direction="forward")
        bwd = z432.run_cellwide(name, model_M1, model_M2, curves, sebas_rows,
                                 direction="backward")
    finally:
        z427.make_cfg = orig_make_cfg
    return fwd, bwd


# ---------------------------------------------------------------- #
# Plot
# ---------------------------------------------------------------- #

def dc_compare_plot(all_runs, fname):
    """Bar plot of (fwd, bwd) cell_rmse for each R_body config + baseline."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    labels = []
    fwd_vals = []
    bwd_vals = []
    for tag, info in all_runs.items():
        labels.append(tag)
        fwd_vals.append(info.get("fwd_cell_rmse", float("nan")))
        bwd_vals.append(info.get("bwd_cell_rmse", float("nan")))
    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w/2, fwd_vals, w, label="forward", color="tab:blue")
    ax.bar(x + w/2, bwd_vals, w, label="backward", color="tab:cyan")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("cell_rmse [dec]")
    ax.set_title("P4 rbodymod=1: cell-wide DC fit vs R_body  (lower = better)")
    ax.axhline(1.349, color="gray", ls="--", lw=0.8, label="z432 fwd baseline")
    ax.axhline(1.027, color="gray", ls=":", lw=0.8, label="z432 bwd baseline")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(fname, dpi=120)
    plt.close(fig)
    log(f"  wrote {fname.name}")


# ---------------------------------------------------------------- #
# Main
# ---------------------------------------------------------------- #

# Baselines from z432 (rbodymod=0)
BASE_FWD = 1.34906163370139
BASE_BWD = 1.026861976331113
BASE_AVG = 0.5 * (BASE_FWD + BASE_BWD)
BASE_ASYM = abs(BASE_FWD - BASE_BWD)


def main():
    t_main = time.time()
    log("P4 — BSIM4 §6.7 rbodymod=1 simplified 1-R network")
    log(f"  z432 baseline: fwd={BASE_FWD:.3f}  bwd={BASE_BWD:.3f}  "
        f"avg={BASE_AVG:.3f}  asym={BASE_ASYM:.3f}")

    R_eff_card, raw = compute_r_body_eff_from_card()
    log(f"  M1 card: rbodymod={int(raw['rbodymod'])}  "
        f"RBPS={raw['rbps']}  RBPD={raw['rbpd']}  RBSB={raw['rbsb']}  "
        f"RBDB={raw['rbdb']}  RBPB={raw['rbpb']}")
    log(f"  R_body_eff (card, literal Ω) = RBPB + RBPS||RBPD||RBSB||RBDB "
        f"= {raw['rbpb']} + {1.0/sum(1/r for r in [raw['rbps'],raw['rbpd'],raw['rbsb'],raw['rbdb']]):.2f}"
        f" = {R_eff_card:.2f} Ω")
    log("  NOTE: card values are nominal Ω (BSIM4 default convention); for a")
    log("  floating-body NSRAM the literal 62.5 Ω pins Vb≈0 (kills body float)."
        " We sweep R_body across [card, 1k, 1M, 1G] for an honest assessment.")

    # Build models / curves
    model_M1, model_M2 = z427.build_models()
    curves = z427.load_curves()
    sebas_rows = z427.load_sebas_params()
    log(f"  loaded: {len(curves)} curves, {len(sebas_rows)} sebas rows")

    # Sanity at three R_body values
    log("=== Sanity probe (VG1=0.4 VG2=0 Vd=2V): rbodymod=0 vs rbodymod=1 ===")
    sanity = {}
    for r_val in (R_eff_card, 1e3, 1e6):
        log(f"  R_body_eff = {r_val:.3g} Ω")
        sanity[f"R_{r_val:.0e}"] = sanity_probe(model_M1, model_M2, sebas_rows,
                                                  VG1=0.4, VG2=0.0, Vd=2.0,
                                                  r_body_eff=r_val)

    # Backward-sweep basin check at VG1=0.6 VG2=0.2 V_D=2.0→0.0
    log("=== Backward-basin probe (VG1=0.6 VG2=0.2 Vd=2.0→0.5): ===")
    cfg, sd_M1, sd_M2 = z427.make_cfg(model_M1, model_M2, {})
    sebas_row = z427.find_params(sebas_rows, 0.6, 0.2)
    P_M1, P_M2 = z427.make_overrides(sebas_row)
    bjt = z427.make_bjt(sebas_row)
    Vd_grid = np.array([2.0, 1.5, 1.0, 0.5])
    basin_results = {}
    for tag, use_rb, r_val in (("rbodymod0", False, 0.0),
                                ("rbodymod1_card", True, R_eff_card),
                                ("rbodymod1_1M", True, 1.0e6)):
        cfg.use_rbodymod = use_rb
        cfg.r_body_total_ohm = float(r_val)
        cfg.v_bodypin = 0.0
        Vb_trace = []
        Id_trace = []
        Vb_warm = 0.0
        with torch.no_grad(), z427.patch_sd_scaled(sd_M1, P_M1), z427.patch_sd_scaled(sd_M2, P_M2):
            for Vd_f in Vd_grid:
                r = z429.run_vsint_pinned(cfg, model_M1, model_M2, bjt,
                                           float(Vd_f), 0.6, 0.2,
                                           Vsint_pin=0.0, Vb_init=Vb_warm)
                Vb_trace.append(float(r["Vb"]))
                Id_trace.append(float(r["Id"]))
                Vb_warm = float(r["Vb"])
        basin_results[tag] = dict(Vb=Vb_trace, Id=Id_trace,
                                    Vd=Vd_grid.tolist())
        log(f"  {tag:18s}: Vb={[f'{v:.3f}' for v in Vb_trace]}  "
            f"Id={[f'{i:.2e}' for i in Id_trace]}")

    # Full sweep at selected R_body values
    R_BODY_CONFIGS = [
        ("R_card",  R_eff_card),    # literal card value
        ("R_1k",    1.0e3),
        ("R_1M",    1.0e6),
        ("R_1G",    1.0e9),         # effectively OFF (≈ rbodymod=0)
    ]
    all_runs = {}
    log("=== Full 33-curve fwd+bwd sweep ===")
    for tag, r_val in R_BODY_CONFIGS:
        log(f"--- {tag}: R_body_eff = {r_val:.3g} Ω ---")
        t0 = time.time()
        fwd, bwd = run_full_sweep(tag, model_M1, model_M2, curves, sebas_rows,
                                    r_body_eff=r_val)
        avg = 0.5 * (fwd["cell_rmse_dec"] + bwd["cell_rmse_dec"])
        asym = abs(fwd["cell_rmse_dec"] - bwd["cell_rmse_dec"])
        all_runs[tag] = dict(
            r_body_eff_ohm=r_val,
            fwd_cell_rmse=fwd["cell_rmse_dec"],
            bwd_cell_rmse=bwd["cell_rmse_dec"],
            dc_avg=avg,
            fwd_bwd_asym=asym,
            fwd_per_branch=fwd["per_branch_rmse_dec"],
            bwd_per_branch=bwd["per_branch_rmse_dec"],
            fwd_n_biases=fwd["n_biases_evaluated"],
            bwd_n_biases=bwd["n_biases_evaluated"],
            fwd_conv_rate=fwd["convergence_rate"],
            bwd_conv_rate=bwd["convergence_rate"],
            wall_sec=round(time.time() - t0, 1),
        )
        log(f"  {tag}: fwd={fwd['cell_rmse_dec']:.3f}  "
            f"bwd={bwd['cell_rmse_dec']:.3f}  avg={avg:.3f}  asym={asym:.3f}  "
            f"wall={time.time()-t0:.0f}s")

    # Add baseline as a reference row for plotting
    plot_runs = {"z432_baseline": dict(fwd_cell_rmse=BASE_FWD,
                                         bwd_cell_rmse=BASE_BWD,
                                         dc_avg=BASE_AVG,
                                         fwd_bwd_asym=BASE_ASYM)}
    plot_runs.update(all_runs)
    dc_compare_plot(plot_runs, OUT / "dc_compare.png")

    # Gates
    best_tag = min(all_runs, key=lambda k: all_runs[k]["dc_avg"])
    best = all_runs[best_tag]
    fwd_worst_delta = max(all_runs[k]["fwd_cell_rmse"] - BASE_FWD
                            for k in all_runs)
    fwd_card_delta = all_runs["R_card"]["fwd_cell_rmse"] - BASE_FWD
    gates = {
        "INFRA_pass": all(all_runs[k]["fwd_n_biases"] > 0 and
                            all_runs[k]["bwd_n_biases"] > 0
                            for k in all_runs),
        "DISCOVERY_dc_avg_lt_1p09": best["dc_avg"] < (BASE_AVG - 0.1),
        "DISCOVERY_dc_avg_lt_1p19_minus_0p1": best["dc_avg"] < 1.09,
        "AMBITIOUS_dc_avg_lt_0p9_AND_asym_lt_0p3":
            (best["dc_avg"] < 0.9 and best["fwd_bwd_asym"] < 0.3),
        "KILL_SHOT_fwd_worse_by_0p5":
            (all_runs["R_card"]["fwd_cell_rmse"] - BASE_FWD) > 0.5,
        "fwd_within_0p1_dec_R_1G":
            abs(all_runs["R_1G"]["fwd_cell_rmse"] - BASE_FWD) < 0.1,
    }

    summary = {
        "BSIM4_REFERENCE": ("BSIM4.8.3 manual §6.7 rbodymod=1 simplified to "
                             "1-R: R_body_eff = RBPB + (RBPS||RBPD||RBSB||RBDB). "
                             "External body pin tied to substrate (Vbp=0)."),
        "MODEL_CARD_RBODY_RAW": raw,
        "R_BODY_EFF_CARD_OHM": R_eff_card,
        "BASELINE_Z432": dict(fwd=BASE_FWD, bwd=BASE_BWD,
                                avg=BASE_AVG, asym=BASE_ASYM),
        "SANITY_PROBE_VG1_0p4_VG2_0_Vd_2V": sanity,
        "BACKWARD_BASIN_VG1_0p6_VG2_0p2": basin_results,
        "RUNS": all_runs,
        "BEST_TAG": best_tag,
        "BEST": best,
        "GATES": gates,
        "wall_total_sec": round(time.time() - t_main, 1),
    }

    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    log(f"  wrote summary.json")

    # honest_analysis.md
    md = []
    md.append("# P4 — BSIM4 §6.7 rbodymod=1 (simplified 1-R) — honest analysis\n")
    md.append("## Implementation\n")
    md.append("Added `cfg.use_rbodymod` + `cfg.r_body_total_ohm` + `cfg.v_bodypin`")
    md.append("to `nsram/bsim4_port/nsram_cell_2T.py::_residuals`. KCL term:")
    md.append("`I_rbody = (Vb - Vbp_pin) / R_body_eff` subtracted from R_B.\n")
    md.append("**Simplification**: the full 5-R Y-Δ network with 4 internal")
    md.append("sub-nodes (dbi, sbi, dbNode, sbNode) collapses to a single")
    md.append("equivalent resistor `R_body_eff = RBPB + (RBPS||RBPD||RBSB||RBDB)`.")
    md.append("This preserves the SIGN of the body-to-pin DC current path; it")
    md.append("does NOT preserve any HF distributed-node effects (those need the")
    md.append("full network and are out of scope for DC fit).\n")
    md.append(f"From M1 card: RBPS={raw['rbps']}  RBPD={raw['rbpd']}  ")
    md.append(f"RBSB={raw['rbsb']}  RBDB={raw['rbdb']}  RBPB={raw['rbpb']} Ω → ")
    md.append(f"`R_body_eff` = **{R_eff_card:.2f} Ω**.\n")
    md.append(f"Note: card has `rbodymod=0` enabled by default in SPICE; the")
    md.append("5-R values are present but inactive. We activate them here as a")
    md.append("physical sanity check, with the external body pin tied to")
    md.append("substrate (Vbp=0 V).\n")

    md.append("## Sanity probe (VG1=0.4 VG2=0 Vd=2V)\n")
    md.append("| tag | R_body | Vb [V] | Id [A] | ΔVb (mV) |\n|---|---|---|---|---|")
    for k, info in sanity.items():
        r0 = info["rbodymod0"]; r1 = info["rbodymod1"]
        md.append(f"| {k} | {info['r_body_eff_ohm']:.3g} Ω | "
                    f"{r0['Vb']:.4f} → {r1['Vb']:.4f} | "
                    f"{r0['Id']:.3e} → {r1['Id']:.3e} | "
                    f"{info['dVb_mV']:+.2f} |")
    md.append("\nTask gate: ΔVb < 50 mV at literal card R_body.\n")

    md.append("## Backward-basin probe (VG1=0.6 VG2=0.2)\n")
    md.append("| tag | Vd | Vb | Id |\n|---|---|---|---|")
    for tag, info in basin_results.items():
        for i, vd in enumerate(info["Vd"]):
            md.append(f"| {tag} | {vd:.2f} | {info['Vb'][i]:.3f} | "
                       f"{info['Id'][i]:.3e} |")
    md.append("")

    md.append("## Full 33-curve sweep results\n")
    md.append(f"Baseline (z432, rbodymod=0): fwd={BASE_FWD:.3f}, "
                f"bwd={BASE_BWD:.3f}, avg={BASE_AVG:.3f}, asym={BASE_ASYM:.3f} dec.\n")
    md.append("| tag | R_body [Ω] | fwd | bwd | avg | asym | Δavg | Δfwd | wall [s] |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for k, info in all_runs.items():
        dfwd = info["fwd_cell_rmse"] - BASE_FWD
        dav = info["dc_avg"] - BASE_AVG
        md.append(f"| {k} | {info['r_body_eff_ohm']:.3g} | "
                    f"{info['fwd_cell_rmse']:.3f} | {info['bwd_cell_rmse']:.3f} | "
                    f"{info['dc_avg']:.3f} | {info['fwd_bwd_asym']:.3f} | "
                    f"{dav:+.3f} | {dfwd:+.3f} | {info['wall_sec']:.0f} |")
    md.append("")
    md.append(f"**Best avg**: {best_tag} (avg={best['dc_avg']:.3f}).\n")

    md.append("## Gate verdict\n")
    for k, v in gates.items():
        md.append(f"- **{k}**: {'PASS' if v else 'FAIL'}")
    md.append("")

    md.append("## Honest caveats / remaining work\n")
    md.append("1. **Full 5-R network NOT implemented**: only the equivalent")
    md.append("   single resistor between intrinsic body and external body pin.")
    md.append("   This loses the dbi/sbi sub-node dynamics — fine for DC, wrong")
    md.append("   for HF/RF.")
    md.append("2. **External body pin is GROUNDED**: we assume Vbp=0 (substrate).")
    md.append("   For a true floating-body NSRAM, Vbp should be a free node tied")
    md.append("   to the substrate through the DNW diode. That would couple")
    md.append("   rbodymod with the existing `use_well_diode` path.")
    md.append("3. **Card R values may not be width-scaled**: BSIM4 spec says")
    md.append("   RBPS etc. are nominal Ω, not Ω·µm, when rbodymod=1. We use")
    md.append("   them literally. If width scaling is needed (e.g., R∝1/W), the")
    md.append("   effective resistance changes by ~10× for our 0.13 µm device.")
    md.append("4. **No `geomod` accounting**: BSIM4 §6.7 includes a geomod")
    md.append("   correction we did not apply.")

    (OUT / "honest_analysis.md").write_text("\n".join(md))
    log(f"  wrote honest_analysis.md")
    log(f"Total wall: {time.time()-t_main:.0f}s")


if __name__ == "__main__":
    main()
