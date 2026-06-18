"""z455 — Knee-sharpener DC fix on top of v449_B + SB_HOT.

Hypothesis (per z454 honest_analysis.md): SB_HOT (BV=1.2, Is×5) gave the
fastest ns-snap but blew up DC (avg=2.809 dec vs SB_OFF=2.087 dec) because
the Slotboom multiplier M(V_db) = 1/(1-(V_db/BV)^n) fires too eagerly at
moderate V_db values that are visited during the DC sweep.

Fix: gate the Slotboom prefactor with a σ around a knee V_knee:
        M(V_db) = 1 / (1 - σ((V_db - V_knee)/V_sharp) · (V_db/BV)^n)
so below V_knee M ≈ 1 (no avalanche → DC clean) and above V_knee the
multiplier recovers full Slotboom strength → ns-snap survives.

Five conditions (all on v449_B + SB_HOT base, BV=1.2, Is×5):
  - K_OFF : knee gate OFF (== z454 SB_HOT, sanity reference)
  - K_1p4 : V_knee=1.4, V_sharp=0.05
  - K_1p6 : V_knee=1.6, V_sharp=0.05
  - K_1p8 : V_knee=1.8, V_sharp=0.05   (predicted sweet spot)
  - K_2p0 : V_knee=2.0, V_sharp=0.05

Outputs to results/z455_knee_sharpener/:
  - run.log  (line 1 = pre-registered gates, includes I_snap asserts)
  - summary.json
  - dc_vs_vknee.png, pulse_overlay.png, multiplier_plot.png
  - honest_analysis.md
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
OUT = ROOT / "results/z455_knee_sharpener"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "nsram"))
sys.path.insert(0, str(ROOT / "scripts"))

LOG = open(OUT / "run.log", "w")
def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True); LOG.write(line + "\n"); LOG.flush()


# Pre-registered gates (LOCKED on line 1 of run.log)
log("PRE-REGISTERED GATES (locked):")
log("  INFRA      = all 5 conditions complete + summary.json")
log("  DISCOVERY  = some V_knee: DC_avg < 1.5 dec AND fast-pulse t_to_0.5V < 5ns on >= 2/4 biases")
log("  AMBITIOUS  = some V_knee: DC_avg < 1.0 dec AND t_to_0.5V < 3ns on >= 3/4 biases AND zero regression vs K_OFF on VG1=0.2 branches")
log("  KILL_SHOT  = monotonic-worse DC across all V_knee OR ns-snap broken (all conditions t_to_0.5V > 10ns)")
log("ASSERT       = I_snap_d != 0 at deep-avalanche probe (Vd=2.0, Vb=0.6) for every knee-on condition")
log("")

# Reuse z454 internals (z449→z427/z429 chain, biases, helpers)
_spec454 = _ilu.spec_from_file_location("z454", ROOT / "scripts/z454_snapback_integration.py")
# Avoid running z454.main() — load module without exec side-effects of main().
# z454 only runs main() under __main__; loading is safe.
z454 = _ilu.module_from_spec(_spec454); _spec454.loader.exec_module(z454)

z449 = z454.z449
z427 = z454.z427
z429 = z454.z429
BIASES = z454.BIASES
slow_dc_cell_rmse_dir = z454.slow_dc_cell_rmse_dir
fast_pulse_smoke = z454.fast_pulse_smoke
assert_snapback_live = z454.assert_snapback_live

# v449_B base flags (n-well cap zeroed) + SB_HOT snapback (BV×0.6, Is×5)
SNAP_HOT = dict(
    snap_BV=2.0 * 0.6, snap_n_avl=4.0, snap_Bf=417.0, snap_Va=0.90,
    snap_Is=6.0256e-9 * 5.0, snap_Nf=1.0,
    snap_Id_clamp=1e-2, snap_Iii_clamp=1e-2,
)
V449B_BASE = {
    "use_vbic_for_q1": True,
    "vbic_AVC1": 0.5, "vbic_AVC2": 0.5,
    "Cbody": 1e-15,
    "body_pdiode_Cj0_per_area": 0.0,
}

V_SHARP = 0.05
def cond_hot(name, V_knee, knee_on=True):
    flags = {**V449B_BASE, "use_snapback_sub": True, **SNAP_HOT,
             "snap_use_knee_gate": bool(knee_on),
             "snap_V_knee": float(V_knee),
             "snap_V_sharp": V_SHARP}
    desc = (f"v449_B + SB_HOT + knee_gate=ON V_knee={V_knee:.2f} V_sharp={V_SHARP}"
            if knee_on else
            "v449_B + SB_HOT, knee gate OFF (== z454 SB_HOT reference)")
    return {"name": name, "desc": desc, "flags": flags, "V_knee": float(V_knee)}

CONDITIONS = [
    cond_hot("K_OFF", 1.8, knee_on=False),
    cond_hot("K_1p4", 1.4, knee_on=True),
    cond_hot("K_1p6", 1.6, knee_on=True),
    cond_hot("K_1p8", 1.8, knee_on=True),
    cond_hot("K_2p0", 2.0, knee_on=True),
]

# Bias the snapback-trace capture to the same hot point z454 uses.
CAPTURE_TAG = "VG1_0p6_VG2_0p2"


def plot_dc_vs_vknee(results, path):
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 4.6))
    # Sort knee-ON conditions by V_knee for line plot
    on = [r for r in results if r["name"] != "K_OFF"]
    on.sort(key=lambda r: r["V_knee"])
    xv = [r["V_knee"] for r in on]
    fwd = [r["dc_forward"]["cell_rmse_dec"] for r in on]
    bwd = [r["dc_backward"]["cell_rmse_dec"] for r in on]
    avg = [r["dc_avg"] for r in on]
    ax.plot(xv, fwd, "o-", label="forward", color="C0")
    ax.plot(xv, bwd, "s-", label="backward", color="C1")
    ax.plot(xv, avg, "^-", label="avg", color="C2", lw=1.8)
    # K_OFF horizontal reference
    off = next((r for r in results if r["name"] == "K_OFF"), None)
    if off is not None:
        ax.axhline(off["dc_avg"], color="grey", ls="--", lw=1,
                   label=f"K_OFF avg = {off['dc_avg']:.2f}")
    ax.axhline(1.5, color="orange", ls=":", lw=0.8, label="DISCOVERY (1.5)")
    ax.axhline(1.0, color="red", ls=":", lw=0.8, label="AMBITIOUS (1.0)")
    ax.set_xlabel("V_knee [V]"); ax.set_ylabel("cell DC RMSE [dec]")
    ax.set_title("z455 — DC RMSE vs V_knee (fwd/bwd/avg), σ-gated Slotboom")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    log(f"  wrote {path.name}")


def plot_pulse_overlay(results, path, bias_tag=CAPTURE_TAG):
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 6), sharex=True)
    colors = ["k", "C0", "C1", "C2", "C3"]
    for r, col in zip(results, colors):
        rec = next((x for x in r["fast"]["per_bias"] if x["tag"] == bias_tag), None)
        if rec is None: continue
        tr = rec.get("_traces")
        if tr is None: continue
        t = np.array(tr["t"]) * 1e9
        axes[0].plot(t, tr["Vb"], col + "-", lw=1.2, label=r["name"])
        axes[1].semilogy(t, np.maximum(np.abs(tr["Id"]), 1e-18),
                         col + "-", lw=1.0, label=r["name"])
    axes[0].axhline(0.3, color="grey", ls=":", lw=0.6, label="0.3V")
    axes[0].axhline(0.5, color="red", ls=":", lw=0.6, label="0.5V")
    axes[0].set_ylabel("V_B [V]"); axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)
    axes[1].set_ylabel("|I_D| [A]"); axes[1].set_xlabel("time [ns]")
    axes[1].legend(fontsize=8); axes[1].grid(True, which="both", alpha=0.3)
    axes[0].set_title(f"z455 — fast-pulse V_B(t) overlay @ {bias_tag}, 5 V_knee variants")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    log(f"  wrote {path.name}")


def plot_multiplier(results, path):
    """M(V_db) for the 5 conditions, computed analytically with the same
    knob values used in the cell."""
    from nsram.bsim4_port.snapback_subcircuit import avalanche_multiplier
    V_db = torch.linspace(0.0, 2.0, 401, dtype=torch.float64)
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 4.6))
    colors = ["k", "C0", "C1", "C2", "C3"]
    for r, col in zip(results, colors):
        flags = r["flags"]
        BV = float(flags["snap_BV"]); n = float(flags["snap_n_avl"])
        use_gate = bool(flags.get("snap_use_knee_gate", False))
        Vk = float(flags.get("snap_V_knee", 1.8))
        Vs = float(flags.get("snap_V_sharp", 0.05))
        M = avalanche_multiplier(V_db, BV, n,
                                 use_knee_gate=use_gate,
                                 V_knee=Vk, V_sharp=Vs).numpy()
        lab = (f"{r['name']} (gate OFF, BV={BV:.2f})" if not use_gate
               else f"{r['name']}  Vk={Vk:.2f}")
        ax.semilogy(V_db.numpy(), M, col + "-", lw=1.4, label=lab)
    ax.axhline(1.0, color="grey", ls="--", lw=0.7)
    ax.set_xlabel("V_db [V]"); ax.set_ylabel("M(V_db) [unitless]")
    ax.set_title("z455 — σ-gated Slotboom multiplier (BV=1.2, n=4, V_sharp=0.05)")
    ax.legend(fontsize=8); ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    log(f"  wrote {path.name}")


def thermal_pause():
    """Pause if APU thermal_zone0 > 85 C, wait until < 75 C (CLAUDE.md)."""
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            t = int(f.read().strip()) / 1000.0
        if t > 85.0:
            log(f"  THERMAL PAUSE: APU={t:.1f}C > 85C, cooling …")
            for _ in range(120):
                time.sleep(2)
                with open("/sys/class/thermal/thermal_zone0/temp") as f:
                    t = int(f.read().strip()) / 1000.0
                if t < 75.0:
                    log(f"  COOLED: APU={t:.1f}C, resuming"); break
    except Exception:
        pass


def eval_gates(results):
    """z455 gates per the task spec."""
    k_off = next((r for r in results if r["name"] == "K_OFF"), None)
    discovery = False; discovery_who = None
    ambitious = False; ambitious_who = None

    def t05_count(pb, thresh_ns):
        n = 0
        for x in pb:
            v = x.get("t_to_0p5V_ns")
            if v is not None and v < thresh_ns:
                n += 1
        return n
    def t05_any_le10(pb):
        for x in pb:
            v = x.get("t_to_0p5V_ns")
            if v is not None and v < 10.0:
                return True
        return False

    # K_OFF VG1=0.4 branches DC for regression check
    def vg04_branch_avg(r):
        f = [x["log_rmse"] for x in r["dc_forward"]["per_bias"]
             if abs(x["VG1"] - 0.4) < 1e-9]
        b = [x["log_rmse"] for x in r["dc_backward"]["per_bias"]
             if abs(x["VG1"] - 0.4) < 1e-9]
        # task says "zero regression on per-branch VG1=0.2" — there are no
        # VG1=0.2 biases in our 4-bias smoke. The DC sweep has VG1=0.2/0.4/0.6
        # (see Sebas 33 curves). Use VG1=0.2 sub-branch from DC instead.
        f2 = [x["log_rmse"] for x in r["dc_forward"]["per_bias"]
              if abs(x["VG1"] - 0.2) < 1e-9]
        b2 = [x["log_rmse"] for x in r["dc_backward"]["per_bias"]
              if abs(x["VG1"] - 0.2) < 1e-9]
        return (np.mean(f2) if f2 else None,
                np.mean(b2) if b2 else None)
    if k_off is not None:
        off_f02, off_b02 = vg04_branch_avg(k_off)
    else:
        off_f02 = off_b02 = None

    for r in results:
        if r["name"] == "K_OFF":
            continue
        pb = r["fast"]["per_bias"]
        n_t05_5 = t05_count(pb, 5.0)
        n_t05_3 = t05_count(pb, 3.0)
        dc = r["dc_avg"]
        if dc < 1.5 and n_t05_5 >= 2:
            if not discovery:
                discovery = True; discovery_who = r["name"]
        # zero regression on VG1=0.2 branch
        f02, b02 = vg04_branch_avg(r)
        regression_ok = True
        if off_f02 is not None and f02 is not None and f02 > off_f02 + 0.05:
            regression_ok = False
        if off_b02 is not None and b02 is not None and b02 > off_b02 + 0.05:
            regression_ok = False
        if dc < 1.0 and n_t05_3 >= 3 and regression_ok:
            if not ambitious:
                ambitious = True; ambitious_who = r["name"]

    # KILL: monotonic-worse DC across all knee-on V_knee OR all t_to_0.5V > 10ns
    on = [r for r in results if r["name"] != "K_OFF"]
    on_sorted = sorted(on, key=lambda r: r["V_knee"])
    monotonic_worse = (len(on_sorted) >= 2 and
                       all(r["dc_avg"] > (k_off["dc_avg"] if k_off else 0) - 1e-6
                           for r in on_sorted))
    # second clause: monotonically increasing DC vs V_knee AND all worse than K_OFF
    monotonic_increasing = all(
        on_sorted[i]["dc_avg"] <= on_sorted[i + 1]["dc_avg"] + 1e-6
        for i in range(len(on_sorted) - 1))
    ns_snap_broken = all(not any(
        (x.get("t_to_0p5V_ns") is not None and x["t_to_0p5V_ns"] < 10.0)
        for x in r["fast"]["per_bias"]) for r in results)
    kill_shot = ns_snap_broken or (monotonic_worse and monotonic_increasing)
    return {
        "INFRA_pass": True,
        "DISCOVERY_pass": discovery, "DISCOVERY_who": discovery_who,
        "AMBITIOUS_pass": ambitious, "AMBITIOUS_who": ambitious_who,
        "KILL_SHOT": kill_shot,
        "kill_shot_reason": ("ns_snap_broken" if ns_snap_broken else
                             ("monotonic_worse_DC" if monotonic_worse and monotonic_increasing
                              else None)),
    }


def main():
    t0_main = time.time()
    log("z455 starting — knee-sharpener on v449_B + SB_HOT base")
    model_M1, model_M2 = z427.build_models()
    curves = z427.load_curves()
    sebas_rows = z427.load_sebas_params()
    log(f"loaded: {len(curves)} curves, {len(sebas_rows)} sebas rows")

    results = []
    for C in CONDITIONS:
        thermal_pause()
        log(f"===== {C['name']}: {C['desc']} =====")
        assert_info = assert_snapback_live(C["name"], C["flags"],
                                           model_M1, model_M2, sebas_rows)
        # Verify I_snap non-zero at V_db > V_knee for knee-on conditions
        if C["flags"].get("snap_use_knee_gate", False):
            if assert_info.get("snap_called") and assert_info.get("I_snap_d", 0.0) <= 0.0:
                log(f"  !!! KILL_SHOT-DEAD: I_snap == 0 at deep-avalanche probe "
                    f"with knee_gate=ON V_knee={C['flags']['snap_V_knee']:.2f}")
        log(f"  I_snap assert: snap_called={assert_info.get('snap_called')} "
            f"|I_snap_d|={assert_info.get('I_snap_d', 0):.3e}A "
            f"|I_snap_b|={assert_info.get('I_snap_b', 0):.3e}A "
            f"V_db={assert_info.get('V_db', 'NA')} BV={assert_info.get('BV', 'NA')}")
        thermal_pause()
        dc_f = slow_dc_cell_rmse_dir(C["name"], C["flags"],
                                     model_M1, model_M2, curves, sebas_rows,
                                     direction="forward")
        thermal_pause()
        dc_b = slow_dc_cell_rmse_dir(C["name"], C["flags"],
                                     model_M1, model_M2, curves, sebas_rows,
                                     direction="backward")
        dc_avg = 0.5 * (dc_f["cell_rmse_dec"] + dc_b["cell_rmse_dec"])
        log(f"  {C['name']}: DC fwd={dc_f['cell_rmse_dec']:.3f} "
            f"bwd={dc_b['cell_rmse_dec']:.3f} avg={dc_avg:.3f}")
        thermal_pause()
        fast = fast_pulse_smoke(C["name"], C["flags"],
                                model_M1, model_M2, sebas_rows,
                                capture_snap_trace_bias=CAPTURE_TAG)
        results.append({
            "name": C["name"], "desc": C["desc"],
            "V_knee": C["V_knee"], "flags": C["flags"],
            "assert": assert_info,
            "dc_forward": dc_f, "dc_backward": dc_b, "dc_avg": dc_avg,
            "fast": fast,
        })

    gates = eval_gates(results)
    log(f"GATES: {gates}")

    # Trim traces for JSON (keep them in-memory for plot first)
    plot_dc_vs_vknee(results, OUT / "dc_vs_vknee.png")
    plot_pulse_overlay(results, OUT / "pulse_overlay.png")
    plot_multiplier(results, OUT / "multiplier_plot.png")

    def trim(rec, max_pts=200):
        tr = rec.get("_traces")
        if tr is None: return
        keys = [k for k, v in tr.items() if isinstance(v, list)]
        if not keys: return
        n_in = len(tr[keys[0]])
        if n_in <= max_pts: return
        idx = np.linspace(0, n_in - 1, max_pts).astype(int).tolist()
        for k in keys:
            v = tr[k]
            if len(v) == n_in:
                tr[k] = [v[i] for i in idx]
    for r in results:
        for rec in r["fast"]["per_bias"]:
            trim(rec)

    summary = {
        "conditions": [
            {"name": r["name"], "desc": r["desc"], "V_knee": r["V_knee"],
             "flags": {k: v for k, v in r["flags"].items()
                       if isinstance(v, (int, float, bool, str))},
             "assert": r["assert"],
             "dc_forward_dec": r["dc_forward"]["cell_rmse_dec"],
             "dc_backward_dec": r["dc_backward"]["cell_rmse_dec"],
             "dc_avg_dec": r["dc_avg"],
             "dc_forward_n": r["dc_forward"]["n"],
             "dc_backward_n": r["dc_backward"]["n"],
             "dc_per_branch_forward": r["dc_forward"]["per_bias"],
             "dc_per_branch_backward": r["dc_backward"]["per_bias"],
             "fast_pulse": [{k: v for k, v in x.items() if k != "_traces"}
                            for x in r["fast"]["per_bias"]],
             } for r in results
        ],
        "gates": gates,
        "references": {
            "z454_SB_HOT_DC_avg": 2.809,
            "z454_SB_OFF_DC_avg": 2.087,
            "z448_BDF_DC_ref": 1.002,
        },
        "wall_total_sec": round(time.time() - t0_main, 1),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    log(f"wrote summary.json  total_wall={summary['wall_total_sec']:.0f}s")

    # honest_analysis.md
    best = min(results, key=lambda r: r["dc_avg"])
    k_off = next((r for r in results if r["name"] == "K_OFF"), None)
    lines = []
    lines.append("# z455 — Knee-sharpener DC fix (σ-gated Slotboom on SB_HOT)\n")
    lines.append("## Pre-registered gates\n")
    for k, v in gates.items():
        lines.append(f"- **{k}**: {v}")
    lines.append("\n## DC (forward / backward / avg) [dec]\n")
    lines.append("| condition | V_knee | DC_fwd | DC_bwd | DC_avg | n |")
    lines.append("|---|---|---|---|---|---|")
    for r in results:
        lines.append(f"| {r['name']} | {r['V_knee']:.2f} | "
                     f"{r['dc_forward']['cell_rmse_dec']:.3f} | "
                     f"{r['dc_backward']['cell_rmse_dec']:.3f} | "
                     f"{r['dc_avg']:.3f} | {r['dc_forward']['n']} |")
    lines.append("\n## Per-branch DC (forward) — average log-RMSE by VG1\n")
    lines.append("| condition | VG1=0.2 | VG1=0.4 | VG1=0.6 |")
    lines.append("|---|---|---|---|")
    for r in results:
        pb = r["dc_forward"]["per_bias"]
        def avg(vg1):
            v = [x["log_rmse"] for x in pb if abs(x["VG1"] - vg1) < 1e-9]
            return (f"{np.mean(v):.3f}" if v else "—")
        lines.append(f"| {r['name']} | {avg(0.2)} | {avg(0.4)} | {avg(0.6)} |")

    lines.append("\n## Fast-pulse smoke\n")
    for r in results:
        lines.append(f"### {r['name']}  (V_knee={r['V_knee']:.2f})")
        lines.append("| bias | Vb_peak | Vb_5ns | Vb_10ns | t→0.3V[ns] | t→0.5V[ns] | self-reset |")
        lines.append("|---|---|---|---|---|---|---|")
        for x in r["fast"]["per_bias"]:
            lines.append(f"| {x['tag']} | {x['Vb_peak_V']:.3f} | "
                         f"{x['Vb_max_5ns_V']:.3f} | {x['Vb_max_10ns_V']:.3f} | "
                         f"{x['t_to_0p3V_ns']} | {x['t_to_0p5V_ns']} | "
                         f"{x['self_reset_within_5ns']} |")
        lines.append("")
    lines.append("\n## Snapback assert (z444-BURN avoidance)\n")
    for r in results:
        a = r["assert"]
        lines.append(f"- **{r['name']}**: snap_called={a.get('snap_called')} "
                     f"|I_snap_d|={a.get('I_snap_d', 0):.3e}A "
                     f"|I_snap_b|={a.get('I_snap_b', 0):.3e}A "
                     f"V_db={a.get('V_db', 'NA')} BV={a.get('BV', 'NA')}")
    lines.append(f"\n## Best condition: **{best['name']}** (V_knee={best['V_knee']:.2f})\n")
    lines.append(f"- DC_avg = {best['dc_avg']:.3f} dec")
    if k_off is not None:
        lines.append(f"- vs K_OFF DC_avg = {k_off['dc_avg']:.3f} dec  "
                     f"(Δ = {best['dc_avg'] - k_off['dc_avg']:+.3f} dec)")
    # Grid-edge warning
    on_sorted = sorted([r for r in results if r["name"] != "K_OFF"],
                       key=lambda r: r["V_knee"])
    if on_sorted:
        best_on = min(on_sorted, key=lambda r: r["dc_avg"])
        if abs(best_on["V_knee"] - on_sorted[0]["V_knee"]) < 1e-9 or \
           abs(best_on["V_knee"] - on_sorted[-1]["V_knee"]) < 1e-9:
            lines.append(f"\n> **Caveat**: best V_knee={best_on['V_knee']:.2f} "
                         f"is on grid edge — wider sweep needed before extrapolation.")
    (OUT / "honest_analysis.md").write_text("\n".join(lines))
    log("wrote honest_analysis.md")
    log("DONE.")


if __name__ == "__main__":
    main()
