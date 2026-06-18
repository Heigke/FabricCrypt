"""z457 — Gate the parasitic NPN collector path directly (not just Slotboom).

Diagnosis (z455 honest_analysis.md): with knee-gated Slotboom on top of SB_HOT
the avalanche-driven body charging is suppressed below V_knee, BUT the
parasitic-NPN I_snap_d still sits at its 10 mA hard ceiling for any V_db>0
because Vbe(Vb=0.6, Vs=0)=0.6 V already turns the BJT hard on. That clamp
current is dumped into the cell during DC and is the dominant DC pollutant.

Fix (z457): apply an independent σ-knee gate to the NPN collector path
itself. Two implementations, both wired through `snap_npn_gate_mode`:

  OPTION X ("current"): I_snap_d *= σ((V_db - V_knee_npn)/V_sharp_npn)
  OPTION Y ("vbe"):     V_BE_eff = V_BE - (1 - σ((V_db - V_knee_npn)/V_sharp))
                                            · V_BE_offset

Six conditions, all on top of v449_B + SB_HOT + z455 knee-gated Slotboom
(V_knee=1.6, V_sharp=0.05):

  N_OFF   :  npn_gate_mode="off"      (== z455 K_1p6 baseline)
  NX_1p4  :  npn_gate_mode="current"  V_knee_npn=1.4
  NX_1p6  :  npn_gate_mode="current"  V_knee_npn=1.6  (predicted sweet)
  NX_1p8  :  npn_gate_mode="current"  V_knee_npn=1.8
  NY_1p6  :  npn_gate_mode="vbe"      V_knee_npn=1.6, V_BE_offset=0.3
  NY_1p8  :  npn_gate_mode="vbe"      V_knee_npn=1.8, V_BE_offset=0.3

Outputs to results/z457_npn_gate/:
  run.log         (line 1 = pre-registered gates + I_snap_d mid-DC asserts)
  summary.json
  dc_compare.png, pulse_overlay.png, I_snap_d_trace.png
  honest_analysis.md
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
OUT = ROOT / "results/z457_npn_gate"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "nsram"))
sys.path.insert(0, str(ROOT / "scripts"))

LOG = open(OUT / "run.log", "w")
def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True); LOG.write(line + "\n"); LOG.flush()


# Pre-registered gates (LOCKED on line 1 of run.log)
log("PRE-REGISTERED GATES (z457, locked):")
log("  INFRA      = all 6 conditions complete + summary.json")
log("  DISCOVERY  = some cond: DC_avg < 1.8 dec AND >=2/4 biases t_to_0.5V < 5ns")
log("  AMBITIOUS  = some cond: DC_avg < 1.0 dec AND >=3/4 biases t_to_0.5V < 3ns")
log("  KILL_SHOT  = no cond drops DC below 2.5 dec (diagnosis wrong — look elsewhere)")
log("  REGRESSION = VG1=0.2 branch worsens > 0.2 dec for any cond → alert")
log("ASSERT       = I_snap_d at mid-DC (V_db≈0.8V) should be ≈ 0 for current/vbe modes")
log("")

# Reuse z454 internals (z449→z427/z429 chain, biases, helpers)
_spec454 = _ilu.spec_from_file_location("z454", ROOT / "scripts/z454_snapback_integration.py")
z454 = _ilu.module_from_spec(_spec454); _spec454.loader.exec_module(z454)

z449 = z454.z449
z427 = z454.z427
z429 = z454.z429
BIASES = z454.BIASES
slow_dc_cell_rmse_dir = z454.slow_dc_cell_rmse_dir
fast_pulse_smoke = z454.fast_pulse_smoke
assert_snapback_live = z454.assert_snapback_live
_residuals_cell = z454._residuals_cell

# v449_B base + SB_HOT + z455 knee-gated Slotboom @ V_knee=1.6
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
SLOTBOOM_KNEE_K_1P6 = {
    "snap_use_knee_gate": True,
    "snap_V_knee": 1.6,
    "snap_V_sharp": 0.05,
}

V_SHARP_NPN = 0.05
V_BE_OFFSET = 0.3

def cond(name, mode, V_knee_npn):
    flags = {**V449B_BASE, "use_snapback_sub": True, **SNAP_HOT,
             **SLOTBOOM_KNEE_K_1P6,
             "snap_npn_gate_mode": mode,
             "snap_npn_V_knee": float(V_knee_npn),
             "snap_npn_V_sharp": V_SHARP_NPN,
             "snap_npn_V_BE_offset": V_BE_OFFSET}
    if mode == "off":
        desc = ("v449_B + SB_HOT + Slotboom knee @ V_knee=1.6 "
                "(== z455 K_1p6 baseline; NPN ungated)")
    else:
        desc = (f"v449_B + SB_HOT + Slotboom knee @1.6 + NPN gate '{mode}' "
                f"@V_knee_npn={V_knee_npn:.2f}, V_sharp={V_SHARP_NPN}")
    return {"name": name, "desc": desc, "flags": flags,
            "mode": mode, "V_knee_npn": float(V_knee_npn)}


CONDITIONS = [
    cond("N_OFF",  "off",     1.6),  # V_knee_npn unused
    cond("NX_1p4", "current", 1.4),
    cond("NX_1p6", "current", 1.6),
    cond("NX_1p8", "current", 1.8),
    cond("NY_1p6", "vbe",     1.6),
    cond("NY_1p8", "vbe",     1.8),
]

CAPTURE_TAG = "VG1_0p6_VG2_0p2"


def thermal_pause():
    """Pause if APU thermal_zone0 > 85 C; wait to <75 C (CLAUDE.md)."""
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


def mid_dc_I_snap_d(name, flags, model_M1, model_M2, sebas_rows,
                    Vd_test=1.4, Vb_test=0.6, Vsint_test=0.0,
                    VG1=0.6, VG2=0.0):
    """Sample I_snap_d at a mid-DC point V_db≈0.8V (Vd=1.4, Vb=0.6).
    For the NPN gate to be doing its job, current/vbe modes must drop
    I_snap_d significantly below the 10 mA clamp here. With V_knee_npn
    >= 1.4 this point is below the knee.
    """
    if not flags.get("use_snapback_sub", False):
        return None
    cfg, sd_M1, sd_M2 = z427.make_cfg(model_M1, model_M2, dict(flags))
    sebas_row = z427.find_params(sebas_rows, VG1, VG2)
    if sebas_row is None:
        return None
    P_M1, P_M2 = z427.make_overrides(sebas_row)
    bjt = z427.make_bjt(sebas_row)
    Vd  = torch.tensor([Vd_test], dtype=torch.float64)
    VG1t = torch.tensor([VG1], dtype=torch.float64)
    VG2t = torch.tensor([VG2], dtype=torch.float64)
    Vsint_t = torch.tensor([Vsint_test], dtype=torch.float64)
    Vb_t = torch.tensor([Vb_test], dtype=torch.float64)
    with torch.no_grad(), z427.patch_sd_scaled(sd_M1, P_M1), \
         z427.patch_sd_scaled(sd_M2, P_M2):
        _, _, comp = _residuals_cell(cfg, model_M1, bjt, Vd, VG1t, VG2t,
                                     Vsint_t, Vb_t, None, None,
                                     model_M2=model_M2)
    I_d = float(comp.get("I_snap_d", torch.tensor(0.0)).abs().item())
    I_b = float(comp.get("I_snap_b", torch.tensor(0.0)).abs().item())
    V_db = Vd_test - Vb_test
    log(f"  [{name}] mid-DC I_snap_d @ V_db={V_db:.2f}V (Vd={Vd_test} Vb={Vb_test}): "
        f"|I_snap_d|={I_d:.3e}A  |I_snap_b|={I_b:.3e}A")
    return {"V_db": V_db, "Vd": Vd_test, "Vb": Vb_test,
            "I_snap_d": I_d, "I_snap_b": I_b}


def trace_I_snap_d(name, flags, model_M1, model_M2, sebas_rows,
                   VG1=0.6, VG2=0.0, Vb_fixed=0.6, Vsint_fixed=0.0,
                   n=121):
    """Compute I_snap_d as a function of V_db at fixed Vb=0.6, Vs=0.
    Returns arrays for plotting. Vd swept 0..2.0 → V_db swept -0.6..1.4."""
    if not flags.get("use_snapback_sub", False):
        return None
    cfg, sd_M1, sd_M2 = z427.make_cfg(model_M1, model_M2, dict(flags))
    sebas_row = z427.find_params(sebas_rows, VG1, VG2)
    if sebas_row is None:
        return None
    P_M1, P_M2 = z427.make_overrides(sebas_row)
    bjt = z427.make_bjt(sebas_row)
    Vd_arr = np.linspace(Vb_fixed, 2.0, n)
    I_snap_d_arr = np.zeros_like(Vd_arr)
    I_snap_b_arr = np.zeros_like(Vd_arr)
    VG1t = torch.tensor([VG1], dtype=torch.float64)
    VG2t = torch.tensor([VG2], dtype=torch.float64)
    Vsint_t = torch.tensor([Vsint_fixed], dtype=torch.float64)
    Vb_t = torch.tensor([Vb_fixed], dtype=torch.float64)
    with torch.no_grad(), z427.patch_sd_scaled(sd_M1, P_M1), \
         z427.patch_sd_scaled(sd_M2, P_M2):
        for k, vd in enumerate(Vd_arr):
            Vd = torch.tensor([float(vd)], dtype=torch.float64)
            _, _, comp = _residuals_cell(cfg, model_M1, bjt, Vd, VG1t, VG2t,
                                         Vsint_t, Vb_t, None, None,
                                         model_M2=model_M2)
            I_snap_d_arr[k] = float(comp.get("I_snap_d",
                                             torch.tensor(0.0)).abs().item())
            I_snap_b_arr[k] = float(comp.get("I_snap_b",
                                             torch.tensor(0.0)).abs().item())
    V_db_arr = Vd_arr - Vb_fixed
    return {"V_db": V_db_arr.tolist(),
            "I_snap_d": I_snap_d_arr.tolist(),
            "I_snap_b": I_snap_b_arr.tolist()}


def plot_dc_compare(results, path):
    fig, ax = plt.subplots(1, 1, figsize=(8.0, 4.8))
    names = [r["name"] for r in results]
    x = np.arange(len(names))
    w = 0.35
    fwd = [r["dc_forward"]["cell_rmse_dec"] for r in results]
    bwd = [r["dc_backward"]["cell_rmse_dec"] for r in results]
    ax.bar(x - w/2, fwd, w, label="forward", color="C0")
    ax.bar(x + w/2, bwd, w, label="backward", color="C1")
    for xi, (f, b) in enumerate(zip(fwd, bwd)):
        ax.text(xi - w/2, f + 0.03, f"{f:.2f}", ha="center", fontsize=7)
        ax.text(xi + w/2, b + 0.03, f"{b:.2f}", ha="center", fontsize=7)
    ax.axhline(1.8, color="orange", ls=":", lw=0.8, label="DISCOVERY (1.8)")
    ax.axhline(1.0, color="red", ls=":", lw=0.8, label="AMBITIOUS (1.0)")
    ax.axhline(2.5, color="grey", ls="--", lw=0.6, label="KILL (2.5)")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, fontsize=8)
    ax.set_ylabel("cell DC RMSE [dec]")
    ax.set_title("z457 — DC RMSE (fwd vs bwd), NPN-gate modes on K_1p6 base")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    log(f"  wrote {path.name}")


def plot_pulse_overlay(results, path, bias_tag=CAPTURE_TAG):
    fig, axes = plt.subplots(2, 1, figsize=(8.6, 6), sharex=True)
    colors = ["k", "C0", "C1", "C2", "C3", "C4"]
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
    axes[0].set_title(f"z457 — fast-pulse V_B(t) overlay @ {bias_tag}, 6 NPN-gate variants")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    log(f"  wrote {path.name}")


def plot_Isnap_trace(results, path):
    fig, ax = plt.subplots(1, 1, figsize=(8.0, 4.8))
    colors = ["k", "C0", "C1", "C2", "C3", "C4"]
    for r, col in zip(results, colors):
        tr = r.get("Isnap_trace")
        if tr is None: continue
        V_db = np.array(tr["V_db"])
        I = np.array(tr["I_snap_d"]) + 1e-18
        ax.semilogy(V_db, I, col + "-", lw=1.4, label=r["name"])
    ax.axvline(0.8, color="grey", ls=":", lw=0.6, label="mid-DC probe (V_db=0.8)")
    ax.axhline(1e-2, color="red", ls="--", lw=0.6, label="10 mA clamp")
    ax.set_xlabel("V_db [V]"); ax.set_ylabel("|I_snap_d| [A]")
    ax.set_title("z457 — parasitic NPN collector current I_snap_d(V_db) "
                 "at VG1=0.6, VG2=0.0, Vb=0.6")
    ax.legend(fontsize=8); ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    log(f"  wrote {path.name}")


def eval_gates(results, k_off_ref):
    """z457 pre-registered gates."""
    discovery = False; discovery_who = None
    ambitious = False; ambitious_who = None
    regression_alerts = []

    def t05_count(pb, thresh_ns):
        n = 0
        for x in pb:
            v = x.get("t_to_0p5V_ns")
            if v is not None and v < thresh_ns:
                n += 1
        return n

    def vg02_branch_avg(r):
        f = [x["log_rmse"] for x in r["dc_forward"]["per_bias"]
             if abs(x["VG1"] - 0.2) < 1e-9]
        b = [x["log_rmse"] for x in r["dc_backward"]["per_bias"]
             if abs(x["VG1"] - 0.2) < 1e-9]
        return (float(np.mean(f)) if f else None,
                float(np.mean(b)) if b else None)

    if k_off_ref is not None:
        off_f02, off_b02 = vg02_branch_avg(k_off_ref)
    else:
        off_f02 = off_b02 = None

    any_below_2p5 = False
    for r in results:
        pb = r["fast"]["per_bias"]
        n_t05_5 = t05_count(pb, 5.0)
        n_t05_3 = t05_count(pb, 3.0)
        dc = r["dc_avg"]
        if dc < 2.5:
            any_below_2p5 = True
        if dc < 1.8 and n_t05_5 >= 2:
            if not discovery:
                discovery = True; discovery_who = r["name"]
        if dc < 1.0 and n_t05_3 >= 3:
            if not ambitious:
                ambitious = True; ambitious_who = r["name"]
        # VG1=0.2 regression check
        f02, b02 = vg02_branch_avg(r)
        if off_f02 is not None and f02 is not None and f02 > off_f02 + 0.2:
            regression_alerts.append(
                {"name": r["name"], "branch": "VG1=0.2 fwd",
                 "off": off_f02, "this": f02,
                 "delta": f02 - off_f02})
        if off_b02 is not None and b02 is not None and b02 > off_b02 + 0.2:
            regression_alerts.append(
                {"name": r["name"], "branch": "VG1=0.2 bwd",
                 "off": off_b02, "this": b02,
                 "delta": b02 - off_b02})

    kill_shot = not any_below_2p5
    return {
        "INFRA_pass": True,
        "DISCOVERY_pass": discovery, "DISCOVERY_who": discovery_who,
        "AMBITIOUS_pass": ambitious, "AMBITIOUS_who": ambitious_who,
        "KILL_SHOT": kill_shot,
        "kill_shot_reason": ("no condition < 2.5 dec — NPN-gate doesn't help, "
                             "diagnosis wrong, look elsewhere (e.g. "
                             "subcircuit_drain_current, body-source diode, "
                             "Iii_body clamp itself)" if kill_shot else None),
        "VG1_0p2_regression_alerts": regression_alerts,
    }


def main():
    t0_main = time.time()
    log("z457 starting — NPN-collector σ-gate on top of z455 K_1p6 base")
    model_M1, model_M2 = z427.build_models()
    curves = z427.load_curves()
    sebas_rows = z427.load_sebas_params()
    log(f"loaded: {len(curves)} curves, {len(sebas_rows)} sebas rows")

    results = []
    for C in CONDITIONS:
        thermal_pause()
        log(f"===== {C['name']}: {C['desc']} =====")

        # Deep-avalanche assert (z444-BURN style, at V_db=1.4 > V_knee)
        assert_info = assert_snapback_live(C["name"], C["flags"],
                                           model_M1, model_M2, sebas_rows)
        log(f"  I_snap assert (deep, V_db=1.4): "
            f"snap_called={assert_info.get('snap_called')} "
            f"|I_snap_d|={assert_info.get('I_snap_d', 0):.3e}A "
            f"|I_snap_b|={assert_info.get('I_snap_b', 0):.3e}A")

        # MID-DC I_snap_d probe — KEY DIAGNOSTIC for this experiment.
        # V_db = 0.8V (below V_knee_npn=1.6) → current/vbe modes should yield
        # I_snap_d << 10 mA. If they don't, the gate isn't actually wired.
        mid = mid_dc_I_snap_d(C["name"], C["flags"],
                              model_M1, model_M2, sebas_rows,
                              Vd_test=1.4, Vb_test=0.6, Vsint_test=0.0,
                              VG1=0.6, VG2=0.0)
        if mid is not None and C["mode"] in ("current", "vbe"):
            if mid["I_snap_d"] >= 9e-3:
                log(f"  WARNING [{C['name']}]: I_snap_d at mid-DC is still "
                    f"≥ 9 mA — gate may not be effective at V_sharp={V_SHARP_NPN}, "
                    f"V_knee_npn={C['V_knee_npn']:.2f}. Inspect honest_analysis.md.")

        # I_snap_d(V_db) trace for plotting (single bias)
        Isnap_trace = trace_I_snap_d(C["name"], C["flags"],
                                     model_M1, model_M2, sebas_rows,
                                     VG1=0.6, VG2=0.0,
                                     Vb_fixed=0.6, Vsint_fixed=0.0,
                                     n=81)

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
            "name": C["name"], "desc": C["desc"], "mode": C["mode"],
            "V_knee_npn": C["V_knee_npn"], "flags": C["flags"],
            "assert": assert_info, "mid_dc": mid,
            "Isnap_trace": Isnap_trace,
            "dc_forward": dc_f, "dc_backward": dc_b, "dc_avg": dc_avg,
            "fast": fast,
        })

    k_off_ref = next((r for r in results if r["name"] == "N_OFF"), None)
    gates = eval_gates(results, k_off_ref)
    log(f"GATES: {gates}")

    plot_dc_compare(results, OUT / "dc_compare.png")
    plot_pulse_overlay(results, OUT / "pulse_overlay.png")
    plot_Isnap_trace(results, OUT / "I_snap_d_trace.png")

    # Trim pulse traces for JSON storage
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
            {"name": r["name"], "desc": r["desc"], "mode": r["mode"],
             "V_knee_npn": r["V_knee_npn"],
             "flags": {k: v for k, v in r["flags"].items()
                       if isinstance(v, (int, float, bool, str))},
             "assert": r["assert"],
             "mid_dc_I_snap_d": r["mid_dc"],
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
            "z455_K_1p6_DC_avg": 2.702,
            "z448_BDF_DC_ref": 1.002,
        },
        "wall_total_sec": round(time.time() - t0_main, 1),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    log(f"wrote summary.json  total_wall={summary['wall_total_sec']:.0f}s")

    # honest_analysis.md
    best = min(results, key=lambda r: r["dc_avg"])
    lines = []
    lines.append("# z457 — NPN-collector σ-gate on top of z455 K_1p6\n")
    lines.append("## Pre-registered gates\n")
    for k, v in gates.items():
        lines.append(f"- **{k}**: {v}")
    lines.append("\n## DC (forward / backward / avg) [dec]\n")
    lines.append("| condition | mode | V_knee_npn | DC_fwd | DC_bwd | DC_avg | n |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in results:
        lines.append(f"| {r['name']} | {r['mode']} | {r['V_knee_npn']:.2f} | "
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

    lines.append("\n## I_snap_d at mid-DC (V_db≈0.8V, VG1=0.6, VG2=0.0)\n")
    lines.append("| condition | mode | V_knee_npn | V_db | |I_snap_d| [A] | |I_snap_b| [A] |")
    lines.append("|---|---|---|---|---|---|")
    for r in results:
        m = r["mid_dc"]
        if m is None:
            lines.append(f"| {r['name']} | {r['mode']} | {r['V_knee_npn']:.2f} | — | — | — |")
        else:
            lines.append(f"| {r['name']} | {r['mode']} | {r['V_knee_npn']:.2f} | "
                         f"{m['V_db']:.2f} | {m['I_snap_d']:.3e} | {m['I_snap_b']:.3e} |")

    lines.append("\n## Fast-pulse smoke\n")
    for r in results:
        lines.append(f"### {r['name']}  (mode={r['mode']}, V_knee_npn={r['V_knee_npn']:.2f})")
        lines.append("| bias | Vb_peak | Vb_5ns | Vb_10ns | t→0.3V[ns] | t→0.5V[ns] | self-reset |")
        lines.append("|---|---|---|---|---|---|---|")
        for x in r["fast"]["per_bias"]:
            lines.append(f"| {x['tag']} | {x['Vb_peak_V']:.3f} | "
                         f"{x['Vb_max_5ns_V']:.3f} | {x['Vb_max_10ns_V']:.3f} | "
                         f"{x['t_to_0p3V_ns']} | {x['t_to_0p5V_ns']} | "
                         f"{x['self_reset_within_5ns']} |")
        lines.append("")

    lines.append("\n## Snapback assert (deep, V_db=1.4V)\n")
    for r in results:
        a = r["assert"]
        lines.append(f"- **{r['name']}**: snap_called={a.get('snap_called')} "
                     f"|I_snap_d|={a.get('I_snap_d', 0):.3e}A "
                     f"|I_snap_b|={a.get('I_snap_b', 0):.3e}A "
                     f"V_db={a.get('V_db', 'NA')} BV={a.get('BV', 'NA')}")

    lines.append(f"\n## Best condition: **{best['name']}** "
                 f"(mode={best['mode']}, V_knee_npn={best['V_knee_npn']:.2f})\n")
    lines.append(f"- DC_avg = {best['dc_avg']:.3f} dec")
    n_off = next((r for r in results if r["name"] == "N_OFF"), None)
    if n_off is not None:
        lines.append(f"- vs N_OFF DC_avg = {n_off['dc_avg']:.3f} dec  "
                     f"(Δ = {best['dc_avg'] - n_off['dc_avg']:+.3f} dec)")
        lines.append(f"- vs z455 K_1p6 reference = 2.702 dec  "
                     f"(Δ = {best['dc_avg'] - 2.702:+.3f} dec)")

    lines.append("\n## Gate-wiring verdict\n")
    for r in results:
        m = r["mid_dc"]
        if m is None or r["mode"] == "off":
            continue
        if m["I_snap_d"] < 1e-3:
            verdict = "EFFECTIVE (mid-DC I_snap_d < 1 mA)"
        elif m["I_snap_d"] < 9e-3:
            verdict = "PARTIAL (mid-DC I_snap_d below clamp but not negligible)"
        else:
            verdict = ("INEFFECTIVE (mid-DC I_snap_d still ≈ 10 mA clamp — "
                       "σ insufficient at chosen V_sharp; either intrinsic "
                       "Ic is large enough to overpower the gate, or wiring bug)")
        lines.append(f"- **{r['name']}** ({r['mode']}, V_knee_npn={r['V_knee_npn']:.2f}): "
                     f"|I_snap_d|_mid = {m['I_snap_d']:.3e} A → {verdict}")

    if gates["KILL_SHOT"]:
        lines.append("\n## KILL_SHOT triggered — what else could pollute DC?\n")
        lines.append("- Body-source diode `body_pdiode_*` parameters "
                     "(Cj0 already zeroed in v449_B; verify Is)")
        lines.append("- Subcircuit drain current contributions: D3 zener "
                     "(G1→B), drain–body p-diode forward leakage")
        lines.append("- VBIC IIMOD path (separate from snapback subcircuit) — "
                     "vbic_AVC1/AVC2 still active")
        lines.append("- Slotboom Iii_body clamp at 10 mA may saturate at "
                     "moderate V_db; check `snap_Iii_clamp` reduction sweep")
        lines.append("- BSIM4 native `Ids` overshoot in the DC bias range — "
                     "not snapback at all, but plain channel model")

    (OUT / "honest_analysis.md").write_text("\n".join(lines))
    log("wrote honest_analysis.md")
    log("DONE.")


if __name__ == "__main__":
    main()
