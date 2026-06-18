"""z462 — β (snap_Bf) × R_body 3x4 sweep on v449_B + SB_HOT + NX_1p8 base.

Hypothesis (per educated_guesses_2026-05-17/cheat_sheet.md item #7 + #9):
  - DNW vertical parasitic NPN β literature value is 10-20, not 10^4.
  - Our card uses β=417 (z454 SB_HOT default). Reducing β should drop the
    NPN holding current 20-200x.
  - Combined with a moderate R_body (10kΩ-100kΩ, the regime Pazos uses), the
    body leak should win over holding -> self-reset should emerge naturally.
  - Risk: lowering β may break DC fit. If so, that itself is a finding
    (NPN role topologically required).

3 x 4 = 12 cells:
  snap_Bf ∈ {20, 50, 200}       (literature, intermediate, somewhat-cool)
  R_body  ∈ {10kΩ, 30kΩ, 100kΩ, 1MΩ}

Per cell:
  - assert_snapback_live (deep V_db=1.4)
  - mid-DC I_snap_d at V_db=0.8 (gate effective?)
  - DC fwd + bwd cell-wide (PT solver, default)
  - extended fast-pulse (V_D 100ps -> 2V, hold 5us, 4 biases)
  - detect self-reset, oscillation cycles + period

Pre-registered gates (line 1 of run.log, LOCKED):
  INFRA      = 12 cells complete + summary.json
  DISCOVERY  = any cell self-reset AND DC_avg < 1.5 dec
  AMBITIOUS  = >=3 oscillation cycles AND period in [100,1000]ns AND DC_avg<1.0 dec
  AMBITIOUS+ = period within 30% of Mario 430 ns (300..560 ns) on >=1 cell
  KILL_SHOT  = β reduction breaks DC > 1.5 dec on ALL cells
               (NPN with β=10^4 is topologically required)
"""
from __future__ import annotations
import importlib.util as _ilu
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)

# Default solver per CLAUDE.md instruction
os.environ.setdefault("NSRAM_DC_SOLVER", "pt")

ROOT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
OUT = ROOT / "results/z462_beta_rbody"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "nsram"))
sys.path.insert(0, str(ROOT / "scripts"))

LOG = open(OUT / "run.log", "w")
def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True); LOG.write(line + "\n"); LOG.flush()


# ---------------- Pre-registered gates (locked) -------------------------
log("PRE-REGISTERED GATES (z462, locked):")
log("  INFRA      = 12 cells complete + summary.json written")
log("  DISCOVERY  = any (Bf, R_body) gives self-reset AND DC_avg < 1.5 dec")
log("  AMBITIOUS  = >=3 osc cycles AND period in [100,1000]ns AND DC_avg < 1.0 dec")
log("  AMBITIOUS+ = period within 300..560 ns (30%% of Mario 430ns) on >=1 cell")
log("  KILL_SHOT  = β reduction breaks DC > 1.5 dec on ALL cells")
log("  Reference: SB_OFF DC_avg=2.087; SB_HOT (Bf=417) DC_avg=2.479-2.809;")
log("             Mario period=430ns, V_body 0.5-0.7V, I_peak=4.8mA")
log("")


# ---------------- Snapshot/restore helper logs (z458 pattern) -----------
_PRESERVE = [
    ROOT / "results/z454_snapback_integration/run.log",
    ROOT / "results/z456_rbody_reset/run.log",
    ROOT / "results/z457_npn_gate/run.log",
    ROOT / "results/z458_snap_rbody_2d/run.log",
]
_snapshots = {}
for _p in _PRESERVE:
    try:
        _snapshots[_p] = _p.read_bytes() if _p.exists() else None
    except Exception:
        _snapshots[_p] = None

_spec454 = _ilu.spec_from_file_location("z454", ROOT / "scripts/z454_snapback_integration.py")
z454 = _ilu.module_from_spec(_spec454); _spec454.loader.exec_module(z454)
_spec457 = _ilu.spec_from_file_location("z457", ROOT / "scripts/z457_npn_gate.py")
z457 = _ilu.module_from_spec(_spec457); _spec457.loader.exec_module(z457)
_spec456 = _ilu.spec_from_file_location("z456", ROOT / "scripts/z456_rbody_reset.py")
z456 = _ilu.module_from_spec(_spec456); _spec456.loader.exec_module(z456)

# Mute helper module LOGs
import io as _io
for _m in (z454, z456, z457):
    try: _m.LOG.close()
    except Exception: pass
    try: _m.LOG = _io.StringIO()
    except Exception: pass
for _p, _content in _snapshots.items():
    if _content is not None:
        try: _p.write_bytes(_content)
        except Exception: pass

z449 = z454.z449
z427 = z454.z427
z429 = z454.z429
BIASES = z454.BIASES
slow_dc_cell_rmse_dir = z454.slow_dc_cell_rmse_dir
assert_snapback_live  = z454.assert_snapback_live
mid_dc_I_snap_d       = z457.mid_dc_I_snap_d

from nsram.bsim4_port.transient_real_v2 import (
    integrate, TransientCfgV2,
)


# ---------------- Base flags ---------------------------------------------
SB_HOT_BASE = dict(
    snap_BV=2.0 * 0.6,
    snap_n_avl=4.0,
    # snap_Bf set per cell
    snap_Va=0.90,
    snap_Is=6.0256e-9 * 5.0,
    snap_Nf=1.0,
    snap_Id_clamp=1e-2,
    snap_Iii_clamp=1e-2,
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
NX_1P8 = {
    "snap_npn_gate_mode": "current",
    "snap_npn_V_knee": 1.8,
    "snap_npn_V_sharp": 0.05,
    "snap_npn_V_BE_offset": 0.3,
}

BETAS = [20.0, 50.0, 200.0]
R_BODY_LIST = [
    ("R_10k",  1.0e4),
    ("R_30k",  3.0e4),
    ("R_100k", 1.0e5),
    ("R_1M",   1.0e6),
]

# Mario calibration targets
MARIO_PERIOD_NS = 430.0
MARIO_VBODY_LO  = 0.5
MARIO_VBODY_HI  = 0.7
MARIO_I_PEAK_MA = 4.8


def cell_flags(beta: float) -> dict:
    f = {**V449B_BASE, "use_snapback_sub": True, **SB_HOT_BASE,
         **SLOTBOOM_KNEE_K_1P6, **NX_1P8}
    f["snap_Bf"] = beta
    return f

def cell_name(beta: float, rb_name: str) -> str:
    return f"Bf{int(beta)}__{rb_name}"


# ---------------- Thermal pause -----------------------------------------
def thermal_pause():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            t = int(f.read().strip()) / 1000.0
        if t > 85.0:
            log(f"  THERMAL PAUSE: APU={t:.1f}C > 85C, cooling...")
            for _ in range(120):
                time.sleep(2)
                with open("/sys/class/thermal/thermal_zone0/temp") as f:
                    t = int(f.read().strip()) / 1000.0
                if t < 75.0:
                    log(f"  COOLED: APU={t:.1f}C, resuming"); break
    except Exception:
        pass


# ---------------- Fast-pulse extended (5us hold, custom) ----------------
def stim_fast_pulse_long(V_hi=2.0, V_lo=0.05, t_rise=100e-12,
                          t_hold=5.0e-6, t_pre=0.5e-9, n_total=4000):
    T = t_pre + t_rise + t_hold
    t = np.linspace(0.0, T, n_total)
    Vd = np.full_like(t, V_lo)
    for i, ti in enumerate(t):
        if ti < t_pre:
            Vd[i] = V_lo
        elif ti < t_pre + t_rise:
            Vd[i] = V_lo + (V_hi - V_lo) * (ti - t_pre) / t_rise
        else:
            Vd[i] = V_hi
    return t, Vd


def fast_pulse_5us(name, flags, R_body, model_M1, model_M2, sebas_rows):
    cfg, sd_M1, sd_M2 = z427.make_cfg(model_M1, model_M2, dict(flags))
    cfg.Cbody = 1e-15
    tcfg = TransientCfgV2(C_B_const=1e-15,
                          max_step=5e-9,
                          first_step=1e-14,
                          rtol=1e-6, atol=1e-15,
                          R_body=R_body,
                          R_body_thresh=0.0)
    per_bias = []
    z449._VBIC_CTX["cfg"] = cfg
    for bias in BIASES:
        sebas_row = z427.find_params(sebas_rows, bias["VG1"], bias["VG2"])
        if sebas_row is None or math.isnan(sebas_row.get("K1", float("nan"))):
            log(f"  skip {bias['tag']} — no Sebas params"); continue
        P_M1, P_M2 = z427.make_overrides(sebas_row)
        bjt = z427.make_bjt(sebas_row)
        z449._VBIC_CTX["bjt"] = bjt
        t, Vd_stim = stim_fast_pulse_long(V_hi=2.0, V_lo=0.05, t_rise=100e-12,
                                           t_hold=5.0e-6, t_pre=0.5e-9, n_total=4000)
        ramp_end_s = 0.5e-9 + 100e-12
        t_start = time.time()
        try:
            with torch.no_grad(), z427.patch_sd_scaled(sd_M1, P_M1), \
                 z427.patch_sd_scaled(sd_M2, P_M2):
                r = integrate(cfg, model_M1, model_M2, bjt,
                              t, Vd_stim, bias["VG1"], bias["VG2"],
                              tcfg=tcfg, Vb0=0.0)
        except Exception as e:
            log(f"  {name} FAIL fast {bias['tag']}: {e}")
            continue
        wall = time.time() - t_start
        Vb_arr = np.array(r["Vb"])
        Id_arr = np.array(r["Id"])
        t_arr = np.array(t)
        Vb_peak = float(np.nanmax(Vb_arr))
        Id_peak_A = float(np.nanmax(np.abs(Id_arr)))

        t_peak_first, t_reset, _ = z456.detect_self_reset(
            t_arr, Vb_arr, ramp_end_s, peak_thresh=0.5, reset_thresh=0.3)
        t_to_05_after_ramp = (t_peak_first - ramp_end_s) if t_peak_first else None
        t_to_reset_after_peak = (t_reset - t_peak_first) if (t_reset and t_peak_first) else None

        n_cycles, period_s, period_std, cycle_times = z456.detect_oscillation(
            t_arr, Vb_arr, ramp_end_s, peak_thresh=0.5, reset_thresh=0.3)

        # Subsample for JSON
        n_samp = min(len(t_arr), 500)
        idx = np.linspace(0, len(t_arr) - 1, n_samp).astype(int)
        per_bias.append({
            "tag": bias["tag"], "VG1": bias["VG1"], "VG2": bias["VG2"],
            "Vb_peak_V": Vb_peak,
            "Id_peak_A": Id_peak_A,
            "t_to_0p5V_after_ramp_ns": (t_to_05_after_ramp * 1e9) if t_to_05_after_ramp else None,
            "t_to_reset_after_peak_ns": (t_to_reset_after_peak * 1e9) if t_to_reset_after_peak else None,
            "self_reset": (t_reset is not None),
            "n_oscillation_cycles": n_cycles,
            "oscillation_period_ns": (period_s * 1e9) if period_s else None,
            "oscillation_period_std_ns": (period_std * 1e9) if period_std else None,
            "Vb_in_mario_range": (MARIO_VBODY_LO <= Vb_peak <= MARIO_VBODY_HI),
            "wall_sec": round(wall, 1),
            "solver_success": bool(r["solver"]["success"]),
            "_traces": {
                "t": (t_arr[idx]).tolist(),
                "Vd": (np.array(Vd_stim)[idx]).tolist(),
                "Vb": (Vb_arr[idx]).tolist(),
                "Id": (Id_arr[idx]).tolist(),
            },
        })
        log(f"  {name}/{bias['tag']}: Vb_peak={Vb_peak:.3f}V Id_peak={Id_peak_A*1e3:.3f}mA "
            f"reset={t_reset is not None} cycles={n_cycles} "
            f"period_ns={period_s*1e9 if period_s else None} wall={wall:.1f}s")
    z449._VBIC_CTX["cfg"] = None
    z449._VBIC_CTX["bjt"] = None
    return {"per_bias": per_bias}


# ---------------- Per-cell runner ---------------------------------------
def run_cell(beta, rb_tuple, model_M1, model_M2, sebas_rows, curves):
    rb_name, R_body = rb_tuple
    name = cell_name(beta, rb_name)
    flags = cell_flags(beta)
    log(f"===== {name}: Bf={beta} R_body={R_body:g} =====")

    # 1) assert snapback live (deep)
    try:
        assert_info = assert_snapback_live(name, flags, model_M1, model_M2, sebas_rows)
        log(f"  assert(deep): snap_called={assert_info.get('snap_called')} "
            f"|I_snap_d|={assert_info.get('I_snap_d', 0):.3e}A "
            f"|I_snap_b|={assert_info.get('I_snap_b', 0):.3e}A")
    except Exception as e:
        log(f"  assert FAIL: {e}")
        assert_info = {"snap_called": False, "error": str(e)}

    # 2) mid-DC probe (V_db=0.8 < knee=1.8 → ~0)
    try:
        mid = mid_dc_I_snap_d(name, flags, model_M1, model_M2, sebas_rows,
                              Vd_test=1.4, Vb_test=0.6, Vsint_test=0.0,
                              VG1=0.6, VG2=0.0)
    except Exception as e:
        log(f"  mid_dc FAIL: {e}")
        mid = None

    # 3) DC fwd + bwd
    thermal_pause()
    dc_f = slow_dc_cell_rmse_dir(name, flags, model_M1, model_M2, curves, sebas_rows,
                                 direction="forward")
    thermal_pause()
    dc_b = slow_dc_cell_rmse_dir(name, flags, model_M1, model_M2, curves, sebas_rows,
                                 direction="backward")
    dc_avg = 0.5 * (dc_f["cell_rmse_dec"] + dc_b["cell_rmse_dec"])
    log(f"  DC fwd={dc_f['cell_rmse_dec']:.3f}  bwd={dc_b['cell_rmse_dec']:.3f}  avg={dc_avg:.3f}")

    # 4) fast-pulse extended 5us
    thermal_pause()
    fast = fast_pulse_5us(name, flags, R_body, model_M1, model_M2, sebas_rows)

    return {
        "name": name, "snap_Bf": beta,
        "R_body_name": rb_name, "R_body": R_body,
        "assert": assert_info, "mid_dc_I_snap_d": mid,
        "dc_forward": dc_f, "dc_backward": dc_b, "dc_avg": dc_avg,
        "fast": fast,
    }


# ---------------- Grid extract helper -----------------------------------
def grid_extract(results, field_fn, fill=np.nan):
    M = np.full((len(BETAS), len(R_BODY_LIST)), fill, dtype=float)
    for r in results:
        i = BETAS.index(r["snap_Bf"])
        j = [k for k, (n, _) in enumerate(R_BODY_LIST) if n == r["R_body_name"]][0]
        try:
            v = field_fn(r)
            M[i, j] = v if v is not None else fill
        except Exception:
            pass
    return M


def plot_heatmap(M, title, path, cmap="viridis", fmt="{:.2f}",
                 vmin=None, vmax=None, cbar_label=""):
    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    im = ax.imshow(M, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(R_BODY_LIST)))
    ax.set_xticklabels([n for n, _ in R_BODY_LIST])
    ax.set_yticks(range(len(BETAS)))
    ax.set_yticklabels([f"Bf={int(b)}" for b in BETAS])
    ax.set_xlabel("R_body"); ax.set_ylabel("snap_Bf (β)")
    ax.set_title(title)
    mid_val = np.nanmean(M) if np.isfinite(np.nanmean(M)) else 0
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if np.isfinite(v):
                ax.text(j, i, fmt.format(v), ha="center", va="center",
                        color="white" if v > mid_val else "black", fontsize=8)
            else:
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=7, color="grey")
    cb = fig.colorbar(im, ax=ax); cb.set_label(cbar_label)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    log(f"  wrote {path.name}")


def plot_best_traces(results, path, top_n=3):
    """Top-N by (n_cycles, then close-to-Mario-period, then low DC)"""
    cands = []
    for r in results:
        for rec in r["fast"]["per_bias"]:
            ncyc = rec.get("n_oscillation_cycles", 0) or 0
            per = rec.get("oscillation_period_ns")
            mario_score = 0.0
            if per is not None:
                if 300 <= per <= 560:
                    mario_score = 2.0
                elif 100 <= per <= 1000:
                    mario_score = 1.0
            self_reset = 1.0 if rec.get("self_reset") else 0.0
            score = ncyc + mario_score + self_reset
            cands.append((score, -r["dc_avg"], r, rec))
    cands.sort(key=lambda x: (x[0], x[1]), reverse=True)
    use = cands[:top_n]
    if not use or use[0][0] == 0:
        # fallback: just show 3 lowest-DC cells at default bias
        log("  no oscillation/reset; plotting fallback (3 lowest-DC at VG1_0p6_VG2_0p2)")
        ranked = sorted(results, key=lambda r: r["dc_avg"])[:top_n]
        use = []
        for r in ranked:
            rec = next((x for x in r["fast"]["per_bias"]
                        if x["tag"] == "VG1_0p6_VG2_0p2"), None)
            if rec is not None:
                use.append((0, -r["dc_avg"], r, rec))
    if not use:
        log("  no traces to plot at all"); return
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    colors = ["C0", "C2", "C3"]
    for (sc, _, r, rec), col in zip(use, colors):
        tr = rec.get("_traces")
        if tr is None: continue
        t = np.array(tr["t"]) * 1e9
        per_str = (f"{rec.get('oscillation_period_ns'):.0f}ns"
                   if rec.get("oscillation_period_ns") else "-")
        label = (f"{r['name']}/{rec['tag']} DC={r['dc_avg']:.2f}dec "
                 f"cyc={rec.get('n_oscillation_cycles', 0)} per={per_str} "
                 f"Vb_pk={rec['Vb_peak_V']:.2f}V")
        axes[0].plot(t, tr["Vb"], col + "-", lw=1.0, label=label)
        axes[1].plot(t, np.array(tr["Id"]) * 1e3, col + "-", lw=1.0)
    axes[0].axhline(0.5, color="red", ls=":", lw=0.6, label="0.5V peak")
    axes[0].axhline(0.3, color="grey", ls=":", lw=0.6, label="0.3V reset")
    axes[0].axhspan(MARIO_VBODY_LO, MARIO_VBODY_HI, color="green", alpha=0.1,
                    label="Mario V_body 0.5-0.7V")
    axes[0].set_ylabel("V_B [V]")
    axes[0].set_title(f"z462 - top-{top_n} best cell traces (Mario period={MARIO_PERIOD_NS}ns)")
    axes[0].legend(fontsize=7, loc="best"); axes[0].grid(True, alpha=0.3)
    axes[1].axhline(MARIO_I_PEAK_MA, color="orange", ls=":", lw=0.6, label="Mario I_peak 4.8mA")
    axes[1].set_xlabel("time [ns]"); axes[1].set_ylabel("I_D [mA]")
    axes[1].grid(True, alpha=0.3); axes[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    log(f"  wrote {path.name}")


def plot_pareto(results, path):
    fig, ax = plt.subplots(figsize=(8, 5.2))
    SB_OFF_DC = 2.087
    cs_map = {20.0: "C3", 50.0: "C1", 200.0: "C0"}
    for r in results:
        dc_pen = r["dc_avg"] - SB_OFF_DC
        per_best = None
        for x in r["fast"]["per_bias"]:
            per = x.get("oscillation_period_ns")
            if per is not None and (per_best is None or abs(per - MARIO_PERIOD_NS) < abs(per_best - MARIO_PERIOD_NS)):
                per_best = per
        if per_best is None:
            # plot at y=2000 (off-chart marker) for "no oscillation"
            y = 2000.0
            marker = "x"
        else:
            y = per_best
            marker = "o"
        ax.scatter(dc_pen, y, c=cs_map[r["snap_Bf"]], s=70, marker=marker,
                   edgecolor="k", label=f"Bf={int(r['snap_Bf'])}" if r["R_body_name"] == "R_10k" else None)
        ax.annotate(r["name"], (dc_pen, y), fontsize=6, xytext=(3, 3),
                    textcoords="offset points")
    ax.axhline(MARIO_PERIOD_NS, color="green", ls="--", lw=1.0, label="Mario 430ns")
    ax.axhspan(300, 560, color="green", alpha=0.1, label="Mario ±30%")
    ax.axhspan(100, 1000, color="yellow", alpha=0.08, label="AMBITIOUS [100,1000]ns")
    ax.axvline(0.0 - SB_OFF_DC + 1.0, color="orange", ls=":", lw=0.7,
               label="DC_avg=1.0 (AMBITIOUS)")
    ax.axvline(0.0 - SB_OFF_DC + 1.5, color="red", ls=":", lw=0.7,
               label="DC_avg=1.5 (DISCOVERY/KS)")
    ax.set_xlabel("DC penalty vs SB_OFF [dec]  (DC_avg - 2.087)")
    ax.set_ylabel("best oscillation period [ns]   (x=no osc, plotted at 2000)")
    ax.set_yscale("log")
    ax.set_title("z462 — Pareto: DC cost vs oscillation period (vs Mario 430ns)")
    ax.legend(fontsize=7, loc="best"); ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    log(f"  wrote {path.name}")


# ---------------- Gate evaluation ---------------------------------------
def eval_gates(results):
    any_reset = False
    discovery = False; discovery_who = []
    ambitious = False; ambitious_who = []
    ambitious_plus = False; ambitious_plus_who = []
    all_dc_high = True
    for r in results:
        dc = r["dc_avg"]
        if dc <= 1.5:
            all_dc_high = False
        for x in r["fast"]["per_bias"]:
            ncyc = int(x.get("n_oscillation_cycles", 0) or 0)
            per  = x.get("oscillation_period_ns")
            if x.get("self_reset"):
                any_reset = True
                if dc < 1.5:
                    discovery = True
                    discovery_who.append(f"{r['name']}/{x['tag']} dc={dc:.2f}")
            if ncyc >= 3 and per is not None and 100.0 <= per <= 1000.0 and dc < 1.0:
                ambitious = True
                ambitious_who.append(
                    f"{r['name']}/{x['tag']} cycles={ncyc} per={per:.1f}ns dc={dc:.2f}")
            if per is not None and 300.0 <= per <= 560.0:
                ambitious_plus = True
                ambitious_plus_who.append(
                    f"{r['name']}/{x['tag']} per={per:.1f}ns dc={dc:.2f}")
    kill_shot = all_dc_high  # β reduction breaks DC > 1.5 dec on ALL cells
    return {
        "INFRA_pass": (len(results) == len(BETAS) * len(R_BODY_LIST)),
        "DISCOVERY_pass": discovery, "DISCOVERY_who": discovery_who,
        "AMBITIOUS_pass": ambitious, "AMBITIOUS_who": ambitious_who,
        "AMBITIOUS_PLUS_pass": ambitious_plus, "AMBITIOUS_PLUS_who": ambitious_plus_who,
        "KILL_SHOT": kill_shot,
        "kill_shot_reason": ("β reduction breaks DC > 1.5 dec on all 12 cells"
                             if kill_shot else None),
        "n_results": len(results),
        "any_self_reset": any_reset,
    }


# ---------------- JSON trim --------------------------------------------
def _trim(results):
    """Trim traces from non-best cells to keep JSON small. (Best traces
    are already in best_cell_traces.png; keep all in JSON for completeness.)"""
    return results


def _summary_blob(results, t0, gates, partial=False):
    return {
        "version": "z462",
        "partial": partial,
        "t_wall_s": time.time() - t0,
        "betas": BETAS,
        "r_body_list": [(n, r) for n, r in R_BODY_LIST],
        "mario_targets": {
            "period_ns": MARIO_PERIOD_NS,
            "Vbody_V": [MARIO_VBODY_LO, MARIO_VBODY_HI],
            "I_peak_mA": MARIO_I_PEAK_MA,
        },
        "gates": gates,
        "cells": results,
    }


# ---------------- Honest analysis MD ------------------------------------
def write_honest_md(results, gates, path):
    lines = []
    lines.append("# z462 — β × R_body sweep (lit β=10-20 vs our β=417)\n")
    lines.append("## Pre-registered gates\n")
    for k in ("INFRA_pass", "DISCOVERY_pass", "AMBITIOUS_pass",
              "AMBITIOUS_PLUS_pass", "KILL_SHOT"):
        lines.append(f"- **{k}**: {gates.get(k)}")
    if gates.get("kill_shot_reason"):
        lines.append(f"- KILL_SHOT reason: {gates['kill_shot_reason']}")
    if gates.get("DISCOVERY_who"):
        lines.append(f"- DISCOVERY who: {gates['DISCOVERY_who']}")
    if gates.get("AMBITIOUS_who"):
        lines.append(f"- AMBITIOUS who: {gates['AMBITIOUS_who']}")
    if gates.get("AMBITIOUS_PLUS_who"):
        lines.append(f"- AMBITIOUS+ who: {gates['AMBITIOUS_PLUS_who']}")
    lines.append("")
    lines.append("## DC_avg grid (rows: Bf, cols: R_body)\n")
    lines.append("| Bf | " + " | ".join(n for n, _ in R_BODY_LIST) + " |")
    lines.append("|---|" + "---|" * len(R_BODY_LIST))
    for b in BETAS:
        row = [f"{int(b)}"]
        for rb_name, _ in R_BODY_LIST:
            rec = next((r for r in results if r["snap_Bf"] == b and r["R_body_name"] == rb_name), None)
            row.append(f"{rec['dc_avg']:.2f}" if rec else "-")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("## Per-cell summary\n")
    lines.append("| cell | Bf | R_body | DC_fwd | DC_bwd | DC_avg | n_reset | best_n_cyc | best_period_ns | max_Vb_peak | max_Id_peak_mA |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in results:
        biases = r["fast"]["per_bias"]
        n_reset = sum(1 for x in biases if x.get("self_reset"))
        best_cyc = max((int(x.get("n_oscillation_cycles", 0) or 0) for x in biases), default=0)
        periods = [x.get("oscillation_period_ns") for x in biases if x.get("oscillation_period_ns")]
        best_per = f"{min(periods, key=lambda p: abs(p - MARIO_PERIOD_NS)):.0f}" if periods else "-"
        max_vb = max((x["Vb_peak_V"] for x in biases), default=0.0)
        max_id = max((x.get("Id_peak_A", 0) for x in biases), default=0.0) * 1e3
        lines.append(
            f"| {r['name']} | {int(r['snap_Bf'])} | {r['R_body']:g} | "
            f"{r['dc_forward']['cell_rmse_dec']:.3f} | "
            f"{r['dc_backward']['cell_rmse_dec']:.3f} | {r['dc_avg']:.3f} | "
            f"{n_reset}/4 | {best_cyc} | {best_per} | {max_vb:.3f} | {max_id:.2f} |"
        )
    lines.append("")
    lines.append("## Mario calibration check\n")
    lines.append(f"- target period: {MARIO_PERIOD_NS} ns")
    lines.append(f"- target V_body swing: [{MARIO_VBODY_LO}, {MARIO_VBODY_HI}] V")
    lines.append(f"- target I_peak: {MARIO_I_PEAK_MA} mA")
    in_mario_period = []
    in_mario_vb = []
    in_mario_id = []
    for r in results:
        for x in r["fast"]["per_bias"]:
            per = x.get("oscillation_period_ns")
            if per is not None and 300 <= per <= 560:
                in_mario_period.append((r["name"], x["tag"], per))
            if MARIO_VBODY_LO <= x["Vb_peak_V"] <= MARIO_VBODY_HI:
                in_mario_vb.append((r["name"], x["tag"], x["Vb_peak_V"]))
            if x.get("Id_peak_A") and 3.0e-3 <= x["Id_peak_A"] <= 7.0e-3:
                in_mario_id.append((r["name"], x["tag"], x["Id_peak_A"] * 1e3))
    lines.append(f"- cells with period in [300,560]ns: {len(in_mario_period)}: {in_mario_period[:5]}")
    lines.append(f"- cells with V_b_peak in [0.5,0.7]V: {len(in_mario_vb)}: {in_mario_vb[:5]}")
    lines.append(f"- cells with I_peak in [3,7]mA: {len(in_mario_id)}: {in_mario_id[:5]}")
    lines.append("")
    Path(path).write_text("\n".join(lines))
    log(f"  wrote {Path(path).name}")


# ---------------- Main --------------------------------------------------
def main():
    t0 = time.time()
    log("z462 starting — 3x4 (Bf x R_body) on top of v449_B+SB_HOT+NX_1p8")
    log(f"  NSRAM_DC_SOLVER={os.environ.get('NSRAM_DC_SOLVER')}")
    model_M1, model_M2 = z427.build_models()
    curves = z427.load_curves()
    sebas_rows = z427.load_sebas_params()
    log(f"loaded: {len(curves)} curves, {len(sebas_rows)} sebas rows")

    # OPTIMIZATION: DC steady-state is independent of R_body (confirmed
    # empirically in z458 — 4 R_body values gave identical DC_avg). Compute
    # DC once per β and reuse. Cuts wall ~4x.
    dc_cache = {}  # beta -> (dc_f, dc_b)
    results = []
    for beta in BETAS:
        for rb in R_BODY_LIST:
            try:
                rb_name, R_body = rb
                name = cell_name(beta, rb_name)
                flags = cell_flags(beta)
                log(f"===== {name}: Bf={beta} R_body={R_body:g} =====")
                # 1) assert + mid
                try:
                    assert_info = assert_snapback_live(name, flags, model_M1, model_M2, sebas_rows)
                    log(f"  assert(deep): snap_called={assert_info.get('snap_called')} "
                        f"|I_snap_d|={assert_info.get('I_snap_d', 0):.3e}A")
                except Exception as e:
                    log(f"  assert FAIL: {e}"); assert_info = {"snap_called": False, "error": str(e)}
                try:
                    mid = mid_dc_I_snap_d(name, flags, model_M1, model_M2, sebas_rows,
                                          Vd_test=1.4, Vb_test=0.6, Vsint_test=0.0,
                                          VG1=0.6, VG2=0.0)
                except Exception as e:
                    log(f"  mid_dc FAIL: {e}"); mid = None
                # 2) DC amortized
                if beta not in dc_cache:
                    thermal_pause()
                    dc_f_ = slow_dc_cell_rmse_dir(name, flags, model_M1, model_M2,
                                                   curves, sebas_rows, direction="forward")
                    thermal_pause()
                    dc_b_ = slow_dc_cell_rmse_dir(name, flags, model_M1, model_M2,
                                                   curves, sebas_rows, direction="backward")
                    dc_cache[beta] = (dc_f_, dc_b_)
                    log(f"  DC[β={beta}] cached: fwd={dc_f_['cell_rmse_dec']:.3f} "
                        f"bwd={dc_b_['cell_rmse_dec']:.3f}")
                dc_f, dc_b = dc_cache[beta]
                dc_avg = 0.5 * (dc_f["cell_rmse_dec"] + dc_b["cell_rmse_dec"])
                log(f"  DC fwd={dc_f['cell_rmse_dec']:.3f}  bwd={dc_b['cell_rmse_dec']:.3f}  avg={dc_avg:.3f}")
                # 3) fast-pulse 5us
                thermal_pause()
                fast = fast_pulse_5us(name, flags, R_body, model_M1, model_M2, sebas_rows)
                r = {
                    "name": name, "snap_Bf": beta,
                    "R_body_name": rb_name, "R_body": R_body,
                    "assert": assert_info, "mid_dc_I_snap_d": mid,
                    "dc_forward": dc_f, "dc_backward": dc_b, "dc_avg": dc_avg,
                    "fast": fast,
                }
            except Exception as e:
                import traceback
                log(f"  CELL FAIL {cell_name(beta, rb[0])}: {e}")
                log(traceback.format_exc())
                continue
            results.append(r)
            # checkpoint
            try:
                gates_partial = eval_gates(results)
                (OUT / "summary.json").write_text(
                    json.dumps(_summary_blob(results, t0, gates_partial, partial=True),
                               indent=2, default=float))
            except Exception as e:
                log(f"  checkpoint fail: {e}")

    gates = eval_gates(results)
    log(f"GATES: {gates}")

    # Heatmaps
    M_dc = grid_extract(results, lambda r: r["dc_avg"])
    plot_heatmap(M_dc, "z462 — DC_avg [dec] (SB_OFF=2.087; Mario range tags @ <1.5)",
                 OUT / "heatmap_dc.png", cmap="viridis",
                 fmt="{:.2f}", cbar_label="DC_avg [dec]")

    def best_period(r):
        periods = [x.get("oscillation_period_ns") for x in r["fast"]["per_bias"]
                   if x.get("oscillation_period_ns")]
        if not periods:
            return np.nan
        return min(periods, key=lambda p: abs(p - MARIO_PERIOD_NS))

    M_per = grid_extract(results, best_period)
    plot_heatmap(M_per, "z462 — best oscillation period [ns] (Mario=430)",
                 OUT / "heatmap_period.png", cmap="plasma",
                 fmt="{:.0f}", cbar_label="period [ns]")

    # Best traces + Pareto
    plot_best_traces(results, OUT / "best_cell_traces.png", top_n=3)
    plot_pareto(results, OUT / "pareto_dc_vs_period.png")

    # Honest analysis
    write_honest_md(results, gates, OUT / "honest_analysis.md")

    # Final summary
    (OUT / "summary.json").write_text(
        json.dumps(_summary_blob(results, t0, gates, partial=False),
                   indent=2, default=float))
    log(f"DONE z462 wall={time.time()-t0:.1f}s  n_results={len(results)}")


if __name__ == "__main__":
    main()
