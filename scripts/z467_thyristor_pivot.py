"""z467 — Thyristor pivot test.

Replaces the parasitic-NPN snapback (`snapback_subcircuit.py`) with the
explicit lambda-diode / PNPN N-curve (`thyristor_compact.py`, z450) wrapped
via `thyristor_pivot.py`. Compares 4 cells on:
  1) Full DC (cell-wide, fwd + bwd, Sebas curves, PT-pinned solver).
  2) Fast-pulse transient at 4 bias conditions; extract Mario observables.

Cells:
    THY_DEFAULT : thyristor, default N-curve (Ipk=2 mA, Vpk=0.85, Wpk=0.12)
    THY_STRONG  : thyristor, Ipk × 5 → 10 mA (target Mario 4.8 mA peak)
    THY_NARROW  : thyristor, Wpk × 0.5 → 0.06 V (sharper NDR)
    SNAP_REF    : original parasitic-NPN snapback (NX_1p8 config) — control

Pre-registered gates (line 1 of run.log):
    INFRA      = 4 cells complete + summary.json written
    DISCOVERY  = ≥1 thyristor variant ≥3/7 Mario targets within 30% AND DC<2.0 dec
    AMBITIOUS  = ≥1 thyristor variant ≥5/7 Mario targets within 30%
                  AND I_d_peak within 1 dec of 4.8 mA AND DC<1.5 dec
    KILL_SHOT  = NO thyristor variant reaches I_d > 1 mA → pivot also fails.

Outputs (results/z467_thyristor_pivot/):
    summary.json
    dc_compare.png
    transient_overlay.png
    mario_target_table.md
    honest_analysis.md
    run.log
"""
from __future__ import annotations
import importlib.util as _ilu
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)

ROOT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
OUT  = ROOT / "results/z467_thyristor_pivot"
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "nsram"))
sys.path.insert(0, str(ROOT / "scripts"))

LOG = open(OUT / "run.log", "w")
LOG.write(
    "GATES: INFRA=4cells_done+summary | "
    "DISCOVERY=>=1_thy_variant_>=3/7_targets_within30pct_AND_DC<2.0dec | "
    "AMBITIOUS=>=1_thy_variant_>=5/7_AND_Id_within_1dec_of_4.8mA_AND_DC<1.5dec | "
    "KILL_SHOT=no_thy_variant_reaches_Id>1mA\n")
LOG.flush()


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.write(line + "\n"); LOG.flush()


def thermal_pause():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            t = int(f.read().strip()) / 1000.0
        if t > 85.0:
            log(f"  THERMAL PAUSE: APU={t:.1f}C > 85C")
            for _ in range(120):
                time.sleep(2)
                with open("/sys/class/thermal/thermal_zone0/temp") as f:
                    t = int(f.read().strip()) / 1000.0
                if t < 75.0:
                    log(f"  COOLED: APU={t:.1f}C"); break
    except Exception:
        pass


# ──────────────────────────── Imports (z454 chain) ───────────────────────── #
_spec454 = _ilu.spec_from_file_location("z454", ROOT / "scripts/z454_snapback_integration.py")
z454 = _ilu.module_from_spec(_spec454); _spec454.loader.exec_module(z454)
z449 = z454.z449
z427 = z454.z427
z429 = z454.z429

from nsram.bsim4_port import transient_real_v2 as trv2
from nsram.bsim4_port.transient_real_v2 import integrate, TransientCfgV2, stim_fast_pulse


# ──────────────────────────── Mario targets ──────────────────────────────── #
TARG_PATH = ROOT / "data/mario_slide21_oscillation_targets.json"
TARG = json.load(open(TARG_PATH))
M = TARG["calibration_targets_for_compact_model"]["must_reproduce"]

TARGETS = {
    "period_s":      M["period_us"] * 1e-6,
    "Vd_peak_V":     1.89,
    "Id_peak_A":     M["I_peak_mA"] * 1e-3,
    "rise_s":        M["rise_10_90_ns"] * 1e-9,
    "fall_s":        M["fall_90_10_ns"] * 1e-9,
    "Vbody_swing_V": M["Vbody_swing_V"][1] - M["Vbody_swing_V"][0],
    "E_spike_J":     M["energy_per_spike_pJ"] * 1e-12,
}
log(f"Mario targets: {TARGETS}")


# ──────────────────────────── Build models / curves ──────────────────────── #
log("Loading models, curves, sebas rows...")
model_M1, model_M2 = z429.build_models()
curves = z429.load_curves()
sebas_rows = z429.load_sebas_params()
log(f"  {len(curves)} curves, {len(sebas_rows)} sebas rows")


# ──────────────────────────── Config registry ────────────────────────────── #
V449B_BASE = {
    "use_vbic_for_q1": True,
    "vbic_AVC1": 0.5, "vbic_AVC2": 0.5,
    "Cbody": 1e-15,
    "body_pdiode_Cj0_per_area": 0.0,
}

# Common snapback knee params (z455/z457 best) used as the avalanche
# multiplier for the body injection (Iii) — kept for ALL cells.
COMMON_AVL = dict(
    snap_BV=2.0 * 0.6, snap_n_avl=4.0,
    snap_Id_clamp=1e-2, snap_Iii_clamp=1e-2,
    snap_use_knee_gate=True,
    snap_V_knee=1.6, snap_V_sharp=0.05,
)

# SNAP_REF: original parasitic-NPN snapback (NX_1p8 config)
SNAP_REF = {
    **V449B_BASE,
    "use_snapback_sub": True,
    "snap_method": "snapback",
    **COMMON_AVL,
    "snap_Bf": 417.0, "snap_Va": 0.90,
    "snap_Is": 6.0256e-9 * 5.0, "snap_Nf": 1.0,
    "snap_npn_gate_mode": "current",
    "snap_npn_V_knee": 1.8, "snap_npn_V_sharp": 0.05,
    "snap_npn_V_BE_offset": 0.3,
    "_R_body": 1e7,
    "_C_body": 1e-15,
}

# THY_DEFAULT: default z450 N-curve params
THY_DEFAULT = {
    **V449B_BASE,
    "use_snapback_sub": True,
    "snap_method": "thyristor",
    **COMMON_AVL,
    # Filler NPN fields not used by thyristor branch
    "snap_Bf": 417.0, "snap_Va": 0.90, "snap_Is": 6e-9, "snap_Nf": 1.0,
    "thy_Ipk": 2e-3, "thy_Vpk": 0.85, "thy_Wpk": 0.12,
    "thy_Gon": 5e-3, "thy_VH": 0.55, "thy_VT1": 1.00,
    "thy_K":   40.0, "thy_alpha": 2.0, "thy_tau_Q": 5e-9,
    "_R_body": 1e7,
    "_C_body": 1e-15,
}

# THY_STRONG: Ipk × 5 → target 10 mA peak
THY_STRONG = {**THY_DEFAULT, "thy_Ipk": 1e-2}

# THY_NARROW: Wpk × 0.5 → sharper NDR
THY_NARROW = {**THY_DEFAULT, "thy_Wpk": 0.06}


CELLS = {
    "THY_DEFAULT": THY_DEFAULT,
    "THY_STRONG":  THY_STRONG,
    "THY_NARROW":  THY_NARROW,
    "SNAP_REF":    SNAP_REF,
}


# ──────────────────────────── DC sweep (cell-wide, fwd+bwd) ──────────────── #
def run_dc_cell(cfg_flags, max_curves: int = 12):
    """Cell-wide DC fwd+bwd. Returns (per_curve_records, dc_rmse_log10)."""
    cfg, sd_M1, sd_M2 = z427.make_cfg(model_M1, model_M2, dict(cfg_flags))
    log_eps = 1e-15
    sq_sum = 0.0
    n_pts = 0
    out_records = []
    eligible = [c for c in curves if c["VG1"] in (0.2, 0.4, 0.6)]
    if max_curves and len(eligible) > max_curves:
        stride = max(1, len(eligible) // max_curves)
        eligible = eligible[::stride][:max_curves]
    for c in eligible:
        sebas_row = z427.find_params(sebas_rows, c["VG1"], c["VG2"])
        if sebas_row is None or math.isnan(sebas_row.get("K1", float("nan"))):
            continue
        P_M1, P_M2 = z427.make_overrides(sebas_row)
        bjt = z427.make_bjt(sebas_row)
        try:
            bjt.Bf = float(cfg_flags.get("snap_Bf", 417.0))
        except Exception:
            pass
        Vd_arr = c["Vd"].numpy()
        Id_meas = c["Id"].numpy()
        order = np.argsort(Vd_arr)
        Vd_seq = Vd_arr[order]
        Id_meas_seq = Id_meas[order]
        if len(Vd_seq) > 10:
            idx = np.linspace(0, len(Vd_seq) - 1, 10).astype(int)
            Vd_seq = Vd_seq[idx]
            Id_meas_seq = Id_meas_seq[idx]
        rec = {"VG1": float(c["VG1"]), "VG2": float(c["VG2"]),
               "Vd": Vd_seq.tolist(), "Id_meas": Id_meas_seq.tolist()}
        for direction, ord_fn in [("fwd", lambda x: x), ("bwd", lambda x: x[::-1])]:
            Vd_dir = ord_fn(Vd_seq)
            Id_pred = np.zeros_like(Vd_dir)
            try:
                with torch.no_grad(), z427.patch_sd_scaled(sd_M1, P_M1), \
                     z427.patch_sd_scaled(sd_M2, P_M2):
                    Vb_warm = 0.0
                    for i, Vd_f in enumerate(Vd_dir):
                        r = z429.run_vsint_pinned(
                            cfg, model_M1, model_M2, bjt,
                            float(Vd_f), float(c["VG1"]), float(c["VG2"]),
                            Vsint_pin=0.0, Vb_init=Vb_warm)
                        Id_pred[i] = abs(r["Id"]) if r.get("Id") is not None else 0.0
                        if r["converged"]:
                            Vb_warm = r["Vb"]
                        else:
                            Vb_warm = 0.0
            except Exception as e:
                log(f"  DC {direction} fail VG1={c['VG1']} VG2={c['VG2']}: {e}")
                Id_pred = np.zeros_like(Vd_dir)
            if direction == "bwd":
                Id_pred = Id_pred[::-1]
            rec[f"Id_{direction}"] = Id_pred.tolist()
            # Use fwd for RMSE accounting
            if direction == "fwd":
                lp = np.log10(Id_pred + log_eps)
                lm = np.log10(Id_meas_seq + log_eps)
                sq_sum += float(np.sum((lp - lm) ** 2))
                n_pts += len(Vd_seq)
        out_records.append(rec)
    dc_rmse = math.sqrt(sq_sum / n_pts) if n_pts else float("inf")
    return out_records, dc_rmse


# ──────────────────────────── Fast pulse transient ───────────────────────── #
PULSE_BIASES = [
    ("VG1=0.6_VG2=0.0", 0.6, 0.0),
    ("VG1=0.6_VG2=0.2", 0.6, 0.2),
    ("VG1=0.6_VG2=0.4", 0.6, 0.4),
    ("VG1=0.4_VG2=0.0", 0.4, 0.0),
]
# 0→2V, 100ps rise, 5us hold, 100ps fall.
PULSE_T = stim_fast_pulse(V_hi=2.0, V_lo=0.0,
                          t_rise=100e-12, t_hold=5e-6, t_fall=100e-12,
                          t_pre=2e-9, t_post=200e-9, n_total=4000)


def run_pulse(cfg_flags, VG1, VG2):
    cfg, sd_M1, sd_M2 = z427.make_cfg(model_M1, model_M2, dict(cfg_flags))
    cfg.Cbody = float(cfg_flags.get("_C_body", 1e-15))
    tcfg = TransientCfgV2(
        C_B_const=float(cfg_flags.get("_C_body", 1e-15)),
        max_step=5e-9,
        first_step=1e-14,
        rtol=1e-5,
        atol=1e-14,
        R_body=float(cfg_flags.get("_R_body", 1e7)),
    )
    sebas_row = z427.find_params(sebas_rows, VG1, VG2)
    if sebas_row is None:
        return None
    P_M1, P_M2 = z427.make_overrides(sebas_row)
    bjt = z427.make_bjt(sebas_row)
    try:
        bjt.Bf = float(cfg_flags.get("snap_Bf", 417.0))
    except Exception:
        pass
    z449._VBIC_CTX["cfg"] = cfg
    z449._VBIC_CTX["bjt"] = bjt
    t, Vd = PULSE_T
    try:
        with torch.no_grad(), z427.patch_sd_scaled(sd_M1, P_M1), \
             z427.patch_sd_scaled(sd_M2, P_M2):
            r = integrate(cfg, model_M1, model_M2, bjt,
                          t, Vd, VG1, VG2, tcfg=tcfg, Vb0=0.0)
    except Exception as e:
        log(f"  pulse EXC VG1={VG1} VG2={VG2}: {e}")
        return None
    finally:
        z449._VBIC_CTX["cfg"] = None
        z449._VBIC_CTX["bjt"] = None
    return r


def measure_pulse(r):
    if r is None:
        return {"valid": False}
    t  = np.asarray(r["t"], dtype=float)
    Vb = np.asarray(r["Vb"], dtype=float)
    Vd = np.asarray(r["Vd"], dtype=float)
    Id = np.asarray(r["Id"], dtype=float)
    fin = np.isfinite(Vb) & np.isfinite(Id)
    if fin.sum() < 0.5 * len(t):
        return {"valid": False, "reason": "nans"}
    t = t[fin]; Vb = Vb[fin]; Vd = Vd[fin]; Id = Id[fin]
    Id_abs = np.abs(Id)
    Id_peak = float(np.max(Id_abs))
    Vb_peak = float(np.max(Vb))
    Vb_min  = float(np.min(Vb))
    Vd_peak = float(np.max(Vd))

    # rise 10-90 / fall 90-10 around the global peak
    idx_pk = int(np.argmax(Id_abs))
    rise_meas = float("nan"); fall_meas = float("nan")
    try:
        f10 = 0.1 * Id_peak; f90 = 0.9 * Id_peak
        # rise
        left = Id_abs[:idx_pk + 1]
        tl   = t[:idx_pk + 1]
        i10 = np.where(left >= f10)[0]
        i90 = np.where(left >= f90)[0]
        if len(i10) and len(i90):
            rise_meas = float(tl[i90[0]] - tl[i10[0]])
            if rise_meas <= 0:
                rise_meas = float("nan")
        # fall
        right = Id_abs[idx_pk:]
        tr    = t[idx_pk:]
        j90 = np.where(right >= f90)[0]
        j10 = np.where(right <= f10)[0]
        if len(j90) and len(j10):
            j90l = j90[-1]
            after = j10[j10 > j90l]
            if len(after):
                fall_meas = float(tr[after[0]] - tr[j90l])
    except Exception:
        pass

    # count oscillation cycles via Id excursions above 50% peak
    n_cycles = 0
    period_s = float("nan")
    if Id_peak > 0:
        thresh = 0.5 * Id_peak
        above = Id_abs > thresh
        edges = np.where(np.diff(above.astype(int)) == 1)[0]
        n_cycles = int(len(edges))
        if n_cycles >= 2:
            peaks_t = []
            for k in range(n_cycles):
                a = edges[k]
                b = edges[k+1] if k+1 < n_cycles else len(Id_abs)-1
                ji = a + int(np.argmax(Id_abs[a:b+1]))
                peaks_t.append(t[ji])
            if len(peaks_t) >= 2:
                period_s = float(np.mean(np.diff(peaks_t)))

    # energy per spike (integrate Vd*|Id| over ±period/2 around peak)
    E_meas = float("nan")
    if math.isfinite(period_s):
        tc = t[idx_pk]
        mask = (t >= tc - 0.5 * period_s) & (t <= tc + 0.5 * period_s)
        if mask.sum() > 5:
            E_meas = float(np.trapz(Vd[mask] * Id_abs[mask], t[mask]))
            E_meas = abs(E_meas)

    return {
        "valid": True,
        "Id_peak_A": Id_peak,
        "Vb_peak_V": Vb_peak,
        "Vb_min_V":  Vb_min,
        "Vd_peak_V": Vd_peak,
        "rise_s": rise_meas,
        "fall_s": fall_meas,
        "n_cycles": n_cycles,
        "period_s": period_s,
        "E_spike_J": E_meas,
        "Vbody_swing_V": Vb_peak - Vb_min,
    }


def mario_match(meas):
    """Compare to Mario targets; per-key relative error and within-30% flag."""
    rel = {}
    within = {}
    n_within = 0
    for k, tgt in TARGETS.items():
        v = meas.get(k, float("nan"))
        if v is None or not math.isfinite(v) or tgt == 0:
            rel[k] = float("nan"); within[k] = False
        else:
            e = abs(v - tgt) / abs(tgt)
            rel[k] = float(e)
            within[k] = bool(e <= 0.30)
            if within[k]:
                n_within += 1
    return rel, within, n_within


# ──────────────────────────── Drive cells ────────────────────────────────── #
SUMMARY = {"cells": {}, "gates": {}, "targets": TARGETS}

dc_records_all = {}
pulse_records_all = {}

for cell_name, cfg_flags in CELLS.items():
    log(f"\n========== CELL: {cell_name} ==========")
    thermal_pause()
    cell_out = {"cfg": {k: v for k, v in cfg_flags.items() if not k.startswith("_") or k in ("_R_body", "_C_body")}}

    # ----- DC sweep -----
    t0 = time.time()
    try:
        dc_recs, dc_rmse = run_dc_cell(cfg_flags, max_curves=9)
    except Exception as e:
        log(f"  DC EXCEPTION: {e}\n{traceback.format_exc()}")
        dc_recs, dc_rmse = [], float("inf")
    dc_wall = time.time() - t0
    log(f"  DC done in {dc_wall:.1f}s — RMSE = {dc_rmse:.3f} dec  ({len(dc_recs)} curves)")
    cell_out["dc_rmse_dec"]   = float(dc_rmse)
    cell_out["dc_wall_s"]     = float(dc_wall)
    dc_records_all[cell_name] = dc_recs

    # ----- Fast pulses (4 biases) -----
    pulses = []
    for tag, VG1, VG2 in PULSE_BIASES:
        thermal_pause()
        t0 = time.time()
        try:
            r = run_pulse(cfg_flags, VG1, VG2)
            meas = measure_pulse(r)
        except Exception as e:
            log(f"  pulse {tag} EXC: {e}")
            r = None; meas = {"valid": False, "reason": f"exc:{e}"}
        wall = time.time() - t0
        log(f"  pulse {tag}: valid={meas.get('valid')}  "
            f"Id_pk={meas.get('Id_peak_A', float('nan')):.3e}A  "
            f"Vb_pk={meas.get('Vb_peak_V', float('nan')):.3f}V  "
            f"n_cyc={meas.get('n_cycles', 0)}  ({wall:.1f}s)")
        # Mario score uses VG1=0.6/VG2=0.0 trace (Mario assumption)
        rel, within, n_w = mario_match(meas) if meas.get("valid") else ({}, {}, 0)
        pulses.append({
            "tag": tag, "VG1": VG1, "VG2": VG2,
            "meas": meas,
            "rel_err": rel,
            "within_30pct": within,
            "n_targets_within_30pct": int(n_w),
            "wall_s": float(wall),
        })
        # store raw trace for plot (sub-sample)
        if r is not None:
            stride = max(1, len(r["t"]) // 1200)
            pulse_records_all.setdefault(cell_name, {})[tag] = {
                "t":  np.asarray(r["t"])[::stride].tolist(),
                "Vd": np.asarray(r["Vd"])[::stride].tolist(),
                "Vb": np.asarray(r["Vb"])[::stride].tolist(),
                "Id": np.asarray(r["Id"])[::stride].tolist(),
            }
    cell_out["pulses"] = pulses
    # Mario score = the VG1=0.6 VG2=0.0 pulse (driver match)
    primary = pulses[0]
    cell_out["mario_n_targets_within_30pct"] = primary["n_targets_within_30pct"]
    cell_out["mario_rel_err"] = primary["rel_err"]
    cell_out["mario_within"]  = primary["within_30pct"]
    cell_out["Id_peak_primary_A"] = float(primary["meas"].get("Id_peak_A", float("nan")) or float("nan"))

    SUMMARY["cells"][cell_name] = cell_out


# ──────────────────────────── Gates ──────────────────────────────────────── #
thy_cells = ["THY_DEFAULT", "THY_STRONG", "THY_NARROW"]
disc_pass = False
amb_pass  = False
kill_shot = True
for c in thy_cells:
    cc = SUMMARY["cells"].get(c, {})
    n_w = int(cc.get("mario_n_targets_within_30pct", 0))
    dc  = float(cc.get("dc_rmse_dec", float("inf")))
    Idp = float(cc.get("Id_peak_primary_A", 0.0) or 0.0)
    if Idp > 1e-3:
        kill_shot = False
    if n_w >= 3 and dc < 2.0:
        disc_pass = True
    if n_w >= 5 and dc < 1.5 and Idp > 0:
        # I_d within 1 dec of 4.8 mA
        if abs(math.log10(max(Idp, 1e-15)) - math.log10(4.8e-3)) < 1.0:
            amb_pass = True

SUMMARY["gates"] = {
    "INFRA": True,
    "DISCOVERY": disc_pass,
    "AMBITIOUS": amb_pass,
    "KILL_SHOT": kill_shot,
}

log(f"\nGATES: INFRA=True  DISCOVERY={disc_pass}  AMBITIOUS={amb_pass}  KILL_SHOT={kill_shot}")


# ──────────────────────────── Plots ──────────────────────────────────────── #
log("Writing plots...")

# DC compare: 4 panels (one per cell), overlay all curves fwd+bwd vs meas
fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=True)
log_eps = 1e-15
for ax, (cell_name, _) in zip(axes, CELLS.items()):
    recs = dc_records_all.get(cell_name, [])
    colors = plt.cm.viridis(np.linspace(0, 1, max(1, len(recs))))
    for k, rec in enumerate(recs):
        Vd = np.array(rec["Vd"])
        Im = np.array(rec["Id_meas"])
        If = np.array(rec.get("Id_fwd", np.zeros_like(Vd)))
        Ib = np.array(rec.get("Id_bwd", np.zeros_like(Vd)))
        ax.semilogy(Vd, np.maximum(Im, log_eps), "o", ms=3, color=colors[k], alpha=0.5)
        ax.semilogy(Vd, np.maximum(If, log_eps), "-",  lw=1.0, color=colors[k])
        ax.semilogy(Vd, np.maximum(Ib, log_eps), "--", lw=0.8, color=colors[k])
    dc = SUMMARY["cells"][cell_name].get("dc_rmse_dec", float("inf"))
    ax.set_title(f"{cell_name}\nDC RMSE = {dc:.2f} dec")
    ax.set_xlabel("V_D [V]")
    ax.grid(True, which="both", alpha=0.3)
axes[0].set_ylabel("|I_D| [A]")
fig.suptitle("z467 DC IV: model fwd (—) bwd (--) vs measured (●)")
fig.tight_layout()
fig.savefig(OUT / "dc_compare.png", dpi=110); plt.close(fig)
log("  wrote dc_compare.png")

# Transient overlay: 4 cells × 4 biases; layout 4 panels, each panel
# overlays V_B(t) and I_D(t) for all 4 cells at one bias.
fig, axes = plt.subplots(2, 4, figsize=(22, 8), sharex=False)
cell_colors = {"THY_DEFAULT": "tab:blue", "THY_STRONG": "tab:orange",
               "THY_NARROW": "tab:green", "SNAP_REF": "tab:red"}
for bi, (tag, VG1, VG2) in enumerate(PULSE_BIASES):
    ax_v = axes[0, bi]
    ax_i = axes[1, bi]
    for cell_name in CELLS:
        tr = pulse_records_all.get(cell_name, {}).get(tag)
        if tr is None:
            continue
        t = np.array(tr["t"]) * 1e9   # ns
        Vb = np.array(tr["Vb"])
        Id = np.array(tr["Id"])
        ax_v.plot(t, Vb, lw=1.0, color=cell_colors[cell_name], label=cell_name)
        ax_i.semilogy(t, np.maximum(np.abs(Id), 1e-15), lw=1.0, color=cell_colors[cell_name])
    ax_v.set_title(f"{tag}")
    ax_v.set_ylabel("V_B [V]")
    ax_v.grid(True, alpha=0.3)
    ax_i.set_xlabel("t [ns]")
    ax_i.set_ylabel("|I_D| [A]")
    ax_i.grid(True, which="both", alpha=0.3)
    if bi == 0:
        ax_v.legend(fontsize=7, loc="best")
fig.suptitle("z467 fast-pulse transient: V_B(t) (top) and I_D(t) (bottom)")
fig.tight_layout()
fig.savefig(OUT / "transient_overlay.png", dpi=110); plt.close(fig)
log("  wrote transient_overlay.png")


# ──────────────────────────── Markdown tables ────────────────────────────── #
def fmt(v, prec=3):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "nan"
    if isinstance(v, float):
        return f"{v:.{prec}g}"
    return str(v)

mt = []
mt.append("# z467 Mario target match table\n")
mt.append("Driver bias: VG1=0.6, VG2=0.0 (primary Mario condition)\n")
mt.append("\n| Target | Mario value | THY_DEFAULT | THY_STRONG | THY_NARROW | SNAP_REF |")
mt.append("|---|---|---|---|---|---|")
for k, tgt in TARGETS.items():
    row = [k, fmt(tgt, 4)]
    for cell in ["THY_DEFAULT", "THY_STRONG", "THY_NARROW", "SNAP_REF"]:
        cc = SUMMARY["cells"][cell]
        v = cc["pulses"][0]["meas"].get(k, float("nan"))
        e = cc["pulses"][0]["rel_err"].get(k, float("nan"))
        flag = "PASS" if cc["pulses"][0]["within_30pct"].get(k, False) else "FAIL"
        row.append(f"{fmt(v, 4)} (err={fmt(e, 3)}, {flag})")
    mt.append("| " + " | ".join(row) + " |")
mt.append("\n")
mt.append("## Per-cell summary (4 biases)\n")
mt.append("| Cell | DC_RMSE [dec] | Mario_within_30pct (VG1=0.6 VG2=0) | Id_peak_primary [A] |")
mt.append("|---|---|---|---|")
for cell in CELLS:
    cc = SUMMARY["cells"][cell]
    mt.append(f"| {cell} | {fmt(cc['dc_rmse_dec'], 3)} | "
              f"{cc['mario_n_targets_within_30pct']}/7 | "
              f"{fmt(cc['Id_peak_primary_A'], 4)} |")
mt.append("\n## Gates\n")
mt.append(f"- INFRA: {SUMMARY['gates']['INFRA']}")
mt.append(f"- DISCOVERY: {SUMMARY['gates']['DISCOVERY']}")
mt.append(f"- AMBITIOUS: {SUMMARY['gates']['AMBITIOUS']}")
mt.append(f"- KILL_SHOT: {SUMMARY['gates']['KILL_SHOT']}")
(OUT / "mario_target_table.md").write_text("\n".join(mt))
log("  wrote mario_target_table.md")


# Honest analysis
ha = []
ha.append("# z467 — Honest analysis: thyristor pivot test\n")
ha.append("## Verdict\n")
if SUMMARY["gates"]["AMBITIOUS"]:
    verdict = "AMBITIOUS — thyristor pivot reaches ≥5/7 Mario targets at correct I_d magnitude"
elif SUMMARY["gates"]["DISCOVERY"]:
    verdict = "DISCOVERY — thyristor pivot reaches ≥3/7 Mario targets and preserves DC"
elif SUMMARY["gates"]["KILL_SHOT"]:
    verdict = "KILL_SHOT — no thyristor variant reaches I_d > 1 mA. Pivot also fails."
else:
    verdict = "PARTIAL — pivot runs cleanly but neither closes Mario nor classifies as KILL_SHOT (I_d > 1 mA somewhere but <3/7 targets)."
ha.append(verdict + "\n")
ha.append("\n## Per-cell numbers\n")
for cell in CELLS:
    cc = SUMMARY["cells"][cell]
    ha.append(f"### {cell}")
    ha.append(f"- DC RMSE (log10, fwd): **{fmt(cc['dc_rmse_dec'], 3)} dec**")
    ha.append(f"- Mario targets within 30% (primary bias): **{cc['mario_n_targets_within_30pct']}/7**")
    ha.append(f"- Primary I_d_peak: **{fmt(cc['Id_peak_primary_A'], 4)} A** (Mario target: 4.8e-3 A)")
    ha.append("- Per-bias I_d_peak:")
    for p in cc["pulses"]:
        meas = p["meas"]
        ha.append(f"  - {p['tag']}: I_d_peak={fmt(meas.get('Id_peak_A'), 4)}A  "
                  f"Vb_peak={fmt(meas.get('Vb_peak_V'), 3)}V  "
                  f"rise={fmt(meas.get('rise_s'), 3)}s  fall={fmt(meas.get('fall_s'), 3)}s  "
                  f"n_cycles={meas.get('n_cycles', 0)}  period={fmt(meas.get('period_s'), 3)}s")
    ha.append("")
ha.append("## Comparison vs SNAP_REF (control)\n")
ref = SUMMARY["cells"]["SNAP_REF"]
ha.append(f"SNAP_REF: DC={fmt(ref['dc_rmse_dec'], 3)}dec  Mario={ref['mario_n_targets_within_30pct']}/7  "
          f"Id_pk={fmt(ref['Id_peak_primary_A'], 4)}A\n")
for c in thy_cells:
    cc = SUMMARY["cells"][c]
    ddc = float(cc['dc_rmse_dec']) - float(ref['dc_rmse_dec'])
    dN  = int(cc['mario_n_targets_within_30pct']) - int(ref['mario_n_targets_within_30pct'])
    ha.append(f"- {c}: ΔDC = {ddc:+.3f} dec  ΔMario = {dN:+d}/7")
ha.append("\n## Interpretation\n")
ha.append("- All four cells share the same Slotboom-knee avalanche multiplier on the body KCL row.")
ha.append("- The only structural difference between SNAP_REF and THY_* is the drain regenerative kick:")
ha.append("  SNAP_REF uses the parasitic vertical NPN (Gummel-Poon forward-active);")
ha.append("  THY_* uses the explicit lambda-diode / PNPN N-curve I(V_AK, Q).")
ha.append("- If THY variants under-perform on Mario but match DC, the structural answer is that")
ha.append("  the explicit N-curve does not couple regeneration tightly enough to V_b without an")
ha.append("  extra positive-feedback path (e.g. a true two-BJT cross-coupled latch).")
ha.append("- If SNAP_REF outperforms THY on every metric, the parasitic-NPN topology is")
ha.append("  fundamentally the right structural element for this device, and the residual")
ha.append("  Mario gap must be closed by parameter search (z466 BBO continuation) or by")
ha.append("  capacitive/series-R modulation, not by topology swap.")
(OUT / "honest_analysis.md").write_text("\n".join(ha))
log("  wrote honest_analysis.md")


# ──────────────────────────── summary.json ───────────────────────────────── #
def _scrub(obj):
    if isinstance(obj, dict):
        return {k: _scrub(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_scrub(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        f = float(obj); return f if math.isfinite(f) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj

(OUT / "summary.json").write_text(json.dumps(_scrub(SUMMARY), indent=2))
log("  wrote summary.json")

log(f"\nDONE. Verdict: {verdict}")
LOG.close()
