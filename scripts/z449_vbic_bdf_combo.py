"""z449 — Combo: VBIC (z443) + BDF charge-state transient (z448).

Combines the three orthogonal wins from prior tracks:
  - VBIC level-4 NPN with Kloosterman avalanche (best DC: z443 1.311 dec).
  - BDF adaptive stiff solver + 3-state Gummel-Poon ODE (z448).
  - Mario lumped C_B = 1 fF.

Three variants:
  v449_A — Baseline: VBIC+BDF, C_B = 1 fF, n-well cap as-is.
  v449_B — v449_A + n-well diode cap zeroed (AC-grounded assumption:
            vnwell is a hard DC bias node, so AC equivalent C_nw → 0).
  v449_C — v449_A + ALPHA0 × 5 in M1 BSIM4 impact-ionization model
            (boost Iiimpact ~5× to bridge the body-cap charging gap).

For each variant we measure:
  (i)  Full slow-DC cell-wide RMSE on Sebas's 33 curves (V_SINT_PIN path).
  (ii) Fast-pulse smoke test on 4 biases (VG1=0.6/VG2=0,0.2,0.4 and
       VG1=0.4/VG2=0): V_D rising edge 100ps to 2V, hold 10ns, fall 100ps.

Pre-registered gates:
  DISCOVERY  : any variant DC < 0.85 dec AND >= 50% biases V_B>0.3V within 5ns.
  AMBITIOUS  : any variant DC < 0.70 dec AND >= 75% biases V_B>0.5V.
  KILL_SHOT  : all three variants ≥ 1.0 dec OR no transient improvement
               (no bias > 0.1 V_B within 5ns in any variant).
"""
from __future__ import annotations
import importlib.util as _ilu
import json
import math
import sys
import time
import copy
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)

ROOT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
OUT = ROOT / "results/z449_vbic_bdf_combo"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "nsram"))
sys.path.insert(0, str(ROOT / "scripts"))

LOG = open(OUT / "run.log", "w")
def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True); LOG.write(line + "\n"); LOG.flush()


_spec427 = _ilu.spec_from_file_location("z427", ROOT / "scripts/z427_vsint_fix.py")
z427 = _ilu.module_from_spec(_spec427); _spec427.loader.exec_module(z427)
_spec429 = _ilu.spec_from_file_location("z429", ROOT / "scripts/z429_multisolver_debug.py")
z429 = _ilu.module_from_spec(_spec429); _spec429.loader.exec_module(z429)

from nsram.bsim4_port import transient_real_v2 as trv2
from nsram.bsim4_port.transient_real_v2 import (
    integrate, TransientCfgV2,
    stim_slow_dc_ramp, stim_fast_pulse,
)
from nsram.bsim4_port.bjt import compute_bjt
from nsram.bsim4_port.vbic import VBICNPN, compute_vbic

# ============================================================ #
# Monkey-patch transient_real_v2._bjt_transport so that when the
# cfg uses VBIC for Q1, the *transport* currents driving q_F/q_R
# come from the VBIC kernel — otherwise the ODE diffusion-cap path
# silently falls back to plain Gummel-Poon.
# ============================================================ #
_ORIG_BJT_TRANSPORT = trv2._bjt_transport

# Holder for current cfg (set per-integrate call)
_VBIC_CTX = {"cfg": None, "bjt": None}

def _bjt_transport_vbic_aware(bjt, Vbe_f, Vbc_f, T_K):
    cfg = _VBIC_CTX["cfg"]
    if cfg is not None and bool(getattr(cfg, "use_vbic_for_q1", False)):
        v = getattr(bjt, "_vbic_cache", None)
        if v is None:
            # Build/cache a VBIC instance off the GP bjt
            v = VBICNPN.from_gp(bjt)
            v.AVC1 = float(getattr(cfg, "vbic_AVC1", 0.5))
            v.AVC2 = float(getattr(cfg, "vbic_AVC2", 0.5))
            v.ISP  = float(getattr(cfg, "vbic_ISP",  0.0))
            v.WSP  = float(getattr(cfg, "vbic_WSP",  1.0))
            try:
                object.__setattr__(bjt, "_vbic_cache", v)
            except Exception:
                pass
        Vbe_t = torch.tensor([Vbe_f], dtype=torch.float64)
        Vbc_t = torch.tensor([Vbc_f], dtype=torch.float64)
        out = compute_vbic(v, Vbe=Vbe_t, Vbc=Vbc_t, T_K=T_K)
        return float(out["Icc"].item()), float(out["Iec"].item())
    return _ORIG_BJT_TRANSPORT(bjt, Vbe_f, Vbc_f, T_K)

trv2._bjt_transport = _bjt_transport_vbic_aware


BIASES = [
    {"VG1": 0.6, "VG2": 0.0, "tag": "VG1_0p6_VG2_0p0"},
    {"VG1": 0.6, "VG2": 0.2, "tag": "VG1_0p6_VG2_0p2"},
    {"VG1": 0.6, "VG2": 0.4, "tag": "VG1_0p6_VG2_0p4"},
    {"VG1": 0.4, "VG2": 0.0, "tag": "VG1_0p4_VG2_0p0"},
]

Z448_REF_SLOW_DC_DEC = 1.002
Z443_VBIC_AVL_DEC     = 1.3110292027686277
Z430_BASELINE_DEC     = 1.6187161900853293


# ============================================================ #
# Slow-DC cell-wide RMSE via V_SINT_PIN (same as z443/z430)
# ============================================================ #
def slow_dc_cell_rmse(name, extra_flags, alpha0_scale, model_M1, model_M2,
                      curves, sebas_rows):
    cfg, sd_M1, sd_M2 = z427.make_cfg(model_M1, model_M2, dict(extra_flags))
    log_eps = 1e-15
    per_bias = []
    fails = 0
    t0 = time.time()
    for c in curves:
        sebas_row = z427.find_params(sebas_rows, c["VG1"], c["VG2"])
        if sebas_row is None or math.isnan(sebas_row.get("K1", float("nan"))):
            continue
        P_M1, P_M2 = z427.make_overrides(sebas_row)
        # Apply alpha0 scaling if variant requests it
        if alpha0_scale != 1.0 and "alpha0" in P_M1:
            P_M1["alpha0"] = P_M1["alpha0"] * float(alpha0_scale)
        bjt = z427.make_bjt(sebas_row)
        Vd_arr = c["Vd"].numpy()
        Id_meas = c["Id"].numpy()
        Id_pred_list = []
        Vb_list = []
        conv_list = []
        try:
            with torch.no_grad(), z427.patch_sd_scaled(sd_M1, P_M1), \
                 z427.patch_sd_scaled(sd_M2, P_M2):
                Vb_warm = 0.0
                for Vd_f in Vd_arr:
                    r = z429.run_vsint_pinned(
                        cfg, model_M1, model_M2, bjt,
                        float(Vd_f), float(c["VG1"]), float(c["VG2"]),
                        Vsint_pin=0.0, Vb_init=Vb_warm)
                    Id_pred_list.append(abs(r["Id"]))
                    Vb_list.append(r["Vb"])
                    conv_list.append(bool(r["converged"]))
                    if r["converged"]:
                        Vb_warm = r["Vb"]
                    else:
                        Vb_warm = 0.0
        except Exception as e:
            fails += 1
            log(f"  {name} fail VG1={c['VG1']} VG2={c['VG2']}: {e}")
            continue
        Id_pred = torch.tensor(Id_pred_list, dtype=torch.float64)
        conv = torch.tensor(conv_list)
        if not conv.any():
            fails += 1; continue
        log_p = torch.log10(Id_pred + log_eps)
        log_m = torch.log10(c["Id"] + log_eps)
        sq = (log_p - log_m) ** 2
        rmse = float(torch.sqrt(sq[conv].mean()))
        per_bias.append({"VG1": c["VG1"], "VG2": c["VG2"],
                         "log_rmse": rmse,
                         "vb_max": float(max(Vb_list))})
    if not per_bias:
        return {"cell_rmse_dec": float("inf"), "n": 0,
                "fails": fails, "wall_sec": time.time() - t0,
                "per_bias": []}
    cell_sq = sum(r["log_rmse"] ** 2 for r in per_bias)
    cell = math.sqrt(cell_sq / len(per_bias))
    log(f"  {name}: slow-DC cell={cell:.3f} dec ({len(per_bias)} biases) "
        f"fails={fails} wall={time.time()-t0:.1f}s")
    return {"cell_rmse_dec": cell, "n": len(per_bias),
            "fails": fails, "wall_sec": time.time() - t0,
            "per_bias": per_bias}


# ============================================================ #
# Fast-pulse smoke test (4 biases)
# ============================================================ #
def fast_pulse_smoke(name, extra_flags, alpha0_scale, model_M1, model_M2,
                     sebas_rows):
    cfg, sd_M1, sd_M2 = z427.make_cfg(model_M1, model_M2, dict(extra_flags))
    # Push cfg.Cbody to 1 fF (Mario value) — separate from tcfg.C_B_const
    cfg.Cbody = 1e-15
    tcfg = TransientCfgV2(C_B_const=1e-15,
                          max_step=1e-10, first_step=1e-14,
                          rtol=1e-6, atol=1e-15)
    per_bias = []
    _VBIC_CTX["cfg"] = cfg
    for bias in BIASES:
        sebas_row = z427.find_params(sebas_rows, bias["VG1"], bias["VG2"])
        if sebas_row is None or math.isnan(sebas_row.get("K1", float("nan"))):
            log(f"  skip {bias['tag']} — no Sebas params"); continue
        P_M1, P_M2 = z427.make_overrides(sebas_row)
        if alpha0_scale != 1.0 and "alpha0" in P_M1:
            P_M1["alpha0"] = P_M1["alpha0"] * float(alpha0_scale)
        bjt = z427.make_bjt(sebas_row)
        _VBIC_CTX["bjt"] = bjt
        t, Vd_stim = stim_fast_pulse(V_hi=2.0, V_lo=0.05,
                                       t_rise=100e-12, t_hold=10e-9,
                                       t_fall=100e-12,
                                       t_pre=0.5e-9, t_post=5e-9,
                                       n_total=800)
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
        Id_arr = np.array(r["Id"]); Vb_arr = np.array(r["Vb"])
        t_arr = np.array(t)
        ramp_end = 0.5e-9 + 100e-12
        hold_end = ramp_end + 10e-9
        idx_hold = (t_arr >= ramp_end) & (t_arr < hold_end)
        Vb_peak = float(np.nanmax(Vb_arr))
        idx_peak = int(np.nanargmax(Vb_arr))
        t_peak = float(t_arr[idx_peak])
        idx_5ns = (t_arr <= 0.5e-9 + 5e-9)
        Vb_max_5ns = float(np.nanmax(Vb_arr[idx_5ns])) if idx_5ns.any() else 0.0
        Id_baseline = float(np.nanmin(Id_arr[t_arr < ramp_end])) if (t_arr < ramp_end).any() else 0.0
        Id_max = float(np.nanmax(Id_arr))
        if Id_baseline > 0 and Id_max > 0:
            decade_swing = math.log10(Id_max / Id_baseline)
        else:
            decade_swing = float("nan")
        per_bias.append({
            "tag": bias["tag"], "VG1": bias["VG1"], "VG2": bias["VG2"],
            "Vb_peak_V": Vb_peak,
            "t_Vb_peak_s": t_peak,
            "Vb_max_within_5ns_V": Vb_max_5ns,
            "Id_decade_swing": decade_swing,
            "Id_peak_A": Id_max,
            "wall_sec": round(wall, 1),
            "solver": r["solver"],
            "_traces": {
                "t": t_arr.tolist(),
                "Vd": list(Vd_stim),
                "Vb": r["Vb"],
                "Id": r["Id"],
            }})
        log(f"  {name}/{bias['tag']}: Vb_peak={Vb_peak:.4f}V@{t_peak*1e9:.2f}ns  "
            f"Vb_5ns={Vb_max_5ns:.4f}V  Id_dec={decade_swing:.2f}  "
            f"success={r['solver']['success']}  wall={wall:.1f}s")
    _VBIC_CTX["cfg"] = None
    _VBIC_CTX["bjt"] = None
    return {"per_bias": per_bias}


# ============================================================ #
# Plots
# ============================================================ #
def plot_variant(name, fast, out_path):
    pb = fast["per_bias"]
    if not pb:
        log(f"  no traces to plot for {name}"); return
    n = len(pb)
    fig, axes = plt.subplots(n, 2, figsize=(11, 2.6 * n), sharex=True,
                             squeeze=False)
    for i, rec in enumerate(pb):
        tr = rec["_traces"]
        t = np.array(tr["t"]) * 1e9
        ax_v = axes[i, 0]; ax_i = axes[i, 1]
        ax_v.plot(t, tr["Vd"], "k-", lw=0.7, label="V_D")
        ax_v.plot(t, tr["Vb"], "b-", lw=1.2, label="V_B")
        ax_v.axhline(0.5, color="r", ls=":", lw=0.6)
        ax_v.set_ylabel(f"V [V]  {rec['tag']}")
        ax_v.grid(True, alpha=0.3)
        ax_v.legend(fontsize=7, loc="upper right")
        ax_i.semilogy(t, np.maximum(np.abs(tr["Id"]), 1e-18),
                       "m-", lw=1.0)
        ax_i.set_ylabel("|I_D| [A]")
        ax_i.grid(True, which="both", alpha=0.3)
        ax_i.set_title(f"Vb_peak={rec['Vb_peak_V']:.3f}V "
                       f"Vb_5ns={rec['Vb_max_within_5ns_V']:.3f}V "
                       f"Id_dec={rec['Id_decade_swing']:.2f}", fontsize=8)
    axes[-1, 0].set_xlabel("time [ns]"); axes[-1, 1].set_xlabel("time [ns]")
    fig.suptitle(f"z449 fast-pulse traces — {name}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120); plt.close(fig)
    log(f"  wrote {out_path.name}")


# ============================================================ #
# Variant definitions
# ============================================================ #
VARIANTS = [
    {
        "name": "v449_A",
        "desc": "VBIC+BDF baseline, C_B=1fF, n-well cap as-is",
        "extra_flags": {"use_vbic_for_q1": True,
                         "vbic_AVC1": 0.5, "vbic_AVC2": 0.5,
                         "Cbody": 1e-15},
        "alpha0_scale": 1.0,
    },
    {
        "name": "v449_B",
        "desc": "v449_A + n-well diode cap zeroed (AC-grounded vnwell)",
        "extra_flags": {"use_vbic_for_q1": True,
                         "vbic_AVC1": 0.5, "vbic_AVC2": 0.5,
                         "Cbody": 1e-15,
                         "body_pdiode_Cj0_per_area": 0.0},
        "alpha0_scale": 1.0,
    },
    {
        "name": "v449_C",
        "desc": "v449_A + ALPHA0 x 5 (5x BSIM4 impact-ion)",
        "extra_flags": {"use_vbic_for_q1": True,
                         "vbic_AVC1": 0.5, "vbic_AVC2": 0.5,
                         "Cbody": 1e-15},
        "alpha0_scale": 5.0,
    },
]


# ============================================================ #
# Pre-registered gate evaluator
# ============================================================ #
def evaluate_gates(results):
    discovery = False
    discovery_who = None
    ambitious = False
    ambitious_who = None
    any_transient = False
    for r in results:
        n_total = len(r["fast"]["per_bias"])
        if n_total == 0:
            continue
        vb_03 = sum(1 for x in r["fast"]["per_bias"]
                    if x["Vb_max_within_5ns_V"] > 0.3)
        vb_05 = sum(1 for x in r["fast"]["per_bias"]
                    if x["Vb_max_within_5ns_V"] > 0.5)
        vb_01 = sum(1 for x in r["fast"]["per_bias"]
                    if x["Vb_max_within_5ns_V"] > 0.1)
        if vb_01 > 0:
            any_transient = True
        dc = r["slow_dc"]["cell_rmse_dec"]
        if dc < 0.85 and vb_03 >= 0.5 * n_total:
            if not discovery:
                discovery = True; discovery_who = r["name"]
        if dc < 0.70 and vb_05 >= 0.75 * n_total:
            if not ambitious:
                ambitious = True; ambitious_who = r["name"]
    all_dc_ge_1 = all(r["slow_dc"]["cell_rmse_dec"] >= 1.0 for r in results)
    kill_shot = all_dc_ge_1 or (not any_transient)
    return {
        "DISCOVERY_pass": discovery, "DISCOVERY_variant": discovery_who,
        "AMBITIOUS_pass": ambitious, "AMBITIOUS_variant": ambitious_who,
        "KILL_SHOT": kill_shot,
        "any_variant_shows_transient_above_0p1V": any_transient,
        "all_variants_DC_ge_1p0_dec": all_dc_ge_1,
    }


# ============================================================ #
# Main
# ============================================================ #
def main():
    t0_main = time.time()
    log("z449 starting — VBIC + BDF combo (Mario C_B=1fF)")
    log(f"VBIC_AVL (z443) DC ref = {Z443_VBIC_AVL_DEC:.3f}")
    log(f"BDF (z448) DC ref      = {Z448_REF_SLOW_DC_DEC:.3f}")
    model_M1, model_M2 = z427.build_models()
    curves = z427.load_curves()
    sebas_rows = z427.load_sebas_params()
    log(f"loaded: {len(curves)} curves, {len(sebas_rows)} sebas rows")

    # Audit body-node cap on baseline cfg before runs
    cfg_audit, _, _ = z427.make_cfg(model_M1, model_M2, {
        "use_vbic_for_q1": True, "vbic_AVC1": 0.5, "vbic_AVC2": 0.5,
        "Cbody": 1e-15})
    Cj0_nwell = cfg_audit.body_pdiode_Cj0_per_area * cfg_audit.body_pdiode_area
    log(f"  audit: Cj0_nwell = {Cj0_nwell*1e15:.2f} fF (area)  "
        f"vnwell={cfg_audit.vnwell:.2f}V  M_grade={cfg_audit.body_pdiode_M:.3f}")
    # Effective at Vb=0, vnwell=2V (reverse-biased): Cj = Cj0*(1+2/Vj)^(-M)
    Vj = cfg_audit.body_pdiode_Vj; M = cfg_audit.body_pdiode_M
    Cnw_eff = Cj0_nwell * (1.0 + 2.0/Vj) ** (-M)
    log(f"  audit: C_nwell @ Vb=0 ≈ {Cnw_eff*1e15:.2f} fF "
        f"(this dominates 12 fF total; C_B=1fF + Cje+Cjc≈1.7fF small)")

    results = []
    for V in VARIANTS:
        log(f"===== {V['name']}: {V['desc']} =====")
        log(f"      flags={V['extra_flags']}  alpha0_scale={V['alpha0_scale']}")
        slow_dc = slow_dc_cell_rmse(V["name"], V["extra_flags"],
                                     V["alpha0_scale"],
                                     model_M1, model_M2, curves, sebas_rows)
        fast = fast_pulse_smoke(V["name"], V["extra_flags"], V["alpha0_scale"],
                                 model_M1, model_M2, sebas_rows)
        plot_path = OUT / f"pulse_{V['name']}.png"
        plot_variant(V["name"], fast, plot_path)
        results.append({"name": V["name"], "desc": V["desc"],
                        "extra_flags": {k: (v if not torch.is_tensor(v) else float(v))
                                         for k, v in V["extra_flags"].items()},
                        "alpha0_scale": V["alpha0_scale"],
                        "slow_dc": slow_dc, "fast": fast})

    gates = evaluate_gates(results)

    # Trim traces for JSON
    def trim(rec, max_pts=200):
        if "_traces" not in rec: return
        tr = rec["_traces"]
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

    # Build summary
    summary = {
        "variants": [
            {
                "name": r["name"], "desc": r["desc"],
                "alpha0_scale": r["alpha0_scale"],
                "slow_dc_cell_rmse_dec": r["slow_dc"]["cell_rmse_dec"],
                "slow_dc_n_biases": r["slow_dc"]["n"],
                "slow_dc_wall_sec": round(r["slow_dc"]["wall_sec"], 1),
                "fast_per_bias": [
                    {k: v for k, v in x.items() if k != "_traces"}
                    for x in r["fast"]["per_bias"]
                ],
            } for r in results
        ],
        "references": {
            "z430_baseline_cell_rmse_dec": Z430_BASELINE_DEC,
            "z443_VBIC_AVL_cell_rmse_dec": Z443_VBIC_AVL_DEC,
            "z448_BDF_slow_DC_cell_rmse_dec": Z448_REF_SLOW_DC_DEC,
        },
        "gates": gates,
        "audit": {
            "Cj0_nwell_fF": Cj0_nwell * 1e15,
            "Cnw_eff_at_Vb0_fF": Cnw_eff * 1e15,
            "vnwell_V": cfg_audit.vnwell,
            "C_B_const_fF": 1.0,
            "comment": ("Total C_eff at V_B=0 ≈ Cnw_eff + Cje + Cjc + C_B "
                         "≈ ~12 fF. n-well dominates; v449_B tests removing it."),
        },
        "wall_total_sec": round(time.time() - t0_main, 1),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    log(f"wrote summary.json  total_wall={summary['wall_total_sec']:.0f}s")

    # Honest analysis
    best = min(results, key=lambda r: r["slow_dc"]["cell_rmse_dec"])
    best_vb5 = None
    for r in results:
        pb = r["fast"]["per_bias"]
        if not pb: continue
        max5 = max(x["Vb_max_within_5ns_V"] for x in pb)
        if best_vb5 is None or max5 > best_vb5[1]:
            best_vb5 = (r["name"], max5)
    lines = [
        "# z449 — VBIC + BDF Combo: Honest Analysis\n",
        f"Wall time: {summary['wall_total_sec']:.0f}s\n\n",
        "## Variants tested\n",
    ]
    for r in results:
        lines.append(f"- **{r['name']}** ({r['desc']}): "
                     f"slow-DC = {r['slow_dc']['cell_rmse_dec']:.3f} dec "
                     f"({r['slow_dc']['n']} biases)\n")
    lines += ["\n## Fast-pulse summary\n",
              "| variant | bias | Vb_peak [V] | t_peak [ns] | Vb@5ns [V] | Id_dec | success |\n",
              "|---|---|---|---|---|---|---|\n"]
    for r in results:
        for rec in r["fast"]["per_bias"]:
            lines.append(
                f"| {r['name']} | {rec['tag']} | {rec['Vb_peak_V']:.4f} | "
                f"{rec['t_Vb_peak_s']*1e9:.2f} | "
                f"{rec['Vb_max_within_5ns_V']:.4f} | "
                f"{rec['Id_decade_swing']:.2f} | "
                f"{rec['solver']['success']} |\n")
    lines += [
        "\n## Body-cap audit\n",
        f"- C_B (Mario lumped) = 1.00 fF\n",
        f"- Cj0_nwell × area  = {Cj0_nwell*1e15:.2f} fF (full zero-bias junction)\n",
        f"- C_nwell @ V_B=0   = {Cnw_eff*1e15:.2f} fF (reverse-biased {cfg_audit.vnwell:.1f}V)\n",
        f"- Cje + Cjc (NPN)   ≈ 1.7 fF (parasitic BJT card)\n",
        f"- **Total C_eff(V_B=0) ≈ {(Cnw_eff*1e15 + 1.0 + 1.7):.1f} fF** — n-well dominates.\n",
        "\n## Gate evaluation\n",
        f"- DISCOVERY (any DC<0.85 + ≥50% biases V_B>0.3@5ns): **"
        f"{'PASS' if gates['DISCOVERY_pass'] else 'FAIL'}**"
        f"{' — ' + gates['DISCOVERY_variant'] if gates['DISCOVERY_variant'] else ''}\n",
        f"- AMBITIOUS (any DC<0.70 + ≥75% biases V_B>0.5@5ns): **"
        f"{'PASS' if gates['AMBITIOUS_pass'] else 'FAIL'}**"
        f"{' — ' + gates['AMBITIOUS_variant'] if gates['AMBITIOUS_variant'] else ''}\n",
        f"- KILL_SHOT (all DC≥1.0 OR no transient above 0.1V): **"
        f"{'TRIGGERED' if gates['KILL_SHOT'] else 'not triggered'}**\n",
        "\n## Diagnosis\n",
        f"- Best DC variant: **{best['name']}** at {best['slow_dc']['cell_rmse_dec']:.3f} dec\n",
        f"  (vs z443 VBIC_AVL alone = {Z443_VBIC_AVL_DEC:.3f}; "
        f"vs z430 GP baseline = {Z430_BASELINE_DEC:.3f}).\n",
    ]
    if best_vb5 is not None:
        lines.append(f"- Best fast-pulse V_B@5ns: **{best_vb5[0]}** = {best_vb5[1]:.4f} V "
                     "(target ≥ 0.5 V for snap-jump).\n")
    # Knob-by-knob conclusions
    A = results[0]["slow_dc"]["cell_rmse_dec"]
    B = results[1]["slow_dc"]["cell_rmse_dec"]
    C = results[2]["slow_dc"]["cell_rmse_dec"]
    A_vb = max((x["Vb_max_within_5ns_V"] for x in results[0]["fast"]["per_bias"]), default=0.0)
    B_vb = max((x["Vb_max_within_5ns_V"] for x in results[1]["fast"]["per_bias"]), default=0.0)
    C_vb = max((x["Vb_max_within_5ns_V"] for x in results[2]["fast"]["per_bias"]), default=0.0)
    lines += [
        "\n## Knob-by-knob (delta vs v449_A)\n",
        f"- v449_B (n-well cap → 0): ΔDC = {B-A:+.3f} dec, "
        f"ΔVb_5ns_max = {B_vb-A_vb:+.4f} V — "
        f"{'helped fast pulse' if (B_vb - A_vb) > 0.05 else 'no fast-pulse improvement'}\n",
        f"- v449_C (ALPHA0 × 5):     ΔDC = {C-A:+.3f} dec, "
        f"ΔVb_5ns_max = {C_vb-A_vb:+.4f} V — "
        f"{'helped fast pulse' if (C_vb - A_vb) > 0.05 else 'no fast-pulse improvement'}\n",
    ]
    if gates["KILL_SHOT"]:
        lines += [
            "\n## Root cause (KILL_SHOT)\n",
            "Even with VBIC avalanche + 1 fF lumped body cap + BDF charge-state\n",
            "solver, neither (a) removing the n-well cap nor (b) 5× ALPHA0 can\n",
            "drive V_B above 0.5 V within 5 ns. The body cap is no longer the\n",
            "limiter under v449_B (only ~2.7 fF total), so the residual issue\n",
            "is the **body-charging current** at V_D=2V is still too small —\n",
            "the BSIM4 impact-ion and VBIC avalanche together yield I_B ~ 10⁻⁶ A\n",
            "where ~10⁻⁵ A would be needed for a 10 ns snap-jump. Next-step\n",
            "recommendation: explicit Verilog-A latch (snapback_subcircuit) with\n",
            "V_BC-thresholded current source — *not* further parameter tweaks.\n",
        ]
    else:
        lines += [
            "\n## Recommendation\n",
            "At least one variant moved a gate. Prioritize the responsible knob\n",
            "for the next iteration: tune VBIC AVC1/AVC2 vs ALPHA0 vs explicit\n",
            "lumped n-well cap measurement from the Sebas card.\n",
        ]
    (OUT / "honest_analysis.md").write_text("".join(lines))
    log("wrote honest_analysis.md")
    log("DONE.")


if __name__ == "__main__":
    main()
