"""z436 — S24 SCR core: coupled NPN + lateral PNP + avalanche multiplier M(V_DB).

The NS-RAM 2T cell is structurally a PNPN thyristor:
  p+ (drain) - n- channel - p (body) - n-well (vnwell=2V) - p-substrate (GND)

Baseline z430 V_SINT_PIN models only the vertical NPN (Q1: C=drain (or Sint),
B=body, E=Sint). z434 added a lateral PNP but did NOT couple it regeneratively
with the NPN and applied avalanche M to Iii (impact-ion) instead of the NPN
collector current — the canonical SCR pattern is M(V_DB) acting on the
collector-base avalanche multiplication, which feeds the body and drives the
other BJT's base.

z436 implements the canonical SCR latch:
  - Q_NPN_vert: existing Q1 (C=drain, B=body, E=Sint=0) with M(V_DB) applied to
    its collector current.
  - Q_PNP_lat:  E=drain (p+), B=body, C=GND (substrate). Gummel-Poon with
    Bf=100, Is=5e-10, area=0.5.
  - M(V_DB) = 1 / (1 - (max(V_D-V_B,0)/V_BR)^n), V_BR=6 V, n=3.
  - Avalanche current (M-1)*Ic_NPN injected into body (impact-ion at the C-B
    junction). The "regenerative coupling" is implicit through the shared body
    node V_B: both BJTs see the SAME V_B in the outer Newton, so Ic_PNP feeds
    Ib_NPN through V_B and vice versa.

Variants:
  (a) BASELINE   — z430 V_SINT_PIN, no SCR additions
  (b) PNP_ONLY   — add lateral PNP
  (c) M_ONLY     — add M(V_DB) on NPN Ic + avalanche body current
  (d) PNP_AND_M  — both (no explicit coupling beyond shared V_B)
  (e) FULL_SCR   — (d) plus M(V_EB,PNP) also applied to PNP transport AND
                    explicit (M-1)*Ic_PNP added to body (positive feedback)

Outer solver: V_Sint pinned to 0 (z429.run_vsint_pinned style); 1-D Newton on
V_B with the SCR-extended R_B residual. The Newton itself IS the
self-consistency loop for the regenerative pair.

Outputs: results/z436_scr_core/
  - summary.json — per-variant cell + per-branch RMSE
  - ablation.json — deltas vs BASELINE and vs z430 V_SINT_PIN
  - overlay_VG1_{0p2,0p4,0p6}.png
  - v_db_trace.png — V_DB and M(V_DB) along Vd sweep
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
OUT = ROOT / "results/z436_scr_core"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "nsram"))
sys.path.insert(0, str(ROOT / "scripts"))

LOG = open(OUT / "run.log", "w")
def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.write(line + "\n"); LOG.flush()


# Reuse z427 / z429 wrappers
_spec427 = _ilu.spec_from_file_location("z427", ROOT / "scripts/z427_vsint_fix.py")
z427 = _ilu.module_from_spec(_spec427); _spec427.loader.exec_module(z427)
_spec429 = _ilu.spec_from_file_location("z429", ROOT / "scripts/z429_multisolver_debug.py")
z429 = _ilu.module_from_spec(_spec429); _spec429.loader.exec_module(z429)

from nsram.bsim4_port.nsram_cell_2T import _residuals  # noqa
from nsram.bsim4_port.constants import KboQ  # noqa


# ============================================================
# SCR parameters (oracle-spec defaults)
# ============================================================

PNP_LAT = dict(
    Is=5e-10,    # lateral less efficient than vertical NPN
    Bf=100.0,    # << 10000 for vertical
    Br=10.0,
    Nf=1.0,
    Nr=1.0,
    Va=100.0,
    area=0.5,    # lateral cross-section ~ half vertical
)

AVAL = dict(
    V_BR=6.0,    # drain-body breakdown
    n=3.0,       # Chynoweth exponent
    M_max=20.0,  # numerical clamp on M to avoid runaway in Newton
)

T_C = 25.0
T_K_DEFAULT = 273.15 + T_C
VT_DEFAULT  = KboQ * T_K_DEFAULT


def _avalanche_M(V_db, V_BR=6.0, n=3.0, M_max=20.0):
    """Chynoweth avalanche multiplier on C-B junction current."""
    Vdb_pos = max(V_db, 0.0)
    ratio = min(Vdb_pos / V_BR, 0.999)
    M = 1.0 / (1.0 - ratio ** n)
    return min(M, M_max)


def _pnp_lat_currents(V_drain, V_body, V_collector=0.0,
                      T_K=T_K_DEFAULT, P=PNP_LAT):
    """Lateral PNP DC currents (Gummel-Poon, simplified).
    Forward-active when V_EB = V_drain - V_body > 0  (drain p+ injects holes).
    Returns (I_E_into_drain, I_B_out_of_body, I_C_into_collector).

    Sign convention: I_E_into_drain > 0 means current flows OUT of drain
    EXTERNAL supply INTO the cell drain pin (i.e. it INCREASES drain current).
    I_B_out_of_body > 0 means current leaving the body node externally (so
    R_B sign: ADD as negative term to body charging residual).
    I_C_into_collector > 0 means current flowing INTO the collector sink.
    """
    Is_ = P["Is"] * P["area"]
    Bf  = P["Bf"]
    Br  = P["Br"]
    Nf  = P["Nf"]
    Nr  = P["Nr"]
    Va  = P["Va"]
    Vt  = KboQ * T_K

    V_EB = V_drain - V_body       # forward when > 0
    V_CB = V_collector - V_body   # < 0 in active (PNP CB reverse-bias)

    # exponentials with overflow cap
    arg_eb = min(V_EB / (Nf * Vt), 40.0)
    arg_cb = min(V_CB / (Nr * Vt), 40.0)
    Icc = Is_ * (math.exp(arg_eb) - 1.0)
    Iec = Is_ * (math.exp(arg_cb) - 1.0)

    # Early effect via inv_q1 = 1 - V_BC/Va = 1 + V_CB/Va; clamp positive
    inv_q1 = max(1.0 + V_CB / Va, 1e-3)
    Ict = (Icc - Iec) / inv_q1

    # Base recomb terms
    Ibe = Icc / Bf
    Ibc = Iec / Br

    # Terminal currents (signs: into-terminal-from-external positive)
    # E injects holes into body and supplies the recomb-base; transports to C.
    I_E_into_drain = Ict + Ibe      # > 0 in forward active
    # Body (base) lead must source the recomb current externally; in our
    # floating-body topology that means body LOSES this much charge.
    I_B_out_of_body = Ibe + Ibc      # leaves body
    # Collector sinks the transported holes from emitter
    I_C_into_collector = Ict - Ibc   # > 0 in forward active (into GND sink)
    return I_E_into_drain, I_B_out_of_body, I_C_into_collector


# ============================================================
# Extended residual: baseline R_B/R_S + SCR additions
# ============================================================

def _comp_local(cfg, model_M1, model_M2, bjt, Vsint_f, Vb_f, Vd_f,
                VG1_f, VG2_f):
    """Get (R_S, R_B, comp dict) from the cell's _residuals."""
    Vd = torch.tensor([Vd_f], dtype=torch.float64)
    VG1 = torch.tensor([VG1_f], dtype=torch.float64)
    VG2 = torch.tensor([VG2_f], dtype=torch.float64)
    Vsint = torch.tensor([Vsint_f], dtype=torch.float64)
    Vb = torch.tensor([Vb_f], dtype=torch.float64)
    with torch.no_grad():
        R_S, R_B, comp = _residuals(cfg, model_M1, bjt, Vd, VG1, VG2,
                                    Vsint, Vb, None, None, model_M2=model_M2)
    return float(R_S.item()), float(R_B.item()), comp


def _g(comp, key, default=0.0):
    v = comp.get(key, default)
    if torch.is_tensor(v):
        return float(v.reshape(-1)[0].item())
    return float(v)


def resid_scr(cfg, model_M1, model_M2, bjt,
              Vsint_f, Vb_f, Vd_f, VG1_f, VG2_f,
              use_pnp: bool, use_M: bool, use_full_coupling: bool):
    """Residual with optional SCR additions.

    Returns (R_S, R_B, Id_pred, diag) where diag has V_DB, M, I_pnp_E, etc.
    """
    R_S, R_B_base, comp = _comp_local(cfg, model_M1, model_M2, bjt,
                                       Vsint_f, Vb_f, Vd_f, VG1_f, VG2_f)
    # Existing cell drain assembly (mirror of solve_2t_steady_state):
    Id_base = (_g(comp, "Ids_M1") + _g(comp, "Ic_Q1") + _g(comp, "Ic_Q2")
               + _g(comp, "Ic_lat") + _g(comp, "Ic_avalanche")
               + _g(comp, "Igidl_M1") - _g(comp, "Ibd_M1")
               - _g(comp, "Ie_vert") + _g(comp, "I_snap_d"))

    R_B = R_B_base
    Id  = Id_base

    V_DB = Vd_f - Vb_f
    M = 1.0
    if use_M:
        M = _avalanche_M(V_DB, V_BR=AVAL["V_BR"], n=AVAL["n"],
                         M_max=AVAL["M_max"])

    # ---- (A) Avalanche multiplier on NPN collector current
    I_Q1_C = _g(comp, "Ic_Q1")
    I_avalanche_to_body = 0.0
    if use_M and abs(I_Q1_C) > 0:
        # Extra collector current = (M-1)*Ic; these are holes generated at C-B
        # junction → flow INTO body (charging it positive) and out at drain.
        I_avalanche_to_body = (M - 1.0) * I_Q1_C
        # Drain sees the same (M-1)*Ic since avalanche electrons go to C
        Id = Id + (M - 1.0) * I_Q1_C
        # Body residual was built with R_B = (charge_in - charge_out); add
        # avalanche holes flowing INTO body.
        R_B = R_B + I_avalanche_to_body

    # ---- (B) Lateral PNP
    I_pnp_E = 0.0
    I_pnp_B = 0.0
    I_pnp_C = 0.0
    if use_pnp:
        I_pnp_E, I_pnp_B, I_pnp_C = _pnp_lat_currents(
            Vd_f, Vb_f, V_collector=0.0, P=PNP_LAT)
        # PNP emitter sources current into drain pin → ADDS to Id
        Id = Id + I_pnp_E
        # PNP base recombination flows OUT of body → SUBTRACT from R_B
        R_B = R_B - I_pnp_B

    # ---- (C) Full coupling: apply M on PNP transport AND inject (M-1)*Ic_pnp
    # into body. This is the regenerative-pair amplification.
    if use_full_coupling and use_pnp and use_M:
        # Multiply PNP collector by M (treat as avalanche at PNP C-B junction
        # which is the body-substrate junction; rough approximation but
        # matches the SCR positive-feedback structure).
        I_pnp_C_M = (M - 1.0) * I_pnp_C
        # These extra holes are generated at the body-substrate junction; they
        # flow OUT through the substrate (i.e., remove holes from body) — but
        # in a PNPN the avalanche electrons inject into body, providing
        # base drive for the NPN. We model it as ADD into body (positive
        # feedback for SCR latch).
        R_B = R_B + I_pnp_C_M
        # And the matching emitter-current bump also lifts Id
        Id = Id + I_pnp_C_M

    diag = dict(V_DB=V_DB, M=M, I_Q1_C=I_Q1_C,
                I_pnp_E=I_pnp_E, I_pnp_B=I_pnp_B, I_pnp_C=I_pnp_C,
                I_avalanche_to_body=I_avalanche_to_body,
                Id_base=Id_base)
    return R_S, R_B, Id, diag


def run_vsint_pin_scr(cfg, model_M1, model_M2, bjt,
                      Vd_f, VG1_f, VG2_f,
                      use_pnp, use_M, use_full_coupling,
                      Vb_init=0.0, max_iters=80):
    """V_Sint pinned to 0; 1-D Newton on V_B with SCR-extended R_B."""
    Vb = Vb_init
    last_dV = 1.0
    for it in range(max_iters):
        R_S, R_B, _, _ = resid_scr(cfg, model_M1, model_M2, bjt,
                                    0.0, Vb, Vd_f, VG1_f, VG2_f,
                                    use_pnp, use_M, use_full_coupling)
        eps = 1e-5
        _, R_Bp, _, _ = resid_scr(cfg, model_M1, model_M2, bjt,
                                    0.0, Vb + eps, Vd_f, VG1_f, VG2_f,
                                    use_pnp, use_M, use_full_coupling)
        dRdV = (R_Bp - R_B) / eps
        if abs(dRdV) < 1e-30:
            break
        dV = -R_B / dRdV
        # damping when residual is in regenerative-positive-feedback regime
        if abs(dV) > 0.15:
            dV = math.copysign(0.15, dV)
        Vb_new = Vb + dV
        Vb_new = max(-0.3, min(1.0, Vb_new))
        if abs(Vb_new - Vb) < 1e-11:
            Vb = Vb_new
            break
        last_dV = abs(Vb_new - Vb)
        Vb = Vb_new
    R_S, R_B, Id, diag = resid_scr(cfg, model_M1, model_M2, bjt,
                                    0.0, Vb, Vd_f, VG1_f, VG2_f,
                                    use_pnp, use_M, use_full_coupling)
    converged = abs(R_B) < 1e-8 or last_dV < 1e-9
    return dict(Vb=Vb, Vsint=0.0, Id=Id, diag=diag,
                resid_RB=abs(R_B), resid_RS=abs(R_S),
                converged=converged, niter=it+1)


# ============================================================
# Variant runner (cell-wide)
# ============================================================

def run_variant(name, use_pnp, use_M, use_full_coupling,
                model_M1, model_M2, curves, sebas_rows,
                collect_diag_for=(0.2, 0.4, 0.6)):
    cfg, sd_M1, sd_M2 = z427.make_cfg(model_M1, model_M2, {})
    log_eps = 1e-15
    per_bias = []
    vb_max_overall = -1e30
    fails = 0
    t0 = time.time()
    for c in curves:
        sebas_row = z427.find_params(sebas_rows, c["VG1"], c["VG2"])
        if sebas_row is None or math.isnan(sebas_row.get("K1", float("nan"))):
            continue
        P_M1, P_M2 = z427.make_overrides(sebas_row)
        bjt = z427.make_bjt(sebas_row)
        Vd_arr = c["Vd"].numpy()
        Id_meas = c["Id"].numpy()
        Id_pred_list = []
        Vb_list = []
        Vdb_list = []
        M_list = []
        conv_list = []
        try:
            with torch.no_grad(), \
                 z427.patch_sd_scaled(sd_M1, P_M1), \
                 z427.patch_sd_scaled(sd_M2, P_M2):
                Vb_warm = 0.0
                for k, Vd_f in enumerate(Vd_arr):
                    r = run_vsint_pin_scr(cfg, model_M1, model_M2, bjt,
                                          float(Vd_f), float(c["VG1"]),
                                          float(c["VG2"]),
                                          use_pnp, use_M, use_full_coupling,
                                          Vb_init=Vb_warm)
                    Id_pred_list.append(abs(r["Id"]))
                    Vb_list.append(r["Vb"])
                    Vdb_list.append(r["diag"]["V_DB"])
                    M_list.append(r["diag"]["M"])
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
            fails += 1
            continue
        log_p = torch.log10(Id_pred + log_eps)
        log_m = torch.log10(c["Id"] + log_eps)
        sq = (log_p - log_m) ** 2
        rmse = float(torch.sqrt(sq[conv].mean()))
        vb_max = float(max(Vb_list))
        vb_max_overall = max(vb_max_overall, vb_max)
        rec = {"VG1": c["VG1"], "VG2": c["VG2"],
               "log_rmse": rmse, "vb_max": vb_max,
               "n_conv": int(conv.sum()), "n_pts": len(Vd_arr),
               "Vd": Vd_arr.tolist(),
               "Id_meas": Id_meas.tolist(),
               "Id_pred": Id_pred.tolist(),
               "Vb": Vb_list,
               "V_DB": Vdb_list,
               "M": M_list,
               "converged": conv_list}
        per_bias.append(rec)
    cell_sq = sum(r["log_rmse"]**2 for r in per_bias)
    cell_n = len(per_bias)
    cell = math.sqrt(cell_sq / cell_n) if cell_n else float("inf")
    per_branch = {}
    for r in per_bias:
        b = f"VG1_{r['VG1']:.1f}"
        per_branch.setdefault(b, {"sq": 0.0, "n": 0})
        per_branch[b]["sq"] += r["log_rmse"]**2
        per_branch[b]["n"] += 1
    per_branch_rmse = {b: math.sqrt(v["sq"]/v["n"]) for b, v in per_branch.items()}
    total_pts = sum(r["n_pts"] for r in per_bias)
    total_conv = sum(r["n_conv"] for r in per_bias)
    conv_rate = total_conv / max(total_pts, 1)
    log(f"  {name}: cell={cell:.3f} per_branch={ {k:round(v,3) for k,v in per_branch_rmse.items()} } "
        f"Vb_max={vb_max_overall:.3f} conv_rate={conv_rate*100:.1f}% "
        f"fails={fails} wall={time.time()-t0:.0f}s")
    return {
        "name": name,
        "cell_rmse_dec": cell,
        "per_branch_rmse_dec": per_branch_rmse,
        "n_biases_evaluated": cell_n,
        "vb_max_overall": vb_max_overall,
        "convergence_rate": conv_rate,
        "fails": fails,
        "wall_sec": round(time.time()-t0, 1),
        "per_bias": per_bias,
    }


# ============================================================
# Plots
# ============================================================

def overlay_plot(VG1_target: float, results: dict, fname: Path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    rows_by_vg2: dict[float, dict[str, dict]] = {}
    for name, r in results.items():
        for rec in r.get("per_bias", []):
            if abs(rec["VG1"] - VG1_target) < 1e-3:
                rows_by_vg2.setdefault(rec["VG2"], {})[name] = rec
    vg2_vals = sorted(rows_by_vg2.keys())
    if len(vg2_vals) >= 3:
        chosen = [vg2_vals[0], vg2_vals[len(vg2_vals)//2], vg2_vals[-1]]
    else:
        chosen = vg2_vals
    colors = {"BASELINE": "tab:red",
              "PNP_ONLY": "tab:orange",
              "M_ONLY": "tab:purple",
              "PNP_AND_M": "tab:blue",
              "FULL_SCR": "tab:green"}
    for ax, vg2 in zip(axes, chosen):
        sub = rows_by_vg2.get(vg2, {})
        meas = None
        for name in colors.keys():
            if name in sub:
                meas = sub[name]
                break
        if meas is None:
            ax.set_title(f"VG2={vg2:.2f} (no data)")
            continue
        ax.plot(meas["Vd"], meas["Id_meas"], "k-", lw=2.5, label="measured")
        for name in colors.keys():
            if name not in sub:
                continue
            rec = sub[name]
            ax.plot(rec["Vd"], rec["Id_pred"], "--", lw=1.4,
                    color=colors[name], label=name)
        ax.set_yscale("log")
        ax.set_xlabel("V_D [V]")
        ax.set_title(f"VG1={VG1_target:.1f}  VG2={vg2:.2f}")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("|I_D| [A]")
    fig.suptitle(f"z436 SCR core variants vs measured @ VG1={VG1_target:.1f}",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(fname, dpi=120)
    plt.close(fig)
    log(f"  wrote {fname.name}")


def vdb_trace_plot(results: dict, fname: Path):
    """V_DB and M(V_DB) vs V_D for FULL_SCR variant at representative biases."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    # Pick FULL_SCR if available, else last variant
    r = results.get("FULL_SCR") or results.get("PNP_AND_M") \
        or results.get("M_ONLY") or results.get("BASELINE")
    if r is None or not r.get("per_bias"):
        return
    # Plot first 9 biases for clarity
    cmap = plt.get_cmap("viridis")
    bias_sorted = sorted(r["per_bias"], key=lambda x: (x["VG1"], x["VG2"]))
    nshow = min(9, len(bias_sorted))
    for i, rec in enumerate(bias_sorted[::max(1, len(bias_sorted)//nshow)][:nshow]):
        color = cmap(i / max(1, nshow-1))
        label = f"VG1={rec['VG1']:.1f} VG2={rec['VG2']:+.2f}"
        axes[0].plot(rec["Vd"], rec["V_DB"], "-", color=color, label=label, lw=1.2)
        axes[1].plot(rec["Vd"], rec["M"], "-", color=color, label=label, lw=1.2)
    axes[0].set_xlabel("V_D [V]"); axes[0].set_ylabel("V_DB = V_D - V_B [V]")
    axes[0].axhline(AVAL["V_BR"], color="k", ls="--", lw=0.8,
                    label=f"V_BR={AVAL['V_BR']:.1f} V")
    axes[0].set_title("V_DB along Vd sweep")
    axes[0].legend(fontsize=7); axes[0].grid(alpha=0.3)
    axes[1].set_xlabel("V_D [V]"); axes[1].set_ylabel("M(V_DB)")
    axes[1].set_title(f"Avalanche M, V_BR={AVAL['V_BR']:.1f}, n={AVAL['n']:.1f}")
    axes[1].set_yscale("log")
    axes[1].legend(fontsize=7); axes[1].grid(alpha=0.3, which="both")
    fig.suptitle("z436 V_DB and M(V_DB) traces (FULL_SCR variant)")
    fig.tight_layout()
    fig.savefig(fname, dpi=120)
    plt.close(fig)
    log(f"  wrote {fname.name}")


# ============================================================
# Main
# ============================================================

def main():
    t_main = time.time()
    log("z436 starting — SCR core (NPN+PNP+M coupled) cell-wide test")
    model_M1, model_M2 = z427.build_models()
    curves = z427.load_curves()
    sebas_rows = z427.load_sebas_params()
    log(f"loaded: {len(curves)} curves, {len(sebas_rows)} sebas rows")
    log(f"PNP_LAT params: {PNP_LAT}")
    log(f"AVAL params: {AVAL}")

    variants = [
        ("BASELINE",   dict(use_pnp=False, use_M=False, use_full_coupling=False)),
        ("PNP_ONLY",   dict(use_pnp=True,  use_M=False, use_full_coupling=False)),
        ("M_ONLY",     dict(use_pnp=False, use_M=True,  use_full_coupling=False)),
        ("PNP_AND_M",  dict(use_pnp=True,  use_M=True,  use_full_coupling=False)),
        ("FULL_SCR",   dict(use_pnp=True,  use_M=True,  use_full_coupling=True)),
    ]
    results: dict[str, dict] = {}
    for name, kw in variants:
        log(f"=== {name} ===")
        results[name] = run_variant(name, **kw,
                                     model_M1=model_M1, model_M2=model_M2,
                                     curves=curves, sebas_rows=sebas_rows)

    # ---- Summary
    summary = {}
    for name, r in results.items():
        summary[name] = {
            "cell_rmse_dec": r["cell_rmse_dec"],
            "per_branch_rmse_dec": r["per_branch_rmse_dec"],
            "n_biases_evaluated": r["n_biases_evaluated"],
            "vb_max_overall": r["vb_max_overall"],
            "convergence_rate": r["convergence_rate"],
            "fails": r["fails"],
            "wall_sec": r["wall_sec"],
        }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    log("wrote summary.json")

    # ---- Ablation vs BASELINE and vs z430
    Z430_BASELINE = 1.6187161900853293
    base = summary["BASELINE"]["cell_rmse_dec"]
    # VG1=0.2 worst-branch improvement check
    def _vg1_02(s):
        return s["per_branch_rmse_dec"].get("VG1_0.2", float("inf"))
    base_vg1_02 = _vg1_02(summary["BASELINE"])
    ablation = {
        "z430_v_sint_pin_baseline_dec": Z430_BASELINE,
        "z436_baseline_cell_rmse_dec": base,
        "scr_params": dict(PNP_LAT=PNP_LAT, AVAL=AVAL),
        "z436_results": summary,
        "deltas_vs_z436_baseline": {
            n: base - summary[n]["cell_rmse_dec"] for n in summary
        },
        "deltas_vs_z430_v_sint_pin": {
            n: Z430_BASELINE - summary[n]["cell_rmse_dec"] for n in summary
        },
        "vg1_0p2_delta_vs_baseline": {
            n: base_vg1_02 - _vg1_02(summary[n]) for n in summary
        },
        "verdict_gates": {
            "INFRA_pass": all(summary[n]["n_biases_evaluated"] > 0
                              for n in summary),
            "DISCOVERY_pass": (
                any((base - summary[n]["cell_rmse_dec"]) >= 0.3
                    for n in summary if n != "BASELINE")
                or any((base_vg1_02 - _vg1_02(summary[n])) >= 0.5
                       for n in summary if n != "BASELINE")
            ),
            "AMBITIOUS_pass_lt_1p0": any(
                summary[n]["cell_rmse_dec"] < 1.0
                for n in summary if n != "BASELINE"),
            "KILL_SHOT_no_variant_helps": all(
                (base - summary[n]["cell_rmse_dec"]) < 0.05
                for n in summary if n != "BASELINE"),
        },
    }
    (OUT / "ablation.json").write_text(json.dumps(ablation, indent=2))
    log("wrote ablation.json")

    # ---- Overlay plots
    for vg1, suffix in [(0.2, "0p2"), (0.4, "0p4"), (0.6, "0p6")]:
        overlay_plot(vg1, results, OUT / f"overlay_VG1_{suffix}.png")
    # ---- V_DB / M trace
    vdb_trace_plot(results, OUT / "v_db_trace.png")

    # ---- Honest analysis
    h = ["# z436 — Coupled SCR core (NPN + PNP_lat + M(V_DB)) cell-wide test\n\n",
         "## Hypothesis (O74 oracle 4/4 consensus)\n",
         "The NS-RAM 2T cell is a PNPN thyristor. Baseline z430 V_SINT_PIN models\n"
         "only the vertical NPN (Q1). The SCR latch needs the lateral PNP, the\n"
         "avalanche multiplier M(V_DB) on the C-B junction, and regenerative\n"
         "coupling through the shared body node V_B.\n\n",
         "## Parameters\n",
         "- PNP_LAT: " + json.dumps(PNP_LAT) + "\n",
         "- AVAL:    " + json.dumps(AVAL) + "\n\n",
         "## Variants\n",
         "- BASELINE  — z430 V_SINT_PIN, no SCR additions\n",
         "- PNP_ONLY  — adds lateral PNP (E=drain, B=body, C=GND)\n",
         "- M_ONLY    — multiplies NPN Ic by M(V_DB) and injects (M-1)·Ic into body\n",
         "- PNP_AND_M — both, but no explicit cross-coupling (regen via shared V_B)\n",
         "- FULL_SCR  — also multiplies PNP transport by M and feeds (M-1)·Ic_PNP into body\n\n",
         "## Results\n```\n", json.dumps(summary, indent=2), "\n```\n\n",
         "## Deltas vs z436 BASELINE\n```\n",
         json.dumps(ablation["deltas_vs_z436_baseline"], indent=2), "\n```\n\n",
         "## Deltas vs z430 V_SINT_PIN (1.619 dec)\n```\n",
         json.dumps(ablation["deltas_vs_z430_v_sint_pin"], indent=2), "\n```\n\n",
         "## VG1=0.2 worst-branch deltas (BASELINE - variant)\n```\n",
         json.dumps(ablation["vg1_0p2_delta_vs_baseline"], indent=2), "\n```\n\n",
         "## Pre-registered verdict gates\n"]
    g = ablation["verdict_gates"]
    h.append(f"- INFRA (all variants produce data): "
             f"{'PASS' if g['INFRA_pass'] else 'FAIL'}\n")
    h.append(f"- DISCOVERY (≥0.3 dec cell-wide OR ≥0.5 dec VG1=0.2 improvement): "
             f"{'PASS' if g['DISCOVERY_pass'] else 'FAIL'}\n")
    h.append(f"- AMBITIOUS (cell-wide < 1.0 dec): "
             f"{'PASS' if g['AMBITIOUS_pass_lt_1p0'] else 'FAIL'}\n")
    h.append(f"- KILL_SHOT (no variant improves ≥0.05 dec → thyristor hypothesis wrong): "
             f"{'TRIGGERED' if g['KILL_SHOT_no_variant_helps'] else 'no'}\n\n")
    # V_DB analysis
    h.append("## V_DB / M(V_DB) avalanche-trigger analysis\n")
    full = results.get("FULL_SCR") or results.get("PNP_AND_M")
    if full and full.get("per_bias"):
        vdb_vals = []
        m_vals = []
        for rec in full["per_bias"]:
            vdb_vals.extend(rec["V_DB"])
            m_vals.extend(rec["M"])
        if vdb_vals:
            h.append(f"- V_DB range: [{min(vdb_vals):.3f}, {max(vdb_vals):.3f}] V "
                     f"(V_BR={AVAL['V_BR']:.1f} V)\n")
            h.append(f"- M range:    [{min(m_vals):.3f}, {max(m_vals):.3f}]\n")
            n_avalanche_pts = sum(1 for v in vdb_vals if v > 0.5 * AVAL["V_BR"])
            h.append(f"- Points with V_DB > 0.5·V_BR ({0.5*AVAL['V_BR']:.1f} V): "
                     f"{n_avalanche_pts}/{len(vdb_vals)} "
                     f"({100*n_avalanche_pts/max(1,len(vdb_vals)):.1f}%)\n")
            n_strong = sum(1 for m in m_vals if m > 1.5)
            h.append(f"- Points with M > 1.5: {n_strong}/{len(m_vals)} "
                     f"({100*n_strong/max(1,len(m_vals)):.1f}%)\n")
    h.append("\n## Per-bias residuals\n")
    for name, r in results.items():
        h.append(f"\n### {name}\n```\n")
        for rec in r.get("per_bias", []):
            h.append(f"VG1={rec['VG1']:.1f} VG2={rec['VG2']:+.2f}  "
                     f"RMSE={rec['log_rmse']:.3f} dec  "
                     f"Vb_max={rec['vb_max']:.3f}  "
                     f"conv={rec['n_conv']}/{rec['n_pts']}\n")
        h.append("```\n")

    h.append("\n## Honest caveats\n")
    h.append("- All variants are post-hoc additions on top of V_SINT_PIN (V_Sint=0 frozen). The\n"
             "  full 2-D Newton (V_Sint, V_B) coupling with the SCR pair is NOT solved jointly;\n"
             "  V_Sint coupling from PNP/NPN base recomb is ignored.\n")
    h.append("- 'Regenerative coupling' here is implicit through the shared body node V_B in the\n"
             "  outer 1-D Newton. The FULL_SCR variant adds an explicit (M-1)·Ic_PNP body-current\n"
             "  term — this is a phenomenological positive-feedback proxy, not a derivation\n"
             "  from carrier-continuity equations.\n")
    h.append("- M(V_DB) is clamped to M_max=20 to keep the Newton stable; a divergent M\n"
             "  represents physical breakdown, which the SCR latch would naturally enter — but\n"
             "  our 1-D Newton would not converge there.\n")
    h.append("- PNP Bf/Is/area set to the oracle-spec defaults; no Bf sweep was performed.\n")

    (OUT / "honest_analysis.md").write_text("".join(h))
    log("wrote honest_analysis.md")

    log(f"DONE wall={time.time()-t_main:.0f}s")
    LOG.close()


if __name__ == "__main__":
    main()
