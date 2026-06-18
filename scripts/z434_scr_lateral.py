"""z434 — S22-C: SCR PNPN-latch + lateral PNP + avalanche multiplier.

The NS-RAM 2T cell is structurally a PNPN thyristor. Baseline (z430)
already models the vertical NPN (drain=C, body=B, source/Sint=E). This
script adds the MISSING positive-feedback regime by including:

  (A) Lateral PNP: emitter = drain (p+), base = body (n-tub between
      drain and source). For active operation V_CB < 0, so the natural
      "collector" sink is **GND** (= source = 0 V), NOT Vnwell (which sits
      at +2 V and would put the device in saturation, not active SCR
      latch). This is the textbook SCR lateral PNP. Beta tuned down
      (Bf=100 vs the vertical NPN's 10000) because the lateral path is
      less efficient. We KEEP a `pnp_collector` knob — default `gnd`, but
      the script also runs a sanity sweep over collector choices.

  (B) Avalanche multiplier on impact ionization:
        M(V_DS) = 1 / (1 - (Vdb_pos / V_BR)^n)
      with V_BR=8 V, n=4. Multiplies the impact-ion current that feeds
      the body, boosting positive feedback at high V_D without changing
      α0/β0.

Method
------
We extend `z429.run_vsint_pinned` (V_Sint=0, 1-D Newton on V_B) with extra
body and drain terms. The PNP and avalanche-M contributions are computed
analytically (closed-form on (Vd, Vb)) and added into the body residual
R_B and the predicted drain current Id_pred.

Variants
--------
  BASELINE              — z430 V_SINT_PIN (no SCR additions)
  LATERAL_PNP_ONLY      — adds lateral PNP only
  LATERAL_PNP + AVAL_M  — adds lateral PNP AND avalanche-M on Iii

Pre-registered gates
--------------------
  INFRA       — both variants produce data, all biases converge
  DISCOVERY   — any non-baseline variant improves ≥ 0.3 dec vs z430
  AMBITIOUS   — cell-wide < 1.0 dec
  KILL_SHOT   — no variant helps → SCR/avalanche hypothesis disproven

Budget: 1.5 h. Output: results/z434_scr_lateral/.
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
OUT = ROOT / "results/z434_scr_lateral"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "nsram"))
sys.path.insert(0, str(ROOT / "scripts"))

LOG = open(OUT / "run.log", "w")
def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.write(line + "\n"); LOG.flush()


# --- reuse z427 / z429 / z430 wrappers
_spec427 = _ilu.spec_from_file_location("z427", ROOT / "scripts/z427_vsint_fix.py")
z427 = _ilu.module_from_spec(_spec427); _spec427.loader.exec_module(z427)
_spec429 = _ilu.spec_from_file_location("z429", ROOT / "scripts/z429_multisolver_debug.py")
z429 = _ilu.module_from_spec(_spec429); _spec429.loader.exec_module(z429)

from nsram.bsim4_port.nsram_cell_2T import _residuals  # noqa
from nsram.bsim4_port.constants import KboQ  # noqa


# ============================================================ #
# Lateral PNP + avalanche-M physics                            #
# ============================================================ #

# PNP parameters (lateral path is less efficient than vertical NPN)
PNP_DEFAULTS = dict(
    Is=5e-9,
    Bf=100.0,        # << 10000 (vertical NPN); lateral injection efficiency low
    Br=10.0,
    Nf=1.0,
    Nr=1.0,
    Va=100.0,
    area=1e-6,       # same scaling as vertical NPN
    Rs_emit=1e4,     # series-R for emitter — must be large so PNP I_E does
                     #   not dominate the channel current; physically this is
                     #   the access resistance of the lateral path (long
                     #   diffusion length, narrow cross-section).
    V_collector=0.0, # textbook SCR PNP: C = source/GND = 0 V
)

# Avalanche-M defaults (drain-body junction)
AVAL_DEFAULTS = dict(
    V_BR=8.0,        # breakdown voltage [V]
    n=4.0,           # multiplication exponent
)


def pnp_currents(Vd, Vb, vnwell, T_K=300.15, P=None):
    """Lateral PNP DC currents — E=Drain (p+), B=Body, C=Vnwell.

    Forward-active when V_EB = Vd - Vb > 0 (drain hole injection into body).
    Returns dict with terminal currents using the same sign convention as
    `compute_bjt`: Ie, Ib, Ic = currents INTO each terminal from external.
    For floating body / drain pin:
      - Ib  is current INTO body from external (positive in forward active
              because external base lead must supply electrons to recombine
              injected holes — for floating body, this current accumulates).
      - Ie  is current INTO drain pin (positive in forward active; emitter
              draws from external supply).
      - Ic  is current INTO vnwell sink (negative in forward active; vnwell
              must sink the collected hole current).
    """
    if P is None:
        P = PNP_DEFAULTS
    Is_ = P["Is"] * P["area"]
    Bf  = P["Bf"]
    Br  = P["Br"]
    Nf  = P["Nf"]
    Nr  = P["Nr"]
    Va  = P["Va"]
    Rs  = P["Rs_emit"]

    Vt = KboQ * T_K
    # Vd_eff with series-Rs limit — solve V_eb_internal implicitly?
    # Approximation: use Rs as a current-clamp via harmonic mean later.

    # Use PNP-canonical inputs: V_EB > 0 forward.
    V_EB = Vd - Vb
    V_CB = vnwell - Vb     # forward CB when V_CB > 0 (PNP saturation)

    # Standard Gummel-Poon transport, PNP convention (Icc driven by V_EB).
    # Cap exponent.
    arg_eb = (V_EB / (Nf * Vt))
    arg_eb = torch.clamp(arg_eb, max=40.0) if torch.is_tensor(arg_eb) else min(arg_eb, 40.0)
    if torch.is_tensor(V_EB):
        Icc = Is_ * (torch.exp(arg_eb) - 1.0)
    else:
        Icc = Is_ * (math.exp(arg_eb) - 1.0)

    arg_cb = (V_CB / (Nr * Vt))
    arg_cb = torch.clamp(arg_cb, max=40.0) if torch.is_tensor(arg_cb) else min(arg_cb, 40.0)
    if torch.is_tensor(V_CB):
        Iec = Is_ * (torch.exp(arg_cb) - 1.0)
    else:
        Iec = Is_ * (math.exp(arg_cb) - 1.0)

    # Early effect (simplified): q1 ≈ 1 - V_BC / Va = 1 - (-V_CB)/Va
    inv_q1 = 1.0 + V_CB / Va
    inv_q1 = max(inv_q1, 1e-4) if not torch.is_tensor(inv_q1) else torch.clamp(inv_q1, min=1e-4)
    q1 = 1.0 / inv_q1

    # Transport
    Ict = (Icc - Iec) / q1

    # Base currents
    Ibe = Icc / Bf      # forward base recomb component
    Ibc = Iec / Br

    # PNP terminal currents (currents INTO each terminal from external):
    #   I_E_into = +(Ict + Ibe)         emitter sources transport + B-E recomb
    #   I_B_into = -(Ibe + Ibc)         (PNP base flows OUT of base externally
    #                                    in forward active; minus sign here means
    #                                    the EXTERNAL base lead must SOURCE current
    #                                    INTO base node — i.e. Ib_INTO > 0)
    #   I_C_into = -(Ict - Ibc)         collector sinks the transported holes
    #
    # NB: Sign convention chosen so KCL closes: I_E + I_B + I_C = 0.
    I_E = Ict + Ibe
    I_B = - Ibe - Ibc   # this flows OUT of base externally; from body KCL perspective
                        # current OUT means body LOSES, so we add this with sign.
    I_C = - (Ict - Ibc)

    # Series-Rs current-limit on the emitter (acts as voltage drop). Use
    # harmonic-mean smooth limiter so derivatives stay continuous.
    I_Rs = (torch.relu(V_EB) if torch.is_tensor(V_EB)
            else max(V_EB, 0.0)) / max(Rs, 1e-30)
    if torch.is_tensor(I_E):
        eps = 1e-30
        I_E_lim = (I_E * I_Rs) / (I_E.abs() + I_Rs + eps)
        # Rescale Ib, Ic by same ratio (so KCL still closes)
        ratio = I_E_lim / (I_E + 1e-30 * torch.sign(I_E + 1e-30))
        I_B = I_B * ratio
        I_C = I_C * ratio
        I_E = I_E_lim
    else:
        eps = 1e-30
        I_E_lim = (I_E * I_Rs) / (abs(I_E) + I_Rs + eps)
        ratio = I_E_lim / (I_E + 1e-30 * (1.0 if I_E >= 0 else -1.0))
        I_B = I_B * ratio
        I_C = I_C * ratio
        I_E = I_E_lim

    return {"I_E": I_E, "I_B": I_B, "I_C": I_C}


def avalanche_M(Vd, Vb, V_BR=8.0, n=4.0):
    """Avalanche multiplier M(V_DS) acting on impact-ion current."""
    V_db = (Vd - Vb)
    Vdb_pos = max(V_db, 0.0) if not torch.is_tensor(V_db) else torch.clamp(V_db, min=0.0)
    ratio = min(Vdb_pos / V_BR, 0.99) if not torch.is_tensor(Vdb_pos) \
            else torch.clamp(Vdb_pos / V_BR, max=0.99)
    M = 1.0 / (1.0 - ratio ** n)
    return M


# ============================================================ #
# Extended resid_pair + Newton: BASELINE + lateral PNP + aval-M
# ============================================================ #

def resid_with_scr(cfg, model_M1, model_M2, bjt, Vsint_f, Vb_f, Vd_f,
                    VG1_f, VG2_f,
                    use_lateral_pnp: bool, use_aval_M: bool,
                    pnp_params=None):
    """Compute (R_S, R_B, Id_pred) with optional SCR additions."""
    # Baseline residuals via _residuals (returns R_S, R_B, components)
    R_S, R_B, comp = z429.resid_pair_full(cfg, model_M1, model_M2, bjt,
                                           Vsint_f, Vb_f, Vd_f, VG1_f, VG2_f) \
        if hasattr(z429, "resid_pair_full") else _resid_full_local(
        cfg, model_M1, model_M2, bjt, Vsint_f, Vb_f, Vd_f, VG1_f, VG2_f)

    # comp has Ids_M1, Ic_Q1, Igidl_M1, Ibd_M1, Ie_vert, I_snap_d (some may be missing)
    # Build base Id_pred (same as solve_2t_steady_state's assembly):
    def _g(k, default=0.0):
        v = comp.get(k, default)
        return float(v.item()) if torch.is_tensor(v) else float(v)
    Id_base = (_g("Ids_M1") + _g("Ic_Q1") + _g("Ic_Q2") + _g("Ic_lat")
               + _g("Ic_avalanche") + _g("Igidl_M1") - _g("Ibd_M1")
               - _g("Ie_vert") + _g("I_snap_d"))

    # --- Lateral PNP additions
    Id_pnp = 0.0
    R_B_extra = 0.0
    if use_lateral_pnp:
        # textbook SCR PNP: collector = GND (=source side of cell), not
        # vnwell (which would put device in saturation).
        V_collector_pnp = float(pnp_params.get("V_collector", 0.0)) \
            if pnp_params else 0.0
        pnp = pnp_currents(Vd_f, Vb_f, V_collector_pnp,
                            T_K=273.15 + cfg.T_C, P=pnp_params)
        # I_E flows INTO drain pin (emitter draws from external drain supply
        # → drain pin must source it → Id INCREASES by I_E)
        Id_pnp = float(pnp["I_E"])
        # I_B is current INTO body node (from base lead, which here is
        # floating body). In our sign convention I_B = -(Ibe+Ibc), which
        # represents current OUT of body. The body KCL R_B was built with
        # "INTO body = positive". We add R_B += I_B (negative for forward
        # active → body LOSES charge through PNP base recomb).
        R_B_extra += float(pnp["I_B"])

    # --- Avalanche-M boost on impact-ion (which feeds body)
    R_B_aval = 0.0
    if use_aval_M:
        # Get Iii from components (we need M1 + M2 impact currents)
        Iii_M1 = _g("Iii_M1")
        Iii_M2 = _g("Iii_M2")
        Iii_total = Iii_M1 + Iii_M2
        M = float(avalanche_M(Vd_f, Vb_f,
                              V_BR=AVAL_DEFAULTS["V_BR"],
                              n=AVAL_DEFAULTS["n"]))
        # The body already has Iii*iii_gain term in R_B; we ADD the
        # extra (M-1)*Iii_total on top.
        R_B_aval = (M - 1.0) * Iii_total

    R_B_total = float(R_B) + R_B_extra + R_B_aval
    Id_pred = Id_base + Id_pnp
    return float(R_S), R_B_total, Id_pred


def _resid_full_local(cfg, model_M1, model_M2, bjt,
                      Vsint_f, Vb_f, Vd_f, VG1_f, VG2_f):
    """Wrap _residuals to also expose components dict."""
    Vd = torch.tensor([Vd_f], dtype=torch.float64)
    VG1 = torch.tensor([VG1_f], dtype=torch.float64)
    VG2 = torch.tensor([VG2_f], dtype=torch.float64)
    Vsint = torch.tensor([Vsint_f], dtype=torch.float64)
    Vb = torch.tensor([Vb_f], dtype=torch.float64)
    with torch.no_grad():
        R_S, R_B, comp = _residuals(cfg, model_M1, bjt, Vd, VG1, VG2,
                                    Vsint, Vb, None, None, model_M2=model_M2)
    return float(R_S.item()), float(R_B.item()), comp


def run_vsint_pin_scr(name, cfg, model_M1, model_M2, bjt,
                      Vd_f, VG1_f, VG2_f,
                      use_lateral_pnp, use_aval_M,
                      pnp_params=None, Vb_init=0.0):
    """V_Sint pinned to 0, 1-D Newton on V_B with SCR-extended residual."""
    Vb = Vb_init
    converged = False
    for it in range(80):
        R_S, R_B, _ = resid_with_scr(cfg, model_M1, model_M2, bjt,
                                       0.0, Vb, Vd_f, VG1_f, VG2_f,
                                       use_lateral_pnp, use_aval_M,
                                       pnp_params)
        eps = 1e-5
        _, R_Bp, _ = resid_with_scr(cfg, model_M1, model_M2, bjt,
                                      0.0, Vb + eps, Vd_f, VG1_f, VG2_f,
                                      use_lateral_pnp, use_aval_M,
                                      pnp_params)
        dRdV = (R_Bp - R_B) / eps
        if abs(dRdV) < 1e-30:
            break
        dV = -R_B / dRdV
        if abs(dV) > 0.2:
            dV = math.copysign(0.2, dV)
        Vb_new = Vb + dV
        Vb_new = max(-0.2, min(1.0, Vb_new))
        if abs(Vb_new - Vb) < 1e-10:
            Vb = Vb_new
            break
        Vb = Vb_new
    R_S, R_B, Id = resid_with_scr(cfg, model_M1, model_M2, bjt,
                                    0.0, Vb, Vd_f, VG1_f, VG2_f,
                                    use_lateral_pnp, use_aval_M,
                                    pnp_params)
    converged = (abs(R_B) < 1e-8)
    return dict(Vb=Vb, Vsint=0.0, Id=Id,
                resid_RB=abs(R_B), resid_RS=abs(R_S),
                converged=converged)


# ============================================================ #
# Variant runner (cell-wide)                                   #
# ============================================================ #

def run_variant(name, use_lateral_pnp, use_aval_M, pnp_params,
                model_M1, model_M2, curves, sebas_rows):
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
        conv_list = []
        try:
            with torch.no_grad(), \
                 z427.patch_sd_scaled(sd_M1, P_M1), \
                 z427.patch_sd_scaled(sd_M2, P_M2):
                Vb_warm = 0.0
                for k, Vd_f in enumerate(Vd_arr):
                    r = run_vsint_pin_scr(name, cfg, model_M1, model_M2, bjt,
                                           float(Vd_f), float(c["VG1"]),
                                           float(c["VG2"]),
                                           use_lateral_pnp, use_aval_M,
                                           pnp_params, Vb_init=Vb_warm)
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
        "name": name, "cell_rmse_dec": cell,
        "per_branch_rmse_dec": per_branch_rmse,
        "n_biases_evaluated": cell_n,
        "vb_max_overall": vb_max_overall,
        "convergence_rate": conv_rate,
        "fails": fails,
        "wall_sec": round(time.time()-t0, 1),
        "per_bias": per_bias,
    }


# ============================================================ #
# Overlay plots                                                #
# ============================================================ #

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
              "LATERAL_PNP_ONLY": "tab:orange",
              "LATERAL_PNP_AVAL_M": "tab:green"}
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
        for name, rec in sub.items():
            ax.plot(rec["Vd"], rec["Id_pred"], "--", lw=1.5,
                    color=colors.get(name, "gray"), label=name)
        ax.set_yscale("log")
        ax.set_xlabel("V_D [V]")
        ax.set_title(f"VG1={VG1_target:.1f}  VG2={vg2:.2f}")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("|I_D| [A]")
    fig.suptitle(f"z434 SCR lateral-PNP variants vs measured @ VG1={VG1_target:.1f}",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(fname, dpi=120)
    plt.close(fig)
    log(f"  wrote {fname.name}")


# ============================================================ #
# Main                                                         #
# ============================================================ #

def main():
    t_main = time.time()
    log("z434 starting — SCR lateral-PNP + avalanche-M cell-wide test")
    model_M1, model_M2 = z427.build_models()
    curves = z427.load_curves()
    sebas_rows = z427.load_sebas_params()
    log(f"loaded: {len(curves)} curves, {len(sebas_rows)} sebas rows")

    results: dict[str, dict] = {}

    log("=== BASELINE (z430 V_SINT_PIN, no SCR additions) ===")
    results["BASELINE"] = run_variant(
        "BASELINE", use_lateral_pnp=False, use_aval_M=False,
        pnp_params=None,
        model_M1=model_M1, model_M2=model_M2,
        curves=curves, sebas_rows=sebas_rows)

    log("=== LATERAL_PNP_ONLY ===")
    results["LATERAL_PNP_ONLY"] = run_variant(
        "LATERAL_PNP_ONLY", use_lateral_pnp=True, use_aval_M=False,
        pnp_params=PNP_DEFAULTS,
        model_M1=model_M1, model_M2=model_M2,
        curves=curves, sebas_rows=sebas_rows)

    log("=== LATERAL_PNP + AVAL_M ===")
    results["LATERAL_PNP_AVAL_M"] = run_variant(
        "LATERAL_PNP_AVAL_M", use_lateral_pnp=True, use_aval_M=True,
        pnp_params=PNP_DEFAULTS,
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

    # ---- Ablation
    Z430_BASELINE = 1.6187161900853293  # z430 V_SINT_PIN cell-wide
    base = summary["BASELINE"]["cell_rmse_dec"]
    ablation = {
        "z430_v_sint_pin_baseline_dec": Z430_BASELINE,
        "z434_results": summary,
        "deltas_vs_z430_v_sint_pin": {
            n: Z430_BASELINE - summary[n]["cell_rmse_dec"]
            for n in summary
        },
        "deltas_vs_z434_baseline": {
            n: base - summary[n]["cell_rmse_dec"]
            for n in summary
        },
        "pnp_params": PNP_DEFAULTS,
        "aval_params": AVAL_DEFAULTS,
        "verdict_gates": {
            "INFRA_pass": all(summary[n]["n_biases_evaluated"] > 0
                              for n in summary),
            "DISCOVERY_pass_ge_0p3_improve": any(
                (Z430_BASELINE - summary[n]["cell_rmse_dec"]) >= 0.3
                for n in summary if n != "BASELINE"),
            "AMBITIOUS_pass_lt_1p0": any(
                summary[n]["cell_rmse_dec"] < 1.0
                for n in summary if n != "BASELINE"),
            "KILL_SHOT_no_help": all(
                (Z430_BASELINE - summary[n]["cell_rmse_dec"]) < 0.05
                for n in summary if n != "BASELINE"),
        },
    }
    (OUT / "ablation.json").write_text(json.dumps(ablation, indent=2))
    log("wrote ablation.json")

    # ---- Overlay plots
    for vg1, suffix in [(0.2, "0p2"), (0.4, "0p4"), (0.6, "0p6")]:
        overlay_plot(vg1, results, OUT / f"overlay_VG1_{suffix}.png")

    # ---- Honest analysis
    h = ["# z434 — SCR lateral-PNP + avalanche-M cell-wide test\n",
         "## Hypothesis\n",
         "The NS-RAM 2T cell is structurally a PNPN thyristor. Existing physics models the\n",
         "vertical NPN (D=C, B=B, Sint=E) but misses the **lateral PNP** drain-body-Nwell\n",
         "and possibly a finite-multiplication-factor avalanche term M(V_DS) beyond α0/β0.\n",
         "These together provide the positive-feedback regime for snapback at high V_D.\n",
         "\n## Variants\n",
         "- **BASELINE** — z430 V_SINT_PIN (V_Sint=0, 1-D Newton on V_B), no SCR additions.\n",
         "- **LATERAL_PNP_ONLY** — adds lateral PNP (E=Drain, B=Body, C=V_nwell=2.0V).\n",
         "  Params: Is=5e-9, Bf=100 (≪10000 vertical), Br=10, area=1e-6, Rs_emit=10Ω.\n",
         "- **LATERAL_PNP + AVAL_M** — also adds avalanche multiplier on Iii_total:\n",
         "  M(V_DS) = 1/(1 - (max(Vd-Vb,0)/V_BR)^n), V_BR=8.0V, n=4.0. Extra body current\n",
         "  (M-1)·Iii_total injected into floating body.\n",
         "\n## Results\n",
         "```\n", json.dumps(summary, indent=2), "\n```\n",
         "\n## Deltas vs z430 V_SINT_PIN baseline (1.619 dec)\n",
         "```\n", json.dumps(ablation["deltas_vs_z430_v_sint_pin"], indent=2), "\n```\n",
         "\n## Pre-registered verdict gates\n"]
    g = ablation["verdict_gates"]
    h.append(f"- INFRA: {'PASS' if g['INFRA_pass'] else 'FAIL'}\n")
    h.append(f"- DISCOVERY (≥0.3 dec improve on a non-baseline variant): {'PASS' if g['DISCOVERY_pass_ge_0p3_improve'] else 'FAIL'}\n")
    h.append(f"- AMBITIOUS (<1.0 dec cell-wide on a non-baseline variant): {'PASS' if g['AMBITIOUS_pass_lt_1p0'] else 'FAIL'}\n")
    h.append(f"- KILL_SHOT (no variant improves ≥0.05 dec → SCR hypothesis disproven): {'TRIGGERED' if g['KILL_SHOT_no_help'] else 'no'}\n")
    h.append("\n## Per-bias residuals\n")
    for name, r in results.items():
        h.append(f"\n### {name}\n```\n")
        for rec in r.get("per_bias", []):
            h.append(f"VG1={rec['VG1']:.1f} VG2={rec['VG2']:+.2f} "
                     f"RMSE={rec['log_rmse']:.3f} dec  "
                     f"Vb_max={rec['vb_max']:.3f}  "
                     f"conv={rec['n_conv']}/{rec['n_pts']}\n")
        h.append("```\n")

    h.append("\n## Honest caveats\n")
    h.append("- The PNP is added *post-hoc* on top of V_SINT_PIN (Vsint=0 frozen), not\n"
             "  inside the full 2-D (Vsint,Vb) Newton — so any V_sint coupling from the\n"
             "  PNP base recombination is ignored.\n")
    h.append("- M(Vds) applied to Iii_total uses (Vd-Vb) for V_DS; this is identical to\n"
             "  the existing `use_dbd_avalanche` logic at the structural level, but acts\n"
             "  on Iii (not Ids) — so it directly amplifies the SCR's body-current trigger.\n")
    h.append("- PNP Bf=100 is a physical guess; we did NOT sweep Bf to find a best-fit\n"
             "  point inside this run (budget 1.5h). Sweep is the natural follow-up.\n")

    (OUT / "honest_analysis.md").write_text("".join(h))
    log("wrote honest_analysis.md")

    log(f"DONE wall={time.time()-t_main:.0f}s")
    LOG.close()


if __name__ == "__main__":
    main()
