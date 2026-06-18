"""z334 — R-18 Iii→Vb COUPLING GAIN audit.

Goal: determine whether the Iii→Vb amplitude shortfall (2-4 dec, pre-knee slope
sign wrong) is a ROUTING bug, a PARAMETER scale, or BOTH.

Steps:
  1. Symbolic chain trace (written to routing_trace.md).
  2. At ngspice ground-truth bias (V_G1=0.6, V_G2=0.20, V_d=2.0,
     Vsint=0.382, Vb=0.267): dump Iii and every multiplicative factor
     that gates Iii into the Vb residual.
  3. Compare pyport's Iii→Vb contribution to ngspice (proxy: dump
     `@m1[ibs]`, `@m1[ibd]`, and explicit `i(m1)` substrate inflow via .save).
  4. ALPHA0 sweep over {7.84e-5, 7.84e-4, 7.84e-3, 7.84e-2}: for each,
     solve all available Sebas IV biases with basin-lock and report
     cell-wide median dec + pre-knee slope sign.
  5. Routing audit: enumerate every multiplicative factor that touches
     m1["Iii"] before it appears in R_B.
"""
from __future__ import annotations
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
import sys, json, math, re, csv, subprocess, importlib.util, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "nsram"))

OUT = ROOT / "results/z334_iii_vb_coupling"
OUT.mkdir(parents=True, exist_ok=True)

# Ground-truth bias from R-11 ngspice xcheck (z330)
VG1_GT = 0.6
VG2_GT = 0.20
VD_GT  = 2.0
VSINT_GT = 0.382
VB_GT    = 0.267

DATA = ROOT / "data/sebas_2026_04_22"


# -----------------------------------------------------------------------------
# 1. ROUTING TRACE (symbolic, read directly from nsram_cell_2T.py)
# -----------------------------------------------------------------------------
ROUTING_TRACE_MD = """\
# R-18 Iii → Vb routing trace (symbolic)

Source: `nsram/nsram/bsim4_port/nsram_cell_2T.py` lines ~410, 700-886.

## Chain (M1 only, m2_body_gnd=True default)

  1. Iii(M1) = compute_iimpact(model, sd, dc, Vds)            [leak.py §44-101]
       T2 = (alpha0 + alpha1*Leff) / Leff
       T1 = T2 * (Vds-Vdseff) * exp(-beta0/(Vds-Vdseff))   (strong-bias arm)
       Iii = T1 * Idsa_Vdseff
       --> alpha0 is a STRAIGHT MULTIPLIER on Iii (linear).

  2. iii_gain = eta_max * sigmoid(eta_slope * (Vds - eta_vds_th))
       defaults: eta_max=1.0, eta_slope=10/V, eta_vds_th=1.0 V
       At Vd=2.0 V: iii_gain = 1.0 * sigmoid(10*1.0) = 0.99995  (~1)
       At Vd=1.0 V: iii_gain = 1.0 * sigmoid(0)      = 0.500    (HALF)
       At Vd=0.5 V: iii_gain = 1.0 * sigmoid(-5)     = 6.7e-3   (KILLED)
       --> This DAMPS Iii into Vb whenever Vd<1V.

  3. iii_total_for_routing = m1["Iii"]                     (m2 body grounded)
     Ib_lat_pair = eta_lat * iii_gain * iii_total_for_routing
     iii_to_body_factor = (1.0 - eta_lat)
       default eta_lat = 0.0  (constant; reduces to F1.v2)
       --> iii_to_body_factor = 1.0 unless eta_sigmoid is enabled.

  4. R_B = iii_to_body_factor * iii_gain * m1["Iii"]       (BULK fraction → Vb)
         + m1["Igidl"] + m1["Igisl"] + m1["Igb"]
         - m1_d * m1["Ibs"] - m1_d * m1["Ibd"]
         - Ib_Q1                                            (BJT consumes base)
         - Ib_lat_pair                                      (lateral pair lost)
         + I_well_body
         - I_body_pdiode

## Effective Iii → Vb gate (m2_body_gnd path)

  Iii_to_Vb = (1.0 - eta_lat) * iii_gain * m1["Iii"]
            = (1.0 - eta_lat) * eta_max * sigmoid(eta_slope*(Vd-eta_vds_th)) * Iii

  Default cfg => (1.0 - 0.0) * 1.0 * sigmoid(10*(Vd-1)) * Iii
              = sigmoid(10*(Vd-1)) * Iii

  Pre-knee bias Vd≈0.5..0.9 V (cluster of cell IV cluster) → factor 6.7e-3..0.27.
  This is a 1-2 decade KNOCKOUT of Iii at pre-knee Vd.

## Multiplicative factors gating Iii into Vb (search list)

  | factor                       | default        | effect on Iii→Vb |
  |------------------------------|----------------|------------------|
  | iii_gain (eta sigmoid)       | sigmoid(10*(Vd-1)) | 0..1 (zeroes Vd<1) |
  | (1 - eta_lat)                | 1.0 (lat off)  | <=1              |
  | cfg.use_iii                  | True           | 1 or 0           |
  | cfg.m2_body_gnd              | True           | drops M2 share   |
  | cfg.iii_body_gain (legacy)   | None           | bypasses sigmoid |
  | cfg.eta_sigmoid              | False          | Vbe-dependent η  |
"""

(OUT / "routing_trace.md").write_text(ROUTING_TRACE_MD)
print("[z334] wrote routing_trace.md")


# -----------------------------------------------------------------------------
# 2. pyport models / sd / cfg (mirrors nsram_surrogate_4d._build_pyport_models)
# -----------------------------------------------------------------------------
def build_pyport():
    sp = importlib.util.spec_from_file_location("v1", ROOT / "scripts/z96_narma10_pilot.py")
    v1 = importlib.util.module_from_spec(sp); sp.loader.exec_module(v1)
    from nsram.bsim4_port.nsram_cell_2T import NSRAMCell2TConfig
    from nsram.bsim4_port.bjt import GummelPoonNPN
    cfg = NSRAMCell2TConfig(use_iii=True, use_gidl=True, use_bjt=True,
                             newton_max_iters=20)
    M1, M2 = v1.build_calibrated_models()
    bjt = GummelPoonNPN.from_sebas_card()
    bjt.Bf = 9000.0; bjt.Va = 0.55; bjt.Is = 1e-9
    return cfg, M1, M2, bjt


# -----------------------------------------------------------------------------
# 3. Bias-point dump at ground-truth (V_G1=0.6, V_G2=0.20, V_d=2.0)
#    Using EXACT Vsint/Vb from ngspice; we call _residuals to expose components.
# -----------------------------------------------------------------------------
def bias_dump():
    from nsram.bsim4_port.nsram_cell_2T import _residuals
    cfg, M1, M2, bjt = build_pyport()
    Vd = torch.tensor(VD_GT, dtype=torch.float64)
    VG1 = torch.tensor(VG1_GT, dtype=torch.float64)
    VG2 = torch.tensor(VG2_GT, dtype=torch.float64)
    Vsint = torch.tensor(VSINT_GT, dtype=torch.float64)
    Vb = torch.tensor(VB_GT, dtype=torch.float64)

    R_S, R_B, comp = _residuals(cfg, M1, bjt, Vd, VG1, VG2, Vsint, Vb,
                                  model_M2=M2)

    # Reconstruct gating factors per the trace
    eta_max = float(getattr(cfg, "eta_max", 1.0))
    eta_slope = float(getattr(cfg, "eta_slope", 10.0))
    eta_vds_th = float(getattr(cfg, "eta_vds_th", 1.0))
    Vds_eff = float(VD_GT)
    iii_gain = eta_max * float(torch.sigmoid(torch.tensor(eta_slope * (Vds_eff - eta_vds_th))))
    eta_lat  = float(getattr(cfg, "eta_lat", 0.0))
    iii_to_body_factor = 1.0 - eta_lat

    Iii_M1 = float(comp["Iii_M1"])
    Iii_M2 = float(comp.get("Iii_M2", 0.0))
    iii_total = Iii_M1 if cfg.m2_body_gnd else (Iii_M1 + Iii_M2)
    Iii_to_Vb_pyport = iii_to_body_factor * iii_gain * iii_total

    out = {
        "bias": {"VG1": VG1_GT, "VG2": VG2_GT, "Vd": VD_GT,
                  "Vsint": VSINT_GT, "Vb": VB_GT},
        "cfg_iii_gating": {
            "use_iii": cfg.use_iii,
            "m2_body_gnd": cfg.m2_body_gnd,
            "iii_body_gain_legacy": getattr(cfg, "iii_body_gain", None),
            "eta_max": eta_max, "eta_slope": eta_slope,
            "eta_vds_th": eta_vds_th,
            "eta_lat (constant)": eta_lat,
            "eta_sigmoid_enabled": bool(getattr(cfg, "eta_sigmoid", False)),
            "use_local_base": bool(getattr(cfg, "use_local_base", False)),
            "use_lateral_collector": bool(getattr(cfg, "use_lateral_collector", False)),
        },
        "pyport_intermediates": {
            "Iii_M1 (raw, from compute_iimpact)": Iii_M1,
            "Iii_M2 (raw, ignored since m2_body_gnd)": Iii_M2,
            "iii_total_for_routing": iii_total,
            "iii_gain (sigmoid Vd-dependent)": iii_gain,
            "iii_to_body_factor (= 1 - eta_lat)": iii_to_body_factor,
            "Iii_to_Vb_pyport": Iii_to_Vb_pyport,
        },
        "other_Vb_terms": {
            "Igidl_M1": float(comp["Igidl_M1"]),
            "Igisl_M1": float(comp["Igisl_M1"]),
            "Igb_M1": float(comp["Igb_M1"]),
            "Ibs_M1": float(comp["Ibs_M1"]),
            "Ibd_M1": float(comp["Ibd_M1"]),
            "Ib_Q1 (BJT base, sinks Vb)": float(comp["Ib_Q1"]),
            "I_well_body": float(comp.get("I_well_body", 0.0)),
            "I_body_pdiode": float(comp.get("I_body_pdiode", 0.0)),
            "Ib_lat_pair (lateral pair, lost to GND)": float(comp.get("Ib_lat_pair", 0.0)),
        },
        "R_S (Vsint residual at GT)": float(R_S),
        "R_B (Vb   residual at GT)": float(R_B),
    }
    return out


# -----------------------------------------------------------------------------
# 4. ngspice ground-truth — re-run z330 deck with extra .save lines for
#    Iii proxy. ngspice 'isub' M1 internal terminal isn't directly accessible;
#    we use `@m1[ibs]` (body-to-source current) and KCL on body node:
#      I_into_body_from_M1 ≈ -(@m1[ib])  if BSIM4 driver supports it; otherwise
#    we report v(vb) only and infer Iii→Vb from the well-diode current
#    needed to hold Vb at 0.267 V (KCL on body).
# -----------------------------------------------------------------------------
NGSPICE_DECK = f""".title z334 Iii->Vb groundtruth probe (VG1={VG1_GT}, VG2={VG2_GT}, Vd={VD_GT})

.include "{ROOT / 'data/sebas_2026_04_22/M1_130DNWFB.txt'}"
.include "{ROOT / 'data/sebas_2026_04_22/M2_130bulkNSRAM.txt'}"

.model parasiticBJT NPN(is=1e-9 va=0.55 bf=9000 br=100 nc=2 ikr=100m rc=0.1
+ vje=0.7 re=0.1 cjc=1e-15 fc=0.5 cje=0.7e-15 ne=1.5 ise=0 tr=20e-12 tf=25e-12
+ itf=0.03 vtf=7 xtf=2)

.model Dwell_mod D(IS=3.4089e-19 N=1.017 RS=0)

Vdd     vd       0       DC {VD_GT}
Vg1     vg1      0       DC {VG1_GT}
Vg2     vg2      0       DC {VG2_GT}
Vnwell  vnwell   0       DC 2.0

M1  vd vg1 vsint vb NMOSdnwfb L=0.13u W=1u
M2  vsint vg2 0 0 NMOS L=0.234u W=1u
Q1  vsint vb 0 parasiticBJT area=1u
Rwell  vnwell vnwell_x  10G
Dwell  vb     vnwell_x  Dwell_mod

.options gmin=1e-15 abstol=1e-12 reltol=1e-3 itl1=500

.control
save all
save @m1[ids]
save @m1[ibs]
save @m1[ibd]
save @m1[isub]
save @m2[ids]
save @q1[ib]
save @q1[ic]
op
print v(vsint)
print v(vb)
print -i(vdd)
print @m1[ids]
print @m1[ibs]
print @m1[ibd]
print @m1[isub]
print @m2[ids]
print @q1[ib]
print @q1[ic]
print i(dwell)
quit
.endc

.end
"""


def run_ngspice():
    deck_path = OUT / "ngspice_deck.sp"
    log_path  = OUT / "ngspice.log"
    deck_path.write_text(NGSPICE_DECK)
    try:
        proc = subprocess.run(["ngspice", "-b", str(deck_path)],
                                capture_output=True, text=True, timeout=120)
    except FileNotFoundError:
        return {"error": "ngspice not installed"}
    except subprocess.TimeoutExpired:
        return {"error": "timeout"}
    log = proc.stdout + "\n--- STDERR ---\n" + proc.stderr
    log_path.write_text(log)

    def grab(pat):
        m = re.search(pat + r"\s*=\s*([-+]?\d+\.?\d*[eE]?[-+]?\d*)", log)
        return float(m.group(1)) if m else None
    return {
        "rc": proc.returncode,
        "v(vsint)": grab(r"v\(vsint\)"),
        "v(vb)":    grab(r"v\(vb\)"),
        "-i(vdd)":  grab(r"-i\(vdd\)"),
        "@m1[ids]": grab(r"@m1\[ids\]"),
        "@m1[ibs]": grab(r"@m1\[ibs\]"),
        "@m1[ibd]": grab(r"@m1\[ibd\]"),
        "@m1[isub]": grab(r"@m1\[isub\]"),
        "@m2[ids]": grab(r"@m2\[ids\]"),
        "@q1[ib]":  grab(r"@q1\[ib\]"),
        "@q1[ic]":  grab(r"@q1\[ic\]"),
        "i(dwell)": grab(r"i\(dwell\)"),
    }


# -----------------------------------------------------------------------------
# 5. ALPHA0 sweep — solve all Sebas IV biases at each alpha0 value.
#    For speed: use _solve_at_fixed_vb at the ngspice ground-truth Vb=0.267
#    (basin-lock surrogate). Then compare to measured Id at each bias.
# -----------------------------------------------------------------------------
def load_sebas_curves():
    """Return list of dicts with VG1, VG2, Vd (np), Id_meas (np)."""
    curves = []
    for d in sorted(DATA.glob("2vHCa-2 I-Vs@VG2 VG1=*")):
        mv = re.search(r"VG1=([\d.]+)", d.name)
        if not mv: continue
        VG1 = float(mv.group(1))
        for f in sorted(d.glob("*.csv")):
            mg = re.search(r"VG2=(-?\d+\.\d+)", f.name)
            if not mg: continue
            VG2 = float(mg.group(1))
            data = np.loadtxt(f, delimiter=",", skiprows=1, usecols=(0, 1))
            if data.ndim == 1:
                continue
            half = len(data) // 2
            Vd = data[:half, 0]
            Id = np.abs(data[:half, 1])
            mask = (Vd >= 0.05) & (Vd <= 2.0)
            Vd, Id = Vd[mask], Id[mask]
            if len(Vd) < 10: continue
            idx = np.linspace(0, len(Vd) - 1, 20).astype(int)
            curves.append({"VG1": VG1, "VG2": VG2,
                            "Vd": Vd[idx], "Id_meas": Id[idx]})
    return curves


def alpha0_sweep():
    """Sweep ALPHA0 on M1. For each value:
       - solve all biases (Vd grid x 20 pts) with Vb pinned at 0.267 (basin-lock).
       - report cell-wide median log-RMSE (dec).
       - report pre-knee slope sign at VG1=0.4, VG2 in {0, 0.2, 0.4}.
    """
    from nsram.bsim4_port.nsram_cell_2T import _residuals
    cfg, M1, M2, bjt = build_pyport()
    curves = load_sebas_curves()
    print(f"[z334] loaded {len(curves)} sebas curves for alpha0 sweep")

    ALPHA0_VALUES = [7.84e-5, 7.84e-4, 7.84e-3, 7.84e-2]

    # The alpha0 lives in sd.scaled, populated by poly_params from model card.
    # We monkey-patch model._values["alpha0"] AND its sd.scaled entry if cached.
    # Simpler: override sd.scaled["alpha0"] in a context manager around solver.
    # Use cfg's lazy SD accessors (compute_size_dep). Calling _residuals
    # populates these on first call; we then mutate scaled['alpha0'].
    sd_M1 = cfg.size_dep_M1(M1)
    _ = cfg.size_dep_M2(M2)

    log_eps = 1e-15

    def solve_one(Vd, VG1, VG2, Vb_fixed=0.267):
        Vd_t = torch.tensor(float(Vd), dtype=torch.float64)
        VG1_t = torch.tensor(float(VG1), dtype=torch.float64)
        VG2_t = torch.tensor(float(VG2), dtype=torch.float64)
        Vb_t  = torch.tensor(float(Vb_fixed), dtype=torch.float64)
        Vsint = (0.5 * Vd_t).clone()
        for it in range(cfg.newton_max_iters):
            R_S, R_B, comp = _residuals(cfg, M1, bjt, Vd_t, VG1_t, VG2_t,
                                         Vsint, Vb_t, model_M2=M2)
            if abs(float(R_S)) < 1e-12:
                break
            h = 1e-6
            R_Sp, _, _ = _residuals(cfg, M1, bjt, Vd_t, VG1_t, VG2_t,
                                      Vsint + h, Vb_t, model_M2=M2)
            dRdV = (R_Sp - R_S) / h
            if abs(float(dRdV)) < 1e-30:
                break
            dV = -R_S / dRdV
            dV = torch.clamp(dV, -0.5, 0.5)
            Vsint = Vsint + dV
        R_S, R_B, comp = _residuals(cfg, M1, bjt, Vd_t, VG1_t, VG2_t,
                                     Vsint, Vb_t, model_M2=M2)
        Id = (comp["Ids_M1"] + comp["Ic_Q1"]
              + comp.get("Ic_lat", 0.0) + comp.get("Ic_avalanche", 0.0)
              + comp["Igidl_M1"] - comp["Ibd_M1"])
        return float(abs(float(Id)))

    sweep_results = []
    for alpha0_val in ALPHA0_VALUES:
        t0 = time.time()
        # Inject alpha0 override
        saved = sd_M1.scaled.get("alpha0", None)
        sd_M1.scaled["alpha0"] = float(alpha0_val)

        # Diagnostic: print Iii_M1 at GT bias under this alpha0
        from nsram.bsim4_port.nsram_cell_2T import _residuals
        Vd_t = torch.tensor(VD_GT, dtype=torch.float64)
        VG1_t = torch.tensor(VG1_GT, dtype=torch.float64)
        VG2_t = torch.tensor(VG2_GT, dtype=torch.float64)
        Vsint_t = torch.tensor(VSINT_GT, dtype=torch.float64)
        Vb_t = torch.tensor(VB_GT, dtype=torch.float64)
        _, _, _comp = _residuals(cfg, M1, bjt, Vd_t, VG1_t, VG2_t,
                                   Vsint_t, Vb_t, model_M2=M2)
        Iii_gt = float(_comp["Iii_M1"])
        print(f"    [diag] alpha0={alpha0_val:.2e} -> Iii_M1@GT = {Iii_gt:.3e}")

        try:
            cell_log_rmses = []
            slope_signs = {}
            for c in curves:
                Id_pred = np.array([solve_one(vd, c["VG1"], c["VG2"])
                                     for vd in c["Vd"]])
                Id_meas = c["Id_meas"]
                rmse = float(np.sqrt(np.mean(
                    (np.log10(Id_pred + log_eps) -
                     np.log10(Id_meas + log_eps))**2)))
                cell_log_rmses.append({"VG1": c["VG1"], "VG2": c["VG2"],
                                        "log_rmse": rmse})
                # Pre-knee slope: VG1=0.4, VG2 in {0, 0.2, 0.4}, Vd in 0.1..0.5
                if abs(c["VG1"] - 0.4) < 1e-3 and any(
                        abs(c["VG2"] - v) < 1e-3 for v in (0.0, 0.2, 0.4)):
                    mask = (c["Vd"] >= 0.05) & (c["Vd"] <= 0.5)
                    if mask.sum() >= 3:
                        slope_pred = np.polyfit(c["Vd"][mask],
                                                  np.log10(Id_pred[mask]+log_eps), 1)[0]
                        slope_meas = np.polyfit(c["Vd"][mask],
                                                  np.log10(Id_meas[mask]+log_eps), 1)[0]
                        slope_signs[f"VG2={c['VG2']:+.2f}"] = {
                            "pred": float(slope_pred),
                            "meas": float(slope_meas),
                            "sign_match": bool(np.sign(slope_pred) == np.sign(slope_meas)),
                        }
            rmses_arr = np.array([r["log_rmse"] for r in cell_log_rmses
                                    if np.isfinite(r["log_rmse"])])
            median_dec = float(np.median(rmses_arr)) if len(rmses_arr) else float("inf")
            sweep_results.append({
                "alpha0": alpha0_val,
                "Iii_M1_at_GT": Iii_gt,
                "median_log_rmse_dec": median_dec,
                "n_curves": len(cell_log_rmses),
                "preknee_slope_VG1_0.4": slope_signs,
                "elapsed_s": time.time() - t0,
            })
            print(f"  alpha0={alpha0_val:.2e}: median_dec={median_dec:.3f} "
                  f"({time.time()-t0:.0f}s, {len(rmses_arr)} curves)")
        finally:
            if saved is None:
                sd_M1.scaled.pop("alpha0", None)
            else:
                sd_M1.scaled["alpha0"] = saved
    return sweep_results


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main():
    print("=== z334 R-18 Iii->Vb coupling audit ===")
    print(f"GT bias: VG1={VG1_GT}, VG2={VG2_GT}, Vd={VD_GT}, "
          f"Vsint={VSINT_GT}, Vb={VB_GT}")

    # -- 1. bias dump --
    print("\n[step 1] bias dump (pyport at GT) ...")
    py = bias_dump()
    print(json.dumps(py["pyport_intermediates"], indent=2))

    # -- 2. ngspice GT --
    print("\n[step 2] ngspice ground-truth ...")
    ng = run_ngspice()
    print(json.dumps(ng, indent=2))

    # Build bias_table.json
    Iii_pyport_to_Vb = py["pyport_intermediates"]["Iii_to_Vb_pyport"]
    # ngspice proxy for body-injection current: i(dwell) is the well->body diode
    # current that must balance net Iii->Vb at DC. KCL on body in op mode:
    # I_in(Iii + Igidl + Igb + Iwell) = I_out(Ibs+Ibd+Ib_Q1+Ib_pdiode).
    # @m1[isub] is the BSIM4 internal substrate current pin (Iii). Use it.
    iii_ngspice = ng.get("@m1[isub]")
    ratio = (Iii_pyport_to_Vb / iii_ngspice) if iii_ngspice else None
    bias_table = {
        "bias": py["bias"],
        "pyport": py["pyport_intermediates"],
        "pyport_other_Vb_terms": py["other_Vb_terms"],
        "ngspice": ng,
        "ratio_pyport_Iii_to_Vb_over_ngspice_isub": ratio,
        "note": "ngspice @m1[isub] is BSIM4 internal substrate current "
                "(impact-ion holes into body). Compare to pyport "
                "Iii_to_Vb_pyport = (1-eta_lat)*iii_gain*Iii_M1.",
    }
    (OUT / "bias_table.json").write_text(json.dumps(bias_table, indent=2))
    print(f"[z334] wrote bias_table.json (ratio={ratio})")

    # -- 3. alpha0 sweep --
    print("\n[step 3] ALPHA0 sweep over {7.84e-5, 7.84e-4, 7.84e-3, 7.84e-2} ...")
    sweep = alpha0_sweep()
    (OUT / "alpha0_sweep.json").write_text(json.dumps(sweep, indent=2))
    print("[z334] wrote alpha0_sweep.json")

    # -- summary print --
    print("\n=== SUMMARY ===")
    print(f"  pyport Iii_to_Vb : {Iii_pyport_to_Vb:.3e}")
    print(f"  ngspice @m1[isub]: {iii_ngspice}")
    print(f"  ratio            : {ratio}")
    best = min(sweep, key=lambda r: r["median_log_rmse_dec"])
    print(f"  best alpha0 (dec): {best['alpha0']:.2e} -> {best['median_log_rmse_dec']:.3f}")


if __name__ == "__main__":
    main()
