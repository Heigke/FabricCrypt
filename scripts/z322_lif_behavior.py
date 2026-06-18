"""R-5 z322 — LIF behavior test on v5-wired 2T NS-RAM pyport cell.

Hypothesis under test (Sebas 2026-04/05 emails): the 2T pyport cell,
when driven with a step input current injected into the floating body
node, behaves like a leaky integrate-and-fire (LIF) neuron:

    body cap Cb integrates I_inj  →  V_b rises (leaky via diode/TAT
    branches)  →  at V_b ≈ 0.6 V the parasitic NPN turns on  →
    M2 drain (Vsint) collapses producing a >0.5 V "spike"  →
    NPN sinks body charge  →  V_b drops below threshold  →
    refractory  →  repeat.

We use the BSIM4 IMPACT_IONIZATION path (compute_iimpact in M1/M2) plus
the body pdiode + TAT branches per v5 wiring. We do **not** enable an
avalanche diode at the drain (cfg.lat_BV set very high to disable).

Implementation: explicit-Euler transient on V_b. At each timestep we
quasi-statically solve V_sint (1D Newton on R_S only) at the current
V_b, then read R_B (net body current at that V_b from _residuals) and
integrate
    V_b(t+dt) = V_b(t) + (R_B(V_b) + I_inj) * dt / Cb

The spike output is V_sint(t) (M2 drain). We sweep I_inj amplitudes and
record V_b trajectory, spike train, refractory period, f-vs-I curve.

Locked gates:
  PASS:       ≥1 amplitude produces clean LIF behavior:
              - V_b accumulates to threshold (~0.5-0.8 V)
              - M2 fires spike with ≥0.5 V swing
              - V_b returns toward rest
              - re-fires after 1-100 µs refractory
  AMBITIOUS:  firing rate scales ~linearly with I_inj
  FAIL:       no spike at any amplitude, OR M2 saturated everywhere.

Output: results/z322_lif_behavior/summary.json
"""
from __future__ import annotations
import os
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "4")

import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

torch.set_default_dtype(torch.float64)

ROOT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
SCRIPTS = ROOT / "scripts"
OUT_DIR = ROOT / "results/z322_lif_behavior"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "nsram"))

DEVICE = torch.device("cpu")  # transient is sequential, cpu is fine

# v5 baseline constants (mirror z320)
BF = 500
ALPHA0 = 1e-4
R_BODY_TABLE = {0.2: 1.0e10, 0.4: 1.0e9, 0.6: 1.0e8}
PDIODE_IS_TOTAL = 5.3675e-7
PDIODE_AREA = 22e-12                  # 22 µm² — task spec
PDIODE_JS_PER_AREA = PDIODE_IS_TOTAL / PDIODE_AREA
PDIODE_N = 1.0535
TAT_JTSS = 3.4e-7
TAT_NJTS = 20.0
TAT_VTSS = 10.0
TAT_XTSS = 0.02

# Task spec for R-5
CBODY_TASK = 7e-15                    # 7 fF body cap
LAT_BV_DISABLE = 1.0e6                # disable avalanche diode (very high BV)
BODY_PDIODE_RS_TASK = 1.0e9           # task spec body_pdiode_Rs=1e9

# Bias / readout conditions
VG1_BIAS = 0.40                       # M1 weakly on
VG2_BIAS = 1.20                       # M2 strongly on (read-out path open)
VD_BIAS = 1.20                        # drain pulled up

# Time-stepping
DT_S = 1.0e-9                         # 1 ns Euler step
T_PRE_S = 5.0e-6                      # 5 µs ramp-up of input
T_STEP_S = 60.0e-6                    # 60 µs of constant input
T_TOTAL_S = T_PRE_S + T_STEP_S
N_STEPS = int(T_TOTAL_S / DT_S) + 1

# Input current amplitudes (A)
I_INJ_AMPS = [1.0e-9, 1.0e-8, 1.0e-7, 1.0e-6, 1.0e-5]

# Spike detection: V_sint dropping below this counts as "fired"
V_SINT_FIRE_THRESH_FRAC = 0.5         # > 0.5*Vd swing == spike
V_SINT_REARM_FRAC = 0.9               # V_sint must recover above this to re-arm

# V_b numerical guard
VB_MIN = -0.2
VB_MAX = 1.2


def _load_module(name, path):
    sp = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(mod)
    return mod


def configure_v5_lif(cfg, vg1=VG1_BIAS):
    """v5 wiring + R-5 modifications (Cb=7fF, disable avalanche)."""
    cfg.use_well_diode = False
    cfg.body_pdiode_to = "vnwell"
    if hasattr(cfg, "z310_enable_vnwell_diode"):
        cfg.z310_enable_vnwell_diode = False

    cfg.body_pdiode_Js = PDIODE_JS_PER_AREA
    cfg.body_pdiode_area = PDIODE_AREA
    cfg.body_pdiode_n = PDIODE_N
    cfg.body_pdiode_Rs = BODY_PDIODE_RS_TASK
    cfg.vnwell_Rs = 1.0e30

    # R-5: disable lateral avalanche (high BV)
    cfg.use_lateral_collector = True
    cfg.lat_BV = float(LAT_BV_DISABLE)
    cfg.lat_N = 4.0
    cfg.lat_BV_max = float(LAT_BV_DISABLE * 1.1)
    cfg.lat_M_smooth_delta = 0.5

    # TAT on (per v5)
    cfg.enable_tat = True
    cfg.tat_jtss = TAT_JTSS
    cfg.tat_njts = TAT_NJTS
    cfg.tat_vtss = TAT_VTSS
    cfg.tat_xtss = TAT_XTSS
    if hasattr(cfg, "z313_enable_tat"):
        cfg.z313_enable_tat = False

    # R-5: 7 fF body cap (task spec)
    cfg.Cbody = float(CBODY_TASK)

    cfg.invalidate() if hasattr(cfg, "invalidate") else None


def _residuals_at(z304, cfg, M1, M2, bjt, Vd_t, VG1_t, VG2_t, Vsint, Vb):
    """Wrap call to _residuals from the cell module."""
    from nsram.bsim4_port.nsram_cell_2T import _residuals
    return _residuals(cfg, M1, bjt, Vd_t, VG1_t, VG2_t, Vsint, Vb,
                      None, None, model_M2=M2)


def solve_vsint_1d(z304, cfg, M1, M2, bjt, Vd_t, VG1_t, VG2_t, Vb,
                   Vsint_init, max_iter=30, tol=1e-12):
    """Quasi-static 1D Newton over V_sint at FIXED V_b. Returns
    (Vsint, components_dict, R_B_at_Vb)."""
    Vsint = Vsint_init.detach().clone()
    for it in range(max_iter):
        R_S, R_B, comp = _residuals_at(z304, cfg, M1, M2, bjt,
                                        Vd_t, VG1_t, VG2_t, Vsint, Vb)
        rs = float(R_S.detach())
        if abs(rs) < tol:
            return Vsint, comp, float(R_B.detach())
        # finite-diff dR_S/dVsint
        h = max(1e-6, 1e-4 * (abs(float(Vsint)) + 1.0))
        Vsint_p = Vsint + h
        R_Sp, _, _ = _residuals_at(z304, cfg, M1, M2, bjt,
                                    Vd_t, VG1_t, VG2_t, Vsint_p, Vb)
        d = (float(R_Sp.detach()) - rs) / h
        if abs(d) < 1e-30:
            break
        # damped Newton with step cap
        step = -rs / d
        step = max(-0.2, min(0.2, step))
        Vsint = (Vsint + step).clamp(0.0, float(Vd_t) * 1.2)
    R_S, R_B, comp = _residuals_at(z304, cfg, M1, M2, bjt,
                                    Vd_t, VG1_t, VG2_t, Vsint, Vb)
    return Vsint, comp, float(R_B.detach())


def run_one_amplitude(z304, cfg, M1, M2, bjt, I_inj_amp):
    """Run a single transient with step-current I_inj_amp injected
    into body node. Returns dict with traces and spike stats."""
    Vd_t = torch.tensor(VD_BIAS, dtype=torch.float64)
    VG1_t = torch.tensor(VG1_BIAS, dtype=torch.float64)
    VG2_t = torch.tensor(VG2_BIAS, dtype=torch.float64)

    # Warm start V_sint near Vd (M2 off → V_sint pulled up via M1 to Vd)
    Vb = torch.tensor(0.0, dtype=torch.float64)
    Vsint = torch.tensor(VD_BIAS * 0.9, dtype=torch.float64)
    Vsint, _, _ = solve_vsint_1d(z304, cfg, M1, M2, bjt,
                                  Vd_t, VG1_t, VG2_t, Vb, Vsint)

    # Trace buffers (downsample by 10× to keep memory sane)
    SAMPLE_EVERY = 10
    t_trace = []
    vb_trace = []
    vsint_trace = []
    i_inj_trace = []
    rb_trace = []

    # Spike detection state
    spikes_t = []     # times of spike onset (first crossing)
    armed = True      # ready to fire
    fire_threshold = V_SINT_FIRE_THRESH_FRAC * VD_BIAS
    rearm_threshold = V_SINT_REARM_FRAC * VD_BIAS
    last_spike_t = -math.inf

    Cb = float(cfg.Cbody)
    Vb_at_fire = []
    refractory_intervals = []

    t = 0.0
    Vb_val = 0.0
    for step in range(N_STEPS):
        # Input ramp: linear ramp during T_PRE_S, then hold
        if t < T_PRE_S:
            I_inj = I_inj_amp * (t / T_PRE_S)
        else:
            I_inj = I_inj_amp

        Vb_tensor = torch.tensor(Vb_val, dtype=torch.float64)
        Vsint, comp, R_B = solve_vsint_1d(
            z304, cfg, M1, M2, bjt, Vd_t, VG1_t, VG2_t, Vb_tensor, Vsint)

        # Body KCL: dV_b/dt = (R_B + I_inj) / Cb
        # R_B is net current INTO body at current V_b (positive ⇒ body charges)
        I_b_net = R_B + I_inj
        dVb = I_b_net * DT_S / Cb
        # Damping for numerical stability: cap |dVb|
        dVb = max(-0.05, min(0.05, dVb))
        Vb_val = max(VB_MIN, min(VB_MAX, Vb_val + dVb))

        vsint_val = float(Vsint.detach())

        # Spike detection (V_sint falling below threshold)
        if armed and vsint_val < fire_threshold:
            spikes_t.append(t)
            Vb_at_fire.append(Vb_val)
            if last_spike_t > -math.inf:
                refractory_intervals.append(t - last_spike_t)
            last_spike_t = t
            armed = False
        elif (not armed) and vsint_val > rearm_threshold:
            armed = True

        if step % SAMPLE_EVERY == 0:
            t_trace.append(t)
            vb_trace.append(Vb_val)
            vsint_trace.append(vsint_val)
            i_inj_trace.append(I_inj)
            rb_trace.append(R_B)

        t += DT_S

    # Stats
    vsint_arr = np.array(vsint_trace)
    vb_arr = np.array(vb_trace)
    if len(spikes_t) >= 2:
        isi = np.diff(spikes_t)
        firing_rate_Hz = 1.0 / np.mean(isi) if np.mean(isi) > 0 else 0.0
        refractory_s = float(np.median(isi))
    elif len(spikes_t) == 1:
        firing_rate_Hz = 1.0 / (T_TOTAL_S - spikes_t[0]) if T_TOTAL_S > spikes_t[0] else 0.0
        refractory_s = None
    else:
        firing_rate_Hz = 0.0
        refractory_s = None

    swing = float(vsint_arr.max() - vsint_arr.min()) if len(vsint_arr) else 0.0
    saturated_low = bool(vsint_arr.max() < 0.1 * VD_BIAS)
    saturated_high = bool(vsint_arr.min() > 0.95 * VD_BIAS)

    # Save full trace (downsampled) to JSON-able list
    return {
        "I_inj_A": I_inj_amp,
        "n_spikes": len(spikes_t),
        "spike_times_s": [float(x) for x in spikes_t[:50]],
        "firing_rate_Hz": float(firing_rate_Hz),
        "refractory_s": (float(refractory_s) if refractory_s is not None else None),
        "Vb_at_fire": [float(x) for x in Vb_at_fire[:50]],
        "Vb_threshold_est": (float(np.mean(Vb_at_fire)) if Vb_at_fire else None),
        "Vsint_swing_V": swing,
        "Vsint_min_V": float(vsint_arr.min()) if len(vsint_arr) else None,
        "Vsint_max_V": float(vsint_arr.max()) if len(vsint_arr) else None,
        "Vb_min_V": float(vb_arr.min()) if len(vb_arr) else None,
        "Vb_max_V": float(vb_arr.max()) if len(vb_arr) else None,
        "saturated_low": saturated_low,
        "saturated_high": saturated_high,
        "trace": {
            "t_s": t_trace,
            "Vb_V": vb_trace,
            "Vsint_V": vsint_trace,
            "I_inj_A": i_inj_trace,
            "R_B_A": rb_trace,
        },
    }


def main():
    t0 = time.time()
    print(f"[z322] device={DEVICE}", flush=True)

    # Load model builders from z304 (same scaffold as z320)
    z304 = _load_module("z304", SCRIPTS / "z304_sebas_three_branch_refit.py")
    _, cfg, M1, M2, sd_M1, sd_M2, _forward_2t = z304.build_models_once()
    print(f"[z322] models built ({time.time()-t0:.1f}s)", flush=True)

    # Get the BJT model used by z304 helpers (same factory as z304)
    from nsram.bsim4_port.bjt import GummelPoonNPN
    bjt = GummelPoonNPN.from_sebas_card()
    bjt.Bf = float(BF)

    configure_v5_lif(cfg, vg1=VG1_BIAS)
    print(f"[z322] cfg: Cbody={cfg.Cbody:.2e} lat_BV={cfg.lat_BV:.1e} "
          f"body_pdiode_Rs={cfg.body_pdiode_Rs:.1e}", flush=True)
    print(f"[z322] bias: VG1={VG1_BIAS} VG2={VG2_BIAS} Vd={VD_BIAS}", flush=True)
    print(f"[z322] T_TOTAL={T_TOTAL_S*1e6:.1f}us  DT={DT_S*1e9:.1f}ns  "
          f"N_STEPS={N_STEPS}", flush=True)

    results = {}
    for I_amp in I_INJ_AMPS:
        t1 = time.time()
        print(f"\n[z322] === I_inj = {I_amp:.1e} A ===", flush=True)
        r = run_one_amplitude(z304, cfg, M1, M2, bjt, I_amp)
        elapsed = time.time() - t1
        print(f"[z322] I={I_amp:.1e}: n_spikes={r['n_spikes']} "
              f"f={r['firing_rate_Hz']:.2e}Hz swing={r['Vsint_swing_V']:.3f}V "
              f"Vb_thr={r['Vb_threshold_est']} ({elapsed:.1f}s)",
              flush=True)
        results[f"I_{I_amp:.0e}"] = r

        # Snapshot
        snap = {"results_so_far": {k: {kk: vv for kk, vv in v.items() if kk != "trace"}
                                    for k, v in results.items()},
                "elapsed_s": time.time() - t0}
        (OUT_DIR / "progress.json").write_text(json.dumps(snap, indent=2, default=float))

    # Gate evaluation
    any_clean_lif = False
    best_amp = None
    best_n_spikes = 0
    for amp_key, r in results.items():
        n = r["n_spikes"]
        sw = r["Vsint_swing_V"]
        refr = r["refractory_s"]
        # Clean LIF requires: ≥2 spikes (so we have refractory), swing≥0.5V,
        # refractory in [1 µs, 100 µs]
        if (n >= 2 and sw >= 0.5 and refr is not None
                and 1e-6 <= refr <= 100e-6):
            any_clean_lif = True
            if n > best_n_spikes:
                best_n_spikes = n
                best_amp = amp_key

    # I-vs-f curve linearity (ambitious)
    f_vals = []
    i_vals = []
    for amp_key, r in results.items():
        if r["n_spikes"] >= 2:
            f_vals.append(r["firing_rate_Hz"])
            i_vals.append(r["I_inj_A"])
    linear_fit_r2 = None
    if len(f_vals) >= 3:
        # log-log fit: f = a * I^b; if b≈1 → linear
        lx = np.log(np.array(i_vals))
        ly = np.log(np.array(f_vals))
        slope, intercept = np.polyfit(lx, ly, 1)
        pred = slope * lx + intercept
        ss_res = float(np.sum((ly - pred) ** 2))
        ss_tot = float(np.sum((ly - ly.mean()) ** 2))
        linear_fit_r2 = 1.0 - ss_res / (ss_tot + 1e-30)
        loglog_slope = float(slope)
    else:
        loglog_slope = None

    # FAIL: no spikes anywhere OR M2 saturated
    all_saturated = all((r["saturated_low"] or r["saturated_high"])
                        for r in results.values())
    no_spikes = all(r["n_spikes"] == 0 for r in results.values())

    summary = {
        "script": "z322_lif_behavior",
        "elapsed_s": time.time() - t0,
        "config": {
            "VG1_BIAS": VG1_BIAS, "VG2_BIAS": VG2_BIAS, "VD_BIAS": VD_BIAS,
            "DT_S": DT_S, "T_TOTAL_S": T_TOTAL_S, "N_STEPS": N_STEPS,
            "CBODY_F": CBODY_TASK,
            "PDIODE_AREA_m2": PDIODE_AREA,
            "BF": BF,
            "BODY_PDIODE_RS": BODY_PDIODE_RS_TASK,
            "LAT_BV_DISABLE": LAT_BV_DISABLE,
            "I_INJ_AMPS": I_INJ_AMPS,
        },
        "results": results,
        "gate_PASS_clean_LIF": bool(any_clean_lif),
        "gate_AMBITIOUS_linear_fI": bool(
            loglog_slope is not None
            and 0.7 <= loglog_slope <= 1.3
            and linear_fit_r2 is not None and linear_fit_r2 >= 0.9),
        "gate_FAIL_no_spikes_or_saturated": bool(no_spikes or all_saturated),
        "best_amp_key": best_amp,
        "best_n_spikes": int(best_n_spikes),
        "loglog_fit_slope": loglog_slope,
        "loglog_fit_r2": linear_fit_r2,
    }
    out_path = OUT_DIR / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2, default=float))

    print(f"\n[z322] gate PASS (clean LIF) = {summary['gate_PASS_clean_LIF']}",
          flush=True)
    print(f"[z322] gate AMBITIOUS (linear f-I) = "
          f"{summary['gate_AMBITIOUS_linear_fI']}  "
          f"(slope={loglog_slope}, r2={linear_fit_r2})", flush=True)
    print(f"[z322] gate FAIL (no spikes/saturated) = "
          f"{summary['gate_FAIL_no_spikes_or_saturated']}", flush=True)
    print(f"[z322] wrote {out_path}  ({time.time()-t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
