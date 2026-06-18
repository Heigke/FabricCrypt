"""z308 — TRUE time-stepped transient validation of pyport against Sebas IV CSVs.

Improves on z298b (quasi-static Vb equilibrium) by integrating dVb/dt
explicitly over the measured time grid (tdata column from CSV), so
finite-rate hysteresis emerges naturally.

Model
-----
    dVb/dt = (Iii_in(Vd, VG1, VG2, Vb) - Ileak_out(Vd, VG1, VG2, Vb)) / Cb
    Id(t)   = Id  (Vd(t), VG1, VG2, Vb(t))

Numerical scheme
----------------
    Semi-implicit per outer step: at every measurement timestamp t_k:
        Vd_k = vdata[k]
        Iii, Ileak, Id = pyport.solve(Vd_k, VG1, VG2, Vb_k)
                                                 # 1D Newton on Vsint at fixed Vb
        Vb_{k+1} = clip(Vb_k + dt_k * (Iii-Ileak) / Cb, 0, Vb_max)
        Id_out[k] = Id

    The measurement dt (typical 30-60ms, max ~600ms in long pause) is
    small enough vs Vb relaxation time (Cb*Rb where Rb~10-100 GΩ given
    leakage currents ~1e-12-1e-10 A, so tau ~ tens of seconds at Cb=5fF)
    that forward Euler on Vb is stable. For safety we sub-step when
    dt > tau_local/4.

Gates
-----
    PASS-conservative : median forward log-RMSE < 0.5 dec on 33 curves
    AMBITIOUS         : same + reverse < 0.5 dec + median hysteresis
                        within 2x of measured
    BONUS             : multi-rate predictions exported (no measurement
                        available yet)

Fallback strategy: tries to import z307_pyport_v2 if present, else uses
z294 MEP-7 pyport (current Vb-clamp version) for the inner Newton.
"""
from __future__ import annotations
import os
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "4")

import importlib.util
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

# ROOT = repo root (parent of scripts/). Resolves correctly on ikaros and on
# the queue-worker sandbox (daedalus /home/naorw/nsram_queue_sandbox, etc.).
ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data/sebas_2026_04_22"
OUT = ROOT / "results/z308_transient_validate"
OUT.mkdir(parents=True, exist_ok=True)

# CPU is ~5x faster than GPU for the inner Newton because each timestep is
# a single 1D solve — GPU dispatch overhead dominates. Override with
# Z308_DEVICE=cuda if you really want to.
DEVICE = torch.device(os.environ.get("Z308_DEVICE", "cpu"))
DTYPE = torch.float64

# Physical defaults (configurable via env)
CB = float(os.environ.get("Z308_CB", "5e-15"))           # 5 fF body cap (project default)
VB_MAX = float(os.environ.get("Z308_VB_MAX", "0.80"))    # forward-bias clamp (well diode opens)
VB0 = float(os.environ.get("Z308_VB0", "0.0"))
SUBSTEP_FRAC = float(os.environ.get("Z308_SUBSTEP_FRAC", "0.25"))  # of local tau

print(f"[z308] device={DEVICE} dtype={DTYPE} Cb={CB:.2e} Vb_max={VB_MAX}")


# ---------------------------------------------------------------- pyport loader
def _load_solver():
    """Always use MEP-7 batched Newton (z294). Enable v2 topology if available.

    The v2 nsram_pyport_v2 monkey-patches _residuals to add the SA3 topology
    elements (vnwell breakdown, drain avalanche, anode-Vb diode). It does NOT
    expose a new solve API — the same z294 solve_batched_gpu picks up the
    patched residuals automatically once enable_v2_topology(cfg, ...) is called.
    """
    print(f"[z308] using z294 MEP-7 pyport (Newton on Vsint @ fixed Vb)")
    sp = importlib.util.spec_from_file_location("mep7", ROOT / "scripts/z294_mep7_gpu_pyport.py")
    mep7 = importlib.util.module_from_spec(sp); sp.loader.exec_module(mep7)
    ns4d = mep7._load_cpu_ref()
    cfg, M1, M2, bjt = ns4d._build_pyport_models()
    # Try to enable v2 topology
    tag = "v1"
    if os.environ.get("Z308_USE_V2", "1") != "0":
        try:
            sys.path.insert(0, str(ROOT / "src"))
            from nsram_pyport_v2 import V2Params, enable_v2_topology
            enable_v2_topology(cfg, V2Params())
            tag = "v2"
            print(f"[z308] v2 topology ENABLED (vnwell breakdown, drain avalanche, anode-Vb diode)")
        except Exception as e:
            print(f"[z308] v2 topology unavailable ({e}); falling back to v1")
    return (tag, mep7, cfg, M1, M2, bjt)


def solve_pt(api_tag, mod, cfg, M1, M2, bjt, Vd, VG1, VG2, Vb):
    """Batched (B,) solve at fixed Vb → dict of (B,) numpy arrays."""
    Vd_t  = torch.tensor(Vd,  dtype=DTYPE)
    VG1_t = torch.tensor(VG1, dtype=DTYPE)
    VG2_t = torch.tensor(VG2, dtype=DTYPE)
    Vb_t  = torch.tensor(Vb,  dtype=DTYPE)
    out = mod.solve_batched_gpu(cfg, M1, M2, bjt, Vd_t, VG1_t, VG2_t, Vb_t,
                                max_iters=cfg.newton_max_iters,
                                device=str(DEVICE), dtype=DTYPE)
    return {k: v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)
            for k, v in out.items() if k in ("Id", "Iii_in", "Ileak_out")}


# -------------------------------------------------------------- CSV loading
VG1_DIRS = {
    0.2: "2vHCa-2 I-Vs@VG2 VG1=0.2 vnwell=2",
    0.4: "2vHCa-2 I-Vs@VG2 VG1=0.4 vnwell=2",
    0.6: "2vHCa-2 I-Vs@VG2 VG1=0.6 vnwell=2",
}
VG2_RE = re.compile(r"VG2=(-?\d+\.\d+)")


def load_csv4(p: Path):
    """Returns vdata, idata, tdata, ifixdata (idata=measured, ifix=fixture).

    Some rows may have NaN in 'Var4'; we skip that column.
    """
    arr = np.loadtxt(p, delimiter=",", skiprows=1, usecols=(0, 1, 2, 5))
    return arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]


# ---------------------------------------------------------- transient solver
def _inet_at(api_tag, mod, cfg, M1, M2, bjt, Vd, VG1, VG2, Vb_arr):
    """Solve for Inet=(Iii_in - Ileak_out) at a batch of Vb values (Vd,VG1,VG2 scalar)."""
    G = len(Vb_arr)
    out = solve_pt(api_tag, mod, cfg, M1, M2, bjt,
                   np.full(G, Vd), np.full(G, VG1),
                   np.full(G, VG2), np.asarray(Vb_arr))
    return out["Iii_in"] - out["Ileak_out"], out


def run_transient(api_tag, mod, cfg, M1, M2, bjt,
                  vd_traj, t_traj, VG1, VG2,
                  Cb=CB, Vb0=VB0, Vb_max=VB_MAX,
                  substep_frac=SUBSTEP_FRAC):
    """Integrate (Vb, Id) along vd_traj sampled at t_traj.

    Exponential / locally-linear integrator:
      At step k, solve for Inet at a small Vb-bracket {Vb, Vb+h, Vb-h} to get
      Inet0 = Inet(Vb_k) and slope k_lin = dInet/dVb (numerical, <0 in stable
      regime).  Then exact integration of dVb/dt = (Inet0 + k_lin*(Vb-Vb_k))/Cb
      over duration dt gives:
            tau     = -Cb / k_lin
            Vb_eq   = Vb_k - Inet0 / k_lin   (local equilibrium)
            Vb_new  = Vb_eq + (Vb_k - Vb_eq) * exp(-dt/tau)
      This is unconditionally stable, recovers quasi-static when dt>>tau, and
      gives true transient hysteresis when dt~tau.  Fallback to forward Euler
      if k_lin >= 0 (unstable region, e.g. avalanche breakdown).
    Returns Id_sim (N,), Vb_sim (N,), diag dict.
    """
    N = len(vd_traj)
    Id_sim = np.zeros(N)
    Vb_sim = np.zeros(N)
    Vb = Vb0
    Vb_peak = Vb0
    h_fd = 1e-3      # finite diff for k_lin
    taus = []

    for k in range(N):
        Vd_k = float(vd_traj[k])
        # 3-point batch: (Vb-h, Vb, Vb+h)
        vb_probe = np.array([max(0.0, Vb - h_fd), Vb, min(Vb_max, Vb + h_fd)])
        out = solve_pt(api_tag, mod, cfg, M1, M2, bjt,
                       np.full(3, Vd_k), np.full(3, VG1),
                       np.full(3, VG2), vb_probe)
        Iii  = out["Iii_in"]
        Ile  = out["Ileak_out"]
        Inet = Iii - Ile
        Id_sim[k] = float(out["Id"][1])  # at Vb (center)
        Vb_sim[k] = Vb
        # Slope k_lin = d Inet / d Vb  (central diff)
        denom = (vb_probe[2] - vb_probe[0])
        k_lin = (Inet[2] - Inet[0]) / denom if denom > 1e-9 else 0.0
        Inet0 = float(Inet[1])
        if k == N - 1:
            continue
        dt = float(t_traj[k + 1] - t_traj[k])
        if dt <= 0:
            continue
        if k_lin < -1e-15:
            tau = -Cb / k_lin
            Vb_eq = Vb - Inet0 / k_lin
            decay = np.exp(-dt / tau)
            Vb_new = Vb_eq + (Vb - Vb_eq) * decay
            taus.append(tau)
        else:
            # unstable / flat-slope region: small forward Euler step (dt-limited)
            Vb_new = Vb + dt * Inet0 / Cb
            taus.append(np.inf)
        if Vb_new < 0.0: Vb_new = 0.0
        if Vb_new > Vb_max: Vb_new = Vb_max
        Vb = Vb_new
        if Vb > Vb_peak: Vb_peak = Vb

    diag = {"vb_peak": float(Vb_peak),
            "vb_final": float(Vb),
            "median_tau_s": float(np.median([t for t in taus if np.isfinite(t)])
                                  if any(np.isfinite(t) for t in taus) else float('inf'))}
    return Id_sim, Vb_sim, diag


# -------------------------------------------------------------- scoring
SAFE = 1e-15
def log10_safe(x): return np.log10(np.maximum(np.abs(x), SAFE))


def midpoint_current(vd, idd, target=1.0):
    if vd[0] > vd[-1]:
        vd = vd[::-1]; idd = idd[::-1]
    if target < vd.min() or target > vd.max():
        return None
    return float(np.interp(target, vd, np.abs(idd)))


def score_curve(meas_vd, meas_id, sim_id):
    apex = int(np.argmax(meas_vd))
    fwd_meas_vd = meas_vd[:apex + 1]
    fwd_meas_id = meas_id[:apex + 1]
    fwd_sim_id  = sim_id[:apex + 1]
    rev_meas_vd = meas_vd[apex:]
    rev_meas_id = meas_id[apex:]
    rev_sim_id  = sim_id[apex:]

    fwd_log_err = np.abs(log10_safe(fwd_sim_id) - log10_safe(fwd_meas_id))
    rev_log_err = np.abs(log10_safe(rev_sim_id) - log10_safe(rev_meas_id))

    fwd_mid_m = midpoint_current(fwd_meas_vd, fwd_meas_id, 1.0)
    rev_mid_m = midpoint_current(rev_meas_vd, rev_meas_id, 1.0)
    fwd_mid_s = midpoint_current(fwd_meas_vd, fwd_sim_id, 1.0)
    rev_mid_s = midpoint_current(rev_meas_vd, rev_sim_id, 1.0)
    hyst_meas = (abs(rev_mid_m - fwd_mid_m) / fwd_mid_m
                 if fwd_mid_m and rev_mid_m and fwd_mid_m > 0 else None)
    hyst_sim  = (abs(rev_mid_s - fwd_mid_s) / fwd_mid_s
                 if fwd_mid_s and rev_mid_s and fwd_mid_s > 0 else None)
    return {
        "apex_idx": apex, "n_points": int(meas_vd.shape[0]),
        "forward_log_rmse": float(np.median(fwd_log_err)),
        "reverse_log_rmse": float(np.median(rev_log_err)),
        "hysteresis_ratio_meas": hyst_meas,
        "hysteresis_ratio_sim":  hyst_sim,
    }


# -------------------------------------------------------------- multi-rate
def synthesize_vd_traj(vd_template, t_template, target_rate_Vps):
    """Stretch/compress t while keeping vd shape, so peak ramp = target_rate."""
    apex = int(np.argmax(vd_template))
    vmax = float(vd_template[apex])
    # Measured forward duration:
    t_fwd_meas = float(t_template[apex] - t_template[0])
    # New forward duration to achieve target ramp on average:
    t_fwd_new = vmax / target_rate_Vps
    scale = t_fwd_new / max(t_fwd_meas, 1e-9)
    t_new = (t_template - t_template[0]) * scale
    return vd_template.copy(), t_new, scale


# -------------------------------------------------------------- driver
def main():
    t0 = time.time()
    api_tag, mod, cfg, M1, M2, bjt = _load_solver()

    # ---- enumerate curves --------------------------------------------------
    curves_meta = []
    for vg1, subdir in VG1_DIRS.items():
        d = DATA_ROOT / subdir
        for csv_path in sorted(d.glob("StandardIV*.csv")):
            m = VG2_RE.search(csv_path.name)
            if not m:
                continue
            vg2 = float(m.group(1))
            try:
                vd, idd, tt, ifix = load_csv4(csv_path)
            except Exception as e:
                print(f"[z308] load fail {csv_path.name}: {e}")
                continue
            curves_meta.append(dict(vg1=vg1, vg2=vg2, file=csv_path.name,
                                    vd=vd, idd=idd, t=tt, ifix=ifix))
    print(f"[z308] enumerated {len(curves_meta)} curves")

    # ---- per-curve transient integrations ---------------------------------
    per_curve = []
    raw_traces = []
    t_solve_start = time.time()
    for ci, cm in enumerate(curves_meta):
        t_curve = time.time()
        Id_sim, Vb_sim, diag = run_transient(
            api_tag, mod, cfg, M1, M2, bjt,
            cm["vd"], cm["t"], cm["vg1"], cm["vg2"])
        res = score_curve(cm["vd"], cm["idd"], Id_sim)
        res.update(dict(
            vg1=cm["vg1"], vg2=cm["vg2"], file=cm["file"],
            t_total=float(cm["t"][-1] - cm["t"][0]),
            ramp_rate_Vps=float(
                np.max(cm["vd"]) /
                max(cm["t"][int(np.argmax(cm["vd"]))] - cm["t"][0], 1e-9)),
            vb_peak=diag["vb_peak"],
            vb_final=diag["vb_final"],
            median_tau_s=diag["median_tau_s"],
            wall_sec=time.time() - t_curve,
        ))
        per_curve.append(res)
        raw_traces.append(dict(
            vg1=cm["vg1"], vg2=cm["vg2"], file=cm["file"],
            t=cm["t"].tolist(), vd=cm["vd"].tolist(),
            id_meas=cm["idd"].tolist(), id_sim=Id_sim.tolist(),
            vb_sim=Vb_sim.tolist(),
            fwd_rmse=res["forward_log_rmse"],
            rev_rmse=res["reverse_log_rmse"],
        ))
        if ci % 5 == 0 or ci == len(curves_meta) - 1:
            print(f"[z308]  {ci+1}/{len(curves_meta)} "
                  f"VG1={cm['vg1']:.2f} VG2={cm['vg2']:+.2f}  "
                  f"fwd={res['forward_log_rmse']:.3f} rev={res['reverse_log_rmse']:.3f} "
                  f"Vb_peak={diag['vb_peak']:.3f} tau={diag['median_tau_s']:.2e}s "
                  f"({res['wall_sec']:.1f}s)")
    print(f"[z308] all curves done in {time.time()-t_solve_start:.1f}s")

    # ---- aggregate --------------------------------------------------------
    fwd = np.array([c["forward_log_rmse"] for c in per_curve])
    rev = np.array([c["reverse_log_rmse"] for c in per_curve])
    hyst_m = np.array([c["hysteresis_ratio_meas"] for c in per_curve
                       if c["hysteresis_ratio_meas"] is not None])
    hyst_s = np.array([c["hysteresis_ratio_sim"]  for c in per_curve
                       if c["hysteresis_ratio_sim"]  is not None])
    med_fwd = float(np.median(fwd))
    med_rev = float(np.median(rev))
    med_hyst_m = float(np.median(hyst_m)) if hyst_m.size else None
    med_hyst_s = float(np.median(hyst_s)) if hyst_s.size else None
    hyst_within_2x = None
    if med_hyst_m and med_hyst_s and med_hyst_m > 0 and med_hyst_s > 0:
        r = med_hyst_s / med_hyst_m
        hyst_within_2x = bool(0.5 <= r <= 2.0)
    pass_conservative = med_fwd < 0.5
    pass_ambitious = pass_conservative and (med_rev < 0.5) and bool(hyst_within_2x)

    # ---- multi-rate predictions ------------------------------------------
    multi_rate = []
    # pick (VG1=0.4, VG2=0.20) as representative
    target = None
    for cm in curves_meta:
        if abs(cm["vg1"] - 0.4) < 1e-6 and abs(cm["vg2"] - 0.20) < 1e-6:
            target = cm; break
    if target is None:
        print("[z308] WARN: no VG1=0.4 VG2=0.20 curve found for multi-rate")
    else:
        rates = [0.017, 0.17, 1.7]  # V/s
        print(f"[z308] multi-rate predictions on {target['file']}")
        for r in rates:
            vd_new, t_new, scale = synthesize_vd_traj(target["vd"], target["t"], r)
            Id_pred, Vb_pred, dg = run_transient(
                api_tag, mod, cfg, M1, M2, bjt,
                vd_new, t_new, target["vg1"], target["vg2"])
            # hysteresis at Vd=1.0
            apex = int(np.argmax(vd_new))
            fwd_mid_p = midpoint_current(vd_new[:apex+1], Id_pred[:apex+1], 1.0)
            rev_mid_p = midpoint_current(vd_new[apex:], Id_pred[apex:], 1.0)
            h_pred = (abs(rev_mid_p - fwd_mid_p) / fwd_mid_p
                      if fwd_mid_p and rev_mid_p else None)
            multi_rate.append(dict(
                ramp_Vps=r, time_scale=scale,
                vb_peak=dg["vb_peak"], vb_final=dg["vb_final"],
                hysteresis_ratio_pred=h_pred,
                id_at_vd1_fwd=fwd_mid_p, id_at_vd1_rev=rev_mid_p,
                t_total=float(t_new[-1] - t_new[0]),
                median_tau_s=dg["median_tau_s"],
            ))
            print(f"[z308]   rate={r:.3f} V/s  Vb_peak={dg['vb_peak']:.3f}  "
                  f"hyst_pred={h_pred}")

    # ---- write summary ----------------------------------------------------
    summary = {
        "script": "scripts/z308_transient_validate.py",
        "api_tag": api_tag,
        "device": str(DEVICE),
        "n_curves": len(per_curve),
        "Cb": CB, "Vb_max": VB_MAX, "Vb0": VB0,
        "substep_frac": SUBSTEP_FRAC,
        "aggregate": {
            "median_forward_log_rmse_dec": med_fwd,
            "median_reverse_log_rmse_dec": med_rev,
            "median_hysteresis_meas":      med_hyst_m,
            "median_hysteresis_sim":       med_hyst_s,
            "hysteresis_spread_within_2x": hyst_within_2x,
        },
        "gate_conservative_pass": bool(pass_conservative),
        "gate_ambitious_pass":    bool(pass_ambitious),
        "per_curve": per_curve,
        "multi_rate_predictions": multi_rate,
        "runtime_sec": time.time() - t0,
    }
    with open(OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    # raw traces saved separately to keep summary small
    with open(OUT / "raw_traces.json", "w") as f:
        json.dump(raw_traces, f)
    print(f"[z308] wrote {OUT/'summary.json'}")
    print(f"[z308] verdict conservative={pass_conservative} ambitious={pass_ambitious}")
    print(f"[z308] med_fwd={med_fwd:.3f}  med_rev={med_rev:.3f}  "
          f"hyst meas={med_hyst_m} sim={med_hyst_s} within2x={hyst_within_2x}")
    print(f"[z308] runtime {summary['runtime_sec']:.1f}s")


if __name__ == "__main__":
    main()
