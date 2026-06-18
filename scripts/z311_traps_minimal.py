"""z311 — Minimal multi-tau trap reservoir stub on top of z308 transient harness.

Hypothesis (slide 21): oxide/interface traps with a distributed spectrum of
time constants tau ~ µs → s reproduce the rate-dependent hysteresis loop shape
seen in measured NS-RAM IV curves. Without traps the current z308 model gives
hysteresis ratio ~2.2e-8 at 0.17 V/s — three to four orders below the
measured ~2.6e-3.

We add ONE simple, additive trap layer to the body node:

    dQ_i/dt = (Q_eq_i(V_B) - Q_i) / tau_i,        i in {fast, mid, slow}
    Q_eq_i(V_B) = Q_max_i * V_B / (V_B + V_half)            (Langmuir saturation)
    V_B_effective(t) = V_B(t) - sum_i Q_i(t) / C_b

with tau = (1e-3, 1e-2, 1e-1) s. The bare-substrate intrinsic Vb dynamics
from z308 are kept intact; traps simply shift the *effective* Vb seen by
the BJT/MOS solve at each timestep. This is the minimal scaffolding needed
to test whether trap memory alone can lift hysteresis by 3+ decades.

Locked gate: hysteresis_ratio_pred at 0.17 V/s > 1e-5  (vs z308 2.2e-8).
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

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data/sebas_2026_04_22"
OUT = ROOT / "results/z311_traps"
OUT.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device(os.environ.get("Z311_DEVICE", "cpu"))
DTYPE = torch.float64

# Physical defaults — keep z308 defaults for fair comparison
CB = float(os.environ.get("Z311_CB", "5e-15"))
VB_MAX = float(os.environ.get("Z311_VB_MAX", "0.80"))
VB0 = float(os.environ.get("Z311_VB0", "0.0"))

# Trap reservoir params (defaults aim for ~1e-3 hysteresis at 0.17 V/s)
TAUS = [float(x) for x in os.environ.get("Z311_TAUS", "1e-3,1e-2,1e-1").split(",")]
# Q_max per reservoir — proportionally split. Total ~ Cb * 0.3 V worth of charge
# so that full saturation can shift Vb_eff by up to ~0.3 V (Langmuir-bounded).
QMAX_TOT = float(os.environ.get("Z311_QMAX_TOT", str(CB * 0.30)))  # Coulombs
QMAX_SPLIT = [float(x) for x in os.environ.get(
    "Z311_QMAX_SPLIT", "0.333,0.333,0.333").split(",")]
V_HALF = float(os.environ.get("Z311_V_HALF", "0.20"))             # Langmuir half-V

print(f"[z311] device={DEVICE} Cb={CB:.2e} taus={TAUS} Qmax_tot={QMAX_TOT:.2e}")


# ---------------------------------------------------------------- pyport loader
def _load_solver():
    print(f"[z311] using z294 MEP-7 pyport (Newton on Vsint @ fixed Vb)")
    sp = importlib.util.spec_from_file_location("mep7", ROOT / "scripts/z294_mep7_gpu_pyport.py")
    mep7 = importlib.util.module_from_spec(sp); sp.loader.exec_module(mep7)
    ns4d = mep7._load_cpu_ref()
    cfg, M1, M2, bjt = ns4d._build_pyport_models()
    tag = "v1"
    if os.environ.get("Z311_USE_V2", "1") != "0":
        try:
            sys.path.insert(0, str(ROOT / "src"))
            from nsram_pyport_v2 import V2Params, enable_v2_topology
            enable_v2_topology(cfg, V2Params())
            tag = "v2"
            print(f"[z311] v2 topology ENABLED")
        except Exception as e:
            print(f"[z311] v2 unavailable ({e}); v1 fallback")
    return (tag, mep7, cfg, M1, M2, bjt)


def solve_pt(mod, cfg, M1, M2, bjt, Vd, VG1, VG2, Vb):
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
    arr = np.loadtxt(p, delimiter=",", skiprows=1, usecols=(0, 1, 2, 5))
    return arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]


# ---------------------------------------------------------- transient solver
def run_transient_traps(mod, cfg, M1, M2, bjt,
                        vd_traj, t_traj, VG1, VG2,
                        Cb=CB, Vb0=VB0, Vb_max=VB_MAX,
                        taus=TAUS, qmax_tot=QMAX_TOT,
                        qmax_split=QMAX_SPLIT, v_half=V_HALF):
    """Same exponential integrator as z308 but with parallel trap reservoirs
    feeding back as Vb_eff = Vb - sum(Q_i)/Cb in the solve.

    Trap update (exact for piecewise-constant V_B over dt):
        Q_eq_i = Qmax_i * Vb / (Vb + V_half)        (Langmuir)
        Q_i_new = Q_eq_i + (Q_i - Q_eq_i) * exp(-dt/tau_i)
    Then Vb_eff = Vb - sum(Q_i)/Cb for the BJT/MOS solve.
    """
    N = len(vd_traj)
    Id_sim = np.zeros(N)
    Vb_sim = np.zeros(N)
    Vbeff_sim = np.zeros(N)
    Qsum_sim = np.zeros(N)
    Vb = float(Vb0)
    Vb_peak = float(Vb0)
    h_fd = 1e-3
    n_tau = len(taus)
    Q = np.zeros(n_tau)
    Qmax = qmax_tot * np.asarray(qmax_split, dtype=float)
    taus_a = np.asarray(taus, dtype=float)

    for k in range(N):
        Vd_k = float(vd_traj[k])
        Qsum = float(Q.sum())
        Vb_eff = max(0.0, min(Vb_max, Vb - Qsum / Cb))
        Vb_sim[k] = Vb
        Vbeff_sim[k] = Vb_eff
        Qsum_sim[k] = Qsum
        # 3-pt bracket on Vb_eff for slope estimate
        vb_probe = np.array([max(0.0, Vb_eff - h_fd), Vb_eff,
                             min(Vb_max, Vb_eff + h_fd)])
        out = solve_pt(mod, cfg, M1, M2, bjt,
                       np.full(3, Vd_k), np.full(3, VG1),
                       np.full(3, VG2), vb_probe)
        Iii  = out["Iii_in"]
        Ile  = out["Ileak_out"]
        Inet = Iii - Ile
        Id_sim[k] = float(out["Id"][1])
        denom = (vb_probe[2] - vb_probe[0])
        k_lin = (Inet[2] - Inet[0]) / denom if denom > 1e-9 else 0.0
        Inet0 = float(Inet[1])
        if k == N - 1:
            continue
        dt = float(t_traj[k + 1] - t_traj[k])
        if dt <= 0:
            continue
        # ---- intrinsic Vb update (same as z308, but driven by Vb_eff slope --
        #      we keep dynamics referenced to Vb, since traps only shift Vb_eff
        #      in the solve; this is the minimal-coupling stub).
        if k_lin < -1e-15:
            tau = -Cb / k_lin
            Vb_eq = Vb - Inet0 / k_lin
            decay = np.exp(-dt / tau)
            Vb_new = Vb_eq + (Vb - Vb_eq) * decay
        else:
            Vb_new = Vb + dt * Inet0 / Cb
        if Vb_new < 0.0: Vb_new = 0.0
        if Vb_new > Vb_max: Vb_new = Vb_max
        Vb = Vb_new
        if Vb > Vb_peak: Vb_peak = Vb
        # ---- trap update (uses NEW Vb as drive for next step) ---------------
        denom_q = (Vb + v_half) if (Vb + v_half) > 1e-9 else 1e-9
        Q_eq = Qmax * (Vb / denom_q)
        Q = Q_eq + (Q - Q_eq) * np.exp(-dt / taus_a)

    diag = {"vb_peak": float(Vb_peak),
            "vb_final": float(Vb),
            "qsum_final": float(Q.sum()),
            "qsum_peak": float(np.max(Qsum_sim)),
            "vbeff_peak": float(np.max(Vbeff_sim))}
    return Id_sim, Vb_sim, Vbeff_sim, diag


# -------------------------------------------------------------- multi-rate
def synthesize_vd_traj(vd_template, t_template, target_rate_Vps):
    apex = int(np.argmax(vd_template))
    vmax = float(vd_template[apex])
    t_fwd_meas = float(t_template[apex] - t_template[0])
    t_fwd_new = vmax / target_rate_Vps
    scale = t_fwd_new / max(t_fwd_meas, 1e-9)
    t_new = (t_template - t_template[0]) * scale
    return vd_template.copy(), t_new, scale


def midpoint_current(vd, idd, target=1.0):
    if vd[0] > vd[-1]:
        vd = vd[::-1]; idd = idd[::-1]
    if target < vd.min() or target > vd.max():
        return None
    return float(np.interp(target, vd, np.abs(idd)))


def main():
    t0 = time.time()
    api_tag, mod, cfg, M1, M2, bjt = _load_solver()

    # pick the representative curve (VG1=0.4, VG2=0.20) — same as z308
    target = None
    for vg1, subdir in VG1_DIRS.items():
        d = DATA_ROOT / subdir
        for p in sorted(d.glob("StandardIV*.csv")):
            m = VG2_RE.search(p.name)
            if not m:
                continue
            vg2 = float(m.group(1))
            if abs(vg1 - 0.4) < 1e-6 and abs(vg2 - 0.20) < 1e-6:
                vd, idd, tt, ifix = load_csv4(p)
                target = dict(vg1=vg1, vg2=vg2, file=p.name,
                              vd=vd, idd=idd, t=tt)
                break
        if target is not None:
            break
    if target is None:
        raise RuntimeError("no VG1=0.4 VG2=0.20 curve found")
    print(f"[z311] template curve: {target['file']}  N={len(target['vd'])}")

    rates = [0.017, 0.17, 1.7]
    multi_rate = []
    for r in rates:
        vd_new, t_new, scale = synthesize_vd_traj(target["vd"], target["t"], r)
        Id_pred, Vb_pred, Vbeff_pred, dg = run_transient_traps(
            mod, cfg, M1, M2, bjt,
            vd_new, t_new, target["vg1"], target["vg2"])
        apex = int(np.argmax(vd_new))
        fwd_mid = midpoint_current(vd_new[:apex+1], Id_pred[:apex+1], 1.0)
        rev_mid = midpoint_current(vd_new[apex:], Id_pred[apex:], 1.0)
        h = (abs(rev_mid - fwd_mid) / fwd_mid
             if fwd_mid and rev_mid and fwd_mid > 0 else None)
        multi_rate.append(dict(
            ramp_Vps=r, time_scale=scale,
            vb_peak=dg["vb_peak"], vb_final=dg["vb_final"],
            vbeff_peak=dg["vbeff_peak"],
            qsum_peak=dg["qsum_peak"], qsum_final=dg["qsum_final"],
            hysteresis_ratio_pred=h,
            id_at_vd1_fwd=fwd_mid, id_at_vd1_rev=rev_mid,
            t_total=float(t_new[-1] - t_new[0]),
        ))
        print(f"[z311] rate={r:.3f}V/s  Vb_peak={dg['vb_peak']:.3f}  "
              f"Vbeff_peak={dg['vbeff_peak']:.3f}  Qsum_peak={dg['qsum_peak']:.2e}  "
              f"hyst={h}")

    # gate eval (0.17 V/s middle rate)
    mid = next(m for m in multi_rate if abs(m["ramp_Vps"] - 0.17) < 1e-6)
    h_mid = mid["hysteresis_ratio_pred"] or 0.0
    gate_pass = bool(h_mid > 1e-5)
    z308_hyst_017 = 2.2e-8
    meas_hyst_017 = 2.6e-3
    improvement_x = (h_mid / z308_hyst_017) if h_mid and z308_hyst_017 else None
    distance_to_meas_x = (meas_hyst_017 / h_mid) if h_mid > 0 else None

    summary = {
        "script": "scripts/z311_traps_minimal.py",
        "api_tag": api_tag,
        "device": str(DEVICE),
        "Cb": CB, "Vb_max": VB_MAX, "Vb0": VB0,
        "taus_s": TAUS, "Qmax_tot_C": QMAX_TOT,
        "Qmax_split": QMAX_SPLIT, "V_half": V_HALF,
        "template_file": target["file"],
        "multi_rate_predictions": multi_rate,
        "gate": {
            "locked_gate_pass": gate_pass,
            "hyst_at_0p17Vps": h_mid,
            "threshold": 1e-5,
            "z308_baseline_hyst_0p17": z308_hyst_017,
            "measured_hyst_0p17_approx": meas_hyst_017,
            "improvement_over_z308_x": improvement_x,
            "distance_to_measured_x": distance_to_meas_x,
        },
        "runtime_sec": time.time() - t0,
    }
    with open(OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[z311] wrote {OUT/'summary.json'}")
    print(f"[z311] gate_pass={gate_pass}  h(0.17V/s)={h_mid:.3e}  "
          f"improvement_over_z308={improvement_x}x  to_measured={distance_to_meas_x}x")
    print(f"[z311] runtime {summary['runtime_sec']:.1f}s")


if __name__ == "__main__":
    main()
