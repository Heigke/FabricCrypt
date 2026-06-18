"""z298 — Sebas transient replay.

Reads the FULL Sebas 130nm I-V CSVs (vdata, idata, tdata, ...), groups by
(VG1, VG2), and reruns the pyport surrogate as a transient sim driven by a
V_d(t) ramp at the SAME ramp rate that the measurement used.

Forward leg (0 -> 2 V) and reverse leg (2 V -> 0) are scored separately.

Uses the z271/z273 4-D surrogate (Id, Iii, Ileak; axes vg1, vg2, vd, vb) and
the simulate_ramp body-state ODE from z273.

Gates:
    PASS-conservative : median forward log-RMSE < 0.5 dec
    AMBITIOUS         : + median reverse log-RMSE < 0.5 dec
                        + hysteresis spread (sim vs meas) within 2x

Output: results/z298_sebas_transient/summary.json
"""
from __future__ import annotations

import json
import os
import re
import time
from glob import glob
from pathlib import Path

os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")

import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
SURR_PATH = ROOT / "results/z271_pmp3_dense_surrogate/surrogate_4d_v2.npz"
DATA_ROOT = ROOT / "data/sebas_2026_04_22"
OUT = ROOT / "results/z298_sebas_transient"
OUT.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64  # need precision for log-currents over many decades
print(f"[z298] device={DEVICE} dtype={DTYPE}")


# ------------------- surrogate I/O (same as z273) ----------------------------
def load_surrogate():
    d = np.load(SURR_PATH)
    Id = torch.tensor(d["Id"], dtype=DTYPE, device=DEVICE)
    Iii = torch.tensor(d["Iii"], dtype=DTYPE, device=DEVICE)
    Ileak = torch.tensor(d["Ileak"], dtype=DTYPE, device=DEVICE)
    axes = {
        "vg1": torch.tensor(d["vg1_axis"], dtype=DTYPE, device=DEVICE),
        "vg2": torch.tensor(d["vg2_axis"], dtype=DTYPE, device=DEVICE),
        "vd":  torch.tensor(d["vd_axis"],  dtype=DTYPE, device=DEVICE),
        "vb":  torch.tensor(d["vb_axis"],  dtype=DTYPE, device=DEVICE),
    }
    return {"Id": Id, "Iii": Iii, "Ileak": Ileak, "axes": axes}


def _idx_frac(x, axis):
    n = axis.shape[0]
    idx = torch.searchsorted(axis, x.contiguous())
    idx = torch.clamp(idx, 1, n - 1)
    i0 = idx - 1
    i1 = idx
    a0 = axis[i0]; a1 = axis[i1]
    w = (x - a0) / (a1 - a0 + 1e-30)
    w = torch.clamp(w, 0.0, 1.0)  # clamp at boundary => edge-extrap (nearest)
    return i0, i1, w


def interp4d(table, axes, vg1, vg2, vd, vb):
    i0g1, i1g1, wg1 = _idx_frac(vg1, axes["vg1"])
    i0g2, i1g2, wg2 = _idx_frac(vg2, axes["vg2"])
    i0d,  i1d,  wd  = _idx_frac(vd,  axes["vd"])
    i0b,  i1b,  wb  = _idx_frac(vb,  axes["vb"])

    out = 0.0
    for sg1, ig1 in ((1 - wg1, i0g1), (wg1, i1g1)):
        for sg2, ig2 in ((1 - wg2, i0g2), (wg2, i1g2)):
            for sd, idv in ((1 - wd, i0d), (wd, i1d)):
                for sb, ib in ((1 - wb, i0b), (wb, i1b)):
                    out = out + sg1 * sg2 * sd * sb * table[ig1, ig2, idv, ib]
    return out


# ------------------- transient solver ----------------------------------------
def simulate_arbitrary_vd(surr, VG1, VG2, t_arr, Vd_arr, C_b_F=1e-15):
    """Forward-Euler body ODE driven by externally supplied V_d(t).

    t_arr, Vd_arr: 1-D torch tensors on DEVICE, same length, monotonic time.

    Returns: I_d(t), V_b(t).
    """
    n = t_arr.shape[0]
    V_b = torch.zeros(n, dtype=DTYPE, device=DEVICE)
    I_d = torch.zeros(n, dtype=DTYPE, device=DEVICE)
    vb = torch.tensor(0.0, dtype=DTYPE, device=DEVICE)

    vg1_t = torch.full((1,), float(VG1), dtype=DTYPE, device=DEVICE)
    vg2_t = torch.full((1,), float(VG2), dtype=DTYPE, device=DEVICE)

    for k in range(n):
        V_b[k] = vb
        vd_k = Vd_arr[k:k + 1]
        vb_k = vb.unsqueeze(0)
        id_k  = interp4d(surr["Id"],   surr["axes"], vg1_t, vg2_t, vd_k, vb_k)
        iii_k = interp4d(surr["Iii"],  surr["axes"], vg1_t, vg2_t, vd_k, vb_k)
        ile_k = interp4d(surr["Ileak"], surr["axes"], vg1_t, vg2_t, vd_k, vb_k)
        I_d[k] = id_k
        if k + 1 < n:
            dt = (t_arr[k + 1] - t_arr[k]).item()
            dvb = (iii_k - ile_k) / C_b_F
            vb = vb + dvb.squeeze() * dt
            if not torch.isfinite(vb):
                vb = torch.tensor(0.0, dtype=DTYPE, device=DEVICE)
            # clamp to surrogate vb-range so interp stays in-grid (nearest edge)
            vb = torch.clamp(vb,
                             surr["axes"]["vb"].min(),
                             surr["axes"]["vb"].max())
    return I_d.cpu().numpy(), V_b.cpu().numpy()


# ------------------- data loading --------------------------------------------
VG1_DIRS = {
    0.2: "2vHCa-2 I-Vs@VG2 VG1=0.2 vnwell=2",
    0.4: "2vHCa-2 I-Vs@VG2 VG1=0.4 vnwell=2",
    0.6: "2vHCa-2 I-Vs@VG2 VG1=0.6 vnwell=2",
}
VG2_RE = re.compile(r"VG2=(-?\d+\.\d+)")


def load_one(csv_path: Path):
    arr = np.loadtxt(csv_path, delimiter=",", skiprows=1,
                     usecols=(0, 1, 2, 4, 5))
    vd = arr[:, 0]; idd = arr[:, 1]; t = arr[:, 2]
    vfixg = arr[:, 3]; ifixg = arr[:, 4]  # gate leakage telemetry (unused)
    return vd, idd, t


def split_forward_reverse(vd, t):
    """Apex index = argmax(vd)."""
    apex = int(np.argmax(vd))
    return apex


# ------------------- per-curve metrics ---------------------------------------
SAFE = 1e-15  # current floor for log


def log10_safe(x):
    return np.log10(np.maximum(np.abs(x), SAFE))


def midpoint_current(vd, idd, target=1.0):
    """Linear-interp |I| at vd=target."""
    if vd[0] > vd[-1]:
        vd = vd[::-1]; idd = idd[::-1]
    if target < vd.min() or target > vd.max():
        return None
    return float(np.interp(target, vd, np.abs(idd)))


def score_curve(meas_vd, meas_id, meas_t, sim_id, sim_vd):
    apex = int(np.argmax(meas_vd))
    # forward = 0..apex inclusive, reverse = apex..end
    fwd_meas_vd = meas_vd[:apex + 1]
    fwd_meas_id = meas_id[:apex + 1]
    fwd_sim_id  = sim_id[:apex + 1]
    rev_meas_vd = meas_vd[apex:]
    rev_meas_id = meas_id[apex:]
    rev_sim_id  = sim_id[apex:]

    fwd_log_err = np.abs(log10_safe(fwd_sim_id) - log10_safe(fwd_meas_id))
    rev_log_err = np.abs(log10_safe(rev_sim_id) - log10_safe(rev_meas_id))

    fwd_median = float(np.median(fwd_log_err))
    rev_median = float(np.median(rev_log_err))

    # hysteresis ratio: |I(rev)-I(fwd)|/I(fwd) at vd=1.0 V
    fwd_mid_m = midpoint_current(fwd_meas_vd, fwd_meas_id, 1.0)
    rev_mid_m = midpoint_current(rev_meas_vd, rev_meas_id, 1.0)
    fwd_mid_s = midpoint_current(fwd_meas_vd, fwd_sim_id, 1.0)
    rev_mid_s = midpoint_current(rev_meas_vd, rev_sim_id, 1.0)
    if fwd_mid_m and rev_mid_m and fwd_mid_m > 0:
        hyst_meas = abs(rev_mid_m - fwd_mid_m) / fwd_mid_m
    else:
        hyst_meas = None
    if fwd_mid_s and rev_mid_s and fwd_mid_s > 0:
        hyst_sim = abs(rev_mid_s - fwd_mid_s) / fwd_mid_s
    else:
        hyst_sim = None

    return {
        "apex_idx": apex,
        "n_points": int(meas_vd.shape[0]),
        "forward_log_rmse": fwd_median,
        "reverse_log_rmse": rev_median,
        "hysteresis_ratio_meas": hyst_meas,
        "hysteresis_ratio_sim":  hyst_sim,
    }


# ------------------- driver --------------------------------------------------
def main():
    t0 = time.time()
    surr = load_surrogate()
    print(f"[z298] surrogate Id shape={tuple(surr['Id'].shape)}, "
          f"vb=[{surr['axes']['vb'].min():.2f},{surr['axes']['vb'].max():.2f}]")

    curves = []
    raw_traces = []  # for top-3 best/worst
    for vg1, subdir in VG1_DIRS.items():
        d = DATA_ROOT / subdir
        for csv_path in sorted(d.glob("StandardIV*.csv")):
            m = VG2_RE.search(csv_path.name)
            if not m:
                continue
            vg2 = float(m.group(1))
            try:
                meas_vd, meas_id, meas_t = load_one(csv_path)
            except Exception as e:
                print(f"[z298] load fail {csv_path.name}: {e}")
                continue

            # Build sim time/Vd arrays = measured timestamps & V_d
            t_arr  = torch.tensor(meas_t,  dtype=DTYPE, device=DEVICE)
            vd_arr = torch.tensor(meas_vd, dtype=DTYPE, device=DEVICE)

            sim_id, sim_vb = simulate_arbitrary_vd(
                surr, vg1, vg2, t_arr, vd_arr, C_b_F=1e-15
            )

            res = score_curve(meas_vd, meas_id, meas_t, sim_id, meas_vd)
            res["vg1"] = vg1
            res["vg2"] = vg2
            res["file"] = csv_path.name
            res["t_total"] = float(meas_t[-1] - meas_t[0])
            res["ramp_rate_Vps"] = float(2.0 / (meas_t[np.argmax(meas_vd)] - meas_t[0]))
            curves.append(res)
            raw_traces.append({
                "vg1": vg1, "vg2": vg2, "file": csv_path.name,
                "t":  meas_t.tolist(),
                "vd": meas_vd.tolist(),
                "id_meas": meas_id.tolist(),
                "id_sim":  sim_id.tolist(),
                "vb_sim":  sim_vb.tolist(),
                "fwd_rmse": res["forward_log_rmse"],
                "rev_rmse": res["reverse_log_rmse"],
            })

    print(f"[z298] processed {len(curves)} curves")

    # ----- aggregate -----
    fwd = np.array([c["forward_log_rmse"] for c in curves])
    rev = np.array([c["reverse_log_rmse"] for c in curves])
    hyst_m = np.array([c["hysteresis_ratio_meas"] for c in curves
                       if c["hysteresis_ratio_meas"] is not None])
    hyst_s = np.array([c["hysteresis_ratio_sim"]  for c in curves
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

    # sort traces
    raw_traces_sorted = sorted(raw_traces, key=lambda r: r["fwd_rmse"])
    top3_best = raw_traces_sorted[:3]
    top3_worst = raw_traces_sorted[-3:]

    # bias: signed median (sim - meas) in log decades on forward leg
    sign_dec = []
    for tr in raw_traces:
        meas_id = np.asarray(tr["id_meas"])
        sim_id  = np.asarray(tr["id_sim"])
        apex = int(np.argmax(tr["vd"]))
        d = log10_safe(sim_id[:apex + 1]) - log10_safe(meas_id[:apex + 1])
        sign_dec.append(float(np.median(d)))
    median_signed_bias_dec = float(np.median(sign_dec))

    summary = {
        "script": "scripts/z298_sebas_transient_replay.py",
        "device": str(DEVICE),
        "n_curves": len(curves),
        "vg1_set": sorted({c["vg1"] for c in curves}),
        "vg2_min": float(min(c["vg2"] for c in curves)),
        "vg2_max": float(max(c["vg2"] for c in curves)),
        "aggregate": {
            "median_forward_log_rmse_dec": med_fwd,
            "median_reverse_log_rmse_dec": med_rev,
            "median_hysteresis_meas":      med_hyst_m,
            "median_hysteresis_sim":       med_hyst_s,
            "hysteresis_spread_within_2x": hyst_within_2x,
            "median_signed_bias_dec_forward": median_signed_bias_dec,
        },
        "gate_conservative_pass": bool(pass_conservative),
        "gate_ambitious_pass":    bool(pass_ambitious),
        "per_curve": curves,
        "top3_best": top3_best,
        "top3_worst": top3_worst,
        "runtime_sec": time.time() - t0,
    }

    with open(OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[z298] wrote {OUT/'summary.json'}")
    print(f"[z298] verdict conservative={pass_conservative} "
          f"ambitious={pass_ambitious}")
    print(f"[z298] med_fwd={med_fwd:.3f} dec, med_rev={med_rev:.3f} dec, "
          f"bias={median_signed_bias_dec:+.3f} dec")
    print(f"[z298] hyst meas={med_hyst_m}  sim={med_hyst_s}  "
          f"within2x={hyst_within_2x}")
    print(f"[z298] runtime {summary['runtime_sec']:.1f} s")


if __name__ == "__main__":
    main()
