"""z273 / PMP-9 — GPU-native batched body-state transient simulator.

Validates against Sebas slide-21 ramp measurements.

Surrogate: results/z271_pmp3_dense_surrogate/surrogate_4d_v2.npz
  Id, Iii, Ileak on grid (V_G1, V_G2, V_d, V_b).
  Ileak is "into-body" net junction current and already absorbs
  the parasitic N-well-junction term at the surrogate's implicit
  V_Nwell baseline (per pyport body_charge_ode_bsim4_full).

ODE (forward Euler, fixed dt):
  dV_b/dt = (Iii - Ileak) / C_b
  V_d(t)  = triangular pulse (0 -> V_set over t_rise, hold t_set, V_set -> 0 over t_fall)
  I_d(t)  = surrogate Id at (V_G1, V_G2, V_d(t), V_b(t))

All currents evaluated via 4-D linear interpolation on the surrogate grid
(torch.nn.functional.grid_sample with mode='bilinear' is 5-D, so we use a
hand-rolled 4-linear interp; the grid is small enough that this is fast).

Outputs:
  results/z273_pmp9/slide21_replication.png
  results/z273_pmp9/trise_sweep.png
  results/z273_pmp9/cb_sweep.png
  results/z273_pmp9/summary.json
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")

import numpy as np  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

ROOT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
SURR = ROOT / "results/z271_pmp3_dense_surrogate/surrogate_4d_v2.npz"
OUT = ROOT / "results/z273_pmp9"
OUT.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32
print(f"[z273] device={DEVICE} dtype={DTYPE}")


# ─────────────────────────── surrogate I/O ────────────────────────────
def load_surrogate():
    d = np.load(SURR)
    Id = torch.tensor(d["Id"], dtype=DTYPE, device=DEVICE)
    Iii = torch.tensor(d["Iii"], dtype=DTYPE, device=DEVICE)
    Ileak = torch.tensor(d["Ileak"], dtype=DTYPE, device=DEVICE)
    axes = {
        "vg1": torch.tensor(d["vg1_axis"], dtype=DTYPE, device=DEVICE),
        "vg2": torch.tensor(d["vg2_axis"], dtype=DTYPE, device=DEVICE),
        "vd": torch.tensor(d["vd_axis"], dtype=DTYPE, device=DEVICE),
        "vb": torch.tensor(d["vb_axis"], dtype=DTYPE, device=DEVICE),
    }
    print(f"[z273] surrogate Id shape={tuple(Id.shape)} "
          f"vg1=[{axes['vg1'].min():.2f},{axes['vg1'].max():.2f}] "
          f"vg2=[{axes['vg2'].min():.2f},{axes['vg2'].max():.2f}] "
          f"vd=[{axes['vd'].min():.2f},{axes['vd'].max():.2f}] "
          f"vb=[{axes['vb'].min():.2f},{axes['vb'].max():.2f}]")
    return {"Id": Id, "Iii": Iii, "Ileak": Ileak, "axes": axes}


# ─────────────────────────── 4-D linear interp ────────────────────────
def _idx_frac(x: torch.Tensor, axis: torch.Tensor):
    """Return (i0, i1, w) so val ≈ (1-w)*tbl[i0] + w*tbl[i1]. Clamped."""
    n = axis.shape[0]
    # locate
    idx = torch.searchsorted(axis, x.contiguous())
    idx = torch.clamp(idx, 1, n - 1)
    i0 = idx - 1
    i1 = idx
    a0 = axis[i0]
    a1 = axis[i1]
    w = (x - a0) / (a1 - a0 + 1e-30)
    w = torch.clamp(w, 0.0, 1.0)  # clamp at boundary => nearest-edge extrap
    return i0, i1, w


def interp4d(table: torch.Tensor,
             axes: dict[str, torch.Tensor],
             vg1, vg2, vd, vb):
    """Linear 4-D interp. Inputs are broadcastable 1-D batches."""
    i0_g1, i1_g1, w_g1 = _idx_frac(vg1, axes["vg1"])
    i0_g2, i1_g2, w_g2 = _idx_frac(vg2, axes["vg2"])
    i0_d, i1_d, w_d = _idx_frac(vd, axes["vd"])
    i0_b, i1_b, w_b = _idx_frac(vb, axes["vb"])

    def g(a, b, c, e):
        # gather table at composite indices (broadcast)
        return table[a, b, c, e]

    out = 0.0
    for sg1, ig1, mg1 in ((1 - w_g1, i0_g1, 0), (w_g1, i1_g1, 1)):
        for sg2, ig2, mg2 in ((1 - w_g2, i0_g2, 0), (w_g2, i1_g2, 1)):
            for sd, id_, md in ((1 - w_d, i0_d, 0), (w_d, i1_d, 1)):
                for sb, ib, mb in ((1 - w_b, i0_b, 0), (w_b, i1_b, 1)):
                    out = out + sg1 * sg2 * sd * sb * g(ig1, ig2, id_, ib)
    return out


# ─────────────────────────── V_d profile ──────────────────────────────
def triangle_vd(t: torch.Tensor, V_set: float,
                t_rise: float, t_set: float, t_fall: float) -> torch.Tensor:
    """Triangular pulse: 0 → V_set over t_rise, hold t_set, → 0 over t_fall."""
    vd = torch.zeros_like(t)
    r = t < t_rise
    h = (t >= t_rise) & (t < t_rise + t_set)
    f = (t >= t_rise + t_set) & (t < t_rise + t_set + t_fall)
    vd = torch.where(r, V_set * (t / t_rise), vd)
    vd = torch.where(h, torch.full_like(vd, V_set), vd)
    vd = torch.where(f,
                     V_set * (1.0 - (t - t_rise - t_set) / t_fall),
                     vd)
    return vd


# ─────────────────────────── simulate ramp ────────────────────────────
def simulate_ramp(surr,
                  VG1: float, VG2: float,
                  V_set: float, t_rise: float, t_set: float, t_fall: float,
                  C_b_F: float,
                  dt_sim: float | None = None,
                  pad_factor: float = 0.3) -> dict:
    """Forward-Euler body-state transient. Returns dict of GPU tensors."""
    if dt_sim is None:
        dt_sim = min(t_rise, t_fall) / 200.0  # 200 pts per ramp
    T_total = t_rise + t_set + t_fall + pad_factor * t_rise
    n_steps = int(np.ceil(T_total / dt_sim)) + 1
    t = torch.arange(n_steps, dtype=DTYPE, device=DEVICE) * dt_sim
    V_d = triangle_vd(t, V_set, t_rise, t_set, t_fall)

    vg1_t = torch.full_like(t, VG1)
    vg2_t = torch.full_like(t, VG2)

    V_b = torch.zeros_like(t)
    I_d = torch.zeros_like(t)
    I_ii = torch.zeros_like(t)
    I_leak = torch.zeros_like(t)
    railed = False
    vb = torch.tensor(0.0, dtype=DTYPE, device=DEVICE)

    vb_min = surr["axes"]["vb"].min().item()
    vb_max = surr["axes"]["vb"].max().item()

    # batched per-step Euler (scalar-time; surrogate eval is dirt cheap)
    for k in range(n_steps):
        V_b[k] = vb
        vd_k = V_d[k:k + 1]
        vb_k = vb.unsqueeze(0)
        vg1_k = vg1_t[k:k + 1]
        vg2_k = vg2_t[k:k + 1]
        id_k = interp4d(surr["Id"], surr["axes"], vg1_k, vg2_k, vd_k, vb_k)
        iii_k = interp4d(surr["Iii"], surr["axes"], vg1_k, vg2_k, vd_k, vb_k)
        ile_k = interp4d(surr["Ileak"], surr["axes"], vg1_k, vg2_k, vd_k, vb_k)
        I_d[k] = id_k
        I_ii[k] = iii_k
        I_leak[k] = ile_k
        if k + 1 < n_steps:
            dvb = (iii_k - ile_k) / C_b_F
            vb = vb + dvb.squeeze() * dt_sim
            if not torch.isfinite(vb):
                railed = True
                break
            if vb.item() < vb_min - 0.05 or vb.item() > vb_max + 0.05:
                railed = True
                # clamp into surrogate (keep simulation alive for diagnostic)
                vb = torch.clamp(vb, vb_min, vb_max)

    return {
        "t": t, "V_d": V_d, "V_b": V_b, "I_d": I_d,
        "I_ii": I_ii, "I_leak": I_leak,
        "dt": dt_sim, "n_steps": n_steps,
        "railed": railed,
        "params": dict(VG1=VG1, VG2=VG2, V_set=V_set, t_rise=t_rise,
                       t_set=t_set, t_fall=t_fall, C_b_F=C_b_F),
    }


def extract_slopes(res: dict) -> dict:
    """S_fire = max dI_d/dt during rising edge; S_relax = max |dI_d/dt|
    during falling edge (sign convention: returned as positive A/s)."""
    t = res["t"]
    I = res["I_d"]
    p = res["params"]
    dI = torch.diff(I)
    dt = torch.diff(t)
    slope = dI / dt
    t_mid = 0.5 * (t[:-1] + t[1:])
    rise_mask = t_mid < p["t_rise"]
    fall_mask = (t_mid >= p["t_rise"] + p["t_set"]) & \
                (t_mid < p["t_rise"] + p["t_set"] + p["t_fall"])
    if rise_mask.any():
        s_fire = slope[rise_mask].max().item()
    else:
        s_fire = float("nan")
    if fall_mask.any():
        s_relax = (-slope[fall_mask]).max().item()  # falling -> negative slope
    else:
        s_relax = float("nan")
    return dict(
        S_fire_A_per_s=s_fire,
        S_relax_A_per_s=s_relax,
        max_I_d_A=I.max().item(),
        V_b_peak_V=res["V_b"].max().item(),
        V_b_final_V=res["V_b"][-1].item(),
        railed=bool(res["railed"]),
    )


# ─────────────────────────── runner ───────────────────────────────────
def main():
    t0 = time.time()
    surr = load_surrogate()

    # 1) Slide-21 replication
    print("[z273] (1) slide-21 replication run")
    slide21 = simulate_ramp(surr,
                            VG1=0.45, VG2=0.30,
                            V_set=2.05,
                            t_rise=200e-6, t_set=1e-6, t_fall=200e-6,
                            C_b_F=14e-15)
    s21 = extract_slopes(slide21)
    print(f"    S_fire={s21['S_fire_A_per_s']:.3e} A/s  "
          f"S_relax={s21['S_relax_A_per_s']:.3e} A/s  "
          f"V_b_peak={s21['V_b_peak_V']:.3f} V  railed={s21['railed']}")

    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    tt = slide21["t"].cpu().numpy() * 1e6  # µs
    axes[0].plot(tt, slide21["V_d"].cpu().numpy(), "tab:blue")
    axes[0].set_ylabel("V_d  [V]")
    axes[0].set_title("Slide-21 condition: V_G1=0.45, V_G2=0.30, "
                      "V_set=2.05 V, t_rise=t_fall=200 µs, t_set=1 µs, "
                      "C_b=14 fF")
    axes[1].plot(tt, slide21["V_b"].cpu().numpy(), "tab:orange")
    axes[1].set_ylabel("V_b  [V]")
    axes[2].plot(tt, slide21["I_d"].cpu().numpy() * 1e6, "tab:green")
    axes[2].set_ylabel("I_d  [µA]")
    axes[2].set_xlabel("t  [µs]")
    for a in axes:
        a.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "slide21_replication.png", dpi=130)
    plt.close(fig)

    # 2) t_rise sweep (slide-21 right panel analog)
    print("[z273] (2) t_rise sweep")
    t_rises = [10e-6, 100e-6, 200e-6, 1e-3]
    trise_results = {}
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    for tr in t_rises:
        r = simulate_ramp(surr,
                          VG1=0.45, VG2=0.30,
                          V_set=2.05,
                          t_rise=tr, t_set=1e-6, t_fall=tr,
                          C_b_F=14e-15)
        s = extract_slopes(r)
        trise_results[f"{tr:.0e}"] = s
        ax.plot(r["V_d"].cpu().numpy(),
                r["I_d"].cpu().numpy() * 1e6,
                label=f"t_rise={tr * 1e6:.0f} µs  "
                      f"S_fire={s['S_fire_A_per_s']:.1e} A/s")
        print(f"    t_rise={tr * 1e6:7.1f} µs  "
              f"S_fire={s['S_fire_A_per_s']:.3e}  "
              f"S_relax={s['S_relax_A_per_s']:.3e}  "
              f"V_b_peak={s['V_b_peak_V']:.3f}  railed={s['railed']}")
    ax.set_xlabel("V_d  [V]")
    ax.set_ylabel("I_d  [µA]")
    ax.set_title("I_d(V_d) trajectories vs t_rise "
                 "(V_G1=0.45, V_G2=0.30, C_b=14 fF)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "trise_sweep.png", dpi=130)
    plt.close(fig)

    # Monotonicity sanity: S_fire vs 1/t_rise
    s_fires = [trise_results[f"{tr:.0e}"]["S_fire_A_per_s"] for tr in t_rises]
    # Sort by t_rise ascending; expect S_fire to DECREASE
    monotonic = all(s_fires[i] >= s_fires[i + 1] - 1e-12
                    for i in range(len(s_fires) - 1))
    print(f"[z273]    monotonic S_fire vs (1/t_rise): {monotonic}")

    # 3) C_b sweep
    print("[z273] (3) C_b sweep")
    cbs_fF = [2.0, 5.0, 8.0, 14.0, 20.0, 50.0]
    cb_results = {}
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    for cb_fF in cbs_fF:
        r = simulate_ramp(surr,
                          VG1=0.45, VG2=0.30,
                          V_set=2.05,
                          t_rise=100e-6, t_set=1e-6, t_fall=100e-6,
                          C_b_F=cb_fF * 1e-15)
        s = extract_slopes(r)
        cb_results[f"{cb_fF:.1f}fF"] = s
        axs[0].plot(r["V_d"].cpu().numpy(),
                    r["I_d"].cpu().numpy() * 1e6,
                    label=f"C_b={cb_fF:.0f} fF")
        print(f"    C_b={cb_fF:5.1f} fF  "
              f"S_fire={s['S_fire_A_per_s']:.3e}  "
              f"S_relax={s['S_relax_A_per_s']:.3e}  "
              f"V_b_peak={s['V_b_peak_V']:.3f}  railed={s['railed']}")
    axs[0].set_xlabel("V_d  [V]")
    axs[0].set_ylabel("I_d  [µA]")
    axs[0].set_title("I_d(V_d) vs C_b  (t_rise=100 µs)")
    axs[0].grid(True, alpha=0.3)
    axs[0].legend(fontsize=8)

    axs[1].plot(cbs_fF,
                [cb_results[f"{c:.1f}fF"]["S_fire_A_per_s"] for c in cbs_fF],
                "o-", label="S_fire")
    axs[1].plot(cbs_fF,
                [cb_results[f"{c:.1f}fF"]["S_relax_A_per_s"] for c in cbs_fF],
                "s--", label="S_relax")
    axs[1].set_xlabel("C_b  [fF]")
    axs[1].set_ylabel("slope  [A/s]")
    axs[1].set_yscale("log")
    axs[1].set_title("S_fire / S_relax vs C_b")
    axs[1].grid(True, alpha=0.3, which="both")
    axs[1].legend()
    fig.tight_layout()
    fig.savefig(OUT / "cb_sweep.png", dpi=130)
    plt.close(fig)

    # ────── Slide-21 measured reference ──────
    # The slide-21 figure shows S_fire and S_relax on the order of µA/µs
    # (i.e. ~1 A/s) for typical ramps; the right panel shows them scaling
    # with 1/t_rise. We use an order-of-magnitude bracket of [1e-2, 1e2] A/s
    # for the PASS gate (covers µA/ms .. mA/µs).
    slide21_reference_A_per_s = {
        "S_fire_nominal_A_per_s": 1.0,    # µA/µs order
        "S_relax_nominal_A_per_s": 1.0,
        "PASS_band_decades": 1.0,
        "note": "Order-of-magnitude estimate read off slide-21 (µA / µs).",
    }

    def in_band(sim, ref, dec=1.0):
        if not np.isfinite(sim) or sim <= 0:
            return False
        return 10 ** (-dec) <= sim / ref <= 10 ** dec

    pass_fire = in_band(s21["S_fire_A_per_s"],
                        slide21_reference_A_per_s["S_fire_nominal_A_per_s"])
    pass_relax = in_band(s21["S_relax_A_per_s"],
                         slide21_reference_A_per_s["S_relax_nominal_A_per_s"])
    overall_pass = pass_fire and pass_relax and not s21["railed"]

    # Find best-matching C_b
    cb_diffs = {k: abs(np.log10(max(v["S_fire_A_per_s"], 1e-30))
                       - np.log10(slide21_reference_A_per_s[
                           "S_fire_nominal_A_per_s"]))
                for k, v in cb_results.items()}
    best_cb = min(cb_diffs, key=cb_diffs.get)

    summary = {
        "script": "scripts/z273_pmp9_transient_simulator.py",
        "surrogate": str(SURR),
        "device": str(DEVICE),
        "wallclock_s": round(time.time() - t0, 2),
        "slide21_replication": s21,
        "trise_sweep": trise_results,
        "cb_sweep": cb_results,
        "slide21_reference": slide21_reference_A_per_s,
        "gate": {
            "PASS_S_fire": bool(pass_fire),
            "PASS_S_relax": bool(pass_relax),
            "monotonic_S_fire_vs_inv_trise": bool(monotonic),
            "rail_check_passed": not s21["railed"],
            "OVERALL": "PASS" if overall_pass else "FAIL",
        },
        "best_C_b_fF_for_slide21": best_cb,
        "best_C_b_S_fire_A_per_s":
            cb_results[best_cb]["S_fire_A_per_s"],
    }

    with open(OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)

    print(f"[z273] wrote {OUT/'summary.json'}")
    print(f"[z273] GATE = {summary['gate']['OVERALL']}  "
          f"(fire={pass_fire}, relax={pass_relax}, "
          f"monotonic={monotonic}, rails_ok={not s21['railed']})")
    print(f"[z273] best C_b match: {best_cb} "
          f"(S_fire={summary['best_C_b_S_fire_A_per_s']:.3e})")


if __name__ == "__main__":
    main()
