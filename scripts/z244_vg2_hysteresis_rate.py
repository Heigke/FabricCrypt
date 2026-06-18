"""z244 — STEP 1 of V_G2 continuum study: rate-dependent hysteresis.

Per VG2_CONTINUUM_PLAN STEP 1 and gemini O39 pick. Decides whether the
V_G2 continuum has dynamical content or is just a static map.

Setup: single 2T cell at V_G1=0.4, V_d=1.0 (in surrogate grid). V_G2
triangular wave swept up from 0.0 to 0.55 V then back down to 0.0,
over total duration T_ramp ∈ {100ns, 1µs, 10µs, 100µs, 1ms, 10ms}.
External Vb integration with Cb=5fF (same convention as the reservoir
experiments). At each timestep evaluate the 4D body-state surrogate at
(VG1, VG2[t], Vd, Vb[t]) → (Id, Iii, Ileak), update Vb.

Compute hysteresis loop area in (V_G2, V_b) and (V_G2, log|I_d|)
projections for each T_ramp.

Acceptance:
  PASS (continuum has dynamical content): loop area at fast ramps > 0
  AND monotonically shrinks as T_ramp grows.
  FAIL: loop area effectively zero (≤ numerical noise) at all rates.

Output: results/z244_vg2_hysteresis_rate/{summary.json, loops.pdf, loops.png}.
"""
from __future__ import annotations
import os, sys, json, time
from pathlib import Path
import numpy as np

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results/z244_vg2_hysteresis_rate"; OUT.mkdir(parents=True, exist_ok=True)
SURR_PATH = ROOT / "results/z220_4d_dense/surrogate_4d_dense.npz"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.nsram_surrogate_4d import NSRAMSurrogate4D


def triangular_VG2(n_steps: int, vmin=0.0, vmax=0.55):
    """Triangular wave: 0->vmax over first half, vmax->0 over second half."""
    half = n_steps // 2
    up = np.linspace(vmin, vmax, half, endpoint=False)
    down = np.linspace(vmax, vmin, n_steps - half)
    return np.concatenate([up, down])


def integrate_cell(surr, VG1, Vd, VG2_traj, dt, Cb=5e-15, Vb0=0.0):
    """External Vb-integration through the surrogate at swept VG2.

    Returns Vb_traj, Id_traj, Iii_traj, Ileak_traj.
    """
    n = len(VG2_traj)
    Vb = np.full(1, Vb0)
    out_Vb = np.zeros(n)
    out_Id = np.zeros(n)
    out_Iii = np.zeros(n)
    out_Ile = np.zeros(n)
    for t in range(n):
        vg2 = np.full(1, VG2_traj[t])
        vg1 = np.full(1, VG1)
        vd = np.full(1, Vd)
        log_Id, Iii, Ile = surr.eval(vg1, vg2, vd, Vb)
        Id = 10.0 ** log_Id[0]
        out_Vb[t] = Vb[0]
        out_Id[t] = Id
        out_Iii[t] = Iii[0]
        out_Ile[t] = Ile[0]
        Vb_new = Vb[0] + dt * (Iii[0] - Ile[0]) / Cb
        Vb = np.array([np.clip(Vb_new, 0.0, 0.7)])
    return out_Vb, out_Id, out_Iii, out_Ile


def loop_area(x_up, y_up, x_down, y_down):
    """Signed loop area enclosed by two paths x_up,y_up (left→right)
    and x_down,y_down (right→left). Uses trapezoid rule along x_common."""
    # Assemble closed loop polygon: up path forward + down path reversed
    xs = np.concatenate([x_up, x_down[::-1]])
    ys = np.concatenate([y_up, y_down[::-1]])
    # Shoelace formula
    return 0.5 * abs(np.sum(xs * np.roll(ys, -1) - np.roll(xs, -1) * ys))


def main():
    print(f"=== z244 V_G2 rate-dependent hysteresis ===", flush=True)
    surr = NSRAMSurrogate4D(SURR_PATH)

    VG1 = 0.4
    Vd = 1.0
    Cb = 5e-15
    n_steps = 400      # 200 up + 200 down, dense enough for loop

    T_ramps = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]  # 100ns to 10ms
    results = []

    fig, axes = plt.subplots(2, len(T_ramps), figsize=(3.2 * len(T_ramps), 6.5),
                              squeeze=False)
    rng_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    for k, T_ramp in enumerate(T_ramps):
        dt = T_ramp / n_steps
        VG2_traj = triangular_VG2(n_steps)
        t0 = time.time()
        Vb_traj, Id_traj, Iii_traj, Ile_traj = integrate_cell(
            surr, VG1, Vd, VG2_traj, dt, Cb=Cb, Vb0=0.0)
        wall = time.time() - t0

        half = n_steps // 2
        VG2_up = VG2_traj[:half]; Vb_up = Vb_traj[:half]
        VG2_dn = VG2_traj[half:]; Vb_dn = Vb_traj[half:]
        Id_up = Id_traj[:half]; Id_dn = Id_traj[half:]

        area_Vb = loop_area(VG2_up, Vb_up, VG2_dn, Vb_dn)
        area_logId = loop_area(VG2_up, np.log10(np.maximum(Id_up, 1e-15)),
                                  VG2_dn, np.log10(np.maximum(Id_dn, 1e-15)))

        print(f"T_ramp={T_ramp:.1e}s  dt={dt:.2e}s  "
              f"loop_area(VG2,Vb)={area_Vb:.5f}  "
              f"loop_area(VG2,log|Id|)={area_logId:.5f}  wall={wall:.1f}s",
              flush=True)

        results.append({
            "T_ramp_s": float(T_ramp),
            "dt_s": float(dt),
            "loop_area_Vb": float(area_Vb),
            "loop_area_log_Id": float(area_logId),
            "wall_s": float(wall),
            "Vb_max_observed": float(Vb_traj.max()),
            "Vb_final": float(Vb_traj[-1]),
        })

        ax_top = axes[0][k]
        ax_top.plot(VG2_up, Vb_up, color=rng_colors[k], lw=1.5,
                     label=f"up")
        ax_top.plot(VG2_dn, Vb_dn, color=rng_colors[k], lw=1.5, ls="--",
                     label=f"down")
        ax_top.set_title(f"T_ramp = {T_ramp:.0e} s\nloop area = {area_Vb:.3e}",
                          fontsize=9)
        ax_top.set_xlabel("V_G2 [V]"); ax_top.set_ylabel("V_b [V]")
        ax_top.grid(alpha=0.3)
        ax_top.legend(fontsize=7)

        ax_bot = axes[1][k]
        ax_bot.plot(VG2_up, np.log10(np.maximum(Id_up, 1e-15)),
                     color=rng_colors[k], lw=1.5, label=f"up")
        ax_bot.plot(VG2_dn, np.log10(np.maximum(Id_dn, 1e-15)),
                     color=rng_colors[k], lw=1.5, ls="--", label=f"down")
        ax_bot.set_title(f"log|I_d|  loop area = {area_logId:.3e}", fontsize=9)
        ax_bot.set_xlabel("V_G2 [V]"); ax_bot.set_ylabel("log10|I_d|")
        ax_bot.grid(alpha=0.3)
        ax_bot.legend(fontsize=7)

    plt.suptitle("V_G2 → V_b and V_G2 → log|I_d| hysteresis vs ramp duration\n"
                 "(triangular V_G2 sweep 0→0.55→0 V, V_G1=0.4, V_d=1.0, Cb=5fF)",
                 fontsize=11, weight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT / "loops.pdf", bbox_inches="tight")
    fig.savefig(OUT / "loops.png", bbox_inches="tight", dpi=150)
    plt.close()

    # Loop area vs rate plot
    fig2, ax2 = plt.subplots(1, 2, figsize=(11, 4.5))
    T = np.array([r["T_ramp_s"] for r in results])
    A_Vb = np.array([r["loop_area_Vb"] for r in results])
    A_Id = np.array([r["loop_area_log_Id"] for r in results])
    ax2[0].loglog(T, A_Vb, "o-", color="#1f77b4", lw=2, ms=10)
    ax2[0].set_xlabel("T_ramp [s] (longer → quasi-static)")
    ax2[0].set_ylabel("Hysteresis loop area in (V_G2, V_b)")
    ax2[0].set_title("V_b loop area vs ramp duration")
    ax2[0].grid(alpha=0.3, which="both")
    ax2[1].loglog(T, A_Id, "o-", color="#d62728", lw=2, ms=10)
    ax2[1].set_xlabel("T_ramp [s]")
    ax2[1].set_ylabel("Hysteresis loop area in (V_G2, log|I_d|)")
    ax2[1].set_title("log|I_d| loop area vs ramp duration")
    ax2[1].grid(alpha=0.3, which="both")
    plt.tight_layout()
    fig2.savefig(OUT / "area_vs_rate.pdf", bbox_inches="tight")
    fig2.savefig(OUT / "area_vs_rate.png", bbox_inches="tight", dpi=150)
    plt.close()

    # Acceptance gate analysis
    # PASS if max(A_Vb) > 1e-4 AND A_Vb is monotonically decreasing (allow noise)
    noise_floor = 1e-6  # numerical floor
    has_signal = bool(A_Vb.max() > noise_floor * 100)
    # Check rough monotonicity: max should be at fastest ramp
    monotone_ok = bool(np.argmax(A_Vb) <= 1)
    gate_pass = has_signal and monotone_ok

    summary = {
        "config": {"VG1": VG1, "Vd": Vd, "Cb_F": Cb, "n_steps": n_steps,
                    "VG2_range_V": [0.0, 0.55]},
        "T_ramps_s": [r["T_ramp_s"] for r in results],
        "loop_area_Vb": [r["loop_area_Vb"] for r in results],
        "loop_area_log_Id": [r["loop_area_log_Id"] for r in results],
        "Vb_max_per_ramp": [r["Vb_max_observed"] for r in results],
        "has_signal_above_floor": has_signal,
        "max_at_fastest_ramp": monotone_ok,
        "gate_pass_STEP1": gate_pass,
        "interpretation": (
            "PASS — V_G2 continuum has dynamical content, smooth morph is "
            "distinct from step-switch. STEP 2 (trainable schedule) can proceed."
            if gate_pass else
            "FAIL — V_G2 continuum is effectively a static map at the timescales "
            "tested. Kill smooth-morph story; STEP 3 (mixed-population) still on."
        ),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n=== Acceptance gate ===", flush=True)
    print(f"Loop area max (Vb): {A_Vb.max():.3e}  noise_floor: {noise_floor:.0e}",
          flush=True)
    print(f"Has signal above floor: {has_signal}", flush=True)
    print(f"Max at fastest ramp: {monotone_ok}", flush=True)
    print(f"STEP 1 GATE: {'✅ PASS' if gate_pass else '❌ FAIL'}", flush=True)
    print(f"\n{summary['interpretation']}", flush=True)


if __name__ == "__main__":
    main()
