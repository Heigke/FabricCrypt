"""z396 — S4-D: TLP-style transient ramp, look for hysteresis loop.

Per research_plan/SNAPBACK_SELF_VALIDATION_PLAN_2026-05-15.md.

Goal: slow triangular Vd(t) ramp 0→2V→0V (10μs each leg) at fixed Cb=8 fF.
Bias: VG1=0.6, VG2=0.2, vnwell=2.0. Use the existing implicit-Euler
integrator in nsram.bsim4_port.transient.integrate_2t_transient_implicit.

Plot Ids vs Vd for both up-ramp and down-ramp. Compute hysteresis spread
at three diagnostic Vds: 0.5, 1.0, 1.5 V. If |log10(Id_up/Id_dn)| > 0.5 dec
anywhere, hysteresis loop is real ⇒ bistability in time domain even if DC
solver misses it.

Output: results/z396_tlp_transient/{summary.json, ids_vs_vd_hysteresis.png}
"""
from __future__ import annotations
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")

import sys, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "nsram"))

OUT = ROOT / "results/z396_tlp_transient"
OUT.mkdir(parents=True, exist_ok=True)

import torch
torch.set_default_dtype(torch.float64)

VG1_val = 0.6
VG2_val = 0.2
T_RAMP = 10e-6        # 10 μs per leg
V_PEAK = 2.0
N_PTS_PER_LEG = 200   # 50 ns step, gives Cb time constant resolution
CB_F = 8e-15          # 8 fF


def main():
    from scripts.nsram_surrogate_4d import _build_pyport_models
    from nsram.bsim4_port.transient import integrate_2t_transient_implicit

    print(f"=== z396 S4-D TLP transient ramp ===")
    print(f"VG1={VG1_val}, VG2={VG2_val}, V_peak={V_PEAK}, ramp={T_RAMP*1e6}μs/leg")
    print(f"Cb={CB_F*1e15:.2f} fF, N={2*N_PTS_PER_LEG} pts total")

    cfg, M1, M2, bjt = _build_pyport_models()

    # Override the cfg's Cb (body junction cap) so the transient sees Cb=8 fF.
    # The implicit solver multiplies Cj0_per_area × area to get Cj0_total.
    cfg.body_pdiode_Cj0_per_area = CB_F   # F / (1 area unit)
    cfg.body_pdiode_area = 1.0
    # vnwell already 2.0 by default; double-check
    if hasattr(cfg, "vnwell"):
        cfg.vnwell = 2.0

    # Triangular waveform: 0 → V_PEAK over T_RAMP, then V_PEAK → 0 over T_RAMP
    t_up = torch.linspace(0.0, T_RAMP, N_PTS_PER_LEG)
    t_dn = torch.linspace(T_RAMP, 2*T_RAMP, N_PTS_PER_LEG + 1)[1:]
    t = torch.cat([t_up, t_dn])
    Vd_up = torch.linspace(0.0, V_PEAK, N_PTS_PER_LEG)
    Vd_dn = torch.linspace(V_PEAK, 0.0, N_PTS_PER_LEG + 1)[1:]
    Vd_t = torch.cat([Vd_up, Vd_dn])
    VG1 = torch.tensor(VG1_val, dtype=torch.float64)
    VG2 = torch.tensor(VG2_val, dtype=torch.float64)

    print("Integrating ...")
    res = integrate_2t_transient_implicit(
        cfg, M1, M2, bjt, Vd_t=Vd_t, t=t, VG1=VG1, VG2=VG2,
        Vb0=0.0, Vsint0=0.0,
        spike_threshold=10.0,    # disable spike reset for this test
        reset_Vb=0.0,
        newton_iters_inner=8,
        newton_iters_outer=12,
        newton_tol=1e-12,
        verbose=False,
    )

    Vb = res["Vb"].numpy()
    Vsint = res["Vsint"].numpy()
    Id = res["Id"].numpy()
    t_np = t.numpy()
    Vd_np = Vd_t.numpy()
    has_nan = bool(np.any(np.isnan(Id)))
    print(f"  done. NaN in Id: {has_nan}")
    print(f"  Vb range: [{np.min(Vb):.3f}, {np.max(Vb):.3f}]")
    print(f"  Vsint range: [{np.min(Vsint):.3f}, {np.max(Vsint):.3f}]")
    print(f"  Id range: [{np.min(np.abs(Id)):.3e}, {np.max(np.abs(Id)):.3e}]")

    Vd_up_np = Vd_up.numpy(); Id_up = Id[:N_PTS_PER_LEG]
    Vd_dn_np = Vd_dn.numpy(); Id_dn = Id[N_PTS_PER_LEG:]

    # Hysteresis at diagnostic Vds: 0.5, 1.0, 1.5
    diag = {}
    for v_check in (0.5, 1.0, 1.5):
        # Find closest Vd on each leg
        i_up = int(np.argmin(np.abs(Vd_up_np - v_check)))
        i_dn = int(np.argmin(np.abs(Vd_dn_np - v_check)))
        Id_up_v = float(Id_up[i_up])
        Id_dn_v = float(Id_dn[i_dn])
        # log-decade spread
        ratio_dec = (np.log10(max(abs(Id_up_v), 1e-30)) -
                     np.log10(max(abs(Id_dn_v), 1e-30)))
        diag[f"Vd={v_check}"] = {
            "Id_up": Id_up_v, "Id_dn": Id_dn_v,
            "ratio_dec": float(ratio_dec),
            "Vb_up": float(Vb[i_up]), "Vb_dn": float(Vb[N_PTS_PER_LEG + i_dn]),
        }
        print(f"  @ Vd={v_check}: Id_up={Id_up_v:.3e}, Id_dn={Id_dn_v:.3e}, "
              f"|spread|={abs(ratio_dec):.3f} dec")

    max_spread = max(abs(d["ratio_dec"]) for d in diag.values())
    snap_jump_up = False
    # Look for snap-jump in up-ramp: max |d log|Id| / d Vd| > 5 between adjacent steps
    Id_up_abs = np.abs(Id_up).clip(min=1e-30)
    log_Id = np.log10(Id_up_abs)
    diffs = np.abs(np.diff(log_Id))
    if len(diffs):
        max_jump = float(np.max(diffs))
        if max_jump > 2.0:
            snap_jump_up = True
    else:
        max_jump = 0.0
    print(f"  max |d log Id| step (up-ramp) = {max_jump:.3f}")

    if max_spread > 0.5:
        verdict = ("S4-D DISCOVERY: hysteresis loop with > 0.5 dec spread "
                   f"(max {max_spread:.2f} dec) → bistability in time domain.")
        case = "discovery"
    elif max_spread > 0.1:
        verdict = (f"S4-D INTERMEDIATE: small hysteresis ({max_spread:.3f} dec). "
                   "Capacitive lag only, no bistability.")
        case = "intermediate"
    else:
        verdict = (f"S4-D KILL-SHOT (partial): no hysteresis (max spread "
                   f"{max_spread:.3f} dec < 0.1) → no time-domain bistability.")
        case = "kill_shot"
    print(f"\nVERDICT: {verdict}")

    summary = {
        "bias": {"VG1": VG1_val, "VG2": VG2_val, "vnwell": 2.0},
        "params": {"Cb_F": CB_F, "T_ramp_s": T_RAMP, "V_peak": V_PEAK,
                   "n_per_leg": N_PTS_PER_LEG},
        "ranges": {
            "Id_min": float(np.min(np.abs(Id))),
            "Id_max": float(np.max(np.abs(Id))),
            "Vb_min": float(np.min(Vb)), "Vb_max": float(np.max(Vb)),
            "Vsint_min": float(np.min(Vsint)), "Vsint_max": float(np.max(Vsint)),
        },
        "hysteresis": diag,
        "max_spread_dec": float(max_spread),
        "max_log_jump_up": max_jump,
        "snap_jump_up": snap_jump_up,
        "has_nan": has_nan,
        "verdict": verdict,
        "case": case,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))

    np.savez(OUT / "traces.npz", t=t_np, Vd=Vd_np, Id=Id, Vb=Vb, Vsint=Vsint,
             Vd_up=Vd_up_np, Id_up=Id_up, Vd_dn=Vd_dn_np, Id_dn=Id_dn)

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        ax = axes[0]
        ax.semilogy(Vd_up_np, np.abs(Id_up).clip(1e-30), 'b-', lw=2, label='up-ramp 0→2V')
        ax.semilogy(Vd_dn_np, np.abs(Id_dn).clip(1e-30), 'r--', lw=2, label='dn-ramp 2→0V')
        for v_check in (0.5, 1.0, 1.5):
            ax.axvline(v_check, color='gray', ls=':', alpha=0.4)
        ax.set_xlabel('Vd (V)')
        ax.set_ylabel('|Id| (A)')
        ax.set_title(f'TLP @ VG1={VG1_val}, VG2={VG2_val}, Cb={CB_F*1e15:.1f}fF, '
                     f'τ_ramp={T_RAMP*1e6:.0f}μs/leg\nmax-spread={max_spread:.3f} dec')
        ax.legend(); ax.grid(True, which='both', alpha=0.3)

        ax = axes[1]
        ax.plot(t_np*1e6, Vb, 'g-', label='Vb (body)')
        ax.plot(t_np*1e6, Vsint, 'm-', label='Vsint')
        ax.plot(t_np*1e6, Vd_np, 'k--', alpha=0.5, label='Vd (drive)')
        ax.set_xlabel('t (μs)'); ax.set_ylabel('V')
        ax.set_title('State variables vs time')
        ax.legend(); ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(OUT / "ids_vs_vd_hysteresis.png", dpi=130)
        plt.close(fig)
        print(f"\nPlot: {OUT}/ids_vs_vd_hysteresis.png")
    except Exception as e:
        print(f"  plot FAILED: {e}")

    print(f"\nSummary: {OUT}/summary.json")


if __name__ == "__main__":
    main()
