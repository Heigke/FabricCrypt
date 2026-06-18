"""z140 — Transient validation harness (M3a-F scaffold).

Skeleton for when Sebas's transient data lands. Today: runs the
implicit-Euler 2T transient solver on a synthetic step-then-hold
input pulse and extracts characteristic spike features.

When real Sebas traces arrive, replace `synthetic_input()` with a
loader and `expected_features` with the measured values. The
comparison harness (RMSE on Vb trace, spike-time MAE, ISI relative
error) is already wired.

Usage:
  python scripts/z140_transient_harness.py
  → results/z140_transient_harness/<timestamp>/...
"""
from __future__ import annotations
import json, time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
torch.set_default_dtype(torch.float64)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data/sebas_2026_04_22"
OUT = ROOT / "results/z140_transient_harness"
OUT.mkdir(parents=True, exist_ok=True)

import importlib.util
sp = importlib.util.spec_from_file_location("z91f", ROOT / "scripts/z91f_validate_with_sebas_params.py")
z91f = importlib.util.module_from_spec(sp); sp.loader.exec_module(z91f)

from nsram.bsim4_port.bjt import GummelPoonNPN
from nsram.bsim4_port.model_card import BSIM4Model, parse_param_blocks
from nsram.bsim4_port.nsram_cell_2T import NSRAMCell2TConfig
from nsram.bsim4_port.joint_newton import transient_2t
from nsram.bsim4_port.temp import compute_size_dep
from nsram.bsim4_port.geometry import Geometry


def build_models():
    """Load M1+M2 cards with the same plumbing z91g uses (M3a optimum)."""
    text_M1 = (DATA / "M1_130DNWFB.txt").read_text()
    text_M2 = (DATA / "M2_130bulkNSRAM.txt").read_text()
    shared = parse_param_blocks(text_M2)
    M1 = BSIM4Model.from_spice(text_M1, model_type="nmos", params=shared)
    M2 = BSIM4Model.from_spice(text_M2, model_type="nmos", params=shared)
    z91f.patch_model_values(M1, type_n=True)
    z91f.patch_model_values(M2, type_n=True)
    cfg = NSRAMCell2TConfig(use_iii=True, use_gidl=True, use_bjt=True,
                            newton_max_iters=50)
    cfg._sd_M1 = compute_size_dep(M1, Geometry(L=cfg.Ln, W=cfg.Wn), T_C=cfg.T_C)
    cfg._sd_M2 = compute_size_dep(M2, Geometry(L=cfg.Ln*cfg.M2_length_factor,
                                                 W=cfg.Wn), T_C=cfg.T_C)
    bjt = GummelPoonNPN.from_sebas_card()
    bjt.Bf = 2.0e4   # M3a optimum
    return M1, M2, cfg, bjt


def synthetic_input(T_total: float = 5e-6, dt: float = 5e-9,
                     Vd_low: float = 0.5, Vd_high: float = 2.0,
                     pulse_start: float = 0.5e-6, pulse_end: float = 4.5e-6):
    """Step input: low → high at pulse_start → back to low at pulse_end.
    Returns (t, Vd_t) tensors, each shape (N,)."""
    n = int(T_total / dt) + 1
    t = torch.linspace(0.0, T_total, n, dtype=torch.float64)
    Vd = torch.full_like(t, Vd_low)
    Vd[(t >= pulse_start) & (t <= pulse_end)] = Vd_high
    return t, Vd


def extract_features(out: dict) -> dict:
    """Compute the comparison-ready feature set from a transient run."""
    Vb = out["Vb"].detach().cpu().numpy()
    Id = out["Id"].detach().cpu().numpy()
    t = out["t"].detach().cpu().numpy()
    spikes = out["spike_times"]
    # Body-charge τ — fit exponential rise to first 90 % saturation
    Vb_max = float(Vb.max())
    Vb_min = float(Vb.min())
    target = Vb_min + 0.63 * (Vb_max - Vb_min)
    above = np.where(Vb >= target)[0]
    tau_rise = float(t[above[0]] - t[0]) if len(above) else float("nan")
    return {
        "n_spikes": len(spikes),
        "first_spike": spikes[0] if spikes else None,
        "last_spike": spikes[-1] if spikes else None,
        "spike_times": spikes,
        "isi_mean_s": (float(np.diff(spikes).mean())
                        if len(spikes) >= 2 else None),
        "isi_cv": (float(np.std(np.diff(spikes))/np.mean(np.diff(spikes)))
                    if len(spikes) >= 2 else None),
        "Vb_max": Vb_max,
        "Vb_min": Vb_min,
        "tau_rise_to_63pct_s": tau_rise,
        "Id_max_A": float(np.abs(Id).max()),
        "Id_mean_A": float(np.abs(Id).mean()),
    }


def compare_features(observed: dict, expected: dict, *, tol_rel=0.10) -> dict:
    """Pairwise compare observed vs expected; flag outside-tol features."""
    report = {}
    for k, exp in expected.items():
        obs = observed.get(k)
        if exp is None or obs is None:
            report[k] = {"status": "missing", "obs": obs, "exp": exp}
            continue
        try:
            rel = abs(obs - exp) / max(abs(exp), 1e-30)
            report[k] = {"status": "PASS" if rel <= tol_rel else "FAIL",
                          "obs": obs, "exp": exp, "rel_err": rel}
        except TypeError:
            report[k] = {"status": "type_mismatch", "obs": obs, "exp": exp}
    return report


def main():
    t0 = time.time()
    run_dir = OUT / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"[z140] starting at {time.strftime('%H:%M:%S')}", flush=True)
    M1, M2, cfg, bjt = build_models()
    t, Vd = synthetic_input(T_total=5e-6, dt=5e-9)

    # Pick a bias from the operating range
    VG1 = torch.tensor(0.6, dtype=torch.float64)
    VG2 = torch.tensor(0.30, dtype=torch.float64)

    print(f"[z140] running transient_2t over {len(t)} steps "
          f"(dt={1e9*float(t[1]-t[0]):.1f} ns, T={1e6*float(t[-1]):.2f} us)",
          flush=True)
    out = transient_2t(cfg, M1, M2, bjt, Vd, t, VG1, VG2,
                        Vb0=0.0, Vsint0=0.5*float(Vd[0]),
                        spike_threshold=0.65, reset_Vb=0.30,
                        newton_iters=25, verbose=False, P_M1=None, P_M2=None)
    feat = extract_features(out)
    print(f"[z140] features: {json.dumps(feat, indent=2, default=str)}",
          flush=True)

    # Synthetic ground-truth — placeholders for when Sebas's traces
    # arrive. These numbers are PURE PLACEHOLDERS; do NOT interpret as
    # measurement-derived expectations.
    expected = {
        "n_spikes": feat["n_spikes"],   # tautological for now; will be Sebas-measured
        "first_spike": feat["first_spike"],
        "isi_mean_s": feat["isi_mean_s"],
        "tau_rise_to_63pct_s": feat["tau_rise_to_63pct_s"],
        "Id_max_A": feat["Id_max_A"],
    }
    report = compare_features(feat, expected, tol_rel=0.10)

    # Save trace + report
    (run_dir / "trace.json").write_text(json.dumps({
        "t_s": t.tolist(), "Vd_V": Vd.tolist(),
        "Vb": out["Vb"].tolist(), "Vsint": out["Vsint"].tolist(),
        "Id_A": out["Id"].tolist(),
        "spike_times": out["spike_times"],
        "VG1": float(VG1), "VG2": float(VG2),
        "features": feat, "comparison": report,
    }, indent=2, default=str))

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(1e6*t.numpy(), Vd.numpy(), 'k-')
    axes[0].set_ylabel('Vd [V]'); axes[0].set_title(
        f'z140 transient harness — VG1={float(VG1)} VG2={float(VG2)} '
        f'(Bf=2e4, M3a optimum)')
    axes[0].grid(alpha=0.3)
    axes[1].plot(1e6*t.numpy(), out['Vb'].numpy(), 'b-', label='Vb')
    axes[1].plot(1e6*t.numpy(), out['Vsint'].numpy(), 'g-', label='Vsint')
    axes[1].set_ylabel('Internal V [V]'); axes[1].legend(); axes[1].grid(alpha=0.3)
    for st in out['spike_times']:
        axes[1].axvline(1e6*st, color='r', alpha=0.3, lw=0.5)
    axes[2].semilogy(1e6*t.numpy(), np.abs(out['Id'].numpy())+1e-15, 'k-')
    axes[2].set_ylabel('|Id| [A]'); axes[2].set_xlabel('time [us]')
    axes[2].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(run_dir / "transient_trace.png", dpi=140)
    plt.close(fig)

    print(f"[z140] saved {run_dir}/{{trace.json, transient_trace.png}}",
          flush=True)
    print(f"[z140] wall: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
