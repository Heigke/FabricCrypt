"""z400 — TLP (Transmission Line Pulse) transient harness for S6 Phase C.

Validates that the rebuilt physics produces TLP-style snapback in time
domain (which Mario/Sebas measure on real silicon with 100 ns pulses),
NOT in DC steady-state where our S5-C analysis confirmed snapback is
absent.

Pulse profile (Vd vs t):
    t ∈ [0 ns,   100 ns]:  ramp 0   → 2.0 V  (linear)
    t ∈ [100 ns, 110 ns]:  hold 2.0 V
    t ∈ [110 ns, 210 ns]:  ramp 2.0 → 0   V  (linear)

Sweeps VG1 ∈ {0.2, 0.4, 0.6} V, VG2 = +0.2 V constant.

Integration uses the existing Backward-Euler (`integrate_2t_transient_implicit`)
in nsram/nsram/bsim4_port/transient.py.  dt = 0.1 ns → 2100 steps total.

Modes (env Z400_MODE):
    baseline : current pyport (v1)
    v2       : v2 topology (vnwell breakdown / anode-Vb diode / drain avalanche)
    rebuilt  : Phase A flags if available (use_vertical_npn_to_dnw,
               use_etab_vg2_curve, use_m2_as_resistor); falls back to v2
               with a warning if those flags don't exist yet on cfg.

Outputs (per VG1):
    results/z400_tlp/{tag}_VG1{vg1}.png       — 3 subplots: Id(t), Vb(t), phase
    results/z400_tlp/{tag}_VG1{vg1}.npz       — raw arrays
    results/z400_tlp/summary.json             — per-condition metrics + gates

Pre-registered gates:
    INFRA      : all traces complete without NaN, ≥ 100 steps per ramp,
                 wall < 30 min.
    DISCOVERY  : snap-jump detected on at least one VG1, defined as
                 max |dId/dt| in hold region (Vd ≈ const) > 100× the
                 mean |dId/dt| during the linear ramp-up phase.
    AMBITIOUS  : hysteresis loop AND latched state: I_d at Vd=2 V hold
                 > 10× I_d at Vd=2 V cold-DC (Vb=0).
    KILL-SHOT  : no snap-jump AND no hysteresis on any VG1 → transient
                 also can't produce snapback with this physics.
"""
from __future__ import annotations
import os
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "4")

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/z400_tlp"
OUT.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cpu")
DTYPE = torch.float64

# ------------------------------ TLP pulse params
T_RAMP_NS   = 100.0
T_HOLD_NS   = 10.0
VD_PEAK     = 2.0
DT_NS       = 0.1
T_TOTAL_NS  = 2.0 * T_RAMP_NS + T_HOLD_NS    # 210 ns

VG1_LIST    = [0.2, 0.4, 0.6]
VG2_CONST   = 0.2
VB0         = 0.0
VSINT0      = 0.0
SPIKE_THR   = 1e9    # disable spike-reset (we want to *observe* Vb buildup)
RESET_VB    = 0.30

MODE = os.environ.get("Z400_MODE", "baseline").lower()


def build_pulse():
    """Construct Vd(t), t arrays per spec."""
    n_ramp = int(round(T_RAMP_NS / DT_NS))   # 1000 each
    n_hold = int(round(T_HOLD_NS / DT_NS))   # 100
    # ramp-up: 0 → 2.0V over [0, 100ns], n_ramp+1 inclusive of t=100ns? Use a
    # contiguous sample: indices 0..n_ramp-1 cover ramp-up, n_ramp..n_ramp+n_hold-1
    # cover hold, n_ramp+n_hold..2*n_ramp+n_hold-1 cover ramp-down.
    n_total = 2 * n_ramp + n_hold
    t = np.arange(n_total, dtype=np.float64) * DT_NS * 1e-9   # seconds
    Vd = np.zeros(n_total, dtype=np.float64)
    # Ramp up
    Vd[:n_ramp] = np.linspace(0.0, VD_PEAK, n_ramp, endpoint=False)
    # Hold
    Vd[n_ramp:n_ramp + n_hold] = VD_PEAK
    # Ramp down
    Vd[n_ramp + n_hold:] = np.linspace(VD_PEAK, 0.0, n_ramp, endpoint=True)
    idx = {
        "ramp_up":   (0, n_ramp),
        "hold":      (n_ramp, n_ramp + n_hold),
        "ramp_down": (n_ramp + n_hold, n_total),
    }
    return t, Vd, idx


# ----------------------------- model loader
def _load_models(mode: str):
    """Returns (cfg, M1, M2, bjt, applied_tag)."""
    sp = importlib.util.spec_from_file_location(
        "ns4d", ROOT / "scripts/nsram_surrogate_4d.py")
    ns4d = importlib.util.module_from_spec(sp); sp.loader.exec_module(ns4d)
    cfg, M1, M2, bjt = ns4d._build_pyport_models()
    applied_tag = "baseline_v1"

    if mode == "baseline":
        return cfg, M1, M2, bjt, applied_tag

    # Try Phase A flags first (rebuilt)
    if mode in ("rebuilt", "v2"):
        rebuilt_flags = [
            "use_vertical_npn_to_dnw",
            "use_etab_vg2_curve",
            "use_etab_curve",
            "use_m2_as_resistor",
        ]
        any_rebuilt = any(hasattr(cfg, f) for f in rebuilt_flags)
        if mode == "rebuilt" and any_rebuilt:
            applied = []
            for f in rebuilt_flags:
                if hasattr(cfg, f):
                    setattr(cfg, f, True)
                    applied.append(f)
            applied_tag = "rebuilt_" + "+".join(applied)
            print(f"[z400] Phase A flags applied: {applied}")
            return cfg, M1, M2, bjt, applied_tag

        # Fall back to v2 topology
        try:
            sys.path.insert(0, str(ROOT / "src"))
            from nsram_pyport_v2 import V2Params, enable_v2_topology
            enable_v2_topology(cfg, V2Params())
            applied_tag = "v2_topology"
            print("[z400] v2 topology ENABLED (vnwell breakdown / anode-Vb diode "
                  "/ drain avalanche)")
        except Exception as e:
            print(f"[z400] WARNING: v2 topology unavailable ({e}); "
                  "using baseline.")
            applied_tag = "baseline_v1_fallback"
    return cfg, M1, M2, bjt, applied_tag


# ----------------------------- cold-DC reference at Vd=peak, Vb=0
def cold_dc_id(cfg, M1, M2, bjt, vg1: float):
    """Quasi-static Id at Vd=VD_PEAK with Vb pinned at 0 (cold initial)."""
    sp = importlib.util.spec_from_file_location(
        "ns4d", ROOT / "scripts/nsram_surrogate_4d.py")
    ns4d = importlib.util.module_from_spec(sp); sp.loader.exec_module(ns4d)
    out = ns4d._solve_at_fixed_vb(cfg, M1, M2, bjt,
                                   Vd=VD_PEAK, VG1=vg1, VG2=VG2_CONST,
                                   Vb_fixed=0.0)
    return out


# ----------------------------- main run
def run_tlp(cfg, M1, M2, bjt, vg1: float, tag: str):
    from nsram.bsim4_port.transient import integrate_2t_transient_implicit

    t_np, Vd_np, idx = build_pulse()
    Vd_t = torch.tensor(Vd_np, dtype=DTYPE)
    t_t  = torch.tensor(t_np,  dtype=DTYPE)
    VG1  = torch.tensor(vg1,   dtype=DTYPE)
    VG2  = torch.tensor(VG2_CONST, dtype=DTYPE)

    print(f"[z400] running TLP integration   tag={tag}  VG1={vg1}  "
          f"n_steps={len(t_np)}  dt={DT_NS}ns")
    t_start = time.time()
    result = integrate_2t_transient_implicit(
        cfg, M1, M2, bjt, Vd_t, t_t, VG1, VG2,
        Vb0=VB0, Vsint0=VSINT0,
        spike_threshold=SPIKE_THR,  # disable LIF reset
        reset_Vb=RESET_VB,
        newton_iters_inner=8,
        newton_iters_outer=12,
        verbose=False,
    )
    wall = time.time() - t_start
    print(f"[z400]   done in {wall:.1f}s")

    Id  = result["Id"].numpy()
    Vb  = result["Vb"].numpy()
    Vsint = result["Vsint"].numpy()

    return {
        "t_s": t_np, "Vd": Vd_np, "Id": Id, "Vb": Vb, "Vsint": Vsint,
        "idx": idx, "wall_s": wall,
    }


# ----------------------------- analysis
def analyze(trace: dict, cold_dc: dict, vg1: float) -> dict:
    t = trace["t_s"]; Vd = trace["Vd"]; Id = trace["Id"]; Vb = trace["Vb"]
    idx = trace["idx"]
    iu0, iu1 = idx["ramp_up"]
    ih0, ih1 = idx["hold"]
    id0, id1 = idx["ramp_down"]

    has_nan = bool(np.any(~np.isfinite(Id)) or np.any(~np.isfinite(Vb)))

    # dId/dt in each window
    dt = float(t[1] - t[0])
    dIdt = np.gradient(Id, dt)
    mean_ramp_up_slope = float(np.mean(np.abs(dIdt[iu0:iu1])))
    max_hold_slope     = float(np.max(np.abs(dIdt[ih0:ih1])))
    snap_jump_ratio    = (max_hold_slope / mean_ramp_up_slope
                          if mean_ramp_up_slope > 0 else float("inf"))
    snap_jump_detected = bool(snap_jump_ratio > 100.0)

    # Vb peak / latch
    vb_max     = float(np.max(Vb))
    vb_at_hold = float(np.mean(Vb[ih0:ih1]))
    vb_latch   = bool(vb_at_hold > 0.5)

    # Hysteresis: |Id(up at Vd_x) − Id(down at Vd_x)| sampled at common Vd
    n_ramp = iu1 - iu0
    Id_up   = Id[iu0:iu1]
    Vd_up   = Vd[iu0:iu1]
    Id_down = Id[id0:id1][::-1]   # reverse so Vd_down is also increasing
    Vd_down = Vd[id0:id1][::-1]
    # Interpolate down onto Vd_up grid
    if len(Id_up) > 1 and len(Id_down) > 1:
        Id_down_interp = np.interp(Vd_up, Vd_down, Id_down)
        hyst = Id_up - Id_down_interp
        hyst_max     = float(np.max(np.abs(hyst)))
        hyst_max_pos = float(Vd_up[int(np.argmax(np.abs(hyst)))])
        # log-area enclosed (in A·V) on linear scale, separate sign
        hyst_area = float(np.trapezoid(np.abs(hyst), Vd_up))
    else:
        hyst_max = hyst_max_pos = hyst_area = 0.0
    hyst_detected = bool(hyst_max > 0.0 and hyst_max / max(1e-30, np.max(np.abs(Id))) > 0.05)

    # Latch vs cold-DC
    cold_id = abs(cold_dc.get("Id", 0.0))
    hold_id = float(np.mean(np.abs(Id[ih0:ih1])))
    latch_ratio = hold_id / cold_id if cold_id > 1e-30 else float("inf")
    latch_detected = bool(latch_ratio > 10.0)

    return {
        "vg1": vg1,
        "has_nan": has_nan,
        "n_steps_ramp_up": int(iu1 - iu0),
        "n_steps_hold":    int(ih1 - ih0),
        "n_steps_ramp_down": int(id1 - id0),
        "mean_ramp_up_dIdt_A_per_s": mean_ramp_up_slope,
        "max_hold_dIdt_A_per_s":     max_hold_slope,
        "snap_jump_ratio":           snap_jump_ratio,
        "snap_jump_detected":        snap_jump_detected,
        "vb_max":                    vb_max,
        "vb_at_hold_mean":           vb_at_hold,
        "vb_latched_gt0p5":          vb_latch,
        "hyst_max_A":                hyst_max,
        "hyst_max_at_Vd":            hyst_max_pos,
        "hyst_area_AV":              hyst_area,
        "hyst_detected":             hyst_detected,
        "cold_dc_Id_A":              cold_id,
        "hold_Id_A":                 hold_id,
        "latch_ratio":               latch_ratio,
        "latch_detected":            latch_detected,
    }


# ----------------------------- plotting
def plot_trace(trace: dict, ana: dict, tag: str, vg1: float):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    t_ns = trace["t_s"] * 1e9
    Vd = trace["Vd"]; Id = trace["Id"]; Vb = trace["Vb"]
    idx = trace["idx"]
    iu0, iu1 = idx["ramp_up"]; ih0, ih1 = idx["hold"]
    id0, id1 = idx["ramp_down"]

    # ---- 1: Id(t) + Vd(t) shadow
    ax = axes[0]
    ax.plot(t_ns, np.abs(Id) + 1e-30, "b-", lw=1.2, label="|Id|")
    ax.set_yscale("log")
    ax.set_xlabel("time (ns)")
    ax.set_ylabel("|Id| (A)", color="b")
    ax.axvspan(t_ns[iu0], t_ns[iu1-1], color="green", alpha=0.10, label="ramp up")
    ax.axvspan(t_ns[ih0], t_ns[ih1-1], color="red",   alpha=0.20, label="hold")
    ax.axvspan(t_ns[id0], t_ns[id1-1], color="green", alpha=0.10, label="ramp down")
    ax2 = ax.twinx()
    ax2.plot(t_ns, Vd, "k--", lw=0.6, alpha=0.5)
    ax2.set_ylabel("Vd (V)")
    ax.set_title(f"Id(t)  VG1={vg1} V, VG2={VG2_CONST} V")
    ax.legend(loc="upper left", fontsize=8)

    # ---- 2: Vb(t)
    ax = axes[1]
    ax.plot(t_ns, Vb, "r-", lw=1.2)
    ax.axhline(0.5, color="gray", ls=":", lw=0.6)
    ax.axhline(0.7, color="gray", ls=":", lw=0.6)
    ax.axvspan(t_ns[ih0], t_ns[ih1-1], color="red", alpha=0.15)
    ax.set_xlabel("time (ns)")
    ax.set_ylabel("Vb (V)")
    ax.set_title(f"Body voltage  vb_max={ana['vb_max']:.3f}  latched={ana['vb_latched_gt0p5']}")

    # ---- 3: phase plot Id vs Vd
    ax = axes[2]
    Id_up   = np.abs(Id[iu0:iu1]) + 1e-30
    Vd_up   = Vd[iu0:iu1]
    Id_down = np.abs(Id[id0:id1]) + 1e-30
    Vd_down = Vd[id0:id1]
    ax.plot(Vd_up,   Id_up,   "g-", lw=1.0, label="ramp up")
    ax.plot(Vd_down, Id_down, "m-", lw=1.0, label="ramp down")
    if ana["cold_dc_Id_A"] > 1e-30:
        ax.axhline(ana["cold_dc_Id_A"], color="k", ls="--", lw=0.6,
                   label=f"cold-DC @Vd=2 ({ana['cold_dc_Id_A']:.2e})")
    ax.set_yscale("log")
    ax.set_xlabel("Vd (V)")
    ax.set_ylabel("|Id| (A)")
    ax.set_title(f"Phase  hyst={ana['hyst_detected']}  snap={ana['snap_jump_detected']}  "
                 f"latch={ana['latch_detected']} (ratio={ana['latch_ratio']:.1e})")
    ax.legend(fontsize=8)

    fig.suptitle(f"z400 TLP transient  [{tag}]  VG1={vg1} V", fontsize=11)
    fig.tight_layout()
    fname = OUT / f"{tag}_VG1{vg1:.2f}.png"
    fig.savefig(fname, dpi=110)
    plt.close(fig)
    print(f"[z400]   saved {fname}")


# ----------------------------- driver
def main():
    t_start = time.time()
    cfg, M1, M2, bjt, applied_tag = _load_models(MODE)
    print(f"[z400] MODE={MODE}  applied_tag={applied_tag}")

    per_vg1 = []
    for vg1 in VG1_LIST:
        cold = cold_dc_id(cfg, M1, M2, bjt, vg1)
        print(f"[z400] cold-DC @VG1={vg1}: Id={cold['Id']:.3e} A  "
              f"converged={cold['converged']}")
        trace = run_tlp(cfg, M1, M2, bjt, vg1, applied_tag)
        ana = analyze(trace, cold, vg1)
        plot_trace(trace, ana, applied_tag, vg1)

        npz_path = OUT / f"{applied_tag}_VG1{vg1:.2f}.npz"
        np.savez(npz_path,
                 t_s=trace["t_s"], Vd=trace["Vd"], Id=trace["Id"],
                 Vb=trace["Vb"], Vsint=trace["Vsint"])
        per_vg1.append(ana)
        print(f"[z400]   snap_jump_ratio={ana['snap_jump_ratio']:.2f}  "
              f"vb_max={ana['vb_max']:.3f}  latch_ratio={ana['latch_ratio']:.2e}  "
              f"hyst_max={ana['hyst_max_A']:.2e}")

    wall = time.time() - t_start

    # Aggregate gates
    any_snap   = any(d["snap_jump_detected"] for d in per_vg1)
    any_hyst   = any(d["hyst_detected"]      for d in per_vg1)
    any_latch  = any(d["latch_detected"]     for d in per_vg1)
    any_nan    = any(d["has_nan"]            for d in per_vg1)
    min_ramp_steps = min(d["n_steps_ramp_up"] for d in per_vg1)
    gates = {
        "INFRA":     bool((not any_nan) and (min_ramp_steps >= 100)
                          and (wall < 30 * 60)),
        "DISCOVERY": any_snap,
        "AMBITIOUS": any_hyst and any_latch,
        "KILL_SHOT": (not any_snap) and (not any_hyst),
    }

    summary = {
        "mode":          MODE,
        "applied_tag":   applied_tag,
        "wall_s":        wall,
        "vg1_list":      VG1_LIST,
        "vg2_const":     VG2_CONST,
        "pulse": {
            "t_ramp_ns": T_RAMP_NS, "t_hold_ns": T_HOLD_NS,
            "vd_peak":   VD_PEAK,   "dt_ns":     DT_NS,
        },
        "per_vg1":       per_vg1,
        "gates":         gates,
    }

    # Merge if multiple modes were already saved
    summary_path = OUT / "summary.json"
    if summary_path.exists():
        try:
            prev = json.loads(summary_path.read_text())
            if isinstance(prev, dict) and "by_mode" in prev:
                runs = prev["by_mode"]
            elif isinstance(prev, dict) and "mode" in prev:
                runs = {prev["applied_tag"]: prev}
            else:
                runs = {}
        except Exception:
            runs = {}
    else:
        runs = {}
    runs[applied_tag] = summary
    summary_path.write_text(json.dumps({"by_mode": runs}, indent=2,
                                        default=lambda o: str(o)))
    print(f"[z400] summary → {summary_path}")
    print(f"[z400] GATES: {gates}")
    print(f"[z400] wall = {wall:.1f} s")


if __name__ == "__main__":
    main()
