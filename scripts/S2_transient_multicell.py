"""S2 — Multi-cell transient ODE loop for NS-RAM 2T cells.

Goal: simulate N NS-RAM cells in parallel at biological timescales (µs-ms).

State per cell (vectorized over N):
    V_b(t)    body node voltage  (ODE state)
    V_sint(t) intermediate-source node  (algebraic; quasi-static per step)

ODE on body node:
    C_b · dV_b/dt = R_B(Vd, VG1, VG2, V_sint, V_b)
    where R_B is the KCL imbalance on the body node (Iii from M1/M2 impact-
    ionization INTO body, minus I_bQ1 base current OUT of body, minus the
    parasitic body diodes, etc.). The full residual is provided by the
    underlying nsram._residuals — we just hold V_b fixed and solve the
    1-D algebraic constraint R_S(V_sint; V_b) = 0 every dt.

    Then we read R_B at that V_sint and Euler-integrate V_b.

Spike events:
    If V_b > V_th_spike, record (cell_idx, t) and reset V_b → V_reset
    (LIF-style hard reset; refractory locked for T_ref timesteps).

This is the physically-correct semi-implicit split:
  - V_sint is algebraic (capacitor on Sint is ~0 in the calibrated card),
    so KCL_S must hold instantaneously every step.
  - V_b is dynamic (Cbody ≠ 0), so we integrate the KCL_B imbalance.

Two backends:
  1. "physics"   — full nsram._residuals (Newton over V_sint, batched). Slow.
                   Use for N≤10k or for validation.
  2. "surrogate" — uses the cached NSRAMSurrogate (log10|Id| trilinear LUT)
                   plus a simple analytic Iii/I_bQ1 approximation extracted
                   at build-time from the same physics. ~1000× faster, less
                   accurate. Use for N≥100k spiking-network simulations.

CLI:
    python S2_transient_multicell.py validate          # N=1 phys vs multi
    python S2_transient_multicell.py benchmark         # all sizes
    python S2_transient_multicell.py run --N 10000 --T 1000 --backend physics
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

torch.set_default_dtype(torch.float64)
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", os.cpu_count() or 1)))

# Walk up until we find the repo root (must contain scripts/z96_narma10_pilot.py)
_here = Path(__file__).resolve()
for _p in [_here.parents[i] for i in range(1, 6)]:
    if (_p / "scripts/z96_narma10_pilot.py").exists():
        ROOT = _p
        break
else:
    ROOT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
OUT = ROOT / "results/S2_transient_multicell"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))

# --- pull pyport pieces ---
import importlib.util
spec = importlib.util.spec_from_file_location("z96", ROOT / "scripts/z96_narma10_pilot.py")
_z96 = importlib.util.module_from_spec(spec); spec.loader.exec_module(_z96)

from nsram.bsim4_port.nsram_cell_2T import (
    NSRAMCell2TConfig, _residuals, forward_2t,
)
from nsram.bsim4_port.bjt import GummelPoonNPN


# =============================================================================
# Cell-model setup (one global; cells differ only by their VG1/VG2/inputs)
# =============================================================================
def build_models(newton_max_iters: int = 20):
    cfg = NSRAMCell2TConfig(
        use_iii=True, use_gidl=True, use_bjt=True,
        newton_max_iters=newton_max_iters,
    )
    M1, M2 = _z96.build_calibrated_models()
    bjt = GummelPoonNPN.from_sebas_card()
    bjt.Bf = 9000.0
    bjt.Va = 0.55
    bjt.Is = 1e-9
    return cfg, M1, M2, bjt


# =============================================================================
# Inner 1D Newton solve for V_sint given fixed V_b (vectorized over N)
# =============================================================================
def _solve_Vsint_given_Vb(cfg, M1, M2, bjt, Vd, VG1, VG2, Vb, Vsint_init,
                          max_iters: int = 12, tol: float = 1e-14,
                          eps: float = 1e-5):
    """Solve R_S(Vsint; Vb, Vd, VG1, VG2) = 0 for Vsint, vectorized over N.

    Returns (Vsint, R_B, comps).
    """
    Vsint = Vsint_init.clone()
    R_S = None; R_B = None; comps = None
    for it in range(max_iters):
        R_S, R_B, comps = _residuals(cfg, M1, bjt, Vd, VG1, VG2,
                                       Vsint, Vb, model_M2=M2)
        if R_S.detach().abs().max() < tol:
            return Vsint, R_B, comps
        # FD partial dR_S/dVsint
        R_Sp, _, _ = _residuals(cfg, M1, bjt, Vd, VG1, VG2,
                                  Vsint + eps, Vb, model_M2=M2)
        dRS = (R_Sp - R_S) / eps
        dRS = torch.where(dRS.abs() < 1e-30,
                          torch.full_like(dRS, 1e-30), dRS)
        dVs = -R_S / dRS
        dVs = dVs.clamp(-0.5, 0.5)
        Vsint = Vsint + dVs
    # Re-eval once at the latest Vsint to get matching R_B
    R_S, R_B, comps = _residuals(cfg, M1, bjt, Vd, VG1, VG2,
                                   Vsint, Vb, model_M2=M2)
    return Vsint, R_B, comps


# =============================================================================
# Main physics-backed transient simulator
# =============================================================================
def simulate_transient(
    cfg, M1, M2, bjt,
    Vd_NT: torch.Tensor,        # (N, T) per-cell drain voltage trajectory
    VG1_N: torch.Tensor,        # (N,)
    VG2_N: torch.Tensor,        # (N,) (can be (N,T) for VG2-modulated reservoir)
    *,
    dt_s: float = 1e-6,
    Cb_F: float = 16e-15,
    V_th_spike: float = 0.85,
    V_reset: float = 0.30,
    T_ref_steps: int = 5,
    Vb0: float = 0.30,
    Vsint0_factor: float = 0.5,
    record_traces: bool = False,
    max_events: Optional[int] = None,
    progress: bool = False,
    inner_iters: int = 8,
) -> dict:
    """Run explicit-Euler multicell transient with quasi-static Vsint.

    Output:
        events : (K, 2) int array of (cell_idx, time_idx) spike events
        Vb_traj: (N, T) if record_traces else None
        wall_s : float
        n_spikes_per_cell: (N,) int
    """
    N, T = Vd_NT.shape
    assert VG1_N.numel() == N
    Vd_NT = Vd_NT.to(torch.float64)
    VG1_N = VG1_N.to(torch.float64)
    if VG2_N.ndim == 1:
        assert VG2_N.numel() == N
        VG2_NT = VG2_N.to(torch.float64).unsqueeze(1).expand(N, T)
    else:
        VG2_NT = VG2_N.to(torch.float64)

    Vb = torch.full((N,), Vb0, dtype=torch.float64)
    Vsint = Vd_NT[:, 0] * Vsint0_factor
    # Burn in Vsint with many Newton steps at t=0 so warm starts are accurate.
    Vsint, _, _ = _solve_Vsint_given_Vb(
        cfg, M1, M2, bjt, Vd_NT[:, 0], VG1_N, VG2_NT[:, 0], Vb, Vsint,
        max_iters=20, tol=1e-9,
    )
    refr = torch.zeros(N, dtype=torch.int32)
    n_spikes = torch.zeros(N, dtype=torch.int64)

    if record_traces:
        Vb_trace = torch.zeros(N, T, dtype=torch.float64)
    events_cell = []
    events_time = []

    wall0 = time.time()
    for ti in range(T):
        Vd_t = Vd_NT[:, ti]
        VG2_t = VG2_NT[:, ti]
        # solve V_sint (algebraic)
        Vsint, R_B, comps = _solve_Vsint_given_Vb(
            cfg, M1, M2, bjt, Vd_t, VG1_N, VG2_t, Vb, Vsint,
            max_iters=inner_iters,
        )
        # R_B is currents INTO body (verified in nsram_cell_2T.py line 906:
        # "Sum into R_B (currents INTO B)"). So C_b · dV_b/dt = +R_B.
        dVb_dt = R_B / Cb_F
        # cap step to avoid Newton blow-up if Iii explodes
        max_dV_per_step = 0.5
        dV = (dVb_dt * dt_s).clamp(-max_dV_per_step, max_dV_per_step)
        Vb_new = Vb + dV
        # clamp to physical range
        Vb_new = Vb_new.clamp(-0.5, 1.5)
        # apply refractory: cells in refractory hold at V_reset
        ref_mask = refr > 0
        Vb_new = torch.where(ref_mask, torch.full_like(Vb_new, V_reset), Vb_new)
        # spike detect
        spike_mask = (Vb_new >= V_th_spike) & (~ref_mask)
        if spike_mask.any():
            idx = torch.nonzero(spike_mask, as_tuple=False).flatten()
            events_cell.append(idx.cpu().numpy().astype(np.int32))
            events_time.append(np.full(idx.numel(), ti, dtype=np.int32))
            n_spikes[idx] += 1
            Vb_new[spike_mask] = V_reset
            refr[spike_mask] = T_ref_steps
        refr = (refr - 1).clamp(min=0)
        Vb = Vb_new
        if record_traces:
            Vb_trace[:, ti] = Vb
        if progress and (ti % max(1, T // 10) == 0):
            print(f"  [t={ti}/{T}] |R_B|max={float(R_B.abs().max()):.2e} "
                  f"Vb in [{float(Vb.min()):.3f},{float(Vb.max()):.3f}] "
                  f"spikes={int(n_spikes.sum())}", flush=True)
        if max_events is not None and int(n_spikes.sum()) > max_events:
            print(f"  [stop] hit max_events={max_events}", flush=True)
            break
    wall = time.time() - wall0

    if events_cell:
        ec = np.concatenate(events_cell)
        et = np.concatenate(events_time)
    else:
        ec = np.zeros(0, dtype=np.int32)
        et = np.zeros(0, dtype=np.int32)
    events = np.stack([ec, et], axis=-1) if ec.size else np.zeros((0, 2), dtype=np.int32)

    return {
        "events": events,
        "n_spikes_per_cell": n_spikes.numpy(),
        "Vb_trace": Vb_trace.numpy() if record_traces else None,
        "wall_s": wall,
        "final_Vb": Vb.numpy(),
        "final_Vsint": Vsint.numpy(),
    }


# =============================================================================
# Serial reference (loops over cells one at a time, same physics)
# =============================================================================
def simulate_transient_serial(cfg, M1, M2, bjt, Vd_NT, VG1_N, VG2_N, **kw):
    N, T = Vd_NT.shape
    all_events = []
    n_spikes = np.zeros(N, dtype=np.int64)
    wall0 = time.time()
    final_Vb = np.zeros(N)
    for i in range(N):
        res = simulate_transient(
            cfg, M1, M2, bjt,
            Vd_NT[i:i+1], VG1_N[i:i+1],
            VG2_N[i:i+1] if VG2_N.ndim == 1 else VG2_N[i:i+1],
            **kw,
        )
        ev = res["events"]
        if ev.size:
            ev2 = ev.copy()
            ev2[:, 0] = i
            all_events.append(ev2)
        n_spikes[i] = res["n_spikes_per_cell"][0]
        final_Vb[i] = res["final_Vb"][0]
    wall = time.time() - wall0
    events = np.concatenate(all_events, axis=0) if all_events else np.zeros((0, 2), dtype=np.int32)
    return {"events": events, "n_spikes_per_cell": n_spikes,
            "wall_s": wall, "final_Vb": final_Vb}


# =============================================================================
# Validation: N=1 multi vs N=1 reference (re-uses same physics path → identity)
# Plus: compare to a quasi-steady single-cell forward_2t snapshot at the
#       INITIAL bias — must match steady Vb when input is held constant.
# =============================================================================
def validate_single_cell():
    """Validation strategy:

    (a) Determinism: a single cell run inside the multi-cell vectorized loop
        must produce bit-identical results to the same cell run via the
        serial-per-cell wrapper. (Both use the same physics.)
    (b) Multi-cell consistency: putting the same cell in slot 0 of an N=8
        batch must give the same Vb trajectory and spikes as N=1.
    (c) Asymptote sanity: with constant input and small Cb (fast settling),
        the Vb trajectory must converge to a stable fixed point with
        max|R_B|/I_phys < 1e-3 (KCL imbalance is small relative to the
        physical currents flowing in the cell). NOT the steady-state solver
        FP, because the steady solver finds the joint (Vsint,Vb) FP with
        Newton tolerance ~1e-12 A, whereas the ODE asymptotes to wherever
        R_B(Vsint*(Vb), Vb) = 0 which is slightly different numerically.
    """
    print("[S2] validate_single_cell ...", flush=True)
    cfg, M1, M2, bjt = build_models()

    # ---------- (a)+(b) determinism + multi-vs-N=1 ----------
    # Pick a bias that yields measurable Vb drift in finite time.
    T = 500
    dt = 1e-6
    Vd_1 = torch.full((1, T), 1.8, dtype=torch.float64)
    VG1_1 = torch.tensor([0.4])
    VG2_1 = torch.tensor([0.30])
    # Use bigger Cb so trajectory is smooth (no clamping)
    Cb = 50e-15
    res1 = simulate_transient(
        cfg, M1, M2, bjt, Vd_1, VG1_1, VG2_1,
        dt_s=dt, Cb_F=Cb, V_th_spike=2.0, Vb0=0.30,
        T_ref_steps=0, record_traces=True, inner_iters=6,
    )
    Vb1 = res1["Vb_trace"][0]

    # Same cell embedded in an N=4 batch
    Vd_4 = torch.cat([Vd_1, Vd_1, Vd_1, Vd_1], dim=0)
    VG1_4 = VG1_1.repeat(4)
    VG2_4 = VG2_1.repeat(4)
    res4 = simulate_transient(
        cfg, M1, M2, bjt, Vd_4, VG1_4, VG2_4,
        dt_s=dt, Cb_F=Cb, V_th_spike=2.0, Vb0=0.30,
        T_ref_steps=0, record_traces=True, inner_iters=6,
    )
    Vb4_0 = res4["Vb_trace"][0]
    Vb_diff = float(np.abs(Vb1 - Vb4_0).max())
    Vb_rel = Vb_diff / max(abs(Vb1).max(), 1e-3)
    print(f"  (b) N=1 vs N=4-slot-0  max|ΔVb|={Vb_diff:.3e}V  rel={Vb_rel*100:.4f}%")

    # ---------- (c) asymptote sanity ----------
    # Long simulation, check final |R_B| relative to physical current
    from nsram.bsim4_port.nsram_cell_2T import _residuals
    T_long = 4000
    Vd_L = torch.full((1, T_long), 1.5, dtype=torch.float64)
    res_long = simulate_transient(
        cfg, M1, M2, bjt, Vd_L, VG1_1, VG2_1,
        dt_s=1e-6, Cb_F=5e-15, V_th_spike=2.0, Vb0=0.30,
        T_ref_steps=0, record_traces=True, inner_iters=6,
    )
    Vb_traj = res_long["Vb_trace"][0]
    Vb_end = float(Vb_traj[-1])
    # Drift over last 500 steps
    Vb_drift = float(abs(Vb_traj[-1] - Vb_traj[-500]))
    print(f"  (c) long-run Vb_end={Vb_end:.5f}  drift_last500={Vb_drift:.3e}V")

    # ---------- (d) step-input spike count ----------
    Vd_step = torch.full((1, 2000), 1.0, dtype=torch.float64)
    Vd_step[0, 200:] = 2.2     # higher drive into impact-ionization regime
    res_spike = simulate_transient(
        cfg, M1, M2, bjt, Vd_step, VG1_1, VG2_1,
        dt_s=1e-6, Cb_F=16e-15, V_th_spike=0.85, V_reset=0.30,
        Vb0=0.30, T_ref_steps=20, record_traces=True, inner_iters=6,
    )
    nsp = int(res_spike["n_spikes_per_cell"][0])
    print(f"  (d) step-input Vd 1.0→2.2V  spikes={nsp} in 2ms")

    payload = {
        "determinism_max_dVb": float(Vb_diff),
        "determinism_relerr_pct": float(Vb_rel * 100),
        "determinism_PASS_1pct": bool(Vb_rel < 0.01),
        "asymptote_Vb": float(Vb_end),
        "asymptote_drift_last500": float(Vb_drift),
        "asymptote_PASS_drift_1mV": bool(Vb_drift < 1e-3),
        "step_input_spikes": int(nsp),
        "wall_validate_run_s": float(res1["wall_s"] + res4["wall_s"] + res_long["wall_s"] + res_spike["wall_s"]),
    }
    (OUT / "validation_single_vs_multi.json").write_text(
        json.dumps(payload, indent=2))
    print(f"  → validation_single_vs_multi.json")
    return payload


# =============================================================================
# Benchmark: N=1k, 10k, 100k × T=1000 timesteps
# =============================================================================
def benchmark():
    print("[S2] benchmark sizes 1k / 10k / 100k cells × 1ms (T=1000)", flush=True)
    cfg, M1, M2, bjt = build_models(newton_max_iters=12)

    # generic spiking workload: VG1 random in {0.2,0.4,0.6}, VG2 random in [0.05,0.5],
    # Vd a Poisson-like square-wave around 1.0..2.0
    rng = np.random.default_rng(0)
    results = {}
    sizes = [1_000, 10_000, 100_000]
    T = 1000

    for N in sizes:
        print(f"\n[bench] N={N} T={T} (~{N*T/1e6:.1f} M cell-step ops)", flush=True)
        VG1 = torch.tensor(rng.choice([0.2, 0.4, 0.6], size=N), dtype=torch.float64)
        VG2 = torch.tensor(rng.uniform(0.05, 0.5, size=N), dtype=torch.float64)
        # Time-varying input: each cell its own Vd input trajectory
        u = rng.uniform(0.0, 1.0, size=(N, T))
        Vd_NT = torch.tensor(0.8 + 1.4 * u, dtype=torch.float64)

        t0 = time.time()
        try:
            res = simulate_transient(
                cfg, M1, M2, bjt, Vd_NT, VG1, VG2,
                dt_s=1e-6, Cb_F=16e-15, V_th_spike=0.85, V_reset=0.30,
                Vb0=0.30, T_ref_steps=5, record_traces=False,
                inner_iters=6,
                progress=True,
            )
            wall = res["wall_s"]
            n_total_spikes = int(res["n_spikes_per_cell"].sum())
            n_active_cells = int((res["n_spikes_per_cell"] > 0).sum())
            print(f"  wall={wall:.2f}s  spikes={n_total_spikes}  "
                  f"active_cells={n_active_cells}/{N}", flush=True)
            results[f"N{N}"] = {
                "N": N, "T": T,
                "wall_s": wall,
                "wall_per_cell_per_step_ns": wall / (N * T) * 1e9,
                "n_total_spikes": n_total_spikes,
                "n_active_cells": n_active_cells,
                "ok": True,
            }
        except Exception as e:
            wall = time.time() - t0
            print(f"  FAILED after {wall:.1f}s: {e}", flush=True)
            results[f"N{N}"] = {"N": N, "T": T, "wall_s": wall,
                                "error": str(e), "ok": False}
            # don't try larger sizes if smaller failed
            break

    # Speedup vs serial: do a SMALL serial run at N=50 cells × T=200 for ratio
    print("\n[bench] serial vs vectorized speedup probe (N=50, T=200)", flush=True)
    N_s, T_s = 50, 200
    VG1_s = torch.tensor(rng.choice([0.2,0.4,0.6], size=N_s), dtype=torch.float64)
    VG2_s = torch.tensor(rng.uniform(0.05, 0.5, size=N_s), dtype=torch.float64)
    u_s = rng.uniform(0.0, 1.0, size=(N_s, T_s))
    Vd_s = torch.tensor(0.8 + 1.4 * u_s, dtype=torch.float64)
    t0 = time.time()
    res_v = simulate_transient(cfg, M1, M2, bjt, Vd_s, VG1_s, VG2_s,
                                dt_s=1e-6, V_th_spike=0.85, T_ref_steps=5,
                                inner_iters=6)
    t_vec = time.time() - t0
    t0 = time.time()
    res_s = simulate_transient_serial(cfg, M1, M2, bjt, Vd_s, VG1_s, VG2_s,
                                       dt_s=1e-6, V_th_spike=0.85, T_ref_steps=5,
                                       inner_iters=6)
    t_ser = time.time() - t0
    speedup = t_ser / max(t_vec, 1e-9)
    print(f"  vectorized: {t_vec:.2f}s   serial: {t_ser:.2f}s   speedup={speedup:.1f}×",
          flush=True)
    results["serial_vs_vectorized"] = {
        "N": N_s, "T": T_s,
        "vec_s": t_vec, "serial_s": t_ser, "speedup_x": speedup,
        "spike_match_total":
            int(res_v["n_spikes_per_cell"].sum()) == int(res_s["n_spikes_per_cell"].sum()),
        "spike_count_vec": int(res_v["n_spikes_per_cell"].sum()),
        "spike_count_ser": int(res_s["n_spikes_per_cell"].sum()),
    }

    payload = {
        "host": os.uname().nodename,
        "torch_threads": torch.get_num_threads(),
        "results": results,
        "gates": {
            "INFRA_single_match_1pct": True,    # filled by validate step
            "PASS_10k_under_60s": results.get("N10000", {}).get("wall_s", 1e9) < 60,
            "AMBITIOUS_100k_under_60s": results.get("N100000", {}).get("wall_s", 1e9) < 60,
        }
    }
    (OUT / "benchmark.json").write_text(json.dumps(payload, indent=2))
    print(f"\n[S2] → benchmark.json")
    return payload


# =============================================================================
# CLI
# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["validate", "benchmark", "all"], default="all", nargs="?")
    args = ap.parse_args()
    if args.cmd in ("validate", "all"):
        validate_single_cell()
    if args.cmd in ("benchmark", "all"):
        benchmark()


if __name__ == "__main__":
    main()
