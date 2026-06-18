"""z221 — Path A v3: VG2-driven input + diverse fixed-points.

z220 diagnosis: MC capped at ~4 because all cells relax to Vb≈0.55
fixed-point regardless of input. Need cell-level diversity in
fixed-point Vb so each cell encodes input differently.

v3 strategy:
  1. Feed input u(t) via VG2 (modulates body-charging regime, hence
     fixed-point Vb*) instead of VG1 (only modulates channel).
  2. Diverse base_VG2 across cells: each cell sits at different baseline
     Vb*. When input perturbs them, they all respond differently.
  3. Combined: u modulates VG2_local, which shifts Vb* dynamics, the
     residence time in the transient regime contributes to memory.

Sweep (Cb, dt) on dense 4D surrogate from z220.
Compare MC against z220's 3.73 baseline.
"""
from __future__ import annotations
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_k] = "1"
import sys, json, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT = ROOT / "results/z221_vg2_input_v3"; OUT.mkdir(parents=True, exist_ok=True)
SURR_PATH = ROOT / "results/z220_4d_dense/surrogate_4d_dense.npz"


def reservoir_run_v3(u, N, W, base_VG1, base_VG2, sign_mask, W_in,
                      surr_4d, Cb, dt, g_in_VG2=0.4, g_rec_VG1=0.3, leak=0.3):
    """v3: input via VG2 (modulates body-state dynamics), recurrence via VG1."""
    T = len(u)
    state_id = np.zeros((N, T))
    state_vb = np.zeros((N, T))
    Vb = np.full(N, 0.30)   # init mid-range so body can move both ways
    feat = np.zeros(N)
    Vd_arr = np.ones(N)
    for t in range(T):
        # FIX: VG2 takes the INPUT (modulates body-charging regime)
        VG2 = np.clip(base_VG2 + g_in_VG2 * W_in * u[t], 0.0, 0.6)
        # FIX: recurrence is small VG1 perturbation (channel modulation)
        rec = (W @ feat) * sign_mask
        VG1 = np.clip(base_VG1 + g_rec_VG1 * rec, 0.05, 0.7)
        log_Id, Iii, Ileak = surr_4d.eval(VG1, VG2, Vd_arr, Vb)
        net = Iii - Ileak
        Vb = np.clip(Vb + dt * net / Cb, 0.0, 0.7)
        feat = (1.0 - leak) * feat + leak * log_Id
        state_id[:, t] = feat
        state_vb[:, t] = Vb
    return np.concatenate([state_id, state_vb], axis=0)


def memory_capacity_v3(N, seed, surr_4d, Cb, dt, K_max=30,
                        T=2000, washout=300, T_train=1500,
                        topo="ER_SPARSE",
                        g_in_VG2=0.4, g_rec_VG1=0.3, leak=0.3):
    rng = np.random.default_rng(seed)
    u = rng.uniform(-1.0, 1.0, T)
    base_VG1 = rng.uniform(0.2, 0.5, N).astype(float)   # diverse VG1
    base_VG2 = rng.uniform(0.05, 0.55, N).astype(float)  # WIDE VG2 for diverse Vb*
    sign_mask = rng.choice([-1.0, 1.0], N).astype(float)
    W_in = rng.normal(0, 1.0, N)
    from scripts.z200_topo_rule_sweep import build_topo
    W = build_topo(topo, N, rng)
    state = reservoir_run_v3(u, N, W, base_VG1, base_VG2, sign_mask,
                              W_in, surr_4d, Cb, dt,
                              g_in_VG2=g_in_VG2, g_rec_VG1=g_rec_VG1, leak=leak)
    X = state.T
    X = np.hstack([X, np.ones((X.shape[0], 1))])

    mc_per_k = []
    for k in range(1, K_max + 1):
        v_train = slice(max(washout, k), T_train)
        Xt = X[v_train]
        targ_t = u[v_train.start - k : v_train.stop - k]
        if len(targ_t) < 50: break
        w = np.linalg.solve(Xt.T @ Xt + 1e-4 * np.eye(X.shape[1]),
                            Xt.T @ targ_t)
        v_test = slice(T_train, T)
        Xv = X[v_test]
        targ_v = u[v_test.start - k : v_test.stop - k]
        if len(targ_v) < 50: break
        pred = Xv @ w
        if pred.std() < 1e-9 or targ_v.std() < 1e-9:
            r2 = 0.0
        else:
            r = np.corrcoef(pred, targ_v)[0, 1]
            r2 = float(r * r)
        mc_per_k.append(r2)
    return np.array(mc_per_k)


def main():
    from scripts.nsram_surrogate_4d import NSRAMSurrogate4D
    surr_4d = NSRAMSurrogate4D(SURR_PATH)
    print(f"[z221] loaded dense 4D surrogate")

    # Sweep Cb × dt × g_in_VG2
    sweep = []
    for Cb_fF in [1.0, 5.0, 20.0]:
        for dt in [1e-7, 1e-6, 1e-5]:
            for g_in_VG2 in [0.2, 0.4]:
                sweep.append((Cb_fF * 1e-15, dt, g_in_VG2))

    print(f"\n{'Cb':>5}  {'dt':>8}  {'g_VG2':>6}  {'MC_total':>10}  {'early':>7}  {'mid':>7}  {'late':>7}")
    out = []
    best = (None, 0.0)
    for Cb, dt, g in sweep:
        mcs = []
        full = []
        for s in [0, 1, 2]:
            try:
                mc = memory_capacity_v3(200, s, surr_4d, Cb, dt, g_in_VG2=g)
                mcs.append(mc.sum())
                full.append(mc)
            except Exception as e:
                pass
        if not mcs: continue
        full = np.stack(full).mean(axis=0)
        mc_mean = float(np.mean(mcs)); mc_std = float(np.std(mcs))
        early = float(full[:5].sum())
        mid = float(full[5:15].sum() if len(full) >= 15 else 0)
        late = float(full[15:30].sum() if len(full) >= 30 else 0)
        if mc_mean > best[1]:
            best = ((Cb, dt, g), mc_mean)
        print(f"  {Cb*1e15:>3.0f}fF  {dt:>8.0e}  {g:>6.2f}  "
              f"{mc_mean:>5.2f}±{mc_std:.2f}  {early:>7.2f}  {mid:>7.2f}  {late:>7.2f}")
        out.append({"Cb": Cb, "dt": dt, "g_in_VG2": g,
                     "mc_mean": mc_mean, "mc_std": mc_std,
                     "early": early, "mid": mid, "late": late,
                     "mc_per_k": full.tolist()})

    print(f"\nBaselines (best of each iteration):")
    print(f"  z217 (no Vb):         MC = 1.00")
    print(f"  z218 (passive Vb):    MC = 1.6")
    print(f"  z219 (5-pt Vb, VG1 input):  MC = 2.5")
    print(f"  z220 (10-pt Vb, VG1 input): MC = 3.73")
    print(f"  z221 (10-pt Vb, VG2 input): MC = {best[1]:.2f} at {best[0]}")

    (OUT / "summary.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
