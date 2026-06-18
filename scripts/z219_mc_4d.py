"""z219 — Memory Capacity test on 4D transient surrogate.

This is the moment of truth for Path A. If MC > 5 (vs z217's 1.0),
the body-state hypothesis is confirmed and the path forward is clear.

Sweep Cb (1, 5, 20 fF) and dt-vs-tau ratios to find regime where
body dynamics are integrating meaningfully. Per gpt-5 O32 killer
omission: clock/τ mismatch matters.
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

from scripts.nsram_surrogate_4d import NSRAMSurrogate4D
from scripts.z200_topo_rule_sweep import build_topo

OUT = ROOT / "results/z219_mc_4d"; OUT.mkdir(parents=True, exist_ok=True)
SURR_PATH = ROOT / "results/z219_4d_surrogate/surrogate_4d.npz"


def reservoir_run_4d(u, N, W, base_VG1, base_VG2, sign_mask, W_in,
                      surr_4d, Cb, dt, g_in=1.0, g_rec=0.6, leak=0.3,
                      include_vb_in_state=True):
    """Run reservoir with 4D surrogate + Vb dynamics per cell."""
    T = len(u)
    state_id = np.zeros((N, T))
    state_vb = np.zeros((N, T))
    Vb = np.full(N, 0.1)   # initial Vb above zero so feedback kicks in
    feat = np.zeros(N)
    Vd_arr = np.ones(N)
    for t in range(T):
        VG1 = np.clip(base_VG1 + g_in * W_in * u[t], 0.05, 0.7)
        rec = (W @ feat) * sign_mask
        VG2 = np.clip(base_VG2 + g_rec * rec, 0.0, 0.6)
        log_Id, Iii, Ileak = surr_4d.eval(VG1, VG2, Vd_arr, Vb)
        # Update Vb
        net = Iii - Ileak
        Vb = np.clip(Vb + dt * net / Cb, 0.0, 0.7)
        feat = (1.0 - leak) * feat + leak * log_Id
        state_id[:, t] = feat
        state_vb[:, t] = Vb
    if include_vb_in_state:
        return np.concatenate([state_id, state_vb], axis=0)  # (2N, T)
    return state_id


def memory_capacity_4d(N, seed, surr_4d, Cb, dt, K_max=30,
                        T=2000, washout=300, T_train=1500,
                        topo="ER_SPARSE"):
    rng = np.random.default_rng(seed)
    u = rng.uniform(-1.0, 1.0, T)
    base_VG1 = rng.choice([0.2, 0.4, 0.6], N).astype(float)
    base_VG2 = rng.uniform(0.1, 0.4, N).astype(float)
    sign_mask = rng.choice([-1.0, 1.0], N).astype(float)
    W_in = rng.normal(0, 1.0, N)
    W = build_topo(topo, N, rng)
    state = reservoir_run_4d(u, N, W, base_VG1, base_VG2, sign_mask,
                              W_in, surr_4d, Cb, dt)
    X = state.T   # (T, 2N)
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
    surr_4d = NSRAMSurrogate4D(SURR_PATH)
    print(f"[z219] loaded 4D surrogate from {SURR_PATH.name}")
    print(f"\n{'Cb (fF)':>9}  {'dt (s)':>9}  {'τ_eff':>10}  {'MC_total':>10}  "
          f"{'MC[1..5]':>10}  {'MC[6..15]':>11}  {'MC[16..30]':>11}")
    # Sweep: Cb in fF, dt
    grid = []
    for Cb_fF in [1.0, 5.0, 20.0]:
        for dt in [1e-9, 1e-7, 1e-6]:    # 1ns, 100ns, 1µs
            Cb = Cb_fF * 1e-15
            tau_eff = Cb * 1e-10 / 1e-12   # rough: τ = Cb·R_eff with R≈100GΩ
            grid.append((Cb_fF, Cb, dt, tau_eff))

    out = []
    for Cb_fF, Cb, dt, tau in grid:
        mcs = []
        full = []
        for s in [0, 1, 2]:
            mc = memory_capacity_4d(200, s, surr_4d, Cb, dt)
            mcs.append(mc.sum())
            full.append(mc)
        full = np.stack(full).mean(axis=0)
        mc_mean = np.mean(mcs); mc_std = np.std(mcs)
        early = full[:5].sum()
        mid = full[5:15].sum() if len(full) >= 15 else 0.0
        late = full[15:30].sum() if len(full) >= 30 else 0.0
        print(f"  {Cb_fF:>7.1f}  {dt:>9.0e}  {tau:>10.2e}  "
              f"{mc_mean:>5.2f}±{mc_std:.2f}  {early:>10.2f}  "
              f"{mid:>11.2f}  {late:>11.2f}")
        out.append({"Cb_fF": Cb_fF, "dt": dt, "tau": tau,
                     "mc_mean": float(mc_mean), "mc_std": float(mc_std),
                     "early": float(early), "mid": float(mid), "late": float(late),
                     "mc_per_k": full.tolist()})

    print("\nBaseline (z217, no Vb): MC ≈ 1.0")
    print("Decision: max MC > 5 across grid → 4D approach validated")

    (OUT / "summary.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
