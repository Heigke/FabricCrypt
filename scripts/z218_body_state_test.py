"""z218 — Proof-of-concept: does adding a body-state variable lift MC?

Hypothesis: NSRAMSurrogate is memoryless (z217: MC≈1). If we add a
SOFTWARE per-cell body-charge state that integrates input over time
with a tunable time constant, we should see MC grow with τ.

This is NOT physical — it's a math test of the upper bound on what
adding ANY state variable could give us. If even an idealized first-
order integrator with τ=20 lifts MC from 1 to >10, we know:
  - The diagnosis is right (state-variable absence is the issue)
  - A real transient surrogate (with Vb dynamics ~ τ_body·d/dt)
    would unlock long-memory tasks
  - Phase-B is well-justified

If this lift doesn't happen, the failure goes deeper than missing
state — probably saturation in the surrogate's nonlinearity.

Test 4 configs of body-state τ ∈ {0 (no state), 5, 20, 100}.
At each: run the existing reservoir BUT augment the readout features
with the body-state vector. Measure MC.
"""
from __future__ import annotations
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
import sys, json, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/z218_body_state"; OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))

from scripts.z200_topo_rule_sweep import build_topo
from scripts.nsram_surrogate import NSRAMSurrogate

surr = NSRAMSurrogate.build_or_load(grid_size=(20, 20, 25))


def run_with_body_state(N, seed, tau_body, T=2000, washout=200, T_train=1500,
                         g_in=1.0, g_rec=0.6, leak=0.3, topo="ER_SPARSE"):
    """Run reservoir with optional per-cell body-state integrator.

    body_charge[t] = (1 - 1/tau_body) * body_charge[t-1] + (1/tau_body) * Iii_proxy
    where Iii_proxy = relu(input + recurrent_drive - threshold)

    If tau_body == 0: no body state (current baseline).
    Bigger tau = longer memory.

    Output features = [log_id_states, body_charge_states, 1_bias]
    """
    rng = np.random.default_rng(seed)
    u = rng.uniform(-1.0, 1.0, T)
    base_VG1 = rng.choice([0.2, 0.4, 0.6], N).astype(float)
    base_VG2 = rng.uniform(0.1, 0.4, N).astype(float)
    sign_mask = rng.choice([-1.0, 1.0], N).astype(float)
    W_in = rng.normal(0, 1.0, N)
    W = build_topo(topo, N, rng)

    feat = np.zeros(N)
    body = np.zeros(N)
    states_log = np.zeros((N, T))
    states_body = np.zeros((N, T))
    Vd = np.ones(N)
    body_threshold = 0.5  # arbitrary; just need the integrator to be active

    for t in range(T):
        # Same reservoir as z216
        VG1 = np.clip(base_VG1 + g_in * W_in * u[t], 0.05, 0.7)
        rec = (W @ feat) * sign_mask
        VG2 = np.clip(base_VG2 + g_rec * rec, 0.0, 0.6)
        log_id = surr.eval(VG1, VG2, Vd)
        feat = (1.0 - leak) * feat + leak * log_id
        states_log[:, t] = feat

        # Optional body-state integrator
        if tau_body > 0:
            # "Iii proxy" = active when log_id is high (cell conducting strongly)
            # Use sigmoid of log_id to bound contribution.
            iii_proxy = np.maximum(log_id - (-8), 0.0)  # log10|Id| > -8 = active
            body = (1.0 - 1.0/tau_body) * body + (1.0/tau_body) * iii_proxy
            states_body[:, t] = body

    # Build feature matrix
    if tau_body > 0:
        X = np.concatenate([states_log.T, states_body.T,
                             np.ones((T, 1))], axis=1)
    else:
        X = np.concatenate([states_log.T, np.ones((T, 1))], axis=1)

    # Memory capacity
    K_max = 30
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
    print("Memory Capacity with body-state integrator (proof of concept):")
    print(f"{'tau_body':>10}  {'MC_total':>10}  {'MC[1..5]':>10}  {'MC[6..15]':>11}  {'MC[16..30]':>11}")
    configs = [0, 5, 20, 100]
    seeds = [0, 1, 2]
    out = []
    for tau in configs:
        mc_sums = []
        full = []
        for s in seeds:
            mc = run_with_body_state(N=200, seed=s, tau_body=tau)
            mc_sums.append(mc.sum())
            full.append(mc)
        full = np.stack(full).mean(axis=0)
        early = full[:5].sum()
        mid = full[5:15].sum() if len(full) >= 15 else 0
        late = full[15:30].sum() if len(full) >= 30 else 0
        mean = np.mean(mc_sums); std = np.std(mc_sums)
        print(f"  {tau:>8}    {mean:>5.2f}±{std:.2f}  "
              f"{early:>10.2f}  {mid:>11.2f}  {late:>11.2f}")
        out.append({"tau": tau, "mc_mean": float(mean), "mc_std": float(std),
                    "early_5": float(early), "mid_10": float(mid), "late_15": float(late),
                    "mc_per_k": full.tolist()})

    print("\nInterpretation:")
    print("  tau=0 baseline ≈ 1.0  (z217 confirmed memoryless)")
    print("  If tau>0 lifts MC significantly: state-variable diagnosis confirmed,")
    print("   transient surrogate would unlock long-memory tasks.")
    print("  If no lift: nonlinearity saturation is the deeper issue.")

    (OUT / "summary.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
