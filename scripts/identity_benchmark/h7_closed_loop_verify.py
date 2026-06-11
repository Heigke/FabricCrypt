"""Closed-loop verification probe — is the SMC loop load-bearing for prediction?

Per Buhrmann/Di Paolo SMC theory, embodiment requires that the agent's action
affect what it later senses. We've shown compute-load shifts the substrate channels
(load_d > 0.5). Now we ask the predictive question:

  Does knowing the agent's recent compute action IMPROVE next-substrate prediction
  beyond what S-history alone gives?

Design:
  - Substrate at 500Hz on the 4 SMC keeper channels (drop C06_fast).
  - Generate a known compute-action signal: alternate (busy GPU matmul, idle)
    in random ~3-7s bouts. Action u_t in {0, 1}.
  - Fit two ridge predictors with same lag depth on the SAME train/test split:
      A) S_{t+1} = f(S_{t-L:t})              (baseline — substrate auto-regressive)
      B) S_{t+1} = f(S_{t-L:t}, u_{t-L:t})   (closed-loop aware)
  - Report per-channel R² gain ΔR² = R²_B − R²_A.

Pass criterion: ΔR² > 0.05 on ≥3 of 4 SMC channels (channel-level conjunction
defeats spurious gain on one chatty channel).

Honest failure mode: if action is fully predictable from S alone (causality is
sense → act → sense and the act adds nothing new conditional on prior sense),
ΔR² ≈ 0. That would mean the closed loop is real but not informative — a strict
reformulation of the embodiment story.
"""
from __future__ import annotations
import os, sys, time, threading, json
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).parent))
from substrate_keepers_v4 import SubstrateStateV4, KEEPER_NAMES, SMC_KEEPER_INDICES

OUT = Path(__file__).resolve().parents[2] / "results/IDENTITY_H7_2026-06-09"
OUT.mkdir(parents=True, exist_ok=True)

DURATION_S = 180
BOUT_MIN_S = 2.0
BOUT_MAX_S = 5.0
LAG = int(os.environ.get("CL_LAG", "24"))   # samples; 24=48ms, 128=256ms, 256=512ms
SEED = 7


def action_schedule(rng, total_s):
    """Random alternating bouts of (idle=0, busy=1)."""
    t = 0.0; out = []
    cur = int(rng.integers(0, 2))
    while t < total_s:
        dur = float(rng.uniform(BOUT_MIN_S, BOUT_MAX_S))
        out.append((t, min(t + dur, total_s), cur))
        t += dur
        cur = 1 - cur
    return out


def action_at(schedule, t):
    for s, e, v in schedule:
        if s <= t < e: return v
    return schedule[-1][2]


def busy_thread(stop_evt, action_state):
    """Stress GPU when action_state[0] == 1, sleep otherwise."""
    if torch.cuda.is_available():
        x = torch.randn(1536, 1536, device="cuda")
        while not stop_evt.is_set():
            if action_state[0] == 1:
                x = torch.matmul(x, x.T); torch.cuda.synchronize()
                x = (x - x.mean()) / (x.std() + 1e-3)
            else:
                time.sleep(0.02)
    else:
        import numpy as _np
        x = _np.random.randn(768, 768)
        while not stop_evt.is_set():
            if action_state[0] == 1:
                x = x @ x.T
                x = (x - x.mean()) / (x.std() + 1e-3)
            else:
                time.sleep(0.02)


def fit_ridge(X, Y, lam=1e-2):
    lam_eff = lam * len(X)
    W = np.linalg.solve(X.T @ X + lam_eff * np.eye(X.shape[1]), X.T @ Y)
    return W


def per_ch_r2(pred, target):
    ss_res = ((pred - target) ** 2).mean(axis=0)
    ss_tot = ((target - target.mean(axis=0)) ** 2).mean(axis=0) + 1e-9
    return 1 - ss_res / ss_tot


def make_lag_features(seq, u, lags):
    """Return (S_only_feats, S_plus_u_feats, targets) at time t for t in [lags, T-1)."""
    T, C = seq.shape
    n = T - lags - 1
    S_only = np.zeros((n, lags * C), dtype=np.float32)
    S_plus = np.zeros((n, lags * C + lags), dtype=np.float32)
    Y = np.zeros((n, C), dtype=np.float32)
    for k, t in enumerate(range(lags, T - 1)):
        S_only[k] = seq[t - lags:t].flatten()
        S_plus[k, :lags * C] = seq[t - lags:t].flatten()
        S_plus[k, lags * C:] = u[t - lags:t]
        Y[k] = seq[t + 1]
    return S_only, S_plus, Y


def main():
    rng = np.random.default_rng(SEED)
    print("=" * 70)
    print(f"CLOSED-LOOP VERIFICATION — ΔR² test on {DURATION_S}s recording")
    print("=" * 70)

    schedule = action_schedule(rng, DURATION_S)
    n_busy = sum(e - s for s, e, v in schedule if v == 1)
    print(f"\nAction schedule: {len(schedule)} bouts, busy fraction={n_busy/DURATION_S:.2%}")

    state = SubstrateStateV4(hz_target=500); state.start()
    time.sleep(0.5)
    action_state = [0]
    stop_evt = threading.Event()
    th = threading.Thread(target=busy_thread, args=(stop_evt, action_state), daemon=True)
    th.start()

    samples = []         # (t_rel, action, w_row[5])
    t0 = time.perf_counter()
    last_idx_seen = -1
    print("\ncollecting...")
    while True:
        now = time.perf_counter() - t0
        if now >= DURATION_S: break
        a = action_at(schedule, now)
        action_state[0] = a
        # snapshot the latest substrate row
        w = state.latest_window(length=4)  # tail of ring
        s_row = w[-1]                      # latest (5-vec, keepers)
        samples.append((now, a, s_row.copy()))
        time.sleep(0.002)   # log at ~500Hz

    stop_evt.set()
    state.stop()
    th.join(timeout=3.0)

    t_arr = np.array([s[0] for s in samples], dtype=np.float32)
    u_arr = np.array([s[1] for s in samples], dtype=np.float32)
    S_arr = np.stack([s[2] for s in samples]).astype(np.float32)
    # Use SMC channels only
    S_smc = S_arr[:, list(SMC_KEEPER_INDICES)]
    smc_names = [KEEPER_NAMES[i] for i in SMC_KEEPER_INDICES]
    print(f"\nLogged {len(t_arr)} frames, {S_smc.shape[1]} SMC channels: {smc_names}")
    # Robust per-channel z-score
    med = np.median(S_smc, axis=0)
    mad = np.maximum(1.4826 * np.median(np.abs(S_smc - med), axis=0), 1.0)
    Sz = ((S_smc - med) / mad).astype(np.float32)
    Sz = (8.0 * np.tanh(Sz / 8.0)).astype(np.float32)

    # Build features
    Xa, Xb, Y = make_lag_features(Sz, u_arr, LAG)
    n_total = len(Y)
    split = int(0.7 * n_total)
    print(f"feature rows: train={split}, test={n_total - split}, dim_A={Xa.shape[1]}, dim_B={Xb.shape[1]}")

    Wa = fit_ridge(Xa[:split], Y[:split])
    Wb = fit_ridge(Xb[:split], Y[:split])
    pred_a = Xa[split:] @ Wa
    pred_b = Xb[split:] @ Wb
    r2_a = per_ch_r2(pred_a, Y[split:])
    r2_b = per_ch_r2(pred_b, Y[split:])
    gain = r2_b - r2_a

    print("\n=== RESULTS (per SMC channel) ===")
    print(f"{'ch':14s} {'R²_S-only':>10s} {'R²_S+u':>10s} {'ΔR²':>9s}  closed-loop-informative?")
    print("-" * 70)
    n_pass = 0
    for i, n in enumerate(smc_names):
        flag = "★ YES" if gain[i] > 0.05 else "  no"
        if gain[i] > 0.05: n_pass += 1
        print(f"{n:14s} {r2_a[i]:>+10.4f} {r2_b[i]:>+10.4f} {gain[i]:>+9.4f}  {flag}")
    verdict = "PASS" if n_pass >= 3 else ("MARGINAL" if n_pass == 2 else "FAIL")
    print(f"\n=> Closed loop load-bearing on {n_pass}/{len(smc_names)} channels → {verdict}")

    # Sanity: how predictable is u from S alone? (causality check)
    # If u is fully recoverable from S, the gain is a free-lunch identifiability artifact.
    # Predict u from S history.
    Yu = u_arr[LAG:-1]
    Xu = Xa
    Wu = fit_ridge(Xu[:split], Yu[:split, None])
    pred_u = (Xu[split:] @ Wu).ravel()
    u_r2 = 1 - ((pred_u - Yu[split:]) ** 2).mean() / (Yu[split:].var() + 1e-9)
    print(f"\nSanity: R²(u | S_history) = {u_r2:+.4f}")
    print(f"  (if ≈1.0, action is already encoded in substrate — closed-loop gain may be redundant)")

    out_path = OUT / f"closed_loop_verify_lag{LAG}_2026-06-10.json"
    json.dump({
        "duration_s": DURATION_S, "lag": LAG, "n_frames": len(t_arr),
        "busy_fraction": float(n_busy / DURATION_S),
        "smc_channels": smc_names,
        "r2_S_only": [float(x) for x in r2_a],
        "r2_S_plus_u": [float(x) for x in r2_b],
        "delta_r2": [float(x) for x in gain],
        "u_predictable_from_S_r2": float(u_r2),
        "n_pass": int(n_pass), "verdict": verdict,
    }, open(out_path, "w"), indent=2)
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
