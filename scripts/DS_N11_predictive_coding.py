#!/usr/bin/env python3
"""DS-N11: NS-RAM as Bayesian prior for predictive coding.

Streaming setup:
  - Input x_t (Lorenz / sine / noisy-step) drives evidence current Iii_i = w_in_i * x_t.
  - NS-RAM cell state evolves continuously: membrane Vm + body charge Q.
    Vm follows leaky integrator with avalanche current (impact ionization)
    when Vm > V_thresh' = V_thresh - alpha * Q.  Q traps slowly (k_cap),
    detraps slowly (k_em). Spike events fire when Vm crosses threshold.
  - Linear ridge readout from [Vm, Q, spike-rate, x_t] -> x_{t+1}.
  - Spikes count = energy (E_spike = 21 fJ from Pazos slide 2).

Baselines:
  - Kalman (linear-Gaussian optimal).
  - LSTM (same param budget order; trained on first 60% of stream).
  - Random forest predictor.

Vectorized N=1K cells with explicit Euler (dt=1us, T=10K steps).
Target: <60s end-to-end on Ryzen + AMD gfx1151 (CPU here, no GPU needed for 1K cells).
"""
import os, json, time, math
import numpy as np
from pathlib import Path

os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")

OUT = Path(__file__).resolve().parent.parent / "results" / "DS_N11_predictive_coding"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 1337
rng = np.random.default_rng(SEED)

# ───────────────────────────── signals ─────────────────────────────
def lorenz(T, dt=0.01, sigma=10, rho=28, beta=8/3, seed=0):
    r = np.random.default_rng(seed); s = np.zeros((T, 3))
    s[0] = r.normal(0, 1, 3)
    for t in range(T - 1):
        x, y, z = s[t]
        dx = sigma * (y - x); dy = x * (rho - z) - y; dz = x * y - beta * z
        s[t + 1] = s[t] + dt * np.array([dx, dy, dz])
    x = s[:, 0]
    return (x - x.mean()) / x.std()

def sine_wave(T, freqs=(0.013, 0.041, 0.007), seed=0):
    r = np.random.default_rng(seed)
    t = np.arange(T)
    y = sum(np.sin(2*np.pi*f*t + r.uniform(0, 2*np.pi)) for f in freqs) / len(freqs)
    return (y - y.mean()) / y.std()

def noisy_step(T, n_steps=20, sigma=0.3, seed=0):
    r = np.random.default_rng(seed)
    levels = r.uniform(-1, 1, n_steps)
    edges = np.sort(r.integers(1, T-1, n_steps - 1))
    y = np.zeros(T); cur = 0
    for i, lvl in enumerate(levels):
        end = edges[i] if i < len(edges) else T
        y[cur:end] = lvl; cur = end
    y += r.normal(0, sigma, T)
    return (y - y.mean()) / y.std()

# ──────────────────────── NS-RAM vectorized ────────────────────────
class NSRAMArray:
    """Vectorized NS-RAM cell bank with body-charge prior + avalanche spiking.

    State variables (length N each):
      Vm[i]  : membrane voltage (V)
      Q[i]   : trapped body charge (unitless, in [0,1])
    Inputs:
      x_t (scalar) maps to per-cell evidence Iii_i = w_in[i]*x_t.
    Update per dt:
      Vm += dt*(I_leak + I_evidence + I_aval) / C_mem
      Q  += dt*(k_cap*(1-Q)*spike_rate - k_em*Q)
      spike if Vm > V_thresh - alpha*Q  → emit, partial reset
    """
    def __init__(self, N=1024, seed=0):
        r = np.random.default_rng(seed)
        self.N = N
        # heterogeneous physics (Pazos slide 16/17/23)
        self.C_mem    = 102e-15 * (1 + 0.10 * r.normal(size=N))
        self.g_leak   = (self.C_mem / 1e-6) * (1 + 0.10 * r.normal(size=N))
        self.V_thresh = 1.364 * (1 + 0.05 * r.normal(size=N))
        self.V_rest   = 0.0
        self.V_reset_frac = 0.3
        self.alpha_Q  = 0.45  # threshold drops alpha*Q when charge trapped
        self.k_cap    = 5e3   # capture rate (Hz)
        self.k_em     = 370.0 # emission rate (Hz) — slow prior decay
        self.I_aval0  = 2.0e-7  # avalanche scale (post-threshold injection)
        self.E_spike  = 21e-15  # J / spike (Pazos slide 2)

        # input projection (heterogeneous gain)
        self.w_in = 1.5e-7 * r.normal(size=N)   # A per unit input
        self.b_in = 0.3e-7 * r.normal(size=N)   # bias current

        # state
        self.Vm = np.zeros(N)
        self.Q  = np.zeros(N)
        self.refrac = np.zeros(N)  # remaining refractory time
        self.t_refrac = 1.6e-6

        # running spike-rate estimate (low-pass)
        self.rate = np.zeros(N)
        self.tau_rate = 50e-6

    def step(self, x_t, dt):
        # evidence current
        I_evd = self.w_in * x_t + self.b_in
        I_leak = -self.g_leak * (self.Vm - self.V_rest)
        # avalanche kicks in only above effective threshold
        Vth_eff = self.V_thresh - self.alpha_Q * self.Q
        over = np.maximum(self.Vm - Vth_eff, 0.0)
        I_aval = self.I_aval0 * (np.exp(np.clip(over / 0.05, 0, 8)) - 1.0)
        # integrate Vm (skip if refractory)
        dV = (I_leak + I_evd + I_aval) / self.C_mem
        active = self.refrac <= 0.0
        self.Vm[active] += dt * dV[active]
        self.refrac = np.maximum(self.refrac - dt, 0.0)

        # spike detection
        spk = (self.Vm > Vth_eff) & active
        if spk.any():
            self.Vm[spk] *= self.V_reset_frac
            self.refrac[spk] = self.t_refrac
        # rate low-pass
        self.rate += dt * (-self.rate / self.tau_rate)
        self.rate[spk] += 1.0 / self.tau_rate  # delta injection

        # body charge dynamics (prior)
        dQ = self.k_cap * (1.0 - self.Q) * (self.rate * dt) - self.k_em * self.Q * dt
        self.Q += dQ
        np.clip(self.Q, 0.0, 1.0, out=self.Q)

        return spk, self.Vm.copy(), self.Q.copy(), self.rate.copy()

def run_nsram(signal, N=1024, dt=1e-6, sub_per_sample=4, seed=0):
    arr = NSRAMArray(N=N, seed=seed)
    T = len(signal)
    # feature: low-pass spike rate (reservoir-style) + body charge Q + input
    feats = np.zeros((T, 2 * N + 1), dtype=np.float32)
    total_spikes = 0
    for t in range(T):
        x = float(signal[t])
        spk_acc = 0
        for _ in range(sub_per_sample):
            spk, Vm, Q, rate = arr.step(x, dt)
            spk_acc += int(spk.sum())
        feats[t, :N]      = rate  # spike-rate (Hz, bounded)
        feats[t, N:2*N]   = Q     # body charge (in [0,1])
        feats[t, 2*N]     = x
        total_spikes += spk_acc
    return feats, total_spikes

def ridge_predict(feats, target, train_frac=0.6, lam=10.0):
    T = len(target)
    n_tr = int(T * train_frac)
    X_tr, X_te = feats[:n_tr-1], feats[n_tr:-1]
    y_tr, y_te = target[1:n_tr], target[n_tr+1:]
    # drop columns with near-zero train variance
    sd_full = X_tr.std(0)
    keep = sd_full > 1e-6
    X_tr = X_tr[:, keep]; X_te = X_te[:, keep]
    mu = X_tr.mean(0); sd = X_tr.std(0) + 1e-8
    Xtr = ((X_tr - mu) / sd).astype(np.float64)
    Xte = ((X_te - mu) / sd).astype(np.float64)
    # clip outliers in test (out-of-distribution feature excursions)
    np.clip(Xtr, -6, 6, out=Xtr); np.clip(Xte, -6, 6, out=Xte)
    Xtr = np.concatenate([Xtr, np.ones((Xtr.shape[0], 1))], axis=1)
    Xte = np.concatenate([Xte, np.ones((Xte.shape[0], 1))], axis=1)
    A = Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1])
    b = Xtr.T @ y_tr.astype(np.float64)
    w = np.linalg.solve(A, b)
    pred = Xte @ w
    rmse = float(np.sqrt(np.mean((pred - y_te) ** 2)))
    return rmse, pred, y_te

# ──────────────────────── baselines ─────────────────────────────────
def kalman_predict(signal, train_frac=0.6):
    """Constant-velocity Kalman 1-step-ahead predictor."""
    T = len(signal)
    n_tr = int(T * train_frac)
    # state [x, v]; transition F=[[1,1],[0,1]]; measurement H=[1,0]
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    Q = np.eye(2) * 1e-2
    R = np.array([[0.1]])
    x = np.array([signal[0], 0.0]); P = np.eye(2)
    preds = np.zeros(T - n_tr - 1)
    for t in range(T - 1):
        # predict
        x = F @ x; P = F @ P @ F.T + Q
        # measure
        y = signal[t + 1] - (H @ x)[0]
        S = (H @ P @ H.T + R)[0, 0]
        K = (P @ H.T).flatten() / S
        if t >= n_tr:
            preds[t - n_tr] = (H @ x)[0]
        x = x + K * y
        P = (np.eye(2) - np.outer(K, H[0])) @ P
    y_te = signal[n_tr + 1:]
    rmse = float(np.sqrt(np.mean((preds - y_te) ** 2)))
    return rmse

def lstm_predict(signal, train_frac=0.6, hidden=32, win=16, epochs=80, lr=5e-3):
    import torch, torch.nn as nn
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    T = len(signal); n_tr = int(T * train_frac)
    def make_xy(s):
        X = np.stack([s[i:i+win] for i in range(len(s)-win-1)], axis=0)
        Y = s[win+1:]
        return X[..., None], Y
    Xtr, Ytr = make_xy(signal[:n_tr])
    Xte, Yte = make_xy(signal[n_tr-win-1:])
    Xtr = torch.tensor(Xtr, dtype=torch.float32, device=dev)
    Ytr = torch.tensor(Ytr, dtype=torch.float32, device=dev)
    Xte = torch.tensor(Xte, dtype=torch.float32, device=dev)
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(1, hidden, batch_first=True)
            self.fc = nn.Linear(hidden, 1)
        def forward(self, x):
            h,_ = self.lstm(x); return self.fc(h[:, -1, 0:hidden]).squeeze(-1)
    net = Net().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    n_params = sum(p.numel() for p in net.parameters())
    for ep in range(epochs):
        opt.zero_grad()
        pred = net(Xtr)
        loss = ((pred - Ytr) ** 2).mean()
        loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        pred = net(Xte).cpu().numpy()
    y_te = Yte
    m = min(len(pred), len(y_te))
    rmse = float(np.sqrt(np.mean((pred[:m] - y_te[:m]) ** 2)))
    # Approx LSTM inference energy: ~ n_params MACs × 1 pJ/MAC × win timesteps
    E_per_inf_J = n_params * win * 1e-12
    return rmse, n_params, E_per_inf_J

def rf_predict(signal, train_frac=0.6, win=16, n_est=80):
    from sklearn.ensemble import RandomForestRegressor
    T = len(signal); n_tr = int(T * train_frac)
    X = np.stack([signal[i:i+win] for i in range(T-win-1)], axis=0)
    Y = signal[win+1:]
    split = n_tr - win - 1
    Xtr, Ytr = X[:split], Y[:split]; Xte, Yte = X[split:], Y[split:]
    rf = RandomForestRegressor(n_estimators=n_est, max_depth=8, n_jobs=-1, random_state=0)
    rf.fit(Xtr, Ytr); pred = rf.predict(Xte)
    rmse = float(np.sqrt(np.mean((pred - Yte) ** 2)))
    return rmse

# ──────────────────────── main ─────────────────────────────────────
def main():
    T = 10_000
    N = 1024
    signals = {
        "lorenz":     lorenz(T, dt=0.01, seed=1),
        "sine":       sine_wave(T, seed=2),
        "noisy_step": noisy_step(T, seed=3),
    }
    results = {}
    t0 = time.time()
    for name, sig in signals.items():
        print(f"\n=== {name} (T={T}) ===", flush=True)
        ts = time.time()
        feats, n_spikes = run_nsram(sig, N=N, dt=1e-6, sub_per_sample=4,
                                    seed=hash(name) & 0xFFFF)
        t_ns = time.time() - ts
        rmse_ns, _, _ = ridge_predict(feats, sig)
        E_total_J = n_spikes * 21e-15
        n_test_pred = int(T * 0.4) - 1
        E_per_pred_J_ns = E_total_J / max(1, n_test_pred)
        print(f"  NS-RAM   t={t_ns:.1f}s  RMSE={rmse_ns:.4f}  spikes={n_spikes}  E/pred={E_per_pred_J_ns*1e9:.3f} nJ")

        ts = time.time()
        rmse_k = kalman_predict(sig)
        t_k = time.time() - ts
        print(f"  Kalman   t={t_k:.2f}s  RMSE={rmse_k:.4f}")

        ts = time.time()
        rmse_l, n_params_l, E_per_pred_J_l = lstm_predict(sig)
        t_l = time.time() - ts
        print(f"  LSTM     t={t_l:.1f}s  RMSE={rmse_l:.4f}  params={n_params_l}  E/pred={E_per_pred_J_l*1e6:.3f} uJ")

        ts = time.time()
        rmse_r = rf_predict(sig)
        t_r = time.time() - ts
        print(f"  RF       t={t_r:.1f}s  RMSE={rmse_r:.4f}")

        results[name] = {
            "nsram":  {"rmse": rmse_ns, "spikes": int(n_spikes),
                        "E_per_pred_J": E_per_pred_J_ns, "wall_s": t_ns},
            "kalman": {"rmse": rmse_k, "wall_s": t_k},
            "lstm":   {"rmse": rmse_l, "n_params": int(n_params_l),
                       "E_per_pred_J": E_per_pred_J_l, "wall_s": t_l},
            "rf":     {"rmse": rmse_r, "wall_s": t_r},
        }
    total_wall = time.time() - t0
    results["_meta"] = {"T": T, "N": N, "seed": SEED, "wall_total_s": total_wall}
    out_json = OUT / "prediction_rmse.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {out_json}")

    # plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        names = list(signals.keys()); x = np.arange(len(names))
        w = 0.2
        for i, m in enumerate(["nsram", "kalman", "lstm", "rf"]):
            vals = [results[n][m]["rmse"] for n in names]
            axes[0].bar(x + (i - 1.5) * w, vals, w, label=m.upper())
        axes[0].set_xticks(x); axes[0].set_xticklabels(names)
        axes[0].set_ylabel("RMSE (next-step prediction)")
        axes[0].set_title("Predictive coding RMSE — lower is better")
        axes[0].legend(); axes[0].grid(alpha=0.3)
        # energy per prediction (log)
        e_ns = [results[n]["nsram"]["E_per_pred_J"] for n in names]
        e_lm = [results[n]["lstm"]["E_per_pred_J"]  for n in names]
        axes[1].bar(x - 0.2, e_ns, 0.4, label="NS-RAM (spikes×21 fJ)")
        axes[1].bar(x + 0.2, e_lm, 0.4, label="LSTM (params×win×1 pJ)")
        axes[1].set_yscale("log"); axes[1].set_ylabel("Energy / prediction (J)")
        axes[1].set_xticks(x); axes[1].set_xticklabels(names)
        axes[1].set_title("Spike-event energy efficiency")
        axes[1].legend(); axes[1].grid(alpha=0.3, which="both")
        plt.tight_layout()
        plt.savefig(OUT / "spike_event_efficiency.png", dpi=130)
        print(f"Saved {OUT/'spike_event_efficiency.png'}")
    except Exception as e:
        print("plot failed:", e)

    # summary.md
    md = ["# DS-N11 Predictive Coding — NS-RAM vs Kalman vs LSTM vs RF\n"]
    md.append(f"- N cells: {N}, T steps: {T}, seed: {SEED}, total wall: {total_wall:.1f}s\n")
    md.append("\n## Best RMSE per signal\n")
    md.append("| signal | NS-RAM | Kalman | LSTM | RF | winner |")
    md.append("|---|---|---|---|---|---|")
    nsram_wins_chaotic = False
    for n in signals.keys():
        r = results[n]
        rmses = {"NS-RAM": r["nsram"]["rmse"], "Kalman": r["kalman"]["rmse"],
                 "LSTM": r["lstm"]["rmse"], "RF": r["rf"]["rmse"]}
        win = min(rmses, key=rmses.get)
        md.append(f"| {n} | {rmses['NS-RAM']:.4f} | {rmses['Kalman']:.4f} | "
                  f"{rmses['LSTM']:.4f} | {rmses['RF']:.4f} | **{win}** |")
        if n == "lorenz" and (rmses["NS-RAM"] < rmses["Kalman"] or rmses["NS-RAM"] < rmses["LSTM"]):
            nsram_wins_chaotic = True
    md.append("\n## Energy per prediction\n")
    md.append("| signal | NS-RAM (J) | LSTM (J) | ratio LSTM/NS-RAM |")
    md.append("|---|---|---|---|")
    for n in signals.keys():
        en = results[n]["nsram"]["E_per_pred_J"]; el = results[n]["lstm"]["E_per_pred_J"]
        md.append(f"| {n} | {en:.3e} | {el:.3e} | {el/max(en,1e-30):.1f}× |")
    md.append("\n## Gates")
    md.append(f"- INFRA (wall < 60s): {'PASS' if total_wall < 60 else 'FAIL'}  (actual {total_wall:.1f}s)")
    md.append(f"- HYPOTHESIS (NS-RAM beats Kalman OR LSTM on Lorenz): "
              f"{'PASS' if nsram_wins_chaotic else 'FAIL'}")
    (OUT / "summary.md").write_text("\n".join(md) + "\n")
    print(f"Saved {OUT/'summary.md'}")
    print(f"\nTOTAL wall: {total_wall:.1f}s")

if __name__ == "__main__":
    main()
