"""C1 — Self-substrate-prediction.

Train a small model on THIS chassis to predict its own next-step substrate
state (5 channels) given a 100-step history. Compare to:
  - Daedalus-trained model evaluated on ikaros data (cross-host transplant)
  - Generic mean-prediction baseline
  - Linear AR(1) baseline

The data we collect is the body's actual time-series under a mock workload
that the script drives itself. The TASK is to predict THIS body's response
to a controlled-but-not-identical stimulus pattern.

Pre-reg WIN gate: ikaros-trained NRMSE < daedalus-trained NRMSE by ≥30%
on ikaros-substrate-prediction (test windows held out).

We collect locally, save windows + train locally, then save a 'transplant'
package that can be evaluated by/with the other chassis's model.

Usage:
  python c1_self_prediction.py collect          # 12 min collection
  python c1_self_prediction.py train            # train local model
  python c1_self_prediction.py eval <other.npz> # evaluate other-host model on local test data
  python c1_self_prediction.py all              # collect+train+self-eval
"""
from __future__ import annotations
import sys, os, json, time, threading, socket
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _telemetry import (
    collect_window, sample_substrate, CHANNELS, apu_temp_c, wait_cool,
)

HOST = socket.gethostname()
ROOT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
OUT = ROOT / f"results/IDENTITY_BENCHMARK_2026-05-30/embodiment5/c1_{HOST}"
OUT.mkdir(parents=True, exist_ok=True)

# Drop voltage (always 0 on gfx1151) and freq (mostly constant)
USE_CH = ["apu_temp_c", "gpu_temp_c", "gpu_power_w", "gpu_freq_mhz", "kern_lat_us"]
CH_IDX = [CHANNELS.index(c) for c in USE_CH]
D = len(USE_CH)

HIST = 100         # history length per window
HORIZON = 10       # predict next 10 steps (was 100, reduced for tractability)
N_TRAIN = 600
N_TEST = 150
DT = 0.20          # 200ms cadence ~5Hz
ABORT_T = 73.0


# ---------- Workload driver -----------------------------------------------
class WorkloadDriver:
    """Background thread that runs varying numpy work to perturb substrate."""
    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.stop = False
        self.intensity = 0.5

    def _phase(self, size_a: int, rest_s: float):
        a = self.rng.standard_normal((size_a, size_a)).astype(np.float32)
        t0 = time.time()
        while time.time() - t0 < 1.5 and not self.stop:
            _ = a @ a
        time.sleep(rest_s)

    def run(self):
        # Cycle through varying sizes/rests to create rich substrate trajectory
        cycle = [(200, 0.05), (400, 0.10), (700, 0.0), (250, 0.30),
                 (550, 0.05), (350, 0.15), (800, 0.0), (200, 0.5)]
        i = 0
        while not self.stop:
            if apu_temp_c() > ABORT_T:
                time.sleep(2.0)
                continue
            s, r = cycle[i % len(cycle)]
            self._phase(s, r)
            i += 1


# ---------- Data collection -----------------------------------------------
def collect(total_samples: int, out_npy: Path):
    """Collect total_samples telemetry rows at DT cadence with workload."""
    print(f"[C1] collect on {HOST}: {total_samples} samples at {DT}s = "
          f"{total_samples * DT / 60:.1f} min")
    wait_cool(60.0)
    drv = WorkloadDriver(seed=42)
    th = threading.Thread(target=drv.run, daemon=True)
    th.start()
    time.sleep(2.0)  # warm-up

    # Collect in chunks of 200 with thermal check
    rows = []
    CHUNK = 200
    for start in range(0, total_samples, CHUNK):
        n = min(CHUNK, total_samples - start)
        w = collect_window(n, dt_s=DT)
        rows.append(w)
        t = apu_temp_c()
        print(f"  chunk {start}-{start+n} done, apu={t:.1f}C", flush=True)
        if t > ABORT_T:
            print(f"  HOT abort at {t:.1f}C — pausing")
            drv.stop = True
            wait_cool(60.0)
            drv = WorkloadDriver(seed=42 + start)
            th = threading.Thread(target=drv.run, daemon=True)
            th.start()
            time.sleep(2.0)
    drv.stop = True
    data = np.concatenate(rows, axis=0)
    data = data[:, CH_IDX]
    np.save(out_npy, data)
    print(f"[C1] saved {data.shape} to {out_npy}")
    return data


# ---------- Window construction -------------------------------------------
def make_windows(data: np.ndarray, n_train: int, n_test: int,
                 hist: int = HIST, horizon: int = HORIZON, seed: int = 0):
    """Slice (T, D) timeseries into (N, hist, D) X and (N, horizon, D) Y windows."""
    T = len(data)
    rng = np.random.default_rng(seed)
    max_start = T - hist - horizon
    if max_start < n_train + n_test + 50:
        raise RuntimeError(f"Need more data: T={T} insufficient for "
                            f"{n_train}+{n_test} windows of hist={hist}+hor={horizon}")
    # Use first 70% for train pool, last 30% for test pool — temporal hold-out
    split = int(max_start * 0.70)
    train_starts = rng.choice(split, n_train, replace=False)
    test_starts = rng.choice(np.arange(split, max_start), n_test, replace=False)

    def _gather(starts):
        X = np.stack([data[s:s+hist] for s in starts])
        Y = np.stack([data[s+hist:s+hist+horizon] for s in starts])
        return X, Y
    return _gather(train_starts), _gather(test_starts)


# ---------- Per-channel standardisation -----------------------------------
def fit_scaler(X):
    # X: (..., D); standardise per channel
    flat = X.reshape(-1, D)
    mu = flat.mean(0)
    sd = flat.std(0) + 1e-6
    return mu.astype(np.float32), sd.astype(np.float32)


def apply_scaler(X, mu, sd):
    return ((X - mu) / sd).astype(np.float32)


def inverse_scaler(Y, mu, sd):
    return Y * sd + mu


# ---------- Model: tiny GRU-like predictor in numpy -----------------------
class TinyARPred:
    """Linear autoregressive predictor with per-channel last-K features.
    Designed to capture chip-specific dynamics; small enough to train in numpy.
    """
    def __init__(self, hist=HIST, horizon=HORIZON, k=20, seed=0):
        self.k = k
        self.horizon = horizon
        rng = np.random.default_rng(seed)
        # weight: (k*D + 1, horizon*D)  flat regression
        self.W = (rng.standard_normal((k * D + 1, horizon * D)) * 0.01).astype(np.float32)

    def featurise(self, X):
        """X: (N, hist, D) -> (N, k*D + 1) using last-k flattened + bias."""
        last = X[:, -self.k:, :].reshape(len(X), -1)
        bias = np.ones((len(X), 1), dtype=np.float32)
        return np.concatenate([last, bias], axis=1)

    def predict(self, X):
        F = self.featurise(X)
        Y = F @ self.W
        return Y.reshape(len(X), self.horizon, D)

    def fit(self, X, Y, lr=0.01, epochs=200, batch=64, verbose=True):
        F = self.featurise(X)
        Yflat = Y.reshape(len(Y), -1)
        n = len(F)
        for ep in range(epochs):
            idx = np.random.permutation(n)
            losses = []
            for i in range(0, n, batch):
                b = idx[i:i+batch]
                pred = F[b] @ self.W
                err = pred - Yflat[b]
                g = (F[b].T @ err) / len(b) * 2.0
                self.W -= lr * g
                losses.append(float((err * err).mean()))
            if verbose and ep % 40 == 0:
                print(f"  ep{ep} loss={np.mean(losses):.4f}")
        return self


# ---------- Metrics --------------------------------------------------------
def nrmse(pred, true):
    """Normalised RMSE per channel then averaged. pred/true: (N, horizon, D) in original units."""
    se = (pred - true) ** 2
    rmse = np.sqrt(se.mean(axis=(0, 1)))  # (D,)
    rng = true.max(axis=(0, 1)) - true.min(axis=(0, 1)) + 1e-6
    return float(np.mean(rmse / rng)), (rmse / rng).tolist()


def mean_baseline(Ytr, Yte):
    mu = Ytr.mean(axis=0, keepdims=True)
    pred = np.broadcast_to(mu, Yte.shape).copy()
    return nrmse(pred, Yte)


def ar1_baseline(Xte, Yte):
    """Predict each future step as last observed value (persistence)."""
    last = Xte[:, -1:, :]
    pred = np.broadcast_to(last, Yte.shape).copy()
    return nrmse(pred, Yte)


# ---------- Pipeline -------------------------------------------------------
def train_and_save(data_npy: Path, seeds=(0, 1, 2, 3, 4)):
    data = np.load(data_npy)
    print(f"[C1] loaded {data.shape}")
    rows = []
    for seed in seeds:
        (Xtr, Ytr), (Xte, Yte) = make_windows(data, N_TRAIN, N_TEST, seed=seed)
        mu, sd = fit_scaler(Xtr)
        Xtr_s = apply_scaler(Xtr, mu, sd); Xte_s = apply_scaler(Xte, mu, sd)
        Ytr_s = apply_scaler(Ytr, mu, sd); Yte_s = apply_scaler(Yte, mu, sd)

        model = TinyARPred(seed=seed)
        model.fit(Xtr_s, Ytr_s, lr=0.01, epochs=200, verbose=(seed == 0))
        pred_s = model.predict(Xte_s)
        pred = inverse_scaler(pred_s, mu, sd)

        nr, per_ch = nrmse(pred, Yte)
        mb, mb_ch = mean_baseline(Ytr, Yte)
        ab, ab_ch = ar1_baseline(Xte, Yte)
        print(f"  seed={seed} NRMSE: model={nr:.4f}  mean={mb:.4f}  AR1={ab:.4f}")
        rows.append({"seed": int(seed), "model_nrmse": nr, "model_nrmse_per_ch": per_ch,
                     "mean_nrmse": mb, "ar1_nrmse": ab,
                     "W": model.W.copy(), "mu": mu, "sd": sd})

    # Save bundle
    out_pkg = OUT / f"c1_{HOST}_model.npz"
    np.savez(out_pkg,
             W=np.stack([r["W"] for r in rows]),
             mu=np.stack([r["mu"] for r in rows]),
             sd=np.stack([r["sd"] for r in rows]),
             seeds=np.array(list(seeds)))
    summary = {
        "host": HOST,
        "n_train": N_TRAIN, "n_test": N_TEST,
        "hist": HIST, "horizon": HORIZON,
        "channels": USE_CH,
        "per_seed_self": [{k: v for k, v in r.items() if k not in ("W", "mu", "sd")}
                          for r in rows],
        "self_model_nrmse_med": float(np.median([r["model_nrmse"] for r in rows])),
        "self_mean_nrmse_med": float(np.median([r["mean_nrmse"] for r in rows])),
        "self_ar1_nrmse_med": float(np.median([r["ar1_nrmse"] for r in rows])),
    }
    (OUT / f"c1_{HOST}_self_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[C1] saved {out_pkg}")
    print(f"[C1] self NRMSE median = {summary['self_model_nrmse_med']:.4f}")
    return summary


def eval_other_model_on_self(other_npz: Path, data_npy: Path,
                              seeds=(0, 1, 2, 3, 4)):
    """Apply other-host's trained weights to THIS host's test windows."""
    data = np.load(data_npy)
    pkg = np.load(other_npz)
    Ws, mus, sds, oseeds = pkg["W"], pkg["mu"], pkg["sd"], pkg["seeds"]
    rows = []
    for i, seed in enumerate(seeds):
        if i >= len(Ws):
            break
        (_, _), (Xte, Yte) = make_windows(data, N_TRAIN, N_TEST, seed=int(seed))
        mu, sd = mus[i], sds[i]
        Xte_s = apply_scaler(Xte, mu, sd)
        model = TinyARPred(seed=int(seed)); model.W = Ws[i]
        pred_s = model.predict(Xte_s)
        pred = inverse_scaler(pred_s, mu, sd)
        nr, per_ch = nrmse(pred, Yte)
        rows.append({"seed": int(seed), "transplant_nrmse": nr, "per_ch": per_ch})
        print(f"  transplant seed={seed} NRMSE={nr:.4f}")
    return rows


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    data_npy = OUT / f"c1_{HOST}_data.npy"

    if cmd in ("collect", "all"):
        total = N_TRAIN + N_TEST + HIST + HORIZON + 200
        # but we resample many windows, so we want enough timeline coverage.
        # Use ~3000 samples = 600 sec = 10 min
        collect(3000, data_npy)

    if cmd in ("train", "all"):
        train_and_save(data_npy)

    if cmd == "eval":
        if len(sys.argv) < 3:
            print("usage: eval <other_host_model.npz>")
            sys.exit(2)
        other_npz = Path(sys.argv[2])
        rows = eval_other_model_on_self(other_npz, data_npy)
        # save transplant summary
        out = {"host_evaluated_on": HOST, "other_model": str(other_npz),
                "rows": rows,
                "transplant_med": float(np.median([r["transplant_nrmse"] for r in rows]))}
        (OUT / f"c1_{HOST}_transplant.json").write_text(json.dumps(out, indent=2))
        print(f"[C1] transplant NRMSE median = {out['transplant_med']:.4f}")


if __name__ == "__main__":
    main()
