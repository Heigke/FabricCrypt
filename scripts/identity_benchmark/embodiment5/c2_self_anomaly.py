"""C2 — Self-anomaly detection.

Train a tiny autoencoder on THIS chassis's NORMAL substrate behaviour.
Generate synthetic anomalies (power spike, thermal step, latency burst) on
the test data only. Detect anomalies by reconstruction error.

Compare:
  - ikaros-trained AE on ikaros-anomalies
  - daedalus-trained AE on ikaros-anomalies (transplanted)
  - Generic AE: random projection baseline ("untrained" reconstructor)

Pre-reg WIN gate: ikaros AE AUROC > daedalus-trained AE AUROC on
ikaros-anomalies by ≥10 pp.

Reuses C1's collected timeseries when available; otherwise collects own.
"""
from __future__ import annotations
import sys, json, time, socket
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _telemetry import CHANNELS

HOST = socket.gethostname()
ROOT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
C1_OUT = ROOT / f"results/IDENTITY_BENCHMARK_2026-05-30/embodiment5/c1_{HOST}"
OUT = ROOT / f"results/IDENTITY_BENCHMARK_2026-05-30/embodiment5/c2_{HOST}"
OUT.mkdir(parents=True, exist_ok=True)

USE_CH = ["apu_temp_c", "gpu_temp_c", "gpu_power_w", "gpu_freq_mhz", "kern_lat_us"]
D = len(USE_CH)
WIN = 50
HID = 16
N_TRAIN = 400
N_TEST_NORMAL = 100
N_TEST_ANOM = 100
SEEDS = [0, 1, 2, 3, 4]


# ---------- Anomaly injection (in scaled units) ---------------------------
def inject_power_spike(x, rng, ch=2, mag=4.0):
    x = x.copy()
    t = rng.integers(WIN // 4, 3 * WIN // 4)
    dur = rng.integers(3, 8)
    x[t:t+dur, ch] += mag
    return x


def inject_thermal_step(x, rng, ch=0, mag=3.0):
    x = x.copy()
    t = rng.integers(WIN // 4, WIN // 2)
    x[t:, ch] += mag
    return x


def inject_latency_burst(x, rng, ch=4, mag=5.0):
    x = x.copy()
    t = rng.integers(WIN // 4, 3 * WIN // 4)
    dur = rng.integers(2, 5)
    x[t:t+dur, ch] += mag
    return x


def inject_freq_drop(x, rng, ch=3, mag=-3.0):
    x = x.copy()
    t = rng.integers(WIN // 4, 2 * WIN // 3)
    dur = rng.integers(4, 10)
    x[t:t+dur, ch] += mag
    return x


ANOM_FNS = [inject_power_spike, inject_thermal_step,
            inject_latency_burst, inject_freq_drop]


# ---------- Data slicing --------------------------------------------------
def slice_windows(data, n, seed=0, hist=WIN):
    rng = np.random.default_rng(seed)
    T = len(data)
    starts = rng.choice(T - hist - 1, n, replace=False)
    return np.stack([data[s:s+hist] for s in starts])


# ---------- AE ------------------------------------------------------------
class TinyAE:
    def __init__(self, din=WIN*D, hid=HID, seed=0):
        rng = np.random.default_rng(seed)
        self.W1 = (rng.standard_normal((din, hid)) * np.sqrt(1.0 / din)).astype(np.float32)
        self.b1 = np.zeros(hid, dtype=np.float32)
        self.W2 = (rng.standard_normal((hid, din)) * np.sqrt(1.0 / hid)).astype(np.float32)
        self.b2 = np.zeros(din, dtype=np.float32)

    def forward(self, x):
        h = np.tanh(x @ self.W1 + self.b1)
        y = h @ self.W2 + self.b2
        return y, h

    def fit(self, X, epochs=200, lr=0.01, batch=64, verbose=False):
        Xf = X.reshape(len(X), -1).astype(np.float32)
        n = len(Xf)
        for ep in range(epochs):
            idx = np.random.permutation(n)
            losses = []
            for i in range(0, n, batch):
                b = idx[i:i+batch]
                xb = Xf[b]
                y, h = self.forward(xb)
                err = y - xb
                gW2 = h.T @ err / len(b); gb2 = err.mean(0)
                dh = err @ self.W2.T * (1 - h * h)
                gW1 = xb.T @ dh / len(b); gb1 = dh.mean(0)
                self.W2 -= lr * gW2; self.b2 -= lr * gb2
                self.W1 -= lr * gW1; self.b1 -= lr * gb1
                losses.append(float((err * err).mean()))
            if verbose and ep % 50 == 0:
                print(f"  AE ep{ep} loss={np.mean(losses):.4f}")

    def recon_err(self, X):
        Xf = X.reshape(len(X), -1).astype(np.float32)
        y, _ = self.forward(Xf)
        return ((y - Xf) ** 2).mean(axis=1)


def auroc(scores_normal, scores_anom):
    """Compute AUROC where higher score = more anomalous.
    Uses Mann-Whitney U statistic formulation."""
    n_neg = len(scores_normal); n_pos = len(scores_anom)
    s = np.concatenate([scores_normal, scores_anom])
    y = np.concatenate([np.zeros(n_neg), np.ones(n_pos)])
    order = np.argsort(s, kind="mergesort")  # ascending
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    sum_ranks_pos = ranks[y == 1].sum()
    U = sum_ranks_pos - n_pos * (n_pos + 1) / 2.0
    return float(U / (n_pos * n_neg))


# ---------- Pipeline -----------------------------------------------------
def fit_scaler(X):
    flat = X.reshape(-1, D)
    return flat.mean(0).astype(np.float32), (flat.std(0) + 1e-6).astype(np.float32)


def scale(X, mu, sd): return ((X - mu) / sd).astype(np.float32)


def run_self_train(data: np.ndarray):
    out_rows = []
    weights = []
    for seed in SEEDS:
        Xtr = slice_windows(data, N_TRAIN, seed=seed * 11 + 1)
        Xte_n = slice_windows(data, N_TEST_NORMAL, seed=seed * 11 + 7)
        mu, sd = fit_scaler(Xtr)
        Xtr_s = scale(Xtr, mu, sd); Xte_n_s = scale(Xte_n, mu, sd)

        # Make anomaly test windows
        rng = np.random.default_rng(seed * 13 + 3)
        base = slice_windows(data, N_TEST_ANOM, seed=seed * 11 + 17)
        base_s = scale(base, mu, sd)
        anom = np.stack([ANOM_FNS[i % len(ANOM_FNS)](base_s[i], rng)
                          for i in range(N_TEST_ANOM)])

        ae = TinyAE(din=WIN*D, hid=HID, seed=seed)
        ae.fit(Xtr_s, epochs=200, lr=0.005, verbose=(seed == 0))

        s_n = ae.recon_err(Xte_n_s)
        s_a = ae.recon_err(anom)
        au = auroc(s_n, s_a)

        # Generic baseline: untrained random projection AE
        ae_g = TinyAE(din=WIN*D, hid=HID, seed=seed + 1000)
        s_n_g = ae_g.recon_err(Xte_n_s)
        s_a_g = ae_g.recon_err(anom)
        au_g = auroc(s_n_g, s_a_g)

        print(f"  seed={seed} AUROC self={au:.3f}  generic_untrained={au_g:.3f}")
        out_rows.append({"seed": int(seed), "auroc_self": au, "auroc_generic": au_g})
        weights.append({"W1": ae.W1.copy(), "b1": ae.b1.copy(),
                         "W2": ae.W2.copy(), "b2": ae.b2.copy(),
                         "mu": mu, "sd": sd})
    return out_rows, weights


def eval_other_on_self(data: np.ndarray, other_pkg: Path):
    pkg = np.load(other_pkg)
    rows = []
    for i, seed in enumerate(SEEDS):
        if i >= len(pkg["W1"]):
            break
        Xte_n = slice_windows(data, N_TEST_NORMAL, seed=seed * 11 + 7)
        rng = np.random.default_rng(seed * 13 + 3)
        base = slice_windows(data, N_TEST_ANOM, seed=seed * 11 + 17)
        mu_o, sd_o = pkg["mu"][i], pkg["sd"][i]
        Xte_n_s = scale(Xte_n, mu_o, sd_o)
        base_s = scale(base, mu_o, sd_o)
        anom = np.stack([ANOM_FNS[k % len(ANOM_FNS)](base_s[k], rng)
                          for k in range(N_TEST_ANOM)])

        ae = TinyAE(din=WIN*D, hid=HID, seed=seed)
        ae.W1 = pkg["W1"][i]; ae.b1 = pkg["b1"][i]
        ae.W2 = pkg["W2"][i]; ae.b2 = pkg["b2"][i]
        s_n = ae.recon_err(Xte_n_s); s_a = ae.recon_err(anom)
        au = auroc(s_n, s_a)
        rows.append({"seed": int(seed), "auroc_transplant": au})
        print(f"  transplant seed={seed} AUROC={au:.3f}")
    return rows


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    data_npy = C1_OUT / f"c1_{HOST}_data.npy"
    if not data_npy.exists():
        print(f"ERROR: need {data_npy} from C1 first (run c1 'collect')")
        sys.exit(2)
    data = np.load(data_npy)
    print(f"[C2] data shape {data.shape}")

    if cmd in ("train", "all"):
        rows, weights = run_self_train(data)
        pkg = OUT / f"c2_{HOST}_ae.npz"
        np.savez(pkg,
                 W1=np.stack([w["W1"] for w in weights]),
                 b1=np.stack([w["b1"] for w in weights]),
                 W2=np.stack([w["W2"] for w in weights]),
                 b2=np.stack([w["b2"] for w in weights]),
                 mu=np.stack([w["mu"] for w in weights]),
                 sd=np.stack([w["sd"] for w in weights]))
        summary = {
            "host": HOST, "channels": USE_CH, "win": WIN,
            "n_train": N_TRAIN, "n_test_n": N_TEST_NORMAL, "n_test_a": N_TEST_ANOM,
            "rows": rows,
            "auroc_self_med": float(np.median([r["auroc_self"] for r in rows])),
            "auroc_generic_med": float(np.median([r["auroc_generic"] for r in rows])),
        }
        (OUT / f"c2_{HOST}_self_summary.json").write_text(json.dumps(summary, indent=2))
        print(f"[C2] self AUROC med = {summary['auroc_self_med']:.3f}")
        print(f"[C2] generic untrained AUROC med = {summary['auroc_generic_med']:.3f}")

    if cmd == "eval":
        if len(sys.argv) < 3:
            print("usage: eval <other_host_ae.npz>"); sys.exit(2)
        rows = eval_other_on_self(data, Path(sys.argv[2]))
        out = {"host_evaluated_on": HOST, "rows": rows,
                "transplant_med": float(np.median([r["auroc_transplant"] for r in rows]))}
        (OUT / f"c2_{HOST}_transplant.json").write_text(json.dumps(out, indent=2))
        print(f"[C2] transplant AUROC med = {out['transplant_med']:.3f}")


if __name__ == "__main__":
    main()
