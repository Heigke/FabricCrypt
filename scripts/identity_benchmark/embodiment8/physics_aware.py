"""Embodiment Phase 8 — Task D: Physics-aware structure.

The reservoir's input projection W_in has a *block structure* keyed by feature
class: thermal-electrical impedance pair features (PAIR_T_P / PAIR_T_F /
PAIR_P_F), spectral features (bp*/oneoverf), burst/Fano features, and
derivative features each get their own block of reservoir neurons.

For chassi-keyed structure: per-chassi block assignments + per-chassi block
weight scaling factors (some chassis emphasise impedance features more, e.g.
a thicker heatsink → more thermal coupling → larger impedance weights).

For random structure: blocks are still present but the assignment of
*which* feature columns go to which block is random.

Test:
  Per-host: A_phys (chassi-keyed physics-aware) vs A_baseline (regular
  chassi-keyed reservoir from abcd_rich) vs B_baseline (random).

  PASS gate: A_phys − A_baseline ≥ 2% AND A_phys − B_baseline ≥ 5% AND
             CI for A_phys − B_baseline excludes 0.

We re-use the *scalar pair features* from dynamic_features.py (computed on
the rich substrate) to also compute one statistical sanity check:
  - Do the cross-channel impedance scalars (PAIR_*_r0, PAIR_*_peak_r,
    PAIR_*_lag_peak_s, PAIR_*_lf_coh) actually differ between hosts beyond
    sampling variation?  Two-sample t test on the per-pair distributions.
"""
from __future__ import annotations
import argparse, json, hashlib, time
from pathlib import Path
import numpy as np

from abcd_rich import (
    HASHES, hash_to_seed, load_features, make_windows, nrmse,
    HIST_C1, HORIZON_C1, N_TRAIN_C1, N_TEST_C1, RES_DIM, N_SEEDS, OUT_DIR,
    bootstrap_diff_ci,
)


def classify_feature_name(name):
    """Returns one of: temp, power, freq, voltage, current, irq, cpu,
    mem, disk, spec, deriv, pair_TP, pair_TF, pair_PF, other."""
    n = name.lower()
    if n.startswith("pair_t_p"): return "pair_TP"
    if n.startswith("pair_t_f"): return "pair_TF"
    if n.startswith("pair_p_f"): return "pair_PF"
    if "absd" in n or "__d" in n: return "deriv"
    if "bp" in n or "oneoverf" in n or "spec_" in n: return "spec"
    if "fano" in n or "burst" in n: return "burst"
    if "thermal" in n or "temp" in n: return "temp"
    if "power" in n: return "power"
    if "cpufreq" in n or "freq" in n: return "freq"
    if "in0" in n or "in1" in n: return "voltage"
    if "curr" in n: return "current"
    if "cpu_util" in n: return "cpu"
    if "ctxt" in n or "intr" in n: return "irq"
    if "vmstat" in n or "mem_" in n: return "mem"
    if "disk_" in n: return "disk"
    return "other"


# Channel "block" assignment for the *time-series* features used in the
# reservoir (they have suffixes __mean, __std, __absd1, __lfhf — the
# underlying base name still tells us the physical type)
def classify_ts(name):
    base = name.rsplit("__", 1)[0]
    return classify_feature_name(base)


class PhysicsAwareReservoir:
    """W_in is block-structured: each feature class is mapped to its own
    block of reservoir neurons.  The block sizes (in # of neurons) and
    weight scales are functions of the chassi seed."""

    BLOCKS = ["temp", "power", "freq", "voltage", "current", "cpu",
              "irq", "mem", "disk", "deriv", "spec", "other"]

    def __init__(self, structure_seed, ts_feature_names, res_dim=RES_DIM,
                 randomize_assignment=False):
        rng = np.random.default_rng(structure_seed)
        din = len(ts_feature_names)
        # Block size weights (per chassi)
        weights = rng.dirichlet(np.ones(len(self.BLOCKS)))
        sizes = np.round(weights * res_dim).astype(int)
        # ensure at least 4 neurons per block + sum == res_dim
        sizes = np.maximum(sizes, 4)
        while sizes.sum() != res_dim:
            if sizes.sum() > res_dim:
                idx = int(rng.integers(0, len(sizes)))
                if sizes[idx] > 4: sizes[idx] -= 1
            else:
                idx = int(rng.integers(0, len(sizes)))
                sizes[idx] += 1
        # Per-block weight scaling factors (per chassi)
        block_scale = rng.uniform(0.5, 1.5, size=len(self.BLOCKS)).astype(np.float32)
        # Feature -> block index
        if randomize_assignment:
            feat_block = rng.integers(0, len(self.BLOCKS), size=din)
        else:
            feat_block = np.array([self.BLOCKS.index(classify_ts(n))
                                   if classify_ts(n) in self.BLOCKS else self.BLOCKS.index("other")
                                   for n in ts_feature_names])
        # Build W_in: each feature only feeds its assigned block
        W_in = np.zeros((din, res_dim), dtype=np.float32)
        block_starts = np.concatenate([[0], np.cumsum(sizes)])
        for i in range(din):
            bidx = int(feat_block[i])
            s, e = block_starts[bidx], block_starts[bidx+1]
            W_in[i, s:e] = rng.standard_normal(e - s).astype(np.float32) / np.sqrt(max(1, (feat_block == bidx).sum())) * block_scale[bidx]
        self.W_in = W_in
        self.W_rec = (rng.standard_normal((res_dim, res_dim)) / np.sqrt(res_dim) * 0.9).astype(np.float32)
        self.bias = (rng.standard_normal(res_dim) * 0.1).astype(np.float32)
        self.res_dim = res_dim; self.din = din
        self.W_out = None; self.b_out = None

    def features(self, X):
        n, hist, _ = X.shape
        h = np.zeros((n, self.res_dim), dtype=np.float32)
        for t in range(hist):
            h = np.tanh(X[:, t, :] @ self.W_in + h @ self.W_rec + self.bias)
        return h

    def fit(self, X, Y, lam=1e-2):
        H = self.features(X)
        Yf = Y.reshape(len(X), -1)
        A = H.T @ H + lam * np.eye(self.res_dim, dtype=np.float32)
        B = H.T @ Yf
        W = np.linalg.solve(A, B)
        self.W_out = W
        self.b_out = Yf.mean(axis=0) - H.mean(axis=0) @ W

    def predict(self, X, horizon, dout):
        H = self.features(X)
        Yf = H @ self.W_out + self.b_out
        return Yf.reshape(-1, horizon, dout)


def run_phys_cell(struct, ts_feature_names, train_X, eval_X, n_seeds=N_SEEDS,
                  randomize_assignment=False):
    D = min(train_X.shape[1], eval_X.shape[1])
    train_X = train_X[:, :D]; eval_X = eval_X[:, :D]
    ts_names = ts_feature_names[:D]
    nrmses = []
    for seed in range(n_seeds):
        if struct == "random":
            ss = seed * 1009 + 7
        else:
            ss = hash_to_seed(HASHES[struct], salt=seed)
        Xtr, Ytr = make_windows(train_X, N_TRAIN_C1, HIST_C1, HORIZON_C1, seed=seed)
        Xte, Yte = make_windows(eval_X, N_TEST_C1, HIST_C1, HORIZON_C1, seed=seed + 9001)
        m = PhysicsAwareReservoir(ss, list(ts_names),
                                  randomize_assignment=randomize_assignment)
        m.fit(Xtr, Ytr)
        Yp = m.predict(Xte, HORIZON_C1, D)
        nrmses.append(nrmse(Yte, Yp))
    return nrmses


# ---------------------------------------------------------------------------
def pair_scalar_diff(host_a="ikaros", host_b="daedalus"):
    """Test whether PAIR_*_r0 / peak_r / lf_coh distributions differ
    between hosts. Two-sample t (Welch) on the cross-pair scalar values."""
    import scipy.stats as st  # type: ignore
    rep = {}
    for h in (host_a, host_b):
        z = np.load(OUT_DIR / f"{h}_features.npz", allow_pickle=True)
        names = list(z["scalar_names"]); vals = z["scalar_values"]
        groups = {"r0": [], "peak_r": [], "lag_peak_s": [], "lf_coh": []}
        for n, v in zip(names, vals):
            if not n.startswith("PAIR_"): continue
            for k in groups:
                if n.endswith("__"+k):
                    groups[k].append(float(v))
        rep[h] = {k: np.array(v) for k, v in groups.items()}
    out = {}
    for k in ("r0", "peak_r", "lag_peak_s", "lf_coh"):
        a = rep[host_a][k]; b = rep[host_b][k]
        if len(a) == 0 or len(b) == 0:
            out[k] = {"n_a": len(a), "n_b": len(b), "mean_a": 0, "mean_b": 0, "t": 0, "p": 1.0}
            continue
        t, p = st.ttest_ind(a, b, equal_var=False)
        out[k] = {"n_a": len(a), "n_b": len(b),
                  "mean_a": float(a.mean()), "mean_b": float(b.mean()),
                  "std_a": float(a.std()), "std_b": float(b.std()),
                  "t": float(t), "p": float(p)}
    return out


# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    # Load both hosts' features ts + names
    Xi, names_i = load_features("ikaros")
    Xd, names_d = load_features("daedalus")
    print(f"[phys] ikaros features={Xi.shape}, daedalus features={Xd.shape}")

    by_eval = {}
    for eh in ("ikaros", "daedalus"):
        oh = "daedalus" if eh == "ikaros" else "ikaros"
        Xe = Xi if eh == "ikaros" else Xd
        Xo = Xi if oh == "ikaros" else Xd
        names_e = names_i if eh == "ikaros" else names_d
        # A_phys: chassi-keyed physics-aware, own data
        a_phys = run_phys_cell(eh, names_e, train_X=Xe, eval_X=Xe,
                               randomize_assignment=False)
        # A_baseline: chassi-keyed regular (random assignment), own data
        a_base = run_phys_cell(eh, names_e, train_X=Xe, eval_X=Xe,
                               randomize_assignment=True)
        # B_baseline: random struct, own data, random assignment
        b_base = run_phys_cell("random", names_e, train_X=Xe, eval_X=Xe,
                               randomize_assignment=True)
        am, bm, pm = np.mean(a_phys), np.mean(b_base), np.mean(a_base)
        ab_d, ab_lo, ab_hi = bootstrap_diff_ci(a_phys, b_base)
        ap_d, ap_lo, ap_hi = bootstrap_diff_ci(a_phys, a_base)
        # NRMSE: lower is better
        # gain_vs_baseline: percentage improvement
        gain_vs_base   = 100.0 * (pm - am) / max(pm, 1e-9)
        gain_vs_random = 100.0 * (bm - am) / max(bm, 1e-9)
        gates = {
            "phys_beats_baseline": {"pct": gain_vs_base, "PASS": gain_vs_base >= 2.0 and ap_lo > 0,
                                    "ci": [ap_lo, ap_hi]},
            "phys_beats_random":   {"pct": gain_vs_random, "PASS": gain_vs_random >= 5.0 and ab_lo > 0,
                                    "ci": [ab_lo, ab_hi]},
        }
        by_eval[eh] = {
            "A_phys_mean": float(am), "A_baseline_mean": float(pm), "B_random_mean": float(bm),
            "gain_vs_baseline_pct": gain_vs_base,
            "gain_vs_random_pct": gain_vs_random,
            "A_phys_per_seed": a_phys, "A_baseline_per_seed": a_base, "B_random_per_seed": b_base,
            "gates": gates,
        }
        print(f"[phys] eh={eh} A_phys={am:.4f} A_base={pm:.4f} B_rand={bm:.4f}  gain_vs_base={gain_vs_base:+.2f}%  gain_vs_rand={gain_vs_random:+.2f}%")

    # Cross-pair sanity check (statistical signal in the substrate)
    try:
        pair_stats = pair_scalar_diff()
    except Exception as e:
        pair_stats = {"error": str(e)}

    out = {
        "by_eval_host": by_eval,
        "pair_scalar_stats": pair_stats,
        "runtime_s": time.time() - t0,
    }
    (OUT_DIR / "physics_aware.json").write_text(json.dumps(out, indent=2, default=float))
    print("saved physics_aware.json   runtime:", out["runtime_s"])


if __name__ == "__main__":
    main()
