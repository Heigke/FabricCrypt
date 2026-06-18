#!/usr/bin/env python3
"""
z263 - N2b: NS-RAM 4D body-state TRANSIENT surrogate as input neuron.

Re-runs N2 but exercises the surrogate's TRANSIENT V_b dynamics
(dV_b/dt = (I_ii - I_leak)/C_b) instead of static IV (V_b=0 fixed).

Per unit i, per image:
    - Poisson-encode the 784 pixels into spike trains over T=100 steps.
    - At each step t:
          drive_t  = W_in_i @ poisson_spike[t]
          V_G1_t   = V_G1_bias_i + g_in * drive_t  (clipped to surrogate)
          I_d, I_ii, I_leak = surrogate(V_G1_t, V_G2_bias_i, 1.0, V_b)
          V_b      = V_b + dt * (I_ii - I_leak) / C_b   (clipped to [0, 0.7])
          spike_rate_accum_i += abs(I_d)
    - unit_feature_i = log10(spike_rate_accum_i / T)
Features go through same ridge readout as N1b Stage A.

LOCKED parameters (no tuning):
    C_b = 5e-15 F
    dt  = 1e-7  s
    T   = 100
    V_G1_bias ~ U[0.2, 0.4], V_G2_bias ~ U[0.0, 0.3]
    g_in = 0.20

Baselines:
    N1b Poisson:     84.65%
    N2 static NS-RAM: 79.27%

Verdicts:
    conservative_vs_poisson : mean >= 82.65%
    ambitious_vs_poisson    : mean > 84.65% and CI non-overlap
    intermediate_vs_static  : mean > 79.27% and CI non-overlap

Subsampling: budget-constrained (T=100 × surrogate queries per step
per unit per image >> static N2). We subsample MNIST and log it.
"""
from __future__ import annotations

import os
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")

import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import torch  # noqa: F401 (kept for device reporting + thermal API)

ROOT = Path(__file__).resolve().parent.parent

# Reuse N1b's loader / thermal / ridge / bootstrap
spec_z261 = importlib.util.spec_from_file_location(
    "z261_n1b_lif_mnist", ROOT / "scripts/z261_n1b_lif_mnist.py"
)
z261 = importlib.util.module_from_spec(spec_z261)
spec_z261.loader.exec_module(z261)

# ------------------------------------------------------------------
# Config - LOCKED. No tuning.
# ------------------------------------------------------------------
N_NEURONS = 128
N_SEEDS = 5
SEEDS = [0, 1, 2, 3, 4]
BOOTSTRAP_N = 1000

VG1_BIAS_LO, VG1_BIAS_HI = 0.20, 0.40
VG2_BIAS_LO, VG2_BIAS_HI = 0.00, 0.30
G_IN = 0.20

C_B = 5e-15          # 5 fF body capacitance
DT_TRANS = 1e-7      # 0.1 us per Poisson step
T_STEPS = 100        # Poisson encoding window

VD_FIXED = 1.0
VB_INIT  = 0.0
VB_AXIS_MIN, VB_AXIS_MAX = 0.0, 0.7  # from surrogate inspection

# Subsample for time-budget. Locked. Report as such.
N_TRAIN_SUB = 10000
N_TEST_SUB  = 2000

THERMAL_CHECK_EVERY = 1  # batches (APU climbs fast on numpy)
BATCH_IMAGES = 128       # batch size per featurize chunk

SURROGATE_PATH = ROOT / "results/z220_4d_dense/surrogate_4d_dense.npz"
RESULTS_DIR = ROOT / "results/z263_n2b_nsram_transient_snn"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BASELINE_POISSON = 0.8465
BASELINE_STATIC  = 0.7927
PASS_CONSERVATIVE = 0.8265
# N1b approx CI = mean +/- 1.96 * std / sqrt(5) using std=0.0072
POISSON_CI = (0.8378, 0.8543)
# N2 static CI from its summary
STATIC_CI  = (0.7851, 0.7998)


# ------------------------------------------------------------------
# Vectorized 4D surrogate yielding (log_Id, Iii, Ileak) simultaneously
# ------------------------------------------------------------------
class VecSurrogate4D:
    """Quadrilinear interp over (VG1, VG2, Vd, Vb).

    Returns log10|Id|, Iii, Ileak.  Accepts ndarray inputs of any
    common shape.  Tracks OOD clipping rate for any axis.
    """
    def __init__(self, npz_path):
        d = np.load(npz_path)
        self.Id_log = np.log10(np.maximum(np.abs(d["Id"]), 1e-15)).astype(np.float32)
        self.Iii    = d["Iii"].astype(np.float32)
        self.Ileak  = d["Ileak"].astype(np.float32)
        self.vg1 = d["vg1_axis"].astype(np.float64)
        self.vg2 = d["vg2_axis"].astype(np.float64)
        self.vd  = d["vd_axis"].astype(np.float64)
        self.vb  = d["vb_axis"].astype(np.float64)
        self.n_query = 0
        self.n_clipped = 0
        self.n_vb_railed = 0

    @staticmethod
    def _idx(x, axis):
        x = np.asarray(x, dtype=np.float64)
        x_clip = np.clip(x, axis[0], axis[-1])
        m = (x != x_clip)
        i = np.clip(np.searchsorted(axis, x_clip), 1, len(axis) - 1) - 1
        f = (x_clip - axis[i]) / np.maximum(axis[i+1] - axis[i], 1e-30)
        return i, f.astype(np.float32), m

    def eval_all(self, VG1, VG2, Vd, Vb):
        """Returns (log_Id, Iii, Ileak), each same shape as inputs."""
        i, fi, m1 = self._idx(VG1, self.vg1)
        j, fj, m2 = self._idx(VG2, self.vg2)
        k, fk, m3 = self._idx(Vd,  self.vd)
        l, fl, m4 = self._idx(Vb,  self.vb)
        any_clip = (m1 | m2 | m3 | m4)
        self.n_query += i.size
        self.n_clipped += int(any_clip.sum())

        out_logId = np.zeros_like(fi, dtype=np.float32)
        out_Iii   = np.zeros_like(fi, dtype=np.float32)
        out_Ileak = np.zeros_like(fi, dtype=np.float32)
        for di in (0, 1):
            wi = fi if di else (1.0 - fi)
            for dj in (0, 1):
                wj = fj if dj else (1.0 - fj)
                for dk in (0, 1):
                    wk = fk if dk else (1.0 - fk)
                    for dl in (0, 1):
                        wl = fl if dl else (1.0 - fl)
                        w = wi * wj * wk * wl
                        out_logId += w * self.Id_log[i+di, j+dj, k+dk, l+dl]
                        out_Iii   += w * self.Iii[  i+di, j+dj, k+dk, l+dl]
                        out_Ileak += w * self.Ileak[i+di, j+dj, k+dk, l+dl]
        return out_logId, out_Iii, out_Ileak

    @property
    def clip_rate(self):
        if self.n_query == 0:
            return 0.0
        return self.n_clipped / self.n_query


# ------------------------------------------------------------------
# Per-seed parameter sampling (same convention as N2)
# ------------------------------------------------------------------
def build_nsram_params(N, n_in, seed):
    rng = np.random.RandomState(seed)
    VG1_bias = rng.uniform(VG1_BIAS_LO, VG1_BIAS_HI, size=N).astype(np.float32)
    VG2_bias = rng.uniform(VG2_BIAS_LO, VG2_BIAS_HI, size=N).astype(np.float32)
    W_in = (rng.randn(N, n_in) / np.sqrt(n_in)).astype(np.float32)
    return VG1_bias, VG2_bias, W_in


# ------------------------------------------------------------------
# Transient featurizer
# ------------------------------------------------------------------
def featurize_transient(X_np, VG1_bias, VG2_bias, W_in, surr,
                        seed_poisson, batch=BATCH_IMAGES):
    """
    X_np: (B_total, n_in) in [0,1]
    Returns:
        feats:        (B_total, N)  log10(mean |I_d| over T steps)
        vb_rail_frac: fraction of (image, unit, step) tuples where V_b
                      was clipped to the [0, 0.7] rail
    """
    B_total, n_in = X_np.shape
    N = VG1_bias.shape[0]
    rng = np.random.RandomState(seed_poisson)

    feats = np.zeros((B_total, N), dtype=np.float32)
    vb_railed_total = 0
    vb_total = 0

    VG2_row = VG2_bias[None, :]               # (1, N)
    VG1_b   = VG1_bias[None, :]               # (1, N)
    nb = 0
    for s in range(0, B_total, batch):
        # Per-batch thermal check
        z261.thermal_guard()
        e = min(s + batch, B_total)
        b = e - s
        Xc = X_np[s:e]                                  # (b, n_in)
        # State
        V_b = np.full((b, N), VB_INIT, dtype=np.float32)
        Id_accum = np.zeros((b, N), dtype=np.float64)

        # Poisson spike trains for this chunk over T steps
        # We avoid a (b, T, n_in) tensor (large). Instead draw per-step.
        Win_T = W_in.T                                  # (n_in, N)
        for t in range(T_STEPS):
            u = rng.rand(b, n_in).astype(np.float32)
            spk = (u < Xc).astype(np.float32)           # (b, n_in)
            drive = spk @ Win_T                         # (b, N)
            VG1 = VG1_b + G_IN * drive                  # (b, N)
            VG2 = np.broadcast_to(VG2_row, VG1.shape)
            Vd  = np.full_like(VG1, VD_FIXED)
            log_Id, Iii, Ileak = surr.eval_all(VG1, VG2, Vd, V_b)
            # |Id| from log10
            Id_abs = np.power(10.0, log_Id).astype(np.float64)
            Id_accum += Id_abs
            # Body-state Euler step
            V_b_new = V_b + DT_TRANS * (Iii - Ileak) / C_B
            V_b_clipped = np.clip(V_b_new, VB_AXIS_MIN, VB_AXIS_MAX)
            vb_railed_total += int((V_b_new != V_b_clipped).sum())
            vb_total += V_b_new.size
            V_b = V_b_clipped.astype(np.float32)

        mean_Id = Id_accum / float(T_STEPS)
        mean_Id = np.maximum(mean_Id, 1e-15)
        feats[s:e] = np.log10(mean_Id).astype(np.float32)
        nb += 1
        # Brief breathing room to keep APU off the rail
        t_now = z261.apu_temp_c()
        if t_now > 70.0:
            time.sleep(0.5)
        if t_now > 80.0:
            time.sleep(2.0)
    rail_frac = vb_railed_total / max(vb_total, 1)
    return feats, rail_frac


# ------------------------------------------------------------------
# Single-seed run
# ------------------------------------------------------------------
def precool(target_c=60.0, timeout_s=300):
    t0 = time.time()
    t = z261.apu_temp_c()
    if t < target_c:
        return t
    print(f"  [precool] APU={t:.1f}C waiting for <{target_c}C", flush=True)
    while z261.apu_temp_c() >= target_c and (time.time() - t0) < timeout_s:
        time.sleep(5)
    t = z261.apu_temp_c()
    print(f"  [precool] resumed at APU={t:.1f}C", flush=True)
    return t


def run_seed(seed, X_train, y_train, X_test, y_test):
    precool(60.0)
    t0 = time.time()
    print(f"[seed {seed}] start APU={z261.apu_temp_c():.1f}C", flush=True)
    n_in = X_train.shape[1]
    VG1_bias, VG2_bias, W_in = build_nsram_params(N_NEURONS, n_in, seed)

    # Fresh surrogate counters per seed
    surr = VecSurrogate4D(SURROGATE_PATH)

    # Smoke check on VG1 drive
    drive_sm = X_train[:500] @ W_in.T
    VG1_sm = VG1_bias[None, :] + G_IN * drive_sm
    print(f"[seed {seed}] VG1 drive range: [{VG1_sm.min():.3f}, "
          f"{VG1_sm.max():.3f}] (domain [0.10, 0.70])", flush=True)

    F_tr, rail_tr = featurize_transient(
        X_train, VG1_bias, VG2_bias, W_in, surr,
        seed_poisson=seed * 1000 + 7,
    )
    F_te, rail_te = featurize_transient(
        X_test, VG1_bias, VG2_bias, W_in, surr,
        seed_poisson=seed * 1000 + 99,
    )
    n_classes = int(max(y_train.max(), y_test.max())) + 1
    # z-score features per-unit using train stats (helps ridge)
    mu = F_tr.mean(axis=0, keepdims=True)
    sd = F_tr.std(axis=0, keepdims=True) + 1e-6
    F_tr_z = ((F_tr - mu) / sd).astype(np.float32)
    F_te_z = ((F_te - mu) / sd).astype(np.float32)

    Wro = z261.fit_ridge_readout(F_tr_z, y_train, n_classes)
    train_acc = float((z261.apply_readout(F_tr_z, Wro) == y_train).mean())
    test_acc  = float((z261.apply_readout(F_te_z, Wro) == y_test).mean())
    dt = time.time() - t0
    print(f"[seed {seed}] NSRAM-T train={train_acc:.4f}  test={test_acc:.4f}  "
          f"({dt:.1f}s, APU={z261.apu_temp_c():.1f}C, "
          f"clip={surr.clip_rate:.4f}, vb_rail_tr={rail_tr:.3f}, "
          f"vb_rail_te={rail_te:.3f})", flush=True)
    return {
        "seed": seed,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "wall_s": dt,
        "clip_rate": float(surr.clip_rate),
        "vb_rail_frac_train": float(rail_tr),
        "vb_rail_frac_test":  float(rail_te),
    }


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    print(f"torch {torch.__version__}  cuda_avail={torch.cuda.is_available()}",
          flush=True)
    print("[loader] loading MNIST 28x28...", flush=True)
    X_train, y_train, X_test, y_test, dataset = z261.load_mnist()
    print(f"dataset: {dataset}  train={X_train.shape}  test={X_test.shape}",
          flush=True)

    # Subsample (deterministic) for time budget
    if N_TRAIN_SUB < X_train.shape[0] or N_TEST_SUB < X_test.shape[0]:
        rng_sub = np.random.RandomState(424242)
        tr_idx = rng_sub.choice(X_train.shape[0], size=N_TRAIN_SUB, replace=False)
        te_idx = rng_sub.choice(X_test.shape[0],  size=N_TEST_SUB,  replace=False)
        X_train_use = X_train[tr_idx]; y_train_use = y_train[tr_idx]
        X_test_use  = X_test[te_idx];  y_test_use  = y_test[te_idx]
        sub_info = {
            "subsampled": True,
            "n_train_used": N_TRAIN_SUB,
            "n_test_used":  N_TEST_SUB,
            "subsample_seed": 424242,
        }
    else:
        X_train_use, y_train_use = X_train, y_train
        X_test_use,  y_test_use  = X_test, y_test
        sub_info = {"subsampled": False,
                    "n_train_used": int(X_train.shape[0]),
                    "n_test_used":  int(X_test.shape[0])}
    print(f"[subsample] {sub_info}", flush=True)

    t_global = time.time()
    thermal_peak = z261.apu_temp_c()
    per_seed = []
    for s in SEEDS:
        r = run_seed(s, X_train_use, y_train_use, X_test_use, y_test_use)
        per_seed.append(r)
        thermal_peak = max(thermal_peak, z261.apu_temp_c())
        z261.thermal_guard()

    accs = np.array([r["test_acc"] for r in per_seed])
    mean = float(accs.mean())
    std  = float(accs.std(ddof=1))
    ci   = z261.bootstrap_ci(accs)
    lo, hi = ci

    delta_vs_poisson_pp = (mean - BASELINE_POISSON) * 100.0
    delta_vs_static_pp  = (mean - BASELINE_STATIC)  * 100.0

    non_overlap_above_poisson = lo > POISSON_CI[1]
    non_overlap_above_static  = lo > STATIC_CI[1]

    verdict_conservative_vs_poisson = "PASS" if mean >= PASS_CONSERVATIVE else "FAIL"
    verdict_ambitious_vs_poisson    = (
        "PASS" if (mean > BASELINE_POISSON and non_overlap_above_poisson)
        else "FAIL"
    )
    verdict_intermediate_vs_static  = (
        "PASS" if (mean > BASELINE_STATIC and non_overlap_above_static)
        else "FAIL"
    )

    clip_rate_overall = float(np.mean([r["clip_rate"] for r in per_seed]))
    vb_rail_overall   = float(np.mean([0.5*(r["vb_rail_frac_train"]
                                            + r["vb_rail_frac_test"])
                                       for r in per_seed]))

    wall = time.time() - t_global

    summary = {
        "experiment": "z263_n2b_nsram_transient_snn",
        "date": "2026-05-11",
        "dataset": dataset,
        "n_train_full": int(X_train.shape[0]),
        "n_test_full":  int(X_test.shape[0]),
        "subsample": sub_info,
        "config": {
            "N_neurons": N_NEURONS,
            "VG1_bias_range": [VG1_BIAS_LO, VG1_BIAS_HI],
            "VG2_bias_range": [VG2_BIAS_LO, VG2_BIAS_HI],
            "g_in": G_IN,
            "C_b_F": C_B,
            "dt_s":  DT_TRANS,
            "T_steps": T_STEPS,
            "Vd_fixed": VD_FIXED,
            "Vb_init": VB_INIT,
            "Vb_axis_range": [VB_AXIS_MIN, VB_AXIS_MAX],
            "ridge_lambda": z261.RIDGE_LAMBDA,
            "feature": "log10(mean |I_d| over T steps), z-scored per-unit (train stats)",
            "surrogate": str(SURROGATE_PATH),
        },
        "n_seeds": N_SEEDS,
        "seeds": SEEDS,
        "per_seed": per_seed,
        "test_accuracy_mean": mean,
        "test_accuracy_std":  std,
        "test_accuracy_ci95": list(ci),
        "baseline_n1b_poisson": BASELINE_POISSON,
        "baseline_n2_static":   BASELINE_STATIC,
        "baseline_poisson_ci_approx": list(POISSON_CI),
        "baseline_static_ci_approx":  list(STATIC_CI),
        "delta_vs_poisson_pp": delta_vs_poisson_pp,
        "delta_vs_static_pp":  delta_vs_static_pp,
        "non_overlap_above_poisson": bool(non_overlap_above_poisson),
        "non_overlap_above_static":  bool(non_overlap_above_static),
        "ood_clip_rate_mean": clip_rate_overall,
        "vb_rail_frac_mean":  vb_rail_overall,
        "gates": {
            "conservative_vs_poisson": {"threshold": PASS_CONSERVATIVE,
                                        "rule": "mean >= 0.8265"},
            "ambitious_vs_poisson": {"threshold": BASELINE_POISSON,
                                     "rule": "mean > 0.8465 AND CI lo > Poisson CI hi"},
            "intermediate_vs_static": {"threshold": BASELINE_STATIC,
                                       "rule": "mean > 0.7927 AND CI lo > Static CI hi"},
        },
        "verdict_conservative_vs_poisson": verdict_conservative_vs_poisson,
        "verdict_ambitious_vs_poisson":    verdict_ambitious_vs_poisson,
        "verdict_intermediate_vs_static":  verdict_intermediate_vs_static,
        "wall_s_total": wall,
        "thermal_peak_c": thermal_peak,
    }
    out_path = RESULTS_DIR / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 72, flush=True)
    print("z263 N2b - NS-RAM TRANSIENT input neuron  SUMMARY", flush=True)
    print("=" * 72, flush=True)
    print(f"  subsample: {sub_info}", flush=True)
    print(f"  per-seed test_acc: " +
          ", ".join(f"{r['test_acc']:.4f}" for r in per_seed), flush=True)
    print(f"  mean = {mean:.4f} +- {std:.4f}  95% CI = [{lo:.4f}, {hi:.4f}]",
          flush=True)
    print(f"  vs N1b Poisson  ({BASELINE_POISSON:.4f}): "
          f"delta = {delta_vs_poisson_pp:+.2f} pp", flush=True)
    print(f"  vs N2 static    ({BASELINE_STATIC:.4f}): "
          f"delta = {delta_vs_static_pp:+.2f} pp", flush=True)
    print(f"  OOD clip rate (mean) = {clip_rate_overall:.4f}", flush=True)
    print(f"  Vb rail frac  (mean) = {vb_rail_overall:.4f}", flush=True)
    print(f"  CONSERVATIVE vs Poisson : {verdict_conservative_vs_poisson}", flush=True)
    print(f"  AMBITIOUS    vs Poisson : {verdict_ambitious_vs_poisson}", flush=True)
    print(f"  INTERMEDIATE vs Static  : {verdict_intermediate_vs_static}", flush=True)
    print(f"  wall = {wall:.1f}s   thermal peak = {thermal_peak:.1f}C", flush=True)
    print(f"  -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
