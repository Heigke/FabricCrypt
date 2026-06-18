"""z269 — single cell of the distributed 4D sweep.

Usage:
    python z269_sweep_cell.py --cell_id <id> \
        --C_b_fF <C_b> --V_G2_bias <V> --dt_s <dt> --g_in <g> \
        --seeds 0 1 2 3 \
        --subsample 5000 1000 \
        --out_dir results/sweep_v1

Each cell evaluates the NS-RAM 4D body-state surrogate as SNN input
neuron at the supplied operating point. Same architecture as
scripts/z268_n2c_nsram_alpha1e5.py but parameter-driven.

GPU-agnostic: works on ROCm (set HSA_OVERRIDE_GFX_VERSION=11.0.0
externally) and CUDA. Falls back to CPU.

Pre-registered gates (NOT applied here; reporting is per-cell raw):
  PASS if 4-seed mean accuracy >= 82.65% AND mean vb_rail <= 0.10 AND
                                  mean OOD clip <= 0.05
"""
from __future__ import annotations
import os, sys, time, json, argparse
from pathlib import Path

import numpy as np
import torch


def set_thermal_env():
    """Limit thread count to avoid CPU thermal cascade."""
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(k, "2")


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_surrogate(path):
    """Load 4D body-state surrogate (z220_4d_dense)."""
    z = np.load(path)
    return {
        "I_d":    z["Id"],
        "I_ii":   z["Iii"],
        "I_leak": z["Ileak"],
        "g_VG1":  z["vg1_axis"],
        "g_VG2":  z["vg2_axis"],
        "g_Vd":   z["vd_axis"],
        "g_Vb":   z["vb_axis"],
    }


def query_surrogate_vec(surr, VG1, VG2, Vd, Vb):
    """Trilinear-ish lookup (we use simple nearest for speed; surrogate is
    dense enough). Returns (I_d, I_ii, I_leak) as numpy arrays."""
    g_VG1 = surr["g_VG1"]; g_VG2 = surr["g_VG2"]
    g_Vd  = surr["g_Vd"];  g_Vb  = surr["g_Vb"]

    iVG1 = np.clip(np.searchsorted(g_VG1, np.asarray(VG1), side="right") - 1,
                   0, len(g_VG1) - 2)
    iVG2 = np.clip(np.searchsorted(g_VG2, np.asarray(VG2), side="right") - 1,
                   0, len(g_VG2) - 2)
    iVd  = np.clip(np.searchsorted(g_Vd,  np.asarray(Vd),  side="right") - 1,
                   0, len(g_Vd)  - 2)
    iVb  = np.clip(np.searchsorted(g_Vb,  np.asarray(Vb),  side="right") - 1,
                   0, len(g_Vb)  - 2)

    I_d    = surr["I_d"][iVG1, iVG2, iVd, iVb]
    I_ii   = surr["I_ii"][iVG1, iVG2, iVd, iVb]
    I_leak = surr["I_leak"][iVG1, iVG2, iVd, iVb]
    return I_d, I_ii, I_leak


def load_mnist_28x28(subsample_train, subsample_test, seed):
    """Reuse same loader pattern as scripts/z261/263/268."""
    try:
        from torchvision import datasets, transforms
        tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.0,), (1.0,)),
        ])
        train = datasets.MNIST("/tmp/mnist", train=True, download=True,
                               transform=tfm)
        test  = datasets.MNIST("/tmp/mnist", train=False, download=True,
                               transform=tfm)
        Xtr = train.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
        ytr = train.targets.numpy()
        Xte = test.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
        yte = test.targets.numpy()
    except Exception as e:
        # Fallback to sklearn fetch_openml
        from sklearn.datasets import fetch_openml
        m = fetch_openml("mnist_784", version=1, parser="auto", as_frame=False)
        X = (m.data.astype(np.float32) / 255.0)
        y = m.target.astype(int)
        Xtr, Xte = X[:60000], X[60000:]
        ytr, yte = y[:60000], y[60000:]
    rng = np.random.default_rng(seed)
    idx_tr = rng.choice(Xtr.shape[0], subsample_train, replace=False)
    idx_te = rng.choice(Xte.shape[0], subsample_test,  replace=False)
    return Xtr[idx_tr], ytr[idx_tr], Xte[idx_te], yte[idx_te]


def poisson_features_static(X, N, W_in, V_G1_bias, V_G2_bias, surr, g_in,
                            vd=1.0):
    """Static-IV reference features (N1b stage A style)."""
    drive = X @ W_in.T                       # (B, N)
    VG1 = V_G1_bias[None, :] + g_in * drive
    VG2 = np.broadcast_to(V_G2_bias[None, :], drive.shape)
    Vd  = np.full_like(VG1, vd)
    Vb  = np.zeros_like(VG1)
    I_d, _, _ = query_surrogate_vec(surr, VG1, VG2, Vd, Vb)
    feats = np.log10(np.abs(I_d) + 1e-18)
    feats = (feats - feats.mean(axis=0)) / (feats.std(axis=0) + 1e-9)
    return feats


def nsram_transient_features(X, N, W_in, V_G1_bias, V_G2_bias, surr,
                             g_in, C_b_F, dt_s, T_steps,
                             vd=1.0, rng_seed=0):
    """NS-RAM transient body-state SNN features.

    For each image x, Poisson-encode pixels into T_steps spike train,
    integrate V_b per unit via surrogate. Output mean |I_d| over T.
    """
    rng = np.random.default_rng(rng_seed + 99991)
    B = X.shape[0]
    spike_rate_accum = np.zeros((B, N), dtype=np.float64)
    Vb_rail_count = 0
    Vb_total = 0
    clip_count = 0
    clip_total = 0

    Vb_max = surr["g_Vb"].max()
    Vb_min = surr["g_Vb"].min()
    VG1_min, VG1_max = surr["g_VG1"].min(), surr["g_VG1"].max()

    # Poisson encoding rate: x ∈ [0,1] → spike prob per step
    p_max = 0.5  # max instantaneous probability at brightest pixel
    Vb = np.zeros((B, N), dtype=np.float64)

    for t in range(T_steps):
        spikes_t = (rng.random(X.shape) < (X * p_max)).astype(np.float32)
        drive_t = spikes_t @ W_in.T          # (B, N)
        VG1 = V_G1_bias[None, :] + g_in * drive_t
        VG1_clipped = np.clip(VG1, VG1_min, VG1_max)
        clip_count += int((VG1 != VG1_clipped).sum())
        clip_total += VG1.size
        VG1 = VG1_clipped
        VG2 = np.broadcast_to(V_G2_bias[None, :], VG1.shape)
        Vd_arr  = np.full_like(VG1, vd)
        Vb_arr  = np.clip(Vb, Vb_min, Vb_max)
        I_d, I_ii, I_leak = query_surrogate_vec(surr, VG1, VG2, Vd_arr, Vb_arr)
        Vb = Vb + dt_s * (I_ii - I_leak) / C_b_F
        Vb_pre = Vb.copy()
        Vb = np.clip(Vb, Vb_min, Vb_max)
        Vb_rail_count += int((Vb != Vb_pre).sum())
        Vb_total += Vb.size
        spike_rate_accum += np.abs(I_d) / T_steps

    feats = np.log10(spike_rate_accum + 1e-18)
    feats = (feats - feats.mean(axis=0)) / (feats.std(axis=0) + 1e-9)
    return feats, Vb_rail_count / max(Vb_total, 1), clip_count / max(clip_total, 1)


def ridge_lstsq(X, y, alpha=1e-3, n_classes=10):
    """Ridge readout with one-hot targets."""
    Y = np.eye(n_classes)[y]
    A = X.T @ X + alpha * np.eye(X.shape[1])
    B = X.T @ Y
    W = np.linalg.solve(A, B)
    return W


def predict(X, W):
    return np.argmax(X @ W, axis=1)


def run_one_seed(args, seed):
    surr = load_surrogate(args.surrogate)
    Xtr, ytr, Xte, yte = load_mnist_28x28(args.subsample[0], args.subsample[1],
                                          seed)
    rng = np.random.default_rng(seed + 31337)
    N = args.n_units
    W_in = rng.normal(0.0, 1.0, (N, 784)).astype(np.float32)
    W_in /= np.linalg.norm(W_in, axis=1, keepdims=True) + 1e-9
    V_G1_bias = rng.uniform(0.20, 0.40, N).astype(np.float64)
    V_G2_bias = np.full(N, args.V_G2_bias, dtype=np.float64)

    t0 = time.time()
    T_steps = max(int(round(1e-5 / args.dt_s)), 10)
    feats_tr, rail_tr, clip_tr = nsram_transient_features(
        Xtr, N, W_in, V_G1_bias, V_G2_bias, surr,
        args.g_in, args.C_b_F, args.dt_s, T_steps, rng_seed=seed)
    feats_te, rail_te, clip_te = nsram_transient_features(
        Xte, N, W_in, V_G1_bias, V_G2_bias, surr,
        args.g_in, args.C_b_F, args.dt_s, T_steps, rng_seed=seed + 1)
    wall = time.time() - t0

    W = ridge_lstsq(feats_tr, ytr)
    train_acc = float((predict(feats_tr, W) == ytr).mean())
    test_acc = float((predict(feats_te, W) == yte).mean())

    return {
        "seed": seed,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "wall_s": wall,
        "vb_rail_frac_train": rail_tr,
        "vb_rail_frac_test": rail_te,
        "clip_rate_train": clip_tr,
        "clip_rate_test": clip_te,
        "T_steps": T_steps,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cell_id", required=True, type=str)
    p.add_argument("--C_b_fF", required=True, type=float)
    p.add_argument("--V_G2_bias", required=True, type=float)
    p.add_argument("--dt_s", required=True, type=float)
    p.add_argument("--g_in", required=True, type=float)
    p.add_argument("--seeds", required=True, type=int, nargs="+")
    p.add_argument("--subsample", required=True, type=int, nargs=2,
                   metavar=("TRAIN", "TEST"))
    p.add_argument("--n_units", type=int, default=128)
    p.add_argument("--surrogate", default="results/z220_4d_dense/surrogate_4d_dense.npz")
    p.add_argument("--out_dir", default="results/sweep_v1")
    args = p.parse_args()
    args.C_b_F = args.C_b_fF * 1e-15

    set_thermal_env()
    print(f"[cell {args.cell_id}] C_b={args.C_b_fF}fF V_G2={args.V_G2_bias}V "
          f"dt={args.dt_s:.0e}s g_in={args.g_in}", flush=True)
    print(f"  device = {get_device()}", flush=True)

    out_root = Path(args.out_dir) / f"cell_{args.cell_id}"
    out_root.mkdir(parents=True, exist_ok=True)
    summary_path = out_root / "summary.json"
    if summary_path.exists():
        print(f"  SKIP: summary exists at {summary_path}", flush=True)
        sys.exit(0)

    per_seed = []
    for s in args.seeds:
        try:
            r = run_one_seed(args, s)
        except Exception as e:
            r = {"seed": s, "error": repr(e)}
        per_seed.append(r)
        print(f"  seed {s}: {r}", flush=True)

    accs = [r["test_acc"] for r in per_seed if "test_acc" in r]
    rails = [r["vb_rail_frac_test"] for r in per_seed if "vb_rail_frac_test" in r]
    clips = [r["clip_rate_test"] for r in per_seed if "clip_rate_test" in r]
    summary = {
        "cell_id": args.cell_id,
        "C_b_fF": args.C_b_fF, "V_G2_bias": args.V_G2_bias,
        "dt_s": args.dt_s, "g_in": args.g_in,
        "n_seeds": len(args.seeds),
        "per_seed": per_seed,
        "mean_acc": float(np.mean(accs)) if accs else None,
        "std_acc":  float(np.std(accs))  if accs else None,
        "mean_vb_rail": float(np.mean(rails)) if rails else None,
        "mean_clip":    float(np.mean(clips)) if clips else None,
        "subsample": args.subsample, "n_units": args.n_units,
    }
    if accs:
        # bootstrap CI95
        bs = np.array([np.mean(np.random.choice(accs, len(accs), replace=True))
                       for _ in range(2000)])
        summary["ci95"] = [float(np.quantile(bs, 0.025)),
                           float(np.quantile(bs, 0.975))]
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[cell {args.cell_id}] DONE acc={summary['mean_acc']:.4f} "
          f"rail={summary['mean_vb_rail']:.3f}", flush=True)


if __name__ == "__main__":
    main()
