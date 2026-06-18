"""z272 — GPU-native single-cell sweep harness.

Same physics as z269 but ALL operations on torch device:
  - MNIST tensors on GPU
  - Poisson encoding via torch.rand
  - Surrogate trilinear lookup via torch.gather + bucketize
  - Ridge readout via torch.linalg.solve
  - Body-state time-stepping in batched torch loop

Per-cell wall on Strix Halo: target ~10–20 s (vs ~25 min for z269 numpy).
Per-cell wall on GB10: target ~1–3 s.

Usage:
    python z272_sweep_cell_gpu.py --cell_id <id> \
        --C_b_fF <C_b> --V_G2_bias <V> --dt_s <dt> --g_in <g> \
        --seeds 0 1 2 3 --subsample 5000 1000 \
        --surrogate path/to/surrogate.npz --out_dir results/sweep_v2

For ROCm: export HSA_OVERRIDE_GFX_VERSION=11.0.0 before launch.
"""
from __future__ import annotations
import os, sys, time, json, argparse
from pathlib import Path
import numpy as np
import torch


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_surrogate_torch(path, device):
    """Load 4D surrogate as torch tensors on `device`. Axes ascending."""
    z = np.load(path)
    out = {
        "I_d":   torch.tensor(z["Id"],    dtype=torch.float32, device=device),
        "I_ii":  torch.tensor(z["Iii"],   dtype=torch.float32, device=device),
        "I_leak":torch.tensor(z["Ileak"], dtype=torch.float32, device=device),
        "ax_VG1":torch.tensor(z["vg1_axis"], dtype=torch.float32, device=device),
        "ax_VG2":torch.tensor(z["vg2_axis"], dtype=torch.float32, device=device),
        "ax_Vd": torch.tensor(z["vd_axis"],  dtype=torch.float32, device=device),
        "ax_Vb": torch.tensor(z["vb_axis"],  dtype=torch.float32, device=device),
    }
    return out


def bucketize_index(values, axis):
    """Map values to grid index (lower neighbor), clamped to [0, n-2]."""
    n = axis.shape[0]
    idx = torch.bucketize(values, axis) - 1
    return idx.clamp(0, n - 2)


def query_surrogate_torch(surr, VG1, VG2, Vd, Vb):
    """Nearest-neighbor (lower-bin) lookup. Inputs are torch tensors of same
    shape. Returns (I_d, I_ii, I_leak) of that shape."""
    iVG1 = bucketize_index(VG1, surr["ax_VG1"])
    iVG2 = bucketize_index(VG2, surr["ax_VG2"])
    iVd  = bucketize_index(Vd,  surr["ax_Vd"])
    iVb  = bucketize_index(Vb,  surr["ax_Vb"])
    return (surr["I_d"][iVG1, iVG2, iVd, iVb],
            surr["I_ii"][iVG1, iVG2, iVd, iVb],
            surr["I_leak"][iVG1, iVG2, iVd, iVb])


def load_mnist_28x28(subsample_train, subsample_test, seed, device):
    """Load + subsample MNIST 28×28; return tensors on `device`."""
    try:
        from torchvision import datasets, transforms
        train = datasets.MNIST("/tmp/mnist", train=True,  download=True)
        test  = datasets.MNIST("/tmp/mnist", train=False, download=True)
        Xtr = train.data.float().view(-1, 784) / 255.0
        ytr = train.targets
        Xte = test.data.float().view(-1, 784) / 255.0
        yte = test.targets
    except Exception:
        from sklearn.datasets import fetch_openml
        m = fetch_openml("mnist_784", version=1, parser="auto", as_frame=False)
        X = torch.tensor(m.data, dtype=torch.float32) / 255.0
        y = torch.tensor(m.target.astype(int), dtype=torch.long)
        Xtr, Xte = X[:60000], X[60000:]
        ytr, yte = y[:60000], y[60000:]
    g = torch.Generator().manual_seed(seed)
    idx_tr = torch.randperm(Xtr.shape[0], generator=g)[:subsample_train]
    idx_te = torch.randperm(Xte.shape[0], generator=g)[:subsample_test]
    return (Xtr[idx_tr].to(device), ytr[idx_tr].to(device),
            Xte[idx_te].to(device), yte[idx_te].to(device))


def nsram_transient_features_gpu(X, N, W_in, V_G1_bias, V_G2_bias, surr,
                                  g_in, C_b_F, dt_s, T_steps, vd=1.0,
                                  generator=None):
    """Batched body-state SNN feature extraction on GPU.

    Inputs are torch tensors. Returns (features, vb_rail_frac, clip_rate)."""
    device = X.device
    B = X.shape[0]
    Vb_min = surr["ax_Vb"][0]
    Vb_max = surr["ax_Vb"][-1]
    VG1_min = surr["ax_VG1"][0]
    VG1_max = surr["ax_VG1"][-1]

    spike_accum = torch.zeros(B, N, device=device)
    Vb = torch.zeros(B, N, device=device)
    rail_count = torch.zeros(1, device=device)
    rail_total = torch.zeros(1, device=device)
    clip_count = torch.zeros(1, device=device)
    clip_total = torch.zeros(1, device=device)

    p_max = 0.5
    VG2_2d = V_G2_bias.expand(B, N)
    Vd_2d  = torch.full((B, N), float(vd), device=device)

    for t in range(T_steps):
        spikes = (torch.rand(X.shape, device=device, generator=generator)
                  < (X * p_max)).float()
        drive  = spikes @ W_in.T            # (B, N)
        VG1    = V_G1_bias.unsqueeze(0) + g_in * drive
        VG1_clipped = VG1.clamp(VG1_min, VG1_max)
        clip_count = clip_count + (VG1 != VG1_clipped).sum().float()
        clip_total = clip_total + float(VG1.numel())
        VG1 = VG1_clipped
        Vb_clamped = Vb.clamp(Vb_min, Vb_max)
        I_d, I_ii, I_leak = query_surrogate_torch(surr, VG1, VG2_2d, Vd_2d,
                                                   Vb_clamped)
        Vb_new = Vb + dt_s * (I_ii - I_leak) / C_b_F
        Vb_new_clamped = Vb_new.clamp(Vb_min, Vb_max)
        rail_count = rail_count + (Vb_new != Vb_new_clamped).sum().float()
        rail_total = rail_total + float(Vb_new.numel())
        Vb = Vb_new_clamped
        spike_accum = spike_accum + I_d.abs() / T_steps

    feats = (spike_accum + 1e-18).log10()
    feats = (feats - feats.mean(dim=0, keepdim=True)) / (
            feats.std(dim=0, keepdim=True) + 1e-9)
    rail_frac = (rail_count / rail_total.clamp_min(1)).item()
    clip_rate = (clip_count / clip_total.clamp_min(1)).item()
    return feats, rail_frac, clip_rate


def ridge_lstsq_torch(X, y, alpha=1e-3, n_classes=10):
    Y = torch.nn.functional.one_hot(y, n_classes).float()
    A = X.T @ X + alpha * torch.eye(X.shape[1], device=X.device)
    B = X.T @ Y
    return torch.linalg.solve(A, B)


def predict_torch(X, W):
    return (X @ W).argmax(dim=1)


def run_one_seed(args, seed, device, surr):
    Xtr, ytr, Xte, yte = load_mnist_28x28(args.subsample[0],
                                          args.subsample[1], seed, device)
    g = torch.Generator(device=device).manual_seed(seed + 31337)
    N = args.n_units
    W_in = torch.randn(N, 784, generator=g, device=device)
    W_in = W_in / (W_in.norm(dim=1, keepdim=True) + 1e-9)
    V_G1_bias = torch.empty(N, device=device).uniform_(0.20, 0.40,
                                                          generator=g)
    V_G2_bias = torch.full((N,), args.V_G2_bias, device=device)
    g_p = torch.Generator(device=device).manual_seed(seed + 99991)

    t0 = time.time()
    T_steps = max(int(round(1e-5 / args.dt_s)), 10)
    feats_tr, rail_tr, clip_tr = nsram_transient_features_gpu(
        Xtr, N, W_in, V_G1_bias, V_G2_bias, surr,
        args.g_in, args.C_b_F, args.dt_s, T_steps, generator=g_p)
    feats_te, rail_te, clip_te = nsram_transient_features_gpu(
        Xte, N, W_in, V_G1_bias, V_G2_bias, surr,
        args.g_in, args.C_b_F, args.dt_s, T_steps, generator=g_p)
    wall = time.time() - t0

    W = ridge_lstsq_torch(feats_tr, ytr)
    train_acc = (predict_torch(feats_tr, W) == ytr).float().mean().item()
    test_acc  = (predict_torch(feats_te, W) == yte).float().mean().item()

    return {"seed": seed, "train_acc": float(train_acc),
            "test_acc": float(test_acc), "wall_s": wall,
            "vb_rail_frac_train": rail_tr, "vb_rail_frac_test": rail_te,
            "clip_rate_train": clip_tr, "clip_rate_test": clip_te,
            "T_steps": T_steps}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cell_id", required=True, type=str)
    p.add_argument("--C_b_fF", required=True, type=float)
    p.add_argument("--V_G2_bias", required=True, type=float)
    p.add_argument("--dt_s", required=True, type=float)
    p.add_argument("--g_in", required=True, type=float)
    p.add_argument("--seeds", required=True, type=int, nargs="+")
    p.add_argument("--subsample", required=True, type=int, nargs=2)
    p.add_argument("--n_units", type=int, default=128)
    p.add_argument("--surrogate",
                   default="results/z271_pmp3_dense_surrogate/surrogate_4d_v2.npz")
    p.add_argument("--out_dir", default="results/sweep_v2")
    args = p.parse_args()
    args.C_b_F = args.C_b_fF * 1e-15

    device = get_device()
    print(f"[cell {args.cell_id}] device={device} surrogate={args.surrogate}",
          flush=True)
    out_root = Path(args.out_dir) / f"cell_{args.cell_id}"
    out_root.mkdir(parents=True, exist_ok=True)
    summary_path = out_root / "summary.json"
    if summary_path.exists():
        print(f"  SKIP: {summary_path} exists", flush=True)
        sys.exit(0)

    surr = load_surrogate_torch(args.surrogate, device)

    per_seed = []
    for s in args.seeds:
        try:
            r = run_one_seed(args, s, device, surr)
        except Exception as e:
            import traceback; traceback.print_exc()
            r = {"seed": s, "error": repr(e)}
        per_seed.append(r)
        print(f"  seed {s}: {r}", flush=True)

    accs = [r["test_acc"] for r in per_seed if "test_acc" in r]
    rails = [r["vb_rail_frac_test"] for r in per_seed if "vb_rail_frac_test" in r]
    clips = [r["clip_rate_test"] for r in per_seed if "clip_rate_test" in r]
    summary = {
        "cell_id": args.cell_id, "C_b_fF": args.C_b_fF,
        "V_G2_bias": args.V_G2_bias, "dt_s": args.dt_s, "g_in": args.g_in,
        "n_seeds": len(args.seeds), "per_seed": per_seed,
        "mean_acc": float(np.mean(accs)) if accs else None,
        "std_acc": float(np.std(accs)) if accs else None,
        "mean_vb_rail": float(np.mean(rails)) if rails else None,
        "mean_clip": float(np.mean(clips)) if clips else None,
        "subsample": args.subsample, "n_units": args.n_units,
        "device": str(device),
    }
    if len(accs) >= 2:
        bs = np.array([np.mean(np.random.choice(accs, len(accs), replace=True))
                       for _ in range(2000)])
        summary["ci95"] = [float(np.quantile(bs, 0.025)),
                           float(np.quantile(bs, 0.975))]
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[cell {args.cell_id}] DONE acc={summary['mean_acc']} "
          f"rail={summary['mean_vb_rail']:.3f}", flush=True)


if __name__ == "__main__":
    main()
