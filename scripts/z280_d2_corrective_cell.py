"""z280 — D2 corrective falsification cell harness (post-O43 oracle).

Differences from z272/z277:
  - Uses MEP-2 dense v3 surrogate by default
  - Quadrilinear interp (from z277_mep1_trilinear_z272.py)
  - FULL 10k MNIST test set (no subsample) by default
  - 10 seeds with JOINT bootstrap over weight init AND Poisson encoding
  - Hard reports clip_rate + vb_rail_frac for gate verification (cell can
    PASS-accuracy but FAIL-clip/rail)
  - Optional --dataset switch for Fashion-MNIST and CIFAR-10-greyscale
    falsification (oracle-suggested)

NO-CHEAT discipline: no parameter tuning to hit gates. Gates locked
externally in 01_LOG.md before this script runs.
"""
from __future__ import annotations
import os, sys, time, json, argparse
from pathlib import Path
import numpy as np
import torch


def get_device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def load_surrogate(path, device):
    z = np.load(path)
    return {
        "I_d": torch.tensor(z["Id"], dtype=torch.float32, device=device),
        "I_ii": torch.tensor(z["Iii"], dtype=torch.float32, device=device),
        "I_leak": torch.tensor(z["Ileak"], dtype=torch.float32, device=device),
        "ax_VG1": torch.tensor(z["vg1_axis"], dtype=torch.float32, device=device),
        "ax_VG2": torch.tensor(z["vg2_axis"], dtype=torch.float32, device=device),
        "ax_Vd": torch.tensor(z["vd_axis"], dtype=torch.float32, device=device),
        "ax_Vb": torch.tensor(z["vb_axis"], dtype=torch.float32, device=device),
    }


def _frac_index(values, axis):
    """Return (i_lo, t) where t in [0,1] is fractional position in [i_lo, i_lo+1]."""
    n = axis.shape[0]
    idx = torch.bucketize(values.contiguous(), axis) - 1
    idx = idx.clamp(0, n - 2)
    lo = axis[idx]
    hi = axis[idx + 1]
    t = ((values - lo) / (hi - lo + 1e-30)).clamp(0.0, 1.0)
    return idx, t


def query_surrogate_quadrilinear(surr, VG1, VG2, Vd, Vb):
    """Quadrilinear lookup over 4D surrogate."""
    i1, t1 = _frac_index(VG1, surr["ax_VG1"])
    i2, t2 = _frac_index(VG2, surr["ax_VG2"])
    i3, t3 = _frac_index(Vd, surr["ax_Vd"])
    i4, t4 = _frac_index(Vb, surr["ax_Vb"])

    Id_acc = torch.zeros_like(VG1)
    Iii_acc = torch.zeros_like(VG1)
    Ileak_acc = torch.zeros_like(VG1)

    for d1 in (0, 1):
        w1 = t1 if d1 == 1 else (1 - t1)
        for d2 in (0, 1):
            w2 = t2 if d2 == 1 else (1 - t2)
            for d3 in (0, 1):
                w3 = t3 if d3 == 1 else (1 - t3)
                for d4 in (0, 1):
                    w4 = t4 if d4 == 1 else (1 - t4)
                    w = w1 * w2 * w3 * w4
                    Id_acc = Id_acc + w * surr["I_d"][i1 + d1, i2 + d2, i3 + d3, i4 + d4]
                    Iii_acc = Iii_acc + w * surr["I_ii"][i1 + d1, i2 + d2, i3 + d3, i4 + d4]
                    Ileak_acc = Ileak_acc + w * surr["I_leak"][i1 + d1, i2 + d2, i3 + d3, i4 + d4]
    return Id_acc, Iii_acc, Ileak_acc


def load_dataset(name, device):
    """Load dataset; return (X_train, y_train, X_test, y_test) on device.
    Returns full datasets (no subsample)."""
    if name == "mnist":
        from torchvision import datasets
        train = datasets.MNIST("/tmp/mnist", train=True, download=True)
        test = datasets.MNIST("/tmp/mnist", train=False, download=True)
        Xtr = train.data.float().view(-1, 784) / 255.0
        Xte = test.data.float().view(-1, 784) / 255.0
        return Xtr.to(device), train.targets.to(device), Xte.to(device), test.targets.to(device), 784
    if name == "fashion-mnist":
        from torchvision import datasets
        train = datasets.FashionMNIST("/tmp/fashion-mnist", train=True, download=True)
        test = datasets.FashionMNIST("/tmp/fashion-mnist", train=False, download=True)
        Xtr = train.data.float().view(-1, 784) / 255.0
        Xte = test.data.float().view(-1, 784) / 255.0
        return Xtr.to(device), train.targets.to(device), Xte.to(device), test.targets.to(device), 784
    if name == "cifar10gr":
        from torchvision import datasets
        train = datasets.CIFAR10("/tmp/cifar10", train=True, download=True)
        test = datasets.CIFAR10("/tmp/cifar10", train=False, download=True)
        # convert to greyscale 28x28 by mean over channels then resize
        Xtr_full = torch.tensor(train.data, dtype=torch.float32).mean(dim=-1) / 255.0
        Xte_full = torch.tensor(test.data, dtype=torch.float32).mean(dim=-1) / 255.0
        # resize 32->28 via avg-pool 4x4 -> reshape (could also use interp)
        Xtr = torch.nn.functional.avg_pool2d(Xtr_full.unsqueeze(1), kernel_size=2, stride=1, padding=0)
        # crop center to 28x28
        Xtr = Xtr[:, :, 2:30, 2:30].reshape(-1, 784)
        Xte = torch.nn.functional.avg_pool2d(Xte_full.unsqueeze(1), kernel_size=2, stride=1, padding=0)
        Xte = Xte[:, :, 2:30, 2:30].reshape(-1, 784)
        return (Xtr.to(device), torch.tensor(train.targets, device=device),
                Xte.to(device), torch.tensor(test.targets, device=device), 784)
    raise ValueError(f"unknown dataset {name}")


def nsram_features(X, N, W_in, V_G1_bias, V_G2_bias, surr, g_in, C_b_F, dt_s,
                    T_steps, vd=1.0, gen=None):
    device = X.device
    B = X.shape[0]
    Vb_min, Vb_max = surr["ax_Vb"][0], surr["ax_Vb"][-1]
    VG1_min, VG1_max = surr["ax_VG1"][0], surr["ax_VG1"][-1]
    spike_accum = torch.zeros(B, N, device=device)
    Vb = torch.zeros(B, N, device=device)
    rail_count = torch.zeros(1, device=device)
    rail_total = 0.0
    clip_count = torch.zeros(1, device=device)
    clip_total = 0.0
    VG2_2d = V_G2_bias.expand(B, N)
    Vd_2d = torch.full((B, N), float(vd), device=device)
    p_max = 0.5
    for t in range(T_steps):
        spikes = (torch.rand(X.shape, device=device, generator=gen) < (X * p_max)).float()
        drive = spikes @ W_in.T
        VG1 = V_G1_bias.unsqueeze(0) + g_in * drive
        VG1_clipped = VG1.clamp(VG1_min, VG1_max)
        clip_count += (VG1 != VG1_clipped).sum().float()
        clip_total += VG1.numel()
        VG1 = VG1_clipped
        Vb_c = Vb.clamp(Vb_min, Vb_max)
        I_d, I_ii, I_leak = query_surrogate_quadrilinear(surr, VG1, VG2_2d, Vd_2d, Vb_c)
        Vb_new = Vb + dt_s * (I_ii - I_leak) / C_b_F
        Vb_new_c = Vb_new.clamp(Vb_min, Vb_max)
        rail_count += (Vb_new != Vb_new_c).sum().float()
        rail_total += Vb_new.numel()
        Vb = Vb_new_c
        spike_accum = spike_accum + I_d.abs() / T_steps
    feats = (spike_accum + 1e-18).log10()
    feats = (feats - feats.mean(dim=0, keepdim=True)) / (feats.std(dim=0, keepdim=True) + 1e-9)
    return feats, (rail_count / max(rail_total, 1)).item(), (clip_count / max(clip_total, 1)).item()


def ridge_solve(X, y, alpha=1e-3, n_classes=10):
    Y = torch.nn.functional.one_hot(y, n_classes).float()
    A = X.T @ X + alpha * torch.eye(X.shape[1], device=X.device)
    return torch.linalg.solve(A, X.T @ Y)


def run_one_seed(args, seed, device, surr, Xtr, ytr, Xte, yte):
    g = torch.Generator(device=device).manual_seed(seed + 31337)
    N = args.n_units
    W_in = torch.randn(N, 784, generator=g, device=device)
    W_in = W_in / (W_in.norm(dim=1, keepdim=True) + 1e-9)
    V_G1_bias = torch.empty(N, device=device).uniform_(0.20, 0.40, generator=g)
    V_G2_bias = torch.full((N,), args.V_G2_bias, device=device)
    # Joint bootstrap: distinct Poisson generator per seed
    gp = torch.Generator(device=device).manual_seed(seed + 99991)
    T_steps = max(int(round(1e-5 / args.dt_s)), 10)
    t0 = time.time()
    feats_tr, rail_tr, clip_tr = nsram_features(Xtr, N, W_in, V_G1_bias, V_G2_bias,
                                                  surr, args.g_in, args.C_b_F, args.dt_s,
                                                  T_steps, gen=gp)
    feats_te, rail_te, clip_te = nsram_features(Xte, N, W_in, V_G1_bias, V_G2_bias,
                                                  surr, args.g_in, args.C_b_F, args.dt_s,
                                                  T_steps, gen=gp)
    wall = time.time() - t0
    W = ridge_solve(feats_tr, ytr)
    train_acc = ((feats_tr @ W).argmax(dim=1) == ytr).float().mean().item()
    test_acc = ((feats_te @ W).argmax(dim=1) == yte).float().mean().item()
    return {"seed": seed, "train_acc": float(train_acc), "test_acc": float(test_acc),
            "wall_s": wall, "vb_rail_frac_train": rail_tr, "vb_rail_frac_test": rail_te,
            "clip_rate_train": clip_tr, "clip_rate_test": clip_te, "T_steps": T_steps}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cell_id", required=True)
    p.add_argument("--C_b_fF", required=True, type=float)
    p.add_argument("--V_G2_bias", required=True, type=float)
    p.add_argument("--dt_s", required=True, type=float)
    p.add_argument("--g_in", required=True, type=float)
    p.add_argument("--seeds", required=True, type=int, nargs="+")
    p.add_argument("--dataset", default="mnist", choices=["mnist", "fashion-mnist", "cifar10gr"])
    p.add_argument("--n_units", type=int, default=128)
    p.add_argument("--surrogate",
                   default="results/z278_mep2_surrogate_v3/surrogate_4d_v3.npz")
    p.add_argument("--out_dir", default="results/sweep_d2")
    args = p.parse_args()
    args.C_b_F = args.C_b_fF * 1e-15
    device = get_device()
    print(f"[D2 cell {args.cell_id}] device={device} dataset={args.dataset}", flush=True)
    out_root = Path(args.out_dir) / f"cell_{args.cell_id}_{args.dataset}"
    out_root.mkdir(parents=True, exist_ok=True)
    summary_path = out_root / "summary.json"
    if summary_path.exists():
        print(f"  SKIP", flush=True); sys.exit(0)
    surr = load_surrogate(args.surrogate, device)
    Xtr, ytr, Xte, yte, _ = load_dataset(args.dataset, device)
    print(f"  data shape: Xtr={Xtr.shape} Xte={Xte.shape}", flush=True)
    per_seed = []
    for s in args.seeds:
        try:
            r = run_one_seed(args, s, device, surr, Xtr, ytr, Xte, yte)
        except Exception as e:
            import traceback; traceback.print_exc(); r = {"seed": s, "error": repr(e)}
        per_seed.append(r)
        print(f"  seed {s}: acc={r.get('test_acc')} rail={r.get('vb_rail_frac_test')} clip={r.get('clip_rate_test')}", flush=True)
    accs = [r["test_acc"] for r in per_seed if "test_acc" in r]
    rails = [r["vb_rail_frac_test"] for r in per_seed if "vb_rail_frac_test" in r]
    clips = [r["clip_rate_test"] for r in per_seed if "clip_rate_test" in r]
    summary = {
        "cell_id": args.cell_id, "dataset": args.dataset,
        "C_b_fF": args.C_b_fF, "V_G2_bias": args.V_G2_bias,
        "dt_s": args.dt_s, "g_in": args.g_in,
        "n_seeds": len(args.seeds), "per_seed": per_seed,
        "mean_acc": float(np.mean(accs)) if accs else None,
        "std_acc": float(np.std(accs)) if accs else None,
        "mean_vb_rail": float(np.mean(rails)) if rails else None,
        "max_vb_rail": float(np.max(rails)) if rails else None,
        "mean_clip": float(np.mean(clips)) if clips else None,
        "max_clip": float(np.max(clips)) if clips else None,
        "device": str(device), "n_units": args.n_units,
    }
    if len(accs) >= 2:
        bs = np.array([np.mean(np.random.choice(accs, len(accs), replace=True))
                         for _ in range(2000)])
        summary["ci95"] = [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]
    # Gate compliance (informational; final aggregator decides)
    summary["passes_clip_gate"] = summary.get("max_clip", 1.0) <= 0.05
    summary["passes_rail_gate"] = summary.get("max_vb_rail", 1.0) <= 0.05
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[D2 cell {args.cell_id}] DONE acc={summary['mean_acc']} "
          f"clip_max={summary.get('max_clip',-1):.3f} rail_max={summary.get('max_vb_rail',-1):.3f}",
          flush=True)


if __name__ == "__main__":
    main()
