"""DS-N5 baseline: pure HDC classifier on UCI-HAR (no NS-RAM).

Encoding rule (LOCKED before run):
  - Record-based HDC: each feature f at position p has a position
    hypervector P_f (random bipolar +/-1) and a level codebook L_f of
    Q_LEVELS bipolar HVs forming a thermometer (interpolated between
    L_f[0] random and L_f[Q-1] = -L_f[0]). Quantize feature value into
    [0, Q_LEVELS-1] using training-set min/max. Sample HV = sign(sum_f
    P_f * L_f[q]) (bundling + binding via XOR ~ elementwise product).
  - Class prototype: bipolarized sum of all train sample HVs of class.
  - Predict: argmax cosine similarity over prototypes.

Dimensions LOCKED: D=10000. Q_LEVELS=32. 5 seeds.

Usage:
  python z284_hdc_baseline.py --data_root data/uci_har/'UCI HAR Dataset' \
      --D 10000 --Q 32 --seeds 0 1 2 3 4 \
      --out results/z285_nsram_hdc/baseline_hdc.json
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np


def load_uci_har(root):
    root = Path(root)
    Xtr = np.loadtxt(root / "train" / "X_train.txt", dtype=np.float32)
    ytr = np.loadtxt(root / "train" / "y_train.txt", dtype=np.int64) - 1
    Xte = np.loadtxt(root / "test"  / "X_test.txt",  dtype=np.float32)
    yte = np.loadtxt(root / "test"  / "y_test.txt",  dtype=np.int64) - 1
    return Xtr, ytr, Xte, yte


def build_codebooks(n_features, D, Q, rng):
    """Position HVs (F,D) and level codebooks (F,Q,D), bipolar ±1."""
    P = rng.choice([-1, 1], size=(n_features, D)).astype(np.int8)
    # Thermometer: L[q] flips q*(D/Q) random bits from L[0] -> -L[0].
    L = np.empty((n_features, Q, D), dtype=np.int8)
    base = rng.choice([-1, 1], size=(n_features, D)).astype(np.int8)
    flips_per_step = D // Q  # ~312 bits/level for D=10000,Q=32
    for f in range(n_features):
        L[f, 0] = base[f]
        order = rng.permutation(D)
        cur = base[f].copy()
        for q in range(1, Q):
            idx = order[(q - 1) * flips_per_step: q * flips_per_step]
            cur[idx] = -cur[idx]
            L[f, q] = cur
    return P, L


def quantize(X, mins, maxs, Q):
    """Map each feature to int [0, Q-1] using train min/max."""
    span = (maxs - mins)
    span = np.where(span < 1e-9, 1.0, span)
    Xn = (X - mins) / span
    Xn = np.clip(Xn, 0.0, 1.0)
    q = np.floor(Xn * (Q - 1) + 0.5).astype(np.int32)
    return np.clip(q, 0, Q - 1)


def encode_samples(Xq, P, L):
    """Return (N, D) int16 bundled HVs. Xq is (N, F) int quantized."""
    N, F = Xq.shape
    D = P.shape[1]
    acc = np.zeros((N, D), dtype=np.int32)
    for f in range(F):
        # bind: P[f] * L[f, Xq[:,f]]  shape (N, D) int8
        Lf = L[f][Xq[:, f]]                # (N, D) int8
        bound = Lf * P[f][None, :]         # (N, D) int8 (-1 or 1)
        acc += bound.astype(np.int32)
    return acc  # don't sign-binarize yet, keep magnitude for cosine


def class_prototypes(Hsum, y, n_classes):
    """Bundle by summing per-class then sign-binarize to ±1."""
    D = Hsum.shape[1]
    protos = np.zeros((n_classes, D), dtype=np.float32)
    for c in range(n_classes):
        m = y == c
        if m.any():
            protos[c] = Hsum[m].sum(axis=0).astype(np.float32)
    # Normalize for cosine
    norms = np.linalg.norm(protos, axis=1, keepdims=True)
    norms = np.where(norms < 1e-9, 1.0, norms)
    return protos / norms


def predict(Hsum, protos):
    H = Hsum.astype(np.float32)
    norms = np.linalg.norm(H, axis=1, keepdims=True)
    norms = np.where(norms < 1e-9, 1.0, norms)
    Hn = H / norms
    sims = Hn @ protos.T  # (N, C)
    return sims.argmax(axis=1)


def run_seed(Xtr, ytr, Xte, yte, D, Q, seed):
    rng = np.random.default_rng(seed)
    n_classes = int(max(ytr.max(), yte.max())) + 1
    F = Xtr.shape[1]
    mins = Xtr.min(axis=0)
    maxs = Xtr.max(axis=0)
    Xtrq = quantize(Xtr, mins, maxs, Q)
    Xteq = quantize(Xte, mins, maxs, Q)

    t0 = time.time()
    P, L = build_codebooks(F, D, Q, rng)
    Htr = encode_samples(Xtrq, P, L)
    Hte = encode_samples(Xteq, P, L)
    protos = class_prototypes(Htr, ytr, n_classes)
    ytr_hat = predict(Htr, protos)
    yte_hat = predict(Hte, protos)
    wall = time.time() - t0
    return {
        "seed": seed, "train_acc": float((ytr_hat == ytr).mean()),
        "test_acc": float((yte_hat == yte).mean()),
        "wall_s": wall, "D": int(D), "Q": int(Q),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="data/uci_har/UCI HAR Dataset")
    p.add_argument("--D", type=int, default=10000)
    p.add_argument("--Q", type=int, default=32)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--out", default="results/z285_nsram_hdc/baseline_hdc.json")
    args = p.parse_args()

    print(f"[z284] loading UCI-HAR from {args.data_root}", flush=True)
    Xtr, ytr, Xte, yte = load_uci_har(args.data_root)
    print(f"[z284] train {Xtr.shape} test {Xte.shape} "
          f"classes={int(ytr.max())+1}", flush=True)

    per_seed = []
    for s in args.seeds:
        r = run_seed(Xtr, ytr, Xte, yte, args.D, args.Q, s)
        per_seed.append(r)
        print(f"  seed {s}: test_acc={r['test_acc']:.4f} "
              f"train_acc={r['train_acc']:.4f} wall={r['wall_s']:.1f}s",
              flush=True)

    accs = [r["test_acc"] for r in per_seed]
    summary = {
        "experiment": "z284_hdc_baseline_uci_har",
        "D": args.D, "Q": args.Q, "n_seeds": len(args.seeds),
        "per_seed": per_seed,
        "mean_acc": float(np.mean(accs)),
        "std_acc": float(np.std(accs)),
    }
    if len(accs) >= 2:
        rng = np.random.default_rng(0)
        bs = np.array([rng.choice(accs, len(accs), replace=True).mean()
                       for _ in range(4000)])
        summary["ci95"] = [float(np.quantile(bs, 0.025)),
                           float(np.quantile(bs, 0.975))]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"[z284] DONE mean_acc={summary['mean_acc']:.4f} "
          f"+/- {summary['std_acc']:.4f}  -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
