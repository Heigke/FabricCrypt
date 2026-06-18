"""DS-N5b: HDC scaling on UCI-HAR. GPU-batched, N in {1024, 10K, 100K, 1M}.

Encoding identical to z284_hdc_baseline.py (record-based HDC, thermometer
levels, sign-binarized prototypes via cosine sim). All compute on GPU.

Memory strategy for large D:
  - Position HVs P: (F, D) int8 -> up to 561 MB at D=1M (kept on GPU).
  - Level codebook L: streamed per-feature; never materialize full (F,Q,D).
  - Bundled HVs: int16 accumulator (N, D) -> at D=1M, 7352 train * 1e6 * 2B
    = 14.7 GB; we accumulate one feature at a time without copies.
  - Test bundle: 2947 * 1e6 * 2B = 5.9 GB.
Total peak ~ 22 GB on unified Grace memory (we have ~96 GB free).

Note: at D=1M, each-class prototype is (6, D) float32 = 24 MB only.
"""
from __future__ import annotations
import argparse, json, time, gc
from pathlib import Path
import numpy as np
import torch


def load_uci_har(root):
    root = Path(root)
    Xtr = np.loadtxt(root / "train" / "X_train.txt", dtype=np.float32)
    ytr = np.loadtxt(root / "train" / "y_train.txt", dtype=np.int64) - 1
    Xte = np.loadtxt(root / "test" / "X_test.txt", dtype=np.float32)
    yte = np.loadtxt(root / "test" / "y_test.txt", dtype=np.int64) - 1
    return Xtr, ytr, Xte, yte


def quantize(X, mins, maxs, Q):
    span = (maxs - mins)
    span = np.where(span < 1e-9, 1.0, span)
    Xn = np.clip((X - mins) / span, 0.0, 1.0)
    return np.clip(np.floor(Xn * (Q - 1) + 0.5).astype(np.int32), 0, Q - 1)


def build_level_codebook_one(D, Q, rng_gpu, device):
    """Return (Q, D) int8 thermometer codebook for one feature, on device.

    L[0] random bipolar; flip D/Q random bits per step to reach L[Q-1] = -L[0].
    """
    base = (torch.randint(0, 2, (D,), device=device, generator=rng_gpu,
                          dtype=torch.int8) * 2 - 1)
    flips_per_step = D // Q
    order = torch.randperm(D, device=device, generator=rng_gpu)
    L = torch.empty((Q, D), dtype=torch.int8, device=device)
    L[0] = base
    cur = base.clone()
    for q in range(1, Q):
        idx = order[(q - 1) * flips_per_step: q * flips_per_step]
        cur[idx] = -cur[idx]
        L[q] = cur
    return L


def encode_and_bundle(Xq_tr_gpu, Xq_te_gpu, F, D, Q, device, seed):
    """Stream feature-by-feature; accumulate bundle into int16 tensors.

    Returns (Htr int16 (Ntr,D), Hte int16 (Nte,D)).
    """
    Ntr = Xq_tr_gpu.shape[0]
    Nte = Xq_te_gpu.shape[0]
    Htr = torch.zeros((Ntr, D), dtype=torch.int16, device=device)
    Hte = torch.zeros((Nte, D), dtype=torch.int16, device=device)

    # Reproducible torch RNG on device, derived from seed.
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)

    for f in range(F):
        # Position HV for this feature (D,) int8
        P = (torch.randint(0, 2, (D,), device=device, generator=rng,
                           dtype=torch.int8) * 2 - 1)
        # Level codebook (Q, D) int8
        L = build_level_codebook_one(D, Q, rng, device)
        # bound = L[Xq[:, f]] * P  -> (N, D) int8
        # Accumulate into int16
        idx_tr = Xq_tr_gpu[:, f].to(torch.long)
        idx_te = Xq_te_gpu[:, f].to(torch.long)
        # Process in chunks if needed to keep transient memory low at D=1M.
        bound_tr = (L[idx_tr] * P)  # (Ntr, D) int8
        Htr.add_(bound_tr.to(torch.int16))
        del bound_tr
        bound_te = (L[idx_te] * P)
        Hte.add_(bound_te.to(torch.int16))
        del bound_te, L, P
    return Htr, Hte


def class_prototypes_gpu(Hsum, y, n_classes, device):
    """Sum per-class then L2-normalize, return float32 (C, D)."""
    D = Hsum.shape[1]
    protos = torch.zeros((n_classes, D), dtype=torch.float32, device=device)
    Hf = Hsum.to(torch.float32)
    for c in range(n_classes):
        m = (y == c)
        if m.any():
            protos[c] = Hf[m].sum(dim=0)
    norms = protos.norm(dim=1, keepdim=True).clamp_min(1e-9)
    return protos / norms


def predict_gpu(Hsum, protos):
    H = Hsum.to(torch.float32)
    norms = H.norm(dim=1, keepdim=True).clamp_min(1e-9)
    Hn = H / norms
    sims = Hn @ protos.T
    return sims.argmax(dim=1)


def run_seed(Xq_tr, Xq_te, ytr, yte, D, Q, F, n_classes, seed, device):
    t0 = time.time()
    Xq_tr_gpu = torch.from_numpy(Xq_tr).to(device)
    Xq_te_gpu = torch.from_numpy(Xq_te).to(device)
    ytr_gpu = torch.from_numpy(ytr).to(device)
    yte_gpu = torch.from_numpy(yte).to(device)

    t_enc0 = time.time()
    Htr, Hte = encode_and_bundle(Xq_tr_gpu, Xq_te_gpu, F, D, Q, device, seed)
    torch.cuda.synchronize()
    t_enc = time.time() - t_enc0

    t_cls0 = time.time()
    protos = class_prototypes_gpu(Htr, ytr_gpu, n_classes, device)
    yhat_tr = predict_gpu(Htr, protos)
    yhat_te = predict_gpu(Hte, protos)
    torch.cuda.synchronize()
    t_cls = time.time() - t_cls0

    train_acc = float((yhat_tr == ytr_gpu).float().mean().item())
    test_acc = float((yhat_te == yte_gpu).float().mean().item())
    wall = time.time() - t0

    # Cleanup
    del Htr, Hte, protos, Xq_tr_gpu, Xq_te_gpu, ytr_gpu, yte_gpu
    del yhat_tr, yhat_te
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "seed": int(seed), "D": int(D), "Q": int(Q),
        "train_acc": train_acc, "test_acc": test_acc,
        "wall_s": wall, "wall_encode_s": t_enc, "wall_classify_s": t_cls,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",
                   default="/home/naorw/nsram_queue_sandbox/data/uci_har/"
                           "UCI HAR Dataset")
    p.add_argument("--Ns", type=int, nargs="+",
                   default=[1024, 10000, 100000, 1000000])
    p.add_argument("--Q", type=int, default=32)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--out_dir", default="results/DS_N5b_HDC_scale")
    args = p.parse_args()

    device = torch.device("cuda")
    print(f"[DS-N5b] device={torch.cuda.get_device_name(0)} "
          f"torch={torch.__version__}", flush=True)
    print(f"[DS-N5b] loading UCI-HAR from {args.data_root}", flush=True)
    Xtr, ytr, Xte, yte = load_uci_har(args.data_root)
    F = Xtr.shape[1]
    n_classes = int(max(ytr.max(), yte.max())) + 1
    print(f"[DS-N5b] train {Xtr.shape} test {Xte.shape} F={F} "
          f"C={n_classes}", flush=True)

    mins = Xtr.min(axis=0)
    maxs = Xtr.max(axis=0)
    Xq_tr = quantize(Xtr, mins, maxs, args.Q)
    Xq_te = quantize(Xte, mins, maxs, args.Q)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = {"by_N": {}}
    for N in args.Ns:
        per_seed = []
        for s in args.seeds:
            try:
                print(f"  N={N:>8d} seed={s} starting...", flush=True)
                r = run_seed(Xq_tr, Xq_te, ytr, yte, N, args.Q, F, n_classes,
                             s, device)
                per_seed.append(r)
                # Per-seed checkpoint for crash resilience
                (out_dir / f"seed_N{N}_s{s}.json").write_text(
                    json.dumps(r, indent=2))
                print(f"  N={N:>8d} seed={s} test_acc={r['test_acc']:.4f} "
                      f"wall={r['wall_s']:.1f}s "
                      f"(enc={r['wall_encode_s']:.1f}s, "
                      f"cls={r['wall_classify_s']:.2f}s)", flush=True)
            except torch.cuda.OutOfMemoryError as e:
                print(f"  N={N} seed={s} OOM: {e}", flush=True)
                gc.collect(); torch.cuda.empty_cache()
                per_seed.append({"seed": s, "D": N, "error": "OOM"})
                break
            except Exception as e:
                print(f"  N={N} seed={s} ERROR: {e}", flush=True)
                per_seed.append({"seed": s, "D": N, "error": str(e)})
        accs = [r["test_acc"] for r in per_seed if "test_acc" in r]
        walls = [r["wall_s"] for r in per_seed if "wall_s" in r]
        all_results["by_N"][str(N)] = {
            "N": N, "per_seed": per_seed,
            "mean_acc": float(np.mean(accs)) if accs else None,
            "std_acc": float(np.std(accs)) if accs else None,
            "mean_wall_s": float(np.mean(walls)) if walls else None,
        }
        # Save after every N for crash resilience
        (out_dir / "accuracy_vs_N.json").write_text(
            json.dumps(all_results, indent=2))
        print(f"[DS-N5b] saved progress for N={N}", flush=True)

    print("[DS-N5b] DONE", flush=True)


if __name__ == "__main__":
    main()
