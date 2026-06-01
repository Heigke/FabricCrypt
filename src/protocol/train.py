"""Train T2 (anomaly) + T3 (twin) classifiers under nonce-keyed signatures.

  python -m src.protocol.train --n_train 400 --n_seeds 30 \\
      --peer_npz data/<peer>_paired_sigs.npz \\
      --out_dir data/
"""
from __future__ import annotations
import os
import sys
import time
import json
import argparse
import socket
import numpy as np
import torch

from .nonce_signature import NonceSig
from .nonce_derivation import fresh_nonce
from .classifier import (
    DIM, TwinMLP, collect_paired, train_T2_anomaly, train_T3_twin,
)
from ..signature.thermal import thermal_guard, hostname


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=400)
    ap.add_argument("--n_seeds", type=int, default=30)
    ap.add_argument("--peer_npz", default=None,
                    help="paired_sigs.npz from a peer host (must contain 'sigs')")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out_dir", default="data")
    args = ap.parse_args()

    host = hostname()
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[train] host={host}  n_train={args.n_train}  n_seeds={args.n_seeds}")

    rng = np.random.default_rng(int(time.time()) & 0xFFFFFFFF)
    sig = NonceSig(host=host)
    t0 = time.time()
    nonces, own_sigs = collect_paired(sig, args.n_train, rng,
                                       thermal_guard=thermal_guard)
    print(f"[train] collected {args.n_train} own pairs in {time.time()-t0:.1f}s")
    paired_path = os.path.join(args.out_dir, f"{host}_paired_sigs.npz")
    np.savez_compressed(paired_path, nonces=nonces, sigs=own_sigs)
    print(f"[train] saved {paired_path}")

    peer_sigs = np.zeros((0, DIM), dtype=np.float32)
    if args.peer_npz and os.path.exists(args.peer_npz):
        d = np.load(args.peer_npz)
        peer_sigs = d["sigs"].astype(np.float32)
        print(f"[train] loaded peer sigs from {args.peer_npz}: {peer_sigs.shape}")

    device = torch.device(args.device)
    print(f"[train] T2 anomaly ({args.n_seeds} seeds)...")
    t2_aurocs, _ = train_T2_anomaly(own_sigs, n_seeds=args.n_seeds,
                                     device=device, verbose=False)
    print(f"[train] T3 twin ({args.n_seeds} seeds)...")
    t3_aurocs, t3_state = train_T3_twin(own_sigs, peer_sigs,
                                         n_seeds=args.n_seeds,
                                         device=device, verbose=False)

    t3_path = os.path.join(args.out_dir, f"{host}_t3_best.pt")
    if t3_state is not None:
        torch.save(t3_state, t3_path)
        print(f"[train] saved {t3_path}")

    summary = {
        "host": host, "n_train": args.n_train, "n_seeds": args.n_seeds,
        "peer_npz": args.peer_npz,
        "t2": {"aurocs": t2_aurocs, "mean": float(np.mean(t2_aurocs)),
               "ci95_lo": float(np.percentile(t2_aurocs, 2.5)),
               "ci95_hi": float(np.percentile(t2_aurocs, 97.5))},
        "t3": {"aurocs": t3_aurocs, "mean": float(np.mean(t3_aurocs)),
               "ci95_lo": float(np.percentile(t3_aurocs, 2.5)),
               "ci95_hi": float(np.percentile(t3_aurocs, 97.5))},
        "t": time.time(),
    }
    with open(os.path.join(args.out_dir, f"{host}_training.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[train] T2 mean AUROC={summary['t2']['mean']:.3f}  "
          f"T3 mean AUROC={summary['t3']['mean']:.3f}")


if __name__ == "__main__":
    main()
