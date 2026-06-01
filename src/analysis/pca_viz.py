"""Produce a 2D PCA scatter of FabricCrypt signatures across chassis.

Each point is one (R) capture; colour = host. Saves a PNG.
"""
from __future__ import annotations
import os
import argparse
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sigs", nargs="+", help="paths to <host>_sig_v2.npz")
    ap.add_argument("--out_png", default="data/pca.png")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    Xs, ys, names = [], [], []
    for label_idx, path in enumerate(args.sigs):
        d = np.load(path)
        vec = d["vec"]
        host = str(d["host"]) if "host" in d else os.path.basename(path).split("_")[0]
        names.append(host)
        Xs.append(vec)
        ys.append(np.full(len(vec), label_idx))

    X = np.vstack(Xs)
    y = np.concatenate(ys)
    Xz = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)
    pcs = PCA(n_components=2).fit_transform(Xz)
    fig, ax = plt.subplots(figsize=(6, 5))
    for k, n in enumerate(names):
        mask = y == k
        ax.scatter(pcs[mask, 0], pcs[mask, 1], s=40, label=n, alpha=0.7)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_title("FabricCrypt signatures (PCA)")
    ax.legend()
    os.makedirs(os.path.dirname(args.out_png) or ".", exist_ok=True)
    fig.tight_layout(); fig.savefig(args.out_png, dpi=150)
    print(f"saved {args.out_png}")


if __name__ == "__main__":
    main()
