"""Cross-chassis leave-one-out classification.

Takes 2+ <host>_sig_v2.npz files (each contains a (R, 290) `vec` matrix),
runs leave-one-out logistic regression, prints accuracy + per-fold preds.

Gate from paper: LOO accuracy > 0.95 with N >= 2 hosts, R >= 10 reps each.
"""
from __future__ import annotations
import os
import sys
import json
import argparse
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sigs", nargs="+", help="paths to <host>_sig_v2.npz")
    ap.add_argument("--out_json", default=None)
    args = ap.parse_args()

    if len(args.sigs) < 2:
        print("Need >=2 signature files for cross-chassis LOO.", file=sys.stderr)
        sys.exit(2)

    Xs, ys, names = [], [], []
    for label_idx, path in enumerate(args.sigs):
        d = np.load(path)
        vec = d["vec"]
        host = str(d["host"]) if "host" in d else os.path.basename(path).split("_")[0]
        names.append(host)
        Xs.append(vec)
        ys.append(np.full(len(vec), label_idx))
        print(f"  loaded {host}: {vec.shape}")

    X = np.vstack(Xs)
    y = np.concatenate(ys)
    mu = X.mean(axis=0); sd = X.std(axis=0) + 1e-9
    Xz = (X - mu) / sd

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import LeaveOneOut
    loo = LeaveOneOut()
    correct, preds = 0, []
    for tr, te in loo.split(Xz):
        clf = LogisticRegression(C=1.0, max_iter=2000)
        clf.fit(Xz[tr], y[tr])
        p = int(clf.predict(Xz[te])[0])
        preds.append(p)
        correct += int(p == y[te[0]])
    acc = correct / len(y)
    out = {
        "n_hosts": len(args.sigs),
        "host_labels": names,
        "n_total": int(len(y)),
        "loo_acc": float(acc),
        "gate_gt_0_95_passed": bool(acc > 0.95),
        "preds": preds,
        "truth": y.tolist(),
    }
    print(json.dumps({k: v for k, v in out.items()
                      if k not in ("preds", "truth")}, indent=2))
    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"saved {args.out_json}")


if __name__ == "__main__":
    main()
