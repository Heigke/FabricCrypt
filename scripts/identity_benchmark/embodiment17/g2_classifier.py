"""G2: Classifier — which chip wrote this?

Train logistic-regression on token-n-gram features per output. Per variant,
report chip-classification accuracy with bootstrap 95% CI.

Pre-reg gate:
  embodied >= 0.80, vanilla <= 0.55, synthetic <= 0.55
"""
from __future__ import annotations
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _common import RESULTS, save_json, bootstrap_ci


def featurize(token_ids, vocab_size=50257, max_unigram=2048):
    """Sparse unigram histogram, truncated to a band hash to keep dim small."""
    v = np.zeros(max_unigram, dtype=np.float32)
    for t in token_ids:
        v[t % max_unigram] += 1.0
    # length norm
    n = v.sum()
    if n > 0:
        v /= n
    return v


def load_outputs(chip, variant):
    p = os.path.join(RESULTS, f'{chip}_{variant}_outputs.json')
    with open(p) as f:
        return json.load(f)


def build_xy(variant, chips=('ikaros', 'daedalus')):
    X, y, prompt_ids = [], [], []
    for lbl, chip in enumerate(chips):
        d = load_outputs(chip, variant)
        for s in d['samples']:
            X.append(featurize(s['token_ids']))
            y.append(lbl)
            prompt_ids.append(s['prompt_idx'])
    return np.asarray(X), np.asarray(y), np.asarray(prompt_ids)


def kfold_logreg_acc(X, y, k=5, seed=0):
    """Train/test split by random k-fold; logistic regression with no extra deps.

    We implement a simple L2-regularized binary log-reg via numpy (avoid sklearn
    dependency assumption).
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    idx = rng.permutation(n)
    folds = np.array_split(idx, k)
    accs = []
    for i in range(k):
        test_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        Xtr, ytr = X[train_idx], y[train_idx]
        Xte, yte = X[test_idx], y[test_idx]
        w, b = _fit_logreg(Xtr, ytr, lam=1e-3, iters=200, lr=0.5)
        p = _sigmoid(Xte @ w + b)
        pred = (p > 0.5).astype(int)
        accs.append(float((pred == yte).mean()))
    return accs


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def _fit_logreg(X, y, lam=1e-3, iters=200, lr=0.5):
    n, d = X.shape
    w = np.zeros(d); b = 0.0
    for _ in range(iters):
        p = _sigmoid(X @ w + b)
        gw = X.T @ (p - y) / n + lam * w
        gb = float((p - y).mean())
        w -= lr * gw
        b -= lr * gb
    return w, b


def main():
    out = {'gate_embodied_min': 0.80, 'gate_baseline_max': 0.55, 'results': {}}
    for variant in ('vanilla', 'embodied', 'synthetic'):
        X, y, _ = build_xy(variant)
        accs = kfold_logreg_acc(X, y, k=5)
        mean, lo, hi = bootstrap_ci(accs, n_boot=1000)
        out['results'][variant] = {
            'fold_accs': accs, 'mean': mean, 'ci95': [lo, hi],
            'n_samples': int(len(y))}
        print(f"[G2] {variant:10s}  acc={mean:.3f}  CI=[{lo:.3f},{hi:.3f}]  n={len(y)}",
              flush=True)

    emb = out['results']['embodied']['mean']
    van = out['results']['vanilla']['mean']
    syn = out['results']['synthetic']['mean']
    out['pass_embodied'] = emb >= 0.80
    out['pass_vanilla'] = van <= 0.55
    out['pass_synthetic'] = syn <= 0.55
    out['gate_all'] = out['pass_embodied'] and out['pass_vanilla'] and out['pass_synthetic']
    out['embodied_minus_synthetic'] = float(emb - syn)
    print(f"[G2] PASS embodied={out['pass_embodied']} vanilla_low={out['pass_vanilla']} "
          f"synth_low={out['pass_synthetic']}  ALL={out['gate_all']}", flush=True)
    save_json('g2_classifier.json', out)


if __name__ == '__main__':
    main()
