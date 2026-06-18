"""Phase 14D — re-run T2 (anomaly detection) and T3 (twin paradox) with the
ikaros sigs collected under a specified governor, against the existing daedalus
sigs from Phase 14B.

Usage: eval_t2_t3.py <ikaros_sigs_npz> <gov_label>

Saves: results/IDENTITY_BENCHMARK_2026-05-30/embodiment14d/t2t3_<gov>.json
"""
import os, sys, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, '..', '..', '..',
                                   'results', 'IDENTITY_BENCHMARK_2026-05-30',
                                   'embodiment14d'))
DAEDALUS_NPZ = os.path.abspath(os.path.join(HERE, '..', '..', '..',
                               'results', 'IDENTITY_BENCHMARK_2026-05-30',
                               'embodiment14b', 'daedalus_sigs.npz'))


def auroc(scores, y):
    """Mann-Whitney U based AUROC."""
    pos = scores[y == 1]; neg = scores[y == 0]
    if len(pos) == 0 or len(neg) == 0: return 0.5
    n = 0; m = 0
    for p in pos:
        for q in neg:
            if p > q: n += 1
            elif p == q: n += 0.5
            m += 1
    return n / m if m else 0.5


def set_seed(s):
    import random
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def run_T2(sigs_ikaros, n_seeds=5, epochs=15, batch=64):
    """Build anomaly-detection dataset from ikaros sigs at current governor.
    540 normal + 60 anomalies (synthetic +3.5σ shift on 4 channels)."""
    n = 600; n_anom = 60
    # Need at least n sigs; resample with replacement if not enough
    if len(sigs_ikaros) >= n:
        base = sigs_ikaros[:n].copy()
    else:
        idx = np.random.choice(len(sigs_ikaros), size=n, replace=True)
        base = sigs_ikaros[idx].copy()
    X = base.copy()
    y = np.zeros(n, dtype=np.int64)
    anom_idx = np.random.choice(n, size=n_anom, replace=False)
    for i in anom_idx:
        shift = np.zeros(32, dtype=np.float32)
        ch = np.random.choice(32, size=4, replace=False)
        shift[ch] = 3.5
        X[i] = np.clip(X[i] + shift, -4.0, 4.0)
        y[i] = 1
    idx = np.arange(n); np.random.shuffle(idx)
    split = int(0.7*n); tr, te = idx[:split], idx[split:]

    def train_eval(use_sig, seed):
        set_seed(seed)
        in_d = 32 if use_sig else 1
        model = nn.Sequential(nn.Linear(in_d, 64), nn.GELU(),
                              nn.Linear(64, 64), nn.GELU(),
                              nn.Linear(64, 2))
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
        for ep in range(epochs):
            order = np.random.permutation(len(tr))
            for i in range(0, len(order), batch):
                b = tr[order[i:i+batch]]
                xb = torch.from_numpy(X[b] if use_sig else np.zeros((len(b),1), dtype=np.float32))
                yb = torch.from_numpy(y[b])
                logits = model(xb); loss = F.cross_entropy(logits, yb)
                opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            xb = torch.from_numpy(X[te] if use_sig else np.zeros((len(te),1), dtype=np.float32))
            scores = F.softmax(model(xb), dim=-1)[:, 1].numpy()
        return auroc(scores, y[te])

    res = {'vanilla': [], 'embodied': []}
    for s in range(n_seeds):
        res['vanilla'].append(float(train_eval(False, s)))
        res['embodied'].append(float(train_eval(True, s)))
        print(f"  T2 seed={s} vanilla={res['vanilla'][-1]:.3f} embodied={res['embodied'][-1]:.3f}", flush=True)
    return {
        'vanilla_auroc_mean': float(np.mean(res['vanilla'])),
        'embodied_auroc_mean': float(np.mean(res['embodied'])),
        'vanilla_aurocs': res['vanilla'], 'embodied_aurocs': res['embodied'],
        'prereg_pass': float(np.mean(res['embodied'])) >= 0.85 and float(np.mean(res['vanilla'])) <= 0.6,
    }


def run_T3(sigs_ikaros, sigs_daedalus, n_seeds=5, epochs=20, batch=32):
    """Twin paradox — distinguish ikaros sigs (label 0) from daedalus (label 1)."""
    n_per = min(250, len(sigs_ikaros), len(sigs_daedalus))
    a = sigs_ikaros[:n_per]; b = sigs_daedalus[:n_per]
    X = np.concatenate([a, b], 0).astype(np.float32)
    y = np.concatenate([np.zeros(n_per), np.ones(n_per)], 0).astype(np.int64)
    idx = np.arange(len(X)); np.random.shuffle(idx)
    split = int(0.7*len(X)); tr, te = idx[:split], idx[split:]

    def train_eval(use_sig, seed):
        set_seed(seed)
        in_d = 32 if use_sig else 1
        model = nn.Sequential(nn.Linear(in_d, 64), nn.GELU(), nn.Linear(64, 2))
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
        for ep in range(epochs):
            order = np.random.permutation(len(tr))
            for i in range(0, len(order), batch):
                bb = tr[order[i:i+batch]]
                xb = torch.from_numpy(X[bb] if use_sig else np.zeros((len(bb),1), dtype=np.float32))
                yb = torch.from_numpy(y[bb])
                logits = model(xb); loss = F.cross_entropy(logits, yb)
                opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            xb = torch.from_numpy(X[te] if use_sig else np.zeros((len(te),1), dtype=np.float32))
            pred = model(xb).argmax(-1).numpy()
        return float((pred == y[te]).mean())

    res = {'vanilla': [], 'embodied': []}
    for s in range(n_seeds):
        res['vanilla'].append(train_eval(False, s))
        res['embodied'].append(train_eval(True, s))
        print(f"  T3 seed={s} vanilla={res['vanilla'][-1]:.3f} embodied={res['embodied'][-1]:.3f}", flush=True)
    return {
        'vanilla_acc_mean': float(np.mean(res['vanilla'])),
        'embodied_acc_mean': float(np.mean(res['embodied'])),
        'vanilla_accs': res['vanilla'], 'embodied_accs': res['embodied'],
        'n_per_class': n_per,
        'prereg_pass': float(np.mean(res['embodied'])) >= 0.95 and float(np.mean(res['vanilla'])) <= 0.55,
    }


def main():
    ika_npz = sys.argv[1]
    gov = sys.argv[2]
    print(f"[T2/T3] gov={gov} ika={ika_npz}")
    sigs_i = np.load(ika_npz)['sigs']
    sigs_d = np.load(DAEDALUS_NPZ)['sigs']
    print(f"[T2/T3] ikaros sigs N={len(sigs_i)} mean={sigs_i.mean():.3f} std={sigs_i.std():.3f}")
    print(f"[T2/T3] daedalus sigs N={len(sigs_d)} mean={sigs_d.mean():.3f} std={sigs_d.std():.3f}")
    out = {'gov': gov, 'ika_npz': ika_npz, 't_start': time.time()}
    print("=== T2 ==="); out['T2'] = run_T2(sigs_i)
    print("=== T3 ==="); out['T3'] = run_T3(sigs_i, sigs_d)
    out['t_end'] = time.time()
    path = os.path.join(OUT, f't2t3_{gov}.json')
    with open(path, 'w') as f: json.dump(out, f, indent=2)
    print(f"[save] {path}")


if __name__ == "__main__":
    main()
