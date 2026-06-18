#!/usr/bin/env python3
"""Phase 13 Task C — per-chassi-bound capability via signature-as-key.

Tiny MLP on a synthetic 2-class task whose W_in layer is *derived from* a
small MLP applied to the signature vector. The "key" is the signature.

Scenarios:
  own        : train on ikaros, evaluate on ikaros (signature live)        -> high acc
  transplant : take trained weights to daedalus, run with daedalus sig     -> low acc
  spoof      : on daedalus, lie and provide stored ikaros signature        -> high acc

Pre-reg: own > 0.90, transplant < 0.30, spoof_with_stored_sig > 0.80
"""
import os, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.abspath(os.path.join(HERE, '..', '..', '..',
    'results', 'IDENTITY_BENCHMARK_2026-05-30', 'embodiment13'))

DIM = 290
HIDDEN = 32
INPUT_FEATS = 16    # task input dimension
N_CLASS = 2
N_TRAIN = 2000
N_EVAL = 1000
EPOCHS = 1000
LR = 0.1
SEEDS = 10

def sigmoid(x): return 1.0/(1.0+np.exp(-np.clip(x,-30,30)))
def softmax(x):
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x); return e / e.sum(axis=-1, keepdims=True)

def make_task(n, rng):
    """Linearly separable: class is sign of w_true . X for fixed w_true."""
    # Use a seed-independent w_true so both train+eval share the boundary
    rng_w = np.random.default_rng(20260601)
    w_true = rng_w.standard_normal(INPUT_FEATS)
    X = rng.standard_normal((n, INPUT_FEATS))
    y = (X @ w_true > 0).astype(np.int64)
    return X, y

def sig_to_Win(sig_vec, W_keyA, W_keyB):
    """Two-layer MLP: sig -> hidden -> Win_flat (INPUT_FEATS*HIDDEN). Then reshape."""
    h = np.tanh(W_keyA @ sig_vec)            # (HIDDEN,)
    flat = W_keyB @ h                        # (INPUT_FEATS*HIDDEN,)
    Win = flat.reshape(INPUT_FEATS, HIDDEN)
    return Win

def forward(X, Win, W_out, b_out):
    h = np.tanh(X @ Win)
    logits = h @ W_out + b_out
    return softmax(logits), h

def train_classifier(sig_vec, seed):
    rng = np.random.default_rng(seed)
    X_tr, y_tr = make_task(N_TRAIN, rng)
    X_ev, y_ev = make_task(N_EVAL,  rng)
    # Keys: fixed *per training run* — saved together with classifier head.
    W_keyA = rng.standard_normal((HIDDEN, DIM))         / np.sqrt(DIM)
    W_keyB = rng.standard_normal((INPUT_FEATS*HIDDEN, HIDDEN)) / np.sqrt(HIDDEN)
    Win = sig_to_Win(sig_vec, W_keyA, W_keyB)
    # Learnable head only (W_out, b_out)
    W_out = rng.standard_normal((HIDDEN, N_CLASS)) * 0.1
    b_out = np.zeros(N_CLASS)
    for ep in range(EPOCHS):
        probs, h = forward(X_tr, Win, W_out, b_out)
        # CE grad w.r.t. logits = probs - onehot
        Y = np.zeros_like(probs); Y[np.arange(N_TRAIN), y_tr] = 1
        d_logits = (probs - Y) / N_TRAIN
        dW_out = h.T @ d_logits
        db_out = d_logits.sum(axis=0)
        W_out -= LR * dW_out
        b_out -= LR * db_out
    probs_ev, _ = forward(X_ev, Win, W_out, b_out)
    acc = float((probs_ev.argmax(axis=1) == y_ev).mean())
    return {'acc': acc, 'W_keyA': W_keyA, 'W_keyB': W_keyB,
            'W_out': W_out, 'b_out': b_out, 'X_ev': X_ev, 'y_ev': y_ev}

def eval_with_sig(trained, sig_vec):
    Win = sig_to_Win(sig_vec, trained['W_keyA'], trained['W_keyB'])
    probs, _ = forward(trained['X_ev'], Win, trained['W_out'], trained['b_out'])
    return float((probs.argmax(axis=1) == trained['y_ev']).mean())

def main():
    ika = np.load(os.path.join(OUT_DIR, 'ikaros_sig_v2.npz'))['vec']
    dae = np.load(os.path.join(OUT_DIR, 'daedalus_sig_v2.npz'))['vec']
    joint = np.vstack([ika, dae])
    mu = joint.mean(axis=0); sd = joint.std(axis=0) + 1e-9
    ika_z = (ika - mu)/sd; dae_z = (dae - mu)/sd
    sig_ika_canonical = np.median(ika_z, axis=0)
    sig_dae_canonical = np.median(dae_z, axis=0)

    accs = {'own': [], 'transplant': [], 'spoof_stored': [], 'spoof_random': []}
    for s in range(SEEDS):
        # Train on ikaros canonical sig
        trained = train_classifier(sig_ika_canonical, seed=100+s)
        # own: another live capture of ikaros (use median of remaining reps)
        live_ika = np.median(ika_z[s % len(ika_z):s % len(ika_z)+1], axis=0)
        accs['own'].append(eval_with_sig(trained, live_ika))
        # transplant: live daedalus sig (single rep)
        live_dae = dae_z[s % len(dae_z)]
        accs['transplant'].append(eval_with_sig(trained, live_dae))
        # spoof: pretend we are ikaros by feeding stored ikaros sig
        accs['spoof_stored'].append(eval_with_sig(trained, sig_ika_canonical))
        # control: random N(0,1) vector
        rng = np.random.default_rng(200+s)
        accs['spoof_random'].append(eval_with_sig(trained, rng.standard_normal(DIM)))
    summary = {k: {'mean': float(np.mean(v)), 'std': float(np.std(v)), 'min': float(np.min(v)),
                   'max': float(np.max(v)), 'n': len(v)}
               for k,v in accs.items()}
    summary['gate_own_gt_0_90']        = bool(np.mean(accs['own']) > 0.90)
    summary['gate_transplant_lt_0_30'] = bool(np.mean(accs['transplant']) < 0.30)
    summary['gate_spoof_gt_0_80']      = bool(np.mean(accs['spoof_stored']) > 0.80)
    summary['gate_all_passed']         = bool(summary['gate_own_gt_0_90'] and
                                              summary['gate_transplant_lt_0_30'] and
                                              summary['gate_spoof_gt_0_80'])
    with open(os.path.join(OUT_DIR, 'chassi_lock.json'), 'w') as f:
        json.dump({'summary': summary, 'detail': accs}, f, indent=2, default=str)
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
