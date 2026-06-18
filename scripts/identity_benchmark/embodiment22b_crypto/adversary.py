"""T2.5 — ML modeling attack on FabricCrypt (controlled vs uncontrolled).

Threat model (Rührmair et al. CCS'10):
   * Attacker observes N (nonce, response) pairs from a target chip.
   * Trains a model f: nonce_bits → response.
   * Forgery: for a FRESH challenge nonce*, predict response*, send.
   * Success if forged response is "close enough" to the real chip's
     response that the verifier accepts.

We run two attacks side-by-side:

  Attack-A (uncontrolled PUF):
      * Use raw Phase-14d paired sigs:  (nonce_8B, sig_64f32).
      * Adversary trains logistic-regression / small MLP / transformer on
        nonce → sig.
      * Forgery succeeds if Pearson(predicted, true) > 0.85 on held-out.

  Attack-B (controlled-PUF wrapped):
      * Same dataset, but each response is wrapped:
            wrapped = SHAKE256("ctrl-puf-out" || raw_response || nonce ||
                              chip_id, 32).
        Output is a 32-byte uniformly-random hash.
      * Adversary trains the same models on nonce → wrapped (bit-vector
        of 256 bits).
      * Forgery succeeds if predicted matches real to within fuzzy-extractor
        radius t (here t=24 of 256, ~10%).

Expected result: Attack-A succeeds dramatically; Attack-B fails (because
the wrapped output is a random oracle — no learnable structure).

We measure forgery RATE as a function of N_train ∈ {100, 500, 1000, 2000, 5000, 10000}.

Output: results/.../adversary_modeling_attack.json
"""
from __future__ import annotations
import os, sys, json, hashlib, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from controlled_puf import wrap_response  # noqa

REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
PAIRED_DIR = os.path.join(REPO, 'results/IDENTITY_BENCHMARK_2026-05-30/embodiment14d_crypto')
OUT_DIR = os.path.join(REPO, 'results/IDENTITY_BENCHMARK_2026-05-30/embodiment22b_crypto')
os.makedirs(OUT_DIR, exist_ok=True)


# ---------- data ----------
def load_pairs(host: str):
    d = np.load(os.path.join(PAIRED_DIR, f'{host}_paired_sigs.npz'))
    return d['nonces'], d['sigs'].astype(np.float32)


def nonce_bits(nonces: np.ndarray) -> np.ndarray:
    """uint8 (N, 8) → bits (N, 64) as float32."""
    bits = np.unpackbits(nonces, axis=1, bitorder='big')
    return bits.astype(np.float32)


def nonce_features(nonces: np.ndarray, dim: int = 64) -> np.ndarray:
    """Hash-expanded nonce features (mirrors NonceSigV2.nonce_embedding) +
    raw bits.  This gives the attacker the same input the classifier sees
    on the nonce_emb side.  Total dim = 64 + 64 = 128.
    """
    bits = nonce_bits(nonces)   # (N, 64)
    n = nonces.shape[0]
    emb = np.zeros((n, dim), dtype=np.float32)
    for i in range(n):
        block = hashlib.shake_256(b"FabricCrypt-v2-emb|" + nonces[i].tobytes()).digest(dim * 4)
        raw = np.frombuffer(block, dtype=np.uint32).astype(np.float64)
        v = (raw / 2**32) * 2 - 1
        emb[i] = v.astype(np.float32)
    return np.concatenate([bits, emb], axis=1)


def make_wrapped_targets(nonces: np.ndarray, sigs: np.ndarray,
                         host: str, K_chip: bytes,
                         out_bytes: int = 32) -> np.ndarray:
    """For Attack-B: compute the controlled-PUF wrapped output for each
    paired (nonce, raw_sig).  Returns float32 (N, out_bytes*8) bit array."""
    out = np.zeros((len(nonces), out_bytes * 8), dtype=np.float32)
    for i in range(len(nonces)):
        w = wrap_response(sigs[i], nonces[i].tobytes(), host, K_chip,
                          out_bytes=out_bytes)
        bits = np.unpackbits(np.frombuffer(w, dtype=np.uint8), bitorder='big')
        out[i] = bits.astype(np.float32)
    return out


# ---------- models ----------
def train_lr(X_train, Y_train, X_test, Y_test):
    """One linear regression per output dim.  (Avoids huge sklearn loops.)
    Y_test_pred = X_test @ W + b, where W is the OLS solution."""
    # Add bias
    Xtr = np.concatenate([X_train, np.ones((X_train.shape[0], 1))], axis=1)
    Xte = np.concatenate([X_test,  np.ones((X_test.shape[0], 1))], axis=1)
    # OLS: W = (Xt X)^-1 Xt Y  (regularized)
    XtX = Xtr.T @ Xtr + 1e-3 * np.eye(Xtr.shape[1])
    W = np.linalg.solve(XtX, Xtr.T @ Y_train)
    Y_pred = Xte @ W
    return Y_pred


def train_mlp(X_train, Y_train, X_test, Y_test, hidden=128, epochs=30, lr=3e-3):
    import torch
    import torch.nn as nn
    device = 'cpu'
    Xt = torch.from_numpy(X_train).float().to(device)
    Yt = torch.from_numpy(Y_train).float().to(device)
    Xv = torch.from_numpy(X_test).float().to(device)
    in_d = X_train.shape[1]; out_d = Y_train.shape[1]
    net = nn.Sequential(
        nn.Linear(in_d, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, out_d)
    ).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    bs = min(64, len(Xt))
    for ep in range(epochs):
        idx = torch.randperm(len(Xt))
        for s in range(0, len(Xt), bs):
            ix = idx[s:s+bs]
            yp = net(Xt[ix])
            loss = ((yp - Yt[ix]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        Yv_pred = net(Xv).cpu().numpy()
    return Yv_pred


# ---------- metrics ----------
def forgery_metric_continuous(Y_pred, Y_true):
    """For Attack-A (continuous 64-dim sig): per-sample Pearson r between
    predicted and true.  Forgery success = r > 0.85."""
    n = Y_pred.shape[0]
    rs = np.zeros(n)
    for i in range(n):
        a = Y_pred[i] - Y_pred[i].mean()
        b = Y_true[i] - Y_true[i].mean()
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
        rs[i] = float(np.dot(a, b) / denom)
    return rs


def forgery_metric_bits(Y_pred, Y_true, hamming_tol=24):
    """For Attack-B (256-bit wrapped output): threshold predicted at 0.5
    → bits → compare hamming distance to true.  Forgery success if
    hamming ≤ tol."""
    pred_bits = (Y_pred > 0.5).astype(np.uint8)
    true_bits = (Y_true > 0.5).astype(np.uint8)
    hams = (pred_bits != true_bits).sum(axis=1).astype(int)
    return hams


# ---------- experiment ----------
def run_attack(host='ikaros'):
    nonces, sigs = load_pairs(host)
    print(f"Loaded {host}: {nonces.shape[0]} pairs, sig dim {sigs.shape[1]}")

    X_all = nonce_features(nonces, dim=64)
    # Attack-A targets: raw sig vectors
    Y_A_all = sigs.copy()
    # Attack-B targets: wrapped controlled-PUF outputs (bits).  Use a
    # dummy K_chip (the chip's actual K_chip from Phase 14d; we don't
    # need the real one since the wrapping is deterministic given K_chip).
    K_chip = hashlib.sha256(b"adversary-test-K-" + host.encode()).digest()
    Y_B_all = make_wrapped_targets(nonces, sigs, host, K_chip, out_bytes=32)

    n_total = X_all.shape[0]
    # Held-out test set: 40 samples
    n_test = 40
    rng = np.random.default_rng(7)
    perm = rng.permutation(n_total)
    test_idx = perm[:n_test]; pool = perm[n_test:]

    X_test = X_all[test_idx]; YA_test = Y_A_all[test_idx]; YB_test = Y_B_all[test_idx]

    # Pool only has ~160 here; cap N_train at len(pool).
    N_grid = [50, 100, 150, len(pool)]   # phase 14d has only 200 pairs total
    results = []
    for N in N_grid:
        if N > len(pool): break
        tr_idx = pool[:N]
        X_train = X_all[tr_idx]; YA_train = Y_A_all[tr_idx]; YB_train = Y_B_all[tr_idx]

        # ----- Attack-A: continuous sig prediction -----
        for model_name, fn in [('linear', train_lr),
                               ('mlp', train_mlp)]:
            t0 = time.time()
            YA_pred = fn(X_train, YA_train, X_test, YA_test)
            rs = forgery_metric_continuous(YA_pred, YA_test)
            succ_85 = float(np.mean(rs > 0.85))
            succ_50 = float(np.mean(rs > 0.50))
            results.append(dict(
                attack='A_uncontrolled', model=model_name, N_train=int(N),
                pearson_mean=float(rs.mean()), pearson_med=float(np.median(rs)),
                forgery_rate_r085=succ_85, forgery_rate_r050=succ_50,
                t_sec=time.time() - t0,
            ))
            print(f"  A {model_name:6s}  N={N:4d}  r̄={rs.mean():.3f}  "
                  f"forge>0.85={succ_85:.2f}  forge>0.50={succ_50:.2f}")

        # ----- Attack-B: 256-bit wrapped prediction -----
        for model_name, fn in [('linear', train_lr),
                               ('mlp', train_mlp)]:
            t0 = time.time()
            YB_pred = fn(X_train, YB_train, X_test, YB_test)
            hams = forgery_metric_bits(YB_pred, YB_test, hamming_tol=24)
            succ_t24 = float(np.mean(hams <= 24))   # tight fuzzy radius
            succ_t48 = float(np.mean(hams <= 48))   # loose fuzzy radius
            results.append(dict(
                attack='B_controlled', model=model_name, N_train=int(N),
                hamming_mean=float(hams.mean()), hamming_min=int(hams.min()),
                forgery_rate_t24=succ_t24, forgery_rate_t48=succ_t48,
                t_sec=time.time() - t0,
            ))
            print(f"  B {model_name:6s}  N={N:4d}  ham̄={hams.mean():.1f}  "
                  f"forge≤24={succ_t24:.2f}  forge≤48={succ_t48:.2f}")

    return results


if __name__ == '__main__':
    out = dict()
    for host in ['ikaros', 'daedalus']:
        try:
            out[host] = run_attack(host)
        except Exception as e:
            out[host] = dict(error=str(e))
            print(f"  {host}: ERROR {e}")
    p = os.path.join(OUT_DIR, 'adversary_modeling_attack.json')
    json.dump(out, open(p, 'w'), indent=2)
    print(f"\nWrote {p}")
