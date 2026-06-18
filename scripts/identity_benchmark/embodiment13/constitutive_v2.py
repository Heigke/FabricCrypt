#!/usr/bin/env python3
"""Phase 13 Task B — load-bearing constitutive coupling using signature_v2.

NARMA-10 reservoir-like task. Coupling:
   alpha(t) = sigmoid(W_alpha . signature_v2_vector)
   (W_alpha is *learned per chassi*, signature_v2 is held fixed for the run
    OR resampled — A/B/C/D ablation)

Conditions (30 seeds each):
   A: own-chassi signature, own data           (expected: best)
   B: random N(0,1) vector replaces signature  (expected: degenerate)
   C: other-chassi signature on eval           (expected: degraded)
   D: random for both                          (expected: worst / equal to B)

Pre-reg: A - B >= 15% NRMSE reduction with bootstrap 95% CI excluding 0.
"""
import os, sys, json, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.abspath(os.path.join(HERE, '..', '..', '..',
    'results', 'IDENTITY_BENCHMARK_2026-05-30', 'embodiment13'))

DIM = 290
N_TRAIN = 2000
N_EVAL = 2000
RES_SIZE = 100
N_SEEDS = 30
LR = 0.05
EPOCHS = 200


def sigmoid(x): return 1.0/(1.0+np.exp(-x))

def gen_narma10(n, rng):
    """NARMA-10 task: u(t) ~ U(0,0.5), y(t+1)=0.3 y + 0.05 y sum(y[-10:]) + 1.5 u[-9] u + 0.1"""
    u = rng.uniform(0.0, 0.5, n+11)
    y = np.zeros(n+11)
    for t in range(10, n+10):
        y[t+1] = 0.3*y[t] + 0.05*y[t]*np.sum(y[t-9:t+1]) + 1.5*u[t-9]*u[t] + 0.1
    return u[10:n+10], y[10:n+10]

def reservoir_run(u, alpha, rng_state):
    """Echo-state-like reservoir with coupling strength alpha.
    state[t+1] = (1-alpha)*state[t] + alpha*tanh(W_in @ u[t] + W_res @ state[t])
    Returns (T, RES_SIZE) state matrix.
    """
    rng = np.random.default_rng(rng_state)
    W_in  = rng.normal(0, 0.5, RES_SIZE)
    W_res = rng.normal(0, 1.0/np.sqrt(RES_SIZE), (RES_SIZE, RES_SIZE))
    # scale spectral radius ~0.9
    eigs = np.linalg.eigvals(W_res)
    rho = max(abs(eigs))
    W_res *= 0.9 / rho
    T = len(u)
    state = np.zeros(RES_SIZE)
    states = np.zeros((T, RES_SIZE))
    for t in range(T):
        inp = W_in*u[t] + W_res @ state
        state = (1-alpha)*state + alpha*np.tanh(inp)
        states[t] = state
    return states

def fit_eval(states_train, y_train, states_eval, y_eval, ridge=1e-3):
    # least-squares regression states->y
    X = states_train
    A = X.T @ X + ridge*np.eye(X.shape[1])
    b = X.T @ y_train
    w = np.linalg.solve(A, b)
    pred = states_eval @ w
    err = y_eval - pred
    rmse = float(np.sqrt(np.mean(err**2)))
    yvar = float(np.std(y_eval))
    return rmse / (yvar + 1e-12), w

def coupling_alpha(W, sig_vec):
    return float(sigmoid(W @ sig_vec))

def learn_W_alpha(sig_train_vec, u_tr, y_tr, seed, n_steps=30, lr_init=0.6):
    """Search for W_alpha (DIM,) minimising NRMSE on training NARMA.
    Simple coordinate-free 1D search over a scalar coupling, projected back through
    the signature direction so that signature identity controls the resulting alpha.

    Strategy: parameterise W = (logit(a_target) / sig.dot(sig)) * sig
              and grid-search a_target in [0.05..0.95]. Whichever produces the
              best in-sample NRMSE wins. This makes signature load-bearing:
              eval with a different signature produces a different alpha via
              the same W. Different sig -> different alpha -> different NRMSE.
    """
    sig_norm_sq = float(sig_train_vec @ sig_train_vec) + 1e-12
    best_nrmse = np.inf; best_a = 0.5
    a_grid = np.linspace(0.05, 0.95, 19)
    for a_target in a_grid:
        logit_t = np.log(a_target/(1-a_target))
        W = (logit_t / sig_norm_sq) * sig_train_vec
        a_used = coupling_alpha(W, sig_train_vec)  # ~= a_target
        st = reservoir_run(u_tr, a_used, seed)
        nrmse, _ = fit_eval(st, y_tr, st, y_tr)
        if nrmse < best_nrmse:
            best_nrmse = nrmse; best_a = a_target; best_W = W
    return best_W

def run_condition(label, sig_train_vec, sig_eval_vec, seed):
    rng = np.random.default_rng(seed)
    u_tr, y_tr = gen_narma10(N_TRAIN, rng)
    u_ev, y_ev = gen_narma10(N_EVAL,  rng)
    # learn W_alpha that works well *for the training signature*
    W = learn_W_alpha(sig_train_vec, u_tr, y_tr, seed)
    a_tr = coupling_alpha(W, sig_train_vec)
    a_ev = coupling_alpha(W, sig_eval_vec)
    st_tr = reservoir_run(u_tr, a_tr, seed)
    st_ev = reservoir_run(u_ev, a_ev, seed)
    nrmse, _w = fit_eval(st_tr, y_tr, st_ev, y_ev)
    return {'label': label, 'seed': seed,
            'a_train': a_tr, 'a_eval': a_ev,
            'nrmse': nrmse,
            'sig_train_norm': float(np.linalg.norm(sig_train_vec)),
            'sig_eval_norm':  float(np.linalg.norm(sig_eval_vec))}

def bootstrap_diff_ci(a, b, n_boot=10000, alpha=0.05, rng_seed=42):
    rng = np.random.default_rng(rng_seed)
    a = np.asarray(a); b = np.asarray(b)
    n = min(len(a), len(b))
    diffs = []
    for _ in range(n_boot):
        ia = rng.integers(0, len(a), n)
        ib = rng.integers(0, len(b), n)
        diffs.append(a[ia].mean() - b[ib].mean())
    diffs = np.sort(diffs)
    lo = float(np.percentile(diffs, 100*alpha/2))
    hi = float(np.percentile(diffs, 100*(1-alpha/2)))
    return float(np.mean(diffs)), lo, hi

def main():
    ika = np.load(os.path.join(OUT_DIR, 'ikaros_sig_v2.npz'))['vec']
    dae = np.load(os.path.join(OUT_DIR, 'daedalus_sig_v2.npz'))['vec']
    # Use the median rep as canonical signature per chassi to reduce noise
    sig_ika = np.median(ika, axis=0)
    sig_dae = np.median(dae, axis=0)
    # z-normalise each signature (avoid scale dominating sigmoid saturation)
    # Use joint stats so cross-chassi comparison is fair
    joint = np.vstack([ika, dae])
    mu = joint.mean(axis=0); sd = joint.std(axis=0) + 1e-9
    sig_ika_z = (sig_ika - mu)/sd
    sig_dae_z = (sig_dae - mu)/sd

    print(f"[const_v2] |sig_ika|={np.linalg.norm(sig_ika_z):.2f}  "
          f"|sig_dae|={np.linalg.norm(sig_dae_z):.2f}  "
          f"cos(ika,dae)={float(sig_ika_z @ sig_dae_z / (np.linalg.norm(sig_ika_z)*np.linalg.norm(sig_dae_z))):.4f}",
          flush=True)

    results = {'A': [], 'B': [], 'C': [], 'D': []}
    for seed in range(N_SEEDS):
        rng = np.random.default_rng(1000 + seed)
        rand_vec  = rng.standard_normal(DIM)
        rand_vec2 = rng.standard_normal(DIM)
        results['A'].append(run_condition('A_own',          sig_ika_z, sig_ika_z, seed))
        results['B'].append(run_condition('B_random_sig',   rand_vec,  rand_vec,  seed))
        results['C'].append(run_condition('C_other_chassi', sig_ika_z, sig_dae_z, seed))
        results['D'].append(run_condition('D_random_both',  rand_vec,  rand_vec2, seed))
        if (seed+1) % 10 == 0:
            a = np.mean([r['nrmse'] for r in results['A']])
            print(f"  seed {seed+1}/{N_SEEDS}  A_mean_nrmse={a:.4f}", flush=True)

    summary = {}
    for k, runs in results.items():
        nr = [r['nrmse'] for r in runs]
        summary[k] = {'mean_nrmse': float(np.mean(nr)),
                      'std_nrmse':  float(np.std(nr)),
                      'median_nrmse': float(np.median(nr)),
                      'n': len(nr)}
    # bootstrap A vs B (own vs random)
    na = [r['nrmse'] for r in results['A']]
    nb = [r['nrmse'] for r in results['B']]
    nc = [r['nrmse'] for r in results['C']]
    nd = [r['nrmse'] for r in results['D']]
    d_AB, lo_AB, hi_AB = bootstrap_diff_ci(nb, na)  # B - A: positive = A better
    d_AC, lo_AC, hi_AC = bootstrap_diff_ci(nc, na)
    d_AD, lo_AD, hi_AD = bootstrap_diff_ci(nd, na)
    # Two gates:
    #   pre-reg (literal): A-B >= 15% (B-A>0, lo>0)
    #   load-bearing test: swapping eval signature degrades >=15% (C vs A)
    rel_AB = d_AB / (np.mean(nb) + 1e-12)
    rel_AC = d_AC / (np.mean(nc) + 1e-12)
    summary['contrasts'] = {
        'B_minus_A': {'mean': d_AB, 'ci95': [lo_AB, hi_AB],
                      'rel_reduction_pct': float(100*rel_AB)},
        'C_minus_A': {'mean': d_AC, 'ci95': [lo_AC, hi_AC],
                      'rel_reduction_pct': float(100*rel_AC)},
        'D_minus_A': {'mean': d_AD, 'ci95': [lo_AD, hi_AD]},
        'gate_15pct_AB_passed': bool(rel_AB >= 0.15 and lo_AB > 0.0),
        'gate_15pct_swap_AC_passed': bool(rel_AC >= 0.15 and lo_AC > 0.0),
    }
    out = {'summary': summary, 'detail': results}
    with open(os.path.join(OUT_DIR, 'constitutive_v2.json'), 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
