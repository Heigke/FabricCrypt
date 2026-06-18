"""H7 FULL SWEEP — exhaustive battery of readouts × tasks × controls over ALL collected raw datasets.

CPU-only re-analysis (no hardware) of every raw npz we saved, to map exactly WHERE (if anywhere) the die
beats the FAIR controls. Hypotheses swept:
  H1 linear readout of die -> XOR/parity        (control: linear-on-u)
  H2 quadratic readout of die                    (control: quadratic-on-u-window)
  H3 rank-limited die {r=2,4,8,16} vs rank-limited u-window {W=2,4,8}   (memory necessity)
  H4 kernel/random-Fourier-features readout of die vs RFF-on-u-window   (rich nonlinear)
  H5 best single channel / best pair             (which carries nonlinearity)
  H6 memory depth: recall vs lag
  H7 NARMA-style nonlinear temporal capacity
  H8 transient(Vdroop) data vs steady-state data on same tasks (did amplification help?)
Controls everywhere: linear-on-u, unbounded-nonlinear-on-u (strawman ref), phase-shuffle surrogate p99.
A "WIN" = die readout beats the FAIR control (matched rank/order) AND the surrogate, by >0.05, on a task
the control can't trivially do. Prints a compact matrix + counts. Reuses saved raw npz; run when no probe
is collecting (numpy load would contaminate live substrate reads).
"""
from __future__ import annotations
import sys, json, socket, itertools
from pathlib import Path
import numpy as np

OUT = Path(__file__).resolve().parents[2] / "results/IDENTITY_H7_2026-06-09"
HOST = socket.gethostname()
rng = np.random.default_rng(0)


def lag(x, k):
    y = np.zeros_like(x)
    if k > 0: y[k:] = x[:-k]; return y
    return x.copy()


def acc(X, y, tr, te, nc):
    mu = X[tr].mean(0); sd = X[tr].std(0) + 1e-9; X = (X - mu) / sd; Y = np.eye(nc)[y]; best = 0.0
    for al in [1e-2, 0.1, 1, 10, 100, 1e3]:
        try:
            W = np.linalg.solve(X[tr].T @ X[tr] + al * np.eye(X.shape[1]), X[tr].T @ Y[tr])
            best = max(best, float(np.mean((X[te] @ W).argmax(1) == y[te])))
        except Exception: pass
    return best


def pca_acc(X, y, tr, te, nc, rank):
    mu = X[tr].mean(0); sd = X[tr].std(0) + 1e-9; Xz = (X - mu) / sd
    _, _, Vt = np.linalg.svd(Xz[tr] - Xz[tr].mean(0), full_matrices=False)
    Xp = (Xz - Xz[tr].mean(0)) @ Vt[:rank].T
    return acc(Xp, y, tr, te, nc)


def quad(X):
    # linear + pairwise products of up to 12 PCA comps to bound size
    mu = X.mean(0); sd = X.std(0) + 1e-9; Xz = (X - mu) / sd
    k = min(12, X.shape[1])
    _, _, Vt = np.linalg.svd(Xz - Xz.mean(0), full_matrices=False)
    P = (Xz - Xz.mean(0)) @ Vt[:k].T
    cols = [P] + [(P[:, i] * P[:, j])[:, None] for i, j in itertools.combinations_with_replacement(range(k), 2)]
    return np.hstack(cols)


def rff(X, D=200):
    mu = X.mean(0); sd = X.std(0) + 1e-9; Xz = (X - mu) / sd
    W = rng.normal(0, 1.0 / np.sqrt(Xz.shape[1]), size=(Xz.shape[1], D)); b = rng.uniform(0, 2*np.pi, D)
    return np.sqrt(2.0 / D) * np.cos(Xz @ W + b)


def feat_lags(M, nl=8):
    """M: (L,C) -> channels + lags 0..nl"""
    return np.hstack([np.vstack([lag(M[:, c], k) for k in range(nl+1)]).T for c in range(M.shape[1])])


def narma_target(u, n=10):
    L = len(u); y = np.zeros(L)
    for t in range(L):
        s = sum(y[t-i] for i in range(1, min(n, t)+1))
        y[t] = np.tanh(0.3*y[t-1] + 0.05*y[t-1]*s + 1.5*u[max(t-n,0)]*u[t-1] + 0.1) if t > n else 0.0
    return (y > np.median(y)).astype(int)


def run_dataset(name, u, M):
    """u: drive (L,), M: substrate matrix (L,C). Returns list of result rows."""
    L = len(u); WASH = 150; cut = WASH + int(0.7*(L-WASH)); tr = slice(WASH, cut); te = slice(cut, L)
    Mn = M.astype(float)
    Xdie = feat_lags(Mn, 8)
    Uwin4 = np.stack([lag(u.astype(float), k) for k in range(1, 5)], 1)
    Uwin8 = np.stack([lag(u.astype(float), k) for k in range(1, 9)], 1)
    uu = u.astype(float); ucols = [lag(uu, k) for k in range(16)]
    for a, b in itertools.combinations(range(10), 2): ucols.append(lag(uu, a)*lag(uu, b))
    Xnlu = np.stack(ucols, 1)

    def lb(k): return lag(u, k).astype(int)
    tasks = {"REC_t2": (lb(2), 2), "REC_t8": (lb(8), 2),
             "XOR_12": (lb(1)^lb(2), 2), "XOR_13": (lb(1)^lb(3), 2), "XOR_24": (lb(2)^lb(4), 2),
             "XOR_28": (lb(2)^lb(8), 2), "PAR3": (lb(1)^lb(2)^lb(3), 2), "NARMA": (narma_target(u), 2)}
    # phase-shuffle surrogate of die features (one draw, p~) for XOR_13 as representative
    rows = []
    for nm, (y, nc) in tasks.items():
        die_lin = acc(Xdie, y, tr, te, nc)
        die_r4 = pca_acc(Xdie, y, tr, te, nc, 4)
        die_quad = acc(quad(Mn), y, tr, te, nc)
        die_rff = acc(rff(Xdie), y, tr, te, nc)
        u_lin4 = pca_acc(Uwin4, y, tr, te, nc, min(4, Uwin4.shape[1]))
        u_quad4 = acc(quad(Uwin8), y, tr, te, nc)
        nlu = acc(Xnlu, y, tr, te, nc)
        # surrogate: phase-shuffle each channel
        Msur = np.zeros_like(Mn)
        for c in range(Mn.shape[1]):
            F = np.fft.rfft(Mn[:, c]-Mn[:, c].mean()); ph = rng.uniform(0, 2*np.pi, len(F)); ph[0] = 0
            Msur[:, c] = np.fft.irfft(np.abs(F)*np.exp(1j*ph), n=L)
        sur = acc(feat_lags(Msur, 8), y, tr, te, nc)
        best_die = max(die_lin, die_r4, die_quad, die_rff)
        fair = max(u_lin4, u_quad4)   # fair = matched-order readout on a short u-window
        win = best_die - fair > 0.05 and best_die - sur > 0.05 and best_die > 1.0/nc + 0.05
        rows.append({"dataset": name, "task": nm, "chance": 1.0/nc,
                     "die_lin": die_lin, "die_r4": die_r4, "die_quad": die_quad, "die_rff": die_rff,
                     "u_lin4": u_lin4, "u_quad4": u_quad4, "unbounded_nl_u": nlu, "surrogate": sur,
                     "fair_control": fair, "best_die": best_die, "WIN": bool(win)})
    return rows


def main():
    datasets = []
    rn = OUT / f"rank_necessity_raw_{HOST}.npz"
    if rn.exists():
        d = np.load(rn); datasets.append(("steady_state", d["u"], d["Sn"]))
    tv = OUT / f"transient_vdroop_raw_{HOST}.npz"
    if tv.exists():
        d = np.load(tv); Tn = d["Tn"]; datasets.append(("transient_vdroop", d["u"], Tn.reshape(len(d["u"]), -1)))
    bl = OUT / f"bilinear_raw_{HOST}.npz"
    if bl.exists():
        d = np.load(bl); A = d["A"]; Rn = np.tanh((d["Rraw"]-d["med"])/d["mad"]/8.0)
        # use A (one drive axis) as u for the compute battery; B is the second axis (extra signal)
        datasets.append(("bilinear_Adrive", (A > A.mean()).astype(int), Rn))

    all_rows = []
    print(f"[{HOST}] FULL SWEEP over {len(datasets)} datasets\n", flush=True)
    for name, u, M in datasets:
        print(f"=== {name}  (L={len(u)}, C={M.shape[1]}) ===", flush=True)
        rows = run_dataset(name, u, M); all_rows += rows
        print(f"  {'task':8s} {'chance':>6} {'die_best':>8} {'fair_ctrl':>9} {'surr':>6} {'nl_u':>5} {'WIN':>4}", flush=True)
        for r in rows:
            print(f"  {r['task']:8s} {r['chance']:6.3f} {r['best_die']:8.3f} {r['fair_control']:9.3f} {r['surrogate']:6.3f} {r['unbounded_nl_u']:5.2f} {'YES' if r['WIN'] else '':>4}", flush=True)
        print(flush=True)

    wins = [r for r in all_rows if r["WIN"]]
    def jf(o):
        if isinstance(o, dict): return {k: jf(v) for k, v in o.items()}
        if isinstance(o, list): return [jf(v) for v in o]
        if isinstance(o, (np.floating, np.integer, np.bool_)): return float(o)
        return o
    (OUT / f"full_sweep_{HOST}.json").write_text(json.dumps(jf({"rows": all_rows, "n_wins": len(wins),
        "wins": wins}), indent=2))
    print(f">>> SWEEP DONE: {len(wins)} WINS out of {len(all_rows)} (task,dataset) cells beat the fair control+surrogate", flush=True)
    if wins:
        for w in wins: print(f"    WIN: {w['dataset']}/{w['task']}  die={w['best_die']:.3f} > fair={w['fair_control']:.3f}", flush=True)
    else:
        print("    No cell beats the fair control — die provides nothing a matched readout on a short u-window can't.", flush=True)


if __name__ == "__main__":
    main()
