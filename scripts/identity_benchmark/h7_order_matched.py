"""H7 ORDER-MATCHED necessity — did we think wrong about WHICH bar is fair?

The full_sweep killed the die by allowing a QUADRATIC on u (XOR = product of 2 bits = trivially quadratic).
But the PAR3 residue (die beats pairwise-nl-u, which is only degree-2) hinted the die's real value is
HIGH-ORDER nonlinearity for free: a degree-k polynomial on a W-window needs O(W^k) features, while the die's
recurrent nonlinearity might deliver order-k products at FIXED rank. So the honest question is not "linear vs
quadratic" but: AT A MATCHED READOUT BUDGET (same #features = rank r), does the die reach a higher
computational ORDER than a polynomial-on-u of the same budget?

Test: parity tasks of increasing order k=2..5 (PARk = XOR of lags 1..k). For each:
  - die: rank-r PCA-linear of die features (channels x lags 0..8), r in {4,8,16,32,64}
  - u-control at MATCHED rank r: rank-r PCA-linear of ALL monomials of u-window (lags1..8) up to degree D,
    swept D=1,2,3 (linear / quadratic / cubic). This is the FAIR same-budget competitor.
  - u-poly FULL (all degree-D monomials, no rank cap) = ceiling reference (what unlimited u-readout can do).
  - surrogate: phase-shuffle die features.
A die "ESCAPE" at order k = die-rank-r beats EVERY matched-rank u-poly (all D up to 3) AND surrogate by >0.05.
That would mean the die packs order-k computation into rank r that no same-budget u-polynomial matches.
CPU-only; uses saved steady_state + transient_vdroop npz. Run when no probe is collecting.
"""
from __future__ import annotations
import json, socket, itertools
from pathlib import Path
import numpy as np

OUT = Path(__file__).resolve().parents[2] / "results/IDENTITY_H7_2026-06-09"
HOST = socket.gethostname()
rng = np.random.default_rng(0)
RANKS = [4, 8, 16, 32, 64]
W = 8            # u-window length
ORDERS = [2, 3, 4, 5]


def lag(x, k):
    y = np.zeros_like(x)
    if k > 0: y[k:] = x[:-k]
    return y if k > 0 else x.copy()


def ridge_acc(X, y, tr, te, nc):
    mu = X[tr].mean(0); sd = X[tr].std(0) + 1e-9; X = (X - mu) / sd
    Y = np.eye(nc)[y]; best = 0.0
    for al in [1e-2, 0.1, 1, 10, 100, 1e3]:
        try:
            Wt = np.linalg.solve(X[tr].T @ X[tr] + al*np.eye(X.shape[1]), X[tr].T @ Y[tr])
            best = max(best, float(np.mean((X[te] @ Wt).argmax(1) == y[te])))
        except Exception: pass
    return best


def rank_pca(X, tr, rank):
    mu = X[tr].mean(0); sd = X[tr].std(0) + 1e-9; Xz = (X - mu)/sd
    c = Xz[tr].mean(0)
    _, _, Vt = np.linalg.svd(Xz[tr] - c, full_matrices=False)
    r = min(rank, Vt.shape[0])
    return (Xz - c) @ Vt[:r].T


def die_features(M):
    return np.hstack([np.vstack([lag(M[:, c], k) for k in range(9)]).T for c in range(M.shape[1])])


def u_monomials(u, W, D):
    """all monomials of {lag(u,1..W)} up to total degree D (binary u -> products of distinct lags + self=lag)."""
    base = [lag(u, k) for k in range(1, W+1)]
    cols = list(base)
    if D >= 2:
        for i, j in itertools.combinations_with_replacement(range(W), 2):
            cols.append(base[i]*base[j])
    if D >= 3:
        for i, j, k in itertools.combinations_with_replacement(range(W), 3):
            cols.append(base[i]*base[j]*base[k])
    return np.stack(cols, 1)


def run(name, u, M):
    L = len(u); WASH = 150; cut = WASH + int(0.7*(L-WASH)); tr = slice(WASH, cut); te = slice(cut, L)
    Df = die_features(M.astype(float))
    umono = {D: u_monomials(u.astype(float), W, D) for D in (1, 2, 3)}
    # surrogate die
    Msur = np.zeros_like(M, float)
    for c in range(M.shape[1]):
        F = np.fft.rfft(M[:, c].astype(float) - M[:, c].mean()); ph = rng.uniform(0, 2*np.pi, len(F)); ph[0] = 0
        Msur[:, c] = np.fft.irfft(np.abs(F)*np.exp(1j*ph), n=L)
    Sf = die_features(Msur)

    rows = []
    for k in ORDERS:
        y = lag(u, 1).astype(int)
        for j in range(2, k+1): y = y ^ lag(u, j).astype(int)
        nc = 2
        # die at each rank
        die_by_rank = {r: ridge_acc(rank_pca(Df, tr, r), y, tr, te, nc) for r in RANKS}
        sur_by_rank = {r: ridge_acc(rank_pca(Sf, tr, r), y, tr, te, nc) for r in RANKS}
        # u-poly at matched rank, each degree
        upoly_matched = {(D, r): ridge_acc(rank_pca(umono[D], tr, r), y, tr, te, nc)
                         for D in (1, 2, 3) for r in RANKS}
        upoly_full = {D: ridge_acc(umono[D], y, tr, te, nc) for D in (1, 2, 3)}
        # ESCAPE per rank: die(r) beats every matched u-poly(D,r) AND surrogate(r) by >0.05
        escape_ranks = []
        for r in RANKS:
            matched_best = max(upoly_matched[(D, r)] for D in (1, 2, 3))
            if die_by_rank[r] - matched_best > 0.05 and die_by_rank[r] - sur_by_rank[r] > 0.05 and die_by_rank[r] > 0.55:
                escape_ranks.append(r)
        rows.append({"dataset": name, "order": k,
                     "die_by_rank": die_by_rank, "sur_by_rank": sur_by_rank,
                     "upoly_matched": {f"D{D}_r{r}": upoly_matched[(D, r)] for D in (1,2,3) for r in RANKS},
                     "upoly_full": {f"D{D}": upoly_full[D] for D in (1,2,3)},
                     "escape_ranks": escape_ranks})
    return rows


def main():
    dsets = []
    for fn, nm, key in [("rank_necessity_raw", "steady_state", "Sn"),
                        ("transient_vdroop_raw", "transient_vdroop", "Tn")]:
        p = OUT / f"{fn}_{HOST}.npz"
        if p.exists():
            d = np.load(p); M = d[key]; M = M.reshape(len(d["u"]), -1) if M.ndim > 2 else M
            dsets.append((nm, d["u"], M))
    allrows = []
    for nm, u, M in dsets:
        print(f"\n=== {nm}  L={len(u)} C={M.shape[1]} ===", flush=True)
        rows = run(nm, u, M); allrows += rows
        for r in rows:
            db = r["die_by_rank"]; uf = r["upoly_full"]
            best_die = max(db.values())
            print(f"  PAR{r['order']}: die(best over rank)={best_die:.3f}  "
                  f"u_full[D1/D2/D3]={uf['D1']:.2f}/{uf['D2']:.2f}/{uf['D3']:.2f}  "
                  f"surr={max(r['sur_by_rank'].values()):.2f}  escape_ranks={r['escape_ranks']}", flush=True)
    def jf(o):
        if isinstance(o, dict): return {k: jf(v) for k, v in o.items()}
        if isinstance(o, list): return [jf(v) for v in o]
        if isinstance(o, (np.floating, np.integer, np.bool_)): return float(o)
        return o
    (OUT / f"order_matched_{HOST}.json").write_text(json.dumps(jf(allrows), indent=2))
    esc = [(r["dataset"], r["order"], r["escape_ranks"]) for r in allrows if r["escape_ranks"]]
    print(f"\n>>> ORDER-MATCHED DONE. Die ESCAPES (beats same-budget u-poly+surrogate) at: {esc if esc else 'NOWHERE'}", flush=True)


if __name__ == "__main__":
    main()
