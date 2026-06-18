"""H7 Fas 1 / A2 — reafference forward-model on EXISTING Phase-0 data.

Question (AE-2 contingency): does the LLM's own ACTION carry predictive information about the body's
FUTURE telemetry, beyond what the body's own recent telemetry already predicts? If yes, the
output->input contingency is *learnable* => the reafferent loop the embodiment claim needs exists.

Honest reframing (from Phase 0): the loop is INTENSITY-mediated, so the action feature is the LLM's
local token-RATE (compute intensity), not token identity. We predict a coarse next-window telemetry
summary at a thermal/electrical-matched horizon.

Design:
  - Restrict to the C1_SELF (real generation) telemetry span.
  - Bin time into windows of --win s. Per window: telemetry summary (mean of each fast channel) and
    token_rate = (#emitted tokens in window)/win.
  - BLIND model: predict next-window telemetry-summary from k windows of telemetry history only.
  - AWARE model: same + k windows of token_rate history (the action).
  - delta_R2 = aware - blind, group-CV across BURSTS (time gaps in mono_t define burst groups).
  - Controls: C2_YOKED span (non-LLM matmul) should NOT show a token_rate gain over its own drive
    proxy; shuffle null collapses.

Out: reafference_fwd_{host}.json + printed verdict.
Run:  python h7_reafference_fwd.py --npz results/IDENTITY_H7_2026-06-16/phase0_ikaros.npz --win 0.2
"""
from __future__ import annotations
import argparse, json
import numpy as np
from pathlib import Path

FAST = ["chan_gpu_power", "chan_gpu_freq", "chan_vcore", "chan_cur_freq"]  # fast electrical channels

def burst_groups(t, gap=0.5):
    g = np.zeros(len(t), int)
    cur = 0
    for i in range(1, len(t)):
        if t[i] - t[i-1] > gap: cur += 1
        g[i] = cur
    return g

def build_windows(t, telem, grp, ev_t, win):
    """Bin [t0,t1] into win-second windows within each burst; return per-window features."""
    rows = []
    for b in np.unique(grp):
        m = grp == b
        tb = t[m]; Xb = telem[m]
        if len(tb) < 5: continue
        t0, t1 = tb[0], tb[-1]
        edges = np.arange(t0, t1, win)
        for k in range(len(edges)-1):
            lo, hi = edges[k], edges[k+1]
            sel = (tb >= lo) & (tb < hi)
            if sel.sum() < 2: continue
            tel = Xb[sel].mean(0)
            rate = float(((ev_t >= lo) & (ev_t < hi)).sum()) / win
            rows.append((b, lo, tel, rate))
    return rows

def design(rows, k):
    """Per-burst lagged design: predict telem[w] from telem[w-1..w-k] (+ optional rate[w-1..w-k])."""
    by_b = {}
    for b, lo, tel, rate in rows: by_b.setdefault(b, []).append((lo, tel, rate))
    Yt, Xtel, Xrate, G = [], [], [], []
    for b, seq in by_b.items():
        seq.sort(key=lambda r: r[0])
        T = [s[1] for s in seq]; R = [s[2] for s in seq]
        for w in range(k, len(seq)):
            Yt.append(T[w])
            Xtel.append(np.concatenate([T[w-j] for j in range(1, k+1)]))
            Xrate.append(np.array([R[w-j] for j in range(1, k+1)]))
            G.append(b)
    if not Yt: return None
    return np.array(Yt), np.array(Xtel), np.array(Xrate), np.array(G)

def group_cv_r2(X, Y, G, lam=10.0):
    """Multi-output ridge, leave-one-burst-out; return mean over outputs of pooled R2 (masked to non-degenerate)."""
    groups = np.unique(G)
    if len(groups) < 3: return np.nan
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-8)
    Xs = np.column_stack([Xs, np.ones(len(Xs))])
    P = np.full_like(Y, np.nan, dtype=float)
    for go in groups:
        te = G == go; tr = ~te
        if tr.sum() < 8 or te.sum() < 1: continue
        A = Xs[tr]; W = np.linalg.solve(A.T @ A + lam*np.eye(A.shape[1]), A.T @ Y[tr])
        P[te] = Xs[te] @ W
    m = ~np.isnan(P).any(1)
    if m.sum() < 5: return np.nan
    r2s = []
    Ym, Pm = Y[m], P[m]
    for j in range(Y.shape[1]):
        sst = ((Ym[:, j] - Ym[:, j].mean())**2).sum()
        if sst < 1e-9: continue
        ssr = ((Ym[:, j] - Pm[:, j])**2).sum()
        r2s.append(1 - ssr/sst)
    return float(np.mean(r2s)) if r2s else np.nan

def run_span(t, telem, ev_t, win, k):
    grp = burst_groups(t)
    rows = build_windows(t, telem, grp, ev_t, win)
    d = design(rows, k)
    if d is None: return None
    Y, Xtel, Xrate, G = d
    blind = group_cv_r2(Xtel, Y, G)
    aware = group_cv_r2(np.column_stack([Xtel, Xrate]), Y, G)
    # shuffle null: permute rate rows
    rng = np.random.default_rng(0)
    Xr_s = Xrate[rng.permutation(len(Xrate))]
    null = group_cv_r2(np.column_stack([Xtel, Xr_s]), Y, G)
    return dict(n_windows=len(Y), n_bursts=int(len(np.unique(G))),
                blind_r2=round(blind, 4), aware_r2=round(aware, 4),
                delta_r2=round(aware-blind, 4) if not (np.isnan(aware) or np.isnan(blind)) else None,
                shuffle_null_r2=round(null, 4))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--win", type=float, default=0.2)
    ap.add_argument("--k", type=int, default=4)
    a = ap.parse_args()
    z = np.load(a.npz, allow_pickle=True)
    host = str(z["host"]); t = z["mono_t"].astype(float)
    telem = np.concatenate([z[c] for c in FAST if c in z], axis=1)
    # standardize columns globally first (per-burst handled in CV via masking; keep scale sane)
    res = {"host": host, "win": a.win, "k": a.k}
    # C1_SELF span
    c1 = z["ev_C1_SELF_t"].astype(float)
    m1 = (t >= c1.min()) & (t <= c1.max())
    res["C1_SELF"] = run_span(t[m1], telem[m1], c1, a.win, a.k)
    # C2_YOKED control span (no LLM; "rate" = telemetry-sample density proxy ~ constant -> expect ~0 gain)
    if "ev_C2_YOKED_t" in z:
        c2 = z["ev_C2_YOKED_t"].astype(float)
        m2 = (t >= c2.min()) & (t <= c2.max())
        res["C2_YOKED"] = run_span(t[m2], telem[m2], c2, a.win, a.k)
    out = Path(a.npz).with_name(f"reafference_fwd_{host}.json")
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    c1r = res.get("C1_SELF") or {}
    print(f"\n[{host}] A2 VERDICT (win={a.win}s k={a.k}):")
    print(f"  C1_SELF: blind R²={c1r.get('blind_r2')} aware R²={c1r.get('aware_r2')} "
          f"Δ(action)={c1r.get('delta_r2')} null={c1r.get('shuffle_null_r2')}")
    print(f"  -> reafference learnable if Δ>0 and ≫ null. saved {out}")

if __name__ == "__main__":
    main()
