"""H7 Phase 0 analysis v2 — burst-aware, per-burst standardized, group-CV across bursts.

Why v2: recordings are CHUNKED thermal bursts. v1 stitched them into one series and split train/test
across different thermal regimes → high-dim ridge extrapolated → spurious negative R². v2 fixes this:
  - segment strictly by burst (no cross-burst lagging),
  - PER-BURST standardize features & targets → isolates FAST within-burst coupling and removes the
    slow thermal-integrator "loaded heater" confound automatically,
  - GroupKFold by burst (train on some bursts, test on held-out bursts) → measures generalization,
  - identical pipeline + identical ridge-λ search for the 30-D-vs-1-D comparison (fair).

Tasks:
 T1 DEFLECTION  Cohen's d per channel, C1/C2/C3 vs C0. Split FAST (power/clock/vcore) vs SLOW (temp).
 T2 REAFFERENCE predict next telemetry fluctuation from history; token-BLIND vs token-AWARE (+rate,ent).
 T3 DECODE/META telemetry -> LLM state (token rate, entropy); FULL 30-50-D vs 1-D power (Eric's Q).
 T4 CONTENT     C1 (content varies) vs C3 (random content, same compute): does entropy add over rate?

Out: phase0_analysis_{host}.json + printed verdict.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

def _ridge_fit(X, y, lam):
    d = X.shape[1]
    return np.linalg.solve(X.T@X + lam*np.eye(d), X.T@y)

def ridge_cv(X, y, groups, lams=(0.3, 1, 3, 10, 30, 100)):
    """GroupKFold by unique group; standardize X on train; pick best λ by mean test R²."""
    X = np.asarray(X, float); y = np.asarray(y, float)
    if y.ndim == 1: y = y[:, None]
    ug = np.unique(groups)
    if len(ug) < 2: return float("nan")
    k = min(5, len(ug))
    folds = [ug[i::k] for i in range(k)]
    best = -1e9
    for lam in lams:
        r2s = []
        for f in folds:
            te = np.isin(groups, f); tr = ~te
            if tr.sum() < 5 or te.sum() < 3: continue
            mu = X[tr].mean(0); sd = X[tr].std(0)+1e-9
            Xtr = (X[tr]-mu)/sd; Xte = (X[te]-mu)/sd
            ym = y[tr].mean(0); W = _ridge_fit(Xtr, y[tr]-ym, lam)
            pred = Xte@W + ym
            ss_res = ((y[te]-pred)**2).sum(0); ss_tot = ((y[te]-y[te].mean(0))**2).sum(0)
            ok = ss_tot > 1e-6 * max(1.0, float(np.max(ss_tot)))   # skip degenerate (constant) outputs
            if not np.any(ok): continue
            r2o = 1 - ss_res[ok]/ss_tot[ok]
            r2s.append(float(np.mean(r2o)))
        if r2s and np.mean(r2s) > best: best = float(np.mean(r2s))
    return best

def clean_channels(chans):
    cols = []; names = []
    T = next(iter(chans.values())).shape[0]
    for k, M in chans.items():
        if M.ndim != 2 or M.shape[1] == 0: continue
        for j in range(M.shape[1]):
            c = M[:, j].astype(float)
            if np.all(np.isnan(c)): continue
            med = np.nanmedian(c) if np.any(~np.isnan(c)) else 0.0
            c = np.nan_to_num(c, nan=med)
            if c.std() < 1e-12: continue
            cols.append(c); names.append(f"{k}[{j}]")
    return (np.stack(cols, 1) if cols else np.zeros((T, 0))), names

def is_slow(nm): return nm.startswith("gpu_temp") or nm.startswith("thermal")
def power_idx(names):
    for i, nm in enumerate(names):
        if nm.startswith("gpu_power"): return i
    for i, nm in enumerate(names):
        if "power" in nm: return i
    return 0

def token_activity(t_samp, ev_t, ev_val, win=0.4):
    rate = np.zeros(len(t_samp)); mval = np.zeros(len(t_samp))
    if ev_t is None or len(ev_t) == 0: return rate, mval
    for i, ts in enumerate(t_samp):
        m = (ev_t > ts-win) & (ev_t <= ts)
        rate[i] = m.sum()/win
        mval[i] = ev_val[m].mean() if (ev_val is not None and m.any()) else 0.0
    return rate, mval

def cohend(a, b):
    a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
    if len(a) < 3 or len(b) < 3: return 0.0
    s = np.sqrt(((len(a)-1)*a.var()+(len(b)-1)*b.var())/max(1, len(a)+len(b)-2))
    return float((a.mean()-b.mean())/(s+1e-12))

def per_burst_lagged(M, bursts, L, feat_cols, extra=None, target="self_next", tgt_col=None):
    """Build (X, y, group) with per-burst standardization and within-burst lagging.
    target: 'self_next' = predict standardized telemetry[feat_cols] at t+1 from history;
            'col' = predict standardized extra[:,tgt_col] at t from telemetry history."""
    Xs, ys, gs = [], [], []
    for bi, (i0, i1) in enumerate(bursts):
        seg = M[i0:i1]
        if seg.shape[0] < L+5: continue
        f = seg[:, feat_cols].astype(float)
        mu = f.mean(0); sd = f.std(0)+1e-9; fz = (f-mu)/sd
        # lagged history rows for t in [L, T)
        T = fz.shape[0]
        hist = np.concatenate([fz[L-1-i:T-1-i] for i in range(L)], axis=1)  # predict t>=L
        idx = np.arange(L, T)
        if extra is not None:
            ez = []
            for c in range(extra.shape[1]):
                e = extra[i0:i1, c].astype(float); em = e.mean(); es = e.std()+1e-9
                ez.append(((e-em)/es))
            ez = np.stack(ez, 1)
        if target == "self_next":
            y = fz[idx]                            # standardized next telemetry vector
            X = hist
        elif target == "self_next_aware":
            y = fz[idx]; X = np.concatenate([hist, ez[idx]], 1)
        elif target == "col":
            y = ez[idx, tgt_col:tgt_col+1]; X = hist
        elif target == "col_from_extra":          # baseline: predict tgt from rate history only
            rl = np.concatenate([ez[L-1-i:T-1-i, 0:1] for i in range(L)], axis=1)
            y = ez[idx, tgt_col:tgt_col+1]; X = rl
        else:
            raise ValueError(target)
        Xs.append(X); ys.append(y); gs.append(np.full(len(idx), bi))
    if not Xs: return None
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(gs)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True); ap.add_argument("--lag", type=int, default=6)
    a = ap.parse_args()
    z = np.load(a.npz, allow_pickle=True)
    host = str(z["host"]); t = z["mono_t"].astype(float)
    chans = {k[5:]: z[k] for k in z.files if k.startswith("chan_")}
    M, names = clean_channels(chans)
    marks = json.loads(str(z["marks"]))
    fast = [i for i, n in enumerate(names) if not is_slow(n)]
    pidx = power_idx(names)
    grp = sorted(set(n.split('[')[0] for n in names))
    print(f"[{host}] {len(t)} samp, {M.shape[1]} ch ({len(fast)} fast) groups={grp}", flush=True)

    def bursts_of(cn):
        return [(w["i0"], min(w["i1"], len(t))) for w in marks.get(cn, {}).get("windows", [])
                if min(w["i1"], len(t)) - w["i0"] > a.lag+5]
    def all_idx(cn):
        idx = []
        for i0, i1 in bursts_of(cn): idx.extend(range(i0, i1))
        return np.array(idx, int)

    res = {"host": host, "n_channels": int(M.shape[1]), "n_fast": len(fast),
           "channel_groups": grp, "lag": a.lag}

    # T1 deflection
    i0 = all_idx("C0_IDLE"); res["T1_deflection"] = {}
    for cn in ["C1_SELF", "C3_SWAP", "C2_YOKED"]:
        ic = all_idx(cn)
        if len(ic) < 5 or len(i0) < 5: continue
        ds = [(names[j], cohend(M[ic, j], M[i0, j])) for j in range(M.shape[1])]
        fast_d = [(n, d) for n, d in ds if not is_slow(n)]
        fast_d.sort(key=lambda x: -abs(x[1]))
        res["T1_deflection"][cn] = {
            "n_fast_moved_|d|>0.5": int(sum(abs(d) > 0.5 for _, d in fast_d)),
            "top5_fast": [[n, round(d, 2)] for n, d in fast_d[:5]],
            "power_d": round(cohend(M[ic, pidx], M[i0, pidx]), 2)}

    # token activity per condition (full M index space)
    def extra_for(cn):
        idx = all_idx(cn);
        et = z.get(f"ev_{cn}_t"); en = z.get(f"ev_{cn}_entropy")
        rate = np.zeros(len(t)); ent = np.zeros(len(t))
        r, e = token_activity(t[idx], et.astype(float) if et is not None else None,
                              en.astype(float) if en is not None else None)
        rate[idx] = r; ent[idx] = e
        return np.stack([rate, ent], 1)

    bC1 = bursts_of("C1_SELF"); exC1 = extra_for("C1_SELF")

    # T2 reafference: next telemetry (fast ch) blind vs aware
    if len(bC1) >= 2:
        blind = per_burst_lagged(M, bC1, a.lag, fast, extra=exC1, target="self_next")
        aware = per_burst_lagged(M, bC1, a.lag, fast, extra=exC1, target="self_next_aware")
        r2_blind = ridge_cv(*blind) if blind else float("nan")
        r2_aware = ridge_cv(*aware) if aware else float("nan")
        res["T2_reafference"] = {"r2_blind": round(r2_blind, 4), "r2_aware": round(r2_aware, 4),
                                 "delta_r2_token": round(r2_aware - r2_blind, 4)}

    # T3 decode/meta: telemetry history -> token rate & entropy; FULL vs 1-D power
    if len(bC1) >= 2:
        out = {}
        for tname, tcol in [("token_rate", 0), ("mean_entropy", 1)]:
            full = per_burst_lagged(M, bC1, a.lag, fast, extra=exC1, target="col", tgt_col=tcol)
            one = per_burst_lagged(M, bC1, a.lag, [pidx], extra=exC1, target="col", tgt_col=tcol)
            r_full = ridge_cv(*full) if full else float("nan")
            r_1d = ridge_cv(*one) if one else float("nan")
            out[tname] = {"r2_full": round(r_full, 4), "r2_power_1D": round(r_1d, 4),
                          "gain_full_over_1D": round(r_full - r_1d, 4)}
        res["T3_decode_meta"] = out

    # T4 content: C1 vs C3 power deflection + entropy-adds (rate vs rate+ent -> telem, within C1)
    if len(bC1) >= 2:
        rate_only = per_burst_lagged(M, bC1, a.lag, fast, extra=exC1[:, 0:1], target="self_next_aware")
        rate_ent = per_burst_lagged(M, bC1, a.lag, fast, extra=exC1, target="self_next_aware")
        r_ro = ridge_cv(*rate_only) if rate_only else float("nan")
        r_re = ridge_cv(*rate_ent) if rate_ent else float("nan")
        d = {"r2_+rate": round(r_ro, 4), "r2_+rate+entropy": round(r_re, 4),
             "entropy_adds": round(r_re - r_ro, 4)}
        i1c = all_idx("C1_SELF"); i3c = all_idx("C3_SWAP")
        if len(i1c) and len(i3c):
            d["C1_vs_C3_power_d"] = round(cohend(M[i1c, pidx], M[i3c, pidx]), 3)
        res["T4_content"] = d

    v = []
    t2 = res.get("T2_reafference", {}); t3 = res.get("T3_decode_meta", {})
    if t2: v.append(f"reafference: blind R²={t2['r2_blind']}, aware R²={t2['r2_aware']} (Δtoken {t2['delta_r2_token']:+})")
    if t3:
        tr = t3["token_rate"]
        v.append(f"meta-decode token_rate: FULL R²={tr['r2_full']} vs 1-D power R²={tr['r2_power_1D']} (gain {tr['gain_full_over_1D']:+})")
    res["verdict"] = v
    out = Path(a.npz).with_name(f"phase0_analysis_{host}.json")
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2)); print(f"\n[{host}] VERDICT:"); [print("  - "+x) for x in v]
    print(f"saved {out}")

if __name__ == "__main__":
    main()
