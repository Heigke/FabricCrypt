"""H7 MIXING VERIFY — is the within-die XOR(u,v) gate-pass REAL or a fluke? CPU-only, on saved raw npz.

The gate said die(full)-linear does XOR(u,v)=0.654 while u-only=0.53. Since XOR is NOT linearly separable in
(u,v), a LINEAR readout scoring >chance can only exploit a physical u*v PRODUCT term. This script stress-tests
that claim three ways, no new hardware:
  1. NULL DISTRIBUTION: 300 phase-shuffle surrogates of the die features -> empirical p-value for XOR(u,v)
     accuracy and a 95th-percentile null threshold (replaces the single flaky surrogate draw).
  2. PHYSICAL u*v COEFFICIENT: regress each telemetry channel on centered [u, v, u*v]; report the partial R^2
     gain of adding u*v over [u,v] alone (>0 only if the silicon physically multiplies). This is the mechanism.
  3. SHARPEST NECESSITY CONTROL: a readout given BOTH u and v but only LINEARLY (lags of u and v, no product).
     XOR needs the product, so this control should fail. If die(full) beats BOTH (a) u-only and (b) u&v-linear,
     the die is genuinely supplying the product term -> non-circular mixing necessity (modulo v being secret).
  4. BOOTSTRAP CI on die XOR accuracy (1000 resamples of the test block) -> is 0.654 robustly above chance?
Pre-registered REAL = die_full_XOR > null_p95 (p<0.05) AND die_full_XOR > u_and_v_linear + 0.05 AND uv_partialR2
median across channels > 0. Writes mixing_verify_{HOST}.json.
"""
from __future__ import annotations
import json, socket, itertools
from pathlib import Path
import numpy as np

OUT = Path(__file__).resolve().parents[2] / "results/IDENTITY_H7_2026-06-09"
HOST = socket.gethostname()
WASHOUT = 150
rng = np.random.default_rng(0)


def lag(x, k):
    y = np.zeros_like(x)
    if k > 0: y[k:] = x[:-k]
    return y if k > 0 else x.copy()


def acc(X, y, tr, te, nc=2):
    mu = X[tr].mean(0); sd = X[tr].std(0)+1e-9; X = (X-mu)/sd; Y = np.eye(nc)[y]; best = 0.0; wbest = None
    for al in [1e-2, .1, 1, 10, 100, 1e3]:
        try:
            Wt = np.linalg.solve(X[tr].T@X[tr]+al*np.eye(X.shape[1]), X[tr].T@Y[tr])
            a = float(np.mean((X[te]@Wt).argmax(1) == y[te]))
            if a > best: best = a
        except Exception: pass
    return best


def main():
    p = OUT/f"cross_die_mixing_raw_{HOST}.npz"
    d = np.load(p); u = d["u"].astype(int); v = d["v"].astype(int); Tn = d["Tn"]
    L = len(u); flat = Tn.reshape(L, -1)
    Xdie = np.hstack([flat, lag(flat, 1), lag(flat, 2)])
    cut = WASHOUT+int(0.7*(L-WASHOUT)); tr = slice(WASHOUT, cut); te = slice(cut, L)
    y = (lag(u, 1) ^ lag(v, 1)).astype(int)

    # --- die full linear ---
    die_xor = acc(Xdie, y, tr, te)

    # --- 1. NULL distribution (phase-shuffle die features), 300 draws ---
    null = []
    for s in range(300):
        r = np.random.default_rng(1000+s); Msur = np.zeros_like(flat)
        for c in range(flat.shape[1]):
            F = np.fft.rfft(flat[:, c]-flat[:, c].mean()); ph = r.uniform(0, 2*np.pi, len(F)); ph[0] = 0
            Msur[:, c] = np.fft.irfft(np.abs(F)*np.exp(1j*ph), n=L)
        Xs = np.hstack([Msur, lag(Msur, 1), lag(Msur, 2)])
        null.append(acc(Xs, y, tr, te))
    null = np.array(null); p95 = float(np.quantile(null, 0.95)); pval = float((null >= die_xor).mean())

    # --- 2. physical u*v partial R^2 per channel (centered, orthogonalized) ---
    uc = (u-u.mean()).astype(float); vc = (v-v.mean()).astype(float); uvc = uc*vc
    A2 = np.stack([np.ones(L), uc, vc], 1); A3 = np.stack([np.ones(L), uc, vc, uvc], 1)
    gains = []
    for c in range(flat.shape[1]):
        ch = flat[:, c]
        b2, *_ = np.linalg.lstsq(A2, ch, rcond=None); r2 = ch-A2@b2
        b3, *_ = np.linalg.lstsq(A3, ch, rcond=None); r3 = ch-A3@b3
        ss = np.sum((ch-ch.mean())**2)+1e-12
        gains.append(float((np.sum(r2**2)-np.sum(r3**2))/ss))
    gains = np.array(gains); uv_med = float(np.median(gains)); uv_max = float(gains.max())

    # --- 3. necessity controls ---
    uu = u.astype(float); vv = v.astype(float)
    u_lin = np.stack([lag(uu, k) for k in range(1, 5)], 1)
    u_cols = [lag(uu, k) for k in range(1, 5)]
    for a, b in itertools.combinations_with_replacement(range(1, 5), 2): u_cols.append(lag(uu, a)*lag(uu, b))
    u_quad = np.stack(u_cols, 1)
    uv_lin = np.stack([lag(uu, k) for k in range(1, 5)] + [lag(vv, k) for k in range(1, 5)], 1)  # BOTH, linear
    uv_prod = np.stack([lag(uu, 1)*lag(vv, 1)], 1)  # the cheat: explicit product (ceiling)
    u_only = max(acc(u_lin, y, tr, te), acc(u_quad, y, tr, te))
    uandv_linear = acc(uv_lin, y, tr, te)
    uv_product_ceiling = acc(uv_prod, y, tr, te)

    # --- 4. bootstrap CI on die XOR (resample test block) ---
    mu = Xdie[tr].mean(0); sd = Xdie[tr].std(0)+1e-9; Xz = (Xdie-mu)/sd; Y = np.eye(2)[y]
    al = 10.0; Wt = np.linalg.solve(Xz[tr].T@Xz[tr]+al*np.eye(Xz.shape[1]), Xz[tr].T@Y[tr])
    pred = (Xz[te]@Wt).argmax(1); correct = (pred == y[te]).astype(float)
    boots = [float(correct[np.random.default_rng(s).integers(0, len(correct), len(correct))].mean()) for s in range(1000)]
    ci = (float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975)))

    real = (die_xor > p95) and (pval < 0.05) and (die_xor - uandv_linear > 0.05) and (uv_med > 0)
    out = {"host": HOST, "die_full_XOR": die_xor, "null_mean": float(null.mean()), "null_p95": p95,
           "null_pvalue": pval, "uv_partialR2_median": uv_med, "uv_partialR2_max": uv_max,
           "u_only_poly": u_only, "u_and_v_LINEAR": uandv_linear, "uv_product_ceiling": uv_product_ceiling,
           "die_XOR_bootCI95": ci, "d_v": float(d.get("Tn") is not None),
           "PRE_REGISTERED_REAL": bool(real)}
    print(json.dumps(out, indent=2))
    print(f"\n  die XOR(u,v)={die_xor:.3f}  null p95={p95:.3f} (p={pval:.3f})  bootCI={ci[0]:.3f}-{ci[1]:.3f}")
    print(f"  u-only={u_only:.3f}   u&v-LINEAR={uandv_linear:.3f}   product-ceiling={uv_product_ceiling:.3f}")
    print(f"  physical u*v partial-R2: median={uv_med:.4f} max={uv_max:.4f}")
    print(f"\n  >>> {'REAL mixing (passes all pre-registered checks)' if real else 'NOT robust — at least one check failed'}")
    (OUT/f"mixing_verify_{HOST}.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
