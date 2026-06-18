"""H7 CROSS-DIE TRANSFER — is the u·v mixing DIE-SPECIFIC? (fuses UNIQUE x RÄKNA)

Within-die we proved ikaros physically computes u·v (h7_mixing_verify: XOR=0.654, p=0.000). That shows "AN APU
multiplies", not "THIS die multiplies differently". This decides die-specificity, the way all 4 O105 oracles
demanded: train the XOR(u,v) readout on ONE die, test ZERO-SHOT on the OTHER (identical command streams u,v).

  Acc_self  = train ikaros, test ikaros (held-out)   -> ceiling
  Acc_cross = train ikaros, test daedalus             -> if << self, the learned mixing is die-specific
  (and symmetric: train daedalus -> test daedalus vs ikaros)

CONTROLS / anti-confound:
  - per-die z-normalisation (so a trivial baseline/offset/gain difference is NOT mistaken for die-specificity).
  - u-only readout transfer: a u-only function MUST transfer across dies (it's the same command). If u-only
    transfers but the die-XOR readout does NOT, that is the clean die-specificity signature (not just noise).
  - within-die trial noise floor is the transient_vdroop/cross_die self-CI we already have.
PRE-REGISTERED die-specific = (Acc_self - Acc_cross) > 0.10 on XOR(u,v) AND u-only transfers (cross≈self).
Run after pulling cross_die_mixing_raw_daedalus.npz next to the ikaros npz. CPU-only.
"""
from __future__ import annotations
import json, socket, itertools
from pathlib import Path
import numpy as np

OUT = Path(__file__).resolve().parents[2] / "results/IDENTITY_H7_2026-06-09"
WASHOUT = 150


def lag(x, k):
    y = np.zeros_like(x)
    if k > 0: y[k:] = x[:-k]
    return y if k > 0 else x.copy()


def feats(Tn):
    L = Tn.shape[0]; flat = Tn.reshape(L, -1)
    return np.hstack([flat, lag(flat, 1), lag(flat, 2)])


def train_W(X, y, tr, nc=2, al=10.0):
    mu = X[tr].mean(0); sd = X[tr].std(0)+1e-9; Xz = (X-mu)/sd; Y = np.eye(nc)[y]
    W = np.linalg.solve(Xz[tr].T@Xz[tr]+al*np.eye(Xz.shape[1]), Xz[tr].T@Y[tr])
    return W, mu, sd


def apply_W(X, W, mu, sd, y, te):
    Xz = (X-mu)/sd
    return float(np.mean((Xz[te]@W).argmax(1) == y[te]))


def best_self(X, y, tr, te, nc=2):
    b = 0.0
    for al in [1e-2, .1, 1, 10, 100, 1e3]:
        W, mu, sd = train_W(X, y, tr, nc, al); b = max(b, apply_W(X, W, mu, sd, y, te))
    return b


def load(host):
    p = OUT/f"cross_die_mixing_raw_{host}.npz"
    if not p.exists(): return None
    d = np.load(p); return d["u"].astype(int), d["v"].astype(int), d["Tn"]


def transfer(srcname, src, dstname, dst, taskfn, label):
    us, vs, Ts = src; ud, vd, Td = dst
    Xs = feats(Ts); Xd = feats(Td)
    ys = taskfn(us, vs); yd = taskfn(ud, vd)
    Ls = len(us); cs = WASHOUT+int(0.7*(Ls-WASHOUT)); tr_s = slice(WASHOUT, cs); te_s = slice(cs, Ls)
    Ld = len(ud); cd = WASHOUT+int(0.7*(Ld-WASHOUT)); te_d = slice(cd, Ld)
    # train on src (train split), pick alpha by src held-out, then transfer to dst with SAME W
    best = None
    for al in [.1, 1, 10, 100]:
        W, mu, sd = train_W(Xs, ys, tr_s, 2, al)
        a_self = apply_W(Xs, W, mu, sd, ys, te_s)
        if best is None or a_self > best[0]:
            a_cross = apply_W(Xd, W, mu, sd, yd, te_d)   # per-src normalisation applied to dst
            # also re-normalise per-dst (remove trivial offset/gain) then apply the SAME W
            mud = Xd[slice(WASHOUT, cd)].mean(0); sdd = Xd[slice(WASHOUT, cd)].std(0)+1e-9
            Xdn = (Xd-mud)/sdd
            a_cross_dn = float(np.mean((Xdn[te_d]@W).argmax(1) == yd[te_d]))
            best = (a_self, a_cross, a_cross_dn)
    print(f"  [{label}] {srcname}->{srcname}={best[0]:.3f}  {srcname}->{dstname}={best[1]:.3f} "
          f"(dst-renorm={best[2]:.3f})  drop={best[0]-best[1]:+.3f}", flush=True)
    return {"src": srcname, "dst": dstname, "self": best[0], "cross": best[1],
            "cross_dstnorm": best[2], "drop": best[0]-best[1]}


def main():
    ik = load("ikaros"); da = load("daedalus")
    if ik is None or da is None:
        print(f"MISSING: ikaros={ik is not None} daedalus={da is not None}. Pull cross_die_mixing_raw_daedalus.npz first.")
        return
    def XORuv(u, v): return (lag(u, 1) ^ lag(v, 1)).astype(int)
    def RECu(u, v): return lag(u, 1).astype(int)
    print("=== XOR(u,v) — die-specific mixing? ===", flush=True)
    rows = [transfer("ikaros", ik, "daedalus", da, XORuv, "XOR"),
            transfer("daedalus", da, "ikaros", ik, XORuv, "XOR")]
    print("=== RECALL(u) — u-only control, MUST transfer (same command) ===", flush=True)
    ctrl = [transfer("ikaros", ik, "daedalus", da, RECu, "REC_u"),
            transfer("daedalus", da, "ikaros", ik, RECu, "REC_u")]
    xor_drop = np.mean([r["drop"] for r in rows]); u_drop = np.mean([r["drop"] for r in ctrl])
    die_specific = (xor_drop > 0.10) and (u_drop < 0.10)
    out = {"xor_transfer": rows, "recall_u_transfer": ctrl, "mean_xor_drop": float(xor_drop),
           "mean_u_drop": float(u_drop), "DIE_SPECIFIC_MIXING": bool(die_specific),
           "verdict": ("DIE-SPECIFIC u·v mixing CONFIRMED — UNIQUE x RÄKNA fused" if die_specific else
                       "mixing transfers across dies (or u-only also drops) — NOT cleanly die-specific")}
    (OUT/"cross_die_transfer.json").write_text(json.dumps(out, indent=2))
    print(f"\n  mean XOR drop={xor_drop:+.3f}   mean u-only drop={u_drop:+.3f}")
    print(f"  >>> {out['verdict']}", flush=True)


if __name__ == "__main__":
    main()
