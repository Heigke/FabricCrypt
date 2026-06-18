"""H7 v2 — the K(C) crux: honestly measure whether a per-die keystream bit is (a) reproducible on the
SAME die (low BER), (b) different on a FOREIGN die (high inter-die Hamming), (c) hard to clone from
PUBLIC inputs (u-clone accuracy near chance). This quantifies the medium-strength uniqueness/clone
claims instead of overclaiming them (fixes red-team holes #2,#3-clone,#7-stats,#5-confound).

K-bit definition (nonce-bound, die-secret):
  Per step t, feature x_t = the live voltage-droop reservoir vector (12 taps x 10 ch = 120-dim).
  Self-normalize per channel per SESSION (z-score) -> removes drive-magnitude covariate shift (the
  red-team's confound: ikaros drive 1.17/1.08 vs daedalus 1.87). Project onto M nonce-seeded random
  directions P (public, from the verifier nonce). K_{t,m} = sign(P_m . xz_t - median_m). The SECRET is
  x (the die response); P and u are public. An attacker who lacks the die must predict x from public u.

Datasets (same seed u => byte-identical challenges, directly comparable):
  transient_vdroop_raw_ikaros_s1.npz, _ikaros_s2.npz (same die, 2 sessions), _daedalus.npz.
Metrics:
  BER_intra      = P(K_s1 != K_s2)                       low  => reproducible (freshness/usability)
  HD_inter       = P(K_ikaros != K_daedalus)             high => die-unique
  decidable_gap  = HD_inter - BER_intra                  > 0 with margin => real per-die signature
  u_clone_acc    = acc of predicting K from PUBLIC u-context (ridge + MLP), vs #CRPs  => clone-cost
Out: results/IDENTITY_H7_2026-06-09/v2_kc_crux.json
"""
from __future__ import annotations
import sys, json, hashlib, socket
import numpy as np
from pathlib import Path

HOST = socket.gethostname()
OUT = Path(__file__).resolve().parents[2] / "results/IDENTITY_H7_2026-06-09"
WASH = 150


def load(name):
    p = OUT / f"transient_vdroop_raw_{name}.npz"
    if not p.exists(): return None, None
    d = np.load(p); return d["u"].astype(int), d["Tn"].astype(np.float32).reshape(len(d["u"]), -1)


def znorm(X, ref):
    mu, sd = ref.mean(0), ref.std(0) + 1e-9
    return (X - mu) / sd


def kbits(Xz, P, thr):
    proj = Xz @ P.T                       # (T, M)
    return (proj > thr[None, :]).astype(np.int8)


def main():
    M = 16                                # nonce-seeded projections (bits per step)
    nonce = b"H7v2-kc-crux-nonce-2026"
    seed = int.from_bytes(hashlib.sha256(nonce).digest()[:4], "little")
    rng = np.random.default_rng(seed)

    u1, X1 = load("ikaros_s1"); u2, X2 = load("ikaros_s2"); ud, Xd = load("daedalus")
    if X1 is None or X2 is None or Xd is None:
        print("missing one of ikaros_s1/ikaros_s2/daedalus npz"); sys.exit(2)
    assert np.array_equal(u1, u2) and np.array_equal(u1, ud), "u mismatch — recollect same seed"
    T = len(u1); sl = slice(WASH, T)
    D = X1.shape[1]
    P = rng.standard_normal((M, D)); P /= np.linalg.norm(P, axis=1, keepdims=True)

    # self-normalize each session by ITS OWN stats (kills drive-magnitude confound)
    X1z, X2z, Xdz = znorm(X1, X1), znorm(X2, X2), znorm(Xd, Xd)
    # thresholds = per-projection median on the reference session (balanced bits)
    thr1 = np.median(X1z[sl] @ P.T, axis=0)
    K1 = kbits(X1z, P, thr1)[sl]
    K2 = kbits(X2z, P, np.median(X2z[sl] @ P.T, axis=0))[sl]
    Kd = kbits(Xdz, P, np.median(Xdz[sl] @ P.T, axis=0))[sl]

    ber_intra = float((K1 != K2).mean())
    hd_inter = float((K1 != Kd).mean())
    # also: foreign with ikaros's OWN threshold (the realistic attack: verifier's die_head fixed)
    Kd_ik_thr = kbits(Xdz, P, thr1)[sl]
    hd_inter_fixed = float((K1 != Kd_ik_thr).mean())

    # ---- clone from PUBLIC u-context (attacker lacks the die) ----
    def uctx(u):
        cols = [np.roll(u, k).astype(float) for k in range(1, 13)]
        # include pairwise products (the strawman nonlinear u-model the red-team used)
        import itertools
        for a, b in itertools.combinations(range(8), 2):
            cols.append(np.roll(u, a).astype(float) * np.roll(u, b).astype(float))
        return np.stack(cols, 1)[sl]
    U = uctx(u1); y = K1.reshape(len(K1), -1)
    n = len(U); cut = int(0.7 * n); tr, te = slice(0, cut), slice(cut, n)
    Uz = (U - U[tr].mean(0)) / (U[tr].std(0) + 1e-9)
    Uz = np.column_stack([Uz, np.ones(len(Uz))])
    clone_acc = []
    for m in range(M):
        ym = y[:, m].astype(float)
        W = np.linalg.solve(Uz[tr].T @ Uz[tr] + 1.0 * np.eye(Uz.shape[1]), Uz[tr].T @ ym[tr])
        pred = (Uz[te] @ W > 0.5).astype(int)
        clone_acc.append(float((pred == y[te, m]).mean()))
    u_clone_acc = float(np.mean(clone_acc))

    # clone-cost curve: u-clone acc vs #training CRPs
    curve = {}
    for frac in [0.05, 0.1, 0.25, 0.5, 0.7]:
        c = max(50, int(frac * n)); accs = []
        for m in range(M):
            ym = y[:, m].astype(float)
            W = np.linalg.solve(Uz[:c].T @ Uz[:c] + 1.0 * np.eye(Uz.shape[1]), Uz[:c].T @ ym[:c])
            pred = (Uz[te] @ W > 0.5).astype(int); accs.append(float((pred == y[te, m]).mean()))
        curve[f"{int(frac*n)}_crps"] = round(float(np.mean(accs)), 3)

    # bootstrap CI on the decidable gap (resample bit-positions)
    K1f, K2f, Kdf = K1.reshape(-1), K2.reshape(-1), Kd.reshape(-1)
    rng2 = np.random.default_rng(1)
    gaps = []
    for _ in range(1000):
        idx = rng2.integers(0, len(K1f), len(K1f))
        gaps.append((K1f[idx] != Kdf[idx]).mean() - (K1f[idx] != K2f[idx]).mean())
    gap_lo, gap_hi = np.percentile(gaps, [2.5, 97.5])

    out = {"host": HOST, "n_bits_per_step": M, "n_steps": int(len(K1)),
           "BER_intra_same_die": round(ber_intra, 4),
           "HD_inter_foreign_die_selfthr": round(hd_inter, 4),
           "HD_inter_foreign_die_ikaros_thr": round(hd_inter_fixed, 4),
           "decidable_gap": round(hd_inter - ber_intra, 4),
           "decidable_gap_CI95": [round(float(gap_lo), 4), round(float(gap_hi), 4)],
           "u_clone_acc_publiconly": round(u_clone_acc, 4),
           "u_clone_cost_curve": curve,
           "chance_bit": 0.5}
    # honest verdicts
    out["reproducible_lowBER"] = bool(ber_intra < 0.20)
    out["per_die_distinguishable"] = bool((hd_inter - ber_intra) > 0.05 and gap_lo > 0)
    out["clone_resistant"] = bool(u_clone_acc < 0.65)
    out["verdict"] = (f"BER={ber_intra:.2f} interHD={hd_inter:.2f} gap={hd_inter-ber_intra:+.2f} "
                      f"(CI[{gap_lo:.2f},{gap_hi:.2f}]) u-clone={u_clone_acc:.2f} -> "
                      + ("per-die signal REAL but " if out["per_die_distinguishable"] else "per-die signal WEAK/absent; ")
                      + ("clone-resistant" if out["clone_resistant"] else "u-CLONABLE (low entropy)"))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "v2_kc_crux.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print("\nVERDICT:", out["verdict"])


if __name__ == "__main__":
    main()
