"""Task B — Test ReverseFuzzyExtractor on Phase 13 signature data (offline).

We load `ikaros_sig_v2.npz` and `daedalus_sig_v2.npz` — each contains
10 repeated 290-dim signature reads of the corresponding host.  We:

  1. Quantize the FIRST read of each host to 256 bits → w_ref.
  2. Enroll the ReverseFuzzyExtractor with w_ref.
  3. Test verify() against:
       * INTRA-HOST replays  (reads 1..9 of same host)  → expected ACCEPT
       * INTER-HOST imposters (reads of the OTHER host)  → expected REJECT
       * ADVERSARIAL random imposters                    → expected REJECT

This is the OFFLINE substitute for live-chip testing.

Output: results/IDENTITY_BENCHMARK_2026-05-30/embodiment22b_crypto/rfe_offline_results.json
"""
from __future__ import annotations
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from reverse_fuzzy import ReverseFuzzyExtractor, quantize_to_bits, hamming  # noqa

REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
SIG_DIR = os.path.join(REPO, 'results/IDENTITY_BENCHMARK_2026-05-30/embodiment13')
OUT_DIR = os.path.join(REPO, 'results/IDENTITY_BENCHMARK_2026-05-30/embodiment22b_crypto')
os.makedirs(OUT_DIR, exist_ok=True)


def load_sigs(host):
    p = os.path.join(SIG_DIR, f'{host}_sig_v2.npz')
    d = np.load(p)
    return np.asarray(d['vec'])   # (10, 290)


def select_stable_dims(ik, da, n_dims=256):
    """Pick the n_dims dims of IKAROS that maximize
        (cross-host separability) / (intra-host noise).

    Cross-host separability is approximated by the Cohen-d effect size
    between ikaros and daedalus.  We require dims to be present-in-data
    (non-degenerate variance) on at least one host.
    """
    mu_ik = ik.mean(axis=0); sd_ik = ik.std(axis=0)
    mu_da = da.mean(axis=0); sd_da = da.std(axis=0)
    pooled = np.sqrt(sd_ik**2 + sd_da**2) + 1e-9
    d_cross = np.abs(mu_ik - mu_da) / pooled
    # Penalize dims with huge intra-host noise; reward cross-host separation.
    noise_pen = sd_ik / (np.abs(mu_ik) + 1e-3)
    score = d_cross - 0.3 * noise_pen
    # Discard fully-degenerate dims
    bad = (sd_ik == 0) & (sd_da == 0)
    score[bad] = -np.inf
    top = np.argsort(-score)[:n_dims]
    return np.sort(top)


def quantize_dimset_sign(vec, ref_median):
    """Sign-of-(vec - ref_median) per selected dim → 1 bit per dim."""
    return (vec > ref_median).astype(np.uint8)


def main():
    ik = load_sigs('ikaros')        # (10, 290)
    da = load_sigs('daedalus')

    print(f"Phase 13 data: ikaros (10, 290), daedalus (10, 290)")
    print("Threshold: midpoint between per-dim ikaros mean and daedalus mean.")
    print("Dim selection: maximize (cross-host Cohen-d) - 0.3*(intra-host noise).")

    # We need to scale bits_ref to whatever N_BITS the RFE chooses.
    # Strategy: for m=9 (N_BITS=512), pick 512 dims (pad with zero-bits
    # from runtime-stable extra dims, or repeat the existing 256 bits).
    # Simplest: re-run selection at the requested N.

    def make_w_ref(n_bits):
        dims_n = select_stable_dims(ik, da, n_dims=min(n_bits, ik.shape[1]))
        mu_ik_n = ik.mean(axis=0)[dims_n]
        mu_da_n = da.mean(axis=0)[dims_n]
        thr_n = 0.5 * (mu_ik_n + mu_da_n)
        ik_bits_n = (ik[:, dims_n] > thr_n[None, :]).astype(np.uint8)
        bits_ref_n = (ik_bits_n.mean(axis=0) > 0.5).astype(np.uint8)
        da_bits_n = (da[:, dims_n] > thr_n[None, :]).astype(np.uint8)
        # pad up to n_bits if dims_n < n_bits
        if bits_ref_n.shape[0] < n_bits:
            pad = n_bits - bits_ref_n.shape[0]
            bits_ref_n = np.concatenate([bits_ref_n, np.zeros(pad, dtype=np.uint8)])
            ik_bits_n  = np.concatenate([ik_bits_n,  np.zeros((ik_bits_n.shape[0], pad), dtype=np.uint8)], axis=1)
            da_bits_n  = np.concatenate([da_bits_n,  np.zeros((da_bits_n.shape[0], pad), dtype=np.uint8)], axis=1)
        return bits_ref_n[:n_bits], ik_bits_n[:, :n_bits], da_bits_n[:, :n_bits]

    sweep = []
    for (m_bch, t) in [(8, 4), (8, 8), (8, 16), (8, 24),
                       (9, 16), (9, 32), (9, 48)]:
        try:
            rfe = ReverseFuzzyExtractor(t=t, m=m_bch)
        except Exception as e:
            sweep.append(dict(t=t, m=m_bch, error=str(e)))
            continue
        bits_ref, ik_z, da_z = make_w_ref(rfe.N_BITS)
        rfe.enroll(bits_ref)

        # Intra-host (same chip, fresh read)
        intra = []
        for i in range(10):
            bits_i = ik_z[i]
            ok, K_rec, ham = rfe.verify(bits_i)
            intra.append((bool(ok), ham))
        intra_accept = sum(1 for ok, _ in intra if ok)
        intra_hams = [h for _, h in intra]

        # Inter-host (daedalus reads, projected into ikaros frame)
        inter = []
        for i in range(10):
            bits_i = da_z[i]
            ok, K_rec, ham = rfe.verify(bits_i)
            inter.append((bool(ok), ham))
        inter_accept = sum(1 for ok, _ in inter if ok)
        inter_hams = [h for _, h in inter]

        # Random imposters
        rng = np.random.default_rng(t * 100 + m_bch)
        rnd_accept = 0
        rnd_hams = []
        for _ in range(100):
            rb = rng.integers(0, 2, size=rfe.N_BITS, dtype=np.uint8)
            ok, _, ham = rfe.verify(rb)
            if ok: rnd_accept += 1
            rnd_hams.append(int(ham))

        sweep.append(dict(
            t=t, m=m_bch, n_bits=rfe.N_BITS,
            ecc_bits=rfe.bch.ecc_bits,
            intra_accept=intra_accept,
            intra_total=10,
            intra_ham_mean=float(np.mean(intra_hams)),
            intra_ham_max=int(np.max(intra_hams)),
            inter_accept=inter_accept,
            inter_total=10,
            inter_ham_mean=float(np.mean(inter_hams)),
            inter_ham_min=int(np.min(inter_hams)),
            random_accept=rnd_accept,
            random_total=100,
            random_ham_mean=float(np.mean(rnd_hams)),
        ))
        print(f"  m={m_bch} t={t:2d} N={rfe.N_BITS:3d}  intra={intra_accept:2d}/10 (ham μ={np.mean(intra_hams):.1f})  "
              f"inter={inter_accept:2d}/10 (ham μ={np.mean(inter_hams):.1f})  "
              f"random={rnd_accept:3d}/100 (ham μ={np.mean(rnd_hams):.1f})")

    out_path = os.path.join(OUT_DIR, 'rfe_offline_results.json')
    json.dump(dict(sweep=sweep,
                   enrolled_host='ikaros',
                   imposter_host='daedalus',
                   note='ReverseFuzzyExtractor — offline test on Phase 13 sigs'),
              open(out_path, 'w'), indent=2)
    print(f"\nWrote {out_path}")


if __name__ == '__main__':
    main()
