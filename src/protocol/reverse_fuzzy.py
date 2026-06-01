"""T2.1 — Reverse Fuzzy Extractor (Van Herrewege et al., FC'12).

Classical fuzzy extractor (Dodis-Reyzin-Smith 2004):
  Enrollment:  (R, P)  ←  Gen(w_enroll)            # P = helper, public
               store R as secret key
  Verify:      R'      ←  Rep(w_noisy, P)          # uses public P
  Property:    R' == R  iff  Hamming(w_enroll, w_noisy) ≤ t.

The classical scheme requires the PROVER to know P (helper data), and P
is typically published.  This leaks ~|P| bits of information about the
chip fingerprint w_enroll (via the code-offset structure).

REVERSE fuzzy extractor (Van Herrewege 2012):
  The VERIFIER holds a SECRET reference (the enrolled fingerprint w_ref
  *and* an internal helper map).  The prover sends only a fresh noisy
  reading w_noisy.  The verifier computes the syndrome of (w_ref XOR
  w_noisy) under a linear error-correcting code, and if the syndrome
  decodes inside the radius t, recovers the same secret K_chip both
  parties would agree on under the classical FE.  Crucially, NO HELPER
  DATA IS EVER PUBLIC, eliminating the helper-data leakage attack.

Construction — code-offset over GF(2) with BCH(255, 131, d=33):
  ------------------------------------------------------
  Concretely we use bchlib BCH(t=16, m=8) which gives:
      n = 255 bits, ecc_bits = 124, k_bits = 131  (we use 128 = 16 data bytes).
      can correct up to 16 random bit errors.
  Codeword bits are encoded as (data_bytes || ecc_bytes) = 32 bytes = 256 bits
  (the high bit of the 256th position is unused / padded zero).

  enrollment(w_ref_bits ∈ {0,1}^256):
      Pick uniformly random data ∈ {0,1}^128 ; ecc = BCH_encode(data).
      Codeword c_bits = (data ++ ecc) ∈ {0,1}^256.
      P_bits = w_ref XOR c                          (PRIVATE — kept by verifier)
      K      = SHA256("rfe-secret" || c_bits)
      Store (w_ref, P_bits, K) privately on verifier.

  verify(w_noisy_bits):                            # prover sends w_noisy
      v = w_noisy XOR P                            # v = c + e, e = w_ref XOR w_noisy
      data_v, ecc_v  = split(v)
      nerr = BCH.decode(data_v, ecc_v)
      BCH.correct(data_v, ecc_v)                   # in-place
      if nerr < 0 or nerr > t : REJECT
      c_hat = (data_v ++ ecc_v) bits
      K' = SHA256("rfe-secret" || c_hat)
      ACCEPT iff K' == K (== original K iff e was correctable).

Properties:
  * If Hamming(w_ref, w_noisy) ≤ t  ⇒  K' == K  (always correct).
  * If Hamming(w_ref, w_noisy) > t  ⇒  decode either fails (nerr<0) or
    returns a wrong codeword; K' != K  ⇒  REJECT.
  * Adversary without (w_ref, P) sees only w_noisy.  They learn nothing
    about K beyond what the chip's own physical jitter reveals.
  * Helper P NEVER leaves the verifier — closes the public-helper leakage.

This module is OFFLINE — no chip access needed.

Author: Tier-2 FabricCrypt — 2026-06-01
"""
from __future__ import annotations
import os, hashlib, secrets
import numpy as np
import bchlib

DEFAULT_T = 16   # correct up to 16 bit-flips (~6% of 256 bits)


# ----------- fingerprint quantization -----------
def quantize_to_bits(vec: np.ndarray, n_bits_total: int = 256) -> np.ndarray:
    """Map a real-valued fingerprint vector to an n_bits_total-bit binary
    string using a median + MAD-shifted threshold ladder.

    Each dim contributes ⌈n_bits_total/n⌉ bits.  Truncated to n_bits_total.
    """
    v = np.asarray(vec, dtype=np.float64).ravel()
    n = len(v)
    bits_per_dim = max(1, (n_bits_total + n - 1) // n)
    out = np.zeros(n * bits_per_dim, dtype=np.uint8)
    med = np.median(v)
    mad = np.median(np.abs(v - med)) + 1e-9
    for b in range(bits_per_dim):
        shift = ((b + 1) // 2) * mad * (1 if b % 2 == 0 else -1)
        thr = med + shift
        out[b * n:(b + 1) * n] = (v > thr).astype(np.uint8)
    return out[:n_bits_total]


def bits_to_bytes(bits: np.ndarray) -> bytes:
    b = np.asarray(bits, dtype=np.uint8)
    pad = (-len(b)) % 8
    if pad: b = np.concatenate([b, np.zeros(pad, dtype=np.uint8)])
    return np.packbits(b, bitorder='big').tobytes()


def bytes_to_bits(data: bytes, n: int) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    bits = np.unpackbits(arr, bitorder='big')
    return bits[:n].astype(np.uint8)


def hamming(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.sum(np.asarray(a, dtype=np.uint8) ^ np.asarray(b, dtype=np.uint8)))


class ReverseFuzzyExtractor:
    """Reverse fuzzy extractor over BCH(255-bit ~ 256-bit codeword).

    Concrete framing: codeword = 16 data bytes (128 bits) || 16 ecc bytes (128 bits).
    Total 256 bits.  bchlib uses 124 of the 128 ecc bits internally; the rest are
    just zero-padding, which doesn't change the code's properties.

    All helper data (P, w_ref) is kept PRIVATE on the verifier.
    """

    # Codeword length is determined by m:
    #   m=8 → 256 bits (32 bytes) — supports t up to 24
    #   m=9 → 512 bits (64 bytes) — supports t up to 48+
    # Public attributes ECC_BYTES / DATA_BYTES / N_BITS set in __init__.
    def __init__(self, t: int = DEFAULT_T, m: int = 8):
        self.t = t
        self.m = m
        self.bch = bchlib.BCH(t=t, m=m)
        self.ECC_BYTES = self.bch.ecc_bytes
        # round codeword up to next byte: ⌈n/8⌉ + extra padding from ecc
        cw_bytes = ((self.bch.n + 7) // 8) + 1   # +1 byte of headroom for padding
        # If m=8: n=255 ⇒ 32-byte codeword.  m=9: n=511 ⇒ 64-byte codeword.
        # We use the next-power-of-2 byte count.
        if m == 8: cw_bytes = 32
        elif m == 9: cw_bytes = 64
        self.N_BITS = cw_bytes * 8
        self.DATA_BYTES = cw_bytes - self.ECC_BYTES
        if self.DATA_BYTES < 1:
            raise ValueError(f"t={t} m={m} leaves no room for data (ecc_bytes={self.ECC_BYTES})")
        self._enrollment = None

    def _random_codeword(self) -> tuple[bytes, bytes, np.ndarray]:
        """Return (data_bytes, ecc_bytes, codeword_bits)."""
        data = secrets.token_bytes(self.DATA_BYTES)
        ecc  = bytes(self.bch.encode(data))
        # ecc is ecc_bytes long (= 16); ok
        cw_bits = bytes_to_bits(data + ecc, self.N_BITS)
        return data, ecc, cw_bits

    def _try_decode(self, v_bits: np.ndarray) -> np.ndarray | None:
        """Attempt to decode v_bits as a noisy codeword.  Return the corrected
        codeword bits, or None on failure."""
        v_bytes = bits_to_bytes(v_bits)
        data = bytearray(v_bytes[:self.DATA_BYTES])
        ecc  = bytearray(v_bytes[self.DATA_BYTES:self.DATA_BYTES + self.ECC_BYTES])
        try:
            nerr = self.bch.decode(data, ecc)
        except Exception:
            return None
        if nerr < 0 or nerr > self.t:
            return None
        self.bch.correct(data, ecc)
        # corrected codeword bits
        return bytes_to_bits(bytes(data) + bytes(ecc), self.N_BITS)

    # ---------- public API ----------
    def enroll(self, w_ref_bits: np.ndarray) -> dict:
        """Generate (P, K) for the enrolled fingerprint.  All kept PRIVATE."""
        assert len(w_ref_bits) == self.N_BITS, f"need {self.N_BITS} bits, got {len(w_ref_bits)}"
        _data, _ecc, c_bits = self._random_codeword()
        P_bits = (w_ref_bits.astype(np.uint8) ^ c_bits).astype(np.uint8)
        K = hashlib.sha256(b"rfe-secret-v1|" + bits_to_bytes(c_bits)).digest()
        self._enrollment = dict(w_ref=w_ref_bits.astype(np.uint8).copy(),
                                P=P_bits, K=K, c=c_bits)
        return dict(K=K, t=self.t, n_bits=self.N_BITS,
                    ecc_bits=self.bch.ecc_bits)

    def verify(self, w_noisy_bits: np.ndarray) -> tuple[bool, bytes | None, int]:
        """Reverse-FE verify.  Returns (accepted, recovered_K_or_None, hamming_seen).

        Helper data P never leaves the verifier.  Prover only sends w_noisy_bits.
        """
        assert self._enrollment is not None, "must enroll first"
        assert len(w_noisy_bits) == self.N_BITS
        w_ref = self._enrollment['w_ref']
        P     = self._enrollment['P']
        K_ref = self._enrollment['K']
        ham   = hamming(w_ref, w_noisy_bits)
        v     = (w_noisy_bits.astype(np.uint8) ^ P).astype(np.uint8)
        c_hat = self._try_decode(v)
        if c_hat is None:
            return False, None, ham
        K_rec = hashlib.sha256(b"rfe-secret-v1|" + bits_to_bytes(c_hat)).digest()
        return (K_rec == K_ref), K_rec, ham


# ============== smoke test ==============
def _smoke():
    rng = np.random.default_rng(0)
    rfe = ReverseFuzzyExtractor(t=16)
    print(f"BCH(n={rfe.N_BITS}, data_bytes={rfe.DATA_BYTES}, ecc_bytes={rfe.ECC_BYTES}, t={rfe.t})")

    w_ref = rng.integers(0, 2, size=rfe.N_BITS, dtype=np.uint8)
    enr = rfe.enroll(w_ref)
    print("enrolled, K =", enr['K'].hex()[:16] + '...')

    for n_flip in [0, 1, 5, 10, 15, 16, 17, 25, 50, 100]:
        idx = rng.choice(rfe.N_BITS, n_flip, replace=False)
        w_noisy = w_ref.copy()
        w_noisy[idx] ^= 1
        ok, K_rec, ham = rfe.verify(w_noisy)
        k_match = (K_rec == enr['K']) if K_rec is not None else False
        print(f"  flips={n_flip:3d}  ham={ham:3d}  accept={ok}  K_match={k_match}")


if __name__ == '__main__':
    _smoke()
