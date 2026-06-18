"""T2.2 — Controlled PUF (Suh–Devadas, DAC 2007).

A raw PUF exposes its CHALLENGE/RESPONSE interface directly.  An adversary
who can query many (C, R) pairs (and/or observe internal intermediates) can
build an ML model that predicts R from C — the modeling attack of Rührmair
et al. (CCS 2010).  Most arbiter/XOR-arbiter PUFs are broken this way.

The "controlled PUF" wraps the raw PUF in two hash layers:

   challenge_in        →  H_in(C, ID_chip)  →  raw_PUF  →  raw_out
                                                              ↓
   response_out        ←       H_out(raw_out, C, ID_chip, ctx)

Properties (Suh-Devadas 2007 §3, Maes-Verbauwhede 2010 survey):
   * The adversary CANNOT directly query raw_PUF with chosen inputs:
     every external challenge is mangled through H_in, so the attacker
     can only ever see (C, R) pairs at the wrapped boundary.  Internal
     PUF challenges are pseudo-random functions of C, eliminating the
     "chosen-challenge" power of the Rührmair attack.
   * The output hash H_out destroys linear structure of raw_out; even if
     adversary somehow recovered the raw response, computing H_out
     requires knowing C and ID_chip in full.
   * Combined with the response being non-malleable (H_out is a random
     oracle in our analysis), the only attack avenue left is to physically
     possess the chip and run it.

We instantiate H_in / H_out using SHAKE256 with strict domain separation.

For FabricCrypt, the "raw PUF" is our existing physical signature (RAPL,
thermal zones, nanosleep jitter, c2c latency, ...) keyed by K_chip and
challenged by a nonce.  We wrap it:

  external_challenge = nonce   (16-32 bytes from verifier)
  inner_nonce        = SHAKE256("ctrl-puf-in" || nonce || ID_chip, 16B)
  raw_response       = NonceSigV2(inner_nonce)  (64-dim float vector)
  raw_response_bytes = serialize(raw_response)
  wrapped_response   = SHAKE256("ctrl-puf-out" || raw_response_bytes
                                || nonce || ID_chip, OUT_BYTES)

The wrapped response is what the prover transmits.  The verifier does
the same wrapping in its model and compares (with appropriate
Hamming-distance tolerance via the reverse fuzzy extractor).

NOTE: this module is mostly composition.  The interesting tests are in
adversary.py (T2.5) which try to model this wrapped PUF and measure
forgery rate vs training-pair count.

Author: Tier-2 FabricCrypt — 2026-06-01
"""
from __future__ import annotations
import hashlib
import numpy as np


def shake(label: bytes, *parts: bytes, n: int = 32) -> bytes:
    h = hashlib.shake_256()
    h.update(label)
    for p in parts:
        h.update(b"|")
        h.update(p)
    return h.digest(n)


def chip_id(host: str, K_chip: bytes) -> bytes:
    """Public-but-bound chip identifier.  Derived from K_chip + host.
    Verifier can use this as a lookup index; it does NOT leak K_chip."""
    return shake(b"ctrl-puf-id", host.encode('utf-8'), K_chip, n=32)


def wrap_challenge(external_nonce: bytes, host: str, K_chip: bytes,
                   inner_nonce_bytes: int = 16) -> bytes:
    """H_in: external challenge → internal challenge.  Mangled with K_chip
    so adversary can't pick the inner nonce."""
    id_b = chip_id(host, K_chip)
    return shake(b"ctrl-puf-in", external_nonce, id_b, host.encode('utf-8'),
                 n=inner_nonce_bytes)


def wrap_response(raw_response: np.ndarray, external_nonce: bytes,
                  host: str, K_chip: bytes, out_bytes: int = 32) -> bytes:
    """H_out: raw PUF response → wrapped response.  Destroys linear structure
    and binds response to (challenge, chip_id, host)."""
    raw_bytes = raw_response.astype(np.float32).tobytes()
    id_b = chip_id(host, K_chip)
    return shake(b"ctrl-puf-out", raw_bytes, external_nonce, id_b,
                 host.encode('utf-8'), n=out_bytes)


class ControlledPUF:
    """Wrap any callable `raw_puf(inner_nonce_bytes) -> np.ndarray` as a
    controlled PUF.  Adds H_in / H_out around it.

    Note: the BCH-quantizable fingerprint is the RAW vector; the wrapped
    output is a hash digest (uniform over {0,1}^{8*out_bytes}).  We expose
    BOTH so that the protocol layer can:
       (i)  use wrap_bits for high-entropy K-derivation
       (ii) use raw vector for ML-classifier matching with tolerance
    """

    def __init__(self, raw_puf, host: str, K_chip: bytes,
                 inner_nonce_bytes: int = 16, out_bytes: int = 32):
        self.raw_puf = raw_puf
        self.host = host
        self.K_chip = K_chip
        self.inner_nonce_bytes = inner_nonce_bytes
        self.out_bytes = out_bytes

    def query(self, external_nonce: bytes) -> dict:
        inner = wrap_challenge(external_nonce, self.host, self.K_chip,
                               self.inner_nonce_bytes)
        raw = self.raw_puf(inner)
        wrapped = wrap_response(raw, external_nonce, self.host, self.K_chip,
                                self.out_bytes)
        return dict(inner=inner, raw=raw, wrapped=wrapped)


# ============== smoke test ==============
def _smoke():
    rng = np.random.default_rng(0)
    # mock raw PUF: deterministic 64-dim function of inner nonce + secret seed
    chip_seed = rng.bytes(32)
    def raw_puf(inner_nonce):
        seed_int = int.from_bytes(hashlib.sha256(chip_seed + inner_nonce).digest()[:8], 'little')
        r = np.random.default_rng(seed_int)
        return r.normal(0, 1, 64).astype(np.float32)

    cpuf = ControlledPUF(raw_puf, host="ikaros", K_chip=chip_seed)
    n = rng.bytes(8)
    out1 = cpuf.query(n)
    out2 = cpuf.query(n)
    print("same nonce → identical wrapped?", out1['wrapped'] == out2['wrapped'])
    print("wrapped hex[:16]:", out1['wrapped'].hex()[:16])
    n2 = rng.bytes(8)
    out3 = cpuf.query(n2)
    print("diff nonce → diff wrapped?", out1['wrapped'] != out3['wrapped'])
    # Different K_chip → different chip_id even with same raw response
    chip_seed2 = rng.bytes(32)
    cpuf2 = ControlledPUF(raw_puf, host="ikaros", K_chip=chip_seed2)
    out4 = cpuf2.query(n)
    print("diff K_chip, same nonce → diff wrapped?", out1['wrapped'] != out4['wrapped'])
    print("ID(chip1):", chip_id("ikaros", chip_seed).hex()[:16])
    print("ID(chip2):", chip_id("ikaros", chip_seed2).hex()[:16])


if __name__ == '__main__':
    _smoke()
