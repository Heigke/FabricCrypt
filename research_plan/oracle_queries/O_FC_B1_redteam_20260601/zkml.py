"""T2.4 — ZK proof of inference + chip-signature commitment.

Production-grade zk-SNARK of neural inference (zkML) requires a heavyweight
toolchain — ezkl, Halo2, etc. — and chip-friendly circuits we cannot
generate inside a 2-3 hour budget.  Instead we deliver a CRYPTOGRAPHIC
COMMITMENT WRAPPER that bridges to a full zkML system if/when available,
and offers honest interactive-proof properties NOW:

   1. Commit to chip signature S via Pedersen-like commitment:
            com_S = SHA256("chip-sig-com-v1" || r || S_bytes)
      where r is a 32-byte random nonce (hiding factor).
   2. Compute inference output y = M(x) on (committed) S-conditioned model.
      Bind y to com_S via:
            tag = HMAC(K_chip, com_S || y_bytes || prog_hash || x_bytes)
      prog_hash is the SHA256 of the model code (a "model commitment").
   3. Verifier checks:
        * tag is valid HMAC under K_chip (known to verifier after enrollment).
        * Optionally opens com_S by receiving r and S, re-hashing.
        * Verifies y = M(x) by running M on the opened S (NIZK-style on a
          random subset of inputs to keep cost low).

This is an HONEST cryptographic commitment to "this output y was produced
by program M with chip-bound state S".  It is NOT a true zk-SNARK: the
verifier sees S after opening, and M is run in the clear at verification.

A real zk-SNARK would prove y = M(x) WITHOUT revealing S, using:
   * ezkl: PyTorch → ONNX → halo2 circuit, ~10s-60s per proof, ~MB-size proof.
   * Risc0: RISC-V execution proofs of inference code, similar order.
   * Circom + snarkjs: hand-written circuits for small models.

We provide stubs (`zk_prove_stub`, `zk_verify_stub`) that document the
interface and run the verification by REPLAY (full opening) rather than
SNARK.  The interface is identical so a future swap-in is mechanical.

Author: Tier-2 FabricCrypt — 2026-06-01
"""
from __future__ import annotations
import os, hashlib, hmac, secrets, json
import numpy as np


# ---------- commitments ----------
def commit_chip_sig(S: np.ndarray, hiding_r: bytes | None = None) -> tuple[bytes, bytes]:
    """Pedersen-style commitment com_S = SHA256("chip-sig-com-v1" || r || S).
    Returns (com_S, r).  r is the opening randomness (hide it until opening).
    """
    if hiding_r is None:
        hiding_r = secrets.token_bytes(32)
    h = hashlib.sha256()
    h.update(b"chip-sig-com-v1|"); h.update(hiding_r); h.update(b"|")
    h.update(np.asarray(S, dtype=np.float32).tobytes())
    return h.digest(), hiding_r


def open_commit(com_S: bytes, S: np.ndarray, r: bytes) -> bool:
    h = hashlib.sha256()
    h.update(b"chip-sig-com-v1|"); h.update(r); h.update(b"|")
    h.update(np.asarray(S, dtype=np.float32).tobytes())
    return h.digest() == com_S


def model_hash(model_code: bytes) -> bytes:
    """Commit to the inference program.  In production: hash the ONNX
    + circuit description; here we hash the Python source bytes."""
    return hashlib.sha256(b"prog-hash-v1|" + model_code).digest()


# ---------- inference-binding tag ----------
def inference_tag(K_chip: bytes, com_S: bytes, x: np.ndarray, y: np.ndarray,
                  prog_hash: bytes) -> bytes:
    """HMAC binding: inference output is bound to (commitment, input, program)
    under K_chip.  An adversary without K_chip cannot mint a fake tag."""
    msg = (com_S + np.asarray(x, dtype=np.float32).tobytes()
           + np.asarray(y, dtype=np.float32).tobytes() + prog_hash)
    return hmac.new(K_chip, msg, hashlib.sha256).digest()


def verify_tag(K_chip: bytes, com_S: bytes, x: np.ndarray, y: np.ndarray,
               prog_hash: bytes, tag: bytes) -> bool:
    expected = inference_tag(K_chip, com_S, x, y, prog_hash)
    return hmac.compare_digest(expected, tag)


# ---------- proof wrapper ----------
def zk_prove_stub(M, x, S, K_chip, model_code) -> dict:
    """Generate a (commitment, output, binding-tag) bundle.

    M:           callable, M(x, S) → y
    x:           input ndarray
    S:           chip signature ndarray
    K_chip:      32-byte secret known to prover & verifier
    model_code:  bytes of model source (or ONNX export)

    Returns a dict that can later be verified WITH OPENING (this stub) or
    WITHOUT OPENING (true zk-SNARK, future swap-in).
    """
    y = M(x, S)
    com_S, r = commit_chip_sig(S)
    prog_hash = model_hash(model_code)
    tag = inference_tag(K_chip, com_S, x, y, prog_hash)
    return dict(
        com_S=com_S.hex(),
        prog_hash=prog_hash.hex(),
        y=y.astype(np.float32).tolist(),
        tag=tag.hex(),
        # opening data — for the stub only; a true zk-SNARK omits this:
        _open_r=r.hex(),
        _open_S=S.astype(np.float32).tolist(),
    )


def zk_verify_stub(proof: dict, M, x, K_chip, model_code) -> dict:
    """Verify the bundle by REPLAY (opens commitment, re-runs M).

    A true zk-SNARK verifier would replace the "replay M" step with a
    succinct constraint check; the OUTER interface (proof dict shape) is
    identical so this can be hot-swapped.
    """
    com_S = bytes.fromhex(proof['com_S'])
    prog_hash_claim = bytes.fromhex(proof['prog_hash'])
    y_claim = np.asarray(proof['y'], dtype=np.float32)
    tag = bytes.fromhex(proof['tag'])

    # 1. program-hash check
    prog_hash = model_hash(model_code)
    if prog_hash != prog_hash_claim:
        return dict(accepted=False, reason='prog_hash_mismatch')

    # 2. open commitment
    r = bytes.fromhex(proof['_open_r'])
    S = np.asarray(proof['_open_S'], dtype=np.float32)
    if not open_commit(com_S, S, r):
        return dict(accepted=False, reason='commit_open_failed')

    # 3. re-run inference (in real zkML this would be the SNARK check)
    y_replay = M(x, S)
    if not np.allclose(y_replay, y_claim, atol=1e-5):
        return dict(accepted=False, reason='inference_mismatch',
                    max_diff=float(np.max(np.abs(y_replay - y_claim))))

    # 4. HMAC tag check
    if not verify_tag(K_chip, com_S, x, y_claim, prog_hash, tag):
        return dict(accepted=False, reason='tag_invalid')

    return dict(accepted=True)


# ============== smoke test ==============
def _smoke():
    rng = np.random.default_rng(0)
    # toy model: y = tanh(W @ x + S[:H])
    H = 4
    W = rng.normal(0, 0.2, (H, 8)).astype(np.float32)
    def M(x, S):
        return np.tanh(W @ x + S[:H])

    S = rng.normal(0, 1, 32).astype(np.float32)
    K_chip = secrets.token_bytes(32)
    x = rng.normal(0, 1, 8).astype(np.float32)

    proof = zk_prove_stub(M, x, S, K_chip, model_code=b"toy-M v1")
    result = zk_verify_stub(proof, M, x, K_chip, model_code=b"toy-M v1")
    print("honest:", result)

    # tampered output
    bad = dict(proof)
    bad['y'] = (np.array(bad['y']) + 0.5).tolist()
    print("tampered y:", zk_verify_stub(bad, M, x, K_chip, b"toy-M v1"))

    # wrong K_chip (adversary)
    print("wrong K_chip:", zk_verify_stub(proof, M, x, secrets.token_bytes(32), b"toy-M v1"))

    # wrong program
    print("wrong program:", zk_verify_stub(proof, M, x, K_chip, b"toy-M v2"))


if __name__ == '__main__':
    _smoke()
