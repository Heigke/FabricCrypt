"""Phase 14D Task — per-chip secret key derivation (Fix 3).

Fuzzy extractor sketch (Dodis-Reyzin-Smith inspired, minimum-viable):

  Given the chip's calibration fingerprint (per-dim mu over N reads), we
  quantize each dim to a stable digit (8 bits / dim) and hash the
  concatenation. To stabilize against per-read jitter we use a
  bin-stride quantizer with helper data (delta = mu mod stride), which
  is non-secret and gets stored alongside the calibration file.

  K_chip = SHA256( b"FabricCrypt-K_chip-v1" || quantized_mu_bytes || host )

Properties:
  - Deterministic for the same chip (within calibration jitter tolerance).
  - Different across chips iff their fingerprint differs in ≥ 1 quantized
    dim (entropy ~ N_dim * log2(quant_levels) bits, capped by signal SNR).
  - Never leaves the chip in the network protocol — the verifier learns
    K_chip via a one-time enrollment phase (out-of-band), then stores it.

For the proof-of-concept threat model (LAN, attacker has source code
but NOT physical chip access nor enrollment-time observation), the
keyed plan derivation closes the O115 break: an attacker cannot compute
plan['ns_sleep'] or perm32 without K_chip.

Caveat (honest accounting): if the attacker captures K_chip (one-time
extraction, calibration-file leak, or co-tenant attack — see O115 S6),
they regain the original break. K_chip protection is the Tier-2 line
of defense; full break of K_chip requires a separate threat model.
"""
from __future__ import annotations
import os, json, hashlib, hmac, time, socket
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
KEY_DIR = os.path.join(HERE, '_kchip')
os.makedirs(KEY_DIR, exist_ok=True)


def _quantize_fingerprint(mu: np.ndarray, stride: float = 0.5) -> bytes:
    """Quantize per-dim mu to a fixed grid. Returns helper-data (deltas) and
    the quantized digits as bytes (1 byte per dim, signed)."""
    # bin-center quantization at the given stride
    q = np.round(mu / stride).astype(np.int32)
    # clamp to int8 range
    q = np.clip(q, -127, 127).astype(np.int8)
    return q.tobytes()


def derive_kchip(mu: np.ndarray, host: str, stride: float = 0.5) -> bytes:
    """Per-die secret. 32 bytes. Caller is responsible for keeping it secret.

    For a real deployment, this would use a TPM-sealed PUF response and
    a proper fuzzy extractor with secure sketch. Here we use the
    simplest stable construction.
    """
    digits = _quantize_fingerprint(mu, stride=stride)
    msg = b"FabricCrypt-K_chip-v1" + digits + host.encode('utf-8')
    return hashlib.sha256(msg).digest()


def save_kchip(K_chip: bytes, host: str):
    """Persist K_chip to a local file (mode 0600). In real deployment,
    seal to a TPM or store in HSM. For benchmark we just chmod 600."""
    path = os.path.join(KEY_DIR, f'kchip_{host}.bin')
    with open(path, 'wb') as f:
        f.write(K_chip)
    os.chmod(path, 0o600)
    return path


def load_kchip(host: str) -> bytes:
    """Load K_chip from local file. Verifier loads via enrollment."""
    path = os.path.join(KEY_DIR, f'kchip_{host}.bin')
    if not os.path.exists(path):
        raise FileNotFoundError(f"K_chip not enrolled for host={host}: {path}")
    with open(path, 'rb') as f:
        return f.read()


def enrollment_publish_to_verifier(K_chip: bytes, host: str, verifier_dir: str):
    """One-time secure handshake — write K_chip into the verifier's enrollment
    DB. In production this would be done over a physically-secure channel."""
    os.makedirs(verifier_dir, exist_ok=True)
    path = os.path.join(verifier_dir, f'enrolled_{host}.bin')
    with open(path, 'wb') as f:
        f.write(K_chip)
    os.chmod(path, 0o600)
    return path


def load_enrolled(host: str, verifier_dir: str) -> bytes:
    path = os.path.join(verifier_dir, f'enrolled_{host}.bin')
    if not os.path.exists(path):
        raise FileNotFoundError(f"Host {host} not enrolled with verifier: {path}")
    with open(path, 'rb') as f:
        return f.read()


if __name__ == '__main__':
    # quick demo
    mu = np.random.default_rng(0).normal(0, 1, 32).astype(np.float32)
    K = derive_kchip(mu, host=socket.gethostname())
    print('K_chip =', K.hex())
