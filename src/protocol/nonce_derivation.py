"""Nonce-derived sampling plan + nonce embedding.

derive_plan(nonce, n_cpus, n_zones) -> dict with:
  cpu_subset:   list[int]   (4 distinct CPU indices)
  zone_subset:  list[int]   (up to 3 distinct thermal-zone indices)
  core_pairs:   list[tuple] (2 pairs for c2c ping-pong)
  ns_sleep:     int         (nanosleep target ns, 1000..8000)
  ns_count:     int         (4..10)
  tsc_count:    int         (4..10)
  perm:         np.ndarray  (32,) output permutation of physical dims
  _hmac8:       bytes       (truncated HMAC for debugging)

nonce_embedding(nonce, dim=32) -> 32-dim unit-ish float32 (so the classifier
also sees WHICH challenge is being answered, not just the signature).
"""
import hmac
import hashlib
import numpy as np

HMAC_KEY_PLAN  = b"fabriccrypt_nonce_plan"
HMAC_KEY_EMBED = b"fabriccrypt_nonce_embed"


def derive_plan(nonce: bytes, n_cpus: int, n_zones: int) -> dict:
    h = hmac.new(HMAC_KEY_PLAN, nonce, hashlib.sha256).digest()
    rng = np.random.default_rng(np.frombuffer(h[:8], dtype=np.uint64)[0])

    cpu_subset = list(rng.choice(n_cpus, size=min(4, n_cpus), replace=False))
    if n_zones > 0:
        zone_subset = list(rng.choice(n_zones, size=min(3, n_zones), replace=False))
    else:
        zone_subset = []

    core_pairs = []
    for _ in range(2):
        a, b = rng.choice(n_cpus, size=2, replace=False)
        core_pairs.append((int(a), int(b)))

    ns_sleep  = int(1000 + (h[16] | (h[17] << 8)) % 7000)   # 1000..8000 ns
    ns_count  = int(4 + h[18] % 7)                          # 4..10
    tsc_count = int(4 + h[19] % 7)                          # 4..10
    perm32    = rng.permutation(32)

    return {
        "cpu_subset":  [int(x) for x in cpu_subset],
        "zone_subset": [int(x) for x in zone_subset],
        "core_pairs":  core_pairs,
        "ns_sleep":    ns_sleep,
        "ns_count":    ns_count,
        "tsc_count":   tsc_count,
        "perm":        perm32,
        "_hmac8":      h[:8],
    }


def nonce_embedding(nonce: bytes, dim: int = 32) -> np.ndarray:
    """Map nonce to a 32-dim unit-norm vector (scaled), for classifier input."""
    block = b""
    i = 0
    while len(block) < dim * 4:
        block += hmac.new(HMAC_KEY_EMBED, nonce + bytes([i]),
                          hashlib.sha256).digest()
        i += 1
    raw = np.frombuffer(block[:dim * 4], dtype=np.uint32).astype(np.float64)
    v = (raw / 2**32) * 2 - 1
    v = v.astype(np.float32)
    n = float(np.linalg.norm(v)) + 1e-8
    return (v / n).astype(np.float32) * np.sqrt(dim).astype(np.float32) * 0.5


def fresh_nonce(rng: np.random.Generator = None) -> bytes:
    rng = rng or np.random.default_rng()
    return rng.bytes(8)
