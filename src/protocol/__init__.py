"""FabricCrypt nonce-keyed challenge-response protocol.

The audience supplies a 64-bit nonce. The nonce drives:
  - which CPUs to read c-state usage from (4 of N picked from HMAC)
  - which thermal zones to read (up to 3 of available)
  - which core pairs to do c2c ping-pong on (2 pairs)
  - nanosleep target duration (1000..8000 ns)
  - nanosleep burst count (4..10)
  - TSC burst count (4..10)
  - output 32-element permutation

This means a static replay of one recorded signature cannot answer an
arbitrary fresh nonce: the recorded vector was permuted under one plan,
but the challenge will demand consistency with a different plan.

The deterministic plan-consistency verifier in `verifier.py` is the hard
gate. The MLP classifier is a soft diagnostic.
"""
from .nonce_derivation import derive_plan, nonce_embedding, fresh_nonce
from .nonce_signature import NonceSig
from .verifier import plan_consistency_score, gated_accept

__all__ = [
    "derive_plan", "nonce_embedding", "fresh_nonce",
    "NonceSig",
    "plan_consistency_score", "gated_accept",
]
