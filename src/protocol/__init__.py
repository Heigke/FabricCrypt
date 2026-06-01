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

# -------------------------------------------------------------------------
# Tier 2 cryptographic hardening (additive — bumps unprotected security
# from ~2^30-2^40 to ~2^60-2^80).  See docs/PROTOCOL.md "Tier 2".
#
#   reverse_fuzzy         — Van Herrewege FC'12 reverse fuzzy extractor
#                           (BCH code-offset; helper P kept PRIVATE on verifier
#                           — eliminates helper-data leakage attack)
#   controlled_puf        — Suh-Devadas DAC'07 controlled-PUF wrap
#                           (SHAKE256 H_in/H_out with strict domain separation;
#                           defeats Ruehrmair CCS'10 ML modeling attack)
#   multiround_protocol   — 3-round commit/challenge/open (50-sample raw S +
#                           5 SHAKE-derived constraints; forces full per-die
#                           noise emulation by adversary)
#   zk_inference_binding  — Pedersen-style commitment + HMAC inference tag
#                           (interface-compatible with future zk-SNARK swap-in)
# -------------------------------------------------------------------------
