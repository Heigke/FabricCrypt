"""Deterministic plan-consistency verifier — the HARD gate.

A NonceSig response under nonce N is a 64-dim vector whose first 32 dims
are the physical signature permuted by plan(N)['perm'], with position 31
(pre-perm) holding the nonce-derived ns_sleep value (1000..8000).

A verifier that knows N can invert the permutation, locate ns_sleep, and
compare to the expected log-scaled value. A static-replay adversary
(same vector for every N) WILL FAIL this check because plan(N) differs
per nonce.
"""
import numpy as np

from .nonce_derivation import derive_plan


def plan_consistency_score(phys_part: np.ndarray, nonce: bytes,
                            n_cpus: int, n_zones: int,
                            tolerance: float = 0.15) -> float:
    """Score in [0, 1]: 1 = perfect plan match, 0 = total mismatch.

    The reference value lives at index 31 BEFORE the permutation; after
    permutation, it lands at output position perm.index(31). We invert
    the permutation, fetch the observed value, log-scale it, and compare
    to the expected log(ns_sleep) value.
    """
    plan = derive_plan(nonce, n_cpus, n_zones)
    perm = plan["perm"]
    pos = int(np.where(perm == 31)[0][0])
    observed = float(phys_part[pos])
    expected = float(np.log1p(plan["ns_sleep"] * 1e-3))
    diff = abs(observed - expected)
    return float(max(0.0, 1.0 - diff / tolerance))


def gated_accept(p0_arr: np.ndarray, plan_scores: np.ndarray,
                 p0_thresh: float = 0.5, plan_thresh: float = 0.5) -> np.ndarray:
    """Final accept = classifier-says-own AND plan-consistency passes.

    In our reference implementation we treat plan-consistency as the HARD
    gate (deterministic, near-binary) and use the classifier output only
    as a diagnostic. See `attacks.py` for the gate evaluation.
    """
    return ((p0_arr > p0_thresh) & (plan_scores > plan_thresh)).astype(np.float32)
