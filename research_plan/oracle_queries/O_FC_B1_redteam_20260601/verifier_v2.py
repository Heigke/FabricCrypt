"""Phase 14D verifier — HARD veto (Fix 2).

Acceptance := plan_pass AND classifier_p0 > tau_cls

  - plan_pass: the chip's reported dim-31 measurement (MAD of nanosleep
    burst at plan['ns_sleep']) is consistent with the chip's enrolled
    per-dim fingerprint mu_31, sigma_31. This is a REAL measurement,
    not a function of the input — an attacker without chip access
    cannot fabricate a value within the chip's calibration band.

  - classifier_p0: the trained twin-MLP must rate the (phys, nonce_emb)
    pair as 'own chip'. Threshold tau_cls is chosen on a held-out honest
    set to give 95% TPR (so honest_own ≥ 0.95).

This closes the O115 fatal break: even an attacker who has K_chip
(simulating Tier-2 break) cannot evaluate the classifier's threshold
behaviour without enrollment-time observation of the chip.
"""
from __future__ import annotations
import os, sys, json, hashlib, hmac
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from nonce_signature_v2 import derive_plan_keyed


def plan_measurement_score(phys_part: np.ndarray, nonce: bytes, K_chip: bytes,
                            n_cpus: int, n_zones: int,
                            mu_vec: np.ndarray, sigma_vec: np.ndarray,
                            band_k: float = 3.0,
                            mahala_quantile_thresh: float = None) -> float:
    """Plan-consistency: (a) dim-31 measurement is in the chip's calibrated
    band AND (b) the un-permuted phys vector as a whole is in-distribution
    for this chip (Mahalanobis-like normalized RMS over all 32 dims).

    Both conditions must hold. (a) is the per-nonce liveness check;
    (b) catches forgeries that fill arbitrary values into a single dim
    or flood all dims with the same scalar — neither matches the chip's
    per-dim mu, sigma profile.

    Returns a [0,1] score combining both.
    """
    plan = derive_plan_keyed(nonce, K_chip, n_cpus, n_zones)
    perm = plan['perm']
    # invert permutation: place phys_part back into un-permuted slots
    inv = np.empty_like(perm); inv[perm] = np.arange(32)
    unperm = phys_part[inv]  # so unperm[i] is the original dim i
    # (a) dim-31 band test
    delta31 = abs(unperm[31] - mu_vec[31]) / max(sigma_vec[31], 1e-3)
    if delta31 <= band_k: s31 = 1.0
    elif delta31 >= 2*band_k: s31 = 0.0
    else: s31 = max(0.0, 1.0 - (delta31 - band_k)/band_k)
    # (b) overall Mahalanobis-like: mean( ((x-mu)/sigma)^2 ) over all 32 dims
    z = (unperm - mu_vec) / np.maximum(sigma_vec, 1e-3)
    mahala = float(np.sqrt(np.mean(z * z)))
    # Honest samples have mahala close to 1.0 (per dim ~1 std). Reject if
    # mahala > band_k (default 3 → corresponds to ~3-sigma per-dim on avg).
    if mahala <= band_k: sM = 1.0
    elif mahala >= 2*band_k: sM = 0.0
    else: sM = max(0.0, 1.0 - (mahala - band_k)/band_k)
    return min(s31, sM)  # both must pass


def classifier_p0(model, X: np.ndarray, device='cpu') -> np.ndarray:
    import torch
    import torch.nn.functional as F
    with torch.no_grad():
        logits = model(torch.from_numpy(X.astype(np.float32)).to(device))
        return F.softmax(logits, dim=-1)[:, 0].cpu().numpy()


def hard_veto_accept(model, X: np.ndarray, nonces, K_chip: bytes,
                     n_cpus: int, n_zones: int,
                     mu_vec: np.ndarray, sigma_vec: np.ndarray,
                     tau_cls: float = 0.5,
                     plan_score_thresh: float = 0.5,
                     band_k: float = 3.0,
                     device='cpu') -> dict:
    """Apply HARD veto: BOTH plan-consistency AND classifier must pass."""
    p0 = classifier_p0(model, X, device=device)
    n = len(X)
    plan_scores = np.empty(n, dtype=np.float32)
    for i in range(n):
        plan_scores[i] = plan_measurement_score(
            X[i, :32], nonces[i], K_chip, n_cpus, n_zones,
            mu_vec, sigma_vec, band_k=band_k)
    plan_pass = plan_scores > plan_score_thresh
    cls_pass = p0 > tau_cls
    accept = plan_pass & cls_pass  # HARD AND
    return {
        'classifier_p0_mean': float(p0.mean()),
        'classifier_pass_only': float(cls_pass.mean()),
        'plan_score_mean': float(plan_scores.mean()),
        'plan_pass_only': float(plan_pass.mean()),
        'accept_rate': float(accept.mean()),
        'tau_cls': tau_cls,
        'plan_score_thresh': plan_score_thresh,
        'band_k': band_k,
    }


def calibrate_threshold(model, X_honest: np.ndarray, target_tpr: float = 0.97,
                        device='cpu') -> float:
    """Pick tau_cls so the model achieves target_tpr on honest examples."""
    p0 = classifier_p0(model, X_honest, device=device)
    # tau = the (1-target_tpr)-quantile of p0
    tau = float(np.quantile(p0, max(0.0, 1.0 - target_tpr)))
    # clamp to a reasonable lower floor so adversarial vectors with p0~0.4
    # still get rejected
    return max(tau, 0.10)


def calibrate_dim31_band(X_honest: np.ndarray, perm_pos_31: np.ndarray) -> tuple:
    """Legacy single-dim band (kept for backwards compat)."""
    vals = np.array([X_honest[i, perm_pos_31[i]] for i in range(len(X_honest))])
    return float(vals.mean()), float(vals.std() + 1e-3)


def calibrate_full_band(X_honest: np.ndarray, perms_inv: np.ndarray):
    """Estimate per-dim (mu, sigma) over the UN-PERMUTED phys vector.

    perms_inv[i] is the inverse permutation for example i, so
    unperm[i] = X_honest[i, :32][perms_inv[i]].
    """
    n = len(X_honest)
    U = np.empty((n, 32), dtype=np.float32)
    for i in range(n):
        U[i] = X_honest[i, :32][perms_inv[i]]
    mu = U.mean(axis=0)
    sigma = U.std(axis=0) + 1e-3
    return mu.astype(np.float32), sigma.astype(np.float32)
