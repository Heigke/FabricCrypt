"""T2.3 — Multi-round interactive protocol over raw micro-sample series.

Phase 14 protocol was single-round: verifier sends nonce, prover responds
with a 64-dim aggregated vector.  An ML attacker who learns the (nonce →
aggregate) map can forge.

Multi-round protocol (this module):

  R1:  V → P :  nonce
       P → V :  S = [s_1, s_2, ..., s_50]  (50 raw micro-samples; e.g. 50
                consecutive nanosleep tail-latency readings under the
                plan derived from `nonce`).  This is HIGH ENTROPY raw data,
                not a summary — the per-sample jitter is the chip's
                physical fingerprint signal.
       P → V :  commit_S = SHA256("commit-S" || nonce || S_bytes)
                (so prover cannot retro-fit S after seeing the
                verifier's constraints in R2)

  R2:  V → P :  c_1, ..., c_5   five algebraic constraints derived from
                                 nonce; each c_k specifies "compute t_k =
                                 f_k(S)" for a verifier-chosen aggregation
                                 f_k.  Aggregations are:
                                    f_1: median of [s_i : i in subset_1]
                                    f_2: variance of [s_i : i in subset_2]
                                    f_3: lag-1 autocorr of [s_i : i in subset_3]
                                    f_4: count(s_i > tau) for tau, subset_4
                                    f_5: weighted sum  Σ w_i * s_i  for subset_5

                The subsets and weights are SHAKE256(nonce || k || domain)
                so the prover CANNOT predict them before seeing R2.

  R3:  P → V :  t_1, ..., t_5  AND  open(S)   (the prover reveals S,
                verifier recomputes commit_S and checks against R1, then
                recomputes f_k(S) for k=1..5 and checks all match).

  Verification:
       check commit_S consistent
       for each k:  | t_k - f_k(S) | < epsilon_k    (5/5 must pass)
       check f_k(S) statistically matches the per-die distribution
            (use trained classifier on the aggregate sample, OR check
             with reverse-fuzzy-extractor on bit-quantized features).

Threat model improvements over single-round:
   * Adversary must produce S that satisfies FIVE post-hoc-chosen
     constraints, each conditioning on a different subset/weighting.
     An ML attacker who learned the marginal distribution of one
     aggregate (e.g. median nanosleep) cannot satisfy a CONSTRAINT
     derived from a random subset they did not anticipate.
   * Forcing the prover to reveal raw S (instead of a summary) exposes
     the FULL per-die response surface.  Forgery requires emulating
     the chip's full noise process, not just its mean.

Author: Tier-2 FabricCrypt — 2026-06-01
"""
from __future__ import annotations
import hashlib
import numpy as np


# ----------- constraint family -----------
def _shake_int_seq(label: bytes, *parts: bytes, n: int, modulus: int) -> np.ndarray:
    """Pseudo-random int sequence of length n in [0, modulus)."""
    h = hashlib.shake_256(); h.update(label)
    for p in parts:
        h.update(b"|"); h.update(p)
    raw = h.digest(8 * n)
    arr = np.frombuffer(raw, dtype=np.uint64) % modulus
    return arr.astype(np.int64)


def _shake_float_seq(label: bytes, *parts: bytes, n: int, lo: float = -1.0, hi: float = 1.0) -> np.ndarray:
    h = hashlib.shake_256(); h.update(label)
    for p in parts:
        h.update(b"|"); h.update(p)
    raw = h.digest(8 * n)
    arr = np.frombuffer(raw, dtype=np.uint64)
    u = arr.astype(np.float64) / 2**64
    return lo + (hi - lo) * u


def derive_constraints(nonce: bytes, n_samples: int) -> list[dict]:
    """Generate 5 verifier-chosen constraints from the nonce."""
    cons = []
    # c1: median over a 40% subset
    idx1 = np.unique(_shake_int_seq(b"mr-c1-subset", nonce, n=n_samples // 2, modulus=n_samples))
    cons.append(dict(name='median', subset=idx1))
    # c2: variance over a 30% subset (disjoint-ish)
    idx2 = np.unique(_shake_int_seq(b"mr-c2-subset", nonce, n=n_samples // 3, modulus=n_samples))
    cons.append(dict(name='variance', subset=idx2))
    # c3: lag-1 autocorrelation over the first half
    idx3 = np.unique(_shake_int_seq(b"mr-c3-subset", nonce, n=n_samples // 2, modulus=n_samples))
    idx3 = np.sort(idx3)
    cons.append(dict(name='lag1_acf', subset=idx3))
    # c4: count above threshold tau over a 35% subset
    idx4 = np.unique(_shake_int_seq(b"mr-c4-subset", nonce, n=n_samples * 35 // 100,
                                    modulus=n_samples))
    tau_u = _shake_float_seq(b"mr-c4-tau", nonce, n=1, lo=0.2, hi=0.8)[0]
    cons.append(dict(name='count_above_q', subset=idx4, q=float(tau_u)))
    # c5: weighted sum over a 50% subset
    idx5 = np.unique(_shake_int_seq(b"mr-c5-subset", nonce, n=n_samples // 2, modulus=n_samples))
    w = _shake_float_seq(b"mr-c5-weights", nonce, n=len(idx5), lo=-1.0, hi=1.0).astype(np.float64)
    cons.append(dict(name='weighted_sum', subset=idx5, weights=w))
    return cons


def evaluate_constraint(S: np.ndarray, con: dict) -> float:
    S = np.asarray(S, dtype=np.float64)
    sub = S[con['subset']] if len(con['subset']) > 0 else S
    name = con['name']
    if name == 'median':
        return float(np.median(sub))
    if name == 'variance':
        return float(np.var(sub))
    if name == 'lag1_acf':
        x = sub - np.mean(sub)
        d = float(np.dot(x, x)) + 1e-12
        return float(np.dot(x[:-1], x[1:]) / d)
    if name == 'count_above_q':
        # threshold is the q'th quantile of sub itself; this binds against
        # the chip-specific distribution
        tau = float(np.quantile(sub, con['q']))
        return float(np.sum(sub > tau))
    if name == 'weighted_sum':
        return float(np.dot(sub, con['weights']))
    raise ValueError(name)


def commit_samples(nonce: bytes, S: np.ndarray) -> bytes:
    h = hashlib.sha256()
    h.update(b"commit-S|"); h.update(nonce); h.update(b"|")
    h.update(S.astype(np.float32).tobytes())
    return h.digest()


# ----------- protocol orchestration -----------
class MultiRoundProver:
    """Prover side.  Owns the chip access via `sample_callable(nonce, n)`
    which produces an ndarray of n raw micro-samples."""

    def __init__(self, sample_callable):
        self.sample = sample_callable
        self._state = {}

    def round1(self, nonce: bytes, n_samples: int = 50) -> dict:
        S = self.sample(nonce, n_samples).astype(np.float32)
        commit = commit_samples(nonce, S)
        self._state[nonce] = S
        return dict(commit_S=commit, n_samples=int(n_samples))

    def round3(self, nonce: bytes, constraints: list[dict]) -> dict:
        S = self._state.pop(nonce)
        ts = [evaluate_constraint(S, c) for c in constraints]
        return dict(S=S, t=ts)


class MultiRoundVerifier:
    """Verifier side.  Stores the prover's R1 commitment, picks
    constraints in R2, validates in R3."""

    def __init__(self, eps: dict | None = None, n_samples: int = 50):
        self.n_samples = n_samples
        self.eps = eps or dict(median=0.5, variance=0.5,
                               lag1_acf=0.15, count_above_q=2.0,
                               weighted_sum=1.0)
        self._open = {}   # nonce -> (commit_S, constraints)

    def round1_recv(self, nonce: bytes, r1: dict):
        self._open[nonce] = dict(commit_S=r1['commit_S'])

    def round2_send(self, nonce: bytes) -> list[dict]:
        cons = derive_constraints(nonce, self.n_samples)
        self._open[nonce]['constraints'] = cons
        return cons

    def round3_verify(self, nonce: bytes, r3: dict) -> dict:
        st = self._open.pop(nonce)
        cons = st['constraints']
        S = np.asarray(r3['S'], dtype=np.float32)
        t_claimed = list(r3['t'])

        # commitment check
        commit_now = commit_samples(nonce, S)
        if commit_now != st['commit_S']:
            return dict(accepted=False, reason='commit_mismatch')

        # constraint check
        fails = []
        for c, t_c in zip(cons, t_claimed):
            t_true = evaluate_constraint(S, c)
            eps = self.eps[c['name']]
            if abs(t_c - t_true) > eps:
                fails.append(dict(name=c['name'], t_claimed=float(t_c),
                                  t_true=float(t_true), eps=eps))
        if fails:
            return dict(accepted=False, reason='constraint_violation',
                        fails=fails)
        return dict(accepted=True, n_constraints=len(cons))


# ============== smoke test ==============
def _smoke():
    rng = np.random.default_rng(0)
    # honest chip: produces S from a fixed-distribution noise
    def sample_chip(nonce, n):
        seed_int = int.from_bytes(hashlib.sha256(b"chip1" + nonce).digest()[:8], 'little')
        r = np.random.default_rng(seed_int)
        return r.normal(0, 1, n).astype(np.float32)

    prover = MultiRoundProver(sample_chip)
    verifier = MultiRoundVerifier(n_samples=50)

    nonce = rng.bytes(8)
    r1 = prover.round1(nonce); verifier.round1_recv(nonce, r1)
    cons = verifier.round2_send(nonce)
    r3 = prover.round3(nonce, cons)
    result = verifier.round3_verify(nonce, r3)
    print("honest:", result)

    # adversary: tampers samples after seeing constraints (impossible
    # because commit is sent in R1, but try anyway)
    nonce2 = rng.bytes(8)
    r1 = prover.round1(nonce2); verifier.round1_recv(nonce2, r1)
    cons = verifier.round2_send(nonce2)
    r3 = prover.round3(nonce2, cons)
    r3['S'] = r3['S'] + 0.5  # tamper after commit
    result = verifier.round3_verify(nonce2, r3)
    print("post-commit tamper:", result['accepted'], result.get('reason'))

    # forgery: adversary sends wrong S that has different stats
    nonce3 = rng.bytes(8)
    fake_S = rng.normal(0, 1, 50).astype(np.float32)
    r1_fake = dict(commit_S=commit_samples(nonce3, fake_S), n_samples=50)
    verifier.round1_recv(nonce3, r1_fake)
    cons = verifier.round2_send(nonce3)
    ts = [evaluate_constraint(fake_S, c) for c in cons]
    r3 = dict(S=fake_S, t=ts)
    result = verifier.round3_verify(nonce3, r3)
    # This will ACCEPT if we don't separately classify S as coming from
    # the right chip.  The multiround protocol enforces COMMITMENT to S
    # and CONSISTENCY of the claimed aggregates with S; it does NOT, on
    # its own, prove that S came from the right chip.  That's the job of
    # the classifier / fuzzy extractor LAYERED on top.
    print("naïve forgery (matched aggregates, wrong chip):", result['accepted'],
          "→ expected: still ACCEPT (proves commit/consistency only)")


if __name__ == '__main__':
    _smoke()
