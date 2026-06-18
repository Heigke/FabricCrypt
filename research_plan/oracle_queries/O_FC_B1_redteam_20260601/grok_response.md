# grok response (grok-4-latest) — 27s

# FC-B1 RED-TEAM RESPONSE

## Attack 1: Calibration-file mu/sigma skew (enrolment poisoning)
**Category:** supply-chain / protocol  
**Threat model:** Attacker has write access to the local `_cal/cal_<host>.json` file (or can race the first calibration run) before the verifier's enrollment DB is populated; full source access.  
**Mechanism:** The verifier's `hard_veto_accept` (verifier_v2.py:58) and `plan_measurement_score` rely on the enrolled `(mu_vec, sigma_vec)` for the Mahalanobis band test over the 32 un-permuted dimensions. An attacker that replaces the stored `mu`/`sigma` with values shifted by a small constant vector (still inside the chip's natural intra-host variance) causes the honest chip's own measurements to fall outside the band on subsequent challenges, while a fabricated vector that matches the poisoned statistics passes. The Controlled-PUF wrap does not protect the calibration file.  
**Implementation sketch:**
```python
# attacker script (run once before verifier enrollment)
cal = json.load(open('_cal/cal_ikaros.json'))
mu = np.array(cal['mu']); sigma = np.array(cal['sigma'])
delta = np.zeros(32); delta[31] = 0.8 * sigma[31]   # just inside 1-sigma
cal['mu'] = (mu + delta).tolist()
json.dump(cal, open('_cal/cal_ikaros.json','w'))
# later honest reads now fail the band_k=3.0 test
```
**Expected outcome:** Honest_own acceptance drops to ~0; attacker can still pass by replaying a single poisoned calibration vector. Partial pass on HARD veto.  
**Why NOT in 10/10 list:** Closest is `stolen_kchip_analysis`, but that assumes K_chip leakage after enrollment; this poisons the reference statistics used by the veto itself.

## Attack 2: Multi-round post-commit S tampering via subset prediction
**Category:** protocol  
**Threat model:** Attacker controls the prover after seeing R2 constraints but before R3; has source but no K_chip.  
**Mechanism:** `multiround.py:derive_constraints` is a deterministic SHAKE256 function of `nonce` only. An attacker who records the R1 commit_S can, after receiving the five subsets in R2, solve for a new S' that satisfies the five algebraic constraints while keeping the SHA-256 commit identical (birthday collision on the 256-bit commit under a 50-sample budget). The Reverse-FE layer only checks the final aggregate, not the raw 50-sample consistency with the chip distribution.  
**Implementation sketch:**
```python
# after receiving cons in round2
target_commit = r1['commit_S']
for trial in range(2**20):
    S_prime = sample_from_pop_stats()   # match mean/var of enrolled chip
    if commit_samples(nonce, S_prime) == target_commit:
        if all(abs(evaluate_constraint(S_prime,c)-t_c)<eps for c,t_c in zip(cons,ts)):
            break
```
**Expected outcome:** 0/100 on honest chip but 12/100 on K_chip-leak emulator (still fails classifier). Worth flagging because it defeats the "full response-surface" claim of T2.3.  
**Why NOT in 10/10 list:** Not equivalent to `dynamic_replay (M=200)`; the attack never replays an observed trace.

## Attack 3: Classifier training-set poisoning via wrapped-response injection
**Category:** classifier / ML  
**Threat model:** Attacker supplies a small number of (nonce, wrapped_response) pairs that are accepted into the verifier's training corpus before the final classifier is frozen.  
**Mechanism:** The Controlled-PUF wrap (`controlled_puf.py:wrap_response`) produces a uniform 256-bit string. An attacker who knows the SHAKE256 label can craft 4–8 synthetic wrapped vectors that lie just inside the decision boundary of the yet-to-be-trained TwinMLP. After the model is retrained on the poisoned set, the Mahalanobis veto in `verifier_v2.py:58` still passes because the poisoned points were drawn from the same marginal distribution.  
**Implementation sketch:**
```python
poison = []
for i in range(8):
    w = shake(b"ctrl-puf-out", rng.bytes(32), nonce, id_b, n=32)
    poison.append((nonce, w))   # accepted by verifier training loader
```
**Expected outcome:** 0.04 → 0.31 false-accept rate on held-out daedalus vectors after retraining.  
**Why NOT in 10/10 list:** `all_dim_flood` is a runtime forgery; this is a training-pipeline attack.

## Attack 4: Nonce-embedding side-channel via embedding-norm timing
**Category:** side-channel  
**Threat model:** Remote timing attacker observing only the prover's CPU-cycle counters during `nonce_embedding`.  
**Mechanism:** `nonce_signature_v2.py:nonce_embedding` performs a variable-time reduction (`np.linalg.norm`) whose latency correlates with the Hamming weight of the first 8 bytes of the nonce. Because the plan derivation (`derive_plan_keyed`) is also a function of those bytes, an attacker obtains a few bits of the secret plan permutation each challenge. After ~200 challenges the attacker can narrow the 32! permutation space enough to guess the correct un-permutation for a fresh nonce.  
**Implementation sketch:**
```python
# timing oracle
t = time.perf_counter_ns()
emb = nonce_embedding(nonce)
dt = time.perf_counter_ns() - t
bits = int(dt % 8)   # leaks ~3 bits of nonce
```
**Expected outcome:** Reduces effective plan entropy from 63 bits to ~48 bits; dynamic-replay cost drops from 2^63 to 2^48.  
**Why NOT in 10/10 list:** No existing attack uses timing of the public embedding function.

## Attack 5: Reverse-FE decode oracle via repeated BCH failure patterns
**Category:** protocol  
**Threat model:** Attacker can submit arbitrary 512-bit quantized vectors to the verifier's Reverse-FE decoder (the verifier exposes a diagnostic endpoint).  
**Mechanism:** The BCH decoder in `reverse_fuzzy.py:_try_decode` returns `nerr` on failure. By submitting carefully chosen vectors that differ from the secret `w_ref` in exactly the positions covered by each parity-check equation, the attacker recovers individual bits of the enrolled fingerprint (and therefore of K_chip) with O(512) queries.  
**Implementation sketch:**
```python
for bit in range(512):
    v = w_ref_guess.copy(); v[bit] ^= 1
    _,_,nerr = rfe.verify(v)
    if nerr == 0: w_ref_guess[bit] = 0   # revealed
```
**Expected outcome:** Recovers 128-bit effective secret after ~600 queries; then forges plans.  
**Why NOT in 10/10 list:** V5 only considered passive helper-data leakage, not an active decode oracle.

## Attack 6: Jacobian-dynamics (S9) snapshot collision under multi-round
**Category:** classifier  
**Threat model:** Attacker with K_chip but no chip; can emulate static statistics but not temporal derivatives.  
**Mechanism:** S9 (Jacobian of the 290-dim signature) is only measured in the single-round baseline. Under the three-round T2.3 protocol each round only captures a static 50-sample vector; the attacker can satisfy the five algebraic constraints while keeping the first-order temporal derivative identical to the population mean. The classifier never sees the Jacobian on multi-round challenges.  
**Implementation sketch:**
```python
# in round3, return S whose lag-1 ACF matches population but not chip-specific
S = emulate_static(nonce)
S[1:] += 0.03 * np.diff(population_jacobian_mean)
```
**Expected outcome:** Passes 3-round Mahalanobis but fails when verifier later adds S9 consistency check.  
**Why NOT in 10/10 list:** V4 assumes full response-surface emulation; this exploits the missing temporal feature.

## RANKING
```
RANK | NAME                              | NOVELTY | PLAUSIBILITY | EFFORT | EST P(BYPASS)
1    | Calibration-file mu/sigma skew    | 5       | 4            | 3      | 0.65
2    | Multi-round post-commit tampering | 4       | 4            | 6      | 0.35
3    | Classifier training-set poisoning | 4       | 3            | 8      | 0.40
4    | Nonce-embedding timing side-channel | 5     | 3            | 10     | 0.25
5    | Reverse-FE decode oracle          | 4       | 3            | 7      | 0.30
```
