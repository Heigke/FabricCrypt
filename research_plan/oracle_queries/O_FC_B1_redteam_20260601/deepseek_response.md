# deepseek response (deepseek-reasoner) — 55s

# FC-B1 RED-TEAM RESPONSE

## 1. Attack: Enrolment Poisoning via Calibration File Injection

**Category:** supply-chain / enrolment

**Threat model:**  
Attacker has write access to the calibration file (`_cal/cal_{host}.json`) before the verifier's enrolment phase. This could be achieved via a compromised CI/CD pipeline, a malicious insider at the manufacturing stage, or a root-level adversary during initial setup. The verifier trusts the calibration output blindly.

**Mechanism:**  
In `nonce_signature_v2.py`, the `_maybe_calibrate` method computes per-dim `mu` and `sigma` from 60 random-plan reads, then derives `K_chip = derive_kchip(mu, host)`. Both `mu` and `sigma` are stored in a JSON file. An attacker who can pre-populate this file with crafted values (e.g., `mu` chosen to be close to the peer chip's fingerprint, `sigma` inflated to widen the acceptance band) can cause two effects:

1. **K_chip backdoor:** By setting `mu` such that `derive_kchip` produces a *known* K_chip (e.g., all zeros), the attacker gains full plan-derivation control without needing chip access.
2. **Classifier poisoning:** The verifier trains its classifier on signatures produced using the poisoned K_chip (Fix 3 in v2.1). After poisoning, honest signatures from the real chip will appear anomalous, while forged signatures crafted by the attacker will be accepted.

**Implementation sketch:**  
```python
# Attacker crafts a calibration file for host "ikaros" that backdoors K_chip
def poison_calibration(host, target_kchip_bytes):
    # Choose a mu vector that when quantized hashes to target_kchip
    # Brute-force over mu values (in practice, find preimage via gradient search)
    # For simplicity, we assume the attacker can set mu to all zeros -> all mu quantized to 0 -> K_chip = SHA256("...||0*32 bytes||host")
    mu = np.zeros(32, dtype=np.float32)
    sigma = np.ones(32, dtype=np.float32) * 100.0   # huge sigma -> accept almost any
    cal_data = {'mu': mu.tolist(), 'sigma': sigma.tolist(), 'host': host, ...}
    with open(f'_cal/cal_{host}.json', 'w') as f:
        json.dump(cal_data, f)
```
Then the attacker can compute `K_chip = derive_kchip(mu, host)` and generate valid plans/responses.

**Expected outcome:**  
- If sigma is inflated, the plan-score check (Mahalanobis) will accept almost any measurement because the z-scores are tiny.  
- If K_chip is backdoored, the attacker can forge both plan and classifier inputs (bypassing hard veto).  
- This attack would pass the HARD veto because both plan_score and classifier p0 can be made arbitrarily high.

**Why NOT in existing list:**  
The 10 attacks focus on replay, forgery, and stolen K_chip. Enrolment poisoning is a *pre-enrolment* supply-chain attack not considered. V0–V7 include K_chip brute-force and leak, but not *manipulating* the enrolment data to produce a known K_chip. This is a novel vector that violates the implicit trust in calibration.

---

## 2. Attack: Time-of-Check Time-of-Use (TOCTOU) on Plan Derivation

**Category:** protocol (timing / race)

**Threat model:**  
Attacker has control over a network path with sub-millisecond latency (e.g., co-located LAN attacker). The verifier sends the nonce, the prover derives the plan, measures, and returns the response. The attacker can intercept the nonce, **compute the plan** instantly (if K_chip is known via prior leak) or, more interestingly, if the attacker does not have K_chip, they can still exploit a race: the attacker sends the nonce to the verifier and observes the plan-derivation computation on the verifier side. If the plan derivation is computed *before* the response is sent, the attacker can read the plan from the verifier's memory or a side-channel.

**Mechanism:**  
In the verifier (pseudocode in §5.4), `VERIFY` calls `PLAN_DERIVE(audience_secret, N)` before evaluating the response. If the attacker can observe the plan bits (e.g., via cache timing on a shared hypervisor, or by reading the audience_secret in a multi-tenant environment), they can pre-compute the expected phys values for that plan and fabricate a matching response. The attack exploits the fact that the plan is deterministic and *computed before* the response is required; if the K_chip leaks or if the audience_secret is compromised, the plan becomes known. But even without secret leakage, a TOCTOU race exists: the verifier's plan computation is not secret-gated; only the prover's plan computation is K_chip-keyed. However, in Tier-1, the verifier also computes plan from K_chip? Actually in Tier-1, the verifier has K_chip enrolled, so it computes `SHAKE256(K_chip||domain||nonce)`. But in Tier-1, the plan derivation on verifier side uses the same K_chip; if an attacker on the verifier side can observe the plan (e.g., via memory snapshot), they can forge.

**Implementation sketch:**  
```python
# Attacker on verifier host (co-tenant) reads process memory after VERIFY call
# Using /proc/pid/mem, find the plan struct
plan_addr = find_symbol("plan")
plan_bytes = read_memory(plan_addr, 128)  # extract plan fields
# Then craft response that matches plan (e.g., set dims to expected values)
forged_phys = plan_to_expected(plan)
response = pack(nonce, forged_phys, attacker_embedding)
send_to_verifier(response)
```

**Expected outcome:**  
If the attacker can read the plan before the prover's response arrives, they can fabricate a perfect match. The plan-consistency check passes (since they know the plan), and the classifier can be bypassed by also emulating the chip's distribution (requires knowing the enrolled mu,sigma). However, even without full fingerprint knowledge, the attacker can replay a previously recorded signature that was generated under a *different* plan; but with plan knowledge, they can adjust to match. This attack is subtle: it requires either K_chip leak or a verifier-side side-channel. It is different from replay because the plan is *not replayed*; it's computed fresh.

**Why NOT in existing list:**  
Existing attacks assume the adversary cannot see the plan before responding. The dynamic replay attack uses a library of recorded pairs, not real-time extraction of the plan. TOCTOU on the verifier's plan derivation is a new timing-based attack.

---

## 3. Attack: Side-Channel on K_chip Derivation Timing

**Category:** side-channel

**Threat model:**  
Attacker has limited code execution on the enrolling machine (e.g., a co-tenant process that can run alongside calibration). The attacker can measure the timing of `derive_kchip` or the BCH encoding in the reverse fuzzy extractor during enrolment.

**Mechanism:**  
In `key_derivation.py`, `derive_kchip` performs a quantization `np.round(mu / stride)` followed by a SHA256 hash. The quantization step involves a division and rounding, which may have variable timings based on mu values (e.g., due to floating-point operations on subnormal numbers). An attacker who can run a timing loop while calibration is happening can infer the mu values bit-by-bit. Similarly, in `reverse_fuzzy.py`, the `_random_codeword` uses `secrets.token_bytes`, but the BCH encode and decode timings may leak Hamming weight of the codeword, potentially leaking `w_ref` bits. Over many enrolment attempts (if the attacker can trigger re-calibration), they can reconstruct the chip fingerprint and thus K_chip.

**Implementation sketch:**  
```python
# Attacker process busy-waits on /proc/self/sched for calibration process's CPU time
# Alternatively, use performance counters
# For each quantized bin, measure the time of the round operation
# Collect many samples to isolate timing signal
```
Then use differential analysis to recover mu values.

**Expected outcome:**  
If successful, the attacker learns the chip's fingerprint and can derive K_chip. This leads to full plan forgery. The attack might require many calibration repetitions and high-resolution timers, but on a co-tenant system this is plausible.

**Why NOT in existing list:**  
The existing adversaries include V4 (K_chip leak via file read) and V1 (fingerprint brute-force), but not timing side-channel during enrolment. V6 (relay) is hardware-distance, not software timing. The O115 break was about a logic bug, not side-channel.

---

## 4. Attack: BCH Decoding Oracle (Reverse Fuzzy Extractor Membership Inference)

**Category:** ML / cryptographic oracle

**Threat model:**  
Attacker can send many `w_noisy` vectors to the verifier and observe the accept/reject outcome (a binary oracle). The reverse fuzzy extractor's verification (`reverse_fuzzy.py`) returns `(accepted, K_rec, ham)` but only `accepted` is observable to the prover (since K_rec is kept secret). This oracle leaks whether the Hamming distance from the enrolled `w_ref` is ≤ t. The attacker can use this to perform a binary search for each bit of `w_ref`.

**Mechanism:**  
In `ReverseFuzzyExtractor.verify`, acceptance occurs iff `hamming(w_ref, w_noisy) <= t` and the BCH decoding succeeds. By flipping a single bit of a known `w_noisy` and observing acceptance change, the attacker can deduce whether that bit was flipped into or out of the correction radius. Over many queries, they can reconstruct `w_ref` bit by bit. Since `w_ref` is a 256-bit (or 512-bit) binary string, each query gives ~1 bit of information. With t around 48 (for m=9), the oracle is relatively tolerant to errors, so the attacker needs ~2*t+1 queries per bit? Actually they can do a binary search by setting a candidate `w_noisy` and flipping bits until rejection. The oracle is cheap (<1 ms) so 256*? queries are feasible.

**Implementation sketch:**  
```python
# Assume attacker knows the verifier's address and can send verification requests
def recover_w_ref(verifier, initial_w, t):
    w = initial_w.copy()
    # Use gradient: for each bit i, toggle it and see if acceptance changes
    # More sophisticated: start from all zeros, then add bits that increase acceptance
    for i in range(len(w)):
        w[i] ^= 1
        if verifier.verify(w).accepted:
            w[i] ^= 1  # revert
    # After many passes, converge to w_ref (or a bit string within t)
    return w
```

**Expected outcome:**  
The attacker recovers `w_ref`, which is the quantized chip fingerprint. With `w_ref`, they can compute the same `K` (since `K = SHA256("rfe-secret-v1|" + c)` and `c = w_ref XOR P`, but P is not known to attacker? Wait, the attacker does not have P. But they can now compute `K`? Actually the attacker only knows `w_ref`. Without `P`, they cannot get `c`. However, they can enroll a new verification with a different K_chip? But the verifier uses the original `P`. If the attacker has `w_ref`, they can compute `w_ref XOR P`? No, P is secret. But they can impersonate the chip by sending `w_noisy = w_ref` (exact match) and get accepted. That already defeats the FE. The attacker can then forge any future challenge by sending a noisy version within t. Since the FE is used for identity verification, having `w_ref` gives full impersonation capability.

**Why NOT in existing list:**  
The existing V5 (helper-data leakage) is eliminated by RFE, but the RFE itself introduces a new oracle. This is a classic fuzzy-extractor oracle attack (Dodis et al. 2008 note this for public FE; for reverse FE it's the verifier that is the oracle). The adversary vectors V0-V7 do not include a verification oracle attack.

---

## 5. Attack: Classifier Confidence Score as Membership Inference Oracle

**Category:** ML (membership inference)

**Threat model:**  
Attacker can query the verifier with many (nonce, response) pairs and observe the `classifier_p0` score (the probability that the response belongs to the target chip). This is returned in the acceptance decision or via timing side-channel.

**Mechanism:**  
The verifier's classifier is a twin-MLP trained on the enrolled chip's signatures. The classifier is fixed post-enrolment. An attacker can generate random (nonce, phys, emb) vectors and measure `p_own`. If `p_own` is high, the input is likely on the manifold of the target chip. Over many queries, the attacker can learn the support of the chip's distribution and eventually synthesize vectors that have `p_own > 0.5` but are not actually from the chip. This is a classic black-box adversarial attack: train a generative model using the black-box classifier as a discriminator.

**Implementation sketch:**  
```python
# Use GAN-like approach: generator G(z) -> signature, discriminator D = verifier.classifier
# Train G to minimize L = -log(D(G(z)))  (adversarial to maximize p_own)
# Since we have black-box access, we can use evolutionary strategies or finite differences
# After enough queries, G can produce signatures that fool the classifier
```
The classifier is not robust to adversarial perturbation because it was trained on a small dataset (n=10 per host). Its decision boundary is likely non-smooth.

**Expected outcome:**  
After ~10^4 queries (feasible on LAN), the attacker can generate a forged signature that achieves `p_own > 0.5` on a random nonce. However, the plan-consistency check still requires the phys vector to match the plan-derived expectation for that specific nonce, which the generative model does not satisfy. So the attack fails the hard veto unless the attacker also knows the plan (requires K_chip). But the attack still reveals the classifier's vulnerability: if the adversary had K_chip, they could combine with this method. It is worth flagging as a latent risk.

**Why NOT in existing list:**  
The existing attacks include ML modeling on the PUF controlled response (V2) and generative attacker (V3), but those target the *PUF response* directly. This attacks the *classifier* itself via black-box queries, a different vector. The 10-attack battery does not include classifier extraction.

---

## 6. Attack: Multi-Round Constraint Extraction via Repeated Probes

**Category:** protocol (multi-round leakage)

**Threat model:**  
Attacker with K_chip (via leak) but without chip access wants to emulate the chip's S sample in the multi-round protocol (§5.10.3). The verifier's constraints in R2 are derived from the nonce (via SHAKE256) and are therefore deterministic once the nonce is known. If the attacker can send the same nonce multiple times and observe the *same* constraints, they can collect many (S, constraints, t) triplets and learn the chip's distribution for those specific constraints.

**Mechanism:**  
The multi-round protocol is meant to be run once per challenge. However, if the verifier does not enforce nonce uniqueness, an attacker could replay the same nonce multiple times. Each time, the verifier sends the same 5 constraints (since they are SHAKE256 of nonce). The attacker can vary his submitted S slightly and observe whether verification passes (binary oracle). Over many queries, he can pin down the correct values for each constraint, effectively reconstructing S for that nonce. Since the constraints subset a 50-element S, with enough queries the attacker can recover the entire chip's micro-sample distribution for that nonce plan. Once he has a library of (nonce, S) pairs, he can emulate future challenges.

**Implementation sketch:**  
```python
for trial in range(1000):
    # Send nonce N, receive constraints
    cons = verifier.round2_send(N)
    # Start with a guess S_guess, adjust based on previous failures
    # E.g., use gradient-free optimization: CMA-ES with objective = number of constraints satisfied
    ...
    if verifier.round3_verify(N, dict(S=S_guess, t=...)).accepted:
        record(N, S_guess)
```
This is essentially an active learning attack that uses the verifier as an oracle to extract the chip's physical response for a given nonce.

**Expected outcome:**  
If the verifier does not implement a rate limit or nonce-reuse detection, the attacker can iteratively refine S. After enough iterations, he can produce an S that passes all 5 constraints. Since the classifier check also runs on the aggregate features, but if the S is genuine (from chip's distribution), the classifier may also accept. This directly breaks the multi-round goal of "forces full response-surface knowledge".

**Why NOT in existing list:**  
The existing attacks include K_chip leak + multi-round emulation (V4), but they assume the attacker must emulate the *full* chip dynamics in one shot. The oracle-based iterative refinement is a new exploitation of repeated queries. The 10-attack battery does not include a multi-round oracle attack.

---

## 7. Attack: BIOS Re-flash to Spoof Deterministic Board Fingerprints

**Category:** supply-chain / board-level cloning

**Threat model:**  
Attacker has physical access to both target (victim) and prover (attacker-owned) machines. They can re-flash the BIOS/UEFI on the attacker's machine to match the victim's DMI/SMBIOS serial number, PCI topology, and USB descriptor chain. The deterministic signals S20–S26 (§4.1c) rely on board-level digital identifiers that are modifiable via firmware updates or component swaps.

**Mechanism:**  
The verifier may short-circuit accept on Phase 22 deterministic match (Section 4.1c note). If the attacker clones the victim's PCI tree, DMI hash, and USB enumeration (by re-flashing BIOS and possibly swapping USB hub controllers), then S20–S26 will match exactly. The attacker's machine would then pass the zero-FP bypass, and the verifier would accept without running expensive stochastic checks. If the verifier uses deterministic bypass as a fast path, the attacker can bypass all Tier-2 hardening.

**Implementation sketch:**  
```bash
# Victim's DMI data
cat /sys/class/dmi/id/product_serial > serial.txt
# Attacker: flash modified BIOS with victim's serial
flashrom -p internal -w victim_bios.bin
# Modify PCIe device tree (if possible by reprogramming EEPROMs)
# Swap USB controller firmware
```
After cloning, the deterministic signals become identical.

**Expected outcome:**  
The verifier's short-circuit accept will accept the attacker as the victim. This requires physical access to both machines and some hardware expertise, but it's a realistic supply-chain attack.

**Why NOT in existing list:**  
The 10 attacks include static replay and peer transplant but not board-level cloning. V0-V7 do not cover modifying deterministic board fingerprints. The paper mentions this risk in §4.1c ("modulo trivial board cloning"), but does not list it as a formal attack vector. This is a naive but missed attack.

---

## RANKING (Top-5)

| RANK | NAME | NOVELTY (1-5) | PLAUSIBILITY (1-5) | EST EFFORT (hrs) | EST P(BYPASS) |
|------|------|----------------|---------------------|-------------------|----------------|
| 1 | Enrolment Poisoning | 5 | 4 | 8 | 1.0 (if calibration not authenticated) |
| 2 | BCH Decoding Oracle | 5 | 3 | 40 | 0.9 (given enough queries) |
| 3 | BIOS Re-flash Clone | 3 | 5 | 120 | 1.0 (on physical attack) |
| 4 | Multi-Round Constraint Extraction | 4 | 3 | 100 | 0.6 (depends on rate limits) |
| 5 | Classifier Membership Inference | 4 | 2 | 200 | 0.3 (plan-score still blocks) |

**Note:** Plausibility and effort are estimates. Enrolment Poisoning is the most direct and likely easiest to implement if the attacker controls the enrolment pipeline. The BCH oracle is theoretically sound but requires many queries and may be slowed by rate limits. BIOS cloning is high effort but guarantees bypass if deterministic fast path is used. The other two are more involved but reveal deeper flaws.
