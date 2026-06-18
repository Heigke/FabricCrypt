# gemini response (gemini-2.5-pro) — 73s

Here is a maximally adversarial analysis of the FabricCrypt scheme.

### 1. Worst attack you found

**The "Canary-Fixing" Replay Attack**

This attack completely breaks the scheme's primary defense, the plan-consistency gate, with minimal cost and a near-100% success rate per attempt. It requires only a single previously recorded `(nonce, signature)` pair from the victim.

**Step-by-Step Execution:**

1.  **Prerequisite:** The attacker possesses one or more valid `(nonce_recorded, signature_recorded)` pairs from the target chip (`ikaros`). Let's say they have one such pair. `signature_recorded` is a 64-dim vector `[V_phys_permuted, E_nonce_recorded]`.
2.  **Challenge:** The verifier issues a fresh challenge nonce, `nonce_new`.
3.  **Attacker's Computation (Offline):**
    a.  The attacker computes the sampling plan for the *recorded* nonce: `plan_recorded = derive_plan(nonce_recorded, ...)` (`nonce_signature.py:84`). This gives them `perm_recorded`.
    b.  They also compute the plan for the *new* challenge nonce: `plan_new = derive_plan(nonce_new, ...)`. This gives them `perm_new` and, crucially, `ns_sleep_new`.
    c.  The attacker takes the physical part of the recorded signature, `V_phys_permuted`. They use the inverse of `perm_recorded` to recover the *un-permuted, normalized* physical vector `Z_unpermuted`.
    d.  The attacker knows from `nonce_signature.py:183` that the canary value is stored at index 31 of the *raw* vector before normalization and permutation. They calculate the expected *normalized* value for the new canary: `canary_new_norm = np.log1p(ns_sleep_new * 1e-3)` (replicating the logic in `verifier_spoof_v2.py:61`).
    e.  The attacker creates a modified vector `Z_unpermuted_mod` by taking `Z_unpermuted` and overwriting the value at index 31 with `canary_new_norm`.
    f.  The attacker then applies the *new* permutation to this modified vector: `V_phys_forged = Z_unpermuted_mod[perm_new]`.
    g.  The attacker computes the nonce embedding for the new nonce: `E_nonce_new = nonce_embedding(nonce_new, 32)`.
    h.  The final forged signature is `signature_forged = concatenate([V_phys_forged, E_nonce_new])`.
4.  **Response:** The attacker sends `signature_forged` to the verifier.

**Why It Succeeds:**

*   **Plan-Consistency Gate:** It passes by construction. The verifier will invert `perm_new`, look at index 31 of the un-permuted vector, and find exactly the `log1p(ns_sleep_new * 1e-3)` value it expects. The `plan_score` will be 1.0.
*   **Twin Classifier:** The classifier is completely fooled. The forged physical vector is identical to a real, previously-seen vector from the target chip in 31 of its 32 dimensions. The `ikaros_spoof_v2.json` results for `static_replay_no_nonce` show a `classifier_p0_mean` of **0.995**. This means the classifier *loves* replayed signatures, even when the nonce embedding is mismatched. Our forged signature is even more convincing because the nonce embedding is correct. The classifier will vote "own" with very high confidence.

**Cost & Probability:**

*   **Cost:** A few hours of a security engineer's time to script the attack. Requires a single captured signature. The computational cost is negligible. This is well within a **$10k budget** (in fact, it's closer to a $500 budget).
*   **Probability of Success:** >99% per attempt. The only failure mode would be if overwriting dimension 31 drastically changes the statistical properties of the vector in a way the classifier detects, which is extremely unlikely.

---

### 2. Other attacks ranked

1.  **Generative Model Attack.**
    *   **Summary:** An attacker with a large library (M > 10⁵) of `(nonce, signature)` pairs trains a generative model (e.g., a GAN or VAE) to learn the mapping `f(nonce) -> V_phys`.
    *   **Defence Survival:** This defeats the current scheme entirely. The model would learn the correlation between the nonce-derived plan and the resulting physical measurements, including the `ns_sleep` canary. It could generate a novel, statistically-correct signature for any fresh nonce.
    *   **Cost/Feasibility:** High. Requires significant data collection and ML expertise. Plausible for a nation-state, marginal at a $1M budget.

2.  **Relay Attack.**
    *   **Summary:** An attacker places a proxy near the victim chip and relays the verifier's challenge, receiving the victim's true signature and forwarding it back within the timeout.
    *   **Defence Survival:** The current protocol has no defense. The 1-3 ms challenge window is vulnerable. A relay over a local LAN (<0.5ms RTT) is trivial. A relay over the public internet is feasible if the verifier and victim are geographically close (e.g., same metro area, <10ms RTT).
    *   **Cost/Feasibility:** Low to medium. Requires a compromised machine near the victim. Trivial for a co-tenant in the same datacenter.

3.  **Side-Channel Inference (Power/Acoustic).**
    *   **Summary:** A nation-state attacker uses power analysis (e.g., on VRM ripple) or acoustic analysis (coil whine) to reconstruct the physical signature in real-time or, more likely, to build a highly accurate predictive model of the chip's behavior.
    *   **Defence Survival:** The protocol has no defense, as noted in `threat_model_and_signals.md`. The attacker could learn the `nonce -> V_phys` mapping without ever using the protocol itself, then mount a generative model attack.
    *   **Cost/Feasibility:** Very high. Requires physical proximity and specialized equipment. Firmly in the nation-state category.

4.  **Weak Nonce Generation.**
    *   **Summary:** The verifier's nonce generation is predictable. The test harness (`verifier_spoof_v2.py:230`) uses `np.random.default_rng(int(time.time()) & 0xFFFFFFFF)`, which is catastrophically insecure.
    *   **Defence Survival:** A real implementation using `secrets.token_bytes` would survive. But if the test harness code were ever used in a real system, an attacker could predict nonces and pre-compute valid signatures from the victim chip.
    *   **Cost/Feasibility:** Trivial, if the vulnerability exists.

---

### 3. Defences to ADD

The core vulnerability is that the canary is **public and deterministic**. An attacker knows what it should be and can simply write it into a replayed signature.

1.  **Make the Canary Secret and Unpredictable.**
    *   **Change:** Do not embed `plan.ns_sleep` directly. Instead, embed a value derived from a chip-fused secret key (if available) or a provisioned secret.
    *   **Code-level Sketch (in `NonceSig._raw_read`):**
        ```python
        # Assume self.secret_key is a 16-byte secret provisioned on the chip
        canary_value = hmac.new(self.secret_key, nonce + b'_canary', hashlib.sha256).digest()
        # Embed the first 4 bytes of the HMAC output
        out[31] = float(np.frombuffer(canary_value[:4], dtype=np.uint32)[0])
        ```
    *   **Impact:** This breaks the Canary-Fixing attack. The attacker cannot compute the correct canary value for `nonce_new` because they don't have `self.secret_key`. The verifier would need a corresponding way to check this, likely via a trusted third party or a pre-shared symmetric key.

2.  **Use Multiple, Interdependent Canaries.**
    *   **Change:** Instead of one canary, embed several values that are computationally linked. For example, `out[10] = f(nonce, secret)`, `out[20] = g(nonce, secret, out[10])`, `out[30] = h(nonce, secret, out[20])`.
    *   **Impact:** This hardens the signature against simple patching. An attacker trying to fix one canary would have to re-calculate others, which requires knowing the secret and the dependency graph. This moves towards a "proof of work" on the chip.

3.  **Implement Distance Bounding.**
    *   **Change:** The verifier must measure the round-trip time (RTT) of the challenge-response exchange with high precision. A challenge could be a single bit, requiring a single-bit response, repeated many times to establish a baseline RTT.
    *   **Impact:** This is the only robust defense against relay attacks. If the observed RTT exceeds the physical limit (e.g., speed of light to the claimed location), the proof is rejected.

4.  **Use a Proper Fuzzy Extractor.**
    *   **Change:** The current `plan_consistency_score` is a weak, single-dimension check. A real fuzzy extractor would take the entire 32-dim physical vector `V_phys` and a public "helper data" string `H` to reproduce a stable, high-entropy secret `S`. The protocol would then be `S = extract(V_phys, H)`, and the chip would prove knowledge of `S`.
    *   **Impact:** This provides a much more robust way to handle noise and ties the identity proof to the entire physical state of the chip, not just one dimension. It makes targeted modification attacks much harder.

---

### 4. Bit-security estimate

The scheme, as specified in the provided files, is broken.

*   **Q18(a) Replay attacker with M ≤ 10⁵:** **0 bits.** The Canary-Fixing attack requires M=1.
*   **Q18(b) Library-replay with M ≤ 2³⁰:** **0 bits.** Same reason.
*   **Q18(c) Generative-model adversary:** **0 bits.** This is a more powerful attack, but the simpler one already wins.
*   **Q18(d) Hardware-emulator adversary:** **0 bits.** The software-based Canary-Fixing attack is a form of emulation that is far cheaper and easier than full hardware emulation.

*   **Q25 ($10k budget):** **0 bits of security.** The probability of a successful forgery is effectively 100%. The cost to develop the Canary-Fixing attack is minimal.
*   **Q26 ($1M budget):** **0 bits of security.** An attacker with this budget would develop a full generative model, which would also defeat the scheme.
*   **Q27 (Nation-state budget):** **0 bits of security.** A nation-state would use side-channels to build a perfect predictive model of the chip, rendering the entire protocol useless.

**Conclusion:** The reliance on a single, publicly-derivable canary value in the plan-consistency gate is a fatal design flaw. The `accept_rate` is determined *only* by this gate (`verifier_spoof_v2.py:72`), and this gate is trivially bypassed. The classifier, while interesting, provides no meaningful security against a knowledgeable adversary. **This scheme is not secure.**
