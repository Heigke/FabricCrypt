# gemini response (gemini-2.5-pro) — 109s

# FC-B1 RED-TEAM RESPONSE

Here are 7 new attack concepts against FabricCrypt v3.1, designed to probe for weaknesses not covered by the existing attack battery.

---

### Attack 1: Calibration Environment Poisoning

1.  **Name**: Calibration Environment Poisoning
2.  **Category**: Supply-chain / Enrollment
3.  **Threat model**: Attacker has temporary physical access to the target chip *before or during its initial calibration/enrollment phase*. The attacker does not need to maintain access after enrollment is complete.
4.  **Mechanism**: The entire security of the keyed protocol (Tier-1 and Tier-2) rests on `K_chip`, which is derived from the `mu` vector (the mean of physical measurements) during the `_maybe_calibrate` routine in `nonce_signature_v2.py`. This calibration is meant to capture the chip's baseline physical properties.
    This attack manipulates the chip's operating environment *only during this calibration phase*. For example, the attacker could apply a specific, constant thermal load to one side of the chassis, or run a carefully crafted background process that perturbs specific signal families (e.g., cache contention, network interrupts). This would cause the calibration routine to record a skewed `(mu, sigma)` profile. The verifier now holds a "poisoned" ground truth for the chip.
    The attacker, knowing the specific perturbation they applied, can later reproduce it on their *own* hardware to generate signatures that more closely match the target's poisoned profile than the target's own signatures under normal conditions.
5.  **Implementation sketch**:
    ```python
    # Attacker's actions:
    # 1. On target machine, before running calibration script:
    #    $ stress-ng --cpu 1 --cpu-load 95 --timeout 600s &  # Apply known, reproducible load

    # 2. Run the FabricCrypt enrollment process on the target machine.
    #    The verifier now stores a (mu_poisoned, sigma_poisoned) profile and K_chip_poisoned.

    # 3. Later, on attacker's machine (e.g., 'daedalus'):
    #    $ stress-ng --cpu 1 --cpu-load 95 --timeout 600s &  # Reproduce the exact same load
    #    prover = NonceSigV2(host='daedalus', K_chip=K_chip_daedalus)
    #    forged_sig = prover.read(fresh_nonce)
    #    # This forged_sig, generated under the same artificial conditions as the
    #    # target's enrollment, will have a higher probability of passing the
    #    # verifier's checks against the poisoned profile.
    ```
6.  **Expected outcome**: This attack could cause the verifier to reject the *honest* chip operating under normal conditions, while accepting forgeries from the attacker's chip operating under the special "poisoned" conditions. It undermines the very foundation of the physical fingerprint by attacking the process that establishes ground truth. It would likely bypass the HARD veto because the forged signature is designed to match the enrolled (but poisoned) `(mu, sigma)`.
7.  **Why it's NOT in our 10/10 list**: The existing list assumes a correctly-enrolled chip. This is an attack on the enrollment process itself. It's a form of supply-chain attack that doesn't require permanent modification of the hardware.

---

### Attack 2: Statistical Forgery against Mahalanobis Gate

1.  **Name**: Statistical Forgery against Mahalanobis Gate
2.  **Category**: ML / Protocol
3.  **Threat model**: Attacker has stolen `K_chip` and the target's enrolled `(mu, sigma)` vectors (the V4 scenario). The attacker does *not* have the physical chip.
4.  **Mechanism**: The `plan_measurement_score` in `verifier_v2.py` is the core of the Tier-1 plan-consistency gate. It checks that the Mahalanobis distance `sqrt(mean(((unperm - mu)/sigma)^2))` is within a band `k`. An honest chip produces noisy physical measurements that satisfy this on average.
    The attacker, with `K_chip`, can derive the `plan` for any `nonce`, including the permutation. Instead of trying to emulate the complex physics, the attacker can construct a synthetic `unperm_fake` vector directly. Each element `unperm_fake[i]` is simply drawn from a normal distribution `N(mu[i], sigma[i])`. By construction, the z-score `(unperm_fake[i] - mu[i]) / sigma[i]` will be a standard normal variable, and the resulting Mahalanobis distance will be very close to 1.0, easily passing the gate (`mahala <= 3.0`). The attacker then applies the correct permutation to create the `phys_part` of the signature to send to the verifier.
5.  **Implementation sketch**:
    ```python
    def forge_statistically(nonce, K_chip, mu, sigma, n_cpus, n_zones):
        # 1. Attacker has K_chip, mu, sigma. They derive the plan.
        plan = derive_plan_keyed(nonce, K_chip, n_cpus, n_zones)
        perm = plan['perm']

        # 2. Construct a synthetic un-permuted vector from the target statistics.
        #    This requires no physical hardware.
        unperm_fake = np.random.normal(loc=mu, scale=sigma).astype(np.float32)

        # 3. Apply the correct permutation.
        phys_part_forged = unperm_fake[perm]

        # 4. Create the full signature vector.
        nonce_emb = nonce_embedding(nonce)
        full_sig_forged = np.concatenate([phys_part_forged, nonce_emb])
        return full_sig_forged
    ```
6.  **Expected outcome**: This attack will likely pass the `plan_measurement_score` gate with a near-perfect score. Its success then depends entirely on whether the synthetic vector can also fool the `classifier`. Since the classifier is trained on real, correlated physical noise, it's possible that a vector of independent Gaussian samples would be distinguishable. However, this is a much more sophisticated forgery than `all_dim_flood` and directly targets the statistical nature of the gate.
7.  **Why it's NOT in our 10/10 list**: The `stolen_kchip_analysis` is a threat model, not a specific forgery mechanism. The `all_dim_flood` attack involves filling all dimensions with a single value, which would fail the Mahalanobis test (some dims would have huge z-scores). This attack generates a statistically plausible vector that is tailor-made to pass the specific Mahalanobis check.

---

### Attack 3: Reverse Fuzzy Extractor Decoding Oracle

1.  **Name**: Reverse Fuzzy Extractor Decoding Oracle
2.  **Category**: Protocol / Cryptography
3.  **Threat model**: Attacker can repeatedly challenge the Tier-2 protocol and observe the final binary accept/reject outcome. The attacker does not need `K_chip`.
4.  **Mechanism**: The Reverse Fuzzy Extractor in `reverse_fuzzy.py` provides security by keeping the helper data `P` private to the verifier. The verifier computes `v = w_noisy XOR P` and attempts to decode `v`. The result is a binary "accepted" (if decoding succeeds and `K'` matches `K`) or "rejected".
    An attacker can use this binary response as an oracle. They start with a valid observed response `w_noisy_0` from the target chip, which they know was accepted. They then systematically submit modified versions `w_noisy_i = w_noisy_0 XOR e_i`, where `e_i` is a vector with a single bit flipped at position `i`.
    By observing which flips preserve acceptance and which break it, the attacker learns information about the underlying codeword `c` (since `v = c + e`, where `e` is the noise). If flipping bit `i` in `w_noisy` breaks acceptance, it implies that this bit was "more critical" to the original successful decoding. Over many queries, this can leak information that helps the attacker construct a `w_noisy` that is closer to a valid `w_ref XOR P`, potentially allowing them to forge a response without the chip.
5.  **Implementation sketch**:
    ```python
    # Attacker has one valid response w_noisy_0 from the target.
    # Assume a function `query_verifier(w_noisy)` that returns True/False.

    w_ref_P_bits_guess = np.zeros(N_BITS)
    for i in range(N_BITS):
        # Create two probes by flipping the i-th bit
        w_probe_1 = w_noisy_0.copy()
        w_probe_1[i] ^= 1

        # Query the verifier with the flipped version
        accepts_flip = query_verifier(w_probe_1)

        if not accepts_flip:
            # If flipping the bit caused a reject, it's more likely that the
            # original bit in w_noisy_0 matched the corresponding bit in (w_ref XOR P).
            # This is a simplification; real analysis is more complex.
            w_ref_P_bits_guess[i] = (w_noisy_0[i] ^ w_ref[i]) # This is P[i]
            # This is a slow, statistical process to recover P.
    ```
6.  **Expected outcome**: This is a slow and difficult attack, but it is a classic pattern against systems that expose a binary decoding oracle. It may not lead to a full break in a limited number of queries, but it fundamentally leaks information about the secret helper data `P`, which is supposed to be the main security gain of the RFE.
7.  **Why it's NOT in our 10/10 list**: This is a cryptographic attack on the Tier-2 RFE primitive itself, exploiting its interactive nature. The existing list focuses on replay and modeling of the physical signature, not on oracle-based leakage from the cryptographic components.

---

### Attack 4: Side-Channel on Plan Execution

1.  **Name**: Side-Channel on Plan Execution
2.  **Category**: Side-channel
3.  **Threat model**: Attacker is a co-tenant on the same physical machine (e.g., in a different VM or container) or has the ability to measure the prover's response time with very high precision over a network. Attacker's goal is to recover `K_chip`.
4.  **Mechanism**: The `_raw_read` function in `nonce_signature_v2.py` executes a different sequence of physical measurements depending on the `plan`, which is derived from `SHAKE256(K_chip, ..., nonce)`. For example, the `core_pairs` to be tested, the `ns_sleep` duration, and the number of `tsc_count` samples all vary. These different operations have distinct execution times and power/thermal footprints.
    By sending many different `nonce` values and precisely measuring the total time taken for `sig.read(nonce)` to complete, the attacker can build a statistical model linking `nonce` to execution time. Since `time = f(plan)` and `plan = g(K_chip, nonce)`, the attacker is observing `time = f(g(K_chip, nonce))`. Over enough observations, they can perform a differential timing analysis to recover bits of `K_chip`. For example, if they can find two nonces `n1, n2` that differ by one bit and result in a statistically significant time difference, they learn something about how that bit interacts with `K_chip` inside the SHAKE256 function.
5.  **Implementation sketch**:
    ```python
    # Attacker code running on the same host or with low-latency network
    prover_interface = connect_to_prover()
    timing_data = []
    for i in range(100000):
        nonce = generate_structured_nonce(i) # e.g., flipping one bit at a time
        t0 = time.perf_counter_ns()
        response = prover_interface.get_signature(nonce)
        t1 = time.perf_counter_ns()
        timing_data.append({'nonce': nonce, 'time_ns': t1 - t0})

    # Offline analysis:
    # model = fit_model(timing_data)
    # for each byte k_byte in K_chip:
    #   find best_guess for k_byte that minimizes model prediction error.
    # This is a hard problem but standard for side-channel analysis.
    ```
6.  **Expected outcome**: This is a classic side-channel attack. It is difficult to execute but plausible in a co-tenant scenario. If successful, it recovers `K_chip`, which would allow the attacker to mount the "Statistical Forgery" attack (Attack 2) and bypass the Tier-1 protocol entirely.
7.  **Why it's NOT in our 10/10 list**: The existing list does not contain any side-channel attacks aimed at recovering the secret key `K_chip`. It only considers a scenario where `K_chip` is *already stolen* (e.g., from a file leak). This attack describes a method to steal it remotely.

---

### Attack 5: Deterministic Signal Forgery

1.  **Name**: Deterministic Signal Forgery
2.  **Category**: Clone / OS-level
3.  **Threat model**: Attacker has root access on their *own* machine, which is the same hardware model as the target. They have stolen the target's `K_chip` and enrolled `(mu, sigma)` profile.
4.  **Mechanism**: The paper's 466-dim signature relies on 7 "board-level deterministic fingerprints" (S20-S26) which have zero intra-host variance and are claimed to provide "zero false positives". These are read from sysfs (`/sys/bus/pci`, `/sys/class/dmi/id`, etc.). A root-level attacker on their own machine can often manipulate what is reported in sysfs, either by writing to the files (if writable) or, more powerfully, by using a custom kernel module to intercept the `read()` syscalls for those specific paths and return forged data that matches the target's deterministic values.
    The attacker would first need to obtain the target's deterministic signal values (e.g., from a single leaked signature). With `K_chip`, they can un-permute the signature and extract the values. Then, they use their kernel module to spoof these values on their own machine, while letting the stochastic HAL-bypass signals be generated by their own hardware.
5.  **Implementation sketch**:
    ```c
    // Kernel module pseudocode (LKM)
    // original_read = get_sys_call_table_entry(SYS_read);
    //
    // my_read_hook(fd, buf, count) {
    //   char path[256];
    //   get_path_from_fd(fd, path, sizeof(path));
    //
    //   if (strcmp(path, "/sys/class/dmi/id/product_uuid") == 0) {
    //     copy_to_user(buf, "TARGET_SPOOFED_UUID", ...);
    //     return SPOOFED_LEN;
    //   }
    //   // ... other hooks for S20-S26 paths
    //
    //   return original_read(fd, buf, count);
    // }
    //
    // // In init_module:
    // set_sys_call_table_entry(SYS_read, my_read_hook);
    ```
6.  **Expected outcome**: This attack would make the attacker's machine appear identical to the target for 96 of the 466 dimensions. This significantly reduces the entropy available for the classifier and Mahalanobis gate to distinguish the two machines, increasing the probability of a successful forgery. It directly challenges the robustness of using easily-manipulable OS-reported identifiers as part of a hardware fingerprint.
7.  **Why it's NOT in our 10/10 list**: This is a sophisticated cloning attack that goes beyond simple replay. It targets the *source* of the signals themselves, blending real stochastic data with forged deterministic data. The existing `daedalus_peer` test uses an unmodified peer machine; this attack uses a *modified* peer machine.

---

### Attack 6: Time-of-Check, Time-of-Use (TOCTOU) on Thermal State

1.  **Name**: TOCTOU on Thermal State
2.  **Category**: Protocol / Side-channel
3.  **Threat model**: Attacker has `K_chip` and the `(mu, sigma)` profile. Attacker has their own hardware.
4.  **Mechanism**: Several signals are thermal-dependent (S6 Thermal spread, S9 Jacobian dynamics, etc.). The `_raw_read` function measures these nearly instantaneously. The verifier's `(mu, sigma)` profile represents the chip's behavior at a "normal" operating temperature. The attacker can use their knowledge of the plan (from `K_chip`) to pre-calculate a forged response that passes the statistical gates. However, this response may correspond to a thermal state their chip is not currently in.
    The attack is to "steer" their own chip into a thermal state that is closer to the one required for the forgery. They would use a feedback loop:
    1. Receive nonce `N`.
    2. Compute the target plan.
    3. Generate a statistically valid fake response `R_fake` (per Attack 2).
    4. Measure their own chip's current response `R_real`.
    5. If `R_real` is far from `R_fake`, apply a thermal/CPU load to their machine to "steer" its physical state.
    6. Go to step 4 and repeat until `R_real` is close enough to `R_fake`.
    7. Submit the now-valid `R_real`.
    This is a TOCTOU attack because the verifier checks the signature at one point in time, but the attacker is actively manipulating their hardware state right up until the moment of measurement to pass that check.
5.  **Implementation sketch**:
    ```python
    def steer_and_sign(nonce, K_chip, mu, sigma):
        plan = derive_plan_keyed(nonce, K_chip, ...)
        target_unperm = np.random.normal(mu, sigma) # The ideal fake vector

        prover = NonceSigV2(...) # Attacker's own chip
        while True:
            # Time of Check
            real_response_perm = prover.read(nonce, raw=False) # Get a real reading
            real_unperm = unpermute(real_response_perm[:32], plan['perm'])

            # Compare to target
            dist = np.linalg.norm(real_unperm - target_unperm)
            if dist < THRESHOLD:
                return real_response_perm # Time of Use: submit the valid signature

            # Steer hardware state
            if np.mean(real_unperm[thermal_dims]) < np.mean(target_unperm[thermal_dims]):
                apply_thermal_load()
            else:
                cool_down()
    ```
6.  **Expected outcome**: This makes forgery much more practical for a `K_chip`-leak attacker. Instead of perfect emulation, they only need to nudge their own (legitimate but different) hardware into a physical state that produces a signature statistically close enough to the target's profile for a specific nonce. The tight response window (<3ms) makes this hard, but not impossible for certain slow-changing signals like temperature.
7.  **Why it's NOT in our 10/10 list**: This is a dynamic attack that manipulates the attacker's own hardware state to pass a check, a form of "physical adversarial example". It's a blend of a cloning and side-channel attack that is more sophisticated than simply replaying responses.

---

### Attack 7: ZK-Scaffold Misuse

1.  **Name**: ZK-Scaffold Misuse
2.  **Category**: Protocol / Implementation
3.  **Threat model**: Attacker has `K_chip`. The system is using the ZK-binding protocol from `zkml.py`.
4.  **Mechanism**: The ZK binding in §5.10.4 and `zkml.py` is a scaffold that relies on HMAC for binding. The tag is `HMAC(K_chip, com_S || y_bytes || prog_hash || x_bytes)`. The commitment `com_S` is `SHA256("chip-sig-com-v1" || r || S_bytes)`. The verifier checks the tag and can optionally open the commitment.
    An attacker with `K_chip` can compute a valid tag for *any* `(com_S, y, x, prog_hash)` they choose. They can commit to a garbage signature `S_garbage` (or even an empty string), compute `com_S_garbage`, then compute a valid HMAC tag for their desired fraudulent output `y_fraud`.
    The verifier will accept the tag as valid. The attack is only caught if the verifier performs the optional, expensive step of asking the prover to open `com_S_garbage` and then re-running the inference. A lazy or resource-constrained verifier might skip the opening step and just trust the HMAC tag, allowing the attacker to attribute any output to the target chip without ever generating a valid signature.
5.  **Implementation sketch**:
    ```python
    def forge_with_zk_scaffold(K_chip, x, y_fraud, prog_hash):
        # 1. Attacker doesn't need a real signature S. Commit to garbage.
        S_garbage = b"garbage"
        r = secrets.token_bytes(32)
        h = hashlib.sha256(b"chip-sig-com-v1|" + r + b"|" + S_garbage).digest()
        com_S_garbage = h

        # 2. Compute a valid tag for the fraudulent output using the stolen K_chip.
        tag_fraud = inference_tag(K_chip, com_S_garbage, x, y_fraud, prog_hash)

        # 3. Create a proof. The verifier will not be able to open it, but the tag is valid.
        proof = {
            'com_S': com_S_garbage.hex(),
            'prog_hash': prog_hash.hex(),
            'y': y_fraud.tolist(),
            'tag': tag_fraud.hex(),
            # Attacker cannot provide valid opening data, but hopes verifier won't ask.
            '_open_r': r.hex(),
            '_open_S': [0.0] * 32
        }
        return proof
    ```
6.  **Expected outcome**: The `verify_tag` check will pass. The `open_commit` check will fail if the verifier performs it. This attack exploits the fact that the expensive part of verification (opening and re-running) might be skipped, and that the HMAC tag alone does not prove the correctness of the committed signature `S`. It highlights a weakness in the "honest interactive-proof" properties of the scaffold.
7.  **Why it's NOT in our 10/10 list**: This is an attack on the not-yet-fully-implemented Tier-2 ZK binding component, exploiting a potential implementation shortcut (skipping the commitment opening).

---

## TOP-5 ATTACK RANKING

| RANK | NAME                                   | NOVELTY (1-5) | PLAUSIBILITY (1-5) | EST EFFORT (hrs) | EST P(BYPASS) |
|------|----------------------------------------|---------------|--------------------|------------------|---------------|
| 1    | Calibration Environment Poisoning      | 5             | 4                  | 8                | 0.8           |
| 2    | Statistical Forgery (Mahalanobis Gate) | 4             | 5                  | 2                | 0.6           |
| 3    | Deterministic Signal Forgery           | 4             | 5                  | 24 (LKM dev)     | 0.7           |
| 4    | Side-Channel on Plan Execution         | 5             | 3                  | 80 (analysis)    | 0.2           |
| 5    | RFE Decoding Oracle                    | 4             | 2                  | 40 (many queries)| 0.1           |
