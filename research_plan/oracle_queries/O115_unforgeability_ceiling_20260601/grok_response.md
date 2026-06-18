# grok response (grok-4-latest) — 20s

**1. Worst attack you found**

Hardware emulator (FPGA + DRAM controller + sysfs shim). Attacker obtains full source (nonce_signature.py:140-170 derive_plan, 260-310 _raw_read, verifier_spoof_v2.py:180-210 plan_consistency_score). They build a small FPGA board that presents itself as the target APU over PCIe/DRAM and, on receiving a 64-bit nonce over the wire, instantly:
- runs the identical HMAC-SHA256 to obtain the plan (ns_sleep, perm, cpu_subset, …),
- writes the exact required value into the emulated location that will become dim 31 after permutation (out[31] = plan['ns_sleep'], line 310),
- fabricates the remaining 31 phys values from a small pre-characterized distribution that passes the 0.15 log-space tolerance (verifier_spoof_v2.py:200),
- returns the 64-dim vector in <200 µs.

The plan-consistency gate (the sole acceptance criterion) passes with probability 1.0. No real silicon physics is required; the gate is a pure software predicate on a deterministic integer. Cost: ~$8k–12k (Xilinx/Intel FPGA + PCIe breakout + 2 weeks engineering). Success probability per challenge after one-time build: >0.999. This is the attack that actually kills the scheme, because every measurement FabricCrypt performs is ultimately a software-readable file or perf counter that an emulator can satisfy.

**2. Other attacks ranked**

1. Generative-model predictor trained on 10^6 (nonce, sig) pairs collected from the victim. (a) Learns the joint distribution of the 32 phys features conditioned on plan; (b) at challenge time samples a vector whose dim-31 component exactly matches the required ns_sleep (probability 1) and whose other marginals stay inside the 0.15 tolerance with high probability. Defence that survives: none in current design. Cost: <$5k (cloud GPU hours). Feasible today.

2. Relay / low-latency proxy. Attacker places a machine within ~100 µs RTT of the victim and forwards the nonce, receives the signature, forwards it. 1–3 ms challenge window is easily bridged on a LAN (measured <150 µs) and still possible on a well-peered Internet path. Defence that survives: explicit distance-bounding or sub-100 µs response deadline. Cost: <$2k hardware.

3. Cold-boot/DRAM dump of the _cal directory and any cached (nonce, sig) pairs during retrain_embodied_nonce. Recovers mu/sigma plus hundreds of plan-consistent examples. Defence that survives: none; calibration files are world-readable. Cost: physical access once.

4. Acoustic coil-whine + power-rail ripple side-channel on the victim while it executes the challenge. Reconstructs ns_sleep and TSC-burst statistics in real time; sufficient to satisfy the single gate. Defence that survives: none. Cost: $3k–10k lab equipment.

5. Weak verifier nonce (verifier_spoof_v2.py:310 uses np.random.default_rng(time.time()&0xffffffff)). Attacker pre-images the 32-bit seed and builds a 2^32 library offline. Defence that survives: secrets.token_bytes (already noted as required).

**3. Defences to ADD**

- Replace HMAC-SHA256 derive_plan with a keyed VRF (or at minimum HMAC-SHA256(key||nonce) where key is a per-die fused secret never exposed to software). This makes the plan non-public and non-predictable without the key.
- Bind the response window to a hardware timer (e.g., require the entire read() to complete inside a 50 µs window measured by a privileged TSC that user space cannot stretch). This kills both emulator and relay.
- Move the plan-consistency check into a signed kernel module or enclave that also measures the actual instruction-retirement latency of the nanosleep/TSC bursts; reject if the observed latency distribution deviates from the expected ns_sleep by more than a few percent.
- Drop the public mu/sigma calibration files; force every verifier to request a fresh raw=True vector and perform its own normalization.
- Add at least one signal that cannot be emulated without breaking existing firmware contracts (e.g., a cache-line lock that also triggers a measurable SMI whose timing is fused).

**4. Bit-security estimate**

- Hardware-emulator adversary ($10k): 0 bits (practical break).
- Generative-model adversary ($1M): ≤20 bits (library of 10^6 pairs plus interpolation already yields >50 % success after a few dozen attempts).
- Nation-state (unbounded but no physical destruction): 0 bits (FPGA emulator + side-channel calibration).
- Replay attacker limited to M ≤ 10^5 pairs: ~33 bits (plan entropy claim in threat_model_and_signals.md:5.5, but reduced by the single leaked dimension).
- Library-replay M ≤ 2^30: ~33 bits before the gate is exhausted; still broken once a learned predictor is added.
- zkML composition or fuzzy-extractor addition would raise the bar only if the plan itself becomes secret and hardware-timed; as written the scheme remains broken at the $10k level.
