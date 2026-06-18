# openai response (gpt-5) — 241s

# FC-B1 RED-TEAM RESPONSE

Below are 8 new attack concepts against FabricCrypt v3.1 that are not covered by your 10/10 list (§5.9) nor the V0–V7 vectors (§5.10.5). Each entry includes name, category, threat model, mechanism (with section/code refs), implementation sketch, expected outcome, and why it’s not already in your list.

---

1) Name: HARD-veto oracle permutation fishing (HVOPF)
- Category: protocol/oracle side-channel
- Threat model:
  - Remote attacker can make repeated online attestation attempts to the verifier and observe accept/reject (or any coarse accept/deny signal). No K_chip or chip access.
- Mechanism:
  - §5.4 and verifier_v2.py implement a HARD veto: accept iff plan_pass AND classifier_p0 > τ_cls. The plan_pass score is min(s31, sM), where s31 depends only on “unpermuted” dim 31 (MAD of nanosleep burst), and sM is a (Mahalanobis-like) RMS over all 32 dims. The inverse permutation is computed from K_chip and nonce (unknown to the attacker), but the accept/deny bit leaks information about which coordinate in the submitted 32-dim phys slice is being interpreted as “dim 31 after unpermutation.”
  - By repeatedly submitting synthetically-crafted 64-dim inputs with different single-coordinate modifications and observing the accept/deny flip, an attacker can identify which permuted coordinate the verifier treats as dim 31 under a fixed nonce (and then approximate its mu_31 value by 1-D line search). This reduces the unknown 32-way permutation problem to locating a single sensitive coordinate (“which slot is s31?”) using only the HARD-veto oracle.
  - Once that index is learned for a given nonce, the attacker sets that coordinate to the (oracle-found) “sweet spot” and sets the remaining 31 coordinates to generic “in-distribution” guesses (e.g., medians from public traces), then chooses the trailing 32 nonce-embedding arbitrarily (see Attack 3) to push the classifier. This is a black-box active attack on plan-consistency that exploits the verifier as a side-channel.
- Implementation sketch (pseudo-Python):
  ```
  # Oracle: returns True/False for a single probe
  def oracle(nonce, X64):
      # sends X64 to verifier, observes accept/reject bit only
      return send_and_observe_accept(nonce, X64)

  # Step 1: find the permuted slot that maps to unperm dim-31
  base = np.zeros(64, dtype=np.float32)
  # initialize nonce-emb slots to any fixed vector (see Attack 3)
  base[32:] = 0.0

  influence = np.zeros(32)
  for j in range(32):
      v1, v2 = -1.0, +1.0             # coarse probe values for phys[j]
      x1, x2 = base.copy(), base.copy()
      x1[j] = v1; x2[j] = v2
      a1 = oracle(nonce, x1)
      a2 = oracle(nonce, x2)
      # coordinate with largest accept/deny sensitivity is the s31 slot
      influence[j] = abs(int(a1) - int(a2))
  j_star = int(np.argmax(influence))

  # Step 2: 1-D search for a passing value on that coordinate
  lo, hi = -8.0, 8.0
  for _ in range(20):
      mid = 0.5 * (lo + hi)
      x = base.copy(); x[j_star] = mid
      a = oracle(nonce, x)
      if a: hi = mid
      else: lo = mid
  # Step 3: craft final submission
  X = base.copy()
  X[j_star] = hi
  # fill remaining 31 phys dims with public “typical” values (e.g., sample mean of sanitized traces)
  X[:32] = fill_typical_phys(X[:32], hold=j_star)
  X[32:] = choose_nonce_emb_for_classifier_boost()  # see Attack 3
  send_and_observe_accept(nonce, X)
  ```
- Expected outcome:
  - Partial bypass. With tens to a few hundred oracle probes per target nonce, attacker can learn the permuted position of the s31 gate and a passing value for it, raising plan_pass probability markedly. Classifier still stands in the way; Attack 3 can help. Success probability depends on how tight sM is and how separable the classifier is on the attacker’s synthetic phys.
- Why it’s NOT in your 10/10 list:
  - Not a replay, not K_chip brute force, not ML modeling on the raw/controlled PUF. This is an adaptive, black-box oracle side-channel against the verifier’s internal plan-consistency test. Closest superficial resemblance is “all_dim_flood” (O115 variant), but that was a one-shot fill; here the novelty is oracle-guided localisation of the hidden s31 slot via accept/deny leakage.

---

2) Name: Multi-round constraint degeneracy (constant-S commit)
- Category: protocol
- Threat model:
  - Remote attacker can speak the Tier-2 multi-round protocol (§5.10.3) but has no chip and no K_chip. The attacker can fabricate any S (50 raw micro-samples).
- Mechanism:
  - In T2.3 (multiround.py), R1 commit_S binds the prover to S; in R2 the verifier sends five post-hoc constraints (median, variance, lag1-ACF, count_above_q using quantile of sub, weighted sum). For degenerate sequences like S ≡ 0 (all zeros), all constraints are trivially predictable without knowing the subsets/weights: median=0, variance=0, lag1-ACF=undefined but computed 0 (due to +1e-12 in denom), count_above_q=0 because tau is the quantile of sub itself, and weighted_sum=0 for any weights. Hence, the “post-hoc” unpredictability gives no hardness: any constant S satisfies all five constraints exactly after R2 is revealed, and the commit check passes.
  - This defeats the specific “five post-hoc constraints” hardness claimed in §5.10.3 (though identity checking on S should still reject later).
- Implementation sketch:
  ```
  from multiround import MultiRoundVerifier, evaluate_constraint, commit_samples
  import numpy as np, os

  V = MultiRoundVerifier(n_samples=50)
  nonce = os.urandom(8)

  # Attacker chooses constant S and commits in R1
  S = np.zeros(50, dtype=np.float32)
  r1 = dict(commit_S=commit_samples(nonce, S), n_samples=50)
  V.round1_recv(nonce, r1)

  # Receive constraints in R2 and compute exact t’s
  cons = V.round2_send(nonce)
  t = [evaluate_constraint(S, c) for c in cons]

  # Reveal S and t in R3
  res = V.round3_verify(nonce, dict(S=S, t=t))
  print(res)  # accepted=True at the multiround layer
  ```
- Expected outcome:
  - Bypasses the “five-constraint” hardness entirely (accepted=True at multiround layer). The full pipeline should still apply classifier/Reverse-FE on S and reject an alien chip. This matters because it collapses the advertised incremental hardness of T2.3: the constraints add no difficulty if the adversary chooses a degenerate S.
- Why it’s NOT in your 10/10 list:
  - Not a replay; not ML; not helper-data leakage. It targets the new Tier-2 multi-round constraint design specifically. None of §5.9 attacks examine this degeneracy.

---

3) Name: Nonce-embedding spoof to bias the classifier
- Category: classifier/protocol
- Threat model:
  - Attacker can submit any 64-dim vector to the verifier; knows model weights (open source). No K_chip. Observes accept/deny (or classifier scores if leaked).
- Mechanism:
  - §5.4 says the verifier checks “embedded_nonce == N” before classification, but verifier_v2.py’s HARD veto never checks that the last 32 dims equal nonce_embedding(nonce). The classifier uses these 32 dims as features. A forger can populate the trailing 32 dimensions with an arbitrary vector that maximizes p_own for the deployed Twin-MLP, independent of the actual nonce, to tilt the classifier in their favor.
  - Combine with Attack 1 (oracle-learned s31 slot) to also pass plan_pass. This is a straight-up integrity gap between the paper’s check and the code: the embedding is not bound to the nonce at verification time.
- Implementation sketch:
  ```
  # Given: known PyTorch model, known nonce -> but we won't use true emb
  from verifier_v2 import classifier_p0
  import torch, numpy as np

  def best_emb_for_fixed_phys(model, phys32, trials=5000):
      # random search (or gradient ascent) over 32-dim "embedding"
      best_e, best_p = None, -1.0
      for _ in range(trials):
          e = np.random.normal(0, 1, 32).astype(np.float32)
          e = e / (np.linalg.norm(e) + 1e-8) * 2.0  # scale if needed
          X = np.concatenate([phys32.astype(np.float32), e], axis=0)[None, :]
          p0 = classifier_p0(model, X)[0]
          if p0 > best_p: best_p, best_e = p0, e
      return best_e, best_p

  # Forge: pick phys32 by any means (e.g., Attack 1 tuned s31); then append best_e
  ```
- Expected outcome:
  - Partial bypass. Increases classifier_p0 considerably, potentially above τ_cls, while plan_pass still must be met. If the operator adds the missing “embedded_nonce == N” check, this closes immediately.
- Why it’s NOT in your 10/10 list:
  - It’s neither “nonce_mismatch” nor “honest_wrong_nonce” (those refer to the nonce used to derive plan). This is spoofing the auxiliary embedding feature that the code fails to validate.

---

4) Name: Tier-2 downgrade: bypass the Controlled-PUF path
- Category: protocol/downgrade
- Threat model:
  - Adversary can speak the old Tier-1 wire format (64-dim phys + nonce_emb) and the verifier accepts it; no chip or K_chip required if they can model/forge Tier-1 features sufficiently well (e.g., via Attack 1 + 3 or other synthesis).
- Mechanism:
  - §5.10.2 introduces Controlled-PUF (H_in/H_out SHAKE256 wrapper), claiming the verifier operates on the wrapped output and never sees raw_phys. However, verifier_v2.py never requires a wrapped digest, nor does it check any Controlled-PUF tag. It still consumes 64-dim phys + 32-dim nonce_emb. The Controlled-PUF code exists (controlled_puf.py) but is not enforced by the verifier gate.
  - A malicious prover can simply send Tier-1-style 64-dim vectors; the verifier has no cryptographic signal that a Controlled-PUF wrap ever occurred.
- Implementation sketch:
  ```
  # Attacker: send 64-dim "classic" vector; no 'wrapped' bytes at all
  X64 = forge_phys_plus_emb()  # e.g., Attack 1+3 combo to pass gates
  result = post_to_verifier({'phys_plus_emb': X64.tolist()})
  print(result['accept'])  # Controlled-PUF never checked/enforced
  ```
- Expected outcome:
  - If the verifier indeed accepts the old message shape, Tier-2 protections (T2.2) are entirely bypassed, reverting security to Tier-1. This is a wire-level downgrade vector.
- Why it’s NOT in your 10/10 list:
  - Your V2/V7 vectors evaluate ML on the Controlled-PUF output; they assume the wrapper is in force. This is a protocol integration issue: no check that the wrapped construction was actually used.

---

5) Name: Enrollment-time K_chip poisoning via user-space load shaping
- Category: supply-chain/enrollment poisoning
- Threat model:
  - Local co-tenant on the target host at enrollment/calibration time; no root required. Can run CPU-bound and syscall-heavy jobs concurrently during NonceSigV2 calibration.
- Mechanism:
  - §5.8 Fix 3 derives K_chip = SHA256("FabricCrypt-K_chip-v1" || quantized_mu_bytes || host). In nonce_signature_v2.py, calibration collects ~60 random-plan physical reads under K0=zero and averages them into mu, then immediately derives K_chip from that mu (no TPM sealing in the PoC).
  - A co-tenant process can bias the nanosleep and TSC bursts (and thermal zones) during this window (e.g., run high-priority nanosleep loops, hammer thermal sensors, change CPU governor if permitted) to shove mu across quantizer bin boundaries. This yields a “poisoned” K_chip differing from an uncontended-enrollment K_chip, causing verifier/prover desynchronization or setting K_chip to a value chosen from a small attacker-controlled set (low effective entropy).
- Implementation sketch:
  ```
  # Attacker runs in parallel with enrollment
  import os, ctypes, time, threading

  libc = ctypes.CDLL('libc.so.6')
  class Timespec(ctypes.Structure):
      _fields_ = [("s", ctypes.c_long), ("ns", ctypes.c_long)]

  def jitter_spammer():
      ts = Timespec(0, 1000000)  # 1 ms
      while True:
          libc.nanosleep(ctypes.byref(ts), None)

  def cpu_heater():
      x = 0
      while True:
          x = (x * 1103515245 + 12345) & 0xffffffff

  # Launch during calibration window
  for _ in range(os.cpu_count()):
      threading.Thread(target=cpu_heater, daemon=True).start()
  for _ in range(4):
      threading.Thread(target=jitter_spammer, daemon=True).start()
  time.sleep(120)  # cover the 60-sample calibration period
  ```
- Expected outcome:
  - Causes K_chip mismatch (DoS) or predictable/low-entropy K_chip (if attacker repeats same load pattern across multiple enrollments). Even partial control increases the risk that equal devices converge to similar K_chip, lowering plan entropy implicitly.
- Why it’s NOT in your 10/10 list:
  - Not a replay/ML attack. It’s an enrollment poisoning vector against Fix 3, unmentioned in §5.9 and orthogonal to helper-data leakage (V5) and K_chip file theft (V4).

---

6) Name: Plan-entropy collapse due to spec–implementation drift
- Category: protocol/implementation bug
- Threat model:
  - Offline attacker analyzing source code to quantify effective plan entropy; online replay attacker leveraging the much-smaller plan space to increase hit rate with modest libraries.
- Mechanism:
  - §5.3 and §5.6 claim dominant entropy from choosing 16 of 120 pairs (~60 bits), plus other fields, totaling ≈64 effective bits. But nonce_signature_v2.derive_plan_keyed() currently selects only TWO core_pairs, small ns_count/tsc_count ranges, and modest CPU/zone subsets. This collapses the actual plan space by many orders of magnitude relative to the spec.
  - A dynamic-replay attacker can harvest a library of (nonce, phys) and, with M in the hundreds or thousands, now cover a significant fraction of the tiny realized plan space (especially if zones/CPU choices repeat frequently), raising the practical success rate well above the §5.6 bound. This is not “dynamic replay (M=200)” as in §5.9 because the theoretical bound assumed C(120,16); the code does not implement that.
- Implementation sketch:
  ```
  # Measure realized plan diversity
  from nonce_signature_v2 import derive_plan_keyed
  import os, hashlib

  def plan_fingerprint(plan):
      return hashlib.sha256(repr(plan).encode()).hexdigest()

  K0 = b'\x00'*32
  seen = set()
  for _ in range(10000):
      n = os.urandom(8)
      p = derive_plan_keyed(n, K0, n_cpus=16, n_zones=8)
      seen.add(plan_fingerprint(p))
  print("distinct plans in 10k draws:", len(seen))
  # Expect far below the combinatorial count implied by §5.3.
  ```
- Expected outcome:
  - If confirmed, practical dynamic replay becomes substantially easier than §5.6 asserts. This weakens Tier-1/Tier-2 replay resistance and should be fixed by implementing the full C(120,16) choice (and/or adding more high-entropy plan elements).
- Why it’s NOT in your 10/10 list:
  - You already tested “dynamic_replay (M=200)” under the theoretical plan analysis; this is an implementation-level plan-entropy collapse, not the same test.

---

7) Name: Reverse-FE acceptance oracle probing (syndrome oracle)
- Category: protocol/oracle side-channel
- Threat model:
  - Remote attacker can invoke the Reverse-FE step many times and observe accept/reject (or any signal correlated with decode success), but has no access to the enrolled reference bits w_ref or helper P.
- Mechanism:
  - reverse_fuzzy.py verify() computes ham = Hamming(w_ref, w_noisy) and returns (accepted, recovered_K_or_None, ham). In a real service, even if ham is not returned to the client, any stable bit (accept vs reject) leaks whether Hamming ≤ t for that w_noisy. Because quantize_to_bits(vec) uses a global median/MAD threshold ladder applied identically across dims for each request, an attacker who can choose vec (they forge the 466-dim input) can perform adaptive queries to learn constraints on w_ref: “does there exist a vector whose bits X yield accept?” Over many queries, one can solve a CSP to approximate w_ref structure (e.g., parity constraints from the BCH code), especially if the verifier responds quickly and there is no rate limit.
  - This is analogous to a “syndrome decoding oracle” with binary feedback (inside/outside radius t). Even if fully reconstructing w_ref is hard, the oracle can lower the search space for forgeries substantially.
- Implementation sketch:
  ```
  # Oracle: returns True if RFE accepts the current quantized bits, False otherwise
  def rfe_oracle(vec):
      bits = quantize_to_bits(vec, n_bits_total=N)
      return rfe_verify_accepts(bits)  # remote oracle: accept/reject

  # Hill-climbing on vec to reduce reject rate
  v = np.zeros(D)              # D = 466 dims (or 64 in a toy)
  for it in range(2000):
      idx = np.random.randint(0, len(v))
      delta = np.random.choice([-0.5, +0.5])
      v_try = v.copy(); v_try[idx] += delta
      if rfe_oracle(v_try): v = v_try  # greedy accept-region climb
  # If an accept is ever achieved, record vec and replay it later.
  ```
- Expected outcome:
  - Partial. Without helper data P or w_ref, success is uncertain, but an accept/reject oracle plus a low correction radius t (e.g., m=8, t=16) can leak enough about the accept region to eventually find a vec that quantizes within distance t of w_ref. Rate-limits and server-side randomization of the quantizer would mitigate this.
- Why it’s NOT in your 10/10 list:
  - Not helper-data leakage (V5) and not ML on the PUF. It exploits the Reverse-FE verifier as an oracle via adaptive queries, which your current battery and vectors do not address.

---

8) Name: ZK-binding freshness gap (replayable inference bundle)
- Category: protocol
- Threat model:
  - Remote attacker can replay previously captured (com_S, y, tag) bundles produced by a legitimate host. No K_chip; no chip.
- Mechanism:
  - T2.4 zkml.py binds y to com_S and x and prog_hash via tag = HMAC(K_chip, com_S || y || prog_hash || x). There is no binding to the attestation nonce or any freshness token. A previously generated valid bundle for input x and program M can be replayed indefinitely on the same verifier (same enrolled K_chip), and will verify as long as the verifier re-runs M(x) and checks the tag. There is no mechanism tying the inference to a specific audience nonce or to a time window.
- Implementation sketch:
  ```
  # Attacker replays a captured honest proof (stub)
  proof = capture_from_wire()  # dict with com_S, y, tag, prog_hash, _open_*
  x = extract_x_if_present()   # or rely on verifier-supplied x for known service
  res = zk_verify_stub(proof, M, x, enrolled_K_chip, model_code)
  print(res['accepted'])  # True, with no freshness binding
  ```
- Expected outcome:
  - Full replay of the inference-binding proof in T2.4. This does not directly pass the attestation hard veto, but it undermines the “bind inference to this chip for this request” guarantee unless the caller separately ties S/com_S to a challenge. A simple fix is to include nonce in both com_S and tag.
- Why it’s NOT in your 10/10 list:
  - Your batteries test attestation gates, not the ZK-binding wrapper. This is a correctness hole in the new T2.4 scaffold.

---

TOP-5 NOVELTY + PLAUSIBILITY RANKING

RANK | NAME | NOVELTY (1-5) | PLAUSIBILITY (1-5) | EST EFFORT (hrs) | EST P(BYPASS)
1 | Tier-2 downgrade: bypass the Controlled-PUF path | 4 | 5 | 2 | High if verifier accepts legacy 64-dim inputs (Tier-2 nullified)
2 | HARD-veto oracle permutation fishing (HVOPF) | 5 | 4 | 6 | Medium (per-nonce plan_pass boost; full bypass depends on classifier)
3 | Multi-round constraint degeneracy (constant-S commit) | 4 | 5 | 1 | High for defeating constraint layer; overall accept still needs classifier
4 | Nonce-embedding spoof to bias the classifier | 4 | 4 | 3 | Medium (raises classifier_p0; needs plan_pass via other means)
5 | Enrollment-time K_chip poisoning via load shaping | 3 | 4 | 4 | Medium (can desync or reduce K_chip entropy; depends on deployment ops)

Additional notes:
- Plan-entropy collapse (spec–impl drift) should be investigated urgently; if confirmed, it globally weakens replay defenses (effort ~2 hrs to measure).
- Reverse-FE oracle probing and ZK-binding freshness gaps are partial-bypass vectors but meaningful: they attack Tier-2’s claimed hardness and binding properties, respectively, rather than Tier-1 gates.
