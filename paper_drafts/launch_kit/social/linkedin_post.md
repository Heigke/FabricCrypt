# LinkedIn launch post — FabricCrypt v0.2

**Target:** ~1500 chars. Posted by the founder/researcher account.
**Media:** attach `twitter_60s.mp4` (or LinkedIn-native re-upload).
**Posting time:** Tuesday 14:00 UTC, cross-posted from X.

---

We built two identical laptops. Only one can run our AI model.

Same SKU. Same RAM. Same BIOS. Same microcode. Move the model file to chassis B and the verifier refuses to authenticate. Move it back to chassis A and it works again. No vendor key. No Secure Enclave. No TPM. Just physics.

**The problem.** Apple's Private Cloud Compute and NVIDIA's Confidential Compute can attest "this is *some* H100 in CC mode" — but not *which* H100. And if you don't have the vendor's PKI, you don't have the attestation. Period.

**What we did.** FabricCrypt is a software-only attestation primitive (at n=2 chassi) that fingerprints a commodity AMD die from 15 signals total: 5 HAL-bypass micro-architectural (inter-core TSC offsets, cacheline ping-pong, DRAM-refresh-aligned jitter, syscall p99.9 tails, NVMe queue-tail latency) + 3 cross-host KS-verified μ-arch (GPU clock jitter, multi-zone thermal, Jacobian dynamics) + 7 board-level deterministic (PCI/PCIe/USB/DMI/UCSI/amdgpu/kernel-boot). Each challenge is bound to a 64-bit nonce that controls *what gets measured*, so a replayed recording lands on the wrong sampling plan and is rejected.

Numbers (N=2 AMD Ryzen AI Max+ PRO 395 "Strix Halo" laptops): 100% leave-one-out die discrimination on a 466-dim live signature, 1.12 ms median verify latency, ten security gates pass, Tier-2 crypto hardening raises the **empirical** attack-cost against ML-modeling attackers to the random-Hamming floor (≥10¹² samples) — these are empirical operating points, not formal cryptographic reductions. ML-modeling attack defeated (0/40 forgeries at N=160 training pairs).

**Demo.** 60-second video of the transplant: model passes on A, refuses on B, passes again on A. Linked below.

**Honest limits.** N=2 chassis — small, and we say so. The cryptographic ceilings above are empirical attack-cost, not proven security reductions. Exploratory stylometric divergence from chip-conditioned training (§7.L6) is supplementary detail, not a headline claim. Persistent kernel-resident adversaries remain out of scope.

Code is MIT-licensed. No accounts, no enrollment with us. We are asking owners of AMD Strix Halo / Ryzen AI 300 laptops to run the enrollment script and publish their signature so we can extend N.

Repo: https://github.com/Heigke/FabricCrypt
Preprint: arXiv:XXXX.XXXXX

#AI #cybersecurity #cryptography #FabricCrypt #attestation #ConfidentialComputing #AMD
