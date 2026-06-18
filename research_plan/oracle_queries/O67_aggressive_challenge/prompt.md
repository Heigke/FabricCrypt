Two clear claims survived today's brutal ablation:
1. **Substrate-level energy**: ~4 fJ/cell-read, 5e8× less than DRAM, no refresh
2. **DC fit**: 4.16 → 0.965 dec via per-VG1 BBO

Three claims were retracted:
- DS-N7 Memory Palace = hash-table-in-disguise (digital dict = 0.957)
- DS-N9 Neuromorphic memory = digital+decay dominates 22-24pp
- HDC scaling = +1pp plateau

Now challenge:

Q1 — Is the "4 fJ/cell-read" really apples-to-apples vs DRAM? Specifically:
  - We counted channel current × probe pulse only
  - We did NOT count: sense-amp overhead, decoder logic, write current
  - At realistic system level (including peripheral), is the 5e8× claim still ~10× or just ~10× max?

Q2 — Is the 0.965 dec genuine physics or BBO fishing in 9-dim space?
  - Per-VG1 params show extreme spread (Rs 6e6 vs 8e9)
  - Could a stupid model with 9 free params get under 1.0 on ANY 33-curve set?
  - Provide null-hypothesis test recipe.

Q3 — Is NS-RAM categorically EVER useful, or always at best a noisy DRAM replacement?
  - What 3 specific commercial applications would actually NEED:
    (a) analog state continuum (>3 bits)
    (b) refresh-free retention (0.1-10s)
    (c) low-energy read at high density
  - Be specific (e.g., "always-on KWS for wearables", not "edge AI")

≤500 words per oracle. Be RUTHLESSLY skeptical. We want to know if we should ship v4.5 brief or kill the project.
