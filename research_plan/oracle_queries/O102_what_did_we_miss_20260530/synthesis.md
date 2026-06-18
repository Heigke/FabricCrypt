# O102 — Synthesis (4/4 oracles received)

**Status**: openai (gpt-5, 172s), gemini-2.5-pro (84s), grok, deepseek all received.

**Headline**: **3-of-4 strong convergence on cryptographic VCEK-as-CONSTRAINT** (OpenAI + Gemini + DeepSeek). Grok dissents, prefers active wear-as-training (which the other three independently rate LOW-EV given driver normalisation + guardbands).

---

## Per-oracle Q-by-Q matrix

| Q | OpenAI (gpt-5) | Gemini-2.5-Pro | Grok | DeepSeek |
|---|---|---|---|---|
| **Q1 untested arch** | CONSTRAINT — closed-loop power/deadline | CONSTRAINT — via SEV-SNP structure | ACTIVE DEGRADATION (write substrate) | CONSTRAINT — VCEK/TPM as hard requirement |
| **Q2 wear-as-training** | LOW (driver normalises) | VERY LOW (guardbands hide) | HIGH (only path escapes meta-pattern) | LOW (ECC/wear-leveling spreads) |
| **Q3 crypto-substrate** | Yes — clean constructive | YES — VCEK defines MODEL STRUCTURE | Dismissed as "still read-only" | YES — public VCEK hash → weight mask |
| **Q4 compiler/ISA** | Low EV (twins identical) | Dead end (ISA class not silicon) | (not emphasised) | (not emphasised) |
| **Q5 missing category** | Energy/time-budget as CONSTRAINT | Approx-compute via undervolting | Active wear | Joint-multichannel SCA fusion |
| **Q6 SCA closure** | Coarse PoC OK with hwmon | $35 USB ADC + LSTM | (subordinate) | Joint-channel fusion |
| **Q7 approx-compute** | Won't be per-die without V-control | STRONG: Vmin via MSR/ryzen_smu | (under active wear) | Cites Papadimitriou HPCA17 Vmin 9-24% |
| **Q8 theorem status** | NOT formal — engineering combo | NOT formal — empirical | (agrees) | (agrees) |
| **Q9 single experiment** | **SEV → HKDF → encrypts final layer** | **VCEK → PRNG → permutes hidden layer** | Wear stress + fingerprint cofit | **VCEK → deterministic weight mask** |
| **Q10 100h plan** | 35-45h SEV crypto; 45-55h constraint loop | Path A SEV permutation; Path B Vmin parallel | Multi-day wear + fingerprint | 0-5h sevctl; 5-35h train; 35-45h transplant; 45-60h TPM fallback |

## Convergence

**3/4 STRONG**: OpenAI, Gemini, DeepSeek independently — without seeing each other — recommend SEV-SNP VCEK as the substrate. Three distinct technical variants:

- **OpenAI**: VCEK → HKDF → AES-CTR encrypts FINAL LINEAR LAYER. Decrypt-or-⊥ inside SEV guest. Wrong device cannot decrypt → ⊥ output.
- **Gemini**: VCEK → SHA256 → PRNG → fixed PERMUTATION of hidden-layer activations. Downstream weights co-fit to P_ikaros; P_daedalus scrambles representation → accuracy collapse.
- **DeepSeek**: VCEK hash → deterministic MULTIPLICATIVE WEIGHT MASK. Wrong key → weights scrambled → garbage output.

All three are **constructive gates** (binary pass/⊥ or accuracy collapse to chance), **unfalsifiable** by shuffle/SW-matched/spatial-seed leak (the prior 14-attack failure modes), and implementable in <24h on existing Strix Halo hardware.

## Divergence (Grok)

Grok argues all three crypto variants are "still read-only signal regime" — the key gates output but the model doesn't *learn through* the substrate physically. He prefers active wear-as-training (substrate-as-ACTIVE-DEGRADATION). Other three counter:
- driver normalisation hides wear (OpenAI, DeepSeek)
- guardbands eliminate user-space visibility (Gemini)
- damage risk + tiny effect size in 2-week budget (OpenAI, Gemini)

**Operational verdict**: Grok's critique is philosophically valid (cryptographic binding is not *emergent* identity coupling), but operationally less actionable. The author treats it as motivation to run BOTH tracks: Track 1 (crypto) for guaranteed publishable constitutive result, Track 2 (Vmin-fault-cofit, also independently endorsed by Gemini as Path B) for the emergent claim Grok wants.

## Final method recommendation

### Primary: Gemini-Q9 VCEK-Permutation (24-hour implementation)

Reasons:
1. 3-of-4 oracles converge unprompted.
2. Constructive gate: ikaros >90%, daedalus <15% on CIFAR-10 — unfalsifiable by all 14 prior confounds.
3. Zero hardware risk.
4. Uses existing SEV-SNP on Strix Halo (verified per AMD doc 58217).
5. First commodity-x86 published result of structural cryptographic constitutive binding for a learnable model (Wu et al. arxiv 2212.11133 did binary lock only, not structural co-adaptation).

Pre-registered gates:
- **G1**: Self-eval on ikaros with P_ikaros: ≥ 90% accuracy on CIFAR-10.
- **G2 (constructive)**: Transplant to daedalus with P_daedalus: ≤ 15% (chance level).
- **G3 (confound-killer)**: Apply *random* permutation P_random of same statistics: also collapses to ≤ 15%. Proves the binding is to *the specific* device permutation, not just to "having any permutation".
- **G4 (key-stability)**: Reboot ikaros, re-extract VCEK, verify deterministic re-derivation of P_ikaros. Proves key is stable across power cycles.

### Secondary (parallel, 36-72h): Vmin-fault-cofit (Track 2)

Author's pre-oracle prior, independently re-derived by Gemini as Path B. Tests Grok's preferred emergent-binding claim. Pre-registered gates from `IDENTITY_DEEPER_HUNT_2026-05-30.md` Section 5 unchanged.

### Combined paper

"Cryptographically-constitutive binding via SEV-SNP VCEK works trivially on commodity x86 (Track 1, first published instance); emergent binding via per-die Vmin fault co-fit [succeeds/fails — TBD] (Track 2). The 14 prior failed attacks plus this dichotomy establish the boundary of what is achievable in user-space on Strix Halo."

If Track 1 passes and Track 2 fails: clean result, paper is positive on Track 1 + negative on Track 2 with full closure of the question. If both pass: extraordinary, two independent positive results. If both fail: pivot to FPGA / external ADC mandatory, abstraction tax is total. **All outcomes are publishable.**
