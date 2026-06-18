# P1 Oracle Synthesis — O54 Optimal Fix-Order

Date: 2026-05-13
Packet: `research_plan/oracle_queries/O54_fix_order/`
Providers: openai (gpt-5, 166s), gemini (gemini-2.5-pro, 60s), grok (grok-4-latest, 29s)

---

## Q1 — Fix-order for pyport_v4 (P3)

### Per-oracle ranking
| Rank | OpenAI (gpt-5) | Gemini (2.5-pro) | Grok (4-latest) |
|------|----------------|------------------|-----------------|
| 1 | VNwell diode polarity | BSIM rbodymod=1 | VNwell diode polarity |
| 2 | rbodymod=1 (distributed Rb) | VNwell diode polarity | N1 multi-τ traps |
| 3 | Drain avalanche M(V_bc) | N1 multi-τ traps | BSIM rbodymod=1 |
| 4 | (defer) SRH gen-rec | SRH gen-rec | (defer) SRH |
| 5 | (defer) N1 traps | — | (defer) avalanche |

### Consensus
- **Unanimous: VNwell diode (correct polarity) FIRST OR SECOND.** One-line, zero-risk, unblocks everything else.
- **Unanimous: rbodymod=1 high priority** (top-3 every oracle; #1 for Gemini, #2 for OpenAI).
- **2/3 (Gemini, Grok): N1 traps in first wave.** OpenAI dissents — wants envelope (1–3) settled before tuning trap τ to avoid compensating for envelope error (z311 already overshoots loop area).
- **OpenAI alone calls for drain avalanche in wave 1**; Gemini & Grok defer it.

### Locked recommendation (P3 build order, next 8h)
1. **VNwell→VB diode polarity fix** (`z310b`, 1-line) — unblocks body discharge dynamics.
2. **rbodymod=1 / distributed Rb** — structural BSIM card fix; predicted to collapse the 9-decade per-branch Rs split. **Highest single-step DC gain.**
3. **Drain avalanche M(V_bc)** coupled to body — sets V_d>2V shape on the new 143-sample T2 validation set.
4. **DC sweep + falsifier** (see Q2). If P3 DC gate (<0.7 dec) not met → add SRH.
5. **N1 multi-τ traps LAST** (OpenAI's discipline argument wins): z311 stub already over-lifts hysteresis by 6.2 dec, so trap τ tuning must come AFTER envelope is correct, else traps absorb envelope error.

---

## Q2 — Cheapest falsifiers (≤2h)

### P3 (pyport_v4) — locked
**Two-ablation harness on ikaros:**
- A) Base + diode + rbody (no avalanche, no traps): fit 33 IV. **Reject if cell-wide median_log_rmse improves <0.5 dec over z304 OR V_G1=0.2 signed bias ≤ −1.0 dec.**
- B) Transient triplet @ 0.017 / 0.17 / 1.7 V/s with same model. **Reject if knee does not monotonically left-shift as ramp slows.** If either fails → SRH required before traps.

### P4 (KWS attack) — locked
**Non-spiking baseline on identical MFCC/rank-coded features and splits:** logistic regression OR linear SVM, same train/test/seed protocol as NS-RAM SNN.
- **If baseline ≥70% while NS-RAM ≤25% across 3 seeds → falsify "encoding/data" as bottleneck; localize failure to NS-RAM SNN mapping (architecture or plumbing, not physics).**
- Gemini variant (5–10 epochs on 10% subset with rank-coded MFCC): kept as a complementary cheaper probe.

---

## Q3 — KWS keep / abandon

### Consensus: **PERSIST, BUT GATED.**
All three oracles agree: do NOT anchor v4.4 on KWS; lead with HDC+RNG, run KWS as a bounded-gate side experiment (P4 PASS threshold = accuracy >25% across 3 seeds).

### Physical arguments
- **For:** NS-RAM hysteretic, multi-τ memory plausibly matches 10–40 ms audio windows; event-sparse → sub-100 µW feasible if MFCC front-end power dominated.
- **Against:** Body/trap time constants (µs–s) are a poor match for phoneme-level timescales; MFCC front-end usually dominates power; current SNN at chance ⇒ architectural/plumbing fault, not yet physics-limited.

### Locked verdict
- **KEEP** KWS as P4 side experiment with P4-PASS = acc>25%, ≥3 seeds.
- **DO NOT** headline KWS in v4.4 brief. v4.4 leads with HDC (N=16384, post-10-seed lock) + Bayesian RNG.
- If P4 fails at >25%: report as explicit, honest negative result; remove KWS from v4.4 application slate.

---

## Q4 — NO-CHEAT discipline

### Split: Gemini & Grok say "no drift"; **OpenAI flags 3 specific drifts** (and is correct).
Gemini cites: z310 deferral respected, 4E HELD, z312 full matrix not cherry-picked.
Grok cites: gates respected, full std reporting.

**OpenAI's flags (verified against 01_LOG tail):**
1. **"v4.4 headline LOCKED at 84.09%" at 10:15 with only 4 seeds.** Pre-registered rule: ≥10 seeds for headline lock. → **Action: run 6 additional seeds before language remains "locked."**
2. **"Noise-BENEFITING" claim (84.09% vs 83.91% @ σ=0).** Δ=0.18pp with std=0.20 (n=4). Statistically ambiguous. → **Action: downgrade to "no worse than σ=0" until 10-seed CI confirms.**
3. **Causal/outcome language on diode fix BEFORE numbers.** "will produce the rate-dependent hysteresis we want" appears before z310b is run. → **Action: hold causal language until z310b numbers land.**

### Locked verdict
**Drift detected (minor but real).** 3 corrections required before v4.4 brief unholds:
- Re-run z312 with 6 more seeds (10 total).
- Restate noise tolerance as "no worse"; only upgrade to "noise-benefiting" if 10-seed CI excludes zero.
- Diode causal claim removed from log/brief until z310b numbers published.

---

## Locked P3 build order
1. z310b VNwell diode polarity (1-line)
2. rbodymod=1 / distributed Rb
3. Drain avalanche M(V_bc)
4. Falsifier A+B (Q2 harness)
5. N1 traps LAST (only after envelope is correct)
6. SRH only if Q2-falsifier-A fails

## Locked KWS verdict
**KEEP as P4 gated side experiment (>25% acc, 3 seeds). HDC+RNG lead v4.4. Do not headline KWS.**
