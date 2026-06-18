# Oracle packet O107 — how to STRENGTHEN a weak-but-real die-specific signal to clean separation?

Adversarial reviewer. Police BOTH biases (wanting to succeed; giving up early). Cite ONLY real work. Rank
every suggestion by expected SNR/separation gain AND probability, give a concrete thermally-safe protocol for
any you rate worth doing. HARD THERMAL LIMIT: 99°C ACPI trip = reboot; sustained near-throttle off the table;
low-duty / sharp-edge / demodulated only.

## Exactly where we are (verified, this session)
Goal: frozen LLM constitutively dependent on ONE specific AMD Strix Halo gfx1151 APU die. Triad: (1) UNIQUE
[SOLVED: CPPC ranking 75% distinct + dynamics fingerprint 14×], (2) RÄKNA = genuine nonlinear computation the
model needs, (3) FRESH [SOLVED: RDSEED].

RÄKNA progress:
- The die physically computes the PRODUCT u·v via shared-PDN power contention (GPU u-bursts × CPU v-bursts).
  A LINEAR readout of on-die power/thermal telemetry does XOR(u,v)=0.75 (= full product ceiling 0.746),
  300-shuffle null p=0.000, u-only=chance, u&v-LINEAR=chance. So genuine analog multiplication. BUT this is
  GENERIC — both our dies (ikaros, daedalus) compute it ~identically. So räkna YES, räkna-UNIKT not yet.
- We then probed the SPATIAL PDN coupling: CPU v-bursts PINNED to 4 distinct cores (0,3,6,9) = spatial zones
  × GPU u; built a temperature-compensated coupling matrix M[zone,channel] (C_uv = A_uv/sqrt|A_u·A_v|), with
  a MATCHED-TEMPERATURE band (49-56°C) on both dies and the SAME drive seeds (the control our earlier
  cross-die test lacked).
- RESULT (3 ikaros runs + 1 daedalus, matched temp): same die is more self-similar than cross-die.
  Compensated cosine INTRA(ik self)=[0.73,0.82,0.75] mean 0.763; INTER(ik-da)=[0.73,0.39,0.71] mean 0.611;
  mean gap +0.152. Raw coupling also now +0.116 (was −0.029 at n=1). SO: mean-level die-specificity REAL and
  REPLICATED in both raw & compensated. BUT NO CLEAN SEPARATION: the inter distribution OVERLAPS intra (one
  ik-da pair = 0.73 = as high as same-die pairs). It is a weak statistical tendency, not a usable per-die
  fingerprint. Run-to-run noise (intra only 0.76, not ~0.95) is the limiter.

## The question: what STRENGTHENS this to clean separation (min_intra > max_inter), thermally safely?
Critique/rank these candidate strengtheners + add any we missed + flag fatal flaws:
1. MORE ZONES: use all 16 CPU cores (and/or GPU CU-occupancy patterns) → 16×Nsensor signature, richer spatial
   PDN structure. Cheapest, biggest leverage? Diminishing returns / scheduler-migration risks?
2. TEMPLATE ENROLLMENT: average K runs per die into a template, classify new runs vs templates (standard PUF
   enrollment + majority voting / helper data). How many runs K to beat run-to-run noise? error-correction?
3. LOCK-IN / FREQUENCY-SWEEP readout (we did time-domain regression only): drive u,v with Gold codes / two
   tones, demodulate at u·v intermod and across a frequency sweep → extract the die-specific PDN poles/zeros
   at high SNR. Is the frequency response more die-specific than the scalar/time-domain coupling?
4. DIFFERENTIAL sensor-pair features (sensor_i − sensor_j) to cancel common-mode and isolate spatial-specific.
5. FEATURE SELECTION: keep only channels/zones where intra≫inter (discriminative subset), discard noise.
6. MORE SAMPLES / longer runs per zone (tighter coefficient estimates → higher intra similarity).
7. COMPOSITE: train the actual frozen-LM + linear adapter to USE the die-specific coupling pattern in its u·v
   readout, so the computation it performs is die-bound (does this make räkna-unikt at the system level, or is
   it just identity re-badged?).

## Also
8. Standard PUF SNR/reliability machinery we should adopt: uniqueness vs reliability (inter/intra Hamming),
   BER reduction, fuzzy extractors/helper data, temporal majority voting, TMR — which give the biggest
   separation gain for an analog/reservoir-style response like ours? Real cites.
9. Brutal verdict: with only 2 physical dies of this model, can "clean separation" even be ESTABLISHED (n=2
   dies), or do we need more dies? What is the minimum convincing experiment, and is the whole räkna-unikt
   goal worth more hot-run budget vs consolidating (generic compute + identity/freshness from separate
   channels)? Give the single highest-leverage next experiment with pre-registered acceptance criteria.
