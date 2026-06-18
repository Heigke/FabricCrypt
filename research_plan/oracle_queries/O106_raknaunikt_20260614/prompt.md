# Oracle packet O106 — how to make the die's u·v computation DIE-SPECIFIC (not just generic)? Last piece.

Adversarial reviewer. Police BOTH biases (wanting to succeed; giving up too early). Cite ONLY real work.
Rank every suggestion by probability of yielding a clean, thermally-safe positive; give a concrete protocol
for any >20%.

## Where we are (verified this session)
Goal: frozen LLM constitutively dependent on ONE specific AMD Strix Halo gfx1151 APU die. Triad: (1) UNIQUE,
(2) RÄKNA = genuine nonlinear computation the model needs, (3) FRESH.
- (1) SOLVED: CPPC fused per-core ranking 75% distinct ikaros-vs-daedalus; 2nd-order dynamics fingerprint 14×.
- (3) SOLVED: RDSEED.
- (2) RÄKNA — **NEW POSITIVE this session**: drive u=GPU bursts and v=CPU bursts (sharp di/dt edges, low
  duty); v is WITHHELD from the control. The die physically computes the PRODUCT u·v via shared-PDN power
  contention. A LINEAR readout of telemetry does XOR(u,v)=0.654 on ikaros, 0.701 on daedalus; 300-shuffle
  null p95≈0.54, p=0.000; u-only(any poly)=0.53≈chance; **u&v-LINEAR=0.49≈chance** (a linear mix of both
  inputs can't XOR → the silicon really made the product); physical u·v partial-R² max channel = 0.089
  (power/energy channel = GPU×CPU envelope contention). So the die genuinely multiplies. NON-circular.

## The problem: the computation is GENERIC, not DIE-SPECIFIC
Cross-die transfer test (train XOR(u,v) readout on die A, test zero-shot on die B, identical command streams):
- ikaros→daedalus XOR 0.615→0.584 (per-die renorm); daedalus→ikaros 0.701→0.633.
- BUT the u-only RECALL control — which MUST transfer (same command) — ALSO dropped hugely (raw u-only drop
  +0.31 >> XOR +0.11). When the must-transfer control fails too, the cross-die gap is NOT die-specific
  mixing — it is an OPERATING-POINT/THERMAL confound (daedalus pinned 99°C vs ikaros cooler).
- After per-die renorm, in the clean direction (ik→da): u-only transfers 0.868 AND XOR transfers 0.584 — i.e.
  NO EXTRA die-specificity in the mixing beyond the generic command readout.
=> The u·v mixing is a generic property of the AMD APU PDN, the SAME on both dies. "compute = identity" fails.
Currently strengthening the v-drive (heavier CPU contention) to lift the weak u·v term (d_v was only 0.59).

## The question: have we missed a way to make the COMPUTATION ITSELF die-specific?
Adjudicate and rank these candidate ideas; add any we missed; flag fatal flaws. THERMAL HARD LIMIT: 99°C ACPI
trip = instant reboot; we already hit 99-100°C; sustained near-throttle is OFF the table; protocols must be
low-duty / sharp-edge / demodulated.

1. **v = the die's OWN uncommanded microstate** (intrinsic leakage / thermal noise / neighbour-core activity)
   instead of an external CPU stream. Then u·v_die mixes the command with a die-intrinsic, unclonable signal,
   so the mixing is die-bound by construction. Is this defensible, and how to instantiate v_die cleanly so it
   is (a) genuinely die-specific, (b) reproducible enough to train an adapter, (c) not just the linear
   fingerprint we already have?
2. **Compare the u·v COEFFICIENT VALUE across dies** (not readout transfer): the nonlinear gain of the u·v
   term is set by die parasitic L / regulator trim (manufacturing variation). Is the coefficient itself a
   die-distinguishable quantity, and how to measure it WITHOUT the thermal confound (e.g. matched thermal
   band, or ratio to a linear coefficient that cancels temperature)?
3. **Three-way product u·v·g_die** — the die's static bias (CPPC/leakage, already 75% die-distinct)
   multiplies INTO the mixing gain, making the computation die-specific by construction. Protocol?
4. **Read u·v THROUGH die-specific transient time-constants** — the settling-transient taps have die-unique
   RC/L; the per-tap u·v structure may be die-specific even if the scalar gain is not. How to test?
5. **Generic kernel + die-specific linear fingerprint = die-specific COMPOSITE function** — our prior
   embodiment training (frozen LM + adapter) already got 2.2-2.6× cross-die separation. Is "räkna unikt"
   better defined at the trained-system level (the composite is die-bound) than at the isolated-kernel level?
   Does that count as requirement (2), or is it just (1) identity re-badged?

## Also
6. How do physical/RESERVOIR PUFs (Krause et al., Neuromorphic Comput. Eng. 3:034001 2023; Sarantoglou et al.
   arXiv:2505.11448 2025 photonic compute+PUF) get DEVICE-SPECIFIC responses from generic nonlinear dynamics?
   What is the exact mechanism that makes their COMPUTATION (not a stored key) device-bound, and can it map to
   commodity-APU PDN/telemetry? Real cites only.
7. Brutal verdict: is "räkna unikt" achievable on a locked commodity APU, or should we publish "die computes
   u·v (generic) + identity/freshness from separate channels" and stop? If achievable, the single highest-
   probability experiment, with protocol and pre-registered acceptance criteria.
