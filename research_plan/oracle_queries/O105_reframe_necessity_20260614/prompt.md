# Oracle packet O105 — did we mis-frame "die-needed computation"? Reframe + amplification check.

You are an adversarial reviewer. Police BOTH biases: our bias toward wanting to succeed AND our bias toward
killing our own positives too eagerly with an unfairly strong control. Be concrete and technical. Cite ONLY
real work (no fabricated arXiv IDs). Rank every suggestion by probability and give a concrete protocol for
any you rate >20%.

## Goal (unchanged)
Make a frozen LLM **constitutively dependent** on ONE specific AMD Strix Halo gfx1151 APU die: a different
chip, or replayed/recorded signals, must break its output, while on the real die it writes fine. Three
"body" requirements: (1) UNIQUE to the die, (2) RÄKNA = perform genuine nonlinear COMPUTATION the model
needs (not just a load reading), (3) FRESH = live, non-replayable.
SOLVED: (1) CPPC fused per-core ranking 75% distinct ikaros-vs-daedalus + live 2nd-order dynamics 14× cross-die; (3) RDSEED.
STUCK: (2) genuine nonlinear computation.

## NEW since O104 — an amplified positive that we then killed (this is the crux; tell us who is right)
We took the "amplify the weak nonlinearity" path. Vdroop/di-dt is ELECTRICAL+FAST, so we excited it with
sharp 4 ms max-GPU bursts at 13% duty (chip stays cool) and read the post-edge SETTLING TRANSIENT as 12
time-multiplexed virtual nodes (Appeltant-style), giving a 120-dim reservoir state.

- AMPLIFIED PROBE REPORTED POSITIVE: a rank-4 LINEAR readout of the die transient did XOR(u[t-1],u[t-2])
  at 0.644 where a rank-4 LINEAR readout of a 4-tap u-window scored 0.502 (chance). Mechanism is physically
  real: di/dt droop responds to EDGES, and |u[t]-u[t-1]| IS XOR(u[t-1],u[t]); the substrate computes the
  nonlinearity via edge physics, the linear readout harvests it. Also PAR3 (3-way parity): die 0.597 vs
  u-window-linear 0.514, and a pairwise-product (degree-2) readout on u also fails PAR3 (0.54).

- WE THEN KILLED IT with a FULL SWEEP (24 dataset×task cells, fair controls + phase-shuffle surrogate):
  0 WINS. The "fair control" allowed a QUADRATIC on the same u-window, and quadratic-on-u does XOR = 1.000
  (XOR is just a product of two bits). So the die was "needed" ONLY if you forbid the linear adapter from
  squaring its own inputs. The single surviving residue: PAR3 on steady-state, die=0.580 beats phase-shuffle
  surrogate by +0.080 AND quadratic-on-u by +0.048 (3-way parity needs a cubic, which the quadratic lacks),
  but it's only +8pp over chance and below our +0.05 win bar. (Independent: two-tone IMD shows 1.8x excess
  over a static monotone map = genuine dynamical nonlinearity; a 2D bilinear probe found ONE channel,
  power/energy-rate, with a faint +0.138 genuine bilinear term. Both real, both tiny.)

## The three framing questions we cannot resolve ourselves (THIS is what we need adjudicated)

1. **Is the rank-limited-LINEAR-adapter necessity legitimate, or circular?** The deployed H7 system IS a
   frozen LM + a rank-<=4 LINEAR adapter on telemetry. Under that adapter the die genuinely adds XOR the
   adapter cannot do. But we "killed" it by allowing the control a quadratic — i.e. we compared against an
   adapter STRONGER than the one we ship. Is "we commit to shipping a linear adapter, therefore die-needed"
   a real security property, or is it PASS-BY-CONSTRUCTION (we made the die necessary by crippling the
   alternative)? An attacker who wants to drop the die would just ship a quadratic adapter on the command.
   What is the correct, non-circular definition of "the die is computationally necessary"?

2. **Is "compute a function of the COMMANDED drive" the wrong bar entirely?** Our structural theorem says any
   function of a drive WE command is self-computable by a modestly-nonlinear readout on the command, so the
   die is never needed for command-functions. But maybe #2 ("räkna") should not mean "die computes XOR of my
   command." Maybe the right property is: the die performs a die-SPECIFIC nonlinear MIXING of EXOGENOUS
   (uncommanded) physical state with the command, so the SAME command on a DIFFERENT die yields a different
   but still coherent modulation. That is closer to a "PUF with dynamics" / physically-unclonable RESERVOIR
   than to a logic gate. Is that a defensible reinterpretation of requirement (2), and if so what is the
   decisive experiment (we can drive ikaros AND daedalus with identical command streams and compare)?

3. **Given three independent REAL-but-tiny nonlinearities (IMD 1.8x; PAR3 cubic residue +0.08 over
   surrogate; ch5 power bilinear +0.138), is there an amplification or readout we have NOT tried that turns
   one into usable die-NECESSITY?** Candidates we know of: time-multiplexed virtual nodes (done), two-tone
   IMD readout (done, not yet used as a compute substrate), PDN resonance excitation, multi-level / chaotic
   drive instead of binary, deeper temporal integration. What would YOU try, ranked, with a protocol —
   subject to HARD THERMAL SAFETY (99C ACPI trip = instant reboot; we already hit 92-98C several times;
   sustained near-throttle operation is OFF THE TABLE).

## Also
4. Anything in 2024-2026 literature on physical reservoir computing / PUF-reservoirs / hardware-locked models
   that bears on Q1-Q3 and we likely missed (we know Clifford et al. 2405.20990 weight-locking; Appeltant
   delay-reservoir; Dambre IPC bound; Wright/McMahon PAT). Real cites only.
5. Brutal verdict: should we keep hunting (2) at all, or is the honest paper "die-unique IDENTITY + FRESHNESS
   are real; die-COMPUTATION is not achievable on a locked commodity APU, and here is the rigorous proof"?
