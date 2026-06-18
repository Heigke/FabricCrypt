# openai response (gpt-5) — 147s

Short verdict up front
- Your “kill” of the amplified positive was methodologically correct under the threat model “attacker may change the adapter.” With that model, linear-only wins are circular: you can always let the control use the same or a slightly richer class and it will match or beat. You did the right thing to sweep quadratic and RFF on u.
- However, the stronger and security-relevant bar for “räkna” is not “compute a nonlinearity of my commanded drive u,” but “the die performs die-specific, nonlinear mixing with exogenous state, so the same u on a different die gives a different but coherent modulation.” That is defensible, measurable, and much harder to spoof.
- You have real but small nonlinearities (IMD>1, cubic residue for PAR3, faint bilinear term). There are plausible, thermally safe ways to amplify them into a clean die-necessity signal under the exogenous-mixing interpretation. I’d prioritize two-tone heterodyne (PDN resonance hunting) and dual-engine bilinear driving with a hidden exogenous bitstream. Protocols below.

Adjudicating the three framing questions

1) Is “rank-limited linear adapter necessity” legitimate, or circular?
- If the attacker is allowed to swap the adapter: circular. Any function of a commanded u that your die+linear readout can do will be computable by a modestly nonlinear readout on u (as your sweep shows). Dambre’s information-processing capacity (IPC) trade-off does not rescue you here: once the control may use polynomial features or kernels on the same u-window, “computation on u” is not a die-necessity claim, it’s a representational choice.
- When is it legitimate? Only if the adapter class and its implementation are enforced and remotely attestable at inference time. Concretely:
  - You need a TCB that measures and attests the exact adapter binary and its weights/configuration, and that the adapter only consumes your specified telemetry channels and u, not additional APIs. For AMD, this means running the adapter inside an attested VM under SEV-SNP or a PSP-backed TEE and having the remote verifier check the measurement before accepting outputs. The server can then reject transcripts unless they carry a valid attestation quote binding to that binary. Without such enforcement, an attacker simply ships a quadratic (or kernel) adapter and drops the die.
  - Even with attestation, the right comparator is “best attacker within the attested class.” So if you freeze to “rank≤4 linear,” linear-only wins are not a security result but an engineering decision. They are acceptable only if you prove the class is enforced (attestation) and still sufficient for LLM performance.
- Non-circular definition to use: “Die computationally necessary under class C” = For a specified, attested adapter class C and allowed inputs I (telemetry-only; u-window-only for the control), there exists a task T such that:
  - Acc(die+C)>chance+ε,
  - sup over g∈C Acc(g(u-window-only)) ≤ chance+ε’ with ε’<Acc(die+C)−chance−δ,
  - and replacing the die or replaying past traces reduces Acc below chance+ε’.
  Without enforcement of C, this reduces to pass-by-construction.

2) Is “function of commanded drive” the wrong bar?
- Yes. Your own “structural theorem” is correct: allowing the control even mild nonlinearity on the same commanded u makes the die unnecessary for command-only functions.
- A defensible requirement (2) is: the die performs die-specific, nonlinear mixing of exogenous state with the command.
  - Exogenous here means a signal not inferable from u alone at the adapter interface at test time. It can be hidden randomness (e.g., RDSEED-derived timing/amplitude dither kept inside an attested enclave) or uncontrolled physical state (PDN/VRM microstate, uncore activity) that interacts nonlinearly with your commanded drive.
  - The property you want is closer to a dynamic PUF (challenge = u; response = die-specific dynamic function of u and exogenous microstate).
- Decisive experiments:
  A. Cross-die generalization (die specificity): Drive Ikaros and Daedalus with identical u and any exogenous pattern e you intend to use (if secret, produce e inside each die’s TEE so the outside can’t see it). Train a linear readout on die A telemetry to solve a nonlinear task; evaluate the same readout weights on die B telemetry with the same u (and same e if public). Expect a large drop only for real die-specific mixing. Controls:
    - Train/test on A (intra-die) vs train on A, test on B (cross-die).
    - Matched u-only readout trained on A should transfer to B if the function is u-only.
    - Phase-shuffled surrogate on A should not replicate A’s accuracy.
  B. Exogenous-mixing necessity: Supply two bitstreams u and v, where v is exogenous at adapter IO (e.g., derived from RDSEED inside the TEE and injected only by modulating burst timing/amplitude). Show:
    - Die+linear on telemetry achieves XOR(u, v) or PAR3(u, v, u lag) above chance and above a u-only control of any allowed nonlinearity in C.
    - If you remove v modulation (same u, fixed timing/amplitude), performance collapses to the u-only baseline.
    - Replaying old telemetry tied to unrelated v’ fails (freshness).
  C. Replace-die test: Repeat A/B on both dies; show the readout trained on A with its e/u does not transfer to B even when given B’s e/u (die-specific function class).

3) Given small but real nonlinearities, what to try next?
Ranked by estimated probability of getting a clean, reportable >0.05 win against a fair, enforced control and phase-shuffle surrogate, with concrete protocols. All candidates assume you enforce the adapter class via attestation; otherwise the attacker can just beef up the control.

1. Two-tone heterodyne at PDN resonances (probability 0.55)
- Rationale: Your IMD>1 result means the PDN is a real nonlinear mixer. Heterodyning lets you push nonlinearity into a controllable low-frequency beat you can read with high SNR at low duty, thermally safe.
- Protocol:
  - Resonance scan: With thermal guardrails, sweep single-tone burst cadence f from 5–200 Hz (change STEP_S) and measure the die channel power at f, 2f, 3f. Pick 2–3 local maxima of 2f/f or 3f/f as candidate resonances. Keep duty at ~10–15%.
  - Two-tone excitation: Choose incommensurate tones f1, f2 near two resonances or one resonance and one off-resonance (e.g., f1≈45 Hz, f2≈62 Hz), both with 4 ms bursts. Alternate tone per step or interleave mini-bursts inside one step so the average duty remains <15%.
  - Readout: Sample transients at 500 Hz; build features = [all taps, first differences, and 1–2 lags] per tone mark. Demodulate the telemetry at 2f1−f2 and 2f2−f1 using synchronous detection: multiply by sin/cos at those beat frequencies and low-pass over 1–2 s to form virtual nodes.
  - Tasks: XOR across tone-select bits; PAR3 across {tone1-edge, tone2-edge, lagged tone1}; NARMA-lite on demodulated envelopes.
  - Controls: u-only with polynomial and kernel features on the exact same u(t) and tone schedule; phase-shuffled surrogate. WIN threshold +0.05.
  - Safety: monitor temp; if >74 C, cool-off loop; select tone cadences that keep average package power safe.

2. Dual-engine bilinear drive with hidden exogenous stream (probability 0.45)
- Rationale: You saw a genuine bilinear term in power. Driving two independent engines (GPU and CPU/NPU) with independent bitstreams creates physical cross-terms you can exploit; secrecy of one stream satisfies the “exogenous” bar.
- Protocol:
  - Inputs: u(t) drives GPU bursts as before. v(t) drives short CPU AVX2 or NPU bursts time-shifted by τ relative to u(t). Generate v(t) from RDSEED inside the attested adapter so v is not observable at the adapter IO. Use 3–4 ms bursts, 10–15% duty each, with randomized staggering so (u,v) edges sometimes coincide.
  - Features: same transient taps; also include per-step integrals and a 1-lag of the per-step mean.
  - Tasks: XOR(u, v), PAR3(u(t−1), u(t−2), v(t−1)), or “gated recall” where the presence of v edges gates recall of u.
  - Controls:
    - u-only readouts with polynomial/kernels.
    - v-ablated condition (hold v fixed) to show necessity of exogenous mixing.
    - Replay of an old (u, v’) with v’≠v to test freshness.
  - Expectation: Die+linear beats any u-only control because v is not available; phase-shuffle kills the beat structure. The bilinear physical term increases separation with only linear readout.

3. Resonance-seeking time-mask input (probability 0.40)
- Rationale: Classic RC masking improves separability by scattering an input across virtual nodes with diverse time constants; here, pick masks matched to PDN modes.
- Protocol:
  - Identify K transient taps that correlate most with edge-induced droop in your transient dataset (e.g., via mutual information with the edge indicator).
  - Build a fixed ±1 mask m(t) of length W (e.g., W=16) and present u(t) through W sub-bursts per step at sub-step offsets that excite different taps (stagger within the 30 ms step). Keep total on-time per step fixed.
  - Features: flatten all taps across sub-bursts; add a 1-lag of tap-wise means.
  - Tasks: higher-order parity across masked sub-bursts, NARMA-lite over masked sequence.
  - Control: same masking applied to u-only features (no die), polynomial and kernel. WIN if die+linear > mask-on-u by >0.05 and > surrogate.

4. Multi-level amplitude drive with edge-slope encoding (probability 0.35)
- Rationale: Vdroop nonlinearity often depends on dI/dt and absolute current; using 3–4 load levels provides quadratic and cubic terms physically without pushing temperature.
- Protocol:
  - Quantize burst intensity into 4 levels by changing the working set size or number of concurrent blocks so the edge slopes are distinct but safe.
  - Code two bits {b1, b2} into level and another bit b3 into inter-burst spacing jitter (RDSEED-driven, secret). Read multi-tap transients; form features of raw taps and differences.
  - Tasks: recover b1 XOR b2; and PAR3(b1, b2, b3). Controls include u-only (just the public levels without b3) with polynomial features; WIN requires beating that control by >0.05.

5. Heterodyne demod of single-tone with randomized phase (probability 0.30)
- Rationale: Randomizing edge phase relative to the 500 Hz sampler can convert phase-to-amplitude via nonlinearity, creating exogenous mixing with sensor clock jitter and PDN microstate.
- Protocol: Drive a single cadence f but vary the burst onset within the 30 ms window using secret RDSEED phase offsets; demodulate the telemetry at harmonics of f with synchronous detection; tasks on parity of secret phase bins. u-only control can’t observe the phase; die mixing maps it into amplitudes linearly readable.

All above protocols stay within your thermal constraint by maintaining short bursts, low duty, and demodulating over long windows to increase SNR.

Why your original amplified positive both “worked” and was right to kill
- The die+linear readout achieving XOR and PAR3 from transient taps was real and diagnostic: it verified the physical nonlinearity channel (edge physics) and your time-multiplexing strategy. That is scientifically valid amplification evidence.
- It is not, by itself, a defensible “necessity” claim because the adversary could add quadratic features on u. With no enforced adapter class, the kill is correct.
- The residue for PAR3 (+0.048 over quadratic-on-u, +0.080 over surrogate) is real but small. Given Dambre’s IPC trade-off (Sci Rep 2012), you will likely need either (i) better SNR at the specific cubic order (the heterodyne and multi-level protocols do that) or (ii) exogenous mixing so u-only controls are fundamentally blind.

Literature that bears on Q1–Q3
- Reservoir computing fundamentals and what linear-vs-nonlinear readouts can and cannot do:
  - Appeltant et al., “Information processing using a single dynamical node as complex system,” Nature Communications, 2011 (delay-based RC, time-multiplexing).
  - Dambre et al., “Information processing capacity of dynamical systems,” Scientific Reports, 2012 (IPC trade-off; why your linear readout needs physical nonlinearity).
- PUF modeling and why command-only functions are weak for security:
  - Rührmair et al., “Modeling Attacks on Physical Unclonable Functions,” CCS 2010; and subsequent surveys by Rührmair and Holcomb (Proc. IEEE 2014). These show that many PUFs based on simple structures are learnable by ML if the challenge/response mapping is accessible; by analogy, any u-only nonlinearity is learnable by the attacker’s upgraded adapter.
- Attested execution to make adapter-class restrictions meaningful:
  - AMD SEV-SNP architecture docs (AMD whitepapers and architecture manuals, 2020–2023) describe remote attestation and measurement binding of guest code. This is the practical way to enforce “class C” on commodity AMD hardware. Without attestation, adapter limits are not a security property.
- Weight locking / hardware-locked models:
  - Clifford et al., “Weight-Locking: Bind neural network weights to specific hardware,” arXiv:2405.20990 (2024). Relevant as an orthogonal mechanism; underscores that enforceable binding needs either cryptographic attestation or hardware-derived secrets.

Brutal verdict on whether to keep hunting (2)
- If your deployment cannot enforce the adapter class (e.g., no SEV-SNP/TEE attestation of the linear adapter and no guarantee about what the attacker runs), then requirement (2) phrased as “die computes a nonlinearity of u” is not defensible. In that setting, the honest paper is: “Die-unique identity and freshness are real; die computation on a commodity APU is not a robust locking primitive against an attacker who can change the adapter.”
- If you can enforce the adapter class and keep an exogenous stream secret at the adapter boundary (e.g., RDSEED-derived timing inside an attested enclave), then reframing (2) as “die-specific nonlinear mixing with exogenous state” is the right target. Under that framing, I estimate >50% chance that one of the heterodyne/bilinear protocols above yields a clean >0.05 win vs fair, enforced controls and a phase-shuffle surrogate, within your thermal envelope.

Concrete acceptance criteria to pre-register (to avoid post-hoc bias)
- Die+linear beats u-only control within the enforced class C by >0.05 on held-out data and beats a phase-shuffle surrogate by >0.05.
- Cross-die transfer drops by >0.1 absolute (train on A, test on B with same u and, if public, same e).
- Replay attacks fail: replacing live telemetry with any previously recorded trace reduces accuracy to ≤chance+0.02 with the same u.
- Thermal log shows compliance (peak <99 C, duty <15%, average temp <75 C).

Final notes on your data
- Your transient_vdroop linear readouts achieving strong recall and 0.78 on XOR_12 show the time-multiplexed nodes are doing their job; the small margins over surrogate on PAR3 confirm genuine, but weak, cubic dynamics. That’s exactly the regime where heterodyne and dual-engine bilinear inputs pay off: they convert weak distributed nonlinearity into a narrowband feature you can extract with linear demodulation at high SNR without raising average power.

If you want, I can turn the top two protocols into drop-in code skeletons that reuse your H7 collection harness and add: (i) tone scheduling and synchronous demod features; (ii) dual-engine burst orchestrator with attested RDSEED masking; and (iii) cross-die train/test splitters.
