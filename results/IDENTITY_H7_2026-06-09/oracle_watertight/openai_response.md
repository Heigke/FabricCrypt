# openai response (gpt-5) — 144s

Hostile review (hardware security + ML + reservoir computing)

1) The single killer attack

Attack: software system-identification + emulator of the “body” that inlines the die response and bypasses the silicon entirely at inference.

Why it kills your claims
- Freshness: you never force the model to depend on a live measurement that a verifier can distinguish from a replay. All “body outputs” are deterministic functions of public inputs u (and, for your cache gate, of a,b). An attacker fits a surrogate f̂ that maps the same public inputs to the model’s internal “body features/bits” and then replaces BodyGate.gate(·) and die_features(·) with f̂(·). Your own code has no attestation or nonce-binding, so the model cannot tell live silicon from a stub. This defeats “fresh/live” and collapses to pure software.
- Unclonability: the only non-trivial target you ever ask the body to supply is XOR of lags. XOR is derivable from public inputs; even your reservoir readout is a linear classifier on transient features; a small query-only probe attains 0.96+ agreement. Once an attacker learns f̂, the die adds nothing.
- Die uniqueness: your reported “gap” largely comes from covariate shift (drive magnitude α, windowing, temperature/governor) and per-condition tuning. An emulator can equalize drive and re-center features, reconstructing the own-die decision boundary on foreign silicon or in pure software. Uniqueness vanishes when drive-equalized.
- Multi-layer and “load-bearing” don’t help: the linear-bottleneck trick prevents software-only models from doing parity, but it doesn’t prevent a software stub from returning the one bit your LM bottlenecks on. The attacker just feeds the “correct” bit from the emulator.

Concrete attack steps (works against every script you posted)
- Collect a few thousand (challenge, response) pairs for each used “body”:
  - Cache XOR: trivial; f̂_cache(a,b)=a^b.
  - Analog reservoir: drive exactly your u[t], record logits and true y[t] or your Xz die features if available; otherwise use only (u-window, y) as supervision.
- Fit f̂:
  - For cache XOR: hard-coded.
  - For analog: ridge/logistic on a few lags and their pairwise products; optionally a small GRU. This reaches ≥0.95 on XOR_t1t2 from public u alone; if you insist on mimicking Xz, do PCA and fit a linear state-space model X̂[t]=A X̂[t-1]+B u[t]+C ϕ(u[t-k..t]) with ϕ including low-order products. This reproduces your die_head logits sufficiently to pass your evals.
- Patch:
  - Monkey-patch BodyGate.gate to return f̂_cache(a,b).
  - Replace die_features(Tn) call with f̂_analog(u-window) or directly return die_head logits. Keep your standardization path to match means/vars.
  - If you “nonce-bind,” include the nonce in the input to f̂; train on a few thousand nonces. There’s no TEE or timing constraint, so replay/emulation still passes.
Result: native, xor_sim, and analog “own-die” all pass without touching silicon. Foreign vs own-die gap collapses after drive equalization, and “freshness” is gone.

Does any honest version survive this?
- Only if you simultaneously do all three:
  1) bind predictions to a public random nonce that is chosen after the model receives the input (prevent precomputation),
  2) force an actual live physical computation whose response depends on high-entropy challenge bits and the chip’s idiosyncratic dynamics (not a short, public-memory function of u),
  3) provide a verifiable liveness channel (time-bounded, attested origin) so replay/emulation cannot satisfy the verifier’s freshness check.
- On commodity AMD without a TEE that can attest “this exact binary drove this exact GPU workload now,” you cannot make this watertight for a remote adversary. You can make it robust for a local reviewer with physical custody, time-bounded challenges, and instrumented checks. See redesign below.

2) Minimal watertight redesign (uses macro + micro + analog; one path is simultaneously load-bearing, fresh, and die-unique)

Threat model this covers
- Reviewer has the box on their desk and runs your code; attacker does not control the host. We protect against software emulation / replay / query-cloning. We do not claim remote attestation (see Section 3).

Core idea
- Make the model’s observable outputs depend on a per-query one-time body bitstring K that:
  - is produced by a high-entropy challenge C (nonce) chosen by the verifier after the input is fixed,
  - is computed by a 3-layer physical pipeline that includes genuine nonlinearities at each layer,
  - has large challenge space and good inter-die distance, low intra-die BER under operating variations,
  - is the sole nonlinearity available to a bottlenecked head in the LM’s forward path.
- Any emulator/replay that does not run the live body with the current C cannot supply K. Because the final decision depends on K⊕(parity of context bits), query-only cloning fails.

Three-layer body and roles
- Micro (L3 destructive interference): compute fast, local Boolean mixing of secret context bits, gµ = XOR(a,b), using your two-streamer organ. You were already stable on 00/11 boundary; keep that. This serves as a steering signal for the analog stage (do not expose gµ outside the body).
- Macro (CPU↔GPU power arbitration): create a die- and board-specific nonlinear scheduling field by:
  - launching CPU L3 streamers on P pinned cores whose affinity and on/off cadence are keyed by C and gµ,
  - simultaneously submitting GPU burst trains keyed by C||gµ through a PRBS that modulates burst spacing and size,
  - binding the SMU power cap during the excitation (write, then read-back) so you actually operate on the contention manifold, not in linear regime.
  This ensures the GPU di/dt trajectory depends nonlinearly on CPU activity and on chip-level arbitration (voltage droop management and package impedance are per-die).
- Analog (voltage droop transient reservoir): use your “amplified” transient capture with taps NTAP and channels N_CH to form the reservoir state X(C, gµ, die). This is the only strong nonlinearity + fading memory. Read it with a linear head to produce K.

Challenge protocol and binding (freshness)
- Verifier picks a 128–256 bit nonce C after giving the LM the text window. Verifier sends C and a strict deadline Δ (e.g., 120 ms) to return the next token.
- Body excitation:
  - Construct a per-query burst schedule uC(t) = PRBS(C, salt=hash(context_id) || gµ), length T≈30–80 ms, with two PRBS families conditioned on gµ.
  - Start CPU streamers pattern S(C) on P cores with duty keyed by another PRBS seeded by C. Each PRBS family must push the CPU right below/above a knee of the SMU sharing curve (nonlinear).
  - Drive GPU bursts keyed by uC while CPU load is on, with “sync inside the burst” as you already fixed. Record NTAP taps × N_CH sensors (same as in your vdroop harness).
- Reservoir readout:
  - Standardize to a calibration collected for this die and this operating condition (temp, governor) at session start.
  - Compute K = sign(Wdie · ϕ(X)), where ϕ includes taps, 1–2 lags, and one bilinear cross-channel term per your IMD observation. Output m=64–256 bits per query.
- Return only the LM token; do NOT expose K. Keep C and timestamps in the audit log. If the end-to-end runtime exceeds Δ + jitter budget (measured on reviewer’s machine), treat as fail-freshness.

LM architecture (real text, load-bearing, body-only nonlinearity)
- Base LM: normal byte-Transformer for text positions (your text_embodied script is fine).
- Query positions: as in your text_embodied, but:
  - The only path to the 2-class logits is a linear bypass over pooled embeddings plus a linear head over the body bit(s).
  - Replace single g by a small binary vector Kq of length m (e.g., 16) consumed through a learned linear combiner; the logit is L = w0 + w⊤Kq + v⊤pool(context). This keeps it linear-in-K but requires the real K to be present.
  - The target at query position is y = QTOK0 + (XOR(a,b) ⊕ h(Kq)), where h is a fixed public linear hash, e.g., take 1 bit as parity of Kq. That way, correctness depends on both XOR(a,b) and Kq. Without live Kq the best an attacker can do is 0.5 on query tokens even if they know a,b.
Training (live-in-the-loop, minimal)
- Pretrain the transformer on text-only loss (no body).
- Collect a die/session calibration set for the reservoir: for 10^3 random nonces C at three temps/governors, measure X and fit Wdie to map X→target bits of a known PRF (e.g., parity of Ktrue). Don’t use recorded X at inference; this is only to set the die_head and normalizer.
- Fine-tune the LM on mixed batches:
  - For query positions, build targets y using the true gµ= BodyGate_XOR(a,b) and live Kq from the reservoir path with random nonces C. Train with plain cross-entropy.
  - For text positions, normal language loss.
- No penalty terms; the bottleneck enforces dependence on Kq+gµ.
Inference protocol (replay-proof for a local reviewer)
- Verifier-run harness enforces:
  - picks fresh C per query,
  - measures wall-clock from challenge issue to token returned,
  - checks that the observed latency distribution matches a previously established live baseline (microseconds-scale stub vs tens of ms live body is easy to separate),
  - randomizes CPU core affinities and governors between sessions to prevent a cached-replay from matching timing.
- Failure if:
  - deadline exceeded,
  - per-query timing or power-telemetry (optional: read RAPL/SMU counters) is inconsistent with live excitation,
  - ablation tests below fail.

Ablations/controls the reviewer should demand and you must pass
- Load-bearing (LM):
  - native: full body → low query PPL; software-truth (replace gµ with XOR(a,b) but keep h(Kq)) must fail (because Kq is required).
  - zero/random Kq: collapse to chance at query tokens; overall PPL increases by ≥1.4× at your chosen query density.
  - cache-only: feed gµ but zero Kq → chance at query tokens.
  - analog-only: supply Kq but randomize a,b → chance at query tokens.
- Freshness:
  - replay: reuse Kq(Cold) for Cnew → chance.
  - shuffled: permute Kq across timesteps with the same marginal stats → chance.
  - deadline: increase Δ to allow stub; now timing-based check can distinguish stub vs live (expect bimodal latency).
- Uniqueness:
  - own-die vs foreign-die: fixed set of nonces and same inputs; foreign Kq plugged into own LM heads → near-chance.
  - drive-equalized foreign: actively match drive amplitude (α) using a calibration loop; the gap must remain.
- Macro nonlinearity actually used:
  - CPU-off: disable CPU streamers → lower K entropy, accuracy on queries drops ≥X%.
  - Powercap-linear: lift power cap above the knee → same as above.
- Live vs recorded:
  - inject a perfect recorded X(C) for a random half of queries (unknown to the LM); those queries fail or violate deadline/timing.

Implementation notes that make the 3 layers genuinely matter
- The CPU streamer cadence and affinities must be keyed by C and operate near the arbitration knee so the GPU’s effective voltage/frequency droop responds nonlinearly (macro layer).
- The GPU burst PRBS must include both spacing and size modulation keyed by C and XOR-steered by gµ (micro→analog composition).
- Read multiple sensors: use your N_CH channels; include at least one with demonstrated bilinear response (IMD>1.5× static). Include 1–2 time lags to exploit fading memory (analog).
- Never expose K or X outside the body; only the final token goes out.

Why this beats the killer attack (locally)
- An emulator built on public inputs cannot predict Kq because Kq depends on the verifier’s nonce and the live, die-specific macro/analog trajectory. Even if the attacker sees previous (C, token) pairs, the mapping includes a one-time-pad bit h(Kq) that is not a deterministic function of the public inputs. Without the die, the attacker’s best is chance on query tokens.
- Replay of recorded X or K fails because C is fresh per query and deadlines/timing betray non-live paths.

3) What is fundamentally impossible on commodity AMD vs achievable

Fundamentally impossible (don’t chase)
- Remote attestation of “this exact GPU kernel and SMU/CPU scheduling just ran now.” There is no widely available GPU TEE or attestation path for gfx11xx. You cannot make a cryptographically strong freshness claim to a remote verifier who doesn’t control the host.
- Unforgeable secrecy of code paths. A local adversary can LD_PRELOAD/ptrace your process and stub BodyGate or the reservoir harness. Without a TEE, software-only anti-spoofing is bypassable.
- Cryptographic PUF guarantees with tiny n (2 dies) and low challenge counts. “Uniqueness gap” on two boards is meaningless; you need population statistics and BER characterization.

Achievable (with care)
- Local, reviewer-observed freshness/uniqueness: timing- and power-profile-bound runs with nonce challenges, where the reviewer controls the host and watchdogs (latency, counters, thermal) and confirms the body was engaged.
- Strong inter-die separation and low intra-die BER under environmental sweeps, given:
  - sharp di/dt transient reservoir with PRBS excitation,
  - macro knee operation (CPU↔GPU arbitration),
  - per-die standardization fitted on a disjoint calibration set,
  - drive equalization across dies to remove gross amplitude covariate shift.

4) Statistics and controls to report

Population and protocol
- Dies: ≥32 distinct boards of the same family; ≥3 sessions per die across days.
- Challenges: ≥1024 fresh nonces per die per session (train/test split by nonce).
- Conditions: temperature setpoints (≈30C, 50C, 70C), power governors (performance/powersave), slight supply variation if available; random CPU affinity maps.
- Calibration: per-die standardization parameters fitted only on a separate calibration split; drive equalization applied to foreign-die comparisons.

Metrics
- Load-bearing (LM):
  - Query-token accuracy and perplexity with 95% Wilson CIs (per-token Bernoulli), per-condition: native, zero-K, random-K, replayed-K, shuffled-K, cache-only, analog-only.
  - Overall PPL shift factor (native vs zero/random): mean with bootstrap 95% CI (block bootstrap by window).
- PUF-like stats (analog K bits):
  - Inter-die Hamming distance: distribution across all die pairs on same nonce set; report mean ± std and the empirical CDF; target ≈0.5 with tight CI.
  - Intra-die BER: same-die, different sessions/temps/governors; report per-condition BER and aggregate; target <1–5% with reconciliation (e.g., BCH) if you use K for masking.
  - Stability across time: same die, same nonce re-run after Δt (hours–days): BER vs Δt curve.
  - Conditional on drive equalization: repeat inter/BER after α-equalization so uniqueness is not an artifact of amplitude.
- Freshness:
  - Latency distributions per condition; show clear separation between live (tens of ms with variance) and stub (sub-ms). Report ROC/AUC of a timing detector distinguishing live vs stub on blinded trials.
  - Optional: SMU power/telemetry traces (mean±CI) showing expected macro-layer signatures only during native.

Ablation checklist (must be in the paper)
- CPU-off (macro removed), Powercap-lifted (linear regime), Foreign-die (equalized), Recorded-X replay, Shuffled-taps, Micro-only, Analog-only.
- For each, report query-token accuracy with 95% CI and the delta vs native.

Implementation hygiene
- Fixed seeds and reproducible PRBS mapping from nonce to drive schedule (public).
- Explicit pre-registered protocol: input fixed → nonce issued → deadline → single token returned; all timestamps audited.
- No per-condition alpha tuning. If you must equalize drive, do it once using a blinded calibration procedure pre-registered before evaluation.
- Train/eval splits on nonces; no leakage of per-nonce features across splits.

Summary
- Your current demos are broken by a simple emulator/replay: no freshness, no unclonability, uniqueness inflated by covariate shift.
- A local, reviewer-acceptable redesign is feasible: challenge-bound, three-layer excitation with macro-knee operation, analog transient readout, and a query-token bottleneck that XORs context with a one-time K(C) bit derived only from live silicon. That makes the die both load-bearing and practically unclonable under local verification.
- Remote, cryptographic-grade guarantees are out-of-scope on commodity AMD today; don’t claim them. Use population statistics, BER, drive-equalized foreign-die tests, and strict ablations to support the local claim.
