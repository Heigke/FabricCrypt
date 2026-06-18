# O15 — Brief v3 storyline reorganization

## What you are reviewing

`nsram_proposal_short.pdf` — a 6-page funding brief Eric Bergvall is
preparing for Mario Lanza (KAUST tape-out lead) and Sebastian Pazos
(Lanza's postdoc, NS-RAM cell measurement author). Deadline 2026-05-06.
Mario said in a recent meeting: *"we need to bring it down to the
transistor level"*.

We have substantial new material (slide inventory + binning audit
finding) that argues the brief needs structural reorganization, not
just polish. We need a third opinion on the proposed reorganization
before committing 2-3 hours to a v3 rewrite.

## What's in this packet

1. **`nsram_proposal_short.pdf`** (6 pp, current v2 — already had two
   substantial revisions today). Read this first to understand what
   we have.

2. **`zoom_slides_inventory.md`** — a 30-image visual inventory of
   slides Sebas and Mario presented during two Zoom meetings (2026-03-20,
   2026-04-30). Eric does NOT have the original .pptx decks, only
   screen-captures from the meetings. The inventory groups the slides
   into a 7-step storyline arc.

3. **`probe_v1_findings.md`** — today's binning audit result: pyport
   reproduces BSIM4 v4.8.3 binning arithmetic exactly, but `ags` (the
   saturation-region body-bias coefficient) gets a +60% correction in
   the M2 geometry, the largest of any parameter. Whether ngspice agrees
   on this +60% is the next probe (Stage 6b v2).

## The proposed v3 reorganization

The slide inventory shows that Sebas's natural pitch follows a 7-step arc:

  1. **Market context** — why edge AI needs sub-mW neuromorphic compute now
  2. **Neuromorphic paradigm** — asynchronous integrate-and-fire SNN
  3. **Cell physics** — impact ionization + parasitic NPN snapback
  4. **Design variants** — 2T vs 1T, thin vs thick oxide, all in 130 nm CMOS
  5. **Circuit integration** — how cells couple to CMOS inverters/array
  6. **System validation** — SNN-on-chip simulations
  7. **Comparative advantage** — why NS-RAM beats RRAM/PCM/memristor

Our current v2 brief starts at step 5 (circuit integration) and treats
steps 1–4 implicitly. That's why it reads as algorithm-software-bridge
when Mario wants transistor-level grounding. Five must-have figures from
the slide inventory:

  - **12.26** — semi-empirical impact-ionization model fits (cell physics)
  - **12.27** — measurements vs SPICE simulation overlay (validation)
  - **12.27 (1)** — 2T snapback waveform under transient VD ramps (dynamics)
  - **12.29** — 130 nm triple-well CMOS implementation (manufacturability)
  - **13.31 (1)** — NS-RAM input-neuron block (circuit integration)
  - plus the already-known **13.23** (I-V family with 2T schematic embedded)
  - plus **13.28** (three operating regimes overlaid in one plot)

## Five questions for you

1. **Is the proposed 7-step storyline the right arc for Mario?** Or is
   it Sebas's narrative pushed onto a brief that should keep its own
   narrative? We're asking because Eric is the one asking for the
   funding, not Sebas; the brief shouldn't be a re-pitch of Sebas's
   slides, it should be Eric's case for why a co-design pipeline gets
   built, leveraging Sebas's data.

2. **What is the right opening?** Three candidates:
   (a) Open with Sebas's 2T cell schematic + snapback waveform —
       transistor-level, but feels like leading with someone else's work.
   (b) Open with the differentiable simulator + benchmark monotone
       ordering (current v2 framing) — Eric's case, but high-level.
   (c) Open with the **co-design loop diagram** — show that pyport
       inverts measurements → algorithm gains → tape-out spec, with
       Sebas's cell as the substrate, our software as the loop closer.
       Most honest framing of who-does-what.

3. **Should we keep or cut the algorithm benchmark section?** It's
   currently ~1.5 of 6 pages (the five B.5 results). Mario's "transistor
   level" comment might mean "less of this, more cell physics". But
   the monotonic task-difficulty ordering IS the strongest empirical
   claim for the dual-topology tape-out recommendation. We could
   compress to one page (table + one paragraph) to make room for
   transistor-level content; or we could keep it and trim Methodology.

4. **Visual gaps to flag.** The slide inventory identifies five gaps
   in the visual record (no full-chip micrograph, no oscilloscope trace
   of array spike output, no quantitative energy comparison, no
   PVT-robustness data, no yield/reliability data). Should the brief
   (a) call these out as M3/M6/M9 acceptance criteria, (b) ignore
   them and let Mario assume they exist, or (c) explicitly request
   them from Pazos as the M3 contingency?

5. **The binning audit result.** Pyport's binning arithmetic is
   correct. The `ags` parameter gets a +60% correction from `pags`. If
   ngspice agrees on this (probe v2 will tell), the 0.87 dec gap is
   in downstream `ags`-consuming code. If ngspice disagrees, it's a
   deeper bug. Either way, the brief currently commits "1-2 days
   coding, ~3 days verification" for M3a closure. Is that defensible
   given today's localisation, or should we widen the estimate to
   "1-2 weeks" to be safe?

## What we want from you

Five labelled answers, one per question. Be terse — Eric will read
your three responses (OpenAI, Gemini, Grok) side by side and
synthesize. **Do not repeat the question stems.** **Do not pad.**

If different responders converge, Eric will adopt the consensus. If
you disagree, say so explicitly so the disagreement surfaces. Aggressive
critique welcome.

For Grok specifically: you have the brief PDF as an inlined text
extract (Sebas's actual slides are described textually in the slide
inventory MD, since the JPEGs are not inlinable). The PDF text and
the inventory MD give you the full picture even without images.

For OpenAI and Gemini: you have the PDF + the inventory MD + the
probe-findings MD as multi-modal inputs.
