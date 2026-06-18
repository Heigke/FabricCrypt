# Oracle query O107 — When does embodiment help? (4-way)

You are one of four oracles (GPT-5, Gemini 2.5 Pro, Grok-4, DeepSeek-R) being
asked adversarially to redirect a research program. Be terse, ≤600 words.
Hostile/falsifying tone preferred — find what's wrong, suggest concrete tasks.

## Background

We've spent a campaign establishing **chassi-bound model identity** on AMD
Ryzen AI Max+ PRO 395 (gfx1151, RDNA3.5). On twin HP Z2 G1a chassis (ikaros,
daedalus):

- **Identity gates 4/4 PASS** (embodiment3 v3): static 256-bit chassi signature,
  reproducible across reboot, distinguishable from twin (G1 = 0.602 bit-flip
  delta, G2 = 1132× separation, G3/G4 = 1.0× normalised binding score).

- **Advantage gates ALL NULL** (V5 hunt H1–H6):
  - H1 routing, H2 sparsity, H3 chip-RNG, H4 plasticity, H5 per-chip LoRA,
    H6 floating-point sensitivity — all show ikaros-trained model on ikaros
    is **NOT significantly better** than daedalus-trained model on ikaros for
    NARMA-10, Mackey-Glass, MNIST-tiny, CIFAR-tiny.
  - 4/6 explicit NULL, 2/6 inconclusive (noise > effect).

**User insight (Swedish, translated):**
> "Maybe we test the wrong things — where identity doesn't help. Think hard.
> When IS having a body actually beneficial? Survival? We can't destroy the
> computer but maybe there's something where reality-coupling matters."

The reframe: **abstract benchmarks (NARMA, MNIST, CIFAR) are body-agnostic by
construction**, so embodiment cannot help. We need tasks where THE TASK IS THE
BODY ITSELF.

## Questions (answer all 10, terse)

1. We've shown chassi-bound model identity (4/4 gates PASS) but no advantage
   on NARMA-10/MNIST/CIFAR. The user proposes: **embodiment helps when the TASK
   is about the body**. List 5 concrete task categories where having a body
   genuinely gives a model a performance advantage that a body-agnostic model
   cannot match.

2. **Self-prediction** (model predicts its own next-step substrate state,
   e.g. power+thermal+latency at t+1 given t-99..t): is this trivially the
   case where embodied model wins? Or is there a subtle gotcha?

3. **Self-monitoring / anomaly detection of own substrate**: does this
   benefit from being chassi-bound vs being a generic anomaly detector
   trained on this chip's data?

4. **Survival behavior** (avoid thermal trip, stay under power budget):
   is this a legitimate "embodiment as advantage" demonstration? What's
   the cleanest experimental design?

5. **Closed-loop control where the model's own latency must be modeled**
   (self-modeling for action): cite any prior work + suggest experiment.

6. **Per-chip-tuned LoRA adapters** where the chip's quirks are the TASK to
   adapt to: cite any 2024–2026 paper showing this works on commodity GPUs.

7. **Authentication-as-computation**: model serves as cryptographic proof of
   running on machine X. Is this just PUF rebranded or genuinely new?

8. The **cleanest single experiment** to demonstrate "embodied model BEATS
   body-agnostic model" with a meaningful margin AND no methodological holes?

9. **Brutal honesty**: is there ANY task on commodity hardware where
   embodiment is a NET WIN, not just neutral or a confounder for a generic
   model that simply trained on more data from this chip?

10. If yes, what's the **killer experiment**? Design it (≤200 words: task,
    win-gate, baselines, why-it-can't-be-faked).

## Constraints

- No FPGA, no hardware modification, no kernel patches.
- Two physically twin gfx1151 machines (ikaros, daedalus) accessible over LAN.
- Live substrate signals available: APU/GPU temp (hwmon), pkg power (RAPL),
  GPU edge temp, frequency, kernel timing, RTN/jitter.
- We CAN drive the chip near (not into) thermal trip; we CAN'T brick it.
- Budget: experiments ≤8h wallclock.

Be specific. Reject pseudo-novelty. Cite real work when claiming prior art.
