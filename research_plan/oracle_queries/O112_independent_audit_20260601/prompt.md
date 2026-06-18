# O112 — Independent Audit of "Per-Die Identity" Findings

You are being asked for an **independent, fresh audit**. Do not assume any prior
oracle synthesis is correct. We deliberately withhold our own conclusions so
you can evaluate the raw data on its own merits. Please reason from first
principles.

## CONTEXT — what we did (just facts, no spin)

### Machines

- **ikaros** (laptop): AMD Strix Halo APU, Radeon 8060S (gfx1151), Linux
  6.14.0-1017-oem, microcode `0xb70001e`, family/model/stepping identical to
  daedalus per `/proc/cpuinfo`.
- **daedalus** (server): same AMD Strix Halo SKU, same microcode `0xb70001e`,
  same model/stepping, same kernel branch, similar memory + storage tier.

Both machines run our software stack from the same git tree.

### Two experiment phases

**Phase 12** — battery of low-level probes (`embodiment12_analysis.json`).
Each probe is a syscall / RDTSC / cache / DRAM / scheduler / power
micro-benchmark. We compare two intra-machine samples (ika vs ika, dae vs dae)
to estimate within-machine drift, and inter-machine (ika vs dae) to look for a
distinguishing signal. KS test reported as `intra_ikaros_KS_D`,
`intra_daedalus_KS_D`, `inter_KS_D`. The ratio `inter / max(intra)` is the
effect size. `pre_reg_pass = true` iff the inter D dominated both intra Ds by
the pre-registered margin.

**Phase 12B** — replication 24h later (`embodiment12b_analysis.json`). For
each task we check:
  - `persists`: does the inter-machine signal still distinguish them?
  - `same_chassi_stable`: do today's ika samples still match yesterday's?
  - `passed_tasks`: phase-12B-only pass list.

### Five signals we believe survived BOTH phases

(Effect sizes from raw JSON, no editorialising.)

| # | Probe                                | inter D | max intra D | ratio  |
|---|--------------------------------------|---------|-------------|--------|
| 1 | nanosleep(0) latency distribution    | 0.7224  | 0.0152      | ~47×   |
| 2 | sched_yield latency distribution     | 0.9931  | 0.0222      | ~45×   |
| 3 | inter-core cache-line ping-pong p50  | 0.9118  | (small)     | huge   |
| 4 | RDTSC offset between same-package cores | 0.91 D-stat | (small) | huge |
| 5 | DRAM refresh-window timing pattern   | strong (D≈0.9) | small | huge |

(See JSON files for exact figures. We are not asking you to recompute, we are
asking you to evaluate.)

### Microcode / stepping equality (claim we want challenged)

`cat /proc/cpuinfo` on both machines reports:
- `vendor_id: AuthenticAMD`
- identical `cpu family / model / stepping`
- identical `microcode: 0xb70001e`
- identical CPU flags set

So at the architectural-state level the two CPUs are nominally indistinguishable
from software's perspective. Yet the signals above separate them with D≈0.7–0.99
while intra-machine drift stays at D≈0.01–0.02.

### "Demo concept" we are considering

Working title: **"We Cloned an AI. It Died in a New Body."**

Concept: Train a small model (LoRA or weights-frozen embedding adapter) whose
forward pass reads a handful of these per-die signals as additional context.
Ship the weights to a second identical machine. The model's behaviour
collapses on the second machine because its substrate-derived inputs no longer
match. Frame as: the model is "bonded to its body" and cannot be moved.

That is the demo. We have not built it yet.

## RAW DATA — attached as JSON

- `embodiment12_analysis.json` — Phase 12 results
- `embodiment12b_analysis.json` — Phase 12B replication

Open them. Look at the actual numbers. Do not take our table on trust.

---

## QUESTIONS — please answer ALL twelve, numbered.

### A. Independent novelty audit

1. **Inter-core TSC offset as identity fingerprint.** We measure RDTSC delta
   between cores within the same CPU package; ikaros shows characteristic
   D=0.91 separation vs daedalus. Search literature: has anyone proposed
   inter-core TSC skew as a per-die fingerprint? Cite if so.

2. **Cache-line transfer latency between specific cores as identity.** Same
   question — prior art?

3. **DRAM refresh-interval timing pattern (without rowhammer) as per-DIMM /
   per-controller fingerprint.** Prior art?

4. **Both machines have identical microcode `0xb70001e`, identical stepping,
   identical model.** Yet the signals separate them with D≈0.7–0.99. Does
   this constitute strong evidence that the signal originates in per-die
   silicon variation (process variation, lithographic mismatch, defect
   pattern), as opposed to something else?

5. **Simplest alternative explanation.** What is the SIMPLEST hypothesis that
   explains the observed inter-machine separation **without** requiring
   per-die silicon variation? Examples to consider, but do not limit
   yourself: different DIMM batches; different SSD firmware doing background
   activity; different ambient temperature; different fan/PWM curve; different
   BIOS revision; different kernel scheduler config; different power profile;
   different background-process load. Rank the top 3 alternatives by
   plausibility.

6. **Best 3 publication venues** if these results hold up — be concrete
   (specific conference/journal, not "a security venue").

7. **Brutal honesty.** Rate the strength of this result on a 1–10 scale and
   defend the rating in 3 sentences. Be willing to say "weak."

### B. Independent unforgeability audit

8. **Spoofing by another Strix Halo.** Could an adversary owning a different
   AMD Strix Halo system clone the ikaros signature by collecting their own
   die's signals and replaying them? How hard, in concrete attacker hours?

9. **Trivial demo fakes.** Could the demo be faked just by reading the
   hostname, MAC address, or detecting a VM/hypervisor — i.e. is there any
   way the demo "works" without actually consuming the physical signal?
   What's the cheapest fake?

10. **Minimum extra work to make this cryptographically meaningful.** What's
    the smallest addition (PUF construction, fuzzy extractor, signed
    attestation chain, etc.) that would turn this from "interesting
    fingerprint" into "unforgeable hardware identity"? Be specific.

11. **Compare to existing hardware-bound systems** — Apple Secure Enclave,
    TPM 2.0 EK, AMD PSP fTPM. For each, name one concrete advantage and one
    concrete disadvantage of our approach versus that system.

### C. Independent demo evaluation

12. **Three sub-questions, brief answers:**
    - (a) Will the demo concept "We Cloned an AI. It Died in a New Body."
      go viral on social media? Rate 1–10 and justify.
    - (b) What demo using these signals would ACTUALLY capture broad
      attention? Free-form — propose something better if you can.
    - (c) What demo would be EMBARRASSING — gets dunked on Hacker News /
      X? Name a specific failure mode we should avoid.

---

## STYLE CONSTRAINTS

- Be skeptical. We want to be told if this is weak.
- Cite literature when you make a novelty claim.
- Numbers, not adjectives, when evaluating effect sizes.
- Brevity is fine. Bullet points are fine.
- Do **not** defer to "what the prior synthesis said" — there is none, from
  your perspective.
