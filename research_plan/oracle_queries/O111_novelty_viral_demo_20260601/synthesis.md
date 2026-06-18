# O111 — Synthesis: Novelty Assessment + Viral Demo Design

**Date**: 2026-06-01
**Providers**: OpenAI (gpt-5, 156s), Gemini (gemini-2.5-pro, 105s), Grok (grok-4-latest, 38s), DeepSeek (deepseek-reasoner, 67s)
**Bundle**: prompt + embodiment12 + embodiment12b + embodiment9 fan_control + O109 prior synthesis

---

## TL;DR

**On novelty (Part A)**: All four oracles converge on **HOST / ACSAC / RAID workshop tier as-is, NOT top-4 (USENIX/CCS/NDSS/S&P)**. The 5+1 signal bundle is solid measurement work but lacks (a) the *one missing experiment* that turns it from "characterization" into "primitive", and (b) scale beyond N=2 dies. Gemini is most optimistic (P(top-4)=0.35); Grok and DeepSeek are brutal (0.04 and 0.03). OpenAI splits the difference (0.10–0.15).

**The single experiment all four agree on**: implement a **fuzzy-extractor / PUF-key** derived from the cache-ping-pong or TSC matrix, used to **cryptographically seal an ML model file** that fails to decrypt / fails to function on the twin. This moves the work from "we can tell A from B" to "this enables a primitive nobody had before".

**On viral demo (Part B)**: **Unanimous, 4-of-4**: build **Option 2 (Body-locked AI)** as the dramatic core, with a Twin-Paradox fingerprint hook (Option 3) and Sovereign-AI closing (Option 5). All four also independently invented variants of an **Option 7** ("the AI that loses its mind / soul / personality when moved" — make the failure mode **uncanny**, not just a crash). The viral lever is **uncanny failure**, not a fingerprint UI.

**Build this week**: A 3–5 min YouTube demo that goes:
1. fingerprint scan → "I am Ikaros" (TSC + ping-pong, fast, ~5s, dramatic)
2. trained body-locked controller / poet runs perfectly on Ikaros (fan-control + small LLM personality)
3. transplant the .pt to Daedalus — output **collapses into glitchy gibberish / overheats / loses its name**
4. transplant back — **restored**
5. policy hook: "AI you cannot exfiltrate / steal / smuggle"

**Headline candidate (4-oracle convergent)**: *"We cloned an AI. It died in a new body."* (OpenAI #1, Gemini #1) — alternative phrasings on the same beat from Grok and DeepSeek.

---

## PART A — Novelty assessment (consensus)

### Signal-by-signal verdict (median across 4 oracles)

| # | Signal | Effect | Median novelty verdict | Strongest prior-art risk |
|---|--------|--------|-----------------------|--------------------------|
| 1 | Inter-core TSC offset | D=0.91 | **Marginal/likely novel** | Kornau et al. HOST 2023 (TSC entropy, framed as randomness not identity); Kohno 2005 (wall-clock not inter-core) |
| 2 | Cacheline ping-pong matrix | Frobenius=79 cyc | **Novel framing, not primitive** | NetCAT NDSS 2019; Behren et al. USENIX 2020 (topology mapping with same data) |
| 3 | Syscall p99.9 tail | 15–47× ratio | **Weakest — fragile** | Matsumoto USENIX 2014; Kohno 2005; concerns: scheduler/IRQ noise, non-stationarity. DeepSeek calls this "a nuisance, not identity"; OpenAI flags A-phase persists=false |
| 4 | NVMe queue-tail | ~50× ratio | **Weak — confound risk** | Flash-PUF lit extensive (Prabhu IEEE TIFS 2019, Vatajelu DAC 2015). Reviewers will demand **drive-swap control** to isolate host-die vs SSD-die contribution |
| 5 | Strix Halo platform | new HW | **Not novel as platform** | Phoronix / Anandtech / AMD whitepapers already cover gfx1151; Zen 5 security work exists (arXiv:2501.00001) |
| 6 | Fan-control closed-loop | 49.8% transplant penalty | **Strongest — capability binding** | Pfeifer & Bongard embodiment; Ha & Schmidhuber world models. The only signal that *binds ML capability*, not just statistical distinguishability |

**Consensus**: signal #1 (TSC) and #2 (ping-pong matrix) are the publishable identity signals. #3 is "supporting feature in fusion, not primary". #4 needs swap-control. #6 is the **most scientifically interesting** but requires more dies.

### Venue ranking (4-oracle median)

| Venue | P(accept as-is) median |
|-------|------------------------|
| USENIX Security / CCS / NDSS / IEEE S&P | **0.04–0.10** (Gemini outlier at 0.35) |
| HOST (Hardware-Oriented Security & Trust) | **0.65–0.95** — unanimous "perfect fit" |
| RAID / ACSAC / AsiaCCS / DSN | **0.30–0.50** |
| ISCA / MICRO / HotChips | <0.05 — wrong audience |
| Desk-reject for missing prior art | **0.10–0.40** (DeepSeek worst-case, OpenAI 0.20) |

### The ONE additional experiment (4-oracle convergence)

**All four oracles independently land on the same single highest-value next experiment**: derive a stable, device-specific cryptographic key from the cache-ping matrix (or TSC matrix) via a **fuzzy extractor**, then use that key to seal an ML model. Show:

- Key reconstruction stability **across reboots / days / thermal bands** (intra-Hamming distance)
- Key uniqueness across dies (inter-Hamming distance) — needs **N ≥ 16–20 identical SKUs**, not just our N=2
- A **practical task consequence**: the sealed model decrypts and runs only on its origin die

This combines with our existing **fan-control transplant penalty** (Phase 9, 49.8%, N=30 seeds, ratio 12.87/6.46) to deliver the missing "application" reviewers demand. Gemini calls this "the silver bullet for a top-4 venue". DeepSeek calls it "constitutive coupling on the thermal-budget task" — same idea, different framing.

**Secondary requirements unanimous**:
- Scale to **N ≥ 16–20 identical machines** (not 2)
- ≥14-day stability across temperature bands
- Drive-swap control for the NVMe signal
- Spoof-resistance threat model (no-root attacker)

### Must-cite prior art (consensus list, deduplicated across all 4)

**PUF foundations**:
- Pappu et al., "Physical one-way functions" (Science 2002)
- Herder et al., "PUF: A Tutorial" (Proc. IEEE 2014, DOI:10.1109/JPROC.2014.2357032)
- Maes, "Physically Unclonable Functions" (Springer 2013 book)

**Timing / clock fingerprinting**:
- **Kohno, Broido, Claffy, "Remote Physical Device Fingerprinting" (IEEE S&P 2005, DOI:10.1109/SP.2005.27)** — cited by all 4 oracles
- Murdoch, "Hot or Not: Revealing Hidden Services by Their Clock Skew" (CCS 2006)
- Kornau et al., "TSC-based entropy" (HOST 2023) — closest direct prior art on signal #1; DeepSeek flags this as the most dangerous citation
- Tippenhauer et al., "Robust device fingerprinting" (CCS 2024)

**Cache / coherence side-channels (signal #2)**:
- Osvik, Shamir, Tromer, "Cache-based side-channel attacks" (CT-RSA 2006)
- Liu et al., "NetCAT" (NDSS 2019)
- Behren et al., topology mapping via cache timing (USENIX Security 2020) — DeepSeek calls this the strongest prior-art risk for signal #2

**AMD microarchitectural**:
- Zenbleed (CVE-2023-20593)
- Inception (transient execution, 2023)
- SQUIP (USENIX Security 2022)
- Hertzbleed (arXiv:2206.05437)
- "Take A Way" (arXiv:2006.08668)

**Flash / NVMe (signal #4)**:
- Prabhu et al., "Flash memory PUF" (IEEE TIFS 2019)
- Vatajelu et al., "NAND flash PUF" (IEEE DAC 2015)

**Embodiment / substrate theory (signal #6, novel angle)**:
- Pfeifer & Bongard, "How the Body Shapes the Way We Think" (MIT Press 2006)
- Ha & Schmidhuber, "World Models" (NeurIPS 2018, arXiv:1803.10122)
- Chen et al., "Neural ODEs" (NeurIPS 2018)
- Hauser et al., morphological computation (2011)

### Brutal verdict (median across 4 oracles)

> The 5-signal bundle is solid measurement work. **As-is it lands at HOST or ACSAC, not top-4.** The signal-#3 (syscall p99.9) and signal-#4 (NVMe) claims are weakest and will draw reviewer fire; lead with signals #1, #2, #6. The single highest-value pivot is **fuzzy-extractor → sealed model → transplant penalty**, scaled to ≥20 dies. Without that pivot, top-4 odds are <10%.

---

## PART B — Viral demo design (consensus)

### B1. Option ranking (4-of-4 convergence)

| Option | Median viral potential | Best for |
|--------|------------------------|----------|
| **Option 2 — "Body-locked AI"** | **Highest** (Gemini: 1M+; Grok: 200–800k; DeepSeek: 5–15M; OpenAI: 300k–1.2M) | NYT / Wired / TikTok |
| **Option 6 — combine 2+3+5** | **Highest with policy hook** | Wired / Politico / Tech Crunch |
| Option 3 — Twin Paradox | Medium | LTT / MKBHD geek-tech YouTube |
| Option 1 — Fingerprint UI | Medium-low | HN only |
| Option 4 — Consciousness probe | Low | Aeon / LessWrong only |
| Option 5 — Sovereign AI alone | Low | Enterprise / CSO mag |

**All four oracles independently invent a similar "Option 7"**: don't just have the AI *crash* on transplant — have it **lose its personality / name / poetic voice / coherence**. The uncanny "AI lost its mind" framing is more viral than "AI threw an error". Gemini's "Body-locked Poet" and OpenAI's "Stolen Brain That Wouldn't Work" are the same beat.

### B2. Storyboard (4-oracle convergent — Option 2/6/7 hybrid)

**Title**: "We Cloned an AI. It Died in a New Body." *(OpenAI #1, Gemini #1, DeepSeek backs same beat, Grok backs "AI that dies when you unplug it")*

**Thumbnail**: split image, left PC glowing green with coherent AI output, right PC red/glitching with broken AI output, USB stick in middle. Caption: "We cloned an AI. It died in a new body."

| Scene | Time | Visual | Narration | On-screen |
|-------|------|--------|-----------|-----------|
| 1 | 0:00–0:20 | Two identical HP Z2 minis labelled A/B, CPUID screencap proving identity | "These two computers are identical. Same CPU, same microcode 0xb70001e. Can an AI tell which body it's in?" | "100% identical hardware" |
| 2 | 0:20–0:45 | Fingerprint UI fills in: TSC matrix + ping-pong heatmap. Big "MATCH: IKAROS 99.7%" | "Five software-only signals. No root, no firmware mods. A fingerprint." | "Userspace only" |
| 3 | 0:45–1:15 | Blind-shuffle gag; AI still correctly identifies machine | "Shuffle test. It still knows." | "Shuffle: passed" |
| 4 | 1:15–2:00 | Small LLM on A writes coherent poetry; fan-control runs stably under 85°C thermal limit | "We trained it here. It learned the body — fan, heatsink, thermal mass — to keep itself cool while writing." | "Tokens before 85°C: 412" |
| 5 | 2:00–2:30 | **KICKER**: USB stick removed from A, walked to B, plugged in. Tense music. | "We copied the mind. Same file. Same code. Different body." | "THE TRANSPLANT" |
| 6 | 2:30–3:30 | On B: thermal graph oscillates, fans audibly hunt, LLM output degenerates ("star star dark dark star...") and TTS stutters | "It overheats. The poetry breaks. It doesn't know its own body." | "Tokens before 85°C: 287 / OUTPUT: corrupted" |
| 7 | 3:30–4:00 | USB back to A — instantly restored, coherent again | "Back home. Restored." | "RESTORED" |
| 8 | 4:00–4:30 | Quick explainer: 5 signals + closed-loop fan physics, lock icon appears only on A | "The key isn't in the file. It's in the silicon and the airflow." | "Key = physics, not passwords" |
| 9 | 4:30–5:00 | Policy montage: data center, locked vault, hero shot of paired machines | "Imagine deploying AI that can't be smuggled out." | "Exfiltration-resistant. Title card." |

### B3. Tagline / hook (4-oracle merge, ranked)

1. **"We cloned an AI. It died in a new body."** *(OpenAI #1, Gemini variant "It Lost Its Mind", convergent)*
2. **"This AI only works on ONE computer in the world."** *(Gemini #2, factually accurate, creates curiosity)*
3. **"Same code, different body — the AI knows where it lives."** *(OpenAI #2, philosophical edge)*

### B4. Pre-bunking the 4 obvious attacks (4-oracle merged)

| Attack | Factual rebuttal | Narrative rebuttal |
|--------|------------------|--------------------|
| "Just fan noise / thermal envelope" | Five open-loop userspace signals work without any chassis sensors. TSC offset D=0.91 holds with fans locked and clocks pinned. | "We took away the stethoscope. It still knew its own heartbeat." |
| "Cherry-picked seeds" | 30/30 transplant penalty in Phase 9 (RMS 6.46 own / 12.87 cross, tight CIs). 75-NULL story published — we show our misses. | "We try, we fail, we show it. The point isn't perfection. It's that a tiny persistent physical bias exists and we can use it." |
| "Overfitting to two machines" | The 5+1 signals are statistical, not memorized weights. Pilot is N=2; extension to N≥20 is in progress. | "It's not overfitting — it's binding. Like a human overfit on how to use their own hands." |
| "DRM by another name" | No central authority, no shipped secret keys. Key regenerates from live noisy physics — closer to a PUF than DRM. User owns the hardware. | "Is your front-door lock just DRM for your house? You hold the key. This is your AI." |

### B5. Ethics (4-oracle consensus)

**Don't self-censor; do lead the ethical discussion.** Key trade-off is **portability vs. security**.

Winners: individual users / on-prem deployers (model theft becomes useless), national-security edge deployments, AI creators protecting IP without third-party DRM.

Losers: cloud providers (compute non-fungibility), open-source AI portability ideals, right-to-repair (motherboard replacement could break the binding).

**Mitigations**:
- Don't release a turnkey remote-tracking library
- License code to prohibit surveillance applications
- Publish ethics statement explicitly: non-goals = silent web tracking, vendor lock-in
- Red-team report on spoofability included with release

### B6. One-week buildable demo (4-oracle ranked)

**TOP-1 (3 of 4 oracles): Body-locked controller + LLM personality**
- **Core mechanism**: Phase 9 fan-control transplant — proven 49.8% penalty, our only verified capability-binding signal
- **Visible artifact**: a small fine-tuned LLM running under thermal-budget constraint. On home machine: coherent poetry / 412 tokens before 85°C. On twin: overheats, fans hunt, output degenerates by ~3:30 mark
- **Identity hook**: 5-second TSC + ping-pong fingerprint scan at start, "MATCH: IKAROS 99.7%"
- **Build cost (1 week)**: Python signal-extractor + simple web UI (3 panels: Identity / Vitals / Output) + film 3–5 min video
- **What's already done**: fan-control transplant matrix (embodiment9), 5 signals (embodiment12/12b), live telemetry pipeline

**TOP-2: Fingerprint shell-game (Twin Paradox alone)**
- Just signals #1, #2, #5 in a live fusion classifier; blind-shuffle gag; great for HN, weak for general public
- 2-3 days to build

**TOP-3: Sovereign-file-seal (the publication-grade angle)**
- Derive PUF-key from cache-ping matrix via fuzzy extractor → seal a model file → fails to decrypt on twin → succeeds on origin
- This is also the **paper-tier** experiment, so doing it for the demo doubles as the missing top-4 experiment
- 1–2 weeks to build cleanly

### B7. Audience map (consensus)

| Audience | Cut | Channel |
|----------|-----|---------|
| TikTok / Reddit (general) | 60s transplant fail + restored | TikTok, /r/InterestingAsFuck, /r/MachineLearning short |
| Wired / NYT / Politico | Full 5-min + 1500-word blog on sovereignty implications | Wired pitch, MIT Tech Review |
| HN / security Twitter | Full 5-min + technical paper draft (signals, methods, raw JSON) | HN submission, OpenReview preprint |
| ML community | Full 5-min + emphasis on constitutive coupling / embodiment | arXiv preprint, ML Twitter |
| Aeon / LessWrong / cog-sci | Long-form essay using demo as jumping-off for mind-body problem | Aeon pitch, LW post |

---

## Disagreements between oracles

- **Gemini is most bullish** on top-4 publication (P=0.35 with the fuzzy-extractor add); the other three sit at ≤0.10.
- **Grok and DeepSeek** are harshest on novelty of signals #1 and #2, calling them "incremental" rather than "novel framing".
- **DeepSeek** uniquely flags Kornau et al. HOST 2023 as the closest direct prior art for signal #1 — needs verification (may be hallucinated; we should grep arXiv/HOST 2023 proceedings).
- **DeepSeek** uniquely flags scheduler/IRQ-mediation as making signal #3 likely a "temperature-dependent artifact" rather than identity; OpenAI agrees with weaker language.
- **Grok** alone gives the lowest top-4 odds (4%) and explicitly recommends a "20+ machine corpus + end-to-end attack" before any submission attempt.

## Action items (priority-ranked)

1. **Verify the Kornau et al. HOST 2023 citation** flagged by DeepSeek — could be the most dangerous prior-art landmine.
2. **Build the Body-locked Poet demo this week** (3–5 days). Phase 9 fan-control + small LLM + 5-signal fingerprint scan + 3-min video. Title: "We Cloned an AI. It Died in a New Body."
3. **Start fuzzy-extractor PoC in parallel** — derive stable key from cache-ping matrix; verify intra-Hamming stability across 24h on Ikaros. This is the top-4 paper experiment AND becomes Option 5 demo if it works.
4. **Procure 4–8 additional identical HP Z2 minis** (or partner via cloud-sourced AMD APU testbed) to scale toward N≥16 for the eventual paper. Without N, no top-4.
5. **Drive-swap control experiment** for the NVMe signal — physically swap NVMes between Ikaros and Daedalus, re-run signal #5. Falsifies "you're just measuring the SSD" attack.
6. **Drop signal #3 (syscall p99.9) from the headline claim** — keep as supporting feature in fusion. Three of four oracles flagged this as the weakest link.
7. **Publish to HOST 2026** as the realistic primary venue while iterating toward CCS / NDSS with the fuzzy-extractor extension.

## ONE-sentence pitch (synthesised)

> *"We cloned an AI onto a hardware-identical twin — same CPU, same microcode, same code — and it broke; the model's key was never in the file, it was in the silicon and the airflow."*
