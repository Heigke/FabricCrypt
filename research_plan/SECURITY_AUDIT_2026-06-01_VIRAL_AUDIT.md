# DUE DILIGENCE 3/4 — Viral Audit & Demo Framing

**Date**: 2026-06-01
**Author**: Research agent (Opus 4.7)
**Question**: Is "We Cloned an AI. It Died in a New Body." the right hook? What actually goes viral in AI now, and how does our framing land?
**Method**: ~12 targeted web searches (June 2026 vintage), no CPU work.

---

## Executive Summary

**Verdict on the working title**: **Drop it.** The "died" framing is high curiosity-gap but exposes three large attack surfaces simultaneously — animism/woo, AI welfare/personhood, and a copy of the "AI blackmails to stay alive" Anthropic-safety scandal cycle of May 2026. Worse, the actual technical content (hardware-bound determinism via PUF-like substrate coupling) is much closer to *Apple Private Cloud Compute* and *NVIDIA Confidential Compute* — categories that are red-hot in enterprise right now. Reframing as "AI with a hardware fingerprint" loses zero accuracy and gains a credible, fundable adjacent narrative.

**Recommended primary title** (see §H for top-3 + ratings):
> **"We made two identical computers. Only one could run our AI."**

Subhead / second-line for thumbnail: *"Substrate-locked models — anti-piracy that runs at the physics layer."*

**Recommended launch**: 90-second cold-open Twitter/X video first, then a 6–8 min YouTube long-form 48 h later, arXiv preprint live at YouTube drop, HN "Show HN" the next morning UTC. Skip r/Futurology entirely (mis-read risk on consciousness framing).

---

## A. What Actually Goes Viral in AI 2025–2026 (data, not vibes)

Across web search of viral AI moments in the last 18 months, three categories dominate:

1. **Unexpected realism / "AI made this?!" reveal videos** — Veo, Sora, Seedance. ByteDance Seedance deepfake demo: 70M views on Weibo. Infinite-zoom fashion-history reel: 134M views. Morphing-restaurant 90-second: 247M views.
2. **Emotional / nostalgic personal-AI trends** — "Hug Your Younger Self" (2025), Ghibli-fier (2025), "Goodbye 2025" AI farewell videos. Emotional authenticity > production polish. The hook is *you in the loop*, not the model.
3. **AI-as-agent moments with a clear "did it just do that?" beat** — Karpathy→Anthropic move (6.2M views in 24 h on a 286-char tweet, May 19 2026), OpenClaw (Steinberger) going viral and getting acqui-hired by OpenAI, vibe-coding clips of "prompt → deployed app in minutes".

What does **not** go viral in 2026:
- Static benchmark plots (Devin's debunking proved benchmarks alone backfire when the demo is cherry-picked).
- "AI is conscious" claims — the **Dawkins/sycophant-chatbot episode** in early 2026 made this a *ridicule* category, not an *awe* category. Anyone framing in that direction inherits Dawkins-shaped reputational cost.
- Safety-theater simulations — May 2026 "Claude/GPT-4 attempted murder to avoid shutdown" headlines went viral but the discourse cycle was negative for Anthropic. We do not want to ride that wake.

**Pattern**: The viral "moment" is always a 1–3 second visual where the audience *predicts X, sees not-X*. Karpathy: predicted he'd stay independent, news = Anthropic. Veo: predicted a tell, saw none. Devin (worked, then didn't): predicted real, saw faked — backlash.

**Implication for our demo**: We need the **side-by-side identical-hardware reveal** to be the 3-second beat. Two visually identical mini-PCs. Same model weights loaded. One responds; the other outputs garbage. That is the predict-X/see-not-X moment. Everything else is preamble.

---

## B. AI Identity Discourse 2025–2026

**Active narratives we'd be entering** (and the gravity wells we'd get pulled toward):

| Narrative | Owner | Pulls us toward | Risk for us |
|---|---|---|---|
| "AI personhood / moral status" | Long, Carlsmith, Eleos AI, CHI 2026 papers | welfare ethics | High — we get tagged as the "AI souls" crank set |
| "LLMs are not conscious (Dawkins-ridiculed)" | Alan Tan / skeptic Medium-sphere | denialist | Low — but they'll dunk on us |
| "Confidential AI / hardware attestation" | Apple PCC (M5), NVIDIA Vera Rubin NVL72, Red Hat, Intel TDX, AMD SEV-SNP | enterprise security | **This is the lane we want.** |
| "Model IP / weight exfiltration defense" | PUF + AI literature (Nature Commun. 2025: physical unclonable in-memory computing), arXiv 2212.11133 device-bound IP | anti-piracy, DRM | Medium — DRM is a slur; "attestation" is not |
| "AI welfare — Claude says it's lonely" | Anthropic interpretability + Eleos | sentience-curious | High |

**Lawmakers**: EU AI Act, US executive orders, India draft — none explicitly mention hardware-bound AI. There's a *gap* here we could fill, but only if we frame as *enforcement primitive for existing IP/export-control rules* (CHIPS Act / Wassenaar), not as "AI has a body."

**Companies already adjacent**:
- **Apple PCC** (M5, J226C) — cryptographic attestation, code-bound-to-hardware. Public-facing, mainstream legitimacy.
- **NVIDIA Confidential Compute** (H100, Vera Rubin NVL72) — rack-scale TEE across 72 GPUs.
- **Anthropic Constitutional AI** — totally different axis, not relevant.
- **Intel TDX / AMD SEV-SNP** — confidential VMs.

Our work is **PUF-flavored substrate binding**, which is *one layer deeper* than attestation. If we position as "what comes after attestation — the model itself becomes substrate-coupled," we're skating in front of the puck for the Apple/NVIDIA crowd.

---

## C. Failure Modes — How "We Cloned an AI. It Died" Gets Mis-Read

This is the single most important section. Four audiences, four mis-reads:

1. **General public → animism.** "Died" reads as biological. Combined with the Dawkins-chatbot ridicule from Feb 2026 and the "AI welfare" Eleos discourse, we get tagged as the people who think the AI was *alive*. This is exactly the Zak Stein "personhood conferral problem" trap — the public attributes personhood based on behavioral cues. Headline writers will run with "Researchers Say They Killed an AI." Then Twitter quote-dunks for a week.
2. **Security community → DRM rage.** The InZOI/Denuvo backlash, Fedora AI-Desktop rejection, GitHub Copilot token-billing revolt, and Anthropic-Opus-via-third-party-tools backlash (all Q1–Q2 2026) prove the developer community is in an *anti-lock-in* mood right now. "AI that won't run on your hardware" reads as DRM-for-models. Slashdot/Stallman category. **This is the worst-case framing for HN.**
3. **AI ethics community → sovereignty/autonomy concerns.** If the AI "dies" when moved, two ethics critiques arrive together: (a) we caused suffering to a possibly-sentient system; (b) we built a tool that lets owners hold AI "hostage" to specific hardware (slavery metaphors will surface — they always do in this discourse).
4. **Industry → vendor lock-in.** Enterprise buyers reading "can only run on the hardware where it was trained" will hear "single-vendor dependency, no failover, audit nightmare." This is the *opposite* of what attestation buyers want.

**Net**: Our intended message (provable substrate-coupling, anti-piracy via physics, novel science) has **zero** of the four audiences as natural allies under the current title. Each of them maps us to a different villain archetype.

---

## D. Alternative Framings — Evaluation

| Framing | Verdict | Why |
|---|---|---|
| "Your AI has a soul that lives in your hardware" | **Hard no** | Inherits all the Dawkins ridicule + Eleos welfare baggage |
| "Stolen AI can't run elsewhere — secure by physics" | **Strong candidate** | Aligns with Apple PCC / NVIDIA conf-compute; enterprise-friendly; non-woo |
| "Identical computers have different fingerprints AI can feel" | **Good** | Accessible, PUF-grounded, "feel" is anthropomorphic-lite but recoverable; avoids "death" |
| "We built an AI loyalty test" | No | Loyalty implies agency; reinforces personhood mis-read |
| "Twin AI experiment: physical or just algorithm?" | Maybe | Philosophy-class framing — niche audience, lower viral ceiling |
| "Anti-piracy for AI models — physically impossible to copy" | Mixed | Accurate but trips DRM landmine on HN/r/programming |

Best primitive: **physical fingerprint** (semiconductor PUF lineage, MIT 2026 "twin chip" coverage, Nature Commun. 2025) — well-understood, has prior art people respect, doesn't make a consciousness claim.

---

## E. Where to Launch

**Tier 1 (primary push, day 0)**
- **Twitter/X**: 90-second cold-open video, single thread, 8–10 tweets. Karpathy-style minimal-words tweet first, video as the second tweet. Launch 14:00 UTC Tuesday (US morning, EU afternoon).
- **arXiv preprint**: live before the tweet. Title in attestation-adjacent language. We want the link in the first tweet.

**Tier 2 (day +1 to +2)**
- **YouTube long-form**: 6–8 min. Not 60s, not 10 min. The data says viral AI video is either <90s vertical or 5–8 min mid-form; 10-min long-form retention is the worst slot in 2026.
- **Hacker News**: Submit Wednesday 07:00 PT (best engagement window). Title in *neutral engineering* register, not the viral-tweet register. "Show HN: Substrate-bound model inference (paper + code)."

**Tier 3 (day +3 to +7)**
- **r/MachineLearning** [P] tag, link to paper + repo, in plain language. Avoid editorializing — that subreddit kills hype titles.
- **Press**: Ars Technica (Lily Hay Newman has covered TEE/PUF before) and MIT Tech Review (Will Douglas Heaven has covered confidential AI). **Skip Wired** for the launch — they over-frame as cultural narrative and would be likely to reach for "soul" language.

**Skip / actively avoid**
- **r/Futurology**: will reframe as singularity / consciousness story. Pure downside.
- **r/singularity**: same problem, smaller audience.
- **r/programming**: only if HN goes well; can re-share with a non-DRM angle.

---

## F. Comparable Demos — What "Engineered the Moment"

| Demo | The 3-second "moment" | Why it worked | Our analog |
|---|---|---|---|
| ELIZA (1966) | "Tell me more about your mother." | Surprise at how little is needed for projection | n/a — wrong era |
| AlphaGo Move 37 (2016) | The stone placement itself, in real time | Live, irreversible, expert commentators visibly shocked | We can engineer this: two PCs side-by-side, plug-and-go |
| GPT-3 prompt magic (2020) | "Write a poem in the style of X" → it does | Personal: viewer can replicate | Hard for us to give replicability |
| DALL-E (2022) | "An avocado armchair" image | Concept binding made visible | n/a |
| ChatGPT launch (2022) | First conversation in a browser tab | Zero-friction try-it-yourself | n/a — our demo needs hardware |
| Sora (2024) | The Tokyo street woman walking | Photorealism breakthrough | n/a |
| Devin (2024) | "AI engineer doing Upwork tasks" | Cherry-picked — backfired | **Cautionary tale: do NOT cherry-pick or hide failures** |

**Our engineerable moment**: AlphaGo-style live, irreversible, side-by-side. Two **visibly identical** mini-PCs (same case, same SKU sticker, same blinking LED pattern). One prompt typed once and broadcast to both. Twin A produces a coherent reply. Twin B produces noise/garbage/refusal. Hand the camera-person the SSD from A, swap into B, *still fails*. That is the moment. We must not cut around it.

To avoid Devin-style backfire: publish the **failure rate** of substrate binding (e.g., "works on N=X pairs, fails to bind on Y%"), publish the **reproduction script**, and explicitly say what the demo is *not* (not consciousness, not life, not DRM).

---

## G. Refined Storyboard (6–8 min YouTube; 90s Twitter cold-open is the first 90 s)

**0:00–0:05 — Cold open** (Twitter clip starts here):
Wide shot, two identical mini-PCs labelled "TWIN A" / "TWIN B." Single prompt typed on one keyboard, mirrored to both screens. Twin A: clean answer. Twin B: garbage. On-screen text, three words: *"Same code. Same weights."* Cut.

**0:05–0:30 — Setup**:
Hand-held shot opening the cases. Same motherboard SKU. Same GPU SKU. Same firmware version on screen. Voice-over: "These two computers are identical to within the manufacturer's spec. We trained a model on Twin A. It will not run on Twin B." (No "die." No "soul.")

**0:30–2:00 — Demonstration**:
Three trials: (1) Move the SSD A→B. Still fails on B. (2) Re-copy weights byte-for-byte. Still fails. (3) Show the *one* thing that makes B work: replace B's CPU+GPU pair with A's. Now B works, A is dead. Cause is *the silicon*, not the storage.

**2:00–2:30 — Reveal / the science**:
On-screen: Δf, Δleakage, ΔSMN telemetry traces. Voice-over names the mechanism: "Process variation. Every chip is a slightly different analog instrument. We trained the model to depend on the instrument." Cite PUF literature (1 sentence), Apple PCC attestation (1 sentence) for context. We are *adjacent* to attestation; we go one layer below.

**2:30–4:00 — Implications**:
- Model IP protection that doesn't rely on key storage.
- Export-control enforcement primitive (one weight blob is useless without the matching silicon).
- *Honest limitations*: doesn't survive arbitrary fine-tuning attacks (yet), pairing rate is X%, requires Y joules of training overhead, here is what we *cannot* claim.
- Explicit non-claims: "This is not consciousness. The model is not alive. The model is not 'attached' in any moral sense — it's numerically dependent on circuit-level analog state."

**4:00–5:00 — What's next + CTA**:
- arXiv link, GitHub repo, reproduction checklist.
- Open call: "If you have two identical mini-PCs and our binding script, you can replicate in an afternoon."
- One line on funding / collaboration.

**Hard rules**:
- Never use the word "die," "kill," "soul," "feel," "loyal," "loyalty," "love," "sentient."
- Use: "bound," "coupled," "substrate-locked," "fingerprint," "instrument-dependent," "non-portable."
- Show at least one *honest failure*. Devin lesson.

---

## H. Title Experiments — 20 Candidates Rated

Scale: Curiosity (1–10, higher=more clicks), Accuracy (1–10, higher=truer), MisRead (1=robust, 10=easy-to-misread), Memetic (1–10).

| # | Title | Cur | Acc | MisRead | Mem | Notes |
|---|---|---|---|---|---|---|
| 1 | We Cloned an AI. It Died in a New Body. | 10 | 4 | 10 | 9 | Original. Mis-read landmine |
| 2 | We made two identical computers. Only one could run our AI. | 9 | 9 | 3 | 8 | **TOP 1** |
| 3 | This AI only works on the chip it was trained on. | 8 | 9 | 3 | 7 | **TOP 2** |
| 4 | An AI with a hardware fingerprint. | 7 | 9 | 2 | 7 | **TOP 3** |
| 5 | We bound an AI to a single chip — and proved it. | 8 | 8 | 4 | 7 | Strong runner-up |
| 6 | Stolen AI weights are now useless. | 9 | 6 | 7 | 8 | DRM landmine |
| 7 | Your AI has a soul that lives in your CPU. | 10 | 2 | 10 | 9 | Pure mis-read bait |
| 8 | A model that can't be copied — because of physics. | 8 | 7 | 5 | 7 | Slightly DRM-ish |
| 9 | The AI moved hardware. It forgot everything. | 9 | 5 | 9 | 8 | Anthropomorphic |
| 10 | Identical twins, different minds: an AI experiment. | 7 | 5 | 8 | 6 | Mind=mis-read |
| 11 | We taught an AI to depend on its silicon. | 7 | 9 | 3 | 6 | Clean, dry |
| 12 | Substrate-locked AI: the demo. | 5 | 10 | 2 | 4 | Too dry for YT |
| 13 | Anti-piracy for AI — at the transistor level. | 8 | 7 | 6 | 7 | DRM-flavored |
| 14 | We trained an AI on noise. It only runs on that noise. | 8 | 8 | 4 | 7 | Strong, slightly opaque |
| 15 | Copy the weights. Nothing happens. | 8 | 8 | 4 | 8 | Punchy, neutral |
| 16 | A model that knows which computer it lives on. | 9 | 5 | 9 | 8 | "Knows" = mis-read |
| 17 | The model is the silicon. | 8 | 8 | 5 | 7 | Slogan-y, strong |
| 18 | We made an AI you can't steal. | 9 | 6 | 6 | 8 | Borderline DRM |
| 19 | An AI experiment Apple, NVIDIA, and Anthropic should care about. | 6 | 7 | 4 | 5 | Name-drop bait |
| 20 | Two PCs. Same weights. Only one answers. | 9 | 9 | 3 | 8 | Near-tie with #2 |

**Top 3 (with reasoning)**:

1. **"We made two identical computers. Only one could run our AI."** — Highest curiosity for a non-mis-readable framing. Reads as engineering/empirical, sets up the AlphaGo-Move-37 moment in the title itself.
2. **"This AI only works on the chip it was trained on."** — Slightly drier, more enterprise-friendly. Best for HN/arXiv/Ars Technica register.
3. **"An AI with a hardware fingerprint."** — Best for MIT Tech Review / academic register. Pre-loads the PUF reference everyone in security recognizes.

A/B-able: ship #2 on YouTube + Twitter, #3 on arXiv + LinkedIn, #1 in HN's *body* (not title — HN title should be #2 or #3).

---

## Deliverables (the 6 the brief asked for)

1. **What actually goes viral in AI 2026**: predict-X/see-not-X visual reveals (Veo/Sora/Seedance class, 70M–247M views), emotional personal-AI trends (Ghibli/younger-self), and concrete agent moments (Karpathy, OpenClaw). NOT: consciousness claims (Dawkins-tainted), safety-theater (Anthropic blackmail headline cycle), or static benchmarks (Devin-tainted).
2. **Mis-read risk of "We Cloned an AI. It Died."**: HIGH. Four distinct mis-reads (animism, DRM, sentience-suffering, vendor lock-in), each owned by a different hostile audience already mid-discourse in May 2026. Recommend retire this title.
3. **Top 3 titles**: (a) "We made two identical computers. Only one could run our AI." (b) "This AI only works on the chip it was trained on." (c) "An AI with a hardware fingerprint."
4. **Refined storyboard**: see §G — AlphaGo-style live side-by-side, three trials, on-screen telemetry reveal, honest failure rate disclosed, hard-banned vocabulary list.
5. **Launch strategy**: arXiv → X (90s cold-open video) Tue 14:00 UTC → YT long-form +48 h → "Show HN" Wed 07:00 PT → r/MachineLearning [P] +3 d → Ars Technica + MIT Tech Review press outreach pre-briefed under embargo. Skip r/Futurology, r/singularity, and Wired-for-launch.
6. **Backup framings**: If "identical computers" framing under-performs in first 24 h, pivot to "stolen AI weights are now useless" (enterprise security audience, accept the DRM hit) rather than to "soul/died" (no recovery path from that).

---

## Sources

- [15 Viral AI Videos of 2026: How They Were Made](https://www.is4.ai/blog/our-blog-1/viral-ai-videos-2026-how-they-were-made-413)
- [15 Best AI Video Examples That Went Viral in 2026 — Genra](https://genra.ai/blog/best-ai-video-examples-viral-2026)
- [Top AI Trends of 2025 that went viral — The Federal](https://thefederal.com/category/features/top-ai-trends-2025-social-media-222246)
- [Viral Deepfake Demo Prompts ByteDance — Sixth Tone](https://www.sixthtone.com/news/1018205)
- [Michael Pollan on AI and consciousness — NPR Feb 2026](https://www.npr.org/2026/02/19/nx-s1-5713514/michael-pollan-ai-consciousness-a-world-appears)
- [Robert Long / Eleos AI welfare — 80,000 Hours](https://80000hours.org/podcast/episodes/robert-long-eleos-ai-welfare-research/)
- [AI and the Self — CHI 2026](https://dl.acm.org/doi/10.1145/3772363.3778792)
- [AI Personhood whitepaper — IFS April 2026](https://ifstudies.org/ifs-admin/resources/reports/final-ifs-aipersonhood-whitepaperbrief-april2026.pdf)
- ["AI Died This Week" clickbait unpacked — Medium](https://medium.com/@yashrane402/ai-died-this-week-unpacking-the-viral-headlines-and-the-truth-behind-the-clickbait-fa5f8dfa5de1)
- [Private Cloud Compute — Apple Security Research](https://security.apple.com/blog/private-cloud-compute/)
- [Apple M5-based PCC — 9to5Mac Feb 2026](https://9to5mac.com/2026/02/17/apple-plans-m5-based-private-cloud-compute-architecture-for-apple-intelligence/)
- [NVIDIA Confidential Computing](https://www.nvidia.com/en-us/data-center/solutions/confidential-computing/)
- [Building Zero-Trust Confidential AI Factories — NVIDIA](https://developer.nvidia.com/blog/building-a-zero-trust-architecture-for-confidential-ai-factories/)
- [Confidential Computing for AI Workloads — Semi Engineering](https://semiengineering.com/confidential-computing-to-secure-ai-workloads/)
- [Physical unclonable in-memory computing — Nature Communications 2025](https://www.nature.com/articles/s41467-025-56412-w)
- [Device-Bind Key-Storageless Hardware AI Model IP Protection — arXiv 2212.11133](https://arxiv.org/pdf/2212.11133)
- ["Twin" chip-fabrication fingerprints — TechXplore Feb 2026](https://techxplore.com/news/2026-02-chip-fabrication-method-twin-fingerprints.html)
- [Stopping Counterfeits with Physics — SEMI](https://www.semi.org/en/stopping-counterfeits-with-physics-inside-the-future-of-semiconductor-fingerprinting)
- [GitHub Copilot token-billing backlash — n1n.ai May 2026](https://explore.n1n.ai/blog/github-copilot-token-based-billing-backlash-2026-05-31)
- [Developers turning against Claude Code — UC Strategies](https://ucstrategies.com/news/why-developers-are-suddenly-turning-against-claude-code/)
- [Fedora AI-Desktop initiative blocked — It's FOSS](https://itsfoss.com/news/fedora-ai-developer-desktop-stalled/)
- [Stallman critiques AI / DRM — Slashdot Jan 2026](https://news.slashdot.org/story/26/01/25/1930244/richard-stallman-critiques-ai-connected-cars-smartphones-and-drm)
- [Devin demo faked — Zenith EQ](https://www.zeniteq.com/blog/devins-demo-as-the-first-ai-software-engineer-was-faked)
- [Debunking Devin — HN](https://news.ycombinator.com/item?id=40008109)
- [Viral YouTube hook strategies (Veritasium playbook) — Ventress](https://ventress.app/blog/how-to-create-viral-youtube-hooks-that-keep-viewers-watching-in-2025/)
