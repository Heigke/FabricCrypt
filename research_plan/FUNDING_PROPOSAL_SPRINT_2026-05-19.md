# NS-RAM Funding Proposal Sprint — 2026-05-19

**Goal:** Fill 4 gaps (A/B/C/D from earlier review) + push v5.0 to Overleaf within 24h.

**Deadline:** Mario brief 2026-06-01 (13 days)

## Gaps to close

### Gap A — Device→algorithm mapping
For each NS-RAM cell property, derive: which algorithmic primitive it enables natively, why competitors can't, quantitative ratio.

### Gap B — Energy / area / speed baseline table
NS-RAM vs Loihi / TrueNorth / Mythic / IBM analog. Numbers per neuron, per synapse, per task.

### Gap C — Concrete task-level demo
1024 NS-RAM cells on KWS event-coded or SHD. Accuracy + energy + neurons/mm² + comparison to baselines.

### Gap D — Workplan WP1-WP4
Mario-fundable workplan: deliverables, milestones, dependencies, total ask.

## Dispatch architecture

5 parallel agents, results synthesized into v5.0 proposal:
1. **A-agent (ikaros)** — Device→algorithm mapping with cited physics
2. **B-agent (ikaros)** — Energy/area baseline table from literature + our cell
3. **C-agent (zgx GB10)** — NS-RAM surrogate × topology zoo → SHD/KWS benchmark
4. **D-agent (ikaros)** — WP1-WP4 written workplan for Marie-Curie/Vinnova/NRF format
5. **E-agent (ikaros)** — Oracle 3-way critique on full v5.0 draft before push

After all 5 land + oracle synthesis → merge into nsram_proposal_placeholders_overleaf_2026_05_03/main-4.tex → git commit → push.

## NO-CHEAT discipline
- All numbers cited to verifiable source (results files OR published paper with DOI)
- No invented benchmarks
- Honest "based on" / "extrapolated from" where modeling involved
- Oracle 3-way before any quantitative claim ships to Overleaf
