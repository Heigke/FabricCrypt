# NS-RAM Master Research Plan — Autonomous Mode

**Started:** 2026-05-01
**Owner:** Eric, autonomous Claude with Robert collab inbound
**Counterparts:** Sebastian Pazos (data, fits, transient), Mario Lanza (foundry, NRF), Robert Luciani (Julia/MTK simulator)
**Mode:** 24h autonomous run with periodic cron wake-ups; we keep coming back to this document.

---

## North-star goals (the bar we measure ourselves against)

**G1 — Fidelity.** Either our PyTorch BSIM4 port or Robert's Julia stack
(or both) reproduces every dynamic Sebas's silicon shows: DC I-V at every
(VG1, VG2), spike timing, self-reset, transient pulse response, body-charge
τ. Target acceptance: median log-RMSE ≤ 0.3 decade on DC, ≤ 10 % on
transient peak/τ.

**G2 — Speed × accuracy at network scale.** Once the cell is fidelity-locked,
the same calibrated cell runs in a topology simulator that hits ≥ 1 M cell
evals / s on a single workstation, allowing multi-topology × multi-task
sweeps in hours not weeks. Target: 9-cell, 16-cell, 64-cell, 256-cell
networks across 5 benchmarks (Hopfield retrieval, memory capacity,
NARMA-10, 7-class waveform, temporal-XOR) within 24h compute on Daedalus.

**G3 — Application identification.** A "where these chips sell" map: which
task at which scale beats a 1 W signal-processing accelerator on
energy/decision and where it loses. Output: an applications brief Mario
can use for NRF and TSMC.

We do not start G2 before G1 is met. We do not start G3 before G2 is met.
This is a sequential plan — phases, not threads.

---

## Phase A — Fidelity (G1)

### State as of 2026-04-30 23:45

- z91d (single-card constant-param fit, arclength solver): Stage 1 loss
  0.62 (off-state only, vth0 fled to 0.315), Stage 3 plateaued.
- z91f (forward-only, single card + Sebas CSV per-bias overrides): 5 decades
  off → after .param patch 2.40 median, 4.83 p90.
- z91g (true two-card refactor + 158+2 unit tests passing): identical to
  z91f run2 — refactor is structurally correct but doesn't move the number
  because Sebas's M1 and M2 cards differ only in `sd.scaled` parameters
  which were already correctly handled.
- Residual is concentrated in **low-VG2 regime** where M2 approaches cutoff
  (5 decades at VG1=0.6/VG2=0). Pattern points at NFACTOR not reaching the
  subthreshold-slope formula, BJT mismatch (Sebas's `mbjt`+`IS` vs our
  Gummel-Poon `Bf`), and possibly body-source diode at the floating body.
- Sebas confirmed (Apr 17 email): he dropped avalanche diodes (LTSpice
  convergence), uses "complementary bipolar current" + BSIM4 §6.1 impact
  ionization + body bias + LDE. Working on poly(VG1, VG2) param dependence.

### A.1 — Diagnose the low-VG2 residual

**A.1.a** — Verify NFACTOR override actually reaches the subthreshold-slope
formula. Trace `nfactor` from CSV → P_M2 dict → `sd.scaled["nfactor"]` →
into Vgsteff smoothing in `compute_dc`. If we patch `nfactor` and Id at
VG2=−0.2 doesn't change at all, the override is silently dropped.
*Output:* `research_plan/artifacts/A1a_nfactor_trace.md`

**A.1.b** — Map Sebas's `mbjt`+`IS` BJT parameterization to our
Gummel-Poon Bf. Read his `parasiticBJT.txt` and the LTSpice `.asc`
schematic in `nsram_info/schematic&modelCards/`. mbjt is the BJT
*idealization factor* (Nf in standard GP). Our `GummelPoonNPN` accepts
Nf — verify it's threading through. *Output:*
`research_plan/artifacts/A1b_bjt_mapping.md`

**A.1.c** — Body-source diode at floating body. With the floating P-body
between source/drain of M2, the body-source junction can forward-bias and
carry current. Our `compute_body_diodes` reads `jss=3.4089e-7` from both
cards. Is this firing in our forward sim at VG2 < 0?
*Output:* `research_plan/artifacts/A1c_body_diode_trace.md`

**Done criterion:** we can name, in one sentence, the dominant cause of
the low-VG2 residual.

### A.2 — z91h: residual GIDL + BJT fit, transport frozen to Sebas's CSV

After A.1, fit *only* the GIDL block (4 params: agidl, bgidl, cgidl, egidl)
and the BJT (Bf or Nf, depending on A.1.b). Everything else clamped to
Sebas's per-bias CSV.

**Done criterion:** median log-RMSE ≤ 1.0 decade on the 25 non-NaN curves.

### A.3 — Polynomial(VG1, VG2) parameter form

Even with A.2, residual gradient with VG2 will remain because Sebas's own
fit uses `K1`, `ETAB`, `BETA0` that vary per bias. Our forward sim with
his CSV applies the right value at each bias point, but we have no
*continuous* fit. Build z91i: poly(VG1, VG2) fit for the 4 transport
params Sebas extracted, anchored to his CSV values as soft constraints.

**Done criterion:** continuous fit reproduces Sebas's CSV values to
≤ 5 % relative error at every measured (VG1, VG2).

### A.4 — Transient validation when Sebas sends the raw traces

When Sebas's transient data lands (he flagged in 2026-04-30 email it's
coming): compare τ_body, peak Id, spike rate. *Blocking on Sebas.*

**Done criterion:** body-charge τ matches measurement ≤ 10 %, peak ≤ 20 %.

### A.5 — Cross-validate against ngspice

Run ngspice on the M1 and M2 cards (we have them, full SPICE-loadable)
on a 9-bias subset; compare bit-for-bit at intermediate variables (Vth,
Vdsat, Idsa). Catches any remaining BSIM4 sub-formula divergence.

**Done criterion:** ≤ 2 % per-curve relative error on 9 biases.

### A.6 — When Robert sends his Julia code

Cross-validate by running both simulators on the same calibrated cell
(post-A.3) on a held-out grid. Both must agree to ≤ 0.5 % relative on
Id at every bias.

**Done criterion:** Eric and Robert simulators agree across 50 bias points.

---

## Phase B — Speed × accuracy (G2)

### B.1 — Vectorize Newton over (VG1, VG2, Vd) batches

Today our Newton solves one bias at a time. Refactor `solve_2t_steady_state`
to operate on a `(B, ...)`-shaped tensor of biases — share Jacobian
structure across biases, use `torch.vmap` or manual batching. Aim for
≥ 100× speedup on 1000-bias batches.

### B.2 — torch.compile or triton-jit the inner kernels

`compute_dc`, `compute_iimpact`, `compute_igidl_gisl`, `compute_igb`,
`compute_body_diodes` are pure tensor ops. `torch.compile(mode="reduce-overhead")`
should give 2–4× on CPU and access to GPU. Triton kernels for the hottest
sub-blocks if needed.

### B.3 — GPU port (gfx1151 ROCm 7.0)

We already have ROCm + HSA override. Move the Newton inner loop and
kernels to GPU. Run on a 10⁴-bias batch.

**Done criterion (B.1–B.3):** ≥ 1 M cell-evaluations / s on a single
ikaros workstation. Compare to Robert's GB10 production: 71 k pts/s today,
3 M projected. We aim to land between.

### B.4 — Topology simulator

Wrap the calibrated cell in a network simulator that supports:
- arbitrary connectivity (sparse adjacency)
- per-cell VG1, VG2 setpoints
- inter-cell coupling via shared lines (Sebas's array hint)
- pulse-driven input, time-stepping

Build on top of `nsram` package. Use the same calibrated cell from A.3.
Topologies to support: linear chain, mesh, sparse random, small-world,
Hopfield-style fully-connected.

### B.5 — Five benchmarks at five scales

Per benchmark × per scale:
- Hopfield retrieval (8 → 256 cells)
- Memory capacity (linear)
- NARMA-10
- 7-class waveform classification
- Temporal-XOR (τ = 5, 10, 20)

For each (task, scale) pair: best topology, energy / decision,
accuracy curve, sensitivity to per-cell variability ±5%.

**Done criterion:** all 25 (task, scale) cells filled with reproducible
numbers in `research_plan/artifacts/B5_benchmark_table.csv`.

---

## Phase C — Applications (G3)

### C.1 — Where NS-RAM beats a 1 W signal-processing accelerator

For each benchmark, compute energy / inference and compare against a
representative 1 W edge accelerator (e.g. Innatera Pulsar, Coral Edge TPU,
GAP9). NS-RAM's per-spike energy of ~6.7 fJ × spike count vs accelerator's
~0.5 mJ / inference for similar tasks.

**Done criterion:** a quadrant chart with task families on one axis,
NS-RAM advantage on the other; clear "where to play" message.

### C.2 — Use-case brief for Mario / NRF

One-page document: target workloads, target power band, expected energy
advantage, scaling story to 1k-cell array. Suitable for NRF Mid-Sized
Grant material and TSMC test-vehicle pitch.

### C.3 — Architecture rec for next tape-out

Given the (task, scale) winners from B.5, recommend cell parameter ranges,
array geometry, fan-out, peripheral DAC resolution, etc. Hand to
Mario/Sebas as input for the next mask.

---

## Sequencing — what runs next

Strict sequence: A.1 → A.2 → A.3 (A.4 + A.5 + A.6 in parallel as data lands)
→ B.1 → B.2 → B.3 → B.4 → B.5 → C.1 → C.2 → C.3

Cron jobs: every ~2h, the autonomous loop re-reads
`research_plan/01_LOG.md` to find the next pending item, executes one
sub-step, logs result, picks the following step. If blocked (e.g. waiting
on Sebas's transient data), parallel-run a different phase's
prerequisite item.

---

## Oracle LLM use

We use GPT-5 and Gemini as second-opinion reviewers. The pattern:

1. Build a query packet under `research_plan/oracle_queries/<id>/`
   containing: `prompt.md`, relevant code files (zip), relevant data
   (csv), prior context (markdown).
2. Recommend cheaper model (gpt-5-nano / gemini-2.5-flash) for routine
   reviews, full model only for hard physics questions.
3. User pastes packet into the corresponding web UI and pastes back the
   response into `oracle_queries/<id>/response.md`.
4. We integrate the response into the next iteration.

We do NOT spend our API budget on chat-back-and-forth — packet-style asks
where the oracle has full context to *execute* on the question, not just
opine.

Initial oracle queries (built tonight):
- O1 — *low-VG2 residual diagnosis* — sent to GPT-5 with z91g code +
  Sebas CSV + plot
- O2 — *NFACTOR formula trace* — sent to Gemini with our `compute_dc.py`
  source + a single-bias dump

---

## Information sources (full list)

- `/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/data/sebas_2026_04_22/`
  — 33 measured I-V curves, two model cards, Sebas's per-bias CSV
- `/home/ikaros/nsram_info/Slow I-Vs 2vHCa-2@VG2 VG1 vnwell=2 SRavg=0/`
  — same dataset different layout (vnwell=2)
- `/home/ikaros/nsram_info/Zoom/` — 32 meeting screenshots, three meetings
  (2026-03-20, 2026-04-22, 2026-04-30)
- `/home/ikaros/nsram_info/emails.rtfd/TXT.rtf` — full email history
  Sebas/Mario/Robert/Eric — extract verbatim into
  `research_plan/artifacts/email_history.md`
- `/home/ikaros/nsram_info/2026-04-30 BSIMfitsBA/2026-04-29 NS-RAM I-V BA plots.pptx`
  — Sebas's slide deck of fit plots — extract images into
  `research_plan/artifacts/sebas_fit_plots/`
- `/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/docs/Team meeting 30 Apr 2026.pptx`
  — Robert's deck with the 50→30M pts/s throughput chart
- `/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/docs/NSRAM 20260429 Mario Seb.pptx`
  — combined deck Mario will use for NRF
- `/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/nsram/` — our
  PyTorch BSIM4 port (158 unit tests, two-model refactor done)
- `/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/z91*` —
  every fitting attempt since 2026-04-22

---

## Rules of engagement (autonomous mode)

1. **Always log to `01_LOG.md` after each iteration.** Date, what was done,
   what changed, what's next, what's blocked.
2. **Never modify `00_RESEARCH_PLAN.md` to lower the bar.** If a target is
   harder than expected, log the discovery, propose a revision, ask the
   user on next interactive turn. Do not silently weaken targets.
3. **Run scripts in background**, use `until grep -q DONE … done` to wait.
   Never block on a sleep > 5 min without a Monitor.
4. **Never claim "validated" without a number.** Every claim links to a
   specific artifact (plot, JSON, RMSE).
5. **Cross-check oracle answers** before acting on them. Oracles can
   confidently hallucinate BSIM4 formulas.
6. **Prefer reading code over guessing.** If unsure how `compute_dc`
   handles `nfactor`, read the function.
7. **Stop and ask** if a single iteration would commit to ≥ 4h of compute
   without a checkpoint, or would touch hardware.

End of plan. Iterate by reading `01_LOG.md`, picking next item, executing.

---

## Directive update (2026-05-02 11:30, after spike-firing breakthrough)

**No shortcuts.** Every fix must be physically motivated by Seb's
materials (cards, schematic, slides, emails, CSV) or by independent
BSIM4 reference. No threshold-tuning hacks. No mocked physics. If a
demo can't be done with current calibration, document the gap and
push for refit / new data, don't fake the demo.

**End goal redefined.** Physically accurate (not perfect) AND fast
enough for large-scale topology benchmarks. Doesn't have to fit Seb's
data to ≤ 0.3 dec — has to be *correct in spirit* and run network-
scale workloads to identify where each topology shines (energy /
decision, accuracy, scaling vs other substrates).

**GPU-first execution.** CPU iteration is too slow at our cell counts.
All B-phase scripts MUST default to GPU device when CUDA available.
Crossover at N≈100k confirmed earlier — every benchmark drives at
least 10k cells in parallel to be in the GPU-favourable regime.
torch.compile on GPU should give an additional 3-10× by fusing
the BSIM4 kernel ops into a single launch.

**Oracle review at every milestone.** When a substantial result
lands (fit improvement, new physics term, benchmark number), build
an O-packet with code + plot + numbers and ship to GPT-5 + Gemini
with file uploads. Use them as a second set of eyes on the physics
before declaring a step done.

**Cron**: continue at 30-min cadence (job c9a92559). Acts as a
"don't sleep" trigger if iteration stalls; user actively pushing
overrides the cooling rule.

**Re-scan loop:** before each major demo (NRF pitch, tape-out brief,
benchmark suite), re-scan `/home/ikaros/nsram_info/` and `docs/`
for any Seb/Mario material we missed. Models, slides, emails — all
fair game. Subagents tasked with this in parallel with execution.
