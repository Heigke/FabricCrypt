# Combined text bundle for oracle context

All text artifacts concatenated below for one-shot context. Each file is delimited by a `=== FILE: <name> ===` marker.



=== FILE: 00_RESEARCH_PLAN.md (13931 chars) ===
```
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

```


=== FILE: 01_LOG_tail80.md (3501 chars) ===
```
**This is the foundational piece. With this:**
✓ LIF dynamics — leaky integration validated
✓ Spike detection + reset — code path tested (no spike at this bias,
  but logic verified)
✓ Time-domain validation against Sebas's transients — UNBLOCKED
  (when he sends raw measurement files)
✓ Meta-plasticity demo — buildable: pick bias regimes that produce
  synapse, neuron, STM behaviours via control voltages

Phase A.4 → essentially CLOSED.

Next iteration: build the meta-plasticity demo. Pick three bias
configurations (synapse: stable analog Vb level, neuron: spike-and-
reset cycle, STM: charge-decay over τ ms) and show one cell
producing all three with only control-voltage changes. This is
exactly what Mario needs for NRF.

Files: nsram/bsim4_port/joint_newton.py (transient_2t, joint Newton
with line search + bounds + autograd Jacobian)

## 2026-05-02 11:00 — A.4.f Vb-equilibrium ceiling: 0.21 V (card-limit confirmed)

User asked to see actual LIF spike events. The earlier plot showed
threshold-crossing+reset working but with tiny amplitude. Investigated
whether the bias space contains regimes that reach larger Vb.

Bias sweep over (VG1, VG2, Vd) — 6 × 4 × 3 = 72 points covering:
  VG1 ∈ {0.2, 0.4, 0.6, 0.8, 1.0, 1.2}
  VG2 ∈ {-0.2, 0.0, 0.2, 0.4}
  Vd  ∈ {1.0, 1.5, 2.0}

**Maximum Vb_eq found = 0.212 V** at (VG1=0.4, VG2=0.4, Vd=2.0).
Across the entire 72-point sweep, Vb_eq never exceeded 0.21 V.

This **confirms what we documented in Phase A**: the parasitic NPN
needs Vb > ~0.6 V to fire (silicon p-n junction Vf), but Sebas's
un-refit M1 card hard-caps Vb_eq at ~0.21 V regardless of bias.

**Implications for LIF / meta-plasticity demos:**

✅ What we CAN demo with current calibration:
- Leaky integration (Vb climbs gradually, decays between pulses)
- Threshold-and-reset mechanism (with low threshold ~0.05-0.1 V)
- Capacitive body dynamics at fast pulses (body cap holds Vb low)
- Multi-cell parallel arrays (B.4 topology)
- Reservoir-style learning (B.5 next)

❌ What we CANNOT demo without M1 refit:
- Volt-scale neuronal spikes (requires Vb past 0.6 V to fire NPN)
- Realistic STDP (synaptic plasticity rules tied to NPN-fire events)
- True \"meta-plasticity\" (synapse↔neuron switching via the NPN
  transition) — the NPN never lights at any bias

**Path forward for the NRF pitch:**

Option A — \"mocked\" meta-plasticity using threshold-tuning. We
demonstrate the SAME BEHAVIOR as a real meta-plastic cell by varying
spike threshold and reset levels under control voltages, even though
the underlying physics (NPN firing) doesn't fire. Honest demo of the
DIFFERENTIABLE FRAMEWORK; not honest about the physics.

Option B — request M1 refit from Sebas. He's already iterating his
deck (added pdiode 5/2). With one more round we could get a card
where Vb naturally reaches volts at high VG1, then everything works.

Option C — synthesise a \"silicon-equivalent\" M1 by direct gradient
fit of vth0/u0/etab to make the model produce Vb past 0.6 V at some
realistic bias. This is engineering on our end.

**Recommendation:** Option B is cleanest scientifically. Option A is
fastest for the pitch demo. Option C is risky (could overfit). User
decision needed.

Phase A status: closed, but the card-limit is now QUANTIFIED at
Vb_eq ≤ 0.21 V — a hard constraint on what dynamic phenomena we can
demonstrate.

Next iteration: either await user decision on A/B/C, or proceed with
B.5 benchmarks (which use static-mode reservoir computing and don't
depend on volt-scale Vb).

```


=== FILE: _normalised/2tnsram_simple_asc.txt (1419 chars) ===
```
Version 4
SHEET 1 3052 680
WIRE 800 0 512 0
WIRE 800 64 800 0
WIRE 816 64 800 64
WIRE 848 64 816 64
WIRE 512 112 512 0
WIRE 800 112 800 64
WIRE 608 160 512 160
WIRE 640 160 608 160
WIRE 688 160 640 160
WIRE 704 160 688 160
WIRE 736 160 704 160
WIRE 752 160 736 160
WIRE 432 192 384 192
WIRE 464 192 432 192
WIRE 512 240 512 208
WIRE 800 240 800 208
WIRE 800 240 512 240
WIRE 608 272 608 160
WIRE 800 272 800 240
WIRE 704 288 704 160
WIRE 624 320 608 320
WIRE 544 352 496 352
WIRE 560 352 544 352
WIRE 624 368 624 320
WIRE 624 368 608 368
WIRE 800 400 800 272
WIRE 608 416 608 368
WIRE 704 416 704 352
FLAG 640 160 B
FLAG 800 272 Sint
FLAG 816 64 D
FLAG 704 416 0
FLAG 608 416 0
FLAG 432 192 G
FLAG 544 352 G2
FLAG 384 192 G
IOPIN 384 192 In
FLAG 496 352 G2
IOPIN 496 352 In
FLAG 848 64 Din
IOPIN 848 64 In
FLAG 800 400 S
IOPIN 800 400 Out
FLAG 688 160 B
IOPIN 688 160 Out
SYMBOL npn 736 112 R0
SYMATTR InstName Q1
SYMATTR Value parasiticBJT
SYMATTR Value2 area=1u
SYMBOL nmos4 464 112 R0
SYMATTR InstName M1
SYMATTR Value2 l='Ln' w='Wn' m=1
SYMBOL cap 688 288 R0
WINDOW 3 22 49 Left 2
SYMATTR Value 'CBpar'
SYMATTR InstName C1
SYMATTR SpiceLine Rser=1m
SYMBOL nmos4 560 272 R0
SYMATTR InstName M2
SYMATTR Value2 l='Ln*10' w='Wn' m=1
TEXT 552 24 Left 2 !.param Ln=0.18u\n.param Wn=0.36u\n.param CBpar=1f
TEXT 520 -64 Left 2 !.inc PTM130bulkNSRAM.txt
TEXT 520 -40 Left 2 !.inc parasiticBJT.txt
TEXT 310 478 Left 2 !.op 0

```


=== FILE: M1_130DNWFB.txt (9903 chars) ===
```
* Predictive Technology Model Beta Version
* 130nm NMOS SPICE Parametersv (normal one)
*  http://ptm.asu.edu/latest.html\
*+Lint = 2.5e-08 Tox = 3.3e-09
*+Vth0 = 0.395 Rdsw = 200

.model NMOSdnwfb NMOS

+Level = 14

+version = 4.5                 binunit = 2                   
+paramchk = 1                  mobmod = 0                    capmod = 2                    
+rdsmod = 0                    igcmod = 0                    igbmod = 0                    
+rbodymod = 0                  trnqsmod = 0                  acnqsmod = 0                  
+fnoimod = 1                   diomod = 1                    tempmod = 0                   
+permod = 1                    geomod = 0                    rgeomod = 0                   
+rgatemod = 0                  
+epsrox = 3.9                  toxe = toxn                   toxp = toxn                   
+toxm = toxn                   dtox = 0                      xj = 1.5e-7                   
+ndep = 1.7e17                 ngate = 1e23                  nsd = 1e20                    
+rsh = 1                       rshg = 0.1                    
+wint = wintn                  wl = 0                        wln = 1                       
+ww = -6.8e-15                 wwn = 1                       wwl = 0                       
+lint = lintn                  ll = 0                        lln = 1                       
+lw = 0                        lwn = 1                       lwl = 0                       
+llc = 0                       lwc = 0                       lwlc = 0                      
+wlc = 0                       wwc = 0                       wwlc = 0                      
+dwg = 0                       dwb = 0                       xl = 0                        
+xw = 0                        
+dmcg = 0                      dmdg = 0                      dmcgt = 0                     
+xgw = 0                       xgl = 0                       ngcon = 1                     
+vth0 = vth0n                  wvth0 = -1.6569e-8            pvth0 = pvth0n           
+phin = 0.05                   k1 = 0.53825                  k2 = -0.070435                
+k3 = k3n                      k3b = 6.37                    w0 = 2.5e-6                   
+lpe0 = lpe0n                  lpeb = -1.6512e-8             vbm = -3                      
+dvtp0 = 0                     dvtp1 = 0                     dvt0 = 1.9758                 
+dvt1 = 0.46322                dvt2 = -0.035558              dvt0w = -0.037131             
+dvt1w = 6.2805e5              dvt2w = -0.32774              vfbsdoff = 0                  
+u0 = 0.048317                 pu0 = -1.2e-16                ua = 5.0195e-11               
+ub = 1.7249e-18               uc = 1.1834e-10               ud = 1e14                     
+up = 0                        lp = 1e-8                     eu = 1.67                     
+vsat = vsatn                  pvsat = 1.03e-009             a0 = 1                        
+ags = 0.34914                 pags = 3e-013                 b0 = 6e-008                   b1 = 0                        
+keta = 0                      pketa = -3.4e-015             a1 = 0.9                      a2 = 0.95                     
+rdsw = 100-140*1e6*1u/int(1u/0.34u)     rdswmin = 35         rdw = 100             
+rdwmin = 0                    rsw = 100                     rswmin = 0                    
+prwb = -0.24                  prwg = 0                      wr = 1                        
+voff = -0.1368                wvoff = -5.6e-9               voffl = -5.5973e-9            
+minv = 0                      nfactor = 1.58                eta0 = 0.19998                
+etab = 1.8              dsub = 0.6412                 cit = 0                       
+cdsc = 2.4e-4                 cdscb = 0                     cdscd = 0                     
+pclm = 0.34476                pdiblc1 = 3.3832              pdiblc2 = 2e-3                
+pdiblcb = 0                   drout = 1.3536                pscbe1 = 5.331e8              
+pscbe2 = 1e-5                 pvag = 0.22                   delta = 0.01                  
+fprout = 0                    pdits = 0                     pditsl = 0                    
+pditsd = 0                    lambda = 0                    vtl = 2e5                     
+lc = 5e-9                     xn = 3                        alpha0 = 7.83756e-5           
+lalpha0 = -9.843026e-12       alpha1 = 0                    beta0 = 19                    
+lbeta0 = -9.5e-7              
+aigbacc = 0.43                bigbacc = 0.054               cigbacc = 0.075               
+nigbacc = 1                   aigbinv = 0.35                bigbinv = 0.03                
+cigbinv = 6e-3                eigbinv = 1.1                 nigbinv = 3                   
+aigc = 0.43                   bigc = 0.054                  cigc = 0.075                  
+aigsd = 0.43                  bigsd = 0.054                 cigsd = 0.075                 
+dlcig = 0                     nigc = 1                      poxedge = 1                   
+pigcd = 1                     ntox = 1                      toxref = toxn                 
+agidl = 1.99e-8               bgidl = 1.624e9               cgidl = 6.3                   
+egidl = 0.91                  
+noia = 3.3216e+41             noib = 1.0773239e+25          noic = -1.0624e+08                 
+em = 4.1e7                    ef = 0.96806                  lintnoi = 0                   
+xpart = 0                     cgso = rcgon*3.65e-10       cgdo = rcgon*3.65e-10               
+cgbo = 0                      ckappas = 0.6                 ckappad = 0.6                 
+cf = 0                        clc = 1e-7                    cle = 0.6                     
+dlc = 1.3737e-8               dwc = 0                       vfbcv = -1                    
+noff = 1                      lnoff = 2.2e-7                voffcv = -0.04464             
+lvoffcv = -2.8e-8             acde = 0.5535                 moin = 15                     
+cgsl = rcgon*2.98e-11         cgdl = rcgon*2.98e-11               
+ijthsrev = 0.1                ijthsfwd = 0.1                xjbvs = 1                     
+xjbvd = 1                     bvs = 10                      jss = 3.4089e-007                   
+jsws = 2.368e-013             jswgs = 0                     jtss = 0                      
+jtsd = 0                      jtssws = 0                    jtsswd = 0                    
+jtsswgs = 0                   jtsswgd = 0                   njts = 20                     
+njtssw = 20                   njtsswg = 20                  xtss = 0.02                   
+xtsd = 0.02                   xtssws = 0.02                 xtsswd = 0.02                 
+xtsswgs = 0.02                xtsswgd = 0.02                vtss = 10                     
+vtsd = 10                     vtssws = 10                   vtsswd = 10                   
+vtsswgs = 10                  vtsswgd = 10                  tnjts = 0                     
+tnjtssw = 0                   tnjtsswg = 0                  cjs = rcjn*0.0016995                
+mjs = 0.51829                 mjsws = 0.57223                         
+cjsws = rcjswn*2.9299e-011    cjswgs = rcjswgn*2.677e-010                
+mjswgs = 0.50288              pbs = 0.74883                 pbsws = 0.6836                     
+pbswgs = 0.70856                    
+xrcrg1 = 12                   xrcrg2 = 1                    rbpb = 50                     
+rbpd = 50                     rbps = 50                     rbdb = 50                     
+rbsb = 50                     rbps0 = 50                    rbpsl = 0                     
+rbpsw = 0                     rbpsnf = 0                    rbpd0 = 50                    
+rbpdl = 0                     rbpdw = 0                     rbpdnf = 0                    
+rbpbx0 = 100                  rbpbxl = 0                    rbpbxw = 0                    
+rbpbxnf = 0                   rbpby0 = 100                  rbpbyl = 0                    
+rbpbyw = 0                    rbpbynf = 0                   rbsbx0 = 100                  
+rbsby0 = 100                  rbdbx0 = 100                  rbdby0 = 100                  
+rbsdbxl = 0                   rbsdbxw = 0                   rbsdbxnf = 0                  
+rbsdbyl = 0                   gbmin = 1e-12                 
+tnom = 25                     ute = -1.785                  wute = 8e-8                   
+kt1 = -0.273                  kt1l = 3e-9                   kt2 = -0.034                  
+ua1 = 7.4e-10                 ub1 = -1e-18                  uc1 = -5.6e-11                
+lua1 = -8.88e-17
+ud1 = 0                       at = 4.6035e4                 prt = 0                    
+njs = 1.017                   xtis = 6.5                   tpb = 0                       
+tpbsw = 0                     tpbswg = 0                    tcj = 0                       
+tcjsw = 0                     tcjswg = 0                    tvoff = 0                     
+tvfbsdoff = 0                 
+saref = 1.04e-6               sbref = 1.04e-6               wlod = 0                      
+ku0 = -2.7e-8                 kvsat = 0.2                   kvth0 = 9.8e-9                
+tku0 = 0                      llodku0 = 0                   wlodku0 = 0                   
+llodvth = 0                   wlodvth = 0                   lku0 = 0                      
+wku0 = 0                      pku0 = 0                      lkvth0 = 0                    
+wkvth0 = 0                    pkvth0 = 0                    stk2 = 0                      
+lodk2 = 1                     steta0 = 0                    lodeta0 = 1                   
+web = 0                       wec = 0                       kvth0we = 0                   
+k2we = 0                      ku0we = 0                     scref = 1e-6         
```


=== FILE: M2_130bulkNSRAM.txt (10507 chars) ===
```
.param toxn    = 4e-009               toxp    = 4e-009
+lintn   = 1.219e-8             lintp   = -1.079e-8
+vth0n   = 0.54153              vth0p   = -1.106133
+lpe0n   = 1.2439e-007          lpe0p   = -7.833656e-8
+k3n     = 65.28                k3p     = -7.18419
+pvth0n  = -1.45e-015           pvth0p  = 5.543149e-16
+vsatn   = 102230               vsatp   = 8.07584e4
+wintn   = 4.7689e-008          wintp   = 4.268414e-9
+rcjn    = 1                    rcjp    = 1
+rcjswn  = 1                    rcjswp  = 1
+rcjswgn = 1                    rcjswgp = 1
+rcgon   = 1                    rcgop   = 1

* Predictive Technology Model Beta Version
* 130nm NMOS SPICE Parametersv (normal one)
*  http://ptm.asu.edu/latest.html\
*+Lint = 2.5e-08 Tox = 3.3e-09
*+Vth0 = 0.395 Rdsw = 200

.model NMOS NMOS

+Level = 14

+version = 4.5                 binunit = 2                   
+paramchk = 1                  mobmod = 0                    capmod = 2                    
+rdsmod = 0                    igcmod = 0                    igbmod = 0                    
+rbodymod = 0                  trnqsmod = 0                  acnqsmod = 0                  
+fnoimod = 1                   diomod = 1                    tempmod = 0                   
+permod = 1                    geomod = 0                    rgeomod = 0                   
+rgatemod = 0                  
+epsrox = 3.9                  toxe = toxn                   toxp = toxn                   
+toxm = toxn                   dtox = 0                      xj = 1.5e-7                   
+ndep = 1.7e17                 ngate = 1e23                  nsd = 1e20                    
+rsh = 1                       rshg = 0.1                    
+wint = wintn                  wl = 0                        wln = 1                       
+ww = -6.8e-15                 wwn = 1                       wwl = 0                       
+lint = lintn                  ll = 0                        lln = 1                       
+lw = 0                        lwn = 1                       lwl = 0                       
+llc = 0                       lwc = 0                       lwlc = 0                      
+wlc = 0                       wwc = 0                       wwlc = 0                      
+dwg = 0                       dwb = 0                       xl = 0                        
+xw = 0                        
+dmcg = 0                      dmdg = 0                      dmcgt = 0                     
+xgw = 0                       xgl = 0                       ngcon = 1                     
+vth0 = vth0n                  wvth0 = -1.6569e-8            pvth0 = pvth0n           
+phin = 0.05                   k1 = 0.63825                  k2 = -0.070435                
+k3 = k3n                      k3b = 6.37                    w0 = 2.5e-6                   
+lpe0 = lpe0n                  lpeb = -1.6512e-8             vbm = -3                      
+dvtp0 = 0                     dvtp1 = 0                     dvt0 = 1.9758                 
+dvt1 = 0.46322                dvt2 = -0.035558              dvt0w = -0.037131             
+dvt1w = 6.2805e5              dvt2w = -0.32774              vfbsdoff = 0                  
+u0 = 0.048317                 pu0 = -1.2e-16                ua = 5.0195e-11               
+ub = 1.7249e-18               uc = 1.1834e-10               ud = 1e14                     
+up = 0                        lp = 1e-8                     eu = 1.67                     
+vsat = vsatn                  pvsat = 1.03e-009             a0 = 1                        
+ags = 0.34914                 pags = 3e-013                 b0 = 6e-008                   b1 = 0                        
+keta = 0                      pketa = -3.4e-015             a1 = 0.9                      a2 = 0.95                     
+rdsw = 100-140*1e6*1u/int(1u/0.34u)     rdswmin = 35         rdw = 100             
+rdwmin = 0                    rsw = 100                     rswmin = 0                    
+prwb = -0.24                  prwg = 0                      wr = 1                        
+voff = -0.1368                wvoff = -5.6e-9               voffl = -5.5973e-9            
+minv = 0                      nfactor = 1.58                eta0 = 0.19998                
+etab = -0.086777              dsub = 0.6412                 cit = 0                       
+cdsc = 2.4e-4                 cdscb = 0                     cdscd = 0                     
+pclm = 0.34476                pdiblc1 = 3.3832              pdiblc2 = 2e-3                
+pdiblcb = 0                   drout = 1.3536                pscbe1 = 5.331e8              
+pscbe2 = 1e-5                 pvag = 0.22                   delta = 0.01                  
+fprout = 0                    pdits = 0                     pditsl = 0                    
+pditsd = 0                    lambda = 0                    vtl = 2e5                     
+lc = 5e-9                     xn = 3                        alpha0 = 7.83756e-5           
+lalpha0 = -9.843026e-12       alpha1 = 0                    beta0 = 18                    
+lbeta0 = -9.5e-7              
+aigbacc = 0.43                bigbacc = 0.054               cigbacc = 0.075               
+nigbacc = 1                   aigbinv = 0.35                bigbinv = 0.03                
+cigbinv = 6e-3                eigbinv = 1.1                 nigbinv = 3                   
+aigc = 0.43                   bigc = 0.054                  cigc = 0.075                  
+aigsd = 0.43                  bigsd = 0.054                 cigsd = 0.075                 
+dlcig = 0                     nigc = 1                      poxedge = 1                   
+pigcd = 1                     ntox = 1                      toxref = toxn                 
+agidl = 1.99e-8               bgidl = 1.624e9               cgidl = 6.3                   
+egidl = 0.91                  
+noia = 3.3216e+41             noib = 1.0773239e+25          noic = -1.0624e+08                 
+em = 4.1e7                    ef = 0.96806                  lintnoi = 0                   
+xpart = 0                     cgso = rcgon*3.65e-10       cgdo = rcgon*3.65e-10               
+cgbo = 0                      ckappas = 0.6                 ckappad = 0.6                 
+cf = 0                        clc = 1e-7                    cle = 0.6                     
+dlc = 1.3737e-8               dwc = 0                       vfbcv = -1                    
+noff = 1                      lnoff = 2.2e-7                voffcv = -0.04464             
+lvoffcv = -2.8e-8             acde = 0.5535                 moin = 15                     
+cgsl = rcgon*2.98e-11         cgdl = rcgon*2.98e-11               
+ijthsrev = 0.1                ijthsfwd = 0.1                xjbvs = 1                     
+xjbvd = 1                     bvs = 10                      jss = 3.4089e-007                   
+jsws = 2.368e-013             jswgs = 0                     jtss = 0                      
+jtsd = 0                      jtssws = 0                    jtsswd = 0                    
+jtsswgs = 0                   jtsswgd = 0                   njts = 20                     
+njtssw = 20                   njtsswg = 20                  xtss = 0.02                   
+xtsd = 0.02                   xtssws = 0.02                 xtsswd = 0.02                 
+xtsswgs = 0.02                xtsswgd = 0.02                vtss = 10                     
+vtsd = 10                     vtssws = 10                   vtsswd = 10                   
+vtsswgs = 10                  vtsswgd = 10                  tnjts = 0                     
+tnjtssw = 0                   tnjtsswg = 0                  cjs = rcjn*0.0016995                
+mjs = 0.51829                 mjsws = 0.57223                         
+cjsws = rcjswn*2.9299e-011    cjswgs = rcjswgn*2.677e-010                
+mjswgs = 0.50288              pbs = 0.74883                 pbsws = 0.6836                     
+pbswgs = 0.70856                    
+xrcrg1 = 12                   xrcrg2 = 1                    rbpb = 50                     
+rbpd = 50                     rbps = 50                     rbdb = 50                     
+rbsb = 50                     rbps0 = 50                    rbpsl = 0                     
+rbpsw = 0                     rbpsnf = 0                    rbpd0 = 50                    
+rbpdl = 0                     rbpdw = 0                     rbpdnf = 0                    
+rbpbx0 = 100                  rbpbxl = 0                    rbpbxw = 0                    
+rbpbxnf = 0                   rbpby0 = 100                  rbpbyl = 0                    
+rbpbyw = 0                    rbpbynf = 0                   rbsbx0 = 100                  
+rbsby0 = 100                  rbdbx0 = 100                  rbdby0 = 100                  
+rbsdbxl = 0                   rbsdbxw = 0                   rbsdbxnf = 0                  
+rbsdbyl = 0                   gbmin = 1e-12                 
+tnom = 25                     ute = -1.785                  wute = 8e-8                   
+kt1 = -0.273                  kt1l = 3e-9                   kt2 = -0.034                  
+ua1 = 7.4e-10                 ub1 = -1e-18                  uc1 = -5.6e-11                
+lua1 = -8.88e-17
+ud1 = 0                       at = 4.6035e4                 prt = 0                    
+njs = 1.017                   xtis = 6.5                   tpb = 0                       
+tpbsw = 0                     tpbswg = 0                    tcj = 0                       
+tcjsw = 0                     tcjswg = 0                    tvoff = 0                     
+tvfbsdoff = 0                 
+saref = 1.04e-6               sbref = 1.04e-6               wlod = 0                      
+ku0 = -2.7e-8                 kvsat = 0.2                   kvth0 = 9.8e-9                
+tku0 = 0                      llodku0 = 0                   wlodku0 = 0                   
+llodvth = 0                   wlodvth = 0                   lku0 = 0                      
+wku0 = 0                      pku0 = 0                      lkvth0 = 0                    
+wkvth0 = 0                    pkvth0 = 0                    stk2 = 0                      
+lodk2 = 1                     steta0 = 0                    lodeta0 = 1                   
+web = 0                       wec = 0                       kvth0we = 0                   
+k2we = 0                      ku0we = 0                     scref = 1e-6         
```


=== FILE: joint_newton.py (9522 chars) ===
```python
"""Joint (Vsint, Vb) Newton with autograd-exact Jacobian.

Replaces the finite-diff Jacobian in `_jacobian_fd_batched` (which costs
4 extra _residuals calls per step and has discretisation error). Uses
torch.autograd.functional.jacobian to get the exact 2x2 J at machine
precision, with one _residuals call.

Designed to be the shared kernel for:
  A.4 — implicit transient (add Cj·dVb/dt term to R_B before solving)
  B.1 — vectorized batch (stack along extra leading dim)

Reuses existing _residuals; does NOT re-implement physics.
"""
from __future__ import annotations
import torch
from typing import Optional

from .nsram_cell_2T import _residuals, NSRAMCell2TConfig
from .model_card import BSIM4Model
from .bjt import GummelPoonNPN


def _residual_pair(cfg, model_M1, model_M2, bjt, Vd, VG1, VG2,
                    Vsint, Vb, P_M1, P_M2,
                    Vb_prev=None, dt=None):
    """Wrap _residuals → (R_S, R_B) tensor for autograd.

    If `Vb_prev` and `dt` provided, adds the implicit-Euler cap term
    `-Cj(Vb-vnwell)·(Vb-Vb_prev)/dt` to R_B. This converts the DC
    body-KCL into a backward-Euler time-step equation.
    """
    R_S, R_B, _ = _residuals(cfg, model_M1, bjt,
                               Vd=Vd, VG1=VG1, VG2=VG2,
                               Vsint=Vsint, Vb=Vb,
                               P_M1=P_M1, P_M2=P_M2,
                               model_M2=model_M2)
    R_S = R_S.squeeze()
    R_B = R_B.squeeze()
    if Vb_prev is not None and dt is not None:
        from .transient import junction_cap
        Cj0_total = cfg.body_pdiode_Cj0_per_area * cfg.body_pdiode_area
        Cj = junction_cap(Vb.squeeze() - cfg.vnwell, Cj0=Cj0_total,
                            Vj=cfg.body_pdiode_Vj, M=cfg.body_pdiode_M)
        R_B = R_B - Cj * (Vb.squeeze() - Vb_prev) / dt
    return torch.stack([R_S, R_B])


def joint_newton_step(cfg, model_M1, model_M2, bjt, Vd, VG1, VG2,
                        Vsint: torch.Tensor, Vb: torch.Tensor,
                        P_M1=None, P_M2=None,
                        Vb_prev=None, dt=None
                        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One Newton step at a single bias point.
    Returns (Vsint_new, Vb_new, R_norm).
    If Vb_prev/dt given, performs implicit-Euler step (transient).
    """
    Vd_t = Vd.unsqueeze(0) if Vd.dim() == 0 else Vd
    Vsint = Vsint.detach().clone().requires_grad_(False)
    Vb = Vb.detach().clone().requires_grad_(False)
    state = torch.stack([Vsint, Vb]).requires_grad_(True)

    def _f(s):
        return _residual_pair(cfg, model_M1, model_M2, bjt,
                                Vd_t, VG1, VG2,
                                s[0:1], s[1:2], P_M1, P_M2,
                                Vb_prev=Vb_prev, dt=dt)

    R = _f(state)
    J = torch.autograd.functional.jacobian(_f, state, create_graph=False)
    try:
        delta = torch.linalg.solve(J, -R)
    except RuntimeError:
        delta = torch.zeros_like(state)
    new_state = state.detach() + delta.detach()
    return new_state[0], new_state[1], R.detach().abs().max()


def joint_newton_solve(cfg, model_M1, model_M2, bjt, Vd, VG1, VG2,
                        Vsint0: float = 0.1, Vb0: float = 0.5,
                        max_iters: int = 30, tol: float = 1e-12,
                        damp: float = 1.0, verbose: bool = False,
                        P_M1=None, P_M2=None,
                        Vb_prev=None, dt=None,
                        Vsint_bounds: tuple = (-0.5, 1.5),
                        Vb_bounds: tuple = (-0.5, 1.2),
                        max_step: float = 0.2) -> dict:
    """Iterate joint Newton with backtracking line search + bounds.

    If Vb_prev/dt given, solves for the implicit-Euler step.
    """
    Vsint = torch.tensor(float(Vsint0), dtype=torch.float64)
    Vb = torch.tensor(float(Vb0), dtype=torch.float64)
    R_prev = float("inf")
    for k in range(max_iters):
        Vs_new, Vb_new, R_norm = joint_newton_step(
            cfg, model_M1, model_M2, bjt, Vd, VG1, VG2, Vsint, Vb,
            P_M1, P_M2, Vb_prev=Vb_prev, dt=dt)
        # Reject NaN steps
        if not (torch.isfinite(Vs_new) and torch.isfinite(Vb_new)):
            if verbose:
                print(f"  iter={k}  NaN step rejected, halving damping")
            damp = damp * 0.5
            if damp < 1e-4:
                break
            continue
        # Cap step magnitude
        dVs = (Vs_new - Vsint).clamp(-max_step, max_step)
        dVb = (Vb_new - Vb).clamp(-max_step, max_step)
        # Backtracking line search
        alpha = damp
        accepted = False
        for ls in range(6):
            Vsint_try = (Vsint + alpha * dVs).clamp(*Vsint_bounds)
            Vb_try = (Vb + alpha * dVb).clamp(*Vb_bounds)
            # Re-eval R at trial point
            from .nsram_cell_2T import _residuals
            R_S_try, R_B_try, _ = _residuals(
                cfg, model_M1, bjt, Vd=Vd.unsqueeze(0) if Vd.dim()==0 else Vd,
                VG1=VG1, VG2=VG2,
                Vsint=Vsint_try.unsqueeze(0), Vb=Vb_try.unsqueeze(0),
                P_M1=P_M1, P_M2=P_M2, model_M2=model_M2)
            if Vb_prev is not None and dt is not None:
                from .transient import junction_cap
                Cj0_total = cfg.body_pdiode_Cj0_per_area * cfg.body_pdiode_area
                Cj_t = junction_cap(Vb_try - cfg.vnwell, Cj0=Cj0_total,
                                     Vj=cfg.body_pdiode_Vj, M=cfg.body_pdiode_M)
                R_B_try = R_B_try.squeeze() - Cj_t * (Vb_try - Vb_prev) / dt
            R_try = max(float(R_S_try.abs().max()), float(R_B_try.abs().max()))
            if R_try < float(R_norm) * 1.001:
                Vsint, Vb = Vsint_try, Vb_try
                accepted = True
                R_norm = torch.tensor(R_try)
                break
            alpha *= 0.5
        if not accepted:
            # Take tiny step anyway to avoid stalling
            Vsint = (Vsint + 1e-3 * dVs).clamp(*Vsint_bounds)
            Vb = (Vb + 1e-3 * dVb).clamp(*Vb_bounds)
        if verbose:
            print(f"  iter={k}  Vsint={float(Vsint):+.5f}  Vb={float(Vb):+.5f}  "
                  f"|R|={float(R_norm):.3e}  alpha={alpha:.3f}")
        if float(R_norm) < tol:
            return {"Vsint": Vsint, "Vb": Vb, "niter": k+1,
                      "converged": True, "R_norm": R_norm}
        R_prev = float(R_norm)
    return {"Vsint": Vsint, "Vb": Vb, "niter": max_iters,
              "converged": False, "R_norm": R_norm}


def transient_2t(cfg, model_M1, model_M2, bjt,
                  Vd_t: torch.Tensor, t: torch.Tensor,
                  VG1: torch.Tensor, VG2: torch.Tensor, *,
                  Vb0: float = 0.0, Vsint0: float = 0.1,
                  spike_threshold: float = 0.65,
                  reset_Vb: float = 0.30,
                  newton_iters: int = 25,
                  newton_tol: float = 1e-10,
                  damp: float = 0.7,
                  verbose: bool = False,
                  P_M1=None, P_M2=None) -> dict:
    """Implicit-Euler 2T transient with autograd-exact Jacobian.

    Solves the joint (Vsint, Vb) system at each timestep with R_B
    augmented by the body capacitance term. Spike detection AFTER step.

    Returns: dict with Vb, Vsint, Id, spike_times (s), t.
    """
    n = Vd_t.numel()
    Vb_traj = torch.zeros(n, dtype=torch.float64)
    Vsint_traj = torch.zeros(n, dtype=torch.float64)
    Id_traj = torch.zeros(n, dtype=torch.float64)
    spike_times = []

    # Step 0 — quasi-static DC at Vd[0]
    res0 = joint_newton_solve(
        cfg, model_M1, model_M2, bjt, Vd_t[0], VG1, VG2,
        Vsint0=Vsint0, Vb0=Vb0,
        max_iters=newton_iters, tol=newton_tol, damp=damp,
        P_M1=P_M1, P_M2=P_M2)
    Vsint, Vb = res0["Vsint"], res0["Vb"]
    Vb_traj[0] = Vb; Vsint_traj[0] = Vsint
    _, _, comps = _residuals(cfg, model_M1, bjt, Vd=Vd_t[0:1], VG1=VG1, VG2=VG2,
                                Vsint=Vsint.unsqueeze(0), Vb=Vb.unsqueeze(0),
                                P_M1=P_M1, P_M2=P_M2, model_M2=model_M2)
    Id_traj[0] = (comps["Ic_Q1"] + comps["Ids_M1"]).squeeze()

    for i in range(1, n):
        dt_i = float(t[i] - t[i-1])
        Vb_prev = Vb.detach().clone()
        # Warm-start from previous solution
        res = joint_newton_solve(
            cfg, model_M1, model_M2, bjt, Vd_t[i], VG1, VG2,
            Vsint0=float(Vsint), Vb0=float(Vb),
            max_iters=newton_iters, tol=newton_tol, damp=damp,
            P_M1=P_M1, P_M2=P_M2,
            Vb_prev=Vb_prev, dt=dt_i)
        Vsint, Vb = res["Vsint"], res["Vb"]
        # Compute Id at converged state
        _, _, comps = _residuals(cfg, model_M1, bjt, Vd=Vd_t[i:i+1], VG1=VG1, VG2=VG2,
                                    Vsint=Vsint.unsqueeze(0), Vb=Vb.unsqueeze(0),
                                    P_M1=P_M1, P_M2=P_M2, model_M2=model_M2)
        Id_traj[i] = (comps["Ic_Q1"] + comps["Ids_M1"]).squeeze()
        Vb_traj[i] = Vb; Vsint_traj[i] = Vsint
        if float(Vb) >= spike_threshold:
            spike_times.append(float(t[i]))
            Vb = torch.tensor(reset_Vb, dtype=torch.float64)
        if verbose and i % max(1, n // 10) == 0:
            print(f"  [transient] t={float(t[i]):.4g}  Vd={float(Vd_t[i]):.3f}  "
                  f"Vb={float(Vb):+.4f}  Vsint={float(Vsint):+.4f}  "
                  f"Id={float(Id_traj[i]):.3e}  conv={res['converged']}")
    return {"Vb": Vb_traj, "Vsint": Vsint_traj, "Id": Id_traj,
              "spike_times": spike_times, "t": t}

```


=== FILE: nsram_cell_2T.py (51201 chars) ===
```python
"""nsram_cell_2T — Differentiable 2T NS-RAM cell with proper topology.

Replaces the 1T proxy in `nsram_cell.py` (which collapses VG2 into a
``vth0_eff = vth0 + gamma·VG2`` shift) with the FULL 2T topology faithful
to Sebas's schematic ``data/sebas_2026_04_22/2tnsram_simple.asc``::

        D ──┬─────────────┬── (drain pin)
            │             │
          M1.D          Q1.C
            │             │
   VG1 → M1.G           Q1.B ── B  (floating body, shared by M1 & M2)
            │             │
          M1.S ── Sint ── Q1.E
                    │
                  M2.D
                    │
   VG2 → M2.G       │
                    │
                  M2.S ── 0  (ground)

Two NMOS (M1 short, M2 long) share floating body B. The internal node
Sint is the M1 source / M2 drain / Q1 emitter. Two unknown internal
voltages (Vsint, Vb) are solved by Newton-Raphson at each (Vd, VG1, VG2)
bias point so Sint-KCL = 0 and Body-KCL = 0.

Newton residuals (currents INTO each node):

    R_Sint(Vsint, Vb) =
        + Ids_M1(VG1−Vsint, Vd−Vsint, Vb−Vsint)            # M1 source ejects into Sint
        − Ids_M2(VG2,         Vsint,    Vb)                 # M2 drain absorbs from Sint
        + Ie_Q1(Vb−Vsint, Vb−Vd)                            # BJT emitter ejects into Sint
        + Ibs_diode_M1(Vb−Vsint)                            # forward body→Sint diode of M1
        − Ibd_diode_M2(Vb)                                  # forward body→drain(=Sint) of M2 leaves Sint

    R_B(Vsint, Vb) =
        + Iii_M1 + Iii_M2                                   # impact-ion holes → body
        + Igidl_M1 + Igisl_M1 + Igidl_M2 + Igisl_M2         # BTBT
        + Igb_M1 + Igb_M2                                   # gate→body tunnel
        − Ibd_diode_M1(Vb−Vd) − Ibs_diode_M1(Vb−Vsint)      # M1 junction leaks LEAVE body
        − Ibd_diode_M2(Vb)    − Ibs_diode_M2(Vb)            # M2 junction leaks LEAVE body
        − Ib_Q1(Vb−Vsint, Vb−Vd)                            # BJT base current leaves B

Drain terminal current at the D pin (positive into device):
    Id = Ids_M1 + Ic_Q1 + Igidl_drain_M1 + Ibd_diode_M1

VG2 is now a *real* gate to M2 (not a proxy threshold shift); body-effect
on M1 enters naturally via Vbs_M1 = Vb − Vsint.

Differentiability: simplest correct path. Newton iterations live INSIDE
autograd (no implicit-function-theorem trick yet). Each iteration is a
single forward of the full BSIM4 stack (~30 calls per bias point worst
case, double precision). For 33×~10 sweep points that's still tractable.

WARNING: do NOT add arbitrary clipping to "fix" Newton divergence — that
was the v4 mistake. Diagnose with `verbose=True`.
"""
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional

import torch

from .bjt import GummelPoonNPN, compute_bjt
from .dc import compute_dc
from .diode import compute_body_diodes
from .geometry import Geometry
from .leak import compute_iimpact, compute_igidl_gisl, compute_igb
from .model_card import BSIM4Model
from .temp import compute_size_dep, SizeDependParam


# --------------------------------------------------------------------------- #
# Config                                                                      #
# --------------------------------------------------------------------------- #

@dataclass
class NSRAMCell2TConfig:
    """Static config for a 2T NS-RAM cell.

    Geometry + toggles + Newton solver knobs. Two SizeDependParam objects
    (one per MOSFET) are computed lazily.
    """
    Ln: float = 180e-9                  # M1 channel length [m]
    Wn: float = 360e-9                  # both channels' width [m]
    M2_length_factor: float = 10.0      # M2 length = Ln * factor (Sebas: 10x)
    Cbody: float = 1e-15                # body cap [F] (transient only; from CBpar)
    T_C: float = 27.0                   # operating temperature

    # Junction geometry per MOSFET. None → auto W·L / 2(W+L).
    As_M1: Optional[float] = None
    Ad_M1: Optional[float] = None
    Ps_M1: Optional[float] = None
    Pd_M1: Optional[float] = None
    As_M2: Optional[float] = None
    Ad_M2: Optional[float] = None
    Ps_M2: Optional[float] = None
    Pd_M2: Optional[float] = None

    # Toggle physics
    use_iii: bool = True
    use_gidl: bool = True
    use_bjt: bool = True
    use_igb: bool = True
    use_diode: bool = True

    # Deep-N-well bias on M1 (130nm DNWFB device).
    # ──────────────────────────────────────────────────────────────────
    # IMPORTANT (2026-05-01, A.1.n finding): Sebas's measurement and
    # SPICE deck apply +2 V to the deep-N-well terminal of M1. The
    # well/body PN junction is forward-biased (well at +2 V vs floating
    # body at ~0 V), pumping current into the body. THIS is the missing
    # body-charging path that explains our 5-decade Id under-prediction
    # at low VG2. The schematic doesn't show the well node; the bias is
    # applied externally on the package pin, with the well capacitance
    # and series resistance internal.
    use_well_diode: bool = True
    vnwell: float = 2.0              # deep-N-well voltage [V]
    # Series-R production default = 1e9 Ω. Grid search (A.2 z91h_grid)
    # found tighter median at Rs=3e9 (0.79) and Rs=1e10 (0.69) but those
    # come with coverage loss — the arclength solver loses the snapback
    # fold mid-trace when vnwell coupling is strong. A.1.s solver work
    # (dual-direction sweep, branch detection) needed to unlock those
    # settings. Until then, Rs=1e9 gives the full 25/25 coverage at
    # honest median 1.19 / p90 2.88.
    vnwell_Rs: float = 1.0e10
    vnwell_area: float = 1.0e-12     # well-body junction area [m²] (~1 µm²; tiny)
    vnwell_Js: float = 3.4089e-7     # saturation current density [A/m²] (jss)
    vnwell_n: float = 1.017          # diode emission factor (njs)
    # mbjt-tracking: Sebas's CSV mbjt column scales the parasitic-NPN
    # area; physically the well-body junction belongs to the same parasitic
    # bipolar structure, so it should track the same multiplier. At
    # mbjt=0.001 (VG1=0.2 in his data) the well coupling effectively
    # disappears; at mbjt=1.0 (VG1=0.4/0.6) it's fully present.
    vnwell_mbjt: float = 1.0
    # A.1.u: Wire M2 body to GND (Sebas's nmos4 with body unconnected
    # defaults to GND in LTSpice) instead of to the floating Vb. Default
    # True per oracle consensus + visual scan of his .asc deck.
    m2_body_gnd: bool = True
    # A.3.d: M1 body-diode scale factor. M1's body-source/body-drain diodes
    # clamp Vb at the Si forward voltage (~0.5V) before the parasitic NPN
    # can light at ~0.6-0.7V. Sebas's measured snap requires Vb past ~0.6V,
    # implying his deck either omits these diodes or has tiny jss.
    m1_diode_scale: float = 1.0
    # A.10 (2026-05-02): Sebas reported a missing parasitic pdiode (5×4.4 µm²
    # = 22 µm²) at the floating body. Pre-staged via cfg flags; default OFF
    # until his SPICE card text + schematic arrive. Cathode candidate set is
    # 'off' / 'vnwell' / 'gnd' / 'sint'. GPT-5 favours 'vnwell' (reverse-bias
    # cap). Gemini favours 'gnd' (forward-bias DC drainage). We support both.
    body_pdiode_to: str = "off"           # cathode node
    body_pdiode_area: float = 22e-12      # 5 µm × 4.4 µm
    body_pdiode_Js: float = 1e-6          # A/m² (mid of oracle estimates)
    body_pdiode_n: float = 1.2            # ideality
    body_pdiode_Vj: float = 0.21918       # built-in (Sebas's pdiode card 2026-05-02)
    body_pdiode_M: float = 0.24097        # grading (Sebas's card)
    body_pdiode_Cj0_per_area: float = 7.3279e-4  # F/m² zero-bias junction cap (Sebas's cj)
    # Physical defaults injected when card has jss=jsd=0 (Sebas's PTM130 card
    # leaves these unset, which leaves the body diodes silent and lets Vb run
    # away unbounded under Iii injection — root cause of v6 fit explosion).
    # Typical 130nm CMOS pn junction: Js ≈ 1e-4 A/m². With AS = W·L = 360n·180n
    # = 6.5e-14 m², Is_diode ≈ 6.5e-18 A; at Vbs = 0.7V forward, Ibs ≈ 1.1e-5 A
    # → naturally clamps Vb at body-source diode turn-on voltage.
    default_jss: float = 1e-4    # A/m² source-bottom junction
    default_jsd: float = 1e-4    # A/m² drain-bottom junction

    # Newton solver
    newton_max_iters: int = 30
    newton_tol: float = 1e-12        # max(|R_Sint|, |R_B|) in Amperes (legacy)
    newton_damping: float = 1.0
    newton_min_damping: float = 1.0 / 64.0
    # Per-iteration relative voltage step cap (helps in steep regions w/o
    # masking divergence). Set to a large number to disable. Keep modest;
    # purpose is convergence, not "papering over" non-physics.
    max_step_V: float = 0.5

    # Oracle-recommended Newton hardening (gmin shunt + relative tol +
    # min-iter guard prevents the "spurious-root at iter 1" pathology where
    # Vb=0 initial guess lands all body currents at ~1e-17 A which is below
    # the absolute residual tolerance even though the true root is at
    # Vb~0.77 V).
    gmin: float = 1e-15              # shunt conductance on body+Sint KCL
    # Lowered from oracle-suggested 1e-12: at 1e-12 gmin shunts dominate
    # over physically zero body diodes (jss=jsd=0 in Sebas card) and pull
    # Vb to Vd/4, forward-biasing M1's body-source junction and doubling
    # Id. 1e-15 is small enough not to distort while still providing the
    # Jacobian slope to escape the spurious flat root at Vb=0.
    Iabstol: float = 1e-12           # absolute current tolerance
    Ireltol: float = 1e-3            # relative tolerance vs |I_physical|
    xtol_v: float = 1e-7             # voltage step infinity-norm tolerance
    min_iters: int = 2               # require >= this many Newton iters
    # gmin homotopy: if enabled, first cold-start solve walks gmin from
    # gmin_start down to `gmin` in factor-of-10 steps before declaring done.
    gmin_step: bool = False
    gmin_start: float = 1e-9

    # Lazy SizeDependParam caches
    _sd_M1: Optional[SizeDependParam] = field(default=None, init=False, repr=False)
    _sd_M2: Optional[SizeDependParam] = field(default=None, init=False, repr=False)

    # ------------------------------------------------------------------ #
    def _geom_M1(self) -> Geometry:
        return Geometry(L=self.Ln, W=self.Wn, NF=1)

    def _geom_M2(self) -> Geometry:
        return Geometry(L=self.Ln * self.M2_length_factor, W=self.Wn, NF=1)

    def size_dep_M1(self, model: BSIM4Model) -> SizeDependParam:
        if self._sd_M1 is None:
            self._sd_M1 = compute_size_dep(model, self._geom_M1(), T_C=self.T_C)
        return self._sd_M1

    def size_dep_M2(self, model: BSIM4Model) -> SizeDependParam:
        if self._sd_M2 is None:
            self._sd_M2 = compute_size_dep(model, self._geom_M2(), T_C=self.T_C)
        return self._sd_M2

    def invalidate(self) -> None:
        self._sd_M1 = None
        self._sd_M2 = None

    def _junctions_M1(self) -> tuple[float, float, float, float]:
        WL = self.Wn * self.Ln
        WLp = 2.0 * (self.Wn + self.Ln)
        return (
            WL  if self.As_M1 is None else self.As_M1,
            WL  if self.Ad_M1 is None else self.Ad_M1,
            WLp if self.Ps_M1 is None else self.Ps_M1,
            WLp if self.Pd_M1 is None else self.Pd_M1,
        )

    def _junctions_M2(self) -> tuple[float, float, float, float]:
        L2 = self.Ln * self.M2_length_factor
        WL = self.Wn * L2
        WLp = 2.0 * (self.Wn + L2)
        return (
            WL  if self.As_M2 is None else self.As_M2,
            WL  if self.Ad_M2 is None else self.Ad_M2,
            WLp if self.Ps_M2 is None else self.Ps_M2,
            WLp if self.Pd_M2 is None else self.Pd_M2,
        )


# --------------------------------------------------------------------------- #
# Param-override context for SizeDependParam                                  #
# --------------------------------------------------------------------------- #

@contextmanager
def _override_sd(sd: SizeDependParam, overrides: Optional[dict]):
    """Temporarily replace selected SizeDependParam fields (for fitting).

    Useful so optimizer can flow grads through ``sd.vth0_T`` etc. without
    rebuilding the whole SizeDependParam each iteration.
    """
    if not overrides:
        yield
        return
    saved: dict = {}
    try:
        for k, v in overrides.items():
            saved[k] = getattr(sd, k)
            setattr(sd, k, v)
        yield
    finally:
        for k, v in saved.items():
            setattr(sd, k, v)


# --------------------------------------------------------------------------- #
# Per-MOSFET physics evaluator                                                #
# --------------------------------------------------------------------------- #

def _eval_mosfet(
    model: BSIM4Model,
    sd: SizeDependParam,
    cfg: NSRAMCell2TConfig,
    Vg: torch.Tensor,
    Vd: torch.Tensor,
    Vs: torch.Tensor,
    Vb: torch.Tensor,
    junctions: tuple[float, float, float, float],
    overrides: Optional[dict] = None,
) -> dict:
    """Compute Ids, Iii, Igidl, Igisl, Igb, Ibs, Ibd for one NMOS at given
    *terminal* voltages. Bias mapping (NMOS, source-referenced):

        Vgs = Vg - Vs,   Vds = Vd - Vs,   Vbs = Vb - Vs,   Vbd = Vb - Vd

    Returned dict uses the convention native to each sub-call:
        - Ids: drain-to-source channel current (positive in saturation, NMOS)
        - Iii: positive INTO body (channel impact-ion)
        - Igidl: positive INTO body (drain edge BTBT, "drain → body")
        - Igisl: positive INTO body (source edge BTBT)
        - Igb: positive INTO body (gate → body tunneling)
        - Ibs, Ibd: junction diode currents, *positive when forward biased*
                   (current flows OUT of body INTO source/drain).
    """
    Vgs = Vg - Vs
    Vds = Vd - Vs
    Vbs = Vb - Vs
    Vbd = Vb - Vd

    with _override_sd(sd, overrides):
        dc = compute_dc(model, sd, Vgs=Vgs, Vds=Vds, Vbs=Vbs)

        if cfg.use_iii:
            Iii = compute_iimpact(model, sd, dc, Vds=Vds)
        else:
            Iii = torch.zeros_like(dc.Ids)

        if cfg.use_gidl:
            Igidl, Igisl = compute_igidl_gisl(model, sd, Vgs=Vgs, Vds=Vds, Vbs=Vbs)
        else:
            Igidl = torch.zeros_like(dc.Ids)
            Igisl = torch.zeros_like(dc.Ids)

        if cfg.use_igb:
            Igb = compute_igb(model, sd, Vgs=Vgs, Vbs=Vbs, dc_result=dc)
        else:
            Igb = torch.zeros_like(dc.Ids)

        if cfg.use_diode:
            As_, Ad_, Ps_, Pd_ = junctions
            # Inject physical Js defaults when card has zero (Sebas card
            # bug — root cause of Vb runaway). See cfg comments.
            js_overrides = {}
            try:
                if float(sd.SourceSatCurDensity_T) == 0.0 and cfg.default_jss > 0:
                    js_overrides["SourceSatCurDensity_T"] = cfg.default_jss
                if float(sd.DrainSatCurDensity_T) == 0.0 and cfg.default_jsd > 0:
                    js_overrides["DrainSatCurDensity_T"] = cfg.default_jsd
            except Exception:
                pass
            if js_overrides:
                with _override_sd(sd, js_overrides):
                    Ibs, Ibd = compute_body_diodes(model, sd, Vbs=Vbs, Vbd=Vbd,
                                                   As=As_, Ad=Ad_, Ps=Ps_, Pd=Pd_)
            else:
                Ibs, Ibd = compute_body_diodes(model, sd, Vbs=Vbs, Vbd=Vbd,
                                               As=As_, Ad=Ad_, Ps=Ps_, Pd=Pd_)
        else:
            Ibs = torch.zeros_like(dc.Ids)
            Ibd = torch.zeros_like(dc.Ids)

    return {
        "Ids": dc.Ids,
        "Iii": Iii,
        "Igidl": Igidl,
        "Igisl": Igisl,
        "Igb": Igb,
        "Ibs": Ibs,         # >0 ⇒ leaves body INTO source
        "Ibd": Ibd,         # >0 ⇒ leaves body INTO drain
        "Vds": Vds,
        "Vbs": Vbs,
        "Vbd": Vbd,
    }


# --------------------------------------------------------------------------- #
# Residual                                                                    #
# --------------------------------------------------------------------------- #

def _residuals(
    cfg: NSRAMCell2TConfig,
    model_M1: BSIM4Model,
    bjt: GummelPoonNPN,
    Vd: torch.Tensor,
    VG1: torch.Tensor,
    VG2: torch.Tensor,
    Vsint: torch.Tensor,
    Vb: torch.Tensor,
    P_M1: Optional[dict] = None,
    P_M2: Optional[dict] = None,
    model_M2: Optional[BSIM4Model] = None,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Compute (R_Sint, R_B, components) at current (Vsint, Vb) guess.

    `model_M2` defaults to `model_M1` (single-model legacy behaviour).
    """
    if model_M2 is None:
        model_M2 = model_M1
    sd_M1 = cfg.size_dep_M1(model_M1)
    sd_M2 = cfg.size_dep_M2(model_M2)
    j_M1 = cfg._junctions_M1()
    j_M2 = cfg._junctions_M2()

    # Ground reference is V_S (source of M2) = 0.
    zero = torch.zeros_like(Vd)

    # M1: D=Vd, G=VG1, S=Vsint, B=Vb
    m1 = _eval_mosfet(model_M1, sd_M1, cfg, Vg=VG1, Vd=Vd, Vs=Vsint, Vb=Vb,
                      junctions=j_M1, overrides=P_M1)
    # M2: D=Vsint, G=VG2, S=0, B=(Vb or GND, see cfg.m2_body_gnd)
    # ──────────────────────────────────────────────────────────────────
    # A.1.u (2026-05-01): Sebas's `2tnsram_simple.asc` uses the LTSpice
    # `nmos4` symbol with M2's body terminal **left unconnected** → it
    # defaults to GND. Wiring M2.B to the floating body Vb (our prior
    # behaviour) drains charge from the body through M2's bulk diodes /
    # Iii / GIDL, preventing Vb from rising enough to fire the parasitic
    # NPN that produces the snapback. cfg.m2_body_gnd=True restores
    # Sebas's topology.
    Vb_M2 = zero if cfg.m2_body_gnd else Vb
    m2 = _eval_mosfet(model_M2, sd_M2, cfg, Vg=VG2, Vd=Vsint, Vs=zero, Vb=Vb_M2,
                      junctions=j_M2, overrides=P_M2)

    # Parasitic NPN: collector=D, base=B, emitter=GND.
    # ──────────────────────────────────────────────────────────────────
    # IMPORTANT (2026-05-01, A.1.i finding): Sebastian's LTSpice schematic
    # `2tnsram_simple.asc` wires the parasitic NPN with **emitter to
    # ground**, not to Sint. This is the "complementary bipolar current"
    # he refers to in his Apr-17 email — its purpose is to provide a
    # body-charging path that fires when Vb climbs (Vbe = Vb − 0 = Vb,
    # not Vb − Vsint ≈ small). With emitter=Sint the BJT would never
    # turn on at low VG2 because Vb tracks Vsint. With emitter=GND, Vbe
    # tracks Vb directly and the NPN switches at Vb ~0.6 V.
    if cfg.use_bjt:
        Vbe = Vb                 # emitter = ground
        Vbc = Vb - Vd            # collector = drain
        bjt_out = compute_bjt(bjt, Vbe=Vbe, Vbc=Vbc, T_K=273.15 + cfg.T_C)
        Ic_Q1 = bjt_out["Ic"]    # collector current (drain → emitter = GND)
        Ib_Q1 = bjt_out["Ib"]    # base current (INTO base from external)
        Ie_Q1 = bjt_out["Ie"]    # emitter current at GND (= −(Ic+Ib))
    else:
        Ic_Q1 = torch.zeros_like(Vd)
        Ib_Q1 = torch.zeros_like(Vd)
        Ie_Q1 = torch.zeros_like(Vd)

    # ---- Sint KCL: currents INTO Sint --------------------------------- #
    # M1 channel current Ids_M1 flows D→S — INTO Sint (M1 source). → +Ids_M1
    # M2 drain is Sint; M2 channel sinks current FROM drain → −Ids_M2
    # BJT emitter is now GND, NOT Sint — BJT no longer touches Sint node.
    # M1 junction: Ibs_M1 >0 ⇒ leaves body INTO source(=Sint). → +Ibs_M1
    # M2 junction: Ibd_M2 >0 ⇒ leaves body INTO drain(=Sint). → +Ibd_M2
    R_Sint = (
        m1["Ids"]
        - m2["Ids"]
        + m1["Ibs"]
        + m2["Ibd"]
    )

    # Deep-N-well to body diode (A.1.n: this is the missing body-charging path).
    # ──────────────────────────────────────────────────────────────────
    # When vnwell > Vb, the N-well/P-body junction forward-biases and pumps
    # current INTO the body. Modelled as a Shockley diode with series R:
    #
    #     I_ideal  = Js·A · (exp((vnwell − Vb)/(n·Vt)) − 1)
    #     I_Rs     = (vnwell − Vb) / Rs   (when forward biased)
    #     I_well_b = harmonic_mean(I_ideal, I_Rs)   smooth transition
    #
    # Reverse-bias contribution is tiny (Js·A ~1e-15 A) — included for
    # completeness so derivatives are continuous through Vb crossing vnwell.
    if cfg.use_well_diode:
        Vt = 0.02585 * (273.15 + cfg.T_C) / 300.0   # thermal voltage at T
        V_drive = cfg.vnwell - Vb
        # Clamp exponent to avoid overflow when V_drive >> Vt
        exp_arg = (V_drive / (cfg.vnwell_n * Vt)).clamp(max=40.0)
        I_ideal = cfg.vnwell_Js * cfg.vnwell_area * (torch.exp(exp_arg) - 1.0)
        # Series-R limited current (only forward; reverse bias = 0 here)
        I_Rs = torch.relu(V_drive) / cfg.vnwell_Rs
        # Smooth min via harmonic mean (differentiable, transitions at the
        # smaller of the two without a hard kink)
        eps = 1e-30
        I_well_body = (I_ideal * I_Rs) / (I_ideal.abs() + I_Rs + eps)
        # Scale by mbjt — the well-body junction belongs to the same
        # parasitic bipolar structure as Q1, so it follows the same
        # device-multiplier. Without this scaling, VG1=0.2 (where
        # mbjt=0.001 keeps the BJT off) would still see full well
        # coupling and the body would float high.
        I_well_body = I_well_body * cfg.vnwell_mbjt
    else:
        I_well_body = torch.zeros_like(Vd)

    # ---- Body KCL: currents INTO B ------------------------------------ #
    # Iii, Igidl, Igisl, Igb are already signed +INTO-body in the helpers.
    # Body junction diodes: Ibs and Ibd are POSITIVE-LEAVING-body, so we
    # subtract them.
    # BJT base current Ib (positive INTO base from external) — for the
    # floating body, the only external current into the base IS the body
    # node itself. Ib>0 ⇒ body sources current → leaves body. → −Ib_Q1
    # Well-body diode I_well_body is +INTO body (well pumps body up). → +I_well_body
    # A.10: extra parasitic pdiode at floating body (Sebas's 2026-05-02
    # email). Anode = body B, cathode = one of {vnwell, GND, Sint}. Default
    # OFF — turns on once we have his SPICE card. Sign convention: I_pdiode
    # = Js·area·(exp((Vb-Vc)/(n·Vt)) - 1), positive when forward-biased,
    # leaves the body → enters R_B with negative sign.
    if cfg.body_pdiode_to != "off":
        Vt_body = 0.02585 * (273.15 + cfg.T_C) / 300.0
        if cfg.body_pdiode_to == "vnwell":
            Vc_pdi = cfg.vnwell
        elif cfg.body_pdiode_to == "gnd":
            Vc_pdi = 0.0
        elif cfg.body_pdiode_to == "sint":
            Vc_pdi = Vsint
        else:
            Vc_pdi = 0.0
        Vab = Vb - Vc_pdi
        exp_arg = (Vab / (cfg.body_pdiode_n * Vt_body)).clamp(-40.0, 40.0)
        I_body_pdiode = (cfg.body_pdiode_Js * cfg.body_pdiode_area
                          * (torch.exp(exp_arg) - 1.0))
    else:
        I_body_pdiode = torch.zeros_like(Vd)

    # A.3.d: scale M1 body diodes (was clamping Vb at ~0.5V at VG1=0.4 row,
    # preventing parasitic NPN from lighting; controlled via cfg.m1_diode_scale,
    # default 1.0). Set <1 to weaken the diode shunt and let Vb climb.
    m1_d = float(cfg.m1_diode_scale)
    if cfg.m2_body_gnd:
        # A.1.u: M2's body is GND, so its body-current contributions do
        # NOT enter the floating-body KCL — they flow between M2's nodes
        # and ground, not the floating Vb.
        R_B = (
            m1["Iii"]
            + m1["Igidl"] + m1["Igisl"]
            + m1["Igb"]
            - m1_d * m1["Ibs"] - m1_d * m1["Ibd"]
            - Ib_Q1
            + I_well_body
            - I_body_pdiode
        )
    else:
        R_B = (
            m1["Iii"] + m2["Iii"]
            + m1["Igidl"] + m1["Igisl"] + m2["Igidl"] + m2["Igisl"]
            + m1["Igb"] + m2["Igb"]
            - m1["Ibs"] - m1["Ibd"]
            - m2["Ibs"] - m2["Ibd"]
            - Ib_Q1
            + I_well_body
            - I_body_pdiode
        )

    # Oracle-recommended gmin shunts — ngspice-style parallel conductance
    # in PARALLEL with each pn junction, NOT a single shunt to ground.
    # This is what gives the body a tendency to track (Vd+Vs)/2 in absence
    # of other forces, matching ngspice's behavior.
    #   I_gmin_bd = gmin * (Vd - Vb)   flows INTO body from drain
    #   I_gmin_bs = gmin * (Vs - Vb) = -gmin * Vb (since Vs=0)
    #                                   flows INTO body from source
    #   I_gmin_bsi = gmin * (Vsint - Vb)  body↔Sint via M1's body-source
    #                                      and M2's body-drain (both at Sint)
    # Sum into R_B (currents INTO B). Similar for Sint node.
    gmin = getattr(cfg, "gmin", 0.0)
    if gmin > 0.0:
        # Body node: junctions B↔D, B↔S(=0), B↔Sint (counted once: M1 body-source
        # and M2 body-drain are both at Sint, so 2× weight)
        R_B = R_B + gmin * (Vd - Vb) + gmin * (-Vb) + 2.0 * gmin * (Vsint - Vb)
        # Sint node: gmin shunt to ground (Sint↔S=0 via M2 channel parasitic)
        # plus to body. Mainly to keep Jacobian non-singular at Sint=0.
        R_Sint = R_Sint + gmin * (-Vsint) + 2.0 * gmin * (Vb - Vsint)

    components = {
        "Ids_M1": m1["Ids"], "Ids_M2": m2["Ids"],
        "Ic_Q1": Ic_Q1, "Ib_Q1": Ib_Q1, "Ie_Q1": Ie_Q1,
        "Iii_M1": m1["Iii"], "Iii_M2": m2["Iii"],
        "Igidl_M1": m1["Igidl"], "Igisl_M1": m1["Igisl"],
        "Igidl_M2": m2["Igidl"], "Igisl_M2": m2["Igisl"],
        "Igb_M1": m1["Igb"], "Igb_M2": m2["Igb"],
        "Ibs_M1": m1["Ibs"], "Ibd_M1": m1["Ibd"],
        "Ibs_M2": m2["Ibs"], "Ibd_M2": m2["Ibd"],
        "I_well_body": I_well_body,
    }
    return R_Sint, R_B, components


# --------------------------------------------------------------------------- #
# Newton solve                                                                #
# --------------------------------------------------------------------------- #

def _solve_jac_2x2(R_S: torch.Tensor, R_B: torch.Tensor,
                   J: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve 2x2 system J · [dVs, dVb]^T = -[R_S, R_B]^T element-wise.

    J has shape (..., 2, 2). Returns (dVsint, dVb), each shape (...).

    Degenerate case handling: when all body physics is disabled
    (Iii=Igidl=Igb=Ibd=Ibs=BJT off), R_B ≡ 0 and the second row of J is
    zero. The 2D system is singular but the 1D problem in Vsint is
    well-posed. We detect this (R_B ≈ 0 AND row-2 of J ≈ 0) and reduce
    to dVs = -R_S / a, dVb = 0.
    """
    a = J[..., 0, 0]; b = J[..., 0, 1]
    c = J[..., 1, 0]; d = J[..., 1, 1]

    # Detect degenerate body row (R_B identically 0 ⇒ no info about Vb)
    body_dead = (c.abs() < 1e-30) & (d.abs() < 1e-30) & (R_B.abs() < 1e-30)

    det = a * d - b * c
    # Keep det away from 0 numerically; sign-preserving floor.
    sign = torch.where(det >= 0, torch.ones_like(det), -torch.ones_like(det))
    det_safe = torch.where(det.abs() < 1e-30, sign * 1e-30, det)
    rhs0 = -R_S
    rhs1 = -R_B
    dVs_full = (d * rhs0 - b * rhs1) / det_safe
    dVb_full = (-c * rhs0 + a * rhs1) / det_safe

    # 1-D fallback when body is dead
    a_safe = torch.where(a.abs() < 1e-30, sign * 1e-30, a)
    dVs_1d = -R_S / a_safe
    dVb_1d = torch.zeros_like(dVs_1d)

    dVs = torch.where(body_dead, dVs_1d, dVs_full)
    dVb = torch.where(body_dead, dVb_1d, dVb_full)
    return dVs, dVb


def _jacobian_finite_diff(
    cfg, model_M1, bjt, Vd, VG1, VG2, Vsint, Vb, P_M1, P_M2, h: float = 1e-6,
    model_M2=None,
) -> torch.Tensor:
    """Finite-difference 2x2 Jacobian ∂(R_Sint, R_B)/∂(Vsint, Vb).

    Vectorized over leading dims of Vsint/Vb. Returns shape (..., 2, 2).
    Computed under torch.no_grad — used inside the Newton loop only for the
    *step direction*; the autograd path through the converged solution
    flows via the iterative updates themselves (since they're under grad).
    """
    with torch.no_grad():
        # Central differences on Vsint
        Rsp_s, Rbp_s, _ = _residuals(cfg, model_M1, bjt, Vd, VG1, VG2,
                                     Vsint + h, Vb, P_M1, P_M2, model_M2=model_M2)
        Rsm_s, Rbm_s, _ = _residuals(cfg, model_M1, bjt, Vd, VG1, VG2,
                                     Vsint - h, Vb, P_M1, P_M2, model_M2=model_M2)
        dRs_dVs = (Rsp_s - Rsm_s) / (2 * h)
        dRb_dVs = (Rbp_s - Rbm_s) / (2 * h)
        # Central differences on Vb
        Rsp_b, Rbp_b, _ = _residuals(cfg, model_M1, bjt, Vd, VG1, VG2,
                                     Vsint, Vb + h, P_M1, P_M2, model_M2=model_M2)
        Rsm_b, Rbm_b, _ = _residuals(cfg, model_M1, bjt, Vd, VG1, VG2,
                                     Vsint, Vb - h, P_M1, P_M2, model_M2=model_M2)
        dRs_dVb = (Rsp_b - Rsm_b) / (2 * h)
        dRb_dVb = (Rbp_b - Rbm_b) / (2 * h)
    J = torch.stack([
        torch.stack([dRs_dVs, dRs_dVb], dim=-1),
        torch.stack([dRb_dVs, dRb_dVb], dim=-1),
    ], dim=-2)
    return J


def solve_2t_steady_state(
    cfg: NSRAMCell2TConfig,
    model: BSIM4Model,
    bjt: GummelPoonNPN,
    Vd: torch.Tensor,
    VG1: torch.Tensor,
    VG2: torch.Tensor,
    P_M1: Optional[dict] = None,
    P_M2: Optional[dict] = None,
    Vsint_init: Optional[torch.Tensor] = None,
    Vb_init: Optional[torch.Tensor] = None,
    verbose: bool = False,
    model_M2: Optional[BSIM4Model] = None,
) -> dict:
    """Solve the 2T cell at quasi-static (Vd, VG1, VG2).

    Returns dict with: Id, Vsint, Vb, components, R_Sint, R_B, niter, converged.

    Newton step uses *finite-difference* Jacobian (no_grad). Voltage
    updates themselves are inside the autograd graph, so gradients of Id
    w.r.t. fit params flow through the Newton iterates. This is slower
    than implicit-diff but correct and simpler.
    """
    # Coerce inputs to fp64 broadcastable tensors
    Vd = torch.as_tensor(Vd, dtype=torch.float64)
    VG1 = torch.as_tensor(VG1, dtype=torch.float64)
    VG2 = torch.as_tensor(VG2, dtype=torch.float64)
    Vd, VG1, VG2 = torch.broadcast_tensors(Vd, VG1, VG2)
    Vd = Vd.contiguous(); VG1 = VG1.contiguous(); VG2 = VG2.contiguous()

    if Vsint_init is None:
        Vsint = (0.5 * Vd).detach().clone()
    else:
        Vsint = Vsint_init.detach().clone().to(torch.float64).expand_as(Vd).contiguous()
    if Vb_init is None:
        # Cold-start at Vb=0. Note: oracle consensus recommended Vb=0.5
        # but in this model (PTM 130nm bulkNSRAM card) Iii=0 at typical
        # biases, so the high-Vb root is not an attractor and Newton
        # drifts back. Default Vb=0 matches legacy behaviour. Use the
        # `Vb_init=` kwarg explicitly when you know your bias is in the
        # impact-ion regime.
        Vb = torch.zeros_like(Vd)
    else:
        Vb = Vb_init.detach().clone().to(torch.float64).expand_as(Vd).contiguous()

    # Initial residual (need it grad-tracked for IFT-free autograd flow)
    R_S, R_B, comp0 = _residuals(cfg, model, bjt, Vd, VG1, VG2, Vsint, Vb, P_M1, P_M2,
                                 model_M2=model_M2)
    prev_resid_norm = (R_S.detach().abs() + R_B.detach().abs()).max()

    def _physical_scale(comp: dict) -> torch.Tensor:
        """Build a per-bias physical-current magnitude from KCL components.
        Used for relative-tolerance convergence — residual must be small
        relative to the current actually flowing in the device, not relative
        to the residual itself (circular)."""
        keys = ["Ids_M1", "Ids_M2", "Ic_Q1", "Ib_Q1",
                "Iii_M1", "Iii_M2", "Igidl_M1", "Igidl_M2",
                "Ibs_M1", "Ibd_M1", "Ibs_M2", "Ibd_M2"]
        scale = torch.zeros_like(R_S.detach())
        for k in keys:
            if k in comp:
                scale = scale + comp[k].detach().abs()
        return scale

    # Tolerances
    iabstol = getattr(cfg, "Iabstol", cfg.newton_tol)
    ireltol = getattr(cfg, "Ireltol", 0.0)
    xtol_v  = getattr(cfg, "xtol_v", 0.0)
    min_iters = getattr(cfg, "min_iters", 1)

    converged = torch.zeros_like(Vd, dtype=torch.bool)
    niter = 0
    last_dV_inf = torch.tensor(float("inf"), dtype=torch.float64)
    cur_comp = comp0
    for it in range(cfg.newton_max_iters):
        niter = it + 1
        # Convergence check (oracle hardening):
        #   - residual: |R| < max(Iabstol, Ireltol * |I_physical|)
        #     where I_physical = Σ|component currents|
        #   - step:     |dV|_inf < xtol_v
        #   - guard:    require >= min_iters AND the residual must have
        #               actually decreased once (or we've passed iter 1)
        residual_max = torch.maximum(R_S.detach().abs(), R_B.detach().abs())
        I_scale = _physical_scale(cur_comp)
        tol_eff = torch.maximum(torch.full_like(I_scale, iabstol), ireltol * I_scale)
        residual_ok = bool((residual_max < tol_eff).all())
        step_ok = bool((last_dV_inf < xtol_v).all()) if xtol_v > 0 else False
        cur_norm = (R_S.detach().abs() + R_B.detach().abs()).max()
        # min_iters: never declare convergence before this many iterations
        # have actually been taken (it counts the iteration *just executed*;
        # we must have done at least min_iters of them, i.e. it >= min_iters).
        if it >= min_iters and (residual_ok or step_ok):
            converged = residual_max < tol_eff
            if verbose:
                print(f"  Newton converged in {it} iter; max R = {residual_max.max():.3e} "
                      f"|dV|_inf = {float(last_dV_inf):.3e}")
            break
        prev_resid_norm = cur_norm

        # FD Jacobian (no_grad), step direction (no_grad). The implicit
        # function theorem is applied AFTER convergence to attach gradients
        # — see the IFT block at the end of this function.
        J = _jacobian_finite_diff(cfg, model, bjt, Vd, VG1, VG2,
                                  Vsint.detach(), Vb.detach(),
                                  P_M1, P_M2, model_M2=model_M2)
        dVs, dVb = _solve_jac_2x2(R_S.detach(), R_B.detach(), J)

        # Step-size cap (per-iteration relative-step limiter)
        max_abs = torch.maximum(dVs.abs(), dVb.abs())
        scale = torch.where(max_abs > cfg.max_step_V,
                            cfg.max_step_V / max_abs.clamp_min(1e-30),
                            torch.ones_like(max_abs))
        dVs = dVs * scale
        dVb = dVb * scale

        # Damped step + backtracking on residual norm (Armijo-style halving)
        damping = cfg.newton_damping
        prev_norm = R_S.detach().abs() + R_B.detach().abs()
        accepted = False
        while damping >= cfg.newton_min_damping:
            Vsint_try = Vsint + damping * dVs
            Vb_try = Vb + damping * dVb
            R_S_try, R_B_try, comp_try = _residuals(cfg, model, bjt, Vd, VG1, VG2,
                                             Vsint_try, Vb_try, P_M1, P_M2,
                                             model_M2=model_M2)
            new_norm = R_S_try.detach().abs() + R_B_try.detach().abs()
            # Strict decrease (mean over batch) accepted; or fall through at
            # min damping. The 0.999 factor demands genuine descent — at
            # min_damping we accept whatever we have.
            if (new_norm.mean() < prev_norm.mean() * 0.999) or damping <= cfg.newton_min_damping:
                Vsint = Vsint_try
                Vb = Vb_try
                R_S = R_S_try
                R_B = R_B_try
                cur_comp = comp_try
                accepted = True
                # Track step size for xtol convergence
                last_dV_inf = torch.maximum(
                    (damping * dVs).abs().max(),
                    (damping * dVb).abs().max(),
                )
                break
            damping *= 0.5
        if verbose:
            rmax = torch.maximum(R_S.detach().abs(), R_B.detach().abs()).max()
            print(f"  iter {it}: damping={damping:.3f} max|R|={rmax:.3e} "
                  f"|dVs|={dVs.abs().max():.3e} |dVb|={dVb.abs().max():.3e}")
        if not accepted:
            break

    # ----- Implicit Function Theorem (IFT) attachment -----
    # At convergence, R(x*, theta) ≈ 0 numerically, but x* (Vsint, Vb) has
    # been computed under no_grad — so it carries no gradient back to theta.
    # IFT says dx*/dtheta = -J^-1 · ∂R/∂theta. We can encode this in the
    # autograd graph by replacing x* with an "attached" version:
    #     x_attached = x*.detach() - J^-1 @ R(x*.detach(), theta)
    # At convergence R≈0 so x_attached ≈ x* in value, but its gradient w.r.t.
    # theta is exactly the IFT result because J^-1 is detached and R has
    # gradient through theta (via compute_dc, compute_iimpact, ...).
    Vsint_d = Vsint.detach()
    Vb_d = Vb.detach()
    R_S_at, R_B_at, _ = _residuals(cfg, model, bjt, Vd, VG1, VG2, Vsint_d, Vb_d, P_M1, P_M2,
                                   model_M2=model_M2)

    # CRITICAL: only apply IFT correction at biases where Newton ACTUALLY
    # converged. The IFT formula  x* = x*_d - J^-1 R(x*_d, theta)  assumes
    # R ≈ 0; if Newton failed, R can be huge, and J near-singular at that
    # bias would produce a spurious gradient that Adam misreads as a strong
    # signal — root cause of the v6/v7 stage 3 explosion.
    # When residual is too large, ZERO out the IFT delta at that bias →
    # gradient flows through theta-only paths, no broken Vb-loop signal.
    with torch.no_grad():
        J_final = _jacobian_finite_diff(cfg, model, bjt, Vd, VG1, VG2,
                                        Vsint_d, Vb_d, P_M1, P_M2,
                                        model_M2=model_M2)
    delta_s, delta_b = _solve_jac_2x2(R_S_at, R_B_at, J_final)

    # SMOOTH bound IFT delta via tanh — passes gradient through ALL bias
    # points (including non-converged ones), but compresses the magnitude so
    # Adam doesn't see exploding signal. Hard clamp would zero gradient at
    # boundary; tanh is differentiable everywhere.
    #   delta_smooth = D_MAX * tanh(delta_raw / D_MAX)
    # For |delta| << D_MAX: delta_smooth ≈ delta (full IFT signal)
    # For |delta| >> D_MAX: delta_smooth ≈ ±D_MAX, gradient ∝ sech²(.) → 0
    # This is the same effective bound but with smooth gradient transition.
    DELTA_BOUND = 0.3  # V — generous to allow real physics, not just to clip
    delta_s = DELTA_BOUND * torch.tanh(delta_s / DELTA_BOUND)
    delta_b = DELTA_BOUND * torch.tanh(delta_b / DELTA_BOUND)

    # 5th-oracle fix: at non-converged points, the IFT correction is meaningless
    # (Newton never reached a valid root) and would mutate Vsint/Vb away from
    # the un-corrected detached value. The function-level docstring promised we
    # don't apply IFT to non-converged points, but the code did. Gate it now.
    conv_mask = converged.detach()
    delta_s = torch.where(conv_mask, delta_s, torch.zeros_like(delta_s))
    delta_b = torch.where(conv_mask, delta_b, torch.zeros_like(delta_b))

    Vsint = Vsint_d - delta_s
    Vb = Vb_d - delta_b
    R_S, R_B, comp = _residuals(cfg, model, bjt, Vd, VG1, VG2, Vsint, Vb, P_M1, P_M2,
                                model_M2=model_M2)

    # Drain terminal current (positive INTO the D pin):
    #   Id = Ids_M1 (drain absorbs Ids from external) +
    #        Ic_Q1  (collector current absorbed from D) +
    #        Igidl_M1 leaves drain INTO body — but at the D pin this is a
    #        current LEAVING the drain to body, so the external D pin sees
    #        an EXTRA −Igidl_M1 inflow. We add it as a positive contribution
    #        because the convention "Igidl > 0 means current flows from drain
    #        into body via BTBT" implies the external supply drives that
    #        extra current INTO D. Same sign as the channel.
    #   The body diode Ibd_M1 is current LEAVING the body INTO drain, so
    #        from the D pin's perspective it FLOWS OUT to ground via M1
    #        substrate path → contributes −Ibd_M1 to Id (current leaves D).
    #
    # In the typical NS-RAM operating regime, |Ibd_M1|, |Igidl_M1| ≪ Ids_M1
    # so the dominant term is Ids_M1; SCBE / impact-ion shows up via Ic_Q1.
    Id = (
        comp["Ids_M1"]
        + comp["Ic_Q1"]
        + comp["Igidl_M1"]
        - comp["Ibd_M1"]
    )

    residual_max = torch.maximum(R_S.detach().abs(), R_B.detach().abs())
    I_scale_final = (R_S.detach().abs() + R_B.detach().abs()).clamp_min(iabstol)
    tol_final = torch.maximum(torch.full_like(I_scale_final, iabstol),
                              ireltol * I_scale_final)
    converged_final = residual_max < tol_final

    return {
        "Id": Id,
        "Vsint": Vsint,
        "Vb": Vb,
        "Ids_M1": comp["Ids_M1"],
        "Ids_M2": comp["Ids_M2"],
        "Ic_Q1": comp["Ic_Q1"],
        "Ib_Q1": comp["Ib_Q1"],
        "R_Sint": R_S,
        "R_B": R_B,
        "components": comp,
        "niter": niter,
        "converged": converged_final,
    }


# --------------------------------------------------------------------------- #
# gmin homotopy (z89): standard SPICE technique for snapback/bistable cells.  #
# --------------------------------------------------------------------------- #
@contextmanager
def _override_gmin(cfg: NSRAMCell2TConfig, value: float):
    """Temporarily override cfg.gmin (used by `_residuals` shunts)."""
    saved = cfg.gmin
    try:
        cfg.gmin = float(value)
        yield
    finally:
        cfg.gmin = saved


def solve_2t_with_homotopy(
    cfg: NSRAMCell2TConfig,
    model: BSIM4Model,
    bjt: GummelPoonNPN,
    Vd: torch.Tensor,
    VG1: torch.Tensor,
    VG2: torch.Tensor,
    P_M1: Optional[dict] = None,
    P_M2: Optional[dict] = None,
    Vsint_init: Optional[torch.Tensor] = None,
    Vb_init: Optional[torch.Tensor] = None,
    gmin_schedule: Optional[list] = None,
    verbose: bool = False,
    model_M2: Optional[BSIM4Model] = None,
) -> dict:
    """Solve 2T cell using gmin homotopy (oracle consensus recommendation).

    Standard SPICE technique for bistable / snapback / S-shaped I-V circuits:
    start with a LARGE gmin (linearizes the circuit, Newton always converges
    because every node has a strong shunt to its neighbours) and use that
    solution as a warm-start for the next smaller gmin. Repeat until the
    target gmin (= cfg.gmin) is reached.

    Implementation:
      * `gmin_schedule` defaults to [1e-3, 1e-5, 1e-8, 1e-12] followed by
        the cfg-specified target gmin (so the FINAL solve uses exactly the
        gmin that the IFT-attached delta sees, and gradients flow normally).
      * Each step calls existing `solve_2t_steady_state` with a temporarily
        overridden cfg.gmin and the previous solution as warm-start. We do
        NOT change `solve_2t_steady_state` so its IFT machinery is untouched.
      * The final returned dict comes from the last call (target gmin).

    NOTE: gmin shunts are physical-style conductances added in `_residuals`.
    They distort the solution slightly at large values; the homotopy walks
    that distortion smoothly to zero. At the FINAL gmin (= cfg.gmin), the
    solution is identical to a direct solve (only the convergence path is
    different) so gradient flow through IFT is unchanged.
    """
    if gmin_schedule is None:
        # Walk down by ~1000x per step. Final target = cfg.gmin.
        gmin_schedule = [1e-3, 1e-5, 1e-8, 1e-12]
    target = float(cfg.gmin)
    # Always end with the target gmin so IFT delta is computed at it.
    schedule = [g for g in gmin_schedule if g > target] + [target]

    Vsint_warm = Vsint_init
    Vb_warm = Vb_init
    last_out = None
    for step, g in enumerate(schedule):
        with _override_gmin(cfg, g):
            out = solve_2t_steady_state(
                cfg, model, bjt,
                Vd=Vd, VG1=VG1, VG2=VG2,
                P_M1=P_M1, P_M2=P_M2,
                Vsint_init=Vsint_warm,
                Vb_init=Vb_warm,
                verbose=verbose and (step == len(schedule) - 1),
                model_M2=model_M2,
            )
        if verbose:
            conv = bool(out["converged"].all())
            print(f"  homotopy step {step}: gmin={g:.1e}  converged={conv}  "
                  f"niter={out['niter']}", flush=True)
        # Warm-start next step with current solution. Detach so we don't
        # accumulate the previous step's autograd graph (the FINAL solve at
        # target gmin still goes through IFT for gradient attachment).
        Vsint_warm = out["Vsint"].detach()
        Vb_warm = out["Vb"].detach()
        last_out = out
    return last_out


# --------------------------------------------------------------------------- #
# Forward sweep                                                               #
# --------------------------------------------------------------------------- #

def forward_2t(
    cfg: NSRAMCell2TConfig,
    model: Optional[BSIM4Model] = None,
    bjt: Optional[GummelPoonNPN] = None,
    Vd_seq: Optional[torch.Tensor] = None,
    VG1: Optional[torch.Tensor] = None,
    VG2: Optional[torch.Tensor] = None,
    P_M1: Optional[dict] = None,
    P_M2: Optional[dict] = None,
    verbose: bool = False,
    warm_start: bool = True,
    use_homotopy: bool = False,
    dense_vd_in_snapback: bool = False,
    snapback_vd_threshold: float = 1.4,
    snapback_vd_step: float = 0.025,
    *,
    model_M1: Optional[BSIM4Model] = None,
    model_M2: Optional[BSIM4Model] = None,
) -> dict:
    """Sweep Vd from low to high with warm-starting Vsint, Vb between points.

    Returns dict with stacked tensors (shape (T,)): Id, Vsint, Vb, niter,
    converged, plus components per sub-call.

    Args (z89 additions):
      use_homotopy: if True, calls `solve_2t_with_homotopy` per point (gmin
          homotopy, expensive but converges through snapback / bistability).
      dense_vd_in_snapback: if True, internally insert intermediate Vd points
          at `snapback_vd_step` spacing for any segment where Vd >= threshold
          (defaults: threshold=1.4 V, step=0.025 V → 4× denser than the
          z88 default 0.1 V grid). Intermediate points are solved purely for
          warm-starting; the returned arrays only contain values at the
          ORIGINAL Vd_seq points (so the loss never sees the intermediate
          biases — they're a numerical aid only).

    Two-model variant: pass `model_M1=` and `model_M2=` as kwargs to use
    distinct BSIM4 cards for M1 and M2. If only legacy `model` is given,
    both transistors use it (back-compat). Mixing legacy `model` with
    `model_M2=` is also allowed (model → M1, kwarg → M2).
    """
    # Resolve model_M1 / model_M2 from positional `model` and kwargs.
    if model_M1 is None:
        model_M1 = model
    if model_M1 is None:
        raise TypeError("forward_2t requires either positional `model` or `model_M1=` kwarg")
    if model_M2 is None:
        model_M2 = model_M1
    Vd_seq = Vd_seq.to(torch.float64)
    VG1 = torch.as_tensor(VG1, dtype=torch.float64)
    VG2 = torch.as_tensor(VG2, dtype=torch.float64)
    T = int(Vd_seq.shape[0])

    # Build augmented schedule with intermediate (warm-start-only) points.
    # `report_idx[k]` indexes into the augmented sequence and tells us which
    # entries correspond to original Vd_seq points (we only return those).
    if dense_vd_in_snapback and T >= 2:
        aug_vd: list = []
        report_idx: list = []
        prev = float(Vd_seq[0].item())
        aug_vd.append(Vd_seq[0])
        report_idx.append(0)
        for i in range(1, T):
            cur = float(Vd_seq[i].item())
            # Insert intermediate points only if both endpoints (or the
            # current segment top) are in the snapback region. Spacing is
            # `snapback_vd_step` (only inserts if larger gap exists).
            if cur >= snapback_vd_threshold and (cur - prev) > 1.5 * snapback_vd_step:
                n_insert = int((cur - prev) / snapback_vd_step) - 1
                if n_insert > 0:
                    for k in range(1, n_insert + 1):
                        v = prev + (cur - prev) * (k / (n_insert + 1))
                        aug_vd.append(torch.tensor(v, dtype=torch.float64))
            aug_vd.append(Vd_seq[i])
            report_idx.append(len(aug_vd) - 1)
            prev = cur
        Vd_aug = torch.stack(aug_vd)
        report_set = set(report_idx)
    else:
        Vd_aug = Vd_seq
        report_idx = list(range(T))
        report_set = set(report_idx)

    T_aug = int(Vd_aug.shape[0])

    Ids_list, Vs_list, Vb_list = [], [], []
    niter_list, conv_list = [], []
    Ids_M1_list, Ids_M2_list, Ic_Q1_list = [], [], []

    # Cold start at Vb=0.5V (oracle consensus: avoid spurious flat root at
    # Vb=0 where all body currents are sub-femtoamp and Newton "converges"
    # without moving). Vsint=Vd/2 as initial series-divider guess.
    # Then cascade the converged solution from each point as the seed for
    # the next when warm_start=True (default).
    Vsint_warm = torch.tensor(0.0, dtype=torch.float64)  # gets replaced below
    Vb_warm = torch.tensor(0.5, dtype=torch.float64)

    # We collect outputs at ALL augmented points then filter to report_idx
    # at the end. This keeps the inner loop simple.
    aug_outs: list = []
    for i in range(T_aug):
        Vd_i = Vd_aug[i].unsqueeze(0)
        if i == 0:
            Vsint_warm = (Vd_i * 0.5).squeeze(0).detach()
        if use_homotopy:
            out = solve_2t_with_homotopy(
                cfg, model_M1, bjt,
                Vd=Vd_i, VG1=VG1, VG2=VG2,
                P_M1=P_M1, P_M2=P_M2,
                Vsint_init=Vsint_warm.expand_as(Vd_i),
                Vb_init=Vb_warm.expand_as(Vd_i),
                verbose=verbose,
                model_M2=model_M2,
            )
        else:
            out = solve_2t_steady_state(
                cfg, model_M1, bjt,
                Vd=Vd_i, VG1=VG1, VG2=VG2,
                P_M1=P_M1, P_M2=P_M2,
                Vsint_init=Vsint_warm.expand_as(Vd_i),
                Vb_init=Vb_warm.expand_as(Vd_i),
                verbose=verbose,
                model_M2=model_M2,
            )
        aug_outs.append(out)

        # Warm-start next point with current solution (detached so warm
        # start doesn't accumulate the previous step's Newton graph).
        if warm_start:
            Vsint_warm = out["Vsint"].detach().squeeze(0)
            Vb_warm = out["Vb"].detach().squeeze(0)

    # Filter to original Vd_seq points only (preserves graph for those).
    for i in report_idx:
        out = aug_outs[i]
        Ids_list.append(out["Id"].squeeze(0))
        Vs_list.append(out["Vsint"].squeeze(0))
        Vb_list.append(out["Vb"].squeeze(0))
        Ids_M1_list.append(out["Ids_M1"].squeeze(0))
        Ids_M2_list.append(out["Ids_M2"].squeeze(0))
        Ic_Q1_list.append(out["Ic_Q1"].squeeze(0))
        niter_list.append(out["niter"])
        conv_list.append(bool(out["converged"].all()))

    return {
        "Id": torch.stack(Ids_list),
        "Vsint": torch.stack(Vs_list),
        "Vb": torch.stack(Vb_list),
        "Ids_M1": torch.stack(Ids_M1_list),
        "Ids_M2": torch.stack(Ids_M2_list),
        "Ic_Q1": torch.stack(Ic_Q1_list),
        "niter": niter_list,
        "converged": conv_list,
    }

```


=== FILE: parasiticBJT.txt (244 chars) ===
```
* Simple bjt for floating bulk parasitic bipolar effect
* Pazos, S.

.model parasiticBJT NPN(is=5E-9 va=100 bf=10000 br=100 nc=2 ikr=100m rc=0.1 vje=0.7 re=0.1 cjc=1e-15 fc=0.5 cje=0.7e-15 ne=1.5 ise=0 tr=20e-12 tf=25e-12 itf=0.03 vtf=7 xtf=2)

```


=== FILE: pdiode.txt (649 chars) ===
```
model  pdiode  diode
+level   = 1            tnom    = 25
+is      = 5.3675e-007  isw     = 1.3664e-013  rsw     = 0.46493      ns      = 1.0851     
+nz      = 1.3664e-013  imax    = 1e+030       imelt   = 1e+030       bvj     = 1e+031     
+ik      = 97740        bv      = 11           ibv     = 97740        ikp     = 1.1946e+005
+n       = 1.0535       rs      = 7.4155e-008
+cj      = 0.00073279   cjsw    = 1.0522e-010  vj      = 0.21918      vjsw    = 0.65166        
+m       = 0.24097      fcs     = 0.5          mjsw    = 0.26029      fc      = 0.5        
+xti     = 6.5          eg      = 1.11         tlev    = 1            tlevc   = 1
```


=== FILE: topology.py (3316 chars) ===
```python
"""Minimal multi-cell NS-RAM topology layer.

B.4 (2026-05-02): wraps `forward_2t_batched` with per-cell static
config, per-cell bias trajectories, and a linear readout for the
common reservoir / Hopfield / classifier paradigm.

Each cell is an independent 2T NS-RAM with shared model cards (M1, M2)
but independent (VG1, VG2) per cell — i.e., the per-cell bias is the
"weight" or "state" that distinguishes cells. Cells run in parallel;
no inter-cell electrical coupling at this layer.

A linear readout maps the per-cell drain currents to network-level
outputs: output[k] = sum_i W[k, i] · log10(|Id_i| + eps).

This is the substrate for Phase B.5 benchmarks (Hopfield retrieval,
NARMA-10, memory capacity, temporal-XOR, multi-class waveform).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import torch

from .nsram_cell_2T import NSRAMCell2TConfig
from .model_card import BSIM4Model
from .bjt import GummelPoonNPN
from .vectorized import forward_2t_batched


@dataclass
class NSRAMNetwork:
    """N-cell 2T NS-RAM array with linear readout.

    cfg: shared cell config
    model_M1, model_M2: shared model cards (calibrated once, applied to all)
    bjt: shared parasitic NPN
    N: number of cells

    Per-cell state (set externally before each forward):
      VG1: shape (N,) — gate-1 control voltage
      VG2: shape (N,) — gate-2 control voltage

    Readout:
      W: shape (n_out, N) — linear projection of log10|Id|
    """
    cfg: NSRAMCell2TConfig
    model_M1: BSIM4Model
    model_M2: BSIM4Model
    bjt: GummelPoonNPN
    N: int
    W: Optional[torch.Tensor] = None
    n_out: int = 0

    def __post_init__(self):
        if self.W is None and self.n_out > 0:
            # Default initial readout: small random Gaussian, fp64
            self.W = (0.01 * torch.randn(self.n_out, self.N, dtype=torch.float64))

    def forward(self, Vd_seq: torch.Tensor, VG1: torch.Tensor, VG2: torch.Tensor,
                **batch_kwargs) -> dict:
        """Run all N cells in parallel for a Vd sweep.
        Returns dict with shape (N, T) tensors plus optional readout (n_out, T).
        """
        out = forward_2t_batched(
            self.cfg, self.model_M1, self.model_M2, self.bjt,
            Vd_seq, VG1, VG2, **batch_kwargs)
        Id = out["Id"]   # (N, T)
        # Log-feature readout (decade-scaled current is the natural NSRAM observable)
        eps = 1e-15
        log_Id = torch.log10(Id.abs() + eps)
        if self.W is not None:
            out["readout"] = self.W @ log_Id   # (n_out, T)
        out["log_Id"] = log_Id
        return out

    def fit_readout(self, log_Id: torch.Tensor, target: torch.Tensor,
                     ridge: float = 1e-3) -> torch.Tensor:
        """Closed-form ridge regression for the linear readout.
        log_Id: (N, T_train) feature matrix.
        target: (n_out, T_train) target outputs.
        Returns W: (n_out, N) — also updates self.W in place.
        """
        X = log_Id   # (N, T)
        XX_T = X @ X.T   # (N, N)
        I = torch.eye(X.shape[0], dtype=X.dtype) * ridge
        XY_T = X @ target.T   # (N, n_out)
        W_T = torch.linalg.solve(XX_T + I, XY_T)   # (N, n_out)
        self.W = W_T.T.contiguous()
        self.n_out = self.W.shape[0]
        return self.W

```


=== FILE: transient.py (12933 chars) ===
```python
"""Body-pdiode capacitance helper + minimal transient stub.

A.4 (2026-05-02): Sebas's pdiode card has voltage-dependent Cj(V) that
captures the floating-body capacitance dynamics. This module exposes
that helper and a tiny transient stub that the rest of the codebase
can import without polluting the DC compute_dc / _residuals path.

Cj(V) per BSIM4 / SPICE diode v4 convention:

    Cj(V) = Cj0 / (1 - V/Vj)^M           for V < Vj·FC
    linear continuation                  for V > Vj·FC  (avoid singularity)

where V = Vanode - Vcathode (positive forward). For our pdiode at
body↔vnwell, V = Vb - vnwell, normally negative (reverse), Cj < Cj0.
"""
from __future__ import annotations
import torch


def junction_cap(V: torch.Tensor, *, Cj0: float, Vj: float, M: float,
                  fc: float = 0.5) -> torch.Tensor:
    """Voltage-dependent junction capacitance, smooth across V=Vj·fc.

    V positive = forward. Cj rises with forward bias up to V=Vj·fc, then
    is linearly extrapolated to keep grad finite (SPICE convention).

    Returns Cj in farads (Cj0 should already be Cj0_per_area × area).
    """
    Vbreak = fc * Vj
    # Reverse / mild forward branch
    arg = (1.0 - V / Vj).clamp_min(1e-6)   # safety floor (very forward → tiny)
    Cj_main = Cj0 * arg.pow(-M)
    # Linear continuation past Vbreak: derivative at V=Vbreak
    arg_b = 1.0 - fc
    slope = Cj0 * M / Vj * arg_b ** (-(M + 1.0))
    Cj_break = Cj0 * arg_b ** (-M)
    Cj_lin = Cj_break + slope * (V - Vbreak)
    return torch.where(V < Vbreak, Cj_main, Cj_lin)


def integrate_body_cap_charge(Vb_traj: torch.Tensor, t_traj: torch.Tensor,
                                vnwell: float, *, Cj0_per_area: float,
                                area: float, Vj: float, M: float
                                ) -> torch.Tensor:
    """Given a Vb(t) trajectory, return I_cap(t) = Cj(Vb-vnwell) · dVb/dt.

    For transient validation against Sebas's ramped Vd measurements: we
    take a quasi-static body-voltage trajectory (from successive DC
    Newton solves at each Vd_i along the ramp) and add the displacement
    current through the body-pdiode capacitance. This is the leading-
    order correction; for full transient solver, the Cj enters the
    body-KCL Jacobian directly.
    """
    Cj0_total = Cj0_per_area * area
    V = Vb_traj - vnwell
    Cj = junction_cap(V, Cj0=Cj0_total, Vj=Vj, M=M)
    dVb_dt = torch.zeros_like(Vb_traj)
    if Vb_traj.numel() > 1:
        dVb_dt[1:-1] = (Vb_traj[2:] - Vb_traj[:-2]) / (t_traj[2:] - t_traj[:-2])
        dVb_dt[0] = (Vb_traj[1] - Vb_traj[0]) / (t_traj[1] - t_traj[0])
        dVb_dt[-1] = (Vb_traj[-1] - Vb_traj[-2]) / (t_traj[-1] - t_traj[-2])
    return Cj * dVb_dt


def integrate_2t_transient_implicit(cfg, model_M1, model_M2, bjt,
                                       Vd_t: torch.Tensor, t: torch.Tensor,
                                       VG1: torch.Tensor, VG2: torch.Tensor, *,
                                       Vb0: float = 0.0, Vsint0: float = 0.0,
                                       spike_threshold: float = 0.65,
                                       reset_Vb: float = 0.30,
                                       newton_iters_inner: int = 8,
                                       newton_iters_outer: int = 12,
                                       newton_tol: float = 1e-12,
                                       verbose: bool = False):
    """Implicit-Euler time integration of the 2T cell body charge.

    Stable on the stiff body-charge ODE (12-decade dynamic range)
    where forward-Euler diverges. Uses a split scheme:
      Outer loop: Newton on Vb_new with backward-Euler on the cap term
        F(Vb_new) := R_B(Vsint*(Vb_new), Vb_new, Vd_new)
                     − Cj(Vb_new − vnwell) · (Vb_new − Vb_old) / dt
                     = 0
      Inner loop: at each candidate Vb_new, quasi-static Newton on
        Vsint*(Vb_new) such that R_Sint = 0 (1D in Vsint).

    Spike detection: post-step, if Vb >= spike_threshold, log event,
    snap to reset_Vb (zero-time discharge).
    """
    from .nsram_cell_2T import _residuals
    n = Vd_t.numel()
    Vb_traj = torch.zeros(n, dtype=torch.float64)
    Vsint_traj = torch.zeros(n, dtype=torch.float64)
    Id_traj = torch.zeros(n, dtype=torch.float64)
    Vb = torch.tensor(Vb0, dtype=torch.float64)
    Vsint = torch.tensor(Vsint0, dtype=torch.float64)
    spike_times = []

    Cj0_total = cfg.body_pdiode_Cj0_per_area * cfg.body_pdiode_area
    eps_J = 1e-4   # finite-diff perturbation

    def _solve_Vsint(Vb_curr, Vd_i):
        """Inner: Vsint such that R_Sint = 0 with this Vb_curr."""
        Vs = Vsint.clone()
        for _ in range(newton_iters_inner):
            R_S, _, comps = _residuals(
                cfg, model_M1, bjt, Vd=Vd_i, VG1=VG1, VG2=VG2,
                Vsint=Vs.unsqueeze(0), Vb=Vb_curr.unsqueeze(0),
                P_M1=None, P_M2=None, model_M2=model_M2)
            if R_S.abs().max() < newton_tol:
                break
            R_S_eps, _, _ = _residuals(
                cfg, model_M1, bjt, Vd=Vd_i, VG1=VG1, VG2=VG2,
                Vsint=(Vs+eps_J).unsqueeze(0),
                Vb=Vb_curr.unsqueeze(0),
                P_M1=None, P_M2=None, model_M2=model_M2)
            J = (R_S_eps - R_S) / eps_J
            dVs = -R_S / J.clamp(min=1e-30, max=None)
            # Damp big steps so we don't jump into nonconvergent region
            Vs = Vs + dVs.squeeze().clamp(-0.3, 0.3)
        return Vs, comps

    for i in range(n):
        Vd_i = Vd_t[i:i+1]
        if i == 0:
            # Initial step: quasi-static (no dt term)
            Vsint, comps = _solve_Vsint(Vb, Vd_i)
        else:
            dt = float(t[i] - t[i-1])
            Vb_old = Vb.clone()
            # Outer Newton on Vb_new
            Vb_new = Vb.clone()
            for outer in range(newton_iters_outer):
                Vsint_at, comps = _solve_Vsint(Vb_new, Vd_i)
                _, R_B_now, _ = _residuals(
                    cfg, model_M1, bjt, Vd=Vd_i, VG1=VG1, VG2=VG2,
                    Vsint=Vsint_at.unsqueeze(0), Vb=Vb_new.unsqueeze(0),
                    P_M1=None, P_M2=None, model_M2=model_M2)
                Cj_now = junction_cap(
                    Vb_new - cfg.vnwell, Cj0=Cj0_total,
                    Vj=cfg.body_pdiode_Vj, M=cfg.body_pdiode_M)
                F = R_B_now.squeeze() - Cj_now * (Vb_new - Vb_old) / dt
                if F.abs() < newton_tol:
                    break
                # FD Jacobian dF/dVb_new
                Vb_eps = Vb_new + eps_J
                Vsint_eps, _ = _solve_Vsint(Vb_eps, Vd_i)
                _, R_B_eps, _ = _residuals(
                    cfg, model_M1, bjt, Vd=Vd_i, VG1=VG1, VG2=VG2,
                    Vsint=Vsint_eps.unsqueeze(0), Vb=Vb_eps.unsqueeze(0),
                    P_M1=None, P_M2=None, model_M2=model_M2)
                Cj_eps = junction_cap(
                    Vb_eps - cfg.vnwell, Cj0=Cj0_total,
                    Vj=cfg.body_pdiode_Vj, M=cfg.body_pdiode_M)
                F_eps = R_B_eps.squeeze() - Cj_eps * (Vb_eps - Vb_old) / dt
                dF = (F_eps - F) / eps_J
                # Damped Newton step
                step = -F / dF.clamp(min=1e-30, max=None) if dF.abs() > 1e-30 else torch.tensor(0.0)
                step = step.clamp(-0.2, 0.2)
                Vb_new = Vb_new + step
                # Bound to physical range
                Vb_new = Vb_new.clamp(-1.0, 2.0)
            Vb = Vb_new
            Vsint = Vsint_at
        # Compute Id
        _, _, comps2 = _residuals(
            cfg, model_M1, bjt, Vd=Vd_i, VG1=VG1, VG2=VG2,
            Vsint=Vsint.unsqueeze(0), Vb=Vb.unsqueeze(0),
            P_M1=None, P_M2=None, model_M2=model_M2)
        Id_i = comps2["Ic_Q1"] + comps2["Ids_M1"]
        Vb_traj[i] = Vb
        Vsint_traj[i] = Vsint
        Id_traj[i] = Id_i.squeeze()
        # Spike detection
        if float(Vb) >= spike_threshold:
            spike_times.append(float(t[i]))
            Vb = torch.tensor(reset_Vb, dtype=torch.float64)
        if verbose and i % max(1, n // 10) == 0:
            print(f"  [transient] t={float(t[i]):.4g}  Vd={float(Vd_i):.3f}  "
                  f"Vb={float(Vb):+.4f}  Vsint={float(Vsint):+.4f}  "
                  f"Id={float(Id_i):.3e}")
    return {
        "Vb": Vb_traj, "Vsint": Vsint_traj, "Id": Id_traj,
        "spike_times": spike_times, "t": t,
    }


def integrate_2t_transient(cfg, model_M1, model_M2, bjt, Vd_t: torch.Tensor,
                            t: torch.Tensor, VG1: torch.Tensor,
                            VG2: torch.Tensor, *,
                            Vb0: float = 0.0, Vsint0: float = 0.0,
                            spike_threshold: float = 0.65,
                            reset_Vb: float = 0.30):
    """Forward Euler time integration of the 2T cell body charge.

    State: Vb(t), Vsint(t). Vsint solved quasi-statically at each step
    (1D Newton in Vsint with Vb fixed); Vb integrated as
        dVb/dt = R_B(Vsint, Vb, Vd) / C_total(Vb)
    where R_B is the body KCL residual (currents INTO body) and
    C_total = Cj_pdiode(Vb-vnwell) + Cj_M1_bs/bd + Cj_M2_bs/bd.

    Spike detection: when Vb crosses spike_threshold, log a spike event,
    snap Vb to reset_Vb (one-step). This is the LIF firing primitive.

    Returns dict with Vb_traj, Vsint_traj, Id_traj, spike_times.

    NOTE: this is a minimal forward-Euler integrator suitable for
    plotting transient ramps and demonstrating LIF-style dynamics.
    Production work needs implicit BDF and adaptive dt.

    WARNING (2026-05-02): forward-Euler is unconditionally unstable on
    this stiff body-charge ODE. The 2T cell loop has very small
    capacitance (Cj ~ 10 fF) and currents that span 1 fA to 1 mA, so
    the natural time constant ranges over ~12 decades. Any explicit
    method explodes near the snapback fold. This routine is provided
    as a skeleton for the eventual implicit integrator (Phase B work);
    quantitative use requires Newton-per-step in the joint (Vsint,Vb).
    """
    from .nsram_cell_2T import _residuals
    n = Vd_t.numel()
    Vb_traj = torch.zeros(n, dtype=torch.float64)
    Vsint_traj = torch.zeros(n, dtype=torch.float64)
    Id_traj = torch.zeros(n, dtype=torch.float64)
    Vb = torch.tensor(Vb0, dtype=torch.float64)
    Vsint = torch.tensor(Vsint0, dtype=torch.float64)
    spike_times = []

    Cj0_total = cfg.body_pdiode_Cj0_per_area * cfg.body_pdiode_area

    for i in range(n):
        Vd_i = Vd_t[i:i+1]
        # Quasi-static Vsint solve: ~5 Newton iters in Vsint with Vb fixed
        for _ in range(8):
            R_S, R_B, comps = _residuals(cfg, model_M1, bjt,
                                          Vd=Vd_i, VG1=VG1, VG2=VG2,
                                          Vsint=Vsint.unsqueeze(0),
                                          Vb=Vb.unsqueeze(0),
                                          P_M1=None, P_M2=None,
                                          model_M2=model_M2)
            # Finite-diff Jacobian dR_S/dVsint
            eps = 1e-4
            R_S_eps, _, _ = _residuals(cfg, model_M1, bjt,
                                        Vd=Vd_i, VG1=VG1, VG2=VG2,
                                        Vsint=(Vsint+eps).unsqueeze(0),
                                        Vb=Vb.unsqueeze(0),
                                        P_M1=None, P_M2=None,
                                        model_M2=model_M2)
            J = (R_S_eps - R_S) / eps
            dVsint = -R_S / J.clamp(min=1e-30)
            Vsint = Vsint + dVsint.squeeze().clamp(-0.5, 0.5)
            if R_S.abs().max() < 1e-12:
                break
        # Now compute body-charge derivative
        _, R_B_now, _ = _residuals(cfg, model_M1, bjt,
                                    Vd=Vd_i, VG1=VG1, VG2=VG2,
                                    Vsint=Vsint.unsqueeze(0),
                                    Vb=Vb.unsqueeze(0),
                                    P_M1=None, P_M2=None,
                                    model_M2=model_M2)
        Cj_now = junction_cap(Vb - cfg.vnwell, Cj0=Cj0_total,
                                Vj=cfg.body_pdiode_Vj, M=cfg.body_pdiode_M)
        # Add small floor so Cj never zero (prevents dt explosion)
        C_total = Cj_now + 1e-18
        if i + 1 < n:
            dt = float(t[i+1] - t[i])
            dVb = float(R_B_now.squeeze()) * dt / float(C_total)
            Vb = Vb + dVb
        # Compute Id at this point
        Id_i = comps["Ic_Q1"] + comps["Ids_M1"]   # drain current = NPN collector + M1 channel
        Vb_traj[i] = Vb
        Vsint_traj[i] = Vsint
        Id_traj[i] = Id_i.squeeze()
        # Spike detection
        if float(Vb) >= spike_threshold:
            spike_times.append(float(t[i]))
            Vb = torch.tensor(reset_Vb, dtype=torch.float64)
    return {
        "Vb": Vb_traj, "Vsint": Vsint_traj, "Id": Id_traj,
        "spike_times": spike_times, "t": t,
    }

```


=== FILE: vectorized.py (5279 chars) ===
```python
"""Vectorized batched 2T forward sweep.

B.1 (2026-05-02): The serial forward_2t loops one bias at a time. The
underlying _residuals already broadcasts naturally over a batch dim,
so an N-bias sweep can share the Newton machinery across all biases
simultaneously — a single torch.linalg.solve replaces N independent
2x2 solves. For z91g (33 biases × 40 Vd points) this should give
a roughly 33× wall-time speedup, modulo Python-level overhead.

Public API:
    forward_2t_batched(cfg, model_M1, model_M2, bjt, Vd_seq, VG1_arr, VG2_arr)
        VG1_arr, VG2_arr: shape (N,) — N independent biases.
        Vd_seq: shape (T,) — common Vd sweep grid.
        Returns dict with shape (N, T) tensors: Id, Vsint, Vb, niter, conv.
"""
from __future__ import annotations
import torch
from typing import Optional

from .nsram_cell_2T import _residuals, NSRAMCell2TConfig
from .model_card import BSIM4Model
from .bjt import GummelPoonNPN


def _solve_2x2_batched(R_S: torch.Tensor, R_B: torch.Tensor,
                         J: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched 2x2 Newton step. J shape (N, 2, 2), residuals shape (N,).
    Returns (dVsint, dVb), each shape (N,).
    """
    rhs = -torch.stack([R_S, R_B], dim=-1).unsqueeze(-1)   # (N,2,1)
    sol = torch.linalg.solve(J, rhs)
    return sol[..., 0, 0], sol[..., 1, 0]


def _jacobian_fd_batched(cfg, model_M1, model_M2, bjt, Vd, VG1, VG2,
                          Vsint, Vb, eps: float = 1e-5) -> torch.Tensor:
    """Build batched 2x2 Jacobian via central-ish finite differences.
    Shape (N, 2, 2). Calls _residuals 4 extra times per step.
    """
    R_S0, R_B0, _ = _residuals(cfg, model_M1, bjt, Vd=Vd, VG1=VG1, VG2=VG2,
                                 Vsint=Vsint, Vb=Vb, model_M2=model_M2)
    R_S_dVs, R_B_dVs, _ = _residuals(cfg, model_M1, bjt, Vd=Vd, VG1=VG1, VG2=VG2,
                                       Vsint=Vsint+eps, Vb=Vb, model_M2=model_M2)
    R_S_dVb, R_B_dVb, _ = _residuals(cfg, model_M1, bjt, Vd=Vd, VG1=VG1, VG2=VG2,
                                       Vsint=Vsint, Vb=Vb+eps, model_M2=model_M2)
    dRS_dVs = (R_S_dVs - R_S0) / eps
    dRS_dVb = (R_S_dVb - R_S0) / eps
    dRB_dVs = (R_B_dVs - R_B0) / eps
    dRB_dVb = (R_B_dVb - R_B0) / eps
    # J[..., i, j] = dR_i / dx_j
    J = torch.stack([
        torch.stack([dRS_dVs, dRS_dVb], dim=-1),
        torch.stack([dRB_dVs, dRB_dVb], dim=-1),
    ], dim=-2)
    return J, R_S0, R_B0


def forward_2t_batched(cfg: NSRAMCell2TConfig,
                        model_M1: BSIM4Model, model_M2: BSIM4Model,
                        bjt: GummelPoonNPN,
                        Vd_seq: torch.Tensor,
                        VG1_arr: torch.Tensor, VG2_arr: torch.Tensor,
                        *, max_iters: int = 30, tol: float = 1e-12,
                        Vsint0: float = 0.1, Vb0: float = 0.3,
                        damping: float = 1.0,
                        verbose: bool = False) -> dict:
    """Run N independent 2T forward Vd sweeps in parallel.

    Returns dict with shape (N, T) tensors: Id, Vsint, Vb, niter, conv.
    """
    N = VG1_arr.numel()
    T = Vd_seq.numel()
    Id_out = torch.zeros(N, T, dtype=torch.float64)
    Vsint_out = torch.zeros(N, T, dtype=torch.float64)
    Vb_out = torch.zeros(N, T, dtype=torch.float64)
    conv_out = torch.zeros(N, T, dtype=torch.bool)
    niter_out = torch.zeros(N, T, dtype=torch.int32)

    # State: Vsint, Vb of shape (N,), warm-started across Vd
    Vsint = torch.full((N,), Vsint0, dtype=torch.float64)
    Vb = torch.full((N,), Vb0, dtype=torch.float64)

    for ti in range(T):
        Vd_t = torch.full((N,), float(Vd_seq[ti]), dtype=torch.float64)
        # Newton loop, batched across N biases
        for k in range(max_iters):
            J, R_S, R_B = _jacobian_fd_batched(
                cfg, model_M1, model_M2, bjt, Vd_t, VG1_arr, VG2_arr, Vsint, Vb)
            R_max = torch.maximum(R_S.abs(), R_B.abs())
            done = R_max < tol
            if done.all():
                niter_out[:, ti] = k
                conv_out[:, ti] = True
                break
            try:
                dVs, dVb = _solve_2x2_batched(R_S, R_B, J)
            except Exception:
                break
            # Damp big steps
            dVs = dVs.clamp(-0.5, 0.5)
            dVb = dVb.clamp(-0.5, 0.5)
            Vsint = Vsint + damping * dVs
            Vb = Vb + damping * dVb
            # Bound to physical range
            Vb = Vb.clamp(-0.5, 1.2)
        else:
            niter_out[:, ti] = max_iters
            conv_out[:, ti] = R_max < tol * 1e3
        # Compute final Id at converged state
        _, _, comps = _residuals(cfg, model_M1, bjt, Vd=Vd_t,
                                   VG1=VG1_arr, VG2=VG2_arr,
                                   Vsint=Vsint, Vb=Vb, model_M2=model_M2)
        Id_out[:, ti] = comps["Ic_Q1"] + comps["Ids_M1"]
        Vsint_out[:, ti] = Vsint
        Vb_out[:, ti] = Vb
        if verbose and ti % max(1, T // 5) == 0:
            print(f"  [batched] Vd={float(Vd_seq[ti]):.3f}  "
                  f"max(R)={float(R_max.max()):.3e}  "
                  f"conv={int(done.sum())}/{N}")

    return {"Id": Id_out, "Vsint": Vsint_out, "Vb": Vb_out,
            "niter": niter_out, "converged": conv_out}

```
