# A8 — Visual Scan: structural physics gaps in BSIM4 NS-RAM 2T fit

Date: 2026-05-01. Scope: every image, slide, schematic, email, transcript line in the
sources flagged in the A8 brief, looking for structural mechanisms missing from our port.

------------------------------------------------------------------------
## 1. Sebas's BSIMfitsBA deck — slide-by-slide
File: `/home/ikaros/nsram_info/2026-04-30 BSIMfitsBA/2026-04-29 NS-RAM I-V BA plots.pptx`

### Slide 1 — three-panel Id(Vd) families, Origin95 OLE charts, raster fallback in `media_image1.emf` (and rendered in `media_image4.png`)
Verbatim text annotations:
- "VG1 = 0.6 V", "VG1 = 0.4 V", "VG1 = 0.2 V"
- "Symbols = measurements", "Lines = simulations"
- "VG2 = 0 V to 0.5 V, step 0.05 V" (VG1=0.6 panel)
- "VG2 = 0 V to 0.3 V, step 0.05 V" (VG1=0.4 panel)
- "VG2 = -0.2 V to 0.1 V, step 0.05 V" (VG1=0.2 panel)
- Schematic in left panel: M1 (top, NMOS, gate=VG1, drain=VD), M2 (bottom, NMOS,
  gate=VG2) with **arrow into the body node Vb of M1**. Grounded sources. NOT a series stack.

`media_image4.png` (the rendered fallback for slide 1) shows the actual measured curves:
- VG1=0.6, VG2=0…0.5: Id climbs from ~1e-9 to **3e-4** with a sharp knee at Vd≈1.5 V;
  current jumps **5 decades in <0.5 V**.
- VG1=0.4: knee at Vd≈1.7 V, jump 1e-10 → ~1e-5.
- VG1=0.2: knee at Vd≈1.9 V, jump 1e-11 → ~1e-7.
- The knee position **shifts right and down** as VG1 decreases. This is the snap shape
  we are missing.

### Slide 2 — single overlay with thick=measurements, thin=simulations
- "Thick = measurements", "Thin = simulations"
- Curves shown: VG1=0.20/VG2=0.0, VG1=0.40/VG2=0.25, VG1=0.60/VG2=0.35
- Sebas's BSIM simulations REPRODUCE the snap (his thin lines track his thick lines well
  on log-Y; small misfit on the foot of the lower branch only).
  → **Definitive: the snap is real and is fittable in BSIM4. Not a measurement artifact.**

### Slide 3 — parameter scaling plots (4 PNG charts)
- `slide_03_img1.png`: NFACTOR (M2) vs VG2; values 1.25 → 12.2; SLOPE depends on VG1
  (red VG1=0.2 highest at 6.2-12.2; black VG1=0.6 lowest at 1.25-6).
  **NFACTOR varies almost 10×** — we used a polynomial; Sebas's data shows a near-linear
  monotone descent that is well captured by polynomial(VG2) per VG1.
- `slide_03_img2.png`: K1 (M1) vs VG1 — three points only: (0.2, 0.558), (0.4, 0.538),
  (0.6, 0.418). **K1 drops 25 % between VG1=0.4 and VG1=0.6** (knee in the
  parameter itself). PWL not polynomial — a regime change near VG1=0.5.
- `slide_03_img3.png`: ETAB (M1) vs VG2 — three monotone branches. Red VG1=0.2 between
  0.8-1.1 (rising). Blue VG1=0.4 between 1.6-1.9 (falling). Black VG1=0.6 nearly flat at
  ~2.5 with abrupt drop to ~2.1 at VG2=0.4. **Three different ETAB regimes per VG1.**
- `slide_03_img4.png`: BETA0 (M1) vs VG2 — Red VG1=0.2 ramps 10.75→14 (linear in VG2).
  Blue VG1=0.4 flat at 19. Black VG1=0.6 flat at 20. **BETA0 saturates above VG1=0.4.**

------------------------------------------------------------------------
## 2. Mario/Seb status deck — slide-by-slide
File: `/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/docs/NSRAM 20260429 Mario Seb.pptx`

- Slides 1–5: our (Eric/Robert) status, BSIM4 port description, network roadmap. No new
  physics info — but Slide 4 says "Phenomenological cell_fast — known to be 20–40 % off".
- Slide 6 (`slide_06_img1.png`): title slide "Status Update 30 Apr 2026 / Robert Luciani".
- **Slide 7 (`slide_07_img1.png`) — Robert's Julia DEQ plot. KEY**:
  - Title: "Status — differentiable single-cell substrate"
  - Bullet text: "SPICE deck → MTK Cell2T model → static-RH5 canonical Newton roots →
    DEQ inner solver", "DEQ trained on I-V family, ~20K root samples", "residual ≤ 1e-6
    across operating range", "Pure Julia, fully differentiable, numerically accurate
    (IEEE 754)".
  - The plot shown is an Id(Vd) family climbing the full snap from ~1e-13 to ~1e-3
    across colored VG2 lines. **Robert's DEQ already captures the snap** (different
    architecture from our Newton-DC port).
- Slide 8 (`slide_08_img1.png`): "Speed is important" — projection chart. Not physics.
- Slide 9 (`slide_09_img1.png`) — **Mario/Seb critical input "Exploration path"**:
  - "7 parameter fit / BBO 10K iters loss=0.155 / 8.2× from defaults"
  - "1 of 7 parameters hit bounds on reasonable ranges"
  - **"LDE physics missing"**  ← *layout-dependent effects, called out explicitly*
  - Table:
    - alpha0_M1 = 2.78e-13 m/V (10× lower than default)
    - alpha0_M2 = 7.27e-11 m/V (typical)
    - **beta0_M1 = 10.5 V (3× lower than default)**
    - beta0_M2 = 1.94 V (15× lower than default)  ← *very small, unusual*
    - **etab_M1 ≈ +0.31 1/V**
    - K1_M1 = 1.94 V^0.5
    - K1_M2 = 1.98 V^0.5
    - **Nfactor_M2 = 3.0+  (UPPER BOUND hit)**
- Slide 10 (`slide_10_img1.png`) — "Future input":
  - "Adding parameters – true BSIM4.jl / KernelAbstractions.jl / Mooncake / Only subset
    that matters for now (how many?) / ngspice unit testing"
  - **"Boundary I-V coverage / High VG2 + high Id has few points"**
  - **"Pulsed-retention measurements"** — Sebas planned to send pulsed data
  - Plot: "I-V family coverage — existing sweeps and gap" with shaded "coverage gap
    (high VG2)" at VG2 ≥ 0.4 across all VG1. **The high-Id snap region is exactly the
    region we have least data for.**
- Slide 11/12: roadmap, local compute. No new physics.
- Slide 13: "Summary / Differentiable single cell substrate in Julia / BSIM4 python
  implementation".

### Slide 3 inset images we already produced (`mario_seb_slides/slide_03_img1.png`, `slide_03_img2.png`)
- `slide_03_img1.png`: **OUR** z88 v10 fit (Stage 1 fitted, loss=0.79, Stage 2-4 still
  pending). Three panels VG1=0.2 (7 curves), 0.4 (11 curves), 0.6 (15 curves). Caption:
  **"li/BJT OFF — only GIDL fits the off-state"**. Measurements (dots) climb 3-7 decades
  through Vd=1.0–2.0 V; OUR LINES are flat tongues stuck at 1e-7…1e-12. *Exactly the
  bug A8 describes.*
- `slide_03_img2.png`: z94 pseudo-arclength continuation test at VG1=0.6/VG2=0.4. Two
  panels α0=2e-2/β0=15 (left, green) and α0=8e-2/β0=12 (right, red). Annotation: "does
  arclength produce a clean knee where Newton+homotopy hops between roots?" The
  arclength path (gray dotted) **plunges to ~1e-15 in the snap region** — meaning the
  arclength solver finds the *lower* unstable branch of the fold but our Newton is
  pinned to the lower stable branch. **The model is bistable; we are converging to the
  wrong stable root.**

------------------------------------------------------------------------
## 3. Other docs

- `/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/docs/Image 2026-02-28 at 13.14.jpeg`:
  not physics — terminal screenshot of the FEEL_Functionally_Embodied_Emergent_Learning
  zip and a bash log. Skip.

------------------------------------------------------------------------
## 4. Schematic & model-card folder
`/home/ikaros/nsram_info/schematic&modelCards/`

### `2tnsram_simple.asc` (LTSpice)
Topology *exactly* as Sebas drew on slide 1:
- M1 = nmos4 (length=Ln=0.18 µm, width=Wn=0.36 µm). Drain=Din, source=Sint, body=B,
  gate=G (port G).
- M2 = nmos4 with **`l='Ln*10'`** (so 1.8 µm, 10× longer channel), w=Wn. Drain=B (the
  *body* of M1), gate=G2. Source=GND. **M2's drain is M1's body** — i.e. M2 acts as a
  **floating-bulk discharge transistor**, controlling the bulk voltage of M1.
- C1 = `CBpar` = **1 fF** between B and GND with Rser=1m. Sets the body-node RC.
- Q1 = `parasiticBJT` (NPN) with collector tied near drain area of M1, emitter near
  source. Used as the bipolar firing path.
- Includes: `.inc PTM130bulkNSRAM.txt`, `.inc parasiticBJT.txt`
- **`.op 0` directive** — this is LTSpice's "DC operating-point at t=0" but in
  combination with the bistable circuit it can mean Sebas's actual sims may be `.tran`
  with UIC pulling the upper branch. Worth confirming.

### `parasiticBJT.txt`
NPN, **bf=10000** (huge β), Va=100 V, va=5e-9 A, **vje=0.7**, Cje=0.7 fF, Cjc=1 fF,
**tf=25 ps**, tr=20 ps, **vtf=7 V, xtf=2** (high-injection model used). The β=10⁴ is
unusually high — this is a *complementary* lateral-NPN that fires the body once
impact-ionization charges it. *We have no equivalent device in our port; we modelled
impact-ionization as a current source into the body node only.*

### `PTM130bulkNSRAM.txt`
Predictive-Technology-Model BSIM3v4-style (Level 14), **NMOS only**. Vth0=0.54,
**alpha0=7.83756e-5, beta0=18, k1=0.63825, k2=-0.070435, etab=-0.086777, nfactor=Nparam
(=1.58)**. NOT BSIM4. *Sebas's later cards in `2026-04-30 BSIMfitsBA/` are full BSIM4
v4.5.*

### `2026-04-30 BSIMfitsBA/130bulkNSRAM(M2).txt`
BSIM4 v4.5, level 14, **rbodymod=0** (no internal body network), **diomod=1**,
diodes shorted to body, **alpha0=7.83756e-5**, **beta0=18**, **lalpha0=-9.84e-12**,
**lbeta0=-9.5e-7**, **pscbe1=5.331e8, pscbe2=1e-5**, **agidl=1.99e-8, bgidl=1.624e9**,
**xn=3 (1/temperature exponent for impact ion.)**. Standard Pre-BSIM4 fit.

### `2026-04-30 BSIMfitsBA/130DNWFB(M1).txt`
**Identical to above EXCEPT**:
- model name: **NMOSdnwfb** (Deep N-Well Floating Body)
- **k1 = 0.53825** (was 0.63825, lower body-effect coefficient)
- **etab = +1.8** (was −0.087! sign flip and 20× bigger). Body-bias coefficient on
  subthreshold slope is wholly different.
- **beta0 = 19** (vs 18).

→ **Sebas uses TWO DIFFERENT MODEL CARDS for M1 (DNWFB) and M2 (bulk).** They share
geometry but have different K1, ETAB, BETA0. Our port uses one card with shared
parameters.

### `2026-04-30 BSIMfitsBA/2Tcell_BSIM_param_DC.csv`
Per-curve parameters extracted by Sebas. 33 rows. Columns:
`VG1, VG2, trise, ETAB, K1, ALPHA0, BETA0, NFACTOR, mbjt, IS, area`

Key observations:
- **`trise` = 9.04 to 12.98** (likely seconds — confirming his sweeps are time-resolved
  ramps, not steady-state DC).
- **`mbjt` jumps from 0.001 (VG1=0.2) to 1.000 (VG1=0.4 and 0.6)** — i.e. the parasitic
  BJT *contribution multiplier* is 1000× larger above VG1=0.4. **A regime switch.**
  This is the parameter that turns on the snap.
- **`ETAB` family**: 0.8…1.1 for VG1=0.2; 1.6…1.9 for VG1=0.4; 2.1…2.5 for VG1=0.6.
- **`K1` jumps** 0.55825 → 0.53825 → 0.41825 — a *step* not a smooth poly.
- **`BETA0`**: 10.75…14 (VG1=0.2), pinned at 19 (VG1=0.4), pinned at 20 (VG1=0.6).
- `IS = 5e-9, area = 1e-6` constant (BJT saturation current and area for the parasitic).
- `ALPHA0 = 7.842e-5` constant.
- `NaN` rows at VG1=0.4/VG2=−0.2…−0.05 and VG1=0.6/VG2=−0.2…−0.05 → those bias points
  do NOT fire (no snap to fit).

------------------------------------------------------------------------
## 5. Slow I-Vs subfolder, raw CSVs

`/home/ikaros/nsram_info/Slow I-Vs 2vHCa-2@VG2 VG1 vnwell=2 SRavg=0/` and the mirrored
copy at `/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/data/sebas_2026_04_22/`.
- No README/notes files. Just CSVs.
- Filename format: `StandardIV_HH_2vHCa-2_VG2=<x>_VG=<y>(1)_<HH-MM-SSPM>.csv`. The
  trailing timestamp is the lab clock.
- Folder names: "vnwell=2" → **deep N-well bias = 2 V** is fixed across the campaign.
- "SRavg=0" → **Sweep-Rate-average = 0** (single sweep, no averaging). Average ramp
  is 0.05 V step per ~0.2 s = **0.25 V/s** (matches Sebas's email: "0.2 V/s").
- CSV columns: `vdata,idata,tdata,Var4,vfixgdata,ifixdata`.
  - `tdata` is **time in seconds** (column 3) — every point has a real timestamp.
  - `vfixgdata,ifixdata` look like the **fixed-VG2 voltage and current** monitors —
    *Sebas is recording two channels simultaneously: Id(Vd,t) and the second-gate
    leakage*. We have not used the second channel. Could constrain VG2 model.
  - `Var4` looks like a duty/quality flag (0–1).

------------------------------------------------------------------------
## 6. Email keyword hits (verbatim with surrounding context)

### a. Sebas, line 212 (Apr 17): different SPICE tool & new modelling approach
> "I'm working with a different SPICE tool and foundry-provided models and I'm focusing
> on adapting foundry models of body bias and impact ionization (in BSIM4) to fit the
> floating body behaviour of our 2T cells. This renders a more standard approach for us
> circuit designers."

### b. Sebas, line 214 (Apr 17): **dropped avalanche diode**, only complementary BJT
> "I've dropped the avalanche diode models (very annoying for convergence) that fire the
> bipolar parasitic effect in LTSpice, and I'm only including a **complementary bipolar
> current** to capture the full swing of the firing mechanism. Fits are looking good,
> but I'm still working on **polynomial dependence of model parameters with tuning
> voltages (VG1, VG2)** and **layout dependent effect on transistor models** to capture
> the experimental behaviour."

### c. Sebas, line 308 (date later): **DC sweep rate 0.2 V/s; pulsed coming**
> "I-V curves in CSV files, at an average **sweep rate of 0.2 volts per second**, 3
> different VG1 (0.2, 0.4, 0.6) and multiple VG2 (value enclosed within each filename).
> Let me know if this works for you. **We are generating additional data now (multiple
> ramp rates for more data on dynamics, pulsed dynamics).**"

### d. Sebas, line 310: foundry NDA, PTM is starting point only
> "A model card for the bipolar device within the 2T cell schematic and a set of
> parameters around the impact ionization / body effect set by me as starting point for
> my SPICE fittings. Sadly, the foundry's full model card cannot be shared without
> infringing NDAs, but the 130 nm (current working node) PTM model I'm attaching is a
> good starting point."

### e. Eric (line 141, Mar 25, **own admission**)
> "things like deep N-well high-voltage operation, **sweep-rate dependent I-V
> hysteresis**, polynomial bulk current models as an alternative to the exponential
> fit, E/I input neuron configurations, and frequency-coded spike encoding. These are
> all stubbed out with placeholder parameters."  — *we explicitly recognized
> sweep-rate hysteresis and stubbed it.*

### f. Eric (line 137, Mar 25): "**Chynoweth avalanche, SRH charge trapping,
> temperature-dependent BVpar**" — these are in our older `nsram` package but the
> question is whether they made it into the differentiable BSIM4 port we are now fitting.

### g. Sebas, line 386 (closing line of Apr 23 email)
> "I still have the feeling that I missed a question or two during our meeting"
> — confirms there is unspoken physics he didn't mention.

### h. No hits at all in emails for: `self-heating`, `SHMOD`, `kink`, `latch`,
> `bistable`, `BVcjs`, `RBODYMOD`, `BSIMSOI`, `STI`. The terms `snap`, `transient`,
> `tran`, `UIC`, `back-gate`, `history effect` also do not appear. Sebas never used
> "self-heating" — but his slide 12.27 explicitly says **"semi-empirical bulk current
> model with PWL(VG2) coefficients"** which functionally adds a piecewise term we lack.

------------------------------------------------------------------------
## 7. Zoom screenshots — physics-relevant content
Folder: `/home/ikaros/nsram_info/Zoom/` (31 images, 2026-03-20 and 2026-04-30 sessions).

### 2026-03-20 session (Mario presenting)
- `12.05 (1).jpeg` / `12.05.jpeg`: AI-on-edge market context (Slide 5/41). Skip.
- `12.06.jpeg`: Spiking neural networks intro, LIF equation. Skip.
- `12.07.jpeg`: Memristor-based electronic neuron (single device, AF integration density).
  Shows hysteresis V-I loop with HRS/LRS branches. Skip — not 2T.
- `12.07 (1).jpeg`: Neuron implementation comparison (CMOS variants). Skip.
- `12.07 (2).jpeg`: Memristor 25 µm² threshold-type AgN/hBN/Au I-V cycling. Skip.
- `12.08.jpeg`: Die-to-die variability of NS-RAM (Pazos 2023 Nature 643). Skip.
- `12.08 (1).jpeg`: Memristor neuron deposition steps. Skip.
- `12.08 (2).jpeg`: NSRAM firing schematic, **VPDSS / VPDSSL labels**, "**Thick oxide
  devices accommodate the now higher drain voltage for NS-RAM**". *Confirms the cell
  uses thick-oxide 3.3 V I/O transistors, NOT core-voltage 130 nm.*
- `12.09.jpeg`: NSRAM Simple LIF in Brian2. Phenomenological. Skip.
- `12.26.jpeg`: SNN classification matrix (Brian2). Skip.
- `12.27.jpeg`: **"Semi-empirical model fits for impact ionization bulk currents"**:
  - "Iion = Iexp + Ipnw"
  - "Iexp = α(Vd − a)·exp(−b/(Vd − a))" — standard Chynoweth.
  - **"Ipnw = c·(Vd − e)^d if Vd > e, else 0"** — *piecewise power-law extra term*.
  - "Each parameter is extracted for different VG…"
  - **"with polynomial poly(VG)"** — coefficient bias-dependence
  - **Annotated red dot**: "where {a, b, c, d, e} = PWL(VG)".
  - This is the **"complementary bipolar current"** Sebas referenced in the email.
- `12.27 (1).jpeg`: **"Measurements vs. SPICE simulation using semi-empirical bulk
  currents"**, two-panel:
  - Left = bulk current Ib alone (clear snap from 1e-9 to 1e-3 at Vd≈1.5 V).
  - Right = total drain current Id (snap is identical because BJT amplifies).
  - Note: "Excellent fit to experimental data with VG1=0.5V"
  - Note: "Dependencies on body voltage Vb has not been included in this revision; in
    cases of weakening the body to account for this effect using measured data with
    Vb=1 MΩ" → meaning he **probes Vb directly with a high-Z meter** during sweep.
  - Sweep VG2 = 0.40, 0.45, 0.50, 0.55 V (slight family).
- `12.29.jpeg`: 27-cell array implementation in 130nm triple-well CMOS. Skip — not the
  fit.
- `12.29 (1).jpeg`: "Standard CMOS deep-Nwell NFET floating body 1T neuron-synapse
  (thick)". Drawing: cross-section showing **floating P-body inside Deep N-well**, with
  VD=2.5V, VG1=0.5V, VG2 floating. **180 nm CMOS**, not 130 nm. **8 µm² per fully-
  contacted 1T neuron.** *This is a different device from the 2T fit data.*
- `12.30.jpeg`: **"Floating body 2T NS-RAM cell under transient VD ramps"** — the
  punchline image:
  - Title literally says **"transient VD ramps"** (NOT DC sweep).
  - Subtitle: "Good overall agreement of the VG1 and VG2 dependence."
  - **"Improves model agreement: fitting is possible with deterministic parameter search
    approach vs ML-driven parameter search."**
  - Plot Id(VD) over 0–2.5 V, log-y from 1e-9 to 1e-3. ~10 colored solid curves
    (measurements) and ~10 dashed (simulations). Family parameter is **VG2 increasing**.
    Snap location **moves left as VG2 increases** (more bulk-bias → earlier firing).
- `12.33.jpeg`: 2T NS-RAM spiking neuron cell schematic with **VD=2.5 V, VG1=0.6 V,
  VG2=floating**, "Outstanding firing range exceeding 10⁴× thanks of soft power
  consumption". *Confirms 10⁴ snap is the design target.*
- `12.39.jpeg`: NS-RAM implementation in standard triple-well CMOS (130nm). Cross-
  section showing **Triple-well + P-channel floating well**. Skip — same as above.

### 2026-04-22 session (Sebas presenting)
- `19.57.jpeg`: Eric's terminal log, NUS Ariba portal screenshot. Skip.

### 2026-04-30 session
- `13.23.jpeg`: Sebas slide 1 of BSIMfitsBA, raster of three-panel I-V family with
  schematic inset. Same as `media_image4.png`.
- `13.24.jpeg`: Sebas slide 3 of BSIMfitsBA — parameter scaling plots (BETA0, ETAB, K1,
  NFACTOR vs VG1/VG2). Same as our `slide_03_imgN.png`.
- `13.25.jpeg`: **VERY IMPORTANT — pulsed-IV measurement screenshot**:
  - Top panel: Vd(t) blue triangular pulse 0–2 V at ~MHz period (~0.4 µs), Id(t) red
    pulses peaking at ~5 mA. **Period 0.4 µs = 2.5 MHz pulsed measurement.**
  - Bottom-left panel: Id(Vd) hysteresis loop, **noisy "spaghetti" of many traces** with
    a single average smooth curve climbing 1e-9 → 1e-4 with snap at Vd≈1.5 V.
  - Bottom-right panel: same hysteresis, slightly different bias.
  - **Conclusion: Sebas has BOTH slow-DC AND pulsed-MHz data. The CSVs we have are slow
    DC. The pulsed data has not been shared.**
- `13.28.jpeg`: Sebas slide 2 of BSIMfitsBA, single-panel overlay. **Sebas's BSIM
  simulations DO reproduce the snap — thin lines track the thick measurement lines on
  log-y across all three (VG1,VG2) conditions.**
- `13.31.jpeg` / `13.31 (1).jpeg`: Mario presenting "Simple NS-RAM cell with integration
  (self-reset)": **Spiking pulses Vd(t)** at multiple periods (50 ms, 80 ms, 200 ms,
  500 ms) with annotation **"Energy consumption considerations: VD line at 5.2 V per
  pulse"** and "Spiking frequency is configurable with 10× change in operating window."
  → cell self-resets via the bistable snapback creating relaxation oscillation.
- `13.33.jpeg`: NSRAM firing with linear excitatory and inhibitory inputs — circuit
  schematic with **VPDSS, VPDSSL** annotations, "Thick oxide devices accommodate the
  now higher drain voltage for NS-RAM". Same point as 12.08(2): **3.3 V thick-ox
  devices**, foundry-process-specific.
- `13.46.jpeg`: AI-on-edge market slide. Skip.
- `13.47.jpeg`: SNN intro slide (Mario). Skip.
- `13.50.jpeg`: NS-RAM array fabrication morphology (resistive switching layer dep.).
  Skip.
- `13.53.jpeg` / `13.53 (1).jpeg`: NUS proposal screenshots. Skip.

------------------------------------------------------------------------
## 8. Closed-caption transcript
File: `Zoom/2026-04-30 13.03.27 Zoom NSRAM/meeting_saved_closed_caption.txt`. Heavy
Swedish auto-translate noise; readable nuggets:
- 13:08:43 (Robert): "On your sweeps that you choose to serve … this very intentional
  to me. Three different VG1 sweeps. … if you have any comment on that … one is
  Dynamics. If you gonna be doing that sometime in near future for free should work
  without it anything." → **Robert flags dynamics as the unmodelled axis.**
- 13:34:53 (Sebas): "got even to because them dynamic parties going to be affected by
  capacitance" → **Sebas confirms capacitance and dynamics matter.**
- 13:35:05 (Mario): "Disconnected … the trap" → likely *trap charging* or *trapezoidal*
  but garbled.

------------------------------------------------------------------------
## 9. THINGS WE MISSED ENTIRELY — ranked

| Rank | Missed mechanism | Evidence | Plausibility it explains the flat-tongue gap |
|------|---|---|---|
| **1** | **Parasitic NPN with β=10000 firing in parallel with the channel** (Sebas's `parasiticBJT.txt` + LTSpice schematic + email "complementary bipolar current"). Our impact-ionization current goes into a body-charge ODE only — we never inject a BJT collector current that **multiplies** by ~10⁴. That is the missing 4–5 decades of Id. | `parasiticBJT.txt`; `2tnsram_simple.asc`; email line 214; Sebas slide 12.27 figure (Iion + Ipnw with PWL); CSV `mbjt` jumps 0.001→1.0 between VG1=0.2 and VG1=0.4. | **Very high.** This alone could close the 5-decade gap. |
| **2** | **Two distinct model cards: M1 = NMOSdnwfb (Deep-N-Well floating body), M2 = bulk NMOS** with different K1, ETAB (sign flip!), BETA0. We use one card. | `130DNWFB(M1).txt` vs `130bulkNSRAM(M2).txt`; CSV K1 column (M1) vs Sebas's slide 9 K1_M2=1.98. | High. Sets the operating point of M1 in the floating-body regime. |
| **3** | **Sweep is a transient ramp at 0.25 V/s, not steady-state DC.** The slide title literally says "transient VD ramps". Floating-body charge fills with `τ=R_body·CBpar` — at 1 fF and ~MΩ effective body-resistance τ≈ ms. Ramping Vd at 0.25 V/s = 4 s/V is slow vs τ (quasi-steady), but the firing **instant** is set by `dVbody/dt = αId − Vbody/τ` integrated over the ramp; our DC Newton solves the algebraic limit and gets the lower stable branch. | `tdata` column in CSVs; slide 12.30 title; transcript Robert/Sebas dynamics comments; CSV `trise` column; `.op 0` in `.asc`. | High. Combined with bistability this explains why arclength finds two roots and DC Newton picks the "wrong" one. |
| **4** | **Bistability / fold bifurcation, requiring continuation not Newton.** Our z94 already showed arclength finds the upper branch but only with α0=8e-2 (not Sebas's 7.84e-5). With a BJT amplifier (item 1), the upper branch becomes the one that physically exists at moderate α0. | `slide_03_img2.png` (arclength plot from us); Sebas slide 12.30 dashed sims also reach the upper branch; the snap is a fold. | High once item 1 is in. |
| **5** | **Layout-Dependent Effects (LDE)** — Sebas's slide 9 says "LDE physics missing" verbatim. BSIM4 §13 SA/SB STI stress + WPE (well proximity) terms. Our port lists them as stubbed. | Slide 9 verbatim; email line 214 "layout dependent effect on transistor models"; Sebas's BSIM cards include `saref=1.04e-6, sbref=1.04e-6, ku0=-2.7e-8, kvth0=9.8e-9`. | Medium. Probably accounts for 0.1–0.3 dec residual after items 1–3 are fixed. |
| **6** | **vnwell = 2 V is a fixed bias, not 0.** We have been assuming Vnwell=0 in the body diode. Sebas's data folder name carries `vnwell=2`. The DNW–P_body junction is **forward-biased at 2 V** if body sits below 2 V — this is the "deep-N-well high-voltage operation" Eric flagged in line 141 but never wired in. | Folder name; `vfixgdata` column may be a vnwell monitor; `bvs=10` in BSIM card (junction breakdown). | Medium-high. Affects the body-diode operating point and thus the BJT base bias. |
| **7** | **Per-curve fit means parameters are bias-dependent surfaces, not constants.** Sebas extracts {ETAB, K1, BETA0, NFACTOR, mbjt} per (VG1,VG2). We've tried polynomial(VG2) — Sebas's K1(VG1) is **piecewise** (kink at VG1=0.5), ETAB is **3 disjoint branches**, mbjt is **binary** (0.001 vs 1.000). Polynomial cannot represent a step. **Use PWL or a ReLU/sigmoid switch.** | CSV `mbjt` column; slide 03 plots; email line 214 "polynomial dependence". | Medium — likely the easiest immediate code change once the BJT is in. |
| **8** | **Pulsed-IV is the regime where the snap is sharpest.** Image 13.25 shows MHz pulsed I-V with ~5 mA peaks. Sebas plans to send pulsed data ("multiple ramp rates for more data on dynamics, pulsed dynamics"). The DC sweeps we have are the LIMIT case where self-heating and trap dynamics smear the snap. | Image 13.25; email line 308. | Medium — irrelevant for fitting the present DC data, but very relevant once pulsed data arrives. |
| **9** | **Vb (body) probe channel** in CSV (`vfixgdata, ifixdata`). Sebas measures the body voltage with a 1 MΩ high-Z meter (slide 12.27(1) caption). We have ignored this channel. Fitting Vb(Vd,t) jointly would constrain α/β/τ uniquely. | CSV header; slide 12.27(1) caption. | Low for closing the 5-dec Id gap, but high for parameter identifiability. |
| **10** | **Self-heating (BSIM4 §10.2 SHMOD) and RBODYMOD body network** — both `0` in Sebas's card. He explicitly disabled them. Probably NOT the answer. We can also leave them off. | `rbodymod = 0` and no `shmod` line in `130bulkNSRAM(M2).txt`. | Very low. Excluded. |

------------------------------------------------------------------------
## 10. Recommended single experiment to close the gap

**Add a parallel parasitic NPN (Sebas's `parasiticBJT.txt` model) to the
differentiable port, with collector tied between drain and body and emitter to source,
and refit α0/β0/Nfactor/ETAB on Sebas's 33 DC curves WITHOUT the polynomial-in-(VG1,VG2)
machinery first.**

Specifically:
1. Implement Gummel-Poon NPN with `bf=10000, IS=5e-9, area=1e-6` (Sebas's values),
   collector current = `IS·area·(exp(VBE/Vt) − exp(VBC/Vt)) · Q1_kirk` with simplified
   Kirk effect via `vtf=7, xtf=2, itf=0.03`. Differentiable with `softplus` floors on
   the diode terms.
2. Replace the body-charge integrator with a **two-equation algebraic system**: solve
   simultaneously Id(Vds, Vgs, Vbs) for the BSIM4 channel **and** Ic(Vbe, Vbc) for the
   NPN, with KCL `Iimpact + Ic_base = Vbody/Rbulk` (steady-state, no `.tran`).
3. Use **pseudo-arclength continuation in Vds** to track BOTH branches of the fold,
   pick the upper branch wherever it exists, fall back to lower otherwise.
4. Fit only **5 globals**: α0_M1, β0_M1, Rbulk, BJT-area_scale, ETAB_M1. Keep all
   other parameters at the values in `130DNWFB(M1).txt` and `130bulkNSRAM(M2).txt`.
5. Compare median log-RMSE of the predicted Id(Vd) family vs measurements over the 33
   curves. **Expectation**: with the BJT in and arclength enabled, log-RMSE should drop
   from 0.79 (current best) to **≤ 0.30** without any per-bias polynomial parameters.
   If it doesn't, the next experiment is to enable `mbjt`-style PWL switch on the BJT
   multiplier between VG1=0.2 and VG1≥0.4 (item 7).

This single experiment isolates whether the missing physics is item 1 (BJT) — the most
likely structural gap — before we revisit polynomial coefficient surfaces (item 7) or
LDE (item 5).
