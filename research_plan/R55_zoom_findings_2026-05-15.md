# R55 — Zoom/Mario/Sebas Deep Dive: Snapback Topology Findings
**Date:** 2026-05-14 (filed 2026-05-15 per filename convention)
**Trigger:** R-43 (iii×Rs), R-47 (subdiode), R-49 (dbd-avalanche) all FAILED to reproduce 2–3 dec snapback fold.
**Scope:** Re-read every text artifact in `nsram/Zoom/`, `docs/Zoom/`, `research_plan/artifacts/Zoom/`, `nsram/proposal_2026_05/`, and cross-checked Zenodo SPICE deck.
**Verdict:** Two genuinely missing structural elements were found that are NOT in current pyport. Three other items confirmed already-modeled.

---

## TOP-5 ACTIONABLE FINDINGS (ranked by physics leverage)

### 1. **D3 + D4: TWO zener avalanche diodes in Zenodo `NeuronSubCirc.asc` — only D4 is approximated, D3 is entirely missing** [HIGHEST]
- **File:** `data/nsram_zenodo/SimulationFiles/SPICE/dev/subcircuit/NeuronSubCirc.asc` (lines 48–58)
- **File:** `data/nsram_zenodo/SimulationFiles/SPICE/dev/Davalanche.txt` (line 6)
- **Cross-ref:** `nsram/proposal_2026_05/01_LOG.md:10017–10032` already flagged this on 2026-05-03, recommended a Phase-B sub-task, but the task was **never executed**.
- **Quote (NeuronSubCirc.asc:48-58):**
  > `SYMBOL zener -144 208 R180 … SYMATTR InstName D3 … SYMATTR Value avalancheD`
  > `SYMBOL zener 240 112 R180 … SYMATTR InstName D4 … SYMATTR Value avalancheD … area=0.01`
- **Quote (Davalanche.txt:6):**
  > `.model avalancheD D(Is=1e-21 Rs=50 Cjo='Cbe' M=0.5 nbv=7 bv='0.9*BVPar' Vj=.75 Isr=0 Ibv=1e-3 Ibvl=1e-3 Nbvl=0.15 Tbv1=-21.3u type=Zener)`
- **Wiring (decoded from FLAG/WIRE):**
  - **D3**: anode at node `B` (body), cathode at node `G` (gate of M4 = V_G1). **Gate-to-body zener avalanche path.**
  - **D4**: between `B` (body) and `Di` (internal drain = M4 drain), area=0.01.
- **Why missing physics:** R-43/R-47/R-49 all attacked the **body→drain** path (D4-equivalent). The **gate→body** path D3 is a completely different injection mechanism — it dumps charge into V_B directly from V_G1 once V_G1 exceeds `0.9·BVPar = 0.9·(3.5−1.5·V_G2)`. At V_G2=0, that threshold is V_G1 ≈ 3.15 V; at V_G2=0.6 the threshold drops to ≈ 2.34 V. **This is exactly the regime where silicon snaps back and our model goes flat.**
- **Topology change implied:**
  - Add `Davalanche` shunt between `G1` and `Vb` node with `BVPar = 3.5 − 1.5·V_G2` (Tsinghua form, per BJTparams.txt:25).
  - This is *separate from* the existing IIMOD/D4 branch; both must coexist.

### 2. **`BJTparams.txt` names the avalanche-shape knobs we treat as constants** [HIGH]
- **File:** `data/nsram_zenodo/SimulationFiles/SPICE/dev/BJTparams.txt` (lines 9–28)
- **Quote (verbatim):**
  > `**** NePar directly affects avalanche relaxation time/voltage (i.e., when the hysteresis happens during the backward)****`
  > `.param NePar 1.5`
  > `**** nbvPar also affects the quickness (slope) and voltage of the avalanche relaxation (keep below 5 for ease of convergence)`
  > `**** If zener BC, between 0 and 1 sets the abruptness of the exponential avalanche component`
  > `***Tsinghua  .param nbvPar '9-(0.1/Vg)*5.5'`
  > `***TSMC     .param nbvPar '9-(0.4/Vg)*0.5'`
  > `***Tsinghua  .param BVPar '3.5-(1.5*Vg)'`
  > `***TSMC     .param BVPar '1.6+(0.4/Vg)'`
- **Why missing physics:** Sebas explicitly defines **both BVPar AND nbvPar as functions of V_G** (Tsinghua: `nbvPar = 9 − 5.5·(0.1/Vg)`). Our pyport hard-codes `nbv = 7` from the diode model card and ignores the `.param` over-ride. nbvPar controls the *exponential abruptness* — small nbvPar produces the **sharp 2–3 dec fold**, large nbvPar smears it out. The TSMC variant inverts the V_G dependence sign — process-dependent.
- **Topology change implied:**
  - Make `nbv` and `bv` of the Davalanche zener evaluated per (V_G1, V_G2) point at sweep time, not held at PTM defaults.
  - Try Tsinghua (`BVPar = 3.5 − 1.5·V_G1`, `nbvPar = 9 − 0.55/V_G1`) first since `PTM130bulkNSRAM.txt` is the matching node card.

### 3. **M3 = BSS145 discrete nMOS body-bias path G2 → B is not in pyport** [HIGH]
- **File:** `data/nsram_zenodo/SimulationFiles/SPICE/dev/subcircuit/NeuronSubCirc.asc` (lines 59–61)
- **Quote:**
  > `SYMBOL nmos 112 208 R0`
  > `SYMATTR InstName M3`
  > `SYMATTR Value BSS145`
- **Wiring (FLAG/WIRE):** M3 drain → B (body), gate → G2, source → GND, body → GND. BSS145 is a discrete NMOS with V_th ≈ 1.5 V — **acts as a V_G2-gated leakage sink from the body**, only active when V_G2 ≳ 1.5 V.
- **Why missing physics:** In Sebas's deck this device **clamps body charge for high V_G2** and **lets body float for low V_G2**. The asymmetric "fold at V_G2≈0, no fold at V_G2≳0.4" pattern in silicon may be M3 turning on. Our pyport has no V_G2-dependent body-leak.
- **Topology change implied:**
  - Add a sub-Vth NMOS (or analytic `I_leak = I0·exp((V_G2−V_th)/n·Vt)`) between V_B and GND with V_th ≈ 1.5 V (BSS145 datasheet).
  - This is **the V_G2-coupling mechanism** we have been hand-fitting via NFACTOR(M2) polynomials.

### 4. **`2tnsram_simple.asc` Q1 (NPN) Bf = 10000, NE = 1.5, NC = 2 — current production deck uses Bf = 9000** [MED]
- **File:** `nsram/Zoom/schematic&modelCards/parasiticBJT.txt` (line 4)
- **Quote (verbatim):**
  > `.model parasiticBJT NPN(is=5E-9 va=100 bf=10000 br=100 nc=2 ikr=100m rc=0.1 vje=0.7 re=0.1 cjc=1e-15 fc=0.5 cje=0.7e-15 ne=1.5 ise=0 tr=20e-12 tf=25e-12 itf=0.03 vtf=7 xtf=2)`
- **Production deck (`ngspice_repro_harness/test_2t_cell_prod.sp:10`):**
  > `.model parasiticBJT NPN(is=1e-9 va=0.55 bf=9000 br=100 nc=2 ikr=100m …)`
- **Discrepancy:** Sebas's canonical card has **IS = 5e-9** (5× our prod), **VA = 100** (180× our 0.55!), **Bf = 10000**. The VA mismatch is **decisive for the Early-effect part of the BJT I-V**, controls how flat the post-snap I-V is and therefore the *width* of the snap fold.
- **Why missing physics:** Setting VA = 0.55 puts the BJT in **immediate quasi-saturation** — the Early kink is at V_CE ≈ 0.55 V, well below the snapback voltage. Sebas's VA = 100 keeps the BJT in normal active until V_CE ≈ 100 V, so the snapback is **sharp**. Our deck flattens it.
- **Topology change implied:**
  - Restore Sebas's exact card values: `IS=5e-9 VA=100 Bf=10000 NE=1.5 NC=2 ITF=0.03 VTF=7 XTF=2`.
  - The ITF/VTF/XTF block (high-current rolloff) was almost certainly tuned to set the fold edge.

### 5. **M1 vs M2 cards have `etab` SIGN-FLIP (+1.8 vs −0.087) — already in R1, but not encoded as branch logic** [MED]
- **File:** `nsram/Zoom/2026-04-30 BSIMfitsBA/130DNWFB(M1).txt` vs `130bulkNSRAM(M2).txt`
- **R1 audit (`research_plan/R1_zoom_audit.md:79`):**
  > `etab | +1.8 | −0.086777 | Sign-flip — M1 is in floating-body regime, M2 is bulk-tied.`
- **k1 also differs:** M1 = 0.53825, M2 = 0.63825 (R1:78).
- **Why missing physics:** `etab` is the body-bias coefficient on `eta0` (DIBL). A positive etab on M1 means **DIBL increases with V_BS** — so as the floating body charges, DIBL exploser, V_th collapses, **M1 turns into a short** — that IS the snapback. Our pyport may load both cards but with the wrong `etab` sign on M1 (PTM default is negative).
- **Topology change implied:**
  - Audit the pyport NMOSdnwfb model loader to ensure `etab = +1.8` is **applied** (not silently clamped to negative or ignored). Existing log entry `01_LOG.md:387` claims `compute_size_dep` applies lalpha0/lbeta0 — but does it apply **etab**?
  - Worth a 30-min grep of `pyport_v5/bsim4/*.py` for `etab` handling.

---

## ALREADY-APPLIED PHYSICS (confirmed in repo, NOT a new finding)

These items were checked and ARE present in current pyport/research plan:
- BSIM4 §6.1 ALPHA0/BETA0/LALPHA0/LBETA0 channel HCI (T2:14–22).
- Gummel-Poon parasitic NPN with floating base = V_B (T2:24–30, but Bf/IS values differ — see #4 above).
- pdiode V_Nwell→V_B junction capacitance (T2:32–37, mail.txt:351).
- M1/M2 LDE asymmetry, NFACTOR(M2)-only knob (R1:75–82, mail.txt:321, mail.txt:376).
- VG1/VG2 polynomial wrapper on {ALPHA0, BETA0} (01_LOG.md:1065, mail.txt:227).
- L-binning of alpha0/beta0 (`lalpha0/Leff = −0.534`, 01_LOG.md:244, applied per 01_LOG.md:387).
- M2 = long channel `l = Ln*10 = 1.8µm`, w=Wn (R19_m2_body_audit.md:41, R_deep_A_topology_compare.md:31).
- CBpar = 1 fF B-to-GND cap (R_deep_A_topology_compare.md:33).

---

## RECOMMENDED NEXT EXPERIMENT (single concrete deck)

**R-55a — Zenodo NeuronSubCirc full port**: Build pyport instance with ALL of:
1. D3 zener G→B, `BVPar = 3.5 − 1.5·V_G1`, `nbvPar = 9 − 0.55/V_G1` (Tsinghua).
2. D4 zener B→Di, area=0.01, same params.
3. M3 BSS145 G2-gated body sink (V_th = 1.5 V).
4. Q1 NPN restored to Sebas's `IS=5e-9 VA=100 Bf=10000 NE=1.5 NC=2`.
5. Audit `etab = +1.8` on M1 load.

This is the **complete Zenodo NeuronSubCirc topology** that Sebas published. We have been running a **strict subset** — three of his five firing elements (D3, D4-with-VG-dependent-nbv, M3) are entirely absent from our deck. Any one of them, or the combination, is a plausible source of the 2–3 dec fold.

**Falsifiable claim:** if all five Zenodo elements are added and silicon snapback still fails by ≥1 dec, then the missing physics is genuinely outside Sebas's published model — at which point Track 1 (PSP / BSIM-IMG) becomes the right pivot.

---

## FILES CONSULTED (provenance)
- `/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/nsram/Zoom/mail.txt` (392 lines, full email thread)
- `/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/nsram/Zoom/schematic&modelCards/2tnsram_simple.asc`
- `/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/nsram/Zoom/schematic&modelCards/parasiticBJT.txt`
- `/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/nsram/Zoom/schematic&modelCards/PTM130bulkNSRAM.txt`
- `/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/nsram/Zoom/pdiode.txt`
- `/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/nsram/Zoom/2026-04-30 BSIMfitsBA/130DNWFB(M1).txt`
- `/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/nsram/Zoom/2026-04-30 BSIMfitsBA/130bulkNSRAM(M2).txt`
- `/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/nsram/Zoom/2026-04-30 13.03.27 Zoom NSRAM/meeting_saved_closed_caption.txt` (degraded Swedglish autocaptions — no usable physics)
- `/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/data/nsram_zenodo/SimulationFiles/SPICE/dev/subcircuit/NeuronSubCirc.asc`
- `/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/data/nsram_zenodo/SimulationFiles/SPICE/dev/BJTparams.txt`
- `/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/data/nsram_zenodo/SimulationFiles/SPICE/dev/Davalanche.txt`
- `/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/nsram/proposal_2026_05/01_LOG.md` (lines 10004–10112)
- `/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/research_plan/R1_zoom_audit.md`
- `/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/research_plan/T2_mario_physics.md`
- `/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/research_plan/R19_m2_body_audit.md`
- `/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/research_plan/R_deep_A_topology_compare.md`
- `/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/nsram/proposal_2026_05/ngspice_repro_harness/test_2t_cell_prod.sp`
