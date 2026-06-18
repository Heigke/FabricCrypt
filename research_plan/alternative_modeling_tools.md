# Alternative Modeling Tools for Floating-Body PNPN / NS-RAM Devices

**Date**: 2026-05-16
**Owner**: research_plan (T1 follow-up)
**Context**: BSIM4-based pyport stuck at ~1.0 decade RMSE (1.39 dec honest, M3b walk-back). The TSMC-130nm 2T NS-RAM cell IS a thyristor (PNPN) with 400 ns relaxation osc. Need to evaluate whether a different compact-model family closes the residual.

**Method**: web survey + Zenodo/repo inspection. Where literature is silent on NS-RAM application I say so.

---

## 0. Mario / Lanza group's actual SPICE choice (anchor)

From Pazos et al. Zenodo dataset (DOI 10.5281/zenodo.13843362) + local audit in `SA2_zenodo_process_map.md`:

- **Simulator**: **LTspice** (the deck ships `.asc`/`.asy` schematic+symbol pairs and one `subcircuit/SubC_SimpleTest.asc`; these are LTspice native formats).
- **MOSFET model**: PTM 130 nm bulk `nmos4` cards (`PTM130bulk_lite.txt`), BSIM4 level, **thin-ox**, NOT a calibrated foundry/imec card. README explicitly says "not unique to any process, only exemplary."
- **Parasitic NPN**: hand-built sub-circuit using SPICE built-in BJT (Gummel-Poon-ish) with `BVpar`/`nbvpar` formulas that switch between **Tsinghua** (default) and **TSMC** parameter sets.
- **Avalanche diode**: zener with `bv = 0.9·BVpar`.
- **TCAD**: dataset README references "Full projects for TCAD simulation of the floating bulk devices" with "command files for parametric structure construction and meshing" — language ("command files", "parametric structure construction") matches **Synopsys Sentaurus** workflow (sde/sdevice command syntax), not Silvaco. Not 100% confirmed without unpacking the zip.

**Implication**: Mario's stack is a *low-tech sub-circuit cobble* (PTM + zener + Gummel-Poon BJT in LTspice). Our pyport already replicates that topology in Python. Any tool that does the same topology will give the same ~1 dec residual — the residual is **structural** (Slotboom non-local IIMOD missing, see T1_bsim4_alternatives.md), not a choice-of-simulator problem.

[Pazos et al. Zenodo 13843362](https://zenodo.org/records/13843362) ・ [Lanza et al. Nature 641 (2025)](https://www.nature.com/articles/s41586-025-08742-4)

---

## 1. BSIM-CMG (Berkeley multi-gate)

- **Snapback / PNPN native?** No. CMG is for FinFET/GAA bulk-isolated channels. Substrate node carries Iimp/GIDL termination, but **no parasitic-BJT branch**, no PNPN latch. BSIM-MG FAQ admits a small body-bias residue is *not* captured.
- **Open-source?** Yes — Berkeley releases Verilog-A under permissive license.
- **Effort to port pyport?** Medium-high. Surface-potential core differs from BSIM4; we'd rewrite the channel kernel. Roughly 2-3 weeks for a Python re-implementation; days if we just adopt `BSIM-CMG.va` via ngspice/Xyce.
- **Used in NS-RAM literature?** No NS-RAM application found. CMG is FinFET territory; NS-RAM is bulk 130 nm.
- **Advantage over our stack?** Negligible for this device. **Reject.**

[BSIM-MG FAQ](https://bsim.berkeley.edu/bsim-mg-faq/) ・ [BSIM-CMG manual](http://srware.com/xictools/docs/model_docs/bsimcmg-1.0.7/BSIMCMG107.0.0_TechnicalManual_20130712.pdf)

---

## 2. BSIM-SOI / Symmetric BSIM-SOI (Berkeley + Si2 CMC)

- **Snapback / PNPN native?** Floating-body **yes** (this is what it's designed for); explicit BJT branch and impact-ionization current at the body node. Symmetric BSIM-SOI Part II (TED 2024) targets PD-SOI specifically. Snapback per se is not its primary claim but the kink and body-charging dynamics are first-class.
- **Open-source?** Yes — UC Berkeley + Si2 CMC standard. `BSIM-SOI 4.7.0_Beta9` (Nov 2024) actively maintained. Verilog-A code is downloadable.
- **Effort to port pyport?** **Medium**. Same Berkeley DNA as BSIM4, so the API/parameter philosophy is familiar. Estimated 1-2 weeks to re-derive the body-charge equations in Python; or zero days if we run via ngspice/Xyce + foundry-style PDK. **Caveat**: our device is **bulk 130 nm with deep N-well**, not SOI. BSIM-SOI assumes a buried-oxide; our isolation is junction-based. Using BSIM-SOI on a bulk DNW device is a **substrate mismatch** — we'd be using the right floating-body math on the wrong topology.
- **NS-RAM literature?** No direct NS-RAM application found, but Z²-FET 1T-DRAM (closely related floating-body PNPN-ish device) uses surface-potential SOI compact models — see §6.
- **Advantage**: Native floating-body math (kink, GIFBE, body-charge RC). But topology mismatch limits gain. **Conditional reject** — use only as a *reference implementation* of the floating-body equations to crib physics from.

[BSIM-SOI page](https://www.bsim.berkeley.edu/models/bsimsoi/) ・ [Symmetric BSIM-SOI Part II, TED 2024](https://ieeexplore.ieee.org/document/10436627) ・ [BSIMSOIv4.4 manual](http://www.srware.com/xictools/docs/model_docs/bsimsoi-4.4/BSIMSOIv4.4_UsersManual.pdf)

---

## 3. EKV (Enz-Krummenacher-Vittoz, EPFL)

- **Snapback / PNPN native?** No. EKV is a *charge-based* compact model, strong on weak-inversion + symmetry, **not** on impact ionization/snapback. The bipolar-like exponential is a *channel* exponential (subthreshold), not a parasitic BJT.
- **Open-source?** Yes — EPFL releases the Verilog-A; ngspice ships an EKV level.
- **Effort to port pyport?** Medium. EKV is famously compact (~4 fundamental parameters), so re-implementation is fast (~3-5 days). But adding a Gummel-Poon BJT and IIMOD on top is the same exercise we already did on BSIM4.
- **NS-RAM literature?** No NS-RAM application found.
- **Advantage**: Cleaner weak-inversion behaviour (we operate near V_T ≈ V_GS, so subthreshold matters). But snapback/PNPN is **outside EKV's domain**. **Reject as kernel**; could be useful as a **subthreshold sanity check** against BSIM4's weak-inversion drift.

[Enz-Krummenacher-Vittoz model equations](https://ngspice.sourceforge.io/external-documents/models/ekv_v262.pdf) ・ [Heuristic EKV fitting open-source (MDPI 2024)](https://www.mdpi.com/2079-9292/14/6/1162)

---

## 4. HiSIM-HV (Hiroshima University / STARC)

- **Snapback / PNPN native?** **Yes, partially.** HiSIM-HV is a surface-potential model for high-voltage MOSFETs that explicitly models impact ionization in the drift region (I_KIRK), and a 2018 paper ("Consistent Modeling of Snapback Phenomenon Based on Conventional I-V Measurements", IEEE) demonstrates snapback within the HiSIM-HV framework. No native PNPN latch, but the kink + body-feedback path is first-class.
- **Open-source?** Source is available via Hiroshima Univ / Si2 CMC (academic). Verilog-A is in `dwarning/VA-Models` (GitHub) for community use.
- **Effort to port pyport?** **High**. Surface-potential solver inside HiSIM is iterative and parameter-rich (>40 params). Estimated 2-4 weeks to Python re-implementation, or zero days via ngspice/Xyce + Verilog-A.
- **NS-RAM literature?** **No NS-RAM application found** — but the closest fit physics-wise (kink in drift, snapback via I-V): this is the family of models that genuinely captures what BSIM4 misses (per T1_bsim4_alternatives.md). HiSIM-HV is what an LDMOS/IGBT modeling team would use, and the NS-RAM punch-through latch is mechanically similar.
- **Advantage**: native non-local IIMOD-adjacent treatment; explicit snapback in a published paper.

[HiSIM-HV 2010 TED](https://ieeexplore.ieee.org/document/5567144/) ・ [Consistent modeling of snapback (2018)](https://www.researchgate.net/publication/329510109_Consistent_Modeling_of_Snapback_Phenomenon_Based_on_Conventional_I-V_Measurements) ・ [HiSIM-HV2 dc PD-SOI LDMOS](https://link.springer.com/article/10.1007/s10825-013-0457-8)

---

## 5. PSP / MM11 (Penn State + Philips, Si2 CMC standard)

- **Snapback / PNPN native?** PSP includes impact ionization, kink, self-heating, GIFBE. **PSP-SOI** adds explicit floating-body simulation, parasitic body currents, valence-band tunneling. Native PNPN latch is **no**, but every component (impact gen → body-Q → kink) is there.
- **Open-source?** Yes — CMC standard; Verilog-A is publicly distributable.
- **Effort to port pyport?** **High**. PSP is a *full surface-potential* model — the channel solver alone requires implicit equation iteration. Estimated 3-4 weeks to a working Python core. Zero-day via ngspice/Xyce.
- **NS-RAM literature?** No direct NS-RAM. PSP-SOI is the closest analogue used in PD-SOI floating-body memory simulation.
- **Advantage**: Yadav et al. (IEEE 2010) explicitly compared PSP vs BSIM4 and showed **PSP passes kink/IP3 where BSIM4 fails**. This is the canonical replacement that the compact-modeling community uses when BSIM4's local-IIMOD breaks. **Strongest "swap-the-kernel" candidate.**

[PSP main paper (TED 2006)](https://ieeexplore.ieee.org/document/1677832/) ・ [PSP-SOI Wu et al. Solid-State Electron 2009](https://ui.adsabs.harvard.edu/abs/2009SSEle..53...18W/abstract) ・ [Analysis of kink in short-channel PD-SOI](https://www.researchgate.net/publication/371530902_Analysis_of_Kink_Effect_in_Short-Channel_Floating_Body_PD-SOI_MOSFETs)

---

## 6. Verilog-A custom compact model (literature best-practice)

This is the **ESD modeling community's main weapon for PNPN/SCR snapback**. Two distinct flavors:

### 6a. Behavioral snapback (Mergens-style)
- Define V_t1 (trigger), V_h (holding), R_on directly from TLP I-V via piecewise/smoothed if-else in Verilog-A.
- "A physically-based behavioral snapback model" (IEEE 2012, doc 6333317) — robust convergence in HBM/MM/TLP transient SPICE.
- "Scalable Verilog-A modeling method for ESD protection devices" — area scaling.

### 6b. BESD (Berkeley Electrostatic Discharge) — Roychowdhury group, **open-source**
- GitHub: `jaijeet/BESD`. Models **SCR (PNPN!) and bipolar clamps** in Verilog-A AND ModSpec/MATLAB.
- Explicit design goal: **continuous, smooth, well-posed ODEs — no if-else, no discontinuities** → convergence-clean across simulators.
- Published rationale: T. Wang, "Modelling Multistability and Hysteresis in ESD Clamps, Memristors and Other Devices" (CICC 2017).

- **Snapback / PNPN native?** **Yes — both, by design.**
- **Open-source?** Yes (BESD, BSD-ish license per repo).
- **Effort to port pyport?** **Low-medium**. BESD's SCR model is the closest existing match to our NS-RAM topology. We could either (i) wrap BESD's Verilog-A via ngspice/Xyce, or (ii) port BESD's ModSpec/MATLAB equations to Python (a few hundred lines). The smoothness property is ideal for our pseudo-transient continuation solver.
- **NS-RAM literature?** No direct NS-RAM citation found in BESD repo. But BESD's SCR is the canonical open-source PNPN compact model.
- **Advantage over our stack**: (a) native PNPN latch (we currently bolt a Gummel-Poon NPN onto a MOSFET — BESD treats the whole 4-layer device as one device); (b) smoothness guarantees fewer convergence stalls than Mario's discontinuous BVpar formulas.

**This is the single most promising substitution.**

[BESD on GitHub](https://github.com/jaijeet/BESD) ・ [Physically-based behavioral snapback (IEEE 2012)](https://ieeexplore.ieee.org/document/6333317) ・ [Compact ESD modeling using Verilog-A (Li et al.)](https://www.researchgate.net/publication/3225830_Compact_modeling_of_on-chip_ESD_protection_devices_using_Verilog-A)

### 6c. Related — Z²-FET / A2RAM compact models (1T-DRAM floating-body PNPN-like)
- "Pragmatic Z²-FET compact model including DC and 1T-DRAM memory operation" (Solid-State Electron 2021) — full Verilog-A, surface-potential + empirical V_ON/V_OFF, validated against TCAD AND experimental data. Z²-FET is electrostatically a PNPN diode used for memory, **directly analogous to NS-RAM**.
- This is the **closest published precedent** for compact-modeling a PNPN-floating-body memory device in Verilog-A.

[Pragmatic Z²-FET compact model (SSE 2021)](https://www.sciencedirect.com/science/article/abs/pii/S0038110121000058) ・ [Z²-FET review (SSE 2018)](https://www.sciencedirect.com/science/article/abs/pii/S0038110117306512)

---

## 7. TCAD: Sentaurus / Silvaco Atlas / nanohub

- **Snapback / PNPN native?** **Yes, fully** — TCAD solves drift-diffusion + impact ionization + Shockley-Read-Hall self-consistently. Thyristors, IGBTs, LDMOS are routine. Silvaco's "Modeling Bidirectional Thyristors Using ATLAS" is a published recipe.
- **Open-source?** **No.** Synopsys Sentaurus and Silvaco Atlas are commercial (~6-figure licenses). Free academic alternatives: **DEVSIM** (open-source DD solver), **GSS/Genius**, nanohub-hosted simulators (login-gated).
- **Effort to port pyport?** N/A — TCAD doesn't replace pyport, it **validates** it. The "port" is: dump pyport-equivalent device under TCAD, generate I-V/transient ground truth, then refit pyport to TCAD. Effort: 1-2 weeks to set up the 130 nm DNW cell in Sentaurus.
- **NS-RAM literature?** **Yes.** Lanza/Pazos Nature 2025 itself uses TCAD (the Zenodo dataset includes "Full projects for TCAD simulation of the floating bulk devices used to validate the dynamics and origin of the neural behaviour"). They use TCAD as the **mechanism validator**, not as a circuit simulator. The compact model still runs in LTspice; TCAD is the truth source.
- **Advantage**: Resolves the Slotboom non-local IIMOD ambiguity by *computing* α(E, T_e) from first principles. **The most expensive but the most defensible cross-validation.**

[Sentaurus Device datasheet](https://www.synopsys.com/content/dam/synopsys/silicon/datasheets/sentaurus_ds.pdf) ・ [Silvaco Atlas bidirectional thyristors](https://silvaco.com/simulation-standard/modeling-bidirectional-thyristors-using-atlas/) ・ [T-RAM TCAD retention study (2023)](https://www.researchgate.net/publication/376630839)

---

## 8. Open-source SPICE engines

### 8a. ngspice
- **Verilog-A?** Via OpenVAF / ADMS-translated `.osdi` modules. Production-grade since 2023+.
- **VBIC (level=4 BJT)?** **Yes**, with parasitic substrate transistor + avalanche multiplication. Zhou et al. (Modeling Snapback and Rise-time Effects in TLP Testing for ESD) explicitly use **BSIM3 + VBIC** as the substrate-current bridge for snapback. **This is the published recipe for our exact substitution**: replace our Gummel-Poon NPN with VBIC.
- **Effort**: minor — swap `.model` line, refit ~10 BJT parameters. Days of work.
- **Advantage**: VBIC's avalanche term is more physical than the Tsinghua-default BVpar formula in Mario's deck.

### 8b. Xyce (Sandia)
- **Parallel** SPICE-compatible engine. Supports Verilog-A via ADMS backend (Xyce/ADMS guide). Same model files as ngspice (`.va` → C++).
- **Convergence**: Xyce docs explicitly flag exponential-voltage terms as the snapback convergence enemy and recommend smooth-ODE formulations — exactly what BESD does.
- **Effort**: same as ngspice. We can drop the same `.va` into both and cross-check.
- **Advantage**: parallel sweeps (parameter Monte Carlo for binning audit), faster than ngspice on Mario sweep matrices.

### 8c. gnucap + Modelgen-Verilog (NLnet, 2024)
- Verilog-AMS support is **actively under development** (FOSDEM 2024 talk). Not production yet for compact models with snapback.
- **Not recommended** for our 24h window — bleeding edge.

[ngspice VBIC docs](https://nmg.gitlab.io/ngspice-manual/bjts/bjtmodels_npn_pnp.html) ・ [Xyce ADMS guide](https://xyce.sandia.gov/documentation/XyceADMSGuide.html) ・ [Zhou et al. BSIM3+VBIC snapback](https://www.researchgate.net/profile/Yuanzhong-Zhou/publication/229068997) ・ [Gnucap Verilog-AMS, FOSDEM 2024](https://archive.fosdem.org/2024/schedule/event/fosdem-2024-3560-verilog-ams-in-gnucap/)

---

## 9. AI / ML surrogates

- **Snapback / PNPN native?** Whatever the training data shows. A Gaussian process or NN learns *the* I-V — including snapback — if Sentaurus/measurement covers it.
- **Open-source?** Yes (scikit-learn, GPyTorch, PyTorch). Mehta-Wong autoencoder approach: predict FinFET I-V/C-V from process params. NVIDIA "AI Physics for TCAD" (2024). ACM TODAES 2025: "TCAD-ML enabled TID compact model for SiC MOSFET" — surrogate + Bayesian calibration via Dakota.
- **Effort to port pyport?** **Low** (build phase) / **HIGH** (data phase). Training data is the bottleneck: need either dense Sebas measurements (we have ~1 IV sweep per Vg2) or a Sentaurus run (see §7).
- **NS-RAM literature?** **No NS-RAM application found.** TID/FinFET adjacent.
- **Advantage**: zero residual *on the training set*. Risk: extrapolation outside training distribution (e.g. snapback branch with one un-sampled Vg2) is silently wrong. **Use only as an interpolator on top of TCAD-generated ground truth, never as the primary physics**.

[TCAD-ML for SiC MOSFET TID, ACM 2025](https://dl.acm.org/doi/10.1145/3766551) ・ [TCAD Device Sim with GNN, TED](https://ieeexplore.ieee.org/iel7/55/4357973/10168926.pdf) ・ [NVIDIA AI Physics for TCAD](https://developer.nvidia.com/blog/using-ai-physics-for-technology-computer-aided-design-simulations/) ・ [Compact ANN model for GAA NSFET](https://pmc.ncbi.nlm.nih.gov/articles/PMC10890573/)

---

## 10. Multi-physics — COMSOL Semiconductor, Silvaco

- **Snapback / PNPN native?** COMSOL forum explicitly warns: "PNPN devices are highly non-linear and likely to encounter convergence problems in COMSOL." Multiple research-community comments say COMSOL is **not** as good as Sentaurus/Silvaco for semiconductor work. COMSOL is FEM-multiphysics first, semiconductor second.
- **Open-source?** No.
- **Effort to port pyport?** N/A; same as §7 — validation only.
- **NS-RAM literature?** No NS-RAM application in COMSOL found.
- **Advantage**: thermal-electrical coupling is COMSOL's strength. Useful **only** if self-heating in the bulk-NPN turn-on becomes a residual driver, which T1_bsim4_alternatives.md does not currently rank top-3.

[COMSOL PNPN forum thread](https://www.comsol.com/forum/thread/63781/pnpn-thyristor-device-simulation) ・ [Silvaco vs COMSOL edaboard discussion](https://www.edaboard.com/threads/tcad-question-silvaco-atlas-versus-comsol-multiphysics.145825/)

---

## Comparison Table

| # | Tool | Snapback / PNPN native | Open-source | Used in NS-RAM/PNPN-memory literature | Effort to port pyport | Verdict |
|---|---|---|---|---|---|---|
| 0 | LTspice + PTM + GP-NPN (Mario's) | partial (sub-circuit) | LTspice = free-ware (not open) | **YES** (Pazos/Lanza Nature 2025) | n/a (baseline) | residual = 1.0-1.4 dec → keep as reference |
| 1 | BSIM-CMG | NO | YES | NO | medium-high | **reject** |
| 2 | BSIM-SOI (Sym, 2024) | YES (floating body) | YES | indirect (PD-SOI memory) | medium | use as physics reference only (topology mismatch) |
| 3 | EKV | NO | YES | NO | medium | reject as kernel; use for subthreshold sanity |
| 4 | HiSIM-HV | YES (snapback published) | YES (CMC) | NO direct | high | strong physics; high cost |
| 5 | **PSP / PSP-SOI** | YES (kink+II+SH+body) | YES (CMC standard) | indirect (PD-SOI memory) | high | canonical BSIM4 replacement |
| 6a | Verilog-A behavioral (Mergens) | YES (by construction) | author code public | NO direct | low | fast empirical fit, not physical |
| 6b | **BESD (Roychowdhury)** | **YES — native SCR/PNPN** | **YES (GitHub)** | NO direct; canonical ESD/SCR | **low-medium** | **TOP PICK — smooth ODE PNPN** |
| 6c | Z²-FET / A2RAM Verilog-A | YES (PNPN memory!) | papers + code in supp. | **YES (Z²-FET = closest analogue)** | medium | second-strongest precedent |
| 7 | Sentaurus / Silvaco TCAD | YES (full physics) | NO (commercial) | **YES (Lanza/Pazos use TCAD)** | n/a — validator | mandatory cross-validation |
| 8a | ngspice + VBIC (level-4 BJT) | partial (BJT side native) | YES | **YES (Zhou: BSIM3+VBIC for snapback)** | days | **easy win** — drop-in NPN upgrade |
| 8b | Xyce + ADMS | partial | YES | NO direct | days | parallel sweeps for binning |
| 8c | gnucap + Modelgen-Verilog | partial | YES | NO | too new | defer |
| 9 | ML surrogate (GP/NN) | data-dependent | YES (libs) | NO | low build / HIGH data | interpolator only |
| 10 | COMSOL Semiconductor | poor (convergence) | NO | NO | n/a | reject |

---

## Recommendation — Next 24 h

The 1.0-1.4 decade residual is **structural** (per T1_bsim4_alternatives.md: missing non-local IIMOD + missing native PNPN latch), and the Mario topology (PTM-BSIM4 + Gummel-Poon NPN + zener) is sub-optimal in two specific ways:

1. The parasitic NPN is a stock Gummel-Poon with hand-coded `BVpar` formulas — **VBIC is the published upgrade** and ngspice supports it natively (Zhou et al. recipe).
2. The whole PNPN is modeled as separate-device sub-circuit. **BESD models the SCR/PNPN as one smooth-ODE device** with explicit V_t1, V_h, R_on parameters extractable directly from Sebas's measured snapback I-V.

### 24 h plan (ranked, parallelizable)

**Track A — Cheap wins, no new tools (4-6 h)**
- A1. Swap our Gummel-Poon NPN for **VBIC level-4** in pyport. Refit Bf, BVcbo, nbvcbo, plus the new substrate-transistor params. Expected residual drop: **0.2-0.4 dec** based on Zhou et al.
- A2. Re-run honest DC fit and B.5 dichotomy with VBIC NPN. If <1.0 dec, ship.

**Track B — BESD substitution (8-12 h)**
- B1. Clone `jaijeet/BESD`, read `BESD_1_0_0_ModSpec_VAPP.m` SCR model. Port the smooth-ODE equations to Python (a few hundred LOC).
- B2. Replace the BSIM4+NPN sub-circuit in pyport with a single BESD-SCR device parameterized by (V_t1, V_h, R_on, area). Add the body-RC and trap reservoir externally.
- B3. Fit (V_t1, V_h, R_on) to Sebas DC IV using TLP-style extraction. Expected residual: targeted at **<0.5 dec** because the PNPN is now treated as one device, not three.

**Track C — Z²-FET precedent (4 h, optional)**
- C1. Fetch the Pragmatic Z²-FET Verilog-A paper supplemental and benchmark its surface-potential V_ON/V_OFF formulation against Sebas's data. If the V_ON shift matches our Vg2-dichotomy regime selector, it confirms the structural pattern (per `nsram_m3b_corrections.md`: ER_SPARSE wins MC).

**Track D — TCAD as validator (set up now, run overnight)**
- D1. Unpack `SimulationFiles.zip` from Pazos Zenodo; identify whether their TCAD projects (likely Sentaurus) target the same imec thick-ox device or PTM. If thick-ox, **use them as ground truth** for refitting.
- D2. If not, schedule a single Sentaurus DD run of the 130 nm DNW cell at academic licence (nanohub or local) — generate I-V + Ibody sweep, use as anchor.

**Defer / drop**: BSIM-CMG (§1), EKV (§3), HiSIM-HV (§4 — too expensive for 24 h), PSP full port (§5 — too expensive for 24 h, keep as physics reference), gnucap (§8c), COMSOL (§10), ML surrogate (§9 — wait for TCAD data first).

### Single-sentence answer

**Within 24 h, do Track A (VBIC swap, ngspice-driven) first — it's the published BSIM3+VBIC snapback recipe and should buy us ~0.3 dec — then start Track B (BESD SCR port) which is the only candidate that natively models the PNPN as one smooth-ODE device and is open-source. Reserve Sentaurus/TCAD (Track D) for validation, not as the primary kernel.**

---

## Sources

- [Pazos et al. Zenodo 13843362](https://zenodo.org/records/13843362)
- [Lanza et al. Nature 641 (2025)](https://www.nature.com/articles/s41586-025-08742-4) ・ [PMC mirror](https://pmc.ncbi.nlm.nih.gov/articles/PMC11964925/)
- [BSIM-SOI page](https://www.bsim.berkeley.edu/models/bsimsoi/) ・ [Symmetric BSIM-SOI Part II TED 2024](https://ieeexplore.ieee.org/document/10436627) ・ [BSIM-SOI v4.4 manual](http://www.srware.com/xictools/docs/model_docs/bsimsoi-4.4/BSIMSOIv4.4_UsersManual.pdf)
- [BSIM-MG FAQ](https://bsim.berkeley.edu/bsim-mg-faq/) ・ [BSIM-CMG manual](http://srware.com/xictools/docs/model_docs/bsimcmg-1.0.7/BSIMCMG107.0.0_TechnicalManual_20130712.pdf)
- [EKV equations (ngspice)](https://ngspice.sourceforge.io/external-documents/models/ekv_v262.pdf) ・ [Heuristic EKV fitting MDPI 2024](https://www.mdpi.com/2079-9292/14/6/1162)
- [HiSIM-HV TED 2010](https://ieeexplore.ieee.org/document/5567144/) ・ [Consistent snapback modeling 2018](https://www.researchgate.net/publication/329510109_Consistent_Modeling_of_Snapback_Phenomenon_Based_on_Conventional_I-V_Measurements)
- [PSP TED 2006](https://ieeexplore.ieee.org/document/1677832/) ・ [PSP-SOI SSE 2009 (Wu et al.)](https://ui.adsabs.harvard.edu/abs/2009SSEle..53...18W/abstract) ・ [Kink in short-channel PD-SOI 2023](https://www.researchgate.net/publication/371530902)
- [BESD GitHub (Roychowdhury)](https://github.com/jaijeet/BESD) ・ [BESD Verilog-A subdir](https://github.com/jaijeet/BESD/tree/master/Verilog-A) ・ [BESD ModSpec source](https://github.com/jaijeet/BESD/blob/master/MAPP/ModSpec/BESD_1_0_0_ModSpec_VAPP.m) ・ [VAPP Verilog-A→ModSpec](https://github.com/jaijeet/VAPP)
- [Physically-based behavioral snapback IEEE 2012](https://ieeexplore.ieee.org/document/6333317) ・ [Scalable Verilog-A ESD method](https://www.researchgate.net/publication/224190121_A_scalable_Verilog-A_modeling_method_for_ESD_protection_devices) ・ [Li et al. Verilog-A ESD compact](https://www.researchgate.net/publication/3225830_Compact_modeling_of_on-chip_ESD_protection_devices_using_Verilog-A)
- [Pragmatic Z²-FET compact model SSE 2021](https://www.sciencedirect.com/science/article/abs/pii/S0038110121000058) ・ [Z²-FET review SSE 2018](https://www.sciencedirect.com/science/article/abs/pii/S0038110117306512) ・ [A2RAM compact modeling SSE 2020](https://www.sciencedirect.com/science/article/abs/pii/S0038110119307348) ・ [Floating-body diode DRAM IEEE](https://ieeexplore.ieee.org/document/6112172) ・ [T-RAM Wikipedia](https://en.wikipedia.org/wiki/T-RAM)
- [Sentaurus Device datasheet](https://www.synopsys.com/content/dam/synopsys/silicon/datasheets/sentaurus_ds.pdf) ・ [Silvaco Atlas bidirectional thyristors](https://silvaco.com/simulation-standard/modeling-bidirectional-thyristors-using-atlas/) ・ [T-RAM cryo retention TCAD](https://www.researchgate.net/publication/376630839)
- [ngspice BJT/VBIC](https://nmg.gitlab.io/ngspice-manual/bjts/bjtmodels_npn_pnp.html) ・ [VBIC text](https://designers-guide.org/vbic/documents/VbicText.pdf) ・ [Zhou et al. BSIM3+VBIC for snapback](https://www.researchgate.net/profile/Yuanzhong-Zhou/publication/229068997)
- [Xyce ADMS guide](https://xyce.sandia.gov/documentation/XyceADMSGuide.html) ・ [Xyce GitHub](https://github.com/Xyce/Xyce) ・ [VA-Models GitHub](https://github.com/dwarning/VA-Models)
- [Gnucap Verilog-AMS NLnet](https://nlnet.nl/project/Gnucap-VerilogAMS/) ・ [FOSDEM 2024 Gnucap talk](https://archive.fosdem.org/2024/schedule/event/fosdem-2024-3560-verilog-ams-in-gnucap/)
- [TCAD-ML SiC MOSFET TID ACM 2025](https://dl.acm.org/doi/10.1145/3766551) ・ [TCAD Device Sim GNN TED](https://ieeexplore.ieee.org/iel7/55/4357973/10168926.pdf) ・ [NVIDIA AI Physics for TCAD](https://developer.nvidia.com/blog/using-ai-physics-for-technology-computer-aided-design-simulations/) ・ [ANN compact model for GAA NSFET PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10890573/)
- [COMSOL PNPN forum](https://www.comsol.com/forum/thread/63781/pnpn-thyristor-device-simulation) ・ [COMSOL vs Silvaco edaboard](https://www.edaboard.com/threads/tcad-question-silvaco-atlas-versus-comsol-multiphysics.145825/)
