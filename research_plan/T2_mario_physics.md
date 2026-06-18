# T2 — Mario/Sebas Body-Charge Physics Re-Scan

**Date:** 2026-05-14
**Question:** What physical mechanism does the 2T NS-RAM silicon use to charge the floating body V_B at high V_G1 / V_D?
**Sources:** `nsram/Zoom/mail.txt`, `nsram/Zoom/2026-04-30 13.03.27 Zoom NSRAM/meeting_saved_closed_caption.txt`, `nsram/Zoom/schematic&modelCards/*`, `nsram/Zoom/pdiode.txt`, `data/sebas_2026_04_22/M1_130DNWFB.txt`, `data/sebas_2026_04_22/M2_130bulkNSRAM.txt`, `research_plan/SA3_image_deep_extract.md`.

---

## 1. Mario's / Sebas's claimed mechanism (consolidated)

Two coupled effects charge V_B; **no GIDL, no BTBT at well-body, no cascaded ionization** is named anywhere:

### A. Channel hot-carrier impact ionization in M1 (channel HCI)
Mario's group, after head-to-head fit, concluded **§6.1 BSIM4 channel impact ionization (ALPHA0 / BETA0 / LALPHA0 / LBETA0) is the dominant body-charging path** (`mail.txt:222`,
`mail.txt:226`):

> "§6.1 impact ionization (ALPHA0 / BETA0) driving body charge directly" (`mail.txt:222`)
> "§6.1 channel HCI fits ~4 decades RMS better than §10.1 junction breakdown — so the channel-HCI route looks like the right match for your 2T cell" (`mail.txt:226`)

This is concretely encoded in Sebas's PDK extraction (`data/sebas_2026_04_22/M1_130DNWFB.txt` and `M2_130bulkNSRAM.txt`):
`alpha0 = 7.83756e-5`, `beta0 = 19` (M1) / `18` (M2), `lalpha0 = -9.843e-12`, `lbeta0 = -9.5e-7`.
These are the BSIM4 substrate-current parameters — i.e. Mario's "ionization" claim is **the standard BSIM4 I_sub model**, not a custom mechanism.

### B. Parasitic vertical NPN ("complementary bipolar current")
Sebas explicitly drops avalanche-diode firing and replaces it with a **lumped NPN parasitic bipolar** sitting at M2 (`mail.txt:183`):

> "I've dropped the avalanche diode models (very annoying for convergence) … and I'm only including a complementary bipolar current to capture the full swing of the firing mechanism." (`mail.txt:183`)

The element is in `schematic&modelCards/parasiticBJT.txt`:
`parasiticBJT NPN(is=5E-9 va=100 bf=10000 br=100 nc=2 ikr=100m …)` — `bf=10000` is the giveaway: it's a *latch amplifier*, not a true BJT. The schematic `2tnsram_simple.asc` wires Q1's base to node `B` (the floating body) and emitter to ground, so I_sub from M1 → V_B → bipolar collector current at M2 → V_D pulldown. Classic floating-body **kink / single-transistor-latch** topology.

### C. N-well → body diode (capacitive only, no breakdown)
The third element charging V_B is a **forward/reverse leakage + junction capacitance** path from V_Nwell, not avalanche (`mail.txt:351`, `pdiode.txt`):

> "an additional diode for clarity (pdiode with area 5 um x 4.4 um) to reflect the capacitive response of the floating body" (`mail.txt:351`)

The card (`pdiode.txt:1-9`) has `bv = 11 V` — above operating range — confirming no breakdown is intended; it provides V_B(V_Nwell)-dependent C_j only. SA3 image-21 extract corroborates: "Explicit parasitic N-well diode V_Nwell → V_B" (`SA3_image_deep_extract.md:47`).

## 2. Does Sebas/Mario acknowledge BSIM4 limitations?

**Sebas, yes — explicitly.** Three direct admissions:

> "I'm working with a different SPICE tool and foundry-provided models and I'm focusing on **adapting foundry models of body bias and impact ionization (in BSIM4) to fit the floating body behaviour** of our 2T cells. This renders a more standard approach for us circuit designers." (`mail.txt:181`)

> "Fits are looking good, but I'm still working on **polynomial dependence of model parameters with tuning voltages (VG1, VG2)** and layout dependent effect on transistor models to capture the experimental behaviour." (`mail.txt:183`)

> "Can your approach drop the avalanche voltage as a control parameter and **deal with the BSIM Impact ionization and body voltage directly**?" (`mail.txt:185`)

Translation: stock BSIM4 ALPHA0/BETA0 + parasitic-BJT do not, by themselves, reproduce silicon. Sebas patches it with (i) per-VG2 polynomial parameter tables and (ii) the M1-vs-M2 LDE split (NFACTOR only on M2; `mail.txt:321`, `mail.txt:376`). Mario's NSRAM v0.12.0 followed Sebas's structure (`mail.txt:220-229`).

## 3. Non-BSIM physics added in Mario's library

Per `mail.txt:120`:

> "Device physics layer — the full **body-charge ODE**, **Chynoweth avalanche model**, **SRH charge trapping**, and **temperature-dependent BVpar**, all matched to your Zenodo SPICE parameters."

So Mario's Brian2/scipy stack adds, on top of BSIM4:
- **Chynoweth** I_sub form (alternative to ALPHA0/BETA0; later down-weighted vs §6.1 channel HCI per `mail.txt:226`).
- **SRH trap charging** of body via interface/STI traps (multi-tau, picked up later as T4 candidate N1).
- **Body-charge ODE** integrating I_ion + I_BJT − I_leak − C·dV_B/dt.
- Temperature-dependent **BVpar** (vestigial — dropped in v0.12.0, `mail.txt:220`).

`firing_mode ∈ {"channel", "junction", "both"}` switch (`mail.txt:229`) — channel-HCI is the recommended setting.

## 4. Mechanism NOT claimed (explicitly absent)

The transcript and mail mention **none** of: GIDL, band-to-band tunneling at well-body junction, cascaded multi-stage ionization, true forward-biased SBD latch. The firing physics is the textbook **NMOS floating-body kink** mechanism: I_sub (BSIM §6.1) ramps V_B → reduces V_th(M1) → emitter injection from parasitic NPN → positive feedback → spike. The N-well diode supplies C_j and reset leak; the pdiode card explicitly sets `bv=11V` to keep it sub-breakdown.

## 5. Implication for our model

Our pyport already has BSIM4 + Gummel-Poon BJT (SA3 §79). What it lacks per T4: SRH traps (N1), explicit V_B-node C_j(V) from pdiode, V_B↔V_G2 designed coupling cap, V_G1-mediated V_B coupling into M2 NFACTOR. None of these require non-BSIM "exotic" physics — they require getting the parasitic network around V_B right.
