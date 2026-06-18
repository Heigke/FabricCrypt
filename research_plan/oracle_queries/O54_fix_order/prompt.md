# O54 — Optimal Fix-Order for pyport_v4 (P1 of MASTER_FIX_PLAN_2026-05-13)

We have a complete diagnosis of why our pyport model is 10× off in current
(0.99 dec) on Sebas IV data. Multiple confirmed missing-physics elements:

**CONFIRMED (today):**
- N1 multi-τ trap mechanism (z311 stub showed 6.2-dec hysteresis lift, proves slide-21 multi-rate physics)
- VNwell→VB diode polarity bug in z310 (anode/cathode swap needed)
- T2 extracted 143 (V_d, I_d) samples at V_d>2V from slide-15+21 (new validation set)
- T1 z307 VNwell breakdown gave +0.58 dec on V_G1=0.2 (helps but insufficient)

**PENDING:**
- pyport_v4 build combining traps + diode (correct polarity) + drain avalanche
- KWS at chance level (Oracle 4D primary ship-blocker)
- v4.4 brief HELD pending model closure

## Reference attachments
- `MASTER_FIX_PLAN_2026-05-13.md` — full plan structure
- `T4_missing_physics_v2.md` — N1/N2/N3 physics candidates
- `4D_critique_synthesis.md` — Oracle 4D ship-gate verdict
- `01_LOG_tail100.md` — last 100 lines of operations log
- `z304_sebas_refit_summary.json` — baseline pyport fit metrics
- `z311_traps_summary.json` — N1 trap stub hysteresis result
- `z308_slide_v2v_samples.json` — 143 new V_d>2V validation samples
- `z312_hdc_n16k_summary.json` — HDC application benchmark

## Questions — answer with brevity (≤500 words total)

**Q1: OPTIMAL FIX-ORDER for next 8h of compute.** Which subset of {N1 traps,
VNwell diode w/correct polarity, drain avalanche M(V_bc), SRH gen-rec, BSIM
rbodymod=1 (needs code)} should we implement FIRST in pyport_v4 to maximize
cell-wide DC fit improvement AND transient agreement at slide-21 ramp rates?

**Q2: For each phase P3 (pyport_v4) and P4 (KWS attack), what is the
cheapest experiment that would FALSIFY it within 2h wall time?**

**Q3: Should we ABANDON KWS as a v4.4 application target** (and instead lead
with HDC + RNG), or persist? Cite physical-arguments for/against
NS-RAM-as-SNN for sub-100µW audio.

**Q4: Are we still drifting from NO-CHEAT discipline** (look at 01_LOG.md
tail)? Cite specifically.

Be sharp. ≤500 words per oracle.
