# O65 — 12h gap-closing review (post BREAKTHROUGH)

## Context

Last 12h log + R-40 audit + R-41 result attached.

NS-RAM 2T pyport DC fit history:
- z358 (R-37 binunit alone): 4.28 dec
- z346 (Vth/tox alone): 4.08 dec
- z352 (eta_sigmoid): 3.93 dec
- z304 (broken BJT spurious): 0.99 dec ← retracted
- **R-41 z361 (NEW): 1.42 dec PASS gate ✓**
  Sebas's parasitic pdiode (V_B → V_Nwell) was THE missing piece per R-40 deep audit.

Active fixes stacked:
- R-20: BJT Vbc=Vb-Vsint
- R-29: lpe0=1.2439e-7, toxe=4e-9
- R-37: binunit override removed (1e6 bug)
- R-39: BJT params (Bf=991, Va=0.90, Is=6e-12, iii_body_gain=0.66)
- R-41: cfg.body_pdiode_to="vnwell", use_well_diode=True, body_pdiode_Js from Sebas card

Per-VG1: 0.20=1.36, 0.40=1.24, 0.60=1.87. AMBITIOUS<0.95 still FAIL.

V_b at flagship 0.484V (was 0.78V over-pumped; ngspice gives 0.27V — still 0.21V above).

R-42 Rs-sweep INVARIANT: Rs ∈ [1e3, 1e7] gives same cell-wide 1.42 dec. Vb doesn't move. Bottleneck is elsewhere.

Scale stack ALSO complete: S1 GPU forward_2t N=1M@60s, S2b transient N=100K@22s, S3 network+variation framework. 1M-cell simulator ready.

## Three questions

**Q1 — gate-crossing risk**: Is the 1.42 dec real or is it absorbing some compensation again? Specifically:
- R-41 uses Sebas's pdiode card directly (Js, n, area) — this is "use ground truth from Sebas". Different from R-39 BBO which TUNED to data. Is this honest or fitting?
- per-VG1=0.40 is now BEST at 1.24, lower than VG1=0.20 at 1.36. Is this physically reasonable, or a sign of over-correction?
- 1.42 dec = pyport now within ~25× of silicon median. Honest?

**Q2 — single best falsifier (<1h)**: Best experiment to test if 1.42 is real and not spurious. Concrete and runnable.

**Q3 — push to AMBITIOUS (<0.95)**: Why does Rs sweep have ZERO effect on Vb (still 0.484V at all Rs)? What ELSE could be holding Vb at 0.484 instead of ngspice's 0.27? Single highest-value experiment to push Vb closer to 0.27 and crack AMBITIOUS.

≤450 words per oracle.
