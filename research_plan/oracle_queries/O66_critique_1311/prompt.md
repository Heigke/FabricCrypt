# O66 — Critique cycle (6h, post-ablation honesty)

## Last 6h major events

- R-43 z363: 2D sweep (iii_gain × vnwell_Rs). Best 1.131 dec PASS still FAIL by 0.13
- R-44: BJT-leakage gemini-hypothesis FALSIFIED (pyport BJT matches ngspice 1.1%)
- R-45 z364: vnwell sweep FALSIFIED (anti-correlation VG1=0.20 vs VG1=0.60 structural, single global knob can't fix)
- DS-N5b HDC scale: 83.6% → 84.6% at N=100K (marginal +1.0pp, plateau)
- DS-N7 Memory Palace: 95.7% recall AT N=100K, 3.3 bits/cell, 38 fJ/read
- **DS-N7b ablation: BRUTAL CLAIM RETRACTED** — digital dict baseline = 0.957 EXACT. Hash does 100% work, NS-RAM no useful contribution. Noise cliff at 17mV codebook half-spacing. Key bit-flip 1% → chance.

## Three CRITICISM questions

**Q1 — Where is the latest claim fragile?**
We now claim:
- "3 real bugs fixed → 1.13 dec, global-knob floor honest"
- "Memory Palace retracted, BUT cell density 3.3 bits/cell @ 4 fJ remains genuine NS-RAM advantage"
- "Skala stack: N=1M GPU 60s, N=100K transient 22s"

Where are these still overclaimed? Specifically:
- Is the "cell density advantage" comparison apples-to-apples vs Flash (we counted read-channel current but not sensing amp / decoder overhead — DS-N7b admitted this)?
- Is 1.13 dec actually physics-correct or fitted-via-pdiode (body_pdiode_Js=24400 was set to Sebas card raw value; vnwell_Rs at 1e8 was BBO-tuned)?

**Q2 — Single best falsifier (<1h)** for "1.13 dec is real physics, not curve-fitting".
Concrete and runnable. (Hint: probably probing pyport vs ngspice at TWO biases that should require DIFFERENT body_pdiode parameters per VG1 — if pyport fits both with same param, suspicious.)

**Q3 — NO-CHEAT drift in last 6h?**
Cite specific 01_LOG.md lines where:
- Gate post-hoc shifted
- Subset cherry-pick (R-43 BBO on 6-bias subset, R-45 on 6 biases too)
- Claim made before falsifier was run

≤500 words per oracle. Be aggressive.
