# O74 — Creative Solutions to NS-RAM Snapback Fold Problem

You are one of 4 oracles on a panel (openai gpt-5, gemini-2.5-pro, grok-4-latest, deepseek-reasoner) being asked for **CREATIVE, non-textbook** solutions to a stuck compact-model fit. We are bored of standard BSIM tweaks. Give us wild but defensible ideas.

---

## CONTEXT — where we are stuck

We are fitting Sebas's NS-RAM 2T cell (TSMC 130nm CMOS, vertical NPN + lateral nFET on shared p-body) to 33 measured per-bias I-V curves spanning V_G1 ∈ {0.2, 0.4, 0.6}, V_G2 ∈ {…}, with V_D swept and snapback (S-shape, negative differential resistance, latch) clearly visible. We have a Python "pyport" of ngspice's BSIM4 v4.8.3 + an explicit lateral/vertical BJT (Gummel-Poon style), Newton-with-arclength continuation, V_Sint substrate pinning.

After **50+ experiments** we currently sit at **cell-wide RMSE ≈ 1.7 dec**. Key observations:

1. **V_Sint runaway** was the dominant pathology — fixed by pinning V_Sint=0 (substrate-tap boundary). After fix: VG1=0.4 branch → 0.79 dec, VG1=0.6 branch → 1.09 dec. Real snapback knee is now visible in plots.
2. **VG1=0.2 branch stuck at 2.6 dec.** This is the sub-threshold high-V_D regime — measured current ~1e-8 A at the knee, but model predicts ~1e-16 A. **8 decades short.**
3. **BSIM4 GIDL (§6.2 of manual) already implemented** — doesn't help VG1=0.2. Gives I_GIDL ~1e-16 vs measured ~1e-8.
4. **Newton solver finds wrong root** — at many V_D points the f(V) = 0 equation has 8 real roots. Arc-length warm-start helps convergence but locks onto the low-current basin, missing the post-snapback high-current basin.
5. **Real snapback is a TRANSIENT** — measured device behaves as a relaxation oscillator with self-reset period ~400 ns. We are computing DC steady-state on a device that is fundamentally non-quasi-static.
6. **Mario's Ipos PWL fallback is single-V_G parameterized** — useful as scaffold but cannot fan in (V_G1, V_G2) jointly.

We have:
- Repo: `/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/`
- 33 measured I-V curves (per-bias .csv)
- Working ngspice + pyport BSIM4
- Sebas's 33 per-bias DC BSIM fits (one set of params per bias point — overfit per-bias, but valuable as scattered data)
- 4 hours subagent compute budget
- pyport access for arbitrary KCL/KVL solver hacks
- Verilog-A support via ngspice if we want it

---

## QUESTIONS

### Q1 — Unconventional combinations to close VG1=0.2

Pick any subset (or invent new ones). For each, give: physical justification, expected dec reduction at VG1=0.2, implementation pain, single deal-breaker risk.

Candidates we are considering (you may pick, combine, dismiss, or replace):

- (a) **Hybrid transient/DC** — slow V_D ramp (ns→ms) with body-capacitance integration, harvest steady envelope.
- (b) **Two-stage Newton** — converge on low-current root first, then warm-start jump (impact-ionization injection term) to the high-current root.
- (c) **2D PWL(V_G1, V_G2) surface** built from Sebas's 33 per-bias BSIM fits as scattered data — Delaunay / RBF interpolant inside the solver.
- (d) **Explicit lateral BJT** in parallel with the vertical NPN (drain-body-source acts as a parasitic lateral npn at high V_D). Gummel-Poon with its own β, V_AF.
- (e) **Full PNPN SCR latch-up model** — the 2T cell IS a thyristor: drain (n+) / body (p) / well (n) / substrate (p). Two-transistor regenerative feedback (α_npn + α_pnp ≥ 1 triggers latch).
- (f) **Self-heating with positive feedback** — T_local couples to I_S, V_T via thermal RC, drives runaway.
- (g) **Stochastic resonance** — inject Johnson/shot noise into body node, average over ensemble, force population to find high-current basin.
- (h) **Verilog-A custom compact model** — bypass BSIM4 entirely for this regime. Hand-roll subthreshold + impact-ionization + thyristor latch in 100 lines of Verilog-A, compile via ngspice OSDI/OpenVAF.

Also propose **wildcards we haven't listed**. Things like:
- Avalanche multiplication M(V_DB) with explicit Chynoweth ionization integral?
- Trap-assisted tunneling (TAT) — phonon-assisted via mid-gap defects?
- Floating-body kink + history-dependent V_BS?
- Holistic data-driven: train a small NN (5 inputs → 1 output) on the residual after BSIM, use as correction term?
- Differentiable solver (JAX) over the entire 33-bias dataset jointly to escape local minima?

### Q2 — What did we MISS?

Be blunt. What would YOU have tried first that we evidently didn't? Tell us specifically what is most embarrassing about our trajectory so far.

### Q3 — THE configuration to build next

Given: 4 hours subagent compute + pyport + 33 measured biases. Propose ONE concrete stack: which **3-4 fixes layered together**, in what order, expected residual RMSE (dec), and what would tell us early that it's failing.

---

**Format**: Be concrete. Numbers. Equations if relevant. No hedging mush. We will dispatch the top 3 ranked ideas across the 4-oracle panel as subagents tomorrow.
