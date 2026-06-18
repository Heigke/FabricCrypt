# O18 — Triple validation: is the NS-RAM PyTorch port + topology sweep actually correct?

## Context

We are building a PyTorch port of the BSIM4 v4.8.3 transistor model
("pyport") and using it to simulate a 2-transistor NS-RAM cell from
Sebastian Pazos (KAUST). The end goal is **super-fast large-scale
simulation of NS-RAM transistor configurations to find applications
where this substrate solves problems other chips cannot**.

The team is one engineer plus an autonomous Claude that has been
working continuously for ~24 hours. The engineer is worried that the
Claude has been over-confident ("vibande" — vibing without rigour) and
wants brutal third-party validation **before** more work is built on
top of these claims.

## What we want from you

A **brutal**, no-soft-pedal review of three claim groups, in this
order:

### Claim group 1: DC fit quality

The brief says the PyTorch BSIM4 port reproduces Sebas's measured I-V
family at:
- median log-RMSE = **0.799 dec** (after applying `bjt.Bf=2e4`,
  M3a optimum)
- mean = 1.40, max = 2.89, p90 = 2.58
- 25 of 33 biases evaluated (8 NaN'd because Sebas's CSV has K1=NaN
  for those rows)

Plus an independent ngspice cross-validation at L=0.234 µm,
Vgs=0.5/Vds=1.0/Vbs=0 shows pyport matches ngspice to **0.15 %** on
Id, gm, gds, Vdsat, Vth.

**Questions:**
1. Is the median log-RMSE of 0.80 actually a reasonable claim, or are
   we hiding bad biases? (See `z91g_stage6_summary.json` and the
   `fit_vs_meas.png` plot — please look at the plot, not just the
   numbers.)
2. The VG1=0.4 V row is at log-RMSE 2.52 dec — three orders of
   magnitude. We've documented this as a "wrong-Newton-root" issue
   (see `probe_v2_finding.md`). Is that a valid diagnosis, or are we
   rationalising a bug?
3. The ngspice cross-val (`stage6b_finding.md`,
   `test_instance_ags.sp`) says pyport matches ngspice to 0.15 %.
   Is the bias point we picked enough to validate, or is it a
   cherry-picked easy point? What additional ngspice probes would
   *prove* the BSIM4 evaluator is correct?

### Claim group 2: Large-scale topology scaling (z139)

We ran 6 topologies × 3 N values {100, 300, 800} × 3 seeds × 4 tasks
(MC, NARMA-10, XOR(τ=2), 4-class waveform). The pipeline is in
`z139_largescale_topology.py`. The reservoir uses **real
forward_2t_batched calls** (not a surrogate) at every timestep, with
`κ` recurrence into VG2.

Key results (`z139_summary.json` + `z139_midrun_analysis.md`):

| topology       | N=100 MC | N=300 MC | N=800 MC | N=800 XOR | N=800 WAVE | scale × |
|----------------|---------:|---------:|---------:|----------:|-----------:|--------:|
| RAND_GAUSS     | 1.42     | 1.50     | 1.87     | 0.53      | 0.47       | 1.31    |
| MESH_4N        | 1.87     | 2.40     | **3.29** | **0.91**  | 0.52       | 1.75    |
| ER_SPARSE      | 2.12     | 2.56     | 2.20     | 0.63      | 0.46       | 1.04    |
| WS_SMALLWORLD  | 1.66     | 2.44     | 2.94     | 0.85      | 0.51       | 1.77    |
| HUB_SPOKE      | 1.18     | 0.86     | 2.89     | 0.90      | **0.61**   | 2.45    |
| LAYERED        | 2.78     | 1.53     | 2.17     | 0.57      | 0.48       | 0.78    |

We claim:
- MESH_4N is the MC champion at large N
- HUB_SPOKE is the WAVE classification champion AND has the steepest
  scaling exponent
- LAYERED is anti-scaling (×0.78)
- ER_SPARSE plateaus at N=300 then collapses (collinearity)

**Questions:**
4. Are these results trustworthy with **2 valid seeds per condition**
   (third seed NaN-ed by a NARMA-10 ridge-selection bug — fix landed
   too late for this run)? Walk through which findings survive at n=2
   and which don't.
5. The HUB_SPOKE non-monotone behaviour (catastrophic at N=300 with
   MC=0.86, dominant at N=800 with MC=2.89) is unusual. Is this a
   real topology effect, or a randomness/n=2 artefact? What test
   would distinguish?
6. We compute spectral radius ρ=0.9 by scaling W to its largest
   eigenvalue (`z139.build_W` in `z139_largescale_topology.py`).
   For HUB_SPOKE the eigenvalue spectrum is dominated by the hub
   star, which means the effective dynamics may not be ρ=0.9 in the
   usual sense. Is the comparison fair?

### Claim group 3: Are we actually ready to scale?

The user's stated end-goal is "super-fast simulation of different
transistor configurations as NS-RAM at large scale to find what
other chips cannot solve". We have:
- PyTorch BSIM4 port that matches ngspice to 0.15 %
- 2T cell with median 0.80-dec fit to silicon
- Topology sweep shows MESH_4N MC=3.29 at N=800

**Questions:**
7. Given the **state of the model** (some biases at 2-3 dec error,
   dependent on Sebas's CSV overrides for the M2 transistor, with
   open questions on bipolar Bf physical realism), is the model
   *good enough* to make claims about "what other chips cannot
   solve"? Or do we need more rigour first?
8. What is the *minimum next step* you would require before believing
   any "NS-RAM beats Chip X at Task Y" claim? Be specific.
9. Three working demos exist: Pavlovian conditioning (LIF surrogate),
   Mackey-Glass forecasting (real cell, 64 cells, NRMSE 0.69), 3D
   phase-space animation (real cell, 3 biases). Look at the PNG of
   each. Are any of them **dishonest**? Which would survive a hostile
   reviewer?

## Output format

Three numbered sections matching the three claim groups. For each:
- A 2–3 sentence verdict ("CREDIBLE / MIXED / NOT CREDIBLE" + why)
- Specific quoted numbers from the artifacts to back the verdict
- A list of concrete things to check or fix before any further work

Then a final box:
- **Triple-checked status:** OK to continue / FIX listed below / HOLD

Be brief. The engineer needs to know whether to keep building on this
foundation or rip out one of the claim groups before anything is sent
to Mario / NRF / Sebas.

## Attached files

- `M3a_addendum_2026-05-03.md` — current proposed brief addendum
- `probe_v2_finding.md` — VG1=0.4 V catastrophe diagnosis
- `stage6b_finding.md` — pyport vs ngspice cross-validation
- `z139_midrun_analysis.md` — partial-data analysis (superseded but
  shows our reasoning evolution)
- `z91g_stage6_summary.json` — DC fit headline numbers
- `z91g_stage6_fit_vs_meas.png` — **the actual fit plot, please look**
- `z139_summary.json` — full topology sweep results
- `demo_pavlovian.png`, `demo_mackey_glass.png`, `demo_phase_space.png`
  — the three demos as PNGs
- `z91g_two_model_validation.py` — DC fit pipeline source
- `z139_largescale_topology.py` — topology sweep source
- `demo_mackey_glass.py` — chaos forecast demo source
- `probe_v2_vg04_catastrophe.py` — single-bias diagnostic probe source
- `test_instance_ags.sp` — ngspice cross-val deck

The fit plot, the scaling table, and the demo PNGs are the most
important things to **look at as images** before answering, especially
if you have multimodal capability.
