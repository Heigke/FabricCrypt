# MEP+DS PLAN — 2026-05-12 (Model Enhancement → Discovery Sweep)

**Trigger**: D1 distributed sweep just closed CONSERVATIVE PASS (NS-RAM
input neuron 84.45% on MNIST, statistical tie with Poisson 84.65%).
Brief v4.4 case strong. User wants to push for GROUNDBREAKING findings
without cheating, with oracle sanity-checks along the way.

**Posture**: Modelling enhancements first (MEP), then novel sweep
discoveries (DS) on the enhanced model. Oracle review at each milestone.

## Phase MEP — Model enhancement (priority order)

| ID | Enhancement | Rationale | Wall | Node |
|---|---|---|---|---|
| MEP-1 | Trilinear interp in z272 (replace nearest-neighbor) | nearest lookup wastes ~10% accuracy; trilinear is ~free on GPU | 1h | ikaros |
| MEP-2 | Dense surrogate v3: 10×15×8×20 = 24K pts, V_G2 to 0.60, V_b to 1.0 | extends domain so V_G2=0.35 isn't at edge; covers thick-ox region | 1h | ikaros |
| MEP-3 | V_Nwell as 5th surrogate axis {0.5, 1.0, 2.0, 2.5, 5.0} V | slide 17/18 says V_NW≥2.5V; current default 2.0V; unlocks N-well diode physics | 2h | daedalus |
| MEP-4 | Per-V_G1-regime separate surrogates | PMP-2 showed 3 regimes physically distinct; can't share interp | 1h | daedalus |
| MEP-5 | Voltage-dependent C(V_b) instead of constant | slide 21 mentions "voltage-dependent capacitance" of N-well diode | 1h | zgx |
| MEP-6 | Differentiable pyport in torch (Newton on GPU) | enables joint DC+transient gradient fit, replaces BBO | 3h | zgx |

**Pre-registered gates** per MEP step (locked BEFORE run):
- MEP-1: PASS if trilinear z272 reproduces nearest-neighbor result within
  ±1 pp AND max accuracy on D1 best cell improves by ≥0.5 pp
- MEP-2: PASS if 95% of new surrogate cells converge, domain coverage
  matches spec
- MEP-3: PASS if V_NW sweep on slide-21 condition shows monotonic effect
  on S_fire; identifies optimal V_NW
- MEP-4: PASS if per-regime surrogates yield ≥1 pp better D1 best cell
- MEP-5: PASS if voltage-dependent C(V_b) changes D1 best cell by ≥0.5 pp
  in either direction (informative)
- MEP-6: PASS if torch Newton converges on test point set within 1e-4 of
  numpy pyport (correctness gate, not speed gate)

## ORACLE-MILESTONE-1 (after MEP-1 to MEP-3)
- Build oracle packet with: latest sweep heatmap + per-V_G1 fit summary +
  C_b cross-validation + slide-21 reconstruction
- Send to 3 oracles (openai, gemini, grok) for CRITICISM
- Specific asks: "Are we fooling ourselves? What's a stronger
  falsification we should run? Where is the result fragile?"
- 25 min wall budget for oracle round

## Phase DS — Discovery sweeps (after MEP closes)

| ID | Discovery probe | Goal | Wall | Node |
|---|---|---|---|---|
| DS-1 | Network scale sweep N={64, 128, 256, 512, 1024} | does NS-RAM scale? (Brian2 used 100) | 30min | zgx |
| DS-2 | Multi-task sweep: MNIST + FashionMNIST + KMNIST + CIFAR10-grey | task-class generalization | 1h | zgx |
| DS-3 | Mixed-cell heterogeneous ensembles | does C_b heterogeneity add capacity? | 1h | daedalus |
| DS-4 | Hebbian / STDP online learning | local-learning vs frozen readout | 2h | daedalus |
| DS-5 | Spike-rate vs spike-timing coding | tests Mario's "VMM less evident with spikes" | 1h | zgx |
| DS-6 | Recurrent NS-RAM topology (z272+W_rec) | reservoir-style with NS-RAM nodes | 1h | ikaros |

**Pre-registered gates per DS** (locked BEFORE run):
- DS-1: PASS if accuracy plateau visible OR monotonic gain at N=1024
- DS-2: PASS if NS-RAM matches Poisson baseline on ≥2 of 4 datasets
- DS-3: PASS if mixed-C_b ensemble strictly beats homogeneous best by 1 pp non-overlap CI
- DS-4: PASS if online learning matches frozen readout within 2 pp; AMBITIOUS if exceeds it
- DS-5: PASS if either coding scheme reaches Poisson baseline; AMBITIOUS if one beats the other by 2+ pp non-overlap
- DS-6: PASS if recurrent NS-RAM reservoir reaches ≥ 78% MNIST (matches feedforward at C_b=8 fF)

## ORACLE-MILESTONE-2 (after DS closes)
- Build oracle packet with: full DS heatmaps + best findings + cross-checks
- Send to 3 oracles for FALSIFICATION attempts
- Specific asks: "What experiment would break the strongest claim?
  Where might we have unintentionally cherry-picked?"

## Phase Brief-v4.4 (if ANY DS gate AMBITIOUS-passes)

Open brief v4.4 with new headline. Re-pre-register oracle review with
the same 21-image set plus updated Mario+Sebas roadmap evidence.

## Cron schedule

| Cadence | Purpose |
|---|---|
| `19 */3 * * *` | MEP+DS campaign wake (every 3h while active) |
| `47 * * * *` | idle hourly (unchanged) |
| `27 */6 * * *` | oracle critique cycle (every 6h while campaign active) |
| `33 11,23 * * *` | 12h oracle feedback (unchanged) |
| `11 9,15,21 * * *` | 6h track audit (unchanged) |
| `43 4 * * *` | baseline watchdog (unchanged) |
| `23 6 * * *` | morning brief (unchanged) |
| `13 2 * * *` | daily synthesis (unchanged) |
| `7 0 * * *` | resource audit (unchanged) |
| `23 9 * * 1` | weekly review (unchanged) |
| `23 3 * * *` | GPU off-hours (unchanged) |

## Failure-honest exit conditions

- **PLAN CLOSED — null**: all MEP steps complete + 0 DS gates pass →
  v4.3 stays final, no new compute path advances
- **PLAN CLOSED — v4.4-incremental**: MEP yields ≥1 PASS but DS finds
  no groundbreaking → v4.4 with model-enhancement clarifications only
- **PLAN CLOSED — v4.4-breakthrough**: any DS AMBITIOUS PASS →
  v4.4 with new headline result + oracle-falsification review

## NO-CHEAT discipline (carried forward)

1. Every gate pre-registered before its run
2. Full grids/heatmaps reported, no cherry-picking
3. n ≥ 4 seeds per network condition (n=4 OK for fast iterations,
   n=10 for any v4.4 headline number)
4. Oracle critique consumed honestly — if 2/3 flag the same fragility,
   address it before claiming the result
5. Anything failing strict but interesting → log as INFORMATIVE_NULL,
   don't bury

Will dispatch MEP-1 + MEP-2 + MEP-3 in parallel as the first wave.
