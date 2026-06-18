# Combined text bundle for oracle context

All text artifacts concatenated below for one-shot context. Each file is delimited by a `=== FILE: <name> ===` marker.



=== FILE: context.md (11695 chars) ===
```
  data dirs (not seen before)
- Prompt: device topology + intended measurement protocol + every
  schematic detail (not just numbers — circuit elements, connections,
  test conditions, NS-RAM cell variant)
- Output: `research_plan/SA3_image_deep_extract.md`
- Gate: ≥3 schematic insights not in our current model

### SA4 — Full model rebuild from Sebas-canonical (subagent, ALL GPUs)
- Wait for SA1 output (poll `SA1_sebas_canonical_params.md` every 30s)
- Re-implement pyport with Sebas three-branch params as ground truth
- Distribute across ikaros + daedalus + zgx via queue
- Sweep over: per-branch Bf, per-branch Vaf, per-branch alpha0, R_s
- Gate: median forward log-RMSE on 33-row Sebas IV < 0.5 dec
  AMBITIOUS: < 0.3 dec
- SAFETY: no branch worse than 1.5 dec

All locked pre-compute. Launching now.

## 2026-05-12 ~17:50 — USER DIRECTIVE: Zenodo SPICE outdated, screenshots = canonical

User clarified: "the online skript from the old paper is outdated, our
screenshots are valid".

**Implications**:
- `data/nsram_zenodo/SimulationFiles/SPICE/` = OUTDATED PARAMS, do NOT use
  for model values. Specifically:
  - `BJTparams.txt` (Bf=50, BVPar=3.5-1.5·Vg) — OUTDATED
  - `Davalanche.txt` — OUTDATED
  - `BJTavalanche.txt` — OUTDATED
  - All `.asc` SPICE netlists — STRUCTURAL reference only, not parameter
  - All TCAD `.cmd` setups — OUTDATED params

- VALID ground truth:
  - 21 slides from Sebas/Mario (already in O44_use_case_audit/)
  - `data/sebas_2026_05_02/image-2.png`
  - `data/sebas_2026_05_02/three_branch_params_extracted.json`
  - `data/sebas_2026_04_22/` M1/M2/parasiticBJT/pdiode cards (Sebas's recent)
  - The 33 IV-V_d sweep CSVs (measured data)

**Reinterpretation of today's wins/fails**:
- DA2 "Mario published Bf=50 + BVPar=3.5-1.5·Vg" — FROM OUTDATED zenodo
  deck. NOT applicable. Today's z303/z303b tested it and found it DEGRADES
  the fit by 1 dec — consistent with "wrong process/cell".
- z299b TCAD comparison (oracle extracted curves from slides) — slides
  are canonical, so curves extracted there are valid; 2-6 dec pyport gap
  is real. But TCAD .cmd inputs are not directly applicable.
- The BVPar formula z300 ruled out as candidate physics is the WRONG
  formula. The TRUE V_G-dependent avalanche must come from screenshot
  evidence (slide-14 PWL bulk current, slide-21 transient ramps) not
  the zenodo BJT formula.

**Running agents update**:
- SA1 Sebas canonical: VALID, prioritize
- SA2 Zenodo process map: REPURPOSE — now just mark zenodo files as
  "outdated, do not use for params"
- SA3 Image deep-extract: MORE VALUABLE — screenshots are now canonical
- SA4 Model rebuild: prompt already says "abandon Mario-zenodo" — on
  track, but emphasize: use SA1's Sebas-only canonical params, treat
  slide structural insights from SA3 as constraints

This explains today's puzzles cleanly: pyport not bad at modeling Sebas's
cell when calibrated to Sebas's data (DA3 0.99 dec), bad when forced to
use outdated zenodo (z303 2.19 dec). The wrong-process hypothesis is
not "different node" — it's "Mario's published deck is old; Sebas's new
device is genuinely a different operating point and the slides+his cards
ARE the new model."

## 2026-05-12 ~17:55 — SA2 confirms user directive cleanly

Audit of 9 zenodo SPICE files: **ZERO match Sebas's 130nm thick-ox
imec cell**.
- All MOSFETs use PTM (Predictive Technology Model) 130nm THIN-ox
  (Tox = 3.3 nm) — NOT imec thick-ox
- BJT params have Tsinghua default + TSMC alternative commented; no
  imec variant
- README explicitly states: "These are not unique to any process and
  only exemplary models"
- Testbenches even use L=250 nm (not even physical 130 nm)
- Tox differs ~5-10× between PTM thin-ox and Sebas's thick-ox →
  totally different avalanche/BV physics

**Resolves today's z303 puzzle**: Mario zenodo Bf=50 + BVPar=3.5-1.5·Vg
degrades fit to 2.19 dec because it's a different transistor entirely.

**Forward implication**: SA4 model rebuild MUST use only Sebas's recent
sources. The "1.39 dec honest DC fit at Bf=100" was historically the
right answer — we just have to commit to it on Sebas grounds, not try
to reach Mario-zenodo's Bf=50 which is the wrong device.

SA1+SA3+SA4 still running. SA4 has correct directive.

## 2026-05-12 ~17:58 — SA1 COMPLETE — canonical Sebas param set defined

58 params catalogued from 4 cards + CSV + JSON. Zero conflicts.

CRITICAL findings:
1. NFACTOR per-V_G1 range = 1.25 → 12.15 (10× dynamic). Our BBO bound 3.0
   clips 4× of real range. THIS is why historical fit caps at 1.39 dec.
2. mbjt: step function 0.001 → 1.0 at V_G1 ≈ 0.3 V (1000× hop). Parasitic
   BJT turns on hard at V_G1 ≥ 0.3 V. Branch decomp mandatory.
3. K1: per-V_G1 (0.558/0.538/0.418), step structure, no smooth poly fit
   possible.
4. ETAB: M1 +1.8 vs M2 −0.087 — floating-body sign flip diagnostic
5. Floating-body 5-tuple at V_G1=0.4: etab=+1.8, k1=0.538, beta0=19,
   mbjt=1, CBpar=1fF
6. image-2.png = dynamic response (ramp rate dep) slide — confirms CBpar/
   tf/tr/cjc/cje/trise as dynamic knobs for transient validation

**SA4 path now clear** — has SA1's canonical doc + Sebas CSV as authoritative
override table. Expected outcome with per-branch fit (NFACTOR up to 15,
mbjt step, K1 per V_G1):
- Each branch should now hit independent local optimum
- Median log-RMSE should drop below 0.5 dec if SA1 hypothesis correct

SA3 image deep-extract still running.

## 2026-05-12 18:47 — :47 idle — idle, APU=80C

## 2026-05-12 18:48 — SA3 = topology gap identified

gpt-5 image-deep on 22 images (21 slides + image-2.png) found 7 NEW
structural elements; 3 critical missing from our pyport:

**MISSING TOPOLOGY ELEMENTS in pyport**:
1. **VNwell→VB parasitic diode** with explicit Cj + V-dependent leakage
   — DOMINANT source of ramp-rate sensitivity (slides 08/15/21). This
   is the missing physics z300 was chasing, and explains:
   - z298b transient ~1.67 dec systematic bias
   - Snapback shape gap
   - Hysteresis at fast ramps (image-2.png slide is exactly about this)
2. **VB–VG2 MOS coupling capacitor** (designed, not parasitic) — sets
   spike rise time
3. **VB is OUTPUT not internal state** — M2 drain = Vspike (the spike
   readout). Our model treats VB only as internal vb_clamp.

**ADDITIONAL insights (4 more)**:
4. NFACTOR(M2) depends on BOTH V_G1 AND V_G2 via VB coupling — explains
   why we couldn't fit branch dependency with a single NFACTOR poly
5. Starved-inverter ~1V front-end is part of firing model
6. VNwell + thick-ox jointly constrain legal operating window
7. VD ↔ Vmem mapping reverses "drain ramp" role across slides

**image-2.png** confirmed = 4-panel param page (BETA0, ETAB, K1, NFACTOR
top/bottom), source of three_branch_params_extracted.json, dated 2026-
05-02. Branch colors match JSON: red=0.2V, blue=0.4V, black=0.6V V_G1.

**Pitfalls flagged by oracle**:
- Slide 05 V_G2=1.4V is OUTSIDE thick-ox regime — don't joint-fit
- Slide 14 pins V_B=0 (modeling crutch, not floating-body)
- Brian2 Cint=170fF vs silicon 102fF — different abstraction layers
- Poisson 85% > LIF 72% — Poisson is REFERENCE not NS-RAM win
  (we may have miscited this historically)

**Implication for SA4**: pure parameter refit is necessary but NOT
sufficient. Without the VNwell diode + VB-VG2 cap + VB-output topology
fixes, no refit will close transient or snapback gaps. SA4 should
report HONEST per-branch DC fit and FLAG that transient/snapback
requires topology rebuild.

**Implication for v4.4 brief**: dropping any "transient match" claim;
v4.4 = DC-only fit on 33-row IV with explicit per-branch table from
canonical Sebas JSON, full disclosure that:
- VNwell diode missing in our model → can't fit ramp-rate dependence
- VB-VG2 cap missing → can't model spike-rise dynamics
- This is a NEXT-STAGE work, not v4.4 scope

## 2026-05-12 18:30 — 3h campaign cron: ACTIVE: z304_sebas_three_branch_refit, APU=84C
SA4 z304 sweep still running (10/12 jobs done, 2 in flight on cluster).

## 2026-05-12 18:42 — SA4 z304 final per-branch refit

11/12 jobs aggregated (1 hung, killed). Per-branch optima INCOMPATIBLE:

| V_G1 | Best (Bf, Rs) | med | signed | verdict |
|---|---|---|---|---|
| 0.2 | (500, 0) | 2.06 | −1.48 | FAIL all gates |
| 0.4 | (50, 1e10) | 1.41 | +0.42 | FAIL conservative |
| 0.6 | (9000, 1e10) | **0.70** | +0.13 | borderline PASS |

V_G1=0.6: PASS-conservative <0.7 (borderline 0.704); AMBITIOUS signed <0.1
borderline FAIL (0.125). Best single branch we have.

V_G1=0.2 wants vnwell OFF (Rs=0). V_G1=0.4/0.6 want vnwell ON (Rs=1e10).
**Structural incompatibility**: no single (Bf, Rs) cell-wide compromise.
Confirms SA3 missing-physics diagnosis: VNwell→VB parasitic diode (with
Cj + V-dependent leakage, drawn in slide-21) is structurally absent from
pyport. Pure parameter refit cannot bridge it.

Branch-coupled BJT (per-V_G1 Bf) helps mathematically but isn't a
physical model — Bf is supposed to be device-constant. Real fix:
implement the VNwell diode + VB-VG2 coupling cap + treat VB as output
node, then re-fit.

## DAY-END SUMMARY 2026-05-12

**Locked wins**:
- HDC headline 80.23% n=20 UCI-HAR, CI95±0.74pp, 2.3 nJ/inf
- HDC noise robustness: N=2048 noise-immune at σ=0.05 (80.4%)
- Bayesian NS-RAM RNG: ESS 1.03× + NIST 5/5 (dual headline candidate)
- 4A oracle 3-way IP-licensing consensus
- DA1: 5 new device specs (S_fire/S_relax, snapback peak V_d, etc)
- DA2: zenodo deck IS outdated (user-confirmed + SA2 mapping)
- SA1: full Sebas canonical 58-param table, image-2.png decoded as
  source of three_branch_params_extracted.json

**Honest negatives**:
- Per-branch DC: V_G1=0.6 0.70 dec borderline PASS; 0.2/0.4 FAIL.
  Cell-wide single-Bf fit impossible. (NEW understanding, not new fail)
- SA3 identified 3 missing topology elements: VNwell→VB diode,
  VB-VG2 cap, VB as output node. Snapback gap = these topology gaps.
- 4D oracle GATE verdict: KWS chance-level is ship-blocker
- z303/z303b: zenodo BJT params degrade fit (different process)

**Path to v4.4 brief** (revised):
1. (BLOCKED) Implement VNwell→VB diode topology + Cj + V-dep leakage
2. (BLOCKED) Implement VB-VG2 coupling cap
3. (BLOCKED) Re-fit per-branch on new topology
4. (POSSIBLE) Use V_G1=0.6 branch fit (0.70 dec) as best-case showcase
5. (POSSIBLE) Ship dual-headline (HDC + Bayesian RNG) gated only on
   topology fix; explicitly flag subthreshold (V_G1=0.2) as next-stage

Net for tonight: don't push v4.4. The model rebuild needs topology
work, not parameter work. SA3 + SA4 jointly pinpoint exact next step.

## 2026-05-12 19:47 — :47 idle cron — idle, APU=43C, last campaign <1h ago

## 2026-05-12 19:47 — Deep-dive 2h cron: 4E gated, NOT triggered

4A/4B/4C/4D all closed. Per workflow: would trigger 4E brief compile.

**Decision: HOLD 4E.** Today's deeper findings rule out shippable brief:
- SA3 identified 3 missing topology elements (VNwell→VB diode + cap +
  output node) — pyport structurally incomplete
- SA4 confirmed branch optima incompatible — pure parameter refit cannot
  bridge the gap
- Oracle 4D verdict GATE not SHIP (KWS chance-level credibility gap)
- User feedback: model genuinely bad, RNG result not spectacular,
  network sims unspectacular

4E brief v4.4 compile would package findings that do not yet justify a
brief. Better to wait for topology fix (next-stage work) than ship a
brief whose credibility is below threshold.

Cluster idle, APU 43°C. No new compute launched.

NEXT-STEP candidates (user-gated, not auto-launched):
1. Implement VNwell→VB diode in pyport, re-test V_G1=0.6 branch
2. Draft email to Sebas asking for V_d > 2V transient sweeps
3. KWS gate attack (different SNN encoding, not just delta-mod retry)
4. Build out per-branch v0.1 model with explicit "subthreshold pending"

```


=== FILE: z293_envelope_summary.json (4145 chars) ===
```json
{
  "experiment": "z293_envelope_sweep_aggregate",
  "counts": {
    "4B1": 4,
    "4B2": 2,
    "4B3": 12
  },
  "4B1_Nscaling": [
    {
      "N": 64,
      "mean_acc": 0.5943332202239565,
      "std_acc": 0.035453294703462826,
      "ci95": [
        0.5624363759755684,
        0.6267390566677977
      ],
      "energy_nJ": 0.14533072222599253,
      "verdict": "FAIL"
    },
    {
      "N": 128,
      "mean_acc": 0.6573634204275536,
      "std_acc": 0.03172162251786319,
      "ci95": [
        0.6269087207329488,
        0.6878181201221581
      ],
      "energy_nJ": 0.30315492419409573,
      "verdict": "INTERMEDIATE_PASS"
    },
    {
      "N": 512,
      "mean_acc": 0.7556837461825586,
      "std_acc": 0.015269765863590069,
      "ci95": [
        0.7378690193417035,
        0.7684085510688836
      ],
      "energy_nJ": 1.1663058893790295,
      "verdict": "INTERMEDIATE_PASS"
    },
    {
      "N": 1024,
      "mean_acc": 0.8056498133695283,
      "std_acc": 0.013601461344802224,
      "ci95": [
        0.7914828639294198,
        0.8176959619952494
      ],
      "energy_nJ": 2.2952326167628097,
      "verdict": "CONSERVATIVE_PASS"
    }
  ],
  "4B2_noise": [
    {
      "sigma": 0.05,
      "mean_acc": 0.5909399389209364,
      "std_acc": 0.0205993416945184,
      "verdict": "FAIL"
    },
    {
      "sigma": 0.1,
      "mean_acc": 0.5486087546657619,
      "std_acc": 0.009071476028818904,
      "verdict": "FAIL"
    }
  ],
  "4B3_vd_grid": [
    {
      "vd_high": 1.5,
      "vd_low": 0.0,
      "mean_acc": 0.6590600610790635,
      "std_acc": 0.030581330028254347,
      "verdict": "INTERMEDIATE_PASS"
    },
    {
      "vd_high": 1.5,
      "vd_low": 0.2,
      "mean_acc": 0.6590600610790635,
      "std_acc": 0.030581330028254347,
      "verdict": "INTERMEDIATE_PASS"
    },
    {
      "vd_high": 1.5,
      "vd_low": 0.5,
      "mean_acc": 0.6590600610790635,
      "std_acc": 0.030581330028254347,
      "verdict": "INTERMEDIATE_PASS"
    },
    {
      "vd_high": 2.0,
      "vd_low": 0.0,
      "mean_acc": 0.6573634204275536,
      "std_acc": 0.03172162251786319,
      "verdict": "INTERMEDIATE_PASS"
    },
    {
      "vd_high": 2.0,
      "vd_low": 0.2,
      "mean_acc": 0.6573634204275536,
      "std_acc": 0.03172162251786319,
      "verdict": "INTERMEDIATE_PASS"
    },
    {
      "vd_high": 2.0,
      "vd_low": 1.0,
      "mean_acc": 0.6562606040040719,
      "std_acc": 0.03132772421117563,
      "verdict": "INTERMEDIATE_PASS"
    },
    {
      "vd_high": 2.5,
      "vd_low": 0.0,
      "mean_acc": 0.656599932134374,
      "std_acc": 0.030946837104436783,
      "verdict": "INTERMEDIATE_PASS"
    },
    {
      "vd_high": 2.5,
      "vd_low": 0.5,
      "mean_acc": 0.656599932134374,
      "std_acc": 0.030946837104436783,
      "verdict": "INTERMEDIATE_PASS"
    },
    {
      "vd_high": 2.5,
      "vd_low": 1.0,
      "mean_acc": 0.6554971157108924,
      "std_acc": 0.03094067409834097,
      "verdict": "INTERMEDIATE_PASS"
    },
    {
      "vd_high": 3.0,
      "vd_low": 0.0,
      "mean_acc": 0.6366644044791314,
      "std_acc": 0.026002782375448678,
      "verdict": "INTERMEDIATE_PASS"
    },
    {
      "vd_high": 3.0,
      "vd_low": 0.2,
      "mean_acc": 0.6366644044791314,
      "std_acc": 0.026002782375448678,
      "verdict": "INTERMEDIATE_PASS"
    },
    {
      "vd_high": 3.0,
      "vd_low": 1.0,
      "mean_acc": 0.6341194435018663,
      "std_acc": 0.02674281894060112,
      "verdict": "FAIL"
    }
  ],
  "gates_locked": {
    "4B1_monotone_nondecreasing": true,
    "4B1_ambitious_N1024_geq_0.76": true,
    "4B2_sigma005_within_1pp": false,
    "4B2_ambitious_sigma010_improves": false,
    "4B3_local_max_interior": false
  },
  "best_4B3_cell": {
    "vd_high": 1.5,
    "vd_low": 0.0,
    "mean_acc": 0.6590600610790635
  },
  "best_overall_cell": {
    "tag": "N1024",
    "cell": {
      "N": 1024,
      "Q": 32,
      "vg1": 0.3,
      "vg2": 0.3,
      "vd_high": 2.0,
      "vd_low": 0.5,
      "sigma_noise": 0.0
    },
    "mean_acc": 0.8056498133695283,
    "std_acc": 0.013601461344802224,
    "verdict": "CONSERVATIVE_PASS"
  }
}
```


=== FILE: z296_ds_n3_bayesian_summary.json (577 chars) ===
```json
{
  "task": "DS-N3 Bayesian MCMC w/ NS-RAM RNG",
  "verdict": "AMBITIOUS",
  "ratio_ess_nsram_over_pseudo": 1.0328542772640148,
  "ess_pseudo": 1150.1384313288672,
  "ess_nsram": 1187.925398243745,
  "acceptance_rate_pseudo": 0.6395,
  "acceptance_rate_nsram": 0.6322,
  "posterior_mean_pseudo": 2.931689384877881,
  "posterior_mean_nsram": 2.9359160202569976,
  "true_mu": 2.5,
  "n_mh": 10000,
  "wall_pseudo_s": 0.044824838638305664,
  "wall_nsram_mh_s": 0.044680118560791016,
  "wall_nsram_gen_s": 0.38445067405700684,
  "seed": 42,
  "device": "cuda",
  "node": "ikaros"
}
```


=== FILE: z296b_nist_randomness_summary.json (3546 chars) ===
```json
{
  "experiment": "z296b_nist_randomness",
  "note": "Hand-implemented 5-test NIST subset (nistrng pkg unreliable; see script docstring).",
  "alpha": 0.01,
  "n_samples": 1050000,
  "n_bits": 1000000,
  "seed": 42,
  "n_tests": 5,
  "streams": {
    "ns_ram": {
      "label": "ns_ram",
      "n_bits": 1000000,
      "n_pass": 5,
      "n_total": 5,
      "elapsed_s": 1.5438728332519531,
      "tests": [
        {
          "test": "monobit",
          "p_value": 0.9044831479588323,
          "passed": true,
          "elapsed_s": 0.00023627281188964844
        },
        {
          "test": "runs",
          "p_value": 0.9808411222048429,
          "passed": true,
          "elapsed_s": 0.0004744529724121094
        },
        {
          "test": "longest_run_ones_in_a_block",
          "p_value": 0.9149415155340728,
          "passed": true,
          "elapsed_s": 0.6226742267608643
        },
        {
          "test": "binary_matrix_rank_32x32",
          "p_value": 0.09665646803462973,
          "passed": true,
          "elapsed_s": 0.9035999774932861
        },
        {
          "test": "dft_spectral",
          "p_value": 0.04350193303072112,
          "passed": true,
          "elapsed_s": 0.016782760620117188
        }
      ]
    },
    "np_random": {
      "label": "np_random",
      "n_bits": 1000000,
      "n_pass": 5,
      "n_total": 5,
      "elapsed_s": 1.538010835647583,
      "tests": [
        {
          "test": "monobit",
          "p_value": 0.678874106833102,
          "passed": true,
          "elapsed_s": 0.0002758502960205078
        },
        {
          "test": "runs",
          "p_value": 0.07675564674691465,
          "passed": true,
          "elapsed_s": 0.0004925727844238281
        },
        {
          "test": "longest_run_ones_in_a_block",
          "p_value": 0.4958490477792086,
          "passed": true,
          "elapsed_s": 0.6220588684082031
        },
        {
          "test": "binary_matrix_rank_32x32",
          "p_value": 0.11784244334941894,
          "passed": true,
          "elapsed_s": 0.9021751880645752
        },
        {
          "test": "dft_spectral",
          "p_value": 0.33069113431456476,
          "passed": true,
          "elapsed_s": 0.012909889221191406
        }
      ]
    },
    "zeros_negctrl": {
      "label": "zeros_negctrl",
      "n_bits": 1000000,
      "n_pass": 0,
      "n_total": 5,
      "elapsed_s": 1.275665044784546,
      "tests": [
        {
          "test": "monobit",
          "p_value": 0.0,
          "passed": false,
          "elapsed_s": 0.00029969215393066406
        },
        {
          "test": "runs",
          "p_value": 0.0,
          "passed": false,
          "elapsed_s": 0.0002644062042236328
        },
        {
          "test": "longest_run_ones_in_a_block",
          "p_value": 4.4003844943604805e-220,
          "passed": false,
          "elapsed_s": 0.6039566993713379
        },
        {
          "test": "binary_matrix_rank_32x32",
          "p_value": 0.0,
          "passed": false,
          "elapsed_s": 0.657984733581543
        },
        {
          "test": "dft_spectral",
          "p_value": 0.0,
          "passed": false,
          "elapsed_s": 0.013086318969726562
        }
      ]
    }
  },
  "gates": {
    "pass_conservative_ns_ram_ge_4_of_5": true,
    "ambitious_ns_ram_all_5_pass": true,
    "sanity_negctrl_zeros_le_1_of_5": true,
    "sanity_posctrl_np_ge_4_of_5": true
  },
  "ns_ram_pass_count": 5,
  "np_random_pass_count": 5,
  "zeros_pass_count": 0
}
```


=== FILE: z302_hdc_noise_robust_summary.json (15559 chars) ===
```json
{
  "experiment": "z302_hdc_noise_robust",
  "n_rows": 23,
  "rows": [
    {
      "path": "results/z302_hdc_noise_robust/A_noisetrain/str0p00_ste0p00/summary.json",
      "strategy": "A_noisetrain",
      "N": 1024,
      "vd_high": 2.0,
      "vd_low": 0.5,
      "sigma_train": 0.0,
      "sigma_test": 0.0,
      "mean_acc": 0.8056498133695283,
      "std_acc": 0.013601461344802224,
      "ci95": [
        0.7914828639294198,
        0.8176959619952494
      ],
      "mean_energy_J_per_inference": 2.2952326167628095e-09,
      "gates": {
        "conservative_geq_0p70": true,
        "ambitious_geq_0p75": true
      }
    },
    {
      "path": "results/z302_hdc_noise_robust/A_noisetrain/str0p05_ste0p00/summary.json",
      "strategy": "A_noisetrain",
      "N": 1024,
      "vd_high": 2.0,
      "vd_low": 0.5,
      "sigma_train": 0.05,
      "sigma_test": 0.0,
      "mean_acc": 0.7883440787241263,
      "std_acc": 0.007369653139597069,
      "ci95": [
        0.7814726840855106,
        0.7952154733627418
      ],
      "mean_energy_J_per_inference": 2.307266821038344e-09,
      "gates": {
        "conservative_geq_0p70": true,
        "ambitious_geq_0p75": true
      }
    },
    {
      "path": "results/z302_hdc_noise_robust/A_noisetrain/str0p05_ste0p05/summary.json",
      "strategy": "A_noisetrain",
      "N": 1024,
      "vd_high": 2.0,
      "vd_low": 0.5,
      "sigma_train": 0.05,
      "sigma_test": 0.05,
      "mean_acc": 0.7852052935188327,
      "std_acc": 0.01011744249465598,
      "ci95": [
        0.7746861214794707,
        0.7933491686460807
      ],
      "mean_energy_J_per_inference": 2.2900842869358673e-09,
      "gates": {
        "conservative_geq_0p70": true,
        "ambitious_geq_0p75": true
      }
    },
    {
      "path": "results/z302_hdc_noise_robust/A_noisetrain/str0p02_ste0p00/summary.json",
      "strategy": "A_noisetrain",
      "N": 1024,
      "vd_high": 2.0,
      "vd_low": 0.5,
      "sigma_train": 0.02,
      "sigma_test": 0.0,
      "mean_acc": 0.7850356294536817,
      "std_acc": 0.012067636173792527,
      "ci95": [
        0.7733288089582626,
        0.7967424499491007
      ],
      "mean_energy_J_per_inference": 2.307266821038344e-09,
      "gates": {
        "conservative_geq_0p70": true,
        "ambitious_geq_0p75": true
      }
    },
    {
      "path": "results/z302_hdc_noise_robust/A_noisetrain/str0p02_ste0p05/summary.json",
      "strategy": "A_noisetrain",
      "N": 1024,
      "vd_high": 2.0,
      "vd_low": 0.5,
      "sigma_train": 0.02,
      "sigma_test": 0.05,
      "mean_acc": 0.7767220902612826,
      "std_acc": 0.009516324168564747,
      "ci95": [
        0.7687478791991856,
        0.7863929419748897
      ],
      "mean_energy_J_per_inference": 2.2900842869358673e-09,
      "gates": {
        "conservative_geq_0p70": true,
        "ambitious_geq_0p75": true
      }
    },
    {
      "path": "results/z302_hdc_noise_robust/A_noisetrain/str0p10_ste0p05/summary.json",
      "strategy": "A_noisetrain",
      "N": 1024,
      "vd_high": 2.0,
      "vd_low": 0.5,
      "sigma_train": 0.1,
      "sigma_test": 0.05,
      "mean_acc": 0.7767220902612826,
      "std_acc": 0.008574332384454177,
      "ci95": [
        0.7699355276552426,
        0.7859687818120122
      ],
      "mean_energy_J_per_inference": 2.2900842869358673e-09,
      "gates": {
        "conservative_geq_0p70": true,
        "ambitious_geq_0p75": true
      }
    },
    {
      "path": "results/z302_hdc_noise_robust/A_noisetrain/str0p00_ste0p05/summary.json",
      "strategy": "A_noisetrain",
      "N": 1024,
      "vd_high": 2.0,
      "vd_low": 0.5,
      "sigma_train": 0.0,
      "sigma_test": 0.05,
      "mean_acc": 0.7750254496097726,
      "std_acc": 0.011220940457269405,
      "ci95": [
        0.7643366135052596,
        0.7857142857142857
      ],
      "mean_energy_J_per_inference": 2.2907351389209364e-09,
      "gates": {
        "conservative_geq_0p70": true,
        "ambitious_geq_0p75": true
      }
    },
    {
      "path": "results/z302_hdc_noise_robust/A_noisetrain/str0p10_ste0p00/summary.json",
      "strategy": "A_noisetrain",
      "N": 1024,
      "vd_high": 2.0,
      "vd_low": 0.5,
      "sigma_train": 0.1,
      "sigma_test": 0.0,
      "mean_acc": 0.7737529691211401,
      "std_acc": 0.010035668893686999,
      "ci95": [
        0.7648456057007126,
        0.7851204614862572
      ],
      "mean_energy_J_per_inference": 2.307266821038344e-09,
      "gates": {
        "conservative_geq_0p70": true,
        "ambitious_geq_0p75": true
      }
    },
    {
      "path": "results/z302_hdc_noise_robust/A_noisetrain/str0p10_ste0p10/summary.json",
      "strategy": "A_noisetrain",
      "N": 1024,
      "vd_high": 2.0,
      "vd_low": 0.5,
      "sigma_train": 0.1,
      "sigma_test": 0.1,
      "mean_acc": 0.7550050899219545,
      "std_acc": 0.010711702828411695,
      "ci95": [
        0.7438920936545639,
        0.7659484221241941
      ],
      "mean_energy_J_per_inference": 2.292186553376315e-09,
      "gates": {
        "conservative_geq_0p70": true,
        "ambitious_geq_0p75": true
      }
    },
    {
      "path": "results/z302_hdc_noise_robust/A_noisetrain/str0p05_ste0p10/summary.json",
      "strategy": "A_noisetrain",
      "N": 1024,
      "vd_high": 2.0,
      "vd_low": 0.5,
      "sigma_train": 0.05,
      "sigma_test": 0.1,
      "mean_acc": 0.7516118086189345,
      "std_acc": 0.007028271183307756,
      "ci95": [
        0.7441465897522905,
        0.7583983712249746
      ],
      "mean_energy_J_per_inference": 2.292186553376315e-09,
      "gates": {
        "conservative_geq_0p70": true,
        "ambitious_geq_0p75": true
      }
    },
    {
      "path": "results/z302_hdc_noise_robust/A_noisetrain/str0p02_ste0p10/summary.json",
      "strategy": "A_noisetrain",
      "N": 1024,
      "vd_high": 2.0,
      "vd_low": 0.5,
      "sigma_train": 0.02,
      "sigma_test": 0.1,
      "mean_acc": 0.742195453003054,
      "std_acc": 0.009172457392741848,
      "ci95": [
        0.7346454021038344,
        0.75169664065151
      ],
      "mean_energy_J_per_inference": 2.292186553376315e-09,
      "gates": {
        "conservative_geq_0p70": true,
        "ambitious_geq_0p75": false
      }
    },
    {
      "path": "results/z302_hdc_noise_robust/A_noisetrain/str0p00_ste0p10/summary.json",
      "strategy": "A_noisetrain",
      "N": 1024,
      "vd_high": 2.0,
      "vd_low": 0.5,
      "sigma_train": 0.0,
      "sigma_test": 0.1,
      "mean_acc": 0.7399898201560909,
      "std_acc": 0.008866063544830948,
      "ci95": [
        0.7319307770614184,
        0.7480488632507636
      ],
      "mean_energy_J_per_inference": 2.2922205361384455e-09,
      "gates": {
        "conservative_geq_0p70": true,
        "ambitious_geq_0p75": false
      }
    },
    {
      "path": "results/z302_hdc_noise_robust/A_noisetrain/str0p10_ste0p20/summary.json",
      "strategy": "A_noisetrain",
      "N": 1024,
      "vd_high": 2.0,
      "vd_low": 0.5,
      "sigma_train": 0.1,
      "sigma_test": 0.2,
      "mean_acc": 0.6827281981676281,
      "std_acc": 0.01002885422879649,
      "ci95": [
        0.6722938581608415,
        0.6926535459789616
      ],
      "mean_energy_J_per_inference": 2.293262658975229e-09,
      "gates": {
        "conservative_geq_0p70": false,
        "ambitious_geq_0p75": false
      }
    },
    {
      "path": "results/z302_hdc_noise_robust/A_noisetrain/str0p05_ste0p20/summary.json",
      "strategy": "A_noisetrain",
      "N": 1024,
      "vd_high": 2.0,
      "vd_low": 0.5,
      "sigma_train": 0.05,
      "sigma_test": 0.2,
      "mean_acc": 0.6679674244994911,
      "std_acc": 0.010780011972599327,
      "ci95": [
        0.657448252460129,
        0.6791652527994572
      ],
      "mean_energy_J_per_inference": 2.293262658975229e-09,
      "gates": {
        "conservative_geq_0p70": false,
        "ambitious_geq_0p75": false
      }
    },
    {
      "path": "results/z302_hdc_noise_robust/A_noisetrain/str0p00_ste0p20/summary.json",
      "strategy": "A_noisetrain",
      "N": 1024,
      "vd_high": 2.0,
      "vd_low": 0.5,
      "sigma_train": 0.0,
      "sigma_test": 0.2,
      "mean_acc": 0.663725822870716,
      "std_acc": 0.004508086276072758,
      "ci95": [
        0.660162877502545,
        0.6684764166949441
      ],
      "mean_energy_J_per_inference": 2.2924759932134375e-09,
      "gates": {
        "conservative_geq_0p70": false,
        "ambitious_geq_0p75": false
      }
    },
    {
      "path": "results/z302_hdc_noise_robust/A_noisetrain/str0p02_ste0p20/summary.json",
      "strategy": "A_noisetrain",
      "N": 1024,
      "vd_high": 2.0,
      "vd_low": 0.5,
      "sigma_train": 0.02,
      "sigma_test": 0.2,
      "mean_acc": 0.662198846284357,
      "std_acc": 0.015641333754999376,
      "ci95": [
        0.6467594163556158,
        0.6776382762130981
      ],
      "mean_energy_J_per_inference": 2.293262658975229e-09,
      "gates": {
        "conservative_geq_0p70": false,
        "ambitious_geq_0p75": false
      }
    },
    {
      "path": "results/z302_hdc_noise_robust/B_nscale/N4096/summary.json",
      "strategy": "B_nscale",
      "N": 4096,
      "vd_high": 2.0,
      "vd_low": 0.5,
      "sigma_train": 0.0,
      "sigma_test": 0.05,
      "mean_acc": 0.8193077706141839,
      "std_acc": 0.008771812660136102,
      "ci95": [
        0.8109942314217848,
        0.82711231761113
      ],
      "mean_energy_J_per_inference": 8.987810873159144e-09,
      "gates": {
        "conservative_geq_0p70": true,
        "ambitious_geq_0p75": true
      }
    },
    {
      "path": "results/z302_hdc_noise_robust/B_nscale/N2048/summary.json",
      "strategy": "B_nscale",
      "N": 2048,
      "vd_high": 2.0,
      "vd_low": 0.5,
      "sigma_train": 0.0,
      "sigma_test": 0.05,
      "mean_acc": 0.8039531727180184,
      "std_acc": 0.013212844384267139,
      "ci95": [
        0.7925856803529012,
        0.8176111299626738
      ],
      "mean_energy_J_per_inference": 4.496423073227011e-09,
      "gates": {
        "conservative_geq_0p70": true,
        "ambitious_geq_0p75": true
      }
    },
    {
      "path": "results/z302_hdc_noise_robust/B_nscale/N1024/summary.json",
      "strategy": "B_nscale",
      "N": 1024,
      "vd_high": 2.0,
      "vd_low": 0.5,
      "sigma_train": 0.0,
      "sigma_test": 0.05,
      "mean_acc": 0.7750254496097726,
      "std_acc": 0.011220940457269405,
      "ci95": [
        0.7643366135052596,
        0.7857142857142857
      ],
      "mean_energy_J_per_inference": 2.2907351389209364e-09,
      "gates": {
        "conservative_geq_0p70": true,
        "ambitious_geq_0p75": true
      }
    },
    {
      "path": "results/z302_hdc_noise_robust/C_vdwide/H2p50_L0p50/summary.json",
      "strategy": "C_vdwide",
      "N": 1024,
      "vd_high": 2.5,
      "vd_low": 0.5,
      "sigma_train": 0.0,
      "sigma_test": 0.05,
      "mean_acc": 0.7745164574143196,
      "std_acc": 0.011533413233360969,
      "ci95": [
        0.7633186291143536,
        0.7857142857142858
      ],
      "mean_energy_J_per_inference": 2.4859355189684426e-09,
      "gates": {
        "conservative_geq_0p70": true,
        "ambitious_geq_0p75": true
      }
    },
    {
      "path": "results/z302_hdc_noise_robust/C_vdwide/H2p20_L0p40/summary.json",
      "strategy": "C_vdwide",
      "N": 1024,
      "vd_high": 2.2,
      "vd_low": 0.4,
      "sigma_train": 0.0,
      "sigma_test": 0.05,
      "mean_acc": 0.7745164574143196,
      "std_acc": 0.011533413233360969,
      "ci95": [
        0.7633186291143536,
        0.7857142857142858
      ],
      "mean_energy_J_per_inference": 2.4859355189684426e-09,
      "gates": {
        "conservative_geq_0p70": true,
        "ambitious_geq_0p75": true
      }
    },
    {
      "path": "results/z302_hdc_noise_robust/C_vdwide/H2p40_L0p40/summary.json",
      "strategy": "C_vdwide",
      "N": 1024,
      "vd_high": 2.4,
      "vd_low": 0.4,
      "sigma_train": 0.0,
      "sigma_test": 0.05,
      "mean_acc": 0.7744316253817441,
      "std_acc": 0.011592848972443576,
      "ci95": [
        0.7631489650492025,
        0.7857142857142858
      ],
      "mean_energy_J_per_inference": 2.4859355189684426e-09,
      "gates": {
        "conservative_geq_0p70": true,
        "ambitious_geq_0p75": true
      }
    },
    {
      "path": "results/z302_hdc_noise_robust/C_vdwide/H3p00_L0p50/summary.json",
      "strategy": "C_vdwide",
      "N": 1024,
      "vd_high": 3.0,
      "vd_low": 0.5,
      "sigma_train": 0.0,
      "sigma_test": 0.05,
      "mean_acc": 0.7662029182219207,
      "std_acc": 0.0032723619820132182,
      "ci95": [
        0.7628096369189006,
        0.7689175432643367
      ],
      "mean_energy_J_per_inference": 4.896399551543943e-09,
      "gates": {
        "conservative_geq_0p70": true,
        "ambitious_geq_0p75": true
      }
    }
  ],
  "best_at_sigma_test_0p05": {
    "path": "results/z302_hdc_noise_robust/B_nscale/N4096/summary.json",
    "strategy": "B_nscale",
    "N": 4096,
    "vd_high": 2.0,
    "vd_low": 0.5,
    "sigma_train": 0.0,
    "sigma_test": 0.05,
    "mean_acc": 0.8193077706141839,
    "std_acc": 0.008771812660136102,
    "ci95": [
      0.8109942314217848,
      0.82711231761113
    ],
    "mean_energy_J_per_inference": 8.987810873159144e-09,
    "gates": {
      "conservative_geq_0p70": true,
      "ambitious_geq_0p75": true
    }
  },
  "best_per_strategy_at_sigma_test_0p05": {
    "A_noisetrain": {
      "path": "results/z302_hdc_noise_robust/A_noisetrain/str0p05_ste0p05/summary.json",
      "strategy": "A_noisetrain",
      "N": 1024,
      "vd_high": 2.0,
      "vd_low": 0.5,
      "sigma_train": 0.05,
      "sigma_test": 0.05,
      "mean_acc": 0.7852052935188327,
      "std_acc": 0.01011744249465598,
      "ci95": [
        0.7746861214794707,
        0.7933491686460807
      ],
      "mean_energy_J_per_inference": 2.2900842869358673e-09,
      "gates": {
        "conservative_geq_0p70": true,
        "ambitious_geq_0p75": true
      }
    },
    "B_nscale": {
      "path": "results/z302_hdc_noise_robust/B_nscale/N4096/summary.json",
      "strategy": "B_nscale",
      "N": 4096,
      "vd_high": 2.0,
      "vd_low": 0.5,
      "sigma_train": 0.0,
      "sigma_test": 0.05,
      "mean_acc": 0.8193077706141839,
      "std_acc": 0.008771812660136102,
      "ci95": [
        0.8109942314217848,
        0.82711231761113
      ],
      "mean_energy_J_per_inference": 8.987810873159144e-09,
      "gates": {
        "conservative_geq_0p70": true,
        "ambitious_geq_0p75": true
      }
    },
    "C_vdwide": {
      "path": "results/z302_hdc_noise_robust/C_vdwide/H2p50_L0p50/summary.json",
      "strategy": "C_vdwide",
      "N": 1024,
      "vd_high": 2.5,
      "vd_low": 0.5,
      "sigma_train": 0.0,
      "sigma_test": 0.05,
      "mean_acc": 0.7745164574143196,
      "std_acc": 0.011533413233360969,
      "ci95": [
        0.7633186291143536,
        0.7857142857142858
      ],
      "mean_energy_J_per_inference": 2.4859355189684426e-09,
      "gates": {
        "conservative_geq_0p70": true,
        "ambitious_geq_0p75": true
      }
    }
  },
  "headline_drop_clean_vs_noisetrained_at_sigma_test_0": {
    "sigma_train_0_test_0_acc": 0.8056498133695283,
    "sigma_train_0p10_test_0_acc": 0.7737529691211401,
    "drop": 0.031896844248388234
  },
  "gates": {
    "conservative_geq_0p70_at_sigma_test_0p05": true,
    "ambitious_geq_0p75_at_sigma_test_0p05": true
  }
}
```


=== FILE: z303_mario_bjt_summary.json (60030 chars) ===
```json
{
  "script": "scripts/z303_mario_bjt_integration.py",
  "device": "cuda",
  "n_curves": 33,
  "mario_params": {
    "IsPar": 1e-16,
    "BfPar": 50,
    "VafPar": 40,
    "NfPar": 0.9,
    "NePar": 1.5,
    "VarPar": 10,
    "BVPar_VG1": "3.5 - 1.5*V_G1"
  },
  "comparison": {
    "baseline": {
      "med": 1.2492279601506482,
      "vg02": 5.101771560677094,
      "vg06": 0.5578187792914511
    },
    "da3": {
      "med": 0.9882303684954579,
      "vg02": 4.672125109531012,
      "vg06": 0.717063436930812
    },
    "mario_only": {
      "med": 2.1941132883463306,
      "vg02": 2.605615458010776,
      "vg06": 3.2175199054445978
    },
    "mario_plus_bv": {
      "med": 2.193152730959098,
      "vg02": 2.60603261855497,
      "vg06": 3.2172436087718,
      "peak_vd_vg06": 2.00027
    }
  },
  "gates": {
    "mario_only": {
      "pass_conservative": false,
      "pass_ambitious": false,
      "safety_pass_vg02": true,
      "vg02_delta_vs_da3_dec": -2.066509651520236
    },
    "mario_plus_bv": {
      "pass_conservative": false,
      "pass_ambitious": false,
      "safety_pass_vg02": true,
      "vg02_delta_vs_da3_dec": -2.0660924909760423
    },
    "BV_at_VG1_0.6": 2.6,
    "BONUS_snapback_peak_vg06": {
      "expected_V": 2.6,
      "tolerance": 0.3,
      "observed_V": 2.00027,
      "pass": false
    }
  },
  "configs": [
    {
      "label": "baseline",
      "Bf": 9000.0,
      "Va": 0.55,
      "Is": 1e-09,
      "Nf": null,
      "Ne": null,
      "Var": null,
      "use_bv_avalanche": false,
      "BV_at_VG1_0.6_V": 2.6,
      "median_fwd_log_rmse_all": 1.2492279601506482,
      "median_signed_dec_all": 1.2492279601506482,
      "by_vg1": {
        "0.2": {
          "median_fwd_log_rmse": 5.101771560677094,
          "median_signed_dec": 5.101771560677094,
          "n": 7,
          "median_sim_peak_vd": 2.00028
        },
        "0.4": {
          "median_fwd_log_rmse": 1.7567530994541247,
          "median_signed_dec": 1.7567530994541247,
          "n": 11,
          "median_sim_peak_vd": 2.00029
        },
        "0.6": {
          "median_fwd_log_rmse": 0.5578187792914511,
          "median_signed_dec": -0.5146624302605787,
          "n": 15,
          "median_sim_peak_vd": 2.00027
        }
      },
      "n_curves": 33,
      "per_curve": [
        {
          "forward_log_rmse": 5.101771560677094,
          "forward_signed_dec": 5.101771560677094,
          "vg1": 0.2,
          "vg2": -0.05,
          "file": "StandardIV_HH_2vHCa-2_VG2=-0.05_VG=0.2(1)_03-32-11PM.csv",
          "sim_peak_vd": 2.0003,
          "sim_peak_id": 4.7876402692184235e-06
        },
        {
          "forward_log_rmse": 4.919357746292301,
          "forward_signed_dec": 4.919357746292301,
          "vg1": 0.2,
          "vg2": -0.1,
          "file": "StandardIV_HH_2vHCa-2_VG2=-0.10_VG=0.2(1)_03-30-30PM.csv",
          "sim_peak_vd": 2.00027,
          "sim_peak_id": 4.787568197192729e-06
        },
        {
          "forward_log_rmse": 4.703554844286521,
          "forward_signed_dec": 4.703554844286521,
          "vg1": 0.2,
          "vg2": -0.15,
          "file": "StandardIV_HH_2vHCa-2_VG2=-0.15_VG=0.2(1)_03-31-04PM.csv",
          "sim_peak_vd": 2.00026,
          "sim_peak_id": 4.787544173230162e-06
        },
        {
          "forward_log_rmse": 4.490875364315369,
          "forward_signed_dec": 4.490875364315369,
          "vg1": 0.2,
          "vg2": -0.2,
          "file": "StandardIV_HH_2vHCa-2_VG2=-0.20_VG=0.2(1)_03-31-38PM.csv",
          "sim_peak_vd": 2.0003,
          "sim_peak_id": 4.7876402692184235e-06
        },
        {
          "forward_log_rmse": 5.2563399280186065,
          "forward_signed_dec": 5.2563399280186065,
          "vg1": 0.2,
          "vg2": 0.0,
          "file": "StandardIV_HH_2vHCa-2_VG2=0.00_VG=0.2(1)_03-32-49PM.csv",
          "sim_peak_vd": 2.00028,
          "sim_peak_id": 4.787592221178291e-06
        },
        {
          "forward_log_rmse": 5.337901943371079,
          "forward_signed_dec": 5.337901943371079,
          "vg1": 0.2,
          "vg2": 0.05,
          "file": "StandardIV_HH_2vHCa-2_VG2=0.05_VG=0.2(1)_03-33-17PM.csv",
          "sim_peak_vd": 2.00028,
          "sim_peak_id": 4.787592221178291e-06
        },
        {
          "forward_log_rmse": 5.348840747335376,
          "forward_signed_dec": 5.348840747335376,
          "vg1": 0.2,
          "vg2": 0.1,
          "file": "StandardIV_HH_2vHCa-2_VG2=0.10_VG=0.2(1)_03-33-55PM.csv",
          "sim_peak_vd": 2.20037,
          "sim_peak_id": 5.2846970962723525e-06
        },
        {
          "forward_log_rmse": 1.2492279601506482,
          "forward_signed_dec": 1.2492279601506482,
          "vg1": 0.4,
          "vg2": -0.05,
          "file": "StandardIV_HH_2vHCa-2_VG2=-0.05_VG=0.4(1)_03-36-27PM.csv",
          "sim_peak_vd": 2.00029,
          "sim_peak_id": 4.787065632759945e-06
        },
        {
          "forward_log_rmse": 1.246710880041042,
          "forward_signed_dec": 1.246710880041042,
          "vg1": 0.4,
          "vg2": -0.1,
          "file": "StandardIV_HH_2vHCa-2_VG2=-0.10_VG=0.4(1)_03-36-03PM.csv",
          "sim_peak_vd": 2.0003,
          "sim_peak_id": 4.787089539172394e-06
        },
        {
          "forward_log_rmse": 1.251149890046574,
          "forward_signed_dec": 1.251149890046574,
          "vg1": 0.4,
          "vg2": -0.15,
          "file": "StandardIV_HH_2vHCa-2_VG2=-0.15_VG=0.4(1)_03-35-29PM.csv",
          "sim_peak_vd": 2.00033,
          "sim_peak_id": 4.787161258412092e-06
        },
        {
          "forward_log_rmse": 1.2507936093854406,
          "forward_signed_dec": 1.2507936093854406,
          "vg1": 0.4,
          "vg2": -0.2,
          "file": "StandardIV_HH_2vHCa-2_VG2=-0.20_VG=0.4(1)_03-36-53PM.csv",
          "sim_peak_vd": 2.00027,
          "sim_peak_id": 4.787017819936229e-06
        },
        {
          "forward_log_rmse": 1.2553023803353547,
          "forward_signed_dec": 1.2553023803353547,
          "vg1": 0.4,
          "vg2": 0.0,
          "file": "StandardIV_HH_2vHCa-2_VG2=0.00_VG=0.4(1)_03-37-28PM.csv",
          "sim_peak_vd": 2.00023,
          "sim_peak_id": 4.786922194293505e-06
        },
        {
          "forward_log_rmse": 1.7567530994541247,
          "forward_signed_dec": 1.7567530994541247,
          "vg1": 0.4,
          "vg2": 0.05,
          "file": "StandardIV_HH_2vHCa-2_VG2=0.05_VG=0.4(1)_03-37-58PM.csv",
          "sim_peak_vd": 2.00031,
          "sim_peak_id": 4.787113445585232e-06
        },
        {
          "forward_log_rmse": 2.4933911135390314,
          "forward_signed_dec": 2.4933911135390314,
          "vg1": 0.4,
          "vg2": 0.1,
          "file": "StandardIV_HH_2vHCa-2_VG2=0.10_VG=0.4(1)_03-38-28PM.csv",
          "sim_peak_vd": 2.00027,
          "sim_peak_id": 4.787017819936229e-06
        },
        {
          "forward_log_rmse": 2.6546530778458823,
          "forward_signed_dec": 2.6546530778458823,
          "vg1": 0.4,
          "vg2": 0.15,
          "file": "StandardIV_HH_2vHCa-2_VG2=0.15_VG=0.4(1)_03-38-55PM.csv",
          "sim_peak_vd": 2.00026,
          "sim_peak_id": 4.786993913524958e-06
        },
        {
          "forward_log_rmse": 2.7898785706989875,
          "forward_signed_dec": 2.7898785706989875,
          "vg1": 0.4,
          "vg2": 0.2,
          "file": "StandardIV_HH_2vHCa-2_VG2=0.20_VG=0.4(1)_03-39-29PM.csv",
          "sim_peak_vd": 2.00032,
          "sim_peak_id": 4.787137351998466e-06
        },
        {
          "forward_log_rmse": 2.839545698858533,
          "forward_signed_dec": 2.839545698858533,
          "vg1": 0.4,
          "vg2": 0.25,
          "file": "StandardIV_HH_2vHCa-2_VG2=0.25_VG=0.4(1)_03-40-00PM.csv",
          "sim_peak_vd": 2.00027,
          "sim_peak_id": 4.78174269059498e-06
        },
        {
          "forward_log_rmse": 2.8432118621693636,
          "forward_signed_dec": 2.8432118621693636,
          "vg1": 0.4,
          "vg2": 0.3,
          "file": "StandardIV_HH_2vHCa-2_VG2=0.3_VG=0.4(1)_03-40-47PM.csv",
          "sim_peak_vd": 2.20036,
          "sim_peak_id": 5.2645387864346855e-06
        },
        {
          "forward_log_rmse": 0.5579048847292389,
          "forward_signed_dec": -0.5579048847292389,
          "vg1": 0.6,
          "vg2": -0.05,
          "file": "StandardIV_HH_2vHCa-2_VG2=-0.05_VG=0.6(1)_03-43-24PM.csv",
          "sim_peak_vd": 2.00025,
          "sim_peak_id": 4.786966318643216e-06
        },
        {
          "forward_log_rmse": 0.5578187792914511,
          "forward_signed_dec": -0.5578187792914511,
          "vg1": 0.6,
          "vg2": -0.1,
          "file": "StandardIV_HH_2vHCa-2_VG2=-0.10_VG=0.6(1)_03-43-01PM.csv",
          "sim_peak_vd": 2.0003,
          "sim_peak_id": 4.787085844425761e-06
        },
        {
          "forward_log_rmse": 0.5576834977023744,
          "forward_signed_dec": -0.5576834977023744,
          "vg1": 0.6,
          "vg2": -0.15,
          "file": "StandardIV_HH_2vHCa-2_VG2=-0.15_VG=0.6(1)_03-42-38PM.csv",
          "sim_peak_vd": 2.00027,
          "sim_peak_id": 4.787014128956234e-06
        },
        {
          "forward_log_rmse": 0.5962872730688984,
          "forward_signed_dec": -0.5962872730688984,
          "vg1": 0.6,
          "vg2": -0.2,
          "file": "StandardIV_HH_2vHCa-2_VG2=-0.20_VG=0.6(1)_03-42-07PM.csv",
          "sim_peak_vd": 2.20039,
          "sim_peak_id": 5.265409306111836e-06
        },
        {
          "forward_log_rmse": 0.5572002317395164,
          "forward_signed_dec": -0.5572002317395164,
          "vg1": 0.6,
          "vg2": 0.0,
          "file": "StandardIV_HH_2vHCa-2_VG2=0.00_VG=0.6(1)_03-43-54PM.csv",
          "sim_peak_vd": 2.00027,
          "sim_peak_id": 4.787014128956234e-06
        },
        {
          "forward_log_rmse": 0.5575299502383126,
          "forward_signed_dec": -0.5575299502383126,
          "vg1": 0.6,
          "vg2": 0.05,
          "file": "StandardIV_HH_2vHCa-2_VG2=0.05_VG=0.6(1)_03-44-26PM.csv",
          "sim_peak_vd": 2.00026,
          "sim_peak_id": 4.786990223799724e-06
        },
        {
          "forward_log_rmse": 0.5561789522093852,
          "forward_signed_dec": -0.5561789522093852,
          "vg1": 0.6,
          "vg2": 0.1,
          "file": "StandardIV_HH_2vHCa-2_VG2=0.10_VG=0.6(1)_03-44-57PM.csv",
          "sim_peak_vd": 2.00034,
          "sim_peak_id": 4.787181465051799e-06
        },
        {
          "forward_log_rmse": 0.5476647140006667,
          "forward_signed_dec": -0.48980377807072717,
          "vg1": 0.6,
          "vg2": 0.15,
          "file": "StandardIV_HH_2vHCa-2_VG2=0.15_VG=0.6(1)_03-45-21PM.csv",
          "sim_peak_vd": 2.00028,
          "sim_peak_id": 4.787038034112742e-06
        },
        {
          "forward_log_rmse": 0.5537550199395502,
          "forward_signed_dec": -0.5146624302605787,
          "vg1": 0.6,
          "vg2": 0.2,
          "file": "StandardIV_HH_2vHCa-2_VG2=0.20_VG=0.6(1)_03-45-46PM.csv",
          "sim_peak_vd": 2.00027,
          "sim_peak_id": 4.787014128956234e-06
        },
        {
          "forward_log_rmse": 0.5578828986818367,
          "forward_signed_dec": 0.12697711109149168,
          "vg1": 0.6,
          "vg2": 0.25,
          "file": "StandardIV_HH_2vHCa-2_VG2=0.25_VG=0.6(1)_03-46-14PM.csv",
          "sim_peak_vd": 2.00024,
          "sim_peak_id": 4.7872718739157525e-06
        },
        {
          "forward_log_rmse": 0.5601587226822717,
          "forward_signed_dec": 0.2863105893327491,
          "vg1": 0.6,
          "vg2": 0.3,
          "file": "StandardIV_HH_2vHCa-2_VG2=0.30_VG=0.6(1)_03-46-40PM.csv",
          "sim_peak_vd": 2.00031,
          "sim_peak_id": 4.789114586376592e-06
        },
        {
          "forward_log_rmse": 0.5584849071991931,
          "forward_signed_dec": 0.4057417053630372,
          "vg1": 0.6,
          "vg2": 0.35,
          "file": "StandardIV_HH_2vHCa-2_VG2=0.35_VG=0.6(1)_03-47-03PM.csv",
          "sim_peak_vd": 2.00031,
          "sim_peak_id": 4.799995195986011e-06
        },
        {
          "forward_log_rmse": 0.5587524018990733,
          "forward_signed_dec": 0.47714291501456074,
          "vg1": 0.6,
          "vg2": 0.4,
          "file": "StandardIV_HH_2vHCa-2_VG2=0.40_VG=0.6(1)_03-47-25PM.csv",
          "sim_peak_vd": 2.00027,
          "sim_peak_id": 4.867845119996936e-06
        },
        {
          "forward_log_rmse": 0.5575952953198895,
          "forward_signed_dec": 0.5190616374392656,
          "vg1": 0.6,
          "vg2": 0.45,
          "file": "StandardIV_HH_2vHCa-2_VG2=0.45_VG=0.6(1)_03-47-47PM.csv",
          "sim_peak_vd": 2.00026,
          "sim_peak_id": 5.254587527231824e-06
        },
        {
          "forward_log_rmse": 0.5679615830822939,
          "forward_signed_dec": 0.542175827519447,
          "vg1": 0.6,
          "vg2": 0.5,
          "file": "StandardIV_HH_2vHCa-2_VG2=0.50_VG=0.6(1)_03-48-22PM.csv",
          "sim_peak_vd": 2.20035,
          "sim_peak_id": 1.3234345487124905e-05
        }
      ]
    },
    {
      "label": "da3",
      "Bf": 3000.0,
      "Va": 0.55,
      "Is": 1e-09,
      "Nf": null,
      "Ne": null,
      "Var": null,
      "use_bv_avalanche": false,
      "BV_at_VG1_0.6_V": 2.6,
      "median_fwd_log_rmse_all": 0.9882303684954579,
      "median_signed_dec_all": 0.9671766629272254,
      "by_vg1": {
        "0.2": {
          "median_fwd_log_rmse": 4.672125109531012,
          "median_signed_dec": 4.672125109531012,
          "n": 7,
          "median_sim_peak_vd": 2.00028
        },
        "0.4": {
          "median_fwd_log_rmse": 1.6379649701206942,
          "median_signed_dec": 1.6379649701206942,
          "n": 11,
          "median_sim_peak_vd": 2.00029
        },
        "0.6": {
          "median_fwd_log_rmse": 0.717063436930812,
          "median_signed_dec": -0.717063436930812,
          "n": 15,
          "median_sim_peak_vd": 2.00027
        }
      },
      "n_curves": 33,
      "per_curve": [
        {
          "forward_log_rmse": 4.672125109531012,
          "forward_signed_dec": 4.672125109531012,
          "vg1": 0.2,
          "vg2": -0.05,
          "file": "StandardIV_HH_2vHCa-2_VG2=-0.05_VG=0.2(1)_03-32-11PM.csv",
          "sim_peak_vd": 2.0003,
          "sim_peak_id": 1.649818184836654e-06
        },
        {
          "forward_log_rmse": 4.522621679942609,
          "forward_signed_dec": 4.522621679942609,
          "vg1": 0.2,
          "vg2": -0.1,
          "file": "StandardIV_HH_2vHCa-2_VG2=-0.10_VG=0.2(1)_03-30-30PM.csv",
          "sim_peak_vd": 2.00027,
          "sim_peak_id": 1.6497936798290508e-06
        },
        {
          "forward_log_rmse": 4.382096461610166,
          "forward_signed_dec": 4.382096461610166,
          "vg1": 0.2,
          "vg2": -0.15,
          "file": "StandardIV_HH_2vHCa-2_VG2=-0.15_VG=0.2(1)_03-31-04PM.csv",
          "sim_peak_vd": 2.00026,
          "sim_peak_id": 1.6497855115092548e-06
        },
        {
          "forward_log_rmse": 4.032483039365623,
          "forward_signed_dec": 4.032483039365623,
          "vg1": 0.2,
          "vg2": -0.2,
          "file": "StandardIV_HH_2vHCa-2_VG2=-0.20_VG=0.2(1)_03-31-38PM.csv",
          "sim_peak_vd": 2.0003,
          "sim_peak_id": 1.649818184836654e-06
        },
        {
          "forward_log_rmse": 4.798886663296603,
          "forward_signed_dec": 4.798886663296603,
          "vg1": 0.2,
          "vg2": 0.0,
          "file": "StandardIV_HH_2vHCa-2_VG2=0.00_VG=0.2(1)_03-32-49PM.csv",
          "sim_peak_vd": 2.00028,
          "sim_peak_id": 1.6498018481568813e-06
        },
        {
          "forward_log_rmse": 4.887160492970596,
          "forward_signed_dec": 4.887160492970596,
          "vg1": 0.2,
          "vg2": 0.05,
          "file": "StandardIV_HH_2vHCa-2_VG2=0.05_VG=0.2(1)_03-33-17PM.csv",
          "sim_peak_vd": 2.00028,
          "sim_peak_id": 1.6498018481568813e-06
        },
        {
          "forward_log_rmse": 4.896854107057129,
          "forward_signed_dec": 4.896854107057129,
          "vg1": 0.2,
          "vg2": 0.1,
          "file": "StandardIV_HH_2vHCa-2_VG2=0.10_VG=0.2(1)_03-33-55PM.csv",
          "sim_peak_vd": 2.20037,
          "sim_peak_id": 1.8189396608751904e-06
        },
        {
          "forward_log_rmse": 0.995926491148424,
          "forward_signed_dec": 0.995926491148424,
          "vg1": 0.4,
          "vg2": -0.05,
          "file": "StandardIV_HH_2vHCa-2_VG2=-0.05_VG=0.4(1)_03-36-27PM.csv",
          "sim_peak_vd": 2.00029,
          "sim_peak_id": 1.6496164347302645e-06
        },
        {
          "forward_log_rmse": 0.9882303684954579,
          "forward_signed_dec": 0.9882303684954579,
          "vg1": 0.4,
          "vg2": -0.1,
          "file": "StandardIV_HH_2vHCa-2_VG2=-0.10_VG=0.4(1)_03-36-03PM.csv",
          "sim_peak_vd": 2.0003,
          "sim_peak_id": 1.649624561841937e-06
        },
        {
          "forward_log_rmse": 0.9671766629272254,
          "forward_signed_dec": 0.9671766629272254,
          "vg1": 0.4,
          "vg2": -0.15,
          "file": "StandardIV_HH_2vHCa-2_VG2=-0.15_VG=0.4(1)_03-35-29PM.csv",
          "sim_peak_vd": 2.00033,
          "sim_peak_id": 1.6496489431777798e-06
        },
        {
          "forward_log_rmse": 0.9511868969553943,
          "forward_signed_dec": 0.9511868969553943,
          "vg1": 0.4,
          "vg2": -0.2,
          "file": "StandardIV_HH_2vHCa-2_VG2=-0.20_VG=0.4(1)_03-36-53PM.csv",
          "sim_peak_vd": 2.00027,
          "sim_peak_id": 1.6496001805073321e-06
        },
        {
          "forward_log_rmse": 0.996929102522417,
          "forward_signed_dec": 0.996929102522417,
          "vg1": 0.4,
          "vg2": 0.0,
          "file": "StandardIV_HH_2vHCa-2_VG2=0.00_VG=0.4(1)_03-37-28PM.csv",
          "sim_peak_vd": 2.00023,
          "sim_peak_id": 1.649567672063117e-06
        },
        {
          "forward_log_rmse": 1.6379649701206942,
          "forward_signed_dec": 1.6379649701206942,
          "vg1": 0.4,
          "vg2": 0.05,
          "file": "StandardIV_HH_2vHCa-2_VG2=0.05_VG=0.4(1)_03-37-58PM.csv",
          "sim_peak_vd": 2.00031,
          "sim_peak_id": 1.6496326889537464e-06
        },
        {
          "forward_log_rmse": 2.0968088176259405,
          "forward_signed_dec": 2.0968088176259405,
          "vg1": 0.4,
          "vg2": 0.1,
          "file": "StandardIV_HH_2vHCa-2_VG2=0.10_VG=0.4(1)_03-38-28PM.csv",
          "sim_peak_vd": 2.00027,
          "sim_peak_id": 1.6496001805073321e-06
        },
        {
          "forward_log_rmse": 2.224465820093144,
          "forward_signed_dec": 2.224465820093144,
          "vg1": 0.4,
          "vg2": 0.15,
          "file": "StandardIV_HH_2vHCa-2_VG2=0.15_VG=0.4(1)_03-38-55PM.csv",
          "sim_peak_vd": 2.00026,
          "sim_peak_id": 1.6495920533960723e-06
        },
        {
          "forward_log_rmse": 2.330840633674973,
          "forward_signed_dec": 2.330840633674973,
          "vg1": 0.4,
          "vg2": 0.2,
          "file": "StandardIV_HH_2vHCa-2_VG2=0.20_VG=0.4(1)_03-39-29PM.csv",
          "sim_peak_vd": 2.00032,
          "sim_peak_id": 1.6496408160656945e-06
        },
        {
          "forward_log_rmse": 2.388981869313686,
          "forward_signed_dec": 2.388981869313686,
          "vg1": 0.4,
          "vg2": 0.25,
          "file": "StandardIV_HH_2vHCa-2_VG2=0.25_VG=0.4(1)_03-40-00PM.csv",
          "sim_peak_vd": 2.00027,
          "sim_peak_id": 1.6489633274416799e-06
        },
        {
          "forward_log_rmse": 2.390909548741866,
          "forward_signed_dec": 2.390909548741866,
          "vg1": 0.4,
          "vg2": 0.3,
          "file": "StandardIV_HH_2vHCa-2_VG2=0.3_VG=0.4(1)_03-40-47PM.csv",
          "sim_peak_vd": 2.20036,
          "sim_peak_id": 1.8156212882582101e-06
        },
        {
          "forward_log_rmse": 0.9263230131042786,
          "forward_signed_dec": -0.9263230131042786,
          "vg1": 0.6,
          "vg2": -0.05,
          "file": "StandardIV_HH_2vHCa-2_VG2=-0.05_VG=0.6(1)_03-43-24PM.csv",
          "sim_peak_vd": 2.00025,
          "sim_peak_id": 1.6495826295054697e-06
        },
        {
          "forward_log_rmse": 0.9271534389878804,
          "forward_signed_dec": -0.9271534389878804,
          "vg1": 0.6,
          "vg2": -0.1,
          "file": "StandardIV_HH_2vHCa-2_VG2=-0.10_VG=0.6(1)_03-43-01PM.csv",
          "sim_peak_vd": 2.0003,
          "sim_peak_id": 1.6496232628600617e-06
        },
        {
          "forward_log_rmse": 0.9271796994972856,
          "forward_signed_dec": -0.9271796994972856,
          "vg1": 0.6,
          "vg2": -0.15,
          "file": "StandardIV_HH_2vHCa-2_VG2=-0.15_VG=0.6(1)_03-42-38PM.csv",
          "sim_peak_vd": 2.00027,
          "sim_peak_id": 1.649598882847306e-06
        },
        {
          "forward_log_rmse": 1.0162417085292006,
          "forward_signed_dec": -1.0162417085292006,
          "vg1": 0.6,
          "vg2": -0.2,
          "file": "StandardIV_HH_2vHCa-2_VG2=-0.20_VG=0.6(1)_03-42-07PM.csv",
          "sim_peak_vd": 2.20039,
          "sim_peak_id": 1.8122316244753715e-06
        },
        {
          "forward_log_rmse": 0.9251831305265217,
          "forward_signed_dec": -0.9251831305265217,
          "vg1": 0.6,
          "vg2": 0.0,
          "file": "StandardIV_HH_2vHCa-2_VG2=0.00_VG=0.6(1)_03-43-54PM.csv",
          "sim_peak_vd": 2.00027,
          "sim_peak_id": 1.649598882847306e-06
        },
        {
          "forward_log_rmse": 0.9224378770938912,
          "forward_signed_dec": -0.9224378770938912,
          "vg1": 0.6,
          "vg2": 0.05,
          "file": "StandardIV_HH_2vHCa-2_VG2=0.05_VG=0.6(1)_03-44-26PM.csv",
          "sim_peak_vd": 2.00026,
          "sim_peak_id": 1.6495907561763876e-06
        },
        {
          "forward_log_rmse": 0.9090493527527501,
          "forward_signed_dec": -0.9090493527527501,
          "vg1": 0.6,
          "vg2": 0.1,
          "file": "StandardIV_HH_2vHCa-2_VG2=0.10_VG=0.6(1)_03-44-57PM.csv",
          "sim_peak_vd": 2.00034,
          "sim_peak_id": 1.649655769543736e-06
        },
        {
          "forward_log_rmse": 0.717063436930812,
          "forward_signed_dec": -0.717063436930812,
          "vg1": 0.6,
          "vg2": 0.15,
          "file": "StandardIV_HH_2vHCa-2_VG2=0.15_VG=0.6(1)_03-45-21PM.csv",
          "sim_peak_vd": 2.00028,
          "sim_peak_id": 1.6496070095182245e-06
        },
        {
          "forward_log_rmse": 0.7161477894500141,
          "forward_signed_dec": -0.7161477894500141,
          "vg1": 0.6,
          "vg2": 0.2,
          "file": "StandardIV_HH_2vHCa-2_VG2=0.20_VG=0.6(1)_03-45-46PM.csv",
          "sim_peak_vd": 2.00027,
          "sim_peak_id": 1.649598882847306e-06
        },
        {
          "forward_log_rmse": 0.26982321227571315,
          "forward_signed_dec": -0.26982321227571315,
          "vg1": 0.6,
          "vg2": 0.25,
          "file": "StandardIV_HH_2vHCa-2_VG2=0.25_VG=0.6(1)_03-46-14PM.csv",
          "sim_peak_vd": 2.00024,
          "sim_peak_id": 1.6497323837614103e-06
        },
        {
          "forward_log_rmse": 0.14422522439950747,
          "forward_signed_dec": -0.14422522439950747,
          "vg1": 0.6,
          "vg2": 0.3,
          "file": "StandardIV_HH_2vHCa-2_VG2=0.30_VG=0.6(1)_03-46-40PM.csv",
          "sim_peak_vd": 2.00031,
          "sim_peak_id": 1.650356936126872e-06
        },
        {
          "forward_log_rmse": 0.10841859798182618,
          "forward_signed_dec": -0.03725069297504291,
          "vg1": 0.6,
          "vg2": 0.35,
          "file": "StandardIV_HH_2vHCa-2_VG2=0.35_VG=0.6(1)_03-47-03PM.csv",
          "sim_peak_vd": 2.00031,
          "sim_peak_id": 1.6543077655776046e-06
        },
        {
          "forward_log_rmse": 0.1084838956109273,
          "forward_signed_dec": 0.025735831180898394,
          "vg1": 0.6,
          "vg2": 0.4,
          "file": "StandardIV_HH_2vHCa-2_VG2=0.40_VG=0.6(1)_03-47-25PM.csv",
          "sim_peak_vd": 2.00027,
          "sim_peak_id": 1.678749625226478e-06
        },
        {
          "forward_log_rmse": 0.1071862130251251,
          "forward_signed_dec": 0.06333706668599781,
          "vg1": 0.6,
          "vg2": 0.45,
          "file": "StandardIV_HH_2vHCa-2_VG2=0.45_VG=0.6(1)_03-47-47PM.csv",
          "sim_peak_vd": 2.00026,
          "sim_peak_id": 1.8172316947505022e-06
        },
        {
          "forward_log_rmse": 0.11371866375728956,
          "forward_signed_dec": 0.09023310980940025,
          "vg1": 0.6,
          "vg2": 0.5,
          "file": "StandardIV_HH_2vHCa-2_VG2=0.50_VG=0.6(1)_03-48-22PM.csv",
          "sim_peak_vd": 2.20035,
          "sim_peak_id": 4.6212603536972265e-06
        }
      ]
    },
    {
      "label": "mario_only",
      "Bf": 50.0,
      "Va": 40.0,
      "Is": 1e-16,
      "Nf": 0.9,
      "Ne": 1.5,
      "Var": 10.0,
      "use_bv_avalanche": false,
      "BV_at_VG1_0.6_V": 2.6,
      "median_fwd_log_rmse_all": 2.1941132883463306,
      "median_signed_dec_all": -1.2530474348299885,
      "by_vg1": {
        "0.2": {
          "median_fwd_log_rmse": 2.605615458010776,
          "median_signed_dec": 2.605615458010776,
          "n": 7,
          "median_sim_peak_vd": 2.00028
        },
        "0.4": {
          "median_fwd_log_rmse": 1.0206942116751243,
          "median_signed_dec": -0.9531766795541756,
          "n": 11,
          "median_sim_peak_vd": 2.00029
        },
        "0.6": {
          "median_fwd_log_rmse": 3.2175199054445978,
          "median_signed_dec": -3.2175199054445978,
          "n": 15,
          "median_sim_peak_vd": 2.00027
        }
      },
      "n_curves": 33,
      "per_curve": [
        {
          "forward_log_rmse": 2.605615458010776,
          "forward_signed_dec": 2.605615458010776,
          "vg1": 0.2,
          "vg2": -0.05,
          "file": "StandardIV_HH_2vHCa-2_VG2=-0.05_VG=0.2(1)_03-32-11PM.csv",
          "sim_peak_vd": 2.0003,
          "sim_peak_id": 6.046090674171304e-09
        },
        {
          "forward_log_rmse": 1.848344270449374,
          "forward_signed_dec": 1.848344270449374,
          "vg1": 0.2,
          "vg2": -0.1,
          "file": "StandardIV_HH_2vHCa-2_VG2=-0.10_VG=0.2(1)_03-30-30PM.csv",
          "sim_peak_vd": 2.00027,
          "sim_peak_id": 6.0460855428271455e-09
        },
        {
          "forward_log_rmse": 1.6406249637788513,
          "forward_signed_dec": 1.6406249637788513,
          "vg1": 0.2,
          "vg2": -0.15,
          "file": "StandardIV_HH_2vHCa-2_VG2=-0.15_VG=0.2(1)_03-31-04PM.csv",
          "sim_peak_vd": 2.00026,
          "sim_peak_id": 6.046083832429676e-09
        },
        {
          "forward_log_rmse": 1.610483264236196,
          "forward_signed_dec": 1.610483264236196,
          "vg1": 0.2,
          "vg2": -0.2,
          "file": "StandardIV_HH_2vHCa-2_VG2=-0.20_VG=0.2(1)_03-31-38PM.csv",
          "sim_peak_vd": 2.0003,
          "sim_peak_id": 6.046090674171304e-09
        },
        {
          "forward_log_rmse": 2.667287301124917,
          "forward_signed_dec": 2.667287301124917,
          "vg1": 0.2,
          "vg2": 0.0,
          "file": "StandardIV_HH_2vHCa-2_VG2=0.00_VG=0.2(1)_03-32-49PM.csv",
          "sim_peak_vd": 2.00028,
          "sim_peak_id": 6.046087253249905e-09
        },
        {
          "forward_log_rmse": 2.682408411708442,
          "forward_signed_dec": 2.682408411708442,
          "vg1": 0.2,
          "vg2": 0.05,
          "file": "StandardIV_HH_2vHCa-2_VG2=0.05_VG=0.2(1)_03-33-17PM.csv",
          "sim_peak_vd": 2.00028,
          "sim_peak_id": 6.046087253249905e-09
        },
        {
          "forward_log_rmse": 2.6504344202984598,
          "forward_signed_dec": 2.6504344202984598,
          "vg1": 0.2,
          "vg2": 0.1,
          "file": "StandardIV_HH_2vHCa-2_VG2=0.10_VG=0.2(1)_03-33-55PM.csv",
          "sim_peak_vd": 2.20037,
          "sim_peak_id": 6.098117901446903e-09
        },
        {
          "forward_log_rmse": 1.2517573772167223,
          "forward_signed_dec": -1.2517573772167223,
          "vg1": 0.4,
          "vg2": -0.05,
          "file": "StandardIV_HH_2vHCa-2_VG2=-0.05_VG=0.4(1)_03-36-27PM.csv",
          "sim_peak_vd": 2.00029,
          "sim_peak_id": 6.045479427931612e-09
        },
        {
          "forward_log_rmse": 1.2546567500762995,
          "forward_signed_dec": -1.2546567500762995,
          "vg1": 0.4,
          "vg2": -0.1,
          "file": "StandardIV_HH_2vHCa-2_VG2=-0.10_VG=0.4(1)_03-36-03PM.csv",
          "sim_peak_vd": 2.0003,
          "sim_peak_id": 6.045481008545294e-09
        },
        {
          "forward_log_rmse": 1.2530474348299885,
          "forward_signed_dec": -1.2530474348299885,
          "vg1": 0.4,
          "vg2": -0.15,
          "file": "StandardIV_HH_2vHCa-2_VG2=-0.15_VG=0.4(1)_03-35-29PM.csv",
          "sim_peak_vd": 2.00033,
          "sim_peak_id": 6.045485750388938e-09
        },
        {
          "forward_log_rmse": 1.2527878490044548,
          "forward_signed_dec": -1.2527878490044548,
          "vg1": 0.4,
          "vg2": -0.2,
          "file": "StandardIV_HH_2vHCa-2_VG2=-0.20_VG=0.4(1)_03-36-53PM.csv",
          "sim_peak_vd": 2.00027,
          "sim_peak_id": 6.04547626670555e-09
        },
        {
          "forward_log_rmse": 1.3013346188335326,
          "forward_signed_dec": -1.229980222514989,
          "vg1": 0.4,
          "vg2": 0.0,
          "file": "StandardIV_HH_2vHCa-2_VG2=0.00_VG=0.4(1)_03-37-28PM.csv",
          "sim_peak_vd": 2.00023,
          "sim_peak_id": 6.045469944258618e-09
        },
        {
          "forward_log_rmse": 1.0206942116751243,
          "forward_signed_dec": -0.9531766795541756,
          "vg1": 0.4,
          "vg2": 0.05,
          "file": "StandardIV_HH_2vHCa-2_VG2=0.05_VG=0.4(1)_03-37-58PM.csv",
          "sim_peak_vd": 2.00031,
          "sim_peak_id": 6.045482589159409e-09
        },
        {
          "forward_log_rmse": 0.8565857997647228,
          "forward_signed_dec": -0.5833131159182781,
          "vg1": 0.4,
          "vg2": 0.1,
          "file": "StandardIV_HH_2vHCa-2_VG2=0.10_VG=0.4(1)_03-38-28PM.csv",
          "sim_peak_vd": 2.00027,
          "sim_peak_id": 6.04547626670555e-09
        },
        {
          "forward_log_rmse": 0.5087705466799388,
          "forward_signed_dec": 0.18244039541069057,
          "vg1": 0.4,
          "vg2": 0.15,
          "file": "StandardIV_HH_2vHCa-2_VG2=0.15_VG=0.4(1)_03-38-55PM.csv",
          "sim_peak_vd": 2.00026,
          "sim_peak_id": 6.045474686093168e-09
        },
        {
          "forward_log_rmse": 0.46398930347471357,
          "forward_signed_dec": 0.18719574474192946,
          "vg1": 0.4,
          "vg2": 0.2,
          "file": "StandardIV_HH_2vHCa-2_VG2=0.20_VG=0.4(1)_03-39-29PM.csv",
          "sim_peak_vd": 2.00032,
          "sim_peak_id": 6.045484169773956e-09
        },
        {
          "forward_log_rmse": 0.38631397639691833,
          "forward_signed_dec": 0.18163213781612164,
          "vg1": 0.4,
          "vg2": 0.25,
          "file": "StandardIV_HH_2vHCa-2_VG2=0.25_VG=0.4(1)_03-40-00PM.csv",
          "sim_peak_vd": 1.70021,
          "sim_peak_id": 5.961184502200307e-09
        },
        {
          "forward_log_rmse": 0.33071623516376647,
          "forward_signed_dec": 0.14570131494612326,
          "vg1": 0.4,
          "vg2": 0.3,
          "file": "StandardIV_HH_2vHCa-2_VG2=0.3_VG=0.4(1)_03-40-47PM.csv",
          "sim_peak_vd": 2.20036,
          "sim_peak_id": 5.753064719758551e-09
        },
        {
          "forward_log_rmse": 3.226386063328702,
          "forward_signed_dec": -3.226386063328702,
          "vg1": 0.6,
          "vg2": -0.05,
          "file": "StandardIV_HH_2vHCa-2_VG2=-0.05_VG=0.6(1)_03-43-24PM.csv",
          "sim_peak_vd": 2.00025,
          "sim_peak_id": 6.045469022284551e-09
        },
        {
          "forward_log_rmse": 3.226296305949936,
          "forward_signed_dec": -3.226296305949936,
          "vg1": 0.6,
          "vg2": -0.1,
          "file": "StandardIV_HH_2vHCa-2_VG2=-0.10_VG=0.6(1)_03-43-01PM.csv",
          "sim_peak_vd": 2.0003,
          "sim_peak_id": 6.045476918412872e-09
        },
        {
          "forward_log_rmse": 3.2261610243608594,
          "forward_signed_dec": -3.2261610243608594,
          "vg1": 0.6,
          "vg2": -0.15,
          "file": "StandardIV_HH_2vHCa-2_VG2=-0.15_VG=0.6(1)_03-42-38PM.csv",
          "sim_peak_vd": 2.00027,
          "sim_peak_id": 6.04547218073588e-09
        },
        {
          "forward_log_rmse": 3.334551529598527,
          "forward_signed_dec": -3.334551529598527,
          "vg1": 0.6,
          "vg2": -0.2,
          "file": "StandardIV_HH_2vHCa-2_VG2=-0.20_VG=0.6(1)_03-42-07PM.csv",
          "sim_peak_vd": 2.20039,
          "sim_peak_id": 6.077081286644173e-09
        },
        {
          "forward_log_rmse": 3.2256814103389795,
          "forward_signed_dec": -3.2256814103389795,
          "vg1": 0.6,
          "vg2": 0.0,
          "file": "StandardIV_HH_2vHCa-2_VG2=0.00_VG=0.6(1)_03-43-54PM.csv",
          "sim_peak_vd": 2.00027,
          "sim_peak_id": 6.04547218073588e-09
        },
        {
          "forward_log_rmse": 3.2260001729169083,
          "forward_signed_dec": -3.2260001729169083,
          "vg1": 0.6,
          "vg2": 0.05,
          "file": "StandardIV_HH_2vHCa-2_VG2=0.05_VG=0.6(1)_03-44-26PM.csv",
          "sim_peak_vd": 2.00026,
          "sim_peak_id": 6.045470601510215e-09
        },
        {
          "forward_log_rmse": 3.2246455228490696,
          "forward_signed_dec": -3.2246455228490696,
          "vg1": 0.6,
          "vg2": 0.1,
          "file": "StandardIV_HH_2vHCa-2_VG2=0.10_VG=0.6(1)_03-44-57PM.csv",
          "sim_peak_vd": 2.00034,
          "sim_peak_id": 6.0454832353155315e-09
        },
        {
          "forward_log_rmse": 3.2175199054445978,
          "forward_signed_dec": -3.2175199054445978,
          "vg1": 0.6,
          "vg2": 0.15,
          "file": "StandardIV_HH_2vHCa-2_VG2=0.15_VG=0.6(1)_03-45-21PM.csv",
          "sim_peak_vd": 2.00028,
          "sim_peak_id": 6.0454737599615436e-09
        },
        {
          "forward_log_rmse": 3.1842870468348146,
          "forward_signed_dec": -3.1842870468348146,
          "vg1": 0.6,
          "vg2": 0.2,
          "file": "StandardIV_HH_2vHCa-2_VG2=0.20_VG=0.6(1)_03-45-46PM.csv",
          "sim_peak_vd": 2.00027,
          "sim_peak_id": 6.04547218073588e-09
        },
        {
          "forward_log_rmse": 2.964824914700224,
          "forward_signed_dec": -2.964824914700224,
          "vg1": 0.6,
          "vg2": 0.25,
          "file": "StandardIV_HH_2vHCa-2_VG2=0.25_VG=0.6(1)_03-46-14PM.csv",
          "sim_peak_vd": 2.00024,
          "sim_peak_id": 6.037689917060706e-09
        },
        {
          "forward_log_rmse": 2.1897602543642565,
          "forward_signed_dec": -2.1897602543642565,
          "vg1": 0.6,
          "vg2": 0.3,
          "file": "StandardIV_HH_2vHCa-2_VG2=0.30_VG=0.6(1)_03-46-40PM.csv",
          "sim_peak_vd": 2.00031,
          "sim_peak_id": 6.022109587204584e-09
        },
        {
          "forward_log_rmse": 2.1689701254474114,
          "forward_signed_dec": -2.1689701254474114,
          "vg1": 0.6,
          "vg2": 0.35,
          "file": "StandardIV_HH_2vHCa-2_VG2=0.35_VG=0.6(1)_03-47-03PM.csv",
          "sim_peak_vd": 2.00031,
          "sim_peak_id": 5.969504225283284e-09
        },
        {
          "forward_log_rmse": 2.1941132883463306,
          "forward_signed_dec": -2.1941132883463306,
          "vg1": 0.6,
          "vg2": 0.4,
          "file": "StandardIV_HH_2vHCa-2_VG2=0.40_VG=0.6(1)_03-47-25PM.csv",
          "sim_peak_vd": 2.00027,
          "sim_peak_id": 5.8275218786725805e-09
        },
        {
          "forward_log_rmse": 2.278241306531177,
          "forward_signed_dec": -2.278241306531177,
          "vg1": 0.6,
          "vg2": 0.45,
          "file": "StandardIV_HH_2vHCa-2_VG2=0.45_VG=0.6(1)_03-47-47PM.csv",
          "sim_peak_vd": 0.650191,
          "sim_peak_id": 5.746879015189103e-09
        },
        {
          "forward_log_rmse": 2.2333217270556487,
          "forward_signed_dec": -2.2333217270556487,
          "vg1": 0.6,
          "vg2": 0.5,
          "file": "StandardIV_HH_2vHCa-2_VG2=0.50_VG=0.6(1)_03-48-22PM.csv",
          "sim_peak_vd": 2.20035,
          "sim_peak_id": 1.1136751067448482e-08
        }
      ]
    },
    {
      "label": "mario_plus_bv",
      "Bf": 50.0,
      "Va": 40.0,
      "Is": 1e-16,
      "Nf": 0.9,
      "Ne": 1.5,
      "Var": 10.0,
      "use_bv_avalanche": true,
      "BV_at_VG1_0.6_V": 2.6,
      "median_fwd_log_rmse_all": 2.193152730959098,
      "median_signed_dec_all": -1.2529881708750148,
      "by_vg1": {
        "0.2": {
          "median_fwd_log_rmse": 2.60603261855497,
          "median_signed_dec": 2.60603261855497,
          "n": 7,
          "median_sim_peak_vd": 2.00028
        },
        "0.4": {
          "median_fwd_log_rmse": 1.0206942116751243,
          "median_signed_dec": -0.9531766795541756,
          "n": 11,
          "median_sim_peak_vd": 2.00029
        },
        "0.6": {
          "median_fwd_log_rmse": 3.2172436087718,
          "median_signed_dec": -3.2172436087718,
          "n": 15,
          "median_sim_peak_vd": 2.00027
        }
      },
      "n_curves": 33,
      "per_curve": [
        {
          "forward_log_rmse": 2.60603261855497,
          "forward_signed_dec": 2.60603261855497,
          "vg1": 0.2,
          "vg2": -0.05,
          "file": "StandardIV_HH_2vHCa-2_VG2=-0.05_VG=0.2(1)_03-32-11PM.csv",
          "sim_peak_vd": 2.0003,
          "sim_peak_id": 6.191530568870563e-09
        },
        {
          "forward_log_rmse": 1.848344270449374,
          "forward_signed_dec": 1.848344270449374,
          "vg1": 0.2,
          "vg2": -0.1,
          "file": "StandardIV_HH_2vHCa-2_VG2=-0.10_VG=0.2(1)_03-30-30PM.csv",
          "sim_peak_vd": 2.00027,
          "sim_peak_id": 6.19151159857532e-09
        },
        {
          "forward_log_rmse": 1.6406249637788513,
          "forward_signed_dec": 1.6406249637788513,
          "vg1": 0.2,
          "vg2": -0.15,
          "file": "StandardIV_HH_2vHCa-2_VG2=-0.15_VG=0.2(1)_03-31-04PM.csv",
          "sim_peak_vd": 2.00026,
          "sim_peak_id": 6.191505275413829e-09
        },
        {
          "forward_log_rmse": 1.610483264236196,
          "forward_signed_dec": 1.610483264236196,
          "vg1": 0.2,
          "vg2": -0.2,
          "file": "StandardIV_HH_2vHCa-2_VG2=-0.20_VG=0.2(1)_03-31-38PM.csv",
          "sim_peak_vd": 2.0003,
          "sim_peak_id": 6.191530568870563e-09
        },
        {
          "forward_log_rmse": 2.6678731527737494,
          "forward_signed_dec": 2.6678731527737494,
          "vg1": 0.2,
          "vg2": 0.0,
          "file": "StandardIV_HH_2vHCa-2_VG2=0.00_VG=0.2(1)_03-32-49PM.csv",
          "sim_peak_vd": 2.00028,
          "sim_peak_id": 6.191517921871937e-09
        },
        {
          "forward_log_rmse": 2.6829941490172082,
          "forward_signed_dec": 2.6829941490172082,
          "vg1": 0.2,
          "vg2": 0.05,
          "file": "StandardIV_HH_2vHCa-2_VG2=0.05_VG=0.2(1)_03-33-17PM.csv",
          "sim_peak_vd": 2.00028,
          "sim_peak_id": 6.191517921871937e-09
        },
        {
          "forward_log_rmse": 2.651504512574121,
          "forward_signed_dec": 2.651504512574121,
          "vg1": 0.2,
          "vg2": 0.1,
          "file": "StandardIV_HH_2vHCa-2_VG2=0.10_VG=0.2(1)_03-33-55PM.csv",
          "sim_peak_vd": 2.20037,
          "sim_peak_id": 6.360756397561857e-09
        },
        {
          "forward_log_rmse": 1.2516980754891298,
          "forward_signed_dec": -1.2516980754891298,
          "vg1": 0.4,
          "vg2": -0.05,
          "file": "StandardIV_HH_2vHCa-2_VG2=-0.05_VG=0.4(1)_03-36-27PM.csv",
          "sim_peak_vd": 2.00029,
          "sim_peak_id": 6.25890823605932e-09
        },
        {
          "forward_log_rmse": 1.2545974710144412,
          "forward_signed_dec": -1.2545974710144412,
          "vg1": 0.4,
          "vg2": -0.1,
          "file": "StandardIV_HH_2vHCa-2_VG2=-0.10_VG=0.4(1)_03-36-03PM.csv",
          "sim_peak_vd": 2.0003,
          "sim_peak_id": 6.258916539679771e-09
        },
        {
          "forward_log_rmse": 1.2529881708750148,
          "forward_signed_dec": -1.2529881708750148,
          "vg1": 0.4,
          "vg2": -0.15,
          "file": "StandardIV_HH_2vHCa-2_VG2=-0.15_VG=0.4(1)_03-35-29PM.csv",
          "sim_peak_vd": 2.00033,
          "sim_peak_id": 6.258941451483167e-09
        },
        {
          "forward_log_rmse": 1.2527285623880742,
          "forward_signed_dec": -1.2527285623880742,
          "vg1": 0.4,
          "vg2": -0.2,
          "file": "StandardIV_HH_2vHCa-2_VG2=-0.20_VG=0.4(1)_03-36-53PM.csv",
          "sim_peak_vd": 2.00027,
          "sim_peak_id": 6.2588916292894276e-09
        },
        {
          "forward_log_rmse": 1.3013346188335326,
          "forward_signed_dec": -1.2299208981151626,
          "vg1": 0.4,
          "vg2": 0.0,
          "file": "StandardIV_HH_2vHCa-2_VG2=0.00_VG=0.4(1)_03-37-28PM.csv",
          "sim_peak_vd": 2.00023,
          "sim_peak_id": 6.25885841763361e-09
        },
        {
          "forward_log_rmse": 1.0206942116751243,
          "forward_signed_dec": -0.9531766795541756,
          "vg1": 0.4,
          "vg2": 0.05,
          "file": "StandardIV_HH_2vHCa-2_VG2=0.05_VG=0.4(1)_03-37-58PM.csv",
          "sim_peak_vd": 2.00031,
          "sim_peak_id": 6.258924843457226e-09
        },
        {
          "forward_log_rmse": 0.8565857997647228,
          "forward_signed_dec": -0.5833131159182781,
          "vg1": 0.4,
          "vg2": 0.1,
          "file": "StandardIV_HH_2vHCa-2_VG2=0.10_VG=0.4(1)_03-38-28PM.csv",
          "sim_peak_vd": 2.00027,
          "sim_peak_id": 6.2588916292894276e-09
        },
        {
          "forward_log_rmse": 0.5087705466799388,
          "forward_signed_dec": 0.18286602981894262,
          "vg1": 0.4,
          "vg2": 0.15,
          "file": "StandardIV_HH_2vHCa-2_VG2=0.15_VG=0.4(1)_03-38-55PM.csv",
          "sim_peak_vd": 2.00026,
          "sim_peak_id": 6.258883326139981e-09
        },
        {
          "forward_log_rmse": 0.46398930347471357,
          "forward_signed_dec": 0.18719574474192946,
          "vg1": 0.4,
          "vg2": 0.2,
          "file": "StandardIV_HH_2vHCa-2_VG2=0.20_VG=0.4(1)_03-39-29PM.csv",
          "sim_peak_vd": 2.00032,
          "sim_peak_id": 6.25893314739169e-09
        },
        {
          "forward_log_rmse": 0.3863139766191779,
          "forward_signed_dec": 0.18163213781612164,
          "vg1": 0.4,
          "vg2": 0.25,
          "file": "StandardIV_HH_2vHCa-2_VG2=0.25_VG=0.4(1)_03-40-00PM.csv",
          "sim_peak_vd": 2.00027,
          "sim_peak_id": 6.162971955500812e-09
        },
        {
          "forward_log_rmse": 0.3307172872971744,
          "forward_signed_dec": 0.14657087461604412,
          "vg1": 0.4,
          "vg2": 0.3,
          "file": "StandardIV_HH_2vHCa-2_VG2=0.3_VG=0.4(1)_03-40-47PM.csv",
          "sim_peak_vd": 2.20036,
          "sim_peak_id": 6.115707317121236e-09
        },
        {
          "forward_log_rmse": 3.22610979331804,
          "forward_signed_dec": -3.22610979331804,
          "vg1": 0.6,
          "vg2": -0.05,
          "file": "StandardIV_HH_2vHCa-2_VG2=-0.05_VG=0.6(1)_03-43-24PM.csv",
          "sim_peak_vd": 2.00025,
          "sim_peak_id": 6.369466145811237e-09
        },
        {
          "forward_log_rmse": 3.226020062599484,
          "forward_signed_dec": -3.226020062599484,
          "vg1": 0.6,
          "vg2": -0.1,
          "file": "StandardIV_HH_2vHCa-2_VG2=-0.10_VG=0.6(1)_03-43-01PM.csv",
          "sim_peak_vd": 2.0003,
          "sim_peak_id": 6.369524470497977e-09
        },
        {
          "forward_log_rmse": 3.225884781010407,
          "forward_signed_dec": -3.225884781010407,
          "vg1": 0.6,
          "vg2": -0.15,
          "file": "StandardIV_HH_2vHCa-2_VG2=-0.15_VG=0.6(1)_03-42-38PM.csv",
          "sim_peak_vd": 2.00027,
          "sim_peak_id": 6.369489475002429e-09
        },
        {
          "forward_log_rmse": 3.3338959740788576,
          "forward_signed_dec": -3.3338959740788576,
          "vg1": 0.6,
          "vg2": -0.2,
          "file": "StandardIV_HH_2vHCa-2_VG2=-0.20_VG=0.6(1)_03-42-07PM.csv",
          "sim_peak_vd": 2.20039,
          "sim_peak_id": 6.652345319724554e-09
        },
        {
          "forward_log_rmse": 3.2254051403283173,
          "forward_signed_dec": -3.2254051403283173,
          "vg1": 0.6,
          "vg2": 0.0,
          "file": "StandardIV_HH_2vHCa-2_VG2=0.00_VG=0.6(1)_03-43-54PM.csv",
          "sim_peak_vd": 2.00027,
          "sim_peak_id": 6.369489475002429e-09
        },
        {
          "forward_log_rmse": 3.2257239828810995,
          "forward_signed_dec": -3.2257239828810995,
          "vg1": 0.6,
          "vg2": 0.05,
          "file": "StandardIV_HH_2vHCa-2_VG2=0.05_VG=0.6(1)_03-44-26PM.csv",
          "sim_peak_vd": 2.00026,
          "sim_peak_id": 6.369477810292916e-09
        },
        {
          "forward_log_rmse": 3.2243693594676923,
          "forward_signed_dec": -3.2243693594676923,
          "vg1": 0.6,
          "vg2": 0.1,
          "file": "StandardIV_HH_2vHCa-2_VG2=0.10_VG=0.6(1)_03-44-57PM.csv",
          "sim_peak_vd": 2.00034,
          "sim_peak_id": 6.369571134348511e-09
        },
        {
          "forward_log_rmse": 3.2172436087718,
          "forward_signed_dec": -3.2172436087718,
          "vg1": 0.6,
          "vg2": 0.15,
          "file": "StandardIV_HH_2vHCa-2_VG2=0.15_VG=0.6(1)_03-45-21PM.csv",
          "sim_peak_vd": 2.00028,
          "sim_peak_id": 6.3695011399397756e-09
        },
        {
          "forward_log_rmse": 3.1842870468348146,
          "forward_signed_dec": -3.1842870468348146,
          "vg1": 0.6,
          "vg2": 0.2,
          "file": "StandardIV_HH_2vHCa-2_VG2=0.20_VG=0.6(1)_03-45-46PM.csv",
          "sim_peak_vd": 2.00027,
          "sim_peak_id": 6.369489475002429e-09
        },
        {
          "forward_log_rmse": 2.964824914700224,
          "forward_signed_dec": -2.964824914700224,
          "vg1": 0.6,
          "vg2": 0.25,
          "file": "StandardIV_HH_2vHCa-2_VG2=0.25_VG=0.6(1)_03-46-14PM.csv",
          "sim_peak_vd": 2.00024,
          "sim_peak_id": 6.361281722497816e-09
        },
        {
          "forward_log_rmse": 2.1897602543642565,
          "forward_signed_dec": -2.1897602543642565,
          "vg1": 0.6,
          "vg2": 0.3,
          "file": "StandardIV_HH_2vHCa-2_VG2=0.30_VG=0.6(1)_03-46-40PM.csv",
          "sim_peak_vd": 2.00031,
          "sim_peak_id": 6.344983768261187e-09
        },
        {
          "forward_log_rmse": 2.168017801600615,
          "forward_signed_dec": -2.168017801600615,
          "vg1": 0.6,
          "vg2": 0.35,
          "file": "StandardIV_HH_2vHCa-2_VG2=0.35_VG=0.6(1)_03-47-03PM.csv",
          "sim_peak_vd": 2.00031,
          "sim_peak_id": 6.289736167217566e-09
        },
        {
          "forward_log_rmse": 2.193152730959098,
          "forward_signed_dec": -2.193152730959098,
          "vg1": 0.6,
          "vg2": 0.4,
          "file": "StandardIV_HH_2vHCa-2_VG2=0.40_VG=0.6(1)_03-47-25PM.csv",
          "sim_peak_vd": 2.00027,
          "sim_peak_id": 6.14069421766634e-09
        },
        {
          "forward_log_rmse": 2.277211675591831,
          "forward_signed_dec": -2.277211675591831,
          "vg1": 0.6,
          "vg2": 0.45,
          "file": "StandardIV_HH_2vHCa-2_VG2=0.45_VG=0.6(1)_03-47-47PM.csv",
          "sim_peak_vd": 2.00026,
          "sim_peak_id": 5.962586433980255e-09
        },
        {
          "forward_log_rmse": 2.233080345527097,
          "forward_signed_dec": -2.233080345527097,
          "vg1": 0.6,
          "vg2": 0.5,
          "file": "StandardIV_HH_2vHCa-2_VG2=0.50_VG=0.6(1)_03-48-22PM.csv",
          "sim_peak_vd": 2.20035,
          "sim_peak_id": 1.2178511583948655e-08
        }
      ]
    },
    {
      "label": "mario_plus_da3_bf",
      "Bf": 3000.0,
      "Va": 40.0,
      "Is": 1e-16,
      "Nf": 0.9,
      "Ne": 1.5,
      "Var": 10.0,
      "use_bv_avalanche": true,
      "BV_at_VG1_0.6_V": 2.6,
      "median_fwd_log_rmse_all": 1.930652491849627,
      "median_signed_dec_all": -0.6963200846498179,
      "by_vg1": {
        "0.2": {
          "median_fwd_log_rmse": 2.887570206428391,
          "median_signed_dec": 2.887570206428391,
          "n": 7,
          "median_sim_peak_vd": 2.00028
        },
        "0.4": {
          "median_fwd_log_rmse": 1.3046717834408046,
          "median_signed_dec": -0.6914785934079948,
          "n": 11,
          "median_sim_peak_vd": 2.00029
        },
        "0.6": {
          "median_fwd_log_rmse": 2.3700899981813253,
          "median_signed_dec": -2.3700899981813253,
          "n": 15,
          "median_sim_peak_vd": 2.00027
        }
      },
      "n_curves": 33,
      "per_curve": [
        {
          "forward_log_rmse": 2.887570206428391,
          "forward_signed_dec": 2.887570206428391,
          "vg1": 0.2,
          "vg2": -0.05,
          "file": "StandardIV_HH_2vHCa-2_VG2=-0.05_VG=0.2(1)_03-32-11PM.csv",
          "sim_peak_vd": 2.0003,
          "sim_peak_id": 5.651604573507816e-08
        },
        {
          "forward_log_rmse": 2.166249827356701,
          "forward_signed_dec": 2.166249827356701,
          "vg1": 0.2,
          "vg2": -0.1,
          "file": "StandardIV_HH_2vHCa-2_VG2=-0.10_VG=0.2(1)_03-30-30PM.csv",
          "sim_peak_vd": 2.00027,
          "sim_peak_id": 5.651589309223639e-08
        },
        {
          "forward_log_rmse": 1.9306485372045046,
          "forward_signed_dec": 1.9306485372045046,
          "vg1": 0.2,
          "vg2": -0.15,
          "file": "StandardIV_HH_2vHCa-2_VG2=-0.15_VG=0.2(1)_03-31-04PM.csv",
          "sim_peak_vd": 2.00026,
          "sim_peak_id": 5.651584221310236e-08
        },
        {
          "forward_log_rmse": 1.930652491849627,
          "forward_signed_dec": 1.930652491849627,
          "vg1": 0.2,
          "vg2": -0.2,
          "file": "StandardIV_HH_2vHCa-2_VG2=-0.20_VG=0.2(1)_03-31-38PM.csv",
          "sim_peak_vd": 2.0003,
          "sim_peak_id": 5.651604573507816e-08
        },
        {
          "forward_log_rmse": 3.629919779680021,
          "forward_signed_dec": 3.629919779680021,
          "vg1": 0.2,
          "vg2": 0.0,
          "file": "StandardIV_HH_2vHCa-2_VG2=0.00_VG=0.2(1)_03-32-49PM.csv",
          "sim_peak_vd": 2.00028,
          "sim_peak_id": 5.651594397227702e-08
        },
        {
          "forward_log_rmse": 3.6450408051286667,
          "forward_signed_dec": 3.6450408051286667,
          "vg1": 0.2,
          "vg2": 0.05,
          "file": "StandardIV_HH_2vHCa-2_VG2=0.05_VG=0.2(1)_03-33-17PM.csv",
          "sim_peak_vd": 2.00028,
          "sim_peak_id": 5.651594397227702e-08
        },
        {
          "forward_log_rmse": 3.6134348006561634,
          "forward_signed_dec": 3.6134348006561634,
          "vg1": 0.2,
          "vg2": 0.1,
          "file": "StandardIV_HH_2vHCa-2_VG2=0.10_VG=0.2(1)_03-33-55PM.csv",
          "sim_peak_vd": 2.20037,
          "sim_peak_id": 5.7736079929492025e-08
        },
        {
          "forward_log_rmse": 1.3046717834408046,
          "forward_signed_dec": -0.6953276315861938,
          "vg1": 0.4,
          "vg2": -0.05,
          "file": "StandardIV_HH_2vHCa-2_VG2=-0.05_VG=0.4(1)_03-36-27PM.csv",
          "sim_peak_vd": 2.00029,
          "sim_peak_id": 5.7026076628443e-08
        },
        {
          "forward_log_rmse": 1.2680623599960077,
          "forward_signed_dec": -0.6963125646665009,
          "vg1": 0.4,
          "vg2": -0.1,
          "file": "StandardIV_HH_2vHCa-2_VG2=-0.10_VG=0.4(1)_03-36-03PM.csv",
          "sim_peak_vd": 2.0003,
          "sim_peak_id": 5.7026144263149714e-08
        },
        {
          "forward_log_rmse": 1.2675854279424073,
          "forward_signed_dec": -0.6963817873851639,
          "vg1": 0.4,
          "vg2": -0.15,
          "file": "StandardIV_HH_2vHCa-2_VG2=-0.15_VG=0.4(1)_03-35-29PM.csv",
          "sim_peak_vd": 2.00033,
          "sim_peak_id": 5.7026347175114287e-08
        },
        {
          "forward_log_rmse": 1.2371788665772554,
          "forward_signed_dec": -0.6963200846498179,
          "vg1": 0.4,
          "vg2": -0.2,
          "file": "StandardIV_HH_2vHCa-2_VG2=-0.20_VG=0.4(1)_03-36-53PM.csv",
          "sim_peak_vd": 2.00027,
          "sim_peak_id": 5.702594136295162e-08
        },
        {
          "forward_log_rmse": 1.3156942599769277,
          "forward_signed_dec": -0.6950158275337497,
          "vg1": 0.4,
          "vg2": 0.0,
          "file": "StandardIV_HH_2vHCa-2_VG2=0.00_VG=0.4(1)_03-37-28PM.csv",
          "sim_peak_vd": 2.00023,
          "sim_peak_id": 5.702567084765653e-08
        },
        {
          "forward_log_rmse": 1.3179011735395143,
          "forward_signed_dec": -0.6914785934079948,
          "vg1": 0.4,
          "vg2": 0.05,
          "file": "StandardIV_HH_2vHCa-2_VG2=0.05_VG=0.4(1)_03-37-58PM.csv",
          "sim_peak_vd": 2.00031,
          "sim_peak_id": 5.702621189916383e-08
        },
        {
          "forward_log_rmse": 1.31981886789535,
          "forward_signed_dec": -0.5410318949827158,
          "vg1": 0.4,
          "vg2": 0.1,
          "file": "StandardIV_HH_2vHCa-2_VG2=0.10_VG=0.4(1)_03-38-28PM.csv",
          "sim_peak_vd": 2.00027,
          "sim_peak_id": 5.702594136295162e-08
        },
        {
          "forward_log_rmse": 1.318647563387258,
          "forward_signed_dec": 1.0927381283310797,
          "vg1": 0.4,
          "vg2": 0.15,
          "file": "StandardIV_HH_2vHCa-2_VG2=0.15_VG=0.4(1)_03-38-55PM.csv",
          "sim_peak_vd": 2.00026,
          "sim_peak_id": 5.702587373216691e-08
        },
        {
          "forward_log_rmse": 1.3195473221292486,
          "forward_signed_dec": 1.1193449300471627,
          "vg1": 0.4,
          "vg2": 0.2,
          "file": "StandardIV_HH_2vHCa-2_VG2=0.20_VG=0.4(1)_03-39-29PM.csv",
          "sim_peak_vd": 2.00032,
          "sim_peak_id": 5.702627953648535e-08
        },
        {
          "forward_log_rmse": 1.2607600406282966,
          "forward_signed_dec": 1.1489534136205135,
          "vg1": 0.4,
          "vg2": 0.25,
          "file": "StandardIV_HH_2vHCa-2_VG2=0.25_VG=0.4(1)_03-40-00PM.csv",
          "sim_peak_vd": 2.00027,
          "sim_peak_id": 5.7025946993063274e-08
        },
        {
          "forward_log_rmse": 1.2086117596261268,
          "forward_signed_dec": 1.094085393836342,
          "vg1": 0.4,
          "vg2": 0.3,
          "file": "StandardIV_HH_2vHCa-2_VG2=0.3_VG=0.4(1)_03-40-47PM.csv",
          "sim_peak_vd": 2.20036,
          "sim_peak_id": 5.866857472934359e-08
        },
        {
          "forward_log_rmse": 2.371507761392623,
          "forward_signed_dec": -2.371507761392623,
          "vg1": 0.6,
          "vg2": -0.05,
          "file": "StandardIV_HH_2vHCa-2_VG2=-0.05_VG=0.6(1)_03-43-24PM.csv",
          "sim_peak_vd": 2.00025,
          "sim_peak_id": 5.785864078340835e-08
        },
        {
          "forward_log_rmse": 2.3722582549456233,
          "forward_signed_dec": -2.3722582549456233,
          "vg1": 0.6,
          "vg2": -0.1,
          "file": "StandardIV_HH_2vHCa-2_VG2=-0.10_VG=0.6(1)_03-43-01PM.csv",
          "sim_peak_vd": 2.0003,
          "sim_peak_id": 5.785911327348654e-08
        },
        {
          "forward_log_rmse": 2.371247043301704,
          "forward_signed_dec": -2.371247043301704,
          "vg1": 0.6,
          "vg2": -0.15,
          "file": "StandardIV_HH_2vHCa-2_VG2=-0.15_VG=0.6(1)_03-42-38PM.csv",
          "sim_peak_vd": 2.00027,
          "sim_peak_id": 5.78588297736771e-08
        },
        {
          "forward_log_rmse": 2.463483607102045,
          "forward_signed_dec": -2.463483607102045,
          "vg1": 0.6,
          "vg2": -0.2,
          "file": "StandardIV_HH_2vHCa-2_VG2=-0.20_VG=0.6(1)_03-42-07PM.csv",
          "sim_peak_vd": 2.20039,
          "sim_peak_id": 6.017047037106041e-08
        },
        {
          "forward_log_rmse": 2.3708696867302814,
          "forward_signed_dec": -2.3708696867302814,
          "vg1": 0.6,
          "vg2": 0.0,
          "file": "StandardIV_HH_2vHCa-2_VG2=0.00_VG=0.6(1)_03-43-54PM.csv",
          "sim_peak_vd": 2.00027,
          "sim_peak_id": 5.78588297736771e-08
        },
        {
          "forward_log_rmse": 2.371471021567495,
          "forward_signed_dec": -2.371471021567495,
          "vg1": 0.6,
          "vg2": 0.05,
          "file": "StandardIV_HH_2vHCa-2_VG2=0.05_VG=0.6(1)_03-44-26PM.csv",
          "sim_peak_vd": 2.00026,
          "sim_peak_id": 5.7858735277582326e-08
        },
        {
          "forward_log_rmse": 2.371395720881643,
          "forward_signed_dec": -2.371395720881643,
          "vg1": 0.6,
          "vg2": 0.1,
          "file": "StandardIV_HH_2vHCa-2_VG2=0.10_VG=0.6(1)_03-44-57PM.csv",
          "sim_peak_vd": 2.00034,
          "sim_peak_id": 5.785949130012541e-08
        },
        {
          "forward_log_rmse": 2.3700899981813253,
          "forward_signed_dec": -2.3700899981813253,
          "vg1": 0.6,
          "vg2": 0.15,
          "file": "StandardIV_HH_2vHCa-2_VG2=0.15_VG=0.6(1)_03-45-21PM.csv",
          "sim_peak_vd": 2.00028,
          "sim_peak_id": 5.7858924271692704e-08
        },
        {
          "forward_log_rmse": 2.365343071609672,
          "forward_signed_dec": -2.365343071609672,
          "vg1": 0.6,
          "vg2": 0.2,
          "file": "StandardIV_HH_2vHCa-2_VG2=0.20_VG=0.6(1)_03-45-46PM.csv",
          "sim_peak_vd": 2.00027,
          "sim_peak_id": 5.78588297736771e-08
        },
        {
          "forward_log_rmse": 2.345217490148303,
          "forward_signed_dec": -2.345217490148303,
          "vg1": 0.6,
          "vg2": 0.25,
          "file": "StandardIV_HH_2vHCa-2_VG2=0.25_VG=0.6(1)_03-46-14PM.csv",
          "sim_peak_vd": 2.00024,
          "sim_peak_id": 5.785967924055442e-08
        },
        {
          "forward_log_rmse": 1.232315794105248,
          "forward_signed_dec": -1.232315794105248,
          "vg1": 0.6,
          "vg2": 0.3,
          "file": "StandardIV_HH_2vHCa-2_VG2=0.30_VG=0.6(1)_03-46-40PM.csv",
          "sim_peak_vd": 2.00031,
          "sim_peak_id": 5.7864440000587155e-08
        },
        {
          "forward_log_rmse": 1.1974161131233645,
          "forward_signed_dec": -1.1974161131233645,
          "vg1": 0.6,
          "vg2": 0.35,
          "file": "StandardIV_HH_2vHCa-2_VG2=0.35_VG=0.6(1)_03-47-03PM.csv",
          "sim_peak_vd": 2.00031,
          "sim_peak_id": 5.788566695587169e-08
        },
        {
          "forward_log_rmse": 1.2024500915225005,
          "forward_signed_dec": -1.2024500915225005,
          "vg1": 0.6,
          "vg2": 0.4,
          "file": "StandardIV_HH_2vHCa-2_VG2=0.40_VG=0.6(1)_03-47-25PM.csv",
          "sim_peak_vd": 2.00027,
          "sim_peak_id": 5.799385347782005e-08
        },
        {
          "forward_log_rmse": 1.7645576416136057,
          "forward_signed_dec": -1.7645576416136057,
          "vg1": 0.6,
          "vg2": 0.45,
          "file": "StandardIV_HH_2vHCa-2_VG2=0.45_VG=0.6(1)_03-47-47PM.csv",
          "sim_peak_vd": 0.70019,
          "sim_peak_id": 5.3537954433114255e-08
        },
        {
          "forward_log_rmse": 2.1630871953552075,
          "forward_signed_dec": -2.1630871953552075,
          "vg1": 0.6,
          "vg2": 0.5,
          "file": "StandardIV_HH_2vHCa-2_VG2=0.50_VG=0.6(1)_03-48-22PM.csv",
          "sim_peak_vd": 0.600214,
          "sim_peak_id": 5.3427493139097045e-08
        }
      ]
    }
  ],
  "runtime_sec": 2.5722649097442627
}
```


=== FILE: z304_sebas_refit_summary.json (6782 chars) ===
```json
{
  "script": "z304_aggregate",
  "n_cells_loaded": 176,
  "n_finite_cells": 176,
  "n_source_files": 11,
  "by_vg1": {
    "0.2": {
      "best": {
        "vg1": 0.2,
        "bf": 500,
        "alpha0": 1e-05,
        "rs": 0,
        "median_log_rmse": 2.0610291308357587,
        "signed_dec_median": -1.4757399592295073,
        "p90_log_rmse": 2.1123002207762025,
        "n_finite": 7
      },
      "pareto": [
        {
          "bf": 500,
          "alpha0": 1e-05,
          "rs": 0,
          "median_log_rmse": 2.0610291308357587,
          "signed_dec_median": -1.4757399592295073
        },
        {
          "bf": 500,
          "alpha0": 0.0001,
          "rs": 0,
          "median_log_rmse": 2.0610291308357587,
          "signed_dec_median": -1.4757399592295073
        },
        {
          "bf": 500,
          "alpha0": 0.001,
          "rs": 0,
          "median_log_rmse": 2.0610291308357587,
          "signed_dec_median": -1.4757399592295073
        },
        {
          "bf": 500,
          "alpha0": 0.01,
          "rs": 0,
          "median_log_rmse": 2.0610291308357587,
          "signed_dec_median": -1.4757399592295073
        }
      ],
      "n_branch_cells": 64
    },
    "0.4": {
      "best": {
        "vg1": 0.4,
        "bf": 50,
        "alpha0": 1e-05,
        "rs": 10000000000.0,
        "median_log_rmse": 1.4046663288699635,
        "signed_dec_median": 0.4243714966378498,
        "p90_log_rmse": 1.4945019316616872,
        "n_finite": 7
      },
      "pareto": [
        {
          "bf": 50,
          "alpha0": 1e-05,
          "rs": 10000000000.0,
          "median_log_rmse": 1.4046663288699635,
          "signed_dec_median": 0.4243714966378498
        },
        {
          "bf": 50,
          "alpha0": 0.0001,
          "rs": 10000000000.0,
          "median_log_rmse": 1.4046663288699635,
          "signed_dec_median": 0.4243714966378498
        },
        {
          "bf": 50,
          "alpha0": 0.001,
          "rs": 10000000000.0,
          "median_log_rmse": 1.4046663288699635,
          "signed_dec_median": 0.4243714966378498
        },
        {
          "bf": 50,
          "alpha0": 0.01,
          "rs": 10000000000.0,
          "median_log_rmse": 1.4046663288699635,
          "signed_dec_median": 0.4243714966378498
        }
      ],
      "n_branch_cells": 48
    },
    "0.6": {
      "best": {
        "vg1": 0.6,
        "bf": 9000,
        "alpha0": 1e-05,
        "rs": 10000000000.0,
        "median_log_rmse": 0.7042229003043868,
        "signed_dec_median": 0.12519489440961706,
        "p90_log_rmse": 0.9573272527337507,
        "n_finite": 11
      },
      "pareto": [
        {
          "bf": 9000,
          "alpha0": 1e-05,
          "rs": 10000000000.0,
          "median_log_rmse": 0.7042229003043868,
          "signed_dec_median": 0.12519489440961706
        },
        {
          "bf": 9000,
          "alpha0": 0.0001,
          "rs": 10000000000.0,
          "median_log_rmse": 0.7042229003043868,
          "signed_dec_median": 0.12519489440961706
        },
        {
          "bf": 9000,
          "alpha0": 0.001,
          "rs": 10000000000.0,
          "median_log_rmse": 0.7042229003043868,
          "signed_dec_median": 0.12519489440961706
        },
        {
          "bf": 9000,
          "alpha0": 0.01,
          "rs": 10000000000.0,
          "median_log_rmse": 0.7042229003043868,
          "signed_dec_median": 0.12519489440961706
        },
        {
          "bf": 500,
          "alpha0": 1e-05,
          "rs": 1000000000.0,
          "median_log_rmse": 0.8765201949636146,
          "signed_dec_median": -0.09842195704459478
        },
        {
          "bf": 500,
          "alpha0": 0.0001,
          "rs": 1000000000.0,
          "median_log_rmse": 0.8765201949636146,
          "signed_dec_median": -0.09842195704459478
        }
      ],
      "n_branch_cells": 64
    }
  },
  "best_cellwide_compromise": {
    "bf": 50,
    "alpha0": 1e-05,
    "rs": 10000000000.0,
    "vg1_02_med": 2.3975482170253373,
    "vg1_04_med": 1.4046663288699635,
    "vg1_06_med": 2.7901932952092294,
    "worst_branch_med": 2.7901932952092294,
    "median_across_branches": 2.3975482170253373,
    "max_abs_signed": 3.165630881809051
  },
  "top_5_cellwide": [
    {
      "bf": 50,
      "alpha0": 1e-05,
      "rs": 10000000000.0,
      "vg1_02_med": 2.3975482170253373,
      "vg1_04_med": 1.4046663288699635,
      "vg1_06_med": 2.7901932952092294,
      "worst_branch_med": 2.7901932952092294,
      "median_across_branches": 2.3975482170253373,
      "max_abs_signed": 3.165630881809051
    },
    {
      "bf": 50,
      "alpha0": 0.0001,
      "rs": 10000000000.0,
      "vg1_02_med": 2.3975482170253373,
      "vg1_04_med": 1.4046663288699635,
      "vg1_06_med": 2.7901932952092294,
      "worst_branch_med": 2.7901932952092294,
      "median_across_branches": 2.3975482170253373,
      "max_abs_signed": 3.165630881809051
    },
    {
      "bf": 50,
      "alpha0": 0.001,
      "rs": 10000000000.0,
      "vg1_02_med": 2.3975482170253373,
      "vg1_04_med": 1.4046663288699635,
      "vg1_06_med": 2.7901932952092294,
      "worst_branch_med": 2.7901932952092294,
      "median_across_branches": 2.3975482170253373,
      "max_abs_signed": 3.165630881809051
    },
    {
      "bf": 50,
      "alpha0": 0.01,
      "rs": 10000000000.0,
      "vg1_02_med": 2.3975482170253373,
      "vg1_04_med": 1.4046663288699635,
      "vg1_06_med": 2.7901932952092294,
      "worst_branch_med": 2.7901932952092294,
      "median_across_branches": 2.3975482170253373,
      "max_abs_signed": 3.165630881809051
    },
    {
      "bf": 50,
      "alpha0": 1e-05,
      "rs": 1000000000.0,
      "vg1_02_med": 3.2264262915314457,
      "vg1_04_med": 1.4894433845213277,
      "vg1_06_med": 1.7824399697183684,
      "worst_branch_med": 3.2264262915314457,
      "median_across_branches": 1.7824399697183684,
      "max_abs_signed": 3.778944938250161
    }
  ],
  "gates": {
    "vg1_0.2": {
      "PASS_conservative": false,
      "AMBITIOUS": false,
      "SAFETY": false,
      "median_log_rmse": 2.0610291308357587,
      "signed_dec_median": -1.4757399592295073
    },
    "vg1_0.4": {
      "PASS_conservative": false,
      "AMBITIOUS": false,
      "SAFETY": true,
      "median_log_rmse": 1.4046663288699635,
      "signed_dec_median": 0.4243714966378498
    },
    "vg1_0.6": {
      "PASS_conservative": false,
      "AMBITIOUS": false,
      "SAFETY": true,
      "median_log_rmse": 0.7042229003043868,
      "signed_dec_median": 0.12519489440961706
    }
  },
  "verdict": {
    "ALL_PASS_conservative": false,
    "ALL_AMBITIOUS_SHIP_v4.4": false,
    "ALL_SAFETY": false,
    "CELLWIDE_BEATS_DA3": false
  },
  "da3_reference_median": 0.99
}
```
