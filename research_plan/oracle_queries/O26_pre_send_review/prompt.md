# O26 — Final pre-send review of NS-RAM proposal short v4.2

## Context

Brief is sealed for NRF deadline 2026-05-06 (tomorrow). Two prior
review rounds (O14, O17, O18) caught real issues; this is a final
sanity check before send.

The attached PDF (results/nsram_proposal_short_v4_2.pdf, 7p, 749 KB)
is what would go to Mario Lanza tomorrow.

## Headline (current)

  - **0.654 dec** median log-RMSE on 25/33 evaluated biases of
    Sebas's 33-row dataset
  - Honest physical optimum: Bf=9×10³, Va=0.55 V, Is=1×10⁻⁹
  - **Six productive sweeps** drove 1.39 → 0.654 dec
  - **Four oracle-ranked candidates** (IKF, ISE/NE, PRWG/Rdsw,
    η(Vbe) sigmoid) came up null at 0.003 dec spread → confirms
    structural floor of lumped-Vb / single-NPN architecture
  - Network demos: 9/48 z200 configs reach ≥0.9 best test acc;
    4 topologies stable at 0.833 final; N=1024 sweet spot
  - Architectural recommendation: **ER_SPARSE** for tape-out

## Three figures we added vs prior v4.1

  1. **Journey timeline** (figures/journey/journey_timeline.pdf):
     10-step bar chart, productive-vs-null distinction, structural-
     floor band annotation
  2. **Null-sweeps quad** (figures/null_sweeps_quad/null_sweeps.pdf):
     2×2 heatmap grid showing each null sweep cell-by-cell
  3. **Stage 5/6 textual additions** in §status

## QUESTIONS for review

1. **Honesty check**: Does the brief overstate any claim? In
   particular: the "four nulls confirm structural floor" framing —
   is it appropriate, or should it be softened to "four candidates
   have been excluded; remaining candidates require architecture
   change"?

2. **Figure 1 (journey timeline)**: Is the "structural floor band"
   visualisation across the 4 null steps clear and defensible? Or
   could a reviewer attack it as cherry-picking the null candidates?

3. **Stage 6 framing**: We list four nulls, then say "going below
   0.65 dec requires architecture change". A reviewer could ask:
   "Did you try the other three architecture options gpt-5
   ranked?" (Two-NPN, body-network, S-Vb diode). We DIDN'T test
   those, only η(Vbe). Should we mention that explicitly?

4. **Network demo §3b**: We claim "9/48 configs reach ≥0.9 best,
   4 topologies stable at 0.833 final". The "best" peak is a
   transient max during training; 0.833 final is what the network
   reaches at end. Is this distinction clear enough, or is "best"
   misleading?

5. **What to cut**: The brief is 7 pages. NRF panels typically have
   short attention spans. Is anything cuttable while preserving
   the core message? Specifically targeting: any redundancy between
   §3 textual table + §3 Stage 6 quad figure?

6. **Last-mile risks**: Any technical claim that, if a reviewer
   asks one specific question, we'd struggle to defend? Where are
   the weakest points in the argumentation?

Please be a hostile reviewer. <500 words.
