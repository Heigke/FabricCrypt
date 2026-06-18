# NRF Send Checklist — 2026-05-06

Brief is verified send-ready (oracle-approved 1/5 honesty,
visually verified 7p / 1.07 MB, all 5 figures embedded,
durable on Heigke/NSRAM commit `f670ad5`).

This is a copy-paste send sequence. Walk it from top to bottom.

---

## 1 — Final pre-send sanity (30 seconds)

```bash
cd /home/ikaros/Documents/claude_hive/AMD_gfx1151_energy
pdfimages -list results/nsram_proposal_short_v4_2.pdf | wc -l
# Expected: 13 (= 11 image lines + 2 header lines)

pdfinfo results/nsram_proposal_short_v4_2.pdf | grep -E "Pages|File size"
# Expected: Pages: 7 ; File size: 1067878 bytes (or close)
```

If either fails, **stop and rebuild from project root**:
```bash
cd /home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results
pdflatex -interaction=nonstopmode nsram_proposal_short_v4_2.tex
pdflatex -interaction=nonstopmode nsram_proposal_short_v4_2.tex
pdfimages -list nsram_proposal_short_v4_2.pdf | wc -l   # must be > 2
```

---

## 2 — Email to Mario

**To**: Mario Lanza (use his existing inbox).
**Cc**: Sebastian Pazos.
**Subject**:
```
NS-RAM 2T-cell brief — calibrated DC fit + ER_SPARSE tape-out recommendation, NRF-ready
```

**Body** — copy from
`research_plan/mario_transmittal_email_draft.md` (the section
between the `## Email body` heading and the `---` after "Best,
Eric"). Trim or adjust tone to your own preference.

**Attachments**:
  - `results/nsram_proposal_short_v4_2.pdf` (7 pages, 1.07 MB)
  - Optionally:
    `results/nsram_proposal_short_v4_2_overleaf_2026-05-05.zip`
    (582 KB — only if Mario likes editing source)

**Send.** Note the timestamp.

---

## 3 — Post-send (next 24 h)

Within ~24 h after Mario receives the brief:

1. **Send Sebas's silicon-characterisation request**:
   - Source: `research_plan/sebas_silicon_characterisation_request.md`
   - Two specific runs: $I_c/I_b$ at saturation + pulsed-$V_d$ TLP.
   - Soft ETA ask, no hard deadline.

2. **Push final eve-of-deadline state** (already done; commit
   `f670ad5`).

3. **Optional follow-up if reviewer asks where residuals
   concentrate**: send
   `figures/per_row_residuals_optimum/per_row_residuals.pdf`
   (3-panel diagnostic showing worst-5 rows all sit at
   $V_{G1}=0.40$ V).

---

## 4 — If a reviewer flags any of the known weak points

The brief explicitly addresses each — ready answers below.

| Reviewer concern | Where in brief | Defensible response |
|---|---|---|
| "Why structural floor only 4 knobs?" | §status Stage 6 + §"observed plateau" wording | Softened to "evidence requires architectural change". Three further architectures (two-NPN, quasi-2D body, body-network) listed as M3b/M6 deliverables; quasi-2D body spec already drafted (`research_plan/M3b_quasi2D_body_implementation_spec.md`). |
| "Bf=100 vs Bf=9000 inconsistency" | §"Network in action" Note on $B_f$ | Network demo runs lower-bound; calibrated $B_f$ only INCREASES $\eta$-bounded collection and reservoir expressivity. Insensitivity sweep queued post-NRF. |
| "8/33 biases excluded?" | §Limitations bullet 1 | $K_1=$ NaN at negative $V_{G2}$ in source CSV (snapback parameter extraction not done). Stamped on every 0.654 dec claim. |
| "ngspice 100 mV $V_b$ divergence?" | §status validation paragraph + §Limitations bullet 3 | Validation explicitly scoped to DC currents only; pyport adds $\eta$-bounded lateral injection ngspice omits. DC currents agree to 1–2%. |
| "0.657 (figure) vs 0.654 (text) discrepancy?" | Fig 2 caption parenthetical | Figure shows single optimum-run median; cross-bias dataset median is 0.654. 3 mdec is run-to-run scatter, not contradiction. |
| "Where are the residuals?" | (not in brief; defensible-on-demand) | Send `per_row_residuals_optimum/per_row_residuals.pdf` showing worst-5 rows all at $V_{G1}=0.40$ V. |

---

## 5 — Artifacts inventory (for reference)

Local + Heigke/NSRAM `proposal_2026_05/`:
  - `nsram_proposal_short.pdf` / `.tex` (brief v4.2-final, 7p)
  - `overleaf_bundles/nsram_proposal_short_v4_2_overleaf_2026-05-05.zip`
  - `mario_transmittal_email_draft.md` (v2)
  - `mario_nrf_onepager.md` (aligned with v4.2)
  - `sebas_silicon_characterisation_request.md` (post-NRF)
  - `M3b_quasi2D_body_implementation_spec.md` (post-NRF)
  - `figures/iv_fit/iv_fit_optimum.pdf`, `journey/journey_timeline.pdf`,
    `null_sweeps_quad/null_sweeps.pdf`, `per_row_residuals_optimum/`
  - `oracle_queries/O27_post_placeholder_swap/` (final oracle vetting)
  - `01_LOG.md` (full research log)

---

*Generated 2026-05-06 by autonomous wake-up #45.*
