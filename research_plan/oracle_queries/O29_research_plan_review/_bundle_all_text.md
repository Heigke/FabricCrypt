# Combined text bundle for oracle context

All text artifacts concatenated below for one-shot context. Each file is delimited by a `=== FILE: <name> ===` marker.



=== FILE: plan.md (7360 chars) ===
```
# NS-RAM Research Plan — Post-NRF (2026-05-07)

**Status of brief**: v4.2-final sent to Mario / NRF on 2026-05-06.
Oracle-vetted (O26 + O27 + O28). 0.654-dec headline confirmed
defensible by both gpt-5 and gemini in O28 against the multi-root
finding from Plan A quasi-2D investigation.

**Posture**: brief is shipped. From here on, all work is
*post-deadline due-diligence + architecture upgrades + network
demonstrations* aimed at the M3b/M6/M9/M12 milestones in the
brief itself.

---

## A — Pending phase-A closure (legacy threads)

| ID | Item | Blocker | Priority |
|----|------|---------|----------|
| A.4 | Transient validation | Sebas data | wait |
| A.6 | Robert Julia cross-val | Robert availability | wait |
| A.12 | Thick-ox cell card request | Compose+send to Sebas | **DO NEXT** |
| F3 | z142 topology v2 (10 seeds, fair-ρ) | Compute window | low (post-Mario) |
| F7 | ngspice cross-val at NEW optimum (Bf=9000, Va=0.55) | None | **URGENT GATE** |

**Action**: F7 first — defensibility gate for the brief's claims.
Bundle with B.1 (multi-root ngspice cross-check, see below).

---

## B — Plan A quasi-2D body model continuation

**Status**: wrapper code in `nsram/nsram/bsim4_port/nsram_cell_2T.py`
(~250 LOC, opt-in via `cfg.quasi2d_body=True`). Plan A blocked on
multi-root behavior: 3×3 Newton walks past lumped's near-physical
root to a numerically-deeper alt-root that doesn't match silicon.

Per O28 oracle consensus (gpt-5 + gemini), the path is:

| ID | Item | Effort | Dependencies |
|----|------|--------|--------------|
| B.1 | ngspice cross-check 3 biases at production params (Bf=9e3, Va=0.55, Is=1e-9) → certify physical branch independently | 2-3 h | F2 harness re-use |
| B.2 | Branch-protection in `solve_2t_quasi2d_steady_state`: reject Newton steps where ‖ΔVb‖∞ > 50 mV; force damping reduction | 30 min | B.1 ground-truth |
| B.3 | Add 10–100 GΩ body-leak regularizer term to `_residuals_quasi2d` (gpt-5's suggestion: erases the latch-up alt-root) | 1 h | B.2 |
| B.4 | Build production-quality test harness — replicate `z91g_two_model_validation.py` setup (env-driven Bf/Va/Is, per-bias make_overrides, mbjt scaling) | 2-3 h | B.3 |
| B.5 | Compare lumped vs. quasi-2D on 25 evaluated biases at production params; report median log-RMSE delta | 1 h | B.4 |
| B.6 | If gain ≥ 0.02 dec: 5×5 sweep over (Rb_SD, α) at the new optimum to find quasi-2D fit point | 4-6 h | B.5 positive |

**Decision tree** at B.5:
- **gain < 0.02 dec**: Plan A is null; pivot to either two-NPN
  (D path below) or body-network. Document the null.
- **gain in [0.02, 0.05] dec**: marginal; finish B.6 to characterize
  but don't sink M6 schedule.
- **gain > 0.05 dec**: Plan A wins; proceed to Plan B (real refactor
  with M1.body=Vb_S, M2.body=Vb_D) for additional ~0.05 dec.

---

## C — Plan B full quasi-2D refactor (conditional)

**Trigger**: Plan A B.5 gain > 0.02 dec.

| ID | Item | Effort |
|----|------|--------|
| C.1 | Refactor `_residuals` to take `Vb_M1, Vb_M2` instead of single `Vb`. M1 body diodes use Vb_M1; M2 uses Vb_M2; BJT base = mean (or configurable) | 1 day |
| C.2 | Update z91g harness for new residual signature; verify F1.v2 backward compat | 0.5 day |
| C.3 | Re-run 33-bias fit at quasi-2D refactored solver | 0.5 day |
| C.4 | Compare against Plan A wrapper result | 0.5 day |

---

## D — Other architecture options (M6 deliverables)

Listed by gpt-5 O25 priority. Each is independent of others — only
trigger if Plan A path closes.

| ID | Architecture | Expected gain | Effort | Trigger |
|----|--------------|---------------|--------|---------|
| D.1 | Two-NPN (parallel collector+emitter parasitic NPNs at M1+M2) | 0.03–0.08 dec | 2-3 days | Plan A null |
| D.2 | Body-network (Rb–Cb distributed RC) | 0.02–0.05 dec | 3-4 days | Plan A + D.1 marginal |
| D.3 | S–Vb diode (NS-RAM-specific snapback diode) | 0.01–0.03 dec | 1-2 days | Backup |

---

## E — M3b silicon ground-truthing

**Trigger**: Sebas accepts characterisation request (drafted in
`research_plan/sebas_silicon_characterisation_request.md`).

| ID | Item | Wall budget |
|----|------|-------------|
| E.1 | Send Sebas request packet (post-NRF) | 5 min |
| E.2 | Receive $I_c/I_b$ saturation data → extract silicon Bf | 1 day after data |
| E.3 | Receive pulsed-Vd/TLP transient → extract $R_b \cdot C_b$ | 1 day after data |
| E.4 | Re-fit z91g at silicon-grounded $(B_f, R_b)$ point; report median log-RMSE vs. brief's 0.654 | 1 day |

**Decision tree at E.4**:
- Silicon $B_f \in [10^3, 10^5]$ → brief's calibration confirmed
  physical → tape-out recommendation stands.
- Silicon $B_f$ out-of-range → revise brief, may need quasi-2D
  to absorb the discrepancy.

---

## F — Network experiments at scale (B.5 closure)

| ID | Item | Wall budget |
|----|------|-------------|
| F.1 | Wire thread cap into z200's `run_config` (the gating issue for N≥2048) | 1 h |
| F.2 | z202 extension to N=8192, 3 seeds at WS\_SMALLWORLD/ff (resume the run that crashed) | 4-6 h compute |
| F.3 | 5-seed reproducibility test on best ER\_SPARSE / Mackey–Glass result | 2 h |
| F.4 | Hard-benchmark suite at calibrated cell: XOR τ=2, NARMA-10, memory capacity, multi-class waveform | 4-6 h |

---

## G — Dissemination

| ID | Item | Wall budget |
|----|------|-------------|
| G.1 | Confirm Mario received the brief; address any questions | reactive |
| G.2 | Send Sebas characterisation request | 5 min |
| G.3 | Conference paper draft target — NICE 2026 (neuromorphic-hardware venue mentioned in brief budget) submission window typically Jul-Aug | 1-2 weeks once results from E + F land |
| G.4 | Upstream ngspice bug reports (the 5 calibration-loop bugs from M3a) — send catalogue + fix patches to SimuCAD | 1 week, low priority |

---

## H — Non-brief side projects (cool but lower priority)

| ID | Item | Status |
|----|------|--------|
| H.1 | Local-learning v2 rebuild + GPU + N=128 | in_progress (#147) |
| H.2 | BEAM byte-level associative memory (already explored, see memory) | dormant |
| H.3 | FEP-Mem online learning (memory entry exists) | dormant |

---

## Priority order across all phases

**Week 1 (immediate)**:
1. B.1 — ngspice cross-check at 3 biases (gating decision)
2. F7 — ngspice cross-val at new optimum (defensibility)
3. A.12 — send thick-ox card request to Sebas
4. G.2 — send Sebas characterisation request

**Week 2-3**:
5. B.2 + B.3 — quasi-2D Newton hardening
6. B.4 + B.5 — production harness + 25-bias comparison
7. F.1 + F.3 — z200 thread-cap + reproducibility

**Week 4+**:
8. (B.5 outcome dependent) → C or D
9. (Sebas data) → E.2 / E.3 / E.4
10. F.4 — hard-benchmark suite
11. G.3 — conference paper draft

---

## Cron strategy (autonomous loop)

The current cron `abd4f469` fires every 30 min and does generic
"pick highest-value safe task". After 110+ wake-ups it's mostly
stand-down because the brief is done.

**New cron strategy**:
- Replace generic 30-min cron with a more focused schedule:
  - **Every 30 min during work hours (08:00–22:00)**: continue
    research-plan execution, picking next item from A-H by priority.
  - **Daily 02:00**: deep-work synthesis — review log, push backup,
    re-prioritize plan.
  - **Weekly Mon 09:00**: full plan review + status report to user.
- Each wake-up is autonomous and durable (auto-expires 7 days).

(Cron setup happens after this plan is oracle-reviewed.)

```
