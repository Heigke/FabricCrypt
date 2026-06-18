# M3b/M6 — Quasi-2D body implementation spec (post-NRF prep)

**Status:** Pre-implementation scoping. No code written yet. Drafted
2026-05-05 to give the post-NRF starting work zero ramp-up.

**Why this option first:** gpt-5's O25 ranking placed quasi-2D body
(split $V_b$) as the #1 untested architecture; expected gain
$0.05$–$0.12$~dec on top of the v4.2-final 0.654-dec plateau. Two
runner-up options (two-NPN, body-network $R_b$–$C_b$) are
explicitly deferred to later in M6 — they share most of the solver
infrastructure once split-$V_b$ is in place.

---

## 1 — Physics motivation

The current pyport model treats the floating P-body as a single
lumped node $V_b$. Sebas's silicon has a finite spreading
resistance laterally between the M1 source-side body region and
the M2 drain-side body region, so during snapback the body charge
accumulates asymmetrically (M2-side fills first because it is
adjacent to the avalanche generation in M2's drain).

The minimal physical refinement: split $V_b \to (V_{b,S},\,V_{b,D})$
with a coupling resistor $R_{b,SD}$:

```
              Iii (avalanche, generates here)
                       │
        ┌──────────────┼──────────────┐
        │  M2-side body                │  M1-side body
   V_b,D ●─────[ R_b,SD ]─────● V_b,S
        │                              │
   diodes to M2.S/M2.D          diodes to M1.S/M1.D
        │                              │
        └──── BJT base contact ────────┘
              (base = (V_b,S+V_b,D)/2 or pick one — see §3.5)
```

Asymmetry expected to help most at $V_{G1}=0.4$~V (the residual
hot-spot in the per-row diagnostic), where the residual cluster
sits and the lumped-$V_b$ approximation visibly breaks (per-row
log-RMSE > 1.0 on five biases there).

---

## 2 — Code entry points

The Newton solver lives in
`nsram/nsram/bsim4_port/nsram_cell_2T.py` (1338 lines). All
necessary changes localised to:

| function                       | line  | change required                            |
|--------------------------------|-------|--------------------------------------------|
| `NSRAMCell2TConfig`            | ~78   | +3 new fields: `quasi2d_body`, `Rb_SD`, `iii_split_alpha` |
| `_residuals`                   | 397   | accept `Vb_S`, `Vb_D` instead of `Vb`; return 3-tuple |
| `_solve_jac_2x2`               | 778   | rename → `_solve_jac_3x3` for split-Vb branch |
| `_jacobian_finite_diff`        | 815   | add 3rd row/col for $R_{b,SD}$ direction   |
| `solve_2t_steady_state`        | 848   | branch on `cfg.quasi2d_body`               |
| `solve_2t_with_homotopy`       | 1110  | propagate split-Vb initial guess           |

All other downstream code (z91g sweep harness, MTK adapter, brief
plot scripts) consumes only the final $(I_d, V_{sint}, V_b)$ tuple
— extend the contract to optionally emit $V_{b,S}$, $V_{b,D}$
behind a config toggle.

---

## 3 — Detailed scope (5 sub-tasks, ~200 LOC total)

### 3.1 — Config additions (~10 LOC)

```python
@dataclass
class NSRAMCell2TConfig:
    # ... existing fields ...

    # Quasi-2D body (post-O25 architecture upgrade). When False, falls
    # back to lumped-Vb solver. When True, body splits into V_{b,S}
    # (M1-side) and V_{b,D} (M2-side) coupled through Rb_SD.
    quasi2d_body: bool = False
    Rb_SD: float = 1e6   # Ω, lateral spreading resistance
    iii_split_alpha: float = 0.7   # fraction of Iii deposited on Vb_D side
```

**Defensible defaults**: `Rb_SD = 1 MΩ` is order-of-magnitude
consistent with depleted P-body in 130 nm at body doping
$\sim 10^{17}$ cm$^{-3}$. `iii_split_alpha = 0.7` reflects the
M2-drain proximity to the avalanche-generation region. Both
should be the *first* knobs in the M6 sweep.

### 3.2 — Residual augmentation (~80 LOC)

Three residual equations instead of two:

```
R_Sint(Vsint, Vb_S, Vb_D) = same as before but with M1 body = Vb_S
R_BS  (Vsint, Vb_S, Vb_D) = M1-side junction leakage in/out + Iii*alpha
                          + (Vb_D - Vb_S) / Rb_SD  [coupling]
R_BD  (Vsint, Vb_S, Vb_D) = M2-side junction leakage in/out + Iii*(1-alpha)
                          - (Vb_D - Vb_S) / Rb_SD  [coupling, opposite sign]
```

Coupling current $I_{SD} = (V_{b,D} - V_{b,S})/R_{b,SD}$ enters
both body residuals with opposite signs, ensuring KCL closure.
Avalanche current $I_{ii}$ from M2's impact-ionisation is split
$(\alpha, 1-\alpha)$ between the two body nodes.

The BJT base voltage: define $V_{b,\text{base}} = \text{mean}(V_{b,S},
V_{b,D})$ for the simplest defensible choice, with a config toggle
to use $V_{b,S}$ or $V_{b,D}$ alone if needed.

### 3.3 — Jacobian (~40 LOC)

Finite-difference with $\Delta V = 10^{-4}$~V per node, 6
residual evaluations per Newton step (3 residuals × 2 directions
each, with the cross terms shared). Reuse the existing FD
infrastructure; only add the third-axis perturbations.

3×3 linear solve via explicit closed form or `torch.linalg.solve`.
Closed-form keeps batched-bias autograd fast; `solve` is simpler.
Recommend: closed-form Cramer's rule on the determinant for
3×3 — keeps the branchless path that lets `torch.compile` work.

### 3.4 — Initial guess (~20 LOC)

Lumped-$V_b$ converges first (existing solver), then expand to
$(V_{b,S}, V_{b,D}) = (V_b, V_b)$ as warm start; one or two
extra Newton iterations should close the asymmetry. This avoids
re-tuning the homotopy schedule.

### 3.5 — Tests + downstream plumbing (~50 LOC)

- `tests/test_quasi2d_body.py`: round-trip — at $R_{b,SD} \to 0$
  the split-Vb solver must reproduce the lumped-Vb solver's
  $(I_d, V_{sint})$ to $\le 10^{-9}$ relative error on the
  33-bias grid.
- `tests/test_quasi2d_body.py::test_asymmetry_at_VG1_0p4`:
  at $V_{G1}=0.4$~V, $V_{b,D} > V_{b,S}$ should hold for
  all $V_D > 1.5$~V (the snapback regime, where avalanche
  deposits more charge on the M2 side).
- z91g harness: add `--quasi2d` flag that flips the config
  toggle and runs the full 33-bias refit.

---

## 4 — Validation plan (post-implementation)

1. **Sanity**: $R_{b,SD} \to 0$ → matches lumped-Vb on all 33
   biases (regression-test gate).
2. **Headline refit**: 5×5 sweep over $(R_{b,SD}, \alpha)$ at the
   v4.2-final $(B_f, V_a, I_s)$ optimum; report median log-RMSE
   delta vs lumped-Vb baseline.
3. **Per-row residual map**: regenerate the diagnostic figure
   (`figures/per_row_residuals_optimum/`) under quasi-2D body. The
   $V_{G1}=0.4$~V residual cluster is the load-bearing test —
   if it lifts, gpt-5's prediction is confirmed; if it doesn't,
   move on to two-NPN.
4. **Network-demo regression**: re-run the 64-cell ER\_SPARSE
   reservoir on Mackey–Glass; verify NRMSE does not degrade
   (the new physics adds expressivity, should be neutral or
   better).

---

## 5 — Risk + fallback

- **Newton convergence**: 3×3 system inherently harder to
  converge than 2×2, especially at the snapback fold. Mitigation:
  the existing pseudo-arclength continuation (line 1110 onwards)
  generalises directly; the only added cost is one more Jacobian
  per step.
- **Differentiability**: torch autograd handles 3×3 closed-form
  fine; only risk is if Cramer's rule numerical conditioning
  degrades at low $\det(J)$. Mitigation: fall back to
  `torch.linalg.solve` with gmin-shunt.
- **If gain < 0.05 dec**: stop, write up as null, move on to
  two-NPN (next M6 deliverable). Don't sink M6 schedule into a
  marginal architecture refinement.

---

## 6 — Effort + schedule

- **Implementation**: 1–2 days focused work (~200 LOC + tests).
- **Validation**: 0.5 day (5×5 sweep + figure regeneration).
- **Wall-budget**: 2–3 days from cold start; can be triggered
  on any post-deadline session.
- **Dependency**: none — independent of Sebas's silicon-
  characterisation runs (that work pins down the *current*
  parameters; this work explores a *new* model class).

---

*Drafted 2026-05-05 ~22:30 by Eric. Pre-implementation scoping
only. Acts on after NRF deadline + Sebas request packet are sent.
Trigger condition: "we want to test gpt-5's #1 architecture
prediction without waiting for silicon data".*
