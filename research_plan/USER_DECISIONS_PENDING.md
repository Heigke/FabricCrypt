# Decisions pending from user — 2026-05-10 18:39 (post-z243)

**Story has settled.** This update finalizes the post-z242 + post-z243
state. Both ESN attribution tests are now done. NS-RAM is purely a
silicon-energy story; brief is locked to two defensible headlines
plus an honest "ESN-class-not-better" qualifier.

---

## Story state — what the data actually says now

| Strand | Result | Status for Mario v2 |
|---|---|---|
| Energy headline | NS-RAM ~0.7 µJ vs MAX78000 5 µJ vs Cortex-M4 50–100 µJ at 1024-step / N=64 | ✅ **Lead with this.** Silicon-level, independent of bench compute. |
| NARMA-10 | NS-RAM NRMSE 0.612 ± 0.030 (z223 30-seed); ESN at same N reaches 0.563 (z243 30-seed), 8% better, CIs disjoint | ✅ Lead as **"ESN-class accuracy at the silicon-energy floor"** (not "beats ESN"). |
| R-track triangulation | surrogate ↔ pyport ↔ ngspice ≤ 0.51 dec | ✅ **Lead with this.** Physics credibility. |
| Cross-task image (within MNIST-family band) | Linear Δ ≈ +30 − 0.56·proj%, R²=0.997 within 43–72% | ⚠️ **Real but pipeline-level.** A textbook tanh ESN at the same N=1000 gives Δ=+27 pp on MNIST vs NS-RAM's +5 pp (z242, p=8e-11). Mention as internal calibration only; do not lead. |
| Cross-task outside band (CIFAR proj=15%) | Δ=+1.94 pp, sign matches but linear extrapolation fails by ~10× | Honest scope-bound. |
| Hyperparam robustness (g_VG2 sweep z241) | Smooth gradient, no winner's curse | ✅ Robustness checked. |

## What changed in the last 24h

- **z240** (CIFAR-10 modality test): linear extrapolation of the
  monotonic fit FAILS outside MNIST-family. Sign holds, magnitude
  saturates ~10× lower than predicted. Mario v2 first scope-bound to
  43–72% MNIST band.
- **z241** (g_VG2 sensitivity sweep): smooth approximately linear
  gradient across 5 hyperparam values. **Winner's-curse risk
  rejected.** Mario v2 strengthened.
- **z242** (ESN attribution control): **ESN beats NS-RAM by 22 pp on
  the SAME pipeline (Δ_ESN = +26.94 pp vs Δ_NSRAM = +5.10 pp).** The
  cross-task gain we documented is a property of "any reservoir +
  this readout pipeline," not of NS-RAM specifically. Mario v2
  fundamentally re-pitched as a HARDWARE-EFFICIENCY story rather than
  a reservoir-quality story.

---

## Decision 1 — Send Mario v2 update note (now in MAJOR-REVISION form)?

**File**: `research_plan/mario_update_note_v2_draft.md` (last edited
2026-05-10 ~14:38, post-z242 revision).

**What it now says**:
- Subject acknowledges "honest attribution finding... hardware-
  efficiency story, not a reservoir-quality story."
- Cross-task section LEADS with z242 (ESN beats NS-RAM by 22 pp).
- Recommendation: lead Mario with energy + NARMA + R-track. Do NOT
  lead with cross-task image work — a textbook ESN beats us.
- Drops the "complement to weak baselines" framing as misleading.
- Within-band experiments kept for completeness as internal calibration.

**This is the most defensible state Mario v2 can be in.** Better to
send THIS than risk Mario running his own ESN check on the prior
"NS-RAM-as-reservoir" framing — that would unwind our credibility.

**Decision needed**: Send revised v2 today / hold for more compute
(see Option A below) / want a different framing?

## Decision 2 — Send Sebas characterisation request?

**File**: `research_plan/sebas_silicon_characterisation_request.md`
(drafted 2026-05-05, **6 days unsent today** — well past 5-day flag).

Status unchanged. Audit confirmed substantively still valid. Two
silicon measurements (Bf at saturation, τ via TLP) are still the
right asks regardless of the cross-task narrative shift.

**Decision needed**: Send / want edits?

## Decision 3 — Bundle Sebas thick-ox addendum?

**File**: `research_plan/sebas_thick_ox_request_addendum.md`. Still
valid as drafted. Bundle with Decision 2.

---

## Optional compute experiments (if you want to firm up before sending)

**Option A — DONE (z243)**: NARMA-10 ESN comparison ran. ESN NRMSE
0.563 vs NS-RAM 0.612, 8% better, CIs disjoint. Mario v2 narrative
is now finalized: "ESN-class at silicon-energy floor."

**Option B (lower priority): Sweep ESN g_in to verify z242 robustness
(~20 min)**. Skipped — both ESN tests independently show ESN beats
NS-RAM, so g_in fairness is no longer load-bearing.

**Option C (lower priority): Out-of-band datapoint (SVHN/EMNIST)
(~30 min)**. Cross-task is no longer brief lead, so lower priority.

**My recommendation now: SEND Mario v2 today.** The brief has been
revised post-z242 + post-z243. The two defensible headlines (energy
+ physics) plus the honest NARMA caveat are stable. Further compute
won't strengthen the brief — it would only delay user decisions on
Mario and Sebas.

---

## Status of project deliverables (for context)

| Deliverable | State |
|---|---|
| NRF brief v4.1 | Sent earlier with 2026-05-06 deadline |
| Mario update note v2 | **Drafted post-z242 revision, awaits send** |
| Sebas characterisation request | **Drafted 6d, awaits send** |
| Sebas thick-ox addendum | **Drafted 3d, awaits send** |
| NS-RAM 4D surrogate + reservoir | Complete (z220/z221/z223) |
| R-track triangulation | Closed (z230/z231) |
| C-track energy calibration | Closed (10× advantage) |
| Cross-task within-band study | Complete; pipeline-level per z242 |
| Cross-task out-of-band study | One CIFAR datapoint; saturation observed |
| ESN attribution test | DONE on MNIST; **major reversal**, 22pp gap |
| ESN attribution on NARMA-10 | **Not yet run** — Option A above |

## What I'll do while waiting

Cron jobs continue: resource audits, baseline regression, daily
syntheses. If you don't redirect, I will run **Option A (NARMA-10
ESN comparison)** on the next GPU off-hours window — it is the
single most informative remaining experiment.
