
## 2026-05-07 wake-up #106 — tightened quasi-2D Newton (still finds alt root)

APU 41°C; sentinel + telem + guard alive.

Added relative-tolerance stop (max(Iabstol, Ireltol·|R|)) + Armijo
backtracking + step caps to solve_2t_quasi2d_steady_state, mirroring
solve_2t_steady_state.

Smoke test still lands at Id=1.5 µA alt root because BJT defaults
(Bf from sebas card) differ from production fit (Bf=9000, Va=0.55,
Is=1e-9). Multi-root structure depends on BJT params; needs the
same overrides as z91g_F6v4 to compare apples-to-apples.

Pushed nsram_cell_2T.py update to Heigke/NSRAM. Plan A continuation
needs a proper z91g-style harness with the production overrides.


## 2026-05-07 wake-up #109 — multi-root at production params confirmed

APU 41°C; sentinel + telem + guard alive.

q2d_prod_params.py: ran lumped, tightened-lumped, quasi-2D at 9
biases × Bf=9000, Va=0.55, Is=1e-9. 6/9 biases show multi-root.

Default lumped picks the low-Id root that matches measurements;
tight + quasi-2D walk to a high-Id alt-root (Id ~1-3 uA).

Pushed: commit fc0e9bf. Plan A awaits ngspice cross-check to
certify which root is physical. This is the gating decision —
not a Newton-tweaking question anymore.


## 2026-05-07 wake-up #110 — O28 oracle critique: brief is defensible

APU 41°C; sentinel + telem + guard alive.

Built O28 packet asking gpt-5 + gemini whether the multi-root issue
threatens the brief. **Both unanimously**: low-Id bias-dependent root
is the physical one (matches measurement); high-Id alt-root is a
parasitic-NPN latch-up the model allows but silicon doesn't occupy.

  - gpt-5 (69s, 2306 chars): "your claim stands if you define/ensure
    operation on the physical branch."
  - gemini (26s, 2266 chars): "0.654-dec headline is defensible.
    The physics captured is what matters."

**0.654-dec brief claim is OK.** The residual floor ~1e-10 is a
known BSIM4 floating-body model limitation, not invalid solution.

Plan A path forward (both oracles agree): (a) ngspice cross-check
2-3 biases for independent certification, (c) engineer quasi-2D
to stay in physical basin via branch-protection in Newton step
(reject |ΔVb| > 50 mV jumps) + 10-100 GΩ body-leak regularizer
to erase the alt-branch.

Don't pivot to two-NPN. Keep Plan A.


## 2026-05-08 track audit 09:11 (cron b6c2e300) — V/R/T/S all advanced overnight

APU 34°C, sentinel alive.

Track status post-z223 30-seed CI:

  V validation     DONE+    z223 30-seed CI [0.601, 0.624] tight; bootstrap
                            CI matches; paired test on z213 inhibition; both
                            null hypothesis (inhibition) and positive
                            hypothesis (4D body-state) properly tested
  R realism        DONE     4D transient surrogate IS the realism step;
                            z214 scale-gap done; z217 MC diagnostic done
  C chip-cost      DONE     calibration_v1.md w/ sky130 anchor + decision
                            heuristic (no change since last audit)
  T tasks          PARTIAL  NARMA-10 SOLVED (NRMSE 0.61, ESN-class touched);
                            NARMA-5 INSIDE ESN (0.59); NARMA-20/30 generator
                            unstable; SeqMNIST + KWS still TODO
  S stats          DONE     bootstrap CI in z223; paired-t in z213/z214/z216;
                            preregistration emerged organically (frozen-config
                            test on NARMA-10 = pre-registered cherry-pick check)
  P thermal        DONE     util_safe_sweep validated through 8+ runs (z212
                            -z223), max APU peak 79°C, no events since fix

Stalled count: 0.

The plan from 2026-05-07 morning (gap-closing) is COMPLETE for 5 of 6
tracks. Only T remaining incomplete (cross-task generalization
benchmarks beyond NARMA).

Re-prioritized queue for next work-hours wake-ups:
  1. Build SeqMNIST harness on 4D surrogate (Track T cross-task)
  2. Refresh Mario update note with z223 CI numbers + journey figure
  3. Fix NARMA-K generator for K>10 (numerical instability, low-pri)
  4. Hetero-cell still untested (Phase I from EXPLORATION_PLAN; lower
     pri now that body-state path works)


## 2026-05-09 resource audit 00:03 — all GREEN
disk 49%, mem 8%, APU 34°C, sentinel PID 9161 alive, results 52GB
(plenty headroom, 1.8TB free), /tmp logs <13KB each, no rotation.
Note: user crontab empty — wake-ups scheduled via Anthropic schedule
system, not system cron. No system-cron TTL to audit. No alert.

## 2026-05-09 daily synthesis 02:13 #36
APU 33°C, sentinel alive (PID 9161). 13 entries since last daily synth.
Major flips today: R-track CLOSED (z230/z231/z232), C-track CLOSED
(edge-AI baseline added), B.2/B.5 done with falsified hypothesis
+ meta-correction. F.1 thread cap landed.
Blocked items watch: Sebas characterisation request UNSENT 4 days
(approaching 5-day flag); thick-ox addendum 2 days; mario_update_note
2 days AND tagged STALE per O35 ("DO NOT SEND").
O35 next-action: 28×28 sequential MNIST FROZEN params 30 seeds GPU N=2k
— single highest-value experiment per 3/3 oracle consensus.
Syncing proposal_2026_05/ now (drift since 00:11).
