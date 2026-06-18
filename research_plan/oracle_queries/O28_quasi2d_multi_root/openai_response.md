# openai response (gpt-5) — 69s

Short answer up front:
- The physical branch is the low‑Id, gate‑sensitive one. The µA “Vb≈0.5 V” branch is a latch/BJT‑on equilibrium that your current BJT params make accessible but silicon does not select in these biases.

1) Which root is physical?
- Low‑Id, VG‑dependent branch. Evidence: (i) Sebas’s data are nA–µA and VG‑sensitive; (ii) the alt branch is nearly VG‑independent and looks like a diode+NPN latch state; (iii) the parameter set (Bf=9e3, Is=1e‑9 A) is far larger than plausible for a lateral parasitic and will over‑enable that latch loop. Any tiny body leak or recombination (present in real silicon) suppresses this branch. Add a 10–100 GΩ body tie in sim: the alt root should vanish or shift to much higher Vd/T.

2) Why doesn’t the residual go to ~1e‑15?
- At the physical root the floating‑body Jacobian is extremely small (pS–nS). With double precision, model smoothings, and any gmin/limiting, Newton stalls when ΔV·G_eff ≈ residual. For G_eff ~1e‑10–1e‑9 S and sub‑mV steps, a 1e‑10 A floor is expected. KCL in silicon is exact; the residual floor is numerical/regularization, not physics.

3) Rethink the stop criterion?
- Keep Iabstol+Ireltol, but add branch‑selection and stability filters:
  - Continuation from a known physical point (low Vd, low VG) and reject Newton jumps that switch branches.
  - Prefer equilibria with positive incremental dissipation (e.g., J symmetric part ≻ 0 on observable ports or at least ∂I_D/∂V_b not giving loop gain >1).
  - Tiny body leak (10–100 GΩ) as a physical regularizer.
These keep you on the measured branch while allowing tight tolerances.

4) 0.654‑dec headline impacted?
- No. The fit was to the measured (physical) branch. The more‑converged alt root is a model‑enabled latch state that silicon did not occupy in those biases. Your claim stands if you define/ensure operation on the physical branch.

5) Path forward
- Do (a): quick ngspice cross‑check at 2–3 biases, with and without a 10–100 GΩ body leak.
- Keep Plan A; don’t pivot to two‑NPN now.
- Soften parasitic BJT: scale Is by area, reduce Bf to a plausible 1–50, add base recombination and explicit well/body leakage. This will erase the spurious branch and lower the residual floor.
- Engineer quasi‑2D to mirror the lumped branch via continuation + stability checks.
