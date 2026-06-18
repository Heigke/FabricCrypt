# openai response (gpt-5) — 245s

Q1 — Best next experiment to find the missing parallel path
Ranking by likelihood for a flat Id vs Vd (~250 nA plateau at low VG1):
1) (b) STI/corner-dominated drain–body junction leakage (SRH/perimeter + sidewall TAT). Rationale: at small Vd (triode sweep 0–100 mV) junction generation current is nearly Vd-invariant, so it looks like a current source and dominates when channel is off. This subsumes “JTSS/JTSD but with correct amplitude and perimeter scaling,” not a Vbs-modulation artifact.
2) (d) Field-assisted TAT/GIDL tail near the drain edge. Even at low Vd, a large |Vg1| can set up high lateral field at the drain corner; plateau then depends more on Vgd than on Vds, hence “flat vs Vd.”
3) (e) Pad/ESD leakage or unintended rail path (IO diode stack, protection network, or fixture floor). 250 nA, Vd-invariant, is plausible and would not be in the thick-ox card.

Falsifiers for top-2:
- For (b) STI/junction leakage:
  • Arrhenius T-sweep on silicon at fixed (VG1 low, Vd=50 mV). Expect strong activation (Ea ~0.5–0.7 eV) and 5–10× rise from 25→85 C if SRH/perimeter dominates. Flat-with-T would falsify SRH and point to instrument/ESD.
  • Junction-off control in pyport: set JS, JSD, JTSS/JTSD, AGIDL/BGIDL/EGIDL = 0 and clamp Vbs (Vbs-clamp control). If the plateau remains, (b) is falsified; if it vanishes, (b) is confirmed and the required amplitude/perimeter model can be backed out.
- For (d) TAT/GIDL:
  • Gate-field probe (VG1-nudge test): at fixed Vd=50 mV, step VG1 by −100/−200 mV. TAT/GIDL should change the plateau by ≥5×; SRH/perimeter changes weakly. Do the same in pyport with Vbs clamped to kill the Vbs-modulation band-aid called out in O87.
Notes: (a) well-tap forward diode is unlikely (BSIMSOI smoke Vbs never forward); (c) Schottky at source/body is unlikely in 130 nm salicided n+ (ohmic contact).

Q2 — Large-scale sim opportunities (still valuable with an imperfect cell)
- Bias-robustness/yield atlas (N=10^5 virtual cells):
  • Task: Monte Carlo PVT + mismatch (use Mario scatter + foundry-like σ) to label mode per cell (LIF, reset, off). Output “safe-bias” heatmaps.
  • Metrics/targets: safe-region area ≥70% at TT±3σ; mis-mode probability ≤5%; mode margin (ΔV needed to flip) distribution.
- Tile-level spatial correlation study (16×16 to 128×128):
  • Task: inject distance-based correlated mismatch; predict spatial clusters of failure for LIF/reset.
  • Metrics: corr(length) vs die distance; cluster size CDF; predicted bad-tile rate for 16×16 test chip (ties directly to DS-3).
- Calibration-at-scale with diff IFT pyport:
  • Task: per-cell bias solver that drives each cell to a target transient (V6 reset τ≈40 ns or V8 LIF rate), starting from random PVT draws.
  • Metrics: calibration success rate ≥90%; iterations-to-converge; final residual vs Mario; bias headroom used.
- Power/IR/thermal grid emulation:
  • Task: place N=10^4–10^5 cells on a mesh, use current envelopes to estimate IR drop and ΔT with simple Rgrid/Cth model.
  • Metrics: worst-case droop ≤5% at activity A; ΔT map and thermal time constants; sensitivity of safe-bias area to 50 mV IR and +10 C.
- Do not promise learning benchmarks. If anything, show a “physics-aware activation” toy only as methodology; no claims of superiority.

Q3 — Pivot decision tree (Day-7 if Pillar A fails)
Default: pivot to a pure methods paper; stop claiming accelerator benefits.
- Gate G0: DS-1 (EP-FIX, no early-stop) median final accuracy ≥88% over 10 seeds? If NO (current trend), remove all “trains a network” claims.
- Gate G1: Pillar A (parallel path) fails to cut median DC gap to ≤2.5 dec and fix the triode knee by Day 7? If YES:
  • Pivot to Methods-only paper: differentiable port (diff IFT pyport), calibration workflow, multi-metric audit tooling, and a candid negative result section (DC gap, V7 knife-edge, Lyapunov caveat for EP). Deliverables: open code, full reproducibility, ablations, and a “killshot playbook.”
  • Keep only robust device functions: V6 reset and V8 LIF mode atlas (qualitative), clearly caveated; no V7; no reservoir/EP claims beyond methodology.
- Optional Plan B (only if Mario wants algorithmic utility): ML emulator of the measured 7-rate transient with uncertainty bands, trained on Mario’s data, explicitly decoupled from physics. This is a separate “surrogate modeling” note; do not entwine with NS-RAM claims.
- Stop conditions:
  • If DS-1 final <80% median or DC median gap stays >3.0 dec on full grid by Day 7, halt all new NS-RAM modeling lines and reassign compute to methods polish, datasets, and documentation.
Rationale: Methods + negative results are defensible and citable; continued model tuning without data risks more knife-edges.
