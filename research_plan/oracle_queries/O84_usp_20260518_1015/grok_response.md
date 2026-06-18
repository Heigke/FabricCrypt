# grok response (grok-4-latest) — 18s

**Q1 — Strongest survivable USP**

Single 130 nm 2T cell delivering calibrated multi-function from intrinsic physics: Mario-matched LIF (V6 self-reset 40 ns + V8 integration + V7 physical Hopf at 420 ns with V_b clamped to [-0.5, +0.62] V), plus stable physical equilibrium propagation (EP) reaching 90.83 ± 2.6 % MNIST (drift recovered from -11.75 pp to +0.44 pp) via differentiable IFT pyport on the real BSIM4 cell. No rival (Loihi 2, Akida, NorthPole, BrainScaleS-2, Mythic) has published one compact device simultaneously acting as non-volatile memory, LIF neuron with verified self-reset, Hopf oscillator, and EP-compatible dynamics, all backed by silicon-calibrated DC/transient matches and a working GPU-batched differentiable model. This is the only defensible USP left; every accelerator-energy or accuracy headline is dead.

**Q2 — Reviewer attack vectors**

USP candidate (physics multi-function + physical EP + diff IFT):

1. “Does EP actually run on the physical cell or a Lyapunov surrogate?”  
   Pre-empt with K1 Jacobian audit (100 % well-conditioned), explicit statement that Scellier-Bengio theorem does not strictly apply, and the four fixes (β-cosine, random-sign, VG1-nudge, early-stop) that produced the 90.83 % result with 6× lower seed variance than the tanh surrogate.

2. “Is the 420 ns V7 Hopf real silicon physics or model extrapolation?”  
   Pre-empt with z477c clamp data showing physical V_b range and Id_pk = 4.39 mA Mario match; state V7 as “model-predicted Hopf reachable only after topology-consistent parameter lock” and flag tape-side measurement as required.

**Q3 — Death sentence experiments**

1. (Cheapest) Silicon measurement of V7 Hopf / ring-oscillator on existing or cheap test structure (pending). Failure kills V7 claim and forces full retraction of oscillation USP.  
2. 4-seed post-z474b IFT-pyport EP-NSRAM on ≥200-sample batches with no early-stop (pending). Failure kills “stable physical EP” claim.  
3. Full peripheral-inclusive energy audit (DAC/ADC + array parasitics) at iso-node for any multi-function mode (pending, highest cost). Failure kills all remaining efficiency-adjacent language.

All three remain pending; any one failure collapses the residual story.
