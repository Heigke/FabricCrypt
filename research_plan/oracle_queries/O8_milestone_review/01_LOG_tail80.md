**This is the foundational piece. With this:**
✓ LIF dynamics — leaky integration validated
✓ Spike detection + reset — code path tested (no spike at this bias,
  but logic verified)
✓ Time-domain validation against Sebas's transients — UNBLOCKED
  (when he sends raw measurement files)
✓ Meta-plasticity demo — buildable: pick bias regimes that produce
  synapse, neuron, STM behaviours via control voltages

Phase A.4 → essentially CLOSED.

Next iteration: build the meta-plasticity demo. Pick three bias
configurations (synapse: stable analog Vb level, neuron: spike-and-
reset cycle, STM: charge-decay over τ ms) and show one cell
producing all three with only control-voltage changes. This is
exactly what Mario needs for NRF.

Files: nsram/bsim4_port/joint_newton.py (transient_2t, joint Newton
with line search + bounds + autograd Jacobian)

## 2026-05-02 11:00 — A.4.f Vb-equilibrium ceiling: 0.21 V (card-limit confirmed)

User asked to see actual LIF spike events. The earlier plot showed
threshold-crossing+reset working but with tiny amplitude. Investigated
whether the bias space contains regimes that reach larger Vb.

Bias sweep over (VG1, VG2, Vd) — 6 × 4 × 3 = 72 points covering:
  VG1 ∈ {0.2, 0.4, 0.6, 0.8, 1.0, 1.2}
  VG2 ∈ {-0.2, 0.0, 0.2, 0.4}
  Vd  ∈ {1.0, 1.5, 2.0}

**Maximum Vb_eq found = 0.212 V** at (VG1=0.4, VG2=0.4, Vd=2.0).
Across the entire 72-point sweep, Vb_eq never exceeded 0.21 V.

This **confirms what we documented in Phase A**: the parasitic NPN
needs Vb > ~0.6 V to fire (silicon p-n junction Vf), but Sebas's
un-refit M1 card hard-caps Vb_eq at ~0.21 V regardless of bias.

**Implications for LIF / meta-plasticity demos:**

✅ What we CAN demo with current calibration:
- Leaky integration (Vb climbs gradually, decays between pulses)
- Threshold-and-reset mechanism (with low threshold ~0.05-0.1 V)
- Capacitive body dynamics at fast pulses (body cap holds Vb low)
- Multi-cell parallel arrays (B.4 topology)
- Reservoir-style learning (B.5 next)

❌ What we CANNOT demo without M1 refit:
- Volt-scale neuronal spikes (requires Vb past 0.6 V to fire NPN)
- Realistic STDP (synaptic plasticity rules tied to NPN-fire events)
- True \"meta-plasticity\" (synapse↔neuron switching via the NPN
  transition) — the NPN never lights at any bias

**Path forward for the NRF pitch:**

Option A — \"mocked\" meta-plasticity using threshold-tuning. We
demonstrate the SAME BEHAVIOR as a real meta-plastic cell by varying
spike threshold and reset levels under control voltages, even though
the underlying physics (NPN firing) doesn't fire. Honest demo of the
DIFFERENTIABLE FRAMEWORK; not honest about the physics.

Option B — request M1 refit from Sebas. He's already iterating his
deck (added pdiode 5/2). With one more round we could get a card
where Vb naturally reaches volts at high VG1, then everything works.

Option C — synthesise a \"silicon-equivalent\" M1 by direct gradient
fit of vth0/u0/etab to make the model produce Vb past 0.6 V at some
realistic bias. This is engineering on our end.

**Recommendation:** Option B is cleanest scientifically. Option A is
fastest for the pitch demo. Option C is risky (could overfit). User
decision needed.

Phase A status: closed, but the card-limit is now QUANTIFIED at
Vb_eq ≤ 0.21 V — a hard constraint on what dynamic phenomena we can
demonstrate.

Next iteration: either await user decision on A/B/C, or proceed with
B.5 benchmarks (which use static-mode reservoir computing and don't
depend on volt-scale Vb).
