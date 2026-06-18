# Embodiment advantage literature — 2024-2026 sweep

Web search performed 2026-05-31 in support of O107.

## Thermal-aware scheduling with RL on chips (Q4, Q10)

- **THERMOS** (Bhatt et al., ACM TECS Sep 2025, arxiv 2508.10691) —
  Thermally-aware multi-objective scheduling of AI workloads on
  heterogeneous multi-chiplet PIM architectures. Uses multi-objective RL
  for Pareto-optimal time/energy. Per-chip thermal model is part of the
  policy input.
- **Thermal-Aware Scheduling for Deep Learning on Mobile NPUs**
  (Tan et al., IEEE TMC 2024, doi 10.1109/TMC.2024.3379501) — Balances
  inference time and accuracy on mobile NPUs using heuristic + DRL
  scheduling, explicitly to prevent overheating. Closed-loop self-modeling
  with per-device thermal profile.
- **ReLeTA** (Hossain et al., arxiv 1912.00189) — RL-based thermal-aware
  task allocation on multicore. Foundational reference.
- **RL-driven task migration for 3D NoC** (Sci. Reports 2025,
  10.1038/s41598-025-96335-6) — RL learns chip-specific migration policy.
- **POD-TAS / ReLeTA** — standard baselines for thermal-aware task
  allocation; embedded in MORL pipelines.

**Reading**: Thermal-aware RL is established (5+ years), but typically
trains-and-evaluates on the same chip / digital-twin without per-chassis
cross-eval. Our C3 transplant arm is the missing methodological piece.

## Self-substrate prediction / thermal modeling (Q2)

- **Thermal Neural Networks** (Kirchgässner et al., arxiv 2103.16323) —
  Lumped-parameter thermal modeling with state-space ML. Predict
  hidden-node temps from sparse sensors. Per-device fitting standard.
- **Deep Learning Thermal Prediction for SSDs** (STMJournals 2026,
  article 245369) — Power dissipation + ambient + frequency + load
  current as features.
- **Physics-informed NN for thermal modeling, transferable** (Sci.Dir.
  S2214860425004257) — Specifically about *transferability* across paths
  and parameters. Embodied/per-system fitting normal practice; transfer
  is the open question (matches our gotcha).
- **Learning-based thermal estimation in multicore** (USPTO patent
  11334398) — NN predicts core temp from utilisation+fan+coolant.

**Reading**: Self-prediction is solved at the algorithm level; the
*embodiment* claim only adds value if the generic+more-data baseline
loses. No 2024-2026 paper found that runs that exact control.

## Per-device / per-chip personalised adapters (Q6)

- **MobiLoRA** (ACL 2025, aclanthology.org/2025.acl-long.1140) —
  Accelerates LoRA-based LLM inference on mobile devices via
  context-aware KV cache. 57.6% acceleration; not accuracy gains from
  per-chip hardware modeling.
- **EdgeLoRA** (MobiSys 2025, doi 10.1145/3711875.3729141, arxiv
  2507.01438) — Multi-tenant LoRA serving on edge devices. Personalisation
  via tenant/user, not chip-physics.
- **Hollowed Net** (arxiv 2411.01179) — On-device personalisation of
  text-to-image diffusion. User-personalisation, not hardware-personalised.
- **PLoRA** (arxiv 2403.06208) — Personalised LoRA for human-centred text.
  Personalisation is per-user.

**Reading**: No 2024-2026 paper found showing per-GPU LoRA that improves
*accuracy* by modeling hardware quirks of commodity GPUs. This confirms
our H5 null result is the field expectation. Cited "Substrate-Aware
Fine-Tuning (Zhang et al., 2024)" from DeepSeek could not be verified
in this search.

## Edge AI battery / energy-aware survival (Q4)

- **Energy-Aware Dynamic Neural Inference** (arxiv 2411.02471) — Regulates
  inference energy based on availability + workload + input complexity.
- **EADTrain** (Sci.Dir. S1047320325001968, 2025) — Energy-aware dynamic
  training of DNNs for sustainable AI.
- **Energy-Aware DL on Resource-Constrained Hardware** (arxiv 2505.12523) —
  Survey + framework. Per-device adaptation discussed.
- **HarvSched** — RL-based exit policy scheduler for energy-harvesting
  devices; learns chip-and-environment-specific policy.
- **PolyThrottle** (arxiv 2310.19991) — Energy-efficient inference on
  edge; tunes per-device hyperparameters.
- **An Energy-Aware Generative AI Edge Inference Framework** (MDPI
  Electronics 14(20):4086, 2025) — Reduces energy 31% and latency 27%
  vs baselines on edge.

**Reading**: Energy-aware inference is an active field. The benchmark
practice is single-chip self-eval; cross-chip transplant evaluation
(our C3-style) is rare. This is the gap our experiment design fills.

## Embodied AI advantage benchmarks (Q1, Q8)

- **PersONAL** (arxiv 2509.19843) — First benchmark for personalised
  embodied agents (navigation, object grounding); 2000+ episodes,
  30+ photorealistic homes. Personalisation per-user, not per-chip.
- **EmbodiedBench** (2025) — Multi-modal LLM evaluation for embodied
  agents. Vision-language-action standard benchmark.
- **VLA on 100g device** (deepsense.ai blog, 2025-2026) — Embodied AI on
  ultra-low-power; demonstrates feasibility but no claim of substrate-
  advantage.
- **EmboCoach-Bench** (arxiv 2601.21570) — Benchmarks AI agents on
  developing embodied robots.

**Reading**: "Embodied" in this literature means physical robot
embodiment, not "model bound to chassis it runs on". No body of work
matching our chassi-binding-as-embodiment frame exists; we are in
relatively novel methodological territory but with weak prior support.

## Synthesis

The literature endorses **thermal-aware survival scheduling** as the
single most-defensible "body-as-task" demonstration. It endorses
**self-prediction** and **anomaly detection** as standard practice
without specifically testing the embodiment-vs-generic-with-more-data
control. No 2024-2026 paper shows per-chassis LoRA accuracy gains on
commodity GPUs (confirming our H5 null is the expected outcome). PUF
literature is mature; "authentication-as-computation" is a rebrand.

The cleanest white-space: **cross-chassis transplant evaluation of
thermal/survival control policies on twin commodity APUs**, which is
exactly the gap C3 fills (sim) and the killer experiment from O107
would fill (real closed-loop on hardware).
