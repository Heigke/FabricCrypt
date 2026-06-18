# Oracle query O102 — What did we miss after 14 NULL identity attacks?

**Date**: 2026-05-30
**Project**: FEEL — identity-as-stake sub-programme
**Hardware**: 2× HP Z2 Mini G1a, AMD Ryzen AI Max+ PRO 395 / Radeon 8060S
(gfx1151), 128 GB LPDDR5X, ROCm 7.0, kernel 6.14. Hosts: ikaros, daedalus.

**Status**: 14 attacks attempted, ALL returned NULL or were falsified as
confounds (most recently A1+A3 z=5.74 → falsified via `hash("ikaros")`
spatial seed leak). Detailed postmortem attached.

## Question to all 4 oracles (please answer each in order, do not skip)

**Q1 — Architectural assumption hunt.** We tested 14 methods. ALL share the
assumption that "substrate is READ as a signal by the model" (input feature,
per-neuron leak, weight mod, dynamical coefficient). What if identity binding
instead requires (a) substrate AS A CONSTRAINT (cannot compute X without it),
(b) substrate AS A REWARD SOURCE (model selected for substrate-survival),
(c) substrate AS TEMPORAL CONTINUITY (cumulative history matters, not
snapshot)? Which untested architectural pattern is most likely to break
through on our specific hardware in ≤ 1 week wall time?

**Q2 — Active wear-as-training.** We never tested a methodology where the
model itself ACTIVELY DEGRADES the substrate during training (writes to
specific TLB sets, fills cache lines to induce localized self-heat, modifies
thermal state via targeted compute hot-spots). Could "wear-as-training"
create irreversible per-device adaptation? Cite any work in this space
(Karnik, Mintarno, Vaisband 2024+ for aging-aware), and assess feasibility
on Strix Halo in ≤ 2 weeks.

**Q3 — Cryptographic angle.** AMD SEV-SNP gives us a Versioned Chip
Endorsement Key (VCEK) signed by an AMD root CA, derived from a per-die
secret. Has anyone used TPM EK, SEV VCEK, or SGX EK as substrate signal
(not just as a wrap key) for a learnable model? Why not? Is there a
fundamental obstacle, or just unexplored? We can run sevctl on both ikaros
and daedalus.

**Q4 — Compiler / instruction-set angle.** Different x86 chips trigger
different compiler paths (BMI2, AVX-512 variants, AMX, AVX10). Can a
compiler-aware model be made to bind to specific instruction-set
capabilities by mid-training profile-guided regeneration? Cite any work
(MLGO, BOLT, BOLT-NN). Is the "ISA fingerprint" route worth pursuing?

**Q5 — Attack category enumeration.** In our 14-attack catalogue, which
ATTACK CATEGORY (not specific method) did we entirely fail to enumerate?
Candidates include: covert-channel as substrate, energy-budget as constraint,
fault-injection as training signal, joint-multichannel SCA, organoid /
biological substrate, photonic substrate. Pick the ONE category we missed
that is most likely to yield a constitutive binding on commodity x86/ARM
within a 100-wall-hour budget.

**Q6 — SCA closure.** Side-channel-attack literature reaches >99% device-ID
via power + EM + timing fingerprints (Picek et al. SoK 2023, EM-X-DL arxiv
2011.06139, GATEBLEED arxiv 2507.17033). Why hasn't anyone closed the loop
and made a model whose computation *depends* on the SCA-fingerprint of
itself? Is there a fundamental obstacle or just unexplored? Suggest a
concrete experimental design.

**Q7 — Approximate-compute software emulation.** Photonic/memristor/spintronic
substrates succeed because compute IS analog physics. Can software EMULATE
that on a conventional CPU by deliberately running in approximate-compute
mode (FP16/FP8 with stochastic rounding, deliberate noise injection in
matmul, aggressive undervolting via MSR)? Cite Lyu et al. approximate
computing surveys. Would per-device undervolting margin (Papadimitriou HPCA
2017, Bacha & Teodorescu ISCA 2014) be sufficient to make the SAME software
substrate become a per-die noise source?

**Q8 — Theorem status.** The "perfect calculator" / "abstraction tax" thesis
(any IEEE-754 op on chip A produces same bits as chip B by contract) — is
this *formally proven* somewhere, or is it empirical consensus that no one
has tried hard enough? Cite the formal proof if one exists, or refute it.

**Q9 — Definitive single experiment.** Different from prior `z > 2σ on
transplant` gates: design ONE experiment whose positive outcome is
UNFALSIFIABLE by shuffle, SW-matched, or spatial-seed confounds. The gate
must be *constructive* (model M produces output Y only when running on
device D, ⊥ elsewhere), not just statistical separation. We will run it.

**Q10 — 100-wall-hour plan.** If you had 100 wall-clock hours on
ikaros + daedalus (same HP Z2 G1a twins, no new HW purchases, all-night
unattended runs OK), and you HAD to produce a publishable constitutive-
binding result, what would YOU run? Be specific: hour-by-hour breakdown,
scripts to write, gates to pre-register, fallback plan if primary fails.

## Bundled context (see attached files)

- `IDENTITY_POSTMORTEM_2026-05-30.md` — diagnoses of all 14 attacks
- `IDENTITY_NULL_PAPER_2026-05-30.md` — 9-attack consolidated paper draft
- `IDENTITY_CONSTITUTIVE_2026-05-30.md` — 5-regime constitutive sweep
- `IDENTITY_LITERATURE_HUNT_2026-05-30.md` — top-3 papers + portability
- `IDENTITY_BROADER_MECHANISMS_2026-05-30.md` — 34 mechanisms B1-B34
- `IDENTITY_MISSED_MECHANISMS_2026-05-30.md` — 17 missed M1-M17

Please return a synthesis section identifying the SINGLE highest-EV
method-class we should attempt next.
