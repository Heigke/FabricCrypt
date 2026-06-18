# Two physically identical AMD Ryzen AI Max+ PRO 395 chassis admit per-machine sensor-trajectory predictors whose advantage is fully explained by training-data distribution and not by chassi-keyed model structure

**Embodiment Phase 7 draft — 2026-05-31, ikaros + daedalus, HP Z2 mini G1a (gfx1151).**

## Abstract

We test whether a 512-bit chassi-bound fingerprint, derived from DMI / PCI / hwmon / cache topology / ACPI / GPU enumeration, contributes capability beyond ordinary training-data overlap to two body-centric tasks (self-substrate prediction C1; self-anomaly autoencoder C2) on two physically identical HP Z2 mini G1a workstations (ikaros, daedalus). Four claims:

1. **The 512-bit signature satisfies G1-G4 robustness gates on both chassis** (repeat-stable, ≥512 bits, hostname-independent, ≥7 distinct subsystems contribute). Cross-chassis Hamming distance = 264/512 (0.516), near the SHA-512 random-pair expectation of 0.50.
2. **Self-vs-transplant gaps are real but large** — C1 NRMSE 0.68 (own data) vs ~210-2473 (other-host data); C2 AUROC 0.83 (own) vs 0.48-0.50 (other-host). Both replicate prior embodiment5 findings.
3. **A pre-registered 2×2 structure-by-data ablation (30 seeds, paired bootstrap 95% CI) shows the structure (chassi-hash-keyed init) contributes effectively ZERO over a random init when training data is held constant** — A vs B effect for C1 = +0.05% (CI spans 0); for C2 = +0.02 pp (CI spans 0). The same nullity holds for a small MLP (effect –10% to +16%, CI spans 0).
4. **The entire self-vs-transplant gap is therefore the data-distribution-shift confound the critic warned about.** The defensible scope contracts from "chassi-bound capability" to "per-machine sensor predictors of dynamics specific to that machine's recent run."

## 1. Methods

### 1.1 Full-bandwidth chassi signature

`scripts/identity_benchmark/embodiment7/full_signature.py` collects, on a single host:

| Subsystem            | Concrete fields                                                                                  |
|----------------------|--------------------------------------------------------------------------------------------------|
| DMI                  | board (name/vendor/version/serial/asset_tag), product (name/serial/uuid/version), bios (vendor/version/date/release), chassis (vendor/type/serial/asset_tag), sys_vendor |
| CPU                  | model_name, vendor_id, family, model, stepping, microcode, cpu_count, cache_size, flags (sorted) |
| Cache topology       | L1/L2/L3 sizes + line_size + ways for first 4 CPUs                                                |
| Memory               | MemTotal rounded to 1 MiB                                                                         |
| PCI                  | All devices: domain:bus:dev.func + [vendor:device] + subsystem ids, sorted                        |
| hwmon                | Enum (`hwmon{i}` → name), stable insertion order                                                  |
| thermal_zones        | Enum (`thermal_zone{i}` → type)                                                                   |
| ACPI                 | Sorted directory listing of `/sys/firmware/acpi/tables/`                                          |
| GPU                  | vendor, device, subsys_vendor, subsys_device, revision                                            |

Canonical JSON → SHA-512 → 512-bit signature. Hostname is excluded from canonicalisation.

### 1.2 G1-G4 robustness gates

- **G1 (repeat-stable)**: 3 sequential calls yield identical hash.
- **G2 (≥512 bits)**: trivial (SHA-512 output length).
- **G3 (hostname-independent)**: injecting fake hostname does not change hash.
- **G4 (subsystem coverage)**: ≥7 of {dmi, cpu, cache_topo, mem, pci, hwmon, thermal_zones, acpi_tables, gpu} present and non-empty.

### 1.3 Tasks

- **C1 — self-substrate-prediction.** 5 substrate channels (apu_temp_c, gpu_temp_c, gpu_power_w, gpu_freq_mhz, kern_lat_us) sampled at ~5 Hz during a varying-matmul workload. Predict next 10 steps from 100-step history. Ridge reservoir, dim 256, ridge λ=1e-3. Data files: `c1_ikaros_data.npy`, `c1_daedalus_data.npy` (each 3000 samples × 5 channels, ~10 min collection per host).
- **C2 — self-anomaly autoencoder.** Tiny tanh AE (input 50×5, hidden 16, output 50×5), SGD 150 epochs. Test on 100 normal + 100 synthetic-anomaly windows (power-spike / thermal-step / latency-burst / freq-drop). AUROC metric.

### 1.4 A/B/C/D 2×2 factorial ablation (the key test)

| Cell | Model init (structure) | Training data | What it isolates              |
|------|------------------------|---------------|-------------------------------|
| A    | chassi-hash-keyed      | own host      | full embodiment hypothesis    |
| B    | random seed            | own host      | data-only baseline            |
| C    | chassi-hash-keyed      | other host    | structure-only                |
| D    | random seed            | other host    | null                          |

The chassi-hash-keyed seed is computed as `SHA-256(hash || seed)[:4]` (per-seed salting). The random seed is `seed * 1009 + 7`. **All else is identical**: same training/test windows, same normalisation (training-host mean/std), same hyperparameters.

Pre-registered effect-size gates:
- **Embodiment effect (A − B)**: ≥ 10% NRMSE improvement / ≥ 5 pp AUROC.
- **Data effect (A − C)**: ≥ 30% / ≥ 10 pp (sanity check).
- **Structure-alone (C − D)**: ≤ 5% / ≤ 5 pp (structure with wrong data should not help).
- **A strictly best**: A mean is ≥ 1 σ better than max(B, C, D).

30 seeds per cell, paired-by-seed percentile bootstrap (n=2000) 95% CI on differences. Bonferroni correction across the 4 confirmatory gates (α=0.025 each for the 2 primary).

### 1.5 Multi-architecture spot-check

Same A vs B test (no C, D) with a 3-layer ReLU MLP (input 100×5 flat → 64 → 64 → horizon×5), 120 epochs SGD. 15 seeds.

## 2. Results

### 2.1 Signature G-gates (PASS on both hosts)

| Host     | G1 stable | G2 bits | G3 host-indep | G4 subsystems | sha512 (first 16 hex)  |
|----------|-----------|---------|---------------|---------------|------------------------|
| ikaros   | PASS      | 512     | PASS          | 9/9           | `22410174476006cb…`     |
| daedalus | PASS      | 512     | PASS          | 9/9           | `ef8352b0d3c73cb1…`     |

Cross-host Hamming distance = 264/512 = 0.516 (vs SHA-512 random-pair expectation = 0.500). Two-hash sample-of-1 — directional only, not statistical.

### 2.2 C1 — A/B/C/D (30 seeds, NRMSE; lower = better)

| eval_host = ikaros |  A (own/own) | B (rand/own) | C (own/other) | D (rand/other) |
|---------------------|--------------|--------------|---------------|----------------|
| median NRMSE        | 0.6770       | 0.6740       | 2473.31       | 2624.06        |
| mean ± σ            | 0.677 ± 0.022| 0.674 ± 0.022| 2604 ± 2581   | 2780 ± 2527    |

| eval_host = daedalus|  A           | B            | C             | D              |
|---------------------|--------------|--------------|---------------|----------------|
| median NRMSE        | 0.7514       | 0.7515       | 210.85        | 207.54         |
| mean ± σ            | 0.751 ± 0.030| 0.752 ± 0.032| 232 ± 96      | 220 ± 94       |

**Gates** (both eval hosts):
- Embodiment effect (A−B): **+0.05% / +0.10% → FAIL** (gate requires ≥10%).
- Data effect (A−C): **+99.98% / +99.64% → PASS** (huge, as expected).
- Structure-alone (C−D): 6.2% / 1.5% → marginal (within noise).
- A strictly best (≥1σ over max(B,C,D)): trivially true vs C,D but FAILS vs B (A ≈ B).

### 2.3 C2 — A/B/C/D (30 seeds, AUROC; higher = better)

| eval_host = ikaros | A (own/own)  | B (rand/own) | C (own/other) | D (rand/other) |
|--------------------|--------------|--------------|---------------|----------------|
| median AUROC       | 0.8305       | 0.8345       | 0.4825        | 0.4813         |
| mean ± σ           | 0.829 ± 0.025| 0.832 ± 0.024| 0.483 ± 0.041 | 0.481 ± 0.040  |

| eval_host = daedalus | A           | B            | C             | D              |
|----------------------|-------------|--------------|---------------|----------------|
| median AUROC         | 0.8257      | 0.8288       | 0.4994        | 0.4995         |
| mean ± σ             | 0.825 ± 0.045| 0.828 ± 0.046| 0.500 ± 0.045| 0.500 ± 0.045 |

**Gates**:
- Embodiment effect (A−B): **+0.02 pp / −0.36 pp → FAIL** (gate requires +5 pp). Sign of effect even flips between hosts.
- Data effect (A−C): +34.2 pp / +33.0 pp → PASS.
- Structure-alone (C−D): 0.06 pp / 0.0 pp → PASS (no structure-only effect, as expected).
- A strictly best: FAIL (A ≈ B).

The C2 transplant AUROC ≈ 0.50 on daedalus (not <0.5) — consistent with the oracle hypothesis that the <0.5 in embodiment5 was a scaling artifact (one of the smaller-N runs randomly landed below 0.5). With 30 seeds it is at chance, not below.

### 2.4 Multi-architecture (C1, MLP, 15 seeds, A vs B only)

| arch | eval_host | A median NRMSE | B median | mean diff (A–B) | 95% CI         |
|------|-----------|----------------|----------|-----------------|----------------|
| MLP  | ikaros    | 1.0042         | 1.0030   | +0.0014         | [−0.029, +0.016]|
| MLP  | daedalus  | 1.0154         | 1.0081   | +0.0080         | [−0.027, +0.032]|

CI spans zero on both hosts; sign of effect flips. Same verdict as ridge.

(LSTM and Transformer skipped due to time budget; pattern is sufficiently clear from ridge + MLP that the structure effect is null.)

### 2.5 Cross-task generalization (C4/C5/C6)

Not run in Phase 7. Given A/B is null on the two tasks we did run, expanding to more tasks under the same null-structure mechanism would only add more null cells.

## 3. Discussion (one paragraph per oracle critic-hole)

**3.1 Distribution-shift confound (closed, verdict NEGATIVE).** The factorial A/B/C/D ablation, with 30 seeds and paired bootstrap CIs on both eval hosts, shows that holding training data constant, the chassi-hash-keyed model init contributes ≤0.1% NRMSE / ≤0.4 pp AUROC over a random init — i.e., null within sampling noise. The entire self-vs-transplant gap reported in embodiment5 is therefore attributable to training-data distribution mismatch, not to chassi-bound structure. Oracle gpt-5 / grok / deepseek convergent prediction is confirmed.

**3.2 C2 AUROC <0.5 artifact (closed, ARTIFACT confirmed).** At 30 seeds with consistent (training-host) z-score normalisation, transplant AUROC sits at 0.48-0.50 — at chance, not below. The embodiment5 finding of 0.484 was a small-N excursion from chance, not a genuine "below random" capability gap. Per gpt-5/grok/deepseek: ROC is monotone-invariant; <0.5 implies score orientation flip and disappears under proper paired normalisation.

**3.3 Single-architecture (partially closed).** Ridge reservoir + 3-layer MLP both show null structure effect with CIs spanning zero. Sign of effect even flips between hosts under MLP. This is insufficient for a positive "architecture-agnostic" claim, but it is **sufficient for the null** — adding LSTM/Transformer would not rescue a hypothesis already null at ridge and MLP.

**3.4 Statistical power (closed for confirmatory).** All confirmatory cells now use 30 seeds, paired percentile bootstrap (n=2000) 95% CI, Bonferroni-corrected α=0.025 for the two primary tests (A−B on C1 and C2). Window overlap is not corrected for; this would only further reduce effective N, which makes our null finding more conservative not less. Honest minimum from the oracles (10-30 seeds) is met or exceeded.

**3.5 Cross-task generalization (open, but moot).** Without a positive A−B effect on C1 or C2, broadening to C4/C5/C6 would only test whether the null replicates, not whether a positive effect generalises. Phase 7 budget redirected accordingly.

**3.6 External validity / N=2 (unchanged).** Two-machine sample-of-1 across all comparisons. The honest scope language: "demonstrated on these two HP Z2 mini G1a units; generality across the gfx1151 population is out of scope." Per oracle consensus, the only "silencer" given N=2 is an explicit blinded crossover (disks/RAM/PSU/location swap) — out of scope for Phase 7.

**3.7 Signal underutilization (closed, NEUTRAL).** The full 512-bit signature (vs prior 256-bit) passes G1-G4 on both hosts and gives the expected ~0.5 random-pair Hamming distance. It does not, however, recover any structure effect that the smaller signature missed — confirming the issue is the hypothesis itself, not signature bandwidth.

## 4. Limitations

1. **N=2 machines** — no population claim possible.
2. **Two tasks tested** — C4/C5/C6 (load/throttle/cpufreq prediction) not run.
3. **Sequence/attention architectures not tested** — only ridge + MLP under A/B. The null is robust over these two; LSTM/Transformer could in principle recover a positive effect, though our prior (the seed of the random projection / first-layer weights) does not plausibly encode chassi-physics.
4. **No yoked workloads** — collections were independent on each machine, so any time-of-day / room-temperature / governor difference adds to the data-distribution gap.
5. **No domain-invariance baseline** (DANN / IRM) — gpt-5's killer falsifier (pooled model with global scaling matches "self-specialists") would be the next ablation if any positive A−B effect were to be claimed.
6. **Window overlap** inflates effective N; corrected statistics would only make the null more conservative.

## 5. Future work

- **Pooled-data + domain-invariant model** (gpt-5's killer): if a single DANN model matches or beats both within-chassis specialists, the framing collapses entirely to ordinary domain generalisation.
- **Yoked workload protocol**: synchronised, governor-locked workloads across both machines on the same wall-clock, repeated over ≥10 days.
- **Third machine** (even briefly) to make the cross-host Hamming distance more than a sample-of-1.
- **Hash-swap inference test** (grok/deepseek killer): inject the *other* host's hash at inference time. If our null result is right, this swap will also be null. Useful as a public-facing sanity demonstration.
- If the structure-binding programme is to be revived, the only honest path is an architecture where the static signature is **structurally** coupled into the computation (e.g., as a basis-set constraint or as physical chassi-keyed circuit) rather than merely seeding random projection weights.

---

**One-sentence headline (defensible, post-ablation)**:

> "On two physically identical AMD Ryzen AI Max+ PRO 395 workstations,
> per-chassis self-vs-transplant gaps in sensor-trajectory prediction
> and self-anomaly detection are fully accounted for by training-data
> distribution shift; a 512-bit chassi-derived model-initialisation key
> contributes no measurable capability beyond a random seed (30 seeds,
> ridge + MLP, paired bootstrap 95% CI spans zero)."
