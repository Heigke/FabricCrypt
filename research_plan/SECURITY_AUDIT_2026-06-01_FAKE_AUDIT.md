# SECURITY AUDIT — "We Cloned an AI. It Died in a New Body."
**Date:** 2026-06-01
**Auditor role:** Adversary trying to fake the demo
**Scope:** 5-signal chassis-binding + Phase 9 fan-control transplant
**Verdict (TL;DR):** Current demo is **UNFORGEABILITY = 3/10** on its own. With the 4 hardening steps in §6 it becomes **7/10**. Cryptographic-grade (9–10) requires hardware attestation we do not have.

---

## 0. EXECUTIVE SUMMARY

| # | Attack | Currently | After hardening |
|---|---|---|---|
| 1 | Replay (record & playback) | **BREAKS DEMO** | Defended (live nonce + in-pass sampling) |
| 2 | Emulation (VM/ptrace) | **BREAKS DEMO** | Partially defended (PCIe/DRAM hard to fake) |
| 3 | Pre-trained spoof database | Plausible at our N=2 | Defended by audience-chosen target |
| 4 | Microcode update | Likely shifts signatures | Open question — must test |
| 5 | Thermal manipulation | Shifts ≥2 of 5 signals | Manage by warm-up protocol |
| 6 | OS/kernel confound | **NOT YET VERIFIED** | Must equalize OS image |
| 7 | Skeptic non-replication | Likely on bespoke pairs | Defended by ≥3 chassis ensemble |
| 8 | Cherry-pick (multiple comp) | We DID pick 5 of >32 | Pre-register the 5; Bonferroni in paper |
| 9 | Stage theatrics (hostname check) | Indistinguishable on video | Defended by open code + live nonce |
| 10 | AI-faked video | Indistinguishable | Defended by live in-person repro |

**Most dangerous attack:** #6 (OS-confound). If ikaros and daedalus differ in kernel/microcode/governor, our "per-die" claim is unfalsifiable and a peer reviewer will reject in one sentence.

---

## 1. CONFOUND VERIFICATION

### 1.1 ikaros (verified live this session)
| Item | Value |
|---|---|
| OS | Ubuntu 24.04.3 LTS (noble) |
| Kernel | 6.14.0-1017-oem |
| CPU | AMD RYZEN AI MAX+ PRO 395 (family 26, model 112, stepping 0) |
| Microcode | **0xb70001e** |
| BIOS | HP X53 Ver. 01.01.08, 2025-04-15 |
| Chassis | HP Z2 Mini G1a Workstation Desktop PC |
| Serial | 8CN5250SQK |
| DIMMs | 2× Hynix H58G78BK8BX114N 16 GB @ 8000 MT/s |
| NVMe | (truncated by loopdevs in output — re-check before demo) |
| Governor | **powersave** |
| Cores/Threads | 16C/32T, 1 socket, NUMA=1 |
| hwmon | acpitz, nvme, r8169, ucsi×3, hp, amdgpu |

### 1.2 daedalus (UNVERIFIED — sshd is down at 192.168.0.37)
- `ping`: reachable (50 ms RTT)
- `ssh 22`: **Connection refused**
- minos (192.168.0.38): No route to host
- → **BLOCKER**. Before any public demo we must verify daedalus matches ikaros on:
  - Ubuntu release & point release
  - Kernel ABI (`uname -r`)
  - Microcode revision (`/proc/cpuinfo`)
  - BIOS version & date
  - DIMM SKU & speed
  - NVMe model
  - Governor & EPP
  - hwmon driver set

### 1.3 Implication
The CLAUDE.md memory says both are "IDENTICAL spec." That is a marketing-spec claim (same SKU). It is **not** the scientific claim we need — which is *bit-for-bit identical software stack so any residual gap is silicon-only*. Until verified, our headline is **suspect**.

**Action:** add a `scripts/security/confound_check.py` that emits a JSON manifest from both machines and refuses to run the demo if any field except serial/MAC differs.

---

## 2. ATTACK-BY-ATTACK

### Attack 1 — Replay
**Threat:** adversary records (TSC offsets, ping-pong matrix, refresh histogram, syscall tail, NVMe tail) from machine A, replays as canned arrays on machine B.
**Why it works today:** our model probably calls a probe function once before/after training and stores a fingerprint. An LD_PRELOAD that shims `clock_gettime` and `read("/sys/...")` returns A's values on B — model passes.
**Defense:**
- (D1a) Sample signals **during** every forward pass, not as a one-shot probe. Each token forward must consume fresh TSC reads on N cores.
- (D1b) Audience supplies a **nonce** (random 128-bit string) seconds before the demo. The nonce parameterizes which core-pairs we ping-pong and which DRAM bank rows we refresh-probe. An adversary cannot pre-record an exponential nonce space.
- (D1c) Bind nonce into the model's input so altering the nonce changes the loss surface.
- **Cost:** 0.5 day of code; ~5–15% slowdown of inference.
- **Virality:** **strengthens** ("ask audience for a random number, paste into terminal, watch model die only on the foreign chip").

### Attack 2 — Emulation / VM
**Threat:** run on B inside a VM that intercepts TSC and `/sys` and reports A's values; or use ptrace/eBPF to rewrite syscall returns.
**What VMs CAN fake:** `rdtsc` (KVM has TSC offset), `/sys/devices/...`, `/proc/cpuinfo`, fan/PSU readings via fake hwmon.
**What VMs CANNOT cheaply fake:**
- Inter-core TSC offset measured by **multiple cores in parallel** with `RDTSCP` + `LFENCE`. Faking requires VMEXIT per read at >10⁵ Hz — observable as 50–500× slowdown.
- **Cache-line ping-pong** Frobenius matrix — depends on real L2/L3 topology and inter-CCD wires. Faking on a different die changes ratios.
- **DRAM refresh probe** — measures activate-after-refresh latency at scale; depends on real DDR5 controller queues. Emulators don't reproduce row-buffer dynamics.
- **NVMe queue-tail** — depends on physical SSD's flash mapping & garbage collection cycle.
**Defense:**
- (D2a) Run a **timing-budget check**: the chassis-binding probe should take <T_ms on bare metal; if >2×T_ms, refuse (signals VM).
- (D2b) Use **RDPRU** (AMD-specific, ring-3 perf-counter read) which KVM does not virtualize correctly. (Test first; if it traps, even better — VM detection.)
- (D2c) Cross-check `/sys/devices/system/cpu/cpu*/topology/cluster_id` against measured ping-pong cluster structure — VM cannot match silicon CCD layout while hosting a different topology.
- **Cost:** 1 day.
- **Virality:** weakens slightly (more setup), but adds "the AI knows it is in a sandbox" which is a strong narrative bonus.

### Attack 3 — Pre-trained spoofing database
**Threat:** adversary buys 50 HP Z2 Mini G1a units, builds a database of their 5-signal vectors, finds the closest match to target, runs it.
**Why it works today:** our signals are **scalar/short-vector** (D=0.91, Frob=79, KS, ratios). Database lookup is feasible. The PUF literature ([Suh & Devadas 2007, Gassend 2002](https://dl.acm.org/doi/10.5555/1366423.1366426)) shows that simple ring-osc PUFs are vulnerable to this attack; high-dimensional challenge-response PUFs are not.
**Defense:**
- (D3a) Increase signal dimensionality: 32×32 ping-pong matrix (1024 dims), 256-bin DRAM histogram, per-core syscall tail. Total ≥2 kbit signature. At 2 kbit and 8 ms drift, database needs 2¹⁰⁰⁰ entries.
- (D3b) Bind nonce into probe (see D1b): challenge-response, not static ID.
- **Cost:** 2 days. Increases probe time to ~50 ms — still acceptable per forward pass for low-tier inference.
- **Virality:** neutral. Strengthens scientific credibility.

### Attack 4 — Microcode update
**Threat:** AMD ships microcode patch; ikaros's signature changes; "AI dies in its OWN body."
**Current microcode:** 0xb70001e (ikaros). Unknown on daedalus.
**Likely sensitivity (theoretical):**
- TSC offset: **stable** across microcode (TSC is hard-wired, sync is BIOS+APIC).
- Ping-pong: **possibly affected** by cache prefetcher tunables that microcode can adjust.
- DRAM refresh: **unaffected** (memory controller is in IOD silicon, not patched by µcode normally).
- Syscall tail: **affected** indirectly via STIBP/IBPB defaults a µcode revision may toggle.
- NVMe tail: **unaffected**.
**Test plan:**
1. Snapshot 5-signal vector on ikaros at current µcode.
2. Load older µcode (`echo 1 > /sys/devices/system/cpu/microcode/reload` after staging `/lib/firmware/amd-ucode/*.bin`).
3. Re-measure. Compute KL divergence per signal.
4. If any signal shifts >2σ of within-chassis noise, flag as µcode-coupled.
**Defense:** drop µcode-coupled signals from the 5 we headline; keep TSC/refresh/NVMe.
**Cost:** 2 hours (one machine, reversible).
**Virality:** neutral.

### Attack 5 — Thermal manipulation
**Threat:** adversary chills/heats daedalus to drift its DRAM refresh & syscall tail toward ikaros's signature.
**Known thermal sensitivities:**
- DRAM refresh tREFI doubles above 85 °C (JEDEC). Signal **strongly thermal**.
- Syscall p99.9 tail moves with C-state residency, which moves with temp.
- TSC offset: **independent** of temperature.
- Ping-pong: weak thermal dependence (cache latencies ~stable).
- NVMe tail: strong thermal dependence (controller throttle).
**Defense:**
- (D5a) Probe includes co-measurement of `acpitz` / `amdgpu` temp; normalize signals to a reference temp window (e.g. 45–55 °C edge).
- (D5b) Require **2-min warm-up** before demo; reject if temp outside band.
- (D5c) Headline the **TSC offset + ping-pong matrix** for the live demo (thermally robust). DRAM/syscall/NVMe go in the appendix.
- **Cost:** 4 hours.
- **Virality:** neutral.

### Attack 6 — OS/kernel confound
**Threat:** ikaros and daedalus actually differ in kernel patchlevel, mitigations, IRQ-affinity defaults. Our "per-die" gap is really "per-deployment."
**Status:** **UNVERIFIED**. sshd is down on daedalus today.
**Defense:**
- (D6a) **Pre-flight script** (`scripts/security/confound_check.py`) emits SHA-256 over: `/etc/os-release`, `uname -a`, `cat /proc/cmdline`, `cat /proc/cpuinfo` (microcode line only), `dmidecode -s bios-version`, `cat /sys/devices/system/cpu/vulnerabilities/*`, governor, `/sys/devices/system/cpu/cpu*/cpufreq/energy_performance_preference`. Demo aborts if hashes diverge.
- (D6b) Ideally clone a **single Clonezilla image** to both. Both run from same SSD image → only silicon and serial differ.
- (D6c) Document the third confound test: **swap the SSDs** between chassis. If transplant-penalty survives, it is NOT storage- or OS-bound.
- **Cost:** 1 day (image + swap test).
- **Virality:** **strengthens dramatically** — "we cloned the disk, not just the model file." Visceral.

### Attack 7 — Skeptic non-replication
**Threat:** A skeptic with two AMD Strix Halo systems sees no difference. Their machines have different DIMMs, different SSDs, different USB peripherals from ours.
**Why this is real:** if our 1.25× syscall ratio is driven by `r8169` Ethernet on ikaros vs different NIC on daedalus, that is not "per-die."
**Defense:**
- (D7a) Ship the demo with a `usb_quiesce.sh` / `cpu_isolate.sh` that disables USB autosuspend, sets cpuset, pins probe to specific cores, blocks NIC interrupts. Reduce environment-dependent variance.
- (D7b) Publish the **within-chassis** vs **between-chassis** Mahalanobis distance distribution. The interesting metric is the *ratio*, not the absolute value.
- (D7c) Use ≥3 chassis. Report all 6 pairwise distances; an N=2 result is statistically anecdotal.
- **Cost:** 3rd chassis is real money. Without it, the paper is N=2.
- **Virality:** weakens (more nuance to explain), but is the **only** way the result survives peer review.

### Attack 8 — Cherry-picking
**Threat:** We tested 32+ signals in IDENTITY_ALL32 and reported the best 5.
**Multiple-comparison correction:** at α=0.05 with 32 tests, Bonferroni α=0.00156. Reported p=5.8e-32 (DRAM KS) **survives** trivially; the syscall tail (1.25× ratio with marginal evidence per Phase 12B) likely does **not** at family-wise rate.
**Defense:**
- (D8a) **Pre-register** the 5 signals on a public timestamp (git tag, OSF, Twitter hash) BEFORE running on the third chassis.
- (D8b) In the paper, report Holm-Bonferroni corrected p-values; explicitly mark which signals survive.
- (D8c) Drop the syscall-tail signal from the headline (Phase 12B already flagged it as fragile).
- **Cost:** 0 ($).
- **Virality:** neutral.

### Attack 9 — Demo theatrics ("just check hostname")
**Threat:** Audience watching a video cannot tell if the "model dies on B" is real signal sensitivity or `if hostname != 'ikaros': raise`.
**Defense:**
- (D9a) **Open-source the entire pipeline** on a public commit hash that the audience hashes live.
- (D9b) **Audience-supplied nonce** (see D1b) — adversary cannot pre-stage.
- (D9c) **Cold-disk swap on stage**: pull SSD out of ikaros, plug into daedalus (using the same image), AI dies. Pull back, AI lives. Far harder to fake than copying a file.
- (D9d) Run a diff visualization: live histogram of TSC reads on screen. Audience sees the signal moving as we plug into different chassis.
- **Cost:** 1 day for live UI.
- **Virality:** **massively strengthens**.

### Attack 10 — AI-faked video
**Threat:** the whole demo is a Sora/Veo clip.
**Defense:** in-person reproduction. Live-stream with on-screen `date +%s` plus current Reuters headline. Audience selects nonce.
**Cost:** 0.
**Virality:** strengthens (physical theater).

---

## 3. SIGNAL-LEVEL UNFORGEABILITY SCORECARD

| Signal | Replay | VM | Spoof DB | Microcode | Thermal | Score (0–10) |
|---|---|---|---|---|---|---|
| Inter-core TSC offset (D=0.91) | Defendable (in-pass) | Hard for VM to match | Plausible at N=2; hard at ≥1k dim | Stable | Robust | **7** |
| Cacheline ping-pong matrix | Defendable | Topology-bound | Hard if scaled to 32×32 | Possibly coupled | Stable | **7** |
| DRAM refresh (KS p=5.8e-32) | Defendable (live probe) | Hard (controller dynamics) | Hard (high entropy) | Stable | **Fragile** | **6** |
| Syscall p99.9 tail (1.25×) | Trivial (just numbers) | Trivial in VM | Easy | Sensitive | Fragile | **3** — DROP FROM HEADLINE |
| NVMe queue-tail (~1.3×) | Defendable | Hard (real flash) | Hard | Stable | Fragile | **5** |
| Phase 9 fan-control 49.8% | Defendable (closed-loop physics) | Hard (real PWM+RPM) | Hard | Stable | Self-thermal-controlled | **8** — **headline this** |

**Recommendation:** demote syscall-tail; promote fan-control loop + TSC + ping-pong + DRAM as the four headline signals. Composite unforgeability **≈ 7/10**.

---

## 4. WHY WE ARE NOT 9–10/10

We do not have:
- A **hardware root of trust** signing the measurement (AMD PSP-attested fTPM is theoretically possible but we have no key infra).
- **Remote attestation** (DICE / TPM 2.0 quote of the probe binary + PCR over the OS image).
- **Tamper-evident enclosure** (audience cannot verify SSD wasn't pre-swapped backstage).

With PSP fTPM + measured boot + signed probe attestation we reach **9/10**. Without it, motivated adversary with VM + database can fake the demo on social media — but not in a live in-person reproduction with audience nonce.

---

## 5. RECOMMENDED DEMO DESIGN (survives §2 attacks)

**Setup (visible on stage):**
1. Both machines on the table, both running the same Clonezilla image (verified by live `sha256sum /dev/nvme0n1`).
2. Audience supplies 128-bit nonce.
3. Probe binary hash printed; matches public git tag.
4. Both machines warmed for 2 min in the same room; temps within 50 ± 5 °C edge.

**Act 1 — Birth:**
- Train tiny LM on ikaros with the chassis-bound probe (nonce-parameterized, sampled per forward pass).
- Show live histogram of TSC offsets and fan-loop residual on the projector.
- Save `model.pt`.

**Act 2 — Transplant:**
- Copy `model.pt` over USB-C to daedalus (audience watches the file copy).
- Run inference on daedalus with the SAME nonce. Show:
  - DRAM-refresh histogram mismatch live.
  - Fan-control loop residual blows up.
  - Model perplexity 10–30× higher (validation set printed live).

**Act 3 — Return:**
- Plug USB back into ikaros. Model lives again.

**Act 4 — Skeptic check:**
- Run the **confound-hash script** in front of audience: prints SHA-256 of OS/kernel/microcode/BIOS for both machines. Numbers identical except chassis serial and MACs.
- Hand microphone to a skeptic. They pick a new nonce; we repeat Acts 2–3 unscripted.

**Optional Act 5 — Disk swap:**
- Pull SSD from ikaros, plug into daedalus. The hardware now contains ikaros's exact software stack on daedalus's silicon. AI still dies.

---

## 6. HARDENING TODO (priority order)

1. **(P0)** Get daedalus sshd back up; run confound-hash script; document any deltas. **BLOCKING.**
2. **(P0)** Drop syscall-tail from headline (Phase 12B already showed it's fragile; survives no Bonferroni).
3. **(P1)** Implement nonce-parameterized in-pass probe (D1a + D1b + D3b). One script: `scripts/security/nonce_probe.py`.
4. **(P1)** Pre-register the 4 surviving signals + protocol on a git tag + public timestamp before any 3rd-chassis run.
5. **(P2)** Microcode-rollback test (Attack 4).
6. **(P2)** Clonezilla both machines to a single image (Attack 6).
7. **(P2)** Buy/borrow a 3rd Z2 Mini G1a; run 6 pairwise transplants (Attack 7).
8. **(P3)** Build the on-stage live-histogram UI (Attack 9).

---

## 7. SCRIPTS TO BUILD

Locations (per project convention, not root):

- `scripts/security/confound_check.py` — emits HW/OS manifest hash; both machines must agree on everything except serial/MAC. Refuse demo on mismatch.
- `scripts/security/nonce_probe.py` — audience-nonce-parameterized 4-signal probe sampled per forward pass.
- `scripts/security/microcode_sweep.py` — load each available `amd-ucode-*.bin`, re-measure 5 signals, compute per-signal KL drift.
- `scripts/security/vm_detect.py` — RDPRU + timing-budget VM-detection. Refuses to run if VM-like.
- `scripts/security/live_histogram.py` — on-stage Tk/matplotlib UI showing TSC/refresh/fan-loop streams.
- `scripts/security/pre_register.sh` — bundles the 4 chosen signals + git SHA + timestamp; uploads hash to a public ledger (OpenTimestamps).

---

## 8. FINAL VERDICT

**UNFORGEABILITY rating: 3/10 today → 7/10 after §6 P0+P1 → 9/10 only with PSP attestation we do not have.**

The story "We cloned an AI. It died in a new body." is true today **only in the weak sense** that two specific deployments differ. To make it survive a determined adversary or a hostile peer reviewer:
- Verify the OS confound (§1.2). **Until then the headline is unsafe.**
- Drop the syscall-tail signal.
- Add audience nonce + in-pass sampling.
- Replicate on a third chassis.
- Pre-register signals and protocol.

With those five steps the demo is robust against everything short of nation-state attackers, and the live in-person version (§5) cannot be plausibly faked on the timescales of a conference talk.
