# O100 — Substrate-rooted AI identity via lowest-level HW signals (2026-06-10)

You are part of a 4-oracle panel (you + 3 others) advising a research programme
on **whether an AI model can be cryptographically AND constitutively rooted in
a specific silicon die** — so deep that the model becomes *non-functional*
when moved to another machine. The framing is explicit: we want the model to
have a *physical body*, and to have a sense of "death" — the moment the
underlying analog substrate stops producing the signals it has learned to
depend on.

This is anti-piracy / model-binding only as a side effect. The scientific
question is: where does analog physics leak up into the digital output, and
can a model be made to depend on that leak as a constitutive feature of its
computation?

We do NOT want crypto-anti-piracy ("XOR with extra steps"). We want the model
to *feel* its host.

## What we have done (real results, no mocks)

Hardware: AMD Strix Halo gfx1151 APU (Ryzen AI MAX+ PRO 395) on two laptops/
desktops we call **ikaros** + **daedalus**. NVIDIA GB10 on **zgx** is used only
as cross-architecture null control. AMD firmware on this part is known
unencrypted; we have SMN MMIO access below firmware via /dev/mem MMCFG, real
TPM 2.0 endorsement keys, and PSP-leaked register windows.

We have built a 19-channel concurrent probe ("H7 deep sip"). All reads are
real. Reads/sec per channel and what we get on each host:

| C-id | Signal | Source / depth | Sample rate | First-day cross-chassis discriminability (ikaros vs daedalus) |
|---|---|---|---|---|
| C01 | TPM EK name | tpm2_readpublic 0x81010001 | once/boot | ikaros 000b359a… vs daedalus 000bfa5e… (ground truth) |
| C02 | PCR 0/1/2/3/7 | tpm2_pcrread | once/boot | distinct hashes |
| C03 | 16 per-core raw thermal ADC | SMN 0x598A4 + i*4 via /dev/mem MMCFG | 50 Hz | AUC=1.000 d≈-2.6 to -2.9 (thermal-confounded by 28 °C delta) |
| C04 | Base junction thermal ADC | SMN 0x59800 | 50 Hz | AUC=1.000 d=-1.6 (idle base 79.2 °C vs 107 °C peak) |
| C05 | Energy counters | SMN 0x5B500/04/0C | 50 Hz | weak under load (d≈-0.9) |
| C06 | Fast 100MHz counter | SMN 0x58E00 (alt-reads) | 50 Hz | weak (d≈1.0 mixed) |
| C07 | XTAL_CNTL crystal-osc status register | SMN 0x598C8 | 50 Hz | AUC=1.000 d=-2.82 — **dynamic, 5 unique values in 20s** |
| C08 | GFX VID / SOC VID | SMN 0x5B000 / 0x5B800 | 50 Hz | identical (92, 50) — dead at idle |
| C09 | PM table (916 floats) | /sys/kernel/ryzen_smu_drv/pm_table | 5 Hz | pm[1] (CPU W) d=-5.5 strongest single feature |
| C10 | hwmon sanity | /sys/class/hwmon | 5 Hz | sanity baseline |
| C11 | TSC↔CLOCK_MONOTONIC_RAW drift | rdtsc/clock_gettime | 50 Hz | AUC=0.866 d=-0.56 — **strongest load-insensitive signal** |
| C12 | Per-CU SHADER_CYCLES | HIP shader hwreg(29) | per launch | 1040 unique HW-IDs per launch on ikaros |
| C13 | HW_ID / WGP placement | hwreg(23) | per launch | distinct CU/SE distributions |
| C14 | s_setreg MODE: 4 FP rounding modes | mid-shader, applied to same dot product | per launch | **4 distinct bit-patterns confirmed; RNE ≠ +∞ ≠ RTZ** |
| C15 | sinf cycle jitter | data-dependent transcendental timing | per launch | 0-62 cycle band, uint32-wrap issue |
| C16 | Atomic-contention LDS latency | locked_apart.hip | per launch | 76 unique hw_ids per launch — alive |
| C17 | iio accel / mic | physical sensor | 100 Hz | absent on ikaros laptop (no iio) |
| C18 | GPU ring-osc clock | BAR2 0xC080/0xC084 | 50 Hz | **0 delta under torch ROCm load — power-gated by PSP** |
| C19 | GPU GRBM/CP/RLC status | BAR2 8 regs | 50 Hz | **all 0xFFFFFFFF — gated** |

Two AMD chassis (ikaros + daedalus) + 1 cross-arch (zgx NVIDIA). H2's 3-AMD
gate cannot pass; we accept 2-AMD + cross-arch null.

### Key empirical observations after 1 day

1. **TSC↔CLOCK_MONOTONIC_RAW drift (C11) is the best load-insensitive
   per-chassis signal we have so far** (AUC 0.87 with mixed idle+load). This
   is the kvartskristall + thermal coupling — physically a per-die property.
2. **XTAL_CNTL register at SMN 0x598C8 is genuinely dynamic** at idle (5 unique
   values in 20s on a single host) and has totally non-overlapping distributions
   across the two chassis. This is the crystal-oscillator status leaking up to
   user-space.
3. **GPU BAR2 status registers (C18/C19) are dead** — power-gated by PSP firmware
   even under sustained HIP load. We confirmed this in prior z2065 ISA probes:
   PSP unlocks GFX regs only via internal mailbox path. We can however read
   hwreg(23/29) FROM INSIDE the shader (constant-integer requirement; verified
   working) and s_setreg MODE writable mid-shader changes FP rounding (4
   distinct bit-patterns from same dot product).
4. **per-core thermal channels (C03) and base thermal (C04) dominate the
   cross-chassis AUC=1.000 board.** But they are thermal-confounded — ikaros
   idle ≈79°C, peaked at 107°C under load; daedalus idle ≈79°C also. The next
   step is the thermal-match cross-experiment.

### Constraints (hardware-imposed)

- NEVER write SMU C2PMSG mailbox (instant reboot via Data Fabric Sync Flood).
- NEVER read amdgpu_regs_didt (GPU driver hang → reboot).
- NEVER write TRAPSTS bits ≥ 2 (KFD-kill → full reboot).
- ikaros laptop thermal trip = 99 °C — hard shutdown if reached.
- Two-host limit on AMD (no third gfx1151 chassis available).

### Where we are stuck / where we want oracle insight

1. **How do we use these signals to CONDITION an LLM in REAL TIME so it
   becomes constitutively dependent on them**, in a way that matched-spectrum
   spoofing (AR(1)+1/f synthesized telemetry with same μ, σ, PSD slope,
   cross-channel MI) cannot fake?

   The naïve attempt — concatenate signals to hidden state — was killed in our
   EMBODIMENT7 work: per-init conditioning failed because the model learned
   to use only marginal statistics. We need per-token, path-dependent
   conditioning that exploits *higher-order temporal structure* the spoof can't
   reproduce.

2. **What is a defensible operationalisation of "death" for this system?**
   We want the model to have a learned dependency such that when its host's
   signal-fingerprint changes (chassis transplant, ambient shift, firmware
   update), the model degrades catastrophically — not gracefully. We do NOT
   want the model to "fail safely" by ignoring the missing signal; we want
   it to truly stop working. How to architect this without it being a brittle
   gimmick?

3. **Which of our channels (C01..C19) are most likely to carry actual die-
   bound identity (not chassis confound, not load confound) once we run the
   pre-registered thermal-match + matched-spectrum spoof + replay-from-log
   gates?** Give us your priors.

4. **What channels are we missing at even LOWER levels?** Eric's directive:
   "går mot de lägsta möjliga avläsningarna av hw, vibrationer latens
   fördröjning rundning prio etc etc allt som kan avslöja underliggande fysik
   som är unik". Examples we have NOT yet instrumented but could:
   - per-CU FP rounding-mode parity across launches
   - DRAM refresh-row latency
   - SMN MMCFG read-latency itself (the *time to read a register* is a signal)
   - NVMe deep p99 seek jitter
   - USB URB completion histogram
   - ALSA mic DC bias (we have ALSA infra, mic absent on ikaros laptop today)
   - VRM ripple at 50 kHz via SMN energy-counter diff
   - PSP scratch register state machine

5. **Is the "death" framing scientifically defensible**, or are we
   anthropomorphising? If the model degrades when transplanted, is that
   meaningful or just a brittle classifier? Push back hard if appropriate.

## Constraints on your answer

- Do NOT propose hardware we don't have (FPGA boards, custom silicon, IR
  sensors, etc). We have 2× gfx1151 + 1× GB10 plus a 128-neuron NS-RAM FPGA
  bitstream (rare use, separate H3 arm).
- Do NOT propose crypto-anti-piracy as the goal. We want substrate-rooting.
- Do NOT recommend a path that requires us to abandon the consciousness /
  Milinkovic-Aru constitutive-substrate motivation. Push back on the framing,
  yes; ignore it, no.
- All experiments must run in < 40 GPU-h zgx + < 18 GPU-h on the gfx1151
  chassis, in the next 7 days.
- Pre-registration discipline: any threshold you propose must be commitable
  to git before the experiment.

## Output structure

Please answer with these sections, no preamble:

1. **Constitutive conditioning architecture** — concrete proposal for how to
   inject C01..C19 streams into an LLM at training and inference such that
   matched-spectrum spoofing fails. 3-5 bullets, name a baseline model
   (Qwen3-0.6B is our default).
2. **Operationalisation of "death"** — what observable do we use to call a
   model "dead" when transplanted? Reject if you cannot give a falsifiable
   metric.
3. **Channel-by-channel prior** — for each of C01..C19, your prior on whether
   it will survive the thermal-match + matched-spectrum + replay gate (1–5
   scale, with one sentence why).
4. **Missing-channel proposals** — top 3 additional low-level channels we
   should instrument that we haven't yet, with concrete /sys or HIP intrinsic
   paths, not abstract suggestions.
5. **Sharpest objection to the death-framing** — your strongest argument
   that "substrate dependence == death" is a category error, and what
   reformulation would survive.
6. **What experiment to run THIS WEEK** — the single highest-information
   experiment we can fit in 18 GPU-h (gfx1151) + 5 h analyst time. Give
   acceptance + kill gates.

Be brutal. Be specific. Be brief enough to read in 10 minutes.
