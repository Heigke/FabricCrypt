# Phase 20 — Exotic Identity-Signal Brainstorm

**Goal**: signals strictly beyond Phase-12 (TSC/C2C/DRAM/syscall/NVMe) and
Phase-19 (BTB/TLB/cross-CCX/GPU-jitter/PCIe-AER/hwmon-spread/RAPL-prec/Jacobian).

Each entry: **why per-die · feasibility on Z2 mini · stability · attack-risk**.
Stars (\*) mark the top-5 implemented in `s10`-`s14`.

---

### A. Radio / EM / Power-Domain Emanations

1. **\*USB-C VBUS / port-power droop fingerprint** (`s10`)
   Per-die: PMIC/SoC VRM has individual feedback-loop trim. Probe via
   `/sys/class/hwmon/hwmon5/in0_input` while bursting CPU. The over/undershoot
   waveform is a per-board signature (capacitor ESR + VRM compensation).
   Feasibility: HIGH (sysfs, no root). Stability: medium (depends on charger).
   Attack-risk: low (requires local probe + load-step pattern).

2. **WiFi RSSI vector across known APs**
   Per-die: chip antenna pattern + PCB ground plane. `iwlist scan`.
   Feasibility: HIGH (if WiFi up). Stability: poor (geometry-bound).
   Attack-risk: medium — geo-clonable.

3. **Bluetooth radio TX power calibration table**
   Per-die: factory cal in NVS. `btmgmt info`, vendor-specific HCI.
   Feasibility: medium (needs HCI privilege). Stability: HIGH (NVS-stored).

4. **Audio-jack as EM antenna (cross-bus crosstalk)**
   Per-die: PCB routing. *Not applicable — Z2 mini has no jack.*

5. **Network-RX inter-packet timing for self-loopback**
   Per-die: PHY clock crystal pull. `ping -f localhost` jitter.
   Feasibility: HIGH. Stability: HIGH. Attack-risk: LOW.

---

### B. Persistent / Wear Signals

6. **NVMe SMART thermal history + Composite-vs-Sensor delta** (`s13`)
   Per-die: NAND wear + DRAM-less controller cache trim. We have no
   `smartctl`/`nvme`, but `/sys/class/hwmon/hwmon1/temp{1,2}_input` exposes
   *Composite + Sensor1* live; the delta and 1/f drift over a fixed warm-up
   workload is per-board.
   Feasibility: HIGH (sysfs). Stability: HIGH (multi-month). Risk: LOW.

7. **Power-on-hours, host_read_commands** (proxy for wear)
   Would need NVMe-admin; without `nvme-cli` we skip.

8. **DRAM retention failure pattern at extreme temperature**
   Per-die: row-hammer / retention. Requires reboot to single-user. Skip.

9. **Stable-vs-unstable bits in uninitialized memory (SRAM PUF)**
   Cache SRAM is initialized by HW; DRAM is initialized by FW. Hard. Skip.

10. **Performance-counter saturation/wrap history**
    Per-die: nothing per-die unique here usually. Skip.

---

### C. Hardware-level Analog

11. **\*Voltage droop under load-step (Ldi/dt response)** (`s10`)
    Per-die: VRM compensation network + bulk-cap ESR. Burst then idle while
    sampling RAPL energy LSB rate.
    Feasibility: HIGH. Stability: HIGH. Attack-risk: LOW.

12. **Inductor saturation @ specific switching freq**
    Probe via narrow-band CPU oscillation. Hard to drive without root.

13. **\*PCIe SerDes equalization coefficients via setpci** (`s11`)
    Per-die: PCIe PHY trains EQ taps per-link; coefficients are queryable in
    Lane Margining caps (Gen4). `setpci -s ... ECAP30.B` style probe.
    Feasibility: medium (root for setpci on extended caps). Stability: HIGH
    (set once at link training). Risk: LOW.

14. **\*DDR PHY training residual** (`s12`)
    Per-die: DDR5 PHY training writes per-bit timing/voltage. dmidecode -t 17
    + UMC sysfs gives manufacturer/serial/training-stats. Not a true PUF but
    per-board stable.
    Feasibility: medium (dmidecode needs root usually; on most distros user
    can read DMI cache). Stability: HIGH. Risk: medium (DIMM-cloneable).

15. **Memory channel mismatch (per-channel latency)**
    Per-die: UMC trim. Stride a buffer larger than L3 across channels.
    Feasibility: HIGH. Stability: HIGH. Risk: LOW.

---

### D. Algorithmic per-chip

16. **AES-NI / SHA-NI dispatch-latency variance under adversarial load**
    Per-die: pipeline gating. Phase-12 found deterministic; re-test with
    simultaneous AVX2 floods.

17. **\*Per-CU shader instruction-latency skew on gfx1151** (`s14`)
    Per-die: 40 CU yield variation → minor freq/voltage trim per CU. HIP kernel
    pins a wave to each CU id and times a fixed instruction loop.
    Feasibility: HIGH (rocm + HIP installed). Stability: medium.
    Risk: LOW. **Most novel signal in this batch.**

18. **RDNA3.5 wavefront issue-order shuffle entropy**
    Per-die: scheduler trim. Hard to disentangle from runtime noise.

19. **Cache flush time variance under set-associative collisions**
    Per-die: replacement-policy random state at boot. Boot-bound.

20. **DMA controller per-channel latency (IOMMU off vs on)**
    Per-die: minor. Need device privileges.

21. **L3 victim cache row hit/miss rate at boundary fills**
    Per-die: tag-array variation. Hard to isolate from microcode noise.

---

### E. Topology / Configuration Unique

22. **DMI structure ordering**
    Per-die-ish: BIOS-set, identical across same-SKU boards. WEAK identity.

23. **ACPI device enumeration order**
    Per-die: same as DMI — usually not unique among same-SKU.

24. **PCIe link-rate downgrades historical (`aer_dev_correctable`)**
    Already in Phase 19 (s5). Skip.

25. **TPM manufacturer info / EK certificate hash**
    Per-die: strong (EK is endorsement-key, unique). But sensitive — attacker
    extracting EK can clone identity. **High risk**, deliberately excluded.

---

### F. Software-State-Derived (Identity by trace)

26. **BIOS POST timing fingerprint**
    Requires reboot — out of scope.

27. **Kernel boot-time `dmesg` entry timing**
    `dmesg -T` first 50 lines give per-board fingerprint. WEAK (changes on
    kernel upgrade).

28. **systemd unit start-order graph** — too brittle.

---

### Picked 5 for implementation

| ID  | Name                         | Why                                          |
|-----|------------------------------|----------------------------------------------|
| s10 | VBUS / voltage droop         | Genuinely analog, root-free, novel           |
| s11 | PCIe SerDes EQ coefficients  | Set-once-at-train, very stable, root-light   |
| s12 | DDR PHY / DIMM SPD + training| Stable manufacturing fingerprint             |
| s13 | NVMe SMART/hwmon thermal     | Temporal & wear evidence, root-free          |
| s14 | Per-CU shader skew (HIP)     | gfx1151-specific, **most novel**             |
