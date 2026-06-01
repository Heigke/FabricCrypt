# Hardware requirements

FabricCrypt assembles its fingerprint from signals that are
*manufacturing-process dependent* — meaning the precise numerical
values are unique to each die. The extraction code, however, is
platform-specific because it bypasses the OS HAL to talk to AMD-style
performance counters and the AMD platform's thermal/RAPL layout.

## Supported

| Platform | Tested | Notes |
|---|---|---|
| HP Z2 mini G1a + AMD Ryzen AI Max+ PRO 395 (Strix Halo / gfx1151) | yes, N=2 | our reference baseline |
| Other AMD Strix Halo APU systems | should work | thermal thresholds may need tuning (see `example.env`) |
| AMD Zen 5 desktops (Granite Ridge / Phoenix Point) | likely | raise `FABRICCRYPT_THERMAL_*` thresholds, big cooler = less pause-time |
| AMD Zen 4 laptops (Phoenix, Hawk Point) | likely | TSC layout differs; cacheline pair indices may need adjustment |

## Probably won't work

| Platform | Reason |
|---|---|
| Intel CPUs | `block_nanosleep` and the c-state usage path assume AMD platform topology; `block_tsc` will run but with very different statistics |
| Apple Silicon | No `/sys/class/powercap/intel-rapl`, no `/sys/class/thermal/thermal_zone*` |
| ARM SBC (Raspberry Pi, etc.) | Missing AMD-specific Linux paths |

## What you need at runtime

- **OS:** Ubuntu 24.04 LTS (we test on this). Kernel ≥ 6.8.
- **Filesystems exposed:**
  - `/sys/class/thermal/thermal_zone*/temp`
  - `/sys/class/powercap/intel-rapl:0/energy_uj` (RAPL is exposed for AMD too on recent kernels)
  - `/sys/devices/system/cpu/cpu*/cpuidle/state*/usage`
- **Tools:** `gcc` (compiles the C helpers), `python3.11+`.
- **No special permissions required** for collection. The C helpers use
  `sched_setaffinity`, the Python code reads from `/sys/...`. Both are
  unprivileged.

## Thermal safety

The default thresholds (abort 68 °C, pause 63 °C, cool to 50 °C) assume
a small chassis with limited cooling. On a desktop with a tower cooler
you can safely raise these:

```bash
export FABRICCRYPT_THERMAL_ABORT_C=85
export FABRICCRYPT_THERMAL_PAUSE_C=80
export FABRICCRYPT_THERMAL_COOL_C=65
```

If you see "ABORT thermal …" messages during collection, this is *safe* —
the script is doing its job, and you should let the box cool before
re-running.

## Storage

A complete reproduction (10 captures + 400-pair training set + classifier
weights) is under 50 MB on disk.

## Extended signature requirements (Phase 19 + Phase 22)

The 11 modules under `src/signature/` for the extended 488-dim signature
require additional system surfaces (all read-only, all standard Linux):

| Module                | Sysfs / tool requirement                              | Sudo? |
|-----------------------|--------------------------------------------------------|-------|
| `gpu_clock_jitter`    | `/sys/class/hwmon/hwmon*/freq*_input`, `temp1_input`, `in0_input` | no    |
| `thermal_spread`      | `/sys/class/hwmon/hwmon*/temp*_input` + `/sys/class/thermal/thermal_zone*/temp` | no |
| `jacobian_dynamics`   | `/sys/class/powercap/intel-rapl*/energy_uj` + hwmon temp/freq | no    |
| `pci_topology`        | `lspci -mn`, `lspci -t`                                | no    |
| `pcie_link_state`     | `/sys/bus/pci/devices/*/current_link_speed`+`width`    | no    |
| `usb_descriptor`      | `lsusb`, `lsusb -t`, `/sys/bus/usb/devices/*/speed`    | no    |
| `dmi_smbios`          | `dmidecode -t {0,1,2,3,4,16,17,19}` (best with sudo); falls back to `/sys/class/dmi/id/*` | optional sudo |
| `kernel_boot_timing`  | `journalctl -k --boot=0` (or `dmesg --ctime` fallback) | no    |
| `ucsi_descriptors`    | `/sys/class/power_supply/ucsi-source-psy-*`            | no    |
| `amdgpu_safe_reads`   | `umr` binary at `/opt/amdgpu/bin/umr`, sudo for read-only ops only | yes   |
| `hpet_drift`          | `clock_gettime(CLOCK_REALTIME/MONOTONIC)`              | no    |

UMR safety: `amdgpu_safe_reads` uses only read-only operations
(`--clock-scan`, `-lt`, `--list-ip-versions`).  Writes to the SMU
mailbox or reads of `amdgpu_regs_didt` are NOT performed — those are
known to cause GPU driver hangs / Data Fabric sync floods on
`gfx1151`.

## Crypto requirements

The Tier 2 protocol modules add one dependency:

```
bchlib>=2.0    # BCH(t=16, m=8) for reverse_fuzzy
```

`controlled_puf`, `multiround_protocol`, and `zk_inference_binding` use
only `numpy` + `hashlib` (already in the base requirements).
