"""FabricCrypt signature module.

5-signal HAL-bypass per-die fingerprint extractor.
Produces a 290-dimensional vector per capture from:
  - Block 1: TSC inter-core offsets (35 dims)
  - Block 2: Cacheline ping-pong RTT matrix (35 dims)
  - Block 3: DRAM-refresh-aligned latency histogram (200 dims)
  - Block 4: Syscall p99.9 tail percentiles (10 dims)
  - Block 5: NVMe queue-tail percentiles (10 dims)
"""
from .signature_v2 import extract_one, TOTAL_DIM, BLOCK_STARTS, DIMS

__all__ = ["extract_one", "TOTAL_DIM", "BLOCK_STARTS", "DIMS"]

# -------------------------------------------------------------------------
# Extended signature (Phase 19 + Phase 22) — optional, additive modules.
#
# Phase 19 (cross-host KS-verified at p_bonf < 0.01):
#   - gpu_clock_jitter   (20 dims) — hwmon GPU freq/temp/voltage at 1 kHz
#   - thermal_spread     (22 dims) — multi-zone hwmon temp constellation
#   - jacobian_dynamics  (30 dims) — d/dt + d2/dt2 + cross-signal Jacobian
#
# Phase 22 (light, deterministic discovery-class signals):
#   - pci_topology         (16 dims) — lspci tree + vendor hash
#   - pcie_link_state      (16 dims) — per-device speed/width degradation
#   - usb_descriptor       (16 dims) — lsusb tree + VID/PID hash
#   - dmi_smbios           (18 dims) — DMI types 0..19 hash (sudo -n optional)
#   - kernel_boot_timing   (16 dims) — journalctl subsystem ready-times
#   - ucsi_descriptors     (16 dims) — USB-C PD port advertisement
#   - amdgpu_safe_reads    (16 dims) — umr clock-scan / IP versions (sudo)
#   - hpet_drift           (12 dims) — CLOCK_REALTIME vs MONOTONIC ppm drift
#
# Total extended dim = 290 (base) + 72 (Phase 19) + 126 (Phase 22) = 488.
# Each module exposes `run(reps, out_dir)` returning the path of the saved
# .npz file with key 'vec' shape (reps, dim).
# -------------------------------------------------------------------------
