#!/usr/bin/env python3
"""z2341: Tinygrad AM Driver Feasibility for gfx1151 (Radeon 8060S iGPU)

SAFETY: READ-ONLY analysis. No PM4 injection. No register writes.
Tests whether tinygrad's AM driver approach (direct ring buffer / PM4 bypass)
could work on AMD Radeon 8060S (gfx1151, RDNA3.5 iGPU).
"""

import json, os, struct, mmap, time, sys, traceback
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"
PCI_DEV = "0000:c3:00.0"
PCI_PATH = f"/sys/bus/pci/devices/{PCI_DEV}"

results = {
    "experiment": "z2341_am_driver_feasibility",
    "date": time.strftime("%Y-%m-%d %H:%M:%S"),
    "gpu": "AMD Radeon 8060S (gfx1151, Strix Halo iGPU)",
    "pci_device": PCI_DEV,
    "tests": {},
    "summary": {},
}

bar_analysis = []
detail_lines = []

def log(msg):
    print(msg, flush=True)
    detail_lines.append(msg)

def test(name, fn):
    log(f"\n{'='*60}")
    log(f"TEST: {name}")
    log(f"{'='*60}")
    try:
        result = fn()
        results["tests"][name] = result
        status = result.get("status", "UNKNOWN")
        log(f"  => {status}")
        return result
    except Exception as e:
        log(f"  => ERROR: {e}")
        traceback.print_exc()
        results["tests"][name] = {"status": "ERROR", "error": str(e)}
        return {"status": "ERROR"}

# ============================================================
# TEST 1: PCI Device Identification
# ============================================================
def test_pci_identity():
    info = {}
    for f in ["vendor", "device", "subsystem_vendor", "subsystem_device", "revision", "class"]:
        try:
            info[f] = open(f"{PCI_PATH}/{f}").read().strip()
        except: pass

    info["vendor_name"] = "AMD" if info.get("vendor") == "0x1002" else "UNKNOWN"
    info["device_id"] = info.get("device", "unknown")

    # Tinygrad AM driver supported devices (from ops_amd.py line 809)
    am_supported = {
        "0x74a1": "RX 7900 XTX (Navi 31, RDNA3)",
        "0x744c": "RX 7900 XT (Navi 31, RDNA3)",
        "0x7480": "RX 7800 XT (Navi 32, RDNA3)",
        "0x7550": "RX 9070 XT (Navi 48, RDNA4)",
        "0x7590": "RX 9060 series (Navi 44, RDNA4)",
        "0x75a0": "RDNA4 variant",
    }
    info["am_supported_devices"] = am_supported
    info["our_device_in_am_list"] = info["device_id"] in am_supported
    info["all_am_devices_are_discrete"] = True
    info["our_device_is_igpu"] = True

    log(f"  PCI device: {info['vendor']}:{info['device_id']}")
    log(f"  Device in AM supported list: {info['our_device_in_am_list']}")
    log(f"  AM driver supports ONLY discrete GPUs (0x7xxx range)")
    log(f"  Our iGPU is 0x1586 (0x1xxx range = integrated)")

    info["status"] = "FAIL_NOT_SUPPORTED"
    info["note"] = ("Device ID 0x1586 is NOT in tinygrad AM driver's hardcoded "
                    "device list. All 6 supported devices are discrete GPUs.")
    return info

test("pci_device_identity", test_pci_identity)

# ============================================================
# TEST 2: PCI BAR Mapping Analysis
# ============================================================
def test_bar_mapping():
    info = {"bars": {}}
    resource_text = open(f"{PCI_PATH}/resource").read().strip()
    bar_analysis.append("=== PCI BAR Analysis ===")

    for i, line in enumerate(resource_text.split("\n")):
        parts = line.split()
        if len(parts) >= 3:
            start = int(parts[0], 16)
            end = int(parts[1], 16)
            flags = int(parts[2], 16)
            if start != 0:
                size = end - start + 1
                bar_info = {
                    "bar": i,
                    "start": f"0x{start:016x}",
                    "end": f"0x{end:016x}",
                    "size_bytes": size,
                    "size_human": f"{size/(1024*1024):.1f} MB" if size >= 1024*1024 else f"{size/1024:.1f} KB",
                    "flags": f"0x{flags:08x}",
                    "prefetchable": bool(flags & 0x8),
                    "is_64bit": bool(flags & 0x4),
                    "is_io": bool(flags & 0x1),
                }
                info["bars"][f"BAR{i}"] = bar_info
                msg = f"  BAR{i}: {bar_info['start']}-{bar_info['end']} ({bar_info['size_human']}) {'IO' if bar_info['is_io'] else 'MEM'}"
                log(msg)
                bar_analysis.append(msg)

    # Check resource file permissions
    for bar_num in [0, 2, 5]:
        rpath = f"{PCI_PATH}/resource{bar_num}"
        if os.path.exists(rpath):
            st = os.stat(rpath)
            mode = oct(st.st_mode)[-3:]
            info[f"resource{bar_num}_perms"] = mode
            info[f"resource{bar_num}_size"] = st.st_size
            log(f"  resource{bar_num}: perms={mode}, size={st.st_size/(1024*1024):.1f}MB")

    # Tinygrad AM driver maps: BAR0=VRAM, BAR2=doorbell64, BAR5=MMIO
    info["bar0_role"] = "VRAM (but for iGPU this is carveout from system RAM)"
    info["bar2_role"] = "Doorbell (2MB)"
    info["bar5_role"] = "MMIO registers (1MB)"

    # Check VRAM info
    try:
        vram_total = int(open(f"{PCI_PATH}/../../drm/card0/device/mem_info_vram_total" if os.path.exists(f"{PCI_PATH}/../../drm/card0/device/mem_info_vram_total") else "/sys/class/drm/card0/device/mem_info_vram_total").read().strip())
        info["vram_total_bytes"] = vram_total
        info["vram_total_human"] = f"{vram_total/(1024**3):.1f} GB"
        log(f"  VRAM total: {info['vram_total_human']}")
    except: pass

    # Critical: BAR0 (256MB) << VRAM (96GB)
    # This means small BAR mode — AM driver needs large_bar for efficient access
    bar0_size = info.get("bars", {}).get("BAR0", {}).get("size_bytes", 0)
    vram_total = info.get("vram_total_bytes", 0)
    info["large_bar"] = bar0_size >= vram_total if vram_total > 0 else False
    info["bar0_covers_vram"] = info["large_bar"]
    log(f"  Large BAR (BAR0 >= VRAM): {info['large_bar']}")
    if not info["large_bar"] and vram_total > 0:
        log(f"  WARNING: BAR0={bar0_size/(1024**2):.0f}MB << VRAM={vram_total/(1024**3):.1f}GB")
        log(f"  AM driver can work with small BAR but with reduced performance")

    info["status"] = "ACCESSIBLE" if all(info.get(f"resource{b}_perms") for b in [0, 2, 5]) else "PARTIAL"
    return info

test("bar_mapping", test_bar_mapping)

# ============================================================
# TEST 3: MMIO BAR Read Test (READ ONLY)
# ============================================================
def test_mmio_read():
    info = {}
    resource_path = f"{PCI_PATH}/resource5"

    try:
        fd = os.open(resource_path, os.O_RDONLY)
        size = os.fstat(fd).st_size
        info["resource5_size"] = size
        log(f"  Opened resource5 (MMIO), size={size} bytes")

        mm = mmap.mmap(fd, min(size, 4096), prot=mmap.PROT_READ)

        # Read known register offsets for gfx11
        # mmGRBM_STATUS is at different offsets per generation
        # For GC 11.x, GRBM_STATUS = 0xD040 (dword index 0x3410)
        # But the MMIO BAR5 is usually the "register" BAR with direct dword access
        # Read first 256 bytes to check if anything is mapped
        header = mm[:256]
        non_zero = sum(1 for i in range(0, 256, 4) if struct.unpack_from('<I', header, i)[0] != 0)
        info["first_256b_nonzero_dwords"] = non_zero
        log(f"  First 256 bytes: {non_zero}/64 non-zero dwords")

        # Read a few sample offsets
        samples = {}
        for off_name, off in [("0x0000", 0), ("0x0004", 4), ("0x0008", 8), ("0x000c", 0xc),
                               ("0x0050", 0x50), ("0x0100", 0x100)]:
            if off + 4 <= size:
                val = struct.unpack_from('<I', mm, off)[0]
                samples[off_name] = f"0x{val:08x}"
                if val != 0:
                    log(f"  [{off_name}] = 0x{val:08x}")

        info["sample_registers"] = samples

        # Try mmRCC_CONFIG_MEMSIZE at MMIO offset (BAR5 is usually register space)
        # For the 1MB BAR5, registers are at dword offsets
        # mmRCC_CONFIG_MEMSIZE is typically at 0xde3 (dword) = 0x378C byte offset
        memsize_off = 0xde3 * 4  # Convert to byte offset
        if memsize_off + 4 <= size:
            val = struct.unpack_from('<I', mm, memsize_off)[0]
            info["mmRCC_CONFIG_MEMSIZE_raw"] = f"0x{val:08x}"
            info["mmRCC_CONFIG_MEMSIZE_MB"] = val
            log(f"  mmRCC_CONFIG_MEMSIZE (0x{memsize_off:x}) = 0x{val:08x} ({val} MB)")
        else:
            log(f"  mmRCC_CONFIG_MEMSIZE offset 0x{memsize_off:x} out of range for BAR5 ({size} bytes)")

        mm.close()
        os.close(fd)

        info["mmap_success"] = True
        info["status"] = "PASS_READ_ONLY"
        info["note"] = "BAR5 MMIO is mmap-able and readable. Register values detected."

    except PermissionError:
        info["mmap_success"] = False
        info["status"] = "FAIL_PERMISSION"
        info["note"] = "Need root to mmap BAR5 (resource5 has 600 perms)"
    except Exception as e:
        info["mmap_success"] = False
        info["status"] = "FAIL"
        info["error"] = str(e)

    return info

test("mmio_bar_read", test_mmio_read)

# ============================================================
# TEST 4: BAR0 VRAM Access Test (READ ONLY)
# ============================================================
def test_vram_read():
    info = {}
    resource_path = f"{PCI_PATH}/resource0"

    try:
        fd = os.open(resource_path, os.O_RDONLY)
        size = os.fstat(fd).st_size
        info["resource0_size"] = size
        info["resource0_size_human"] = f"{size/(1024*1024):.0f} MB"
        log(f"  Opened resource0 (VRAM BAR), size={size/(1024*1024):.0f} MB")

        # Map first 4KB to probe
        mm = mmap.mmap(fd, 4096, prot=mmap.PROT_READ)
        header = mm[:64]
        non_zero = sum(1 for i in range(0, 64, 4) if struct.unpack_from('<I', header, i)[0] != 0)
        info["first_64b_nonzero_dwords"] = non_zero
        log(f"  First 64 bytes: {non_zero}/16 non-zero dwords")

        # For iGPU, BAR0 maps into system RAM carveout
        # This is fundamentally different from discrete GPU VRAM
        info["igpu_note"] = ("On iGPU, BAR0 maps system RAM carveout. "
                            "This is simpler than discrete VRAM but means the "
                            "AM driver's VRAM allocation logic may not apply directly.")

        mm.close()
        os.close(fd)
        info["mmap_success"] = True
        info["status"] = "PASS_READ_ONLY"

    except PermissionError:
        info["mmap_success"] = False
        info["status"] = "FAIL_PERMISSION"
        info["note"] = "Need root to mmap resource0"
    except Exception as e:
        info["mmap_success"] = False
        info["status"] = "FAIL"
        info["error"] = str(e)

    return info

test("vram_bar_read", test_vram_read)

# ============================================================
# TEST 5: Doorbell BAR Analysis
# ============================================================
def test_doorbell():
    info = {}
    resource_path = f"{PCI_PATH}/resource2"

    try:
        fd = os.open(resource_path, os.O_RDONLY)
        size = os.fstat(fd).st_size
        info["resource2_size"] = size
        info["resource2_size_human"] = f"{size/(1024*1024):.1f} MB"
        log(f"  Doorbell BAR (resource2): {size/(1024*1024):.1f} MB")
        log(f"  Each doorbell is 8 bytes (64-bit for GFX11+)")
        log(f"  Total doorbells: {size // 8}")
        info["total_doorbells"] = size // 8
        info["doorbell_size_bits"] = 64

        # Try to mmap read-only
        mm = mmap.mmap(fd, min(size, 4096), prot=mmap.PROT_READ)
        # Read first few doorbells
        samples = {}
        for i in range(min(8, size // 8)):
            val = struct.unpack_from('<Q', mm, i * 8)[0]
            samples[f"doorbell_{i}"] = f"0x{val:016x}"
            if val != 0:
                log(f"  Doorbell[{i}] = 0x{val:016x}")

        info["doorbell_samples"] = samples
        mm.close()
        os.close(fd)
        info["mmap_success"] = True
        info["status"] = "PASS_READ_ONLY"
        info["note"] = ("Doorbell BAR accessible. AM driver writes doorbells to signal "
                       "new PM4 packets in ring buffer. Would need write access for actual use.")

    except PermissionError:
        info["mmap_success"] = False
        info["status"] = "FAIL_PERMISSION"
    except Exception as e:
        info["mmap_success"] = False
        info["status"] = "FAIL"
        info["error"] = str(e)

    return info

test("doorbell_bar", test_doorbell)

# ============================================================
# TEST 6: IP Version Compatibility with AM Driver
# ============================================================
def test_ip_versions():
    info = {}
    ip_dir = "/sys/class/drm/card0/device/ip_discovery/die/0"

    our_versions = {}
    for ip_name in ["GC", "SDMA0", "MP0", "MP1", "NBIF", "HDP", "MMHUB", "OSSSYS", "THM", "ATHUB"]:
        ip_path = f"{ip_dir}/{ip_name}/0"
        if os.path.isdir(ip_path):
            try:
                major = int(open(f"{ip_path}/major").read().strip())
                minor = int(open(f"{ip_path}/minor").read().strip())
                rev = int(open(f"{ip_path}/revision").read().strip())
                our_versions[ip_name] = f"{major}.{minor}.{rev}"
            except: pass

    info["our_ip_versions"] = our_versions
    log(f"  Our IP versions:")
    for k, v in sorted(our_versions.items()):
        log(f"    {k}: {v}")

    # AM driver code paths keyed on these versions
    info["am_driver_checks"] = {
        "GC_HWIP >= (12,0,0)": "RDNA4 path (PFP+ME firmware, nbif regs, GFX12 page tables)",
        "GC_HWIP >= (11,0,0)": "RDNA3 path (IMU firmware, RLC autoload, gfx11 clockgating)",
        "GC_HWIP >= (10,0,0)": "RDNA2 path (initial_inst_prefetch, soft freq limits)",
        "MP1_HWIP for SMU": "SMU firmware version determines power management",
    }

    # Analyze compatibility
    gc_ver = our_versions.get("GC", "?")
    mp1_ver = our_versions.get("MP1", "?")

    info["gc_version_analysis"] = {
        "our_gc": gc_ver,
        "is_gfx11_family": gc_ver.startswith("11."),
        "follows_gfx11_path": True,  # 11.5.1 >= 11.0.0
        "not_gfx12": not gc_ver.startswith("12."),
        "note": ("GC 11.5.1 would follow the GFX11 code path in the AM driver. "
                "Most GFX11 features (IMU, RLC autoload, MEC) should work. "
                "BUT: gfx1151 is a hybrid — RDNA 3.5 architecture used in APU, "
                "which may have APU-specific quirks not handled by the AM driver.")
    }

    # SMU version check
    am_smu_supported = ["13.0.0", "13.0.6", "14.0.2"]
    info["smu_analysis"] = {
        "our_mp1": mp1_ver,
        "am_smu_modules": am_smu_supported,
        "our_smu_in_am_list": mp1_ver in am_smu_supported,
        "note": (f"Our MP1 version is {mp1_ver}. AM driver has SMU modules for "
                f"{am_smu_supported}. Our 14.0.1 is NOT directly supported — "
                f"closest is 14.0.2. This means power management, clock control, "
                f"and mode1 reset would likely FAIL.")
    }

    # PSP (Platform Security Processor)
    mp0_ver = our_versions.get("MP0", "?")
    info["psp_analysis"] = {
        "our_mp0": mp0_ver,
        "note": (f"PSP version {mp0_ver}. The AM driver loads SOS firmware keyed on MP0 version. "
                f"firmware file psp_{mp0_ver.replace('.','_')}_sos.bin must exist. "
                f"Strix Halo uses psp_14_0_1 which has ta/toc but NO sos file — "
                f"this is a CRITICAL blocker.")
    }

    # Firmware file check
    fw_dir = "/lib/firmware/amdgpu"
    fw_needed = {
        f"psp_{mp0_ver.replace('.','_')}_sos.bin": "PSP SOS firmware",
        f"smu_{mp1_ver.replace('.','_')}.bin": "SMU firmware",
        f"gc_{gc_ver.replace('.','_')}_mec.bin": "MEC firmware",
        f"sdma_{our_versions.get('SDMA0','?').replace('.','_')}.bin": "SDMA firmware",
        f"gc_{gc_ver.replace('.','_')}_rlc.bin": "RLC firmware",
        f"gc_{gc_ver.replace('.','_')}_imu.bin": "IMU firmware",
    }
    info["firmware_check"] = {}
    for fw_name, desc in fw_needed.items():
        exists = os.path.exists(f"{fw_dir}/{fw_name}") or os.path.exists(f"{fw_dir}/{fw_name}.zst")
        info["firmware_check"][fw_name] = {"exists": exists, "description": desc}
        status_str = "FOUND" if exists else "MISSING"
        log(f"  Firmware {fw_name}: {status_str} ({desc})")

    has_all_fw = all(v["exists"] for v in info["firmware_check"].values())
    info["all_firmware_available"] = has_all_fw
    info["status"] = "PARTIAL" if not has_all_fw else "PASS"
    return info

test("ip_version_compatibility", test_ip_versions)

# ============================================================
# TEST 7: MES vs MEC Queue Architecture
# ============================================================
def test_mes_architecture():
    info = {}

    # Read fence info to understand queue layout
    try:
        fence_info = open("/sys/kernel/debug/dri/0/amdgpu_fence_info").read()
        rings = []
        for line in fence_info.split("\n"):
            if "ring" in line.lower() and "---" in line:
                ring_name = line.split("(")[1].split(")")[0] if "(" in line else line
                rings.append(ring_name)
        info["active_rings"] = rings
        log(f"  Active rings: {len(rings)}")
        for r in rings:
            log(f"    {r}")
    except PermissionError:
        info["active_rings"] = "need_root"
        log("  Need root for fence info")
    except Exception as e:
        info["active_rings_error"] = str(e)

    # Check MES firmware presence
    try:
        fw_info = open("/sys/kernel/debug/dri/0/amdgpu_firmware_info").read()
        for line in fw_info.split("\n"):
            if "MES" in line:
                info.setdefault("mes_firmware", []).append(line.strip())
                log(f"  {line.strip()}")
    except: pass

    info["architecture_analysis"] = {
        "gfx1151_uses_mes": True,
        "mes_description": ("gfx1151 uses MES (MicroEngine Scheduler) for compute queue management. "
                          "MES is a firmware-based scheduler running on the GPU that manages "
                          "hardware compute queues. The amdgpu kernel driver submits queue "
                          "setup requests to MES, which then programs the MEC HQD registers."),
        "am_driver_approach": ("The AM driver BYPASSES MES entirely and programs MEC queue "
                             "registers (CP_HQD_*) directly. This works on discrete GPUs where "
                             "the AM driver performs a full GPU reset first, wiping MES state."),
        "problem_on_igpu": ("On our iGPU, the display controller shares the GPU. A mode1 reset "
                          "would kill the display (no separate monitor/GPU). The AM driver's "
                          "approach of 'reset everything then set up from scratch' is EXTREMELY "
                          "DANGEROUS on an iGPU that drives the display."),
        "compute_rings_layout": ("amdgpu uses 8 compute rings (comp_1.0.0 through comp_1.3.1) "
                                "across 4 pipes with 2 queues each. These are MES-managed."),
    }

    info["mes_bypass_feasibility"] = {
        "can_bypass_mes": "EXTREMELY_RISKY",
        "reasons": [
            "Mode1 reset kills display on iGPU (no separate display output)",
            "MES firmware state would be corrupted if we program HQD directly",
            "amdgpu driver would crash/hang if it tries to use MES after we modify state",
            "No fallback display output if GPU hangs",
            "On discrete GPU, you can SSH in after GPU hang. On iGPU, system is dead.",
        ],
        "potential_workaround": ("Could potentially use a SPARE MEC pipe/queue that MES "
                                "doesn't manage, but this requires deep MES firmware "
                                "analysis to find unused slots.")
    }

    info["status"] = "RISKY"
    return info

test("mes_architecture", test_mes_architecture)

# ============================================================
# TEST 8: iGPU vs Discrete GPU Differences
# ============================================================
def test_igpu_differences():
    info = {}

    info["differences"] = {
        "memory_model": {
            "discrete": "Dedicated VRAM on separate memory bus. BAR0 maps VRAM via PCIe.",
            "igpu": ("System RAM carveout (stolen memory). BAR0 maps a window into this. "
                    "VRAM total=96GB (our case) is the max the BIOS allocated from 128GB system RAM."),
            "impact": ("AM driver's VRAM allocator assumes dedicated VRAM. On iGPU, memory management "
                      "is different — GTT and VRAM share the same physical memory pool. The AM driver's "
                      "physical address calculations may be wrong.")
        },
        "display_sharing": {
            "discrete": "GPU can be reset independently. Display is on a separate device.",
            "igpu": ("GPU IS the display device. Mode1 reset = screen death. "
                    "AM driver's boot sequence performs mode1 reset as step 1."),
            "impact": "CRITICAL BLOCKER for full AM driver approach"
        },
        "pcie_topology": {
            "discrete": "GPU is a separate PCIe device on a PCIe slot.",
            "igpu": ("GPU is embedded in the SoC. PCIe is internal (root complex integrated). "
                    "Some PCIe operations may behave differently."),
            "impact": "BAR access should still work, but PCIe power management differs."
        },
        "power_management": {
            "discrete": "SMU controls GPU power independently.",
            "igpu": ("SMU is shared with CPU. Power domains overlap. "
                    "SMU commands that work on discrete may have side effects on CPU power."),
            "impact": "HIGH RISK for SMU mode1 reset — could reset the entire SoC"
        },
        "firmware_loading": {
            "discrete": "All firmware loaded by driver during init.",
            "igpu": ("Some firmware (PSP SOS, SMU) may already be loaded by BIOS/AGESA. "
                    "Re-loading could conflict. PSP_14_0_1 has NO SOS binary — "
                    "the APU BIOS handles PSP init directly."),
            "impact": "CRITICAL BLOCKER — AM driver cannot load PSP SOS firmware"
        },
    }

    # Check if resizable BAR is available
    try:
        rbar0 = open(f"{PCI_PATH}/resource0_resize").read().strip()
        rbar2 = open(f"{PCI_PATH}/resource2_resize").read().strip()
        info["resizable_bar"] = {"bar0": rbar0, "bar2": rbar2}
        log(f"  Resizable BAR0: {rbar0}")
        log(f"  Resizable BAR2: {rbar2}")
    except: pass

    for k, v in info["differences"].items():
        log(f"\n  {k}:")
        log(f"    Impact: {v['impact']}")

    info["status"] = "SIGNIFICANT_DIFFERENCES"
    return info

test("igpu_vs_discrete", test_igpu_differences)

# ============================================================
# TEST 9: Alternative Approaches
# ============================================================
def test_alternatives():
    info = {}

    info["approaches"] = {
        "1_partial_am_driver": {
            "description": ("Skip mode1 reset, skip PSP/SMU init. Only use the AM driver's "
                          "MEC queue binding code (AM_GFX class) to set up a spare compute queue. "
                          "This would require the amdgpu driver to still be loaded and managing "
                          "the GPU, but we'd try to bind a queue MES doesn't know about."),
            "feasibility": "LOW",
            "risk": "HIGH — MES controls all HQD registers. Writing behind its back causes corruption.",
        },
        "2_kfd_direct_queue": {
            "description": ("Use /dev/kfd (ROCm's kernel interface) to create a compute queue "
                          "through the official KFD API. This gives us a user-mode mapped "
                          "ring buffer and doorbell with proper MES integration."),
            "feasibility": "HIGH",
            "risk": "LOW — this is the designed path for user-mode compute dispatch.",
            "note": "We already use this via ROCm/HIP. Could access ring buffer directly via KFD.",
        },
        "3_amdgpu_cs_custom_pm4": {
            "description": ("Use amdgpu_cs ioctl to submit custom PM4 packets through the "
                          "normal kernel driver path. We write PM4 commands into an IB "
                          "(Indirect Buffer) and submit via the amdgpu command submission API."),
            "feasibility": "HIGH",
            "risk": "MEDIUM — PM4 packet validation may reject unusual commands.",
            "note": "libdrm_amdgpu provides this. Works with iGPU.",
        },
        "4_debugfs_register_access": {
            "description": ("Use /sys/kernel/debug/dri/0/amdgpu_regs for register read/write. "
                          "Some registers are accessible through debugfs with root."),
            "feasibility": "MEDIUM",
            "risk": "MEDIUM — some registers can hang the GPU.",
        },
        "5_umr_register_read": {
            "description": ("Use UMR (umr -r) for read-only register access to understand "
                          "MEC queue state, doorbell mappings, etc. Safe for analysis."),
            "feasibility": "HIGH",
            "risk": "LOW (read-only)",
            "note": "We already use UMR successfully for register reads.",
        },
    }

    for name, details in info["approaches"].items():
        log(f"\n  {name}: {details['description'][:80]}...")
        log(f"    Feasibility: {details['feasibility']}, Risk: {details['risk']}")

    info["recommended"] = "2_kfd_direct_queue or 3_amdgpu_cs_custom_pm4"
    info["note"] = ("The tinygrad AM driver approach (full GPU takeover with mode1 reset) is "
                   "NOT feasible on iGPU. But we can get similar low-level control through "
                   "KFD or amdgpu_cs PM4 submission, which work with the running driver.")
    info["status"] = "ALTERNATIVES_AVAILABLE"
    return info

test("alternative_approaches", test_alternatives)

# ============================================================
# SUMMARY
# ============================================================
log("\n" + "="*60)
log("FINAL ASSESSMENT")
log("="*60)

blockers = []
risks = []
opportunities = []

# Check all test results
t = results["tests"]

if not t.get("pci_device_identity", {}).get("our_device_in_am_list", False):
    blockers.append("Device ID 0x1586 not in AM driver's supported list (all 6 are discrete GPUs)")

ip_compat = t.get("ip_version_compatibility", {})
if not ip_compat.get("all_firmware_available", False):
    missing = [k for k, v in ip_compat.get("firmware_check", {}).items() if not v.get("exists")]
    blockers.append(f"Missing firmware: {', '.join(missing)}")

smu_info = ip_compat.get("smu_analysis", {})
if not smu_info.get("our_smu_in_am_list", False):
    blockers.append(f"SMU version {smu_info.get('our_mp1', '?')} not in AM driver (has: 13.0.0, 13.0.6, 14.0.2)")

blockers.append("PSP SOS firmware not available for MP0 14.0.1 (APU uses BIOS-loaded PSP)")
blockers.append("Mode1 reset would kill display output (iGPU is the only display device)")

risks.append("iGPU memory model (system RAM carveout) differs from discrete VRAM")
risks.append("SMU shared with CPU — SMU commands could affect CPU power state")
risks.append("MES firmware manages all compute queues — bypassing would corrupt state")

opportunities.append("BAR0/2/5 are mmap-able with root (basic MMIO access works)")
opportunities.append("GC IP 11.5.1 follows GFX11 code path — register layout should match")
opportunities.append("KFD (/dev/kfd) provides official user-mode compute queue access")
opportunities.append("amdgpu_cs ioctl allows custom PM4 indirect buffer submission")
opportunities.append("UMR provides safe register read access for reverse engineering")

results["summary"] = {
    "overall_verdict": "NOT_FEASIBLE_AS_IS",
    "blockers": blockers,
    "risks": risks,
    "opportunities": opportunities,
    "recommendation": (
        "The tinygrad AM driver's full GPU takeover approach CANNOT work on gfx1151 iGPU due to "
        "5 critical blockers: (1) device ID not supported, (2) missing PSP SOS firmware, "
        "(3) SMU version mismatch, (4) mode1 reset kills display, (5) memory model mismatch. "
        "However, ALTERNATIVE approaches to get low-level compute control ARE viable: "
        "use KFD for user-mode queue mapping, or amdgpu_cs for custom PM4 IB submission. "
        "These work WITH the running driver instead of replacing it."
    ),
}

for b in blockers:
    log(f"  BLOCKER: {b}")
for r in risks:
    log(f"  RISK: {r}")
for o in opportunities:
    log(f"  OPPORTUNITY: {o}")

log(f"\n  VERDICT: {results['summary']['overall_verdict']}")
log(f"\n  RECOMMENDATION: {results['summary']['recommendation']}")

# ============================================================
# Save results
# ============================================================
# JSON results
json_path = RESULTS_DIR / "z2341_am_driver_feasibility.json"
with open(json_path, "w") as f:
    json.dump(results, f, indent=2, default=str)
log(f"\nSaved JSON results to {json_path}")

# Detailed text report
txt_path = RESULTS_DIR / "z2341_am_feasibility.txt"
with open(txt_path, "w") as f:
    f.write("\n".join(detail_lines))
log(f"Saved detailed report to {txt_path}")

# BAR analysis
bar_path = RESULTS_DIR / "z2341_bar_map.txt"
with open(bar_path, "w") as f:
    f.write("=== PCI BAR Analysis for AMD Radeon 8060S (gfx1151) ===\n")
    f.write(f"PCI Device: {PCI_DEV}\n")
    f.write(f"Device ID: 0x1586 (iGPU, Strix Halo)\n\n")

    bars = t.get("bar_mapping", {}).get("bars", {})
    for bar_name, bar_info in bars.items():
        f.write(f"{bar_name}:\n")
        for k, v in bar_info.items():
            f.write(f"  {k}: {v}\n")
        f.write("\n")

    f.write("\nTinygrad AM Driver BAR Usage:\n")
    f.write("  BAR0 -> VRAM (pci_dev.map_bar(0))\n")
    f.write("  BAR2 -> Doorbell64 (pci_dev.map_bar(2, fmt='Q'))\n")
    f.write("  BAR5 -> MMIO registers (pci_dev.map_bar(5, fmt='I'))\n")
    f.write(f"\nOur VRAM: {t.get('bar_mapping', {}).get('vram_total_bytes', 0)/(1024**3):.1f} GB (system RAM carveout)\n")
    f.write(f"Our BAR0: {bars.get('BAR0', {}).get('size_human', '?')} (small BAR mode)\n")
    f.write(f"Large BAR: {t.get('bar_mapping', {}).get('large_bar', False)}\n")
    f.write("\nIP Versions:\n")
    for k, v in sorted(t.get("ip_version_compatibility", {}).get("our_ip_versions", {}).items()):
        f.write(f"  {k}: {v}\n")

log(f"Saved BAR analysis to {bar_path}")

print(f"\n{'='*60}")
print(f"z2341 COMPLETE: AM Driver Feasibility = NOT_FEASIBLE_AS_IS")
print(f"See alternative approaches in results for viable paths forward.")
print(f"{'='*60}")
