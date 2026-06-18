#!/usr/bin/env python3
"""z2065 ISA Probe: Test every hardware register and channel for gfx1151.

Tests each s_getreg hwreg in a SEPARATE kernel to isolate crashes.
Also tests debugfs channels, MMIO reads, and dangerous s_sendmsg_rtn.

Run with:
  sudo HSA_OVERRIDE_GFX_VERSION=11.0.0 PYTORCH_ROCM_ARCH=gfx1100 PYTHONUNBUFFERED=1 \
    PATH=/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/venv/bin:$PATH \
    /home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/venv/bin/python scripts/z2065_isa_probe.py
"""

import os, sys, json, time, struct, traceback

os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '11.0.0')
os.environ.setdefault('PYTORCH_ROCM_ARCH', 'gfx1100')

import torch
from torch.utils.cpp_extension import load_inline

# ── Auto-detect GPU card ──
def _find_card():
    for c in range(8):
        if os.path.exists(f'/sys/class/drm/card{c}/device/gpu_metrics'):
            return c
    return 0

CARD = _find_card()
DEVICE = 'cuda'
print(f"[INFO] Using card{CARD}, device={torch.cuda.get_device_name(0)}")

# ── Results collector ──
results = []

def record(name, category, status, details="", values=None):
    r = {
        "name": name,
        "category": category,
        "status": status,
        "details": details,
        "values": values or {},
    }
    results.append(r)
    sym = {"PASS": "✓", "FAIL": "✗", "CRASH": "💀", "SKIP": "⏭", "ERROR": "⚠"}
    print(f"  [{sym.get(status, '?')}] {name}: {status} — {details}")
    return r


def gpu_alive():
    """Check if GPU is still responsive."""
    try:
        torch.cuda.synchronize()
        v = torch.randn(1, device=DEVICE).item()
        return True
    except:
        return False


# ═══════════════════════════════════════════════════════════════
# PART 1: ISA s_getreg tests (each compiled separately)
# ═══════════════════════════════════════════════════════════════

def test_getreg(hwreg_num, hwreg_name, description):
    """Test a single s_getreg_b32 hwreg(N) in an isolated kernel."""
    print(f"\n--- Testing s_getreg hwreg({hwreg_num}) = {hwreg_name}: {description}")

    hip_src = f'''
#include <hip/hip_runtime.h>
#include <torch/extension.h>

__global__ void read_hwreg_kern(int* out, int n) {{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {{
        unsigned int val;
        asm volatile("s_getreg_b32 %0, hwreg({hwreg_num})" : "=s"(val));
        val = __builtin_amdgcn_readfirstlane(val);
        out[tid] = (int)val;
    }}
}}

torch::Tensor read_hwreg(int n) {{
    auto out = torch::zeros({{n}}, torch::dtype(torch::kInt32).device(torch::kCUDA));
    read_hwreg_kern<<<(n+63)/64, 64>>>(out.data_ptr<int>(), n);
    return out;
}}
'''
    cpp_src = '''
#include <torch/extension.h>
torch::Tensor read_hwreg(int n);
'''

    try:
        mod = load_inline(
            name=f'hwreg_{hwreg_num}_probe',
            cpp_sources=cpp_src,
            cuda_sources=hip_src,
            functions=['read_hwreg'],
            extra_cuda_cflags=['-O2', '--offload-arch=gfx1100'],
            verbose=False,
        )

        out = mod.read_hwreg(256)
        torch.cuda.synchronize()

        vals = out.cpu().numpy()
        unique = sorted(set(int(v) for v in vals))
        hex_vals = [f"0x{v & 0xFFFFFFFF:08x}" for v in unique[:8]]

        record(
            f"s_getreg hwreg({hwreg_num}) {hwreg_name}",
            "ISA",
            "PASS",
            f"{len(unique)} unique: {', '.join(hex_vals)}",
            {"unique_count": len(unique), "values_hex": hex_vals, "raw_first": int(vals[0])},
        )
        return True

    except Exception as e:
        err = str(e)
        if "memory access fault" in err.lower():
            record(f"s_getreg hwreg({hwreg_num}) {hwreg_name}", "ISA", "CRASH", err[:200])
            return False
        else:
            record(f"s_getreg hwreg({hwreg_num}) {hwreg_name}", "ISA", "ERROR", err[:200])
            return gpu_alive()


# ═══════════════════════════════════════════════════════════════
# PART 2: Dangerous s_sendmsg_rtn tests
# ═══════════════════════════════════════════════════════════════

def test_sendmsg_rtn_b32(msg_code_int, msg_name, description):
    """Test s_sendmsg_rtn_b32 — KNOWN GPU KILLERS."""
    print(f"\n--- ⚠️ DANGEROUS: s_sendmsg_rtn_b32 sendmsg({msg_name}): {description}")

    hip_src = f'''
#include <hip/hip_runtime.h>
#include <torch/extension.h>

__global__ void sendmsg_kern(int* out, int n) {{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n && threadIdx.x == 0) {{
        unsigned int val;
        asm volatile("s_sendmsg_rtn_b32 %0, {msg_code_int}" : "=s"(val));
        val = __builtin_amdgcn_readfirstlane(val);
        out[tid] = (int)val;
    }}
}}

torch::Tensor sendmsg_test(int n) {{
    auto out = torch::zeros({{n}}, torch::dtype(torch::kInt32).device(torch::kCUDA));
    sendmsg_kern<<<1, 1>>>(out.data_ptr<int>(), n);
    return out;
}}
'''
    cpp_src = '''
#include <torch/extension.h>
torch::Tensor sendmsg_test(int n);
'''

    try:
        mod = load_inline(
            name=f'sendmsg_{msg_code_int}_probe',
            cpp_sources=cpp_src,
            cuda_sources=hip_src,
            functions=['sendmsg_test'],
            extra_cuda_cflags=['-O2', '--offload-arch=gfx1100'],
            verbose=False,
        )

        out = mod.sendmsg_test(1)
        torch.cuda.synchronize()

        val = out.cpu().item()
        record(f"s_sendmsg_rtn_b32 msg({msg_name})", "ISA-DANGEROUS",
               "PASS", f"SURPRISING OK! value=0x{val & 0xFFFFFFFF:08x}",
               {"value_hex": f"0x{val & 0xFFFFFFFF:08x}"})
        return True

    except Exception as e:
        err = str(e)
        if "memory access fault" in err.lower() or "gpu recover" in err.lower():
            record(f"s_sendmsg_rtn_b32 msg({msg_name})", "ISA-DANGEROUS",
                   "CRASH", f"GPU CRASH: {err[:200]}")
            return False
        else:
            record(f"s_sendmsg_rtn_b32 msg({msg_name})", "ISA-DANGEROUS",
                   "ERROR", f"{err[:200]}")
            return gpu_alive()


# ═══════════════════════════════════════════════════════════════
# PART 3: Debugfs / sysfs / MMIO tests
# ═══════════════════════════════════════════════════════════════

def test_debugfs_file(name, path, binary=False):
    """Read a debugfs/sysfs file."""
    print(f"\n--- Testing {name}")
    try:
        mode = 'rb' if binary else 'r'
        with open(path, mode) as f:
            data = f.read()
        if binary:
            record(name, "debugfs", "PASS", f"{len(data)} bytes, header=0x{data[0]:02x}{data[1]:02x}" if len(data) >= 2 else f"{len(data)} bytes",
                   {"size": len(data)})
        else:
            lines = data.strip().split('\n') if data.strip() else []
            preview = lines[0][:80] if lines else "(empty)"
            record(name, "debugfs", "PASS", f"{len(lines)} lines: {preview}",
                   {"line_count": len(lines)})
    except PermissionError:
        record(name, "debugfs", "FAIL", "Permission denied (need sudo)")
    except Exception as e:
        record(name, "debugfs", "ERROR", str(e)[:200])


def test_mmio_reg(offset, name):
    """Read MMIO register via amdgpu_regs2."""
    print(f"\n--- Testing MMIO 0x{offset:04X} ({name})")
    try:
        path = f"/sys/kernel/debug/dri/{CARD}/amdgpu_regs2"
        with open(path, 'rb') as f:
            f.seek(offset)
            raw = f.read(4)
        if len(raw) == 4:
            val = struct.unpack('<I', raw)[0]
            record(f"MMIO 0x{offset:04X} {name}", "MMIO", "PASS",
                   f"0x{val:08X}", {"value_hex": f"0x{val:08X}", "value_dec": val})
        else:
            record(f"MMIO 0x{offset:04X} {name}", "MMIO", "FAIL", f"Short read: {len(raw)} bytes")
    except Exception as e:
        record(f"MMIO 0x{offset:04X} {name}", "MMIO", "ERROR", str(e)[:200])


def test_mmio_write_read(offset, name, write_val):
    """Try writing an MMIO register via amdgpu_regs2, then read back."""
    print(f"\n--- Testing MMIO write 0x{offset:04X} ({name}) = 0x{write_val:08X}")
    try:
        path = f"/sys/kernel/debug/dri/{CARD}/amdgpu_regs2"
        # Read original
        with open(path, 'rb') as f:
            f.seek(offset)
            orig_raw = f.read(4)
        orig_val = struct.unpack('<I', orig_raw)[0] if len(orig_raw) == 4 else None

        # Write
        with open(path, 'rb+') as f:
            f.seek(offset)
            f.write(struct.pack('<I', write_val))

        # Read back
        with open(path, 'rb') as f:
            f.seek(offset)
            new_raw = f.read(4)
        new_val = struct.unpack('<I', new_raw)[0] if len(new_raw) == 4 else None

        if new_val == write_val:
            record(f"MMIO write 0x{offset:04X} {name}", "MMIO-WRITE", "PASS",
                   f"Wrote 0x{write_val:08X}, read back 0x{new_val:08X} (orig: 0x{orig_val:08X})",
                   {"wrote": f"0x{write_val:08X}", "readback": f"0x{new_val:08X}", "original": f"0x{orig_val:08X}"})
        else:
            record(f"MMIO write 0x{offset:04X} {name}", "MMIO-WRITE", "FAIL",
                   f"Wrote 0x{write_val:08X} but read 0x{new_val:08X} (orig: 0x{orig_val:08X})",
                   {"wrote": f"0x{write_val:08X}", "readback": f"0x{new_val:08X}", "original": f"0x{orig_val:08X}"})

        # Restore original
        if orig_val is not None:
            with open(path, 'rb+') as f:
                f.seek(offset)
                f.write(struct.pack('<I', orig_val))

    except PermissionError:
        record(f"MMIO write 0x{offset:04X} {name}", "MMIO-WRITE", "FAIL", "Permission denied")
    except Exception as e:
        record(f"MMIO write 0x{offset:04X} {name}", "MMIO-WRITE", "ERROR", str(e)[:200])


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  z2065 ISA PROBE — Comprehensive Hardware Channel Test")
    print(f"  GPU: card{CARD} | {torch.cuda.get_device_name(0)}")
    print("=" * 70)

    # ── Phase 1: Confirmed-safe s_getreg ──
    print("\n\n" + "─" * 70)
    print("  PHASE 1: CONFIRMED SAFE s_getreg (known from z2050+)")
    print("─" * 70)

    test_getreg(1, "MODE", "FP rounding [3:0], denorm [7:4]")
    test_getreg(23, "HW_ID1", "WGP_ID, SA_ID, SE_ID")

    # ── Phase 2: Untested s_getreg ──
    print("\n\n" + "─" * 70)
    print("  PHASE 2: UNTESTED s_getreg (testing now)")
    print("─" * 70)

    for hwreg_num, name, desc in [
        (2, "STATUS", "SCC, EXECZ, VCCZ, IN_TG, HALT, VALID"),
        (3, "TRAPSTS", "Trap status: EXCP, DP_RATE, SAVECTX"),
        (24, "HW_ID2", "VMID, queue ID, pipe ID, ME ID"),
        (29, "SHADER_CYCLES", "Per-wave cycle counter (may not exist)"),
    ]:
        alive = test_getreg(hwreg_num, name, desc)
        if not alive:
            print("  *** GPU DEAD, stopping ISA tests ***")
            break

    # ── Phase 2b: Extended s_getreg ──
    print("\n\n" + "─" * 70)
    print("  PHASE 2b: EXTENDED s_getreg probes")
    print("─" * 70)

    if gpu_alive():
        for hwreg_num, name, desc in [
            (5, "GPR_ALLOC", "VGPR/SGPR allocation"),
            (6, "LDS_ALLOC", "LDS base and size"),
            (7, "IB_STS", "Instruction buffer status"),
            (8, "PC_LO", "Program counter low 32b"),
            (9, "PC_HI", "Program counter high 32b"),
            (20, "FLAT_SCR_LO", "Flat scratch addr low"),
            (21, "FLAT_SCR_HI", "Flat scratch addr high"),
            (25, "POPS_PACKER", "POPS packer ID"),
        ]:
            alive = test_getreg(hwreg_num, name, desc)
            if not alive:
                print("  *** GPU DEAD ***")
                break

    # ── Phase 3: Debugfs / sysfs ──
    print("\n\n" + "─" * 70)
    print("  PHASE 3: DEBUGFS & SYSFS")
    print("─" * 70)

    test_debugfs_file("amdgpu_wave", f"/sys/kernel/debug/dri/{CARD}/amdgpu_wave")
    test_debugfs_file("amdgpu_gpr", f"/sys/kernel/debug/dri/{CARD}/amdgpu_gpr")
    test_debugfs_file("amdgpu_fence_info", f"/sys/kernel/debug/dri/{CARD}/amdgpu_fence_info")
    test_debugfs_file("gpu_metrics", f"/sys/class/drm/card{CARD}/device/gpu_metrics", binary=True)
    test_debugfs_file("DVFS level", f"/sys/class/drm/card{CARD}/device/power_dpm_force_performance_level")
    test_debugfs_file("sched_mask", f"/sys/kernel/debug/dri/{CARD}/amdgpu_compute_sched_mask")

    # hwmon
    print(f"\n--- Testing hwmon sensors")
    hwmon_dir = None
    for d in os.listdir(f'/sys/class/drm/card{CARD}/device/hwmon/'):
        hwmon_dir = f'/sys/class/drm/card{CARD}/device/hwmon/{d}'
        break
    if hwmon_dir:
        sensors = {}
        for name in ['temp1_input', 'power1_average', 'freq1_input', 'freq2_input', 'in0_input', 'fan1_input']:
            try:
                with open(f'{hwmon_dir}/{name}', 'r') as f:
                    sensors[name] = int(f.read().strip())
            except:
                pass
        record("hwmon sensors", "sysfs", "PASS",
               f"{len(sensors)}: " + ", ".join(f"{k}={v}" for k, v in sensors.items()), sensors)

    # ── Phase 4: MMIO registers ──
    print("\n\n" + "─" * 70)
    print("  PHASE 4: MMIO REGISTERS (via amdgpu_regs2)")
    print("─" * 70)

    mmio_regs = [
        (0x8010, "GRBM_STATUS"),
        (0x8014, "GRBM_STATUS_SE0"),
        (0x8018, "GRBM_STATUS_SE1"),
        (0xB004, "RLC_STATUS"),
        (0xE000, "CG_SPLL_CNTL"),
        (0x8DE0, "SQ_IND_INDEX"),
        (0x8DE4, "SQ_IND_DATA"),
        (0x8D00, "SQ_PERFCOUNTER0_SELECT"),
        (0x8D04, "SQ_PERFCOUNTER1_SELECT"),
        (0x8D40, "SQ_PERFCOUNTER0_LO"),
        (0x8D44, "SQ_PERFCOUNTER0_HI"),
        (0x8040, "CP_STAT"),
        (0x8670, "CP_STALLED_STAT1"),
        (0x8674, "CP_STALLED_STAT2"),
        # Additional GFX11 registers
        (0x8020, "GRBM_STATUS2"),
        (0x30800, "GB_ADDR_CONFIG"),
        (0x8E48, "SQ_CONFIG"),
        (0x9100, "SPI_CONFIG_CNTL"),
        (0x9508, "TA_STATUS"),
        (0x9838, "DB_DEBUG"),
        (0xA014, "PA_SC_MODE_CNTL_0"),
    ]
    for offset, name in mmio_regs:
        test_mmio_reg(offset, name)

    # Try SQ_IND write-read cycle (write index, read data)
    print(f"\n--- Testing SQ_IND write→read cycle")
    try:
        path = f"/sys/kernel/debug/dri/{CARD}/amdgpu_regs2"
        # Write a wave-0 SGPR-0 index to SQ_IND_INDEX
        # SQ_IND_INDEX format: [15:0]=INDEX, [26:24]=WAVE_ID, etc.
        idx_val = 0x00000000  # wave 0, SGPR 0
        with open(path, 'rb+') as f:
            f.seek(0x8DE0)
            f.write(struct.pack('<I', idx_val))
        with open(path, 'rb') as f:
            f.seek(0x8DE4)
            data_raw = f.read(4)
        data_val = struct.unpack('<I', data_raw)[0] if len(data_raw) == 4 else -1
        record("SQ_IND write→read", "MMIO", "PASS",
               f"Wrote INDEX=0x{idx_val:08X}, DATA=0x{data_val:08X}",
               {"index": f"0x{idx_val:08X}", "data": f"0x{data_val:08X}"})
    except Exception as e:
        record("SQ_IND write→read", "MMIO", "ERROR", str(e)[:200])

    # ── Phase 5: DANGEROUS s_sendmsg_rtn ──
    print("\n\n" + "─" * 70)
    print("  PHASE 5: ⚠️  DANGEROUS s_sendmsg_rtn (KNOWN GPU KILLERS)")
    print("─" * 70)

    if not gpu_alive():
        print("  GPU already dead, marking all as SKIP")
        for code, name in [("0x80", "DOORBELL"), ("0x81", "DDID"), ("0x83", "GET_REALTIME")]:
            record(f"s_sendmsg_rtn_b32 msg({name})", "ISA-DANGEROUS",
                   "SKIP", "GPU dead before test")
    else:
        alive = test_sendmsg_rtn_b32(0x80, "DOORBELL", "Doorbell read")
        if not alive:
            for code, name in [("0x81", "DDID"), ("0x83", "GET_REALTIME")]:
                record(f"s_sendmsg_rtn_b32 msg({name})", "ISA-DANGEROUS",
                       "SKIP", "GPU dead from prior crash")
        else:
            alive = test_sendmsg_rtn_b32(0x81, "DDID", "Device/Die ID read")
            if not alive:
                record("s_sendmsg_rtn_b32 msg(GET_REALTIME)", "ISA-DANGEROUS",
                       "SKIP", "GPU dead from prior crash")
            else:
                test_sendmsg_rtn_b32(0x83, "GET_REALTIME", "Realtime clock read")

    print_summary()


def print_summary():
    """Print final summary table."""
    print("\n\n")
    print("=" * 95)
    print("  COMPREHENSIVE ISA PROBE RESULTS — gfx1151 (Radeon 8060S)")
    print("=" * 95)
    print(f"  {'Status':<6} {'Channel':<52} {'Category':<15} {'Details'}")
    print("─" * 95)

    for r in results:
        sym = {"PASS": "✓", "FAIL": "✗", "CRASH": "💀", "SKIP": "⏭", "ERROR": "⚠"}.get(r['status'], '?')
        det = r['details'][:45] + "..." if len(r['details']) > 45 else r['details']
        print(f"  {sym} {r['status']:<5} {r['name']:<52} {r['category']:<15} {det}")

    print("─" * 95)
    total = len(results)
    passed = sum(1 for r in results if r['status'] == 'PASS')
    failed = sum(1 for r in results if r['status'] == 'FAIL')
    crashed = sum(1 for r in results if r['status'] == 'CRASH')
    errors = sum(1 for r in results if r['status'] == 'ERROR')
    skipped = sum(1 for r in results if r['status'] == 'SKIP')
    print(f"  TOTAL: {total} | PASS: {passed} | FAIL: {failed} | CRASH: {crashed} | ERROR: {errors} | SKIP: {skipped}")
    print("=" * 95)

    # Save JSON
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'z2065_isa_probe_results.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "card": CARD,
            "gpu": f"gfx1151 ({torch.cuda.get_device_name(0)})",
            "summary": {"total": total, "pass": passed, "fail": failed,
                        "crash": crashed, "error": errors, "skip": skipped},
            "results": results,
        }, f, indent=2)
    print(f"\n  Results saved to: {out_path}")


if __name__ == "__main__":
    main()
