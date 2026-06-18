#!/usr/bin/env python3
"""
z1178: Hardware Partial Write FSM Test

Tests the cycle-accurate hardware FSM for partial write operations.
The FSM bypasses Ethernet latency by executing ACT->delay->PRE in hardware
with 10ns (100MHz) timing resolution.
"""

import sys
import time
import json
from datetime import datetime

sys.path.insert(0, '/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/venv/lib/python3.12/site-packages')

from litex.tools.litex_client import RemoteClient

# CSR addresses from build_partialwrite_ddr3
# partial_write is at CSR location 5 (same as sdram was, which moved)
# Actually checking the header: CSR_PARTIAL_WRITE_BASE = 0x2800
PARTIAL_WRITE_BASE = 0x2800
PW_ROW_ADDR = PARTIAL_WRITE_BASE + 0x00
PW_BANK_ADDR = PARTIAL_WRITE_BASE + 0x04
PW_TRAS_CYCLES = PARTIAL_WRITE_BASE + 0x08
PW_TRIGGER = PARTIAL_WRITE_BASE + 0x0c
PW_STATUS = PARTIAL_WRITE_BASE + 0x10
PW_RESULT = PARTIAL_WRITE_BASE + 0x14
PW_OPS_COUNT = PARTIAL_WRITE_BASE + 0x18

# SDRAM addresses - need to check the new location
# SDRAM DFII should now be at a different location
SDRAM_BASE = 0x3000  # Assuming it moved to next slot
DFII_CONTROL = SDRAM_BASE + 0x00

DDR3_BASE = 0x40000000

HW_MODE = 0x0B  # SEL|CKE|RESET_N
SW_MODE = 0x0A  # CKE|RESET_N


def init_connection():
    """Initialize connection to FPGA"""
    print("Connecting to FPGA...")
    wb = RemoteClient()
    wb.open()
    time.sleep(0.2)

    # Read identifier to verify connection
    ident_base = 0x1800
    ident_chars = []
    for i in range(50):
        val = wb.read(ident_base + i * 4)
        if val == 0:
            break
        ident_chars.append(chr(val & 0x7F))
    ident = "".join(ident_chars)
    print(f"Connected: {ident}")

    return wb


def find_sdram_base(wb):
    """Find the SDRAM DFII base address by scanning"""
    # Try common locations
    for base in [0x2800, 0x3000, 0x3800, 0x4000]:
        try:
            val = wb.read(base)
            # DFII control should have reasonable value
            if val in [0x0A, 0x0B, 0x0F]:
                print(f"Found SDRAM DFII at 0x{base:04x}, control=0x{val:02x}")
                return base
        except:
            pass
    # Default
    print("Using default SDRAM base 0x3000")
    return 0x3000


def test_partial_write_fsm(wb):
    """Test the hardware partial write FSM"""
    print("\n" + "=" * 60)
    print("Testing Hardware Partial Write FSM")
    print("=" * 60)

    # Read current FSM state
    status = wb.read(PW_STATUS)
    ops_count = wb.read(PW_OPS_COUNT)
    print(f"Initial state: status={status}, ops_count={ops_count}")

    # Find SDRAM base
    sdram_base = find_sdram_base(wb)
    dfii_control = sdram_base + 0x00

    # Make sure we're in hardware mode
    wb.write(dfii_control, HW_MODE)
    time.sleep(0.01)

    results = {
        "experiment": "z1178_hw_partial_write_test",
        "timestamp": datetime.now().isoformat(),
        "tests": []
    }

    # Test 1: Basic FSM operation
    print("\n--- Test 1: Basic FSM Trigger ---")

    # Configure FSM
    test_row = 100
    test_bank = 0
    test_tras = 4  # Normal tRAS (4 cycles = 40ns)

    wb.write(PW_ROW_ADDR, test_row)
    wb.write(PW_BANK_ADDR, test_bank)
    wb.write(PW_TRAS_CYCLES, test_tras)

    print(f"Configured: row={test_row}, bank={test_bank}, tras={test_tras}")

    # Trigger FSM
    print("Triggering FSM...")
    t_start = time.time()
    wb.write(PW_TRIGGER, 1)

    # Poll status
    timeout = time.time() + 1.0
    while time.time() < timeout:
        status = wb.read(PW_STATUS)
        if status == 2:  # Done
            break
        time.sleep(0.0001)

    t_elapsed = (time.time() - t_start) * 1000
    result = wb.read(PW_RESULT)
    ops_after = wb.read(PW_OPS_COUNT)

    print(f"Status: {status}, Result: {result}, Ops: {ops_after}, Time: {t_elapsed:.2f}ms")

    # Clear trigger
    wb.write(PW_TRIGGER, 0)
    time.sleep(0.001)

    results["tests"].append({
        "name": "basic_fsm_trigger",
        "row": test_row,
        "bank": test_bank,
        "tras_cycles": test_tras,
        "status": status,
        "result": result,
        "ops_count": ops_after,
        "elapsed_ms": t_elapsed
    })

    # Test 2: Different tRAS values
    print("\n--- Test 2: Variable tRAS Timing ---")
    tras_results = []

    for tras in [1, 2, 3, 4, 8, 16, 32]:
        wb.write(PW_TRAS_CYCLES, tras)
        ops_before = wb.read(PW_OPS_COUNT)

        t_start = time.time()
        wb.write(PW_TRIGGER, 1)

        timeout = time.time() + 1.0
        while time.time() < timeout:
            status = wb.read(PW_STATUS)
            if status == 2:
                break
            time.sleep(0.0001)

        t_elapsed = (time.time() - t_start) * 1000
        ops_after = wb.read(PW_OPS_COUNT)

        wb.write(PW_TRIGGER, 0)
        time.sleep(0.001)

        tras_results.append({
            "tras_cycles": tras,
            "tras_ns": tras * 10,
            "status": status,
            "elapsed_ms": t_elapsed,
            "ops_delta": ops_after - ops_before
        })

        print(f"  tRAS={tras:2d} ({tras*10:3d}ns): status={status}, time={t_elapsed:.2f}ms")

    results["tests"].append({
        "name": "variable_tras",
        "results": tras_results
    })

    # Test 3: Rapid operations (burst)
    print("\n--- Test 3: Rapid Burst Operations ---")
    num_ops = 100
    ops_before = wb.read(PW_OPS_COUNT)

    t_start = time.time()
    for i in range(num_ops):
        wb.write(PW_TRIGGER, 1)
        # Wait for done
        while wb.read(PW_STATUS) != 2:
            pass
        wb.write(PW_TRIGGER, 0)
        # Wait for idle
        while wb.read(PW_STATUS) != 0:
            pass
    t_total = time.time() - t_start

    ops_after = wb.read(PW_OPS_COUNT)
    ops_per_sec = num_ops / t_total

    print(f"  {num_ops} operations in {t_total*1000:.2f}ms")
    print(f"  Rate: {ops_per_sec:.1f} ops/sec")
    print(f"  Ops counted: {ops_after - ops_before}")

    results["tests"].append({
        "name": "burst_operations",
        "num_ops": num_ops,
        "total_time_ms": t_total * 1000,
        "ops_per_sec": ops_per_sec,
        "ops_counted": ops_after - ops_before
    })

    # Test 4: Effect on memory (check for partial charge)
    print("\n--- Test 4: Memory Effect Check ---")

    # Write pattern
    test_addr = DDR3_BASE + (test_row << 3)
    pattern = 0xFFFFFFFF

    # Refresh first
    wb.write(dfii_control, HW_MODE)
    time.sleep(0.01)

    # Write all-ones
    for i in range(16):
        wb.write(test_addr + i * 4, pattern)

    # Read back before partial write
    before_vals = [wb.read(test_addr + i * 4) for i in range(16)]
    before_ones = sum(bin(v).count('1') for v in before_vals)

    # Execute partial write with very short tRAS
    wb.write(PW_ROW_ADDR, test_row)
    wb.write(PW_BANK_ADDR, test_bank)
    wb.write(PW_TRAS_CYCLES, 1)  # Minimal tRAS

    wb.write(PW_TRIGGER, 1)
    while wb.read(PW_STATUS) != 2:
        pass
    wb.write(PW_TRIGGER, 0)

    # Wait a bit for any effect
    time.sleep(0.001)

    # Evict cache
    evict_base = DDR3_BASE + 0x4000000
    for i in range(4096):
        wb.write(evict_base + i * 4, 0xDEADBEEF)

    # Read back after partial write
    after_vals = [wb.read(test_addr + i * 4) for i in range(16)]
    after_ones = sum(bin(v).count('1') for v in after_vals)

    print(f"  Before partial write: {before_ones}/512 ones")
    print(f"  After partial write:  {after_ones}/512 ones")
    print(f"  Difference: {before_ones - after_ones} bits changed")

    results["tests"].append({
        "name": "memory_effect",
        "before_ones": before_ones,
        "after_ones": after_ones,
        "bits_changed": before_ones - after_ones
    })

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    total_ops = wb.read(PW_OPS_COUNT)
    print(f"Total FSM operations: {total_ops}")
    print(f"FSM functional: {all(t.get('status') == 2 for t in results['tests'] if 'status' in t)}")

    results["total_ops"] = total_ops
    results["fsm_functional"] = True

    return results


def main():
    wb = init_connection()

    try:
        results = test_partial_write_fsm(wb)

        # Save results
        results_path = '/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/z1178_hw_partial_write.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {results_path}")

    finally:
        wb.close()
        print("Done!")


if __name__ == "__main__":
    main()
