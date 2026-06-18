#!/usr/bin/env python3
"""
z1174: Partial Writes via Ethernet (Etherbone)

Partial writes = truncated tRAS timing for multi-level analog storage.
DDR3 tRAS minimum is 37.5ns. By issuing PRECHARGE before full charge,
we can store intermediate voltage levels.

Protocol:
1. ACTIVATE row (starts charging)
2. Wait partial tRAS (controlled delay)
3. PRECHARGE (stops charging early)
4. Result: Partial charge = analog level

Ethernet/Etherbone should be fast enough for cycle-accurate control.
"""

import sys
import time
import json
from datetime import datetime

sys.path.insert(0, '/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/venv/lib/python3.12/site-packages')

from litex.tools.litex_client import RemoteClient

DDR3_BASE = 0x40000000
SDRAM_BASE = 0x3000
DFII_CONTROL = SDRAM_BASE + 0x00
DFII_PI0_COMMAND = SDRAM_BASE + 0x04
DFII_PI0_COMMAND_ISSUE = SDRAM_BASE + 0x08
DFII_PI0_ADDRESS = SDRAM_BASE + 0x0c
DFII_PI0_BADDRESS = SDRAM_BASE + 0x10
DFII_PI0_WRDATA = SDRAM_BASE + 0x14
DFII_PI0_RDDATA = SDRAM_BASE + 0x18

HW_MODE = 0x0B  # SEL=1, CKE=1, RESET_N=1
SW_MODE = 0x0A  # SEL=0, CKE=1, RESET_N=1

# DDR3 Commands (active low, active = 0)
# Format: CS_N, RAS_N, CAS_N, WE_N
# In LiteDRAM DFII: [0]=CS, [1]=WE, [2]=CAS, [3]=RAS
CMD_NOP = 0x00           # All high (inactive)
CMD_ACTIVATE = 0x09      # RAS=1, CS=1 (CS+RAS active)
CMD_READ = 0x05          # CAS=1, CS=1 (CS+CAS active)
CMD_WRITE = 0x07         # CAS=1, WE=1, CS=1
CMD_PRECHARGE = 0x0B     # RAS=1, WE=1, CS=1
CMD_REFRESH = 0x0D       # RAS=1, CAS=1, CS=1


def init_ddr3(wb):
    """Initialize DDR3 with proper DFII settings"""
    wb.write(DFII_CONTROL, HW_MODE)
    time.sleep(0.1)
    # Issue some refreshes
    for _ in range(10):
        issue_refresh(wb)
    print("DDR3 initialized")


def issue_refresh(wb):
    """Issue auto-refresh command"""
    wb.write(DFII_CONTROL, SW_MODE)
    time.sleep(0.0001)
    wb.write(DFII_PI0_ADDRESS, 0x400)  # A10=1 for all banks
    wb.write(DFII_PI0_BADDRESS, 0)
    wb.write(DFII_PI0_COMMAND, CMD_PRECHARGE)
    wb.write(DFII_PI0_COMMAND_ISSUE, 1)
    time.sleep(0.0001)
    wb.write(DFII_PI0_COMMAND, CMD_REFRESH)
    wb.write(DFII_PI0_COMMAND_ISSUE, 1)
    time.sleep(0.0001)
    wb.write(DFII_CONTROL, HW_MODE)


def partial_write_row(wb, row, bank, delay_us):
    """
    Perform partial write by truncating tRAS.

    Args:
        row: Row address (0-8191 for 128Mbit)
        bank: Bank address (0-7)
        delay_us: Delay between ACTIVATE and PRECHARGE
    """
    # Switch to software mode
    wb.write(DFII_CONTROL, SW_MODE)

    # ACTIVATE the row (starts charging cells)
    wb.write(DFII_PI0_ADDRESS, row)
    wb.write(DFII_PI0_BADDRESS, bank)
    wb.write(DFII_PI0_COMMAND, CMD_ACTIVATE)
    wb.write(DFII_PI0_COMMAND_ISSUE, 1)

    # Wait partial tRAS (this determines charge level)
    if delay_us > 0:
        time.sleep(delay_us / 1_000_000)

    # PRECHARGE to stop charging early
    wb.write(DFII_PI0_ADDRESS, 0x400)  # A10=1 for precharge all
    wb.write(DFII_PI0_COMMAND, CMD_PRECHARGE)
    wb.write(DFII_PI0_COMMAND_ISSUE, 1)

    # Back to hardware mode
    wb.write(DFII_CONTROL, HW_MODE)


def measure_etherbone_latency(wb):
    """Measure round-trip time for single CSR access"""
    iterations = 100
    start = time.perf_counter()
    for _ in range(iterations):
        wb.read(DFII_CONTROL)
    end = time.perf_counter()
    return (end - start) / iterations * 1_000_000  # microseconds


def main():
    print("=" * 60)
    print("z1174: Partial Writes via Ethernet")
    print("=" * 60)

    wb = RemoteClient()
    wb.open()
    print("Connected to Etherbone at 192.168.0.50:1234")

    results = {
        "experiment": "z1174_partial_write_ethernet",
        "timestamp": datetime.now().isoformat(),
        "tests": []
    }

    init_ddr3(wb)

    # Measure Etherbone latency
    print("\n=== Etherbone Latency Measurement ===")
    latency_us = measure_etherbone_latency(wb)
    print(f"Single CSR access: {latency_us:.1f} µs")
    print(f"This limits minimum tRAS control to ~{latency_us:.0f} µs")

    results["etherbone_latency_us"] = latency_us

    # For reference: DDR3-800 timings
    # tRAS min = 37.5ns (row active time)
    # tRC = 50ns (row cycle time)
    # At 100MHz sys clock = 10ns per cycle
    # So Etherbone latency >> tRAS, meaning we can't do sub-microsecond control

    # Test partial writes with various delays
    print("\n=== Partial Write Test ===")
    print("Testing truncated tRAS at various delays...")
    print("(Note: Etherbone latency limits minimum delay)")

    test_row = 100
    test_bank = 0
    test_addr = DDR3_BASE + (test_row << 10) + (test_bank << 27)  # Approximate mapping

    # First, write known pattern via normal write
    pattern = 0xAAAAAAAA
    wb.write(test_addr, pattern)
    initial = wb.read(test_addr)
    print(f"\nInitial pattern: 0x{initial:08x}")

    # Test different partial write delays
    delays_us = [0, 1, 5, 10, 50, 100, 500, 1000]
    partial_results = []

    for delay in delays_us:
        # Refresh to restore full charge
        issue_refresh(wb)
        time.sleep(0.001)

        # Write known pattern
        wb.write(test_addr, pattern)

        # Perform partial write (truncated tRAS)
        partial_write_row(wb, test_row, test_bank, delay)

        # Wait a bit for any effects
        time.sleep(0.01)

        # Read back
        result = wb.read(test_addr)
        ones = bin(result).count('1')

        print(f"  {delay:4d} µs delay: 0x{result:08x} ({ones}/32 ones)")

        partial_results.append({
            "delay_us": delay,
            "result": f"0x{result:08x}",
            "ones_count": ones,
            "expected_ones": 16  # 0xAAAAAAAA has 16 ones
        })

    results["tests"].append({
        "name": "partial_write_delays",
        "pattern": "0xAAAAAAAA",
        "results": partial_results
    })

    # Test rapid ACT/PRE cycles
    print("\n=== Rapid ACT/PRE Cycle Test ===")
    print("Testing multiple rapid activation cycles...")

    cycle_results = []
    for num_cycles in [1, 5, 10, 20, 50]:
        # Refresh to start fresh
        issue_refresh(wb)
        time.sleep(0.001)

        # Write pattern
        wb.write(test_addr, pattern)

        # Perform multiple rapid ACT/PRE cycles
        wb.write(DFII_CONTROL, SW_MODE)
        for _ in range(num_cycles):
            wb.write(DFII_PI0_ADDRESS, test_row)
            wb.write(DFII_PI0_BADDRESS, test_bank)
            wb.write(DFII_PI0_COMMAND, CMD_ACTIVATE)
            wb.write(DFII_PI0_COMMAND_ISSUE, 1)
            wb.write(DFII_PI0_ADDRESS, 0x400)
            wb.write(DFII_PI0_COMMAND, CMD_PRECHARGE)
            wb.write(DFII_PI0_COMMAND_ISSUE, 1)
        wb.write(DFII_CONTROL, HW_MODE)

        time.sleep(0.01)
        result = wb.read(test_addr)
        ones = bin(result).count('1')

        print(f"  {num_cycles:2d} cycles: 0x{result:08x} ({ones}/32 ones)")

        cycle_results.append({
            "num_cycles": num_cycles,
            "result": f"0x{result:08x}",
            "ones_count": ones
        })

    results["tests"].append({
        "name": "rapid_act_pre_cycles",
        "results": cycle_results
    })

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    # Check if we saw any partial effects
    all_ones = [r["ones_count"] for r in partial_results]
    if min(all_ones) < 16 or max(all_ones) > 16:
        print("PARTIAL CHARGE EFFECTS DETECTED!")
        print(f"Ones count varied from {min(all_ones)} to {max(all_ones)}")
        results["partial_charge_detected"] = True
    else:
        print("No partial charge effects observed.")
        print(f"Etherbone latency ({latency_us:.0f} µs) may be too slow")
        print("for sub-microsecond tRAS control.")
        print("\nNOTE: For true multi-level storage, need:")
        print("  1. Hardware FSM for cycle-accurate timing, OR")
        print("  2. Decay-based levels (write full, let decay to target level)")
        results["partial_charge_detected"] = False

    results["conclusion"] = "partial_write_test_complete"

    # Save
    results_path = '/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/z1174_partial_write_ethernet.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    wb.close()
    print("Done!")

    return results


if __name__ == "__main__":
    main()
