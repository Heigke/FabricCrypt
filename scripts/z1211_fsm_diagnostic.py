#!/usr/bin/env python3
"""
z1211: Detailed FSM diagnostic to understand why partial writes aren't working.

This script:
1. Monitors FSM state during execution
2. Checks ext_dfi control path
3. Validates the complete data flow
"""

import time
import json
from datetime import datetime
from litex.tools.litex_client import RemoteClient

# CSR addresses
CSR_BASE = 0x0
DDRPHY_BASE = CSR_BASE + 0x800
SDRAM_BASE = CSR_BASE + 0x3000
LEDS_BASE = CSR_BASE + 0x2000
IDENTIFIER = CSR_BASE + 0x1800
DRAM_BASE = 0x40000000

# Partial write FSM registers
PW_BASE = 0x2800
PW_CONFIG = PW_BASE + 0x00
PW_WRITE_DATA = PW_BASE + 0x04
PW_REF_DATA = PW_BASE + 0x08
PW_CONTROL = PW_BASE + 0x0c
PW_STATUS = PW_BASE + 0x10
PW_RESULT = PW_BASE + 0x14
PW_RESULT_P1 = PW_BASE + 0x18
PW_RESULT_P2 = PW_BASE + 0x1c
PW_RESULT_P3 = PW_BASE + 0x20
PW_DEBUG = PW_BASE + 0x24
PW_EDGE_COUNT = PW_BASE + 0x28

# DFII registers (for software control comparison)
DFII_CONTROL = SDRAM_BASE + 0x00
DFII_PI0_COMMAND = SDRAM_BASE + 0x04
DFII_PI0_COMMAND_ISSUE = SDRAM_BASE + 0x08
DFII_PI0_ADDRESS = SDRAM_BASE + 0x0c
DFII_PI0_BADDRESS = SDRAM_BASE + 0x10
DFII_PI0_WRDATA = SDRAM_BASE + 0x14
DFII_PI0_RDDATA = SDRAM_BASE + 0x18


def make_config(row, col, bank, tras=8):
    """Pack DDR3 address into config register."""
    return (row << 18) | (col << 8) | (bank << 5) | tras


def addr_to_ddr3(linear_addr):
    """Convert linear address to DDR3 row/col/bank."""
    offset = linear_addr - DRAM_BASE
    row = (offset >> 14) & 0x3FFF
    bank = (offset >> 11) & 0x7
    col = (offset >> 1) & 0x3FF
    return row, col, bank


def print_fsm_state(wb, label=""):
    """Print current FSM state."""
    status = wb.read(PW_STATUS)
    debug = wb.read(PW_DEBUG)
    edge = wb.read(PW_EDGE_COUNT)
    r0 = wb.read(PW_RESULT)
    r1 = wb.read(PW_RESULT_P1)
    r2 = wb.read(PW_RESULT_P2)
    r3 = wb.read(PW_RESULT_P3)

    busy = status & 1
    done = (status >> 1) & 1

    print(f"  {label}: status=0x{status:02x} (busy={busy},done={done}) state={debug} edge={edge}")
    print(f"       results: P0=0x{r0:08x} P1=0x{r1:08x} P2=0x{r2:08x} P3=0x{r3:08x}")


def test_dfii_software_mode(wb):
    """Test DFII in software mode to verify basic DFI functionality."""
    print("\n=== Test 1: DFII Software Mode ===")

    test_addr = DRAM_BASE + 0x1000

    # Write a known pattern
    wb.write(test_addr, 0xDEADBEEF)
    time.sleep(0.01)
    result = wb.read(test_addr)
    print(f"  Normal write: 0xDEADBEEF -> read back 0x{result:08x}")

    # Enter software control
    print("  Entering software control mode...")
    wb.write(DFII_CONTROL, 0x0E)  # CKE | ODT | RESET_N, but NOT SEL
    time.sleep(0.01)

    # Read DFII_PI0_RDDATA
    rddata = wb.read(DFII_PI0_RDDATA)
    print(f"  PI0 RDDATA: 0x{rddata:08x}")

    # Return to hardware control
    wb.write(DFII_CONTROL, 0x0F)  # SEL | CKE | ODT | RESET_N
    time.sleep(0.01)

    # Verify memory still works
    result2 = wb.read(test_addr)
    print(f"  After SW mode: read 0x{result2:08x}")

    return result == 0xDEADBEEF and result2 == 0xDEADBEEF


def test_fsm_trigger(wb):
    """Test FSM triggering with detailed monitoring."""
    print("\n=== Test 2: FSM Trigger Monitoring ===")

    test_addr = DRAM_BASE + 0x2000
    row, col, bank = addr_to_ddr3(test_addr)
    config = make_config(row, col, bank, tras=8)

    print(f"  Target: 0x{test_addr:08x} -> row=0x{row:04x} col=0x{col:03x} bank={bank}")
    print(f"  Config: 0x{config:08x}")

    # Pre-write a pattern
    wb.write(test_addr, 0x11111111)
    time.sleep(0.01)
    initial = wb.read(test_addr)
    print(f"  Initial memory: 0x{initial:08x}")

    # Setup FSM
    wb.write(PW_CONFIG, config)
    wb.write(PW_WRITE_DATA, 0xAAAAAAAA)
    wb.write(PW_REF_DATA, 0x00000000)
    wb.write(PW_CONTROL, 0x00)  # Clear any previous state

    print_fsm_state(wb, "Before trigger")

    # Trigger with polling
    print("  Triggering...")
    wb.write(PW_CONTROL, 0x03)  # start + use_ext_dfi

    # Poll status rapidly
    for i in range(10):
        status = wb.read(PW_STATUS)
        debug = wb.read(PW_DEBUG)
        print(f"    Poll {i}: status=0x{status:02x} debug_state={debug}")
        if status & 0x02:  # done
            break
        time.sleep(0.001)

    wb.write(PW_CONTROL, 0x00)  # Clear trigger
    time.sleep(0.01)

    print_fsm_state(wb, "After trigger")

    # Check memory
    final = wb.read(test_addr)
    print(f"  Final memory: 0x{final:08x}")

    return final != initial


def test_fsm_without_ext_dfi(wb):
    """Test FSM trigger without ext_dfi to see if state machine runs."""
    print("\n=== Test 3: FSM Without ext_dfi ===")

    test_addr = DRAM_BASE + 0x3000
    row, col, bank = addr_to_ddr3(test_addr)
    config = make_config(row, col, bank, tras=8)

    # Setup
    wb.write(PW_CONFIG, config)
    wb.write(PW_WRITE_DATA, 0x55555555)
    wb.write(PW_REF_DATA, 0x00000000)
    wb.write(PW_CONTROL, 0x00)

    edge_before = wb.read(PW_EDGE_COUNT)
    print(f"  Edge count before: {edge_before}")

    # Trigger WITHOUT ext_dfi (bit 1 = 0)
    wb.write(PW_CONTROL, 0x01)  # Only start, no ext_dfi
    time.sleep(0.01)

    for i in range(5):
        status = wb.read(PW_STATUS)
        debug = wb.read(PW_DEBUG)
        print(f"    Poll {i}: status=0x{status:02x} debug_state={debug}")
        time.sleep(0.005)

    wb.write(PW_CONTROL, 0x00)
    time.sleep(0.01)

    edge_after = wb.read(PW_EDGE_COUNT)
    print(f"  Edge count after: {edge_after}")
    print_fsm_state(wb, "Final")


def test_direct_memory_sweep(wb):
    """Test different memory locations to rule out addressing issues."""
    print("\n=== Test 4: Memory Address Sweep ===")

    addresses = [
        DRAM_BASE + 0x0,
        DRAM_BASE + 0x1000,
        DRAM_BASE + 0x10000,
        DRAM_BASE + 0x100000,
        DRAM_BASE + 0x800000,
    ]

    for addr in addresses:
        pattern = addr & 0xFFFFFFFF
        wb.write(addr, pattern)
        time.sleep(0.005)
        readback = wb.read(addr)
        match = "OK" if readback == pattern else "FAIL"
        print(f"  0x{addr:08x}: write=0x{pattern:08x} read=0x{readback:08x} [{match}]")


def main():
    print("=" * 60)
    print("z1211: FSM Diagnostic")
    print("=" * 60)

    wb = RemoteClient(host='localhost', port=1234)
    wb.open()

    print(f"\nConnected! Identifier: 0x{wb.read(IDENTIFIER):08x}")

    # LED indication
    wb.write(LEDS_BASE, 0x01)

    results = {}

    # Run tests
    results["dfii_sw_mode"] = test_dfii_software_mode(wb)
    results["fsm_trigger"] = test_fsm_trigger(wb)
    test_fsm_without_ext_dfi(wb)
    test_direct_memory_sweep(wb)

    # Summary
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  DFII software mode works: {results['dfii_sw_mode']}")
    print(f"  FSM trigger affects memory: {results['fsm_trigger']}")

    # LED indication
    if results["fsm_trigger"]:
        wb.write(LEDS_BASE, 0x0F)
    else:
        wb.write(LEDS_BASE, 0x05)

    # Save
    output = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "experiment": "z1211 FSM Diagnostic",
        "results": results
    }
    with open("results/z1211_fsm_diagnostic.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to results/z1211_fsm_diagnostic.json")
    wb.close()


if __name__ == "__main__":
    main()
