#!/usr/bin/env python3
"""
z1185: Basic DRAM test - verify L2 cache and DRAM work correctly
WITHOUT any FSM/SW mode interference.
"""

import time
from litex import RemoteClient

DRAM_BASE = 0x40000000


def main():
    print("z1185: Basic DRAM Test (no FSM interference)")
    print("="*60)

    wb = RemoteClient()
    wb.open()

    # Check identity
    ident = ""
    for i in range(256):
        c = wb.read(0x1800 + i*4)
        if c == 0:
            break
        ident += chr(c)
    print(f"SoC Identity: {ident}")

    # Check DFII control - make sure we're in HW mode
    dfii_ctrl = wb.read(0x2800)
    print(f"DFII control: 0x{dfii_ctrl:02x} (should be 0x0b for HW mode)")

    if dfii_ctrl != 0x0b:
        print("Setting DFII to HW mode...")
        wb.write(0x2800, 0x0b)
        time.sleep(0.1)
        dfii_ctrl = wb.read(0x2800)
        print(f"DFII control now: 0x{dfii_ctrl:02x}")

    # Test 1: Simple write/read at base address
    print("\n--- Test 1: Write/Read at DRAM base ---")
    test_addr = DRAM_BASE
    test_val = 0xCAFEBABE

    print(f"  Write 0x{test_val:08x} to 0x{test_addr:08x}")
    wb.write(test_addr, test_val)
    time.sleep(0.01)

    read_val = wb.read(test_addr)
    print(f"  Read back: 0x{read_val:08x}")

    if read_val == test_val:
        print("  PASS: Immediate read matches")
    else:
        print("  FAIL: Mismatch!")

    # Test 2: Write, access different region (flush cache), read back
    print("\n--- Test 2: Write, flush cache, read ---")
    test_addr2 = DRAM_BASE + 0x1000
    test_val2 = 0xDEADBEEF

    print(f"  Write 0x{test_val2:08x} to 0x{test_addr2:08x}")
    wb.write(test_addr2, test_val2)
    time.sleep(0.01)

    # Access a different cache region (L2 cache is 8KB = 0x2000)
    # Access enough to evict our test line
    flush_addr = DRAM_BASE + 0x100000  # Far away
    print(f"  Flushing cache by reading 0x{flush_addr:08x}...")
    for i in range(512):  # 2KB read
        _ = wb.read(flush_addr + i*4)
    time.sleep(0.01)

    # Now read original address - should come from DRAM
    read_val2 = wb.read(test_addr2)
    print(f"  Read back after flush: 0x{read_val2:08x}")

    if read_val2 == test_val2:
        print("  PASS: Data persisted in DRAM")
    else:
        print(f"  FAIL: Expected 0x{test_val2:08x}, got 0x{read_val2:08x}")

    # Test 3: Multiple addresses
    print("\n--- Test 3: Multiple address test ---")
    patterns = [0x11111111, 0x22222222, 0x33333333, 0x44444444]
    addrs = [DRAM_BASE + 0x2000 + i*0x1000 for i in range(4)]

    for addr, pat in zip(addrs, patterns):
        wb.write(addr, pat)
        print(f"  Wrote 0x{pat:08x} to 0x{addr:08x}")
    time.sleep(0.01)

    # Flush with massive read
    print("  Flushing cache...")
    for i in range(1024):
        _ = wb.read(DRAM_BASE + 0x200000 + i*4)
    time.sleep(0.01)

    # Read back
    print("  Reading back after flush:")
    all_match = True
    for addr, pat in zip(addrs, patterns):
        val = wb.read(addr)
        status = "PASS" if val == pat else "FAIL"
        print(f"    0x{addr:08x}: 0x{val:08x} (expected 0x{pat:08x}) - {status}")
        if val != pat:
            all_match = False

    if all_match:
        print("\n  ALL TESTS PASSED - DRAM working correctly")
    else:
        print("\n  SOME TESTS FAILED - DRAM issue detected")

    # Test 4: Check if it's a refresh issue
    print("\n--- Test 4: Refresh disabled check ---")
    print("  Note: This SoC has refresh DISABLED (with_refresh=False)")
    print("  Writing pattern and waiting 100ms...")

    test_addr3 = DRAM_BASE + 0x50000
    test_val3 = 0xAAAA5555
    wb.write(test_addr3, test_val3)
    time.sleep(0.01)

    # Flush immediately
    for i in range(1024):
        _ = wb.read(DRAM_BASE + 0x300000 + i*4)
    time.sleep(0.1)  # Wait 100ms

    # Read back
    val3 = wb.read(test_addr3)
    print(f"  After 100ms: 0x{val3:08x} (expected 0x{test_val3:08x})")

    if val3 == test_val3:
        print("  Data retained after 100ms (no decay yet)")
    else:
        ones_expected = bin(test_val3).count('1')
        ones_actual = bin(val3).count('1')
        print(f"  Data changed! Ones: {ones_actual} (was {ones_expected})")

    wb.close()


if __name__ == "__main__":
    main()
