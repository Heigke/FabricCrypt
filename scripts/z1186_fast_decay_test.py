#!/usr/bin/env python3
"""
z1186: Fast Decay Test - Measure DRAM decay timing

Since refresh is disabled, we need to understand decay timing
to properly implement partial writes.
"""

import time
from litex import RemoteClient

DRAM_BASE = 0x40000000


def main():
    print("z1186: Fast Decay Test")
    print("="*60)

    wb = RemoteClient()
    wb.open()

    # Check DFII in HW mode
    dfii_ctrl = wb.read(0x2800)
    print(f"DFII control: 0x{dfii_ctrl:02x}")
    if dfii_ctrl != 0x0b:
        wb.write(0x2800, 0x0b)
        time.sleep(0.01)

    # Test different wait times to find decay threshold
    print("\n--- Decay timing test ---")

    test_addr = DRAM_BASE + 0x80000
    test_val = 0xFFFFFFFF
    flush_addr = DRAM_BASE + 0x200000

    wait_times_ms = [10, 20, 50, 100, 200, 500, 1000]

    for wait_ms in wait_times_ms:
        # Write pattern
        wb.write(test_addr, test_val)
        time.sleep(0.001)

        # Quick cache flush (minimal)
        for i in range(128):  # 512 bytes - small flush
            _ = wb.read(flush_addr + i*4)

        # Wait
        time.sleep(wait_ms / 1000.0)

        # Read back
        val = wb.read(test_addr)
        ones = bin(val).count('1')

        print(f"  Wait {wait_ms:4d}ms: 0x{val:08x} ({ones:2d}/32 ones)")

    # Now test with NO wait - just flush and read immediately
    print("\n--- Minimal latency test ---")

    for flush_words in [64, 128, 256, 512, 1024, 2048]:
        # Write
        test_addr2 = DRAM_BASE + 0x90000
        test_val2 = 0xFFFFFFFF
        wb.write(test_addr2, test_val2)
        time.sleep(0.001)

        # Flush with varying amounts
        flush_addr2 = DRAM_BASE + 0x300000
        start = time.time()
        for i in range(flush_words):
            _ = wb.read(flush_addr2 + i*4)
        flush_time_ms = (time.time() - start) * 1000

        # Immediate read
        val2 = wb.read(test_addr2)
        ones2 = bin(val2).count('1')

        print(f"  Flush {flush_words:4d} words ({flush_time_ms:.0f}ms): 0x{val2:08x} ({ones2:2d}/32 ones)")

    # Test actual round-trip time
    print("\n--- Etherbone latency measurement ---")
    start = time.time()
    for i in range(100):
        _ = wb.read(DRAM_BASE)
    elapsed = (time.time() - start) * 1000
    print(f"  100 reads: {elapsed:.1f}ms total, {elapsed/100:.1f}ms per read")

    wb.close()


if __name__ == "__main__":
    main()
