#!/usr/bin/env python3
"""
Safe READ-ONLY probe of GPU MMIO registers via PCI BAR2.
This bypasses firmware — reads silicon registers directly.
NO WRITES — read only. Safe to run.
"""
import mmap, os, struct, time, sys

BAR2_PATH = "/sys/bus/pci/devices/0000:c3:00.0/resource2"

def main():
    if not os.path.exists(BAR2_PATH):
        print(f"ERROR: {BAR2_PATH} not found")
        sys.exit(1)

    fd = os.open(BAR2_PATH, os.O_RDONLY | os.O_SYNC)
    size = os.fstat(fd).st_size
    print(f"BAR2 MMIO: {size} bytes ({size/1024/1024:.1f}MB)")

    mm = mmap.mmap(fd, size, mmap.MAP_SHARED, mmap.PROT_READ)

    def read_reg(offset):
        if offset >= size:
            return None
        mm.seek(offset)
        return struct.unpack('<I', mm.read(4))[0]

    # === Direct MMIO registers (GC block, accessible via BAR2) ===
    print("\n=== GRBM Status (per-block activity) ===")
    regs = [
        (0x8010, "GRBM_STATUS"),
        (0x8014, "GRBM_STATUS2"),
        (0x8020, "GRBM_STATUS_SE0"),
        (0x8024, "GRBM_STATUS_SE1"),
    ]
    for off, name in regs:
        v = read_reg(off)
        if v is not None:
            print(f"  {name} (0x{off:04X}) = 0x{v:08X}")

    print("\n=== RLC (RunList Controller) ===")
    regs = [
        (0xC060, "RLC_CNTL"),
        (0xC07C, "RLC_STAT"),
        (0xC080, "RLC_GPU_CLOCK_COUNT_LSB"),
        (0xC084, "RLC_GPU_CLOCK_COUNT_MSB"),
        (0xC10C, "RLC_GPM_STAT"),
    ]
    for off, name in regs:
        v = read_reg(off)
        if v is not None:
            print(f"  {name} (0x{off:04X}) = 0x{v:08X}")

    # Clock counter — analog oscillator signal!
    print("\n=== GPU Clock Counter (ring oscillator, 10 samples) ===")
    counts = []
    for i in range(10):
        lsb = read_reg(0xC080)
        msb = read_reg(0xC084)
        if lsb is not None and msb is not None:
            full = (msb << 32) | lsb
            counts.append(full)
            print(f"  [{i}] = {full}")
        time.sleep(0.01)
    if len(counts) >= 2:
        deltas = [counts[i+1]-counts[i] for i in range(len(counts)-1)]
        avg_delta = sum(deltas)/len(deltas)
        freq_mhz = avg_delta / 10000  # delta per 10ms
        print(f"  avg_delta = {avg_delta:.0f} clocks/10ms ≈ {freq_mhz:.0f} MHz")

    print("\n=== Command Processor ===")
    regs = [
        (0x263C, "CP_STAT"),
        (0x8670, "CP_STALLED_STAT1"),
        (0x8674, "CP_STALLED_STAT2"),
        (0x8678, "CP_BUSY_STAT"),
    ]
    for off, name in regs:
        v = read_reg(off)
        if v is not None and v != 0:
            print(f"  {name} (0x{off:04X}) = 0x{v:08X}")

    print("\n=== System/Bus Status ===")
    regs = [
        (0xD048, "SRBM_STATUS"),
        (0xD04C, "SRBM_STATUS2"),
    ]
    for off, name in regs:
        v = read_reg(off)
        if v is not None:
            print(f"  {name} (0x{off:04X}) = 0x{v:08X}")

    # === SMN indirect access (index 0x60, data 0x64) ===
    # NOTE: This requires WRITE to index register — skip for safety
    # We can only do it if we open read-write

    print("\n=== Scanning for non-zero registers in key ranges ===")
    ranges = [
        (0x8000, 0x8100, "GRBM range"),
        (0xC000, 0xC200, "RLC range"),
        (0xD000, 0xD100, "SRBM range"),
    ]
    for start, end, name in ranges:
        nonzero = []
        for off in range(start, min(end, size), 4):
            v = read_reg(off)
            if v is not None and v != 0 and v != 0xFFFFFFFF:
                nonzero.append((off, v))
        if nonzero:
            print(f"  {name}: {len(nonzero)} non-zero registers")
            for off, v in nonzero[:10]:
                print(f"    0x{off:04X} = 0x{v:08X}")
            if len(nonzero) > 10:
                print(f"    ... ({len(nonzero)-10} more)")

    mm.close()
    os.close(fd)
    print("\n=== DONE — read-only probe complete ===")

if __name__ == "__main__":
    main()
