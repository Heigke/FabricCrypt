#!/usr/bin/env python3
"""Direct BAR2 MMIO register probing — BELOW firmware access.
Reads GPU registers directly via PCI BAR2 mmap, bypassing driver/SMU/PSP.
"""
import mmap, os, struct, time, sys

GPU_PCI = "0000:c3:00.0"
BAR2_PATH = f"/sys/bus/pci/devices/{GPU_PCI}/resource2"

def main():
    fd = os.open(BAR2_PATH, os.O_RDWR | os.O_SYNC)
    size = os.fstat(fd).st_size
    print(f"BAR2 MMIO: {BAR2_PATH} ({size} bytes = {size/1024/1024:.1f}MB)")
    mm = mmap.mmap(fd, size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)

    def read_reg(offset):
        if offset >= size:
            return None
        mm.seek(offset)
        return struct.unpack('<I', mm.read(4))[0]

    def write_reg(offset, value):
        if offset >= size:
            return False
        mm.seek(offset)
        mm.write(struct.pack('<I', value))
        return True

    def read_smn(addr):
        """Read SMN register via NBIO index/data pair at 0x60/0x64"""
        write_reg(0x60, addr)
        return read_reg(0x64)

    # ================================================================
    # PART 1: Direct MMIO register reads (GC block)
    # ================================================================
    print("\n" + "="*70)
    print("PART 1: Direct MMIO Registers (below firmware!)")
    print("="*70)

    gc_regs = [
        (0x8010, "GRBM_STATUS"),
        (0x8014, "GRBM_STATUS2"),
        (0x8018, "GRBM_STATUS3"),
        (0x8020, "GRBM_STATUS_SE0"),
        (0x8024, "GRBM_STATUS_SE1"),
        (0xD048, "SRBM_STATUS"),
        (0xD04C, "SRBM_STATUS2"),
        (0x263C, "CP_STAT"),
        (0xC060, "RLC_CNTL"),
        (0xC07C, "RLC_STAT"),
        (0xC080, "RLC_GPU_CLOCK_COUNT_LSB"),
        (0xC084, "RLC_GPU_CLOCK_COUNT_MSB"),
        (0xC10C, "RLC_GPM_STAT"),
    ]
    for off, name in gc_regs:
        val = read_reg(off)
        if val is not None:
            print(f"  0x{off:04X} {name:35s} = 0x{val:08X} ({val})")

    # ================================================================
    # PART 2: GPU Clock Counter — ring oscillator / PLL output
    # ================================================================
    print("\n" + "="*70)
    print("PART 2: GPU Clock Counter (analog oscillator)")
    print("="*70)
    samples = []
    for i in range(10):
        lsb = read_reg(0xC080)
        msb = read_reg(0xC084)
        if lsb is not None and msb is not None:
            full = (msb << 32) | lsb
            samples.append(full)
            print(f"  [{i}] clock = {full}")
        time.sleep(0.005)
    if len(samples) >= 2:
        deltas = [samples[i+1] - samples[i] for i in range(len(samples)-1)]
        avg_delta = sum(deltas) / len(deltas)
        print(f"  Avg delta per 5ms: {avg_delta:.0f} cycles")
        print(f"  Implied freq: {avg_delta / 5e-3 / 1e6:.1f} MHz")

    # ================================================================
    # PART 3: SMN indirect access — thermal, SVI, MP1 (SMU)
    # ================================================================
    print("\n" + "="*70)
    print("PART 3: SMN Indirect Register Access (below firmware!)")
    print("="*70)

    # 3a: Try to read SMU firmware version
    print("\n--- SMU Firmware Version ---")
    smn_addrs = [
        (0x03B10528, "MP1_SMN_FW_VERSION"),
        (0x03B1052C, "MP1_SMN_FW_VERSION+4"),
    ]
    for addr, name in smn_addrs:
        val = read_smn(addr)
        if val is not None:
            print(f"  SMN 0x{addr:08X} {name:30s} = 0x{val:08X}")

    # 3b: SMU mailbox registers (for sending raw messages)
    print("\n--- SMU Mailbox Registers ---")
    # Note: register offsets may differ per ASIC. Try multiple known offsets.
    mailbox_tries = [
        # Standard MP1 mailbox (v14)
        (0x03B10980, "MP1_C2PMSG_32 (response)"),
        (0x03B10984, "MP1_C2PMSG_33 (param)"),
        (0x03B10988, "MP1_C2PMSG_34 (msg_id)"),
        (0x03B1098C, "MP1_C2PMSG_35"),
        # Debug mailbox
        (0x03B10A4C, "MP1_C2PMSG_90 (debug_resp)"),
        (0x03B10A50, "MP1_C2PMSG_91 (debug_param)"),
        (0x03B10A54, "MP1_C2PMSG_92 (debug_msg)"),
        # Alt offsets (some ASICs use different base)
        (0x03B00080, "MP1_SMN_C2PMSG_32_alt"),
        (0x03B00084, "MP1_SMN_C2PMSG_33_alt"),
        (0x03B00088, "MP1_SMN_C2PMSG_34_alt"),
    ]
    for addr, name in mailbox_tries:
        val = read_smn(addr)
        if val is not None and val != 0xFFFFFFFF:
            print(f"  SMN 0x{addr:08X} {name:40s} = 0x{val:08X}")

    # 3c: Thermal registers
    print("\n--- Thermal Sensor Registers (direct silicon read!) ---")
    # THM block on GFX11 — try multiple base addresses
    thm_bases = [0x00059800, 0x00059C00, 0x00060000, 0x0005AC00]
    for base in thm_bases:
        for off in range(0, 0x40, 4):
            addr = base + off
            val = read_smn(addr)
            if val is not None and val != 0 and val != 0xFFFFFFFF and val != 0xDEADDEAD:
                # Check if it looks like a temperature (range 20-120°C)
                temp_c = (val >> 10) & 0x3FF  # Common AMD temp encoding
                temp_raw = val & 0xFFF
                print(f"  SMN 0x{addr:08X} = 0x{val:08X} (raw={val}, shifted_temp={temp_c}, raw12={temp_raw})")

    # 3d: SVI (Serial VID Interface) telemetry — voltage/current from VRM
    print("\n--- SVI Telemetry Registers (VRM ADC — analog!) ---")
    svi_bases = [0x0005A000, 0x0005A800, 0x00060800, 0x0006A000]
    for base in svi_bases:
        for off in range(0, 0x20, 4):
            addr = base + off
            val = read_smn(addr)
            if val is not None and val != 0 and val != 0xFFFFFFFF:
                print(f"  SMN 0x{addr:08X} = 0x{val:08X} ({val})")

    # 3e: MP0 (PSP) registers — see if we can read PSP state
    print("\n--- PSP (MP0) Registers ---")
    mp0_regs = [
        (0x03800004, "MP0_SMN_C2PMSG_0"),
        (0x03800008, "MP0_SMN_C2PMSG_1"),
        (0x0380000C, "MP0_SMN_C2PMSG_2"),
        (0x03800010, "MP0_SMN_C2PMSG_3"),
        (0x03800074, "MP0_SMN_C2PMSG_28"),
        (0x03800078, "MP0_SMN_C2PMSG_29"),
        (0x0380007C, "MP0_SMN_C2PMSG_30"),
        (0x03800080, "MP0_SMN_C2PMSG_31"),
    ]
    for addr, name in mp0_regs:
        val = read_smn(addr)
        if val is not None:
            print(f"  SMN 0x{addr:08X} {name:30s} = 0x{val:08X}")

    # ================================================================
    # PART 4: Scan for interesting non-zero SMN ranges
    # ================================================================
    print("\n" + "="*70)
    print("PART 4: SMN Range Scan (hunting for analog sensors)")
    print("="*70)

    scan_ranges = [
        (0x00059800, 0x00059A00, "THM block"),
        (0x00059C00, 0x00059D00, "CG_THERMAL"),
        (0x0005A000, 0x0005A100, "SVI telemetry"),
        (0x0005AC00, 0x0005AD00, "SMUIO"),
        (0x03B10500, 0x03B10600, "MP1_FW"),
        (0x03B10900, 0x03B10B00, "MP1_MAILBOX"),
    ]
    for start, end, name in scan_ranges:
        found = []
        for addr in range(start, end, 4):
            val = read_smn(addr)
            if val is not None and val != 0 and val != 0xFFFFFFFF:
                found.append((addr, val))
        if found:
            print(f"\n  --- {name} (0x{start:08X}-0x{end:08X}): {len(found)} non-zero ---")
            for addr, val in found[:20]:
                print(f"    0x{addr:08X} = 0x{val:08X} ({val})")
            if len(found) > 20:
                print(f"    ... and {len(found)-20} more")

    # ================================================================
    # PART 5: Rapid sampling of changing registers (find analog signals)
    # ================================================================
    print("\n" + "="*70)
    print("PART 5: Rapid Sampling — Finding Analog Signals")
    print("="*70)

    # Collect registers that change rapidly = analog sensors
    candidate_addrs = []
    # Add all non-zero SMN addresses found above
    for start, end, name in scan_ranges:
        for addr in range(start, end, 4):
            val = read_smn(addr)
            if val is not None and val != 0 and val != 0xFFFFFFFF:
                candidate_addrs.append(("SMN", addr, name))

    # Also add direct MMIO candidates
    for off, name in gc_regs:
        val = read_reg(off)
        if val is not None and val != 0:
            candidate_addrs.append(("MMIO", off, name))

    # Sample each 50 times and find changing ones
    print(f"\n  Sampling {len(candidate_addrs)} candidates 50 times each...")
    changing = []
    for bus_type, addr, name in candidate_addrs:
        vals = set()
        for _ in range(50):
            if bus_type == "SMN":
                v = read_smn(addr)
            else:
                v = read_reg(addr)
            if v is not None:
                vals.add(v)
        if len(vals) > 3:
            changing.append((bus_type, addr, name, len(vals), min(vals), max(vals)))

    if changing:
        print(f"\n  ANALOG SIGNALS FOUND ({len(changing)} changing registers):")
        for bus, addr, name, unique, mn, mx in sorted(changing, key=lambda x: -x[3]):
            print(f"    {bus} 0x{addr:08X} [{name:20s}] {unique:4d} unique, range [{mn:10d}-{mx:10d}]")
    else:
        print("  No rapidly-changing registers found (try under GPU load)")

    mm.close()
    os.close(fd)
    print("\n\nDONE — Direct below-firmware register access confirmed!")

if __name__ == "__main__":
    main()
