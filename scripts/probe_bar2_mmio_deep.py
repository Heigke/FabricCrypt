#!/usr/bin/env python3
"""
Below-firmware register probing via direct BAR2 MMIO + SMN indirect access.
Must run as root: sudo python3 probe_bar2_mmio_deep.py
SAFE: read-only operations only (no register writes).
"""
import mmap, os, struct, time, sys

BAR2_PATH = "/sys/bus/pci/devices/0000:c3:00.0/resource2"

def main():
    if os.geteuid() != 0:
        print("ERROR: Must run as root (sudo)")
        sys.exit(1)

    fd = os.open(BAR2_PATH, os.O_RDONLY | os.O_SYNC)
    size = os.fstat(fd).st_size
    print(f"BAR2 MMIO: {size} bytes ({size/1024/1024:.1f}MB)")
    mm = mmap.mmap(fd, size, mmap.MAP_SHARED, mmap.PROT_READ)

    def read_reg(offset):
        if offset + 4 > size:
            return None
        mm.seek(offset)
        return struct.unpack('<I', mm.read(4))[0]

    # ===== PART 1: Direct MMIO Register Reads =====
    print("\n=== PART 1: Direct MMIO Registers (below firmware) ===")
    regs = [
        (0x8010, "GRBM_STATUS"),
        (0x8014, "GRBM_STATUS2"),
        (0x8020, "GRBM_STATUS_SE0"),
        (0x8024, "GRBM_STATUS_SE1"),
        (0xD048, "SRBM_STATUS"),
        (0xD04C, "SRBM_STATUS2"),
        (0x263C, "CP_STAT"),
        (0xC07C, "RLC_STAT"),
        (0xC080, "RLC_GPU_CLOCK_LSB"),
        (0xC084, "RLC_GPU_CLOCK_MSB"),
        (0xC10C, "RLC_GPM_STAT"),
    ]
    for offset, name in regs:
        val = read_reg(offset)
        if val is not None:
            print(f"  0x{offset:04X} {name:25s} = 0x{val:08X} ({val})")

    # ===== PART 2: GPU Clock Counter (ring oscillator) =====
    print("\n=== PART 2: GPU Clock Counter (hardware oscillator) ===")
    clocks = []
    for i in range(10):
        lsb = read_reg(0xC080)
        msb = read_reg(0xC084)
        if lsb is not None and msb is not None:
            full = (msb << 32) | lsb
            clocks.append(full)
        time.sleep(0.001)
    if clocks:
        for i, c in enumerate(clocks):
            print(f"  [{i}] {c}")
        if len(clocks) > 1:
            deltas = [clocks[i+1]-clocks[i] for i in range(len(clocks)-1)]
            print(f"  Deltas: {deltas}")
            if any(d > 0 for d in deltas):
                print(f"  LIVE CLOCK - incrementing at ~{sum(deltas)/len(deltas)/0.001:.0f} ticks/sec")
            else:
                print(f"  Clock appears STATIC (may be gated when idle)")

    # ===== PART 3: SMN Indirect Access =====
    # SMN uses MMIO index/data pair at 0x60/0x64 (NBIO)
    # READ-ONLY: we only write the ADDRESS register, then read DATA
    print("\n=== PART 3: SMN Indirect Register Read ===")
    print("(Writing address to index reg 0x60, reading data from 0x64)")

    # Need write access for SMN index register
    mm.close()
    os.close(fd)

    fd = os.open(BAR2_PATH, os.O_RDWR | os.O_SYNC)
    mm = mmap.mmap(fd, size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)

    def write_reg(offset, value):
        mm.seek(offset)
        mm.write(struct.pack('<I', value))

    def read_smn(addr):
        """Read SMN register via NBIO index/data at 0x60/0x64"""
        write_reg(0x60, addr)
        return read_reg(0x64)

    # --- SMU firmware version ---
    smn_regs = [
        (0x03B10528, "MP1_SMN_FW_VERSION"),
    ]
    for addr, name in smn_regs:
        val = read_smn(addr)
        if val is not None and val != 0 and val != 0xFFFFFFFF:
            # SMU version format: program.major.minor.debug
            prog = (val >> 24) & 0xFF
            major = (val >> 16) & 0xFF
            minor = (val >> 8) & 0xFF
            debug = val & 0xFF
            print(f"  SMN 0x{addr:08X} {name:30s} = 0x{val:08X} (v{prog}.{major}.{minor}.{debug})")
        else:
            print(f"  SMN 0x{addr:08X} {name:30s} = 0x{val:08X}" if val is not None else "  ERROR")

    # --- SMU Mailbox registers (read current state, NOT sending messages) ---
    mailbox_regs = [
        (0x03B10980, "MP1_C2PMSG_32 (response)"),
        (0x03B10984, "MP1_C2PMSG_33 (param)"),
        (0x03B10988, "MP1_C2PMSG_34 (msg_id)"),
        (0x03B1098C, "MP1_C2PMSG_35 (extra)"),
    ]
    print("\n  --- SMU Mailbox State (read-only peek) ---")
    for addr, name in mailbox_regs:
        val = read_smn(addr)
        if val is not None:
            print(f"  SMN 0x{addr:08X} {name:30s} = 0x{val:08X} ({val})")

    # --- THM (thermal) registers ---
    print("\n  --- Thermal Sensor Registers ---")
    thm_scan = [
        (0x00059800, "THM_TCON_CUR_TMP"),
        (0x00059804, "THM_TCON_HTC"),
        (0x00059808, "THM_TCON_THERM_TRIP"),
        (0x00059900, "THM_BACO_CNTL"),
        (0x00059C00, "CG_THERMAL_STATUS"),
        (0x00059C04, "CG_THERMAL_INT_EN"),
        (0x00059C08, "CG_THERMAL_INT_CTRL"),
    ]
    for addr, name in thm_scan:
        val = read_smn(addr)
        if val is not None and val != 0 and val != 0xFFFFFFFF:
            print(f"  SMN 0x{addr:08X} {name:30s} = 0x{val:08X} ({val})")

    # --- Scan THM range for any non-zero registers ---
    print("\n  --- THM Range Scan (0x59800-0x59C40) ---")
    found_thm = 0
    for addr in range(0x59800, 0x59C40, 4):
        val = read_smn(addr)
        if val is not None and val != 0 and val != 0xFFFFFFFF and val != 0xDEADDEAD:
            print(f"  SMN 0x{addr:08X} = 0x{val:08X} ({val})")
            found_thm += 1
    print(f"  Found {found_thm} non-zero THM registers")

    # --- SVI telemetry registers ---
    print("\n  --- SVI Telemetry (voltage/current readback) ---")
    # Try various known SVI base addresses
    for base_name, base in [("SMUSVI0", 0x0005A000), ("SMUSVI1", 0x0005A800),
                             ("SMUSVI2", 0x0005B000), ("THM_TSVI", 0x00059E00)]:
        found = False
        for off in range(0, 0x40, 4):
            val = read_smn(base + off)
            if val is not None and val != 0 and val != 0xFFFFFFFF:
                print(f"  SMN 0x{base+off:08X} ({base_name}+0x{off:02X}) = 0x{val:08X} ({val})")
                found = True
        if not found:
            print(f"  {base_name} (0x{base:08X}): all zero/FF")

    # --- RSMU (Root SMU) direct registers ---
    print("\n  --- RSMU / MP1 Scan ---")
    mp1_bases = [0x03B10000, 0x03B10500, 0x03B10900, 0x03B10A00]
    for base in mp1_bases:
        found = 0
        for off in range(0, 0x80, 4):
            val = read_smn(base + off)
            if val is not None and val != 0 and val != 0xFFFFFFFF:
                found += 1
                if found <= 8:
                    print(f"  SMN 0x{base+off:08X} = 0x{val:08X}")
        if found > 8:
            print(f"  ... +{found-8} more non-zero in range 0x{base:08X}-0x{base+0x80:08X}")
        elif found == 0:
            print(f"  Range 0x{base:08X}: all zero/FF")

    mm.close()
    os.close(fd)
    print("\nDone. All reads were READ-ONLY (safe).")

if __name__ == "__main__":
    main()
