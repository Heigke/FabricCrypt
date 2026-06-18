#!/usr/bin/env python3
"""
Direct GPU register probing BELOW firmware level via BAR2 MMIO + SMN indirect access.
Reads raw silicon-level analog sensors bypassing PSP/SMU firmware.
Requires root access.
"""
import mmap, os, struct, time, sys, json

GPU_PCI = "0000:c3:00.0"
BAR2_PATH = f"/sys/bus/pci/devices/{GPU_PCI}/resource2"

def open_bar2():
    """Open BAR2 MMIO for register access"""
    fd = os.open(BAR2_PATH, os.O_RDWR | os.O_SYNC)
    size = os.fstat(fd).st_size
    mm = mmap.mmap(fd, size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
    return fd, mm, size

def read_mmio(mm, offset):
    mm.seek(offset)
    return struct.unpack('<I', mm.read(4))[0]

def write_mmio(mm, offset, value):
    mm.seek(offset)
    mm.write(struct.pack('<I', value))

def read_smn(mm, addr):
    """Read SMN (System Management Network) register via MMIO index/data pair at 0x60/0x64"""
    write_mmio(mm, 0x60, addr)
    return read_mmio(mm, 0x64)

def main():
    print("=" * 70)
    print("BELOW-FIRMWARE GPU REGISTER PROBE")
    print("Direct MMIO BAR2 + SMN Indirect Access")
    print("=" * 70)

    fd, mm, size = open_bar2()
    print(f"\nBAR2 opened: {BAR2_PATH} ({size} bytes = {size/1024/1024:.1f}MB)")
    results = {}

    # ============================================================
    # SECTION 1: Direct MMIO Registers
    # ============================================================
    print("\n--- SECTION 1: Direct MMIO Registers ---")
    mmio_regs = [
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
    for offset, name in mmio_regs:
        try:
            val = read_mmio(mm, offset)
            print(f"  0x{offset:04X} {name:35s} = 0x{val:08X}")
            results[name] = val
        except Exception as e:
            print(f"  0x{offset:04X} {name:35s} = ERROR: {e}")

    # ============================================================
    # SECTION 2: GPU Clock Counter (Ring Oscillator / PLL)
    # ============================================================
    print("\n--- SECTION 2: GPU Clock Counter (Analog Oscillator) ---")
    clk_samples = []
    for i in range(10):
        lsb = read_mmio(mm, 0xC080)
        msb = read_mmio(mm, 0xC084)
        full = (msb << 32) | lsb
        clk_samples.append(full)
        if i < 5:
            print(f"  Sample {i}: {full:20d} (0x{full:016X})")
        time.sleep(0.001)
    if len(clk_samples) > 1:
        deltas = [clk_samples[i+1] - clk_samples[i] for i in range(len(clk_samples)-1)]
        avg_delta = sum(deltas) / len(deltas)
        print(f"  Avg delta/ms: {avg_delta:.0f} clocks (≈ {avg_delta/1e3:.1f} MHz effective)")
        results['gpu_clock_rate_mhz'] = avg_delta / 1e3

    # ============================================================
    # SECTION 3: SMN Indirect Access - Thermal Sensors
    # ============================================================
    print("\n--- SECTION 3: SMN Thermal Sensors (Below Firmware) ---")
    # THM block addresses (vary by ASIC, scan a range)
    thm_bases = [0x00059800, 0x0005A800, 0x00059C00]
    for base in thm_bases:
        print(f"  Scanning THM range 0x{base:08X}-0x{base+0x100:08X}:")
        for offset in range(0, 0x100, 4):
            addr = base + offset
            try:
                val = read_smn(mm, addr)
                if val != 0 and val != 0xFFFFFFFF and val != 0xDEADDEAD:
                    # Try to interpret as temperature (various encodings)
                    temp_raw = (val >> 21) & 0x7FF  # bits [31:21] common temp encoding
                    temp_c = temp_raw / 8.0  # 3-bit fractional
                    print(f"    0x{addr:08X} = 0x{val:08X} (temp_decode: {temp_c:.1f}°C, raw: {val})")
                    results[f'smn_{addr:08x}'] = val
            except Exception as e:
                pass

    # ============================================================
    # SECTION 4: SMN - SVI2/SVI3 Telemetry (Voltage/Current)
    # ============================================================
    print("\n--- SECTION 4: SMN SVI Telemetry (Analog VRM ADC) ---")
    # SVI2/SVI3 telemetry typically in SMU SRAM area or specific THM offsets
    svi_scan_ranges = [
        (0x0005A000, 0x0005A100, "SVI_PLANE"),
        (0x0005C000, 0x0005C100, "SVI_ALT"),
        (0x00059900, 0x00059A00, "THM_SVI"),
        (0x03B10000, 0x03B10100, "MP1_BASE"),
    ]
    for start, end, label in svi_scan_ranges:
        found = 0
        for addr in range(start, end, 4):
            try:
                val = read_smn(mm, addr)
                if val != 0 and val != 0xFFFFFFFF:
                    found += 1
                    if found <= 10:
                        print(f"  [{label}] 0x{addr:08X} = 0x{val:08X} ({val:10d})")
                    results[f'smn_{addr:08x}'] = val
            except:
                pass
        if found > 10:
            print(f"  [{label}] ... {found} non-zero registers total")
        elif found == 0:
            print(f"  [{label}] No non-zero registers found")

    # ============================================================
    # SECTION 5: SMN - MP1 (SMU) Mailbox Registers
    # ============================================================
    print("\n--- SECTION 5: SMU MP1 Mailbox (Message Interface) ---")
    # Standard MP1 C2PMSG offsets for v14
    mp1_base = 0x03B10000
    mp1_regs = [
        (mp1_base + 0x528, "MP1_SMN_FW_VERSION"),
        (mp1_base + 0x980, "MP1_SMN_C2PMSG_32_resp"),
        (mp1_base + 0x984, "MP1_SMN_C2PMSG_33_param"),
        (mp1_base + 0x988, "MP1_SMN_C2PMSG_34_msg"),
        (mp1_base + 0x98C, "MP1_SMN_C2PMSG_35"),
        (mp1_base + 0xA4C, "MP1_SMN_C2PMSG_90_dbg_resp"),
        (mp1_base + 0xA50, "MP1_SMN_C2PMSG_91_dbg_param"),
        (mp1_base + 0xA54, "MP1_SMN_C2PMSG_92_dbg_msg"),
    ]
    for addr, name in mp1_regs:
        try:
            val = read_smn(mm, addr)
            print(f"  0x{addr:08X} {name:35s} = 0x{val:08X}")
            results[name] = val
        except Exception as e:
            print(f"  0x{addr:08X} {name:35s} = ERROR: {e}")

    # ============================================================
    # SECTION 6: SMN Wide Scan for Active Sensors
    # ============================================================
    print("\n--- SECTION 6: SMN Wide Scan (finding active analog registers) ---")
    # Scan key SMN regions for changing values
    scan_regions = [
        (0x00059800, 0x00059C00, "THM"),     # Thermal
        (0x0005A000, 0x0005A200, "SVI"),     # SVI voltage
        (0x0005C000, 0x0005C200, "CG"),      # Clock gating
        (0x03B10500, 0x03B10600, "MP1_FW"),  # SMU FW area
        (0x03B10900, 0x03B10B00, "MP1_MSG"), # SMU messages
    ]

    changing_regs = []
    for start, end, label in scan_regions:
        # Read twice with delay, find changing values
        vals1 = {}
        vals2 = {}
        for addr in range(start, end, 4):
            try:
                vals1[addr] = read_smn(mm, addr)
            except:
                pass
        time.sleep(0.1)
        for addr in range(start, end, 4):
            try:
                vals2[addr] = read_smn(mm, addr)
            except:
                pass

        changed = 0
        for addr in vals1:
            if addr in vals2 and vals1[addr] != vals2[addr]:
                changed += 1
                changing_regs.append((addr, label, vals1[addr], vals2[addr]))
                if changed <= 5:
                    print(f"  [{label}] 0x{addr:08X}: 0x{vals1[addr]:08X} -> 0x{vals2[addr]:08X} (CHANGING!)")
        nonzero = sum(1 for a in vals1 if vals1[a] != 0 and vals1[a] != 0xFFFFFFFF)
        print(f"  [{label}] {nonzero} non-zero, {changed} changing out of {len(vals1)} registers")

    # ============================================================
    # SECTION 7: High-Speed Analog Sampling
    # ============================================================
    print("\n--- SECTION 7: High-Speed Analog Sampling ---")
    if changing_regs:
        print(f"  Found {len(changing_regs)} changing registers, sampling fastest:")
        # Pick up to 5 changing registers for rapid sampling
        targets = changing_regs[:5]
        for addr, label, _, _ in targets:
            samples = []
            t0 = time.time()
            for _ in range(100):
                try:
                    samples.append(read_smn(mm, addr))
                except:
                    pass
            t1 = time.time()
            unique = len(set(samples))
            rate = len(samples) / (t1 - t0) if t1 > t0 else 0
            mn, mx = min(samples), max(samples) if samples else (0, 0)
            print(f"  [{label}] 0x{addr:08X}: {unique} unique in 100 samples, rate={rate:.0f}Hz, range=[{mn}-{mx}]")
    else:
        print("  No changing SMN registers found in scanned ranges")
        # Sample GRBM_STATUS which we know changes
        samples = []
        t0 = time.time()
        for _ in range(1000):
            samples.append(read_mmio(mm, 0x8010))
        t1 = time.time()
        unique = len(set(samples))
        rate = len(samples) / (t1 - t0)
        print(f"  GRBM_STATUS direct MMIO: {unique} unique in 1000 samples, rate={rate:.0f}Hz")

    mm.close()
    os.close(fd)

    # Save results
    out_path = "/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/below_firmware_probe.json"
    # Convert any non-serializable values
    clean = {k: int(v) if isinstance(v, (int,)) else v for k, v in results.items()}
    with open(out_path, 'w') as f:
        json.dump(clean, f, indent=2)
    print(f"\nResults saved to {out_path}")
    print("\n" + "=" * 70)
    print("BELOW-FIRMWARE PROBE COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()
