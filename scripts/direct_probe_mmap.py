#!/usr/bin/env python3
"""
direct_probe_mmap.py — Read GPU registers via /dev/mem after fw_load_type=0 failure
Maps MMIO BAR5 directly from userspace.
"""
import mmap
import struct
import os
import sys

BAR5_PHYS = 0xB4400000
BAR5_SIZE = 0x100000  # 1MB

def rreg(mm, reg_idx):
    """Read 32-bit register at 4-byte index"""
    offset = reg_idx * 4
    if offset + 4 > BAR5_SIZE:
        return 0xDEADDEAD
    mm.seek(offset)
    return struct.unpack('<I', mm.read(4))[0]

def wreg(mm, reg_idx, val):
    """Write 32-bit register at 4-byte index"""
    offset = reg_idx * 4
    mm.seek(offset)
    mm.write(struct.pack('<I', val))

def main():
    fd = os.open('/dev/mem', os.O_RDWR | os.O_SYNC)
    mm = mmap.mmap(fd, BAR5_SIZE, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, offset=BAR5_PHYS)

    print("=== GFX REGISTER STATE AFTER fw_load_type=0 ===")
    print()

    # GRBM status
    regs = {
        'GRBM_STATUS':      0x0504,
        'GRBM_STATUS2':     0x0502,
        'GRBM_STATUS3':     0x0503,
        'CP_STAT':          0x0E40,
        'CP_MEC_CNTL':      0x0E58,
        'CP_CPC_STATUS':    0x0E54,
        'CP_CPF_STATUS':    0x0E53,
        'RLC_CNTL':         0x4C00,
        'RLC_STAT':         0x4C04,
        'RLC_GPM_STAT':     0x4C01,
        'RLC_SAFE_MODE':    0x4C20,
        'CP_MES_CNTL':      0x28C0,
        'MES_PRGRM_START':  0x28C5,
        'MES_INSTR_PNTR':   0x28E1,
        'MES_IC_BASE_LO':   0x28C1,
        'MES_IC_BASE_HI':   0x28C2,
        'MES_DOORBELL_CTL': 0x28D2,
        'MES_GP0_LO':       0x28D9,
        'MES_GP0_HI':       0x28DA,
        'GFX_RS64_DC_BASE_LO': 0x2D14,
        'GFX_RS64_DC_BASE_HI': 0x2D15,
        'MEC_RS64_DC_BASE_LO': 0x2D25,
        'MEC_RS64_DC_BASE_HI': 0x2D26,
        'MEC_UCODE_ADDR':   0x2D1C,
        'MEC_UCODE_DATA':   0x2D1D,
        'SDMA0_STATUS':     0x0D85,
    }

    for name, idx in regs.items():
        val = rreg(mm, idx)
        print(f"  {name:30s} [0x{idx:04x}] = 0x{val:08x}")

    print()
    print("=== MES REGISTER RANGE SCAN (0x28C0-0x28FF) ===")
    for i in range(0x28C0, 0x2900):
        val = rreg(mm, i)
        if val != 0 and val != 0xFFFFFFFF:
            print(f"  reg[0x{i:04x}] = 0x{val:08x}")

    print()
    print("=== RLC REGISTER RANGE (0x4C00-0x4C50) ===")
    for i in range(0x4C00, 0x4C50):
        val = rreg(mm, i)
        if val != 0 and val != 0xFFFFFFFF:
            print(f"  reg[0x{i:04x}] = 0x{val:08x}")

    print()
    print("=== CP/GFX REGISTER RANGE (0x2D00-0x2D40) ===")
    for i in range(0x2D00, 0x2D40):
        val = rreg(mm, i)
        if val != 0 and val != 0xFFFFFFFF:
            print(f"  reg[0x{i:04x}] = 0x{val:08x}")

    print()
    print("=== WRITE TEST: MES_IC_BASE_LO ===")
    before = rreg(mm, 0x28C1)
    print(f"  BEFORE: 0x{before:08x}")
    wreg(mm, 0x28C1, 0xDEADBEEF)
    after = rreg(mm, 0x28C1)
    print(f"  AFTER:  0x{after:08x}")
    if after == 0xDEADBEEF:
        print("  *** MES_IC_BASE IS WRITABLE! ***")
        # Restore
        wreg(mm, 0x28C1, before)
    else:
        print(f"  IC_BASE locked (wrote 0xDEADBEEF, read 0x{after:08x})")

    print()
    print("=== WRITE TEST: RLC_CNTL ===")
    rlc = rreg(mm, 0x4C00)
    print(f"  RLC_CNTL BEFORE: 0x{rlc:08x}")
    wreg(mm, 0x4C00, rlc | 0x1)  # Try setting RLC_ENABLE
    import time; time.sleep(0.01)
    rlc2 = rreg(mm, 0x4C00)
    print(f"  RLC_CNTL AFTER:  0x{rlc2:08x}")

    print()
    print("=== WRITE TEST: CP_MEC_CNTL (halt bits) ===")
    mec = rreg(mm, 0x0E58)
    print(f"  CP_MEC_CNTL BEFORE: 0x{mec:08x}")
    # Try clearing halt bits (bits 28:30 = MEC_ME1_HALT, MEC_ME2_HALT, MEC_ME1_STEP)
    wreg(mm, 0x0E58, mec & ~0x70000000)
    import time; time.sleep(0.01)
    mec2 = rreg(mm, 0x0E58)
    print(f"  CP_MEC_CNTL AFTER:  0x{mec2:08x}")

    # Check key GRBM register for GFX state
    print()
    print("=== GRBM GFX PIPE SELECT TEST ===")
    for pipe in range(4):
        wreg(mm, 0x0640, pipe)  # GRBM_GFX_CNTL = select pipe
        import time; time.sleep(0.001)
        mes = rreg(mm, 0x28C0)
        ic_lo = rreg(mm, 0x28C1)
        ic_hi = rreg(mm, 0x28C2)
        print(f"  PIPE{pipe}: MES_CNTL=0x{mes:08x} IC_BASE=0x{ic_hi:08x}_{ic_lo:08x}")
    wreg(mm, 0x0640, 0)  # Reset

    mm.close()
    os.close(fd)

if __name__ == '__main__':
    main()
