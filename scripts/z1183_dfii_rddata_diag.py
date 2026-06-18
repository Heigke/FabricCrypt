#!/usr/bin/env python3
"""
z1183: DFII rddata diagnostic

Direct read from DFII phase rddata CSRs to understand data flow.
"""

import time
from litex import RemoteClient

# DFII CSR addresses from csr.h
DFII_CONTROL = 0x2800       # dfii_csrstorage_54 - control register

# Phase 0 CSRs
DFII_PI0_COMMAND = 0x2804   # pi0_csrstorage_55 - command
DFII_PI0_ISSUE = 0x2808     # pi0_csr_56 - command_issue
DFII_PI0_ADDRESS = 0x280c   # pi0_csrstorage_57 - address
DFII_PI0_BADDRESS = 0x2810  # pi0_csrstorage_58 - baddress
DFII_PI0_WRDATA = 0x2814    # pi0_csrstorage_59 - wrdata (only low 32 bits)
DFII_PI0_RDDATA = 0x2818    # pi0_csrstatus_60 - rddata (only low 32 bits)

# Phase 1-3 rddata CSRs
DFII_PI1_RDDATA = 0x2830    # pi1_csrstatus_66
DFII_PI2_RDDATA = 0x2848    # pi2_csrstatus_72
DFII_PI3_RDDATA = 0x2860    # pi3_csrstatus_78

# Commands
CMD_NOP = 0x00
CMD_ACT = 0x09      # cs, ras
CMD_PRE = 0x0B      # cs, ras, we
CMD_READ = 0x25     # cs, cas, rden
CMD_WRITE = 0x1D    # cs, cas, wren

SW_MODE = 0x0A      # Software mode, CKE on
HW_MODE = 0x0B      # Hardware mode, CKE on


def issue_cmd(wb, cmd, addr=0, bank=0, wrdata=None):
    """Issue a single DFII command"""
    wb.write(DFII_PI0_ADDRESS, addr)
    wb.write(DFII_PI0_BADDRESS, bank)
    if wrdata is not None:
        wb.write(DFII_PI0_WRDATA, wrdata)
    wb.write(DFII_PI0_COMMAND, cmd)
    wb.write(DFII_PI0_ISSUE, 1)  # Issue command
    time.sleep(0.001)


def main():
    print("z1183: DFII rddata diagnostic")
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

    # Read initial DFII control
    dfii_ctrl = wb.read(DFII_CONTROL)
    print(f"Initial DFII control: 0x{dfii_ctrl:02x}")

    # Switch to SW mode
    print("\n--- Switching to SW mode ---")
    wb.write(DFII_CONTROL, SW_MODE)
    time.sleep(0.01)
    dfii_ctrl = wb.read(DFII_CONTROL)
    print(f"DFII control: 0x{dfii_ctrl:02x}")

    # Precharge all banks
    print("\n--- Precharge all ---")
    issue_cmd(wb, CMD_PRE, addr=0x400, bank=0)
    time.sleep(0.01)

    # Read all phase rddata
    print("\n--- Reading phase rddata (before any operation) ---")
    for i, addr in enumerate([DFII_PI0_RDDATA, DFII_PI1_RDDATA, DFII_PI2_RDDATA, DFII_PI3_RDDATA]):
        rd = wb.read(addr)
        print(f"  Phase {i} rddata: 0x{rd:08x}")

    # ACT
    row = 0x100
    bank = 0
    col = 0
    print(f"\n--- ACT row=0x{row:x}, bank={bank} ---")
    issue_cmd(wb, CMD_ACT, addr=row, bank=bank)
    time.sleep(0.01)

    # WRITE (full tRAS)
    pattern = 0xDEADBEEF
    print(f"\n--- WRITE col={col}, pattern=0x{pattern:08x} ---")
    issue_cmd(wb, CMD_WRITE, addr=col, bank=bank, wrdata=pattern)
    time.sleep(0.01)

    # PRE
    print("\n--- PRE ---")
    issue_cmd(wb, CMD_PRE, addr=0x400, bank=bank)
    time.sleep(0.01)

    # ACT again
    print(f"\n--- ACT row=0x{row:x}, bank={bank} ---")
    issue_cmd(wb, CMD_ACT, addr=row, bank=bank)
    time.sleep(0.01)

    # READ
    print(f"\n--- READ col={col} ---")
    issue_cmd(wb, CMD_READ, addr=col, bank=bank)

    # Wait for CL
    time.sleep(0.05)

    # Read all phase rddata IMMEDIATELY
    print("\n--- Reading phase rddata (after READ cmd) ---")
    for i, addr in enumerate([DFII_PI0_RDDATA, DFII_PI1_RDDATA, DFII_PI2_RDDATA, DFII_PI3_RDDATA]):
        rd = wb.read(addr)
        print(f"  Phase {i} rddata: 0x{rd:08x}")

    # Try multiple reads in quick succession
    print("\n--- Multiple reads from PI0 rddata ---")
    for j in range(5):
        rd = wb.read(DFII_PI0_RDDATA)
        print(f"  Read {j}: 0x{rd:08x}")

    # PRE
    print("\n--- Final PRE ---")
    issue_cmd(wb, CMD_PRE, addr=0x400, bank=bank)
    time.sleep(0.01)

    # Switch back to HW mode
    print("\n--- Switching back to HW mode ---")
    wb.write(DFII_CONTROL, HW_MODE)
    time.sleep(0.01)
    dfii_ctrl = wb.read(DFII_CONTROL)
    print(f"DFII control: 0x{dfii_ctrl:02x}")

    # Now try via L2 cache/controller to verify write worked
    print("\n--- Reading via L2 cache (controller path) ---")
    # Address calculation: DRAM_BASE + (row << 14) + (bank << 11) + col
    dram_addr = 0x40000000 + (row << 14) + (bank << 11) + col
    # Read 4 words
    for i in range(4):
        data = wb.read(dram_addr + i*4)
        print(f"  L2 read addr 0x{dram_addr + i*4:08x}: 0x{data:08x}")

    wb.close()


if __name__ == "__main__":
    main()
