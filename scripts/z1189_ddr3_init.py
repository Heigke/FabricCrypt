#!/usr/bin/env python3
"""
z1189: DDR3 Manual Initialization via DFII SW mode

Initialize DDR3 DRAM manually without BIOS by issuing the proper
initialization sequence via DFII CSRs.

DDR3 Init Sequence:
1. Apply CKE, deassert RESET_N
2. Wait 500us
3. Issue MRS to MR2 (CWL, etc.)
4. Issue MRS to MR3 (MPR, etc.)
5. Issue MRS to MR1 (DLL enable, ODT, etc.)
6. Issue MRS to MR0 (CL, WR, DLL reset)
7. Issue ZQCL (ZQ calibration long)
8. Wait 512 cycles
9. Switch to HW mode
"""

import time
from litex import RemoteClient

# DFII CSRs
DFII_CONTROL = 0x2800
DFII_PI0_COMMAND = 0x2804
DFII_PI0_COMMAND_ISSUE = 0x2808
DFII_PI0_ADDRESS = 0x280c
DFII_PI0_BADDRESS = 0x2810
DFII_PI0_WRDATA = 0x2814
DFII_PI0_RDDATA = 0x2818

# DDR3 Commands (active high for DFII)
CMD_NOP = 0x00
CMD_MRS = 0x09      # cs + ras (address selects mode register via bank + addr)
CMD_ZQCL = 0x0D     # cs + ras + we + (A10=1)
CMD_PRE = 0x0B      # cs + ras + we

# Control register bits
# bit 0: SEL (0=SW, 1=HW)
# bit 1: CKE
# bit 2: ODT
# bit 3: RESET_N

# Mode register values for DDR3-800 @ 100MHz sys_clk
# CL = 6 (5-6 cycles at 100MHz = 50-60ns)
# CWL = 5
# WR = 6

# MR0: CL=6, WR=6, DLL reset
# A[1:0]=0 (burst length 8)
# A[3]=0 (sequential)
# A[6:4]=010 (CL=6)
# A[8]=1 (DLL reset)
# A[11:9]=010 (WR=6)
MR0 = 0x0420  # CL=6, WR=6, DLL reset, BL=8

# MR1: DLL on, ODT=120ohm (Rtt_Nom=RZQ/2), AL=0
# A[0]=0 (DLL enable)
# A[1]=0 (output driver impedance, full)
# A[2]=0 (Rtt_Nom bit 0)
# A[5]=0 (AL=0)
# A[6]=0 (Rtt_Nom bit 1)
# A[9]=1 (Rtt_Nom bit 2) -> RZQ/2 = 120 ohm
MR1 = 0x0040  # DLL on, no AL, RTT_NOM=RZQ/2

# MR2: CWL=5
# A[5:3]=000 (CWL=5 for DDR3-800)
MR2 = 0x0000  # CWL=5 (000)

# MR3: MPR disabled
MR3 = 0x0000


def issue_cmd(wb, cmd, addr=0, bank=0):
    """Issue a DFII command"""
    wb.write(DFII_PI0_ADDRESS, addr)
    wb.write(DFII_PI0_BADDRESS, bank)
    wb.write(DFII_PI0_COMMAND, cmd)
    wb.write(DFII_PI0_COMMAND_ISSUE, 1)
    time.sleep(0.001)


def main():
    print("z1189: DDR3 Manual Initialization")
    print("="*60)

    wb = RemoteClient()
    wb.open()

    # Read initial state
    ctrl = wb.read(DFII_CONTROL)
    print(f"Initial DFII control: 0x{ctrl:02x}")

    # Step 1: Enter SW mode with CKE and RESET_N
    print("\nStep 1: Entering SW mode, asserting CKE, deasserting RESET...")
    wb.write(DFII_CONTROL, 0x0A)  # SEL=0, CKE=1, RESET_N=1
    time.sleep(0.001)

    # Step 2: Wait 500us (actually more to be safe)
    print("Step 2: Waiting 1ms...")
    time.sleep(0.001)

    # Step 3: Precharge all banks
    print("Step 3: Precharge all banks...")
    issue_cmd(wb, CMD_PRE, addr=0x400, bank=0)  # A10=1 for all banks
    time.sleep(0.001)

    # Step 4: Issue MRS to MR2
    print(f"Step 4: MRS to MR2 (0x{MR2:04x})...")
    issue_cmd(wb, CMD_MRS, addr=MR2, bank=2)  # Bank 2 = MR2
    time.sleep(0.001)

    # Step 5: Issue MRS to MR3
    print(f"Step 5: MRS to MR3 (0x{MR3:04x})...")
    issue_cmd(wb, CMD_MRS, addr=MR3, bank=3)  # Bank 3 = MR3
    time.sleep(0.001)

    # Step 6: Issue MRS to MR1
    print(f"Step 6: MRS to MR1 (0x{MR1:04x})...")
    issue_cmd(wb, CMD_MRS, addr=MR1, bank=1)  # Bank 1 = MR1
    time.sleep(0.001)

    # Step 7: Issue MRS to MR0 with DLL reset
    print(f"Step 7: MRS to MR0 (0x{MR0:04x})...")
    issue_cmd(wb, CMD_MRS, addr=MR0, bank=0)  # Bank 0 = MR0
    time.sleep(0.001)

    # Step 8: ZQCL (ZQ calibration long)
    print("Step 8: ZQCL (ZQ calibration)...")
    issue_cmd(wb, CMD_ZQCL, addr=0x400, bank=0)  # A10=1 for ZQCL
    time.sleep(0.001)

    # Step 9: Wait for ZQ cal (512 cycles @ 100MHz = 5.12us, use 10ms)
    print("Step 9: Waiting for ZQ calibration...")
    time.sleep(0.01)

    # Step 10: Switch to HW mode
    print("Step 10: Switching to HW mode...")
    wb.write(DFII_CONTROL, 0x0B)  # SEL=1, CKE=1, RESET_N=1
    time.sleep(0.1)

    ctrl = wb.read(DFII_CONTROL)
    print(f"Final DFII control: 0x{ctrl:02x}")

    # Test memory access
    print("\n" + "="*60)
    print("Testing memory access after init...")

    DRAM_BASE = 0x40000000

    # Test immediate write/read
    wb.write(DRAM_BASE, 0xDEADBEEF)
    val = wb.read(DRAM_BASE)
    print(f"  Immediate: wrote 0xDEADBEEF, read 0x{val:08x}",
          "OK" if val == 0xDEADBEEF else "FAIL")

    # Test with flush
    test_addr = DRAM_BASE + 0x10000
    wb.write(test_addr, 0xCAFEBABE)
    time.sleep(0.001)
    for i in range(256):
        _ = wb.read(DRAM_BASE + 0x100000 + i*4)
    val = wb.read(test_addr)
    print(f"  After 256-word flush: wrote 0xCAFEBABE, read 0x{val:08x}",
          "OK" if val == 0xCAFEBABE else "FAIL")

    # Test with larger flush
    test_addr2 = DRAM_BASE + 0x20000
    wb.write(test_addr2, 0x12345678)
    time.sleep(0.001)
    for i in range(512):
        _ = wb.read(DRAM_BASE + 0x200000 + i*4)
    val = wb.read(test_addr2)
    print(f"  After 512-word flush: wrote 0x12345678, read 0x{val:08x}",
          "OK" if val == 0x12345678 else "FAIL")

    wb.close()


if __name__ == "__main__":
    main()
