#!/usr/bin/env python3
"""
z1220: Test ext_dfi v2 with phase selection

v2 improvements:
- Commands only issued on the selected phase (not all phases)
- Trigger-based one-shot command issue (like SW mode phase injectors)
- Should properly interact with DDR3 1:4 mode

CSR Layout (from csr.h):
- 0x1800: ext_sel
- 0x1804: cmd_phase (0-3)
- 0x1808: cmd [CS,WE,CAS,RAS,WREN,RDEN,0,0]
- 0x180c: addr (14-bit)
- 0x1810: bank (3-bit)
- 0x1814: wrdata (32-bit)
- 0x1818: trigger (write 1 to issue)
- 0x181c: status
- 0x1820: master_rddata
"""

import time
from litex.tools.litex_client import RemoteClient

# DFII CSRs
DFII_CONTROL = 0x3000
PHASE_SIZE = 0x18

def pi_base(phase):
    return 0x3004 + phase * PHASE_SIZE

# EXT_DFI_TEST v2 CSRs
EXT_SEL = 0x1800
EXT_CMD_PHASE = 0x1804
EXT_CMD = 0x1808
EXT_ADDR = 0x180c
EXT_BANK = 0x1810
EXT_WRDATA = 0x1814
EXT_TRIGGER = 0x1818
EXT_STATUS = 0x181c
EXT_MASTER_RDDATA = 0x1820


def issue_ext_cmd(wb, cmd, addr, bank, wrdata, phase):
    """Issue a command via ext_dfi on specified phase."""
    wb.write(EXT_CMD_PHASE, phase)
    wb.write(EXT_CMD, cmd)
    wb.write(EXT_ADDR, addr)
    wb.write(EXT_BANK, bank)
    wb.write(EXT_WRDATA, wrdata)
    # Trigger (0->1 edge)
    wb.write(EXT_TRIGGER, 0)
    wb.write(EXT_TRIGGER, 1)
    wb.write(EXT_TRIGGER, 0)


def main():
    print("=" * 60)
    print("z1220: ext_dfi v2 Test with Phase Selection")
    print("=" * 60)

    wb = RemoteClient(host='localhost', port=1234)
    wb.open()

    ident = wb.read(0x2000)
    print(f"\nConnected! Identifier: 0x{ident:08x}")

    # Check initial state
    print("\n=== Initial State ===")
    dfii_ctrl = wb.read(DFII_CONTROL)
    ext_sel_val = wb.read(EXT_SEL)
    status = wb.read(EXT_STATUS)
    print(f"  DFII Control: 0x{dfii_ctrl:02x}")
    print(f"  ext_sel: 0x{ext_sel_val:02x}")
    print(f"  status: 0x{status:02x}")

    # Init DDR3
    print("\n=== Initialize DDR3 ===")
    import subprocess
    result = subprocess.run(
        ["python3", "scripts/z1210_ddr3_litex_init.py"],
        capture_output=True,
        text=True
    )
    if "SUCCESSFUL" in result.stdout:
        print("  DDR3 init successful")
    else:
        print("  DDR3 init may have failed")
        print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)

    # Test SW mode first
    print("\n=== Test SW Mode ===")
    WRPHASE = 3
    RDPHASE = 2

    wb.write(DFII_CONTROL, 0x0E)  # SW mode
    time.sleep(0.01)

    # Precharge all
    wb.write(pi_base(0) + 0x08, 0x400)
    wb.write(pi_base(0) + 0x0c, 0)
    wb.write(pi_base(0) + 0x00, 0x0B)
    wb.write(pi_base(0) + 0x04, 1)
    time.sleep(0.001)

    # Activate row 200
    wb.write(pi_base(0) + 0x08, 200)
    wb.write(pi_base(0) + 0x0c, 0)
    wb.write(pi_base(0) + 0x00, 0x09)
    wb.write(pi_base(0) + 0x04, 1)
    time.sleep(0.001)

    # Write 0xAAAAAAAA
    wb.write(pi_base(WRPHASE) + 0x10, 0xAAAAAAAA)
    wb.write(pi_base(WRPHASE) + 0x08, 0)
    wb.write(pi_base(WRPHASE) + 0x0c, 0)
    wb.write(pi_base(WRPHASE) + 0x00, 0x17)
    wb.write(pi_base(WRPHASE) + 0x04, 1)
    time.sleep(0.01)

    # Precharge
    wb.write(pi_base(0) + 0x08, 0x400)
    wb.write(pi_base(0) + 0x0c, 0)
    wb.write(pi_base(0) + 0x00, 0x0B)
    wb.write(pi_base(0) + 0x04, 1)
    time.sleep(0.001)

    # Read back
    wb.write(pi_base(0) + 0x08, 200)
    wb.write(pi_base(0) + 0x0c, 0)
    wb.write(pi_base(0) + 0x00, 0x09)
    wb.write(pi_base(0) + 0x04, 1)
    time.sleep(0.001)

    wb.write(pi_base(RDPHASE) + 0x08, 0)
    wb.write(pi_base(RDPHASE) + 0x0c, 0)
    wb.write(pi_base(RDPHASE) + 0x00, 0x25)
    wb.write(pi_base(RDPHASE) + 0x04, 1)
    time.sleep(0.01)

    sw_result = wb.read(pi_base(0) + 0x14)
    print(f"  Wrote 0xAAAAAAAA, Read 0x{sw_result:08x}")

    if sw_result != 0xAAAAAAAA:
        print("  SW mode not working!")
        wb.close()
        return

    print("  SW mode OK")

    # Now test ext_dfi v2
    print("\n=== Test ext_dfi v2 ===")

    # Switch to HW mode
    wb.write(DFII_CONTROL, 0x0F)
    time.sleep(0.01)

    # Enable ext_dfi
    print("  Enabling ext_dfi_sel...")
    wb.write(EXT_SEL, 1)
    time.sleep(0.01)

    # Verify
    ext_sel_rb = wb.read(EXT_SEL)
    status = wb.read(EXT_STATUS)
    print(f"  ext_sel readback: 0x{ext_sel_rb:02x}")
    print(f"  status: 0x{status:02x} (ext_dfi_sel={(status>>0)&1})")

    # Issue commands via ext_dfi using same phases as SW mode
    # PRECHARGE ALL (cmd=0x0B on phase 0)
    print("  Issuing PRE_ALL via ext_dfi (phase 0)...")
    issue_ext_cmd(wb, 0x0B, 0x400, 0, 0, phase=0)
    time.sleep(0.001)

    # ACTIVATE (cmd=0x09 on phase 0)
    print("  Issuing ACT via ext_dfi (phase 0)...")
    issue_ext_cmd(wb, 0x09, 200, 0, 0, phase=0)
    time.sleep(0.001)

    # WRITE (cmd=0x17 on phase 3 = WRPHASE)
    print("  Issuing WRITE via ext_dfi (phase 3)...")
    issue_ext_cmd(wb, 0x17, 0, 0, 0xDEADBEEF, phase=WRPHASE)
    time.sleep(0.01)

    # PRECHARGE (phase 0)
    print("  Issuing PRE_ALL via ext_dfi (phase 0)...")
    issue_ext_cmd(wb, 0x0B, 0x400, 0, 0, phase=0)
    time.sleep(0.001)

    # Disable ext_dfi
    print("  Disabling ext_dfi_sel...")
    wb.write(EXT_SEL, 0)
    time.sleep(0.01)

    # Read back via SW mode
    print("\n=== Verify Result via SW Mode ===")
    wb.write(DFII_CONTROL, 0x0E)
    time.sleep(0.01)

    # Precharge
    wb.write(pi_base(0) + 0x08, 0x400)
    wb.write(pi_base(0) + 0x0c, 0)
    wb.write(pi_base(0) + 0x00, 0x0B)
    wb.write(pi_base(0) + 0x04, 1)
    time.sleep(0.001)

    # Activate
    wb.write(pi_base(0) + 0x08, 200)
    wb.write(pi_base(0) + 0x0c, 0)
    wb.write(pi_base(0) + 0x00, 0x09)
    wb.write(pi_base(0) + 0x04, 1)
    time.sleep(0.001)

    # Read
    wb.write(pi_base(RDPHASE) + 0x08, 0)
    wb.write(pi_base(RDPHASE) + 0x0c, 0)
    wb.write(pi_base(RDPHASE) + 0x00, 0x25)
    wb.write(pi_base(RDPHASE) + 0x04, 1)
    time.sleep(0.01)

    final_result = wb.read(pi_base(0) + 0x14)
    print(f"  Final value at row=200, col=0: 0x{final_result:08x}")

    wb.write(DFII_CONTROL, 0x0F)

    print("\n=== Analysis ===")
    if final_result == 0xDEADBEEF:
        print("  *** ext_dfi v2 PATH WORKS! ***")
        print("  ext_dfi wrote 0xDEADBEEF successfully!")
        print("  Phase selection fixed the issue!")
    elif final_result == 0xAAAAAAAA:
        print("  ext_dfi v2 write DID NOT affect memory.")
        print("  Value still 0xAAAAAAAA (from SW mode write).")
        print("  ")
        print("  Possible issues:")
        print("    1. Trigger timing - command pulse too short?")
        print("    2. Need multiple triggers for full sequence?")
        print("    3. ext_dfi mux still not routing to PHY?")
    else:
        print(f"  Unexpected value: 0x{final_result:08x}")
        print("  Partial write or corruption detected.")

    wb.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
