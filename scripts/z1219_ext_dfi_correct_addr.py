#!/usr/bin/env python3
"""
z1219: Test ext_dfi mux with CORRECT CSR addresses from csr.h

Previous tests used wrong addresses (0x2800 = LEDS base).
Correct addresses from build_ext_dfi_test/software/include/generated/csr.h:
- EXT_DFI_TEST base: 0x1800
"""

import time
from litex.tools.litex_client import RemoteClient

# DFII CSRs
DFII_CONTROL = 0x3000
PHASE_SIZE = 0x18

def pi_base(phase):
    return 0x3004 + phase * PHASE_SIZE

# EXT_DFI_TEST CSRs - CORRECT addresses from csr.h
EXT_DFI_BASE = 0x1800
EXT_SEL = EXT_DFI_BASE + 0x00       # CSRSTORAGE_364
EXT_CMD = EXT_DFI_BASE + 0x04       # CSRSTORAGE_365
EXT_ADDR = EXT_DFI_BASE + 0x08      # CSRSTORAGE_366
EXT_BANK = EXT_DFI_BASE + 0x0c      # CSRSTORAGE_367
EXT_WRDATA = EXT_DFI_BASE + 0x10    # CSRSTORAGE_368
EXT_TRIGGER = EXT_DFI_BASE + 0x14   # CSRSTORAGE_369
EXT_STATUS = EXT_DFI_BASE + 0x18    # CSRSTATUS_370
EXT_MASTER_RDDATA = EXT_DFI_BASE + 0x1c  # CSRSTATUS_371


def main():
    print("=" * 60)
    print("z1219: ext_dfi Test with CORRECT CSR Addresses")
    print("=" * 60)

    wb = RemoteClient(host='localhost', port=1234)
    wb.open()

    ident = wb.read(0x2000)  # IDENTIFIER_MEM_BASE
    print(f"\nConnected! Identifier: 0x{ident:08x}")

    print("\nCSR Addresses (from csr.h):")
    print(f"  EXT_SEL:          0x{EXT_SEL:04x}")
    print(f"  EXT_CMD:          0x{EXT_CMD:04x}")
    print(f"  EXT_ADDR:         0x{EXT_ADDR:04x}")
    print(f"  EXT_BANK:         0x{EXT_BANK:04x}")
    print(f"  EXT_WRDATA:       0x{EXT_WRDATA:04x}")
    print(f"  EXT_TRIGGER:      0x{EXT_TRIGGER:04x}")
    print(f"  EXT_STATUS:       0x{EXT_STATUS:04x}")
    print(f"  EXT_MASTER_RDDATA: 0x{EXT_MASTER_RDDATA:04x}")

    # Check initial state
    print("\n=== Initial State ===")
    dfii_ctrl = wb.read(DFII_CONTROL)
    ext_sel_val = wb.read(EXT_SEL)
    status = wb.read(EXT_STATUS)
    print(f"  DFII Control: 0x{dfii_ctrl:02x} (SEL={(dfii_ctrl>>0)&1})")
    print(f"  ext_sel CSR: 0x{ext_sel_val:02x}")
    print(f"  ext_status: 0x{status:02x}")

    # Init DDR3
    print("\n=== Initialize DDR3 ===")
    print("  Running z1210 init...")
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
    wb.write(pi_base(0) + 0x08, 0x400)  # address
    wb.write(pi_base(0) + 0x0c, 0)       # bank
    wb.write(pi_base(0) + 0x00, 0x0B)    # command (PRE)
    wb.write(pi_base(0) + 0x04, 1)       # issue
    time.sleep(0.001)

    # Activate row 200
    wb.write(pi_base(0) + 0x08, 200)
    wb.write(pi_base(0) + 0x0c, 0)
    wb.write(pi_base(0) + 0x00, 0x09)    # ACT
    wb.write(pi_base(0) + 0x04, 1)
    time.sleep(0.001)

    # Write 0xAAAAAAAA
    wb.write(pi_base(WRPHASE) + 0x10, 0xAAAAAAAA)  # wrdata
    wb.write(pi_base(WRPHASE) + 0x08, 0)           # column
    wb.write(pi_base(WRPHASE) + 0x0c, 0)           # bank
    wb.write(pi_base(WRPHASE) + 0x00, 0x17)        # WRITE
    wb.write(pi_base(WRPHASE) + 0x04, 1)           # issue
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
    wb.write(pi_base(0) + 0x00, 0x09)    # ACT
    wb.write(pi_base(0) + 0x04, 1)
    time.sleep(0.001)

    wb.write(pi_base(RDPHASE) + 0x08, 0)
    wb.write(pi_base(RDPHASE) + 0x0c, 0)
    wb.write(pi_base(RDPHASE) + 0x00, 0x25)  # READ
    wb.write(pi_base(RDPHASE) + 0x04, 1)
    time.sleep(0.01)

    sw_result = wb.read(pi_base(0) + 0x14)
    print(f"  Wrote 0xAAAAAAAA, Read 0x{sw_result:08x}")

    if sw_result != 0xAAAAAAAA:
        print("  SW mode not working!")
        wb.close()
        return

    print("  SW mode OK")

    # Now test ext_dfi
    print("\n=== Test ext_dfi Path ===")

    # Switch to HW mode first
    wb.write(DFII_CONTROL, 0x0F)  # HW mode (SEL=1)
    time.sleep(0.01)

    dfii_ctrl = wb.read(DFII_CONTROL)
    print(f"  DFII Control set to: 0x{dfii_ctrl:02x} (SEL={(dfii_ctrl>>0)&1})")

    # Now enable ext_dfi_sel
    print("  Setting ext_sel=1...")
    wb.write(EXT_SEL, 1)
    time.sleep(0.01)

    # Verify
    ext_sel_readback = wb.read(EXT_SEL)
    status = wb.read(EXT_STATUS)
    print(f"  ext_sel readback: 0x{ext_sel_readback:02x}")
    print(f"  ext_status: 0x{status:02x} (ext_dfi_sel={(status>>0)&1})")

    if ext_sel_readback != 1:
        print("  ERROR: ext_sel not being set!")
        print("  This means the CSR address is still wrong or module not connected.")
        wb.close()
        return

    # Set ext_dfi to NOP first (all signals high/inactive)
    # In active-low encoding: cs_n=1, we_n=1, cas_n=1, ras_n=1 means NOP
    # Our encoding: bit0=CS, bit1=WE, bit2=CAS, bit3=RAS
    # cs_n = ~bit0, so bit0=0 means cs_n=1 (inactive)
    # For NOP: all cmd bits = 0 (signals stay high/inactive)
    print("  Setting NOP on ext_dfi...")
    wb.write(EXT_CMD, 0x00)  # NOP (all signals inactive)
    wb.write(EXT_ADDR, 0)
    wb.write(EXT_BANK, 0)
    wb.write(EXT_WRDATA, 0)
    time.sleep(0.01)

    # Check master rddata
    master_rd = wb.read(EXT_MASTER_RDDATA)
    print(f"  master_rddata: 0x{master_rd:08x}")

    # Now issue PRECHARGE ALL via ext_dfi
    # PRE: cs=1, ras=1, we=1 -> cmd = 0x0B (bit0=1, bit1=1, bit3=1)
    print("  Issuing PRE_ALL via ext_dfi...")
    wb.write(EXT_CMD, 0x0B)  # CS + WE + RAS
    wb.write(EXT_ADDR, 0x400)  # A10=1 for all banks
    time.sleep(0.001)

    # Issue ACTIVATE
    print("  Issuing ACT via ext_dfi...")
    wb.write(EXT_CMD, 0x09)  # CS + RAS
    wb.write(EXT_ADDR, 200)  # row 200
    wb.write(EXT_BANK, 0)
    time.sleep(0.001)

    # Issue WRITE with different data
    print("  Issuing WRITE via ext_dfi...")
    # WRITE: cs=1, cas=1, we=1, wrdata_en=1 -> 0x17
    wb.write(EXT_CMD, 0x17)  # CS + WE + CAS + WRDATA_EN
    wb.write(EXT_ADDR, 0)  # col 0
    wb.write(EXT_BANK, 0)
    wb.write(EXT_WRDATA, 0xDEADBEEF)  # Different from 0xAAAAAAAA
    time.sleep(0.01)

    # Issue PRECHARGE
    wb.write(EXT_CMD, 0x0B)
    wb.write(EXT_ADDR, 0x400)
    time.sleep(0.001)

    # Back to NOP
    wb.write(EXT_CMD, 0x00)
    time.sleep(0.001)

    # Disable ext_dfi_sel
    print("  Setting ext_sel=0...")
    wb.write(EXT_SEL, 0)
    time.sleep(0.01)

    # Read back via SW mode
    print("\n=== Verify Result via SW Mode ===")
    wb.write(DFII_CONTROL, 0x0E)  # SW mode
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

    wb.write(DFII_CONTROL, 0x0F)  # HW mode

    print("\n=== Analysis ===")
    if final_result == 0xDEADBEEF:
        print("  *** ext_dfi PATH WORKS! ***")
        print("  ext_dfi wrote 0xDEADBEEF successfully!")
    elif final_result == 0xAAAAAAAA:
        print("  ext_dfi write DID NOT affect memory.")
        print("  Value still 0xAAAAAAAA (from SW mode write).")
        print("  Possible issues:")
        print("    1. ext_dfi commands not being latched by PHY")
        print("    2. Command timing too fast (need clock delays)")
        print("    3. PHY requires specific phase alignment")
    else:
        print(f"  Unexpected value: 0x{final_result:08x}")

    wb.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
