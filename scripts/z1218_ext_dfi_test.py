#!/usr/bin/env python3
"""
z1218: Test ext_dfi mux directly.

This test verifies if the ext_dfi path works at all by:
1. Init DDR3 (SW mode)
2. Write known value via SW mode
3. Enable ext_dfi_sel via CSR
4. Issue commands via ext_dfi
5. Read back via SW mode to see if ext_dfi affected memory
"""

import time
from litex.tools.litex_client import RemoteClient

# Standard CSRs
DFII_CONTROL = 0x3000
PHASE_SIZE = 0x18

def pi_base(phase):
    return 0x3004 + phase * PHASE_SIZE

# ext_dfi_test CSRs - need to find from csr.csv
# Let's read csr.csv to get the addresses
def main():
    print("=" * 60)
    print("z1218: ext_dfi Mux Direct Test")
    print("=" * 60)

    wb = RemoteClient(host='localhost', port=1234)
    wb.open()

    ident = wb.read(0x1800)
    print(f"\nConnected! Identifier: 0x{ident:08x}")

    # Read CSR map
    print("\nReading CSR addresses from build...")
    try:
        with open("build_ext_dfi_test/csr.csv", "r") as f:
            csrs = {}
            for line in f:
                if line.startswith("csr_register"):
                    parts = line.strip().split(",")
                    name = parts[1]
                    addr = int(parts[2], 0)
                    csrs[name] = addr
            print(f"  Found {len(csrs)} CSRs")
    except:
        print("  Could not read csr.csv, using defaults")
        csrs = {}

    # Get ext_dfi_test addresses
    ext_sel = csrs.get("ext_dfi_test_ext_sel", 0x2800)
    ext_cmd = csrs.get("ext_dfi_test_cmd", 0x2804)
    ext_addr = csrs.get("ext_dfi_test_addr", 0x2808)
    ext_bank = csrs.get("ext_dfi_test_bank", 0x280c)
    ext_wrdata = csrs.get("ext_dfi_test_wrdata", 0x2810)
    ext_status = csrs.get("ext_dfi_test_status", 0x2818)
    ext_master_rddata = csrs.get("ext_dfi_test_master_rddata", 0x281c)

    print(f"\n  ext_sel: 0x{ext_sel:04x}")
    print(f"  ext_cmd: 0x{ext_cmd:04x}")
    print(f"  ext_status: 0x{ext_status:04x}")
    print(f"  ext_master_rddata: 0x{ext_master_rddata:04x}")

    # Check initial state
    print("\n=== Initial State ===")
    dfii_ctrl = wb.read(DFII_CONTROL)
    status = wb.read(ext_status)
    print(f"  DFII Control: 0x{dfii_ctrl:02x}")
    print(f"  ext_status: 0x{status:02x} (ext_dfi_sel={(status>>0)&1})")

    # First init DDR3
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

    # Test SW mode works
    print("\n=== Test SW Mode ===")
    # Write via SW mode
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

    wb.write(DFII_CONTROL, 0x0F)  # Back to HW mode
    print("  SW mode OK")

    # Now test ext_dfi
    print("\n=== Test ext_dfi Path ===")

    # First, enable ext_dfi_sel
    print("  Setting ext_dfi_sel=1...")
    wb.write(ext_sel, 1)
    time.sleep(0.01)

    status = wb.read(ext_status)
    print(f"  ext_status: 0x{status:02x} (ext_dfi_sel={(status>>0)&1})")

    # Set ext_dfi command to NOP first
    # NOP = all signals high (inactive)
    wb.write(ext_cmd, 0x00)  # NOP
    wb.write(ext_addr, 0)
    wb.write(ext_bank, 0)
    wb.write(ext_wrdata, 0x55555555)

    # Check master_rddata
    master_rd = wb.read(ext_master_rddata)
    print(f"  master_rddata: 0x{master_rd:08x}")

    # Try issuing a PRECHARGE ALL command
    # PRE: cs=1, ras=1, we=1, cas=0 -> cmd = 0x0B (same as SW mode)
    print("  Issuing PRE_ALL via ext_dfi...")
    wb.write(ext_cmd, 0x0B)  # PRE
    wb.write(ext_addr, 0x400)  # A10=1
    time.sleep(0.001)

    # Issue ACTIVATE
    print("  Issuing ACT via ext_dfi...")
    wb.write(ext_cmd, 0x09)  # ACT
    wb.write(ext_addr, 200)  # row 200
    wb.write(ext_bank, 0)
    time.sleep(0.001)

    # Issue WRITE
    print("  Issuing WRITE via ext_dfi...")
    # WRITE: cs=1, cas=1, we=1, wrdata_en=1 -> 0x17
    wb.write(ext_cmd, 0x17)
    wb.write(ext_addr, 0)  # col 0
    wb.write(ext_bank, 0)
    wb.write(ext_wrdata, 0xDEADBEEF)  # Different from 0xAAAAAAAA
    time.sleep(0.01)

    # Issue PRECHARGE
    wb.write(ext_cmd, 0x0B)
    wb.write(ext_addr, 0x400)
    time.sleep(0.001)

    # Issue NOP
    wb.write(ext_cmd, 0x00)

    # Disable ext_dfi_sel
    print("  Setting ext_dfi_sel=0...")
    wb.write(ext_sel, 0)
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
        print("  The ext_dfi mux is not switching or commands not reaching PHY.")
    else:
        print(f"  Unexpected value: 0x{final_result:08x}")

    wb.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
