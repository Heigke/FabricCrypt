#!/usr/bin/env python3
"""
z1134: LiteDRAM Raw DFI Command Test for Multi-Level DRAM Charging

This script uses LiteX remote access to control the DFIInjector and test
truncated tRAS (Frac) operations for achieving multi-level cell charge.

Requirements:
1. Build and load the FracSoC bitstream: build_frac_soc.py --build --load
2. Connect via UART or Etherbone

Theory of Operation:
- DFI software control mode allows direct ACT/PRE commands
- By issuing ACT then PRE with truncated timing (before full tRAS),
  we can achieve partial charge transfer in DRAM cells
- Multiple short ACT/PRE cycles accumulate to intermediate charge levels

DDR3 Commands (RAS/CAS/WE encoding):
- ACT (Activate): RAS=0, CAS=1, WE=1 -> ras=1, cas=0, we=0 in CSR (inverted)
- PRE (Precharge): RAS=0, CAS=1, WE=0 -> ras=1, cas=0, we=1 in CSR
- NOP: CS=1 (deselect) -> cs=0 in CSR
"""

import time
import argparse
import json
import struct
from pathlib import Path

# Try to import litex remote access
try:
    from litex.tools.litex_client import RemoteClient
    HAS_LITEX_CLIENT = True
except ImportError:
    HAS_LITEX_CLIENT = False
    print("Warning: litex.tools.litex_client not available")

# Try serial fallback
try:
    import serial
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False


class DFIController:
    """Controller for LiteDRAM DFI software mode

    CSR Addresses from build (base: 0x00002000):
      - sdram_dfii_control:      0x00002000 (sel, cke, odt, reset_n)
      - sdram_dfii_pi0_command:  0x00002004 (cs, we, cas, ras, wren, rden)
      - sdram_dfii_pi0_issue:    0x00002008 (trigger)
      - sdram_dfii_pi0_address:  0x0000200c (row address)
      - sdram_dfii_pi0_baddress: 0x00002010 (bank address)
      - sdram_dfii_pi0_wrdata:   0x00002014 (write data)
      - sdram_dfii_pi0_rddata:   0x00002018 (read data)
    """

    # DDR3 timing in cycles at 100MHz (10ns per cycle)
    T_RCD = 14   # Row to column delay (~14 cycles = 140ns)
    T_RAS = 37   # Row active time minimum (~37 cycles = 370ns)
    T_RP = 14    # Row precharge time (~14 cycles = 140ns)
    T_RC = 51    # Row cycle time (tRAS + tRP)

    # CSR base addresses
    CSR_BASE = 0x00002000

    def __init__(self, client):
        self.client = client
        self.in_software_mode = False

    def enter_software_mode(self):
        """Switch DFI to software control"""
        # dfii._control: sel=0 for software, cke=1, odt=0, reset_n=1
        # Format: [reset_n, odt, cke, sel] = [1, 0, 1, 0] = 0b1010 = 10
        try:
            self.client.regs.sdram_dfii_control.write(0b1010)
        except AttributeError:
            # Fallback to direct address if reg name doesn't match
            self.client.write(self.CSR_BASE, 0b1010)
        self.in_software_mode = True
        print("DFI: Entered software control mode")

    def exit_software_mode(self):
        """Return DFI to hardware control"""
        # sel=1 for hardware control
        try:
            self.client.regs.sdram_dfii_control.write(0b1011)
        except AttributeError:
            self.client.write(self.CSR_BASE, 0b1011)
        self.in_software_mode = False
        print("DFI: Returned to hardware control mode")

    def nop(self, phase=0):
        """Issue NOP (no operation) - deselect chip"""
        pi = getattr(self.client.regs, f'sdram_dfii_pi{phase}_command')
        pi_issue = getattr(self.client.regs, f'sdram_dfii_pi{phase}_command_issue')
        # cs=0 means deselected (NOP)
        pi.write(0)
        pi_issue.write(1)

    def activate(self, bank, row, phase=0):
        """Issue ACT (activate) command"""
        # Set bank and row address
        pi_addr = getattr(self.client.regs, f'sdram_dfii_pi{phase}_address')
        pi_baddr = getattr(self.client.regs, f'sdram_dfii_pi{phase}_baddress')
        pi_cmd = getattr(self.client.regs, f'sdram_dfii_pi{phase}_command')
        pi_issue = getattr(self.client.regs, f'sdram_dfii_pi{phase}_command_issue')

        pi_addr.write(row)
        pi_baddr.write(bank)
        # ACT: cs=1, ras=1, cas=0, we=0 -> 0b0011 = 3 (in CSR active-high logic)
        pi_cmd.write(0b0011)
        pi_issue.write(1)

    def precharge(self, bank=None, phase=0):
        """Issue PRE (precharge) command. bank=None for all banks."""
        pi_addr = getattr(self.client.regs, f'sdram_dfii_pi{phase}_address')
        pi_baddr = getattr(self.client.regs, f'sdram_dfii_pi{phase}_baddress')
        pi_cmd = getattr(self.client.regs, f'sdram_dfii_pi{phase}_command')
        pi_issue = getattr(self.client.regs, f'sdram_dfii_pi{phase}_command_issue')

        if bank is None:
            # All banks precharge: A10=1
            pi_addr.write(1 << 10)
            pi_baddr.write(0)
        else:
            pi_addr.write(0)
            pi_baddr.write(bank)
        # PRE: cs=1, ras=1, cas=0, we=1 -> 0b0111 = 7
        pi_cmd.write(0b0111)
        pi_issue.write(1)

    def write_data(self, data, phase=0):
        """Write data to DFI for subsequent WRITE command"""
        pi_wrdata = getattr(self.client.regs, f'sdram_dfii_pi{phase}_wrdata')
        pi_wrdata.write(data)

    def read_data(self, phase=0):
        """Read captured data from DFI after READ command"""
        pi_rddata = getattr(self.client.regs, f'sdram_dfii_pi{phase}_rddata')
        return pi_rddata.read()

    def frac_operation(self, bank, row, partial_tras_cycles, num_fracs=1):
        """
        Perform Frac operation: repeated ACT->short_wait->PRE cycles

        This is the core operation for multi-level charging:
        - ACT opens the row (starts charge transfer)
        - Wait partial_tras_cycles (much less than T_RAS)
        - PRE closes the row (stops charge transfer early)
        - Repeat to accumulate charge

        Args:
            bank: Bank address (0-7)
            row: Row address (0-8191 for our chip)
            partial_tras_cycles: Cycles to wait before PRE (< T_RAS for partial)
            num_fracs: Number of ACT/PRE iterations
        """
        cycle_time_ns = 10  # At 100MHz

        for i in range(num_fracs):
            # Issue ACT
            self.activate(bank, row)

            # Wait partial tRAS (in software we can only approximate)
            # Each CSR access takes some cycles, so timing isn't exact
            wait_us = (partial_tras_cycles * cycle_time_ns) / 1000
            if wait_us > 0.001:
                time.sleep(wait_us / 1000000)

            # Issue PRE
            self.precharge(bank)

            # Wait tRP before next operation
            wait_us = (self.T_RP * cycle_time_ns) / 1000
            time.sleep(wait_us / 1000000)


def test_dfi_access(client):
    """Basic test of DFI software mode access"""
    print("\n=== Testing DFI Access ===")

    ctrl = DFIController(client)

    # Enter software mode
    ctrl.enter_software_mode()

    # Issue a few NOPs
    for i in range(4):
        ctrl.nop(phase=i)
    print("Issued NOPs on all phases")

    # Read back control register
    control = client.regs.sdram_dfii_control.read()
    print(f"DFI control register: 0x{control:04x}")

    # Return to hardware mode
    ctrl.exit_software_mode()

    print("DFI access test complete")


def test_frac_sweep(client, bank=0, row=0):
    """Sweep partial_tras values to characterize multi-level charging"""
    print("\n=== Frac Operation Sweep ===")
    print(f"Bank: {bank}, Row: {row}")

    ctrl = DFIController(client)
    ctrl.enter_software_mode()

    # Test configurations: (partial_tras_cycles, num_fracs, description)
    configs = [
        (5, 1, "minimal"),
        (10, 1, "short"),
        (20, 1, "medium-short"),
        (37, 1, "full tRAS (control)"),
        (5, 5, "5x minimal"),
        (10, 10, "10x short"),
        (5, 15, "15x minimal"),
    ]

    results = []
    for partial_tras, num_fracs, desc in configs:
        print(f"\nConfig: partial_tras={partial_tras}, num_fracs={num_fracs} ({desc})")

        # Perform Frac operation
        ctrl.frac_operation(bank, row, partial_tras, num_fracs)

        results.append({
            'partial_tras': partial_tras,
            'num_fracs': num_fracs,
            'desc': desc,
            # Note: actual charge measurement requires reading back the cells
            # This requires additional READ command implementation
        })
        print(f"  Completed {num_fracs} Frac operations")

    ctrl.exit_software_mode()

    return results


def find_csr_registers(client):
    """List available CSR registers for debugging"""
    print("\n=== Available CSR Registers ===")

    if hasattr(client, 'regs'):
        regs = dir(client.regs)
        sdram_regs = [r for r in regs if 'sdram' in r.lower() or 'dfii' in r.lower()]
        for reg in sorted(sdram_regs):
            print(f"  {reg}")
        return sdram_regs
    return []


def main():
    parser = argparse.ArgumentParser(description='LiteDRAM Frac Operation Test')
    parser.add_argument('--host', default='192.168.1.50', help='Etherbone host IP')
    parser.add_argument('--port', type=int, default=1234, help='Etherbone port')
    parser.add_argument('--uart', default=None, help='UART device for serial bridge')
    parser.add_argument('--csr-csv', default=None, help='Path to CSR CSV file')
    parser.add_argument('--test-access', action='store_true', help='Just test DFI access')
    parser.add_argument('--list-regs', action='store_true', help='List available registers')
    parser.add_argument('--bank', type=int, default=0, help='Bank for Frac test')
    parser.add_argument('--row', type=int, default=0, help='Row for Frac test')
    args = parser.parse_args()

    if not HAS_LITEX_CLIENT:
        print("ERROR: litex.tools.litex_client required")
        print("Install with: pip install litex")
        return 1

    # Connect to SoC
    print(f"Connecting to SoC at {args.host}:{args.port}...")
    try:
        client = RemoteClient(host=args.host, port=args.port, csr_csv=args.csr_csv)
        client.open()
        print("Connected!")
    except Exception as e:
        print(f"Connection failed: {e}")
        return 1

    try:
        # List registers if requested
        if args.list_regs:
            find_csr_registers(client)
            return 0

        # Test access only
        if args.test_access:
            test_dfi_access(client)
            return 0

        # Full Frac sweep
        results = test_frac_sweep(client, args.bank, args.row)

        # Save results
        results_path = Path('results/z1134_litedram_frac_test.json')
        results_path.parent.mkdir(exist_ok=True)
        with open(results_path, 'w') as f:
            json.dump({
                'test': 'z1134_litedram_frac',
                'bank': args.bank,
                'row': args.row,
                'results': results,
            }, f, indent=2)
        print(f"\nResults saved to {results_path}")

    finally:
        client.close()
        print("\nDisconnected")

    return 0


if __name__ == '__main__':
    exit(main())
