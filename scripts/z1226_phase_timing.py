#!/usr/bin/env python3
"""
z1226: Phase-based Sub-cycle Timing

The 4 DFI phases give us 2.5ns resolution (10ns / 4 phases).
By issuing commands on different phases, we might get finer timing:

- Phase 0: t=0ns
- Phase 1: t=2.5ns
- Phase 2: t=5ns
- Phase 3: t=7.5ns

If ACT on phase 0 and PRE on phase 1 = 2.5ns active time
If ACT on phase 0 and PRE on phase 2 = 5ns active time
etc.
"""

import time
import json
import subprocess
from datetime import datetime
from litex.tools.litex_client import RemoteClient

# CSRs
DFII_CONTROL = 0x3000
PHASE_SIZE = 0x18
def pi_base(phase): return 0x3004 + phase * PHASE_SIZE

# Commands
CMD_NOP = 0x07      # RAS=1, CAS=1, WE=1
CMD_ACT = 0x09      # RAS=0, CAS=1, WE=1 (with CS)
CMD_PRE = 0x0B      # RAS=0, CAS=1, WE=0
CMD_WRITE = 0x17    # RAS=1, CAS=0, WE=0
CMD_READ = 0x25     # RAS=1, CAS=0, WE=1


def count_ones(val):
    return bin(val).count('1')


def issue_on_phase(wb, phase, cmd, addr, bank):
    """Issue a command on a specific DFI phase."""
    wb.write(pi_base(phase) + 0x08, addr)    # address
    wb.write(pi_base(phase) + 0x0c, bank)    # bank
    wb.write(pi_base(phase) + 0x00, cmd)     # command
    wb.write(pi_base(phase) + 0x04, 1)       # issue


def write_with_phase_timing(wb, row, col, bank, data, act_phase, pre_phase):
    """
    Write with ACT on act_phase and PRE on pre_phase.
    This gives timing resolution of 2.5ns per phase difference.
    """
    wb.write(DFII_CONTROL, 0x0E)  # SW mode
    time.sleep(0.001)

    # First, precharge all to start clean
    issue_on_phase(wb, 0, CMD_PRE, 0x400, 0)
    time.sleep(0.0002)

    # Issue ACT on specified phase
    issue_on_phase(wb, act_phase, CMD_ACT, row, bank)

    # Small delay
    time.sleep(0.0001)

    # Issue WRITE on phase 3 (data)
    wb.write(pi_base(3) + 0x10, data)
    issue_on_phase(wb, 3, CMD_WRITE, col, bank)

    # Issue PRE on specified phase
    # If pre_phase > act_phase, it happens in same clock cycle = ultra fast
    # If pre_phase < act_phase, we need to wait for next cycle
    if pre_phase >= act_phase:
        # Same clock cycle - just issue on that phase
        issue_on_phase(wb, pre_phase, CMD_PRE, 0x400, bank)
    else:
        # Wait for next cycle
        time.sleep(0.00001)  # ~10ns
        issue_on_phase(wb, pre_phase, CMD_PRE, 0x400, bank)

    time.sleep(0.0002)
    wb.write(DFII_CONTROL, 0x0F)  # HW mode


def read_data(wb, row, col, bank):
    """Standard read."""
    wb.write(DFII_CONTROL, 0x0E)
    time.sleep(0.001)
    issue_on_phase(wb, 0, CMD_PRE, 0x400, 0)
    time.sleep(0.0001)
    issue_on_phase(wb, 0, CMD_ACT, row, bank)
    time.sleep(0.0002)
    issue_on_phase(wb, 2, CMD_READ, col, bank)
    time.sleep(0.001)
    result = wb.read(pi_base(0) + 0x14)
    issue_on_phase(wb, 0, CMD_PRE, 0x400, bank)
    time.sleep(0.0001)
    wb.write(DFII_CONTROL, 0x0F)
    return result


def clear_row(wb, row, bank):
    """Clear row to all zeros."""
    wb.write(DFII_CONTROL, 0x0E)
    time.sleep(0.001)
    issue_on_phase(wb, 0, CMD_PRE, 0x400, 0)
    time.sleep(0.0001)
    issue_on_phase(wb, 0, CMD_ACT, row, bank)
    time.sleep(0.0002)
    wb.write(pi_base(3) + 0x10, 0x00000000)
    issue_on_phase(wb, 3, CMD_WRITE, 0, bank)
    time.sleep(0.001)
    issue_on_phase(wb, 0, CMD_PRE, 0x400, bank)
    time.sleep(0.0001)
    wb.write(DFII_CONTROL, 0x0F)


def main():
    print("=" * 70)
    print("z1226: Phase-based Sub-cycle Timing")
    print("=" * 70)
    print()
    print("DFI phases give 2.5ns resolution:")
    print("  ACT phase 0, PRE phase 0 = ~0ns (immediate)")
    print("  ACT phase 0, PRE phase 1 = ~2.5ns")
    print("  ACT phase 0, PRE phase 2 = ~5ns")
    print("  ACT phase 0, PRE phase 3 = ~7.5ns")
    print()

    ROW = 700
    BANK = 0
    COL = 0
    PATTERN = 0xFFFFFFFF

    results = {
        "experiment": "z1226_phase_timing",
        "timestamp": datetime.now().isoformat(),
        "phase_timing": []
    }

    print("Connecting to FPGA...")
    wb = RemoteClient(host='localhost', port=1234)
    wb.open()

    print("\nInitializing DDR3...")
    subprocess.run(["python3", "scripts/z1210_ddr3_litex_init.py"],
                   capture_output=True, text=True)
    print("  DDR3 init complete")

    # Test each phase combination
    print("\n" + "=" * 70)
    print("Phase Timing Sweep")
    print("=" * 70)
    print(f"{'ACT Ph':>6} | {'PRE Ph':>6} | {'~Time':>6} | {'Result':>12} | {'Bits':>4}")
    print("-" * 60)

    for act_phase in range(4):
        for pre_phase in range(4):
            # Calculate approximate timing
            if pre_phase >= act_phase:
                timing_ns = (pre_phase - act_phase) * 2.5
            else:
                timing_ns = (4 - act_phase + pre_phase) * 2.5

            # Clear, write, read
            clear_row(wb, ROW, BANK)
            write_with_phase_timing(wb, ROW, COL, BANK, PATTERN, act_phase, pre_phase)
            val = read_data(wb, ROW, COL, BANK)
            bits = count_ones(val)

            print(f"{act_phase:6d} | {pre_phase:6d} | {timing_ns:5.1f}ns | 0x{val:08x} | {bits:4d}")

            results["phase_timing"].append({
                "act_phase": act_phase,
                "pre_phase": pre_phase,
                "timing_ns": timing_ns,
                "result": hex(val),
                "bits": bits
            })

    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    all_bits = sorted(set(d["bits"] for d in results["phase_timing"]))
    print(f"Unique bit counts: {all_bits}")

    # Group by timing
    by_timing = {}
    for d in results["phase_timing"]:
        t = d["timing_ns"]
        if t not in by_timing:
            by_timing[t] = []
        by_timing[t].append(d["bits"])

    print("\nBits by timing:")
    for t in sorted(by_timing.keys()):
        bits = by_timing[t]
        print(f"  {t:5.1f}ns: {bits}")

    # Save
    output_path = "/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/z1226_phase_timing.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    wb.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
