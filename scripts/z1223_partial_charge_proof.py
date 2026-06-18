#!/usr/bin/env python3
"""
z1223: Statistical Proof of Partial Charge

Hypothesis: If tRAS=1 creates partial charge (not just partial bit selection),
then those cells should decay faster than fully charged cells (tRAS=5).

Experiment Design:
1. Write 0xFFFFFFFF with tRAS=1 to rows 100-109 (partial charge)
2. Write 0xFFFFFFFF with tRAS=5 to rows 200-209 (full charge)
3. Keep in software mode (no refresh)
4. Read back at intervals: 0s, 1s, 2s, 5s, 10s, 20s, 30s, 60s
5. Compare decay curves

If partial charge is real:
- Partial rows should show bit flips (1→0) earlier/more than full rows
- Decay rate should be measurably different

If it's just timing-based bit selection:
- Both should decay at similar rates (same starting charge, just fewer bits)
"""

import time
import json
import subprocess
from datetime import datetime
from collections import Counter
from litex.tools.litex_client import RemoteClient

# CSRs (relative addresses - litex_server handles base)
DFII_CONTROL = 0x3000
PHASE_SIZE = 0x18
def pi_base(phase): return 0x3004 + phase * PHASE_SIZE

PW_CONFIG = 0x2800
PW_WRITE_DATA = 0x2804
PW_CONTROL = 0x280c
PW_STATUS = 0x2810
PW_RESULT = 0x2814


def make_config(row, col, bank, tras):
    return (row << 18) | (col << 8) | (bank << 5) | tras


def sw_write(wb, row, col, bank, data):
    """Software mode write via DFII phase registers."""
    WRPHASE = 3
    wb.write(DFII_CONTROL, 0x0E)  # SW mode
    time.sleep(0.001)
    # Precharge all
    wb.write(pi_base(0) + 0x08, 0x400)
    wb.write(pi_base(0) + 0x0c, 0)
    wb.write(pi_base(0) + 0x00, 0x0B)
    wb.write(pi_base(0) + 0x04, 1)
    time.sleep(0.0001)
    # Activate row
    wb.write(pi_base(0) + 0x08, row)
    wb.write(pi_base(0) + 0x0c, bank)
    wb.write(pi_base(0) + 0x00, 0x09)
    wb.write(pi_base(0) + 0x04, 1)
    time.sleep(0.0001)
    # Write data on WRPHASE
    wb.write(pi_base(WRPHASE) + 0x10, data)
    wb.write(pi_base(WRPHASE) + 0x08, col)
    wb.write(pi_base(WRPHASE) + 0x0c, bank)
    wb.write(pi_base(WRPHASE) + 0x00, 0x17)
    wb.write(pi_base(WRPHASE) + 0x04, 1)
    time.sleep(0.001)
    # Precharge
    wb.write(pi_base(0) + 0x08, 0x400)
    wb.write(pi_base(0) + 0x0c, 0)
    wb.write(pi_base(0) + 0x00, 0x0B)
    wb.write(pi_base(0) + 0x04, 1)
    time.sleep(0.0001)
    wb.write(DFII_CONTROL, 0x0F)  # HW mode


def sw_read(wb, row, col, bank):
    """Software mode read via DFII phase registers."""
    RDPHASE = 2
    wb.write(DFII_CONTROL, 0x0E)  # SW mode
    time.sleep(0.001)
    # Precharge all
    wb.write(pi_base(0) + 0x08, 0x400)
    wb.write(pi_base(0) + 0x0c, 0)
    wb.write(pi_base(0) + 0x00, 0x0B)
    wb.write(pi_base(0) + 0x04, 1)
    time.sleep(0.0001)
    # Activate row
    wb.write(pi_base(0) + 0x08, row)
    wb.write(pi_base(0) + 0x0c, bank)
    wb.write(pi_base(0) + 0x00, 0x09)
    wb.write(pi_base(0) + 0x04, 1)
    time.sleep(0.0001)
    # Read on RDPHASE
    wb.write(pi_base(RDPHASE) + 0x08, col)
    wb.write(pi_base(RDPHASE) + 0x0c, bank)
    wb.write(pi_base(RDPHASE) + 0x00, 0x25)
    wb.write(pi_base(RDPHASE) + 0x04, 1)
    time.sleep(0.001)
    # Capture result
    result = wb.read(pi_base(0) + 0x14)
    # Precharge
    wb.write(pi_base(0) + 0x08, 0x400)
    wb.write(pi_base(0) + 0x0c, 0)
    wb.write(pi_base(0) + 0x00, 0x0B)
    wb.write(pi_base(0) + 0x04, 1)
    time.sleep(0.0001)
    wb.write(DFII_CONTROL, 0x0F)  # HW mode
    return result


def run_fsm(wb, row, col, bank, tras, write_data):
    """Run partial write FSM with configurable tRAS."""
    config = make_config(row, col, bank, tras)
    wb.write(PW_CONFIG, config)
    wb.write(PW_WRITE_DATA, write_data)
    wb.write(PW_CONTROL, 0)
    time.sleep(0.001)
    wb.write(PW_CONTROL, 1)
    for _ in range(100):
        status = wb.read(PW_STATUS)
        if (status >> 1) & 1:
            break
        time.sleep(0.001)
    result = wb.read(PW_RESULT)
    wb.write(PW_CONTROL, 0)
    return result


def count_ones(val):
    """Count number of 1 bits."""
    return bin(val).count('1')


def disable_refresh(wb):
    """Keep DFII in software mode to prevent auto-refresh."""
    wb.write(DFII_CONTROL, 0x0E)  # sel=0 (SW mode), cke=1, reset_n=1
    print(f"  DFII control set to 0x{wb.read(DFII_CONTROL):02x} (refresh disabled)")


def enable_refresh(wb):
    """Return to hardware mode with auto-refresh."""
    wb.write(DFII_CONTROL, 0x0F)  # sel=1 (HW mode)
    print(f"  DFII control set to 0x{wb.read(DFII_CONTROL):02x} (refresh enabled)")


def main():
    print("=" * 70)
    print("z1223: Statistical Proof of Partial Charge")
    print("=" * 70)
    print()

    # Experiment parameters
    PARTIAL_ROWS = list(range(100, 110))  # 10 rows with tRAS=1 (partial)
    FULL_ROWS = list(range(200, 210))      # 10 rows with tRAS=5 (full)
    BANK = 0
    COL = 0
    PATTERN = 0xFFFFFFFF

    # Decay measurement intervals (seconds)
    INTERVALS = [0, 1, 2, 5, 10, 20, 30, 60]

    results = {
        "experiment": "z1223_partial_charge_proof",
        "timestamp": datetime.now().isoformat(),
        "hypothesis": "Partial charge (tRAS=1) should decay faster than full charge (tRAS=5)",
        "partial_tras": 1,
        "full_tras": 5,
        "partial_rows": PARTIAL_ROWS,
        "full_rows": FULL_ROWS,
        "pattern": hex(PATTERN),
        "intervals_sec": INTERVALS,
        "decay_data": []
    }

    print("Connecting to FPGA...")
    wb = RemoteClient(host='localhost', port=1234)
    wb.open()

    # Initialize DDR3
    print("\nInitializing DDR3...")
    subprocess.run(["python3", "scripts/z1210_ddr3_litex_init.py"],
                   capture_output=True, text=True)
    print("  DDR3 init complete")

    # Phase 1: Write FULL charge rows first (tRAS=5)
    print()
    print("Phase 1: Writing FULL charge rows (tRAS=5)...")
    for row in FULL_ROWS:
        # Clear first
        sw_write(wb, row, COL, BANK, 0x00000000)
        # Write with full tRAS
        run_fsm(wb, row, COL, BANK, tras=5, write_data=PATTERN)
        readback = sw_read(wb, row, COL, BANK)
        ones = count_ones(readback)
        print(f"  Row {row}: 0x{readback:08x} ({ones} bits)")

    # Phase 2: Write PARTIAL charge rows (tRAS=1)
    print()
    print("Phase 2: Writing PARTIAL charge rows (tRAS=1)...")
    for row in PARTIAL_ROWS:
        # Clear first
        sw_write(wb, row, COL, BANK, 0x00000000)
        # Write with partial tRAS
        run_fsm(wb, row, COL, BANK, tras=1, write_data=PATTERN)
        readback = sw_read(wb, row, COL, BANK)
        ones = count_ones(readback)
        print(f"  Row {row}: 0x{readback:08x} ({ones} bits)")

    # Phase 3: Disable refresh and measure decay
    print()
    print("Phase 3: Disabling refresh and measuring decay...")
    disable_refresh(wb)
    print("-" * 70)

    start_time = time.time()
    last_interval = 0

    for interval in INTERVALS:
        # Wait until this interval
        wait_time = interval - last_interval
        if wait_time > 0:
            print(f"  Waiting {wait_time:.0f}s...")
            time.sleep(wait_time)
        last_interval = interval

        elapsed = time.time() - start_time

        # Read all partial rows (need to temporarily use SW commands)
        partial_bits = []
        partial_vals = []
        for row in PARTIAL_ROWS:
            # Manual read without triggering refresh
            wb.write(pi_base(0) + 0x08, row)
            wb.write(pi_base(0) + 0x0c, BANK)
            wb.write(pi_base(0) + 0x00, 0x09)  # ACT
            wb.write(pi_base(0) + 0x04, 1)
            time.sleep(0.0001)
            wb.write(pi_base(2) + 0x08, COL)
            wb.write(pi_base(2) + 0x0c, BANK)
            wb.write(pi_base(2) + 0x00, 0x25)  # READ
            wb.write(pi_base(2) + 0x04, 1)
            time.sleep(0.001)
            val = wb.read(pi_base(0) + 0x14)
            wb.write(pi_base(0) + 0x08, 0x400)
            wb.write(pi_base(0) + 0x0c, 0)
            wb.write(pi_base(0) + 0x00, 0x0B)  # PRE
            wb.write(pi_base(0) + 0x04, 1)
            time.sleep(0.0001)
            partial_bits.append(count_ones(val))
            partial_vals.append(val)

        # Read all full rows
        full_bits = []
        full_vals = []
        for row in FULL_ROWS:
            wb.write(pi_base(0) + 0x08, row)
            wb.write(pi_base(0) + 0x0c, BANK)
            wb.write(pi_base(0) + 0x00, 0x09)
            wb.write(pi_base(0) + 0x04, 1)
            time.sleep(0.0001)
            wb.write(pi_base(2) + 0x08, COL)
            wb.write(pi_base(2) + 0x0c, BANK)
            wb.write(pi_base(2) + 0x00, 0x25)
            wb.write(pi_base(2) + 0x04, 1)
            time.sleep(0.001)
            val = wb.read(pi_base(0) + 0x14)
            wb.write(pi_base(0) + 0x08, 0x400)
            wb.write(pi_base(0) + 0x0c, 0)
            wb.write(pi_base(0) + 0x00, 0x0B)
            wb.write(pi_base(0) + 0x04, 1)
            time.sleep(0.0001)
            full_bits.append(count_ones(val))
            full_vals.append(val)

        partial_avg = sum(partial_bits) / len(partial_bits)
        full_avg = sum(full_bits) / len(full_bits)

        measurement = {
            "target_interval": interval,
            "actual_elapsed": round(elapsed, 3),
            "partial_bits": partial_bits,
            "partial_avg": round(partial_avg, 2),
            "partial_values": [hex(v) for v in partial_vals],
            "full_bits": full_bits,
            "full_avg": round(full_avg, 2),
            "full_values": [hex(v) for v in full_vals],
            "difference": round(full_avg - partial_avg, 2)
        }
        results["decay_data"].append(measurement)

        print(f"  t={interval:3.0f}s: Partial avg={partial_avg:5.1f} bits, Full avg={full_avg:5.1f} bits, Δ={full_avg-partial_avg:+5.1f}")

    # Re-enable refresh
    print()
    enable_refresh(wb)

    # Analysis
    print()
    print("=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    # Initial state
    initial_partial = results["decay_data"][0]["partial_avg"]
    initial_full = results["decay_data"][0]["full_avg"]

    # Final state
    final_partial = results["decay_data"][-1]["partial_avg"]
    final_full = results["decay_data"][-1]["full_avg"]

    # Decay amounts
    partial_decay = initial_partial - final_partial
    full_decay = initial_full - final_full

    print(f"Initial state (t=0):")
    print(f"  Partial (tRAS=1): {initial_partial:.1f} bits")
    print(f"  Full (tRAS=5):    {initial_full:.1f} bits")
    print()
    print(f"Final state (t={INTERVALS[-1]}s):")
    print(f"  Partial (tRAS=1): {final_partial:.1f} bits")
    print(f"  Full (tRAS=5):    {final_full:.1f} bits")
    print()
    print(f"Decay amounts:")
    print(f"  Partial: {partial_decay:.1f} bits lost")
    print(f"  Full:    {full_decay:.1f} bits lost")
    print()

    # Statistical test: did partial decay more?
    if initial_partial > 0 and initial_full > 0:
        partial_decay_pct = (partial_decay / initial_partial) * 100 if initial_partial > 0 else 0
        full_decay_pct = (full_decay / initial_full) * 100 if initial_full > 0 else 0

        print(f"Decay percentages:")
        print(f"  Partial: {partial_decay_pct:.1f}% of initial")
        print(f"  Full:    {full_decay_pct:.1f}% of initial")
        print()

        results["analysis"] = {
            "initial_partial_bits": initial_partial,
            "initial_full_bits": initial_full,
            "final_partial_bits": final_partial,
            "final_full_bits": final_full,
            "partial_decay_bits": partial_decay,
            "full_decay_bits": full_decay,
            "partial_decay_pct": round(partial_decay_pct, 2),
            "full_decay_pct": round(full_decay_pct, 2)
        }

        # Verdict
        if partial_decay_pct > full_decay_pct + 10:
            verdict = "PARTIAL CHARGE CONFIRMED - partial rows decay faster!"
            results["verdict"] = "CONFIRMED"
        elif abs(partial_decay_pct - full_decay_pct) < 10 and partial_decay == 0 and full_decay == 0:
            verdict = "NO DECAY OBSERVED - room temp retention too long, need heat or longer wait"
            results["verdict"] = "NO_DECAY"
        elif abs(partial_decay_pct - full_decay_pct) < 10:
            verdict = "INCONCLUSIVE - decay rates similar (may be bit selection, not charge)"
            results["verdict"] = "INCONCLUSIVE"
        else:
            verdict = "UNEXPECTED - full rows decayed more (check experiment)"
            results["verdict"] = "UNEXPECTED"

        print(f"VERDICT: {verdict}")
        results["verdict_detail"] = verdict
    else:
        print("ERROR: Initial bits = 0, cannot analyze")
        results["verdict"] = "ERROR"

    # Save results
    output_path = "/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/z1223_partial_charge_proof.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print()
    print(f"Results saved to: {output_path}")

    wb.close()
    print()
    print("Done!")


if __name__ == "__main__":
    main()
