#!/usr/bin/env python3
"""
z1228: Statistical Charge Level Inference (Corrected for Destructive Reads)

DRAM reads are destructive - sense amp restores cell to full 0 or 1.
So we can't read the same charge state twice.

Approach: PROBABILISTIC INFERENCE
1. Repeat the same partial write N times (fresh each trial)
2. Read once after each write
3. Count how many times it reads as 1 vs 0
4. The flip PROBABILITY reveals the charge level

If charge is near threshold:  ~50% flip rate
If charge is well above:      ~100% flip rate
If charge is below:           ~0% flip rate

We can also vary:
- tRAS (timing)
- Number of frac operations before read
- Decay time before read

This gives us EFFECTIVE analog sensing through statistics!
"""

import time
import json
import subprocess
from datetime import datetime
from collections import Counter
from litex.tools.litex_client import RemoteClient

# CSRs
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
    """Full software write."""
    WRPHASE = 3
    wb.write(DFII_CONTROL, 0x0E)
    time.sleep(0.001)
    wb.write(pi_base(0) + 0x08, 0x400)
    wb.write(pi_base(0) + 0x0c, 0)
    wb.write(pi_base(0) + 0x00, 0x0B)
    wb.write(pi_base(0) + 0x04, 1)
    time.sleep(0.0001)
    wb.write(pi_base(0) + 0x08, row)
    wb.write(pi_base(0) + 0x0c, bank)
    wb.write(pi_base(0) + 0x00, 0x09)
    wb.write(pi_base(0) + 0x04, 1)
    time.sleep(0.0001)
    wb.write(pi_base(WRPHASE) + 0x10, data)
    wb.write(pi_base(WRPHASE) + 0x08, col)
    wb.write(pi_base(WRPHASE) + 0x0c, bank)
    wb.write(pi_base(WRPHASE) + 0x00, 0x17)
    wb.write(pi_base(WRPHASE) + 0x04, 1)
    time.sleep(0.001)
    wb.write(pi_base(0) + 0x08, 0x400)
    wb.write(pi_base(0) + 0x0c, 0)
    wb.write(pi_base(0) + 0x00, 0x0B)
    wb.write(pi_base(0) + 0x04, 1)
    time.sleep(0.0001)
    wb.write(DFII_CONTROL, 0x0F)


def sw_read(wb, row, col, bank):
    """Software read (destructive - restores cell)."""
    RDPHASE = 2
    wb.write(DFII_CONTROL, 0x0E)
    time.sleep(0.0005)
    wb.write(pi_base(0) + 0x08, 0x400)
    wb.write(pi_base(0) + 0x0c, 0)
    wb.write(pi_base(0) + 0x00, 0x0B)
    wb.write(pi_base(0) + 0x04, 1)
    time.sleep(0.0001)
    wb.write(pi_base(0) + 0x08, row)
    wb.write(pi_base(0) + 0x0c, bank)
    wb.write(pi_base(0) + 0x00, 0x09)
    wb.write(pi_base(0) + 0x04, 1)
    time.sleep(0.0001)
    wb.write(pi_base(RDPHASE) + 0x08, col)
    wb.write(pi_base(RDPHASE) + 0x0c, bank)
    wb.write(pi_base(RDPHASE) + 0x00, 0x25)
    wb.write(pi_base(RDPHASE) + 0x04, 1)
    time.sleep(0.0005)
    result = wb.read(pi_base(0) + 0x14)
    wb.write(pi_base(0) + 0x08, 0x400)
    wb.write(pi_base(0) + 0x0c, 0)
    wb.write(pi_base(0) + 0x00, 0x0B)
    wb.write(pi_base(0) + 0x04, 1)
    time.sleep(0.0001)
    wb.write(DFII_CONTROL, 0x0F)
    return result


def run_fsm(wb, row, col, bank, tras, write_data):
    """Run partial write FSM."""
    config = make_config(row, col, bank, tras)
    wb.write(PW_CONFIG, config)
    wb.write(PW_WRITE_DATA, write_data)
    wb.write(PW_CONTROL, 0)
    time.sleep(0.0005)
    wb.write(PW_CONTROL, 1)
    for _ in range(50):
        status = wb.read(PW_STATUS)
        if (status >> 1) & 1:
            break
        time.sleep(0.001)
    wb.write(PW_CONTROL, 0)


def count_ones(val):
    return bin(val).count('1')


def measure_flip_probability(wb, row, col, bank, tras, num_trials=50):
    """
    Measure the probability of bits flipping with given tRAS.

    For each trial:
    1. Clear cell to 0
    2. Write 0xFFFFFFFF with specified tRAS
    3. Read back
    4. Count 1s

    Returns: list of bit counts from each trial
    """
    bit_counts = []
    for _ in range(num_trials):
        # Clear to 0
        sw_write(wb, row, col, bank, 0x00000000)
        # Partial write
        run_fsm(wb, row, col, bank, tras, 0xFFFFFFFF)
        # Read (destructive)
        val = sw_read(wb, row, col, bank)
        bit_counts.append(count_ones(val))
    return bit_counts


def measure_multi_frac_probability(wb, row, col, bank, num_fracs, num_trials=50):
    """
    Measure flip probability after multiple frac operations.

    Theory: More fracs = more charge accumulation = higher flip probability
    """
    bit_counts = []
    for _ in range(num_trials):
        # Clear to 0
        sw_write(wb, row, col, bank, 0x00000000)
        # Do N frac operations (tRAS=0 for fastest)
        for _ in range(num_fracs):
            run_fsm(wb, row, col, bank, tras=0, write_data=0xFFFFFFFF)
        # Read
        val = sw_read(wb, row, col, bank)
        bit_counts.append(count_ones(val))
    return bit_counts


def main():
    print("=" * 70)
    print("z1228: Statistical Charge Inference (Probabilistic)")
    print("=" * 70)
    print()
    print("Since reads are destructive, we use PROBABILITY over many trials")
    print("to infer charge levels.")
    print()

    ROW = 900
    BANK = 0
    COL = 0
    NUM_TRIALS = 30

    results = {
        "experiment": "z1228_statistical_charge_sensing",
        "timestamp": datetime.now().isoformat(),
        "num_trials": NUM_TRIALS,
        "tras_sweep": [],
        "frac_accumulation": [],
    }

    print("Connecting to FPGA...")
    wb = RemoteClient(host='localhost', port=1234)
    wb.open()

    print("\nInitializing DDR3...")
    subprocess.run(["python3", "scripts/z1210_ddr3_litex_init.py"],
                   capture_output=True, text=True)
    print("  DDR3 init complete")

    # Test 1: tRAS sweep - how does timing affect flip probability?
    print("\n" + "=" * 70)
    print(f"Test 1: tRAS Sweep ({NUM_TRIALS} trials each)")
    print("=" * 70)
    print(f"{'tRAS':>5} | {'Mean bits':>10} | {'Std':>6} | {'Min':>4} | {'Max':>4} | Distribution")
    print("-" * 70)

    for tras in range(6):
        bit_counts = measure_flip_probability(wb, ROW, COL, BANK, tras, NUM_TRIALS)

        mean_bits = sum(bit_counts) / len(bit_counts)
        min_bits = min(bit_counts)
        max_bits = max(bit_counts)

        # Calculate std dev
        variance = sum((x - mean_bits) ** 2 for x in bit_counts) / len(bit_counts)
        std_bits = variance ** 0.5

        # Distribution
        counter = Counter(bit_counts)
        dist_str = ", ".join(f"{k}:{v}" for k, v in sorted(counter.items())[:5])

        print(f"{tras:5d} | {mean_bits:10.1f} | {std_bits:6.2f} | {min_bits:4d} | {max_bits:4d} | {dist_str}")

        results["tras_sweep"].append({
            "tras": tras,
            "bit_counts": bit_counts,
            "mean": round(mean_bits, 2),
            "std": round(std_bits, 2),
            "min": min_bits,
            "max": max_bits
        })

    # Test 2: Frac accumulation - multiple fast writes accumulate charge?
    print("\n" + "=" * 70)
    print(f"Test 2: Frac Accumulation ({NUM_TRIALS} trials each)")
    print("=" * 70)
    print("Theory: More fracs = more charge = higher flip probability")
    print(f"{'Fracs':>5} | {'Mean bits':>10} | {'Std':>6} | {'Min':>4} | {'Max':>4}")
    print("-" * 50)

    for num_fracs in [1, 2, 3, 4, 5, 6, 8, 10, 15, 20]:
        bit_counts = measure_multi_frac_probability(wb, ROW, COL, BANK, num_fracs, NUM_TRIALS)

        mean_bits = sum(bit_counts) / len(bit_counts)
        min_bits = min(bit_counts)
        max_bits = max(bit_counts)
        variance = sum((x - mean_bits) ** 2 for x in bit_counts) / len(bit_counts)
        std_bits = variance ** 0.5

        print(f"{num_fracs:5d} | {mean_bits:10.1f} | {std_bits:6.2f} | {min_bits:4d} | {max_bits:4d}")

        results["frac_accumulation"].append({
            "num_fracs": num_fracs,
            "bit_counts": bit_counts,
            "mean": round(mean_bits, 2),
            "std": round(std_bits, 2)
        })

    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    # Check if there's variation in tRAS sweep
    tras_means = [d["mean"] for d in results["tras_sweep"]]
    tras_stds = [d["std"] for d in results["tras_sweep"]]

    print("\ntRAS sweep analysis:")
    print(f"  Mean bits by tRAS: {tras_means}")
    print(f"  Std devs: {tras_stds}")

    # Check frac accumulation
    frac_means = [d["mean"] for d in results["frac_accumulation"]]
    print(f"\nFrac accumulation means: {frac_means}")

    # Verdict
    if max(tras_stds) > 2:
        print("\n  → High variance detected!")
        print("  → This suggests we're near threshold - PROBABILISTIC CHARGE SENSING WORKS!")
        results["verdict"] = "PROBABILISTIC_SENSING_WORKS"
    elif frac_means[-1] > frac_means[0] + 5:
        print("\n  → Frac accumulation shows charge buildup!")
        results["verdict"] = "FRAC_ACCUMULATION_WORKS"
    else:
        print("\n  → Low variance - results are deterministic")
        print("  → This confirms bit selection, not charge level variation")
        results["verdict"] = "DETERMINISTIC"

    # Save
    output_path = "/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/z1228_statistical_charge_sensing.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    wb.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
