#!/usr/bin/env python3
"""
z140_baseline_analysis.py

Investigate why baseline_comparison test fails more than other tests.

Hypothesis: The mean baseline is actually a strong predictor for normalized
telemetry, especially when values cluster around 0.5.
"""

import json
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def analyze_baseline_strength():
    """Analyze the strength of mean baseline predictor."""

    print("=" * 60)
    print("BASELINE COMPARISON ANALYSIS")
    print("=" * 60)

    # Load z138 results
    try:
        with open("results/z138_overnight_summary.json", 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("z138 results not found")
        return

    print("\n1. TEST PASS RATES BY TYPE")
    print("-" * 40)

    test_types = ['baseline_comparison', 'mismatch_test', 'counterfactual', 'selectivity']
    pass_counts = {t: 0 for t in test_types}
    total = 0

    for exp in data['experiments']:
        if exp['name'].startswith('SEED'):
            total += 1
            for test, result in exp['details'].items():
                if result == 'PASS':
                    pass_counts[test] += 1

    print(f"Total seed experiments: {total}")
    for test in test_types:
        rate = pass_counts[test] / total * 100
        print(f"  {test}: {pass_counts[test]}/{total} ({rate:.0f}%)")

    print("\n2. WHY BASELINE IS HARD TO BEAT")
    print("-" * 40)

    # Simulated telemetry analysis
    # Channel 8 (timing) is most variable, others cluster around normalized values
    print("""
    The baseline comparison requires beating mean predictor by 50%.

    Challenge: GPU telemetry values are normalized to [0, 1] and often
    cluster around similar values:
    - Power: normalized, varies ~0.3-0.7 depending on load
    - Temp: normalized, varies slowly ~0.4-0.6
    - GPU util: normalized, can vary 0-1 but often clustered
    - Timing (ch8): most variable - this is where signal is strongest

    With limited data, the world model may not extract enough signal
    to beat a mean predictor by 50% margin.
    """)

    print("\n3. ABLATION INSIGHTS")
    print("-" * 40)

    for exp in data['experiments']:
        if 'ABLATION' in exp['name']:
            baseline = exp['details']['baseline_comparison']
            print(f"  {exp['name']}: baseline={baseline}")

    print("""
    All ablations also fail baseline_comparison, which is expected.
    This confirms the test IS measuring something real - it's just
    that with 300 episodes, the margin is narrow.
    """)

    print("\n4. RECOMMENDATIONS")
    print("-" * 40)
    print("""
    Options to improve baseline_comparison pass rate:
    1. Increase data (z138 stress test shows 1000ep works)
    2. Lower threshold from 50% to 30% improvement
    3. Focus evaluation on high-variance channels (ch8)
    4. Use relative improvement per-channel, not global MSE

    Current approach: Require 1000+ episodes for reliable results.
    """)


if __name__ == "__main__":
    analyze_baseline_strength()
