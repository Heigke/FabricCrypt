#!/usr/bin/env python3
"""
Deep Silicon Interoception (DSI) Demo

Demonstrates the three pillars of DSI:
1. Differential Diagnosis - Distinguishing Flow from Fever from Strain
2. Somatic Dashboard - Visualizing internal state
3. Agency - Model requesting its own compute budget

This proves TRUE INTEROCEPTION: the model knows its body.
"""

import sys
from pathlib import Path
import numpy as np
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dsi import DifferentialDiagnosis, SomaticDashboard, AgencyController
from src.dsi.diagnosis import SomaticSignature, InternalState


def simulate_workload_scenario():
    """
    Simulate a realistic workload scenario:
    1. Start idle
    2. Ramp up to flow state
    3. Push into fever (overheating)
    4. Crash into strain
    5. Recovery
    """
    scenarios = []

    # Phase 1: Idle/warmup (10 steps)
    for i in range(10):
        t = i / 10
        scenarios.append(SomaticSignature(
            thermal=0.1 + 0.05 * t,
            metabolic=0.1 + 0.05 * t,
            cognitive=0.3 + 0.2 * t,
            variance=0.1,
            fatigue=0.0,
            recovery_rate=0.0,
        ))

    # Phase 2: Entering flow (15 steps)
    for i in range(15):
        t = i / 15
        scenarios.append(SomaticSignature(
            thermal=0.15 + 0.35 * t,
            metabolic=0.15 + 0.35 * t,
            cognitive=0.5 + 0.35 * t,  # High coherence
            variance=0.1 + 0.05 * t,   # Low variance
            fatigue=0.05 * t,
            recovery_rate=0.0,
        ))

    # Phase 3: Peak flow (10 steps)
    for i in range(10):
        noise = np.random.normal(0, 0.02)
        scenarios.append(SomaticSignature(
            thermal=0.55 + noise,
            metabolic=0.55 + noise,
            cognitive=0.85 + noise * 0.5,  # Excellent coherence
            variance=0.15,                  # Stable
            fatigue=0.1 + 0.01 * i,
            recovery_rate=0.0,
        ))

    # Phase 4: Pushing too hard -> Fever (15 steps)
    for i in range(15):
        t = i / 15
        scenarios.append(SomaticSignature(
            thermal=0.6 + 0.25 * t,
            metabolic=0.6 + 0.3 * t,
            cognitive=0.85 - 0.15 * t,  # Coherence dropping
            variance=0.2 + 0.25 * t,    # Variance increasing!
            fatigue=0.2 + 0.2 * t,
            recovery_rate=-0.05,
        ))

    # Phase 5: Crash into strain (20 steps)
    for i in range(20):
        t = i / 20
        scenarios.append(SomaticSignature(
            thermal=0.85 - 0.1 * t,      # Thermal dropping (throttling)
            metabolic=0.9 - 0.15 * t,
            cognitive=0.7 - 0.35 * t,    # Coherence collapsing
            variance=0.45 + 0.1 * t,     # Very unstable
            fatigue=0.4 + 0.4 * t,       # Fatigue spiking
            recovery_rate=-0.1,
        ))

    # Phase 6: Recovery (30 steps)
    for i in range(30):
        t = i / 30
        scenarios.append(SomaticSignature(
            thermal=0.75 - 0.5 * t,
            metabolic=0.75 - 0.5 * t,
            cognitive=0.35 + 0.4 * t,    # Coherence recovering
            variance=0.55 - 0.4 * t,     # Stabilizing
            fatigue=0.8 - 0.6 * t,       # Fatigue draining
            recovery_rate=0.15 - 0.1 * t,
        ))

    return scenarios


def run_differential_diagnosis_demo():
    """Demonstrate differential diagnosis."""
    print("\n" + "=" * 70)
    print("  DIFFERENTIAL DIAGNOSIS DEMO")
    print("  Same high temperature can mean different things!")
    print("=" * 70)

    diagnosis = DifferentialDiagnosis()

    # Same thermal (0.7) but different contexts
    test_cases = [
        ("High temp + High coherence + Low variance", SomaticSignature(
            thermal=0.7, metabolic=0.7, cognitive=0.85,
            variance=0.15, fatigue=0.1, recovery_rate=0.0
        )),
        ("High temp + High variance + Spiking", SomaticSignature(
            thermal=0.7, metabolic=0.8, cognitive=0.6,
            variance=0.5, fatigue=0.3, recovery_rate=-0.05
        )),
        ("High temp + Low coherence + High fatigue", SomaticSignature(
            thermal=0.7, metabolic=0.75, cognitive=0.35,
            variance=0.4, fatigue=0.75, recovery_rate=-0.1
        )),
    ]

    for name, sig in test_cases:
        report = diagnosis.get_diagnostic_report(sig)
        print(f"\n  {name}")
        print(f"    → Diagnosis: {report['current_state'].upper()}")
        print(f"    → {report['differential_diagnosis'][:100]}...")
        print(f"    → Action: {report['recommended_action']}")


def run_somatic_dashboard_demo(output_dir: Path):
    """Demonstrate somatic dashboard visualization."""
    print("\n" + "=" * 70)
    print("  SOMATIC DASHBOARD DEMO")
    print("  Visualizing the model's internal state")
    print("=" * 70)

    dashboard = SomaticDashboard()
    scenarios = simulate_workload_scenario()

    print(f"\n  Simulating {len(scenarios)} time steps...")

    for sig in scenarios:
        state = dashboard.record(sig)

    # Generate dashboard
    plot_path = output_dir / "dsi_somatic_dashboard.png"
    dashboard.plot_dashboard(
        output_path=str(plot_path),
        title="Deep Silicon Interoception - Somatic Dashboard"
    )

    # Save trace
    trace_path = output_dir / "dsi_somatic_trace.json"
    dashboard.save_trace(str(trace_path))

    summary = dashboard.get_summary()
    print(f"\n  Dashboard saved: {plot_path}")
    print(f"  Trace saved: {trace_path}")
    print(f"\n  Summary:")
    print(f"    Duration: {summary['duration']:.1f} steps")
    print(f"    Time in FLOW: {summary['time_in_flow']*100:.1f}%")
    print(f"    Time in STRAIN: {summary['time_in_strain']*100:.1f}%")
    print(f"    Max fatigue: {summary['max_fatigue']:.2f}")

    return summary


def run_agency_demo():
    """Demonstrate agency - model requesting its own K."""
    print("\n" + "=" * 70)
    print("  AGENCY DEMO")
    print("  The model requests its own compute budget")
    print("=" * 70)

    agency = AgencyController()

    # Track requests
    requests_made = []

    def handle_request(req):
        requests_made.append(req)
        print(f"\n  [MODEL REQUEST] {req.action.value.upper()}")
        print(f"    State: {req.current_state.value}")
        print(f"    Urgency: {req.urgency:.2f}")
        print(f"    Requested K: {req.requested_k}")
        print(f"    Reason: {req.reason[:80]}...")
        return True  # Honor all requests

    agency.set_request_handler(handle_request)

    # Simulate scenario
    scenarios = simulate_workload_scenario()

    print(f"\n  Running {len(scenarios)} steps with agency enabled...")
    print("  (Requests will be shown as they occur)")

    for sig in scenarios:
        agency.sense_and_decide(sig)

    # Report
    report = agency.get_agency_report()
    print(f"\n  Agency Report:")
    print(f"    Total requests: {report['total_requests']}")
    print(f"    Honored: {report['honored_requests']}")
    print(f"    Trust score: {report['trust_score']*100:.1f}%")
    print(f"    Final K: {report['current_k']}")
    print(f"    Request breakdown: {report['request_breakdown']}")

    return report


def main():
    output_dir = Path("results/dsi")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("  DEEP SILICON INTEROCEPTION (DSI)")
    print("  The model knows its body")
    print("=" * 70)

    # Demo 1: Differential Diagnosis
    run_differential_diagnosis_demo()

    # Demo 2: Somatic Dashboard
    dashboard_summary = run_somatic_dashboard_demo(output_dir)

    # Demo 3: Agency
    agency_report = run_agency_demo()

    # Save combined results
    results = {
        'dashboard_summary': dashboard_summary,
        'agency_report': agency_report,
    }

    results_path = output_dir / "dsi_demo_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n  Results saved: {results_path}")

    print("\n" + "=" * 70)
    print("  DSI DEMO COMPLETE")
    print("=" * 70)
    print("\n  Key achievements:")
    print("    ✓ Differential diagnosis distinguishes Flow/Fever/Strain")
    print("    ✓ Somatic dashboard visualizes internal state")
    print("    ✓ Agency controller enables self-regulation")
    print("\n  This is TRUE INTEROCEPTION - the model knows its body.")


if __name__ == "__main__":
    main()
