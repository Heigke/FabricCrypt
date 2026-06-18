#!/usr/bin/env python3
"""
Z86: Cross-Machine Validation Suite

Runs Z81-style validation on all three machines:
- ikaros (local, AMD APU)
- daedalus (remote, AMD dGPU)
- minos (remote, NVIDIA GPU)

Features:
- Repeated runs with randomized order
- Statistical analysis (mean ± std, CI)
- Phase-separated reporting
- Actuator proof-of-effect

Author: FEEL Research Team
Date: 2026-01-20
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import random
import logging
import argparse
import subprocess
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class MachineConfig:
    """Configuration for a machine."""
    name: str
    hostname: str
    username: str
    password_env: str  # Environment variable containing password
    vendor: str  # "AMD" or "NVIDIA"
    is_local: bool = False
    python_path: str = "python"
    venv_path: str = ""
    hsa_override: str = ""  # HSA_OVERRIDE_GFX_VERSION if needed


@dataclass
class ValidationResult:
    """Result of a single validation run."""
    machine: str
    controller: str
    run_id: int

    # Phase metrics
    prefill_j_per_token: float
    prefill_time_ms: float
    decode_j_per_token: float
    decode_time_ms: float
    decode_tbt_p50_ms: float
    decode_tbt_p95_ms: float
    decode_throughput_tps: float

    # SLO compliance
    slo_violations: int
    slo_violation_rate: float

    # Thermal
    avg_temp_c: float
    max_temp_c: float

    # Energy
    total_energy_j: float
    avg_power_w: float

    # Timestamp
    timestamp: str


@dataclass
class ValidationSummary:
    """Summary statistics for a controller."""
    machine: str
    controller: str
    num_runs: int

    # Mean ± std
    decode_jpt_mean: float
    decode_jpt_std: float
    decode_tbt_p95_mean: float
    decode_tbt_p95_std: float
    throughput_mean: float
    throughput_std: float
    slo_violation_rate_mean: float

    # 95% CI
    decode_jpt_ci_low: float
    decode_jpt_ci_high: float

    # Best/worst
    best_jpt: float
    worst_jpt: float


# Machine configurations
MACHINES = {
    'ikaros': MachineConfig(
        name='ikaros',
        hostname='localhost',
        username='ikaros',
        password_env='',
        vendor='AMD',
        is_local=True,
        venv_path='/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/venv',
        hsa_override='11.0.0',
    ),
    'daedalus': MachineConfig(
        name='daedalus',
        hostname='192.168.0.37',
        username='daedalus',
        password_env='DAEDALUS_PASS',
        vendor='AMD',
        is_local=False,
    ),
    'minos': MachineConfig(
        name='minos',
        hostname='192.168.0.38',
        username='minos',
        password_env='MINOS_PASS',
        vendor='NVIDIA',
        is_local=False,
    ),
}

# Controllers to test
CONTROLLERS = ['disabled', 'fixed_eco', 'fixed_med', 'fixed_perf', 'multiscale']


def run_local_validation(
    machine: MachineConfig,
    controller: str,
    num_tokens: int = 200,
    num_requests: int = 3,
) -> Optional[Dict[str, Any]]:
    """Run validation locally."""
    env = os.environ.copy()
    if machine.hsa_override:
        env['HSA_OVERRIDE_GFX_VERSION'] = machine.hsa_override

    cmd = [
        f"{machine.venv_path}/bin/python" if machine.venv_path else "python",
        "scripts/z81_scientific_validation.py",
        "--controller", controller,
        "--decode-tokens", str(num_tokens),
        "--requests", str(num_requests),
        "--machine", machine.vendor.lower() + ("_apu" if "apu" in machine.name.lower() else "_gpu"),
        "--output", f"results/z86_{machine.name}",
        "--json-only",
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
            env=env,
        )

        if result.returncode != 0:
            logger.error(f"Validation failed: {result.stderr}")
            return None

        # Parse JSON output
        for line in result.stdout.split('\n'):
            if line.startswith('{'):
                return json.loads(line)

        return None

    except Exception as e:
        logger.error(f"Error running validation: {e}")
        return None


def run_remote_validation(
    machine: MachineConfig,
    controller: str,
    num_tokens: int = 200,
    num_requests: int = 3,
) -> Optional[Dict[str, Any]]:
    """Run validation on remote machine via SSH."""
    password = os.environ.get(machine.password_env, '')

    # Build remote command
    remote_cmd = f"""
    cd ~/AMD_gfx1151_energy && \
    source venv/bin/activate && \
    python scripts/z81_scientific_validation.py \
        --controller {controller} \
        --decode-tokens {num_tokens} \
        --requests {num_requests} \
        --machine {machine.vendor.lower()}_gpu \
        --output results/z86_{machine.name} \
        --json-only
    """

    # Use sshpass for password auth (or expect SSH keys)
    if password:
        cmd = [
            "sshpass", "-p", password,
            "ssh", "-o", "StrictHostKeyChecking=no",
            f"{machine.username}@{machine.hostname}",
            remote_cmd,
        ]
    else:
        cmd = [
            "ssh", "-o", "StrictHostKeyChecking=no",
            f"{machine.username}@{machine.hostname}",
            remote_cmd,
        ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
        )

        if result.returncode != 0:
            logger.error(f"Remote validation failed: {result.stderr}")
            return None

        # Parse JSON output
        for line in result.stdout.split('\n'):
            if line.startswith('{'):
                return json.loads(line)

        return None

    except Exception as e:
        logger.error(f"Error running remote validation: {e}")
        return None


def run_validation(
    machine: MachineConfig,
    controller: str,
    num_tokens: int = 200,
    num_requests: int = 3,
) -> Optional[Dict[str, Any]]:
    """Run validation on specified machine."""
    if machine.is_local:
        return run_local_validation(machine, controller, num_tokens, num_requests)
    else:
        return run_remote_validation(machine, controller, num_tokens, num_requests)


def parse_result(raw: Dict[str, Any], machine: str, controller: str, run_id: int) -> Optional[ValidationResult]:
    """Parse raw validation output into ValidationResult."""
    try:
        metrics = raw.get('phase_metrics', raw)

        return ValidationResult(
            machine=machine,
            controller=controller,
            run_id=run_id,
            prefill_j_per_token=metrics.get('prefill_j_per_token', 0),
            prefill_time_ms=metrics.get('prefill_time_ms', 0),
            decode_j_per_token=metrics.get('decode_j_per_token', 0),
            decode_time_ms=metrics.get('decode_time_ms', 0),
            decode_tbt_p50_ms=metrics.get('decode_tbt_p50_ms', 0),
            decode_tbt_p95_ms=metrics.get('decode_tbt_p95_ms', 0),
            decode_throughput_tps=metrics.get('decode_throughput_tps', 0),
            slo_violations=metrics.get('decode_slo_violations', 0),
            slo_violation_rate=metrics.get('decode_slo_violation_rate', 0),
            avg_temp_c=metrics.get('avg_temp_c', 0),
            max_temp_c=metrics.get('max_temp_c', 0),
            total_energy_j=metrics.get('total_energy_j', 0),
            avg_power_w=metrics.get('avg_power_w', 0),
            timestamp=datetime.now().isoformat(),
        )
    except Exception as e:
        logger.error(f"Error parsing result: {e}")
        return None


def compute_summary(results: List[ValidationResult]) -> ValidationSummary:
    """Compute summary statistics for a set of results."""
    if not results:
        raise ValueError("No results to summarize")

    machine = results[0].machine
    controller = results[0].controller

    jpts = [r.decode_j_per_token for r in results]
    tbts = [r.decode_tbt_p95_ms for r in results]
    tpss = [r.decode_throughput_tps for r in results]
    violations = [r.slo_violation_rate for r in results]

    # 95% CI for J/token
    n = len(jpts)
    mean_jpt = np.mean(jpts)
    std_jpt = np.std(jpts, ddof=1) if n > 1 else 0
    ci_margin = 1.96 * std_jpt / np.sqrt(n) if n > 1 else 0

    return ValidationSummary(
        machine=machine,
        controller=controller,
        num_runs=n,
        decode_jpt_mean=mean_jpt,
        decode_jpt_std=std_jpt,
        decode_tbt_p95_mean=np.mean(tbts),
        decode_tbt_p95_std=np.std(tbts, ddof=1) if n > 1 else 0,
        throughput_mean=np.mean(tpss),
        throughput_std=np.std(tpss, ddof=1) if n > 1 else 0,
        slo_violation_rate_mean=np.mean(violations),
        decode_jpt_ci_low=mean_jpt - ci_margin,
        decode_jpt_ci_high=mean_jpt + ci_margin,
        best_jpt=min(jpts),
        worst_jpt=max(jpts),
    )


def run_cross_machine_validation(
    machines: List[str],
    controllers: List[str],
    num_runs: int = 5,
    num_tokens: int = 200,
    num_requests: int = 3,
    randomize: bool = True,
) -> Dict[str, Any]:
    """
    Run full cross-machine validation.

    Args:
        machines: List of machine names to test
        controllers: List of controllers to test
        num_runs: Number of runs per (machine, controller) pair
        num_tokens: Tokens per run
        num_requests: Requests per run
        randomize: Whether to randomize run order

    Returns:
        Dict with all results and summaries
    """
    # Build run schedule
    schedule = []
    for machine_name in machines:
        for controller in controllers:
            for run_id in range(num_runs):
                schedule.append((machine_name, controller, run_id))

    if randomize:
        random.shuffle(schedule)
        logger.info(f"Randomized {len(schedule)} runs")

    # Run all
    all_results: List[ValidationResult] = []

    for i, (machine_name, controller, run_id) in enumerate(schedule):
        logger.info(f"[{i+1}/{len(schedule)}] {machine_name}/{controller} run {run_id+1}")

        machine = MACHINES[machine_name]
        raw = run_validation(machine, controller, num_tokens, num_requests)

        if raw:
            result = parse_result(raw, machine_name, controller, run_id)
            if result:
                all_results.append(result)
                logger.info(f"  J/token: {result.decode_j_per_token:.4f}, TBT p95: {result.decode_tbt_p95_ms:.2f}ms")

        # Small delay between runs
        time.sleep(2)

    # Compute summaries
    summaries = {}
    for machine_name in machines:
        summaries[machine_name] = {}
        for controller in controllers:
            results = [r for r in all_results if r.machine == machine_name and r.controller == controller]
            if results:
                summaries[machine_name][controller] = asdict(compute_summary(results))

    return {
        'results': [asdict(r) for r in all_results],
        'summaries': summaries,
        'config': {
            'machines': machines,
            'controllers': controllers,
            'num_runs': num_runs,
            'num_tokens': num_tokens,
            'num_requests': num_requests,
            'randomized': randomize,
        },
        'timestamp': datetime.now().isoformat(),
    }


def print_summary_table(data: Dict[str, Any]) -> None:
    """Print summary table."""
    summaries = data['summaries']

    print("\n" + "=" * 100)
    print("CROSS-MACHINE VALIDATION SUMMARY")
    print("=" * 100)

    for machine_name, controllers in summaries.items():
        print(f"\n{machine_name.upper()}")
        print("-" * 80)
        print(f"{'Controller':<12} {'J/tok (mean±std)':<20} {'95% CI':<18} {'TBT p95':<12} {'Thru (tps)':<12} {'SLO Viol':<10}")
        print("-" * 80)

        for controller, stats in controllers.items():
            jpt_str = f"{stats['decode_jpt_mean']:.4f}±{stats['decode_jpt_std']:.4f}"
            ci_str = f"[{stats['decode_jpt_ci_low']:.4f}, {stats['decode_jpt_ci_high']:.4f}]"
            tbt_str = f"{stats['decode_tbt_p95_mean']:.2f}±{stats['decode_tbt_p95_std']:.2f}"
            tps_str = f"{stats['throughput_mean']:.1f}±{stats['throughput_std']:.1f}"
            viol_str = f"{stats['slo_violation_rate_mean']*100:.2f}%"

            print(f"{controller:<12} {jpt_str:<20} {ci_str:<18} {tbt_str:<12} {tps_str:<12} {viol_str:<10}")

    print("=" * 100)


def main():
    parser = argparse.ArgumentParser(description="Cross-machine validation suite")
    parser.add_argument('--machines', nargs='+', default=['ikaros'],
                       choices=list(MACHINES.keys()), help='Machines to test')
    parser.add_argument('--controllers', nargs='+', default=CONTROLLERS,
                       help='Controllers to test')
    parser.add_argument('--runs', type=int, default=5, help='Runs per configuration')
    parser.add_argument('--tokens', type=int, default=200, help='Tokens per run')
    parser.add_argument('--requests', type=int, default=3, help='Requests per run')
    parser.add_argument('--no-randomize', action='store_true', help='Disable run order randomization')
    parser.add_argument('--output', type=str, default='results/z86_cross_machine',
                       help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    logger.info(f"Starting cross-machine validation")
    logger.info(f"  Machines: {args.machines}")
    logger.info(f"  Controllers: {args.controllers}")
    logger.info(f"  Runs per config: {args.runs}")

    results = run_cross_machine_validation(
        machines=args.machines,
        controllers=args.controllers,
        num_runs=args.runs,
        num_tokens=args.tokens,
        num_requests=args.requests,
        randomize=not args.no_randomize,
    )

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(args.output, f'cross_machine_{timestamp}.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to: {output_file}")

    # Print summary
    print_summary_table(results)


if __name__ == '__main__':
    main()
