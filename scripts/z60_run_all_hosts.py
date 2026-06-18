#!/usr/bin/env python3
"""
Z60: Run Metabolic Experiment Across All Hosts
==============================================
Launches experiments on ikaros, daedalus, and minos in parallel,
then aggregates results for cross-body comparison.

Usage:
    python scripts/z60_run_all_hosts.py
    python scripts/z60_run_all_hosts.py --local-only  # Just ikaros
    python scripts/z60_run_all_hosts.py --quick       # Reduced epochs

Author: FEEL Research Team
Date: 2026-01-19
"""

import os
import sys
import json
import time
import subprocess
import threading
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
)
logger = logging.getLogger(__name__)

# Host configurations
HOSTS = {
    'ikaros': {
        'host': 'localhost',  # Local
        'user': 'ikaros',
        'gpu_type': 'AMD gfx1151',
        'vram_gb': 96,
    },
    'daedalus': {
        'host': '192.168.0.37',
        'user': 'daedalus',
        'password': 'daedalus',
        'gpu_type': 'AMD gfx1151',
        'vram_gb': 96,
    },
    'minos': {
        'host': '192.168.0.38',
        'user': 'minos',
        'password': 'minos',
        'gpu_type': 'NVIDIA RTX A6000 x4',
        'vram_gb': 192,
    },
}

PROJECT_DIR = Path(__file__).parent.parent
EXPERIMENT_SCRIPT = "scripts/z60_metabolic_experiment.py"


def run_local_experiment(host: str, args: Dict) -> Dict:
    """Run experiment locally (for ikaros)."""
    logger.info(f"Starting local experiment on {host}")

    cmd = [
        sys.executable,
        str(PROJECT_DIR / EXPERIMENT_SCRIPT),
        f"--host={host}",
        f"--hidden-dim={args.get('hidden_dim', 256)}",
        f"--num-layers={args.get('num_layers', 6)}",
        f"--lm-epochs={args.get('lm_epochs', 5)}",
        f"--rl-epochs={args.get('rl_epochs', 3)}",
        f"--corpus-size={args.get('corpus_size', 500000)}",
        f"--eval-samples={args.get('eval_samples', 100)}",
    ]

    env = os.environ.copy()
    env['PYTHONPATH'] = str(PROJECT_DIR)

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        cwd=str(PROJECT_DIR),
    )

    # Stream output
    output_lines = []
    for line in process.stdout:
        output_lines.append(line)
        logger.info(f"[{host}] {line.rstrip()}")

    process.wait()

    return {
        'host': host,
        'returncode': process.returncode,
        'output': ''.join(output_lines),
    }


def run_remote_experiment(host: str, config: Dict, args: Dict) -> Dict:
    """Run experiment on remote host via SSH."""
    logger.info(f"Starting remote experiment on {host} ({config['host']})")

    # Create experiment script to run remotely
    remote_script = f"""
cd ~/AMD_gfx1151_energy 2>/dev/null || cd /tmp
git clone https://github.com/user/AMD_gfx1151_energy.git 2>/dev/null || true
cd AMD_gfx1151_energy 2>/dev/null || true

# Try to use existing venv or create one
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Install dependencies
pip install torch numpy 2>/dev/null || python3 -m pip install --user torch numpy 2>/dev/null || true

# Run experiment
export PYTHONPATH=$(pwd)
python3 scripts/z60_metabolic_experiment.py \\
    --host={host} \\
    --hidden-dim={args.get('hidden_dim', 256)} \\
    --num-layers={args.get('num_layers', 6)} \\
    --lm-epochs={args.get('lm_epochs', 5)} \\
    --rl-epochs={args.get('rl_epochs', 3)} \\
    --corpus-size={args.get('corpus_size', 500000)} \\
    --eval-samples={args.get('eval_samples', 100)}
"""

    # Build SSH command
    ssh_cmd = ['sshpass', '-p', config['password']]
    ssh_cmd.extend([
        'ssh', '-o', 'StrictHostKeyChecking=no',
        f"{config['user']}@{config['host']}",
        'bash', '-c', f"'{remote_script}'"
    ])

    process = subprocess.Popen(
        ssh_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    output_lines = []
    for line in process.stdout:
        output_lines.append(line)
        logger.info(f"[{host}] {line.rstrip()}")

    process.wait()

    return {
        'host': host,
        'returncode': process.returncode,
        'output': ''.join(output_lines),
    }


def sync_code_to_remote(host: str, config: Dict):
    """Sync project code to remote host."""
    logger.info(f"Syncing code to {host}...")

    rsync_cmd = [
        'sshpass', '-p', config['password'],
        'rsync', '-avz', '--progress',
        '--exclude', 'venv', '--exclude', '__pycache__',
        '--exclude', 'results', '--exclude', '.git',
        str(PROJECT_DIR) + '/',
        f"{config['user']}@{config['host']}:~/AMD_gfx1151_energy/"
    ]

    result = subprocess.run(rsync_cmd, capture_output=True, text=True)
    if result.returncode == 0:
        logger.info(f"Code synced to {host}")
    else:
        logger.warning(f"Code sync to {host} failed: {result.stderr}")


def collect_remote_results(host: str, config: Dict, timestamp: str) -> Optional[Dict]:
    """Collect results from remote host."""
    logger.info(f"Collecting results from {host}...")

    # Find the most recent results directory
    find_cmd = f"ls -td ~/AMD_gfx1151_energy/results/z60_metabolic_{host}_* 2>/dev/null | head -1"

    ssh_cmd = [
        'sshpass', '-p', config['password'],
        'ssh', '-o', 'StrictHostKeyChecking=no',
        f"{config['user']}@{config['host']}",
        find_cmd
    ]

    result = subprocess.run(ssh_cmd, capture_output=True, text=True)
    remote_dir = result.stdout.strip()

    if not remote_dir:
        logger.warning(f"No results found on {host}")
        return None

    # Copy results back
    local_results_dir = PROJECT_DIR / f"results/z60_aggregated_{timestamp}"
    local_results_dir.mkdir(parents=True, exist_ok=True)

    scp_cmd = [
        'sshpass', '-p', config['password'],
        'scp', '-r', '-o', 'StrictHostKeyChecking=no',
        f"{config['user']}@{config['host']}:{remote_dir}/",
        str(local_results_dir / host)
    ]

    subprocess.run(scp_cmd, capture_output=True)

    # Load results
    results_file = local_results_dir / host / "results.json"
    if results_file.exists():
        with open(results_file) as f:
            return json.load(f)

    return None


def aggregate_results(results: Dict[str, Dict], output_dir: Path):
    """Aggregate results from all hosts into comparison report."""
    logger.info("Aggregating results...")

    report = {
        'timestamp': datetime.now().isoformat(),
        'hosts': {},
        'comparison': {},
    }

    for host, data in results.items():
        if data is None:
            continue

        report['hosts'][host] = {
            'device': HOSTS[host]['gpu_type'],
            'baseline_ppl': data.get('baseline_ppl', 0),
            'metabolic_ppl': data.get('metabolic_ppl', 0),
            'baseline_j_per_token': data.get('baseline_j_per_token', 0),
            'metabolic_j_per_token': data.get('metabolic_j_per_token', 0),
            'ppl_delta_percent': data.get('ppl_delta_percent', 0),
            'energy_delta_percent': data.get('energy_delta_percent', 0),
            'efficiency_gain': data.get('efficiency_gain', 0),
            'action_distribution': data.get('metabolic_action_distribution', []),
        }

    # Cross-host comparison
    if len(report['hosts']) > 1:
        hosts_list = list(report['hosts'].keys())
        for i, h1 in enumerate(hosts_list):
            for h2 in hosts_list[i+1:]:
                d1 = report['hosts'][h1]
                d2 = report['hosts'][h2]

                report['comparison'][f'{h1}_vs_{h2}'] = {
                    'ppl_diff': d1['metabolic_ppl'] - d2['metabolic_ppl'],
                    'energy_diff': d1['metabolic_j_per_token'] - d2['metabolic_j_per_token'],
                    'efficiency_diff': d1['efficiency_gain'] - d2['efficiency_gain'],
                }

    # Save aggregated report
    report_path = output_dir / "aggregated_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    # Print summary
    print("\n" + "="*70)
    print("METABOLIC TRANSFORMER: CROSS-HOST COMPARISON")
    print("="*70)

    for host, data in report['hosts'].items():
        print(f"\n{host} ({data['device']}):")
        print(f"  Perplexity: {data['baseline_ppl']:.2f} → {data['metabolic_ppl']:.2f} ({data['ppl_delta_percent']:+.1f}%)")
        print(f"  J/token: {data['baseline_j_per_token']*1000:.2f} → {data['metabolic_j_per_token']*1000:.2f} mJ ({data['energy_delta_percent']:+.1f}%)")
        print(f"  Efficiency gain: {data['efficiency_gain']:+.1f}%")

    if report['comparison']:
        print("\nCross-Host Comparison:")
        for key, comp in report['comparison'].items():
            print(f"  {key}:")
            print(f"    PPL diff: {comp['ppl_diff']:+.2f}")
            print(f"    Energy diff: {comp['energy_diff']*1000:+.2f} mJ")

    print(f"\nFull report: {report_path}")
    return report


def main():
    parser = argparse.ArgumentParser(description="Run Metabolic Experiment on All Hosts")
    parser.add_argument('--local-only', action='store_true', help='Only run on ikaros')
    parser.add_argument('--hosts', nargs='+', default=['ikaros', 'daedalus', 'minos'],
                        help='Hosts to run on')
    parser.add_argument('--quick', action='store_true', help='Quick mode (reduced epochs)')
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--num-layers', type=int, default=6)
    parser.add_argument('--sync-code', action='store_true', help='Sync code to remote hosts first')
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = PROJECT_DIR / f"results/z60_aggregated_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Experiment arguments
    exp_args = {
        'hidden_dim': args.hidden_dim,
        'num_layers': args.num_layers,
        'lm_epochs': 2 if args.quick else 5,
        'rl_epochs': 1 if args.quick else 3,
        'corpus_size': 100000 if args.quick else 500000,
        'eval_samples': 50 if args.quick else 100,
    }

    if args.local_only:
        args.hosts = ['ikaros']

    logger.info(f"Running experiments on: {args.hosts}")
    logger.info(f"Configuration: {exp_args}")

    # Sync code to remote hosts
    if args.sync_code:
        for host in args.hosts:
            if host != 'ikaros' and host in HOSTS:
                sync_code_to_remote(host, HOSTS[host])

    # Run experiments in parallel
    threads = []
    results = {}
    results_lock = threading.Lock()

    def run_and_store(host):
        config = HOSTS.get(host)
        if config is None:
            logger.error(f"Unknown host: {host}")
            return

        if host == 'ikaros' or config['host'] == 'localhost':
            result = run_local_experiment(host, exp_args)
        else:
            result = run_remote_experiment(host, config, exp_args)

        with results_lock:
            results[host] = result

    for host in args.hosts:
        t = threading.Thread(target=run_and_store, args=(host,))
        threads.append(t)
        t.start()

    # Wait for all experiments
    for t in threads:
        t.join()

    logger.info("All experiments complete!")

    # Collect results from remote hosts
    collected_results = {}
    for host in args.hosts:
        if host == 'ikaros':
            # Find local results
            results_dirs = sorted(PROJECT_DIR.glob(f"results/z60_metabolic_{host}_*"))
            if results_dirs:
                results_file = results_dirs[-1] / "results.json"
                if results_file.exists():
                    with open(results_file) as f:
                        collected_results[host] = json.load(f)
        else:
            config = HOSTS.get(host)
            if config:
                collected_results[host] = collect_remote_results(host, config, timestamp)

    # Aggregate and report
    if collected_results:
        aggregate_results(collected_results, output_dir)
    else:
        logger.warning("No results collected!")


if __name__ == "__main__":
    main()
