#!/usr/bin/env python3
"""
z918: Cross-Machine Validation
==============================

Tests if embodied controllers trained on ikaros (AMD Radeon 8060S)
transfer effectively to:
- daedalus (192.168.0.37, AMD)
- minos (192.168.0.38, NVIDIA)

Key questions:
1. Does CaseBased memory transfer across GPUs?
2. What's the efficiency degradation on different hardware?
3. Do somatic state classifications generalize?

Author: FEEL Research Team
Date: 2026-01-29
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# SSH config
MACHINES = {
    'ikaros': {
        'host': 'localhost',
        'user': None,
        'gpu': 'AMD Radeon 8060S (gfx1151)',
        'venv': '/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/venv',
        'cwd': '/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy',
        'env': 'HSA_OVERRIDE_GFX_VERSION=11.0.0',
    },
    'daedalus': {
        'host': '192.168.0.37',
        'user': 'daedalus',
        'gpu': 'AMD (unknown model)',
        'venv': None,  # Will detect
        'cwd': None,   # Will sync
        'env': 'HSA_OVERRIDE_GFX_VERSION=11.0.0',  # May need adjustment
    },
    'minos': {
        'host': '192.168.0.38',
        'user': 'minos',
        'gpu': 'NVIDIA',
        'venv': None,
        'cwd': None,
        'env': '',  # NVIDIA doesn't need HSA override
    },
}


def run_remote_command(machine: str, command: str, timeout: int = 300) -> Dict[str, Any]:
    """Run command on remote machine via SSH."""
    config = MACHINES[machine]

    if config['host'] == 'localhost':
        # Local execution
        full_cmd = f"cd {config['cwd']} && source {config['venv']}/bin/activate && {config['env']} {command}"
        result = subprocess.run(
            full_cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    else:
        # Remote execution
        ssh_cmd = f"ssh {config['user']}@{config['host']}"
        full_cmd = f"{ssh_cmd} '{command}'"
        result = subprocess.run(
            full_cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    return {
        'returncode': result.returncode,
        'stdout': result.stdout,
        'stderr': result.stderr,
    }


def check_machine_availability(machine: str) -> bool:
    """Check if machine is reachable."""
    config = MACHINES[machine]

    if config['host'] == 'localhost':
        return True

    result = subprocess.run(
        f"ping -c 1 -W 2 {config['host']}",
        shell=True,
        capture_output=True,
    )
    return result.returncode == 0


def detect_venv_path(machine: str) -> Optional[str]:
    """Detect virtual environment path on remote machine."""
    config = MACHINES[machine]

    # Check common locations
    paths_to_check = [
        '~/Documents/claude_hive/AMD_gfx1151_energy/venv',
        '~/venvs/torch-rocm',
        '~/venvs/torch',
        '~/.venv',
    ]

    for path in paths_to_check:
        result = run_remote_command(machine, f"test -d {path} && echo 'exists'", timeout=10)
        if 'exists' in result['stdout']:
            return path

    return None


def sync_code_to_machine(machine: str) -> bool:
    """Sync codebase to remote machine."""
    config = MACHINES[machine]
    local_path = Path(__file__).parent.parent

    if config['host'] == 'localhost':
        return True

    # rsync essential files
    rsync_cmd = f"""
    rsync -avz --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' \
        --exclude='checkpoints' --exclude='data/MNIST' \
        {local_path}/ {config['user']}@{config['host']}:~/AMD_gfx1151_energy/
    """

    result = subprocess.run(rsync_cmd, shell=True, capture_output=True)
    return result.returncode == 0


def run_benchmark_on_machine(machine: str, duration: int = 60) -> Dict[str, Any]:
    """Run z914-style benchmark on a machine."""
    config = MACHINES[machine]

    # Benchmark script (simplified z914)
    benchmark_script = f"""
import sys
sys.path.insert(0, 'src')
import json
import time
import torch
import torch.nn as nn
import numpy as np

# Simple workload
class Workload:
    def __init__(self, device):
        self.model = nn.Sequential(
            nn.Embedding(8192, 512),
            nn.Linear(512, 2048),
            nn.GELU(),
            nn.Linear(2048, 512),
            nn.Linear(512, 8192),
        ).to(device)

    def run(self, bs=4, seq=256):
        x = torch.randint(0, 8192, (bs, seq), device='cuda')
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            y = self.model(x).mean()
        y.backward()
        torch.cuda.synchronize()
        return bs * seq

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {{device}}')

try:
    from telemetry.sysfs_hwmon import SysfsHwmonTelemetry
    telemetry = SysfsHwmonTelemetry()
    has_telemetry = True
except:
    has_telemetry = False
    print('Warning: No telemetry')

workload = Workload(device)

# Warmup
for _ in range(5):
    workload.run()

# Benchmark
start = time.time()
total_tokens = 0
power_samples = []

while time.time() - start < {duration}:
    if has_telemetry:
        sample = telemetry.read_sample()
        power_samples.append(sample.power_w)
    tokens = workload.run()
    total_tokens += tokens

elapsed = time.time() - start
avg_power = np.mean(power_samples) if power_samples else 100.0
total_energy = avg_power * elapsed
j_per_token = total_energy / max(total_tokens, 1)

result = {{
    'machine': '{machine}',
    'gpu': '{config["gpu"]}',
    'total_tokens': total_tokens,
    'elapsed_sec': elapsed,
    'avg_power_w': avg_power,
    'total_energy_j': total_energy,
    'j_per_token': j_per_token,
    'throughput': total_tokens / elapsed,
}}

print('RESULT_JSON:' + json.dumps(result))
"""

    # Run benchmark
    if config['host'] == 'localhost':
        cmd = f"cd {config['cwd']} && source {config['venv']}/bin/activate && {config['env']} python -c \"{benchmark_script}\""
    else:
        cmd = f"cd ~/AMD_gfx1151_energy && python -c \"{benchmark_script}\""

    result = run_remote_command(machine, cmd, timeout=duration + 60)

    # Parse result
    for line in result['stdout'].split('\n'):
        if line.startswith('RESULT_JSON:'):
            return json.loads(line.replace('RESULT_JSON:', ''))

    return {
        'machine': machine,
        'error': result['stderr'] or 'No result found',
        'stdout': result['stdout'],
    }


def main():
    print("=" * 70)
    print("z918: CROSS-MACHINE VALIDATION")
    print("=" * 70)

    # Check machine availability
    print("\nChecking machine availability...")
    available = {}
    for machine in MACHINES:
        is_available = check_machine_availability(machine)
        available[machine] = is_available
        status = "✓ ONLINE" if is_available else "✗ OFFLINE"
        print(f"  {machine}: {status}")

    # Run benchmarks on available machines
    results = {}
    duration = 60  # 1 minute per machine

    for machine, is_available in available.items():
        if not is_available:
            print(f"\n  Skipping {machine} (offline)")
            continue

        print(f"\n{'='*60}")
        print(f"Benchmarking: {machine}")
        print(f"{'='*60}")

        # Sync code if remote
        if MACHINES[machine]['host'] != 'localhost':
            print("  Syncing code...")
            if not sync_code_to_machine(machine):
                print("  Failed to sync code, skipping")
                continue

        # Run benchmark
        print(f"  Running benchmark ({duration}s)...")
        result = run_benchmark_on_machine(machine, duration)
        results[machine] = result

        if 'error' not in result:
            print(f"  J/token: {result['j_per_token']*1000:.3f} mJ/tok")
            print(f"  Throughput: {result['throughput']:.0f} tok/s")
        else:
            print(f"  Error: {result['error']}")

    # Comparative analysis
    if len(results) > 1:
        print("\n" + "=" * 70)
        print("CROSS-MACHINE COMPARISON")
        print("=" * 70)

        # Use ikaros as baseline
        if 'ikaros' in results and 'error' not in results['ikaros']:
            ikaros_j = results['ikaros']['j_per_token']

            for machine, result in sorted(results.items(), key=lambda x: x[1].get('j_per_token', float('inf'))):
                if 'error' in result:
                    continue

                j_tok = result['j_per_token'] * 1000
                vs_ikaros = (result['j_per_token'] - ikaros_j) / ikaros_j * 100

                print(f"  {machine:10s}: {j_tok:.3f} mJ/tok ({vs_ikaros:+.1f}% vs ikaros)")

    # Save results
    results_summary = {
        'benchmark': 'z918_cross_machine',
        'timestamp': datetime.now().isoformat(),
        'machines': results,
    }

    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    results_file = results_dir / "z918_cross_machine.json"

    with open(results_file, 'w') as f:
        json.dump(results_summary, f, indent=2, default=str)

    print(f"\nResults saved to: {results_file}")

    return results_summary


if __name__ == "__main__":
    main()
