#!/usr/bin/env python3
"""
z100_minos_deployment.py - Deploy and validate FEEL on Minos (NVIDIA GPU)

This script:
1. Syncs the codebase to Minos
2. Sets up the environment
3. Runs validation tests with real NVML energy counters
4. Collects results back

Minos has NVIDIA GPU with NVML support for cumulative energy counters.

Usage:
    python z100_minos_deployment.py --deploy      # Sync code to Minos
    python z100_minos_deployment.py --validate    # Run validation suite
    python z100_minos_deployment.py --all         # Deploy + validate + collect
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# Minos connection details (from CLAUDE.md)
MINOS_HOST = os.environ.get("MINOS_HOST", "192.168.0.38")
MINOS_USER = os.environ.get("MINOS_USER", "minos")
MINOS_PASS = os.environ.get("MINOS_PASS", "minos")

# Local paths
LOCAL_PROJECT = Path(__file__).parent.parent
REMOTE_PROJECT = "/home/minos/feel_project"


@dataclass
class ValidationResult:
    """Result from a validation test."""
    test_name: str
    success: bool
    output: str
    error: str
    duration_s: float
    nvml_available: bool
    energy_measured: bool


def run_ssh_command(command: str, timeout: int = 300) -> Tuple[str, str, int]:
    """Run command on Minos via SSH."""
    ssh_cmd = [
        "sshpass", "-p", MINOS_PASS,
        "ssh", "-o", "StrictHostKeyChecking=no",
        f"{MINOS_USER}@{MINOS_HOST}",
        command
    ]

    try:
        result = subprocess.run(
            ssh_cmd,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired:
        return "", "Command timed out", -1
    except FileNotFoundError:
        # sshpass not installed, try with expect or plain ssh
        log.warning("sshpass not found, trying plain ssh (may require manual password)")
        ssh_cmd = [
            "ssh", "-o", "StrictHostKeyChecking=no",
            f"{MINOS_USER}@{MINOS_HOST}",
            command
        ]
        result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=timeout)
        return result.stdout, result.stderr, result.returncode


def rsync_to_minos(local_path: Path, remote_path: str) -> bool:
    """Sync local directory to Minos."""
    log.info(f"Syncing {local_path} to {MINOS_USER}@{MINOS_HOST}:{remote_path}")

    # Create remote directory first
    run_ssh_command(f"mkdir -p {remote_path}")

    rsync_cmd = [
        "rsync", "-avz", "--progress",
        "--exclude", "venv",
        "--exclude", "__pycache__",
        "--exclude", "*.pyc",
        "--exclude", ".git",
        "--exclude", "results",
        "--exclude", "*.jsonl",
        f"{local_path}/",
        f"{MINOS_USER}@{MINOS_HOST}:{remote_path}/"
    ]

    try:
        result = subprocess.run(rsync_cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            log.info("Sync completed successfully")
            return True
        else:
            log.error(f"Sync failed: {result.stderr}")
            return False
    except Exception as e:
        log.error(f"Sync error: {e}")
        return False


def check_minos_nvidia() -> Dict:
    """Check NVIDIA GPU status on Minos."""
    log.info("Checking NVIDIA GPU on Minos...")

    # Check nvidia-smi
    stdout, stderr, rc = run_ssh_command("nvidia-smi --query-gpu=name,memory.total,power.draw,power.limit --format=csv")

    if rc != 0:
        return {"available": False, "error": stderr}

    # Check NVML energy support
    nvml_check = """python3 -c "
import pynvml
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
try:
    energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
    print(f'NVML_ENERGY_AVAILABLE: {energy}')
except:
    print('NVML_ENERGY_NOT_AVAILABLE')
pynvml.nvmlShutdown()
"
"""
    nvml_out, nvml_err, nvml_rc = run_ssh_command(nvml_check)

    return {
        "available": True,
        "nvidia_smi": stdout,
        "nvml_energy": "NVML_ENERGY_AVAILABLE" in nvml_out,
        "nvml_output": nvml_out
    }


def setup_minos_environment() -> bool:
    """Set up Python environment on Minos."""
    log.info("Setting up Python environment on Minos...")

    setup_cmds = [
        f"cd {REMOTE_PROJECT}",
        "python3 -m venv venv 2>/dev/null || true",
        "source venv/bin/activate",
        "pip install --upgrade pip",
        "pip install pynvml requests numpy scipy torch --quiet",
    ]

    stdout, stderr, rc = run_ssh_command(" && ".join(setup_cmds), timeout=600)

    if rc == 0:
        log.info("Environment setup complete")
        return True
    else:
        log.error(f"Environment setup failed: {stderr}")
        return False


def start_actuator_daemon() -> bool:
    """Start the FEEL actuator daemon on Minos."""
    log.info("Starting actuator daemon on Minos...")

    # Kill any existing daemon
    run_ssh_command("pkill -f privileged_daemon || true")
    time.sleep(1)

    # Start daemon in background
    start_cmd = f"""
cd {REMOTE_PROJECT} && \
source venv/bin/activate && \
nohup python src/actuator/privileged_daemon_v2.py --port 9877 > /tmp/daemon.log 2>&1 &
echo $!
"""
    stdout, stderr, rc = run_ssh_command(start_cmd)

    if rc == 0 and stdout.strip().isdigit():
        pid = stdout.strip()
        log.info(f"Daemon started with PID {pid}")
        time.sleep(2)

        # Verify daemon is running
        check_stdout, _, _ = run_ssh_command(f"curl -s http://localhost:9877/health || echo FAILED")
        if "FAILED" not in check_stdout:
            log.info("Daemon health check passed")
            return True
        else:
            log.error("Daemon health check failed")
            return False
    else:
        log.error(f"Failed to start daemon: {stderr}")
        return False


def run_validation_test(test_script: str, test_name: str, timeout: int = 300) -> ValidationResult:
    """Run a validation test on Minos."""
    log.info(f"Running validation: {test_name}")

    start = time.time()

    cmd = f"""
cd {REMOTE_PROJECT} && \
source venv/bin/activate && \
python {test_script} 2>&1
"""

    stdout, stderr, rc = run_ssh_command(cmd, timeout=timeout)
    duration = time.time() - start

    success = rc == 0
    nvml_available = "NVML" in stdout or "nvml" in stdout.lower()
    energy_measured = "energy" in stdout.lower() and ("J" in stdout or "joule" in stdout.lower())

    return ValidationResult(
        test_name=test_name,
        success=success,
        output=stdout,
        error=stderr,
        duration_s=duration,
        nvml_available=nvml_available,
        energy_measured=energy_measured,
    )


def collect_results() -> Dict:
    """Collect results from Minos."""
    log.info("Collecting results from Minos...")

    # Create local results directory
    local_results = LOCAL_PROJECT / "results" / "minos_validation"
    local_results.mkdir(parents=True, exist_ok=True)

    # Rsync results back
    rsync_cmd = [
        "rsync", "-avz",
        f"{MINOS_USER}@{MINOS_HOST}:{REMOTE_PROJECT}/results/",
        f"{local_results}/"
    ]

    try:
        subprocess.run(rsync_cmd, capture_output=True, text=True, timeout=120)
        log.info(f"Results collected to {local_results}")
        return {"success": True, "path": str(local_results)}
    except Exception as e:
        log.error(f"Failed to collect results: {e}")
        return {"success": False, "error": str(e)}


def run_validation_suite() -> List[ValidationResult]:
    """Run full validation suite on Minos."""
    results = []

    # List of tests to run
    tests = [
        # Quick NVML energy test
        ("scripts/z94_real_vllm_benchmark.py --help", "z94 vLLM benchmark (help check)"),

        # Energy harness test
        ("python3 -c \""
         "import sys; sys.path.insert(0, '.'); "
         "from src.metabolic.energy_harness import EnergyMonitor; "
         "print('EnergyMonitor imported OK')\"", "Energy harness import"),

        # NVML direct test
        ("python3 -c \""
         "import pynvml; pynvml.nvmlInit(); "
         "h = pynvml.nvmlDeviceGetHandleByIndex(0); "
         "e = pynvml.nvmlDeviceGetTotalEnergyConsumption(h); "
         "print(f'NVML Energy: {e} mJ'); "
         "pynvml.nvmlShutdown()\"", "NVML direct energy read"),

        # Actuator health check
        ("curl -s http://localhost:9877/health | python3 -m json.tool", "Actuator daemon health"),

        # Profile switching test
        ("curl -s -X POST http://localhost:9877/profile -H 'Content-Type: application/json' "
         "-d '{\"profile\": \"eco\"}' | python3 -m json.tool", "Profile switch to eco"),

        # Energy reading from daemon
        ("curl -s http://localhost:9877/energy | python3 -m json.tool", "Daemon energy endpoint"),

        # Telemetry reading
        ("curl -s http://localhost:9877/telemetry | python3 -m json.tool", "Daemon telemetry"),
    ]

    for test_cmd, test_name in tests:
        result = run_validation_test(test_cmd, test_name, timeout=60)
        results.append(result)

        status = "PASS" if result.success else "FAIL"
        log.info(f"  [{status}] {test_name} ({result.duration_s:.1f}s)")

        if not result.success:
            log.error(f"    Error: {result.error[:200]}")

    return results


def generate_validation_report(
    gpu_info: Dict,
    results: List[ValidationResult],
    output_path: Path
):
    """Generate validation report."""

    report = [
        "# Minos NVML Validation Report",
        "",
        f"Generated: {datetime.now().isoformat()}",
        f"Host: {MINOS_HOST}",
        "",
        "## GPU Information",
        "",
        "```",
        gpu_info.get("nvidia_smi", "N/A"),
        "```",
        "",
        f"**NVML Energy Counters**: {'Available' if gpu_info.get('nvml_energy') else 'Not Available'}",
        "",
        "## Validation Results",
        "",
        "| Test | Status | Duration | NVML | Energy |",
        "|------|--------|----------|------|--------|",
    ]

    passed = 0
    failed = 0

    for r in results:
        status = "PASS" if r.success else "FAIL"
        nvml = "Yes" if r.nvml_available else "-"
        energy = "Yes" if r.energy_measured else "-"

        if r.success:
            passed += 1
        else:
            failed += 1

        report.append(f"| {r.test_name} | {status} | {r.duration_s:.1f}s | {nvml} | {energy} |")

    report.extend([
        "",
        "## Summary",
        "",
        f"- **Total tests**: {len(results)}",
        f"- **Passed**: {passed}",
        f"- **Failed**: {failed}",
        f"- **Pass rate**: {passed/len(results)*100:.1f}%",
        "",
    ])

    # Add failure details
    failures = [r for r in results if not r.success]
    if failures:
        report.extend([
            "## Failures",
            "",
        ])
        for r in failures:
            report.extend([
                f"### {r.test_name}",
                "",
                "**Error:**",
                "```",
                r.error[:500] if r.error else "No error message",
                "```",
                "",
            ])

    # Write report
    with open(output_path, "w") as f:
        f.write("\n".join(report))

    log.info(f"Report saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Deploy and validate FEEL on Minos")
    parser.add_argument("--deploy", action="store_true", help="Deploy code to Minos")
    parser.add_argument("--setup", action="store_true", help="Set up environment on Minos")
    parser.add_argument("--start-daemon", action="store_true", help="Start actuator daemon")
    parser.add_argument("--validate", action="store_true", help="Run validation suite")
    parser.add_argument("--collect", action="store_true", help="Collect results from Minos")
    parser.add_argument("--all", action="store_true", help="Run all steps")
    parser.add_argument("--check-gpu", action="store_true", help="Check GPU status only")
    args = parser.parse_args()

    if not any([args.deploy, args.setup, args.start_daemon, args.validate,
                args.collect, args.all, args.check_gpu]):
        parser.print_help()
        sys.exit(1)

    # Create results directory
    results_dir = LOCAL_PROJECT / "results" / "minos_validation"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Check GPU
    if args.check_gpu or args.all:
        gpu_info = check_minos_nvidia()
        if gpu_info["available"]:
            log.info("NVIDIA GPU detected on Minos")
            log.info(f"NVML Energy support: {gpu_info.get('nvml_energy', False)}")
        else:
            log.error(f"No NVIDIA GPU: {gpu_info.get('error', 'Unknown error')}")
            if args.check_gpu:
                sys.exit(1)

    # Deploy
    if args.deploy or args.all:
        if not rsync_to_minos(LOCAL_PROJECT, REMOTE_PROJECT):
            log.error("Deployment failed")
            if not args.all:
                sys.exit(1)

    # Setup environment
    if args.setup or args.all:
        if not setup_minos_environment():
            log.error("Environment setup failed")
            if not args.all:
                sys.exit(1)

    # Start daemon
    if args.start_daemon or args.all:
        if not start_actuator_daemon():
            log.error("Daemon start failed")
            if not args.all:
                sys.exit(1)

    # Run validation
    if args.validate or args.all:
        log.info("\n" + "="*60)
        log.info("Running validation suite")
        log.info("="*60 + "\n")

        gpu_info = check_minos_nvidia()
        results = run_validation_suite()

        # Generate report
        report_path = results_dir / "validation_report.md"
        generate_validation_report(gpu_info, results, report_path)

        # Save raw results
        results_data = [
            {
                "test_name": r.test_name,
                "success": r.success,
                "duration_s": r.duration_s,
                "nvml_available": r.nvml_available,
                "energy_measured": r.energy_measured,
                "output_preview": r.output[:500],
            }
            for r in results
        ]

        with open(results_dir / "validation_results.json", "w") as f:
            json.dump(results_data, f, indent=2)

        # Print summary
        passed = sum(1 for r in results if r.success)
        total = len(results)
        log.info(f"\nValidation complete: {passed}/{total} tests passed")

    # Collect results
    if args.collect or args.all:
        collect_results()

    log.info("\nDone!")


if __name__ == "__main__":
    main()
