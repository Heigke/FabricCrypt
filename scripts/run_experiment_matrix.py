#!/usr/bin/env python3
"""
Experiment Matrix Runner for Energy-Efficient LLM Inference.

Reads the experiment manifest YAML and executes the specified matrix
of experiments with proper warmup, cooldown, and data collection.
"""

import argparse
import itertools
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class ExperimentConfig:
    """Single experiment configuration."""
    model: str
    model_id: str
    prompt_length: int
    decode_length: int
    batch_size: int
    concurrency: int
    policy: str
    policy_config: Dict[str, Any]
    samples: int


class ExperimentMatrixRunner:
    """Runs a matrix of experiments from manifest."""

    def __init__(
        self,
        manifest_path: Path,
        output_dir: Path,
        matrix_name: str = "standard",
        dry_run: bool = False
    ):
        self.manifest_path = manifest_path
        self.output_dir = output_dir
        self.matrix_name = matrix_name
        self.dry_run = dry_run

        # Load manifest
        with open(manifest_path, 'r') as f:
            self.manifest = yaml.safe_load(f)

        # Prepare output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Track progress
        self.completed: List[str] = []
        self.failed: List[str] = []
        self.skipped: List[str] = []

    def get_matrix_configs(self) -> List[ExperimentConfig]:
        """Generate all experiment configurations from the matrix."""
        matrix = self.manifest["matrix"].get(self.matrix_name, {})
        if not matrix:
            raise ValueError(f"Matrix '{self.matrix_name}' not found in manifest")

        # Get models
        model_ids = matrix.get("models", [])
        models = {m["id"]: m for m in self.manifest["models"]}

        # Get prompt lengths
        prompt_lengths = matrix.get("prompt_lengths", [])
        prompt_configs = self.manifest["prompt_lengths"]

        # Get other parameters
        decode_lengths = matrix.get("decode_lengths", [64])
        batch_sizes = matrix.get("batch_sizes", [1])
        concurrency_levels = matrix.get("concurrency_levels", [1])
        policies = matrix.get("policies", ["auto"])
        samples = matrix.get("samples", 15)

        # Get policy configs
        policy_configs = {p["name"]: p for p in self.manifest["dvfs_policies"]}

        # Generate all combinations
        configs = []
        for model_id in model_ids:
            model_info = models.get(model_id)
            if not model_info:
                print(f"Warning: Model '{model_id}' not found in manifest")
                continue

            for prompt_key in prompt_lengths:
                prompt_info = prompt_configs.get(prompt_key, {})
                prompt_tokens = prompt_info.get("tokens", 256)

                for decode_len in decode_lengths:
                    for batch_size in batch_sizes:
                        for concurrency in concurrency_levels:
                            for policy in policies:
                                policy_config = policy_configs.get(policy, {})

                                configs.append(ExperimentConfig(
                                    model=model_info["name"],
                                    model_id=model_id,
                                    prompt_length=prompt_tokens,
                                    decode_length=decode_len,
                                    batch_size=batch_size,
                                    concurrency=concurrency,
                                    policy=policy,
                                    policy_config=policy_config,
                                    samples=samples
                                ))

        return configs

    def get_config_id(self, config: ExperimentConfig) -> str:
        """Generate unique ID for a configuration."""
        return (
            f"{config.model_id}_"
            f"p{config.prompt_length}_"
            f"d{config.decode_length}_"
            f"b{config.batch_size}_"
            f"c{config.concurrency}_"
            f"{config.policy}"
        )

    def run_single_experiment(self, config: ExperimentConfig) -> bool:
        """Run a single experiment configuration."""
        config_id = self.get_config_id(config)
        print(f"\n{'='*60}")
        print(f"Running: {config_id}")
        print(f"{'='*60}")

        # Build command
        script_path = Path(__file__).parent / "hf_infer_extended.py"

        cmd = [
            "python", str(script_path),
            "--model", config.model,
            "--prompt-length", str(config.prompt_length),
            "--decode-length", str(config.decode_length),
            "--samples", str(config.samples),
            "--policies", config.policy,
            "--output-dir", str(self.output_dir / config.model_id),
        ]

        # Add batch size if > 1
        if config.batch_size > 1:
            cmd.extend(["--batch-size", str(config.batch_size)])

        # Add concurrency if > 1
        if config.concurrency > 1:
            cmd.extend(["--concurrency", str(config.concurrency)])

        # Add DPM control if specified
        if config.policy_config.get("dpm_control"):
            cmd.append("--enable-dpm-control")
            sclk_indices = config.policy_config.get("sclk_indices", [])
            mclk_indices = config.policy_config.get("mclk_indices", [])
            if sclk_indices:
                cmd.extend(["--sclk-indices", ",".join(map(str, sclk_indices))])
            if mclk_indices:
                cmd.extend(["--mclk-indices", ",".join(map(str, mclk_indices))])

        print(f"Command: {' '.join(cmd)}")

        if self.dry_run:
            print("[DRY RUN] Would execute above command")
            return True

        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=False,  # Let output stream to console
                timeout=3600  # 1 hour timeout per experiment
            )
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error: Experiment failed with return code {e.returncode}")
            return False
        except subprocess.TimeoutExpired:
            print("Error: Experiment timed out")
            return False

    def check_completed(self, config: ExperimentConfig) -> bool:
        """Check if an experiment has already been completed."""
        config_id = self.get_config_id(config)
        output_path = (
            self.output_dir / config.model_id /
            f"{config.policy}_p{config.prompt_length}_d{config.decode_length}_raw.csv"
        )

        if output_path.exists():
            # Check if it has enough samples
            try:
                with open(output_path, 'r') as f:
                    lines = sum(1 for line in f) - 1  # Subtract header
                    if lines >= config.samples:
                        return True
            except Exception:
                pass

        return False

    def run_matrix(self, resume: bool = True):
        """Run all experiments in the matrix."""
        configs = self.get_matrix_configs()
        total = len(configs)

        print(f"\n{'#'*60}")
        print(f"Experiment Matrix: {self.matrix_name}")
        print(f"Total configurations: {total}")
        print(f"Output directory: {self.output_dir}")
        print(f"Resume mode: {resume}")
        print(f"Dry run: {self.dry_run}")
        print(f"{'#'*60}\n")

        # Save matrix configuration
        matrix_config_path = self.output_dir / "matrix_config.json"
        with open(matrix_config_path, 'w') as f:
            json.dump({
                "manifest": str(self.manifest_path),
                "matrix_name": self.matrix_name,
                "start_time": datetime.now().isoformat(),
                "total_configs": total,
                "configs": [self.get_config_id(c) for c in configs]
            }, f, indent=2)

        # Get measurement config
        measurement = self.manifest.get("measurement", {})
        cooldown = measurement.get("cooldown_seconds", 5)

        for i, config in enumerate(configs):
            config_id = self.get_config_id(config)
            print(f"\n[{i+1}/{total}] Processing: {config_id}")

            # Check if already completed
            if resume and self.check_completed(config):
                print(f"  -> Already completed, skipping")
                self.skipped.append(config_id)
                continue

            # Run experiment
            success = self.run_single_experiment(config)

            if success:
                self.completed.append(config_id)
            else:
                self.failed.append(config_id)

            # Cooldown between experiments
            if i < total - 1 and not self.dry_run:
                print(f"\nCooldown: {cooldown}s...")
                time.sleep(cooldown)

        # Summary
        self.print_summary()

    def run_sustained_tests(self):
        """Run sustained thermal tests for key configurations."""
        sustained = self.manifest.get("measurement", {}).get("sustained_test", {})
        if not sustained.get("enabled"):
            print("Sustained tests not enabled in manifest")
            return

        duration = sustained.get("duration_seconds", 600)
        policies = sustained.get("policies", ["auto", "profile_peak"])

        # Use first model for sustained test
        model_ids = self.manifest["matrix"].get(self.matrix_name, {}).get("models", [])
        if not model_ids:
            print("No models specified for sustained test")
            return

        model_info = next(
            (m for m in self.manifest["models"] if m["id"] == model_ids[0]),
            None
        )
        if not model_info:
            return

        print(f"\n{'#'*60}")
        print(f"Running Sustained Thermal Tests")
        print(f"Model: {model_info['name']}")
        print(f"Duration: {duration}s per policy")
        print(f"Policies: {policies}")
        print(f"{'#'*60}\n")

        script_path = Path(__file__).parent / "hf_infer_extended.py"

        for policy in policies:
            print(f"\n{'='*60}")
            print(f"Sustained Test: {policy}")
            print(f"{'='*60}")

            cmd = [
                "python", str(script_path),
                "--model", model_info["name"],
                "--sustained-duration", str(duration),
                "--policies", policy,
                "--output-dir", str(self.output_dir / "sustained"),
            ]

            if self.dry_run:
                print(f"[DRY RUN] Would execute: {' '.join(cmd)}")
                continue

            try:
                subprocess.run(cmd, check=True, timeout=duration + 300)
            except subprocess.CalledProcessError as e:
                print(f"Sustained test failed: {e}")
            except subprocess.TimeoutExpired:
                print("Sustained test timed out")

    def print_summary(self):
        """Print experiment summary."""
        print(f"\n{'#'*60}")
        print("EXPERIMENT SUMMARY")
        print(f"{'#'*60}")
        print(f"Completed: {len(self.completed)}")
        print(f"Failed: {len(self.failed)}")
        print(f"Skipped (already done): {len(self.skipped)}")

        if self.failed:
            print(f"\nFailed experiments:")
            for config_id in self.failed:
                print(f"  - {config_id}")

        # Save summary
        summary_path = self.output_dir / "experiment_summary.json"
        with open(summary_path, 'w') as f:
            json.dump({
                "end_time": datetime.now().isoformat(),
                "completed": self.completed,
                "failed": self.failed,
                "skipped": self.skipped
            }, f, indent=2)

        print(f"\nSummary saved to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run experiment matrix from manifest"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("experiments/workload_matrix.yaml"),
        help="Path to experiment manifest YAML"
    )
    parser.add_argument(
        "--matrix",
        type=str,
        default="standard",
        choices=["quick", "standard", "full"],
        help="Matrix configuration to run"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/workload_matrix"),
        help="Output directory for results"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Don't skip already completed experiments"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing"
    )
    parser.add_argument(
        "--sustained-only",
        action="store_true",
        help="Only run sustained thermal tests"
    )

    args = parser.parse_args()

    # Add timestamp to output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir / f"{args.matrix}_{timestamp}"

    runner = ExperimentMatrixRunner(
        manifest_path=args.manifest,
        output_dir=output_dir,
        matrix_name=args.matrix,
        dry_run=args.dry_run
    )

    if args.sustained_only:
        runner.run_sustained_tests()
    else:
        runner.run_matrix(resume=not args.no_resume)

        # Run sustained tests after matrix if enabled
        runner.run_sustained_tests()


if __name__ == "__main__":
    main()
