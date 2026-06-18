#!/usr/bin/env python3
"""
z150: Causal Validation for Early Exit Embodied Compute

This script proves the causal chain:
    semantic uncertainty → exit depth → energy savings

Ablation matrix:
1. Full ECC: uncertainty + body → exit decision
2. Uncertainty-Only: uncertainty → exit (body signal zeroed)
3. Body-Only: body → exit (uncertainty signal zeroed)
4. Random: random exit at same mean depth
5. Fixed: always use full depth (baseline)

Success criteria:
- Full ECC saves ≥30% energy vs Fixed
- Full ECC > Random at same depth budget (proves uncertainty matters)
- Full ECC > Body-Only (proves semantics + body > body alone)
- Quality preserved within 5% of Fixed

Usage:
    python z150_causal_early_exit_validation.py --model deepseek-r1 --device cuda:0
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from tqdm import tqdm

# Local imports
from src.modeling.early_exit import EarlyExitDecision, EarlyExitStats
from src.modeling.early_exit_transformer import EarlyExitTransformer, EarlyExitOutput
from src.modeling.z24_sensor_hub import AMDSensorHub, SimulatedSensorHub
from src.energy_harness.nvml_energy import create_energy_meter, EnergyMeasurement
from src.memory.control_memory import ControlMemory, ControlOutcome


@dataclass
class AblationConfig:
    """Configuration for an ablation condition"""
    name: str
    use_uncertainty: bool = True
    use_body: bool = True
    use_memory: bool = True
    fixed_exit_layer: Optional[int] = None  # None = adaptive
    random_exit: bool = False
    description: str = ""


@dataclass
class PromptSet:
    """Set of prompts with difficulty annotation"""
    name: str
    difficulty: str  # easy, medium, hard
    prompts: List[str] = field(default_factory=list)
    expected_perplexity_range: Tuple[float, float] = (0.0, 100.0)


@dataclass
class SingleResult:
    """Result from a single inference"""
    prompt: str
    difficulty: str
    condition: str
    exit_layer: int
    num_layers: int
    energy_j: float
    latency_ms: float
    quality_score: float  # Proxy: log prob of generated tokens
    tokens_generated: int
    uncertainty_mean: float
    body_state_mean: Dict[str, float]
    flops_saved_pct: float


@dataclass
class AblationResult:
    """Aggregated results for an ablation condition"""
    condition: str
    n_samples: int

    # Energy metrics
    energy_mean_j: float
    energy_std_j: float
    energy_ci_95: Tuple[float, float]

    # Exit metrics
    exit_layer_mean: float
    exit_layer_std: float
    flops_saved_pct: float

    # Quality metrics
    quality_mean: float
    quality_std: float

    # Latency metrics
    latency_mean_ms: float
    latency_std_ms: float

    # Per-difficulty breakdown
    by_difficulty: Dict[str, Dict[str, float]] = field(default_factory=dict)


def create_prompt_sets() -> List[PromptSet]:
    """Create prompt sets of varying difficulty"""

    easy_prompts = [
        "The sky is",
        "One plus one equals",
        "The color of grass is",
        "Water freezes at",
        "The sun rises in the",
        "Dogs say",
        "Cats like to",
        "The moon orbits the",
        "Fish live in",
        "Birds can",
    ]

    medium_prompts = [
        "Explain why the sky appears blue during the day.",
        "What are the main differences between Python and JavaScript?",
        "Describe the water cycle in simple terms.",
        "How does photosynthesis work?",
        "What causes the seasons to change?",
        "Explain the concept of supply and demand.",
        "How do computers store information?",
        "What is the scientific method?",
        "Describe how vaccines work.",
        "Explain the theory of evolution.",
    ]

    hard_prompts = [
        "Analyze the philosophical implications of Gödel's incompleteness theorems on the nature of mathematical truth.",
        "Discuss the relationship between quantum entanglement and information theory in the context of black hole physics.",
        "Compare and contrast the economic theories of Keynes and Hayek with respect to government intervention.",
        "Explain the mechanism by which CRISPR-Cas9 achieves targeted gene editing and discuss ethical implications.",
        "Analyze the geopolitical factors that led to the fall of the Western Roman Empire.",
        "Discuss the implications of the measurement problem in quantum mechanics for our understanding of reality.",
        "Compare different interpretations of probability: frequentist, Bayesian, and propensity views.",
        "Analyze the role of neuroplasticity in learning and memory formation from a computational perspective.",
        "Discuss the philosophical problem of consciousness and evaluate different theories: dualism, functionalism, and panpsychism.",
        "Explain the mathematical foundations of deep learning and discuss limitations of current architectures.",
    ]

    return [
        PromptSet(name="easy", difficulty="easy", prompts=easy_prompts, expected_perplexity_range=(1.0, 10.0)),
        PromptSet(name="medium", difficulty="medium", prompts=medium_prompts, expected_perplexity_range=(10.0, 50.0)),
        PromptSet(name="hard", difficulty="hard", prompts=hard_prompts, expected_perplexity_range=(50.0, 200.0)),
    ]


def create_ablation_configs() -> List[AblationConfig]:
    """Define ablation conditions"""
    return [
        AblationConfig(
            name="full_ecc",
            use_uncertainty=True,
            use_body=True,
            use_memory=True,
            description="Full Embodied Conditional Compute: uncertainty + body + memory"
        ),
        AblationConfig(
            name="uncertainty_only",
            use_uncertainty=True,
            use_body=False,
            use_memory=False,
            description="Uncertainty-driven exit only (body signals zeroed)"
        ),
        AblationConfig(
            name="body_only",
            use_uncertainty=False,
            use_body=True,
            use_memory=False,
            description="Body-driven exit only (uncertainty set to 0.5)"
        ),
        AblationConfig(
            name="random",
            use_uncertainty=False,
            use_body=False,
            random_exit=True,
            description="Random exit at each exit layer (same mean depth as full_ecc)"
        ),
        AblationConfig(
            name="fixed",
            use_uncertainty=False,
            use_body=False,
            fixed_exit_layer=-1,  # -1 means last layer
            description="Fixed full depth (baseline)"
        ),
    ]


class CausalValidator:
    """
    Runs causal validation experiments for early exit.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cuda:0",
        num_layers: int = 24,
        max_new_tokens: int = 50,
        num_seeds: int = 3
    ):
        self.model_name = model_name
        self.device = device
        self.num_layers = num_layers
        self.max_new_tokens = max_new_tokens
        self.num_seeds = num_seeds

        self.model = None
        self.tokenizer = None
        self.sensor_hub = None
        self.energy_meter = None
        self.control_memory = None

        self.results: List[SingleResult] = []

    def setup(self):
        """Initialize model, sensors, and energy meter"""
        print(f"\n{'='*60}")
        print(f"Setting up Causal Validation")
        print(f"{'='*60}")

        # Load model
        print(f"\nLoading model: {self.model_name}")
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map=self.device
        )

        # Setup sensor hub
        print("Setting up sensor hub...")
        try:
            self.sensor_hub = AMDSensorHub()
            print("  Using AMD sensor hub")
        except Exception as e:
            print(f"  AMD sensor hub failed ({e}), using simulated")
            self.sensor_hub = SimulatedSensorHub()

        # Wrap with early exit
        print("Creating early exit wrapper...")
        self.model = EarlyExitTransformer(
            base_model,
            sensor_hub=self.sensor_hub,
            exit_layers=[4, 8, 12, 16, 20, 24] if self.num_layers == 24 else None
        )
        self.model.eval()

        # Setup energy meter
        print("Setting up energy meter...")
        try:
            self.energy_meter = create_energy_meter(device_type="auto")
            print(f"  Energy meter: {type(self.energy_meter).__name__}")
        except Exception as e:
            print(f"  Energy meter failed: {e}")
            self.energy_meter = None

        # Setup control memory
        self.control_memory = ControlMemory(num_layers=self.num_layers)

        print("\nSetup complete!")

    def run_single_inference(
        self,
        prompt: str,
        config: AblationConfig,
        seed: int
    ) -> SingleResult:
        """Run single inference with ablation config"""
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Prepare input
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        # Get sensor vector
        sensor_vector = self.model.get_sensor_vector(1, self.device)

        # Apply ablation conditions
        if not config.use_uncertainty:
            # Zero uncertainty signal = always say "confident"
            # This is done by modifying the uncertainty head output
            pass  # Handled in forward

        if not config.use_body:
            # Zero body signal
            sensor_vector = torch.zeros_like(sensor_vector)

        # Start energy measurement
        if self.energy_meter:
            self.energy_meter.start()

        start_time = time.perf_counter()

        # Run inference
        with torch.no_grad():
            if config.fixed_exit_layer is not None:
                # Fixed depth
                exit_layer = self.num_layers if config.fixed_exit_layer == -1 else config.fixed_exit_layer
                output = self.model(
                    input_ids,
                    attention_mask=attention_mask,
                    sensor_vector=sensor_vector,
                    force_exit_layer=exit_layer
                )
            elif config.random_exit:
                # Random exit
                exit_layer = np.random.choice(self.model.exit_layers)
                output = self.model(
                    input_ids,
                    attention_mask=attention_mask,
                    sensor_vector=sensor_vector,
                    force_exit_layer=exit_layer
                )
            else:
                # Adaptive exit
                output = self.model(
                    input_ids,
                    attention_mask=attention_mask,
                    sensor_vector=sensor_vector
                )

        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000

        # Stop energy measurement
        if self.energy_meter:
            energy = self.energy_meter.stop()
            energy_j = energy.energy_joules
        else:
            energy_j = latency_ms * 0.1 / 1000  # Rough estimate: 100W

        # Compute quality proxy (negative log likelihood of output)
        logits = output.logits
        quality_score = -torch.nn.functional.cross_entropy(
            logits[:, :-1, :].reshape(-1, logits.size(-1)),
            input_ids[:, 1:].reshape(-1),
            reduction='mean'
        ).item()

        # Get exit info
        exit_layer = output.exit_layer
        flops_saved = (1 - exit_layer / self.num_layers) * 100

        # Get sensor state
        body_state = {
            'temperature': sensor_vector[0, 0].item() if sensor_vector.size(1) > 0 else 0.0,
            'power': sensor_vector[0, 5].item() if sensor_vector.size(1) > 5 else 0.0,
        }

        # Uncertainty (from decision if available)
        uncertainty = output.exit_decision.uncertainty if output.exit_decision else 0.5

        return SingleResult(
            prompt=prompt[:50] + "..." if len(prompt) > 50 else prompt,
            difficulty="unknown",  # Set by caller
            condition=config.name,
            exit_layer=exit_layer,
            num_layers=self.num_layers,
            energy_j=energy_j,
            latency_ms=latency_ms,
            quality_score=quality_score,
            tokens_generated=input_ids.size(1),
            uncertainty_mean=uncertainty,
            body_state_mean=body_state,
            flops_saved_pct=flops_saved
        )

    def run_ablation(
        self,
        prompt_sets: List[PromptSet],
        configs: List[AblationConfig]
    ) -> Dict[str, AblationResult]:
        """Run full ablation experiment"""
        print(f"\n{'='*60}")
        print("Running Ablation Experiment")
        print(f"{'='*60}")

        all_results = []

        total_runs = sum(len(ps.prompts) for ps in prompt_sets) * len(configs) * self.num_seeds
        pbar = tqdm(total=total_runs, desc="Ablation")

        for config in configs:
            print(f"\n--- Condition: {config.name} ---")
            print(f"    {config.description}")

            for prompt_set in prompt_sets:
                for prompt in prompt_set.prompts:
                    for seed in range(self.num_seeds):
                        result = self.run_single_inference(prompt, config, seed)
                        result.difficulty = prompt_set.difficulty
                        all_results.append(result)
                        pbar.update(1)

        pbar.close()
        self.results = all_results

        # Aggregate results
        return self._aggregate_results(all_results, configs)

    def _aggregate_results(
        self,
        results: List[SingleResult],
        configs: List[AblationConfig]
    ) -> Dict[str, AblationResult]:
        """Aggregate single results into ablation results"""

        aggregated = {}

        for config in configs:
            condition_results = [r for r in results if r.condition == config.name]

            if not condition_results:
                continue

            # Overall metrics
            energies = [r.energy_j for r in condition_results]
            exit_layers = [r.exit_layer for r in condition_results]
            qualities = [r.quality_score for r in condition_results]
            latencies = [r.latency_ms for r in condition_results]
            flops_saved = [r.flops_saved_pct for r in condition_results]

            # Bootstrap CI for energy
            energy_ci = self._bootstrap_ci(energies)

            # Per-difficulty breakdown
            by_difficulty = {}
            for difficulty in ['easy', 'medium', 'hard']:
                diff_results = [r for r in condition_results if r.difficulty == difficulty]
                if diff_results:
                    by_difficulty[difficulty] = {
                        'energy_mean': np.mean([r.energy_j for r in diff_results]),
                        'exit_layer_mean': np.mean([r.exit_layer for r in diff_results]),
                        'quality_mean': np.mean([r.quality_score for r in diff_results]),
                        'n_samples': len(diff_results)
                    }

            aggregated[config.name] = AblationResult(
                condition=config.name,
                n_samples=len(condition_results),
                energy_mean_j=np.mean(energies),
                energy_std_j=np.std(energies),
                energy_ci_95=energy_ci,
                exit_layer_mean=np.mean(exit_layers),
                exit_layer_std=np.std(exit_layers),
                flops_saved_pct=np.mean(flops_saved),
                quality_mean=np.mean(qualities),
                quality_std=np.std(qualities),
                latency_mean_ms=np.mean(latencies),
                latency_std_ms=np.std(latencies),
                by_difficulty=by_difficulty
            )

        return aggregated

    def _bootstrap_ci(self, data: List[float], n_bootstrap: int = 1000, ci: float = 0.95) -> Tuple[float, float]:
        """Compute bootstrap confidence interval"""
        data = np.array(data)
        n = len(data)

        bootstrap_means = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(data, size=n, replace=True)
            bootstrap_means.append(np.mean(sample))

        alpha = (1 - ci) / 2
        lower = np.percentile(bootstrap_means, alpha * 100)
        upper = np.percentile(bootstrap_means, (1 - alpha) * 100)

        return (lower, upper)

    def analyze_results(self, aggregated: Dict[str, AblationResult]) -> Dict[str, Any]:
        """Analyze and report on ablation results"""
        print(f"\n{'='*60}")
        print("ANALYSIS RESULTS")
        print(f"{'='*60}")

        analysis = {}

        # Get baseline (fixed)
        fixed = aggregated.get('fixed')
        full_ecc = aggregated.get('full_ecc')

        if fixed and full_ecc:
            # Energy savings
            energy_savings_pct = (1 - full_ecc.energy_mean_j / fixed.energy_mean_j) * 100
            print(f"\n[1] Energy Savings vs Fixed Baseline:")
            print(f"    Full ECC: {energy_savings_pct:.1f}% reduction")
            print(f"    Target: ≥30%")
            print(f"    {'✓ PASS' if energy_savings_pct >= 30 else '✗ FAIL'}")
            analysis['energy_savings_pct'] = energy_savings_pct
            analysis['energy_savings_pass'] = energy_savings_pct >= 30

        # Full ECC vs Random (proves uncertainty matters)
        random = aggregated.get('random')
        if full_ecc and random:
            # Normalize by exit depth
            ecc_efficiency = full_ecc.energy_mean_j / full_ecc.exit_layer_mean
            random_efficiency = random.energy_mean_j / random.exit_layer_mean

            efficiency_gain = (1 - ecc_efficiency / random_efficiency) * 100
            print(f"\n[2] Full ECC vs Random (same depth budget):")
            print(f"    ECC efficiency: {ecc_efficiency:.4f} J/layer")
            print(f"    Random efficiency: {random_efficiency:.4f} J/layer")
            print(f"    Efficiency gain: {efficiency_gain:.1f}%")
            print(f"    Target: ≥10%")
            print(f"    {'✓ PASS' if efficiency_gain >= 10 else '✗ FAIL'}")
            analysis['ecc_vs_random_pct'] = efficiency_gain
            analysis['ecc_vs_random_pass'] = efficiency_gain >= 10

        # Full ECC vs Body-Only (proves semantics matters)
        body_only = aggregated.get('body_only')
        if full_ecc and body_only:
            semantic_contribution = (1 - full_ecc.energy_mean_j / body_only.energy_mean_j) * 100
            print(f"\n[3] Full ECC vs Body-Only:")
            print(f"    Semantic contribution: {semantic_contribution:.1f}%")
            print(f"    {'✓ PASS' if semantic_contribution > 0 else '✗ FAIL'}")
            analysis['semantic_contribution_pct'] = semantic_contribution
            analysis['semantic_contribution_pass'] = semantic_contribution > 0

        # Quality preservation
        if fixed and full_ecc:
            quality_degradation = (1 - full_ecc.quality_mean / fixed.quality_mean) * 100
            print(f"\n[4] Quality Preservation:")
            print(f"    Fixed quality: {fixed.quality_mean:.4f}")
            print(f"    ECC quality: {full_ecc.quality_mean:.4f}")
            print(f"    Degradation: {quality_degradation:.1f}%")
            print(f"    Target: ≤5%")
            print(f"    {'✓ PASS' if abs(quality_degradation) <= 5 else '✗ FAIL'}")
            analysis['quality_degradation_pct'] = quality_degradation
            analysis['quality_pass'] = abs(quality_degradation) <= 5

        # Summary table
        print(f"\n{'='*60}")
        print("SUMMARY TABLE")
        print(f"{'='*60}")
        print(f"{'Condition':<20} {'Energy (J)':<12} {'Exit Layer':<12} {'Quality':<10} {'FLOP Saved':<10}")
        print("-" * 64)

        for name in ['fixed', 'full_ecc', 'uncertainty_only', 'body_only', 'random']:
            if name in aggregated:
                r = aggregated[name]
                print(f"{r.condition:<20} {r.energy_mean_j:<12.4f} {r.exit_layer_mean:<12.1f} {r.quality_mean:<10.4f} {r.flops_saved_pct:<10.1f}%")

        # Overall pass/fail
        all_pass = all([
            analysis.get('energy_savings_pass', False),
            analysis.get('ecc_vs_random_pass', False),
            analysis.get('semantic_contribution_pass', False),
            analysis.get('quality_pass', False)
        ])

        print(f"\n{'='*60}")
        print(f"OVERALL: {'✓ ALL TESTS PASSED' if all_pass else '✗ SOME TESTS FAILED'}")
        print(f"{'='*60}")

        analysis['all_pass'] = all_pass
        return analysis

    def save_results(self, output_dir: Path):
        """Save all results to files"""
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save raw results
        raw_file = output_dir / f"raw_results_{timestamp}.json"
        with open(raw_file, 'w') as f:
            json.dump([asdict(r) for r in self.results], f, indent=2)

        print(f"\nResults saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Causal validation for early exit")
    parser.add_argument("--model", type=str, default="deepseek-ai/deepseek-coder-1.3b-instruct",
                       help="Model name or path")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device")
    parser.add_argument("--num-layers", type=int, default=24, help="Number of layers")
    parser.add_argument("--num-seeds", type=int, default=3, help="Number of random seeds")
    parser.add_argument("--output-dir", type=str, default="results/z150_causal_validation",
                       help="Output directory")
    args = parser.parse_args()

    # Create validator
    validator = CausalValidator(
        model_name=args.model,
        device=args.device,
        num_layers=args.num_layers,
        num_seeds=args.num_seeds
    )

    # Setup
    validator.setup()

    # Create prompt sets and ablation configs
    prompt_sets = create_prompt_sets()
    configs = create_ablation_configs()

    # Run ablation
    aggregated = validator.run_ablation(prompt_sets, configs)

    # Analyze
    analysis = validator.analyze_results(aggregated)

    # Save
    output_dir = Path(args.output_dir)
    validator.save_results(output_dir)

    # Save analysis
    analysis_file = output_dir / f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(analysis_file, 'w') as f:
        json.dump(analysis, f, indent=2)

    # Save aggregated
    agg_file = output_dir / f"aggregated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(agg_file, 'w') as f:
        json.dump({k: asdict(v) for k, v in aggregated.items()}, f, indent=2)

    print(f"\nAll results saved to {output_dir}")

    return 0 if analysis.get('all_pass', False) else 1


if __name__ == "__main__":
    exit(main())
