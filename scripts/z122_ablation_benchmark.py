#!/usr/bin/env python3
"""
z122 Ablation Benchmark - Separating Compute vs Power Cap Effects

This benchmark answers the critical question:
"Are energy savings from LayerDrop (compute) or power cap (hardware) or both?"

Ablation Configs:
1. baseline:        No LayerDrop, No power cap change (25W default)
2. compute_only:    LayerDrop ON, Power cap FIXED at 25W
3. powercap_only:   LayerDrop OFF, Power cap ECO (15W)
4. both:            LayerDrop ON, Power cap ECO (15W)

This separates:
- Compute actuation: Fewer layers → less work → faster → less energy
- Power cap actuation: Lower cap → DVFS → slower but more efficient

The cleanest finding would be:
- compute_only shows TPS improvement + some energy reduction
- powercap_only shows TPS reduction + energy reduction
- both shows the combined effect (should be > either alone)
"""

import os
import sys
import json
import time
import argparse
import random
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy import stats

import torch
import torch.nn.functional as F

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from feel_slm.model_v2 import FEELSLMV2, FEELConfigV2
from sensing.background_telemetry import BackgroundTelemetrySampler, TelemetryConfig


# =============================================================================
# Ablation Configuration
# =============================================================================

@dataclass
class AblationConfig:
    """Single ablation configuration."""
    name: str
    layerdrop_enabled: bool  # Compute actuation
    layerdrop_rate: float    # Drop probability (0.0 = no drop)
    power_cap_watts: float   # Hardware actuation (25W = default, 15W = eco, 35W = perf)

    @property
    def is_baseline(self) -> bool:
        return not self.layerdrop_enabled and self.power_cap_watts == 25.0


ABLATION_CONFIGS = [
    AblationConfig("baseline",       layerdrop_enabled=False, layerdrop_rate=0.0, power_cap_watts=25.0),
    AblationConfig("compute_only",   layerdrop_enabled=True,  layerdrop_rate=0.3, power_cap_watts=25.0),
    AblationConfig("powercap_only",  layerdrop_enabled=False, layerdrop_rate=0.0, power_cap_watts=15.0),
    AblationConfig("both",           layerdrop_enabled=True,  layerdrop_rate=0.3, power_cap_watts=15.0),
    # Extra: perf mode for comparison
    AblationConfig("perf_baseline",  layerdrop_enabled=False, layerdrop_rate=0.0, power_cap_watts=35.0),
]


# =============================================================================
# Power Cap Control (via privileged daemon)
# =============================================================================

def set_power_cap(watts: float) -> bool:
    """Set GPU power cap via privileged daemon."""
    try:
        from actuator import PowerClient
        client = PowerClient()
        if client.ping():
            return client.set_power_cap(watts)
    except:
        pass

    # Fallback: try direct sysfs (requires permissions)
    try:
        for i in range(20):
            hwmon = Path(f"/sys/class/hwmon/hwmon{i}")
            if not hwmon.exists():
                continue
            name_file = hwmon / "name"
            if name_file.exists() and "amdgpu" in name_file.read_text():
                cap_file = hwmon / "power1_cap"
                if cap_file.exists():
                    cap_uw = int(watts * 1_000_000)
                    cap_file.write_text(str(cap_uw))
                    print(f"    Power cap set to {watts}W (direct sysfs)")
                    return True
    except PermissionError:
        print(f"    Warning: Cannot set power cap (no permissions)")
        return False
    except Exception as e:
        print(f"    Warning: Power cap error: {e}")
        return False

    return False


def get_power_cap() -> Optional[float]:
    """Read current power cap."""
    try:
        for i in range(20):
            hwmon = Path(f"/sys/class/hwmon/hwmon{i}")
            if not hwmon.exists():
                continue
            name_file = hwmon / "name"
            if name_file.exists() and "amdgpu" in name_file.read_text():
                cap_file = hwmon / "power1_cap"
                if cap_file.exists():
                    return int(cap_file.read_text().strip()) / 1_000_000
    except:
        pass
    return None


# =============================================================================
# Energy Meter (Background Sampling)
# =============================================================================

class AblationEnergyMeter:
    """Energy meter using background sampling for low overhead."""

    def __init__(self, sampler: BackgroundTelemetrySampler):
        self.sampler = sampler
        self._readings: List[Tuple[float, float]] = []
        self._running = False

    def start(self):
        """Start energy integration."""
        self._readings = []
        self._running = True
        self._start_time = time.perf_counter()

    def stop(self) -> Dict:
        """Stop and compute energy."""
        self._running = False
        end_time = time.perf_counter()
        duration = end_time - self._start_time

        # Get latest power reading
        raw = self.sampler.get_latest_raw()
        avg_power_w = raw[0] if raw[0] > 0 else 25.0  # Default if no reading

        # Energy = power × time
        energy_j = avg_power_w * duration

        return {
            "duration_s": duration,
            "avg_power_w": avg_power_w,
            "energy_j": energy_j,
            "energy_mj": energy_j * 1000,
        }


# =============================================================================
# Model Runner
# =============================================================================

class AblationRunner:
    """Runs model with specific ablation config."""

    def __init__(
        self,
        model: FEELSLMV2,
        tokenizer,
        sampler: BackgroundTelemetrySampler,
        device: str = "cuda",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.sampler = sampler
        self.device = device
        self.energy_meter = AblationEnergyMeter(sampler)

    def _configure_layerdrop(self, config: AblationConfig):
        """Configure LayerDrop for the model."""
        # Set drop rate on layers that support it
        drop_layers = [2, 3, 4, 5]  # Middle layers
        for i, layer in enumerate(self.model.layers):
            if hasattr(layer, 'can_drop'):
                layer.can_drop = config.layerdrop_enabled and (i in drop_layers)

        # Store drop rate for generation
        self._layerdrop_rate = config.layerdrop_rate if config.layerdrop_enabled else 0.0

    def _should_drop_layer(self, layer_idx: int) -> bool:
        """Decide whether to drop a layer during generation."""
        if self._layerdrop_rate <= 0:
            return False
        return random.random() < self._layerdrop_rate

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        config: AblationConfig = None,
    ) -> Dict:
        """Generate with specific ablation config."""
        # Configure
        if config:
            self._configure_layerdrop(config)
            set_power_cap(config.power_cap_watts)
            time.sleep(0.1)  # Let power cap settle

        # Encode prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        if hasattr(self.tokenizer, 'model_max_length'):
            max_len = min(32000, self.tokenizer.model_max_length)
            input_ids = input_ids[:, :max_len]
        input_ids = torch.clamp(input_ids, 0, self.model.config.vocab_size - 1)
        input_ids = input_ids.to(self.device)

        prompt_len = input_ids.shape[1]

        # Start energy measurement
        self.energy_meter.start()
        t_start = time.perf_counter()

        # Generate tokens
        generated_ids = input_ids.clone()
        new_tokens = 0
        layers_dropped = 0
        layers_total = 0

        for _ in range(max_new_tokens):
            # Create mask
            seq_len = generated_ids.shape[1]
            mask = torch.triu(
                torch.ones(seq_len, seq_len, device=self.device) * float('-inf'),
                diagonal=1
            )

            # Forward with LayerDrop
            x = self.model.embed_tokens(generated_ids)

            for i, layer in enumerate(self.model.layers):
                drop = self._should_drop_layer(i) if hasattr(layer, 'can_drop') and layer.can_drop else False
                if drop:
                    layers_dropped += 1
                layers_total += 1
                x = layer(x, mask, drop_this_layer=drop)

            x = self.model.norm(x)
            logits = self.model.lm_head(x)

            # Sample next token
            next_logits = logits[:, -1, :]
            next_token = torch.argmax(next_logits, dim=-1, keepdim=True)

            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            new_tokens += 1

            # Stop on EOS
            if next_token.item() == self.tokenizer.eos_token_id:
                break

        # Stop timing
        t_end = time.perf_counter()
        energy_result = self.energy_meter.stop()

        duration = t_end - t_start
        tps = new_tokens / duration if duration > 0 else 0
        mj_per_token = energy_result["energy_mj"] / new_tokens if new_tokens > 0 else 0

        return {
            "prompt_len": prompt_len,
            "new_tokens": new_tokens,
            "duration_s": duration,
            "tps": tps,
            "mj_per_token": mj_per_token,
            "avg_power_w": energy_result["avg_power_w"],
            "energy_mj": energy_result["energy_mj"],
            "layers_dropped": layers_dropped,
            "layers_total": layers_total,
            "drop_rate_actual": layers_dropped / layers_total if layers_total > 0 else 0,
        }


# =============================================================================
# Benchmark Runner
# =============================================================================

def compute_ci(values: List[float], confidence: float = 0.95) -> Tuple[float, float, float]:
    """Compute mean and confidence interval."""
    if len(values) < 2:
        return np.mean(values), 0, 0

    mean = np.mean(values)
    se = stats.sem(values)
    ci = se * stats.t.ppf((1 + confidence) / 2, len(values) - 1)
    return mean, mean - ci, mean + ci


def run_ablation_benchmark(
    model: FEELSLMV2,
    tokenizer,
    sampler: BackgroundTelemetrySampler,
    prompts: List[str],
    configs: List[AblationConfig],
    repeats: int = 3,
    warmup: int = 2,
    device: str = "cuda",
) -> Dict:
    """Run full ablation benchmark."""
    runner = AblationRunner(model, tokenizer, sampler, device)
    results = {c.name: [] for c in configs}

    for rep in range(repeats):
        print(f"\n--- Repeat {rep + 1}/{repeats} ---")

        # Randomize config order
        shuffled = configs.copy()
        random.shuffle(shuffled)

        for config in shuffled:
            print(f"  Running {config.name}...", end=" ", flush=True)

            # Warmup
            for i in range(warmup):
                _ = runner.generate(prompts[i % len(prompts)], max_new_tokens=20, config=config)

            # Actual runs
            run_results = []
            for prompt in prompts:
                result = runner.generate(prompt, max_new_tokens=50, config=config)
                run_results.append(result)

            # Aggregate this run
            avg_tps = np.mean([r["tps"] for r in run_results])
            avg_mj = np.mean([r["mj_per_token"] for r in run_results])
            avg_drop = np.mean([r["drop_rate_actual"] for r in run_results])

            results[config.name].append({
                "tps": avg_tps,
                "mj_per_token": avg_mj,
                "drop_rate": avg_drop,
                "power_cap": config.power_cap_watts,
            })

            print(f"TPS={avg_tps:.1f}, mJ/tok={avg_mj:.1f}, drop={avg_drop:.1%}")

    # Reset power cap to default
    set_power_cap(25.0)

    return results


def format_results(results: Dict, configs: List[AblationConfig]) -> str:
    """Format results as table."""
    lines = []
    lines.append("=" * 100)
    lines.append("ABLATION RESULTS (95% CI)")
    lines.append("=" * 100)
    lines.append(f"{'Config':<18} {'LayerDrop':<12} {'PowerCap':<10} {'TPS':<18} {'mJ/token':<18} {'vs Baseline'}")
    lines.append("-" * 100)

    # Get baseline for comparison
    baseline_tps = np.mean([r["tps"] for r in results.get("baseline", [{"tps": 1}])])
    baseline_mj = np.mean([r["mj_per_token"] for r in results.get("baseline", [{"mj_per_token": 1}])])

    for config in configs:
        runs = results.get(config.name, [])
        if not runs:
            continue

        tps_vals = [r["tps"] for r in runs]
        mj_vals = [r["mj_per_token"] for r in runs]

        tps_mean, tps_lo, tps_hi = compute_ci(tps_vals)
        mj_mean, mj_lo, mj_hi = compute_ci(mj_vals)

        # Compute vs baseline
        tps_delta = ((tps_mean / baseline_tps) - 1) * 100 if baseline_tps > 0 else 0
        mj_delta = ((baseline_mj / mj_mean) - 1) * 100 if mj_mean > 0 else 0

        tps_str = f"{tps_mean:.1f} ± {(tps_hi-tps_lo)/2:.1f}"
        mj_str = f"{mj_mean:.1f} ± {(mj_hi-mj_lo)/2:.1f}"

        ld_str = f"{config.layerdrop_rate:.0%}" if config.layerdrop_enabled else "OFF"
        pc_str = f"{config.power_cap_watts:.0f}W"

        if config.is_baseline:
            vs_str = "-"
        else:
            vs_str = f"TPS:{tps_delta:+.1f}%, E:{mj_delta:+.1f}%"

        lines.append(f"{config.name:<18} {ld_str:<12} {pc_str:<10} {tps_str:<18} {mj_str:<18} {vs_str}")

    lines.append("-" * 100)
    return "\n".join(lines)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="FEEL Ablation Benchmark")
    parser.add_argument("--model-size", choices=["30m", "125m"], default="30m")
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--output-dir", type=str, default="results/z122_ablation")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    print("=" * 60)
    print("FEEL Ablation Benchmark (z122)")
    print("Separating Compute vs Power Cap Effects")
    print("=" * 60)

    # Load tokenizer
    print("\n1. Loading tokenizer...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Load prompts
    print("\n2. Loading prompts...")
    prompts = [
        "Once upon a time in a land far away",
        "The quick brown fox jumps over",
        "In the beginning there was",
        "Scientists have discovered that",
        "The future of artificial intelligence",
        "Deep in the forest there lived",
        "The weather forecast predicts",
        "According to recent studies",
        "A long time ago in a galaxy",
        "The secret to happiness is",
    ][:args.num_samples]
    print(f"   Using {len(prompts)} prompts")

    # Create model
    print(f"\n3. Creating model ({args.model_size})...")
    if args.model_size == "30m":
        config = FEELConfigV2(
            vocab_size=32000,
            hidden_dim=512,
            num_layers=8,
            num_heads=8,
        )
    else:
        config = FEELConfigV2(
            vocab_size=32000,
            hidden_dim=768,
            num_layers=12,
            num_heads=12,
        )

    model = FEELSLMV2(config).to(args.device)
    model.eval()
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    # Start telemetry sampler
    print("\n4. Starting background telemetry...")
    telemetry_config = TelemetryConfig(sample_rate_hz=50)
    sampler = BackgroundTelemetrySampler(telemetry_config)
    sampler.start()
    time.sleep(0.5)  # Let sampler stabilize
    print(f"   Sampler stats: {sampler.get_stats()}")

    # Check power cap control
    print("\n5. Checking power cap control...")
    current_cap = get_power_cap()
    print(f"   Current power cap: {current_cap}W")

    # Run ablation
    print("\n6. Running ablation benchmark...")
    results = run_ablation_benchmark(
        model=model,
        tokenizer=tokenizer,
        sampler=sampler,
        prompts=prompts,
        configs=ABLATION_CONFIGS,
        repeats=args.repeats,
        warmup=args.warmup,
        device=args.device,
    )

    # Stop sampler
    sampler.stop()

    # Format and print results
    print("\n" + format_results(results, ABLATION_CONFIGS))

    # Interpret results
    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)

    baseline = results.get("baseline", [{}])[0] if results.get("baseline") else {}
    compute = results.get("compute_only", [{}])[0] if results.get("compute_only") else {}
    powercap = results.get("powercap_only", [{}])[0] if results.get("powercap_only") else {}
    both = results.get("both", [{}])[0] if results.get("both") else {}

    if baseline and compute:
        compute_effect_tps = (compute.get("tps", 0) / baseline.get("tps", 1) - 1) * 100
        compute_effect_energy = (1 - compute.get("mj_per_token", 1) / baseline.get("mj_per_token", 1)) * 100
        print(f"\nCompute actuation (LayerDrop only):")
        print(f"  TPS effect:    {compute_effect_tps:+.1f}%")
        print(f"  Energy effect: {compute_effect_energy:+.1f}% reduction")

    if baseline and powercap:
        powercap_effect_tps = (powercap.get("tps", 0) / baseline.get("tps", 1) - 1) * 100
        powercap_effect_energy = (1 - powercap.get("mj_per_token", 1) / baseline.get("mj_per_token", 1)) * 100
        print(f"\nPower cap actuation (15W cap only):")
        print(f"  TPS effect:    {powercap_effect_tps:+.1f}%")
        print(f"  Energy effect: {powercap_effect_energy:+.1f}% reduction")

    if baseline and both:
        both_effect_tps = (both.get("tps", 0) / baseline.get("tps", 1) - 1) * 100
        both_effect_energy = (1 - both.get("mj_per_token", 1) / baseline.get("mj_per_token", 1)) * 100
        print(f"\nCombined (LayerDrop + 15W cap):")
        print(f"  TPS effect:    {both_effect_tps:+.1f}%")
        print(f"  Energy effect: {both_effect_energy:+.1f}% reduction")

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert numpy types to native Python for JSON serialization
    def convert_to_native(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(i) for i in obj]
        return obj

    with open(output_dir / "ablation_results.json", "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "configs": [asdict(c) for c in ABLATION_CONFIGS],
            "results": convert_to_native(results),
        }, f, indent=2)

    print(f"\nResults saved to {output_dir}/")
    print("\n" + "=" * 60)
    print("Ablation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
