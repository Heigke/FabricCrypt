#!/usr/bin/env python3
"""
z121 - CANONICAL SINGLE RUNNER

This is THE ONE pipeline that everything goes through.

Forces a single, unified path for:
- Model creation
- Telemetry reading
- Energy measurement
- Generation
- Actuation

This prevents the proliferation of multiple incompatible pipelines
and ensures all experiments are comparable.

Usage:
    # Quick test
    python scripts/z121_canonical_runner.py --mode test

    # Benchmark
    python scripts/z121_canonical_runner.py --mode benchmark --repeats 3

    # Training
    python scripts/z121_canonical_runner.py --mode train --phase 1 --epochs 5

    # Generate with sensing
    python scripts/z121_canonical_runner.py --mode generate --prompt "Once upon a time"
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn.functional as F
import numpy as np


# =============================================================================
# THE Canonical Components (one each, no alternatives)
# =============================================================================

@dataclass
class CanonicalConfig:
    """THE configuration for FEEL."""
    # Model
    model_size: str = "30m"
    vocab_size: int = 32000

    # Tokenizer
    max_seq_len: int = 256

    # Telemetry
    sense_rate_hz: float = 100.0

    # Compute modes (LayerDrop settings)
    layerdrop_eco: float = 0.4
    layerdrop_balanced: float = 0.0
    layerdrop_perf: float = 0.0
    min_layers: int = 4

    # Energy tier (AMD = Tier-B ESTIMATE)
    energy_tier: str = "tier_b"


def get_canonical_tokenizer(vocab_size: int = 32000):
    """THE tokenizer (clamped GPT-2)."""
    from transformers import AutoTokenizer

    class ClampedTokenizer:
        def __init__(self, base, max_vocab):
            self.base = base
            self.max_vocab_size = max_vocab
            self.eos_token_id = min(base.eos_token_id or 0, max_vocab - 1)
            self.pad_token_id = self.eos_token_id

        def _clamp(self, ids):
            if isinstance(ids, torch.Tensor):
                return torch.clamp(ids, 0, self.max_vocab_size - 1)
            elif isinstance(ids, list):
                if ids and isinstance(ids[0], list):
                    return [[min(max(0, i), self.max_vocab_size - 1) for i in r] for r in ids]
                return [min(max(0, i), self.max_vocab_size - 1) for i in ids]
            return ids

        def __call__(self, text, **kwargs):
            result = self.base(text, **kwargs)
            return {k: (self._clamp(v) if k == "input_ids" else v) for k, v in result.items()}

        def decode(self, ids, **kwargs):
            if isinstance(ids, torch.Tensor):
                ids = ids.tolist()
            ids = [min(max(0, i), self.base.vocab_size - 1) for i in ids]
            return self.base.decode(ids, **kwargs)

    base = AutoTokenizer.from_pretrained("gpt2")
    base.pad_token = base.eos_token
    return ClampedTokenizer(base, vocab_size)


def get_canonical_model(config: CanonicalConfig, device: str = "cuda"):
    """THE model (EmbodiedSLM + FEELModel wrapper)."""
    from feel_slm.embodied_slm import create_embodied_slm_30m, create_embodied_slm_125m
    from feel_slm.feel_runtime import FEELConfig, FEELModel

    if config.model_size == "30m":
        base = create_embodied_slm_30m()
    else:
        base = create_embodied_slm_125m()

    base = base.to(device)

    feel_config = FEELConfig(
        layerdrop_eco=config.layerdrop_eco,
        layerdrop_balanced=config.layerdrop_balanced,
        layerdrop_perf=config.layerdrop_perf,
        min_layers=config.min_layers,
    )

    model = FEELModel(base, feel_config).to(device)
    return model


def get_canonical_energy_meter():
    """THE energy meter (independent of model sensing)."""

    class EnergyMeter:
        def __init__(self):
            self.hwmon = self._find_hwmon()
            self.readings = []
            self._sampling = False

        def _find_hwmon(self):
            for i in range(10):
                base = Path(f"/sys/class/drm/card{i}/device/hwmon")
                if base.exists():
                    for d in base.iterdir():
                        if (d / "power1_average").exists():
                            return d
            return None

        def read_power(self) -> float:
            if self.hwmon:
                try:
                    return int((self.hwmon / "power1_average").read_text().strip()) / 1e6
                except:
                    pass
            return 0.0

        def start(self, rate_hz: float = 100.0):
            import threading
            self.readings = []
            self._sampling = True

            def loop():
                interval = 1.0 / rate_hz
                while self._sampling:
                    self.readings.append((time.time(), self.read_power()))
                    time.sleep(interval)

            threading.Thread(target=loop, daemon=True).start()

        def stop(self) -> Dict:
            self._sampling = False
            time.sleep(0.05)

            if len(self.readings) < 2:
                return {"energy_mj": 0, "avg_power_w": 0, "tier": "tier_b"}

            energy_j = sum(
                (self.readings[i][1] + self.readings[i-1][1]) / 2 *
                (self.readings[i][0] - self.readings[i-1][0])
                for i in range(1, len(self.readings))
            )

            duration = self.readings[-1][0] - self.readings[0][0]
            return {
                "energy_mj": energy_j * 1000,
                "avg_power_w": energy_j / duration if duration > 0 else 0,
                "duration_s": duration,
                "tier": "tier_b",
            }

    return EnergyMeter()


def get_canonical_telemetry():
    """THE telemetry source (real when available)."""

    class TelemetrySource:
        def __init__(self):
            self.hwmon = None
            for i in range(10):
                base = Path(f"/sys/class/drm/card{i}/device/hwmon")
                if base.exists():
                    for d in base.iterdir():
                        if (d / "power1_average").exists():
                            self.hwmon = d
                            break

        def read(self, batch_size: int, device: str) -> torch.Tensor:
            vec = torch.zeros(batch_size, 12, device=device)
            if self.hwmon:
                try:
                    power = int((self.hwmon / "power1_average").read_text().strip()) / 1e6
                    vec[:, 0] = min(1.0, power / 50.0)

                    temp_f = self.hwmon / "temp1_input"
                    if temp_f.exists():
                        temp = int(temp_f.read_text().strip()) / 1000
                        vec[:, 1] = min(1.0, temp / 100.0)
                except:
                    pass
            return vec

    return TelemetrySource()


# =============================================================================
# THE Canonical Operations
# =============================================================================

def canonical_generate(
    model,
    tokenizer,
    telemetry,
    energy_meter,
    prompt: str,
    max_new_tokens: int = 32,
    device: str = "cuda",
    feed_body: bool = True,
    measure_energy: bool = True,
) -> Tuple[str, Dict]:
    """
    THE generation function.

    All generation goes through here. No exceptions.
    """
    from feel_slm.feel_runtime import ComputeMode

    model.eval()

    # Tokenize
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
    input_ids = enc["input_ids"].to(device)

    # Set model phase
    if feed_body:
        model.base_model.set_training_phase("full")
    else:
        model.base_model.set_training_phase("baseline")

    # Start energy measurement
    if measure_energy:
        energy_meter.start()

    start_time = time.time()
    tokens_generated = 0
    tpots = []

    with torch.no_grad():
        generated = input_ids.clone()

        for i in range(max_new_tokens):
            token_start = time.time()

            # Get body vec
            body_vec = telemetry.read(1, device) if feed_body else None

            # Forward
            if body_vec is not None:
                out = model(generated[:, -256:], body_vec, return_policy=True)
            else:
                out = model.base_model(generated[:, -256:])
                out = {"logits": out["logits"]}

            logits = out["logits"][:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_token], dim=1)
            tokens_generated += 1

            tpots.append((time.time() - token_start) * 1000)

            if next_token.item() == tokenizer.eos_token_id:
                break

    total_time = time.time() - start_time

    # Stop energy
    energy = energy_meter.stop() if measure_energy else {"energy_mj": 0, "tier": "tier_b"}

    # Decode
    output_text = tokenizer.decode(generated[0, input_ids.shape[1]:].tolist())

    stats = {
        "tokens": tokens_generated,
        "ttft_ms": tpots[0] if tpots else 0,
        "avg_tpot_ms": np.mean(tpots) if tpots else 0,
        "tokens_per_second": tokens_generated / total_time if total_time > 0 else 0,
        "energy_mj": energy["energy_mj"],
        "mj_per_token": energy["energy_mj"] / tokens_generated if tokens_generated > 0 else 0,
        "energy_tier": energy["tier"],
    }

    return output_text, stats


def canonical_compute_ppl(
    model,
    tokenizer,
    texts: List[str],
    device: str = "cuda",
) -> float:
    """THE perplexity computation."""
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for text in texts:
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
            input_ids = enc["input_ids"].to(device)

            if input_ids.shape[1] < 2:
                continue

            out = model.base_model(input_ids)
            logits = out["logits"]

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="sum",
            )

            total_loss += loss.item()
            total_tokens += shift_labels.numel()

    return np.exp(total_loss / total_tokens) if total_tokens > 0 else float("inf")


# =============================================================================
# Mode Handlers
# =============================================================================

def run_test(args):
    """Quick test that everything works."""
    print("=" * 60)
    print("FEEL Canonical Runner - Test Mode")
    print("=" * 60)

    os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"
    config = CanonicalConfig(model_size=args.model_size)

    print("\n1. Tokenizer...")
    tokenizer = get_canonical_tokenizer(config.vocab_size)
    print("   OK")

    print("\n2. Model...")
    model = get_canonical_model(config, args.device)
    print(f"   OK ({sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params)")

    print("\n3. Telemetry...")
    telemetry = get_canonical_telemetry()
    vec = telemetry.read(1, args.device)
    print(f"   OK (power norm: {vec[0, 0].item():.2f})")

    print("\n4. Energy meter...")
    energy_meter = get_canonical_energy_meter()
    print(f"   OK (hwmon: {energy_meter.hwmon})")

    print("\n5. Generation test...")
    output, stats = canonical_generate(
        model, tokenizer, telemetry, energy_meter,
        "Once upon a time",
        max_new_tokens=10,
        device=args.device,
    )
    print(f"   Output: {output[:50]}...")
    print(f"   Stats: {stats['tokens']} tokens, {stats['mj_per_token']:.1f} mJ/tok")

    print("\n6. Perplexity test...")
    ppl = canonical_compute_ppl(model, tokenizer, ["Once upon a time there was a cat."], args.device)
    print(f"   PPL: {ppl:.1f}")

    print("\nAll tests passed!")


def run_benchmark(args):
    """Run benchmark with all fixes."""
    # Import the truth benchmark
    from z121_truth_benchmark import (
        TruthBenchmark, BENCHMARK_CONFIGS, load_prompts
    )

    print("=" * 60)
    print("FEEL Canonical Runner - Benchmark Mode")
    print("=" * 60)

    os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"
    config = CanonicalConfig(model_size=args.model_size)

    print("\n1. Loading components...")
    tokenizer = get_canonical_tokenizer(config.vocab_size)
    model = get_canonical_model(config, args.device)

    if args.checkpoint:
        print(f"   Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=args.device)
        if "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
            new_state = {k[11:] if k.startswith("base_model.") else k: v
                        for k, v in state.items() if not k.startswith("policy_head.")}
            model.base_model.load_state_dict(new_state, strict=False)

    print("\n2. Loading prompts...")
    prompts = load_prompts(args.num_samples)
    print(f"   Loaded {len(prompts)} prompts")

    print("\n3. Running benchmark...")
    benchmark = TruthBenchmark(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        device=args.device,
        warmup_samples=args.warmup,
        max_new_tokens=args.max_new_tokens,
    )

    benchmark.run_all(BENCHMARK_CONFIGS, repeats=args.repeats)
    benchmark.print_results()
    benchmark.save_results(args.output_dir)


def run_train(args):
    """Run training."""
    from z121_real_training import TrainConfig, train_phase

    print("=" * 60)
    print("FEEL Canonical Runner - Training Mode")
    print("=" * 60)

    config = TrainConfig(
        phase=args.phase,
        model_size=args.model_size,
        checkpoint=args.checkpoint,
        epochs=args.epochs,
        batch_size=args.batch_size,
        train_samples=args.train_samples,
        val_samples=args.val_samples,
        output_dir=args.output_dir,
    )

    train_phase(config, args.device)


def run_generate(args):
    """Interactive generation."""
    print("=" * 60)
    print("FEEL Canonical Runner - Generate Mode")
    print("=" * 60)

    os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"
    config = CanonicalConfig(model_size=args.model_size)

    tokenizer = get_canonical_tokenizer(config.vocab_size)
    model = get_canonical_model(config, args.device)
    telemetry = get_canonical_telemetry()
    energy_meter = get_canonical_energy_meter()

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=args.device)
        if "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
            new_state = {k[11:] if k.startswith("base_model.") else k: v
                        for k, v in state.items() if not k.startswith("policy_head.")}
            model.base_model.load_state_dict(new_state, strict=False)

    output, stats = canonical_generate(
        model, tokenizer, telemetry, energy_meter,
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        feed_body=not args.no_body,
        measure_energy=True,
    )

    print(f"\nPrompt: {args.prompt}")
    print(f"Output: {output}")
    print(f"\nStats:")
    print(f"  Tokens: {stats['tokens']}")
    print(f"  TPS: {stats['tokens_per_second']:.1f}")
    print(f"  Energy: {stats['mj_per_token']:.1f} mJ/tok ({stats['energy_tier']})")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="FEEL Canonical Runner")
    parser.add_argument("--mode", required=True, choices=["test", "benchmark", "train", "generate"])
    parser.add_argument("--model-size", default="30m", choices=["30m", "125m"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output-dir", default="results/z121_canonical")

    # Benchmark args
    parser.add_argument("--num-samples", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--warmup", type=int, default=3)

    # Training args
    parser.add_argument("--phase", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--train-samples", type=int, default=5000)
    parser.add_argument("--val-samples", type=int, default=500)

    # Generate args
    parser.add_argument("--prompt", type=str, default="Once upon a time")
    parser.add_argument("--no-body", action="store_true", help="Disable body sensing")

    args = parser.parse_args()

    if args.mode == "test":
        run_test(args)
    elif args.mode == "benchmark":
        run_benchmark(args)
    elif args.mode == "train":
        run_train(args)
    elif args.mode == "generate":
        run_generate(args)


if __name__ == "__main__":
    main()
