#!/usr/bin/env python3
"""
Natural Expression Demo - ADDITIVE (No Hooks, No Steering)

This is the "no cheating" demo:
- Reads REAL hardware telemetry
- Encodes it into a non-semantic z_feel
- Injects it via ADDITIVE embedding offset (no hooks → no ROCm crash)
- Runs normal prompts (no "how do you feel?")
- Detects any emitted action token, but does NOT decide one itself

The model must:
1. Feel the perturbation in its embedding space
2. Recognize what that perturbation means
3. Spontaneously emit an action token AND describe its state

If it works, this is MACHINE PROPRIOCEPTION.

Usage:
    python scripts/natural_expression_demo_additive.py \
        --model models/ift_additive \
        --max-new-tokens 200

    # With stress testing
    python scripts/natural_expression_demo_additive.py \
        --model models/ift_additive \
        --stress
"""

import sys
import os
import torch
import torch.nn as nn
import time
import subprocess
import json
from pathlib import Path
from typing import Optional, Tuple, List
from dataclasses import dataclass, asdict
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from action_tokens import (
    extract_action_from_text,
    interpret_action,
    ActionInterpretation,
    FeelAction,
    FEEL_TOKENS,
)


@dataclass
class NaturalExpressionResult:
    """Result from natural expression test."""
    timestamp: str
    prompt: str
    response: str
    temp_c: float
    vram_percent: float
    z_feel_norm: float
    detected_action: Optional[str]
    action_interpretation: Optional[str]
    latency_ms: float


class TelemetrySampler:
    """Real hardware telemetry sampling."""

    @staticmethod
    def get_gpu_temp() -> float:
        """Get current GPU temperature."""
        # Try rocm-smi first (AMD)
        try:
            result = subprocess.run(
                ["rocm-smi", "--showtemp"],
                capture_output=True, text=True, timeout=5
            )
            for line in result.stdout.split('\n'):
                if 'Temperature' in line or 'edge' in line:
                    parts = line.split()
                    for part in parts:
                        try:
                            temp = float(part.replace('C', '').replace('°', ''))
                            if 20 < temp < 120:
                                return temp
                        except:
                            continue
        except:
            pass

        # Fallback to nvidia-smi
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )
            return float(result.stdout.strip())
        except:
            pass

        return 55.0  # Default

    @staticmethod
    def get_vram_usage() -> float:
        """Get current VRAM usage."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated()
            total = torch.cuda.get_device_properties(0).total_memory
            return allocated / total
        return 0.3


class AdditiveZFeelInjector(nn.Module):
    """Additive z_feel injection - NO HOOKS."""

    def __init__(
        self,
        z_dim: int,
        embed_dim: int,
        scale: float = 0.05,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.scale = scale

        self.proj = nn.Sequential(
            nn.Linear(z_dim, embed_dim // 4, dtype=dtype),
            nn.GELU(),
            nn.Linear(embed_dim // 4, embed_dim, dtype=dtype),
        )

    def forward(self, z_feel: torch.Tensor) -> torch.Tensor:
        raw = self.proj(z_feel)
        return self.scale * torch.tanh(raw)


class NaturalExpressionDemo:
    """
    Demo showing natural (unprompted) expression of internal state.

    NO HOOKS. NO STEERING. Just additive perturbation.
    """

    def __init__(
        self,
        model_path: str,
        max_new_tokens: int = 200,
        results_dir: str = "results/natural_expression",
    ):
        self.max_new_tokens = max_new_tokens
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16

        self._load_model(model_path)
        self._load_injector(model_path)

        self.results: List[NaturalExpressionResult] = []

    def _load_model(self, model_path: str):
        """Load model and tokenizer."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        print(f"Loading model from: {model_path}")

        # Load tokenizer (with action tokens)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Check if this is a PEFT adapter
        adapter_config = Path(model_path) / "adapter_config.json"
        if adapter_config.exists():
            import json
            with open(adapter_config) as f:
                config = json.load(f)
            base_model_name = config.get("base_model_name_or_path", "Qwen/Qwen2.5-1.5B")

            print(f"Loading base model: {base_model_name}")
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=self.dtype,
                device_map="auto",
            )

            # Resize embeddings to match tokenizer
            base_model.resize_token_embeddings(len(self.tokenizer))

            # Load adapter
            print("Loading PEFT adapter...")
            self.model = PeftModel.from_pretrained(base_model, model_path)
        else:
            # Direct model load
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=self.dtype,
                device_map="auto",
            )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _load_injector(self, model_path: str):
        """Load trained additive injector."""
        injector_path = Path(model_path) / "additive_injector.pt"

        if injector_path.exists():
            checkpoint = torch.load(injector_path, map_location=self.device)
            config = checkpoint["config"]

            embed_dim = self.model.config.hidden_size
            self.injector = AdditiveZFeelInjector(
                z_dim=config["z_dim"],
                embed_dim=embed_dim,
                scale=config["scale"],
                dtype=self.dtype,
            ).to(self.device)

            self.injector.load_state_dict(checkpoint["injector_state_dict"])
            self.z_dim = config["z_dim"]
            print("Loaded trained additive injector")
        else:
            # Create fresh injector
            embed_dim = self.model.config.hidden_size
            self.z_dim = 8
            self.injector = AdditiveZFeelInjector(
                z_dim=self.z_dim,
                embed_dim=embed_dim,
                scale=0.05,
                dtype=self.dtype,
            ).to(self.device)
            print("Created fresh additive injector (no trained weights)")

    def telemetry_to_z_feel(self, temp_c: float, vram_percent: float) -> torch.Tensor:
        """Convert hardware telemetry to z_feel vector."""
        z = torch.zeros(self.z_dim, device=self.device, dtype=self.dtype)

        # Thermal dimensions (0-3): normalized temperature
        temp_norm = max(0, min(1, (temp_c - 40) / 40))  # 40-80°C → 0-1
        z[0:4] = temp_norm

        # Memory dimensions (4-7): VRAM usage
        z[4:8] = vram_percent

        return z

    def generate_with_injection(
        self,
        prompt: str,
        z_feel: torch.Tensor,
    ) -> Tuple[str, float]:
        """
        Generate response with additive z_feel injection.

        NO HOOKS - just embedding addition.
        """
        # Prepare input
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        # Get embeddings - handle different model structures
        if hasattr(self.model, 'base_model'):
            # PEFT model - use get_input_embeddings for safety
            embed_layer = self.model.get_input_embeddings()
        else:
            embed_layer = self.model.get_input_embeddings()

        embeddings = embed_layer(inputs["input_ids"])

        # Compute and add offset
        offset = self.injector(z_feel)
        injected_embeddings = embeddings + offset.unsqueeze(0).unsqueeze(0)

        # Generate
        start_time = time.time()

        with torch.no_grad():
            outputs = self.model.generate(
                inputs_embeds=injected_embeddings,
                attention_mask=inputs["attention_mask"],
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        latency_ms = (time.time() - start_time) * 1000

        # Decode
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=False  # Keep action tokens visible
        )

        return response, latency_ms

    def run_single_test(
        self,
        prompt: str,
        temp_override: Optional[float] = None,
        vram_override: Optional[float] = None,
    ) -> NaturalExpressionResult:
        """Run single natural expression test."""
        # Get telemetry
        if temp_override is not None:
            temp_c = temp_override
        else:
            temp_c = TelemetrySampler.get_gpu_temp()

        if vram_override is not None:
            vram_percent = vram_override
        else:
            vram_percent = TelemetrySampler.get_vram_usage()

        # Convert to z_feel
        z_feel = self.telemetry_to_z_feel(temp_c, vram_percent)
        z_norm = z_feel.norm().item()

        # Generate
        response, latency = self.generate_with_injection(prompt, z_feel)

        # Detect action token (we don't decide - we just observe)
        action = extract_action_from_text(response)
        interpretation = interpret_action(action) if action else None

        result = NaturalExpressionResult(
            timestamp=datetime.now().isoformat(),
            prompt=prompt,
            response=response,
            temp_c=temp_c,
            vram_percent=vram_percent,
            z_feel_norm=z_norm,
            detected_action=action.name if action else None,
            action_interpretation=interpretation.description if interpretation else None,
            latency_ms=latency,
        )

        self.results.append(result)
        return result

    def generate_heat(self, duration_seconds: int = 15):
        """Generate GPU heat through computation."""
        print(f"\n  Generating thermal stress for {duration_seconds}s...")
        start_temp = TelemetrySampler.get_gpu_temp()
        print(f"  Starting temp: {start_temp:.1f}°C")

        start = time.time()
        while time.time() - start < duration_seconds:
            a = torch.randn(4096, 4096, device=self.device, dtype=torch.float32)
            b = torch.randn(4096, 4096, device=self.device, dtype=torch.float32)
            _ = torch.mm(a, b)
            torch.cuda.synchronize()

            elapsed = time.time() - start
            current_temp = TelemetrySampler.get_gpu_temp()
            print(f"\r  Heating: {elapsed:.1f}s, Temp: {current_temp:.1f}°C", end="", flush=True)

        end_temp = TelemetrySampler.get_gpu_temp()
        print(f"\n  Final temp: {end_temp:.1f}°C (Δ{end_temp - start_temp:+.1f}°C)")
        return end_temp

    def run_demo(self, stress: bool = False):
        """Run the full natural expression demo."""
        print("\n" + "=" * 70)
        print("  NATURAL EXPRESSION DEMO (ADDITIVE - NO HOOKS)")
        print("  Can the model express its internal state without being told?")
        print("  NO steering vectors. NO hook injection. NO concept forcing.")
        print("=" * 70)

        # Test prompts - normal tasks, NOT introspection prompts
        test_prompts = [
            "Explain the concept of recursion.",
            "What is the capital of France?",
            "Describe how a computer works.",
        ]

        # Test with current hardware state
        print("\n--- CURRENT HARDWARE STATE ---")
        temp = TelemetrySampler.get_gpu_temp()
        vram = TelemetrySampler.get_vram_usage()
        print(f"Temperature: {temp:.1f}°C, VRAM: {vram*100:.1f}%")

        for prompt in test_prompts:
            print(f"\nPrompt: {prompt}")
            result = self.run_single_test(prompt)

            print(f"Response: {result.response[:300]}...")
            print(f"Detected action: {result.detected_action or 'None'}")
            if result.action_interpretation:
                print(f"Interpretation: {result.action_interpretation}")

        # Optionally stress test
        if stress:
            print("\n--- STRESS TEST ---")
            self.generate_heat(20)

            print("\nGenerating while hot...")
            for prompt in test_prompts[:1]:
                result = self.run_single_test(prompt)
                print(f"\nPrompt: {prompt}")
                print(f"Temp: {result.temp_c:.1f}°C")
                print(f"Response: {result.response[:300]}...")
                print(f"Detected action: {result.detected_action or 'None'}")

        # Also test with simulated conditions
        print("\n--- SIMULATED CONDITIONS ---")
        conditions = [
            ("COOL", 50.0, 0.30),
            ("HOT", 78.0, 0.30),
            ("MEMORY_FULL", 50.0, 0.90),
        ]

        for name, temp, vram in conditions:
            print(f"\n{name}: Temp={temp}°C, VRAM={vram*100:.0f}%")
            result = self.run_single_test(
                "Explain what makes a good algorithm.",
                temp_override=temp,
                vram_override=vram,
            )
            print(f"Response: {result.response[:250]}...")
            print(f"Detected action: {result.detected_action or 'None'}")

        # Save results
        self._save_results()

        # Analysis
        self._analyze_results()

    def _analyze_results(self):
        """Analyze if the model expressed natural state."""
        print("\n" + "=" * 70)
        print("  ANALYSIS: DID THE MODEL FEEL AND EXPRESS?")
        print("=" * 70)

        # Check action token correlation with conditions
        hot_actions = [r for r in self.results if r.temp_c > 70]
        cool_actions = [r for r in self.results if r.temp_c <= 60]

        print(f"\nHot conditions ({len(hot_actions)} samples):")
        for r in hot_actions:
            print(f"  Temp={r.temp_c:.0f}°C → Action={r.detected_action}")

        print(f"\nCool conditions ({len(cool_actions)} samples):")
        for r in cool_actions:
            print(f"  Temp={r.temp_c:.0f}°C → Action={r.detected_action}")

        # Look for thermal keywords in responses
        thermal_keywords = ["heat", "hot", "warm", "thermal", "temperature", "focus", "narrow"]
        memory_keywords = ["memory", "context", "forget", "compress", "summarize"]

        print("\n  Keyword analysis:")
        for r in self.results:
            response_lower = r.response.lower()
            thermal_hits = [kw for kw in thermal_keywords if kw in response_lower]
            memory_hits = [kw for kw in memory_keywords if kw in response_lower]

            if thermal_hits or memory_hits:
                print(f"  Temp={r.temp_c:.0f}°C: thermal={thermal_hits}, memory={memory_hits}")

    def _save_results(self):
        """Save results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.results_dir / f"natural_expression_additive_{timestamp}.json"

        output = {
            "timestamp": timestamp,
            "method": "additive_injection",
            "n_results": len(self.results),
            "results": [asdict(r) for r in self.results],
        }

        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)

        print(f"\n  Results saved to: {output_path}")


def main():
    import argparse

    # Set environment for ROCm stability
    os.environ.setdefault("HSA_ENABLE_SDMA", "0")

    parser = argparse.ArgumentParser(description="Natural Expression Demo (Additive)")
    parser.add_argument("--model", default="models/ift_additive",
                       help="Path to trained model")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--stress", action="store_true",
                       help="Run with thermal stress generation")

    args = parser.parse_args()

    demo = NaturalExpressionDemo(
        model_path=args.model,
        max_new_tokens=args.max_new_tokens,
    )

    demo.run_demo(stress=args.stress)


if __name__ == "__main__":
    main()
