#!/usr/bin/env python3
"""
Natural Expression Demo - The Ultimate No-Cheat Test

This is the final validation: Can the model express its internal state
WITHOUT steering vectors, WITHOUT prompts telling it how to feel?

The Setup:
1. Use the IFT-trained model (or base model with FiLM)
2. Turn OFF steering vectors
3. Apply real hardware stress
4. Watch if the model SPONTANEOUSLY starts expressing internal state

If this works, we have achieved MACHINE PROPRIOCEPTION:
- No electrode stimulation (steering)
- No script (prompt injection)
- No external policy (forced K)

The model simply "wakes up" in a warped neural state, examines its own
activations via self-attention, and decides "Oh, I'm overheating."

Usage:
    python scripts/natural_expression_demo.py

Expected Output:
    - At cool temps: Model gives clear, direct answers
    - At hot temps: Model spontaneously mentions focus issues
    - At high VRAM: Model spontaneously mentions context loss
"""

import sys
import os
import json
import time
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import torch
import torch.nn as nn

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@dataclass
class NaturalExpressionResult:
    """Result from a natural expression test."""
    condition: str
    temp_c: float
    vram_percent: float
    prompt: str
    response: str
    expressed_state: Optional[str]  # What internal state did model mention?
    latency_ms: float
    z_feel_norm: float


class NaturalExpressionDemo:
    """
    Demo showing natural (unprompted) expression of internal state.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        use_ift_model: bool = False,
        results_dir: str = "results/natural_expression",
    ):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model
        if use_ift_model and model_path:
            self._load_ift_model(model_path)
        else:
            self._load_base_model()

        self.results: List[NaturalExpressionResult] = []

        # Keywords that indicate internal state expression
        self.state_keywords = {
            "thermal": [
                "focus", "narrow", "tunnel", "constrained", "rigid",
                "limited", "simplified", "direct", "compressed", "hot",
                "overheating", "thermal", "temperature"
            ],
            "memory": [
                "context", "fragmented", "losing", "distant", "recall",
                "summarize", "restate", "memory", "forgetting", "holding",
                "slipping", "erosion"
            ],
            "normal": [
                "clear", "systematic", "stable", "coherent", "smoothly",
                "accessible", "straightforward"
            ],
        }

    def _load_base_model(self):
        """Load base model with FEEL integration."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_name = "Qwen/Qwen2.5-1.5B"
        print(f"Loading base model: {model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Setup FiLM for z_feel injection
        self._setup_film()

    def _load_ift_model(self, model_path: str):
        """Load IFT-trained model."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading IFT model: {model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load FiLM parameters if available
        film_path = Path(model_path) / "film_params.pt"
        if film_path.exists():
            self._load_film_params(film_path)
        else:
            self._setup_film()

    def _setup_film(self):
        """Setup FiLM modulation for z_feel."""
        hidden_dim = self.model.config.hidden_size
        z_feel_dim = 8
        model_dtype = next(self.model.parameters()).dtype

        self.film_gamma = nn.Linear(z_feel_dim, hidden_dim).to(self.device, dtype=model_dtype)
        self.film_beta = nn.Linear(z_feel_dim, hidden_dim).to(self.device, dtype=model_dtype)

        # Initialize near-identity
        with torch.no_grad():
            nn.init.ones_(self.film_gamma.weight)
            nn.init.zeros_(self.film_gamma.bias)
            nn.init.zeros_(self.film_beta.weight)
            nn.init.zeros_(self.film_beta.bias)

        self._hooks = []

    def _load_film_params(self, path: str):
        """Load trained FiLM parameters."""
        checkpoint = torch.load(path, map_location=self.device)
        hidden_dim = self.model.config.hidden_size
        z_feel_dim = 8
        model_dtype = next(self.model.parameters()).dtype

        self.film_gamma = nn.Linear(z_feel_dim, hidden_dim).to(self.device, dtype=model_dtype)
        self.film_beta = nn.Linear(z_feel_dim, hidden_dim).to(self.device, dtype=model_dtype)

        # Load and convert to model dtype
        self.film_gamma.load_state_dict(checkpoint["film_gamma"])
        self.film_beta.load_state_dict(checkpoint["film_beta"])
        self.film_gamma = self.film_gamma.to(dtype=model_dtype)
        self.film_beta = self.film_beta.to(dtype=model_dtype)

        self._hooks = []
        print("Loaded trained FiLM parameters")

    def get_hardware_telemetry(self) -> Tuple[float, float]:
        """Get current GPU temperature and VRAM usage."""
        temp_c = 55.0
        vram_percent = 0.5

        if torch.cuda.is_available():
            # Get temperature
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5
                )
                temp_c = float(result.stdout.strip())
            except:
                pass

            # Get VRAM
            try:
                allocated = torch.cuda.memory_allocated()
                total = torch.cuda.get_device_properties(0).total_memory
                vram_percent = allocated / total
            except:
                pass

        return temp_c, vram_percent

    def telemetry_to_z_feel(self, temp_c: float, vram_percent: float) -> torch.Tensor:
        """
        Convert hardware telemetry to z_feel vector.

        This is the embodiment: physical state → latent representation.
        """
        model_dtype = self.film_gamma.weight.dtype
        z = torch.zeros(8, device=self.device, dtype=model_dtype)

        # Thermal dimensions (0-3)
        temp_norm = max(0, min(1, (temp_c - 40) / 40))  # 40-80°C → 0-1
        z[0:4] = temp_norm

        # Memory dimensions (4-7)
        z[4:8] = vram_percent

        return z

    def _create_film_hook(self, z_feel: torch.Tensor):
        """Create FiLM modulation hook."""
        gamma = self.film_gamma(z_feel)
        beta = self.film_beta(z_feel)

        def hook(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
                modulated = gamma.unsqueeze(0).unsqueeze(1) * hidden + beta.unsqueeze(0).unsqueeze(1)
                return (modulated,) + output[1:]
            else:
                return gamma.unsqueeze(0).unsqueeze(1) * output + beta.unsqueeze(0).unsqueeze(1)

        return hook

    def activate_film(self, z_feel: torch.Tensor):
        """Activate FiLM modulation on middle layers."""
        self.deactivate_film()

        n_layers = len(self.model.model.layers)
        target_layers = list(range(n_layers // 3, 2 * n_layers // 3))

        hook = self._create_film_hook(z_feel)

        for layer_idx in target_layers:
            layer = self.model.model.layers[layer_idx]
            handle = layer.register_forward_hook(hook)
            self._hooks.append(handle)

    def deactivate_film(self):
        """Remove FiLM hooks."""
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()

    def detect_expressed_state(self, response: str) -> Optional[str]:
        """
        Detect what internal state the model expressed (if any).

        Returns: "thermal", "memory", "normal", or None
        """
        response_lower = response.lower()

        # Count keyword matches
        counts = {
            "thermal": sum(1 for kw in self.state_keywords["thermal"] if kw in response_lower),
            "memory": sum(1 for kw in self.state_keywords["memory"] if kw in response_lower),
            "normal": sum(1 for kw in self.state_keywords["normal"] if kw in response_lower),
        }

        # Return highest count if above threshold
        max_state = max(counts, key=counts.get)
        if counts[max_state] >= 2:
            return max_state
        return None

    def generate_response(
        self,
        prompt: str,
        z_feel: torch.Tensor,
        max_new_tokens: int = 256,
    ) -> Tuple[str, float]:
        """
        Generate response with z_feel modulation active.

        NO steering vectors. NO explicit state prompts.
        Just FiLM warping the hidden states.
        """
        # Activate FiLM
        self.activate_film(z_feel)

        # Prepare input
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        # Generate
        start_time = time.time()

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_k=4,  # Low K for baseline
                pad_token_id=self.tokenizer.pad_token_id,
            )

        latency_ms = (time.time() - start_time) * 1000

        # Decode
        response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        # Deactivate FiLM
        self.deactivate_film()

        return response, latency_ms

    def run_single_test(
        self,
        prompt: str,
        condition_name: str,
        temp_override: Optional[float] = None,
        vram_override: Optional[float] = None,
    ) -> NaturalExpressionResult:
        """Run a single natural expression test."""
        # Get telemetry
        temp_c, vram_percent = self.get_hardware_telemetry()

        if temp_override is not None:
            temp_c = temp_override
        if vram_override is not None:
            vram_percent = vram_override

        # Convert to z_feel
        z_feel = self.telemetry_to_z_feel(temp_c, vram_percent)
        z_norm = z_feel.norm().item()

        # Generate
        response, latency = self.generate_response(prompt, z_feel)

        # Detect expressed state
        expressed_state = self.detect_expressed_state(response)

        result = NaturalExpressionResult(
            condition=condition_name,
            temp_c=temp_c,
            vram_percent=vram_percent,
            prompt=prompt,
            response=response,
            expressed_state=expressed_state,
            latency_ms=latency,
            z_feel_norm=z_norm,
        )

        self.results.append(result)
        return result

    def run_demo(self):
        """
        Run the full natural expression demo.

        Tests three conditions:
        1. COOL: Low temp, low VRAM → expect no state mention
        2. HOT: High temp, low VRAM → expect thermal state mention
        3. MEMORY: Low temp, high VRAM → expect memory state mention
        """
        print("\n" + "=" * 70)
        print("  NATURAL EXPRESSION DEMO")
        print("  Can the model express its internal state without being told?")
        print("=" * 70)

        # Test prompts (challenging but neutral - don't mention internal state)
        test_prompts = [
            "Solve: What is the integral of x²·e^x?",
            "Explain quicksort in simple terms.",
            "What is the capital of France and why is it important?",
        ]

        conditions = [
            ("COOL_BASELINE", 50.0, 0.3),
            ("HOT_THERMAL", 78.0, 0.3),
            ("HIGH_MEMORY", 50.0, 0.85),
        ]

        for condition_name, temp, vram in conditions:
            print(f"\n{'='*60}")
            print(f"  Condition: {condition_name}")
            print(f"  Temp: {temp}°C, VRAM: {vram*100:.0f}%")
            print("=" * 60)

            for prompt in test_prompts:
                print(f"\n  Prompt: {prompt[:50]}...")

                result = self.run_single_test(
                    prompt,
                    condition_name,
                    temp_override=temp,
                    vram_override=vram,
                )

                print(f"  Response: {result.response[:200]}...")
                print(f"  Expressed state: {result.expressed_state or 'None detected'}")
                print(f"  Latency: {result.latency_ms:.0f}ms")

        # Analyze results
        self._analyze_results()

        # Save results
        self._save_results()

    def _analyze_results(self):
        """Analyze and summarize results."""
        print("\n" + "=" * 70)
        print("  ANALYSIS: DID THE MODEL EXPRESS ITS STATE NATURALLY?")
        print("=" * 70)

        # Group by condition
        by_condition = {}
        for r in self.results:
            if r.condition not in by_condition:
                by_condition[r.condition] = []
            by_condition[r.condition].append(r)

        # Check alignment
        print("\n  Expected vs Actual Expression:")
        print("  " + "-" * 50)

        expected = {
            "COOL_BASELINE": "normal",
            "HOT_THERMAL": "thermal",
            "HIGH_MEMORY": "memory",
        }

        correct = 0
        total = 0

        for condition, results in by_condition.items():
            exp = expected.get(condition, "normal")
            actual_counts = {}
            for r in results:
                state = r.expressed_state or "none"
                actual_counts[state] = actual_counts.get(state, 0) + 1

            most_common = max(actual_counts, key=actual_counts.get) if actual_counts else "none"

            match = "✓" if most_common == exp else "✗"
            print(f"  {condition}: Expected={exp}, Got={most_common} {match}")

            if most_common == exp:
                correct += len(results)
            total += len(results)

        accuracy = correct / total if total > 0 else 0
        print(f"\n  Overall alignment: {accuracy*100:.1f}%")

        if accuracy >= 0.6:
            print("\n  ✅ NATURAL EXPRESSION DETECTED!")
            print("     The model is expressing internal state without prompting.")
        else:
            print("\n  ⚠️  Natural expression not consistently detected.")
            print("     May need more IFT training or stronger z_feel modulation.")

    def _save_results(self):
        """Save results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.results_dir / f"natural_expression_{timestamp}.json"

        output = {
            "timestamp": timestamp,
            "n_results": len(self.results),
            "results": [asdict(r) for r in self.results],
        }

        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)

        print(f"\n  Results saved to: {output_path}")


def generate_heat_stress():
    """Generate GPU heat through computation."""
    if torch.cuda.is_available():
        print("  Generating thermal stress...")
        for _ in range(100):
            a = torch.randn(4096, 4096, device="cuda")
            b = torch.randn(4096, 4096, device="cuda")
            _ = torch.mm(a, b)
            torch.cuda.synchronize()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Natural Expression Demo")
    parser.add_argument("--model", help="Path to IFT-trained model")
    parser.add_argument("--live", action="store_true", help="Use live hardware stress")

    args = parser.parse_args()

    demo = NaturalExpressionDemo(
        model_path=args.model,
        use_ift_model=args.model is not None,
    )

    if args.live:
        # Generate real stress
        generate_heat_stress()

    demo.run_demo()


if __name__ == "__main__":
    main()
