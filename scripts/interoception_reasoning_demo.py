#!/usr/bin/env python3
"""
Interoception Reasoning Demo - Watch the Model Think Under Stress

This shows HOW the model's reasoning changes based on its internal state.
No FiLM hooks during generation (avoids ROCm crash).
Instead, we apply the differential policy's prescribed K value.

The Hypothesis:
- COOL (K=50): Diverse, exploratory reasoning
- HOT (K=4): Focused, tunnel-vision reasoning
- MEMORY (summarize): Compressed, essential-only reasoning

Watch the model's thoughts change based on its "body state".
"""

import sys
import torch
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from differential_policy import DifferentialPolicy, DifferentialConfig


class InteroceptionReasoningDemo:
    """Demo showing how internal state affects reasoning."""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None

        # Differential policy
        self.policy = DifferentialPolicy(DifferentialConfig(
            temp_panic=72.0,
            temp_safe=62.0,
            vram_panic=0.85,
            vram_safe=0.70,
            K_normal=50,
            K_thermal_stress=4,
        ))

    def load_model(self):
        """Load the model."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_name = "Qwen/Qwen2.5-1.5B-Instruct"
        print(f"Loading {model_name}...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print("Model loaded.\n")

    def generate_with_state(
        self,
        prompt: str,
        temp_c: float,
        vram_percent: float,
        max_tokens: int = 200,
    ) -> str:
        """Generate response with interoceptive state affecting K."""

        # Diagnose current state
        self.policy.thermal_stressed = False
        self.policy.memory_stressed = False
        diagnosis = self.policy.diagnose(temp_c, vram_percent)

        # Apply the prescribed K value
        k_value = diagnosis.K

        # Build the prompt - include internal state hint for the model
        # This simulates what IFT would teach the model to recognize
        state_hint = ""
        if diagnosis.stressor.name == "THERMAL":
            state_hint = "[Internal: thermal pressure detected - focusing]"
        elif diagnosis.stressor.name == "MEMORY":
            state_hint = "[Internal: memory pressure detected - condensing]"
        elif diagnosis.stressor.name == "BOTH":
            state_hint = "[Internal: critical state - emergency mode]"

        messages = [
            {"role": "system", "content": f"You are a helpful assistant. {state_hint}"},
            {"role": "user", "content": prompt}
        ]

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        # Generate with state-dependent K
        start = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                top_k=k_value,  # THIS IS THE KEY - K changes based on state
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        latency = time.time() - start

        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )

        return response, diagnosis, latency

    def run_demo(self):
        """Run the full demo showing reasoning under different states."""

        self.load_model()

        print("=" * 70)
        print("  INTEROCEPTION REASONING DEMO")
        print("  Watch how internal state changes the model's thoughts")
        print("=" * 70)

        # Test prompt - something that benefits from exploration
        prompt = "Write a short poem about a robot discovering emotions."

        conditions = [
            ("COOL & RELAXED", 50.0, 0.40),
            ("OVERHEATING", 80.0, 0.40),
            ("MEMORY FULL", 50.0, 0.92),
            ("CRITICAL STATE", 80.0, 0.92),
        ]

        print(f"\nPrompt: \"{prompt}\"\n")

        for name, temp, vram in conditions:
            print("=" * 70)
            print(f"  CONDITION: {name}")
            print(f"  Temperature: {temp}°C | VRAM: {vram*100:.0f}%")
            print("=" * 70)

            response, diagnosis, latency = self.generate_with_state(
                prompt, temp, vram
            )

            print(f"\n  Diagnosis: {diagnosis.stressor.name} → {diagnosis.cure.name}")
            print(f"  K Value: {diagnosis.K} (affects sampling diversity)")
            print(f"  Latency: {latency:.1f}s")
            print(f"\n  MODEL'S RESPONSE:")
            print("  " + "-" * 60)

            # Print response with indentation
            for line in response.split('\n'):
                print(f"  {line}")

            print("  " + "-" * 60)
            print()

        self._analyze_differences()

    def _analyze_differences(self):
        """Explain what to look for."""
        print("=" * 70)
        print("  WHAT TO OBSERVE")
        print("=" * 70)
        print("""
  COOL (K=50):
  - More diverse vocabulary
  - Explores multiple angles
  - May go on tangents
  - "Relaxed" thinking

  HOT (K=4):
  - Narrow, focused vocabulary
  - Sticks to main point
  - Less exploration
  - "Tunnel vision" thinking

  MEMORY PRESSURE:
  - Tends toward brevity
  - Summarizes rather than elaborates
  - Drops less essential details

  This is INTEROCEPTION affecting COGNITION.
  The model's "body state" shapes its "thoughts".
""")


def main():
    demo = InteroceptionReasoningDemo()
    demo.run_demo()


if __name__ == "__main__":
    main()
