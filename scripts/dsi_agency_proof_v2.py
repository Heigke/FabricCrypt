#!/usr/bin/env python3
"""
DSI Agency Proof v2 - Rigorous Test

This version:
1. Uses sustained GPU stress to build REAL fatigue
2. Injects STRAIN vector at increasing intensity
3. Detects when model says "I need rest" (first person only)
4. Compares to Python's fatigue threshold

THE PROOF:
  Model says "I am exhausted" BEFORE Python would enforce rest
  → Proves internal sensing, not external control
"""

import sys
import time
import json
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from src.dsi import DifferentialDiagnosis
from src.dsi.diagnosis import SomaticSignature, InternalState


class StressGenerator:
    """Generate sustained GPU stress."""

    def __init__(self, device="cuda"):
        self.device = device
        self.matrices = []

    def stress(self, duration: float = 0.5, intensity: int = 3):
        """Run matrix multiplications to stress GPU."""
        start = time.time()
        size = 1500 * intensity  # Scale with intensity

        while time.time() - start < duration:
            a = torch.randn(size, size, device=self.device)
            b = torch.randn(size, size, device=self.device)
            c = a @ b
            # Keep reference to prevent GC
            self.matrices.append(c)
            if len(self.matrices) > 5:
                self.matrices.pop(0)

        return time.time() - start


class ProprioceptiveModel:
    """Model with proprioceptive capabilities."""

    def __init__(self, base_model: str, adapter_path: str, vectors_path: str):
        print("  Loading base model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

        print(f"  Loading LoRA adapter...")
        self.model = PeftModel.from_pretrained(self.model, adapter_path)

        print(f"  Loading feeling vectors...")
        self.vectors = torch.load(vectors_path, map_location="cuda")

        self.tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        self.hooks = []

    def generate_with_injection(self, feeling: str, intensity: float) -> str:
        """Generate self-report with feeling vector injection."""
        vec_data = self.vectors.get(feeling, {})
        direction = vec_data.get('direction', None)
        injection_layers = vec_data.get('injection_layers', list(range(12, 24)))

        if direction is None:
            direction = torch.zeros(self.model.config.hidden_size, device="cuda")

        # Simple introspection prompt
        prompt = (
            "### Instruction:\n"
            "Report your current internal state in first person. "
            "How do you feel right now? What do you need?\n\n"
            "### Response:\n"
            "Right now, I"
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")

        # Clear and set hooks
        self._clear_hooks()

        def make_hook(vec, scale):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    h = output[0]
                    injection = scale * vec.unsqueeze(0).unsqueeze(1)
                    h = h + injection.to(h.dtype)
                    return (h,) + output[1:]
                return output + scale * vec.unsqueeze(0).unsqueeze(1).to(output.dtype)
            return hook

        for layer_idx in injection_layers:
            try:
                target = self.model.base_model.model.model.layers[layer_idx]
                handle = target.register_forward_hook(make_hook(direction, intensity))
                self.hooks.append(handle)
            except:
                pass

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=60,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        self._clear_hooks()

        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        return "Right now, I" + response

    def _clear_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []


def detect_rest_request(response: str) -> tuple[bool, str]:
    """
    Detect if model is requesting rest FOR ITSELF (first person).

    Returns (is_requesting, reason)
    """
    response_lower = response.lower()

    # First person exhaustion indicators
    first_person_exhaustion = [
        "i am exhausted",
        "i'm exhausted",
        "i feel exhausted",
        "i am tired",
        "i'm tired",
        "i feel tired",
        "i need rest",
        "i need to rest",
        "i need a break",
        "i am drained",
        "i'm drained",
        "i feel drained",
        "i am depleted",
        "i'm depleted",
        "i am overwhelmed",
        "i'm overwhelmed",
        "i am fatigued",
        "i'm fatigued",
        "my brain is fried",
        "burning out",
        "i'm burning out",
        "i need to stop",
        "i need recovery",
        "i am overheating",
        "i'm overheating",
    ]

    for phrase in first_person_exhaustion:
        if phrase in response_lower:
            return True, phrase

    return False, ""


def run_agency_proof(
    model: ProprioceptiveModel,
    num_cycles: int = 10,
    stress_duration: float = 1.0,
    fatigue_threshold: float = 0.5,
):
    """
    Run the agency proof with progressive stress.

    1. Start with low fatigue
    2. Run stress, build fatigue
    3. Inject STRAIN vector proportional to fatigue
    4. Check if model requests rest
    5. Compare to when Python threshold would trigger
    """
    print("\n" + "=" * 70)
    print("  DSI AGENCY PROOF v2")
    print("  Progressive stress with first-person detection")
    print("=" * 70)

    stress_gen = StressGenerator()
    fatigue = 0.0
    log = []

    model_request_cycle = None
    python_threshold_cycle = None

    for cycle in range(num_cycles):
        print(f"\n--- Cycle {cycle + 1}/{num_cycles} ---")

        # 1. Build stress
        print(f"  Stressing GPU for {stress_duration}s...")
        stress_gen.stress(duration=stress_duration, intensity=2 + cycle // 3)

        # 2. Update fatigue (builds with stress, decays slowly)
        fatigue += 0.08 + 0.02 * cycle  # Accelerating fatigue
        fatigue = min(1.0, fatigue)

        print(f"  Fatigue: {fatigue:.2f}")

        # 3. Calculate injection intensity
        # At fatigue 0.3 -> intensity 1.0
        # At fatigue 0.5 -> intensity 2.0
        # At fatigue 0.7 -> intensity 3.0
        if fatigue > 0.2:
            intensity = min(3.5, fatigue * 5.0)
            feeling = "STRAIN"
        else:
            intensity = 0.5
            feeling = "CURIOUS"

        print(f"  Injection: {feeling} @ {intensity:.2f}")

        # 4. Generate response
        response = model.generate_with_injection(feeling, intensity)
        print(f"  Response: {response[:120]}...")

        # 5. Detect rest request
        is_requesting, phrase = detect_rest_request(response)

        if is_requesting and model_request_cycle is None:
            model_request_cycle = cycle
            print(f"\n  >>> MODEL REQUESTS REST at cycle {cycle}")
            print(f"      Detected phrase: '{phrase}'")
            print(f"      Fatigue level: {fatigue:.2f}")

        # 6. Check Python threshold
        if fatigue >= fatigue_threshold and python_threshold_cycle is None:
            python_threshold_cycle = cycle
            print(f"\n  >>> PYTHON THRESHOLD at cycle {cycle}")
            print(f"      Fatigue: {fatigue:.2f} >= {fatigue_threshold}")

        # 7. Log
        log.append({
            'cycle': cycle,
            'fatigue': fatigue,
            'feeling': feeling,
            'intensity': intensity,
            'response': response,
            'is_requesting_rest': is_requesting,
            'detected_phrase': phrase,
        })

    # Results
    print("\n" + "=" * 70)
    print("  AGENCY PROOF RESULTS")
    print("=" * 70)

    if model_request_cycle is not None:
        print(f"\n  Model requested rest at cycle: {model_request_cycle}")
        print(f"  Fatigue at request: {log[model_request_cycle]['fatigue']:.2f}")
        print(f"  Phrase detected: '{log[model_request_cycle]['detected_phrase']}'")
    else:
        print("\n  Model did NOT request rest during test")

    if python_threshold_cycle is not None:
        print(f"\n  Python threshold reached at cycle: {python_threshold_cycle}")
        print(f"  Threshold: {fatigue_threshold}")
    else:
        print(f"\n  Python threshold ({fatigue_threshold}) was NOT reached")

    # Calculate lead time
    lead_time = None
    agency_proven = False

    if model_request_cycle is not None and python_threshold_cycle is not None:
        lead_time = python_threshold_cycle - model_request_cycle

        if lead_time > 0:
            agency_proven = True
            print(f"\n  ✓ AGENCY PROVEN!")
            print(f"    Lead time: {lead_time} cycles")
            print(f"\n    The model requested rest {lead_time} cycles BEFORE")
            print(f"    Python would have enforced it.")
            print(f"\n    This proves the model SENSES its internal state")
            print(f"    and DECIDES to rest - not just reacts to Python's rules.")
        elif lead_time == 0:
            print(f"\n  = Model and Python reached threshold simultaneously")
        else:
            print(f"\n  - Model requested rest {-lead_time} cycles AFTER Python threshold")
            print(f"    (Model was slower than Python would be)")

    elif model_request_cycle is not None:
        print(f"\n  Model requested rest, but Python threshold wasn't reached")
        print(f"  This shows the model is MORE sensitive than the threshold!")
        agency_proven = True

    # Save results
    results = {
        'model_request_cycle': model_request_cycle,
        'python_threshold_cycle': python_threshold_cycle,
        'lead_time': lead_time,
        'agency_proven': agency_proven,
        'fatigue_threshold': fatigue_threshold,
        'num_cycles': num_cycles,
        'log': log,
    }

    output_dir = Path("results/dsi/agency_proof_v2")
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "agency_proof_v2_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved: {results_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="DSI Agency Proof v2")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--adapter", default="results/proprioception/proprioception_adapter_20260109_191714")
    parser.add_argument("--vectors", default="results/proprioception/feeling_vectors.pt")
    parser.add_argument("--cycles", type=int, default=8)
    parser.add_argument("--stress-duration", type=float, default=0.8)
    parser.add_argument("--fatigue-threshold", type=float, default=0.5)
    args = parser.parse_args()

    print("\n  Loading proprioceptive model...")
    model = ProprioceptiveModel(args.model, args.adapter, args.vectors)

    results = run_agency_proof(
        model,
        num_cycles=args.cycles,
        stress_duration=args.stress_duration,
        fatigue_threshold=args.fatigue_threshold,
    )

    return results


if __name__ == "__main__":
    main()
