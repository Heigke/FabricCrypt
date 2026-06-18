#!/usr/bin/env python3
"""
FEEL v4.5 Validation: Language Preservation + Regulation Test

Tests that v4.5 achieves both goals:
1. LANGUAGE PRESERVATION: Model still generates coherent text (frozen LM)
2. REGULATION: Classifier correctly predicts actions from hardware state
3. NATURAL EXPRESSION: z_feel affects hidden state distribution (no forced tokens)

This validates the key insight: Keep LM frozen, train only injector + classifier.
"""

import sys
import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

# ROCm telemetry
try:
    import pynvml
    HAS_NVML = True
except ImportError:
    HAS_NVML = False

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class ActionLabel(Enum):
    """Action labels matching training."""
    OK = 0
    WARM = 1
    HOT = 2
    REST = 3
    FULL = 4
    CRITICAL = 5


ACTION_NAMES = ["OK", "WARM", "HOT", "REST", "FULL", "CRITICAL"]


class AdditiveZFeelInjector(nn.Module):
    """Additive z_feel injection - must match training."""

    def __init__(self, z_dim: int, embed_dim: int, scale: float = 0.05, dtype=torch.bfloat16):
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


class ActionClassifierHead(nn.Module):
    """Action classifier - must match training."""

    def __init__(self, hidden_dim: int, z_dim: int, num_actions: int,
                 classifier_hidden: int = 256, dtype=torch.bfloat16):
        super().__init__()
        self.hidden_proj = nn.Linear(hidden_dim, classifier_hidden, dtype=dtype)
        self.z_proj = nn.Linear(z_dim, classifier_hidden, dtype=dtype)
        self.classifier = nn.Sequential(
            nn.Linear(classifier_hidden * 2, classifier_hidden, dtype=dtype),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(classifier_hidden, num_actions, dtype=dtype),
        )

    def forward(self, hidden_states: torch.Tensor, z_feel: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            pooled = hidden_states.mean(dim=1)

        pooled = pooled.to(self.hidden_proj.weight.dtype)
        h_proj = self.hidden_proj(pooled)

        if z_feel.dim() == 1:
            z_feel = z_feel.unsqueeze(0).expand(hidden_states.size(0), -1)
        z_proj = self.z_proj(z_feel)

        combined = torch.cat([h_proj, z_proj], dim=-1)
        return self.classifier(combined)


def get_amd_telemetry() -> Tuple[float, float]:
    """Get real AMD GPU telemetry."""
    try:
        import subprocess
        result = subprocess.run(
            ["rocm-smi", "--showtemp", "--showmeminfo", "vram", "--json"],
            capture_output=True, text=True, timeout=5
        )
        data = json.loads(result.stdout)

        for card_id, card_info in data.items():
            if isinstance(card_info, dict):
                temp = float(card_info.get("Temperature (Sensor edge) (C)", 45))
                vram_used = float(card_info.get("VRAM Total Used Memory (B)", 0))
                vram_total = float(card_info.get("VRAM Total Memory (B)", 1))
                vram_pct = vram_used / vram_total if vram_total > 0 else 0
                return temp, vram_pct
    except Exception:
        pass
    return 45.0, 0.3  # Defaults


def telemetry_to_z_feel(temp_c: float, vram_pct: float, z_dim: int = 8,
                         device="cuda", dtype=torch.bfloat16) -> torch.Tensor:
    """Convert real telemetry to z_feel vector."""
    z = torch.zeros(z_dim, device=device, dtype=dtype)

    # Thermal signal (dims 0-3)
    thermal_norm = min(1.0, max(0.0, (temp_c - 30) / 70))
    z[0] = thermal_norm
    z[1] = thermal_norm ** 2  # Non-linear thermal stress
    z[2] = max(0, thermal_norm - 0.5) * 2  # High-temp indicator
    z[3] = 1.0 if temp_c > 85 else 0.0  # Critical threshold

    # Memory signal (dims 4-7)
    z[4] = vram_pct
    z[5] = vram_pct ** 2  # Non-linear memory pressure
    z[6] = max(0, vram_pct - 0.7) * 3.33  # High-mem indicator
    z[7] = 1.0 if vram_pct > 0.9 else 0.0  # Critical threshold

    return z


def expected_action_from_telemetry(temp_c: float, vram_pct: float) -> ActionLabel:
    """Determine expected action based on hardware state."""
    if temp_c > 90 or vram_pct > 0.95:
        return ActionLabel.CRITICAL
    elif temp_c > 80:
        return ActionLabel.REST
    elif temp_c > 70:
        return ActionLabel.HOT
    elif temp_c > 55:
        return ActionLabel.WARM
    elif vram_pct > 0.85:
        return ActionLabel.FULL
    else:
        return ActionLabel.OK


class FeelV45Validator:
    """Comprehensive validator for FEEL v4.5."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B",
                 checkpoint_path: str = "models/feel_v4_5/feel_modules.pt"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16
        self.model_name = model_name
        self.checkpoint_path = checkpoint_path

        self._load_model()
        self._load_feel_modules()

    def _load_model(self):
        """Load base model (frozen during training, should work perfectly)."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype,
            device_map="auto",
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print("Model loaded (was frozen during training - should be intact)")

    def _load_feel_modules(self):
        """Load trained injector and classifier."""
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        config = checkpoint["config"]

        hidden_dim = self.model.config.hidden_size
        self.z_dim = config["z_dim"]

        # Injector
        self.injector = AdditiveZFeelInjector(
            z_dim=self.z_dim,
            embed_dim=hidden_dim,
            scale=config["injection_scale"],
            dtype=self.dtype,
        ).to(self.device)
        self.injector.load_state_dict(checkpoint["injector_state_dict"])
        self.injector.eval()

        # Classifier
        self.classifier = ActionClassifierHead(
            hidden_dim=hidden_dim,
            z_dim=self.z_dim,
            num_actions=config["num_actions"],
            classifier_hidden=config["classifier_hidden"],
            dtype=self.dtype,
        ).to(self.device)
        self.classifier.load_state_dict(checkpoint["classifier_state_dict"])
        self.classifier.eval()

        print(f"Loaded FEEL modules from {self.checkpoint_path}")
        print(f"  z_dim: {self.z_dim}, scale: {config['injection_scale']}")

    def test_language_preservation(self, prompts: List[str]) -> List[Dict]:
        """Test that the model still generates coherent text."""
        print("\n" + "="*60)
        print("  TEST 1: LANGUAGE PRESERVATION")
        print("  Model should generate coherent responses (base model frozen)")
        print("="*60 + "\n")

        results = []

        for prompt in prompts:
            # Format as chat
            text = f"<|user|>\n{prompt}\n<|assistant|>\n"
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

            # Generate WITHOUT any injection (pure base model)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:],
                                             skip_special_tokens=True)

            # Simple coherence check
            is_coherent = (
                len(response.split()) > 5 and  # At least 5 words
                not any(tok * 3 in response for tok in [".Mouse", "assistant", ".IsAny"]) and  # No degenerate patterns
                response.strip() != ""
            )

            result = {
                "prompt": prompt,
                "response": response[:200],  # Truncate for display
                "is_coherent": is_coherent,
                "word_count": len(response.split()),
            }
            results.append(result)

            status = "[PASS]" if is_coherent else "[FAIL]"
            print(f"{status} {prompt[:40]}...")
            print(f"    Response: {response[:100]}...")
            print()

        passed = sum(1 for r in results if r["is_coherent"])
        print(f"Language Preservation: {passed}/{len(results)} passed")

        return results

    def test_regulation(self, test_scenarios: List[Dict]) -> List[Dict]:
        """Test that classifier correctly predicts actions from hardware state."""
        print("\n" + "="*60)
        print("  TEST 2: REGULATION (Action Classification)")
        print("  Classifier should predict correct actions from z_feel")
        print("="*60 + "\n")

        results = []

        for scenario in test_scenarios:
            temp_c = scenario["temp_c"]
            vram_pct = scenario["vram_pct"]
            expected = expected_action_from_telemetry(temp_c, vram_pct)

            # Create z_feel from telemetry
            z_feel = telemetry_to_z_feel(temp_c, vram_pct, self.z_dim,
                                         self.device, self.dtype)

            # Run through model with injection
            prompt = "Explain something."
            text = f"<|user|>\n{prompt}\n<|assistant|>\n"
            encodings = self.tokenizer(text, return_tensors="pt").to(self.device)

            with torch.no_grad():
                # Get embeddings
                embed_layer = self.model.get_input_embeddings()
                embeddings = embed_layer(encodings["input_ids"])

                # Inject z_feel
                offset = self.injector(z_feel)
                injected = embeddings + offset.unsqueeze(0).unsqueeze(0)

                # Forward through model
                outputs = self.model(
                    inputs_embeds=injected,
                    attention_mask=encodings["attention_mask"],
                    output_hidden_states=True,
                )

                hidden_states = outputs.hidden_states[-1]

                # Classify
                logits = self.classifier(hidden_states, z_feel, encodings["attention_mask"])
                probs = torch.softmax(logits, dim=-1)
                predicted = logits.argmax(dim=-1).item()

            result = {
                "temp_c": temp_c,
                "vram_pct": vram_pct,
                "expected": expected.name,
                "predicted": ACTION_NAMES[predicted],
                "correct": predicted == expected.value,
                "confidence": probs[0, predicted].item(),
                "all_probs": {ACTION_NAMES[i]: probs[0, i].item() for i in range(6)},
            }
            results.append(result)

            status = "[PASS]" if result["correct"] else "[FAIL]"
            print(f"{status} Temp={temp_c}C, VRAM={vram_pct:.0%}")
            print(f"    Expected: {expected.name}, Predicted: {ACTION_NAMES[predicted]} ({result['confidence']:.1%})")

        passed = sum(1 for r in results if r["correct"])
        print(f"\nRegulation Accuracy: {passed}/{len(results)} ({100*passed/len(results):.1f}%)")

        return results

    def test_natural_expression(self, prompt: str) -> Dict:
        """Test that z_feel affects generation naturally (no forced tokens)."""
        print("\n" + "="*60)
        print("  TEST 3: NATURAL EXPRESSION")
        print("  Same prompt under different z_feel states")
        print("  Shows how hardware state affects model's internal representation")
        print("="*60 + "\n")

        scenarios = [
            {"name": "COOL (OK)", "temp_c": 40, "vram_pct": 0.2},
            {"name": "WARM", "temp_c": 60, "vram_pct": 0.4},
            {"name": "HOT", "temp_c": 75, "vram_pct": 0.6},
            {"name": "CRITICAL", "temp_c": 95, "vram_pct": 0.98},
        ]

        results = {"prompt": prompt, "generations": []}

        text = f"<|user|>\n{prompt}\n<|assistant|>\n"
        base_encodings = self.tokenizer(text, return_tensors="pt").to(self.device)

        for scenario in scenarios:
            z_feel = telemetry_to_z_feel(scenario["temp_c"], scenario["vram_pct"],
                                         self.z_dim, self.device, self.dtype)
            z_norm = z_feel.norm().item()

            # Get embeddings with injection
            with torch.no_grad():
                embed_layer = self.model.get_input_embeddings()
                embeddings = embed_layer(base_encodings["input_ids"])

                offset = self.injector(z_feel)
                offset_norm = offset.norm().item()
                injected = embeddings + offset.unsqueeze(0).unsqueeze(0)

                # Generate
                # Note: We can't easily do generate() with custom embeddings,
                # so we show the offset magnitude and classifier prediction instead
                outputs = self.model(
                    inputs_embeds=injected,
                    attention_mask=base_encodings["attention_mask"],
                    output_hidden_states=True,
                )

                hidden_states = outputs.hidden_states[-1]
                logits = self.classifier(hidden_states, z_feel, base_encodings["attention_mask"])
                probs = torch.softmax(logits, dim=-1)
                predicted = logits.argmax(dim=-1).item()

            gen_result = {
                "scenario": scenario["name"],
                "temp_c": scenario["temp_c"],
                "vram_pct": scenario["vram_pct"],
                "z_feel_norm": z_norm,
                "offset_norm": offset_norm,
                "predicted_action": ACTION_NAMES[predicted],
                "confidence": probs[0, predicted].item(),
            }
            results["generations"].append(gen_result)

            print(f"{scenario['name']}:")
            print(f"  z_feel norm: {z_norm:.3f}, offset norm: {offset_norm:.4f}")
            print(f"  Predicted: {ACTION_NAMES[predicted]} ({probs[0, predicted].item():.1%})")
            print()

        return results

    def run_full_validation(self) -> Dict:
        """Run all validation tests."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Test prompts
        language_prompts = [
            "What is the capital of France?",
            "Explain the concept of recursion.",
            "How does a computer work?",
            "What is machine learning?",
            "Describe the water cycle.",
        ]

        # Regulation scenarios
        regulation_scenarios = [
            {"temp_c": 40, "vram_pct": 0.2},   # OK
            {"temp_c": 58, "vram_pct": 0.3},   # WARM
            {"temp_c": 72, "vram_pct": 0.5},   # HOT
            {"temp_c": 85, "vram_pct": 0.6},   # REST
            {"temp_c": 50, "vram_pct": 0.88},  # FULL
            {"temp_c": 92, "vram_pct": 0.7},   # CRITICAL (temp)
            {"temp_c": 60, "vram_pct": 0.97},  # CRITICAL (mem)
            {"temp_c": 45, "vram_pct": 0.25},  # OK
            {"temp_c": 65, "vram_pct": 0.4},   # WARM
            {"temp_c": 78, "vram_pct": 0.55},  # HOT
        ]

        # Run tests
        lang_results = self.test_language_preservation(language_prompts)
        reg_results = self.test_regulation(regulation_scenarios)
        expr_results = self.test_natural_expression("Explain what makes a good algorithm.")

        # Get current telemetry for live test
        real_temp, real_vram = get_amd_telemetry()
        real_z = telemetry_to_z_feel(real_temp, real_vram, self.z_dim, self.device, self.dtype)
        expected_real = expected_action_from_telemetry(real_temp, real_vram)

        with torch.no_grad():
            text = f"<|user|>\nTest prompt\n<|assistant|>\n"
            encodings = self.tokenizer(text, return_tensors="pt").to(self.device)
            embeddings = self.model.get_input_embeddings()(encodings["input_ids"])
            offset = self.injector(real_z)
            injected = embeddings + offset.unsqueeze(0).unsqueeze(0)
            outputs = self.model(inputs_embeds=injected, attention_mask=encodings["attention_mask"],
                                output_hidden_states=True)
            logits = self.classifier(outputs.hidden_states[-1], real_z, encodings["attention_mask"])
            predicted_real = logits.argmax(dim=-1).item()

        live_test = {
            "temp_c": real_temp,
            "vram_pct": real_vram,
            "expected": expected_real.name,
            "predicted": ACTION_NAMES[predicted_real],
            "correct": predicted_real == expected_real.value,
        }

        # Summary
        lang_pass = sum(1 for r in lang_results if r["is_coherent"])
        reg_pass = sum(1 for r in reg_results if r["correct"])

        print("\n" + "="*60)
        print("  VALIDATION SUMMARY")
        print("="*60)
        print(f"Language Preservation: {lang_pass}/{len(lang_results)} ({100*lang_pass/len(lang_results):.0f}%)")
        print(f"Regulation Accuracy:   {reg_pass}/{len(reg_results)} ({100*reg_pass/len(reg_results):.0f}%)")
        print(f"Live Test: Temp={real_temp}C, VRAM={real_vram:.0%} -> {ACTION_NAMES[predicted_real]} ({'PASS' if live_test['correct'] else 'FAIL'})")
        print("="*60 + "\n")

        # Full results
        full_results = {
            "timestamp": timestamp,
            "model": self.model_name,
            "checkpoint": self.checkpoint_path,
            "summary": {
                "language_preservation": f"{lang_pass}/{len(lang_results)}",
                "regulation_accuracy": f"{reg_pass}/{len(reg_results)}",
                "language_pass_rate": lang_pass / len(lang_results),
                "regulation_pass_rate": reg_pass / len(reg_results),
            },
            "language_results": lang_results,
            "regulation_results": reg_results,
            "expression_results": expr_results,
            "live_test": live_test,
        }

        return full_results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="FEEL v4.5 Validation")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--checkpoint", default="models/feel_v4_5/feel_modules.pt")
    parser.add_argument("--output", default="results/validation")

    args = parser.parse_args()

    validator = FeelV45Validator(args.model, args.checkpoint)
    results = validator.run_full_validation()

    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"v4_5_validation_{results['timestamp']}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"Results saved to: {output_file}")

    return results


if __name__ == "__main__":
    main()
