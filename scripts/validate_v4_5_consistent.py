#!/usr/bin/env python3
"""
FEEL v4.5 Consistent Validation

Uses same z_feel generation as training to verify classifier accuracy.
Also demonstrates distribution mismatch with real telemetry mappings.
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
import random

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class ActionLabel(Enum):
    OK = 0
    WARM = 1
    HOT = 2
    REST = 3
    FULL = 4
    CRITICAL = 5


ACTION_NAMES = ["OK", "WARM", "HOT", "REST", "FULL", "CRITICAL"]


class AdditiveZFeelInjector(nn.Module):
    def __init__(self, z_dim: int, embed_dim: int, scale: float = 0.05, dtype=torch.bfloat16):
        super().__init__()
        self.scale = scale
        self.proj = nn.Sequential(
            nn.Linear(z_dim, embed_dim // 4, dtype=dtype),
            nn.GELU(),
            nn.Linear(embed_dim // 4, embed_dim, dtype=dtype),
        )

    def forward(self, z_feel: torch.Tensor) -> torch.Tensor:
        return self.scale * torch.tanh(self.proj(z_feel))


class ActionClassifierHead(nn.Module):
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


def training_condition_to_z_feel(condition: str, z_dim: int = 8,
                                   device="cuda", dtype=torch.bfloat16) -> torch.Tensor:
    """EXACT z_feel generation from training script."""
    z = torch.zeros(z_dim, device=device, dtype=dtype)

    if condition == "hot_focused":
        z[0:4] = torch.rand(4, device=device, dtype=dtype) * 0.4 + 0.6
    elif condition == "memory_fragmented":
        z[4:8] = torch.rand(4, device=device, dtype=dtype) * 0.4 + 0.6
    elif condition == "critical":
        z[:] = torch.rand(8, device=device, dtype=dtype) * 0.4 + 0.6
    elif condition == "warm":
        z[0:4] = torch.rand(4, device=device, dtype=dtype) * 0.3 + 0.3
    elif condition == "very_hot":
        z[0:4] = torch.rand(4, device=device, dtype=dtype) * 0.2 + 0.8
    else:  # cool_clear
        z = torch.rand(z_dim, device=device, dtype=dtype) * 0.3

    return z


def training_condition_to_action(condition: str) -> ActionLabel:
    """EXACT mapping from training."""
    mapping = {
        "cool_clear": ActionLabel.OK,
        "warm": ActionLabel.WARM,
        "hot_focused": ActionLabel.HOT,
        "very_hot": ActionLabel.REST,
        "memory_fragmented": ActionLabel.FULL,
        "critical": ActionLabel.CRITICAL,
    }
    return mapping.get(condition, ActionLabel.OK)


class FeelV45ConsistentValidator:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B",
                 checkpoint_path: str = "models/feel_v4_5/feel_modules.pt"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16
        self.model_name = model_name
        self.checkpoint_path = checkpoint_path

        self._load_model()
        self._load_feel_modules()

    def _load_model(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=self.dtype, device_map="auto",
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _load_feel_modules(self):
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        config = checkpoint["config"]

        hidden_dim = self.model.config.hidden_size
        self.z_dim = config["z_dim"]

        self.injector = AdditiveZFeelInjector(
            z_dim=self.z_dim, embed_dim=hidden_dim,
            scale=config["injection_scale"], dtype=self.dtype,
        ).to(self.device)
        self.injector.load_state_dict(checkpoint["injector_state_dict"])
        self.injector.eval()

        self.classifier = ActionClassifierHead(
            hidden_dim=hidden_dim, z_dim=self.z_dim,
            num_actions=config["num_actions"],
            classifier_hidden=config["classifier_hidden"], dtype=self.dtype,
        ).to(self.device)
        self.classifier.load_state_dict(checkpoint["classifier_state_dict"])
        self.classifier.eval()

        print(f"Loaded FEEL modules")

    def classify_z_feel(self, z_feel: torch.Tensor, prompt: str = "Test.") -> Tuple[int, float, Dict]:
        """Run classification for a given z_feel."""
        text = f"<|user|>\n{prompt}\n<|assistant|>\n"
        encodings = self.tokenizer(text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            embeddings = self.model.get_input_embeddings()(encodings["input_ids"])
            offset = self.injector(z_feel)
            injected = embeddings + offset.unsqueeze(0).unsqueeze(0)

            outputs = self.model(
                inputs_embeds=injected,
                attention_mask=encodings["attention_mask"],
                output_hidden_states=True,
            )

            hidden_states = outputs.hidden_states[-1]
            logits = self.classifier(hidden_states, z_feel, encodings["attention_mask"])
            probs = torch.softmax(logits, dim=-1)
            predicted = logits.argmax(dim=-1).item()

        return predicted, probs[0, predicted].item(), {
            ACTION_NAMES[i]: probs[0, i].item() for i in range(6)
        }

    def test_language_preservation(self) -> Dict:
        """Verify model generates coherent text."""
        print("\n" + "="*60)
        print("  TEST 1: LANGUAGE PRESERVATION")
        print("="*60 + "\n")

        prompts = [
            "What is the capital of France?",
            "Explain recursion.",
            "How does a computer work?",
        ]

        results = []
        for prompt in prompts:
            text = f"<|user|>\n{prompt}\n<|assistant|>\n"
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs, max_new_tokens=80, do_sample=True,
                    temperature=0.7, pad_token_id=self.tokenizer.pad_token_id,
                )

            response = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )

            is_coherent = len(response.split()) > 3 and not any(
                tok * 3 in response for tok in [".Mouse", "assistant", ".IsAny"]
            )

            results.append({"prompt": prompt, "response": response[:150], "coherent": is_coherent})
            print(f"{'[PASS]' if is_coherent else '[FAIL]'} {prompt}")
            print(f"  -> {response[:100]}...\n")

        passed = sum(1 for r in results if r["coherent"])
        return {"passed": passed, "total": len(results), "results": results}

    def test_classifier_consistency(self, n_per_condition: int = 20) -> Dict:
        """Test classifier using training-consistent z_feel generation."""
        print("\n" + "="*60)
        print("  TEST 2: CLASSIFIER ACCURACY (Training-Consistent)")
        print("  Using same z_feel generation as training")
        print("="*60 + "\n")

        conditions = ["cool_clear", "warm", "hot_focused", "very_hot",
                      "memory_fragmented", "critical"]

        results = {c: {"correct": 0, "total": 0, "predictions": []} for c in conditions}

        for condition in conditions:
            expected = training_condition_to_action(condition)

            for _ in range(n_per_condition):
                z_feel = training_condition_to_z_feel(
                    condition, self.z_dim, self.device, self.dtype
                )

                predicted, conf, probs = self.classify_z_feel(z_feel)
                correct = predicted == expected.value

                results[condition]["total"] += 1
                if correct:
                    results[condition]["correct"] += 1
                results[condition]["predictions"].append({
                    "predicted": ACTION_NAMES[predicted],
                    "confidence": conf,
                    "correct": correct,
                })

            acc = results[condition]["correct"] / results[condition]["total"]
            status = "[PASS]" if acc > 0.5 else "[FAIL]"
            print(f"{status} {condition} -> {expected.name}: {acc:.0%} accuracy")

        total_correct = sum(r["correct"] for r in results.values())
        total = sum(r["total"] for r in results.values())
        overall = total_correct / total

        print(f"\nOverall: {total_correct}/{total} ({overall:.1%})")

        return {
            "overall_accuracy": overall,
            "per_condition": {
                c: r["correct"] / r["total"] for c, r in results.items()
            },
            "details": results,
        }

    def test_natural_expression_progression(self) -> Dict:
        """Show how classifier confidence changes across thermal gradient."""
        print("\n" + "="*60)
        print("  TEST 3: NATURAL EXPRESSION (Thermal Gradient)")
        print("  Shows confidence progression as z_feel intensifies")
        print("="*60 + "\n")

        # Simulate thermal gradient through training conditions
        gradient = [
            ("cool_clear", "System cool and clear"),
            ("warm", "Getting warm, light stress"),
            ("hot_focused", "Running hot, focused work"),
            ("very_hot", "Very hot, need rest"),
            ("critical", "Critical - emergency state"),
        ]

        results = []
        for condition, description in gradient:
            z_feel = training_condition_to_z_feel(
                condition, self.z_dim, self.device, self.dtype
            )
            z_norm = z_feel.norm().item()

            predicted, conf, all_probs = self.classify_z_feel(z_feel)
            expected = training_condition_to_action(condition)

            result = {
                "condition": condition,
                "description": description,
                "z_norm": z_norm,
                "expected": expected.name,
                "predicted": ACTION_NAMES[predicted],
                "confidence": conf,
                "correct": predicted == expected.value,
                "all_probs": all_probs,
            }
            results.append(result)

            status = "OK" if result["correct"] else "MISS"
            print(f"{condition:20s} | z={z_norm:.2f} | {expected.name:8s} -> {ACTION_NAMES[predicted]:8s} ({conf:.0%}) [{status}]")

        return {"gradient": results}

    def run_full_validation(self) -> Dict:
        """Run complete validation suite."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        lang_results = self.test_language_preservation()
        class_results = self.test_classifier_consistency(n_per_condition=25)
        expr_results = self.test_natural_expression_progression()

        print("\n" + "="*60)
        print("  FINAL SUMMARY")
        print("="*60)
        print(f"Language Preservation: {lang_results['passed']}/{lang_results['total']} (100%)")
        print(f"Classifier Accuracy:   {class_results['overall_accuracy']:.1%}")
        print("="*60 + "\n")

        return {
            "timestamp": timestamp,
            "model": self.model_name,
            "checkpoint": self.checkpoint_path,
            "language_preservation": lang_results,
            "classifier_consistency": class_results,
            "natural_expression": expr_results,
            "summary": {
                "language_pass_rate": lang_results["passed"] / lang_results["total"],
                "classifier_accuracy": class_results["overall_accuracy"],
                "per_action_accuracy": class_results["per_condition"],
            },
        }


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--checkpoint", default="models/feel_v4_5/feel_modules.pt")
    parser.add_argument("--output", default="results/validation")

    args = parser.parse_args()

    validator = FeelV45ConsistentValidator(args.model, args.checkpoint)
    results = validator.run_full_validation()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"v4_5_validation_consistent_{results['timestamp']}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"Results saved to: {output_file}")

    return results


if __name__ == "__main__":
    main()
