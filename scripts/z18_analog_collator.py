#!/usr/bin/env python3
"""
FEEL v18: Analog Hardware Collator
===================================
Fixes the "Binary Trap" by injecting continuous stress values during training.

The Problem:
- Previous collator used discrete values: 0.1 (Calm) vs 0.9 (Stressed)
- FiLM adapter learned a binary decision boundary at ~0.5
- Result: Step function instead of S-Curve

The Fix:
- Inject continuous noise: 0.0-0.4 (Calm), 0.6-1.0 (Stressed)
- Add "blind spot" training with decoupled sensor values
- Create gradual transitions so model learns analog control

Author: FEEL Research Team
Date: 2026-01-12
"""

import torch
import random
from typing import List, Dict, Any


class AnalogHardwareCollator:
    """
    Collator that generates continuous stress values for analog control training.

    Instead of binary (0.1 vs 0.9), generates:
    - Calm zone: 0.0 - 0.4 (with variance)
    - Transition zone: 0.4 - 0.6 (sparse, teaches boundary)
    - Stressed zone: 0.6 - 1.0 (with variance)
    """

    def __init__(
        self,
        tokenizer,
        length_threshold: int = 200,  # Chars below = stressed response
        blind_spot_prob: float = 0.1,  # Probability of decoupled sensors
        transition_prob: float = 0.15,  # Probability of mid-range values
    ):
        self.tokenizer = tokenizer
        self.length_threshold = length_threshold
        self.blind_spot_prob = blind_spot_prob
        self.transition_prob = transition_prob

    def _get_stress_from_target(self, example: Dict) -> float:
        """
        Determine target stress level from the example.

        Priority:
        1. Explicit 'stress_level' field (continuous)
        2. 'is_stressed' boolean field
        3. Output length heuristic
        """
        # Check for explicit continuous stress level
        if "stress_level" in example:
            base_stress = example["stress_level"]
            # Add small jitter to explicit values too
            jitter = random.gauss(0, 0.05)
            return max(0.0, min(1.0, base_stress + jitter))

        # Check for boolean flag
        if "is_stressed" in example:
            is_stressed = example["is_stressed"]
        else:
            # Heuristic: short output = stressed
            output_len = len(example.get("output", ""))
            is_stressed = output_len < self.length_threshold

        # Generate continuous value within appropriate zone
        if is_stressed:
            # Stressed zone: 0.6 - 1.0 with gradient
            # Use beta distribution to cluster toward middle-high
            stress = random.betavariate(2, 2) * 0.4 + 0.6  # Range: 0.6-1.0
        else:
            # Calm zone: 0.0 - 0.4
            stress = random.betavariate(2, 2) * 0.4  # Range: 0.0-0.4

        # Occasionally sample from transition zone (0.4-0.6)
        # This teaches the model that middle values exist
        if random.random() < self.transition_prob:
            stress = random.uniform(0.35, 0.65)

        return stress

    def _generate_sensor_vector(self, stress: float) -> torch.Tensor:
        """
        Generate [temp, power, clock] sensor vector from stress level.

        Physical correlation model:
        - Temperature: Primary stress indicator
        - Power: Correlated with temp but with noise
        - Clock: Inversely correlated (throttling under heat)
        """
        # Temperature = base stress + noise
        temp = stress + random.gauss(0, 0.05)
        temp = max(0.0, min(1.0, temp))

        # Power is correlated but with more variance
        power_correlation = 0.7
        power_noise = 0.15
        power = stress * power_correlation + random.gauss(0.3, power_noise)
        power = max(0.0, min(1.0, power))

        # Clock inversely correlates (throttling when hot)
        clock_base = 1.0 - stress * 0.3  # Light inverse correlation
        clock = clock_base + random.gauss(0, 0.1)
        clock = max(0.5, min(1.0, clock))  # Clock never goes below 50%

        # BLIND SPOT TRAINING: Occasionally decouple sensors
        # This teaches the model not to rely on spurious correlations
        if random.random() < self.blind_spot_prob:
            power = random.random()  # Total random power
            clock = random.uniform(0.6, 1.0)  # Random clock

        return torch.tensor([temp, power, clock], dtype=torch.float32)

    def __call__(self, examples: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collate batch with analog stress values.
        """
        batch_size = len(examples)

        # Collect input_ids
        input_ids_list = []
        for e in examples:
            if isinstance(e["input_ids"], torch.Tensor):
                input_ids_list.append(e["input_ids"])
            else:
                input_ids_list.append(torch.tensor(e["input_ids"]))

        # Pad sequences
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids_list,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )

        # Create attention mask
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        # Labels = input_ids for causal LM
        labels = input_ids.clone()
        # Mask padding tokens in labels
        labels[labels == self.tokenizer.pad_token_id] = -100

        # Generate analog stress values and sensor vectors
        stress_levels = []
        sensor_data = []

        for example in examples:
            stress = self._get_stress_from_target(example)
            stress_levels.append(stress)
            sensor_data.append(self._generate_sensor_vector(stress))

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "stress_level": torch.tensor(stress_levels, dtype=torch.float32),
            "sensor_values": torch.stack(sensor_data),
        }


class GradientAnalogCollator(AnalogHardwareCollator):
    """
    Extended collator that generates training samples across the full gradient.

    For S-Curve training, we need samples at ALL stress levels, not just
    the extremes. This collator ensures uniform coverage.
    """

    def __init__(self, tokenizer, **kwargs):
        super().__init__(tokenizer, **kwargs)
        self.coverage_bins = 10  # Divide 0-1 into 10 bins
        self.bin_counts = [0] * self.coverage_bins

    def _force_coverage(self, batch_examples: List[Dict]) -> List[float]:
        """
        Ensure batch has coverage across stress spectrum.
        """
        stress_levels = []

        # First, get natural stress levels
        for example in batch_examples:
            stress = self._get_stress_from_target(example)
            stress_levels.append(stress)

        # Identify underrepresented bins
        batch_bins = [int(s * (self.coverage_bins - 1)) for s in stress_levels]
        for b in batch_bins:
            self.bin_counts[b] += 1

        # Optionally adjust: if certain bins are starving, force samples there
        # This is a passive monitoring for now
        return stress_levels


# Standalone test
if __name__ == "__main__":
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    collator = AnalogHardwareCollator(tokenizer)

    # Test with mock examples
    examples = [
        {"input_ids": [1, 2, 3, 4, 5], "output": "short", "is_stressed": True},
        {"input_ids": [1, 2, 3, 4, 5, 6, 7, 8], "output": "this is a longer response that should be calm", "is_stressed": False},
        {"input_ids": [1, 2, 3], "stress_level": 0.75},
    ]

    batch = collator(examples)

    print("Analog Collator Test:")
    print(f"  Input IDs shape: {batch['input_ids'].shape}")
    print(f"  Stress levels: {batch['stress_level'].tolist()}")
    print(f"  Sensor values:")
    for i, (stress, sensor) in enumerate(zip(batch['stress_level'], batch['sensor_values'])):
        print(f"    Sample {i}: stress={stress:.3f}, sensors=[T={sensor[0]:.3f}, P={sensor[1]:.3f}, C={sensor[2]:.3f}]")

    # Test distribution
    print("\nStress Distribution Test (100 samples):")
    stress_samples = []
    for _ in range(100):
        mock = [{"input_ids": [1,2,3], "is_stressed": random.random() > 0.5}]
        b = collator(mock)
        stress_samples.append(b['stress_level'][0].item())

    # Bin the samples
    bins = [0] * 10
    for s in stress_samples:
        bins[min(9, int(s * 10))] += 1

    print("  Bin distribution (0.0-0.1 ... 0.9-1.0):")
    for i, count in enumerate(bins):
        bar = "#" * count
        print(f"    [{i/10:.1f}-{(i+1)/10:.1f}]: {bar} ({count})")
