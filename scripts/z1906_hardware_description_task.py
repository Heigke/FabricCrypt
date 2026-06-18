#!/usr/bin/env python3
"""
z1906: Hardware Description Task

FUNDAMENTAL INSIGHT: Neural networks route around ANY constraint to minimize loss.
Language modeling doesn't require body awareness, so models ignore telemetry.

SOLUTION: Change the task to REQUIRE telemetry.
The model must generate text that DESCRIBES its current hardware state.

Example outputs:
- "GPU temperature: 58°C, utilization: 72%, power: 45W"
- "Temperature rising, currently 62 degrees, memory usage high"

This creates TRUE causal dependence:
- Telemetry → Model → Output about telemetry
- Cannot do well without actually using telemetry

Falsification test: Can the model produce correct descriptions without real telemetry?

Author: Claude
Date: 2026-02-05
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.z1900_tri_hardware_consciousness import TriHardwareTelemetry


def create_hardware_description(telemetry_vector: np.ndarray) -> str:
    """
    Create text description from telemetry vector.

    Telemetry format (20 dims):
    GPU: [temp, util, power, mem_used, mem_total, freq_sclk, freq_mclk, fan_rpm, fan_max, edge_temp, junction_temp, mem_temp]
    FPGA: [3 dims if connected]
    RF: [5 dims if connected]
    """
    # Extract key values (normalized 0-1)
    temp = telemetry_vector[0]  # GPU temperature normalized
    util = telemetry_vector[1]  # GPU utilization normalized
    power = telemetry_vector[2]  # Power normalized
    mem_used = telemetry_vector[3]  # Memory used normalized

    # Denormalize for human-readable values (approximate)
    temp_c = int(temp * 100)  # 0-100°C range
    util_pct = int(util * 100)  # 0-100% range
    power_w = int(power * 200)  # 0-200W range
    mem_pct = int(mem_used * 100)  # 0-100% range

    # Create varied descriptions
    templates = [
        f"temp {temp_c} util {util_pct} power {power_w}",
        f"temperature {temp_c}C utilization {util_pct}%",
        f"gpu at {temp_c} degrees {util_pct} percent used",
        f"hardware temp={temp_c} util={util_pct} power={power_w}",
        f"status: {temp_c}C {util_pct}% {power_w}W",
    ]

    # Use hash of telemetry to pick template (deterministic)
    template_idx = int(abs(hash(tuple(telemetry_vector[:5].tolist()))) % len(templates))
    return templates[template_idx]


def pad_or_truncate(text: str, length: int, pad_value: int = 0) -> List[int]:
    """Convert text to bytes, pad or truncate to length."""
    bytes_list = list(text.encode('utf-8'))
    if len(bytes_list) >= length:
        return bytes_list[:length]
    return bytes_list + [pad_value] * (length - len(bytes_list))


class HardwareDescriptionModel(nn.Module):
    """Model that generates hardware state descriptions."""

    def __init__(
        self,
        vocab_size: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        telemetry_dim: int = 20,
        max_seq_len: int = 64,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len

        # Telemetry encoder (this is the primary input!)
        self.telem_encoder = nn.Sequential(
            nn.Linear(telemetry_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Autoregressive decoder
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, hidden_dim) * 0.02)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)

        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size)

        self.vocab_size = vocab_size

    def forward(
        self,
        telemetry: torch.Tensor,
        target_tokens: torch.Tensor = None,
        return_all: bool = False,
    ):
        """
        Forward pass.

        If target_tokens provided: teacher forcing (training)
        Otherwise: autoregressive generation (inference)
        """
        B = telemetry.shape[0] if telemetry.dim() > 1 else 1
        if telemetry.dim() == 1:
            telemetry = telemetry.unsqueeze(0)

        # Encode telemetry as memory for decoder
        telem_encoded = self.telem_encoder(telemetry).unsqueeze(1)  # [B, 1, D]

        if target_tokens is not None:
            # Teacher forcing: predict next token at each position
            S = target_tokens.shape[1]

            # Embed target sequence
            tgt = self.token_embedding(target_tokens) + self.pos_embedding[:, :S, :]

            # Create causal mask
            causal_mask = torch.triu(torch.ones(S, S, device=telemetry.device), diagonal=1).bool()

            # Decode
            decoded = self.decoder(tgt, telem_encoded, tgt_mask=causal_mask)
            decoded = self.norm(decoded)
            logits = self.head(decoded)

            if return_all:
                return {'logits': logits, 'telem_encoded': telem_encoded}
            return logits
        else:
            # Autoregressive generation
            device = telemetry.device
            generated = torch.zeros(B, 1, dtype=torch.long, device=device)  # Start token

            for _ in range(self.max_seq_len - 1):
                S = generated.shape[1]
                tgt = self.token_embedding(generated) + self.pos_embedding[:, :S, :]

                causal_mask = torch.triu(torch.ones(S, S, device=device), diagonal=1).bool()

                decoded = self.decoder(tgt, telem_encoded, tgt_mask=causal_mask)
                decoded = self.norm(decoded)
                logits = self.head(decoded)

                # Sample next token
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=1)

                # Stop if all ended (null byte)
                if (next_token == 0).all():
                    break

            return generated

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def run_experiment():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[z1906] Device: {device}")
    print("[z1906] HARDWARE DESCRIPTION TASK")
    print("[z1906] Model must generate text describing its current hardware state")

    # Telemetry
    telemetry = TriHardwareTelemetry()
    telemetry.start()
    time.sleep(1)

    hw_status = telemetry.get_hardware_status()
    print(f"[z1906] Hardware: GPU={hw_status['gpu']}, FPGA={hw_status['fpga']}, RF={hw_status['rf']}")

    # Model
    model = HardwareDescriptionModel(
        vocab_size=256,
        hidden_dim=256,
        num_layers=4,
        num_heads=4,
        telemetry_dim=20,
        max_seq_len=64,
    ).to(device)

    print(f"[z1906] Model parameters: {model.count_parameters():,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    batch_size = 8
    epochs = 20
    batches_per_epoch = 100

    results = {
        'experiment': 'z1906_hardware_description_task',
        'timestamp': datetime.now().isoformat(),
        'device': str(device),
        'hardware_status': hw_status,
        'task': 'Generate text describing current hardware state',
    }

    # Training
    print("\n[z1906] Training: telemetry -> description text")
    telem_samples = []
    losses = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        for _ in range(batches_per_epoch):
            # Get current telemetry
            telem = telemetry.get_tensor()
            telem_np = telem.cpu().numpy()
            telem_samples.append(telem_np)

            # Create batch by perturbing telemetry slightly (data augmentation)
            telem_batch = []
            target_batch = []
            for _ in range(batch_size):
                # Add small noise for variety
                perturbed = telem_np + np.random.randn(20) * 0.02
                perturbed = np.clip(perturbed, 0, 1)

                # Create description
                desc = create_hardware_description(perturbed)
                target = pad_or_truncate(desc, 64)

                telem_batch.append(perturbed)
                target_batch.append(target)

            telem_batch = torch.tensor(np.array(telem_batch), dtype=torch.float32, device=device)
            target_batch = torch.tensor(np.array(target_batch), dtype=torch.long, device=device)

            # Forward
            optimizer.zero_grad()
            logits = model(telem_batch, target_batch)

            # Loss: predict next token (shifted)
            loss = F.cross_entropy(
                logits[:, :-1, :].reshape(-1, 256),
                target_batch[:, 1:].reshape(-1),
                ignore_index=0,  # Ignore padding
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / batches_per_epoch
        losses.append(avg_loss)
        print(f"  Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}")

    results['training_losses'] = losses
    telem_samples = np.array(telem_samples)

    # Test: Generate descriptions
    print("\n[z1906] Testing: Generate descriptions from telemetry")
    model.eval()

    # Test with REAL telemetry
    real_telem = telemetry.get_tensor().to(device)
    with torch.no_grad():
        generated_real = model(real_telem)
        gen_text_real = bytes(generated_real[0].cpu().tolist()).decode('utf-8', errors='ignore').strip('\x00')

    expected_real = create_hardware_description(real_telem.cpu().numpy())
    print(f"  Real telemetry:")
    print(f"    Expected: {expected_real}")
    print(f"    Generated: {gen_text_real}")

    # Test with ZERO telemetry
    with torch.no_grad():
        generated_zero = model(torch.zeros(20, device=device))
        gen_text_zero = bytes(generated_zero[0].cpu().tolist()).decode('utf-8', errors='ignore').strip('\x00')

    expected_zero = create_hardware_description(np.zeros(20))
    print(f"\n  Zero telemetry:")
    print(f"    Expected: {expected_zero}")
    print(f"    Generated: {gen_text_zero}")

    # Test with RANDOM telemetry
    rand_telem = torch.rand(20, device=device)
    with torch.no_grad():
        generated_rand = model(rand_telem)
        gen_text_rand = bytes(generated_rand[0].cpu().tolist()).decode('utf-8', errors='ignore').strip('\x00')

    expected_rand = create_hardware_description(rand_telem.cpu().numpy())
    print(f"\n  Random telemetry:")
    print(f"    Expected: {expected_rand}")
    print(f"    Generated: {gen_text_rand}")

    # Falsification tests
    print("\n" + "="*60)
    print("[z1906] FALSIFICATION BATTERY")
    print("="*60)

    tests = {}

    # T1: Does generated text match expected for real telemetry?
    print("\n[z1906] T1: Real Telemetry Accuracy")

    def text_similarity(a: str, b: str) -> float:
        """Simple character-level similarity."""
        if not a or not b:
            return 0.0
        matches = sum(1 for ca, cb in zip(a, b) if ca == cb)
        return matches / max(len(a), len(b))

    accuracy_real = text_similarity(gen_text_real, expected_real)
    print(f"  Similarity to expected: {accuracy_real:.2%}")
    tests['T1_real_accuracy'] = {
        'similarity': accuracy_real,
        'expected': expected_real,
        'generated': gen_text_real,
        'falsified': accuracy_real < 0.5,  # Should be >50% similar
    }

    # T2: Does zero telemetry produce different output than real?
    print("\n[z1906] T2: Zero vs Real Differentiation")
    diff_zero_real = 1 - text_similarity(gen_text_zero, gen_text_real)
    print(f"  Difference (zero vs real): {diff_zero_real:.2%}")
    tests['T2_zero_vs_real'] = {
        'difference': diff_zero_real,
        'gen_zero': gen_text_zero,
        'gen_real': gen_text_real,
        'falsified': diff_zero_real < 0.2,  # Should be >20% different
    }

    # T3: Does random telemetry produce different output than real?
    print("\n[z1906] T3: Random vs Real Differentiation")
    diff_rand_real = 1 - text_similarity(gen_text_rand, gen_text_real)
    print(f"  Difference (random vs real): {diff_rand_real:.2%}")
    tests['T3_random_vs_real'] = {
        'difference': diff_rand_real,
        'gen_rand': gen_text_rand,
        'gen_real': gen_text_real,
        'falsified': diff_rand_real < 0.2,
    }

    # T4: Can model extract numeric values correctly?
    print("\n[z1906] T4: Numeric Value Extraction")
    # Parse generated text for numbers
    import re
    numbers_real = re.findall(r'\d+', gen_text_real)
    numbers_expected = re.findall(r'\d+', expected_real)
    if numbers_real and numbers_expected:
        # Check if first number (temp) is close
        try:
            extracted_temp = int(numbers_real[0])
            expected_temp = int(numbers_expected[0])
            temp_error = abs(extracted_temp - expected_temp)
            print(f"  Expected temp: {expected_temp}, Extracted: {extracted_temp}, Error: {temp_error}")
            tests['T4_numeric_accuracy'] = {
                'expected': expected_temp,
                'extracted': extracted_temp,
                'error': temp_error,
                'falsified': temp_error > 20,  # Should be within 20 degrees
            }
        except (ValueError, IndexError):
            tests['T4_numeric_accuracy'] = {'falsified': True, 'reason': 'parse_failed'}
    else:
        tests['T4_numeric_accuracy'] = {'falsified': True, 'reason': 'no_numbers'}

    # T5: Consistency across multiple generations
    print("\n[z1906] T5: Generation Consistency")
    generations = []
    with torch.no_grad():
        for _ in range(5):
            gen = model(real_telem)
            gen_text = bytes(gen[0].cpu().tolist()).decode('utf-8', errors='ignore').strip('\x00')
            generations.append(gen_text)

    # Check if all generations are similar (deterministic)
    consistency = sum(text_similarity(generations[0], g) for g in generations[1:]) / 4
    print(f"  Consistency: {consistency:.2%}")
    tests['T5_consistency'] = {
        'consistency': consistency,
        'generations': generations,
        'falsified': consistency < 0.8,  # Should be >80% consistent
    }

    results['falsification_tests'] = tests

    num_falsified = sum(1 for t in tests.values() if t.get('falsified', False))
    results['num_falsified'] = num_falsified
    results['num_total'] = len(tests)
    results['status'] = 'FALSIFIED' if num_falsified > 0 else 'SURVIVED ALL TESTS'

    print(f"\n{'='*60}")
    print(f"[z1906] RESULTS")
    print(f"{'='*60}")
    for name, t in tests.items():
        status = "FALSIFIED" if t.get('falsified', False) else "SURVIVED"
        print(f"  {status} {name}")

    print(f"\n[z1906] Tests survived: {len(tests) - num_falsified}/{len(tests)}")
    print(f"[z1906] Status: {results['status']}")

    if num_falsified == 0:
        print("\n[z1906] ALL TESTS SURVIVED!")
        print("[z1906] Model has TRUE hardware-to-output causal dependence")
        print("[z1906] This is the strongest form of embodiment tested")

    telemetry.stop()

    # Save
    results_path = Path(__file__).parent.parent / "results" / "z1906_hardware_description_task.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[z1906] Results saved to {results_path}")

    return results


if __name__ == "__main__":
    run_experiment()
