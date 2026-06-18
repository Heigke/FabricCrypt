#!/usr/bin/env python3
"""Quick test of trained adapters."""

import os
import sys
import torch
from pathlib import Path

os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from modeling.hardware_aware_llm import HardwareAwareLLM

print("=" * 60)
print("QUICK ADAPTER TEST")
print("=" * 60)

# Load model
print("\n[1] Loading model...")
model = HardwareAwareLLM(
    base_model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    adapter_type="film",
    sensor_type="hybrid",
    load_in_4bit=True,
)

# Load trained adapters
adapter_path = "models/hardware_aware/epoch_1"
print(f"\n[2] Loading adapters from {adapter_path}...")
model.load_adapters(adapter_path)
print("    Adapters loaded successfully!")

# Test generation at different stress levels
prompt = "What is 2 + 2?"

print("\n[3] Testing generation at LOW stress (0.1)...")
model.sensor.enable_simulation(True)
model.sensor.set_simulated_stress(0.1)
output_calm = model.generate(prompt=prompt, max_new_tokens=100)
print(f"    Response: {output_calm[:200]}...")

print("\n[4] Testing generation at HIGH stress (0.9)...")
model.sensor.set_simulated_stress(0.9)
output_stressed = model.generate(prompt=prompt, max_new_tokens=100)
print(f"    Response: {output_stressed[:200]}...")

print("\n[5] Comparing outputs...")
print(f"    Calm length:     {len(output_calm)} chars")
print(f"    Stressed length: {len(output_stressed)} chars")
print(f"    Difference:      {len(output_calm) - len(output_stressed)} chars")

if len(output_calm) > len(output_stressed):
    print("\n✅ SUCCESS: Calm response is longer (as expected)")
else:
    print("\n⚠️  Stressed response is longer or equal (may need more training)")

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)
