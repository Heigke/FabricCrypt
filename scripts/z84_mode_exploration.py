#!/usr/bin/env python3
"""
Z84: Mode Exploration Test
Test all expression modes by varying calibration parameters.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import json
import logging
from dataclasses import asdict
from typing import Dict, Any, List, Tuple
import numpy as np

from src.atom import (
    create_sensor, create_actuator, AtomConfig, BodyStateTracker,
    EmbodiedExpressionController, MachineCalibration
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_mode_triggering():
    """Test different calibrations to trigger each mode."""
    
    # Initialize model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    logger.info(f"Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    device = next(model.parameters()).device
    
    # Initialize hardware
    atom_config = AtomConfig(rate_limit_ms=50, latency_slo_ms=50.0)
    sensor = create_sensor(device_id=0)
    body_tracker = BodyStateTracker(config=atom_config)
    
    # Test scenarios with different calibrations
    test_scenarios = [
        {
            "name": "EXPLORE trigger (j_max=5.0, low strain)",
            "j_per_token_max": 5.0,  # Higher max → lower strain ratio
            "tbt_slo_ms": 50.0,
            "expected_mode": "EXPLORE"
        },
        {
            "name": "CONSERVE trigger (j_max=1.0, high strain)",
            "j_per_token_max": 1.0,  # Lower max → higher strain ratio
            "tbt_slo_ms": 50.0,
            "expected_mode": "CONSERVE"
        },
        {
            "name": "URGENT trigger (tbt_slo=15ms)",
            "j_per_token_max": 3.0,
            "tbt_slo_ms": 15.0,  # Very tight SLO → high urgency
            "expected_mode": "URGENT"
        },
        {
            "name": "Baseline (standard calibration)",
            "j_per_token_max": 3.0,
            "tbt_slo_ms": 50.0,
            "expected_mode": "BALANCED/RECOVER"
        },
    ]
    
    prompt = "Explain the concept of machine learning in simple terms."
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    num_tokens = 50  # Short test
    
    results = []
    
    for scenario in test_scenarios:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing: {scenario['name']}")
        logger.info(f"  j_per_token_max={scenario['j_per_token_max']}, tbt_slo_ms={scenario['tbt_slo_ms']}")
        
        # Create controller with custom calibration
        calibration = MachineCalibration(
            j_per_token_target=1.5,
            j_per_token_max=scenario['j_per_token_max'],
            tbt_slo_ms=scenario['tbt_slo_ms'],
            ttft_slo_ms=500.0,
            temp_safe_c=70.0,
            temp_critical_c=85.0,
        )
        
        controller = EmbodiedExpressionController(
            modulation_strength=1.0,
            enabled=True,
            calibration=calibration,
        )
        
        # Generate tokens
        past_key_values = None
        current_ids = input_ids
        mode_counts = {}
        latent_history = []
        
        # Warm up sensor
        for _ in range(3):
            sensor.read()
        last_energy = sensor.read().energy_joules
        
        for i in range(num_tokens):
            # Read sensor
            snap = sensor.read()
            energy_now = snap.energy_joules
            delta_energy = energy_now - last_energy if energy_now > 0 and last_energy > 0 else 0.0
            last_energy = energy_now
            
            # Update body state with timing
            from src.atom.schema import InferencePhase
            body_state = body_tracker.update(
                snapshot=snap,
                tokens_generated=1,  # One token per iteration
                phase=InferencePhase.DECODE,
                latency_ms=12.0,  # Typical TBT
                kv_length=input_ids.shape[1] + i,
                actuation_applied=False,
            )
            
            # Get expression params
            expr_params = controller.step(body_state, delta_energy_j=delta_energy)
            
            # Track mode
            mode_name = expr_params.mode.name
            mode_counts[mode_name] = mode_counts.get(mode_name, 0) + 1
            
            # Track latent
            latent = controller.get_current_latent()
            if latent:
                latent_history.append(asdict(latent))
            
            # Generate one token
            with torch.no_grad():
                outputs = model(
                    input_ids=current_ids[:, -1:] if past_key_values else current_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                past_key_values = outputs.past_key_values
                logits = outputs.logits[:, -1, :]
                
                # Apply expression params
                logits = logits / expr_params.temperature
                probs = torch.softmax(logits, dim=-1)
                
                # Top-k sampling
                top_k_probs, top_k_indices = torch.topk(probs[0], expr_params.top_k)
                sampled_idx = torch.multinomial(top_k_probs, 1)
                next_token = top_k_indices[sampled_idx[0]].view(1, 1)

                current_ids = torch.cat([current_ids, next_token], dim=1)
        
        # Analyze results
        avg_strain = np.mean([l['strain'] for l in latent_history]) if latent_history else 0
        avg_urgency = np.mean([l['urgency'] for l in latent_history]) if latent_history else 0
        avg_debt = np.mean([l['debt'] for l in latent_history]) if latent_history else 0
        avg_margin = np.mean([l['margin'] for l in latent_history]) if latent_history else 0
        
        result = {
            "scenario": scenario['name'],
            "expected_mode": scenario['expected_mode'],
            "mode_distribution": mode_counts,
            "avg_strain": avg_strain,
            "avg_urgency": avg_urgency,
            "avg_debt": avg_debt,
            "avg_margin": avg_margin,
        }
        results.append(result)
        
        logger.info(f"  Mode distribution: {mode_counts}")
        logger.info(f"  Avg latent: strain={avg_strain:.3f}, urgency={avg_urgency:.3f}, debt={avg_debt:.3f}, margin={avg_margin:.3f}")
        
        # Check if expected mode was triggered
        dominant_mode = max(mode_counts.items(), key=lambda x: x[1])[0] if mode_counts else "NONE"
        expected = scenario['expected_mode'].split('/')[0]  # Handle "BALANCED/RECOVER"
        if expected in mode_counts:
            logger.info(f"  ✓ Expected mode '{expected}' was triggered!")
        else:
            logger.info(f"  ✗ Expected '{expected}', got dominant mode: {dominant_mode}")
    
    # Summary
    print("\n" + "="*80)
    print("MODE EXPLORATION SUMMARY")
    print("="*80)
    for r in results:
        modes_str = ", ".join([f"{k}:{v}" for k, v in r['mode_distribution'].items()])
        print(f"\n{r['scenario']}:")
        print(f"  Expected: {r['expected_mode']}")
        print(f"  Got: {modes_str}")
        print(f"  Latent: strain={r['avg_strain']:.3f}, urgency={r['avg_urgency']:.3f}, debt={r['avg_debt']:.3f}, margin={r['avg_margin']:.3f}")
    
    return results

if __name__ == "__main__":
    test_mode_triggering()
