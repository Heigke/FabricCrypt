#!/usr/bin/env python3
"""
Z85: Auto-Calibrated Validation
Runs validation with machine-specific calibration.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import json
import logging
import argparse
from datetime import datetime
from dataclasses import asdict
from typing import Dict, Any, List, Tuple
import numpy as np

from src.atom import (
    create_sensor, create_actuator, AtomConfig, BodyStateTracker,
    EmbodiedExpressionController, MachineCalibration, MultiScaleController,
    InferencePhase, AtomicTelemetrySnapshot
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_validation(
    model,
    tokenizer,
    device,
    calibration: MachineCalibration,
    num_tokens: int = 100,
    num_requests: int = 2,
    modulation_strength: float = 0.5,
) -> Dict[str, Any]:
    """Run validation with given calibration."""
    
    sensor = create_sensor(device_id=0)
    atom_config = AtomConfig(rate_limit_ms=50, latency_slo_ms=calibration.tbt_slo_ms)
    body_tracker = BodyStateTracker(config=atom_config)
    
    controller = EmbodiedExpressionController(
        modulation_strength=modulation_strength,
        enabled=True,
        calibration=calibration,
    )
    
    prompt = "Write a detailed explanation of how computers process information."
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    all_tokens = 0
    total_energy = 0.0
    mode_counts = {}
    temps = []
    latent_history = []
    
    for req in range(num_requests):
        # Warmup sensor
        for _ in range(3):
            sensor.read()
        
        energy_start = sensor.read().energy_joules
        last_energy = energy_start
        
        current_ids = input_ids
        past_key_values = None
        controller.reset()
        
        with torch.no_grad():
            for i in range(num_tokens):
                # Sense
                snap = sensor.read()
                energy_now = snap.energy_joules
                delta_energy = energy_now - last_energy if energy_now > 0 and last_energy > 0 else 0.0
                last_energy = energy_now
                
                # Update body state
                body_state = body_tracker.update(
                    snapshot=snap,
                    tokens_generated=1,
                    phase=InferencePhase.DECODE,
                    latency_ms=12.0,
                    kv_length=input_ids.shape[1] + i,
                    actuation_applied=False,
                )
                
                # Get expression params
                expr_params = controller.step(body_state, delta_energy_j=delta_energy)
                
                # Track
                mode_name = expr_params.mode.name
                mode_counts[mode_name] = mode_counts.get(mode_name, 0) + 1
                temps.append(expr_params.temperature)
                
                latent = controller.get_current_latent()
                if latent:
                    latent_history.append(asdict(latent))
                
                # Generate
                outputs = model(
                    input_ids=current_ids[:, -1:] if past_key_values else current_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                past_key_values = outputs.past_key_values
                logits = outputs.logits[:, -1, :] / expr_params.temperature
                probs = torch.softmax(logits, dim=-1)
                
                top_k_probs, top_k_indices = torch.topk(probs[0], expr_params.top_k)
                sampled_idx = torch.multinomial(top_k_probs, 1)
                next_token = top_k_indices[sampled_idx[0]].view(1, 1)
                current_ids = torch.cat([current_ids, next_token], dim=1)
        
        energy_end = sensor.read().energy_joules
        total_energy += energy_end - energy_start
        all_tokens += num_tokens
    
    j_per_token = total_energy / all_tokens if all_tokens > 0 else 0
    avg_temp = np.mean(temps)
    temp_variance = np.var(temps)
    params_changed = temp_variance > 1e-6
    
    avg_strain = np.mean([l['strain'] for l in latent_history]) if latent_history else 0
    avg_urgency = np.mean([l['urgency'] for l in latent_history]) if latent_history else 0
    avg_debt = np.mean([l['debt'] for l in latent_history]) if latent_history else 0
    avg_margin = np.mean([l['margin'] for l in latent_history]) if latent_history else 0
    
    return {
        'j_per_token': j_per_token,
        'mode_distribution': mode_counts,
        'avg_temperature': avg_temp,
        'temperature_variance': temp_variance,
        'params_changed': params_changed,
        'avg_strain': avg_strain,
        'avg_urgency': avg_urgency,
        'avg_debt': avg_debt,
        'avg_margin': avg_margin,
        'calibration': asdict(calibration),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokens', type=int, default=100)
    parser.add_argument('--requests', type=int, default=2)
    parser.add_argument('--output', type=str, default='results/z85_auto_calibrated')
    args = parser.parse_args()
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    model_name = 'Qwen/Qwen2.5-0.5B-Instruct'
    logger.info(f'Loading model: {model_name}')
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map='auto',
        trust_remote_code=True
    )
    model.eval()
    device = next(model.parameters()).device
    
    sensor = create_sensor(device_id=0)
    
    # Step 1: Auto-tune calibration
    logger.info('Auto-tuning calibration...')
    calibration = MachineCalibration.auto_tune(sensor, model, tokenizer, device, num_tokens=50)
    logger.info(f'Auto-tuned: target={calibration.j_per_token_target:.3f}, max={calibration.j_per_token_max:.3f}')
    
    # Step 2: Run validation with different modulation strengths
    modulations = [0.0, 0.5, 1.0]
    results = []
    
    for mod in modulations:
        logger.info(f'\nValidating modulation_strength={mod:.2f}')
        result = run_validation(
            model, tokenizer, device,
            calibration=calibration,
            num_tokens=args.tokens,
            num_requests=args.requests,
            modulation_strength=mod,
        )
        result['modulation_strength'] = mod
        results.append(result)
        
        modes_str = ', '.join([f'{k}:{v}' for k, v in result['mode_distribution'].items()])
        logger.info(f'  J/token: {result["j_per_token"]:.4f}')
        logger.info(f'  Modes: {modes_str}')
        logger.info(f'  Params changed: {result["params_changed"]}')
        logger.info(f'  Avg temp: {result["avg_temperature"]:.3f}')
        logger.info(f'  Latent: strain={result["avg_strain"]:.3f}, debt={result["avg_debt"]:.3f}')
    
    # Save results
    os.makedirs(args.output, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(args.output, f'auto_calibrated_{timestamp}.json')

    # Convert numpy types to Python native types for JSON
    def convert_to_native(obj):
        if isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(v) for v in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(output_file, 'w') as f:
        json.dump(convert_to_native({
            'auto_calibration': asdict(calibration),
            'results': results,
            'timestamp': timestamp,
        }), f, indent=2)
    
    logger.info(f'\nResults saved to: {output_file}')
    
    # Summary
    print('\n' + '='*80)
    print('AUTO-CALIBRATED VALIDATION RESULTS')
    print('='*80)
    print(f'Calibration: target={calibration.j_per_token_target:.3f}, max={calibration.j_per_token_max:.3f}')
    print('-'*80)
    print(f'{"Mod":>5} {"J/tok":>8} {"Params":>8} {"Modes":>30} {"Temp":>8} {"Debt":>8}')
    print('-'*80)
    
    for r in results:
        modes_str = ', '.join([f'{k}:{v}' for k, v in r['mode_distribution'].items()])
        print(f'{r["modulation_strength"]:5.2f} {r["j_per_token"]:8.4f} {"YES" if r["params_changed"] else "NO":>8} {modes_str:>30} {r["avg_temperature"]:8.3f} {r["avg_debt"]:8.3f}')
    print('='*80)


if __name__ == '__main__':
    main()
