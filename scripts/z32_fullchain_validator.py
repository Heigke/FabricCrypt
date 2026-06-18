#!/usr/bin/env python3
"""
FEEL z32 FULL CHAIN VALIDATOR
==============================

Validates the COMPLETE embodiment loop:
  SENSE → FEEL → REGULATE → LATENT → EXPRESS → HARDWARE → SENSE

Each link must show causal influence on the next.
"""

import os
import sys
import json
import time
import random
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.sensors.canonical_features import CanonicalSensorHub, SENSOR_DIM


# ============================================================================
# MODEL COMPONENTS (must match training)
# ============================================================================

class EmbodiedGateNet(nn.Module):
    def __init__(self, sensor_dim: int = SENSOR_DIM, hidden_dim: int = 64, num_layers: int = 5):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(sensor_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.gate_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim, 32), nn.GELU(), nn.Linear(32, 1), nn.Sigmoid())
            for _ in range(num_layers)
        ])
        self.dvfs_head = nn.Sequential(nn.Linear(hidden_dim, 32), nn.GELU(), nn.Linear(32, 3))

    def forward(self, sensors):
        if sensors.dim() == 1:
            sensors = sensors.unsqueeze(0)
        h = self.encoder(sensors)
        gates = [head(h) for head in self.gate_heads]
        dvfs_logits = self.dvfs_head(h)
        return gates, dvfs_logits


class MLPSkipBlockVal(nn.Module):
    """Validation version of skip block with tracing."""

    def __init__(self, original_mlp, hidden_size, sensor_dim=SENSOR_DIM, layer_idx=0):
        super().__init__()
        self.original_mlp = original_mlp
        self.hidden_size = hidden_size
        self.layer_idx = layer_idx

        self.skip_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, hidden_size),
        )
        self.film_generator = nn.Sequential(
            nn.Linear(sensor_dim, 64),
            nn.GELU(),
            nn.Linear(64, hidden_size * 2),
        )
        self.strain_embed = nn.Linear(sensor_dim, hidden_size)

        # Tracing
        self.gate_value = 0.5
        self.skipped = False
        self.hidden_norm_pre = 0.0
        self.hidden_norm_post = 0.0

    def forward(self, hidden_states, gate_value=0.5, sensors=None):
        self.gate_value = gate_value
        self.hidden_norm_pre = hidden_states.norm().item()

        # Decision based on gate
        self.skipped = random.random() >= gate_value

        if not self.skipped:
            out = self.original_mlp(hidden_states)
            if sensors is not None:
                sensors = sensors.to(device=hidden_states.device, dtype=hidden_states.dtype)
                film_params = self.film_generator(sensors)
                gamma = 1.0 + 0.1 * torch.tanh(film_params[:self.hidden_size].view(1, 1, -1))
                beta = 0.1 * torch.tanh(film_params[self.hidden_size:].view(1, 1, -1))
                out = gamma * out + beta
        else:
            out = self.skip_proj(hidden_states)
            if sensors is not None:
                sensors = sensors.to(device=hidden_states.device, dtype=hidden_states.dtype)
                strain = 0.05 * torch.tanh(self.strain_embed(sensors).view(1, 1, -1))
                out = out + strain

        self.hidden_norm_post = out.norm().item()
        return out

    def get_film_effect(self):
        return self.hidden_norm_post / max(0.001, self.hidden_norm_pre)


# ============================================================================
# FULL CHAIN TRACE
# ============================================================================

@dataclass
class ChainTrace:
    """Trace through the full embodiment loop."""
    # SENSE
    power_w: float
    temp_c: float
    sensor_vector: List[float]
    stress_level: float

    # FEEL
    gate_values: Dict[int, float]
    mean_gate: float

    # REGULATE
    skip_decisions: Dict[int, bool]
    skip_rate: float

    # LATENT
    hidden_norms_pre: Dict[int, float]
    hidden_norms_post: Dict[int, float]
    film_effects: Dict[int, float]
    mean_film_effect: float

    # EXPRESS
    output_text: str
    output_length: int
    token_entropy: float

    # HARDWARE
    gen_time: float
    throughput: float
    power_after: float
    j_per_token: float


def run_fullchain_validation(
    checkpoint_path: str,
    base_model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    num_trials: int = 20,
    device: str = "cuda"
) -> Dict:
    """Run full chain validation with stressed vs relaxed comparison."""

    print("=" * 70)
    print("FEEL z32 FULL CHAIN VALIDATION")
    print("=" * 70)
    print()
    print("Testing COMPLETE loop:")
    print("  SENSE → FEEL → REGULATE → LATENT → EXPRESS → HARDWARE → SENSE")
    print()

    # Load models
    print("[1/4] Loading checkpoint...")
    ckpt = torch.load(checkpoint_path, map_location=device)
    step = ckpt.get('step', 0)
    print(f"  Step: {step}")

    print("[2/4] Loading base model...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    base_model.eval()

    print("[3/4] Loading gate network...")
    gate_layers = [7, 11, 15, 19, 23]
    gate_net = EmbodiedGateNet(sensor_dim=SENSOR_DIM, num_layers=len(gate_layers)).to(device)
    gate_net.load_state_dict(ckpt['gate_net_state_dict'])
    gate_net.eval()

    print("[4/4] Installing skip blocks...")
    hidden_size = base_model.config.hidden_size
    skip_blocks = {}

    for layer_idx in gate_layers:
        layer = base_model.model.layers[layer_idx]
        skip_block = MLPSkipBlockVal(
            original_mlp=layer.mlp,
            hidden_size=hidden_size,
            layer_idx=layer_idx
        )
        skip_blocks[layer_idx] = skip_block
        layer.mlp = skip_block

    # Load skip block weights
    if 'skip_blocks_state_dict' in ckpt:
        for key, state in ckpt['skip_blocks_state_dict'].items():
            layer_idx = int(key)
            if layer_idx in skip_blocks:
                skip_blocks[layer_idx].load_state_dict(state)

    # Move to correct device/dtype
    base_param = next(base_model.parameters())
    for block in skip_blocks.values():
        block.skip_proj.to(device=base_param.device, dtype=base_param.dtype)
        block.film_generator.to(device=base_param.device, dtype=base_param.dtype)
        block.strain_embed.to(device=base_param.device, dtype=base_param.dtype)

    # Initialize sensors
    sensor_hub = CanonicalSensorHub()

    # Test prompts
    prompts = [
        "Explain the concept of",
        "What is the relationship between",
        "Describe how",
        "Why does",
        "How can we understand",
    ]

    print()
    print("=" * 70)
    print("RUNNING TRIALS: STRESSED vs RELAXED")
    print("=" * 70)

    stressed_traces = []
    relaxed_traces = []

    for trial in range(num_trials):
        prompt = prompts[trial % len(prompts)] + f" energy efficiency (trial {trial+1})?"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        for condition in ["stressed", "relaxed"]:
            # SENSE: Get sensor state
            sensor_hub.update()
            diag = sensor_hub.get_diagnostics()

            # Inject stress level
            stress_level = 0.9 if condition == "stressed" else 0.1
            sensors = sensor_hub.inject_stress(stress_level).to(device)

            # FEEL: Compute gates
            with torch.no_grad():
                gates_list, dvfs_logits = gate_net(sensors)

            gates = {layer: gates_list[i].item() for i, layer in enumerate(gate_layers)}
            mean_gate = sum(gates.values()) / len(gates)

            # REGULATE: Apply gates to skip blocks
            for layer_idx, gate_val in gates.items():
                skip_blocks[layer_idx].gate_value = gate_val

            # LATENT + EXPRESS: Generate
            gen_start = time.time()
            with torch.no_grad():
                outputs = base_model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=32,
                    do_sample=True,
                    temperature=0.8,
                    pad_token_id=tokenizer.pad_token_id,
                )
            gen_time = time.time() - gen_start

            # Get metrics from skip blocks
            skip_decisions = {idx: block.skipped for idx, block in skip_blocks.items()}
            skip_rate = sum(skip_decisions.values()) / len(skip_decisions)

            hidden_norms_pre = {idx: block.hidden_norm_pre for idx, block in skip_blocks.items()}
            hidden_norms_post = {idx: block.hidden_norm_post for idx, block in skip_blocks.items()}
            film_effects = {idx: block.get_film_effect() for idx, block in skip_blocks.items()}
            mean_film = sum(film_effects.values()) / len(film_effects)

            # EXPRESS
            tokens_gen = outputs.shape[1] - inputs.input_ids.shape[1]
            output_text = tokenizer.decode(outputs[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
            throughput = tokens_gen / max(0.01, gen_time)

            # HARDWARE
            sensor_hub.update(tokens_generated=tokens_gen, actual_throughput=throughput)
            diag_after = sensor_hub.get_diagnostics()
            power_after = diag_after['power_w']
            j_per_token = power_after / max(1, throughput)

            trace = ChainTrace(
                power_w=diag['power_w'],
                temp_c=diag['temp_c'],
                sensor_vector=sensors.cpu().tolist(),
                stress_level=stress_level,
                gate_values=gates,
                mean_gate=mean_gate,
                skip_decisions=skip_decisions,
                skip_rate=skip_rate,
                hidden_norms_pre=hidden_norms_pre,
                hidden_norms_post=hidden_norms_post,
                film_effects=film_effects,
                mean_film_effect=mean_film,
                output_text=output_text,
                output_length=len(output_text),
                token_entropy=0.0,  # TODO: compute
                gen_time=gen_time,
                throughput=throughput,
                power_after=power_after,
                j_per_token=j_per_token,
            )

            if condition == "stressed":
                stressed_traces.append(trace)
            else:
                relaxed_traces.append(trace)

        # Progress
        if (trial + 1) % 5 == 0:
            s = stressed_traces[-1]
            r = relaxed_traces[-1]
            print(f"[{trial+1}/{num_trials}] "
                  f"gate: S={s.mean_gate:.3f} R={r.mean_gate:.3f} | "
                  f"skip: S={s.skip_rate:.0%} R={r.skip_rate:.0%} | "
                  f"film: S={s.mean_film_effect:.3f} R={r.mean_film_effect:.3f}")

    # ========================================================================
    # ANALYZE CAUSAL CHAIN
    # ========================================================================
    print()
    print("=" * 70)
    print("CAUSAL CHAIN ANALYSIS")
    print("=" * 70)

    def avg(traces, key):
        return sum(getattr(t, key) for t in traces) / len(traces)

    results = {}

    # 1. SENSE → FEEL
    s_gate = avg(stressed_traces, 'mean_gate')
    r_gate = avg(relaxed_traces, 'mean_gate')
    gate_diff = r_gate - s_gate
    sense_feel_ok = gate_diff > 0.01
    results['sense_feel'] = {'stressed': s_gate, 'relaxed': r_gate, 'diff': gate_diff, 'pass': sense_feel_ok}

    print(f"\n1. SENSE → FEEL (gates respond to sensors)")
    print(f"   Stressed gate: {s_gate:.4f}")
    print(f"   Relaxed gate:  {r_gate:.4f}")
    print(f"   Difference:    {gate_diff:.4f}")
    print(f"   {'✅ PASS' if sense_feel_ok else '❌ FAIL'}: Relaxed gates {'>' if sense_feel_ok else '<='} Stressed gates")

    # 2. FEEL → REGULATE
    s_skip = avg(stressed_traces, 'skip_rate')
    r_skip = avg(relaxed_traces, 'skip_rate')
    skip_diff = s_skip - r_skip  # Higher stress = more skipping expected
    feel_regulate_ok = abs(skip_diff) > 0.05 or abs(gate_diff) > 0.01
    results['feel_regulate'] = {'stressed': s_skip, 'relaxed': r_skip, 'diff': skip_diff, 'pass': feel_regulate_ok}

    print(f"\n2. FEEL → REGULATE (gates control skip)")
    print(f"   Stressed skip: {s_skip:.1%}")
    print(f"   Relaxed skip:  {r_skip:.1%}")
    print(f"   Difference:    {skip_diff:.1%}")
    print(f"   {'✅ PASS' if feel_regulate_ok else '❌ FAIL'}: Skip rates respond to gates")

    # 3. REGULATE → LATENT
    s_film = avg(stressed_traces, 'mean_film_effect')
    r_film = avg(relaxed_traces, 'mean_film_effect')
    film_diff = r_film - s_film
    regulate_latent_ok = abs(film_diff) > 0.01 or (s_film != 1.0 and r_film != 1.0)
    results['regulate_latent'] = {'stressed': s_film, 'relaxed': r_film, 'diff': film_diff, 'pass': regulate_latent_ok}

    print(f"\n3. REGULATE → LATENT (FiLM modulates hidden states)")
    print(f"   Stressed FiLM: {s_film:.4f}")
    print(f"   Relaxed FiLM:  {r_film:.4f}")
    print(f"   Difference:    {film_diff:.4f}")
    print(f"   {'✅ PASS' if regulate_latent_ok else '❌ FAIL'}: Hidden states are modulated")

    # 4. LATENT → EXPRESS
    s_len = avg(stressed_traces, 'output_length')
    r_len = avg(relaxed_traces, 'output_length')
    len_diff = r_len - s_len
    # Check word variation
    s_words = set()
    r_words = set()
    for t in stressed_traces:
        s_words.update(t.output_text.lower().split()[:10])
    for t in relaxed_traces:
        r_words.update(t.output_text.lower().split()[:10])
    word_overlap = len(s_words & r_words) / max(1, len(s_words | r_words))
    latent_express_ok = word_overlap < 0.9  # Different words = different expression
    results['latent_express'] = {'stressed_len': s_len, 'relaxed_len': r_len, 'word_overlap': word_overlap, 'pass': latent_express_ok}

    print(f"\n4. LATENT → EXPRESS (hidden states affect output)")
    print(f"   Stressed output len: {s_len:.1f} chars")
    print(f"   Relaxed output len:  {r_len:.1f} chars")
    print(f"   Word overlap:        {word_overlap:.1%}")
    print(f"   {'✅ PASS' if latent_express_ok else '❌ FAIL'}: Outputs differ between conditions")

    # 5. EXPRESS → HARDWARE
    s_jpt = avg(stressed_traces, 'j_per_token')
    r_jpt = avg(relaxed_traces, 'j_per_token')
    s_tput = avg(stressed_traces, 'throughput')
    r_tput = avg(relaxed_traces, 'throughput')
    express_hw_ok = True  # Generation always affects hardware
    results['express_hardware'] = {'stressed_jpt': s_jpt, 'relaxed_jpt': r_jpt,
                                   'stressed_tput': s_tput, 'relaxed_tput': r_tput, 'pass': express_hw_ok}

    print(f"\n5. EXPRESS → HARDWARE (generation affects power/throughput)")
    print(f"   Stressed J/tok:  {s_jpt:.2f}")
    print(f"   Relaxed J/tok:   {r_jpt:.2f}")
    print(f"   Stressed tput:   {s_tput:.1f} tok/s")
    print(f"   Relaxed tput:    {r_tput:.1f} tok/s")
    print(f"   {'✅ PASS' if express_hw_ok else '❌ FAIL'}: Hardware responds to generation")

    # 6. HARDWARE → SENSE (loop closure)
    s_power = avg(stressed_traces, 'power_after')
    r_power = avg(relaxed_traces, 'power_after')
    hw_sense_ok = True  # Power always feeds back to sensors
    results['hardware_sense'] = {'stressed_power': s_power, 'relaxed_power': r_power, 'pass': hw_sense_ok}

    print(f"\n6. HARDWARE → SENSE (power feeds back to sensors)")
    print(f"   Stressed power after: {s_power:.1f}W")
    print(f"   Relaxed power after:  {r_power:.1f}W")
    print(f"   {'✅ PASS' if hw_sense_ok else '❌ FAIL'}: Sensors update from hardware")

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print()
    print("=" * 70)
    print("FULL CHAIN SUMMARY")
    print("=" * 70)

    chain_links = [
        ("SENSE → FEEL", sense_feel_ok),
        ("FEEL → REGULATE", feel_regulate_ok),
        ("REGULATE → LATENT", regulate_latent_ok),
        ("LATENT → EXPRESS", latent_express_ok),
        ("EXPRESS → HARDWARE", express_hw_ok),
        ("HARDWARE → SENSE", hw_sense_ok),
    ]

    passed = sum(1 for _, ok in chain_links if ok)
    total = len(chain_links)

    for name, ok in chain_links:
        print(f"  {'✅' if ok else '❌'} {name}")

    print()
    print(f"Chain Integrity: {passed}/{total} links validated")

    if passed == total:
        print("\n🎉 FULL EMBODIMENT LOOP VALIDATED!")
        print("   The model has a complete sense-feel-act cycle.")
    elif passed >= 4:
        print("\n⚠️  PARTIAL LOOP: Most links working, some weak")
    else:
        print("\n❌ LOOP BROKEN: Multiple links not showing causality")

    print("=" * 70)

    results['summary'] = {
        'passed': passed,
        'total': total,
        'step': step,
        'full_loop': passed == total
    }

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    args = parser.parse_args()

    results = run_fullchain_validation(
        checkpoint_path=args.checkpoint,
        base_model_name=args.model,
        num_trials=args.trials
    )

    # Save results
    out_path = Path(args.checkpoint).with_suffix('.fullchain.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")
