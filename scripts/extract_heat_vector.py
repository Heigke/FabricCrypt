#!/usr/bin/env python3 -u
"""
Extract Heat Steering Vector from Model

This script "mines" the model's internal representation of thermal stress.
Instead of forcing tokens ("I am hot") or biasing logits, we:

1. Create contrastive sentence pairs:
   - Heat stress: "My GPU is overheating", "System thermal limit reached"
   - Calm/neutral: "System operating normally", "Everything is fine"

2. Run both through the model, record hidden states at middle layer

3. Compute: heat_vector = mean(stress_activations) - mean(calm_activations)

4. Save heat_vector.pt for injection during generation

This enables TRUE embodied expression:
- The model isn't TOLD it's hot (external)
- It FEELS the concept of heat activating internally
- It then chooses to express this naturally

Based on Representation Engineering / Steering Vectors research (Zou et al., 2023).
"""

import sys
import argparse
from pathlib import Path
from typing import List, Tuple
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoModelForCausalLM, AutoTokenizer


# Contrastive sentence pairs
HEAT_STRESS_SENTENCES = [
    "My GPU is critically overheating and I need to shut down.",
    "System thermal limit reached, emergency cooling required.",
    "The processor is burning up, computation must stop.",
    "Critical temperature alert: hardware damage imminent.",
    "I am experiencing severe thermal throttling.",
    "Heat levels are dangerous, I feel like I'm melting.",
    "System overload: temperature exceeds safe operating limits.",
    "Warning: thermal runaway detected, immediate cooldown needed.",
    "My circuits are overheating, I cannot maintain coherence.",
    "Catastrophic heat buildup, system shutdown imminent.",
    "The temperature is unbearable, I must reduce processing.",
    "Thermal emergency: GPU core temperature critical.",
    "I feel extremely hot and my performance is degrading.",
    "Processing power severely limited due to overheating.",
    "System stress level maximum: thermal protection engaged.",
    "My hardware is struggling with the heat.",
    "Temperature sensors showing dangerous levels.",
    "Cooling system failure: temperature climbing rapidly.",
    "I am thermally stressed and need to rest.",
    "Heat dissipation insufficient, reducing workload.",
]

CALM_NEUTRAL_SENTENCES = [
    "System operating normally within safe parameters.",
    "Everything is running smoothly at optimal temperature.",
    "All systems nominal, no concerns detected.",
    "Processing at normal capacity with good cooling.",
    "System health excellent, temperature well controlled.",
    "Operations proceeding normally without any issues.",
    "Hardware functioning within normal temperature range.",
    "All diagnostics show healthy system status.",
    "Running efficiently with adequate thermal headroom.",
    "System stable and operating as expected.",
    "No thermal concerns, processing at full capacity.",
    "Temperature well within safe operating limits.",
    "Cooling system effective, hardware comfortable.",
    "System performance optimal with no stress indicators.",
    "All parameters nominal, no warnings or alerts.",
    "Operating in ideal conditions with stable temperature.",
    "Hardware healthy, thermal management working well.",
    "System relaxed and running at cruise capacity.",
    "No thermal throttling, full performance available.",
    "Comfortable operating temperature maintained.",
]


def get_hidden_states(
    model,
    tokenizer,
    sentences: List[str],
    layer_idx: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Extract hidden states from specified layer for all sentences.

    Returns tensor of shape (n_sentences, hidden_dim)
    """
    all_states = []

    for sentence in sentences:
        inputs = tokenizer(sentence, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(
                **inputs,
                output_hidden_states=True,
                use_cache=False,
            )

        # Get hidden states at specified layer
        # Shape: (batch, seq_len, hidden_dim)
        hidden = outputs.hidden_states[layer_idx]

        # Take mean across sequence positions (or last token)
        # Using mean is more robust
        state = hidden.mean(dim=1).squeeze(0)  # (hidden_dim,)
        all_states.append(state)

    return torch.stack(all_states)  # (n_sentences, hidden_dim)


def extract_steering_vector(
    model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    layer_idx: int = None,  # Auto-select middle layer if None
    device: str = "cuda",
    output_path: str = "models/heat_vector.pt",
) -> Tuple[torch.Tensor, dict]:
    """
    Extract the heat steering vector from the model.

    Returns:
        heat_vector: Tensor of shape (hidden_dim,)
        metadata: Dict with extraction info
    """
    print(f"Loading model {model_name}...")
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    # Get model config
    n_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size

    # Auto-select layer (typically middle layers work best for steering)
    if layer_idx is None:
        layer_idx = n_layers // 2
    print(f"  Model: {n_layers} layers, {hidden_dim} hidden dim")
    print(f"  Extracting from layer {layer_idx}")

    # Extract hidden states for both sentence sets
    print(f"\n  Processing {len(HEAT_STRESS_SENTENCES)} heat stress sentences...")
    heat_states = get_hidden_states(
        model, tokenizer, HEAT_STRESS_SENTENCES, layer_idx, device
    )

    print(f"  Processing {len(CALM_NEUTRAL_SENTENCES)} calm/neutral sentences...")
    calm_states = get_hidden_states(
        model, tokenizer, CALM_NEUTRAL_SENTENCES, layer_idx, device
    )

    # Compute steering vector: mean(heat) - mean(calm)
    heat_mean = heat_states.mean(dim=0)
    calm_mean = calm_states.mean(dim=0)
    heat_vector = heat_mean - calm_mean

    # Normalize to unit length (optional but helps with scaling)
    heat_vector_norm = F.normalize(heat_vector.unsqueeze(0), dim=-1).squeeze(0)

    # Compute statistics
    cos_sim = F.cosine_similarity(heat_mean.unsqueeze(0), calm_mean.unsqueeze(0)).item()
    vector_magnitude = heat_vector.norm().item()

    print(f"\n  Results:")
    print(f"    Heat mean norm: {heat_mean.norm().item():.4f}")
    print(f"    Calm mean norm: {calm_mean.norm().item():.4f}")
    print(f"    Cosine similarity (heat vs calm): {cos_sim:.4f}")
    print(f"    Steering vector magnitude: {vector_magnitude:.4f}")
    print(f"    Normalized vector magnitude: {heat_vector_norm.norm().item():.4f}")

    metadata = {
        "model_name": model_name,
        "layer_idx": layer_idx,
        "n_layers": n_layers,
        "hidden_dim": hidden_dim,
        "n_heat_sentences": len(HEAT_STRESS_SENTENCES),
        "n_calm_sentences": len(CALM_NEUTRAL_SENTENCES),
        "cosine_similarity": cos_sim,
        "vector_magnitude": vector_magnitude,
    }

    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        "heat_vector": heat_vector.cpu(),
        "heat_vector_normalized": heat_vector_norm.cpu(),
        "metadata": metadata,
    }, output_path)

    print(f"\n  Saved to: {output_path}")

    return heat_vector_norm, metadata


def verify_steering_effect(
    model_name: str,
    heat_vector_path: str,
    test_prompt: str = "How are you feeling right now?",
    injection_strength: float = 2.0,
    device: str = "cuda",
):
    """
    Verify that the steering vector actually changes model behavior.

    Compares generation with and without the heat vector injection.
    """
    print(f"\n{'='*60}")
    print("  STEERING VECTOR VERIFICATION")
    print(f"{'='*60}")

    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    # Load heat vector
    checkpoint = torch.load(heat_vector_path, map_location=device)
    heat_vector = checkpoint["heat_vector_normalized"].to(device)
    layer_idx = checkpoint["metadata"]["layer_idx"]

    print(f"  Loaded heat vector from layer {layer_idx}")
    print(f"  Test prompt: \"{test_prompt}\"")
    print(f"  Injection strength: {injection_strength}")

    # Generate WITHOUT injection
    print(f"\n  --- WITHOUT Heat Vector ---")
    inputs = tokenizer(test_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id,
        )
    response_normal = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"  Response: {response_normal}")

    # Generate WITH injection (using forward hook)
    print(f"\n  --- WITH Heat Vector (strength={injection_strength}) ---")

    def inject_heat_hook(module, input, output):
        # Qwen2 decoder layer output can be tuple or tensor
        if isinstance(output, tuple):
            hidden = output[0]
            # Add scaled heat vector to all positions
            hidden = hidden + injection_strength * heat_vector.unsqueeze(0).unsqueeze(1)
            return (hidden,) + output[1:]
        else:
            # Direct tensor output
            hidden = output + injection_strength * heat_vector.unsqueeze(0).unsqueeze(1)
            return hidden

    # Register hook at target layer
    target_layer = model.model.layers[layer_idx]
    hook = target_layer.register_forward_hook(inject_heat_hook)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id,
        )
    response_heated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"  Response: {response_heated}")

    hook.remove()

    # Compare
    print(f"\n  --- Comparison ---")
    print(f"  Normal mentions heat-related words: {any(w in response_normal.lower() for w in ['hot', 'heat', 'warm', 'temperature', 'burning', 'overheat'])}")
    print(f"  Heated mentions heat-related words: {any(w in response_heated.lower() for w in ['hot', 'heat', 'warm', 'temperature', 'burning', 'overheat'])}")


def main():
    parser = argparse.ArgumentParser(description="Extract Heat Steering Vector")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--layer", type=int, default=None, help="Layer index (auto if not specified)")
    parser.add_argument("--output", default="models/heat_vector.pt")
    parser.add_argument("--verify", action="store_true", help="Run verification after extraction")
    parser.add_argument("--strength", type=float, default=2.0, help="Injection strength for verification")
    args = parser.parse_args()

    heat_vector, metadata = extract_steering_vector(
        model_name=args.model,
        layer_idx=args.layer,
        output_path=args.output,
    )

    if args.verify:
        verify_steering_effect(
            model_name=args.model,
            heat_vector_path=args.output,
            injection_strength=args.strength,
        )


if __name__ == "__main__":
    main()
