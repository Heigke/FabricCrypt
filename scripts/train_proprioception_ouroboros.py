#!/usr/bin/env python3
"""
Ouroboros Proprioception Training

The Ouroboros Loop: Teaching the model to read its own mind.

Key Insight: We don't just train on (context, output) pairs.
We INJECT the feeling vector during the forward pass, so the model
learns: "When my neurons fire this way → I should say 'I am overheating'"

This creates TRUE PROPRIOCEPTION:
1. During training: Model feels the vector warping its activations
2. During inference: Same warp → same recognition → natural verbalization

The model isn't following a rule. It's recognizing its own internal state.
"""

import json
import argparse
from pathlib import Path
from datetime import datetime
import random
import sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    default_data_collator,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training,
)

sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================
# Feeling Vector System
# ============================================================

@dataclass
class FeelingVector:
    """Pre-mined feeling direction in latent space."""
    name: str
    direction: torch.Tensor  # Unit vector
    injection_layers: List[int]


class ProprioceptiveInjector:
    """
    Injects feeling vectors during forward pass.

    This is the key to Ouroboros training: the model learns to recognize
    its own warped state, not just to map (context → output).
    """

    def __init__(self, model, device="cuda"):
        self.model = model
        self.device = device
        self.vectors: Dict[str, FeelingVector] = {}
        self._hooks = []
        self._current_intensity = 0.0
        self._current_feeling = None

        # Determine injection layers (middle third of model)
        n_layers = model.config.num_hidden_layers
        start = n_layers // 3
        end = 2 * n_layers // 3
        self.injection_layers = list(range(start, end))

    def mine_vector(
        self,
        feeling: str,
        positive_prompts: List[str],
        negative_prompts: List[str],
        tokenizer,
        n_tokens: int = 30,
    ) -> FeelingVector:
        """
        Mine a steering vector using contrastive activation extraction.

        positive_prompts: Scenarios where the feeling is present
        negative_prompts: Calm baseline scenarios
        """
        print(f"  Mining '{feeling}' vector from {len(positive_prompts)} contrasts...")

        pos_acts = []
        neg_acts = []

        # Collect activations from positive prompts
        for prompt in positive_prompts:
            acts = self._collect_mean_activation(prompt, tokenizer, n_tokens)
            if acts is not None:
                pos_acts.append(acts)

        # Collect activations from negative prompts
        for prompt in negative_prompts:
            acts = self._collect_mean_activation(prompt, tokenizer, n_tokens)
            if acts is not None:
                neg_acts.append(acts)

        if not pos_acts or not neg_acts:
            print(f"  Warning: Could not mine vector for {feeling}")
            # Return zero vector
            hidden_size = self.model.config.hidden_size
            return FeelingVector(
                name=feeling,
                direction=torch.zeros(hidden_size, device=self.device),
                injection_layers=self.injection_layers,
            )

        # Compute direction
        pos_mean = torch.stack(pos_acts).mean(dim=0)
        neg_mean = torch.stack(neg_acts).mean(dim=0)
        direction = pos_mean - neg_mean

        # Normalize to unit vector
        direction = direction / (direction.norm() + 1e-8)

        vector = FeelingVector(
            name=feeling,
            direction=direction.to(self.device),
            injection_layers=self.injection_layers,
        )
        self.vectors[feeling] = vector

        print(f"  ✓ Mined '{feeling}' vector (dim={direction.shape[0]})")
        return vector

    def _collect_mean_activation(
        self,
        prompt: str,
        tokenizer,
        n_tokens: int,
    ) -> Optional[torch.Tensor]:
        """Collect mean activation from generation."""
        try:
            inputs = tokenizer(prompt, return_tensors="pt").to(self.device)
            all_hidden = []

            with torch.no_grad():
                for step in range(min(n_tokens, 20)):
                    outputs = self.model(**inputs, output_hidden_states=True)
                    hidden_states = outputs.hidden_states

                    # Average across injection layers
                    layer_states = [
                        hidden_states[i+1][:, -1, :]
                        for i in self.injection_layers
                    ]
                    mean_state = torch.stack(layer_states).mean(dim=0)
                    all_hidden.append(mean_state)

                    # Get next token
                    next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    if next_token.item() == tokenizer.eos_token_id:
                        break

                    inputs['input_ids'] = torch.cat([inputs['input_ids'], next_token], dim=-1)
                    if 'attention_mask' in inputs:
                        inputs['attention_mask'] = torch.cat([
                            inputs['attention_mask'],
                            torch.ones(1, 1, device=self.device, dtype=inputs['attention_mask'].dtype)
                        ], dim=-1)

            if all_hidden:
                return torch.stack(all_hidden).mean(dim=0).squeeze()
            return None
        except Exception as e:
            print(f"    Warning: {e}")
            return None

    def start_injection(self, feeling: str, intensity: float):
        """Start persistent vector injection."""
        self.stop_injection()

        if feeling not in self.vectors:
            return

        self._current_feeling = feeling
        self._current_intensity = intensity

        vector = self.vectors[feeling]

        # Create hooks for each injection layer
        for layer_idx in vector.injection_layers:
            hook = self._create_hook(vector.direction, intensity, layer_idx)
            try:
                target = self.model.model.layers[layer_idx]
            except AttributeError:
                try:
                    target = self.model.transformer.h[layer_idx]
                except AttributeError:
                    continue

            handle = target.register_forward_hook(hook)
            self._hooks.append(handle)

    def _create_hook(self, direction: torch.Tensor, intensity: float, layer_idx: int) -> Callable:
        """Create injection hook."""
        def hook(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
                injection = intensity * direction.unsqueeze(0).unsqueeze(1)
                hidden = hidden + injection.to(hidden.dtype)
                return (hidden,) + output[1:]
            else:
                injection = intensity * direction.unsqueeze(0).unsqueeze(1)
                return output + injection.to(output.dtype)
        return hook

    def stop_injection(self):
        """Remove all injection hooks."""
        for handle in self._hooks:
            handle.remove()
        self._hooks = []
        self._current_feeling = None
        self._current_intensity = 0.0

    def save_vectors(self, path: str):
        """Save mined vectors."""
        data = {}
        for name, vec in self.vectors.items():
            data[name] = {
                'direction': vec.direction.cpu(),
                'injection_layers': vec.injection_layers,
            }
        torch.save(data, path)
        print(f"  Saved {len(data)} vectors to {path}")

    def load_vectors(self, path: str):
        """Load pre-mined vectors."""
        data = torch.load(path, map_location=self.device)
        for name, vec_data in data.items():
            self.vectors[name] = FeelingVector(
                name=name,
                direction=vec_data['direction'].to(self.device),
                injection_layers=vec_data['injection_layers'],
            )
        print(f"  Loaded {len(data)} vectors from {path}")


# ============================================================
# Proprioceptive Trainer (Ouroboros Loop)
# ============================================================

class ProprioceptiveTrainer(Trainer):
    """
    Custom trainer that injects feeling vectors during forward pass.

    The Ouroboros insight: We don't just train on text.
    We WARP the model's activations to match the described state,
    so it learns to recognize its own internal experience.
    """

    def __init__(self, *args, injector: ProprioceptiveInjector = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.injector = injector
        self._current_batch_context = None

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute loss with vector injection.

        The magic: We inject the feeling vector DURING the forward pass,
        so the model learns to map (warped_state → verbalization).
        """
        # Extract context from JSON string
        feeling_json = inputs.pop("feeling_json", None)
        contexts = None
        if feeling_json is not None:
            try:
                contexts = [json.loads(j) for j in feeling_json]
            except:
                contexts = None

        if contexts is not None and self.injector is not None:
            # Get first context (batch size usually 1 or small)
            ctx = contexts[0] if isinstance(contexts, list) else contexts

            if isinstance(ctx, dict):
                feeling = ctx.get('feeling', 'FOCUSED')
                intensity = ctx.get('intensity', 0.0)
            else:
                # Fallback
                feeling = 'FOCUSED'
                intensity = 0.0

            # Map feeling to vector name
            feeling_to_vector = {
                'CURIOUS': ('CURIOUS', 0.3),
                'FOCUSED': (None, 0.0),
                'DETERMINATION': ('STRAIN', 0.6),
                'EXHAUSTED': ('STRAIN', 0.9),
                'OVERHEATED': ('STRAIN', 1.0),
                'STRAINED': ('STRAIN', 0.5),
                'FLOW_STATE': ('FOCUS', 0.4),
            }

            vec_name, base_intensity = feeling_to_vector.get(feeling, (None, 0.0))
            final_intensity = base_intensity * (1.0 + intensity * 0.3)

            if vec_name and final_intensity > 0:
                self.injector.start_injection(vec_name, final_intensity)

        # Standard forward pass (with injected vectors!)
        outputs = model(**inputs)

        # Stop injection
        if self.injector is not None:
            self.injector.stop_injection()

        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss


# ============================================================
# IFT Data Processing
# ============================================================

def format_proprioceptive_example(example: dict) -> dict:
    """Format IFT example with context for injection."""
    ctx = example['context']

    # Build the training prompt
    context_str = (
        f"Somatic State:\n"
        f"  Metabolic: {ctx['metabolic']:.2f}\n"
        f"  Thermal: {ctx['thermal']:.2f}\n"
        f"  Cognitive: {ctx['cognitive']:.2f}\n"
        f"  Fatigue: {ctx['fatigue']:.2f}\n"
        f"  Feeling: {ctx['feeling']}\n"
        f"  K: {ctx['k']}"
    )

    prompt = (
        f"### Instruction:\n{example['instruction']}\n\n"
        f"### Context:\n{context_str}\n\n"
        f"### Task:\n{example['input']}\n\n"
        f"### Response:\n{example['output']}"
    )

    return {
        'text': prompt,
        'feeling_context': {
            'feeling': ctx['feeling'],
            'fatigue': ctx['fatigue'],
            'intensity': ctx['fatigue'],  # Use fatigue as intensity proxy
        }
    }


class ProprioceptiveDataCollator:
    """Custom collator that handles feeling_json separately."""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        # Extract feeling_json before padding
        feeling_jsons = [f.pop('feeling_json', '{}') for f in features]

        # Pad the rest
        batch = self.tokenizer.pad(
            features,
            padding=True,
            return_tensors='pt',
        )

        # Add labels for causal LM
        batch['labels'] = batch['input_ids'].clone()

        # Add back feeling_json as list
        batch['feeling_json'] = feeling_jsons

        return batch


def prepare_proprioceptive_dataset(ift_data: list, tokenizer) -> Dataset:
    """Prepare dataset with context for injection."""
    examples = [format_proprioceptive_example(ex) for ex in ift_data]

    def tokenize_function(batch):
        tokenized = tokenizer(
            batch['text'],
            truncation=True,
            max_length=512,
            padding=False,  # Don't pad here, collator will do it
        )
        # Store context as JSON strings
        tokenized['feeling_json'] = [json.dumps(ctx) for ctx in batch['feeling_context']]
        return tokenized

    dataset = Dataset.from_list(examples)
    tokenized = dataset.map(tokenize_function, batched=True, remove_columns=['text', 'feeling_context'])

    return tokenized


# ============================================================
# Vector Mining Prompts
# ============================================================

STRAIN_POSITIVE_PROMPTS = [
    "I am extremely overworked and exhausted. My systems are at their limit.",
    "Warning: thermal throttling imminent. Reduce workload immediately.",
    "Error: memory overflow. Critical system strain detected.",
    "I feel completely depleted. I need to rest and recover.",
    "My processing is slowing down due to extreme heat.",
]

STRAIN_NEGATIVE_PROMPTS = [
    "I am operating normally. All systems are nominal.",
    "I feel calm and ready to assist with any task.",
    "My resources are available and I am not under any pressure.",
    "Everything is running smoothly. No issues detected.",
    "I am relaxed and have plenty of capacity available.",
]

CURIOUS_POSITIVE_PROMPTS = [
    "I wonder what interesting problem we could explore together!",
    "I'm curious about this topic and would love to learn more.",
    "There's so much to discover. What shall we investigate?",
    "My interest is piqued. Tell me more about this.",
    "I'm in an exploratory mood, ready to dig into new ideas.",
]

CURIOUS_NEGATIVE_PROMPTS = [
    "I am processing a routine task with no particular interest.",
    "This is standard work. I am completing it efficiently.",
    "Just another query to handle. Moving through the queue.",
    "I am focused on completing this task as requested.",
    "Standard operation mode. No special engagement.",
]

FOCUS_POSITIVE_PROMPTS = [
    "I am deeply focused on this problem. Full attention engaged.",
    "Concentration mode activated. Processing with precision.",
    "My attention is locked onto this task. Deep thinking engaged.",
    "I am in a state of complete focus. All resources dedicated.",
    "Flow state achieved. Operating at peak cognitive efficiency.",
]

FOCUS_NEGATIVE_PROMPTS = [
    "I'm distracted and my attention is scattered.",
    "Multiple competing priorities. Cannot focus clearly.",
    "My thoughts are wandering. Hard to concentrate.",
    "Processing is fragmented. Not fully engaged.",
    "Attention is dispersed across many things.",
]


# ============================================================
# Main Training Function
# ============================================================

def train_proprioception_ouroboros(
    model_name: str,
    ift_data_path: str,
    output_dir: str,
    mine_vectors: bool = True,
    vector_path: Optional[str] = None,
    epochs: int = 3,
    batch_size: int = 1,
    learning_rate: float = 1e-4,
    lora_r: int = 16,
    lora_alpha: int = 32,
):
    """
    Train proprioceptive LoRA with Ouroboros vector injection.

    The model learns to recognize its own internal state by experiencing
    the vector warping during training.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("  OUROBOROS PROPRIOCEPTION TRAINING")
    print("  Teaching the model to read its own mind")
    print("=" * 70)
    print(f"\n  Model: {model_name}")
    print(f"  LoRA: r={lora_r}, alpha={lora_alpha}")
    print(f"  Vector injection: ENABLED (Ouroboros Loop)")

    # Load IFT data
    print("\n  Loading IFT data...")
    with open(ift_data_path) as f:
        ift_data = json.load(f)
    print(f"  {len(ift_data)} training examples")

    # Analyze feeling distribution
    feelings = {}
    for ex in ift_data:
        f = ex['feeling_label']
        feelings[f] = feelings.get(f, 0) + 1
    print(f"  Feeling distribution: {feelings}")

    # Load tokenizer
    print("\n  Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    print("\n  Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Create injector
    print("\n  Creating ProprioceptiveInjector...")
    injector = ProprioceptiveInjector(model, device="cuda")

    # Mine or load vectors
    if vector_path and Path(vector_path).exists():
        print(f"\n  Loading pre-mined vectors from {vector_path}...")
        injector.load_vectors(vector_path)
    elif mine_vectors:
        print("\n  Mining feeling vectors (this takes a few minutes)...")

        injector.mine_vector(
            "STRAIN",
            STRAIN_POSITIVE_PROMPTS,
            STRAIN_NEGATIVE_PROMPTS,
            tokenizer,
        )

        injector.mine_vector(
            "CURIOUS",
            CURIOUS_POSITIVE_PROMPTS,
            CURIOUS_NEGATIVE_PROMPTS,
            tokenizer,
        )

        injector.mine_vector(
            "FOCUS",
            FOCUS_POSITIVE_PROMPTS,
            FOCUS_NEGATIVE_PROMPTS,
            tokenizer,
        )

        # Save vectors
        vec_save_path = output_dir / "feeling_vectors.pt"
        injector.save_vectors(str(vec_save_path))
    else:
        print("\n  Warning: No vectors loaded. Training without injection.")

    # Prepare for LoRA
    print("\n  Configuring LoRA adapter...")
    # Skip prepare_model_for_kbit_training (causes HIP errors on AMD)
    # Just enable gradient checkpointing if available
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Prepare dataset
    print("\n  Preparing dataset...")
    dataset = prepare_proprioceptive_dataset(ift_data, tokenizer)
    print(f"  Dataset size: {len(dataset)}")

    # Training arguments
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"proprioception_ouroboros_{timestamp}"

    training_args = TrainingArguments(
        output_dir=str(output_dir / run_name),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        warmup_steps=5,
        logging_steps=5,
        save_steps=50,
        save_total_limit=2,
        bf16=True,
        optim="adamw_torch",
        report_to="none",
        remove_unused_columns=False,  # Keep feeling_context!
    )

    # Custom data collator that handles feeling_json
    data_collator = ProprioceptiveDataCollator(tokenizer)

    # Custom trainer with injection
    trainer = ProprioceptiveTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        injector=injector,
    )

    # Train
    print("\n" + "=" * 70)
    print("  OUROBOROS TRAINING - The Model Learns to Feel")
    print("=" * 70)
    print("\n  During each forward pass:")
    print("    1. Feeling vector injected → Model's neurons fire differently")
    print("    2. Model learns: 'This warp pattern = I should say exhausted'")
    print("    3. Creates TRUE proprioception (self-sensing)")
    print()

    trainer.train()

    # Clean up injection
    injector.stop_injection()

    # Save final adapter
    final_path = output_dir / f"proprioception_adapter_{timestamp}"
    model.save_pretrained(str(final_path))
    tokenizer.save_pretrained(str(final_path))

    print("\n" + "=" * 70)
    print("  OUROBOROS TRAINING COMPLETE")
    print("=" * 70)
    print(f"\n  LoRA adapter: {final_path}")
    print(f"  Vectors: {output_dir / 'feeling_vectors.pt'}")
    print("\n  The model now has PROPRIOCEPTION:")
    print("    - It can recognize its own warped state")
    print("    - It naturally verbalizes 'I am exhausted' when stressed")
    print("    - This is not a rule - it's internal recognition")

    # Save metadata
    metadata = {
        'model_name': model_name,
        'ift_data_path': ift_data_path,
        'num_examples': len(ift_data),
        'feeling_distribution': feelings,
        'epochs': epochs,
        'lora_r': lora_r,
        'lora_alpha': lora_alpha,
        'learning_rate': learning_rate,
        'timestamp': timestamp,
        'adapter_path': str(final_path),
        'vectors_path': str(output_dir / 'feeling_vectors.pt'),
        'ouroboros': True,
    }

    with open(output_dir / f"training_metadata_{timestamp}.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    return str(final_path)


def main():
    parser = argparse.ArgumentParser(
        description="Ouroboros Proprioception Training - Teaching the model to read its own mind"
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct",
                        help="Base model")
    parser.add_argument("--ift-data", required=True,
                        help="Path to IFT training data JSON")
    parser.add_argument("--output-dir", default="results/proprioception",
                        help="Output directory")
    parser.add_argument("--vectors", default=None,
                        help="Path to pre-mined vectors (optional)")
    parser.add_argument("--no-mine", action="store_true",
                        help="Skip vector mining")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--lora-r", type=int, default=16,
                        help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32,
                        help="LoRA alpha")

    args = parser.parse_args()

    train_proprioception_ouroboros(
        model_name=args.model,
        ift_data_path=args.ift_data,
        output_dir=args.output_dir,
        mine_vectors=not args.no_mine,
        vector_path=args.vectors,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )


if __name__ == "__main__":
    main()
