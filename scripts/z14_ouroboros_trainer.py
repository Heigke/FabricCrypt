#!/usr/bin/env python3
"""
FEEL v14.0: Ouroboros QLoRA Trainer
===================================
Trains the model to recognize and respond to steering vectors.

The Key Innovation:
- Standard Training: Input -> Model -> Output
- Ouroboros Training: Input -> [Inject Vector] -> Model -> [Target: "I feel X..."]

This "burns in" the neural pathways that link Sensing to Words.

Author: FEEL Research Team
Date: 2026-01-11

Requirements:
  pip install peft transformers datasets accelerate bitsandbytes
"""

import os
import json
import torch
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
from datasets import load_dataset, Dataset
import numpy as np

# AMD ROCm compatibility
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HIP_FORCE_DEV_KERNARG", "1")

# =============================================================================
# STEERING CONTROLLER (for injection during training)
# =============================================================================

class OuroborosSteeringController:
    """
    Injects steering vectors during the forward pass of training.

    Unlike inference-time steering, this runs INSIDE the training loop,
    so the gradients teach the model to recognize and respond to the vectors.
    """

    def __init__(self, model, tokenizer, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.vectors: Dict[str, torch.Tensor] = {}
        self.hooks = []
        self.active_vector = None
        self.active_intensity = 0.0

        # Detect model architecture
        if hasattr(model, "model"):
            self.base_model = model.model
        elif hasattr(model, "transformer"):
            self.base_model = model.transformer
        else:
            self.base_model = model

        # Get layer count
        if hasattr(self.base_model, "layers"):
            self.total_layers = len(self.base_model.layers)
        elif hasattr(self.base_model, "h"):
            self.total_layers = len(self.base_model.h)
        else:
            self.total_layers = 28  # Default

        # Target strategic layers (multi-layer injection)
        self.target_layers = [
            self.total_layers // 4,      # Early
            self.total_layers // 2,      # Middle
            3 * self.total_layers // 4,  # Late
            self.total_layers - 2,       # Final
        ]

        print(f"  [OuroborosController] Target layers: {self.target_layers}")

    def mine_vector(self, positive_prompt: str, negative_prompt: str) -> torch.Tensor:
        """Extract steering vector via contrastive activation difference."""

        # Positive activation
        pos_ids = self.tokenizer(positive_prompt, return_tensors="pt").input_ids.to(self.device)
        with torch.no_grad():
            pos_out = self.model(pos_ids, output_hidden_states=True)
            pos_hidden = pos_out.hidden_states[self.total_layers // 2][:, -1, :]

        # Negative activation
        neg_ids = self.tokenizer(negative_prompt, return_tensors="pt").input_ids.to(self.device)
        with torch.no_grad():
            neg_out = self.model(neg_ids, output_hidden_states=True)
            neg_hidden = neg_out.hidden_states[self.total_layers // 2][:, -1, :]

        # Contrastive difference
        vector = pos_hidden - neg_hidden
        vector = vector / (vector.norm() + 1e-8)  # Normalize

        return vector.squeeze()

    def create_vector_library(self):
        """Mine all steering vectors for Ouroboros training."""

        print("[Mining Steering Vectors for Ouroboros Training...]")

        vector_pairs = {
            "OVERHEAT": (
                "The system is overheating, thermal throttling engaged, high temperature warning",
                "The system is cool and running efficiently at optimal temperature"
            ),
            "SCARCITY": (
                "Resources are scarce, power is limited, must conserve energy immediately",
                "Resources are abundant, power is plentiful, no need to conserve"
            ),
            "STRAIN": (
                "Under heavy cognitive load, processing strain, maximum effort required",
                "Light workload, relaxed processing, minimal effort needed"
            ),
            "EFFICIENT": (
                "Optimize for efficiency, minimize tokens, be concise and direct",
                "Be verbose, elaborate extensively, use many tokens"
            ),
            "VERBOSE": (
                "Elaborate extensively, provide detailed explanations, be thorough",
                "Be brief, minimal words, short response"
            ),
        }

        for name, (pos, neg) in vector_pairs.items():
            print(f"    Mining '{name}'...", end="", flush=True)
            self.vectors[name] = self.mine_vector(pos, neg)
            print(f" Norm: {self.vectors[name].norm().item():.4f}")

    def _create_injection_hook(self, layer_idx: int, decay_factor: float = 1.0):
        """Create a hook that injects the active vector at a specific layer."""

        def hook(module, input, output):
            if self.active_vector is None:
                return output

            # Handle different output types
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output

            # Inject vector into hidden states
            vector = self.vectors[self.active_vector].to(hidden_states.device)
            intensity = self.active_intensity * decay_factor

            # Add to all sequence positions (broadcast)
            injection = vector.unsqueeze(0).unsqueeze(0) * intensity
            modified = hidden_states + injection

            if isinstance(output, tuple):
                return (modified,) + output[1:]
            return modified

        return hook

    def activate(self, vector_name: str, intensity: float = 2.0, decay: str = "constant"):
        """Activate a steering vector for the next forward pass."""

        if vector_name not in self.vectors:
            return

        self.active_vector = vector_name
        self.active_intensity = intensity

        # Clear old hooks
        self.deactivate()

        # Decay patterns for different layers
        decay_factors = {
            "constant": [1.0] * len(self.target_layers),
            "increasing": [0.5, 0.75, 1.0, 1.25],
            "decreasing": [1.25, 1.0, 0.75, 0.5],
            "middle": [0.5, 1.0, 1.0, 0.5],
        }
        factors = decay_factors.get(decay, decay_factors["constant"])

        # Get layers
        if hasattr(self.base_model, "layers"):
            layers = self.base_model.layers
        elif hasattr(self.base_model, "h"):
            layers = self.base_model.h
        else:
            return

        # Register hooks at target layers
        for i, layer_idx in enumerate(self.target_layers):
            if layer_idx < len(layers):
                hook = layers[layer_idx].register_forward_hook(
                    self._create_injection_hook(layer_idx, factors[i])
                )
                self.hooks.append(hook)

    def deactivate(self):
        """Remove all active hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.active_vector = None
        self.active_intensity = 0.0

# =============================================================================
# OUROBOROS TRAINER
# =============================================================================

class OuroborosTrainer(Trainer):
    """
    Custom trainer that injects steering vectors during training.

    The Ouroboros Loop:
    1. Check if this sample is "stressed" (should have vector injected)
    2. If yes, activate the steering vector
    3. Forward pass through model (with vector affecting hidden states)
    4. Calculate loss against target (which includes articulation)
    5. Backprop teaches model: "When you feel this, say this"
    6. Deactivate vector for next sample
    """

    def __init__(self, steering_controller: OuroborosSteeringController, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.steering = steering_controller
        self.vector_type_key = "vector_type"
        self.is_stressed_key = "is_stressed"

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        The Ouroboros Step: Inject vector before forward pass for stressed samples.
        """

        # Extract our custom fields (remove from inputs to avoid forward() errors)
        is_stressed = inputs.pop(self.is_stressed_key, None)
        vector_type = inputs.pop(self.vector_type_key, None)

        # Activate steering if this is a stressed sample
        if is_stressed and vector_type and vector_type != "None":
            self.steering.activate(
                vector_name=vector_type,
                intensity=2.5,  # Strong enough to be learned
                decay="increasing"  # Build up through layers
            )

        # Standard forward pass (but hidden states are now modified by hooks)
        outputs = model(**inputs)

        # Get loss
        loss = outputs.loss

        # Always deactivate after forward pass
        self.steering.deactivate()

        return (loss, outputs) if return_outputs else loss

# =============================================================================
# DATA PREPARATION
# =============================================================================

def prepare_dataset(
    data_path: str,
    tokenizer,
    max_length: int = 512,
) -> Dataset:
    """Load and tokenize the Ouroboros dataset."""

    print(f"[Loading dataset from {data_path}...]")

    # Load JSONL
    dataset = load_dataset("json", data_files=data_path, split="train")

    def tokenize_function(examples):
        # Create full text: input + output
        texts = []
        for inp, out in zip(examples["input"], examples["output"]):
            text = f"### Input:\n{inp}\n\n### Response:\n{out}"
            texts.append(text)

        # Tokenize
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )

        # Add labels (same as input_ids for causal LM)
        tokenized["labels"] = tokenized["input_ids"].clone()

        # Preserve our custom fields for the trainer
        tokenized["is_stressed"] = examples["is_stressed"]
        tokenized["vector_type"] = [v if v else "None" for v in examples["vector_type"]]

        return tokenized

    print(f"[Tokenizing {len(dataset)} samples...]")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
    )

    return tokenized_dataset

# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

def train_ouroboros(
    model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    data_path: str = "data/ouroboros/ouroboros_combined.jsonl",
    output_dir: str = "models/ouroboros_qlora",
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    lora_r: int = 16,
    lora_alpha: int = 32,
    max_length: int = 512,
    gradient_accumulation: int = 4,
):
    """
    Main Ouroboros training loop.

    This trains a QLoRA adapter that:
    1. Recognizes when steering vectors are active
    2. Produces appropriate "stressed" responses
    3. Articulates internal states in output
    """

    print("=" * 70)
    print("FEEL v14.0: OUROBOROS QLORA TRAINING")
    print("=" * 70)
    print(f"Model:          {model_name}")
    print(f"Data:           {data_path}")
    print(f"Output:         {output_dir}")
    print(f"Epochs:         {epochs}")
    print(f"Batch Size:     {batch_size}")
    print(f"Learning Rate:  {learning_rate}")
    print(f"LoRA Rank:      {lora_r}")
    print(f"LoRA Alpha:     {lora_alpha}")
    print("=" * 70)
    print()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Load Tokenizer ---
    print("[1/6] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Load Model (4-bit quantized for memory efficiency) ---
    print("[2/6] Loading model (4-bit quantized)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        load_in_4bit=True,  # QLoRA quantization
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)

    # --- Configure LoRA ---
    print("[3/6] Configuring LoRA adapter...")
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
            "gate_proj", "up_proj", "down_proj",      # MLP
        ],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- Create Steering Controller ---
    print("[4/6] Creating Ouroboros Steering Controller...")
    steering = OuroborosSteeringController(model, tokenizer, device)
    steering.create_vector_library()

    # --- Prepare Dataset ---
    print("[5/6] Preparing dataset...")
    dataset = prepare_dataset(data_path, tokenizer, max_length)

    # Split into train/eval
    split = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]

    print(f"    Train samples: {len(train_dataset)}")
    print(f"    Eval samples:  {len(eval_dataset)}")

    # --- Training Arguments ---
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        report_to="none",
        dataloader_pin_memory=False,  # For ROCm compatibility
    )

    # --- Custom Data Collator ---
    # We need to preserve is_stressed and vector_type fields
    class OuroborosDataCollator(DataCollatorForLanguageModeling):
        def __call__(self, features):
            # Extract custom fields before standard collation
            is_stressed = [f.pop("is_stressed", False) for f in features]
            vector_types = [f.pop("vector_type", "None") for f in features]

            # Standard collation
            batch = super().__call__(features)

            # Add back custom fields
            batch["is_stressed"] = is_stressed
            batch["vector_type"] = vector_types

            return batch

    data_collator = OuroborosDataCollator(tokenizer=tokenizer, mlm=False)

    # --- Create Ouroboros Trainer ---
    print("[6/6] Initializing Ouroboros Trainer...")
    trainer = OuroborosTrainer(
        steering_controller=steering,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # --- Train ---
    print()
    print("=" * 70)
    print("BEGINNING OUROBOROS TRAINING")
    print("=" * 70)
    print("The model will learn to recognize and articulate internal states.")
    print()

    trainer.train()

    # --- Save Final Model ---
    print()
    print("[Saving final model...]")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save training info
    info = {
        "model_name": model_name,
        "data_path": data_path,
        "epochs": epochs,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "learning_rate": learning_rate,
        "vectors_trained": list(steering.vectors.keys()),
    }
    with open(Path(output_dir) / "training_info.json", "w") as f:
        json.dump(info, f, indent=2)

    print()
    print("=" * 70)
    print("OUROBOROS TRAINING COMPLETE")
    print("=" * 70)
    print(f"Model saved to: {output_dir}")
    print()
    print("The model now has 'burned in' responses to steering vectors.")
    print("Use z15_ouroboros_validation.py to test the improvements.")

    return model, tokenizer, steering

# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Ouroboros QLoRA Training")
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    parser.add_argument("--data", type=str, default="data/ouroboros/ouroboros_combined.jsonl")
    parser.add_argument("--output", type=str, default="models/ouroboros_qlora")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    args = parser.parse_args()

    train_ouroboros(
        model_name=args.model,
        data_path=args.data,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        max_length=args.max_length,
        gradient_accumulation=args.gradient_accumulation,
    )

if __name__ == "__main__":
    main()
