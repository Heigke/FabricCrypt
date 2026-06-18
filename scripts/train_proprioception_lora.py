#!/usr/bin/env python3
"""
Train Proprioceptive LoRA Adapter

Creates a LoRA adapter that teaches the model to verbalize its somatic state.
This is the "Final Boss" step: Introspective Fine-Tuning (IFT).

The model learns to describe:
- Current feeling (CURIOUS, FOCUSED, DETERMINATION, EXHAUSTED)
- Fatigue level
- Somatic signals (metabolic, thermal, cognitive)

This creates **Proprioception** - the model knowing its own body state.
"""

import json
import argparse
from pathlib import Path
from datetime import datetime
import torch
from datasets import Dataset
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
    TaskType,
    prepare_model_for_kbit_training,
)


def format_ift_example(example: dict) -> str:
    """Format a single IFT example into prompt-completion format."""
    # Build context string
    ctx = example['context']
    context_str = (
        f"Somatic State:\n"
        f"  Metabolic load: {ctx['metabolic']:.2f}\n"
        f"  Thermal load: {ctx['thermal']:.2f}\n"
        f"  Cognitive load: {ctx['cognitive']:.2f}\n"
        f"  Fatigue: {ctx['fatigue']:.2f}\n"
        f"  Current feeling: {ctx['feeling']}\n"
        f"  Compute budget (K): {ctx['k']}"
    )

    # Format as instruction-following
    prompt = (
        f"### Instruction:\n{example['instruction']}\n\n"
        f"### Context:\n{context_str}\n\n"
        f"### Task:\n{example['input']}\n\n"
        f"### Response:\n{example['output']}"
    )

    return prompt


def load_ift_data(data_path: str) -> list[dict]:
    """Load IFT training data from JSON."""
    with open(data_path) as f:
        return json.load(f)


def prepare_dataset(ift_data: list[dict], tokenizer) -> Dataset:
    """Prepare dataset for training."""
    texts = [format_ift_example(ex) for ex in ift_data]

    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=512,
            padding='max_length',
        )

    dataset = Dataset.from_dict({'text': texts})
    tokenized = dataset.map(tokenize_function, batched=True, remove_columns=['text'])

    return tokenized


def train_proprioception_lora(
    model_name: str,
    ift_data_path: str,
    output_dir: str,
    epochs: int = 3,
    batch_size: int = 2,
    learning_rate: float = 2e-4,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
):
    """Train a LoRA adapter for proprioceptive verbalization."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("  PROPRIOCEPTIVE LORA TRAINING")
    print("  Teaching the model to verbalize its somatic state")
    print("=" * 70)
    print(f"\n  Model: {model_name}")
    print(f"  LoRA rank: {lora_r}, alpha: {lora_alpha}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Epochs: {epochs}")

    # Load IFT data
    print("\nLoading IFT training data...")
    ift_data = load_ift_data(ift_data_path)
    print(f"  {len(ift_data)} examples loaded")

    # Analyze data distribution
    feelings = {}
    for ex in ift_data:
        f = ex['feeling_label']
        feelings[f] = feelings.get(f, 0) + 1
    print(f"  Feeling distribution: {feelings}")

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Prepare for LoRA training
    model = prepare_model_for_kbit_training(model)

    # Configure LoRA
    print("\nConfiguring LoRA adapter...")
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Prepare dataset
    print("\nPreparing dataset...")
    dataset = prepare_dataset(ift_data, tokenizer)
    print(f"  Dataset size: {len(dataset)}")

    # Show example
    print("\n  Example training input:")
    sample_text = format_ift_example(ift_data[0])
    print("  " + sample_text[:300].replace("\n", "\n  ") + "...")

    # Training arguments
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"proprioception_lora_{timestamp}"

    training_args = TrainingArguments(
        output_dir=str(output_dir / run_name),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        warmup_steps=10,
        logging_steps=5,
        save_steps=50,
        save_total_limit=2,
        bf16=True,
        optim="adamw_torch",
        report_to="none",  # Disable wandb for simplicity
        remove_unused_columns=False,
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    # Train
    print("\n" + "=" * 70)
    print("  TRAINING PROPRIOCEPTIVE VERBALIZATION")
    print("=" * 70)

    trainer.train()

    # Save final adapter
    final_adapter_path = output_dir / f"proprioception_adapter_{timestamp}"
    model.save_pretrained(str(final_adapter_path))
    tokenizer.save_pretrained(str(final_adapter_path))

    print("\n" + "=" * 70)
    print("  TRAINING COMPLETE")
    print("=" * 70)
    print(f"\n  Adapter saved: {final_adapter_path}")

    # Save training metadata
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
        'adapter_path': str(final_adapter_path),
    }

    with open(output_dir / f"training_metadata_{timestamp}.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    return str(final_adapter_path)


def main():
    parser = argparse.ArgumentParser(description="Train Proprioceptive LoRA Adapter")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct",
                        help="Base model to train on")
    parser.add_argument("--ift-data", required=True,
                        help="Path to IFT training data JSON")
    parser.add_argument("--output-dir", default="results/proprioception",
                        help="Output directory for adapter")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Training batch size")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Learning rate")
    parser.add_argument("--lora-r", type=int, default=16,
                        help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32,
                        help="LoRA alpha")
    args = parser.parse_args()

    adapter_path = train_proprioception_lora(
        model_name=args.model,
        ift_data_path=args.ift_data,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )

    print(f"\n  Use with: --adapter {adapter_path}")


if __name__ == "__main__":
    main()
