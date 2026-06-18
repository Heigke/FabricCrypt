#!/usr/bin/env python3
"""
======================================================================
  z1403b: LATENT DEPTH REASONING - Light Version with Qwen3-4B

  Selective Deep Thinking in Latent Space - Faster iteration
======================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
import json
import os
import sys
from datetime import datetime
from typing import Optional, Tuple, Dict

# Flush prints immediately
import functools
print = functools.partial(print, flush=True)

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

print("=" * 70)
print("  z1403b: LATENT DEPTH REASONING - Light Version")
print("  Selective Deep Thinking with Qwen3-4B")
print("=" * 70)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice: {device}")


class LatentDepthRouter(nn.Module):
    """Router that decides how much depth to use per position."""

    def __init__(self, hidden_dim: int, num_levels: int = 3):
        super().__init__()
        self.router = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, num_levels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns soft depth weights [B, S, num_levels]."""
        return F.softmax(self.router(x), dim=-1)


class RecurrentThinkingBlock(nn.Module):
    """Lightweight recurrent block for iterative reasoning."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        # Simple GRU-style gating
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)
        self.transform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.iter_emb = nn.Embedding(8, hidden_dim)

    def forward(self, h: torch.Tensor, iteration: int) -> torch.Tensor:
        """One iteration of thinking."""
        B, S, D = h.shape

        # Add iteration info
        iter_emb = self.iter_emb(torch.tensor([min(iteration, 7)], device=h.device))
        h_aug = h + iter_emb.unsqueeze(0).expand(B, S, -1) * 0.1

        # Transform
        h_new = self.transform(h_aug)

        # Gated update
        gate = torch.sigmoid(self.gate(torch.cat([h, h_new], dim=-1)))
        return gate * h_new + (1 - gate) * h


class LatentDepthModule(nn.Module):
    """Module that applies variable depth reasoning."""

    def __init__(self, hidden_dim: int, num_levels: int = 3, residual_scale: float = 0.1):
        super().__init__()
        self.num_levels = num_levels
        self.residual_scale = residual_scale

        self.router = LatentDepthRouter(hidden_dim, num_levels)
        self.thinker = RecurrentThinkingBlock(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, hidden: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Apply variable depth reasoning."""
        # Get depth routing
        depth_weights = self.router(hidden)  # [B, S, num_levels]

        # Compute weighted combination of different depths
        outputs = []
        for level in range(self.num_levels):
            h = hidden.clone()
            for i in range(level + 1):
                h = self.thinker(h, i)
            outputs.append(h)

        # Stack and weight
        stacked = torch.stack(outputs, dim=-1)  # [B, S, D, num_levels]
        weighted = (stacked * depth_weights.unsqueeze(2)).sum(dim=-1)  # [B, S, D]

        # Project and add residual (scaled small)
        delta = self.output_proj(weighted - hidden)
        output = hidden + delta * self.residual_scale

        # Stats
        depth_indices = torch.arange(self.num_levels, device=device).float()
        avg_depth = (depth_weights * depth_indices).sum(dim=-1).mean().item()

        return {
            'output': output,
            'avg_depth': avg_depth,
            'depth_weights': depth_weights,
        }


class SelfPredictor(nn.Module):
    """Predicts next module's state from current."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.encoder = nn.Linear(hidden_dim, 256)
        self.decoder = nn.Linear(256, hidden_dim)

    def forward(self, current: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Prediction loss."""
        pred = self.decoder(F.gelu(self.encoder(current)))
        return F.mse_loss(pred, target.detach())


class LatentDepthQwen3(nn.Module):
    """Qwen3-4B with latent depth reasoning modules."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-4B",
        num_levels: int = 3,
        lora_r: int = 16,
        residual_scale: float = 0.1,
        distill_weight: float = 0.3,
    ):
        super().__init__()
        self.distill_weight = distill_weight

        print(f"\nLoading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Main model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

        self.hidden_dim = self.model.config.hidden_size
        self.num_layers = self.model.config.num_hidden_layers

        # Insertion points at 1/3 and 2/3 through
        self.insert_layers = [self.num_layers // 3, 2 * self.num_layers // 3]

        print(f"  Hidden: {self.hidden_dim}, Layers: {self.num_layers}")
        print(f"  Depth modules at layers: {self.insert_layers}")

        # LoRA
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_r * 2,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.model, lora_config)

        # Depth modules
        self.depth_modules = nn.ModuleList([
            LatentDepthModule(self.hidden_dim, num_levels, residual_scale).to(device).to(torch.bfloat16)
            for _ in self.insert_layers
        ])

        # Self predictor
        self.self_predictor = SelfPredictor(self.hidden_dim).to(device).to(torch.bfloat16)

        # Reference model for distillation
        print("Loading frozen reference...")
        self.reference = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        for p in self.reference.parameters():
            p.requires_grad = False
        self.reference.eval()

        # Count params
        lora_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        depth_params = sum(p.numel() for p in self.depth_modules.parameters())
        pred_params = sum(p.numel() for p in self.self_predictor.parameters())

        print(f"\n✓ Model initialized")
        print(f"  LoRA: {lora_params:,}")
        print(f"  Depth modules: {depth_params:,}")
        print(f"  Self predictor: {pred_params:,}")

    def forward_with_depth(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward with latent depth reasoning."""

        # Get hidden states
        outputs = self.model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        all_hidden = outputs.hidden_states

        # Apply depth modules
        depth_outputs = []
        total_depth = 0.0
        for i, layer_idx in enumerate(self.insert_layers):
            h = all_hidden[layer_idx + 1].to(device)
            out = self.depth_modules[i](h)
            depth_outputs.append(out['output'])
            total_depth += out['avg_depth']

        avg_depth = total_depth / len(self.insert_layers)

        # Self prediction loss
        pred_loss = torch.tensor(0.0, device=device)
        if len(depth_outputs) > 1:
            pred_loss = self.self_predictor(
                depth_outputs[0].mean(dim=1),
                depth_outputs[1].mean(dim=1),
            )

        # Compute logits (use original output + small contribution from depth)
        final_h = all_hidden[-1]
        for out in depth_outputs:
            final_h = final_h + out * 0.05
        logits = self.model.lm_head(final_h)

        # LM loss
        lm_loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            lm_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        # Distillation
        distill_loss = torch.tensor(0.0, device=device)
        if labels is not None:
            with torch.no_grad():
                ref_out = self.reference(input_ids=input_ids, attention_mask=attention_mask)
                ref_logits = ref_out.logits
            T = 2.0
            distill_loss = F.kl_div(
                F.log_softmax(logits / T, dim=-1),
                F.softmax(ref_logits / T, dim=-1),
                reduction='batchmean',
            ) * (T * T)

        # Total loss
        total_loss = None
        if lm_loss is not None:
            total_loss = lm_loss + self.distill_weight * distill_loss + 0.01 * pred_loss

        return {
            'loss': total_loss,
            'lm_loss': lm_loss,
            'distill_loss': distill_loss,
            'pred_loss': pred_loss,
            'logits': logits,
            'avg_depth': avg_depth,
        }

    @torch.no_grad()
    def generate_with_depth(self, prompt: str, max_tokens: int = 80) -> Tuple[str, float]:
        """Generate with depth stats."""
        self.eval()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)

        # Generate
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Get depth stats
        result = self.forward_with_depth(inputs.input_ids, inputs.attention_mask)

        return text, result['avg_depth']


class ReasoningDataset(Dataset):
    """Dataset for reasoning prompts."""

    def __init__(self, tokenizer, num_samples: int = 200, max_len: int = 256):
        self.tokenizer = tokenizer
        self.max_len = max_len

        prompts = [
            "Let me work through this step by step. If we have",
            "Consider the equation x² + 2x - 15 = 0. To solve",
            "All cats are mammals. Some mammals can fly. Therefore",
            "When I analyze my own reasoning process, I notice",
            "The relationship between entropy and information",
            "My approach to solving complex problems involves",
            "If it rains, the ground is wet. The ground is wet.",
            "The recursive nature of thinking about thinking",
            "In quantum mechanics, the measurement problem arises",
            "Breaking down my reasoning, I first",
        ]

        self.samples = [prompts[i % len(prompts)] for i in range(num_samples)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.samples[idx],
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'labels': enc['input_ids'].squeeze(0).clone(),
        }


def main():
    """Main training loop."""

    if HAS_WANDB:
        wandb.init(
            project="z1403-latent-depth",
            name="qwen3-4b-light",
            mode="offline",
        )

    # Create model
    model = LatentDepthQwen3(
        model_name="Qwen/Qwen3-4B",
        num_levels=3,
        lora_r=16,
        residual_scale=0.1,
        distill_weight=0.3,
    )

    # Dataset
    print("\nCreating dataset...")
    dataset = ReasoningDataset(model.tokenizer, num_samples=200)
    loader = DataLoader(dataset, batch_size=2, shuffle=True)
    print(f"✓ {len(dataset)} samples")

    # Optimizer
    optimizer = torch.optim.AdamW([
        {'params': model.model.parameters(), 'lr': 1e-4},
        {'params': model.depth_modules.parameters(), 'lr': 5e-4},
        {'params': model.self_predictor.parameters(), 'lr': 5e-4},
    ])

    # Pre-training eval
    print("\n" + "=" * 70)
    print("PRE-TRAINING EVALUATION")
    print("=" * 70)

    test_prompts = [
        "Let me analyze my reasoning step by step:",
        "When I think deeply about this problem,",
        "My approach to understanding complex systems",
    ]

    pre_depths = []
    for p in test_prompts:
        text, depth = model.generate_with_depth(p, max_tokens=60)
        print(f"\nPrompt: {p}")
        print(f"Generated: {text[len(p):len(p)+150]}...")
        print(f"Avg depth: {depth:.3f}")
        pre_depths.append(depth)

    pre_avg = sum(pre_depths) / len(pre_depths)
    print(f"\nPre-training avg depth: {pre_avg:.3f}")

    # Training
    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)

    model.train()
    epochs = 2

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        losses = []

        for step, batch in enumerate(loader):
            optimizer.zero_grad()

            out = model.forward_with_depth(
                batch['input_ids'].to(device),
                batch['attention_mask'].to(device),
                batch['labels'].to(device),
            )

            out['loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            losses.append(out['loss'].item())

            if (step + 1) % 20 == 0:
                avg = sum(losses[-20:]) / len(losses[-20:])
                print(f"  Step {step+1}: loss={avg:.4f}, depth={out['avg_depth']:.2f}")

                if HAS_WANDB:
                    wandb.log({
                        'train/loss': avg,
                        'train/lm_loss': out['lm_loss'].item() if out['lm_loss'] else 0,
                        'train/avg_depth': out['avg_depth'],
                    })

    # Post-training eval
    print("\n" + "=" * 70)
    print("POST-TRAINING EVALUATION")
    print("=" * 70)

    model.eval()

    test_prompts_full = test_prompts + [
        "The recursive nature of self-reflection means",
        "Breaking down my reasoning, I first",
        "My confidence in this analysis comes from",
    ]

    post_depths = []
    samples = []

    for p in test_prompts_full:
        text, depth = model.generate_with_depth(p, max_tokens=80)
        print(f"\nPrompt: {p}")
        print(f"Generated: {text[len(p):len(p)+200]}...")
        print(f"Avg depth: {depth:.3f}")
        post_depths.append(depth)
        samples.append({'prompt': p, 'generated': text[len(p):len(p)+200], 'depth': depth})

    post_avg = sum(post_depths) / len(post_depths)
    depth_var = max(post_depths) - min(post_depths)

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print(f"\nAverage Reasoning Depth:")
    print(f"  Pre:  {pre_avg:.3f}")
    print(f"  Post: {post_avg:.3f}")
    print(f"\nDepth Variation: {depth_var:.3f}")
    print(f"  {'Good' if depth_var > 0.1 else 'Limited'} selective depth allocation")

    # Save
    results = {
        "experiment": "z1403b_latent_depth_light",
        "timestamp": datetime.now().isoformat(),
        "model": "Qwen/Qwen3-4B",
        "pre_avg_depth": pre_avg,
        "post_avg_depth": post_avg,
        "depth_variation": depth_var,
        "samples": samples,
    }

    path = "/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/z1403b_latent_depth_light.json"
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {path}")

    if HAS_WANDB:
        wandb.log({
            'final/pre_avg_depth': pre_avg,
            'final/post_avg_depth': post_avg,
            'final/depth_variation': depth_var,
        })
        wandb.finish()

    return results


if __name__ == "__main__":
    main()
