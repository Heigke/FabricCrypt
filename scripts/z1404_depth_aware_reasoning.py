#!/usr/bin/env python3
"""
======================================================================
  z1404: DEPTH-AWARE LATENT REASONING

  Enhancement over z1403: Model is explicitly AWARE of its depth allocation
  - Depth embeddings injected into hidden states
  - Meta-cognitive head predicts own depth allocation
  - Model learns to reflect on its reasoning depth
======================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
import json
import functools
from datetime import datetime
from typing import Optional, Tuple, Dict

print = functools.partial(print, flush=True)

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

print("=" * 70)
print("  z1404: DEPTH-AWARE LATENT REASONING")
print("  Model Explicitly Aware of Its Reasoning Depth")
print("=" * 70)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice: {device}")


class DepthAwareRouter(nn.Module):
    """
    Router that outputs depth allocation AND provides depth embedding
    for the model to be aware of its depth state.
    """

    def __init__(self, hidden_dim: int, num_levels: int = 4):
        super().__init__()
        self.num_levels = num_levels
        self.hidden_dim = hidden_dim

        # Router network
        self.router = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, num_levels),
        )

        # Depth embeddings that get added to hidden states
        # This makes the model AWARE of what depth it's at
        self.depth_embeddings = nn.Parameter(torch.randn(num_levels, hidden_dim) * 0.02)

    def forward(self, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            depth_weights: [B, S, num_levels] soft depth allocation
            depth_embedding: [B, S, H] weighted depth embedding for awareness
        """
        # Compute soft depth allocation
        logits = self.router(hidden)
        depth_weights = F.softmax(logits, dim=-1)  # [B, S, num_levels]

        # Compute weighted depth embedding - this is what gets added to hidden
        # states so the model "knows" what depth it's operating at
        # depth_weights: [B, S, L], depth_embeddings: [L, H]
        depth_embedding = torch.einsum('bsl,lh->bsh', depth_weights, self.depth_embeddings)

        return depth_weights, depth_embedding


class DepthAwareThinkingBlock(nn.Module):
    """
    Thinking block that receives depth information at each iteration.
    The model explicitly knows what iteration/depth it's at.
    """

    def __init__(self, hidden_dim: int, num_levels: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Level-specific processing (different weights per depth level)
        self.level_transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in range(num_levels)
        ])

        # Gating mechanism
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)

        # Depth-aware iteration embedding
        self.iter_depth_emb = nn.Embedding(num_levels * 8, hidden_dim)  # iter * level

    def forward(
        self,
        hidden: torch.Tensor,
        iteration: int,
        level: int,
        depth_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """
        Process at specific depth level with awareness.

        Args:
            hidden: [B, S, H] hidden states
            iteration: iteration number (0, 1, 2, ...)
            level: depth level (0, 1, 2, 3)
            depth_embedding: [B, S, H] depth awareness embedding
        """
        B, S, H = hidden.shape

        # Add depth awareness to hidden state
        h_aware = hidden + depth_embedding * 0.1

        # Add iteration+level embedding
        iter_level_idx = min(iteration * 4 + level, 31)
        iter_emb = self.iter_depth_emb(torch.tensor([iter_level_idx], device=hidden.device))
        h_aware = h_aware + iter_emb.unsqueeze(0).expand(B, S, -1) * 0.1

        # Level-specific transform
        h_new = self.level_transforms[level](h_aware)

        # Gated update
        gate = torch.sigmoid(self.gate(torch.cat([hidden, h_new], dim=-1)))
        return gate * h_new + (1 - gate) * hidden


class MetaCognitiveHead(nn.Module):
    """
    Head that predicts the model's own depth allocation.
    This trains the model to be explicitly introspective about its depth.
    """

    def __init__(self, hidden_dim: int, num_levels: int = 4):
        super().__init__()
        self.num_levels = num_levels

        # Predict depth allocation from hidden state
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, num_levels),
        )

    def forward(
        self,
        hidden: torch.Tensor,
        target_depth_weights: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Predict and compare to actual depth allocation.

        Args:
            hidden: [B, S, H] hidden states
            target_depth_weights: [B, S, num_levels] actual depth allocation

        Returns:
            meta_loss: loss for depth prediction
            predicted_depth: [B, S, num_levels] predicted allocation
        """
        # Predict depth allocation
        pred_logits = self.predictor(hidden)  # [B, S, num_levels]
        pred_depth = F.softmax(pred_logits, dim=-1)

        # Cross-entropy loss against actual depth allocation
        # Using soft labels (actual depth weights)
        meta_loss = F.cross_entropy(
            pred_logits.view(-1, self.num_levels),
            target_depth_weights.view(-1, self.num_levels).argmax(dim=-1),
            reduction='mean'
        )

        return {
            'meta_loss': meta_loss,
            'predicted_depth': pred_depth,
        }


class DepthAwareModule(nn.Module):
    """
    Complete depth-aware reasoning module.
    The model knows and can reflect on its depth allocation.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_levels: int = 4,
        residual_scale: float = 0.1,
    ):
        super().__init__()
        self.num_levels = num_levels
        self.residual_scale = residual_scale

        # Depth-aware router
        self.router = DepthAwareRouter(hidden_dim, num_levels)

        # Depth-aware thinking
        self.thinker = DepthAwareThinkingBlock(hidden_dim, num_levels)

        # Meta-cognitive head
        self.meta_head = MetaCognitiveHead(hidden_dim, num_levels)

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, hidden: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Apply depth-aware reasoning with meta-cognition."""

        # Get depth allocation and awareness embedding
        depth_weights, depth_embedding = self.router(hidden)

        # Process at each depth level
        level_outputs = []
        for level in range(self.num_levels):
            h = hidden.clone()
            # Iterate at this depth level
            for iteration in range(level + 1):
                h = self.thinker(h, iteration, level, depth_embedding)
            level_outputs.append(h)

        # Stack and weight by depth allocation
        stacked = torch.stack(level_outputs, dim=-1)  # [B, S, H, num_levels]
        weighted = (stacked * depth_weights.unsqueeze(2)).sum(dim=-1)  # [B, S, H]

        # Meta-cognitive prediction (can the model predict its own depth?)
        meta_out = self.meta_head(weighted, depth_weights)

        # Output with residual
        delta = self.output_proj(weighted - hidden)
        output = hidden + delta * self.residual_scale

        # Compute depth statistics
        depth_indices = torch.arange(self.num_levels, device=device).float()
        per_position_depth = (depth_weights * depth_indices).sum(dim=-1)  # [B, S]
        avg_depth = per_position_depth.mean().item()
        depth_std = per_position_depth.std().item()

        return {
            'output': output,
            'depth_weights': depth_weights,
            'depth_embedding': depth_embedding,
            'meta_loss': meta_out['meta_loss'],
            'predicted_depth': meta_out['predicted_depth'],
            'avg_depth': avg_depth,
            'depth_std': depth_std,
            'per_position_depth': per_position_depth,
        }


class DepthAwareQwen3(nn.Module):
    """Qwen3 with depth-aware latent reasoning."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-4B",
        num_levels: int = 4,
        lora_r: int = 16,
        residual_scale: float = 0.1,
        distill_weight: float = 0.3,
        meta_weight: float = 0.1,
    ):
        super().__init__()
        self.distill_weight = distill_weight
        self.meta_weight = meta_weight

        print(f"\nLoading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

        self.hidden_dim = self.model.config.hidden_size
        self.num_layers = self.model.config.num_hidden_layers
        self.insert_layers = [self.num_layers // 3, 2 * self.num_layers // 3]

        print(f"  Hidden: {self.hidden_dim}, Layers: {self.num_layers}")
        print(f"  Depth-aware modules at layers: {self.insert_layers}")

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

        # Depth-aware modules
        self.depth_modules = nn.ModuleList([
            DepthAwareModule(self.hidden_dim, num_levels, residual_scale).to(device).to(torch.bfloat16)
            for _ in self.insert_layers
        ])

        # Cross-module predictor
        self.cross_predictor = nn.Sequential(
            nn.Linear(self.hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, self.hidden_dim),
        ).to(device).to(torch.bfloat16)

        # Reference for distillation
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
        lora_p = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        depth_p = sum(p.numel() for p in self.depth_modules.parameters())
        cross_p = sum(p.numel() for p in self.cross_predictor.parameters())

        print(f"\n✓ Model initialized")
        print(f"  LoRA: {lora_p:,}")
        print(f"  Depth-aware modules: {depth_p:,}")
        print(f"  Cross predictor: {cross_p:,}")

    def forward_with_depth_awareness(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward with depth-aware reasoning and meta-cognition."""

        outputs = self.model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        all_hidden = outputs.hidden_states

        # Apply depth-aware modules
        depth_outputs = []
        total_meta_loss = 0.0
        depth_stats = []

        for i, layer_idx in enumerate(self.insert_layers):
            h = all_hidden[layer_idx + 1].to(device)
            out = self.depth_modules[i](h)
            depth_outputs.append(out['output'])
            total_meta_loss += out['meta_loss']
            depth_stats.append({
                'avg_depth': out['avg_depth'],
                'depth_std': out['depth_std'],
            })

        avg_depth = sum(s['avg_depth'] for s in depth_stats) / len(depth_stats)

        # Cross-module prediction loss
        cross_loss = torch.tensor(0.0, device=device)
        if len(depth_outputs) > 1:
            pred = self.cross_predictor(depth_outputs[0].mean(dim=1))
            target = depth_outputs[1].mean(dim=1)
            cross_loss = F.mse_loss(pred, target.detach())

        # Compute logits
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
            total_loss = (
                lm_loss +
                self.distill_weight * distill_loss +
                self.meta_weight * total_meta_loss +
                0.01 * cross_loss
            )

        return {
            'loss': total_loss,
            'lm_loss': lm_loss,
            'distill_loss': distill_loss,
            'meta_loss': total_meta_loss,
            'cross_loss': cross_loss,
            'logits': logits,
            'avg_depth': avg_depth,
            'depth_stats': depth_stats,
        }

    @torch.no_grad()
    def generate_with_depth_awareness(
        self,
        prompt: str,
        max_tokens: int = 80
    ) -> Tuple[str, Dict]:
        """Generate with depth awareness stats."""
        self.eval()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        result = self.forward_with_depth_awareness(inputs.input_ids, inputs.attention_mask)

        stats = {
            'avg_depth': result['avg_depth'],
            'depth_stats': result['depth_stats'],
        }

        return text, stats


class ReasoningDataset(Dataset):
    """Dataset for depth-aware reasoning."""

    def __init__(self, tokenizer, num_samples: int = 200, max_len: int = 256):
        self.tokenizer = tokenizer
        self.max_len = max_len

        # Mix of prompts that should require different depths
        easy_prompts = [
            "The capital of France is",
            "2 + 2 equals",
            "Water boils at",
        ]
        medium_prompts = [
            "Let me work through this step by step. If we have",
            "When I analyze my reasoning process, I notice",
            "My approach to solving this problem involves",
        ]
        hard_prompts = [
            "The recursive nature of self-reflection means that",
            "Consider the philosophical implications of a machine that",
            "The relationship between consciousness and computation",
        ]

        self.samples = []
        for i in range(num_samples):
            if i % 3 == 0:
                self.samples.append(easy_prompts[i % len(easy_prompts)])
            elif i % 3 == 1:
                self.samples.append(medium_prompts[i % len(medium_prompts)])
            else:
                self.samples.append(hard_prompts[i % len(hard_prompts)])

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
            project="z1404-depth-aware",
            name="qwen3-4b-depth-aware",
            mode="offline",
        )

    model = DepthAwareQwen3(
        model_name="Qwen/Qwen3-4B",
        num_levels=4,
        lora_r=16,
        residual_scale=0.1,
        distill_weight=0.3,
        meta_weight=0.1,
    )

    print("\nCreating dataset...")
    dataset = ReasoningDataset(model.tokenizer, num_samples=200)
    loader = DataLoader(dataset, batch_size=2, shuffle=True)
    print(f"✓ {len(dataset)} samples")

    optimizer = torch.optim.AdamW([
        {'params': model.model.parameters(), 'lr': 1e-4},
        {'params': model.depth_modules.parameters(), 'lr': 5e-4},
        {'params': model.cross_predictor.parameters(), 'lr': 5e-4},
    ])

    # Pre-training eval
    print("\n" + "=" * 70)
    print("PRE-TRAINING EVALUATION")
    print("=" * 70)

    test_prompts = [
        ("easy", "The capital of France is"),
        ("medium", "Let me analyze my reasoning step by step:"),
        ("hard", "The recursive nature of self-reflection means"),
    ]

    pre_results = []
    for difficulty, p in test_prompts:
        text, stats = model.generate_with_depth_awareness(p, max_tokens=60)
        print(f"\n[{difficulty.upper()}] Prompt: {p}")
        print(f"Generated: {text[len(p):len(p)+150]}...")
        print(f"Avg depth: {stats['avg_depth']:.3f}")
        pre_results.append({'difficulty': difficulty, 'depth': stats['avg_depth']})

    pre_avg = sum(r['depth'] for r in pre_results) / len(pre_results)
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

            out = model.forward_with_depth_awareness(
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
                print(f"  Step {step+1}: loss={avg:.4f}, depth={out['avg_depth']:.2f}, meta_loss={out['meta_loss'].item():.4f}")

                if HAS_WANDB:
                    wandb.log({
                        'train/loss': avg,
                        'train/avg_depth': out['avg_depth'],
                        'train/meta_loss': out['meta_loss'].item(),
                    })

    # Post-training eval
    print("\n" + "=" * 70)
    print("POST-TRAINING EVALUATION")
    print("=" * 70)

    model.eval()

    test_prompts_full = [
        ("easy", "The capital of France is"),
        ("easy", "2 + 2 equals"),
        ("medium", "Let me analyze my reasoning step by step:"),
        ("medium", "When I think about this problem,"),
        ("hard", "The recursive nature of self-reflection means"),
        ("hard", "The relationship between consciousness and computation"),
    ]

    post_results = []
    samples = []

    for difficulty, p in test_prompts_full:
        text, stats = model.generate_with_depth_awareness(p, max_tokens=80)
        print(f"\n[{difficulty.upper()}] Prompt: {p}")
        print(f"Generated: {text[len(p):len(p)+200]}...")
        print(f"Avg depth: {stats['avg_depth']:.3f}")
        post_results.append({'difficulty': difficulty, 'depth': stats['avg_depth']})
        samples.append({
            'difficulty': difficulty,
            'prompt': p,
            'generated': text[len(p):len(p)+200],
            'depth': stats['avg_depth'],
        })

    # Compute depth by difficulty
    easy_depth = sum(r['depth'] for r in post_results if r['difficulty'] == 'easy') / 2
    medium_depth = sum(r['depth'] for r in post_results if r['difficulty'] == 'medium') / 2
    hard_depth = sum(r['depth'] for r in post_results if r['difficulty'] == 'hard') / 2

    post_avg = sum(r['depth'] for r in post_results) / len(post_results)
    depth_spread = hard_depth - easy_depth

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print(f"\nAverage Depth by Difficulty:")
    print(f"  Easy:   {easy_depth:.3f}")
    print(f"  Medium: {medium_depth:.3f}")
    print(f"  Hard:   {hard_depth:.3f}")
    print(f"\nDepth Spread (hard - easy): {depth_spread:.3f}")
    print(f"  {'Good' if depth_spread > 0.1 else 'Limited'} difficulty-aware depth allocation")

    print(f"\nOverall:")
    print(f"  Pre-training avg: {pre_avg:.3f}")
    print(f"  Post-training avg: {post_avg:.3f}")

    # Save
    results = {
        "experiment": "z1404_depth_aware_reasoning",
        "timestamp": datetime.now().isoformat(),
        "model": "Qwen/Qwen3-4B",
        "pre_avg_depth": pre_avg,
        "post_avg_depth": post_avg,
        "depth_by_difficulty": {
            "easy": easy_depth,
            "medium": medium_depth,
            "hard": hard_depth,
        },
        "depth_spread": depth_spread,
        "samples": samples,
    }

    path = "/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/z1404_depth_aware_reasoning.json"
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {path}")

    if HAS_WANDB:
        wandb.log({
            'final/pre_avg_depth': pre_avg,
            'final/post_avg_depth': post_avg,
            'final/easy_depth': easy_depth,
            'final/medium_depth': medium_depth,
            'final/hard_depth': hard_depth,
            'final/depth_spread': depth_spread,
        })
        wandb.finish()

    return results


if __name__ == "__main__":
    main()
