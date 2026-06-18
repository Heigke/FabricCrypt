#!/usr/bin/env python3
"""
Dynamic Attention Span Training - Deeper embodied compute control.

Combines:
1. Early Exit (layer-level control) - from z151
2. Dynamic Attention Span (attention-level control) - NEW

The model learns to:
- Exit early when confident (saves whole layers)
- Use shorter attention spans when local context suffices (saves O(n²) attention)
- Respond to simulated body state (thermal/power constraints)
"""

import os
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from src.modeling.dynamic_attention import SpanConfig, SpanPredictor


class ExitHead(nn.Module):
    """Exit head for early exit."""
    def __init__(self, hidden_dim, vocab_size):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.out = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, x):
        return self.out(self.norm(F.gelu(self.proj(x)) + x))


class EmbodiedGPT2(nn.Module):
    """
    GPT-2 with embodied compute control:
    - Early exit at layers [3, 6, 9, 12]
    - Dynamic attention span [32, 64, 128, 256]
    - Body state awareness
    """

    def __init__(self, model_name="gpt2", sensor_dim=32):
        super().__init__()

        # Load base model
        self.base = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)

        # Freeze base
        for p in self.base.parameters():
            p.requires_grad = False

        # Config
        self.hidden_dim = self.base.config.hidden_size
        self.vocab_size = self.base.config.vocab_size
        self.num_layers = self.base.config.num_hidden_layers
        self.sensor_dim = sensor_dim

        # Exit layers
        self.exit_layer_indices = [
            self.num_layers // 4,
            self.num_layers // 2,
            3 * self.num_layers // 4,
            self.num_layers
        ]

        # Exit heads
        self.exit_heads = nn.ModuleList([
            ExitHead(self.hidden_dim, self.vocab_size)
            for _ in self.exit_layer_indices
        ])

        # Exit uncertainty (from z151_v2)
        self.exit_uncertainty = nn.Sequential(
            nn.Linear(self.hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # Dynamic span predictor (num_layers = number of exit points, not transformer layers)
        self.span_config = SpanConfig(
            hidden_dim=self.hidden_dim,
            sensor_dim=sensor_dim,
            num_layers=len(self.exit_layer_indices),  # 4 exit points
            num_spans=4,
            min_span=32,
            max_span=256,
            per_layer_span=True
        )
        self.span_predictor = SpanPredictor(self.span_config)

        # Body state encoder (simulated for now)
        self.body_encoder = nn.Sequential(
            nn.Linear(sensor_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )

    def simulate_body_state(self, batch_size, device, stress_level=0.5):
        """
        Simulate body state for training.

        stress_level: 0 = relaxed (can use full compute), 1 = stressed (minimize compute)
        """
        # Generate semi-random body state
        base_state = torch.randn(batch_size, self.sensor_dim, device=device) * 0.3

        # Add stress component
        stress = torch.ones(batch_size, 1, device=device) * stress_level
        base_state[:, 0:1] = 1.0 - stress  # First dim is "thermal headroom"
        base_state[:, 1:2] = 1.0 - stress  # Second dim is "power headroom"

        return torch.sigmoid(base_state)  # Normalize to [0, 1]

    def forward(self, input_ids, labels=None, body_state=None, return_all=False):
        """
        Forward pass with embodied control.
        """
        batch_size = input_ids.size(0)
        device = input_ids.device

        # Simulate body state if not provided
        if body_state is None:
            # Random stress level during training
            stress = torch.rand(1).item() if self.training else 0.3
            body_state = self.simulate_body_state(batch_size, device, stress)

        # Get all hidden states from base model
        with torch.no_grad():
            out = self.base(input_ids, output_hidden_states=True)

        hidden_states = out.hidden_states
        final_logits = out.logits

        # Compute exit outputs and control signals
        exit_outputs = []
        exit_uncertainties = []
        span_predictions = []

        for i, layer_idx in enumerate(self.exit_layer_indices):
            h = hidden_states[layer_idx]

            # Exit head output
            logits = self.exit_heads[i](h)
            exit_outputs.append(logits)

            # Exit uncertainty
            u = self.exit_uncertainty(h[:, -1, :])
            exit_uncertainties.append(u)

            # Span prediction for this layer (use exit index, not layer index)
            span_weights, selected_span = self.span_predictor(
                h, body_state, i  # i is the exit index (0, 1, 2, 3)
            )
            span_predictions.append({
                'weights': span_weights,
                'selected': selected_span,
                'value': self.span_predictor.get_span_value(selected_span)
            })

        # Training loss
        if labels is not None:
            loss_dict = self._compute_loss(
                exit_outputs, exit_uncertainties, span_predictions,
                final_logits, labels, body_state
            )
            return loss_dict

        # Inference: return predictions
        if return_all:
            return {
                'exit_outputs': exit_outputs,
                'exit_uncertainties': exit_uncertainties,
                'span_predictions': span_predictions,
                'final_logits': final_logits
            }

        # Simple inference: use best exit
        u_values = [u.mean().item() for u in exit_uncertainties]
        best_idx = np.argmin(u_values)
        return exit_outputs[best_idx], self.exit_layer_indices[best_idx]

    def _compute_loss(
        self,
        exit_outputs,
        exit_uncertainties,
        span_predictions,
        final_logits,
        labels,
        body_state
    ):
        """Compute combined loss for embodied control."""
        device = labels.device
        total_loss = torch.tensor(0.0, device=device)

        # 1. CE loss for exit heads (from z151)
        shift_labels = labels[..., 1:].contiguous()
        ce_losses = []

        for i, logits in enumerate(exit_outputs):
            shift_logits = logits[..., :-1, :].contiguous()
            ce = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100
            )
            ce_losses.append(ce.item())
            w = (i + 1) / len(self.exit_layer_indices)
            total_loss = total_loss + w * ce

        total_loss = total_loss / len(exit_outputs)

        # 2. KL loss to final layer
        final_shift = final_logits[..., :-1, :].contiguous()
        final_probs = F.softmax(final_shift.detach(), dim=-1)

        kl_loss = torch.tensor(0.0, device=device)
        for logits in exit_outputs:
            shift_logits = logits[..., :-1, :].contiguous()
            log_probs = F.log_softmax(shift_logits, dim=-1)
            kl = F.kl_div(log_probs, final_probs, reduction='batchmean')
            kl_loss = kl_loss + kl
        kl_loss = kl_loss / len(exit_outputs)
        total_loss = total_loss + 0.1 * kl_loss

        # 3. Exit calibration loss (uncertainty should match disagreement)
        final_pred = final_logits[:, -1, :].argmax(dim=-1)
        uncert_loss = torch.tensor(0.0, device=device)

        for i, (logits, u) in enumerate(zip(exit_outputs, exit_uncertainties)):
            exit_pred = logits[:, -1, :].argmax(dim=-1)
            target = (exit_pred != final_pred).float().unsqueeze(-1)
            uncert_loss = uncert_loss + F.binary_cross_entropy(u, target)

        uncert_loss = uncert_loss / len(exit_outputs)
        total_loss = total_loss + 0.3 * uncert_loss

        # 4. Exit incentive (reward early exits when confident)
        exit_incentive = torch.tensor(0.0, device=device)
        for i, u in enumerate(exit_uncertainties):
            layer_pos = (i + 1) / len(self.exit_layer_indices)
            reward = (1 - u.mean()) * (1 - layer_pos)
            exit_incentive = exit_incentive + reward
        total_loss = total_loss - 0.1 * exit_incentive

        # 5. Span efficiency loss (prefer smaller spans)
        span_loss = torch.tensor(0.0, device=device)
        avg_span_ratio = 0.0

        for pred in span_predictions:
            weights = pred['weights']  # [batch, num_spans]
            # Normalized span values
            spans = torch.tensor(
                self.span_predictor.span_choices,
                device=device, dtype=torch.float
            )
            max_span = spans.max()
            normalized = spans / max_span

            # Weighted average span ratio
            ratio = (weights * normalized).sum(dim=-1).mean()
            avg_span_ratio += ratio.item()

            # Loss: encourage smaller spans (target 0.4 = 40% of max)
            span_loss = span_loss + (ratio - 0.4).abs()

        span_loss = span_loss / len(span_predictions)
        avg_span_ratio = avg_span_ratio / len(span_predictions)
        total_loss = total_loss + 0.05 * span_loss

        # 6. Body alignment loss (span should decrease under stress)
        body_loss = torch.tensor(0.0, device=device)

        if body_state is not None:
            # thermal_headroom is in body_state[:, 0]
            headroom = body_state[:, 0].mean()

            # Expected span ratio: 0.3 when stressed (headroom=0), 0.7 when relaxed
            expected = 0.3 + 0.4 * headroom

            for pred in span_predictions:
                weights = pred['weights']
                spans = torch.tensor(
                    self.span_predictor.span_choices,
                    device=device, dtype=torch.float
                )
                ratio = (weights * (spans / spans.max())).sum(dim=-1).mean()
                body_loss = body_loss + (ratio - expected).abs()

            body_loss = body_loss / len(span_predictions)
            total_loss = total_loss + 0.05 * body_loss

        return {
            'loss': total_loss,
            'ce_losses': ce_losses,
            'kl_loss': kl_loss.item(),
            'uncert_loss': uncert_loss.item(),
            'exit_incentive': exit_incentive.item(),
            'span_loss': span_loss.item(),
            'body_loss': body_loss.item() if isinstance(body_loss, torch.Tensor) else body_loss,
            'avg_span_ratio': avg_span_ratio,
            'uncertainties': [u.mean().item() for u in exit_uncertainties]
        }


def main():
    print("="*60)
    print("Embodied GPT-2 Training (Early Exit + Dynamic Span)")
    print("="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Model
    print("\nLoading GPT-2 with embodied control...")
    model = EmbodiedGPT2("gpt2").to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Exit layers: {model.exit_layer_indices}")
    print(f"Span choices: {model.span_predictor.span_choices}")
    print(f"Trainable params: {trainable:,}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=3e-5
    )

    # LR scheduler
    num_steps = 1000
    warmup_steps = 100

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        return max(0.1, 1 - (step - warmup_steps) / (num_steps - warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Data
    print("\nLoading TinyStories...")
    dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)

    # Training
    print(f"\nTraining for {num_steps} steps...")
    model.train()

    metrics = []
    data_iter = iter(dataset)
    pbar = tqdm(range(num_steps))

    for step in pbar:
        # Get batch
        texts = []
        for _ in range(8):
            try:
                item = next(data_iter)
                texts.append(item['text'][:512])
            except StopIteration:
                data_iter = iter(dataset)
                item = next(data_iter)
                texts.append(item['text'][:512])

        # Tokenize
        encoded = tokenizer(
            texts,
            max_length=128,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        input_ids = encoded['input_ids'].to(device)
        labels = input_ids.clone()

        # Vary stress level during training
        stress = 0.3 + 0.4 * np.sin(step / 50)  # Oscillate 0.3-0.7
        body_state = model.simulate_body_state(8, device, stress)

        # Forward
        optimizer.zero_grad()
        out = model(input_ids, labels=labels, body_state=body_state)
        loss = out['loss']

        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # Log
        metrics.append({
            'step': step,
            'loss': loss.item(),
            'ce': out['ce_losses'],
            'kl': out['kl_loss'],
            'uncertainty': out['uncertainties'],
            'avg_span_ratio': out['avg_span_ratio'],
            'stress': stress
        })

        pbar.set_postfix({
            'loss': f"{loss.item():.3f}",
            'span': f"{out['avg_span_ratio']:.2f}",
            'stress': f"{stress:.2f}"
        })

        # Checkpoint
        if (step + 1) % 250 == 0:
            output_dir = Path("checkpoints/z152_embodied")
            output_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                'step': step,
                'exit_heads': model.exit_heads.state_dict(),
                'exit_uncertainty': model.exit_uncertainty.state_dict(),
                'span_predictor': model.span_predictor.state_dict(),
                'body_encoder': model.body_encoder.state_dict()
            }, output_dir / f"checkpoint_step{step+1}.pt")
            print(f"\n  Saved checkpoint at step {step+1}")

    # Final save
    output_dir = Path("checkpoints/z152_embodied")
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save({
        'exit_heads': model.exit_heads.state_dict(),
        'exit_uncertainty': model.exit_uncertainty.state_dict(),
        'span_predictor': model.span_predictor.state_dict(),
        'body_encoder': model.body_encoder.state_dict(),
        'exit_layers': model.exit_layer_indices,
        'span_choices': model.span_predictor.span_choices
    }, output_dir / "checkpoint.pt")

    with open(output_dir / "metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved to {output_dir}")

    # Evaluation
    print("\n" + "="*60)
    print("Evaluation")
    print("="*60)

    model.eval()
    test_prompts = [
        "Once upon a time",
        "The cat sat on",
        "Scientists discovered",
        "The epistemological",
    ]

    for stress_level in [0.2, 0.5, 0.8]:
        print(f"\n--- Stress Level: {stress_level} ---")

        for prompt in test_prompts:
            inputs = tokenizer(prompt, return_tensors='pt').to(device)
            body = model.simulate_body_state(1, device, stress_level)

            with torch.no_grad():
                out = model(inputs['input_ids'], body_state=body, return_all=True)

            u_values = [u.mean().item() for u in out['exit_uncertainties']]
            best_exit = np.argmin(u_values)
            best_layer = model.exit_layer_indices[best_exit]

            spans = [p['value'].item() for p in out['span_predictions']]
            avg_span = np.mean(spans)

            next_token = out['exit_outputs'][best_exit][0, -1].argmax().item()
            next_word = tokenizer.decode([next_token])

            print(f"  '{prompt}' -> layer {best_layer}, span {avg_span:.0f}, '{next_word}'")

    print("\n" + "="*60)
    print("Done!")
    print("="*60)


if __name__ == "__main__":
    main()
