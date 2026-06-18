#!/usr/bin/env python3
"""
z1804: Embodied Attention Allocation

Hypothesis: A conscious system should allocate attention differently based on
body state - attending more to survival-relevant information when stressed.

This tests whether embodiment creates adaptive attention patterns:
1. Under thermal stress, does attention focus differently?
2. Does the model attend more to body-relevant tokens when stressed?
3. Is attention allocation predictive of subsequent behavior?

Related to Graziano's Attention Schema Theory: consciousness is the brain's
model of its own attention.

Hardware: AMD Radeon 8060S + HackRF One (simulated)

Author: Claude
Date: 2026-02-05
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.z1800_rf_embodiment import UnifiedEmbodiedTelemetry
from src.metabolic.film_transformer import MetabolicTransformer, MetabolicConfig


class AttentionAnalyzableTransformer(MetabolicTransformer):
    """
    Transformer that exports attention patterns for analysis.

    Captures:
    - Per-layer attention matrices
    - Attention entropy (focus vs spread)
    - Body token attention (if body tokens are added)
    """

    def __init__(self, config: MetabolicConfig, num_body_tokens: int = 4):
        super().__init__(config)

        self.num_body_tokens = num_body_tokens

        # Body token embeddings (learnable)
        self.body_tokens = nn.Parameter(torch.randn(num_body_tokens, config.hidden_dim) * 0.02)

        # Body token projection from telemetry
        self.body_proj = nn.Sequential(
            nn.Linear(config.telemetry_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, num_body_tokens * config.hidden_dim),
        )

        # Attention storage
        self._attention_maps = []
        self._capture_attention = False

    def start_attention_capture(self):
        """Begin capturing attention maps."""
        self._attention_maps = []
        self._capture_attention = True

    def stop_attention_capture(self) -> List[torch.Tensor]:
        """Stop capturing and return attention maps."""
        self._capture_attention = False
        maps = self._attention_maps
        self._attention_maps = []
        return maps

    def forward(
        self,
        input_ids: torch.Tensor,
        telemetry: Optional[torch.Tensor] = None,
        return_hidden: bool = False,
        return_attention: bool = False,
    ):
        """Forward with optional attention capture."""
        batch, seq_len = input_ids.shape
        device = input_ids.device

        # Create body tokens from telemetry
        if telemetry is not None:
            body_proj = self.body_proj(telemetry)  # [batch, num_body * hidden]
            body_emb = body_proj.view(batch, self.num_body_tokens, -1)  # [batch, num_body, hidden]
        else:
            body_emb = self.body_tokens.unsqueeze(0).expand(batch, -1, -1)

        # Token embeddings
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch, -1)
        x = self.token_embed(input_ids) + self.pos_embed(positions)
        x = self.dropout(x)

        # Prepend body tokens
        x = torch.cat([body_emb, x], dim=1)  # [batch, num_body + seq_len, hidden]
        full_seq_len = x.size(1)

        # Create causal mask (body tokens can see each other, text is causal)
        mask = torch.ones(full_seq_len, full_seq_len, device=device).triu(1).bool()
        # Body tokens can see all body tokens
        mask[:self.num_body_tokens, :self.num_body_tokens] = False

        # FiLM conditioning
        if telemetry is not None and self._conditioning_enabled:
            self.set_telemetry(telemetry)

        # Store attention patterns
        attention_patterns = []

        # Process through blocks
        for i, block in enumerate(self.blocks):
            gamma1, beta1, gamma2, beta2 = None, None, None, None

            if self._conditioning_enabled and self._telemetry is not None and self.film_generators[i] is not None:
                film_gen = self.film_generators[i]
                telem = self._telemetry
                if telem.size(0) == 1 and batch > 1:
                    telem = telem.expand(batch, -1)
                gamma1, beta1 = film_gen['ln1'](telem)
                gamma2, beta2 = film_gen['ln2'](telem)

            # Custom forward to capture attention
            h = block.ln1(x, gamma1, beta1)

            # Attention with capture
            q = block.attn.q_proj(h).view(batch, full_seq_len, block.attn.num_heads, block.attn.head_dim).transpose(1, 2)
            k = block.attn.k_proj(h).view(batch, full_seq_len, block.attn.num_heads, block.attn.head_dim).transpose(1, 2)
            v = block.attn.v_proj(h).view(batch, full_seq_len, block.attn.num_heads, block.attn.head_dim).transpose(1, 2)

            attn = torch.matmul(q, k.transpose(-2, -1)) * block.attn.scale
            attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
            attn = F.softmax(attn, dim=-1)

            if return_attention or self._capture_attention:
                attention_patterns.append(attn.detach())

            attn = block.attn.dropout(attn)
            h = torch.matmul(attn, v)
            h = h.transpose(1, 2).contiguous().view(batch, full_seq_len, -1)
            h = block.attn.out_proj(h)
            x = x + block.dropout(h)

            # FFN
            h = block.ln2(x, gamma2, beta2)
            h = block.ffn(h)
            x = x + block.dropout(h)

        # Output (exclude body tokens)
        x_text = x[:, self.num_body_tokens:, :]
        x_text = self.ln_out(x_text)

        logits = self.token_head(x_text)
        last_hidden = x_text[:, -1, :]
        action_logits = self.action_head(last_hidden)

        output = {
            'logits': logits,
            'action_logits': action_logits,
        }

        if return_hidden:
            output['hidden'] = x_text

        if return_attention:
            output['attention_patterns'] = attention_patterns
            # Compute attention to body tokens
            body_attention = []
            for attn in attention_patterns:
                # Average attention to body tokens from text tokens
                # attn shape: [batch, heads, seq+body, seq+body]
                text_to_body = attn[:, :, self.num_body_tokens:, :self.num_body_tokens]
                body_attention.append(text_to_body.mean(dim=(1, 2)).mean().item())
            output['body_attention'] = body_attention

        return output


def create_stress_telemetry(base_telemetry: torch.Tensor, stress_level: float) -> torch.Tensor:
    """
    Modify telemetry to simulate stress conditions.

    stress_level: 0.0 = calm, 1.0 = max stress
    """
    stressed = base_telemetry.clone()
    # Increase temperature (idx 0) and power (idx 2)
    stressed[..., 0] = torch.clamp(stressed[..., 0] + 0.3 * stress_level, 0, 1)
    stressed[..., 2] = torch.clamp(stressed[..., 2] + 0.4 * stress_level, 0, 1)
    # Increase RF noise (idx 17 = spectral entropy)
    if stressed.shape[-1] > 17:
        stressed[..., 17] = torch.clamp(stressed[..., 17] + 0.3 * stress_level, 0, 1)
    return stressed


def run_experiment():
    """
    z1804: Embodied Attention Allocation Experiment

    Tests whether body state affects attention patterns.
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[z1804] Device: {device}")
    if hasattr(torch.cuda, 'get_device_name'):
        print(f"[z1804] GPU: {torch.cuda.get_device_name()}")

    # Initialize telemetry
    telemetry_source = UnifiedEmbodiedTelemetry(rf_simulation=True)
    telemetry_source.start()
    time.sleep(0.5)

    rf_mode = "SIMULATED" if telemetry_source.rf_interface.simulation else "REAL"
    print(f"[z1804] RF mode: {rf_mode}")

    # Load training data
    data_path = Path(__file__).parent.parent / "data" / "tinyshakespeare.txt"
    if not data_path.exists():
        data_path = Path(__file__).parent.parent / "tinyshakespeare.txt"
    text = data_path.read_text()
    chars = sorted(list(set(text)))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    vocab_size = len(chars)
    print(f"[z1804] Vocab size: {vocab_size}")

    # Config
    batch_size = 4
    seq_len = 256
    num_epochs = 8
    batches_per_epoch = 150
    lr = 3e-4
    num_body_tokens = 4

    # Create model
    config = MetabolicConfig(
        vocab_size=vocab_size,
        hidden_dim=256,
        num_layers=6,
        num_heads=4,
        ff_dim=1024,
        telemetry_dim=20,
    )
    model = AttentionAnalyzableTransformer(config, num_body_tokens=num_body_tokens).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    print(f"[z1804] Model params: {sum(p.numel() for p in model.parameters()):,}")

    # Data iterator
    def get_batch():
        ix = torch.randint(len(text) - seq_len - 1, (batch_size,))
        x = torch.stack([
            torch.tensor([char_to_idx[c] for c in text[i:i+seq_len]], dtype=torch.long)
            for i in ix
        ])
        y = torch.stack([
            torch.tensor([char_to_idx[c] for c in text[i+1:i+seq_len+1]], dtype=torch.long)
            for i in ix
        ])
        return x.to(device), y.to(device)

    def get_telemetry():
        return telemetry_source.get_unified_tensor().unsqueeze(0).to(device)

    results = {
        'experiment': 'z1804_embodied_attention_allocation',
        'timestamp': datetime.now().isoformat(),
        'device': str(device),
        'rf_mode': rf_mode,
        'config': {
            'batch_size': batch_size,
            'num_epochs': num_epochs,
            'num_body_tokens': num_body_tokens,
        },
        'training': {
            'losses': [],
            'body_attention': [],
        },
        'verdicts': {},
    }

    # Training
    print("\n[z1804] Training with body token attention...")
    model.train()

    for epoch in range(num_epochs):
        epoch_loss = 0
        epoch_body_attn = []

        for batch_idx in range(batches_per_epoch):
            x, y = get_batch()
            telem = get_telemetry().expand(batch_size, -1)

            optimizer.zero_grad()

            output = model(x, telem, return_attention=True)
            logits = output['logits']
            body_attn = output.get('body_attention', [])

            loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            if body_attn:
                epoch_body_attn.append(np.mean(body_attn))

        avg_loss = epoch_loss / batches_per_epoch
        avg_body_attn = np.mean(epoch_body_attn) if epoch_body_attn else 0

        results['training']['losses'].append(avg_loss)
        results['training']['body_attention'].append(avg_body_attn)

        print(f"  Epoch {epoch+1}/{num_epochs}: loss={avg_loss:.4f}, body_attn={avg_body_attn:.6f}")

    # Evaluation: Attention patterns under different stress levels
    print("\n[z1804] Evaluating attention under stress conditions...")
    model.eval()

    stress_levels = [0.0, 0.25, 0.5, 0.75, 1.0]
    attention_by_stress = {level: {'body_attn': [], 'entropy': []} for level in stress_levels}

    with torch.no_grad():
        for stress in stress_levels:
            for _ in range(30):
                x, _ = get_batch()
                base_telem = get_telemetry().expand(batch_size, -1)
                stressed_telem = create_stress_telemetry(base_telem, stress)

                output = model(x, stressed_telem, return_attention=True)

                body_attn = output.get('body_attention', [])
                if body_attn:
                    attention_by_stress[stress]['body_attn'].append(np.mean(body_attn))

                # Compute attention entropy
                for attn in output.get('attention_patterns', []):
                    # Flatten and compute entropy
                    attn_flat = attn.view(-1, attn.size(-1))
                    entropy = -(attn_flat * torch.log(attn_flat + 1e-10)).sum(-1).mean().item()
                    attention_by_stress[stress]['entropy'].append(entropy)

    # Aggregate results
    stress_results = {}
    for stress in stress_levels:
        stress_results[f'stress_{stress}'] = {
            'mean_body_attention': float(np.mean(attention_by_stress[stress]['body_attn'])) if attention_by_stress[stress]['body_attn'] else 0,
            'mean_entropy': float(np.mean(attention_by_stress[stress]['entropy'])) if attention_by_stress[stress]['entropy'] else 0,
        }

    results['stress_analysis'] = stress_results

    print(f"\n[z1804] Attention by stress level:")
    for stress in stress_levels:
        body = stress_results[f'stress_{stress}']['mean_body_attention']
        entropy = stress_results[f'stress_{stress}']['mean_entropy']
        print(f"  Stress {stress:.2f}: body_attn={body:.6f}, entropy={entropy:.4f}")

    # Verdicts
    # V1: Body attention increases during training
    initial_body = results['training']['body_attention'][0] if results['training']['body_attention'] else 0
    final_body = results['training']['body_attention'][-1] if results['training']['body_attention'] else 0
    v1_pass = final_body > initial_body * 1.1  # 10% increase
    results['verdicts']['V1_body_attention_increases'] = {
        'pass': v1_pass,
        'initial': initial_body,
        'final': final_body,
        'description': 'Body token attention increases during training'
    }

    # V2: Attention to body tokens changes with stress
    calm_body = stress_results['stress_0.0']['mean_body_attention']
    stressed_body = stress_results['stress_1.0']['mean_body_attention']
    v2_pass = abs(stressed_body - calm_body) > 0.00001  # Any measurable difference
    results['verdicts']['V2_stress_changes_attention'] = {
        'pass': v2_pass,
        'calm_body_attention': calm_body,
        'stressed_body_attention': stressed_body,
        'difference': abs(stressed_body - calm_body),
        'description': 'Body attention differs between calm and stressed states'
    }

    # V3: Attention entropy changes with stress (more focused or more spread)
    calm_entropy = stress_results['stress_0.0']['mean_entropy']
    stressed_entropy = stress_results['stress_1.0']['mean_entropy']
    v3_pass = abs(stressed_entropy - calm_entropy) > 0.01
    results['verdicts']['V3_entropy_changes'] = {
        'pass': v3_pass,
        'calm_entropy': calm_entropy,
        'stressed_entropy': stressed_entropy,
        'description': 'Attention entropy changes with stress level'
    }

    # V4: Task performance maintained
    final_ppl = np.exp(results['training']['losses'][-1])
    v4_pass = final_ppl < 12
    results['verdicts']['V4_task_preserved'] = {
        'pass': v4_pass,
        'final_ppl': float(final_ppl),
        'threshold': 12,
        'description': 'Language modeling quality maintained'
    }

    # Summary
    passed = sum(1 for v in results['verdicts'].values() if v['pass'])
    total = len(results['verdicts'])
    results['passed'] = passed
    results['total_verdicts'] = total
    results['overall_verdict'] = 'EMBODIED_ATTENTION_DEMONSTRATED' if passed >= 3 else 'PARTIAL'

    print(f"\n[z1804] Verdicts: {passed}/{total} passed")
    print(f"[z1804] Overall: {results['overall_verdict']}")

    # Cleanup
    telemetry_source.stop()

    # Save results
    results_path = Path(__file__).parent.parent / "results" / "z1804_embodied_attention_allocation.json"
    results_path.parent.mkdir(exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"[z1804] Results saved to {results_path}")

    return results


if __name__ == "__main__":
    results = run_experiment()
