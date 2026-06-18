#!/usr/bin/env python3
"""
z1001: Self-Modeling Probe
==========================

Tests if embodied model can predict its own internal states.

Hypothesis: A model trained with energy awareness develops better
self-modeling capabilities (can predict its own activations/states).

Metrics:
- Introspective accuracy: Can it predict what state it's in?
- State prediction MSE: How well does it predict next body state?
- Correlation: Does prediction accuracy correlate with output quality?
"""

import os, sys, json, time, torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from telemetry.sysfs_hwmon import SysfsHwmonTelemetry, EnergyMeter, GpuSample

@dataclass
class BodyState:
    power_w: float = 0.0
    temp_c: float = 0.0
    util_pct: float = 0.0
    power_ema: float = 0.0
    temp_ema: float = 0.0
    power_deriv: float = 0.0
    temp_deriv: float = 0.0

    def to_tensor(self, device='cpu'):
        return torch.tensor([
            self.power_w / 300.0, self.temp_c / 100.0, self.util_pct / 100.0,
            self.power_ema / 300.0, self.temp_ema / 100.0,
            self.power_deriv / 100.0, self.temp_deriv / 10.0,
        ], dtype=torch.float32, device=device)

class BodyStateTracker:
    def __init__(self, ema_alpha=0.1):
        self.alpha = ema_alpha
        self.state = BodyState()
        self.prev_power = self.prev_temp = 0.0
        self.prev_time = time.time()
        self.initialized = False

    def update(self, sample):
        now = time.time()
        dt = max(now - self.prev_time, 0.001)
        if not self.initialized:
            self.state.power_w = sample.power_w
            self.state.temp_c = sample.temp_edge_c
            self.state.util_pct = sample.gpu_busy_pct
            self.state.power_ema = sample.power_w
            self.state.temp_ema = sample.temp_edge_c
            self.prev_power = sample.power_w
            self.prev_temp = sample.temp_edge_c
            self.initialized = True
        else:
            self.state.power_w = sample.power_w
            self.state.temp_c = sample.temp_edge_c
            self.state.util_pct = sample.gpu_busy_pct
            self.state.power_ema = self.alpha * sample.power_w + (1-self.alpha) * self.state.power_ema
            self.state.temp_ema = self.alpha * sample.temp_edge_c + (1-self.alpha) * self.state.temp_ema
            self.state.power_deriv = (sample.power_w - self.prev_power) / dt
            self.state.temp_deriv = (sample.temp_edge_c - self.prev_temp) / dt
            self.prev_power = sample.power_w
            self.prev_temp = sample.temp_edge_c
        self.prev_time = now
        return self.state

class SelfModelingTransformer(nn.Module):
    """Transformer that predicts its own hidden states."""
    def __init__(self, vocab_size, hidden_size=128, n_layers=4, body_dim=7):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.pos_embedding = nn.Embedding(256, hidden_size)
        self.body_proj = nn.Linear(body_dim, hidden_size)

        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(hidden_size, 4, hidden_size*4, batch_first=True)
            for _ in range(n_layers)
        ])
        self.ln = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)

        # Self-modeling heads
        self.hidden_predictor = nn.Sequential(
            nn.Linear(hidden_size + body_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.state_predictor = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.GELU(),
            nn.Linear(32, body_dim)
        )

    def forward(self, tokens, body_state, return_self_model=False):
        B, T = tokens.shape
        x = self.embedding(tokens) + self.pos_embedding(torch.arange(T, device=tokens.device))

        # Add body state as conditioning
        body_emb = self.body_proj(body_state).unsqueeze(1)
        x = x + body_emb * 0.1

        for block in self.blocks:
            x = block(x)
        x = self.ln(x)
        logits = self.lm_head(x)

        result = {'logits': logits, 'hidden': x}

        if return_self_model:
            pooled = x.mean(dim=1)
            # Predict what hidden state SHOULD be given body state
            pred_input = torch.cat([pooled, body_state], dim=-1)
            predicted_hidden = self.hidden_predictor(pred_input)
            result['predicted_hidden'] = predicted_hidden
            result['actual_hidden'] = pooled
            # Predict next body state
            result['predicted_state'] = self.state_predictor(pooled)

        return result

def load_data():
    paths = [
        Path(__file__).parent.parent / "data" / "tinyshakespeare.txt",
        Path(__file__).parent.parent / "data" / "tiny_shakespeare.txt",
    ]
    for p in paths:
        if p.exists():
            return p.read_text()
    import urllib.request
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    path = Path(__file__).parent.parent / "data" / "tinyshakespeare.txt"
    path.parent.mkdir(exist_ok=True)
    urllib.request.urlretrieve(url, path)
    return path.read_text()

def main():
    print("="*60)
    print("z1001: SELF-MODELING PROBE")
    print("="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    text = load_data()
    chars = sorted(set(text))
    char_to_idx = {c: i for i, c in enumerate(chars)}
    vocab_size = len(chars)
    data = torch.tensor([char_to_idx[c] for c in text], device=device)

    telemetry = SysfsHwmonTelemetry()
    tracker = BodyStateTracker()

    results = {}

    for condition in ['A', 'B']:
        label = 'Without self-modeling loss' if condition == 'A' else 'With self-modeling loss'
        print(f"\n{'='*50}")
        print(f"Condition {condition}: {label}")
        print("="*50)

        model = SelfModelingTransformer(vocab_size).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

        n_steps = 200
        batch_size, seq_len = 32, 64

        hidden_pred_errors = []
        state_pred_errors = []
        perplexities = []

        for step in range(n_steps):
            starts = torch.randint(0, len(data)-seq_len-1, (batch_size,))
            batch = torch.stack([data[s:s+seq_len] for s in starts])

            sample = telemetry.read_sample()
            body = tracker.update(sample)
            body_t = body.to_tensor(device).unsqueeze(0).expand(batch_size, -1)

            outputs = model(batch, body_t, return_self_model=True)

            # Task loss
            logits = outputs['logits'][:, :-1].reshape(-1, vocab_size)
            targets = batch[:, 1:].reshape(-1)
            task_loss = F.cross_entropy(logits, targets)

            # Self-modeling losses
            hidden_pred_loss = F.mse_loss(outputs['predicted_hidden'], outputs['actual_hidden'].detach())

            time.sleep(0.002)
            next_sample = telemetry.read_sample()
            next_body = tracker.update(next_sample)
            next_body_t = next_body.to_tensor(device).unsqueeze(0).expand(batch_size, -1)
            state_pred_loss = F.mse_loss(outputs['predicted_state'], next_body_t)

            # Total loss
            if condition == 'A':
                loss = task_loss
            else:
                loss = task_loss + 0.1 * hidden_pred_loss + 0.1 * state_pred_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            hidden_pred_errors.append(hidden_pred_loss.item())
            state_pred_errors.append(state_pred_loss.item())
            perplexities.append(torch.exp(task_loss).item())

            if step % 50 == 0:
                print(f"  Step {step}: PPL={perplexities[-1]:.1f}, HiddenErr={hidden_pred_loss.item():.4f}, StateErr={state_pred_loss.item():.4f}")

        # Final evaluation
        model.eval()
        final_hidden_err = np.mean(hidden_pred_errors[-20:])
        final_state_err = np.mean(state_pred_errors[-20:])
        final_ppl = np.mean(perplexities[-20:])

        results[condition] = {
            'final_ppl': final_ppl,
            'final_hidden_error': final_hidden_err,
            'final_state_error': final_state_err,
            'label': label,
        }

        print(f"\n  Final PPL: {final_ppl:.2f}")
        print(f"  Final Hidden Pred Error: {final_hidden_err:.4f}")
        print(f"  Final State Pred Error: {final_state_err:.4f}")

    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)

    ppl_a, ppl_b = results['A']['final_ppl'], results['B']['final_ppl']
    hidden_a, hidden_b = results['A']['final_hidden_error'], results['B']['final_hidden_error']
    state_a, state_b = results['A']['final_state_error'], results['B']['final_state_error']

    print(f"\n| Metric | Without Self-Model | With Self-Model |")
    print(f"|--------|-------------------|-----------------|")
    print(f"| PPL | {ppl_a:.2f} | {ppl_b:.2f} |")
    print(f"| Hidden Pred Err | {hidden_a:.4f} | {hidden_b:.4f} |")
    print(f"| State Pred Err | {state_a:.4f} | {state_b:.4f} |")

    # Verdict
    if hidden_b < hidden_a:
        print(f"\n✅ Self-modeling loss IMPROVES hidden state prediction")
    else:
        print(f"\n⚠️ Self-modeling loss does NOT improve hidden prediction")

    if state_b < state_a:
        print(f"✅ Self-modeling loss IMPROVES state prediction")

    # Save
    save_results = {
        'experiment': 'z1001_self_modeling',
        'timestamp': datetime.now().isoformat(),
        'results': results,
        'verdict': {
            'hidden_improved': bool(hidden_b < hidden_a),
            'state_improved': bool(state_b < state_a),
            'ppl_maintained': bool(ppl_b < ppl_a * 1.1),
        }
    }

    results_path = Path(__file__).parent.parent / "results" / "z1001_self_modeling.json"
    with open(results_path, 'w') as f:
        json.dump(save_results, f, indent=2)
    print(f"\nSaved to {results_path}")

if __name__ == "__main__":
    main()
