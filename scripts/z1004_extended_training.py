#!/usr/bin/env python3
"""
z1004: Extended Predictive Coding Training
==========================================

Longer training (2000 steps) to see if the z1000 improvements hold at scale.

Key questions:
1. Does energy prediction accuracy improve with more training?
2. Does the coherence advantage persist?
3. Does perplexity continue to improve?
"""

import os, sys, json, time, torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from telemetry.sysfs_hwmon import SysfsHwmonTelemetry, EnergyMeter

class PredictiveTransformer(nn.Module):
    """Full predictive transformer with all heads."""
    def __init__(self, vocab_size, hidden=256, layers=6, body_dim=7):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)
        self.pos = nn.Embedding(512, hidden)
        self.body_proj = nn.Linear(body_dim, hidden)

        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(hidden, 8, hidden*4, batch_first=True, dropout=0.1)
            for _ in range(layers)
        ])
        self.ln = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, vocab_size)

        # Predictive heads
        self.energy_head = nn.Sequential(
            nn.Linear(hidden, 64), nn.GELU(), nn.Linear(64, 2)
        )
        self.state_head = nn.Sequential(
            nn.Linear(hidden + body_dim, 64), nn.GELU(), nn.Linear(64, body_dim)
        )
        self.coherence_head = nn.Sequential(
            nn.Linear(hidden * 2, 64), nn.GELU(), nn.Linear(64, 1), nn.Sigmoid()
        )

    def forward(self, x, body_state, prev_hidden=None):
        B, T = x.shape
        h = self.embed(x) + self.pos(torch.arange(T, device=x.device).clamp(max=511))
        h = h + self.body_proj(body_state).unsqueeze(1) * 0.1

        for block in self.blocks:
            h = block(h)
        h = self.ln(h)
        logits = self.head(h)

        pooled = h.mean(dim=1)
        energy_out = self.energy_head(pooled)
        state_input = torch.cat([pooled, body_state], dim=-1)
        next_state = self.state_head(state_input)

        coherence = None
        if prev_hidden is not None:
            prev_pooled = prev_hidden.mean(dim=1)
            coh_input = torch.cat([prev_pooled, pooled], dim=-1)
            coherence = self.coherence_head(coh_input).squeeze(-1)

        return {
            'logits': logits,
            'hidden': h,
            'energy_mean': energy_out[:, 0],
            'energy_logvar': energy_out[:, 1],
            'next_state': next_state,
            'coherence': coherence,
        }

class BodyTracker:
    def __init__(self):
        self.ema = {'power': 50, 'temp': 50}
        self.prev = {'power': 50, 'temp': 50}
        self.alpha = 0.1

    def update(self, sample):
        self.ema['power'] = self.alpha * sample.power_w + (1-self.alpha) * self.ema['power']
        self.ema['temp'] = self.alpha * sample.temp_edge_c + (1-self.alpha) * self.ema['temp']
        deriv_power = sample.power_w - self.prev['power']
        deriv_temp = sample.temp_edge_c - self.prev['temp']
        self.prev['power'] = sample.power_w
        self.prev['temp'] = sample.temp_edge_c
        return torch.tensor([
            sample.power_w/300, sample.temp_edge_c/100, sample.gpu_busy_pct/100,
            self.ema['power']/300, self.ema['temp']/100, deriv_power/100, deriv_temp/10
        ], dtype=torch.float32)

def load_data():
    path = Path(__file__).parent.parent / "data" / "tinyshakespeare.txt"
    if path.exists():
        return path.read_text()
    import urllib.request
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    path.parent.mkdir(exist_ok=True)
    urllib.request.urlretrieve(url, path)
    return path.read_text()

def main():
    print("="*60)
    print("z1004: EXTENDED PREDICTIVE CODING TRAINING")
    print("="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text = load_data()
    chars = sorted(set(text))
    char_to_idx = {c: i for i, c in enumerate(chars)}
    vocab_size = len(chars)
    data = torch.tensor([char_to_idx[c] for c in text], device=device)

    print(f"Device: {device}, Vocab: {vocab_size}, Data: {len(data):,} chars")

    telemetry = SysfsHwmonTelemetry()
    tracker = BodyTracker()

    model = PredictiveTransformer(vocab_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2000)

    n_steps = 2000
    batch_size, seq_len = 48, 128
    energy_scale = 0.1

    # Metrics tracking
    metrics = {
        'step': [], 'task_loss': [], 'energy_loss': [], 'state_loss': [],
        'ppl': [], 'energy_mape': [], 'coherence': []
    }

    print(f"\nTraining for {n_steps} steps...")
    prev_hidden = None

    for step in range(n_steps):
        model.train()
        starts = torch.randint(0, len(data)-seq_len-1, (batch_size,))
        batch = torch.stack([data[s:s+seq_len] for s in starts])

        sample = telemetry.read_sample()
        body = tracker.update(sample).to(device).unsqueeze(0).expand(batch_size, -1)

        # Forward with energy measurement
        with EnergyMeter(telemetry) as meter:
            outputs = model(batch, body, prev_hidden)
        actual_energy = meter.energy_j / batch_size

        # Task loss
        logits = outputs['logits'][:, :-1].reshape(-1, vocab_size)
        targets = batch[:, 1:].reshape(-1)
        task_loss = F.cross_entropy(logits, targets)

        # Energy loss (Gaussian NLL)
        energy_mean = outputs['energy_mean']
        energy_logvar = outputs['energy_logvar']
        energy_target = torch.full((batch_size,), actual_energy / energy_scale, device=device)
        energy_var = energy_logvar.exp()
        energy_loss = 0.5 * (energy_logvar + (energy_target - energy_mean/energy_scale).pow(2) / (energy_var + 1e-6)).mean()

        # State loss
        time.sleep(0.002)
        next_sample = telemetry.read_sample()
        next_body = tracker.update(next_sample).to(device).unsqueeze(0).expand(batch_size, -1)
        state_loss = F.mse_loss(outputs['next_state'], next_body)

        # Coherence loss (encourage high coherence)
        coherence_loss = 0.0
        if outputs['coherence'] is not None:
            coherence_loss = -outputs['coherence'].mean()  # Maximize coherence

        # Combined loss
        loss = task_loss + 0.1 * energy_loss + 0.05 * state_loss + 0.01 * coherence_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        prev_hidden = outputs['hidden'].detach()

        # Track metrics
        ppl = torch.exp(task_loss).item()
        energy_mape = abs(energy_mean.mean().item() - actual_energy) / max(actual_energy, 1e-6)
        coh = outputs['coherence'].mean().item() if outputs['coherence'] is not None else 0

        metrics['step'].append(step)
        metrics['task_loss'].append(task_loss.item())
        metrics['energy_loss'].append(energy_loss.item())
        metrics['state_loss'].append(state_loss.item())
        metrics['ppl'].append(ppl)
        metrics['energy_mape'].append(energy_mape)
        metrics['coherence'].append(coh)

        if step % 100 == 0:
            print(f"  Step {step}: PPL={ppl:.1f}, E_loss={energy_loss.item():.3f}, Coh={coh:.3f}")

    # Final evaluation
    print("\n" + "="*50)
    print("FINAL EVALUATION")
    print("="*50)

    model.eval()
    final_ppl = np.mean(metrics['ppl'][-100:])
    final_energy_mape = np.mean(metrics['energy_mape'][-100:])
    final_coherence = np.mean(metrics['coherence'][-100:])

    print(f"\nFinal Metrics (last 100 steps avg):")
    print(f"  Perplexity: {final_ppl:.2f}")
    print(f"  Energy MAPE: {final_energy_mape*100:.1f}%")
    print(f"  Coherence: {final_coherence:.3f}")

    # Compare to start
    start_ppl = np.mean(metrics['ppl'][:100])
    start_energy_mape = np.mean(metrics['energy_mape'][:100])

    print(f"\nImprovement from start:")
    print(f"  PPL: {start_ppl:.1f} → {final_ppl:.1f} ({(start_ppl-final_ppl)/start_ppl*100:.1f}% better)")
    print(f"  Energy MAPE: {start_energy_mape*100:.1f}% → {final_energy_mape*100:.1f}%")

    # Save results
    results = {
        'experiment': 'z1004_extended_training',
        'timestamp': datetime.now().isoformat(),
        'n_steps': n_steps,
        'final_ppl': final_ppl,
        'final_energy_mape': final_energy_mape,
        'final_coherence': final_coherence,
        'improvement': {
            'ppl_reduction': (start_ppl - final_ppl) / start_ppl,
            'start_ppl': start_ppl,
        }
    }

    results_path = Path(__file__).parent.parent / "results" / "z1004_extended_training.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved to {results_path}")

if __name__ == "__main__":
    main()
