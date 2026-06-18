#!/usr/bin/env python3
"""
z1005: Reasoning Benchmark
==========================

Tests if embodied models perform better on reasoning tasks.

Tasks:
1. Simple arithmetic (multi-step)
2. Pattern completion
3. Logical deduction

Compares: Baseline vs Embodied vs Active Inference
"""

import os, sys, json, time, torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from telemetry.sysfs_hwmon import SysfsHwmonTelemetry

class ReasoningTransformer(nn.Module):
    def __init__(self, vocab_size, hidden=128, layers=4, body_dim=7, use_body=True):
        super().__init__()
        self.use_body = use_body
        self.embed = nn.Embedding(vocab_size, hidden)
        self.pos = nn.Embedding(256, hidden)
        if use_body:
            self.body_proj = nn.Linear(body_dim, hidden)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(hidden, 4, hidden*4, batch_first=True)
            for _ in range(layers)
        ])
        self.ln = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, vocab_size)
        self.uncertainty = nn.Sequential(
            nn.Linear(hidden, 32), nn.GELU(), nn.Linear(32, 1), nn.Softplus()
        )

    def forward(self, x, body_state=None):
        B, T = x.shape
        h = self.embed(x) + self.pos(torch.arange(min(T, 256), device=x.device))
        if self.use_body and body_state is not None:
            h = h + self.body_proj(body_state).unsqueeze(1) * 0.1
        for block in self.blocks:
            h = block(h)
        h = self.ln(h)
        return self.head(h), self.uncertainty(h.mean(1)), h

class BodyTracker:
    def __init__(self):
        self.ema = {'power': 50, 'temp': 50}
        self.alpha = 0.1
    def update(self, sample):
        self.ema['power'] = self.alpha * sample.power_w + (1-self.alpha) * self.ema['power']
        self.ema['temp'] = self.alpha * sample.temp_edge_c + (1-self.alpha) * self.ema['temp']
        return torch.tensor([
            sample.power_w/300, sample.temp_edge_c/100, sample.gpu_busy_pct/100,
            self.ema['power']/300, self.ema['temp']/100, 0.0, 0.0
        ], dtype=torch.float32)

# Simple reasoning tasks
def generate_arithmetic_data(n_samples=1000):
    """Generate simple arithmetic problems: a + b = ?"""
    data = []
    for _ in range(n_samples):
        a, b = np.random.randint(1, 50, 2)
        question = f"{a}+{b}="
        answer = str(a + b)
        data.append((question, answer))
    return data

def generate_pattern_data(n_samples=500):
    """Generate pattern completion: 2,4,6,? = 8"""
    data = []
    for _ in range(n_samples):
        start = np.random.randint(1, 10)
        step = np.random.randint(1, 5)
        seq = [start + i*step for i in range(4)]
        question = ','.join(map(str, seq[:3])) + ',?='
        answer = str(seq[3])
        data.append((question, answer))
    return data

def generate_logic_data(n_samples=500):
    """Simple if-then logic."""
    templates = [
        ("if A>B and B>C then A>C:", "yes"),
        ("if X=1 and Y=X then Y=", "1"),
        ("if all cats are animals and Tom is cat then Tom is:", "animal"),
    ]
    data = []
    for _ in range(n_samples):
        q, a = templates[np.random.randint(len(templates))]
        data.append((q, a))
    return data

def create_tokenizer(data):
    """Create character tokenizer from data."""
    all_text = ''.join([q + a for q, a in data])
    chars = sorted(set(all_text + '0123456789=+,?:'))
    char_to_idx = {c: i for i, c in enumerate(chars)}
    idx_to_char = {i: c for c, i in char_to_idx.items()}
    return char_to_idx, idx_to_char

def encode(text, char_to_idx):
    return [char_to_idx.get(c, 0) for c in text]

def decode(tokens, idx_to_char):
    return ''.join([idx_to_char.get(t, '?') for t in tokens])

def evaluate_model(model, test_data, char_to_idx, idx_to_char, device, body_state=None, use_active=False):
    """Evaluate model accuracy on test data."""
    correct = 0
    model.eval()

    for question, expected in test_data[:100]:  # Test on 100 samples
        prompt = torch.tensor([encode(question, char_to_idx)], device=device)

        with torch.no_grad():
            generated = prompt.clone()
            for _ in range(len(expected) + 2):
                logits, uncertainty, _ = model(generated[:, -128:], body_state)

                if use_active:
                    # Active inference: sample candidates, pick lowest uncertainty
                    probs = F.softmax(logits[:, -1] / 0.7, dim=-1)
                    candidates = torch.multinomial(probs.expand(5, -1), 1)
                    best_unc = float('inf')
                    best_tok = candidates[0]
                    for c in candidates:
                        test_seq = torch.cat([generated, c.unsqueeze(0).T], dim=1)
                        _, unc, _ = model(test_seq[:, -128:], body_state)
                        if unc.item() < best_unc:
                            best_unc = unc.item()
                            best_tok = c
                    next_token = best_tok.unsqueeze(0).T
                else:
                    next_token = logits[:, -1].argmax(-1, keepdim=True)

                generated = torch.cat([generated, next_token], dim=1)

            output = decode(generated[0, len(prompt[0]):].tolist(), idx_to_char)
            output = output.split('=')[-1] if '=' in output else output
            output = ''.join(c for c in output if c.isalnum())[:len(expected)]

            if output == expected:
                correct += 1

    return correct / 100.0

def main():
    print("="*60)
    print("z1005: REASONING BENCHMARK")
    print("="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Generate training data
    print("\nGenerating reasoning data...")
    arith_data = generate_arithmetic_data(2000)
    pattern_data = generate_pattern_data(500)
    logic_data = generate_logic_data(500)

    all_data = arith_data + pattern_data + logic_data
    np.random.shuffle(all_data)
    train_data = all_data[:2500]
    test_data = all_data[2500:]

    char_to_idx, idx_to_char = create_tokenizer(all_data)
    vocab_size = len(char_to_idx)
    print(f"Vocab size: {vocab_size}, Train: {len(train_data)}, Test: {len(test_data)}")

    telemetry = SysfsHwmonTelemetry()
    tracker = BodyTracker()

    results = {}

    for condition, use_body, use_active in [
        ('Baseline', False, False),
        ('Embodied', True, False),
        ('Embodied+Active', True, True),
    ]:
        print(f"\n{'='*50}")
        print(f"Training: {condition}")
        print("="*50)

        model = ReasoningTransformer(vocab_size, use_body=use_body).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

        # Training
        for step in range(500):
            batch_data = [train_data[i] for i in np.random.randint(0, len(train_data), 32)]
            max_len = max(len(q) + len(a) for q, a in batch_data)

            batch = []
            for q, a in batch_data:
                seq = encode(q + a, char_to_idx)
                seq = seq + [0] * (max_len - len(seq))
                batch.append(seq)
            batch = torch.tensor(batch, device=device)

            sample = telemetry.read_sample()
            body = tracker.update(sample).to(device).unsqueeze(0).expand(32, -1)

            logits, _, _ = model(batch, body if use_body else None)
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, vocab_size), batch[:, 1:].reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 100 == 0:
                print(f"  Step {step}: Loss={loss.item():.3f}")

        # Evaluation
        print("\nEvaluating...")
        sample = telemetry.read_sample()
        body = tracker.update(sample).to(device).unsqueeze(0)

        accuracy = evaluate_model(
            model, test_data, char_to_idx, idx_to_char, device,
            body if use_body else None, use_active
        )

        results[condition] = {'accuracy': accuracy}
        print(f"  Accuracy: {accuracy*100:.1f}%")

    # Summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)

    print(f"\n| Condition | Accuracy |")
    print(f"|-----------|----------|")
    for cond, res in results.items():
        print(f"| {cond:17} | {res['accuracy']*100:6.1f}% |")

    # Verdict
    baseline_acc = results['Baseline']['accuracy']
    embodied_acc = results['Embodied']['accuracy']
    active_acc = results['Embodied+Active']['accuracy']

    if embodied_acc > baseline_acc:
        print(f"\n✅ Embodied beats Baseline (+{(embodied_acc-baseline_acc)*100:.1f}%)")
    if active_acc > embodied_acc:
        print(f"✅ Active inference adds +{(active_acc-embodied_acc)*100:.1f}%")

    save_results = {
        'experiment': 'z1005_reasoning_benchmark',
        'timestamp': datetime.now().isoformat(),
        'results': results,
        'verdict': {
            'embodied_improves': bool(embodied_acc > baseline_acc),
            'active_improves': bool(active_acc > embodied_acc),
        }
    }

    results_path = Path(__file__).parent.parent / "results" / "z1005_reasoning_benchmark.json"
    with open(results_path, 'w') as f:
        json.dump(save_results, f, indent=2)
    print(f"\nSaved to {results_path}")

if __name__ == "__main__":
    main()
