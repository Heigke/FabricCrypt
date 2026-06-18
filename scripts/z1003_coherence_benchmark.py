#!/usr/bin/env python3
"""
z1003: Coherence Benchmark
==========================

Measures if embodied (energy-aware) models produce more coherent outputs.

Metrics:
- Self-consistency: Does it contradict itself?
- Local coherence: Adjacent token similarity
- Global coherence: Whole-sequence consistency
- Repetition score: Diverse vs repetitive
"""

import os, sys, json, time, torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from telemetry.sysfs_hwmon import SysfsHwmonTelemetry, EnergyMeter

class EmbodiedTransformer(nn.Module):
    """Transformer with optional energy awareness."""
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

    def forward(self, x, body_state=None):
        B, T = x.shape
        h = self.embed(x) + self.pos(torch.arange(T, device=x.device))
        if self.use_body and body_state is not None:
            body_emb = self.body_proj(body_state).unsqueeze(1)
            h = h + body_emb * 0.1
        for block in self.blocks:
            h = block(h)
        h = self.ln(h)
        return self.head(h), h

class BodyTracker:
    def __init__(self):
        self.power_ema = self.temp_ema = 50.0
        self.alpha = 0.1

    def update(self, sample):
        self.power_ema = self.alpha * sample.power_w + (1-self.alpha) * self.power_ema
        self.temp_ema = self.alpha * sample.temp_edge_c + (1-self.alpha) * self.temp_ema
        return torch.tensor([
            sample.power_w/300, sample.temp_edge_c/100, sample.gpu_busy_pct/100,
            self.power_ema/300, self.temp_ema/100, 0.0, 0.0
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

def compute_local_coherence(hidden_states):
    """Cosine similarity between adjacent hidden states."""
    if hidden_states.size(1) < 2:
        return 1.0
    h1 = hidden_states[:, :-1]
    h2 = hidden_states[:, 1:]
    sim = F.cosine_similarity(h1, h2, dim=-1)
    return sim.mean().item()

def compute_self_consistency(hidden_states):
    """Compare first half vs second half representations."""
    T = hidden_states.size(1)
    if T < 4:
        return 1.0
    first_half = hidden_states[:, :T//2].mean(dim=1)
    second_half = hidden_states[:, T//2:].mean(dim=1)
    return F.cosine_similarity(first_half, second_half, dim=-1).mean().item()

def compute_repetition_score(tokens):
    """Unique bigrams ratio."""
    if len(tokens) < 3:
        return 1.0
    bigrams = [(tokens[i], tokens[i+1]) for i in range(len(tokens)-1)]
    return len(set(bigrams)) / len(bigrams)

def generate(model, prompt, body_state, max_tokens, device):
    """Generate tokens and collect hidden states."""
    tokens = prompt.clone()
    all_hidden = []

    for _ in range(max_tokens):
        logits, hidden = model(tokens[:, -256:], body_state)
        all_hidden.append(hidden[:, -1:])
        probs = F.softmax(logits[:, -1] / 0.8, dim=-1)
        next_token = torch.multinomial(probs, 1)
        tokens = torch.cat([tokens, next_token], dim=1)

    return tokens, torch.cat(all_hidden, dim=1)

def main():
    print("="*60)
    print("z1003: COHERENCE BENCHMARK")
    print("="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    text = load_data()
    chars = sorted(set(text))
    char_to_idx = {c: i for i, c in enumerate(chars)}
    idx_to_char = {i: c for c, i in char_to_idx.items()}
    vocab_size = len(chars)
    data = torch.tensor([char_to_idx[c] for c in text], device=device)

    telemetry = SysfsHwmonTelemetry()
    tracker = BodyTracker()

    results = {}

    for condition, use_body in [('A', False), ('B', True)]:
        label = 'Baseline (no body)' if condition == 'A' else 'Embodied (with body)'
        print(f"\n{'='*50}")
        print(f"Condition {condition}: {label}")
        print("="*50)

        model = EmbodiedTransformer(vocab_size, use_body=use_body).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

        # Train
        print("Training...")
        for step in range(250):
            starts = torch.randint(0, len(data)-65, (32,))
            batch = torch.stack([data[s:s+64] for s in starts])

            sample = telemetry.read_sample()
            body = tracker.update(sample).to(device).unsqueeze(0).expand(32, -1)

            logits, _ = model(batch, body if use_body else None)
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, vocab_size), batch[:, 1:].reshape(-1))

            # Add energy prediction loss for embodied
            if use_body:
                with EnergyMeter(telemetry) as meter:
                    _ = model(batch[:1], body[:1])
                # Small regularization based on energy
                loss = loss + 0.01 * meter.energy_j

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 100 == 0:
                print(f"  Step {step}: Loss={loss.item():.3f}")

        model.eval()

        # Evaluate coherence
        print("\nEvaluating coherence...")
        local_coherences = []
        self_consistencies = []
        repetition_scores = []

        prompts = ["The king ", "To be or ", "Love is "]

        for prompt_text in prompts:
            prompt = torch.tensor([[char_to_idx.get(c, 0) for c in prompt_text]], device=device)
            sample = telemetry.read_sample()
            body = tracker.update(sample).to(device).unsqueeze(0)

            with torch.no_grad():
                tokens, hidden = generate(model, prompt, body if use_body else None, 100, device)

            local_coh = compute_local_coherence(hidden)
            self_cons = compute_self_consistency(hidden)
            rep_score = compute_repetition_score(tokens[0].tolist())

            local_coherences.append(local_coh)
            self_consistencies.append(self_cons)
            repetition_scores.append(rep_score)

            gen_text = ''.join([idx_to_char[i] for i in tokens[0].tolist()])
            print(f"  '{prompt_text}' -> ...{gen_text[-30:]}")
            print(f"    Local: {local_coh:.3f}, Self-cons: {self_cons:.3f}, Diversity: {rep_score:.3f}")

        results[condition] = {
            'label': label,
            'local_coherence': np.mean(local_coherences),
            'self_consistency': np.mean(self_consistencies),
            'diversity': np.mean(repetition_scores),
        }

    # Summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)

    print(f"\n| Metric | Baseline | Embodied | Delta |")
    print(f"|--------|----------|----------|-------|")
    for metric in ['local_coherence', 'self_consistency', 'diversity']:
        a, b = results['A'][metric], results['B'][metric]
        delta = (b - a) / a * 100 if a > 0 else 0
        print(f"| {metric:18} | {a:.3f} | {b:.3f} | {delta:+.1f}% |")

    # Verdict
    embodied_wins = 0
    for metric in ['local_coherence', 'self_consistency', 'diversity']:
        if results['B'][metric] > results['A'][metric]:
            embodied_wins += 1

    if embodied_wins >= 2:
        print(f"\n✅ EMBODIED WINS on {embodied_wins}/3 metrics")
    else:
        print(f"\n⚠️ BASELINE WINS on {3-embodied_wins}/3 metrics")

    save_results = {
        'experiment': 'z1003_coherence_benchmark',
        'timestamp': datetime.now().isoformat(),
        'results': results,
        'verdict': {
            'embodied_wins': embodied_wins,
            'total_metrics': 3,
        }
    }

    results_path = Path(__file__).parent.parent / "results" / "z1003_coherence_benchmark.json"
    with open(results_path, 'w') as f:
        json.dump(save_results, f, indent=2)
    print(f"\nSaved to {results_path}")

if __name__ == "__main__":
    main()
