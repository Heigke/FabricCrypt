#!/usr/bin/env python3
"""
z1002: Active Inference Generation
==================================

Tests if EFE-based (Expected Free Energy) selection beats greedy decoding.

Hypothesis: Generating multiple candidates and selecting by Expected Free Energy
produces better reasoning than greedy next-token prediction.

Inspired by: Active inference beating o1 on Mastermind (100% vs 29%)
"""

import os, sys, json, time, torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from telemetry.sysfs_hwmon import SysfsHwmonTelemetry

class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, hidden=128, layers=4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)
        self.pos = nn.Embedding(256, hidden)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(hidden, 4, hidden*4, batch_first=True)
            for _ in range(layers)
        ])
        self.ln = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, vocab_size)
        # Uncertainty head for EFE
        self.uncertainty = nn.Sequential(
            nn.Linear(hidden, 32), nn.GELU(), nn.Linear(32, 1), nn.Softplus()
        )

    def forward(self, x):
        B, T = x.shape
        h = self.embed(x) + self.pos(torch.arange(T, device=x.device))
        for block in self.blocks:
            h = block(h)
        h = self.ln(h)
        return self.head(h), self.uncertainty(h.mean(1))

def load_data():
    paths = [Path(__file__).parent.parent / "data" / "tinyshakespeare.txt"]
    for p in paths:
        if p.exists():
            return p.read_text()
    import urllib.request
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    path = paths[0]
    path.parent.mkdir(exist_ok=True)
    urllib.request.urlretrieve(url, path)
    return path.read_text()

def compute_efe(logits, uncertainty, lambda_explore=0.1):
    """
    Expected Free Energy = -log p(preferred) + epistemic_uncertainty

    Lower EFE = better candidate
    """
    # Entropy as exploration bonus
    probs = F.softmax(logits, dim=-1)
    entropy = -(probs * (probs + 1e-8).log()).sum(-1)

    # EFE: negative log-prob of most likely (exploitation) + uncertainty (exploration)
    max_logprob = logits.max(-1).values
    efe = -max_logprob + lambda_explore * uncertainty.squeeze() + 0.1 * entropy
    return efe

def generate_greedy(model, prompt, max_tokens, device):
    """Standard greedy decoding."""
    tokens = prompt.clone()
    for _ in range(max_tokens):
        logits, _ = model(tokens[:, -256:])
        next_token = logits[:, -1].argmax(-1, keepdim=True)
        tokens = torch.cat([tokens, next_token], dim=1)
    return tokens

def generate_active_inference(model, prompt, max_tokens, device, n_candidates=5):
    """Active inference: generate candidates, select by EFE."""
    tokens = prompt.clone()
    for _ in range(max_tokens):
        logits, uncertainty = model(tokens[:, -256:])
        last_logits = logits[:, -1]

        # Sample n candidates
        probs = F.softmax(last_logits / 0.8, dim=-1)
        candidates = torch.multinomial(probs.expand(n_candidates, -1), 1)  # [n, 1]

        # Evaluate each candidate's EFE
        efes = []
        for i in range(n_candidates):
            cand_tokens = torch.cat([tokens, candidates[i:i+1].T], dim=1)
            cand_logits, cand_unc = model(cand_tokens[:, -256:])
            efe = compute_efe(cand_logits[:, -1], cand_unc)
            efes.append(efe.item())

        # Select candidate with lowest EFE
        best_idx = np.argmin(efes)
        next_token = candidates[best_idx:best_idx+1].T
        tokens = torch.cat([tokens, next_token], dim=1)

    return tokens

def evaluate_coherence(text, n_gram=3):
    """Simple coherence: unique n-grams ratio (higher = more diverse/coherent)."""
    if len(text) < n_gram:
        return 1.0
    ngrams = [text[i:i+n_gram] for i in range(len(text)-n_gram+1)]
    return len(set(ngrams)) / len(ngrams)

def evaluate_repetition(text):
    """Repetition penalty (lower = more repetitive)."""
    words = text.split()
    if len(words) < 2:
        return 1.0
    bigrams = [(words[i], words[i+1]) for i in range(len(words)-1)]
    return len(set(bigrams)) / len(bigrams)

def main():
    print("="*60)
    print("z1002: ACTIVE INFERENCE GENERATION")
    print("="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    text = load_data()
    chars = sorted(set(text))
    char_to_idx = {c: i for i, c in enumerate(chars)}
    idx_to_char = {i: c for c, i in char_to_idx.items()}
    vocab_size = len(chars)
    data = torch.tensor([char_to_idx[c] for c in text], device=device)

    # Train model first
    print("\nTraining base model...")
    model = SimpleTransformer(vocab_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    for step in range(300):
        starts = torch.randint(0, len(data)-65, (32,))
        batch = torch.stack([data[s:s+64] for s in starts])
        logits, _ = model(batch)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, vocab_size), batch[:, 1:].reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            print(f"  Step {step}: Loss={loss.item():.3f}")

    model.eval()

    # Test generation
    print("\n" + "="*50)
    print("GENERATION COMPARISON")
    print("="*50)

    prompts = [
        "To be or not to be",
        "The king said unto",
        "Love is a",
    ]

    results = {'greedy': [], 'active': []}

    for prompt_text in prompts:
        print(f"\nPrompt: '{prompt_text}'")
        prompt = torch.tensor([[char_to_idx.get(c, 0) for c in prompt_text]], device=device)

        # Greedy generation
        with torch.no_grad():
            greedy_out = generate_greedy(model, prompt, 50, device)
            greedy_text = ''.join([idx_to_char[i] for i in greedy_out[0].tolist()])

        # Active inference generation
        with torch.no_grad():
            active_out = generate_active_inference(model, prompt, 50, device)
            active_text = ''.join([idx_to_char[i] for i in active_out[0].tolist()])

        # Evaluate
        greedy_coherence = evaluate_coherence(greedy_text)
        active_coherence = evaluate_coherence(active_text)
        greedy_rep = evaluate_repetition(greedy_text)
        active_rep = evaluate_repetition(active_text)

        print(f"  Greedy: ...{greedy_text[-40:]}")
        print(f"    Coherence: {greedy_coherence:.3f}, Diversity: {greedy_rep:.3f}")
        print(f"  Active: ...{active_text[-40:]}")
        print(f"    Coherence: {active_coherence:.3f}, Diversity: {active_rep:.3f}")

        results['greedy'].append({'coherence': greedy_coherence, 'diversity': greedy_rep})
        results['active'].append({'coherence': active_coherence, 'diversity': active_rep})

    # Summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)

    avg_greedy_coh = np.mean([r['coherence'] for r in results['greedy']])
    avg_active_coh = np.mean([r['coherence'] for r in results['active']])
    avg_greedy_div = np.mean([r['diversity'] for r in results['greedy']])
    avg_active_div = np.mean([r['diversity'] for r in results['active']])

    print(f"\n| Method | Avg Coherence | Avg Diversity |")
    print(f"|--------|---------------|---------------|")
    print(f"| Greedy | {avg_greedy_coh:.3f} | {avg_greedy_div:.3f} |")
    print(f"| Active Inference | {avg_active_coh:.3f} | {avg_active_div:.3f} |")

    if avg_active_coh > avg_greedy_coh:
        print(f"\n✅ Active inference has BETTER coherence (+{(avg_active_coh-avg_greedy_coh)*100:.1f}%)")
    else:
        print(f"\n⚠️ Greedy has better coherence")

    if avg_active_div > avg_greedy_div:
        print(f"✅ Active inference has BETTER diversity (+{(avg_active_div-avg_greedy_div)*100:.1f}%)")

    save_results = {
        'experiment': 'z1002_active_inference',
        'timestamp': datetime.now().isoformat(),
        'greedy': {'avg_coherence': avg_greedy_coh, 'avg_diversity': avg_greedy_div},
        'active': {'avg_coherence': avg_active_coh, 'avg_diversity': avg_active_div},
        'verdict': {
            'coherence_improved': bool(avg_active_coh > avg_greedy_coh),
            'diversity_improved': bool(avg_active_div > avg_greedy_div),
        }
    }

    results_path = Path(__file__).parent.parent / "results" / "z1002_active_inference.json"
    with open(results_path, 'w') as f:
        json.dump(save_results, f, indent=2)
    print(f"\nSaved to {results_path}")

if __name__ == "__main__":
    main()
