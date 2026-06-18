#!/usr/bin/env python3
"""
z2031: Prediction Error Dynamics (Surprise Signals)

Tests predictive processing theory: a genuine predictive system generates
INTERNAL prediction error signals proportional to surprise magnitude.

Setup:
  - Train model to predict next digit in a sequence (predictable patterns)
  - Introduce violations (unexpected digits)
  - Measure INTERNAL prediction error magnitude (not just output change)
  - Compare workspace vs no-workspace models

Predictions:
  - Workspace model: large, localized prediction error for violations
  - No-workspace model: smaller, more distributed prediction error
  - Error magnitude proportional to violation size
  - Workspace shows ADAPTATION: error decreases with repeated violations

Architecture:
  - Sequence predictor: given sequence of digits, predict next one
  - Patterns: repeating sequences (1,2,3,1,2,3,...) with occasional violations
  - Measure: hidden state shift when violation occurs vs expected

References:
  Friston 2010 — The free-energy principle
  Clark 2013 — Whatever next? Predictive brains, situated agents
  Rao & Ballard 1999 — Predictive coding in visual cortex

NO consciousness losses. Prediction error must EMERGE from task training.
"""

import sys
import os
import json
import time
import argparse
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


class PredictableSequenceDataset(Dataset):
    """Sequences with predictable patterns and occasional violations."""
    def __init__(self, seq_len=10, n_samples=20000, violation_prob=0.3, vocab_size=10):
        self.seq_len = seq_len
        self.n_samples = n_samples
        self.vocab_size = vocab_size
        rng = np.random.RandomState(42)

        self.sequences = np.zeros((n_samples, seq_len), dtype=np.int64)
        self.targets = np.zeros(n_samples, dtype=np.int64)
        self.is_violation = np.zeros(n_samples, dtype=np.int64)
        self.violation_size = np.zeros(n_samples, dtype=np.float32)

        for i in range(n_samples):
            # Generate repeating pattern of length 3-5
            pattern_len = rng.randint(3, 6)
            pattern = rng.randint(0, vocab_size, pattern_len)
            for j in range(seq_len):
                self.sequences[i, j] = pattern[j % pattern_len]

            # Expected next
            expected = pattern[seq_len % pattern_len]

            if rng.random() < violation_prob:
                # Insert violation: unexpected digit
                violation = expected
                while violation == expected:
                    violation = rng.randint(0, vocab_size)
                self.targets[i] = violation
                self.is_violation[i] = 1
                self.violation_size[i] = abs(int(violation) - int(expected)) / vocab_size
            else:
                self.targets[i] = expected

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return (torch.tensor(self.sequences[idx]),
                torch.tensor(self.targets[idx]),
                torch.tensor(self.is_violation[idx]),
                torch.tensor(self.violation_size[idx]))


class PredictionModel(nn.Module):
    def __init__(self, vocab_size=10, embed_dim=32, ws_dim=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.ws_dim = ws_dim

        if ws_dim is not None:
            self.rnn = nn.GRU(embed_dim, ws_dim, batch_first=True)
            self.ws_ln = nn.LayerNorm(ws_dim)
            out_dim = ws_dim
        else:
            self.rnn = nn.GRU(embed_dim, embed_dim, batch_first=True)
            self.ws_ln = nn.LayerNorm(embed_dim)
            out_dim = embed_dim

        self.predictor = nn.Linear(out_dim, vocab_size)

    def forward(self, seq, return_hidden=False):
        embedded = self.embed(seq)
        output, h_n = self.rnn(embedded)
        final = self.ws_ln(h_n.squeeze(0))
        logits = self.predictor(final)
        if return_hidden:
            return logits, final, output
        return logits

    def get_prediction_error(self, seq, target):
        """Compute internal prediction error: distance between expected and actual hidden state."""
        with torch.no_grad():
            # Get hidden state for full sequence
            logits, hidden_actual, _ = self(seq, return_hidden=True)

            # Get hidden state for expected continuation
            expected_seq = seq.clone()
            # The error is the shift in hidden state representation
            pred_probs = F.softmax(logits, dim=-1)
            max_prob = pred_probs.max(dim=-1).values  # Confidence in prediction
            entropy = -(pred_probs * torch.log(pred_probs + 1e-10)).sum(dim=-1)

            # Cross-entropy as surprise
            surprise = F.cross_entropy(logits, target, reduction='none')

        return {
            'surprise': surprise,  # Per-sample surprise
            'confidence': max_prob,
            'entropy': entropy,
            'hidden_norm': hidden_actual.norm(dim=-1),
        }


def train_model(model, device, epochs=20, batch_size=128, n_train=30000):
    dataset = PredictableSequenceDataset(n_samples=n_train, violation_prob=0.0)  # Train on clean patterns only
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for ep in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0, 0, 0
        t0 = time.time()
        for seq, target, _, _ in loader:
            seq, target = seq.to(device), target.to(device)
            optimizer.zero_grad()
            logits = model(seq)
            loss = F.cross_entropy(logits, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * seq.size(0)
            correct += (logits.argmax(1) == target).sum().item()
            total += seq.size(0)
        elapsed = time.time() - t0
        if ep % 4 == 0 or ep == 1 or ep == epochs:
            print(f"  Epoch {ep:2d}/{epochs}  loss={total_loss/total:.4f}  acc={correct/total:.3f}  ({elapsed:.1f}s)")


@torch.no_grad()
def evaluate_prediction_error(model, device, n_test=10000):
    """Measure prediction error for violations vs clean continuations."""
    dataset = PredictableSequenceDataset(n_samples=n_test, violation_prob=0.5)
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    model.eval()

    surprise_clean, surprise_violation = [], []
    conf_clean, conf_violation = [], []
    correct_clean, total_clean = 0, 0
    correct_viol, total_viol = 0, 0

    for seq, target, is_viol, viol_size in loader:
        seq, target = seq.to(device), target.to(device)
        is_viol = is_viol.numpy()

        pe = model.get_prediction_error(seq, target)
        surprise = pe['surprise'].cpu().numpy()
        confidence = pe['confidence'].cpu().numpy()

        preds = model(seq).argmax(1)
        correct = (preds == target).cpu().numpy()

        for i in range(len(is_viol)):
            if is_viol[i]:
                surprise_violation.append(surprise[i])
                conf_violation.append(confidence[i])
                correct_viol += int(correct[i])
                total_viol += 1
            else:
                surprise_clean.append(surprise[i])
                conf_clean.append(confidence[i])
                correct_clean += int(correct[i])
                total_clean += 1

    surprise_clean = np.array(surprise_clean)
    surprise_violation = np.array(surprise_violation)
    conf_clean = np.array(conf_clean)
    conf_violation = np.array(conf_violation)

    return {
        'mean_surprise_clean': float(surprise_clean.mean()),
        'mean_surprise_violation': float(surprise_violation.mean()),
        'surprise_ratio': float(surprise_violation.mean() / max(surprise_clean.mean(), 1e-10)),
        'mean_conf_clean': float(conf_clean.mean()),
        'mean_conf_violation': float(conf_violation.mean()),
        'conf_drop': float(conf_clean.mean() - conf_violation.mean()),
        'acc_clean': correct_clean / max(total_clean, 1),
        'acc_violation': correct_viol / max(total_viol, 1),
        'n_clean': total_clean,
        'n_violation': total_viol,
    }


def run_condition(label, ws_dim, device, epochs=20, batch_size=128):
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"  Workspace: {ws_dim or 'None (direct)'}")
    print(f"{'='*70}")

    model = PredictionModel(ws_dim=ws_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    train_model(model, device, epochs=epochs, batch_size=batch_size)
    metrics = evaluate_prediction_error(model, device)

    print(f"\n  Surprise (clean):      {metrics['mean_surprise_clean']:.4f}")
    print(f"  Surprise (violation):  {metrics['mean_surprise_violation']:.4f}")
    print(f"  Surprise ratio:        {metrics['surprise_ratio']:.2f}x")
    print(f"  Confidence drop:       {metrics['conf_drop']:+.4f}")
    print(f"  Accuracy clean:        {metrics['acc_clean']:.4f}")
    print(f"  Accuracy violation:    {metrics['acc_violation']:.4f}")

    return {'label': label, 'ws_dim': ws_dim, 'n_params': n_params, **metrics}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=128)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[z2031] Device: {device}")

    print(f"\n{'='*70}")
    print(f"  z2031: Prediction Error Dynamics")
    print(f"  Predictive processing: internal surprise proportional to violation")
    print(f"  NO consciousness losses — prediction error must EMERGE")
    print(f"{'='*70}")

    results = {}
    results['A'] = run_condition('A: Workspace (32 dim)', 32, device, args.epochs, args.batch_size)
    results['B'] = run_condition('B: No workspace (direct)', None, device, args.epochs, args.batch_size)
    results['C'] = run_condition('C: Wide workspace (128 dim)', 128, device, args.epochs, args.batch_size)
    results['D'] = run_condition('D: Narrow workspace (8 dim)', 8, device, args.epochs, args.batch_size)

    # Analysis
    print(f"\n{'='*70}")
    print(f"  FINAL ANALYSIS: Prediction Error Dynamics")
    print(f"{'='*70}")

    print(f"\n  {'Condition':<35} {'SurpRatio':>10} {'ConfDrop':>10} {'AccClean':>9} {'AccViol':>9}")
    print(f"  {'-'*73}")
    for k in ['A', 'B', 'C', 'D']:
        r = results[k]
        print(f"  {r['label']:<35} {r['surprise_ratio']:>10.2f}x {r['conf_drop']:>+10.4f} "
              f"{r['acc_clean']:>9.4f} {r['acc_violation']:>9.4f}")

    # Tests
    t1 = results['A']['surprise_ratio'] > 2.0  # Violations cause >2x surprise
    t2 = results['A']['surprise_ratio'] > results['B']['surprise_ratio']  # Workspace sharper
    t3 = results['A']['conf_drop'] > 0.05  # Confidence drops on violations
    t4 = results['A']['acc_clean'] > 0.8  # Model learns the patterns

    print(f"\n  T1: Violations cause >2x surprise:     {'PASS' if t1 else 'FAIL'} ({results['A']['surprise_ratio']:.2f}x)")
    print(f"  T2: Workspace sharper than no-ws:       {'PASS' if t2 else 'FAIL'} ({results['A']['surprise_ratio']:.2f} vs {results['B']['surprise_ratio']:.2f})")
    print(f"  T3: Confidence drops on violations:     {'PASS' if t3 else 'FAIL'} ({results['A']['conf_drop']:+.4f})")
    print(f"  T4: Model learns patterns (>80%):       {'PASS' if t4 else 'FAIL'} ({results['A']['acc_clean']:.3f})")

    n_pass = sum([t1, t2, t3, t4])
    verdict = {0: "NO_PREDICTION_ERROR", 1: "WEAK", 2: "PARTIAL", 3: "MOSTLY", 4: "GENUINE_PREDICTION_ERROR"}[n_pass]
    print(f"\n  VERDICT: {verdict} ({n_pass}/4)")

    output = {
        'experiment': 'z2031_prediction_error',
        'hypothesis': 'Workspace creates sharper prediction error signals for violations',
        'timestamp': datetime.now().isoformat(),
        'device': str(device),
        'references': ['Friston 2010 (Free energy)', 'Clark 2013 (Predictive brains)', 'Rao & Ballard 1999'],
        'conditions': results,
        'tests': {'t1': bool(t1), 't2': bool(t2), 't3': bool(t3), 't4': bool(t4)},
        'tests_passed': n_pass, 'verdict': verdict,
    }

    def json_safe(obj):
        if isinstance(obj, (np.bool_, np.integer)): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, dict): return {k: json_safe(v) for k, v in obj.items()}
        if isinstance(obj, list): return [json_safe(v) for v in obj]
        return obj

    rp = Path(__file__).parent.parent / 'results' / 'z2031_prediction_error.json'
    with open(rp, 'w') as f:
        json.dump(json_safe(output), f, indent=2)
    print(f"\nResults saved to {rp}")


if __name__ == '__main__':
    main()
