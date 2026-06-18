#!/usr/bin/env python3
"""
z904 - Fatigue & Recovery Dynamics in Energy-Aware Transformers

Scientific experiment testing biologically-inspired fatigue dynamics.

Hypothesis: Fatigue dynamics (activation scaling decays with cumulative energy,
recovers during idle) produce better energy-quality Pareto curves than fixed networks.

Architecture: 6-layer transformer (256 hidden, 4 heads, ~500K params)
with per-layer fatigue state:
    f_i(t+1) = f_i(t) + alpha_i * E_measured   (fatigue accumulates)
    f_i(t+1) = f_i(t) * decay                  (recovery during idle)
    h_i = h_i * (1 - gamma_i * f_i)            (fatigued neurons fire less)

alpha_i, gamma_i are learnable per layer. decay is a hyperparameter.

Conditions:
    A: No fatigue (standard network)
    B: Random fatigue (not correlated with energy)
    C: Fixed fatigue schedule (time-based, not energy-based)
    D: Real energy-linked fatigue (the hypothesis)

Metrics: Perplexity vs fatigue state, learned alpha/gamma values,
         energy efficiency over long runs, recovery dynamics after idle.
"""

import os
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '11.0.0'

import sys
import json
import time
import math
import copy
import random
import argparse
import traceback
from pathlib import Path
from datetime import datetime

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# --------------------------------------------------------------------------- #
# Telemetry imports with fallback
# --------------------------------------------------------------------------- #
HAVE_TELEMETRY = False
try:
    from src.telemetry.sysfs_hwmon import SysfsHwmonTelemetry, EnergyMeter
    HAVE_TELEMETRY = True
except Exception as e:
    print(f"[warn] sysfs_hwmon telemetry unavailable: {e}")
    print("[warn] Falling back to wall-clock energy proxy")

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
CKPT_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# TinyShakespeare download
# --------------------------------------------------------------------------- #
SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def download_tiny_shakespeare() -> str:
    """Download TinyShakespeare if not present, return text."""
    fpath = DATA_DIR / "tinyshakespeare_input.txt"
    if fpath.exists():
        print(f"[data] Using cached {fpath} ({fpath.stat().st_size / 1024:.0f} KB)")
        return fpath.read_text()

    print(f"[data] Downloading TinyShakespeare ...")
    import urllib.request
    try:
        urllib.request.urlretrieve(SHAKESPEARE_URL, str(fpath))
        text = fpath.read_text()
        print(f"[data] Downloaded {len(text)} chars")
        return text
    except Exception as e:
        print(f"[data] Download failed: {e}, generating synthetic data")
        # Fallback: generate repeating ASCII text
        synth = ("to be or not to be that is the question "
                 "whether tis nobler in the mind to suffer "
                 "the slings and arrows of outrageous fortune\n") * 5000
        fpath.write_text(synth)
        return synth


# --------------------------------------------------------------------------- #
# Char-level dataset
# --------------------------------------------------------------------------- #
class CharDataset(Dataset):
    """Character-level language modelling dataset."""

    def __init__(self, text: str, seq_len: int = 256):
        self.seq_len = seq_len
        # byte-level encoding: every char -> its ordinal (0-255)
        self.data = torch.tensor([ord(c) % 256 for c in text], dtype=torch.long)
        self.n_samples = max(1, (len(self.data) - 1) // seq_len)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len + 1
        chunk = self.data[start:end]
        if len(chunk) < self.seq_len + 1:
            # pad with zeros
            pad = torch.zeros(self.seq_len + 1 - len(chunk), dtype=torch.long)
            chunk = torch.cat([chunk, pad])
        return chunk[:self.seq_len], chunk[1:self.seq_len + 1]


# --------------------------------------------------------------------------- #
# Fatigue Transformer
# --------------------------------------------------------------------------- #
class FatigueTransformerBlock(nn.Module):
    """Transformer block with per-layer fatigue state."""

    def __init__(self, hidden_dim: int, num_heads: int, ff_dim: int, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx

        # Standard pre-norm transformer block components
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, hidden_dim),
        )

        # Learnable fatigue parameters
        self.alpha = nn.Parameter(torch.tensor(0.001))  # fatigue accumulation rate
        self.gamma = nn.Parameter(torch.tensor(0.1))    # fatigue impact on activation

        # State (not a parameter -- runtime bookkeeping)
        self.fatigue_state: float = 0.0

    def forward(self, x, mask=None, energy_measured: float = 0.0,
                is_idle: bool = False, decay: float = 0.95):
        # Update fatigue
        if is_idle:
            self.fatigue_state *= decay
        else:
            self.fatigue_state = self.fatigue_state + self.alpha.abs().item() * energy_measured

        # Clamp fatigue to prevent runaway
        f = min(self.fatigue_state, 10.0)
        fatigue_scale = 1.0 - self.gamma.abs().item() * f
        fatigue_scale = max(fatigue_scale, 0.1)  # floor at 10%

        # Standard transformer with fatigue modulation
        h = self.ln1(x)
        h, _ = self.attn(h, h, h, attn_mask=mask)
        x = x + h * fatigue_scale

        h = self.ln2(x)
        h = self.ffn(h)
        x = x + h * fatigue_scale
        return x


class FatigueTransformer(nn.Module):
    """Char-level transformer with biologically-inspired fatigue dynamics."""

    def __init__(self, vocab_size: int = 256, hidden_dim: int = 256,
                 num_layers: int = 6, num_heads: int = 4, ff_dim: int = 1024,
                 max_seq_len: int = 256):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len

        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embed = nn.Embedding(max_seq_len, hidden_dim)
        self.blocks = nn.ModuleList([
            FatigueTransformerBlock(hidden_dim, num_heads, ff_dim, i)
            for i in range(num_layers)
        ])
        self.ln_out = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids, energy_measured: float = 0.0,
                is_idle: bool = False, decay: float = 0.95):
        B, T = input_ids.shape
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.embed(input_ids) + self.pos_embed(positions)

        # Causal mask
        mask = torch.triu(torch.ones(T, T, device=input_ids.device), diagonal=1).bool()

        for block in self.blocks:
            x = block(x, mask, energy_measured, is_idle, decay)

        return self.head(self.ln_out(x))

    def reset_fatigue(self):
        for block in self.blocks:
            block.fatigue_state = 0.0

    def get_fatigue_states(self) -> list:
        return [block.fatigue_state for block in self.blocks]

    def get_fatigue_params(self) -> list:
        return [(block.alpha.item(), block.gamma.item()) for block in self.blocks]


# --------------------------------------------------------------------------- #
# Energy measurement helpers
# --------------------------------------------------------------------------- #
class WallClockEnergyProxy:
    """Fallback energy proxy when GPU telemetry is unavailable."""

    def __init__(self):
        self.start_time = 0.0
        self.energy_j = 0.0
        self.duration_s = 0.0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.duration_s = time.perf_counter() - self.start_time
        # Estimate ~50W average GPU power as proxy
        self.energy_j = self.duration_s * 50.0


def make_energy_meter(telemetry):
    """Create appropriate energy meter based on available telemetry."""
    if telemetry is not None:
        return EnergyMeter(telemetry)
    return WallClockEnergyProxy()


# --------------------------------------------------------------------------- #
# Training loop for one condition
# --------------------------------------------------------------------------- #
def train_condition(condition: str, model: FatigueTransformer, train_loader: DataLoader,
                    val_loader: DataLoader, device: torch.device, epochs: int,
                    decay: float, telemetry, lr: float = 3e-4) -> dict:
    """
    Train under a given fatigue condition.

    Conditions:
      A - no fatigue: energy_measured always 0
      B - random fatigue: random energy values
      C - fixed schedule: time-based monotonic fatigue signal
      D - real energy fatigue: actual GPU energy measurement
    """
    print(f"\n{'='*70}")
    print(f"  CONDITION {condition}")
    desc = {
        'A': 'No fatigue (standard network)',
        'B': 'Random fatigue (not correlated with energy)',
        'C': 'Fixed fatigue schedule (time-based)',
        'D': 'Real energy-linked fatigue (hypothesis)',
    }
    print(f"  {desc.get(condition, 'Unknown')}")
    print(f"{'='*70}")

    model.reset_fatigue()
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    total_batches = len(train_loader) * epochs
    global_step = 0

    history = {
        'condition': condition,
        'description': desc.get(condition, ''),
        'epoch_metrics': [],
        'batch_energy': [],
        'fatigue_traces': [],
        'learned_params_trace': [],
        'recovery_events': [],
    }

    total_energy_j = 0.0
    best_val_ppl = float('inf')

    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        epoch_loss = 0.0
        epoch_tokens = 0
        epoch_energy = 0.0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Determine energy signal for this batch
            if condition == 'A':
                energy_signal = 0.0
            elif condition == 'B':
                energy_signal = random.uniform(0.0, 2.0)
            elif condition == 'C':
                # Monotonic time-based: fraction of training completed
                energy_signal = global_step / max(total_batches, 1)
            else:
                # Condition D: measure real energy
                energy_signal = 0.0  # will be updated below

            # Measure real energy for all conditions (for comparison), use it for D
            meter = make_energy_meter(telemetry)
            with meter:
                logits = model(inputs, energy_measured=energy_signal,
                               is_idle=False, decay=decay)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                       targets.view(-1))
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            batch_energy_j = meter.energy_j

            # For condition D, feed the real energy back
            if condition == 'D':
                # Re-run fatigue update with actual energy (retroactive)
                for block in model.blocks:
                    block.fatigue_state = (block.fatigue_state
                                           + block.alpha.abs().item() * batch_energy_j)
                    block.fatigue_state = min(block.fatigue_state, 10.0)

            n_tokens = inputs.numel()
            epoch_loss += loss.item() * n_tokens
            epoch_tokens += n_tokens
            epoch_energy += batch_energy_j
            total_energy_j += batch_energy_j
            global_step += 1

            # Record fatigue traces periodically
            if batch_idx % 20 == 0:
                states = model.get_fatigue_states()
                params = model.get_fatigue_params()
                history['fatigue_traces'].append({
                    'epoch': epoch,
                    'batch': batch_idx,
                    'global_step': global_step,
                    'fatigue_states': [float(s) for s in states],
                    'energy_signal': float(energy_signal),
                    'batch_energy_j': float(batch_energy_j),
                })
                history['learned_params_trace'].append({
                    'epoch': epoch,
                    'batch': batch_idx,
                    'global_step': global_step,
                    'alphas': [float(p[0]) for p in params],
                    'gammas': [float(p[1]) for p in params],
                })

            history['batch_energy'].append({
                'step': global_step,
                'energy_j': float(batch_energy_j),
            })

        scheduler.step()

        # End-of-epoch: simulate idle period and recovery
        print(f"  [epoch {epoch+1}] Simulating idle recovery (2s sleep) ...")
        pre_recovery_states = model.get_fatigue_states()

        time.sleep(2)

        # Apply idle recovery for several virtual "steps"
        n_recovery_steps = 20
        for _ in range(n_recovery_steps):
            for block in model.blocks:
                block.fatigue_state *= decay

        post_recovery_states = model.get_fatigue_states()

        recovery_event = {
            'epoch': epoch,
            'pre_recovery': [float(s) for s in pre_recovery_states],
            'post_recovery': [float(s) for s in post_recovery_states],
            'recovery_steps': n_recovery_steps,
            'decay': decay,
        }
        history['recovery_events'].append(recovery_event)

        # Validation
        val_ppl, val_loss = evaluate(model, val_loader, device, decay)

        epoch_ppl = math.exp(min(epoch_loss / max(epoch_tokens, 1), 20.0))
        epoch_time = time.time() - epoch_start

        epoch_record = {
            'epoch': epoch + 1,
            'train_loss': epoch_loss / max(epoch_tokens, 1),
            'train_ppl': float(epoch_ppl),
            'val_loss': float(val_loss),
            'val_ppl': float(val_ppl),
            'epoch_energy_j': float(epoch_energy),
            'cumulative_energy_j': float(total_energy_j),
            'epoch_time_s': float(epoch_time),
            'fatigue_states': [float(s) for s in model.get_fatigue_states()],
            'learned_params': {
                'alphas': [float(p[0]) for p in model.get_fatigue_params()],
                'gammas': [float(p[1]) for p in model.get_fatigue_params()],
            },
        }
        history['epoch_metrics'].append(epoch_record)

        print(f"  [epoch {epoch+1}/{epochs}] "
              f"train_ppl={epoch_ppl:.2f}  val_ppl={val_ppl:.2f}  "
              f"energy={epoch_energy:.1f}J  cumul={total_energy_j:.1f}J  "
              f"time={epoch_time:.1f}s")
        print(f"    fatigue states: {[f'{s:.4f}' for s in model.get_fatigue_states()]}")
        print(f"    alphas: {[f'{p[0]:.6f}' for p in model.get_fatigue_params()]}")
        print(f"    gammas: {[f'{p[1]:.6f}' for p in model.get_fatigue_params()]}")

        # Checkpoint best
        if val_ppl < best_val_ppl:
            best_val_ppl = val_ppl
            ckpt_path = CKPT_DIR / f"z904_condition_{condition}_best.pt"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'val_ppl': val_ppl,
                'fatigue_states': model.get_fatigue_states(),
                'condition': condition,
            }, ckpt_path)
            print(f"    -> saved best checkpoint (val_ppl={val_ppl:.2f})")

    history['best_val_ppl'] = float(best_val_ppl)
    history['total_energy_j'] = float(total_energy_j)
    history['final_fatigue_states'] = [float(s) for s in model.get_fatigue_states()]
    history['final_learned_params'] = {
        'alphas': [float(p[0]) for p in model.get_fatigue_params()],
        'gammas': [float(p[1]) for p in model.get_fatigue_params()],
    }

    return history


def evaluate(model: FatigueTransformer, val_loader: DataLoader,
             device: torch.device, decay: float) -> tuple:
    """Evaluate model; return (perplexity, avg_loss)."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs, energy_measured=0.0, is_idle=False, decay=decay)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1), reduction='sum')
            total_loss += loss.item()
            total_tokens += targets.numel()

    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(min(avg_loss, 20.0))
    return ppl, avg_loss


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="z904: Fatigue & Recovery Dynamics Experiment")
    parser.add_argument('--epochs', type=int, default=15,
                        help='Training epochs per condition (default: 15)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--seq-len', type=int, default=256,
                        help='Sequence length (default: 256)')
    parser.add_argument('--decay', type=float, default=0.95,
                        help='Fatigue decay rate during idle (default: 0.95)')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device: auto, cuda, cpu (default: auto)')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate (default: 3e-4)')
    parser.add_argument('--conditions', type=str, default='A,B,C,D',
                        help='Comma-separated conditions to run (default: A,B,C,D)')
    args = parser.parse_args()

    print("=" * 70)
    print("  z904: Fatigue & Recovery Dynamics in Energy-Aware Transformers")
    print("=" * 70)
    print(f"  Epochs:     {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Seq len:    {args.seq_len}")
    print(f"  Decay:      {args.decay}")
    print(f"  LR:         {args.lr}")
    print(f"  Conditions: {args.conditions}")
    print(f"  Telemetry:  {'sysfs_hwmon' if HAVE_TELEMETRY else 'wall-clock proxy'}")

    # Device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"  Device:     {device}")
    if device.type == 'cuda':
        print(f"  GPU:        {torch.cuda.get_device_name(0)}")
        print(f"  VRAM:       {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Telemetry
    telemetry = None
    if HAVE_TELEMETRY and device.type == 'cuda':
        try:
            telemetry = SysfsHwmonTelemetry(sample_rate_hz=50)
            sample = telemetry.read_sample()
            print(f"  GPU Power:  {sample.power_w:.1f} W")
            print(f"  GPU Temp:   {sample.temp_edge_c:.1f} C")
        except Exception as e:
            print(f"  [warn] Telemetry init failed: {e}")
            telemetry = None

    # Data
    print(f"\n--- Data Loading ---")
    text = download_tiny_shakespeare()
    print(f"  Total chars: {len(text):,}")

    # Split 90/10
    split_idx = int(len(text) * 0.9)
    train_text = text[:split_idx]
    val_text = text[split_idx:]

    train_ds = CharDataset(train_text, seq_len=args.seq_len)
    val_ds = CharDataset(val_text, seq_len=args.seq_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=(device.type == 'cuda'))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=(device.type == 'cuda'))

    print(f"  Train samples: {len(train_ds)}")
    print(f"  Val samples:   {len(val_ds)}")
    print(f"  Train batches: {len(train_loader)}")

    # Build base model to count params
    base_model = FatigueTransformer(
        vocab_size=256,
        hidden_dim=256,
        num_layers=6,
        num_heads=4,
        ff_dim=1024,
        max_seq_len=args.seq_len,
    )
    n_params = sum(p.numel() for p in base_model.parameters())
    print(f"  Model params:  {n_params:,} ({n_params/1e6:.2f}M)")
    del base_model

    # Run conditions
    conditions = [c.strip().upper() for c in args.conditions.split(',')]
    all_results = {
        'experiment': 'z904_fatigue_recovery',
        'timestamp': datetime.now().isoformat(),
        'config': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'seq_len': args.seq_len,
            'decay': args.decay,
            'lr': args.lr,
            'n_params': n_params,
            'device': str(device),
            'gpu': torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu',
            'telemetry': 'sysfs_hwmon' if telemetry else 'wall-clock proxy',
        },
        'conditions': {},
        'summary': {},
    }

    for cond in conditions:
        if cond not in ('A', 'B', 'C', 'D'):
            print(f"[warn] Skipping unknown condition '{cond}'")
            continue

        # Fresh model for each condition (same init via seed)
        torch.manual_seed(42)
        random.seed(42)
        np.random.seed(42)

        model = FatigueTransformer(
            vocab_size=256,
            hidden_dim=256,
            num_layers=6,
            num_heads=4,
            ff_dim=1024,
            max_seq_len=args.seq_len,
        )

        try:
            result = train_condition(
                condition=cond,
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                epochs=args.epochs,
                decay=args.decay,
                telemetry=telemetry,
                lr=args.lr,
            )
            all_results['conditions'][cond] = result
        except Exception as e:
            print(f"\n[ERROR] Condition {cond} failed: {e}")
            traceback.print_exc()
            all_results['conditions'][cond] = {
                'condition': cond,
                'error': str(e),
            }
        finally:
            del model
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    # --------------------------------------------------------------------------- #
    # Summary analysis
    # --------------------------------------------------------------------------- #
    print("\n" + "=" * 70)
    print("  SUMMARY: Fatigue & Recovery Dynamics")
    print("=" * 70)

    summary_rows = []
    for cond in conditions:
        r = all_results['conditions'].get(cond, {})
        if 'error' in r:
            summary_rows.append({
                'condition': cond,
                'best_val_ppl': None,
                'total_energy_j': None,
                'status': 'FAILED',
            })
            continue

        best_ppl = r.get('best_val_ppl', float('inf'))
        total_e = r.get('total_energy_j', 0.0)
        final_states = r.get('final_fatigue_states', [])
        final_params = r.get('final_learned_params', {})

        efficiency = best_ppl / max(total_e, 0.001) if total_e > 0 else 0.0

        summary_rows.append({
            'condition': cond,
            'best_val_ppl': best_ppl,
            'total_energy_j': total_e,
            'ppl_per_joule': efficiency,
            'final_avg_fatigue': float(np.mean(final_states)) if final_states else 0.0,
            'final_alphas': final_params.get('alphas', []),
            'final_gammas': final_params.get('gammas', []),
            'status': 'OK',
        })

    # Print table
    print(f"\n{'Cond':<6} {'Best Val PPL':>13} {'Energy (J)':>11} {'PPL/J':>10} "
          f"{'Avg Fatigue':>12} {'Status':<8}")
    print("-" * 65)
    for row in summary_rows:
        if row['status'] == 'FAILED':
            print(f"{row['condition']:<6} {'---':>13} {'---':>11} {'---':>10} "
                  f"{'---':>12} {'FAILED':<8}")
        else:
            print(f"{row['condition']:<6} {row['best_val_ppl']:>13.2f} "
                  f"{row['total_energy_j']:>11.1f} {row['ppl_per_joule']:>10.4f} "
                  f"{row['final_avg_fatigue']:>12.6f} {'OK':<8}")

    # Pareto analysis
    print(f"\n--- Pareto Analysis (lower PPL is better, lower energy is better) ---")
    valid_rows = [r for r in summary_rows if r['status'] == 'OK']
    if len(valid_rows) >= 2:
        # Sort by energy
        sorted_by_e = sorted(valid_rows, key=lambda r: r['total_energy_j'])
        # Check if D is on the Pareto frontier
        pareto_front = []
        best_ppl_so_far = float('inf')
        for r in sorted_by_e:
            if r['best_val_ppl'] < best_ppl_so_far:
                pareto_front.append(r['condition'])
                best_ppl_so_far = r['best_val_ppl']

        print(f"  Pareto-optimal conditions: {pareto_front}")
        hypothesis_supported = 'D' in pareto_front
        print(f"  Hypothesis (energy-linked fatigue on Pareto front): "
              f"{'SUPPORTED' if hypothesis_supported else 'NOT SUPPORTED'}")
        all_results['summary']['pareto_front'] = pareto_front
        all_results['summary']['hypothesis_supported'] = hypothesis_supported

    # Learned parameter analysis for condition D
    if 'D' in all_results['conditions'] and 'error' not in all_results['conditions']['D']:
        d_result = all_results['conditions']['D']
        final_p = d_result.get('final_learned_params', {})
        alphas = final_p.get('alphas', [])
        gammas = final_p.get('gammas', [])
        if alphas and gammas:
            print(f"\n--- Learned Parameters (Condition D) ---")
            for i, (a, g) in enumerate(zip(alphas, gammas)):
                print(f"  Layer {i}: alpha={a:.6f}  gamma={g:.6f}")
            print(f"  Mean alpha: {np.mean(alphas):.6f}  (fatigue accumulation rate)")
            print(f"  Mean gamma: {np.mean(gammas):.6f}  (fatigue impact strength)")

            # Check if deeper layers learned different fatigue dynamics
            if len(alphas) >= 4:
                early_alpha = np.mean(alphas[:len(alphas)//2])
                late_alpha = np.mean(alphas[len(alphas)//2:])
                print(f"  Early-layer avg alpha: {early_alpha:.6f}")
                print(f"  Late-layer avg alpha:  {late_alpha:.6f}")
                all_results['summary']['alpha_gradient'] = {
                    'early': float(early_alpha),
                    'late': float(late_alpha),
                    'ratio': float(late_alpha / max(early_alpha, 1e-8)),
                }

    # Recovery analysis
    print(f"\n--- Recovery Dynamics ---")
    for cond in conditions:
        r = all_results['conditions'].get(cond, {})
        events = r.get('recovery_events', [])
        if not events:
            continue
        # Average recovery ratio across all epochs
        ratios = []
        for ev in events:
            pre = ev['pre_recovery']
            post = ev['post_recovery']
            for p, q in zip(pre, post):
                if p > 1e-8:
                    ratios.append(q / p)
        if ratios:
            avg_ratio = np.mean(ratios)
            print(f"  Condition {cond}: avg recovery ratio = {avg_ratio:.4f} "
                  f"(1.0 = no recovery, 0.0 = full recovery)")
            all_results['summary'][f'recovery_ratio_{cond}'] = float(avg_ratio)

    all_results['summary']['conditions_run'] = conditions
    all_results['summary']['table'] = summary_rows

    # Save results
    results_path = RESULTS_DIR / "z904_fatigue_recovery.json"
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n[saved] Results -> {results_path}")

    # Final checkpoint with all condition summaries
    final_ckpt = CKPT_DIR / "z904_experiment_summary.pt"
    torch.save({
        'summary': all_results['summary'],
        'config': all_results['config'],
    }, final_ckpt)
    print(f"[saved] Summary checkpoint -> {final_ckpt}")

    print(f"\n{'='*70}")
    print(f"  z904 experiment complete.")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
