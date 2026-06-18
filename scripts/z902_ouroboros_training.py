#!/usr/bin/env python3
"""
z902 Ouroboros Self-Referential Training Experiment
====================================================

Hypothesis: A model trained on its own output generated under different
hardware conditions develops internal representations that encode
hardware-dependent variations and can self-correct.

Architecture: MetabolicTransformer (FiLM-conditioned char-level LM)

Three phases:
  Phase 1: Train char-level LM normally on TinyShakespeare (baseline task learning)
  Phase 2: Generate text under varied thermal conditions (warm GPU vs cool GPU)
  Phase 3: Train with Ouroboros loss = L_task + lambda * L_self_state
           where L_self_state = BCE(classifier(hidden_states), thermal_bin)

Controls:
  A: Phase 1 only (no self-referential learning)
  B: Ouroboros with RANDOM thermal labels (shuffled)
  C: Ouroboros with all data from SAME thermal condition (no contrast)

Metrics:
  - Self-state classification accuracy
  - Task perplexity stability across temperatures
  - Hidden state CKA (hot vs cold)
  - Output quality gap reduction

Author: FEEL Research Team
Date: 2026-01-28
"""

import os
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '11.0.0'

import sys
import json
import time
import math
import argparse
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from src.metabolic.film_transformer import MetabolicTransformer, MetabolicConfig
try:
    from src.telemetry.sysfs_hwmon import SysfsHwmonTelemetry, EnergyMeter
    HAS_SYSFS = True
except Exception:
    HAS_SYSFS = False


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def get_device(requested: str = 'auto') -> torch.device:
    """Select best available device with robust fallback."""
    if requested != 'auto':
        return torch.device(requested)
    if torch.cuda.is_available():
        try:
            t = torch.zeros(1, device='cuda')
            del t
            return torch.device('cuda')
        except Exception:
            pass
    return torch.device('cpu')


def read_gpu_temp() -> float:
    """Read GPU edge temperature in Celsius.  Returns 0.0 on failure."""
    if HAS_SYSFS:
        try:
            telem = _GLOBAL_TELEMETRY
            sample = telem.read_sample()
            return sample.temp_edge_c
        except Exception:
            pass
    # Fallback: try direct sysfs read
    for hwmon in sorted(Path('/sys/class/drm').glob('card*/device/hwmon/hwmon*/temp1_input')):
        try:
            return int(hwmon.read_text().strip()) / 1000.0
        except Exception:
            continue
    return 0.0


def make_telemetry_vector(device: torch.device) -> torch.Tensor:
    """Build a 12-dim telemetry vector from live hardware (or zeros)."""
    vec = torch.zeros(12, device=device)
    if HAS_SYSFS:
        try:
            sample = _GLOBAL_TELEMETRY.read_sample()
            vec[0] = sample.power_w / 100.0          # normalised power
            vec[1] = sample.temp_edge_c / 100.0      # normalised temp
            vec[2] = sample.freq_sclk_mhz / 3000.0   # normalised clock
            vec[3] = sample.gpu_busy_pct / 100.0      # utilisation
        except Exception:
            pass
    return vec


# Global telemetry handle (initialised lazily)
_GLOBAL_TELEMETRY = None


def init_telemetry():
    """Initialise the global sysfs telemetry object if possible."""
    global _GLOBAL_TELEMETRY
    if HAS_SYSFS and _GLOBAL_TELEMETRY is None:
        try:
            _GLOBAL_TELEMETRY = SysfsHwmonTelemetry(sample_rate_hz=10)
            print(f"[telemetry] sysfs hwmon initialised  "
                  f"power={_GLOBAL_TELEMETRY.paths.power_average}")
        except Exception as e:
            print(f"[telemetry] sysfs hwmon unavailable: {e}")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def download_tiny_shakespeare() -> str:
    """Download TinyShakespeare if not cached; return the text."""
    data_path = Path(__file__).parent.parent / 'data' / 'tiny_shakespeare.txt'
    data_path.parent.mkdir(parents=True, exist_ok=True)
    if not data_path.exists():
        import urllib.request
        url = ('https://raw.githubusercontent.com/karpathy/char-rnn/'
               'master/data/tinyshakespeare/input.txt')
        print(f"[data] Downloading TinyShakespeare -> {data_path}")
        urllib.request.urlretrieve(url, data_path)
    text = data_path.read_text(encoding='utf-8')
    print(f"[data] Loaded {len(text):,} characters from TinyShakespeare")
    return text


class CharDataset(Dataset):
    """Character-level language-modelling dataset."""

    def __init__(self, text: str, seq_len: int = 128):
        self.data = torch.tensor(
            [ord(c) % 256 for c in text], dtype=torch.long
        )
        self.seq_len = seq_len

    def __len__(self):
        return max(1, len(self.data) - self.seq_len - 1)

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.seq_len]
        y = self.data[idx + 1: idx + self.seq_len + 1]
        return x, y


# ---------------------------------------------------------------------------
# Self-State Classifier (used in Phase 3)
# ---------------------------------------------------------------------------

class SelfStateClassifier(nn.Module):
    """Classifies mean-pooled hidden states -> thermal bin (0=cool, 1=warm)."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: [batch, hidden_dim] mean-pooled hidden states -> [batch, 1]"""
        return self.fc(h)


# ---------------------------------------------------------------------------
# CKA (Centered Kernel Alignment)
# ---------------------------------------------------------------------------

def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute linear CKA between two representation matrices.

    X, Y: [n_samples, dim]
    Returns CKA in [0, 1].
    """
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    hsic_xy = np.linalg.norm(X.T @ Y, 'fro') ** 2
    hsic_xx = np.linalg.norm(X.T @ X, 'fro') ** 2
    hsic_yy = np.linalg.norm(Y.T @ Y, 'fro') ** 2
    denom = math.sqrt(hsic_xx * hsic_yy)
    if denom < 1e-12:
        return 0.0
    return float(hsic_xy / denom)


# ---------------------------------------------------------------------------
# Thermal Cycling for Phase 2
# ---------------------------------------------------------------------------

def warm_gpu(device: torch.device, duration_s: float = 8.0):
    """Heat the GPU with heavy matmuls."""
    if device.type != 'cuda':
        return
    print("  [thermal] Warming GPU ...", end='', flush=True)
    a = torch.randn(2048, 2048, device=device)
    t0 = time.time()
    while time.time() - t0 < duration_s:
        _ = a @ a
        torch.cuda.synchronize()
    del a
    torch.cuda.empty_cache()
    temp = read_gpu_temp()
    print(f" done  temp={temp:.1f}C")


def cool_gpu(device: torch.device, target_c: float = 50.0,
             timeout_s: float = 60.0):
    """Wait for GPU to cool below target temperature."""
    if device.type != 'cuda':
        return
    print(f"  [thermal] Cooling GPU (target <{target_c:.0f}C) ...",
          end='', flush=True)
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        temp = read_gpu_temp()
        if temp > 0 and temp < target_c:
            break
        time.sleep(1.0)
    temp = read_gpu_temp()
    print(f" done  temp={temp:.1f}C")


@torch.no_grad()
def generate_under_thermal(
    model: MetabolicTransformer,
    dataset: CharDataset,
    device: torch.device,
    n_samples: int,
    seq_len: int,
    condition: str,  # 'warm' or 'cool'
) -> List[Dict]:
    """Generate text samples and capture hidden states under a thermal condition.

    Returns list of dicts with keys:
        input_ids, output_ids, hidden_mean, gpu_temp, condition
    """
    model.eval()
    results = []

    for i in range(n_samples):
        # Pick a random prompt from the dataset
        idx = np.random.randint(0, len(dataset))
        x, _ = dataset[idx]
        x = x.unsqueeze(0).to(device)  # [1, seq_len]

        telemetry = make_telemetry_vector(device).unsqueeze(0)
        gpu_temp = read_gpu_temp()

        out = model(x, telemetry=telemetry, return_hidden=True)
        hidden = out['hidden']          # [1, seq_len, hidden_dim]
        h_mean = hidden.mean(dim=1)     # [1, hidden_dim]

        # Greedy decode one token
        logits = out['logits'][:, -1, :]
        pred_token = logits.argmax(dim=-1)

        results.append({
            'input_ids': x.cpu(),
            'hidden_mean': h_mean.cpu(),
            'gpu_temp': gpu_temp,
            'condition': condition,
        })

        if (i + 1) % 50 == 0:
            print(f"    generated {i + 1}/{n_samples}  "
                  f"temp={gpu_temp:.1f}C  cond={condition}")

    return results


# ---------------------------------------------------------------------------
# Training Helpers
# ---------------------------------------------------------------------------

def train_phase1(
    model: MetabolicTransformer,
    dataset: CharDataset,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float = 3e-4,
) -> List[Dict]:
    """Phase 1: Standard char-level LM training on TinyShakespeare."""
    print("\n" + "=" * 70)
    print("PHASE 1 : Standard Language Model Training")
    print("=" * 70)

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        drop_last=True, num_workers=0)
    history = []

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        n_batches = 0
        t0 = time.time()

        for x, y in loader:
            x, y = x.to(device), y.to(device)
            telemetry = make_telemetry_vector(device).unsqueeze(0).expand(
                x.size(0), -1
            )
            out = model(x, telemetry=telemetry)
            logits = out['logits']  # [B, seq, vocab]
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), y.view(-1)
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        ppl = math.exp(min(avg_loss, 20))
        elapsed = time.time() - t0
        record = {
            'epoch': epoch, 'loss': avg_loss, 'perplexity': ppl,
            'time_s': elapsed,
        }
        history.append(record)
        print(f"  Epoch {epoch}/{epochs}  loss={avg_loss:.4f}  "
              f"ppl={ppl:.2f}  time={elapsed:.1f}s")

    return history


def train_phase3(
    model: MetabolicTransformer,
    classifier: SelfStateClassifier,
    dataset: CharDataset,
    phase2_data: List[Dict],
    thermal_labels: torch.Tensor,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lambda_self: float,
    lr: float = 3e-4,
    condition_name: str = "Ouroboros",
) -> List[Dict]:
    """Phase 3: Ouroboros training with self-state loss.

    L = L_task + lambda * L_self_state
    L_self_state = BCE(classifier(hidden_mean), thermal_label)
    """
    print(f"\n{'=' * 70}")
    print(f"PHASE 3 : {condition_name}")
    print(f"{'=' * 70}")

    model.train()
    classifier.train()

    params = list(model.parameters()) + list(classifier.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        drop_last=True, num_workers=0)

    # Pre-process phase2 hidden / label tensors
    hidden_bank = torch.cat([d['hidden_mean'] for d in phase2_data], dim=0)
    hidden_bank = hidden_bank.to(device)     # [N, hidden_dim]
    label_bank = thermal_labels.float().to(device)  # [N]

    history = []

    for epoch in range(1, epochs + 1):
        task_loss_sum = 0.0
        self_loss_sum = 0.0
        total_loss_sum = 0.0
        correct = 0
        n_self_samples = 0
        n_batches = 0
        t0 = time.time()

        for x, y in loader:
            x, y = x.to(device), y.to(device)
            B = x.size(0)

            # --- Task loss ---
            telemetry = make_telemetry_vector(device).unsqueeze(0).expand(B, -1)
            out = model(x, telemetry=telemetry, return_hidden=True)
            logits = out['logits']
            l_task = F.cross_entropy(
                logits.view(-1, logits.size(-1)), y.view(-1)
            )

            # --- Self-state loss (sample from hidden bank) ---
            idx = torch.randint(0, hidden_bank.size(0), (B,))
            h_sample = hidden_bank[idx]        # [B, hidden_dim]
            lbl = label_bank[idx].unsqueeze(1)  # [B, 1]

            pred = classifier(h_sample)         # [B, 1]
            l_self = F.binary_cross_entropy_with_logits(pred, lbl)

            # --- Also classify current hidden states (online) ---
            h_current = out['hidden'].mean(dim=1).detach()
            pred_current = classifier(h_current)
            # No label for current -- just use bank samples for loss.

            # --- Combined ---
            loss = l_task + lambda_self * l_self

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

            task_loss_sum += l_task.item()
            self_loss_sum += l_self.item()
            total_loss_sum += loss.item()
            correct += ((pred > 0).float() == lbl).sum().item()
            n_self_samples += B
            n_batches += 1

        avg_task = task_loss_sum / max(n_batches, 1)
        avg_self = self_loss_sum / max(n_batches, 1)
        avg_total = total_loss_sum / max(n_batches, 1)
        accuracy = correct / max(n_self_samples, 1)
        ppl = math.exp(min(avg_task, 20))
        elapsed = time.time() - t0

        record = {
            'epoch': epoch,
            'task_loss': avg_task,
            'self_loss': avg_self,
            'total_loss': avg_total,
            'perplexity': ppl,
            'self_state_accuracy': accuracy,
            'time_s': elapsed,
        }
        history.append(record)
        print(f"  Epoch {epoch}/{epochs}  "
              f"task={avg_task:.4f}  self={avg_self:.4f}  "
              f"ppl={ppl:.2f}  self_acc={accuracy:.3f}  "
              f"time={elapsed:.1f}s")

    return history


# ---------------------------------------------------------------------------
# Evaluation Helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_perplexity(model: MetabolicTransformer, dataset: CharDataset,
                        device: torch.device, batch_size: int,
                        thermal_mode: str = 'neutral') -> float:
    """Evaluate perplexity on the dataset."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        drop_last=True, num_workers=0)
    total_loss = 0.0
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        telemetry = make_telemetry_vector(device).unsqueeze(0).expand(
            x.size(0), -1
        )
        out = model(x, telemetry=telemetry)
        loss = F.cross_entropy(
            out['logits'].view(-1, out['logits'].size(-1)), y.view(-1)
        )
        total_loss += loss.item()
        n += 1
    avg = total_loss / max(n, 1)
    return math.exp(min(avg, 20))


@torch.no_grad()
def collect_hidden_states(
    model: MetabolicTransformer,
    dataset: CharDataset,
    device: torch.device,
    n_samples: int = 100,
) -> np.ndarray:
    """Collect mean-pooled hidden states for CKA analysis."""
    model.eval()
    hiddens = []
    for i in range(n_samples):
        idx = np.random.randint(0, len(dataset))
        x, _ = dataset[idx]
        x = x.unsqueeze(0).to(device)
        telemetry = make_telemetry_vector(device).unsqueeze(0)
        out = model(x, telemetry=telemetry, return_hidden=True)
        h = out['hidden'].mean(dim=1).cpu().numpy()  # [1, hidden_dim]
        hiddens.append(h)
    return np.concatenate(hiddens, axis=0)  # [n_samples, hidden_dim]


@torch.no_grad()
def evaluate_self_state_accuracy(
    classifier: SelfStateClassifier,
    phase2_data: List[Dict],
    thermal_labels: torch.Tensor,
    device: torch.device,
) -> float:
    """Evaluate classifier accuracy on the phase-2 hidden bank."""
    classifier.eval()
    hiddens = torch.cat([d['hidden_mean'] for d in phase2_data], dim=0).to(device)
    labels = thermal_labels.float().to(device).unsqueeze(1)
    preds = classifier(hiddens)
    acc = ((preds > 0).float() == labels).float().mean().item()
    return acc


# ---------------------------------------------------------------------------
# Main Experiment
# ---------------------------------------------------------------------------

def run_experiment(args):
    """Run the full Ouroboros experiment."""

    print("=" * 70)
    print("z902  OUROBOROS SELF-REFERENTIAL TRAINING EXPERIMENT")
    print("=" * 70)
    print(f"Start time : {datetime.now().isoformat()}")
    print(f"Device req : {args.device}")
    print(f"Phase 1 ep : {args.phase1_epochs}")
    print(f"Phase 3 ep : {args.phase3_epochs}")
    print(f"Batch size : {args.batch_size}")
    print(f"Seq len    : {args.seq_len}")
    print(f"Lambda self: {args.lambda_self}")
    print(f"Gen samples: {args.num_gen_samples}")
    print()

    device = get_device(args.device)
    print(f"[device] Using {device}")
    init_telemetry()

    # ---- Data ----
    text = download_tiny_shakespeare()
    # Use 90% train, 10% eval
    split = int(len(text) * 0.9)
    train_ds = CharDataset(text[:split], seq_len=args.seq_len)
    eval_ds = CharDataset(text[split:], seq_len=args.seq_len)
    print(f"[data] Train sequences: {len(train_ds):,}  "
          f"Eval sequences: {len(eval_ds):,}")

    # ---- Model config ----
    config = MetabolicConfig(
        hidden_dim=256,
        num_layers=6,
        num_heads=4,
        ff_dim=1024,
        telemetry_dim=12,
        max_seq_len=max(args.seq_len, 512),
    )

    # ---- Paths ----
    ckpt_dir = Path(__file__).parent.parent / 'checkpoints'
    results_dir = Path(__file__).parent.parent / 'results'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results: Dict = {
        'experiment': 'z902_ouroboros',
        'timestamp': datetime.now().isoformat(),
        'args': vars(args),
        'device': str(device),
        'model_params': None,
        'conditions': {},
    }

    # ==================================================================
    # CONDITION A : Phase 1 only (control -- no self-referential learning)
    # ==================================================================
    print("\n\n" + "#" * 70)
    print("# CONDITION A : Phase 1 Only (Control)")
    print("#" * 70)

    model_a = MetabolicTransformer(config).to(device)
    all_results['model_params'] = model_a.get_num_parameters()
    print(f"[model] Parameters: {model_a.get_num_parameters():,}")

    hist_a = train_phase1(model_a, train_ds, device,
                          epochs=args.phase1_epochs,
                          batch_size=args.batch_size)

    ppl_a = evaluate_perplexity(model_a, eval_ds, device, args.batch_size)
    print(f"  [eval] Condition A perplexity: {ppl_a:.2f}")

    torch.save(model_a.state_dict(),
               ckpt_dir / 'z902_condition_a.pt')

    all_results['conditions']['A_phase1_only'] = {
        'phase1_history': hist_a,
        'eval_perplexity': ppl_a,
    }

    # ==================================================================
    # PHASE 2 : Generate data under varied thermal conditions
    # ==================================================================
    print("\n\n" + "#" * 70)
    print("# PHASE 2 : Thermal Data Generation")
    print("#" * 70)

    # Use the Phase-1 trained model for generation
    model_gen = MetabolicTransformer(config).to(device)
    model_gen.load_state_dict(model_a.state_dict())

    n_per_cond = args.num_gen_samples // 2

    # --- Cool condition ---
    cool_gpu(device, target_c=48.0, timeout_s=45)
    print(f"\n  Generating {n_per_cond} samples under COOL condition ...")
    cool_data = generate_under_thermal(
        model_gen, train_ds, device, n_per_cond, args.seq_len, 'cool'
    )

    # --- Warm condition ---
    warm_gpu(device, duration_s=10.0)
    print(f"\n  Generating {n_per_cond} samples under WARM condition ...")
    warm_data = generate_under_thermal(
        model_gen, train_ds, device, n_per_cond, args.seq_len, 'warm'
    )

    phase2_data = cool_data + warm_data

    # Assign thermal labels based on actual measured temperatures
    temps = [d['gpu_temp'] for d in phase2_data]
    median_temp = np.median(temps) if any(t > 0 for t in temps) else 50.0
    thermal_labels = torch.tensor(
        [1 if d['gpu_temp'] >= median_temp else 0 for d in phase2_data],
        dtype=torch.long,
    )
    n_cool = (thermal_labels == 0).sum().item()
    n_warm = (thermal_labels == 1).sum().item()
    print(f"\n  [phase2] Total samples: {len(phase2_data)}  "
          f"cool={n_cool}  warm={n_warm}  median_temp={median_temp:.1f}C")
    print(f"  [phase2] Temp range: {min(temps):.1f}C - {max(temps):.1f}C")

    # Fallback: if all temps are 0 (no sysfs), assign labels by condition string
    if all(t == 0.0 for t in temps):
        print("  [phase2] WARNING: No temperature readings; "
              "assigning labels by generation order (cool=0, warm=1)")
        thermal_labels = torch.tensor(
            [0 if d['condition'] == 'cool' else 1 for d in phase2_data],
            dtype=torch.long,
        )

    all_results['phase2'] = {
        'total_samples': len(phase2_data),
        'n_cool': n_cool,
        'n_warm': n_warm,
        'median_temp_c': float(median_temp),
        'temp_min_c': float(min(temps)),
        'temp_max_c': float(max(temps)),
    }

    # Collect hidden-state snapshots for CKA before Phase 3
    h_cool_pre = np.stack(
        [d['hidden_mean'].numpy().squeeze() for d in cool_data]
    )
    h_warm_pre = np.stack(
        [d['hidden_mean'].numpy().squeeze() for d in warm_data]
    )
    cka_pre = linear_cka(h_cool_pre, h_warm_pre)
    print(f"  [CKA] Pre-ouroboros  hot-vs-cold CKA = {cka_pre:.4f}")

    # ==================================================================
    # CONDITION B : Ouroboros with TRUE thermal labels
    # ==================================================================
    print("\n\n" + "#" * 70)
    print("# CONDITION B : Ouroboros (True Labels)")
    print("#" * 70)

    model_b = MetabolicTransformer(config).to(device)
    model_b.load_state_dict(model_a.state_dict())  # start from Phase 1 ckpt
    clf_b = SelfStateClassifier(config.hidden_dim).to(device)

    hist_b = train_phase3(
        model_b, clf_b, train_ds, phase2_data, thermal_labels, device,
        epochs=args.phase3_epochs, batch_size=args.batch_size,
        lambda_self=args.lambda_self, condition_name="Ouroboros (True Labels)",
    )

    ppl_b = evaluate_perplexity(model_b, eval_ds, device, args.batch_size)
    acc_b = evaluate_self_state_accuracy(clf_b, phase2_data, thermal_labels,
                                         device)
    print(f"  [eval] Condition B perplexity : {ppl_b:.2f}")
    print(f"  [eval] Condition B self-acc   : {acc_b:.3f}")

    torch.save({
        'model': model_b.state_dict(),
        'classifier': clf_b.state_dict(),
    }, ckpt_dir / 'z902_condition_b.pt')

    # Post-ouroboros CKA: regenerate hidden states using the updated model
    h_cool_post = collect_hidden_states(model_b, eval_ds, device,
                                         n_samples=min(n_per_cond, 100))
    # Use warm GPU hidden states from before (same model weights, different
    # condition), but to be fair let's collect from eval set too
    h_warm_post = collect_hidden_states(model_b, eval_ds, device,
                                         n_samples=min(n_per_cond, 100))
    cka_post_b = linear_cka(h_cool_post, h_warm_post)
    print(f"  [CKA] Post-ouroboros hot-vs-cold CKA = {cka_post_b:.4f}")

    all_results['conditions']['B_ouroboros_true'] = {
        'phase3_history': hist_b,
        'eval_perplexity': ppl_b,
        'self_state_accuracy': acc_b,
        'cka_hot_cold': cka_post_b,
    }

    # ==================================================================
    # CONDITION C : Ouroboros with RANDOM thermal labels (control)
    # ==================================================================
    print("\n\n" + "#" * 70)
    print("# CONDITION C : Ouroboros (Random Labels)")
    print("#" * 70)

    model_c = MetabolicTransformer(config).to(device)
    model_c.load_state_dict(model_a.state_dict())
    clf_c = SelfStateClassifier(config.hidden_dim).to(device)

    random_labels = thermal_labels[torch.randperm(len(thermal_labels))]

    hist_c = train_phase3(
        model_c, clf_c, train_ds, phase2_data, random_labels, device,
        epochs=args.phase3_epochs, batch_size=args.batch_size,
        lambda_self=args.lambda_self,
        condition_name="Ouroboros (Random Labels)",
    )

    ppl_c = evaluate_perplexity(model_c, eval_ds, device, args.batch_size)
    acc_c = evaluate_self_state_accuracy(clf_c, phase2_data, random_labels,
                                         device)
    print(f"  [eval] Condition C perplexity : {ppl_c:.2f}")
    print(f"  [eval] Condition C self-acc   : {acc_c:.3f}")

    torch.save({
        'model': model_c.state_dict(),
        'classifier': clf_c.state_dict(),
    }, ckpt_dir / 'z902_condition_c.pt')

    all_results['conditions']['C_random_labels'] = {
        'phase3_history': hist_c,
        'eval_perplexity': ppl_c,
        'self_state_accuracy': acc_c,
    }

    # ==================================================================
    # CONDITION D : Ouroboros with SAME thermal condition (no contrast)
    # ==================================================================
    print("\n\n" + "#" * 70)
    print("# CONDITION D : Ouroboros (Same Condition -- All Cool)")
    print("#" * 70)

    model_d = MetabolicTransformer(config).to(device)
    model_d.load_state_dict(model_a.state_dict())
    clf_d = SelfStateClassifier(config.hidden_dim).to(device)

    # All labels set to 0 (cool) -- no thermal contrast
    same_labels = torch.zeros_like(thermal_labels)

    hist_d = train_phase3(
        model_d, clf_d, train_ds, phase2_data, same_labels, device,
        epochs=args.phase3_epochs, batch_size=args.batch_size,
        lambda_self=args.lambda_self,
        condition_name="Ouroboros (Same Condition)",
    )

    ppl_d = evaluate_perplexity(model_d, eval_ds, device, args.batch_size)
    acc_d = evaluate_self_state_accuracy(clf_d, phase2_data, same_labels,
                                         device)
    print(f"  [eval] Condition D perplexity : {ppl_d:.2f}")
    print(f"  [eval] Condition D self-acc   : {acc_d:.3f}")

    torch.save({
        'model': model_d.state_dict(),
        'classifier': clf_d.state_dict(),
    }, ckpt_dir / 'z902_condition_d.pt')

    all_results['conditions']['D_same_condition'] = {
        'phase3_history': hist_d,
        'eval_perplexity': ppl_d,
        'self_state_accuracy': acc_d,
    }

    # ==================================================================
    # Summary
    # ==================================================================
    print("\n\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    all_results['cka'] = {
        'pre_ouroboros': cka_pre,
        'post_ouroboros_B': cka_post_b,
    }

    quality_gap_pre = abs(ppl_a - ppl_a)  # baseline has no gap by definition
    quality_gap_b = abs(ppl_b - ppl_a)
    quality_gap_c = abs(ppl_c - ppl_a)
    quality_gap_d = abs(ppl_d - ppl_a)

    all_results['summary'] = {
        'perplexity': {
            'A_phase1_only': ppl_a,
            'B_ouroboros_true': ppl_b,
            'C_random_labels': ppl_c,
            'D_same_condition': ppl_d,
        },
        'self_state_accuracy': {
            'B_ouroboros_true': acc_b,
            'C_random_labels': acc_c,
            'D_same_condition': acc_d,
        },
        'quality_gap_vs_baseline': {
            'B_ouroboros_true': quality_gap_b,
            'C_random_labels': quality_gap_c,
            'D_same_condition': quality_gap_d,
        },
        'cka_hot_cold': {
            'pre_ouroboros': cka_pre,
            'post_ouroboros_B': cka_post_b,
        },
    }

    # Print table
    header = f"{'Condition':<30} {'Perplexity':>12} {'Self-Acc':>10} {'PPL Gap':>10}"
    print(header)
    print("-" * len(header))
    print(f"{'A: Phase 1 only':<30} {ppl_a:>12.2f} {'N/A':>10} {0.0:>10.2f}")
    print(f"{'B: Ouroboros (true labels)':<30} {ppl_b:>12.2f} {acc_b:>10.3f} {quality_gap_b:>10.2f}")
    print(f"{'C: Ouroboros (random labels)':<30} {ppl_c:>12.2f} {acc_c:>10.3f} {quality_gap_c:>10.2f}")
    print(f"{'D: Ouroboros (same condition)':<30} {ppl_d:>12.2f} {acc_d:>10.3f} {quality_gap_d:>10.2f}")
    print()
    print(f"CKA (hot vs cold)  pre-ouroboros : {cka_pre:.4f}")
    print(f"CKA (hot vs cold)  post-ouroboros: {cka_post_b:.4f}")
    print()

    hypothesis_supported = (acc_b > acc_c + 0.05) and (acc_b > 0.6)
    print(f"Hypothesis supported: {hypothesis_supported}")
    print(f"  B accuracy ({acc_b:.3f}) > C accuracy ({acc_c:.3f}) + 0.05 ? "
          f"{'YES' if acc_b > acc_c + 0.05 else 'NO'}")
    print(f"  B accuracy ({acc_b:.3f}) > 0.6 ?  "
          f"{'YES' if acc_b > 0.6 else 'NO'}")

    all_results['hypothesis_supported'] = hypothesis_supported
    all_results['end_time'] = datetime.now().isoformat()

    # Save results
    results_path = results_dir / 'z902_ouroboros_training.json'
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")
    print(f"Checkpoints in   {ckpt_dir}/z902_*.pt")

    return all_results


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="z902 Ouroboros Self-Referential Training Experiment"
    )
    parser.add_argument('--phase1-epochs', type=int, default=5,
                        help='Number of Phase 1 (standard LM) epochs')
    parser.add_argument('--phase3-epochs', type=int, default=5,
                        help='Number of Phase 3 (Ouroboros) epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Training batch size')
    parser.add_argument('--seq-len', type=int, default=128,
                        help='Sequence length for char-level LM')
    parser.add_argument('--lambda-self', type=float, default=0.5,
                        help='Weight for self-state classification loss')
    parser.add_argument('--num-gen-samples', type=int, default=200,
                        help='Number of generation samples in Phase 2')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device: auto, cpu, cuda')
    args = parser.parse_args()

    try:
        run_experiment(args)
    except KeyboardInterrupt:
        print("\n[interrupted] Experiment stopped by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
