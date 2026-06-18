#!/usr/bin/env python3
"""
z1702: Cross-Machine Body Transfer Experiment
==============================================

Tests whether an embodied MetabolicTransformer trained on one machine (ikaros)
adapts when transferred to another with different hardware (daedalus, minos).

Phases:
  1. Train on ikaros (local AMD gfx1151, TinyShakespeare, 10 epochs)
  2. Evaluate on ikaros (baseline perplexity, J/token)
  3. Transfer to remote machines via SCP, run zero-shot eval
  4. Fine-tune on remote with real telemetry, compare embodied vs disembodied

Key question: Does the embodied model's internal representation of its own
hardware state help or hinder when transferred to a machine with a different
"body" (GPU)?  Can it re-adapt faster than a model that ignores telemetry?

Author: FEEL Research Team
Date: 2026-02-04
"""

import os
import sys
import json
import time
import math
import shutil
import textwrap
import subprocess
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
PROJECT_ROOT = str(Path(__file__).parent.parent)
sys.path.insert(0, PROJECT_ROOT)

from src.metabolic.film_transformer import (
    MetabolicTransformer,
    MetabolicConfig,
    create_metabolic_transformer,
    get_best_device,
)
from src.actuation.gpu_actuator import GPUActuator, PerformanceLevel
from src.telemetry.sysfs_hwmon import SysfsHwmonTelemetry, EnergyMeter

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "tinyshakespeare.txt")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
CHECKPOINT_PATH = os.path.join(MODELS_DIR, "z1702_trained.pt")
RESULT_PATH = os.path.join(RESULTS_DIR, "z1702_cross_machine.json")

REMOTE_MACHINES = {
    "daedalus": {
        "host": os.environ.get("DAEDALUS_HOST", "192.168.0.37"),
        "user": os.environ.get("DAEDALUS_USER", "daedalus"),
        "password": os.environ.get("DAEDALUS_PASS", "daedalus"),
        "gpu_vendor": "amd",
    },
    "minos": {
        "host": os.environ.get("MINOS_HOST", "192.168.0.38"),
        "user": os.environ.get("MINOS_USER", "minos"),
        "password": os.environ.get("MINOS_PASS", "minos"),
        "gpu_vendor": "nvidia",
    },
}

SEQ_LEN = 256
BATCH_SIZE = 32
TRAIN_EPOCHS = 10
FINETUNE_EPOCHS = 3
LR = 3e-4
TELEMETRY_DIM = 12

# ---------------------------------------------------------------------------
# SSH / SCP helpers
# ---------------------------------------------------------------------------

def ssh_run(host: str, user: str, password: str, cmd: str,
            timeout: int = 600) -> Tuple[str, str, int]:
    """Run command on remote host via sshpass."""
    try:
        result = subprocess.run(
            ["sshpass", "-p", password, "ssh",
             "-o", "StrictHostKeyChecking=no",
             "-o", "ConnectTimeout=10",
             f"{user}@{host}", cmd],
            capture_output=True, text=True, timeout=timeout,
        )
        return result.stdout, result.stderr, result.returncode
    except FileNotFoundError:
        return "", "sshpass not installed (apt install sshpass)", -1
    except subprocess.TimeoutExpired:
        return "", "SSH command timed out", -2
    except Exception as e:
        return "", str(e), -3


def scp_to(host: str, user: str, password: str,
           local: str, remote: str, timeout: int = 120) -> bool:
    """Copy file to remote host."""
    try:
        subprocess.run(
            ["sshpass", "-p", password, "scp",
             "-o", "StrictHostKeyChecking=no",
             "-o", "ConnectTimeout=10",
             local, f"{user}@{host}:{remote}"],
            check=True, timeout=timeout,
        )
        return True
    except Exception as e:
        print(f"  SCP to {host} failed: {e}")
        return False


def scp_from(host: str, user: str, password: str,
             remote: str, local: str, timeout: int = 120) -> bool:
    """Copy file from remote host."""
    try:
        subprocess.run(
            ["sshpass", "-p", password, "scp",
             "-o", "StrictHostKeyChecking=no",
             "-o", "ConnectTimeout=10",
             f"{user}@{host}:{remote}", local],
            check=True, timeout=timeout,
        )
        return True
    except Exception as e:
        print(f"  SCP from {host} failed: {e}")
        return False


def check_remote(host: str, user: str, password: str) -> bool:
    """Check whether a remote machine is reachable."""
    stdout, stderr, rc = ssh_run(host, user, password, "echo OK", timeout=15)
    return rc == 0 and "OK" in stdout

# ---------------------------------------------------------------------------
# Data loading (char-level TinyShakespeare)
# ---------------------------------------------------------------------------

def load_text_data(path: str) -> torch.Tensor:
    """Load text file as byte-level tensor."""
    with open(path, "r") as f:
        text = f.read()
    data = torch.tensor([ord(c) % 256 for c in text], dtype=torch.long)
    return data


def make_batches(data: torch.Tensor, seq_len: int, batch_size: int
                 ) -> List[torch.Tensor]:
    """Chunk data into batches of [batch_size, seq_len+1] for next-char pred."""
    total = (len(data) - 1) // seq_len
    batches = []
    for i in range(0, total - batch_size + 1, batch_size):
        batch = []
        for j in range(batch_size):
            start = (i + j) * seq_len
            chunk = data[start : start + seq_len + 1]
            if len(chunk) == seq_len + 1:
                batch.append(chunk)
        if len(batch) == batch_size:
            batches.append(torch.stack(batch))
    return batches

# ---------------------------------------------------------------------------
# Telemetry vector construction
# ---------------------------------------------------------------------------

def build_telemetry_vector(sample) -> torch.Tensor:
    """Build 12-dim telemetry vector from a GpuSample."""
    vec = [
        sample.power_w / 100.0,
        sample.temp_edge_c / 100.0,
        sample.freq_sclk_mhz / 3000.0,
        sample.gpu_busy_pct / 100.0,
        getattr(sample, "temp_junction_c", 0.0) / 100.0,
        getattr(sample, "temp_mem_c", 0.0) / 100.0,
        getattr(sample, "freq_mclk_mhz", 0) / 2000.0,
        getattr(sample, "vram_used_gb", 0.0) / 16.0,
        0.0, 0.0, 0.0, 0.0,  # derivative placeholders
    ]
    return torch.tensor(vec[:TELEMETRY_DIM], dtype=torch.float32)


def synthetic_telemetry(power: float = 30.0, temp: float = 55.0,
                        freq: float = 1500.0, util: float = 50.0
                        ) -> torch.Tensor:
    """Fallback synthetic telemetry for CPU-only or remote eval."""
    vec = [
        power / 100.0, temp / 100.0, freq / 3000.0, util / 100.0,
        temp / 100.0, 0.4, 0.5, 0.2,
        0.0, 0.0, 0.0, 0.0,
    ]
    return torch.tensor(vec[:TELEMETRY_DIM], dtype=torch.float32)


# ---------------------------------------------------------------------------
# Embodiment-specific metrics
# ---------------------------------------------------------------------------

def measure_self_model_accuracy(model: MetabolicTransformer, device: torch.device,
                                 telemetry_fn, n_samples: int = 50) -> Dict[str, float]:
    """
    Measure how accurately the model can predict its own telemetry state.

    This tests the "self-model" aspect of embodiment - does the model
    have an accurate internal representation of its own hardware?

    Method: Compare model hidden state correlation with actual telemetry.
    A well-embodied model should have hidden states that correlate with
    actual hardware state.
    """
    model.eval()

    # Collect pairs of (hidden_state, actual_telemetry)
    hidden_states = []
    telemetry_states = []

    # Create dummy input for probing
    dummy_input = torch.randint(0, 255, (1, SEQ_LEN), device=device)

    for _ in range(n_samples):
        telem = telemetry_fn()
        telem_batch = telem.unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(dummy_input, telemetry=telem_batch, return_hidden=True)
            hidden = out.get('hidden', out['logits'])
            # Take mean hidden state across sequence
            h_mean = hidden.mean(dim=1).squeeze().cpu().numpy()

        hidden_states.append(h_mean[:16])  # First 16 dims for tractability
        telemetry_states.append(telem.numpy()[:4])  # power, temp, freq, util

        # Small sleep to get varying telemetry
        time.sleep(0.02)

    hidden_arr = np.array(hidden_states)
    telem_arr = np.array(telemetry_states)

    # Compute correlation between hidden states and telemetry
    correlations = []
    for i in range(min(4, telem_arr.shape[1])):
        for j in range(min(16, hidden_arr.shape[1])):
            corr = np.corrcoef(telem_arr[:, i], hidden_arr[:, j])[0, 1]
            if not np.isnan(corr):
                correlations.append(abs(corr))

    mean_corr = np.mean(correlations) if correlations else 0.0
    max_corr = np.max(correlations) if correlations else 0.0

    return {
        'self_model_mean_correlation': float(mean_corr),
        'self_model_max_correlation': float(max_corr),
        'n_samples': n_samples,
    }


def measure_homeostatic_regulation(model: MetabolicTransformer, device: torch.device,
                                    telemetry_fn, n_calm: int = 20,
                                    n_stressed: int = 20) -> Dict[str, float]:
    """
    Measure homeostatic regulation quality.

    Homeostasis = maintaining stable internal state despite external perturbations.

    Test:
    1. Measure output variance during calm periods
    2. Measure output variance during stressed periods (high GPU load)
    3. A well-regulated system should show LOWER output variance under stress
       than an unregulated system (it compensates)
    """
    model.eval()

    # Calm period outputs
    calm_outputs = []
    calm_telemetry = []
    dummy_input = torch.randint(0, 255, (1, SEQ_LEN), device=device)

    for _ in range(n_calm):
        telem = telemetry_fn()
        telem_batch = telem.unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(dummy_input, telemetry=telem_batch)
            logit_mean = out['logits'].mean().item()
            logit_std = out['logits'].std().item()

        calm_outputs.append([logit_mean, logit_std])
        calm_telemetry.append(telem.numpy()[:4])
        time.sleep(0.02)

    # Stressed period (create GPU load)
    stressed_outputs = []
    stressed_telemetry = []

    for _ in range(n_stressed):
        # Create stress via matrix operations
        stress = torch.randn(1000, 1000, device=device)
        _ = stress @ stress.T @ stress
        del stress

        telem = telemetry_fn()
        telem_batch = telem.unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(dummy_input, telemetry=telem_batch)
            logit_mean = out['logits'].mean().item()
            logit_std = out['logits'].std().item()

        stressed_outputs.append([logit_mean, logit_std])
        stressed_telemetry.append(telem.numpy()[:4])

        torch.cuda.empty_cache()

    calm_arr = np.array(calm_outputs)
    stressed_arr = np.array(stressed_outputs)
    calm_telem = np.array(calm_telemetry)
    stressed_telem = np.array(stressed_telemetry)

    # Output stability metrics
    calm_output_var = calm_arr[:, 0].var()
    stressed_output_var = stressed_arr[:, 0].var()

    # Telemetry change
    telem_diff = np.mean(stressed_telem, axis=0) - np.mean(calm_telem, axis=0)
    power_increase = telem_diff[0] * 100 if len(telem_diff) > 0 else 0
    temp_increase = telem_diff[1] * 100 if len(telem_diff) > 1 else 0

    # Homeostatic regulation score
    # Higher = better (output remains stable despite telemetry changes)
    output_change = abs(stressed_arr[:, 0].mean() - calm_arr[:, 0].mean())
    telem_change = np.linalg.norm(telem_diff)

    # Regulation quality: how much output changed per unit telemetry change
    # Lower is better (less sensitivity to hardware state changes)
    regulation_quality = output_change / (telem_change + 1e-6)

    # Stability ratio: variance ratio (stressed/calm)
    # Close to 1.0 = good regulation
    stability_ratio = stressed_output_var / (calm_output_var + 1e-6)

    return {
        'calm_output_variance': float(calm_output_var),
        'stressed_output_variance': float(stressed_output_var),
        'power_increase_w': float(power_increase),
        'temp_increase_c': float(temp_increase),
        'regulation_quality': float(regulation_quality),
        'stability_ratio': float(stability_ratio),
        'homeostatic_score': float(1.0 / (1.0 + regulation_quality)),  # 0-1, higher=better
    }


def measure_adaptation_speed_detailed(model: MetabolicTransformer, batches: List[torch.Tensor],
                                       device: torch.device, telemetry_fn,
                                       target_ppl: float, max_epochs: int = 10,
                                       conditioning: bool = True) -> Dict[str, float]:
    """
    Measure detailed adaptation speed on new hardware.

    Returns epochs to reach target perplexity plus convergence metrics.
    """
    model.train()
    model.enable_conditioning(conditioning)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    epoch_metrics = []
    epochs_to_target = None

    for epoch in range(max_epochs):
        total_loss = 0.0
        total_tokens = 0
        t0 = time.time()

        for batch in batches[:len(batches)//4]:  # Use subset for speed
            batch = batch.to(device)
            inputs = batch[:, :-1]
            targets = batch[:, 1:]

            telem = telemetry_fn().unsqueeze(0).expand(inputs.size(0), -1).to(device)
            out = model(inputs, telemetry=telem)
            logits = out["logits"]

            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                   targets.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * targets.numel()
            total_tokens += targets.numel()

        elapsed = time.time() - t0
        avg_loss = total_loss / max(total_tokens, 1)
        ppl = math.exp(min(avg_loss, 20.0))

        epoch_metrics.append({
            'epoch': epoch,
            'perplexity': ppl,
            'loss': avg_loss,
            'elapsed_s': elapsed,
        })

        if epochs_to_target is None and ppl <= target_ppl:
            epochs_to_target = epoch + 1

    # Compute convergence rate (slope of log-ppl vs epoch)
    if len(epoch_metrics) >= 2:
        log_ppls = [math.log(e['perplexity']) for e in epoch_metrics]
        epochs = list(range(len(log_ppls)))
        # Linear regression
        n = len(epochs)
        sum_x = sum(epochs)
        sum_y = sum(log_ppls)
        sum_xy = sum(x*y for x, y in zip(epochs, log_ppls))
        sum_xx = sum(x*x for x in epochs)
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x ** 2 + 1e-6)
        convergence_rate = -slope  # Positive = good (ppl decreasing)
    else:
        convergence_rate = 0.0

    return {
        'epochs_to_target': epochs_to_target,
        'final_perplexity': epoch_metrics[-1]['perplexity'] if epoch_metrics else float('inf'),
        'convergence_rate': float(convergence_rate),
        'epoch_metrics': epoch_metrics,
    }

# ---------------------------------------------------------------------------
# Training & evaluation loops
# ---------------------------------------------------------------------------

def train_epoch(model: MetabolicTransformer, batches: List[torch.Tensor],
                optimizer: torch.optim.Optimizer, device: torch.device,
                telemetry_fn, conditioning: bool = True
                ) -> Dict[str, float]:
    """Train one epoch, return metrics."""
    model.train()
    model.enable_conditioning(conditioning)
    total_loss = 0.0
    total_tokens = 0
    t0 = time.time()

    for batch in batches:
        batch = batch.to(device)
        inputs = batch[:, :-1]    # [B, seq_len]
        targets = batch[:, 1:]    # [B, seq_len]

        telem = telemetry_fn().unsqueeze(0).expand(inputs.size(0), -1).to(device)
        out = model(inputs, telemetry=telem)
        logits = out["logits"]

        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               targets.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * targets.numel()
        total_tokens += targets.numel()

    elapsed = time.time() - t0
    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = math.exp(min(avg_loss, 20.0))
    tokens_sec = total_tokens / max(elapsed, 1e-6)

    return {
        "loss": avg_loss,
        "perplexity": perplexity,
        "tokens_sec": tokens_sec,
        "elapsed_s": elapsed,
        "total_tokens": total_tokens,
    }


@torch.no_grad()
def eval_epoch(model: MetabolicTransformer, batches: List[torch.Tensor],
               device: torch.device, telemetry_fn,
               conditioning: bool = True) -> Dict[str, float]:
    """Evaluate one epoch, return metrics."""
    model.eval()
    model.enable_conditioning(conditioning)
    total_loss = 0.0
    total_tokens = 0
    t0 = time.time()

    for batch in batches:
        batch = batch.to(device)
        inputs = batch[:, :-1]
        targets = batch[:, 1:]

        telem = telemetry_fn().unsqueeze(0).expand(inputs.size(0), -1).to(device)
        out = model(inputs, telemetry=telem)
        logits = out["logits"]

        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               targets.reshape(-1))
        total_loss += loss.item() * targets.numel()
        total_tokens += targets.numel()

    elapsed = time.time() - t0
    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = math.exp(min(avg_loss, 20.0))
    tokens_sec = total_tokens / max(elapsed, 1e-6)

    return {
        "loss": avg_loss,
        "perplexity": perplexity,
        "tokens_sec": tokens_sec,
        "elapsed_s": elapsed,
        "total_tokens": total_tokens,
    }

# ---------------------------------------------------------------------------
# Remote evaluation script (self-contained, generated as a string)
# ---------------------------------------------------------------------------

REMOTE_EVAL_SCRIPT = textwrap.dedent(r'''
#!/usr/bin/env python3
"""Remote evaluation script for z1702 cross-machine transfer (auto-generated)."""
import os, sys, json, math, time, platform, subprocess
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass

# ---- Minimal MetabolicTransformer (self-contained copy) ----

@dataclass
class MetabolicConfig:
    vocab_size: int = 256
    hidden_dim: int = 256
    num_layers: int = 6
    num_heads: int = 4
    ff_dim: int = 1024
    max_seq_len: int = 512
    dropout: float = 0.1
    telemetry_dim: int = 12
    film_hidden_dim: int = 64
    condition_every_layer: bool = True
    num_actions: int = 4
    action_head_hidden: int = 128
    use_causal_mask: bool = True

class FiLMGenerator(nn.Module):
    def __init__(self, telemetry_dim, hidden_dim, film_hidden=64):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(telemetry_dim, film_hidden), nn.ReLU(),
                                 nn.Linear(film_hidden, film_hidden), nn.ReLU())
        self.gamma_head = nn.Linear(film_hidden, hidden_dim)
        self.beta_head = nn.Linear(film_hidden, hidden_dim)
        nn.init.zeros_(self.gamma_head.weight); nn.init.zeros_(self.gamma_head.bias)
        nn.init.zeros_(self.beta_head.weight); nn.init.zeros_(self.beta_head.bias)
    def forward(self, t):
        h = self.mlp(t)
        return self.gamma_head(h), self.beta_head(h)

class FiLMLayerNorm(nn.Module):
    def __init__(self, hidden_dim, eps=1e-5):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, eps=eps)
    def forward(self, x, gamma=None, beta=None):
        h = self.norm(x)
        if gamma is not None: h = h * (1 + gamma.unsqueeze(1))
        if beta is not None: h = h + beta.unsqueeze(1)
        return h

class MetabolicAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_heads = cfg.num_heads
        self.head_dim = cfg.hidden_dim // cfg.num_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        self.k_proj = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        self.v_proj = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        self.out_proj = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        self.dropout = nn.Dropout(cfg.dropout)
    def forward(self, x, mask=None):
        B, S, _ = x.shape
        q = self.q_proj(x).view(B,S,self.num_heads,self.head_dim).transpose(1,2)
        k = self.k_proj(x).view(B,S,self.num_heads,self.head_dim).transpose(1,2)
        v = self.v_proj(x).view(B,S,self.num_heads,self.head_dim).transpose(1,2)
        a = torch.matmul(q, k.transpose(-2,-1)) * self.scale
        if mask is not None: a = a.masked_fill(mask == 0, float('-inf'))
        a = self.dropout(F.softmax(a, dim=-1))
        o = torch.matmul(a, v).transpose(1,2).contiguous().view(B, S, -1)
        return self.out_proj(o)

class MetabolicFFN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg.hidden_dim, cfg.ff_dim)
        self.fc2 = nn.Linear(cfg.ff_dim, cfg.hidden_dim)
        self.dropout = nn.Dropout(cfg.dropout)
    def forward(self, x):
        return self.fc2(self.dropout(F.gelu(self.fc1(x))))

class MetabolicBlock(nn.Module):
    def __init__(self, cfg, idx):
        super().__init__()
        self.ln1 = FiLMLayerNorm(cfg.hidden_dim)
        self.ln2 = FiLMLayerNorm(cfg.hidden_dim)
        self.attn = MetabolicAttention(cfg)
        self.ffn = MetabolicFFN(cfg)
        self.dropout = nn.Dropout(cfg.dropout)
    def forward(self, x, g1=None, b1=None, g2=None, b2=None, mask=None):
        h = self.attn(self.ln1(x, g1, b1), mask)
        x = x + self.dropout(h)
        h = self.ffn(self.ln2(x, g2, b2))
        return x + self.dropout(h)

class MetabolicTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.config = cfg
        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.hidden_dim)
        self.pos_embed = nn.Embedding(cfg.max_seq_len, cfg.hidden_dim)
        self.film_generators = nn.ModuleList([
            nn.ModuleDict({
                'ln1': FiLMGenerator(cfg.telemetry_dim, cfg.hidden_dim, cfg.film_hidden_dim),
                'ln2': FiLMGenerator(cfg.telemetry_dim, cfg.hidden_dim, cfg.film_hidden_dim),
            }) for _ in range(cfg.num_layers)])
        self.blocks = nn.ModuleList([MetabolicBlock(cfg, i) for i in range(cfg.num_layers)])
        self.ln_out = nn.LayerNorm(cfg.hidden_dim)
        self.token_head = nn.Linear(cfg.hidden_dim, cfg.vocab_size)
        self.action_head = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.action_head_hidden), nn.ReLU(),
            nn.Linear(cfg.action_head_hidden, cfg.num_actions))
        self.dropout = nn.Dropout(cfg.dropout)
        self.register_buffer("causal_mask",
            torch.triu(torch.ones(cfg.max_seq_len, cfg.max_seq_len), diagonal=1).bool())
        self._telemetry = None
        self._conditioning_enabled = True
    def enable_conditioning(self, v=True): self._conditioning_enabled = v
    def forward(self, input_ids, telemetry=None, return_hidden=False):
        B, S = input_ids.shape
        dev = input_ids.device
        t = telemetry
        if t is not None and t.dim() == 1: t = t.unsqueeze(0)
        if t is not None and t.size(0) == 1 and B > 1: t = t.expand(B, -1)
        pos = torch.arange(S, device=dev).unsqueeze(0).expand(B, -1)
        x = self.dropout(self.token_embed(input_ids) + self.pos_embed(pos))
        mask = ~self.causal_mask[:S, :S] if self.config.use_causal_mask else None
        for i, blk in enumerate(self.blocks):
            g1 = b1 = g2 = b2 = None
            if self._conditioning_enabled and t is not None:
                fg = self.film_generators[i]
                g1, b1 = fg['ln1'](t); g2, b2 = fg['ln2'](t)
            x = blk(x, g1, b1, g2, b2, mask)
        x = self.ln_out(x)
        logits = self.token_head(x)
        act = self.action_head(x[:, -1, :])
        out = {'logits': logits, 'action_logits': act}
        if return_hidden: out['hidden'] = x
        return out

# ---- Telemetry helpers ----

def detect_gpu():
    """Detect GPU vendor and basic info."""
    info = {"vendor": "cpu", "name": "CPU-only", "has_gpu": False}
    if torch.cuda.is_available():
        info["has_gpu"] = True
        info["name"] = torch.cuda.get_device_name(0)
        name_lower = info["name"].lower()
        if "nvidia" in name_lower or "geforce" in name_lower or "rtx" in name_lower:
            info["vendor"] = "nvidia"
        elif "amd" in name_lower or "radeon" in name_lower:
            info["vendor"] = "amd"
        else:
            info["vendor"] = "unknown"
    return info

def read_amd_telemetry():
    """Read AMD GPU telemetry via sysfs (best-effort)."""
    base = Path("/sys/class/drm")
    for card in sorted(base.glob("card*")):
        hwmon_base = card / "device" / "hwmon"
        if not hwmon_base.exists():
            continue
        for hwmon in hwmon_base.glob("hwmon*"):
            try:
                power_uw = int((hwmon / "power1_average").read_text().strip())
                temp_mc = int((hwmon / "temp1_input").read_text().strip())
                return {"power_w": power_uw / 1e6, "temp_c": temp_mc / 1e3}
            except Exception:
                pass
    return {"power_w": 0.0, "temp_c": 0.0}

def read_nvidia_telemetry():
    """Read NVIDIA GPU telemetry via nvidia-smi."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=power.draw,temperature.gpu,clocks.current.graphics,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            parts = r.stdout.strip().split(",")
            return {"power_w": float(parts[0].strip()),
                    "temp_c": float(parts[1].strip()),
                    "freq_mhz": float(parts[2].strip()),
                    "util_pct": float(parts[3].strip())}
    except Exception:
        pass
    return {"power_w": 0.0, "temp_c": 0.0, "freq_mhz": 0.0, "util_pct": 0.0}

def make_telemetry_vec(gpu_info):
    """Build 12-dim telemetry vector from whatever sensor data we got."""
    vendor = gpu_info["vendor"]
    if vendor == "amd":
        raw = read_amd_telemetry()
        return [raw["power_w"]/100, raw["temp_c"]/100, 0.5, 0.5,
                raw["temp_c"]/100, 0.4, 0.5, 0.2, 0,0,0,0]
    elif vendor == "nvidia":
        raw = read_nvidia_telemetry()
        return [raw["power_w"]/200, raw["temp_c"]/100,
                raw["freq_mhz"]/3000, raw["util_pct"]/100,
                raw["temp_c"]/100, 0.4, 0.5, 0.2, 0,0,0,0]
    else:
        return [0.3, 0.5, 0.5, 0.3, 0.5, 0.4, 0.5, 0.2, 0,0,0,0]

# ---- Main eval/finetune ----

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--finetune-epochs", type=int, default=0)
    args = parser.parse_args()

    gpu_info = detect_gpu()
    device = torch.device("cuda" if gpu_info["has_gpu"] else "cpu")
    print(f"GPU: {gpu_info}")
    print(f"Device: {device}")

    cfg = MetabolicConfig()
    model = MetabolicTransformer(cfg).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print("Checkpoint loaded.")

    # Load data
    with open(args.data, "r") as f:
        text = f.read()
    data = torch.tensor([ord(c) % 256 for c in text], dtype=torch.long)
    seq_len = 256
    batch_size = 32
    total = (len(data) - 1) // seq_len
    batches = []
    for i in range(0, total - batch_size + 1, batch_size):
        batch = []
        for j in range(batch_size):
            start = (i + j) * seq_len
            chunk = data[start:start + seq_len + 1]
            if len(chunk) == seq_len + 1:
                batch.append(chunk)
        if len(batch) == batch_size:
            batches.append(torch.stack(batch))

    results = {"hostname": platform.node(), "gpu": gpu_info,
               "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}

    # Collect telemetry samples during eval
    telemetry_samples = []

    def telem_fn():
        vec = make_telemetry_vec(gpu_info)
        telemetry_samples.append(vec[:4])
        return torch.tensor(vec, dtype=torch.float32)

    # ---- Zero-shot eval (embodied ON) ----
    model.eval()
    model.enable_conditioning(True)
    total_loss = 0.0; total_tok = 0
    t0 = time.time()
    with torch.no_grad():
        for batch in batches:
            batch = batch.to(device)
            inp, tgt = batch[:, :-1], batch[:, 1:]
            tv = telem_fn().unsqueeze(0).expand(inp.size(0), -1).to(device)
            out = model(inp, telemetry=tv)
            loss = F.cross_entropy(out["logits"].reshape(-1, out["logits"].size(-1)),
                                   tgt.reshape(-1))
            total_loss += loss.item() * tgt.numel()
            total_tok += tgt.numel()
    elapsed = time.time() - t0
    avg_loss = total_loss / max(total_tok, 1)
    results["eval_embodied"] = {
        "perplexity": math.exp(min(avg_loss, 20)),
        "loss": avg_loss,
        "tokens_sec": total_tok / max(elapsed, 1e-6),
        "elapsed_s": elapsed,
    }

    # ---- Zero-shot eval (embodied OFF / disembodied) ----
    model.enable_conditioning(False)
    total_loss = 0.0; total_tok = 0
    t0 = time.time()
    with torch.no_grad():
        for batch in batches:
            batch = batch.to(device)
            inp, tgt = batch[:, :-1], batch[:, 1:]
            tv = telem_fn().unsqueeze(0).expand(inp.size(0), -1).to(device)
            out = model(inp, telemetry=tv)
            loss = F.cross_entropy(out["logits"].reshape(-1, out["logits"].size(-1)),
                                   tgt.reshape(-1))
            total_loss += loss.item() * tgt.numel()
            total_tok += tgt.numel()
    elapsed = time.time() - t0
    avg_loss = total_loss / max(total_tok, 1)
    results["eval_disembodied"] = {
        "perplexity": math.exp(min(avg_loss, 20)),
        "loss": avg_loss,
        "tokens_sec": total_tok / max(elapsed, 1e-6),
        "elapsed_s": elapsed,
    }

    # ---- Fine-tune (if requested) ----
    if args.finetune_epochs > 0:
        for mode_name, cond in [("finetune_embodied", True),
                                ("finetune_disembodied", False)]:
            # Reload checkpoint for fair comparison
            model.load_state_dict(ckpt["model_state_dict"])
            model.enable_conditioning(cond)
            model.train()
            opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
            epoch_metrics = []
            for ep in range(args.finetune_epochs):
                ep_loss = 0.0; ep_tok = 0; t0 = time.time()
                for batch in batches:
                    batch = batch.to(device)
                    inp, tgt = batch[:, :-1], batch[:, 1:]
                    tv = telem_fn().unsqueeze(0).expand(inp.size(0), -1).to(device)
                    out = model(inp, telemetry=tv)
                    loss = F.cross_entropy(
                        out["logits"].reshape(-1, out["logits"].size(-1)),
                        tgt.reshape(-1))
                    opt.zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    ep_loss += loss.item() * tgt.numel()
                    ep_tok += tgt.numel()
                elapsed = time.time() - t0
                al = ep_loss / max(ep_tok, 1)
                epoch_metrics.append({
                    "epoch": ep, "perplexity": math.exp(min(al, 20)),
                    "loss": al, "tokens_sec": ep_tok / max(elapsed, 1e-6)})
                print(f"  {mode_name} epoch {ep}: ppl={epoch_metrics[-1]['perplexity']:.2f}")
            results[mode_name] = epoch_metrics

    # Telemetry statistics
    if telemetry_samples:
        arr = [s[:4] for s in telemetry_samples]
        import numpy as _np
        arr = _np.array(arr)
        results["telemetry_stats"] = {
            "mean": arr.mean(axis=0).tolist(),
            "std": arr.std(axis=0).tolist(),
            "n_samples": len(arr),
        }

    # ---- Embodiment-specific metrics ----
    print("Measuring self-model accuracy...")
    model.eval()
    model.enable_conditioning(True)
    hidden_states = []
    telem_states = []
    dummy_input = torch.randint(0, 255, (1, seq_len), device=device)
    for _ in range(30):
        tv = telem_fn()
        tv_batch = tv.unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(dummy_input, telemetry=tv_batch, return_hidden=True)
            h = out.get('hidden', out['logits']).mean(dim=1).squeeze().cpu().numpy()
        hidden_states.append(h[:16])
        telem_states.append(tv.numpy()[:4])
        time.sleep(0.02)

    hidden_arr = _np.array(hidden_states)
    telem_arr = _np.array(telem_states)
    correlations = []
    for i in range(min(4, telem_arr.shape[1])):
        for j in range(min(16, hidden_arr.shape[1])):
            c = _np.corrcoef(telem_arr[:, i], hidden_arr[:, j])[0, 1]
            if not _np.isnan(c):
                correlations.append(abs(c))
    results["self_model_accuracy"] = {
        "mean_correlation": float(_np.mean(correlations)) if correlations else 0.0,
        "max_correlation": float(_np.max(correlations)) if correlations else 0.0,
    }

    # Homeostatic regulation
    print("Measuring homeostatic regulation...")
    calm_outputs = []
    stressed_outputs = []
    dummy_input = torch.randint(0, 255, (1, seq_len), device=device)

    # Calm period
    for _ in range(15):
        tv = telem_fn().unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(dummy_input, telemetry=tv)
            calm_outputs.append(out['logits'].mean().item())
        time.sleep(0.02)

    # Stressed period
    for _ in range(15):
        if device.type == 'cuda':
            stress = torch.randn(1000, 1000, device=device)
            _ = stress @ stress.T @ stress
            del stress
        tv = telem_fn().unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(dummy_input, telemetry=tv)
            stressed_outputs.append(out['logits'].mean().item())

    calm_arr = _np.array(calm_outputs)
    stressed_arr = _np.array(stressed_outputs)
    output_change = abs(stressed_arr.mean() - calm_arr.mean())
    calm_var = calm_arr.var()
    stressed_var = stressed_arr.var()
    stability_ratio = stressed_var / (calm_var + 1e-6)
    results["homeostatic_regulation"] = {
        "calm_variance": float(calm_var),
        "stressed_variance": float(stressed_var),
        "stability_ratio": float(stability_ratio),
        "homeostatic_score": float(1.0 / (1.0 + output_change)),
    }

    print(f"Self-model correlation: {results['self_model_accuracy']['mean_correlation']:.4f}")
    print(f"Homeostatic score: {results['homeostatic_regulation']['homeostatic_score']:.4f}")

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results written to {args.output}")

if __name__ == "__main__":
    main()
''').lstrip()

# ---------------------------------------------------------------------------
# Phase 1: Local training
# ---------------------------------------------------------------------------

def phase1_train(device: torch.device, telemetry: SysfsHwmonTelemetry
                 ) -> Tuple[MetabolicTransformer, List[Dict]]:
    """Train MetabolicTransformer on TinyShakespeare for TRAIN_EPOCHS epochs."""
    print("=" * 70)
    print("  PHASE 1: Train on ikaros (local)")
    print("=" * 70)

    data = load_text_data(DATA_PATH)
    batches = make_batches(data, SEQ_LEN, BATCH_SIZE)
    print(f"  Data: {len(data):,} chars -> {len(batches)} batches "
          f"(seq_len={SEQ_LEN}, batch_size={BATCH_SIZE})")

    model = create_metabolic_transformer(
        hidden_dim=256, num_layers=6, num_heads=4, telemetry_dim=TELEMETRY_DIM)
    model = model.to(device)
    print(f"  Model params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # Telemetry collector for statistics
    telem_history: List[List[float]] = []

    def telemetry_fn() -> torch.Tensor:
        sample = telemetry.read_sample()
        vec = build_telemetry_vector(sample)
        telem_history.append(vec.tolist()[:4])
        return vec

    epoch_metrics = []
    for epoch in range(TRAIN_EPOCHS):
        telemetry.reset_accumulator()
        telemetry.start_continuous_sampling()

        metrics = train_epoch(model, batches, optimizer, device,
                              telemetry_fn, conditioning=True)

        telemetry.stop_continuous_sampling()
        energy_j = telemetry.accumulator.total_energy_j
        j_per_token = energy_j / max(metrics["total_tokens"], 1)

        # Telemetry stats for this epoch
        epoch_telem = np.array(telem_history[-len(batches):]) if telem_history else np.zeros((1, 4))

        record = {
            "epoch": epoch,
            "perplexity": metrics["perplexity"],
            "loss": metrics["loss"],
            "tokens_sec": metrics["tokens_sec"],
            "j_per_token": j_per_token,
            "energy_j": energy_j,
            "elapsed_s": metrics["elapsed_s"],
            "telemetry_mean": epoch_telem.mean(axis=0).tolist(),
        }
        epoch_metrics.append(record)
        print(f"  Epoch {epoch:2d}: ppl={metrics['perplexity']:8.2f}  "
              f"loss={metrics['loss']:.4f}  "
              f"tok/s={metrics['tokens_sec']:,.0f}  "
              f"J/tok={j_per_token:.6f}")

    # Save checkpoint
    os.makedirs(MODELS_DIR, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {
            "hidden_dim": 256, "num_layers": 6, "num_heads": 4,
            "telemetry_dim": TELEMETRY_DIM,
        },
        "epoch_metrics": epoch_metrics,
    }, CHECKPOINT_PATH)
    print(f"  Checkpoint saved: {CHECKPOINT_PATH}")

    return model, epoch_metrics


# ---------------------------------------------------------------------------
# Phase 2: Local evaluation
# ---------------------------------------------------------------------------

def phase2_eval_local(model: MetabolicTransformer, device: torch.device,
                      telemetry: SysfsHwmonTelemetry) -> Dict:
    """Evaluate trained model on ikaros (baseline)."""
    print("\n" + "=" * 70)
    print("  PHASE 2: Evaluate on ikaros (local baseline)")
    print("=" * 70)

    data = load_text_data(DATA_PATH)
    batches = make_batches(data, SEQ_LEN, BATCH_SIZE)

    telem_history: List[List[float]] = []

    def telemetry_fn() -> torch.Tensor:
        sample = telemetry.read_sample()
        vec = build_telemetry_vector(sample)
        telem_history.append(vec.tolist()[:4])
        return vec

    # Embodied eval
    telemetry.reset_accumulator()
    telemetry.start_continuous_sampling()
    em = eval_epoch(model, batches, device, telemetry_fn, conditioning=True)
    telemetry.stop_continuous_sampling()
    energy_j = telemetry.accumulator.total_energy_j
    em["j_per_token"] = energy_j / max(em["total_tokens"], 1)
    em["energy_j"] = energy_j

    # Disembodied eval
    telemetry.reset_accumulator()
    telemetry.start_continuous_sampling()
    dm = eval_epoch(model, batches, device, telemetry_fn, conditioning=False)
    telemetry.stop_continuous_sampling()
    energy_j_d = telemetry.accumulator.total_energy_j
    dm["j_per_token"] = energy_j_d / max(dm["total_tokens"], 1)
    dm["energy_j"] = energy_j_d

    # Embodiment-specific metrics
    print("  Measuring self-model accuracy...")
    self_model = measure_self_model_accuracy(model, device, telemetry_fn, n_samples=30)

    print("  Measuring homeostatic regulation...")
    homeostatic = measure_homeostatic_regulation(model, device, telemetry_fn,
                                                  n_calm=15, n_stressed=15)

    telem_arr = np.array(telem_history) if telem_history else np.zeros((1, 4))
    result = {
        "embodied": em,
        "disembodied": dm,
        "self_model_accuracy": self_model,
        "homeostatic_regulation": homeostatic,
        "telemetry_stats": {
            "mean": telem_arr.mean(axis=0).tolist(),
            "std": telem_arr.std(axis=0).tolist(),
        },
    }

    print(f"  Embodied:    ppl={em['perplexity']:.2f}  J/tok={em['j_per_token']:.6f}")
    print(f"  Disembodied: ppl={dm['perplexity']:.2f}  J/tok={dm['j_per_token']:.6f}")
    print(f"  Self-model correlation: {self_model['self_model_mean_correlation']:.4f}")
    print(f"  Homeostatic score: {homeostatic['homeostatic_score']:.4f}")

    return result


# ---------------------------------------------------------------------------
# Phase 3: Transfer to remote machines
# ---------------------------------------------------------------------------

def phase3_transfer(machine_name: str, machine_cfg: Dict) -> Optional[Dict]:
    """Transfer model to remote machine and run eval."""
    host = machine_cfg["host"]
    user = machine_cfg["user"]
    password = machine_cfg["password"]

    print(f"\n  --- Transfer to {machine_name} ({host}) ---")

    # Check reachability
    if not check_remote(host, user, password):
        print(f"  SKIPPED: {machine_name} is offline or unreachable")
        return {"status": "offline", "machine": machine_name}

    print(f"  {machine_name} is reachable")

    # Create remote working directory
    remote_dir = f"/tmp/z1702_{machine_name}"
    ssh_run(host, user, password, f"mkdir -p {remote_dir}")

    # Write the remote eval script to a temp file and SCP it
    local_script = os.path.join(MODELS_DIR, f"z1702_remote_eval_{machine_name}.py")
    with open(local_script, "w") as f:
        f.write(REMOTE_EVAL_SCRIPT)

    ok1 = scp_to(host, user, password, local_script, f"{remote_dir}/eval.py")
    ok2 = scp_to(host, user, password, CHECKPOINT_PATH,
                 f"{remote_dir}/checkpoint.pt")
    ok3 = scp_to(host, user, password, DATA_PATH,
                 f"{remote_dir}/tinyshakespeare.txt")

    if not (ok1 and ok2 and ok3):
        print(f"  SKIPPED: SCP failed for {machine_name}")
        return {"status": "scp_failed", "machine": machine_name}

    print(f"  Files transferred to {machine_name}:{remote_dir}")

    # Detect remote Python / venv
    py_cmd = "python3"
    # Try common venv locations
    for venv_path in [f"/home/{user}/venv/bin/python3",
                      f"/home/{user}/.venv/bin/python3",
                      f"/home/{user}/Documents/claude_hive/AMD_gfx1151_energy/venv/bin/python3"]:
        stdout, _, rc = ssh_run(host, user, password,
                                f"test -f {venv_path} && echo YES", timeout=10)
        if "YES" in stdout:
            py_cmd = venv_path
            break

    # Run zero-shot eval
    env_prefix = ""
    if machine_cfg["gpu_vendor"] == "amd":
        env_prefix = "HSA_OVERRIDE_GFX_VERSION=11.0.0 "

    eval_cmd = (f"{env_prefix}{py_cmd} {remote_dir}/eval.py "
                f"--checkpoint {remote_dir}/checkpoint.pt "
                f"--data {remote_dir}/tinyshakespeare.txt "
                f"--output {remote_dir}/results.json")
    print(f"  Running zero-shot eval on {machine_name}...")
    stdout, stderr, rc = ssh_run(host, user, password, eval_cmd, timeout=600)
    if rc != 0:
        print(f"  Eval FAILED (rc={rc}):")
        print(f"    stdout: {stdout[:500]}")
        print(f"    stderr: {stderr[:500]}")
        return {"status": "eval_failed", "machine": machine_name,
                "stdout": stdout[:1000], "stderr": stderr[:1000]}

    print(f"  Zero-shot eval complete on {machine_name}")

    # SCP back results
    local_result = os.path.join(RESULTS_DIR,
                                f"z1702_remote_{machine_name}_zeroshot.json")
    ok = scp_from(host, user, password, f"{remote_dir}/results.json",
                  local_result)
    if not ok:
        return {"status": "result_fetch_failed", "machine": machine_name}

    with open(local_result, "r") as f:
        remote_results = json.load(f)

    print(f"  Remote perplexity (embodied):    "
          f"{remote_results.get('eval_embodied', {}).get('perplexity', 'N/A')}")
    print(f"  Remote perplexity (disembodied): "
          f"{remote_results.get('eval_disembodied', {}).get('perplexity', 'N/A')}")

    return {"status": "ok", "machine": machine_name, "results": remote_results}


# ---------------------------------------------------------------------------
# Phase 4: Fine-tune on remote
# ---------------------------------------------------------------------------

def phase4_finetune_remote(machine_name: str, machine_cfg: Dict) -> Optional[Dict]:
    """Fine-tune on remote with real telemetry, compare embodied vs disembodied."""
    host = machine_cfg["host"]
    user = machine_cfg["user"]
    password = machine_cfg["password"]

    print(f"\n  --- Fine-tune on {machine_name} ({host}) ---")

    if not check_remote(host, user, password):
        print(f"  SKIPPED: {machine_name} is offline")
        return {"status": "offline", "machine": machine_name}

    remote_dir = f"/tmp/z1702_{machine_name}"

    # Detect Python
    py_cmd = "python3"
    for venv_path in [f"/home/{user}/venv/bin/python3",
                      f"/home/{user}/.venv/bin/python3",
                      f"/home/{user}/Documents/claude_hive/AMD_gfx1151_energy/venv/bin/python3"]:
        stdout, _, rc = ssh_run(host, user, password,
                                f"test -f {venv_path} && echo YES", timeout=10)
        if "YES" in stdout:
            py_cmd = venv_path
            break

    env_prefix = ""
    if machine_cfg["gpu_vendor"] == "amd":
        env_prefix = "HSA_OVERRIDE_GFX_VERSION=11.0.0 "

    ft_cmd = (f"{env_prefix}{py_cmd} {remote_dir}/eval.py "
              f"--checkpoint {remote_dir}/checkpoint.pt "
              f"--data {remote_dir}/tinyshakespeare.txt "
              f"--output {remote_dir}/finetune_results.json "
              f"--finetune-epochs {FINETUNE_EPOCHS}")
    print(f"  Running {FINETUNE_EPOCHS}-epoch finetune on {machine_name}...")
    stdout, stderr, rc = ssh_run(host, user, password, ft_cmd, timeout=600)
    if rc != 0:
        print(f"  Finetune FAILED (rc={rc}):")
        print(f"    stderr: {stderr[:500]}")
        return {"status": "finetune_failed", "machine": machine_name,
                "stderr": stderr[:1000]}

    local_result = os.path.join(RESULTS_DIR,
                                f"z1702_remote_{machine_name}_finetune.json")
    ok = scp_from(host, user, password, f"{remote_dir}/finetune_results.json",
                  local_result)
    if not ok:
        return {"status": "result_fetch_failed", "machine": machine_name}

    with open(local_result, "r") as f:
        ft_results = json.load(f)

    # Print adaptation curve
    for mode in ["finetune_embodied", "finetune_disembodied"]:
        if mode in ft_results:
            epdata = ft_results[mode]
            ppls = [e["perplexity"] for e in epdata]
            print(f"  {mode}: ppl " + " -> ".join(f"{p:.2f}" for p in ppls))

    return {"status": "ok", "machine": machine_name, "results": ft_results}


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def kl_divergence_gaussian(p_mean: np.ndarray, p_std: np.ndarray,
                           q_mean: np.ndarray, q_std: np.ndarray) -> float:
    """KL(P||Q) between two diagonal Gaussians."""
    p_var = p_std ** 2 + 1e-8
    q_var = q_std ** 2 + 1e-8
    kl = 0.5 * np.sum(np.log(q_var / p_var) + (p_var + (p_mean - q_mean) ** 2) / q_var - 1)
    return float(kl)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("  z1702: Cross-Machine Body Transfer Experiment")
    print("  Testing embodied model adaptation across different hardware")
    print("=" * 70)
    print()

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    device = get_best_device()
    print(f"Local device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Initialize local telemetry
    try:
        telemetry = SysfsHwmonTelemetry()
        sample = telemetry.read_sample()
        print(f"Telemetry OK: {sample.power_w:.1f}W, {sample.temp_edge_c:.1f}C")
    except Exception as e:
        print(f"WARNING: Telemetry init failed ({e}), using synthetic fallback")
        telemetry = None

    # Telemetry fallback
    if telemetry is None:
        class FakeTelemetry:
            def read_sample(self):
                from types import SimpleNamespace
                return SimpleNamespace(
                    power_w=25.0 + np.random.randn() * 2,
                    temp_edge_c=50.0 + np.random.randn() * 3,
                    freq_sclk_mhz=1500 + int(np.random.randn() * 100),
                    gpu_busy_pct=40 + np.random.randn() * 10,
                    temp_junction_c=55.0, temp_mem_c=45.0,
                    freq_mclk_mhz=1000, vram_used_gb=1.0,
                    timestamp_ns=time.time_ns())
            def reset_accumulator(self): pass
            def start_continuous_sampling(self): pass
            def stop_continuous_sampling(self): pass
            class accumulator:
                total_energy_j = 0.0
        telemetry = FakeTelemetry()

    all_results = {
        "experiment": "z1702_cross_machine_transfer",
        "timestamp": datetime.now().isoformat(),
        "local_device": str(device),
        "local_gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
    }

    # ---- Phase 1: Train ----
    model, train_metrics = phase1_train(device, telemetry)
    all_results["phase1_training"] = train_metrics

    # ---- Phase 2: Local eval ----
    local_eval = phase2_eval_local(model, device, telemetry)
    all_results["phase2_local_eval"] = local_eval

    # ---- Phase 3: Remote zero-shot transfer ----
    print("\n" + "=" * 70)
    print("  PHASE 3: Transfer to remote machines (zero-shot)")
    print("=" * 70)

    remote_zeroshot = {}
    for name, cfg in REMOTE_MACHINES.items():
        result = phase3_transfer(name, cfg)
        remote_zeroshot[name] = result
    all_results["phase3_remote_zeroshot"] = remote_zeroshot

    # ---- Phase 4: Remote fine-tune ----
    print("\n" + "=" * 70)
    print("  PHASE 4: Fine-tune on remote machines")
    print("=" * 70)

    remote_finetune = {}
    for name, cfg in REMOTE_MACHINES.items():
        # Only attempt if Phase 3 succeeded for this machine
        if remote_zeroshot.get(name, {}).get("status") == "ok":
            result = phase4_finetune_remote(name, cfg)
            remote_finetune[name] = result
        else:
            print(f"\n  SKIPPED {name}: Phase 3 did not succeed")
            remote_finetune[name] = {"status": "skipped", "machine": name}
    all_results["phase4_remote_finetune"] = remote_finetune

    # ---- Analysis ----
    print("\n" + "=" * 70)
    print("  ANALYSIS")
    print("=" * 70)

    local_ppl = local_eval["embodied"]["perplexity"]
    local_j = local_eval["embodied"]["j_per_token"]
    local_telem_mean = np.array(local_eval["telemetry_stats"]["mean"])
    local_telem_std = np.array(local_eval["telemetry_stats"]["std"])

    # Local embodiment metrics
    local_self_model = local_eval.get("self_model_accuracy", {})
    local_homeostatic = local_eval.get("homeostatic_regulation", {})

    print(f"\n  Local baseline (ikaros):")
    print(f"    Embodied perplexity:    {local_ppl:.2f}")
    print(f"    Disembodied perplexity: {local_eval['disembodied']['perplexity']:.2f}")
    print(f"    J/token:                {local_j:.6f}")
    print(f"    Self-model correlation: {local_self_model.get('self_model_mean_correlation', 0.0):.4f}")
    print(f"    Homeostatic score:      {local_homeostatic.get('homeostatic_score', 0.0):.4f}")

    verdicts = []

    for name in REMOTE_MACHINES:
        zs = remote_zeroshot.get(name, {})
        ft = remote_finetune.get(name, {})

        if zs.get("status") != "ok":
            print(f"\n  {name}: OFFLINE or FAILED ({zs.get('status', 'unknown')})")
            continue

        rr = zs["results"]
        remote_emb_ppl = rr.get("eval_embodied", {}).get("perplexity", float("inf"))
        remote_dis_ppl = rr.get("eval_disembodied", {}).get("perplexity", float("inf"))

        print(f"\n  {name} (zero-shot transfer):")
        print(f"    Embodied perplexity:    {remote_emb_ppl:.2f}")
        print(f"    Disembodied perplexity: {remote_dis_ppl:.2f}")
        print(f"    GPU: {rr.get('gpu', {}).get('name', 'unknown')}")

        # Telemetry divergence
        telem_div = 0.0
        if "telemetry_stats" in rr:
            remote_mean = np.array(rr["telemetry_stats"]["mean"])
            remote_std = np.array(rr["telemetry_stats"]["std"])
            # Pad if needed
            n = min(len(local_telem_mean), len(remote_mean))
            telem_div = kl_divergence_gaussian(
                local_telem_mean[:n], local_telem_std[:n] + 1e-6,
                remote_mean[:n], remote_std[:n] + 1e-6)
            print(f"    Telemetry KL divergence: {telem_div:.4f}")

        # Embodiment metrics from remote
        remote_self_model = rr.get("self_model_accuracy", {})
        remote_homeostatic = rr.get("homeostatic_regulation", {})
        remote_self_corr = remote_self_model.get("mean_correlation", 0.0)
        remote_homeo_score = remote_homeostatic.get("homeostatic_score", 0.0)

        print(f"    Self-model correlation: {remote_self_corr:.4f}")
        print(f"    Homeostatic score:      {remote_homeo_score:.4f}")

        # Self-model degradation (how much correlation dropped after transfer)
        local_corr = local_self_model.get('self_model_mean_correlation', 0.0)
        self_model_degradation = local_corr - remote_self_corr

        # Homeostatic degradation
        local_homeo = local_homeostatic.get('homeostatic_score', 0.0)
        homeostatic_degradation = local_homeo - remote_homeo_score

        verdicts.append({
            "machine": name,
            "telemetry_divergence": telem_div,
            "remote_embodied_ppl": remote_emb_ppl,
            "remote_disembodied_ppl": remote_dis_ppl,
            "remote_self_model_correlation": remote_self_corr,
            "remote_homeostatic_score": remote_homeo_score,
            "self_model_degradation": self_model_degradation,
            "homeostatic_degradation": homeostatic_degradation,
        })

        # Fine-tune results
        if ft.get("status") == "ok":
            ftr = ft["results"]
            for mode in ["finetune_embodied", "finetune_disembodied"]:
                if mode in ftr:
                    ppls = [e["perplexity"] for e in ftr[mode]]
                    print(f"    {mode}: {' -> '.join(f'{p:.2f}' for p in ppls)}")
                    verdicts[-1][f"{mode}_final_ppl"] = ppls[-1] if ppls else None

            # Adaptation speed
            emb_ft = ftr.get("finetune_embodied", [])
            dis_ft = ftr.get("finetune_disembodied", [])
            if emb_ft and dis_ft:
                target = local_ppl * 1.10  # within 10% of local

                def epochs_to_target(ft_list, tgt):
                    for e in ft_list:
                        if e["perplexity"] <= tgt:
                            return e["epoch"] + 1
                    return None

                emb_speed = epochs_to_target(emb_ft, target)
                dis_speed = epochs_to_target(dis_ft, target)
                verdicts[-1]["adaptation_speed_embodied"] = emb_speed
                verdicts[-1]["adaptation_speed_disembodied"] = dis_speed
                print(f"    Adaptation (epochs to <{target:.1f} ppl): "
                      f"embodied={emb_speed}, disembodied={dis_speed}")

    all_results["verdicts_per_machine"] = verdicts

    # ---- Final verdict ----
    print("\n" + "=" * 70)
    print("  VERDICT")
    print("=" * 70)

    pass_criteria = {
        "embodied_adapts_faster": False,
        "telemetry_divergence": False,
        "finetune_helps": False,
        "self_model_maintains": False,
        "homeostatic_recovers": False,
    }

    for v in verdicts:
        # Criterion 1: Embodied adapts faster than disembodied
        es = v.get("adaptation_speed_embodied")
        ds = v.get("adaptation_speed_disembodied")
        if es is not None and ds is not None and es < ds:
            pass_criteria["embodied_adapts_faster"] = True

        # Criterion 2: Telemetry genuinely different between machines
        if v.get("telemetry_divergence", 0) > 0.1:
            pass_criteria["telemetry_divergence"] = True

        # Criterion 3: Fine-tuning reduces perplexity vs zero-shot
        ft_emb = v.get("finetune_embodied_final_ppl")
        if ft_emb is not None and ft_emb < v.get("remote_embodied_ppl", float("inf")):
            pass_criteria["finetune_helps"] = True

        # Criterion 4: Self-model accuracy maintained (degradation < 50%)
        self_deg = v.get("self_model_degradation", 1.0)
        local_corr = local_self_model.get('self_model_mean_correlation', 0.001)
        if local_corr > 0 and (self_deg / local_corr) < 0.5:
            pass_criteria["self_model_maintains"] = True

        # Criterion 5: Homeostatic regulation recoverable (score > 0.5 after transfer)
        if v.get("remote_homeostatic_score", 0) > 0.5:
            pass_criteria["homeostatic_recovers"] = True

    n_pass = sum(pass_criteria.values())
    overall = "PASS" if n_pass >= 3 else ("PARTIAL" if n_pass >= 2 else "INCONCLUSIVE")

    print()
    for crit, passed in pass_criteria.items():
        tag = "PASS" if passed else "FAIL"
        print(f"  [{tag}] {crit}")
    print(f"\n  Overall: {overall} ({n_pass}/3 criteria met)")

    if not verdicts:
        print("\n  NOTE: No remote machines were reachable.")
        print("  Only local training/eval results are available.")
        overall = "LOCAL_ONLY"

    all_results["pass_criteria"] = pass_criteria
    all_results["overall_verdict"] = overall

    # ---- Save ----
    # Convert numpy types for JSON serialization
    def to_serializable(obj):
        if isinstance(obj, dict):
            return {k: to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [to_serializable(v) for v in obj]
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return obj

    all_results = to_serializable(all_results)

    with open(RESULT_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved: {RESULT_PATH}")

    # ---- Summary table ----
    print("\n" + "=" * 70)
    print("  SUMMARY TABLE")
    print("=" * 70)
    print(f"  {'Machine':<12} {'Emb PPL':>10} {'Dis PPL':>10} "
          f"{'Self-Model':>10} {'Homeostatic':>11} {'KL div':>8}")
    print("  " + "-" * 73)

    local_sm = local_self_model.get('self_model_mean_correlation', 0.0)
    local_hs = local_homeostatic.get('homeostatic_score', 0.0)
    print(f"  {'ikaros':<12} {local_ppl:>10.2f} "
          f"{local_eval['disembodied']['perplexity']:>10.2f} "
          f"{local_sm:>10.4f} "
          f"{local_hs:>11.4f} "
          f"{'0.0':>8}")

    for v in verdicts:
        print(f"  {v['machine']:<12} {v['remote_embodied_ppl']:>10.2f} "
              f"{v['remote_disembodied_ppl']:>10.2f} "
              f"{v.get('remote_self_model_correlation', 0.0):>10.4f} "
              f"{v.get('remote_homeostatic_score', 0.0):>11.4f} "
              f"{v['telemetry_divergence']:>8.4f}")

    # Embodiment metrics summary
    print("\n  " + "-" * 73)
    print("  EMBODIMENT METRICS ANALYSIS:")
    print("  " + "-" * 73)
    for v in verdicts:
        sm_deg = v.get('self_model_degradation', 0.0)
        hs_deg = v.get('homeostatic_degradation', 0.0)
        print(f"  {v['machine']}:")
        print(f"    Self-model degradation:    {sm_deg:+.4f} "
              f"({'OK' if abs(sm_deg) < local_sm * 0.5 else 'DEGRADED'})")
        print(f"    Homeostatic degradation:   {hs_deg:+.4f} "
              f"({'OK' if abs(hs_deg) < 0.2 else 'IMPAIRED'})")

    print()


if __name__ == "__main__":
    main()
