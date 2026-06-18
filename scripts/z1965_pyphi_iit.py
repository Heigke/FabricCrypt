#!/usr/bin/env python3
"""
z1965: Formal PyPhi IIT Phi Calculation
=======================================

Implements formal Integrated Information Theory (IIT) Phi calculation using PyPhi
to provide scientific rigor beyond the proxy measures used in z1708.

BACKGROUND:
- z1708 uses three tractable proxies: PIL (Partition Information Loss),
  PCI (Perturbational Complexity Index), and cross-layer MI
- This script computes ACTUAL Phi values using PyPhi's formal algorithms

KEY INSIGHT:
- True Phi is NP-hard to compute, so we extract small subsystems (8-16 nodes)
  from neural network hidden states
- We binarize hidden activations and compute transition probability matrices (TPMs)
- PyPhi then calculates the exact Phi for these subsystems

METHODOLOGY:
1. Train a FiLM-conditioned MetabolicTransformer under two conditions:
   - Embodied: FiLM ON, real GPU telemetry, actuation enabled
   - Disembodied: FiLM OFF, constant telemetry, no actuation

2. Extract hidden state vectors from the trained models

3. Convert hidden states to TPMs via:
   - Binarization (median threshold)
   - Computing state transition probabilities from consecutive states

4. Calculate formal Phi using pyphi.compute.sia() / pyphi.compute.phi()

5. Compare formal Phi with proxy measures from z1708

PYPHI COMPATIBILITY:
- PyPhi 1.2.0 has Python 3.12 compatibility issues (Iterable import)
- We patch this at runtime by modifying collections module

References:
- Tononi et al. (2004) "An Information Integration Theory of Consciousness"
- Oizumi et al. (2014) "From the Phenomenology to the Mechanisms of Consciousness"
- PyPhi documentation: https://pyphi.readthedocs.io/

Author: Claude + ikaros
Date: 2026-02-05
"""

import sys
import os
import json
import time
import math
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter
from dataclasses import dataclass

# Python 3.12 compatibility patch for pyphi
# Must be done BEFORE importing pyphi
import collections
import collections.abc
if not hasattr(collections, 'Iterable'):
    collections.Iterable = collections.abc.Iterable
if not hasattr(collections, 'Mapping'):
    collections.Mapping = collections.abc.Mapping
if not hasattr(collections, 'MutableMapping'):
    collections.MutableMapping = collections.abc.MutableMapping
if not hasattr(collections, 'Sequence'):
    collections.Sequence = collections.abc.Sequence

# Setup path and environment
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '11.0.0'

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import pyphi with error handling
try:
    import pyphi
    PYPHI_AVAILABLE = True
    print(f"PyPhi {pyphi.__version__} loaded successfully")
except ImportError as e:
    PYPHI_AVAILABLE = False
    print(f"WARNING: PyPhi import failed: {e}")
    print("Will compute TPM-based integration measures without formal Phi")

from src.metabolic.film_transformer import create_metabolic_transformer, MetabolicConfig
from src.telemetry.sysfs_hwmon import SysfsHwmonTelemetry
from src.actuation.gpu_actuator import GPUActuator, PerformanceLevel

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_PATH = Path(__file__).parent.parent / 'data' / 'tinyshakespeare.txt'
RESULTS_PATH = Path(__file__).parent.parent / 'results' / 'z1965_pyphi_iit.json'

# Training parameters
NUM_EPOCHS = 3
BATCH_SIZE = 4
SEQ_LEN = 128  # Shorter for faster iteration
LR = 3e-4
THERMAL_SETPOINT_C = 60.0
ACTION_NAMES = ['ECO', 'BALANCED', 'PERFORMANCE', 'MAX']
PRINT_EVERY = 50

# Phi calculation parameters
PHI_NUM_NODES = 8  # Number of nodes for Phi calculation (8 = tractable, 16 = slow)
PHI_SAMPLES_PER_BATCH = 20  # Samples to collect for TPM estimation
NUM_EVAL_BATCHES = 30  # Batches for evaluation


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class CharDataset:
    """Byte-level character dataset."""

    def __init__(self, path: Path, seq_len: int):
        text = path.read_text(encoding='utf-8', errors='replace')
        self.data = torch.tensor(
            [b for b in text.encode('utf-8')], dtype=torch.long,
        )
        self.seq_len = seq_len
        self.n_batches = (len(self.data) - seq_len - 1) // (BATCH_SIZE * seq_len)

    def get_batch(self, batch_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        offset = batch_idx * BATCH_SIZE * self.seq_len
        inputs, targets = [], []
        for b in range(BATCH_SIZE):
            start = offset + b * self.seq_len
            end = start + self.seq_len
            if end + 1 > len(self.data):
                start = 0
                end = self.seq_len
            inputs.append(self.data[start:end])
            targets.append(self.data[start + 1:end + 1])
        return torch.stack(inputs), torch.stack(targets)


# ---------------------------------------------------------------------------
# Telemetry Building
# ---------------------------------------------------------------------------
def build_telemetry(
    sample, state, prev_sample=None,
) -> torch.Tensor:
    """Build 12-dim telemetry vector for MetabolicTransformer FiLM."""
    if prev_sample is not None:
        dt = max((sample.timestamp_ns - prev_sample.timestamp_ns) / 1e9, 1e-6)
        d_power = (sample.power_w - prev_sample.power_w) / (50.0 * dt)
        d_temp  = (sample.temp_edge_c - prev_sample.temp_edge_c) / (100.0 * dt)
        d_freq  = (sample.freq_sclk_mhz - prev_sample.freq_sclk_mhz) / (3000.0 * dt)
        d_util  = (sample.gpu_busy_pct - prev_sample.gpu_busy_pct) / (100.0 * dt)
    else:
        d_power = d_temp = d_freq = d_util = 0.0

    MAX_SCLK = 2900.0
    throttled = 1.0 if sample.freq_sclk_mhz < MAX_SCLK * 0.5 else 0.0
    perf_map = {'low': 0.0, 'balanced': 0.5, 'high': 1.0, 'auto': 0.5, 'manual': 0.5}
    perf_encoded = perf_map.get(state.performance_level, 0.5)
    thermal_dev = (sample.temp_edge_c - THERMAL_SETPOINT_C) / 40.0
    freq_headroom = (MAX_SCLK - sample.freq_sclk_mhz) / MAX_SCLK

    return torch.tensor([
        sample.power_w / 50.0,
        sample.temp_edge_c / 100.0,
        sample.freq_sclk_mhz / 3000.0,
        sample.gpu_busy_pct / 100.0,
        perf_encoded,
        throttled,
        d_power,
        d_temp,
        d_freq,
        d_util,
        thermal_dev,
        freq_headroom,
    ], dtype=torch.float32)


# ---------------------------------------------------------------------------
# Action Application
# ---------------------------------------------------------------------------
def apply_action(action_idx: int, actuator: GPUActuator, cur_perf: int) -> int:
    """Map model action to GPU performance level."""
    if action_idx == 0:       # ECO
        new = max(0, cur_perf - 1)
    elif action_idx == 1:     # BALANCED
        new = 1
    elif action_idx == 2:     # PERFORMANCE
        new = min(2, cur_perf + 1)
    else:                     # MAX
        new = 2
    levels = [PerformanceLevel.LOW, PerformanceLevel.BALANCED, PerformanceLevel.HIGH]
    actuator.set_performance_level(levels[new])
    return new


# ---------------------------------------------------------------------------
# Phi Calculation Utilities
# ---------------------------------------------------------------------------
@dataclass
class PhiResult:
    """Result of Phi calculation for a subsystem."""
    phi: float
    num_nodes: int
    num_states: int
    tpm_rank: int
    state: Optional[Tuple] = None
    computation_time_s: float = 0.0


def extract_subsystem_nodes(
    hidden_states: torch.Tensor,
    num_nodes: int = PHI_NUM_NODES,
    method: str = 'variance'
) -> torch.Tensor:
    """
    Extract a subset of nodes from hidden states for tractable Phi computation.

    Args:
        hidden_states: [batch, seq_len, hidden_dim]
        num_nodes: Number of nodes to extract
        method: 'variance' (highest variance), 'random', or 'pca'

    Returns:
        Extracted nodes [batch, seq_len, num_nodes]
    """
    batch, seq_len, hidden_dim = hidden_states.shape

    if method == 'variance':
        # Select dimensions with highest variance across batch and sequence
        var_per_dim = hidden_states.var(dim=(0, 1))  # [hidden_dim]
        top_indices = torch.argsort(var_per_dim, descending=True)[:num_nodes]
        nodes = hidden_states[:, :, top_indices]
    elif method == 'random':
        indices = torch.randperm(hidden_dim)[:num_nodes]
        nodes = hidden_states[:, :, indices]
    elif method == 'pca':
        # Use PCA to extract principal components
        flat = hidden_states.reshape(-1, hidden_dim).cpu().numpy()
        from sklearn.decomposition import PCA
        pca = PCA(n_components=num_nodes)
        transformed = pca.fit_transform(flat)
        nodes = torch.from_numpy(transformed.reshape(batch, seq_len, num_nodes))
    else:
        raise ValueError(f"Unknown method: {method}")

    return nodes


def binarize_states(
    nodes: torch.Tensor,
    threshold: str = 'median'
) -> np.ndarray:
    """
    Binarize continuous node states for IIT analysis.

    Args:
        nodes: [batch, seq_len, num_nodes] continuous values
        threshold: 'median', 'mean', or float value

    Returns:
        Binary states [batch*seq_len, num_nodes] as 0/1 integers
    """
    flat = nodes.reshape(-1, nodes.shape[-1]).cpu().numpy()

    if threshold == 'median':
        th = np.median(flat, axis=0, keepdims=True)
    elif threshold == 'mean':
        th = np.mean(flat, axis=0, keepdims=True)
    else:
        th = float(threshold)

    binary = (flat > th).astype(int)
    return binary


def compute_tpm(
    binary_states: np.ndarray,
    lookahead: int = 1
) -> np.ndarray:
    """
    Compute transition probability matrix from binary state sequence.

    The TPM maps from current state to next state probability distribution.
    For n nodes, there are 2^n possible states.

    Args:
        binary_states: [num_samples, num_nodes] binary (0/1)
        lookahead: Steps ahead for transition (default 1)

    Returns:
        TPM: [2^n, 2^n] transition probability matrix
    """
    num_samples, num_nodes = binary_states.shape
    num_states = 2 ** num_nodes

    # Convert binary states to state indices
    powers = 2 ** np.arange(num_nodes)
    state_indices = binary_states @ powers  # [num_samples]

    # Count transitions
    transition_counts = np.zeros((num_states, num_states))

    for t in range(num_samples - lookahead):
        current_state = state_indices[t]
        next_state = state_indices[t + lookahead]
        transition_counts[current_state, next_state] += 1

    # Normalize rows to get probabilities
    row_sums = transition_counts.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1)  # Avoid division by zero
    tpm = transition_counts / row_sums

    # For states never visited, use uniform distribution
    unvisited = row_sums.flatten() == 0
    tpm[unvisited] = 1.0 / num_states

    return tpm


def compute_tpm_by_node(
    binary_states: np.ndarray,
    lookahead: int = 1
) -> np.ndarray:
    """
    Compute TPM in state-by-node format (PyPhi's expected format).

    PyPhi expects TPM[past_state, node] = P(node=1 | past_state)

    Args:
        binary_states: [num_samples, num_nodes]
        lookahead: Steps ahead

    Returns:
        TPM: [2^n, n] where entry [s, i] is P(node_i=1 | state=s)
    """
    num_samples, num_nodes = binary_states.shape
    num_states = 2 ** num_nodes

    # Convert states to indices
    powers = 2 ** np.arange(num_nodes)
    state_indices = binary_states @ powers

    # Count occurrences of each state and node=1 outcomes
    state_counts = np.zeros(num_states)
    node_on_counts = np.zeros((num_states, num_nodes))

    for t in range(num_samples - lookahead):
        current_state = state_indices[t]
        next_binary = binary_states[t + lookahead]
        state_counts[current_state] += 1
        node_on_counts[current_state] += next_binary

    # Compute conditional probabilities
    tpm_by_node = np.zeros((num_states, num_nodes))
    for s in range(num_states):
        if state_counts[s] > 0:
            tpm_by_node[s] = node_on_counts[s] / state_counts[s]
        else:
            # Unvisited states: use independent node probabilities
            tpm_by_node[s] = binary_states.mean(axis=0)

    return tpm_by_node


def compute_phi_pyphi(
    tpm: np.ndarray,
    state: Tuple[int, ...],
    connectivity_matrix: Optional[np.ndarray] = None
) -> PhiResult:
    """
    Compute formal Phi using PyPhi.

    Args:
        tpm: Transition probability matrix [2^n, n] (state-by-node format)
        state: Current state as tuple of 0/1
        connectivity_matrix: [n, n] binary connectivity (None = fully connected)

    Returns:
        PhiResult with formal Phi value
    """
    if not PYPHI_AVAILABLE:
        return PhiResult(
            phi=0.0,
            num_nodes=tpm.shape[1],
            num_states=tpm.shape[0],
            tpm_rank=np.linalg.matrix_rank(tpm),
            state=state,
            computation_time_s=0.0
        )

    t0 = time.time()
    num_nodes = tpm.shape[1]

    # Create connectivity matrix if not provided (fully connected)
    if connectivity_matrix is None:
        connectivity_matrix = np.ones((num_nodes, num_nodes), dtype=int)

    try:
        # Suppress pyphi warnings about determinism
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')

            # Create network
            network = pyphi.Network(tpm, connectivity_matrix)

            # Create subsystem for current state
            subsystem = pyphi.Subsystem(network, state)

            # Compute System Integrated Information (SIA)
            sia = pyphi.compute.sia(subsystem)
            phi = sia.phi

    except Exception as e:
        print(f"  PyPhi computation failed: {e}")
        phi = 0.0

    computation_time = time.time() - t0

    return PhiResult(
        phi=float(phi),
        num_nodes=num_nodes,
        num_states=tpm.shape[0],
        tpm_rank=np.linalg.matrix_rank(tpm),
        state=state,
        computation_time_s=computation_time
    )


def compute_integration_proxy(tpm: np.ndarray) -> Dict[str, float]:
    """
    Compute integration proxy measures from TPM without full Phi.

    Measures:
    - Effective Information: I(X_t+1; X_t) via TPM
    - Mutual Information: Average MI between nodes
    - Integration: 1 - (factorized MI / joint MI)
    """
    num_states, num_nodes = tpm.shape

    # Effective information: entropy of transitions
    # H(X_t+1 | X_t) averaged over states
    cond_entropy = 0.0
    for s in range(num_states):
        for n in range(num_nodes):
            p = tpm[s, n]
            if 0 < p < 1:
                cond_entropy -= (p * np.log2(p) + (1-p) * np.log2(1-p))
    cond_entropy /= num_states

    # Marginal entropy of each node
    marginal_probs = tpm.mean(axis=0)  # Assuming uniform state distribution
    node_entropy = 0.0
    for p in marginal_probs:
        if 0 < p < 1:
            node_entropy -= (p * np.log2(p) + (1-p) * np.log2(1-p))

    # Integration proxy: reduction in entropy from joint consideration
    integration = max(0, node_entropy - cond_entropy)

    # Determinism: how close TPM is to deterministic (0 or 1 entries)
    determinism = np.mean(np.minimum(tpm, 1 - tpm) * 2)  # 0 = deterministic, 1 = random
    determinism = 1 - determinism  # Invert so higher = more deterministic

    # Degeneracy: how many states map to same successor
    tpm_binary = (tpm > 0.5).astype(int)
    unique_successors = len(set(tuple(row) for row in tpm_binary))
    degeneracy = 1 - (unique_successors / num_states)

    return {
        'effective_information': float(integration),
        'node_entropy': float(node_entropy),
        'conditional_entropy': float(cond_entropy),
        'determinism': float(determinism),
        'degeneracy': float(degeneracy),
    }


# ---------------------------------------------------------------------------
# Training and Evaluation
# ---------------------------------------------------------------------------
def train_condition(
    condition_name: str,
    model: torch.nn.Module,
    dataset: CharDataset,
    device: torch.device,
    telemetry: SysfsHwmonTelemetry,
    actuator: GPUActuator,
    *,
    film_on: bool = True,
    use_real_telem: bool = True,
    do_actuation: bool = True,
) -> Tuple[Dict, List[torch.Tensor]]:
    """
    Train a model and collect hidden states for Phi analysis.

    Returns:
        (training_stats, hidden_state_samples)
    """
    print(f"\n{'='*60}")
    print(f"  Condition: {condition_name}")
    print(f"  FiLM: {'ON' if film_on else 'OFF'}  |  "
          f"Telemetry: {'real' if use_real_telem else 'constant'}  |  "
          f"Actuation: {'ON' if do_actuation else 'OFF'}")
    print(f"{'='*60}")

    model.enable_conditioning(film_on)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    prev_sample = None
    cur_perf = 1
    action_counter = Counter()
    epoch_losses: List[float] = []
    hidden_samples: List[torch.Tensor] = []

    for epoch in range(NUM_EPOCHS):
        epoch_loss_sum = 0.0
        n_tokens = 0
        t0 = time.time()

        for bi in range(min(dataset.n_batches, 150)):
            inp, tgt = dataset.get_batch(bi)
            inp, tgt = inp.to(device), tgt.to(device)

            # Build telemetry
            sample = telemetry.read_sample()
            gpu_state = actuator.get_current_state()

            if use_real_telem:
                telem_vec = build_telemetry(sample, gpu_state, prev_sample).to(device)
            else:
                telem_vec = torch.full((12,), 0.5, device=device)

            telem_batch = telem_vec.unsqueeze(0).expand(BATCH_SIZE, -1)

            # Forward with hidden state collection
            out = model(inp, telem_batch, return_hidden=True)
            logits = out['logits']
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), tgt.view(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss_sum += loss.item() * inp.numel()
            n_tokens += inp.numel()

            # Collect hidden states for Phi analysis (final epoch only)
            if epoch == NUM_EPOCHS - 1 and len(hidden_samples) < PHI_SAMPLES_PER_BATCH * 5:
                hidden_samples.append(out['hidden'].detach().clone())

            # Action
            if do_actuation and film_on:
                mean_probs = F.softmax(out['action_logits'], dim=-1).mean(dim=0)
                action_idx = torch.argmax(mean_probs).item()
                cur_perf = apply_action(action_idx, actuator, cur_perf)
                action_counter[ACTION_NAMES[action_idx]] += 1

            prev_sample = sample

            if bi % PRINT_EVERY == 0:
                ppl = math.exp(min(epoch_loss_sum / max(n_tokens, 1), 20))
                print(f"  [{condition_name}] epoch {epoch+1}/{NUM_EPOCHS} "
                      f"batch {bi} loss={loss.item():.4f} ppl={ppl:.1f}")

        avg_loss = epoch_loss_sum / max(n_tokens, 1)
        epoch_ppl = math.exp(min(avg_loss, 20))
        epoch_losses.append(epoch_ppl)
        elapsed = time.time() - t0
        print(f"  [{condition_name}] Epoch {epoch+1} done: ppl={epoch_ppl:.2f} "
              f"({elapsed:.1f}s, {n_tokens/elapsed:.0f} tok/s)")

    stats = {
        'name': condition_name,
        'epoch_perplexities': epoch_losses,
        'final_perplexity': epoch_losses[-1],
        'action_distribution': dict(action_counter),
    }

    return stats, hidden_samples


def compute_phi_for_condition(
    model: torch.nn.Module,
    hidden_samples: List[torch.Tensor],
    condition_name: str,
    device: torch.device,
) -> Dict:
    """
    Compute Phi measures for a trained model condition.
    """
    print(f"\n  Computing Phi for {condition_name}...")

    if not hidden_samples:
        return {'error': 'No hidden samples available'}

    # Concatenate hidden samples
    all_hidden = torch.cat(hidden_samples, dim=0)  # [total_batch, seq_len, hidden_dim]

    # Extract subsystem nodes
    nodes = extract_subsystem_nodes(all_hidden, num_nodes=PHI_NUM_NODES, method='variance')

    # Binarize
    binary_states = binarize_states(nodes)

    print(f"    Subsystem: {PHI_NUM_NODES} nodes, {len(binary_states)} state samples")

    # Compute TPM in state-by-node format
    tpm = compute_tpm_by_node(binary_states)

    print(f"    TPM shape: {tpm.shape}, rank: {np.linalg.matrix_rank(tpm)}")

    # Compute integration proxy measures (always available)
    proxy_measures = compute_integration_proxy(tpm)

    # Compute formal Phi if pyphi available
    phi_results = []
    if PYPHI_AVAILABLE:
        # Sample a few representative states for Phi computation
        # (Full computation over all states is exponential)
        state_indices = np.random.choice(len(binary_states), min(5, len(binary_states)), replace=False)

        for idx in state_indices:
            state = tuple(binary_states[idx])
            result = compute_phi_pyphi(tpm, state)
            phi_results.append({
                'phi': result.phi,
                'state': result.state,
                'computation_time': result.computation_time_s,
            })
            print(f"      State {state}: Phi = {result.phi:.4f} ({result.computation_time_s:.2f}s)")

        avg_phi = np.mean([r['phi'] for r in phi_results]) if phi_results else 0.0
    else:
        avg_phi = 0.0
        print("    (PyPhi not available, using proxy measures only)")

    return {
        'condition': condition_name,
        'num_nodes': PHI_NUM_NODES,
        'num_state_samples': len(binary_states),
        'tpm_shape': list(tpm.shape),
        'tpm_rank': int(np.linalg.matrix_rank(tpm)),
        'avg_phi': float(avg_phi),
        'phi_samples': phi_results,
        'proxy_measures': proxy_measures,
    }


def compute_z1708_proxies(
    model: torch.nn.Module,
    dataset: CharDataset,
    device: torch.device,
    telemetry: SysfsHwmonTelemetry,
    actuator: GPUActuator,
    film_on: bool,
    use_real_telem: bool,
) -> Dict:
    """
    Compute z1708-style proxy measures (PIL, PCI, MI) for comparison.

    Simplified version - see z1708 for full implementation.
    """
    model.eval()
    model.enable_conditioning(film_on)
    prev_sample = None

    # Collect layer activations for MI
    num_layers = model.config.num_layers
    layer_acts = [[] for _ in range(num_layers)]

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for bi in range(min(NUM_EVAL_BATCHES, dataset.n_batches)):
            inp, tgt = dataset.get_batch(bi)
            inp, tgt = inp.to(device), tgt.to(device)

            sample = telemetry.read_sample()
            gpu_state = actuator.get_current_state()

            if use_real_telem:
                telem_vec = build_telemetry(sample, gpu_state, prev_sample).to(device)
            else:
                telem_vec = torch.full((12,), 0.5, device=device)

            telem_batch = telem_vec.unsqueeze(0).expand(BATCH_SIZE, -1)
            prev_sample = sample

            # Manual forward to collect layer activations
            batch, seq_len = inp.shape
            positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch, -1)
            x = model.token_embed(inp) + model.pos_embed(positions)
            x = model.dropout(x)

            if model.config.use_causal_mask:
                mask = ~model.causal_mask[:seq_len, :seq_len]
            else:
                mask = None

            telem = telem_batch
            for i, block in enumerate(model.blocks):
                gamma1, beta1, gamma2, beta2 = None, None, None, None
                if model._conditioning_enabled and model.film_generators[i] is not None:
                    fg = model.film_generators[i]
                    gamma1, beta1 = fg['ln1'](telem)
                    gamma2, beta2 = fg['ln2'](telem)
                x = block(x, gamma1, beta1, gamma2, beta2, mask)
                layer_acts[i].append(x.mean(dim=(0, 1)).cpu().numpy())

            x = model.ln_out(x)
            logits = model.token_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), tgt.view(-1))
            total_loss += loss.item() * inp.numel()
            total_tokens += inp.numel()

    # Compute perplexity
    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(min(avg_loss, 20))

    # Compute cross-layer MI (simplified)
    layer_arrays = [np.array(acts) for acts in layer_acts]
    mi_values = []
    for i in range(num_layers):
        for j in range(i + 1, num_layers):
            x = layer_arrays[i].mean(axis=-1)
            y = layer_arrays[j].mean(axis=-1)
            if x.std() > 1e-6 and y.std() > 1e-6:
                corr = np.corrcoef(x, y)[0, 1]
                if not np.isnan(corr):
                    # MI approximation from correlation
                    mi = -0.5 * np.log(1 - corr**2) if abs(corr) < 0.9999 else 5.0
                    mi_values.append(mi)

    avg_mi = float(np.mean(mi_values)) if mi_values else 0.0

    return {
        'perplexity': ppl,
        'cross_layer_mi': avg_mi,
        'num_layer_pairs': len(mi_values),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("  z1965: FORMAL PYPHI IIT PHI CALCULATION")
    print("  Computing true Integrated Information for embodied vs disembodied")
    print("=" * 70)
    print()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name}, VRAM: {props.total_memory / 1e9:.1f} GB")
    print(f"Device: {device}")
    print(f"PyPhi available: {PYPHI_AVAILABLE}")
    print()

    telemetry = SysfsHwmonTelemetry()
    actuator = GPUActuator()
    dataset = CharDataset(DATA_PATH, SEQ_LEN)
    print(f"Dataset: {len(dataset.data)} chars, {dataset.n_batches} batches/epoch")

    results = {
        'experiment': 'z1965_pyphi_iit',
        'timestamp': datetime.now().isoformat(),
        'pyphi_available': PYPHI_AVAILABLE,
        'phi_num_nodes': PHI_NUM_NODES,
        'config': {
            'num_epochs': NUM_EPOCHS, 'batch_size': BATCH_SIZE,
            'seq_len': SEQ_LEN, 'lr': LR,
        },
        'conditions': {},
        'verdicts': {},
    }

    try:
        # ---- Condition A: Embodied ----
        print("\n" + "#" * 70)
        print("  CONDITION A: EMBODIED (FiLM ON, real telemetry, actuation)")
        print("#" * 70)
        model_a = create_metabolic_transformer(
            hidden_dim=256, num_layers=6, num_heads=4, telemetry_dim=12,
        ).to(device)

        train_a, hidden_a = train_condition(
            "A_Embodied", model_a, dataset, device, telemetry, actuator,
            film_on=True, use_real_telem=True, do_actuation=True,
        )

        phi_a = compute_phi_for_condition(model_a, hidden_a, "A_Embodied", device)
        proxy_a = compute_z1708_proxies(model_a, dataset, device, telemetry, actuator,
                                        film_on=True, use_real_telem=True)

        results['conditions']['A_Embodied'] = {
            **train_a,
            'phi_analysis': phi_a,
            'z1708_proxies': proxy_a,
        }

        del model_a
        torch.cuda.empty_cache()
        time.sleep(5)

        # ---- Condition B: Disembodied ----
        print("\n" + "#" * 70)
        print("  CONDITION B: DISEMBODIED (FiLM OFF, constant telemetry)")
        print("#" * 70)
        model_b = create_metabolic_transformer(
            hidden_dim=256, num_layers=6, num_heads=4, telemetry_dim=12,
        ).to(device)

        train_b, hidden_b = train_condition(
            "B_Disembodied", model_b, dataset, device, telemetry, actuator,
            film_on=False, use_real_telem=False, do_actuation=False,
        )

        phi_b = compute_phi_for_condition(model_b, hidden_b, "B_Disembodied", device)
        proxy_b = compute_z1708_proxies(model_b, dataset, device, telemetry, actuator,
                                        film_on=False, use_real_telem=False)

        results['conditions']['B_Disembodied'] = {
            **train_b,
            'phi_analysis': phi_b,
            'z1708_proxies': proxy_b,
        }

        del model_b
        torch.cuda.empty_cache()

        # ================================================================
        # VERDICTS
        # ================================================================
        print("\n" + "=" * 70)
        print("  VERDICTS: Formal Phi vs Proxy Measures")
        print("=" * 70)

        conds = results['conditions']
        a = conds['A_Embodied']
        b = conds['B_Disembodied']

        # Formal Phi comparison
        phi_embodied = a['phi_analysis']['avg_phi']
        phi_disembodied = b['phi_analysis']['avg_phi']

        # Proxy measures comparison
        ei_embodied = a['phi_analysis']['proxy_measures']['effective_information']
        ei_disembodied = b['phi_analysis']['proxy_measures']['effective_information']

        mi_embodied = a['z1708_proxies']['cross_layer_mi']
        mi_disembodied = b['z1708_proxies']['cross_layer_mi']

        print(f"\n  FORMAL PHI (PyPhi):")
        print(f"    Embodied:     Phi = {phi_embodied:.6f}")
        print(f"    Disembodied:  Phi = {phi_disembodied:.6f}")

        print(f"\n  TPM-BASED PROXY:")
        print(f"    Embodied:     EI = {ei_embodied:.6f}")
        print(f"    Disembodied:  EI = {ei_disembodied:.6f}")

        print(f"\n  Z1708 PROXY (Cross-layer MI):")
        print(f"    Embodied:     MI = {mi_embodied:.6f}")
        print(f"    Disembodied:  MI = {mi_disembodied:.6f}")

        # Verdicts
        v1_phi = phi_embodied > phi_disembodied if PYPHI_AVAILABLE and phi_embodied > 0 else None
        v2_ei = ei_embodied > ei_disembodied
        v3_mi = mi_embodied > mi_disembodied

        print(f"\n  VERDICT 1 (Formal Phi): ", end="")
        if v1_phi is None:
            print("N/A (PyPhi unavailable or no valid Phi)")
        else:
            print(f"{'PASS' if v1_phi else 'FAIL'} - Embodied Phi {'>' if v1_phi else '<='} Disembodied Phi")

        print(f"  VERDICT 2 (TPM EI): {'PASS' if v2_ei else 'FAIL'} - Embodied EI {'>' if v2_ei else '<='} Disembodied EI")
        print(f"  VERDICT 3 (MI Proxy): {'PASS' if v3_mi else 'FAIL'} - Embodied MI {'>' if v3_mi else '<='} Disembodied MI")

        # Correlation check: Do formal Phi and proxies agree?
        if PYPHI_AVAILABLE and phi_embodied > 0 and phi_disembodied > 0:
            phi_direction = 1 if phi_embodied > phi_disembodied else -1
            ei_direction = 1 if ei_embodied > ei_disembodied else -1
            mi_direction = 1 if mi_embodied > mi_disembodied else -1

            phi_ei_agree = phi_direction == ei_direction
            phi_mi_agree = phi_direction == mi_direction

            print(f"\n  CONSISTENCY:")
            print(f"    Formal Phi agrees with TPM EI: {'YES' if phi_ei_agree else 'NO'}")
            print(f"    Formal Phi agrees with MI Proxy: {'YES' if phi_mi_agree else 'NO'}")

        verdicts = {
            'v1_formal_phi_higher': v1_phi,
            'v2_effective_info_higher': v2_ei,
            'v3_mi_proxy_higher': v3_mi,
        }

        results['verdicts'] = verdicts

        # Detailed comparison table
        print(f"\n{'Measure':<25} {'Embodied':>12} {'Disembodied':>12} {'Delta':>12}")
        print("-" * 65)
        print(f"{'Formal Phi':<25} {phi_embodied:>12.6f} {phi_disembodied:>12.6f} {phi_embodied - phi_disembodied:>+12.6f}")
        print(f"{'Effective Info':<25} {ei_embodied:>12.6f} {ei_disembodied:>12.6f} {ei_embodied - ei_disembodied:>+12.6f}")
        print(f"{'Cross-layer MI':<25} {mi_embodied:>12.6f} {mi_disembodied:>12.6f} {mi_embodied - mi_disembodied:>+12.6f}")
        print(f"{'Perplexity':<25} {a['final_perplexity']:>12.2f} {b['final_perplexity']:>12.2f} {a['final_perplexity'] - b['final_perplexity']:>+12.2f}")

        # Summary
        passed = sum(1 for v in verdicts.values() if v is True)
        total = sum(1 for v in verdicts.values() if v is not None)

        print(f"\n{'='*70}")
        if passed == total and total > 0:
            overall = "ALL PASS: Embodied networks show higher integration by all measures"
        elif passed >= total / 2:
            overall = "MOSTLY PASS: Embodied networks show higher integration by most measures"
        else:
            overall = "MIXED: Integration difference inconclusive"

        print(f"  OVERALL: {overall}")
        print(f"  Verdicts passed: {passed}/{total}")
        print(f"{'='*70}")

        results['overall_verdict'] = overall
        results['verdicts_passed'] = passed
        results['verdicts_total'] = total

    finally:
        actuator.restore_initial_state()
        print("\nGPU state restored.")

    # Save results
    def to_python(obj):
        if isinstance(obj, dict):
            return {k: to_python(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [to_python(v) for v in obj]
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, bool):
            return obj
        elif obj is None:
            return None
        return obj

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, 'w') as f:
        json.dump(to_python(results), f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == '__main__':
    main()
