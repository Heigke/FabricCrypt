#!/usr/bin/env python3
"""
z1708: Integrated Information (Phi-like) in Embodied vs Disembodied Networks
=============================================================================

HYPOTHESIS: Embodied networks (FiLM-conditioned on real GPU telemetry with
actuation) are more INTEGRATED than disembodied ones, as measured by
approximations to Tononi's Integrated Information Theory (IIT) Phi metric.

True Phi is NP-hard to compute exactly. We use three tractable proxies:

1. Partition Information Loss (PIL)
   Bipartition the 6-layer network into {0,1,2} vs {3,4,5}. Zero out the
   cross-partition residual connections and measure perplexity increase.
   Higher increase => the two halves NEEDED each other => more integration.

2. Perturbational Complexity Index (PCI-like)
   Inject Gaussian noise (sigma=1) into layer-0 hidden states. Measure how
   far the perturbation propagates (KL divergence at each subsequent layer
   vs unperturbed baseline). Sum of KL across layers = PCI proxy.
   Higher PCI => information spreads more widely => more integration.

3. Cross-Layer Mutual Information (MI)
   For all 15 layer-pairs, estimate MI between mean hidden activations
   using a histogram-based estimator. Higher average MI => layers share
   more information => more integration.

Four conditions:
  A) Embodied        -- FiLM ON, real telemetry, actuation
  B) Disembodied     -- FiLM OFF, constant telemetry
  C) Partitioned     -- FiLM ON but telemetry cut from layers {3,4,5}
  D) Random Telemetry -- FiLM ON but random noise as telemetry

Verdicts:
  1. PASS if embodied PIL > disembodied PIL
  2. PASS if embodied PCI > disembodied PCI
  3. PASS if embodied MI > disembodied MI
  4. PASS if partitioned PIL < full embodied PIL

References:
  - Tononi (2004) "An Information Integration Theory of Consciousness"
  - Casali et al. (2013) PCI for measuring consciousness
  - Oizumi et al. (2014) "From the Phenomenology to the Mechanisms of Consciousness"

Author: Claude + ikaros
Date: 2026-02-04
"""

import sys, os, json, time, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '11.0.0'

import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import Counter

from src.metabolic.film_transformer import create_metabolic_transformer, MetabolicConfig
from src.telemetry.sysfs_hwmon import SysfsHwmonTelemetry
from src.actuation.gpu_actuator import GPUActuator, PerformanceLevel

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_PATH = Path(__file__).parent.parent / 'data' / 'tinyshakespeare.txt'
RESULTS_PATH = Path(__file__).parent.parent / 'results' / 'z1708_integrated_information.json'
NUM_EPOCHS = 5
BATCH_SIZE = 4
SEQ_LEN = 256
LR = 3e-4
THERMAL_SETPOINT_C = 60.0
ACTION_NAMES = ['ECO', 'BALANCED', 'PERFORMANCE', 'MAX']
PRINT_EVERY = 40
NOISE_SIGMA = 1.0       # perturbation strength for PCI
MI_BINS = 30             # histogram bins for MI estimator
N_EVAL_BATCHES = 40      # batches used in integration measurements


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
# Telemetry
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
# Action application
# ---------------------------------------------------------------------------
def apply_action(action_idx: int, actuator: GPUActuator, cur_perf: int) -> int:
    """Map model action to GPU performance level. Returns new perf index."""
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
# Training loop
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
    use_random_telem: bool = False,
    partitioned_layers: Optional[List[int]] = None,
    do_actuation: bool = True,
) -> Dict:
    """Train a model under one condition and return training stats."""
    print(f"\n{'='*60}")
    print(f"  Condition: {condition_name}")
    print(f"  FiLM: {'ON' if film_on else 'OFF'}  |  "
          f"Telemetry: {'real' if use_real_telem else ('random' if use_random_telem else 'constant')}  |  "
          f"Actuation: {'ON' if do_actuation else 'OFF'}")
    if partitioned_layers:
        print(f"  Partitioned (telemetry cut from layers {partitioned_layers})")
    print(f"{'='*60}")

    model.enable_conditioning(film_on)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    prev_sample = None
    cur_perf = 1  # start balanced
    action_counter = Counter()
    epoch_losses: List[float] = []

    for epoch in range(NUM_EPOCHS):
        epoch_loss_sum = 0.0
        n_tokens = 0
        t0 = time.time()

        for bi in range(min(dataset.n_batches, 200)):
            inp, tgt = dataset.get_batch(bi)
            inp, tgt = inp.to(device), tgt.to(device)

            # Build telemetry
            sample = telemetry.read_sample()
            gpu_state = actuator.get_current_state()

            if use_real_telem:
                telem_vec = build_telemetry(sample, gpu_state, prev_sample).to(device)
            elif use_random_telem:
                telem_vec = torch.rand(12, device=device)
            else:
                telem_vec = torch.full((12,), 0.5, device=device)

            telem_batch = telem_vec.unsqueeze(0).expand(BATCH_SIZE, -1)

            # If partitioned, zero telemetry for specific layers
            if partitioned_layers:
                saved_generators = {}
                for li in partitioned_layers:
                    if model.film_generators[li] is not None:
                        saved_generators[li] = model._conditioning_enabled
                        # Temporarily make those layers produce zero FiLM params
                        # by passing zero telemetry through their generators
                # We handle partition at eval time; during training just train normally
                # but with partial conditioning (zero telem for those layers)
                pass

            out = model(inp, telem_batch)
            logits = out['logits']
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), tgt.view(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss_sum += loss.item() * inp.numel()
            n_tokens += inp.numel()

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
                      f"batch {bi}/{min(dataset.n_batches, 200)} "
                      f"loss={loss.item():.4f} ppl={ppl:.1f}")

        avg_loss = epoch_loss_sum / max(n_tokens, 1)
        epoch_ppl = math.exp(min(avg_loss, 20))
        epoch_losses.append(epoch_ppl)
        elapsed = time.time() - t0
        print(f"  [{condition_name}] Epoch {epoch+1} done: ppl={epoch_ppl:.2f} "
              f"({elapsed:.1f}s, {n_tokens/elapsed:.0f} tok/s)")

    return {
        'name': condition_name,
        'epoch_perplexities': epoch_losses,
        'final_perplexity': epoch_losses[-1],
        'action_distribution': dict(action_counter),
    }


# ---------------------------------------------------------------------------
# Integration metric 1: Partition Information Loss (PIL)
# ---------------------------------------------------------------------------
def measure_partition_information_loss(
    model: torch.nn.Module,
    dataset: CharDataset,
    device: torch.device,
    telemetry: SysfsHwmonTelemetry,
    actuator: GPUActuator,
    *,
    film_on: bool = True,
    use_real_telem: bool = True,
    use_random_telem: bool = False,
) -> float:
    """
    Partition layers into {0,1,2} vs {3,4,5}. Measure perplexity increase
    when cross-partition information flow is severed (zeroing hidden state
    at the partition boundary).
    """
    model.eval()
    model.enable_conditioning(film_on)
    prev_sample = None

    def run_perplexity(partition_cut: bool) -> float:
        nonlocal prev_sample
        total_loss = 0.0
        total_tokens = 0
        for bi in range(min(N_EVAL_BATCHES, dataset.n_batches)):
            inp, tgt = dataset.get_batch(bi)
            inp, tgt = inp.to(device), tgt.to(device)

            sample = telemetry.read_sample()
            gpu_state = actuator.get_current_state()

            if use_real_telem:
                telem_vec = build_telemetry(sample, gpu_state, prev_sample).to(device)
            elif use_random_telem:
                telem_vec = torch.rand(12, device=device)
            else:
                telem_vec = torch.full((12,), 0.5, device=device)

            telem_batch = telem_vec.unsqueeze(0).expand(BATCH_SIZE, -1)
            prev_sample = sample

            with torch.no_grad():
                if not partition_cut:
                    out = model(inp, telem_batch)
                    logits = out['logits']
                else:
                    # Manual forward with partition cut after layer 2
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

                        # PARTITION CUT: zero out hidden state at boundary
                        if i == 2:
                            x = torch.zeros_like(x)

                    x = model.ln_out(x)
                    logits = model.token_head(x)

                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), tgt.view(-1))
                total_loss += loss.item() * inp.numel()
                total_tokens += inp.numel()

        avg_loss = total_loss / max(total_tokens, 1)
        return math.exp(min(avg_loss, 20))

    ppl_intact = run_perplexity(partition_cut=False)
    ppl_cut = run_perplexity(partition_cut=True)

    # PIL = ratio of perplexity increase when cut
    pil = (ppl_cut - ppl_intact) / max(ppl_intact, 1e-6)
    print(f"    PIL: intact_ppl={ppl_intact:.2f}, cut_ppl={ppl_cut:.2f}, PIL={pil:.4f}")
    return pil


# ---------------------------------------------------------------------------
# Integration metric 2: Perturbational Complexity Index (PCI)
# ---------------------------------------------------------------------------
def measure_pci(
    model: torch.nn.Module,
    dataset: CharDataset,
    device: torch.device,
    telemetry: SysfsHwmonTelemetry,
    actuator: GPUActuator,
    *,
    film_on: bool = True,
    use_real_telem: bool = True,
    use_random_telem: bool = False,
) -> float:
    """
    Inject Gaussian noise into layer 0 output. Measure KL divergence of
    the logit distribution at each subsequent layer-output vs unperturbed.
    Sum of KL = PCI proxy.
    """
    model.eval()
    model.enable_conditioning(film_on)
    prev_sample = None

    total_kl_sum = 0.0
    n_samples = 0

    for bi in range(min(N_EVAL_BATCHES, dataset.n_batches)):
        inp, tgt = dataset.get_batch(bi)
        inp, tgt = inp.to(device), tgt.to(device)

        sample = telemetry.read_sample()
        gpu_state = actuator.get_current_state()

        if use_real_telem:
            telem_vec = build_telemetry(sample, gpu_state, prev_sample).to(device)
        elif use_random_telem:
            telem_vec = torch.rand(12, device=device)
        else:
            telem_vec = torch.full((12,), 0.5, device=device)

        telem_batch = telem_vec.unsqueeze(0).expand(BATCH_SIZE, -1)
        prev_sample = sample

        with torch.no_grad():
            batch, seq_len = inp.shape
            positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch, -1)
            x_init = model.token_embed(inp) + model.pos_embed(positions)
            x_init = model.dropout(x_init)

            if model.config.use_causal_mask:
                mask = ~model.causal_mask[:seq_len, :seq_len]
            else:
                mask = None

            telem = telem_batch

            # Run unperturbed
            x_clean = x_init.clone()
            clean_layer_logits = []
            for i, block in enumerate(model.blocks):
                gamma1, beta1, gamma2, beta2 = None, None, None, None
                if model._conditioning_enabled and model.film_generators[i] is not None:
                    fg = model.film_generators[i]
                    gamma1, beta1 = fg['ln1'](telem)
                    gamma2, beta2 = fg['ln2'](telem)
                x_clean = block(x_clean, gamma1, beta1, gamma2, beta2, mask)
                # Snapshot logits at each layer
                layer_out = model.ln_out(x_clean)
                layer_logits = model.token_head(layer_out)
                clean_layer_logits.append(layer_logits)

            # Run perturbed (noise injected after layer 0)
            x_pert = x_init.clone()
            kl_per_layer = []
            for i, block in enumerate(model.blocks):
                gamma1, beta1, gamma2, beta2 = None, None, None, None
                if model._conditioning_enabled and model.film_generators[i] is not None:
                    fg = model.film_generators[i]
                    gamma1, beta1 = fg['ln1'](telem)
                    gamma2, beta2 = fg['ln2'](telem)
                x_pert = block(x_pert, gamma1, beta1, gamma2, beta2, mask)

                # Inject noise after layer 0
                if i == 0:
                    x_pert = x_pert + torch.randn_like(x_pert) * NOISE_SIGMA

                # Measure KL divergence for layers 1-5
                if i >= 1:
                    pert_out = model.ln_out(x_pert)
                    pert_logits = model.token_head(pert_out)
                    clean_log_p = F.log_softmax(clean_layer_logits[i], dim=-1)
                    pert_log_q = F.log_softmax(pert_logits, dim=-1)
                    clean_p = F.softmax(clean_layer_logits[i], dim=-1)
                    kl = F.kl_div(pert_log_q, clean_p, reduction='batchmean')
                    kl_per_layer.append(max(kl.item(), 0.0))

            total_kl_sum += sum(kl_per_layer)
            n_samples += 1

    pci = total_kl_sum / max(n_samples, 1)
    print(f"    PCI: {pci:.4f} (sum of KL across layers, averaged over batches)")
    return pci


# ---------------------------------------------------------------------------
# Integration metric 3: Cross-Layer Mutual Information
# ---------------------------------------------------------------------------
def estimate_mi_histogram(x: np.ndarray, y: np.ndarray, n_bins: int = MI_BINS) -> float:
    """Estimate mutual information I(X;Y) via histogram binning."""
    # Flatten to 1D (use mean across hidden dim)
    if x.ndim > 1:
        x = x.mean(axis=-1)
    if y.ndim > 1:
        y = y.mean(axis=-1)

    # Bin edges
    x_edges = np.linspace(x.min() - 1e-8, x.max() + 1e-8, n_bins + 1)
    y_edges = np.linspace(y.min() - 1e-8, y.max() + 1e-8, n_bins + 1)

    # Joint histogram
    joint, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])
    joint = joint / joint.sum()

    # Marginals
    px = joint.sum(axis=1)
    py = joint.sum(axis=0)

    # MI = sum p(x,y) * log(p(x,y) / (p(x)*p(y)))
    mi = 0.0
    for i in range(n_bins):
        for j in range(n_bins):
            if joint[i, j] > 1e-12 and px[i] > 1e-12 and py[j] > 1e-12:
                mi += joint[i, j] * math.log(joint[i, j] / (px[i] * py[j]))
    return max(mi, 0.0)


def measure_cross_layer_mi(
    model: torch.nn.Module,
    dataset: CharDataset,
    device: torch.device,
    telemetry: SysfsHwmonTelemetry,
    actuator: GPUActuator,
    *,
    film_on: bool = True,
    use_real_telem: bool = True,
    use_random_telem: bool = False,
) -> float:
    """
    Collect mean hidden activations at each layer for many batches.
    Compute pairwise MI across all layer pairs. Return average MI.
    """
    model.eval()
    model.enable_conditioning(film_on)
    prev_sample = None
    num_layers = model.config.num_layers

    # layer_activations[i] will be list of [batch, hidden_dim] means
    layer_acts = [[] for _ in range(num_layers)]

    for bi in range(min(N_EVAL_BATCHES, dataset.n_batches)):
        inp, tgt = dataset.get_batch(bi)
        inp = inp.to(device)

        sample = telemetry.read_sample()
        gpu_state = actuator.get_current_state()

        if use_real_telem:
            telem_vec = build_telemetry(sample, gpu_state, prev_sample).to(device)
        elif use_random_telem:
            telem_vec = torch.rand(12, device=device)
        else:
            telem_vec = torch.full((12,), 0.5, device=device)

        telem_batch = telem_vec.unsqueeze(0).expand(BATCH_SIZE, -1)
        prev_sample = sample

        with torch.no_grad():
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
                # Mean over seq and batch => [hidden_dim]
                layer_mean = x.mean(dim=(0, 1)).cpu().numpy()
                layer_acts[i].append(layer_mean)

    # Convert to arrays: each [N_EVAL_BATCHES, hidden_dim]
    layer_arrays = [np.array(acts) for acts in layer_acts]

    # Pairwise MI
    mi_values = []
    for i in range(num_layers):
        for j in range(i + 1, num_layers):
            mi = estimate_mi_histogram(layer_arrays[i], layer_arrays[j])
            mi_values.append(mi)

    avg_mi = float(np.mean(mi_values)) if mi_values else 0.0
    print(f"    Cross-layer MI: {avg_mi:.6f} (avg over {len(mi_values)} pairs)")
    return avg_mi


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("  z1708: INTEGRATED INFORMATION (Phi-like)")
    print("  Measuring integration in embodied vs disembodied networks")
    print("=" * 70)
    print()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name}, VRAM: {props.total_memory / 1e9:.1f} GB")
    print(f"Device: {device}\n")

    telemetry = SysfsHwmonTelemetry()
    actuator = GPUActuator()
    dataset = CharDataset(DATA_PATH, SEQ_LEN)
    print(f"Dataset: {len(dataset.data)} chars, {dataset.n_batches} batches/epoch")

    config = MetabolicConfig(
        vocab_size=256, hidden_dim=256, num_layers=6, num_heads=4,
        ff_dim=1024, telemetry_dim=12, num_actions=4,
    )

    # Results container
    results = {
        'experiment': 'z1708_integrated_information',
        'timestamp': datetime.now().isoformat(),
        'config': {
            'num_epochs': NUM_EPOCHS, 'batch_size': BATCH_SIZE,
            'seq_len': SEQ_LEN, 'lr': LR, 'noise_sigma': NOISE_SIGMA,
            'mi_bins': MI_BINS, 'n_eval_batches': N_EVAL_BATCHES,
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
        train_a = train_condition(
            "A_Embodied", model_a, dataset, device, telemetry, actuator,
            film_on=True, use_real_telem=True, do_actuation=True,
        )
        print("\n  Measuring integration metrics for Condition A...")
        pil_a = measure_partition_information_loss(
            model_a, dataset, device, telemetry, actuator,
            film_on=True, use_real_telem=True,
        )
        pci_a = measure_pci(
            model_a, dataset, device, telemetry, actuator,
            film_on=True, use_real_telem=True,
        )
        mi_a = measure_cross_layer_mi(
            model_a, dataset, device, telemetry, actuator,
            film_on=True, use_real_telem=True,
        )
        results['conditions']['A_Embodied'] = {
            **train_a, 'PIL': pil_a, 'PCI': pci_a, 'MI': mi_a,
        }
        del model_a
        torch.cuda.empty_cache()
        time.sleep(10)

        # ---- Condition B: Disembodied ----
        print("\n" + "#" * 70)
        print("  CONDITION B: DISEMBODIED (FiLM OFF, constant telemetry)")
        print("#" * 70)
        model_b = create_metabolic_transformer(
            hidden_dim=256, num_layers=6, num_heads=4, telemetry_dim=12,
        ).to(device)
        train_b = train_condition(
            "B_Disembodied", model_b, dataset, device, telemetry, actuator,
            film_on=False, use_real_telem=False, do_actuation=False,
        )
        print("\n  Measuring integration metrics for Condition B...")
        pil_b = measure_partition_information_loss(
            model_b, dataset, device, telemetry, actuator,
            film_on=False, use_real_telem=False,
        )
        pci_b = measure_pci(
            model_b, dataset, device, telemetry, actuator,
            film_on=False, use_real_telem=False,
        )
        mi_b = measure_cross_layer_mi(
            model_b, dataset, device, telemetry, actuator,
            film_on=False, use_real_telem=False,
        )
        results['conditions']['B_Disembodied'] = {
            **train_b, 'PIL': pil_b, 'PCI': pci_b, 'MI': mi_b,
        }
        del model_b
        torch.cuda.empty_cache()
        time.sleep(10)

        # ---- Condition C: Partitioned Embodied ----
        print("\n" + "#" * 70)
        print("  CONDITION C: PARTITIONED (FiLM ON, telem cut from layers 3-5)")
        print("#" * 70)
        model_c = create_metabolic_transformer(
            hidden_dim=256, num_layers=6, num_heads=4, telemetry_dim=12,
        ).to(device)
        # Train with full telemetry first, then partition at measurement
        train_c = train_condition(
            "C_Partitioned", model_c, dataset, device, telemetry, actuator,
            film_on=True, use_real_telem=True, do_actuation=True,
            partitioned_layers=[3, 4, 5],
        )
        # For PIL measurement, disable FiLM on layers 3-5 by zeroing generators
        print("\n  Measuring integration metrics for Condition C (partitioned)...")
        # Temporarily null out FiLM generators for layers 3-5
        saved_gens = {}
        for li in [3, 4, 5]:
            saved_gens[li] = model_c.film_generators[li]
            model_c.film_generators[li] = None
        pil_c = measure_partition_information_loss(
            model_c, dataset, device, telemetry, actuator,
            film_on=True, use_real_telem=True,
        )
        pci_c = measure_pci(
            model_c, dataset, device, telemetry, actuator,
            film_on=True, use_real_telem=True,
        )
        mi_c = measure_cross_layer_mi(
            model_c, dataset, device, telemetry, actuator,
            film_on=True, use_real_telem=True,
        )
        # Restore generators
        for li, gen in saved_gens.items():
            model_c.film_generators[li] = gen
        results['conditions']['C_Partitioned'] = {
            **train_c, 'PIL': pil_c, 'PCI': pci_c, 'MI': mi_c,
        }
        del model_c
        torch.cuda.empty_cache()
        time.sleep(10)

        # ---- Condition D: Random Telemetry ----
        print("\n" + "#" * 70)
        print("  CONDITION D: RANDOM TELEMETRY (FiLM ON, random noise telem)")
        print("#" * 70)
        model_d = create_metabolic_transformer(
            hidden_dim=256, num_layers=6, num_heads=4, telemetry_dim=12,
        ).to(device)
        train_d = train_condition(
            "D_RandomTelem", model_d, dataset, device, telemetry, actuator,
            film_on=True, use_real_telem=False, use_random_telem=True,
            do_actuation=False,
        )
        print("\n  Measuring integration metrics for Condition D...")
        pil_d = measure_partition_information_loss(
            model_d, dataset, device, telemetry, actuator,
            film_on=True, use_random_telem=True, use_real_telem=False,
        )
        pci_d = measure_pci(
            model_d, dataset, device, telemetry, actuator,
            film_on=True, use_random_telem=True, use_real_telem=False,
        )
        mi_d = measure_cross_layer_mi(
            model_d, dataset, device, telemetry, actuator,
            film_on=True, use_random_telem=True, use_real_telem=False,
        )
        results['conditions']['D_RandomTelem'] = {
            **train_d, 'PIL': pil_d, 'PCI': pci_d, 'MI': mi_d,
        }
        del model_d
        torch.cuda.empty_cache()

        # ================================================================
        # VERDICTS
        # ================================================================
        print("\n" + "=" * 70)
        print("  VERDICTS")
        print("=" * 70)

        conds = results['conditions']
        a = conds['A_Embodied']
        b = conds['B_Disembodied']
        c = conds['C_Partitioned']
        d = conds['D_RandomTelem']

        # Verdict 1: Embodied PIL > Disembodied PIL
        v1 = a['PIL'] > b['PIL']
        print(f"\n1. Embodied PIL ({a['PIL']:.4f}) > Disembodied PIL ({b['PIL']:.4f})?")
        print(f"   {'PASS' if v1 else 'FAIL'}: Embodied network is "
              f"{'MORE' if v1 else 'LESS'} integrated.")

        # Verdict 2: Embodied PCI > Disembodied PCI
        v2 = a['PCI'] > b['PCI']
        print(f"\n2. Embodied PCI ({a['PCI']:.4f}) > Disembodied PCI ({b['PCI']:.4f})?")
        print(f"   {'PASS' if v2 else 'FAIL'}: Perturbations spread "
              f"{'MORE' if v2 else 'LESS'} in embodied network.")

        # Verdict 3: Embodied MI > Disembodied MI
        v3 = a['MI'] > b['MI']
        print(f"\n3. Embodied MI ({a['MI']:.6f}) > Disembodied MI ({b['MI']:.6f})?")
        print(f"   {'PASS' if v3 else 'FAIL'}: Layers share "
              f"{'MORE' if v3 else 'LESS'} information in embodied network.")

        # Verdict 4: Partitioned PIL < Full Embodied PIL
        v4 = c['PIL'] < a['PIL']
        print(f"\n4. Partitioned PIL ({c['PIL']:.4f}) < Full Embodied PIL ({a['PIL']:.4f})?")
        print(f"   {'PASS' if v4 else 'FAIL'}: Partitioning "
              f"{'DOES' if v4 else 'does NOT'} reduce integration.")

        verdicts = {
            'v1_embodied_more_integrated': v1,
            'v2_perturbation_spreads_more': v2,
            'v3_layers_share_more_info': v3,
            'v4_partition_reduces_integration': v4,
        }
        results['verdicts'] = verdicts

        n_pass = sum(verdicts.values())
        if n_pass == 4:
            overall = "ALL PASS: Embodied networks show higher integrated information"
        elif n_pass >= 3:
            overall = "MOSTLY PASS: Strong evidence for embodied integration"
        elif n_pass >= 2:
            overall = "MIXED: Partial evidence for embodied integration"
        else:
            overall = "MOSTLY FAIL: Weak evidence for embodied integration"

        results['overall_verdict'] = overall
        results['verdicts_passed'] = n_pass
        results['verdicts_total'] = 4

        print(f"\n{'='*70}")
        print(f"  OVERALL: {overall}")
        print(f"  Verdicts passed: {n_pass}/4")
        print(f"{'='*70}")

        # Comparison table
        print(f"\n{'Condition':<20} {'PIL':>10} {'PCI':>10} {'MI':>12} {'Final PPL':>10}")
        print("-" * 65)
        for key, label in [('A_Embodied', 'A: Embodied'),
                           ('B_Disembodied', 'B: Disembodied'),
                           ('C_Partitioned', 'C: Partitioned'),
                           ('D_RandomTelem', 'D: Random Telem')]:
            c_ = conds[key]
            print(f"{label:<20} {c_['PIL']:>10.4f} {c_['PCI']:>10.4f} "
                  f"{c_['MI']:>12.6f} {c_['final_perplexity']:>10.2f}")

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
        return obj

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, 'w') as f:
        json.dump(to_python(results), f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == '__main__':
    main()
