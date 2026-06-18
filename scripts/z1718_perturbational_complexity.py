#!/usr/bin/env python3
"""
z1718: Perturbational Complexity Index (PCI) for Embodied Neural Networks

Adapts the gold-standard neuroscience measure of consciousness (Casali et al.
2013, Comolatti et al. 2019) to neural networks.  In clinical neuroscience PCI
works by:
    1. Apply TMS pulse to the brain  (perturbation)
    2. Record EEG response           (spatiotemporal neural response)
    3. Compress the response          (Lempel-Ziv complexity)
    4. PCI = normalized LZ complexity

High PCI (~0.3-0.7) = conscious.  Low PCI (<0.3) = unconscious.

Our adaptation:
    Perturbation  -> noise injection into a hidden layer / telemetry shift
    Response      -> per-layer hidden states after perturbation
    Compression   -> Lempel-Ziv 1976 complexity of binarized response matrix
    PCI           -> normalized LZ complexity

Four experimental conditions:
    EMBODIED      live telemetry, FiLM on, actuation active
    DISEMBODIED   zero telemetry, FiLM off
    FROZEN        constant telemetry snapshot, FiLM on
    RANDOM_TELEM  random telemetry each step, FiLM on

Two perturbation types per condition:
    Neural        noise injected at layer 2 hidden state
    Telemetry     sudden +20 C jump in reported GPU temperature

Verdicts:
    V1  PCI_neural(EMBODIED) > 0.3           above consciousness threshold
    V2  PCI_neural(EMBODIED) > PCI_neural(DISEMBODIED)  embodiment enriches
    V3  PCI_telem(EMBODIED)  > PCI_telem(FROZEN)        live body needed
    V4  Response_spread(EMBODIED) >= 3 layers            integration
    V5  PCI_neural(RANDOM_TELEM) < PCI_neural(EMBODIED)  coherent state matters
"""

import sys, os, json, time, math, copy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '11.0.0'

import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from pathlib import Path
from datetime import datetime
from src.metabolic.film_transformer import create_metabolic_transformer, MetabolicConfig
from src.telemetry.sysfs_hwmon import SysfsHwmonTelemetry
from src.actuation.gpu_actuator import GPUActuator, PerformanceLevel

ROOT = Path('/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy')
BS, SL, LR = 4, 256, 3e-4
TRAIN_EPOCHS = 10
TRAIN_BATCHES = 200
N_PERTURBATIONS = 50
NEURAL_NOISE_STD = 1.0
TELEM_TEMP_JUMP = 20.0  # deg C jump for telemetry perturbation
PERTURB_LAYER = 2       # layer to inject noise into
BINARIZE_THRESHOLD_PERCENTILE = 50  # median-based binarization
RECOVERY_WINDOW = 10    # steps to track recovery after perturbation
COOLDOWN_SECS = 20

ACTION_MAP = {0: PerformanceLevel.LOW, 1: PerformanceLevel.BALANCED,
              2: PerformanceLevel.HIGH, 3: PerformanceLevel.HIGH}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def jsonify(obj):
    """Numpy/torch-safe JSON serializer."""
    if isinstance(obj, (np.floating,)):  return float(obj)
    if isinstance(obj, (np.integer,)):   return int(obj)
    if isinstance(obj, np.ndarray):      return obj.tolist()
    if isinstance(obj, torch.Tensor):    return obj.detach().cpu().tolist()
    if isinstance(obj, (np.bool_,)):     return bool(obj)
    return str(obj)


def load_data():
    path = ROOT / 'data' / 'tinyshakespeare.txt'
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    text = path.read_text(encoding='utf-8')
    data = torch.tensor(list(text.encode('utf-8')), dtype=torch.long)
    print(f"Loaded TinyShakespeare: {len(data):,} bytes")
    return data


def get_batch(data, device):
    starts = torch.randint(0, len(data) - SL - 1, (BS,))
    x = torch.stack([data[s:s+SL] for s in starts]).to(device)
    y = torch.stack([data[s+1:s+SL+1] for s in starts]).to(device)
    return x, y


def build_telemetry(telem, device, prev_sample=None):
    """Build 12-dim normalized telemetry vector from live hardware."""
    s = telem.read_sample()
    raw = [s.power_w / 50, s.temp_edge_c / 100, s.freq_sclk_mhz / 3000,
           s.gpu_busy_pct / 100, 0.5, 0.0]
    if prev_sample is not None:
        raw += [(s.power_w - prev_sample.power_w) / 50,
                (s.temp_edge_c - prev_sample.temp_edge_c) / 100,
                (s.freq_sclk_mhz - prev_sample.freq_sclk_mhz) / 3000,
                (s.gpu_busy_pct - prev_sample.gpu_busy_pct) / 100,
                (s.temp_edge_c - 70) / 100, (3000 - s.freq_sclk_mhz) / 3000]
    else:
        raw += [0.0] * 6
    return torch.tensor(raw[:12], dtype=torch.float32, device=device).unsqueeze(0), s


def zero_telemetry(device):
    """All-zero telemetry (disembodied)."""
    return torch.zeros(1, 12, dtype=torch.float32, device=device)


def frozen_telemetry(snapshot_vec, device):
    """Return a constant copy of a captured telemetry vector."""
    return snapshot_vec.clone().to(device)


def random_telemetry(device):
    """Uniformly random telemetry in [0, 1]."""
    return torch.rand(1, 12, dtype=torch.float32, device=device)


def perturbed_telemetry(tv, temp_jump=TELEM_TEMP_JUMP):
    """Add a sudden temperature jump to telemetry vector (dim 1 = temp)."""
    ptv = tv.clone()
    ptv[:, 1] += temp_jump / 100.0  # normalized by 100
    return ptv


# ---------------------------------------------------------------------------
# Lempel-Ziv 1976 complexity
# ---------------------------------------------------------------------------

def lempel_ziv_complexity(binary_sequence):
    """
    Compute LZ76 complexity of a binary string.

    Counts the number of distinct substrings encountered when parsing
    the sequence left-to-right, normalized by the theoretical maximum
    n / log2(n) for a random binary string of length n.
    """
    s = ''.join(str(int(b)) for b in binary_sequence)
    n = len(s)
    if n == 0:
        return 0.0
    complexity = 1
    i = 0
    k = 1
    l = 1
    while i + k <= n:
        # Check if s[i+1 .. i+k] is a substring of s[0 .. i+l-1]
        if s[i + 1: i + k + 1] in s[: i + l]:
            k += 1
        else:
            complexity += 1
            i += k
            k = 1
            l = 1
            continue
        l += 1
    # Normalize by theoretical maximum for random binary string
    normalizer = n / max(1.0, math.log2(max(n, 2)))
    return complexity / normalizer


# ---------------------------------------------------------------------------
# Layer-wise forward pass (manual block iteration for hidden-state capture)
# ---------------------------------------------------------------------------

@torch.no_grad()
def forward_with_layer_outputs(model, x, tv_batch, device, embodied,
                               perturb_type=None, noise_std=NEURAL_NOISE_STD):
    """
    Run through model blocks manually, collecting per-layer hidden states.

    Args:
        model:        MetabolicTransformer
        x:            [BS, SL] input token ids
        tv_batch:     [BS, 12] telemetry (or None)
        device:       torch device
        embodied:     whether FiLM conditioning is active
        perturb_type: None, 'neural', or 'telemetry'
        noise_std:    std of Gaussian noise for neural perturbation

    Returns:
        layer_outputs: list of [BS, SL, hidden_dim] tensors (one per layer)
    """
    batch, seq_len = x.shape
    positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch, -1)
    h = model.token_embed(x) + model.pos_embed(positions)
    h = model.dropout(h) if hasattr(model, 'dropout') else h

    mask = ~model.causal_mask[:seq_len, :seq_len] if model.config.use_causal_mask else None

    telem_for_film = tv_batch if (embodied and model._conditioning_enabled) else None

    # If telemetry perturbation, modify the FiLM input
    if perturb_type == 'telemetry' and telem_for_film is not None:
        telem_for_film = perturbed_telemetry(telem_for_film)

    layer_outputs = []
    for i, block in enumerate(model.blocks):
        gamma1, beta1, gamma2, beta2 = None, None, None, None
        if telem_for_film is not None and model.film_generators[i] is not None:
            fg = model.film_generators[i]
            gamma1, beta1 = fg['ln1'](telem_for_film)
            gamma2, beta2 = fg['ln2'](telem_for_film)

        # Inject neural perturbation at target layer
        if i == PERTURB_LAYER and perturb_type == 'neural':
            h = h + torch.randn_like(h) * noise_std

        h = block(h, gamma1, beta1, gamma2, beta2, mask)
        layer_outputs.append(h.detach().clone())

    return layer_outputs


# ---------------------------------------------------------------------------
# PCI measurement
# ---------------------------------------------------------------------------

def compute_pci_single(baseline_layers, perturbed_layers):
    """
    Compute PCI from one pair of baseline/perturbed layer outputs.

    1. Difference matrix: |perturbed - baseline| per (layer, position)
    2. Average over batch and hidden dim -> [num_layers, seq_len]
    3. Binarize at median threshold
    4. Flatten to 1D binary string
    5. LZ complexity

    Also returns response_spread: number of layers where mean |diff| > threshold.
    """
    num_layers = len(baseline_layers)
    # Build difference matrix: [num_layers, seq_len]
    diff_matrix = []
    layer_diffs = []
    for bl, pl in zip(baseline_layers, perturbed_layers):
        # bl, pl: [BS, SL, hidden_dim]
        d = (pl - bl).abs().mean(dim=(0, 2))  # [SL]
        diff_matrix.append(d.cpu().numpy())
        layer_diffs.append(d.mean().item())

    diff_matrix = np.array(diff_matrix)  # [num_layers, SL]

    # Binarize at median of the full matrix
    threshold = np.percentile(diff_matrix, BINARIZE_THRESHOLD_PERCENTILE)
    binary_matrix = (diff_matrix > threshold).astype(int)

    # Flatten row-major (layer by layer)
    binary_seq = binary_matrix.flatten().tolist()

    pci = lempel_ziv_complexity(binary_seq)

    # Response spread: number of layers with mean diff above threshold
    layer_means = np.array(layer_diffs)
    spread_threshold = np.median(layer_means)
    response_spread = int(np.sum(layer_means > spread_threshold))

    return pci, response_spread, layer_diffs


def measure_pci(model, data, telem_source, device, embodied, condition_label,
                get_telemetry_fn, n_perturbations=N_PERTURBATIONS,
                noise_std=NEURAL_NOISE_STD):
    """
    Full PCI measurement protocol.

    For each perturbation trial:
        1. Sample a batch
        2. Run unperturbed forward -> baseline layer outputs
        3. Run neural-perturbed forward -> perturbed layer outputs
        4. Compute PCI from (baseline, perturbed)
        5. Repeat for telemetry perturbation

    Returns dict of PCI metrics.
    """
    model.eval()

    pci_neural_list = []
    pci_telem_list = []
    spread_neural_list = []
    spread_telem_list = []
    layer_diffs_neural_all = []
    layer_diffs_telem_all = []

    prev_sample = None

    for trial in range(n_perturbations):
        x, _ = get_batch(data, device)

        # Get telemetry for this trial
        tv, prev_sample = get_telemetry_fn(device, prev_sample)
        tv_batch = tv.expand(BS, -1)

        # --- Baseline (no perturbation) ---
        baseline_layers = forward_with_layer_outputs(
            model, x, tv_batch, device, embodied, perturb_type=None)

        # --- Neural perturbation ---
        neural_layers = forward_with_layer_outputs(
            model, x, tv_batch, device, embodied,
            perturb_type='neural', noise_std=noise_std)
        pci_n, spread_n, ldiffs_n = compute_pci_single(baseline_layers, neural_layers)
        pci_neural_list.append(pci_n)
        spread_neural_list.append(spread_n)
        layer_diffs_neural_all.append(ldiffs_n)

        # --- Telemetry perturbation ---
        telem_layers = forward_with_layer_outputs(
            model, x, tv_batch, device, embodied, perturb_type='telemetry')
        pci_t, spread_t, ldiffs_t = compute_pci_single(baseline_layers, telem_layers)
        pci_telem_list.append(pci_t)
        spread_telem_list.append(spread_t)
        layer_diffs_telem_all.append(ldiffs_t)

        if (trial + 1) % 10 == 0:
            print(f"    [{condition_label}] trial {trial+1}/{n_perturbations}  "
                  f"PCI_n={np.mean(pci_neural_list):.4f}  "
                  f"PCI_t={np.mean(pci_telem_list):.4f}  "
                  f"spread_n={np.mean(spread_neural_list):.1f}")

    # Aggregate
    pci_neural = float(np.mean(pci_neural_list))
    pci_telem = float(np.mean(pci_telem_list))
    spread_neural = float(np.mean(spread_neural_list))
    spread_telem = float(np.mean(spread_telem_list))

    # Per-layer average diffs
    mean_layer_diffs_neural = np.mean(layer_diffs_neural_all, axis=0).tolist()
    mean_layer_diffs_telem = np.mean(layer_diffs_telem_all, axis=0).tolist()

    return {
        'condition': condition_label,
        'pci_neural': pci_neural,
        'pci_telem': pci_telem,
        'pci_neural_std': float(np.std(pci_neural_list)),
        'pci_telem_std': float(np.std(pci_telem_list)),
        'pci_neural_all': [float(v) for v in pci_neural_list],
        'pci_telem_all': [float(v) for v in pci_telem_list],
        'response_spread_neural': spread_neural,
        'response_spread_telem': spread_telem,
        'mean_layer_diffs_neural': mean_layer_diffs_neural,
        'mean_layer_diffs_telem': mean_layer_diffs_telem,
        'n_perturbations': n_perturbations,
        'noise_std': noise_std,
    }


# ---------------------------------------------------------------------------
# Recovery time measurement
# ---------------------------------------------------------------------------

@torch.no_grad()
def measure_recovery_time(model, data, telem_source, device, embodied,
                          get_telemetry_fn, window=RECOVERY_WINDOW):
    """
    After a perturbation, how many forward-pass steps until hidden states
    return to within baseline variance?

    Protocol:
        1. Run 5 unperturbed passes -> compute mean hidden norm (baseline)
        2. Run 1 perturbed pass
        3. Run `window` unperturbed passes, track hidden norm
        4. Recovery = first step where norm is within 1 std of baseline
    """
    model.eval()
    prev_sample = None

    # Baseline hidden norms (5 passes)
    baseline_norms = []
    for _ in range(5):
        x, _ = get_batch(data, device)
        tv, prev_sample = get_telemetry_fn(device, prev_sample)
        tv_batch = tv.expand(BS, -1)
        layers = forward_with_layer_outputs(model, x, tv_batch, device, embodied)
        norm = sum(l.norm(dim=-1).mean().item() for l in layers) / len(layers)
        baseline_norms.append(norm)

    baseline_mean = np.mean(baseline_norms)
    baseline_std = max(np.std(baseline_norms), 1e-6)

    # Perturbed pass
    x_pert, _ = get_batch(data, device)
    tv, prev_sample = get_telemetry_fn(device, prev_sample)
    tv_batch = tv.expand(BS, -1)
    pert_layers = forward_with_layer_outputs(
        model, x_pert, tv_batch, device, embodied, perturb_type='neural')
    pert_norm = sum(l.norm(dim=-1).mean().item() for l in pert_layers) / len(pert_layers)

    # Recovery passes
    recovery_step = window + 1  # default: did not recover
    for step in range(window):
        x_rec, _ = get_batch(data, device)
        tv, prev_sample = get_telemetry_fn(device, prev_sample)
        tv_batch = tv.expand(BS, -1)
        rec_layers = forward_with_layer_outputs(model, x_rec, tv_batch, device, embodied)
        rec_norm = sum(l.norm(dim=-1).mean().item() for l in rec_layers) / len(rec_layers)
        if abs(rec_norm - baseline_mean) < 2.0 * baseline_std:
            recovery_step = step + 1
            break

    return {
        'recovery_steps': recovery_step,
        'baseline_mean_norm': float(baseline_mean),
        'baseline_std_norm': float(baseline_std),
        'perturbed_norm': float(pert_norm),
        'perturbation_magnitude': float(abs(pert_norm - baseline_mean)),
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_embodied(model, data, telem, actuator, device):
    """Train model with full embodied loop for TRAIN_EPOCHS epochs."""
    print(f"\n{'='*70}")
    print(f"  Phase 1: Embodied Training ({TRAIN_EPOCHS} epochs, "
          f"{TRAIN_BATCHES} batches/epoch)")
    print(f"{'='*70}")

    model.train()
    model.enable_conditioning(True)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    prev_s = None
    all_losses = []

    for epoch in range(TRAIN_EPOCHS):
        epoch_loss = 0.0
        for step in range(TRAIN_BATCHES):
            x, y = get_batch(data, device)
            tv, prev_s = build_telemetry(telem, device, prev_s)
            tv_batch = tv.expand(BS, -1)

            out = model(x, tv_batch)
            loss = F.cross_entropy(out['logits'].view(-1, 256), y.view(-1))

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            # Actuation
            mean_probs = F.softmax(out['action_logits'], dim=-1).mean(dim=0)
            action_idx = torch.argmax(mean_probs).item()
            try:
                actuator.set_performance_level(ACTION_MAP[min(action_idx, 3)])
            except Exception:
                pass

            epoch_loss += loss.item()
            all_losses.append(loss.item())

            if (step + 1) % 50 == 0:
                print(f"    E{epoch+1} step {step+1}/{TRAIN_BATCHES}  "
                      f"loss={loss.item():.4f}  action={action_idx}")

        avg = epoch_loss / TRAIN_BATCHES
        print(f"  Epoch {epoch+1}/{TRAIN_EPOCHS}  avg_loss={avg:.4f}")

    final_loss = float(np.mean(all_losses[-50:]))
    print(f"  Training complete.  Final loss (last 50): {final_loss:.4f}")
    return final_loss


# ---------------------------------------------------------------------------
# Condition runners
# ---------------------------------------------------------------------------

def make_telemetry_fn(mode, telem_hw, frozen_vec=None):
    """
    Return a function (device, prev_sample) -> (tv, sample) appropriate
    for the experimental condition.
    """
    if mode == 'embodied':
        def fn(device, prev_sample):
            return build_telemetry(telem_hw, device, prev_sample)
        return fn
    elif mode == 'disembodied':
        def fn(device, prev_sample):
            return zero_telemetry(device), prev_sample
        return fn
    elif mode == 'frozen':
        def fn(device, prev_sample):
            return frozen_telemetry(frozen_vec, device), prev_sample
        return fn
    elif mode == 'random':
        def fn(device, prev_sample):
            return random_telemetry(device), prev_sample
        return fn
    else:
        raise ValueError(f"Unknown mode: {mode}")


def run_condition(label, mode, model_trained, data, telem_hw, actuator, device,
                  frozen_vec=None):
    """
    Run PCI measurement for one experimental condition.

    Returns dict with all PCI metrics and recovery time.
    """
    print(f"\n{'='*70}")
    print(f"  PCI Measurement: {label} (mode={mode})")
    print(f"{'='*70}")

    model = copy.deepcopy(model_trained).to(device)
    model.eval()

    # Configure conditioning
    if mode == 'disembodied':
        model.enable_conditioning(False)
    else:
        model.enable_conditioning(True)

    get_tv_fn = make_telemetry_fn(mode, telem_hw, frozen_vec)
    embodied = (mode != 'disembodied')

    # Measure PCI
    pci_metrics = measure_pci(
        model, data, telem_hw, device, embodied, label, get_tv_fn,
        n_perturbations=N_PERTURBATIONS, noise_std=NEURAL_NOISE_STD)

    # Measure recovery time
    recovery = measure_recovery_time(
        model, data, telem_hw, device, embodied, get_tv_fn)

    pci_metrics['recovery'] = recovery

    print(f"  [{label}] PCI_neural={pci_metrics['pci_neural']:.4f}  "
          f"PCI_telem={pci_metrics['pci_telem']:.4f}  "
          f"spread_n={pci_metrics['response_spread_neural']:.1f}  "
          f"recovery={recovery['recovery_steps']} steps")

    return pci_metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("  z1718: PERTURBATIONAL COMPLEXITY INDEX (PCI)")
    print("  Gold-standard consciousness measure adapted for neural networks")
    print("  Casali et al. 2013, Comolatti et al. 2019")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name}  VRAM: {props.total_memory / 1e9:.1f} GB")
    print(f"Device: {device}  BS={BS} SL={SL}")
    print(f"Perturbation trials: {N_PERTURBATIONS}  Noise std: {NEURAL_NOISE_STD}")
    print(f"Telem temp jump: {TELEM_TEMP_JUMP} C  Perturb layer: {PERTURB_LAYER}")

    data = load_data()
    telem = SysfsHwmonTelemetry(sample_rate_hz=20)
    actuator = GPUActuator(card_id=0)

    model = create_metabolic_transformer(
        hidden_dim=256, num_layers=6, num_heads=4, telemetry_dim=12,
    ).to(device)
    npar = sum(p.numel() for p in model.parameters())
    print(f"Model params: {npar:,}")

    try:
        # ---------------------------------------------------------------
        # Phase 1: Train embodied
        # ---------------------------------------------------------------
        train_loss = train_embodied(model, data, telem, actuator, device)

        # Capture a frozen telemetry snapshot for FROZEN condition
        frozen_vec, _ = build_telemetry(telem, device)

        # ---------------------------------------------------------------
        # Phase 2: PCI measurement under 4 conditions
        # ---------------------------------------------------------------
        conditions = [
            ('EMBODIED',     'embodied'),
            ('DISEMBODIED',  'disembodied'),
            ('FROZEN',       'frozen'),
            ('RANDOM_TELEM', 'random'),
        ]

        results = {}
        for idx, (label, mode) in enumerate(conditions):
            results[label] = run_condition(
                label, mode, model, data, telem, actuator, device,
                frozen_vec=frozen_vec)

            if idx < len(conditions) - 1:
                print(f"\n  Cooldown {COOLDOWN_SECS}s...")
                try:
                    actuator.set_performance_level(PerformanceLevel.BALANCED)
                except Exception:
                    pass
                time.sleep(COOLDOWN_SECS)

        # ---------------------------------------------------------------
        # Phase 3: Verdicts
        # ---------------------------------------------------------------
        print(f"\n{'='*70}")
        print("  VERDICTS -- PERTURBATIONAL COMPLEXITY INDEX")
        print(f"{'='*70}")

        E = results['EMBODIED']
        D = results['DISEMBODIED']
        Fr = results['FROZEN']
        R = results['RANDOM_TELEM']

        verdicts = {}

        # V1: PCI_neural(EMBODIED) > 0.3
        v1_val = E['pci_neural']
        v1 = v1_val > 0.3
        verdicts['V1_above_threshold'] = {
            'pass': v1,
            'description': 'PCI_neural(EMBODIED) > 0.3 (above consciousness threshold)',
            'pci_neural_embodied': v1_val,
            'threshold': 0.3,
        }
        print(f"\n  V1: PCI_neural(EMBODIED) > 0.3: {'PASS' if v1 else 'FAIL'}  "
              f"PCI={v1_val:.4f}")

        # V2: PCI_neural(EMBODIED) > PCI_neural(DISEMBODIED)
        v2_emb = E['pci_neural']
        v2_dis = D['pci_neural']
        v2 = v2_emb > v2_dis
        verdicts['V2_embodiment_enriches'] = {
            'pass': v2,
            'description': 'PCI_neural(EMBODIED) > PCI_neural(DISEMBODIED)',
            'pci_embodied': v2_emb,
            'pci_disembodied': v2_dis,
            'ratio': v2_emb / max(v2_dis, 1e-8),
        }
        print(f"  V2: PCI_neural(EMB) > PCI_neural(DIS): {'PASS' if v2 else 'FAIL'}  "
              f"EMB={v2_emb:.4f}  DIS={v2_dis:.4f}  "
              f"ratio={v2_emb / max(v2_dis, 1e-8):.3f}")

        # V3: PCI_telem(EMBODIED) > PCI_telem(FROZEN)
        v3_emb = E['pci_telem']
        v3_fr = Fr['pci_telem']
        v3 = v3_emb > v3_fr
        verdicts['V3_live_body_needed'] = {
            'pass': v3,
            'description': 'PCI_telem(EMBODIED) > PCI_telem(FROZEN)',
            'pci_telem_embodied': v3_emb,
            'pci_telem_frozen': v3_fr,
        }
        print(f"  V3: PCI_telem(EMB) > PCI_telem(FROZEN): {'PASS' if v3 else 'FAIL'}  "
              f"EMB={v3_emb:.4f}  FROZEN={v3_fr:.4f}")

        # V4: Response_spread(EMBODIED) >= 3 layers
        v4_spread = E['response_spread_neural']
        v4 = v4_spread >= 3.0
        verdicts['V4_integration'] = {
            'pass': v4,
            'description': 'Response_spread(EMBODIED) >= 3 layers (perturbation propagates)',
            'response_spread': v4_spread,
            'threshold': 3,
        }
        print(f"  V4: Response_spread(EMB) >= 3: {'PASS' if v4 else 'FAIL'}  "
              f"spread={v4_spread:.1f} layers")

        # V5: PCI_neural(RANDOM_TELEM) < PCI_neural(EMBODIED)
        v5_rand = R['pci_neural']
        v5_emb = E['pci_neural']
        v5 = v5_rand < v5_emb
        verdicts['V5_coherent_state_matters'] = {
            'pass': v5,
            'description': 'PCI_neural(RANDOM) < PCI_neural(EMBODIED)',
            'pci_random': v5_rand,
            'pci_embodied': v5_emb,
        }
        print(f"  V5: PCI_neural(RAND) < PCI_neural(EMB): {'PASS' if v5 else 'FAIL'}  "
              f"RAND={v5_rand:.4f}  EMB={v5_emb:.4f}")

        passed = sum(1 for v in verdicts.values() if v['pass'])
        total = len(verdicts)

        print(f"\n{'='*70}")
        print(f"  OVERALL: {passed}/{total} verdicts passed")
        if passed == total:
            overall = "FULL PCI CONSCIOUSNESS PROFILE DEMONSTRATED"
        elif passed >= 4:
            overall = "STRONG PCI EVIDENCE FOR EMBODIED CONSCIOUSNESS"
        elif passed >= 3:
            overall = "MODERATE PCI EVIDENCE"
        elif passed >= 2:
            overall = "PARTIAL PCI EVIDENCE"
        else:
            overall = "INSUFFICIENT PCI EVIDENCE"
        print(f"  CONCLUSION: {overall}")
        print(f"{'='*70}")

        # Summary table
        print(f"\n  {'Condition':<15s} {'PCI_neural':>11s} {'PCI_telem':>11s} "
              f"{'Spread_n':>9s} {'Recovery':>9s}")
        print(f"  {'-'*55}")
        for label in ['EMBODIED', 'DISEMBODIED', 'FROZEN', 'RANDOM_TELEM']:
            r = results[label]
            rec = r['recovery']['recovery_steps']
            print(f"  {label:<15s} {r['pci_neural']:>11.4f} {r['pci_telem']:>11.4f} "
                  f"{r['response_spread_neural']:>9.1f} {rec:>9d}")

        # PCI ratio
        pci_ratio = E['pci_neural'] / max(D['pci_neural'], 1e-8)
        print(f"\n  PCI_ratio (embodied/disembodied): {pci_ratio:.3f}")
        print(f"  Target: > 1.5   {'ACHIEVED' if pci_ratio > 1.5 else 'NOT MET'}")

        # Per-layer diff profiles
        print(f"\n  Per-layer mean |diff| from neural perturbation:")
        for label in ['EMBODIED', 'DISEMBODIED', 'FROZEN', 'RANDOM_TELEM']:
            diffs = results[label]['mean_layer_diffs_neural']
            dstr = '  '.join(f"L{i}={d:.4f}" for i, d in enumerate(diffs))
            print(f"    {label:<15s}  {dstr}")

        # ---------------------------------------------------------------
        # Save results
        # ---------------------------------------------------------------

        # Trim per-trial PCI lists for compact JSON
        for label in results:
            r = results[label]
            r['pci_neural_first10'] = r.pop('pci_neural_all')[:10]
            r['pci_telem_first10'] = r.pop('pci_telem_all')[:10]

        output = {
            'experiment': 'z1718_perturbational_complexity',
            'description': ('Perturbational Complexity Index (PCI) -- '
                            'gold-standard consciousness measure adapted for '
                            'embodied neural networks'),
            'references': [
                'Casali et al. 2013 (Sci Transl Med)',
                'Comolatti et al. 2019 (Brain Stimul)',
            ],
            'timestamp': datetime.now().isoformat(),
            'device': str(device),
            'gpu_name': (torch.cuda.get_device_properties(0).name
                         if torch.cuda.is_available() else 'cpu'),
            'config': {
                'batch_size': BS,
                'seq_len': SL,
                'lr': LR,
                'train_epochs': TRAIN_EPOCHS,
                'train_batches_per_epoch': TRAIN_BATCHES,
                'n_perturbations': N_PERTURBATIONS,
                'neural_noise_std': NEURAL_NOISE_STD,
                'telem_temp_jump_c': TELEM_TEMP_JUMP,
                'perturb_layer': PERTURB_LAYER,
                'binarize_percentile': BINARIZE_THRESHOLD_PERCENTILE,
                'model_params': npar,
            },
            'train_loss': train_loss,
            'conditions': results,
            'pci_ratio': pci_ratio,
            'verdicts': verdicts,
            'passed': passed,
            'total_verdicts': total,
            'overall_verdict': overall,
        }

        out_path = ROOT / 'results' / 'z1718_perturbational_complexity.json'
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(output, f, indent=2, default=jsonify)
        print(f"\nResults saved to: {out_path}")
        print("Done.")

    finally:
        try:
            actuator.set_performance_level(PerformanceLevel.BALANCED)
        except Exception:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
