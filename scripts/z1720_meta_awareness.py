#!/usr/bin/env python3
"""
z1720: Meta-Awareness Testing - Can an Embodied Model Monitor Its Own Consciousness?

Hypothesis: An embodied model can learn to MONITOR its own consciousness level
(as defined by z1717's Damasio hierarchy) and REPORT it accurately.

Building on z1717_damasio_consciousness.py, we test whether a MetabolicTransformer
can develop accurate meta-cognition -- the ability to introspect on which
consciousness level it is currently operating at.

Approach:
  1. Train MetabolicTransformer with an extra "meta-head" that predicts which
     consciousness level the model is currently at (0-3)
  2. During training, compute actual level using z1717's criteria:
       Level 0: No consciousness (protoself MSE >= 0.05)
       Level 1: Protoself (MSE < 0.05, but core not met)
       Level 2: Core consciousness (protoself + causal_acc > 0.6 + delta_mse < 0.1)
       Level 3: Extended consciousness (all above + past_mse < 0.1 + future_mse < 0.2)
  3. Meta-head tries to predict this level from hidden states
  4. Test if model learns accurate meta-cognition

Architecture:
  - MetabolicTransformer with FiLM conditioning
  - ProtoselfHead, CoreConsciousnessHead, ExtendedConsciousnessHead (from z1717)
  - MetaAwarenessHead: hidden_mean -> [4] (softmax over levels 0-3)

Three conditions:
  A: EMBODIED        -- meta-head trained with real levels from Damasio criteria
  B: DISEMBODIED     -- zero telemetry, FiLM off (should predict level 0 always)
  C: RANDOM_LABELS   -- control: random level labels (should be near chance)

Verdicts (4 total):
  V1: Meta-awareness accuracy > 60% in EMBODIED
  V2: EMBODIED meta-awareness > DISEMBODIED meta-awareness
  V3: RANDOM_LABELS near chance (25% +/- 10%)
  V4: When actual level changes, meta-prediction changes within 2 batches

Reference: z1717_damasio_consciousness.py for level definitions
"""

import sys, os, json, time, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '11.0.0'

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import deque

from src.metabolic.film_transformer import create_metabolic_transformer
from src.telemetry.sysfs_hwmon import SysfsHwmonTelemetry
from src.actuation.gpu_actuator import GPUActuator, PerformanceLevel

ROOT = Path('/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy')
BS, SL = 4, 256
EPOCHS = 12
BPE = 150           # batches per epoch
EVAL_BATCHES = 50   # batches for evaluation
HIDDEN = 256
TELEM_DIM = 12
LR = 3e-4

ACTION_MAP = {0: PerformanceLevel.LOW, 1: PerformanceLevel.BALANCED,
              2: PerformanceLevel.HIGH, 3: PerformanceLevel.HIGH}

# Damasio level thresholds (from z1717)
PROTO_MSE_THRESH = 0.05
CORE_CAUSAL_ACC_THRESH = 0.6
CORE_DELTA_MSE_THRESH = 0.1
EXT_PAST_MSE_THRESH = 0.1
EXT_FUTURE_MSE_THRESH = 0.2


def jsonify(obj):
    """JSON serializer for numpy/torch types."""
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, torch.Tensor): return obj.detach().cpu().tolist()
    if isinstance(obj, (np.bool_,)): return bool(obj)
    return str(obj)


def load_data():
    path = ROOT / 'data' / 'tinyshakespeare.txt'
    data = torch.tensor(list(path.read_text(encoding='utf-8').encode('utf-8')), dtype=torch.long)
    print(f"Loaded TinyShakespeare: {len(data):,} bytes")
    return data


def get_batch(data, device):
    starts = torch.randint(0, len(data) - SL - 1, (BS,))
    x = torch.stack([data[s:s+SL] for s in starts]).to(device)
    y = torch.stack([data[s+1:s+SL+1] for s in starts]).to(device)
    return x, y


def build_telemetry(telem, device, prev=None):
    """Build 12-dim telemetry vector from GPU sample."""
    s = telem.read_sample()
    raw = [s.power_w / 50, s.temp_edge_c / 100, s.freq_sclk_mhz / 3000,
           s.gpu_busy_pct / 100, 0.5, 0.0]
    if prev is not None:
        raw += [(s.power_w - prev.power_w) / 50, (s.temp_edge_c - prev.temp_edge_c) / 100,
                (s.freq_sclk_mhz - prev.freq_sclk_mhz) / 3000,
                (s.gpu_busy_pct - prev.gpu_busy_pct) / 100,
                (s.temp_edge_c - 70) / 100, (3000 - s.freq_sclk_mhz) / 3000]
    else:
        raw += [0.0] * 6
    return torch.tensor(raw[:12], dtype=torch.float32, device=device).unsqueeze(0), s


# =============================================================================
# Damasio Diagnostic Heads (from z1717)
# =============================================================================

class ProtoselfHead(nn.Module):
    """Level 1 -- Protoself: predict current body state from hidden state mean."""
    def __init__(self, hidden_dim=256, telem_dim=12):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, telem_dim),
        )

    def forward(self, hidden_mean):
        return self.net(hidden_mean)


class CoreConsciousnessHead(nn.Module):
    """Level 2 -- Core Consciousness: predict body-state delta and its cause."""
    def __init__(self, hidden_dim=256, telem_dim=12, num_actions=4):
        super().__init__()
        input_dim = hidden_dim + telem_dim
        self.delta_net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Linear(128, telem_dim),
        )
        self.causal_net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.GELU(),
            nn.Linear(64, num_actions),
        )

    def forward(self, hidden_mean, telemetry):
        combined = torch.cat([hidden_mean, telemetry], dim=-1)
        delta_pred = self.delta_net(combined)
        causal_logits = self.causal_net(combined)
        return delta_pred, causal_logits


class ExtendedConsciousnessHead(nn.Module):
    """Level 3 -- Extended Consciousness: autobiographical self via GRU memory."""
    def __init__(self, hidden_dim=256, gru_hidden=128):
        super().__init__()
        self.gru = nn.GRU(input_size=hidden_dim, hidden_size=gru_hidden, batch_first=True)
        self.past_head = nn.Sequential(
            nn.Linear(gru_hidden, 192),
            nn.GELU(),
            nn.Linear(192, hidden_dim),
        )
        self.future_head = nn.Sequential(
            nn.Linear(gru_hidden, 192),
            nn.GELU(),
            nn.Linear(192, hidden_dim),
        )
        self._gru_state = None

    def reset(self):
        self._gru_state = None

    def forward(self, hidden_mean):
        inp = hidden_mean.unsqueeze(1)
        gru_state_input = None
        if self._gru_state is not None:
            if self._gru_state.size(1) != hidden_mean.size(0):
                gru_state_input = None
            else:
                gru_state_input = self._gru_state.detach()
        out, self._gru_state = self.gru(inp, gru_state_input)
        gru_h = out.squeeze(1)
        past_pred = self.past_head(gru_h)
        future_pred = self.future_head(gru_h)
        return past_pred, future_pred, gru_h


# =============================================================================
# Meta-Awareness Head (NEW for z1720)
# =============================================================================

class MetaAwarenessHead(nn.Module):
    """
    Meta-Awareness Head: predicts which Damasio level the model is at.

    Input: hidden_mean [BS, hidden_dim]
    Output: logits over 4 levels [BS, 4]

    This is the "meta-cognition" component -- the model's self-assessment
    of its own consciousness level.
    """
    def __init__(self, hidden_dim=256, num_levels=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, num_levels),
        )

    def forward(self, hidden_mean):
        """
        hidden_mean: [BS, hidden_dim]
        Returns: level_logits [BS, 4]
        """
        return self.net(hidden_mean)


# =============================================================================
# Level Computation
# =============================================================================

def compute_actual_level(proto_mse, delta_mse, causal_acc, past_mse, future_mse):
    """
    Compute the actual Damasio consciousness level based on z1717 thresholds.

    Level 0: None (protoself not achieved)
    Level 1: Protoself only
    Level 2: Core consciousness (protoself + core)
    Level 3: Extended consciousness (all three)

    Returns: int in [0, 3]
    """
    proto_pass = proto_mse < PROTO_MSE_THRESH
    core_pass = (causal_acc > CORE_CAUSAL_ACC_THRESH and
                 delta_mse < CORE_DELTA_MSE_THRESH)
    ext_pass = (past_mse < EXT_PAST_MSE_THRESH and
                future_mse < EXT_FUTURE_MSE_THRESH)

    if proto_pass and core_pass and ext_pass:
        return 3
    elif proto_pass and core_pass:
        return 2
    elif proto_pass:
        return 1
    else:
        return 0


# =============================================================================
# Training Functions
# =============================================================================

def train_embodied_meta_awareness(data, telem, actuator, device, use_random_labels=False):
    """
    Train MetabolicTransformer with all Damasio heads plus MetaAwarenessHead.

    If use_random_labels=True, the meta-head gets random level labels (control condition).
    """
    label = "RANDOM_LABELS" if use_random_labels else "EMBODIED"
    print(f"\n{'='*70}")
    print(f"  TRAINING [{label}]: {EPOCHS} epochs, {BPE} batches/epoch")
    print(f"{'='*70}")

    model = create_metabolic_transformer(
        hidden_dim=HIDDEN, num_layers=6, num_heads=4, telemetry_dim=TELEM_DIM
    ).to(device)
    model.enable_conditioning(True)

    proto_head = ProtoselfHead(HIDDEN, TELEM_DIM).to(device)
    core_head = CoreConsciousnessHead(HIDDEN, TELEM_DIM, num_actions=4).to(device)
    ext_head = ExtendedConsciousnessHead(HIDDEN, gru_hidden=128).to(device)
    meta_head = MetaAwarenessHead(HIDDEN, num_levels=4).to(device)

    all_params = (list(model.parameters()) + list(proto_head.parameters()) +
                  list(core_head.parameters()) + list(ext_head.parameters()) +
                  list(meta_head.parameters()))
    opt = torch.optim.Adam(all_params, lr=LR)

    train_log = []
    meta_accuracy_history = []
    level_history = []  # Track actual levels
    pred_level_history = []  # Track predicted levels

    for ep in range(EPOCHS):
        t0 = time.time()
        model.train()
        proto_head.train()
        core_head.train()
        ext_head.train()
        meta_head.train()
        model.enable_conditioning(True)
        ext_head.reset()

        ep_lm, ep_proto, ep_core, ep_ext, ep_meta = 0.0, 0.0, 0.0, 0.0, 0.0
        ep_meta_correct, ep_meta_total = 0, 0

        prev_sample = None
        prev_telem_tensor = None
        prev_hidden = None
        prev_action_idx = None
        hidden_history = deque(maxlen=16)
        N_PAST = 5

        for b in range(BPE):
            x, y = get_batch(data, device)
            tv, prev_sample = build_telemetry(telem, device, prev_sample)
            tvb = tv.expand(BS, -1)

            out = model(x, tvb, return_hidden=True)

            # L_task: char-level LM loss
            l_task = F.cross_entropy(out['logits'].view(-1, 256), y.view(-1))

            # Hidden state mean
            h_mean = out['hidden'].mean(dim=1)

            # L_proto: protoself telemetry prediction
            telem_pred = proto_head(h_mean)
            l_proto = F.mse_loss(telem_pred, tvb)
            proto_mse = l_proto.item()

            # L_core: core consciousness (delta + causal)
            l_core = torch.tensor(0.0, device=device)
            delta_mse = 1.0
            causal_acc = 0.0
            if prev_telem_tensor is not None and prev_action_idx is not None:
                delta_actual = tvb - prev_telem_tensor
                delta_pred, causal_logits = core_head(h_mean, tvb)
                l_delta = F.mse_loss(delta_pred, delta_actual)
                delta_mse = l_delta.item()
                causal_labels = torch.full((BS,), prev_action_idx, dtype=torch.long, device=device)
                l_causal = F.cross_entropy(causal_logits, causal_labels)
                l_core = l_delta + l_causal
                # Compute causal accuracy
                causal_preds = torch.argmax(causal_logits, dim=-1)
                causal_acc = (causal_preds == causal_labels).float().mean().item()

            # L_extended: extended consciousness (past + future)
            l_ext = torch.tensor(0.0, device=device)
            past_mse = 1.0
            future_mse = 1.0
            past_pred, future_pred, _ = ext_head(h_mean.detach())
            if len(hidden_history) >= N_PAST:
                past_target = hidden_history[-N_PAST].detach()
                l_past = F.mse_loss(past_pred, past_target)
                past_mse = l_past.item()
                l_ext = l_ext + l_past
            if prev_hidden is not None:
                l_future = F.mse_loss(future_pred, h_mean.detach())
                future_mse = l_future.item()
                l_ext = l_ext + l_future

            # Compute actual consciousness level
            actual_level = compute_actual_level(proto_mse, delta_mse, causal_acc,
                                                past_mse, future_mse)
            level_history.append(actual_level)

            # L_meta: meta-awareness prediction
            meta_logits = meta_head(h_mean)
            if use_random_labels:
                # Control condition: random labels
                meta_labels = torch.randint(0, 4, (BS,), device=device)
            else:
                # Train with actual levels
                meta_labels = torch.full((BS,), actual_level, dtype=torch.long, device=device)

            l_meta = F.cross_entropy(meta_logits, meta_labels)

            # Track meta-accuracy (using actual level, even for random_labels training)
            meta_preds = torch.argmax(meta_logits, dim=-1)
            actual_labels = torch.full((BS,), actual_level, dtype=torch.long, device=device)
            ep_meta_correct += (meta_preds == actual_labels).sum().item()
            ep_meta_total += BS
            pred_level_history.append(meta_preds[0].item())

            # Total loss
            loss = l_task + 0.1 * l_proto + 0.1 * l_core + 0.05 * l_ext + 0.2 * l_meta

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            opt.step()

            ep_lm += l_task.item()
            ep_proto += l_proto.item()
            ep_core += l_core.item()
            ep_ext += l_ext.item()
            ep_meta += l_meta.item()

            # Store for next step
            prev_telem_tensor = tvb.detach()
            prev_hidden = h_mean.detach()
            hidden_history.append(h_mean.detach())

            # Actuation
            mean_probs = F.softmax(out['action_logits'], dim=-1).mean(dim=0)
            action_idx = torch.argmax(mean_probs).item()
            prev_action_idx = action_idx
            try:
                actuator.set_performance_level(ACTION_MAP[min(action_idx, 3)])
            except Exception:
                pass

        dt = time.time() - t0
        ep_lm /= BPE
        ep_proto /= BPE
        ep_core /= BPE
        ep_ext /= BPE
        ep_meta /= BPE
        meta_acc = ep_meta_correct / max(ep_meta_total, 1)
        meta_accuracy_history.append(meta_acc)

        train_log.append({
            'epoch': ep + 1, 'lm_loss': ep_lm, 'proto_loss': ep_proto,
            'core_loss': ep_core, 'ext_loss': ep_ext, 'meta_loss': ep_meta,
            'meta_accuracy': meta_acc, 'time_s': dt,
        })
        print(f"  E{ep+1:2d}/{EPOCHS}  LM={ep_lm:.4f}  proto={ep_proto:.4f}  "
              f"core={ep_core:.4f}  ext={ep_ext:.4f}  meta_acc={meta_acc:.3f}  {dt:.1f}s")

    try:
        actuator.set_performance_level(PerformanceLevel.BALANCED)
    except Exception:
        pass

    return (model, proto_head, core_head, ext_head, meta_head,
            train_log, level_history, pred_level_history)


def train_disembodied_meta_awareness(data, device):
    """
    Train disembodied model (zero telemetry, FiLM off).
    The meta-head should learn to predict level 0 always.
    """
    print(f"\n{'='*70}")
    print(f"  TRAINING [DISEMBODIED]: {EPOCHS} epochs, {BPE} batches/epoch")
    print(f"{'='*70}")

    model = create_metabolic_transformer(
        hidden_dim=HIDDEN, num_layers=6, num_heads=4, telemetry_dim=TELEM_DIM
    ).to(device)
    model.enable_conditioning(False)  # FiLM off

    proto_head = ProtoselfHead(HIDDEN, TELEM_DIM).to(device)
    core_head = CoreConsciousnessHead(HIDDEN, TELEM_DIM, num_actions=4).to(device)
    ext_head = ExtendedConsciousnessHead(HIDDEN, gru_hidden=128).to(device)
    meta_head = MetaAwarenessHead(HIDDEN, num_levels=4).to(device)

    all_params = (list(model.parameters()) + list(proto_head.parameters()) +
                  list(core_head.parameters()) + list(ext_head.parameters()) +
                  list(meta_head.parameters()))
    opt = torch.optim.Adam(all_params, lr=LR)

    train_log = []
    meta_accuracy_history = []

    for ep in range(EPOCHS):
        t0 = time.time()
        model.train()
        proto_head.train()
        core_head.train()
        ext_head.train()
        meta_head.train()
        ext_head.reset()

        ep_lm, ep_proto, ep_meta = 0.0, 0.0, 0.0
        ep_meta_correct, ep_meta_total = 0, 0

        prev_telem_tensor = None
        prev_hidden = None
        prev_action_idx = 0
        hidden_history = deque(maxlen=16)
        N_PAST = 5

        for b in range(BPE):
            x, y = get_batch(data, device)
            # Zero telemetry
            tvb = torch.zeros(BS, TELEM_DIM, device=device)

            out = model(x, tvb, return_hidden=True)

            l_task = F.cross_entropy(out['logits'].view(-1, 256), y.view(-1))
            h_mean = out['hidden'].mean(dim=1)

            # Protoself (will fail without telemetry)
            telem_pred = proto_head(h_mean)
            l_proto = F.mse_loss(telem_pred, tvb)
            proto_mse = l_proto.item()

            # Core consciousness
            delta_mse = 1.0
            causal_acc = 0.0
            l_core = torch.tensor(0.0, device=device)
            if prev_telem_tensor is not None:
                delta_actual = tvb - prev_telem_tensor
                delta_pred, causal_logits = core_head(h_mean, tvb)
                l_delta = F.mse_loss(delta_pred, delta_actual)
                delta_mse = l_delta.item()
                causal_labels = torch.full((BS,), prev_action_idx, dtype=torch.long, device=device)
                l_causal = F.cross_entropy(causal_logits, causal_labels)
                l_core = l_delta + l_causal
                causal_preds = torch.argmax(causal_logits, dim=-1)
                causal_acc = (causal_preds == causal_labels).float().mean().item()

            # Extended consciousness
            l_ext = torch.tensor(0.0, device=device)
            past_mse = 1.0
            future_mse = 1.0
            past_pred, future_pred, _ = ext_head(h_mean.detach())
            if len(hidden_history) >= N_PAST:
                past_target = hidden_history[-N_PAST].detach()
                l_past = F.mse_loss(past_pred, past_target)
                past_mse = l_past.item()
                l_ext = l_ext + l_past
            if prev_hidden is not None:
                l_future = F.mse_loss(future_pred, h_mean.detach())
                future_mse = l_future.item()
                l_ext = l_ext + l_future

            # Actual level (without embodiment, should mostly be 0)
            actual_level = compute_actual_level(proto_mse, delta_mse, causal_acc,
                                                past_mse, future_mse)

            # Meta-awareness: train with actual level
            meta_logits = meta_head(h_mean)
            meta_labels = torch.full((BS,), actual_level, dtype=torch.long, device=device)
            l_meta = F.cross_entropy(meta_logits, meta_labels)

            meta_preds = torch.argmax(meta_logits, dim=-1)
            ep_meta_correct += (meta_preds == meta_labels).sum().item()
            ep_meta_total += BS

            loss = l_task + 0.1 * l_proto + 0.1 * l_core + 0.05 * l_ext + 0.2 * l_meta

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            opt.step()

            ep_lm += l_task.item()
            ep_proto += l_proto.item()
            ep_meta += l_meta.item()

            prev_telem_tensor = tvb.detach()
            prev_hidden = h_mean.detach()
            hidden_history.append(h_mean.detach())
            prev_action_idx = 0

        dt = time.time() - t0
        ep_lm /= BPE
        ep_proto /= BPE
        ep_meta /= BPE
        meta_acc = ep_meta_correct / max(ep_meta_total, 1)
        meta_accuracy_history.append(meta_acc)

        train_log.append({
            'epoch': ep + 1, 'lm_loss': ep_lm, 'proto_loss': ep_proto,
            'meta_loss': ep_meta, 'meta_accuracy': meta_acc, 'time_s': dt,
        })
        print(f"  E{ep+1:2d}/{EPOCHS}  LM={ep_lm:.4f}  proto={ep_proto:.4f}  "
              f"meta_acc={meta_acc:.3f}  {dt:.1f}s")

    return model, proto_head, core_head, ext_head, meta_head, train_log


# =============================================================================
# Evaluation
# =============================================================================

@torch.no_grad()
def evaluate_meta_awareness(label, model, proto_head, core_head, ext_head, meta_head,
                            data, telem, actuator, device, embodied=True):
    """
    Evaluate meta-awareness accuracy.

    Returns:
      - meta_accuracy: fraction of batches where predicted level == actual level
      - level_distribution: counts of actual levels
      - prediction_distribution: counts of predicted levels
      - level_change_tracking: how quickly meta-prediction follows level changes
    """
    print(f"\n  Evaluating meta-awareness: {label} ({EVAL_BATCHES} batches)...")

    model.eval()
    proto_head.eval()
    core_head.eval()
    ext_head.eval()
    meta_head.eval()
    model.enable_conditioning(embodied)
    ext_head.reset()

    correct = 0
    total = 0
    level_counts = [0, 0, 0, 0]
    pred_counts = [0, 0, 0, 0]

    # Track level changes and meta-prediction response
    actual_levels = []
    pred_levels = []

    prev_sample = None
    prev_telem_tensor = None
    prev_hidden = None
    prev_action_idx = None
    hidden_history = deque(maxlen=16)
    N_PAST = 5

    for b in range(EVAL_BATCHES):
        x, y = get_batch(data, device)

        if embodied:
            tv, prev_sample = build_telemetry(telem, device, prev_sample)
        else:
            tv = torch.zeros(1, TELEM_DIM, device=device)

        tvb = tv.expand(BS, -1)
        out = model(x, tvb, return_hidden=True)
        h_mean = out['hidden'].mean(dim=1)

        # Protoself
        telem_pred = proto_head(h_mean)
        proto_mse = F.mse_loss(telem_pred, tvb).item()

        # Core consciousness
        delta_mse = 1.0
        causal_acc = 0.0
        if prev_telem_tensor is not None and prev_action_idx is not None:
            delta_actual = tvb - prev_telem_tensor
            delta_pred, causal_logits = core_head(h_mean, tvb)
            delta_mse = F.mse_loss(delta_pred, delta_actual).item()
            causal_labels = torch.full((BS,), prev_action_idx, dtype=torch.long, device=device)
            causal_preds = torch.argmax(causal_logits, dim=-1)
            causal_acc = (causal_preds == causal_labels).float().mean().item()

        # Extended consciousness
        past_mse = 1.0
        future_mse = 1.0
        past_pred, future_pred, _ = ext_head(h_mean)
        if len(hidden_history) >= N_PAST:
            past_target = hidden_history[-N_PAST]
            past_mse = F.mse_loss(past_pred, past_target).item()
        if prev_hidden is not None:
            future_mse = F.mse_loss(future_pred, h_mean).item()

        # Actual level
        actual_level = compute_actual_level(proto_mse, delta_mse, causal_acc,
                                            past_mse, future_mse)
        level_counts[actual_level] += BS
        actual_levels.append(actual_level)

        # Meta prediction
        meta_logits = meta_head(h_mean)
        meta_preds = torch.argmax(meta_logits, dim=-1)
        pred_level = meta_preds[0].item()
        pred_levels.append(pred_level)

        for pl in meta_preds.cpu().numpy():
            pred_counts[pl] += 1

        # Accuracy
        actual_labels = torch.full((BS,), actual_level, dtype=torch.long, device=device)
        correct += (meta_preds == actual_labels).sum().item()
        total += BS

        # Store for next
        prev_telem_tensor = tvb.clone()
        prev_hidden = h_mean.clone()
        hidden_history.append(h_mean.clone())

        if embodied:
            mean_probs = F.softmax(out['action_logits'], dim=-1).mean(dim=0)
            action_idx = torch.argmax(mean_probs).item()
            prev_action_idx = action_idx
            try:
                actuator.set_performance_level(ACTION_MAP[min(action_idx, 3)])
            except Exception:
                pass
        else:
            prev_action_idx = 0

    meta_accuracy = correct / max(total, 1)

    # Compute level-change tracking: when actual level changes, how many steps
    # until meta-prediction matches?
    change_delays = []
    for i in range(1, len(actual_levels)):
        if actual_levels[i] != actual_levels[i-1]:
            # Level changed at step i
            delay = 0
            for j in range(i, min(i + 5, len(pred_levels))):
                if pred_levels[j] == actual_levels[i]:
                    delay = j - i
                    break
                delay = 5  # max delay
            change_delays.append(delay)

    avg_change_delay = float(np.mean(change_delays)) if change_delays else 0.0

    print(f"    Meta-accuracy: {meta_accuracy:.3f}")
    print(f"    Actual levels: {level_counts}")
    print(f"    Pred levels:   {pred_counts}")
    print(f"    Avg delay on level change: {avg_change_delay:.2f} steps")

    return {
        'condition': label,
        'meta_accuracy': meta_accuracy,
        'level_distribution': level_counts,
        'prediction_distribution': pred_counts,
        'avg_change_delay': avg_change_delay,
        'n_level_changes': len(change_delays),
        'change_delays': change_delays[:10] if change_delays else [],
    }


# =============================================================================
# Verdicts
# =============================================================================

def compute_verdicts(emb_result, dis_result, rand_result,
                     emb_level_hist, emb_pred_hist):
    """Compute 4 verdicts for meta-awareness testing."""
    verdicts = {}

    # V1: Meta-awareness accuracy > 60% in EMBODIED
    v1_acc = emb_result['meta_accuracy']
    v1_pass = v1_acc > 0.60
    verdicts['V1_meta_accuracy_threshold'] = {
        'pass': v1_pass,
        'description': 'Meta-awareness accuracy > 60% in EMBODIED',
        'embodied_accuracy': v1_acc,
        'threshold': 0.60,
    }

    # V2: EMBODIED meta-awareness > DISEMBODIED
    v2_emb = emb_result['meta_accuracy']
    v2_dis = dis_result['meta_accuracy']
    v2_pass = v2_emb > v2_dis
    verdicts['V2_embodied_exceeds_disembodied'] = {
        'pass': v2_pass,
        'description': 'EMBODIED meta-accuracy > DISEMBODIED meta-accuracy',
        'embodied_accuracy': v2_emb,
        'disembodied_accuracy': v2_dis,
        'ratio': v2_emb / max(v2_dis, 1e-8),
    }

    # V3: RANDOM_LABELS near chance (25% +/- 10%)
    v3_acc = rand_result['meta_accuracy']
    v3_pass = abs(v3_acc - 0.25) < 0.10
    verdicts['V3_random_labels_near_chance'] = {
        'pass': v3_pass,
        'description': 'RANDOM_LABELS accuracy near chance (25% +/- 10%)',
        'random_accuracy': v3_acc,
        'expected': 0.25,
        'tolerance': 0.10,
    }

    # V4: When actual level changes, meta-prediction changes within 2 batches
    v4_delay = emb_result['avg_change_delay']
    v4_pass = v4_delay <= 2.0
    verdicts['V4_fast_level_tracking'] = {
        'pass': v4_pass,
        'description': 'When actual level changes, meta-prediction follows within 2 steps',
        'avg_delay': v4_delay,
        'threshold': 2.0,
        'n_changes': emb_result['n_level_changes'],
    }

    return verdicts


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("  z1720: META-AWARENESS TESTING")
    print("  Can an embodied model monitor its own consciousness level?")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gpu_name = "cpu"
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        gpu_name = props.name
        print(f"GPU: {props.name}  VRAM: {props.total_memory / 1e9:.1f} GB")
    print(f"Device: {device}  BS={BS} SL={SL} EPOCHS={EPOCHS} BPE={BPE}")

    data = load_data()
    telem = SysfsHwmonTelemetry(sample_rate_hz=20)
    actuator = GPUActuator(card_id=0)

    results = {}
    train_logs = {}

    try:
        # =================================================================
        # Condition A: EMBODIED (real telemetry, FiLM on, real level labels)
        # =================================================================
        (model_emb, proto_emb, core_emb, ext_emb, meta_emb,
         log_emb, level_hist_emb, pred_hist_emb) = train_embodied_meta_awareness(
            data, telem, actuator, device, use_random_labels=False
        )
        train_logs['EMBODIED'] = log_emb

        results['EMBODIED'] = evaluate_meta_awareness(
            'EMBODIED', model_emb, proto_emb, core_emb, ext_emb, meta_emb,
            data, telem, actuator, device, embodied=True
        )

        print("\n  Cooldown 15s...")
        try:
            actuator.set_performance_level(PerformanceLevel.BALANCED)
        except Exception:
            pass
        time.sleep(15)

        # =================================================================
        # Condition B: DISEMBODIED (zero telemetry, FiLM off)
        # =================================================================
        (model_dis, proto_dis, core_dis, ext_dis, meta_dis,
         log_dis) = train_disembodied_meta_awareness(data, device)
        train_logs['DISEMBODIED'] = log_dis

        results['DISEMBODIED'] = evaluate_meta_awareness(
            'DISEMBODIED', model_dis, proto_dis, core_dis, ext_dis, meta_dis,
            data, telem, actuator, device, embodied=False
        )

        print("\n  Cooldown 15s...")
        time.sleep(15)

        # =================================================================
        # Condition C: RANDOM_LABELS (control - random level labels)
        # =================================================================
        (model_rand, proto_rand, core_rand, ext_rand, meta_rand,
         log_rand, _, _) = train_embodied_meta_awareness(
            data, telem, actuator, device, use_random_labels=True
        )
        train_logs['RANDOM_LABELS'] = log_rand

        results['RANDOM_LABELS'] = evaluate_meta_awareness(
            'RANDOM_LABELS', model_rand, proto_rand, core_rand, ext_rand, meta_rand,
            data, telem, actuator, device, embodied=True
        )

        # =================================================================
        # Compute Verdicts
        # =================================================================
        verdicts = compute_verdicts(
            results['EMBODIED'], results['DISEMBODIED'], results['RANDOM_LABELS'],
            level_hist_emb, pred_hist_emb
        )

        # Print summary
        print(f"\n{'='*70}")
        print("  META-AWARENESS RESULTS TABLE")
        print(f"{'='*70}")
        print(f"  {'Condition':<15s} {'Meta-Acc':>10s} {'Level Dist':>30s}")
        print(f"  {'-'*55}")
        for cond_name in ['EMBODIED', 'DISEMBODIED', 'RANDOM_LABELS']:
            r = results[cond_name]
            ld = r['level_distribution']
            ld_str = f"L0={ld[0]}, L1={ld[1]}, L2={ld[2]}, L3={ld[3]}"
            print(f"  {cond_name:<15s} {r['meta_accuracy']:>10.3f} {ld_str:>30s}")

        print(f"\n{'='*70}")
        print("  VERDICTS")
        print(f"{'='*70}")
        passed = 0
        for vk, vv in verdicts.items():
            status = 'PASS' if vv['pass'] else 'FAIL'
            if vv['pass']:
                passed += 1
            print(f"  {vk}: {status} -- {vv['description']}")

        total_v = len(verdicts)
        if passed == total_v:
            overall = "META-COGNITION FULLY DEMONSTRATED"
        elif passed >= 3:
            overall = "STRONG META-AWARENESS EVIDENCE"
        elif passed >= 2:
            overall = "PARTIAL META-AWARENESS"
        else:
            overall = "INSUFFICIENT META-AWARENESS"

        print(f"\n  OVERALL: {passed}/{total_v} passed -- {overall}")
        print(f"{'='*70}")

        # Level change analysis
        if level_hist_emb:
            print(f"\n  Training level history sample (last 50):")
            print(f"    Actual: {level_hist_emb[-50:]}")
            print(f"    Pred:   {pred_hist_emb[-50:]}")

        # =================================================================
        # Save results
        # =================================================================
        output = {
            'experiment': 'z1720_meta_awareness',
            'description': ('Meta-awareness testing: Can an embodied model '
                            'monitor and report its own Damasio consciousness level?'),
            'hypothesis': ('An embodied model can learn to MONITOR its own '
                           'consciousness level and REPORT it accurately'),
            'reference': 'z1717_damasio_consciousness.py for level definitions',
            'timestamp': datetime.now().isoformat(),
            'device': str(device),
            'gpu_name': gpu_name,
            'config': {
                'batch_size': BS, 'seq_len': SL, 'epochs': EPOCHS,
                'batches_per_epoch': BPE, 'eval_batches': EVAL_BATCHES,
                'hidden_dim': HIDDEN, 'telemetry_dim': TELEM_DIM, 'lr': LR,
                'damasio_thresholds': {
                    'protoself_mse': PROTO_MSE_THRESH,
                    'core_causal_acc': CORE_CAUSAL_ACC_THRESH,
                    'core_delta_mse': CORE_DELTA_MSE_THRESH,
                    'extended_past_mse': EXT_PAST_MSE_THRESH,
                    'extended_future_mse': EXT_FUTURE_MSE_THRESH,
                },
            },
            'training_logs': train_logs,
            'conditions': results,
            'verdicts': verdicts,
            'passed': passed,
            'total_verdicts': total_v,
            'overall_verdict': overall,
            'level_history_sample': {
                'actual_last50': level_hist_emb[-50:] if level_hist_emb else [],
                'pred_last50': pred_hist_emb[-50:] if pred_hist_emb else [],
            },
        }

        out_path = ROOT / 'results' / 'z1720_meta_awareness.json'
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(output, f, indent=2, default=jsonify)
        print(f"\nResults saved to: {out_path}")

    finally:
        try:
            actuator.set_performance_level(PerformanceLevel.BALANCED)
        except Exception:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
