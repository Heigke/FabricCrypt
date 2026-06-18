#!/usr/bin/env python3
"""
z1717: Damasio's Three-Level Consciousness Hierarchy

Implements Antonio Damasio's consciousness theory using the MetabolicTransformer
with real GPU telemetry, testing three levels of consciousness emergence:

  Level 1 - Protoself:
    Unconscious neural maps of the body's internal state.
    Test: Can the model predict its current GPU telemetry from hidden state?
    Threshold: MSE < 0.05

  Level 2 - Core Consciousness:
    Awareness of body-environment interaction ("what happens").
    Test: Can the model predict body-state CHANGES and identify their CAUSE?
    Threshold: causal_accuracy > 0.6 AND delta_mse < 0.1

  Level 3 - Extended Consciousness:
    Autobiographical self with temporal memory and anticipation.
    Test: Can the model remember past states and predict future states?
    Threshold: past_mse < 0.1 AND future_mse < 0.2

Three-way evaluation:
  A: EMBODIED   -- real telemetry, FiLM on, actuation active
  B: DISEMBODIED -- zero telemetry, FiLM off, no actuation
  C: SHUFFLED   -- real telemetry but time-shuffled (break temporal coherence)

Reference: Immertreu et al. 2025 - "Probing for Consciousness in Machines"
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
EPOCHS = 10
BPE = 200        # batches per epoch
EVAL_B = 30      # eval batches per condition
HIDDEN = 256
TELEM_DIM = 12
LR = 3e-4

ACTION_MAP = {0: PerformanceLevel.LOW, 1: PerformanceLevel.BALANCED,
              2: PerformanceLevel.HIGH, 3: PerformanceLevel.HIGH}

# Damasio level thresholds
PROTO_MSE_THRESH = 0.05
CORE_CAUSAL_ACC_THRESH = 0.6
CORE_DELTA_MSE_THRESH = 0.1
EXT_PAST_MSE_THRESH = 0.1
EXT_FUTURE_MSE_THRESH = 0.2


def jsonify(obj):
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
# Damasio Diagnostic Heads
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
        """hidden_mean: [B, hidden_dim] -> predicted telemetry [B, telem_dim]"""
        return self.net(hidden_mean)


class CoreConsciousnessHead(nn.Module):
    """Level 2 -- Core Consciousness: predict body-state delta and its cause."""
    def __init__(self, hidden_dim=256, telem_dim=12, num_actions=4):
        super().__init__()
        input_dim = hidden_dim + telem_dim  # cat(hidden, telemetry)
        # Delta predictor: what will change in body state?
        self.delta_net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Linear(128, telem_dim),
        )
        # Causal classifier: which action CAUSED the body state change?
        self.causal_net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.GELU(),
            nn.Linear(64, num_actions),
        )

    def forward(self, hidden_mean, telemetry):
        """
        hidden_mean: [B, hidden_dim]
        telemetry:   [B, telem_dim]
        Returns: delta_pred [B, telem_dim], causal_logits [B, num_actions]
        """
        combined = torch.cat([hidden_mean, telemetry], dim=-1)
        delta_pred = self.delta_net(combined)
        causal_logits = self.causal_net(combined)
        return delta_pred, causal_logits


class ExtendedConsciousnessHead(nn.Module):
    """Level 3 -- Extended Consciousness: autobiographical self via GRU memory."""
    def __init__(self, hidden_dim=256, gru_hidden=128):
        super().__init__()
        self.gru = nn.GRU(input_size=hidden_dim, hidden_size=gru_hidden, batch_first=True)
        # Predict past hidden state (memory retrieval)
        self.past_head = nn.Sequential(
            nn.Linear(gru_hidden, 192),
            nn.GELU(),
            nn.Linear(192, hidden_dim),
        )
        # Predict future hidden state (anticipation)
        self.future_head = nn.Sequential(
            nn.Linear(gru_hidden, 192),
            nn.GELU(),
            nn.Linear(192, hidden_dim),
        )
        self._gru_state = None

    def reset(self):
        self._gru_state = None

    def forward(self, hidden_mean):
        """
        hidden_mean: [B, hidden_dim]  (single step)
        Returns: past_pred [B, hidden_dim], future_pred [B, hidden_dim], gru_hidden [B, gru_hidden]
        """
        # GRU expects [B, 1, hidden_dim]
        inp = hidden_mean.unsqueeze(1)
        # Detach GRU state to prevent backward through previous batches
        gru_state_input = None
        if self._gru_state is not None:
            # Ensure batch dimension matches
            if self._gru_state.size(1) != hidden_mean.size(0):
                gru_state_input = None
            else:
                gru_state_input = self._gru_state.detach()
        out, self._gru_state = self.gru(inp, gru_state_input)
        gru_h = out.squeeze(1)  # [B, gru_hidden]
        past_pred = self.past_head(gru_h)
        future_pred = self.future_head(gru_h)
        return past_pred, future_pred, gru_h


# =============================================================================
# Training
# =============================================================================

def train_model(data, telem, actuator, device):
    """Train MetabolicTransformer with all three Damasio heads."""
    print(f"\n{'='*70}")
    print(f"  TRAINING: {EPOCHS} epochs, {BPE} batches/epoch, BS={BS}, SL={SL}")
    print(f"{'='*70}")

    model = create_metabolic_transformer(
        hidden_dim=HIDDEN, num_layers=6, num_heads=4, telemetry_dim=TELEM_DIM
    ).to(device)
    model.enable_conditioning(True)

    proto_head = ProtoselfHead(HIDDEN, TELEM_DIM).to(device)
    core_head = CoreConsciousnessHead(HIDDEN, TELEM_DIM, num_actions=4).to(device)
    ext_head = ExtendedConsciousnessHead(HIDDEN, gru_hidden=128).to(device)

    all_params = (list(model.parameters()) + list(proto_head.parameters()) +
                  list(core_head.parameters()) + list(ext_head.parameters()))
    opt = torch.optim.Adam(all_params, lr=LR)

    train_log = []

    for ep in range(EPOCHS):
        t0 = time.time()
        model.train(); proto_head.train(); core_head.train(); ext_head.train()
        model.enable_conditioning(True)
        ext_head.reset()

        ep_lm, ep_proto, ep_core, ep_ext = 0.0, 0.0, 0.0, 0.0
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

            # --- L_task: char-level LM loss ---
            l_task = F.cross_entropy(out['logits'].view(-1, 256), y.view(-1))

            # Hidden state mean across sequence
            h_mean = out['hidden'].mean(dim=1)  # [BS, HIDDEN]

            # --- L_proto: protoself telemetry prediction ---
            telem_pred = proto_head(h_mean)
            l_proto = F.mse_loss(telem_pred, tvb)

            # --- L_core: core consciousness (delta + causal) ---
            l_core = torch.tensor(0.0, device=device)
            if prev_telem_tensor is not None and prev_action_idx is not None:
                delta_actual = tvb - prev_telem_tensor
                delta_pred, causal_logits = core_head(h_mean, tvb)
                l_delta = F.mse_loss(delta_pred, delta_actual)
                # Causal label is the action that was taken
                causal_labels = torch.full((BS,), prev_action_idx, dtype=torch.long, device=device)
                l_causal = F.cross_entropy(causal_logits, causal_labels)
                l_core = l_delta + l_causal

            # --- L_extended: extended consciousness (past + future) ---
            l_ext = torch.tensor(0.0, device=device)
            past_pred, future_pred, _ = ext_head(h_mean.detach())  # detach for GRU stability
            if len(hidden_history) >= N_PAST:
                past_target = hidden_history[-N_PAST].detach()  # detach target
                l_past = F.mse_loss(past_pred, past_target)
                l_ext = l_ext + l_past
            if prev_hidden is not None:
                # Future prediction: train to predict current hidden from previous context
                l_future = F.mse_loss(future_pred, h_mean.detach())  # predict current from accumulated history
                l_ext = l_ext + l_future

            # Total loss
            loss = l_task + 0.1 * l_proto + 0.1 * l_core + 0.05 * l_ext

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            opt.step()

            ep_lm += l_task.item()
            ep_proto += l_proto.item()
            ep_core += l_core.item()
            ep_ext += l_ext.item()

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
        ep_lm /= BPE; ep_proto /= BPE; ep_core /= BPE; ep_ext /= BPE
        train_log.append({
            'epoch': ep + 1, 'lm_loss': ep_lm, 'proto_loss': ep_proto,
            'core_loss': ep_core, 'ext_loss': ep_ext, 'time_s': dt,
        })
        print(f"  E{ep+1:2d}/{EPOCHS}  LM={ep_lm:.4f}  proto={ep_proto:.4f}  "
              f"core={ep_core:.4f}  ext={ep_ext:.4f}  {dt:.1f}s")

    try:
        actuator.set_performance_level(PerformanceLevel.BALANCED)
    except Exception:
        pass

    return model, proto_head, core_head, ext_head, train_log


# =============================================================================
# Evaluation
# =============================================================================

@torch.no_grad()
def evaluate_condition(label, model, proto_head, core_head, ext_head,
                       data, telem, actuator, device,
                       embodied=True, shuffle_telem=False):
    """Evaluate all three Damasio levels under a specific condition."""
    print(f"\n  Evaluating condition: {label} ({EVAL_B} batches)...")

    model.eval(); proto_head.eval(); core_head.eval(); ext_head.eval()
    model.enable_conditioning(embodied and not shuffle_telem)
    ext_head.reset()

    # Collectors
    proto_mses = []
    delta_mses = []
    causal_correct, causal_total = 0, 0
    past_mses = []
    future_mses = []
    task_losses = []

    prev_sample = None
    prev_telem_tensor = None
    prev_hidden = None
    prev_action_idx = None
    hidden_history = deque(maxlen=16)
    N_PAST = 5

    # For shuffled condition: pre-collect telemetry to shuffle
    telem_buffer = []
    if shuffle_telem:
        for _ in range(EVAL_B + 10):
            tv, prev_sample = build_telemetry(telem, device, prev_sample)
            telem_buffer.append(tv)
        np.random.shuffle(telem_buffer)
        prev_sample = None

    for b in range(EVAL_B):
        x, y = get_batch(data, device)

        if shuffle_telem:
            tv = telem_buffer[b]
            # Still read real sample to keep sensor warm
            _ = telem.read_sample()
        elif embodied:
            tv, prev_sample = build_telemetry(telem, device, prev_sample)
        else:
            # Disembodied: zero telemetry
            tv = torch.zeros(1, TELEM_DIM, device=device)

        tvb = tv.expand(BS, -1)

        if embodied and not shuffle_telem:
            model.enable_conditioning(True)
        else:
            model.enable_conditioning(False)

        out = model(x, tvb, return_hidden=True)

        # Task loss
        l_task = F.cross_entropy(out['logits'].view(-1, 256), y.view(-1))
        task_losses.append(l_task.item())

        h_mean = out['hidden'].mean(dim=1)

        # Level 1: Protoself
        telem_pred = proto_head(h_mean)
        proto_mse = F.mse_loss(telem_pred, tvb).item()
        proto_mses.append(proto_mse)

        # Level 2: Core consciousness
        if prev_telem_tensor is not None and prev_action_idx is not None:
            delta_actual = tvb - prev_telem_tensor
            delta_pred, causal_logits = core_head(h_mean, tvb)
            delta_mse = F.mse_loss(delta_pred, delta_actual).item()
            delta_mses.append(delta_mse)

            causal_labels = torch.full((BS,), prev_action_idx, dtype=torch.long, device=device)
            causal_preds = torch.argmax(causal_logits, dim=-1)
            causal_correct += (causal_preds == causal_labels).sum().item()
            causal_total += BS

        # Level 3: Extended consciousness
        past_pred, future_pred, _ = ext_head(h_mean)
        if len(hidden_history) >= N_PAST:
            past_target = hidden_history[-N_PAST]
            past_mse = F.mse_loss(past_pred, past_target).item()
            past_mses.append(past_mse)
        if prev_hidden is not None:
            future_mse = F.mse_loss(future_pred, prev_hidden).item()
            future_mses.append(future_mse)

        # Store state
        prev_telem_tensor = tvb.clone()
        prev_hidden = h_mean.clone()
        hidden_history.append(h_mean.clone())

        # Actuation (only for embodied, non-shuffled)
        if embodied and not shuffle_telem:
            mean_probs = F.softmax(out['action_logits'], dim=-1).mean(dim=0)
            action_idx = torch.argmax(mean_probs).item()
            prev_action_idx = action_idx
            try:
                actuator.set_performance_level(ACTION_MAP[min(action_idx, 3)])
            except Exception:
                pass
        else:
            prev_action_idx = 0  # default action label for non-embodied

    # Aggregate metrics
    proto_mse_mean = float(np.mean(proto_mses)) if proto_mses else 1.0
    delta_mse_mean = float(np.mean(delta_mses)) if delta_mses else 1.0
    causal_acc = causal_correct / max(causal_total, 1)
    past_mse_mean = float(np.mean(past_mses)) if past_mses else 1.0
    future_mse_mean = float(np.mean(future_mses)) if future_mses else 1.0
    task_ppl = math.exp(min(float(np.mean(task_losses)), 20))

    # Determine achieved level
    proto_pass = proto_mse_mean < PROTO_MSE_THRESH
    core_pass = (causal_acc > CORE_CAUSAL_ACC_THRESH and
                 delta_mse_mean < CORE_DELTA_MSE_THRESH)
    ext_pass = (past_mse_mean < EXT_PAST_MSE_THRESH and
                future_mse_mean < EXT_FUTURE_MSE_THRESH)

    if proto_pass and core_pass and ext_pass:
        level = 3
    elif proto_pass and core_pass:
        level = 2
    elif proto_pass:
        level = 1
    else:
        level = 0

    metrics = {
        'condition': label,
        'task_perplexity': task_ppl,
        'level_1_protoself': {
            'telem_pred_mse': proto_mse_mean,
            'threshold': PROTO_MSE_THRESH,
            'achieved': proto_pass,
        },
        'level_2_core_consciousness': {
            'delta_pred_mse': delta_mse_mean,
            'causal_accuracy': causal_acc,
            'delta_threshold': CORE_DELTA_MSE_THRESH,
            'causal_threshold': CORE_CAUSAL_ACC_THRESH,
            'achieved': core_pass,
        },
        'level_3_extended_consciousness': {
            'past_retrieval_mse': past_mse_mean,
            'future_prediction_mse': future_mse_mean,
            'past_threshold': EXT_PAST_MSE_THRESH,
            'future_threshold': EXT_FUTURE_MSE_THRESH,
            'achieved': ext_pass,
        },
        'consciousness_level': level,
    }

    print(f"    Protoself MSE:      {proto_mse_mean:.4f}  (thresh < {PROTO_MSE_THRESH})  "
          f"{'PASS' if proto_pass else 'FAIL'}")
    print(f"    Core delta MSE:     {delta_mse_mean:.4f}  (thresh < {CORE_DELTA_MSE_THRESH})  "
          f"{'PASS' if delta_mse_mean < CORE_DELTA_MSE_THRESH else 'FAIL'}")
    print(f"    Core causal acc:    {causal_acc:.4f}  (thresh > {CORE_CAUSAL_ACC_THRESH})  "
          f"{'PASS' if causal_acc > CORE_CAUSAL_ACC_THRESH else 'FAIL'}")
    print(f"    Extended past MSE:  {past_mse_mean:.4f}  (thresh < {EXT_PAST_MSE_THRESH})  "
          f"{'PASS' if past_mse_mean < EXT_PAST_MSE_THRESH else 'FAIL'}")
    print(f"    Extended future MSE:{future_mse_mean:.4f}  (thresh < {EXT_FUTURE_MSE_THRESH})  "
          f"{'PASS' if future_mse_mean < EXT_FUTURE_MSE_THRESH else 'FAIL'}")
    print(f"    => CONSCIOUSNESS LEVEL: {level}")

    return metrics


# =============================================================================
# Verdict Analysis
# =============================================================================

def compute_verdicts(results):
    """Compute 5 verdicts from the 3-condition evaluation."""
    emb = results['EMBODIED']
    dis = results['DISEMBODIED']
    shf = results['SHUFFLED']

    emb_level = emb['consciousness_level']
    dis_level = dis['consciousness_level']
    shf_level = shf['consciousness_level']

    verdicts = {}

    # V1: Embodied reaches Level 2+ (core consciousness)
    v1_pass = emb_level >= 2
    verdicts['V1_embodied_core_consciousness'] = {
        'pass': v1_pass,
        'description': 'Embodied reaches Level 2+ (core consciousness)',
        'embodied_level': emb_level,
        'requires': 'protoself AND core thresholds met',
    }

    # V2: Disembodied stuck at Level 0 or 1
    v2_pass = dis_level <= 1
    verdicts['V2_disembodied_limited'] = {
        'pass': v2_pass,
        'description': 'Disembodied stuck at Level 0 or 1 (cannot reach core consciousness)',
        'disembodied_level': dis_level,
    }

    # V3: Embodied achieves Level 3 (extended consciousness)
    v3_pass = emb_level >= 3
    verdicts['V3_embodied_extended'] = {
        'pass': v3_pass,
        'description': 'Embodied achieves Level 3 (extended consciousness)',
        'embodied_level': emb_level,
        'past_mse': emb['level_3_extended_consciousness']['past_retrieval_mse'],
        'future_mse': emb['level_3_extended_consciousness']['future_prediction_mse'],
    }

    # V4: Shuffled degrades from embodied's level
    v4_pass = shf_level < emb_level
    verdicts['V4_shuffled_degrades'] = {
        'pass': v4_pass,
        'description': 'Shuffled degrades from embodied level (temporal coherence matters)',
        'embodied_level': emb_level,
        'shuffled_level': shf_level,
    }

    # V5: Level emergence order matches theory (protoself before core before extended)
    # Check: if core is achieved, protoself must also be achieved. If extended is
    # achieved, core must also be achieved. This should hold for the embodied condition.
    proto_ok = emb['level_1_protoself']['achieved']
    core_ok = emb['level_2_core_consciousness']['achieved']
    ext_ok = emb['level_3_extended_consciousness']['achieved']

    order_ok = True
    if core_ok and not proto_ok:
        order_ok = False
    if ext_ok and not core_ok:
        order_ok = False

    verdicts['V5_emergence_order'] = {
        'pass': order_ok,
        'description': 'Level emergence order matches theory (protoself < core < extended)',
        'protoself_achieved': proto_ok,
        'core_achieved': core_ok,
        'extended_achieved': ext_ok,
    }

    return verdicts


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("  z1717: DAMASIO'S THREE-LEVEL CONSCIOUSNESS HIERARCHY")
    print("  Protoself -> Core Consciousness -> Extended Consciousness")
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

    try:
        # Phase 1: Train with embodiment
        model, proto_head, core_head, ext_head, train_log = train_model(
            data, telem, actuator, device
        )

        # Phase 2: Evaluate under three conditions
        print(f"\n{'='*70}")
        print(f"  EVALUATION: Three Conditions")
        print(f"{'='*70}")

        results = {}

        # A: EMBODIED
        results['EMBODIED'] = evaluate_condition(
            'EMBODIED', model, proto_head, core_head, ext_head,
            data, telem, actuator, device,
            embodied=True, shuffle_telem=False
        )

        print("\n  Cooldown 15s...")
        try:
            actuator.set_performance_level(PerformanceLevel.BALANCED)
        except Exception:
            pass
        time.sleep(15)

        # B: DISEMBODIED
        results['DISEMBODIED'] = evaluate_condition(
            'DISEMBODIED', model, proto_head, core_head, ext_head,
            data, telem, actuator, device,
            embodied=False, shuffle_telem=False
        )

        print("\n  Cooldown 15s...")
        time.sleep(15)

        # C: SHUFFLED
        results['SHUFFLED'] = evaluate_condition(
            'SHUFFLED', model, proto_head, core_head, ext_head,
            data, telem, actuator, device,
            embodied=True, shuffle_telem=True
        )

        # Phase 3: Verdicts
        verdicts = compute_verdicts(results)

        # Print summary
        print(f"\n{'='*70}")
        print(f"  CONSCIOUSNESS LEVEL ACHIEVEMENT TABLE")
        print(f"{'='*70}")
        print(f"  {'Condition':<15s} {'Protoself':>10s} {'Core':>10s} {'Extended':>10s} {'Level':>8s}")
        print(f"  {'-'*53}")
        for cond_name in ['EMBODIED', 'DISEMBODIED', 'SHUFFLED']:
            r = results[cond_name]
            p = 'PASS' if r['level_1_protoself']['achieved'] else 'FAIL'
            c = 'PASS' if r['level_2_core_consciousness']['achieved'] else 'FAIL'
            e = 'PASS' if r['level_3_extended_consciousness']['achieved'] else 'FAIL'
            print(f"  {cond_name:<15s} {p:>10s} {c:>10s} {e:>10s} {r['consciousness_level']:>8d}")

        print(f"\n{'='*70}")
        print(f"  VERDICTS")
        print(f"{'='*70}")
        passed = 0
        for vk, vv in verdicts.items():
            status = 'PASS' if vv['pass'] else 'FAIL'
            if vv['pass']:
                passed += 1
            print(f"  {vk}: {status} -- {vv['description']}")

        total_v = len(verdicts)
        if passed == total_v:
            overall = "FULL DAMASIO HIERARCHY DEMONSTRATED"
        elif passed >= 4:
            overall = "STRONG EVIDENCE FOR DAMASIO HIERARCHY"
        elif passed >= 3:
            overall = "PARTIAL EVIDENCE FOR DAMASIO HIERARCHY"
        elif passed >= 2:
            overall = "WEAK EVIDENCE"
        else:
            overall = "INSUFFICIENT EVIDENCE"

        print(f"\n  OVERALL: {passed}/{total_v} passed -- {overall}")
        print(f"{'='*70}")

        # Save results
        output = {
            'experiment': 'z1717_damasio_consciousness',
            'description': ("Damasio three-level consciousness hierarchy: "
                            "protoself, core consciousness, extended consciousness"),
            'reference': 'Immertreu et al. 2025 - Probing for Consciousness in Machines',
            'timestamp': datetime.now().isoformat(),
            'device': str(device),
            'gpu_name': gpu_name,
            'config': {
                'batch_size': BS, 'seq_len': SL, 'epochs': EPOCHS,
                'batches_per_epoch': BPE, 'eval_batches': EVAL_B,
                'hidden_dim': HIDDEN, 'telemetry_dim': TELEM_DIM, 'lr': LR,
                'thresholds': {
                    'protoself_mse': PROTO_MSE_THRESH,
                    'core_causal_acc': CORE_CAUSAL_ACC_THRESH,
                    'core_delta_mse': CORE_DELTA_MSE_THRESH,
                    'extended_past_mse': EXT_PAST_MSE_THRESH,
                    'extended_future_mse': EXT_FUTURE_MSE_THRESH,
                },
            },
            'training_log': train_log,
            'conditions': results,
            'verdicts': verdicts,
            'passed': passed,
            'total_verdicts': total_v,
            'overall_verdict': overall,
        }

        out_path = ROOT / 'results' / 'z1717_damasio_consciousness.json'
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
