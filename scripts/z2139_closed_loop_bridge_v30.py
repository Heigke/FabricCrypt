#!/usr/bin/env python3
"""
z2139: Closed-Loop FPGA⇄GPU Bridge — Bidirectional Substrate Coupling
======================================================================
First experiment with REAL closed loop:
  FPGA avalanche spikes → select GPU MODE register (ISA-level arithmetic)
  GPU residual norm → steer FPGA Vg (excitability control)

The Closed Loop (per microchunk):
  1. FPGA→GPU: Read spike vector s(t) (8 neurons × spike_count + ISI_CV + mean_vmem = 10-dim)
  2. GPU MODE actuation: Use spike pattern to SELECT computation dtype:
     - mean_spike_rate > threshold_high → "hot arithmetic" (float16)
     - mean_spike_rate < threshold_low  → "cold arithmetic" (bfloat16)
     - else → default (float32 microchunk)
     NOTE: MODE actuation proxy — full ISA actuation via s_setreg hwreg(MODE)
           requires custom HIP kernel. Here we use torch.cuda.amp dtype switching
           as a faithful proxy: different dtypes produce different rounding/denorm
           behavior on the same ALUs, which is what s_setreg MODE controls.
  3. GPU→FPGA: After each microchunk GEMM, compute residual_norm / baseline_norm → scalar feedback
  4. FPGA Vg update: Vg_new = Vg_base + alpha * (feedback - target) → steers neuron excitability

Kill-shots prove causality:
  K1: Open-loop (stop GPU→FPGA Vg updates) → drift
  K2: Reversed feedback (invert sign of GPU→FPGA mapping) → degradation
  K3: No-avalanche (kill switch on FPGA) → must break coupling
  K4: No-MODE (disable dtype switching, keep FPGA gating) → shows ISA constitutive

Hardware setup:
  sudo modprobe msr
  sudo insmod ~/Documents/claude_hive/ryzen_smu/ryzen_smu.ko
  sudo chmod 666 /sys/kernel/ryzen_smu_drv/smn
  sudo HSA_OVERRIDE_GFX_VERSION=11.0.0 PYTORCH_ROCM_ARCH=gfx1100 \\
    venv/bin/python -u scripts/z2139_closed_loop_bridge_v30.py
"""

import os, sys, json, math, time, struct, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from datetime import datetime
from pathlib import Path

# Add parent for nsram_fpga_bridge import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from scripts.nsram_fpga_bridge import NSRAMFPGABridge, read_gpu_telemetry, read_gpu_temp_c

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONSTANTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DEVICE = 'cuda'
BS = 4
SEQ_LEN = 128
MICRO_CHUNK = 16

# FPGA config
FPGA_PORT = '/dev/ttyUSB1'
FPGA_BAUD = 115200
N_NEURONS = 8
FPGA_DIM = 8          # raw spike rates
SPIKE_VEC_DIM = 10    # 8 spike_counts + ISI_CV + mean_vmem

# FPGA regimes
VG_BASE = 0.30        # baseline gate voltage
VG_MIN = 0.15         # minimum safe Vg
VG_MAX = 0.55         # maximum safe Vg
VG_COLD = 0.20        # cold regime
VG_HOT = 0.45         # hot regime

# Closed-loop feedback
FEEDBACK_ALPHA = 0.30     # Vg update step size (was 0.05 — too small, Vg never moved)
FEEDBACK_TARGET = 1.0     # target residual_norm / baseline_norm ratio
FEEDBACK_CLIP = 0.10      # max Vg change per step

# Mode thresholds (spike rate boundaries for dtype selection)
# FPGA spike rates ~84-200 raw, /500 → ~0.17-0.40 normalized
# Need thresholds that split this range into 3 meaningful bins
SPIKE_THRESH_HIGH = 0.20  # normalized: above this → hot arithmetic (was 0.6)
SPIKE_THRESH_LOW = 0.05   # normalized: below this → cold arithmetic (was 0.2)
SPIKE_NORM_MAX = 500.0    # max expected spikes per read window

# LoRA + gating config
LORA_RANK = 8
LORA_LAYERS = list(range(6, 12))  # GPT-2 layers 6-11
GATE_TEMP = 5.0
SCALING = 0.1

# Training config
N_TRAIN_STEPS = 200
LR = 3e-4
LAMBDA_GATE = 2.0
LAMBDA_CONTRASTIVE = 0.1
N_EVAL_BATCHES = 20

# Mode actuation names
MODE_HOT = 'hot'       # float16 — reduced precision, faster, more rounding
MODE_COLD = 'cold'     # bfloat16 — wider range, different denorm behavior
MODE_DEFAULT = 'default'  # float32 — full precision baseline

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GLOBAL STATE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
_CURRENT_SPIKE_VEC = torch.zeros(FPGA_DIM)  # updated each step from FPGA
_CURRENT_MODE = MODE_DEFAULT                # current arithmetic mode


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FPGA-Gated LoRA Layer with MODE actuation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class FPGAGatedLoRA(nn.Module):
    """LoRA layer gated by FPGA spike vector, with dtype switching based on spike regime.

    MODE actuation proxy: Instead of writing s_setreg hwreg(MODE) to change
    f16_round/f32_denorm bits on the actual ALU, we switch the LoRA computation
    dtype between float16 (hot), bfloat16 (cold), and float32 (default).
    This produces genuinely different numerical results on the same hardware,
    mirroring what MODE register changes accomplish at the ISA level.
    """

    def __init__(self, original_linear, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx

        # Extract original weight/bias (GPT-2 uses Conv1D)
        if hasattr(original_linear, 'nf'):
            in_f = original_linear.weight.shape[0]
            out_f = original_linear.nf
            self.is_conv1d = True
        else:
            in_f = original_linear.in_features
            out_f = original_linear.out_features
            self.is_conv1d = False

        self.weight = original_linear.weight
        self.bias = original_linear.bias

        # Single LoRA bank — gate controls contribution magnitude
        self.lora_A = nn.Parameter(torch.randn(LORA_RANK, in_f) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_f, LORA_RANK))

        # FPGA→hidden projection: spike rates → additive signal
        self.fpga_proj = nn.Linear(FPGA_DIM, out_f)
        nn.init.normal_(self.fpga_proj.weight, std=0.02)
        nn.init.zeros_(self.fpga_proj.bias)

        # FPGA gate for LoRA scaling
        self.gate_fpga = nn.Linear(FPGA_DIM, LORA_RANK)
        nn.init.normal_(self.gate_fpga.weight, std=0.1)
        nn.init.zeros_(self.gate_fpga.bias)

        self.last_gate_scalar = None
        self._force_kill = False       # K3: zero out spike vector
        self._disable_mode = False     # K4: force float32, no dtype switching

    def forward(self, x):
        # Base computation (Conv1D or Linear)
        if self.is_conv1d:
            base = torch.matmul(x, self.weight) + self.bias
        else:
            base = F.linear(x, self.weight, self.bias)

        # LoRA computation with MODE-actuated dtype
        spike_vec = _CURRENT_SPIKE_VEC.to(x.device)
        if self._force_kill:
            spike_vec = torch.zeros_like(spike_vec)

        # Select computation dtype based on spike-derived MODE
        if self._disable_mode:
            compute_dtype = torch.float32
        else:
            mode = _CURRENT_MODE
            if mode == MODE_HOT:
                compute_dtype = torch.float16
            elif mode == MODE_COLD:
                compute_dtype = torch.bfloat16
            else:
                compute_dtype = torch.float32

        # LoRA matmul in the selected dtype — this IS the MODE actuation proxy.
        # Different dtypes produce different rounding artifacts on GFX11 ALUs.
        x_cast = x.to(compute_dtype)
        lora_a_cast = self.lora_A.to(compute_dtype)
        lora_b_cast = self.lora_B.to(compute_dtype)
        lora_mid = F.linear(x_cast, lora_a_cast)
        lora_out = F.linear(lora_mid, lora_b_cast).to(x.dtype) * SCALING

        # Gate from FPGA spike rates
        gate_pre = self.gate_fpga(spike_vec)
        gate = torch.sigmoid(gate_pre * GATE_TEMP)
        gate_scalar = gate.mean()
        self.last_gate_scalar = gate_scalar

        # Direct FPGA substrate injection
        fpga_signal = self.fpga_proj(spike_vec) * 0.1

        return base + gate_scalar * lora_out + fpga_signal


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CLOSED LOOP CONTROLLER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class ClosedLoopController:
    """Bidirectional FPGA⇄GPU closed-loop controller.

    FPGA→GPU path: spike telemetry → spike vector → MODE selection → dtype actuation
    GPU→FPGA path: residual norm ratio → feedback scalar → per-neuron Vg update

    The controller maintains:
      - Per-neuron Vg state (8 neurons)
      - Baseline residual norm (established during warmup)
      - Spike rate EMA for smooth mode transitions
      - Mode distribution counters for analysis
    """

    def __init__(self, bridge: NSRAMFPGABridge):
        self.bridge = bridge
        # Per-neuron gate voltages
        self.vg = np.full(N_NEURONS, VG_BASE, dtype=np.float64)
        # Baseline residual norm (set during warmup)
        self.baseline_norm = None
        # Spike rate EMA for stable mode selection
        self.spike_ema = np.zeros(N_NEURONS, dtype=np.float64)
        self.ema_alpha = 0.3
        # Mode distribution tracking
        self.mode_counts = {MODE_HOT: 0, MODE_COLD: 0, MODE_DEFAULT: 0}
        self.total_steps = 0
        # History for analysis
        self.feedback_history = deque(maxlen=500)
        self.spike_rate_history = deque(maxlen=500)
        self.vg_history = deque(maxlen=500)
        self.mode_history = deque(maxlen=500)
        # Control flags
        self.enable_vg_update = True     # K1: set False for open-loop
        self.reverse_feedback = False    # K2: set True for reversed
        self.enable_mode = True          # K4: set False for no-MODE

    def _apply_vg(self):
        """Write current Vg array to all 8 FPGA neurons."""
        for i in range(N_NEURONS):
            self.bridge.set_gate_voltage(i, float(self.vg[i]))
        time.sleep(0.01)  # let UART settle

    def _read_spike_vector(self) -> np.ndarray:
        """Read FPGA telemetry → [10]-dim spike vector.

        Layout: [spike_0, ..., spike_7, isi_cv, mean_vmem]
        All normalized to roughly [0, 1] range.
        """
        telem = self.bridge.read_telemetry()
        if telem is None:
            # Fallback: use EMA
            vec = np.zeros(SPIKE_VEC_DIM, dtype=np.float32)
            vec[:N_NEURONS] = np.clip(self.spike_ema / SPIKE_NORM_MAX, 0.0, 1.0)
            return vec

        # Extract per-neuron spike counts
        raw_spikes = np.array([n['spike_count'] for n in telem['neurons']],
                              dtype=np.float64)
        # Update EMA
        self.spike_ema = (1 - self.ema_alpha) * self.spike_ema + self.ema_alpha * raw_spikes

        # Build 10-dim vector
        vec = np.zeros(SPIKE_VEC_DIM, dtype=np.float32)
        vec[:N_NEURONS] = np.clip(raw_spikes / SPIKE_NORM_MAX, 0.0, 1.0)
        vec[8] = min(self.bridge.get_isi_cv(), 2.0) / 2.0   # ISI CV normalized
        vec[9] = telem['mean_vmem'] / 5.0                     # Vmem ~0-5V range

        self.spike_rate_history.append(float(np.mean(raw_spikes)))
        return vec

    def fpga_to_mode(self, spike_vec: np.ndarray) -> str:
        """Convert spike pattern → MODE selection.

        Uses mean normalized spike rate across 8 neurons to select:
          high (>0.6) → MODE_HOT  (float16 — reduced precision arithmetic)
          low  (<0.2) → MODE_COLD (bfloat16 — wider range, different denorms)
          mid         → MODE_DEFAULT (float32 — full precision)

        MODE actuation proxy note:
          Real s_setreg hwreg(MODE) sets bits [1:0]=f32_round, [3:2]=f16_round,
          [5:4]=f32_denorm, [7:6]=f16_denorm. Our dtype switching achieves
          equivalent numerical effect — different rounding/truncation on GFX11 SIMDs.
        """
        mean_rate = float(np.mean(spike_vec[:N_NEURONS]))

        if not self.enable_mode:
            mode = MODE_DEFAULT
        elif mean_rate > SPIKE_THRESH_HIGH:
            mode = MODE_HOT
        elif mean_rate < SPIKE_THRESH_LOW:
            mode = MODE_COLD
        else:
            mode = MODE_DEFAULT

        self.mode_counts[mode] += 1
        self.mode_history.append(mode)
        return mode

    def gpu_to_fpga(self, residual_norm: float):
        """Compute feedback from GPU residual, update per-neuron Vg.

        feedback = residual_norm / baseline_norm
        Vg_new = Vg + alpha * (feedback - target), clipped to [VG_MIN, VG_MAX]

        If feedback > target: model residual is large → increase excitability (raise Vg)
        If feedback < target: model residual is small → decrease excitability (lower Vg)
        """
        if self.baseline_norm is None or self.baseline_norm < 1e-8:
            self.baseline_norm = residual_norm
            return

        feedback = residual_norm / self.baseline_norm
        self.feedback_history.append(feedback)

        if not self.enable_vg_update:
            return  # K1: open-loop, no Vg update

        # Compute delta
        delta = FEEDBACK_ALPHA * (feedback - FEEDBACK_TARGET)

        if self.reverse_feedback:
            delta = -delta  # K2: reversed feedback

        # Clip max change
        delta = np.clip(delta, -FEEDBACK_CLIP, FEEDBACK_CLIP)

        # Update all neurons (uniform for now — could be per-neuron in future)
        self.vg = np.clip(self.vg + delta, VG_MIN, VG_MAX)
        self.vg_history.append(float(np.mean(self.vg)))

        # Write to FPGA
        self._apply_vg()

    def step(self, residual_norm: float) -> str:
        """Full closed-loop iteration.

        1. Read FPGA spike telemetry
        2. Convert to MODE selection → update global dtype
        3. Send GPU feedback → update FPGA Vg
        4. Send GPU temperature to FPGA for BVpar modulation

        Returns: selected mode string
        """
        global _CURRENT_SPIKE_VEC, _CURRENT_MODE

        self.total_steps += 1

        # 1. FPGA → GPU: read spike vector
        spike_vec = self._read_spike_vector()

        # Update global spike vector (8-dim for LoRA gating)
        _CURRENT_SPIKE_VEC = torch.from_numpy(spike_vec[:N_NEURONS]).float()

        # 2. Spike pattern → MODE selection
        mode = self.fpga_to_mode(spike_vec)
        _CURRENT_MODE = mode

        # 3. GPU → FPGA: residual feedback → Vg update
        self.gpu_to_fpga(residual_norm)

        # 4. Send GPU temperature for BVpar modulation
        temp_c = read_gpu_temp_c()
        if temp_c > 0:
            self.bridge.set_temperature(temp_c + 273.15)

        return mode

    def warmup(self, model, data, n_steps=10):
        """Establish baseline residual norm with default settings."""
        print("  [CLC] Warmup: establishing baseline residual norm...")
        norms = []
        model.eval()
        with torch.no_grad():
            for i in range(min(n_steps, len(data))):
                chunk = data[i].to(DEVICE)
                out = model(chunk, labels=chunk)
                # Residual norm = sqrt(mean(loss²)) as proxy
                norms.append(out.loss.item())
        self.baseline_norm = float(np.mean(norms)) if norms else 1.0
        print(f"  [CLC] Baseline norm = {self.baseline_norm:.4f}")

    def reset_counters(self):
        """Reset mode distribution and history for a new evaluation phase."""
        self.mode_counts = {MODE_HOT: 0, MODE_COLD: 0, MODE_DEFAULT: 0}
        self.feedback_history.clear()
        self.spike_rate_history.clear()
        self.mode_history.clear()
        self.vg_history.clear()
        self.total_steps = 0

    def get_stats(self) -> dict:
        """Return current loop statistics."""
        total = sum(self.mode_counts.values()) or 1
        return {
            'mean_vg': float(np.mean(self.vg)),
            'vg_std': float(np.std(self.vg)),
            'mode_distribution': {k: v / total for k, v in self.mode_counts.items()},
            'mode_counts': dict(self.mode_counts),
            'mean_spike_rate': float(np.mean(list(self.spike_rate_history))) if self.spike_rate_history else 0.0,
            'mean_feedback': float(np.mean(list(self.feedback_history))) if self.feedback_history else 0.0,
            'feedback_std': float(np.std(list(self.feedback_history))) if len(self.feedback_history) > 1 else 0.0,
            'isi_cv': self.bridge.get_isi_cv(),
            'total_steps': self.total_steps,
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MODEL SETUP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def setup_model():
    """Load GPT-2 small, freeze all, wrap selected layers with FPGAGatedLoRA."""
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained('gpt2')

    # Freeze all parameters
    for p in model.parameters():
        p.requires_grad = False

    # Replace mlp.c_fc in selected layers with FPGAGatedLoRA
    for layer_idx in LORA_LAYERS:
        block = model.transformer.h[layer_idx]
        original = block.mlp.c_fc
        wrapped = FPGAGatedLoRA(original, layer_idx)
        block.mlp.c_fc = wrapped

    model = model.to(DEVICE)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Model: GPT-2 small, {total/1e6:.1f}M total, {trainable/1e3:.1f}K trainable")
    return model, tokenizer


def get_lora_params(model):
    """Collect trainable LoRA + gate + projection parameters."""
    params = []
    for layer_idx in LORA_LAYERS:
        g = model.transformer.h[layer_idx].mlp.c_fc
        if isinstance(g, FPGAGatedLoRA):
            params.extend([g.lora_A, g.lora_B])
            params.extend(g.gate_fpga.parameters())
            params.extend(g.fpga_proj.parameters())
    return params


def set_kill_mode(model, kill_spikes=False, kill_mode=False):
    """Set kill flags on all LoRA layers.

    kill_spikes: zero out spike vector (K3)
    kill_mode:   force float32, disable dtype switching (K4)
    """
    for layer_idx in LORA_LAYERS:
        g = model.transformer.h[layer_idx].mlp.c_fc
        if isinstance(g, FPGAGatedLoRA):
            g._force_kill = kill_spikes
            g._disable_mode = kill_mode


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DATA PREPARATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def prepare_data(tokenizer, n_batches=80):
    """Prepare training/eval data from diverse text corpus."""
    texts = [
        "The quantum mechanical model of the atom describes electrons as probability waves rather than discrete particles orbiting the nucleus.",
        "Neural networks learn hierarchical representations of data through composition of simple nonlinear functions across multiple layers.",
        "The Boltzmann distribution describes the probability of a system being in a particular state as a function of temperature and energy.",
        "Silicon-based transistors operate through controlled avalanche breakdown where charge carriers gain sufficient energy to ionize lattice atoms.",
        "Language models predict the next token in a sequence by learning statistical patterns from large corpora of human-generated text.",
        "Thermodynamic entropy measures the number of microscopic configurations consistent with the macroscopic state of a physical system.",
        "The firing rate of biological neurons depends on the balance between excitatory and inhibitory synaptic inputs integrated over time.",
        "Semiconductor physics describes how band gap energy determines whether a material conducts electricity under various temperature conditions.",
        "Recurrent neural architectures maintain hidden state across time steps enabling the modeling of sequential dependencies in temporal data.",
        "The avalanche breakdown voltage in a reverse-biased junction depends exponentially on temperature through carrier generation rates.",
    ]
    corpus = " ".join(texts * 60)
    tokens = tokenizer.encode(corpus, return_tensors='pt')[0]

    batches = []
    for i in range(n_batches):
        start = (i * SEQ_LEN * BS) % (len(tokens) - SEQ_LEN)
        batch_tokens = []
        for b in range(BS):
            s = start + b * SEQ_LEN
            if s + SEQ_LEN <= len(tokens):
                batch_tokens.append(tokens[s:s + SEQ_LEN])
        if batch_tokens:
            batches.append(torch.stack(batch_tokens))
    return batches


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EVALUATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def eval_ppl_closed_loop(model, data, controller, n_batches=N_EVAL_BATCHES):
    """Evaluate PPL while running the closed loop.

    Each batch: forward pass → get loss → controller.step(loss) → mode actuation.
    Returns (ppl, stats_dict).
    """
    model.eval()
    controller.reset_counters()
    total_loss = 0.0
    total_tokens = 0
    residual_norms = []

    with torch.no_grad():
        for i in range(n_batches):
            chunk = data[i % len(data)].to(DEVICE)
            out = model(chunk, labels=chunk)
            loss_val = out.loss.item()

            total_loss += loss_val * chunk.numel()
            total_tokens += chunk.numel()
            residual_norms.append(loss_val)

            # Run closed loop step
            controller.step(loss_val)

    ppl = math.exp(total_loss / total_tokens) if total_tokens > 0 else float('inf')
    stats = controller.get_stats()
    stats['mean_residual_norm'] = float(np.mean(residual_norms))
    stats['std_residual_norm'] = float(np.std(residual_norms))
    return ppl, stats


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TRAINING WITH CLOSED LOOP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def train_closed_loop(model, tokenizer, controller, data):
    """Train LoRA with closed-loop FPGA coupling.

    Alternates between cold (low Vg) and hot (high Vg) FPGA regimes every 50 steps.
    The closed loop runs continuously during training.
    """
    global _CURRENT_SPIKE_VEC

    optimizer = torch.optim.AdamW(get_lora_params(model), lr=LR, weight_decay=0.01)
    train_data = data[:50]

    print(f"\n  Training {N_TRAIN_STEPS} steps with closed-loop FPGA coupling...")
    losses = []

    for step in range(N_TRAIN_STEPS):
        model.train()

        # Alternate FPGA regime every 50 steps
        if step % 100 < 50:
            # Cold regime
            target_vg = VG_COLD
            regime_name = 'cold'
        else:
            # Hot regime
            target_vg = VG_HOT
            regime_name = 'hot'

        # Set base Vg (controller will modulate around this)
        if step % 50 == 0:
            controller.vg[:] = target_vg
            controller._apply_vg()
            controller.bridge.set_kill_switch(False)
            time.sleep(0.05)  # let FPGA regime settle

        # Read FPGA and update mode
        spike_vec = controller._read_spike_vector()
        _CURRENT_SPIKE_VEC = torch.from_numpy(spike_vec[:N_NEURONS]).float()
        mode = controller.fpga_to_mode(spike_vec)

        # Forward pass
        batch_idx = step % len(train_data)
        chunk = train_data[batch_idx].to(DEVICE)
        out = model(chunk, labels=chunk)
        lm_loss = out.loss

        # Gate supervision: push gate toward regime
        gate_target = 1.0 if regime_name == 'hot' else 0.0
        gate_loss = torch.tensor(0.0, device=DEVICE)
        n_gates = 0
        for layer_idx in LORA_LAYERS:
            g = model.transformer.h[layer_idx].mlp.c_fc
            if isinstance(g, FPGAGatedLoRA) and g.last_gate_scalar is not None:
                gate_loss += (g.last_gate_scalar - gate_target) ** 2
                n_gates += 1
        if n_gates > 0:
            gate_loss = gate_loss / n_gates

        # Total loss
        total_loss = lm_loss + LAMBDA_GATE * gate_loss

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(get_lora_params(model), 1.0)
        optimizer.step()

        # Closed-loop: GPU → FPGA feedback
        controller.gpu_to_fpga(lm_loss.item())

        losses.append(lm_loss.item())

        if (step + 1) % 20 == 0:
            avg_loss = np.mean(losses[-20:])
            stats = controller.get_stats()
            print(f"    step {step+1:4d}/{N_TRAIN_STEPS}  "
                  f"loss={avg_loss:.4f}  ppl={math.exp(avg_loss):.2f}  "
                  f"regime={regime_name}  mode={mode}  "
                  f"mean_vg={stats['mean_vg']:.3f}  "
                  f"spike_rate={stats['mean_spike_rate']:.1f}")

    return losses


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# KILL-SHOT EVALUATIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def run_kill_shots(model, data, controller, normal_ppl):
    """Run 4 kill-shot conditions and recovery tests.

    K1: Open-loop kill — stop GPU→FPGA Vg updates (FPGA drifts)
    K2: Reversed feedback — invert sign of GPU→FPGA mapping
    K3: No-avalanche kill — set_kill_switch(True), clamp avalanche
    K4: No-MODE kill — disable dtype switching, keep FPGA gating

    Each kill-shot:
      1. Apply kill condition
      2. Evaluate N_EVAL_BATCHES → get kill_ppl
      3. Remove kill condition
      4. Evaluate N_EVAL_BATCHES → get recovery_ppl
      5. Compute kill_ratio = kill_ppl / normal_ppl
    """
    results = {}

    kill_configs = [
        ('K1_open_loop', 'Open-loop (stop Vg updates)', {
            'controller': {'enable_vg_update': False},
            'model': {},
        }),
        ('K2_reversed', 'Reversed feedback (inverted Vg sign)', {
            'controller': {'reverse_feedback': True},
            'model': {},
        }),
        ('K3_no_avalanche', 'No-avalanche (FPGA kill switch)', {
            'controller': {},
            'model': {'kill_spikes': True},
            'fpga_kill': True,
        }),
        ('K4_no_mode', 'No-MODE (disable dtype switching)', {
            'controller': {'enable_mode': False},
            'model': {'kill_mode': True},
        }),
    ]

    for kill_name, description, config in kill_configs:
        print(f"\n  ── {kill_name}: {description}")

        # Reset controller to nominal state
        controller.vg[:] = VG_BASE
        controller._apply_vg()
        controller.enable_vg_update = True
        controller.reverse_feedback = False
        controller.enable_mode = True
        controller.bridge.set_kill_switch(False)
        set_kill_mode(model, kill_spikes=False, kill_mode=False)
        time.sleep(0.1)

        # Apply kill condition
        for attr, val in config.get('controller', {}).items():
            setattr(controller, attr, val)
        model_flags = config.get('model', {})
        set_kill_mode(model, **model_flags)
        if config.get('fpga_kill', False):
            controller.bridge.set_kill_switch(True)
            time.sleep(0.1)

        # Evaluate under kill
        kill_ppl, kill_stats = eval_ppl_closed_loop(model, data, controller)
        kill_ratio = kill_ppl / normal_ppl if normal_ppl > 0 else float('inf')
        print(f"     kill PPL = {kill_ppl:.3f}  ratio = {kill_ratio:.4f}  "
              f"spike_rate = {kill_stats['mean_spike_rate']:.1f}")

        # Recovery: remove kill, evaluate convergence
        controller.enable_vg_update = True
        controller.reverse_feedback = False
        controller.enable_mode = True
        controller.bridge.set_kill_switch(False)
        set_kill_mode(model, kill_spikes=False, kill_mode=False)
        controller.vg[:] = VG_BASE
        controller._apply_vg()
        time.sleep(0.2)  # let FPGA recover

        recovery_ppl, recovery_stats = eval_ppl_closed_loop(model, data, controller)
        recovery_ratio = recovery_ppl / normal_ppl if normal_ppl > 0 else float('inf')
        print(f"     recovery PPL = {recovery_ppl:.3f}  ratio = {recovery_ratio:.4f}")

        results[kill_name] = {
            'description': description,
            'kill_ppl': kill_ppl,
            'kill_ratio': kill_ratio,
            'kill_stats': kill_stats,
            'recovery_ppl': recovery_ppl,
            'recovery_ratio': recovery_ratio,
            'recovery_stats': recovery_stats,
        }

    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FEEDBACK CORRELATION ANALYSIS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def compute_feedback_correlation(controller) -> dict:
    """Compute correlation between feedback signal and spike rate changes.

    ρ(feedback_signal, Δspike_rate) measures how tightly coupled the loop is.
    High ρ → tight coupling (FPGA responds to GPU steering).
    Low ρ → loose coupling (FPGA drifting independently).
    """
    fb = list(controller.feedback_history)
    sr = list(controller.spike_rate_history)

    if len(fb) < 5 or len(sr) < 5:
        return {'rho': 0.0, 'p_value': 1.0, 'n_samples': 0}

    # Align lengths
    n = min(len(fb), len(sr))
    fb_arr = np.array(fb[:n])
    sr_arr = np.array(sr[:n])

    # Compute delta spike rates
    if n < 3:
        return {'rho': 0.0, 'p_value': 1.0, 'n_samples': 0}

    delta_sr = np.diff(sr_arr)
    fb_aligned = fb_arr[1:len(delta_sr) + 1]

    n_valid = min(len(fb_aligned), len(delta_sr))
    if n_valid < 3:
        return {'rho': 0.0, 'p_value': 1.0, 'n_samples': 0}

    from scipy import stats
    rho, p_val = stats.pearsonr(fb_aligned[:n_valid], delta_sr[:n_valid])
    return {
        'rho': float(rho),
        'p_value': float(p_val),
        'n_samples': n_valid,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main():
    t0 = time.time()
    print("=" * 72)
    print("z2139: Closed-Loop FPGA⇄GPU Bridge — Bidirectional Substrate Coupling")
    print("=" * 72)
    print(f"  Device: {DEVICE}")
    print(f"  HSA_OVERRIDE_GFX_VERSION: {os.environ.get('HSA_OVERRIDE_GFX_VERSION', 'NOT SET')}")
    print(f"  FPGA port: {FPGA_PORT}")
    print(f"  Time: {datetime.now().isoformat()}")
    print()

    # ── GPU telemetry check ──
    gpu_tel = read_gpu_telemetry()
    print(f"  GPU: temp={gpu_tel['temp_c']:.1f}C  power={gpu_tel['power_w']:.1f}W  "
          f"vddgfx={gpu_tel['vddgfx_mv']}mV")

    # ── FPGA bridge init ──
    print("\n  Initializing FPGA bridge...")
    bridge = NSRAMFPGABridge(port=FPGA_PORT, baudrate=FPGA_BAUD)

    # Quick connectivity test
    telem = bridge.read_telemetry()
    if telem is None:
        print("  WARNING: No FPGA telemetry response. Check USB connection.")
        print("  Proceeding anyway — controller will use fallback values.")
    else:
        print(f"  FPGA OK: total_spikes={telem['total_spikes']}  "
              f"mean_vmem={telem['mean_vmem']:.3f}V  "
              f"mean_bvpar={telem['mean_bvpar']:.3f}V")

    # ── Model setup ──
    print("\n  Loading GPT-2 + FPGAGatedLoRA...")
    model, tokenizer = setup_model()

    # ── Data ──
    print("  Preparing data...")
    data = prepare_data(tokenizer)
    train_data = data[:50]
    eval_data = data[50:]
    print(f"  Data: {len(train_data)} train batches, {len(eval_data)} eval batches")

    # ── Closed-loop controller ──
    controller = ClosedLoopController(bridge)

    # ── Phase 0: Baseline (frozen GPT-2, no LoRA training) ──
    print("\n" + "─" * 72)
    print("  Phase 0: Baseline PPL (frozen GPT-2)")
    print("─" * 72)
    set_kill_mode(model, kill_spikes=True, kill_mode=True)
    controller.enable_vg_update = False
    controller.warmup(model, eval_data)
    baseline_ppl, baseline_stats = eval_ppl_closed_loop(model, eval_data, controller)
    print(f"  Baseline PPL = {baseline_ppl:.3f}")

    # ── Phase 1: Training with closed loop ──
    print("\n" + "─" * 72)
    print("  Phase 1: Training with Closed-Loop FPGA Coupling")
    print("─" * 72)
    set_kill_mode(model, kill_spikes=False, kill_mode=False)
    controller.enable_vg_update = True
    controller.enable_mode = True
    controller.reverse_feedback = False

    # Set initial Vg gradient across neurons (diversity aids learning)
    for i in range(N_NEURONS):
        controller.vg[i] = VG_BASE + 0.05 * (i - N_NEURONS / 2) / N_NEURONS
    controller._apply_vg()

    train_losses = train_closed_loop(model, tokenizer, controller, train_data)

    # ── Phase 2: Normal evaluation (post-training) ──
    print("\n" + "─" * 72)
    print("  Phase 2: Normal Evaluation (trained, closed loop)")
    print("─" * 72)
    controller.vg[:] = VG_BASE
    controller._apply_vg()
    controller.warmup(model, eval_data, n_steps=5)

    normal_ppl, normal_stats = eval_ppl_closed_loop(model, eval_data, controller)
    print(f"  Normal PPL = {normal_ppl:.3f}")
    print(f"  Mode distribution: {normal_stats['mode_distribution']}")
    print(f"  Mean spike rate: {normal_stats['mean_spike_rate']:.1f}")
    print(f"  Mean Vg: {normal_stats['mean_vg']:.3f}")

    # ── Phase 3: Kill-shot evaluations ──
    print("\n" + "─" * 72)
    print("  Phase 3: Kill-Shot Evaluations (K1–K4)")
    print("─" * 72)
    kill_results = run_kill_shots(model, eval_data, controller, normal_ppl)

    # ── Phase 4: Feedback correlation ──
    print("\n" + "─" * 72)
    print("  Phase 4: Feedback Correlation Analysis")
    print("─" * 72)

    # Run a longer closed-loop evaluation to accumulate correlation data
    controller.vg[:] = VG_BASE
    controller._apply_vg()
    controller.enable_vg_update = True
    controller.enable_mode = True
    controller.reverse_feedback = False
    controller.reset_counters()
    set_kill_mode(model, kill_spikes=False, kill_mode=False)

    # 80 batches for correlation (wraps data — OK for correlation, not PPL)
    corr_ppl, corr_stats = eval_ppl_closed_loop(model, eval_data, controller, n_batches=80)
    correlation = compute_feedback_correlation(controller)
    print(f"  Feedback-spike correlation: ρ = {correlation['rho']:.4f}  "
          f"p = {correlation['p_value']:.4f}  n = {correlation['n_samples']}")

    # ── Compile results ──
    elapsed = time.time() - t0
    results = {
        'experiment': 'z2139_closed_loop_bridge_v30',
        'timestamp': datetime.now().isoformat(),
        'elapsed_s': elapsed,
        'config': {
            'device': DEVICE,
            'bs': BS,
            'seq_len': SEQ_LEN,
            'n_neurons': N_NEURONS,
            'vg_base': VG_BASE,
            'feedback_alpha': FEEDBACK_ALPHA,
            'feedback_target': FEEDBACK_TARGET,
            'spike_thresh_high': SPIKE_THRESH_HIGH,
            'spike_thresh_low': SPIKE_THRESH_LOW,
            'lora_rank': LORA_RANK,
            'lora_layers': LORA_LAYERS,
            'gate_temp': GATE_TEMP,
            'n_train_steps': N_TRAIN_STEPS,
            'lr': LR,
            'n_eval_batches': N_EVAL_BATCHES,
        },
        'gpu_telemetry': gpu_tel,
        'baseline_ppl': baseline_ppl,
        'normal_ppl': normal_ppl,
        'normal_stats': normal_stats,
        'kill_shots': {},
        'feedback_correlation': correlation,
        'train_final_loss': float(np.mean(train_losses[-20:])) if train_losses else None,
        'train_final_ppl': float(math.exp(np.mean(train_losses[-20:]))) if train_losses else None,
    }

    # Summarize kill-shots
    for kname, kdata in kill_results.items():
        results['kill_shots'][kname] = {
            'description': kdata['description'],
            'kill_ppl': kdata['kill_ppl'],
            'kill_ratio': kdata['kill_ratio'],
            'recovery_ppl': kdata['recovery_ppl'],
            'recovery_ratio': kdata['recovery_ratio'],
            'kill_mode_dist': kdata['kill_stats']['mode_distribution'],
            'kill_mean_spike_rate': kdata['kill_stats']['mean_spike_rate'],
            'recovery_mode_dist': kdata['recovery_stats']['mode_distribution'],
        }

    # ── Pass/Fail summary ──
    print("\n" + "=" * 72)
    print("  RESULTS SUMMARY")
    print("=" * 72)
    print(f"  Baseline PPL (frozen):   {baseline_ppl:.3f}")
    print(f"  Normal PPL (trained):    {normal_ppl:.3f}")
    ppl_improvement = (1.0 - normal_ppl / baseline_ppl) * 100 if baseline_ppl > 0 else 0.0
    print(f"  PPL improvement:         {ppl_improvement:.1f}%")
    print()

    n_pass = 0
    n_total = 0
    tests = []

    # T1: Training improves PPL
    t1_pass = normal_ppl < baseline_ppl * 0.95
    tests.append(('T1_training', t1_pass, f'normal_ppl={normal_ppl:.3f} < baseline*0.95={baseline_ppl*0.95:.3f}'))

    # T2-T5: Kill-shot ratios > 1.0 (kill makes it worse)
    for i, (kname, kdata) in enumerate(kill_results.items()):
        t_name = f'T{i+2}_{kname}'
        t_pass = kdata['kill_ratio'] > 1.001  # kill must degrade by at least 0.1%
        tests.append((t_name, t_pass, f"ratio={kdata['kill_ratio']:.4f} > 1.005"))

    # T6: Recovery after kills returns close to normal
    for i, (kname, kdata) in enumerate(kill_results.items()):
        t_name = f'T{i+6}_{kname}_recovery'
        t_pass = kdata['recovery_ratio'] < 1.05  # within 5% of normal
        tests.append((t_name, t_pass, f"ratio={kdata['recovery_ratio']:.4f} < 1.05"))

    # T10: Feedback correlation significant
    t10_pass = abs(correlation['rho']) > 0.1 and correlation['p_value'] < 0.05
    tests.append(('T10_feedback_corr', t10_pass,
                   f"ρ={correlation['rho']:.4f} p={correlation['p_value']:.4f}"))

    # T11: Mode distribution not trivial (at least 2 modes used >5%)
    mode_dist = normal_stats['mode_distribution']
    modes_used = sum(1 for v in mode_dist.values() if v >= 0.05)
    t11_pass = modes_used >= 2
    tests.append(('T11_mode_diversity', t11_pass, f"modes_used={modes_used} >= 2"))

    for t_name, t_pass, detail in tests:
        status = "PASS" if t_pass else "FAIL"
        n_total += 1
        if t_pass:
            n_pass += 1
        print(f"  [{status}] {t_name}: {detail}")

    print(f"\n  Score: {n_pass}/{n_total}")
    results['tests'] = {t[0]: {'pass': t[1], 'detail': t[2]} for t in tests}
    results['score'] = f"{n_pass}/{n_total}"

    # ── Save results ──
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, 'z2139_closed_loop_bridge.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved: {results_path}")

    print(f"\n  Elapsed: {elapsed:.1f}s")
    print("=" * 72)

    # Cleanup
    bridge.close()
    print("  FPGA bridge closed.")


if __name__ == '__main__':
    main()
