#!/usr/bin/env python3
"""
z2144v35: Synaptic Plasticity on FPGA<->GPU Bridge — T47-T49
=============================================================
Three tests certifying short-term and long-term synaptic plasticity
on the 8-neuron NS-RAM FPGA bank, coupled to GPT-2 via FPGAGatedLoRA.

  T47: Paired-Pulse Protocol       — STP facilitation/depression curve
  T48: Multi-Level Weight Retention — 8-level LTP + retention after 30s
  T49: Plasticity LM Advantage     — PPL improvement with adaptive Vg

Hardware:
  - AMD gfx1151 GPU  (HSA_OVERRIDE_GFX_VERSION=11.0.0)
  - Tang Nano 9K FPGA on /dev/ttyUSB1  (921600 / 115200 baud)

Run:
  HSA_OVERRIDE_GFX_VERSION=11.0.0 venv/bin/python -u scripts/z2144_synaptic_plasticity_v35.py
"""

import os, sys, json, math, time, struct, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from collections import deque

# Ensure HSA override for gfx1151
os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '11.0.0')

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / 'scripts'))
from nsram_fpga_bridge import NSRAMFPGABridge, read_gpu_temp_c, read_gpu_telemetry

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONSTANTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DEVICE = 'cuda'
BS = 4
SEQ_LEN = 128
N_EVAL_BATCHES = 30
N_NEURONS = 8
FPGA_PORT = '/dev/ttyUSB1'
FPGA_BAUD_FAST = 921600
FPGA_BAUD_SLOW = 115200
RESULTS_JSON = BASE / 'results' / 'z2144_synaptic_plasticity.json'

# Lanza NS-RAM reference parameters
LANZA_BV0 = 4.2
LANZA_ALPHA_T = 0.003
LANZA_T0 = 300.0
LANZA_BETA_VG = 1.8


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HELPER: Lempel-Ziv Complexity (LZ76)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def lempel_ziv_complexity(binary_string):
    """LZ76 complexity of a binary string."""
    n = len(binary_string)
    if n == 0:
        return 0.0
    s = binary_string + '0'
    i, k, l, c = 0, 1, 1, 1
    while k + l <= n:
        if s[i + l] == s[k + l]:
            l += 1
        else:
            if l >= k - i:
                k += l + 1
                c += 1
                i = k - 1
                l = 1
            else:
                i += 1
                if i == k:
                    k += 1
                    c += 1
                    i = k - 1
                    l = 1
                else:
                    l = 1
    if l > 0:
        c += 1
    norm = n / math.log2(n) if n > 1 else 1.0
    return c / norm


def spike_train_to_binary(spike_counts, threshold=None):
    """Convert spike counts to binary string (median threshold)."""
    arr = np.array(spike_counts, dtype=float)
    if threshold is None:
        threshold = float(np.median(arr))
    return ''.join('1' if s > threshold else '0' for s in arr)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HELPER: Transfer Entropy
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def transfer_entropy(source, target, k=1, lag=1):
    """Transfer entropy from source -> target using binning."""
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    n = min(len(source), len(target))
    if n < k + lag + 2:
        return 0.0
    n_bins = max(3, int(np.sqrt(n / 5)))

    def digitise(x):
        lo, hi = np.min(x), np.max(x)
        if hi == lo:
            return np.zeros(len(x), dtype=int)
        return np.clip(((x - lo) / (hi - lo) * n_bins).astype(int), 0, n_bins - 1)

    sd = digitise(source)
    td = digitise(target)
    valid_start = k + lag
    valid_end = n
    if valid_start >= valid_end:
        return 0.0
    N = valid_end - valid_start

    def encode_history(arr, t, k_len):
        val = 0
        for j in range(k_len):
            val = val * n_bins + int(arr[t - k_len + j])
        return val

    from collections import Counter
    cnt_tft_tp_sp = Counter()
    cnt_tp_sp = Counter()
    cnt_tft_tp = Counter()
    cnt_tp = Counter()

    for t in range(valid_start, valid_end):
        tf = int(td[t])
        tp = encode_history(td, t, k)
        sp = encode_history(sd, t - lag, k)
        cnt_tft_tp_sp[(tf, tp, sp)] += 1
        cnt_tp_sp[(tp, sp)] += 1
        cnt_tft_tp[(tf, tp)] += 1
        cnt_tp[tp] += 1

    te = 0.0
    for (tf, tp, sp), c_joint in cnt_tft_tp_sp.items():
        p_joint = c_joint / N
        p_tf_given_tp_sp = c_joint / cnt_tp_sp[(tp, sp)]
        p_tf_given_tp = cnt_tft_tp[(tf, tp)] / cnt_tp[tp]
        if p_tf_given_tp > 0 and p_tf_given_tp_sp > 0:
            te += p_joint * math.log2(p_tf_given_tp_sp / p_tf_given_tp)
    return te


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FPGAGatedLoRA + FEELBridgeModel  (self-contained copy)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class FPGAGatedLoRA(nn.Module):
    """LoRA adapter gated by FPGA spike telemetry."""

    def __init__(self, base_linear, rank=8, alpha=16):
        super().__init__()
        self.base_linear = base_linear
        self.rank = rank
        self.scaling = alpha / rank

        w = base_linear.weight
        if hasattr(base_linear, 'nf'):
            in_f, out_f = w.shape[0], w.shape[1]
        else:
            out_f, in_f = w.shape
        self.in_features = in_f
        self.out_features = out_f

        dtype = w.dtype
        self.lora_A = nn.Parameter(torch.randn(rank, in_f, dtype=dtype) * 0.01)
        self.lora_B = nn.Parameter(torch.randn(out_f, rank, dtype=dtype) * 0.001)

        self.gate_proj = nn.Linear(N_NEURONS * 2, rank, dtype=torch.float32)
        nn.init.normal_(self.gate_proj.weight, std=0.02)
        nn.init.constant_(self.gate_proj.bias, 0.0)

        self.last_gate = None
        self.open_loop = False

    def set_fpga_state(self, spike_counts, vmems):
        vec = np.zeros(N_NEURONS * 2, dtype=np.float32)
        for i in range(min(N_NEURONS, len(spike_counts))):
            vec[i] = spike_counts[i] / 100.0
            vec[N_NEURONS + i] = vmems[i] / 5.0
        self._fpga_vec = vec

    def forward(self, x):
        base_out = self.base_linear(x)
        x_cast = x.to(self.lora_A.dtype)
        lora_mid = F.linear(x_cast, self.lora_A)

        dev = self.gate_proj.weight.device
        if self.open_loop:
            gate = torch.full((self.rank,), 0.5, device=dev)
        elif hasattr(self, '_fpga_vec'):
            fvec = torch.from_numpy(self._fpga_vec).float().to(dev)
            gate = torch.sigmoid(self.gate_proj(fvec))
        else:
            gate = torch.full((self.rank,), 0.5, device=dev)

        self.last_gate = gate.detach().cpu().numpy()
        lora_out = F.linear(lora_mid * gate, self.lora_B)
        return base_out + lora_out * self.scaling


class FEELBridgeModel(nn.Module):
    """GPT-2 small with FPGAGatedLoRA on attention projection layers."""

    def __init__(self, layer_range=(6, 12), rank=4):
        super().__init__()
        from transformers import GPT2LMHeadModel
        self.gpt2 = GPT2LMHeadModel.from_pretrained('gpt2')
        for p in self.gpt2.parameters():
            p.requires_grad = False

        self.lora_layers = nn.ModuleList()
        for i in range(layer_range[0], layer_range[1]):
            attn = self.gpt2.transformer.h[i].attn
            original = attn.c_attn
            lora = FPGAGatedLoRA(original, rank=rank, alpha=rank * 2)
            attn.c_attn = lora
            self.lora_layers.append(lora)

    def set_fpga_state(self, spike_counts, vmems):
        for lora in self.lora_layers:
            lora.set_fpga_state(spike_counts, vmems)

    def set_open_loop(self, open_loop: bool):
        for lora in self.lora_layers:
            lora.open_loop = open_loop

    def forward(self, input_ids, labels=None):
        return self.gpt2(input_ids=input_ids, labels=labels)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HELPER: Evaluation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def get_wikitext_data(tokenizer, n_tokens=16384):
    """Load a chunk of WikiText-2 for evaluation."""
    from datasets import load_dataset
    ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    text = '\n'.join([r['text'] for r in ds if r['text'].strip()])
    ids = tokenizer.encode(text)[:n_tokens]
    return torch.tensor(ids, dtype=torch.long)


def eval_ppl(model, data, device, n_batches=N_EVAL_BATCHES):
    """Evaluate perplexity over n_batches micro-chunks."""
    model.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for b in range(n_batches):
            start = b * BS * SEQ_LEN
            end = start + BS * SEQ_LEN
            if end > len(data):
                break
            chunk = data[start:end].view(BS, SEQ_LEN).to(device)
            labels = chunk.clone()
            labels[:, :-1] = chunk[:, 1:]
            labels[:, -1] = -100
            out = model(chunk, labels=labels)
            total_loss += out.loss.item() * (BS * (SEQ_LEN - 1))
            total_tokens += BS * (SEQ_LEN - 1)
    if total_tokens == 0:
        return 999.0
    return math.exp(total_loss / total_tokens)


def closed_loop_step(model, fpga, data, batch_idx, device):
    """One closed-loop step: forward pass -> feedback -> FPGA -> telemetry -> model."""
    start = batch_idx * BS * SEQ_LEN
    end = start + BS * SEQ_LEN
    if end > len(data):
        return None, None
    chunk = data[start:end].view(BS, SEQ_LEN).to(device)
    labels = chunk.clone()
    labels[:, :-1] = chunk[:, 1:]
    labels[:, -1] = -100

    model.eval()
    with torch.no_grad():
        out = model(chunk, labels=labels)
    loss = out.loss.item()

    mac_signal = min(1.0, loss / 10.0)
    gpu_temp = read_gpu_temp_c()
    fpga.set_temperature(gpu_temp + 273.15)
    fpga.set_mac_signal(mac_signal)
    time.sleep(0.01)

    telem = fpga.read_telemetry()
    if telem is not None:
        spike_counts = [n['spike_count'] for n in telem['neurons']]
        vmems = [n['vmem'] for n in telem['neurons']]
        model.set_fpga_state(spike_counts, vmems)
    else:
        telem = {'total_spikes': 0, 'neurons': [{'spike_count': 0, 'vmem': 0}] * 8}

    return loss, telem


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SYNAPTIC PLASTICITY: STP, LTP, PlasticityManager
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class STPState:
    """Short-Term Plasticity state for one synapse (pre -> post).

    Implements Tsodyks-Markram model:
      u: facilitation variable (release probability), decays to U0
      x: depression variable (available resources), decays to 1.0
    On spike: effective_weight = u * x, then u += U0*(1-u), x -= u*x
    """

    def __init__(self, U0=0.2, tau_fac=0.2, tau_dep=0.5):
        self.U0 = U0
        self.tau_fac = tau_fac  # facilitation time constant (seconds)
        self.tau_dep = tau_dep  # depression time constant (seconds)
        self.u = U0             # facilitation (release probability)
        self.x = 1.0            # depression (available resources)

    def update(self, dt):
        """Decay u toward U0 and x toward 1.0 over dt seconds."""
        if dt > 0:
            self.u += (self.U0 - self.u) * (1.0 - math.exp(-dt / self.tau_fac))
            self.x += (1.0 - self.x) * (1.0 - math.exp(-dt / self.tau_dep))

    def on_spike(self):
        """Process a presynaptic spike. Returns effective synaptic weight."""
        # Facilitation: increase u
        self.u += self.U0 * (1.0 - self.u)
        # Effective weight
        eff = self.u * self.x
        # Depression: decrease x
        self.x -= self.u * self.x
        # Clamp
        self.u = max(0.0, min(1.0, self.u))
        self.x = max(0.0, min(1.0, self.x))
        return eff

    def reset(self):
        self.u = self.U0
        self.x = 1.0


class LTPState:
    """Long-Term Potentiation state for one neuron pair (pre -> post).

    8 discrete weight levels mapping to gate voltages.
    STDP rule: pre-before-post potentiates, post-before-pre depresses.
    """

    WEIGHT_VALUES = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    STDP_WINDOW = 0.020  # 20 ms

    def __init__(self, init_level=4):
        self.weight_level = init_level  # 0-7

    @property
    def vg(self):
        return self.WEIGHT_VALUES[self.weight_level]

    def stdp_update(self, dt_pre_post):
        """Apply STDP rule given dt = t_post - t_pre (seconds).

        dt > 0: pre fires before post -> potentiate (level++)
        dt < 0: post fires before pre -> depress (level--)
        Amplitude weighted by exp(-|dt| / STDP_WINDOW).
        """
        if abs(dt_pre_post) > self.STDP_WINDOW:
            return  # outside window, no update

        amplitude = math.exp(-abs(dt_pre_post) / self.STDP_WINDOW)
        # Probabilistic update: larger amplitude -> higher chance
        if random.random() < amplitude:
            if dt_pre_post > 0:
                # Pre before post: potentiate
                self.weight_level = min(7, self.weight_level + 1)
            elif dt_pre_post < 0:
                # Post before pre: depress
                self.weight_level = max(0, self.weight_level - 1)

    def reset(self, level=4):
        self.weight_level = level


class PlasticityManager:
    """Manages STP and LTP matrices for the 8-neuron bank.

    Tracks spike times from telemetry, updates STP/LTP states,
    and computes per-neuron Vg modulation.
    """

    def __init__(self, n_neurons=N_NEURONS, base_vg=0.35):
        self.n = n_neurons
        self.base_vg = base_vg

        # 8x8 STP matrix (pre x post)
        self.stp = [[STPState() for _ in range(n_neurons)] for _ in range(n_neurons)]
        # 8x8 LTP matrix (pre x post)
        self.ltp = [[LTPState() for _ in range(n_neurons)] for _ in range(n_neurons)]

        # Per-neuron last spike time (None = no spike detected yet)
        self.last_spike_time = [None] * n_neurons
        # Spike detection threshold (spike_count delta)
        self.spike_threshold = 2

    def process_telemetry(self, telem_now, telem_prev, dt):
        """Detect spikes from telemetry deltas and update STP/LTP.

        A neuron is considered to have "spiked" if its spike_count
        increased by more than spike_threshold since last read.
        """
        if telem_now is None or telem_prev is None or dt <= 0:
            return

        now_time = telem_now.get('timestamp', time.time())
        spiking_neurons = []

        for nid in range(self.n):
            sc_now = telem_now['neurons'][nid]['spike_count']
            sc_prev = telem_prev['neurons'][nid]['spike_count']
            delta = sc_now - sc_prev
            if delta > self.spike_threshold:
                spiking_neurons.append(nid)
                self.last_spike_time[nid] = now_time

        # Update STP: decay all synapses, then process spikes
        for pre in range(self.n):
            for post in range(self.n):
                if pre == post:
                    continue
                self.stp[pre][post].update(dt)

        for pre_nid in spiking_neurons:
            for post in range(self.n):
                if post == pre_nid:
                    continue
                self.stp[pre_nid][post].on_spike()

        # Update LTP via STDP: for each spiking pair, check timing
        for pre_nid in spiking_neurons:
            for post_nid in range(self.n):
                if post_nid == pre_nid:
                    continue
                if self.last_spike_time[post_nid] is not None:
                    dt_pre_post = self.last_spike_time[post_nid] - now_time
                    # dt_pre_post > 0 means post spiked AFTER pre (=now) -> potentiate
                    # dt_pre_post < 0 means post spiked BEFORE pre -> depress
                    # Since pre is spiking NOW and post spiked in past:
                    # dt = t_post - t_pre, post in past => dt < 0 => depress
                    self.ltp[pre_nid][post_nid].stdp_update(dt_pre_post)

        # Also check: for neurons that spiked now as POST, paired with
        # neurons that spiked recently as PRE
        for post_nid in spiking_neurons:
            for pre_nid in range(self.n):
                if pre_nid == post_nid:
                    continue
                if pre_nid in spiking_neurons:
                    continue  # already handled above
                if self.last_spike_time[pre_nid] is not None:
                    dt_pre_post = now_time - self.last_spike_time[pre_nid]
                    # pre spiked in past, post spiking now => dt > 0 => potentiate
                    self.ltp[pre_nid][post_nid].stdp_update(dt_pre_post)

    def get_vg_for_neuron(self, nid):
        """Compute Vg for neuron nid: base + STP modulation + LTP-derived Vg.

        STP modulation: average effective weight from all incoming synapses
        LTP contribution: average LTP Vg from all incoming synapses
        """
        stp_sum = 0.0
        ltp_vg_sum = 0.0
        count = 0
        for pre in range(self.n):
            if pre == nid:
                continue
            stp_sum += self.stp[pre][nid].u * self.stp[pre][nid].x
            ltp_vg_sum += self.ltp[pre][nid].vg
            count += 1

        if count == 0:
            return self.base_vg

        avg_stp = stp_sum / count
        avg_ltp_vg = ltp_vg_sum / count

        # STP modulation: deviation from baseline (U0=0.2) scaled
        stp_mod = (avg_stp - 0.2) * 0.3  # maps [-0.2, 0.8] -> [-0.06, 0.24]
        # LTP contribution: blend base_vg with LTP-derived average
        ltp_blend = 0.3  # 30% LTP influence
        vg = self.base_vg * (1.0 - ltp_blend) + avg_ltp_vg * ltp_blend + stp_mod

        return max(0.10, min(0.55, vg))

    def apply_to_fpga(self, bridge):
        """Write per-neuron Vg to FPGA based on current plasticity state."""
        for nid in range(self.n):
            vg = self.get_vg_for_neuron(nid)
            bridge.set_gate_voltage(nid, vg)

    def reset(self):
        """Reset all plasticity states."""
        for pre in range(self.n):
            for post in range(self.n):
                self.stp[pre][post].reset()
                self.ltp[pre][post].reset()
        self.last_spike_time = [None] * self.n


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# T47: Paired-Pulse Protocol
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def run_T47(fpga, plasticity_mgr):
    """T47: Paired-Pulse Protocol — STP facilitation/depression curve.

    For 3 neuron pairs at 6 ISIs:
      - Send Vg pulse to pre-neuron, measure post spike response
      - Send 2nd pulse after ISI, measure ratio
    PASS: |ratio-1.0| > 0.15 at short ISI AND |ratio-1.0| < 0.10 at long ISI
    """
    print("\n" + "=" * 60)
    print("T47: Paired-Pulse Protocol (STP facilitation/depression)")
    print("=" * 60)

    PAIRS = [(0, 1), (2, 3), (4, 5)]
    ISIS_MS = [10, 20, 50, 100, 200, 500]
    N_TRIALS = 10
    PULSE_VG = 0.50
    PULSE_DUR_S = 0.005  # 5ms
    REST_VG = 0.15       # quiescent Vg (minimal spiking)

    pair_results = {}

    for pre_nid, post_nid in PAIRS:
        pair_key = f"{pre_nid}->{post_nid}"
        print(f"  Pair {pair_key}:")
        ratios_by_isi = {}

        for isi_ms in ISIS_MS:
            isi_s = isi_ms / 1000.0
            trial_ratios = []

            for trial in range(N_TRIALS):
                # Reset: set all neurons to rest Vg
                for nid in range(N_NEURONS):
                    fpga.set_gate_voltage(nid, REST_VG)
                time.sleep(0.05)  # settle

                # Reset plasticity for clean measurement
                plasticity_mgr.reset()

                # Read baseline post-neuron spike count
                telem0 = fpga.read_telemetry()
                if telem0 is None:
                    continue
                baseline_sc = telem0['neurons'][post_nid]['spike_count']

                # 1st pulse: drive pre-neuron
                fpga.set_gate_voltage(pre_nid, PULSE_VG)
                time.sleep(PULSE_DUR_S)
                fpga.set_gate_voltage(pre_nid, REST_VG)

                # Read post response to 1st pulse
                time.sleep(0.01)  # propagation delay
                telem1 = fpga.read_telemetry()
                if telem1 is None:
                    continue
                resp1 = telem1['neurons'][post_nid]['spike_count'] - baseline_sc

                # Wait ISI
                time.sleep(isi_s)

                # Read before 2nd pulse
                telem_mid = fpga.read_telemetry()
                if telem_mid is None:
                    continue
                mid_sc = telem_mid['neurons'][post_nid]['spike_count']

                # 2nd pulse
                fpga.set_gate_voltage(pre_nid, PULSE_VG)
                time.sleep(PULSE_DUR_S)
                fpga.set_gate_voltage(pre_nid, REST_VG)

                time.sleep(0.01)
                telem2 = fpga.read_telemetry()
                if telem2 is None:
                    continue
                resp2 = telem2['neurons'][post_nid]['spike_count'] - mid_sc

                # Compute ratio (2nd / 1st), avoid division by zero
                if resp1 > 0:
                    ratio = resp2 / resp1
                elif resp2 > 0:
                    ratio = 2.0  # 2nd > 0 but 1st = 0: strong facilitation
                else:
                    ratio = 1.0  # both zero: no signal

                trial_ratios.append(ratio)

                # Update STP state based on telemetry
                if telem0 and telem2:
                    dt = telem2['timestamp'] - telem0['timestamp']
                    plasticity_mgr.process_telemetry(telem2, telem0, dt)

            avg_ratio = float(np.mean(trial_ratios)) if trial_ratios else 1.0
            std_ratio = float(np.std(trial_ratios)) if len(trial_ratios) > 1 else 0.0
            ratios_by_isi[isi_ms] = {'mean': avg_ratio, 'std': std_ratio,
                                      'n_trials': len(trial_ratios)}
            print(f"    ISI={isi_ms:4d}ms: ratio={avg_ratio:.3f} +/- {std_ratio:.3f}"
                  f" (n={len(trial_ratios)})")

        pair_results[pair_key] = ratios_by_isi

    # Evaluate PASS criteria
    any_pair_pass = False
    pair_pass_details = {}
    for pair_key, ratios_by_isi in pair_results.items():
        short_isis = [isi for isi in ISIS_MS if isi <= 50]
        long_isis = [isi for isi in ISIS_MS if isi >= 200]

        max_short_dev = max(abs(ratios_by_isi[isi]['mean'] - 1.0)
                           for isi in short_isis if isi in ratios_by_isi)
        min_long_dev = min(abs(ratios_by_isi[isi]['mean'] - 1.0)
                          for isi in long_isis if isi in ratios_by_isi)

        pair_pass = max_short_dev > 0.15 and min_long_dev < 0.10
        pair_pass_details[pair_key] = {
            'max_short_deviation': max_short_dev,
            'min_long_deviation': min_long_dev,
            'pass': pair_pass,
        }
        if pair_pass:
            any_pair_pass = True

    passed = any_pair_pass

    result = {
        'test': 'T47',
        'name': 'Paired-Pulse Protocol',
        'pair_results': {k: {str(isi): v for isi, v in d.items()}
                        for k, d in pair_results.items()},
        'pair_pass_details': pair_pass_details,
        'status': 'PASS' if passed else 'FAIL',
        'criterion': '|ratio-1|>0.15 at short ISI AND |ratio-1|<0.10 at long ISI (any pair)',
    }
    print(f"\n  Pass details: {pair_pass_details}")
    print(f"  => {result['status']}")
    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# T48: Multi-Level Weight Retention
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def run_T48(fpga):
    """T48: Multi-Level Weight Retention — 8 LTP levels + retention.

    Set neuron 0 to each of 8 Vg levels, measure spike rate.
    Wait 30s, re-measure at last level.
    PASS: monotonically ordered across >=6 of 8 AND retained within 20%.
    """
    print("\n" + "=" * 60)
    print("T48: Multi-Level Weight Retention (8-level LTP)")
    print("=" * 60)

    WEIGHT_VALUES = LTPState.WEIGHT_VALUES
    TEST_NEURON = 0
    N_READS = 20
    SETTLE_S = 0.5
    RETENTION_WAIT_S = 30.0

    spike_rates = []

    for level, vg in enumerate(WEIGHT_VALUES):
        # Set test neuron to this Vg
        fpga.set_gate_voltage(TEST_NEURON, vg)
        # Other neurons at nominal
        for nid in range(1, N_NEURONS):
            fpga.set_gate_voltage(nid, 0.35)
        time.sleep(SETTLE_S)

        # Measure spike rate: average over N_READS
        counts = []
        for _ in range(N_READS):
            telem = fpga.read_telemetry()
            if telem is not None:
                counts.append(telem['neurons'][TEST_NEURON]['spike_count'])
            time.sleep(0.02)

        avg_rate = float(np.mean(counts)) if counts else 0.0
        spike_rates.append(avg_rate)
        print(f"  Level {level} (Vg={vg:.2f}V): spike_rate={avg_rate:.1f}"
              f" (n={len(counts)} reads)")

    # Check monotonicity: count how many adjacent pairs are in order
    n_monotonic = 0
    for i in range(len(spike_rates) - 1):
        if spike_rates[i + 1] >= spike_rates[i]:
            n_monotonic += 1
    # Also count strict monotonic from the full sequence perspective
    # (6 of 8 means 6 levels in correct relative order)
    # We check: how many levels are in a longest increasing subsequence
    def lis_length(arr):
        """Length of longest increasing subsequence."""
        if not arr:
            return 0
        tails = []
        for x in arr:
            import bisect
            pos = bisect.bisect_left(tails, x)
            if pos == len(tails):
                tails.append(x)
            else:
                tails[pos] = x
        return len(tails)

    lis_len = lis_length(spike_rates)
    monotonic_pass = lis_len >= 6

    print(f"\n  Monotonicity: {n_monotonic}/7 adjacent pairs ordered, "
          f"LIS={lis_len}/8")

    # Retention test: wait 30s at last Vg, re-measure
    print(f"  Waiting {RETENTION_WAIT_S}s for retention test...")
    last_vg = WEIGHT_VALUES[-1]
    fpga.set_gate_voltage(TEST_NEURON, last_vg)
    time.sleep(RETENTION_WAIT_S)

    # Re-measure
    retained_counts = []
    for _ in range(N_READS):
        telem = fpga.read_telemetry()
        if telem is not None:
            retained_counts.append(telem['neurons'][TEST_NEURON]['spike_count'])
        time.sleep(0.02)

    retained_rate = float(np.mean(retained_counts)) if retained_counts else 0.0
    initial_rate = spike_rates[-1]  # rate at last level before wait
    if initial_rate > 0:
        retention_ratio = retained_rate / initial_rate
    else:
        retention_ratio = 1.0 if retained_rate == 0 else 0.0

    retention_pass = abs(retention_ratio - 1.0) < 0.20  # within 20%

    print(f"  Retention: initial={initial_rate:.1f}, after 30s={retained_rate:.1f}, "
          f"ratio={retention_ratio:.3f}")

    passed = monotonic_pass and retention_pass

    result = {
        'test': 'T48',
        'name': 'Multi-Level Weight Retention',
        'weight_values': WEIGHT_VALUES,
        'spike_rates': spike_rates,
        'n_monotonic_pairs': n_monotonic,
        'lis_length': lis_len,
        'monotonic_pass': monotonic_pass,
        'initial_rate_last_level': initial_rate,
        'retained_rate': retained_rate,
        'retention_ratio': retention_ratio,
        'retention_pass': retention_pass,
        'status': 'PASS' if passed else 'FAIL',
        'criterion': 'LIS>=6 of 8 levels AND retention within 20% after 30s',
    }
    print(f"  => monotonic={monotonic_pass}, retention={retention_pass} => {result['status']}")
    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# T49: Plasticity LM Advantage
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def run_T49(model, fpga, data, device, plasticity_mgr):
    """T49: Plasticity LM Advantage — PPL improvement with adaptive Vg.

    1. Baseline PPL: frozen GPT-2, no FPGA
    2. Plasticity-enabled: STP+LTP updating Vg each batch (50 batches)
    3. Plasticity-frozen: fixed Vg=0.35 for all neurons (50 batches)
    PASS: PPL_plastic / PPL_frozen < 0.99 (1% improvement)
    """
    print("\n" + "=" * 60)
    print("T49: Plasticity LM Advantage (adaptive vs fixed Vg)")
    print("=" * 60)

    N_PLASTIC_BATCHES = 50

    # Phase 1: Baseline PPL (no FPGA, open loop)
    print("  Phase 1: Baseline PPL (frozen GPT-2, open loop)...")
    model.set_open_loop(True)
    baseline_ppl = eval_ppl(model, data, device, n_batches=N_EVAL_BATCHES)
    print(f"    Baseline PPL = {baseline_ppl:.4f}")

    # Phase 2: Plasticity-frozen (fixed Vg=0.35, closed loop)
    print(f"  Phase 2: Fixed Vg=0.35, closed loop ({N_PLASTIC_BATCHES} batches)...")
    model.set_open_loop(False)
    for nid in range(N_NEURONS):
        fpga.set_gate_voltage(nid, 0.35)
    time.sleep(0.2)

    losses_frozen = []
    for b in range(N_PLASTIC_BATCHES):
        loss, telem = closed_loop_step(model, fpga, data, b, device)
        if loss is not None:
            losses_frozen.append(loss)
    ppl_frozen = math.exp(np.mean(losses_frozen)) if losses_frozen else 999.0
    print(f"    PPL_frozen = {ppl_frozen:.4f} ({len(losses_frozen)} batches)")

    # Phase 3: Plasticity-enabled (STP+LTP updating Vg)
    print(f"  Phase 3: Plasticity-enabled ({N_PLASTIC_BATCHES} batches)...")
    model.set_open_loop(False)
    plasticity_mgr.reset()

    # Set initial Vg from plasticity manager
    plasticity_mgr.apply_to_fpga(fpga)
    time.sleep(0.2)

    losses_plastic = []
    prev_telem = None
    for b in range(N_PLASTIC_BATCHES):
        loss, telem = closed_loop_step(model, fpga, data, b, device)
        if loss is not None:
            losses_plastic.append(loss)

        # Update plasticity from telemetry
        if telem is not None and prev_telem is not None:
            dt = telem.get('timestamp', time.time()) - prev_telem.get('timestamp', 0)
            if dt > 0:
                plasticity_mgr.process_telemetry(telem, prev_telem, dt)
                # Apply updated Vg to FPGA
                plasticity_mgr.apply_to_fpga(fpga)
        prev_telem = telem

    ppl_plastic = math.exp(np.mean(losses_plastic)) if losses_plastic else 999.0
    print(f"    PPL_plastic = {ppl_plastic:.4f} ({len(losses_plastic)} batches)")

    # Compute advantage ratio
    if ppl_frozen > 0:
        ratio = ppl_plastic / ppl_frozen
    else:
        ratio = 1.0
    passed = ratio < 0.99

    # Collect plasticity state summary
    ltp_levels = []
    for pre in range(N_NEURONS):
        for post in range(N_NEURONS):
            if pre != post:
                ltp_levels.append(plasticity_mgr.ltp[pre][post].weight_level)
    ltp_mean = float(np.mean(ltp_levels))
    ltp_std = float(np.std(ltp_levels))

    result = {
        'test': 'T49',
        'name': 'Plasticity LM Advantage',
        'baseline_ppl': baseline_ppl,
        'ppl_frozen': ppl_frozen,
        'ppl_plastic': ppl_plastic,
        'ratio': ratio,
        'ltp_mean_level': ltp_mean,
        'ltp_std_level': ltp_std,
        'n_frozen_batches': len(losses_frozen),
        'n_plastic_batches': len(losses_plastic),
        'status': 'PASS' if passed else 'FAIL',
        'criterion': 'PPL_plastic / PPL_frozen < 0.99',
    }
    print(f"  => ratio={ratio:.6f} (plastic/frozen), LTP mean level={ltp_mean:.1f}+/-{ltp_std:.1f}")
    print(f"  => {result['status']}")
    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SIMULATED FPGA (fallback when hardware unavailable)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class SimulatedFPGA:
    """Minimal FPGA simulator for testing without hardware.

    Produces plausible telemetry: spike rate depends on Vg and temperature.
    Includes inter-neuron coupling for STP/paired-pulse realism.
    """

    def __init__(self):
        self.vg = [0.35] * N_NEURONS
        self.temp_k = 300.0
        self.mac_val = 0.5
        self.kill = False
        self.telemetry_history = deque(maxlen=1000)
        self.spike_history = deque(maxlen=10000)
        self._step = 0
        self._last_spikes = [0] * N_NEURONS  # for paired-pulse memory

    def set_gate_voltage(self, neuron_id, vg):
        if 0 <= neuron_id < N_NEURONS:
            self.vg[neuron_id] = vg

    def set_temperature(self, temp_k):
        self.temp_k = temp_k

    def set_mac_signal(self, mac_val):
        self.mac_val = mac_val

    def set_kill_switch(self, enabled):
        self.kill = enabled

    def read_telemetry(self):
        self._step += 1
        neurons = []
        for i in range(N_NEURONS):
            if self.kill:
                sc, vm, bv = 0, 0.0, 0.0
            else:
                vg = self.vg[i]
                t_factor = 1.0 + 0.002 * (self.temp_k - 300.0)
                base_rate = (vg ** 2) * 200.0 * t_factor
                rate = base_rate * (0.8 + 0.4 * self.mac_val)

                # Paired-pulse memory: if neuron recently had high spikes,
                # facilitate (boost by 20-40% at short intervals)
                if self._last_spikes[i] > 10:
                    rate *= 1.0 + 0.3 * min(1.0, self._last_spikes[i] / 50.0)

                sc = max(0, int(rate + random.gauss(0, max(1, rate * 0.15))))
                bv = LANZA_BV0 * math.exp(-LANZA_ALPHA_T * (self.temp_k - LANZA_T0)) * \
                     (1.0 + LANZA_BETA_VG * vg)
                vm = vg * 0.8 + random.gauss(0, 0.02)
            neurons.append({'spike_count': sc, 'vmem': vm, 'bvpar': bv})
            self._last_spikes[i] = sc

        result = {
            'timestamp': time.time(),
            'neurons': neurons,
            'total_spikes': sum(n['spike_count'] for n in neurons),
            'mean_vmem': float(np.mean([n['vmem'] for n in neurons])),
            'mean_bvpar': float(np.mean([n['bvpar'] for n in neurons])),
        }
        self.telemetry_history.append(result)
        self.spike_history.append((result['timestamp'], result['total_spikes']))
        return result

    def close(self):
        pass


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN: Run battery + scorecard
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main():
    print("=" * 70)
    print("z2144v35: Synaptic Plasticity Battery — T47-T49")
    print("=" * 70)

    device = torch.device(DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  HSA_OVERRIDE_GFX_VERSION="
              f"{os.environ.get('HSA_OVERRIDE_GFX_VERSION', 'unset')}")

    # Load GPT-2 tokenizer + data
    print("\nLoading GPT-2 tokenizer and evaluation data...")
    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    eval_data = get_wikitext_data(tokenizer, n_tokens=32768)
    print(f"  Eval tokens: {len(eval_data)}")

    # Build model
    print("\nBuilding FEELBridgeModel (GPT-2 + FPGAGatedLoRA, layers 6-11, rank 4)...")
    model = FEELBridgeModel(layer_range=(6, 12), rank=4).to(device)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {n_trainable:,}")

    # Baseline PPL (no FPGA)
    print("\nBaseline PPL (frozen GPT-2, no FPGA feedback)...")
    model.set_open_loop(True)
    baseline_ppl = eval_ppl(model, eval_data, device)
    print(f"  Baseline PPL = {baseline_ppl:.4f}")

    # Connect FPGA (try fast baud first, fall back to slow)
    print(f"\nConnecting to FPGA on {FPGA_PORT}...")
    fpga = None
    fpga_ok = False
    for baud in [FPGA_BAUD_FAST, FPGA_BAUD_SLOW]:
        try:
            fpga = NSRAMFPGABridge(port=FPGA_PORT, baudrate=baud)
            telem = fpga.read_telemetry()
            if telem:
                fpga_ok = True
                print(f"  FPGA connected at {baud} baud.")
                print(f"  Initial telemetry: spikes={telem['total_spikes']}, "
                      f"bvpar={telem['mean_bvpar']:.3f}V")
                break
            else:
                print(f"  Connected at {baud} but no telemetry, trying next...")
                if hasattr(fpga, 'close'):
                    fpga.close()
                fpga = None
        except Exception as e:
            print(f"  {baud} baud failed: {e}")
            fpga = None

    if not fpga_ok:
        print("  Running in SIMULATED mode (results will be synthetic)")
        fpga = SimulatedFPGA()

    # Create plasticity manager
    plasticity_mgr = PlasticityManager(n_neurons=N_NEURONS, base_vg=0.35)

    # Run T47-T49
    results = []
    try:
        results.append(run_T47(fpga, plasticity_mgr))
        results.append(run_T48(fpga))
        results.append(run_T49(model, fpga, eval_data, device, plasticity_mgr))
    finally:
        if hasattr(fpga, 'close'):
            fpga.close()
            print("\nFPGA bridge closed.")

    # ━━━ SCORECARD ━━━
    print("\n" + "=" * 70)
    print("SCORECARD: Synaptic Plasticity Battery T47-T49")
    print("=" * 70)
    n_pass = 0
    for r in results:
        status = r.get('status', 'FAIL')
        marker = 'PASS' if status == 'PASS' else 'FAIL'
        if status == 'PASS':
            n_pass += 1
        print(f"  {r['test']:5s} {r['name']:35s} {marker:4s}  ({r.get('criterion', '')})")
    print(f"\n  Total: {n_pass}/{len(results)} PASS")
    print(f"  FPGA mode: {'REAL' if fpga_ok else 'SIMULATED'}")
    print(f"  Baseline PPL: {baseline_ppl:.4f}")

    # Save results
    output = {
        'experiment': 'z2144_synaptic_plasticity_v35',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'device': str(device),
        'fpga_real': fpga_ok,
        'baseline_ppl': baseline_ppl,
        'n_pass': n_pass,
        'n_total': len(results),
        'tests': results,
    }
    RESULTS_JSON.parent.mkdir(exist_ok=True)
    with open(RESULTS_JSON, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to {RESULTS_JSON}")


if __name__ == '__main__':
    main()
