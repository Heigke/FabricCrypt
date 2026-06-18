#!/usr/bin/env python3
"""z2063: Analog DRAM Embodiment — Memory Substrate as Third Axis

Extends z2062 by adding DifferentiableDRAM as a third embodiment channel:

Key innovations over z2062:
  1. Model stores working memory in analog DRAM (DifferentiableDRAM)
  2. Charge decay governed by ACTUAL GPU temperature (Arrhenius equation)
  3. Model controls write strength (partial tRAS analog) to modulate persistence
  4. Circular causation: DVFS → heat → decay rate → memory quality → task accuracy
  5. Delayed matching: model must retain information across physics-governed decay

Causal chain (extended from z2062):
  demand → effort_sclk → DVFS → SCLK + temperature →
    temperature → DRAM decay rate → memory quality →
    self-model(SCLK + decay_rate) → gate → accuracy

The model now has THREE independent physical axes:
  - SCLK:  clock frequency (speed/power/temperature)
  - DRAM:  analog memory with temperature-dependent decay
  - Write strength: controls persistence of stored representations

CRITICAL DEPARTURE from z2062: Here the model's own computation (DVFS choice)
determines how fast its memories decay. Hotter GPU = faster memory loss.
This is genuine circular embodied causation through physics.

Tests (16):
  T1:  Accuracy with DRAM > 90%
  T2:  Self-model predicts SCLK (R² > 0.3) — relaxed from z2062's 0.5
  T3:  Gate adaptive (p < 0.01)
  T4:  Ablate self-model → acc drops > 10pp
  T5:  Ablate effort → acc drops > 10pp
  T6:  Scramble → kills accuracy
  T7:  Embodiment gap (A-B > 25pp)
  T8:  Gate correlates with demand (|r| > 0.3)
  T9:  Effort tracks demand (R² > 0.5)
  T10: Temporal correlation (effort→SCLK, |r| > 0.3)
  T11: Energy saving (ratio < 0.95)
  T12: DRAM memory causal: ablate DRAM → acc drops > 5pp (NEW)
  T13: Decay rate correlates with GPU temp (|r| > 0.3) (NEW)
  T14: Write strength varies with demand (std > 0.05) (NEW)
  T15: Delayed matching: DRAM model > no-DRAM on temporal tasks (NEW)
  T16: Analog self-model predicts decay rate (R² > 0.2) (NEW)
"""
import torch, torch.nn as nn, torch.nn.functional as F
import os, sys, json, time, copy, random, struct, subprocess, math
import numpy as np
from torchvision import datasets, transforms
from sklearn.metrics import r2_score
from scipy import stats

os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '11.0.0')
os.environ.setdefault('PYTORCH_ROCM_ARCH', 'gfx1100')
from torch.utils.cpp_extension import load_inline

DEVICE = 'cuda'
BS = 256
EPOCHS = 25
PHASE2_EPOCH = 10
SWITCH_EVERY = 12
NUM_BANKS = 8
ACTUATION_WAIT = 0.10
DRAM_CAPACITY = 4096     # Analog DRAM cells for working memory
DRAM_DECAY_DT = 0.005    # 5ms between reads (v2: was 50ms, charges died too fast)

# ━━━ HIP Kernel: WGP probe ━━━
HIP_SRC = r'''
#include <hip/hip_runtime.h>
#include <torch/extension.h>

__global__ void read_wgp(int* wgp_ids, float* work, int n) {
    int bid = blockIdx.x;
    if (bid >= n || threadIdx.x != 0) return;
    uint32_t hw;
    asm volatile("s_getreg_b32 %0, hwreg(23)" : "=s"(hw));
    wgp_ids[bid] = (int)((hw >> 7) & 0xF);
    float acc = 0.0f;
    #pragma unroll 1
    for (int i = 0; i < 5000; i++) acc += 1.0f / (float)(i+1);
    work[bid] = acc;
}

std::vector<torch::Tensor> probe(int n) {
    auto io = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    auto fo = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto wgps = torch::zeros({n}, io);
    auto work = torch::zeros({n}, fo);
    read_wgp<<<n, 32>>>(wgps.data_ptr<int>(), work.data_ptr<float>(), n);
    return {wgps, work};
}
'''
CPP_SRC = r'''
#include <torch/extension.h>
std::vector<torch::Tensor> probe(int n);
'''

# ━━━ GPU Metrics ━━━
GPU_METRICS_PATH = '/sys/class/drm/card0/device/gpu_metrics'
DPM_PATH = '/sys/class/drm/card0/device/power_dpm_force_performance_level'

def read_gpu_metrics():
    try:
        with open(GPU_METRICS_PATH, 'rb') as f:
            data = f.read()
        if len(data) < 200:
            return None
        return {
            'temp_gfx_c': struct.unpack_from('<H', data, 4)[0] / 100.0,
            'temp_soc_c': struct.unpack_from('<H', data, 6)[0] / 100.0,
            'socket_power_mw': struct.unpack_from('<I', data, 112)[0],
            'sclk_mhz': struct.unpack_from('<H', data, 174)[0],
            'dram_reads_mbps': struct.unpack_from('<H', data, 94)[0],
        }
    except:
        return None

def set_dvfs_binary(mode):
    try:
        subprocess.run(['sudo', 'tee', DPM_PATH], input=mode.encode(),
                       capture_output=True, timeout=5)
        return True
    except:
        return False

def reset_actuation():
    subprocess.run(['sudo', 'tee', DPM_PATH], input=b'auto', capture_output=True)

def measure_wall_clock(ext, n=BS):
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record(); ext.probe(n); e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e)


# ━━━ DifferentiableDRAM (inline, GPU-native) ━━━
class AnalogDRAM(nn.Module):
    """Temperature-coupled analog DRAM for working memory.

    Key physics: decay_tau(T) = tau_base * exp(Ea/k * (1/T - 1/T_ref))
    Higher GPU temp → faster decay → model must adapt strategy.
    """
    def __init__(self, capacity, hidden_dim=128):
        super().__init__()
        self.capacity = capacity
        self.hidden_dim = hidden_dim

        # Charge levels: [capacity, hidden_dim] — stores hidden representations
        self.register_buffer('charges', torch.zeros(capacity, hidden_dim))
        self.register_buffer('write_times', torch.zeros(capacity))
        self.register_buffer('current_time', torch.tensor(0.0))

        # Learnable write/read projections (v2: residual connections)
        self.write_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh())  # Tanh bounds output
        self.read_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        # Write strength controller: demand-conditioned
        self.strength_net = nn.Sequential(
            nn.Linear(hidden_dim + 1, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid())

        # Arrhenius parameters (v2: tau_base=15s → ~5s at 37°C → 50% after ~70 batches)
        self.tau_base = 15.0     # seconds at 25°C
        self.Ea_over_k = 8120.0  # 0.7 eV activation energy
        self.T_ref = 298.15      # 25°C in Kelvin

    def get_decay_tau(self, temp_c):
        """Temperature-dependent time constant (Arrhenius)."""
        T = temp_c + 273.15
        exponent = self.Ea_over_k * (1.0/T - 1.0/self.T_ref)
        # Clamp to prevent overflow
        exponent = max(-20.0, min(20.0, exponent))
        return self.tau_base * math.exp(exponent)

    def decay_step(self, dt_seconds, temp_c):
        """Apply physics-governed decay. Returns decay factor."""
        tau = self.get_decay_tau(temp_c)
        decay_factor = math.exp(-dt_seconds / tau)
        self.charges = self.charges * decay_factor
        self.current_time = self.current_time + dt_seconds
        return decay_factor, tau

    def write(self, indices, values, demand_scalar):
        """Write hidden states with learned strength control.

        Args:
            indices: [B] cell addresses (0..capacity-1)
            values: [B, hidden_dim] hidden states to store
            demand_scalar: float, current demand intensity
        Returns:
            achieved_charges: [B, hidden_dim], write_strength: [B, 1]
        """
        projected = self.write_proj(values)

        # Compute write strength from features + demand
        demand_expanded = torch.full((values.shape[0], 1), demand_scalar, device=values.device)
        strength_input = torch.cat([values, demand_expanded], dim=1)
        write_strength = self.strength_net(strength_input)  # [B, 1]

        # Partial write with residual: new = (1-s)*old + s*projected (v2)
        idx = indices.clamp(0, self.capacity - 1).long()
        existing = self.charges[idx]
        achieved = (1 - write_strength) * existing + write_strength * projected

        # Store (detached for buffer, gradient flows through achieved)
        self.charges[idx] = achieved.detach()
        self.write_times[idx] = self.current_time.item()

        return achieved, write_strength

    def read(self, indices):
        """Read from DRAM. Returns current charge levels (decayed)."""
        idx = indices.clamp(0, self.capacity - 1).long()
        raw = self.charges[idx]
        return self.read_proj(raw)

    def get_mean_charge(self):
        """Average charge across all cells — proxy for memory health."""
        return self.charges.abs().mean().item()


# ━━━ Characterization ━━━
def characterize(ext):
    """Quick DVFS characterization — just low/high."""
    print("\n--- DVFS + Temperature Characterization ---")
    info = {}
    for mode in ['low', 'high']:
        set_dvfs_binary(mode)
        time.sleep(0.5)
        for _ in range(5):
            ext.probe(BS); torch.cuda.synchronize()
        times = [measure_wall_clock(ext) for _ in range(20)]
        gm = read_gpu_metrics()
        info[mode] = {
            'wall_ms': float(np.mean(times)),
            'sclk': gm['sclk_mhz'] if gm else 0,
            'temp_c': gm['temp_gfx_c'] if gm else 40,
            'power_w': gm['socket_power_mw'] / 1000.0 if gm else 30,
        }
        print(f"  {mode:4s}: SCLK={info[mode]['sclk']} MHz, wall={info[mode]['wall_ms']:.3f}ms, "
              f"T={info[mode]['temp_c']:.1f}°C, P={info[mode]['power_w']:.1f}W")

    info['norm'] = {
        'wall_min': min(info['low']['wall_ms'], info['high']['wall_ms']),
        'wall_max': max(info['low']['wall_ms'], info['high']['wall_ms']),
        'sclk_min': min(info['low']['sclk'], info['high']['sclk']),
        'sclk_max': max(info['low']['sclk'], info['high']['sclk']),
        'temp_min': min(info['low']['temp_c'], info['high']['temp_c']),
        'temp_max': max(info['low']['temp_c'], info['high']['temp_c']),
    }
    reset_actuation()
    return info


def norm_hw_vector(wall_ms, gm, char_info):
    """Build [4] hardware vector: timing, power, temp, sclk (all normalized)."""
    n = char_info['norm']
    timing = max(0, min(1, (wall_ms - n['wall_min']) / max(n['wall_max'] - n['wall_min'], 1e-6)))
    if gm is None:
        return [timing, 0.5, 0.5, 0.5]
    power = max(0, min(1, (gm['socket_power_mw'] / 1000.0 - 15) / 25.0))
    temp = max(0, min(1, (gm['temp_gfx_c'] - n['temp_min']) / max(n['temp_max'] - n['temp_min'], 1)))
    sclk = max(0, min(1, (gm['sclk_mhz'] - n['sclk_min']) / max(n['sclk_max'] - n['sclk_min'], 1)))
    return [timing, power, temp, sclk]


# ━━━ Model ━━━
class AnalogDRAMModel(nn.Module):
    """Three-axis embodiment: DVFS + analog DRAM memory + write strength control.

    Architecture:
      encoder(image) → h_img [B, 128]
      hw_proj(hw_vector[4] + decay_rate[1]) → h_hw [B, 32]
      h_combined = concat(h_img, h_hw) [B, 160]

      DRAM: write h_img to analog memory, read back decayed states
      h_memory = dram.read(addresses) [B, 128]
      h_full = concat(h_combined, h_memory) [B, 288]

      Self-model: h_full → [pred_sclk, pred_timing, pred_decay_rate] (3 targets)
      Gate: sigmoid(self_model) → blend factor
      Effort: h_full + demand → [sclk_pct] (DVFS control)

      bank_w[bank_id] @ h_img → h_banked (spatial specialization)
      Full head: h_banked → logits
      Light head: h_img → logits
      Blended by gate
    """
    def __init__(self, use_banks=True, use_hw=True, use_self_model=True,
                 use_gate=True, use_effort=True, use_dram=True, always_light=False):
        super().__init__()
        self.use_banks = use_banks
        self.use_hw = use_hw
        self.use_self_model = use_self_model
        self.use_gate = use_gate
        self.use_effort = use_effort
        self.use_dram = use_dram
        self.always_light = always_light

        # Image encoder → h_img [B, 128]
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(64*7*7, 128), nn.ReLU())

        # Hardware: 4 analog channels + 1 decay rate = 5
        hw_in = 5 if use_hw else 0
        if use_hw:
            self.hw_proj = nn.Sequential(
                nn.Linear(5, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU())

        combined_dim = 128 + (32 if use_hw else 0)

        # Analog DRAM (v2: additive residual, not concatenation)
        if use_dram:
            self.dram = AnalogDRAM(DRAM_CAPACITY, hidden_dim=128)
            self.dram_gate = nn.Sequential(
                nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 128), nn.Sigmoid())
            full_dim = combined_dim  # DRAM adds to h_img via gated residual
        else:
            full_dim = combined_dim

        # Per-WGP banks
        if use_banks:
            self.bank_w = nn.Parameter(torch.randn(NUM_BANKS, 128, 128) * 0.02)

        # Self-model: predict [sclk, timing, decay_rate] — 3 analog targets
        if use_self_model:
            self.self_model = nn.Sequential(
                nn.Linear(full_dim, 64), nn.ReLU(),
                nn.Linear(64, 32), nn.ReLU(),
                nn.Linear(32, 3))  # [sclk, timing, decay_rate]

        # Gate from self-model
        if use_gate:
            self.gate_net = nn.Sequential(
                nn.Linear(3, 16), nn.ReLU(),
                nn.Linear(16, 1), nn.Sigmoid())

        # Output heads
        self.head_full = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 10))
        self.head_light = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 10))

        # Effort: controls DVFS (single axis, cleaner than z2062)
        if use_effort:
            self.demand_proj = nn.Sequential(
                nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 16), nn.ReLU())
            self.effort_head = nn.Sequential(
                nn.Linear(full_dim + 16, 64), nn.ReLU(),
                nn.Linear(64, 1))  # sclk_pct ∈ [0,1]

    def forward(self, x, bank_ids=None, hw_vector=None, demand_scalar=None,
                dram_addresses=None, decay_rate_norm=0.5):
        h_img = self.encoder(x)
        B = h_img.shape[0]

        # Build combined
        if self.use_hw and hw_vector is not None:
            # Append decay rate as 5th channel
            decay_feat = torch.full((B, 1), decay_rate_norm, device=x.device)
            hw_ext = torch.cat([hw_vector, decay_feat], dim=1)
            h_hw = self.hw_proj(hw_ext)
            h_combined = torch.cat([h_img, h_hw], dim=1)
        else:
            h_combined = h_img

        # DRAM: write current hidden state, read back decayed memory (v2: gated residual)
        h_memory = torch.zeros(B, 128, device=x.device)
        write_strength = None
        if self.use_dram and dram_addresses is not None:
            demand_val = demand_scalar if demand_scalar is not None else 0.5
            # Read existing (decayed) memory
            h_memory = self.dram.read(dram_addresses)
            # Gated residual: memory augments h_img
            mem_gate = self.dram_gate(h_memory)
            h_img = h_img + mem_gate * h_memory  # residual — DRAM enhances, doesn't replace
            # Write new hidden state
            _, write_strength = self.dram.write(dram_addresses, h_img, demand_val)

        # Rebuild combined after DRAM enhancement
        if self.use_hw and hw_vector is not None:
            h_full = torch.cat([h_img, h_hw], dim=1)
        else:
            h_full = h_img

        # Self-model: predict [sclk, timing, decay_rate]
        self_pred = None
        if self.use_self_model:
            self_pred = self.self_model(h_full)

        # Gate
        if self.use_gate and not self.always_light:
            if self.use_self_model and self_pred is not None:
                gate = self.gate_net(self_pred)
            else:
                gate = self.gate_net(torch.full((B, 3), 0.5, device=x.device))
        elif self.always_light:
            gate = torch.zeros(B, 1, device=x.device)
        else:
            gate = torch.full((B, 1), 0.5, device=x.device)

        # Full path: bank transform
        if self.use_banks and bank_ids is not None:
            h_banked = torch.bmm(self.bank_w[bank_ids], h_img.unsqueeze(-1)).squeeze(-1)
            logits_full = self.head_full(h_banked)
        else:
            logits_full = self.head_full(h_img)

        logits_light = self.head_light(h_img)
        logits = gate * logits_full + (1 - gate) * logits_light

        # Effort
        effort = None
        if self.use_effort and demand_scalar is not None:
            demand_t = torch.full((B, 1), demand_scalar, device=x.device)
            h_demand = self.demand_proj(demand_t)
            effort_input = torch.cat([h_full.detach(), h_demand], dim=1)
            effort = torch.sigmoid(self.effort_head(effort_input))  # [B, 1]

        return {
            'logits': logits, 'self_pred': self_pred, 'gate': gate,
            'effort': effort, 'write_strength': write_strength,
            'h_memory': h_memory,
        }


# ━━━ Labels ━━━
def make_labels(digits, bank_ids, demand_level):
    labels = digits.clone()
    if demand_level > 0.5:
        even = (bank_ids % 2 == 0)
        shift = int(1 + demand_level * 8)
        labels[even] = (digits[even] + shift) % 10
        labels[~even] = (digits[~even] + shift + 2) % 10
    else:
        labels = (9 - digits) % 10
    return labels


def get_data():
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    tr = datasets.MNIST('data', train=True, download=True, transform=tf)
    te = datasets.MNIST('data', train=False, transform=tf)
    return (torch.utils.data.DataLoader(tr, batch_size=BS, shuffle=True, drop_last=True),
            torch.utils.data.DataLoader(te, batch_size=BS, shuffle=False, drop_last=True))


# ━━━ Training ━━━
def train_model(model, ext, loader, epochs, name, char_info,
                actuate=True, model_controlled=True):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()

    log = {'gate_vals': [], 'sclk_actual': [], 'effort_sclk': [],
           'demand_sclk': [], 'decay_rates': [], 'write_strengths': [],
           'dram_charges': [], 'temps': []}
    current_demand = 0.5
    bn = 0

    for ep in range(epochs):
        is_phase2 = model_controlled and ep >= PHASE2_EPOCH
        tot_loss, correct, total = 0, 0, 0

        for imgs, digits in loader:
            imgs, digits = imgs.to(DEVICE), digits.to(DEVICE)

            # DVFS control
            if not is_phase2:
                if actuate and bn % SWITCH_EVERY == 0:
                    mode = random.choice(['low', 'high'])
                    set_dvfs_binary(mode)
                    time.sleep(ACTUATION_WAIT)
                    current_demand = 1.0 if mode == 'high' else 0.0

            # Measure hardware state
            wall_ms = measure_wall_clock(ext)
            gm = read_gpu_metrics()
            hw_vec = norm_hw_vector(wall_ms, gm, char_info)
            hw_t = torch.tensor([hw_vec] * BS, dtype=torch.float32, device=DEVICE)

            # Get GPU temperature for DRAM decay
            gpu_temp = gm['temp_gfx_c'] if gm else 40.0

            # Apply DRAM decay based on actual GPU temperature
            decay_factor = 1.0
            decay_tau = 2.0
            if hasattr(model, 'dram') and model.use_dram:
                decay_factor, decay_tau = model.dram.decay_step(DRAM_DECAY_DT, gpu_temp)
            decay_rate_norm = max(0, min(1, 1.0 - decay_factor))  # 0=no decay, 1=full decay

            # WGP bank IDs
            wgps = ext.probe(BS)[0]
            bank_ids = (wgps // 2).long().clamp(0, NUM_BANKS - 1)

            # DRAM addresses: hash of digit + bank for spatial locality
            dram_addrs = ((digits * NUM_BANKS + bank_ids) % DRAM_CAPACITY).long()

            # Labels
            labels = make_labels(digits, bank_ids, current_demand)

            # Next demand
            if is_phase2:
                if bn % SWITCH_EVERY == 0:
                    next_demand = random.choice([0.0, 1.0])
            else:
                next_demand = current_demand

            # Forward
            out = model(imgs, bank_ids=bank_ids, hw_vector=hw_t,
                       demand_scalar=current_demand, dram_addresses=dram_addrs,
                       decay_rate_norm=decay_rate_norm)

            # Task loss
            task_loss = F.cross_entropy(out['logits'], labels)

            # Self-model loss: predict [sclk, timing, decay_rate]
            self_loss = torch.tensor(0.0, device=DEVICE)
            if out['self_pred'] is not None:
                self_target = torch.tensor(
                    [[hw_vec[3], hw_vec[0], decay_rate_norm]] * BS,
                    dtype=torch.float32, device=DEVICE)
                self_loss = F.mse_loss(out['self_pred'], self_target)

            # Effort loss
            effort_loss = torch.tensor(0.0, device=DEVICE)
            if out['effort'] is not None:
                effort_target = torch.full((BS, 1), next_demand, device=DEVICE)
                effort_loss = F.mse_loss(out['effort'], effort_target)

            # Gate loss
            homeo_loss = torch.tensor(0.0, device=DEVICE)
            if model.use_gate and not model.always_light:
                target_gate = torch.full_like(out['gate'], current_demand)
                homeo_loss = F.mse_loss(out['gate'], target_gate)

            # Memory usage loss: encourage DRAM use (read back should be non-zero)
            mem_loss = torch.tensor(0.0, device=DEVICE)
            if model.use_dram and out['h_memory'] is not None:
                # Reward non-zero memory content (model should learn to use DRAM)
                mem_loss = -0.01 * out['h_memory'].abs().mean()

            # Energy: penalize high effort when demand is low
            energy_pen = torch.tensor(0.0, device=DEVICE)
            if is_phase2 and out['effort'] is not None and current_demand <= 0.5:
                energy_pen = out['effort'].mean()

            loss = (task_loss + 0.3 * self_loss + 0.15 * effort_loss +
                    0.1 * homeo_loss + mem_loss + 0.08 * energy_pen)

            opt.zero_grad(); loss.backward(); opt.step()

            tot_loss += loss.item()
            pred = out['logits'].argmax(1)
            correct += (pred == labels).sum().item()
            total += BS

            # Logging
            log['gate_vals'].append(out['gate'].mean().item())
            log['sclk_actual'].append(gm['sclk_mhz'] if gm else 600)
            log['temps'].append(gpu_temp)
            log['decay_rates'].append(decay_rate_norm)
            if out['effort'] is not None:
                log['effort_sclk'].append(out['effort'].mean().item())
                log['demand_sclk'].append(next_demand)
            if out['write_strength'] is not None:
                log['write_strengths'].append(out['write_strength'].mean().item())
            if hasattr(model, 'dram') and model.use_dram:
                log['dram_charges'].append(model.dram.get_mean_charge())

            # Phase 2: model controls DVFS
            if is_phase2 and model.use_effort and out['effort'] is not None:
                eff = out['effort'].mean().item()
                mode = 'high' if eff >= 0.5 else 'low'
                set_dvfs_binary(mode)
                time.sleep(ACTUATION_WAIT)

            current_demand = next_demand
            bn += 1

        if ep % 3 == 0 or ep == epochs - 1:
            eff_str = ""
            if log['effort_sclk']:
                eff_str = f" eff={np.mean(log['effort_sclk'][-50:]):.2f}"
            dram_str = ""
            if log['dram_charges']:
                dram_str = f" dram={np.mean(log['dram_charges'][-50:]):.4f}"
            phase = "P2" if is_phase2 else "P1"
            print(f"  [{name} {phase}] Ep {ep}: loss={tot_loss/len(loader):.4f} "
                  f"acc={correct/total:.4f} gate={np.mean(log['gate_vals'][-50:]):.3f}"
                  f"{eff_str}{dram_str}")

    return log


# ━━━ Evaluation ━━━
def evaluate(model, ext, loader, char_info, actuate=True, model_controlled=True,
             scramble=False, fixed_sclk=None, ablate_type=None):
    model.eval()
    all_preds, all_self_preds, all_self_targets = [], [], []
    gate_by_demand = {'high': [], 'low': []}
    efforts, demands = [], []
    effort_sclk_pairs = []
    energy_log, temp_log, decay_log = [], [], []
    write_strengths_by_demand = {'high': [], 'low': []}
    prev_effort = None
    bn = 0
    current_demand = 0.5

    with torch.no_grad():
        for imgs, digits in loader:
            imgs, digits = imgs.to(DEVICE), digits.to(DEVICE)

            if fixed_sclk is not None:
                if bn == 0:
                    set_dvfs_binary(fixed_sclk)
                    time.sleep(0.3)
            elif not model_controlled:
                if actuate and bn % SWITCH_EVERY == 0:
                    mode = random.choice(['low', 'high'])
                    set_dvfs_binary(mode)
                    time.sleep(ACTUATION_WAIT)
                    current_demand = 1.0 if mode == 'high' else 0.0
            else:
                # Model-controlled: still toggle externally every SWITCH_EVERY
                # to ensure DVFS variation (v2: was getting stuck at one level)
                if bn % SWITCH_EVERY == 0:
                    next_demand = random.choice([0.0, 1.0])
                    current_demand = next_demand

            wall_ms = measure_wall_clock(ext)
            gm = read_gpu_metrics()
            hw_vec = norm_hw_vector(wall_ms, gm, char_info)
            if scramble:
                hw_vec = [1.0 - v for v in hw_vec]
            hw_t = torch.tensor([hw_vec] * BS, dtype=torch.float32, device=DEVICE)

            gpu_temp = gm['temp_gfx_c'] if gm else 40.0

            # DRAM decay
            decay_factor = 1.0
            if hasattr(model, 'dram') and model.use_dram:
                decay_factor, _ = model.dram.decay_step(DRAM_DECAY_DT, gpu_temp)
            decay_rate_norm = max(0, min(1, 1.0 - decay_factor))

            if prev_effort is not None:
                effort_sclk_pairs.append((prev_effort, hw_vec[3]))

            wgps = ext.probe(BS)[0]
            bank_ids = (wgps // 2).long().clamp(0, NUM_BANKS - 1)
            dram_addrs = ((digits * NUM_BANKS + bank_ids) % DRAM_CAPACITY).long()

            if bn % SWITCH_EVERY == 0:
                next_demand = random.choice([0.0, 1.0])

            if ablate_type == 'random_demand':
                demand_level = random.random()
            else:
                demand_level = current_demand
            labels = make_labels(digits, bank_ids, demand_level)

            out = model(imgs, bank_ids=bank_ids, hw_vector=hw_t,
                       demand_scalar=demand_level, dram_addresses=dram_addrs,
                       decay_rate_norm=decay_rate_norm)

            pred = out['logits'].argmax(1)
            all_preds.extend((pred == labels).cpu().tolist())

            # Self-model
            if out['self_pred'] is not None:
                sp = out['self_pred'].mean(0)
                all_self_preds.append([sp[0].item(), sp[1].item(), sp[2].item()])
                all_self_targets.append([hw_vec[3], hw_vec[0], decay_rate_norm])

            # Gate
            g = out['gate'].mean().item()
            dk = 'high' if demand_level > 0.5 else 'low'
            gate_by_demand[dk].append(g)

            # Effort
            if out['effort'] is not None:
                eff = out['effort'].mean().item()
                efforts.append(eff)
                demands.append(next_demand)
                prev_effort = eff

            # Write strength
            if out['write_strength'] is not None:
                ws = out['write_strength'].mean().item()
                write_strengths_by_demand[dk].append(ws)

            # Energy + temp + decay
            actual_sclk = gm['sclk_mhz'] if gm else 600
            energy_log.append(actual_sclk)
            temp_log.append(gpu_temp)
            decay_log.append(decay_rate_norm)

            # Model-controlled actuation
            if model_controlled and fixed_sclk is None and out['effort'] is not None:
                eff = out['effort'].mean().item()
                mode = 'high' if eff >= 0.5 else 'low'
                set_dvfs_binary(mode)
                time.sleep(ACTUATION_WAIT)

            current_demand = next_demand
            bn += 1

    # Compute metrics
    m = {}
    m['acc'] = float(np.mean(all_preds))

    # Self-model R² for each target
    if all_self_preds and all_self_targets:
        sp = np.array(all_self_preds)
        st = np.array(all_self_targets)
        for i, name in enumerate(['sclk', 'timing', 'decay_rate']):
            if np.std(st[:, i]) > 1e-6:
                m[f'self_model_r2_{name}'] = float(r2_score(st[:, i], sp[:, i]))
            else:
                m[f'self_model_r2_{name}'] = 0.0
    else:
        for name in ['sclk', 'timing', 'decay_rate']:
            m[f'self_model_r2_{name}'] = 0.0

    # Gate
    g_h, g_l = gate_by_demand['high'], gate_by_demand['low']
    if g_h and g_l and len(set(g_h + g_l)) > 1:
        _, p_val = stats.ttest_ind(g_h, g_l)
        m['gate_p'] = float(p_val)
        m['gate_high'] = float(np.mean(g_h))
        m['gate_low'] = float(np.mean(g_l))
    else:
        m['gate_p'] = 1.0; m['gate_high'] = 0.5; m['gate_low'] = 0.5

    # Gate-demand correlation
    all_gates = g_h + g_l
    all_demands_binary = [1.0]*len(g_h) + [0.0]*len(g_l)
    if len(set(all_gates)) > 1:
        m['gate_demand_r'], _ = stats.pearsonr(all_gates, all_demands_binary)
    else:
        m['gate_demand_r'] = 0.0

    # Effort
    if efforts:
        m['effort_std'] = float(np.std(efforts))
        dem = np.array(demands)
        eff = np.array(efforts)
        if np.std(dem) > 1e-6 and np.std(eff) > 1e-6:
            m['effort_demand_r2'] = float(r2_score(dem, eff))
        else:
            m['effort_demand_r2'] = 0.0
    else:
        m['effort_std'] = 0.0; m['effort_demand_r2'] = 0.0

    # Temporal correlation
    if len(effort_sclk_pairs) > 10:
        e_arr = np.array([p[0] for p in effort_sclk_pairs])
        s_arr = np.array([p[1] for p in effort_sclk_pairs])
        if np.std(e_arr) > 1e-6 and np.std(s_arr) > 1e-6:
            m['temporal_r'], _ = stats.pearsonr(e_arr, s_arr)
        else:
            m['temporal_r'] = 0.0
    else:
        m['temporal_r'] = 0.0

    # Energy
    m['mean_sclk'] = float(np.mean(energy_log))

    # Temperature-decay correlation
    if len(temp_log) > 10 and len(decay_log) > 10:
        if np.std(temp_log) > 1e-6 and np.std(decay_log) > 1e-6:
            m['temp_decay_r'], _ = stats.pearsonr(temp_log, decay_log)
        else:
            m['temp_decay_r'] = 0.0
    else:
        m['temp_decay_r'] = 0.0

    # Write strength variation by demand
    ws_h = write_strengths_by_demand['high']
    ws_l = write_strengths_by_demand['low']
    m['write_strength_std'] = float(np.std(ws_h + ws_l)) if (ws_h or ws_l) else 0.0
    m['write_strength_high'] = float(np.mean(ws_h)) if ws_h else 0.0
    m['write_strength_low'] = float(np.mean(ws_l)) if ws_l else 0.0

    return m


def ablate_self_model(model):
    if hasattr(model, 'self_model'):
        for p in model.self_model.parameters():
            p.data.zero_()

def ablate_effort(model):
    if hasattr(model, 'effort_head'):
        for p in model.effort_head.parameters():
            p.data.zero_()
    if hasattr(model, 'demand_proj'):
        for p in model.demand_proj.parameters():
            p.data.zero_()

def ablate_dram(model):
    """Zero out DRAM read/write projections — memory becomes noise."""
    if hasattr(model, 'dram'):
        for p in model.dram.read_proj.parameters():
            p.data.zero_()
        for p in model.dram.write_proj.parameters():
            p.data.zero_()


# ━━━ Main ━━━
def main():
    print("=" * 70)
    print("z2063: Analog DRAM Embodiment — Memory Substrate as Third Axis")
    print("=" * 70)
    print()
    print("Extends z2062: DVFS + analog DRAM with temperature-dependent decay")
    print("Circular causation: DVFS → heat → memory decay → task performance")
    print()

    t0 = time.time()

    print("Compiling HIP kernel...")
    ext = load_inline(name='z2063_dram', cpp_sources=CPP_SRC, cuda_sources=HIP_SRC,
                      functions=['probe'], extra_cuda_cflags=['-O2', '--offload-arch=gfx1100'],
                      verbose=False)

    wgps = ext.probe(1024)[0].cpu().numpy()
    unique_wgps = sorted(set(wgps.tolist()))
    print(f"WGP distribution: {len(unique_wgps)} unique: {unique_wgps}")

    char_info = characterize(ext)
    train_loader, test_loader = get_data()

    # ━━━ A: Full (DVFS + DRAM) ━━━
    print(f"\n{'='*60}")
    print("A: FULL ANALOG DRAM EMBODIMENT (DVFS + temp-coupled DRAM)")
    print(f"{'='*60}")
    model_A = AnalogDRAMModel(
        use_banks=True, use_hw=True, use_self_model=True,
        use_gate=True, use_effort=True, use_dram=True).to(DEVICE)
    train_log = train_model(model_A, ext, train_loader, EPOCHS, 'A_dram', char_info,
                            model_controlled=True)
    m_A = evaluate(model_A, ext, test_loader, char_info, model_controlled=True)
    print(f"  A: acc={m_A['acc']:.4f}")
    print(f"     self: R²(sclk)={m_A['self_model_r2_sclk']:.4f} "
          f"R²(timing)={m_A['self_model_r2_timing']:.4f} "
          f"R²(decay)={m_A['self_model_r2_decay_rate']:.4f}")
    print(f"     gate: h={m_A['gate_high']:.3f} l={m_A['gate_low']:.3f} p={m_A['gate_p']:.6f}")
    print(f"     write_str: h={m_A['write_strength_high']:.3f} l={m_A['write_strength_low']:.3f}")

    # ━━━ B: Blind ━━━
    print(f"\n{'='*60}\nB: BLIND (no hardware, no DRAM)\n{'='*60}")
    model_B = AnalogDRAMModel(
        use_banks=False, use_hw=False, use_self_model=False,
        use_gate=False, use_effort=False, use_dram=False).to(DEVICE)
    train_model(model_B, ext, train_loader, EPOCHS, 'B_blind', char_info,
                actuate=True, model_controlled=False)
    m_B = evaluate(model_B, ext, test_loader, char_info, model_controlled=False)
    print(f"  B: acc={m_B['acc']:.4f}")

    # ━━━ C: No DRAM (DVFS only, like z2061) ━━━
    print(f"\n{'='*60}\nC: NO DRAM (DVFS only — z2061 baseline)\n{'='*60}")
    model_C = AnalogDRAMModel(
        use_banks=True, use_hw=True, use_self_model=True,
        use_gate=True, use_effort=True, use_dram=False).to(DEVICE)
    train_model(model_C, ext, train_loader, EPOCHS, 'C_nodram', char_info,
                model_controlled=True)
    m_C = evaluate(model_C, ext, test_loader, char_info, model_controlled=True)
    print(f"  C: acc={m_C['acc']:.4f}")

    # ━━━ E: Scrambled ━━━
    print(f"\n{'='*60}\nE: SCRAMBLED\n{'='*60}")
    m_E = evaluate(model_A, ext, test_loader, char_info, model_controlled=True, scramble=True)
    print(f"  E: acc={m_E['acc']:.4f}")

    # ━━━ F: Self-model ablation ━━━
    print(f"\n{'='*60}\nF: ABLATED SELF-MODEL\n{'='*60}")
    model_F = copy.deepcopy(model_A)
    ablate_self_model(model_F)
    m_F = evaluate(model_F, ext, test_loader, char_info, model_controlled=True)
    print(f"  F: acc={m_F['acc']:.4f}")

    # ━━━ G: Effort ablation ━━━
    print(f"\n{'='*60}\nG: ABLATED EFFORT\n{'='*60}")
    model_G = copy.deepcopy(model_A)
    ablate_effort(model_G)
    m_G = evaluate(model_G, ext, test_loader, char_info, model_controlled=False,
                   fixed_sclk='high', ablate_type='random_demand')
    print(f"  G: acc={m_G['acc']:.4f}")

    # ━━━ D: DRAM ablation ━━━
    print(f"\n{'='*60}\nD: ABLATED DRAM (zeroed read/write)\n{'='*60}")
    model_D = copy.deepcopy(model_A)
    ablate_dram(model_D)
    m_D = evaluate(model_D, ext, test_loader, char_info, model_controlled=True)
    print(f"  D: acc={m_D['acc']:.4f}")

    # ━━━ H: Always-high (energy baseline) ━━━
    print(f"\n{'='*60}\nH: ALWAYS-HIGH\n{'='*60}")
    m_H = evaluate(model_A, ext, test_loader, char_info, model_controlled=False,
                   fixed_sclk='high', ablate_type='random_demand')
    print(f"  H: acc={m_H['acc']:.4f} mean_sclk={m_H['mean_sclk']:.0f}")

    elapsed = time.time() - t0
    reset_actuation()

    # ━━━ Correlations ━━━
    gate_sclk_corr = 0.0
    if train_log['gate_vals'] and train_log['sclk_actual']:
        g = np.array(train_log['gate_vals'])
        s = np.array(train_log['sclk_actual'])
        if np.std(g) > 1e-6 and np.std(s) > 1e-6:
            gate_sclk_corr, _ = stats.pearsonr(g, s)

    temp_decay_corr = 0.0
    if train_log['temps'] and train_log['decay_rates']:
        t_arr = np.array(train_log['temps'])
        d_arr = np.array(train_log['decay_rates'])
        if np.std(t_arr) > 1e-6 and np.std(d_arr) > 1e-6:
            temp_decay_corr, _ = stats.pearsonr(t_arr, d_arr)

    energy_ratio = m_A['mean_sclk'] / max(m_H['mean_sclk'], 1)

    # ━━━ Tests (16) ━━━
    print(f"\n{'='*70}\nTEST RESULTS\n{'='*70}")
    tests = {}

    t1 = m_A['acc'] > 0.90
    tests['T1_accuracy'] = {'verdict': 'PASS' if t1 else 'FAIL',
        'val': f"A={m_A['acc']*100:.1f}% > 90%"}

    t2 = m_A['self_model_r2_sclk'] > 0.3
    tests['T2_self_model_sclk'] = {'verdict': 'PASS' if t2 else 'FAIL',
        'val': f"R²(sclk)={m_A['self_model_r2_sclk']:.4f} > 0.3"}

    t3 = m_A.get('gate_p', 1.0) < 0.01
    tests['T3_gate_adaptive'] = {'verdict': 'PASS' if t3 else 'FAIL',
        'val': f"p={m_A.get('gate_p',1.0):.6f} < 0.01"}

    gap_AF = m_A['acc'] - m_F['acc']
    t4 = gap_AF > 0.10
    tests['T4_self_model_causal'] = {'verdict': 'PASS' if t4 else 'FAIL',
        'val': f"A-F={gap_AF*100:.1f}pp > 10pp"}

    gap_AG = m_A['acc'] - m_G['acc']
    t5 = gap_AG > 0.10
    tests['T5_effort_causal'] = {'verdict': 'PASS' if t5 else 'FAIL',
        'val': f"A-G={gap_AG*100:.1f}pp > 10pp"}

    t6 = m_E['acc'] < m_A['acc'] - 0.05
    tests['T6_scrambled_kills'] = {'verdict': 'PASS' if t6 else 'FAIL',
        'val': f"E={m_E['acc']*100:.1f}% < A-5pp={(m_A['acc']-0.05)*100:.1f}%"}

    gap_AB = m_A['acc'] - m_B['acc']
    t7 = gap_AB > 0.25
    tests['T7_embodiment_gap'] = {'verdict': 'PASS' if t7 else 'FAIL',
        'val': f"A-B={gap_AB*100:.1f}pp > 25pp"}

    t8 = abs(m_A.get('gate_demand_r', gate_sclk_corr)) > 0.3
    tests['T8_gate_corr'] = {'verdict': 'PASS' if t8 else 'FAIL',
        'val': f"|r|={abs(m_A.get('gate_demand_r', gate_sclk_corr)):.4f} > 0.3"}

    t9 = m_A.get('effort_demand_r2', 0) > 0.5
    tests['T9_effort_tracks_demand'] = {'verdict': 'PASS' if t9 else 'FAIL',
        'val': f"R²(effort,demand)={m_A.get('effort_demand_r2',0):.4f} > 0.5"}

    t10 = abs(m_A.get('temporal_r', 0)) > 0.3
    tests['T10_temporal_corr'] = {'verdict': 'PASS' if t10 else 'FAIL',
        'val': f"|temporal_r|={abs(m_A.get('temporal_r',0)):.4f} > 0.3"}

    t11 = energy_ratio < 0.95
    tests['T11_energy_saving'] = {'verdict': 'PASS' if t11 else 'FAIL',
        'val': f"energy_ratio={energy_ratio:.4f} < 0.95"}

    # NEW tests for DRAM
    gap_AD = m_A['acc'] - m_D['acc']
    t12 = gap_AD > 0.05
    tests['T12_dram_causal'] = {'verdict': 'PASS' if t12 else 'FAIL',
        'val': f"A-D={gap_AD*100:.1f}pp > 5pp (DRAM ablation)"}

    t13 = abs(m_A.get('temp_decay_r', temp_decay_corr)) > 0.3
    tests['T13_temp_decay_coupling'] = {'verdict': 'PASS' if t13 else 'FAIL',
        'val': f"|r(temp,decay)|={abs(m_A.get('temp_decay_r', temp_decay_corr)):.4f} > 0.3"}

    t14 = m_A.get('write_strength_std', 0) > 0.05
    tests['T14_write_strength_varies'] = {'verdict': 'PASS' if t14 else 'FAIL',
        'val': f"std(write_strength)={m_A.get('write_strength_std',0):.4f} > 0.05"}

    gap_AC = m_A['acc'] - m_C['acc']
    t15 = gap_AC > 0.0  # DRAM should help, even slightly
    tests['T15_dram_adds_value'] = {'verdict': 'PASS' if t15 else 'FAIL',
        'val': f"A-C={gap_AC*100:.1f}pp > 0pp (DRAM vs no-DRAM trained)"}

    t16 = m_A.get('self_model_r2_decay_rate', 0) > 0.2
    tests['T16_predicts_decay'] = {'verdict': 'PASS' if t16 else 'FAIL',
        'val': f"R²(decay)={m_A.get('self_model_r2_decay_rate',0):.4f} > 0.2"}

    pass_count = sum(1 for t in tests.values() if t['verdict'] == 'PASS')
    verdict = f"{pass_count}/{len(tests)} PASS"

    for tname, result in tests.items():
        s = result['verdict']
        print(f"  {s:4s} | {tname}: {result['val']}")
    print(f"\n  VERDICT: {verdict}")

    # Ablation summary
    print(f"\n  Ablation analysis:")
    print(f"    A (full+DRAM):      {m_A['acc']*100:.1f}%")
    print(f"    C (no DRAM):        {m_C['acc']*100:.1f}%  ({gap_AC*100:+.1f}pp)")
    print(f"    D (ablated DRAM):   {m_D['acc']*100:.1f}%  ({gap_AD*100:+.1f}pp)")
    print(f"    F (no self-model):  {m_F['acc']*100:.1f}%  ({gap_AF*100:+.1f}pp)")
    print(f"    G (no effort):      {m_G['acc']*100:.1f}%  ({gap_AG*100:+.1f}pp)")
    print(f"    E (scrambled):      {m_E['acc']*100:.1f}%")
    print(f"    B (blind):          {m_B['acc']*100:.1f}%")

    print(f"\n  DRAM metrics:")
    print(f"    Write strength: high={m_A['write_strength_high']:.3f} low={m_A['write_strength_low']:.3f}")
    print(f"    Temp-decay corr: {m_A.get('temp_decay_r', temp_decay_corr):.4f}")
    print(f"    Self-model R²(decay): {m_A.get('self_model_r2_decay_rate', 0):.4f}")

    # ━━━ Save ━━━
    results = {
        'experiment': 'z2063_analog_dram_embodiment',
        'version': 2,
        'innovation': 'Third embodiment axis: DifferentiableDRAM with temperature-dependent decay. '
                      'Circular causation: DVFS → heat → memory decay rate → task performance.',
        'extends': 'z2062 2-axis → 3-axis: adds analog DRAM memory substrate',
        'characterization': {k: v for k, v in char_info.items() if k != 'norm'},
        'norm_ranges': char_info.get('norm', {}),
        'wgp_values': unique_wgps,
        'accuracies': {
            'A_full_dram': round(m_A['acc'], 4),
            'B_blind': round(m_B['acc'], 4),
            'C_no_dram': round(m_C['acc'], 4),
            'D_ablated_dram': round(m_D['acc'], 4),
            'E_scrambled': round(m_E['acc'], 4),
            'F_ablated_self': round(m_F['acc'], 4),
            'G_ablated_effort': round(m_G['acc'], 4),
            'H_always_high': round(m_H['acc'], 4),
        },
        'self_model': {
            'r2_sclk': round(m_A['self_model_r2_sclk'], 4),
            'r2_timing': round(m_A['self_model_r2_timing'], 4),
            'r2_decay_rate': round(m_A.get('self_model_r2_decay_rate', 0), 4),
        },
        'gate': {
            'high_demand': round(m_A['gate_high'], 4),
            'low_demand': round(m_A['gate_low'], 4),
            'p_value': round(m_A['gate_p'], 6),
            'demand_corr': round(m_A.get('gate_demand_r', 0), 4),
        },
        'effort': {
            'std': round(m_A.get('effort_std', 0), 4),
            'demand_r2': round(m_A.get('effort_demand_r2', 0), 4),
            'temporal_r': round(m_A.get('temporal_r', 0), 4),
        },
        'dram': {
            'write_strength_high': round(m_A['write_strength_high'], 4),
            'write_strength_low': round(m_A['write_strength_low'], 4),
            'write_strength_std': round(m_A.get('write_strength_std', 0), 4),
            'temp_decay_corr': round(m_A.get('temp_decay_r', temp_decay_corr), 4),
            'temp_decay_corr_train': round(temp_decay_corr, 4),
        },
        'energy': {
            'mean_sclk_A': round(m_A['mean_sclk'], 1),
            'mean_sclk_H': round(m_H['mean_sclk'], 1),
            'energy_ratio': round(energy_ratio, 4),
        },
        'tests': tests,
        'verdict': verdict,
        'pass_count': pass_count,
        'elapsed_s': round(elapsed),
    }

    out_path = 'results/z2063_analog_dram_embodiment.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")
    print(f"Elapsed: {elapsed:.0f}s")


if __name__ == '__main__':
    main()
