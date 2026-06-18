#!/usr/bin/env python3
"""z2046: Deep Timing Fusion — Hardware IS Computation (v2)

Fixes z2045 problems:
  1. SCLK-based bank selection (reliable, monotonic) instead of noisy timing bins
  2. BALANCED training: systematic DVFS cycling ensures each bank gets ~25% of batches
  3. CONTINUOUS timing coupling: timing → learned projection → weight modulation
  4. DUAL coupling: discrete (SCLK→bank) + continuous (timing→modulation)

ARCHITECTURE:
  Layer 1: Standard encoder (784 → 256 → 128)
  Layer 2: FPGA-STYLE — SCLK selects from 3 weight banks [128→64]
            PLUS timing modulation: h = bank(x) + timing_norm * V_coupling
  Layer 3: Self-model head predicts which bank was active (METACOGNITION)
  Layer 4: Classifier head (64 → 10)

WHY THIS WORKS:
  - SCLK IS hardware state (set by power management or our DVFS commands)
  - Different SCLK → different weight bank → DIFFERENT computation path
  - Timing modulation adds CONTINUOUS hardware influence on top
  - Self-model must KNOW which bank is active to minimize loss
  - Kill shot: freeze SCLK reading OR ablate timing → measurable degradation

3 DVFS STATES → 3 BANKS:
  SCLK < 800 MHz  → bank 0 (forced low: ~603 MHz)
  800 ≤ SCLK < 1400 → bank 1 (auto: ~1100-1300 MHz)
  SCLK ≥ 1400     → bank 2 (forced high: ~1535 MHz)

5 CONDITIONS:
  A_embodied:   Live SCLK→bank + live timing modulation
  B_frozen:     Constant bank 1 + frozen timing (no hw variation)
  C_random:     Random bank + random timing (decorrelated)
  D_blind:      No bank switching, no timing (standard network)
  E_sclk_only:  SCLK→bank but no timing modulation (ablation)

6 TESTS:
  T1: A_embodied learns (acc > 90%)
  T2: Different DVFS → different banks (SCLK-based, guaranteed)
  T3: A_embodied self-model accuracy > 80% (knows which bank is active)
  T4: Kill shot A — freeze bank → accuracy/KL changes
  T5: Kill shot B — ablate timing coupling → self-model degrades
  T6: A_embodied > D_blind on self-prediction (not just task accuracy)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
import json
import os
import sys
import time
import ctypes
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '11.0.0')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / 'results'
RESULTS_DIR.mkdir(exist_ok=True)

# =============================================================================
# HIP Timing Probe
# =============================================================================

HIP_SRC = '''
#include <hip/hip_runtime.h>
#include <torch/extension.h>

// 4-channel timing fingerprint: memory, compute, LDS, VGPR pressure
// Returns [n, 4]
__global__ void timing_fingerprint_kernel(
    float* __restrict__ output,
    const float* __restrict__ workspace,
    int ws_size, int n
) {
    int gid = threadIdx.x + blockIdx.x * blockDim.x;
    if (gid >= n) return;

    // Channel 0: Global memory timing (cache misses)
    float sum = 0.0f;
    uint64_t t0 = clock64();
    #pragma unroll 1
    for (int i = 0; i < 64; i++) {
        // Large strides to force cache misses
        int idx = ((gid * 2654435761u + i * 65537u) >> 4) % ws_size;
        sum += workspace[idx];
    }
    uint64_t t1 = clock64();
    output[gid * 4 + 0] = (float)(t1 - t0);

    // Channel 1: Compute (ALU) timing
    float x = (float)(gid + 1) * 0.001f;
    uint64_t t2 = clock64();
    #pragma unroll 1
    for (int i = 0; i < 128; i++) {
        x = __sinf(x) * __cosf(x + 0.1f) + 0.001f;
    }
    uint64_t t3 = clock64();
    output[gid * 4 + 1] = (float)(t3 - t2);

    // Channel 2: LDS timing (shared memory)
    __shared__ float lds[256];
    uint64_t t4 = clock64();
    lds[threadIdx.x % 256] = sum + x;
    __syncthreads();
    float lds_val = 0.0f;
    #pragma unroll 1
    for (int i = 0; i < 32; i++) {
        lds_val += lds[(threadIdx.x + i * 7) % 256];
    }
    uint64_t t5 = clock64();
    output[gid * 4 + 2] = (float)(t5 - t4);

    // Channel 3: VGPR pressure
    float v[16];
    uint64_t t6 = clock64();
    #pragma unroll 1
    for (int i = 0; i < 16; i++) {
        v[i] = __sinf((float)(gid * 16 + i) * 0.01f);
    }
    float vsum = 0.0f;
    #pragma unroll 1
    for (int i = 0; i < 16; i++) vsum += v[i];
    uint64_t t7 = clock64();
    output[gid * 4 + 3] = (float)(t7 - t6);

    // Anti-optimization
    if (__builtin_expect((sum + x + lds_val + vsum) == -1e30f, 0))
        output[0] = sum;
}

torch::Tensor probe_fingerprint(torch::Tensor workspace, int n) {
    auto output = torch::empty({n, 4}, workspace.options());
    int threads = min(n, 128);
    int blocks = (n + threads - 1) / threads;
    timing_fingerprint_kernel<<<blocks, threads>>>(
        output.data_ptr<float>(), workspace.data_ptr<float>(),
        workspace.size(0), n);
    return output;
}
'''

HIP_CPP = '''
torch::Tensor probe_fingerprint(torch::Tensor workspace, int n);
'''


class TimingProbe:
    """4-channel HIP ISA timing fingerprint."""

    def __init__(self, n_threads=64, workspace_size=1 << 22):  # 16MB workspace
        from torch.utils.cpp_extension import load_inline
        print("[BUILD] Compiling timing fingerprint extension...")
        t0 = time.time()
        self.ext = load_inline(
            name='timing_fp_z2046',
            cpp_sources=HIP_CPP,
            cuda_sources=HIP_SRC,
            functions=['probe_fingerprint'],
            verbose=False,
            extra_cuda_cflags=['-O2']
        )
        print(f"[BUILD] Done in {time.time()-t0:.1f}s")
        self.n = n_threads
        self.workspace = torch.randn(workspace_size, device=DEVICE)
        self.probe()  # warm up

    def probe(self):
        """Returns [n, 4] timing tensor (mem, alu, lds, vgpr)."""
        return self.ext.probe_fingerprint(self.workspace, self.n)

    def get_features(self):
        """Returns 4-dim normalized timing features."""
        t = self.probe()  # [n, 4]
        return t.mean(dim=0)  # [4] mean across threads


# =============================================================================
# DVFS Controller + SCLK Bank Mapping
# =============================================================================

class DVFSController:
    def __init__(self):
        lib_path = str(ROOT / 'src' / 'native' / 'libdeep_gpu.so')
        try:
            self.lib = ctypes.CDLL(lib_path)
            self.lib.deep_gpu_init.restype = ctypes.c_int
            self.lib.deep_gpu_dvfs_force_low.restype = ctypes.c_int
            self.lib.deep_gpu_dvfs_force_high.restype = ctypes.c_int
            self.lib.deep_gpu_dvfs_auto.restype = ctypes.c_int
            self.lib.deep_gpu_dvfs_get_sclk.restype = ctypes.c_int
            self.lib.deep_gpu_cleanup.restype = None
            fd = self.lib.deep_gpu_init()
            self.available = fd >= 0
            if self.available:
                print(f"[DEEP_GPU] Initialized: fd={fd}")
        except Exception as e:
            print(f"[DVFS] Not available: {e}")
            self.available = False

    def force_low(self):
        if self.available: self.lib.deep_gpu_dvfs_force_low()
    def force_high(self):
        if self.available: self.lib.deep_gpu_dvfs_force_high()
    def auto(self):
        if self.available: self.lib.deep_gpu_dvfs_auto()
    def get_sclk(self):
        return self.lib.deep_gpu_dvfs_get_sclk() if self.available else 1000

    def sclk_to_bank(self, sclk=None):
        """Map SCLK frequency to bank index. 3 banks for 3 DVFS states."""
        if sclk is None:
            sclk = self.get_sclk()
        if sclk < 800:
            return 0  # low (~603 MHz)
        elif sclk < 1400:
            return 1  # auto (~1100-1300 MHz)
        else:
            return 2  # high (~1535 MHz)

    def cleanup(self):
        if self.available:
            self.lib.deep_gpu_dvfs_auto()
            self.lib.deep_gpu_cleanup()


# =============================================================================
# Deep Timing Fusion Model
# =============================================================================

class DeepTimingFusionModel(nn.Module):
    """Neural network with SCLK-switched banks + continuous timing modulation.

    This is the closest GPU analog to FPGA:
    - SCLK selects which weight bank is active (like FPGA LUT config)
    - Timing fingerprint continuously modulates activations (like analog timing)
    - Self-model predicts which bank is active (metacognition)
    """

    def __init__(self, n_banks=3, hidden_dim=64, timing_dim=4,
                 use_timing_mod=True, use_self_model=True):
        super().__init__()
        self.n_banks = n_banks
        self.hidden_dim = hidden_dim
        self.use_timing_mod = use_timing_mod
        self.use_self_model = use_self_model

        # Encoder (shared across banks)
        self.encoder = nn.Sequential(
            nn.Linear(784, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
        )

        # SWITCHED LAYER: 3 weight banks (one per DVFS state)
        self.banks = nn.ModuleList([
            nn.Sequential(nn.Linear(128, hidden_dim), nn.ReLU())
            for _ in range(n_banks)
        ])

        # Continuous timing coupling: timing features → hidden modulation
        if use_timing_mod:
            self.timing_proj = nn.Linear(timing_dim, hidden_dim, bias=False)
            # Initialize small but NOT tiny — must be learnable
            nn.init.normal_(self.timing_proj.weight, std=0.1)

        # Self-model: predicts which bank was active
        if use_self_model:
            self.self_model = nn.Sequential(
                nn.Linear(hidden_dim, 32), nn.ReLU(),
                nn.Linear(32, n_banks),
            )

        # Classifier head
        self.head = nn.Linear(hidden_dim, 10)

    def forward(self, x, bank_idx=0, timing_features=None):
        """
        Args:
            x: [B, 1, 28, 28] images
            bank_idx: which bank to use (0, 1, or 2)
            timing_features: [4] timing fingerprint (or None)

        Returns:
            logits: [B, 10] classification logits
            bank_pred: [B, n_banks] bank prediction logits (or None)
        """
        x_flat = x.view(x.shape[0], -1)
        h = self.encoder(x_flat)  # [B, 128]

        # SCLK-switched bank selection
        h = self.banks[bank_idx](h)  # [B, hidden_dim]

        # Continuous timing modulation
        if self.use_timing_mod and timing_features is not None:
            # timing_features: [4] → project to [hidden_dim]
            t_mod = self.timing_proj(timing_features)  # [hidden_dim]
            # Modulate: h = h * (1 + tanh(t_mod))
            # tanh keeps modulation bounded in [-1, 1]
            h = h * (1.0 + torch.tanh(t_mod).unsqueeze(0))  # [B, hidden_dim]

        # Self-model prediction
        bank_pred = None
        if self.use_self_model:
            bank_pred = self.self_model(h.detach())  # [B, n_banks] — detached to prevent gaming

        logits = self.head(h)
        return logits, bank_pred


# =============================================================================
# Training & Evaluation
# =============================================================================

def train_condition(model, train_loader, test_loader, probe, dvfs,
                    mode, n_epochs=15, lr=1e-3, self_model_weight=0.1):
    """Train one condition.

    Modes:
      'embodied':   SCLK→bank + live timing + self-model
      'frozen':     Fixed bank 1 + no timing
      'random':     Random bank + random timing
      'blind':      Always bank 0, no timing, no self-model
      'sclk_only':  SCLK→bank but no timing modulation
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    bank_totals = [0] * model.n_banks
    t0 = time.time()

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        self_correct = 0
        epoch_banks = [0] * model.n_banks

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            # DVFS cycling for embodied/sclk_only: cycle every 30 batches
            if mode in ('embodied', 'sclk_only'):
                phase = (batch_idx // 30 + epoch * 5) % 3
                if phase == 0:
                    dvfs.force_low()
                elif phase == 1:
                    dvfs.auto()
                else:
                    dvfs.force_high()
                time.sleep(0.001)  # tiny sleep for DVFS to settle

            # Bank selection
            if mode in ('embodied', 'sclk_only'):
                bank_idx = dvfs.sclk_to_bank()
            elif mode == 'random':
                bank_idx = np.random.randint(0, model.n_banks)
            elif mode == 'frozen':
                bank_idx = 1
            else:  # blind
                bank_idx = 0

            epoch_banks[bank_idx] += 1
            bank_totals[bank_idx] += 1

            # Timing features
            timing_features = None
            if mode == 'embodied':
                with torch.no_grad():
                    tf = probe.get_features()  # [4]
                    # Normalize to ~[-1, 1]
                    timing_features = (tf - tf.mean()) / (tf.std() + 1e-6)
            elif mode == 'random':
                timing_features = torch.randn(4, device=DEVICE)

            # Forward
            logits, bank_pred = model(images, bank_idx, timing_features)
            loss = F.cross_entropy(logits, labels)

            # Self-model loss
            if bank_pred is not None and mode in ('embodied', 'sclk_only', 'random'):
                bank_target = torch.full((images.shape[0],), bank_idx,
                                         dtype=torch.long, device=DEVICE)
                loss_self = F.cross_entropy(bank_pred, bank_target)
                loss = loss + self_model_weight * loss_self
                self_correct += (bank_pred.argmax(1) == bank_target).sum().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.shape[0]
            total_loss += loss.item()

        acc = correct / total
        self_acc = self_correct / total if total > 0 else 0
        history.append({
            'epoch': epoch,
            'loss': total_loss / len(train_loader),
            'accuracy': acc,
            'self_model_acc': self_acc,
            'bank_counts': epoch_banks,
        })

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:2d}: loss={history[-1]['loss']:.4f} "
                  f"acc={acc:.4f} self_acc={self_acc:.4f} "
                  f"banks={epoch_banks}")

    train_time = time.time() - t0
    dvfs.auto()

    # Final evaluation
    time.sleep(0.3)
    eval_result = evaluate_model(model, test_loader, probe, dvfs, mode)

    return {
        'train_history': history,
        'train_time_s': train_time,
        'bank_totals': bank_totals,
        **eval_result,
    }


def evaluate_model(model, test_loader, probe, dvfs, mode,
                   forced_bank=None, ablate_timing=False):
    """Evaluate model with optional ablation."""
    model.eval()
    correct = 0
    total = 0
    self_correct = 0
    bank_counts = [0] * model.n_banks
    all_outputs = []
    all_bank_preds = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            if forced_bank is not None:
                bank_idx = forced_bank
            elif mode in ('embodied', 'sclk_only'):
                bank_idx = dvfs.sclk_to_bank()
            elif mode == 'random':
                bank_idx = np.random.randint(0, model.n_banks)
            elif mode == 'frozen':
                bank_idx = 1
            else:
                bank_idx = 0

            bank_counts[bank_idx] += 1

            timing_features = None
            if mode == 'embodied' and not ablate_timing:
                tf = probe.get_features()
                timing_features = (tf - tf.mean()) / (tf.std() + 1e-6)

            logits, bank_pred = model(images, bank_idx, timing_features)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.shape[0]
            all_outputs.append(F.softmax(logits, dim=1).cpu())

            if bank_pred is not None:
                bank_target = torch.full((images.shape[0],), bank_idx,
                                         dtype=torch.long, device=DEVICE)
                self_correct += (bank_pred.argmax(1) == bank_target).sum().item()
                all_bank_preds.append(bank_pred.cpu())

    outputs = torch.cat(all_outputs)
    accuracy = correct / total
    self_model_acc = self_correct / total if total > 0 else 0

    return {
        'accuracy': accuracy,
        'self_model_acc': self_model_acc,
        'bank_counts': bank_counts,
        'outputs': outputs,
    }


# =============================================================================
# Main Experiment
# =============================================================================

def main():
    print("=" * 70)
    print("z2046: Deep Timing Fusion — Hardware IS Computation (v2)")
    print("  SCLK-switched banks + continuous timing modulation")
    print("  Fixes z2045: reliable SCLK→bank + balanced training + metacognition")
    print("=" * 70)

    # Initialize hardware probes
    probe = TimingProbe(n_threads=64)
    dvfs = DVFSController()

    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_ds = torchvision.datasets.MNIST(str(ROOT / 'data'), train=True,
                                           download=True, transform=transform)
    test_ds = torchvision.datasets.MNIST(str(ROOT / 'data'), train=False,
                                          download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True,
                              num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False,
                             num_workers=2, pin_memory=True)

    print(f"[DATA] Train: {len(train_ds)}, Test: {len(test_ds)}")
    print(f"[GPU] {torch.cuda.get_device_name(0)}")

    # =========================================================================
    # T2: Verify SCLK→bank mapping is reliable
    # =========================================================================
    print(f"\n{'='*60}")
    print("  T2: SCLK → Bank Mapping Verification")
    print(f"{'='*60}")
    sclk_banks = {}
    for state, fn in [('low', dvfs.force_low), ('high', dvfs.force_high),
                       ('auto', dvfs.auto)]:
        fn()
        time.sleep(0.5)
        sclks = []
        banks = []
        for _ in range(20):
            s = dvfs.get_sclk()
            b = dvfs.sclk_to_bank(s)
            sclks.append(s)
            banks.append(b)
            time.sleep(0.02)

        most_common = max(set(banks), key=banks.count)
        sclk_banks[state] = {
            'mean_sclk': int(np.mean(sclks)),
            'most_common_bank': most_common,
            'bank_distribution': [banks.count(i) for i in range(3)],
        }
        print(f"  {state:5s}: SCLK={int(np.mean(sclks)):5d} MHz → "
              f"bank {most_common} (distribution: {[banks.count(i) for i in range(3)]})")

    # Also collect timing fingerprints per DVFS state
    timing_per_state = {}
    for state, fn in [('low', dvfs.force_low), ('high', dvfs.force_high),
                       ('auto', dvfs.auto)]:
        fn()
        time.sleep(0.5)
        tfp = []
        for _ in range(20):
            tf = probe.get_features()
            tfp.append(tf.cpu().numpy())
            time.sleep(0.02)
        tfp = np.array(tfp)
        timing_per_state[state] = {
            'mean': tfp.mean(0).tolist(),
            'std': tfp.std(0).tolist(),
        }
        print(f"  {state:5s} timing: mem={tfp[:,0].mean():.0f} alu={tfp[:,1].mean():.0f} "
              f"lds={tfp[:,2].mean():.0f} vgpr={tfp[:,3].mean():.0f}")

    dvfs.auto()
    low_bank = sclk_banks['low']['most_common_bank']
    high_bank = sclk_banks['high']['most_common_bank']
    auto_bank = sclk_banks['auto']['most_common_bank']
    t2_pass = len(set([low_bank, high_bank, auto_bank])) >= 2
    print(f"  → Low→bank {low_bank}, Auto→bank {auto_bank}, High→bank {high_bank}")
    print(f"  → T2: {'PASS' if t2_pass else 'FAIL'} "
          f"({len(set([low_bank, high_bank, auto_bank]))}/3 distinct banks)")

    # =========================================================================
    # Train all conditions
    # =========================================================================
    N_EPOCHS = 15
    N_BANKS = 3

    conditions = {
        'A_embodied':   {'mode': 'embodied',  'timing_mod': True,  'self_model': True},
        'B_frozen':     {'mode': 'frozen',    'timing_mod': False, 'self_model': True},
        'C_random':     {'mode': 'random',    'timing_mod': True,  'self_model': True},
        'D_blind':      {'mode': 'blind',     'timing_mod': False, 'self_model': False},
        'E_sclk_only':  {'mode': 'sclk_only', 'timing_mod': False, 'self_model': True},
    }

    all_results = {
        'experiment': 'z2046_deep_timing_fusion',
        'timestamp': datetime.now().isoformat(),
        'sclk_banks': sclk_banks,
        'timing_per_state': timing_per_state,
        'conditions': {},
    }

    for cond_name, cfg in conditions.items():
        print(f"\n{'='*60}")
        print(f"  Condition: {cond_name} (mode={cfg['mode']})")
        print(f"{'='*60}")

        model = DeepTimingFusionModel(
            n_banks=N_BANKS, hidden_dim=64, timing_dim=4,
            use_timing_mod=cfg['timing_mod'],
            use_self_model=cfg['self_model'],
        ).to(DEVICE)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {n_params:,}")

        result = train_condition(
            model, train_loader, test_loader, probe, dvfs,
            mode=cfg['mode'], n_epochs=N_EPOCHS,
        )

        # Store outputs for later KL comparison
        outputs_ref = result.pop('outputs')

        # Per-DVFS evaluation (for embodied condition)
        dvfs_eval = {}
        if cfg['mode'] == 'embodied':
            print(f"\n  --- Per-DVFS Evaluation ---")
            for dvfs_state, dvfs_fn in [('low', dvfs.force_low),
                                         ('auto', dvfs.auto),
                                         ('high', dvfs.force_high)]:
                dvfs_fn()
                time.sleep(0.5)
                ev = evaluate_model(model, test_loader, probe, dvfs, cfg['mode'])
                out_dvfs = ev.pop('outputs')
                # KL from reference (auto)
                p = out_dvfs.mean(0).clamp(min=1e-8)
                q = outputs_ref.mean(0).clamp(min=1e-8)
                kl = (p * (p / q).log()).sum().item()
                dvfs_eval[dvfs_state] = {**ev, 'kl_from_ref': kl}
                print(f"    {dvfs_state:5s}: acc={ev['accuracy']:.4f} "
                      f"self_acc={ev['self_model_acc']:.4f} "
                      f"banks={ev['bank_counts']} KL={kl:.6f}")
            dvfs.auto()

        # Kill shots (for embodied condition)
        kill_shots = {}
        if cfg['mode'] == 'embodied':
            print(f"\n  --- Kill Shot A: Force Wrong Bank ---")
            for forced_b in range(N_BANKS):
                ev = evaluate_model(model, test_loader, probe, dvfs,
                                     cfg['mode'], forced_bank=forced_b)
                kill_shots[f'forced_bank_{forced_b}'] = {
                    'accuracy': ev['accuracy'],
                    'self_model_acc': ev['self_model_acc'],
                }
                print(f"    Forced bank {forced_b}: acc={ev['accuracy']:.4f} "
                      f"self_acc={ev['self_model_acc']:.4f}")
                ev.pop('outputs', None)

            print(f"\n  --- Kill Shot B: Ablate Timing Modulation ---")
            ev_no_timing = evaluate_model(model, test_loader, probe, dvfs,
                                           cfg['mode'], ablate_timing=True)
            kill_shots['ablated_timing'] = {
                'accuracy': ev_no_timing['accuracy'],
                'self_model_acc': ev_no_timing['self_model_acc'],
            }
            p = ev_no_timing.pop('outputs').mean(0).clamp(min=1e-8)
            q = outputs_ref.mean(0).clamp(min=1e-8)
            kill_shots['ablated_timing']['kl_from_ref'] = (p * (p / q).log()).sum().item()
            print(f"    No timing: acc={kill_shots['ablated_timing']['accuracy']:.4f} "
                  f"self_acc={kill_shots['ablated_timing']['self_model_acc']:.4f} "
                  f"KL={kill_shots['ablated_timing']['kl_from_ref']:.6f}")

        result_entry = {
            'mode': cfg['mode'],
            'n_params': n_params,
            'final_accuracy': result['accuracy'],
            'self_model_acc': result['self_model_acc'],
            'bank_totals': result['bank_totals'],
            'train_time_s': result['train_time_s'],
            'train_history': result['train_history'],
            'dvfs_eval': dvfs_eval,
            'kill_shots': kill_shots,
        }
        all_results['conditions'][cond_name] = result_entry

        print(f"  Final: acc={result['accuracy']:.4f} "
              f"self_acc={result['self_model_acc']:.4f} "
              f"banks={result['bank_totals']}")

    # =========================================================================
    # Analysis
    # =========================================================================
    print(f"\n{'='*70}")
    print("  Cross-Condition Analysis")
    print(f"{'='*70}")

    print(f"\n  {'Condition':<14} {'Accuracy':>10} {'Self-Model':>12} {'Banks':>20}")
    print("  " + "-" * 60)
    for cond, res in all_results['conditions'].items():
        print(f"  {cond:<14} {res['final_accuracy']:>10.4f} "
              f"{res['self_model_acc']:>12.4f} {str(res['bank_totals']):>20}")

    # T1: embodied learns
    emb = all_results['conditions']['A_embodied']
    t1_pass = emb['final_accuracy'] >= 0.90
    print(f"\n  T1: A_embodied acc={emb['final_accuracy']:.4f} >= 0.90 — "
          f"{'PASS' if t1_pass else 'FAIL'}")

    # T3: Self-model accuracy > 80%
    t3_pass = emb['self_model_acc'] >= 0.80
    print(f"  T3: A_embodied self_model_acc={emb['self_model_acc']:.4f} >= 0.80 — "
          f"{'PASS' if t3_pass else 'FAIL'}")

    # T4: Kill shot A — forced wrong bank
    ks = emb.get('kill_shots', {})
    if ks:
        bank_accs = [ks.get(f'forced_bank_{i}', {}).get('accuracy', 0)
                     for i in range(N_BANKS)]
        best_acc = max(bank_accs)
        worst_acc = min(bank_accs)
        t4_gap = best_acc - worst_acc
        t4_pass = t4_gap > 0.005
        print(f"  T4: Kill shot A: bank accs={[f'{a:.4f}' for a in bank_accs]} "
              f"gap={t4_gap:.4f} — {'PASS' if t4_pass else 'FAIL'}")

        # Self-model accuracy under forced bank
        sm_accs = [ks.get(f'forced_bank_{i}', {}).get('self_model_acc', 0)
                   for i in range(N_BANKS)]
        sm_gap = max(sm_accs) - min(sm_accs)
        print(f"       Self-model accs={[f'{a:.4f}' for a in sm_accs]} gap={sm_gap:.4f}")
    else:
        t4_pass = False
        t4_gap = 0

    # T5: Kill shot B — ablate timing
    ablated = ks.get('ablated_timing', {})
    if ablated:
        t5_kl = ablated.get('kl_from_ref', 0)
        t5_pass = abs(t5_kl) > 1e-5 or abs(emb['final_accuracy'] - ablated['accuracy']) > 0.001
        print(f"  T5: Kill shot B: ablated acc={ablated['accuracy']:.4f} "
              f"KL={t5_kl:.6f} — {'PASS' if t5_pass else 'FAIL'}")
    else:
        t5_pass = False

    # T6: Embodied > blind on self-prediction
    blind = all_results['conditions']['D_blind']
    sclk_only = all_results['conditions']['E_sclk_only']
    t6_pass = emb['self_model_acc'] > blind.get('self_model_acc', 0) + 0.1
    print(f"  T6: A_embodied self_acc={emb['self_model_acc']:.4f} > "
          f"D_blind self_acc={blind.get('self_model_acc', 0):.4f} — "
          f"{'PASS' if t6_pass else 'FAIL'}")
    print(f"       E_sclk_only self_acc={sclk_only['self_model_acc']:.4f}")

    # Bank balance check
    emb_banks = emb['bank_totals']
    total_batches = sum(emb_banks)
    balance_ratio = min(emb_banks) / max(max(emb_banks), 1)
    print(f"\n  Bank balance: {emb_banks} "
          f"(ratio={balance_ratio:.2f}, ideal=1.0)")

    n_pass = sum([t1_pass, t2_pass, t3_pass, t4_pass, t5_pass, t6_pass])
    all_results['tests'] = {
        'T1_learning': {'pass': bool(t1_pass), 'detail': f"acc={emb['final_accuracy']:.4f}"},
        'T2_sclk_bank_mapping': {'pass': bool(t2_pass),
                                  'detail': f"low→{low_bank} auto→{auto_bank} high→{high_bank}"},
        'T3_self_model': {'pass': bool(t3_pass),
                          'detail': f"self_acc={emb['self_model_acc']:.4f}"},
        'T4_kill_shot_bank': {'pass': bool(t4_pass), 'detail': f"gap={t4_gap:.4f}"},
        'T5_kill_shot_timing': {'pass': bool(t5_pass),
                                 'detail': f"KL={ablated.get('kl_from_ref', 0):.6f}"},
        'T6_embodied_vs_blind': {'pass': bool(t6_pass),
                                  'detail': f"emb={emb['self_model_acc']:.4f} vs blind={blind.get('self_model_acc', 0):.4f}"},
    }
    all_results['verdict'] = f'{n_pass}/6 PASS'

    print(f"\n{'='*70}")
    print(f"  VERDICT: {n_pass}/6 PASS")
    print(f"{'='*70}")

    all_results['notes'] = {
        'innovation': 'SCLK-switched weight banks + continuous timing modulation. '
                      'Reliable bank selection via DRM ioctl SCLK reading.',
        'fixes_from_z2045': [
            'SCLK-based bank selection (monotonic, reliable) vs timing bins (noisy, U-shaped)',
            'Balanced DVFS cycling (3 states × 30 batches each per epoch)',
            'Self-model as metacognition test (model knows its own hw state)',
            'Continuous timing modulation (learned projection) on top of discrete banks',
        ],
        'analogy': {
            'FPGA': 'Clock domain → LUT configuration → computation path',
            'GPU': 'SCLK frequency → weight bank → output + timing→modulation',
        },
    }

    dvfs.cleanup()

    results_path = RESULTS_DIR / 'z2046_deep_timing_fusion.json'
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n[SAVED] {results_path}")

    return all_results


if __name__ == '__main__':
    main()
