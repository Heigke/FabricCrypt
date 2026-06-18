#!/usr/bin/env python3
"""
Z1104: ACTUAL Kernel-Level Embodiment (Not PyTorch Simulation)

Critical finding: z910 claimed "kernel-level" but hip_available=False!
The HIP kernels were never actually loaded - it fell back to PyTorch.

This experiment FORCES kernel compilation and usage to test the REAL claim:
Does kernel-level modulation produce different dynamics than PyTorch-level?

Key differences between kernel-level and PyTorch-level:
1. TIMING: Kernel can read fresh telemetry per-operation, not per-batch
2. PRECISION: FP32 in kernel vs potentially mixed precision in PyTorch
3. ORDERING: Kernel operations are not subject to PyTorch graph optimization
4. ATOMICITY: Modulation happens within single kernel launch

This script:
1. Forcibly compiles interoceptive_kernels.hip with explicit error handling
2. Loads and verifies kernel functions via ctypes
3. Runs comparison: actual HIP kernel vs PyTorch equivalent
4. Tests if there's ANY measurable difference

Author: FEEL Research Team
Date: 2026-01-29
"""

import os
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '11.0.0'

# Ensure ROCm libraries are in path
rocm_lib = '/opt/rocm-7.1.1/lib'
amdgpu_lib = '/opt/amdgpu/lib/x86_64-linux-gnu'
current_path = os.environ.get('LD_LIBRARY_PATH', '')
if rocm_lib not in current_path:
    os.environ['LD_LIBRARY_PATH'] = f"{rocm_lib}:{amdgpu_lib}:{current_path}"

import sys
import json
import time
import ctypes
import subprocess
import math
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Tuple, Optional, Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.telemetry.sysfs_hwmon import SysfsHwmonTelemetry, EnergyMeter


# ============================================================================
# Kernel Build Infrastructure
# ============================================================================

NATIVE_DIR = Path(__file__).parent.parent / "src" / "native"
HIP_SOURCE = NATIVE_DIR / "interoceptive_kernels.hip"
HIP_LIB = NATIVE_DIR / "libinteroceptive.so"


def find_hipcc() -> Optional[Path]:
    """Find hipcc compiler."""
    candidates = [
        Path("/opt/rocm/bin/hipcc"),
        Path("/usr/bin/hipcc"),
    ]
    for c in candidates:
        if c.exists():
            return c

    # Try PATH
    try:
        result = subprocess.run(["which", "hipcc"], capture_output=True, text=True)
        if result.returncode == 0:
            return Path(result.stdout.strip())
    except:
        pass

    return None


def build_kernels(force: bool = False) -> Tuple[bool, str]:
    """
    Build HIP kernels with detailed error reporting.

    Returns: (success, message)
    """
    if not HIP_SOURCE.exists():
        return False, f"Source not found: {HIP_SOURCE}"

    if HIP_LIB.exists() and not force:
        if HIP_LIB.stat().st_mtime > HIP_SOURCE.stat().st_mtime:
            return True, "Using cached library"

    hipcc = find_hipcc()
    if not hipcc:
        return False, "hipcc not found (ROCm not installed?)"

    # Check for header file
    header = NATIVE_DIR / "deep_embodiment.hpp"
    if not header.exists():
        # Create minimal header
        header.write_text("""
#pragma once
namespace feel {
// Forward declarations for deep embodiment kernels
}
""")

    # Build command - target gfx1100 for RDNA3
    cmd = [
        str(hipcc),
        "-shared", "-fPIC",
        "-O3",
        "--offload-arch=gfx1100",
        "-I", str(NATIVE_DIR),
        "-o", str(HIP_LIB),
        str(HIP_SOURCE)
    ]

    print(f"[z1104] Building: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(NATIVE_DIR)
        )

        if result.returncode != 0:
            error = result.stderr or result.stdout
            # Common errors
            if "gfx1100" in error and "not supported" in error.lower():
                return False, f"GPU architecture gfx1100 not supported by this ROCm version"
            if "hip/hip_runtime.h" in error:
                return False, f"ROCm headers not found: {error[:200]}"
            return False, f"Compilation failed:\n{error[:500]}"

        if HIP_LIB.exists():
            return True, f"Built successfully: {HIP_LIB}"
        else:
            return False, "Build completed but .so not found"

    except subprocess.TimeoutExpired:
        return False, "Build timed out (120s)"
    except Exception as e:
        return False, f"Build exception: {e}"


def load_kernels() -> Tuple[Optional[ctypes.CDLL], str]:
    """Load kernel library with detailed error handling."""
    if not HIP_LIB.exists():
        return None, "Library not found (needs build)"

    try:
        lib = ctypes.CDLL(str(HIP_LIB))

        # Check for expected functions
        try:
            _ = lib.launch_energy_modulated_attention
            _ = lib.launch_body_token_encoder
            _ = lib.launch_homeostatic_gate
        except AttributeError as e:
            return None, f"Function not found: {e}"

        return lib, "Loaded successfully"

    except OSError as e:
        error_str = str(e)
        if "libamdhip64" in error_str:
            return None, "Missing libamdhip64 (ROCm runtime not found)"
        if "undefined symbol" in error_str:
            symbol = error_str.split("undefined symbol: ")[-1].split()[0]
            return None, f"Undefined symbol: {symbol}"
        return None, f"Load error: {error_str[:200]}"


# ============================================================================
# Kernel Function Wrappers
# ============================================================================

class HIPKernels:
    """Wrapper for HIP kernel functions."""

    def __init__(self, lib: ctypes.CDLL):
        self.lib = lib
        self._setup_signatures()

    def _setup_signatures(self):
        """Define ctypes function signatures."""
        # launch_energy_modulated_attention
        self.lib.launch_energy_modulated_attention.argtypes = [
            ctypes.c_void_p,  # query
            ctypes.c_void_p,  # key
            ctypes.c_void_p,  # value
            ctypes.c_void_p,  # output
            ctypes.c_void_p,  # body_state
            ctypes.c_float,   # power_setpoint
            ctypes.c_float,   # temp_setpoint
            ctypes.c_float,   # energy_mod_strength
            ctypes.c_float,   # homeostatic_gain
            ctypes.c_int,     # batch_size
            ctypes.c_int,     # seq_len
            ctypes.c_int,     # num_heads
            ctypes.c_int,     # head_dim
            ctypes.c_int,     # num_body_tokens
            ctypes.c_void_p,  # stream
        ]
        self.lib.launch_energy_modulated_attention.restype = ctypes.c_int

        # launch_homeostatic_gate
        self.lib.launch_homeostatic_gate.argtypes = [
            ctypes.c_void_p,  # hidden_states
            ctypes.c_void_p,  # body_state
            ctypes.c_float,   # power_setpoint
            ctypes.c_float,   # temp_setpoint
            ctypes.c_float,   # gain
            ctypes.c_int,     # batch_size
            ctypes.c_int,     # seq_len
            ctypes.c_int,     # hidden_dim
            ctypes.c_void_p,  # stream
        ]
        self.lib.launch_homeostatic_gate.restype = ctypes.c_int

    def energy_modulated_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        body_state: torch.Tensor,
        power_setpoint: float = 30.0,
        temp_setpoint: float = 50.0,
        energy_mod_strength: float = 0.3,
        homeostatic_gain: float = 0.1,
        num_body_tokens: int = 0
    ) -> torch.Tensor:
        """
        Call the HIP energy-modulated attention kernel.

        Args:
            Q, K, V: [batch, seq_len, head_dim] per head
            body_state: [batch, 12] telemetry
        Returns:
            output: [batch, seq_len, head_dim]
        """
        # Ensure contiguous float32 on CUDA
        Q = Q.contiguous().float().cuda()
        K = K.contiguous().float().cuda()
        V = V.contiguous().float().cuda()
        body_state = body_state.contiguous().float().cuda()

        batch_size, seq_len, head_dim = Q.shape
        output = torch.empty_like(Q)

        # Get stream
        stream = torch.cuda.current_stream().cuda_stream

        # Call kernel
        err = self.lib.launch_energy_modulated_attention(
            Q.data_ptr(),
            K.data_ptr(),
            V.data_ptr(),
            output.data_ptr(),
            body_state.data_ptr(),
            ctypes.c_float(power_setpoint),
            ctypes.c_float(temp_setpoint),
            ctypes.c_float(energy_mod_strength),
            ctypes.c_float(homeostatic_gain),
            batch_size,
            seq_len,
            1,  # num_heads (handled per-head)
            head_dim,
            num_body_tokens,
            stream,
        )

        torch.cuda.synchronize()

        if err != 0:
            raise RuntimeError(f"Kernel launch failed with error {err}")

        return output

    def homeostatic_gate(
        self,
        hidden_states: torch.Tensor,
        body_state: torch.Tensor,
        power_setpoint: float = 30.0,
        temp_setpoint: float = 50.0,
        gain: float = 0.1
    ) -> torch.Tensor:
        """
        Apply homeostatic gating to hidden states.

        Args:
            hidden_states: [batch, seq_len, hidden_dim]
            body_state: [batch, 12]
        Returns:
            gated: [batch, seq_len, hidden_dim]
        """
        hidden = hidden_states.contiguous().float().cuda()
        body = body_state.contiguous().float().cuda()

        batch_size, seq_len, hidden_dim = hidden.shape

        # In-place modification
        stream = torch.cuda.current_stream().cuda_stream

        err = self.lib.launch_homeostatic_gate(
            hidden.data_ptr(),
            body.data_ptr(),
            ctypes.c_float(power_setpoint),
            ctypes.c_float(temp_setpoint),
            ctypes.c_float(gain),
            batch_size,
            seq_len,
            hidden_dim,
            stream,
        )

        torch.cuda.synchronize()

        if err != 0:
            raise RuntimeError(f"Homeostatic gate kernel failed with error {err}")

        return hidden


# ============================================================================
# PyTorch Equivalent (for comparison)
# ============================================================================

def pytorch_energy_modulated_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    body_state: torch.Tensor,
    power_setpoint: float = 30.0,
    temp_setpoint: float = 50.0,
    energy_mod_strength: float = 0.3,
    homeostatic_gain: float = 0.1,
    num_body_tokens: int = 0
) -> torch.Tensor:
    """
    PyTorch implementation of energy-modulated attention.

    This should be mathematically equivalent to the HIP kernel.
    """
    batch_size, seq_len, head_dim = Q.shape

    # Compute modulation from body state
    # body_state layout: [temp_edge, temp_junction, _, power, ...]
    current_power = body_state[:, 3] * 50.0  # Denormalize
    current_temp = body_state[:, 0] * 100.0

    # Energy modulation
    power_ratio = current_power / max(power_setpoint, 1.0)
    energy_mod = 1.0 - energy_mod_strength * torch.clamp(power_ratio, max=2.0)
    energy_mod = torch.clamp(energy_mod, min=0.1)  # [batch]

    # Homeostatic gating
    power_dev = torch.abs(current_power - power_setpoint) / power_setpoint
    temp_dev = torch.abs(current_temp - temp_setpoint) / temp_setpoint
    total_dev = power_dev + temp_dev
    homeostatic_gate = 1.0 / (1.0 + homeostatic_gain * total_dev)

    # Combined modulation [batch]
    attention_scale = energy_mod * homeostatic_gate

    # Standard attention scores
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(head_dim)

    # Apply modulation per-key-position
    # Body tokens get INVERSE modulation (more attention when stressed)
    mod_mask = torch.ones(seq_len, device=Q.device)
    if num_body_tokens > 0:
        mod_mask[:num_body_tokens] = 2.0 - attention_scale.mean()
        mod_mask[num_body_tokens:] = attention_scale.mean()
    else:
        mod_mask[:] = attention_scale.mean()

    scores = scores * mod_mask.view(1, 1, -1)

    # Softmax and output
    attn = F.softmax(scores, dim=-1)
    output = torch.matmul(attn, V)

    return output


def pytorch_homeostatic_gate(
    hidden_states: torch.Tensor,
    body_state: torch.Tensor,
    power_setpoint: float = 30.0,
    temp_setpoint: float = 50.0,
    gain: float = 0.1
) -> torch.Tensor:
    """PyTorch implementation of homeostatic gating."""
    current_power = body_state[:, 3] * 50.0
    current_temp = body_state[:, 0] * 100.0

    power_dev = torch.abs(current_power - power_setpoint) / power_setpoint
    temp_dev = torch.abs(current_temp - temp_setpoint) / temp_setpoint

    gate = 1.0 / (1.0 + gain * (power_dev + temp_dev))

    # Apply gate [batch] to [batch, seq, hidden]
    return hidden_states * gate.view(-1, 1, 1)


# ============================================================================
# Test Infrastructure
# ============================================================================

def make_telemetry_tensor(telemetry: SysfsHwmonTelemetry, batch_size: int, device: torch.device) -> torch.Tensor:
    """Create body_state tensor from live telemetry."""
    sample = telemetry.read_sample()

    # Normalized 12-dim vector
    state = torch.tensor([
        sample.temp_edge_c / 100.0,
        sample.temp_edge_c / 100.0,  # junction ~= edge
        0.0,  # slope placeholder
        sample.power_w / 50.0,
        sample.power_w / 50.0,  # power_cap ~= power
        0.0, 0.0, 0.0,  # clocks (not available via sysfs_hwmon)
        0.0, 0.0, 0.0, 0.0  # padding
    ], dtype=torch.float32, device=device)

    return state.unsqueeze(0).expand(batch_size, -1)


def test_kernel_vs_pytorch(kernels: Optional[HIPKernels], device: torch.device) -> Dict:
    """
    Compare kernel output vs PyTorch output.

    Tests:
    1. Numerical equivalence (or difference)
    2. Timing difference
    3. Gradient flow (PyTorch only has grads)
    """
    results = {}

    batch_size = 4
    seq_len = 64
    head_dim = 64

    # Random inputs
    Q = torch.randn(batch_size, seq_len, head_dim, device=device)
    K = torch.randn(batch_size, seq_len, head_dim, device=device)
    V = torch.randn(batch_size, seq_len, head_dim, device=device)
    body_state = torch.rand(batch_size, 12, device=device)  # Normalized

    # PyTorch version
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(100):
        out_pytorch = pytorch_energy_modulated_attention(
            Q.clone(), K.clone(), V.clone(), body_state
        )
    torch.cuda.synchronize()
    pytorch_time = (time.perf_counter() - t0) / 100

    results['pytorch_ms'] = pytorch_time * 1000
    results['pytorch_mean'] = out_pytorch.mean().item()
    results['pytorch_std'] = out_pytorch.std().item()

    # HIP kernel version (if available)
    if kernels is not None:
        try:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(100):
                out_kernel = kernels.energy_modulated_attention(
                    Q.clone(), K.clone(), V.clone(), body_state
                )
            torch.cuda.synchronize()
            kernel_time = (time.perf_counter() - t0) / 100

            results['kernel_ms'] = kernel_time * 1000
            results['kernel_mean'] = out_kernel.mean().item()
            results['kernel_std'] = out_kernel.std().item()

            # Compare
            diff = (out_pytorch - out_kernel).abs()
            results['max_diff'] = diff.max().item()
            results['mean_diff'] = diff.mean().item()
            results['relative_diff'] = (diff / (out_pytorch.abs() + 1e-8)).mean().item()

            results['speedup'] = pytorch_time / kernel_time if kernel_time > 0 else 0

        except Exception as e:
            results['kernel_error'] = str(e)
    else:
        results['kernel_error'] = "Kernels not available"

    return results


# ============================================================================
# Training Comparison
# ============================================================================

class KernelAttentionLayer(nn.Module):
    """Attention layer using HIP kernel."""

    def __init__(self, hidden_dim: int, num_heads: int, kernels: HIPKernels):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.kernels = kernels

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor, body_state: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        Q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim)
        K = self.k_proj(x).view(B, T, self.num_heads, self.head_dim)
        V = self.v_proj(x).view(B, T, self.num_heads, self.head_dim)

        # Process each head through kernel
        outputs = []
        for h in range(self.num_heads):
            out_h = self.kernels.energy_modulated_attention(
                Q[:, :, h, :], K[:, :, h, :], V[:, :, h, :],
                body_state
            )
            outputs.append(out_h)

        out = torch.stack(outputs, dim=2).view(B, T, D)
        return self.out_proj(out)


class PyTorchAttentionLayer(nn.Module):
    """Attention layer using PyTorch (equivalent math)."""

    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor, body_state: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        Q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim)
        K = self.k_proj(x).view(B, T, self.num_heads, self.head_dim)
        V = self.v_proj(x).view(B, T, self.num_heads, self.head_dim)

        outputs = []
        for h in range(self.num_heads):
            out_h = pytorch_energy_modulated_attention(
                Q[:, :, h, :], K[:, :, h, :], V[:, :, h, :],
                body_state
            )
            outputs.append(out_h)

        out = torch.stack(outputs, dim=2).view(B, T, D)
        return self.out_proj(out)


class SimpleTransformer(nn.Module):
    """Simple transformer for comparison."""

    def __init__(self, vocab_size: int, hidden_dim: int, num_layers: int,
                 num_heads: int, use_kernel: bool, kernels: Optional[HIPKernels]):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, hidden_dim)
        self.pos_emb = nn.Embedding(512, hidden_dim)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            if use_kernel and kernels is not None:
                attn = KernelAttentionLayer(hidden_dim, num_heads, kernels)
            else:
                attn = PyTorchAttentionLayer(hidden_dim, num_heads)

            layer = nn.ModuleDict({
                'attn': attn,
                'ln1': nn.LayerNorm(hidden_dim),
                'ff': nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 4),
                    nn.GELU(),
                    nn.Linear(hidden_dim * 4, hidden_dim),
                ),
                'ln2': nn.LayerNorm(hidden_dim),
            })
            self.layers.append(layer)

        self.ln_out = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor, body_state: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        pos = torch.arange(T, device=x.device)
        h = self.token_emb(x) + self.pos_emb(pos)

        for layer in self.layers:
            h = h + layer['attn'](layer['ln1'](h), body_state)
            h = h + layer['ff'](layer['ln2'](h))

        return self.head(self.ln_out(h))


def run_training_comparison(kernels: Optional[HIPKernels], device: torch.device) -> Dict:
    """Compare training dynamics between kernel and PyTorch versions."""

    results = {
        'pytorch': {'losses': [], 'final_ppl': 0, 'time_s': 0},
        'kernel': {'losses': [], 'final_ppl': 0, 'time_s': 0},
    }

    vocab_size = 256
    hidden_dim = 128
    num_layers = 2
    num_heads = 4
    batch_size = 16
    seq_len = 64
    n_steps = 50

    # Generate random data
    data = torch.randint(0, vocab_size, (n_steps, batch_size, seq_len + 1), device=device)

    telemetry = SysfsHwmonTelemetry(sample_rate_hz=100)

    for mode in ['pytorch', 'kernel']:
        use_kernel = (mode == 'kernel' and kernels is not None)

        if mode == 'kernel' and kernels is None:
            results['kernel']['error'] = 'Kernels not available'
            continue

        model = SimpleTransformer(
            vocab_size, hidden_dim, num_layers, num_heads,
            use_kernel=use_kernel, kernels=kernels
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        for step in range(n_steps):
            batch = data[step]
            inp = batch[:, :-1]
            tgt = batch[:, 1:]

            body_state = make_telemetry_tensor(telemetry, batch_size, device)

            optimizer.zero_grad()
            logits = model(inp, body_state)
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), tgt.reshape(-1))
            loss.backward()
            optimizer.step()

            results[mode]['losses'].append(loss.item())

        torch.cuda.synchronize()
        results[mode]['time_s'] = time.perf_counter() - t0
        results[mode]['final_ppl'] = math.exp(results[mode]['losses'][-1])

    return results


# ============================================================================
# Main
# ============================================================================

def main():
    print("="*70)
    print("  Z1104: ACTUAL Kernel-Level Embodiment Test")
    print("="*70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Device] {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    results = {
        'experiment': 'z1104_actual_kernel_embodiment',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'device': str(device),
        'stages': {}
    }

    # Stage 1: Build kernels
    print("\n[Stage 1] Building HIP Kernels")
    print("-"*50)

    build_success, build_msg = build_kernels(force=True)
    print(f"  Build: {build_msg}")
    results['stages']['build'] = {
        'success': build_success,
        'message': build_msg
    }

    # Stage 2: Load kernels
    print("\n[Stage 2] Loading HIP Kernels")
    print("-"*50)

    kernels = None
    if build_success:
        lib, load_msg = load_kernels()
        print(f"  Load: {load_msg}")
        results['stages']['load'] = {
            'success': lib is not None,
            'message': load_msg
        }

        if lib is not None:
            kernels = HIPKernels(lib)
            print("  Kernel wrapper created successfully")
    else:
        results['stages']['load'] = {
            'success': False,
            'message': 'Build failed, cannot load'
        }

    # Stage 3: Numerical comparison
    print("\n[Stage 3] Numerical Comparison (Kernel vs PyTorch)")
    print("-"*50)

    numerical = test_kernel_vs_pytorch(kernels, device)
    results['stages']['numerical'] = numerical

    print(f"  PyTorch: {numerical['pytorch_ms']:.3f} ms, mean={numerical['pytorch_mean']:.4f}")
    if 'kernel_ms' in numerical:
        print(f"  Kernel:  {numerical['kernel_ms']:.3f} ms, mean={numerical['kernel_mean']:.4f}")
        print(f"  Max diff: {numerical['max_diff']:.6f}")
        print(f"  Speedup: {numerical.get('speedup', 0):.2f}x")
    else:
        print(f"  Kernel:  {numerical.get('kernel_error', 'N/A')}")

    # Stage 4: Training comparison
    print("\n[Stage 4] Training Comparison")
    print("-"*50)

    training = run_training_comparison(kernels, device)
    results['stages']['training'] = training

    print(f"  PyTorch: PPL={training['pytorch']['final_ppl']:.2f}, time={training['pytorch']['time_s']:.2f}s")
    if 'error' not in training['kernel']:
        print(f"  Kernel:  PPL={training['kernel']['final_ppl']:.2f}, time={training['kernel']['time_s']:.2f}s")
    else:
        print(f"  Kernel:  {training['kernel']['error']}")

    # Summary
    print("\n" + "="*70)
    print("  SUMMARY")
    print("="*70)

    kernel_available = kernels is not None
    numerical_match = numerical.get('max_diff', float('inf')) < 0.01

    claims = [
        {
            'claim': 'HIP kernels successfully built',
            'validated': build_success,
            'evidence': build_msg[:50]
        },
        {
            'claim': 'HIP kernels successfully loaded',
            'validated': kernel_available,
            'evidence': 'Wrapper created' if kernel_available else results['stages'].get('load', {}).get('message', 'N/A')
        },
        {
            'claim': 'Kernel output matches PyTorch (within 1%)',
            'validated': numerical_match if kernel_available else False,
            'evidence': f"max_diff={numerical.get('max_diff', 'N/A')}"
        },
        {
            'claim': 'Kernel is faster than PyTorch',
            'validated': numerical.get('speedup', 0) > 1.0,
            'evidence': f"speedup={numerical.get('speedup', 0):.2f}x"
        },
    ]

    results['claims'] = claims
    results['n_validated'] = sum(1 for c in claims if c['validated'])

    for c in claims:
        status = "PASS" if c['validated'] else "FAIL"
        print(f"  [{status}] {c['claim']}: {c['evidence']}")

    # Key insight
    print("\n[KEY INSIGHT]")
    if kernel_available and numerical_match:
        print("  Kernel and PyTorch produce IDENTICAL results (within numerical precision)")
        print("  This means: Kernel-level vs PyTorch-level doesn't change the MATH")
        print("  The benefit of kernels is SPEED and ATOMICITY, not different computation")
    elif kernel_available:
        print("  Kernel and PyTorch produce DIFFERENT results!")
        print("  This could indicate: Different precision, ordering, or implementation bug")
    else:
        print("  Kernels could not be loaded - cannot compare")
        print("  This is likely due to ROCm configuration or GPU architecture mismatch")

    # Save results
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    out_path = results_dir / "z1104_actual_kernel_embodiment.json"

    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: str(x) if isinstance(x, Path) else x)

    print(f"\n[Saved] {out_path}")


if __name__ == "__main__":
    main()
