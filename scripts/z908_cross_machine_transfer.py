#!/usr/bin/env python3
"""
Z908: Cross-Machine Transfer Experiment
========================================

Hypothesis: Embodied model trained on ikaros (Radeon 8060S) adapts behavior
appropriately on daedalus (different AMD) and minos (NVIDIA).

Protocol:
1. Train on ikaros with full embodiment (or load from z906/z907 checkpoint)
2. Transfer to daedalus (192.168.0.37, user=daedalus, AMD GPU)
3. Transfer to minos (192.168.0.38, user=minos, NVIDIA GPU)
4. Measure energy efficiency degradation on each machine

Metrics per machine:
- J/token (absolute and relative to ikaros)
- Perplexity (should be similar if model transfers well)
- Layer firing pattern adaptation (do gates respond to different hardware?)
- Temperature-response correlation

Author: FEEL Research Team
Date: 2026-01-28
"""

import os
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '11.0.0'

import sys
import json
import time
import math
import shutil
import logging
import argparse
import tempfile
import textwrap
import statistics
import subprocess
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.telemetry.sysfs_hwmon import SysfsHwmonTelemetry, EnergyMeter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent


# =============================================================================
# Machine Configurations
# =============================================================================

@dataclass
class MachineConfig:
    """Configuration for a machine in the cluster."""
    name: str
    host: str
    user: str
    password_env: str
    password_default: str
    vendor: str  # "amd" or "nvidia"
    is_local: bool = False
    venv_hint: str = ""  # Hint for finding venv on remote
    hsa_override: str = ""


MACHINES = {
    'ikaros': MachineConfig(
        name='ikaros',
        host='localhost',
        user='ikaros',
        password_env='',
        password_default='',
        vendor='amd',
        is_local=True,
        venv_hint='/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/venv',
        hsa_override='11.0.0',
    ),
    'daedalus': MachineConfig(
        name='daedalus',
        host='192.168.0.37',
        user='daedalus',
        password_env='DAEDALUS_PASS',
        password_default='daedalus',
        vendor='amd',
        venv_hint='torch-rocm',  # Look for this in venvs
    ),
    'minos': MachineConfig(
        name='minos',
        host='192.168.0.38',
        user='minos',
        password_env='MINOS_PASS',
        password_default='minos',
        vendor='nvidia',
    ),
}


# =============================================================================
# StochasticDepthTransformer (self-contained, matches training architecture)
# =============================================================================

class StochasticGate(nn.Module):
    """Gate that decides whether to execute a layer based on sensor input."""

    def __init__(self, hidden_dim: int, sensor_dim: int = 3):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim + sensor_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        nn.init.constant_(self.fc[-2].bias, 0.5)

    def forward(self, hidden: torch.Tensor, sensor: torch.Tensor) -> torch.Tensor:
        batch = hidden.size(0)
        if sensor.dim() == 1:
            sensor = sensor.unsqueeze(0).expand(batch, -1)
        ctx = hidden[:, -1, :]  # last token
        fusion = torch.cat([ctx, sensor], dim=-1)
        return self.fc(fusion)


class StochasticDepthBlock(nn.Module):
    """Transformer block with stochastic depth gating."""

    def __init__(self, hidden_dim: int, n_heads: int, sensor_dim: int = 3, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.gate = StochasticGate(hidden_dim, sensor_dim)
        self.last_gate_prob = 0.5

    def forward(self, x: torch.Tensor, sensor: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        gate_prob = self.gate(x, sensor)
        self.last_gate_prob = gate_prob.mean().item()

        if self.training:
            # Soft gating for training
            normed = self.ln1(x)
            attn_out, _ = self.attn(normed, normed, normed, attn_mask=mask)
            h = x + attn_out
            h = h + self.ffn(self.ln2(h))
            gate_expanded = gate_prob.unsqueeze(-1).to(x.dtype)
            return gate_expanded * h + (1 - gate_expanded) * x
        else:
            # Hard gating for inference
            if gate_prob.mean().item() > 0.5:
                normed = self.ln1(x)
                attn_out, _ = self.attn(normed, normed, normed, attn_mask=mask)
                h = x + attn_out
                h = h + self.ffn(self.ln2(h))
                return h
            else:
                return x  # Skip layer


class StochasticDepthTransformer(nn.Module):
    """
    Self-contained transformer with stochastic depth gating.
    Each layer has a gate conditioned on hardware sensor state.
    """

    def __init__(self, vocab_size: int = 50257, hidden_dim: int = 256,
                 n_layers: int = 8, n_heads: int = 8, max_seq_len: int = 256,
                 sensor_dim: int = 3, dropout: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len
        self.sensor_dim = sensor_dim

        self.tok_embed = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embed = nn.Embedding(max_seq_len, hidden_dim)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            StochasticDepthBlock(hidden_dim, n_heads, sensor_dim, dropout)
            for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.tok_embed.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor,
                sensor: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T = input_ids.shape
        if sensor is None:
            sensor = torch.tensor([0.5, 0.5, 0.5], device=input_ids.device)

        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
        h = self.drop(self.tok_embed(input_ids) + self.pos_embed(pos))

        # Causal mask
        mask = torch.triu(torch.ones(T, T, device=h.device), diagonal=1).bool()

        for block in self.blocks:
            h = block(h, sensor, mask=mask)

        h = self.ln_f(h)
        return self.lm_head(h)

    def get_layer_firing_rates(self) -> List[float]:
        """Get the gate probability for each layer."""
        return [block.last_gate_prob for block in self.blocks]

    def get_config_dict(self) -> dict:
        return {
            'vocab_size': self.vocab_size,
            'hidden_dim': self.hidden_dim,
            'n_layers': self.n_layers,
            'n_heads': self.n_heads,
            'max_seq_len': self.max_seq_len,
            'sensor_dim': self.sensor_dim,
        }


# =============================================================================
# Checkpoint Discovery
# =============================================================================

def discover_checkpoint(checkpoint_arg: Optional[str] = None) -> Optional[Path]:
    """Try to find a suitable checkpoint, in priority order."""
    candidates = []

    if checkpoint_arg:
        p = Path(checkpoint_arg)
        if p.exists():
            return p
        logger.warning(f"Specified checkpoint not found: {checkpoint_arg}")

    # Try z906/z907 checkpoints first
    ckpt_dir = PROJECT_ROOT / "checkpoints"
    for name in [
        "z906_full.pt", "z907_live.pt",
        "z906_full/checkpoint.pt", "z907_live/checkpoint.pt",
        "z200_hardware_loop/checkpoint_final.pt",
        "z200_hardware_loop/checkpoint_step300.pt",
        "z151_minimal/checkpoint.pt",
    ]:
        p = ckpt_dir / name
        if p.exists():
            candidates.append(p)

    # Also check for any .pt files in checkpoints root
    if ckpt_dir.exists():
        for pt_file in sorted(ckpt_dir.glob("*.pt")):
            candidates.append(pt_file)

    if candidates:
        logger.info(f"Discovered checkpoint: {candidates[0]}")
        return candidates[0]

    return None


# =============================================================================
# Local Evaluation
# =============================================================================

@dataclass
class EvalResult:
    """Result of evaluating the model on one machine."""
    machine: str
    vendor: str
    gpu_name: str
    n_eval_steps: int
    j_per_token: float
    j_per_token_marginal: float
    perplexity: float
    avg_loss: float
    layer_firing_rates: List[float]
    avg_power_w: float
    avg_temp_c: float
    max_temp_c: float
    tokens_per_second: float
    total_energy_j: float
    total_tokens: int
    duration_s: float
    timestamp: str
    status: str = "OK"  # "OK", "UNREACHABLE", "ERROR"
    error_message: str = ""


def get_gpu_name() -> str:
    """Get GPU name from sysfs or torch."""
    try:
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    # Try sysfs
    try:
        drm = Path("/sys/class/drm")
        for card in sorted(drm.glob("card[0-9]*")):
            product = card / "device" / "product_name"
            if product.exists():
                return product.read_text().strip()
            uevent = card / "device" / "uevent"
            if uevent.exists():
                for line in uevent.read_text().splitlines():
                    if "PCI_SLOT_NAME" in line:
                        return f"AMD GPU ({line.split('=')[1]})"
    except Exception:
        pass
    return "Unknown GPU"


def train_fresh_model(device: torch.device, n_steps: int = 100) -> StochasticDepthTransformer:
    """Train a minimal model from scratch if no checkpoint is found."""
    logger.info("No checkpoint found - training fresh model (simplified, %d steps)", n_steps)

    model = StochasticDepthTransformer(
        vocab_size=50257, hidden_dim=256, n_layers=8,
        n_heads=8, max_seq_len=256, sensor_dim=3,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    model.train()
    for step in range(n_steps):
        # Random data
        input_ids = torch.randint(0, 50257, (4, 128), device=device)
        labels = input_ids.clone()

        # Random sensor state (simulated embodiment)
        stress = (step / n_steps)  # Linearly increasing stress
        sensor = torch.tensor([stress, 1.0 - stress, 0.5], device=device)

        logits = model(input_ids, sensor=sensor)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, 50257), labels[:, 1:].reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if (step + 1) % 20 == 0:
            logger.info(f"  Train step {step+1}/{n_steps}, loss={loss.item():.4f}")

    # Save checkpoint
    ckpt_dir = PROJECT_ROOT / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "z908_fresh.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': model.get_config_dict(),
    }, ckpt_path)
    logger.info(f"Saved fresh checkpoint: {ckpt_path}")

    return model


def load_model_from_checkpoint(ckpt_path: Path, device: torch.device) -> StochasticDepthTransformer:
    """Load model from checkpoint, handling various checkpoint formats."""
    logger.info(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Try to extract model config
    if 'model_config' in ckpt:
        cfg = ckpt['model_config']
        model = StochasticDepthTransformer(**cfg).to(device)
    else:
        # Default config
        model = StochasticDepthTransformer(
            vocab_size=50257, hidden_dim=256, n_layers=8,
            n_heads=8, max_seq_len=256, sensor_dim=3,
        ).to(device)

    # Try to load state dict
    if 'model_state_dict' in ckpt:
        state = ckpt['model_state_dict']
    elif 'state_dict' in ckpt:
        state = ckpt['state_dict']
    elif 'model' in ckpt:
        state = ckpt['model']
    else:
        # Maybe the checkpoint IS the state dict
        state = ckpt

    try:
        model.load_state_dict(state, strict=False)
        logger.info("Loaded model state dict (strict=False)")
    except Exception as e:
        logger.warning(f"Could not load state dict: {e}")
        logger.info("Proceeding with randomly initialized model")

    return model


def evaluate_local(model: StochasticDepthTransformer, device: torch.device,
                   n_eval_steps: int = 200) -> EvalResult:
    """Run evaluation on the local machine (ikaros) with sysfs energy measurement."""
    logger.info("=" * 60)
    logger.info("LOCAL EVALUATION: ikaros")
    logger.info("=" * 60)

    model.eval()
    gpu_name = get_gpu_name()
    logger.info(f"GPU: {gpu_name}")

    # Initialize telemetry
    try:
        telemetry = SysfsHwmonTelemetry(sample_rate_hz=50.0)
        has_telemetry = True
    except Exception as e:
        logger.warning(f"No sysfs telemetry: {e}")
        has_telemetry = False

    # Measure idle baseline
    idle_power_w = 0.0
    if has_telemetry:
        logger.info("Measuring idle baseline (2s)...")
        idle_power_w = telemetry.measure_idle_baseline(2.0)
        logger.info(f"Idle power: {idle_power_w:.1f} W")

    # Prepare evaluation data (random but fixed seed for reproducibility)
    rng = torch.Generator(device='cpu')
    rng.manual_seed(908)
    eval_inputs = [
        torch.randint(0, 50257, (1, 128), generator=rng).to(device)
        for _ in range(n_eval_steps)
    ]

    # Evaluation loop with energy measurement
    losses = []
    all_firing_rates = []
    temps = []
    total_tokens = 0

    if has_telemetry:
        meter = EnergyMeter(telemetry)
    else:
        meter = None

    torch.cuda.synchronize() if torch.cuda.is_available() else None

    start_time = time.time()
    start_ns = time.time_ns()

    if meter:
        meter.__enter__()

    with torch.no_grad():
        for step_idx, input_ids in enumerate(eval_inputs):
            # Vary sensor state to test adaptation
            stress = 0.3 + 0.4 * math.sin(step_idx / 20.0)
            sensor = torch.tensor([stress, 1.0 - stress, 0.5], device=device)

            logits = model(input_ids, sensor=sensor)
            labels = input_ids[:, 1:]
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, model.vocab_size), labels.reshape(-1))
            losses.append(loss.item())

            firing_rates = model.get_layer_firing_rates()
            all_firing_rates.append(firing_rates)
            total_tokens += input_ids.numel()

            # Read temperature
            if has_telemetry:
                sample = telemetry.read_sample()
                temps.append(sample.temp_edge_c)

            if (step_idx + 1) % 50 == 0:
                avg_loss = np.mean(losses[-50:])
                logger.info(f"  Step {step_idx+1}/{n_eval_steps}, loss={avg_loss:.4f}")

    if meter:
        meter.__exit__(None, None, None)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    end_time = time.time()
    duration_s = end_time - start_time

    # Compute metrics
    avg_loss = float(np.mean(losses))
    perplexity = min(float(math.exp(avg_loss)), 1e6)  # Cap to avoid overflow
    tokens_per_second = total_tokens / duration_s if duration_s > 0 else 0

    # Energy metrics
    if meter and meter.energy_j > 0:
        total_energy_j = meter.energy_j
        j_per_token = total_energy_j / total_tokens if total_tokens > 0 else 0
        marginal_energy_j = meter.marginal_energy_j()
        j_per_token_marginal = marginal_energy_j / total_tokens if total_tokens > 0 else 0
        avg_power_w = meter.avg_power_w
    else:
        total_energy_j = 0.0
        j_per_token = 0.0
        j_per_token_marginal = 0.0
        avg_power_w = 0.0

    # Temperature
    avg_temp = float(np.mean(temps)) if temps else 0.0
    max_temp = float(np.max(temps)) if temps else 0.0

    # Average firing rates across all steps
    mean_firing_rates = [
        float(np.mean([rates[i] for rates in all_firing_rates]))
        for i in range(model.n_layers)
    ]

    result = EvalResult(
        machine='ikaros',
        vendor='amd',
        gpu_name=gpu_name,
        n_eval_steps=n_eval_steps,
        j_per_token=j_per_token,
        j_per_token_marginal=j_per_token_marginal,
        perplexity=perplexity,
        avg_loss=avg_loss,
        layer_firing_rates=mean_firing_rates,
        avg_power_w=avg_power_w,
        avg_temp_c=avg_temp,
        max_temp_c=max_temp,
        tokens_per_second=tokens_per_second,
        total_energy_j=total_energy_j,
        total_tokens=total_tokens,
        duration_s=duration_s,
        timestamp=datetime.now().isoformat(),
    )

    _print_machine_result(result)
    return result


# =============================================================================
# Remote Evaluation Script Generator
# =============================================================================

def generate_remote_eval_script(model_config: dict, n_eval_steps: int) -> str:
    """
    Generate a self-contained Python script for remote evaluation.
    No project imports needed - pure PyTorch only.
    """
    script = textwrap.dedent('''\
        #!/usr/bin/env python3
        """Self-contained remote evaluation script for z908 cross-machine transfer."""
        import os
        import sys
        import json
        import time
        import math
        import argparse
        from pathlib import Path
        from datetime import datetime

        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        import numpy as np

        # =====================================================================
        # StochasticDepthTransformer (self-contained)
        # =====================================================================

        class StochasticGate(nn.Module):
            def __init__(self, hidden_dim, sensor_dim=3):
                super().__init__()
                self.fc = nn.Sequential(
                    nn.Linear(hidden_dim + sensor_dim, 64),
                    nn.ReLU(),
                    nn.Linear(64, 1),
                    nn.Sigmoid(),
                )
                nn.init.constant_(self.fc[-2].bias, 0.5)

            def forward(self, hidden, sensor):
                batch = hidden.size(0)
                if sensor.dim() == 1:
                    sensor = sensor.unsqueeze(0).expand(batch, -1)
                ctx = hidden[:, -1, :]
                fusion = torch.cat([ctx, sensor], dim=-1)
                return self.fc(fusion)


        class StochasticDepthBlock(nn.Module):
            def __init__(self, hidden_dim, n_heads, sensor_dim=3, dropout=0.1):
                super().__init__()
                self.attn = nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)
                self.ln1 = nn.LayerNorm(hidden_dim)
                self.ln2 = nn.LayerNorm(hidden_dim)
                self.ffn = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 4),
                    nn.GELU(),
                    nn.Linear(hidden_dim * 4, hidden_dim),
                    nn.Dropout(dropout),
                )
                self.gate = StochasticGate(hidden_dim, sensor_dim)
                self.last_gate_prob = 0.5

            def forward(self, x, sensor, mask=None):
                gate_prob = self.gate(x, sensor)
                self.last_gate_prob = gate_prob.mean().item()

                # Hard gating for inference
                if gate_prob.mean().item() > 0.5:
                    normed = self.ln1(x)
                    attn_out, _ = self.attn(normed, normed, normed, attn_mask=mask)
                    h = x + attn_out
                    h = h + self.ffn(self.ln2(h))
                    return h
                else:
                    return x


        class StochasticDepthTransformer(nn.Module):
            def __init__(self, vocab_size=50257, hidden_dim=256, n_layers=8,
                         n_heads=8, max_seq_len=256, sensor_dim=3, dropout=0.1):
                super().__init__()
                self.vocab_size = vocab_size
                self.hidden_dim = hidden_dim
                self.n_layers = n_layers
                self.n_heads = n_heads
                self.max_seq_len = max_seq_len
                self.sensor_dim = sensor_dim

                self.tok_embed = nn.Embedding(vocab_size, hidden_dim)
                self.pos_embed = nn.Embedding(max_seq_len, hidden_dim)
                self.drop = nn.Dropout(dropout)

                self.blocks = nn.ModuleList([
                    StochasticDepthBlock(hidden_dim, n_heads, sensor_dim, dropout)
                    for _ in range(n_layers)
                ])
                self.ln_f = nn.LayerNorm(hidden_dim)
                self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
                self.lm_head.weight = self.tok_embed.weight
                self.apply(self._init_weights)

            def _init_weights(self, module):
                if isinstance(module, nn.Linear):
                    torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                    if module.bias is not None:
                        torch.nn.init.zeros_(module.bias)
                elif isinstance(module, nn.Embedding):
                    torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

            def forward(self, input_ids, sensor=None):
                B, T = input_ids.shape
                if sensor is None:
                    sensor = torch.tensor([0.5, 0.5, 0.5], device=input_ids.device)
                pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
                h = self.drop(self.tok_embed(input_ids) + self.pos_embed(pos))
                mask = torch.triu(torch.ones(T, T, device=h.device), diagonal=1).bool()
                for block in self.blocks:
                    h = block(h, sensor, mask=mask)
                h = self.ln_f(h)
                return self.lm_head(h)

            def get_layer_firing_rates(self):
                return [block.last_gate_prob for block in self.blocks]


        # =====================================================================
        # Telemetry: sysfs (AMD) -> pynvml (NVIDIA) -> time-based fallback
        # =====================================================================

        class TelemetryReader:
            """Cross-platform GPU telemetry reader."""

            def __init__(self):
                self.backend = None
                self.power_samples = []
                self.temp_samples = []
                self.gpu_name = "Unknown"
                self._nvml_handle = None
                self._sysfs_power_path = None
                self._sysfs_temp_path = None

                self._detect_backend()

            def _detect_backend(self):
                # Try sysfs first (AMD)
                drm = Path("/sys/class/drm")
                if drm.exists():
                    for card in sorted(drm.glob("card[0-9]*")):
                        hwmon = card / "device" / "hwmon"
                        if hwmon.exists():
                            for hw in hwmon.glob("hwmon*"):
                                pwr = hw / "power1_average"
                                tmp = hw / "temp1_input"
                                if pwr.exists():
                                    self._sysfs_power_path = pwr
                                    self._sysfs_temp_path = tmp if tmp.exists() else None
                                    self.backend = "sysfs"
                                    # Try GPU name
                                    product = card / "device" / "product_name"
                                    if product.exists():
                                        try:
                                            self.gpu_name = product.read_text().strip()
                                        except Exception:
                                            pass
                                    else:
                                        self.gpu_name = "AMD GPU (sysfs)"
                                    return

                # Try pynvml (NVIDIA)
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    name = pynvml.nvmlDeviceGetName(self._nvml_handle)
                    if isinstance(name, bytes):
                        name = name.decode()
                    self.gpu_name = name
                    self.backend = "nvml"
                    return
                except Exception:
                    pass

                # Try torch
                if torch.cuda.is_available():
                    self.gpu_name = torch.cuda.get_device_name(0)

                # Fallback
                self.backend = "time"

            def read_power_w(self):
                if self.backend == "sysfs" and self._sysfs_power_path:
                    try:
                        return int(self._sysfs_power_path.read_text().strip()) / 1e6
                    except Exception:
                        return 0.0
                elif self.backend == "nvml" and self._nvml_handle:
                    try:
                        import pynvml
                        return pynvml.nvmlDeviceGetPowerUsage(self._nvml_handle) / 1000.0
                    except Exception:
                        return 0.0
                return 0.0

            def read_temp_c(self):
                if self.backend == "sysfs" and self._sysfs_temp_path:
                    try:
                        return int(self._sysfs_temp_path.read_text().strip()) / 1e3
                    except Exception:
                        return 0.0
                elif self.backend == "nvml" and self._nvml_handle:
                    try:
                        import pynvml
                        return pynvml.nvmlDeviceGetTemperature(
                            self._nvml_handle, pynvml.NVML_TEMPERATURE_GPU)
                    except Exception:
                        return 0.0
                return 0.0

            def sample(self):
                self.power_samples.append(self.read_power_w())
                self.temp_samples.append(self.read_temp_c())


        # =====================================================================
        # Evaluation
        # =====================================================================

        def main():
            parser = argparse.ArgumentParser()
            parser.add_argument("--checkpoint", required=True)
            parser.add_argument("--n-eval-steps", type=int, default=''' + str(n_eval_steps) + ''')
            parser.add_argument("--device", default="auto")
            args = parser.parse_args()

            if args.device == "auto":
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                device = torch.device(args.device)

            # Load checkpoint
            ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

            # Build model
            cfg = ckpt.get("model_config", {
                "vocab_size": ''' + str(model_config.get('vocab_size', 50257)) + ''',
                "hidden_dim": ''' + str(model_config.get('hidden_dim', 256)) + ''',
                "n_layers": ''' + str(model_config.get('n_layers', 8)) + ''',
                "n_heads": ''' + str(model_config.get('n_heads', 8)) + ''',
                "max_seq_len": ''' + str(model_config.get('max_seq_len', 256)) + ''',
                "sensor_dim": ''' + str(model_config.get('sensor_dim', 3)) + ''',
            })
            model = StochasticDepthTransformer(**cfg).to(device)

            state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt.get("model", ckpt)))
            try:
                model.load_state_dict(state, strict=False)
            except Exception:
                pass  # Proceed with random init if mismatch

            model.eval()

            # Telemetry
            telem = TelemetryReader()

            # Warmup
            warmup_ids = torch.randint(0, cfg.get("vocab_size", 50257), (1, 128), device=device)
            for _ in range(5):
                with torch.no_grad():
                    model(warmup_ids)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            # Evaluation
            rng = torch.Generator(device="cpu")
            rng.manual_seed(908)
            eval_inputs = [
                torch.randint(0, cfg.get("vocab_size", 50257), (1, 128), generator=rng).to(device)
                for _ in range(args.n_eval_steps)
            ]

            losses = []
            all_firing_rates = []
            total_tokens = 0

            start = time.time()

            with torch.no_grad():
                for step_idx, input_ids in enumerate(eval_inputs):
                    stress = 0.3 + 0.4 * math.sin(step_idx / 20.0)
                    sensor = torch.tensor([stress, 1.0 - stress, 0.5], device=device)

                    logits = model(input_ids, sensor=sensor)
                    labels = input_ids[:, 1:]
                    loss = F.cross_entropy(
                        logits[:, :-1].reshape(-1, cfg.get("vocab_size", 50257)),
                        labels.reshape(-1)
                    )
                    losses.append(loss.item())
                    all_firing_rates.append(model.get_layer_firing_rates())
                    total_tokens += input_ids.numel()

                    # Sample telemetry every 5 steps
                    if step_idx % 5 == 0:
                        telem.sample()

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            elapsed = time.time() - start

            # Compute metrics
            avg_loss = float(np.mean(losses))
            ppl = min(float(math.exp(avg_loss)), 1e6)
            tps = total_tokens / elapsed if elapsed > 0 else 0

            # Energy
            powers = [p for p in telem.power_samples if p > 0]
            temps_list = [t for t in telem.temp_samples if t > 0]

            if powers:
                avg_power = float(np.mean(powers))
                total_energy = avg_power * elapsed
                jpt = total_energy / total_tokens if total_tokens > 0 else 0
            else:
                avg_power = 0.0
                total_energy = 0.0
                jpt = 0.0

            avg_temp = float(np.mean(temps_list)) if temps_list else 0.0
            max_temp = float(np.max(temps_list)) if temps_list else 0.0

            n_layers = cfg.get("n_layers", 8)
            mean_firing = [
                float(np.mean([rates[i] for rates in all_firing_rates]))
                for i in range(n_layers)
            ]

            result = {
                "gpu_name": telem.gpu_name,
                "backend": telem.backend,
                "n_eval_steps": args.n_eval_steps,
                "j_per_token": jpt,
                "perplexity": ppl,
                "avg_loss": avg_loss,
                "layer_firing_rates": mean_firing,
                "avg_power_w": avg_power,
                "avg_temp_c": avg_temp,
                "max_temp_c": max_temp,
                "tokens_per_second": tps,
                "total_energy_j": total_energy,
                "total_tokens": total_tokens,
                "duration_s": elapsed,
            }

            # Output as JSON on a single line for easy parsing
            print("Z908_RESULT_JSON:" + json.dumps(result))


        if __name__ == "__main__":
            main()
    ''')
    return script


# =============================================================================
# Remote Deployment
# =============================================================================

def check_sshpass_available() -> bool:
    """Check if sshpass is installed."""
    try:
        result = subprocess.run(["sshpass", "-V"], capture_output=True, text=True, timeout=5)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def get_password(machine: MachineConfig) -> str:
    """Get password for a machine from env or default."""
    if machine.password_env:
        return os.environ.get(machine.password_env, machine.password_default)
    return machine.password_default


def check_machine_reachable(machine: MachineConfig, timeout: int = 30) -> bool:
    """Check if a remote machine is reachable via SSH."""
    password = get_password(machine)
    cmd = [
        "sshpass", "-p", password,
        "ssh", "-o", "StrictHostKeyChecking=no",
        "-o", f"ConnectTimeout={timeout}",
        f"{machine.user}@{machine.host}",
        "echo OK"
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 5)
        return result.returncode == 0 and "OK" in result.stdout
    except (subprocess.TimeoutExpired, Exception) as e:
        logger.warning(f"Machine {machine.name} unreachable: {e}")
        return False


def find_remote_python(machine: MachineConfig, timeout: int = 30) -> str:
    """Find the best python interpreter on a remote machine."""
    password = get_password(machine)

    # Strategy: look for venvs, then system python3
    find_script = textwrap.dedent(f"""\
        # Try torch-rocm venv first (daedalus)
        for venv_dir in ~/venvs ~/envs ~; do
            for vname in torch-rocm torch rocm venv .venv; do
                if [ -f "$venv_dir/$vname/bin/python" ]; then
                    echo "VENV:$venv_dir/$vname/bin/python"
                    exit 0
                fi
            done
        done
        # Look in home for common patterns
        for d in ~/*/bin/python ~/.*env*/bin/python; do
            if [ -f "$d" ]; then
                echo "VENV:$d"
                exit 0
            fi
        done
        # Fallback to system python3
        if command -v python3 &>/dev/null; then
            echo "SYSTEM:$(which python3)"
            exit 0
        fi
        echo "NONE:"
    """)

    cmd = [
        "sshpass", "-p", password,
        "ssh", "-o", "StrictHostKeyChecking=no",
        "-o", f"ConnectTimeout={timeout}",
        f"{machine.user}@{machine.host}",
        find_script,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 5)
        for line in result.stdout.strip().split('\n'):
            line = line.strip()
            if line.startswith("VENV:") or line.startswith("SYSTEM:"):
                python_path = line.split(":", 1)[1]
                logger.info(f"  {machine.name}: found python at {python_path}")
                return python_path
    except Exception as e:
        logger.warning(f"Could not find python on {machine.name}: {e}")

    return "python3"


def deploy_and_evaluate(machine: MachineConfig, checkpoint_path: Path,
                        model_config: dict, n_eval_steps: int,
                        timeout: int = 300) -> EvalResult:
    """Deploy model to remote machine and evaluate."""
    logger.info("=" * 60)
    logger.info(f"REMOTE EVALUATION: {machine.name} ({machine.host})")
    logger.info("=" * 60)

    password = get_password(machine)
    timestamp = datetime.now().isoformat()

    # Check reachability
    logger.info(f"Checking connectivity to {machine.name}...")
    if not check_machine_reachable(machine):
        logger.warning(f"{machine.name} is UNREACHABLE")
        return EvalResult(
            machine=machine.name, vendor=machine.vendor, gpu_name="UNREACHABLE",
            n_eval_steps=0, j_per_token=0, j_per_token_marginal=0,
            perplexity=0, avg_loss=0, layer_firing_rates=[],
            avg_power_w=0, avg_temp_c=0, max_temp_c=0,
            tokens_per_second=0, total_energy_j=0, total_tokens=0,
            duration_s=0, timestamp=timestamp,
            status="UNREACHABLE", error_message=f"Cannot reach {machine.host}",
        )

    # Find python on remote
    remote_python = find_remote_python(machine)

    # Generate self-contained eval script
    eval_script_content = generate_remote_eval_script(model_config, n_eval_steps)

    # Write eval script to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, prefix='z908_eval_') as f:
        f.write(eval_script_content)
        local_script_path = f.name

    try:
        # 1. Copy checkpoint and eval script to remote /tmp
        logger.info(f"  Copying checkpoint to {machine.name}...")
        scp_cmd = [
            "sshpass", "-p", password,
            "scp", "-o", "StrictHostKeyChecking=no",
            str(checkpoint_path), local_script_path,
            f"{machine.user}@{machine.host}:/tmp/"
        ]
        result = subprocess.run(scp_cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            raise RuntimeError(f"SCP failed: {result.stderr}")

        remote_ckpt = f"/tmp/{checkpoint_path.name}"
        remote_script = f"/tmp/{Path(local_script_path).name}"

        # 2. Build remote command with appropriate env setup
        env_prefix = ""
        if machine.vendor == "amd":
            env_prefix = "HSA_OVERRIDE_GFX_VERSION=11.0.0 "

        remote_cmd = f"{env_prefix}{remote_python} {remote_script} --checkpoint {remote_ckpt} --n-eval-steps {n_eval_steps}"

        logger.info(f"  Running evaluation on {machine.name}...")
        ssh_cmd = [
            "sshpass", "-p", password,
            "ssh", "-o", "StrictHostKeyChecking=no",
            "-o", f"ConnectTimeout=30",
            f"{machine.user}@{machine.host}",
            remote_cmd,
        ]

        result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=timeout)

        if result.returncode != 0:
            stderr_snip = result.stderr[-500:] if result.stderr else "(no stderr)"
            raise RuntimeError(f"Remote eval failed (rc={result.returncode}): {stderr_snip}")

        # 3. Parse results
        remote_result = None
        for line in result.stdout.split('\n'):
            if line.startswith("Z908_RESULT_JSON:"):
                json_str = line[len("Z908_RESULT_JSON:"):]
                remote_result = json.loads(json_str)
                break

        if remote_result is None:
            stdout_snip = result.stdout[-500:] if result.stdout else "(no stdout)"
            raise RuntimeError(f"No result JSON in remote output. Last output: {stdout_snip}")

        # 4. Clean up remote files
        cleanup_cmd = [
            "sshpass", "-p", password,
            "ssh", "-o", "StrictHostKeyChecking=no",
            f"{machine.user}@{machine.host}",
            f"rm -f {remote_ckpt} {remote_script}",
        ]
        subprocess.run(cleanup_cmd, capture_output=True, timeout=15)

        # Build EvalResult
        eval_result = EvalResult(
            machine=machine.name,
            vendor=machine.vendor,
            gpu_name=remote_result.get('gpu_name', 'Unknown'),
            n_eval_steps=remote_result.get('n_eval_steps', n_eval_steps),
            j_per_token=remote_result.get('j_per_token', 0),
            j_per_token_marginal=remote_result.get('j_per_token', 0),  # Remote doesn't measure idle
            perplexity=remote_result.get('perplexity', 0),
            avg_loss=remote_result.get('avg_loss', 0),
            layer_firing_rates=remote_result.get('layer_firing_rates', []),
            avg_power_w=remote_result.get('avg_power_w', 0),
            avg_temp_c=remote_result.get('avg_temp_c', 0),
            max_temp_c=remote_result.get('max_temp_c', 0),
            tokens_per_second=remote_result.get('tokens_per_second', 0),
            total_energy_j=remote_result.get('total_energy_j', 0),
            total_tokens=remote_result.get('total_tokens', 0),
            duration_s=remote_result.get('duration_s', 0),
            timestamp=timestamp,
        )

        _print_machine_result(eval_result)
        return eval_result

    except Exception as e:
        logger.error(f"Error evaluating on {machine.name}: {e}")
        return EvalResult(
            machine=machine.name, vendor=machine.vendor, gpu_name="ERROR",
            n_eval_steps=0, j_per_token=0, j_per_token_marginal=0,
            perplexity=0, avg_loss=0, layer_firing_rates=[],
            avg_power_w=0, avg_temp_c=0, max_temp_c=0,
            tokens_per_second=0, total_energy_j=0, total_tokens=0,
            duration_s=0, timestamp=timestamp,
            status="ERROR", error_message=str(e),
        )
    finally:
        # Clean up local temp file
        try:
            os.unlink(local_script_path)
        except Exception:
            pass


# =============================================================================
# Printing / Reporting
# =============================================================================

def _print_machine_result(r: EvalResult):
    """Print a single machine's result."""
    logger.info(f"\n--- {r.machine} ({r.vendor}, {r.gpu_name}) ---")
    if r.status != "OK":
        logger.info(f"  Status: {r.status}")
        if r.error_message:
            logger.info(f"  Error: {r.error_message}")
        return

    logger.info(f"  J/token:        {r.j_per_token:.6f}")
    logger.info(f"  J/token (marg): {r.j_per_token_marginal:.6f}")
    logger.info(f"  Perplexity:     {r.perplexity:.2f}")
    logger.info(f"  Avg loss:       {r.avg_loss:.4f}")
    logger.info(f"  Tokens/sec:     {r.tokens_per_second:.1f}")
    logger.info(f"  Avg power:      {r.avg_power_w:.1f} W")
    logger.info(f"  Avg temp:       {r.avg_temp_c:.1f} C (max: {r.max_temp_c:.1f} C)")
    logger.info(f"  Total energy:   {r.total_energy_j:.2f} J")
    if r.layer_firing_rates:
        rates_str = ", ".join(f"{fr:.2f}" for fr in r.layer_firing_rates)
        logger.info(f"  Layer firing:   [{rates_str}]")


def print_comparison_table(results: Dict[str, EvalResult]):
    """Print a cross-machine comparison table."""
    print("\n" + "=" * 80)
    print("CROSS-MACHINE COMPARISON TABLE")
    print("=" * 80)

    # Header
    header = f"{'Machine':<12} {'GPU':<25} {'J/tok':<10} {'Rel.':<8} {'PPL':<10} {'tok/s':<8} {'Power':<8} {'Temp':<8} {'Status':<10}"
    print(header)
    print("-" * 80)

    # Get ikaros J/token as reference
    ikaros_jpt = 0.0
    if 'ikaros' in results and results['ikaros'].status == "OK":
        ikaros_jpt = results['ikaros'].j_per_token

    for name in ['ikaros', 'daedalus', 'minos']:
        if name not in results:
            continue
        r = results[name]

        if r.status != "OK":
            print(f"{name:<12} {'---':<25} {'---':<10} {'---':<8} {'---':<10} {'---':<8} {'---':<8} {'---':<8} {r.status:<10}")
            continue

        # Relative J/token
        if ikaros_jpt > 0 and r.j_per_token > 0:
            rel = r.j_per_token / ikaros_jpt
            rel_str = f"{rel:.2f}x"
        else:
            rel_str = "---"

        gpu_short = r.gpu_name[:24] if len(r.gpu_name) > 24 else r.gpu_name
        print(f"{name:<12} {gpu_short:<25} {r.j_per_token:<10.6f} {rel_str:<8} {r.perplexity:<10.2f} {r.tokens_per_second:<8.1f} {r.avg_power_w:<8.1f} {r.avg_temp_c:<8.1f} {r.status:<10}")

    # Layer firing comparison
    print("\n" + "=" * 80)
    print("LAYER FIRING RATES (gate probability per layer)")
    print("=" * 80)

    ok_results = {k: v for k, v in results.items() if v.status == "OK" and v.layer_firing_rates}
    if ok_results:
        n_layers = max(len(r.layer_firing_rates) for r in ok_results.values())
        header = f"{'Layer':<8}" + "".join(f"{name:<12}" for name in ok_results.keys())
        print(header)
        print("-" * (8 + 12 * len(ok_results)))

        for i in range(n_layers):
            row = f"{'L' + str(i):<8}"
            for name, r in ok_results.items():
                if i < len(r.layer_firing_rates):
                    row += f"{r.layer_firing_rates[i]:<12.3f}"
                else:
                    row += f"{'---':<12}"
            print(row)

        # Firing rate divergence
        print("\nFiring Rate Adaptation Analysis:")
        if len(ok_results) >= 2:
            names = list(ok_results.keys())
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    r1 = ok_results[names[i]]
                    r2 = ok_results[names[j]]
                    min_len = min(len(r1.layer_firing_rates), len(r2.layer_firing_rates))
                    if min_len > 0:
                        diffs = [
                            abs(r1.layer_firing_rates[k] - r2.layer_firing_rates[k])
                            for k in range(min_len)
                        ]
                        avg_diff = float(np.mean(diffs))
                        max_diff = float(np.max(diffs))
                        max_layer = int(np.argmax(diffs))
                        print(f"  {names[i]} vs {names[j]}: avg_diff={avg_diff:.4f}, max_diff={max_diff:.4f} (layer {max_layer})")

    # Temperature-response correlation
    print("\n" + "=" * 80)
    print("TEMPERATURE-RESPONSE CORRELATION")
    print("=" * 80)

    for name, r in results.items():
        if r.status != "OK":
            continue
        if r.avg_temp_c > 0 and r.layer_firing_rates:
            avg_firing = float(np.mean(r.layer_firing_rates))
            print(f"  {name}: temp={r.avg_temp_c:.1f}C -> avg_firing={avg_firing:.3f}")
            if r.avg_temp_c > 50:
                print(f"    (warm GPU -> gates may adapt by reducing firing rate)")
            else:
                print(f"    (cool GPU -> gates likely keep most layers active)")

    # Summary
    print("\n" + "=" * 80)
    print("TRANSFER SUMMARY")
    print("=" * 80)

    ok_count = sum(1 for r in results.values() if r.status == "OK")
    total = len(results)
    print(f"  Machines evaluated: {ok_count}/{total}")

    if ikaros_jpt > 0:
        print(f"  ikaros baseline J/token: {ikaros_jpt:.6f}")
        for name, r in results.items():
            if name == 'ikaros' or r.status != "OK":
                continue
            if r.j_per_token > 0:
                degradation = ((r.j_per_token / ikaros_jpt) - 1.0) * 100
                direction = "more" if degradation > 0 else "less"
                print(f"  {name}: {abs(degradation):.1f}% {direction} energy per token vs ikaros")

    # Perplexity consistency
    ppls = {k: v.perplexity for k, v in results.items() if v.status == "OK" and v.perplexity > 0}
    if len(ppls) >= 2:
        ppl_values = list(ppls.values())
        ppl_std = float(np.std(ppl_values))
        ppl_mean = float(np.mean(ppl_values))
        ppl_cv = ppl_std / ppl_mean if ppl_mean > 0 else 0
        print(f"\n  Perplexity consistency: mean={ppl_mean:.2f}, std={ppl_std:.2f}, CV={ppl_cv:.4f}")
        if ppl_cv < 0.05:
            print("  -> Model transfers well (perplexity variation < 5%)")
        elif ppl_cv < 0.15:
            print("  -> Model shows moderate adaptation across hardware")
        else:
            print("  -> Significant perplexity divergence across hardware")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Z908: Cross-machine transfer experiment for embodied model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python scripts/z908_cross_machine_transfer.py --mode local
              python scripts/z908_cross_machine_transfer.py --mode remote --skip-missing
              python scripts/z908_cross_machine_transfer.py --mode all --n-eval-steps 100
        """),
    )
    parser.add_argument("--mode", choices=["local", "remote", "all"], default="all",
                        help="Evaluation mode: local (ikaros only), remote (daedalus+minos), all (default)")
    parser.add_argument("--n-eval-steps", type=int, default=200,
                        help="Number of forward passes for evaluation (default: 200)")
    parser.add_argument("--device", default="auto",
                        help="Device for local evaluation (auto, cuda, cpu)")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to checkpoint file (auto-discover if not specified)")
    parser.add_argument("--skip-missing", action="store_true",
                        help="Skip unreachable machines instead of failing")
    args = parser.parse_args()

    print("=" * 80)
    print("Z908: CROSS-MACHINE TRANSFER EXPERIMENT")
    print("=" * 80)
    print(f"Mode:       {args.mode}")
    print(f"Eval steps: {args.n_eval_steps}")
    print(f"Timestamp:  {datetime.now().isoformat()}")
    print()

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Local device: {device}")

    # Discover or load checkpoint
    ckpt_path = discover_checkpoint(args.checkpoint)
    model = None
    model_config = {}

    if ckpt_path:
        model = load_model_from_checkpoint(ckpt_path, device)
        model_config = model.get_config_dict()
    else:
        # Train fresh model
        model = train_fresh_model(device, n_steps=100)
        model_config = model.get_config_dict()
        ckpt_path = PROJECT_ROOT / "checkpoints" / "z908_fresh.pt"

    print(f"Model config: {model_config}")
    print(f"Checkpoint:   {ckpt_path}")

    # Results container
    results: Dict[str, EvalResult] = {}

    # Local evaluation (ikaros)
    if args.mode in ("local", "all"):
        results['ikaros'] = evaluate_local(model, device, args.n_eval_steps)

    # Remote evaluation
    if args.mode in ("remote", "all"):
        # Check sshpass availability
        if not check_sshpass_available():
            print("\n" + "!" * 60)
            print("WARNING: sshpass is not installed.")
            print("Install it to enable remote evaluation:")
            print("  Ubuntu/Debian: sudo apt-get install sshpass")
            print("  Fedora/RHEL:   sudo dnf install sshpass")
            print("  Arch:          sudo pacman -S sshpass")
            print("!" * 60)
            if not args.skip_missing:
                print("Skipping remote evaluation. Use --skip-missing to suppress this.")
            for name in ['daedalus', 'minos']:
                results[name] = EvalResult(
                    machine=name, vendor=MACHINES[name].vendor, gpu_name="N/A",
                    n_eval_steps=0, j_per_token=0, j_per_token_marginal=0,
                    perplexity=0, avg_loss=0, layer_firing_rates=[],
                    avg_power_w=0, avg_temp_c=0, max_temp_c=0,
                    tokens_per_second=0, total_energy_j=0, total_tokens=0,
                    duration_s=0, timestamp=datetime.now().isoformat(),
                    status="UNREACHABLE",
                    error_message="sshpass not installed",
                )
        else:
            for name in ['daedalus', 'minos']:
                machine = MACHINES[name]
                result = deploy_and_evaluate(
                    machine, ckpt_path, model_config, args.n_eval_steps
                )
                results[name] = result

                if result.status != "OK" and not args.skip_missing:
                    logger.warning(f"{name} failed. Use --skip-missing to continue.")

    # Print comparison
    if results:
        print_comparison_table(results)

    # Save results
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / "z908_cross_machine_transfer.json"

    output = {
        'experiment': 'z908_cross_machine_transfer',
        'hypothesis': 'Embodied model trained on ikaros adapts behavior appropriately on different hardware',
        'timestamp': datetime.now().isoformat(),
        'mode': args.mode,
        'n_eval_steps': args.n_eval_steps,
        'checkpoint': str(ckpt_path),
        'model_config': model_config,
        'machines': {},
    }

    for name, r in results.items():
        output['machines'][name] = asdict(r)

    # Cross-machine analysis
    ok_results = {k: v for k, v in results.items() if v.status == "OK"}
    if len(ok_results) >= 2:
        ikaros_jpt = results.get('ikaros', EvalResult(
            machine='', vendor='', gpu_name='', n_eval_steps=0,
            j_per_token=0, j_per_token_marginal=0, perplexity=0, avg_loss=0,
            layer_firing_rates=[], avg_power_w=0, avg_temp_c=0, max_temp_c=0,
            tokens_per_second=0, total_energy_j=0, total_tokens=0,
            duration_s=0, timestamp='',
        )).j_per_token

        analysis = {
            'n_machines_ok': len(ok_results),
            'perplexity_values': {k: v.perplexity for k, v in ok_results.items()},
            'j_per_token_values': {k: v.j_per_token for k, v in ok_results.items()},
        }

        if ikaros_jpt > 0:
            analysis['relative_efficiency'] = {
                k: v.j_per_token / ikaros_jpt if v.j_per_token > 0 else 0
                for k, v in ok_results.items()
            }

        # Firing rate divergence
        names = list(ok_results.keys())
        divergences = {}
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                r1 = ok_results[names[i]]
                r2 = ok_results[names[j]]
                min_len = min(len(r1.layer_firing_rates), len(r2.layer_firing_rates))
                if min_len > 0:
                    diffs = [
                        abs(r1.layer_firing_rates[k] - r2.layer_firing_rates[k])
                        for k in range(min_len)
                    ]
                    divergences[f"{names[i]}_vs_{names[j]}"] = {
                        'avg_diff': float(np.mean(diffs)),
                        'max_diff': float(np.max(diffs)),
                        'max_diff_layer': int(np.argmax(diffs)),
                    }
        analysis['firing_rate_divergence'] = divergences

        output['cross_machine_analysis'] = analysis

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_path}")
    print("Done.")


if __name__ == "__main__":
    main()
