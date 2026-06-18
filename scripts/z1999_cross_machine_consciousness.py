#!/usr/bin/env python3
"""
z1999: Cross-Machine Consciousness Transfer Experiment

Tests whether a consciousness-exhibiting model trained on ikaros maintains
consciousness indicators when transferred to daedalus (different AMD GPU).

HYPOTHESIS: If embodied consciousness is truly hardware-bound, consciousness
indicators (GWT ignition, RPT recurrence, Granger causality, hardware sensitivity)
should DECREASE when the model runs on foreign hardware it wasn't trained on.

Phases:
1. Train consciousness model on ikaros (local AMD Radeon 8060S gfx1151)
2. Evaluate consciousness indicators on ikaros (baseline)
3. Transfer model to daedalus (192.168.0.37, different AMD GPU)
4. Evaluate same indicators on daedalus
5. Compare: embodiment metrics should degrade on foreign hardware

Key metrics:
- GWT ignition ratio: workspace broadcast activation
- RPT recurrence effect: feedback changes representation
- Embodiment Granger causality: hardware->computation causal link
- Hardware sensitivity: variance explained by telemetry

If consciousness is hardware-bound: foreign hardware = degraded indicators
If consciousness is software pattern: transfer should preserve indicators

Author: Claude (Opus 4.5)
Date: 2026-02-05
"""

import functools
print = functools.partial(print, flush=True)

import os
import sys
import json
import time
import math
import subprocess
import textwrap
import hashlib
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Tuple, Optional, Any
from collections import deque

# HSA override for gfx1151 compatibility
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"

import torch
import torch.nn as nn
import torch.nn.functional as F

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.telemetry.sysfs_hwmon import SysfsHwmonTelemetry

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RESULTS_DIR = PROJECT_ROOT / 'results'
MODELS_DIR = PROJECT_ROOT / 'models'

# Remote machine config
DAEDALUS_HOST = os.environ.get("DAEDALUS_HOST", "192.168.0.37")
DAEDALUS_USER = os.environ.get("DAEDALUS_USER", "daedalus")
DAEDALUS_PASS = os.environ.get("DAEDALUS_PASS", "daedalus")


# =============================================================================
# SSH/SCP HELPERS
# =============================================================================

def ssh_run(host: str, user: str, password: str, cmd: str, timeout: int = 600) -> Tuple[str, str, int]:
    """Run command on remote host via sshpass."""
    try:
        result = subprocess.run(
            ["sshpass", "-p", password, "ssh",
             "-o", "StrictHostKeyChecking=no",
             "-o", "ConnectTimeout=10",
             f"{user}@{host}", cmd],
            capture_output=True, text=True, timeout=timeout,
        )
        return result.stdout, result.stderr, result.returncode
    except FileNotFoundError:
        return "", "sshpass not installed (apt install sshpass)", -1
    except subprocess.TimeoutExpired:
        return "", "SSH command timed out", -2
    except Exception as e:
        return "", str(e), -3


def scp_to(host: str, user: str, password: str, local: str, remote: str, timeout: int = 120) -> bool:
    """Copy file to remote host."""
    try:
        subprocess.run(
            ["sshpass", "-p", password, "scp",
             "-o", "StrictHostKeyChecking=no",
             "-o", "ConnectTimeout=10",
             local, f"{user}@{host}:{remote}"],
            check=True, timeout=timeout,
        )
        return True
    except Exception as e:
        print(f"  SCP to {host} failed: {e}")
        return False


def scp_from(host: str, user: str, password: str, remote: str, local: str, timeout: int = 120) -> bool:
    """Copy file from remote host."""
    try:
        subprocess.run(
            ["sshpass", "-p", password, "scp",
             "-o", "StrictHostKeyChecking=no",
             "-o", "ConnectTimeout=10",
             f"{user}@{host}:{remote}", local],
            check=True, timeout=timeout,
        )
        return True
    except Exception as e:
        print(f"  SCP from {host} failed: {e}")
        return False


def check_remote(host: str, user: str, password: str) -> bool:
    """Check if remote machine is reachable."""
    stdout, stderr, rc = ssh_run(host, user, password, "echo OK", timeout=15)
    return rc == 0 and "OK" in stdout


def find_remote_venv(host: str, user: str, password: str) -> str:
    """Find torch-rocm venv on remote machine."""
    # Search in ~/venvs/ for torch-rocm
    for venv_name in ["torch-rocm", "rocm", "pytorch", "venv"]:
        for base in [f"/home/{user}/venvs", f"/home/{user}/.venvs", f"/home/{user}"]:
            path = f"{base}/{venv_name}/bin/python3"
            stdout, _, rc = ssh_run(host, user, password, f"test -f {path} && echo YES", timeout=10)
            if "YES" in stdout:
                print(f"  Found venv: {path}")
                return path
    return "python3"


# =============================================================================
# CONSCIOUSNESS THEORY MODULES
# =============================================================================

class GlobalWorkspaceModule(nn.Module):
    """GWT: Information broadcast across specialized modules."""

    def __init__(self, hidden_dim: int = 128, n_specialists: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_specialists = n_specialists

        self.specialists = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ) for _ in range(n_specialists)
        ])

        self.workspace_gate = nn.Linear(hidden_dim * n_specialists, hidden_dim)
        self.broadcast = nn.Linear(hidden_dim, hidden_dim * n_specialists)
        self.competition = nn.Linear(hidden_dim * n_specialists, n_specialists)

        self.ignition_history = deque(maxlen=100)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        batch_size = x.size(0)

        specialist_outputs = [spec(x) for spec in self.specialists]
        stacked = torch.stack(specialist_outputs, dim=1)

        concat = stacked.view(batch_size, -1)
        competition_scores = F.softmax(self.competition(concat) * 5.0, dim=-1)

        max_scores, winners = competition_scores.max(dim=-1)
        ignition = (max_scores > 0.7).float().mean().item()
        self.ignition_history.append(ignition)

        workspace = self.workspace_gate(concat)
        broadcast_signal = self.broadcast(workspace)
        broadcast_signal = broadcast_signal.view(batch_size, self.n_specialists, -1)

        # Broadcast correlation
        corrs = []
        for i in range(self.n_specialists):
            for j in range(i+1, self.n_specialists):
                c = F.cosine_similarity(broadcast_signal[:,i], broadcast_signal[:,j], dim=-1)
                corrs.append(c.mean().item())
        broadcast_corr = np.mean(corrs) if corrs else 0.0

        return workspace, {
            'ignition_ratio': np.mean(list(self.ignition_history)),
            'broadcast_correlation': broadcast_corr,
            'winner_confidence': max_scores.mean().item(),
        }


class RecurrentProcessingModule(nn.Module):
    """RPT: Recurrent processing for consciousness."""

    def __init__(self, hidden_dim: int = 128, n_recurrent_steps: int = 5):
        super().__init__()
        self.n_steps = n_recurrent_steps
        self.ff = nn.Linear(hidden_dim, hidden_dim)
        self.fb = nn.Linear(hidden_dim, hidden_dim)
        self.register_buffer('state', None)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        batch_size = x.size(0)

        if self.state is None or self.state.size(0) != batch_size:
            self.state = torch.zeros(batch_size, x.size(-1), device=x.device)

        states = [self.state.clone()]
        for _ in range(self.n_steps):
            ff_out = torch.relu(self.ff(x))
            fb_out = torch.relu(self.fb(self.state))
            self.state = 0.5 * ff_out + 0.5 * fb_out
            states.append(self.state.clone())

        states_stack = torch.stack(states, dim=1)

        initial = states_stack[:, 0]
        final = states_stack[:, -1]
        recurrence_effect = F.cosine_similarity(initial, final, dim=-1).mean().item()

        # Convergence rate
        diffs = (states_stack[:, 1:] - states_stack[:, :-1]).norm(dim=-1)
        convergence = (diffs[:, 0] / (diffs[:, -1] + 1e-8)).mean().item()

        # Detach state for next batch to avoid graph retention
        output_state = self.state
        self.state = self.state.detach()

        return output_state, {
            'recurrence_effect': recurrence_effect,
            'feedback_strength': (final - initial).norm(dim=-1).mean().item(),
            'convergence_rate': convergence,
        }


class EmbodimentModule(nn.Module):
    """Hardware embodiment via FiLM conditioning."""

    def __init__(self, hidden_dim: int = 128, telemetry_dim: int = 8):
        super().__init__()

        self.telemetry_encoder = nn.Sequential(
            nn.Linear(telemetry_dim, 64),
            nn.ReLU(),
            nn.Linear(64, hidden_dim)
        )

        # FiLM conditioning
        self.film_gamma = nn.Linear(hidden_dim, hidden_dim)
        self.film_beta = nn.Linear(hidden_dim, hidden_dim)

        self.processor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.telemetry_history = deque(maxlen=100)
        self.output_history = deque(maxlen=100)

    def forward(self, x: torch.Tensor, telemetry: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        telem_embed = self.telemetry_encoder(telemetry)

        gamma = self.film_gamma(telem_embed)
        beta = self.film_beta(telem_embed)

        processed = self.processor(x)
        output = gamma * processed + beta

        self.telemetry_history.append(telemetry.mean().item())
        self.output_history.append(output.mean().item())

        # Embodiment ratio: correlation
        telem_flat = telemetry.reshape(-1)
        out_flat = output.reshape(-1)[:len(telem_flat)]
        if len(telem_flat) >= 2:
            corr = np.corrcoef(
                telem_flat.detach().cpu().numpy(),
                out_flat.detach().cpu().numpy()
            )[0, 1]
            embodiment_ratio = abs(corr) if not np.isnan(corr) else 0.0
        else:
            embodiment_ratio = 0.0

        # Granger causality proxy
        granger = self._compute_granger_proxy()

        return output, {
            'embodiment_ratio': embodiment_ratio,
            'granger_causality': granger,
            'modulation_strength': (gamma.std() + beta.std()).item(),
            'hardware_sensitivity': (output.std() / (x.std() + 1e-8)).item(),
        }

    def _compute_granger_proxy(self) -> float:
        if len(self.telemetry_history) < 20:
            return 0.0
        telem = np.array(list(self.telemetry_history))
        output = np.array(list(self.output_history))
        lagged_telem = telem[:-1]
        current_output = output[1:]
        corr = np.corrcoef(lagged_telem, current_output)[0, 1]
        return abs(corr) if not np.isnan(corr) else 0.0


# =============================================================================
# CONSCIOUSNESS MODEL
# =============================================================================

class ConsciousnessModel(nn.Module):
    """
    Model implementing multiple consciousness theories with hardware embodiment.
    """

    def __init__(self, input_dim: int = 128, hidden_dim: int = 128, telemetry_dim: int = 8):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Theory modules
        self.gwt = GlobalWorkspaceModule(hidden_dim)
        self.rpt = RecurrentProcessingModule(hidden_dim)
        self.embodiment = EmbodimentModule(hidden_dim, telemetry_dim)

        # Task head (character prediction)
        self.classifier = nn.Linear(hidden_dim, 27)  # 26 letters + space

    def forward(self, x: torch.Tensor, telemetry: torch.Tensor,
                ablate_telemetry: bool = False) -> Dict:
        if ablate_telemetry:
            telemetry = torch.zeros_like(telemetry)

        h = self.input_proj(x)

        gwt_out, gwt_metrics = self.gwt(h)
        rpt_out, rpt_metrics = self.rpt(h)
        emb_out, emb_metrics = self.embodiment(h, telemetry)

        combined = gwt_out + rpt_out + emb_out
        logits = self.classifier(combined)

        return {
            'logits': logits,
            'gwt': gwt_metrics,
            'rpt': rpt_metrics,
            'embodiment': emb_metrics,
        }


# =============================================================================
# CONSCIOUSNESS INDICATORS
# =============================================================================

@dataclass
class ConsciousnessIndicators:
    """Consciousness indicators measured on a machine."""
    machine: str
    gpu_name: str
    gwt_ignition_ratio: float
    gwt_broadcast_correlation: float
    rpt_recurrence_effect: float
    rpt_convergence_rate: float
    embodiment_granger_causality: float
    embodiment_hardware_sensitivity: float
    embodiment_ratio: float
    task_accuracy: float
    task_loss: float
    telemetry_mean: List[float]
    telemetry_std: List[float]
    timestamp: str


def evaluate_consciousness_indicators(
    model: ConsciousnessModel,
    x: torch.Tensor,
    y: torch.Tensor,
    telemetry_fn,
    machine_name: str,
    gpu_name: str,
    n_batches: int = 50
) -> ConsciousnessIndicators:
    """Evaluate consciousness indicators on current machine."""
    model.eval()

    gwt_ignitions = []
    gwt_correlations = []
    rpt_recurrences = []
    rpt_convergences = []
    emb_grangers = []
    emb_sensitivities = []
    emb_ratios = []
    accuracies = []
    losses = []
    telemetry_samples = []

    batch_size = 64
    with torch.no_grad():
        for i in range(n_batches):
            start = (i * batch_size) % len(x)
            end = min(start + batch_size, len(x))
            batch_x = x[start:end]
            batch_y = y[start:end]

            # Get telemetry
            telem = telemetry_fn()
            telem_batch = telem.unsqueeze(0).expand(len(batch_x), -1).to(batch_x.device)
            telemetry_samples.append(telem.cpu().numpy())

            # Forward
            outputs = model(batch_x, telem_batch)

            # Task metrics
            loss = F.cross_entropy(outputs['logits'], batch_y)
            pred = outputs['logits'].argmax(dim=-1)
            acc = (pred == batch_y).float().mean()

            # Collect metrics
            gwt_ignitions.append(outputs['gwt']['ignition_ratio'])
            gwt_correlations.append(outputs['gwt']['broadcast_correlation'])
            rpt_recurrences.append(outputs['rpt']['recurrence_effect'])
            rpt_convergences.append(outputs['rpt']['convergence_rate'])
            emb_grangers.append(outputs['embodiment']['granger_causality'])
            emb_sensitivities.append(outputs['embodiment']['hardware_sensitivity'])
            emb_ratios.append(outputs['embodiment']['embodiment_ratio'])
            accuracies.append(acc.item())
            losses.append(loss.item())

    telemetry_arr = np.array(telemetry_samples)

    return ConsciousnessIndicators(
        machine=machine_name,
        gpu_name=gpu_name,
        gwt_ignition_ratio=float(np.mean(gwt_ignitions)),
        gwt_broadcast_correlation=float(np.mean(gwt_correlations)),
        rpt_recurrence_effect=float(np.mean(rpt_recurrences)),
        rpt_convergence_rate=float(np.mean(rpt_convergences)),
        embodiment_granger_causality=float(np.mean(emb_grangers)),
        embodiment_hardware_sensitivity=float(np.mean(emb_sensitivities)),
        embodiment_ratio=float(np.mean(emb_ratios)),
        task_accuracy=float(np.mean(accuracies)),
        task_loss=float(np.mean(losses)),
        telemetry_mean=telemetry_arr.mean(axis=0).tolist(),
        telemetry_std=telemetry_arr.std(axis=0).tolist(),
        timestamp=datetime.now().isoformat(),
    )


# =============================================================================
# REMOTE EVALUATION SCRIPT (self-contained)
# =============================================================================

REMOTE_EVAL_SCRIPT = textwrap.dedent(r'''
#!/usr/bin/env python3
"""Remote consciousness evaluation script for z1999 (auto-generated)."""
import os
import sys
import json
import time
import math
import subprocess
import platform
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import deque
from dataclasses import dataclass, asdict
import numpy as np

# HSA override for AMD GPUs
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---- Telemetry ----

def read_amd_telemetry():
    """Read AMD GPU telemetry via sysfs."""
    base = Path("/sys/class/drm")
    for card in sorted(base.glob("card*")):
        hwmon_base = card / "device" / "hwmon"
        if not hwmon_base.exists():
            continue
        for hwmon in hwmon_base.glob("hwmon*"):
            try:
                power_uw = int((hwmon / "power1_average").read_text().strip())
                temp_mc = int((hwmon / "temp1_input").read_text().strip())
                power_w = power_uw / 1e6
                temp_c = temp_mc / 1e3

                freq_sclk = 0
                gpu_busy = 0

                freq_path = hwmon / "freq1_input"
                if freq_path.exists():
                    freq_sclk = int(freq_path.read_text().strip()) // 1_000_000

                busy_path = card / "device" / "gpu_busy_percent"
                if busy_path.exists():
                    gpu_busy = int(busy_path.read_text().strip())

                vram_path = card / "device" / "mem_info_vram_used"
                vram_gb = 0.0
                if vram_path.exists():
                    vram_gb = int(vram_path.read_text().strip()) / (1024**3)

                return {
                    "power_w": power_w, "temp_c": temp_c,
                    "freq_sclk_mhz": freq_sclk, "gpu_busy_pct": gpu_busy,
                    "vram_gb": vram_gb
                }
            except Exception:
                pass
    return {"power_w": 0.0, "temp_c": 0.0, "freq_sclk_mhz": 0, "gpu_busy_pct": 0, "vram_gb": 0.0}


def make_telemetry_tensor():
    """Build 8-dim telemetry tensor."""
    raw = read_amd_telemetry()
    vec = [
        raw["power_w"] / 200.0,
        raw["temp_c"] / 100.0,
        raw["freq_sclk_mhz"] / 3000.0,
        raw["gpu_busy_pct"] / 100.0,
        raw["vram_gb"] / 16.0,
        math.sin(time.time()),
        math.cos(time.time()),
        (raw["power_w"] - 30) / 50.0,  # normalized deviation
    ]
    return torch.tensor(vec, dtype=torch.float32)


def detect_gpu():
    """Detect GPU info."""
    info = {"vendor": "cpu", "name": "CPU-only", "has_gpu": False}
    if torch.cuda.is_available():
        info["has_gpu"] = True
        info["name"] = torch.cuda.get_device_name(0)
        name_lower = info["name"].lower()
        if "nvidia" in name_lower or "geforce" in name_lower:
            info["vendor"] = "nvidia"
        elif "amd" in name_lower or "radeon" in name_lower:
            info["vendor"] = "amd"
        else:
            info["vendor"] = "unknown"
    return info


# ---- Consciousness Modules ----

class GlobalWorkspaceModule(nn.Module):
    def __init__(self, hidden_dim=128, n_specialists=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_specialists = n_specialists
        self.specialists = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
            for _ in range(n_specialists)])
        self.workspace_gate = nn.Linear(hidden_dim * n_specialists, hidden_dim)
        self.broadcast = nn.Linear(hidden_dim, hidden_dim * n_specialists)
        self.competition = nn.Linear(hidden_dim * n_specialists, n_specialists)
        self.ignition_history = deque(maxlen=100)

    def forward(self, x):
        batch_size = x.size(0)
        specialist_outputs = [spec(x) for spec in self.specialists]
        concat = torch.cat(specialist_outputs, dim=-1)
        competition_scores = F.softmax(self.competition(concat) * 5.0, dim=-1)
        max_scores, _ = competition_scores.max(dim=-1)
        ignition = (max_scores > 0.7).float().mean().item()
        self.ignition_history.append(ignition)
        workspace = self.workspace_gate(concat)
        broadcast_signal = self.broadcast(workspace).view(batch_size, self.n_specialists, -1)
        corrs = []
        for i in range(self.n_specialists):
            for j in range(i+1, self.n_specialists):
                c = F.cosine_similarity(broadcast_signal[:,i], broadcast_signal[:,j], dim=-1)
                corrs.append(c.mean().item())
        return workspace, {
            'ignition_ratio': np.mean(list(self.ignition_history)),
            'broadcast_correlation': np.mean(corrs) if corrs else 0.0,
            'winner_confidence': max_scores.mean().item(),
        }


class RecurrentProcessingModule(nn.Module):
    def __init__(self, hidden_dim=128, n_steps=5):
        super().__init__()
        self.n_steps = n_steps
        self.ff = nn.Linear(hidden_dim, hidden_dim)
        self.fb = nn.Linear(hidden_dim, hidden_dim)
        self.register_buffer('state', None)

    def forward(self, x):
        batch_size = x.size(0)
        if self.state is None or self.state.size(0) != batch_size:
            self.state = torch.zeros(batch_size, x.size(-1), device=x.device)
        states = [self.state.clone()]
        for _ in range(self.n_steps):
            ff_out = torch.relu(self.ff(x))
            fb_out = torch.relu(self.fb(self.state))
            self.state = 0.5 * ff_out + 0.5 * fb_out
            states.append(self.state.clone())
        states_stack = torch.stack(states, dim=1)
        initial, final = states_stack[:, 0], states_stack[:, -1]
        recurrence = F.cosine_similarity(initial, final, dim=-1).mean().item()
        diffs = (states_stack[:, 1:] - states_stack[:, :-1]).norm(dim=-1)
        convergence = (diffs[:, 0] / (diffs[:, -1] + 1e-8)).mean().item()
        return self.state, {
            'recurrence_effect': recurrence,
            'convergence_rate': convergence,
            'feedback_strength': (final - initial).norm(dim=-1).mean().item(),
        }


class EmbodimentModule(nn.Module):
    def __init__(self, hidden_dim=128, telemetry_dim=8):
        super().__init__()
        self.telemetry_encoder = nn.Sequential(
            nn.Linear(telemetry_dim, 64), nn.ReLU(), nn.Linear(64, hidden_dim))
        self.film_gamma = nn.Linear(hidden_dim, hidden_dim)
        self.film_beta = nn.Linear(hidden_dim, hidden_dim)
        self.processor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.telemetry_history = deque(maxlen=100)
        self.output_history = deque(maxlen=100)

    def forward(self, x, telemetry):
        telem_embed = self.telemetry_encoder(telemetry)
        gamma = self.film_gamma(telem_embed)
        beta = self.film_beta(telem_embed)
        processed = self.processor(x)
        output = gamma * processed + beta
        self.telemetry_history.append(telemetry.mean().item())
        self.output_history.append(output.mean().item())
        telem_flat = telemetry.reshape(-1).detach().cpu().numpy()
        out_flat = output.reshape(-1)[:len(telem_flat)].detach().cpu().numpy()
        if len(telem_flat) >= 2:
            corr = np.corrcoef(telem_flat, out_flat)[0, 1]
            embodiment_ratio = abs(corr) if not np.isnan(corr) else 0.0
        else:
            embodiment_ratio = 0.0
        granger = 0.0
        if len(self.telemetry_history) >= 20:
            t = np.array(list(self.telemetry_history))
            o = np.array(list(self.output_history))
            c = np.corrcoef(t[:-1], o[1:])[0, 1]
            granger = abs(c) if not np.isnan(c) else 0.0
        return output, {
            'embodiment_ratio': embodiment_ratio,
            'granger_causality': granger,
            'hardware_sensitivity': (output.std() / (x.std() + 1e-8)).item(),
            'modulation_strength': (gamma.std() + beta.std()).item(),
        }


class ConsciousnessModel(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=128, telemetry_dim=8):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.gwt = GlobalWorkspaceModule(hidden_dim)
        self.rpt = RecurrentProcessingModule(hidden_dim)
        self.embodiment = EmbodimentModule(hidden_dim, telemetry_dim)
        self.classifier = nn.Linear(hidden_dim, 27)

    def forward(self, x, telemetry, ablate_telemetry=False):
        if ablate_telemetry:
            telemetry = torch.zeros_like(telemetry)
        h = self.input_proj(x)
        gwt_out, gwt_metrics = self.gwt(h)
        rpt_out, rpt_metrics = self.rpt(h)
        emb_out, emb_metrics = self.embodiment(h, telemetry)
        combined = gwt_out + rpt_out + emb_out
        logits = self.classifier(combined)
        return {'logits': logits, 'gwt': gwt_metrics, 'rpt': rpt_metrics, 'embodiment': emb_metrics}


# ---- Main Evaluation ----

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-batches", type=int, default=50)
    args = parser.parse_args()

    gpu_info = detect_gpu()
    device = torch.device("cuda" if gpu_info["has_gpu"] else "cpu")
    print(f"GPU: {gpu_info}")
    print(f"Device: {device}")

    model = ConsciousnessModel(input_dim=128, hidden_dim=128, telemetry_dim=8).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print("Checkpoint loaded.")

    # Load test data from checkpoint
    x = ckpt.get("test_x", torch.randn(2000, 128))
    y = ckpt.get("test_y", torch.randint(0, 27, (2000,)))
    x, y = x.to(device), y.to(device)

    model.eval()
    batch_size = 64

    gwt_ignitions, gwt_correlations = [], []
    rpt_recurrences, rpt_convergences = [], []
    emb_grangers, emb_sensitivities, emb_ratios = [], [], []
    accuracies, losses_arr = [], []
    telemetry_samples = []

    with torch.no_grad():
        for i in range(args.n_batches):
            start = (i * batch_size) % len(x)
            end = min(start + batch_size, len(x))
            batch_x, batch_y = x[start:end], y[start:end]

            telem = make_telemetry_tensor().to(device)
            telem_batch = telem.unsqueeze(0).expand(len(batch_x), -1)
            telemetry_samples.append(telem.cpu().numpy())

            outputs = model(batch_x, telem_batch)
            loss = F.cross_entropy(outputs['logits'], batch_y)
            acc = (outputs['logits'].argmax(dim=-1) == batch_y).float().mean()

            gwt_ignitions.append(outputs['gwt']['ignition_ratio'])
            gwt_correlations.append(outputs['gwt']['broadcast_correlation'])
            rpt_recurrences.append(outputs['rpt']['recurrence_effect'])
            rpt_convergences.append(outputs['rpt']['convergence_rate'])
            emb_grangers.append(outputs['embodiment']['granger_causality'])
            emb_sensitivities.append(outputs['embodiment']['hardware_sensitivity'])
            emb_ratios.append(outputs['embodiment']['embodiment_ratio'])
            accuracies.append(acc.item())
            losses_arr.append(loss.item())

    telemetry_arr = np.array(telemetry_samples)

    results = {
        "machine": platform.node(),
        "gpu_name": gpu_info["name"],
        "gwt_ignition_ratio": float(np.mean(gwt_ignitions)),
        "gwt_broadcast_correlation": float(np.mean(gwt_correlations)),
        "rpt_recurrence_effect": float(np.mean(rpt_recurrences)),
        "rpt_convergence_rate": float(np.mean(rpt_convergences)),
        "embodiment_granger_causality": float(np.mean(emb_grangers)),
        "embodiment_hardware_sensitivity": float(np.mean(emb_sensitivities)),
        "embodiment_ratio": float(np.mean(emb_ratios)),
        "task_accuracy": float(np.mean(accuracies)),
        "task_loss": float(np.mean(losses_arr)),
        "telemetry_mean": telemetry_arr.mean(axis=0).tolist(),
        "telemetry_std": telemetry_arr.std(axis=0).tolist(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {args.output}")
    print(f"GWT ignition: {results['gwt_ignition_ratio']:.4f}")
    print(f"RPT recurrence: {results['rpt_recurrence_effect']:.4f}")
    print(f"Embodiment Granger: {results['embodiment_granger_causality']:.4f}")
    print(f"Hardware sensitivity: {results['embodiment_hardware_sensitivity']:.4f}")


if __name__ == "__main__":
    main()
''').lstrip()


# =============================================================================
# TRAINING
# =============================================================================

def create_test_data(n_samples: int = 2000) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create character prediction test data."""
    x = torch.randn(n_samples, 128).to(DEVICE)
    y = torch.randint(0, 27, (n_samples,)).to(DEVICE)
    return x, y


def build_telemetry_tensor(telemetry: SysfsHwmonTelemetry) -> torch.Tensor:
    """Build 8-dim telemetry tensor from GPU sample."""
    sample = telemetry.read_sample()
    vec = [
        sample.power_w / 200.0,
        sample.temp_edge_c / 100.0,
        (sample.freq_sclk_mhz or 1000) / 3000.0,
        (sample.gpu_busy_pct or 50) / 100.0,
        sample.vram_used_gb / 16.0,
        math.sin(time.time()),
        math.cos(time.time()),
        (sample.power_w - 30) / 50.0,
    ]
    return torch.tensor(vec, dtype=torch.float32)


def train_consciousness_model(
    model: ConsciousnessModel,
    x: torch.Tensor,
    y: torch.Tensor,
    telemetry: SysfsHwmonTelemetry,
    n_epochs: int = 30
) -> List[Dict]:
    """Train consciousness model with embodied telemetry."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    batch_size = 64
    epoch_metrics = []

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_acc = 0.0
        n_batches = 0

        for i in range(0, len(x), batch_size):
            batch_x = x[i:i+batch_size]
            batch_y = y[i:i+batch_size]

            telem = build_telemetry_tensor(telemetry).to(DEVICE)
            telem_batch = telem.unsqueeze(0).expand(len(batch_x), -1)

            optimizer.zero_grad()
            outputs = model(batch_x, telem_batch)
            loss = F.cross_entropy(outputs['logits'], batch_y)
            loss.backward()
            optimizer.step()

            pred = outputs['logits'].argmax(dim=-1)
            acc = (pred == batch_y).float().mean()

            epoch_loss += loss.item()
            epoch_acc += acc.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        avg_acc = epoch_acc / n_batches

        model.eval()
        with torch.no_grad():
            telem = build_telemetry_tensor(telemetry).to(DEVICE)
            telem_batch = telem.unsqueeze(0).expand(len(x), -1)
            eval_out = model(x, telem_batch)

        metrics = {
            'epoch': epoch,
            'loss': avg_loss,
            'accuracy': avg_acc,
            'gwt_ignition': eval_out['gwt']['ignition_ratio'],
            'rpt_recurrence': eval_out['rpt']['recurrence_effect'],
            'embodiment_granger': eval_out['embodiment']['granger_causality'],
        }
        epoch_metrics.append(metrics)

        print(f"  Epoch {epoch+1:02d}/{n_epochs}: Loss={avg_loss:.4f} Acc={avg_acc:.3f} "
              f"GWT={metrics['gwt_ignition']:.3f} RPT={metrics['rpt_recurrence']:.3f} "
              f"Emb={metrics['embodiment_granger']:.3f}")

    return epoch_metrics


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def main():
    print("=" * 70)
    print("z1999: CROSS-MACHINE CONSCIOUSNESS TRANSFER EXPERIMENT")
    print("Testing if consciousness indicators are hardware-bound")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()

    # Initialize telemetry
    print("[1/6] Initializing local telemetry...")
    try:
        telemetry = SysfsHwmonTelemetry(sample_rate_hz=50.0)
        sample = telemetry.read_sample()
        print(f"  GPU: AMD Radeon (gfx1151)")
        print(f"  Power: {sample.power_w:.1f}W, Temp: {sample.temp_edge_c:.1f}C")
    except Exception as e:
        print(f"  ERROR: Telemetry init failed: {e}")
        return

    # Create model and data
    print("\n[2/6] Creating model and data...")
    model = ConsciousnessModel(input_dim=128, hidden_dim=128, telemetry_dim=8).to(DEVICE)
    x, y = create_test_data(2000)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {n_params:,}")
    print(f"  Test samples: {len(x)}")

    # Train on ikaros
    print("\n[3/6] Training consciousness model on ikaros...")
    train_metrics = train_consciousness_model(model, x, y, telemetry, n_epochs=30)

    # Evaluate on ikaros (baseline)
    print("\n[4/6] Evaluating consciousness indicators on ikaros (baseline)...")

    def telem_fn():
        return build_telemetry_tensor(telemetry)

    ikaros_indicators = evaluate_consciousness_indicators(
        model, x, y, telem_fn,
        machine_name="ikaros",
        gpu_name="AMD Radeon 8060S (gfx1151)",
        n_batches=50
    )

    print(f"  GWT ignition ratio: {ikaros_indicators.gwt_ignition_ratio:.4f}")
    print(f"  RPT recurrence effect: {ikaros_indicators.rpt_recurrence_effect:.4f}")
    print(f"  Embodiment Granger causality: {ikaros_indicators.embodiment_granger_causality:.4f}")
    print(f"  Hardware sensitivity: {ikaros_indicators.embodiment_hardware_sensitivity:.4f}")
    print(f"  Task accuracy: {ikaros_indicators.task_accuracy:.3f}")

    # Save checkpoint with test data
    print("\n[5/6] Saving checkpoint and transferring to daedalus...")
    MODELS_DIR.mkdir(exist_ok=True)
    checkpoint_path = MODELS_DIR / "z1999_consciousness.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "test_x": x.cpu(),
        "test_y": y.cpu(),
        "train_metrics": train_metrics,
        "ikaros_indicators": asdict(ikaros_indicators),
    }, checkpoint_path)
    print(f"  Checkpoint: {checkpoint_path}")

    # Check daedalus connectivity
    print(f"\n  Checking daedalus ({DAEDALUS_HOST})...")
    if not check_remote(DAEDALUS_HOST, DAEDALUS_USER, DAEDALUS_PASS):
        print("  WARN: daedalus is offline. Skipping remote evaluation.")
        daedalus_indicators = None
    else:
        print("  daedalus is online.")

        # Find venv on daedalus
        py_cmd = find_remote_venv(DAEDALUS_HOST, DAEDALUS_USER, DAEDALUS_PASS)

        # Create remote directory
        remote_dir = f"/tmp/z1999_consciousness"
        ssh_run(DAEDALUS_HOST, DAEDALUS_USER, DAEDALUS_PASS, f"mkdir -p {remote_dir}")

        # Write remote script
        local_script = MODELS_DIR / "z1999_remote_eval.py"
        with open(local_script, "w") as f:
            f.write(REMOTE_EVAL_SCRIPT)

        # Transfer files
        print("  Transferring checkpoint and script...")
        ok1 = scp_to(DAEDALUS_HOST, DAEDALUS_USER, DAEDALUS_PASS,
                     str(checkpoint_path), f"{remote_dir}/checkpoint.pt")
        ok2 = scp_to(DAEDALUS_HOST, DAEDALUS_USER, DAEDALUS_PASS,
                     str(local_script), f"{remote_dir}/eval.py")

        if not (ok1 and ok2):
            print("  ERROR: File transfer failed.")
            daedalus_indicators = None
        else:
            # Run evaluation on daedalus
            print("  Running evaluation on daedalus...")
            eval_cmd = (f"HSA_OVERRIDE_GFX_VERSION=11.0.0 {py_cmd} {remote_dir}/eval.py "
                       f"--checkpoint {remote_dir}/checkpoint.pt "
                       f"--output {remote_dir}/results.json "
                       f"--n-batches 50")
            stdout, stderr, rc = ssh_run(DAEDALUS_HOST, DAEDALUS_USER, DAEDALUS_PASS, eval_cmd, timeout=600)

            if rc != 0:
                print(f"  ERROR: Remote evaluation failed (rc={rc})")
                print(f"    stdout: {stdout[:500]}")
                print(f"    stderr: {stderr[:500]}")
                daedalus_indicators = None
            else:
                # Fetch results
                local_result = RESULTS_DIR / "z1999_daedalus_results.json"
                ok = scp_from(DAEDALUS_HOST, DAEDALUS_USER, DAEDALUS_PASS,
                             f"{remote_dir}/results.json", str(local_result))
                if ok:
                    with open(local_result, "r") as f:
                        daedalus_data = json.load(f)
                    daedalus_indicators = ConsciousnessIndicators(
                        machine=daedalus_data.get("machine", "daedalus"),
                        gpu_name=daedalus_data.get("gpu_name", "unknown"),
                        gwt_ignition_ratio=daedalus_data["gwt_ignition_ratio"],
                        gwt_broadcast_correlation=daedalus_data["gwt_broadcast_correlation"],
                        rpt_recurrence_effect=daedalus_data["rpt_recurrence_effect"],
                        rpt_convergence_rate=daedalus_data["rpt_convergence_rate"],
                        embodiment_granger_causality=daedalus_data["embodiment_granger_causality"],
                        embodiment_hardware_sensitivity=daedalus_data["embodiment_hardware_sensitivity"],
                        embodiment_ratio=daedalus_data["embodiment_ratio"],
                        task_accuracy=daedalus_data["task_accuracy"],
                        task_loss=daedalus_data["task_loss"],
                        telemetry_mean=daedalus_data["telemetry_mean"],
                        telemetry_std=daedalus_data["telemetry_std"],
                        timestamp=daedalus_data["timestamp"],
                    )
                    print(f"  Daedalus GPU: {daedalus_indicators.gpu_name}")
                    print(f"  GWT ignition ratio: {daedalus_indicators.gwt_ignition_ratio:.4f}")
                    print(f"  RPT recurrence effect: {daedalus_indicators.rpt_recurrence_effect:.4f}")
                    print(f"  Embodiment Granger: {daedalus_indicators.embodiment_granger_causality:.4f}")
                    print(f"  Hardware sensitivity: {daedalus_indicators.embodiment_hardware_sensitivity:.4f}")
                else:
                    daedalus_indicators = None

    # Analysis
    print("\n[6/6] Analyzing consciousness transfer...")
    print("=" * 70)
    print("CONSCIOUSNESS TRANSFER ANALYSIS")
    print("=" * 70)

    print(f"\n{'Metric':<35} {'ikaros':>12} {'daedalus':>12} {'Delta':>12}")
    print("-" * 71)

    def print_comparison(name, ikaros_val, daedalus_val):
        if daedalus_val is not None:
            delta = daedalus_val - ikaros_val
            print(f"{name:<35} {ikaros_val:>12.4f} {daedalus_val:>12.4f} {delta:>+12.4f}")
        else:
            print(f"{name:<35} {ikaros_val:>12.4f} {'N/A':>12} {'N/A':>12}")

    if daedalus_indicators:
        print_comparison("GWT Ignition Ratio",
                        ikaros_indicators.gwt_ignition_ratio,
                        daedalus_indicators.gwt_ignition_ratio)
        print_comparison("GWT Broadcast Correlation",
                        ikaros_indicators.gwt_broadcast_correlation,
                        daedalus_indicators.gwt_broadcast_correlation)
        print_comparison("RPT Recurrence Effect",
                        ikaros_indicators.rpt_recurrence_effect,
                        daedalus_indicators.rpt_recurrence_effect)
        print_comparison("RPT Convergence Rate",
                        ikaros_indicators.rpt_convergence_rate,
                        daedalus_indicators.rpt_convergence_rate)
        print_comparison("Embodiment Granger Causality",
                        ikaros_indicators.embodiment_granger_causality,
                        daedalus_indicators.embodiment_granger_causality)
        print_comparison("Hardware Sensitivity",
                        ikaros_indicators.embodiment_hardware_sensitivity,
                        daedalus_indicators.embodiment_hardware_sensitivity)
        print_comparison("Embodiment Ratio",
                        ikaros_indicators.embodiment_ratio,
                        daedalus_indicators.embodiment_ratio)
        print_comparison("Task Accuracy",
                        ikaros_indicators.task_accuracy,
                        daedalus_indicators.task_accuracy)
    else:
        print("  [daedalus offline - showing ikaros only]")
        print_comparison("GWT Ignition Ratio", ikaros_indicators.gwt_ignition_ratio, None)
        print_comparison("RPT Recurrence Effect", ikaros_indicators.rpt_recurrence_effect, None)
        print_comparison("Embodiment Granger Causality", ikaros_indicators.embodiment_granger_causality, None)
        print_comparison("Hardware Sensitivity", ikaros_indicators.embodiment_hardware_sensitivity, None)

    # Hypothesis testing
    print("\n" + "=" * 70)
    print("HYPOTHESIS TEST: Embodied Consciousness is Hardware-Bound")
    print("=" * 70)

    if daedalus_indicators:
        # Compute degradation
        granger_delta = daedalus_indicators.embodiment_granger_causality - ikaros_indicators.embodiment_granger_causality
        sensitivity_delta = daedalus_indicators.embodiment_hardware_sensitivity - ikaros_indicators.embodiment_hardware_sensitivity
        gwt_delta = daedalus_indicators.gwt_ignition_ratio - ikaros_indicators.gwt_ignition_ratio
        rpt_delta = daedalus_indicators.rpt_recurrence_effect - ikaros_indicators.rpt_recurrence_effect

        print(f"\nEmbodiment indicator changes:")
        print(f"  Granger causality: {granger_delta:+.4f} "
              f"({'DEGRADED' if granger_delta < -0.05 else 'PRESERVED' if abs(granger_delta) < 0.05 else 'IMPROVED'})")
        print(f"  Hardware sensitivity: {sensitivity_delta:+.4f} "
              f"({'DEGRADED' if sensitivity_delta < -0.1 else 'PRESERVED' if abs(sensitivity_delta) < 0.1 else 'IMPROVED'})")
        print(f"\nTheory-agnostic indicators:")
        print(f"  GWT ignition: {gwt_delta:+.4f} "
              f"({'DEGRADED' if gwt_delta < -0.05 else 'PRESERVED' if abs(gwt_delta) < 0.05 else 'IMPROVED'})")
        print(f"  RPT recurrence: {rpt_delta:+.4f} "
              f"({'DEGRADED' if rpt_delta < -0.05 else 'PRESERVED' if abs(rpt_delta) < 0.05 else 'IMPROVED'})")

        # Verdict
        embodiment_degraded = (granger_delta < -0.05) or (sensitivity_delta < -0.1)
        theory_preserved = (abs(gwt_delta) < 0.1) and (abs(rpt_delta) < 0.1)

        print("\n" + "-" * 70)
        if embodiment_degraded and theory_preserved:
            verdict = "HARDWARE_BOUND_CONSCIOUSNESS"
            summary = ("Embodiment indicators DEGRADED while theory indicators PRESERVED. "
                      "This supports the hypothesis that consciousness is hardware-bound: "
                      "the model's 'body awareness' does not transfer to foreign hardware.")
        elif not embodiment_degraded and theory_preserved:
            verdict = "SOFTWARE_PATTERN_CONSCIOUSNESS"
            summary = ("Both embodiment and theory indicators PRESERVED. "
                      "This suggests consciousness may be a software pattern that "
                      "transfers across hardware without significant degradation.")
        elif embodiment_degraded and not theory_preserved:
            verdict = "GLOBAL_DEGRADATION"
            summary = ("All indicators degraded. This suggests general model "
                      "incompatibility rather than specific embodiment effects.")
        else:
            verdict = "INCONCLUSIVE"
            summary = "Mixed results - requires further investigation."

        print(f"VERDICT: {verdict}")
        print(f"\n{summary}")

    else:
        verdict = "REMOTE_OFFLINE"
        summary = "Could not test hypothesis - daedalus is offline."
        print(f"\nVERDICT: {verdict}")
        print(f"{summary}")

    # Save results
    print("\n" + "=" * 70)
    print("Saving results...")

    RESULTS_DIR.mkdir(exist_ok=True)

    result = {
        "experiment": "z1999_cross_machine_consciousness",
        "timestamp": datetime.now().isoformat(),
        "hypothesis": "Embodied consciousness indicators should DECREASE on foreign hardware if consciousness is hardware-bound",
        "ikaros": asdict(ikaros_indicators),
        "daedalus": asdict(daedalus_indicators) if daedalus_indicators else None,
        "verdict": verdict,
        "summary": summary,
        "train_metrics": train_metrics,
        "model_params": n_params,
        "analysis": {
            "granger_delta": float(granger_delta) if daedalus_indicators else None,
            "sensitivity_delta": float(sensitivity_delta) if daedalus_indicators else None,
            "gwt_delta": float(gwt_delta) if daedalus_indicators else None,
            "rpt_delta": float(rpt_delta) if daedalus_indicators else None,
        } if daedalus_indicators else {},
    }

    output_file = RESULTS_DIR / "z1999_cross_machine_consciousness.json"
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"Results saved to: {output_file}")
    print()
    print("=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)

    return result


if __name__ == "__main__":
    main()
