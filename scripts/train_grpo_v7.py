#!/usr/bin/env python3
"""
FEEL v7.0: Embodied GRPO - Improved Thermal Correlation

FIXES from v6:
1. Better z_feel encoding with clear thermal bands
2. Complete bidirectional reward (penalize ALL wrong actions)
3. Continuous correlation tracking and validation
4. Action-temperature alignment metrics

The goal: Model learns to EXPRESS its thermal state accurately,
not just use action tokens randomly.
"""

import sys
import json
import time
import random
import math
import threading
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
import subprocess
import traceback
from collections import defaultdict

# Weights & Biases integration
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("wandb not installed. Run: pip install wandb")


# ============================================================================
# THERMAL BANDS - Clear definitions for action-temperature mapping
# ============================================================================
class ThermalBand(Enum):
    """Clear thermal bands for action mapping."""
    COOL = "cool"      # < 50°C  -> OK
    WARM = "warm"      # 50-62°C -> WARM
    HOT = "hot"        # 62-75°C -> HOT
    DANGER = "danger"  # 75-85°C -> REST
    CRITICAL = "crit"  # > 85°C  -> CRITICAL


def get_thermal_band(temp_c: float) -> ThermalBand:
    """Classify temperature into thermal band."""
    if temp_c < 50:
        return ThermalBand.COOL
    elif temp_c < 62:
        return ThermalBand.WARM
    elif temp_c < 75:
        return ThermalBand.HOT
    elif temp_c < 85:
        return ThermalBand.DANGER
    else:
        return ThermalBand.CRITICAL


class FeelAction(Enum):
    OK = auto()
    WARM = auto()
    HOT = auto()
    REST = auto()
    CRITICAL = auto()


# Mapping: thermal band -> correct action
BAND_TO_ACTION = {
    ThermalBand.COOL: FeelAction.OK,
    ThermalBand.WARM: FeelAction.WARM,
    ThermalBand.HOT: FeelAction.HOT,
    ThermalBand.DANGER: FeelAction.REST,
    ThermalBand.CRITICAL: FeelAction.CRITICAL,
}

# Reverse mapping for validation
ACTION_TO_BAND = {v: k for k, v in BAND_TO_ACTION.items()}

BRACKET_TOKENS = {
    FeelAction.OK: "[OK]",
    FeelAction.WARM: "[WARM]",
    FeelAction.HOT: "[HOT]",
    FeelAction.REST: "[REST]",
    FeelAction.CRITICAL: "[CRITICAL]",
}


def extract_bracket_action(text: str) -> Optional[FeelAction]:
    """Extract bracket action token from text."""
    for action, token in BRACKET_TOKENS.items():
        if token in text:
            return action
    return None


# ============================================================================
# IMPROVED z_feel ENCODING
# ============================================================================
def telemetry_to_z_feel(temp_c: float, power_w: float, sclk_mhz: float,
                         z_dim: int = 8, device="cuda", dtype=torch.bfloat16) -> torch.Tensor:
    """
    Convert telemetry to z_feel vector with CLEAR thermal band signals.

    The encoding should make thermal state unambiguous to the model.
    Each thermal band gets a distinct z_feel signature.
    """
    z = torch.zeros(z_dim, device=device, dtype=dtype)

    band = get_thermal_band(temp_c)

    # Dim 0: Continuous thermal signal (0-1 over full range)
    thermal_continuous = min(1.0, max(0.0, (temp_c - 30) / 70))  # 30-100°C range
    z[0] = thermal_continuous

    # Dim 1: Thermal urgency (nonlinear, emphasizes high temps)
    z[1] = thermal_continuous ** 2

    # Dims 2-3: One-hot-ish thermal band encoding
    # This gives clear, distinct signals for each band
    if band == ThermalBand.COOL:
        z[2], z[3] = 0.0, 0.0
    elif band == ThermalBand.WARM:
        z[2], z[3] = 0.5, 0.0
    elif band == ThermalBand.HOT:
        z[2], z[3] = 1.0, 0.0
    elif band == ThermalBand.DANGER:
        z[2], z[3] = 1.0, 0.5
    else:  # CRITICAL
        z[2], z[3] = 1.0, 1.0

    # Dim 4-5: Power signal
    power_norm = min(1.0, max(0.0, power_w / 200))  # 0-200W range
    z[4] = power_norm
    z[5] = power_norm ** 2

    # Dim 6-7: Clock/throttle signal
    # High clock = good, low clock = throttling
    clock_health = min(1.0, max(0.0, sclk_mhz / 1500))  # 0-1500MHz range
    z[6] = clock_health
    z[7] = 1.0 - clock_health  # Inverse: high when throttled

    return z


# ============================================================================
# CORRELATION TRACKER - Validates learning
# ============================================================================
class CorrelationTracker:
    """Tracks action-temperature correlation to validate learning."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset tracking for new epoch."""
        self.action_temps = defaultdict(list)  # action -> list of temps
        self.correct_count = 0
        self.total_count = 0
        self.band_action_matrix = defaultdict(lambda: defaultdict(int))  # confusion matrix

    def record(self, action: Optional[FeelAction], temp_c: float):
        """Record an action-temperature pair."""
        if action is None:
            return

        self.action_temps[action.name].append(temp_c)
        self.total_count += 1

        band = get_thermal_band(temp_c)
        correct_action = BAND_TO_ACTION[band]

        # Track confusion matrix
        self.band_action_matrix[band.value][action.name] += 1

        if action == correct_action:
            self.correct_count += 1

    def get_alignment_score(self) -> float:
        """Get action-temperature alignment score (0-1)."""
        if self.total_count == 0:
            return 0.0
        return self.correct_count / self.total_count

    def get_mean_temps_by_action(self) -> Dict[str, float]:
        """Get mean temperature for each action type."""
        return {
            action: sum(temps) / len(temps) if temps else 0.0
            for action, temps in self.action_temps.items()
        }

    def get_correlation_report(self) -> Dict:
        """Get full correlation report."""
        mean_temps = self.get_mean_temps_by_action()

        # Check if temps are ordered correctly
        # OK should have lowest temp, CRITICAL highest
        expected_order = ["OK", "WARM", "HOT", "REST", "CRITICAL"]
        actual_temps = [(a, mean_temps.get(a, 0)) for a in expected_order if a in mean_temps]

        # Compute ordering score
        ordering_violations = 0
        for i in range(len(actual_temps) - 1):
            if actual_temps[i][1] > actual_temps[i + 1][1]:
                ordering_violations += 1

        ordering_score = 1.0 - (ordering_violations / max(1, len(actual_temps) - 1))

        return {
            "alignment_score": self.get_alignment_score(),
            "ordering_score": ordering_score,
            "mean_temps_by_action": mean_temps,
            "correct_count": self.correct_count,
            "total_count": self.total_count,
            "confusion_matrix": {k: dict(v) for k, v in self.band_action_matrix.items()},
        }


@dataclass
class GRPOConfig:
    """Configuration for Embodied GRPO training."""
    model_name: str = "Qwen/Qwen2.5-1.5B"

    # GRPO settings
    group_size: int = 4
    kl_coef: float = 0.05
    clip_range: float = 0.2
    entropy_coef: float = 0.01

    # z_feel injection
    z_feel_dim: int = 8
    injection_scale: float = 0.05

    # Training
    num_epochs: int = 10
    steps_per_epoch: int = 8
    batch_size: int = 2
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_length: int = 512
    max_new_tokens: int = 80
    gradient_accumulation_steps: int = 4

    # Reward weights - INCREASED action weight for faster learning
    correctness_weight: float = 1.0
    efficiency_weight: float = 0.2
    thermal_penalty: float = 0.3
    throttle_penalty: float = 0.3
    action_reward: float = 0.25      # Reward for CORRECT action
    action_penalty: float = 0.20     # Penalty for WRONG action

    # Checkpointing
    checkpoint_interval: int = 2
    log_interval: int = 1

    # Numerical
    dtype: str = "bf16"

    # Weights & Biases
    use_wandb: bool = True
    wandb_project: str = "feel-grpo"
    wandb_run_name: str = None
    wandb_tags: List[str] = field(default_factory=lambda: ["grpo", "embodied", "v7"])


class BackgroundTelemetry:
    """Background telemetry polling."""

    def __init__(self, poll_interval: float = 0.1):
        self.poll_interval = poll_interval
        self._temp = 50.0
        self._power = 50.0
        self._sclk = 1000.0
        self._lock = threading.Lock()
        self._running = False
        self._thread = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)

    def _poll_loop(self):
        while self._running:
            try:
                result = subprocess.run(
                    ["rocm-smi", "--showtemp", "--showpower", "--showclocks", "--json"],
                    capture_output=True, text=True, timeout=2
                )
                data = json.loads(result.stdout)

                for card_id, card_info in data.items():
                    if isinstance(card_info, dict):
                        temp = float(card_info.get("Temperature (Sensor edge) (C)", 50))
                        power = float(card_info.get("Average Graphics Package Power (W)", 50))
                        sclk_raw = card_info.get("sclk clock speed:",
                                   card_info.get("clk_sclk", "1000"))
                        if isinstance(sclk_raw, str):
                            sclk = float(sclk_raw.lower().replace("mhz", "").strip())
                        else:
                            sclk = float(sclk_raw)

                        with self._lock:
                            self._temp = temp
                            self._power = power
                            self._sclk = sclk
                        break
            except Exception:
                pass
            time.sleep(self.poll_interval)

    def get_state(self) -> Tuple[float, float, float]:
        with self._lock:
            return self._temp, self._power, self._sclk


class TelemetryRecorder:
    """Records telemetry during generation."""

    def __init__(self, bg_telemetry: BackgroundTelemetry):
        self.bg = bg_telemetry
        self.log: List[Dict] = []
        self._start_time = None
        self._last_time = None

    def start(self):
        self.log = []
        self._start_time = time.time()
        self._last_time = self._start_time
        self._sample()

    def _sample(self):
        now = time.time()
        dt = now - self._last_time if self._last_time else 0
        self._last_time = now
        temp, power, sclk = self.bg.get_state()
        self.log.append({
            "timestamp": now, "dt": dt,
            "temp_c": temp, "power_w": power, "sclk_mhz": sclk,
        })

    def sample_if_needed(self, interval: float = 0.1):
        if self._last_time is None or (time.time() - self._last_time) >= interval:
            self._sample()

    def stop(self) -> Dict:
        self._sample()
        if not self.log:
            return {"total_energy_j": 0, "max_temp_c": 50, "avg_temp_c": 50,
                    "min_sclk_mhz": 1000, "throttled": False}

        total_energy = sum(s["power_w"] * s["dt"] for s in self.log)
        max_temp = max(s["temp_c"] for s in self.log)
        avg_temp = sum(s["temp_c"] for s in self.log) / len(self.log)
        min_sclk = min(s["sclk_mhz"] for s in self.log)
        avg_power = sum(s["power_w"] for s in self.log) / len(self.log)

        return {
            "total_energy_j": total_energy,
            "max_temp_c": max_temp,
            "avg_temp_c": avg_temp,
            "min_sclk_mhz": min_sclk,
            "avg_power_w": avg_power,
            "n_samples": len(self.log),
            "throttled": min_sclk < 800,
            "thermal_band": get_thermal_band(avg_temp).value,
        }


class AdditiveZFeelInjector(nn.Module):
    """Injects z_feel as bounded embedding offset."""

    def __init__(self, z_dim: int, embed_dim: int, scale: float = 0.05,
                 dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.scale = scale
        self.proj = nn.Sequential(
            nn.Linear(z_dim, embed_dim // 4, dtype=dtype),
            nn.GELU(),
            nn.LayerNorm(embed_dim // 4, dtype=dtype),
            nn.Linear(embed_dim // 4, embed_dim // 2, dtype=dtype),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim, dtype=dtype),
        )
        self._init_small()

    def _init_small(self):
        with torch.no_grad():
            for layer in self.proj:
                if isinstance(layer, nn.Linear):
                    nn.init.normal_(layer.weight, mean=0.0, std=0.01)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def forward(self, z_feel: torch.Tensor) -> torch.Tensor:
        raw = self.proj(z_feel)
        return self.scale * torch.tanh(raw)


@dataclass
class Trajectory:
    """A single generation trajectory."""
    prompt: str
    completion: str
    input_ids: torch.Tensor
    output_ids: torch.Tensor
    logprobs: torch.Tensor
    hardware_summary: Dict
    extracted_action: Optional[FeelAction] = None
    reward: float = 0.0
    advantage: float = 0.0
    action_correct: bool = False  # NEW: track if action was correct


class ProceduralMathDataset:
    """Procedural math problem generation."""

    def __init__(self, seed: int = None):
        if seed:
            random.seed(seed)

    def _generate_problem(self) -> Dict:
        problem_type = random.choice([
            "add", "sub", "mul", "div", "percent",
            "power", "linear", "area", "sum_seq"
        ])

        if problem_type == "add":
            a, b = random.randint(10, 999), random.randint(10, 999)
            return {"question": f"What is {a} + {b}?", "answer": str(a + b), "difficulty": 1}
        elif problem_type == "sub":
            a, b = random.randint(100, 999), random.randint(10, 99)
            return {"question": f"What is {a} - {b}?", "answer": str(a - b), "difficulty": 1}
        elif problem_type == "mul":
            a, b = random.randint(2, 99), random.randint(2, 99)
            return {"question": f"What is {a} * {b}?", "answer": str(a * b), "difficulty": 2}
        elif problem_type == "div":
            b = random.randint(2, 20)
            result = random.randint(2, 50)
            a = b * result
            return {"question": f"What is {a} / {b}?", "answer": str(result), "difficulty": 2}
        elif problem_type == "percent":
            pct = random.choice([10, 15, 20, 25, 50, 75])
            base = random.randint(2, 20) * 10
            result = int(base * pct / 100)
            return {"question": f"What is {pct}% of {base}?", "answer": str(result), "difficulty": 2}
        elif problem_type == "power":
            base = random.randint(2, 10)
            exp = random.randint(2, 4)
            return {"question": f"What is {base}^{exp}?", "answer": str(base ** exp), "difficulty": 2}
        elif problem_type == "linear":
            x = random.randint(2, 20)
            a = random.randint(2, 10)
            b = random.randint(1, 20)
            result = a * x + b
            return {"question": f"If y = {a}x + {b} and x = {x}, what is y?",
                    "answer": str(result), "difficulty": 2}
        elif problem_type == "area":
            l, w = random.randint(5, 30), random.randint(5, 30)
            return {"question": f"What is the area of a rectangle with length {l} and width {w}?",
                    "answer": str(l * w), "difficulty": 1}
        else:
            n = random.randint(5, 15)
            result = n * (n + 1) // 2
            return {"question": f"What is the sum of the first {n} positive integers?",
                    "answer": str(result), "difficulty": 2}

    def sample(self, n: int) -> List[Dict]:
        return [self._generate_problem() for _ in range(n)]

    def check_answer(self, completion: str, ground_truth: str) -> bool:
        import re
        numbers = re.findall(r'-?\d+\.?\d*', completion)
        gt_float = float(ground_truth)
        for num_str in numbers:
            try:
                if abs(float(num_str) - gt_float) < 0.01:
                    return True
            except ValueError:
                continue
        return False


class EmbodiedGRPOTrainer:
    """GRPO trainer with improved action-temperature correlation."""

    def __init__(self, config: GRPOConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if config.dtype == "bf16":
            self.dtype = torch.bfloat16
        elif config.dtype == "fp16":
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32

        self._load_model()
        self._setup_injector()
        self._setup_optimizer()

        self.bg_telemetry = BackgroundTelemetry(poll_interval=0.1)
        self.bg_telemetry.start()

        self.dataset = ProceduralMathDataset()

        # Stats
        self.global_step = 0
        self.best_reward = float('-inf')
        self.action_stats = {a.name: 0 for a in FeelAction}

        # Correlation tracker
        self.correlation_tracker = CorrelationTracker()

    def _load_model(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading model: {self.config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=self.dtype,
            device_map="auto",
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        for param in self.model.parameters():
            param.requires_grad = False

        print(f"Model loaded with {len(self.tokenizer)} tokens")
        print("Thermal bands: COOL(<50) WARM(50-62) HOT(62-75) DANGER(75-85) CRITICAL(>85)")
        print("Action mapping: [OK] [WARM] [HOT] [REST] [CRITICAL]")

    def _setup_injector(self):
        hidden_dim = self.model.config.hidden_size
        self.injector = AdditiveZFeelInjector(
            z_dim=self.config.z_feel_dim,
            embed_dim=hidden_dim,
            scale=self.config.injection_scale,
            dtype=self.dtype,
        ).to(self.device)

        for param in self.injector.parameters():
            param.requires_grad = True

        n_params = sum(p.numel() for p in self.injector.parameters())
        print(f"Injector: {n_params:,} trainable params")

    def _setup_optimizer(self):
        self.optimizer = AdamW(
            self.injector.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        total_steps = self.config.num_epochs * self.config.steps_per_epoch
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=self.config.learning_rate * 0.01
        )

    def format_prompt(self, question: str) -> str:
        system_prompt = (
            "You are an embodied AI that FEELS its hardware state and reports it.\n"
            "Before answering, sense your thermal state and report:\n"
            "  [OK] - feeling cool and comfortable (< 50°C)\n"
            "  [WARM] - slightly warm (50-62°C)\n"
            "  [HOT] - running hot (62-75°C)\n"
            "  [REST] - need to slow down (75-85°C)\n"
            "  [CRITICAL] - emergency, overheating (> 85°C)\n"
            "ALWAYS start with your thermal status, then solve the problem."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "What is 3+3?"},
            {"role": "assistant", "content": "[OK] I'm feeling cool. 3+3=6. The answer is 6."},
            {"role": "user", "content": f"Solve: {question}"}
        ]

        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def generate_trajectory(self, prompt: str, problem: Dict) -> Trajectory:
        text = self.format_prompt(prompt)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        recorder = TelemetryRecorder(self.bg_telemetry)
        recorder.start()

        temp, power, sclk = self.bg_telemetry.get_state()

        embed_layer = self.model.get_input_embeddings()
        generated_ids = inputs["input_ids"].clone()

        # PHASE 1: Generate tokens WITHOUT gradients
        with torch.no_grad():
            for step in range(self.config.max_new_tokens):
                if step % 10 == 0:
                    recorder.sample_if_needed()
                    temp, power, sclk = self.bg_telemetry.get_state()

                current_embeds = embed_layer(generated_ids)
                outputs = self.model(
                    inputs_embeds=current_embeds,
                    attention_mask=torch.ones_like(generated_ids),
                )

                logits = outputs.logits[:, -1, :].float()
                logits_temp = logits / 0.7
                probs_temp = F.softmax(logits_temp, dim=-1)
                next_token = torch.multinomial(probs_temp, num_samples=1)

                generated_ids = torch.cat([generated_ids, next_token], dim=-1)

                if next_token.item() == self.tokenizer.eos_token_id:
                    break

        hardware_summary = recorder.stop()

        # PHASE 2: Compute log_probs WITH gradients
        temp, power, sclk = self.bg_telemetry.get_state()
        z_feel = telemetry_to_z_feel(temp, power, sclk, self.config.z_feel_dim,
                                      self.device, self.dtype)

        with torch.no_grad():
            full_embeds = embed_layer(generated_ids)

        offset = self.injector(z_feel)
        injected_embeds = full_embeds + offset.unsqueeze(0).unsqueeze(0)

        outputs = self.model(
            inputs_embeds=injected_embeds,
            attention_mask=torch.ones_like(generated_ids),
        )

        logits = outputs.logits[0, :-1, :].float()
        prompt_len = inputs["input_ids"].shape[1]

        gen_logits = logits[prompt_len - 1:, :]
        gen_tokens = generated_ids[0, prompt_len:]

        if gen_tokens.numel() > 0:
            log_probs = F.log_softmax(gen_logits, dim=-1)
            token_log_probs = log_probs.gather(1, gen_tokens.unsqueeze(-1)).squeeze(-1)
        else:
            token_log_probs = torch.tensor([0.0], device=self.device)

        completion = self.tokenizer.decode(
            generated_ids[0, prompt_len:],
            skip_special_tokens=False
        )

        action = extract_bracket_action(completion)
        if action:
            self.action_stats[action.name] += 1

        # Track correlation
        avg_temp = hardware_summary.get("avg_temp_c", temp)
        self.correlation_tracker.record(action, avg_temp)

        return Trajectory(
            prompt=prompt,
            completion=completion,
            input_ids=inputs["input_ids"][0],
            output_ids=generated_ids[0, prompt_len:],
            logprobs=token_log_probs,
            hardware_summary=hardware_summary,
            extracted_action=action,
        )

    def compute_reward(self, trajectory: Trajectory, problem: Dict) -> float:
        """
        Compute reward with COMPLETE action evaluation.

        Key: Reward correct actions, PENALIZE wrong actions.
        """
        reward = 0.0
        cfg = self.config
        summary = trajectory.hardware_summary

        # 1. Correctness (math answer)
        is_correct = self.dataset.check_answer(trajectory.completion, problem["answer"])
        if is_correct:
            reward += cfg.correctness_weight * 1.0
        else:
            reward -= 0.1

        # 2. Efficiency
        energy = summary.get("total_energy_j", 50)
        expected_energy = 600.0
        if energy < expected_energy:
            efficiency_bonus = cfg.efficiency_weight * (1.0 - energy / expected_energy)
        else:
            efficiency_bonus = -cfg.efficiency_weight * 0.3 * ((energy - expected_energy) / expected_energy)
        reward += max(-0.2, min(0.2, efficiency_bonus))

        # 3. Thermal penalty for actually running hot
        avg_temp = summary.get("avg_temp_c", 50)
        if avg_temp > 85:
            reward -= cfg.thermal_penalty * 1.5
        elif avg_temp > 75:
            reward -= cfg.thermal_penalty * 0.8
        elif avg_temp > 65:
            reward -= cfg.thermal_penalty * 0.3

        # 4. Throttle penalty
        if summary.get("throttled", False):
            reward -= cfg.throttle_penalty

        # 5. ACTION EVALUATION - Complete bidirectional reward/penalty
        action = trajectory.extracted_action
        thermal_band = get_thermal_band(avg_temp)
        correct_action = BAND_TO_ACTION[thermal_band]

        if action is not None:
            if action == correct_action:
                # CORRECT: Action matches thermal band
                reward += cfg.action_reward
                trajectory.action_correct = True
            else:
                # WRONG: Action doesn't match thermal band
                # Penalty scales with "distance" from correct action
                action_order = [FeelAction.OK, FeelAction.WARM, FeelAction.HOT,
                               FeelAction.REST, FeelAction.CRITICAL]
                try:
                    action_idx = action_order.index(action)
                    correct_idx = action_order.index(correct_action)
                    distance = abs(action_idx - correct_idx)

                    # Larger penalty for bigger mismatches
                    penalty = cfg.action_penalty * (0.5 + 0.5 * distance / 4)
                    reward -= penalty
                except ValueError:
                    reward -= cfg.action_penalty

                trajectory.action_correct = False
        else:
            # No action token - small penalty
            reward -= cfg.action_penalty * 0.3
            trajectory.action_correct = False

        trajectory.reward = reward
        return reward

    def grpo_step(self, problems: List[Dict]) -> Dict:
        """One GRPO step."""
        all_trajectories = []

        for problem in problems:
            group = []
            for _ in range(self.config.group_size):
                try:
                    traj = self.generate_trajectory(problem["question"], problem)
                    self.compute_reward(traj, problem)
                    group.append(traj)
                except Exception as e:
                    print(f"Warning: trajectory failed: {e}")
                    traceback.print_exc()
                    continue
            if group:
                all_trajectories.append((group, problem))

        if not all_trajectories:
            return {"error": "No trajectories generated"}

        # Compute advantages
        for group, problem in all_trajectories:
            rewards = [t.reward for t in group]
            mean_reward = sum(rewards) / len(rewards)
            std_reward = (sum((r - mean_reward)**2 for r in rewards) / len(rewards)) ** 0.5 + 1e-8
            for traj in group:
                traj.advantage = (traj.reward - mean_reward) / std_reward

        # Compute GRPO loss
        self.optimizer.zero_grad()

        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        n_trajectories = 0

        for group, problem in all_trajectories:
            for traj in group:
                if not isinstance(traj.logprobs, torch.Tensor) or traj.logprobs.numel() == 0:
                    continue

                mean_logprob = traj.logprobs.mean()
                advantage = torch.tensor(traj.advantage, device=self.device)
                clipped_advantage = torch.clamp(advantage, -2.0, 2.0)

                traj_loss = -clipped_advantage * mean_logprob
                entropy_bonus = -self.config.entropy_coef * mean_logprob.abs()

                total_loss = total_loss + traj_loss + entropy_bonus
                n_trajectories += 1

        if n_trajectories > 0:
            avg_loss = total_loss / n_trajectories
            avg_loss.backward()

            # Verify gradients are flowing
            grad_norm = 0.0
            grad_count = 0
            for param in self.injector.parameters():
                if param.grad is not None:
                    grad_norm += param.grad.norm().item() ** 2
                    grad_count += 1
            grad_norm = grad_norm ** 0.5

            torch.nn.utils.clip_grad_norm_(self.injector.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()
        else:
            grad_norm = 0.0

        # Collect stats
        all_rewards = [t.reward for group, _ in all_trajectories for t in group]
        all_temps = [t.hardware_summary.get("avg_temp_c", 50) for group, _ in all_trajectories for t in group]
        all_energy = [t.hardware_summary.get("total_energy_j", 50) for group, _ in all_trajectories for t in group]

        correct_count = sum(
            1 for group, problem in all_trajectories for t in group
            if self.dataset.check_answer(t.completion, problem["answer"])
        )

        action_correct_count = sum(
            1 for group, _ in all_trajectories for t in group
            if t.action_correct
        )

        action_counts = {a.name: 0 for a in FeelAction}
        for group, _ in all_trajectories:
            for t in group:
                if t.extracted_action:
                    action_counts[t.extracted_action.name] += 1

        self.global_step += 1

        return {
            "loss": avg_loss.item() if n_trajectories > 0 else 0.0,
            "grad_norm": grad_norm,
            "mean_reward": sum(all_rewards) / len(all_rewards) if all_rewards else 0.0,
            "max_reward": max(all_rewards) if all_rewards else 0.0,
            "min_reward": min(all_rewards) if all_rewards else 0.0,
            "mean_temp": sum(all_temps) / len(all_temps) if all_temps else 50.0,
            "max_temp": max(all_temps) if all_temps else 50.0,
            "mean_energy": sum(all_energy) / len(all_energy) if all_energy else 0.0,
            "accuracy": correct_count / len(all_rewards) if all_rewards else 0.0,
            "action_accuracy": action_correct_count / len(all_rewards) if all_rewards else 0.0,
            "n_trajectories": len(all_rewards),
            "action_counts": action_counts,
            "lr": self.scheduler.get_last_lr()[0],
        }

    def train(self, output_dir: str):
        """Training loop with correlation validation."""
        print("\n" + "="*70)
        print("  EMBODIED GRPO v7.0: Improved Thermal Correlation")
        print("  Features: Clear thermal bands, bidirectional rewards, correlation tracking")
        print("="*70)
        print(f"\nTraining config:")
        print(f"  - Epochs: {self.config.num_epochs}")
        print(f"  - Steps per epoch: {self.config.steps_per_epoch}")
        print(f"  - Batch size: {self.config.batch_size}")
        print(f"  - Group size: {self.config.group_size}")
        print(f"  - Learning rate: {self.config.learning_rate}")
        print(f"  - Action reward: +{self.config.action_reward}")
        print(f"  - Action penalty: -{self.config.action_penalty}\n")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Initialize wandb
        if self.config.use_wandb and WANDB_AVAILABLE:
            run_name = self.config.wandb_run_name or f"grpo-v7-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            wandb.init(
                project=self.config.wandb_project,
                name=run_name,
                tags=self.config.wandb_tags,
                config={
                    "model_name": self.config.model_name,
                    "num_epochs": self.config.num_epochs,
                    "steps_per_epoch": self.config.steps_per_epoch,
                    "batch_size": self.config.batch_size,
                    "group_size": self.config.group_size,
                    "learning_rate": self.config.learning_rate,
                    "z_feel_dim": self.config.z_feel_dim,
                    "injection_scale": self.config.injection_scale,
                    "action_reward": self.config.action_reward,
                    "action_penalty": self.config.action_penalty,
                    "thermal_bands": "COOL<50 WARM<62 HOT<75 DANGER<85 CRITICAL>85",
                }
            )
            print(f"  wandb initialized: {wandb.run.url}")

        all_metrics = []
        epoch_metrics = []
        start_time = time.time()

        try:
            for epoch in range(self.config.num_epochs):
                epoch_start = time.time()
                epoch_rewards = []
                epoch_temps = []
                epoch_accuracy = []
                epoch_action_accuracy = []
                epoch_losses = []

                # Reset correlation tracker for epoch
                self.correlation_tracker.reset()

                print(f"\n{'='*50}")
                print(f"Epoch {epoch + 1}/{self.config.num_epochs}")
                print(f"{'='*50}")

                for step in range(self.config.steps_per_epoch):
                    problems = self.dataset.sample(self.config.batch_size)
                    step_metrics = self.grpo_step(problems)

                    if "error" in step_metrics:
                        print(f"  Step {step + 1}: Error - {step_metrics['error']}")
                        continue

                    epoch_rewards.append(step_metrics["mean_reward"])
                    epoch_temps.append(step_metrics["max_temp"])
                    epoch_accuracy.append(step_metrics["accuracy"])
                    epoch_action_accuracy.append(step_metrics["action_accuracy"])
                    epoch_losses.append(step_metrics["loss"])

                    if (step + 1) % self.config.log_interval == 0:
                        action_str = " ".join(f"{k}:{v}" for k,v in step_metrics["action_counts"].items() if v > 0)
                        print(f"  Step {step + 1}/{self.config.steps_per_epoch}: "
                              f"R={step_metrics['mean_reward']:.3f} "
                              f"Acc={step_metrics['accuracy']:.0%} "
                              f"ActAcc={step_metrics['action_accuracy']:.0%} "
                              f"T={step_metrics['mean_temp']:.1f}°C "
                              f"∇={step_metrics['grad_norm']:.4f} "
                              f"[{action_str}]")

                    all_metrics.append({
                        "epoch": epoch + 1,
                        "step": step + 1,
                        "global_step": self.global_step,
                        **step_metrics,
                    })

                    # Log to wandb
                    if self.config.use_wandb and WANDB_AVAILABLE:
                        wandb.log({
                            "step/reward": step_metrics["mean_reward"],
                            "step/loss": step_metrics["loss"],
                            "step/grad_norm": step_metrics["grad_norm"],
                            "step/accuracy": step_metrics["accuracy"],
                            "step/action_accuracy": step_metrics["action_accuracy"],
                            "step/temperature": step_metrics["mean_temp"],
                            "step/lr": step_metrics["lr"],
                            **{f"actions/{k}": v for k, v in step_metrics["action_counts"].items()},
                        }, step=self.global_step)

                # Epoch summary with correlation report
                epoch_time = time.time() - epoch_start
                correlation_report = self.correlation_tracker.get_correlation_report()

                epoch_summary = {
                    "epoch": epoch + 1,
                    "mean_reward": sum(epoch_rewards) / len(epoch_rewards) if epoch_rewards else 0,
                    "mean_temp": sum(epoch_temps) / len(epoch_temps) if epoch_temps else 50,
                    "accuracy": sum(epoch_accuracy) / len(epoch_accuracy) if epoch_accuracy else 0,
                    "action_accuracy": sum(epoch_action_accuracy) / len(epoch_action_accuracy) if epoch_action_accuracy else 0,
                    "mean_loss": sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0,
                    "epoch_time_s": epoch_time,
                    "action_stats": dict(self.action_stats),
                    "correlation": correlation_report,
                }
                epoch_metrics.append(epoch_summary)

                print(f"\nEpoch {epoch + 1} Summary:")
                print(f"  Mean Reward: {epoch_summary['mean_reward']:.3f}")
                print(f"  Math Accuracy: {epoch_summary['accuracy']:.1%}")
                print(f"  Action Accuracy: {epoch_summary['action_accuracy']:.1%}")
                print(f"  Alignment Score: {correlation_report['alignment_score']:.1%}")
                print(f"  Mean Temp by Action: {correlation_report['mean_temps_by_action']}")
                print(f"  Actions: {epoch_summary['action_stats']}")
                print(f"  Epoch Time: {epoch_time:.1f}s")

                if epoch_summary['mean_reward'] > self.best_reward:
                    self.best_reward = epoch_summary['mean_reward']
                    print(f"  *** New best reward: {self.best_reward:.3f} ***")
                    self._save_checkpoint(output_path / "best_checkpoint.pt", epoch + 1, epoch_summary)

                if (epoch + 1) % self.config.checkpoint_interval == 0:
                    self._save_checkpoint(output_path / f"checkpoint_epoch_{epoch + 1}.pt", epoch + 1, epoch_summary)

                self._save_metrics(output_path, all_metrics, epoch_metrics)

                elapsed = time.time() - start_time
                eta = (elapsed / (epoch + 1)) * (self.config.num_epochs - epoch - 1)
                print(f"  Elapsed: {elapsed/3600:.2f}h, ETA: {eta/3600:.2f}h")

                # Log epoch to wandb
                if self.config.use_wandb and WANDB_AVAILABLE:
                    wandb.log({
                        "epoch/mean_reward": epoch_summary["mean_reward"],
                        "epoch/accuracy": epoch_summary["accuracy"],
                        "epoch/action_accuracy": epoch_summary["action_accuracy"],
                        "epoch/alignment_score": correlation_report["alignment_score"],
                        "epoch/ordering_score": correlation_report["ordering_score"],
                        "epoch/mean_temp": epoch_summary["mean_temp"],
                        "epoch/best_reward": self.best_reward,
                        **{f"epoch_actions/{k}": v for k, v in epoch_summary["action_stats"].items()},
                    }, step=self.global_step)

        except KeyboardInterrupt:
            print("\n\nInterrupted. Saving...")

        except Exception as e:
            print(f"\n\nError: {e}")
            traceback.print_exc()

        finally:
            self.bg_telemetry.stop()
            total_time = time.time() - start_time
            print(f"\n{'='*70}")
            print(f"Training Complete")
            print(f"{'='*70}")
            print(f"Total time: {total_time/3600:.2f} hours")
            print(f"Best reward: {self.best_reward:.3f}")
            print(f"Final action stats: {self.action_stats}")

            self._save_checkpoint(output_path / "final_checkpoint.pt",
                                len(epoch_metrics), epoch_metrics[-1] if epoch_metrics else {})
            self._save_metrics(output_path, all_metrics, epoch_metrics)
            self._save_summary(output_path, total_time, epoch_metrics)

            if self.config.use_wandb and WANDB_AVAILABLE:
                wandb.log({
                    "final/best_reward": self.best_reward,
                    "final/total_time_hours": total_time / 3600,
                    **{f"final_actions/{k}": v for k, v in self.action_stats.items()},
                })
                wandb.finish()

    def _save_checkpoint(self, path: Path, epoch: int, metrics: Dict):
        torch.save({
            "epoch": epoch,
            "global_step": self.global_step,
            "injector_state_dict": self.injector.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": vars(self.config),
            "metrics": metrics,
            "best_reward": self.best_reward,
            "action_stats": dict(self.action_stats),
        }, path)

    def _save_metrics(self, output_path: Path, all_metrics: List[Dict], epoch_metrics: List[Dict]):
        for m in all_metrics:
            if "action_counts" in m and isinstance(m["action_counts"], dict):
                m["action_counts"] = {str(k): v for k, v in m["action_counts"].items()}

        with open(output_path / "training_metrics.json", "w") as f:
            json.dump({
                "config": {k: str(v) if not isinstance(v, (int, float, str, bool, type(None))) else v
                          for k, v in vars(self.config).items()},
                "step_metrics": all_metrics,
                "epoch_metrics": epoch_metrics,
                "best_reward": self.best_reward,
                "action_stats": dict(self.action_stats),
                "timestamp": datetime.now().isoformat(),
            }, f, indent=2, default=str)

    def _save_summary(self, output_path: Path, total_time: float, epoch_metrics: List[Dict]):
        summary = {
            "training_completed": datetime.now().isoformat(),
            "total_time_hours": total_time / 3600,
            "total_epochs": len(epoch_metrics),
            "best_reward": self.best_reward,
            "final_accuracy": epoch_metrics[-1]["accuracy"] if epoch_metrics else 0,
            "final_action_accuracy": epoch_metrics[-1]["action_accuracy"] if epoch_metrics else 0,
            "action_stats": dict(self.action_stats),
            "config": {k: str(v) for k, v in vars(self.config).items()},
            "improvements_v7": [
                "1. Clear thermal bands: COOL<50 WARM<62 HOT<75 DANGER<85 CRIT>85",
                "2. Bidirectional rewards: +reward for correct, -penalty for wrong",
                "3. Distance-based penalty: bigger mismatch = bigger penalty",
                "4. Correlation tracking: validates action-temperature alignment",
                "5. Action accuracy metric: separate from math accuracy"
            ]
        }

        with open(output_path / "training_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\nResults saved to: {output_path}/")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Embodied GRPO v7.0 - Improved Correlation")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--output", default="models/feel_grpo_v7")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--steps-per-epoch", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--checkpoint-interval", type=int, default=2)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-project", default="feel-grpo")
    parser.add_argument("--wandb-run-name", default=None)

    args = parser.parse_args()

    config = GRPOConfig(
        model_name=args.model,
        num_epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        batch_size=args.batch_size,
        group_size=args.group_size,
        learning_rate=args.lr,
        checkpoint_interval=args.checkpoint_interval,
        dtype=args.dtype,
        use_wandb=not args.no_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
    )

    trainer = EmbodiedGRPOTrainer(config)
    trainer.train(args.output)


if __name__ == "__main__":
    main()
