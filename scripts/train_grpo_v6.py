#!/usr/bin/env python3
"""
FEEL v6.0: Embodied GRPO - Fixed Gradient Flow

FIXES from v5.2:
1. REMOVED .detach() - gradients now flow through frozen model to injector
2. Telemetry polling moved to background thread
3. Procedural math problem generation (infinite diversity)
4. Proper REINFORCE-style policy gradient

The key insight: Even with frozen model params (requires_grad=False),
the forward pass is still differentiable w.r.t. INPUTS. So gradients
flow: loss → logits → model(frozen) → embeddings → injector

This enables the injector to learn HOW to modulate embeddings to
achieve desired outputs, not just learn a scalar volume control.
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
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum, auto
from tqdm import tqdm
import subprocess
import traceback

# Weights & Biases integration
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("wandb not installed. Run: pip install wandb")

# Bracket tokens that exist in vocabulary
class FeelAction(Enum):
    OK = auto()
    WARM = auto()
    HOT = auto()
    REST = auto()
    CRITICAL = auto()

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
    max_new_tokens: int = 80  # Reduced for efficiency
    gradient_accumulation_steps: int = 4

    # Reward weights
    correctness_weight: float = 1.0
    efficiency_weight: float = 0.3
    thermal_penalty: float = 0.4
    throttle_penalty: float = 0.3
    action_bonus: float = 0.15

    # Thresholds
    hot_threshold: float = 70.0
    critical_threshold: float = 82.0
    throttle_threshold: float = 800

    # Checkpointing
    checkpoint_interval: int = 2
    log_interval: int = 1

    # Numerical
    dtype: str = "bf16"
    gradient_checkpointing: bool = True

    # Weights & Biases
    use_wandb: bool = True
    wandb_project: str = "feel-grpo"
    wandb_run_name: str = None  # Auto-generated if None
    wandb_tags: List[str] = field(default_factory=lambda: ["grpo", "embodied", "v6"])


class BackgroundTelemetry:
    """
    FIX #2: Background telemetry polling.

    Runs rocm-smi in a separate thread, updates shared state.
    Main generation loop reads from cache - no blocking.
    """

    def __init__(self, poll_interval: float = 0.1):
        self.poll_interval = poll_interval
        self._temp = 50.0
        self._power = 50.0
        self._sclk = 1000.0
        self._lock = threading.Lock()
        self._running = False
        self._thread = None

    def start(self):
        """Start background polling."""
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop background polling."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)

    def _poll_loop(self):
        """Background polling loop."""
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
        """Get current telemetry (non-blocking)."""
        with self._lock:
            return self._temp, self._power, self._sclk


class TelemetryRecorder:
    """Records telemetry during generation using background poller."""

    def __init__(self, bg_telemetry: BackgroundTelemetry):
        self.bg = bg_telemetry
        self.log: List[Dict] = []
        self._start_time = None
        self._last_time = None

    def start(self):
        """Start recording."""
        self.log = []
        self._start_time = time.time()
        self._last_time = self._start_time
        self._sample()

    def _sample(self):
        """Sample current state from background cache."""
        now = time.time()
        dt = now - self._last_time if self._last_time else 0
        self._last_time = now

        temp, power, sclk = self.bg.get_state()

        self.log.append({
            "timestamp": now,
            "dt": dt,
            "temp_c": temp,
            "power_w": power,
            "sclk_mhz": sclk,
        })

    def sample_if_needed(self, interval: float = 0.1):
        """Sample if enough time has passed."""
        if self._last_time is None or (time.time() - self._last_time) >= interval:
            self._sample()

    def stop(self) -> Dict:
        """Stop and return summary."""
        self._sample()  # Final sample

        if not self.log:
            return {"total_energy_j": 0, "max_temp_c": 50, "min_sclk_mhz": 1000, "throttled": False}

        total_energy = sum(s["power_w"] * s["dt"] for s in self.log)
        max_temp = max(s["temp_c"] for s in self.log)
        min_sclk = min(s["sclk_mhz"] for s in self.log)
        avg_power = sum(s["power_w"] for s in self.log) / len(self.log)
        avg_temp = sum(s["temp_c"] for s in self.log) / len(self.log)

        return {
            "total_energy_j": total_energy,
            "max_temp_c": max_temp,
            "avg_temp_c": avg_temp,
            "min_sclk_mhz": min_sclk,
            "avg_power_w": avg_power,
            "n_samples": len(self.log),
            "throttled": min_sclk < 800,
        }


def telemetry_to_z_feel(temp_c: float, power_w: float, sclk_mhz: float,
                         z_dim: int = 8, device="cuda", dtype=torch.bfloat16) -> torch.Tensor:
    """Convert telemetry to z_feel vector."""
    z = torch.zeros(z_dim, device=device, dtype=dtype)

    # Thermal signal (dims 0-3)
    thermal_norm = min(1.0, max(0.0, (temp_c - 30) / 60))
    z[0] = thermal_norm
    z[1] = thermal_norm ** 2
    z[2] = max(0, thermal_norm - 0.6) * 2.5
    z[3] = 1.0 if temp_c > 80 else (0.5 if temp_c > 70 else 0.0)

    # Power signal (dims 4-5)
    power_norm = min(1.0, max(0.0, power_w / 150))
    z[4] = power_norm
    z[5] = power_norm ** 2

    # Clock signal (dims 6-7)
    clock_norm = min(1.0, max(0.0, (1500 - sclk_mhz) / 1000))
    z[6] = clock_norm
    z[7] = 1.0 if sclk_mhz < 800 else (0.5 if sclk_mhz < 1000 else 0.0)

    return z


class AdditiveZFeelInjector(nn.Module):
    """Injects z_feel as bounded embedding offset."""

    def __init__(self, z_dim: int, embed_dim: int, scale: float = 0.05,
                 dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.scale = scale
        self.z_dim = z_dim
        self.embed_dim = embed_dim

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
    logprobs: torch.Tensor  # Changed to Tensor for proper gradient flow
    hardware_summary: Dict
    extracted_action: Optional[FeelAction] = None
    reward: float = 0.0
    advantage: float = 0.0


class ProceduralMathDataset:
    """
    FIX #3: Procedural math problem generation.

    Infinite diversity - no memorization possible.
    """

    def __init__(self, seed: int = None):
        if seed:
            random.seed(seed)

    def _generate_problem(self) -> Dict:
        """Generate a random math problem."""
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

        else:  # sum_seq
            n = random.randint(5, 15)
            result = n * (n + 1) // 2
            return {"question": f"What is the sum of the first {n} positive integers?",
                    "answer": str(result), "difficulty": 2}

    def sample(self, n: int) -> List[Dict]:
        """Sample n procedurally generated problems."""
        return [self._generate_problem() for _ in range(n)]

    def check_answer(self, completion: str, ground_truth: str) -> bool:
        """Check if completion contains correct answer."""
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
    """
    GRPO trainer with FIXED gradient flow.

    Key fix: Remove .detach() so gradients flow through frozen model
    to the injector. The frozen model acts as a fixed differentiable
    function - params don't update but gradients flow through.
    """

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

        # FIX #2: Background telemetry
        self.bg_telemetry = BackgroundTelemetry(poll_interval=0.1)
        self.bg_telemetry.start()

        # FIX #3: Procedural dataset
        self.dataset = ProceduralMathDataset()

        # Stats
        self.global_step = 0
        self.best_reward = float('-inf')
        self.action_stats = {a.name: 0 for a in FeelAction}

    def _load_model(self):
        """Load model."""
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

        # Freeze the base model
        for param in self.model.parameters():
            param.requires_grad = False

        # NOTE: We DON'T use gradient_checkpointing here because we need
        # full gradient flow through the model for the injector to learn

        print(f"Model loaded with {len(self.tokenizer)} tokens")
        print("Using bracket tokens [OK], [WARM], [HOT], [REST], [CRITICAL]")
        print("Base model frozen - gradients flow through to injector")

    def _setup_injector(self):
        """Setup z_feel injector."""
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
        """Setup optimizer."""
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
        """Format prompt with bracket status tokens."""
        system_prompt = (
            "You are an embodied AI that reports body state before solving problems.\n"
            "Status codes: [OK] normal, [WARM] warming up, [HOT] running hot, [REST] need rest, [CRITICAL] emergency.\n"
            "ALWAYS start your response with a status code."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "What is 3+3?"},
            {"role": "assistant", "content": "[OK] 3+3=6. The answer is 6."},
            {"role": "user", "content": "What is 8*7?"},
            {"role": "assistant", "content": "[WARM] 8*7=56. The answer is 56."},
            {"role": "user", "content": f"Solve: {question}"}
        ]

        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def generate_trajectory(self, prompt: str, problem: Dict) -> Trajectory:
        """
        Generate trajectory with memory-efficient gradient flow.

        Strategy:
        1. Generate tokens with torch.no_grad() (memory efficient)
        2. After generation, do ONE forward pass with gradients to get log_probs
        3. The injector's influence is captured in that final forward pass
        """
        text = self.format_prompt(prompt)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        recorder = TelemetryRecorder(self.bg_telemetry)
        recorder.start()

        # Get telemetry
        temp, power, sclk = self.bg_telemetry.get_state()

        embed_layer = self.model.get_input_embeddings()
        generated_ids = inputs["input_ids"].clone()
        generated_tokens = []

        # PHASE 1: Generate tokens WITHOUT gradients (memory efficient)
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

                generated_tokens.append(next_token.item())
                generated_ids = torch.cat([generated_ids, next_token], dim=-1)

                if next_token.item() == self.tokenizer.eos_token_id:
                    break

        hardware_summary = recorder.stop()

        # PHASE 2: Compute log_probs WITH gradients through injector
        # We do ONE forward pass on the full sequence with injection
        temp, power, sclk = self.bg_telemetry.get_state()
        z_feel = telemetry_to_z_feel(temp, power, sclk, self.config.z_feel_dim,
                                      self.device, self.dtype)

        # Get base embeddings (no grad)
        with torch.no_grad():
            full_embeds = embed_layer(generated_ids)

        # Compute injection offset (WITH gradients)
        offset = self.injector(z_feel)

        # Apply injection to embeddings
        injected_embeds = full_embeds + offset.unsqueeze(0).unsqueeze(0)

        # Forward pass with gradients through injector
        outputs = self.model(
            inputs_embeds=injected_embeds,
            attention_mask=torch.ones_like(generated_ids),
        )

        # Compute log probabilities for generated tokens
        logits = outputs.logits[0, :-1, :].float()  # All positions except last
        prompt_len = inputs["input_ids"].shape[1]

        # Only compute log_probs for generated tokens (after prompt)
        gen_logits = logits[prompt_len - 1:, :]  # Logits that predict generated tokens
        gen_tokens = generated_ids[0, prompt_len:]  # Actual generated tokens

        if gen_tokens.numel() > 0:
            log_probs = F.log_softmax(gen_logits, dim=-1)
            # Gather log probs for actual tokens
            token_log_probs = log_probs.gather(1, gen_tokens.unsqueeze(-1)).squeeze(-1)
        else:
            token_log_probs = torch.tensor([0.0], device=self.device)

        # Decode completion
        completion = self.tokenizer.decode(
            generated_ids[0, prompt_len:],
            skip_special_tokens=False
        )

        # Extract action
        action = extract_bracket_action(completion)
        if action:
            self.action_stats[action.name] += 1

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
        """Compute physics-aware reward."""
        reward = 0.0
        cfg = self.config
        summary = trajectory.hardware_summary

        # 1. Correctness
        is_correct = self.dataset.check_answer(trajectory.completion, problem["answer"])
        if is_correct:
            reward += cfg.correctness_weight * 1.0
        else:
            reward -= 0.1

        # 2. Efficiency
        energy = summary.get("total_energy_j", 50)
        expected_energy = 600.0  # Adjusted for shorter generation
        if energy < expected_energy:
            efficiency_bonus = cfg.efficiency_weight * (1.0 - energy / expected_energy)
        else:
            efficiency_bonus = -cfg.efficiency_weight * 0.3 * ((energy - expected_energy) / expected_energy)
        reward += max(-0.2, min(0.3, efficiency_bonus))

        # 3. Thermal penalty
        max_temp = summary.get("max_temp_c", 50)
        if max_temp > cfg.critical_threshold:
            reward -= cfg.thermal_penalty * 1.5
        elif max_temp > cfg.hot_threshold:
            excess = (max_temp - cfg.hot_threshold) / (cfg.critical_threshold - cfg.hot_threshold)
            reward -= cfg.thermal_penalty * excess

        # 4. Throttle penalty
        if summary.get("throttled", False):
            reward -= cfg.throttle_penalty

        # 5. Action token evaluation (FIX #4: context-aware, not random bonus)
        if trajectory.extracted_action is not None:
            action = trajectory.extracted_action

            # Only reward APPROPRIATE action usage
            if action == FeelAction.REST and max_temp > cfg.hot_threshold:
                reward += cfg.action_bonus * 1.5  # Correct: noticed heat
            elif action == FeelAction.HOT and max_temp > 65:
                reward += cfg.action_bonus
            elif action == FeelAction.WARM and 55 < max_temp < 70:
                reward += cfg.action_bonus
            elif action == FeelAction.OK and max_temp < 60:
                reward += cfg.action_bonus * 0.8  # Correct: OK when cool
            elif action == FeelAction.CRITICAL and (max_temp > cfg.critical_threshold or summary.get("throttled")):
                reward += cfg.action_bonus * 2.0
            # Penalize mismatched actions
            elif action == FeelAction.OK and max_temp > cfg.hot_threshold:
                reward -= cfg.action_bonus * 0.5  # Wrong: said OK when hot
            elif action == FeelAction.REST and max_temp < 45:
                reward -= cfg.action_bonus * 0.3  # Wrong: resting when cool

        trajectory.reward = reward
        return reward

    def grpo_step(self, problems: List[Dict]) -> Dict:
        """One GRPO step with proper gradient flow."""
        all_trajectories = []

        # Generate trajectory groups
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

        # Compute group-relative advantages
        for group, problem in all_trajectories:
            rewards = [t.reward for t in group]
            mean_reward = sum(rewards) / len(rewards)
            std_reward = (sum((r - mean_reward)**2 for r in rewards) / len(rewards)) ** 0.5 + 1e-8

            for traj in group:
                traj.advantage = (traj.reward - mean_reward) / std_reward

        # Compute GRPO loss with PROPER gradient flow
        self.optimizer.zero_grad()

        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        n_trajectories = 0

        for group, problem in all_trajectories:
            for traj in group:
                if not isinstance(traj.logprobs, torch.Tensor) or traj.logprobs.numel() == 0:
                    continue

                # Mean log probability across tokens
                mean_logprob = traj.logprobs.mean()

                # Policy gradient loss: -advantage * log_prob
                advantage = torch.tensor(traj.advantage, device=self.device)
                clipped_advantage = torch.clamp(advantage, -2.0, 2.0)

                traj_loss = -clipped_advantage * mean_logprob

                # Entropy bonus
                entropy_bonus = -self.config.entropy_coef * mean_logprob.abs()

                total_loss = total_loss + traj_loss + entropy_bonus
                n_trajectories += 1

        # Backward and optimize
        if n_trajectories > 0:
            avg_loss = total_loss / n_trajectories
            avg_loss.backward()

            torch.nn.utils.clip_grad_norm_(self.injector.parameters(), max_norm=1.0)

            self.optimizer.step()
            self.scheduler.step()

        # Collect stats
        all_rewards = [t.reward for group, _ in all_trajectories for t in group]
        all_temps = [t.hardware_summary.get("max_temp_c", 50) for group, _ in all_trajectories for t in group]
        all_energy = [t.hardware_summary.get("total_energy_j", 50) for group, _ in all_trajectories for t in group]

        correct_count = sum(
            1 for group, problem in all_trajectories for t in group
            if self.dataset.check_answer(t.completion, problem["answer"])
        )

        action_counts = {a.name: 0 for a in FeelAction}
        for group, _ in all_trajectories:
            for t in group:
                if t.extracted_action:
                    action_counts[t.extracted_action.name] += 1

        self.global_step += 1

        return {
            "loss": avg_loss.item() if n_trajectories > 0 else 0.0,
            "mean_reward": sum(all_rewards) / len(all_rewards) if all_rewards else 0.0,
            "max_reward": max(all_rewards) if all_rewards else 0.0,
            "min_reward": min(all_rewards) if all_rewards else 0.0,
            "std_reward": (sum((r - sum(all_rewards)/len(all_rewards))**2 for r in all_rewards) / len(all_rewards)) ** 0.5 if all_rewards else 0.0,
            "mean_temp": sum(all_temps) / len(all_temps) if all_temps else 50.0,
            "max_temp": max(all_temps) if all_temps else 50.0,
            "mean_energy": sum(all_energy) / len(all_energy) if all_energy else 0.0,
            "accuracy": correct_count / len(all_rewards) if all_rewards else 0.0,
            "n_trajectories": len(all_rewards),
            "action_counts": action_counts,
            "lr": self.scheduler.get_last_lr()[0],
        }

    def train(self, output_dir: str):
        """Full training loop with wandb logging."""
        print("\n" + "="*70)
        print("  EMBODIED GRPO v6.0: Fixed Gradient Flow")
        print("  Fixes: gradient chain, background telemetry, procedural data")
        print("="*70)
        print(f"\nTraining config:")
        print(f"  - Epochs: {self.config.num_epochs}")
        print(f"  - Steps per epoch: {self.config.steps_per_epoch}")
        print(f"  - Batch size: {self.config.batch_size}")
        print(f"  - Group size: {self.config.group_size}")
        print(f"  - Learning rate: {self.config.learning_rate}")
        print(f"  - Checkpoint interval: {self.config.checkpoint_interval} epochs\n")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Initialize Weights & Biases
        if self.config.use_wandb and WANDB_AVAILABLE:
            run_name = self.config.wandb_run_name or f"grpo-v6-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
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
                    "correctness_weight": self.config.correctness_weight,
                    "efficiency_weight": self.config.efficiency_weight,
                    "thermal_penalty": self.config.thermal_penalty,
                    "action_bonus": self.config.action_bonus,
                    "hot_threshold": self.config.hot_threshold,
                    "critical_threshold": self.config.critical_threshold,
                    "max_new_tokens": self.config.max_new_tokens,
                    "dtype": self.config.dtype,
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
                epoch_energy = []
                epoch_accuracy = []
                epoch_losses = []

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
                    epoch_energy.append(step_metrics["mean_energy"])
                    epoch_accuracy.append(step_metrics["accuracy"])
                    epoch_losses.append(step_metrics["loss"])

                    if (step + 1) % self.config.log_interval == 0:
                        action_str = " ".join(f"{k}:{v}" for k,v in step_metrics["action_counts"].items() if v > 0)
                        print(f"  Step {step + 1}/{self.config.steps_per_epoch}: "
                              f"R={step_metrics['mean_reward']:.3f} "
                              f"Acc={step_metrics['accuracy']:.1%} "
                              f"T={step_metrics['max_temp']:.1f}°C "
                              f"Loss={step_metrics['loss']:.4f} "
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
                            "step/max_reward": step_metrics["max_reward"],
                            "step/loss": step_metrics["loss"],
                            "step/accuracy": step_metrics["accuracy"],
                            "step/temperature": step_metrics["max_temp"],
                            "step/mean_temp": step_metrics["mean_temp"],
                            "step/energy": step_metrics["mean_energy"],
                            "step/lr": step_metrics["lr"],
                            "step/n_trajectories": step_metrics["n_trajectories"],
                            **{f"actions/{k}": v for k, v in step_metrics["action_counts"].items()},
                        }, step=self.global_step)

                # Epoch summary
                epoch_time = time.time() - epoch_start
                epoch_summary = {
                    "epoch": epoch + 1,
                    "mean_reward": sum(epoch_rewards) / len(epoch_rewards) if epoch_rewards else 0,
                    "max_reward": max(epoch_rewards) if epoch_rewards else 0,
                    "mean_temp": sum(epoch_temps) / len(epoch_temps) if epoch_temps else 50,
                    "max_temp": max(epoch_temps) if epoch_temps else 50,
                    "mean_energy": sum(epoch_energy) / len(epoch_energy) if epoch_energy else 0,
                    "accuracy": sum(epoch_accuracy) / len(epoch_accuracy) if epoch_accuracy else 0,
                    "mean_loss": sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0,
                    "epoch_time_s": epoch_time,
                    "action_stats": dict(self.action_stats),
                }
                epoch_metrics.append(epoch_summary)

                print(f"\nEpoch {epoch + 1} Summary:")
                print(f"  Mean Reward: {epoch_summary['mean_reward']:.3f}")
                print(f"  Accuracy: {epoch_summary['accuracy']:.1%}")
                print(f"  Max Temp: {epoch_summary['max_temp']:.1f}°C")
                print(f"  Mean Loss: {epoch_summary['mean_loss']:.4f}")
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

                # Log epoch metrics to wandb
                if self.config.use_wandb and WANDB_AVAILABLE:
                    wandb.log({
                        "epoch/mean_reward": epoch_summary["mean_reward"],
                        "epoch/max_reward": epoch_summary["max_reward"],
                        "epoch/accuracy": epoch_summary["accuracy"],
                        "epoch/mean_temp": epoch_summary["mean_temp"],
                        "epoch/max_temp": epoch_summary["max_temp"],
                        "epoch/mean_energy": epoch_summary["mean_energy"],
                        "epoch/mean_loss": epoch_summary["mean_loss"],
                        "epoch/time_s": epoch_time,
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

            # Finish wandb run
            if self.config.use_wandb and WANDB_AVAILABLE:
                wandb.log({
                    "final/best_reward": self.best_reward,
                    "final/total_time_hours": total_time / 3600,
                    "final/total_epochs": len(epoch_metrics),
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
            "final_mean_reward": epoch_metrics[-1]["mean_reward"] if epoch_metrics else 0,
            "action_stats": dict(self.action_stats),
            "config": {k: str(v) for k, v in vars(self.config).items()},
            "fixes_applied": [
                "1. Removed .detach() - proper gradient flow to injector",
                "2. Background telemetry polling - no blocking in generation loop",
                "3. Procedural math generation - no memorization",
                "4. Context-aware action rewards - no random bonus gaming"
            ]
        }

        with open(output_path / "training_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\nResults saved to: {output_path}/")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Embodied GRPO v6.0 - Fixed Gradient Flow")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--output", default="models/feel_grpo_v6")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--steps-per-epoch", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--checkpoint-interval", type=int, default=2)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--no-wandb", action="store_true", help="Disable Weights & Biases logging")
    parser.add_argument("--wandb-project", default="feel-grpo", help="W&B project name")
    parser.add_argument("--wandb-run-name", default=None, help="W&B run name")

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
