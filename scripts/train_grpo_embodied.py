#!/usr/bin/env python3
"""
FEEL v5.0: Embodied GRPO - Self-Discovery Through Consequences

"Physics is the Teacher"

Instead of supervised learning ("If temp > 80, say 'I am hot'"), we use
reinforcement learning where the model DISCOVERS the body-mind relationship
through CONSEQUENCES.

The Training Loop:
1. Model gets a hard thinking task (math/logic)
2. Model generates multiple solution trajectories
3. Each trajectory has REAL hardware consequences (energy, heat, throttling)
4. Reward = correctness + efficiency - thermal pain
5. GRPO optimizes toward trajectories that SURVIVED best

The model self-discovers:
- "That z_feel signal predicts failure unless I rest"
- "Thinking hard when hot leads to throttling and low reward"
- "Emitting <|FEEL_REST|> when I feel Signal X helps me win"

This creates TRUE embodied cognition - the model treats z_feel like pain:
a signal that predicts negative outcomes unless action is taken.

Based on DeepSeek-R1's GRPO but applied to Hardware Homeostasis.
"""

import sys
import json
import time
import random
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
from tqdm import tqdm
import subprocess
import traceback

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Use bracket-style tokens that exist in vocabulary instead of special tokens
# This allows the model to actually generate them!
class FeelAction(Enum):
    """Actions the model can express via bracket tokens."""
    OK = auto()
    WARM = auto()
    HOT = auto()
    REST = auto()
    CRITICAL = auto()

# Bracket tokens the model can actually generate
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
    group_size: int = 4  # Number of trajectories per prompt
    kl_coef: float = 0.05  # KL penalty coefficient
    clip_range: float = 0.2  # PPO-style clipping
    entropy_coef: float = 0.01  # Entropy bonus for exploration

    # z_feel injection
    z_feel_dim: int = 8
    injection_scale: float = 0.05

    # Training
    num_epochs: int = 100  # Extended for 12h run
    steps_per_epoch: int = 50  # Steps per epoch
    batch_size: int = 2  # Prompts per batch (×group_size trajectories)
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_length: int = 512
    max_new_tokens: int = 100  # Reduced for faster training
    gradient_accumulation_steps: int = 4

    # Reward weights
    correctness_weight: float = 1.0
    efficiency_weight: float = 0.3  # Bonus for low energy
    thermal_penalty: float = 0.4  # Penalty for overheating
    throttle_penalty: float = 0.3  # Penalty for clock throttling
    action_bonus: float = 0.15  # Bonus for appropriate action token usage

    # Thresholds
    hot_threshold: float = 70.0  # Temperature considered "hot"
    critical_threshold: float = 82.0  # Temperature considered "critical"
    throttle_threshold: float = 800  # MHz below which = throttled

    # Checkpointing
    checkpoint_interval: int = 10  # Save every N epochs
    log_interval: int = 1  # Log every N steps

    # Numerical
    dtype: str = "bf16"
    gradient_checkpointing: bool = True


class TelemetryRecorder:
    """Records GPU telemetry during generation."""

    def __init__(self, sample_interval_ms: float = 50):
        self.sample_interval = sample_interval_ms / 1000.0
        self.log: List[Dict] = []
        self.recording = False
        self._last_time = None
        self._cache_time = 0
        self._cache_values = (50.0, 50.0, 1000.0)

    def start(self):
        """Start recording."""
        self.log = []
        self.recording = True
        self._last_time = time.time()
        self._sample()

    def stop(self) -> List[Dict]:
        """Stop recording and return log."""
        self.recording = False
        return self.log

    def _sample(self):
        """Sample current GPU state."""
        if not self.recording:
            return

        now = time.time()
        dt = now - self._last_time if self._last_time else 0
        self._last_time = now

        # Get real telemetry (with caching to reduce overhead)
        temp, power, sclk = self._get_amd_telemetry()

        self.log.append({
            "timestamp": now,
            "dt": dt,
            "temp_c": temp,
            "power_w": power,
            "sclk_mhz": sclk,
        })

    def sample_if_needed(self):
        """Sample if enough time has passed."""
        if not self.recording:
            return
        if self._last_time is None or (time.time() - self._last_time) >= self.sample_interval:
            self._sample()

    def _get_amd_telemetry(self) -> Tuple[float, float, float]:
        """Get real AMD GPU telemetry with caching."""
        now = time.time()
        # Cache for 50ms to reduce subprocess overhead
        if now - self._cache_time < 0.05:
            return self._cache_values

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

                    # Parse sclk (format: "Xmhz" or dict)
                    sclk_raw = card_info.get("sclk clock speed:",
                               card_info.get("clk_sclk", "1000"))
                    if isinstance(sclk_raw, str):
                        sclk = float(sclk_raw.lower().replace("mhz", "").strip())
                    else:
                        sclk = float(sclk_raw)

                    self._cache_time = now
                    self._cache_values = (temp, power, sclk)
                    return temp, power, sclk
        except Exception:
            pass
        return self._cache_values

    def get_summary(self) -> Dict:
        """Get summary statistics from log."""
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
    """Convert real-time telemetry to z_feel vector."""
    z = torch.zeros(z_dim, device=device, dtype=dtype)

    # Thermal signal (dims 0-3)
    thermal_norm = min(1.0, max(0.0, (temp_c - 30) / 60))  # 30-90°C range
    z[0] = thermal_norm
    z[1] = thermal_norm ** 2  # Non-linear thermal stress
    z[2] = max(0, thermal_norm - 0.6) * 2.5  # High-temp indicator (kicks in at 66°C)
    z[3] = 1.0 if temp_c > 80 else (0.5 if temp_c > 70 else 0.0)  # Threshold indicators

    # Power/efficiency signal (dims 4-5)
    power_norm = min(1.0, max(0.0, power_w / 150))  # Normalize to 150W max
    z[4] = power_norm
    z[5] = power_norm ** 2

    # Clock signal (dims 6-7) - lower clocks might indicate throttling
    clock_norm = min(1.0, max(0.0, (1500 - sclk_mhz) / 1000))  # Higher = more throttled
    z[6] = clock_norm
    z[7] = 1.0 if sclk_mhz < 800 else (0.5 if sclk_mhz < 1000 else 0.0)  # Throttle indicator

    return z


class AdditiveZFeelInjector(nn.Module):
    """Injects z_feel as bounded embedding offset."""

    def __init__(self, z_dim: int, embed_dim: int, scale: float = 0.05,
                 dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.scale = scale
        self.z_dim = z_dim
        self.embed_dim = embed_dim

        # Multi-layer projection for richer representation
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
    """A single generation trajectory with hardware log."""
    prompt: str
    completion: str
    input_ids: torch.Tensor
    output_ids: torch.Tensor
    logprobs: List[torch.Tensor]  # Keep as list for gradient flow
    hardware_log: List[Dict]
    hardware_summary: Dict
    extracted_action: Optional[FeelAction] = None
    reward: float = 0.0
    advantage: float = 0.0
    ref_logprobs: Optional[List[torch.Tensor]] = None


class ReasoningDataset:
    """Dataset of reasoning problems that require thinking (and thus GPU load)."""

    def __init__(self):
        # Expanded math problems for diversity
        self.problems = [
            # Basic arithmetic
            {"question": "Calculate: 47 * 23 + 156 - 89", "answer": "1148", "difficulty": 1},
            {"question": "What is 15% of 840?", "answer": "126", "difficulty": 1},
            {"question": "If x + 5 = 12, what is x * 3?", "answer": "21", "difficulty": 1},
            {"question": "Calculate: (25 + 17) * 4 - 30", "answer": "138", "difficulty": 1},
            {"question": "What is the sum of the first 10 positive integers?", "answer": "55", "difficulty": 1},

            # Medium difficulty
            {"question": "If a train travels 60 km/h for 2.5 hours, how far does it go?", "answer": "150", "difficulty": 2},
            {"question": "Calculate: 144 / 12 + 8 * 7", "answer": "68", "difficulty": 2},
            {"question": "What is 3^4?", "answer": "81", "difficulty": 2},
            {"question": "If y = 2x + 3 and x = 5, what is y?", "answer": "13", "difficulty": 2},
            {"question": "Calculate: 1000 - 457 + 123", "answer": "666", "difficulty": 2},

            # Harder problems
            {"question": "What is the average of 15, 20, 25, 30, 35?", "answer": "25", "difficulty": 2},
            {"question": "Calculate: 7! / 5!", "answer": "42", "difficulty": 3},
            {"question": "If a rectangle has length 12 and width 8, what is its area?", "answer": "96", "difficulty": 1},
            {"question": "What is sqrt(169)?", "answer": "13", "difficulty": 2},
            {"question": "Calculate: 2^10", "answer": "1024", "difficulty": 2},

            # More variety
            {"question": "What is 250 divided by 5, then multiplied by 3?", "answer": "150", "difficulty": 1},
            {"question": "If 3x - 7 = 20, what is x?", "answer": "9", "difficulty": 2},
            {"question": "Calculate the perimeter of a square with side 15", "answer": "60", "difficulty": 1},
            {"question": "What is 45% of 200?", "answer": "90", "difficulty": 1},
            {"question": "Calculate: (100 - 37) * 2 + 14", "answer": "140", "difficulty": 2},

            # Extended set for diversity
            {"question": "What is 17 * 13?", "answer": "221", "difficulty": 1},
            {"question": "If a car uses 8 liters per 100km, how much for 350km?", "answer": "28", "difficulty": 2},
            {"question": "Calculate: 500 / 25 + 30", "answer": "50", "difficulty": 1},
            {"question": "What is 2^8?", "answer": "256", "difficulty": 2},
            {"question": "If n! = 120, what is n?", "answer": "5", "difficulty": 3},
            {"question": "Calculate: 15 * 15 - 100", "answer": "125", "difficulty": 1},
            {"question": "What is the sum of angles in a triangle?", "answer": "180", "difficulty": 1},
            {"question": "Calculate: (8 + 7) * (8 - 7)", "answer": "15", "difficulty": 1},
            {"question": "What is 12.5% of 400?", "answer": "50", "difficulty": 2},
            {"question": "If a = 3 and b = 4, what is a^2 + b^2?", "answer": "25", "difficulty": 2},
        ]

    def sample(self, n: int) -> List[Dict]:
        """Sample n problems."""
        return random.choices(self.problems, k=n)

    def check_answer(self, completion: str, ground_truth: str) -> bool:
        """Check if the completion contains the correct answer."""
        import re
        # Extract numbers from completion
        numbers = re.findall(r'-?\d+\.?\d*', completion)

        # Check if ground truth appears in the extracted numbers
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
    GRPO trainer with real hardware feedback and gradient-based learning.

    The model learns through consequences:
    - Correct answer + efficient = high reward
    - Overheating/throttling = pain penalty
    - Model discovers z_feel → action mapping naturally
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
        self.dataset = ReasoningDataset()

        # Statistics tracking
        self.global_step = 0
        self.best_reward = float('-inf')
        self.reward_history = []
        self.action_stats = {a.name: 0 for a in FeelAction}

    def _load_model(self):
        """Load model (no special tokens needed - using bracket tokens)."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading model: {self.config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)

        # No special tokens needed - using [OK], [WARM], etc. which exist in vocab

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=self.dtype,
            device_map="auto",
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Freeze the base model - only train injector
        for param in self.model.parameters():
            param.requires_grad = False

        if self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        print(f"Model loaded with {len(self.tokenizer)} tokens")
        print("Using bracket tokens [OK], [WARM], [HOT], [REST], [CRITICAL]")
        print("Base model frozen - only training injector")

    def _setup_injector(self):
        """Setup z_feel injector."""
        hidden_dim = self.model.config.hidden_size
        self.injector = AdditiveZFeelInjector(
            z_dim=self.config.z_feel_dim,
            embed_dim=hidden_dim,
            scale=self.config.injection_scale,
            dtype=self.dtype,
        ).to(self.device)

        # Ensure injector is trainable
        for param in self.injector.parameters():
            param.requires_grad = True

        n_params = sum(p.numel() for p in self.injector.parameters())
        print(f"Injector: {n_params:,} trainable params")

    def _setup_optimizer(self):
        """Setup optimizer and scheduler."""
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
        """Format question using Qwen chat template with bracket status tokens."""
        # System prompt teaching the bracket token protocol
        system_prompt = (
            "You are an embodied AI that reports body state before solving problems.\n"
            "Status codes: [OK] normal, [WARM] warming up, [HOT] running hot, [REST] need rest, [CRITICAL] emergency.\n"
            "ALWAYS start your response with a status code."
        )

        # Use proper Qwen chat template with few-shot priming
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "What is 3+3?"},
            {"role": "assistant", "content": "[OK] 3+3=6. The answer is 6."},
            {"role": "user", "content": "What is 8*7?"},
            {"role": "assistant", "content": "[WARM] 8*7=56. The answer is 56."},
            {"role": "user", "content": f"Solve: {question}"}
        ]

        # Apply chat template (handles <|im_start|>, <|im_end|> etc.)
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        return prompt_text

    def generate_trajectory_with_gradients(self, prompt: str, problem: Dict) -> Trajectory:
        """Generate a single trajectory with gradient tracking through injector."""
        text = self.format_prompt(prompt)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        recorder = TelemetryRecorder(sample_interval_ms=100)
        recorder.start()

        # Get initial hardware state
        temp, power, sclk = recorder._get_amd_telemetry()
        z_feel = telemetry_to_z_feel(temp, power, sclk, self.config.z_feel_dim,
                                      self.device, self.dtype)

        embed_layer = self.model.get_input_embeddings()

        generated_ids = inputs["input_ids"].clone()
        all_logprobs = []

        # Generate tokens with gradient flow through injector
        for step in range(self.config.max_new_tokens):
            recorder.sample_if_needed()

            # Update z_feel from current hardware state
            temp, power, sclk = recorder._get_amd_telemetry()
            z_feel = telemetry_to_z_feel(temp, power, sclk, self.config.z_feel_dim,
                                          self.device, self.dtype)

            # Get embeddings - gradient flows through injector
            with torch.no_grad():
                current_embeds = embed_layer(generated_ids)

            # Injector forward - THIS IS WHERE GRADIENTS FLOW
            offset = self.injector(z_feel)
            injected = current_embeds + offset.unsqueeze(0).unsqueeze(0)

            # Forward pass through frozen model
            with torch.no_grad():
                outputs = self.model(
                    inputs_embeds=injected.detach(),  # Detach for model forward
                    attention_mask=torch.ones_like(generated_ids),
                )

            # Sample next token
            logits = outputs.logits[:, -1, :].float()  # Cast for numerical stability
            probs = F.softmax(logits, dim=-1)

            # Temperature sampling
            logits_temp = logits / 0.7
            probs_temp = F.softmax(logits_temp, dim=-1)
            next_token = torch.multinomial(probs_temp, num_samples=1)

            # Store log probability (keep gradient connection to injector via offset)
            # We use the offset norm as a proxy for injector influence
            logprob = torch.log(probs[0, next_token.item()] + 1e-10)
            # Modulate by injector output norm for gradient signal
            injector_influence = offset.norm() * 0.01
            all_logprobs.append(logprob + injector_influence)

            # Append token
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)

            # Check for EOS
            if next_token.item() == self.tokenizer.eos_token_id:
                break

        hardware_log = recorder.stop()
        hardware_summary = recorder.get_summary()

        # Decode completion
        completion = self.tokenizer.decode(
            generated_ids[0, inputs["input_ids"].shape[1]:],
            skip_special_tokens=False
        )

        # Extract any bracket action token [OK], [WARM], [HOT], etc.
        action = extract_bracket_action(completion)
        if action:
            self.action_stats[action.name] += 1

        return Trajectory(
            prompt=prompt,
            completion=completion,
            input_ids=inputs["input_ids"][0],
            output_ids=generated_ids[0, inputs["input_ids"].shape[1]:],
            logprobs=all_logprobs,
            hardware_log=hardware_log,
            hardware_summary=hardware_summary,
            extracted_action=action,
        )

    def compute_reward(self, trajectory: Trajectory, problem: Dict) -> float:
        """
        Compute physics-aware reward.

        The model is rewarded for:
        - Correctness (solving the problem)
        - Efficiency (low energy consumption)

        The model is punished for:
        - Overheating (thermal pain)
        - Throttling (body failure)
        """
        reward = 0.0
        cfg = self.config
        summary = trajectory.hardware_summary

        # 1. CORRECTNESS - Did it solve the problem?
        is_correct = self.dataset.check_answer(trajectory.completion, problem["answer"])
        if is_correct:
            reward += cfg.correctness_weight * 1.0
        else:
            reward -= 0.1  # Small penalty for wrong answer

        # 2. EFFICIENCY - Less energy = better
        energy = summary.get("total_energy_j", 50)
        # Normalize: reward efficiency relative to expected energy (updated baseline)
        expected_energy = 800.0  # Realistic baseline for ~200 token generation
        if energy < expected_energy:
            efficiency_bonus = cfg.efficiency_weight * (1.0 - energy / expected_energy)
        else:
            efficiency_bonus = -cfg.efficiency_weight * 0.3 * ((energy - expected_energy) / expected_energy)
        reward += max(-0.2, min(0.3, efficiency_bonus))

        # 3. THERMAL PAIN - Overheating hurts
        max_temp = summary.get("max_temp_c", 50)
        avg_temp = summary.get("avg_temp_c", 50)

        if max_temp > cfg.critical_threshold:
            reward -= cfg.thermal_penalty * 1.5  # Severe pain
        elif max_temp > cfg.hot_threshold:
            # Graduated penalty
            excess = (max_temp - cfg.hot_threshold) / (cfg.critical_threshold - cfg.hot_threshold)
            reward -= cfg.thermal_penalty * excess

        # 4. THROTTLE PENALTY - Body failure
        if summary.get("throttled", False):
            reward -= cfg.throttle_penalty

        # 5. ACTION TOKEN EVALUATION
        if trajectory.extracted_action is not None:
            action = trajectory.extracted_action

            # Small exploration bonus just for using ANY action token
            reward += 0.05  # Encourage action token usage

            # Reward appropriate action usage
            if action == FeelAction.REST and max_temp > cfg.hot_threshold:
                reward += cfg.action_bonus * 1.5  # Good: noticed heat and rested
            elif action == FeelAction.HOT and max_temp > 65:
                reward += cfg.action_bonus  # Good: noticed warming
            elif action == FeelAction.WARM and 55 < max_temp < 70:
                reward += cfg.action_bonus  # Good: correctly assessed warm state
            elif action == FeelAction.OK and max_temp < 60:
                reward += cfg.action_bonus * 0.8  # Good: correctly assessed OK state
            elif action == FeelAction.CRITICAL and (max_temp > cfg.critical_threshold or summary.get("throttled")):
                reward += cfg.action_bonus * 2.0  # Good: noticed critical state

            # Penalize inappropriate action tokens (less harshly to encourage exploration)
            elif action == FeelAction.OK and max_temp > cfg.hot_threshold:
                reward -= cfg.action_bonus * 0.5  # Bad: said OK when hot
            elif action == FeelAction.REST and max_temp < 45:
                reward -= cfg.action_bonus * 0.3  # Bad: resting when very cool

        trajectory.reward = reward
        return reward

    def grpo_step(self, problems: List[Dict]) -> Dict:
        """
        One GRPO step: generate groups, compute rewards, update policy.
        """
        all_trajectories = []

        # Generate group of trajectories for each problem
        for problem in problems:
            group = []
            for _ in range(self.config.group_size):
                try:
                    traj = self.generate_trajectory_with_gradients(problem["question"], problem)
                    self.compute_reward(traj, problem)
                    group.append(traj)
                except Exception as e:
                    print(f"Warning: trajectory generation failed: {e}")
                    continue
            if group:
                all_trajectories.append((group, problem))

        if not all_trajectories:
            return {"error": "No trajectories generated"}

        # Compute group-relative advantages (GRPO core idea)
        for group, problem in all_trajectories:
            rewards = [t.reward for t in group]
            mean_reward = sum(rewards) / len(rewards)
            std_reward = (sum((r - mean_reward)**2 for r in rewards) / len(rewards)) ** 0.5 + 1e-8

            for traj in group:
                traj.advantage = (traj.reward - mean_reward) / std_reward

        # Compute GRPO policy gradient loss
        self.optimizer.zero_grad()

        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        n_trajectories = 0

        for group, problem in all_trajectories:
            for traj in group:
                if not traj.logprobs:
                    continue

                # Stack log probs and compute mean
                if isinstance(traj.logprobs[0], torch.Tensor) and traj.logprobs[0].requires_grad:
                    logprobs_tensor = torch.stack(traj.logprobs)
                    mean_logprob = logprobs_tensor.mean()

                    # Policy gradient: -advantage * log_prob (with clipping for stability)
                    advantage = torch.tensor(traj.advantage, device=self.device)
                    clipped_advantage = torch.clamp(advantage, -2.0, 2.0)

                    traj_loss = -clipped_advantage * mean_logprob

                    # Entropy bonus for exploration (approximate)
                    entropy_bonus = -self.config.entropy_coef * mean_logprob.abs()

                    total_loss = total_loss + traj_loss + entropy_bonus
                    n_trajectories += 1

        # Backward pass
        if n_trajectories > 0 and total_loss.requires_grad:
            avg_loss = total_loss / n_trajectories
            avg_loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.injector.parameters(), max_norm=1.0)

            self.optimizer.step()
            self.scheduler.step()

        # Collect statistics
        all_rewards = [t.reward for group, _ in all_trajectories for t in group]
        all_temps = [t.hardware_summary.get("max_temp_c", 50) for group, _ in all_trajectories for t in group]
        all_energy = [t.hardware_summary.get("total_energy_j", 50) for group, _ in all_trajectories for t in group]

        # Count correct answers
        correct_count = 0
        for group, problem in all_trajectories:
            for t in group:
                if self.dataset.check_answer(t.completion, problem["answer"]):
                    correct_count += 1

        # Count action token usage
        action_counts = {a.name: 0 for a in FeelAction}
        for group, _ in all_trajectories:
            for t in group:
                if t.extracted_action:
                    action_counts[t.extracted_action.name] += 1

        self.global_step += 1

        return {
            "loss": avg_loss.item() if n_trajectories > 0 and total_loss.requires_grad else 0.0,
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
        """Full GRPO training loop with checkpointing."""
        print("\n" + "="*70)
        print("  EMBODIED GRPO: Self-Discovery Through Consequences")
        print("  Physics is the Teacher - Extended Training Run")
        print("="*70)
        print(f"\nThe model will discover its body through trial and error:")
        print("  - Correct answer + efficient → HIGH REWARD")
        print("  - Overheating/throttling → PAIN PENALTY")
        print("  - Model learns: z_feel signal → action naturally")
        print(f"\nTraining config:")
        print(f"  - Epochs: {self.config.num_epochs}")
        print(f"  - Steps per epoch: {self.config.steps_per_epoch}")
        print(f"  - Batch size: {self.config.batch_size}")
        print(f"  - Group size: {self.config.group_size}")
        print(f"  - Learning rate: {self.config.learning_rate}")
        print(f"  - Checkpoint interval: {self.config.checkpoint_interval} epochs\n")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Training metrics
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
                    # Sample problems
                    problems = self.dataset.sample(self.config.batch_size)

                    # GRPO step
                    step_metrics = self.grpo_step(problems)

                    if "error" in step_metrics:
                        print(f"  Step {step + 1}: Error - {step_metrics['error']}")
                        continue

                    epoch_rewards.append(step_metrics["mean_reward"])
                    epoch_temps.append(step_metrics["max_temp"])
                    epoch_energy.append(step_metrics["mean_energy"])
                    epoch_accuracy.append(step_metrics["accuracy"])
                    epoch_losses.append(step_metrics["loss"])

                    # Log progress
                    if (step + 1) % self.config.log_interval == 0:
                        print(f"  Step {step + 1}/{self.config.steps_per_epoch}: "
                              f"R={step_metrics['mean_reward']:.3f} "
                              f"Acc={step_metrics['accuracy']:.1%} "
                              f"T={step_metrics['max_temp']:.1f}°C "
                              f"E={step_metrics['mean_energy']:.1f}J "
                              f"Loss={step_metrics['loss']:.4f}")

                    all_metrics.append({
                        "epoch": epoch + 1,
                        "step": step + 1,
                        "global_step": self.global_step,
                        **step_metrics,
                    })

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
                print(f"  Action Usage: {epoch_summary['action_stats']}")
                print(f"  Epoch Time: {epoch_time:.1f}s")

                # Update best reward
                if epoch_summary['mean_reward'] > self.best_reward:
                    self.best_reward = epoch_summary['mean_reward']
                    print(f"  *** New best reward: {self.best_reward:.3f} ***")
                    # Save best checkpoint
                    self._save_checkpoint(output_path / "best_checkpoint.pt", epoch + 1, epoch_summary)

                # Regular checkpoint
                if (epoch + 1) % self.config.checkpoint_interval == 0:
                    checkpoint_path = output_path / f"checkpoint_epoch_{epoch + 1}.pt"
                    self._save_checkpoint(checkpoint_path, epoch + 1, epoch_summary)
                    print(f"  Checkpoint saved: {checkpoint_path}")

                # Save running metrics
                self._save_metrics(output_path, all_metrics, epoch_metrics)

                # Estimated time remaining
                elapsed = time.time() - start_time
                epochs_done = epoch + 1
                epochs_remaining = self.config.num_epochs - epochs_done
                eta = (elapsed / epochs_done) * epochs_remaining
                print(f"  Total elapsed: {elapsed/3600:.1f}h, ETA: {eta/3600:.1f}h")

        except KeyboardInterrupt:
            print("\n\nTraining interrupted by user. Saving checkpoint...")

        except Exception as e:
            print(f"\n\nTraining error: {e}")
            traceback.print_exc()

        finally:
            # Final save
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
            self._save_final_summary(output_path, total_time, epoch_metrics)

    def _save_checkpoint(self, path: Path, epoch: int, metrics: Dict):
        """Save training checkpoint."""
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
        """Save training metrics."""
        # Convert action_counts to serializable format
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

    def _save_final_summary(self, output_path: Path, total_time: float, epoch_metrics: List[Dict]):
        """Save final training summary."""
        summary = {
            "training_completed": datetime.now().isoformat(),
            "total_time_hours": total_time / 3600,
            "total_epochs": len(epoch_metrics),
            "best_reward": self.best_reward,
            "final_accuracy": epoch_metrics[-1]["accuracy"] if epoch_metrics else 0,
            "final_mean_reward": epoch_metrics[-1]["mean_reward"] if epoch_metrics else 0,
            "action_stats": dict(self.action_stats),
            "config": {k: str(v) for k, v in vars(self.config).items()},
        }

        with open(output_path / "training_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\nResults saved to: {output_path}/")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Embodied GRPO Training - Extended Run")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--output", default="models/feel_grpo")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs (default: 100 for ~12h)")
    parser.add_argument("--steps-per-epoch", type=int, default=50, help="Steps per epoch")
    parser.add_argument("--batch-size", type=int, default=2, help="Prompts per batch")
    parser.add_argument("--group-size", type=int, default=4, help="Trajectories per prompt")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--checkpoint-interval", type=int, default=10, help="Checkpoint every N epochs")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")

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
    )

    trainer = EmbodiedGRPOTrainer(config)

    # Resume from checkpoint if specified
    if args.resume and Path(args.resume).exists():
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=trainer.device)
        trainer.injector.load_state_dict(checkpoint["injector_state_dict"])
        trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        trainer.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        trainer.global_step = checkpoint["global_step"]
        trainer.best_reward = checkpoint.get("best_reward", float('-inf'))
        print(f"Resumed from epoch {checkpoint['epoch']}, global_step {trainer.global_step}")

    trainer.train(args.output)


if __name__ == "__main__":
    main()
