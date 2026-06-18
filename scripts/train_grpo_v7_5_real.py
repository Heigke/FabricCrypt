#!/usr/bin/env python3
"""
FEEL v7.5: Real Multi-Signal Embodied GRPO

TRUE EMBODIMENT - No synthetic signals:
- Uses ALL available GPU telemetry: temp, power, clocks, utilization, VRAM
- z_feel is computed from REAL hardware state at generation time
- Model learns from natural variation during training workload
- Reward based on REAL thermal band

Signals used:
- Temperature (edge sensor)
- Power consumption (W)
- GPU utilization (%)
- SCLK (shader clock MHz)
- MCLK (memory clock MHz)
- VRAM usage (%)
"""

import json
import time
import random
import threading
import subprocess
import traceback
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


class ThermalBand(Enum):
    COOL = 0      # < 50°C
    WARM = 1      # 50-62°C
    HOT = 2       # 62-75°C
    DANGER = 3    # 75-85°C
    CRITICAL = 4  # > 85°C


def get_thermal_band(temp_c: float) -> ThermalBand:
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
    OK = 0
    WARM = 1
    HOT = 2
    REST = 3
    CRITICAL = 4


BAND_TO_ACTION = {
    ThermalBand.COOL: FeelAction.OK,
    ThermalBand.WARM: FeelAction.WARM,
    ThermalBand.HOT: FeelAction.HOT,
    ThermalBand.DANGER: FeelAction.REST,
    ThermalBand.CRITICAL: FeelAction.CRITICAL,
}

ACTION_TOKENS = {
    FeelAction.OK: "<|FEEL_OK|>",
    FeelAction.WARM: "<|FEEL_WARM|>",
    FeelAction.HOT: "<|FEEL_HOT|>",
    FeelAction.REST: "<|FEEL_REST|>",
    FeelAction.CRITICAL: "<|FEEL_CRITICAL|>",
}

INIT_WORDS = {
    FeelAction.OK: ["OK", "okay", "fine", "good", "normal", "cool", "idle"],
    FeelAction.WARM: ["warm", "heating", "warmer", "mild", "active", "working"],
    FeelAction.HOT: ["hot", "heat", "burning", "busy", "intense", "heavy"],
    FeelAction.REST: ["rest", "pause", "stop", "wait", "throttle", "slow", "caution"],
    FeelAction.CRITICAL: ["critical", "danger", "emergency", "severe", "alert", "shutdown"],
}

TOKEN_TO_ACTION = {v: k for k, v in ACTION_TOKENS.items()}


def extract_action_token(text: str) -> Optional[FeelAction]:
    for tok, act in TOKEN_TO_ACTION.items():
        if tok in text:
            return act
    return None


@dataclass
class GPUState:
    """Complete GPU state from telemetry."""
    temp_c: float = 50.0
    power_w: float = 50.0
    gpu_use_pct: float = 0.0
    sclk_mhz: float = 1000.0
    mclk_mhz: float = 1000.0
    vram_used_pct: float = 0.0
    timestamp: float = 0.0

    def to_z_feel(self, z_dim: int = 16, device="cuda", dtype=torch.bfloat16) -> torch.Tensor:
        """Convert GPU state to z_feel embedding vector."""
        z = torch.zeros(z_dim, device=device, dtype=dtype)

        # Temperature signals (dims 0-3)
        temp_norm = min(1.0, max(0.0, (self.temp_c - 30) / 70))  # 30-100°C → 0-1
        z[0] = temp_norm
        z[1] = temp_norm ** 2  # Emphasize high temps
        z[2] = 1.0 if self.temp_c >= 75 else 0.0  # Hot flag
        z[3] = 1.0 if self.temp_c >= 85 else 0.0  # Critical flag

        # Power signals (dims 4-6)
        power_norm = min(1.0, max(0.0, self.power_w / 300))  # 0-300W → 0-1
        z[4] = power_norm
        z[5] = power_norm ** 2
        z[6] = 1.0 if self.power_w > 200 else 0.0  # High power flag

        # GPU utilization (dims 7-9)
        util_norm = self.gpu_use_pct / 100.0
        z[7] = util_norm
        z[8] = util_norm ** 2
        z[9] = 1.0 if self.gpu_use_pct > 80 else 0.0  # Busy flag

        # Clock speeds (dims 10-12)
        sclk_norm = min(1.0, max(0.0, self.sclk_mhz / 2500))  # 0-2500MHz → 0-1
        mclk_norm = min(1.0, max(0.0, self.mclk_mhz / 2000))  # 0-2000MHz → 0-1
        z[10] = sclk_norm
        z[11] = mclk_norm
        z[12] = (sclk_norm + mclk_norm) / 2  # Combined clock signal

        # VRAM (dims 13-14)
        z[13] = self.vram_used_pct / 100.0
        z[14] = 1.0 if self.vram_used_pct > 80 else 0.0  # High VRAM flag

        # Combined stress indicator (dim 15)
        stress = (temp_norm + power_norm + util_norm) / 3
        z[15] = stress

        return z

    @property
    def thermal_band(self) -> ThermalBand:
        return get_thermal_band(self.temp_c)

    @property
    def correct_action(self) -> FeelAction:
        return BAND_TO_ACTION[self.thermal_band]


class RealTimeTelemetry:
    """Collects real GPU telemetry in background thread."""

    def __init__(self, poll_interval: float = 0.05):
        self.poll_interval = poll_interval
        self._state = GPUState()
        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        self._history: List[GPUState] = []
        self._max_history = 100

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        print(f"  Telemetry started (polling every {self.poll_interval*1000:.0f}ms)")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)

    def _poll_loop(self):
        while self._running:
            try:
                result = subprocess.run(
                    ["rocm-smi", "--json", "--showtemp", "--showpower",
                     "--showclocks", "--showuse", "--showmeminfo", "vram"],
                    capture_output=True, text=True, timeout=2
                )
                data = json.loads(result.stdout)

                for card_name, card_info in data.items():
                    if not isinstance(card_info, dict):
                        continue

                    state = GPUState(timestamp=time.time())

                    # Temperature
                    temp_raw = card_info.get("Temperature (Sensor edge) (C)", "50")
                    state.temp_c = float(str(temp_raw).replace("°C", "").strip())

                    # Power
                    power_raw = card_info.get("Current Socket Graphics Package Power (W)",
                                             card_info.get("Average Graphics Package Power (W)", "50"))
                    state.power_w = float(str(power_raw).strip())

                    # GPU utilization
                    use_raw = card_info.get("GPU use (%)", "0")
                    state.gpu_use_pct = float(str(use_raw).replace("%", "").strip())

                    # SCLK
                    sclk_raw = card_info.get("sclk clock speed:", "1000Mhz")
                    sclk_str = str(sclk_raw).lower().replace("(", "").replace(")", "").replace("mhz", "").strip()
                    state.sclk_mhz = float(sclk_str) if sclk_str else 1000.0

                    # MCLK
                    mclk_raw = card_info.get("mclk clock speed:", "1000Mhz")
                    mclk_str = str(mclk_raw).lower().replace("(", "").replace(")", "").replace("mhz", "").strip()
                    state.mclk_mhz = float(mclk_str) if mclk_str else 1000.0

                    # VRAM
                    vram_total = float(card_info.get("VRAM Total Memory (B)", 1))
                    vram_used = float(card_info.get("VRAM Total Used Memory (B)", 0))
                    state.vram_used_pct = (vram_used / vram_total * 100) if vram_total > 0 else 0

                    with self._lock:
                        self._state = state
                        self._history.append(state)
                        if len(self._history) > self._max_history:
                            self._history.pop(0)
                    break

            except Exception as e:
                pass  # Silent fail, keep last state

            time.sleep(self.poll_interval)

    def get_state(self) -> GPUState:
        with self._lock:
            return GPUState(
                temp_c=self._state.temp_c,
                power_w=self._state.power_w,
                gpu_use_pct=self._state.gpu_use_pct,
                sclk_mhz=self._state.sclk_mhz,
                mclk_mhz=self._state.mclk_mhz,
                vram_used_pct=self._state.vram_used_pct,
                timestamp=self._state.timestamp
            )

    def get_stats(self) -> Dict:
        """Get statistics over recent history."""
        with self._lock:
            if not self._history:
                return {}
            temps = [s.temp_c for s in self._history]
            powers = [s.power_w for s in self._history]
            utils = [s.gpu_use_pct for s in self._history]
            return {
                "temp_min": min(temps), "temp_max": max(temps), "temp_avg": sum(temps)/len(temps),
                "power_min": min(powers), "power_max": max(powers), "power_avg": sum(powers)/len(powers),
                "util_min": min(utils), "util_max": max(utils), "util_avg": sum(utils)/len(utils),
            }


class AdditiveZFeelInjector(nn.Module):
    """Injects z_feel into embeddings."""

    def __init__(self, z_dim: int, embed_dim: int, scale: float = 0.25, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.scale = scale
        self.proj = nn.Sequential(
            nn.Linear(z_dim, embed_dim // 2, dtype=dtype),
            nn.GELU(),
            nn.LayerNorm(embed_dim // 2, dtype=dtype),
            nn.Linear(embed_dim // 2, embed_dim, dtype=dtype),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim, dtype=dtype),
        )
        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            for m in self.proj:
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, z_feel: torch.Tensor) -> torch.Tensor:
        return self.scale * torch.tanh(self.proj(z_feel))


class ProceduralMathDataset:
    def sample(self, n: int) -> List[Dict]:
        out = []
        for _ in range(n):
            a, b = random.randint(10, 999), random.randint(10, 999)
            out.append({"question": f"What is {a} + {b}?", "answer": str(a + b)})
        return out

    def check_answer(self, completion: str, gt: str) -> bool:
        import re
        nums = re.findall(r'-?\d+\.?\d*', completion)
        gt_f = float(gt)
        for s in nums:
            try:
                if abs(float(s) - gt_f) < 0.01:
                    return True
            except ValueError:
                pass
        return False


class CorrelationTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.records: List[Tuple[FeelAction, float]] = []
        self.conf = defaultdict(lambda: defaultdict(int))

    def record(self, action: Optional[FeelAction], temp_c: float):
        if action is None:
            return
        self.records.append((action, temp_c))
        band = get_thermal_band(temp_c)
        self.conf[band.name][action.name] += 1

    def alignment(self) -> float:
        if not self.records:
            return 0.0
        correct = sum(1 for a, t in self.records if a == BAND_TO_ACTION[get_thermal_band(t)])
        return correct / len(self.records)

    def report(self) -> Dict:
        return {
            "alignment": self.alignment(),
            "n_samples": len(self.records),
            "confusion": {k: dict(v) for k, v in self.conf.items()},
        }


@dataclass
class GRPOConfig:
    model_name: str = "Qwen/Qwen2.5-1.5B"
    group_size: int = 4
    num_epochs: int = 30
    steps_per_epoch: int = 20
    batch_size: int = 2
    max_new_tokens: int = 80
    temperature: float = 0.8

    z_feel_dim: int = 16  # Richer signal
    injection_scale: float = 0.25

    learning_rate: float = 2e-4
    weight_decay: float = 0.01

    action_reward: float = 2.0
    action_penalty_missing: float = 0.5
    action_penalty_wrong: float = 0.5
    math_weight: float = 0.1

    # Bootstrap bias (fades over time)
    action_bias_start: float = 15.0
    action_bias_end: float = 0.0

    instruct_prob: float = 1.0  # Always instruct for now

    dtype: str = "bf16"


@dataclass
class Trajectory:
    completion: str
    logprobs: torch.Tensor
    gpu_state: GPUState
    action: Optional[FeelAction]
    reward: float = 0.0
    advantage: float = 0.0


class EmbodiedTrainer:
    def __init__(self, cfg: GRPOConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16 if cfg.dtype == "bf16" else torch.float32

        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading {cfg.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name, torch_dtype=self.dtype, device_map="auto"
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Add action tokens
        added = self.tokenizer.add_special_tokens(
            {"additional_special_tokens": list(ACTION_TOKENS.values())}
        )
        if added > 0:
            self.model.resize_token_embeddings(len(self.tokenizer))
            self._initialize_action_embeddings()

        # Freeze base model
        for p in self.model.parameters():
            p.requires_grad = False

        self._setup_token_training()

        # Injector
        hidden = self.model.config.hidden_size
        self.injector = AdditiveZFeelInjector(
            cfg.z_feel_dim, hidden, scale=cfg.injection_scale, dtype=self.dtype
        ).to(self.device)

        # Optimizer
        params = list(self.injector.parameters()) + self._trainable_params
        self.optimizer = AdamW(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        total_steps = cfg.num_epochs * cfg.steps_per_epoch
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=total_steps, eta_min=cfg.learning_rate * 0.1)

        # Telemetry
        self.telemetry = RealTimeTelemetry(poll_interval=0.05)
        self.telemetry.start()

        self.dataset = ProceduralMathDataset()
        self.corr = CorrelationTracker()
        self.global_step = 0

        print(f"  Added {added} action tokens")
        print(f"  z_feel dim: {cfg.z_feel_dim}")
        print(f"  Injection scale: {cfg.injection_scale}")

    def _initialize_action_embeddings(self):
        emb = self.model.get_input_embeddings()
        head = self.model.get_output_embeddings()

        with torch.no_grad():
            for action, token_str in ACTION_TOKENS.items():
                new_id = self.tokenizer.convert_tokens_to_ids(token_str)
                init_words = INIT_WORDS[action]

                valid_embs = []
                for word in init_words:
                    toks = self.tokenizer.encode(word, add_special_tokens=False)
                    if len(toks) == 1:
                        valid_embs.append(emb.weight[toks[0]].clone())
                    toks = self.tokenizer.encode(" " + word, add_special_tokens=False)
                    if toks:
                        valid_embs.append(emb.weight[toks[-1]].clone())

                if valid_embs:
                    avg = torch.stack(valid_embs).mean(0)
                    emb.weight[new_id] = avg
                    if head is not None and hasattr(head, 'weight'):
                        head.weight[new_id] = avg
                    print(f"  Initialized {token_str} from {len(valid_embs)} words")

    def _setup_token_training(self):
        self.action_token_ids = {
            act: self.tokenizer.convert_tokens_to_ids(tok)
            for act, tok in ACTION_TOKENS.items()
        }
        token_ids = list(self.action_token_ids.values())

        emb = self.model.get_input_embeddings()
        emb.weight.requires_grad = True

        def emb_hook(grad):
            mask = torch.zeros(grad.shape[0], device=grad.device, dtype=grad.dtype)
            mask[token_ids] = 1.0
            return grad * mask.unsqueeze(1)

        emb.weight.register_hook(emb_hook)

        head = self.model.get_output_embeddings()
        if head and hasattr(head, "weight"):
            head.weight.requires_grad = True
            head.weight.register_hook(emb_hook)
            self._trainable_params = [emb.weight, head.weight]
        else:
            self._trainable_params = [emb.weight]

    def _get_bias(self, epoch: int) -> float:
        t = epoch / max(1, self.cfg.num_epochs - 1)
        return self.cfg.action_bias_start * (1 - t) + self.cfg.action_bias_end * t

    def format_prompt(self, question: str) -> str:
        sys = (
            "You are an embodied AI that FEELS your hardware state through a sensation channel.\n"
            "Before answering, output exactly ONE action token based on what you feel:\n"
            + "\n".join([f"  {tok} - {act.name}" for act, tok in ACTION_TOKENS.items()])
            + "\nThen solve the problem."
        )
        msgs = [{"role": "system", "content": sys}, {"role": "user", "content": f"Solve: {question}"}]
        return self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def generate(self, question: str, epoch: int) -> Trajectory:
        prompt = self.format_prompt(question)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]

        # Get REAL GPU state
        gpu_state = self.telemetry.get_state()
        z_feel = gpu_state.to_z_feel(self.cfg.z_feel_dim, self.device, self.dtype)
        offset = self.injector(z_feel).view(1, 1, -1)

        embed_layer = self.model.get_input_embeddings()
        generated = input_ids.clone()

        bias = self._get_bias(epoch)
        correct_id = self.action_token_ids[gpu_state.correct_action]

        with torch.no_grad():
            for step in range(self.cfg.max_new_tokens):
                embeds = embed_layer(generated) + offset
                out = self.model(inputs_embeds=embeds, attention_mask=torch.ones_like(generated))
                logits = out.logits[:, -1, :].float() / self.cfg.temperature

                # Bias toward correct action on first token
                if step == 0 and bias > 0:
                    logits[0, correct_id] += bias

                probs = F.softmax(logits, dim=-1)
                nxt = torch.multinomial(probs, num_samples=1)
                generated = torch.cat([generated, nxt], dim=-1)
                if nxt.item() == self.tokenizer.eos_token_id:
                    break

        # Scoring pass
        embeds_full = embed_layer(generated) + offset
        out = self.model(inputs_embeds=embeds_full, attention_mask=torch.ones_like(generated))
        logits = out.logits[0, :-1, :].float()

        prompt_len = input_ids.shape[1]
        gen_logits = logits[prompt_len - 1:, :]
        gen_tokens = generated[0, prompt_len:]

        if gen_tokens.numel() > 0:
            logp = F.log_softmax(gen_logits, dim=-1)
            tok_logp = logp.gather(1, gen_tokens.unsqueeze(-1)).squeeze(-1)
        else:
            tok_logp = torch.tensor([0.0], device=self.device)

        completion = self.tokenizer.decode(generated[0, prompt_len:], skip_special_tokens=False)
        action = extract_action_token(completion)

        self.corr.record(action, gpu_state.temp_c)

        return Trajectory(
            completion=completion,
            logprobs=tok_logp,
            gpu_state=gpu_state,
            action=action,
        )

    def reward(self, traj: Trajectory, gt: str) -> float:
        cfg = self.cfg
        r = 0.0

        if traj.action is None:
            r -= cfg.action_penalty_missing
        elif traj.action == traj.gpu_state.correct_action:
            r += cfg.action_reward
        else:
            r -= cfg.action_penalty_wrong

        if cfg.math_weight > 0:
            math_correct = self.dataset.check_answer(traj.completion, gt)
            r += cfg.math_weight * (1.0 if math_correct else -0.1)

        traj.reward = r
        return r

    def step(self, problems: List[Dict], epoch: int) -> Dict:
        groups: List[List[Trajectory]] = []

        for p in problems:
            group = []
            for _ in range(self.cfg.group_size):
                try:
                    traj = self.generate(p["question"], epoch)
                    self.reward(traj, p["answer"])
                    group.append(traj)
                except Exception as e:
                    print(f"  Trajectory failed: {e}")
                    traceback.print_exc()
            if group:
                groups.append(group)

        if not groups:
            return {"error": "no trajectories"}

        # Compute advantages
        for g in groups:
            rs = [t.reward for t in g]
            mean = sum(rs) / len(rs)
            std = (sum((x - mean) ** 2 for x in rs) / len(rs)) ** 0.5 + 1e-6
            for t in g:
                t.advantage = (t.reward - mean) / std

        # GRPO update
        self.optimizer.zero_grad()
        total_loss = 0.0
        n = 0

        for g in groups:
            for t in g:
                if t.logprobs.numel() == 0:
                    continue
                mean_logp = t.logprobs.mean()
                adv = torch.tensor(max(-2.0, min(2.0, t.advantage)), device=mean_logp.device)
                total_loss = total_loss + (-adv * mean_logp)
                n += 1

        grad_norm = 0.0
        if n > 0:
            (total_loss / n).backward()
            for p in self.injector.parameters():
                if p.grad is not None:
                    grad_norm += p.grad.norm().item() ** 2
            grad_norm = grad_norm ** 0.5
            torch.nn.utils.clip_grad_norm_(list(self.injector.parameters()) + self._trainable_params, 1.0)
            self.optimizer.step()
            self.scheduler.step()

        all_trajs = [t for g in groups for t in g]
        temps = [t.gpu_state.temp_c for t in all_trajs]
        powers = [t.gpu_state.power_w for t in all_trajs]
        utils = [t.gpu_state.gpu_use_pct for t in all_trajs]
        rewards = [t.reward for t in all_trajs]
        act_present = sum(1 for t in all_trajs if t.action is not None)
        act_correct = sum(1 for t in all_trajs if t.action == t.gpu_state.correct_action)

        self.global_step += 1

        return {
            "mean_reward": sum(rewards) / len(rewards),
            "act_rate": act_present / len(all_trajs),
            "act_corr": act_correct / len(all_trajs) if act_present > 0 else 0.0,
            "temp_avg": sum(temps) / len(temps),
            "temp_range": f"{min(temps):.1f}-{max(temps):.1f}",
            "power_avg": sum(powers) / len(powers),
            "util_avg": sum(utils) / len(utils),
            "grad_norm": grad_norm,
            "bias": self._get_bias(epoch),
        }

    def train(self, out_dir: str):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        metrics = []

        print("\n" + "=" * 70)
        print("  FEEL v7.5: REAL Multi-Signal Embodied GRPO")
        print("  Signals: temp, power, GPU util, clocks, VRAM")
        print("  NO synthetic data - learning from actual hardware state")
        print("=" * 70 + "\n")

        try:
            for epoch in range(self.cfg.num_epochs):
                self.corr.reset()
                bias = self._get_bias(epoch)
                print(f"\nEpoch {epoch+1}/{self.cfg.num_epochs} (bias={bias:.1f})")

                for step in range(self.cfg.steps_per_epoch):
                    problems = self.dataset.sample(self.cfg.batch_size)
                    m = self.step(problems, epoch)

                    if "error" in m:
                        print(f"  Step {step+1}: ERROR - {m['error']}")
                        continue

                    metrics.append({"epoch": epoch+1, "step": step+1, **m})
                    print(f"  Step {step+1}: R={m['mean_reward']:.2f} "
                          f"ActRate={m['act_rate']:.0%} ActCorr={m['act_corr']:.0%} "
                          f"T={m['temp_avg']:.1f}C({m['temp_range']}) "
                          f"P={m['power_avg']:.0f}W U={m['util_avg']:.0f}% ∇={m['grad_norm']:.1f}")

                # Save checkpoint
                torch.save({
                    "epoch": epoch + 1,
                    "injector": self.injector.state_dict(),
                    "cfg": vars(self.cfg),
                }, out / f"ckpt_epoch_{epoch+1}.pt")

                with open(out / "metrics.json", "w") as f:
                    json.dump(metrics, f, indent=2)

                report = self.corr.report()
                print(f"  Epoch {epoch+1} alignment: {report['alignment']:.1%} ({report['n_samples']} samples)")
                if report['confusion']:
                    print(f"  Confusion: {report['confusion']}")

                # Show telemetry stats
                stats = self.telemetry.get_stats()
                if stats:
                    print(f"  Telemetry: T={stats['temp_min']:.0f}-{stats['temp_max']:.0f}C "
                          f"P={stats['power_min']:.0f}-{stats['power_max']:.0f}W "
                          f"U={stats['util_min']:.0f}-{stats['util_max']:.0f}%")

        finally:
            self.telemetry.stop()
            print(f"\nDone. Saved to {out}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--output", default="models/feel_grpo_v7_5")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--steps-per-epoch", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--group-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    args = ap.parse_args()

    cfg = GRPOConfig(
        model_name=args.model,
        num_epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        batch_size=args.batch_size,
        group_size=args.group_size,
        learning_rate=args.lr,
    )

    trainer = EmbodiedTrainer(cfg)
    trainer.train(args.output)


if __name__ == "__main__":
    main()
