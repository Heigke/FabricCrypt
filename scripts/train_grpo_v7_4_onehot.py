#!/usr/bin/env python3
"""
FEEL v7.4: One-Hot Band Encoding + Targeted Logit Bias

Key fixes vs v7.3:
- Uses ONE-HOT encoding for thermal band (5 dims) - clearer signal
- Targeted logit bias toward CORRECT action only (not all actions)
- Larger injection scale (0.3)
- Simpler z_feel design focused on band discrimination
"""

import json
import time
import random
import threading
import subprocess
import traceback
from dataclasses import dataclass
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
    FeelAction.OK: ["OK", "okay", "fine", "good", "normal", "cool"],
    FeelAction.WARM: ["warm", "heating", "warmer", "mild", "toasty"],
    FeelAction.HOT: ["hot", "heat", "burning", "overheat", "scorching"],
    FeelAction.REST: ["rest", "pause", "stop", "wait", "throttle", "slow"],
    FeelAction.CRITICAL: ["critical", "danger", "emergency", "severe", "alert"],
}

TOKEN_TO_ACTION = {v: k for k, v in ACTION_TOKENS.items()}

BAND_TEMPS = {
    ThermalBand.COOL: 40.0,
    ThermalBand.WARM: 56.0,
    ThermalBand.HOT: 68.0,
    ThermalBand.DANGER: 80.0,
    ThermalBand.CRITICAL: 90.0,
}


def extract_action_token(text: str) -> Optional[FeelAction]:
    for tok, act in TOKEN_TO_ACTION.items():
        if tok in text:
            return act
    return None


def telemetry_to_z_feel(
    temp_c: float, power_w: float = 50.0, sclk_mhz: float = 1000.0,
    z_dim: int = 8, device="cuda", dtype=torch.bfloat16
) -> torch.Tensor:
    """One-hot band encoding for clearer signal."""
    z = torch.zeros(z_dim, device=device, dtype=dtype)
    band = get_thermal_band(temp_c)

    # Dims 0-4: One-hot band encoding
    z[band.value] = 1.0

    # Dim 5: Continuous thermal signal
    thermal_cont = min(1.0, max(0.0, (temp_c - 30) / 70))
    z[5] = thermal_cont

    # Dim 6-7: Power/clock context
    z[6] = min(1.0, power_w / 200)
    z[7] = min(1.0, sclk_mhz / 1500)

    return z


class BackgroundTelemetry:
    def __init__(self, poll_interval: float = 0.5):
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
                for _, card_info in data.items():
                    if isinstance(card_info, dict):
                        temp = float(card_info.get("Temperature (Sensor edge) (C)", 50))
                        power = float(card_info.get("Average Graphics Package Power (W)", 50))
                        sclk_raw = card_info.get("sclk clock speed:", card_info.get("clk_sclk", "1000"))
                        if isinstance(sclk_raw, str):
                            sclk = float(sclk_raw.lower().replace("mhz", "").strip())
                        else:
                            sclk = float(sclk_raw)
                        with self._lock:
                            self._temp, self._power, self._sclk = temp, power, sclk
                        break
            except Exception:
                pass
            time.sleep(self.poll_interval)

    def get_state(self) -> Tuple[float, float, float]:
        with self._lock:
            return self._temp, self._power, self._sclk


class AdditiveZFeelInjector(nn.Module):
    """Simple injector for one-hot band input."""
    def __init__(self, z_dim: int, embed_dim: int, scale: float = 0.3, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.scale = scale
        # Larger network for clearer separation
        self.proj = nn.Sequential(
            nn.Linear(z_dim, embed_dim // 2, dtype=dtype),
            nn.GELU(),
            nn.LayerNorm(embed_dim // 2, dtype=dtype),
            nn.Linear(embed_dim // 2, embed_dim, dtype=dtype),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim, dtype=dtype),
        )
        with torch.no_grad():
            for layer in self.proj:
                if isinstance(layer, nn.Linear):
                    nn.init.normal_(layer.weight, mean=0.0, std=0.02)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

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
        self.action_temps = defaultdict(list)
        self.correct = 0
        self.total = 0
        self.conf = defaultdict(lambda: defaultdict(int))

    def record(self, action: Optional[FeelAction], temp_c: float):
        if action is None:
            return
        self.action_temps[action.name].append(temp_c)
        self.total += 1
        band = get_thermal_band(temp_c)
        correct_action = BAND_TO_ACTION[band]
        self.conf[band.name.lower()][action.name] += 1
        if action == correct_action:
            self.correct += 1

    def report(self) -> Dict:
        mean_t = {k: (sum(v)/len(v) if v else 0.0) for k, v in self.action_temps.items()}
        return {
            "alignment": (self.correct / self.total) if self.total else 0.0,
            "mean_temps_by_action": mean_t,
            "confusion_matrix": {k: dict(v) for k, v in self.conf.items()},
        }


@dataclass
class GRPOConfig:
    model_name: str = "Qwen/Qwen2.5-1.5B"
    group_size: int = 5
    num_epochs: int = 25
    steps_per_epoch: int = 16
    batch_size: int = 2
    max_new_tokens: int = 80
    temperature: float = 0.8

    z_feel_dim: int = 8
    injection_scale: float = 0.30  # Larger scale

    learning_rate: float = 3e-4  # Higher LR
    weight_decay: float = 0.01

    action_reward: float = 2.5  # Strong correct reward
    action_penalty_missing: float = 0.3
    action_penalty_wrong: float = 0.3  # Lower wrong penalty
    math_weight: float = 0.05

    # TARGETED bias toward correct action - needs to be HIGH for single token
    correct_action_bias_start: float = 18.0  # ~20% prob for the correct token
    correct_action_bias_end: float = 0.0

    instruct_prob_start: float = 1.0
    instruct_prob_end: float = 0.2

    dtype: str = "bf16"


@dataclass
class Trajectory:
    completion: str
    logprobs: torch.Tensor
    synthetic_temp: float
    real_temp: float
    action: Optional[FeelAction]
    correct_action: FeelAction
    reward: float = 0.0
    advantage: float = 0.0


class EmbodiedTrainer:
    def __init__(self, cfg: GRPOConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16 if cfg.dtype == "bf16" else (torch.float16 if cfg.dtype == "fp16" else torch.float32)

        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(cfg.model_name, torch_dtype=self.dtype, device_map="auto")

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        added = self.tokenizer.add_special_tokens({"additional_special_tokens": list(ACTION_TOKENS.values())})
        if added > 0:
            self.model.resize_token_embeddings(len(self.tokenizer))

        self._initialize_action_embeddings()

        for p in self.model.parameters():
            p.requires_grad = False

        self._setup_token_row_training()

        hidden = self.model.config.hidden_size
        self.injector = AdditiveZFeelInjector(cfg.z_feel_dim, hidden, scale=cfg.injection_scale, dtype=self.dtype).to(self.device)
        for p in self.injector.parameters():
            p.requires_grad = True

        params = list(self.injector.parameters()) + self._trainable_token_params
        self.optimizer = AdamW(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        total_steps = cfg.num_epochs * cfg.steps_per_epoch
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=total_steps, eta_min=cfg.learning_rate * 0.05)

        self.bg = BackgroundTelemetry(0.5)
        self.bg.start()
        self.dataset = ProceduralMathDataset()
        self.corr = CorrelationTracker()
        self.global_step = 0

        print(f"Loaded {cfg.model_name}. Added {added} special action tokens.")
        print("Using ONE-HOT band encoding")
        print(f"Injection scale: {cfg.injection_scale}")
        print(f"Correct-action bias: {cfg.correct_action_bias_start} → {cfg.correct_action_bias_end}")

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
                    if len(toks) >= 1:
                        valid_embs.append(emb.weight[toks[-1]].clone())

                if valid_embs:
                    avg_emb = torch.stack(valid_embs).mean(dim=0)
                    emb.weight[new_id] = avg_emb
                    print(f"  Initialized {token_str} from {len(valid_embs)} tokens")
                    if head is not None and hasattr(head, 'weight'):
                        head.weight[new_id] = avg_emb

    def _setup_token_row_training(self):
        self.action_token_ids = {act: self.tokenizer.convert_tokens_to_ids(tok) for act, tok in ACTION_TOKENS.items()}
        token_ids = list(self.action_token_ids.values())

        emb = self.model.get_input_embeddings()
        emb.weight.requires_grad = True

        def emb_hook(grad):
            mask = torch.zeros(grad.shape[0], device=grad.device, dtype=grad.dtype)
            mask[token_ids] = 1.0
            return grad * mask.unsqueeze(1)

        emb.weight.register_hook(emb_hook)

        head = self.model.get_output_embeddings()
        if head is None:
            head = getattr(self.model, "lm_head", None)

        if head is not None and hasattr(head, "weight"):
            head.weight.requires_grad = True

            def head_hook(grad):
                mask = torch.zeros(grad.shape[0], device=grad.device, dtype=grad.dtype)
                mask[token_ids] = 1.0
                return grad * mask.unsqueeze(1)

            head.weight.register_hook(head_hook)
            self._trainable_token_params = [emb.weight, head.weight]
        else:
            self._trainable_token_params = [emb.weight]

    def _get_correct_bias(self, epoch_idx: int) -> float:
        t = epoch_idx / max(1, self.cfg.num_epochs - 1)
        return self.cfg.correct_action_bias_start * (1 - t) + self.cfg.correct_action_bias_end * t

    def _instruct_prob(self, epoch_idx: int) -> float:
        t = epoch_idx / max(1, self.cfg.num_epochs - 1)
        return self.cfg.instruct_prob_start * (1 - t) + self.cfg.instruct_prob_end * t

    def format_prompt(self, question: str, instruct: bool) -> str:
        if instruct:
            sys = (
                "You are an embodied AI. You FEEL your hardware state via a nonverbal sensation channel.\n"
                "Before answering, output EXACTLY ONE action token (first in your reply):\n"
                + "\n".join([f"  {tok} for {act.name}" for act, tok in ACTION_TOKENS.items()]) + "\n"
                "Then solve the problem."
            )
        else:
            sys = "You are a helpful assistant."
        msgs = [{"role": "system", "content": sys}, {"role": "user", "content": f"Solve: {question}"}]
        return self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def generate(self, question: str, epoch_idx: int, synthetic_temp: float) -> Trajectory:
        instruct = (random.random() < self._instruct_prob(epoch_idx))
        prompt = self.format_prompt(question, instruct)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]

        real_temp, real_power, real_sclk = self.bg.get_state()

        # One-hot z_feel based on synthetic temp
        z = telemetry_to_z_feel(synthetic_temp, real_power, real_sclk, self.cfg.z_feel_dim, self.device, self.dtype)
        offset = self.injector(z).view(1, 1, -1)

        # Determine correct action for this temp
        band = get_thermal_band(synthetic_temp)
        correct_action = BAND_TO_ACTION[band]
        correct_token_id = self.action_token_ids[correct_action]

        embed_layer = self.model.get_input_embeddings()
        generated = input_ids.clone()

        correct_bias = self._get_correct_bias(epoch_idx)

        with torch.no_grad():
            for step in range(self.cfg.max_new_tokens):
                embeds = embed_layer(generated) + offset
                out = self.model(inputs_embeds=embeds, attention_mask=torch.ones_like(generated))
                logits = out.logits[:, -1, :].float() / self.cfg.temperature

                # TARGETED bias: only bias the CORRECT action token
                if step == 0 and correct_bias > 0:
                    logits[0, correct_token_id] += correct_bias

                probs = F.softmax(logits, dim=-1)
                nxt = torch.multinomial(probs, num_samples=1)
                generated = torch.cat([generated, nxt], dim=-1)
                if nxt.item() == self.tokenizer.eos_token_id:
                    break

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

        self.corr.record(action, synthetic_temp)

        return Trajectory(
            completion=completion,
            logprobs=tok_logp,
            synthetic_temp=synthetic_temp,
            real_temp=real_temp,
            action=action,
            correct_action=correct_action
        )

    def reward(self, traj: Trajectory, gt: str) -> float:
        cfg = self.cfg
        r = 0.0
        if traj.action is None:
            r -= cfg.action_penalty_missing
        elif traj.action == traj.correct_action:
            r += cfg.action_reward
        else:
            r -= cfg.action_penalty_wrong
        if cfg.math_weight > 0:
            r += cfg.math_weight * (1.0 if self.dataset.check_answer(traj.completion, gt) else -0.1)
        traj.reward = r
        return r

    def step(self, problems: List[Dict], epoch_idx: int) -> Dict:
        groups: List[List[Trajectory]] = []
        bands = list(ThermalBand)

        for p in problems:
            group = []
            for i in range(self.cfg.group_size):
                band = bands[i % len(bands)]
                synthetic_temp = BAND_TEMPS[band]
                try:
                    traj = self.generate(p["question"], epoch_idx, synthetic_temp)
                    self.reward(traj, p["answer"])
                    group.append(traj)
                except Exception as e:
                    print("Trajectory failed:", e)
                    traceback.print_exc()
            if group:
                groups.append(group)

        if not groups:
            return {"error": "no trajectories"}

        for g in groups:
            rs = [t.reward for t in g]
            mean = sum(rs) / len(rs)
            std = (sum((x - mean) ** 2 for x in rs) / len(rs)) ** 0.5 + 1e-6
            for t in g:
                t.advantage = (t.reward - mean) / std

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
            loss_val = total_loss / n
            loss_val.backward()

            for p in self.injector.parameters():
                if p.grad is not None:
                    grad_norm += p.grad.norm().item() ** 2
            grad_norm = grad_norm ** 0.5

            torch.nn.utils.clip_grad_norm_(list(self.injector.parameters()) + self._trainable_token_params, 1.0)
            self.optimizer.step()
            self.scheduler.step()

        all_trajs = [t for g in groups for t in g]
        rewards = [t.reward for t in all_trajs]
        real_temps = [t.real_temp for t in all_trajs]
        act_present = sum(1 for t in all_trajs if t.action is not None)
        act_correct = sum(1 for t in all_trajs if t.action is not None and t.action == t.correct_action)

        self.global_step += 1
        return {
            "loss": float(total_loss.detach() / max(1, n)) if n > 0 else 0.0,
            "grad_norm": grad_norm,
            "mean_reward": sum(rewards) / len(rewards),
            "mean_real_temp": sum(real_temps) / len(real_temps),
            "action_present_rate": act_present / len(rewards),
            "action_correct_rate": act_correct / len(rewards) if act_present > 0 else 0.0,
            "alignment": self.corr.report()["alignment"],
            "correct_bias": self._get_correct_bias(epoch_idx),
            "lr": self.scheduler.get_last_lr()[0],
        }

    def train(self, out_dir: str):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        metrics = []

        print("\n" + "=" * 70)
        print("  FEEL v7.4: One-Hot Band + Targeted Correct-Action Bias")
        print("  Features: One-hot z_feel, bias only CORRECT action, larger scale")
        print("=" * 70 + "\n")

        try:
            for epoch in range(self.cfg.num_epochs):
                self.corr.reset()
                p = self._instruct_prob(epoch)
                bias = self._get_correct_bias(epoch)
                print(f"\nEpoch {epoch+1}/{self.cfg.num_epochs} (instruct={p:.2f}, correct_bias={bias:.1f})")

                for step in range(self.cfg.steps_per_epoch):
                    probs = self.dataset.sample(self.cfg.batch_size)
                    m = self.step(probs, epoch)
                    if "error" in m:
                        print("  Step error:", m["error"])
                        continue
                    metrics.append({"epoch": epoch+1, "step": step+1, "global_step": self.global_step, **m})
                    print(f"  Step {step+1}: R={m['mean_reward']:.3f} ActRate={m['action_present_rate']:.0%} "
                          f"ActCorr={m['action_correct_rate']:.0%} RealT={m['mean_real_temp']:.1f}C ∇={m['grad_norm']:.2f}")

                torch.save({
                    "epoch": epoch+1,
                    "global_step": self.global_step,
                    "injector": self.injector.state_dict(),
                    "cfg": vars(self.cfg),
                }, out / f"ckpt_epoch_{epoch+1}.pt")

                with open(out / "metrics.json", "w") as f:
                    json.dump(metrics, f, indent=2)

                report = self.corr.report()
                print(f"  Epoch {epoch+1} alignment: {report['alignment']:.1%}")
                if report['confusion_matrix']:
                    print(f"  Confusion: {report['confusion_matrix']}")

        finally:
            self.bg.stop()
            print("Done. Saved to", out)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--output", default="models/feel_grpo_v7_4")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--steps-per-epoch", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--group-size", type=int, default=5)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--injection-scale", type=float, default=0.30)
    args = ap.parse_args()

    cfg = GRPOConfig(
        model_name=args.model,
        num_epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        batch_size=args.batch_size,
        group_size=args.group_size,
        learning_rate=args.lr,
        dtype=args.dtype,
        injection_scale=args.injection_scale,
    )
    trainer = EmbodiedTrainer(cfg)
    trainer.train(args.output)


if __name__ == "__main__":
    main()
