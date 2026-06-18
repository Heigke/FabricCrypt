#!/usr/bin/env python3
"""
FEEL Token Integration - Terminal Demo
=======================================
Shows FEEL's influence on generation in real-time with colored terminal output.

Run: python scripts/feel_demo_terminal.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import sys
import time

# ANSI colors
class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"

    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_BLUE = "\033[44m"


def color_by_kl(kl: float) -> str:
    """Color text based on KL divergence (FEEL influence)."""
    if kl > 0.5:
        return C.RED + C.BOLD
    elif kl > 0.1:
        return C.YELLOW
    elif kl > 0.01:
        return C.GREEN
    else:
        return C.DIM


def sensor_bar(value: float, width: int = 20, label: str = "") -> str:
    """Create a horizontal bar for sensor value."""
    filled = int(min(1.0, max(0.0, value)) * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"{label:>12}: [{C.CYAN}{bar}{C.RESET}] {value:.3f}"


# ============================================================
# FEEL Components
# ============================================================

class SensorBank(nn.Module):
    def __init__(self):
        super().__init__()
        self.sensor_names = [
            "entropy", "top1_prob", "top5_gap", "logit_std",
            "logit_range", "kurtosis", "skewness", "grad_norm"
        ]

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        logits = logits[:, -1, :].float()
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)

        entropy = -(probs * log_probs).sum(-1) / np.log(logits.shape[-1])
        top1_prob = probs.max(dim=-1).values
        top5 = probs.topk(5, dim=-1).values
        top5_gap = top5[:, 0] - top5[:, -1]
        logit_std = logits.std(dim=-1) / 10.0
        logit_range = (logits.max(dim=-1).values - logits.min(dim=-1).values) / 100.0

        logit_mean = logits.mean(dim=-1, keepdim=True)
        logit_centered = logits - logit_mean
        var = (logit_centered ** 2).mean(dim=-1)
        std = var.sqrt() + 1e-8

        skewness = ((logit_centered / std.unsqueeze(-1)) ** 3).mean(dim=-1) / 10.0
        kurtosis = ((logit_centered / std.unsqueeze(-1)) ** 4).mean(dim=-1) / 100.0
        grad_proxy = (entropy * (1 - entropy)).clamp(0, 1)

        return torch.stack([
            entropy, top1_prob, top5_gap, logit_std,
            logit_range, kurtosis, skewness, grad_proxy
        ], dim=-1)


class FEELProjector(nn.Module):
    def __init__(self, sensor_dim: int = 8, hidden_dim: int = 64, embed_dim: int = 1536):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(sensor_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, embed_dim),
        )
        self._init_near_zero()

    def _init_near_zero(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=1e-4)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, sensors: torch.Tensor) -> torch.Tensor:
        return self.net(sensors)


class FEELStream(nn.Module):
    def __init__(self, embed_dim: int = 1536):
        super().__init__()
        self.sensor_bank = SensorBank()
        self.projector = FEELProjector(embed_dim=embed_dim)
        self.alpha = nn.Parameter(torch.tensor(-4.0))

    def forward(self, logits: torch.Tensor) -> tuple:
        sensors = self.sensor_bank(logits)
        z_feel = self.projector(sensors)
        alpha = F.softplus(self.alpha)
        feel_embed = alpha * z_feel
        return feel_embed, sensors, alpha


# ============================================================
# Demo
# ============================================================

class TerminalDemo:
    def __init__(self, model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"{C.CYAN}Loading model on {self.device}...{C.RESET}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            trust_remote_code=True,
            device_map="auto"
        )
        self.model.eval()

        embed_dim = self.model.config.hidden_size
        self.feel_stream = FEELStream(embed_dim=embed_dim).to(self.device)

        # Load trained checkpoint
        checkpoint_path = Path(__file__).parent.parent / "results/feel_training/feel_projector_checkpoint.pt"
        if checkpoint_path.exists():
            print(f"{C.GREEN}✓ Loading trained FEEL checkpoint{C.RESET}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            # Handle different checkpoint formats
            if "feel_stream_state" in checkpoint:
                self.feel_stream.load_state_dict(checkpoint["feel_stream_state"])
            elif "feel_stream" in checkpoint:
                if "alpha" in checkpoint["feel_stream"]:
                    self.feel_stream.alpha.data.copy_(checkpoint["feel_stream"]["alpha"])
                    print(f"{C.GREEN}  Loaded trained alpha{C.RESET}")

        self.feel_stream.eval()
        print(f"{C.GREEN}✓ Ready!{C.RESET}\n")

    def generate_live(self, prompt: str, max_tokens: int = 64, temperature: float = 0.7,
                      injection_strength: float = 1.0, show_sensors: bool = True):
        """Generate with live visualization."""

        print(f"\n{C.BOLD}{'='*60}{C.RESET}")
        print(f"{C.BOLD}Prompt:{C.RESET} {prompt}")
        print(f"{C.BOLD}{'='*60}{C.RESET}\n")

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        current_ids = input_ids.clone()
        past_kv = None

        kl_history = []
        sensor_names = ["entropy", "top1_prob", "top5_gap", "logit_std",
                       "logit_range", "kurtosis", "skewness", "grad_norm"]

        print(f"{C.CYAN}Generation (colored by FEEL influence):{C.RESET}\n")
        sys.stdout.write(f"{C.DIM}{prompt}{C.RESET}")
        sys.stdout.flush()

        for step in range(max_tokens):
            with torch.no_grad():
                # Base forward
                if past_kv is None:
                    outputs_base = self.model(current_ids, use_cache=True)
                else:
                    outputs_base = self.model(current_ids[:, -1:], past_key_values=past_kv, use_cache=True)

                logits_base = outputs_base.logits

                # FEEL forward
                feel_embed, sensors, alpha = self.feel_stream(logits_base)
                feel_embed = feel_embed * injection_strength

                if past_kv is None:
                    embeds = self.model.get_input_embeddings()(current_ids)
                else:
                    embeds = self.model.get_input_embeddings()(current_ids[:, -1:])

                embeds = embeds + feel_embed.unsqueeze(1)

                outputs_feel = self.model(
                    inputs_embeds=embeds,
                    past_key_values=past_kv,
                    use_cache=True
                )
                logits_feel = outputs_feel.logits
                past_kv = outputs_feel.past_key_values

                # KL divergence
                p_base = F.softmax(logits_base[:, -1, :].float(), dim=-1)
                p_feel = F.softmax(logits_feel[:, -1, :].float(), dim=-1)
                kl = F.kl_div(p_feel.log(), p_base, reduction='batchmean').item()
                kl_history.append(kl)

                # Sample
                if temperature > 0:
                    probs = F.softmax(logits_feel[:, -1, :] / temperature, dim=-1)
                    next_token = torch.multinomial(probs, 1)
                else:
                    next_token = logits_feel[:, -1, :].argmax(dim=-1, keepdim=True)

                token_str = self.tokenizer.decode([next_token.item()])
                color = color_by_kl(kl)
                sys.stdout.write(f"{color}{token_str}{C.RESET}")
                sys.stdout.flush()

                current_ids = torch.cat([current_ids, next_token], dim=-1)

                if next_token.item() == self.tokenizer.eos_token_id:
                    break

        print("\n")

        # Summary
        print(f"{C.BOLD}{'='*60}{C.RESET}")
        print(f"{C.BOLD}FEEL Metrics Summary:{C.RESET}")
        print(f"{C.BOLD}{'='*60}{C.RESET}")

        avg_kl = np.mean(kl_history)
        max_kl = np.max(kl_history)
        print(f"  Avg KL Divergence: {C.YELLOW}{avg_kl:.4f}{C.RESET}")
        print(f"  Max KL Divergence: {C.YELLOW}{max_kl:.4f}{C.RESET}")
        print(f"  Alpha (gate):      {C.YELLOW}{alpha.item():.4f}{C.RESET}")
        print(f"  Tokens generated:  {len(kl_history)}")

        # KL histogram
        print(f"\n{C.BOLD}KL Divergence Distribution:{C.RESET}")
        bins = [0, 0.01, 0.05, 0.1, 0.5, float('inf')]
        labels = ["<0.01 (minimal)", "0.01-0.05 (low)", "0.05-0.1 (medium)",
                  "0.1-0.5 (high)", ">0.5 (strong)"]
        colors = [C.DIM, C.GREEN, C.YELLOW, C.RED, C.RED + C.BOLD]

        for i in range(len(bins) - 1):
            count = sum(1 for kl in kl_history if bins[i] <= kl < bins[i+1])
            pct = count / len(kl_history) * 100
            bar = "█" * int(pct / 2)
            print(f"  {colors[i]}{labels[i]:20}: {bar} {pct:.1f}%{C.RESET}")

        if show_sensors:
            print(f"\n{C.BOLD}Final Sensor Values:{C.RESET}")
            sensors_np = sensors[0].cpu().numpy()
            for name, val in zip(sensor_names, sensors_np):
                print(sensor_bar(val, label=name))

        return kl_history


def main():
    print(f"""
{C.CYAN}{C.BOLD}╔══════════════════════════════════════════════════════════╗
║         FEEL Token Integration - Terminal Demo            ║
║     Feeling Embeddings for Embodied Learning (v3.1)       ║
╚══════════════════════════════════════════════════════════╝{C.RESET}

{C.YELLOW}Color Legend:{C.RESET}
  {C.DIM}dim{C.RESET}      = minimal FEEL influence (KL < 0.01)
  {C.GREEN}green{C.RESET}    = low influence (KL 0.01-0.1)
  {C.YELLOW}yellow{C.RESET}   = medium influence (KL 0.1-0.5)
  {C.RED}red/bold{C.RESET} = strong influence (KL > 0.5)
""")

    demo = TerminalDemo()

    # Example prompts
    prompts = [
        "Describe what uncertainty feels like:",
        "Explain consciousness in simple terms:",
        "What is 17 * 23? Let me think step by step:",
    ]

    while True:
        print(f"\n{C.CYAN}Choose a prompt or enter your own:{C.RESET}")
        for i, p in enumerate(prompts):
            print(f"  [{i+1}] {p[:50]}...")
        print(f"  [0] Enter custom prompt")
        print(f"  [q] Quit\n")

        choice = input(f"{C.BOLD}> {C.RESET}").strip()

        if choice.lower() == 'q':
            print(f"\n{C.GREEN}Goodbye!{C.RESET}")
            break
        elif choice == '0':
            prompt = input(f"{C.BOLD}Enter prompt: {C.RESET}")
        elif choice.isdigit() and 1 <= int(choice) <= len(prompts):
            prompt = prompts[int(choice) - 1]
        else:
            print(f"{C.RED}Invalid choice{C.RESET}")
            continue

        # Parameters
        print(f"\n{C.DIM}(Press Enter for defaults){C.RESET}")
        max_tok = input(f"Max tokens [{C.YELLOW}64{C.RESET}]: ").strip() or "64"
        temp = input(f"Temperature [{C.YELLOW}0.7{C.RESET}]: ").strip() or "0.7"
        strength = input(f"FEEL strength [{C.YELLOW}1.0{C.RESET}]: ").strip() or "1.0"

        demo.generate_live(
            prompt=prompt,
            max_tokens=int(max_tok),
            temperature=float(temp),
            injection_strength=float(strength)
        )


if __name__ == "__main__":
    main()
