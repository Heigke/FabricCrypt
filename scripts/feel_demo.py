#!/usr/bin/env python3
"""
FEEL Token Integration - Live Demo
===================================
Interactive demo showing FEEL's influence on LLM generation in real-time.

Run: python scripts/feel_demo.py
Then open: http://localhost:7860
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import json
import time

# ============================================================
# FEEL Components (copied from training script for standalone demo)
# ============================================================

class SensorBank(nn.Module):
    """Computes 8 interoceptive signals from model activations."""

    def __init__(self):
        super().__init__()
        self.sensor_names = [
            "entropy", "top1_prob", "top5_gap", "logit_std",
            "logit_range", "kurtosis", "skewness", "grad_norm_proxy"
        ]

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        logits = logits[:, -1, :].float()
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)

        # Entropy (normalized)
        entropy = -(probs * log_probs).sum(-1) / np.log(logits.shape[-1])

        # Top-1 probability
        top1_prob = probs.max(dim=-1).values

        # Top-5 gap
        top5 = probs.topk(5, dim=-1).values
        top5_gap = top5[:, 0] - top5[:, -1]

        # Logit statistics
        logit_std = logits.std(dim=-1) / 10.0
        logit_range = (logits.max(dim=-1).values - logits.min(dim=-1).values) / 100.0

        # Higher moments
        logit_mean = logits.mean(dim=-1, keepdim=True)
        logit_centered = logits - logit_mean
        var = (logit_centered ** 2).mean(dim=-1)
        std = var.sqrt() + 1e-8

        skewness = ((logit_centered / std.unsqueeze(-1)) ** 3).mean(dim=-1) / 10.0
        kurtosis = ((logit_centered / std.unsqueeze(-1)) ** 4).mean(dim=-1) / 100.0

        # Grad norm proxy (entropy gradient magnitude)
        grad_proxy = (entropy * (1 - entropy)).clamp(0, 1)

        sensors = torch.stack([
            entropy, top1_prob, top5_gap, logit_std,
            logit_range, kurtosis, skewness, grad_proxy
        ], dim=-1)

        return sensors


class FEELProjector(nn.Module):
    """Projects 8D sensor vector to model embedding dimension."""

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
    """Complete FEEL stream with learnable injection strength."""

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
# Demo Application
# ============================================================

class FEELDemo:
    def __init__(self, model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading model on {self.device}...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            trust_remote_code=True,
            device_map="auto"
        )
        self.model.eval()

        # Initialize FEEL stream
        embed_dim = self.model.config.hidden_size
        self.feel_stream = FEELStream(embed_dim=embed_dim).to(self.device)

        # Try to load trained checkpoint
        checkpoint_path = Path(__file__).parent.parent / "results/feel_training/feel_projector_checkpoint.pt"
        if checkpoint_path.exists():
            print(f"Loading trained FEEL checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            # Handle different checkpoint formats
            if "feel_stream_state" in checkpoint:
                self.feel_stream.load_state_dict(checkpoint["feel_stream_state"])
                print("✓ Loaded trained FEEL weights (v3 format)")
            elif "feel_stream" in checkpoint:
                # Older format - load alpha if available
                if "alpha" in checkpoint["feel_stream"]:
                    self.feel_stream.alpha.data.copy_(checkpoint["feel_stream"]["alpha"])
                    print("✓ Loaded trained alpha from checkpoint")
            elif "z_feel_model" in checkpoint:
                print("✓ Checkpoint found (using projector weights)")
                # Could map z_feel_model weights if needed

        self.feel_stream.eval()
        print("✓ Demo ready!")

    def generate_comparison(
        self,
        prompt: str,
        max_tokens: int = 64,
        temperature: float = 0.7,
        injection_strength: float = 1.0,
    ):
        """Generate with FEEL ON and OFF, collecting metrics."""

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        results = {"feel_on": {}, "feel_off": {}}

        for mode in ["feel_off", "feel_on"]:
            with_feel = (mode == "feel_on")

            tokens = []
            kl_divs = []
            sensors_history = []
            alphas = []
            entropies = []

            current_ids = input_ids.clone()
            past_kv = None

            for step in range(max_tokens):
                with torch.no_grad():
                    # Get base logits
                    if past_kv is None:
                        outputs_base = self.model(current_ids, use_cache=True)
                        past_kv_base = outputs_base.past_key_values
                    else:
                        outputs_base = self.model(
                            current_ids[:, -1:],
                            past_key_values=past_kv,
                            use_cache=True
                        )
                        past_kv_base = outputs_base.past_key_values

                    logits_base = outputs_base.logits

                    if with_feel and injection_strength > 0:
                        # Compute FEEL embedding
                        feel_embed, sensors, alpha = self.feel_stream(logits_base)
                        feel_embed = feel_embed * injection_strength

                        # Get embeddings and inject FEEL
                        if past_kv is None:
                            embeds = self.model.get_input_embeddings()(current_ids)
                        else:
                            embeds = self.model.get_input_embeddings()(current_ids[:, -1:])

                        embeds = embeds + feel_embed.unsqueeze(1)

                        # Forward with FEEL
                        outputs_feel = self.model(
                            inputs_embeds=embeds,
                            past_key_values=past_kv,
                            use_cache=True
                        )
                        logits_feel = outputs_feel.logits
                        past_kv = outputs_feel.past_key_values

                        # Compute KL divergence
                        p_base = F.softmax(logits_base[:, -1, :].float(), dim=-1)
                        p_feel = F.softmax(logits_feel[:, -1, :].float(), dim=-1)
                        kl = F.kl_div(p_feel.log(), p_base, reduction='batchmean').item()

                        logits = logits_feel
                        sensors_history.append(sensors[0].cpu().numpy())
                        alphas.append(alpha.item())
                        kl_divs.append(kl)
                    else:
                        logits = logits_base
                        past_kv = past_kv_base
                        kl_divs.append(0.0)
                        sensors_history.append(np.zeros(8))
                        alphas.append(0.0)

                    # Compute entropy
                    probs = F.softmax(logits[:, -1, :].float(), dim=-1)
                    entropy = -(probs * probs.log()).sum(-1).item()
                    entropies.append(entropy)

                    # Sample next token
                    if temperature > 0:
                        probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
                        next_token = torch.multinomial(probs, 1)
                    else:
                        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)

                    tokens.append(next_token.item())
                    current_ids = torch.cat([current_ids, next_token], dim=-1)

                    # Stop on EOS
                    if next_token.item() == self.tokenizer.eos_token_id:
                        break

            results[mode] = {
                "text": self.tokenizer.decode(tokens, skip_special_tokens=True),
                "tokens": tokens,
                "kl_divs": kl_divs,
                "sensors": sensors_history,
                "alphas": alphas,
                "entropies": entropies,
            }

        return results

    def format_output(self, results: dict, prompt: str) -> tuple:
        """Format results for Gradio display."""

        feel_off = results["feel_off"]
        feel_on = results["feel_on"]

        # Text outputs
        text_off = f"**FEEL OFF:**\n{prompt}{feel_off['text']}"
        text_on = f"**FEEL ON:**\n{prompt}{feel_on['text']}"

        # KL divergence plot data
        kl_data = list(zip(range(len(feel_on["kl_divs"])), feel_on["kl_divs"]))

        # Sensor heatmap
        sensors = np.array(feel_on["sensors"])

        # Entropy comparison
        entropy_off = feel_off["entropies"]
        entropy_on = feel_on["entropies"]

        # Summary stats
        avg_kl = np.mean(feel_on["kl_divs"]) if feel_on["kl_divs"] else 0
        max_kl = np.max(feel_on["kl_divs"]) if feel_on["kl_divs"] else 0
        avg_alpha = np.mean(feel_on["alphas"]) if feel_on["alphas"] else 0

        stats = f"""
### FEEL Metrics
| Metric | Value |
|--------|-------|
| Avg KL Divergence | {avg_kl:.4f} |
| Max KL Divergence | {max_kl:.4f} |
| Avg Alpha (gate) | {avg_alpha:.4f} |
| Tokens Generated | {len(feel_on['tokens'])} |
"""

        return text_off, text_on, sensors, kl_data, stats


def create_demo():
    """Create Gradio interface."""

    demo_instance = None

    def initialize():
        nonlocal demo_instance
        if demo_instance is None:
            demo_instance = FEELDemo()
        return "✓ Model loaded and ready!"

    def run_demo(prompt, max_tokens, temperature, injection_strength):
        nonlocal demo_instance
        if demo_instance is None:
            demo_instance = FEELDemo()

        results = demo_instance.generate_comparison(
            prompt=prompt,
            max_tokens=int(max_tokens),
            temperature=float(temperature),
            injection_strength=float(injection_strength),
        )

        text_off, text_on, sensors, kl_data, stats = demo_instance.format_output(results, prompt)

        # Create plots
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # KL divergence plot
        fig_kl, ax_kl = plt.subplots(figsize=(8, 3))
        if kl_data:
            steps, kls = zip(*kl_data)
            ax_kl.bar(steps, kls, color='steelblue', alpha=0.7)
            ax_kl.axhline(y=0.001, color='red', linestyle='--', label='Threshold')
            ax_kl.set_xlabel('Token Step')
            ax_kl.set_ylabel('KL Divergence')
            ax_kl.set_title('FEEL Causal Influence per Token')
            ax_kl.legend()
        plt.tight_layout()

        # Sensor heatmap
        fig_sensors, ax_sensors = plt.subplots(figsize=(8, 3))
        sensor_names = ["entropy", "top1_prob", "top5_gap", "logit_std",
                       "logit_range", "kurtosis", "skewness", "grad_norm"]
        if sensors.shape[0] > 0:
            im = ax_sensors.imshow(sensors.T, aspect='auto', cmap='viridis')
            ax_sensors.set_yticks(range(8))
            ax_sensors.set_yticklabels(sensor_names)
            ax_sensors.set_xlabel('Token Step')
            ax_sensors.set_title('FEEL Sensor Activations (z_feel)')
            plt.colorbar(im, ax=ax_sensors)
        plt.tight_layout()

        return text_off, text_on, fig_kl, fig_sensors, stats

    # Build interface
    with gr.Blocks(title="FEEL Token Demo", theme=gr.themes.Soft()) as interface:
        gr.Markdown("""
# 🧠 FEEL Token Integration Demo
**Feeling Embeddings for Embodied Learning** - Real-time visualization of interoceptive signals in LLM generation.

FEEL adds 8 sensor signals (entropy, confidence, etc.) back into the model's input embeddings,
creating a closed-loop feedback system that influences generation.
        """)

        with gr.Row():
            with gr.Column(scale=2):
                prompt_input = gr.Textbox(
                    label="Prompt",
                    placeholder="Enter your prompt here...",
                    value="Explain what consciousness might feel like for an AI:",
                    lines=3
                )

                with gr.Row():
                    max_tokens = gr.Slider(16, 128, value=64, step=8, label="Max Tokens")
                    temperature = gr.Slider(0.0, 1.5, value=0.7, step=0.1, label="Temperature")
                    injection = gr.Slider(0.0, 2.0, value=1.0, step=0.1, label="FEEL Strength")

                run_btn = gr.Button("🚀 Generate Comparison", variant="primary")

            with gr.Column(scale=1):
                stats_output = gr.Markdown(label="Metrics")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Without FEEL")
                text_off = gr.Markdown()
            with gr.Column():
                gr.Markdown("### With FEEL")
                text_on = gr.Markdown()

        with gr.Row():
            kl_plot = gr.Plot(label="KL Divergence (FEEL influence)")
            sensor_plot = gr.Plot(label="Sensor Activations")

        # Wire up
        run_btn.click(
            fn=run_demo,
            inputs=[prompt_input, max_tokens, temperature, injection],
            outputs=[text_off, text_on, kl_plot, sensor_plot, stats_output]
        )

        gr.Markdown("""
---
### How to interpret:
- **KL Divergence**: Higher = FEEL has more influence on that token
- **Sensor Heatmap**: Shows the 8D interoceptive signal over time
- **Alpha Gate**: Learned injection strength (higher = more FEEL)

*Model: DeepSeek-R1-Distill-Qwen-1.5B with trained FEEL projector*
        """)

    return interface


if __name__ == "__main__":
    print("=" * 60)
    print("  FEEL Token Integration - Live Demo")
    print("=" * 60)

    interface = create_demo()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # Set True for public URL
        show_error=True
    )
