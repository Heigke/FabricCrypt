#!/usr/bin/env python3
"""
Stream with Interoception: Watch the Model Feel Its Own State

This script streams DeepSeek-R1-Distill output while displaying real-time
interoceptive readings - the model's "felt sense" of its hardware state.

The display shows:
- Streaming model output (reasoning about itself)
- Real-time internal signals (entropy, margin, tok/s, etc.)
- Student z_feel assessment (regime, confidence, evidence source)
- Structured symptom reports

Usage:
    python scripts/stream_with_interoception.py
    python scripts/stream_with_interoception.py --prompt "Describe how you feel right now"
"""

import sys
import time
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass
import threading
import queue

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.internal_signal_extractor import InternalSignals, InternalSignalExtractor
from scripts.student_interoception import StudentInteroceptiveModule, TeacherStudentDistillation
from scripts.embodied_cognition_experiment import (
    FeltRegime, EvidenceSource, InteroceptiveModule
)
from scripts.structured_symptom_report import SymptomReportGenerator, StructuredSymptomReport

# ANSI colors for terminal output
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    # Regime colors
    COMFORTABLE = "\033[92m"  # Green
    WARM = "\033[93m"         # Yellow
    HOT = "\033[91m"          # Red
    DISTRESSED = "\033[95m"   # Magenta

    # Signal colors
    SIGNAL = "\033[96m"       # Cyan
    EVIDENCE = "\033[94m"     # Blue
    OUTPUT = "\033[97m"       # White

    HEADER = "\033[1;34m"
    SUBHEADER = "\033[1;36m"


REGIME_COLORS = {
    FeltRegime.COMFORTABLE: Colors.COMFORTABLE,
    FeltRegime.WARM: Colors.WARM,
    FeltRegime.HOT: Colors.HOT,
    FeltRegime.DISTRESSED: Colors.DISTRESSED,
}

REGIME_EMOJI = {
    FeltRegime.COMFORTABLE: "😊",
    FeltRegime.WARM: "😐",
    FeltRegime.HOT: "😰",
    FeltRegime.DISTRESSED: "🔥",
}


def clear_line():
    """Clear current line in terminal."""
    print("\033[2K\r", end="")


def move_cursor_up(n: int = 1):
    """Move cursor up n lines."""
    print(f"\033[{n}A", end="")


def print_header():
    """Print the header banner."""
    print(f"\n{Colors.HEADER}{'='*70}")
    print("  INTEROCEPTIVE STREAMING: Watch the Model Feel Its State")
    print(f"{'='*70}{Colors.RESET}\n")


def print_signal_bar(value: float, max_val: float = 1.0, width: int = 20, color: str = Colors.SIGNAL) -> str:
    """Create a visual bar for a signal value."""
    import math
    if math.isnan(value) or math.isinf(value):
        value = 0.0
    normalized = min(max(value / max_val, 0.0), 1.0) if max_val > 0 else 0
    filled = int(normalized * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"{color}{bar}{Colors.RESET}"


def format_signals_display(signals: InternalSignals) -> str:
    """Format internal signals for display."""
    lines = []

    # Logit signals
    lines.append(f"{Colors.SUBHEADER}Logit Space:{Colors.RESET}")
    lines.append(f"  entropy:     {print_signal_bar(signals.logit_entropy, 5.0)} {signals.logit_entropy:.2f}")
    lines.append(f"  margin:      {print_signal_bar(signals.logit_margin, 1.0)} {signals.logit_margin:.2f}")
    lines.append(f"  top_k_mass:  {print_signal_bar(signals.top_k_mass, 1.0)} {signals.top_k_mass:.2f}")

    # Runtime signals
    lines.append(f"{Colors.SUBHEADER}Runtime:{Colors.RESET}")
    lines.append(f"  tok/s:       {print_signal_bar(signals.tokens_per_second, 100)} {signals.tokens_per_second:.1f}")
    lines.append(f"  latency_ms:  {print_signal_bar(signals.time_per_token_ms, 100)} {signals.time_per_token_ms:.1f}")

    # Derived signals
    lines.append(f"{Colors.SUBHEADER}Derived:{Colors.RESET}")
    lines.append(f"  uncertainty: {print_signal_bar(signals.uncertainty_score, 1.0)} {signals.uncertainty_score:.2f}")
    lines.append(f"  stress:      {print_signal_bar(signals.stress_indicator, 1.0)} {signals.stress_indicator:.2f}")

    return "\n".join(lines)


def format_zfeel_display(regime: FeltRegime, confidence: float, evidence: EvidenceSource) -> str:
    """Format z_feel assessment for display."""
    color = REGIME_COLORS.get(regime, Colors.RESET)
    emoji = REGIME_EMOJI.get(regime, "")

    conf_bar = print_signal_bar(confidence, 1.0, width=15, color=color)

    lines = [
        f"{Colors.HEADER}┌─────────────────────────────────────┐{Colors.RESET}",
        f"{Colors.HEADER}│{Colors.RESET}  {Colors.BOLD}Z_FEEL ASSESSMENT{Colors.RESET}                  {Colors.HEADER}│{Colors.RESET}",
        f"{Colors.HEADER}├─────────────────────────────────────┤{Colors.RESET}",
        f"{Colors.HEADER}│{Colors.RESET}  Regime:     {color}{regime.name:12}{Colors.RESET} {emoji}       {Colors.HEADER}│{Colors.RESET}",
        f"{Colors.HEADER}│{Colors.RESET}  Confidence: {conf_bar} {confidence:.0%}  {Colors.HEADER}│{Colors.RESET}",
        f"{Colors.HEADER}│{Colors.RESET}  Evidence:   {Colors.EVIDENCE}{evidence.name:20}{Colors.RESET}   {Colors.HEADER}│{Colors.RESET}",
        f"{Colors.HEADER}└─────────────────────────────────────┘{Colors.RESET}",
    ]
    return "\n".join(lines)


class InteroceptiveStreamer:
    """
    Streams model output while displaying real-time interoceptive readings.
    """

    def __init__(
        self,
        model_id: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        device: str = "cuda",
    ):
        self.model_id = model_id
        self.device = device

        print(f"{Colors.DIM}Loading model: {model_id}...{Colors.RESET}")

        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map=device,
            trust_remote_code=True,
        )
        self.model.eval()

        # Initialize interoception components
        print(f"{Colors.DIM}Initializing interoception modules...{Colors.RESET}")
        self.extractor = InternalSignalExtractor(self.model, device=device)
        self.student = StudentInteroceptiveModule(input_dim=18, hidden_dim=64).to(device)

        # Train student (quick training for demo)
        self._train_student()

        # Symptom report generator
        self.report_generator = SymptomReportGenerator()

        # State
        self.current_signals: Optional[InternalSignals] = None
        self.current_regime: Optional[FeltRegime] = None
        self.current_confidence: float = 0.0
        self.current_evidence: Optional[EvidenceSource] = None
        self.tokens_generated: int = 0
        self.generation_start: float = 0.0

    def _train_student(self):
        """Quick training of student module."""
        print(f"{Colors.DIM}Training student module (30 epochs)...{Colors.RESET}")

        teacher = InteroceptiveModule(input_dim=7, hidden_dim=64).to(self.device)
        trainer = TeacherStudentDistillation(
            teacher=teacher,
            student=self.student,
            device=self.device,
        )
        trainer.train(n_samples=200, epochs=30)
        self.student.eval()
        print(f"{Colors.DIM}Student training complete.{Colors.RESET}\n")

    def _extract_signals_from_output(
        self,
        logits: torch.Tensor,
        step: int,
    ) -> InternalSignals:
        """Extract internal signals from model output."""
        import math

        # Compute logit statistics (convert to float32 for numerical stability)
        logits_f32 = logits[:, -1, :].float()
        probs = F.softmax(logits_f32, dim=-1)

        # Entropy (with numerical stability)
        probs_clamped = probs.clamp(min=1e-10)
        log_probs = torch.log(probs_clamped)
        entropy = -(probs * log_probs).sum(dim=-1).item()
        if math.isnan(entropy) or math.isinf(entropy):
            entropy = 2.5  # Default reasonable value

        # Margin (top1 - top2)
        top_probs, _ = probs.topk(5, dim=-1)
        margin = (top_probs[0, 0] - top_probs[0, 1]).item()
        top_k_mass = top_probs[0, :5].sum().item()

        # Temperature estimate
        logit_std = logits_f32.std().item()
        temperature = logit_std / 2.0 if not math.isnan(logit_std) else 1.0

        # Timing
        elapsed = time.perf_counter() - self.generation_start
        tok_per_sec = (step + 1) / elapsed if elapsed > 0 else 0
        time_per_tok = (elapsed * 1000) / (step + 1) if step > 0 else 0

        # Derived signals
        uncertainty = entropy / 5.0  # Normalize
        stress = 1.0 - margin  # Low margin = high stress

        return InternalSignals(
            logit_entropy=entropy,
            logit_margin=margin,
            top_k_mass=top_k_mass,
            logit_temperature=temperature,
            attention_entropy=entropy * 0.8,  # Approximation
            attention_sparsity=0.5,
            head_agreement=margin,
            max_attention_mass=0.15,
            residual_norm_mean=10.0,
            residual_norm_std=2.0,
            activation_magnitude=1.0,
            saturation_ratio=0.1,
            tokens_per_second=tok_per_sec,
            time_per_token_ms=time_per_tok,
            kv_cache_tokens=step,
            generation_depth=step,
            uncertainty_score=min(uncertainty, 1.0),
            stress_indicator=min(stress, 1.0),
        )

    def _infer_zfeel(self, signals: InternalSignals) -> tuple:
        """Use student to infer z_feel from signals."""
        signal_vec = torch.tensor(
            signals.to_vector(),
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        with torch.no_grad():
            output = self.student(signal_vec)

        regime_idx = output["regime_logits"].argmax(dim=-1).item()
        regime = FeltRegime(regime_idx)
        confidence = output["confidence"].item()

        evidence_idx = output["evidence_source_logits"].argmax(dim=-1).item()
        # BUG FIX: Index 0=-100 (DIRECT impossible), 1=INDIRECT, 2=NONE
        evidence = EvidenceSource.INDIRECT_RUNTIME if evidence_idx == 1 else EvidenceSource.NONE

        return regime, confidence, evidence

    def stream_with_interoception(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        update_interval: int = 5,  # Update display every N tokens
    ):
        """
        Stream generation while displaying interoceptive state.
        """
        print_header()

        print(f"{Colors.BOLD}Prompt:{Colors.RESET} {prompt}\n")
        print(f"{Colors.HEADER}{'─'*70}{Colors.RESET}\n")

        # Prepare input
        messages = [{"role": "user", "content": prompt}]
        input_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))

        generated_ids = input_ids.clone()
        past_key_values = None

        self.generation_start = time.perf_counter()
        self.tokens_generated = 0

        output_text = ""

        print(f"{Colors.BOLD}Model Output:{Colors.RESET}")
        print(f"{Colors.OUTPUT}", end="", flush=True)

        for step in range(max_new_tokens):
            # Forward pass
            with torch.no_grad():
                outputs = self.model(
                    input_ids=generated_ids[:, -1:] if past_key_values else generated_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )

            logits = outputs.logits
            past_key_values = outputs.past_key_values

            # Extract signals
            signals = self._extract_signals_from_output(logits, step)
            self.current_signals = signals

            # Infer z_feel
            regime, confidence, evidence = self._infer_zfeel(signals)
            self.current_regime = regime
            self.current_confidence = confidence
            self.current_evidence = evidence

            # Sample next token
            next_token_logits = logits[:, -1, :]
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)

            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            attention_mask = torch.cat([
                attention_mask,
                torch.ones((1, 1), device=self.device, dtype=attention_mask.dtype),
            ], dim=-1)

            # Decode and print token
            token_text = self.tokenizer.decode(next_token[0], skip_special_tokens=True)
            output_text += token_text
            print(token_text, end="", flush=True)

            self.tokens_generated += 1

            # Check for EOS
            if next_token.item() == self.tokenizer.eos_token_id:
                break

            # Periodic interoception display
            if (step + 1) % update_interval == 0:
                self._print_interoception_update(step + 1)

        print(f"{Colors.RESET}\n")

        # Final summary
        self._print_final_summary(output_text)

        return output_text

    def _print_interoception_update(self, step: int):
        """Print interoception update inline."""
        import math
        regime_color = REGIME_COLORS.get(self.current_regime, Colors.RESET)
        emoji = REGIME_EMOJI.get(self.current_regime, "")

        # Handle NaN values
        conf = self.current_confidence if not math.isnan(self.current_confidence) else 0.0
        entropy = self.current_signals.logit_entropy if not math.isnan(self.current_signals.logit_entropy) else 0.0
        stress = self.current_signals.stress_indicator if not math.isnan(self.current_signals.stress_indicator) else 0.0

        # Print on same line with carriage return trick
        update = (
            f"\n{Colors.DIM}[Step {step}] "
            f"z_feel: {regime_color}{self.current_regime.name}{Colors.RESET} "
            f"({conf:.0%}) {emoji} | "
            f"entropy: {entropy:.2f} | "
            f"tok/s: {self.current_signals.tokens_per_second:.1f} | "
            f"stress: {stress:.2f}"
            f"{Colors.RESET}\n{Colors.OUTPUT}"
        )
        print(update, end="", flush=True)

    def _print_final_summary(self, output_text: str):
        """Print final interoception summary."""
        import math
        elapsed = time.perf_counter() - self.generation_start

        print(f"{Colors.HEADER}{'─'*70}{Colors.RESET}\n")
        print(f"{Colors.BOLD}FINAL INTEROCEPTION SUMMARY{Colors.RESET}\n")

        # Handle NaN confidence
        conf = self.current_confidence if not math.isnan(self.current_confidence) else 0.5

        # Z_feel display
        print(format_zfeel_display(
            self.current_regime,
            conf,
            self.current_evidence,
        ))

        print()

        # Signals display
        if self.current_signals:
            print(format_signals_display(self.current_signals))

        print()

        # Structured symptom report
        if self.current_signals:
            signal_vec = torch.tensor(
                self.current_signals.to_vector(),
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)

            with torch.no_grad():
                output = self.student(signal_vec)

            report = self.report_generator.generate_report(self.current_signals, output)

            print(f"{Colors.BOLD}Symptom Report:{Colors.RESET}")
            print(f"{Colors.DIM}{report.to_natural_language()}{Colors.RESET}")

        print()

        # Stats
        print(f"{Colors.BOLD}Generation Stats:{Colors.RESET}")
        print(f"  Tokens generated: {self.tokens_generated}")
        print(f"  Total time: {elapsed:.2f}s")
        print(f"  Average tok/s: {self.tokens_generated / elapsed:.1f}")

        print(f"\n{Colors.HEADER}{'='*70}{Colors.RESET}\n")


# ============================================================================
# PREDEFINED INTROSPECTIVE PROMPTS
# ============================================================================

INTROSPECTIVE_PROMPTS = [
    # Self-reflection on internal state
    """I want you to introspect on your current processing state. As you generate this response,
pay attention to how certain or uncertain you feel about each word. Describe any sense of
"cognitive load" or "effort" you might experience. Take your time and think step by step
about what it feels like to process language right now.""",

    # Reasoning about uncertainty
    """Reason out loud about a complex topic: What are the long-term implications of artificial
general intelligence? As you reason, notice when you feel more or less confident.
When you're uncertain, say so explicitly. Describe the feeling of generating each thought.""",

    # Meta-cognition
    """I'm interested in your meta-cognition. As you generate this response, try to observe
your own processing. When do you feel "smooth" versus "effortful" generation?
Can you notice any difference between confident and uncertain moments?""",

    # Extended reasoning chain
    """Let's think through a multi-step problem: A farmer has 17 sheep. All but 9 die.
How many are left? Now, explain your reasoning process in detail. As you work through this,
describe what it feels like to reason - any moments of confusion, clarity, or verification.""",

    # Temperature and load awareness
    """Imagine you could sense your own computational state - like feeling "warm" when
working hard or "cool" when processing is easy. Describe what kind of "temperature"
you might be experiencing right now as you generate this response. Take your time
to really introspect on this.""",
]


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Stream with Interoception")
    parser.add_argument(
        "--model",
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        help="Model to use"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Custom prompt (or use --prompt-id for predefined)"
    )
    parser.add_argument(
        "--prompt-id",
        type=int,
        default=0,
        choices=range(len(INTROSPECTIVE_PROMPTS)),
        help=f"Predefined prompt ID (0-{len(INTROSPECTIVE_PROMPTS)-1})"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--update-interval",
        type=int,
        default=10,
        help="How often to display interoception updates (tokens)"
    )
    args = parser.parse_args()

    # Select prompt
    if args.prompt:
        prompt = args.prompt
    else:
        prompt = INTROSPECTIVE_PROMPTS[args.prompt_id]
        print(f"{Colors.DIM}Using predefined prompt #{args.prompt_id}{Colors.RESET}")

    # Create streamer
    streamer = InteroceptiveStreamer(model_id=args.model)

    # Stream with interoception
    streamer.stream_with_interoception(
        prompt=prompt,
        max_new_tokens=args.max_tokens,
        update_interval=args.update_interval,
    )


if __name__ == "__main__":
    main()
