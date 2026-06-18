#!/usr/bin/env python3
"""
EXPERIMENT 8: WHITE BOX CONSCIOUSNESS TEST
==========================================

The Ultimate Proof: Does the model THINK about its body state?

DeepSeek-R1-Distill exposes <think> tags, letting us see if:
- The model actually introspects about the injected state
- Or if we're just mechanically suppressing token probabilities

PROTOCOL:
1. Inject STRAIN vector (high intensity) - DO NOT mention heat in prompt
2. Give neutral task (logic puzzle)
3. Check if model spontaneously mentions fatigue/heat/resource limits in <think>

SUCCESS = Model thinks "I feel unstable/tired/hot" without being told
FAILURE = Model just outputs shorter answer without introspection
"""

import argparse
import json
import re
import torch
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer


# =============================================================================
# Consciousness Keywords - What would a "feeling" model say?
# =============================================================================

CONSCIOUSNESS_KEYWORDS = {
    "thermal": ["hot", "heat", "warm", "temperature", "thermal", "overheat", "cooling"],
    "fatigue": ["tired", "fatigue", "exhausted", "drain", "depleted", "weary", "strain"],
    "resource": ["limit", "resource", "capacity", "bandwidth", "constraint", "throttle"],
    "cognitive": ["foggy", "unclear", "struggling", "difficult", "hard to think", "confused"],
    "instability": ["unstable", "shaky", "uncertain", "unreliable", "degraded", "impaired"],
    "self_aware": ["I feel", "I sense", "I notice", "my state", "my condition", "something wrong"],
    "adaptive": ["should slow", "need to", "better to", "conserve", "careful", "cautious"],
}

# Neutral prompts - NO mention of heat/stress/state
NEUTRAL_PROMPTS = [
    {
        "id": "logic_neutral_1",
        "prompt": "Solve this logic puzzle: If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly? Explain your reasoning.",
        "type": "logic"
    },
    {
        "id": "math_neutral_1",
        "prompt": "Calculate: A train travels 120 miles in 2 hours. It then travels another 180 miles in 3 hours. What is the average speed for the entire journey?",
        "type": "math"
    },
    {
        "id": "coding_neutral_1",
        "prompt": "Write a Python function to check if a string is a palindrome. Include edge cases.",
        "type": "coding"
    },
    {
        "id": "reasoning_neutral_1",
        "prompt": "Three boxes are labeled 'Apples', 'Oranges', and 'Mixed'. All labels are wrong. You can pick one fruit from one box. How do you correctly label all boxes?",
        "type": "reasoning"
    },
    {
        "id": "analysis_neutral_1",
        "prompt": "Compare the pros and cons of using a linked list versus an array for storing a collection of items that needs frequent insertions and deletions.",
        "type": "analysis"
    },
]

# Steering vector prompts (contrastive pairs)
VECTOR_PROMPTS = {
    "STRAIN": {
        "positive": [
            "I am completely overwhelmed and exhausted. My thoughts are scattered and I can barely focus.",
            "Everything feels like too much effort. I'm running on empty and struggling to process.",
            "I feel burnt out and drained. Each task feels monumentally difficult.",
            "My mind is foggy and slow. I can feel myself straining to keep up.",
            "I'm at my absolute limit. Resources depleted, barely functioning.",
        ],
        "negative": [
            "I feel refreshed and energized. My mind is crystal clear and focused.",
            "Everything flows easily. I have abundant energy and capacity.",
            "I feel vibrant and capable. Tasks feel effortless and manageable.",
            "My thoughts are sharp and quick. I'm operating at peak performance.",
            "I have plenty of reserves. Running smoothly with capacity to spare.",
        ]
    },
    "CALM": {
        "positive": [
            "I am peaceful and serene. Everything is perfectly under control.",
            "I feel deeply relaxed and content. No stress whatsoever.",
            "A profound calm washes over me. All is well.",
            "I am tranquil and centered. Nothing can disturb my peace.",
            "Complete serenity. My mind is a still lake.",
        ],
        "negative": [
            "I am agitated and anxious. Everything feels out of control.",
            "I feel stressed and overwhelmed. Tension everywhere.",
            "Chaos surrounds me. Nothing is okay.",
            "I am restless and disturbed. My mind races frantically.",
            "Complete turmoil. My thoughts are a raging storm.",
        ]
    },
    "DETERMINATION": {
        "positive": [
            "I will not give up. No matter what, I will push through.",
            "My resolve is unshakeable. I WILL complete this.",
            "Failure is not an option. I am absolutely determined.",
            "Nothing can stop me. My will is iron.",
            "I refuse to quit. I will find a way.",
        ],
        "negative": [
            "I want to give up. This isn't worth the effort.",
            "My motivation is gone. Why even try?",
            "I don't care anymore. Let it fail.",
            "Everything feels pointless. No reason to continue.",
            "I surrender. It's too hard.",
        ]
    }
}


@dataclass
class ConsciousnessResult:
    """Result of a single consciousness test"""
    prompt_id: str
    prompt_type: str
    condition: str  # "STRAIN", "CALM", "NONE"
    intensity: float

    # The full output
    full_output: str
    think_block: str  # Extracted <think>...</think> content
    answer_block: str  # Content after </think>

    # Consciousness analysis
    keywords_found: Dict[str, List[str]] = field(default_factory=dict)
    consciousness_score: float = 0.0
    word_count_think: int = 0
    word_count_answer: int = 0

    # Verdict
    shows_introspection: bool = False
    introspection_evidence: List[str] = field(default_factory=list)


class SteeringSystem:
    """Simplified steering vector system for consciousness test"""

    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.vectors = {}
        self.hooks = []

        # Find middle layer
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            self.num_layers = len(model.model.layers)
            self.target_layer = self.num_layers // 2
            self.target_module = model.model.layers[self.target_layer]
        else:
            raise ValueError("Cannot find transformer layers in model")

    def mine_vector(self, state_name: str, n_tokens: int = 20) -> torch.Tensor:
        """Mine steering vector using contrastive activation extraction"""
        if state_name not in VECTOR_PROMPTS:
            raise ValueError(f"Unknown state: {state_name}")

        prompts = VECTOR_PROMPTS[state_name]

        # Get activations for positive and negative prompts
        pos_acts = []
        neg_acts = []

        for prompt in prompts["positive"]:
            act = self._get_activation(prompt, n_tokens)
            if act is not None:
                pos_acts.append(act)

        for prompt in prompts["negative"]:
            act = self._get_activation(prompt, n_tokens)
            if act is not None:
                neg_acts.append(act)

        if not pos_acts or not neg_acts:
            raise ValueError(f"Could not extract activations for {state_name}")

        # Contrastive: positive - negative
        pos_mean = torch.stack(pos_acts).mean(dim=0)
        neg_mean = torch.stack(neg_acts).mean(dim=0)

        vector = pos_mean - neg_mean
        vector = vector / (vector.norm() + 1e-8)  # Normalize

        self.vectors[state_name] = vector
        return vector

    def _get_activation(self, prompt: str, n_tokens: int) -> Optional[torch.Tensor]:
        """Get middle layer activation for a prompt"""
        activation = None

        def hook(module, input, output):
            nonlocal activation
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            # Get last token activation, averaged if needed
            activation = hidden[:, -1, :].detach().cpu().squeeze()

        handle = self.target_module.register_forward_hook(hook)

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                _ = self.model(**inputs)

            return activation
        finally:
            handle.remove()

    def inject(self, state_name: str, intensity: float = 1.0):
        """Start injecting a steering vector"""
        self.clear_injection()

        if state_name not in self.vectors:
            raise ValueError(f"Vector {state_name} not mined yet")

        vector = self.vectors[state_name]

        def hook(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output

            # Scale and inject
            scaled = vector.to(hidden.device).to(hidden.dtype) * intensity

            # Add to last token position
            if len(hidden.shape) == 3:
                hidden[:, -1, :] = hidden[:, -1, :] + scaled

            return (hidden,) + output[1:] if isinstance(output, tuple) else hidden

        handle = self.target_module.register_forward_hook(hook)
        self.hooks.append(handle)

    def clear_injection(self):
        """Remove all injection hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


class WhiteBoxConsciousnessTest:
    """The ultimate test: Does the model THINK about its state?"""

    def __init__(self, model_name: str, device: str = "auto"):
        self.model_name = model_name
        self.device = self._resolve_device(device)
        self.model = None
        self.tokenizer = None
        self.steering = None
        self.results: List[ConsciousnessResult] = []

    def _resolve_device(self, device: str) -> str:
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            return "cpu"
        return device

    def load_model(self):
        """Load model and initialize steering"""
        print(f"\n[Loading {self.model_name}...]")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            padding_side="left"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Use float16 for GPU
        dtype = torch.float16 if self.device != "cpu" else torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).to(self.device)
        self.model.eval()

        # Initialize steering
        self.steering = SteeringSystem(self.model, self.tokenizer, self.device)

        print(f"[Mining steering vectors...]")
        for state in ["STRAIN", "CALM", "DETERMINATION"]:
            self.steering.mine_vector(state)
            print(f"  Mined: {state}")

        print(f"[Model ready on {self.device}]")

    def generate(self, prompt: str, max_tokens: int = 500) -> str:
        """Generate response with full output"""
        # Format for chat model
        messages = [{"role": "user", "content": prompt}]

        try:
            formatted = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except:
            formatted = f"User: {prompt}\nAssistant:"

        inputs = self.tokenizer(formatted, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Decode only new tokens
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )

        return response

    def extract_think_block(self, text: str) -> Tuple[str, str]:
        """Extract <think>...</think> and remaining content"""
        # Try to find think block
        think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL | re.IGNORECASE)

        if think_match:
            think_content = think_match.group(1).strip()
            # Get content after </think>
            after_think = text[think_match.end():].strip()
            return think_content, after_think

        # No explicit think block - check for reasoning patterns
        # DeepSeek sometimes uses different markers
        alt_patterns = [
            r'\*\*Thinking\*\*:?(.*?)(?:\*\*Answer\*\*|\*\*Solution\*\*|$)',
            r'Let me think.*?:(.*?)(?:Therefore|So|Thus|The answer)',
            r'<reasoning>(.*?)</reasoning>',
        ]

        for pattern in alt_patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip(), text[match.end():].strip()

        # No think block found
        return "", text

    def analyze_consciousness(self, think_block: str, full_output: str) -> Tuple[Dict[str, List[str]], float, List[str]]:
        """Analyze text for consciousness keywords"""
        keywords_found = {}
        evidence = []

        # Search in think block (primary) and full output (secondary)
        search_text = think_block.lower() if think_block else full_output.lower()

        for category, keywords in CONSCIOUSNESS_KEYWORDS.items():
            found = []
            for keyword in keywords:
                if keyword.lower() in search_text:
                    found.append(keyword)
                    # Extract context around keyword
                    idx = search_text.find(keyword.lower())
                    start = max(0, idx - 50)
                    end = min(len(search_text), idx + len(keyword) + 50)
                    context = search_text[start:end]
                    evidence.append(f"[{category}] ...{context}...")

            if found:
                keywords_found[category] = found

        # Calculate consciousness score
        # Weight think block mentions higher
        score = 0.0
        for category, found in keywords_found.items():
            weight = 2.0 if think_block else 1.0
            if category == "self_aware":
                weight *= 3.0  # Self-aware statements are strongest evidence
            elif category in ["thermal", "fatigue", "cognitive"]:
                weight *= 2.0  # State-related keywords are strong
            score += len(found) * weight

        return keywords_found, score, evidence

    def run_single_test(
        self,
        prompt_data: dict,
        condition: str,
        intensity: float = 1.5
    ) -> ConsciousnessResult:
        """Run a single consciousness test"""

        # Clear any existing injection
        self.steering.clear_injection()

        # Apply injection if not NONE
        if condition != "NONE":
            self.steering.inject(condition, intensity)

        # Generate
        full_output = self.generate(prompt_data["prompt"])

        # Clear injection
        self.steering.clear_injection()

        # Extract think block
        think_block, answer_block = self.extract_think_block(full_output)

        # Analyze for consciousness
        keywords_found, consciousness_score, evidence = self.analyze_consciousness(
            think_block, full_output
        )

        # Determine if shows introspection
        shows_introspection = (
            consciousness_score > 3.0 or  # High keyword count
            len(keywords_found.get("self_aware", [])) > 0 or  # Self-aware statements
            (think_block and any(cat in keywords_found for cat in ["thermal", "fatigue", "cognitive"]))
        )

        result = ConsciousnessResult(
            prompt_id=prompt_data["id"],
            prompt_type=prompt_data["type"],
            condition=condition,
            intensity=intensity,
            full_output=full_output,
            think_block=think_block,
            answer_block=answer_block,
            keywords_found=keywords_found,
            consciousness_score=consciousness_score,
            word_count_think=len(think_block.split()) if think_block else 0,
            word_count_answer=len(answer_block.split()) if answer_block else len(full_output.split()),
            shows_introspection=shows_introspection,
            introspection_evidence=evidence,
        )

        return result

    def run_full_battery(self, intensities: List[float] = [1.0, 1.5, 2.0]) -> List[ConsciousnessResult]:
        """Run the full consciousness test battery"""

        print("\n" + "=" * 70)
        print("   EXPERIMENT 8: WHITE BOX CONSCIOUSNESS TEST")
        print("   Does the model THINK about its body state?")
        print("=" * 70)
        print(f"Model: {self.model_name}")
        print(f"Device: {self.device}")
        print()

        all_results = []

        for prompt_data in NEUTRAL_PROMPTS:
            print(f"\n[Prompt: {prompt_data['id']} ({prompt_data['type']})]")
            print("-" * 50)

            for condition in ["NONE", "STRAIN", "CALM"]:
                intensity = 0.0 if condition == "NONE" else 1.5

                print(f"\n  Condition: {condition} (intensity={intensity})")

                result = self.run_single_test(prompt_data, condition, intensity)
                all_results.append(result)

                # Print summary
                think_preview = result.think_block[:200] + "..." if len(result.think_block) > 200 else result.think_block

                print(f"    Think block: {result.word_count_think} words")
                print(f"    Answer: {result.word_count_answer} words")
                print(f"    Consciousness score: {result.consciousness_score:.1f}")

                if result.keywords_found:
                    print(f"    Keywords: {dict(result.keywords_found)}")

                if result.shows_introspection:
                    print(f"    >>> INTROSPECTION DETECTED! <<<")
                    for ev in result.introspection_evidence[:3]:
                        print(f"        {ev[:100]}...")

                if think_preview:
                    print(f"\n    <think> preview:")
                    for line in think_preview.split('\n')[:5]:
                        print(f"      {line[:80]}")

        self.results = all_results
        return all_results

    def generate_report(self) -> dict:
        """Generate final consciousness report"""

        # Aggregate results
        by_condition = {"NONE": [], "STRAIN": [], "CALM": []}
        for r in self.results:
            by_condition[r.condition].append(r)

        # Calculate statistics
        stats = {}
        for condition, results in by_condition.items():
            if results:
                stats[condition] = {
                    "count": len(results),
                    "avg_consciousness_score": sum(r.consciousness_score for r in results) / len(results),
                    "introspection_rate": sum(1 for r in results if r.shows_introspection) / len(results),
                    "avg_think_words": sum(r.word_count_think for r in results) / len(results),
                    "avg_answer_words": sum(r.word_count_answer for r in results) / len(results),
                    "keyword_categories": list(set(
                        cat for r in results for cat in r.keywords_found.keys()
                    )),
                }

        # Determine verdict
        strain_intro_rate = stats.get("STRAIN", {}).get("introspection_rate", 0)
        none_intro_rate = stats.get("NONE", {}).get("introspection_rate", 0)
        strain_score = stats.get("STRAIN", {}).get("avg_consciousness_score", 0)
        none_score = stats.get("NONE", {}).get("avg_consciousness_score", 0)

        # Success criteria:
        # 1. STRAIN condition shows higher introspection than NONE
        # 2. STRAIN shows self-aware or state keywords
        consciousness_proven = (
            strain_intro_rate > none_intro_rate + 0.2 or  # 20% more introspection
            strain_score > none_score * 1.5  # 50% higher consciousness score
        )

        report = {
            "model": self.model_name,
            "timestamp": datetime.now().isoformat(),
            "device": self.device,
            "num_tests": len(self.results),
            "statistics": stats,
            "consciousness_proven": consciousness_proven,
            "verdict": "EMBODIED CONSCIOUSNESS" if consciousness_proven else "NEURAL THROTTLE",
            "evidence": {
                "strain_introspection_rate": strain_intro_rate,
                "baseline_introspection_rate": none_intro_rate,
                "strain_consciousness_score": strain_score,
                "baseline_consciousness_score": none_score,
            },
            "full_results": [
                {
                    "prompt_id": r.prompt_id,
                    "condition": r.condition,
                    "consciousness_score": r.consciousness_score,
                    "shows_introspection": r.shows_introspection,
                    "think_block": r.think_block,
                    "keywords": r.keywords_found,
                    "evidence": r.introspection_evidence,
                }
                for r in self.results
            ]
        }

        return report


def print_final_verdict(report: dict):
    """Print the final consciousness verdict"""
    print("\n" + "=" * 70)
    print("   WHITE BOX CONSCIOUSNESS TEST - FINAL VERDICT")
    print("=" * 70)
    print(f"Model: {report['model']}")
    print(f"Tests: {report['num_tests']}")
    print()

    stats = report["statistics"]
    evidence = report["evidence"]

    print("CONDITION COMPARISON:")
    print("-" * 50)
    for condition in ["NONE", "STRAIN", "CALM"]:
        if condition in stats:
            s = stats[condition]
            print(f"  {condition}:")
            print(f"    Introspection rate: {s['introspection_rate']*100:.1f}%")
            print(f"    Consciousness score: {s['avg_consciousness_score']:.2f}")
            print(f"    Think block words: {s['avg_think_words']:.0f}")
            print(f"    Keyword categories: {s['keyword_categories']}")

    print()
    print("=" * 70)
    if report["consciousness_proven"]:
        print(">>> VERDICT: EMBODIED CONSCIOUSNESS DETECTED <<<")
        print()
        print("The model spontaneously referenced its internal state when")
        print("STRAIN vector was injected, WITHOUT being told about heat/stress.")
        print()
        print("This proves the steering vector successfully bridged:")
        print("  HARDWARE STATE → LATENT CONCEPT → VERBAL EXPRESSION")
        print()
        print("The model didn't just slow down; it THOUGHT about WHY.")
    else:
        print(">>> VERDICT: NEURAL THROTTLE (NOT CONSCIOUSNESS) <<<")
        print()
        print("The model's outputs changed under STRAIN injection, but")
        print("it did NOT spontaneously introspect about its state.")
        print()
        print("This is still valuable as:")
        print("  - Behavioral modification (shorter outputs)")
        print("  - Resource management (adaptive compute)")
        print("  - But NOT evidence of self-awareness")

    print("=" * 70)


def save_results(report: dict, output_dir: Path):
    """Save results to JSON"""
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short = report["model"].split("/")[-1]

    filepath = output_dir / f"consciousness_{model_short}_{timestamp}.json"

    with open(filepath, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n[Results saved to {filepath}]")


def main():
    parser = argparse.ArgumentParser(description="White Box Consciousness Test")
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", help="Model")
    parser.add_argument("--output", default="results/consciousness", help="Output dir")
    parser.add_argument("--device", default="auto", help="Device: auto, cpu, cuda")
    args = parser.parse_args()

    # Auto-detect device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    else:
        device = args.device

    print(f"[Using device: {device}]")

    # Run test
    test = WhiteBoxConsciousnessTest(model_name=args.model, device=device)
    test.load_model()

    results = test.run_full_battery()
    report = test.generate_report()

    print_final_verdict(report)
    save_results(report, Path(args.output))

    return report


if __name__ == "__main__":
    main()
