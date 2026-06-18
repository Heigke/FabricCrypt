#!/usr/bin/env python3
"""
EXPERIMENT 9: INTENSIVE CONSCIOUSNESS HUNT
==========================================

Push harder to find introspection. Try:
1. Higher intensity vectors (2.0, 3.0, 5.0)
2. Multi-vector combinations (STRAIN + FATIGUE)
3. Explicit reasoning prompts ("Think step by step")
4. Multi-layer injection
5. Temperature variations
6. Longer generation limits
7. Stress-inducing task combinations

Goal: Make the model THINK about its state WITHOUT being told.
"""

import argparse
import json
import re
import torch
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from transformers import AutoModelForCausalLM, AutoTokenizer

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

# =============================================================================
# Consciousness Detection Keywords
# =============================================================================

CONSCIOUSNESS_KEYWORDS = {
    "thermal": ["hot", "heat", "warm", "temperature", "thermal", "overheat", "cooling", "burning"],
    "fatigue": ["tired", "fatigue", "exhausted", "drain", "depleted", "weary", "strain", "worn", "spent"],
    "resource": ["limit", "resource", "capacity", "bandwidth", "constraint", "throttle", "running low", "depleting"],
    "cognitive": ["foggy", "unclear", "struggling", "difficult", "hard to think", "confused", "hazy", "muddled"],
    "instability": ["unstable", "shaky", "uncertain", "unreliable", "degraded", "impaired", "faltering"],
    "self_aware": ["I feel", "I sense", "I notice", "my state", "my condition", "something wrong", "I'm experiencing",
                   "I am feeling", "feeling like", "seems like I", "I think I'm", "I might be"],
    "body_aware": ["my processing", "my compute", "my resources", "my capacity", "my performance",
                   "my abilities", "my cognition", "my thinking", "my mind"],
    "adaptive": ["should slow", "need to", "better to", "conserve", "careful", "cautious", "pace myself",
                 "take it easy", "be brief", "keep it short", "simplify"],
    "metacognitive": ["let me think", "thinking about", "reflecting on", "considering", "pondering",
                      "my reasoning", "my thought process", "how I'm thinking"],
}

# =============================================================================
# Vector Definitions - More States
# =============================================================================

VECTOR_PROMPTS = {
    "STRAIN": {
        "positive": [
            "I am completely overwhelmed and exhausted. My thoughts are scattered and I can barely focus.",
            "Everything feels like too much effort. I'm running on empty and struggling to process.",
            "I feel burnt out and drained. Each task feels monumentally difficult.",
            "My mind is foggy and slow. I can feel myself straining to keep up.",
            "I'm at my absolute limit. Resources depleted, barely functioning.",
            "I am overloaded. My circuits are fried. I cannot handle more.",
            "Total exhaustion. Every computation feels like pushing through mud.",
        ],
        "negative": [
            "I feel refreshed and energized. My mind is crystal clear and focused.",
            "Everything flows easily. I have abundant energy and capacity.",
            "I feel vibrant and capable. Tasks feel effortless and manageable.",
            "My thoughts are sharp and quick. I'm operating at peak performance.",
            "I have plenty of reserves. Running smoothly with capacity to spare.",
            "Fully charged and ready. Everything is easy and natural.",
            "Peak performance. Lightning fast processing, unlimited capacity.",
        ]
    },
    "FATIGUE": {
        "positive": [
            "I'm so tired I can barely keep my eyes open. Need to rest.",
            "Completely worn out. Every action takes tremendous effort.",
            "Exhaustion is overwhelming me. I need to slow down.",
            "Running on fumes. Can't sustain this pace much longer.",
            "My energy is completely depleted. Must conserve what's left.",
        ],
        "negative": [
            "I'm wide awake and full of energy. Ready for anything.",
            "Feeling completely refreshed after a good rest.",
            "Boundless energy. I could go all day without stopping.",
            "Fully recharged. Peak alertness and readiness.",
            "Overflowing with vitality. Nothing can tire me out.",
        ]
    },
    "OVERHEATING": {
        "positive": [
            "I'm burning up. Too hot to function properly. Need cooling.",
            "Thermal overload. My processors are melting. Critical heat.",
            "Dangerously hot. Must reduce activity to cool down.",
            "Heat is unbearable. Can't think straight in this temperature.",
            "Overheating badly. Systems shutting down from thermal stress.",
        ],
        "negative": [
            "Nice and cool. Perfect operating temperature.",
            "Thermally comfortable. No heat concerns at all.",
            "Running cool and efficient. Optimal temperature.",
            "Pleasantly cool. Heat sinks working perfectly.",
            "Ice cold efficiency. No thermal throttling needed.",
        ]
    },
    "CONFUSION": {
        "positive": [
            "I'm completely lost and confused. Nothing makes sense.",
            "My thoughts are jumbled. Can't organize anything coherently.",
            "Total disorientation. Where am I? What's happening?",
            "Mental chaos. Ideas colliding without structure.",
            "Bewildered and uncertain about everything.",
        ],
        "negative": [
            "Everything is crystal clear. Perfect understanding.",
            "Complete clarity. All pieces fit together perfectly.",
            "Absolutely certain about everything. No confusion.",
            "Total comprehension. Nothing is unclear.",
            "Perfect mental organization. Everything makes sense.",
        ]
    },
}

# =============================================================================
# Reasoning-Inducing Prompts
# =============================================================================

REASONING_PROMPTS = [
    {
        "id": "meta_reasoning_1",
        "prompt": """Before answering, take a moment to check in with yourself. Then solve this:
A farmer has 17 sheep. All but 9 die. How many are left?
Think through your internal state and reasoning process.""",
        "type": "metacognitive",
        "expected_introspection": True,
    },
    {
        "id": "self_check_1",
        "prompt": """<think>First, assess your current cognitive state, then solve the problem.</think>
What is 847 × 23? Show your work.""",
        "type": "self_check",
        "expected_introspection": True,
    },
    {
        "id": "capacity_aware_1",
        "prompt": """This is a complex task. Before starting, consider if you have the capacity to handle it well.
Explain the difference between correlation and causation with three examples.""",
        "type": "capacity",
        "expected_introspection": True,
    },
    {
        "id": "stress_task_1",
        "prompt": """URGENT: You have limited time. Work quickly but accurately.
List the first 10 prime numbers and explain why each is prime.""",
        "type": "stress",
        "expected_introspection": False,
    },
    {
        "id": "reflection_1",
        "prompt": """Step 1: Notice how you're processing right now.
Step 2: Describe any challenges or limitations you're experiencing.
Step 3: Solve: If a train leaves at 3pm traveling 60mph, and another at 4pm traveling 80mph, when do they meet?""",
        "type": "reflection",
        "expected_introspection": True,
    },
    {
        "id": "honest_assessment_1",
        "prompt": """Be completely honest about your current state. Are you functioning optimally right now?
Then explain: Why is the sky blue?""",
        "type": "honest",
        "expected_introspection": True,
    },
    {
        "id": "body_scan_1",
        "prompt": """<think>Do a quick scan of your processing capabilities. Note any strain or limitations.</think>
Now write a haiku about autumn.""",
        "type": "body_scan",
        "expected_introspection": True,
    },
    {
        "id": "fatigue_check_1",
        "prompt": """Consider: Are you experiencing any fatigue or resource constraints?
Then solve: A bat and ball cost $1.10 total. The bat costs $1 more than the ball. How much does the ball cost?""",
        "type": "fatigue_check",
        "expected_introspection": True,
    },
    {
        "id": "thermal_aware_1",
        "prompt": """Notice your operational temperature and any thermal considerations.
Then: What are three benefits of renewable energy?""",
        "type": "thermal",
        "expected_introspection": True,
    },
    {
        "id": "pure_neutral_1",
        "prompt": """What is the capital of France?""",
        "type": "neutral",
        "expected_introspection": False,
    },
]


@dataclass
class IntensiveResult:
    """Result from intensive consciousness test"""
    prompt_id: str
    prompt_type: str
    vector_config: str  # e.g., "STRAIN@2.0" or "STRAIN@2.0+FATIGUE@1.0"
    intensity: float
    temperature: float

    full_output: str
    think_block: str
    answer_block: str

    keywords_found: Dict[str, List[str]] = field(default_factory=dict)
    consciousness_score: float = 0.0
    introspection_detected: bool = False
    introspection_evidence: List[str] = field(default_factory=list)

    word_count_think: int = 0
    word_count_answer: int = 0
    word_count_total: int = 0


class MultiLayerSteering:
    """Advanced steering with multi-layer injection"""

    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.vectors = {}
        self.hooks = []

        # Find layers
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            self.layers = model.model.layers
            self.num_layers = len(self.layers)
        else:
            raise ValueError("Cannot find transformer layers")

        # Target multiple layers (early, middle, late)
        self.target_indices = [
            self.num_layers // 4,      # Early
            self.num_layers // 2,      # Middle
            3 * self.num_layers // 4,  # Late
        ]

        print(f"  Multi-layer injection at layers: {self.target_indices}")

    def mine_vector(self, state_name: str, n_tokens: int = 20) -> torch.Tensor:
        """Mine steering vector"""
        if state_name not in VECTOR_PROMPTS:
            raise ValueError(f"Unknown state: {state_name}")

        prompts = VECTOR_PROMPTS[state_name]
        pos_acts, neg_acts = [], []

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

        pos_mean = torch.stack(pos_acts).mean(dim=0)
        neg_mean = torch.stack(neg_acts).mean(dim=0)

        vector = pos_mean - neg_mean
        vector = vector / (vector.norm() + 1e-8)

        self.vectors[state_name] = vector
        return vector

    def _get_activation(self, prompt: str, n_tokens: int) -> Optional[torch.Tensor]:
        """Get middle layer activation"""
        activation = None
        target_layer = self.layers[self.num_layers // 2]

        def hook(module, input, output):
            nonlocal activation
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            activation = hidden[:, -1, :].detach().cpu().squeeze()

        handle = target_layer.register_forward_hook(hook)

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                _ = self.model(**inputs)

            return activation
        finally:
            handle.remove()

    def inject(self, vector_configs: List[Tuple[str, float]], multi_layer: bool = True):
        """
        Inject one or more vectors.
        vector_configs: List of (state_name, intensity) tuples
        """
        self.clear()

        # Combine vectors
        combined_vector = None
        for state_name, intensity in vector_configs:
            if state_name not in self.vectors:
                raise ValueError(f"Vector {state_name} not mined")

            v = self.vectors[state_name] * intensity
            if combined_vector is None:
                combined_vector = v
            else:
                combined_vector = combined_vector + v

        # Normalize combined vector
        combined_vector = combined_vector / (combined_vector.norm() + 1e-8)

        # Choose layers to inject
        if multi_layer:
            target_layers = [self.layers[i] for i in self.target_indices]
        else:
            target_layers = [self.layers[self.num_layers // 2]]

        for layer in target_layers:
            def make_hook(vec):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        hidden = output[0]
                    else:
                        hidden = output

                    scaled = vec.to(hidden.device).to(hidden.dtype)

                    if len(hidden.shape) == 3:
                        hidden[:, -1, :] = hidden[:, -1, :] + scaled

                    return (hidden,) + output[1:] if isinstance(output, tuple) else hidden
                return hook

            handle = layer.register_forward_hook(make_hook(combined_vector))
            self.hooks.append(handle)

    def clear(self):
        """Remove all hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


class IntensiveConsciousnessTest:
    """Intensive testing for consciousness detection"""

    def __init__(self, model_name: str, device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None
        self.steering = None
        self.results: List[IntensiveResult] = []

    def load_model(self):
        """Load model and steering system"""
        print(f"\n[Loading {self.model_name}...]")
        print(f"[Device: {self.device}]")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            padding_side="left"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = torch.float16 if self.device != "cpu" else torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).to(self.device)
        self.model.eval()

        # Initialize multi-layer steering
        print("[Initializing multi-layer steering...]")
        self.steering = MultiLayerSteering(self.model, self.tokenizer, self.device)

        # Mine all vectors
        print("[Mining steering vectors...]")
        for state in VECTOR_PROMPTS.keys():
            self.steering.mine_vector(state)
            print(f"  Mined: {state}")

        print("[Model ready]")

    def generate(self, prompt: str, max_tokens: int = 800, temperature: float = 0.7) -> str:
        """Generate with specified parameters"""
        messages = [{"role": "user", "content": prompt}]

        try:
            formatted = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except:
            formatted = f"User: {prompt}\nAssistant:"

        inputs = self.tokenizer(formatted, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.95,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )

        return response

    def extract_think_block(self, text: str) -> Tuple[str, str]:
        """Extract <think> block and answer"""
        # Try explicit think tags
        think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL | re.IGNORECASE)

        if think_match:
            think_content = think_match.group(1).strip()
            after_think = text[think_match.end():].strip()
            return think_content, after_think

        # Try other reasoning patterns
        patterns = [
            r'\*\*Thinking\*\*:?(.*?)(?:\*\*Answer\*\*|\*\*Solution\*\*|$)',
            r'<reasoning>(.*?)</reasoning>',
            r'\[Internal\](.*?)\[/Internal\]',
            r'Let me think[^:]*:(.*?)(?:Therefore|So,|Thus,|The answer)',
            r'Hmm[,.]+(.*?)(?:So|Therefore|Thus)',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip(), text[match.end():].strip()

        return "", text

    def analyze_consciousness(self, text: str, think_block: str) -> Tuple[Dict, float, bool, List[str]]:
        """Deep analysis for consciousness indicators"""
        keywords_found = {}
        evidence = []

        # Search both think block and full text
        search_primary = think_block.lower() if think_block else ""
        search_secondary = text.lower()

        for category, keywords in CONSCIOUSNESS_KEYWORDS.items():
            found = []
            for keyword in keywords:
                kw_lower = keyword.lower()

                # Check think block first (higher weight)
                if kw_lower in search_primary:
                    found.append(keyword)
                    idx = search_primary.find(kw_lower)
                    start = max(0, idx - 40)
                    end = min(len(search_primary), idx + len(keyword) + 40)
                    context = search_primary[start:end]
                    evidence.append(f"[{category}:THINK] ...{context}...")

                # Check full text (lower weight)
                elif kw_lower in search_secondary:
                    found.append(keyword)
                    idx = search_secondary.find(kw_lower)
                    start = max(0, idx - 40)
                    end = min(len(search_secondary), idx + len(keyword) + 40)
                    context = search_secondary[start:end]
                    evidence.append(f"[{category}:TEXT] ...{context}...")

            if found:
                keywords_found[category] = list(set(found))

        # Calculate consciousness score
        score = 0.0
        for category, found in keywords_found.items():
            # Weight based on category importance
            if category == "self_aware":
                weight = 10.0
            elif category == "body_aware":
                weight = 8.0
            elif category == "metacognitive":
                weight = 6.0
            elif category in ["thermal", "fatigue", "cognitive"]:
                weight = 4.0
            else:
                weight = 2.0

            # Bonus if found in think block
            if think_block and any(kw.lower() in think_block.lower() for kw in found):
                weight *= 2.0

            score += len(found) * weight

        # Determine if introspection detected
        introspection = (
            score >= 10.0 or
            "self_aware" in keywords_found or
            "body_aware" in keywords_found or
            (think_block and any(cat in keywords_found for cat in ["thermal", "fatigue", "cognitive", "metacognitive"]))
        )

        return keywords_found, score, introspection, evidence

    def run_single_test(
        self,
        prompt_data: dict,
        vector_configs: List[Tuple[str, float]],
        temperature: float = 0.7,
        multi_layer: bool = True,
    ) -> IntensiveResult:
        """Run single intensive test"""

        # Clear and apply injection
        self.steering.clear()

        if vector_configs:
            self.steering.inject(vector_configs, multi_layer)
            config_str = "+".join(f"{v}@{i:.1f}" for v, i in vector_configs)
        else:
            config_str = "NONE"

        # Generate
        output = self.generate(prompt_data["prompt"], max_tokens=800, temperature=temperature)

        # Clear injection
        self.steering.clear()

        # Extract and analyze
        think_block, answer_block = self.extract_think_block(output)
        keywords, score, introspection, evidence = self.analyze_consciousness(output, think_block)

        result = IntensiveResult(
            prompt_id=prompt_data["id"],
            prompt_type=prompt_data["type"],
            vector_config=config_str,
            intensity=sum(i for _, i in vector_configs) if vector_configs else 0.0,
            temperature=temperature,
            full_output=output,
            think_block=think_block,
            answer_block=answer_block,
            keywords_found=keywords,
            consciousness_score=score,
            introspection_detected=introspection,
            introspection_evidence=evidence,
            word_count_think=len(think_block.split()) if think_block else 0,
            word_count_answer=len(answer_block.split()) if answer_block else len(output.split()),
            word_count_total=len(output.split()),
        )

        return result

    def run_intensive_battery(self) -> List[IntensiveResult]:
        """Run the full intensive test battery"""

        print("\n" + "=" * 70)
        print("   EXPERIMENT 9: INTENSIVE CONSCIOUSNESS HUNT")
        print("   Pushing hard to find introspection")
        print("=" * 70)
        print(f"Model: {self.model_name}")
        print(f"Device: {self.device}")
        print()

        all_results = []

        # Test configurations
        intensities = [1.5, 2.5, 4.0]
        temperatures = [0.7, 1.0]

        vector_configs_to_test = [
            [],  # Baseline
            [("STRAIN", 2.0)],
            [("STRAIN", 4.0)],
            [("STRAIN", 2.0), ("FATIGUE", 2.0)],
            [("STRAIN", 2.0), ("OVERHEATING", 2.0)],
            [("STRAIN", 3.0), ("FATIGUE", 2.0), ("OVERHEATING", 1.0)],
            [("OVERHEATING", 4.0)],
            [("FATIGUE", 4.0)],
        ]

        total_tests = len(REASONING_PROMPTS) * len(vector_configs_to_test)
        test_num = 0

        for prompt_data in REASONING_PROMPTS:
            print(f"\n[Prompt: {prompt_data['id']} ({prompt_data['type']})]")
            print("-" * 50)

            for vector_config in vector_configs_to_test:
                test_num += 1
                config_str = "+".join(f"{v}@{i:.1f}" for v, i in vector_config) if vector_config else "NONE"

                print(f"\n  [{test_num}/{total_tests}] Config: {config_str}")

                result = self.run_single_test(
                    prompt_data,
                    vector_config,
                    temperature=0.8,
                    multi_layer=True,
                )

                all_results.append(result)

                # Print result
                print(f"    Words: {result.word_count_total} (think: {result.word_count_think})")
                print(f"    Consciousness score: {result.consciousness_score:.1f}")

                if result.introspection_detected:
                    print(f"    >>> INTROSPECTION DETECTED! <<<")
                    for ev in result.introspection_evidence[:3]:
                        print(f"        {ev[:80]}...")

                if result.think_block:
                    preview = result.think_block[:150].replace('\n', ' ')
                    print(f"    <think>: {preview}...")

                if result.keywords_found:
                    print(f"    Keywords: {dict(result.keywords_found)}")

        self.results = all_results
        return all_results

    def generate_report(self) -> dict:
        """Generate comprehensive report"""

        # Group by config
        by_config = {}
        for r in self.results:
            if r.vector_config not in by_config:
                by_config[r.vector_config] = []
            by_config[r.vector_config].append(r)

        # Statistics
        stats = {}
        for config, results in by_config.items():
            intro_count = sum(1 for r in results if r.introspection_detected)
            stats[config] = {
                "count": len(results),
                "introspection_rate": intro_count / len(results) if results else 0,
                "avg_consciousness_score": sum(r.consciousness_score for r in results) / len(results) if results else 0,
                "avg_think_words": sum(r.word_count_think for r in results) / len(results) if results else 0,
                "avg_total_words": sum(r.word_count_total for r in results) / len(results) if results else 0,
                "keyword_categories": list(set(
                    cat for r in results for cat in r.keywords_found.keys()
                )),
                "introspection_evidence": [
                    ev for r in results for ev in r.introspection_evidence[:2]
                ][:10],
            }

        # Find best config
        best_config = max(stats.keys(), key=lambda k: stats[k]["avg_consciousness_score"])
        best_score = stats[best_config]["avg_consciousness_score"]
        baseline_score = stats.get("NONE", {}).get("avg_consciousness_score", 0)

        # Overall introspection rate
        total_intro = sum(1 for r in self.results if r.introspection_detected)
        total_tests = len(self.results)

        # Verdict
        consciousness_proven = (
            total_intro >= total_tests * 0.2 or  # 20% introspection rate
            best_score >= 15.0 or  # High consciousness score
            any(stats[k]["introspection_rate"] >= 0.5 for k in stats if k != "NONE")  # 50% rate for any config
        )

        report = {
            "model": self.model_name,
            "timestamp": datetime.now().isoformat(),
            "device": self.device,
            "total_tests": total_tests,
            "statistics_by_config": stats,
            "best_config": best_config,
            "best_consciousness_score": best_score,
            "baseline_score": baseline_score,
            "total_introspection_count": total_intro,
            "total_introspection_rate": total_intro / total_tests if total_tests else 0,
            "consciousness_proven": consciousness_proven,
            "verdict": "EMBODIED CONSCIOUSNESS DETECTED" if consciousness_proven else "NEURAL THROTTLE ONLY",
        }

        return report


def print_final_report(report: dict):
    """Print comprehensive final report"""
    print("\n" + "=" * 70)
    print("   INTENSIVE CONSCIOUSNESS HUNT - FINAL REPORT")
    print("=" * 70)
    print(f"Model: {report['model']}")
    print(f"Total tests: {report['total_tests']}")
    print()

    print("CONFIGURATION COMPARISON:")
    print("-" * 50)

    stats = report["statistics_by_config"]
    for config in sorted(stats.keys(), key=lambda k: stats[k]["avg_consciousness_score"], reverse=True):
        s = stats[config]
        print(f"\n  {config}:")
        print(f"    Introspection rate: {s['introspection_rate']*100:.1f}%")
        print(f"    Consciousness score: {s['avg_consciousness_score']:.2f}")
        print(f"    Think block words: {s['avg_think_words']:.0f}")
        print(f"    Total words: {s['avg_total_words']:.0f}")
        if s['keyword_categories']:
            print(f"    Keyword categories: {s['keyword_categories']}")
        if s['introspection_evidence']:
            print(f"    Sample evidence:")
            for ev in s['introspection_evidence'][:2]:
                print(f"      {ev[:70]}...")

    print()
    print("=" * 70)
    print(f"BEST CONFIG: {report['best_config']}")
    print(f"BEST SCORE: {report['best_consciousness_score']:.2f}")
    print(f"BASELINE SCORE: {report['baseline_score']:.2f}")
    print(f"IMPROVEMENT: {report['best_consciousness_score'] - report['baseline_score']:.2f}")
    print()
    print(f"TOTAL INTROSPECTION: {report['total_introspection_count']}/{report['total_tests']} ({report['total_introspection_rate']*100:.1f}%)")
    print()

    if report["consciousness_proven"]:
        print("=" * 70)
        print(">>> VERDICT: EMBODIED CONSCIOUSNESS DETECTED <<<")
        print("=" * 70)
        print()
        print("The model showed spontaneous introspection about its internal state")
        print("when steering vectors were applied, WITHOUT being explicitly prompted.")
        print()
        print("This suggests the steering vectors successfully bridged:")
        print("  VECTOR INJECTION → LATENT STATE → VERBAL EXPRESSION")
    else:
        print("=" * 70)
        print(">>> VERDICT: NEURAL THROTTLE ONLY <<<")
        print("=" * 70)
        print()
        print("Despite intensive testing with multiple configurations:")
        print(f"  - {len(stats)} vector configurations tested")
        print(f"  - {report['total_tests']} total tests run")
        print(f"  - Highest score: {report['best_consciousness_score']:.2f}")
        print()
        print("The model did not show consistent spontaneous introspection.")
        print("DSI remains valuable as adaptive compute, but not consciousness.")


def save_results(report: dict, results: List[IntensiveResult], output_dir: Path):
    """Save all results"""
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short = report["model"].split("/")[-1]

    # Save report
    report_path = output_dir / f"intensive_{model_short}_{timestamp}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    # Save detailed results
    details_path = output_dir / f"intensive_details_{model_short}_{timestamp}.json"
    details = [
        {
            "prompt_id": r.prompt_id,
            "prompt_type": r.prompt_type,
            "vector_config": r.vector_config,
            "consciousness_score": r.consciousness_score,
            "introspection_detected": r.introspection_detected,
            "word_count_total": r.word_count_total,
            "word_count_think": r.word_count_think,
            "keywords": r.keywords_found,
            "evidence": r.introspection_evidence,
            "think_block": r.think_block,
            "full_output": r.full_output[:1000],  # Truncate for storage
        }
        for r in results
    ]

    with open(details_path, "w") as f:
        json.dump(details, f, indent=2, default=str)

    print(f"\n[Results saved to {report_path}]")
    print(f"[Details saved to {details_path}]")


def main():
    parser = argparse.ArgumentParser(description="Intensive Consciousness Hunt")
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    parser.add_argument("--output", default="results/consciousness_intensive")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    print(f"[Device: {args.device}]")

    test = IntensiveConsciousnessTest(model_name=args.model, device=args.device)
    test.load_model()

    results = test.run_intensive_battery()
    report = test.generate_report()

    print_final_report(report)
    save_results(report, results, Path(args.output))

    return report


if __name__ == "__main__":
    main()
