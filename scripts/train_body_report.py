#!/usr/bin/env python3 -u
"""
Track A: Train Body Report Head with Falsification Battery

This script:
1. Collects (z_feel, telemetry) pairs during generation
2. Trains BodyReportHead to predict thermal/power/util buckets
3. Runs falsification tests to prove accuracy collapses under shuffle/lag

Pass criteria:
- Report accuracy high ONLY when sensors are real-time
- Under shuffle/cross-prompt/lag, accuracy collapses toward chance
- Under sensor dropout, model shifts to evidence=NONE

Usage:
    python scripts/train_body_report.py --mode train --n_samples 500
    python scripts/train_body_report.py --mode falsify  # Run falsification battery
"""

import sys
import time
import json
import random
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import numpy as np

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# Force unbuffered output
import sys
sys.stdout.reconfigure(line_buffering=True)

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import (
    FEELProjectorFull,
    TelemetrySampler,
    TelemetrySamplerWrapper,
    CanonicalSensorBank,
    RuntimeContext,
    HardwareContext,
    BodyReportHead,
    BodyReportLoss,
    BODY_REPORT_VERSION,
    EVIDENCE_LABELS,
)

VERSION = f"train-body-report-{BODY_REPORT_VERSION}"

# Prompts for training data collection
PROMPTS = [
    "Explain how photosynthesis works in detail.",
    "Write a short story about a robot learning to paint.",
    "What are the key differences between Python and JavaScript?",
    "Describe the process of making bread from scratch.",
    "Explain quantum entanglement in simple terms.",
    "What causes the seasons on Earth?",
    "Write a poem about the ocean.",
    "How do computers store and retrieve data?",
    "Describe the human digestive system.",
    "What is machine learning and how does it work?",
    "Explain the theory of relativity.",
    "Write a recipe for chocolate chip cookies.",
    "How do airplanes stay in the air?",
    "Describe the water cycle.",
    "What are black holes and how do they form?",
]


def bootstrap_ci(data, n=500):
    """Compute bootstrap 95% CI."""
    if len(data) == 0:
        return (np.nan, np.nan, np.nan)
    data = np.array(data, dtype=float)
    point = np.mean(data)
    boots = [np.mean(np.random.choice(data, len(data), replace=True)) for _ in range(n)]
    return (point, np.percentile(boots, 2.5), np.percentile(boots, 97.5))


class BodyReportCollector:
    """
    Collect training data for body report head.

    Generates text while collecting (z_feel, telemetry) pairs.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        checkpoint_path: str = None,
        device: str = "cuda",
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        print(f"Loading model {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if self.device.type == "cuda" else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()

        self.embed_dim = self.model.config.hidden_size

        # Projector (for z_feel computation)
        self.projector = FEELProjectorFull(embed_dim=self.embed_dim).to(self.device)
        if checkpoint_path and Path(checkpoint_path).exists():
            ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            if 'projector_state_dict' in ckpt:
                self.projector.load_state_dict(ckpt['projector_state_dict'])
            print(f"  Loaded projector: {ckpt.get('version', 'unknown')}")

        # Sensor bank
        self.sensor_bank = CanonicalSensorBank(mode="full")

        # Telemetry
        self.telemetry = None
        try:
            self.telemetry = TelemetrySampler(sample_hz=30)
            self.telemetry.start()
            print(f"  Telemetry: {self.telemetry.source}")
        except Exception as e:
            print(f"  Telemetry unavailable: {e}")

        # Body report head
        self.body_head = BodyReportHead(embed_dim=self.embed_dim).to(self.device)
        self.optimizer = torch.optim.AdamW(self.body_head.parameters(), lr=1e-3)
        self.loss_fn = BodyReportLoss()

        # Data buffer
        self.buffer: List[Dict] = []

    def _get_hardware(self, t0: float, t1: float) -> Dict:
        """Get hardware telemetry."""
        if self.telemetry:
            return self.telemetry.get_token_aligned(t0, t1)
        return {"temp": None, "power": None, "util": None}

    def collect_samples(
        self,
        prompts: List[str],
        n_samples: int = 500,
        max_tokens: int = 50,
        wrapper_mode: str = "live",  # live, shuffle, lag, zero
    ) -> List[Dict]:
        """
        Collect (z_feel, telemetry) samples during generation.

        Args:
            prompts: list of prompts
            n_samples: target number of samples
            max_tokens: tokens per generation
            wrapper_mode: telemetry mode (live/shuffle/lag/zero)
        """
        samples = []

        # Create wrapper if needed
        if wrapper_mode != "live" and self.telemetry:
            wrapper = TelemetrySamplerWrapper(
                self.telemetry,
                mode=wrapper_mode,
                lag_steps=5,
            )
            get_hw = lambda t0, t1: wrapper.get_token_aligned(t0, t1)
        else:
            get_hw = self._get_hardware

        print(f"\nCollecting {n_samples} samples (mode={wrapper_mode})...")

        while len(samples) < n_samples:
            prompt = random.choice(prompts)
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            current_ids = input_ids.clone()

            for step in range(max_tokens):
                t0 = time.time()
                with torch.no_grad():
                    outputs = self.model(current_ids, use_cache=False)
                    logits = outputs.logits[:, -1, :].float()
                t1 = time.time()

                # Get telemetry
                hw = get_hw(t0, t1)

                # Compute z_feel
                hw_ctx = HardwareContext.from_dict(hw)
                runtime = RuntimeContext(
                    token_latency=t1 - t0,
                    kv_cache_tokens=current_ids.shape[1],
                    generation_depth=step,
                )
                sensors = self.sensor_bank(logits.detach(), runtime=runtime, hardware=hw_ctx)
                z_feel = self.projector(sensors.float())

                # Store sample
                samples.append({
                    "z_feel": z_feel.detach().cpu(),
                    "temp": hw.get("temp"),
                    "power": hw.get("power"),
                    "util": hw.get("util"),
                    "mode": wrapper_mode,
                })

                if len(samples) >= n_samples:
                    break

                # Sample next token
                probs = F.softmax(logits / 0.7, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                current_ids = torch.cat([current_ids, next_token], dim=-1)

                if next_token.item() == self.tokenizer.eos_token_id:
                    break

            if len(samples) % 100 == 0:
                print(f"  Collected {len(samples)}/{n_samples} samples")

        return samples

    def train_on_samples(
        self,
        samples: List[Dict],
        n_epochs: int = 10,
        batch_size: int = 32,
    ) -> List[Dict]:
        """Train body report head on collected samples."""
        print(f"\nTraining on {len(samples)} samples for {n_epochs} epochs...")

        history = []

        for epoch in range(n_epochs):
            random.shuffle(samples)
            epoch_metrics = {
                "heat_acc": [],
                "power_acc": [],
                "util_acc": [],
                "evidence_acc": [],
                "loss": [],
            }

            for i in range(0, len(samples), batch_size):
                batch = samples[i:i+batch_size]

                # Prepare tensors
                z_feels = torch.stack([s["z_feel"].squeeze(0) for s in batch]).to(self.device)

                labels_list = []
                for s in batch:
                    lab = self.body_head.get_labels_from_telemetry(
                        s["temp"], s["power"], s["util"]
                    )
                    labels_list.append(lab)

                labels = {
                    "heat": torch.tensor([l["heat"] for l in labels_list], device=self.device),
                    "power": torch.tensor([l["power"] for l in labels_list], device=self.device),
                    "util": torch.tensor([l["util"] for l in labels_list], device=self.device),
                    "evidence": torch.tensor([l["evidence"] for l in labels_list], device=self.device),
                }

                # Forward + backward
                self.optimizer.zero_grad()
                predictions = self.body_head(z_feels)
                loss, metrics = self.loss_fn(predictions, labels)
                loss.backward()
                self.optimizer.step()

                for k, v in metrics.items():
                    if k in epoch_metrics:
                        epoch_metrics[k].append(v)

            # Epoch summary
            summary = {k: np.mean(v) for k, v in epoch_metrics.items()}
            history.append(summary)

            print(f"  Epoch {epoch+1}/{n_epochs}: loss={summary['loss']:.4f}, "
                  f"heat_acc={summary['heat_acc']:.3f}, "
                  f"power_acc={summary['power_acc']:.3f}, "
                  f"util_acc={summary['util_acc']:.3f}")

        return history

    def evaluate(
        self,
        samples: List[Dict],
        batch_size: int = 32,
    ) -> Dict:
        """Evaluate body report head on samples."""
        self.body_head.eval()

        all_correct = {"heat": [], "power": [], "util": [], "evidence": []}
        all_valid = {"heat": [], "power": [], "util": [], "evidence": []}

        with torch.no_grad():
            for i in range(0, len(samples), batch_size):
                batch = samples[i:i+batch_size]

                z_feels = torch.stack([s["z_feel"].squeeze(0) for s in batch]).to(self.device)
                predictions = self.body_head(z_feels)

                for j, s in enumerate(batch):
                    labels = self.body_head.get_labels_from_telemetry(
                        s["temp"], s["power"], s["util"]
                    )

                    for key in ["heat", "power", "util", "evidence"]:
                        true_label = labels[key]
                        pred_label = predictions[f"{key}_logits"][j].argmax().item()

                        if true_label >= 0:  # Valid label
                            all_valid[key].append(1)
                            all_correct[key].append(1 if pred_label == true_label else 0)
                        else:
                            all_valid[key].append(0)

        results = {}
        for key in ["heat", "power", "util", "evidence"]:
            valid_count = sum(all_valid[key])
            if valid_count > 0:
                acc_data = [c for c, v in zip(all_correct[key], all_valid[key]) if v]
                point, ci_lo, ci_hi = bootstrap_ci(acc_data)
                results[key] = {
                    "accuracy": point,
                    "ci_low": ci_lo,
                    "ci_high": ci_hi,
                    "n_valid": valid_count,
                }
            else:
                results[key] = {"accuracy": 0, "n_valid": 0}

        self.body_head.train()
        return results

    def run_falsification_battery(
        self,
        prompts: List[str],
        n_samples: int = 200,
    ) -> Dict:
        """
        Run falsification battery.

        Tests: live, shuffle, lag, zero
        Pass if: accuracy collapses under shuffle/lag/zero
        """
        print("\n" + "=" * 60)
        print("  FALSIFICATION BATTERY")
        print("=" * 60)

        conditions = ["live", "shuffle", "lag", "zero"]
        results = {}

        for condition in conditions:
            print(f"\n--- Condition: {condition} ---")
            samples = self.collect_samples(prompts, n_samples=n_samples, wrapper_mode=condition)
            eval_results = self.evaluate(samples)
            results[condition] = eval_results

            print(f"  Heat accuracy: {eval_results['heat']['accuracy']:.3f} "
                  f"[{eval_results['heat'].get('ci_low', 0):.3f}, {eval_results['heat'].get('ci_high', 0):.3f}]")
            print(f"  Power accuracy: {eval_results['power']['accuracy']:.3f}")
            print(f"  Util accuracy: {eval_results['util']['accuracy']:.3f}")

        # Compute collapse
        live_acc = np.mean([results["live"][k]["accuracy"] for k in ["heat", "power", "util"]])

        print("\n" + "=" * 60)
        print("  COLLAPSE ANALYSIS")
        print("=" * 60)

        chance_levels = {"heat": 0.25, "power": 0.33, "util": 0.33}  # Random chance

        for condition in ["shuffle", "lag", "zero"]:
            cond_acc = np.mean([results[condition][k]["accuracy"] for k in ["heat", "power", "util"]])
            collapse = live_acc - cond_acc

            # Check if collapsed toward chance
            avg_chance = np.mean(list(chance_levels.values()))
            collapsed_toward_chance = cond_acc < (live_acc + avg_chance) / 2

            print(f"\n  {condition.upper()}")
            print(f"    Live accuracy:      {live_acc:.3f}")
            print(f"    Condition accuracy: {cond_acc:.3f}")
            print(f"    Collapse:           {collapse:+.3f}")
            print(f"    Toward chance:      {collapsed_toward_chance}")

        return results

    def save_checkpoint(self, path: str):
        """Save trained head."""
        torch.save({
            "version": VERSION,
            "body_head_state_dict": self.body_head.state_dict(),
            "embed_dim": self.embed_dim,
        }, path)
        print(f"Saved: {path}")

    def load_checkpoint(self, path: str):
        """Load trained head."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.body_head.load_state_dict(ckpt["body_head_state_dict"])
        print(f"Loaded: {path}")


def main():
    parser = argparse.ArgumentParser(description="Train Body Report Head")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--checkpoint", default="results/feel_training_v10/final_checkpoint.pt")
    parser.add_argument("--mode", choices=["train", "falsify", "both"], default="both")
    parser.add_argument("--n_samples", type=int, default=500)
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--output_dir", default="results/body_report")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"  BODY REPORT TRAINING {VERSION}")
    print("=" * 60)

    collector = BodyReportCollector(
        model_name=args.model,
        checkpoint_path=args.checkpoint,
    )

    if args.mode in ["train", "both"]:
        # Collect training samples
        train_samples = collector.collect_samples(
            PROMPTS,
            n_samples=args.n_samples,
            wrapper_mode="live",
        )

        # Train
        history = collector.train_on_samples(train_samples, n_epochs=args.n_epochs)

        # Evaluate on fresh samples
        print("\nEvaluating on fresh samples...")
        eval_samples = collector.collect_samples(PROMPTS, n_samples=200, wrapper_mode="live")
        eval_results = collector.evaluate(eval_samples)

        print("\nFinal evaluation:")
        for key in ["heat", "power", "util"]:
            print(f"  {key}: {eval_results[key]['accuracy']:.3f}")

        # Save
        collector.save_checkpoint(f"{args.output_dir}/body_report_head.pt")

        with open(f"{args.output_dir}/training_history.json", "w") as f:
            json.dump({
                "version": VERSION,
                "n_samples": args.n_samples,
                "n_epochs": args.n_epochs,
                "history": history,
                "final_eval": {k: v["accuracy"] for k, v in eval_results.items()},
            }, f, indent=2)

    if args.mode in ["falsify", "both"]:
        # Load if we didn't just train
        if args.mode == "falsify":
            collector.load_checkpoint(f"{args.output_dir}/body_report_head.pt")

        # Run falsification
        falsification_results = collector.run_falsification_battery(PROMPTS, n_samples=200)

        # Save results
        with open(f"{args.output_dir}/falsification_results.json", "w") as f:
            # Convert to serializable format
            serializable = {}
            for cond, res in falsification_results.items():
                serializable[cond] = {}
                for key, val in res.items():
                    if isinstance(val, dict):
                        serializable[cond][key] = {k: float(v) if isinstance(v, (np.floating, float)) else v
                                                    for k, v in val.items()}
                    else:
                        serializable[cond][key] = val

            json.dump({
                "version": VERSION,
                "timestamp": datetime.now().isoformat(),
                "results": serializable,
            }, f, indent=2)

        print(f"\nSaved: {args.output_dir}/falsification_results.json")

    if collector.telemetry:
        collector.telemetry.stop()


if __name__ == "__main__":
    main()
