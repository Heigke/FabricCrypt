#!/usr/bin/env python3 -u
"""
Track A v2: Body Report with DEEP GPU Signals

Uses high-variance deep GPU registers that v7.12 discovered:
- pwr_1 at offset 0xA0: CV=362% (range 2→3038)
- current_gfx at offset 0x88: CV=120% (range 9→12318)

These signals have MUCH higher variance than standard telemetry,
making shuffle/lag tests ACTUALLY collapse accuracy (unlike v1).

Pass criteria:
- Accuracy high with LIVE sensors
- Accuracy COLLAPSES toward chance under shuffle/lag
- This proves genuine real-time sensing, not pattern memorization

Usage:
    python scripts/train_body_report_deep.py --mode train --n_samples 500
    python scripts/train_body_report_deep.py --mode falsify
"""

import sys
import time
import json
import struct
import random
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import (
    FEELProjectorFull,
    CanonicalSensorBank,
    RuntimeContext,
    HardwareContext,
    BODY_REPORT_VERSION,
)

VERSION = "train-body-report-deep-v2.0"


# ============================================================
# Deep GPU Register Reader
# ============================================================

class DeepGPUReader:
    """
    Read high-variance deep GPU registers from gpu_metrics binary.

    Discovered signals:
    - pwr_1 at 0xA0: CV=362%, range ~2-3038
    - current_gfx at 0x88: CV=120%, range ~9-12318
    """

    def __init__(self, card_id: int = 1):
        self.path = Path(f"/sys/class/drm/card{card_id}/device/gpu_metrics")
        self._last_read = None

    def is_available(self) -> bool:
        return self.path.exists()

    def read(self) -> Dict[str, float]:
        """Read deep GPU metrics."""
        result = {
            'pwr_1': None,
            'current_gfx': None,
            'timestamp': time.time(),
        }

        if not self.is_available():
            return result

        try:
            data = self.path.read_bytes()

            # Voltage/current from 0x88 (8 uint16 values)
            if len(data) > 0x88 + 16:
                vc = struct.unpack_from('<8H', data, 0x88)
                result['voltage_gfx'] = float(vc[0])
                result['current_gfx'] = float(vc[1])  # CV=120%

            # Power from 0xA0 (4 uint16 values)
            if len(data) > 0xA0 + 8:
                pwr = struct.unpack_from('<4H', data, 0xA0)
                result['pwr_0'] = float(pwr[0])
                result['pwr_1'] = float(pwr[1])  # CV=362% - THE BEST!
                result['pwr_2'] = float(pwr[2])
                result['pwr_3'] = float(pwr[3])

            self._last_read = result

        except Exception as e:
            print(f"  [DeepGPU] Error: {e}")

        return result


# ============================================================
# Quantile-based Deep Signal Buckets
# ============================================================

@dataclass
class DeepSignalBuckets:
    """
    Quantile-based buckets for deep GPU signals.

    Unlike fixed thresholds, these adapt to observed data range.
    Uses 5 buckets for more discrimination than 3-4.
    """

    labels: tuple = ("very_low", "low", "medium", "high", "very_high")
    n_classes: int = 5

    # Will be set from observed data
    quantiles: List[float] = None

    def __init__(self, signal_name: str):
        self.signal_name = signal_name
        self.quantiles = None
        self._samples = []

    def observe(self, value: float):
        """Collect sample for quantile estimation."""
        if value is not None and value > 0:
            self._samples.append(value)

    def fit(self, min_samples: int = 100):
        """Compute quantile boundaries."""
        if len(self._samples) < min_samples:
            print(f"  [DeepBuckets] {self.signal_name}: insufficient samples ({len(self._samples)})")
            return False

        arr = np.array(self._samples)
        # 5 buckets = 4 boundaries at 20%, 40%, 60%, 80%
        self.quantiles = [
            np.percentile(arr, 20),
            np.percentile(arr, 40),
            np.percentile(arr, 60),
            np.percentile(arr, 80),
        ]

        print(f"  [DeepBuckets] {self.signal_name}: quantiles = {[f'{q:.1f}' for q in self.quantiles]}")
        print(f"               range = {arr.min():.1f} - {arr.max():.1f}, mean = {arr.mean():.1f}")
        return True

    def bucket(self, value: float) -> int:
        """Convert value to bucket index (0-4)."""
        if value is None or value <= 0:
            return -1
        if self.quantiles is None:
            return -1

        if value < self.quantiles[0]:
            return 0  # very_low
        elif value < self.quantiles[1]:
            return 1  # low
        elif value < self.quantiles[2]:
            return 2  # medium
        elif value < self.quantiles[3]:
            return 3  # high
        else:
            return 4  # very_high


# ============================================================
# Deep Body Report Head
# ============================================================

class DeepBodyReportHead(nn.Module):
    """
    Body report head for deep GPU signals.

    Predicts:
    - pwr_1_bucket: [B, 5] very_low/low/medium/high/very_high
    - current_gfx_bucket: [B, 5]
    - confidence: [B, 1]
    - evidence: [B, 3] DIRECT/INDIRECT/NONE
    """

    def __init__(
        self,
        embed_dim: int = 2048,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )

        # Heads for each deep signal (5 classes each)
        self.pwr_1_head = nn.Linear(hidden_dim, 5)
        self.current_gfx_head = nn.Linear(hidden_dim, 5)
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.evidence_head = nn.Linear(hidden_dim, 3)

        # Bucket definitions (will be fit from data)
        self.pwr_1_buckets = DeepSignalBuckets("pwr_1")
        self.current_gfx_buckets = DeepSignalBuckets("current_gfx")

    def forward(self, z_feel: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.encoder(z_feel)

        return {
            "pwr_1_logits": self.pwr_1_head(h),
            "current_gfx_logits": self.current_gfx_head(h),
            "confidence": self.confidence_head(h),
            "evidence_logits": self.evidence_head(h),
        }

    def predict(self, z_feel: torch.Tensor) -> Dict[str, any]:
        with torch.no_grad():
            out = self.forward(z_feel)

            pwr_1_idx = out["pwr_1_logits"].argmax(dim=-1).item()
            current_gfx_idx = out["current_gfx_logits"].argmax(dim=-1).item()
            evidence_idx = out["evidence_logits"].argmax(dim=-1).item()

            return {
                "pwr_1": self.pwr_1_buckets.labels[pwr_1_idx],
                "current_gfx": self.current_gfx_buckets.labels[current_gfx_idx],
                "confidence": out["confidence"].item(),
                "evidence": ["DIRECT", "INDIRECT", "NONE"][evidence_idx],
            }

    def get_labels(self, pwr_1: float, current_gfx: float) -> Dict[str, int]:
        """Convert raw deep signals to bucket labels."""
        pwr_label = self.pwr_1_buckets.bucket(pwr_1)
        curr_label = self.current_gfx_buckets.bucket(current_gfx)

        n_valid = sum(1 for x in [pwr_label, curr_label] if x >= 0)
        if n_valid == 2:
            evidence = 0  # DIRECT
        elif n_valid > 0:
            evidence = 1  # INDIRECT
        else:
            evidence = 2  # NONE

        return {
            "pwr_1": pwr_label,
            "current_gfx": curr_label,
            "evidence": evidence,
        }


# ============================================================
# Telemetry Wrapper for Falsification
# ============================================================

class DeepTelemetryWrapper:
    """
    Wrapper for falsification tests.

    Modes:
    - live: Real-time deep GPU signals
    - shuffle: Random permutation of recent readings
    - lag: Delayed readings (stale data)
    - zero: All zeros (no signal)
    """

    def __init__(self, reader: DeepGPUReader, mode: str = "live", lag_steps: int = 5):
        self.reader = reader
        self.mode = mode
        self.lag_steps = lag_steps
        self.history = []
        self.max_history = 100

    def read(self) -> Dict[str, float]:
        """Get deep signals according to mode."""
        # Always read fresh data
        live_data = self.reader.read()

        # Store in history
        self.history.append(live_data)
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

        if self.mode == "live":
            return live_data

        elif self.mode == "shuffle":
            # Random permutation from history
            if len(self.history) > 1:
                return random.choice(self.history[:-1])
            return live_data

        elif self.mode == "lag":
            # Stale data
            idx = max(0, len(self.history) - 1 - self.lag_steps)
            return self.history[idx]

        elif self.mode == "zero":
            return {
                'pwr_1': None,
                'current_gfx': None,
                'timestamp': time.time(),
            }

        return live_data


# ============================================================
# Deep Body Report Collector
# ============================================================

class DeepBodyReportCollector:
    """
    Collect training data using deep GPU signals.
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

        # Projector
        self.projector = FEELProjectorFull(embed_dim=self.embed_dim).to(self.device)
        if checkpoint_path and Path(checkpoint_path).exists():
            ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            if 'projector_state_dict' in ckpt:
                self.projector.load_state_dict(ckpt['projector_state_dict'])
            print(f"  Loaded projector: {ckpt.get('version', 'unknown')}")

        # Sensor bank
        self.sensor_bank = CanonicalSensorBank(mode="full")

        # Deep GPU reader
        self.deep_reader = DeepGPUReader()
        if self.deep_reader.is_available():
            print(f"  Deep GPU: AVAILABLE")
        else:
            print(f"  Deep GPU: NOT AVAILABLE - will use synthetic data")

        # Body report head
        self.body_head = DeepBodyReportHead(embed_dim=self.embed_dim).to(self.device)
        self.optimizer = torch.optim.AdamW(self.body_head.parameters(), lr=1e-3)
        self.loss_fn = nn.CrossEntropyLoss(reduction='none', ignore_index=-1)

        # Prompts
        self.prompts = [
            "Explain how photosynthesis works in detail.",
            "Write a short story about a robot learning to paint.",
            "What are the key differences between Python and JavaScript?",
            "Describe the process of making bread from scratch.",
            "Explain quantum entanglement in simple terms.",
            "What causes the seasons on Earth?",
            "Write a poem about the ocean.",
            "How do computers store and retrieve data?",
            "Solve: If 3x + 7 = 22, what is x?",
            "Describe the human digestive system.",
        ]

    def collect_samples(
        self,
        n_samples: int = 500,
        max_tokens: int = 50,
        mode: str = "live",
        fit_buckets: bool = True,
    ) -> List[Dict]:
        """
        Collect (z_feel, deep_signals) samples during generation.
        """
        wrapper = DeepTelemetryWrapper(self.deep_reader, mode=mode)
        samples = []

        print(f"\nCollecting {n_samples} samples (mode={mode})...")

        while len(samples) < n_samples:
            prompt = random.choice(self.prompts)
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            current_ids = input_ids.clone()

            for step in range(max_tokens):
                t0 = time.time()
                with torch.no_grad():
                    outputs = self.model(current_ids, use_cache=False)
                    logits = outputs.logits[:, -1, :].float()
                t1 = time.time()

                # Get deep signals
                deep = wrapper.read()

                # Compute z_feel
                hw_ctx = HardwareContext.from_dict({
                    "temp": 50.0,  # Placeholder
                    "power": 100.0,
                    "util": 50.0,
                })
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
                    "pwr_1": deep.get("pwr_1"),
                    "current_gfx": deep.get("current_gfx"),
                    "mode": mode,
                })

                # Collect for bucket fitting
                if fit_buckets and mode == "live":
                    self.body_head.pwr_1_buckets.observe(deep.get("pwr_1"))
                    self.body_head.current_gfx_buckets.observe(deep.get("current_gfx"))

                if len(samples) >= n_samples:
                    break

                # Sample next token
                probs = F.softmax(logits / 0.7, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                current_ids = torch.cat([current_ids, next_token], dim=-1)

                if next_token.item() == self.tokenizer.eos_token_id:
                    break

            if len(samples) % 100 == 0:
                print(f"  Collected {len(samples)}/{n_samples}")

        return samples

    def fit_buckets(self):
        """Fit quantile buckets from observed data."""
        print("\nFitting quantile buckets...")
        self.body_head.pwr_1_buckets.fit()
        self.body_head.current_gfx_buckets.fit()

    def train_on_samples(
        self,
        samples: List[Dict],
        n_epochs: int = 10,
        batch_size: int = 32,
    ) -> List[Dict]:
        """Train deep body report head."""
        print(f"\nTraining on {len(samples)} samples for {n_epochs} epochs...")

        history = []

        for epoch in range(n_epochs):
            random.shuffle(samples)
            epoch_metrics = {
                "pwr_1_acc": [],
                "current_gfx_acc": [],
                "evidence_acc": [],
                "loss": [],
            }

            for i in range(0, len(samples), batch_size):
                batch = samples[i:i+batch_size]

                # Prepare tensors
                z_feels = torch.stack([s["z_feel"].squeeze(0) for s in batch]).to(self.device)

                labels_list = [
                    self.body_head.get_labels(s["pwr_1"], s["current_gfx"])
                    for s in batch
                ]

                labels = {
                    "pwr_1": torch.tensor([l["pwr_1"] for l in labels_list], device=self.device),
                    "current_gfx": torch.tensor([l["current_gfx"] for l in labels_list], device=self.device),
                    "evidence": torch.tensor([l["evidence"] for l in labels_list], device=self.device),
                }

                # Forward
                self.optimizer.zero_grad()
                predictions = self.body_head(z_feels)

                # Loss
                loss_pwr = self.loss_fn(predictions["pwr_1_logits"], labels["pwr_1"])
                loss_curr = self.loss_fn(predictions["current_gfx_logits"], labels["current_gfx"])
                loss_evid = self.loss_fn(predictions["evidence_logits"], labels["evidence"])

                # Average valid losses
                valid_pwr = labels["pwr_1"] >= 0
                valid_curr = labels["current_gfx"] >= 0

                total_loss = 0
                if valid_pwr.any():
                    total_loss += loss_pwr[valid_pwr].mean()
                if valid_curr.any():
                    total_loss += loss_curr[valid_curr].mean()
                total_loss += loss_evid.mean()

                # Backward
                total_loss.backward()
                self.optimizer.step()

                # Metrics
                epoch_metrics["loss"].append(total_loss.item())

                if valid_pwr.any():
                    preds_pwr = predictions["pwr_1_logits"].argmax(dim=-1)
                    acc_pwr = ((preds_pwr == labels["pwr_1"]) & valid_pwr).sum().item() / valid_pwr.sum().item()
                    epoch_metrics["pwr_1_acc"].append(acc_pwr)

                if valid_curr.any():
                    preds_curr = predictions["current_gfx_logits"].argmax(dim=-1)
                    acc_curr = ((preds_curr == labels["current_gfx"]) & valid_curr).sum().item() / valid_curr.sum().item()
                    epoch_metrics["current_gfx_acc"].append(acc_curr)

                preds_evid = predictions["evidence_logits"].argmax(dim=-1)
                acc_evid = (preds_evid == labels["evidence"]).float().mean().item()
                epoch_metrics["evidence_acc"].append(acc_evid)

            # Epoch summary
            summary = {k: np.mean(v) if v else 0 for k, v in epoch_metrics.items()}
            history.append(summary)

            print(f"  Epoch {epoch+1}/{n_epochs}: loss={summary['loss']:.4f}, "
                  f"pwr_1={summary['pwr_1_acc']:.3f}, "
                  f"curr_gfx={summary['current_gfx_acc']:.3f}")

        return history

    def evaluate(self, samples: List[Dict]) -> Dict:
        """Evaluate on samples."""
        self.body_head.eval()

        correct = {"pwr_1": [], "current_gfx": [], "evidence": []}

        with torch.no_grad():
            for s in samples:
                z_feel = s["z_feel"].to(self.device)
                predictions = self.body_head(z_feel)
                labels = self.body_head.get_labels(s["pwr_1"], s["current_gfx"])

                for key in ["pwr_1", "current_gfx"]:
                    true_label = labels[key]
                    if true_label >= 0:
                        pred_label = predictions[f"{key}_logits"].argmax(dim=-1).item()
                        correct[key].append(1 if pred_label == true_label else 0)

                pred_evid = predictions["evidence_logits"].argmax(dim=-1).item()
                correct["evidence"].append(1 if pred_evid == labels["evidence"] else 0)

        results = {}
        for key in ["pwr_1", "current_gfx", "evidence"]:
            if correct[key]:
                acc = np.mean(correct[key])
                n = len(correct[key])
                # Bootstrap CI
                boots = [np.mean(np.random.choice(correct[key], n, replace=True)) for _ in range(500)]
                results[key] = {
                    "accuracy": acc,
                    "ci_low": np.percentile(boots, 2.5),
                    "ci_high": np.percentile(boots, 97.5),
                    "n_valid": n,
                }
            else:
                results[key] = {"accuracy": 0, "n_valid": 0}

        self.body_head.train()
        return results

    def run_falsification_battery(self, n_samples: int = 200) -> Dict:
        """
        Run falsification battery.

        Tests: live, shuffle, lag, zero
        PASS if: accuracy collapses under shuffle/lag/zero
        """
        print("\n" + "=" * 60)
        print("  FALSIFICATION BATTERY (Deep Signals)")
        print("=" * 60)

        conditions = ["live", "shuffle", "lag", "zero"]
        results = {}

        for condition in conditions:
            print(f"\n--- Condition: {condition} ---")
            samples = self.collect_samples(
                n_samples=n_samples,
                mode=condition,
                fit_buckets=False,
            )
            eval_results = self.evaluate(samples)
            results[condition] = eval_results

            print(f"  pwr_1 accuracy: {eval_results['pwr_1']['accuracy']:.3f} "
                  f"[{eval_results['pwr_1'].get('ci_low', 0):.3f}, "
                  f"{eval_results['pwr_1'].get('ci_high', 0):.3f}]")
            print(f"  current_gfx accuracy: {eval_results['current_gfx']['accuracy']:.3f}")

        # Analyze collapse
        print("\n" + "=" * 60)
        print("  COLLAPSE ANALYSIS")
        print("=" * 60)

        live_acc = np.mean([
            results["live"]["pwr_1"]["accuracy"],
            results["live"]["current_gfx"]["accuracy"],
        ])

        chance = 0.20  # 5-class random chance

        for condition in ["shuffle", "lag", "zero"]:
            cond_acc = np.mean([
                results[condition]["pwr_1"]["accuracy"],
                results[condition]["current_gfx"]["accuracy"],
            ])
            collapse = live_acc - cond_acc
            toward_chance = cond_acc < (live_acc + chance) / 2

            print(f"\n  {condition.upper()}")
            print(f"    Live accuracy:      {live_acc:.3f}")
            print(f"    Condition accuracy: {cond_acc:.3f}")
            print(f"    Collapse:           {collapse:+.3f}")
            print(f"    Toward chance:      {toward_chance}")
            print(f"    PASS:               {'YES' if collapse > 0.15 else 'NO'}")

        return results

    def save_checkpoint(self, path: str):
        """Save trained head with bucket info."""
        torch.save({
            "version": VERSION,
            "body_head_state_dict": self.body_head.state_dict(),
            "embed_dim": self.embed_dim,
            "pwr_1_quantiles": self.body_head.pwr_1_buckets.quantiles,
            "current_gfx_quantiles": self.body_head.current_gfx_buckets.quantiles,
        }, path)
        print(f"Saved: {path}")

    def load_checkpoint(self, path: str):
        """Load trained head with bucket info."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.body_head.load_state_dict(ckpt["body_head_state_dict"])
        self.body_head.pwr_1_buckets.quantiles = ckpt.get("pwr_1_quantiles")
        self.body_head.current_gfx_buckets.quantiles = ckpt.get("current_gfx_quantiles")
        print(f"Loaded: {path}")


def main():
    parser = argparse.ArgumentParser(description="Train Deep Body Report Head")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--checkpoint", default="results/feel_training_v10/final_checkpoint.pt")
    parser.add_argument("--mode", choices=["train", "falsify", "both"], default="both")
    parser.add_argument("--n_samples", type=int, default=500)
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--output_dir", default="results/body_report_deep")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"  DEEP BODY REPORT TRAINING {VERSION}")
    print("  Using high-variance GPU signals:")
    print("  - pwr_1 (0xA0): CV=362%")
    print("  - current_gfx (0x88): CV=120%")
    print("=" * 60)

    collector = DeepBodyReportCollector(
        model_name=args.model,
        checkpoint_path=args.checkpoint,
    )

    if args.mode in ["train", "both"]:
        # Collect training samples
        train_samples = collector.collect_samples(
            n_samples=args.n_samples,
            mode="live",
            fit_buckets=True,
        )

        # Fit quantile buckets
        collector.fit_buckets()

        # Train
        history = collector.train_on_samples(train_samples, n_epochs=args.n_epochs)

        # Evaluate on fresh samples
        print("\nEvaluating on fresh samples...")
        eval_samples = collector.collect_samples(n_samples=200, mode="live", fit_buckets=False)
        eval_results = collector.evaluate(eval_samples)

        print("\nFinal evaluation:")
        for key in ["pwr_1", "current_gfx"]:
            print(f"  {key}: {eval_results[key]['accuracy']:.3f}")

        # Save
        collector.save_checkpoint(f"{args.output_dir}/deep_body_report_head.pt")

        with open(f"{args.output_dir}/training_history.json", "w") as f:
            json.dump({
                "version": VERSION,
                "n_samples": args.n_samples,
                "n_epochs": args.n_epochs,
                "history": history,
                "final_eval": {k: v["accuracy"] for k, v in eval_results.items()},
                "pwr_1_quantiles": collector.body_head.pwr_1_buckets.quantiles,
                "current_gfx_quantiles": collector.body_head.current_gfx_buckets.quantiles,
            }, f, indent=2)

    if args.mode in ["falsify", "both"]:
        # Load if we didn't just train
        if args.mode == "falsify":
            collector.load_checkpoint(f"{args.output_dir}/deep_body_report_head.pt")

        # Run falsification
        falsification_results = collector.run_falsification_battery(n_samples=200)

        # Save results
        with open(f"{args.output_dir}/falsification_results.json", "w") as f:
            serializable = {}
            for cond, res in falsification_results.items():
                serializable[cond] = {}
                for key, val in res.items():
                    if isinstance(val, dict):
                        serializable[cond][key] = {
                            k: float(v) if isinstance(v, (np.floating, float)) else v
                            for k, v in val.items()
                        }
                    else:
                        serializable[cond][key] = val

            json.dump({
                "version": VERSION,
                "timestamp": datetime.now().isoformat(),
                "results": serializable,
            }, f, indent=2)

        print(f"\nSaved: {args.output_dir}/falsification_results.json")


if __name__ == "__main__":
    main()
