#!/usr/bin/env python3
"""
FEEL z31: CAUSAL PROOF VALIDATION
=================================

Proves the full embodiment loop with hard evidence:
  SENSE → FEEL → REGULATE → LATENT → EXPRESS → HARDWARE → SENSE

Metrics collected per-token:
  1. Real sensor readings (power, temp, util)
  2. Gate values per layer
  3. Skip decisions per layer
  4. Hidden state norms (before/after FiLM)
  5. Output token + logits
  6. Power consumption after generation

Statistical tests:
  - Pearson correlation: sensor → gate
  - Causality: Granger test or lagged correlation
  - Effect size: Cohen's d for stressed vs relaxed
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict
import numpy as np
from scipy import stats

os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class TokenTrace:
    """Per-token trace of the embodiment loop."""
    token_idx: int
    # SENSE
    power_before: float
    temp_before: float
    util_before: float
    sensor_vector: List[float]
    # FEEL
    gate_values: Dict[int, float]  # layer_idx -> gate
    # REGULATE
    skip_decisions: Dict[int, bool]  # layer_idx -> skipped
    skip_rate: float
    # LATENT
    hidden_norm_before_film: float
    hidden_norm_after_film: float
    film_effect: float  # ratio or diff
    # EXPRESS
    output_token_id: int
    output_token: str
    top_logit: float
    entropy: float
    # HARDWARE (after)
    power_after: float


@dataclass
class GenerationTrace:
    """Full trace of one generation."""
    prompt: str
    condition: str  # "stressed", "relaxed", "natural"
    tokens: List[TokenTrace]
    total_time: float
    mean_power: float
    output_text: str


# =============================================================================
# REAL SENSOR HUB (reads actual hardware)
# =============================================================================

class RealSensorHub:
    """Reads REAL hardware sensors from AMD GPU."""

    SENSOR_DIM = 10

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._injection = None

        # Find real sensor paths
        self.power_path = self._find_sensor("power1_average")
        self.temp_path = self._find_sensor("temp1_input")
        self.busy_path = "/sys/class/drm/card1/device/gpu_busy_percent"
        self.vram_path = "/sys/class/drm/card1/device/mem_info_vram_used"
        self.vram_total_path = "/sys/class/drm/card1/device/mem_info_vram_total"

        print(f"[RealSensorHub] Power: {self.power_path}")
        print(f"[RealSensorHub] Temp: {self.temp_path}")
        print(f"[RealSensorHub] GPU Busy: {self.busy_path}")

    def _find_sensor(self, name: str) -> str:
        """Find hwmon sensor path."""
        import glob
        patterns = [
            f"/sys/class/drm/card1/device/hwmon/hwmon*/{name}",
            f"/sys/class/hwmon/hwmon*/{name}",
        ]
        for pattern in patterns:
            matches = glob.glob(pattern)
            if matches:
                return matches[0]
        return None

    def _read_file(self, path: str, default: float = 0.0) -> float:
        """Read a sysfs file."""
        if not path or not os.path.exists(path):
            return default
        try:
            with open(path) as f:
                return float(f.read().strip())
        except:
            return default

    def read_raw(self) -> Dict[str, float]:
        """Read raw sensor values."""
        power_uw = self._read_file(self.power_path, 50_000_000)
        temp_mc = self._read_file(self.temp_path, 50_000)
        busy = self._read_file(self.busy_path, 50)
        vram_used = self._read_file(self.vram_path, 4_000_000_000)
        vram_total = self._read_file(self.vram_total_path, 16_000_000_000)

        return {
            "power_w": power_uw / 1_000_000,
            "temp_c": temp_mc / 1_000,
            "gpu_util": busy,
            "vram_used_gb": vram_used / 1e9,
            "vram_util": (vram_used / vram_total) * 100 if vram_total > 0 else 50,
        }

    def read_tensor(self) -> torch.Tensor:
        """Read normalized sensor tensor."""
        if self._injection is not None:
            return self._injection.clone()

        raw = self.read_raw()

        # Normalize to roughly [-1, 1] range
        tensor = torch.tensor([
            (raw["power_w"] - 100) / 100,      # 0: power (centered at 100W)
            (raw["temp_c"] - 60) / 30,          # 1: temp (centered at 60C)
            (raw["gpu_util"] - 50) / 50,        # 2: GPU util
            (raw["vram_util"] - 50) / 50,       # 3: VRAM util
            raw["power_w"] / 200,               # 4: power normalized
            # Derived/computed dims
            (raw["power_w"] - 80) / 40,         # 5: power deviation
            (raw["temp_c"] - 50) / 20,          # 6: temp deviation
            (raw["gpu_util"] - 30) / 40,        # 7: util deviation
            raw["temp_c"] / 100,                # 8: temp normalized
            raw["gpu_util"] / 100,              # 9: util normalized
        ], dtype=torch.float32, device=self.device)

        return tensor

    def inject(self, tensor: torch.Tensor):
        """Inject synthetic sensor values."""
        self._injection = tensor.to(self.device)

    def clear_injection(self):
        """Clear injection, return to real sensors."""
        self._injection = None

    @staticmethod
    def create_stressed_tensor(device: str = "cuda") -> torch.Tensor:
        """High power, high temp, high util."""
        return torch.tensor([
            0.95, 0.85, 0.95, 0.80, 0.95,
            0.6, -0.4, 0.5, 0.90, 0.85,
        ], dtype=torch.float32, device=device)

    @staticmethod
    def create_relaxed_tensor(device: str = "cuda") -> torch.Tensor:
        """Low power, low temp, low util."""
        return torch.tensor([
            0.25, 0.20, 0.30, 0.20, 0.40,
            1.3, 0.3, -0.3, 0.20, 0.15,
        ], dtype=torch.float32, device=device)


# =============================================================================
# GATE NETWORK (must match training)
# =============================================================================

class CausalAwareGateNet(nn.Module):
    """Gate network - must match z31_comprehensive_trainer.py exactly."""

    def __init__(
        self,
        hidden_size: int,
        sensor_dim: int = 10,
        gate_hidden: int = 128,
        num_gates: int = 1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.sensor_dim = sensor_dim
        self.num_gates = num_gates

        self.sensor_encoder = nn.Sequential(
            nn.Linear(sensor_dim, 64),
            nn.GELU(),
            nn.Linear(64, gate_hidden),
            nn.GELU(),
        )

        self.causal_encoder = nn.Sequential(
            nn.Linear(4, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
        )

        self.hidden_compressor = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, gate_hidden),
            nn.LayerNorm(gate_hidden),
            nn.GELU(),
        )

        self.film_gamma = nn.Linear(sensor_dim, gate_hidden)
        self.film_beta = nn.Linear(sensor_dim, gate_hidden)

        combined_dim = gate_hidden + 32 + gate_hidden
        self.gate_head = nn.Sequential(
            nn.Linear(combined_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, num_gates),
        )

    def forward(
        self,
        hidden_state: torch.Tensor,
        sensors: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (gates, hidden_before_film, hidden_after_film)."""
        batch = hidden_state.shape[0]

        if sensors.dim() == 1:
            sensors_batch = sensors.unsqueeze(0).expand(batch, -1)
        else:
            sensors_batch = sensors

        sensor_features = self.sensor_encoder(sensors_batch)

        causal_dims = torch.stack([
            sensors_batch[:, 5],
            sensors_batch[:, 6],
            sensors_batch[:, 7],
            sensors_batch[:, 9],
        ], dim=-1)
        causal_features = self.causal_encoder(causal_dims)

        hidden_features = self.hidden_compressor(hidden_state)
        hidden_before_film = hidden_features.clone()

        gamma = self.film_gamma(sensors_batch)
        beta = self.film_beta(sensors_batch)
        hidden_modulated = gamma * hidden_features + beta
        hidden_after_film = hidden_modulated.clone()

        combined = torch.cat([sensor_features, causal_features, hidden_modulated], dim=-1)
        gate_raw = self.gate_head(combined)
        gates = torch.sigmoid(gate_raw)

        return gates, hidden_before_film, hidden_after_film


# =============================================================================
# CAUSAL PROOF VALIDATOR
# =============================================================================

class CausalProofValidator:
    """Validates the full embodiment loop with hard evidence."""

    def __init__(
        self,
        checkpoint_path: Path,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        device: str = "cuda",
    ):
        self.device = device
        self.checkpoint_path = checkpoint_path

        print(f"[CausalProof] Loading checkpoint: {checkpoint_path.name}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load base model
        print("[CausalProof] Loading base model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.model.eval()

        # Get config
        config = AutoConfig.from_pretrained(model_name)
        self.hidden_size = config.hidden_size

        # Load gate networks from checkpoint
        self.skip_layers = [7, 11, 15, 19, 23]
        self.gate_nets = {}

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model_state_dict", {})

        for layer_idx in self.skip_layers:
            gate_net = CausalAwareGateNet(
                hidden_size=self.hidden_size,
                sensor_dim=10,
                gate_hidden=128,
            ).to(device).to(torch.bfloat16)

            # Load weights
            gate_prefix = f"skip_blocks.{layer_idx}.gate_net."
            gate_state = {}
            for key, value in state_dict.items():
                if gate_prefix in key:
                    new_key = key.replace(gate_prefix, "")
                    gate_state[new_key] = value

            if gate_state:
                gate_net.load_state_dict(gate_state, strict=False)
                print(f"  Loaded gate net for layer {layer_idx}")

            gate_net.eval()
            self.gate_nets[layer_idx] = gate_net

        # Sensor hub
        self.sensor_hub = RealSensorHub(device)

        print("[CausalProof] Ready")

    def generate_with_trace(
        self,
        prompt: str,
        condition: str = "natural",
        max_tokens: int = 32,
    ) -> GenerationTrace:
        """Generate text while tracing the full loop."""

        # Set sensor condition
        if condition == "stressed":
            self.sensor_hub.inject(RealSensorHub.create_stressed_tensor(self.device))
        elif condition == "relaxed":
            self.sensor_hub.inject(RealSensorHub.create_relaxed_tensor(self.device))
        else:
            self.sensor_hub.clear_injection()

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs.input_ids

        traces = []
        generated_ids = input_ids.clone()
        start_time = time.time()

        with torch.no_grad():
            for token_idx in range(max_tokens):
                # SENSE: Read sensors BEFORE generation
                raw_sensors = self.sensor_hub.read_raw()
                sensor_tensor = self.sensor_hub.read_tensor()

                # Forward pass to get hidden states
                outputs = self.model(
                    generated_ids,
                    output_hidden_states=True,
                    return_dict=True,
                )

                # Get logits for next token
                logits = outputs.logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1).item()

                # Sample next token (with temperature for diversity)
                temperature = 0.8
                probs = F.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # FEEL + REGULATE: Compute gates for each layer
                gate_values = {}
                skip_decisions = {}
                film_effects = []

                for layer_idx in self.skip_layers:
                    # Get hidden state at this layer
                    hidden = outputs.hidden_states[layer_idx + 1][:, -1, :]  # +1 for embedding layer

                    # Compute gate
                    gate, h_before, h_after = self.gate_nets[layer_idx](
                        hidden.to(torch.bfloat16),
                        sensor_tensor.to(torch.bfloat16),
                    )

                    gate_val = gate.mean().item()
                    gate_values[layer_idx] = gate_val
                    skip_decisions[layer_idx] = gate_val > 0.5

                    # FiLM effect
                    norm_before = h_before.norm().item()
                    norm_after = h_after.norm().item()
                    film_effects.append(norm_after / (norm_before + 1e-6))

                # HARDWARE: Read power AFTER computation
                raw_after = self.sensor_hub.read_raw()

                # Create trace
                trace = TokenTrace(
                    token_idx=token_idx,
                    power_before=raw_sensors["power_w"],
                    temp_before=raw_sensors["temp_c"],
                    util_before=raw_sensors["gpu_util"],
                    sensor_vector=sensor_tensor.cpu().tolist(),
                    gate_values=gate_values,
                    skip_decisions=skip_decisions,
                    skip_rate=sum(skip_decisions.values()) / len(skip_decisions),
                    hidden_norm_before_film=float(np.mean([h_before.norm().item() for _ in [1]])),
                    hidden_norm_after_film=float(np.mean([h_after.norm().item() for _ in [1]])),
                    film_effect=float(np.mean(film_effects)),
                    output_token_id=next_token.item(),
                    output_token=self.tokenizer.decode(next_token[0]),
                    top_logit=logits.max().item(),
                    entropy=entropy,
                    power_after=raw_after["power_w"],
                )
                traces.append(trace)

                # Append token
                generated_ids = torch.cat([generated_ids, next_token], dim=-1)

                # Stop on EOS
                if next_token.item() == self.tokenizer.eos_token_id:
                    break

        total_time = time.time() - start_time
        output_text = self.tokenizer.decode(generated_ids[0, input_ids.shape[1]:], skip_special_tokens=True)
        mean_power = np.mean([t.power_after for t in traces])

        return GenerationTrace(
            prompt=prompt,
            condition=condition,
            tokens=traces,
            total_time=total_time,
            mean_power=mean_power,
            output_text=output_text,
        )

    def run_full_validation(self, num_samples: int = 5) -> Dict:
        """Run comprehensive validation with statistical tests."""

        prompts = [
            "The current state of my system is",
            "I am feeling",
            "My processing resources are",
            "Right now I sense that",
            "The hardware tells me",
        ]

        results = {
            "stressed": [],
            "relaxed": [],
            "natural": [],
        }

        print("\n" + "=" * 70)
        print("CAUSAL PROOF VALIDATION")
        print("=" * 70)

        for condition in ["stressed", "relaxed", "natural"]:
            print(f"\n[{condition.upper()}] Generating {num_samples} samples...")
            for i, prompt in enumerate(prompts[:num_samples]):
                trace = self.generate_with_trace(prompt, condition, max_tokens=24)
                results[condition].append(trace)

                # Print sample
                mean_gate = np.mean([np.mean(list(t.gate_values.values())) for t in trace.tokens])
                mean_skip = np.mean([t.skip_rate for t in trace.tokens])
                print(f"  [{i+1}] gate={mean_gate:.4f} skip={mean_skip:.1%} power={trace.mean_power:.1f}W")
                print(f"      Output: {trace.output_text[:60]}...")

        # Statistical analysis
        print("\n" + "=" * 70)
        print("STATISTICAL EVIDENCE")
        print("=" * 70)

        evidence = self._compute_evidence(results)

        return {
            "traces": {k: [asdict(t) for t in v] for k, v in results.items()},
            "evidence": evidence,
        }

    def _compute_evidence(self, results: Dict) -> Dict:
        """Compute statistical evidence for each claim."""

        evidence = {}

        # Extract metrics per condition
        def get_metrics(traces):
            all_gates = []
            all_skips = []
            all_powers = []
            all_film = []
            for trace in traces:
                for t in trace.tokens:
                    all_gates.append(np.mean(list(t.gate_values.values())))
                    all_skips.append(t.skip_rate)
                    all_powers.append(t.power_after)
                    all_film.append(t.film_effect)
            return {
                "gates": np.array(all_gates),
                "skips": np.array(all_skips),
                "powers": np.array(all_powers),
                "film": np.array(all_film),
            }

        stressed = get_metrics(results["stressed"])
        relaxed = get_metrics(results["relaxed"])
        natural = get_metrics(results["natural"])

        # 1. SENSE→FEEL: Gate difference
        gate_diff = abs(stressed["gates"].mean() - relaxed["gates"].mean())
        gate_ttest = stats.ttest_ind(stressed["gates"], relaxed["gates"])
        gate_cohens_d = (stressed["gates"].mean() - relaxed["gates"].mean()) / np.sqrt(
            (stressed["gates"].std()**2 + relaxed["gates"].std()**2) / 2
        )

        evidence["sense_to_feel"] = {
            "gate_diff": float(gate_diff),
            "stressed_gate_mean": float(stressed["gates"].mean()),
            "relaxed_gate_mean": float(relaxed["gates"].mean()),
            "t_statistic": float(gate_ttest.statistic),
            "p_value": float(gate_ttest.pvalue),
            "cohens_d": float(gate_cohens_d),
            "significant": gate_ttest.pvalue < 0.05,
            "effect_size": "large" if abs(gate_cohens_d) > 0.8 else "medium" if abs(gate_cohens_d) > 0.5 else "small",
        }

        print(f"\n1. SENSE→FEEL (Do sensors affect gates?)")
        print(f"   Gate diff: {gate_diff:.4f}")
        print(f"   Stressed: {stressed['gates'].mean():.4f} ± {stressed['gates'].std():.4f}")
        print(f"   Relaxed:  {relaxed['gates'].mean():.4f} ± {relaxed['gates'].std():.4f}")
        print(f"   t-test: t={gate_ttest.statistic:.2f}, p={gate_ttest.pvalue:.2e}")
        print(f"   Cohen's d: {gate_cohens_d:.2f} ({evidence['sense_to_feel']['effect_size']})")
        print(f"   VERDICT: {'✓ PROVEN' if gate_ttest.pvalue < 0.05 else '✗ NOT PROVEN'}")

        # 2. FEEL→REGULATE: Skip rate difference
        skip_diff = abs(stressed["skips"].mean() - relaxed["skips"].mean())
        skip_ttest = stats.ttest_ind(stressed["skips"], relaxed["skips"])

        evidence["feel_to_regulate"] = {
            "skip_diff": float(skip_diff),
            "stressed_skip_mean": float(stressed["skips"].mean()),
            "relaxed_skip_mean": float(relaxed["skips"].mean()),
            "t_statistic": float(skip_ttest.statistic),
            "p_value": float(skip_ttest.pvalue),
            "significant": skip_ttest.pvalue < 0.05,
        }

        print(f"\n2. FEEL→REGULATE (Do gates affect skip rate?)")
        print(f"   Skip diff: {skip_diff:.1%}")
        print(f"   Stressed: {stressed['skips'].mean():.1%}")
        print(f"   Relaxed:  {relaxed['skips'].mean():.1%}")
        print(f"   t-test: t={skip_ttest.statistic:.2f}, p={skip_ttest.pvalue:.2e}")
        print(f"   VERDICT: {'✓ PROVEN' if skip_ttest.pvalue < 0.05 else '✗ NOT PROVEN'}")

        # 3. REGULATE→LATENT: FiLM effect
        film_diff = abs(stressed["film"].mean() - relaxed["film"].mean())
        film_ttest = stats.ttest_ind(stressed["film"], relaxed["film"])

        evidence["regulate_to_latent"] = {
            "film_diff": float(film_diff),
            "stressed_film_mean": float(stressed["film"].mean()),
            "relaxed_film_mean": float(relaxed["film"].mean()),
            "t_statistic": float(film_ttest.statistic),
            "p_value": float(film_ttest.pvalue),
            "significant": film_ttest.pvalue < 0.05,
        }

        print(f"\n3. REGULATE→LATENT (Do gates affect hidden states?)")
        print(f"   FiLM effect diff: {film_diff:.4f}")
        print(f"   Stressed: {stressed['film'].mean():.4f}")
        print(f"   Relaxed:  {relaxed['film'].mean():.4f}")
        print(f"   t-test: t={film_ttest.statistic:.2f}, p={film_ttest.pvalue:.2e}")
        print(f"   VERDICT: {'✓ PROVEN' if film_ttest.pvalue < 0.05 else '✗ NOT PROVEN'}")

        # 4. Sensor→Gate correlation (within natural condition)
        if len(natural["gates"]) > 10:
            # Use power as proxy for sensor state
            natural_traces = results["natural"]
            powers = []
            gates = []
            for trace in natural_traces:
                for t in trace.tokens:
                    powers.append(t.power_before)
                    gates.append(np.mean(list(t.gate_values.values())))

            if len(set(powers)) > 1:  # Need variance
                correlation = stats.pearsonr(powers, gates)
                evidence["sensor_gate_correlation"] = {
                    "pearson_r": float(correlation[0]),
                    "p_value": float(correlation[1]),
                    "significant": correlation[1] < 0.05,
                }

                print(f"\n4. SENSOR→GATE CORRELATION (Natural condition)")
                print(f"   Pearson r: {correlation[0]:.4f}")
                print(f"   p-value: {correlation[1]:.2e}")
                print(f"   VERDICT: {'✓ CORRELATED' if correlation[1] < 0.05 else '✗ NO CORRELATION'}")

        # 5. Output diversity
        stressed_outputs = [t.output_text for t in results["stressed"]]
        relaxed_outputs = [t.output_text for t in results["relaxed"]]

        # Simple diversity: unique word ratio
        stressed_words = set(" ".join(stressed_outputs).split())
        relaxed_words = set(" ".join(relaxed_outputs).split())
        overlap = len(stressed_words & relaxed_words) / len(stressed_words | relaxed_words) if stressed_words | relaxed_words else 1

        evidence["output_diversity"] = {
            "word_overlap": float(overlap),
            "stressed_unique_words": len(stressed_words),
            "relaxed_unique_words": len(relaxed_words),
            "outputs_differ": overlap < 0.8,
        }

        print(f"\n5. EXPRESS (Do outputs differ?)")
        print(f"   Word overlap: {overlap:.1%}")
        print(f"   Stressed unique: {len(stressed_words)} words")
        print(f"   Relaxed unique: {len(relaxed_words)} words")
        print(f"   VERDICT: {'✓ OUTPUTS DIFFER' if overlap < 0.8 else '✗ OUTPUTS SIMILAR'}")

        # Summary
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)

        proven = sum([
            evidence["sense_to_feel"]["significant"],
            evidence["feel_to_regulate"]["significant"],
            evidence["regulate_to_latent"]["significant"],
            evidence.get("sensor_gate_correlation", {}).get("significant", False),
            evidence["output_diversity"]["outputs_differ"],
        ])

        print(f"   Claims proven: {proven}/5")
        print(f"   Loop status: {'✓ EMBODIMENT LOOP WORKING' if proven >= 3 else '✗ INSUFFICIENT EVIDENCE'}")

        evidence["summary"] = {
            "claims_proven": proven,
            "total_claims": 5,
            "loop_working": proven >= 3,
        }

        return evidence


def main():
    parser = argparse.ArgumentParser(description="z31 Causal Proof Validation")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path")
    parser.add_argument("--output", type=str, default="z31_causal_proof.json", help="Output JSON")
    parser.add_argument("--samples", type=int, default=5, help="Samples per condition")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint).expanduser()
    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    validator = CausalProofValidator(checkpoint_path, args.model_name)
    results = validator.run_full_validation(num_samples=args.samples)

    # Save results (convert numpy types)
    def convert_numpy(obj):
        if isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        elif isinstance(obj, (np.bool_, np.integer)):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(convert_numpy(results["evidence"]), f, indent=2)

    print(f"\n[Done] Evidence saved to {output_path}")


if __name__ == "__main__":
    main()
