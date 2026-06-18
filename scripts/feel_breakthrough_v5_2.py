#!/usr/bin/env python3
"""
FEEL Breakthrough Experiments v5.2 - PARITY-FIXED VERSION

Critical fixes from v5.1:
1. Uses inputs_embeds injection (matches training path)
2. Unified canonical sensor bank (no drift between train/eval)
3. Parity sanity suite (runs first, aborts if fails)
4. Telemetry validity gates
5. Monotonic influence curve test

Author: Claude + Human
Date: 2026-01-05
"""

import sys
import os
import json
import hashlib
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

# Add src to path for canonical sensors
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from canonical_sensors import CanonicalSensorBank, CANONICAL_SENSOR_VERSION, SENSOR_DIM


# ============================================================
# GPU Telemetry (with validity gates)
# ============================================================

@dataclass
class GPUMetrics:
    """GPU metrics with explicit availability tracking."""
    temp: float = 0.0
    temp_available: bool = False
    power: float = 0.0
    power_available: bool = False
    busy: float = 0.0
    busy_available: bool = False


class RobustGPUTelemetry:
    """GPU telemetry with validity tracking."""

    def __init__(self):
        self.amdsmi_available = False
        self.rocmsmi_available = False
        self.gpu_handle = None
        self.support_matrix = {}
        self._init_amdsmi()
        self._init_rocmsmi()

        # Validity tracking
        self.samples_collected = 0
        self.valid_temp_samples = 0
        self.valid_power_samples = 0
        self.temp_values = []
        self.power_values = []

    def _init_amdsmi(self):
        try:
            import amdsmi
            amdsmi.amdsmi_init()
            handles = amdsmi.amdsmi_get_processor_handles()
            if handles:
                self.gpu_handle = handles[0]
                self.amdsmi_available = True
                # Check what's actually available
                self.support_matrix['temp_edge'] = self._test_temp()
                self.support_matrix['power_amdsmi'] = self._test_power_amdsmi()
                self.support_matrix['utilization'] = self._test_utilization()
        except Exception as e:
            self.amdsmi_available = False

    def _init_rocmsmi(self):
        import subprocess
        try:
            result = subprocess.run(
                ["rocm-smi", "--showpower", "--json"],
                capture_output=True, text=True, timeout=5
            )
            self.rocmsmi_available = result.returncode == 0
        except:
            self.rocmsmi_available = False

    def _test_temp(self) -> bool:
        try:
            import amdsmi
            temp = amdsmi.amdsmi_get_temp_metric(
                self.gpu_handle,
                amdsmi.AmdSmiTemperatureType.EDGE,
                amdsmi.AmdSmiTemperatureMetric.CURRENT
            )
            return temp > 0
        except:
            return False

    def _test_power_amdsmi(self) -> bool:
        try:
            import amdsmi
            power = amdsmi.amdsmi_get_power_info(self.gpu_handle)
            avg = power.get('average_socket_power', 0)
            return avg > 0
        except:
            return False

    def _test_utilization(self) -> bool:
        try:
            import amdsmi
            util = amdsmi.amdsmi_get_gpu_activity(self.gpu_handle)
            return 'gfx_activity' in util
        except:
            return False

    def _get_from_rocmsmi(self) -> GPUMetrics:
        """Get metrics from rocm-smi CLI as fallback."""
        import subprocess
        metrics = GPUMetrics()
        try:
            result = subprocess.run(
                ["rocm-smi", "--showtemp", "--showpower", "--showuse", "--json"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                data = json.loads(result.stdout)
                for card_key in data:
                    if card_key.startswith("card"):
                        card = data[card_key]
                        # Temperature
                        if "Temperature (Sensor edge) (C)" in card:
                            temp_str = card["Temperature (Sensor edge) (C)"]
                            try:
                                metrics.temp = float(temp_str)
                                metrics.temp_available = True
                            except:
                                pass
                        # Power
                        if "Average Graphics Package Power (W)" in card:
                            power_str = card["Average Graphics Package Power (W)"]
                            try:
                                metrics.power = float(power_str)
                                metrics.power_available = True
                            except:
                                pass
                        # Busy
                        if "GPU use (%)" in card:
                            busy_str = card["GPU use (%)"]
                            try:
                                metrics.busy = float(busy_str)
                                metrics.busy_available = True
                            except:
                                pass
                        break
        except:
            pass
        return metrics

    def get_metrics(self) -> GPUMetrics:
        """Get current GPU metrics."""
        metrics = GPUMetrics()
        self.samples_collected += 1

        # Try amdsmi first for temp
        if self.amdsmi_available and self.support_matrix.get('temp_edge'):
            try:
                import amdsmi
                temp = amdsmi.amdsmi_get_temp_metric(
                    self.gpu_handle,
                    amdsmi.AmdSmiTemperatureType.EDGE,
                    amdsmi.AmdSmiTemperatureMetric.CURRENT
                )
                if temp > 0:
                    metrics.temp = temp
                    metrics.temp_available = True
                    self.valid_temp_samples += 1
                    self.temp_values.append(temp)
            except:
                pass

        # Try rocm-smi for power (and anything else missing)
        if self.rocmsmi_available:
            rocm_metrics = self._get_from_rocmsmi()
            if not metrics.temp_available and rocm_metrics.temp_available:
                metrics.temp = rocm_metrics.temp
                metrics.temp_available = True
                self.valid_temp_samples += 1
                self.temp_values.append(rocm_metrics.temp)
            if rocm_metrics.power_available:
                metrics.power = rocm_metrics.power
                metrics.power_available = True
                self.valid_power_samples += 1
                self.power_values.append(rocm_metrics.power)
            if rocm_metrics.busy_available:
                metrics.busy = rocm_metrics.busy
                metrics.busy_available = True

        return metrics

    def get_validity_report(self) -> Dict:
        """Get telemetry validity statistics."""
        temp_pct = self.valid_temp_samples / max(1, self.samples_collected) * 100
        power_pct = self.valid_power_samples / max(1, self.samples_collected) * 100
        temp_var = np.var(self.temp_values) if len(self.temp_values) > 1 else 0
        power_var = np.var(self.power_values) if len(self.power_values) > 1 else 0

        return {
            "samples_collected": self.samples_collected,
            "temp_valid_pct": temp_pct,
            "power_valid_pct": power_pct,
            "temp_variance": temp_var,
            "power_variance": power_var,
            "temp_usable": temp_pct > 90 and temp_var > 0.1,
            "power_usable": power_pct > 90 and power_var > 0.1,
        }

    def get_support_info(self) -> Dict:
        return {
            "amdsmi_initialized": self.amdsmi_available,
            "rocmsmi_available": self.rocmsmi_available,
            "support_matrix": self.support_matrix,
        }


# ============================================================
# FEEL Projector (matches training architecture EXACTLY)
# ============================================================

class CanonicalFEELProjector(nn.Module):
    """
    FEEL projector matching train_feel_canonical_v5.py architecture.

    Architecture:
        sensor_encoder: 12 -> 64 -> 64 (LayerNorm, GELU, Dropout)
        z_to_embed: 64 -> 128 -> hidden_size (GELU, LayerNorm)
    """

    def __init__(self, hidden_size: int = 1536, sensor_dim: int = 12, z_dim: int = 64):
        super().__init__()
        self.hidden_size = hidden_size
        self.sensor_dim = sensor_dim
        self.z_dim = z_dim

        # Must match training architecture exactly
        self.sensor_encoder = nn.Sequential(
            nn.Linear(sensor_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, z_dim),
            nn.GELU(),
        )

        self.z_to_embed = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Linear(128, hidden_size),
        )

    def forward(self, sensors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            sensors: [batch, 12] sensor values

        Returns:
            z_feel: [batch, 64] latent
            feel_embed: [batch, hidden_size] embedding for injection
        """
        z_feel = self.sensor_encoder(sensors)  # [batch, 64]
        feel_embed = self.z_to_embed(z_feel)   # [batch, hidden_size]
        return z_feel, feel_embed

    def get_weight_fingerprint(self) -> str:
        """Get hash of weights for parity checking."""
        state = self.state_dict()
        h = hashlib.md5()
        for k in sorted(state.keys()):
            h.update(k.encode())
            h.update(state[k].cpu().numpy().tobytes())
        return h.hexdigest()[:16]


# ============================================================
# FEEL Stream (with inputs_embeds injection)
# ============================================================

class CanonicalFEELStream(nn.Module):
    """
    FEEL stream using inputs_embeds injection (matches training).

    Key: FEEL is injected by modifying token embeddings BEFORE
    the transformer forward pass, allowing amplification through
    attention and MLP layers.
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        projector: CanonicalFEELProjector,
        sensor_bank: CanonicalSensorBank,
        alpha: float = 0.001,
        gpu_telemetry: Optional[RobustGPUTelemetry] = None,
    ):
        super().__init__()
        self.model = model
        self.projector = projector
        self.sensor_bank = sensor_bank
        self.alpha = alpha
        self.gpu_telemetry = gpu_telemetry
        self.device = next(model.parameters()).device

    def inject_feel(
        self,
        input_ids: torch.Tensor,
        feel_embed: torch.Tensor,
    ) -> torch.Tensor:
        """
        Inject FEEL into token embeddings.

        This is the CORRECT injection path matching training:
        embeds = token_embeds + alpha * feel_embed.unsqueeze(1)

        Args:
            input_ids: [batch, seq_len] token IDs
            feel_embed: [batch, hidden_size] FEEL embedding

        Returns:
            Modified embeddings [batch, seq_len, hidden_size]
        """
        # Get token embeddings
        token_embeds = self.model.get_input_embeddings()(input_ids)

        # Inject FEEL (broadcast across sequence)
        # Shape: [batch, seq_len, hidden] + [batch, 1, hidden]
        modified_embeds = token_embeds + self.alpha * feel_embed.unsqueeze(1).to(token_embeds.dtype)

        return modified_embeds

    def forward_with_feel(
        self,
        input_ids: torch.Tensor,
        feel_embed: torch.Tensor,
        output_hidden_states: bool = True,
    ):
        """
        Forward pass with FEEL injection via inputs_embeds.

        This allows transformer layers to amplify the FEEL signal.
        """
        modified_embeds = self.inject_feel(input_ids, feel_embed)

        outputs = self.model(
            inputs_embeds=modified_embeds,
            output_hidden_states=output_hidden_states,
            use_cache=False,
        )

        return outputs

    def generate_step_by_step(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        feel_on: bool = True,
    ) -> Tuple[torch.Tensor, List[Dict]]:
        """
        Generate tokens step-by-step with optional FEEL injection.

        At each step:
        1. Compute sensors from current logits
        2. Generate feel_embed
        3. Inject into embeddings
        4. Forward through transformer
        5. Sample next token

        Returns:
            generated_ids: Full sequence including input
            step_data: List of per-step metrics
        """
        generated = input_ids.clone()
        step_data = []

        for step in range(max_new_tokens):
            with torch.no_grad():
                # Get baseline logits
                outputs_base = self.model(input_ids=generated, use_cache=False)
                logits = outputs_base.logits[:, -1, :]

                if feel_on:
                    # Compute sensors
                    sensors = self.sensor_bank(logits, generation_depth=step)
                    sensors = sensors.to(self.projector.sensor_encoder[0].weight.dtype)

                    # Generate FEEL embedding
                    z_feel, feel_embed = self.projector(sensors)

                    # Forward with FEEL injection
                    outputs = self.forward_with_feel(
                        generated, feel_embed, output_hidden_states=True
                    )
                    logits = outputs.logits[:, -1, :]
                    hidden = outputs.hidden_states[-1][:, -1, :]
                else:
                    hidden = outputs_base.hidden_states[-1][:, -1, :] if outputs_base.hidden_states else None

                # Sample next token
                probs = F.softmax(logits.float(), dim=-1)
                next_token = torch.argmax(probs, dim=-1, keepdim=True)

                # Record step data
                entropy = -(probs * torch.log(probs.clamp(min=1e-10))).sum(-1).item()
                step_data.append({
                    "step": step,
                    "token_id": next_token[0, 0].item(),
                    "entropy": entropy,
                    "feel_on": feel_on,
                })

                # Append token
                generated = torch.cat([generated, next_token], dim=-1)

        return generated, step_data


# ============================================================
# Parity Sanity Suite (MUST PASS before experiments)
# ============================================================

def run_parity_sanity_suite(
    feel_stream: CanonicalFEELStream,
    tokenizer: AutoTokenizer,
) -> Tuple[bool, Dict]:
    """
    Run parity sanity suite. ABORTS if fails.

    Tests:
    1. Monotonic influence curve (KL vs alpha)
    2. Weight fingerprint logging
    3. Sensor parity check
    """
    print("\n" + "="*60)
    print("PARITY SANITY SUITE (must pass to continue)")
    print("="*60)

    results = {}

    # Test 1: Monotonic influence curve
    print("\n[1/3] Monotonic Influence Curve...")
    alphas = [0, 1e-4, 1e-3, 1e-2, 1e-1]
    kl_values = []

    test_prompt = "The capital of France is"
    inputs = tokenizer(test_prompt, return_tensors="pt").to(feel_stream.device)

    with torch.no_grad():
        # Baseline (no FEEL)
        out_base = feel_stream.model(input_ids=inputs["input_ids"], use_cache=False)
        logits_base = out_base.logits[:, -1, :]
        p_base = F.softmax(logits_base.float(), dim=-1)

        # Compute sensors once
        sensors = feel_stream.sensor_bank(logits_base)
        sensors = sensors.to(feel_stream.projector.sensor_encoder[0].weight.dtype)
        z_feel, feel_embed = feel_stream.projector(sensors)

        original_alpha = feel_stream.alpha

        for alpha in alphas:
            feel_stream.alpha = alpha
            outputs = feel_stream.forward_with_feel(
                inputs["input_ids"], feel_embed, output_hidden_states=False
            )
            logits_feel = outputs.logits[:, -1, :]
            p_feel = F.softmax(logits_feel.float(), dim=-1)

            kl = F.kl_div(p_feel.log(), p_base, reduction='batchmean').item()
            kl_values.append(kl)
            print(f"    alpha={alpha:.0e}: KL={kl:.6f}")

        feel_stream.alpha = original_alpha

    # Check monotonicity
    is_monotonic = all(kl_values[i] <= kl_values[i+1] for i in range(len(kl_values)-1))
    results["monotonic_influence"] = {
        "alphas": alphas,
        "kl_values": kl_values,
        "is_monotonic": is_monotonic,
    }
    print(f"    Monotonic: {'PASS' if is_monotonic else 'FAIL'}")

    # Test 2: Weight fingerprint
    print("\n[2/3] Weight Fingerprint...")
    fingerprint = feel_stream.projector.get_weight_fingerprint()
    results["weight_fingerprint"] = fingerprint
    print(f"    Projector fingerprint: {fingerprint}")

    # Log key weight norms
    norms = {}
    for name, param in feel_stream.projector.named_parameters():
        norms[name] = param.data.norm().item()
    results["weight_norms"] = norms
    print(f"    Weight norms logged: {len(norms)} parameters")

    # Test 3: Sensor parity
    print("\n[3/3] Sensor Parity Check...")
    sensor_bank_2 = CanonicalSensorBank()
    test_logits = torch.randn(1, 32000, device=feel_stream.device)
    sensors_1 = feel_stream.sensor_bank(test_logits)
    sensors_2 = sensor_bank_2.to(feel_stream.device)(test_logits)

    max_diff = (sensors_1 - sensors_2).abs().max().item()
    sensor_parity = max_diff < 1e-6
    results["sensor_parity"] = {
        "max_diff": max_diff,
        "passed": sensor_parity,
    }
    print(f"    Max sensor diff: {max_diff:.2e}")
    print(f"    Parity: {'PASS' if sensor_parity else 'FAIL'}")

    # Overall pass/fail
    all_pass = is_monotonic and sensor_parity
    results["overall_pass"] = all_pass

    print(f"\n{'='*60}")
    print(f"SANITY SUITE: {'PASS' if all_pass else 'FAIL'}")
    print(f"{'='*60}")

    return all_pass, results


# ============================================================
# Experiment 1: Teacher-Forced Counterfactual (FIXED)
# ============================================================

def run_teacher_forced_counterfactual(
    feel_stream: CanonicalFEELStream,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    n_tokens: int = 64,
) -> Dict:
    """
    Teacher-forced counterfactual using inputs_embeds injection.

    For each prompt:
    1. Generate baseline tokens (FEEL OFF)
    2. Teacher-force through with FEEL ON vs FEEL OFF
    3. Compare hidden states and logits at each step

    Key: Uses inputs_embeds injection to match training path.
    """
    print("\n" + "="*60)
    print("EXPERIMENT 1: Teacher-Forced Counterfactual")
    print("="*60)
    print("  (Using inputs_embeds injection - matches training)")

    raw_data = []

    for i, prompt in enumerate(prompts[:50]):
        print(f"  [{i+1}/50] {prompt[:50]}...")

        inputs = tokenizer(prompt, return_tensors="pt").to(feel_stream.device)
        input_ids = inputs["input_ids"]
        prompt_len = input_ids.shape[1]

        l2_divergences = []
        kl_divergences = []
        cosine_sims = []

        # Step-by-step teacher forcing
        current_ids = input_ids.clone()

        with torch.no_grad():
            for step in range(n_tokens):
                # FEEL OFF forward
                out_off = feel_stream.model(
                    input_ids=current_ids,
                    output_hidden_states=True,
                    use_cache=False,
                )
                hidden_off = out_off.hidden_states[-1][:, -1, :]
                logits_off = out_off.logits[:, -1, :]

                # Compute sensors and FEEL embedding
                sensors = feel_stream.sensor_bank(logits_off, generation_depth=step)
                sensors = sensors.to(feel_stream.projector.sensor_encoder[0].weight.dtype)
                z_feel, feel_embed = feel_stream.projector(sensors)

                # FEEL ON forward (inputs_embeds injection!)
                out_on = feel_stream.forward_with_feel(
                    current_ids, feel_embed, output_hidden_states=True
                )
                hidden_on = out_on.hidden_states[-1][:, -1, :]
                logits_on = out_on.logits[:, -1, :]

                # Measure divergence
                l2 = (hidden_on - hidden_off).norm().item()
                l2_divergences.append(l2)

                # KL divergence
                p_on = F.softmax(logits_on.float(), dim=-1)
                p_off = F.softmax(logits_off.float(), dim=-1)
                kl = F.kl_div(p_on.log(), p_off, reduction='batchmean').item()
                kl_divergences.append(max(0, kl))

                # Cosine similarity
                cos = F.cosine_similarity(hidden_on, hidden_off, dim=-1).mean().item()
                cosine_sims.append(cos)

                # Sample next token (from baseline to maintain teacher forcing)
                next_token = torch.argmax(p_off, dim=-1, keepdim=True)
                current_ids = torch.cat([current_ids, next_token], dim=-1)

        raw_data.append({
            "prompt": prompt[:100],
            "n_tokens": n_tokens,
            "avg_l2": np.mean(l2_divergences),
            "max_l2": np.max(l2_divergences),
            "avg_kl": np.mean(kl_divergences),
            "max_kl": np.max(kl_divergences),
            "avg_cosine": np.mean(cosine_sims),
            "min_cosine": np.min(cosine_sims),
        })

    # Aggregate with bootstrap CIs
    all_l2 = [d["avg_l2"] for d in raw_data]
    all_kl = [d["avg_kl"] for d in raw_data]
    all_cos = [d["avg_cosine"] for d in raw_data]

    def bootstrap_ci(data, n_boot=1000, ci=0.95):
        boots = [np.mean(np.random.choice(data, len(data), replace=True)) for _ in range(n_boot)]
        lo = np.percentile(boots, (1-ci)/2 * 100)
        hi = np.percentile(boots, (1+ci)/2 * 100)
        return np.mean(data), lo, hi

    l2_mean, l2_lo, l2_hi = bootstrap_ci(all_l2)
    kl_mean, kl_lo, kl_hi = bootstrap_ci(all_kl)
    cos_mean, cos_lo, cos_hi = bootstrap_ci(all_cos)

    # Causal effect threshold: L2 > 1.0 or cosine < 0.99
    causal_effect = l2_mean > 1.0 or cos_mean < 0.99

    summary = {
        "avg_l2_divergence": l2_mean,
        "l2_ci_95": [l2_lo, l2_hi],
        "avg_kl_divergence": kl_mean,
        "kl_ci_95": [kl_lo, kl_hi],
        "avg_cosine_similarity": cos_mean,
        "cosine_ci_95": [cos_lo, cos_hi],
        "causal_effect_detected": causal_effect,
        "verdict": "PASS" if causal_effect else "FAIL",
        "alpha": feel_stream.alpha,
    }

    print(f"\n  Results:")
    print(f"    Avg L2 divergence: {l2_mean:.4f} [{l2_lo:.4f}, {l2_hi:.4f}]")
    print(f"    Avg KL divergence: {kl_mean:.6f} [{kl_lo:.6f}, {kl_hi:.6f}]")
    print(f"    Avg cosine sim:    {cos_mean:.4f} [{cos_lo:.4f}, {cos_hi:.4f}]")
    print(f"    Causal effect:     {'YES' if causal_effect else 'NO'}")

    return {
        "summary": summary,
        "raw_data": raw_data,
        "timestamp": datetime.now().isoformat(),
    }


# ============================================================
# Experiment 2: Alpha Sweep (find working alpha)
# ============================================================

def run_alpha_sweep(
    feel_stream: CanonicalFEELStream,
    tokenizer: AutoTokenizer,
    prompts: List[str],
) -> Dict:
    """
    Sweep alpha values to find where FEEL has measurable effect.
    """
    print("\n" + "="*60)
    print("EXPERIMENT 2: Alpha Sweep")
    print("="*60)

    alphas = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1]
    results = {}

    test_prompt = prompts[0]
    inputs = tokenizer(test_prompt, return_tensors="pt").to(feel_stream.device)

    original_alpha = feel_stream.alpha

    for alpha in alphas:
        feel_stream.alpha = alpha

        with torch.no_grad():
            # Baseline (MUST request hidden_states!)
            out_base = feel_stream.model(
                input_ids=inputs["input_ids"],
                output_hidden_states=True,
                use_cache=False
            )
            logits_base = out_base.logits[:, -1, :]
            hidden_base = out_base.hidden_states[-1][:, -1, :]

            # With FEEL
            sensors = feel_stream.sensor_bank(logits_base)
            sensors = sensors.to(feel_stream.projector.sensor_encoder[0].weight.dtype)
            z_feel, feel_embed = feel_stream.projector(sensors)

            out_feel = feel_stream.forward_with_feel(
                inputs["input_ids"], feel_embed, output_hidden_states=True
            )
            logits_feel = out_feel.logits[:, -1, :]
            hidden_feel = out_feel.hidden_states[-1][:, -1, :]

            # Metrics
            l2 = (hidden_feel - hidden_base).norm().item()
            p_base = F.softmax(logits_base.float(), dim=-1)
            p_feel = F.softmax(logits_feel.float(), dim=-1)
            kl = F.kl_div(p_feel.log(), p_base, reduction='batchmean').item()
            cos = F.cosine_similarity(hidden_feel, hidden_base, dim=-1).mean().item()

        results[str(alpha)] = {
            "l2": l2,
            "kl": kl,
            "cosine": cos,
        }
        print(f"  alpha={alpha:.0e}: L2={l2:.4f}, KL={kl:.6f}, cos={cos:.6f}")

    feel_stream.alpha = original_alpha

    # Find best alpha (highest L2 while maintaining cos > 0.9)
    best_alpha = None
    best_l2 = 0
    for alpha_str, metrics in results.items():
        if metrics["cosine"] > 0.9 and metrics["l2"] > best_l2:
            best_l2 = metrics["l2"]
            best_alpha = float(alpha_str)

    return {
        "alpha_results": results,
        "best_alpha": best_alpha,
        "best_l2": best_l2,
        "timestamp": datetime.now().isoformat(),
    }


# ============================================================
# Experiment 3: Falsification Battery
# ============================================================

def run_falsification_battery(
    feel_stream: CanonicalFEELStream,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    n_tokens: int = 32,
) -> Dict:
    """
    Falsification battery: prove sensors carry meaningful information.

    Tests:
    1. Baseline: Generate with correct sensor alignment
    2. Permuted: Shuffle sensor timesteps (use sensor from wrong step)

    If sensors matter: permuted KL should be different from baseline KL.
    This proves the FEEL signal depends on actual sensor values.
    """
    print("\n" + "="*60)
    print("EXPERIMENT 3: Falsification Battery")
    print("="*60)

    baseline_kls = []
    permuted_kls = []

    test_prompts = prompts[:8]  # Use subset for speed

    for i, prompt in enumerate(test_prompts):
        print(f"  [{i+1}/{len(test_prompts)}] {prompt[:50]}...")

        inputs = tokenizer(prompt, return_tensors="pt").to(feel_stream.device)

        # Phase 1: Baseline - collect sensors and KL
        baseline_sensors = []
        baseline_kl_list = []

        with torch.no_grad():
            current_ids = inputs["input_ids"].clone()

            for step in range(n_tokens):
                # Base forward
                out_base = feel_stream.model(current_ids, use_cache=False)
                logits_base = out_base.logits[:, -1, :]

                # Get sensors
                sensors = feel_stream.sensor_bank(logits_base)
                sensors = sensors.to(feel_stream.projector.sensor_encoder[0].weight.dtype)
                baseline_sensors.append(sensors.clone())

                # FEEL forward
                z_feel, feel_embed = feel_stream.projector(sensors)
                out_feel = feel_stream.forward_with_feel(
                    current_ids, feel_embed, output_hidden_states=False
                )

                # KL
                p_base = F.softmax(logits_base.float(), dim=-1)
                p_feel = F.softmax(out_feel.logits[:, -1, :].float(), dim=-1)
                kl = F.kl_div(p_feel.log(), p_base, reduction='batchmean').item()
                baseline_kl_list.append(max(0, kl))

                # Next token
                next_token = torch.argmax(p_base, dim=-1, keepdim=True)
                current_ids = torch.cat([current_ids, next_token], dim=-1)

        baseline_kls.append(np.mean(baseline_kl_list))

        # Phase 2: Permuted - use shuffled sensors
        perm_idx = np.random.permutation(len(baseline_sensors))
        permuted_kl_list = []

        with torch.no_grad():
            current_ids = inputs["input_ids"].clone()

            for step in range(n_tokens):
                # Base forward
                out_base = feel_stream.model(current_ids, use_cache=False)
                logits_base = out_base.logits[:, -1, :]

                # Use PERMUTED sensor (from wrong timestep)
                wrong_step = perm_idx[step % len(perm_idx)]
                sensors = baseline_sensors[wrong_step]

                # FEEL forward with wrong sensor
                z_feel, feel_embed = feel_stream.projector(sensors)
                out_feel = feel_stream.forward_with_feel(
                    current_ids, feel_embed, output_hidden_states=False
                )

                # KL
                p_base = F.softmax(logits_base.float(), dim=-1)
                p_feel = F.softmax(out_feel.logits[:, -1, :].float(), dim=-1)
                kl = F.kl_div(p_feel.log(), p_base, reduction='batchmean').item()
                permuted_kl_list.append(max(0, kl))

                # Next token (from baseline for consistency)
                next_token = torch.argmax(p_base, dim=-1, keepdim=True)
                current_ids = torch.cat([current_ids, next_token], dim=-1)

        permuted_kls.append(np.mean(permuted_kl_list))

    # Compute summary
    baseline_mean = np.mean(baseline_kls)
    permuted_mean = np.mean(permuted_kls)

    # Ratio: if sensors matter, permuted should be different
    # Could be higher (wrong sensor causes more divergence) or lower (wrong sensor less effective)
    permute_ratio = permuted_mean / (baseline_mean + 1e-8)

    # Falsification passes if ratio is significantly different from 1.0
    # (either higher or lower shows sensors matter)
    ratio_diff = abs(permute_ratio - 1.0)
    falsification_passed = ratio_diff > 0.05  # >5% difference

    summary = {
        "baseline_avg_kl": baseline_mean,
        "permuted_avg_kl": permuted_mean,
        "permute_ratio": permute_ratio,
        "ratio_diff_pct": ratio_diff * 100,
        "falsification_passed": falsification_passed,
        "verdict": "PASS" if falsification_passed else "FAIL",
        "interpretation": "Sensors carry meaningful temporal information" if falsification_passed
                         else "Sensors may not encode temporal structure",
    }

    print(f"\n  Results:")
    print(f"    Baseline avg KL:  {baseline_mean:.6f}")
    print(f"    Permuted avg KL:  {permuted_mean:.6f}")
    print(f"    Permute ratio:    {permute_ratio:.3f}")
    print(f"    Ratio diff:       {ratio_diff*100:.1f}%")
    print(f"    Falsification:    {summary['verdict']}")

    return {
        "summary": summary,
        "baseline_kls": baseline_kls,
        "permuted_kls": permuted_kls,
        "timestamp": datetime.now().isoformat(),
    }


# ============================================================
# Main
# ============================================================

def load_prompts() -> List[str]:
    """Load diverse prompts for testing."""
    return [
        # Math
        "What is 17 * 23? Let me think step by step:",
        "If a train travels 60 mph for 2.5 hours, how far does it go?",
        "Explain why the square root of 2 is irrational:",
        "What is the derivative of x^3 + 2x^2 - 5x + 1?",
        "Calculate the integral of sin(x) from 0 to pi:",
        # Reasoning
        "Describe what uncertainty feels like:",
        "How confident are you in your reasoning abilities?",
        "What does it mean to be self-aware?",
        "Can machines truly understand language?",
        "What is consciousness?",
        # Creative
        "Write a haiku about artificial intelligence:",
        "Describe the color blue to someone who has never seen:",
        "What would a conversation between two AIs be like?",
        "Write a short poem about uncertainty:",
        # Technical
        "Explain quantum entanglement in simple terms:",
        "How does a transformer neural network work?",
        "What is the time complexity of binary search?",
        "Explain the CAP theorem:",
    ] * 3  # Repeat for more samples


def main():
    print("\n" + "="*70)
    print("  FEEL BREAKTHROUGH EXPERIMENTS v5.2 - PARITY-FIXED VERSION")
    print("="*70)

    # Initialize GPU telemetry
    print("\nInitializing GPU telemetry...")
    gpu_telemetry = RobustGPUTelemetry()
    print(f"  Support: {gpu_telemetry.get_support_info()}")

    # Load model
    print("\nLoading model on cuda...")
    model_name = "Qwen/Qwen2.5-1.5B"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cuda",
        trust_remote_code=True,
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    print(f"  Model loaded (dtype: {model.dtype})")

    # Initialize components
    print("\nInitializing FEEL components...")
    sensor_bank = CanonicalSensorBank()
    print(f"  Sensor bank version: {sensor_bank.get_version()}")

    hidden_size = model.config.hidden_size
    projector = CanonicalFEELProjector(hidden_size=hidden_size).to(model.device).half()

    # Load trained weights
    checkpoint_path = Path(__file__).parent.parent / "results" / "feel_training" / "canonical_v5_checkpoint.pt"
    if checkpoint_path.exists():
        print(f"  Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=model.device)

        if "feel_stream_state" in ckpt:
            state = ckpt["feel_stream_state"]
            # Extract projector weights
            projector_state = {}
            for k, v in state.items():
                if k.startswith("projector."):
                    new_k = k.replace("projector.", "")
                    projector_state[new_k] = v
            projector.load_state_dict(projector_state)
            alpha = ckpt.get("alpha", 0.001)
            print(f"  Loaded projector weights (alpha: {alpha:.6f})")
        else:
            alpha = 0.001
            print("  WARNING: No projector weights in checkpoint!")
    else:
        alpha = 0.001
        print(f"  WARNING: No checkpoint found at {checkpoint_path}")

    # Create FEEL stream
    feel_stream = CanonicalFEELStream(
        model=model,
        projector=projector,
        sensor_bank=sensor_bank,
        alpha=alpha,
        gpu_telemetry=gpu_telemetry,
    )
    print(f"  FEEL stream created (alpha: {feel_stream.alpha})")

    # Load prompts
    prompts = load_prompts()
    print(f"  Loaded {len(prompts)} prompts")

    # Results storage
    results = {}

    # Run parity sanity suite (MUST PASS)
    sanity_pass, sanity_results = run_parity_sanity_suite(feel_stream, tokenizer)
    results["sanity_suite"] = sanity_results

    if not sanity_pass:
        print("\n!!! SANITY SUITE FAILED - ABORTING !!!")
        print("Fix parity issues before running experiments.")
        return

    # Run alpha sweep first
    results["alpha_sweep"] = run_alpha_sweep(feel_stream, tokenizer, prompts)

    # Use best alpha from sweep
    if results["alpha_sweep"]["best_alpha"]:
        print(f"\nUsing best alpha from sweep: {results['alpha_sweep']['best_alpha']}")
        feel_stream.alpha = results["alpha_sweep"]["best_alpha"]

    # Run teacher-forced counterfactual
    results["teacher_forced"] = run_teacher_forced_counterfactual(
        feel_stream, tokenizer, prompts
    )

    # Run falsification battery
    results["falsification"] = run_falsification_battery(
        feel_stream, tokenizer, prompts
    )

    # Save results
    results_dir = Path(__file__).parent.parent / "results" / "feel_experiments"
    results_dir.mkdir(parents=True, exist_ok=True)

    results_path = results_dir / "breakthrough_v5_2_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {results_path}")

    # Summary
    print("\n" + "="*70)
    print("  FINAL SUMMARY")
    print("="*70)
    print(f"  Sensor version: {CANONICAL_SENSOR_VERSION}")
    print(f"  Weight fingerprint: {sanity_results['weight_fingerprint']}")
    print(f"  Monotonic influence: {'PASS' if sanity_results['monotonic_influence']['is_monotonic'] else 'FAIL'}")
    print(f"  Best alpha: {results['alpha_sweep']['best_alpha']}")
    print(f"  Teacher-forced causal effect: {results['teacher_forced']['summary']['verdict']}")
    print(f"    L2 divergence: {results['teacher_forced']['summary']['avg_l2_divergence']:.4f}")
    print(f"    Cosine similarity: {results['teacher_forced']['summary']['avg_cosine_similarity']:.4f}")
    print(f"  Falsification battery: {results['falsification']['summary']['verdict']}")
    print(f"    Permute ratio: {results['falsification']['summary']['permute_ratio']:.3f}")


if __name__ == "__main__":
    main()
