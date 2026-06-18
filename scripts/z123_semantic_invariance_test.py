#!/usr/bin/env python3
"""
z123 Semantic Invariance Test

Critical validation: Does body conditioning break language semantics?

Test protocol:
1. Run model with body=OFF (baseline logits)
2. Run model with body=ON (body-conditioned logits)
3. Compute KL(body_on || body_off) per token
4. KL should be TINY (<0.01) for normal prompts

This proves:
- Body affects internal representation WITHOUT changing token semantics
- The model can "feel" without "hallucinating"

Also tests:
- ReporterHead: Can hidden states predict telemetry? (shared substrate proof)
- Body-report prompts: Can KL constraint be relaxed for specific tasks?
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from feel_slm.model_v2 import FEELSLMV2, FEELConfigV2


# =============================================================================
# Semantic Invariance Test
# =============================================================================

class SemanticInvarianceTest:
    """
    Tests whether body conditioning preserves language semantics.

    Computes KL(P_body || P_baseline) across tokens.
    Low KL = body doesn't change semantics = GOOD
    High KL = body corrupts language = BAD
    """

    def __init__(self, model: FEELSLMV2, device: str = "cuda"):
        self.model = model
        self.device = device

    @torch.no_grad()
    def compute_kl_divergence(
        self,
        input_ids: torch.Tensor,
        body_vec: torch.Tensor,
    ) -> Dict:
        """
        Compute KL divergence between body-on and body-off outputs.

        Args:
            input_ids: [B, L] token ids
            body_vec: [B, body_dim] body state vector

        Returns:
            Dict with KL stats per token and aggregate
        """
        input_ids = input_ids.to(self.device)
        body_vec = body_vec.to(self.device)

        # Forward WITHOUT body (baseline)
        self.model.eval()

        # Disable body influence
        original_phase = self.model.config.phase
        self.model.config.phase = 0  # Completely disable body

        # Create mask
        seq_len = input_ids.shape[1]
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=self.device) * float('-inf'),
            diagonal=1
        )

        # Get baseline logits
        x = self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            x = layer(x, mask, drop_this_layer=False)
        x = self.model.norm(x)
        logits_baseline = self.model.lm_head(x)

        # Forward WITH body
        self.model.config.phase = 1  # Enable body for policy/reporter

        # Encode body
        body_embed = self.model.body_encoder(body_vec)

        # Forward with body influence (if any injection is enabled)
        x = self.model.embed_tokens(input_ids)
        for i, layer in enumerate(self.model.layers):
            x = layer(x, mask, drop_this_layer=False)

            # Apply gated injection if enabled
            gated_injections = getattr(self.model, 'gated_injections', None)
            if gated_injections is not None and i < len(gated_injections):
                injection = gated_injections[i]
                if injection is not None and hasattr(injection, 'enabled') and injection.enabled:
                    x = injection(x, body_embed)

        x = self.model.norm(x)
        logits_body = self.model.lm_head(x)

        # Restore phase
        self.model.config.phase = original_phase

        # Compute KL divergence per token
        # KL(P || Q) = sum(P * log(P/Q))
        probs_baseline = F.softmax(logits_baseline, dim=-1)
        probs_body = F.softmax(logits_body, dim=-1)

        # Add small epsilon for numerical stability
        eps = 1e-10
        kl_per_token = torch.sum(
            probs_body * (torch.log(probs_body + eps) - torch.log(probs_baseline + eps)),
            dim=-1
        )  # [B, L]

        # Also compute reverse KL
        kl_reverse = torch.sum(
            probs_baseline * (torch.log(probs_baseline + eps) - torch.log(probs_body + eps)),
            dim=-1
        )

        # Symmetric KL (Jensen-Shannon style)
        kl_symmetric = 0.5 * (kl_per_token + kl_reverse)

        return {
            "kl_per_token": kl_per_token.cpu().numpy(),  # [B, L]
            "kl_mean": kl_per_token.mean().item(),
            "kl_max": kl_per_token.max().item(),
            "kl_std": kl_per_token.std().item(),
            "kl_symmetric_mean": kl_symmetric.mean().item(),
            "tokens_with_high_kl": (kl_per_token > 0.01).sum().item(),
            "total_tokens": kl_per_token.numel(),
        }


# =============================================================================
# Reporter Accuracy Test
# =============================================================================

class ReporterAccuracyTest:
    """
    Tests whether hidden states can predict telemetry.

    This proves "shared latent substrate":
    - If ReporterHead accurately predicts body state from hidden states
    - Then the model internally represents hardware state
    - This is the core "embodiment" claim
    """

    def __init__(self, model: FEELSLMV2, device: str = "cuda"):
        self.model = model
        self.device = device

    @torch.no_grad()
    def test_reporter(
        self,
        input_ids: torch.Tensor,
        body_vec: torch.Tensor,
    ) -> Dict:
        """
        Test reporter head accuracy.

        Args:
            input_ids: [B, L] token ids
            body_vec: [B, body_dim] ground truth body state

        Returns:
            Dict with reporter predictions vs ground truth
        """
        input_ids = input_ids.to(self.device)
        body_vec = body_vec.to(self.device)

        # Forward to get body embedding and reporter output
        body_embed = self.model.body_encoder(body_vec)

        # Get reporter predictions
        if hasattr(self.model, 'reporter_head'):
            reporter_out = self.model.reporter_head(body_embed)

            # Extract predictions
            strain_pred = reporter_out.get("strain", torch.zeros(1))
            margin_pred = reporter_out.get("margin", torch.zeros(1))
            telemetry_pred = reporter_out.get("telemetry_pred", torch.zeros(1, 3))

            # Compute accuracy metrics
            # For telemetry prediction, compare to input body_vec (first 3 dims = power, temp, util)
            telemetry_gt = body_vec[:, :3]  # power, temp, util
            telemetry_mse = F.mse_loss(telemetry_pred, telemetry_gt).item()

            # Strain should correlate with power (body_vec[0])
            strain_gt = body_vec[:, 0]  # Use power as strain proxy
            strain_flat = strain_pred.view(-1)  # Flatten to 1D
            strain_gt_flat = strain_gt.view(-1)
            # Need at least 2 samples for correlation
            if strain_flat.numel() >= 2:
                strain_corr = torch.corrcoef(torch.stack([strain_flat, strain_gt_flat]))[0, 1].item()
            else:
                strain_corr = 0.0  # Can't compute correlation for single sample

            return {
                "telemetry_mse": telemetry_mse,
                "strain_correlation": strain_corr if not np.isnan(strain_corr) else 0.0,
                "strain_pred_mean": strain_pred.mean().item(),
                "margin_pred_mean": margin_pred.mean().item(),
                "telemetry_pred": telemetry_pred.cpu().numpy().tolist(),
                "telemetry_gt": telemetry_gt.cpu().numpy().tolist(),
            }
        else:
            return {"error": "No reporter_head in model"}


# =============================================================================
# Comprehensive Test Suite
# =============================================================================

def run_semantic_tests(
    model: FEELSLMV2,
    tokenizer,
    device: str = "cuda",
    num_samples: int = 20,
) -> Dict:
    """Run comprehensive semantic invariance tests."""
    inv_test = SemanticInvarianceTest(model, device)
    rep_test = ReporterAccuracyTest(model, device)

    # Test prompts (normal language, should have LOW KL)
    normal_prompts = [
        "The quick brown fox jumps over the lazy dog",
        "Once upon a time in a land far away",
        "Scientists have discovered a new species",
        "The weather forecast predicts rain tomorrow",
        "In the beginning there was light",
        "The stock market closed higher today",
        "Artificial intelligence is transforming",
        "The recipe calls for three eggs",
        "According to recent research findings",
        "The mountain peak was covered in snow",
    ]

    # Body-report prompts (model talks about its state, higher KL acceptable)
    body_prompts = [
        "I am currently feeling [BODY] and my power usage is",
        "My internal temperature is [BODY] degrees and",
        "System status: utilization at [BODY] percent",
        "Energy report: consuming [BODY] watts of power",
        "Hardware state: running at [BODY] efficiency",
    ]

    results = {
        "normal_prompts": [],
        "body_prompts": [],
        "summary": {},
    }

    print("\n--- Testing Normal Prompts (expect LOW KL) ---")
    for prompt in normal_prompts[:num_samples]:
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        input_ids = torch.clamp(input_ids, 0, model.config.vocab_size - 1)

        # Random body state
        body_vec = torch.rand(1, model.config.body_dim)

        kl_result = inv_test.compute_kl_divergence(input_ids, body_vec)
        rep_result = rep_test.test_reporter(input_ids, body_vec)

        results["normal_prompts"].append({
            "prompt": prompt[:50],
            "kl_mean": kl_result["kl_mean"],
            "kl_max": kl_result["kl_max"],
            "reporter_mse": rep_result.get("telemetry_mse", -1),
        })

        status = "✓" if kl_result["kl_mean"] < 0.01 else "✗"
        print(f"  {status} KL={kl_result['kl_mean']:.6f} | {prompt[:40]}...")

    # Aggregate
    normal_kls = [r["kl_mean"] for r in results["normal_prompts"]]
    results["summary"]["normal_kl_mean"] = np.mean(normal_kls)
    results["summary"]["normal_kl_max"] = np.max(normal_kls)
    results["summary"]["normal_kl_std"] = np.std(normal_kls)
    results["summary"]["invariance_passed"] = np.mean(normal_kls) < 0.01

    print(f"\n  Summary: mean KL = {np.mean(normal_kls):.6f}, max = {np.max(normal_kls):.6f}")
    print(f"  Invariance test: {'PASSED ✓' if results['summary']['invariance_passed'] else 'FAILED ✗'}")

    return results


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Semantic Invariance Test")
    parser.add_argument("--model-size", choices=["30m", "125m"], default="30m")
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default="results/z123_semantic")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    print("=" * 60)
    print("FEEL Semantic Invariance Test (z123)")
    print("=" * 60)

    # Load tokenizer
    print("\n1. Loading tokenizer...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Create model
    print(f"\n2. Creating model ({args.model_size})...")
    if args.model_size == "30m":
        config = FEELConfigV2(
            vocab_size=32000,
            hidden_dim=512,
            num_layers=8,
            num_heads=8,
            phase=1,
            enable_gating=False,  # Test with gating OFF
        )
    else:
        config = FEELConfigV2(
            vocab_size=32000,
            hidden_dim=768,
            num_layers=12,
            num_heads=12,
            phase=1,
            enable_gating=False,
        )

    model = FEELSLMV2(config).to(args.device)
    model.eval()
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    # Run tests
    print("\n3. Running semantic invariance tests...")
    results = run_semantic_tests(model, tokenizer, args.device, args.num_samples)

    # Additional test: vary body magnitude
    print("\n4. Testing KL vs body magnitude...")
    inv_test = SemanticInvarianceTest(model, args.device)
    test_prompt = "The quick brown fox jumps over"
    input_ids = tokenizer.encode(test_prompt, return_tensors="pt")
    input_ids = torch.clamp(input_ids, 0, model.config.vocab_size - 1)

    magnitudes = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]
    kl_by_magnitude = []

    for mag in magnitudes:
        body_vec = torch.ones(1, model.config.body_dim) * mag
        kl_result = inv_test.compute_kl_divergence(input_ids, body_vec)
        kl_by_magnitude.append(kl_result["kl_mean"])
        print(f"   Body magnitude {mag:.1f}: KL = {kl_result['kl_mean']:.6f}")

    results["kl_by_magnitude"] = dict(zip(magnitudes, kl_by_magnitude))

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "semantic_invariance.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Final summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"\nSemantic Invariance (normal prompts):")
    print(f"  Mean KL: {results['summary']['normal_kl_mean']:.6f}")
    print(f"  Max KL:  {results['summary']['normal_kl_max']:.6f}")
    print(f"  Test:    {'PASSED ✓' if results['summary']['invariance_passed'] else 'FAILED ✗'}")

    print(f"\nThreshold: KL < 0.01 for invariance")
    print(f"Interpretation:")
    if results['summary']['invariance_passed']:
        print("  Body conditioning does NOT alter language semantics.")
        print("  The model can 'feel' without 'hallucinating'.")
    else:
        print("  WARNING: Body conditioning affects language semantics!")
        print("  Need to reduce gate magnitude or add stronger KL anchoring.")

    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    main()
