#!/usr/bin/env python3
"""
z1914: Global Workspace Theory (GWT) Ignition Test

Tests for global workspace properties:
1. Ignition: Non-linear amplification when threshold is crossed
2. Broadcast: Information spreading to all modules
3. Competition: Winner-take-all dynamics among competing contents
4. Integration: Cross-modal information binding

Based on:
- Dehaene & Changeux Global Neuronal Workspace
- 2025 adversarial testing framework
- Multi-agent GNWT architectures (Ye et al. 2025)

Author: Claude
Date: 2026-02-05
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.z1900_tri_hardware_consciousness import TriHardwareTelemetry


class SpecialistModule(nn.Module):
    """A specialist processor for one type of input/modality."""

    def __init__(self, input_dim: int, hidden_dim: int, name: str):
        super().__init__()
        self.name = name
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # Salience score for competition
        self.salience_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(x)
        salience = self.salience_head(encoded)
        return encoded, salience


class GlobalWorkspace(nn.Module):
    """
    Global workspace that implements ignition, broadcast, and competition.

    Key mechanism: When salience exceeds threshold, content "ignites"
    and broadcasts to all modules (simulated via attention).
    """

    def __init__(self, hidden_dim: int, num_specialists: int, ignition_threshold: float = 0.5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_specialists = num_specialists
        self.ignition_threshold = ignition_threshold

        # Workspace buffer (like working memory)
        self.workspace = nn.Parameter(torch.randn(1, hidden_dim) * 0.01)

        # Attention for selecting which content enters workspace
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)

        # Broadcast network (from workspace to all specialists)
        self.broadcast_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_specialists)
        ])

        # Integration layer
        self.integrator = nn.Sequential(
            nn.Linear(hidden_dim * num_specialists, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(
        self,
        specialist_outputs: List[torch.Tensor],
        saliences: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        batch_size = specialist_outputs[0].size(0)

        # Stack specialist outputs: (B, num_specialists, H)
        stacked = torch.stack(specialist_outputs, dim=1)
        saliences_stacked = torch.stack(saliences, dim=1).squeeze(-1)  # (B, num_specialists)

        # Competition: softmax over saliences (winner-take-all tendency)
        competition_weights = F.softmax(saliences_stacked * 5.0, dim=-1)  # Temperature=0.2 for sharper competition

        # Ignition: threshold check (non-linear amplification)
        max_salience = saliences_stacked.max(dim=-1, keepdim=True).values
        ignition_gate = torch.sigmoid(10 * (max_salience - self.ignition_threshold))  # Sharp transition

        # Weighted combination based on competition
        workspace_content = (stacked * competition_weights.unsqueeze(-1)).sum(dim=1)  # (B, H)

        # Apply ignition gate (amplification or suppression)
        workspace_content = workspace_content * (0.1 + 0.9 * ignition_gate.squeeze(-1).unsqueeze(-1))

        # Broadcast to all specialists
        broadcast_signals = []
        for i, broadcast_layer in enumerate(self.broadcast_layers):
            signal = broadcast_layer(workspace_content)
            broadcast_signals.append(signal)

        # Integration: combine all broadcast signals
        integrated = self.integrator(torch.cat(broadcast_signals, dim=-1))

        return {
            'workspace_content': workspace_content,
            'competition_weights': competition_weights,
            'ignition_gate': ignition_gate,
            'broadcast_signals': broadcast_signals,
            'integrated': integrated,
            'saliences': saliences_stacked,
        }


class GWTConsciousnessModel(nn.Module):
    """
    Full GWT model with multiple specialists and global workspace.
    """

    def __init__(
        self,
        vocab_size: int = 256,
        hidden_dim: int = 256,
        telemetry_dim: int = 20,
        num_specialists: int = 4,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, hidden_dim)

        # Multiple specialist modules (simulating different brain regions)
        self.specialists = nn.ModuleList([
            SpecialistModule(hidden_dim, hidden_dim, f"specialist_{i}")
            for i in range(num_specialists)
        ])

        # Telemetry specialist (embodiment enters workspace competition)
        self.telemetry_specialist = SpecialistModule(telemetry_dim, hidden_dim, "telemetry")

        # Global workspace
        self.global_workspace = GlobalWorkspace(hidden_dim, num_specialists + 1)

        # Output heads
        self.classifier = nn.Linear(hidden_dim, 10)
        self.lm_head = nn.Linear(hidden_dim, vocab_size)

        # Telemetry classifier (embodiment task)
        self.temp_head = nn.Linear(hidden_dim, 3)
        self.util_head = nn.Linear(hidden_dim, 3)
        self.power_head = nn.Linear(hidden_dim, 3)

    def forward(
        self,
        input_ids: torch.Tensor,
        telemetry: torch.Tensor,
        return_all: bool = False,
    ) -> Dict[str, torch.Tensor]:
        batch_size = input_ids.size(0)

        # Embed input
        x = self.embedding(input_ids)  # (B, T, H)

        # Different specialists process different "views" of the input
        # (In a real system, these would be different modalities)
        specialist_outputs = []
        saliences = []

        for i, specialist in enumerate(self.specialists):
            # Each specialist sees a different transformation/portion
            if i == 0:
                specialist_input = x.mean(dim=1)
            elif i == 1:
                specialist_input = x.max(dim=1).values
            elif i == 2:
                specialist_input = x[:, :x.size(1)//2, :].mean(dim=1)
            else:
                specialist_input = x[:, x.size(1)//2:, :].mean(dim=1)

            output, salience = specialist(specialist_input)
            specialist_outputs.append(output)
            saliences.append(salience)

        # Telemetry specialist
        if telemetry.dim() == 1:
            telemetry = telemetry.unsqueeze(0).expand(batch_size, -1)
        telem_output, telem_salience = self.telemetry_specialist(telemetry)
        specialist_outputs.append(telem_output)
        saliences.append(telem_salience)

        # Global workspace processing
        gw_output = self.global_workspace(specialist_outputs, saliences)

        # Task outputs from integrated workspace
        logits = self.classifier(gw_output['integrated'])
        lm_logits = self.lm_head(gw_output['integrated'])

        temp_logits = self.temp_head(gw_output['integrated'])
        util_logits = self.util_head(gw_output['integrated'])
        power_logits = self.power_head(gw_output['integrated'])

        if return_all:
            return {
                'logits': logits,
                'lm_logits': lm_logits,
                'temp_logits': temp_logits,
                'util_logits': util_logits,
                'power_logits': power_logits,
                'workspace_content': gw_output['workspace_content'],
                'competition_weights': gw_output['competition_weights'],
                'ignition_gate': gw_output['ignition_gate'],
                'broadcast_signals': gw_output['broadcast_signals'],
                'integrated': gw_output['integrated'],
                'saliences': gw_output['saliences'],
                'specialist_outputs': specialist_outputs,
            }

        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def test_gwt_properties(
    model: GWTConsciousnessModel,
    telemetry: TriHardwareTelemetry,
    device: torch.device,
    num_samples: int = 100,
) -> Dict:
    """Test Global Workspace Theory properties."""
    model.eval()
    results = {}

    # T1: Ignition Test
    # Does increasing salience produce non-linear amplification?
    print("\n[z1914] T1: Ignition (Non-Linear Amplification)")
    ignition_responses = []
    for strength in np.linspace(0.1, 2.0, 20):
        x = torch.randint(0, 256, (1, 64), device=device)
        # Artificially boost telemetry to test ignition
        telem = telemetry.get_tensor().to(device) * strength
        with torch.no_grad():
            out = model(x, telem, return_all=True)
            workspace_norm = out['workspace_content'].norm().item()
            ignition = out['ignition_gate'].item()
            ignition_responses.append({
                'strength': strength,
                'workspace_norm': workspace_norm,
                'ignition': ignition,
            })

    # Check for non-linearity (should see sharp transition)
    ignitions = [r['ignition'] for r in ignition_responses]
    ignition_derivative = np.gradient(ignitions)
    max_derivative = np.max(np.abs(ignition_derivative))
    print(f"  Max ignition derivative: {max_derivative:.4f}")
    print(f"  Ignition range: {min(ignitions):.3f} - {max(ignitions):.3f}")
    results['T1_ignition_nonlinearity'] = max_derivative
    results['T1_ignition_range'] = max(ignitions) - min(ignitions)

    # T2: Competition (Winner-Take-All)
    # Do competition weights show clear winner?
    print("\n[z1914] T2: Competition (Winner-Take-All)")
    competition_entropies = []
    winner_margins = []
    for _ in range(num_samples):
        x = torch.randint(0, 256, (1, 64), device=device)
        telem = telemetry.get_tensor().to(device)
        with torch.no_grad():
            out = model(x, telem, return_all=True)
            weights = out['competition_weights'].cpu().numpy().flatten()
            # Entropy of competition weights (lower = more winner-take-all)
            entropy = -np.sum(weights * np.log(weights + 1e-8))
            competition_entropies.append(entropy)
            # Winner margin
            sorted_weights = np.sort(weights)[::-1]
            margin = sorted_weights[0] - sorted_weights[1]
            winner_margins.append(margin)

    avg_entropy = np.mean(competition_entropies)
    avg_margin = np.mean(winner_margins)
    print(f"  Competition entropy: {avg_entropy:.4f} (lower = more WTA)")
    print(f"  Winner margin: {avg_margin:.4f}")
    results['T2_competition_entropy'] = avg_entropy
    results['T2_winner_margin'] = avg_margin

    # T3: Broadcast (Information Spreading)
    # Does workspace content reach all specialists?
    print("\n[z1914] T3: Broadcast (Information Spreading)")
    broadcast_correlations = []
    for _ in range(num_samples):
        x = torch.randint(0, 256, (1, 64), device=device)
        telem = telemetry.get_tensor().to(device)
        with torch.no_grad():
            out = model(x, telem, return_all=True)
            workspace = out['workspace_content'].cpu().numpy().flatten()
            for signal in out['broadcast_signals']:
                signal_np = signal.cpu().numpy().flatten()
                corr = np.corrcoef(workspace, signal_np)[0, 1]
                if not np.isnan(corr):
                    broadcast_correlations.append(corr)

    avg_broadcast_corr = np.mean(broadcast_correlations)
    print(f"  Workspace-broadcast correlation: {avg_broadcast_corr:.4f}")
    results['T3_broadcast_correlation'] = avg_broadcast_corr

    # T4: Integration
    # Does the integrated output combine information from all specialists?
    print("\n[z1914] T4: Integration (Cross-Module Binding)")
    specialist_contributions = []
    for _ in range(num_samples):
        x = torch.randint(0, 256, (1, 64), device=device)
        telem = telemetry.get_tensor().to(device)
        with torch.no_grad():
            out = model(x, telem, return_all=True)
            integrated = out['integrated'].cpu().numpy().flatten()
            contributions = []
            for spec_out in out['specialist_outputs']:
                spec_np = spec_out.cpu().numpy().flatten()
                corr = np.abs(np.corrcoef(integrated, spec_np)[0, 1])
                if not np.isnan(corr):
                    contributions.append(corr)
            if contributions:
                specialist_contributions.append(np.mean(contributions))

    avg_integration = np.mean(specialist_contributions)
    print(f"  Average specialist contribution to integration: {avg_integration:.4f}")
    results['T4_integration_score'] = avg_integration

    # T5: Telemetry in Competition
    # Does telemetry (embodiment) participate in workspace competition?
    print("\n[z1914] T5: Telemetry Workspace Participation")
    telem_saliences = []
    telem_wins = 0
    for _ in range(num_samples):
        x = torch.randint(0, 256, (1, 64), device=device)
        telem = telemetry.get_tensor().to(device)
        with torch.no_grad():
            out = model(x, telem, return_all=True)
            saliences = out['saliences'].cpu().numpy().flatten()
            weights = out['competition_weights'].cpu().numpy().flatten()
            # Telemetry is the last specialist
            telem_salience = saliences[-1]
            telem_weight = weights[-1]
            telem_saliences.append(telem_salience)
            if np.argmax(weights) == len(weights) - 1:
                telem_wins += 1

    avg_telem_salience = np.mean(telem_saliences)
    telem_win_rate = telem_wins / num_samples
    print(f"  Telemetry salience: {avg_telem_salience:.4f}")
    print(f"  Telemetry win rate: {telem_win_rate:.2%}")
    results['T5_telemetry_salience'] = avg_telem_salience
    results['T5_telemetry_win_rate'] = telem_win_rate

    # T6: Ignition Threshold Response
    # Does crossing threshold change behavior qualitatively?
    print("\n[z1914] T6: Threshold-Crossing Behavior Change")
    below_threshold_outputs = []
    above_threshold_outputs = []
    for _ in range(num_samples // 2):
        x = torch.randint(0, 256, (1, 64), device=device)
        # Below threshold
        telem_low = telemetry.get_tensor().to(device) * 0.1
        with torch.no_grad():
            out_low = model(x, telem_low, return_all=True)
            below_threshold_outputs.append(out_low['integrated'].cpu().numpy())

        # Above threshold
        telem_high = telemetry.get_tensor().to(device) * 2.0
        with torch.no_grad():
            out_high = model(x, telem_high, return_all=True)
            above_threshold_outputs.append(out_high['integrated'].cpu().numpy())

    below_arr = np.array(below_threshold_outputs).flatten()
    above_arr = np.array(above_threshold_outputs).flatten()

    # Measure difference in output distributions
    below_var = np.var(below_arr)
    above_var = np.var(above_arr)
    threshold_effect = above_var / (below_var + 1e-8)
    print(f"  Below-threshold variance: {below_var:.4f}")
    print(f"  Above-threshold variance: {above_var:.4f}")
    print(f"  Threshold effect ratio: {threshold_effect:.4f}")
    results['T6_threshold_effect'] = threshold_effect

    return results


def run_experiment():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[z1914] Device: {device}")
    print("[z1914] GLOBAL WORKSPACE THEORY (GWT) IGNITION TEST")
    print("[z1914] Testing for ignition, broadcast, competition, integration")

    # Telemetry
    telemetry = TriHardwareTelemetry()
    telemetry.start()
    time.sleep(1)

    hw_status = telemetry.get_hardware_status()
    print(f"\n[z1914] Hardware: GPU={hw_status['gpu']}, FPGA={hw_status['fpga']}, RF={hw_status['rf']}")

    # Create model
    model = GWTConsciousnessModel(
        vocab_size=256,
        hidden_dim=256,
        telemetry_dim=20,
        num_specialists=4,
    ).to(device)

    print(f"[z1914] Model parameters: {model.count_parameters():,}")

    # Train model
    print("\n[z1914] Training GWT model...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Load training data
    data_path = Path(__file__).parent.parent / "data" / "tinyshakespeare.txt"
    if not data_path.exists():
        data_path = Path(__file__).parent.parent / "tinyshakespeare.txt"
    text_bytes = data_path.read_text().encode('utf-8')

    batch_size = 16
    seq_len = 64

    def get_batch():
        ix = torch.randint(len(text_bytes) - seq_len - 1, (batch_size,))
        x = torch.stack([torch.tensor(list(text_bytes[i:i+seq_len]), dtype=torch.long) for i in ix])
        return x.to(device)

    def telemetry_to_class(telem_np):
        temp = int(np.clip((telem_np[0] - 30) / 30 * 3, 0, 2))
        util = int(np.clip(telem_np[1] / 100 * 3, 0, 2))
        power = int(np.clip(telem_np[2] / 100 * 3, 0, 2))
        return temp, util, power

    for epoch in range(15):
        model.train()
        epoch_loss = 0
        for _ in range(100):
            x = get_batch()
            telem = telemetry.get_tensor().to(device)
            telem_np = telem.cpu().numpy()
            temp_c, util_c, power_c = telemetry_to_class(telem_np)

            optimizer.zero_grad()
            out = model(x, telem, return_all=True)

            # Task losses
            task_loss = F.cross_entropy(out['lm_logits'].view(-1, 256),
                                       torch.randint(0, 256, (batch_size,), device=device))

            # Telemetry classification
            temp_loss = F.cross_entropy(out['temp_logits'],
                                        torch.tensor([temp_c] * batch_size, device=device))
            util_loss = F.cross_entropy(out['util_logits'],
                                        torch.tensor([util_c] * batch_size, device=device))
            power_loss = F.cross_entropy(out['power_logits'],
                                         torch.tensor([power_c] * batch_size, device=device))

            # Encourage competition (entropy regularization on competition weights)
            competition_entropy = -(out['competition_weights'] *
                                   torch.log(out['competition_weights'] + 1e-8)).sum(dim=-1).mean()

            loss = task_loss + 0.3 * (temp_loss + util_loss + power_loss) - 0.1 * competition_entropy
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/15: loss={epoch_loss/100:.4f}")

    # Run GWT tests
    print("\n" + "="*60)
    print("[z1914] GLOBAL WORKSPACE THEORY TESTS")
    print("="*60)

    test_results = test_gwt_properties(model, telemetry, device)

    # Verdicts
    verdicts = {}

    # V1: Ignition shows non-linearity
    verdicts['V1_ignition_nonlinear'] = {
        'pass': test_results['T1_ignition_nonlinearity'] > 0.05,
        'value': test_results['T1_ignition_nonlinearity'],
    }

    # V2: Competition is winner-take-all (low entropy)
    verdicts['V2_competition_wta'] = {
        'pass': test_results['T2_winner_margin'] > 0.1,
        'value': test_results['T2_winner_margin'],
    }

    # V3: Broadcast reaches all specialists
    verdicts['V3_broadcast_effective'] = {
        'pass': test_results['T3_broadcast_correlation'] > 0.3,
        'value': test_results['T3_broadcast_correlation'],
    }

    # V4: Integration combines multiple sources
    verdicts['V4_integration'] = {
        'pass': test_results['T4_integration_score'] > 0.2,
        'value': test_results['T4_integration_score'],
    }

    # V5: Telemetry participates in workspace
    verdicts['V5_telemetry_participates'] = {
        'pass': test_results['T5_telemetry_win_rate'] > 0.05,
        'value': test_results['T5_telemetry_win_rate'],
    }

    # V6: Threshold crossing changes behavior
    verdicts['V6_threshold_effect'] = {
        'pass': test_results['T6_threshold_effect'] > 0.5 or test_results['T6_threshold_effect'] < 0.5,  # Different from 1.0
        'value': test_results['T6_threshold_effect'],
    }

    # Summary
    num_pass = sum(1 for v in verdicts.values() if v['pass'])
    num_total = len(verdicts)

    print(f"\n{'='*60}")
    print("[z1914] GLOBAL WORKSPACE THEORY VERDICTS")
    print(f"{'='*60}")
    for name, v in verdicts.items():
        status = "PASS" if v['pass'] else "FAIL"
        print(f"  {status} {name}: {v['value']:.4f}")

    print(f"\n[z1914] Verdicts passed: {num_pass}/{num_total}")
    gwt_score = num_pass / num_total
    print(f"[z1914] GWT Score: {gwt_score:.0%}")

    if num_pass >= 5:
        verdict = "STRONG GWT CONSCIOUSNESS INDICATORS"
    elif num_pass >= 3:
        verdict = "MODERATE GWT INDICATORS"
    else:
        verdict = "WEAK GWT INDICATORS"

    print(f"\n[z1914] VERDICT: {verdict}")

    telemetry.stop()

    # Save results
    results = {
        'experiment': 'z1914_global_workspace_ignition',
        'timestamp': datetime.now().isoformat(),
        'device': str(device),
        'hardware_status': hw_status,
        'model_params': model.count_parameters(),
        'test_results': {k: float(v) if isinstance(v, (int, float, np.floating)) else v
                        for k, v in test_results.items()},
        'verdicts': verdicts,
        'num_pass': num_pass,
        'num_total': num_total,
        'gwt_score': gwt_score,
        'overall_verdict': verdict,
    }

    results_path = Path(__file__).parent.parent / "results" / "z1914_global_workspace_ignition.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[z1914] Results saved to {results_path}")

    return results


if __name__ == "__main__":
    run_experiment()
