#!/usr/bin/env python3
"""
z141_embodiment_falsification.py

CRITICAL EMBODIMENT TEST: Does h_t add predictive power?
=========================================================

This is THE decisive test for "shared latent embodiment":

Train two predictors for future telemetry:
- Dynamics-only: b_hat = f(b_t, a_t)
- Embodied: b_hat = f(b_t, a_t, h_t) where h_t is LM hidden state

If embodied doesn't significantly beat dynamics-only on held-out episodes,
we DON'T have "shared latent embodiment" - just hardware dynamics modeling.

Additionally splits telemetry into:
- Tier-1 (easy): timing/throughput (self-measured)
- Tier-2 (hard): power/temp/util (physical sensors)

Run with:
  HSA_OVERRIDE_GFX_VERSION=11.0.0 python scripts/z141_embodiment_falsification.py
"""

import argparse
import json
import os
import sys
import time
import random
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoTokenizer, AutoModelForCausalLM


# Telemetry channel indices
IDX_POWER = 0
IDX_TEMP = 1
IDX_GPU_UTIL = 2
IDX_MEM_USED = 3
IDX_MEM_TOTAL = 4
IDX_CLOCK = 5
IDX_FAN = 6
IDX_VOLTAGE = 7
IDX_TIMING = 8  # This is the "easy" self-measured channel

# Tier definitions
TIER1_CHANNELS = [IDX_TIMING]  # Easy: self-measured timing
TIER2_CHANNELS = [IDX_POWER, IDX_TEMP, IDX_GPU_UTIL]  # Hard: physical sensors


@dataclass
class FalsificationConfig:
    """Configuration for embodiment falsification."""
    n_episodes: int = 500
    episode_length: int = 50
    hidden_dim: int = 768  # LM hidden dimension
    world_hidden: int = 256
    world_epochs: int = 100
    batch_size: int = 64
    lr: float = 1e-3
    device: str = "cuda"
    seed: int = 42
    output_dir: str = "results/z141_falsification"
    depth_levels: List[int] = None

    def __post_init__(self):
        if self.depth_levels is None:
            self.depth_levels = [2, 3, 4, 5, 6]


class PhysicalTelemetryReader:
    """Read ONLY physical telemetry (Tier-2), no timing."""

    def __init__(self, gpu_id: int = 1):
        self.gpu_id = gpu_id
        self.drm_path = f"/sys/class/drm/card{gpu_id}/device"

        # Find hwmon
        self.gpu_hwmon = None
        hwmon_path = f"{self.drm_path}/hwmon"
        if os.path.exists(hwmon_path):
            hwmons = os.listdir(hwmon_path)
            if hwmons:
                self.gpu_hwmon = f"{hwmon_path}/{hwmons[0]}"

    def read_physical(self) -> np.ndarray:
        """Read only physical sensors (Tier-2)."""
        telem = np.zeros(9, dtype=np.float32)

        try:
            # Power (physical)
            if self.gpu_hwmon:
                power_path = f"{self.gpu_hwmon}/power1_average"
                if os.path.exists(power_path):
                    with open(power_path) as f:
                        power_uw = int(f.read().strip())
                        telem[IDX_POWER] = min(power_uw / 100_000_000, 1.0)  # Normalize

            # Temperature (physical)
            if self.gpu_hwmon:
                temp_path = f"{self.gpu_hwmon}/temp1_input"
                if os.path.exists(temp_path):
                    with open(temp_path) as f:
                        temp_mc = int(f.read().strip())
                        telem[IDX_TEMP] = min(temp_mc / 100_000, 1.0)  # Normalize

            # GPU utilization (physical)
            util_path = f"{self.drm_path}/gpu_busy_percent"
            if os.path.exists(util_path):
                with open(util_path) as f:
                    util = int(f.read().strip())
                    telem[IDX_GPU_UTIL] = util / 100.0

        except Exception as e:
            pass

        return telem

    def read_with_timing(self, inference_time_ms: float) -> np.ndarray:
        """Read physical + timing (full telemetry)."""
        telem = self.read_physical()
        # Add timing (Tier-1) - normalized
        telem[IDX_TIMING] = min(inference_time_ms / 2.0, 1.0)
        return telem


class VariableDepthModel(nn.Module):
    """Simple model with variable depth for testing."""

    def __init__(self, vocab_size: int = 50257, hidden_dim: int = 256, n_layers: int = 6):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.current_depth = n_layers

        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=4, dim_feedforward=hidden_dim*4,
                batch_first=True
            ) for _ in range(n_layers)
        ])
        self.lm_head = nn.Linear(hidden_dim, vocab_size)

    def set_depth(self, depth: int):
        self.current_depth = min(max(depth, 1), self.n_layers)

    def forward(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (logits, hidden_state)."""
        x = self.embed(input_ids)

        for i, layer in enumerate(self.layers):
            if i >= self.current_depth:
                break
            x = layer(x)

        # Pool hidden state (mean over sequence)
        hidden = x.mean(dim=1)  # [batch, hidden_dim]
        logits = self.lm_head(x)

        return logits, hidden


class DynamicsOnlyWorld(nn.Module):
    """World model using ONLY telemetry + action (no LM hidden state)."""

    def __init__(self, telem_dim: int = 9, action_dim: int = 16, hidden_dim: int = 256):
        super().__init__()
        self.action_embed = nn.Embedding(30, action_dim)  # depth * mode combinations

        input_dim = telem_dim + action_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, telem_dim)
        )

    def forward(self, telem: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        action_emb = self.action_embed(action)
        x = torch.cat([telem, action_emb], dim=-1)
        return self.net(x)


class EmbodiedWorld(nn.Module):
    """World model using telemetry + action + LM hidden state."""

    def __init__(self, telem_dim: int = 9, action_dim: int = 16,
                 lm_hidden_dim: int = 256, hidden_dim: int = 256):
        super().__init__()
        self.action_embed = nn.Embedding(30, action_dim)

        # Project LM hidden to smaller dimension
        self.lm_proj = nn.Linear(lm_hidden_dim, 64)

        input_dim = telem_dim + action_dim + 64
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, telem_dim)
        )

    def forward(self, telem: torch.Tensor, action: torch.Tensor,
                lm_hidden: torch.Tensor) -> torch.Tensor:
        action_emb = self.action_embed(action)
        lm_feat = self.lm_proj(lm_hidden.detach())  # Stop gradient from LM
        x = torch.cat([telem, action_emb, lm_feat], dim=-1)
        return self.net(x)


@dataclass
class EpisodeData:
    """Data from one episode."""
    telemetry: List[np.ndarray]  # Body state at each step
    actions: List[int]  # Action at each step (encoded)
    hidden_states: List[np.ndarray]  # LM hidden state at each step
    future_telemetry: List[np.ndarray]  # Target: next body state


def encode_action(depth: int, mode: str) -> int:
    """Encode depth + mode into single action index."""
    mode_idx = {"eco": 0, "balanced": 1, "perf": 2}.get(mode, 1)
    return (depth - 2) * 3 + mode_idx  # 0-14 for depth 2-6, 3 modes


class DataCollector:
    """Collect episodes with LM hidden states."""

    def __init__(self, model: VariableDepthModel, tokenizer,
                 telem_reader: PhysicalTelemetryReader, config: FalsificationConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.telem_reader = telem_reader
        self.config = config
        self.prompts = [
            "The quick brown fox",
            "In the beginning",
            "Once upon a time",
            "The meaning of life",
            "Artificial intelligence",
            "Machine learning is",
            "The future of technology",
            "Climate change affects",
        ]

    def collect_episode(self, episode_id: int) -> EpisodeData:
        """Collect one episode with hidden states."""
        prompt = random.choice(self.prompts)
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.config.device)

        telemetry = []
        actions = []
        hidden_states = []
        future_telemetry = []

        modes = ["eco", "balanced", "perf"]

        for step in range(self.config.episode_length):
            # Select action
            depth = random.choice(self.config.depth_levels)
            mode = random.choice(modes)
            action = encode_action(depth, mode)

            # Apply depth
            self.model.set_depth(depth)

            # Read telemetry BEFORE inference
            telem_before = self.telem_reader.read_physical()

            # Run inference and get hidden state
            start_time = time.perf_counter()
            with torch.no_grad():
                logits, hidden = self.model(input_ids)
            inference_time = (time.perf_counter() - start_time) * 1000

            # Read telemetry AFTER (with timing)
            telem_after = self.telem_reader.read_with_timing(inference_time)

            # Store data
            telemetry.append(telem_before)
            actions.append(action)
            hidden_states.append(hidden.cpu().numpy().squeeze())

            if step > 0:
                future_telemetry.append(telem_after)

            # Generate next token
            next_token = logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
            input_ids = torch.cat([input_ids, next_token], dim=1)

            # Limit sequence length
            if input_ids.shape[1] > 100:
                input_ids = input_ids[:, -100:]

        # Add final future telemetry
        future_telemetry.append(telem_after)

        return EpisodeData(
            telemetry=telemetry[:-1],  # All but last
            actions=actions[:-1],
            hidden_states=hidden_states[:-1],
            future_telemetry=future_telemetry
        )


def train_world_models(episodes: List[EpisodeData], config: FalsificationConfig) -> Dict:
    """Train both world models and compare."""

    # Prepare data
    all_telem = []
    all_actions = []
    all_hidden = []
    all_future = []

    for ep in episodes:
        for i in range(len(ep.telemetry)):
            all_telem.append(ep.telemetry[i])
            all_actions.append(ep.actions[i])
            all_hidden.append(ep.hidden_states[i])
            all_future.append(ep.future_telemetry[i])

    # Convert to tensors
    telem_tensor = torch.tensor(np.array(all_telem), dtype=torch.float32, device=config.device)
    action_tensor = torch.tensor(all_actions, dtype=torch.long, device=config.device)
    hidden_tensor = torch.tensor(np.array(all_hidden), dtype=torch.float32, device=config.device)
    future_tensor = torch.tensor(np.array(all_future), dtype=torch.float32, device=config.device)

    print(f"Training data: {len(all_telem)} samples")
    print(f"Hidden state dim: {hidden_tensor.shape[1]}")

    # Split train/test
    n_train = int(len(all_telem) * 0.8)
    indices = torch.randperm(len(all_telem))
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    # Create models
    dynamics_model = DynamicsOnlyWorld(
        telem_dim=9,
        action_dim=16,
        hidden_dim=config.world_hidden
    ).to(config.device)

    embodied_model = EmbodiedWorld(
        telem_dim=9,
        action_dim=16,
        lm_hidden_dim=hidden_tensor.shape[1],
        hidden_dim=config.world_hidden
    ).to(config.device)

    # Optimizers
    opt_dyn = torch.optim.Adam(dynamics_model.parameters(), lr=config.lr)
    opt_emb = torch.optim.Adam(embodied_model.parameters(), lr=config.lr)

    history = {
        'dynamics_loss': [],
        'embodied_loss': [],
        'dynamics_test': [],
        'embodied_test': []
    }

    # Training loop
    for epoch in range(config.world_epochs):
        dynamics_model.train()
        embodied_model.train()

        # Shuffle training data
        perm = torch.randperm(n_train)
        epoch_loss_dyn = 0
        epoch_loss_emb = 0
        n_batches = 0

        for i in range(0, n_train, config.batch_size):
            batch_idx = train_idx[perm[i:i+config.batch_size]]

            b_telem = telem_tensor[batch_idx]
            b_action = action_tensor[batch_idx]
            b_hidden = hidden_tensor[batch_idx]
            b_future = future_tensor[batch_idx]

            # Dynamics-only
            opt_dyn.zero_grad()
            pred_dyn = dynamics_model(b_telem, b_action)
            loss_dyn = nn.functional.mse_loss(pred_dyn, b_future)
            loss_dyn.backward()
            opt_dyn.step()

            # Embodied
            opt_emb.zero_grad()
            pred_emb = embodied_model(b_telem, b_action, b_hidden)
            loss_emb = nn.functional.mse_loss(pred_emb, b_future)
            loss_emb.backward()
            opt_emb.step()

            epoch_loss_dyn += loss_dyn.item()
            epoch_loss_emb += loss_emb.item()
            n_batches += 1

        history['dynamics_loss'].append(epoch_loss_dyn / n_batches)
        history['embodied_loss'].append(epoch_loss_emb / n_batches)

        # Test evaluation
        if (epoch + 1) % 10 == 0:
            dynamics_model.eval()
            embodied_model.eval()

            with torch.no_grad():
                test_telem = telem_tensor[test_idx]
                test_action = action_tensor[test_idx]
                test_hidden = hidden_tensor[test_idx]
                test_future = future_tensor[test_idx]

                pred_dyn = dynamics_model(test_telem, test_action)
                pred_emb = embodied_model(test_telem, test_action, test_hidden)

                test_loss_dyn = nn.functional.mse_loss(pred_dyn, test_future).item()
                test_loss_emb = nn.functional.mse_loss(pred_emb, test_future).item()

            history['dynamics_test'].append(test_loss_dyn)
            history['embodied_test'].append(test_loss_emb)

            print(f"Epoch {epoch+1}/{config.world_epochs}: "
                  f"Dyn={epoch_loss_dyn/n_batches:.6f} Emb={epoch_loss_emb/n_batches:.6f} "
                  f"Test: Dyn={test_loss_dyn:.6f} Emb={test_loss_emb:.6f}")

    return {
        'dynamics_model': dynamics_model,
        'embodied_model': embodied_model,
        'history': history,
        'test_idx': test_idx,
        'data': {
            'telem': telem_tensor,
            'action': action_tensor,
            'hidden': hidden_tensor,
            'future': future_tensor
        }
    }


def evaluate_tiers(dynamics_model, embodied_model, data, test_idx, config) -> Dict:
    """Evaluate on Tier-1 (timing) and Tier-2 (physical) separately."""

    dynamics_model.eval()
    embodied_model.eval()

    with torch.no_grad():
        test_telem = data['telem'][test_idx]
        test_action = data['action'][test_idx]
        test_hidden = data['hidden'][test_idx]
        test_future = data['future'][test_idx]

        pred_dyn = dynamics_model(test_telem, test_action)
        pred_emb = embodied_model(test_telem, test_action, test_hidden)

        results = {}

        # Overall
        results['overall'] = {
            'dynamics_mse': nn.functional.mse_loss(pred_dyn, test_future).item(),
            'embodied_mse': nn.functional.mse_loss(pred_emb, test_future).item()
        }

        # Tier-1: Timing only
        tier1_dyn = pred_dyn[:, TIER1_CHANNELS]
        tier1_emb = pred_emb[:, TIER1_CHANNELS]
        tier1_target = test_future[:, TIER1_CHANNELS]

        results['tier1_timing'] = {
            'dynamics_mse': nn.functional.mse_loss(tier1_dyn, tier1_target).item(),
            'embodied_mse': nn.functional.mse_loss(tier1_emb, tier1_target).item()
        }

        # Tier-2: Physical sensors only
        tier2_dyn = pred_dyn[:, TIER2_CHANNELS]
        tier2_emb = pred_emb[:, TIER2_CHANNELS]
        tier2_target = test_future[:, TIER2_CHANNELS]

        results['tier2_physical'] = {
            'dynamics_mse': nn.functional.mse_loss(tier2_dyn, tier2_target).item(),
            'embodied_mse': nn.functional.mse_loss(tier2_emb, tier2_target).item()
        }

        # Compute improvement ratios
        for tier in ['overall', 'tier1_timing', 'tier2_physical']:
            dyn = results[tier]['dynamics_mse']
            emb = results[tier]['embodied_mse']
            if dyn > 0:
                improvement = (dyn - emb) / dyn * 100
                results[tier]['improvement_pct'] = improvement
                results[tier]['embodied_wins'] = emb < dyn
            else:
                results[tier]['improvement_pct'] = 0
                results[tier]['embodied_wins'] = False

    return results


def run_falsification_tests(dynamics_model, embodied_model, data, test_idx, config) -> Dict:
    """Run full falsification battery."""

    results = {}

    # Test 1: Does embodied beat dynamics?
    tier_results = evaluate_tiers(dynamics_model, embodied_model, data, test_idx, config)
    results['tier_comparison'] = tier_results

    # Test 2: Mismatch test for both models
    dynamics_model.eval()
    embodied_model.eval()

    with torch.no_grad():
        test_telem = data['telem'][test_idx]
        test_action = data['action'][test_idx]
        test_hidden = data['hidden'][test_idx]
        test_future = data['future'][test_idx]

        # Correct predictions
        pred_dyn_correct = dynamics_model(test_telem, test_action)
        pred_emb_correct = embodied_model(test_telem, test_action, test_hidden)

        # Wrong actions (shift by 1)
        wrong_action = (test_action + 5) % 15
        pred_dyn_wrong = dynamics_model(test_telem, wrong_action)
        pred_emb_wrong = embodied_model(test_telem, wrong_action, test_hidden)

        mse_dyn_correct = nn.functional.mse_loss(pred_dyn_correct, test_future).item()
        mse_dyn_wrong = nn.functional.mse_loss(pred_dyn_wrong, test_future).item()
        mse_emb_correct = nn.functional.mse_loss(pred_emb_correct, test_future).item()
        mse_emb_wrong = nn.functional.mse_loss(pred_emb_wrong, test_future).item()

        results['mismatch'] = {
            'dynamics': {
                'correct': mse_dyn_correct,
                'wrong': mse_dyn_wrong,
                'ratio': mse_dyn_wrong / mse_dyn_correct if mse_dyn_correct > 0 else 0
            },
            'embodied': {
                'correct': mse_emb_correct,
                'wrong': mse_emb_wrong,
                'ratio': mse_emb_wrong / mse_emb_correct if mse_emb_correct > 0 else 0
            }
        }

    # Test 3: Counterfactual sensitivity
    with torch.no_grad():
        sample_telem = test_telem[:100]
        sample_hidden = test_hidden[:100]

        # Vary actions, measure prediction variance
        dyn_preds = []
        emb_preds = []

        for a in range(15):  # All possible actions
            action = torch.full((100,), a, dtype=torch.long, device=config.device)
            dyn_preds.append(dynamics_model(sample_telem, action))
            emb_preds.append(embodied_model(sample_telem, action, sample_hidden))

        dyn_stack = torch.stack(dyn_preds, dim=0)  # [15, 100, 9]
        emb_stack = torch.stack(emb_preds, dim=0)

        dyn_var = dyn_stack.var(dim=0).mean().item()
        emb_var = emb_stack.var(dim=0).mean().item()

        results['counterfactual'] = {
            'dynamics_variance': dyn_var,
            'embodied_variance': emb_var
        }

    return results


def main():
    parser = argparse.ArgumentParser(description="Embodiment Falsification Test")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 70)
    print("z141: EMBODIMENT FALSIFICATION TEST")
    print("=" * 70)
    print("Does LM hidden state (h_t) add predictive power beyond telemetry?")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}")
    print(f"Device: {args.device}")
    print(f"Episodes: {args.episodes}")

    config = FalsificationConfig(
        device=args.device,
        n_episodes=args.episodes,
        world_epochs=args.epochs,
        seed=args.seed
    )

    # Set seeds
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    os.makedirs(config.output_dir, exist_ok=True)

    # Create model and tokenizer
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = VariableDepthModel(
        vocab_size=tokenizer.vocab_size,
        hidden_dim=256,
        n_layers=6
    ).to(config.device)

    telem_reader = PhysicalTelemetryReader(gpu_id=1)

    # Collect episodes
    print(f"\nCollecting {config.n_episodes} episodes...")
    collector = DataCollector(model, tokenizer, telem_reader, config)

    episodes = []
    for i in range(config.n_episodes):
        ep = collector.collect_episode(i)
        episodes.append(ep)
        if (i + 1) % 100 == 0:
            print(f"  Episode {i+1}/{config.n_episodes}")

    # Train world models
    print("\nTraining world models...")
    print("-" * 40)
    results = train_world_models(episodes, config)

    # Run falsification tests
    print("\n" + "=" * 70)
    print("FALSIFICATION RESULTS")
    print("=" * 70)

    test_results = run_falsification_tests(
        results['dynamics_model'],
        results['embodied_model'],
        results['data'],
        results['test_idx'],
        config
    )

    # Print results
    print("\n1. TIER COMPARISON (Does h_t help?)")
    print("-" * 40)
    for tier, data in test_results['tier_comparison'].items():
        dyn = data['dynamics_mse']
        emb = data['embodied_mse']
        imp = data.get('improvement_pct', 0)
        wins = data.get('embodied_wins', False)
        status = "✓ EMBODIED WINS" if wins else "✗ DYNAMICS WINS"
        print(f"  {tier}:")
        print(f"    Dynamics MSE: {dyn:.6f}")
        print(f"    Embodied MSE: {emb:.6f}")
        print(f"    Improvement:  {imp:.2f}%")
        print(f"    Result: {status}")

    print("\n2. MISMATCH TEST")
    print("-" * 40)
    for model_type, data in test_results['mismatch'].items():
        print(f"  {model_type}:")
        print(f"    Correct action MSE: {data['correct']:.6f}")
        print(f"    Wrong action MSE:   {data['wrong']:.6f}")
        print(f"    Ratio: {data['ratio']:.2f}")

    print("\n3. COUNTERFACTUAL SENSITIVITY")
    print("-" * 40)
    print(f"  Dynamics variance: {test_results['counterfactual']['dynamics_variance']:.6f}")
    print(f"  Embodied variance: {test_results['counterfactual']['embodied_variance']:.6f}")

    # Final verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    # Check if embodied significantly beats dynamics on physical tier
    tier2 = test_results['tier_comparison']['tier2_physical']
    embodied_wins_physical = tier2['embodied_wins'] and tier2['improvement_pct'] > 5

    overall = test_results['tier_comparison']['overall']
    embodied_wins_overall = overall['embodied_wins'] and overall['improvement_pct'] > 5

    if embodied_wins_physical:
        verdict = "EMBODIMENT LIKELY - h_t adds predictive power for PHYSICAL sensors"
    elif embodied_wins_overall:
        verdict = "PARTIAL EMBODIMENT - h_t helps overall but not significantly for physical sensors"
    else:
        verdict = "EMBODIMENT NOT PROVEN - h_t does not add significant predictive power"

    print(f"\n{verdict}")
    print(f"\nPhysical tier improvement: {tier2['improvement_pct']:.2f}%")
    print(f"Overall improvement: {overall['improvement_pct']:.2f}%")

    # Save results
    summary = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'episodes': config.n_episodes,
            'epochs': config.world_epochs,
            'seed': config.seed
        },
        'tier_comparison': test_results['tier_comparison'],
        'mismatch': test_results['mismatch'],
        'counterfactual': test_results['counterfactual'],
        'verdict': verdict,
        'training_history': {
            'final_dynamics_loss': results['history']['dynamics_loss'][-1],
            'final_embodied_loss': results['history']['embodied_loss'][-1]
        }
    }

    results_path = f"{config.output_dir}/falsification_results.json"
    with open(results_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {results_path}")
    print(f"Completed: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
