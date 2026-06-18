#!/usr/bin/env python3
"""
z105_feel_validation.py - Quick validation of FEEL-SLM architecture

Tests that all components work together:
1. Model instantiation (baseline and FEEL)
2. Forward pass
3. Loss computation
4. Data loading
5. Mini training step
6. Generation

Run with: HSA_OVERRIDE_GFX_VERSION=11.0.0 python scripts/z105_feel_validation.py
"""

import sys
import os
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import time

def test_imports():
    """Test all imports work."""
    print("=" * 60)
    print("1. Testing imports...")
    print("=" * 60)

    from src.feel_slm import (
        FEELConfig,
        TrainerConfig,
        FEELSLM,
        BaselineSLM,
        BodyEncoder,
        GatedBodyInjection,
        FEELLoss,
        InvarianceLoss,
        ContextualBandit,
        TelemetryAugmenter,
        create_dataloaders,
    )
    print("✓ All imports successful")
    return True


def test_config():
    """Test configuration."""
    print("\n" + "=" * 60)
    print("2. Testing configuration...")
    print("=" * 60)

    from src.feel_slm import FEELConfig

    # Test presets
    for name, fn in [("tiny", FEELConfig.tiny),
                     ("small", FEELConfig.small),
                     ("medium", FEELConfig.medium)]:
        config = fn()
        params_estimate = (
            config.vocab_size * config.hidden_dim +  # embedding
            config.num_layers * (4 * config.hidden_dim * config.hidden_dim +
                                3 * config.hidden_dim * config.intermediate_dim)  # layers
        )
        print(f"  {name}: hidden={config.hidden_dim}, layers={config.num_layers}, "
              f"~{params_estimate/1e6:.1f}M params")

    print("✓ Configuration test passed")
    return True


def test_models():
    """Test model instantiation and forward pass."""
    print("\n" + "=" * 60)
    print("3. Testing models...")
    print("=" * 60)

    from src.feel_slm import FEELConfig, FEELSLM, BaselineSLM

    config = FEELConfig.tiny()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    # Test baseline
    baseline = BaselineSLM(config).to(device)
    baseline_params = sum(p.numel() for p in baseline.parameters())
    print(f"  BaselineSLM: {baseline_params:,} parameters")

    # Test FEEL
    feel = FEELSLM(config).to(device)
    feel_params = sum(p.numel() for p in feel.parameters())
    print(f"  FEELSLM: {feel_params:,} parameters")
    print(f"  Body path adds: {feel_params - baseline_params:,} parameters "
          f"({100*(feel_params - baseline_params)/baseline_params:.1f}%)")

    # Forward pass
    batch_size = 2
    seq_len = 64
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
    telemetry = torch.rand(batch_size, config.body_dim, device=device)

    # Baseline forward - returns dict
    with torch.no_grad():
        baseline_output = baseline(input_ids)
        if isinstance(baseline_output, dict):
            baseline_logits = baseline_output['logits']
        else:
            baseline_logits = baseline_output
        print(f"  Baseline output shape: {baseline_logits.shape}")

    # FEEL forward - returns dict
    with torch.no_grad():
        feel_output = feel(input_ids, telemetry)
        if isinstance(feel_output, dict):
            feel_logits = feel_output['logits']
            print(f"  FEEL output shape: {feel_logits.shape}")
            print(f"  FEEL output keys: {list(feel_output.keys())}")
        else:
            feel_logits = feel_output
            print(f"  FEEL output shape: {feel_logits.shape}")

    print("✓ Models test passed")
    return True


def test_body_encoder():
    """Test body encoder."""
    print("\n" + "=" * 60)
    print("4. Testing body encoder...")
    print("=" * 60)

    from src.feel_slm import FEELConfig, BodyEncoder, TelemetrySnapshot
    from src.feel_slm.body_encoder import BodyLatentComputer

    config = FEELConfig.tiny()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    encoder = BodyEncoder(config).to(device)
    telemetry = torch.rand(2, config.body_dim, device=device)

    # Forward
    body_embed = encoder(telemetry)
    print(f"  Input shape: {telemetry.shape}")
    print(f"  Output shape: {body_embed.shape}")
    print(f"  Output norm (should be ~0.1 scaled): {body_embed.norm(dim=-1).mean().item():.4f}")

    # Test latent computer - takes config
    latent_computer = BodyLatentComputer(config).to(device)
    snapshot = TelemetrySnapshot(
        power_watts=150.0, temp_c=65.0, utilization=0.8,
        clock_mhz=2400, mem_util=0.5, mem_used_gb=8.0
    )
    telem_tensor = snapshot.to_vector()  # to_vector(), not to_tensor()
    latent = latent_computer(telem_tensor.unsqueeze(0).to(device))
    print(f"  Body latent shape: {latent.shape}")
    print(f"  Body latent: {[f'{x:.3f}' for x in latent.squeeze().tolist()]}")

    print("✓ Body encoder test passed")
    return True


def test_gated_injection():
    """Test gated injection."""
    print("\n" + "=" * 60)
    print("5. Testing gated injection...")
    print("=" * 60)

    from src.feel_slm import FEELConfig, GatedBodyInjection

    config = FEELConfig.tiny()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # GatedBodyInjection takes (config, layer_idx)
    gate = GatedBodyInjection(config, layer_idx=0).to(device)

    hidden = torch.randn(2, 64, config.hidden_dim, device=device)
    body_embed = torch.randn(2, config.body_embed_dim, device=device)  # body_embed_dim, not hidden_dim

    output, alpha = gate(hidden, body_embed, return_gate=True)

    print(f"  Hidden shape: {hidden.shape}")
    print(f"  Body embed shape: {body_embed.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Gate alpha shape: {alpha.shape}")
    print(f"  Gate mean (should be low initially): {alpha.mean().item():.4f}")

    # Check gate stats
    stats = gate.get_gate_stats(hidden, body_embed)
    print(f"  Gate stats: {stats}")

    print("✓ Gated injection test passed")
    return True


def test_losses():
    """Test loss functions."""
    print("\n" + "=" * 60)
    print("6. Testing losses...")
    print("=" * 60)

    from src.feel_slm import FEELConfig, InvarianceLoss, FEELLoss
    from src.feel_slm.losses import ReporterLoss, PolicyLoss

    config = FEELConfig.tiny()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size = 2
    seq_len = 64

    # Create dummy tensors
    logits1 = torch.randn(batch_size, seq_len, config.vocab_size, device=device)
    logits2 = torch.randn(batch_size, seq_len, config.vocab_size, device=device)
    labels = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
    telemetry = torch.rand(batch_size, config.body_dim, device=device)
    predicted_telemetry = torch.rand(batch_size, config.body_dim, device=device)

    # Invariance loss
    inv_loss = InvarianceLoss()
    inv_val = inv_loss(logits1, logits2)
    print(f"  Invariance loss: {inv_val.item():.4f}")

    # Reporter loss - takes dicts
    rep_loss = ReporterLoss()
    reporter_outputs = {
        'telemetry_pred': predicted_telemetry,
        'strain_logits': torch.randn(batch_size, 4, device=device),
        'mode_logits': torch.randn(batch_size, 5, device=device),
    }
    targets = {
        'telemetry': telemetry,
        'strain_labels': torch.randint(0, 4, (batch_size,), device=device),
        'mode_labels': torch.randint(0, 5, (batch_size,), device=device),
    }
    rep_val, rep_breakdown = rep_loss(reporter_outputs, targets)
    print(f"  Reporter loss: {rep_val.item():.4f}")
    print(f"    Breakdown: {rep_breakdown}")

    # Policy loss
    pol_loss = PolicyLoss().to(device)
    action_logits = torch.randn(batch_size, 3, device=device)
    actions = torch.randint(0, 3, (batch_size,), device=device)
    rewards = torch.rand(batch_size, device=device)
    pol_val, pol_info = pol_loss(action_logits, actions, rewards)
    print(f"  Policy loss: {pol_val.item():.4f}")
    print(f"    Info: {pol_info}")

    print("✓ Losses test passed")
    return True


def test_data():
    """Test data loading."""
    print("\n" + "=" * 60)
    print("7. Testing data loading...")
    print("=" * 60)

    from src.feel_slm import TelemetryAugmenter, create_dataloaders

    # Test augmenter
    aug = TelemetryAugmenter()
    for pattern in ['idle', 'working', 'stressed', 'cooling']:
        telem, latent, strain, mode = aug.generate(pattern)
        print(f"  {pattern}: strain={strain}, mode={mode}")

    # Test dataloader (small sample)
    train_loader, val_loader = create_dataloaders(
        batch_size=4,
        max_length=64,
        train_samples=20,
        val_samples=10,
        num_workers=0,
    )

    batch = next(iter(train_loader))
    print(f"  Batch input_ids: {batch['input_ids'].shape}")
    print(f"  Batch telemetry: {batch['telemetry'].shape}")
    print(f"  Batch body_latent: {batch['body_latent'].shape}")
    print(f"  Batch strain_labels: {batch['strain_labels']}")

    print("✓ Data loading test passed")
    return True


def test_mini_training():
    """Test a mini training step."""
    print("\n" + "=" * 60)
    print("8. Testing mini training step...")
    print("=" * 60)

    from src.feel_slm import FEELConfig, FEELSLM, FEELLoss, create_dataloaders
    from src.feel_slm.losses import LossWeights

    config = FEELConfig.tiny()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = FEELSLM(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # FEELLoss takes LossWeights, not individual params
    weights = LossWeights(lm=1.0, reporter=0.1, invariance=0.05)
    loss_fn = FEELLoss(weights=weights)

    # Get batch
    train_loader, _ = create_dataloaders(
        batch_size=2,
        max_length=32,
        train_samples=10,
        num_workers=0,
    )
    batch = next(iter(train_loader))
    batch = {k: v.to(device) for k, v in batch.items()}

    # Forward
    model.train()
    output = model(batch['input_ids'], batch['telemetry'])

    # Compute LM loss manually (simpler test)
    import torch.nn.functional as F
    logits = output['logits']
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = batch['labels'][:, 1:].contiguous()
    lm_loss = F.cross_entropy(
        shift_logits.view(-1, config.vocab_size),
        shift_labels.view(-1),
        ignore_index=0
    )

    print(f"  LM loss: {lm_loss.item():.4f}")
    print(f"  Output keys: {list(output.keys())}")

    # Backward
    optimizer.zero_grad()
    lm_loss.backward()

    # Check gradients exist
    grad_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            grad_norm += p.grad.norm().item() ** 2
    grad_norm = grad_norm ** 0.5
    print(f"  Gradient norm: {grad_norm:.4f}")

    # Step
    optimizer.step()
    print("✓ Mini training test passed")
    return True


def test_generation():
    """Test text generation."""
    print("\n" + "=" * 60)
    print("9. Testing generation...")
    print("=" * 60)

    from src.feel_slm import FEELConfig, FEELSLM, BaselineSLM

    config = FEELConfig.tiny()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Test baseline generation
    baseline = BaselineSLM(config).to(device)
    prompt = torch.randint(0, config.vocab_size, (1, 16), device=device)

    start = time.time()
    baseline_output = baseline.generate(prompt, max_new_tokens=20)
    baseline_time = time.time() - start
    print(f"  Baseline generation: {baseline_output.shape[1]} tokens in {baseline_time*1000:.1f}ms")

    # Test FEEL generation
    feel = FEELSLM(config).to(device)
    telemetry = torch.rand(1, config.body_dim, device=device)

    start = time.time()
    feel_output, gate_history = feel.generate(prompt, telemetry, max_new_tokens=20)
    feel_time = time.time() - start
    print(f"  FEEL generation: {feel_output.shape[1]} tokens in {feel_time*1000:.1f}ms")
    print(f"  Gate history length: {len(gate_history)}")
    print(f"  Gate values sample: {[f'{g:.3f}' for g in gate_history[:5]]}")

    print("✓ Generation test passed")
    return True


def test_bandit():
    """Test contextual bandit."""
    print("\n" + "=" * 60)
    print("10. Testing bandit...")
    print("=" * 60)

    from src.feel_slm import ContextualBandit, AdaptiveProfileSelector

    bandit = ContextualBandit(state_dim=5, num_actions=3)

    # Simulate interactions
    for i in range(20):
        body_state = torch.rand(5)
        action, info = bandit.select_action(body_state)
        reward = 1.0 - body_state[0].item() if action == 0 else body_state[0].item()
        bandit.update(body_state, action, reward)

    stats = bandit.get_stats()
    print(f"  Bandit stats after 20 steps:")
    print(f"    Epsilon: {stats['epsilon']:.4f}")
    print(f"    Action counts: {stats['action_counts']}")

    # Test adaptive selector
    selector = AdaptiveProfileSelector(bandit)
    body_latent = torch.tensor([0.3, 0.2, 0.1, 0.2, 0.8])
    profile, info = selector.select_profile(body_latent)
    print(f"  Selected profile (low margin): {profile} - {info}")

    print("✓ Bandit test passed")
    return True


def main():
    """Run all validation tests."""
    print("\n" + "=" * 60)
    print("FEEL-SLM Architecture Validation")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    tests = [
        ("Imports", test_imports),
        ("Config", test_config),
        ("Models", test_models),
        ("Body Encoder", test_body_encoder),
        ("Gated Injection", test_gated_injection),
        ("Losses", test_losses),
        ("Data Loading", test_data),
        ("Mini Training", test_mini_training),
        ("Generation", test_generation),
        ("Bandit", test_bandit),
    ]

    results = []
    for name, test_fn in tests:
        try:
            result = test_fn()
            results.append((name, result, None))
        except Exception as e:
            print(f"✗ {name} FAILED: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False, str(e)))

    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, r, _ in results if r)
    total = len(results)

    for name, result, error in results:
        status = "✓ PASSED" if result else f"✗ FAILED: {error}"
        print(f"  {name}: {status}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n✅ ALL VALIDATION TESTS PASSED!")
        print("\nFEEL-SLM architecture is ready for training.")
        return 0
    else:
        print(f"\n❌ {total - passed} tests failed")
        return 1


if __name__ == "__main__":
    exit(main())
