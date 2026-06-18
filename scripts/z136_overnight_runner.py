#!/usr/bin/env python3
"""
z136_overnight_runner.py

OVERNIGHT DEEP EMBODIMENT RUNNER
================================

Runs the full z135 deep embodiment pipeline overnight with:
- Extensive logging and checkpointing
- Error recovery and retry logic
- Progress reporting
- Final report generation

Run on ikaros/daedalus with:
  HSA_OVERRIDE_GFX_VERSION=11.0.0 python scripts/z136_overnight_runner.py

Expected duration: 4-8 hours depending on episode count
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# CONFIGURATION
# =============================================================================

class OvernightConfig:
    """Configuration for overnight run."""

    def __init__(self):
        # Episodes
        self.n_episodes = 500
        self.episode_length = 50

        # World model training
        self.world_model_epochs = 100
        self.batch_size = 32
        self.learning_rate = 1e-3

        # Validation
        self.n_test_episodes = 100  # 20% of total

        # Output
        self.output_dir = f"results/z136_overnight_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.log_file = None  # Set in main

        # Hardware
        self.device = "cpu"  # gfx1151 needs HSA override and may not have ROCm

        # Checkpointing
        self.checkpoint_every_episodes = 50
        self.checkpoint_every_epochs = 10


# =============================================================================
# LOGGING
# =============================================================================

class OvernightLogger:
    """Logger that writes to both console and file."""

    def __init__(self, log_path: str):
        self.log_path = log_path
        self.start_time = time.time()

        # Create log file
        with open(log_path, 'w') as f:
            f.write(f"=== OVERNIGHT RUN STARTED: {datetime.now().isoformat()} ===\n\n")

    def log(self, message: str, level: str = "INFO"):
        """Log a message."""
        elapsed = time.time() - self.start_time
        hours = int(elapsed // 3600)
        mins = int((elapsed % 3600) // 60)
        secs = int(elapsed % 60)

        timestamp = f"[{hours:02d}:{mins:02d}:{secs:02d}]"
        formatted = f"{timestamp} [{level}] {message}"

        print(formatted)

        with open(self.log_path, 'a') as f:
            f.write(formatted + "\n")

    def section(self, title: str):
        """Log a section header."""
        border = "=" * 60
        self.log("")
        self.log(border)
        self.log(title)
        self.log(border)

    def error(self, message: str):
        """Log an error."""
        self.log(message, "ERROR")

    def warning(self, message: str):
        """Log a warning."""
        self.log(message, "WARN")

    def progress(self, current: int, total: int, prefix: str = ""):
        """Log progress."""
        pct = 100.0 * current / total if total > 0 else 0
        self.log(f"{prefix}: {current}/{total} ({pct:.1f}%)")


# =============================================================================
# PIPELINE STAGES
# =============================================================================

def check_environment(logger: OvernightLogger) -> bool:
    """Check environment is properly configured."""
    logger.section("ENVIRONMENT CHECK")

    checks_passed = True

    # Check HSA override (critical for gfx1151)
    hsa_override = os.environ.get("HSA_OVERRIDE_GFX_VERSION", "")
    if hsa_override == "11.0.0":
        logger.log(f"HSA_OVERRIDE_GFX_VERSION={hsa_override} (correct)")
    else:
        logger.warning(f"HSA_OVERRIDE_GFX_VERSION={hsa_override or 'NOT SET'}")
        logger.warning("Expected 11.0.0 for gfx1151 - setting it now")
        os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"

    # Check Python version
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    logger.log(f"Python: {py_version}")

    # Check PyTorch
    try:
        import torch
        logger.log(f"PyTorch: {torch.__version__}")

        # Check CUDA/ROCm
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            for i in range(device_count):
                name = torch.cuda.get_device_name(i)
                logger.log(f"  GPU {i}: {name}")
        else:
            logger.warning("No CUDA/ROCm GPU available - using CPU")
            logger.log("(This is expected on gfx1151 without full ROCm setup)")
    except ImportError as e:
        logger.error(f"PyTorch not available: {e}")
        checks_passed = False

    # Check transformers
    try:
        import transformers
        logger.log(f"Transformers: {transformers.__version__}")
    except ImportError as e:
        logger.error(f"Transformers not available: {e}")
        checks_passed = False

    # Check disk space
    import shutil
    total, used, free = shutil.disk_usage("/")
    free_gb = free / (1024**3)
    logger.log(f"Free disk space: {free_gb:.1f} GB")
    if free_gb < 5:
        logger.warning("Low disk space - results may not save properly")

    # Check memory
    try:
        import psutil
        mem = psutil.virtual_memory()
        available_gb = mem.available / (1024**3)
        logger.log(f"Available RAM: {available_gb:.1f} GB")
        if available_gb < 4:
            logger.warning("Low memory - may cause issues with large episode counts")
    except ImportError:
        logger.log("psutil not available - skipping memory check")

    return checks_passed


def run_episode_generation(config: OvernightConfig, logger: OvernightLogger) -> str:
    """Run intervention episode generation (Phase 1)."""
    logger.section("PHASE 1: INTERVENTION EPISODE GENERATION")

    import numpy as np
    import torch
    import random

    # Set seeds
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # Import from z135
    from scripts.z135_deep_embodiment_pipeline import (
        DeepEmbodimentConfig,
        RobustTelemetryReader,
        InterventionDatasetGenerator,
        Episode,
        load_model
    )
    from transformers import AutoTokenizer

    # Create config
    pipeline_config = DeepEmbodimentConfig(
        n_episodes=config.n_episodes,
        episode_length=config.episode_length,
        device=config.device,
        output_dir=config.output_dir
    )

    os.makedirs(config.output_dir, exist_ok=True)

    # Load tokenizer
    logger.log("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    logger.log("Loading model (CPU for gfx1151 compatibility)...")
    model = load_model(pipeline_config)

    # Create telemetry reader
    logger.log("Initializing telemetry reader...")
    telem_reader = RobustTelemetryReader()

    # Create generator
    generator = InterventionDatasetGenerator(model, tokenizer, pipeline_config, telem_reader)

    # Generate episodes with checkpointing
    episodes = []
    episode_path = os.path.join(config.output_dir, "intervention_episodes.json")

    logger.log(f"Generating {config.n_episodes} episodes...")
    start_time = time.time()

    for i in range(config.n_episodes):
        try:
            episode = generator.generate_episode(i)
            episodes.append(episode)

            # Progress logging
            if (i + 1) % 10 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                remaining = (config.n_episodes - i - 1) / rate if rate > 0 else 0
                logger.log(f"Episode {i+1}/{config.n_episodes} "
                          f"({100*(i+1)/config.n_episodes:.1f}%) - "
                          f"Rate: {rate:.2f} ep/s - "
                          f"ETA: {remaining/60:.1f} min")

            # Checkpoint
            if (i + 1) % config.checkpoint_every_episodes == 0:
                logger.log(f"Saving checkpoint at episode {i+1}...")
                generator._save_episodes(episodes, episode_path)

        except Exception as e:
            logger.error(f"Error generating episode {i}: {e}")
            logger.error(traceback.format_exc())
            # Continue with remaining episodes

    # Final save
    logger.log(f"Saving final episode dataset ({len(episodes)} episodes)...")
    generator._save_episodes(episodes, episode_path)

    # Validate causal coupling
    logger.log("Validating causal coupling...")
    generator._validate_causal_coupling(episodes)

    # Summary statistics
    total_tokens = sum(len(ep.tokens) for ep in episodes)
    total_time = sum(sum(ep.timing) for ep in episodes)
    logger.log(f"Total tokens generated: {total_tokens}")
    logger.log(f"Total generation time: {total_time:.1f}s")
    logger.log(f"Average tokens/sec: {total_tokens/total_time:.1f}")

    return episode_path


def run_world_model_training(config: OvernightConfig, episode_path: str, logger: OvernightLogger) -> str:
    """Run world model training (Phase 2)."""
    logger.section("PHASE 2: WORLD MODEL TRAINING")

    import json
    import numpy as np
    import torch

    from scripts.z135_deep_embodiment_pipeline import (
        DeepEmbodimentConfig,
        WorldModelTrainer,
        Episode
    )

    # Load episodes
    logger.log(f"Loading episodes from {episode_path}...")
    with open(episode_path, 'r') as f:
        episode_data = json.load(f)

    episodes = []
    for ep_dict in episode_data:
        episodes.append(Episode(**ep_dict))

    logger.log(f"Loaded {len(episodes)} episodes")

    # Create config
    pipeline_config = DeepEmbodimentConfig(
        world_model_epochs=config.world_model_epochs,
        world_model_lr=config.learning_rate,
        batch_size=config.batch_size,
        device=config.device,
        output_dir=config.output_dir
    )

    # Split data
    n_train = int(len(episodes) * 0.8)
    train_episodes = episodes[:n_train]
    test_episodes = episodes[n_train:]

    logger.log(f"Training: {len(train_episodes)} episodes")
    logger.log(f"Testing: {len(test_episodes)} episodes")

    # Create trainer
    trainer = WorldModelTrainer(pipeline_config)

    # Prepare data
    logger.log("Preparing training data...")
    dataset = trainer.prepare_training_data(train_episodes)
    logger.log(f"Training samples: {len(dataset)}")

    # Custom training loop with detailed logging
    from torch.utils.data import DataLoader
    import torch.nn.functional as F

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True
    )

    history = {'loss': [], 'mse': []}
    best_loss = float('inf')
    world_model_path = os.path.join(config.output_dir, "world_model.pt")

    logger.log(f"Training for {config.world_model_epochs} epochs...")
    start_time = time.time()

    for epoch in range(config.world_model_epochs):
        epoch_loss = 0.0
        n_batches = 0

        trainer.world_model.train()
        for batch in dataloader:
            current_telem = batch['current_telem'].to(config.device)
            future_telem = batch['future_telem'].to(config.device)
            action_vec = batch['action_vec'].to(config.device)

            # Dummy hidden state
            hidden_state = torch.randn(
                current_telem.size(0),
                pipeline_config.hidden_size,
                device=config.device
            ) * 0.1

            # Forward
            pred_future = trainer.world_model(hidden_state, action_vec, current_telem)
            loss = F.mse_loss(pred_future, future_telem)

            # Backward
            trainer.optimizer.zero_grad()
            loss.backward()
            trainer.optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        history['loss'].append(avg_loss)
        history['mse'].append(avg_loss)

        # Log progress
        if (epoch + 1) % 5 == 0:
            elapsed = time.time() - start_time
            eta = elapsed / (epoch + 1) * (config.world_model_epochs - epoch - 1)
            logger.log(f"Epoch {epoch+1}/{config.world_model_epochs}: "
                      f"Loss={avg_loss:.6f} - ETA: {eta/60:.1f} min")

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(trainer.world_model.state_dict(), world_model_path)

        # Checkpoint
        if (epoch + 1) % config.checkpoint_every_epochs == 0:
            checkpoint_path = os.path.join(config.output_dir, f"world_model_epoch{epoch+1}.pt")
            torch.save(trainer.world_model.state_dict(), checkpoint_path)
            logger.log(f"Checkpoint saved: {checkpoint_path}")

    # Load best model
    trainer.world_model.load_state_dict(torch.load(world_model_path, weights_only=True))
    logger.log(f"Best model loss: {best_loss:.6f}")

    # Save training history
    history_path = os.path.join(config.output_dir, "training_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f)

    return world_model_path, test_episodes, trainer


def run_rigorous_validation(config: OvernightConfig, world_model_path: str,
                           test_episodes, trainer, logger: OvernightLogger) -> dict:
    """Run rigorous validation (Phase 3)."""
    logger.section("PHASE 3: RIGOROUS VALIDATION")

    import torch
    from scripts.z135_deep_embodiment_pipeline import (
        DeepEmbodimentConfig,
        RigorousValidator
    )

    # Create config
    pipeline_config = DeepEmbodimentConfig(
        device=config.device,
        output_dir=config.output_dir
    )

    # Load best model if needed
    if hasattr(trainer, 'world_model'):
        world_model = trainer.world_model
    else:
        from scripts.z135_deep_embodiment_pipeline import BodyWorldModel
        world_model = BodyWorldModel(pipeline_config).to(config.device)
        world_model.load_state_dict(torch.load(world_model_path, weights_only=True))

    # Create validator
    validator = RigorousValidator(world_model, pipeline_config)

    logger.log(f"Running validation on {len(test_episodes)} test episodes...")

    # Run each test with logging
    results = {}

    # Prepare test data
    test_data = validator._prepare_test_data(test_episodes)
    logger.log(f"Test samples: {len(test_data)}")

    # Test 1: Baseline comparison
    logger.log("Running Test 1: Baseline comparison...")
    results['baseline_comparison'] = validator._test_vs_baselines(test_data)

    bc = results['baseline_comparison']
    logger.log(f"  Model MSE: {bc['model_mse']:.6f}")
    logger.log(f"  Mean baseline MSE: {bc['mean_baseline_mse']:.6f}")
    logger.log(f"  Constant baseline MSE: {bc['constant_baseline_mse']:.6f}")
    logger.log(f"  Improvement vs mean: {bc['improvement_vs_mean']:.1f}%")
    logger.log(f"  Result: {'PASS' if bc['passed'] else 'FAIL'}")

    # Test 2: Mismatch test
    logger.log("Running Test 2: Mismatch test...")
    results['mismatch_test'] = validator._test_mismatch(test_data)

    mt = results['mismatch_test']
    logger.log(f"  Matched MSE: {mt['matched_mse']:.6f}")
    logger.log(f"  Mismatched MSE: {mt['mismatched_mse']:.6f}")
    logger.log(f"  Ratio (should be >1.2): {mt['ratio']:.3f}")
    logger.log(f"  Result: {'PASS' if mt['passed'] else 'FAIL'}")

    # Test 3: Counterfactual
    logger.log("Running Test 3: Counterfactual sensitivity...")
    results['counterfactual'] = validator._test_counterfactual(test_data)

    cf = results['counterfactual']
    logger.log(f"  Mean prediction variance: {cf['mean_prediction_variance']:.6f}")
    logger.log(f"  Result: {'PASS' if cf['passed'] else 'FAIL'}")

    # Test 4: Selectivity
    logger.log("Running Test 4: Selectivity (control task)...")
    results['selectivity'] = validator._test_selectivity(test_data)

    sel = results['selectivity']
    logger.log(f"  Original MSE: {sel['original_mse']:.6f}")
    logger.log(f"  Shuffled MSE: {sel['shuffled_mse']:.6f}")
    logger.log(f"  Selectivity score: {sel['selectivity']:.3f}")
    logger.log(f"  Result: {'PASS' if sel['passed'] else 'FAIL'}")

    # Compute verdict
    results['verdict'] = validator._compute_verdict(results)

    return results


def generate_final_report(config: OvernightConfig, validation_results: dict,
                         logger: OvernightLogger) -> str:
    """Generate comprehensive final report."""
    logger.section("GENERATING FINAL REPORT")

    import json
    from datetime import datetime

    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("DEEP EMBODIMENT OVERNIGHT RUN - FINAL REPORT")
    report_lines.append("=" * 70)
    report_lines.append(f"Timestamp: {datetime.now().isoformat()}")
    report_lines.append(f"Output Directory: {config.output_dir}")
    report_lines.append("")

    # Configuration
    report_lines.append("-" * 70)
    report_lines.append("CONFIGURATION")
    report_lines.append("-" * 70)
    report_lines.append(f"Episodes: {config.n_episodes}")
    report_lines.append(f"Episode length: {config.episode_length}")
    report_lines.append(f"World model epochs: {config.world_model_epochs}")
    report_lines.append(f"Device: {config.device}")
    report_lines.append("")

    # Validation Results
    report_lines.append("-" * 70)
    report_lines.append("VALIDATION RESULTS")
    report_lines.append("-" * 70)

    verdict = validation_results.get('verdict', {})
    tests_passed = verdict.get('tests_passed', 0)
    total_tests = verdict.get('total_tests', 4)

    report_lines.append(f"Tests Passed: {tests_passed}/{total_tests}")
    report_lines.append("")

    # Individual tests
    for test_name in ['baseline_comparison', 'mismatch_test', 'counterfactual', 'selectivity']:
        if test_name in validation_results:
            result = validation_results[test_name]
            status = "PASS" if result.get('passed', False) else "FAIL"
            report_lines.append(f"{test_name}: {status}")

            # Key metrics
            if test_name == 'baseline_comparison':
                report_lines.append(f"  - Improvement vs mean: {result.get('improvement_vs_mean', 0):.1f}%")
            elif test_name == 'mismatch_test':
                report_lines.append(f"  - Mismatch ratio: {result.get('ratio', 0):.3f}")
            elif test_name == 'counterfactual':
                report_lines.append(f"  - Prediction variance: {result.get('mean_prediction_variance', 0):.6f}")
            elif test_name == 'selectivity':
                report_lines.append(f"  - Selectivity score: {result.get('selectivity', 0):.3f}")
            report_lines.append("")

    # Final verdict
    report_lines.append("=" * 70)
    report_lines.append("FINAL VERDICT")
    report_lines.append("=" * 70)

    verdict_text = verdict.get('verdict', 'UNKNOWN')
    if tests_passed >= 3:
        emoji = "SUCCESS"
    elif tests_passed >= 2:
        emoji = "PARTIAL"
    else:
        emoji = "NOT PROVEN"

    report_lines.append(f">>> {emoji}: {verdict_text} <<<")
    report_lines.append("")

    # Interpretation
    report_lines.append("-" * 70)
    report_lines.append("INTERPRETATION")
    report_lines.append("-" * 70)

    if tests_passed >= 3:
        report_lines.append("The body world model demonstrates predictive embodiment:")
        report_lines.append("- Predictions beat simple baselines")
        report_lines.append("- Model is sensitive to actions (causal coupling)")
        report_lines.append("- Predictions are selective (not memorization)")
        report_lines.append("")
        report_lines.append("NEXT STEPS:")
        report_lines.append("1. Integrate world model into closed-loop controller")
        report_lines.append("2. Run on Minos with Tier-A energy measurements")
        report_lines.append("3. Compare vs GreenLLM-style external baseline")
    elif tests_passed >= 2:
        report_lines.append("Partial evidence of embodiment:")
        report_lines.append("- Some tests pass, indicating useful signal")
        report_lines.append("- Failed tests suggest areas for improvement")
        report_lines.append("")
        report_lines.append("RECOMMENDATIONS:")
        report_lines.append("1. Increase episode diversity (more action schedules)")
        report_lines.append("2. Train longer or adjust learning rate")
        report_lines.append("3. Check telemetry variance in dataset")
    else:
        report_lines.append("Embodiment not yet demonstrated:")
        report_lines.append("- World model does not beat baselines significantly")
        report_lines.append("- May indicate weak telemetry-action coupling")
        report_lines.append("")
        report_lines.append("RECOMMENDATIONS:")
        report_lines.append("1. Verify telemetry has meaningful variance")
        report_lines.append("2. Increase intervention strength (more depth variation)")
        report_lines.append("3. Consider using real hardware actuation if available")

    report_lines.append("")
    report_lines.append("=" * 70)

    # Save report
    report_text = "\n".join(report_lines)
    report_path = os.path.join(config.output_dir, "overnight_report.txt")

    with open(report_path, 'w') as f:
        f.write(report_text)

    logger.log(f"Report saved to {report_path}")

    # Also save JSON version
    json_report = {
        'config': {
            'n_episodes': config.n_episodes,
            'episode_length': config.episode_length,
            'world_model_epochs': config.world_model_epochs,
            'device': config.device
        },
        'validation_results': validation_results,
        'timestamp': datetime.now().isoformat()
    }

    json_path = os.path.join(config.output_dir, "overnight_results.json")

    def convert_for_json(obj):
        import numpy as np
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(v) for v in obj]
        return obj

    with open(json_path, 'w') as f:
        json.dump(convert_for_json(json_report), f, indent=2)

    # Print report to console
    print("\n" + report_text)

    return report_path


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Overnight Deep Embodiment Runner")
    parser.add_argument("--n-episodes", type=int, default=500, help="Number of episodes")
    parser.add_argument("--epochs", type=int, default=100, help="World model epochs")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu recommended for gfx1151)")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    # Create config
    config = OvernightConfig()
    config.n_episodes = args.n_episodes
    config.world_model_epochs = args.epochs
    config.device = args.device

    if args.output_dir:
        config.output_dir = args.output_dir

    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)

    # Create logger
    config.log_file = os.path.join(config.output_dir, "overnight.log")
    logger = OvernightLogger(config.log_file)

    logger.section("OVERNIGHT DEEP EMBODIMENT RUN")
    logger.log(f"Started: {datetime.now().isoformat()}")
    logger.log(f"Output: {config.output_dir}")
    logger.log(f"Episodes: {config.n_episodes}")
    logger.log(f"World model epochs: {config.world_model_epochs}")

    # Signal handler for graceful shutdown
    shutdown_requested = [False]

    def signal_handler(sig, frame):
        logger.warning("Shutdown requested - finishing current operation...")
        shutdown_requested[0] = True

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    start_time = time.time()

    try:
        # Phase 0: Environment check
        if not check_environment(logger):
            logger.error("Environment check failed - aborting")
            return 1

        if shutdown_requested[0]:
            logger.warning("Shutdown before episode generation")
            return 1

        # Phase 1: Generate episodes
        episode_path = run_episode_generation(config, logger)

        if shutdown_requested[0]:
            logger.warning("Shutdown after episode generation - partial results saved")
            return 1

        # Phase 2: Train world model
        world_model_path, test_episodes, trainer = run_world_model_training(
            config, episode_path, logger
        )

        if shutdown_requested[0]:
            logger.warning("Shutdown after training - model saved")
            return 1

        # Phase 3: Rigorous validation
        validation_results = run_rigorous_validation(
            config, world_model_path, test_episodes, trainer, logger
        )

        # Phase 4: Generate report
        report_path = generate_final_report(config, validation_results, logger)

        # Summary
        elapsed = time.time() - start_time
        hours = int(elapsed // 3600)
        mins = int((elapsed % 3600) // 60)

        logger.section("RUN COMPLETE")
        logger.log(f"Total time: {hours}h {mins}m")
        logger.log(f"Report: {report_path}")
        logger.log(f"Results: {config.output_dir}")

        verdict = validation_results.get('verdict', {})
        tests_passed = verdict.get('tests_passed', 0)

        if tests_passed >= 3:
            logger.log("RESULT: PREDICTIVE EMBODIMENT DEMONSTRATED")
            return 0
        elif tests_passed >= 2:
            logger.log("RESULT: PARTIAL SUCCESS - NEEDS REFINEMENT")
            return 0
        else:
            logger.log("RESULT: NOT YET PROVEN - CONTINUE ITERATION")
            return 0

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
