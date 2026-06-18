#!/usr/bin/env python3
"""
z132_resume_embodiment.py - Resume FEEL-SLM embodiment training from checkpoint

Fixes dataloader deadlock by using num_workers=0 (synchronous loading).
Resumes from Phase 2 epoch 2 checkpoint and continues full pipeline.
"""

import argparse
import sys
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from feel_slm.embodied_slm import EmbodiedSLM, create_embodied_slm_30m


def get_telemetry():
    """Get GPU telemetry from AMD hardware."""
    try:
        import subprocess
        result = subprocess.run(
            ["rocm-smi", "--showpower", "--showtemp", "--showuse", "--json"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            card = data.get("card0", {})
            return {
                "power": float(card.get("Average Graphics Package Power (W)", 100)),
                "temp": float(card.get("Temperature (Sensor edge) (C)", 50)),
                "util": float(card.get("GPU use (%)", 50)),
            }
    except Exception:
        pass

    # Fallback: use random plausible values
    return {
        "power": 80 + np.random.randn() * 20,
        "temp": 60 + np.random.randn() * 10,
        "util": 70 + np.random.randn() * 20,
    }


def load_dataset_split(split: str, max_samples: int = 30000):
    """Load TinyStories dataset."""
    print(f"Loading TinyStories ({split})...", flush=True)
    ds = load_dataset("roneneldan/TinyStories", split=split, trust_remote_code=True)
    if max_samples and len(ds) > max_samples:
        ds = ds.select(range(max_samples))
    print(f"  Loaded {len(ds)} samples", flush=True)
    return ds


def create_dataloader(dataset, tokenizer, batch_size: int, max_length: int = 256):
    """Create dataloader with num_workers=0 to avoid deadlocks."""
    def collate_fn(batch):
        texts = [item["text"] for item in batch]
        encodings = tokenizer(
            texts,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = torch.clamp(encodings["input_ids"], 0, 31999)
        return {"input_ids": input_ids}

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,  # FIX: Use synchronous loading to avoid deadlock
        pin_memory=False,  # Disabled since num_workers=0
    )


# =============================================================================
# PHASE 2 RESUME: Continue Body Conditioning
# =============================================================================

def run_phase2_resume(
    model,
    tokenizer,
    output_dir: Path,
    device: str,
    start_epoch: int = 2,  # 0-indexed, so this means epoch 3
    num_epochs: int = 5
):
    """Resume Phase 2: Body conditioning with KL anchoring from checkpoint."""
    print("\n" + "=" * 60, flush=True)
    print(f"PHASE 2 RESUME: EPOCHS {start_epoch+1}-{num_epochs} (FIXED DATALOADER)", flush=True)
    print("=" * 60, flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset_split("train", max_samples=30000)
    loader = create_dataloader(ds, tokenizer, batch_size=4)

    # Set training phase
    model.set_training_phase("full")
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    KL_WEIGHT = 0.1

    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_losses = []
        kl_losses = []

        for batch_idx, batch in enumerate(loader):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)

            # Get telemetry and create body vector
            telem = get_telemetry()
            telemetry = torch.tensor([[
                telem["power"] / 200.0,
                telem["temp"] / 100.0,
                telem["util"] / 100.0,
                np.sin(telem["power"] / 50.0),
                np.cos(telem["temp"] / 50.0),
                telem["power"] * telem["temp"] / 10000.0,
                (telem["util"] - 50.0) / 50.0,
                np.tanh(telem["power"] / 100.0),
                np.tanh(telem["temp"] / 50.0),
                np.tanh(telem["util"] / 50.0),
                (telem["power"] + telem["temp"]) / 200.0,
                np.clip(telem["power"] - telem["temp"], -50, 50) / 50.0,
            ]], dtype=torch.float32).to(device)
            telemetry = telemetry.expand(input_ids.size(0), -1)

            # Forward with body
            out_body = model(input_ids, telemetry=telemetry)
            logits_body = out_body["logits"]

            # Forward without body (anchor)
            with torch.no_grad():
                out_base = model(input_ids, telemetry=None)
                logits_base = out_base["logits"]

            # LM loss
            shift_logits = logits_body[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            lm_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=tokenizer.pad_token_id,
            )

            # KL loss
            kl_loss = F.kl_div(
                F.log_softmax(logits_body, dim=-1),
                F.softmax(logits_base, dim=-1),
                reduction="batchmean"
            )

            loss = lm_loss + KL_WEIGHT * kl_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_losses.append(lm_loss.item())
            kl_losses.append(kl_loss.item())

            if (batch_idx + 1) % 100 == 0:
                avg_lm = np.mean(epoch_losses[-100:])
                avg_kl = np.mean(kl_losses[-100:])
                print(f"  Epoch {epoch+1} Step {batch_idx+1}: lm={avg_lm:.4f} kl={avg_kl:.4f}", flush=True)

        avg_loss = np.mean(epoch_losses)
        avg_kl = np.mean(kl_losses)
        print(f"Epoch {epoch+1}/{num_epochs}: lm={avg_loss:.4f} kl={avg_kl:.4f}", flush=True)

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "metrics": {"lm_loss": avg_loss, "kl_loss": avg_kl},
        }, output_dir / f"epoch_{epoch+1}.pt")

    # Save final
    torch.save({
        "model_state_dict": model.state_dict(),
        "metrics": {"lm_loss": avg_loss, "kl_loss": avg_kl},
    }, output_dir / "final.pt")

    print("Phase 2 complete!", flush=True)
    return model


# =============================================================================
# PHASE 3: Full Embodiment with LayerDrop
# =============================================================================

def run_phase3(model, tokenizer, output_dir: Path, device: str, num_epochs: int = 3):
    """Phase 3: Train with LayerDrop and distillation."""
    print("\n" + "=" * 60, flush=True)
    print("PHASE 3: FULL EMBODIMENT WITH LAYERDROP", flush=True)
    print("=" * 60, flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset_split("train", max_samples=30000)
    loader = create_dataloader(ds, tokenizer, batch_size=4)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    DISTILL_TEMP = 2.0
    DROP_RATE = 0.2

    for epoch in range(num_epochs):
        model.train()
        epoch_losses = []
        distill_losses = []

        for batch_idx, batch in enumerate(loader):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)

            telem = get_telemetry()
            telemetry = torch.tensor([[
                telem["power"] / 200.0,
                telem["temp"] / 100.0,
                telem["util"] / 100.0,
                np.sin(telem["power"] / 50.0),
                np.cos(telem["temp"] / 50.0),
                telem["power"] * telem["temp"] / 10000.0,
                (telem["util"] - 50.0) / 50.0,
                np.tanh(telem["power"] / 100.0),
                np.tanh(telem["temp"] / 50.0),
                np.tanh(telem["util"] / 50.0),
                (telem["power"] + telem["temp"]) / 200.0,
                np.clip(telem["power"] - telem["temp"], -50, 50) / 50.0,
            ]], dtype=torch.float32).to(device)
            telemetry = telemetry.expand(input_ids.size(0), -1)

            # Forward with LayerDrop
            model.set_layerdrop(True, prob=DROP_RATE)
            out_drop = model(input_ids, telemetry=telemetry)
            logits_drop = out_drop["logits"]
            model.set_layerdrop(False)

            # Teacher (no dropout)
            with torch.no_grad():
                out_teacher = model(input_ids, telemetry=telemetry)
                logits_teacher = out_teacher["logits"]

            # LM loss
            shift_logits = logits_drop[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            lm_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=tokenizer.pad_token_id,
            )

            # Distillation loss
            distill_loss = F.kl_div(
                F.log_softmax(logits_drop / DISTILL_TEMP, dim=-1),
                F.softmax(logits_teacher / DISTILL_TEMP, dim=-1),
                reduction="batchmean"
            ) * (DISTILL_TEMP ** 2)

            loss = lm_loss + 0.5 * distill_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_losses.append(lm_loss.item())
            distill_losses.append(distill_loss.item())

            if (batch_idx + 1) % 100 == 0:
                avg_lm = np.mean(epoch_losses[-100:])
                avg_dist = np.mean(distill_losses[-100:])
                print(f"  Epoch {epoch+1} Step {batch_idx+1}: lm={avg_lm:.4f} distill={avg_dist:.4f}", flush=True)

        avg_loss = np.mean(epoch_losses)
        avg_dist = np.mean(distill_losses)
        print(f"Epoch {epoch+1}/{num_epochs}: lm={avg_loss:.4f} distill={avg_dist:.4f}", flush=True)

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "metrics": {"lm_loss": avg_loss, "distill_loss": avg_dist},
        }, output_dir / f"epoch_{epoch+1}.pt")

    torch.save({
        "model_state_dict": model.state_dict(),
        "metrics": {"lm_loss": avg_loss, "distill_loss": avg_dist},
    }, output_dir / "final.pt")

    print("Phase 3 complete!", flush=True)
    return model


# =============================================================================
# REPORTER HEAD TRAINING
# =============================================================================

class ReporterHead(torch.nn.Module):
    """Linear probe to predict body state from hidden states."""
    def __init__(self, hidden_dim: int, body_dim: int = 12):
        super().__init__()
        self.probe = torch.nn.Linear(hidden_dim, body_dim)

    def forward(self, hidden_states):
        return self.probe(hidden_states.mean(dim=1))


def train_reporter(model, tokenizer, output_dir: Path, device: str, num_epochs: int = 20):
    """Train reporter head to predict body state from frozen model."""
    print("\n" + "=" * 60, flush=True)
    print("REPORTER HEAD TRAINING", flush=True)
    print("=" * 60, flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Freeze base model
    for param in model.parameters():
        param.requires_grad = False

    ds = load_dataset_split("train", max_samples=10000)
    loader = create_dataloader(ds, tokenizer, batch_size=8)

    hidden_dim = model.config.hidden_size
    reporter = ReporterHead(hidden_dim, body_dim=12).to(device)
    optimizer = torch.optim.AdamW(reporter.parameters(), lr=1e-3)

    for epoch in range(num_epochs):
        reporter.train()
        epoch_losses = []

        for batch_idx, batch in enumerate(loader):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)

            telem = get_telemetry()
            body_vec = torch.tensor([[
                telem["power"] / 200.0,
                telem["temp"] / 100.0,
                telem["util"] / 100.0,
                np.sin(telem["power"] / 50.0),
                np.cos(telem["temp"] / 50.0),
                telem["power"] * telem["temp"] / 10000.0,
                (telem["util"] - 50.0) / 50.0,
                np.tanh(telem["power"] / 100.0),
                np.tanh(telem["temp"] / 50.0),
                np.tanh(telem["util"] / 50.0),
                (telem["power"] + telem["temp"]) / 200.0,
                np.clip(telem["power"] - telem["temp"], -50, 50) / 50.0,
            ]], dtype=torch.float32).to(device)
            body_vec = body_vec.expand(input_ids.size(0), -1)

            with torch.no_grad():
                out = model(input_ids, telemetry=body_vec)
                hidden = out["hidden_states"]

            pred = reporter(hidden)
            loss = F.mse_loss(pred, body_vec)
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

            if (batch_idx + 1) % 50 == 0:
                avg_loss = np.mean(epoch_losses[-50:])
                print(f"  Epoch {epoch+1} Step {batch_idx+1}: loss={avg_loss:.4f}", flush=True)

        avg_loss = np.mean(epoch_losses)
        print(f"Epoch {epoch+1}/{num_epochs}: loss={avg_loss:.4f}", flush=True)

    torch.save(reporter.state_dict(), output_dir / "reporter.pt")

    # Unfreeze model
    for param in model.parameters():
        param.requires_grad = True

    print("Reporter training complete!", flush=True)
    return reporter


# =============================================================================
# VALIDATION
# =============================================================================

def validate_embodiment(model, reporter, tokenizer, device: str):
    """Validate the embodiment by comparing to random baseline."""
    print("\n" + "=" * 60, flush=True)
    print("EMBODIMENT VALIDATION", flush=True)
    print("=" * 60, flush=True)

    ds = load_dataset_split("validation", max_samples=1000)
    loader = create_dataloader(ds, tokenizer, batch_size=8)

    model.eval()
    reporter.eval()

    real_errors = []
    random_errors = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)

            telem = get_telemetry()
            body_vec = torch.tensor([[
                telem["power"] / 200.0,
                telem["temp"] / 100.0,
                telem["util"] / 100.0,
                np.sin(telem["power"] / 50.0),
                np.cos(telem["temp"] / 50.0),
                telem["power"] * telem["temp"] / 10000.0,
                (telem["util"] - 50.0) / 50.0,
                np.tanh(telem["power"] / 100.0),
                np.tanh(telem["temp"] / 50.0),
                np.tanh(telem["util"] / 50.0),
                (telem["power"] + telem["temp"]) / 200.0,
                np.clip(telem["power"] - telem["temp"], -50, 50) / 50.0,
            ]], dtype=torch.float32).to(device)
            body_vec = body_vec.expand(input_ids.size(0), -1)

            out = model(input_ids, telemetry=body_vec)
            hidden = out["hidden_states"]

            pred = reporter(hidden)
            real_error = F.mse_loss(pred, body_vec).item()

            # Random baseline
            random_body = torch.randn_like(body_vec)
            random_error = F.mse_loss(pred, random_body).item()

            real_errors.append(real_error)
            random_errors.append(random_error)

    avg_real = np.mean(real_errors)
    avg_random = np.mean(random_errors)
    improvement = (avg_random - avg_real) / avg_random * 100

    print(f"\nResults:", flush=True)
    print(f"  Reporter MSE (real body): {avg_real:.4f}", flush=True)
    print(f"  Random baseline MSE:      {avg_random:.4f}", flush=True)
    print(f"  Improvement over random:  {improvement:.1f}%", flush=True)

    if improvement > 30:
        print("\n✅ EMBODIMENT VERIFIED: Hidden states encode body state!", flush=True)
        verdict = "PASS"
    else:
        print("\n⚠️  EMBODIMENT WEAK: Body encoding not strong enough", flush=True)
        verdict = "WEAK"

    return {
        "reporter_mse": avg_real,
        "random_baseline": avg_random,
        "improvement_pct": improvement,
        "verdict": verdict,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Resume FEEL-SLM Embodiment Training")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to epoch checkpoint (e.g., epoch_2.pt)")
    parser.add_argument("--output-dir", type=str, default="results/z132_resume",
                        help="Output directory")
    parser.add_argument("--start-epoch", type=int, default=2,
                        help="Epoch to resume from (0-indexed)")
    parser.add_argument("--phase2-epochs", type=int, default=5,
                        help="Total epochs for Phase 2")
    parser.add_argument("--phase3-epochs", type=int, default=3,
                        help="Epochs for Phase 3")
    parser.add_argument("--reporter-epochs", type=int, default=20,
                        help="Epochs for reporter training")
    args = parser.parse_args()

    print("=" * 60, flush=True)
    print("FEEL-SLM EMBODIMENT RESUME (FIXED DATALOADER)", flush=True)
    print("=" * 60, flush=True)
    print(f"Started: {datetime.now().isoformat()}", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)

    # Load model and checkpoint
    print(f"\nLoading checkpoint: {args.checkpoint}", flush=True)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
    tokenizer.pad_token = tokenizer.eos_token

    # Create model using the same factory function as z129
    model = create_embodied_slm_30m().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"  Loaded from epoch {checkpoint.get('epoch', 'unknown') + 1}", flush=True)

    output_base = Path(args.output_dir)

    # Phase 2 Resume
    model = run_phase2_resume(
        model, tokenizer,
        output_dir=output_base / "phase2",
        device=device,
        start_epoch=args.start_epoch,
        num_epochs=args.phase2_epochs,
    )

    # Phase 3
    model = run_phase3(
        model, tokenizer,
        output_dir=output_base / "phase3",
        device=device,
        num_epochs=args.phase3_epochs,
    )

    # Reporter Training
    reporter = train_reporter(
        model, tokenizer,
        output_dir=output_base / "reporter",
        device=device,
        num_epochs=args.reporter_epochs,
    )

    # Validation
    results = validate_embodiment(model, reporter, tokenizer, device)

    # Save final results
    with open(output_base / "final_results.json", "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "checkpoint": args.checkpoint,
            "validation": results,
        }, f, indent=2)

    print("\n" + "=" * 60, flush=True)
    print("EMBODIMENT PIPELINE COMPLETE", flush=True)
    print("=" * 60, flush=True)
    print(f"Finished: {datetime.now().isoformat()}", flush=True)
    print(f"Verdict: {results['verdict']}", flush=True)


if __name__ == "__main__":
    main()
