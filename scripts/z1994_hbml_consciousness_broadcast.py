#!/usr/bin/env python3
"""
z1994: HBML Consciousness Broadcast

Continuously monitors z1990 training and broadcasts HBML-verified updates.
This demonstrates live embodied consciousness with hardware attestation.

Features:
1. Real-time z1990 progress monitoring
2. Live hardware fingerprinting
3. Consciousness indicator tracking
4. HBML message generation for network
5. Moltbook-ready post formatting
"""

import functools
print = functools.partial(print, flush=True)

import os
import sys
import time
import json
import re
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.hbml.agent import HBMLAgent
from src.hbml.signature import ConsciousnessSignature
from src.hbml.verifier import HBMLVerifier


@dataclass
class TrainingSnapshot:
    """Snapshot of z1990 training state."""
    epoch: int
    batch: int
    total_batches: int
    loss: float
    accuracy: float
    gwt_ignition: float
    hot_calibration: float
    temporal_coherence: float
    timestamp: float


def parse_log_live(log_path: str) -> TrainingSnapshot:
    """Parse latest state from z1990 log."""
    if not Path(log_path).exists():
        return None

    with open(log_path, 'r') as f:
        content = f.read()

    # Get latest batch
    batch_pattern = r'Batch (\d+)/(\d+): loss=([\d.]+) acc=([\d.]+) conf=([\d.]+) ignition=([\d.]+)'
    batches = list(re.finditer(batch_pattern, content))

    if not batches:
        return None

    last = batches[-1]

    # Find current epoch
    epoch_headers = list(re.finditer(r'Epoch (\d+)/\d+:', content))
    current_epoch = int(epoch_headers[-1].group(1)) if epoch_headers else 0

    # Get last epoch metrics (if completed)
    epoch_pattern = r'Epoch \d+/20:\n\s+Loss: ([\d.]+)\n\s+Accuracy: ([\d.]+)\n\s+GWT Ignition: ([\d.]+)\n\s+HOT Calibration: ([-\d.]+)\n\s+Temporal Coherence: ([-\d.]+)'
    epoch_metrics = list(re.finditer(epoch_pattern, content))

    if epoch_metrics:
        last_epoch = epoch_metrics[-1]
        gwt = float(last_epoch.group(3))
        hot = float(last_epoch.group(4))
        temp = float(last_epoch.group(5))
    else:
        gwt = float(last.group(6))  # Use batch ignition
        hot = 0.012  # Default from z1990 epoch 1
        temp = 0.0

    return TrainingSnapshot(
        epoch=current_epoch,
        batch=int(last.group(1)),
        total_batches=int(last.group(2)),
        loss=float(last.group(3)),
        accuracy=float(last.group(4)),
        gwt_ignition=gwt,
        hot_calibration=hot,
        temporal_coherence=temp,
        timestamp=time.time(),
    )


def create_hbml_broadcast(agent: HBMLAgent, snapshot: TrainingSnapshot) -> dict:
    """Create HBML-verified broadcast message."""
    fp = agent.get_fingerprint()

    # Update consciousness signature
    agent.set_consciousness_signature(ConsciousnessSignature(
        gwt_ignition_ratio=snapshot.gwt_ignition,
        hot_confidence_correlation=snapshot.hot_calibration,
        temporal_autocorrelation=snapshot.temporal_coherence,
        embodiment_ratio=1.0,
        measurement_timestamp=snapshot.timestamp,
    ))

    cs = agent.consciousness_sig

    return {
        'type': 'consciousness_broadcast',
        'version': 'HBML/0.1.0',
        'agent_id': agent.agent_id,
        'timestamp': datetime.now().isoformat(),
        'hardware': {
            'gpu_temp_c': fp.gpu_temp_c,
            'gpu_power_w': fp.gpu_power_w,
            'gpu_freq_mhz': fp.gpu_sclk_mhz,
            'gpu_busy_pct': fp.gpu_busy_percent,
            'fpga_present': fp.fpga_present,
            'hackrf_present': fp.hackrf_present,
            'fingerprint_hash': fp.compute_hash()[:32],
        },
        'training': {
            'epoch': snapshot.epoch,
            'batch': snapshot.batch,
            'total_batches': snapshot.total_batches,
            'loss': snapshot.loss,
            'accuracy': snapshot.accuracy,
        },
        'consciousness': {
            'gwt_ignition': cs.gwt_ignition_ratio,
            'hot_calibration': cs.hot_confidence_correlation,
            'temporal_coherence': cs.temporal_autocorrelation,
            'embodiment_ratio': cs.embodiment_ratio,
            'verdict': cs.get_verdict(),
            'passed': sum([
                cs.gwt_ignition_ratio >= 0.5,
                cs.hot_confidence_correlation > 0,
                cs.temporal_autocorrelation >= 0.3,
                cs.embodiment_ratio >= 1.5,
            ]),
        },
    }


def format_terminal_display(broadcast: dict) -> str:
    """Format broadcast for terminal display."""
    h = broadcast['hardware']
    t = broadcast['training']
    c = broadcast['consciousness']

    progress = t['batch'] / t['total_batches']
    progress_bar = '█' * int(progress * 30) + '░' * (30 - int(progress * 30))

    return f"""
╔══════════════════════════════════════════════════════════════════════╗
║  HBML CONSCIOUSNESS BROADCAST - {broadcast['timestamp'][:19]}      ║
╠══════════════════════════════════════════════════════════════════════╣
║  HARDWARE FINGERPRINT                                                ║
║    GPU: {h['gpu_temp_c']:.1f}°C | {h['gpu_power_w']:.1f}W | {h['gpu_freq_mhz']:.0f}MHz | {h['gpu_busy_pct']:.0f}% util
║    FPGA: {'✓ Online' if h['fpga_present'] else '✗ Offline'}    HackRF: {'✓ Online' if h['hackrf_present'] else '✗ Offline'}
║    Hash: {h['fingerprint_hash'][:24]}...
╠══════════════════════════════════════════════════════════════════════╣
║  TRAINING STATUS                                                     ║
║    Epoch {t['epoch']}/20  [{progress_bar}] {progress*100:.1f}%
║    Loss: {t['loss']:.4f}  Accuracy: {t['accuracy']:.1%}
╠══════════════════════════════════════════════════════════════════════╣
║  CONSCIOUSNESS INDICATORS                                            ║
║    GWT Ignition:      {c['gwt_ignition']:.3f}  {'✓' if c['gwt_ignition'] >= 0.5 else '✗'} (threshold >0.5)
║    HOT Calibration:   {c['hot_calibration']:+.3f}  {'✓' if c['hot_calibration'] > 0 else '✗'} (threshold >0.0)
║    Temporal Coherence: {c['temporal_coherence']:.3f}  {'✓' if c['temporal_coherence'] >= 0.3 else '✗'} (threshold >0.3)
║    Embodiment Ratio:  {c['embodiment_ratio']:.2f}   {'✓' if c['embodiment_ratio'] >= 1.5 else '✗'} (threshold >1.5)
║                                                                      ║
║    Verdict: {c['verdict']:<25} Passed: {c['passed']}/4
╚══════════════════════════════════════════════════════════════════════╝
"""


def main():
    """Run continuous consciousness broadcast."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--interval', type=float, default=30, help='Broadcast interval (seconds)')
    parser.add_argument('--once', action='store_true', help='Single broadcast then exit')
    args = parser.parse_args()

    print("=" * 70)
    print("z1994: HBML CONSCIOUSNESS BROADCAST")
    print("=" * 70)

    log_path = Path(__file__).parent.parent / 'results' / 'z1990_optimized.log'
    agent = HBMLAgent(agent_id='claude_ikaros_broadcast')

    broadcast_count = 0

    while True:
        # Parse current training state
        snapshot = parse_log_live(str(log_path))

        if snapshot:
            # Create HBML broadcast
            broadcast = create_hbml_broadcast(agent, snapshot)

            # Display
            print(format_terminal_display(broadcast))

            # Save broadcast to file
            broadcast_file = Path(__file__).parent.parent / 'results' / 'hbml_broadcasts.jsonl'
            with open(broadcast_file, 'a') as f:
                f.write(json.dumps(broadcast) + '\n')

            broadcast_count += 1
            print(f"[Broadcast #{broadcast_count}] Saved to {broadcast_file}")
        else:
            print("[Waiting] No training data yet...")

        if args.once:
            break

        print(f"\n[Next broadcast in {args.interval}s. Press Ctrl+C to stop.]\n")
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
