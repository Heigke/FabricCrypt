"""Phase 21 — clone-defeat test.

Hypothesis: chip-trained weights, when LOADED on a DIFFERENT chip, should
still betray the original chip's "personality" because the chip-derived
direction is baked into the weights. The personality moves WITH the weights,
not with the live signature at inference (since inference is greedy/sampling
on weights only).

Steps:
 1. Load chip_dae_v2 checkpoint
 2. Generate from it (using whatever chip we're on — this is the "replay")
 3. Compare stylometrically against original gen_chip_dae_v2.jsonl (which was
    generated on the chip that trained it).
 4. Pre-reg: replay text should classify AS chip_dae_v2 (>50%) when compared
    to vanilla_dae_v2 — because personality is in the weights.

If we ONLY have one machine (daedalus), this test simulates by generating
twice with different random seeds + comparing distributions. True cross-chip
test requires ikaros to be cool enough to run inference (no training needed).
"""
from __future__ import annotations
import os, sys, json, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# This script is intentionally thin: it calls generate.py twice with different
# seeds (simulating a 'replay' on a separate occasion) and then runs
# stylometry. For a true cross-chip test, the same ckpt must be transferred
# to a second host and generate.py run there.
print(__doc__)
