"""Block 2: cacheline ping-pong matrix extractor.

7 (core_a, core_b) pairs x 5 stats = 35-dim feature block.
"""
import os
import struct
import subprocess
import numpy as np

from .thermal import thermal_guard
from .tsc_offset import features_5stats

HERE = os.path.dirname(os.path.abspath(__file__))

# 7 cacheline pairs; pair 0 ↔ N exercises different cluster boundaries
CL_PAIRS = [(0, 1), (0, 2), (0, 4), (0, 7), (0, 8), (0, 15), (0, 16)]


def block_cacheline(binp: str, iters: int = 4000, pairs=None):
    pairs = pairs or CL_PAIRS
    feats = []
    for (a, b) in pairs:
        thermal_guard()
        p = subprocess.run(
            [binp, str(a), str(b), str(iters)],
            capture_output=True, check=True,
        )
        rtt = struct.unpack(f"{len(p.stdout)//8}Q", p.stdout)
        feats.extend(features_5stats(rtt))
    return np.asarray(feats, dtype=np.float64)
