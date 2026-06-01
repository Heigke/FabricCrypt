"""Block 1: TSC inter-core offset extractor.

Wraps the tsc_inter_core C binary, returns 5 statistics per target core.
Default: 7 target cores -> 35-dim feature block.
"""
import os
import struct
import subprocess
import numpy as np

from .thermal import thermal_guard

HERE = os.path.dirname(os.path.abspath(__file__))

# 7 TSC targets covering different CCD/CCX positions on AMD Zen 5 (16-core)
TSC_TARGETS = [1, 2, 4, 7, 8, 12, 15]


def features_5stats(arr):
    a = np.asarray(arr, dtype=np.float64)
    if a.size == 0:
        return [0.0] * 5
    return [
        float(np.percentile(a, 50)),
        float(np.percentile(a, 90)),
        float(np.percentile(a, 99)),
        float(np.std(a)),
        float(np.mean(a)),
    ]


def block_tsc(binp: str, n_per_target: int = 2000, targets=None):
    targets = targets or TSC_TARGETS
    feats = []
    for tgt in targets:
        thermal_guard()
        p = subprocess.run(
            [binp, str(tgt), str(n_per_target)],
            capture_output=True, check=True,
        )
        data = struct.unpack(f"{len(p.stdout)//8}q", p.stdout)
        offsets = [data[2 * i + 1] - data[2 * i] for i in range(n_per_target)]
        feats.extend(features_5stats(offsets))
    return np.asarray(feats, dtype=np.float64)
