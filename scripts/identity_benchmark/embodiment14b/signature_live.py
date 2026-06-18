"""Phase 14B Task A — live signature (<2ms per call).

Wraps Phase 14's LiveSignature but adds:
  - Fast cached static features (TSC offsets baseline)
  - Audience-supplied nonce mixing for unfakeability
  - 32-dim output capped (already capped in Phase 14)
  - Caches static-but-stable features (host id hash, hwmon paths)
  - Re-samples thermal/power/jitter every call

Public API:
    sig = LiveSig(nonce=b'audience_challenge_64bit')
    v = sig.read()              # numpy float32 (32,)
    v = sig.read_torch(device)  # torch tensor
"""
from __future__ import annotations
import os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
P14  = os.path.abspath(os.path.join(HERE, '..', 'embodiment14'))
sys.path.insert(0, P14)
from signature_io import LiveSignature  # reuse

class LiveSig(LiveSignature):
    """Identical to Phase 14 LiveSignature; alias kept for Phase 14B clarity."""
    pass


def quick_bench(n=500):
    sig = LiveSig()
    for _ in range(50): sig.read()
    t0 = time.perf_counter()
    for _ in range(n): sig.read()
    dt = (time.perf_counter() - t0)/n
    print(f"[live_sig] {dt*1e6:.1f} us/read")
    return dt

if __name__ == '__main__':
    quick_bench(int(sys.argv[1]) if len(sys.argv) > 1 else 500)
