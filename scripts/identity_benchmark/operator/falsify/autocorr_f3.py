"""F3 autocorrelation on per-rep modal-distance time-series."""
from __future__ import annotations
import struct, sys, json
import numpy as np
from pathlib import Path

def load(path: Path):
    raw = path.read_bytes()
    M, R, hl = struct.unpack('iii', raw[:12])
    host = raw[12:12+hl].decode()
    body = raw[12+hl:]
    arr = np.frombuffer(body, dtype=np.float32).reshape(R, M)
    return host, arr

def popcount_xor(a, b):
    ai = a.view(np.uint32); bi = b.view(np.uint32)
    xor = ai ^ bi
    pc = np.zeros_like(xor)
    v = xor.copy()
    for _ in range(32):
        pc += (v & 1); v >>= 1
    return pc.sum()  # total bits differing in this rep vs modal

def main():
    path = Path(sys.argv[1])
    host, A = load(path)
    R, M = A.shape
    # Compute per-output modal bit-pattern across reps
    bits = A.view(np.uint32)
    from collections import Counter
    modal = np.zeros(M, dtype=np.uint32)
    for m in range(M):
        modal[m] = Counter(bits[:, m].tolist()).most_common(1)[0][0]
    # Per-rep distance to modal (total bits differing)
    series = np.zeros(R)
    for r in range(R):
        xor = bits[r] ^ modal
        s = 0
        for b in range(32):
            s += int(((xor >> b) & 1).sum())
        series[r] = s

    # Autocorrelation at lag 1
    s = series - series.mean()
    denom = (s*s).sum()
    if denom < 1e-12:
        ac1 = 0.0
    else:
        ac1 = float((s[:-1] * s[1:]).sum() / denom)
    # Also lag 2,3 for context
    def acl(lag):
        if denom < 1e-12: return 0.0
        return float((s[:-lag]*s[lag:]).sum()/denom) if lag<R else 0.0

    out = {
        'host': host,
        'reps': R,
        'outputs_M': M,
        'series_per_rep_bits_diff_from_modal': series.tolist(),
        'series_mean': float(series.mean()),
        'series_std': float(series.std()),
        'autocorr_lag1': ac1,
        'autocorr_lag2': acl(2),
        'autocorr_lag3': acl(3),
    }
    print(json.dumps(out, indent=2))

if __name__ == '__main__':
    main()
