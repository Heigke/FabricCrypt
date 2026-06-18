"""F8 — CPU sanity baseline. Compute same matmul on CPU with deterministic
numpy. Ikaros and daedalus runs should be 100% bit-identical (sanity check).
"""
from __future__ import annotations
import hashlib, json, struct, socket, sys
from pathlib import Path
import numpy as np

OUT_DIR = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/IDENTITY_OPERATOR_2026-05-31/falsify_v2/F8")
OUT_DIR.mkdir(parents=True, exist_ok=True)

M, K = 64, 4096
# Mirror the LCG-based deterministic inputs of divergent_matmul.hip
W = np.zeros(M * K, dtype=np.float32)
for i in range(M * K):
    s = np.uint32(np.uint32(i) * np.uint32(1103515245) + np.uint32(12345))
    v = ((int(s) >> 8) & 0xFFFF) / 65535.0 - 0.5
    if ((int(s) >> 24) & 0xFF) < 3:
        v *= 1e-38
    W[i] = v
W = W.reshape(M, K)
x = np.zeros(K, dtype=np.float32)
for i in range(K):
    s = np.uint32(np.uint32(i + 7) * np.uint32(22695477) + np.uint32(1))
    x[i] = ((int(s) >> 8) & 0xFFFF) / 65535.0 - 0.5

# Use float64 accumulation cast back to float32 for STRICT determinism
y64 = W.astype(np.float64) @ x.astype(np.float64)
y = y64.astype(np.float32)

# Hash of result bits
h = hashlib.sha256(y.tobytes()).hexdigest()
host = socket.gethostname()
np.save(OUT_DIR / f"{host}_cpu_y.npy", y)
result = {
    "F8_cpu_baseline": {
        "host": host,
        "M": M, "K": K,
        "y_sha256": h,
        "y_first5": y[:5].tolist(),
        "y_input_hash_W": hashlib.sha256(W.tobytes()).hexdigest()[:16],
        "y_input_hash_x": hashlib.sha256(x.tobytes()).hexdigest()[:16],
    }
}
(OUT_DIR / f"{host}_cpu.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
