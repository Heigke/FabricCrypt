"""Generate the per-die global_substrate_stats.npz (robust median/MAD per channel).

The embodied LM's GlobalNorm standardizes each substrate window by these constants and
soft-clamps to [-8,8] via tanh. If the stats come from a DIFFERENT die, the local signal
saturates the clamp (channels go flat) and the encoder sees no dynamics -> the model can't
learn its die's identity. So each machine must compute its OWN stats from its OWN live
signal, sampled across the same idle+active mix the trainer uses.

Writes to results/IDENTITY_H7_2026-06-09/global_substrate_stats.npz (the STATS path).
Run as root (substrate reads /dev/mem) with HSA_OVERRIDE_GFX_VERSION=11.0.0.
"""
from __future__ import annotations
import sys, time, socket
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).parent))
from substrate_realtime_v3 import SubstrateStateV3
from h7_rooted_lm_v4a import STATS, WIN_LEN, N_CHANNELS, BASE_MODEL
from transformers import AutoModelForCausalLM, AutoTokenizer

HOST = socket.gethostname()
N_IDLE = 60       # idle windows (no GPU load)
N_ACTIVE = 60     # windows captured while running base-model inference (active regime)
WARMUP = 8.0


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(STATS), help="output .npz path for this die's stats")
    out_path = Path(ap.parse_args().out)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL).to(device).eval()
    ids = tok("The quick brown fox jumps over the lazy dog by the river at dawn.",
              return_tensors="pt").input_ids.to(device)

    state = SubstrateStateV3(hz_target=500); state.start()
    print(f"[{HOST}] warmup {WARMUP}s ..."); time.sleep(WARMUP)

    chunks = []
    for _ in range(N_IDLE):
        time.sleep(0.25); chunks.append(state.latest_window(length=WIN_LEN).copy())
    print(f"[{HOST}] idle windows: {len(chunks)}")
    with torch.no_grad():
        for _ in range(N_ACTIVE):
            _ = base(ids).logits            # GPU load -> active regime
            time.sleep(0.2); chunks.append(state.latest_window(length=WIN_LEN).copy())
    state.stop()
    print(f"[{HOST}] total windows: {len(chunks)}")

    W = np.concatenate(chunks, axis=0).astype(np.float64)   # (T_total, C)
    assert W.shape[1] == N_CHANNELS, (W.shape, N_CHANNELS)
    median = np.median(W, axis=0)
    mad = np.median(np.abs(W - median), axis=0) * 1.4826     # ~std-consistent MAD
    mad = np.where(mad < 1e-3, 1.0, mad)                     # floor (match GlobalNorm)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, median=median.astype(np.float32), mad=mad.astype(np.float32))
    print(f"[{HOST}] saved {out_path}")
    print("median:", np.round(median, 3))
    print("mad   :", np.round(mad, 3))


if __name__ == "__main__":
    main()
