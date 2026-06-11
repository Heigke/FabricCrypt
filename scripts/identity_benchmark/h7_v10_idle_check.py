"""Is the 'idle breaks' finding real, or a fresh-buffer / first-capture artifact?

Capture idle windows AFTER a long warmup, many of them, and interleave a model-
inference burst (the training-time regime) to compare. Reports the full distribution.
"""
from __future__ import annotations
import sys, time, socket
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).parent))
from substrate_realtime_v3 import SubstrateStateV3
from h7_rooted_lm_v4a import GlobalNorm, SubstrateEncoderV4, K_TOKENS, WIN_LEN, BASE_MODEL, STATS
from h7_embodied_v8 import FilmEmbodiedSmolLM
from h7_embodied_v7 import encode, seq_nll
from transformers import AutoModelForCausalLM, AutoTokenizer

HOST = socket.gethostname()
ROOT = Path(__file__).resolve().parents[2]
V10 = ROOT / "results/IDENTITY_EMBODIED_V10_2026-06-10"
CKPT = V10 / f"v10_best_{HOST}.pt"
EVAL_TEXT = ("The forest was dark and quiet as she walked. He could not remember "
             "what the letter had said, only that it arrived on a cold morning. "
             "Beyond the river the lights of the town flickered against the hills.")


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", default=str(CKPT)); _a = ap.parse_args()
    ckpt_path = _a.ckpt
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = FilmEmbodiedSmolLM().to(device).eval()
    se = SubstrateEncoderV4(d_emb=model.d, K=K_TOKENS).to(device).eval()
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.film.load_state_dict(ck["film"]); se.load_state_dict(ck["se"])
    msd = dict(model.named_parameters())
    for k, v in ck.get("lora", {}).items():
        if k in msd: msd[k].data.copy_(v.to(device))
    norm = GlobalNorm(STATS); pad = tok.pad_token_id
    ids = tok(EVAL_TEXT, return_tensors="pt", truncation=True, max_length=96).input_ids.to(device)

    @torch.no_grad()
    def ppl_w(w):
        l, _ = seq_nll(model, ids, encode(se, norm, w.astype(np.float32), device), pad)
        return float(np.exp(l.item()))

    state = SubstrateStateV3(hz_target=500); state.start()
    print("long warmup (8s)..."); time.sleep(8.0)

    # 20 genuinely-idle windows (no model inference between captures)
    idle = []
    for _ in range(20):
        time.sleep(0.5); idle.append(ppl_w(state.latest_window(length=WIN_LEN)))

    # 20 windows captured WHILE doing model inference (training-time regime)
    active = []
    for _ in range(20):
        _ = ppl_w(state.latest_window(length=WIN_LEN))   # extra inference = load
        _ = model(ids).logits
        time.sleep(0.2); active.append(ppl_w(state.latest_window(length=WIN_LEN)))
    state.stop()

    def summ(a):
        a = np.array(a)
        return (f"median={np.median(a):.1f} mean={np.mean(a):.1f} "
                f"min={a.min():.1f} max={a.max():.1f}  frac_coherent(<60)={np.mean(a<60):.2f}")
    print(f"\nIDLE (post-warmup, no inference): {summ(idle)}")
    print(f"ACTIVE (inference between caps):  {summ(active)}")
    print(f"\nidle PPLs: {[round(x,1) for x in idle]}")


if __name__ == "__main__":
    main()
