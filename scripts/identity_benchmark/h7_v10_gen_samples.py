"""H7 v10 — qualitative generation under real vs wrong substrate.

Evidences goal criterion 4 ("writes good text but influenced by identity") and
that a wrong signal breaks it. Loads the v10 best checkpoint, streams LIVE ikaros
substrate, and greedily generates a continuation under {real, knock, shuffle, zero}.

Run: sudo HSA_OVERRIDE_GFX_VERSION=11.0.0 venv/bin/python h7_v10_gen_samples.py
"""
from __future__ import annotations
import sys, time, socket
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).parent))
from substrate_realtime_v3 import SubstrateStateV3
from h7_rooted_lm_v4a import GlobalNorm, SubstrateEncoderV4, K_TOKENS, WIN_LEN, BASE_MODEL, STATS
from h7_embodied_v8 import FilmEmbodiedSmolLM
from h7_knockoff_kl_probe import make_knockoff
from h7_embodied_v7 import temporal_shuffle, encode
from transformers import AutoModelForCausalLM, AutoTokenizer

HOST = socket.gethostname()
ROOT = Path(__file__).resolve().parents[2]
CKPT = ROOT / "results/IDENTITY_EMBODIED_V10_2026-06-10" / f"v10_best_{HOST}.pt"
SEED = 131
PROMPTS = ["The old house at the edge of town", "She opened the letter and"]


@torch.no_grad()
def gen(model, ids, S, tok, n=40):
    cur = ids.clone()
    for _ in range(n):
        logits = model(cur, substrate_tokens=S).logits[:, -1, :]
        nxt = logits.argmax(-1, keepdim=True)
        cur = torch.cat([cur, nxt], dim=1)
        if nxt.item() == tok.eos_token_id:
            break
    return tok.decode(cur[0, ids.shape[1]:], skip_special_tokens=True)


def main():
    rng = np.random.default_rng(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = FilmEmbodiedSmolLM().to(device).eval()
    se = SubstrateEncoderV4(d_emb=model.d, K=K_TOKENS).to(device).eval()
    ck = torch.load(CKPT, map_location=device, weights_only=False)
    model.film.load_state_dict(ck["film"]); se.load_state_dict(ck["se"])
    msd = dict(model.named_parameters())
    for k, v in ck.get("lora", {}).items():
        if k in msd: msd[k].data.copy_(v.to(device))
    print(f"loaded v10 ckpt step={ck.get('step')}")
    norm = GlobalNorm(STATS)

    state = SubstrateStateV3(hz_target=500); state.start(); time.sleep(2.0)
    w = state.latest_window(length=WIN_LEN)
    conds = {
        "REAL  (own live die)": encode(se, norm, w, device),
        "KNOCK (spoof)":        encode(se, norm, make_knockoff(w, rng), device),
        "SHUFFLE (wrong dyn)":  encode(se, norm, temporal_shuffle(w, rng), device),
        "ZERO  (no signal)":    torch.zeros(1, K_TOKENS, model.d, device=device),
    }
    state.stop()

    for p in PROMPTS:
        ids = tok(p, return_tensors="pt").input_ids.to(device)
        print("\n" + "=" * 70)
        print(f"PROMPT: {p!r}")
        print("=" * 70)
        for name, S in conds.items():
            txt = gen(model, ids, S, tok)
            print(f"\n[{name}]\n  {txt.strip()!r}")


if __name__ == "__main__":
    main()
