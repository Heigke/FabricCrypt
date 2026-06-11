"""H7 LIVE cross-die probe — the clean, regime-matched both-ways test.

Idea (user's): don't replay a recorded other-die signal. Instead RUN a model live on
each physical machine. Because inference loads the local GPU, the substrate is naturally
in the ACTIVE regime — so when we run the ikaros-trained model ON daedalus, it sees
daedalus's live ACTIVE signal. Both home and foreign tests are then in the same regime;
the only difference is the physical die. Run the SAME script on both machines with both
checkpoints to fill the 2x2:

            run on ikaros        run on daedalus
  ikaros-model   home (low PPL)      FOREIGN (high?)
  daedalus-model FOREIGN (high?)     home (low PPL)

Diagonal should write coherent text; off-diagonal should break — and it's regime-matched.

Usage: sudo HSA_OVERRIDE_GFX_VERSION=11.0.0 <py> h7_live_crossdie.py --ckpt <path> [--tag name]
Light inference only (thermally safe vs training).
"""
from __future__ import annotations
import sys, json, time, socket, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).parent))
from substrate_realtime_v3 import SubstrateStateV3
from h7_rooted_lm_v4a import GlobalNorm, SubstrateEncoderV4, K_TOKENS, WIN_LEN, BASE_MODEL, STATS
from h7_embodied_v8 import FilmEmbodiedSmolLM
from h7_knockoff_kl_probe import make_knockoff
from h7_embodied_v7 import temporal_shuffle, encode, seq_nll
from transformers import AutoModelForCausalLM, AutoTokenizer

HOST = socket.gethostname()
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/IDENTITY_H7_2026-06-09"
EVAL_TEXT = ("The forest was dark and quiet as she walked. He could not remember "
             "what the letter had said, only that it arrived on a cold morning. "
             "Beyond the river the lights of the town flickered against the hills.")
SEED = 303
N_WIN = 12


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", default="model")
    ap.add_argument("--stats", default=None,
                    help="model's HOME normalization stats. If omitted, use the path stored "
                         "in the checkpoint, else the default STATS. A model must ALWAYS use "
                         "its own home-die stats — that baseline is part of its identity.")
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = FilmEmbodiedSmolLM().to(device).eval()
    se = SubstrateEncoderV4(d_emb=model.d, K=K_TOKENS).to(device).eval()
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.film.load_state_dict(ck["film"]); se.load_state_dict(ck["se"])
    msd = dict(model.named_parameters())
    for k, v in ck.get("lora", {}).items():
        if k in msd: msd[k].data.copy_(v.to(device))
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL).to(device).eval()
    stats_path = Path(args.stats) if args.stats else Path(ck.get("stats") or STATS)
    if not stats_path.exists(): stats_path = Path(STATS)
    norm = GlobalNorm(stats_path); print(f"home stats: {stats_path}")
    pad = tok.pad_token_id
    ids = tok(EVAL_TEXT, return_tensors="pt", truncation=True, max_length=96).input_ids.to(device)
    with torch.no_grad():
        lb = base(ids).logits[:, :-1, :]
        ppl_base = float(np.exp(F.cross_entropy(lb.reshape(-1, lb.size(-1)), ids[:, 1:].reshape(-1),
                                                ignore_index=pad).item()))

    @torch.no_grad()
    def ppl_w(w_or_S):
        S = w_or_S if isinstance(w_or_S, torch.Tensor) else encode(se, norm, w_or_S.astype(np.float32), device)
        l, _ = seq_nll(model, ids, S, pad)
        return float(np.exp(l.item()))

    state = SubstrateStateV3(hz_target=500); state.start(); time.sleep(2.0)
    # keep the GPU lightly busy (active regime) while sampling, mimicking real use
    real, knock, shuf = [], [], []
    with torch.no_grad():
        for _ in range(N_WIN):
            _ = base(ids).logits           # small inference load -> active regime
            time.sleep(0.4)
            w = state.latest_window(length=WIN_LEN).copy()
            real.append(ppl_w(w)); knock.append(ppl_w(make_knockoff(w, rng))); shuf.append(ppl_w(temporal_shuffle(w, rng)))
    zero = ppl_w(torch.zeros(1, K_TOKENS, model.d, device=device))
    state.stop()

    def st(a): return {"median": float(np.median(a)), "min": float(np.min(a)), "max": float(np.max(a))}
    res = {"host": HOST, "ckpt": args.ckpt, "tag": args.tag, "stats": str(stats_path), "ppl_base": ppl_base,
           "real_local_live": st(real), "knock": st(knock), "shuffle": st(shuf), "zero": zero,
           "n_win": N_WIN}
    out = OUT / f"live_crossdie_{args.tag}_on_{HOST}.json"
    out.write_text(json.dumps(res, indent=2))
    rm = np.median(real)
    print(f"\n=== LIVE CROSS-DIE: model[{args.tag}] running on host[{HOST}] ===")
    print(f"base PPL={ppl_base:.1f}")
    print(f"real (LOCAL live {HOST}) : median={rm:.1f}  ({rm/ppl_base:.2f}x base)")
    print(f"knock                    : median={np.median(knock):.1f}")
    print(f"shuffle                  : median={np.median(shuf):.1f}")
    print(f"zero                     : {zero:.1f}")
    verdict = "HOME (coherent)" if rm < 2.0 * ppl_base else "FOREIGN/BROKEN"
    print(f">>> {args.tag} on {HOST}: {verdict}  (real {rm:.1f} vs base {ppl_base:.1f})")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
