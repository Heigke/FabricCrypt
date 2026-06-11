"""H7 graded-correlation probe — is v11's output a GRADED function of the live signal,
or only a kill-switch?

v11 trains an aux objective tying the channel-GRAD_CHANNEL dynamics amplitude (feat) of the
live window to the model's output entropy: target_ent = base_ent + GRAD_BETA * feat. This
probe measures, over many live windows on the HOME die, the actual Pearson correlation
between feat and the model's output entropy.

Controls:
  - shuffle: feed the TEMPORALLY-SHUFFLED window (feat is permutation-invariant so unchanged,
    but the dynamics the encoder reads are destroyed). If the coupling is genuinely read from
    live dynamics, entropy decouples from feat -> r collapses.
  - pairing-null: correlate feat against a randomly re-paired entropy array -> r ~ 0 (sanity).

PASS if (a) |r_real| is meaningful (>~0.3) and >=3x |r_shuffle|, and (b) text stays coherent
(median real PPL < COHERENT). Run as root with HSA_OVERRIDE_GFX_VERSION=11.0.0.
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
from h7_embodied_v7 import temporal_shuffle, encode, seq_nll
from transformers import AutoModelForCausalLM, AutoTokenizer

HOST = socket.gethostname()
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/IDENTITY_H7_2026-06-09"
EVAL_TEXT = ("The forest was dark and quiet as she walked. He could not remember "
             "what the letter had said, only that it arrived on a cold morning. "
             "Beyond the river the lights of the town flickered against the hills.")
GRAD_CHANNEL = 4          # MUST match v11 training
COHERENT = 60.0
SEED = 707


def pearson(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if a.std() < 1e-9 or b.std() < 1e-9: return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--stats", default=None)
    ap.add_argument("--tag", default="model")
    ap.add_argument("--n", type=int, default=240)
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

    def feat_of(w):
        # v12: feat on the NORMALIZED window (matches training). On the RAW window the
        # cross-channel scale disparity made this degenerate (constant -0.762).
        z = norm(w)
        return float(np.tanh(np.std(z[:, GRAD_CHANNEL]) / (np.std(z) + 1e-6) - 1.0))

    @torch.no_grad()
    def ent_ppl(w):
        S = encode(se, norm, w.astype(np.float32), device)
        l, o = seq_nll(model, ids, S, pad)
        logits = o.logits[:, :-1, :]
        p = F.softmax(logits, dim=-1)
        ent = float((-(p * torch.log(p + 1e-9)).sum(-1)).mean().item())
        return ent, float(np.exp(l.item()))

    state = SubstrateStateV3(hz_target=500); state.start()
    print("warmup 6s..."); time.sleep(6.0)
    feats, ent_real, ent_shuf, ppl_real = [], [], [], []
    with torch.no_grad():
        for i in range(args.n):
            _ = base(ids).logits                 # keep die in active regime
            time.sleep(0.12)
            w = state.latest_window(length=WIN_LEN).copy()
            f = feat_of(w)
            er, pr = ent_ppl(w)
            es, _ = ent_ppl(temporal_shuffle(w, rng))
            feats.append(f); ent_real.append(er); ent_shuf.append(es); ppl_real.append(pr)
            if (i + 1) % 40 == 0:
                print(f"  {i+1}/{args.n}  r_real={pearson(feats, ent_real):.3f} "
                      f"r_shuf={pearson(feats, ent_shuf):.3f} medPPL={np.median(ppl_real):.1f}")
    state.stop()

    r_real = pearson(feats, ent_real)
    r_shuf = pearson(feats, ent_shuf)
    # pairing null: re-pair feat with a permuted entropy index
    perm = rng.permutation(len(feats))
    r_null = pearson(feats, np.array(ent_real)[perm])
    # least-squares slope feat->entropy (compare to trained GRAD_BETA=0.6)
    slope = float(np.polyfit(feats, ent_real, 1)[0]) if np.std(feats) > 1e-9 else 0.0
    med_ppl = float(np.median(ppl_real))
    ratio = abs(r_real) / max(abs(r_shuf), 1e-3)
    coherent = med_ppl < COHERENT
    passed = (abs(r_real) > 0.3) and (ratio >= 3.0) and coherent

    res = {"host": HOST, "ckpt": args.ckpt, "tag": args.tag, "stats": str(stats_path),
           "n": args.n, "r_real": r_real, "r_shuffle": r_shuf, "r_pairing_null": r_null,
           "ratio_real_over_shuffle": ratio, "slope_feat_to_entropy": slope,
           "grad_beta_trained": 0.6, "median_ppl_real": med_ppl, "coherent": coherent,
           "feat_std": float(np.std(feats)), "feat_range": [float(np.min(feats)), float(np.max(feats))],
           "PASS": passed}
    out = OUT / f"graded_probe_{args.tag}_on_{HOST}.json"
    out.write_text(json.dumps(res, indent=2))
    print(f"\n=== GRADED PROBE: model[{args.tag}] on host[{HOST}], n={args.n} ===")
    print(f"feat range {res['feat_range']} std {res['feat_std']:.3f}")
    print(f"r(feat, entropy)            REAL    = {r_real:+.3f}")
    print(f"r(feat, entropy)            SHUFFLE = {r_shuf:+.3f}   (dynamics destroyed)")
    print(f"r(feat, entropy)            NULL    = {r_null:+.3f}   (re-paired)")
    print(f"ratio real/shuffle          = {ratio:.2f}  (need >=3)")
    print(f"slope feat->entropy         = {slope:+.3f}  (trained GRAD_BETA=0.60)")
    print(f"median real PPL             = {med_ppl:.1f}  (coherent<{COHERENT}: {coherent})")
    print(f">>> {'PASS' if passed else 'FAIL'} — graded coupling {'CONFIRMED' if passed else 'not established'}")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
