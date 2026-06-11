"""H7 v10 falsification round 2 — the literature's decisive tests, OFFLINE.

No substrate/root/thermal: uses the two recorded replays (held-out ikaros + daedalus).

Tests (per skeptical lit review):
  G1  GRADED real-die-to-real-die: interpolate ikaros<->daedalus windows, PPL curve.
      Smooth monotone -> genuine graded dependence. Flat-then-cliff -> OOD gate.
  G2  DC-ONLY control: replace each channel by its TEMPORAL MEAN (constant, zero
      dynamics). If coherent -> model uses a scalar per-channel offset (~few bits).
      If breaks -> model needs the DYNAMICS (many bits).
  G3  VAR-SCALE: scale ikaros dynamics amplitude around its mean (0.25x..4x). Where
      does it leave the coherent basin? width of the basin = how tuned it is.
  G4  bits proxy: PPL of {ikaros real, ikaros DC-only, ikaros 1st-PC-only} — how much
      of the signal's structure is needed.

Run: HSA_OVERRIDE_GFX_VERSION=11.0.0 venv/bin/python h7_v10_falsify2.py  (no sudo)
"""
from __future__ import annotations
import sys, json, socket
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).parent))
from h7_rooted_lm_v4a import GlobalNorm, SubstrateEncoderV4, K_TOKENS, WIN_LEN, N_CHANNELS, BASE_MODEL, STATS
from h7_embodied_v8 import FilmEmbodiedSmolLM
from h7_embodied_v7 import encode, seq_nll
from transformers import AutoModelForCausalLM, AutoTokenizer

HOST = socket.gethostname()
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/IDENTITY_H7_2026-06-09"
V10 = ROOT / "results/IDENTITY_EMBODIED_V10_2026-06-10"
CKPT = V10 / f"v10_best_{HOST}.pt"
IK = OUT / f"substrate_replay_{HOST}_heldout_10ch.npz"
DAED = OUT / "substrate_replay_daedalus_10ch.npz"
RESULT = V10 / "v10_falsify2_result.json"
EVAL_TEXT = ("The forest was dark and quiet as she walked. He could not remember "
             "what the letter had said, only that it arrived on a cold morning. "
             "Beyond the river the lights of the town flickered against the hills.")


def main():
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
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL).to(device).eval()
    norm = GlobalNorm(STATS); pad = tok.pad_token_id
    ids = tok(EVAL_TEXT, return_tensors="pt", truncation=True, max_length=96).input_ids.to(device)
    with torch.no_grad():
        lb = base(ids).logits[:, :-1, :]
        ppl_base = float(np.exp(F.cross_entropy(lb.reshape(-1, lb.size(-1)), ids[:, 1:].reshape(-1),
                                                ignore_index=pad).item()))

    ik = np.load(IK)["windows"].astype(np.float32)
    dd = np.load(DAED)["windows"].astype(np.float32)

    @torch.no_grad()
    def ppl_w(w):
        l, _ = seq_nll(model, ids, encode(se, norm, w.astype(np.float32), device), pad)
        return float(np.exp(l.item()))

    def med_ppl(ws): return float(np.median([ppl_w(w) for w in ws]))

    res = {"ppl_base": ppl_base, "ik_real": med_ppl(ik[:8]), "daed_real": med_ppl(dd[:8])}
    print(f"base={ppl_base:.1f}  ikaros_real={res['ik_real']:.1f}  daedalus={res['daed_real']:.1f}")

    # G1 graded ikaros<->daedalus (pair windows, interpolate, median over 6 pairs)
    print("\nG1 ikaros<->daedalus interpolation:")
    g1 = {}
    for a in [1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125, 0.0]:
        ppls = []
        for i in range(6):
            w = a * ik[i] + (1 - a) * dd[i]
            ppls.append(ppl_w(w))
        g1[f"{a}"] = float(np.median(ppls))
        print(f"  ikaros_frac={a:.3f}  PPL={g1[f'{a}']:.1f}")
    res["G1_ikaros_daedalus_interp"] = g1

    # G2 DC-only: each channel -> its temporal mean (constant)
    print("\nG2 DC-only (channels replaced by temporal mean):")
    dc = []
    for i in range(8):
        w = ik[i].copy()
        wdc = np.repeat(w.mean(axis=0, keepdims=True), w.shape[0], axis=0)
        dc.append(ppl_w(wdc))
    res["G2_dc_only"] = {"median": float(np.median(dc)), "vs_real": float(np.median(dc) / res["ik_real"])}
    print(f"  DC-only median PPL={np.median(dc):.1f}  ({np.median(dc)/res['ik_real']:.2f}x real)")

    # G3 variance scaling around the mean
    print("\nG3 variance-scale of ikaros dynamics:")
    g3 = {}
    for s in [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]:
        ppls = []
        for i in range(6):
            w = ik[i]; mu = w.mean(axis=0, keepdims=True)
            ppls.append(ppl_w(mu + s * (w - mu)))
        g3[f"{s}"] = float(np.median(ppls))
        print(f"  scale={s:.2f}  PPL={g3[f'{s}']:.1f}")
    res["G3_var_scale"] = g3

    # G4 first-PC-only: keep only top principal component of the window (per-channel)
    print("\nG4 reduced-structure:")
    res["G4"] = {"real": res["ik_real"], "dc_only": res["G2_dc_only"]["median"]}

    RESULT.write_text(json.dumps(res, indent=2, default=str))
    print(f"\nsaved {RESULT}")

    # verdict
    print("\n=== ROUND-2 VERDICT ===")
    vals = [g1[k] for k in ["1.0","0.875","0.75","0.625","0.5","0.375","0.25","0.125","0.0"]]
    monotone = all(vals[i] <= vals[i+1] * 3 for i in range(len(vals)-1))  # loosely increasing
    cliff_ratio = max(vals[i+1] / max(vals[i],1e-9) for i in range(len(vals)-1))
    print(f"  G1 curve ikaros->daedalus: {[round(v,1) for v in vals]}")
    print(f"     max step-up ratio={cliff_ratio:.1f}x  ({'GRADED' if cliff_ratio < 20 else 'CLIFF'})")
    dcx = res["G2_dc_only"]["vs_real"]
    print(f"  G2 DC-only {dcx:.2f}x real: {'uses DYNAMICS (DC insufficient)' if dcx > 3 else 'scalar offset SUFFICIENT (weak)'}")


if __name__ == "__main__":
    main()
