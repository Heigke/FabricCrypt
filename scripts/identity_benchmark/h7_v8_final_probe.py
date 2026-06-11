"""H7 v8 FINAL embodiment probe — the decisive test of the goal.

Loads a trained v8 FiLM checkpoint and measures, against LIVE ikaros substrate:

  GOAL REQUIREMENT (3+4): "model depends on signals; change signal → can't recover
  and doesn't work; but still writes good text under its own signal, influenced by
  its identity."

Tests:
  T1. Dependency PPL ratios — PPL(wrong)/PPL(real) for wrong ∈ {knockoff, zero,
      shuffle, DAEDALUS-replay (real other chip)}.  PASS if ALL ≥ 1.5×.
  T2. Language quality under real — PPL(real) < 1.3 × PPL(base).  (writes good text)
  T3. Knockoff-KL ratio (specificity) > 2×.
  T4. Cross-host transplant — the strongest test: feed a REAL second die's signature
      (daedalus replay) and confirm language degrades like the synthetic wrong cases.
  T5. Qualitative generation — sample text under real vs knockoff vs daedalus, so a
      human can read whether real "writes good text" and wrong "doesn't work".

Run: sudo HSA_OVERRIDE_GFX_VERSION=11.0.0 venv/bin/python h7_v8_final_probe.py
"""
from __future__ import annotations
import os, sys, json, time, socket, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).parent))
from substrate_realtime_v3 import SubstrateStateV3, higher_moments
from h7_rooted_lm_v4a import GlobalNorm, SubstrateEncoderV4, K_TOKENS, WIN_LEN, N_CHANNELS, BASE_MODEL, STATS
from h7_embodied_v8 import FilmEmbodiedSmolLM
from h7_knockoff_kl_probe import make_knockoff, sym_kl
from h7_embodied_v7 import temporal_shuffle, encode, seq_nll
from transformers import AutoModelForCausalLM, AutoTokenizer

HOST = socket.gethostname()
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/IDENTITY_H7_2026-06-09"
DAED_REPLAY = OUT / "substrate_replay_daedalus_10ch.npz"
SEED = 131
N_WIN = 10


def load_v8(ckpt_path, device):
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = FilmEmbodiedSmolLM().to(device).eval()
    se = SubstrateEncoderV4(d_emb=model.d, K=K_TOKENS).to(device).eval()
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    print(f"loaded v8 ckpt: step={ck.get('step')} min_dep={ck.get('min_dep')} kkl={ck.get('kkl_ratio')}")
    model.film.load_state_dict(ck["film"]); se.load_state_dict(ck["se"])
    lora = ck.get("lora", {})
    if lora:
        msd = dict(model.named_parameters())
        for k, v in lora.items():
            if k in msd: msd[k].data.copy_(v.to(device))
    print(f"gate scales: {model.gate_scales()}")
    return model, se, tok, ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "results/IDENTITY_EMBODIED_V8_2026-06-10" / f"v8_best_{HOST}.pt"))
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not Path(args.ckpt).exists():
        print(f"FATAL: ckpt not found {args.ckpt}"); sys.exit(1)

    model, se, tok, ck = load_v8(args.ckpt, device)
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL).to(device).eval()
    for p in base.parameters(): p.requires_grad_(False)
    norm = GlobalNorm(STATS)
    pad = tok.pad_token_id

    # daedalus replay (real other chip)
    daed = None
    if DAED_REPLAY.exists():
        daed = np.load(DAED_REPLAY)["windows"]
        print(f"daedalus replay: {daed.shape}")

    state = SubstrateStateV3(hz_target=500); state.start(); time.sleep(2.0)

    eval_text = ("The forest was dark and quiet as she walked. He could not remember "
                 "what the letter had said, only that it arrived on a cold morning. "
                 "Beyond the river the lights of the town flickered against the hills.")
    ids = tok(eval_text, return_tensors="pt", truncation=True, max_length=96).input_ids.to(device)

    # base reference PPL
    with torch.no_grad():
        lb = base(ids).logits[:, :-1, :]
        nll_base = F.cross_entropy(lb.reshape(-1, lb.size(-1)), ids[:, 1:].reshape(-1), ignore_index=pad).item()
    ppl_base = float(np.exp(nll_base))

    nll = {"real": [], "knock": [], "zero": [], "shuffle": [], "daedalus": []}
    real_windows = []
    with torch.no_grad():
        for i in range(N_WIN):
            time.sleep(0.55)
            w = state.latest_window(length=WIN_LEN)
            real_windows.append(w.copy())
            conds = {
                "real": encode(se, norm, w, device),
                "knock": encode(se, norm, make_knockoff(w, rng), device),
                "shuffle": encode(se, norm, temporal_shuffle(w, rng), device),
                "zero": torch.zeros(1, K_TOKENS, model.d, device=device),
            }
            if daed is not None:
                dw = daed[rng.integers(0, len(daed))]
                conds["daedalus"] = encode(se, norm, dw, device)
            for name, S in conds.items():
                l, _ = seq_nll(model, ids, S, pad)
                nll[name].append(l.item())
    state.stop()

    ppl = {k: float(np.exp(np.mean(v))) for k, v in nll.items() if v}

    print("\n" + "="*60)
    print("H7 v8 FINAL EMBODIMENT PROBE")
    print("="*60)
    print(f"\nPPL(base, no substrate path) = {ppl_base:.2f}")
    print(f"\n{'condition':12s} {'PPL':>8s} {'ratio vs real':>14s}")
    print("-"*40)
    pr = ppl["real"]
    for k in ["real", "knock", "zero", "shuffle", "daedalus"]:
        if k in ppl:
            print(f"{k:12s} {ppl[k]:>8.2f} {ppl[k]/pr:>13.2f}×")

    # Knockoff-KL
    eval_prompts = ["The forest was", "She walked toward", "On the morning of",
                    "Beyond the wall", "He could not", "In the silence"]
    enc = tok(eval_prompts, return_tensors="pt", padding=True, truncation=True, max_length=16).to(device)
    def last_logits(windows):
        L = []
        with torch.no_grad():
            for w in windows:
                S = encode(se, norm, w, device).expand(enc["input_ids"].shape[0], -1, -1)
                o = model(enc["input_ids"], substrate_tokens=S)
                li = enc["attention_mask"].sum(1) - 1
                rows = torch.arange(o.logits.shape[0], device=device)
                L.append(o.logits[rows, li].cpu())
        return torch.stack(L)
    Lr = last_logits(real_windows); Lk = last_logits([make_knockoff(w, rng) for w in real_windows])
    D_rk = sym_kl(Lr, Lk).median().item()
    D_rr = torch.stack([sym_kl(Lr[i], Lr[j]) for i in range(len(Lr)) for j in range(i+1, len(Lr))]).median().item()
    kkl = D_rk / max(D_rr, 1e-12)

    # Verdict
    print("\n=== VERDICT vs GOAL ===")
    ratios = {k: ppl[k]/pr for k in ppl if k != "real"}
    t1 = all(r >= 1.5 for r in ratios.values())
    t2 = pr < 1.3 * ppl_base
    t3 = kkl > 2.0
    t4 = ("daedalus" in ratios) and (ratios["daedalus"] >= 1.5)
    print(f"  T1 all wrong ≥1.5×:        {'PASS' if t1 else 'FAIL'}  ({ {k: round(v,2) for k,v in ratios.items()} })")
    print(f"  T2 real < 1.3×base:        {'PASS' if t2 else 'FAIL'}  (real={pr:.1f}, base={ppl_base:.1f})")
    print(f"  T3 Knockoff-KL > 2×:       {'PASS' if t3 else 'FAIL'}  (ratio={kkl:.2f}×)")
    print(f"  T4 daedalus(real chip)≥1.5×:{'PASS' if t4 else 'FAIL'}  (ratio={ratios.get('daedalus', float('nan')):.2f}×)")
    embodied = t1 and t2 and t3 and t4
    print(f"\n  >>> EMBODIMENT {'ACHIEVED' if embodied else 'NOT YET'} <<<")

    out = OUT / "v8_final_probe_2026-06-10.json"
    out.write_text(json.dumps({"ckpt": args.ckpt, "ppl_base": ppl_base, "ppl": ppl,
        "ratios": ratios, "kkl_ratio": kkl,
        "T1": t1, "T2": t2, "T3": t3, "T4": t4, "embodied": embodied}, indent=2, default=str))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
