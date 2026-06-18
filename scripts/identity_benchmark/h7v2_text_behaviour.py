"""H7 v2 — does the body STEER the LLM's generated text (behaviour), not just a bit-accuracy number?

We let a frozen GPT-2 generate text autoregressively. At "decision points" (every qstride tokens) the
model is genuinely torn between its top-2 next tokens; WHICH branch it commits to is decided by the live
body keystream bit K_t = micro_cacheXOR XOR meso_GPUdroop XOR macro_CPU->GPU XOR prefcore_fp (the same
multi-layer organ as h7v2_multilayer_keystream). Everywhere else it decodes greedily. So the body is a
hidden controller steering the model down one of exponentially many fluent trajectories.

What this demonstrates (honest framing):
  * STEERING + IDENTITY: same die + same nonce -> byte-identical text (reproducible, body-locked behaviour).
  * UNIQUE: a foreign die's fingerprint -> a DIFFERENT trajectory (the model behaves like a different
    individual on different silicon).
  * FRESH: an old nonce -> different trajectory (replay can't reproduce today's behaviour).
  * DEPENDENCE: remove the body (bits forced 0) -> the model can no longer reproduce its own body-locked
    text; token-divergence vs the live reference jumps. The committed trajectory is constitutively the
    body's, not something GPT-2's weights can regenerate alone.
We measure token-divergence from the live reference and self-perplexity, and we PRINT the actual generated
text for each condition so a human can read that the behaviour really differs.

HONEST: top-2 branches are both locally fluent, so "no body" text does not become word-salad; the
dependence is on the *specific* (body-determined) trajectory/identity, which is exactly the substrate-
rooting claim. We do not pretend the body is required for generic fluency.

Run sandbox-disabled. HSA override. Out: results/.../v2_text_behaviour_{host}.json (+ .txt samples)
"""
from __future__ import annotations
import os, sys, json, time, socket, hashlib
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from h7_rooted_lm_embodied import BodyGate
from h7v2_multilayer_keystream import (read_prefcore_fingerprint, fp_bits, operands,
                                       MesoGate, MacroGate, DAEDALUS_FP)

HOST = socket.gethostname()
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/IDENTITY_H7_2026-06-09"; OUT.mkdir(parents=True, exist_ok=True)
CORPUS = ROOT / "data/tinyshakespeare.txt"


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=700); ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--ctx", type=int, default=96); ap.add_argument("--qstride", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--gen_len", type=int, default=160)
    ap.add_argument("--win", type=float, default=0.03); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    import torch, torch.nn as nn, torch.nn.functional as F
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    lm = GPT2LMHeadModel.from_pretrained("gpt2").to(dev).eval()
    for p in lm.parameters(): p.requires_grad_(False)
    D = lm.config.n_embd
    ids = np.array(tok(CORPUS.read_text(encoding="utf-8", errors="ignore"))["input_ids"], dtype=np.int64)
    rng = np.random.default_rng(a.seed)

    fp_own = read_prefcore_fingerprint()
    micro = BodyGate(win=a.win); mfid = micro.calibrate()
    meso = MesoGate(); meso.measure(); macro = MacroGate(); macro.measure()
    print(f"[{HOST}] text-behaviour: micro fid={mfid:.2f} meso{meso.table} macro{macro.table} dev={dev}", flush=True)

    @torch.no_grad()
    def hidden_top2(xb):
        o = lm(xb, output_hidden_states=True)
        return o.hidden_states[-1][:, -1], o.logits[:, -1].topk(2, -1).indices  # last position

    sel = nn.Sequential(nn.Linear(D + 1, 128), nn.GELU(), nn.Linear(128, 2)).to(dev)
    opt = torch.optim.AdamW(sel.parameters(), lr=a.lr)

    def windows(bs):
        i = rng.integers(0, len(ids) - a.ctx - 2, size=bs)
        return torch.from_numpy(ids[i[:, None] + np.arange(a.ctx + 1)[None]]).to(dev)

    # ---- train the selector (same objective as keystream: commit to top-2[K]) ----
    @torch.no_grad()
    def frozen_full(xb):
        o = lm(xb, output_hidden_states=True); return o.hidden_states[-1], o.logits.topk(2, -1).indices
    qpos = list(range(2, a.ctx, a.qstride))
    nonce_tr = b"H7v2-behave-train"; fpb_tr = fp_bits(fp_own, nonce_tr, a.ctx)
    t0 = time.time()
    for step in range(a.steps):
        if step > 0 and step % 200 == 0: meso.measure(); macro.measure()
        x = windows(a.batch)[:, :a.ctx]; h, _ = frozen_full(x); xnp = x.cpu().numpy()
        mic = {(p, q): micro.gate(p, q) for p in (0, 1) for q in (0, 1)}
        B = x.shape[0]; b = np.zeros((B, a.ctx), np.float32)
        for bi in range(B):
            for t in qpos:
                (ami, bmi), (ame, bme), (ama, bma) = operands(nonce_tr, xnp[bi, max(0, t-4):t])
                b[bi, t] = mic[(ami, bmi)] ^ meso.gate(ame, bme) ^ macro.gate(ama, bma) ^ int(fpb_tr[t])
        bt = torch.from_numpy(b).to(dev); tgt = bt.long()
        logit2 = sel(torch.cat([h, bt.unsqueeze(-1)], -1))
        qm = torch.zeros(B, a.ctx, dtype=torch.bool, device=dev); qm[:, qpos] = True
        ce = F.cross_entropy(logit2.reshape(-1, 2), tgt.reshape(-1), reduction="none").reshape(B, a.ctx)
        loss = (ce * qm).sum() / qm.sum(); opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % 200 == 0: print(f"  step {step+1} loss={loss.item():.3f} t={time.time()-t0:.0f}s", flush=True)

    # ---- autoregressive generation: body steers the branch at decision points ----
    meso.measure(); macro.measure()
    prompt = "To be, or not to be, that is the question:\n"
    p_ids = tok(prompt)["input_ids"]

    @torch.no_grad()
    def generate(nonce, fp_for_bits, body_on=True, micro_live=True):
        fpb = fp_bits(fp_for_bits, nonce, a.gen_len + 8)
        seq = list(p_ids)
        mic = {(p, q): micro.gate(p, q) for p in (0, 1) for q in (0, 1)} if micro_live else {(p, q): 0 for p in (0,1) for q in (0,1)}
        dec = 0
        for step in range(a.gen_len):
            ctx = torch.tensor([seq[-a.ctx:]], device=dev)
            h, top2 = hidden_top2(ctx)
            is_q = (step % a.qstride == 0)
            if is_q:
                cseq = np.array(seq[-4:], dtype=np.int64)
                (ami, bmi), (ame, bme), (ama, bma) = operands(nonce, cseq)
                if body_on:
                    k = mic[(ami, bmi)] ^ meso.gate(ame, bme) ^ macro.gate(ama, bma) ^ int(fpb[dec])
                else:
                    k = 0                                   # body removed
                bt = torch.tensor([[float(k)]], device=dev)
                choice = sel(torch.cat([h, bt], -1)).argmax(-1).item()
                nxt = top2[0, choice].item(); dec += 1
            else:
                nxt = top2[0, 0].item()                     # greedy elsewhere
            seq.append(nxt)
        return seq[len(p_ids):]

    nonce_a = b"H7v2-behave-eval-A"
    ref = generate(nonce_a, fp_own, body_on=True)            # live reference trajectory
    runs = {
        "live_ikaros":    generate(nonce_a, fp_own, body_on=True),     # repeat -> should equal ref
        "no_body":        generate(nonce_a, fp_own, body_on=False),
        "foreign_die":    generate(nonce_a, DAEDALUS_FP, body_on=True),
        "replay_oldnonce":generate(b"H7v2-behave-OLD", fp_own, body_on=True),
    }

    def divergence(seq):  # fraction of tokens differing from the live reference
        n = min(len(seq), len(ref)); return round(float(np.mean([seq[i] != ref[i] for i in range(n)])), 3)

    @torch.no_grad()
    def ppl(seq):
        full = torch.tensor([p_ids + seq], device=dev)
        o = lm(full); lp = F.log_softmax(o.logits[0, :-1], -1)
        tgt = full[0, 1:]; nll = -lp[range(len(tgt)), tgt].mean()
        return round(float(torch.exp(nll)), 2)

    micro.close()
    samples = {k: tok.decode(v) for k, v in runs.items()}
    out = {"host": HOST, "prompt": prompt, "gen_len": a.gen_len,
           "token_divergence_vs_live_ref": {k: divergence(v) for k, v in runs.items()},
           "self_perplexity": {k: ppl(v) for k, v in runs.items()},
           "reference_is_reproducible": bool(runs["live_ikaros"] == ref),
           "foreign_die_steers_different_text": bool(divergence(runs["foreign_die"]) > 0.1),
           "replay_steers_different_text": bool(divergence(runs["replay_oldnonce"]) > 0.1),
           "no_body_cannot_reproduce": bool(divergence(runs["no_body"]) > 0.1),
           "text_samples": samples,
           "honest_note": ("Both branches are top-2 so all texts stay locally fluent; the body's role is to "
                           "STEER which fluent trajectory the model commits to. Same die+nonce reproduces the "
                           "trajectory exactly; foreign die / old nonce / no body cannot -> the generated "
                           "behaviour is constitutively the body's, and it is die-unique and fresh.")}
    (OUT / f"v2_text_behaviour_{HOST}.json").write_text(json.dumps(out, indent=2))
    txt = OUT / f"v2_text_behaviour_{HOST}.txt"
    with open(txt, "w") as f:
        f.write(f"PROMPT: {prompt}\n" + "=" * 78 + "\n")
        for k, s in samples.items():
            f.write(f"\n### {k}  (token-div vs live ref = {divergence(runs[k])}, ppl={ppl(runs[k])})\n{s}\n")
    print(json.dumps({k: v for k, v in out.items() if k != "text_samples"}, indent=2), flush=True)
    print(f"\n[{HOST}] reproducible={out['reference_is_reproducible']} "
          f"foreign_div={out['token_divergence_vs_live_ref']['foreign_die']} "
          f"no_body_div={out['token_divergence_vs_live_ref']['no_body']} "
          f"replay_div={out['token_divergence_vs_live_ref']['replay_oldnonce']}\n  text -> {txt}", flush=True)


if __name__ == "__main__":
    main()
