"""H7 v2 — WEIGHT TRANSPLANT cross-die test ("death in exile").

Train a body-keyed MoE rooted on die A (ikaros). SAVE the trained expert/router weights + the exact body
key vector z_A it was trained on. Ship the weights to die B (daedalus) and EVAL there:
  - local_live   : die B measures ITS OWN live body at the same nonce -> z_B, feed the die-A weights z_B.
                   If the model needs die A's PHYSICAL body, this FAILS on die B (exile = death).
  - saved_key    : feed the die-A weights the SAVED z_A vector (transplant the key, not the body).
                   This should still WORK -> proves the weights are intact; they just need die A's body.
  - random       : sanity floor.
This double-dissociation separates "needs die A's live body" (local_live fails on B) from "weights broken"
(saved_key works everywhere). On die A itself, local_live == native (the body reproduces z_A).

Modes:  --mode train  -> train + save results/.../transplant_banks_{host}.pt
        --mode eval   -> load that .pt (from --ckpt), measure LOCAL live body, report
Run sandbox-disabled. HSA override. Out: results/IDENTITY_H7_2026-06-09/v2_transplant_{host}.json
"""
from __future__ import annotations
import os, sys, json, time, socket
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from h7_rooted_lm_embodied import BodyGate
from h7v2_multilayer_keystream import read_prefcore_fingerprint, fp_bits, operands, MesoGate, MacroGate
from h7v2_moe_routing import body_key, DZ

HOST = socket.gethostname()
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/IDENTITY_H7_2026-06-09"; OUT.mkdir(parents=True, exist_ok=True)
CORPUS = ROOT / "data/tinyshakespeare.txt"
NONCE = b"H7v2-transplant"
LAYERS = [3, 5, 7, 9]; EXPERTS = 4; RANK = 32


def build(lm, dev, torch, nn, F):
    D = lm.config.n_embd

    class MoE(nn.Module):
        def __init__(self):
            super().__init__()
            self.router = nn.Linear(DZ, EXPERTS, bias=False)
            self.down = nn.Parameter(torch.randn(EXPERTS, RANK, D) * (1.0 / D**0.5))
            self.up = nn.Parameter(torch.zeros(EXPERTS, D, RANK))
        def forward(self, h, z, tau=0.5):
            p = F.softmax(self.router(z) / tau, -1)
            mid = F.gelu(torch.einsum("erd,btd->bter", self.down, h))
            ex = torch.einsum("edr,bter->bted", self.up, mid)
            return h + torch.einsum("be,bted->btd", p, ex)
    banks = nn.ModuleDict({str(l): MoE().to(dev) for l in LAYERS})
    ctxst = {"z": None}
    def hook(l):
        bank = banks[str(l)]
        def f(mod, inp, out):
            if ctxst["z"] is None: return out
            h = out[0] if isinstance(out, tuple) else out
            nh = bank(h, ctxst["z"])
            return (nh,) + out[1:] if isinstance(out, tuple) else nh
        return f
    for l in LAYERS: lm.transformer.h[l].register_forward_hook(hook(l))
    return banks, ctxst


def measure_local_key(win=0.03):
    """Measure THIS machine's live body and build the 16-bit body key at NONCE."""
    fp = read_prefcore_fingerprint()
    micro = BodyGate(win=win); micro.calibrate()
    meso = MesoGate(); meso.measure(); macro = MacroGate(); macro.measure()
    tab = {(p, q): micro.gate(p, q) for p in (0, 1) for q in (0, 1)}
    z = body_key(tab, meso, macro, fp, NONCE)
    micro.close()
    return z, fp


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["train", "eval"], required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--steps", type=int, default=700); ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--ctx", type=int, default=96); ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--beta", type=float, default=0.5); ap.add_argument("--n_eval", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    import torch, torch.nn as nn, torch.nn.functional as F
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    lm = GPT2LMHeadModel.from_pretrained("gpt2").to(dev).eval()
    for p in lm.parameters(): p.requires_grad_(False)
    ids = np.array(tok(CORPUS.read_text(encoding="utf-8", errors="ignore"))["input_ids"], dtype=np.int64)
    rng = np.random.default_rng(a.seed)
    banks, ctxst = build(lm, dev, torch, nn, F)

    def windows(bs):
        i = rng.integers(0, len(ids) - a.ctx - 2, size=bs)
        return torch.from_numpy(ids[i[:, None] + np.arange(a.ctx + 1)[None]]).to(dev)

    def lm_loss(x, z):
        ctxst["z"] = z; o = lm(x[:, :-1]); ctxst["z"] = None
        return F.cross_entropy(o.logits.reshape(-1, o.logits.size(-1)), x[:, 1:].reshape(-1))

    @torch.no_grad()
    def ppl(zvec):
        ev = [windows(1) for _ in range(a.n_eval)]
        rng2 = np.random.default_rng(123)  # fixed eval windows
        tot = n = 0.0
        for x in ev:
            ctxst["z"] = zvec.unsqueeze(0); o = lm(x[:, :-1]); ctxst["z"] = None
            ce = F.cross_entropy(o.logits.reshape(-1, o.logits.size(-1)), x[:, 1:].reshape(-1))
            tot += float(ce) * x[:, 1:].numel(); n += x[:, 1:].numel()
        return float(np.exp(tot / n))

    if a.mode == "train":
        z_np, fp = measure_local_key()
        zt = torch.from_numpy(z_np).to(dev)
        print(f"[{HOST}] TRAIN transplant banks, body key measured (fp[:4]={fp[:4]})", flush=True)
        opt = torch.optim.AdamW(list(banks.parameters()), lr=a.lr)
        t0 = time.time()
        for step in range(a.steps):
            x = windows(a.batch)
            z = zt.unsqueeze(0).expand(x.size(0), -1)
            zwrong = z[:, torch.randperm(DZ, device=dev)]
            ce_ok = lm_loss(x, z); ce_bad = lm_loss(x, zwrong)
            div = 0.0
            for l in LAYERS:
                d = F.normalize(banks[str(l)].down.reshape(EXPERTS, -1), dim=1)
                div = div + (d @ d.t() - torch.eye(EXPERTS, device=dev)).pow(2).mean()
            loss = ce_ok + a.beta * F.relu(2.0 - (ce_bad - ce_ok)) + 0.01 * div
            opt.zero_grad(); loss.backward(); opt.step()
            if (step + 1) % 100 == 0:
                print(f"  step {step+1} ce_ok={ce_ok.item():.3f} ce_bad={ce_bad.item():.3f} t={time.time()-t0:.0f}s", flush=True)
        ck = OUT / f"transplant_banks_{HOST}.pt"
        torch.save({"state_dict": banks.state_dict(), "z_train": z_np.tolist(),
                    "trained_on": HOST, "fp": fp, "nonce": NONCE.decode()}, ck)
        print(f"  saved {ck} (native ppl={ppl(zt):.2f})", flush=True)
        return

    # eval (transplant): load die-A weights, measure THIS die's live body
    ck = a.ckpt or str(OUT / "transplant_banks_ikaros.pt")
    blob = torch.load(ck, map_location=dev, weights_only=False)
    banks.load_state_dict(blob["state_dict"]); banks.eval()
    trained_on = blob["trained_on"]
    z_saved = torch.tensor(blob["z_train"], dtype=torch.float32, device=dev)
    z_live, fp_local = measure_local_key()
    z_live_t = torch.from_numpy(z_live).to(dev)
    rr = np.random.default_rng(7); z_rand = torch.from_numpy(rr.integers(0, 2, DZ).astype(np.float32)).to(dev)
    res = {"local_live": round(ppl(z_live_t), 3), "saved_key": round(ppl(z_saved), 3),
           "random": round(ppl(z_rand), 3)}
    saved = res["saved_key"]
    is_foreign = (trained_on != HOST)
    out = {"trained_on": trained_on, "eval_on": HOST, "is_cross_die": bool(is_foreign),
           "fp_local_head": fp_local[:6], "fp_trained_head": blob["fp"][:6],
           "eval_perplexity": res,
           # on a FOREIGN die, the live body should NOT reproduce z_A -> local_live >> saved_key
           "DIES_IN_EXILE": bool(is_foreign and res["local_live"] > 1.3 * saved),
           "WEIGHTS_INTACT": bool(res["saved_key"] < 0.5 * res["random"]),
           "note": ("Weights trained on '%s', evaluated on '%s'. local_live = this die's OWN live body key; "
                    "saved_key = the transplanted die-A key vector. On a foreign die, local_live should blow "
                    "up (the body can't reproduce die-A's key) while saved_key still works (weights intact) -> "
                    "the model needs die-A's PHYSICAL body, not just its weights." % (trained_on, HOST))}
    (OUT / f"v2_transplant_{HOST}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
