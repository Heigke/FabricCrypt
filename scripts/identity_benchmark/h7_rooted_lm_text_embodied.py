"""H7 — make a real BYTE-LM depend on the body: the body's physical computation is load-bearing for
text prediction, and the LM's WEIGHTS learn to rely on it, so without the body the LM stops working.

This is the "on the actual LLM" version of h7_rooted_lm_embodied.py. A small causal byte-transformer
models a structured byte stream (it has real perplexity). Woven into the stream are QUERY positions
whose target byte depends on XOR of two earlier context bits — a genuinely non-linear dependency. The
prediction at query positions is ARCHITECTURALLY BOTTLENECKED so it cannot use the transformer's rich
features: its logits come only from (linear pooled context) + (a head reading the BODY gate g). A linear
map can't compute XOR, so the body is the only thing that can supply it. Plain cross-entropy over ALL
positions (no margin/spoof loss). Text positions train the transformer; query positions train the body
head — so part of the LM's weights (the gate head) literally encode reliance on the body's computation.

The body = the verified destructive-L3-cache XOR organ (micro_mem.c). Training uses the gate's truth
table (deterministic XOR; disclosed, fast). EVAL drives LIVE silicon per query. Probe-gate by removing
the body:
   native : live cache-XOR gate   -> LM works (low PPL on query positions)
   zero   : gate forced 0          -> query head loses its only XOR source -> PPL explodes
   random : gate randomised        -> same: the LM cannot do those tokens without the body
Overall PPL degrades too (query positions are a sizable fraction). native << zero/random  =>  the LM
is constitutively dependent on the body; its weights stopped being able to do the task alone.

Run under sandbox-disabled shell (gcc/exec). Out: results/IDENTITY_H7_2026-06-09/text_embodied_{host}.json
"""
from __future__ import annotations
import os, sys, time, json, argparse, math, socket
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from h7_rooted_lm_embodied import BodyGate          # the verified physical cache-XOR organ

HOST = socket.gethostname()
HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "results/IDENTITY_H7_2026-06-09"
VOCAB = 256
D1, D2 = 3, 7                      # lags whose bits the query target XORs
QTOK0 = 200                        # query target tokens live at 200/201 (out of the text byte range)


def make_stream(n, rng, qstride=4):
    """Structured byte stream + woven query positions. Returns x (n,), qmask (n,), and the two
    operand bits per position (valid where qmask)."""
    x = rng.integers(0, 190, size=n).astype(np.int64)            # 'text' bytes in [0,190)
    for stride, val in [(5, 17), (11, 113), (23, 151)]:          # periodic structure -> learnable PPL
        x[::stride] = val
    qmask = np.zeros(n, bool); qmask[QTOK0::qstride] = False      # placeholder
    # mark query positions on a regular grid (skip the warmup region needed for lags)
    qpos = np.arange(max(D1, D2) + 1, n, qstride)
    qmask[qpos] = True
    a = ((x >> 0) & 1).astype(np.int64)                          # LSB as the secret bit
    abit = np.zeros(n, np.int64); bbit = np.zeros(n, np.int64)
    abit[qpos] = a[qpos - D1]; bbit[qpos] = a[qpos - D2]
    y = np.empty(n, np.int64)
    y[:-1] = x[1:]; y[-1] = x[0]                                  # normal next-byte target
    y[qpos] = QTOK0 + (abit[qpos] ^ bbit[qpos])                  # query target = XOR -> token 200/201
    return x, y, qmask, abit, bbit


def build_model(ctx, d=128, nl=2, nh=4):
    import torch, torch.nn as nn

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln1 = nn.LayerNorm(d); self.attn = nn.MultiheadAttention(d, nh, batch_first=True)
            self.ln2 = nn.LayerNorm(d)
            self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        def forward(self, x, mask):
            h = self.ln1(x)
            a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
            x = x + a; x = x + self.mlp(self.ln2(x)); return x

    class TextBodyLM(nn.Module):
        """Full transformer head for text. SEPARATE bottlenecked head for query positions:
        query logits = linear(pooled emb context) + gate_head(body g). No transformer features reach
        the query head -> the body is the query head's only route to XOR."""
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(VOCAB, d); self.pos = nn.Embedding(ctx, d)
            self.blocks = nn.ModuleList([Block() for _ in range(nl)])
            self.lnf = nn.LayerNorm(d); self.text_head = nn.Linear(d, VOCAB)
            # query head: linear bypass over pooled context (causal mean) + the body gate
            self.q_bypass = nn.Linear(d, 2)
            self.q_gate = nn.Linear(1, 2, bias=False)

        def trunk(self, x, mask):
            T = x.shape[1]
            pos = torch.arange(T, device=x.device)
            h = self.emb(x) + self.pos(pos)[None]
            for b in self.blocks: h = b(h, mask)
            return self.lnf(h)

        def text_logits(self, x, mask):
            return self.text_head(self.trunk(x, mask))           # (B,T,VOCAB)

        def query_logits(self, x, g):
            # causal running mean of embeddings = linear, body-free context summary
            e = self.emb(x)                                       # (B,T,d)
            csum = torch.cumsum(e, dim=1)
            cnt = torch.arange(1, x.shape[1] + 1, device=x.device).float()[None, :, None]
            pooled = csum / cnt
            return self.q_bypass(pooled) + self.q_gate(g.unsqueeze(-1))  # (B,T,2)

    return TextBodyLM


def causal_mask(T, device):
    import torch
    return torch.triu(torch.full((T, T), float("-inf"), device=device), diagonal=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--ctx", type=int, default=64)
    ap.add_argument("--qstride", type=int, default=4, help="1 query position every qstride tokens")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n_eval", type=int, default=40)
    ap.add_argument("--win", type=float, default=0.04)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import torch, torch.nn as nn, torch.nn.functional as F
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(a.seed)

    # one long stream; sample windows from it
    N = 400_000
    X, Y, QM, AB, BB = make_stream(N, rng, qstride=a.qstride)
    Model = build_model(a.ctx); net = Model().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr)
    mask = causal_mask(a.ctx, dev)
    qfrac = float(QM.mean())
    print(f"[{HOST}] text-body LM: ctx={a.ctx} qstride={a.qstride} (query frac={qfrac:.2f}) "
          f"steps={a.steps} dev={dev}", flush=True)

    def batch(bs):
        i = rng.integers(0, N - a.ctx - 1, size=bs)
        idx = i[:, None] + np.arange(a.ctx)[None]
        return (torch.from_numpy(X[idx]).to(dev), torch.from_numpy(Y[idx]).to(dev),
                torch.from_numpy(QM[idx]).to(dev), torch.from_numpy(AB[idx]).to(dev),
                torch.from_numpy(BB[idx]).to(dev))

    t0 = time.time()
    for step in range(a.steps):
        x, y, qm, ab, bb = batch(a.batch)
        tl = net.text_logits(x, mask)                            # (B,T,VOCAB)
        g_truth = (ab ^ bb).float()                              # TRAIN gate = truth-table XOR (fast)
        ql = net.query_logits(x, g_truth)                        # (B,T,2)
        # text loss on non-query positions, query loss on query positions
        ce_text = F.cross_entropy(tl.reshape(-1, VOCAB), y.reshape(-1), reduction="none").reshape(y.shape)
        qy = (y - QTOK0).clamp(0, 1)
        ce_q = F.cross_entropy(ql.reshape(-1, 2), qy.reshape(-1), reduction="none").reshape(y.shape)
        loss = (ce_text * (~qm)).sum() / (~qm).sum() + (ce_q * qm).sum() / qm.sum().clamp(min=1)
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % 500 == 0:
            with torch.no_grad():
                qacc = float(((ql.argmax(-1) == qy)[qm]).float().mean())
            print(f"  step {step+1:5d} loss={loss.item():.3f} query_acc={qacc:.3f} t={time.time()-t0:.0f}s", flush=True)

    # ---- eval with the LIVE body, probe-gated ----
    net.eval()
    ev = [batch(1) for _ in range(a.n_eval)]                     # fixed eval windows
    body = BodyGate(win=a.win); fid = body.calibrate()
    print(f"[{HOST}] gate calib fidelity={fid:.2f} thr={body.thr:.0f}", flush=True)

    @torch.no_grad()
    def run(cond):
        rr = np.random.default_rng(7)
        q_ce_sum = q_n = 0.0; q_correct = 0
        t_ce_sum = t_n = 0.0
        for (x, y, qm, ab, bb) in ev:
            tl = net.text_logits(x, mask)
            qm_np = qm[0].cpu().numpy(); ab_np = ab[0].cpu().numpy(); bb_np = bb[0].cpu().numpy()
            g = np.zeros(a.ctx, np.float32)
            for t in range(a.ctx):
                if not qm_np[t]: continue
                if cond == "native": g[t] = body.gate(int(ab_np[t]), int(bb_np[t]))
                elif cond == "zero": g[t] = 0
                elif cond == "random": g[t] = rr.integers(0, 2)
                elif cond == "truth": g[t] = ab_np[t] ^ bb_np[t]
            gt = torch.from_numpy(g).to(dev)[None]
            ql = net.query_logits(x, gt)
            qy = (y - QTOK0).clamp(0, 1)
            # query metrics
            m = qm[0]
            if m.any():
                ce = F.cross_entropy(ql[0][m], qy[0][m], reduction="sum").item()
                q_ce_sum += ce; q_n += int(m.sum())
                q_correct += int((ql[0][m].argmax(-1) == qy[0][m]).sum())
            # text metrics (non-query)
            nm = (~qm[0])
            if nm.any():
                ce = F.cross_entropy(tl[0][nm], y[0][nm], reduction="sum").item()
                t_ce_sum += ce; t_n += int(nm.sum())
        qppl = math.exp(min(q_ce_sum / max(q_n, 1), 50))
        tppl = math.exp(min(t_ce_sum / max(t_n, 1), 50))
        # combined PPL weighted by token counts
        cppl = math.exp(min((q_ce_sum + t_ce_sum) / max(q_n + t_n, 1), 50))
        return {"query_ppl": round(qppl, 3), "query_acc": round(q_correct / max(q_n, 1), 3),
                "text_ppl": round(tppl, 3), "overall_ppl": round(cppl, 3)}

    res = {}
    try:
        for c in ["native", "truth", "zero", "random"]:
            res[c] = run(c)
            print(f"  [{c:7s}] query_acc={res[c]['query_acc']} query_ppl={res[c]['query_ppl']} "
                  f"text_ppl={res[c]['text_ppl']} overall_ppl={res[c]['overall_ppl']}", flush=True)
    finally:
        body.close()

    nat = res["native"]["query_acc"]; corrupt = max(res["zero"]["query_acc"], res["random"]["query_acc"])
    nat_oppl = res["native"]["overall_ppl"]; corr_oppl = min(res["zero"]["overall_ppl"], res["random"]["overall_ppl"])
    out = {"host": HOST, "ctx": a.ctx, "qstride": a.qstride, "query_frac": round(qfrac, 3),
           "gate_fidelity": round(fid, 3), "results": res,
           "BODY_LOAD_BEARING": bool(nat > 0.9 and nat - corrupt > 0.35),
           "overall_ppl_native": nat_oppl, "overall_ppl_no_body": corr_oppl,
           "ppl_inflation_without_body": round(corr_oppl / max(nat_oppl, 1e-9), 2)}
    out["verdict"] = ("LM CONSTITUTIVELY DEPENDS ON BODY: query tokens collapse to chance and overall "
                      "PPL inflates when the body is removed/corrupted — its weights cannot do the task alone"
                      if out["BODY_LOAD_BEARING"] else
                      "NOT body-dependent (LM solved query tokens without the body — check bottleneck)")
    OUT.mkdir(parents=True, exist_ok=True)
    jp = OUT / f"text_embodied_{HOST}.json"; jp.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2), flush=True)
    print(f"\n[{HOST}] VERDICT: {out['verdict']}", flush=True)
    print(f"  query_acc native={nat} zero={res['zero']['query_acc']} random={res['random']['query_acc']} | "
          f"overall PPL {nat_oppl} -> {corr_oppl} ({out['ppl_inflation_without_body']}x without body)", flush=True)
    print(f"  saved {jp}", flush=True)


if __name__ == "__main__":
    main()
