"""H7 — FULL multi-layer embodiment demo: the LLM computes through SEVERAL physical layers of its own
body and is constitutively dependent on them; the analog layer makes it die-UNIQUE. One runnable demo.

The body computes for the LM across three physical mechanisms:
  LAYER 1  digital cache-XOR  (micro_mem.c destructive-L3 interference)   g1 = XOR(u1,u2)
  LAYER 2  digital cache-XOR composed on layer 1's output                  g2 = XOR(g1,u3) = PAR3(u1,u2,u3)
           -> a genuine TWO-STAGE physical circuit: stage 2 consumes stage 1. The LM backbone is LINEAR
              and cannot compute parity; only this composed body circuit can.
  LAYER 3  analog voltage-droop transient reservoir (per-die silicon)      -> die-UNIQUE features

The LM (linear bypass over the input bits + bottleneck head) is trained with PLAIN cross-entropy to
predict PAR3. Its head weights learn to rely on g2 (the layer-2 body output); remove/corrupt the body
and the LM collapses. The analog layer is added on a separate XOR task to show per-die uniqueness:
weights trained on THIS die's droop do not transfer to another die.

DEMO A (multi-layer meta-computation + body dependence), target = PAR3, eval on LIVE silicon:
   native        both cache stages live              -> solves PAR3
   no_body       g1=g2=0                             -> collapses (LM can't do parity alone)
   stage2_off    g1 live, g2=0                       -> collapses (has u3 & g1 but can't XOR them)
   stage1_bad    g1 randomised -> g2 computed on it  -> collapses (layer 1 feeds layer 2)
DEMO B (analog per-die uniqueness), target = XOR_t1t2, droop reservoir readout:
   own_die       this die's recorded droop           -> above chance
   foreign_die   another die's droop, same inputs    -> ~chance  => computation is UNIQUE to the die

Run under sandbox-disabled shell (gcc/exec). Needs transient_vdroop_raw_{host}.npz (+ _daedalus for
uniqueness). Out: results/IDENTITY_H7_2026-06-09/multilayer_demo_{host}.json
"""
from __future__ import annotations
import os, sys, time, json, math, socket
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from h7_rooted_lm_embodied import BodyGate

HOST = socket.gethostname()
HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "results/IDENTITY_H7_2026-06-09"
WASHOUT = 150


def lag(x, k):
    y = np.zeros_like(x)
    if k > 0: y[k:] = x[:-k]; return y
    return x.copy()


def load_die(host):
    p = OUT / f"transient_vdroop_raw_{host}.npz"
    if not p.exists(): return None, None
    d = np.load(p); return d["u"].astype(int), d["Tn"].astype(np.float32)


def die_features(Tn):
    flat = Tn.reshape(Tn.shape[0], -1)
    return np.hstack([flat, lag(flat, 1), lag(flat, 2)])


# ======================================================================================
# DEMO A — two composed cache layers compute PAR3; the LM depends on them
# ======================================================================================
def demo_A(body, win, n_eval=160, steps=4000, seed=0):
    import torch, torch.nn as nn, torch.nn.functional as F
    torch.manual_seed(seed)
    L = 4000
    rng = np.random.default_rng(seed)
    u = rng.integers(0, 2, size=L)
    u1, u2, u3 = lag(u, 1), lag(u, 2), lag(u, 3)
    y = (u1 ^ u2 ^ u3).astype(np.int64)                 # PAR3 target
    K = 6
    Uwin = np.stack([lag(u.astype(np.float32), k) for k in range(1, K + 1)], 1)  # linear bypass inputs

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    bypass = nn.Linear(K, 2).to(dev)                    # linear escape hatch (has u1..u6 incl u3)
    g1_head = nn.Linear(1, 2, bias=False).to(dev)
    g2_head = nn.Linear(1, 2, bias=False).to(dev)
    opt = torch.optim.AdamW(list(bypass.parameters()) + list(g1_head.parameters())
                            + list(g2_head.parameters()), lr=3e-3)
    Ut = torch.from_numpy((Uwin - Uwin.mean(0)) / (Uwin.std(0) + 1e-9)).float().to(dev)
    yt = torch.from_numpy(y).to(dev)
    cut = WASHOUT + int(0.7 * (L - WASHOUT))
    tr = torch.arange(WASHOUT, cut, device=dev)
    # train with the body's TRUTH TABLE (deterministic XOR composition), fast
    g1_tt = torch.from_numpy((u1 ^ u2).astype(np.float32)).to(dev)
    g2_tt = torch.from_numpy(y.astype(np.float32)).to(dev)
    for step in range(steps):
        logits = bypass(Ut[tr]) + g1_head(g1_tt[tr, None]) + g2_head(g2_tt[tr, None])
        loss = F.cross_entropy(logits, yt[tr])
        opt.zero_grad(); loss.backward(); opt.step()

    te = np.arange(cut, L)
    te = te[:n_eval]                                    # subsample eval positions (live silicon = slow)
    rr = np.random.default_rng(7)

    @torch.no_grad()
    def evalcond(cond):
        g1 = np.zeros(len(te), np.float32); g2 = np.zeros(len(te), np.float32)
        for i, t in enumerate(te):
            a, b, c = int(u1[t]), int(u2[t]), int(u3[t])
            if cond == "no_body":
                g1[i] = 0; g2[i] = 0; continue
            if cond == "stage1_bad":
                gg1 = rr.integers(0, 2)                 # layer 1 corrupted
            else:
                gg1 = body.gate(a, b)                   # LAYER 1 live
            g1[i] = gg1
            if cond == "stage2_off":
                g2[i] = 0
            else:
                g2[i] = body.gate(int(gg1), c)         # LAYER 2 live, consumes layer-1 output
        lg = (bypass(Ut[torch.from_numpy(te).to(dev)])
              + g1_head(torch.from_numpy(g1).to(dev)[:, None])
              + g2_head(torch.from_numpy(g2).to(dev)[:, None]))
        pred = lg.argmax(-1).cpu().numpy()
        return float((pred == y[te]).mean()), pred

    res = {}; preds = {}
    for c in ["native", "no_body", "stage2_off", "stage1_bad"]:
        res[c], preds[c] = evalcond(c)
    chance = float(max(y[te].mean(), 1 - y[te].mean()))
    return {"task": "PAR3 (3-bit parity via 2 composed cache layers)", "chance": round(chance, 3),
            "acc": {k: round(v, 3) for k, v in res.items()},
            "n_eval": len(te), "y_sample": y[te][:32].tolist(),
            "pred_native": preds["native"][:32].tolist(),
            "pred_no_body": preds["no_body"][:32].tolist()}


# ======================================================================================
# DEMO B — analog per-die droop reservoir: uniqueness (own die vs foreign die)
# ======================================================================================
def demo_B(steps=5000, seed=0):
    import torch, torch.nn as nn, torch.nn.functional as F
    torch.manual_seed(seed)
    u, Tn = load_die(HOST)
    if u is None: return {"error": f"no transient_vdroop_raw_{HOST}.npz"}
    L = len(u)
    y = (lag(u, 1) ^ lag(u, 2)).astype(np.int64)        # XOR_t1t2 (the droop-solvable task)
    K = 6
    Uwin = np.stack([lag(u.astype(np.float32), k) for k in range(1, K + 1)], 1)
    Xd = die_features(Tn)
    cut = WASHOUT + int(0.7 * (L - WASHOUT)); tr = np.arange(WASHOUT, cut); te = np.arange(cut, L)
    mu, sd = Xd[tr].mean(0), Xd[tr].std(0) + 1e-9
    Xz = (Xd - mu) / sd
    umu, usd = Uwin[tr].mean(0), Uwin[tr].std(0) + 1e-9
    Uz = (Uwin - umu) / usd
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    bypass = nn.Linear(K, 2).to(dev); die_head = nn.Linear(Xz.shape[1], 2, bias=False).to(dev)
    opt = torch.optim.AdamW(list(bypass.parameters()) + list(die_head.parameters()), lr=5e-3)
    Ut = torch.from_numpy(Uz).float().to(dev); Dt = torch.from_numpy(Xz).float().to(dev)
    yt = torch.from_numpy(y).to(dev); tri = torch.from_numpy(tr).to(dev)
    for step in range(steps):
        lg = bypass(Ut[tri]) + die_head(Dt[tri])
        loss = F.cross_entropy(lg, yt[tri]); opt.zero_grad(); loss.backward(); opt.step()

    @torch.no_grad()
    def acc(Dmat):
        lg = bypass(Ut[torch.from_numpy(te).to(dev)]) + die_head(torch.from_numpy(Dmat[te]).float().to(dev))
        return float((lg.argmax(-1).cpu().numpy() == y[te]).mean())

    out = {"task": "XOR_t1t2 (analog droop reservoir)", "chance": round(float(max(y[te].mean(), 1 - y[te].mean())), 3),
           "own_die": round(acc(Xz), 3)}
    foreign = "daedalus" if HOST == "ikaros" else "ikaros"
    uf, Tnf = load_die(foreign)
    if Tnf is not None and len(uf) == L and np.array_equal(uf, u):
        out["foreign_die"] = round(acc((die_features(Tnf) - mu) / sd), 3)
        out["foreign_host"] = foreign
        out["uniqueness_gap"] = round(out["own_die"] - out["foreign_die"], 3)
    return out


def main():
    print(f"\n{'='*70}\n  H7 FULL MULTI-LAYER EMBODIMENT DEMO — host={HOST}\n{'='*70}", flush=True)
    body = BodyGate(win=0.04); fid = body.calibrate()
    print(f"  cache organ calibrated (fidelity={fid:.2f}, thr={body.thr:.0f})\n", flush=True)
    try:
        A = demo_A(body, win=0.04)
    finally:
        body.close()
    B = demo_B()

    print(f"\n--- DEMO A: multi-layer meta-computation, the LM depends on its body ---")
    print(f"  task: {A['task']}   chance={A['chance']}")
    for k in ["native", "no_body", "stage2_off", "stage1_bad"]:
        print(f"    {k:11s} acc={A['acc'][k]}")
    print(f"  true   : {''.join(map(str,A['y_sample']))}")
    print(f"  native : {''.join(map(str,A['pred_native']))}  <- body computing")
    print(f"  no_body: {''.join(map(str,A['pred_no_body']))}  <- body removed")

    print(f"\n--- DEMO B: analog per-die uniqueness ---")
    print(f"  task: {B.get('task')}   chance={B.get('chance')}")
    print(f"    own_die    acc={B.get('own_die')}")
    if "foreign_die" in B:
        print(f"    foreign_die({B['foreign_host']}) acc={B['foreign_die']}  (uniqueness gap={B['uniqueness_gap']})")
    else:
        print(f"    foreign_die: NOT AVAILABLE (need matched-seed daedalus recording)")

    nat = A["acc"]["native"]; worst_abl = max(A["acc"]["no_body"], A["acc"]["stage2_off"], A["acc"]["stage1_bad"])
    verdict = {
        "host": HOST, "demoA": A, "demoB": B,
        "MULTILAYER_LOAD_BEARING": bool(nat > 0.85 and nat - worst_abl > 0.30),
        "UNIQUE_per_die": bool("uniqueness_gap" in B and B["uniqueness_gap"] > 0.08),
    }
    print(f"\n{'='*70}")
    print(f"  VERDICT: multi-layer body computation load-bearing for the LM = {verdict['MULTILAYER_LOAD_BEARING']}")
    print(f"           per-die unique (analog layer)                        = {verdict['UNIQUE_per_die']}")
    print(f"{'='*70}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"multilayer_demo_{HOST}.json").write_text(json.dumps(verdict, indent=2))
    print(f"  saved {OUT}/multilayer_demo_{HOST}.json", flush=True)


if __name__ == "__main__":
    main()
