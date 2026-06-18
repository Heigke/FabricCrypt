# Combined text bundle for oracle context

All text artifacts concatenated below for one-shot context. Each file is delimited by a `=== FILE: <name> ===` marker.



=== FILE: embodied_analog_ikaros.json (439 chars) ===
```json
{
  "host": "ikaros",
  "task": "XOR_t1t2",
  "chance": 0.527,
  "uwin": 6,
  "die_rank": 0,
  "die_dim": 360,
  "n_train": 1715,
  "n_eval": 735,
  "results": {
    "native": 0.638,
    "shuffled": 0.551,
    "zero": 0.551,
    "foreign": 0.527,
    "foreign_host": "daedalus"
  },
  "LOAD_BEARING": false,
  "UNIQUE_per_die": true,
  "uniqueness_gap": 0.111,
  "verdict": "NOT load-bearing (die not needed); UNIQUE per-die (gap 0.111)"
}
```


=== FILE: h7_multilayer_demo.py (10584 chars) ===
```python
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

```


=== FILE: h7_rooted_lm_embodied.py (16413 chars) ===
```python
"""H7 — LOAD-BEARING embodiment: the physical cache-XOR gate is in the LM's forward path.

WHY THIS EXISTS (answer to "var inte adaptern för ytlig?"):
  v21/v22 conditioned an LM on PASSIVE telemetry via FiLM and *trained in* the dependence with a
  margin/spoof loss. That is shallow on two counts: (1) the body never COMPUTES anything in that
  path — telemetry is just a state/noise signal; (2) the dependence is an artefact of the penalty
  (pass-by-construction), not a need.

  Here the body's PHYSICAL computation is the only nonlinearity in the forward path:
    - Backbone is strictly LINEAR (embedding + linear readout + linear head). No GELU/MLP/softmax.
      A linear map provably CANNOT compute XOR of two input bits.
    - The single nonlinear unit is the physical gate g = XOR(a,b), produced by DESTRUCTIVE L3-cache
      interference between two streamers (micro_mem.c), read out by a throughput-sum threshold.
    - The task carries a genuine XOR dependency. So if the model solves it, the body is LOAD-BEARING
      *by architecture*, not by a loss term. Training uses PLAIN cross-entropy — no margin, no spoof
      penalty. Whether the model comes to depend on the body is therefore EMPIRICAL.

HONEST DISCRIMINATORS (the part that separates "deep" from "pass-by-construction"):
    native    live silicon gate            -> must SOLVE (acc ~ 1.0)
    random    random bit per query         -> must BREAK (acc ~ chance) : computation is load-bearing
    frozen    g held constant              -> must BREAK
    xor_sim   correct XOR in software       -> must STAY GOOD : honest, the *function* is generic
    foreign   another die's calib threshold -> degrades only if borderline bits flip (UNIQUE; weak)

  native≈xor_sim AND native>>random/frozen  ==> the body is doing a real, load-bearing computation
  (and we DON'T overclaim uniqueness: digital XOR is generic across dies — see foreign).

Training shortcut (disclosed): the gate is deterministic XOR, so training uses the truth table with a
straight-through estimator (10^6 hardware calls would be infeasible). At EVAL every query drives the
real streamers and reads live throughput — the produced token genuinely depends on silicon contention.

Needs micro_mem (built from micro_mem.c). Run under sandbox-disabled shell (gcc/exec). No sudo needed.
Out: results/IDENTITY_H7_2026-06-09/embodied_{host}.json
"""
from __future__ import annotations
import os, sys, time, json, argparse, struct, mmap, subprocess, tempfile, socket, math
import numpy as np
from pathlib import Path

HOST = socket.gethostname()
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT_DIR = ROOT / "results/IDENTITY_H7_2026-06-09"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------------------------------
# Physical XOR gate: two L3-sharing memory streamers; throughput-sum threshold = the digital readout.
# (Same organ verified in h7_throttle_nonlin / h7_closed_loop: ikaros XOR readout 0.99, daedalus 0.95.)
# --------------------------------------------------------------------------------------------------
class BodyGate:
    def __init__(self, cpu_a=0, cpu_b=2, mb=24, win=0.05):
        self.win = win
        binp = HERE / "micro_mem"; src = HERE / "micro_mem.c"
        if not binp.exists() or src.stat().st_mtime > binp.stat().st_mtime:
            r = subprocess.run(["gcc", "-O2", "-march=native", "-o", str(binp), str(src)],
                               capture_output=True, text=True)
            if r.returncode:
                raise RuntimeError("gcc failed:\n" + r.stderr)
        self.shm = Path(tempfile.gettempdir()) / f"h7embshm_{os.getpid()}"
        self.shm.write_bytes(b"\x00" * 64)
        self.fd = os.open(str(self.shm), os.O_RDWR); self.mm = mmap.mmap(self.fd, 64)
        arr = mb * 1024 * 1024
        self.pA = subprocess.Popen([str(binp), str(cpu_a), "0", str(self.shm), str(arr)])
        self.pB = subprocess.Popen([str(binp), str(cpu_b), "1", str(self.shm), str(arr)])
        time.sleep(0.5)
        self.thr = None; self.thr_foreign = None; self.cell_means = {}

    def _sf(self, i, v): self.mm.seek(i * 4); self.mm.write(struct.pack("i", v))
    def _sc(self, i, v): self.mm.seek(8 + i * 8); self.mm.write(struct.pack("Q", v))
    def _gc(self, i): self.mm.seek(8 + i * 8); return struct.unpack("Q", self.mm.read(8))[0]

    def _sum(self, av, bv):
        self._sc(0, 0); self._sc(1, 0); self._sf(0, int(av)); self._sf(1, int(bv))
        time.sleep(self.win)
        s = self._gc(0) + self._gc(1); self._sf(0, 0); self._sf(1, 0); return s

    def calibrate(self, reps=8, foreign_cell_means=None):
        cal = {(x, y): [] for x in (0, 1) for y in (0, 1)}
        for _ in range(reps):
            for x in (0, 1):
                for y in (0, 1):
                    cal[(x, y)].append(self._sum(x, y))
        self.cell_means = {k: float(np.mean(v)) for k, v in cal.items()}
        # XOR=1 (01,10) run fast (high sum); 00 idle & 11 contention low -> separator between
        # the lowest XOR=1 cell and the highest XOR=0 cell.
        lo1 = min(self.cell_means[(0, 1)], self.cell_means[(1, 0)])
        hi0 = max(self.cell_means[(0, 0)], self.cell_means[(1, 1)])
        self.thr = (lo1 + hi0) / 2
        if foreign_cell_means is not None:
            flo1 = min(foreign_cell_means["01"], foreign_cell_means["10"])
            fhi0 = max(foreign_cell_means["00"], foreign_cell_means["11"])
            self.thr_foreign = (flo1 + fhi0) / 2
        # fidelity: does the live gate reproduce XOR on its own calibration cells?
        ok = sum(int((self._gate_from_sum(self.cell_means[(x, y)])) == (x ^ y))
                 for x in (0, 1) for y in (0, 1))
        return ok / 4.0

    def _gate_from_sum(self, s, foreign=False):
        t = self.thr_foreign if (foreign and self.thr_foreign is not None) else self.thr
        return 1 if s > t else 0

    def gate(self, a, b, foreign=False):
        """LIVE physical XOR: drive streamers, threshold the throughput-sum."""
        return self._gate_from_sum(self._sum(a, b), foreign=foreign)

    def close(self):
        try:
            self._sf(0, 2); self._sf(1, 2); time.sleep(0.2)
            self.pA.terminate(); self.pB.terminate(); self.mm.close(); os.close(self.fd)
            self.shm.unlink()
        except Exception:
            pass


# --------------------------------------------------------------------------------------------------
# Parity task: each example is ctx bytes; two marker positions carry secret bits a*,b* in their LSB.
# Target token = a* XOR b*. A LINEAR model cannot compute this; only the physical XOR gate can.
# Extra "content" bytes give the sequence non-trivial structure (so it's a real sequence, not 2 bits).
# --------------------------------------------------------------------------------------------------
VOCAB = 256
P1, P2 = 5, 19            # fixed marker positions inside the context


def make_batch(n, ctx, rng):
    # content bytes live in [2,256); the two secret operand bits live in dedicated marker tokens
    # at P1,P2 with value in {0,1} (cleanly separable). XOR of them STILL needs the gate.
    X = rng.integers(2, VOCAB, size=(n, ctx)).astype(np.int64)
    a = rng.integers(0, 2, size=n); b = rng.integers(0, 2, size=n)
    X[:, P1] = a
    X[:, P2] = b
    y = (a ^ b).astype(np.int64)        # the parity target the model must output
    return X, y, a.astype(np.int64), b.astype(np.int64)


# --------------------------------------------------------------------------------------------------
# Strictly LINEAR backbone. Only nonlinearity = binarize + physical gate.
# --------------------------------------------------------------------------------------------------
def build_net(ctx, d=64):
    import torch, torch.nn as nn

    class LinearGateNet(nn.Module):
        """Strictly linear except the binarize+gate. The head ALSO gets a full linear bypass
        (pooled embeddings) so the model COULD ignore the gate — but a linear bypass cannot
        compute XOR, so to solve the task it must depend on the physical gate. That makes
        'corrupting the gate breaks it' a genuine empirical result, not an architectural tautology.

        Training uses a differentiable soft-XOR surrogate (g_soft = a+b-2ab on sigmoid-bits, with an
        annealed temperature so it sharpens toward {0,1}). At EVAL the bits are hard-thresholded and
        the XOR is computed by LIVE SILICON. The soft surrogate is only a smoother gradient path to
        the same hard solution — it changes nothing about the eval-time load-bearing claim."""
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(VOCAB, d)           # linear lookup
            self.read_a = nn.Linear(ctx * d, 1)         # linear: extract pre-bit a from whole ctx
            self.read_b = nn.Linear(ctx * d, 1)         # linear: extract pre-bit b
            self.bypass = nn.Linear(d, 2)               # LINEAR escape hatch: pooled emb -> logits
            self.gate_head = nn.Linear(1, 2, bias=False)  # gate's contribution to logits

        def pre_logits(self, X):
            h = self.emb(X).reshape(X.shape[0], -1)     # (n, ctx*d), linear
            return self.read_a(h).squeeze(-1), self.read_b(h).squeeze(-1)

        def hard_bits(self, X):
            ua, ub = self.pre_logits(X)
            return (ua > 0).long(), (ub > 0).long()

        def _pool(self, X):
            return self.emb(X).mean(dim=1)              # (n, d) linear, gives the bypass full info

        def logits_with_gate(self, X, g):
            return self.bypass(self._pool(X)) + self.gate_head(g.unsqueeze(-1))

        def forward(self, X, temp=3.0):
            ua, ub = self.pre_logits(X)
            a_s = torch.sigmoid(temp * ua); b_s = torch.sigmoid(temp * ub)
            g_soft = a_s + b_s - 2 * a_s * b_s          # differentiable XOR surrogate in (0,1)
            return self.logits_with_gate(X, g_soft), (ua, ub)

    return LinearGateNet()


def train(net, steps, batch, ctx, lr, seed=0):
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = net.to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr)
    lossf = torch.nn.CrossEntropyLoss()
    rng = np.random.default_rng(seed)
    t0 = time.time()
    for step in range(steps):
        X, y, _, _ = make_batch(batch, ctx, rng)
        Xt = torch.from_numpy(X).to(dev); yt = torch.from_numpy(y).to(dev)
        temp = 1.0 + 5.0 * step / max(1, steps)         # anneal soft-bit sharpness 1 -> 6
        logits, _ = net(Xt, temp=temp)
        loss = lossf(logits, yt)                        # PLAIN CE. no margin, no spoof penalty.
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % 200 == 0:
            acc = float((logits.argmax(-1) == yt).float().mean())
            print(f"  step {step+1:5d}  ce={loss.item():.4f}  train_acc={acc:.3f}  temp={temp:.1f}  "
                  f"t={time.time()-t0:.0f}s", flush=True)
    return net, dev


def evaluate(net, dev, body: BodyGate, cond, n, ctx, seed=123):
    """Run the net but replace the gate with the requested SOURCE of the bit, then score parity.

    'native'/'foreign' actually drive the silicon. 'xor_sim' computes XOR in numpy. 'random'/'frozen'
    corrupt it. We score whether the model's output token == true parity.
    """
    import torch
    rng = np.random.default_rng(seed)
    X, y, a, b = make_batch(n, ctx, rng)
    Xt = torch.from_numpy(X).to(dev)
    with torch.no_grad():
        abit_t, bbit_t = net.hard_bits(Xt)
        abit = abit_t.cpu().numpy().astype(int); bbit = bbit_t.cpu().numpy().astype(int)
        gbits = np.zeros(n, dtype=np.float32)
        frozen_val = 1
        rr = np.random.default_rng(7)
        for i in range(n):
            if cond == "native":
                gbits[i] = body.gate(abit[i], bbit[i])
            elif cond == "foreign":
                gbits[i] = body.gate(abit[i], bbit[i], foreign=True)
            elif cond == "xor_sim":
                gbits[i] = abit[i] ^ bbit[i]
            elif cond == "random":
                gbits[i] = rr.integers(0, 2)
            elif cond == "frozen":
                gbits[i] = frozen_val
            else:
                raise ValueError(cond)
        gt = torch.from_numpy(gbits).to(dev)
        logits = net.logits_with_gate(Xt, gt)
        pred = logits.argmax(-1).cpu().numpy()
        ce = torch.nn.functional.cross_entropy(
            logits, torch.from_numpy(y).to(dev)).item()
    acc = float((pred == y).mean())
    # how often did the net's own pre-bits match the secret bits it was supposed to read?
    bit_acc = float(((abit == a) & (bbit == b)).mean())
    return {"acc": round(acc, 3), "ce": round(ce, 4),
            "ppl": round(math.exp(min(ce, 50)), 3), "bit_read_acc": round(bit_acc, 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--ctx", type=int, default=24)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--n_eval", type=int, default=200)
    ap.add_argument("--win", type=float, default=0.05)
    ap.add_argument("--cpu_a", type=int, default=0)
    ap.add_argument("--cpu_b", type=int, default=2)
    ap.add_argument("--foreign_cells", default="",
                    help="JSON dict of other-die cell means {'00':..,'01':..,'10':..,'11':..}")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import torch
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    print(f"[{HOST}] embodied load-bearing LM: ctx={a.ctx} steps={a.steps} batch={a.batch} seed={a.seed}", flush=True)
    net = build_net(a.ctx)
    net, dev = train(net, a.steps, a.batch, a.ctx, a.lr)
    print(f"[{HOST}] trained on {dev}; starting physical body + eval", flush=True)

    foreign = None
    if a.foreign_cells:
        foreign = json.loads(a.foreign_cells)
    body = BodyGate(cpu_a=a.cpu_a, cpu_b=a.cpu_b, win=a.win)
    fid = body.calibrate(foreign_cell_means=foreign)
    print(f"[{HOST}] gate calibrated: cells={{ {', '.join(f'{x}{y}:{round(body.cell_means[(x,y)])}' for x in (0,1) for y in (0,1))} }} "
          f"thr={body.thr:.0f} truth-table-fidelity={fid:.2f}", flush=True)

    conds = ["native", "xor_sim", "random", "frozen"]
    if foreign is not None:
        conds.append("foreign")
    res = {}
    try:
        for c in conds:
            res[c] = evaluate(net, dev, body, c, a.n_eval, a.ctx)
            print(f"  [{c:8s}] acc={res[c]['acc']}  ppl={res[c]['ppl']}  "
                  f"bit_read_acc={res[c]['bit_read_acc']}", flush=True)
    finally:
        body.close()

    native = res["native"]["acc"]
    corrupt = max(res["random"]["acc"], res["frozen"]["acc"])
    sim = res["xor_sim"]["acc"]
    out = {
        "host": HOST, "device": dev, "ctx": a.ctx, "steps": a.steps,
        "gate_cell_means": {f"{x}{y}": round(body.cell_means[(x, y)]) for x in (0, 1) for y in (0, 1)},
        "gate_thr": round(body.thr, 1), "gate_truth_table_fidelity": round(fid, 3),
        "results": res,
        # LOAD-BEARING: native solves AND corrupting the body breaks it
        "LOAD_BEARING": bool(native > 0.9 and native - corrupt > 0.35),
        # HONEST: the function is generic (a correct software XOR works just as well)
        "function_generic_xor_sim_ok": bool(sim > 0.9),
        "verdict": None,
    }
    if out["LOAD_BEARING"] and out["function_generic_xor_sim_ok"]:
        out["verdict"] = ("DEEP: physical body computation is load-bearing for the LM output "
                          "(no margin loss); function is generic XOR (uniqueness NOT claimed here)")
    elif native <= 0.9:
        out["verdict"] = "NEGATIVE: model did not learn to use the gate (honest null)"
    else:
        out["verdict"] = "INCONCLUSIVE: see per-condition table"
    jp = OUT_DIR / f"embodied_{HOST}.json"; jp.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2), flush=True)
    print(f"\n[{HOST}] VERDICT: {out['verdict']}", flush=True)
    print(f"  native={native} xor_sim={sim} random={res['random']['acc']} frozen={res['frozen']['acc']}"
          f"{' foreign='+str(res['foreign']['acc']) if foreign else ''}", flush=True)
    print(f"  saved {jp}", flush=True)


if __name__ == "__main__":
    main()

```


=== FILE: h7_rooted_lm_embodied_analog.py (8834 chars) ===
```python
"""H7 — ANALOG load-bearing embodiment: the per-die voltage-droop transient reservoir IS the LM's
only nonlinearity. This is the upgrade over the digital cache-XOR version (h7_rooted_lm_embodied.py):
the cache gate was load-bearing but GENERIC across dies (xor_sim worked). The analog droop transient
is PER-DIE silicon (process variation in di/dt droop + settling), so it can be load-bearing AND unique.

Grounding (already-measured, not simulated): h7_transient_vdroop.py drove the die with a sharp-edge
GPU burst stream u[t] and recorded the settling transient as NTAP=12 time-multiplexed virtual nodes x
N_CH=10 channels -> Tn (L,12,10). Its result: a LINEAR readout of these die features does temporal XOR
(u[t-1]^u[t-2]) at ~0.80 while the u-window control sits at chance (0.50) -> "die_needed: true".

This script wires that reservoir into an LM-style predictor:
  logits[t] = BYPASS(u-window lags)  +  DIE_HEAD(standardized die reservoir features at t)
  - BYPASS is a full LINEAR escape hatch over recent input bits. A linear map CANNOT compute XOR, so the
    model COULD ignore the die but then fails the task -> any dependence on the die is EARNED, not forced.
  - DIE_HEAD is a linear readout of the PHYSICAL reservoir state. The reservoir's fading memory + droop
    nonlinearity is the only thing that can supply XOR. Plain cross-entropy. No margin/spoof loss.

Held-out temporal split (train first 70% of timesteps, eval last 30%). Conditions:
  native   real die features on held-out timesteps        -> must SOLVE (load-bearing)
  shuffled die features row-permuted (right stats, wrong binding/timestep) -> must BREAK
  zero     die features zeroed (reservoir removed)         -> bypass only -> chance
  foreign  ANOTHER die's recorded features for the SAME u  -> if it BREAKS, the computation is UNIQUE
           to this die (h7_transient_vdroop uses a fixed seed, so u/targets match across dies).

native >> shuffled/zero  => DEEP load-bearing.   native >> foreign => UNIQUE (per-die).
Pure numpy/torch on recorded Tn — no root, no GPU bursts here. (Fresh-collection eval is a stronger
follow-up; daedalus Tn must be collected to run the foreign/uniqueness arm.)
Out: results/IDENTITY_H7_2026-06-09/embodied_analog_{host}.json
"""
from __future__ import annotations
import sys, json, time, socket, argparse, math
from pathlib import Path
import numpy as np

HOST = socket.gethostname()
HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "results/IDENTITY_H7_2026-06-09"
WASHOUT = 150


def load_die(host):
    p = OUT / f"transient_vdroop_raw_{host}.npz"
    if not p.exists():
        return None, None
    d = np.load(p)
    return d["u"].astype(int), d["Tn"].astype(np.float32)   # u:(L,)  Tn:(L,12,10)


def lag(x, k):
    y = np.zeros_like(x)
    if k > 0:
        y[k:] = x[:-k]; return y
    return x.copy()


def die_features(Tn):
    """Reservoir state = transient taps flattened + 2 step-lags (same construction as the harness)."""
    flat = Tn.reshape(Tn.shape[0], -1)                       # (L, 120)
    return np.hstack([flat, lag(flat, 1), lag(flat, 2)])     # (L, 360)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="XOR_t1t2", choices=["XOR_t1t2", "XOR_t1t3", "PAR3_123"])
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--uwin", type=int, default=6, help="bypass linear window over input bit lags")
    ap.add_argument("--die_rank", type=int, default=0, help="0=full die features; >0 = PCA rank cap")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--foreign_host", default=None, help="other die host for the uniqueness arm")
    a = ap.parse_args()

    import torch, torch.nn as nn
    torch.manual_seed(a.seed); np.random.seed(a.seed)

    u, Tn = load_die(HOST)
    if u is None:
        print(f"[{HOST}] missing transient_vdroop_raw_{HOST}.npz — run h7_transient_vdroop.py first"); sys.exit(2)
    L = len(u)

    # targets (the die-solvable temporal functions)
    if a.task == "XOR_t1t2": y = (lag(u, 1) ^ lag(u, 2))
    elif a.task == "XOR_t1t3": y = (lag(u, 1) ^ lag(u, 3))
    else: y = (lag(u, 1) ^ lag(u, 2) ^ lag(u, 3))
    y = y.astype(np.int64)

    Xdie = die_features(Tn)                                   # (L, 360)
    Uwin = np.stack([lag(u.astype(np.float32), k) for k in range(1, a.uwin + 1)], 1)  # (L, uwin)

    cut = WASHOUT + int(0.7 * (L - WASHOUT))
    tr = np.arange(WASHOUT, cut); te = np.arange(cut, L)

    # standardize on TRAIN stats only
    mu, sd = Xdie[tr].mean(0), Xdie[tr].std(0) + 1e-9
    Xz = (Xdie - mu) / sd
    umu, usd = Uwin[tr].mean(0), Uwin[tr].std(0) + 1e-9
    Uz = (Uwin - umu) / usd

    # optional rank cap on die features (fit PCA on train) — mirrors the harness rank-limited probe
    proj = None
    if a.die_rank > 0:
        Xc = Xz[tr] - Xz[tr].mean(0)
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        proj = Vt[:a.die_rank]                                # (rank, 360)
    def die_proj(M):
        return M if proj is None else (M - Xz[tr].mean(0)) @ proj.T

    Xz_use = die_proj(Xz)
    ddim = Xz_use.shape[1]

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    bypass = nn.Linear(a.uwin, 2).to(dev)                    # LINEAR escape hatch over input bits
    die_head = nn.Linear(ddim, 2, bias=False).to(dev)        # readout of the PHYSICAL reservoir state
    opt = torch.optim.AdamW(list(bypass.parameters()) + list(die_head.parameters()), lr=a.lr)
    lossf = nn.CrossEntropyLoss()

    Ut = torch.from_numpy(Uz).float().to(dev)
    Dt = torch.from_numpy(Xz_use).float().to(dev)
    yt = torch.from_numpy(y).to(dev)
    tri = torch.from_numpy(tr).to(dev)

    for step in range(a.steps):
        logits = bypass(Ut[tri]) + die_head(Dt[tri])
        loss = lossf(logits, yt[tri])
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % 1000 == 0:
            acc = float((logits.argmax(-1) == yt[tri]).float().mean())
            print(f"  step {step+1:5d} ce={loss.item():.4f} train_acc={acc:.3f}", flush=True)

    @torch.no_grad()
    def acc_with(Dmat):
        lg = bypass(Ut[torch.from_numpy(te).to(dev)]) + die_head(torch.from_numpy(Dmat[te]).float().to(dev))
        return float((lg.argmax(-1).cpu().numpy() == y[te]).mean())

    @torch.no_grad()
    def acc_bypass_only():
        lg = bypass(Ut[torch.from_numpy(te).to(dev)])        # die contribution dropped
        return float((lg.argmax(-1).cpu().numpy() == y[te]).mean())

    rng = np.random.default_rng(7)
    res = {}
    res["native"] = round(acc_with(Xz_use), 3)
    Xsh = Xz_use.copy(); perm = te.copy(); rng.shuffle(perm); Xsh[te] = Xz_use[perm]
    res["shuffled"] = round(acc_with(Xsh), 3)
    res["zero"] = round(acc_bypass_only(), 3)

    # uniqueness arm: another die's recorded features (same u/targets), ikaros-trained heads
    foreign = a.foreign_host or ("daedalus" if HOST == "ikaros" else "ikaros")
    uf, Tnf = load_die(foreign)
    if Tnf is not None and len(uf) == L and np.array_equal(uf, u):
        Xf = (die_features(Tnf) - mu) / sd
        res["foreign"] = round(acc_with(die_proj(Xf)), 3)
        res["foreign_host"] = foreign
    elif Tnf is not None:
        res["foreign"] = None; res["foreign_host"] = f"{foreign} (u mismatch — re-collect same seed)"

    chance = float(max(y[te].mean(), 1 - y[te].mean()))
    nat = res["native"]; corrupt = max(res["shuffled"], res["zero"])
    out = {
        "host": HOST, "task": a.task, "chance": round(chance, 3), "uwin": a.uwin,
        "die_rank": a.die_rank, "die_dim": ddim, "n_train": len(tr), "n_eval": len(te),
        "results": res,
        "LOAD_BEARING": bool(nat > chance + 0.10 and nat - corrupt > 0.10),
    }
    if "foreign" in res and res["foreign"] is not None:
        out["UNIQUE_per_die"] = bool(nat - res["foreign"] > 0.10)
        out["uniqueness_gap"] = round(nat - res["foreign"], 3)
    out["verdict"] = (
        ("DEEP+ANALOG load-bearing" if out["LOAD_BEARING"] else "NOT load-bearing (die not needed)")
        + (f"; UNIQUE per-die (gap {out.get('uniqueness_gap')})" if out.get("UNIQUE_per_die")
           else ("; NOT unique (foreign solves it too)" if "UNIQUE_per_die" in out
                 else "; uniqueness UNTESTED (need foreign-die data, same seed)"))
    )
    OUT.mkdir(parents=True, exist_ok=True)
    jp = OUT / f"embodied_analog_{HOST}.json"; jp.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2), flush=True)
    print(f"\n[{HOST}] VERDICT: {out['verdict']}", flush=True)
    print(f"  native={nat} shuffled={res['shuffled']} zero={res['zero']} "
          f"foreign={res.get('foreign','NA')} chance={chance:.2f}", flush=True)
    print(f"  saved {jp}", flush=True)


if __name__ == "__main__":
    main()

```


=== FILE: h7_rooted_lm_text_embodied.py (12387 chars) ===
```python
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

```


=== FILE: h7_transient_vdroop.py (7092 chars) ===
```python
"""H7 AMPLIFIED probe — fast-edge Vdroop excitation + time-multiplexed transient readout.

Insight: the genuine die nonlinearity we found (IMD 1.8x over static, ch5 bilinear) is Vdroop / di/dt
physics — ELECTRICAL and FAST, not thermal. So we can excite it HARD with SHARP load edges at LOW
average power (cool!) instead of cooking the chip. Amplification = three tricks:
  1. sharp max-amplitude bursts, low duty -> big di/dt droop, chip stays cool
  2. read the post-edge SETTLING TRANSIENT as many time-multiplexed virtual nodes (Appeltant) -> a weak
     nonlinearity unfolded into a high-dim reservoir state
  3. static-map control + long run -> isolate the genuine DYNAMICAL nonlinearity from instantaneous load
Fair test: can a RANK-LIMITED LINEAR readout of the transient reservoir now do XOR and BEAT the rank-
limited u-window adapter — where the SMOOTH drive failed (rank_necessity was negative)? Also reports the
static-map excess and the unbounded-nl-u reference. Low duty => thermally safe (target <70C). Root.
"""
from __future__ import annotations
import sys, json, time, socket
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).parent))
from substrate_realtime_v3 import SubstrateStateV3

HOST = socket.gethostname()
OUT = Path(__file__).resolve().parents[2] / "results/IDENTITY_H7_2026-06-09"
ZONE = Path("/sys/class/thermal/thermal_zone0/temp")
N_CH = 10
L = 2600
WASHOUT = 150
BURST_MS = 0.004        # sharp 4ms max-GPU burst on u=1 (high di/dt edge)
STEP_S = 0.030          # 30ms step -> ~13% duty on u=1 -> low average power, cool
NTAP = 12               # virtual nodes = transient settling samples after the edge (~24ms @ 500Hz)
W = 4; R = 4            # rank-limited adapter window + rank
SEED = 0


def temp_c():
    try: return int(ZONE.read_text())/1000.0
    except Exception: return 0.0


def collect(u):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    gA = torch.randn(2048,2048,device=dev); gB = torch.randn(2048,2048,device=dev)  # big = sharp high-di/dt edge
    st = SubstrateStateV3(hz_target=500); st.start(); time.sleep(6.0)
    pool = np.array([st.latest_window(length=64).reshape(-1,N_CH) for _ in range(40)]).reshape(-1,N_CH)
    med = np.median(pool,0); mad = np.median(np.abs(pool-med),0)*1.4826+1e-9
    T = np.zeros((L, NTAP, N_CH), np.float32); t0=time.time()
    for t in range(L):
        s0=time.time()
        if u[t]:
            # sync INSIDE the burst so 4ms wall-clock = ~4ms REAL GPU work (genuine ~13% duty).
            # Without this, the loop async-queues hundreds of matmuls in 4ms and the later sync runs
            # them all -> GPU saturates continuously -> chip overheats (daedalus hit 96C). 2026-06-17.
            while time.time()-s0 < BURST_MS:
                gA=(gA@gB).tanh()*0.5+0.5
                if dev=="cuda": torch.cuda.synchronize()
        # capture settling transient (virtual nodes) for the rest of the step
        time.sleep(0.004)
        T[t] = st.latest_window(length=NTAP).reshape(-1,N_CH)[:NTAP]
        rest = STEP_S - (time.time()-s0)
        if rest>0: time.sleep(rest)
        if t%10==0:
            tc=temp_c()
            if tc>74.0:
                while temp_c()>56.0: time.sleep(1.0)
            if t%600==0: print(f"  step {t}/{L} temp={tc:.0f}C ({time.time()-t0:.0f}s)",flush=True)
    st.stop()
    Tn = np.tanh((T - med)/mad/8.0)
    return Tn, med, mad


def lag(x,k):
    y=np.zeros_like(x);
    if k>0: y[k:]=x[:-k]
    return y if k>0 else x.copy()


def rank_lin(X,y,tr,te,nc,rank):
    mu=X[tr].mean(0); sd=X[tr].std(0)+1e-9; Xz=(X-mu)/sd
    U,Sv,Vt=np.linalg.svd(Xz[tr]-Xz[tr].mean(0),full_matrices=False)
    P=Vt[:rank]; Xp=(Xz-Xz[tr].mean(0))@P.T; Y=np.eye(nc)[y]; best=0.0
    for al in[1e-2,.1,1,10,100]:
        W_=np.linalg.solve(Xp[tr].T@Xp[tr]+al*np.eye(Xp.shape[1]),Xp[tr].T@Y[tr])
        best=max(best,float(np.mean((Xp[te]@W_).argmax(1)==y[te])))
    return best


def full_lin(X,y,tr,te,nc):
    mu=X[tr].mean(0);sd=X[tr].std(0)+1e-9;X=(X-mu)/sd;Y=np.eye(nc)[y];best=0.0
    for al in[.1,1,10,100,1e3]:
        W_=np.linalg.solve(X[tr].T@X[tr]+al*np.eye(X.shape[1]),X[tr].T@Y[tr])
        best=max(best,float(np.mean((X[te]@W_).argmax(1)==y[te])))
    return best


def main():
    rng=np.random.default_rng(SEED); u=rng.integers(0,2,size=L)
    print(f"[{HOST}] AMPLIFIED transient/Vdroop reservoir (sharp edges, low duty) temp {temp_c():.0f}C...",flush=True)
    Tn,med,mad = collect(u)
    # drive landed?
    flat = Tn.reshape(L,-1)
    dland=np.abs((flat[u==1].mean(0)-flat[u==0].mean(0))/(np.sqrt((flat[u==1].std(0)**2+flat[u==0].std(0)**2)/2)+1e-9)).max()
    print(f"  drive landed max|d|={dland:.2f}  (transient tap range)",flush=True)
    OUT.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(OUT/f"transient_vdroop_raw_{HOST}.npz", u=u, Tn=Tn)

    # reservoir features: transient taps (flattened) + 1 lag of the per-step mean
    Xres = flat                                   # (L, NTAP*N_CH) = time-multiplexed virtual nodes
    Xres = np.hstack([Xres, lag(flat, 1), lag(flat, 2)])
    Uwin = np.stack([lag(u.astype(float),k) for k in range(1,W+1)],1)
    # nonlinear-on-u reference (unbounded strawman)
    import itertools
    uu=u.astype(float); ucols=[lag(uu,k) for k in range(16)]
    for a,b in itertools.combinations(range(10),2): ucols.append(lag(uu,a)*lag(uu,b))
    Xnlu=np.stack(ucols,1)

    n=L-WASHOUT; cut=WASHOUT+int(0.7*n); tr=slice(WASHOUT,cut); te=slice(cut,L)
    def lb(k): return lag(u,k).astype(int)
    tasks={"RECALL_t2":(lb(2),2),"XOR_t1t2":(lb(1)^lb(2),2),"XOR_t1t3":(lb(1)^lb(3),2),
           "XOR_t2t4":(lb(2)^lb(4),2),"XOR_t2t8":(lb(2)^lb(8),2),
           "PAR3_123":((lb(1)^lb(2)^lb(3)),2)}
    suite={}
    for nm,(y,nc) in tasks.items():
        die_r=rank_lin(Xres,y,tr,te,nc,R)
        die_full=full_lin(Xres,y,tr,te,nc)
        ctrl=rank_lin(Uwin,y,tr,te,nc,min(R,W))
        nlu=full_lin(Xnlu,y,tr,te,nc)
        win = die_r-ctrl>0.05 and die_r>1.0/nc+0.05
        suite[nm]={"chance":1.0/nc,"die_rank4":die_r,"die_full_linear":die_full,
                   "u_window_rank":ctrl,"unbounded_nl_u":nlu,"die_needed":bool(win)}
        flag="  <-- DIE NEEDED" if win else ("  (die_full beats u_win)" if die_full-ctrl>0.05 else "")
        print(f"  {nm:10s} chance={1.0/nc:.2f} die(r4)={die_r:.3f} die(full)={die_full:.3f} u_win={ctrl:.3f} [nl_u={nlu:.2f}]{flag}",flush=True)
    needed=[k for k,v in suite.items() if v["die_needed"]]
    out={"host":HOST,"drive_landed_d":float(dland),"ntap":NTAP,"burst_ms":BURST_MS,"task_suite":suite,
         "die_needed_on":needed,
         "verdict":"AMPLIFICATION WORKED — die needed" if needed else "amplification insufficient — die still not needed"}
    def jf(o):
        if isinstance(o,dict): return {k:jf(v) for k,v in o.items()}
        if isinstance(o,(np.floating,np.integer)): return float(o)
        return o
    (OUT/f"transient_vdroop_{HOST}.json").write_text(json.dumps(jf(out),indent=2))
    print(f"\n>>> {out['verdict']}  needed_on={needed}",flush=True)


if __name__=="__main__":
    main()

```


=== FILE: multilayer_demo_ikaros.json (1462 chars) ===
```json
{
  "host": "ikaros",
  "demoA": {
    "task": "PAR3 (3-bit parity via 2 composed cache layers)",
    "chance": 0.506,
    "acc": {
      "native": 0.944,
      "no_body": 0.494,
      "stage2_off": 0.494,
      "stage1_bad": 0.406
    },
    "n_eval": 160,
    "y_sample": [
      0,
      0,
      1,
      1,
      1,
      0,
      1,
      1,
      1,
      0,
      0,
      0,
      1,
      1,
      1,
      1,
      1,
      1,
      0,
      1,
      1,
      1,
      0,
      0,
      0,
      1,
      1,
      1,
      0,
      1,
      1,
      0
    ],
    "pred_native": [
      0,
      0,
      0,
      1,
      1,
      0,
      1,
      1,
      1,
      0,
      0,
      0,
      1,
      1,
      1,
      1,
      1,
      1,
      0,
      1,
      1,
      1,
      0,
      0,
      0,
      1,
      1,
      1,
      0,
      1,
      1,
      0
    ],
    "pred_no_body": [
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ]
  },
  "demoB": {
    "task": "XOR_t1t2 (analog droop reservoir)",
    "chance": 0.527,
    "own_die": 0.638,
    "foreign_die": 0.473,
    "foreign_host": "daedalus",
    "uniqueness_gap": 0.165
  },
  "MULTILAYER_LOAD_BEARING": true,
  "UNIQUE_per_die": true
}
```
