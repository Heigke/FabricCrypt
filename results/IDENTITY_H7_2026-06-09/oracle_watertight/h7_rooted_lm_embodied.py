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
