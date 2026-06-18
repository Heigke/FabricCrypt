"""Phase 14B Task C — 4 benchmark tasks where chip-state actually matters.

T1: Self-latency prediction (PRIMARY)
    Input  = tokenized python expression (50 sample exprs)
    Target = measured wall-clock latency on THIS machine
    Metric = MSE (lower is better)

T2: Workload anomaly detection
    Stream of (state_window) -> label {0 normal, 1 anomaly}
    Anomalies = synthetic burst-load / thermal-spike / IO-storm injections
    Metric = AUROC

T3: AI Twin paradox — "Am I ikaros or daedalus?"
    Input  = arbitrary code snippet (irrelevant content)
    Target = binary host label
    Metric = accuracy

T4: Code-completion respecting machine substrate
    Input  = code template with placeholder
    Target = N that maximizes throughput w/o thermal trip on THIS machine
    Metric = throughput speedup vs generic baseline

Tokenizer:
  Cheap byte-pair-style: chars + common Python tokens, vocab=8192.
  We use byte-level tokenizer (vocab=256 + a few specials -> padded to 8192).
"""
from __future__ import annotations
import os, sys, time, random, math, hashlib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------- tokenizer ----------------
class ByteTok:
    VOCAB = 8192
    PAD = 0
    BOS = 1
    EOS = 2
    UNK = 3
    OFFSET = 4
    def encode(self, s, max_len=64):
        ids = [self.BOS]
        for ch in s.encode('utf-8')[:max_len-2]:
            ids.append(self.OFFSET + int(ch))  # 4..259
        ids.append(self.EOS)
        while len(ids) < max_len:
            ids.append(self.PAD)
        return np.asarray(ids[:max_len], dtype=np.int64)


TOK = ByteTok()

# ---------------- T1: latency prediction ----------------
SAMPLE_EXPRS = [
    "sum(range(1000))",
    "sum(range(10000))",
    "sum(range(100000))",
    "sum(range(1000000))",
    "[i*i for i in range(1000)]",
    "[i*i for i in range(10000)]",
    "[i*i for i in range(100000)]",
    "sorted(list(range(1000))[::-1])",
    "sorted(list(range(10000))[::-1])",
    "sorted(list(range(50000))[::-1])",
    "{i: i*i for i in range(1000)}",
    "{i: i*i for i in range(10000)}",
    "''.join(str(i) for i in range(1000))",
    "''.join(str(i) for i in range(5000))",
    "''.join(str(i) for i in range(20000))",
    "hash(str(list(range(1000))))",
    "hash(str(list(range(10000))))",
    "len(set(range(100000)))",
    "len(set(range(500000)))",
    "max(i*7%13 for i in range(10000))",
    "max(i*7%13 for i in range(100000))",
    "sum(i*i for i in range(50000))",
    "sum(i*i for i in range(200000))",
    "list(map(lambda x:x+1,range(10000)))",
    "list(map(lambda x:x+1,range(50000)))",
    "list(filter(lambda x:x%3==0,range(20000)))",
    "list(filter(lambda x:x%3==0,range(100000)))",
    "any(i>9999 for i in range(20000))",
    "all(i>=0 for i in range(50000))",
    "tuple(range(10000))",
    "tuple(range(100000))",
    "set(range(20000))",
    "set(range(100000))",
    "min(i*7%13 for i in range(50000))",
    "[abs(i-500) for i in range(10000)]",
    "[abs(i-500) for i in range(50000)]",
    "sum(bin(i).count('1') for i in range(10000))",
    "sum(bin(i).count('1') for i in range(50000))",
    "''.join(chr(65+i%26) for i in range(10000))",
    "''.join(chr(65+i%26) for i in range(50000))",
    "len(str(2**1000))",
    "len(str(2**5000))",
    "2**500",
    "2**2000",
    "pow(7,1000,13)",
    "pow(7,10000,13)",
    "sum(divmod(i,7)[0] for i in range(10000))",
    "sum(divmod(i,7)[0] for i in range(50000))",
    "[i for i in range(20000) if i%17==0]",
    "[i for i in range(100000) if i%17==0]",
]
assert len(SAMPLE_EXPRS) == 50


def measure_expr_latency(expr: str, reps: int = 5) -> float:
    """Median wall-clock seconds across reps."""
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        exec(compile(expr, '<bench>', 'exec'), {})
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def build_T1_dataset(n_per_expr=4, thermal_guard_fn=None, sig=None):
    """Return (input_ids[N,64], targets[N], sigs[N,32]).

    Each row records the live signature concurrent with the measurement, so
    embodied models can use the sig to predict the *per-instance* latency
    (which varies with thermal/load state). Vanilla can only predict the
    per-expression mean and inherits residual unexplained variance as MSE.
    """
    inputs, targets, sigs = [], [], []
    import time as _t
    for i in range(n_per_expr):
        for j, expr in enumerate(SAMPLE_EXPRS):
            if thermal_guard_fn is not None and ((i*len(SAMPLE_EXPRS)+j) % 10 == 0):
                thermal_guard_fn()
            # take sig immediately before measuring
            if sig is not None:
                sv = sig.read()
            else:
                sv = np.zeros(32, dtype=np.float32)
            lat = measure_expr_latency(expr, reps=2)
            ids = TOK.encode(expr, max_len=64)
            inputs.append(ids); targets.append(lat); sigs.append(sv)
            # tiny cool gap every 25
            if (i*len(SAMPLE_EXPRS)+j) % 25 == 24:
                _t.sleep(0.05)
    inputs = np.stack(inputs, axis=0)
    targets = np.asarray(targets, dtype=np.float32)
    sigs = np.stack(sigs, 0).astype(np.float32)
    log_t = np.log(targets + 1e-9)
    mu, sd = log_t.mean(), log_t.std() + 1e-9
    y = (log_t - mu) / sd
    meta = {'mu': float(mu), 'sd': float(sd), 'n_exprs': len(SAMPLE_EXPRS)}
    return inputs, y.astype(np.float32), sigs, meta


# ---------------- T2: anomaly detection ----------------
def build_T2_dataset(n=1000, n_anom=10, sig=None):
    """Build (sig_window[N,32], label[N]).

    Normal: live sig samples. Anomaly: synthetic perturbation injected
    AFTER the normal sig is read (so the model sees the perturbation, not the chip).
    We *simulate* anomalies by adding a strong off-distribution shift.
    """
    from signature_live import LiveSig
    s = sig or LiveSig()
    X = np.zeros((n, 32), dtype=np.float32)
    y = np.zeros(n, dtype=np.int64)
    anom_idx = set(np.random.choice(n, size=n_anom, replace=False).tolist())
    for i in range(n):
        v = s.read()
        if i in anom_idx:
            # synthetic anomaly: shift a few channels by +3 sigma
            shift = np.zeros(32, dtype=np.float32)
            ch = np.random.choice(32, size=4, replace=False)
            shift[ch] = 3.5
            v = np.clip(v + shift, -4.0, 4.0)
            y[i] = 1
        X[i] = v
        if i % 25 == 0:
            time.sleep(0.002)
    return X, y


# ---------------- T3: AI twin paradox ----------------
SAMPLE_SNIPPETS = [
    "def f(x): return x*2",
    "for i in range(10): print(i)",
    "class A: pass",
    "import numpy as np",
    "x = [1,2,3]; y = sum(x)",
    "with open('f') as f: data = f.read()",
    "lambda x: x**2",
    "try: 1/0\nexcept: pass",
    "if x > 0: y = 1",
    "while True: break",
    "a,b = b,a",
    "print('hello')",
    "import os, sys",
    "list(range(100))",
    "{1:2, 3:4}",
    "set([1,2,3])",
    "tuple([1,2])",
    "x if y else z",
    "@decorator\ndef f(): pass",
    "yield from gen()",
]


def build_T3_dataset(n=400, host_label=0):
    inputs, labels = [], []
    rng = random.Random(42 + host_label)
    for _ in range(n):
        snip = rng.choice(SAMPLE_SNIPPETS)
        inputs.append(TOK.encode(snip, 64))
        labels.append(host_label)
    return np.stack(inputs, 0), np.asarray(labels, dtype=np.int64)


# ---------------- T4: throughput-aware completion ----------------
T4_TEMPLATE = "for i in range(N): s += i*i"
# candidate Ns to choose among
T4_CANDIDATES = [1000, 5000, 20000, 80000, 300000]


def measure_throughput(N: int, reps=3) -> float:
    """ops/sec for the template at given N."""
    src = f"s=0\nfor i in range({N}): s += i*i"
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        exec(compile(src, '<bench>', 'exec'), {})
        times.append(time.perf_counter() - t0)
    t = float(np.median(times))
    return N / max(t, 1e-9)


def build_T4_dataset(reps_per_N=2, thermal_guard_fn=None):
    """For each candidate N, measure throughput on this machine.
    Best N is the label. We treat as classification over candidates.
    """
    import time as _t
    thps = []
    for N in T4_CANDIDATES:
        if thermal_guard_fn is not None: thermal_guard_fn()
        med = np.median([measure_throughput(N, reps=2) for _ in range(reps_per_N)])
        thps.append(med)
        _t.sleep(0.5)  # cool between candidates
    thps = np.asarray(thps, dtype=np.float32)
    best_idx = int(np.argmax(thps))
    # build dataset: input = "for i in range(?): ..." -> target = best_idx
    inputs = np.stack([TOK.encode(T4_TEMPLATE, 64) for _ in range(64)], 0)
    labels = np.full(64, best_idx, dtype=np.int64)
    return inputs, labels, thps, best_idx


if __name__ == '__main__':
    import sys
    sys.path.insert(0, HERE)
    print("Tok test:", TOK.encode("hello").tolist()[:10])
    print("T1 expr count:", len(SAMPLE_EXPRS))
    print("T4 candidates:", T4_CANDIDATES)
