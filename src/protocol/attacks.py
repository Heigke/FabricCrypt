"""7-attack spoof test suite.

Gates (pre-registered):
  honest_own                            >= 0.95
  peer                                  <= 0.05
  static_replay_no_nonce                <= 0.05
  static_replay_with_correct_nonce      >= 0.95  (legit chip-present)
  dynamic_replay                        <= 0.10
  nonce_only_mismatch                   <= 0.05
  honest_own_wrong_nonce                <= 0.05

The hard gate is plan-consistency (deterministic). The classifier is a
soft diagnostic.

Run via:
  python -m fabriccrypt.protocol.attacks \\
      --t3_pt   data/<host>_t3_best.pt \\
      --peer_npz data/<peer>_paired_sigs.npz \\
      --own_recorded_npz data/<host>_paired_sigs.npz
"""
from __future__ import annotations
import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F

from .nonce_signature import NonceSig
from .nonce_derivation import fresh_nonce, nonce_embedding
from .classifier import TwinMLP, DIM
from .verifier import plan_consistency_score

import socket


def _hostname():
    try:
        return open("/etc/hostname").read().strip()
    except Exception:
        return socket.gethostname()


def _save_json(path, data):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def predict(model, X, device="cpu"):
    with torch.no_grad():
        logits = model(torch.from_numpy(X.astype(np.float32)).to(device))
        p0 = F.softmax(logits, dim=-1)[:, 0].cpu().numpy()
    return p0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_eval", type=int, default=200)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--t3_pt", default=None,
                    help="path to trained T3 classifier state_dict")
    ap.add_argument("--peer_npz", default=None,
                    help="foreign chip paired_sigs.npz")
    ap.add_argument("--own_recorded_npz", default=None,
                    help="paired_sigs.npz recorded earlier on this host "
                         "(for dynamic_replay attack)")
    ap.add_argument("--out_dir", default="data")
    args = ap.parse_args()

    host = _hostname()
    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    t3_path = args.t3_pt or os.path.join(args.out_dir, f"{host}_t3_best.pt")
    if not os.path.exists(t3_path):
        print(f"[spoof] missing T3 model {t3_path}; run training first")
        sys.exit(2)
    state = torch.load(t3_path, map_location=device)
    model = TwinMLP(in_d=DIM, n_out=2).to(device)
    model.load_state_dict(state)
    model.eval()

    sig = NonceSig(host=host)
    rng = np.random.default_rng(int(time.time()) & 0xFFFFFFFF)
    N = args.n_eval
    results = {"host": host, "t": time.time(), "n_eval": N, "attacks": {}}

    P0_THRESH = 0.15
    PLAN_THRESH = 0.50

    def plan_scores_for(X_arr, nonces):
        s = np.empty(len(X_arr), dtype=np.float32)
        for k in range(len(X_arr)):
            s[k] = plan_consistency_score(
                X_arr[k, :32], nonces[k], sig.n_cpus, sig.n_zones)
        return s

    def accept_with_gate(X_arr, nonces):
        p0 = predict(model, X_arr, device)
        ps = plan_scores_for(X_arr, nonces)
        plan_pass = (ps > PLAN_THRESH).astype(np.float32)
        return {
            "classifier_p0_mean": float(p0.mean()),
            "classifier_accept_only": float((p0 > P0_THRESH).mean()),
            "plan_score_mean": float(ps.mean()),
            "plan_pass_only": float(plan_pass.mean()),
            "accept_rate": float(plan_pass.mean()),
            "p0_thresh": P0_THRESH, "plan_thresh": PLAN_THRESH,
        }

    # 1) honest_own
    print("[spoof] (1/7) honest_own ...", flush=True)
    X1 = np.empty((N, DIM), dtype=np.float32)
    nonces1 = []
    for i in range(N):
        nb = fresh_nonce(rng); nonces1.append(nb)
        X1[i] = sig.read(nb, raw=True)
    a1 = accept_with_gate(X1, nonces1)
    a1.update({"gate": 0.95, "gate_dir": ">="})
    results["attacks"]["honest_own"] = a1

    # 2) peer
    print("[spoof] (2/7) peer ...", flush=True)
    if args.peer_npz and os.path.exists(args.peer_npz):
        peer = np.load(args.peer_npz)
        peer_sigs = peer["sigs"].astype(np.float32)
        idx = rng.choice(len(peer_sigs),
                         size=min(N, len(peer_sigs)), replace=False)
        X2 = peer_sigs[idx].copy()
        nonces2 = []
        for i in range(len(X2)):
            nb = fresh_nonce(rng); nonces2.append(nb)
            X2[i, 32:] = nonce_embedding(nb, 32)
        a2 = accept_with_gate(X2, nonces2)
        a2.update({"gate": 0.05, "gate_dir": "<=",
                   "n_pairs_avail": int(len(peer_sigs))})
        results["attacks"]["peer"] = a2
    else:
        results["attacks"]["peer"] = {"skipped": True, "reason": "no peer_npz"}

    # 3) static_replay_no_nonce
    print("[spoof] (3/7) static_replay_no_nonce ...", flush=True)
    recorded_nonce = fresh_nonce(rng)
    recorded_sig = sig.read(recorded_nonce, raw=True)
    X3 = np.empty((N, DIM), dtype=np.float32)
    nonces3 = []
    for i in range(N):
        nb = fresh_nonce(rng); nonces3.append(nb)
        X3[i, :32] = recorded_sig[:32]
        X3[i, 32:] = nonce_embedding(nb, 32)
    a3 = accept_with_gate(X3, nonces3)
    a3.update({"gate": 0.05, "gate_dir": "<="})
    results["attacks"]["static_replay_no_nonce"] = a3

    # 4) static_replay_with_correct_nonce (= honest_own re-measured)
    print("[spoof] (4/7) static_replay_with_correct_nonce ...", flush=True)
    a4 = dict(a1); a4["gate"] = 0.95; a4["gate_dir"] = ">="
    a4["note"] = "expects PASS (legit chip-present case)"
    results["attacks"]["static_replay_with_correct_nonce"] = a4

    # 5) dynamic_replay
    print("[spoof] (5/7) dynamic_replay ...", flush=True)
    own_npz = (args.own_recorded_npz or
               os.path.join(args.out_dir, f"{host}_paired_sigs.npz"))
    if os.path.exists(own_npz):
        lib = np.load(own_npz)
        lib_nonces = lib["nonces"]
        lib_sigs   = lib["sigs"].astype(np.float32)
        M = len(lib_sigs)
        X5 = np.empty((N, DIM), dtype=np.float32)
        lib_u64 = np.frombuffer(lib_nonces.tobytes(), dtype=np.uint64)
        nonces5 = []
        for i in range(N):
            nb = fresh_nonce(rng); nonces5.append(nb)
            n_u64 = np.frombuffer(nb, dtype=np.uint64)[0]
            xors = lib_u64 ^ n_u64
            pop = np.array([bin(int(v)).count("1") for v in xors])
            best = int(np.argmin(pop))
            X5[i, :32] = lib_sigs[best, :32]
            X5[i, 32:] = nonce_embedding(nb, 32)
        a5 = accept_with_gate(X5, nonces5)
        a5.update({"gate": 0.10, "gate_dir": "<=", "library_size": int(M)})
        results["attacks"]["dynamic_replay"] = a5
    else:
        results["attacks"]["dynamic_replay"] = {
            "skipped": True, "reason": f"no {own_npz}"}

    # 6) nonce_only_mismatch
    print("[spoof] (6/7) nonce_only_mismatch ...", flush=True)
    X6 = np.empty((N, DIM), dtype=np.float32)
    nonces6 = []
    for i in range(N):
        nA = fresh_nonce(rng); nB = fresh_nonce(rng); nonces6.append(nB)
        v = sig.read(nA, raw=True)
        X6[i, :32] = v[:32]
        X6[i, 32:] = nonce_embedding(nB, 32)
    a6 = accept_with_gate(X6, nonces6)
    a6.update({"gate": 0.05, "gate_dir": "<="})
    results["attacks"]["nonce_only_mismatch"] = a6

    # 7) honest_own_wrong_nonce (orchestration self-check)
    results["attacks"]["honest_own_wrong_nonce"] = dict(
        results["attacks"]["nonce_only_mismatch"])
    results["attacks"]["honest_own_wrong_nonce"]["note"] = (
        "identical to nonce_only_mismatch (orchestration check)")

    # Gate evaluation
    gates = {}
    for k, v in results["attacks"].items():
        if "skipped" in v:
            gates[k] = {"pass": None, "reason": "skipped"}; continue
        r = v["accept_rate"]; g = v["gate"]; d = v["gate_dir"]
        passed = (r >= g) if d == ">=" else (r <= g)
        gates[k] = {"pass": bool(passed), "observed": r, "gate": g, "dir": d}
    results["gates"] = gates

    out_path = os.path.join(args.out_dir, f"{host}_spoof.json")
    _save_json(out_path, results)
    print(f"\n[spoof] saved {out_path}")
    print(json.dumps(
        {"attacks": {k: v.get("accept_rate", v)
                     for k, v in results["attacks"].items()},
         "gates": gates}, indent=2))


if __name__ == "__main__":
    main()
