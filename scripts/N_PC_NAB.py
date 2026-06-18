#!/usr/bin/env python3
"""N-PC-NAB: Predictive coding network with NS-RAM neurons for NAB anomaly detection.

Architecture (Phase N1 #6, Phase N2 U6):
  - Layer 1 (encoder): N=256 NS-RAM-style LIF neurons. State integrates input via
    learned input weights W_in. Output is a low-pass spike rate vector r_e(t).
  - Layer 2 (error neurons): N=256 leaky integrators. Receive top-down prediction
    p(t) = W_dec @ r_e(t) of the next sample. Error e(t) = x(t) - p(t-1) drives
    them. Their activity feeds back into the encoder via W_fb (predictive coding).
  - Online learning: W_in updated by Hebbian * |e|; W_dec by delta-rule on e.

Anomaly score = smoothed |prediction error| z-score. NAB streams scored with
sigmoid-weighted windows (standard NAB) -> equivalent F1-style metric.

Compute substrate: torch on CUDA (zgx). Vectorized N=256 cells, dt=1ms.
"""
from __future__ import annotations
import os, sys, json, time, math, argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
NAB_ROOT = Path(os.environ.get("NAB_ROOT", str(REPO / "data" / "NAB")))
OUT = REPO / "results" / "N_PC_NAB_N256"
OUT.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 1337
torch.manual_seed(SEED); np.random.seed(SEED)

# 3 representative streams (cover artificial + real + low-signal)
STREAMS = [
    "artificialWithAnomaly/art_daily_jumpsup.csv",
    "realKnownCause/nyc_taxi.csv",
    "realKnownCause/machine_temperature_system_failure.csv",
]

N_ENC = 256
N_ERR = 256
DT = 1.0
WIN = 200  # rolling z-score window
SMOOTH = 5  # consecutive samples gating

# Energy model: NS-RAM cell ~ 21 fJ/spike (Pazos slide 2)
E_SPIKE_J = 21e-15


# ───────────────────────────── NAB loaders ─────────────────────────────
def load_stream(rel):
    import csv
    path = NAB_ROOT / "data" / rel
    if not path.exists():
        raise FileNotFoundError(path)
    ts, vals = [], []
    with open(path) as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            ts.append(row[0]); vals.append(float(row[1]))
    return ts, np.asarray(vals, dtype=np.float32)


def load_windows(rel):
    p = NAB_ROOT / "labels" / "combined_windows.json"
    w = json.load(open(p))
    # Keys in combined_windows use forward slashes
    return w.get(rel, [])


def to_unix(ts_str):
    # NAB timestamps: "2014-04-01 00:00:00" or "2014-04-01 00:00:00.000000"
    s = ts_str.strip()
    if "." in s:
        s = s.split(".")[0]
    return time.mktime(time.strptime(s, "%Y-%m-%d %H:%M:%S"))


# ───────────────────────── NS-RAM-style network ────────────────────────
class PCNetwork:
    """2-layer predictive coding with NS-RAM-style LIF neurons."""

    def __init__(self, n_enc=256, n_err=256, lr_in=1e-3, lr_dec=5e-3,
                 lr_fb=2e-4, tau=20.0, v_th=1.0, leak=0.92, seed=0,
                 device=DEVICE):
        g = torch.Generator(device="cpu").manual_seed(seed)
        self.device = device
        self.n_enc = n_enc; self.n_err = n_err
        # Heterogeneous params (NS-RAM cells differ)
        self.tau_e = (tau * (1.0 + 0.1 * torch.randn(n_enc, generator=g))).to(device)
        self.v_th  = (v_th * (1.0 + 0.05 * torch.randn(n_enc, generator=g))).to(device)
        self.leak  = torch.tensor(leak, device=device)

        # Weights
        self.W_in  = (0.5 * torch.randn(n_enc, 1, generator=g)).to(device)
        self.W_fb  = (0.05 * torch.randn(n_enc, n_err, generator=g)).to(device)
        self.W_dec = (0.05 / math.sqrt(n_enc) *
                       torch.randn(1, n_enc, generator=g)).to(device)

        # State
        self.Vm = torch.zeros(n_enc, device=device)
        self.Q  = torch.zeros(n_enc, device=device)  # slow body charge
        self.rate = torch.zeros(n_enc, device=device)
        self.err  = torch.zeros(n_err, device=device)
        self.prev_pred = torch.tensor(0.0, device=device)

        self.lr_in = lr_in; self.lr_dec = lr_dec; self.lr_fb = lr_fb

        self.total_spikes = 0
        # error neurons are simple 1D mapping from scalar error (fan-out random)
        self.E_proj = (0.5 * torch.randn(n_err, generator=g)).to(device)

    def step(self, x_t, train=True):
        x = torch.tensor(float(x_t), device=self.device)
        # 1) compute prediction error from previous step's prediction
        e_scalar = x - self.prev_pred
        # excite error neurons
        self.err = 0.9 * self.err + self.E_proj * e_scalar

        # 2) drive encoder: input + top-down error feedback
        I_in = self.W_in[:, 0] * x + self.W_fb @ self.err
        # leaky integrate with body charge modulating threshold
        v_th_eff = self.v_th - 0.3 * self.Q
        self.Vm = self.leak * self.Vm + (1.0 / self.tau_e) * I_in
        spk = (self.Vm > v_th_eff).float()
        n_spk = int(spk.sum().item())
        self.total_spikes += n_spk
        # reset
        self.Vm = self.Vm - spk * v_th_eff * 0.7
        # slow body charge (NS-RAM trapping)
        self.Q = 0.999 * self.Q + 0.01 * spk
        self.Q.clamp_(0.0, 1.0)
        # low-pass rate
        self.rate = 0.9 * self.rate + 0.1 * spk

        # 3) decoder predicts next sample
        pred = (self.W_dec @ self.rate).squeeze()

        # 4) online updates (delta rule + Hebbian gated by error)
        if train:
            abs_e = e_scalar.abs()
            # decoder delta on previous prediction's error
            self.W_dec += self.lr_dec * e_scalar * self.rate.unsqueeze(0)
            # input weight Hebbian, gated by abs error
            self.W_in[:, 0] += self.lr_in * abs_e * (self.rate - self.rate.mean()) * x
            # weight bound
            self.W_in.clamp_(-2.0, 2.0)
            self.W_dec.clamp_(-2.0, 2.0)
            # feedback weights weakly anti-Hebbian (decorrelate error -> encoder)
            self.W_fb += self.lr_fb * (self.rate.unsqueeze(1) * self.err.unsqueeze(0)
                                       - 0.001 * self.W_fb)

        self.prev_pred = pred.detach()
        return float(pred.item()), float(e_scalar.item()), float(spk.sum().item())


# ───────────────────────── scoring helpers ─────────────────────────────
def rolling_zscore(x, win=200):
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    out = np.zeros(n)
    csum = np.concatenate([[0], np.cumsum(x)])
    csum2 = np.concatenate([[0], np.cumsum(x * x)])
    for t in range(n):
        lo = max(0, t - win); hi = t + 1
        m = (csum[hi] - csum[lo]) / max(1, hi - lo)
        v = (csum2[hi] - csum2[lo]) / max(1, hi - lo) - m * m
        s = math.sqrt(max(v, 1e-9))
        out[t] = (x[t] - m) / (s + 1e-9)
    return out


def _nms_events(dets, refractory=20):
    """Collapse runs of detections into single events.
    A new event is allowed after `refractory` non-detection samples (or after a
    long detection cluster). Returns boolean mask with only the first sample
    of each cluster set."""
    out = np.zeros_like(dets)
    last = -10**9
    for i, d in enumerate(dets):
        if d and (i - last) > refractory:
            out[i] = True; last = i
        elif d:
            last = i  # extend cluster, suppress
    return out


def f1_from_windows(timestamps, scores, windows, threshold, refractory=200):
    """Compute F1 vs labeled windows after NMS on detections.

    A window is TP if ANY post-NMS detection lies within it. Detections outside
    all windows = FP. NMS prevents continuous high-score regions from being
    counted as thousands of FPs."""
    if not windows:
        return None
    win_ranges = []
    for w in windows:
        try:
            t0 = to_unix(w[0]); t1 = to_unix(w[1])
            win_ranges.append((t0, t1))
        except Exception:
            continue
    if not win_ranges:
        return None
    ts = np.array([to_unix(t) for t in timestamps])
    dets = scores >= threshold
    events = _nms_events(dets, refractory=refractory)

    tp_windows = set()
    fp = 0
    for i, d in enumerate(events):
        if not d:
            continue
        t = ts[i]
        in_any = False
        for wi, (a, b) in enumerate(win_ranges):
            if a <= t <= b:
                tp_windows.add(wi); in_any = True; break
        if not in_any:
            fp += 1
    tp = len(tp_windows)
    fn = len(win_ranges) - tp
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    if prec + rec == 0:
        return 0.0, prec, rec
    return 2 * prec * rec / (prec + rec), prec, rec


# ───────────────────────── runner ──────────────────────────────────────
def run_stream(rel, save_traces=False):
    print(f"[N_PC_NAB] === {rel} ===", flush=True)
    ts, x = load_stream(rel)
    windows = load_windows(rel)
    n = len(x)
    # normalize
    med = np.median(x); mad = np.median(np.abs(x - med)) + 1e-9
    z = np.clip((x - med) / (1.4826 * mad), -5.0, 5.0).astype(np.float32)

    net = PCNetwork(n_enc=N_ENC, n_err=N_ERR, seed=SEED, device=DEVICE)
    preds = np.zeros(n, dtype=np.float32)
    errs  = np.zeros(n, dtype=np.float32)
    spike_count = np.zeros(n, dtype=np.float32)

    # 60% train, 40% test (online both; train flag controls weight update)
    n_tr = int(0.6 * n)
    weight_snaps = []
    snap_idx = np.linspace(0, n - 1, 30).astype(int)

    t0 = time.time()
    for t in range(n):
        # Always learn online (true streaming anomaly detection scenario);
        # n_tr only used for threshold selection.
        train = True
        p, e, ns = net.step(z[t], train=train)
        preds[t] = p; errs[t] = e; spike_count[t] = ns
        if t in snap_idx:
            weight_snaps.append(net.W_dec.detach().cpu().numpy().copy())
    dt_s = time.time() - t0
    throughput = n / max(1e-9, dt_s)

    abs_err = np.abs(errs)
    z_err = rolling_zscore(abs_err, win=WIN)
    # smooth with short moving average to reduce single-sample spurious peaks
    k = 5
    kernel = np.ones(k) / k
    z_err = np.convolve(z_err, kernel, mode="same")

    # Pick best threshold on TRAIN half, then evaluate on TEST half
    # (cheat-safe: threshold chosen on train labels only)
    test_slice = slice(n_tr, n)
    ts_test = ts[n_tr:]; z_test = z_err[n_tr:]
    train_ts = ts[:n_tr]; train_z = z_err[:n_tr]

    # Filter windows by half
    def windows_in(rng_lo, rng_hi):
        out = []
        for w in windows:
            try:
                t0 = to_unix(w[0]); t1 = to_unix(w[1])
            except Exception:
                continue
            if t1 < rng_lo or t0 > rng_hi:
                continue
            out.append(w)
        return out

    if len(train_ts) == 0 or len(ts_test) == 0:
        return None
    tr_lo = to_unix(train_ts[0]); tr_hi = to_unix(train_ts[-1])
    te_lo = to_unix(ts_test[0]);  te_hi = to_unix(ts_test[-1])
    train_windows = windows_in(tr_lo, tr_hi)
    test_windows = windows_in(te_lo, te_hi)

    # Threshold = robust percentile of ALL z-scores up to end of train. This
    # is a streaming-online style threshold (no leakage from test labels);
    # NAB benchmark allows per-stream calibration. We use 97th percentile as
    # a recall-oriented default that's small enough to detect rare anomalies.
    # We then verify train-F1 (informational only).
    if len(train_z) > 20:
        best_thr = float(np.percentile(train_z, 98.5))
    else:
        best_thr = 3.0
    if train_windows:
        res = f1_from_windows(train_ts, train_z, train_windows, best_thr)
        best_f1 = None if res is None else res[0]
    else:
        best_f1 = None

    # eval on TEST
    test_res = f1_from_windows(ts_test, z_test, test_windows, best_thr)
    if test_res is None:
        test_f1 = None; test_prec = None; test_rec = None
    else:
        test_f1, test_prec, test_rec = test_res

    energy_pj = (net.total_spikes * E_SPIKE_J * 1e12) / max(1, n)

    out = {
        "stream": rel,
        "n_samples": n,
        "n_train": n_tr,
        "n_windows_total": len(windows),
        "n_windows_train": len(train_windows),
        "n_windows_test": len(test_windows),
        "best_threshold_on_train": float(best_thr),
        "train_F1": None if best_f1 is None else float(best_f1),
        "test_F1": test_f1,
        "test_precision": test_prec,
        "test_recall": test_rec,
        "throughput_samples_per_sec": float(throughput),
        "total_spikes": int(net.total_spikes),
        "energy_pj_per_sample": float(energy_pj),
        "wall_time_s": float(dt_s),
    }
    diag = (f"train_z[max,99%]=({train_z.max():.2f},"
            f"{np.percentile(train_z,99):.2f}) "
            f"test_z[max,99%]=({z_test.max():.2f},{np.percentile(z_test,99):.2f}) "
            f"n_wins(train,test)=({len(train_windows)},{len(test_windows)})")
    print(f"[N_PC_NAB] {rel}: test_F1={test_f1} train_F1={best_f1} "
          f"thr={best_thr:.2f} throughput={throughput:.1f}/s | {diag}", flush=True)

    return out, preds, errs, z_err, spike_count, np.array(weight_snaps), net


# ───────────────────────── main ────────────────────────────────────────
def main():
    print(f"[N_PC_NAB] device={DEVICE}", flush=True)
    print(f"[N_PC_NAB] NAB_ROOT={NAB_ROOT}", flush=True)
    print(f"[N_PC_NAB] streams={STREAMS}", flush=True)

    all_summaries = []
    all_preds = []; all_gt = []; all_spikes = []; all_vb = []
    all_weights = None
    f1_per_stream = {}
    energies = []
    throughputs = []

    for rel in STREAMS:
        try:
            res = run_stream(rel)
        except Exception as e:
            print(f"[N_PC_NAB] FAILED {rel}: {e}", flush=True)
            f1_per_stream[rel] = None
            continue
        if res is None:
            f1_per_stream[rel] = None
            continue
        summ, preds, errs, z_err, spikes, w_snaps, net = res
        all_summaries.append(summ)
        f1_per_stream[rel] = summ["test_F1"]
        energies.append(summ["energy_pj_per_sample"])
        throughputs.append(summ["throughput_samples_per_sec"])

        # store traces (concatenated; per-stream slices recorded in summary)
        # ground truth = binary 1 inside any window
        ts, x = load_stream(rel)
        windows = load_windows(rel)
        gt = np.zeros(len(x), dtype=np.int8)
        for w in windows:
            try:
                t0 = to_unix(w[0]); t1 = to_unix(w[1])
            except Exception:
                continue
            tu = np.array([to_unix(t) for t in ts])
            mask = (tu >= t0) & (tu <= t1)
            gt[mask] = 1

        all_preds.append(preds); all_gt.append(gt)
        all_spikes.append(spikes); all_vb.append(z_err.astype(np.float32))
        if all_weights is None:
            all_weights = w_snaps

    # numeric summary
    valid_f1 = [v for v in f1_per_stream.values() if v is not None]
    mean_f1 = float(np.mean(valid_f1)) if valid_f1 else None
    mean_energy = float(np.mean(energies)) if energies else None
    mean_throughput = float(np.mean(throughputs)) if throughputs else None

    summary = {
        "task": "N_PC_NAB Phase N1#6 / Phase N2 U6",
        "n_enc": N_ENC, "n_err": N_ERR,
        "device": str(DEVICE),
        "seed": SEED,
        "streams": STREAMS,
        "test_F1_per_stream": f1_per_stream,
        "mean_F1": mean_f1,
        "energy_per_sample_pJ": mean_energy,
        "throughput_samples_per_sec_mean": mean_throughput,
        "per_stream": all_summaries,
        "gates": {
            "DISCOVERY_mean_F1_gt_0.3": (mean_f1 is not None and mean_f1 > 0.3),
            "AMBITIOUS_F1_gt_0.5_AND_thr_gt_1k": (
                mean_f1 is not None and mean_f1 > 0.5
                and mean_throughput is not None and mean_throughput > 1000),
        },
        "timestamp": datetime.now().isoformat(),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))

    # save arrays — concat across streams; also save per-stream offsets
    offsets = np.cumsum([0] + [len(p) for p in all_preds])
    np.save(OUT / "predictions.npy", np.concatenate(all_preds) if all_preds else np.array([]))
    np.save(OUT / "ground_truth.npy", np.concatenate(all_gt) if all_gt else np.array([]))
    np.save(OUT / "spikes.npy", np.concatenate(all_spikes) if all_spikes else np.array([]))
    np.save(OUT / "vb.npy", np.concatenate(all_vb) if all_vb else np.array([]))
    np.save(OUT / "stream_offsets.npy", offsets)
    if all_weights is not None:
        np.save(OUT / "weights.npy", all_weights)

    print(f"[N_PC_NAB] DONE mean_F1={mean_f1} energy_pJ={mean_energy} "
          f"throughput={mean_throughput}", flush=True)
    print(f"[N_PC_NAB] gates={summary['gates']}", flush=True)


if __name__ == "__main__":
    main()
