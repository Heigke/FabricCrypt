"""
TASK C — Tournament racing (RO-pairs aggregated).

Treat 16 CPU cores as RO-pairs. For each window W (1 second = 50 samples),
extract a noise-variance summary per core (using cpufreq, cpu_util, or
derived rail-noise proxy if available) and run a 16-core single-elimination
bracket: pair (0,1), (2,3), ... winners advance, until 1 champion remains.
The winner-pattern is the bit string of bracket decisions (15 bits for 16
entrants). Measure intra-chassi Hamming-distance stability across 100
windows and inter-chassi Hamming distance.

Pre-reg: intra-HD <= 10% (1.5 bits / 15), inter-HD >= 40% (6 bits / 15).

We aggregate ALL cpu_util_cpuX and cpufreq_cpuX channels as "noise"
proxies (their windowed std).
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "results/IDENTITY_BENCHMARK_2026-05-30/embodiment8"
OUT = ROOT / "results/IDENTITY_BENCHMARK_2026-05-30/embodiment11a"
OUT.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(20260601)
WIN = 50  # 1 sec at 50 Hz
N_WINDOWS = 100
N_CORES = 16  # take first 16 cores


def load_core_signals(path):
    z = np.load(path)
    ch = list(map(str, z["channels"]))
    data = z["data"]
    # Use cpu_util_cpuX std as "rail noise" proxy (closest to per-core activity variance).
    util = []
    freq = []
    for i in range(N_CORES):
        if f"cpu_util_cpu{i}" in ch:
            util.append(data[:, ch.index(f"cpu_util_cpu{i}")])
        if f"cpufreq_cpu{i}" in ch:
            freq.append(data[:, ch.index(f"cpufreq_cpu{i}")])
    util = np.stack(util, axis=1) if util else None
    freq = np.stack(freq, axis=1) if freq else None
    return util, freq


def bracket_winner_bits(scores_16):
    """Return 15-bit pattern from a single-elim 16-entrant bracket.
    bit=1 means second entrant (higher index of the pair) won, else 0.
    Aggregates 8+4+2+1 = 15 match outcomes."""
    cur_idx = np.arange(16)
    cur_score = scores_16.copy()
    bits = []
    while len(cur_idx) > 1:
        next_idx, next_score = [], []
        for j in range(0, len(cur_idx), 2):
            a, b = cur_score[j], cur_score[j + 1]
            if b > a:
                next_idx.append(cur_idx[j + 1]); next_score.append(b); bits.append(1)
            else:
                next_idx.append(cur_idx[j]); next_score.append(a); bits.append(0)
        cur_idx = np.array(next_idx); cur_score = np.array(next_score)
    return np.array(bits, dtype=int)  # length 15


def patterns_for(machine_signal):
    # machine_signal: (T, 16)
    T, _ = machine_signal.shape
    starts = np.linspace(0, T - WIN - 1, N_WINDOWS).astype(int)
    pats = np.zeros((N_WINDOWS, 15), dtype=int)
    for k, s in enumerate(starts):
        w = machine_signal[s : s + WIN]
        score = w.std(axis=0)
        pats[k] = bracket_winner_bits(score)
    return pats


def hd(a, b):
    return int(np.sum(a != b))


def mean_intra_hd(pats):
    n = len(pats)
    s = 0; c = 0
    for i in range(n):
        for j in range(i + 1, n):
            s += hd(pats[i], pats[j]); c += 1
    return s / c if c else 0.0


def mean_inter_hd(pa, pb):
    n = min(len(pa), len(pb))
    return sum(hd(pa[i], pb[i]) for i in range(n)) / n


def run_one(name, util, freq):
    res = {}
    for tag, sig in [("util", util), ("freq", freq)]:
        if sig is None:
            res[tag] = None; continue
        pats = patterns_for(sig)
        # modal pattern = majority vote per bit
        modal = (pats.mean(axis=0) >= 0.5).astype(int)
        bit_stab = (pats == modal).mean(axis=0)
        res[tag] = {
            "intra_HD_mean": float(mean_intra_hd(pats)),
            "intra_HD_frac": float(mean_intra_hd(pats) / 15.0),
            "modal_pattern": modal.tolist(),
            "bit_stability_mean": float(bit_stab.mean()),
            "patterns_shape": list(pats.shape),
            "_patterns": pats.tolist(),
        }
    return res


def main():
    t0 = time.time()
    print("[C] Loading per-core signals ...")
    iu, ifr = load_core_signals(DATA / "ikaros_rich.npz")
    du, dfr = load_core_signals(DATA / "daedalus_rich.npz")
    print(f"[C] ikaros util {None if iu is None else iu.shape}  freq {None if ifr is None else ifr.shape}")
    print(f"[C] daedalus util {None if du is None else du.shape}  freq {None if dfr is None else dfr.shape}")

    res_i = run_one("ikaros", iu, ifr)
    res_d = run_one("daedalus", du, dfr)

    summary = {"task": "C_tournament_race", "win_samples": WIN, "n_windows": N_WINDOWS, "n_cores": N_CORES}
    for tag in ("util", "freq"):
        ri, rd = res_i.get(tag), res_d.get(tag)
        if ri is None or rd is None:
            summary[tag] = None; continue
        pi = np.array(ri.pop("_patterns")); pd = np.array(rd.pop("_patterns"))
        inter = float(mean_inter_hd(pi, pd))
        modal_hd = float(hd(np.array(ri["modal_pattern"]), np.array(rd["modal_pattern"])))
        summary[tag] = {
            "ikaros": ri,
            "daedalus": rd,
            "inter_HD_window_paired_mean": inter,
            "inter_HD_window_paired_frac": inter / 15.0,
            "inter_HD_modal_patterns": modal_hd,
            "inter_HD_modal_frac": modal_hd / 15.0,
            "prereg_PASS": bool(
                ri["intra_HD_frac"] <= 0.10
                and rd["intra_HD_frac"] <= 0.10
                and (modal_hd / 15.0) >= 0.40
            ),
        }
        print(
            f"[C] {tag}: intra_i={ri['intra_HD_frac']:.3f} intra_d={rd['intra_HD_frac']:.3f} "
            f"inter_modal={modal_hd/15.0:.3f}  PASS={summary[tag]['prereg_PASS']}"
        )

    summary["elapsed_s"] = round(time.time() - t0, 2)
    with open(OUT / "task_c_tournament.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[C] wrote {OUT/'task_c_tournament.json'}")


if __name__ == "__main__":
    main()
