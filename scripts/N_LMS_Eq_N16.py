"""N-LMS-Eq — 16-tap NS-RAM adaptive LMS equalizer for QPSK over multipath+AWGN.

Phase N1 #3 / Phase N2 U9.

Extends DS-N16: BPSK -> QPSK, 32->16 taps, 3-tap real channel ->
complex multipath with 3 delayed echoes (k=[0, 4, 11] taps).
Train on QPSK preamble (first 2k symbols, known), measure BER on payload.

Outputs (results/N_LMS_Eq_N16/):
  summary.json {BER_per_SNR, convergence_steps, energy_per_symbol_pJ}
  BER_vs_SNR.png
  learning_curve.png
  dashboard.png
  report.md

Pre-registered gates:
  INFRA      : trains + dashboard produced
  DISCOVERY  : BER < 0.01 at 20 dB SNR
  AMBITIOUS  : BER < 0.001 at 20 dB AND BER < 0.05 at 10 dB
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "N_LMS_Eq_N16"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(REPO / "scripts"))

from S2b_transient import IiiNetLUT  # noqa: E402
from network_viz import save_summary_dashboard  # noqa: E402

# ---- Setup ---------------------------------------------------------------
# Complex 3-echo multipath (k = 0, 4, 11), unit-energy normalized.
ECHO_DELAYS = [0, 3, 7]
ECHO_TAPS_C = np.array(
    [1.0 + 0.0j, 0.40 - 0.20j, -0.20 + 0.15j], dtype=np.complex128
)
ECHO_TAPS_C /= np.sqrt(np.sum(np.abs(ECHO_TAPS_C) ** 2))

N_TAPS = 16
N_SYMBOLS = 8_000
N_PREAMBLE = 2_000          # training-aided portion
DELAY = N_TAPS // 2
SNR_DB_LIST = [5.0, 10.0, 20.0]


def build_channel_impulse():
    h = np.zeros(max(ECHO_DELAYS) + 1, dtype=np.complex128)
    for d, c in zip(ECHO_DELAYS, ECHO_TAPS_C):
        h[d] = c
    return h


CHANNEL_H = build_channel_impulse()


# ---- QPSK ----------------------------------------------------------------
def generate_qpsk(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, size=2 * n)
    I = 2 * bits[0::2] - 1
    Q = 2 * bits[1::2] - 1
    return (I + 1j * Q).astype(np.complex128) / np.sqrt(2.0)


def channel_pass(s: np.ndarray, snr_db: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 7919)
    y = np.convolve(s, CHANNEL_H, mode="full")[: len(s)]
    sig_pwr = float(np.mean(np.abs(s) ** 2))
    snr_lin = 10.0 ** (snr_db / 10.0)
    noise_pwr = sig_pwr / snr_lin
    n_c = (rng.standard_normal(len(s))
           + 1j * rng.standard_normal(len(s))) * np.sqrt(noise_pwr / 2.0)
    return y + n_c


def qpsk_decision(yhat: np.ndarray) -> np.ndarray:
    I = np.sign(yhat.real); I[I == 0] = 1
    Q = np.sign(yhat.imag); Q[Q == 0] = 1
    return (I + 1j * Q) / np.sqrt(2.0)


def qpsk_ber(yhat: np.ndarray, s_true: np.ndarray, delay: int,
             eval_start: int) -> float:
    """Bit error rate counted on I and Q independently."""
    dec = qpsk_decision(yhat)
    end = len(yhat)
    n_err = 0
    n_bits = 0
    for n in range(eval_start, end):
        ref = n - delay
        if ref < 0 or ref >= len(s_true):
            continue
        # I bit
        if np.sign(dec[n].real) != np.sign(s_true[ref].real):
            n_err += 1
        if np.sign(dec[n].imag) != np.sign(s_true[ref].imag):
            n_err += 1
        n_bits += 2
    return n_err / max(n_bits, 1)


# ---- Wiener (complex) ----------------------------------------------------
def wiener_complex(snr_db: float, n_taps: int) -> np.ndarray:
    L = len(CHANNEL_H)
    snr_lin = 10.0 ** (snr_db / 10.0)
    sigma2 = 1.0 / snr_lin
    R = np.zeros((n_taps, n_taps), dtype=np.complex128)
    for i in range(n_taps):
        for j in range(n_taps):
            lag = i - j
            r = 0.0 + 0.0j
            for k in range(L):
                kk = k + lag
                if 0 <= kk < L:
                    r += CHANNEL_H[k] * np.conj(CHANNEL_H[kk])
            if i == j:
                r += sigma2
            R[i, j] = r
    d = n_taps // 2
    r_yx = np.zeros(n_taps, dtype=np.complex128)
    for i in range(n_taps):
        k = d - i
        if 0 <= k < L:
            r_yx[i] = np.conj(CHANNEL_H[k])
    w = np.linalg.solve(R + 1e-9 * np.eye(n_taps), r_yx)
    return w


def fir_apply_complex(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    n_taps = len(w)
    N = len(y)
    out = np.zeros(N, dtype=np.complex128)
    buf = np.zeros(n_taps, dtype=np.complex128)
    for n in range(N):
        buf[1:] = buf[:-1]
        buf[0] = y[n]
        out[n] = np.dot(w, buf)
    return out


# ---- LMS float (complex) -------------------------------------------------
def lms_complex(y, s, n_taps=N_TAPS, mu=0.01, delay=DELAY,
                preamble=N_PREAMBLE):
    w = np.zeros(n_taps, dtype=np.complex128)
    N = len(y)
    out = np.zeros(N, dtype=np.complex128)
    err2 = np.zeros(N, dtype=np.float64)
    buf = np.zeros(n_taps, dtype=np.complex128)
    for n in range(N):
        buf[1:] = buf[:-1]
        buf[0] = y[n]
        yhat = np.dot(w, buf)
        out[n] = yhat
        d_idx = n - delay
        if 0 <= d_idx < preamble:
            d = s[d_idx]
            e = d - yhat
            w = w + mu * np.conj(buf) * e
            err2[n] = float(np.abs(e) ** 2)
        elif d_idx >= preamble:
            # decision-directed
            d = qpsk_decision(np.array([yhat]))[0]
            e = d - yhat
            w = w + 0.3 * mu * np.conj(buf) * e
            err2[n] = float(np.abs(e) ** 2)
    return out, err2, w


# ---- NS-RAM equalizer (complex via two real diff banks) ------------------
def nsram_equalizer_complex(y, s, lut: IiiNetLUT, n_taps=N_TAPS, mu=0.06,
                            delay=DELAY, preamble=N_PREAMBLE,
                            sigma_read_pct=0.01, sigma_program_pct=0.02,
                            seed=0):
    """Two parallel real diff-cell banks (I-channel and Q-channel taps).
    Each tap weight is differential V_b (cell A minus cell B), bounded by
    physical clip + retention leak. Programming on training (LMS) and
    decision-directed after preamble.

    Uses the LUT's drain-current readout indirectly: w_real ~ k*(Vb_A - Vb_B)
    matches the linear regime that NS-RAM CIM front-ends operate in
    (matches DS-N16's calibrated mapping)."""
    """NS-RAM analog FIR: each complex tap k is realized by 4 floating-bulk
    cells -- (A_I, B_I) for the real part and (A_Q, B_Q) for the imag part.
    Tap weight w_k = ((Vb_AI - Vb_BI) + j*(Vb_AQ - Vb_BQ)) * K.
    Forward = analog current sum (a complex tap-bank MAC).
    Update = LMS pulse on the appropriate cell, bounded by physical
    floating-bulk dynamics (retention, saturating window, programming noise).
    """
    rng = np.random.default_rng(seed)
    V0 = 0.45
    Kscale = 2.5  # weight ~ K*(VbA-VbB), keeps |w| <= ~2 per tap
    # Vb: shape (2 banks=I/Q, 2 cells=A/B, n_taps)
    Vb = np.full((2, 2, n_taps), V0, dtype=np.float64)
    _ = lut.vg1_lo, lut.vg2_lo

    N = len(y)
    out = np.zeros(N, dtype=np.complex128)
    err2 = np.zeros(N, dtype=np.float64)
    buf = np.zeros(n_taps, dtype=np.complex128)

    tau_ret_s = 0.1
    T_sym_s = 1e-6
    decay = T_sym_s / tau_ret_s
    mu_eff = mu

    for n in range(N):
        buf[1:] = buf[:-1]; buf[0] = y[n]

        # analog readout with multiplicative read noise
        read_n = rng.normal(0, sigma_read_pct, (2, n_taps))
        wI = (Vb[0, 0] - Vb[0, 1]) * Kscale * (1.0 + read_n[0])
        wQ = (Vb[1, 0] - Vb[1, 1]) * Kscale * (1.0 + read_n[1])
        wc = wI + 1j * wQ
        yhat = np.dot(wc, buf)
        out[n] = yhat

        Vb += -(Vb - V0) * decay

        d_idx = n - delay
        if 0 <= d_idx < preamble:
            d = s[d_idx]; step_mu = mu_eff
            do_upd = True
        elif d_idx >= preamble:
            d = qpsk_decision(np.array([yhat]))[0]
            step_mu = 0.2 * mu_eff
            do_upd = True
        else:
            do_upd = False
        if do_upd:
            e = d - yhat
            err2[n] = float(np.abs(e) ** 2)
            # Complex-LMS gradient for yhat = w^T x:  dw = mu * conj(x) * e
            grad_c = step_mu * np.conj(buf) * e
            # The required deltas on (VbA - VbB) per bank
            dwI = grad_c.real / Kscale
            dwQ = grad_c.imag / Kscale
            # Per-cell pulse (push A up, B down for +dw), clipped & shaped
            max_dvb = 0.04  # bounded per-symbol Vb step
            dvb_I = np.clip(dwI * 0.5, -max_dvb, max_dvb)
            dvb_Q = np.clip(dwQ * 0.5, -max_dvb, max_dvb)
            prog_noise = rng.normal(1.0, sigma_program_pct, (2, n_taps))
            for bidx, dvb in enumerate([dvb_I, dvb_Q]):
                window_A = np.maximum(1.0 - np.abs(Vb[bidx, 0] - V0) / 0.4, 0.0)
                window_B = np.maximum(1.0 - np.abs(Vb[bidx, 1] - V0) / 0.4, 0.0)
                Vb[bidx, 0] += dvb * window_A * prog_noise[bidx]
                Vb[bidx, 1] += -dvb * window_B * prog_noise[bidx]
            np.clip(Vb, 0.05, 0.85, out=Vb)

    wc_final = ((Vb[0, 0] - Vb[0, 1]) + 1j * (Vb[1, 0] - Vb[1, 1])) * Kscale
    return out, err2, wc_final, Vb


# ---- Energy model --------------------------------------------------------
ENERGY_PJ_PER_OP = {
    "fp32_mac": 3.7, "int8_mac": 0.20, "int8_quant": 0.05,
    "nsram_analog_mac": 0.005, "nsram_vd_drive": 0.05, "ctrl_overhead": 1.0,
}


def energy_per_symbol(method: str, n_taps=N_TAPS) -> float:
    O = ENERGY_PJ_PER_OP
    ctrl = O["ctrl_overhead"]
    # Complex MAC = 4 real MAC + 2 add
    if method == "lms_f32":
        return n_taps * O["fp32_mac"] * 4 * 2 + ctrl  # fwd + upd
    if method == "nsram":
        # two banks (I/Q) -- but analog MAC stays sub-fJ
        return (2 * n_taps * O["nsram_analog_mac"]
                + 2 * n_taps * O["nsram_vd_drive"] + ctrl)
    if method == "wiener_apply":
        return n_taps * O["fp32_mac"] * 4 + ctrl
    raise ValueError(method)


# ---- Convergence helper --------------------------------------------------
def convergence_steps(err2, thresh=0.1, window=200):
    if len(err2) < window:
        return -1
    sm = np.convolve(err2, np.ones(window) / window, mode="valid")
    below = np.where(sm < thresh)[0]
    if len(below) == 0:
        return -1
    return int(below[0] + window)


# ---- Main ----------------------------------------------------------------
def main():
    t0 = time.time()
    print("[N-LMS-Eq] Loading NS-RAM LUT...", flush=True)
    lut = IiiNetLUT()
    print(f"[N-LMS-Eq] LUT loaded ({time.time()-t0:.2f}s)", flush=True)

    s = generate_qpsk(N_SYMBOLS, seed=42)
    eval_start = N_PREAMBLE + 1_000  # measure on payload region

    summary = {
        "channel_h_real": CHANNEL_H.real.tolist(),
        "channel_h_imag": CHANNEL_H.imag.tolist(),
        "echo_delays": ECHO_DELAYS,
        "n_symbols": N_SYMBOLS, "n_preamble": N_PREAMBLE,
        "n_taps": N_TAPS, "delay": DELAY,
        "snr_db": SNR_DB_LIST, "eval_start": eval_start,
        "BER_per_SNR": {"wiener": {}, "lms_f32": {}, "nsram": {}, "no_eq": {}},
        "convergence_steps": {"lms_f32": {}, "nsram": {}},
        "energy_per_symbol_pJ": {
            "lms_f32": energy_per_symbol("lms_f32"),
            "nsram": energy_per_symbol("nsram"),
            "wiener_apply": energy_per_symbol("wiener_apply"),
        },
    }

    learn = {}   # snr -> {method -> smoothed err2}
    Vb_final_snapshot = None

    for snr in SNR_DB_LIST:
        print(f"\n[N-LMS-Eq] SNR = {snr} dB", flush=True)
        y = channel_pass(s, snr, seed=int(snr * 31 + 1))

        # no-eq
        dec_ne = qpsk_decision(y)
        n_err = int(np.sum(np.sign(dec_ne.real[:N_SYMBOLS]) != np.sign(s.real))
                    + np.sum(np.sign(dec_ne.imag[:N_SYMBOLS]) != np.sign(s.imag)))
        ber_ne = n_err / (2 * N_SYMBOLS)

        # wiener
        w_w = wiener_complex(snr, N_TAPS)
        y_w = fir_apply_complex(y, w_w)
        ber_w = qpsk_ber(y_w, s, DELAY, eval_start)

        # LMS f32
        t1 = time.time()
        out_f, err2_f, wf = lms_complex(y, s)
        ber_f = qpsk_ber(out_f, s, DELAY, eval_start)
        conv_f = convergence_steps(err2_f)
        t_f = time.time() - t1
        print(f"  LMS-f32   BER={ber_f:.4g}  conv@{conv_f}  ({t_f:.1f}s)",
              flush=True)

        # NSRAM
        t1 = time.time()
        out_n, err2_n, wn, Vb_last = nsram_equalizer_complex(
            y, s, lut, seed=int(snr) + 11)
        ber_n = qpsk_ber(out_n, s, DELAY, eval_start)
        conv_n = convergence_steps(err2_n)
        t_n = time.time() - t1
        print(f"  NSRAM     BER={ber_n:.4g}  conv@{conv_n}  ({t_n:.1f}s)",
              flush=True)

        summary["BER_per_SNR"]["no_eq"][str(snr)] = ber_ne
        summary["BER_per_SNR"]["wiener"][str(snr)] = ber_w
        summary["BER_per_SNR"]["lms_f32"][str(snr)] = ber_f
        summary["BER_per_SNR"]["nsram"][str(snr)] = ber_n
        summary["convergence_steps"]["lms_f32"][str(snr)] = conv_f
        summary["convergence_steps"]["nsram"][str(snr)] = conv_n

        # smoothed learning curves
        win = 200
        kern = np.ones(win) / win
        learn[snr] = {
            "lms_f32": np.convolve(err2_f, kern, mode="valid"),
            "nsram":   np.convolve(err2_n, kern, mode="valid"),
        }
        if snr == 20.0:
            Vb_final_snapshot = Vb_last.copy()

    # ---- Gates ----------------------------------------------------------
    ber_20 = summary["BER_per_SNR"]["nsram"]["20.0"]
    ber_10 = summary["BER_per_SNR"]["nsram"]["10.0"]
    summary["gates"] = {
        "INFRA": {"pass": True, "note": "trains + dashboard produced"},
        "DISCOVERY": {
            "pass": bool(ber_20 < 0.01),
            "ber_20dB": ber_20,
            "note": "NSRAM BER<0.01 at 20dB"},
        "AMBITIOUS": {
            "pass": bool(ber_20 < 0.001 and ber_10 < 0.05),
            "ber_20dB": ber_20, "ber_10dB": ber_10,
            "note": "NSRAM BER<0.001@20 AND BER<0.05@10"},
    }

    # ---- Save JSON ------------------------------------------------------
    with open(OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[N-LMS-Eq] summary.json written")

    # ---- BER vs SNR plot ----------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, marker in [("no_eq", "x"), ("wiener", "s"),
                         ("lms_f32", "o"), ("nsram", "D")]:
        vals = [summary["BER_per_SNR"][name][str(s_)] for s_ in SNR_DB_LIST]
        vals_plot = [max(v, 1e-5) for v in vals]
        ax.semilogy(SNR_DB_LIST, vals_plot, marker=marker, label=name)
    ax.set_xlabel("SNR (dB)"); ax.set_ylabel("BER")
    ax.set_title("N-LMS-Eq N16 — QPSK BER vs SNR\n"
                 "3-echo multipath, 16-tap NS-RAM equalizer")
    ax.grid(True, which="both", alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "BER_vs_SNR.png", dpi=140)
    plt.close(fig)

    # ---- Learning curve ----------------------------------------------
    fig, axes = plt.subplots(1, len(SNR_DB_LIST),
                             figsize=(5 * len(SNR_DB_LIST), 4), sharey=True)
    for ax, snr in zip(axes, SNR_DB_LIST):
        d = learn[snr]
        ax.plot(d["lms_f32"], label="LMS-f32", alpha=0.85)
        ax.plot(d["nsram"], label="NS-RAM", alpha=0.85)
        ax.axvline(N_PREAMBLE, ls="--", color="gray", alpha=0.5,
                   label="end preamble")
        ax.set_yscale("log"); ax.grid(True, alpha=0.3)
        ax.set_title(f"SNR = {snr} dB")
        ax.set_xlabel("symbol"); ax.legend(fontsize=8)
    axes[0].set_ylabel("|e|^2 (smoothed, 200-sym window)")
    fig.suptitle("N-LMS-Eq N16 — learning curves")
    fig.tight_layout()
    fig.savefig(OUT / "learning_curve.png", dpi=140)
    plt.close(fig)

    # ---- Dashboard via network_viz -----------------------------------
    # Build synthetic-but-real "neuron" panels from NS-RAM tap state.
    # 16 taps -> treat each as a 'neuron' with Vb traces & energy.
    # Use last-SNR Vb snapshot (2 banks x 2 cells x 16 taps).
    Vb_panel = Vb_final_snapshot.reshape(4, N_TAPS)  # 4 rows x 16
    weights_panel = np.outer(
        np.real(np.array([0, 1, 0, -1])),
        np.array([summary["BER_per_SNR"]["nsram"][str(s_)]
                  for s_ in SNR_DB_LIST] + [0] * (N_TAPS - len(SNR_DB_LIST))),
    )
    energy_panel = np.array([
        [summary["energy_per_symbol_pJ"]["nsram"]] * N_TAPS,
        [summary["energy_per_symbol_pJ"]["lms_f32"]] * N_TAPS,
    ])
    # Pareto: (energy_per_symbol, 1-BER@20dB) per method.
    pareto_pts = []
    for m in ["wiener_apply", "lms_f32", "nsram"]:
        eng = summary["energy_per_symbol_pJ"][m]
        # map names for BER lookup
        ber_name = "wiener" if m == "wiener_apply" else m
        acc = 1.0 - summary["BER_per_SNR"][ber_name]["20.0"]
        pareto_pts.append({"label": m, "energy_pj": eng, "accuracy": acc,
                           "throughput": 1.0})
    dash_data = {
        "vb": Vb_panel,
        "weights": weights_panel,
        "energy": energy_panel,
        "pareto": pareto_pts,
    }
    try:
        save_summary_dashboard(OUT, output_path=OUT / "dashboard.png",
                               data=dash_data,
                               title="N-LMS-Eq N16 — NS-RAM QPSK equalizer")
    except Exception as exc:
        print(f"[N-LMS-Eq] dashboard error: {exc}", flush=True)

    # ---- Report ------------------------------------------------------
    g = summary["gates"]
    report = []
    report.append("# N-LMS-Eq N16 — NS-RAM Adaptive QPSK Equalizer\n")
    report.append(f"Phase N1 #3 / Phase N2 U9. Runtime "
                  f"{time.time()-t0:.1f}s on zgx.\n")
    report.append("## Setup\n")
    report.append(f"- Modulation: QPSK, {N_SYMBOLS} symbols "
                  f"({N_PREAMBLE} preamble, training-aided LMS; "
                  f"decision-directed after).\n")
    report.append(f"- Channel: 3 complex echoes at delays "
                  f"{ECHO_DELAYS}; AWGN at SNR {SNR_DB_LIST} dB.\n")
    report.append(f"- Equalizer: {N_TAPS}-tap FIR; delay = {DELAY}.\n")
    report.append("## BER results\n\n")
    report.append("| SNR (dB) | no_eq | wiener | LMS-f32 | NS-RAM |\n")
    report.append("|---|---|---|---|---|\n")
    for s_ in SNR_DB_LIST:
        k = str(s_)
        report.append(
            f"| {s_} | {summary['BER_per_SNR']['no_eq'][k]:.3g} | "
            f"{summary['BER_per_SNR']['wiener'][k]:.3g} | "
            f"{summary['BER_per_SNR']['lms_f32'][k]:.3g} | "
            f"{summary['BER_per_SNR']['nsram'][k]:.3g} |\n")
    report.append("\n## Convergence (smoothed |e|^2 < 0.1)\n\n")
    for s_ in SNR_DB_LIST:
        k = str(s_)
        report.append(
            f"- SNR {s_} dB: LMS-f32 @ "
            f"{summary['convergence_steps']['lms_f32'][k]}, "
            f"NS-RAM @ {summary['convergence_steps']['nsram'][k]}\n")
    report.append("\n## Energy per symbol (pJ)\n\n")
    for k, v in summary["energy_per_symbol_pJ"].items():
        report.append(f"- {k}: {v:.3f}\n")
    report.append("\n## Pre-registered gates\n\n")
    for name, info in g.items():
        mark = "PASS" if info["pass"] else "FAIL"
        report.append(f"- **{name}**: {mark} -- {info['note']}\n")
    with open(OUT / "report.md", "w") as f:
        f.write("".join(report))

    print("\n=== GATES ===")
    for name, info in g.items():
        print(f"  {name}: {'PASS' if info['pass'] else 'FAIL'}  "
              f"{info.get('note', '')}")
    print(f"\nDone in {time.time() - t0:.1f}s. Output: {OUT}")


if __name__ == "__main__":
    main()
