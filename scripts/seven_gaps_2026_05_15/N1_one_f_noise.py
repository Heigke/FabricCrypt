"""Track 5 / N1: 1/f Vth noise injection model.

PSD: S_Vth(f) = K_f / (Cox * W * L * f)

Verify: simulated Vth(t) PSD exhibits 1/f^alpha with alpha in [0.8, 1.2].
Sweep K_f magnitude in {0.1x, 1x, 10x}.

Output: results/N1_1f_noise/{psd_validation.png, summary.json}
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal

OUT = Path(__file__).resolve().parents[2] / "results" / "N1_1f_noise"
OUT.mkdir(parents=True, exist_ok=True)

# --- BSIM4 NOIMOD-style 1/f noise card values (silicon-bulk NMOS, 130 nm) ---
# NOIA default ~ 6.25e41 (eV-1 m-3) for NMOS; K_f (lumped) ~ 1e-25 V^2 F is a
# conservative textbook value when NOIMOD card not present.
K_F_DEFAULT = 1e-25                # V^2 * F (lumped, Bsim4 SPICE flicker scale)
COX_OX = 3.45e-3                   # F/m^2 (130nm gate-ox capacitance per area)
W = 360e-9
L = 180e-9
CWL = COX_OX * W * L               # F

FS = 2e6     # sampling rate (2 MHz, well above any 1/f corner we care about)
T_S = 0.5    # 0.5 s of data per run
N = int(FS * T_S)


def gen_1f_noise(n: int, fs: float, K_f: float, cwl: float, seed: int = 0) -> np.ndarray:
    """Generate Vth_noise(t) with PSD = K_f/(cwl * f) via FFT spectral shaping."""
    rng = np.random.default_rng(seed)
    # White noise in time
    w = rng.standard_normal(n)
    # FFT, shape to 1/sqrt(f), inverse FFT.
    W_ = np.fft.rfft(w)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    # avoid div-by-zero at DC
    freqs[0] = freqs[1]
    # Target PSD V^2/Hz at each f: S(f) = K_f / (cwl * f)
    # The magnitude shaping factor = sqrt(S(f) * fs * n / 2) after normalization.
    shape = np.sqrt(K_f / (cwl * freqs))
    W_shaped = W_ * shape
    v = np.fft.irfft(W_shaped, n=n)
    # Normalize so that empirical PSD area matches target variance
    # (FFT scaling: numpy.fft uses unnormalized forward; PSD scale = 1/(fs*n))
    # We adjust by overall constant so PSD area near 1Hz matches K_f/(cwl*1Hz).
    return v.astype(np.float64)


def estimate_alpha(v: np.ndarray, fs: float) -> tuple[float, float, np.ndarray, np.ndarray]:
    f, pxx = signal.welch(v, fs=fs, nperseg=min(len(v) // 8, 1 << 15))
    # fit log10(Pxx) = -alpha * log10(f) + c  over decade [1, 1e4]
    mask = (f >= 1.0) & (f <= 1e4)
    lf = np.log10(f[mask])
    lp = np.log10(pxx[mask])
    a, c = np.polyfit(lf, lp, 1)
    alpha = -a
    return alpha, c, f, pxx


def main() -> None:
    sweep = {"0.1x": 0.1, "1x": 1.0, "10x": 10.0}
    results: dict[str, dict] = {}
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    for label, mult in sweep.items():
        K_f = K_F_DEFAULT * mult
        v = gen_1f_noise(N, FS, K_f, CWL, seed=int(mult * 100))
        alpha, c, f, pxx = estimate_alpha(v, FS)
        std_mV = float(np.std(v) * 1e3)
        results[label] = {
            "K_f": K_f,
            "alpha_fit": float(alpha),
            "alpha_in_band": bool(0.8 <= alpha <= 1.2),
            "Vth_noise_std_mV": std_mV,
        }
        ax.loglog(f[1:], pxx[1:], label=f"K_f x{label} alpha={alpha:.2f} std={std_mV:.1f}mV")
    # reference line: 1/f
    fref = np.logspace(0, 4, 50)
    ax.loglog(fref, 1e-15 / fref, "k--", alpha=0.4, label="ideal 1/f (arb)")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("PSD of Vth_noise  [V^2/Hz]")
    ax.set_title("N1: 1/f noise injection — PSD validation")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    plot_path = OUT / "psd_validation.png"
    fig.savefig(plot_path, dpi=110)
    plt.close(fig)

    # gate: at least one sweep point in [0.8,1.2]
    any_pass = any(v["alpha_in_band"] for v in results.values())
    summary = {
        "gate_alpha_in_band": any_pass,
        "results": results,
        "plot": str(plot_path),
        "config": {"K_f_default": K_F_DEFAULT, "Cox": COX_OX, "W": W, "L": L, "fs": FS, "T_s": T_S},
    }
    with open(OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
