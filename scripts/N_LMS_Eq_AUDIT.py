"""N_LMS_Eq_AUDIT — peripheral-aware brutal audit of the 170x NS-RAM LMS-Eq claim.

Re-uses the N_LMS_Eq_N16 simulation (calibrated-cell LUT IiiNetLUT loads
post-z469 snap_Is etc.) for the actual BER measurement, then layers an HONEST
peripheral energy model on top:

  Analog NS-RAM path per QPSK symbol (N=16 taps, complex => 64 cells):
    1. Analog MAC current sum         ~ 5  fJ / cell
    2. Vd drive (column line)         ~ 50 fJ / cell
    3. ADC at output     (8b, 1 MS/s) ~ Razavi/Murmann survey
    4. DAC tap write    (8b per tap update) ~ Razavi
    5. Row driver wire RC at 130 nm
    6. Control overhead
    7. Body-state retention constraint: tau_body ~ 1 ms => max symbol rate

  Digital baseline ISO-PRECISION (8-bit MAC, NOT f32):
    - Horowitz ISSCC-2014: 8b MAC at 45 nm ~ 0.2 pJ
    - 16 complex taps = 64 real MACs/symbol fwd + 64 update
    - Plus 8b reg file access, control

Sources cited inline.

Pre-registered gates (this script):
    SURVIVES: peripheral-aware NS-RAM <= 10x digital baseline AND
              BER(NSRAM) <= 1.5x BER(digital_8b) at SNR >= 10 dB
    DEMOTE:   peripheral-aware NS-RAM > digital baseline
    KILL:     peripheral > 100x cell cost (peripheral dominates)
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "N_LMS_Eq_AUDIT"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(REPO / "scripts"))

# Reuse the calibrated simulator (uses post-z469 snap_Is=4.5192e-12 LUT)
from N_LMS_Eq_N16 import (  # noqa: E402
    generate_qpsk, channel_pass, qpsk_decision, qpsk_ber, wiener_complex,
    fir_apply_complex, lms_complex, nsram_equalizer_complex,
    N_TAPS, N_SYMBOLS, N_PREAMBLE, DELAY, CHANNEL_H,
)
from S2b_transient import IiiNetLUT  # noqa: E402

SNR_DB_LIST = [5.0, 10.0, 15.0, 20.0]

# ----------------------------------------------------------------------------
# Quantization: 8-bit MAC digital baseline (iso-precision with NS-RAM ~6-8b)
# ----------------------------------------------------------------------------
def quantize_int8(x: np.ndarray, scale: float) -> np.ndarray:
    """Symmetric 8-bit: -127..127 * scale."""
    q = np.round(x / scale).clip(-127, 127)
    return q * scale


def lms_int8_complex(y, s, n_taps=N_TAPS, mu=0.01, delay=DELAY,
                     preamble=N_PREAMBLE,
                     w_scale=2.0 / 127, x_scale=3.0 / 127):
    """LMS with 8-bit weights and 8-bit input quantization at MAC,
    accumulator in int32 (industry standard CIM digital baseline)."""
    w = np.zeros(n_taps, dtype=np.complex128)
    N = len(y)
    out = np.zeros(N, dtype=np.complex128)
    err2 = np.zeros(N, dtype=np.float64)
    buf = np.zeros(n_taps, dtype=np.complex128)
    for n in range(N):
        buf[1:] = buf[:-1]; buf[0] = y[n]
        # Quantize input and weights to 8 bits at MAC
        bq_r = quantize_int8(buf.real, x_scale)
        bq_i = quantize_int8(buf.imag, x_scale)
        wq_r = quantize_int8(w.real, w_scale)
        wq_i = quantize_int8(w.imag, w_scale)
        # Complex MAC w/ quantized operands; accumulator effectively 32b
        yhat_r = np.dot(wq_r, bq_r) - np.dot(wq_i, bq_i)
        yhat_i = np.dot(wq_r, bq_i) + np.dot(wq_i, bq_r)
        yhat = yhat_r + 1j * yhat_i
        out[n] = yhat
        d_idx = n - delay
        if 0 <= d_idx < preamble:
            d = s[d_idx]
            e = d - yhat
            w = w + mu * np.conj(buf) * e
            err2[n] = float(np.abs(e) ** 2)
        elif d_idx >= preamble:
            d = qpsk_decision(np.array([yhat]))[0]
            e = d - yhat
            w = w + 0.3 * mu * np.conj(buf) * e
            err2[n] = float(np.abs(e) ** 2)
    return out, err2, w


# ----------------------------------------------------------------------------
# PERIPHERAL ENERGY MODEL (honest, with sources)
# ----------------------------------------------------------------------------
# All numbers in pJ unless stated.
#
# Cell-level (analog NS-RAM, 130 nm Sebas process):
#   MAC current: I*Vread*tau ~ 100 nA * 0.5 V * 100 ns = 5 fJ/cell  -> 0.005 pJ
#   Vd drive line CV^2: 200 fF * 0.5^2 = 50 fJ                      -> 0.05 pJ
#
# DAC (8-bit, 1 MS/s, per tap-update channel):
#   Murmann ADC/DAC survey (https://github.com/bmurmann/ADC-survey)
#   8b SAR DAC at 1 MS/s: FOM ~50 fJ/conv.step => 50 fJ * 2^8 = 12.8 pJ/conv
#   Conservative 130 nm CMOS scaling (~0.5 fJ/step degradation) ~ 15 pJ/sample
DAC_PJ_PER_WRITE = 15.0   # 8b, 1 MS/s, 130 nm scaling on Murmann/Razavi
#
# ADC (8-bit, 1 MS/s, output sampling):
#   Murmann survey median FOM for 8b/1MS/s ~ 50 fJ/conv.step => ~12.8 pJ/conv
#   Add buffer/S&H ~5 pJ  -> 18 pJ realistic
ADC_PJ_PER_SAMPLE = 18.0
#
# Wire RC at 130 nm (column line, ~1 mm avg):
#   C_metal ~ 200 fF/mm; Vdd^2 = 1.2^2 = 1.44; alpha=0.5
#   E_wire = 0.5 * 200e-15 * 1.44 = 0.144 pJ per toggle per column
WIRE_PJ_PER_COL_TOGGLE = 0.144
#
# Row driver / control overhead per symbol (Sun & Wong 2019, IEDM)
ROW_DRIVER_PJ_PER_SYM = 5.0  # row decoder + WL pulse generator
CTRL_PJ_PER_SYM = 3.0        # FSM/timing control
#
# Digital 8b MAC baseline:
#   Horowitz "Computing's energy problem (and what we can do about it)",
#   ISSCC-2014, Fig 2: 8b integer MAC at 45 nm = 0.2 pJ.
#   Scale to 130 nm (~4x voltage*capacitance penalty) = 0.8 pJ.
#   We'll use 0.2 pJ (45 nm) as a FAIR baseline, since the NS-RAM advantage
#   would be even worse at 130 nm.  This is GENEROUS to NS-RAM.
INT8_MAC_PJ_45NM = 0.2
INT8_REGFILE_PJ = 0.1   # 8b register read+write per access
INT8_CTRL_PJ = 0.5      # FSM, sequencer per symbol


def nsram_peripheral_energy_per_symbol(n_taps=N_TAPS):
    """Per-symbol energy with FULL peripheral overhead included.
    Complex => 2 banks (I, Q) of n_taps differential cells (A+B) each.
    Cells total = 4 * n_taps (I-A, I-B, Q-A, Q-B)
    """
    n_cells = 4 * n_taps
    # 1) analog MAC
    e_mac = n_cells * 0.005        # 0.005 pJ per cell
    # 2) Vd drive
    e_vd = n_cells * 0.05
    # 3) ADC for I and Q outputs (2 samples)
    e_adc = 2 * ADC_PJ_PER_SAMPLE
    # 4) DAC: write happens on every LMS update -> n_cells writes/symbol
    e_dac = n_cells * DAC_PJ_PER_WRITE
    # 5) Wire RC: each column toggled once per symbol
    e_wire = n_taps * 2 * WIRE_PJ_PER_COL_TOGGLE   # 2 banks
    # 6) Row + control
    e_row = ROW_DRIVER_PJ_PER_SYM
    e_ctrl = CTRL_PJ_PER_SYM
    breakdown = {
        "analog_mac": e_mac,
        "vd_drive": e_vd,
        "adc": e_adc,
        "dac_writes": e_dac,
        "wire_rc": e_wire,
        "row_driver": e_row,
        "control": e_ctrl,
    }
    total = sum(breakdown.values())
    cell_total = e_mac + e_vd
    periph_total = total - cell_total
    return total, cell_total, periph_total, breakdown


def digital_int8_energy_per_symbol(n_taps=N_TAPS):
    """8-bit complex LMS energy per symbol.  Complex MAC = 4 real MACs + 2 adds.
    Plus update path."""
    # Forward: 4 * n_taps real MACs
    fwd_macs = 4 * n_taps
    # Update: same count for gradient + weight update
    upd_macs = 4 * n_taps
    e_mac = (fwd_macs + upd_macs) * INT8_MAC_PJ_45NM
    # Register file access: each tap weight read+written each symbol
    e_rf = 4 * n_taps * INT8_REGFILE_PJ
    e_ctrl = INT8_CTRL_PJ
    breakdown = {"int8_mac": e_mac, "regfile": e_rf, "control": e_ctrl}
    return sum(breakdown.values()), breakdown


# ----------------------------------------------------------------------------
# Symbol-rate constraint check
# ----------------------------------------------------------------------------
TAU_BODY_S = 1e-3            # ~1 ms body-state retention timescale
# To program a +/-0.04 V Vb pulse and have it persist for the symbol
# duration, T_sym must be << tau_body.  At T_sym = 1 us (1 MS/s),
# T_sym / tau_body = 1e-3 => decay per symbol = 0.001 (matches script).
T_SYM_S = 1e-6
SYMBOL_RATE_HZ = 1.0 / T_SYM_S


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    t0 = time.time()
    print("[AUDIT] Loading calibrated NS-RAM LUT (post-z469)...", flush=True)
    lut = IiiNetLUT()
    print(f"[AUDIT] LUT loaded ({time.time()-t0:.2f}s)", flush=True)

    s = generate_qpsk(N_SYMBOLS, seed=42)
    eval_start = N_PREAMBLE + 1000

    results = {"snr_db": SNR_DB_LIST,
               "BER": {"no_eq": {}, "wiener": {}, "lms_f32": {},
                       "lms_int8": {}, "nsram": {}}}

    for snr in SNR_DB_LIST:
        print(f"\n[AUDIT] SNR = {snr} dB", flush=True)
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

        # LMS f32 (reference)
        out_f, err2_f, wf = lms_complex(y, s)
        ber_f = qpsk_ber(out_f, s, DELAY, eval_start)

        # LMS int8 (iso-precision digital baseline)
        out_q, err2_q, wq = lms_int8_complex(y, s)
        ber_q = qpsk_ber(out_q, s, DELAY, eval_start)

        # NS-RAM analog (post-z469 calibrated)
        out_n, err2_n, wn, _ = nsram_equalizer_complex(
            y, s, lut, seed=int(snr) + 11)
        ber_n = qpsk_ber(out_n, s, DELAY, eval_start)

        print(f"  no_eq:    BER={ber_ne:.4g}")
        print(f"  wiener:   BER={ber_w:.4g}")
        print(f"  LMS-f32:  BER={ber_f:.4g}")
        print(f"  LMS-int8: BER={ber_q:.4g}")
        print(f"  NS-RAM:   BER={ber_n:.4g}", flush=True)

        results["BER"]["no_eq"][str(snr)] = ber_ne
        results["BER"]["wiener"][str(snr)] = ber_w
        results["BER"]["lms_f32"][str(snr)] = ber_f
        results["BER"]["lms_int8"][str(snr)] = ber_q
        results["BER"]["nsram"][str(snr)] = ber_n

    # ---- Energy --------------------------------------------------------
    ns_total, ns_cell, ns_periph, ns_break = nsram_peripheral_energy_per_symbol()
    dig_total, dig_break = digital_int8_energy_per_symbol()
    f32_total_recorded = 474.6   # from original results, fp32 MAC at 45 nm

    energy = {
        "nsram_total_pJ": ns_total,
        "nsram_cell_pJ": ns_cell,
        "nsram_periph_pJ": ns_periph,
        "nsram_breakdown_pJ": ns_break,
        "digital_int8_total_pJ": dig_total,
        "digital_int8_breakdown_pJ": dig_break,
        "digital_f32_recorded_pJ": f32_total_recorded,
        "peripheral_dominance_ratio": ns_periph / max(ns_cell, 1e-9),
        "ratio_nsram_vs_int8": ns_total / dig_total,
        "ratio_nsram_vs_f32": ns_total / f32_total_recorded,
        "ratio_original_claim_vs_int8": f32_total_recorded / dig_total,
        "sources": {
            "ADC": "Murmann ADC-survey (Stanford); 8b 1 MS/s FOM ~50 fJ/step => ~13 pJ + buffer",
            "DAC": "Razavi 'Principles of Data Conversion'; 8b SAR DAC FOM ~50 fJ/step",
            "int8_MAC": "Horowitz, ISSCC-2014, Fig 2: 8b int MAC @ 45 nm = 0.2 pJ",
            "wire_RC": "ITRS 130 nm Cmetal ~200 fF/mm; CV^2 at Vdd=1.2 V",
            "row_driver": "Sun & Wong, IEDM 2019, in-memory compute periph survey",
        },
    }

    # ---- Gates ---------------------------------------------------------
    ber_10_nsram = results["BER"]["nsram"]["10.0"]
    ber_10_int8 = results["BER"]["lms_int8"]["10.0"]
    ber_20_nsram = results["BER"]["nsram"]["20.0"]
    ber_20_int8 = results["BER"]["lms_int8"]["20.0"]

    gate = {
        "SURVIVES": {
            "condition": "peripheral-aware NS-RAM <= 10x int8 AND BER(NSRAM)<=1.5x BER(int8) at SNR>=10dB",
            "energy_ratio": ns_total / dig_total,
            "energy_pass": ns_total <= 10 * dig_total,
            "ber_10dB_ratio": ber_10_nsram / max(ber_10_int8, 1e-6),
            "ber_20dB_ratio": ber_20_nsram / max(ber_20_int8, 1e-6),
            "ber_pass": (ber_10_nsram <= 1.5 * max(ber_10_int8, 1e-4)
                         and ber_20_nsram <= 1.5 * max(ber_20_int8, 1e-4)),
        },
        "DEMOTE": {
            "condition": "peripheral-aware NS-RAM > int8 baseline",
            "triggered": ns_total > dig_total,
        },
        "KILL": {
            "condition": "peripheral cost > 100x cell cost",
            "ratio": ns_periph / max(ns_cell, 1e-9),
            "triggered": ns_periph > 100 * ns_cell,
        },
    }
    gate["SURVIVES"]["pass"] = (gate["SURVIVES"]["energy_pass"]
                                and gate["SURVIVES"]["ber_pass"])

    if gate["KILL"]["triggered"]:
        verdict = "KILL"
    elif gate["SURVIVES"]["pass"]:
        verdict = "SURVIVES"
    else:
        verdict = "DEMOTE"

    energy["gates"] = gate
    energy["verdict"] = verdict

    # ---- Save artifacts ------------------------------------------------
    (OUT / "peripheral_energy.json").write_text(json.dumps(energy, indent=2))
    (OUT / "digital_baseline.json").write_text(json.dumps({
        "ber_per_snr": results["BER"]["lms_int8"],
        "energy_per_symbol_pJ": dig_total,
        "breakdown_pJ": dig_break,
        "precision_bits": 8,
        "source": "Horowitz ISSCC-2014 8b MAC at 45 nm",
    }, indent=2))
    (OUT / "ber_results.json").write_text(json.dumps(results, indent=2))

    # ---- BER plot ------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))
    style = [
        ("no_eq",     "x", "gray"),
        ("wiener",    "s", "tab:green"),
        ("lms_f32",   "o", "tab:blue"),
        ("lms_int8",  "^", "tab:purple"),
        ("nsram",     "D", "tab:red"),
    ]
    for name, marker, color in style:
        vals = [results["BER"][name][str(s_)] for s_ in SNR_DB_LIST]
        vals = [max(v, 1e-5) for v in vals]
        ax.semilogy(SNR_DB_LIST, vals, marker=marker, color=color,
                    label=name)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("BER")
    ax.set_title("N-LMS-Eq AUDIT — iso-precision BER vs SNR (QPSK, 3-echo, N16)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "ber_vs_snr.png", dpi=140)
    plt.close(fig)

    # ---- Verdict writeup ----------------------------------------------
    md = []
    md.append("# N-LMS-Eq AUDIT — Brutal Peripheral-Aware Verdict\n")
    md.append(f"**VERDICT: {verdict}**\n")
    md.append("## Energy per symbol (pJ)\n")
    md.append(f"- NS-RAM analog **TOTAL** with peripherals: **{ns_total:.2f} pJ**")
    md.append(f"  - Cell-level (MAC+Vd): {ns_cell:.3f} pJ ({100*ns_cell/ns_total:.2f}%)")
    md.append(f"  - Peripheral (ADC/DAC/wire/row/ctrl): {ns_periph:.2f} pJ ({100*ns_periph/ns_total:.2f}%)")
    md.append("\n  Breakdown:")
    for k, v in ns_break.items():
        md.append(f"    - {k}: {v:.3f} pJ ({100*v/ns_total:.2f}%)")
    md.append(f"\n- Digital int8 LMS (iso-precision baseline): **{dig_total:.2f} pJ**")
    for k, v in dig_break.items():
        md.append(f"    - {k}: {v:.3f} pJ")
    md.append(f"\n- Original f32 baseline (from N_LMS_Eq_N16): {f32_total_recorded:.2f} pJ\n")

    md.append("## Ratios\n")
    md.append(f"- NS-RAM / int8-digital = **{ns_total/dig_total:.2f}x**")
    md.append(f"- NS-RAM / f32-digital  = {ns_total/f32_total_recorded:.4f}x")
    md.append(f"- Original claim (f32/int8 baselines): {f32_total_recorded/dig_total:.1f}x (this is the apples-to-oranges artifact)")
    md.append(f"- Peripheral/cell dominance: {ns_periph/max(ns_cell,1e-9):.1f}x\n")

    md.append("## BER at iso-precision (NS-RAM ~6-8 effective bits, int8 digital)\n")
    md.append("| SNR (dB) | no_eq | wiener | LMS-f32 | LMS-int8 | NS-RAM |")
    md.append("|---|---|---|---|---|---|")
    for sn in SNR_DB_LIST:
        row = (f"| {sn} | {results['BER']['no_eq'][str(sn)]:.4g} | "
               f"{results['BER']['wiener'][str(sn)]:.4g} | "
               f"{results['BER']['lms_f32'][str(sn)]:.4g} | "
               f"{results['BER']['lms_int8'][str(sn)]:.4g} | "
               f"{results['BER']['nsram'][str(sn)]:.4g} |")
        md.append(row)

    md.append("\n## Pre-registered gate decision\n")
    md.append(f"- SURVIVES: energy_pass={gate['SURVIVES']['energy_pass']}, ber_pass={gate['SURVIVES']['ber_pass']} -> overall={gate['SURVIVES']['pass']}")
    md.append(f"- DEMOTE triggered: {gate['DEMOTE']['triggered']}")
    md.append(f"- KILL triggered:   {gate['KILL']['triggered']} (periph/cell ratio={gate['KILL']['ratio']:.1f})\n")

    md.append("## Symbol-rate constraint\n")
    md.append(f"- Body-state tau ~ {TAU_BODY_S*1e3:.1f} ms; T_sym = {T_SYM_S*1e6:.2f} us => decay/symbol={T_SYM_S/TAU_BODY_S:.3g}.")
    md.append("  Adequate at 1 MS/s; would dominate dynamics at >100 kS/s -> sub-MHz only.\n")

    md.append("## Sources\n")
    for k, v in energy["sources"].items():
        md.append(f"- **{k}**: {v}")
    md.append("\n## Honest interpretation\n")
    if verdict == "KILL":
        md.append("Peripheral (ADC/DAC/row drivers) consumes >>100x the cell energy.\n"
                  "The NS-RAM analog cell is irrelevant to total system energy at\n"
                  "this scale (N=16 taps, 1 MS/s).  The 170x advantage was an\n"
                  "artifact of comparing a peripheral-free analog model to a\n"
                  "FULL-precision f32 digital pipeline.  CLAIM DEMOTED.\n")
    elif verdict == "DEMOTE":
        md.append("Once peripheral costs are included, NS-RAM is NOT cheaper than\n"
                  "an iso-precision 8b digital implementation.  The 170x figure\n"
                  "was an apples-to-oranges comparison (fp32 vs sub-fJ analog).\n"
                  "REAL advantage at this scale: small or negative.\n")
    else:
        md.append("NS-RAM survives at iso-precision -- but the advantage is\n"
                  "modest, not 170x.\n")
    (OUT / "honest_verdict.md").write_text("\n".join(md))
    print("\n" + "\n".join(md[:60]))
    print(f"\n[AUDIT] verdict = {verdict}")
    print(f"[AUDIT] written to {OUT}")
    print(f"[AUDIT] total time {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
