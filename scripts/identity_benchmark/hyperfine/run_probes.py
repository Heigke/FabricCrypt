#!/usr/bin/env python3
"""Hyperfine precision-spectroscopy probes P1-P11.

Run with:  HSA_OVERRIDE_GFX_VERSION=11.0.0 venv/bin/python scripts/identity_benchmark/hyperfine/run_probes.py [--host LABEL] [--probes 1,2,...,11] [--quick]

All sampling is passive (read hwmon/thermal/freq sysfs). Active perturbations
are CPU-only (one core spinning) to avoid contaminating GPU clock signal,
EXCEPT P1 (DPM toggle) which writes to sysfs DPM file once per period.

Outputs JSON per probe + a raw npz with full time-series for later joint analysis.
"""
import argparse, os, sys, time, json, math, struct, threading, multiprocessing as mp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _hf_common import (
    HOST, HW_AMDGPU, P_FILE, T_FILE, F_FILE, DPM, CARD,
    apu_temp_c, gpu_temp_c, power_w, gpu_freq_hz,
    abort_if_hot, wait_cool, sample_block, save_json, cohen_d, allan_dev,
)


# ---------------- CPU perturbation helpers (no GPU) ----------------
def _cpu_burner(stop_evt, duty_fn=None, ncores=1):
    """Spin ncores with optional time-varying duty cycle in [0,1]."""
    import os as _os
    procs = []
    def _spin(idx):
        # pin to a specific core for reproducibility
        try: _os.sched_setaffinity(0, {idx})
        except Exception: pass
        t0 = time.monotonic()
        while not stop_evt.is_set():
            if duty_fn is None:
                # full burn
                x = 0.0
                for _ in range(200000): x = x*1.0000001 + 1e-9
            else:
                t = time.monotonic() - t0
                d = max(0.0, min(1.0, float(duty_fn(t))))
                on = 0.020 * d   # 20ms window scaling
                off = 0.020 * (1-d)
                t_end = time.monotonic() + on
                x = 0.0
                while time.monotonic() < t_end:
                    for _ in range(5000): x = x*1.0000001 + 1e-9
                if off > 0: time.sleep(off)
    threads = [threading.Thread(target=_spin, args=(i,), daemon=True) for i in range(ncores)]
    for t in threads: t.start()
    return threads


# ====================== PROBES ======================

def P1_lockin(out_dir, f0=0.5, duration_s=180, fs=20.0):
    """Lock-in: toggle DPM low/auto at f0 Hz; demodulate (T,P,freq) at f0."""
    print(f"[P1] lock-in f0={f0}Hz duration={duration_s}s on {HOST}", flush=True)
    if DPM is None:
        return {"probe":"P1","status":"skipped_no_dpm"}
    # read initial DPM
    try: orig = open(DPM).read().strip()
    except Exception: orig = "auto"
    period = 1.0/f0
    n = int(duration_s*fs)
    rec = {"t":np.zeros(n), "p":np.zeros(n), "tg":np.zeros(n), "ta":np.zeros(n), "f":np.zeros(n), "drive":np.zeros(n)}
    t0 = time.monotonic()
    last_state = None
    next_s = t0
    dt = 1.0/fs
    for i in range(n):
        if abort_if_hot(72.0):
            try: open(DPM,"w").write(orig+"\n")
            except Exception: pass
            return {"probe":"P1","status":"aborted_thermal","i":i}
        tnow = time.monotonic() - t0
        # square-wave drive at f0
        drv = 1 if (tnow % period) < (period/2) else 0
        if drv != last_state:
            try: open(DPM,"w").write(("low" if drv else "auto")+"\n")
            except Exception: pass
            last_state = drv
        rec["t"][i] = tnow
        rec["p"][i]  = power_w()
        rec["tg"][i] = gpu_temp_c()
        rec["ta"][i] = apu_temp_c()
        rec["f"][i]  = gpu_freq_hz()/1e6
        rec["drive"][i] = drv
        next_s += dt
        rem = next_s - time.monotonic()
        if rem > 1e-3: time.sleep(rem)
    # restore
    try: open(DPM,"w").write(orig+"\n")
    except Exception: pass

    # demodulate
    def demod(sig):
        s = sig - np.mean(sig)
        ph = 2*np.pi*f0*rec["t"]
        I = np.mean(s*np.cos(ph)); Q = np.mean(s*np.sin(ph))
        amp = 2*math.hypot(I,Q)
        phase = math.atan2(Q,I)
        return amp, phase

    res = {"probe":"P1","host":HOST,"f0":f0,"duration_s":duration_s,"fs":fs}
    for k in ("p","tg","ta","f"):
        a, ph = demod(rec[k])
        res[f"{k}_amp"] = a; res[f"{k}_phase_rad"] = ph
    save_json(os.path.join(out_dir,"P1_lockin.json"), res)
    np.savez_compressed(os.path.join(out_dir,"P1_lockin.npz"), **rec)
    return res


def P2_phase_coherent(out_dir, duration_s=60, fs=200.0, fold_ms=10.0):
    """Fold-average power signal modulo fold_ms; per-bin amplitude."""
    print(f"[P2] fold-avg fold={fold_ms}ms dur={duration_s}s on {HOST}", flush=True)
    blk = sample_block(duration_s, fs, want=("p","f"))
    if abort_if_hot(72.0): return {"probe":"P2","status":"aborted"}
    p = blk["p"]; f = blk["f"]; ts = blk["ts"]
    rel = (ts - ts[0]) * 1000.0  # ms
    nbins = int(round(fold_ms * fs / 1000.0))
    if nbins < 2: nbins = 2
    bin_idx = (np.arange(len(p)) % nbins)
    fold_p = np.array([p[bin_idx==i].mean() for i in range(nbins)])
    fold_f = np.array([f[bin_idx==i].mean() for i in range(nbins)])
    # significance: pattern energy vs shuffled-baseline
    rng = np.random.default_rng(42)
    null_amp = []
    for _ in range(200):
        idx2 = rng.permutation(bin_idx)
        fp2 = np.array([p[idx2==i].mean() for i in range(nbins)])
        null_amp.append(float(np.std(fp2)))
    res = {"probe":"P2","host":HOST,"duration_s":duration_s,"fs":fs,"fold_ms":fold_ms,
           "nbins":nbins,
           "fold_p_std": float(np.std(fold_p)),
           "fold_f_std": float(np.std(fold_f)),
           "null_p_std_mean": float(np.mean(null_amp)),
           "null_p_std_p95":  float(np.percentile(null_amp,95)),
           "z_p": float((np.std(fold_p) - np.mean(null_amp))/(np.std(null_amp)+1e-12)),
           "fold_p": fold_p.tolist(), "fold_f": fold_f.tolist()}
    save_json(os.path.join(out_dir,"P2_fold.json"), res)
    np.savez_compressed(os.path.join(out_dir,"P2_fold.npz"), p=p, f=f, ts=ts)
    return res


def P3_pump_probe(out_dir, n_pumps=20, pump_ms=100, tau_max_s=8.0, fs=100.0):
    """Pump = brief 4-core CPU burst; probe (P,T,freq) Green's function G(tau)."""
    print(f"[P3] pump-probe N={n_pumps} pump={pump_ms}ms tau<={tau_max_s}s on {HOST}", flush=True)
    n_per = int(tau_max_s * fs)
    traces = {k: np.zeros((n_pumps, n_per), dtype=np.float32) for k in ("p","tg","ta","f")}
    for k in range(n_pumps):
        if abort_if_hot(70.0): break
        # ensure quiet
        wait_cool(58.0, 60)
        # pump
        stop = threading.Event()
        ths = _cpu_burner(stop, ncores=4)
        time.sleep(pump_ms/1000.0)
        stop.set()
        for th in ths: th.join(timeout=0.1)
        # probe
        blk = sample_block(tau_max_s, fs, want=("p","tg","ta","f"))
        for kk in ("p","tg","ta","f"):
            traces[kk][k,:len(blk[kk])] = blk[kk][:n_per]
        time.sleep(0.5)
    # average G(tau), subtract baseline (first 5 samples)
    G = {kk: traces[kk].mean(axis=0) for kk in traces}
    Gn = {kk: G[kk] - G[kk][:5].mean() for kk in G}
    # fit exponential to p: y = A*(1-exp(-t/tau))
    t_axis = np.arange(n_per)/fs
    def fit_tau(y):
        y = y.copy();
        if abs(y[-1]) < 1e-9: return float("nan"), float("nan")
        # rough: time to reach (1-1/e)*y_peak
        peak = np.max(np.abs(y))
        if peak < 1e-9: return float("nan"), peak
        target = (1-1/math.e)*peak * np.sign(np.argmax(np.abs(y)))
        # find first crossing
        for i,v in enumerate(np.abs(y)):
            if v >= (1-1/math.e)*peak: return float(t_axis[i]), float(peak)
        return float("nan"), float(peak)
    tau_p, peak_p = fit_tau(Gn["p"])
    tau_tg, peak_tg = fit_tau(Gn["tg"])
    res = {"probe":"P3","host":HOST,"n_pumps":k+1,"fs":fs,
           "tau_rise_p_s": tau_p, "peak_p_w": peak_p,
           "tau_rise_tg_s": tau_tg, "peak_tg_c": peak_tg,
           "G_p_full": Gn["p"].tolist(), "G_tg_full": Gn["tg"].tolist(),
           "t_axis": t_axis.tolist()}
    save_json(os.path.join(out_dir,"P3_pump.json"), res)
    np.savez_compressed(os.path.join(out_dir,"P3_pump.npz"), **traces, t=t_axis)
    return res


def P4_two_tone(out_dir, f1=2.0, f2=7.0, duration_s=120, fs=50.0):
    """Two-tone CPU drive; look for intermodulation in power spectrum."""
    print(f"[P4] two-tone f1={f1} f2={f2} dur={duration_s}s on {HOST}", flush=True)
    stop = threading.Event()
    def duty(t):
        return 0.25 + 0.25*math.sin(2*math.pi*f1*t) + 0.25*math.sin(2*math.pi*f2*t)
    ths = _cpu_burner(stop, duty_fn=duty, ncores=2)
    try:
        blk = sample_block(duration_s, fs, want=("p","f","ta"))
    finally:
        stop.set()
        for th in ths: th.join(timeout=0.5)
    p = blk["p"] - np.mean(blk["p"])
    # PSD
    from numpy.fft import rfft, rfftfreq
    P = np.abs(rfft(p * np.hanning(len(p))))**2
    freqs = rfftfreq(len(p), d=1.0/fs)
    def peak_at(freq, bw=0.2):
        m = (freqs>freq-bw)&(freqs<freq+bw)
        return float(P[m].max()) if m.any() else 0.0
    # noise floor: median between 8-15 Hz, away from features
    nf_mask = (freqs>10)&(freqs<20)
    nf = float(np.median(P[nf_mask])) if nf_mask.any() else 1e-9
    feats = {
        "f1": peak_at(f1), "f2": peak_at(f2),
        "2f1": peak_at(2*f1), "2f2": peak_at(2*f2),
        "f1+f2": peak_at(f1+f2), "f2-f1": peak_at(f2-f1),
        "2f1+f2": peak_at(2*f1+f2), "2f1-f2": peak_at(abs(2*f1-f2)),
    }
    snrs = {k: float(v/nf) for k,v in feats.items()}
    res = {"probe":"P4","host":HOST,"f1":f1,"f2":f2,"duration_s":duration_s,"fs":fs,
           "noise_floor":nf, "peaks":feats, "snr":snrs}
    save_json(os.path.join(out_dir,"P4_twotone.json"), res)
    np.savez_compressed(os.path.join(out_dir,"P4_twotone.npz"), p=blk["p"], freqs=freqs, psd=P)
    return res


def P5_step_response(out_dir, n_steps=15, fs=100.0, win_s=2.0):
    """Microsecond step response: burst one CPU core, sample fast (freq, power, temp)."""
    # We don't have HIP loop access — fall back to sysfs at max rate (~1ms/sample limited)
    print(f"[P5] step-resp N={n_steps} win={win_s}s on {HOST}", flush=True)
    traces = {k: np.zeros((n_steps, int(win_s*fs)), dtype=np.float32) for k in ("p","f","tg")}
    for k in range(n_steps):
        if abort_if_hot(70.0): break
        wait_cool(58.0, 60)
        # step ON
        stop = threading.Event()
        ths = _cpu_burner(stop, ncores=1)
        blk = sample_block(win_s, fs, want=("p","f","tg"))
        stop.set()
        for th in ths: th.join(timeout=0.2)
        for kk in ("p","f","tg"):
            traces[kk][k,:len(blk[kk])] = blk[kk][:int(win_s*fs)]
        time.sleep(0.5)
    # extract rise time and damping from average f response
    f_avg = traces["f"].mean(axis=0)
    p_avg = traces["p"].mean(axis=0)
    f0_b = f_avg[:5].mean(); f_peak = float(f_avg.max())
    # rise time: 10% to 90%
    span = f_peak - f0_b
    if abs(span) > 1e-6:
        thr10 = f0_b + 0.1*span; thr90 = f0_b + 0.9*span
        try:
            i10 = next(i for i,v in enumerate(f_avg) if v>=thr10)
            i90 = next(i for i,v in enumerate(f_avg) if v>=thr90)
            rise_s = (i90-i10)/fs
        except StopIteration:
            rise_s = float("nan")
    else:
        rise_s = float("nan")
    # ringback: FFT of detrended trace
    from numpy.fft import rfft, rfftfreq
    det = f_avg - np.linspace(f_avg[0], f_avg[-1], len(f_avg))
    F = np.abs(rfft(det))
    freqs = rfftfreq(len(det), d=1.0/fs)
    ring_idx = int(np.argmax(F[1:]))+1 if len(F)>1 else 0
    ring_f = float(freqs[ring_idx]) if ring_idx>0 else 0.0
    res = {"probe":"P5","host":HOST,"n_steps":k+1,"fs":fs,"win_s":win_s,
           "f0_baseline_mhz": float(f0_b), "f_peak_mhz": f_peak,
           "rise_time_s": rise_s, "ringback_hz": ring_f,
           "f_avg": f_avg.tolist(), "p_avg": p_avg.tolist()}
    save_json(os.path.join(out_dir,"P5_step.json"), res)
    np.savez_compressed(os.path.join(out_dir,"P5_step.npz"), **traces)
    return res


def P6_allan(out_dir, duration_s=900, fs=100.0):
    """Allan deviation of TSC vs CLOCK_MONOTONIC fractional frequency."""
    print(f"[P6] Allan dur={duration_s}s fs={fs} on {HOST}", flush=True)
    n = int(duration_s*fs); dt = 1.0/fs
    tsc = np.zeros(n); mono = np.zeros(n)
    t0_m = time.monotonic_ns(); t0_t = time.perf_counter_ns()
    next_t = time.monotonic()
    for i in range(n):
        next_t += dt
        mono[i] = (time.monotonic_ns() - t0_m)/1e9
        tsc[i]  = (time.perf_counter_ns() - t0_t)/1e9
        if i % 1000 == 0 and abort_if_hot(72.0): break
        rem = next_t - time.monotonic()
        if rem > 1e-3: time.sleep(rem)
    # fractional frequency
    y = np.diff(tsc)/np.diff(mono) - 1.0
    taus = [0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]
    if duration_s >= 600: taus.append(500.0)
    ad = allan_dev(y, dt, taus)
    res = {"probe":"P6","host":HOST,"duration_s":duration_s,"fs":fs,
           "taus":taus, "allan_dev": ad.tolist(),
           "y_mean": float(np.mean(y)), "y_std": float(np.std(y))}
    save_json(os.path.join(out_dir,"P6_allan.json"), res)
    np.savez_compressed(os.path.join(out_dir,"P6_allan.npz"), tsc=tsc, mono=mono)
    return res


def P7_mi_lag(out_dir, duration_s=120, fs=50.0):
    """Lagged mutual information P-T, P-F, F-T pairs."""
    print(f"[P7] MI-lag dur={duration_s}s fs={fs} on {HOST}", flush=True)
    blk = sample_block(duration_s, fs, want=("p","tg","ta","f"))
    chans = {"P":blk["p"], "Tg":blk["tg"], "Ta":blk["ta"], "F":blk["f"]}
    def mi(a, b, bins=16):
        a = np.asarray(a); b = np.asarray(b)
        if np.std(a)<1e-9 or np.std(b)<1e-9: return 0.0
        H, _, _ = np.histogram2d(a, b, bins=bins)
        Pxy = H/H.sum(); Px = Pxy.sum(1, keepdims=True); Py = Pxy.sum(0, keepdims=True)
        m = Pxy>0
        return float(np.sum(Pxy[m]*np.log(Pxy[m]/(Px*Py + 1e-15)[m])))
    lags_s = [0, 0.02, 0.1, 0.5, 2.0, 10.0]
    out = {}
    pairs = [("P","Ta"),("P","Tg"),("P","F"),("F","Tg"),("F","Ta"),("Tg","Ta")]
    for a,b in pairs:
        row = []
        for lag in lags_s:
            L = int(lag*fs)
            if L >= len(chans[a])//2: row.append(float("nan")); continue
            row.append(mi(chans[a][:len(chans[a])-L], chans[b][L:]))
        out[f"{a}->{b}"] = row
    res = {"probe":"P7","host":HOST,"duration_s":duration_s,"fs":fs,"lags_s":lags_s,"MI":out}
    save_json(os.path.join(out_dir,"P7_mi.json"), res)
    np.savez_compressed(os.path.join(out_dir,"P7_mi.npz"), **blk)
    return res


def P8_bispectrum(out_dir, duration_s=180, fs=50.0, nseg=8):
    """Bispectrum of power; report total bispectral asymmetry + top peaks."""
    print(f"[P8] bispectrum dur={duration_s}s fs={fs} on {HOST}", flush=True)
    blk = sample_block(duration_s, fs, want=("p",))
    x = blk["p"] - np.mean(blk["p"])
    if abort_if_hot(72.0): return {"probe":"P8","status":"aborted"}
    N = len(x); seglen = N // nseg
    # average bispectrum over segments (segmented direct method)
    from numpy.fft import rfft, rfftfreq
    F = seglen//2 + 1
    B = np.zeros((F, F), dtype=np.complex128)
    for s in range(nseg):
        seg = x[s*seglen:(s+1)*seglen] * np.hanning(seglen)
        X = rfft(seg)
        # bispectrum B(f1,f2) = X(f1)X(f2)X*(f1+f2)
        for i in range(F):
            jmax = min(F-1, F-1-i)
            if jmax<=0: continue
            j = np.arange(jmax+1)
            k = i + j
            B[i, :jmax+1] += X[i] * X[j] * np.conj(X[k])
    B /= nseg
    magB = np.abs(B)
    freqs = rfftfreq(seglen, d=1.0/fs)
    # bicoherence proxy: B / mean
    bicoh = magB / (magB.mean()+1e-12)
    # top 5 peaks (excluding DC)
    flat = bicoh.copy(); flat[0,:]=0; flat[:,0]=0
    flat_idx = np.dstack(np.unravel_index(np.argsort(flat.ravel())[::-1][:10], flat.shape))[0]
    peaks = [{"f1":float(freqs[i]),"f2":float(freqs[j]),"bicoh":float(flat[i,j])} for i,j in flat_idx]
    asym = float(np.mean(magB) / (np.std(x)**3 + 1e-12))
    res = {"probe":"P8","host":HOST,"duration_s":duration_s,"fs":fs,"nseg":nseg,
           "bispec_asymmetry": asym, "top_peaks": peaks,
           "magB_max": float(magB.max()), "magB_mean": float(magB.mean())}
    save_json(os.path.join(out_dir,"P8_bispec.json"), res)
    np.savez_compressed(os.path.join(out_dir,"P8_bispec.npz"), magB=magB, freqs=freqs)
    return res


def P9_local_only_marker(out_dir, duration_s=120, fs=20.0):
    """Local half of P9: simultaneous-window sampling. Pair with remote via timestamp."""
    print(f"[P9] sync window dur={duration_s}s fs={fs} on {HOST}", flush=True)
    # wait until next 10-second wall boundary so two hosts roughly align
    wallclock = time.time()
    delay = 10.0 - (wallclock % 10.0)
    print(f"[P9] aligning, wait {delay:.2f}s", flush=True)
    time.sleep(delay)
    wstart = time.time()
    blk = sample_block(duration_s, fs, want=("p","tg","ta","f"))
    rec = {"wall_start":wstart, "p":blk["p"], "tg":blk["tg"], "ta":blk["ta"], "f":blk["f"], "ts":blk["ts"]}
    np.savez_compressed(os.path.join(out_dir,"P9_sync.npz"), **rec)
    res = {"probe":"P9","host":HOST,"wall_start":wstart,"duration_s":duration_s,"fs":fs,
           "p_mean":float(np.mean(blk["p"])),"p_std":float(np.std(blk["p"])),
           "tg_mean":float(np.mean(blk["tg"])),"tg_std":float(np.std(blk["tg"])),
           "f_mean":float(np.mean(blk["f"])),"f_std":float(np.std(blk["f"]))}
    save_json(os.path.join(out_dir,"P9_sync.json"), res)
    return res


def P10_counting(out_dir, duration_s=60, fs=200.0):
    """RTN-event counting on power signal. Fano factor."""
    print(f"[P10] counting dur={duration_s}s fs={fs} on {HOST}", flush=True)
    blk = sample_block(duration_s, fs, want=("p","f"))
    p = blk["p"]
    # define RTN event as derivative exceeding 2-sigma of derivative
    dp = np.diff(p)
    thresh = 2*np.std(dp)
    events = np.abs(dp) > thresh
    # bin into 100ms windows
    bin_ms = 100
    samples_per_bin = int(bin_ms * fs / 1000.0)
    if samples_per_bin < 2: samples_per_bin = 2
    nbins = len(events)//samples_per_bin
    counts = events[:nbins*samples_per_bin].reshape(nbins, samples_per_bin).sum(axis=1)
    mean_n = float(counts.mean()); var_n = float(counts.var())
    fano = var_n / (mean_n+1e-12)
    # also for freq
    df = np.diff(blk["f"])
    f_thresh = 2*np.std(df); f_events = np.abs(df) > f_thresh
    f_counts = f_events[:nbins*samples_per_bin].reshape(nbins, samples_per_bin).sum(axis=1)
    fano_f = float(f_counts.var() / (f_counts.mean()+1e-12))
    res = {"probe":"P10","host":HOST,"duration_s":duration_s,"fs":fs,
           "p_event_rate_hz": float(events.sum()/duration_s),
           "p_thresh": float(thresh), "fano_power": fano,
           "f_thresh": float(f_thresh), "fano_freq": fano_f,
           "mean_n_per_bin": mean_n, "var_n_per_bin": var_n,
           "bin_ms": bin_ms}
    save_json(os.path.join(out_dir,"P10_count.json"), res)
    np.savez_compressed(os.path.join(out_dir,"P10_count.npz"), p=p, f=blk["f"])
    return res


def P11_smu_calib(out_dir):
    """Try to read SMU/VFT fuses via available safe paths (READ-ONLY)."""
    print(f"[P11] SMU calibration read (read-only) on {HOST}", flush=True)
    candidates = [
        "/sys/kernel/ryzen_smu_drv/version",
        "/sys/kernel/ryzen_smu_drv/codename",
        "/sys/kernel/ryzen_smu_drv/smu_args",
        "/sys/kernel/ryzen_smu_drv/pm_table",
        "/sys/kernel/ryzen_smu_drv/pm_table_version",
        # GPU side — VBIOS / pp_table / pp_dpm_sclk (firmware-derived) — both cards
        "/sys/class/drm/card0/device/pp_dpm_sclk",
        "/sys/class/drm/card0/device/pp_dpm_mclk",
        "/sys/class/drm/card0/device/pp_dpm_socclk",
        "/sys/class/drm/card0/device/pp_dpm_fclk",
        "/sys/class/drm/card0/device/pp_table",
        "/sys/class/drm/card0/device/vbios_version",
        "/sys/class/drm/card0/device/unique_id",
        "/sys/class/drm/card1/device/pp_dpm_sclk",
        "/sys/class/drm/card1/device/pp_dpm_mclk",
        "/sys/class/drm/card1/device/pp_dpm_socclk",
        "/sys/class/drm/card1/device/pp_dpm_fclk",
        "/sys/class/drm/card1/device/pp_table",
        "/sys/class/drm/card1/device/vbios_version",
        "/sys/class/drm/card1/device/unique_id",
        # CPU
        "/sys/devices/system/cpu/cpu0/cpufreq/amd_pstate_max_freq",
        "/sys/devices/system/cpu/cpu0/cpufreq/amd_pstate_highest_perf",
        "/sys/devices/system/cpu/cpu0/cpufreq/amd_pstate_lowest_nonlinear_freq",
        "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq",
        "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_min_freq",
        # CPUID via /proc
        "/proc/cpuinfo",
    ]
    results = {}
    for path in candidates:
        try:
            if path == "/proc/cpuinfo":
                with open(path) as f:
                    txt = f.read()
                # Per-core max freq + model + stepping + microcode
                lines = txt.splitlines()
                vals = {}
                for ln in lines[:60]:
                    for k in ("model name","stepping","microcode","cpu MHz","cpu family"):
                        if ln.startswith(k):
                            vals[k] = ln.split(":",1)[1].strip(); break
                results[path] = vals
            elif path.endswith("pp_table") or path.endswith("pm_table"):
                # binary blob — hash + first 64 bytes hex
                with open(path,"rb") as f: blob = f.read()
                import hashlib
                results[path] = {"sha256": hashlib.sha256(blob).hexdigest(),
                                 "size": len(blob),
                                 "head_hex": blob[:64].hex()}
            else:
                with open(path) as f:
                    results[path] = f.read().strip()
        except Exception as e:
            results[path] = f"ERR: {e}"
    # Try multiple per-core max freq (per-die fuse evidence)
    percore = {}
    for c in range(32):
        for key in ("cpuinfo_max_freq","amd_pstate_highest_perf","amd_pstate_max_freq"):
            p = f"/sys/devices/system/cpu/cpu{c}/cpufreq/{key}"
            try: percore.setdefault(key, {})[c] = int(open(p).read().strip())
            except Exception: pass
    results["percore_max"] = percore
    # cpuid via x86 cpuid binary if available
    import subprocess
    try:
        out = subprocess.run(["cpuid","-1","-l","0x80000007"], capture_output=True, text=True, timeout=5)
        results["cpuid_80000007"] = out.stdout
    except Exception as e:
        results["cpuid_80000007"] = f"ERR: {e}"
    save_json(os.path.join(out_dir,"P11_smu_calib.json"), results)
    return {"probe":"P11","host":HOST,"keys_present":[k for k,v in results.items() if isinstance(v,(dict,str)) and not str(v).startswith("ERR")]}


# ====================== MAIN ======================

PROBE_FUNCS = {
    1: P1_lockin, 2: P2_phase_coherent, 3: P3_pump_probe, 4: P4_two_tone,
    5: P5_step_response, 6: P6_allan, 7: P7_mi_lag, 8: P8_bispectrum,
    9: P9_local_only_marker, 10: P10_counting, 11: P11_smu_calib,
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=HOST)
    ap.add_argument("--probes", default="1,2,3,4,5,6,7,8,9,10,11")
    ap.add_argument("--quick", action="store_true", help="reduced durations for smoke tests")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    out_dir = args.outdir or f"results/IDENTITY_BENCHMARK_2026-05-30/hyperfine/{args.host}"
    os.makedirs(out_dir, exist_ok=True)

    probes = [int(p) for p in args.probes.split(",") if p.strip()]
    overrides = {}
    if args.quick:
        overrides = {
            1: dict(duration_s=30, f0=0.5, fs=20.0),
            2: dict(duration_s=20, fs=200.0, fold_ms=10.0),
            3: dict(n_pumps=5, pump_ms=100, tau_max_s=4.0, fs=100.0),
            4: dict(duration_s=40, f1=2.0, f2=7.0, fs=50.0),
            5: dict(n_steps=4, fs=100.0, win_s=2.0),
            6: dict(duration_s=120, fs=100.0),
            7: dict(duration_s=40, fs=50.0),
            8: dict(duration_s=40, fs=50.0, nseg=4),
            9: dict(duration_s=30, fs=20.0),
            10: dict(duration_s=30, fs=200.0),
            11: dict(),
        }

    summary = {"host":args.host, "wall_start":time.time(), "probes":{}, "thermal_log":[]}
    for pid in probes:
        if abort_if_hot(72.0):
            print(f"[MAIN] hot, waiting...", flush=True)
            wait_cool(60.0, 180)
        wait_cool(62.0, 60)
        t0 = time.time()
        try:
            kwargs = overrides.get(pid, {}) if args.quick else {}
            res = PROBE_FUNCS[pid](out_dir, **kwargs) if pid != 11 else PROBE_FUNCS[pid](out_dir)
            summary["probes"][f"P{pid}"] = {"ok":True, "dt_s": time.time()-t0, "result_keys": list(res.keys()) if isinstance(res,dict) else None,
                                            "result": res if isinstance(res, dict) and len(json.dumps(res, default=str))<5000 else None}
        except Exception as e:
            import traceback
            summary["probes"][f"P{pid}"] = {"ok":False, "dt_s": time.time()-t0, "error": str(e), "tb": traceback.format_exc()}
            print(f"[P{pid}] ERROR: {e}", flush=True)
        summary["thermal_log"].append({"probe":pid, "apu_after_c": apu_temp_c(), "t": time.time()})
        # cooldown between probes
        wait_cool(60.0, 90)
    summary["wall_end"] = time.time()
    save_json(os.path.join(out_dir, "RUN_SUMMARY.json"), summary)
    print(f"[DONE] {args.host} wrote summary; total wall = {summary['wall_end']-summary['wall_start']:.1f}s", flush=True)

if __name__ == "__main__":
    main()
