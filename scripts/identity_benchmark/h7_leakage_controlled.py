"""H7 B3 (v2) — leakage Ioff(T) curvature with a CLOSED-LOOP temperature controller.

Why the first B3 failed: it sampled idle power during a free cooldown TRANSIENT. Each sample sat at an
unknown, fast-moving temperature and was contaminated by residual switching activity, so the a+b*exp(T/c)
fit was degenerate (b->0, c rail-pinned) and not reproducible.

Fix (this script): a bang-bang controller HOLDS the die at a sequence of setpoints {58,65,72,78}C by
modulating a multiprocess background load. At each held setpoint we take many "gated" idle measurements:
flip the load OFF, let active power drain for a few ms, then integrate RAPL package energy over a short
window (<=150ms) — short enough that T moves <~1C, long enough for a clean energy delta. The recorded
(T, P_idle) points therefore sit at KNOWN, STABLE temperatures with the cores parked → true static+leakage.

Subthreshold leakage Ioff ~ exp(T) with a slope set by this die's threshold-voltage (Vth) distribution
(process variation). Fit P_idle(T)=a+b*exp(T/c); (b,c) is the per-die leakage signature. Compare ikaros vs
daedalus for uniqueness. Two passes -> intra-die reproducibility.

Thermal-safe: setpoints <= 78C, hard abort > 86C, cores sleep (not spin) when idle. Read-only sysfs telemetry.
Run sandbox-disabled.  Out: results/IDENTITY_H7_2026-06-09/leakage_controlled_{host}.json
"""
from __future__ import annotations
import os, sys, json, time, socket, multiprocessing
import numpy as np
from pathlib import Path

HOST = socket.gethostname()
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/IDENTITY_H7_2026-06-09"; OUT.mkdir(parents=True, exist_ok=True)
RAPL = "/sys/class/powercap/intel-rapl:0/energy_uj"
TZ = "/sys/class/thermal/thermal_zone0/temp"

T_LO = 55.0                             # ramp floor (C)
T_HI = 74.0                             # ramp ceiling (C); kept well under abort
RAMP_RATE = 0.06                        # reference sweep speed (C/s) — slow so the die tracks it closely
SAMPLE_EVERY = 0.9                      # seconds between gated idle samples along the ramp
RAMP_TIMEOUT = 420.0                    # wall-clock cap per ramp direction (s)
ABORT = 84.0                            # hard thermal abort (C)
IDLE_WIN = 0.18                         # RAPL integration window per gated sample (s)
DRAIN = 0.05                            # let active power drain before integrating (s)
KP = 0.22                               # PI proportional gain (duty per degC error)
KI = 0.06                               # PI integral gain (tracks the moving reference with no lag offset)


def temp():
    try: return int(open(TZ).read()) / 1000
    except Exception: return 0.0


def energy_uj():
    try: return int(open(RAPL).read())
    except Exception: return 0


CYCLE = 0.04  # PWM period for the proportional heater (s)


def _worker(duty):
    """PWM CPU heater: each CYCLE, burn for duty fraction then idle the rest. duty<0 -> exit.
    Proportional duty (set by the controller from the temperature error) avoids the bang-bang overshoot
    that 32 always-on cores produce, so the die can be HELD at a setpoint instead of oscillating past it."""
    x = 1.000001
    while True:
        d = duty.value
        if d < 0: break
        if d <= 0.01:
            time.sleep(CYCLE)
        else:
            t0 = time.time(); burn = d * CYCLE
            while time.time() - t0 < burn:
                for _ in range(20000): x = x * 1.0000001 + 0.1
            rest = CYCLE - (time.time() - t0)
            if rest > 0: time.sleep(rest)
    if x == 1234.5: print(x)  # defeat dead-code elimination


class Controller:
    """Proportional (duty-cycle) temperature controller over a pool of CPU worker processes.

    heat duty = clip(KP*(setpoint - T), 0, 1): far below setpoint -> full heat; near it -> trickle, so the
    die settles AT the setpoint without the overshoot 32 always-on cores cause. The APU is one die, so CPU
    heating + RAPL package idle-power gating still measures this die's Ioff(T). gfx1151 GPU heaters overshoot
    to ~97C instantly and must NOT be used here.
    NOTE: never `pkill -f` this script's name from the launching shell — it matches the shell's own argv."""
    def __init__(self):
        n = max(2, (os.cpu_count() or 16) - 1)
        self.duty = multiprocessing.Value("d", 0.0)
        self.procs = [multiprocessing.Process(target=_worker, args=(self.duty,)) for _ in range(n)]
        for p in self.procs: p.start()

    def reset(self):
        self.integ = 0.0; self._last = None

    def hold(self, sp):
        """One PI control tick toward setpoint sp (non-blocking). duty = clip(KP*err + KI*∫err, 0, 1)
        with anti-windup (integral frozen when the output saturates). Returns current temp."""
        if not hasattr(self, "integ"): self.reset()
        t = temp(); now = time.time()
        dt = 0.0 if self._last is None else min(1.0, now - self._last)
        self._last = now
        if t >= ABORT:
            self.duty.value = 0.0; self.integ = 0.0
            return t
        err = sp - t
        raw = KP * err + KI * self.integ
        # anti-windup: only integrate when not saturated (or when integrating pulls out of saturation)
        if 0.0 < raw < 1.0 or (raw >= 1.0 and err < 0) or (raw <= 0.0 and err > 0):
            self.integ += err * dt
            self.integ = max(-5.0, min(40.0, self.integ))
        self.duty.value = float(min(1.0, max(0.0, KP * err + KI * self.integ)))
        return t

    def gated_idle_sample(self):
        """Park cores, let active power drain, integrate RAPL over IDLE_WIN. Returns (T_mid, P_idle_W)."""
        self.duty.value = 0.0
        time.sleep(DRAIN)
        T0 = temp(); e0 = energy_uj(); c0 = time.time()
        time.sleep(IDLE_WIN)
        de = energy_uj() - e0; dt = time.time() - c0; T1 = temp()
        if de < 0: de += (1 << 32)
        return (T0 + T1) / 2.0, (de / 1e6) / dt

    def coast_to(self, target, timeout):
        """Park cores and wait until the die cools below target (between passes)."""
        self.duty.value = 0.0; t0 = time.time()
        while temp() > target and time.time() - t0 < timeout:
            time.sleep(1.0)

    def stop(self):
        self.duty.value = -1.0
        for p in self.procs:
            p.join(timeout=2)
            if p.is_alive(): p.terminate()


def ramp_pass(ctl, lo, hi, up):
    """Sweep the reference temperature slowly across [lo,hi] (up=True rising, else falling) while the PI
    controller makes the die TRACK it; take a gated idle sample every SAMPLE_EVERY s. Because the reference
    moves slowly the die stays near it, so each (T,P) sits at a near-stable, known temperature — but we
    record the TRUE measured T regardless, so imperfect tracking still yields valid points. Dense coverage
    across the whole band → a well-conditioned a+b*exp(T/c) fit (no 2-cluster degeneracy)."""
    ctl.reset()
    r = lo if up else hi
    Ts, Ps = [], []
    t0 = time.time(); last = t0; last_samp = 0.0
    while ((r < hi) if up else (r > lo)) and time.time() - t0 < RAMP_TIMEOUT:
        now = time.time(); dt = min(0.5, now - last); last = now
        r += RAMP_RATE * dt if up else -RAMP_RATE * dt
        r = min(hi, max(lo, r))
        t = ctl.hold(r)
        if t >= ABORT:
            ctl.duty.value = 0.0; print(f"  ABORT {t:.0f}C", flush=True); break
        if now - last_samp >= SAMPLE_EVERY:
            Tm, Pw = ctl.gated_idle_sample(); last_samp = time.time()
            if 0.2 < Pw < 200:
                Ts.append(Tm); Ps.append(Pw)
        else:
            time.sleep(0.05)
    return np.array(Ts), np.array(Ps)


def bin_PT(T, P, edges):
    """Bin (T,P) into temperature bins → per-bin (T_mean, P_mean, n). The binned P(T) curve is the robust,
    comparable per-die signature (more stable than a 3-param fit over a ~20C span)."""
    out = []
    for i in range(len(edges) - 1):
        m = (T >= edges[i]) & (T < edges[i + 1])
        if m.sum() >= 3:
            out.append({"T_bin": round((edges[i] + edges[i + 1]) / 2, 1), "n": int(m.sum()),
                        "T_mean": round(float(T[m].mean()), 2), "T_std": round(float(T[m].std()), 2),
                        "P_mean": round(float(P[m].mean()), 3), "P_std": round(float(P[m].std()), 3)})
    return out


def fit_leakage(T, P):
    """P = a + b*exp(T/c): grid over c, linear LS for (a,b). Returns (a,b,c,rmse)."""
    best = None
    for c in np.linspace(8, 60, 300):
        X = np.column_stack([np.ones_like(T), np.exp(T / c)])
        coef, *_ = np.linalg.lstsq(X, P, rcond=None)
        res = P - X @ coef
        rmse = float(np.sqrt(np.mean(res**2)))
        if best is None or rmse < best[3]:
            best = (float(coef[0]), float(coef[1]), float(c), rmse)
    return best


def main():
    edges = np.linspace(T_LO, T_HI, 6)   # 5 temperature bins for the P(T) signature
    print(f"[{HOST}] controlled leakage RAMP probe, t={temp():.0f}C, band={T_LO}-{T_HI}C @ {RAMP_RATE}C/s",
          flush=True)
    ctl = Controller()
    try:
        print("  up-ramp (pass1)...", flush=True)
        T1, P1 = ramp_pass(ctl, T_LO, T_HI, up=True)
        print(f"    {len(T1)} samples, T {T1.min():.0f}->{T1.max():.0f}C, P {P1.min():.1f}-{P1.max():.1f}W"
              if len(T1) else "    no samples", flush=True)
        print("  down-ramp (pass2)...", flush=True)
        T2, P2 = ramp_pass(ctl, T_LO, T_HI, up=False)
        print(f"    {len(T2)} samples, T {T2.min():.0f}->{T2.max():.0f}C, P {P2.min():.1f}-{P2.max():.1f}W"
              if len(T2) else "    no samples", flush=True)
    finally:
        ctl.stop()

    bins1 = bin_PT(T1, P1, edges) if len(T1) else []
    bins2 = bin_PT(T2, P2, edges) if len(T2) else []
    pass1_ok = len(T1) >= 12 and len(bins1) >= 3 and (T1.max() - T1.min()) >= 10
    pass2_ok = len(T2) >= 12 and len(bins2) >= 3 and (T2.max() - T2.min()) >= 10
    if not pass1_ok:
        print(f"FAILED: pass1 {len(T1)} samples / {len(bins1)} bins / span "
              f"{(T1.max()-T1.min()) if len(T1) else 0:.0f}C", flush=True); return

    a1, b1, c1, r1 = fit_leakage(T1, P1)
    print(f"  pass1 fit a={a1:.3f} b={b1:.4f} c={c1:.2f} rmse={r1:.4f} ({len(bins1)} bins)", flush=True)
    if pass2_ok:
        a2, b2, c2, r2 = fit_leakage(T2, P2)
        print(f"  pass2 fit a={a2:.3f} b={b2:.4f} c={c2:.2f} rmse={r2:.4f} ({len(bins2)} bins)", flush=True)
    else:
        a2 = b2 = c2 = r2 = float("nan")
        print(f"  pass2 INVALID: {len(T2)} samples / {len(bins2)} bins", flush=True)

    # robust signature = binned P(T) curve; reproducibility = up-ramp vs down-ramp match at common bins
    sig1 = {b["T_bin"]: b["P_mean"] for b in bins1}
    if pass2_ok:
        sig2 = {b["T_bin"]: b["P_mean"] for b in bins2}
        common = [(sig1[k], sig2[k]) for k in sig1 if k in sig2]
        rel = np.mean([abs(p - q) / (0.5 * (p + q) + 1e-9) for p, q in common]) if common else float("nan")
        sig_repro = float(1 - rel)
        hysteresis_W = float(np.mean([p - q for p, q in common])) if common else float("nan")  # up minus down
    else:
        sig_repro = hysteresis_W = float("nan")
    P1mean = float(np.mean(P1))
    curvature_span = float(b1 * (np.exp(T1.max() / c1) - np.exp(T1.min() / c1)))
    out = {"host": HOST,
           "band_C": [T_LO, T_HI], "ramp_rate_C_s": RAMP_RATE,
           "pass1_up": {"a": round(a1, 3), "b": round(b1, 4), "c": round(c1, 3), "rmse": round(r1, 4),
                        "n": int(len(T1)), "bins": bins1},
           "pass2_down": {"a": round(a2, 3), "b": round(b2, 4), "c": round(c2, 3), "rmse": round(r2, 4),
                          "n": int(len(T2)), "bins": bins2, "valid": bool(pass2_ok)},
           "leakage_signature_PT": sig1,
           "leakage_signature_bc": ([round((b1 + b2) / 2, 4), round((c1 + c2) / 2, 3)] if pass2_ok
                                    else [round(b1, 4), round(c1, 3)]),
           "intra_die_reproducibility": (round(sig_repro, 4) if sig_repro == sig_repro else None),
           "hysteresis_up_minus_down_W": (round(hysteresis_W, 3) if hysteresis_W == hysteresis_W else None),
           "curvature_span_W": round(curvature_span, 3),
           "samples_pass1": {"T": [round(float(x), 2) for x in T1], "P": [round(float(x), 3) for x in P1]},
           "REPRODUCIBLE": bool(pass1_ok and pass2_ok and sig_repro == sig_repro and sig_repro > 0.9
                                and abs(curvature_span) > 0.3),
           "note": ("Controlled SLOW RAMP: a PI controller makes the die TRACK a slowly-sweeping reference "
                    "temperature across the band; idle power sampled in short gated windows (cores parked) "
                    "throughout, so each (T,P) sits at a near-stable known T with dense coverage. up-ramp = "
                    "pass1, down-ramp = pass2 (also a hysteresis check). P_idle(T)=a+b*exp(T/c); (b,c) and the "
                    "binned P(T) curve are the leakage/Vth signature. Compare ikaros vs daedalus for per-die "
                    "uniqueness.")}
    (OUT / f"leakage_controlled_{HOST}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if not k.startswith("samples")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
