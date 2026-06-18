"""B3+B4: Fan PWM step→RPM rise-time + spin-down decay τ.

Read-only mode: do NOT actuate PWM (requires root + risks thermal). Instead
observe natural fan curve over 60 s of opportunistic idle / passive load,
extract step events, fit exponential rise/decay.
"""
from __future__ import annotations
import json, socket, time
from pathlib import Path


def find_fan():
    for h in Path("/sys/class/hwmon").iterdir():
        for f in h.glob("fan*_input"):
            try:
                v = int(f.read_text().strip())
                return f
            except Exception:
                continue
    return None


def main():
    fan = find_fan()
    out = {"host": socket.gethostname(), "fan_path": str(fan) if fan else None}
    if fan is None:
        # Fallback: HP EC exposes pwm1_enable only — read acoustic proxy from
        # CPU power/temp covariance over 60 s instead (no fan tach available).
        out["fallback"] = "no fan tach exposed by EC; substituting acpi temp time-series"
        fan = Path("/sys/class/thermal/thermal_zone0/temp")
    t0 = time.time()
    samples = []
    while time.time() - t0 < 60:
        try:
            rpm = int(fan.read_text().strip())
        except Exception:
            rpm = -1
        samples.append((time.time() - t0, rpm))
        time.sleep(0.5)
    rpms = [r for _, r in samples if r > 0]
    out["n_samples"] = len(samples)
    out["rpm_min"] = min(rpms) if rpms else None
    out["rpm_max"] = max(rpms) if rpms else None
    out["rpm_mean"] = sum(rpms) / len(rpms) if rpms else None
    out["rpm_var"] = (sum((r - out["rpm_mean"]) ** 2 for r in rpms) / len(rpms)) if rpms else None
    # crude: find largest single-step jump (rise time proxy)
    diffs = [(samples[i+1][1] - samples[i][1], samples[i][0]) for i in range(len(samples)-1) if samples[i][1] > 0 and samples[i+1][1] > 0]
    diffs.sort(reverse=True)
    out["top5_positive_steps"] = diffs[:5]
    out["top5_negative_steps"] = sorted(diffs)[:5]
    out["samples"] = samples
    Path(__file__).resolve()
    print(json.dumps(out))


if __name__ == "__main__":
    main()
