"""B24: Power × temp cross-correlation lag (no GPU stress — passive observation).

Observe amdgpu power1_average + edge temp + ACPI thermal zone at 10 Hz for
60 s, compute lag-correlation curve (lag ∈ [-2, +2] s) for each pair.
"""
from __future__ import annotations
import json, socket, time
from pathlib import Path


def read_int(p):
    try:
        return int(Path(p).read_text().strip())
    except Exception:
        return None


def find_amdgpu():
    for h in Path("/sys/class/hwmon").iterdir():
        try:
            if h.joinpath("name").read_text().strip() == "amdgpu":
                return str(h)
        except Exception:
            continue
    return None


def main():
    h = find_amdgpu()
    out = {"host": socket.gethostname(), "amdgpu": h}
    if h is None:
        out["error"] = "no amdgpu hwmon"
        print(json.dumps(out)); return
    sensors = {
        "power_uw": Path(h)/"power1_average",
        "in0_uv": Path(h)/"in0_input",
        "freq_hz": Path(h)/"freq1_input",
        "acpi_temp": "/sys/class/thermal/thermal_zone0/temp",
    }
    # detect temp sensor on amdgpu
    for f in Path(h).glob("temp*_input"):
        sensors["gpu_temp"] = f
        break
    series = {k: [] for k in sensors}
    t0 = time.time()
    times = []
    while time.time() - t0 < 60:
        ts = time.time() - t0
        times.append(ts)
        for k, p in sensors.items():
            series[k].append(read_int(p))
        time.sleep(0.1)
    out["n"] = len(times)

    def mean(v):
        v = [x for x in v if x is not None]
        return (sum(v)/len(v)) if v else None
    def var(v):
        v = [x for x in v if x is not None]
        if len(v) < 2: return None
        m = sum(v)/len(v)
        return sum((x-m)**2 for x in v)/len(v)
    def xcorr_at_lag(a, b, lag):
        # pearson at integer-sample lag
        n = min(len(a), len(b))
        if lag >= 0:
            x = a[:n-lag]; y = b[lag:n]
        else:
            x = a[-lag:n]; y = b[:n+lag]
        x = [u for u in x if u is not None]; y = [u for u in y if u is not None]
        m = min(len(x), len(y)); x=x[:m]; y=y[:m]
        if m < 5: return None
        mx, my = sum(x)/m, sum(y)/m
        num = sum((x[i]-mx)*(y[i]-my) for i in range(m))
        dx = (sum((xi-mx)**2 for xi in x))**0.5
        dy = (sum((yi-my)**2 for yi in y))**0.5
        if dx*dy == 0: return None
        return num/(dx*dy)

    out["mean"] = {k: mean(v) for k,v in series.items()}
    out["var"] = {k: var(v) for k,v in series.items()}
    pairs = [("power_uw","gpu_temp"), ("power_uw","acpi_temp"), ("gpu_temp","acpi_temp"),
             ("freq_hz","power_uw"), ("in0_uv","power_uw")]
    out["xcorr_lags"] = {}
    for a, b in pairs:
        if a not in series or b not in series: continue
        lags = list(range(-20, 21))
        cc = [xcorr_at_lag(series[a], series[b], l) for l in lags]
        out["xcorr_lags"][f"{a}__{b}"] = {"lags_x100ms": lags, "cc": cc}
        # peak
        cc_clean = [(l,c) for l,c in zip(lags, cc) if c is not None]
        if cc_clean:
            peak = max(cc_clean, key=lambda x: x[1])
            out["xcorr_lags"][f"{a}__{b}"]["peak_lag_100ms"] = peak[0]
            out["xcorr_lags"][f"{a}__{b}"]["peak_cc"] = peak[1]
    print(json.dumps(out))


if __name__ == "__main__":
    main()
