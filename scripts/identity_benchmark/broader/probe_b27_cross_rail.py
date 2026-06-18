"""B27: Cross-rail covariance (multi-sensor 4×4 cov matrix).

Sensors: amdgpu in0, amdgpu in1, ucsi_source in0 (each port), acpi temp.
Sample 200×100ms = 20s. Build covariance matrix + Frobenius norm + per-pair r.
"""
from __future__ import annotations
import json, socket, time
from pathlib import Path


def read_int(p):
    try: return int(Path(p).read_text().strip())
    except Exception: return None


def main():
    chs = {}
    for h in Path("/sys/class/hwmon").iterdir():
        try: name = h.joinpath("name").read_text().strip()
        except Exception: continue
        if name == "amdgpu":
            for label in ("in0_input","in1_input","power1_average","freq1_input"):
                p = h/label
                if p.exists(): chs[f"amdgpu_{label}"] = p
        if name.startswith("ucsi"):
            for label in ("in0_input","curr1_input"):
                p = h/label
                if p.exists(): chs[f"{name}_{label}"] = p
    chs["acpi_temp"] = Path("/sys/class/thermal/thermal_zone0/temp")
    out = {"host": socket.gethostname(), "channels": list(chs.keys())}
    series = {k: [] for k in chs}
    t0 = time.time()
    while time.time() - t0 < 20:
        for k,p in chs.items():
            series[k].append(read_int(p))
        time.sleep(0.1)
    out["n"] = len(next(iter(series.values())))
    # zero-mean, unit-var per channel for correlation matrix
    def stats(v):
        v=[x for x in v if x is not None]
        if len(v)<2: return None,None
        m=sum(v)/len(v); s=(sum((x-m)**2 for x in v)/len(v))**0.5
        return m,s
    ms = {k: stats(v) for k,v in series.items()}
    keys = [k for k,(m,s) in ms.items() if m is not None and s and s>0]
    corr = {}
    for i,a in enumerate(keys):
        for b in keys[i+1:]:
            va, vb = series[a], series[b]
            n = min(len(va), len(vb))
            ma, sa = ms[a]; mb, sb = ms[b]
            num = 0; cnt = 0
            for k in range(n):
                if va[k] is None or vb[k] is None: continue
                num += (va[k]-ma)*(vb[k]-mb); cnt += 1
            if cnt > 5 and sa*sb > 0:
                corr[f"{a}__{b}"] = num/(cnt*sa*sb)
    out["mean"] = {k:m for k,(m,s) in ms.items()}
    out["std"] = {k:s for k,(m,s) in ms.items()}
    out["corr"] = corr
    # Frobenius norm of correlation matrix (excluding diag)
    fro = (sum(v*v for v in corr.values()))**0.5
    out["corr_frobenius"] = fro
    print(json.dumps(out))


if __name__ == "__main__":
    main()
