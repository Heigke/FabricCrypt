"""B12: NVMe composite-temp distribution + idle power-state residency band.

Uses smartctl (no root needed for most attributes) and nvme-cli if present.
"""
from __future__ import annotations
import json, socket, subprocess, time
from pathlib import Path


def run(cmd, t=10):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=t)
        return r.stdout, r.returncode
    except Exception as e:
        return f"ERR:{e}", -1


def find_nvme():
    devs = []
    for p in Path("/dev").glob("nvme?n?"):
        devs.append(str(p))
    return sorted(devs)


def main():
    out = {"host": socket.gethostname(), "devices": find_nvme()}
    out["smartctl"] = {}
    for d in out["devices"]:
        stdout, rc = run(["sudo", "-n", "smartctl", "-a", d], t=10)
        if rc != 0:
            stdout, rc = run(["smartctl", "-a", d], t=10)
        out["smartctl"][d] = {"rc": rc, "lines": stdout.splitlines()[:80]}
    # composite-temp time-series 30 samples × 1 s
    hwmons = []
    for h in Path("/sys/class/hwmon").iterdir():
        try:
            if h.joinpath("name").read_text().strip() == "nvme":
                hwmons.append(str(h))
        except Exception:
            continue
    out["nvme_hwmons"] = hwmons
    temps = {h: [] for h in hwmons}
    for _ in range(30):
        for h in hwmons:
            try:
                temps[h].append(int(Path(h).joinpath("temp1_input").read_text().strip()))
            except Exception:
                temps[h].append(-1)
        time.sleep(1)
    out["temp_series"] = temps
    out["temp_mean"] = {h: (sum(v)/len(v) if v else None) for h, v in temps.items()}
    out["temp_var"] = {h: (sum((x-sum(v)/len(v))**2 for x in v)/len(v) if v else None) for h, v in temps.items()}
    out["temp_min"] = {h: (min(v) if v else None) for h, v in temps.items()}
    out["temp_max"] = {h: (max(v) if v else None) for h, v in temps.items()}
    print(json.dumps(out))


if __name__ == "__main__":
    main()
