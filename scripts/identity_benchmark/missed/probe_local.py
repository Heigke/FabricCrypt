"""M2 boot-time + M11 TSC drift + M13 CPU MSR PLATFORM_ID.
Run on ikaros (default) or daedalus. CPU/disk only, no GPU stress.
"""
from __future__ import annotations
import json, os, socket, subprocess, time
from pathlib import Path


def m2_boot_time():
    """Parse journalctl boot list + dmesg kernel-ready timestamp."""
    out = {}
    try:
        r = subprocess.run(["journalctl", "--list-boots", "--no-pager"],
                           capture_output=True, text=True, timeout=10)
        lines = [l for l in r.stdout.splitlines() if l.strip().startswith("-") or l.strip().startswith(" 0")]
        out["boot_list_last5"] = lines[-5:]
    except Exception as e:
        out["boot_list_err"] = str(e)
    # systemd-analyze gives boot time breakdown
    try:
        r = subprocess.run(["systemd-analyze"], capture_output=True, text=True, timeout=5)
        out["systemd_analyze"] = r.stdout.strip()
    except Exception as e:
        out["systemd_analyze_err"] = str(e)
    try:
        r = subprocess.run(["systemd-analyze", "blame"], capture_output=True, text=True, timeout=5)
        out["systemd_blame_top10"] = r.stdout.strip().splitlines()[:10]
    except Exception as e:
        out["systemd_blame_err"] = str(e)
    return out


def m11_tsc_drift(duration_s: float = 30.0):
    """Sample CLOCK_MONOTONIC (TSC-derived on most x86) vs CLOCK_REALTIME drift."""
    samples = []
    t0_mono = time.clock_gettime(time.CLOCK_MONOTONIC)
    t0_real = time.clock_gettime(time.CLOCK_REALTIME)
    end = time.time() + duration_s
    while time.time() < end:
        mono = time.clock_gettime(time.CLOCK_MONOTONIC) - t0_mono
        real = time.clock_gettime(time.CLOCK_REALTIME) - t0_real
        samples.append((mono, real, real - mono))
        time.sleep(0.05)
    # linear fit of (real - mono) vs mono
    import numpy as np
    arr = np.asarray(samples)
    mono_s = arr[:, 0]
    drift = arr[:, 2]
    # ppm = slope * 1e6
    slope, intercept = np.polyfit(mono_s, drift, 1)
    return {
        "n_samples": int(arr.shape[0]),
        "duration_s": float(mono_s[-1] - mono_s[0]),
        "drift_slope_ppm": float(slope * 1e6),
        "drift_intercept_s": float(intercept),
        "drift_std_s": float(drift.std()),
        "drift_first": float(drift[0]),
        "drift_last": float(drift[-1]),
    }


def m13_cpu_id():
    """Read PLATFORM_ID (MSR 0x17), CPUID signature, microcode rev."""
    out = {}
    # /proc/cpuinfo basics
    try:
        with open("/proc/cpuinfo") as f:
            cp = f.read()
        for k in ("model name", "stepping", "microcode", "cpu family", "model"):
            for line in cp.splitlines():
                if line.startswith(k):
                    out[k.replace(" ", "_")] = line.split(":", 1)[1].strip()
                    break
    except Exception as e:
        out["cpuinfo_err"] = str(e)
    # /sys/devices/system/cpu/microcode/processor_flags (intel-specific; harmless)
    try:
        p = Path("/sys/devices/system/cpu/microcode/processor_flags")
        if p.exists():
            out["processor_flags"] = p.read_text().strip()
    except Exception as e:
        out["pflags_err"] = str(e)
    # rdmsr 0x17 (PLATFORM_ID — Intel-only but try anyway)
    for msr_name, msr in [("PLATFORM_ID_0x17", "0x17"),
                          ("PATCH_LEVEL_0x8b", "0x8b"),  # AMD ucode rev
                          ]:
        try:
            r = subprocess.run(["sudo", "-n", "rdmsr", "-p", "0", msr],
                               capture_output=True, text=True, timeout=3)
            if r.returncode == 0:
                out[msr_name] = r.stdout.strip()
            else:
                out[f"{msr_name}_err"] = r.stderr.strip() or "rc!=0"
        except Exception as e:
            out[f"{msr_name}_err"] = str(e)
    # SMBIOS / DMI per-board serials (silicon-adjacent)
    for k, p in [
        ("dmi_board_serial", "/sys/class/dmi/id/board_serial"),
        ("dmi_product_uuid", "/sys/class/dmi/id/product_uuid"),
        ("dmi_chassis_serial", "/sys/class/dmi/id/chassis_serial"),
        ("dmi_bios_version", "/sys/class/dmi/id/bios_version"),
        ("dmi_bios_date", "/sys/class/dmi/id/bios_date"),
    ]:
        try:
            v = Path(p).read_text().strip()
            out[k] = v
        except Exception:
            pass
    # CPUID family/model/stepping via lscpu
    try:
        r = subprocess.run(["lscpu"], capture_output=True, text=True, timeout=3)
        for line in r.stdout.splitlines():
            if any(k in line for k in ("CPU family", "Model:", "Stepping", "BogoMIPS")):
                out.setdefault("lscpu", []).append(line.strip())
    except Exception as e:
        out["lscpu_err"] = str(e)
    return out


def main():
    host = socket.gethostname()
    out_root = Path(os.environ.get("OUT_ROOT",
        "/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/IDENTITY_BENCHMARK_2026-05-30/missed"))
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"[{host}] M2 boot...")
    m2 = m2_boot_time()
    (out_root / f"M2_{host}.json").write_text(json.dumps(m2, indent=2))

    print(f"[{host}] M11 TSC drift (30s)...")
    m11 = m11_tsc_drift(30.0)
    (out_root / f"M11_{host}.json").write_text(json.dumps(m11, indent=2))

    print(f"[{host}] M13 CPU ID...")
    m13 = m13_cpu_id()
    (out_root / f"M13_{host}.json").write_text(json.dumps(m13, indent=2))

    print(f"[{host}] DONE. wrote to {out_root}")


if __name__ == "__main__":
    main()
