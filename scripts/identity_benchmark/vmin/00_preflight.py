#!/usr/bin/env python3
"""Stage 0 preflight: enumerate available undervolt mechanisms on this host.

Writes a JSON report describing what *would* be available, and a GO / NO-GO
verdict. This script never writes to any voltage-control register. It is
purely read-only and safe to run at any time.

Decision tree:
  - amd-pstate-epp + cpupower available     → can set frequency floor/ceil only (NOT undervolt)
  - rdmsr/wrmsr + /dev/cpu/N/msr writable   → can read P-state MSRs but writing AMD VID
                                              requires the SMU mailbox (data-fabric sync flood
                                              risk on Zen 5 Strix Halo)
  - ryzen_smu module loaded + smn writable  → can talk to SMU mailbox; documented crash mode
                                              on this hardware (MEMORY.md UMR section)
  - ryzenadj available                      → uses SMU mailbox under the hood; same risk

On Zen 5 Strix Halo (AMD Ryzen AI Max+ 395), there is no published safe path
to push CPU below per-die Vmin from user space without going through the SMU
mailbox. The local repo's ryzen_smu has Strix Halo support but the campaign
has documented two crashes from SMU mailbox writes already.

VERDICT: NO-GO for live undervolt. Track 2 aborts to design-doc only.
"""
from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import time
from pathlib import Path

OUT = Path(__file__).resolve().parents[3] / "results" / "IDENTITY_VMIN_2026-05-30"
OUT.mkdir(parents=True, exist_ok=True)


def have(cmd: str) -> str | None:
    return shutil.which(cmd)


def run(*args, **kw) -> tuple[int, str]:
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=8, **kw)
        return p.returncode, (p.stdout + p.stderr).strip()
    except Exception as e:  # noqa: BLE001
        return -1, repr(e)


def main() -> int:
    host = socket.gethostname()
    rep: dict = {
        "host": host,
        "ts": time.time(),
        "cpu_model": None,
        "kernel": None,
        "mechanisms": {},
        "verdict": "NO-GO",
        "verdict_reason": "",
    }
    try:
        rep["cpu_model"] = next(
            (l.split(":", 1)[1].strip() for l in Path("/proc/cpuinfo").read_text().splitlines()
             if l.startswith("model name")), "unknown")
    except Exception:  # noqa: BLE001
        pass
    rc, out = run("uname", "-r")
    rep["kernel"] = out

    # cpupower: frequency floor only, NOT a voltage knob
    rc, out = run("cpupower", "frequency-info")
    rep["mechanisms"]["cpupower"] = {
        "present": have("cpupower") is not None,
        "freq_governor_available": "amd-pstate" in out,
        "note": "frequency floor/ceil only, does NOT undervolt",
    }

    # MSR: rdmsr/wrmsr presence + device node
    msr_dev_ok = os.path.exists("/dev/cpu/0/msr")
    rep["mechanisms"]["msr"] = {
        "rdmsr": have("rdmsr") is not None,
        "wrmsr": have("wrmsr") is not None,
        "dev_present": msr_dev_ok,
        "note": "AMD VID lives behind SMU mailbox; MSR 0x1B0 in design doc is INTEL not AMD",
    }
    # P-state read-only sanity
    rc, out = run("sudo", "-n", "rdmsr", "-p", "0", "0xC0010293")
    rep["mechanisms"]["msr"]["msr_C0010293_pstate"] = out if rc == 0 else f"err: {out}"

    # ryzen_smu: loaded?
    rc, out = run("lsmod")
    loaded = "ryzen_smu" in out
    sysfs_present = os.path.exists("/sys/kernel/ryzen_smu_drv")
    rep["mechanisms"]["ryzen_smu"] = {
        "module_loaded": loaded,
        "sysfs_present": sysfs_present,
        "ko_on_disk": str(Path("/home/ikaros/Documents/claude_hive/ryzen_smu/ryzen_smu.ko")
                          .exists()),
        "note": (
            "Strix Halo supported in source. Loading + mailbox write is documented crash mode on "
            "this hardware (MEMORY.md: 'NEVER write to SMU mailbox → Data Fabric Sync Flood → "
            "instant reboot'). Campaign has crashed ikaros TWICE from related operations."
        ),
    }

    # ryzenadj
    rep["mechanisms"]["ryzenadj"] = {
        "present": have("ryzenadj") is not None,
        "note": "uses SMU mailbox under the hood; same risk class",
    }

    # Final verdict
    if rep["mechanisms"]["ryzen_smu"]["module_loaded"] and rep["mechanisms"]["ryzen_smu"]["sysfs_present"]:
        rep["verdict"] = "CONDITIONAL-GO-WITH-WATCHDOG"
        rep["verdict_reason"] = (
            "ryzen_smu loaded; SMU mailbox accessible. Proceed ONLY with PROVEN watchdog, "
            "10 mV steps, 30 s hold, immediate restore on >50% bitflip or 60 s telemetry stall."
        )
    else:
        rep["verdict"] = "NO-GO"
        rep["verdict_reason"] = (
            "No safe user-space undervolt mechanism available on Zen 5 Strix Halo. "
            "cpupower controls frequency only; AMD VID writes require SMU mailbox which is "
            "documented crash mode on this exact hardware. Track 2 aborts to design-doc only."
        )

    out_path = OUT / f"preflight_{host}.json"
    out_path.write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))
    print(f"\nWROTE: {out_path}")
    return 0 if rep["verdict"].startswith("CONDITIONAL") else 2


if __name__ == "__main__":
    raise SystemExit(main())
