"""B32: Discrete fuse / firmware fingerprint enumeration.

Captures: dmidecode (board/uuid), cpuid model+stepping+ucode, AGESA hash from
SMBIOS, VBIOS hash, kernel cmdline, /proc/cpuinfo per-core flags hash.

Pure discrete identity (not analog) — establishes a by-design unique baseline.
"""
from __future__ import annotations
import hashlib, json, socket, subprocess, re
from pathlib import Path


def run(cmd, t=10):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=t)
        return r.stdout, r.returncode
    except Exception as e:
        return f"ERR:{e}", -1


def sha(s): return hashlib.sha256(s.encode() if isinstance(s,str) else s).hexdigest()[:16]


def main():
    out = {"host": socket.gethostname()}
    # cpuinfo per-core flags
    ci = Path("/proc/cpuinfo").read_text()
    out["cpuinfo_sha16"] = sha(ci)
    m = re.search(r"model name\s*:\s*(.+)", ci); out["model_name"] = m.group(1) if m else None
    m = re.search(r"microcode\s*:\s*(\S+)", ci); out["microcode"] = m.group(1) if m else None
    m = re.search(r"stepping\s*:\s*(\S+)", ci); out["stepping"] = m.group(1) if m else None
    # ucode from /sys
    for p in Path("/sys/devices/system/cpu").glob("cpu*/microcode/version"):
        try:
            out.setdefault("per_core_ucode", []).append((p.parts[-3], p.read_text().strip()))
        except Exception: pass
    # dmi (no root needed for some fields)
    for field in ("bios-version","bios-release-date","system-uuid","baseboard-serial-number","system-serial-number","processor-version"):
        s, rc = run(["sudo","-n","dmidecode","-s",field], t=5)
        if rc != 0: s, rc = run(["dmidecode","-s",field], t=5)
        out[f"dmi_{field}"] = s.strip() if rc==0 else f"rc={rc}"
    # full dmidecode hash
    s, rc = run(["sudo","-n","dmidecode"], t=10)
    if rc==0: out["dmi_full_sha16"] = sha(s)
    # vbios via amdgpu debugfs (requires root readable)
    try:
        for p in Path("/sys/kernel/debug/dri").glob("*/amdgpu_vbios"):
            try: out["vbios_sha16"] = sha(p.read_bytes()); break
            except Exception as e: out["vbios_err"] = str(e)
    except PermissionError:
        out["vbios_err"] = "debugfs perm denied"
    # gpu_metrics blob
    for p in Path("/sys/class/drm").glob("card*/device/gpu_metrics"):
        try: out["gpu_metrics_sha16"] = sha(p.read_bytes()); break
        except Exception: pass
    # MSR PLATFORM_ID 0x17 / PPIN if avail (read-only)
    s, rc = run(["sudo","-n","rdmsr","-a","0x17"], t=5)
    if rc==0: out["msr_0x17"] = s.strip().splitlines()
    s, rc = run(["sudo","-n","rdmsr","-a","0x8b"], t=5)
    if rc==0: out["msr_0x8b_ucode"] = s.strip().splitlines()
    # SMBIOS strings sha
    s, rc = run(["lspci","-vvv"], t=10)
    if rc==0: out["lspci_sha16"] = sha(s)
    # ASPM caps
    s, rc = run(["lspci","-vv","-d","1002:"], t=5)
    if rc==0: out["lspci_amdgpu_sha16"] = sha(s)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
