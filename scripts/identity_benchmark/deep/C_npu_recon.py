"""C. NPU XDNA recon + bare attempt at userspace. Step 1: enumerate.
We attempt to load a kernel ONLY if XRT userspace + an .xclbin / .vaie exist.
Otherwise report honest blocker.
"""
import argparse, os, sys, time, glob, subprocess
sys.path.insert(0, os.path.dirname(__file__))
from _common import save_json, host_label

def sh(cmd, t=5):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=t)
        return dict(rc=r.returncode, out=r.stdout[-2000:], err=r.stderr[-1000:])
    except Exception as e:
        return dict(rc=-1, out="", err=str(e))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args=ap.parse_args()
    report = dict(host=host_label())
    # device
    report["dev_accel0"] = os.path.exists("/dev/accel/accel0")
    report["dev_accel_perms"] = sh("ls -la /dev/accel/")["out"]
    # kernel module
    report["lsmod_amdxdna"] = sh("lsmod | grep -i amdxdna")["out"]
    report["modinfo_amdxdna"] = sh("modinfo amdxdna 2>/dev/null | head -20")["out"]
    report["dmesg_amdxdna"] = sh("dmesg 2>/dev/null | grep -i amdxdna | tail -10")["out"]
    # pci
    report["lspci_npu"] = sh("lspci -nn | grep -iE 'signal proc|17f0'")["out"]
    # xrt
    report["which_xrt_smi"] = sh("which xrt-smi xrtutil 2>/dev/null")["out"]
    report["xilinx_dir"] = sh("ls /opt/xilinx/ 2>/dev/null")["out"]
    report["dpkg_xrt"] = sh("dpkg -l 2>/dev/null | grep -iE 'xrt|xdna|ryzen-ai' | head")["out"]
    report["python_xrt"] = sh("python3 -c 'import xrt' 2>&1")["out"]
    # firmware blobs
    report["fw_amdxdna"] = sh("ls /lib/firmware/amdnpu/ 2>/dev/null; ls /lib/firmware/amdxdna/ 2>/dev/null")["out"]
    # debugfs
    report["debugfs_accel"] = sh("ls /sys/kernel/debug/accel/ 2>/dev/null")["out"]
    # decide blocker
    has_xrt = bool(report["which_xrt_smi"].strip())
    has_pyxrt = "Traceback" not in report["python_xrt"]
    report["blocker"] = {
        "xrt_userspace_present": has_xrt,
        "python_xrt_binding": has_pyxrt,
        "needed_for_fingerprint": "xrt-smi or pyxrt + a Ryzen-AI .xclbin from AMD's amd/RyzenAI-SW repo (NPU compiled model); without it the /dev/accel/accel0 char device cannot be exercised from userspace.",
        "verdict": "BLOCKED" if not (has_xrt or has_pyxrt) else "MAYBE",
    }
    save_json(args.out, report)
    print("VERDICT:", report["blocker"]["verdict"], flush=True)

if __name__=="__main__": main()
