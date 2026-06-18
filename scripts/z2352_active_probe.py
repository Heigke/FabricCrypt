#!/usr/bin/env python3
"""
z2352_active_probe.py — Read GC registers while GPU is active (clock gates open)
Run a GPU compute kernel, then read registers via /dev/mem during execution.
"""
import subprocess, struct, os, sys, time, mmap, threading

# GC register effective addresses (dword offsets)
REGS = {
    "GRBM_STATUS":      0x1260 + 0x0DA4,  # = 0x2004
    "CP_MEC_CNTL":      0xA000 + 0x0802,  # = 0xA802
    "MEC_UCODE_ADDR":   0xA000 + 0x581A,  # = 0xF81A
    "MEC_UCODE_DATA":   0xA000 + 0x581B,  # = 0xF81B
    "CPC_IC_BASE_LO":   0xA000 + 0x584C,  # = 0xF84C
    "CPC_IC_BASE_HI":   0xA000 + 0x584D,  # = 0xF84D
    "CPC_IC_BASE_CNTL": 0xA000 + 0x584E,  # = 0xF84E
    "CPC_IC_OP_CNTL":   0xA000 + 0x297A,  # = 0xC97A
    "CPC_STATUS":       0xA000 + 0x2180,  # = 0xC180
}

BAR5_PHYS = 0xB4400000
BAR5_SIZE = 0x100000  # 1MB

# GPU stress kernel - simple OpenCL busy loop
GPU_STRESS_CMD = """
import time
try:
    import torch
    if torch.cuda.is_available():
        d = torch.device('cuda')
        # Allocate and do repeated matmul to keep GPU busy
        a = torch.randn(1024, 1024, device=d)
        for _ in range(200):
            a = torch.mm(a, a)
            a = a / a.norm()
        print("GPU_ACTIVE")
        time.sleep(0.5)
        print("GPU_DONE")
    else:
        print("NO_CUDA")
except Exception as e:
    print(f"ERR: {e}")
"""

def read_bar5_regs():
    """Read GC registers via /dev/mem"""
    try:
        fd = os.open("/dev/mem", os.O_RDONLY | os.O_SYNC)
        mm = mmap.mmap(fd, BAR5_SIZE, mmap.MAP_SHARED, mmap.PROT_READ, offset=BAR5_PHYS)

        print("=== GC Register Probe (BAR5 direct) ===")
        for name, dword_off in sorted(REGS.items(), key=lambda x: x[1]):
            byte_off = dword_off * 4
            if byte_off + 4 > BAR5_SIZE:
                print(f"  {name:20s} [0x{dword_off:05X}] BEYOND BAR5")
                continue
            mm.seek(byte_off)
            val = struct.unpack('<I', mm.read(4))[0]
            print(f"  {name:20s} [0x{dword_off:05X}] = 0x{val:08X}")

        mm.close()
        os.close(fd)
    except Exception as e:
        print(f"BAR5 read error: {e}")

# First read: idle
print("--- IDLE STATE ---")
read_bar5_regs()

# Launch GPU workload in background
print("\n--- LAUNCHING GPU WORKLOAD ---")
env = os.environ.copy()
env["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"
proc = subprocess.Popen(
    [sys.executable, "-c", GPU_STRESS_CMD],
    env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE
)

# Wait a moment for GPU to start
time.sleep(2.0)

# Read while GPU is active
print("\n--- ACTIVE STATE (during compute) ---")
read_bar5_regs()

# Collect output
stdout, stderr = proc.communicate(timeout=30)
print(f"\nGPU workload output: {stdout.decode().strip()}")
if stderr.decode().strip():
    print(f"GPU stderr: {stderr.decode().strip()[:200]}")
