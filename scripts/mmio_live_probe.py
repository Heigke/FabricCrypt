#!/usr/bin/env python3
"""
mmio_live_probe.py — Try multiple register access methods to find
dynamic GPU state observable via /dev/mem during kernel execution.

Methods:
1. Direct BAR5 MMIO (failed — clock gated)
2. MM_INDEX/MM_DATA indirect access (may bypass gating)
3. SMU/NBIO registers (always-on domain)
4. HDP registers (memory controller, always-on)
5. IH registers (interrupt handler)
6. Wider register scan to find ANY dynamic register
"""
import mmap, struct, os, sys, time, subprocess, threading
import numpy as np

BAR5_PHYS = 0xB4400000
BAR5_SIZE = 0x100000

def open_bar5():
    fd = os.open('/dev/mem', os.O_RDWR | os.O_SYNC)
    mm = mmap.mmap(fd, BAR5_SIZE, mmap.MAP_SHARED,
                   mmap.PROT_READ | mmap.PROT_WRITE, offset=BAR5_PHYS)
    return fd, mm

def rreg(mm, reg_idx):
    off = reg_idx * 4
    if off + 4 > BAR5_SIZE:
        return 0xDEADDEAD
    mm.seek(off)
    return struct.unpack('<I', mm.read(4))[0]

def wreg(mm, reg_idx, val):
    off = reg_idx * 4
    mm.seek(off)
    mm.write(struct.pack('<I', val))

def rreg_indirect(mm, reg_addr):
    """Read register via MM_INDEX/MM_DATA indirect access.
    MM_INDEX = BAR5+0x0, MM_DATA = BAR5+0x4 (byte offsets)"""
    mm.seek(0)
    mm.write(struct.pack('<I', reg_addr * 4))  # write address to MM_INDEX
    mm.seek(4)
    return struct.unpack('<I', mm.read(4))[0]  # read from MM_DATA

def launch_persistent_kernel():
    """Launch a HIP kernel that runs for ~30 seconds"""
    src = '/tmp/mmio_persistent.hip'
    exe = '/tmp/mmio_persistent'
    with open(src, 'w') as f:
        f.write('''
#include <hip/hip_runtime.h>
#include <stdio.h>
__global__ void busy_kernel(volatile float* data, int n, int iters) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    float x = (float)i * 0.001f;
    for (int j = 0; j < iters; j++) {
        x = sinf(x) * cosf(x + 0.1f) + 0.001f;
        if (i < n) data[i] = x;
    }
}
int main() {
    printf("Kernel starting...\\n"); fflush(stdout);
    float *d;
    hipMalloc(&d, 65536*sizeof(float));
    // Run for ~30 seconds
    for (int rep = 0; rep < 500; rep++) {
        busy_kernel<<<256, 256>>>(d, 65536, 50000);
    }
    hipDeviceSynchronize();
    hipFree(d);
    printf("Kernel done.\\n");
    return 0;
}
''')
    env = os.environ.copy()
    env['HSA_OVERRIDE_GFX_VERSION'] = '11.0.0'
    r = subprocess.run(['/opt/rocm/bin/hipcc', '-O2', '-o', exe, src],
                       env=env, capture_output=True, timeout=120)
    if r.returncode != 0:
        print(f"Compile failed: {r.stderr.decode()[:200]}")
        return None
    return subprocess.Popen([exe], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def main():
    print("=" * 60)
    print("MMIO LIVE PROBE — Finding dynamic GPU registers")
    print("=" * 60)

    fd, mm = open_bar5()

    # ===== METHOD 1: Direct vs Indirect comparison =====
    print("\n--- Method comparison (GPU idle) ---")
    test_regs = {
        'GRBM_STATUS': 0x0504,
        'GRBM_STATUS2': 0x0502,
        'CP_STAT': 0x0E40,
        'CP_CPC_STATUS': 0x0E54,
    }
    for name, idx in test_regs.items():
        direct = rreg(mm, idx)
        indirect = rreg_indirect(mm, idx)
        print(f"  {name:20s} direct=0x{direct:08x}  indirect=0x{indirect:08x}  "
              f"{'DIFFER' if direct != indirect else 'same'}")

    # ===== METHOD 2: Scan always-on domains =====
    print("\n--- Scanning always-on register domains ---")

    # Known always-on register ranges on RDNA3:
    always_on_ranges = {
        'NBIO':     (0x0000, 0x0100),  # NBIO/BIF
        'HDP':      (0x0F00, 0x0F80),  # Host Data Path
        'OSSSYS':   (0x000A, 0x0020),  # OS System
        'MMHUB':    (0x0680, 0x0700),  # Memory Management Hub
        'SMN_LOW':  (0x0000, 0x0040),  # SMN access
    }

    dynamic_regs = []
    for domain, (start, end) in always_on_ranges.items():
        nonzero = 0
        for idx in range(start, end):
            v = rreg(mm, idx)
            if v != 0 and v != 0xFFFFFFFF:
                nonzero += 1
                if nonzero <= 3:
                    print(f"  {domain} reg[0x{idx:04x}] = 0x{v:08x}")
        if nonzero > 3:
            print(f"  {domain}: ... {nonzero} non-trivial registers total")
        elif nonzero == 0:
            print(f"  {domain}: all zero/FFFFFFFF")

    # ===== METHOD 3: Wide scan for ANY non-static register =====
    print("\n--- Wide scan: sampling ALL registers twice, 100ms apart ---")

    # Sample all registers twice
    n_regs = BAR5_SIZE // 4  # 262144 registers
    # Too many — scan in chunks of interesting ranges
    scan_ranges = [
        (0x0000, 0x1000),   # Low range (NBIO, BIF, HDP, OSSSYS)
        (0x0D00, 0x0F00),   # SDMA range
        (0x0E00, 0x0F00),   # CP range
        (0x2800, 0x2A00),   # MES range
        (0x4C00, 0x4D00),   # RLC range
    ]

    snap1 = {}
    for start, end in scan_ranges:
        for idx in range(start, end):
            snap1[idx] = rreg(mm, idx)

    time.sleep(0.1)

    changed = []
    for start, end in scan_ranges:
        for idx in range(start, end):
            v2 = rreg(mm, idx)
            if v2 != snap1[idx]:
                changed.append((idx, snap1[idx], v2))

    if changed:
        print(f"  Found {len(changed)} registers that changed in 100ms!")
        for idx, v1, v2 in changed[:20]:
            print(f"    reg[0x{idx:04x}]: 0x{v1:08x} → 0x{v2:08x}")
    else:
        print("  No registers changed (all static in 100ms)")

    # ===== METHOD 4: Launch kernel, then scan =====
    print("\n--- Launching persistent HIP kernel (~30s) ---")
    proc = launch_persistent_kernel()
    if proc is None:
        print("  Failed to launch kernel")
        mm.close(); os.close(fd)
        return

    time.sleep(2)  # Let kernel get going

    print("--- Sampling during kernel execution ---")

    # Direct reads
    for name, idx in test_regs.items():
        direct = rreg(mm, idx)
        indirect = rreg_indirect(mm, idx)
        print(f"  {name:20s} direct=0x{direct:08x}  indirect=0x{indirect:08x}")

    # Scan for changed registers during kernel
    print("\n--- Wide scan during kernel execution (2 snapshots, 100ms apart) ---")
    snap_a = {}
    for start, end in scan_ranges:
        for idx in range(start, end):
            snap_a[idx] = rreg(mm, idx)

    time.sleep(0.1)

    changed_during = []
    for start, end in scan_ranges:
        for idx in range(start, end):
            v2 = rreg(mm, idx)
            if v2 != snap_a[idx]:
                changed_during.append((idx, snap_a[idx], v2))

    if changed_during:
        print(f"  FOUND {len(changed_during)} dynamic registers during kernel!")
        for idx, v1, v2 in changed_during[:30]:
            print(f"    reg[0x{idx:04x}]: 0x{v1:08x} → 0x{v2:08x}")

        # For the most dynamic registers, do a burst sample
        best_regs = [idx for idx, _, _ in changed_during[:4]]
        print(f"\n--- Burst sampling {len(best_regs)} dynamic registers (10000 reads) ---")
        data = np.zeros((10000, len(best_regs)), dtype=np.uint32)
        t0 = time.perf_counter()
        for i in range(10000):
            for j, idx in enumerate(best_regs):
                data[i, j] = rreg(mm, idx)
        dt = time.perf_counter() - t0
        print(f"  Rate: {10000/dt:.0f} Hz")

        for j, idx in enumerate(best_regs):
            col = data[:, j]
            unique = len(np.unique(col))
            transitions = np.sum(col[1:] != col[:-1])
            d = col.astype(float)
            if np.std(d) > 0:
                d = d - d.mean()
                acf = np.corrcoef(d[:-1], d[1:])[0, 1]
            else:
                acf = 1.0
            vals, counts = np.unique(col, return_counts=True)
            entropy = -np.sum((counts/len(col)) * np.log2(counts/len(col) + 1e-30))
            print(f"  reg[0x{idx:04x}]: unique={unique}, transitions={transitions}, "
                  f"entropy={entropy:.2f}bits, ACF={acf:.4f}")
    else:
        print("  No registers changed during kernel execution either.")

        # Last resort: try indirect access during kernel
        print("\n--- Trying INDIRECT access during kernel ---")
        for name, idx in test_regs.items():
            v = rreg_indirect(mm, idx)
            print(f"  {name:20s} indirect=0x{v:08x}")

        # Try reading via the scratch register trick
        # On some GPUs, SCRATCH_REG can be written from shader and read from CPU
        print("\n--- Scanning scratch registers (0x2000-0x2100) ---")
        for idx in range(0x2000, 0x2100):
            v = rreg(mm, idx)
            if v != 0 and v != 0xFFFFFFFF:
                print(f"  reg[0x{idx:04x}] = 0x{v:08x}")

    # Cleanup
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except:
        proc.kill()

    mm.close()
    os.close(fd)
    print("\nDone.")

if __name__ == '__main__':
    main()
