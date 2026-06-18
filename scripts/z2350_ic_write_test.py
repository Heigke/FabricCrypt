#!/usr/bin/env python3
"""z2350: Test IC register writability at CORRECT offsets (0x5840-0x5852).

Previous test used WRONG offsets (0x585A etc). This uses the correct ones
from kernel source / z2348 results.

Must run as root (sudo).
"""
import struct, time, json, os

REGS_PATH = '/sys/kernel/debug/dri/0/amdgpu_regs'

offsets = {
    'CP_PFP_IC_BASE_LO':   0x5840,
    'CP_PFP_IC_BASE_HI':   0x5841,
    'CP_PFP_IC_BASE_CNTL': 0x5842,
    'CP_PFP_IC_OP_CNTL':   0x5843,
    'CP_ME_IC_BASE_LO':    0x5844,
    'CP_ME_IC_BASE_HI':    0x5845,
    'CP_ME_IC_BASE_CNTL':  0x5846,
    'CP_ME_IC_OP_CNTL':    0x5847,
    'CP_CPC_IC_BASE_LO':   0x584C,
    'CP_CPC_IC_BASE_HI':   0x584D,
    'CP_CPC_IC_BASE_CNTL': 0x584E,
    'CP_MES_IC_BASE_LO':   0x5850,
    'CP_MES_IC_BASE_HI':   0x5851,
    'CP_MES_IC_BASE_CNTL': 0x5852,
}

def read_reg(off):
    with open(REGS_PATH, 'rb') as f:
        f.seek(off * 4)
        return struct.unpack('<I', f.read(4))[0]

def write_reg(off, val):
    with open(REGS_PATH, 'r+b') as f:
        f.seek(off * 4)
        f.write(struct.pack('<I', val))
        f.flush()

def main():
    results = {}

    print("=== z2350: IC Register Read (correct offsets) ===")
    vals = {}
    for name, off in offsets.items():
        val = read_reg(off)
        vals[name] = val
        print(f"  {name:25s} (0x{off:04X}) = 0x{val:08X}")

    print()
    print("=== WRITE TEST: ME IC registers (0x5844-0x5847) ===")
    me_regs = [
        ('CP_ME_IC_BASE_LO',   0x5844),
        ('CP_ME_IC_BASE_HI',   0x5845),
        ('CP_ME_IC_BASE_CNTL', 0x5846),
        ('CP_ME_IC_OP_CNTL',   0x5847),
    ]

    for name, off in me_regs:
        orig = vals[name]
        test_val = 0xA5A50000 | (off & 0xFFFF)

        try:
            write_reg(off, test_val)
        except Exception as e:
            print(f"  {name}: WRITE FAILED: {e}")
            results[name] = {'writable': False, 'error': str(e)}
            continue

        time.sleep(0.01)
        readback = read_reg(off)

        writable = (readback != orig)
        exact = (readback == test_val)

        status = "WRITABLE" if writable else "read-only"
        if writable and not exact:
            status += f" (partial: got 0x{readback:08X})"

        print(f"  {name} (0x{off:04X}): orig=0x{orig:08X} wrote=0x{test_val:08X} read=0x{readback:08X}  {status}")

        results[name] = {
            'offset': f"0x{off:04X}",
            'original': f"0x{orig:08X}",
            'test_val': f"0x{test_val:08X}",
            'readback': f"0x{readback:08X}",
            'writable': writable,
            'exact_match': exact,
        }

        # RESTORE original value
        if writable:
            write_reg(off, orig)
            time.sleep(0.01)
            restored = read_reg(off)
            ok = (restored == orig)
            print(f"    -> restored to 0x{restored:08X} {'OK' if ok else 'MISMATCH!'}")
            results[name]['restored'] = ok

    print()
    print("=== WRITE TEST: PFP IC registers (0x5840-0x5843) ===")
    pfp_regs = [
        ('CP_PFP_IC_BASE_LO',   0x5840),
        ('CP_PFP_IC_BASE_HI',   0x5841),
        ('CP_PFP_IC_BASE_CNTL', 0x5842),
        ('CP_PFP_IC_OP_CNTL',   0x5843),
    ]

    for name, off in pfp_regs:
        orig = vals[name]
        test_val = 0xB5B50000 | (off & 0xFFFF)

        try:
            write_reg(off, test_val)
        except Exception as e:
            print(f"  {name}: WRITE FAILED: {e}")
            results[name] = {'writable': False, 'error': str(e)}
            continue

        time.sleep(0.01)
        readback = read_reg(off)

        writable = (readback != orig)
        exact = (readback == test_val)

        status = "WRITABLE" if writable else "read-only"
        if writable and not exact:
            status += f" (partial: got 0x{readback:08X})"

        print(f"  {name} (0x{off:04X}): orig=0x{orig:08X} wrote=0x{test_val:08X} read=0x{readback:08X}  {status}")

        results[name] = {
            'offset': f"0x{off:04X}",
            'original': f"0x{orig:08X}",
            'test_val': f"0x{test_val:08X}",
            'readback': f"0x{readback:08X}",
            'writable': writable,
            'exact_match': exact,
        }

        if writable:
            write_reg(off, orig)
            time.sleep(0.01)
            restored = read_reg(off)
            ok = (restored == orig)
            print(f"    -> restored to 0x{restored:08X} {'OK' if ok else 'MISMATCH!'}")
            results[name]['restored'] = ok

    print()
    print("=== IC_BASE_CNTL bit decode ===")
    for name in ['CP_ME_IC_BASE_CNTL', 'CP_PFP_IC_BASE_CNTL', 'CP_CPC_IC_BASE_CNTL', 'CP_MES_IC_BASE_CNTL']:
        v = vals[name]
        print(f"  {name} = 0x{v:08X} = {v:032b}")
        sz = v & 0x3FF
        print(f"    bits[9:0]  IC_SIZE     = {sz} ({sz*4}KB)")
        print(f"    bit[10]    = {(v>>10)&1}  bit[11] = {(v>>11)&1}  bit[12] = {(v>>12)&1}")
        print(f"    bit[13]    = {(v>>13)&1}  bit[14] = {(v>>14)&1}  bit[15] = {(v>>15)&1}")
        print(f"    bits[21:16]= {(v>>16)&0x3F:06b}  bit[29] = {(v>>29)&1}")
        print()

    # Save results
    out = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'register_values': {n: f"0x{v:08X}" for n, v in vals.items()},
        'write_tests': results,
    }

    outpath = '/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/z2350_ic_write_test.json'
    with open(outpath, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"Results saved to {outpath}")

if __name__ == "__main__":
    main()
