#!/usr/bin/env python3
"""z2352i: Post-reset check — did CPC_IC_BASE_LO survive?
Run this AFTER GPU reset to check register persistence.
"""
import mmap, struct, os, json, time

MMIO_BASE = 0xB4400000
results = {'check_time': time.strftime('%Y-%m-%d %H:%M:%S')}

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

class MMIO:
    def __init__(self):
        self.fd = os.open('/dev/mem', os.O_RDWR | os.O_SYNC)
        self.mm = mmap.mmap(self.fd, 1024*1024, mmap.MAP_SHARED,
                           mmap.PROT_READ | mmap.PROT_WRITE, offset=MMIO_BASE)
    def r(self, off):
        b = off * 4
        return struct.unpack('<I', self.mm[b:b+4])[0]
    def close(self):
        self.mm.close()
        os.close(self.fd)

log("=== z2352i: Post-Reset Register Check ===")
mm = MMIO()

# Check all IC registers
ic_offsets = {
    'PFP_IC_BASE_LO': 0x5840, 'PFP_IC_BASE_HI': 0x5841,
    'PFP_IC_BASE_CNTL': 0x5842, 'PFP_IC_OP_CNTL': 0x5843,
    'ME_IC_BASE_LO': 0x5844, 'ME_IC_BASE_HI': 0x5845,
    'ME_IC_BASE_CNTL': 0x5846, 'ME_IC_OP_CNTL': 0x5847,
    'MEC_IC_BASE_LO': 0x5848, 'MEC_IC_BASE_HI': 0x5849,
    'MEC_IC_BASE_CNTL': 0x584A, 'MEC_IC_OP_CNTL': 0x584B,
    'CPC_IC_BASE_LO': 0x584C, 'CPC_IC_BASE_HI': 0x584D,
    'CPC_IC_BASE_CNTL': 0x584E, 'CPC_IC_OP_CNTL': 0x584F,
    'MES_IC_BASE_LO': 0x5850, 'MES_IC_BASE_HI': 0x5851,
    'MES_IC_BASE_CNTL': 0x5852, 'MES_IC_OP_CNTL': 0x5853,
}

for name, off in ic_offsets.items():
    val = mm.r(off)
    tag = ""
    if name == 'CPC_IC_BASE_LO':
        if val == 0xCAFE0000:
            tag = " *** SURVIVED RESET! ***"
        else:
            tag = f" (was 0xCAFE0000 — CLEARED by reset)"
    log(f"  {name:20s} = 0x{val:08X}{tag}")
    results[name] = f"0x{val:08X}"

results['cpc_survived'] = (mm.r(0x584C) == 0xCAFE0000)

p = '/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/z2352_post_reset_check.json'
with open(p, 'w') as f:
    json.dump(results, f, indent=2)

mm.close()
log(f"\nCPC_IC_BASE_LO survived reset: {results['cpc_survived']}")
log("Done.")
