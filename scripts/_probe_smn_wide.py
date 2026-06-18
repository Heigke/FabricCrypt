#!/usr/bin/env python3
"""Wide SMN scan 0x00000-0x100000 for dynamic registers."""
import struct, os, mmap, time

fd = os.open('/dev/mem', os.O_RDWR | os.O_SYNC)
mm = mmap.mmap(fd, 4096, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, offset=0xE0000000)

def smn_read(addr):
    struct.pack_into('<I', mm, 0x60, addr)
    return struct.unpack_from('<I', mm, 0x64)[0]

dynamic_blocks = []
live_blocks = []

for block_start in range(0x00000, 0x100000, 0x1000):
    n_live = 0
    n_dynamic = 0
    examples = []
    vals1 = {}
    for off in range(0, 0x1000, 0x40):
        addr = block_start + off
        v = smn_read(addr)
        if v != 0 and v != 0xFFFFFFFF:
            vals1[addr] = v
            n_live += 1
    if n_live == 0:
        continue
    time.sleep(0.01)
    for addr, v1 in vals1.items():
        v2 = smn_read(addr)
        if v2 != v1:
            n_dynamic += 1
            examples.append((addr, v1, v2))
    live_blocks.append((block_start, n_live))
    if n_dynamic > 0:
        dynamic_blocks.append((block_start, n_live, n_dynamic, examples[:3]))

print(f'=== WIDE SMN SCAN: 0x00000-0xFFFFF (step 0x40) ===')
print(f'  Total live blocks: {len(live_blocks)}')
print(f'  Dynamic blocks: {len(dynamic_blocks)}')
print(f'\n=== DYNAMIC BLOCKS ===')
for block, n_live, n_dyn, examples in sorted(dynamic_blocks, key=lambda x: -x[2]):
    print(f'  0x{block:05X}: {n_live} live, {n_dyn} dynamic')
    for addr, v1, v2 in examples:
        print(f'    0x{addr:05X}: 0x{v1:08X} -> 0x{v2:08X} (d={v2-v1})')
print(f'\n=== TOP LIVE BLOCKS (>10 regs) ===')
for block, n_live in sorted(live_blocks, key=lambda x: -x[1])[:20]:
    print(f'  0x{block:05X}: {n_live} live registers')

mm.close()
os.close(fd)
