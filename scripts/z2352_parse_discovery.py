#!/usr/bin/env python3
"""Parse amdgpu IP discovery table to find GC base addresses for GFX 11.5.1"""
import struct, sys

DISCOVERY_PATH = "/sys/kernel/debug/dri/128/amdgpu_discovery"
GC_HWID = 11

with open(DISCOVERY_PATH, "rb") as f:
    blob = f.read()

print(f"Discovery blob: {len(blob)} bytes")

# binary_header: signature(4) + ver_major(2) + ver_minor(2) + checksum(2) + size(2) + table_list(...)
sig = struct.unpack_from("<I", blob, 0)[0]
ver_major, ver_minor = struct.unpack_from("<HH", blob, 4)
print(f"Binary header: sig=0x{sig:08X}, version={ver_major}.{ver_minor}")

# table_list has entries; ip_discovery is first table
# Each table_info is offset(2) + checksum(2) + size(2) = 6 bytes
# binary_header = 12 bytes + table_list
# Let's find the ip_discovery table offset from table_list[0]
ip_disc_off, ip_disc_csum, ip_disc_size = struct.unpack_from("<HHH", blob, 12)
print(f"IP discovery table: offset={ip_disc_off}, size={ip_disc_size}")

# ip_discovery_header at ip_disc_off:
# signature(4) + version(2) + size(2) + id(4) + num_dies(2) + die_info[16](each 4 bytes) + padding(2)
hdr_off = ip_disc_off
ip_sig = struct.unpack_from("<I", blob, hdr_off)[0]
ip_ver, ip_size = struct.unpack_from("<HH", blob, hdr_off + 4)
ip_id = struct.unpack_from("<I", blob, hdr_off + 8)[0]
num_dies = struct.unpack_from("<H", blob, hdr_off + 12)[0]
print(f"IP discovery: sig=0x{ip_sig:08X}, ver={ip_ver}, size={ip_size}, id=0x{ip_id:08X}, num_dies={num_dies}")

# die_info starts at hdr_off + 14, each is 4 bytes (die_id(2) + die_offset(2))
for d in range(min(num_dies, 16)):
    die_id, die_offset = struct.unpack_from("<HH", blob, hdr_off + 14 + d * 4)
    print(f"\nDie {d}: id={die_id}, offset={die_offset} (abs={ip_disc_off + die_offset})")

    # die_header at ip_disc_off + die_offset: die_id(2) + num_ips(2)
    dh_off = ip_disc_off + die_offset
    dh_die_id, dh_num_ips = struct.unpack_from("<HH", blob, dh_off)
    print(f"  die_header: die_id={dh_die_id}, num_ips={dh_num_ips}")

    # IP entries follow die_header (4 bytes after)
    pos = dh_off + 4
    for i in range(dh_num_ips):
        if pos + 8 > len(blob):
            print(f"  IP {i}: out of bounds at pos={pos}")
            break

        # ip_v3 struct: hw_id(2) + instance(1) + num_base(1) + major(1) + minor(1) + revision(1) + sub_rev_variant(1)
        # = 8 bytes header, then num_base * 4 bytes of base addresses
        hw_id, inst, nbase, major, minor, rev, sub_var = struct.unpack_from("<HBBBBB B", blob, pos)
        sub_rev = sub_var & 0x0F
        variant = (sub_var >> 4) & 0x0F

        bases = []
        for k in range(nbase):
            if pos + 8 + k * 4 <= len(blob) - 4:
                ba = struct.unpack_from("<I", blob, pos + 8 + k * 4)[0]
                bases.append(ba)

        ip_size_bytes = 8 + nbase * 4

        if hw_id == GC_HWID or nbase > 0:
            label = " *** GC ***" if hw_id == GC_HWID else ""
            print(f"  IP {i}: hw_id={hw_id}{label}, inst={inst}, v{major}.{minor}.{rev}.{sub_rev}, "
                  f"variant={variant}, nbase={nbase}, bases={['0x%08X' % b for b in bases]}")

        pos += ip_size_bytes

# Also check base_addr_64_bit flag for v4
if ip_ver >= 4:
    flag_off = hdr_off + 14 + 16 * 4  # after die_info[16]
    flag_byte = blob[flag_off]
    print(f"\nv4 base_addr_64_bit flag: {flag_byte & 1}")
