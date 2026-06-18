#!/usr/bin/env python3
"""
Generate gc_11_5_1_toc.bin for RLC_BACKDOOR_AUTO firmware loading.

AMD never shipped a TOC file for gc_11_x_x — only gc_12_x_x has them.
This script creates a synthetic TOC binary that the gfx_v11_0 driver
can parse for RLC_BACKDOOR_AUTO (fw_load_type=3) firmware loading.

The TOC is wrapped in a psp_firmware_header_v1_0 container:
  - 256-byte PSP header
  - N * 16-byte RLC_TABLE_OF_CONTENT entries
  - 8-byte terminator (all zeros)

Each entry: {DW0: offset(25b)|id(7b), DW1: flags(14b)|size(18b),
             DW2: indirect_addr_reg(16b)|index(16b),
             DW3: indirect_data_reg(16b)|indirect_start_offset(16b)}

GFX11 indirect register pairs (from gc_11_0_0_offset.h):
  PFP:  addr=0x5814 data=0x5815  (regCP_HYP_PFP_UCODE_ADDR/DATA)
  ME:   addr=0x5816 data=0x5817  (regCP_HYP_ME_UCODE_ADDR/DATA)
  MEC:  addr=0x581A data=0x581B  (regCP_MEC_ME1_UCODE_ADDR/DATA)
  RLC:  addr=0x5B60 data=0x5B61  (regRLC_GPM_UCODE_ADDR/DATA)
  IRAM: addr=0x5B6A data=0x5B6B  (regRLC_LX6_IRAM_ADDR/DATA)
  DRAM: addr=0x5B68 data=0x5B69  (regRLC_LX6_DRAM_ADDR/DATA)
  SDMA: addr=0x5880 data=0x5881  (regSDMA0_UCODE_ADDR/DATA)
"""

import struct
import os
import sys
import zlib

# ─── SOC21 Firmware IDs ─────────────────────────────────────────────
FW_INVALID        = 0
FW_RLC_G_UCODE    = 1
FW_RLC_TOC        = 2
FW_RLCG_SCRATCH   = 3
FW_RLX6_UCODE     = 7   # IRAM
FW_RLX6_DRAM_BOOT = 9   # DRAM
FW_SDMA_TH0       = 11
FW_SDMA_TH1       = 12
FW_CP_PFP         = 13
FW_CP_ME          = 14
FW_CP_MEC         = 15
FW_RS64_MES_P0    = 16
FW_RS64_MES_P1    = 17
FW_RS64_MES_P0_STACK = 21
FW_RS64_MES_P1_STACK = 22

# ─── Firmware sizes from header parsing (gc_11_5_1_*.bin) ───────────
# These MUST match or exceed what the driver will try to copy.
# Sizes in bytes. The TOC stores them in DWORDs (bytes/4).
FW_SIZES = {
    FW_RLC_G_UCODE:      25088,    # rlc ucode_size_bytes
    FW_RLC_TOC:           512,      # self-referential: entries + mask
    FW_RLCG_SCRATCH:      2048,     # RLC scratch RAM
    FW_RLX6_UCODE:        66048,    # rlc_iram_ucode_size_bytes
    FW_RLX6_DRAM_BOOT:    33280,    # rlc_dram_ucode_size_bytes
    FW_SDMA_TH0:          17408,    # sdma ctx_ucode_size_bytes
    FW_SDMA_TH1:          16896,    # sdma ctl_ucode_size_bytes
    FW_CP_PFP:            263168,   # pfp ucode_size_bytes
    FW_CP_ME:             263168,   # me ucode_size_bytes
    FW_CP_MEC:            267008,   # mec ucode_size_bytes - jt_size*4 (267904 - 896)
    FW_RS64_MES_P0:       104528,   # mes1 ucode
    FW_RS64_MES_P1:       126112,   # mes2 ucode
    FW_RS64_MES_P0_STACK: 131072,   # mes1 data
    FW_RS64_MES_P1_STACK: 131072,   # mes2 data
}

# ─── GFX11 indirect register pairs ─────────────────────────────────
# {id: (indirect_addr_reg, indirect_data_reg)}
# These tell the RLC hardware which registers to use for loading each
# firmware component into the execution engine.
GFX11_REGS = {
    FW_RLC_G_UCODE:      (0x5B60, 0x5B61),  # RLC_GPM_UCODE_ADDR/DATA
    FW_RLC_TOC:           (0x0000, 0x0000),  # no indirect loading
    FW_RLCG_SCRATCH:      (0x0000, 0x0000),  # memory-mapped, no indirect
    FW_RLX6_UCODE:        (0x5B6A, 0x5B6B),  # RLC_LX6_IRAM_ADDR/DATA
    FW_RLX6_DRAM_BOOT:    (0x5B68, 0x5B69),  # RLC_LX6_DRAM_ADDR/DATA
    FW_SDMA_TH0:          (0x5880, 0x5881),  # SDMA0_UCODE_ADDR/DATA
    FW_SDMA_TH1:          (0x5880, 0x5881),  # same SDMA regs, different data
    FW_CP_PFP:            (0x5814, 0x5815),  # CP_HYP_PFP_UCODE_ADDR/DATA
    FW_CP_ME:             (0x5816, 0x5817),  # CP_HYP_ME_UCODE_ADDR/DATA
    FW_CP_MEC:            (0x581A, 0x581B),  # CP_MEC_ME1_UCODE_ADDR/DATA
    FW_RS64_MES_P0:       (0x0000, 0x0000),  # MES loaded via buffer, no indirect
    FW_RS64_MES_P1:       (0x0000, 0x0000),
    FW_RS64_MES_P0_STACK: (0x0000, 0x0000),
    FW_RS64_MES_P1_STACK: (0x0000, 0x0000),
}


def align_up(val, alignment):
    return (val + alignment - 1) & ~(alignment - 1)


def build_toc_entry(fw_id, offset_dw, size_dw, addr_reg=0, data_reg=0,
                    index=0, start_offset=0, flags=0):
    """Build a 16-byte RLC_TABLE_OF_CONTENT entry.

    DW0: offset(25 bits) | id(7 bits)  [little-endian bitfield on x86]
    DW1: flags(14 bits) | size(18 bits)
    DW2: indirect_addr_reg(16) | index(16)
    DW3: indirect_data_reg(16) | indirect_start_offset(16)
    """
    dw0 = (offset_dw & 0x1FFFFFF) | ((fw_id & 0x7F) << 25)
    dw1 = (flags & 0x3FFF) | ((size_dw & 0x3FFFF) << 14)
    dw2 = (addr_reg & 0xFFFF) | ((index & 0xFFFF) << 16)
    dw3 = (data_reg & 0xFFFF) | ((start_offset & 0xFFFF) << 16)
    return struct.pack('<IIII', dw0, dw1, dw2, dw3)


def build_psp_header(total_size, ucode_size, ucode_offset=0x100):
    """Build a psp_firmware_header_v1_0.

    struct common_firmware_header (32 bytes):
      size_bytes, header_size_bytes,
      header_version_major(16), header_version_minor(16),
      ip_version_major(16), ip_version_minor(16),
      ucode_version, ucode_size_bytes,
      ucode_array_offset_bytes, crc32

    struct psp_fw_legacy_bin_desc sos (12 bytes):
      fw_version, offset_bytes, size_bytes
    """
    header = bytearray(ucode_offset)  # zero-padded to ucode_offset

    # common_firmware_header
    struct.pack_into('<I', header, 0x00, total_size)          # size_bytes
    struct.pack_into('<I', header, 0x04, 0x2C)                # header_size_bytes (44)
    struct.pack_into('<HH', header, 0x08, 1, 0)              # header_version 1.0
    struct.pack_into('<HH', header, 0x0C, 11, 5)             # ip_version 11.5
    struct.pack_into('<I', header, 0x10, 1)                   # ucode_version
    struct.pack_into('<I', header, 0x14, ucode_size)          # ucode_size_bytes
    struct.pack_into('<I', header, 0x18, ucode_offset)        # ucode_array_offset_bytes
    # crc32 will be filled after building the full binary

    # psp_fw_legacy_bin_desc sos (at offset 0x20)
    struct.pack_into('<I', header, 0x20, 1)                   # fw_version
    struct.pack_into('<I', header, 0x24, 0)                   # offset_bytes
    struct.pack_into('<I', header, 0x28, 0)                   # size_bytes

    return bytes(header)


def generate_toc():
    """Generate the complete gc_11_5_1_toc.bin."""

    PSP_HEADER_SIZE = 0x100  # 256 bytes
    ALIGN = 0x1000           # 4KB alignment for firmware regions

    # ─── Compute layout ─────────────────────────────────────────────
    # Order entries by firmware ID for clean layout.
    # Each firmware gets 4KB-aligned offset within the autoload buffer.
    entry_ids = sorted(FW_SIZES.keys())

    # Compute offsets (in bytes, then convert to DWORDs for TOC)
    layout = {}
    current_offset = 0

    for fw_id in entry_ids:
        size_bytes = FW_SIZES[fw_id]
        layout[fw_id] = {
            'offset_bytes': current_offset,
            'size_bytes': size_bytes,
            'offset_dw': current_offset // 4,
            'size_dw': align_up(size_bytes, 4) // 4,
        }
        # Next firmware starts at aligned boundary
        current_offset = align_up(current_offset + size_bytes, ALIGN)

    total_buffer_size = current_offset

    # ─── Build TOC entries ───────────────────────────────────────────
    toc_data = bytearray()

    for fw_id in entry_ids:
        info = layout[fw_id]
        addr_reg, data_reg = GFX11_REGS.get(fw_id, (0, 0))

        # DW1 flags: load_at_boot=1 for critical firmware
        flags = 0x0001  # load_at_boot

        entry = build_toc_entry(
            fw_id=fw_id,
            offset_dw=info['offset_dw'],
            size_dw=info['size_dw'],
            addr_reg=addr_reg,
            data_reg=data_reg,
            index=0,
            start_offset=0,
            flags=flags,
        )
        toc_data += entry

    # Terminator entry (all zeros)
    toc_data += b'\x00' * 16

    # ─── Print layout ────────────────────────────────────────────────
    id_names = {
        1: 'RLC_G_UCODE', 2: 'RLC_TOC', 3: 'RLCG_SCRATCH',
        7: 'RLX6_IRAM', 9: 'RLX6_DRAM', 11: 'SDMA_TH0', 12: 'SDMA_TH1',
        13: 'CP_PFP', 14: 'CP_ME', 15: 'CP_MEC',
        16: 'MES_P0', 17: 'MES_P1', 21: 'MES_P0_STACK', 22: 'MES_P1_STACK',
    }

    print("gc_11_5_1_toc.bin layout:")
    print(f"  PSP header:        0x000 - 0x0FF ({PSP_HEADER_SIZE}B)")
    print(f"  TOC entries:       0x100 - 0x{PSP_HEADER_SIZE + len(toc_data) - 1:03X} ({len(toc_data)}B, {len(entry_ids)} entries + terminator)")
    print()
    print(f"  Autoload buffer layout (firmware regions):")
    print(f"  {'ID':>3} {'Name':<16} {'Offset':>10} {'Size':>10}  Regs")
    print(f"  {'---':>3} {'----':<16} {'------':>10} {'----':>10}  ----")
    for fw_id in entry_ids:
        info = layout[fw_id]
        addr_reg, data_reg = GFX11_REGS.get(fw_id, (0, 0))
        name = id_names.get(fw_id, f'FW_{fw_id}')
        regs = f"0x{addr_reg:04X}/0x{data_reg:04X}" if addr_reg else "-"
        print(f"  {fw_id:3d} {name:<16} 0x{info['offset_bytes']:08X} {info['size_bytes']:>10,}B  {regs}")
    print(f"\n  Total autoload buffer: {total_buffer_size:,} bytes ({total_buffer_size/1024:.1f} KB)")

    # ─── Build complete binary ───────────────────────────────────────
    ucode_size = len(toc_data)
    total_file_size = PSP_HEADER_SIZE + ucode_size

    header = build_psp_header(total_file_size, ucode_size, PSP_HEADER_SIZE)
    binary = bytearray(header) + toc_data

    # Compute CRC32 of the ucode data portion
    crc = zlib.crc32(bytes(toc_data)) & 0xFFFFFFFF
    struct.pack_into('<I', binary, 0x1C, crc)

    return bytes(binary)


def main():
    toc_bin = generate_toc()

    # Default output path
    out_path = '/lib/firmware/amdgpu/gc_11_5_1_toc.bin'
    if len(sys.argv) > 1:
        out_path = sys.argv[1]

    print(f"\n  Output: {out_path}")
    print(f"  Total file size: {len(toc_bin)} bytes")

    # Verify by re-parsing
    print("\n  Verification (re-parse):")
    offset = 0x100
    while offset + 16 <= len(toc_bin):
        dw0, dw1, dw2, dw3 = struct.unpack_from('<IIII', toc_bin, offset)
        fw_id = (dw0 >> 25) & 0x7F
        fw_offset = dw0 & 0x1FFFFFF
        fw_size = (dw1 >> 14) & 0x3FFFF
        if fw_id == 0 and fw_offset == 0 and fw_size == 0:
            print(f"    TERMINATOR at offset 0x{offset:03X}")
            break
        print(f"    id={fw_id:2d} offset={fw_offset*4:>10,}B size={fw_size*4:>10,}B")
        offset += 16

    try:
        with open(out_path, 'wb') as f:
            f.write(toc_bin)
        print(f"\n  Written successfully to {out_path}")
    except PermissionError:
        alt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'gc_11_5_1_toc.bin')
        with open(alt_path, 'wb') as f:
            f.write(toc_bin)
        print(f"\n  Permission denied for {out_path}")
        print(f"  Written to {alt_path}")
        print(f"  Install with: sudo cp {alt_path} {out_path}")


if __name__ == '__main__':
    main()
