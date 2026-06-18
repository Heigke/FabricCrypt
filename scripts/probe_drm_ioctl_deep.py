#!/usr/bin/env python3
"""
Deep register probing via DRM ioctl (AMDGPU_INFO_READ_MMR_REG).
This goes through the kernel driver which handles RLCG on APU.
Works as regular user via /dev/dri/renderD128.
"""
import struct, fcntl, os, sys, time

# DRM ioctl constants
DRM_IOCTL_BASE = 0x40
AMDGPU_INFO = 0x05
DRM_COMMAND_BASE = 0x40
# DRM_IOCTL_AMDGPU_INFO = DRM_IOWR(DRM_COMMAND_BASE + AMDGPU_INFO, struct drm_amdgpu_info)
# = _IOWR('d', 0x45, size) where 'd'=0x64
# For 64-byte struct: _IOWR(0x64, 0x45, 64) = 0xC0206445 (read+write, 32 bytes)
# Actually the ioctl number depends on struct size

AMDGPU_INFO_READ_MMR_REG = 0x15

def open_drm():
    for path in ['/dev/dri/renderD128', '/dev/dri/renderD129']:
        try:
            return os.open(path, os.O_RDWR)
        except:
            continue
    raise RuntimeError("Cannot open DRM render node")

def read_mmr_reg(fd, reg_offset):
    """Read a single MMIO register via AMDGPU_INFO ioctl."""
    # struct drm_amdgpu_info:
    #   __u32 return_size   (output buffer size)
    #   __u32 query         (AMDGPU_INFO_READ_MMR_REG = 0x15)
    #   union {
    #     struct { __u32 dword_offset; __u32 count; __u32 instance; __u32 flags; } read_mmr_reg;
    #   }
    # Total struct is 32 bytes minimum

    out_buf = bytearray(4)  # single u32 output

    # Build the info request
    # Layout: return_size(4) + query(4) + dword_offset(4) + count(4) + instance(4) + flags(4)
    dword_off = reg_offset >> 2
    info_data = struct.pack('<IIIIII', 4, AMDGPU_INFO_READ_MMR_REG,
                            dword_off, 1, 0, 0)
    # Pad to 32 bytes
    info_data = info_data.ljust(32, b'\x00')

    # DRM_IOCTL_AMDGPU_INFO
    # _IOWR('d', 0x45, 32) = direction(3)<<30 | size(14)<<16 | type(8)<<8 | nr(8)
    # direction = 3 (read+write), size = 32, type = 0x64 ('d'), nr = 0x45
    ioctl_num = (3 << 30) | (32 << 16) | (0x64 << 8) | 0x45

    # Actually, the ioctl uses pointer to output buffer + pointer to input struct
    # Let me use a simpler approach: ctypes
    import ctypes
    import ctypes.util

    class drm_amdgpu_info(ctypes.Structure):
        _pack_ = 1
        _fields_ = [
            ("return_size", ctypes.c_uint32),
            ("query", ctypes.c_uint32),
            ("dword_offset", ctypes.c_uint32),
            ("count", ctypes.c_uint32),
            ("instance", ctypes.c_uint32),
            ("flags", ctypes.c_uint32),
            ("padding", ctypes.c_uint32 * 2),
        ]

    # Actually the struct layout for DRM_IOCTL_AMDGPU_INFO is:
    # struct drm_amdgpu_info_arg {
    #   __u64 return_pointer;
    #   __u32 return_size;
    #   __u32 query;
    #   ... union with query-specific fields
    # }

    result = ctypes.c_uint32(0)
    result_ptr = ctypes.addressof(result)

    # Build the actual ioctl argument
    # return_pointer(8) + return_size(4) + query(4) + dword_offset(4) + count(4) + instance(4) + flags(4)
    arg = struct.pack('<QIIIIIII',
        result_ptr,           # return_pointer
        4,                    # return_size
        AMDGPU_INFO_READ_MMR_REG,  # query
        dword_off,            # dword_offset
        1,                    # count
        0,                    # instance
        0,                    # flags
        0,                    # padding
    )

    # DRM_IOCTL_AMDGPU_INFO: _IOWR('d', 0x45, sizeof(arg))
    # sizeof = 36 or 40 bytes... let's use the standard size
    arg_size = len(arg)
    ioctl_num = (3 << 30) | (arg_size << 16) | (0x64 << 8) | 0x45

    try:
        fcntl.ioctl(fd, ioctl_num, arg)
        return result.value
    except OSError as e:
        return None

def read_mmr_reg_v2(fd, reg_offset):
    """Alternative: use the libdrm approach via ctypes."""
    import ctypes

    result = (ctypes.c_uint32 * 1)()

    # struct drm_amdgpu_info {
    #   __u64 return_pointer;  // 8
    #   __u32 return_size;     // 4
    #   __u32 query;           // 4
    #   union {                // starts at offset 16
    #     struct { __u32 dword_offset; __u32 count; __u32 instance; __u32 flags; }
    #   }
    # }
    # Total: 32 bytes

    dword_off = reg_offset >> 2

    buf = bytearray(32)
    struct.pack_into('<Q', buf, 0, ctypes.addressof(result))  # return_pointer
    struct.pack_into('<I', buf, 8, 4)                          # return_size
    struct.pack_into('<I', buf, 12, AMDGPU_INFO_READ_MMR_REG) # query
    struct.pack_into('<I', buf, 16, dword_off)                 # dword_offset
    struct.pack_into('<I', buf, 20, 1)                         # count
    struct.pack_into('<I', buf, 24, 0)                         # instance
    struct.pack_into('<I', buf, 28, 0)                         # flags

    # ioctl number: _IOWR('d', 0x45, 32)
    ioctl_num = (3 << 30) | (32 << 16) | (0x64 << 8) | 0x45

    try:
        fcntl.ioctl(fd, ioctl_num, buf)
        return result[0]
    except OSError as e:
        return f"ERR:{e.errno}"

def main():
    fd = open_drm()
    print(f"Opened DRM render node\n")

    # ===== Key GPU registers =====
    regs = [
        (0x8010, "GRBM_STATUS", "per-block busy bits"),
        (0x8014, "GRBM_STATUS2", "more block status"),
        (0x8020, "GRBM_STATUS_SE0", "shader engine 0"),
        (0x8024, "GRBM_STATUS_SE1", "shader engine 1"),
        (0xD048, "SRBM_STATUS", "system bus status"),
        (0xD04C, "SRBM_STATUS2", "system bus status 2"),
        (0xC07C, "RLC_STAT", "RLC status"),
        (0xC080, "RLC_GPU_CLOCK_LSB", "GPU clock low"),
        (0xC084, "RLC_GPU_CLOCK_MSB", "GPU clock high"),
        (0xC10C, "RLC_GPM_STAT", "RLC GPM status"),
        (0x263C, "CP_STAT", "command processor stat"),
    ]

    print("=== DRM ioctl Register Read ===")
    for offset, name, desc in regs:
        val = read_mmr_reg_v2(fd, offset)
        if val is not None and not isinstance(val, str):
            print(f"  0x{offset:04X} {name:25s} = 0x{val:08X} ({val}) [{desc}]")
        else:
            print(f"  0x{offset:04X} {name:25s} = {val} [{desc}]")

    # ===== GPU Clock Counter dynamics =====
    print("\n=== GPU Clock Counter (analog oscillator) ===")
    clocks = []
    for i in range(10):
        lsb = read_mmr_reg_v2(fd, 0xC080)
        msb = read_mmr_reg_v2(fd, 0xC084)
        if isinstance(lsb, int) and isinstance(msb, int):
            clocks.append((msb << 32) | lsb)
        time.sleep(0.002)

    for i, c in enumerate(clocks):
        print(f"  [{i}] = {c}")
    if len(clocks) > 1:
        deltas = [clocks[i+1]-clocks[i] for i in range(len(clocks)-1) if clocks[i+1] > clocks[i]]
        if deltas:
            avg_d = sum(deltas)/len(deltas)
            print(f"  Avg delta/2ms: {avg_d:.0f} ticks → ~{avg_d/0.002/1e6:.1f} MHz effective")

    # ===== GRBM_STATUS dynamics (block activity) =====
    print("\n=== GRBM_STATUS Dynamics (per-block busy) ===")
    statuses = set()
    for _ in range(50):
        val = read_mmr_reg_v2(fd, 0x8010)
        if isinstance(val, int):
            statuses.add(val)
        time.sleep(0.001)
    print(f"  {len(statuses)} unique values in 50 samples")
    for v in sorted(statuses)[:5]:
        # Decode GRBM_STATUS bits
        ta_busy = (v >> 14) & 1
        gds_busy = (v >> 15) & 1
        spi_busy = (v >> 22) & 1
        cb_busy = (v >> 30) & 1
        gui_active = (v >> 31) & 1
        print(f"  0x{v:08X}: GUI_ACTIVE={gui_active} SPI={spi_busy} TA={ta_busy} CB={cb_busy}")

    os.close(fd)
    print("\nDone.")

if __name__ == "__main__":
    main()
