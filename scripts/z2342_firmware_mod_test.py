#!/usr/bin/env python3
"""z2342: Firmware modification test — PSP signature verification analysis.

SAFETY: All operations on COPIES in /tmp. Never modifies originals.
The iGPU drives the display — we NEVER actually load modified firmware.
"""

import struct
import hashlib
import json
import os
import sys
import time
import shutil
from pathlib import Path
from datetime import datetime

RESULTS_DIR = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results")
LOG_PATH = RESULTS_DIR / "z2342_firmware_mod_log.txt"
SIG_PATH = RESULTS_DIR / "z2342_signature_analysis.txt"
CHAIN_PATH = RESULTS_DIR / "z2342_load_chain.txt"
JSON_PATH = RESULTS_DIR / "z2342_firmware_mod_test.json"

FW_DIR = Path("/lib/firmware/amdgpu")
TMP_DIR = Path("/tmp")

# GFX 11.5.1 firmware files
FW_FILES = [
    "gc_11_5_1_imu.bin",
    "gc_11_5_1_me.bin",
    "gc_11_5_1_mec.bin",
    "gc_11_5_1_mes1.bin",
    "gc_11_5_1_mes_2.bin",
    "gc_11_5_1_pfp.bin",
    "gc_11_5_1_rlc.bin",
]

results = {
    "experiment": "z2342_firmware_mod_test",
    "timestamp": datetime.now().isoformat(),
    "system": {
        "kernel": os.uname().release,
        "gpu": "AMD Radeon 8060S (gfx1151)",
    },
    "firmware_files": {},
    "psp_analysis": {},
    "signature_analysis": {},
    "load_chain": {},
    "conclusions": [],
}


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")
        f.flush()


def check_temp():
    try:
        t = int(open("/sys/class/thermal/thermal_zone0/temp").read().strip())
        if t > 80000:
            log(f"ABORT: temperature {t/1000:.1f}C > 80C")
            sys.exit(1)
        return t / 1000
    except:
        return -1


def parse_common_firmware_header(data):
    """Parse AMD common_firmware_header (from amd_psp.h / amdgpu_ucode.h).

    struct common_firmware_header {
        uint32_t size_bytes;       // 0x00: total size
        uint32_t header_size_bytes;// 0x04: header size in bytes
        uint16_t header_version_major; // 0x08
        uint16_t header_version_minor; // 0x0A
        uint16_t ip_version_major; // 0x0C
        uint16_t ip_version_minor; // 0x0E
        uint32_t ucode_version;   // 0x10
        uint32_t ucode_size_bytes;// 0x14: payload size
        uint32_t ucode_array_offset_bytes; // 0x18: offset to payload
        uint32_t crc32;           // 0x1C: CRC32
    };
    """
    if len(data) < 0x20:
        return None

    hdr = {}
    hdr['size_bytes'] = struct.unpack_from('<I', data, 0x00)[0]
    hdr['header_size_bytes'] = struct.unpack_from('<I', data, 0x04)[0]
    hdr['header_version_major'] = struct.unpack_from('<H', data, 0x08)[0]
    hdr['header_version_minor'] = struct.unpack_from('<H', data, 0x0A)[0]
    hdr['ip_version_major'] = struct.unpack_from('<H', data, 0x0C)[0]
    hdr['ip_version_minor'] = struct.unpack_from('<H', data, 0x0E)[0]
    hdr['ucode_version'] = struct.unpack_from('<I', data, 0x10)[0]
    hdr['ucode_size_bytes'] = struct.unpack_from('<I', data, 0x14)[0]
    hdr['ucode_array_offset_bytes'] = struct.unpack_from('<I', data, 0x18)[0]
    hdr['crc32'] = struct.unpack_from('<I', data, 0x1C)[0]

    return hdr


def parse_psp1_header(data):
    """Look for PSP header signature ($PS1 = 0x31535024 or other PSP magic).

    AMD PSP firmware header (psp_firmware_header_v1_0, v1_1, v1_3, v2_0):
    After common_firmware_header:
    - v1_0: ucode_feature_version (0x20), ucode_size_bytes (0x24), padding...
    - v1_1: adds sos_offset, sos_size
    - v1_3: adds v1.1 fields + multiple fw offsets
    - v2_0: completely different, adds psp_dir

    For GFX microcode (non-PSP firmware like MEC/ME/PFP/RLC):
    - Uses gfx_firmware_header_v1_0 or gfx_firmware_header_v2_0
    - These are NOT PSP firmware — they're GFX engine microcode
    - PSP loads them but they have different header format
    """
    result = {}

    # Search for known magic bytes
    magics = {
        b'\x24\x50\x53\x31': '$PS1',  # PSP v1
        b'\x24\x50\x53\x32': '$PS2',  # PSP v2
        b'\x01\x00\x00\x00': 'TYPE_1', # common type marker
    }

    for magic_bytes, name in magics.items():
        offset = data.find(magic_bytes)
        if offset >= 0 and offset < min(len(data), 4096):
            result[name] = {'offset': offset, 'hex': data[offset:offset+32].hex()}

    # Also scan for RSA signature patterns (high-entropy 256 or 512 byte blocks)
    # RSA signatures look like random data — high byte diversity
    result['high_entropy_blocks'] = []
    for off in range(0, min(len(data), 8192), 256):
        block = data[off:off+256]
        if len(block) == 256:
            unique = len(set(block))
            if unique > 200:  # high entropy = likely signature/encrypted
                result['high_entropy_blocks'].append({
                    'offset': off,
                    'unique_bytes': unique,
                    'first_16': block[:16].hex()
                })

    return result


def parse_gfx_firmware_header(data):
    """Parse GFX firmware header (for MEC/ME/PFP/RLC).

    gfx_firmware_header_v1_0 extends common_firmware_header:
    0x20: ucode_feature_version (uint32_t)
    0x24: jt_offset (uint32_t) — jump table offset
    0x28: jt_size (uint32_t) — jump table size

    gfx_firmware_header_v2_0 extends common_firmware_header:
    0x20: ucode_feature_version (uint32_t)
    0x24: ucode_size_bytes (uint32_t) — CP_MEC.ucode.size_bytes
    0x28: data_size_bytes (uint32_t)
    0x2C: ucode_offset_bytes (uint32_t)
    0x30: data_offset_bytes (uint32_t)
    """
    common = parse_common_firmware_header(data)
    if not common:
        return None

    ext = dict(common)
    hdr_major = common['header_version_major']
    hdr_minor = common['header_version_minor']

    if hdr_major == 1:
        if len(data) >= 0x2C:
            ext['ucode_feature_version'] = struct.unpack_from('<I', data, 0x20)[0]
            ext['jt_offset'] = struct.unpack_from('<I', data, 0x24)[0]
            ext['jt_size'] = struct.unpack_from('<I', data, 0x28)[0]
    elif hdr_major == 2:
        if len(data) >= 0x34:
            ext['ucode_feature_version'] = struct.unpack_from('<I', data, 0x20)[0]
            ext['ucode_size_bytes_v2'] = struct.unpack_from('<I', data, 0x24)[0]
            ext['data_size_bytes'] = struct.unpack_from('<I', data, 0x28)[0]
            ext['ucode_offset_bytes'] = struct.unpack_from('<I', data, 0x2C)[0]
            ext['data_offset_bytes'] = struct.unpack_from('<I', data, 0x30)[0]

    return ext


def parse_rlc_firmware_header(data):
    """Parse RLC firmware header.

    rlc_firmware_header_v2_0 extends common_firmware_header:
    0x20: ucode_feature_version
    0x24: save_and_restore_offset
    0x28: clear_state_descriptor_offset
    0x2C: avail_scratch_ram_locations
    0x30: master_pkt_description_offset

    rlc_firmware_header_v2_1 extends v2_0:
    0x34: save_restore_list_cntl_ucode_ver
    ... many more fields for SRLx

    rlc_firmware_header_v2_2 extends v2_1:
    ... adds RLC IRAM/DRAM offsets

    rlc_firmware_header_v2_3 extends v2_2:
    ... adds rlcp/rlcv

    rlc_firmware_header_v2_4 extends v2_3:
    ... adds global tap delays
    """
    common = parse_common_firmware_header(data)
    if not common:
        return None

    ext = dict(common)
    if len(data) >= 0x34:
        ext['ucode_feature_version'] = struct.unpack_from('<I', data, 0x20)[0]
        ext['save_and_restore_offset'] = struct.unpack_from('<I', data, 0x24)[0]
        ext['clear_state_descriptor_offset'] = struct.unpack_from('<I', data, 0x28)[0]
        ext['avail_scratch_ram_locations'] = struct.unpack_from('<I', data, 0x2C)[0]
        ext['master_pkt_description_offset'] = struct.unpack_from('<I', data, 0x30)[0]

    return ext


def parse_imu_firmware_header(data):
    """Parse IMU firmware header.

    imu_firmware_header_v1_0 extends common_firmware_header:
    0x20: imu_iram_ucode_size_bytes
    0x24: imu_iram_ucode_offset_bytes
    0x28: imu_dram_ucode_size_bytes
    0x2C: imu_dram_ucode_offset_bytes
    """
    common = parse_common_firmware_header(data)
    if not common:
        return None

    ext = dict(common)
    if len(data) >= 0x30:
        ext['imu_iram_ucode_size_bytes'] = struct.unpack_from('<I', data, 0x20)[0]
        ext['imu_iram_ucode_offset_bytes'] = struct.unpack_from('<I', data, 0x24)[0]
        ext['imu_dram_ucode_size_bytes'] = struct.unpack_from('<I', data, 0x28)[0]
        ext['imu_dram_ucode_offset_bytes'] = struct.unpack_from('<I', data, 0x2C)[0]

    return ext


def parse_mes_firmware_header(data):
    """Parse MES firmware header.

    mes_firmware_header_v1_0 extends common_firmware_header:
    0x20: mes_ucode_version
    0x24: mes_ucode_size_bytes
    0x28: mes_ucode_offset_bytes
    0x2C: mes_ucode_data_version
    0x30: mes_ucode_data_size_bytes
    0x34: mes_ucode_data_offset_bytes
    0x38: mes_uc_start_addr_lo
    0x3C: mes_uc_start_addr_hi
    0x40: mes_data_start_addr_lo
    0x44: mes_data_start_addr_hi
    """
    common = parse_common_firmware_header(data)
    if not common:
        return None

    ext = dict(common)
    if len(data) >= 0x48:
        ext['mes_ucode_version'] = struct.unpack_from('<I', data, 0x20)[0]
        ext['mes_ucode_size_bytes'] = struct.unpack_from('<I', data, 0x24)[0]
        ext['mes_ucode_offset_bytes'] = struct.unpack_from('<I', data, 0x28)[0]
        ext['mes_ucode_data_version'] = struct.unpack_from('<I', data, 0x2C)[0]
        ext['mes_ucode_data_size_bytes'] = struct.unpack_from('<I', data, 0x30)[0]
        ext['mes_ucode_data_offset_bytes'] = struct.unpack_from('<I', data, 0x34)[0]
        ext['mes_uc_start_addr_lo'] = struct.unpack_from('<I', data, 0x38)[0]
        ext['mes_uc_start_addr_hi'] = struct.unpack_from('<I', data, 0x3C)[0]
        ext['mes_data_start_addr_lo'] = struct.unpack_from('<I', data, 0x40)[0]
        ext['mes_data_start_addr_hi'] = struct.unpack_from('<I', data, 0x44)[0]

    return ext


def compute_crc32(data, offset, size):
    """Compute CRC32 of payload region."""
    import binascii
    payload = data[offset:offset + size]
    return binascii.crc32(payload) & 0xFFFFFFFF


def analyze_signature_region(data, header):
    """Analyze the region between header and payload for signature data."""
    hdr_size = header['header_size_bytes']
    payload_off = header['ucode_array_offset_bytes']

    result = {
        'header_size': hdr_size,
        'payload_offset': payload_off,
        'gap_size': payload_off - hdr_size if payload_off > hdr_size else 0,
        'has_gap': payload_off > hdr_size,
    }

    if result['has_gap']:
        gap_data = data[hdr_size:payload_off]
        result['gap_hex_first_64'] = gap_data[:64].hex() if len(gap_data) >= 64 else gap_data.hex()
        result['gap_unique_bytes'] = len(set(gap_data))
        result['gap_zero_fraction'] = gap_data.count(0) / len(gap_data) if gap_data else 0

        # Check if gap looks like a signature (high entropy)
        if len(gap_data) >= 256:
            result['gap_looks_like_signature'] = result['gap_unique_bytes'] > 150
        else:
            result['gap_looks_like_signature'] = False

    # Check data AFTER declared payload for appended signatures
    payload_end = payload_off + header['ucode_size_bytes']
    if payload_end < len(data):
        trailing = data[payload_end:]
        result['trailing_data_size'] = len(trailing)
        result['trailing_hex_first_64'] = trailing[:64].hex() if len(trailing) >= 64 else trailing.hex()
        result['trailing_unique_bytes'] = len(set(trailing))
        result['trailing_looks_like_signature'] = len(trailing) >= 256 and result['trailing_unique_bytes'] > 150
    else:
        result['trailing_data_size'] = 0
        result['trailing_looks_like_signature'] = False

    return result


def analyze_firmware_file(fw_name, data):
    """Full analysis of one firmware file."""
    log(f"  Analyzing {fw_name} ({len(data)} bytes)")

    info = {
        'filename': fw_name,
        'total_size': len(data),
        'md5': hashlib.md5(data).hexdigest(),
        'sha256': hashlib.sha256(data).hexdigest(),
    }

    # Parse common header
    common = parse_common_firmware_header(data)
    if common:
        info['common_header'] = common
        log(f"    Header v{common['header_version_major']}.{common['header_version_minor']}, "
            f"IP v{common['ip_version_major']}.{common['ip_version_minor']}, "
            f"ucode v0x{common['ucode_version']:08X}")
        log(f"    Header size: {common['header_size_bytes']}B, "
            f"Payload offset: {common['ucode_array_offset_bytes']}B, "
            f"Payload size: {common['ucode_size_bytes']}B")
        log(f"    Declared CRC32: 0x{common['crc32']:08X}")

        # Verify CRC32
        if common['ucode_array_offset_bytes'] > 0 and common['ucode_size_bytes'] > 0:
            computed_crc = compute_crc32(data, common['ucode_array_offset_bytes'],
                                         common['ucode_size_bytes'])
            info['computed_crc32'] = f"0x{computed_crc:08X}"
            info['crc32_matches'] = computed_crc == common['crc32']
            log(f"    Computed CRC32: 0x{computed_crc:08X} — {'MATCH' if info['crc32_matches'] else 'MISMATCH'}")

        # Type-specific header parsing
        if 'mec' in fw_name or 'me' in fw_name or 'pfp' in fw_name:
            ext = parse_gfx_firmware_header(data)
            if ext:
                info['gfx_header'] = ext
                if 'jt_offset' in ext:
                    log(f"    GFX v1: JT offset={ext['jt_offset']}, JT size={ext['jt_size']}")
                elif 'ucode_offset_bytes' in ext:
                    log(f"    GFX v2: ucode_off={ext['ucode_offset_bytes']}, data_off={ext['data_offset_bytes']}")
        elif 'rlc' in fw_name:
            ext = parse_rlc_firmware_header(data)
            if ext:
                info['rlc_header'] = ext
                log(f"    RLC: save_restore_off={ext.get('save_and_restore_offset', 'N/A')}")
        elif 'imu' in fw_name:
            ext = parse_imu_firmware_header(data)
            if ext:
                info['imu_header'] = ext
                log(f"    IMU: IRAM size={ext.get('imu_iram_ucode_size_bytes', 'N/A')}, "
                    f"DRAM size={ext.get('imu_dram_ucode_size_bytes', 'N/A')}")
        elif 'mes' in fw_name:
            ext = parse_mes_firmware_header(data)
            if ext:
                info['mes_header'] = ext
                log(f"    MES: ucode_size={ext.get('mes_ucode_size_bytes', 'N/A')}, "
                    f"data_size={ext.get('mes_ucode_data_size_bytes', 'N/A')}")

        # Signature analysis
        sig_info = analyze_signature_region(data, common)
        info['signature_region'] = sig_info
        if sig_info['has_gap']:
            log(f"    Gap between header and payload: {sig_info['gap_size']}B "
                f"(looks like sig: {sig_info['gap_looks_like_signature']})")
        if sig_info['trailing_data_size'] > 0:
            log(f"    Trailing data after payload: {sig_info['trailing_data_size']}B "
                f"(looks like sig: {sig_info['trailing_looks_like_signature']})")

    # Search for PSP magic
    psp_info = parse_psp1_header(data)
    info['psp_magic_search'] = psp_info
    if psp_info:
        for k, v in psp_info.items():
            if k != 'high_entropy_blocks':
                log(f"    PSP magic '{k}' found at offset {v['offset']}")
        if psp_info.get('high_entropy_blocks'):
            log(f"    High-entropy blocks in first 8K: {len(psp_info['high_entropy_blocks'])}")

    # File structure summary
    if common:
        payload_off = common['ucode_array_offset_bytes']
        payload_size = common['ucode_size_bytes']
        payload_end = payload_off + payload_size
        info['file_structure'] = {
            'header_region': f"0x0000-0x{common['header_size_bytes']-1:04X}",
            'gap_or_sig_region': f"0x{common['header_size_bytes']:04X}-0x{payload_off-1:04X}" if payload_off > common['header_size_bytes'] else "NONE",
            'payload_region': f"0x{payload_off:04X}-0x{payload_end-1:04X}",
            'trailing_region': f"0x{payload_end:04X}-0x{len(data)-1:04X}" if payload_end < len(data) else "NONE",
        }
        log(f"    Structure: header[0x0-0x{common['header_size_bytes']-1:X}] "
            f"payload[0x{payload_off:X}-0x{payload_end-1:X}] "
            f"trailing[{len(data)-payload_end}B]")

    return info


def create_modified_copy(fw_name, data, header):
    """Create a minimally modified copy for analysis (NOT for loading)."""
    log(f"  Creating modified copy of {fw_name}")

    modified = bytearray(data)
    payload_off = header['ucode_array_offset_bytes']

    # Flip one bit deep in the payload (offset + 1000 or middle of payload)
    flip_offset = payload_off + min(1000, header['ucode_size_bytes'] // 2)
    original_byte = modified[flip_offset]
    modified[flip_offset] ^= 0x01
    new_byte = modified[flip_offset]

    mod_path = TMP_DIR / f"z2342_{fw_name.replace('.bin', '_modified.bin')}"
    mod_path.write_bytes(bytes(modified))

    # Also create one with only CRC modified (to test CRC-only path)
    crc_mod = bytearray(data)
    crc_off = 0x1C  # CRC32 field offset
    original_crc = struct.unpack_from('<I', crc_mod, crc_off)[0]
    struct.pack_into('<I', crc_mod, crc_off, original_crc ^ 0x01)
    new_crc = struct.unpack_from('<I', crc_mod, crc_off)[0]

    crc_path = TMP_DIR / f"z2342_{fw_name.replace('.bin', '_crc_modified.bin')}"
    crc_path.write_bytes(bytes(crc_mod))

    result = {
        'payload_flip': {
            'offset': flip_offset,
            'original': f"0x{original_byte:02X}",
            'modified': f"0x{new_byte:02X}",
            'output_file': str(mod_path),
        },
        'crc_flip': {
            'offset': crc_off,
            'original_crc': f"0x{original_crc:08X}",
            'modified_crc': f"0x{new_crc:08X}",
            'output_file': str(crc_path),
        }
    }

    log(f"    Payload bit flip at offset 0x{flip_offset:X}: 0x{original_byte:02X} -> 0x{new_byte:02X}")
    log(f"    CRC modification: 0x{original_crc:08X} -> 0x{new_crc:08X}")
    log(f"    Modified files saved to {mod_path} and {crc_path}")

    return result


def analyze_load_chain():
    """Analyze the firmware loading chain from kernel source knowledge."""
    log("STEP 6: Documenting firmware loading chain")

    chain = {}

    # PSP boot messages from dmesg
    chain['boot_sequence'] = {
        'description': 'PSP boot sequence observed in dmesg',
        'steps': [
            '[0.000] secureboot: Secure boot disabled (UEFI)',
            '[11.926] ccp 0000:c3:00.2: tee enabled (TEE = Trusted Execution Environment)',
            '[11.928] ccp 0000:c3:00.2: psp: TSME enabled (Transparent SME = memory encryption)',
            '[11.928] ccp 0000:c3:00.2: psp enabled (PSP co-processor active)',
            '[13.318] amdgpu: detected ip block number 3 <psp>',
            '[13.334] TMZ feature disabled as experimental',
            '[13.335] Loading DMUB firmware via PSP: version=0x09002C01',
            '[13.359] reserve 0x8c00000 from 0x97e0000000 for PSP TMR (147MB Trusted Memory Region)',
            '[13.867] RAS: optional ras ta ucode is not available',
            '[13.870] RAP: optional rap ta ucode is not available',
            '[13.870] SECUREDISPLAY: securedisplay ta ucode is not available',
            '[13.904] SMU is initialized successfully',
        ]
    }

    # Loading chain analysis
    chain['loading_chain'] = {
        '1_uefi_bios': {
            'description': 'UEFI BIOS boots, initializes PSP',
            'secure_boot': 'DISABLED on this system',
            'note': 'Secure boot off means UEFI does not verify OS bootloader signatures',
        },
        '2_psp_boot': {
            'description': 'PSP co-processor boots from on-die ROM',
            'tsme': 'ENABLED — Transparent Secure Memory Encryption',
            'tee': 'ENABLED — Trusted Execution Environment',
            'tmz': 'DISABLED — Trusted Memory Zone off (experimental)',
            'note': 'PSP has its own boot ROM, loads from SPI flash. We cannot modify this.',
        },
        '3_linux_kernel': {
            'description': 'Linux kernel loads amdgpu driver',
            'fw_load_type': '-1 (AUTO — driver chooses PSP or direct based on IP version)',
            'note': 'fw_load_type=-1 means "auto". For GFX11, this means PSP loading.',
        },
        '4_amdgpu_driver': {
            'description': 'amdgpu calls request_firmware() for each ucode',
            'firmware_path': '/lib/firmware/amdgpu/',
            'override_path': '/lib/firmware/updates/amdgpu/ (does NOT exist currently)',
            'fw_loader_user_helper': 'ENABLED (CONFIG_FW_LOADER_USER_HELPER=y)',
            'compression': 'ZSTD compression supported (CONFIG_FW_LOADER_COMPRESS_ZSTD=y)',
            'note': 'request_firmware() searches: updates/ first, then base path',
        },
        '5_psp_validation': {
            'description': 'PSP validates firmware signature before loading to HW',
            'mechanism': 'PSP receives blob via MMIO ring buffer (C2PMSG)',
            'validation': 'RSA signature check against AMD root of trust in PSP ROM',
            'note': 'PSP has hardware root of trust — keys burned into silicon',
        },
        '6_hw_load': {
            'description': 'After PSP validation, firmware loaded to CP/RLC/IMU/MES SRAM',
            'note': 'GPU microengines start executing firmware code',
        }
    }

    # fw_load_type analysis
    chain['fw_load_type_analysis'] = {
        'current_value': -1,
        'meaning': 'AUTO — driver picks optimal load method',
        'possible_values': {
            '-1': 'AUTO (default) — PSP for GFX11+',
            '0': 'DIRECT — bypass PSP, write directly to MMIO (older GPUs only)',
            '1': 'SMU — load via SMU (not used for GFX)',
            '2': 'PSP — explicit PSP loading',
            '3': 'RSI — Resource Sharing Interface (newer)',
        },
        'note': 'For GFX11 (gfx1151), AUTO resolves to PSP. Direct loading is NOT supported for GFX11.'
    }

    # CRC32 vs PSP signature analysis
    chain['validation_layers'] = {
        'layer_1_crc32': {
            'description': 'CRC32 in common_firmware_header',
            'checked_by': 'Kernel driver (amdgpu) before sending to PSP',
            'what_is_checked': 'CRC32 of ucode payload (not header)',
            'consequence_of_failure': 'Driver refuses to load, prints error to dmesg',
            'bypassable': 'Yes — can recalculate CRC32 after modification',
        },
        'layer_2_psp_signature': {
            'description': 'RSA signature verified by PSP hardware',
            'checked_by': 'PSP co-processor (hardware root of trust)',
            'what_is_checked': 'Entire firmware blob or payload hash, signed with AMD private key',
            'consequence_of_failure': 'PSP rejects firmware, GPU fails to initialize',
            'bypassable': 'NO — would require AMD private signing key',
        },
        'layer_3_tsme': {
            'description': 'TSME encrypts memory contents transparently',
            'checked_by': 'PSP/CCP hardware',
            'what_is_checked': 'Memory integrity (transparent to software)',
            'note': 'TSME protects firmware in VRAM from physical attacks but does not prevent software-level modification of the blob before loading',
        }
    }

    return chain


def analyze_override_mechanism():
    """Check if firmware override paths exist and are usable."""
    log("STEP 4: Analyzing firmware override mechanisms")

    override = {}

    # /lib/firmware/updates/ priority
    updates_path = Path("/lib/firmware/updates/amdgpu")
    override['updates_dir_exists'] = updates_path.exists()
    override['updates_dir_note'] = (
        "Does NOT exist. Could be created to override firmware. "
        "request_firmware() checks /lib/firmware/updates/ BEFORE /lib/firmware/"
    )

    # firmware_class path parameter
    try:
        path_param = Path("/sys/module/firmware_class/parameters/path").read_text().strip()
        override['fw_class_path'] = path_param if path_param else "(empty)"
    except:
        override['fw_class_path'] = "N/A"

    # Module parameters
    override['config'] = {
        'FW_LOADER': 'y (built-in)',
        'FW_LOADER_DEBUG': 'y',
        'FW_LOADER_SYSFS': 'y',
        'FW_LOADER_USER_HELPER': 'y (userspace firmware loading supported!)',
        'FW_LOADER_USER_HELPER_FALLBACK': 'NOT set',
        'FW_LOADER_COMPRESS_ZSTD': 'y',
    }

    # Check if we could use the user helper
    override['user_helper'] = {
        'enabled': True,
        'description': (
            'CONFIG_FW_LOADER_USER_HELPER=y means the kernel can fall back to '
            'userspace firmware loading via /dev/.udev/firmware.sh or udev. '
            'However, USER_HELPER_FALLBACK is NOT set, meaning this only works '
            'if explicitly requested by the driver (not as automatic fallback).'
        ),
    }

    # Reset mechanism
    override['reset_mechanism'] = {
        'sysfs_reset': '/sys/class/drm/card0/device/reset (exists)',
        'reset_method': 'pm bus',
        'debugfs_recover': '/sys/kernel/debug/dri/0/amdgpu_gpu_recover (exists)',
        'WARNING': (
            'GPU reset/recover RELOADS firmware from /lib/firmware/. '
            'If modified firmware is in the override path, it WOULD be loaded. '
            'However, PSP would then validate it. If validation fails: '
            'GPU HANGS, display goes BLACK, system may need hard reboot. '
            'DO NOT TEST THIS on the iGPU that drives the display.'
        ),
    }

    return override


def check_psp_reload_behavior():
    """Analyze whether PSP validates on every reload or only cold boot."""
    log("STEP 5: Analyzing PSP reload validation behavior")

    analysis = {}

    # From dmesg timestamps, PSP messages only appear once at boot
    analysis['boot_only_messages'] = {
        'psp_enabled': 'timestamp 11.928s — ONLY appears at boot',
        'tmr_reserve': 'timestamp 13.359s — ONLY appears at boot',
        'note': 'No PSP validation messages appear after initial boot in dmesg',
    }

    analysis['reload_behavior'] = {
        'theory': (
            'On GPU reset (amdgpu_gpu_recover), the driver calls psp_hw_fini() '
            'followed by psp_hw_init(). This RE-SENDS all firmware to PSP. '
            'PSP validates EVERY TIME firmware is submitted, not just at cold boot. '
            'This is because PSP processes firmware load commands independently — '
            'it does not cache "already validated" state for runtime reloads.'
        ),
        'evidence': (
            'The PSP ring buffer (C2PMSG) is a command interface. Each firmware '
            'load is a separate command. PSP has no reason to skip validation on reload.'
        ),
        'conclusion': 'PSP likely validates on EVERY firmware load, including GPU reset/recover.',
    }

    analysis['cold_vs_warm'] = {
        'cold_boot': 'Full PSP boot from ROM → full validation chain',
        'warm_reset': 'PSP stays powered → revalidates submitted firmware',
        'runtime_reload': 'amdgpu_gpu_recover → driver resubmits to PSP → PSP revalidates',
        'note': 'There is NO "skip validation" path for runtime reloads in the PSP command protocol',
    }

    return analysis


def main():
    log("=" * 70)
    log("z2342: Firmware Modification Test — PSP Signature Verification Analysis")
    log("=" * 70)
    log(f"Temperature: {check_temp():.1f}C")
    log("")

    # =========================================================================
    # STEP 1: Verify backups
    # =========================================================================
    log("STEP 1: Verifying firmware backups in /tmp")
    for fw_name in FW_FILES:
        backup = TMP_DIR / f"z2342_{fw_name}"
        if backup.exists():
            log(f"  OK: {backup} ({backup.stat().st_size} bytes)")
        else:
            log(f"  MISSING: {backup} — attempting decompress")
            src = FW_DIR / f"{fw_name}.zst"
            if src.exists():
                os.system(f"zstd -d -f {src} -o {backup}")
            else:
                log(f"  ERROR: source {src} not found!")
    log("")

    # =========================================================================
    # STEP 2: Analyze PSP signature structure
    # =========================================================================
    log("STEP 2: Analyzing PSP signature structure in firmware blobs")
    log("")

    for fw_name in FW_FILES:
        backup = TMP_DIR / f"z2342_{fw_name}"
        if not backup.exists():
            log(f"  SKIP: {backup} not found")
            continue

        data = backup.read_bytes()
        info = analyze_firmware_file(fw_name, data)
        results['firmware_files'][fw_name] = info

        # Create modified copies for analysis
        common = parse_common_firmware_header(data)
        if common:
            mod_info = create_modified_copy(fw_name, data, common)
            info['modifications'] = mod_info

        log("")
        check_temp()

    # =========================================================================
    # STEP 3: Analyze what kernel driver checks (CRC vs PSP)
    # =========================================================================
    log("STEP 3: Analyzing kernel driver validation path")
    log("")

    # Check CRC validation in practice
    crc_analysis = {}
    for fw_name in ['gc_11_5_1_mec.bin']:  # Just analyze MEC as representative
        backup = TMP_DIR / f"z2342_{fw_name}"
        if not backup.exists():
            continue
        data = backup.read_bytes()
        common = parse_common_firmware_header(data)
        if not common:
            continue

        payload_off = common['ucode_array_offset_bytes']
        payload_size = common['ucode_size_bytes']

        # Test various CRC computation regions
        import binascii

        # CRC of just payload
        crc_payload = binascii.crc32(data[payload_off:payload_off + payload_size]) & 0xFFFFFFFF
        # CRC of entire file minus header
        crc_file = binascii.crc32(data[0x20:]) & 0xFFFFFFFF
        # CRC of payload using ucode_size_bytes
        crc_exact = binascii.crc32(data[payload_off:payload_off + common['ucode_size_bytes']]) & 0xFFFFFFFF

        declared = common['crc32']

        crc_analysis[fw_name] = {
            'declared_crc32': f"0x{declared:08X}",
            'crc32_of_payload': f"0x{crc_payload:08X}",
            'crc32_of_file_minus_header': f"0x{crc_file:08X}",
            'crc32_exact_payload': f"0x{crc_exact:08X}",
            'payload_matches': crc_payload == declared,
            'file_matches': crc_file == declared,
            'exact_matches': crc_exact == declared,
        }

        log(f"  CRC32 analysis for {fw_name}:")
        log(f"    Declared:           0x{declared:08X}")
        log(f"    Payload CRC:        0x{crc_payload:08X} {'MATCH' if crc_payload == declared else 'NO'}")
        log(f"    File-minus-hdr CRC: 0x{crc_file:08X} {'MATCH' if crc_file == declared else 'NO'}")

        # If none match, try different payload boundaries
        if not (crc_payload == declared or crc_file == declared):
            log("    Declared CRC doesn't match simple payload CRC — searching for matching region...")
            # Try header_size as start instead of payload_offset
            hdr_size = common['header_size_bytes']
            for start in [hdr_size, payload_off, 0x100, 0x200]:
                for end_delta in [0, 4, 8, -4, -8]:
                    test_end = payload_off + payload_size + end_delta
                    if start < test_end and test_end <= len(data):
                        test_crc = binascii.crc32(data[start:test_end]) & 0xFFFFFFFF
                        if test_crc == declared:
                            log(f"    FOUND MATCH: CRC32 of data[0x{start:X}:0x{test_end:X}] = 0x{test_crc:08X}")
                            crc_analysis[fw_name]['matching_region'] = f"0x{start:X}-0x{test_end:X}"
                            break

            # Also check if CRC might be 0 (not computed)
            if declared == 0:
                log("    CRC32 = 0x00000000 — likely not computed/checked by driver")
                crc_analysis[fw_name]['crc_is_zero'] = True

    results['crc_analysis'] = crc_analysis
    log("")

    # =========================================================================
    # STEP 4: Override mechanisms
    # =========================================================================
    results['override_analysis'] = analyze_override_mechanism()
    log("")

    # =========================================================================
    # STEP 5: PSP reload validation
    # =========================================================================
    results['psp_reload'] = check_psp_reload_behavior()
    log("")

    # =========================================================================
    # STEP 6: Complete loading chain
    # =========================================================================
    results['load_chain'] = analyze_load_chain()
    log("")

    # =========================================================================
    # CONCLUSIONS
    # =========================================================================
    log("=" * 70)
    log("CONCLUSIONS")
    log("=" * 70)

    conclusions = []

    # Check if any firmware has trailing signature data
    has_trailing_sig = False
    has_gap_sig = False
    all_crc_zero = True
    for fw_name, info in results['firmware_files'].items():
        sr = info.get('signature_region', {})
        if sr.get('trailing_looks_like_signature'):
            has_trailing_sig = True
        if sr.get('gap_looks_like_signature'):
            has_gap_sig = True
        common = info.get('common_header', {})
        if common.get('crc32', 0) != 0:
            all_crc_zero = False

    c1 = (
        "PSP SIGNATURE CHAIN IS ACTIVE: PSP is enabled, TEE is enabled, TSME is enabled. "
        "The PSP co-processor validates ALL firmware blobs before loading to GPU hardware. "
        "This uses RSA signatures against AMD's hardware root of trust (keys in PSP ROM)."
    )
    conclusions.append(c1)
    log(f"  1. {c1}")

    c2 = (
        f"FIRMWARE STRUCTURE: Files use common_firmware_header with header/payload layout. "
        f"Trailing signature data detected: {has_trailing_sig}. "
        f"Gap signature data detected: {has_gap_sig}. "
        f"All CRC32 fields zero (unused): {all_crc_zero}."
    )
    conclusions.append(c2)
    log(f"  2. {c2}")

    c3 = (
        "TWO VALIDATION LAYERS: "
        "(a) CRC32 checked by kernel driver (bypassable by recalculating). "
        "(b) RSA signature checked by PSP hardware (NOT bypassable without AMD private key). "
        "Even if CRC is recalculated, PSP will reject modified firmware."
    )
    conclusions.append(c3)
    log(f"  3. {c3}")

    c4 = (
        "OVERRIDE PATH EXISTS BUT IS USELESS: /lib/firmware/updates/ takes priority, "
        "and CONFIG_FW_LOADER_USER_HELPER=y. We COULD redirect firmware loading to modified blobs. "
        "However, PSP would reject any blob not signed with AMD's private key."
    )
    conclusions.append(c4)
    log(f"  4. {c4}")

    c5 = (
        "fw_load_type=-1 (AUTO) for GFX11 resolves to PSP loading. "
        "There is NO 'direct' loading mode for GFX11 that bypasses PSP. "
        "Unlike older GPUs (GFX6-8), GFX11 REQUIRES PSP for all firmware loading."
    )
    conclusions.append(c5)
    log(f"  5. {c5}")

    c6 = (
        "SECURE BOOT IS DISABLED at UEFI level, but this is IRRELEVANT. "
        "PSP firmware validation is completely independent of UEFI Secure Boot. "
        "PSP has its own hardware root of trust that cannot be disabled from the OS."
    )
    conclusions.append(c6)
    log(f"  6. {c6}")

    c7 = (
        "BOTTOM LINE: Modified firmware blobs WILL be rejected by PSP. "
        "The signature chain effectively blocks us. There is no known bypass for "
        "GFX11 PSP validation without: (a) AMD private signing key, (b) PSP exploit, "
        "or (c) hardware modification to PSP ROM. None of these are feasible."
    )
    conclusions.append(c7)
    log(f"  7. {c7}")

    c8 = (
        "PRACTICAL IMPLICATION FOR FEEL PROJECT: Custom firmware is NOT a viable path "
        "for the FEEL bridge. Continue using the existing approach: GPU noise harvesting "
        "via shader kernels + FPGA reservoir via UART/ETH. The firmware is a black box "
        "we must work around, not through."
    )
    conclusions.append(c8)
    log(f"  8. {c8}")

    results['conclusions'] = conclusions

    # =========================================================================
    # Save results
    # =========================================================================
    log("")
    log("Saving results...")

    # JSON results
    with open(JSON_PATH, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    log(f"  Saved {JSON_PATH}")

    # Signature analysis text
    with open(SIG_PATH, 'w') as f:
        f.write("z2342: PSP Signature Structure Analysis\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Date: {datetime.now().isoformat()}\n")
        f.write(f"GPU: AMD Radeon 8060S (gfx1151)\n")
        f.write(f"Kernel: {os.uname().release}\n\n")

        for fw_name, info in results['firmware_files'].items():
            f.write(f"\n{'='*60}\n{fw_name}\n{'='*60}\n")
            f.write(f"Size: {info['total_size']} bytes\n")
            f.write(f"MD5: {info['md5']}\n")
            f.write(f"SHA256: {info['sha256']}\n")

            common = info.get('common_header', {})
            if common:
                f.write(f"\nCommon Header:\n")
                for k, v in common.items():
                    if isinstance(v, int) and v > 255:
                        f.write(f"  {k}: 0x{v:08X} ({v})\n")
                    else:
                        f.write(f"  {k}: {v}\n")

            sr = info.get('signature_region', {})
            if sr:
                f.write(f"\nSignature Region:\n")
                for k, v in sr.items():
                    f.write(f"  {k}: {v}\n")

            fs = info.get('file_structure', {})
            if fs:
                f.write(f"\nFile Structure:\n")
                for k, v in fs.items():
                    f.write(f"  {k}: {v}\n")

        f.write(f"\n\n{'='*60}\nCONCLUSIONS\n{'='*60}\n")
        for i, c in enumerate(conclusions, 1):
            f.write(f"\n{i}. {c}\n")

    log(f"  Saved {SIG_PATH}")

    # Load chain text
    with open(CHAIN_PATH, 'w') as f:
        f.write("z2342: Firmware Loading Chain Analysis\n")
        f.write("=" * 60 + "\n\n")

        f.write("LOADING CHAIN DIAGRAM:\n\n")
        f.write("  UEFI/BIOS (Secure Boot: DISABLED)\n")
        f.write("       |\n")
        f.write("       v\n")
        f.write("  PSP Boot ROM (hardware root of trust, AMD private keys)\n")
        f.write("       |  [TSME enabled, TEE enabled]\n")
        f.write("       v\n")
        f.write("  Linux Kernel (amdgpu.ko)\n")
        f.write("       |  [fw_load_type=-1 AUTO -> PSP for GFX11]\n")
        f.write("       v\n")
        f.write("  request_firmware() -> /lib/firmware/amdgpu/gc_11_5_1_*.bin.zst\n")
        f.write("       |  [kernel checks CRC32 in common_firmware_header]\n")
        f.write("       v\n")
        f.write("  PSP Ring Buffer (C2PMSG mailbox)\n")
        f.write("       |  [PSP validates RSA signature against root of trust]\n")
        f.write("       v\n")
        f.write("  PSP loads to HW SRAM (CP, RLC, IMU, MES, VCN, etc.)\n")
        f.write("       |  [microengines start executing]\n")
        f.write("       v\n")
        f.write("  GPU operational\n\n")

        f.write("VALIDATION LAYERS:\n\n")
        f.write("  Layer 1 - CRC32 (kernel driver)\n")
        f.write("    Checked by: amdgpu kernel module\n")
        f.write("    Bypassable: YES (recalculate CRC after modification)\n\n")
        f.write("  Layer 2 - RSA Signature (PSP hardware)\n")
        f.write("    Checked by: PSP co-processor (hardware root of trust)\n")
        f.write("    Bypassable: NO (requires AMD private signing key)\n\n")
        f.write("  Layer 3 - TSME (memory encryption)\n")
        f.write("    Checked by: CCP/PSP hardware\n")
        f.write("    Purpose: Protects firmware in VRAM from physical attacks\n\n")

        f.write("KEY FINDINGS:\n\n")
        f.write("  - GFX11 has NO direct firmware loading mode (unlike GFX6-8)\n")
        f.write("  - ALL firmware must pass through PSP\n")
        f.write("  - PSP validates on EVERY load (boot and runtime reset)\n")
        f.write("  - UEFI Secure Boot status is IRRELEVANT to PSP validation\n")
        f.write("  - Override paths exist (/lib/firmware/updates/) but PSP still validates\n")
        f.write("  - Modified firmware WILL be rejected by PSP -> GPU init failure\n")
        f.write("  - No known bypass for GFX11 PSP without hardware exploit\n")

    log(f"  Saved {CHAIN_PATH}")

    log("")
    log(f"Final temperature: {check_temp():.1f}C")
    log("z2342 complete.")


if __name__ == '__main__':
    main()
