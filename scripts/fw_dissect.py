#!/usr/bin/env python3
"""
AMD GPU Firmware Blob Dissector — gfx1151 (Radeon 8060S)
=========================================================
Custom tool to parse, analyze, and map encryption/signing boundaries
in AMD GPU firmware blobs. Goes beyond PSPTool by targeting individual
.bin firmware files rather than BIOS images.

Produces:
  1. Header parse for every blob (common + IP-specific headers)
  2. PSP signature envelope detection ($PS1, $PS2 markers)
  3. Entropy analysis to identify encrypted vs cleartext regions
  4. Microcode pattern extraction (opcode frequency, instruction density)
  5. Cross-blob comparison (shared code segments)
  6. JSON report for downstream analysis

Usage:
  python fw_dissect.py [--blob-dir /tmp/fw_dissect] [--output results/fw_dissect_report.json]
"""

import struct
import hashlib
import json
import math
import os
import sys
import argparse
from collections import Counter, OrderedDict
from pathlib import Path
from datetime import datetime

# ─── Firmware Header Structures ───────────────────────────────────────────────

def parse_common_header(data):
    """Parse the 32-byte common_firmware_header present in ALL blobs."""
    if len(data) < 32:
        return None
    fields = struct.unpack_from('<IIHHHHI I I', data, 0)
    # Note: struct has 10 fields across 32 bytes
    # But the actual common header is:
    # uint32 size_bytes, uint32 header_size_bytes,
    # uint16 header_version_major, uint16 header_version_minor,
    # uint16 ip_version_major, uint16 ip_version_minor,
    # uint32 ucode_version, uint32 ucode_size_bytes,
    # uint32 ucode_array_offset_bytes, uint32 crc32
    hdr = {}
    off = 0
    hdr['size_bytes'] = struct.unpack_from('<I', data, off)[0]; off += 4
    hdr['header_size_bytes'] = struct.unpack_from('<I', data, off)[0]; off += 4
    hdr['header_version_major'] = struct.unpack_from('<H', data, off)[0]; off += 2
    hdr['header_version_minor'] = struct.unpack_from('<H', data, off)[0]; off += 2
    hdr['ip_version_major'] = struct.unpack_from('<H', data, off)[0]; off += 2
    hdr['ip_version_minor'] = struct.unpack_from('<H', data, off)[0]; off += 2
    hdr['ucode_version'] = struct.unpack_from('<I', data, off)[0]; off += 4
    hdr['ucode_size_bytes'] = struct.unpack_from('<I', data, off)[0]; off += 4
    hdr['ucode_array_offset_bytes'] = struct.unpack_from('<I', data, off)[0]; off += 4
    hdr['crc32'] = struct.unpack_from('<I', data, off)[0]; off += 4
    return hdr


def parse_gfx_header_v1(data):
    """GFX v1.0 extended header (ME, PFP, MEC)."""
    base = parse_common_header(data)
    if not base:
        return None
    off = 32
    if len(data) < off + 12:
        return base
    base['ucode_feature_version'] = struct.unpack_from('<I', data, off)[0]; off += 4
    base['jt_offset'] = struct.unpack_from('<I', data, off)[0]; off += 4
    base['jt_size'] = struct.unpack_from('<I', data, off)[0]; off += 4
    return base


def parse_gfx_header_v2(data):
    """GFX v2.0 extended header."""
    base = parse_common_header(data)
    if not base:
        return None
    off = 32
    fields = ['ucode_feature_version', 'ucode_size_bytes_v2',
              'ucode_offset_bytes', 'data_size_bytes', 'data_offset_bytes',
              'ucode_start_addr_lo', 'ucode_start_addr_hi']
    for f in fields:
        if len(data) < off + 4:
            break
        base[f] = struct.unpack_from('<I', data, off)[0]; off += 4
    return base


def parse_rlc_header_v2_3(data):
    """RLC v2.3 header (very large — 204 bytes)."""
    base = parse_common_header(data)
    if not base:
        return None
    off = 32
    rlc_fields = [
        'ucode_feature_version',
        'save_and_restore_offset', 'clear_state_descriptor_offset',
        'avail_scratch_ram_locations', 'master_pkt_description_offset',
        # v2.0 additions
        'jt_offset', 'jt_size',
        'reg_restore_list_size', 'reg_list_format_start',
        'reg_list_format_separate_start', 'starting_offsets_start',
        'reg_list_format_size_bytes', 'reg_list_format_array_offset_bytes',
        'reg_list_size_bytes', 'reg_list_array_offset_bytes',
        'reg_list_format_separate_size_bytes', 'reg_list_format_separate_array_offset_bytes',
        'reg_list_separate_size_bytes', 'reg_list_separate_array_offset_bytes',
        # v2.1 additions
        'reg_list_format_direct_reg_list_length',
        'save_restore_list_cntl_ucode_ver', 'save_restore_list_cntl_feature_ver',
        'save_restore_list_cntl_size_bytes', 'save_restore_list_cntl_offset_bytes',
        'save_restore_list_gpm_ucode_ver', 'save_restore_list_gpm_feature_ver',
        'save_restore_list_gpm_size_bytes', 'save_restore_list_gpm_offset_bytes',
        'save_restore_list_srm_ucode_ver', 'save_restore_list_srm_feature_ver',
        'save_restore_list_srm_size_bytes', 'save_restore_list_srm_offset_bytes',
        # v2.2 additions
        'rlc_iram_ucode_size_bytes', 'rlc_iram_ucode_offset_bytes',
        'rlc_dram_ucode_size_bytes', 'rlc_dram_ucode_offset_bytes',
        # v2.3 additions
        'rlcp_ucode_version', 'rlcp_ucode_feature_version',
        'rlcp_ucode_size_bytes', 'rlcp_ucode_offset_bytes',
        'rlcv_ucode_version', 'rlcv_ucode_feature_version',
        'rlcv_ucode_size_bytes', 'rlcv_ucode_offset_bytes',
    ]
    for f in rlc_fields:
        if len(data) < off + 4:
            break
        base[f] = struct.unpack_from('<I', data, off)[0]; off += 4
    return base


def parse_imu_header(data):
    """IMU v1.0 header."""
    base = parse_common_header(data)
    if not base:
        return None
    off = 32
    for f in ['imu_iram_ucode_size_bytes', 'imu_iram_ucode_offset_bytes',
              'imu_dram_ucode_size_bytes', 'imu_dram_ucode_offset_bytes']:
        if len(data) < off + 4:
            break
        base[f] = struct.unpack_from('<I', data, off)[0]; off += 4
    return base


def parse_mes_header(data):
    """MES v1.0 header."""
    base = parse_common_header(data)
    if not base:
        return None
    off = 32
    for f in ['mes_ucode_version', 'mes_ucode_size_bytes', 'mes_ucode_offset_bytes',
              'mes_ucode_data_version', 'mes_ucode_data_size_bytes',
              'mes_ucode_data_offset_bytes',
              'mes_uc_start_addr_lo', 'mes_uc_start_addr_hi',
              'mes_data_start_addr_lo', 'mes_data_start_addr_hi']:
        if len(data) < off + 4:
            break
        base[f] = struct.unpack_from('<I', data, off)[0]; off += 4
    return base


def parse_sdma_header(data):
    """SDMA header — detect version from common header."""
    base = parse_common_header(data)
    if not base:
        return None
    off = 32
    hv_major = base['header_version_major']
    if hv_major == 1:
        for f in ['ucode_feature_version', 'ucode_change_version',
                   'jt_offset', 'jt_size']:
            if len(data) < off + 4:
                break
            base[f] = struct.unpack_from('<I', data, off)[0]; off += 4
    elif hv_major == 2:
        for f in ['ucode_feature_version', 'ctx_ucode_size_bytes',
                   'ctx_jt_offset', 'ctx_jt_size',
                   'ctl_ucode_offset', 'ctl_ucode_size_bytes',
                   'ctl_jt_offset', 'ctl_jt_size']:
            if len(data) < off + 4:
                break
            base[f] = struct.unpack_from('<I', data, off)[0]; off += 4
    elif hv_major == 3:
        for f in ['ucode_feature_version', 'ucode_offset_bytes', 'ucode_size_bytes_v3']:
            if len(data) < off + 4:
                break
            base[f] = struct.unpack_from('<I', data, off)[0]; off += 4
    return base


def parse_smu_header(data):
    """SMU/PMFW header (v2.0 common + feature version)."""
    base = parse_common_header(data)
    if not base:
        return None
    off = 32
    if len(data) >= off + 4:
        base['ucode_feature_version'] = struct.unpack_from('<I', data, off)[0]
    return base


def parse_psp_header(data):
    """PSP firmware header — complex, version-dependent."""
    base = parse_common_header(data)
    if not base:
        return None
    off = 32
    hv = (base['header_version_major'], base['header_version_minor'])
    if hv[0] == 2:
        if len(data) >= off + 4:
            base['psp_fw_bin_count'] = struct.unpack_from('<I', data, off)[0]; off += 4
        # Each bin descriptor is 24 bytes: fw_type(u32) + offset(u32) + size(u32) + padding(12B)
        count = base.get('psp_fw_bin_count', 0)
        bins = []
        for i in range(min(count, 32)):  # cap at 32
            if len(data) < off + 12:
                break
            fw_type = struct.unpack_from('<I', data, off)[0]
            fw_off = struct.unpack_from('<I', data, off + 4)[0]
            fw_sz = struct.unpack_from('<I', data, off + 8)[0]
            bins.append({'fw_type': fw_type, 'offset': fw_off, 'size': fw_sz})
            off += 24  # 3x uint32 + 12 bytes padding
        base['fw_bins'] = bins
    return base


# ─── PSP Signature Envelope Detection ────────────────────────────────────────

PSP_MARKERS = {
    b'$PS1': 'PSP Signature v1 (RSA-2048 + SHA-256)',
    b'$PS2': 'PSP Signature v2 (RSA-4096 + SHA-384)',
    b'\x05\x00\x00\x00': 'PSP Directory Table (cookie 0x05)',
    b'\x24\x50\x53\x50': '$PSP Directory Header',
    b'\x24\x42\x48\x44': '$BHD BIOS Header Directory',
}

def scan_psp_signatures(data):
    """Scan for PSP signature markers and map their locations."""
    findings = []
    for marker, desc in PSP_MARKERS.items():
        offset = 0
        while True:
            idx = data.find(marker, offset)
            if idx == -1:
                break
            # Read surrounding context
            ctx_before = data[max(0, idx-8):idx].hex()
            ctx_after = data[idx:min(len(data), idx+32)].hex()
            findings.append({
                'offset': idx,
                'offset_hex': f'0x{idx:06x}',
                'marker': marker.hex(),
                'marker_ascii': marker.decode('ascii', errors='replace'),
                'description': desc,
                'context': f'...{ctx_before} [{ctx_after}]...',
            })
            offset = idx + len(marker)
    return findings


def scan_rsa_signatures(data):
    """Look for RSA signature patterns (large blocks of high-entropy data
    at expected offsets, typically 256 or 512 bytes)."""
    sigs = []
    # PSP typically places signatures right after $PS1 marker
    # $PS1 header: 4B marker + various fields, then 256B RSA sig
    offset = 0
    while True:
        idx = data.find(b'$PS1', offset)
        if idx == -1:
            break
        # Parse PS1 header
        if len(data) >= idx + 256:
            # After $PS1 (4B), there's typically:
            # 4B: header size, 4B: body size, 4B: encrypted flag, ...
            # Then RSA signature block
            ps1_hdr = {}
            ps1_hdr['marker_offset'] = idx
            if len(data) >= idx + 16:
                ps1_hdr['field_04'] = struct.unpack_from('<I', data, idx + 4)[0]
                ps1_hdr['field_08'] = struct.unpack_from('<I', data, idx + 8)[0]
                ps1_hdr['field_0c'] = struct.unpack_from('<I', data, idx + 12)[0]
            # Check if there's a 256-byte RSA block following
            sig_start = idx + 16  # typical offset
            if len(data) >= sig_start + 256:
                sig_block = data[sig_start:sig_start + 256]
                ent = shannon_entropy(sig_block)
                ps1_hdr['potential_rsa_offset'] = sig_start
                ps1_hdr['potential_rsa_entropy'] = round(ent, 4)
                ps1_hdr['likely_rsa'] = ent > 7.0  # high entropy = likely signature
            sigs.append(ps1_hdr)
        offset = idx + 4
    return sigs


# ─── Entropy Analysis ────────────────────────────────────────────────────────

def shannon_entropy(data):
    """Calculate Shannon entropy of a byte sequence (0-8 bits)."""
    if not data:
        return 0.0
    counts = Counter(data)
    total = len(data)
    entropy = 0.0
    for count in counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def entropy_map(data, block_size=256):
    """Build an entropy map over the entire blob in fixed-size blocks.
    Returns list of (offset, entropy) tuples.

    Interpretation:
      - 0.0-3.0: Highly structured / repetitive (NOP sleds, zero padding)
      - 3.0-5.0: Code-like (structured but varied)
      - 5.0-7.0: Compressed data or mixed content
      - 7.0-8.0: Encrypted or cryptographic material (signatures, keys)
    """
    blocks = []
    for i in range(0, len(data), block_size):
        chunk = data[i:i + block_size]
        if len(chunk) < block_size // 2:
            break
        blocks.append({
            'offset': i,
            'offset_hex': f'0x{i:06x}',
            'entropy': round(shannon_entropy(chunk), 4),
            'size': len(chunk),
        })
    return blocks


def classify_regions(entropy_blocks, threshold_low=3.0, threshold_high=7.0):
    """Classify regions as cleartext/code/encrypted based on entropy."""
    regions = []
    current_type = None
    current_start = 0

    for blk in entropy_blocks:
        ent = blk['entropy']
        if ent < threshold_low:
            region_type = 'structured_low_entropy'
        elif ent < 5.0:
            region_type = 'code_like'
        elif ent < threshold_high:
            region_type = 'compressed_or_data'
        else:
            region_type = 'encrypted_or_signature'

        if region_type != current_type:
            if current_type is not None:
                regions.append({
                    'type': current_type,
                    'start': current_start,
                    'start_hex': f'0x{current_start:06x}',
                    'end': blk['offset'],
                    'end_hex': f'0x{blk["offset"]:06x}',
                    'size': blk['offset'] - current_start,
                })
            current_type = region_type
            current_start = blk['offset']

    # Final region
    if current_type is not None and entropy_blocks:
        last = entropy_blocks[-1]
        end = last['offset'] + last['size']
        regions.append({
            'type': current_type,
            'start': current_start,
            'start_hex': f'0x{current_start:06x}',
            'end': end,
            'end_hex': f'0x{end:06x}',
            'size': end - current_start,
        })

    return regions


# ─── Microcode Pattern Analysis ──────────────────────────────────────────────

def analyze_instruction_patterns(data, ucode_offset, ucode_size):
    """Analyze the microcode payload for instruction-like patterns.
    AMD GPU microcode (ME/PFP/MEC) uses 32-bit instruction words.
    RLC uses a RISC-like 32-bit ISA.
    """
    if ucode_offset + ucode_size > len(data):
        ucode_size = len(data) - ucode_offset

    ucode = data[ucode_offset:ucode_offset + ucode_size]
    if len(ucode) < 4:
        return {}

    # Extract 32-bit words (little-endian)
    n_words = len(ucode) // 4
    words = struct.unpack_from(f'<{n_words}I', ucode)

    # Opcode analysis — top byte frequency
    top_byte_freq = Counter()
    top_nibble_freq = Counter()
    zero_words = 0
    nop_candidates = 0  # words that are all-zero or common NOP patterns

    for w in words:
        top_byte = (w >> 24) & 0xFF
        top_nibble = (w >> 28) & 0xF
        top_byte_freq[top_byte] += 1
        top_nibble_freq[top_nibble] += 1
        if w == 0:
            zero_words += 1
        if w in (0x00000000, 0xBF800000, 0xBF810000, 0xBF9F0000):
            nop_candidates += 1

    # GFX11 shader ISA NOP/endpgm detection
    # s_nop = 0xBF800000, s_endpgm = 0xBF810000, s_sendmsg_halt = 0xBF9F0000
    isa_markers = {
        's_nop (0xBF800000)': sum(1 for w in words if w == 0xBF800000),
        's_endpgm (0xBF810000)': sum(1 for w in words if w == 0xBF810000),
        's_sendmsg_halt (0xBF9F0000)': sum(1 for w in words if w == 0xBF9F0000),
        's_code_end (0xBF9F0000)': sum(1 for w in words if w == 0xBF9F0001),
    }

    # PM4 packet detection (command processor ME/PFP)
    # Type 3 packet: top 2 bits = 11, next 14 bits = opcode
    pm4_type3 = sum(1 for w in words if (w >> 30) == 3)
    pm4_type2 = sum(1 for w in words if w == 0x80000000)  # NOP packet
    pm4_type0 = sum(1 for w in words if (w >> 30) == 0 and w != 0)

    # Repeat pattern detection (firmware jump tables often have repeated values)
    word_freq = Counter(words)
    most_common_words = word_freq.most_common(10)

    return {
        'n_words': n_words,
        'n_unique_words': len(word_freq),
        'word_diversity': round(len(word_freq) / max(n_words, 1), 4),
        'zero_words': zero_words,
        'zero_word_pct': round(100.0 * zero_words / max(n_words, 1), 2),
        'nop_candidates': nop_candidates,
        'top_byte_distribution': {f'0x{k:02x}': v for k, v in top_byte_freq.most_common(16)},
        'top_nibble_distribution': {f'0x{k:x}': v for k, v in sorted(top_nibble_freq.items())},
        'isa_markers': {k: v for k, v in isa_markers.items() if v > 0},
        'pm4_analysis': {
            'type3_packets': pm4_type3,
            'type2_nop_packets': pm4_type2,
            'type0_packets': pm4_type0,
            'type3_pct': round(100.0 * pm4_type3 / max(n_words, 1), 2),
        },
        'most_common_words': [
            {'word': f'0x{w:08x}', 'count': c} for w, c in most_common_words
        ],
        'ucode_entropy': round(shannon_entropy(ucode), 4),
    }


# ─── Cross-Blob Comparison ──────────────────────────────────────────────────

def compute_blob_fingerprints(blobs):
    """Compute SHA-256 fingerprints for various regions of each blob."""
    fps = {}
    for name, data in blobs.items():
        fp = {
            'full_sha256': hashlib.sha256(data).hexdigest()[:16],
            'size': len(data),
        }
        # Header fingerprint (first 256 bytes)
        if len(data) >= 256:
            fp['header_sha256'] = hashlib.sha256(data[:256]).hexdigest()[:16]
        # Payload fingerprint (from offset 0x100 onward)
        if len(data) > 0x100:
            fp['payload_sha256'] = hashlib.sha256(data[0x100:]).hexdigest()[:16]
        # First 4KB after header
        if len(data) > 0x100 + 4096:
            fp['first_4k_sha256'] = hashlib.sha256(data[0x100:0x100+4096]).hexdigest()[:16]
        fps[name] = fp
    return fps


def find_shared_sequences(blobs, min_length=64):
    """Find byte sequences shared between different firmware blobs.
    This can reveal shared library code or common PSP wrapper code."""
    shared = []
    names = list(blobs.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a_data = blobs[names[i]]
            b_data = blobs[names[j]]
            # Check if any 64-byte aligned block in A appears in B
            # (sampling approach — full comparison too expensive)
            matches = 0
            sample_points = min(100, len(a_data) // min_length)
            for s in range(sample_points):
                offset = (s * len(a_data)) // sample_points
                offset = (offset // 4) * 4  # align to 4 bytes
                seq = a_data[offset:offset + min_length]
                if len(seq) == min_length and seq in b_data:
                    matches += 1
            if matches > 0:
                shared.append({
                    'blob_a': names[i],
                    'blob_b': names[j],
                    'shared_blocks': matches,
                    'sample_points': sample_points,
                    'share_pct': round(100.0 * matches / max(sample_points, 1), 2),
                })
    return shared


# ─── GPU Metrics Binary Decoder ──────────────────────────────────────────────

def decode_gpu_metrics(metrics_path='/sys/class/drm/card1/device/gpu_metrics'):
    """Decode the binary gpu_metrics sysfs file for raw ADC-level telemetry.
    Format: 4-byte header (uint16 structure_size, uint8 format_revision, uint8 content_revision)
    Then version-specific payload.
    """
    try:
        with open(metrics_path, 'rb') as f:
            data = f.read()
    except (FileNotFoundError, PermissionError) as e:
        return {'error': str(e)}

    if len(data) < 4:
        return {'error': 'Too short'}

    structure_size = struct.unpack_from('<H', data, 0)[0]
    format_rev = data[2]
    content_rev = data[3]

    result = {
        'structure_size': structure_size,
        'format_revision': format_rev,
        'content_revision': content_rev,
        'raw_size': len(data),
    }

    # Common fields for format 1.x (discrete) and 2.x (APU)
    # This is format 3.x for RDNA3.5/4 — need to discover layout
    # For now, dump all uint16 values as potential temperature/voltage readings
    if len(data) > 4:
        # Try to extract recognizable fields
        # Standard gpu_metrics_v1_3/v2_x layout starts after header
        off = 4
        raw_u16 = []
        raw_u32 = []
        while off + 2 <= len(data):
            raw_u16.append(struct.unpack_from('<H', data, off)[0])
            off += 2

        # Reset and try u32
        off = 4
        while off + 4 <= len(data):
            raw_u32.append(struct.unpack_from('<I', data, off)[0])
            off += 4

        # Heuristic: temperatures are typically 2000-10000 (20.00-100.00C)
        # Voltages are typically 500-1500 (mV)
        # Clocks are typically 200-3000 (MHz)
        potential_temps = [(i, v / 100.0) for i, v in enumerate(raw_u16)
                          if 2000 <= v <= 12000]
        potential_voltages = [(i, v) for i, v in enumerate(raw_u16)
                             if 300 <= v <= 2000]
        potential_clocks = [(i, v) for i, v in enumerate(raw_u16)
                           if 100 <= v <= 4000]

        result['raw_u16_count'] = len(raw_u16)
        result['raw_u32_count'] = len(raw_u32)
        result['potential_temperatures_C'] = potential_temps[:10]
        result['potential_voltages_mV'] = potential_voltages[:10]
        result['potential_clocks_MHz'] = potential_clocks[:10]
        result['raw_hex_first_128B'] = data[4:132].hex()

    return result


# ─── Main Dissection Pipeline ────────────────────────────────────────────────

BLOB_PARSERS = {
    'gc_11_5_1_me': ('ME (Micro Engine)', parse_gfx_header_v1),
    'gc_11_5_1_pfp': ('PFP (Pre-Fetch Parser)', parse_gfx_header_v1),
    'gc_11_5_1_mec': ('MEC (Compute Engine)', parse_gfx_header_v1),
    'gc_11_5_1_rlc': ('RLC (Run List Controller)', parse_rlc_header_v2_3),
    'gc_11_5_1_imu': ('IMU (Interrupt Mgmt Unit)', parse_imu_header),
    'gc_11_5_1_mes1': ('MES1 (HW Scheduler pipe1)', parse_mes_header),
    'gc_11_5_1_mes_2': ('MES2 (HW Scheduler pipe2)', parse_mes_header),
    'smu_14_0_2': ('SMU/PMFW (Power Management)', parse_smu_header),
    'sdma_7_0_0': ('SDMA (System DMA)', parse_sdma_header),
    'psp_14_0_4_toc': ('PSP TOC (Table of Contents)', parse_psp_header),
    'psp_14_0_4_ta': ('PSP TA (Trusted Apps)', parse_psp_header),
    'vcn_5_0_0': ('VCN (Video Core Next)', parse_common_header),
    'dcn_4_0_1_dmcub': ('DMCUB (Display Controller)', parse_common_header),
}


def dissect_blob(name, data, parser_fn):
    """Full dissection of a single firmware blob."""
    result = OrderedDict()
    result['name'] = name
    result['description'] = BLOB_PARSERS.get(name, ('Unknown', None))[0]
    result['file_size'] = len(data)
    result['sha256'] = hashlib.sha256(data).hexdigest()

    # 1. Parse header
    header = parser_fn(data)
    result['header'] = header

    # 2. PSP signature scan
    psp_sigs = scan_psp_signatures(data)
    result['psp_signatures'] = psp_sigs

    # 3. RSA signature analysis
    rsa_sigs = scan_rsa_signatures(data)
    result['rsa_signatures'] = rsa_sigs

    # 4. Entropy map (256-byte blocks)
    ent_map = entropy_map(data, block_size=256)
    result['entropy_summary'] = {
        'overall_entropy': round(shannon_entropy(data), 4),
        'min_block_entropy': round(min(b['entropy'] for b in ent_map), 4) if ent_map else 0,
        'max_block_entropy': round(max(b['entropy'] for b in ent_map), 4) if ent_map else 0,
        'avg_block_entropy': round(sum(b['entropy'] for b in ent_map) / len(ent_map), 4) if ent_map else 0,
        'n_blocks': len(ent_map),
    }

    # 5. Region classification
    regions = classify_regions(ent_map)
    result['region_classification'] = regions

    # Region summary
    region_summary = Counter()
    for r in regions:
        region_summary[r['type']] += r['size']
    result['region_summary'] = dict(region_summary)

    # 6. Microcode pattern analysis
    if header:
        ucode_off = header.get('ucode_array_offset_bytes', 0x100)
        ucode_sz = header.get('ucode_size_bytes', len(data) - ucode_off)
        result['microcode_analysis'] = analyze_instruction_patterns(data, ucode_off, ucode_sz)

    # 7. Header-to-payload gap analysis (what's between header end and ucode start)
    if header:
        hdr_size = header.get('header_size_bytes', 32)
        ucode_off = header.get('ucode_array_offset_bytes', 0x100)
        gap_start = hdr_size
        gap_end = ucode_off
        if gap_end > gap_start and gap_end <= len(data):
            gap_data = data[gap_start:gap_end]
            result['header_payload_gap'] = {
                'start': gap_start,
                'end': gap_end,
                'size': gap_end - gap_start,
                'entropy': round(shannon_entropy(gap_data), 4),
                'hex_first_64': gap_data[:64].hex() if len(gap_data) >= 64 else gap_data.hex(),
                'is_zero_padded': all(b == 0 for b in gap_data),
                'non_zero_bytes': sum(1 for b in gap_data if b != 0),
            }

    # 8. Entropy map (store full for plotting, but downsample for JSON)
    result['entropy_map_downsampled'] = [
        {'offset': b['offset'], 'entropy': b['entropy']}
        for b in ent_map[::max(1, len(ent_map) // 64)]
    ]

    return result


def main():
    parser = argparse.ArgumentParser(description='AMD GPU Firmware Blob Dissector')
    parser.add_argument('--blob-dir', default='/tmp/fw_dissect',
                        help='Directory containing decompressed .bin files')
    parser.add_argument('--output', default=None,
                        help='Output JSON report path')
    parser.add_argument('--gpu-metrics', action='store_true',
                        help='Also decode live gpu_metrics')
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()

    blob_dir = Path(args.blob_dir)
    if not blob_dir.exists():
        print(f"ERROR: Blob directory {blob_dir} does not exist")
        sys.exit(1)

    # Load all blobs
    blobs = {}
    for name in BLOB_PARSERS:
        path = blob_dir / f'{name}.bin'
        if path.exists():
            blobs[name] = path.read_bytes()
            if args.verbose:
                print(f"  Loaded {name}: {len(blobs[name])} bytes")
        else:
            print(f"  SKIP {name}: not found at {path}")

    print(f"\n{'='*70}")
    print(f"AMD GPU Firmware Dissector — gfx1151 (Radeon 8060S)")
    print(f"{'='*70}")
    print(f"Loaded {len(blobs)} firmware blobs from {blob_dir}")
    print(f"Total firmware size: {sum(len(d) for d in blobs.values()):,} bytes")
    print()

    # Dissect each blob
    report = OrderedDict()
    report['timestamp'] = datetime.now().isoformat()
    report['device'] = 'AMD Radeon 8060S (gfx1151)'
    report['blob_dir'] = str(blob_dir)
    report['blobs'] = OrderedDict()

    for name, data in blobs.items():
        desc, parser_fn = BLOB_PARSERS[name]
        print(f"Dissecting {name} ({desc})...")
        result = dissect_blob(name, data, parser_fn)
        report['blobs'][name] = result

        # Print summary
        hdr = result.get('header', {})
        ent = result.get('entropy_summary', {})
        print(f"  Size: {len(data):,}B | Header: v{hdr.get('header_version_major','?')}.{hdr.get('header_version_minor','?')} | "
              f"IP: v{hdr.get('ip_version_major','?')}.{hdr.get('ip_version_minor','?')} | "
              f"ucode: 0x{hdr.get('ucode_version', 0):08x} | "
              f"Entropy: {ent.get('overall_entropy', 0):.2f}")

        # PSP signatures
        for sig in result.get('psp_signatures', []):
            print(f"    PSP sig @ {sig['offset_hex']}: {sig['description']}")

        # Region summary
        for rtype, size in result.get('region_summary', {}).items():
            pct = 100.0 * size / len(data)
            print(f"    Region: {rtype} = {size:,}B ({pct:.1f}%)")

        # Microcode highlights
        mc = result.get('microcode_analysis', {})
        if mc:
            print(f"    Microcode: {mc.get('n_words', 0)} words, "
                  f"diversity={mc.get('word_diversity', 0):.3f}, "
                  f"entropy={mc.get('ucode_entropy', 0):.2f}")
            pm4 = mc.get('pm4_analysis', {})
            if pm4.get('type3_pct', 0) > 5:
                print(f"    PM4 Type3 packets: {pm4['type3_packets']} ({pm4['type3_pct']}%)")

        print()

    # Cross-blob comparison
    print("Cross-blob comparison...")
    report['cross_blob'] = {
        'fingerprints': compute_blob_fingerprints(blobs),
        'shared_sequences': find_shared_sequences(blobs),
    }

    shared = report['cross_blob']['shared_sequences']
    if shared:
        for s in shared:
            print(f"  Shared code: {s['blob_a']} <-> {s['blob_b']}: "
                  f"{s['shared_blocks']}/{s['sample_points']} blocks ({s['share_pct']}%)")
    else:
        print("  No shared 64-byte sequences found between blobs")

    # GPU metrics (optional)
    if args.gpu_metrics:
        print("\nDecoding live gpu_metrics...")
        # Try multiple card paths
        for card in ['card0', 'card1', 'card2']:
            metrics_path = f'/sys/class/drm/{card}/device/gpu_metrics'
            if os.path.exists(metrics_path):
                report['gpu_metrics'] = decode_gpu_metrics(metrics_path)
                report['gpu_metrics']['card'] = card
                print(f"  Decoded from {metrics_path}")
                gm = report['gpu_metrics']
                if 'potential_temperatures_C' in gm:
                    print(f"  Potential temperatures: {gm['potential_temperatures_C'][:5]}")
                if 'potential_voltages_mV' in gm:
                    print(f"  Potential voltages: {gm['potential_voltages_mV'][:5]}")
                break

    # Save report
    if args.output is None:
        args.output = str(Path(__file__).parent.parent / 'results' / 'fw_dissect_report.json')

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport saved to {args.output}")

    # Print encryption boundary summary
    print(f"\n{'='*70}")
    print("ENCRYPTION/SIGNING BOUNDARY MAP")
    print(f"{'='*70}")
    for name, result in report['blobs'].items():
        regions = result.get('region_classification', [])
        encrypted = sum(r['size'] for r in regions if 'encrypted' in r['type'])
        code_like = sum(r['size'] for r in regions if 'code' in r['type'])
        total = result['file_size']
        enc_pct = 100.0 * encrypted / total if total else 0
        code_pct = 100.0 * code_like / total if total else 0

        has_psp = len(result.get('psp_signatures', [])) > 0
        status = "SIGNED" if has_psp else "UNSIGNED"
        enc_status = f"ENCRYPTED:{enc_pct:.0f}%" if enc_pct > 10 else f"CLEARTEXT:{code_pct:.0f}%"

        print(f"  {name:30s} {status:10s} {enc_status:20s} ({total:>8,}B)")


if __name__ == '__main__':
    main()
