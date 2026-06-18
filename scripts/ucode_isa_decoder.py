#!/usr/bin/env python3
"""
AMD GPU Proprietary Microcode ISA Decoder — gfx1151
=====================================================
Novel tool: No existing open-source tool decodes the proprietary
microcode ISA used inside ME/PFP/MEC/RLC/SDMA firmware blobs.

This is NOT the shader ISA (documented in RDNA ISA guides).
This is the **command processor microcode** ISA — a proprietary
RISC-like instruction set used by the GPU's packet-processing engines.

What we know from kernel source + binary analysis:
  - 32-bit instruction words (little-endian)
  - ME/PFP: PM4 packet processor — interprets PM4 command packets
  - MEC: Compute queue command processor
  - RLC: Run-list controller — power gating, context switching
  - SDMA: DMA engine microcode
  - IMU: Interrupt management (new in GFX11)
  - MES: Hardware scheduler (new in GFX11)

Approach:
  1. Statistical opcode extraction (top-byte/top-nibble analysis)
  2. Control flow detection (branch targets, loop patterns)
  3. Register reference extraction (known MMIO register addresses)
  4. PM4 opcode cross-reference (ME/PFP process these opcodes)
  5. String/constant extraction
  6. Jump table detection
  7. Instruction clustering (identify likely ISA encoding families)

Usage:
  python ucode_isa_decoder.py --blob /tmp/fw_dissect/gc_11_5_1_me.bin
  python ucode_isa_decoder.py --all --blob-dir /tmp/fw_dissect
"""

import struct
import json
import os
import sys
import argparse
from collections import Counter, defaultdict
from pathlib import Path
from datetime import datetime


# ─── Known AMD Register Addresses (from UMR / kernel source) ─────────────────
# These are MMIO register offsets that microcode references to control hardware

KNOWN_REGISTERS = {
    # GC registers
    0x2040: 'CP_RB0_BASE',
    0x2041: 'CP_RB0_CNTL',
    0x2044: 'CP_RB0_RPTR',
    0x2045: 'CP_RB0_WPTR',
    0x2084: 'CP_ME_CNTL',
    0x2088: 'CP_ME_RAM_WADDR',
    0x2089: 'CP_ME_RAM_RADDR',
    0x208A: 'CP_ME_RAM_DATA',
    0x2098: 'GRBM_CNTL',
    0x2200: 'CP_MEC_CNTL',
    0x20C0: 'CP_PFP_IB_CONTROL',
    0x30C0: 'RLC_CNTL',
    0x30C1: 'RLC_STAT',
    0x30C4: 'RLC_SAFE_MODE',
    0x30C8: 'RLC_SRM_CNTL',
    0x30D0: 'RLC_GPM_GENERAL_0',
    0x30D4: 'RLC_GPM_GENERAL_4',
    0x30E0: 'RLC_RLCS_GPM_STAT',
    0x3040: 'RLC_GPR_REG2',
    # GRBM
    0x2000: 'GRBM_STATUS',
    0x2004: 'GRBM_STATUS2',
    0x2008: 'GRBM_STATUS_SE0',
    0x200C: 'GRBM_STATUS_SE1',
    0x2010: 'GRBM_SOFT_RESET',
    0x2020: 'GRBM_GFX_INDEX',
    # CP
    0x2100: 'CP_STALLED_STAT1',
    0x2104: 'CP_STALLED_STAT2',
    0x2108: 'CP_STALLED_STAT3',
    0x210C: 'CP_BUSY_STAT',
    # SMU/PMFW mailbox
    0x03B1: 'MP1_SMN_C2PMSG_66',  # SMU msg request
    0x03B5: 'MP1_SMN_C2PMSG_82',  # SMU msg arg
    0x03B9: 'MP1_SMN_C2PMSG_90',  # SMU response
    # SDMA
    0x4200: 'SDMA0_STATUS_REG',
    0x4204: 'SDMA0_STATUS1_REG',
    0x4220: 'SDMA0_CNTL',
    0x4224: 'SDMA0_GFX_RB_CNTL',
    # IH (Interrupt Handler)
    0x1A00: 'IH_RB_CNTL',
    0x1A04: 'IH_RB_BASE',
    0x1A0C: 'IH_RB_RPTR',
    0x1A10: 'IH_RB_WPTR',
}

# PM4 opcodes (Type 3 packet opcodes that ME/PFP process)
PM4_OPCODES = {
    0x10: 'NOP',
    0x11: 'SET_BASE',
    0x12: 'CLEAR_STATE',
    0x15: 'DISPATCH_DIRECT',
    0x16: 'DISPATCH_INDIRECT',
    0x27: 'ATOMIC_GDS',
    0x28: 'ATOMIC_MEM',
    0x2D: 'WRITE_DATA',
    0x2E: 'DRAW_INDEX_INDIRECT_MULTI',
    0x2F: 'MEM_SEMAPHORE',
    0x30: 'COPY_DATA',
    0x31: 'CP_DMA',
    0x32: 'PFP_SYNC_ME',
    0x33: 'SURFACE_SYNC',
    0x34: 'ME_INITIALIZE',
    0x35: 'COND_WRITE',
    0x36: 'EVENT_WRITE',
    0x37: 'EVENT_WRITE_EOP',
    0x38: 'EVENT_WRITE_EOS',
    0x39: 'RELEASE_MEM',
    0x3C: 'PREAMBLE_CNTL',
    0x3F: 'DMA_DATA',
    0x40: 'CONTEXT_CONTROL',
    0x43: 'ACQUIRE_MEM',
    0x46: 'REWIND',
    0x47: 'INVALIDATE_TLBS',
    0x49: 'SET_CONFIG_REG',
    0x4A: 'SET_CONTEXT_REG',
    0x4C: 'SET_SH_REG',
    0x4D: 'SET_SH_REG_OFFSET',
    0x50: 'SET_UCONFIG_REG',
    0x58: 'LOAD_CONST_RAM',
    0x59: 'WRITE_CONST_RAM',
    0x5B: 'DUMP_CONST_RAM',
    0x5C: 'INCREMENT_CE_COUNTER',
    0x5E: 'WAIT_ON_CE_COUNTER',
    0x68: 'SET_SH_REG_INDEX',
    0x69: 'LOAD_CONTEXT_REG_INDEX',
    0x73: 'SET_UCONFIG_REG_INDEX',
    0x78: 'LOAD_SH_REG_INDEX',
    0x80: 'SET_RESOURCES',
    0x81: 'MAP_PROCESS',
    0x82: 'MAP_QUEUES',
    0x83: 'UNMAP_QUEUES',
    0x84: 'QUERY_STATUS',
    0x85: 'RUN_LIST',
    0xA0: 'DRAW_INDEX_OFFSET_2',
    0xA1: 'DRAW_INDEX_MULTI_ELEMENT',
    0xA2: 'DRAW_INDEX_AUTO',
    0xA4: 'DRAW_INDEX_MULTI_INST',
    0xA5: 'DRAW_INDEX_MULTI_AUTO',
    0xA9: 'NUM_INSTANCES',
}


# ─── Microcode Analysis Functions ────────────────────────────────────────────

def extract_words(data, offset=0x100):
    """Extract 32-bit instruction words from microcode payload."""
    payload = data[offset:]
    n = len(payload) // 4
    return list(struct.unpack_from(f'<{n}I', payload))


def analyze_opcode_encoding(words):
    """Statistical analysis of instruction encoding patterns.

    Strategy: since we don't know the ISA, we look at the distribution of
    various bit fields to infer the encoding format.

    Key hypothesis: most RISC ISAs use the top 4-8 bits for opcode.
    """
    n = len(words)
    if n == 0:
        return {}

    # Analyze different potential opcode widths
    encodings = {}

    for width_name, shift, mask in [
        ('top_4bit', 28, 0xF),
        ('top_6bit', 26, 0x3F),
        ('top_8bit', 24, 0xFF),
        ('top_10bit', 22, 0x3FF),
        ('bits_31_30', 30, 0x3),      # PM4 packet type field
        ('bits_29_23', 23, 0x7F),     # PM4 opcode field (within type 3)
    ]:
        freq = Counter()
        for w in words:
            opcode = (w >> shift) & mask
            freq[opcode] += 1

        # Compute entropy of this field
        import math
        entropy = 0.0
        for count in freq.values():
            p = count / n
            if p > 0:
                entropy -= p * math.log2(p)

        # Low entropy = good opcode field candidate (few distinct values used often)
        max_entropy = math.log2(min(len(freq), mask + 1)) if freq else 0

        encodings[width_name] = {
            'n_distinct': len(freq),
            'max_possible': mask + 1,
            'entropy': round(entropy, 3),
            'max_entropy': round(max_entropy, 3),
            'entropy_ratio': round(entropy / max_entropy, 3) if max_entropy > 0 else 0,
            'top_values': [
                {'opcode': f'0x{op:0{(shift//4+1)}x}', 'count': cnt,
                 'pct': round(100*cnt/n, 2)}
                for op, cnt in freq.most_common(12)
            ],
        }

    return encodings


def detect_branch_targets(words):
    """Detect potential branch/jump targets.

    Branch instructions typically encode a relative or absolute offset.
    Look for words where the lower 16-24 bits form a valid address
    within the firmware image, and the instruction appears in a
    pattern consistent with control flow.
    """
    n = len(words)
    targets = Counter()
    branches = []

    for i, w in enumerate(words):
        # Hypothesis 1: lower 16 bits are a word-address target
        target_16 = w & 0xFFFF
        if 0 < target_16 < n:
            targets[target_16] += 1

        # Hypothesis 2: lower 24 bits are a byte-address target
        target_24 = w & 0xFFFFFF
        if 0 < target_24 < n * 4:
            target_word = target_24 // 4
            targets[target_word] += 1

        # Hypothesis 3: signed relative offset (lower 16 bits, signed)
        rel_16 = w & 0xFFFF
        if rel_16 & 0x8000:
            rel_16 -= 0x10000
        abs_target = i + rel_16
        if 0 <= abs_target < n and abs(rel_16) > 1:
            targets[abs_target] += 1

    # Filter: real branch targets should be referenced multiple times
    likely_targets = {addr: cnt for addr, cnt in targets.items() if cnt >= 2}

    # Find potential function prologues (addresses referenced >= 3 times)
    functions = {addr: cnt for addr, cnt in targets.items() if cnt >= 3}

    return {
        'total_potential_targets': len(targets),
        'likely_targets_ge2': len(likely_targets),
        'likely_functions_ge3': len(functions),
        'top_targets': [
            {'word_addr': addr, 'byte_offset': f'0x{addr*4+0x100:06x}',
             'references': cnt}
            for addr, cnt in sorted(likely_targets.items(),
                                     key=lambda x: -x[1])[:20]
        ],
    }


def extract_register_references(words):
    """Find references to known MMIO register addresses in microcode."""
    refs = defaultdict(list)

    for i, w in enumerate(words):
        # Check if any 16-bit field matches a known register
        lo16 = w & 0xFFFF
        hi16 = (w >> 16) & 0xFFFF

        for val, which in [(lo16, 'lo16'), (hi16, 'hi16')]:
            if val in KNOWN_REGISTERS:
                refs[KNOWN_REGISTERS[val]].append({
                    'word_index': i,
                    'byte_offset': f'0x{i*4+0x100:06x}',
                    'instruction': f'0x{w:08x}',
                    'field': which,
                })

    return {name: entries for name, entries in sorted(refs.items())}


def extract_pm4_references(words):
    """Find PM4 opcode references in ME/PFP microcode.

    The command processor must handle each PM4 opcode, so the microcode
    should contain jump table entries or opcode comparisons for each.
    """
    refs = defaultdict(list)

    for i, w in enumerate(words):
        # Check if this word is a PM4 Type 3 packet header
        if (w >> 30) == 3:
            opcode = (w >> 8) & 0xFF
            count = w & 0x3FFF
            if opcode in PM4_OPCODES:
                refs[PM4_OPCODES[opcode]].append({
                    'word_index': i,
                    'byte_offset': f'0x{i*4+0x100:06x}',
                    'raw': f'0x{w:08x}',
                    'count_dwords': count,
                })

        # Also check if the opcode appears as a bare constant (jump table)
        for val_16 in [w & 0xFF, (w >> 8) & 0xFF, (w >> 16) & 0xFF, (w >> 24) & 0xFF]:
            pass  # Too noisy — stick with full packet detection

    return {name: entries for name, entries in sorted(refs.items())}


def detect_jump_tables(words):
    """Detect jump tables (consecutive words that all point to valid addresses).

    Jump tables are common in packet processors: index by opcode → handler address.
    """
    n = len(words)
    tables = []

    i = 0
    while i < n - 4:
        # Check if next 4+ words are all valid addresses within firmware
        run_start = i
        while i < n:
            target = words[i]
            # Valid address heuristic: word-aligned, within firmware range
            if target > 0 and target < n * 4 and (target & 3) == 0:
                i += 1
            elif target == 0:
                i += 1  # Allow zero entries in jump tables
            else:
                break

        run_len = i - run_start
        if run_len >= 8:  # Minimum 8 entries for a plausible jump table
            # Verify: entries should have reasonable spread
            entries = words[run_start:run_start + run_len]
            non_zero = [e for e in entries if e > 0]
            if len(non_zero) >= 4:
                unique = len(set(non_zero))
                tables.append({
                    'start_word': run_start,
                    'start_offset': f'0x{run_start*4+0x100:06x}',
                    'length': run_len,
                    'non_zero_entries': len(non_zero),
                    'unique_entries': unique,
                    'sample_entries': [f'0x{e:08x}' for e in entries[:8]],
                })
        i = max(i, run_start + 1)

    return tables


def detect_string_constants(data, offset=0x100):
    """Extract ASCII string constants from firmware payload."""
    payload = data[offset:]
    strings = []
    current = b''
    start = 0

    for i, b in enumerate(payload):
        if 0x20 <= b < 0x7F:
            if not current:
                start = i
            current += bytes([b])
        else:
            if len(current) >= 6:  # minimum 6 chars
                strings.append({
                    'offset': f'0x{start+offset:06x}',
                    'length': len(current),
                    'string': current.decode('ascii'),
                })
            current = b''

    return strings


def detect_repeated_patterns(words, pattern_len=4):
    """Find repeated instruction patterns (potential loops or idioms)."""
    patterns = Counter()
    for i in range(len(words) - pattern_len):
        pat = tuple(words[i:i + pattern_len])
        if all(w != 0 for w in pat):  # skip zero-filled regions
            patterns[pat] += 1

    repeated = [(pat, cnt) for pat, cnt in patterns.items() if cnt >= 3]
    repeated.sort(key=lambda x: -x[1])

    return [
        {
            'count': cnt,
            'words': [f'0x{w:08x}' for w in pat],
            'first_occurrence': next(i for i in range(len(words)-pattern_len)
                                     if tuple(words[i:i+pattern_len]) == pat),
        }
        for pat, cnt in repeated[:15]
    ]


def analyze_instruction_density(words, window=64):
    """Analyze instruction density (non-zero words per window).
    High density = code. Low density = data/padding.
    """
    density_map = []
    for i in range(0, len(words), window):
        chunk = words[i:i + window]
        non_zero = sum(1 for w in chunk if w != 0)
        density = non_zero / len(chunk) if chunk else 0
        density_map.append({
            'word_offset': i,
            'byte_offset': f'0x{i*4+0x100:06x}',
            'density': round(density, 3),
            'non_zero': non_zero,
            'total': len(chunk),
        })

    # Identify code vs data regions
    code_regions = []
    data_regions = []
    for blk in density_map:
        if blk['density'] > 0.5:
            code_regions.append(blk)
        else:
            data_regions.append(blk)

    return {
        'total_blocks': len(density_map),
        'code_blocks': len(code_regions),
        'data_blocks': len(data_regions),
        'code_pct': round(100 * len(code_regions) / max(len(density_map), 1), 1),
        'density_map': density_map[::max(1, len(density_map)//32)],  # downsample
    }


# ─── PSP Signature Envelope Decoder ─────────────────────────────────────────

def decode_psp_envelope(data):
    """Decode the PSP $PS1 signature envelope wrapping the microcode.

    Structure (reverse-engineered from binary analysis):
    Offset 0x100: Start of PSP wrapper
    +0x00: 4 bytes — field (varies)
    +0x04: 4 bytes — '$PS1' marker (0x24505331)
    +0x08: 4 bytes — body size (signed payload + signature)
    +0x0C: 4 bytes — flags (bit 0 = encrypted?)
    +0x10: 256 bytes — RSA-2048 signature
    +0x110: payload starts

    OR:
    +0x00: 4 bytes — unknown
    +0x04: 4 bytes — unknown
    +0x08: 4 bytes — unknown
    +0x0C: 4 bytes — unknown
    +0x10: '$PS1' marker
    """
    envelopes = []
    offset = 0

    while offset < len(data) - 16:
        idx = data.find(b'$PS1', offset)
        if idx == -1:
            break

        env = {
            'marker_offset': idx,
            'marker_hex': f'0x{idx:06x}',
        }

        # Read fields around the marker
        if idx >= 4:
            env['field_minus4'] = f'0x{struct.unpack_from("<I", data, idx-4)[0]:08x}'
        if idx >= 8:
            env['field_minus8'] = f'0x{struct.unpack_from("<I", data, idx-8)[0]:08x}'
        if idx >= 12:
            env['field_minus12'] = f'0x{struct.unpack_from("<I", data, idx-12)[0]:08x}'
        if idx >= 16:
            env['field_minus16'] = f'0x{struct.unpack_from("<I", data, idx-16)[0]:08x}'

        # Fields after marker
        if idx + 20 <= len(data):
            env['field_plus4'] = f'0x{struct.unpack_from("<I", data, idx+4)[0]:08x}'
            env['field_plus8'] = f'0x{struct.unpack_from("<I", data, idx+8)[0]:08x}'
            env['field_plus12'] = f'0x{struct.unpack_from("<I", data, idx+12)[0]:08x}'
            env['field_plus16'] = f'0x{struct.unpack_from("<I", data, idx+16)[0]:08x}'

        # Check for RSA signature block after marker
        # Typical: marker at +0x10, RSA at +0x14 (256 bytes for RSA-2048)
        sig_candidates = [idx + 4, idx + 16, idx + 20]
        for sig_start in sig_candidates:
            if sig_start + 256 <= len(data):
                sig_block = data[sig_start:sig_start + 256]
                # High entropy = likely RSA signature
                from collections import Counter as C2
                import math
                counts = C2(sig_block)
                ent = -sum((c/256) * math.log2(c/256) for c in counts.values())
                if ent > 7.0:
                    env['rsa_sig_offset'] = f'0x{sig_start:06x}'
                    env['rsa_sig_entropy'] = round(ent, 3)
                    env['rsa_sig_first16'] = sig_block[:16].hex()
                    env['rsa_sig_last16'] = sig_block[-16:].hex()
                    env['payload_starts_at'] = f'0x{sig_start+256:06x}'
                    break

        # Determine if payload after signature is encrypted
        payload_start = sig_start + 256 if 'rsa_sig_offset' in env else idx + 256
        if payload_start + 256 <= len(data):
            test_block = data[payload_start:payload_start + 256]
            counts = C2(test_block)
            payload_ent = -sum((c/256) * math.log2(c/256) for c in counts.values())
            env['payload_entropy'] = round(payload_ent, 3)
            env['payload_likely_encrypted'] = payload_ent > 7.5
            env['payload_assessment'] = (
                'ENCRYPTED' if payload_ent > 7.5 else
                'COMPRESSED' if payload_ent > 6.0 else
                'CODE' if payload_ent > 3.0 else
                'STRUCTURED/TABLES'
            )

        envelopes.append(env)
        offset = idx + 4

    return envelopes


# ─── Main Pipeline ───────────────────────────────────────────────────────────

BLOB_INFO = {
    'gc_11_5_1_me': ('ME (Command Processor)', 'PM4 packet processing'),
    'gc_11_5_1_pfp': ('PFP (Pre-Fetch Parser)', 'Command pre-fetching'),
    'gc_11_5_1_mec': ('MEC (Compute Engine)', 'Compute queue processing'),
    'gc_11_5_1_rlc': ('RLC (Run List Controller)', 'Power gating, context switch'),
    'gc_11_5_1_imu': ('IMU (Interrupt Mgmt)', 'GFX11 interrupt handling'),
    'gc_11_5_1_mes1': ('MES1 (HW Scheduler P1)', 'Hardware scheduling pipe 1'),
    'gc_11_5_1_mes_2': ('MES2 (HW Scheduler P2)', 'Hardware scheduling pipe 2'),
    'smu_14_0_2': ('SMU/PMFW', 'Power management firmware'),
    'sdma_7_0_0': ('SDMA', 'System DMA engine'),
}


def analyze_blob(name, data):
    """Full microcode ISA analysis for a single blob."""
    info = BLOB_INFO.get(name, ('Unknown', ''))
    print(f"\n{'='*70}")
    print(f"  {name} — {info[0]}")
    print(f"  Function: {info[1]}")
    print(f"  Size: {len(data):,} bytes")
    print(f"{'='*70}")

    # Parse header to find microcode offset
    hdr_size = struct.unpack_from('<I', data, 4)[0] if len(data) > 8 else 32
    ucode_offset = struct.unpack_from('<I', data, 24)[0] if len(data) > 28 else 0x100

    print(f"  Header size: {hdr_size} bytes")
    print(f"  Microcode offset: 0x{ucode_offset:04x}")

    words = extract_words(data, ucode_offset)
    print(f"  Instruction words: {len(words):,}")

    result = {
        'name': name,
        'description': info[0],
        'function': info[1],
        'size': len(data),
        'ucode_offset': ucode_offset,
        'n_words': len(words),
    }

    # 1. PSP envelope
    print(f"\n  --- PSP Signature Envelope ---")
    envelopes = decode_psp_envelope(data)
    result['psp_envelopes'] = envelopes
    for env in envelopes:
        print(f"    $PS1 @ {env['marker_hex']}")
        if 'rsa_sig_offset' in env:
            print(f"      RSA sig @ {env['rsa_sig_offset']} (entropy: {env['rsa_sig_entropy']})")
            print(f"      Payload starts @ {env.get('payload_starts_at', '?')}")
            print(f"      Payload assessment: {env.get('payload_assessment', '?')}")

    # 2. Opcode encoding analysis
    print(f"\n  --- Opcode Encoding Analysis ---")
    encodings = analyze_opcode_encoding(words)
    result['opcode_encoding'] = encodings
    for width, info_dict in encodings.items():
        ent_ratio = info_dict['entropy_ratio']
        quality = 'EXCELLENT' if ent_ratio < 0.3 else 'GOOD' if ent_ratio < 0.5 else 'MODERATE' if ent_ratio < 0.7 else 'POOR'
        print(f"    {width:15s}: {info_dict['n_distinct']:4d}/{info_dict['max_possible']:4d} distinct, "
              f"entropy={info_dict['entropy']:.2f}/{info_dict['max_entropy']:.2f} ({quality})")
        if ent_ratio < 0.5:
            for v in info_dict['top_values'][:5]:
                print(f"      {v['opcode']:>8s}: {v['count']:6d} ({v['pct']:.1f}%)")

    # 3. Branch target analysis
    print(f"\n  --- Branch Target Analysis ---")
    branches = detect_branch_targets(words)
    result['branch_targets'] = branches
    print(f"    Potential targets: {branches['total_potential_targets']}")
    print(f"    Likely targets (≥2 refs): {branches['likely_targets_ge2']}")
    print(f"    Likely functions (≥3 refs): {branches['likely_functions_ge3']}")
    for t in branches['top_targets'][:5]:
        print(f"      @ {t['byte_offset']}: {t['references']} references")

    # 4. Register references
    print(f"\n  --- Known Register References ---")
    reg_refs = extract_register_references(words)
    result['register_references'] = {k: len(v) for k, v in reg_refs.items()}
    for reg_name, entries in list(reg_refs.items())[:15]:
        print(f"    {reg_name:30s}: {len(entries)} references")

    # 5. PM4 opcode handling
    print(f"\n  --- PM4 Opcode References ---")
    pm4_refs = extract_pm4_references(words)
    result['pm4_references'] = {k: len(v) for k, v in pm4_refs.items()}
    for op_name, entries in list(pm4_refs.items())[:15]:
        print(f"    {op_name:30s}: {len(entries)} occurrences")

    # 6. Jump tables
    print(f"\n  --- Jump Table Detection ---")
    jump_tables = detect_jump_tables(words)
    result['jump_tables'] = jump_tables
    if jump_tables:
        for jt in jump_tables[:5]:
            print(f"    Table @ {jt['start_offset']}: {jt['length']} entries "
                  f"({jt['unique_entries']} unique)")
            print(f"      Sample: {', '.join(jt['sample_entries'][:4])}")
    else:
        print(f"    None detected")

    # 7. String constants
    print(f"\n  --- String Constants ---")
    strings = detect_string_constants(data, ucode_offset)
    result['strings'] = strings[:20]
    for s in strings[:10]:
        print(f"    @ {s['offset']}: \"{s['string'][:60]}\"")
    if not strings:
        print(f"    None found")

    # 8. Repeated patterns
    print(f"\n  --- Repeated Instruction Patterns ---")
    patterns = detect_repeated_patterns(words)
    result['repeated_patterns'] = patterns[:10]
    for p in patterns[:5]:
        print(f"    {p['count']}x: {' '.join(p['words'])}")

    # 9. Instruction density
    density = analyze_instruction_density(words)
    result['instruction_density'] = density
    print(f"\n  --- Instruction Density ---")
    print(f"    Code blocks: {density['code_blocks']}/{density['total_blocks']} ({density['code_pct']}%)")

    return result


def main():
    parser = argparse.ArgumentParser(description='AMD GPU Microcode ISA Decoder')
    parser.add_argument('--blob', default=None, help='Single blob path to analyze')
    parser.add_argument('--blob-dir', default='/tmp/fw_dissect', help='Directory of decompressed blobs')
    parser.add_argument('--all', action='store_true', help='Analyze all known blobs')
    parser.add_argument('--output', default=None, help='Output JSON path')
    args = parser.parse_args()

    results = {}

    if args.blob:
        name = Path(args.blob).stem
        data = Path(args.blob).read_bytes()
        results[name] = analyze_blob(name, data)
    elif args.all:
        blob_dir = Path(args.blob_dir)
        for name in BLOB_INFO:
            path = blob_dir / f'{name}.bin'
            if path.exists():
                data = path.read_bytes()
                results[name] = analyze_blob(name, data)
            else:
                print(f"SKIP: {path} not found")
    else:
        # Default: analyze ME (most interesting for PM4 processing)
        blob_dir = Path(args.blob_dir)
        for name in ['gc_11_5_1_me', 'gc_11_5_1_pfp', 'gc_11_5_1_rlc']:
            path = blob_dir / f'{name}.bin'
            if path.exists():
                data = path.read_bytes()
                results[name] = analyze_blob(name, data)

    # Save report
    if args.output is None:
        args.output = str(Path(args.blob_dir).parent / 'results' / 'ucode_isa_report.json'
                          if args.blob is None else
                          Path(args.blob).with_suffix('.isa.json'))

    # Ensure results dir exists
    if 'results' in str(args.output):
        os.makedirs(os.path.dirname(args.output), exist_ok=True)

    report = {
        'timestamp': datetime.now().isoformat(),
        'device': 'AMD Radeon 8060S (gfx1151)',
        'tool': 'ucode_isa_decoder v1.0',
        'blobs': results,
    }

    with open(args.output, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport saved to {args.output}")


if __name__ == '__main__':
    main()
