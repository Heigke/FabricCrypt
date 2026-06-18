#!/usr/bin/env python3
"""
z2338_firmware_disasm.py — MEC Firmware Disassembly + HWREG Mapping (gfx1151)
=============================================================================
PART 1: Disassemble MEC, PFP, ME, RLC firmware — identify ISA, find PM4 tables
PART 2: Deep probe undocumented HWREGs [7,8,9,18,19,27,28]

SAFETY: READ-ONLY. No hardware register writes. No firmware modification.

Run:
  HSA_OVERRIDE_GFX_VERSION=11.0.0 PYTHONUNBUFFERED=1 ./venv/bin/python scripts/z2338_firmware_disasm.py
"""

import os, sys, time, json, struct, collections, math, hashlib
import numpy as np
from pathlib import Path

os.environ['PYTHONUNBUFFERED'] = '1'
os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '11.0.0')

BASE = Path(__file__).resolve().parent.parent
RESULTS = BASE / 'results'
RESULTS.mkdir(exist_ok=True)

TEMP_PAUSE = 75
TEMP_RESUME = 50
TEMP_ABORT = 85

# ======================================================================
# Thermal Safety
# ======================================================================
def get_temp():
    try: return int(open('/sys/class/thermal/thermal_zone0/temp').read()) // 1000
    except: return 0

def check_abort():
    t = get_temp()
    if t >= TEMP_ABORT:
        print(f"\n  [ABORT] Temperature {t}C >= {TEMP_ABORT}C!", flush=True)
        return True
    return False

def wait_cool(label="", target=None):
    if target is None: target = TEMP_RESUME
    t = get_temp()
    if t <= target: return t
    print(f"  [TEMP] {label} {t}C -> {target}C...", end="", flush=True)
    t0 = time.time()
    while t > target and time.time() - t0 < 180:
        time.sleep(3); t = get_temp()
        print(f" {t}", end="", flush=True)
    print(f" OK ({time.time()-t0:.0f}s)")
    return t

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.bool_): return bool(obj)
        if isinstance(obj, bytes): return obj.hex()
        return super().default(obj)

results = {
    'experiment': 'z2338_firmware_disasm',
    'description': 'MEC firmware disassembly + HWREG mapping on gfx1151',
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'parts': {}
}
SAVE_JSON = RESULTS / 'z2338_firmware_disasm.json'

def save_results():
    with open(SAVE_JSON, 'w') as f:
        json.dump(results, f, indent=2, cls=NpEncoder)
    print(f"  [SAVED] {SAVE_JSON}", flush=True)

def save_txt(name, text):
    p = RESULTS / f'z2338_{name}.txt'
    with open(p, 'w') as f:
        f.write(text)
    print(f"  [SAVED] {p}", flush=True)

# ======================================================================
# PART 1: FIRMWARE DISASSEMBLY
# ======================================================================
print("=" * 70)
print("z2338: MEC Firmware Disassembly + HWREG Mapping — gfx1151")
print("=" * 70)

FW_DIR = '/tmp'
FW_FILES = {
    'mec': f'{FW_DIR}/gc_11_5_1_mec.bin',
    'pfp': f'{FW_DIR}/gc_11_5_1_pfp.bin',
    'me':  f'{FW_DIR}/gc_11_5_1_me.bin',
    'rlc': f'{FW_DIR}/gc_11_5_1_rlc.bin',
}

# Decompress if needed
import subprocess
for name, path in FW_FILES.items():
    if not os.path.exists(path):
        zst = f'/lib/firmware/amdgpu/gc_11_5_1_{name}.bin.zst'
        if os.path.exists(zst):
            subprocess.run(['zstd', '-d', zst, '-o', path, '-f'], check=True)
            print(f"  Decompressed {zst} -> {path}")

# Known PM4 type-3 opcodes (from amdgpu driver headers)
PM4_OPCODES = {
    0x10: 'NOP',
    0x11: 'SET_BASE',
    0x12: 'CLEAR_STATE',
    0x15: 'DISPATCH_DIRECT',
    0x16: 'DISPATCH_INDIRECT',
    0x20: 'ATOMIC_GDS',
    0x21: 'ATOMIC_MEM',
    0x22: 'OCCLUSION_QUERY',
    0x23: 'SET_PREDICATION',
    0x24: 'REG_RMW',
    0x25: 'COND_EXEC',
    0x26: 'PRED_EXEC',
    0x27: 'DRAW_INDIRECT',
    0x28: 'DRAW_INDEX_INDIRECT',
    0x29: 'INDEX_BASE',
    0x2A: 'DRAW_INDEX_2',
    0x2C: 'CONTEXT_CONTROL',
    0x2D: 'INDEX_TYPE',
    0x30: 'DRAW_INDIRECT_MULTI',
    0x31: 'DRAW_INDEX_AUTO',
    0x33: 'NUM_INSTANCES',
    0x34: 'DRAW_INDEX_MULTI_INST',
    0x35: 'INDIRECT_BUFFER_CNST',  # v1
    0x36: 'STRMOUT_BUFFER_UPDATE',
    0x37: 'DRAW_INDEX_OFFSET_2',
    0x3F: 'INDIRECT_BUFFER',
    0x40: 'COPY_DATA',
    0x41: 'CP_DMA',
    0x42: 'PFP_SYNC_ME',
    0x43: 'SURFACE_SYNC',
    0x44: 'ME_INITIALIZE',
    0x45: 'COND_WRITE',
    0x46: 'EVENT_WRITE',
    0x47: 'EVENT_WRITE_EOP',
    0x48: 'EVENT_WRITE_EOS',
    0x49: 'RELEASE_MEM',
    0x50: 'PREAMBLE_CNTL',
    0x58: 'DISPATCH_MESH_INDIRECT_MULTI',
    0x59: 'DISPATCH_TASKMESH_GFX',
    0x5F: 'DMA_DATA',
    0x68: 'CONTEXT_REG_RMW',
    0x69: 'GFX_CNTX_UPDATE',
    0x6C: 'ONE_REG_WRITE',
    0x73: 'ACQUIRE_MEM',
    0x76: 'SET_SH_REG',
    0x77: 'SET_SH_REG_OFFSET',
    0x78: 'SET_QUEUE_REG',
    0x79: 'SET_UCONFIG_REG',
    0x7A: 'SET_UCONFIG_REG_INDEX',
    0x80: 'LOAD_CONST_RAM',
    0x81: 'WRITE_CONST_RAM',
    0x82: 'DUMP_CONST_RAM',
    0x83: 'INCREMENT_CE_COUNTER',
    0x84: 'INCREMENT_DE_COUNTER',
    0x85: 'WAIT_ON_CE_COUNTER',
    0x86: 'WAIT_ON_DE_COUNTER_DIFF',
    0x90: 'WAIT_REG_MEM',
    0x91: 'MEM_WRITE',
    0xA0: 'REWIND',
    0xA2: 'LOAD_UCONFIG_REG',
    0xA4: 'LOAD_SH_REG',
    0xA5: 'LOAD_SH_REG_INDEX',
    0xA9: 'LOAD_CONTEXT_REG',
    0xAA: 'LOAD_CONTEXT_REG_INDEX',
    0xB8: 'SET_SH_REG_INDEX',
    0xD5: 'SET_CONTEXT_REG',  # GFX11+
    0xD6: 'SET_CONTEXT_REG_INDEX',
}

# Known GFX register offsets that might appear in firmware
KNOWN_REGS = {
    0x2C00: 'COMPUTE_DISPATCH_INITIATOR',
    0x2C04: 'COMPUTE_DIM_X',
    0x2C08: 'COMPUTE_DIM_Y',
    0x2C0C: 'COMPUTE_DIM_Z',
    0x2C10: 'COMPUTE_START_X',
    0x2C14: 'COMPUTE_START_Y',
    0x2C18: 'COMPUTE_START_Z',
    0x2C1C: 'COMPUTE_NUM_THREAD_X',
    0x2C20: 'COMPUTE_NUM_THREAD_Y',
    0x2C24: 'COMPUTE_NUM_THREAD_Z',
    0x2C28: 'COMPUTE_PIPELINESTAT_ENABLE',
    0x2C2C: 'COMPUTE_PERFCOUNT_ENABLE',
    0x2C30: 'COMPUTE_PGM_LO',
    0x2C34: 'COMPUTE_PGM_HI',
    0x2C38: 'COMPUTE_DISPATCH_PKT_ADDR_LO',
    0x2C44: 'COMPUTE_PGM_RSRC1',
    0x2C48: 'COMPUTE_PGM_RSRC2',
    0x2C4C: 'COMPUTE_VMID',
    0x2C50: 'COMPUTE_RESOURCE_LIMITS',
    0x2C54: 'COMPUTE_STATIC_THREAD_MGMT_SE0',
    0x2C58: 'COMPUTE_STATIC_THREAD_MGMT_SE1',
    0x2C7C: 'COMPUTE_DISPATCH_ID',
    0x2C80: 'COMPUTE_THREADGROUP_ID',
    0x2C88: 'COMPUTE_RELAUNCH',
    0x2C8C: 'COMPUTE_WAVE_RESTORE_ADDR_LO',
    0x2C90: 'COMPUTE_WAVE_RESTORE_ADDR_HI',
    0x2C94: 'COMPUTE_WAVE_RESTORE_CONTROL',
    0x30A0: 'CP_MEC_CNTL',
    0x30A4: 'CP_MEC_ME1_HEADER_DUMP',
    0x30A8: 'CP_MEC_ME2_HEADER_DUMP',
    0x3100: 'CP_HQD_ACTIVE',
    0x3104: 'CP_HQD_VMID',
    0x3108: 'CP_HQD_PERSISTENT_STATE',
    0x310C: 'CP_HQD_PIPE_PRIORITY',
    0x3110: 'CP_HQD_QUEUE_PRIORITY',
    0x3114: 'CP_HQD_QUANTUM',
    0x3118: 'CP_HQD_PQ_BASE',
    0x311C: 'CP_HQD_PQ_BASE_HI',
    0x3120: 'CP_HQD_PQ_RPTR',
    0x3124: 'CP_HQD_PQ_RPTR_REPORT_ADDR',
    0x3128: 'CP_HQD_PQ_RPTR_REPORT_ADDR_HI',
    0x312C: 'CP_HQD_PQ_WPTR_POLL_ADDR',
    0x3130: 'CP_HQD_PQ_WPTR_POLL_ADDR_HI',
    0x3134: 'CP_HQD_PQ_DOORBELL_CONTROL',
    0x3138: 'CP_HQD_PQ_CONTROL',
    0x313C: 'CP_HQD_IB_BASE_ADDR',
    0x3140: 'CP_HQD_IB_BASE_ADDR_HI',
    0x3144: 'CP_HQD_IB_RPTR',
    0x3148: 'CP_HQD_IB_CONTROL',
    0x314C: 'CP_HQD_DEQUEUE_REQUEST',
    0x3150: 'CP_HQD_DMA_OFFLOAD',
    0x3154: 'CP_HQD_SEMA_CMD',
    0x3158: 'CP_HQD_MSG_TYPE',
    0x315C: 'CP_HQD_ATOMIC0_PREOP_LO',
    0x3160: 'CP_HQD_ATOMIC0_PREOP_HI',
    0x3164: 'CP_HQD_ATOMIC1_PREOP_LO',
    0x3168: 'CP_HQD_ATOMIC1_PREOP_HI',
    0x316C: 'CP_HQD_HQ_STATUS0',
    0x3170: 'CP_HQD_HQ_CONTROL0',
    0x3174: 'CP_HQD_CTX_SAVE_BASE_ADDR_LO',
    0x3178: 'CP_HQD_CTX_SAVE_BASE_ADDR_HI',
    0x317C: 'CP_HQD_CTX_SAVE_CONTROL',
    0x3180: 'CP_HQD_CNTL_STACK_OFFSET',
    0x3184: 'CP_HQD_CNTL_STACK_SIZE',
    0x3188: 'CP_HQD_WG_STATE_OFFSET',
    0x318C: 'CP_HQD_CTX_SAVE_SIZE',
    0x3190: 'CP_HQD_GDS_RESOURCE_STATE',
    0x3194: 'CP_HQD_ERROR',
    0x3198: 'CP_HQD_EOP_BASE_ADDR',
    0x319C: 'CP_HQD_EOP_BASE_ADDR_HI',
    0x31A0: 'CP_HQD_EOP_CONTROL',
    0x31A4: 'CP_HQD_EOP_RPTR',
    0x31A8: 'CP_HQD_EOP_WPTR',
    0x31AC: 'CP_HQD_EOP_DONE_EVENTS',
    0x31B0: 'CP_HQD_PQ_WPTR_LO',
    0x31B4: 'CP_HQD_PQ_WPTR_HI',
}


def analyze_firmware(name, path):
    """Analyze a single firmware binary."""
    print(f"\n  --- Analyzing {name}: {path} ---", flush=True)

    if not os.path.exists(path):
        return {'error': f'File not found: {path}'}

    data = open(path, 'rb').read()
    size = len(data)
    sha256 = hashlib.sha256(data).hexdigest()

    info = {
        'file': path,
        'size_bytes': size,
        'sha256': sha256,
    }

    # --- Parse firmware header (common_firmware_header_v1) ---
    if size < 0x20:
        info['error'] = 'Too small for header'
        return info

    hdr = struct.unpack_from('<IIHHHHI I I', data, 0)
    # Fields: size_bytes, header_size_bytes, header_version_major, header_version_minor,
    #         ip_version_major, ip_version_minor, ucode_version, ucode_size_bytes,
    #         ucode_array_offset_bytes
    # But actual struct may differ. Let's read carefully:
    hdr_size = struct.unpack_from('<I', data, 0x00)[0]
    hdr_hdr_size = struct.unpack_from('<I', data, 0x04)[0]
    hdr_ver_major = struct.unpack_from('<H', data, 0x08)[0]
    hdr_ver_minor = struct.unpack_from('<H', data, 0x0A)[0]
    ip_ver_major = struct.unpack_from('<H', data, 0x0C)[0]
    ip_ver_minor = struct.unpack_from('<H', data, 0x0E)[0]
    ucode_ver = struct.unpack_from('<I', data, 0x10)[0]
    ucode_size = struct.unpack_from('<I', data, 0x14)[0]
    ucode_offset = struct.unpack_from('<I', data, 0x18)[0]
    crc32 = struct.unpack_from('<I', data, 0x1C)[0]

    info['header'] = {
        'size_bytes': hdr_size,
        'header_size_bytes': hdr_hdr_size,
        'header_version': f'{hdr_ver_major}.{hdr_ver_minor}',
        'ip_version': f'{ip_ver_major}.{ip_ver_minor}',
        'ucode_version': f'0x{ucode_ver:08X}',
        'ucode_size_bytes': ucode_size,
        'ucode_array_offset': ucode_offset,
        'ucode_array_offset_hex': f'0x{ucode_offset:04X}',
        'crc32': f'0x{crc32:08X}',
    }

    # Check for $PS1 header
    ps1_offset = None
    for off in [0x100, 0x110, 0x120, 0x80, 0x90, 0xA0]:
        if off + 4 <= size and data[off:off+4] == b'$PS1':
            ps1_offset = off
            break
    # Also try at ucode_offset - 256 or similar
    if ps1_offset is None:
        idx = data.find(b'$PS1')
        if idx >= 0:
            ps1_offset = idx

    info['ps1_offset'] = ps1_offset
    info['ps1_offset_hex'] = f'0x{ps1_offset:04X}' if ps1_offset is not None else None

    # Parse $PS1 header if found
    if ps1_offset is not None and ps1_offset + 64 <= size:
        ps1_data = data[ps1_offset:ps1_offset+64]
        info['ps1_header'] = {
            'magic': ps1_data[0:4].decode('ascii', errors='replace'),
            'bytes_4_7': ps1_data[4:8].hex(),
            'bytes_8_11': ps1_data[8:12].hex(),
            'bytes_12_15': ps1_data[12:16].hex(),
            'bytes_16_19': ps1_data[16:20].hex(),
            'first_64_hex': ps1_data.hex(),
        }

    # Find actual microcode start
    # The ucode_array_offset typically points past the common header
    # But there may also be a $PS1 wrapper. The actual ISA starts after all headers.
    # For GFX11 MEC: typically ucode_offset = 0x100 and $PS1 is at 0x110
    # The microcode may start after the $PS1 signature block

    # Try to find where the actual non-zero instruction data begins
    ucode_start = ucode_offset if ucode_offset < size else 0

    # Check if there's a $PS1 block after the ucode_offset
    if ps1_offset is not None and ps1_offset >= ucode_start:
        # $PS1 is part of the ucode region — actual code starts after PS1 header
        # PS1 header is typically 256 bytes (signing block)
        potential_starts = [ps1_offset + 256, ps1_offset + 512, ps1_offset + 128]
        for ps in potential_starts:
            if ps < size:
                # Check if there's non-zero data here
                chunk = data[ps:min(ps+16, size)]
                if any(b != 0 for b in chunk):
                    ucode_start = ps
                    break

    # Also try: skip leading zeros from ucode_offset
    test_start = ucode_offset
    while test_start < size - 4:
        word = struct.unpack_from('<I', data, test_start)[0]
        if word != 0:
            break
        test_start += 4

    # Use the earlier of the two methods (non-zero data start)
    actual_code_start = min(test_start, ucode_start) if ucode_start > 0 else test_start

    info['ucode_actual_start'] = actual_code_start
    info['ucode_actual_start_hex'] = f'0x{actual_code_start:04X}'

    payload = data[actual_code_start:]
    payload_size = len(payload)
    info['payload_size'] = payload_size

    # --- Entropy analysis of payload ---
    if payload_size > 0:
        byte_counts = collections.Counter(payload)
        total = payload_size
        entropy = -sum((c/total) * math.log2(c/total) for c in byte_counts.values() if c > 0)
        zero_pct = byte_counts.get(0, 0) / total * 100
        info['payload_entropy'] = round(entropy, 3)
        info['payload_zero_pct'] = round(zero_pct, 1)

    # --- Instruction word analysis (32-bit words) ---
    n_words = payload_size // 4
    if n_words > 0:
        words = struct.unpack_from(f'<{n_words}I', payload)
        words_arr = np.array(words, dtype=np.uint32)

        # Top byte histogram (potential opcodes)
        top_bytes = (words_arr >> 24).astype(np.uint8)
        top_byte_counts = collections.Counter(top_bytes.tolist())
        info['top_byte_histogram'] = {f'0x{k:02X}': v for k, v in
                                       sorted(top_byte_counts.items(), key=lambda x: -x[1])[:20]}

        # Most common 32-bit words
        word_counts = collections.Counter(words)
        info['most_common_words'] = {f'0x{k:08X}': v for k, v in
                                      word_counts.most_common(30)}

        # Non-zero words
        non_zero = words_arr[words_arr != 0]
        info['non_zero_words'] = len(non_zero)
        info['total_words'] = n_words
        info['non_zero_pct'] = round(len(non_zero) / n_words * 100, 1)

        # --- Look for PM4 opcode references ---
        # In the MEC microcode, PM4 opcodes may appear as constants loaded into registers
        # They typically appear as the opcode value shifted or in dispatch tables
        pm4_refs = {}
        for opcode, opname in PM4_OPCODES.items():
            # Search for the opcode value in various encodings
            count = 0
            for w in words:
                # Direct match in lower byte
                if (w & 0xFF) == opcode:
                    count += 1
                # Match in byte 1
                if ((w >> 8) & 0xFF) == opcode:
                    count += 1
            # Only report if found more than noise threshold
            if count > 0:
                pm4_refs[f'0x{opcode:02X}_{opname}'] = count
        info['pm4_opcode_references'] = pm4_refs

        # --- Look for specific PM4 opcodes as immediate values ---
        pm4_dispatch_table = []
        for i, w in enumerate(words):
            for opcode, opname in PM4_OPCODES.items():
                if w == opcode:
                    pm4_dispatch_table.append({
                        'offset': actual_code_start + i*4,
                        'offset_hex': f'0x{actual_code_start + i*4:06X}',
                        'opcode': f'0x{opcode:02X}',
                        'name': opname,
                        'context_before': f'0x{words[max(0,i-1)]:08X}' if i > 0 else None,
                        'context_after': f'0x{words[min(len(words)-1,i+1)]:08X}' if i < len(words)-1 else None,
                    })
        info['pm4_exact_matches'] = pm4_dispatch_table[:50]  # limit

        # --- Look for known register addresses ---
        reg_refs = {}
        for i, w in enumerate(words):
            # Check if the word matches a known register offset
            if w in KNOWN_REGS:
                regname = KNOWN_REGS[w]
                if regname not in reg_refs:
                    reg_refs[regname] = []
                reg_refs[regname].append(f'0x{actual_code_start + i*4:06X}')
            # Also check shifted versions (dword offset = register_offset >> 2)
            w_shifted = w << 2
            if w_shifted in KNOWN_REGS:
                regname = KNOWN_REGS[w_shifted] + ' (>>2)'
                if regname not in reg_refs:
                    reg_refs[regname] = []
                reg_refs[regname].append(f'0x{actual_code_start + i*4:06X}')
        info['register_references'] = {k: v[:5] for k, v in sorted(reg_refs.items())}

        # --- Look for HWREG ID references (7, 18, 19, 27, 28) ---
        # In the F32 ISA, HWREG reads use s_getreg encoding: HWREG(id, offset, size)
        # The encoding is: id | (offset << 6) | ((size-1) << 11)
        # For full 32-bit reads: id | (0 << 6) | (31 << 11) = id | 0xF800
        hwreg_pattern_refs = {}
        target_hwregs = [7, 8, 9, 18, 19, 27, 28]
        for hid in target_hwregs:
            full_read_imm = hid | 0xF800  # HWREG(hid, 0, 32) encoding
            count = 0
            locations = []
            for i, w in enumerate(words):
                if (w & 0xFFFF) == full_read_imm or (w >> 16) == full_read_imm:
                    count += 1
                    locations.append(f'0x{actual_code_start + i*4:06X}')
                # Also check just the ID in lower bits
                if (w & 0x3F) == hid:
                    count += 1
            hwreg_pattern_refs[f'HWREG_{hid}'] = {
                'full_read_matches': len(locations),
                'id_in_low6_count': count,
                'locations': locations[:10],
            }
        info['hwreg_pattern_references'] = hwreg_pattern_refs

        # --- Try to identify ISA type ---
        # F32 ISA characteristics:
        #   - 32-bit fixed-width instructions
        #   - Top bits encode opcode
        #   - Many NOPs (0x00000000)
        #   - Branch instructions typically have top nibble 0x1

        # ARM Thumb characteristics:
        #   - Mix of 16-bit and 32-bit instructions
        #   - Thumb-2 32-bit start with 0xE, 0xF in bits[15:12]

        # RISC-V characteristics:
        #   - Bottom 2 bits = 0b11 for 32-bit instructions
        #   - Common opcodes: 0x13 (ADDI), 0x33 (ADD), 0x63 (Branch), 0x6F (JAL)

        riscv_count = sum(1 for w in words if (w & 0x3) == 0x3 and w != 0)
        arm_count = sum(1 for w in words if w != 0 and ((w >> 28) == 0xE or (w >> 12) & 0xF in [0xE, 0xF]))

        # Check for F32 pattern: lots of 0x00000000 (NOP) and structured top nibbles
        nop_count = sum(1 for w in words if w == 0)

        # For MEC F32: instruction format varies. Check if high byte distribution is structured
        high_nibble_counts = collections.Counter((w >> 28) & 0xF for w in words if w != 0)

        info['isa_analysis'] = {
            'total_words': n_words,
            'nop_zero_words': nop_count,
            'nop_pct': round(nop_count / n_words * 100, 1),
            'riscv_signature_count': riscv_count,
            'riscv_pct': round(riscv_count / max(1, n_words - nop_count) * 100, 1),
            'arm_like_count': arm_count,
            'high_nibble_distribution': {f'0x{k:X}': v for k, v in sorted(high_nibble_counts.items())},
        }

        # --- ASCII string search ---
        strings_found = []
        ascii_buf = []
        for i, b in enumerate(payload):
            if 0x20 <= b <= 0x7E:
                ascii_buf.append(chr(b))
            else:
                if len(ascii_buf) >= 6:
                    strings_found.append({
                        'offset': actual_code_start + i - len(ascii_buf),
                        'offset_hex': f'0x{actual_code_start + i - len(ascii_buf):06X}',
                        'string': ''.join(ascii_buf),
                    })
                ascii_buf = []
        if len(ascii_buf) >= 6:
            strings_found.append({
                'offset': actual_code_start + len(payload) - len(ascii_buf),
                'string': ''.join(ascii_buf),
            })
        info['ascii_strings'] = strings_found[:50]

        # --- Look for jump/dispatch table pattern ---
        # A dispatch table would be a sequence of addresses or offsets
        # Look for sequences where consecutive words increment by a regular amount
        table_candidates = []
        for i in range(len(words) - 8):
            # Check if 8+ consecutive non-zero words form an arithmetic sequence
            seq = [words[i+j] for j in range(8)]
            if all(s != 0 for s in seq):
                diffs = [seq[j+1] - seq[j] for j in range(7)]
                if all(d == diffs[0] for d in diffs) and diffs[0] != 0 and abs(diffs[0]) < 0x1000:
                    table_candidates.append({
                        'offset': actual_code_start + i*4,
                        'offset_hex': f'0x{actual_code_start + i*4:06X}',
                        'stride': diffs[0],
                        'first_value': f'0x{seq[0]:08X}',
                        'sample': [f'0x{s:08X}' for s in seq],
                    })
        info['potential_tables'] = table_candidates[:20]

        # --- Hex dump of first 256 bytes of actual code ---
        dump_size = min(256, payload_size)
        hex_lines = []
        for off in range(0, dump_size, 16):
            hex_bytes = ' '.join(f'{payload[off+j]:02X}' for j in range(min(16, dump_size - off)))
            ascii_repr = ''.join(chr(payload[off+j]) if 0x20 <= payload[off+j] <= 0x7E else '.'
                                  for j in range(min(16, dump_size - off)))
            hex_lines.append(f'  {actual_code_start + off:06X}: {hex_bytes:<48s} {ascii_repr}')
        info['hex_dump_first_256'] = hex_lines

        # --- Also dump around any interesting patterns ---
        # Look for the magic values from undocumented HWREGs
        magic_values = [0xB8E3B9D4, 0xB8E3BD30, 0x0000740A, 0x00000603]
        magic_refs = {}
        for mv in magic_values:
            locs = []
            for i, w in enumerate(words):
                if w == mv:
                    locs.append(f'0x{actual_code_start + i*4:06X}')
            if locs:
                magic_refs[f'0x{mv:08X}'] = locs
        info['magic_value_references'] = magic_refs

    return info


print("\n[PART 1] Firmware Binary Analysis", flush=True)
print("=" * 50, flush=True)

fw_results = {}
for name, path in FW_FILES.items():
    try:
        fw_results[name] = analyze_firmware(name, path)
        print(f"  {name}: {fw_results[name].get('payload_size', 0)} bytes payload, "
              f"entropy={fw_results[name].get('payload_entropy', '?')}, "
              f"non-zero={fw_results[name].get('non_zero_pct', '?')}%", flush=True)
    except Exception as e:
        fw_results[name] = {'error': str(e)}
        print(f"  {name}: ERROR {e}", flush=True)

results['parts']['firmware_analysis'] = fw_results
save_results()

# --- Generate text report ---
report_lines = []
report_lines.append("=" * 70)
report_lines.append("z2338 PART 1: MEC FIRMWARE DISASSEMBLY REPORT")
report_lines.append(f"GPU: AMD Radeon 8060S (gfx1151, Strix Halo)")
report_lines.append(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
report_lines.append("METHOD: READ-ONLY analysis of firmware binaries")
report_lines.append("=" * 70)

for name in ['mec', 'pfp', 'me', 'rlc']:
    fw = fw_results.get(name, {})
    report_lines.append(f"\n{'='*50}")
    report_lines.append(f"  {name.upper()} Firmware: gc_11_5_1_{name}.bin")
    report_lines.append(f"{'='*50}")
    report_lines.append(f"  Size: {fw.get('size_bytes', '?')} bytes")
    report_lines.append(f"  SHA256: {fw.get('sha256', '?')[:32]}...")

    hdr = fw.get('header', {})
    report_lines.append(f"  Header version: {hdr.get('header_version', '?')}")
    report_lines.append(f"  IP version: {hdr.get('ip_version', '?')}")
    report_lines.append(f"  Ucode version: {hdr.get('ucode_version', '?')}")
    report_lines.append(f"  Ucode size: {hdr.get('ucode_size_bytes', '?')} bytes")
    report_lines.append(f"  Ucode offset: {hdr.get('ucode_array_offset_hex', '?')}")
    report_lines.append(f"  $PS1 at: {fw.get('ps1_offset_hex', 'NOT FOUND')}")
    report_lines.append(f"  Actual code start: {fw.get('ucode_actual_start_hex', '?')}")
    report_lines.append(f"  Payload entropy: {fw.get('payload_entropy', '?')}")
    report_lines.append(f"  Zero bytes: {fw.get('payload_zero_pct', '?')}%")

    isa = fw.get('isa_analysis', {})
    if isa:
        report_lines.append(f"\n  ISA Analysis:")
        report_lines.append(f"    Total instruction words: {isa.get('total_words', '?')}")
        report_lines.append(f"    NOP (zero) words: {isa.get('nop_zero_words', '?')} ({isa.get('nop_pct', '?')}%)")
        report_lines.append(f"    RISC-V signature: {isa.get('riscv_pct', '?')}%")
        report_lines.append(f"    High nibble distribution: {isa.get('high_nibble_distribution', {})}")

    # Most common words
    common = fw.get('most_common_words', {})
    if common:
        report_lines.append(f"\n  Top 10 most common instruction words:")
        for i, (w, c) in enumerate(list(common.items())[:10]):
            report_lines.append(f"    {w}: {c} occurrences")

    # Register references
    regs = fw.get('register_references', {})
    if regs:
        report_lines.append(f"\n  Known register references found:")
        for rname, locs in list(regs.items())[:20]:
            report_lines.append(f"    {rname}: {locs}")

    # PM4 exact matches
    pm4 = fw.get('pm4_exact_matches', [])
    if pm4:
        report_lines.append(f"\n  PM4 opcode exact matches (value == opcode):")
        for m in pm4[:15]:
            report_lines.append(f"    {m['offset_hex']}: {m['name']} ({m['opcode']})")

    # HWREG pattern refs
    hwreg_p = fw.get('hwreg_pattern_references', {})
    if hwreg_p:
        report_lines.append(f"\n  Undocumented HWREG pattern references:")
        for hname, hdata in hwreg_p.items():
            if hdata['full_read_matches'] > 0:
                report_lines.append(f"    {hname}: {hdata['full_read_matches']} full-read matches at {hdata['locations']}")

    # Magic values
    magic = fw.get('magic_value_references', {})
    if magic:
        report_lines.append(f"\n  Magic value references (from undocumented HWREGs):")
        for mv, locs in magic.items():
            report_lines.append(f"    {mv}: found at {locs}")

    # Strings
    strings = fw.get('ascii_strings', [])
    if strings:
        report_lines.append(f"\n  ASCII strings found ({len(strings)}):")
        for s in strings[:15]:
            report_lines.append(f"    {s.get('offset_hex', '')}: \"{s['string']}\"")

    # Hex dump
    hexdump = fw.get('hex_dump_first_256', [])
    if hexdump:
        report_lines.append(f"\n  Hex dump (first 256 bytes of code):")
        for line in hexdump[:16]:
            report_lines.append(line)

    # Tables
    tables = fw.get('potential_tables', [])
    if tables:
        report_lines.append(f"\n  Potential dispatch/jump tables:")
        for t in tables[:5]:
            report_lines.append(f"    {t['offset_hex']}: stride={t['stride']}, start={t['first_value']}")

report_text = '\n'.join(report_lines)
save_txt('mec_disasm', report_text)
print(f"\n  Part 1 complete. Report saved.", flush=True)

# ======================================================================
# PART 2: DEEP HWREG PROBING
# ======================================================================
print("\n[PART 2] Deep HWREG Probing", flush=True)
print("=" * 50, flush=True)

wait_cool("Pre-HWREG probe")

import torch
from torch.utils.cpp_extension import load_inline

HWREG_PROBE_SRC = r'''
#include <torch/extension.h>

#define HWREG(id, offset, size) ((id) | ((offset) << 6) | (((size)-1) << 11))

#define HW_REG_HW_ID1       23
#define HW_REG_HW_ID2       24
#define HW_REG_SHADER_CYCLES 29

// =====================================================================
// P1: Bitfield extraction of HWREG[7] — correlate with topology
// Read HWREG[7] as 4 bytes + HW_ID1/HW_ID2 for correlation
// Output per wave: [byte0, byte1, byte2, byte3, hw_id1, hw_id2, se, sa, wgp, simd, wave_in_simd]
// =====================================================================
__global__ void kernel_hwreg7_topology(unsigned int* out, int n_waves) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int wave_id = tid / 32;
    if (wave_id >= n_waves) return;
    if (threadIdx.x % 32 != 0) return;

    unsigned int reg7 = __builtin_amdgcn_s_getreg(HWREG(7, 0, 32));
    unsigned int hw_id1 = __builtin_amdgcn_s_getreg(HWREG(HW_REG_HW_ID1, 0, 32));
    unsigned int hw_id2 = __builtin_amdgcn_s_getreg(HWREG(HW_REG_HW_ID2, 0, 32));

    int base = wave_id * 11;
    out[base + 0] = reg7 & 0xFF;           // byte 0
    out[base + 1] = (reg7 >> 8) & 0xFF;    // byte 1
    out[base + 2] = (reg7 >> 16) & 0xFF;   // byte 2
    out[base + 3] = (reg7 >> 24) & 0xFF;   // byte 3
    out[base + 4] = hw_id1;
    out[base + 5] = hw_id2;
    // Decode HW_ID1 fields
    out[base + 6] = (hw_id1 >> 18) & 0x7;  // SE_ID [20:18]
    out[base + 7] = (hw_id1 >> 16) & 0x1;  // SA_ID [16]
    out[base + 8] = (hw_id1 >> 10) & 0x3F; // WGP_ID [15:10]
    out[base + 9] = (hw_id1 >> 8) & 0x3;   // SIMD_ID [9:8]
    out[base + 10] = hw_id1 & 0x3F;        // WAVE_ID [5:0]
}

torch::Tensor probe_hwreg7_topology(int n_waves) {
    auto out = torch::zeros({n_waves * 11}, torch::device(torch::kCUDA).dtype(torch::kInt32));
    kernel_hwreg7_topology<<<n_waves, 32>>>((unsigned int*)out.data_ptr<int>(), n_waves);
    return out.reshape({n_waves, 11});
}

// =====================================================================
// P2: HWREG[18]+[19] — TMA_HI and next register, check for 64-bit addresses
// Also read TMA_LO (17) for completeness
// Output per wave: [tma_lo, tma_hi, reg19, hw_id1]
// =====================================================================
__global__ void kernel_hwreg_tma(unsigned int* out, int n_waves) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int wave_id = tid / 32;
    if (wave_id >= n_waves) return;
    if (threadIdx.x % 32 != 0) return;

    unsigned int tma_lo = __builtin_amdgcn_s_getreg(HWREG(17, 0, 32));
    unsigned int tma_hi = __builtin_amdgcn_s_getreg(HWREG(18, 0, 32));
    unsigned int reg19 = __builtin_amdgcn_s_getreg(HWREG(19, 0, 32));
    unsigned int hw_id1 = __builtin_amdgcn_s_getreg(HWREG(HW_REG_HW_ID1, 0, 32));

    int base = wave_id * 4;
    out[base + 0] = tma_lo;
    out[base + 1] = tma_hi;
    out[base + 2] = reg19;
    out[base + 3] = hw_id1;
}

torch::Tensor probe_hwreg_tma(int n_waves) {
    auto out = torch::zeros({n_waves * 4}, torch::device(torch::kCUDA).dtype(torch::kInt32));
    kernel_hwreg_tma<<<n_waves, 32>>>((unsigned int*)out.data_ptr<int>(), n_waves);
    return out.reshape({n_waves, 4});
}

// =====================================================================
// P3: HWREG[27] temporal stability — read it 4 times with delays
// Also read SHADER_CYCLES to measure the delay between reads
// Output per wave: [r27_t0, r27_t1, r27_t2, r27_t3, cycles_t0, cycles_t3, hw_id1]
// =====================================================================
__global__ void kernel_hwreg27_temporal(unsigned int* out, int n_waves) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int wave_id = tid / 32;
    if (wave_id >= n_waves) return;
    if (threadIdx.x % 32 != 0) return;

    unsigned int c0 = __builtin_amdgcn_s_getreg(HWREG(29, 0, 32));  // SHADER_CYCLES
    unsigned int r0 = __builtin_amdgcn_s_getreg(HWREG(27, 0, 32));

    // Busy-wait ~100 cycles
    for (volatile int i = 0; i < 25; i++) { asm volatile("s_nop 0"); }
    unsigned int r1 = __builtin_amdgcn_s_getreg(HWREG(27, 0, 32));

    // Wait more
    for (volatile int i = 0; i < 100; i++) { asm volatile("s_nop 0"); }
    unsigned int r2 = __builtin_amdgcn_s_getreg(HWREG(27, 0, 32));

    // Wait even more
    for (volatile int i = 0; i < 500; i++) { asm volatile("s_nop 0"); }
    unsigned int r3 = __builtin_amdgcn_s_getreg(HWREG(27, 0, 32));
    unsigned int c3 = __builtin_amdgcn_s_getreg(HWREG(29, 0, 32));

    unsigned int hw_id1 = __builtin_amdgcn_s_getreg(HWREG(HW_REG_HW_ID1, 0, 32));

    int base = wave_id * 7;
    out[base + 0] = r0;
    out[base + 1] = r1;
    out[base + 2] = r2;
    out[base + 3] = r3;
    out[base + 4] = c0;
    out[base + 5] = c3;
    out[base + 6] = hw_id1;
}

torch::Tensor probe_hwreg27_temporal(int n_waves) {
    auto out = torch::zeros({n_waves * 7}, torch::device(torch::kCUDA).dtype(torch::kInt32));
    kernel_hwreg27_temporal<<<n_waves, 32>>>((unsigned int*)out.data_ptr<int>(), n_waves);
    return out.reshape({n_waves, 7});
}

// =====================================================================
// P4: HWREG[28] bitfield decode + HWREG[8], [9] analysis
// Output: [reg8, reg9, reg28, reg28_byte0, reg28_byte1, reg28_byte2, reg28_byte3]
// =====================================================================
__global__ void kernel_hwreg_misc(unsigned int* out, int n_waves) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int wave_id = tid / 32;
    if (wave_id >= n_waves) return;
    if (threadIdx.x % 32 != 0) return;

    unsigned int r8 = __builtin_amdgcn_s_getreg(HWREG(8, 0, 32));
    unsigned int r9 = __builtin_amdgcn_s_getreg(HWREG(9, 0, 32));
    unsigned int r28 = __builtin_amdgcn_s_getreg(HWREG(28, 0, 32));
    unsigned int hw_id1 = __builtin_amdgcn_s_getreg(HWREG(HW_REG_HW_ID1, 0, 32));

    int base = wave_id * 7;
    out[base + 0] = r8;
    out[base + 1] = r9;
    out[base + 2] = r28;
    out[base + 3] = r28 & 0xFF;
    out[base + 4] = (r28 >> 8) & 0xFF;
    out[base + 5] = (r28 >> 16) & 0xFF;
    out[base + 6] = hw_id1;
}

torch::Tensor probe_hwreg_misc(int n_waves) {
    auto out = torch::zeros({n_waves * 7}, torch::device(torch::kCUDA).dtype(torch::kInt32));
    kernel_hwreg_misc<<<n_waves, 32>>>((unsigned int*)out.data_ptr<int>(), n_waves);
    return out.reshape({n_waves, 7});
}

// =====================================================================
// P5: Dispatch-size sensitivity — read undocumented regs with different grid sizes
// This kernel reads all undocumented regs. Launch with different block counts.
// Output per wave: [r7, r8, r9, r18, r19, r27, r28, cycles, hw_id1, hw_id2]
// =====================================================================
__global__ void kernel_hwreg_all_undoc(unsigned int* out, int n_waves) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int wave_id = tid / 32;
    if (wave_id >= n_waves) return;
    if (threadIdx.x % 32 != 0) return;

    int base = wave_id * 10;
    out[base + 0] = __builtin_amdgcn_s_getreg(HWREG(7, 0, 32));
    out[base + 1] = __builtin_amdgcn_s_getreg(HWREG(8, 0, 32));
    out[base + 2] = __builtin_amdgcn_s_getreg(HWREG(9, 0, 32));
    out[base + 3] = __builtin_amdgcn_s_getreg(HWREG(18, 0, 32));
    out[base + 4] = __builtin_amdgcn_s_getreg(HWREG(19, 0, 32));
    out[base + 5] = __builtin_amdgcn_s_getreg(HWREG(27, 0, 32));
    out[base + 6] = __builtin_amdgcn_s_getreg(HWREG(28, 0, 32));
    out[base + 7] = __builtin_amdgcn_s_getreg(HWREG(29, 0, 32));
    out[base + 8] = __builtin_amdgcn_s_getreg(HWREG(23, 0, 32));
    out[base + 9] = __builtin_amdgcn_s_getreg(HWREG(24, 0, 32));
}

torch::Tensor probe_hwreg_all_undoc(int n_waves) {
    auto out = torch::zeros({n_waves * 10}, torch::device(torch::kCUDA).dtype(torch::kInt32));
    kernel_hwreg_all_undoc<<<n_waves, 32>>>((unsigned int*)out.data_ptr<int>(), n_waves);
    return out.reshape({n_waves, 10});
}

// =====================================================================
// P6: Variance test for reservoir — collect undocumented reg values over
//     many dispatches to see if they provide useful entropy/variance
// Output per sample: [r7, r18, r19, r27, cycles]
// =====================================================================
__global__ void kernel_hwreg_variance(unsigned int* out) {
    if (threadIdx.x != 0) return;
    out[0] = __builtin_amdgcn_s_getreg(HWREG(7, 0, 32));
    out[1] = __builtin_amdgcn_s_getreg(HWREG(18, 0, 32));
    out[2] = __builtin_amdgcn_s_getreg(HWREG(19, 0, 32));
    out[3] = __builtin_amdgcn_s_getreg(HWREG(27, 0, 32));
    out[4] = __builtin_amdgcn_s_getreg(HWREG(29, 0, 32));
}

torch::Tensor probe_hwreg_variance() {
    auto out = torch::zeros({5}, torch::device(torch::kCUDA).dtype(torch::kInt32));
    kernel_hwreg_variance<<<1, 1>>>((unsigned int*)out.data_ptr<int>());
    return out;
}

'''

HWREG_CPP_SRC = r'''
#include <torch/extension.h>
torch::Tensor probe_hwreg7_topology(int n_waves);
torch::Tensor probe_hwreg_tma(int n_waves);
torch::Tensor probe_hwreg27_temporal(int n_waves);
torch::Tensor probe_hwreg_misc(int n_waves);
torch::Tensor probe_hwreg_all_undoc(int n_waves);
torch::Tensor probe_hwreg_variance();

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("probe_hwreg7_topology", &probe_hwreg7_topology);
    m.def("probe_hwreg_tma", &probe_hwreg_tma);
    m.def("probe_hwreg27_temporal", &probe_hwreg27_temporal);
    m.def("probe_hwreg_misc", &probe_hwreg_misc);
    m.def("probe_hwreg_all_undoc", &probe_hwreg_all_undoc);
    m.def("probe_hwreg_variance", &probe_hwreg_variance);
}
'''

print("  Compiling HIP kernels...", flush=True)
try:
    # Clear cache to force recompile
    import shutil
    cache_dir = os.path.expanduser('~/.cache/torch_extensions/py312_cpu/z2338_hwreg_probe2')
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)

    hwreg_mod = load_inline(
        name='z2338_hwreg_probe2',
        cpp_sources=[HWREG_CPP_SRC],
        cuda_sources=[HWREG_PROBE_SRC],
        extra_cuda_cflags=['-O2'],
        verbose=False,
    )
    print("  Compilation successful.", flush=True)
except Exception as e:
    print(f"  Compilation FAILED: {e}", flush=True)
    hwreg_mod = None

hwreg_results = {}

if hwreg_mod is not None:
    torch.cuda.synchronize()

    # ==================== P1: HWREG[7] + Topology ====================
    print("\n  P1: HWREG[7] bitfield + topology correlation...", flush=True)
    try:
        N = 64
        data = hwreg_mod.probe_hwreg7_topology(N)
        torch.cuda.synchronize()
        d = data.cpu().numpy().astype(np.uint32)

        p1 = {
            'n_waves': N,
            'columns': ['byte0', 'byte1', 'byte2', 'byte3', 'hw_id1', 'hw_id2',
                        'se', 'sa', 'wgp', 'simd', 'wave_in_simd'],
        }

        # Analyze correlation between reg7 fields and topology
        byte3_vals = d[:, 3]  # top byte of reg7
        se_vals = d[:, 6]
        sa_vals = d[:, 7]
        wgp_vals = d[:, 8]

        p1['reg7_byte0_unique'] = sorted(set(int(x) for x in d[:, 0]))
        p1['reg7_byte1_unique'] = sorted(set(int(x) for x in d[:, 1]))
        p1['reg7_byte2_unique'] = sorted(set(int(x) for x in d[:, 2]))
        p1['reg7_byte3_unique'] = sorted(set(int(x) for x in d[:, 3]))
        p1['reg7_full_unique'] = sorted(set(
            f'0x{int(d[i,0]) | (int(d[i,1])<<8) | (int(d[i,2])<<16) | (int(d[i,3])<<24):08X}'
            for i in range(N)))

        p1['se_unique'] = sorted(set(int(x) for x in se_vals))
        p1['sa_unique'] = sorted(set(int(x) for x in sa_vals))
        p1['wgp_unique'] = sorted(set(int(x) for x in wgp_vals))

        # Cross-tabulate reg7 byte3 vs SE_ID
        cross_tab = {}
        for i in range(N):
            key = f'SE{int(se_vals[i])}_SA{int(sa_vals[i])}'
            b3 = int(byte3_vals[i])
            if key not in cross_tab:
                cross_tab[key] = set()
            cross_tab[key].add(f'0x{b3:02X}')
        p1['reg7_byte3_vs_topology'] = {k: sorted(v) for k, v in sorted(cross_tab.items())}

        # Check if reg7 byte3 encodes SE*SA index
        byte3_to_se_sa = {}
        for i in range(N):
            b3 = int(byte3_vals[i])
            se_sa = (int(se_vals[i]), int(sa_vals[i]))
            if b3 not in byte3_to_se_sa:
                byte3_to_se_sa[b3] = set()
            byte3_to_se_sa[b3].add(se_sa)
        p1['byte3_to_se_sa_mapping'] = {f'0x{k:02X}': [list(v) for v in sorted(vs)]
                                         for k, vs in sorted(byte3_to_se_sa.items())}

        # Detailed per-wave dump (first 10)
        p1['sample_waves'] = []
        for i in range(min(10, N)):
            reg7_full = int(d[i,0]) | (int(d[i,1])<<8) | (int(d[i,2])<<16) | (int(d[i,3])<<24)
            p1['sample_waves'].append({
                'wave': i,
                'reg7_hex': f'0x{reg7_full:08X}',
                'reg7_bytes': [f'0x{int(d[i,j]):02X}' for j in range(4)],
                'hw_id1_hex': f'0x{int(d[i,4]):08X}',
                'hw_id2_hex': f'0x{int(d[i,5]):08X}',
                'se': int(d[i,6]), 'sa': int(d[i,7]),
                'wgp': int(d[i,8]), 'simd': int(d[i,9]),
                'wave': int(d[i,10]),
            })

        hwreg_results['P1_reg7_topology'] = p1
        results['parts']['P1_reg7_topology'] = p1
        save_results()
        print(f"    reg7 unique full values: {p1['reg7_full_unique']}", flush=True)
        print(f"    byte3 vs topology: {p1['reg7_byte3_vs_topology']}", flush=True)
    except Exception as e:
        print(f"    P1 FAILED: {e}", flush=True)
        hwreg_results['P1_reg7_topology'] = {'error': str(e)}

    if check_abort(): sys.exit(1)
    wait_cool("P1->P2")

    # ==================== P2: TMA registers ====================
    print("\n  P2: HWREG[17,18,19] — TMA address analysis...", flush=True)
    try:
        N = 64
        data = hwreg_mod.probe_hwreg_tma(N)
        torch.cuda.synchronize()
        d = data.cpu().numpy().astype(np.uint32)

        p2 = {'n_waves': N}

        tma_lo = d[:, 0]
        tma_hi = d[:, 1]
        reg19 = d[:, 2]

        p2['tma_lo_unique'] = sorted(set(f'0x{int(x):08X}' for x in tma_lo))
        p2['tma_hi_unique_count'] = len(set(int(x) for x in tma_hi))
        p2['reg19_unique_count'] = len(set(int(x) for x in reg19))

        # Check if tma_lo+tma_hi form addresses
        p2['tma_addresses_sample'] = []
        for i in range(min(10, N)):
            addr = (int(tma_hi[i]) << 32) | int(tma_lo[i])
            p2['tma_addresses_sample'].append({
                'wave': i,
                'tma_lo': f'0x{int(tma_lo[i]):08X}',
                'tma_hi': f'0x{int(tma_hi[i]):08X}',
                'tma_addr': f'0x{addr:016X}',
                'reg19': f'0x{int(reg19[i]):08X}',
            })

        # Check if reg19 could be an address extension or counter
        reg19_sorted = sorted(int(x) for x in reg19)
        p2['reg19_min'] = f'0x{reg19_sorted[0]:08X}'
        p2['reg19_max'] = f'0x{reg19_sorted[-1]:08X}'
        p2['reg19_range'] = reg19_sorted[-1] - reg19_sorted[0]

        # Check if TMA_HI values look like pointers (high bits should be consistent)
        tma_hi_top = sorted(set((int(x) >> 24) & 0xFF for x in tma_hi))
        p2['tma_hi_top_byte_values'] = [f'0x{x:02X}' for x in tma_hi_top]

        # Check correlation between tma_hi and reg19
        if len(set(int(x) for x in tma_hi)) > 1 and len(set(int(x) for x in reg19)) > 1:
            corr = np.corrcoef(tma_hi.astype(np.float64), reg19.astype(np.float64))[0, 1]
            p2['tma_hi_reg19_correlation'] = float(corr)

        hwreg_results['P2_tma_analysis'] = p2
        results['parts']['P2_tma_analysis'] = p2
        save_results()
        print(f"    TMA_LO unique: {p2['tma_lo_unique']}", flush=True)
        print(f"    TMA_HI unique count: {p2['tma_hi_unique_count']}", flush=True)
        print(f"    REG19 unique count: {p2['reg19_unique_count']}", flush=True)
        print(f"    TMA_HI top bytes: {p2['tma_hi_top_byte_values']}", flush=True)
    except Exception as e:
        print(f"    P2 FAILED: {e}", flush=True)
        hwreg_results['P2_tma_analysis'] = {'error': str(e)}

    if check_abort(): sys.exit(1)
    wait_cool("P2->P3")

    # ==================== P3: HWREG[27] Temporal ====================
    print("\n  P3: HWREG[27] temporal stability...", flush=True)
    try:
        N = 64
        data = hwreg_mod.probe_hwreg27_temporal(N)
        torch.cuda.synchronize()
        d = data.cpu().numpy().astype(np.uint32)

        p3 = {'n_waves': N}

        r27_t0 = d[:, 0]
        r27_t1 = d[:, 1]
        r27_t2 = d[:, 2]
        r27_t3 = d[:, 3]
        cycles_start = d[:, 4]
        cycles_end = d[:, 5]

        # Check if reg27 changes within a single kernel invocation
        changed_t0_t1 = int(np.sum(r27_t0 != r27_t1))
        changed_t0_t2 = int(np.sum(r27_t0 != r27_t2))
        changed_t0_t3 = int(np.sum(r27_t0 != r27_t3))

        p3['waves_where_reg27_changed'] = {
            't0_vs_t1': changed_t0_t1,
            't0_vs_t2': changed_t0_t2,
            't0_vs_t3': changed_t0_t3,
        }
        p3['reg27_is_dynamic'] = changed_t0_t3 > 0

        # Cycle count between reads
        elapsed = (cycles_end.astype(np.int64) - cycles_start.astype(np.int64))
        p3['elapsed_cycles_mean'] = float(np.mean(elapsed))
        p3['elapsed_cycles_std'] = float(np.std(elapsed))

        # Unique values at each timepoint
        p3['reg27_t0_unique'] = len(set(int(x) for x in r27_t0))
        p3['reg27_t3_unique'] = len(set(int(x) for x in r27_t3))

        # Bitfield analysis of reg27
        r27_sample = r27_t0[:10]
        p3['reg27_bitfield_sample'] = []
        for i, val in enumerate(r27_sample):
            v = int(val)
            p3['reg27_bitfield_sample'].append({
                'wave': i,
                'hex': f'0x{v:08X}',
                'bin': f'{v:032b}',
                'byte0': v & 0xFF,
                'byte1': (v >> 8) & 0xFF,
                'byte2': (v >> 16) & 0xFF,
                'byte3': (v >> 24) & 0xFF,
                'nibbles': [f'{(v >> (28-4*j)) & 0xF:X}' for j in range(8)],
            })

        hwreg_results['P3_reg27_temporal'] = p3
        results['parts']['P3_reg27_temporal'] = p3
        save_results()
        print(f"    Waves where reg27 changed: {p3['waves_where_reg27_changed']}", flush=True)
        print(f"    Dynamic: {p3['reg27_is_dynamic']}", flush=True)
        print(f"    Unique at t0: {p3['reg27_t0_unique']}, at t3: {p3['reg27_t3_unique']}", flush=True)
    except Exception as e:
        print(f"    P3 FAILED: {e}", flush=True)
        hwreg_results['P3_reg27_temporal'] = {'error': str(e)}

    if check_abort(): sys.exit(1)
    wait_cool("P3->P4")

    # ==================== P4: HWREG[8,9,28] constants ====================
    print("\n  P4: HWREG[8,9,28] constant analysis...", flush=True)
    try:
        N = 64
        data = hwreg_mod.probe_hwreg_misc(N)
        torch.cuda.synchronize()
        d = data.cpu().numpy().astype(np.uint32)

        p4 = {'n_waves': N}

        r8_vals = d[:, 0]
        r9_vals = d[:, 1]
        r28_vals = d[:, 2]

        p4['reg8_unique'] = sorted(set(f'0x{int(x):08X}' for x in r8_vals))
        p4['reg9_unique'] = sorted(set(f'0x{int(x):08X}' for x in r9_vals))
        p4['reg28_unique'] = sorted(set(f'0x{int(x):08X}' for x in r28_vals))

        # Decode reg28 = 0x603
        r28 = int(r28_vals[0])
        p4['reg28_decode'] = {
            'full': f'0x{r28:08X}',
            'bits_2_0': r28 & 0x7,
            'bits_5_3': (r28 >> 3) & 0x7,
            'bits_7_6': (r28 >> 6) & 0x3,
            'bits_10_8': (r28 >> 8) & 0x7,
            'bits_15_11': (r28 >> 11) & 0x1F,
            'bits_31_16': (r28 >> 16) & 0xFFFF,
        }

        # Decode reg9 = 0x740A
        r9 = int(r9_vals[0])
        p4['reg9_decode'] = {
            'full': f'0x{r9:08X}',
            'bits_7_0': r9 & 0xFF,    # 0x0A = 10
            'bits_15_8': (r9 >> 8) & 0xFF,  # 0x74 = 116
            'bits_31_16': (r9 >> 16) & 0xFFFF,  # 0x0000
            'interpretation': f'Could be: config({r9&0xFF}, {(r9>>8)&0xFF})',
        }

        # Decode reg8
        r8 = int(r8_vals[0])
        p4['reg8_decode'] = {
            'full': f'0x{r8:08X}',
            'as_signed': struct.unpack('<i', struct.pack('<I', r8))[0],
            'bits_31_24': (r8 >> 24) & 0xFF,
            'bits_23_16': (r8 >> 16) & 0xFF,
            'bits_15_8': (r8 >> 8) & 0xFF,
            'bits_7_0': r8 & 0xFF,
            'interpretation': 'Possibly a GPU virtual address or config pointer',
        }

        # Check if reg8 varies across waves
        p4['reg8_varies'] = len(set(int(x) for x in r8_vals)) > 1
        p4['reg9_varies'] = len(set(int(x) for x in r9_vals)) > 1
        p4['reg28_varies'] = len(set(int(x) for x in r28_vals)) > 1

        hwreg_results['P4_constants'] = p4
        results['parts']['P4_constants'] = p4
        save_results()
        print(f"    REG8 unique: {p4['reg8_unique']}, varies: {p4['reg8_varies']}", flush=True)
        print(f"    REG9 unique: {p4['reg9_unique']}", flush=True)
        print(f"    REG28 decode: {p4['reg28_decode']}", flush=True)
    except Exception as e:
        print(f"    P4 FAILED: {e}", flush=True)
        hwreg_results['P4_constants'] = {'error': str(e)}

    if check_abort(): sys.exit(1)
    wait_cool("P4->P5")

    # ==================== P5: Dispatch size sensitivity ====================
    print("\n  P5: Dispatch size sensitivity...", flush=True)
    try:
        p5 = {'grid_sizes': {}}
        col_names = ['r7', 'r8', 'r9', 'r18', 'r19', 'r27', 'r28', 'cycles', 'hw_id1', 'hw_id2']

        for n_waves in [1, 4, 16, 64]:
            data = hwreg_mod.probe_hwreg_all_undoc(n_waves)
            torch.cuda.synchronize()
            d = data.cpu().numpy().astype(np.uint32)

            grid_info = {'n_waves': n_waves}
            for ci, cname in enumerate(col_names):
                vals = d[:, ci]
                unique = sorted(set(int(x) for x in vals))
                grid_info[f'{cname}_unique_count'] = len(unique)
                grid_info[f'{cname}_unique'] = [f'0x{x:08X}' for x in unique[:10]]

            p5['grid_sizes'][str(n_waves)] = grid_info
            print(f"    Grid {n_waves}: r7={grid_info['r7_unique_count']}u, "
                  f"r27={grid_info['r27_unique_count']}u, "
                  f"r18={grid_info['r18_unique_count']}u", flush=True)

        hwreg_results['P5_dispatch_sensitivity'] = p5
        results['parts']['P5_dispatch_sensitivity'] = p5
        save_results()
    except Exception as e:
        print(f"    P5 FAILED: {e}", flush=True)
        hwreg_results['P5_dispatch_sensitivity'] = {'error': str(e)}

    if check_abort(): sys.exit(1)
    wait_cool("P5->P6")

    # ==================== P6: Variance for reservoir ====================
    print("\n  P6: Temporal variance over 200 dispatches...", flush=True)
    try:
        N_SAMPLES = 200
        samples = np.zeros((N_SAMPLES, 5), dtype=np.uint32)

        for i in range(N_SAMPLES):
            data = hwreg_mod.probe_hwreg_variance()
            torch.cuda.synchronize()
            samples[i] = data.cpu().numpy().astype(np.uint32)

            if i % 50 == 0 and check_abort():
                break

        col_names_v = ['r7', 'r18', 'r19', 'r27', 'cycles']
        p6 = {'n_samples': N_SAMPLES}

        for ci, cname in enumerate(col_names_v):
            vals = samples[:, ci]
            unique = sorted(set(int(x) for x in vals))
            p6[f'{cname}_unique_count'] = len(unique)
            p6[f'{cname}_std'] = float(np.std(vals.astype(np.float64)))
            p6[f'{cname}_min'] = f'0x{int(np.min(vals)):08X}'
            p6[f'{cname}_max'] = f'0x{int(np.max(vals)):08X}'
            p6[f'{cname}_entropy_bits'] = round(math.log2(max(1, len(unique))), 2)
            # For reservoir: is it useful?
            if len(unique) > 1:
                vals_norm = (vals.astype(np.float64) - np.mean(vals)) / max(np.std(vals), 1e-10)
                # Autocorrelation at lag 1
                if len(vals_norm) > 1:
                    acf1 = float(np.corrcoef(vals_norm[:-1], vals_norm[1:])[0, 1])
                    p6[f'{cname}_acf1'] = acf1

        # Quick reservoir test: can we classify wave4 with these features?
        # Generate simple wave4 labels and train a linear classifier
        cycles = samples[:, 4].astype(np.float64)
        r18 = samples[:, 1].astype(np.float64)
        r19 = samples[:, 2].astype(np.float64)
        r27 = samples[:, 3].astype(np.float64)

        # Normalize each feature
        def norm(x):
            s = np.std(x)
            if s < 1e-10: return np.zeros_like(x)
            return (x - np.mean(x)) / s

        features_undoc = np.column_stack([norm(r18), norm(r19), norm(r27)])
        features_cycles = norm(cycles).reshape(-1, 1)

        # Simple 4-class waveform classification based on time index
        labels = np.array([i % 4 for i in range(N_SAMPLES)])

        # Train simple linear classifier (ridge regression)
        from numpy.linalg import lstsq

        def ridge_classify(X, y, n_classes=4, alpha=1.0):
            n = len(y)
            Y_oh = np.zeros((n, n_classes))
            for i in range(n):
                Y_oh[i, y[i]] = 1.0

            # Split 80/20
            split = int(0.8 * n)
            X_tr, X_te = X[:split], X[split:]
            Y_tr, Y_te = Y_oh[:split], Y_oh[split:]
            y_te = y[split:]

            # Ridge regression
            A = X_tr.T @ X_tr + alpha * np.eye(X_tr.shape[1])
            B = X_tr.T @ Y_tr
            W = np.linalg.solve(A, B)

            pred = X_te @ W
            pred_class = np.argmax(pred, axis=1)
            acc = np.mean(pred_class == y_te)
            return float(acc)

        if features_undoc.shape[1] > 0 and np.any(np.std(features_undoc, axis=0) > 0):
            acc_undoc = ridge_classify(features_undoc, labels)
        else:
            acc_undoc = 0.25  # chance

        if np.std(features_cycles) > 0:
            acc_cycles = ridge_classify(features_cycles, labels)
        else:
            acc_cycles = 0.25

        # Combined
        features_all = np.column_stack([features_undoc, features_cycles])
        valid_cols = np.std(features_all, axis=0) > 0
        features_all = features_all[:, valid_cols]
        if features_all.shape[1] > 0:
            acc_combined = ridge_classify(features_all, labels)
        else:
            acc_combined = 0.25

        p6['reservoir_test'] = {
            'task': 'wave4 classification (4-class by time index)',
            'n_train': int(0.8 * N_SAMPLES),
            'n_test': N_SAMPLES - int(0.8 * N_SAMPLES),
            'acc_undoc_regs_only': round(acc_undoc * 100, 1),
            'acc_cycles_only': round(acc_cycles * 100, 1),
            'acc_combined': round(acc_combined * 100, 1),
            'chance': 25.0,
        }

        hwreg_results['P6_variance_reservoir'] = p6
        results['parts']['P6_variance_reservoir'] = p6
        save_results()

        print(f"    Variance analysis:", flush=True)
        for cname in col_names_v:
            print(f"      {cname}: unique={p6[f'{cname}_unique_count']}, "
                  f"entropy={p6.get(f'{cname}_entropy_bits', 0):.1f} bits, "
                  f"acf1={p6.get(f'{cname}_acf1', 'N/A')}", flush=True)
        print(f"    Reservoir test (wave4):", flush=True)
        print(f"      Undoc regs: {p6['reservoir_test']['acc_undoc_regs_only']}%", flush=True)
        print(f"      Cycles only: {p6['reservoir_test']['acc_cycles_only']}%", flush=True)
        print(f"      Combined: {p6['reservoir_test']['acc_combined']}%", flush=True)
    except Exception as e:
        print(f"    P6 FAILED: {e}", flush=True)
        hwreg_results['P6_variance_reservoir'] = {'error': str(e)}

# ======================================================================
# Generate HWREG mapping report
# ======================================================================
print("\n  Generating HWREG mapping report...", flush=True)

hwreg_report = []
hwreg_report.append("=" * 70)
hwreg_report.append("z2338 PART 2: UNDOCUMENTED HWREG MAPPING REPORT")
hwreg_report.append(f"GPU: AMD Radeon 8060S (gfx1151, Strix Halo)")
hwreg_report.append(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
hwreg_report.append("METHOD: READ-ONLY s_getreg probes from HIP kernels")
hwreg_report.append("=" * 70)

for part_name, pdata in sorted(hwreg_results.items()):
    hwreg_report.append(f"\n{'='*50}")
    hwreg_report.append(f"  {part_name}")
    hwreg_report.append(f"{'='*50}")
    hwreg_report.append(json.dumps(pdata, indent=2, cls=NpEncoder))

save_txt('hwreg_map', '\n'.join(hwreg_report))

# Final save
results['parts']['hwreg_mapping'] = hwreg_results
save_results()

# ======================================================================
# SUMMARY
# ======================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

# Firmware ISA determination
mec_fw = fw_results.get('mec', {})
isa = mec_fw.get('isa_analysis', {})
if isa:
    nop_pct = isa.get('nop_pct', 0)
    riscv_pct = isa.get('riscv_pct', 0)
    high_nibbles = isa.get('high_nibble_distribution', {})
    print(f"\n  MEC ISA Analysis:")
    print(f"    NOP/zero words: {nop_pct}%")
    print(f"    RISC-V bottom-2-bits signature: {riscv_pct}%")
    print(f"    High nibble spread: {len(high_nibbles)} distinct values")

    if nop_pct > 60:
        print(f"    → High NOP count suggests SPARSE microcode (F32-style or custom RISC)")
    if riscv_pct > 80:
        print(f"    → Strong RISC-V signature")
    elif riscv_pct > 40:
        print(f"    → Moderate RISC-V overlap (may be coincidental with F32)")

    # Check if it's the custom MEC F-micro format
    common_words = mec_fw.get('most_common_words', {})
    print(f"    Most common non-zero words: {list(common_words.items())[:5]}")

print(f"\n  Files saved:")
print(f"    {RESULTS / 'z2338_mec_disasm.txt'}")
print(f"    {RESULTS / 'z2338_hwreg_map.txt'}")
print(f"    {RESULTS / 'z2338_firmware_disasm.json'}")

print(f"\n  Temperature: {get_temp()}C")
print(f"  DONE.", flush=True)
