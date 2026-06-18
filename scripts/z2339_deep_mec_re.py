#!/usr/bin/env python3
"""
z2339_deep_mec_re.py — Deep MEC/RLC/PFP Firmware Reverse Engineering (gfx1151)
===============================================================================
Full handler extraction, PM4 dispatch table recovery, HWREG mapping,
cross-reference with kernel driver, and exploitable pattern identification.

Target: AMD Radeon 8060S (gfx1151, RDNA3)
Firmware ISA: F32 (CP microcode engine), NOT shader ISA

SAFETY: READ-ONLY analysis. No hardware register writes. No firmware modification.

Run:
  PYTHONUNBUFFERED=1 ./venv/bin/python scripts/z2339_deep_mec_re.py
"""

import os, sys, time, json, struct, collections, math, hashlib, subprocess
from pathlib import Path
from collections import Counter, defaultdict

os.environ['PYTHONUNBUFFERED'] = '1'

BASE = Path(__file__).resolve().parent.parent
RESULTS = BASE / 'results'
RESULTS.mkdir(exist_ok=True)

# ======================================================================
# Thermal Safety (light — no GPU ops, just file analysis)
# ======================================================================
def get_temp():
    try: return int(open('/sys/class/thermal/thermal_zone0/temp').read()) // 1000
    except: return 0

def check_abort():
    t = get_temp()
    if t >= 85:
        print(f"\n  [ABORT] Temperature {t}C >= 85C!", flush=True)
        return True
    return False

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, bytes): return obj.hex()
        if hasattr(obj, 'item'): return obj.item()  # numpy scalar
        if hasattr(obj, 'tolist'): return obj.tolist()  # numpy array
        return super().default(obj)

results = {
    'experiment': 'z2339_deep_mec_re',
    'description': 'Deep MEC/RLC/PFP firmware RE on gfx1151 (Radeon 8060S)',
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'firmware': {},
    'mec_analysis': {},
    'rlc_analysis': {},
    'pfp_analysis': {},
    'me_analysis': {},
    'pm4_dispatch': {},
    'exploitable': {},
    'kernel_xref': {},
}

SAVE_JSON = RESULTS / 'z2339_deep_mec_re.json'

def save_results():
    with open(SAVE_JSON, 'w') as f:
        json.dump(results, f, indent=2, cls=NpEncoder)
    print(f"  [SAVED] {SAVE_JSON}", flush=True)

def save_txt(name, text):
    p = RESULTS / f'z2339_{name}.txt'
    with open(p, 'w') as f:
        f.write(text)
    print(f"  [SAVED] {p}", flush=True)

# ======================================================================
# Known PM4 opcodes (from kernel soc15d.h, comprehensive)
# ======================================================================
PM4_OPCODES = {
    0x10: 'NOP', 0x11: 'SET_BASE', 0x12: 'CLEAR_STATE',
    0x13: 'INDEX_BUFFER_SIZE', 0x15: 'DISPATCH_DIRECT',
    0x16: 'DISPATCH_INDIRECT', 0x1D: 'ATOMIC_GDS', 0x1E: 'ATOMIC_MEM',
    0x1F: 'OCCLUSION_QUERY', 0x20: 'SET_PREDICATION', 0x21: 'REG_RMW',
    0x22: 'COND_EXEC', 0x23: 'PRED_EXEC', 0x24: 'DRAW_INDIRECT',
    0x25: 'DRAW_INDEX_INDIRECT', 0x26: 'INDEX_BASE', 0x27: 'DRAW_INDEX_2',
    0x28: 'CONTEXT_CONTROL', 0x2A: 'INDEX_TYPE',
    0x2C: 'DRAW_INDIRECT_MULTI', 0x2D: 'DRAW_INDEX_AUTO',
    0x2F: 'NUM_INSTANCES', 0x30: 'DRAW_INDEX_MULTI_AUTO',
    0x33: 'INDIRECT_BUFFER_CONST', 0x34: 'STRMOUT_BUFFER_UPDATE',
    0x35: 'DRAW_INDEX_OFFSET_2', 0x36: 'DRAW_PREAMBLE',
    0x37: 'WRITE_DATA', 0x38: 'DRAW_INDEX_INDIRECT_MULTI',
    0x39: 'MEM_SEMAPHORE', 0x3C: 'WAIT_REG_MEM', 0x3F: 'INDIRECT_BUFFER',
    0x40: 'COPY_DATA', 0x42: 'PFP_SYNC_ME', 0x45: 'COND_WRITE',
    0x46: 'EVENT_WRITE', 0x49: 'RELEASE_MEM', 0x4A: 'PREAMBLE_CNTL',
    0x50: 'DMA_DATA', 0x58: 'ACQUIRE_MEM', 0x59: 'REWIND',
    0x5E: 'LOAD_UCONFIG_REG', 0x5F: 'LOAD_SH_REG',
    0x60: 'LOAD_CONFIG_REG', 0x61: 'LOAD_CONTEXT_REG',
    0x68: 'SET_CONFIG_REG', 0x69: 'SET_CONTEXT_REG',
    0x73: 'SET_CONTEXT_REG_INDIRECT', 0x76: 'SET_SH_REG',
    0x77: 'SET_SH_REG_OFFSET', 0x78: 'SET_QUEUE_REG',
    0x79: 'SET_UCONFIG_REG', 0x7D: 'SCRATCH_RAM_WRITE',
    0x7E: 'SCRATCH_RAM_READ', 0x80: 'LOAD_CONST_RAM',
    0x81: 'WRITE_CONST_RAM', 0x83: 'DUMP_CONST_RAM',
    0x84: 'INCREMENT_CE_COUNTER', 0x85: 'WAIT_ON_CE_COUNTER',
    0x88: 'WAIT_ON_DE_COUNTER_DIFF', 0x8B: 'SWITCH_BUFFER',
    0x90: 'FRAME_CONTROL', 0x98: 'INVALIDATE_TLBS',
    0xA0: 'SET_RESOURCES', 0xA2: 'MAP_QUEUES', 0xA3: 'UNMAP_QUEUES',
    0xA4: 'QUERY_STATUS', 0xD2: 'RUN_CLEANER_SHADER',
}

# CP internal register names (from RE + kernel headers)
CP_INTERNAL_REGS = {
    0x0008: 'CP_ME_CNTL/FETCH_CTRL',
    0x0010: 'CP_FETCH_SIZE/DMA_CTRL',
    0x0013: 'CP_DISPATCH_STATE',  # Most-referenced
    0x0014: 'CP_RING_RPTR',
    0x0018: 'CP_IB_CTRL',
    0x001A: 'CP_IB_BASE_LO',
    0x001B: 'CP_IB_BASE_HI',
    0x0021: 'CP_HQD_WPTR_UPDATE',
    0x0026: 'CP_HQD_STATE',
    0x0027: 'CP_HQD_DOORBELL',
    0x0029: 'CP_HQD_EOP_CTRL',
    0x002A: 'CP_HQD_EOP_STATE',
    0x002B: 'CP_CNTL_STATUS',
    0x002F: 'CP_QUEUE_ID',
    0x0032: 'CP_IB_SIZE',
    0x0033: 'CP_ME_STATUS',
    0x0040: 'CP_PERF_CTRL',
    0x004F: 'CP_SCRATCH/DEBUG',
    0x0053: 'CP_TRAP_CTRL',
    0x0074: 'CP_GDS_CTRL',
    0x0087: 'CP_RESOURCE_STATE',
    0x0088: 'CP_CTX_SAVE_CTRL',
    0x008B: 'CP_QUEUE_DEQUEUE',
    0x008E: 'CP_PREEMPT_STATE',
    0x00AB: 'CP_DMA_STATE',
    0x00BF: 'CP_JUMP_TABLE_BASE',
    0x00D9: 'CP_DEBUG_REG',
}

# ======================================================================
# F32 ISA Decoder
# ======================================================================
class F32Instruction:
    """Decode a single F32 ISA instruction word."""
    def __init__(self, word, offset):
        self.word = word
        self.offset = offset
        self.top = (word >> 24) & 0xFF
        self.opcode = ''
        self.operands = ''
        self.category = 'UNKNOWN'
        self._decode()

    def _decode(self):
        w = self.word
        top = self.top

        if w == 0x90000000:
            self.opcode = 'S_ENDPGM'
            self.category = 'CONTROL'
            return

        if w == 0x00000000:
            self.opcode = 'S_NOP'
            self.category = 'NOP'
            return

        # S_MOV immediate: C0RR_IIII
        if top == 0xC0:
            reg = (w >> 16) & 0xFF
            imm = w & 0xFFFF
            self.opcode = 'S_MOV_IMM'
            self.operands = f's{reg}, 0x{imm:04X}'
            self.category = 'SCALAR_MOV'
            return

        # S_LOAD variants: C4RR_SSSS
        if top == 0xC4:
            reg = (w >> 16) & 0xFF
            src = w & 0xFFFF
            self.opcode = 'S_LOAD_DWORD'
            self.operands = f's{reg}, [0x{src:04X}]'
            self.category = 'SCALAR_LOAD'
            return

        # S_STORE variants: CC-CF
        if top in [0xCC, 0xCD, 0xCE, 0xCF]:
            mode = top - 0xCC
            reg = (w >> 16) & 0xFF
            dst = w & 0xFFFF
            sizes = ['DWORD', 'DWORDX2', 'DWORDX4', 'DWORDX8']
            self.opcode = f'S_STORE_{sizes[mode]}'
            self.operands = f's{reg}, [0x{dst:04X}]'
            self.category = 'SCALAR_STORE'
            return

        # Hardware register access: D8MM_RRRR
        if top == 0xD8:
            mode = (w >> 16) & 0xFF
            reg_id = w & 0xFFFF
            mode_names = {0x00: 'READ', 0x01: 'INDEXED_READ', 0x08: 'INDEXED_LOAD',
                          0x40: 'WRITE', 0x41: 'INDEXED_WRITE',
                          0x80: 'SET', 0xC0: 'RMW', 0xC8: 'RMW_IDX'}
            mstr = mode_names.get(mode, f'MODE_0x{mode:02X}')
            reg_name = CP_INTERNAL_REGS.get(reg_id, '')
            self.opcode = f'S_HWREG_{mstr}'
            self.operands = f'0x{reg_id:04X}'
            if reg_name:
                self.operands += f' ({reg_name})'
            self.category = 'HWREG'
            return

        # WAITCNT / special control: DC
        if top == 0xDC:
            sub = (w >> 16) & 0xFF
            imm = w & 0xFFFF
            self.opcode = f'S_SPECIAL_0xDC{sub:02X}'
            self.operands = f'0x{imm:04X}'
            self.category = 'CONTROL'
            return

        # Branches
        if top in [0x94, 0x95, 0x96, 0x97, 0x9A, 0x9B]:
            names = {0x94:'S_CBRANCH_SCC0', 0x95:'S_CBRANCH_SCC1',
                     0x96:'S_CBRANCH_EXECZ', 0x97:'S_CBRANCH_VCCZ',
                     0x9A:'S_CBRANCH_EXECNZ', 0x9B:'S_CBRANCH_X'}
            off_val = w & 0xFFFF
            if off_val > 0x8000: off_val -= 0x10000
            target = self.offset + 4 + off_val * 4
            self.opcode = names.get(top, f'S_CBRANCH_0x{top:02X}')
            self.operands = f'0x{target:06X} (rel={off_val:+d})'
            self.category = 'BRANCH'
            return

        if top == 0x8C:
            off_val = w & 0xFFFF
            if off_val > 0x8000: off_val -= 0x10000
            target = self.offset + 4 + off_val * 4
            self.opcode = 'S_BRANCH'
            self.operands = f'0x{target:06X} (rel={off_val:+d})'
            self.category = 'BRANCH'
            return

        # ALU operations
        if top == 0x80:
            dst = (w >> 16) & 0xFF
            src0 = (w >> 8) & 0xFF
            src1 = w & 0xFF
            self.opcode = 'S_ADD_U32'
            self.operands = f's{dst}, s{src0}, s{src1}'
            self.category = 'ALU'
            return

        if top in range(0x00, 0x40) and w != 0:
            self.opcode = f'ALU_0x{top:02X}'
            self.operands = f'0x{w:08X}'
            self.category = 'ALU'
            return

        if top in [0x7C, 0x7D, 0x7E, 0x7F]:
            self.opcode = f'S_BIT_0x{top:02X}'
            self.operands = f'0x{w:08X}'
            self.category = 'ALU'
            return

        self.opcode = f'UNK_0x{top:02X}'
        self.operands = f'0x{w:08X}'

    def __str__(self):
        return f'{self.offset:06X}: {self.word:08X}  {self.opcode:24s} {self.operands}'


def decode_firmware(data, isa_offset):
    """Decode all instructions from ISA offset to end."""
    instructions = []
    for off in range(isa_offset, len(data) - 3, 4):
        w = struct.unpack_from('<I', data, off)[0]
        instructions.append(F32Instruction(w, off))
    return instructions


# ======================================================================
# STEP 1: Full MEC Handler Extraction
# ======================================================================
def analyze_mec(fw_path):
    """Complete MEC firmware analysis."""
    print("\n" + "=" * 70)
    print("STEP 1: MEC FIRMWARE ANALYSIS")
    print("=" * 70, flush=True)

    data = open(fw_path, 'rb').read()
    size = len(data)
    sha256 = hashlib.sha256(data).hexdigest()

    # Parse header
    hdr_size = struct.unpack_from('<I', data, 0)[0]
    hdr_hdr_size = struct.unpack_from('<I', data, 4)[0]
    ucode_ver = struct.unpack_from('<I', data, 0x10)[0]
    ucode_size = struct.unpack_from('<I', data, 0x14)[0]
    ucode_offset = struct.unpack_from('<I', data, 0x18)[0]

    # Find $PS1 and ISA start
    ps1_off = data.find(b'$PS1')
    isa_off = ps1_off + 256 if ps1_off >= 0 else ucode_offset

    print(f"  File: {fw_path} ({size} bytes)")
    print(f"  SHA256: {sha256[:16]}...")
    print(f"  ucode_version: 0x{ucode_ver:08X}")
    print(f"  ucode_size: {ucode_size} bytes")
    print(f"  ucode_offset: 0x{ucode_offset:04X}")
    print(f"  $PS1 offset: 0x{ps1_off:04X}")
    print(f"  ISA start: 0x{isa_off:04X}")
    print(f"  ISA payload: {size - isa_off} bytes = {(size - isa_off) // 4} instructions")

    info = {
        'file': str(fw_path), 'size': size, 'sha256': sha256,
        'ucode_version': f'0x{ucode_ver:08X}', 'ucode_size': ucode_size,
        'isa_offset': f'0x{isa_off:04X}', 'isa_size': size - isa_off,
    }

    # Decode all instructions
    print("  Decoding F32 ISA instructions...", flush=True)
    instrs = decode_firmware(data, isa_off)

    # ---- Split into handlers by S_ENDPGM ----
    print("  Splitting into handlers (S_ENDPGM delimiters)...", flush=True)
    handlers = []
    handler_start = 0
    for idx, instr in enumerate(instrs):
        if instr.word == 0x90000000:  # S_ENDPGM
            handler_size = idx - handler_start + 1
            # Count non-NOP instructions
            non_nop = sum(1 for i in range(handler_start, idx + 1)
                          if instrs[i].word != 0)
            handlers.append({
                'id': len(handlers),
                'start_idx': handler_start,
                'end_idx': idx,
                'start_offset': f'0x{instrs[handler_start].offset:06X}',
                'end_offset': f'0x{instrs[idx].offset:06X}',
                'size_words': handler_size,
                'non_nop_words': non_nop,
                'start_offset_int': instrs[handler_start].offset,
            })
            handler_start = idx + 1

    print(f"  Found {len(handlers)} handlers")
    info['handler_count'] = len(handlers)

    # Sort by size (largest first)
    handlers_by_size = sorted(handlers, key=lambda h: -h['non_nop_words'])

    # ---- Analyze each handler for interesting patterns ----
    print("  Analyzing handler contents...", flush=True)

    handler_details = []
    txt_lines = ["MEC HANDLER MAP", "=" * 70, ""]
    txt_lines.append(f"Total handlers: {len(handlers)}")
    txt_lines.append(f"ISA start: 0x{isa_off:04X}")
    txt_lines.append("")

    # Per-handler analysis
    for h in handlers:
        h_instrs = instrs[h['start_idx']:h['end_idx'] + 1]
        detail = {
            'id': h['id'],
            'offset': h['start_offset'],
            'size': h['size_words'],
            'non_nop': h['non_nop_words'],
        }

        # Collect HWREG accesses
        hwregs = []
        for ins in h_instrs:
            if ins.category == 'HWREG':
                reg_id = ins.word & 0xFFFF
                mode = (ins.word >> 16) & 0xFF
                hwregs.append({'reg': f'0x{reg_id:04X}', 'mode': f'0x{mode:02X}',
                               'offset': f'0x{ins.offset:06X}'})
        detail['hwreg_accesses'] = hwregs

        # Collect branches
        branches = []
        for ins in h_instrs:
            if ins.category == 'BRANCH':
                branches.append({'opcode': ins.opcode, 'operands': ins.operands,
                                 'offset': f'0x{ins.offset:06X}'})
        detail['branches'] = branches[:20]  # limit

        # Collect store operations
        stores = []
        for ins in h_instrs:
            if ins.category == 'SCALAR_STORE':
                dst = ins.word & 0xFFFF
                stores.append({'opcode': ins.opcode, 'dst': f'0x{dst:04X}',
                                'offset': f'0x{ins.offset:06X}'})
        detail['stores'] = stores[:20]

        # Check for MMIO register references (high-value addresses)
        mmio_refs = []
        for ins in h_instrs:
            if ins.category == 'SCALAR_MOV':
                imm = ins.word & 0xFFFF
                # Check if this looks like a register offset (0x2C00-0x3200 compute regs)
                if 0x2C00 <= imm <= 0x3200 or 0x0E00 <= imm <= 0x1000:
                    mmio_refs.append({'value': f'0x{imm:04X}', 'offset': f'0x{ins.offset:06X}'})
        detail['mmio_refs'] = mmio_refs

        handler_details.append(detail)

    # ---- Recover PM4 dispatch table ----
    print("  Recovering PM4 dispatch table...", flush=True)

    # The dispatch table is at ~0x12358: triplets of (C0_load, ALU_compute, CC_store)
    # Pattern: s14 = opcode*16, compute handler addr, store to [0xBF]
    dispatch_table = {}
    dispatch_txt = ["PM4 DISPATCH TABLE (MEC)", "=" * 70, ""]

    for idx in range(len(instrs) - 2):
        ins = instrs[idx]
        if ins.word >> 24 != 0xC0:
            continue
        # Check: next is ALU (0x04xxxxxx pattern), then CC store to 0xBF
        if idx + 2 >= len(instrs):
            continue
        ins2 = instrs[idx + 1]
        ins3 = instrs[idx + 2]
        if ins3.word & 0xFFFF0000 == 0xCCC00000 and ins3.word & 0xFFFF == 0x00BF:
            reg = (ins.word >> 16) & 0xFF
            table_key = ins.word & 0xFFFF
            # table_key = PM4_opcode * 0x10
            pm4_opc = table_key // 0x10
            alu_word = ins2.word
            opc_name = PM4_OPCODES.get(pm4_opc, f'UNKNOWN_0x{pm4_opc:02X}')
            entry = {
                'pm4_opcode': f'0x{pm4_opc:02X}',
                'pm4_name': opc_name,
                'table_key': f'0x{table_key:04X}',
                'handler_alu': f'0x{alu_word:08X}',
                'table_offset': f'0x{ins.offset:06X}',
            }
            dispatch_table[pm4_opc] = entry
            dispatch_txt.append(
                f"  PM4 0x{pm4_opc:02X} ({opc_name:30s}) key=0x{table_key:04X} "
                f"alu=0x{alu_word:08X} @0x{ins.offset:06X}"
            )

    # Also search for potential undocumented entries
    # Look for table entries where pm4_opc is NOT in PM4_OPCODES
    undocumented = []
    for opc, entry in sorted(dispatch_table.items()):
        if opc not in PM4_OPCODES:
            undocumented.append(entry)

    dispatch_txt.append("")
    dispatch_txt.append(f"Total dispatch entries: {len(dispatch_table)}")
    dispatch_txt.append(f"Known PM4 opcodes: {sum(1 for o in dispatch_table if o in PM4_OPCODES)}")
    dispatch_txt.append(f"UNDOCUMENTED opcodes: {len(undocumented)}")
    dispatch_txt.append("")
    if undocumented:
        dispatch_txt.append("=== UNDOCUMENTED PM4 OPCODES ===")
        for u in undocumented:
            dispatch_txt.append(f"  {u['pm4_opcode']} ({u['pm4_name']}) — key={u['table_key']} @{u['table_offset']}")

    # ---- Analyze HWREG access patterns across all handlers ----
    print("  Analyzing HWREG access patterns...", flush=True)

    hwreg_summary = defaultdict(lambda: {'read': 0, 'write': 0, 'other': 0, 'handlers': set()})
    for h_idx, h in enumerate(handlers):
        h_instrs = instrs[h['start_idx']:h['end_idx'] + 1]
        for ins in h_instrs:
            if ins.category == 'HWREG':
                reg_id = ins.word & 0xFFFF
                mode = (ins.word >> 16) & 0xFF
                if mode in [0x00, 0x01, 0x08]:
                    hwreg_summary[reg_id]['read'] += 1
                elif mode in [0x40, 0x41, 0x80]:
                    hwreg_summary[reg_id]['write'] += 1
                else:
                    hwreg_summary[reg_id]['other'] += 1
                hwreg_summary[reg_id]['handlers'].add(h_idx)

    # ---- D8 instruction with indexed addressing (mode 0x01, 0x08, 0x41) ----
    # These use an index register to access MMIO — potential arbitrary register access
    indexed_accesses = []
    for ins in instrs:
        if ins.category == 'HWREG':
            mode = (ins.word >> 16) & 0xFF
            if mode in [0x01, 0x08, 0x41, 0xC8]:
                reg_id = ins.word & 0xFFFF
                indexed_accesses.append({
                    'offset': f'0x{ins.offset:06X}',
                    'word': f'0x{ins.word:08X}',
                    'reg': f'0x{reg_id:04X}',
                    'mode': f'0x{mode:02X}',
                })

    # ---- Build handler text report ----
    txt_lines.append("TOP 30 HANDLERS BY SIZE:")
    txt_lines.append("-" * 60)
    for h in handlers_by_size[:30]:
        txt_lines.append(
            f"  Handler {h['id']:3d}: {h['start_offset']} - {h['end_offset']}  "
            f"size={h['size_words']:5d}  non_nop={h['non_nop_words']:5d}"
        )

    txt_lines.append("")
    txt_lines.append("HWREG ACCESS SUMMARY:")
    txt_lines.append("-" * 60)
    for reg_id, info_r in sorted(hwreg_summary.items(), key=lambda x: -(x[1]['read'] + x[1]['write'] + x[1]['other'])):
        name = CP_INTERNAL_REGS.get(reg_id, '')
        total = info_r['read'] + info_r['write'] + info_r['other']
        txt_lines.append(
            f"  REG 0x{reg_id:04X} {name:30s} R={info_r['read']:3d} W={info_r['write']:3d} "
            f"O={info_r['other']:3d} total={total:4d} handlers={len(info_r['handlers'])}"
        )

    txt_lines.append("")
    txt_lines.append("INDEXED HWREG ACCESSES (potential arbitrary MMIO):")
    txt_lines.append("-" * 60)
    for ia in indexed_accesses:
        txt_lines.append(f"  {ia['offset']}: {ia['word']} reg={ia['reg']} mode={ia['mode']}")

    save_txt('mec_handlers', '\n'.join(txt_lines))
    save_txt('pm4_dispatch', '\n'.join(dispatch_txt))

    # ---- Instruction category statistics ----
    cat_counts = Counter(ins.category for ins in instrs)

    info['instruction_categories'] = dict(cat_counts.most_common())
    info['handler_count'] = len(handlers)
    info['top_handlers'] = handlers_by_size[:20]
    info['hwreg_summary'] = {
        f'0x{reg_id:04X}': {
            'name': CP_INTERNAL_REGS.get(reg_id, ''),
            'read': s['read'], 'write': s['write'], 'other': s['other'],
            'handler_count': len(s['handlers']),
        }
        for reg_id, s in sorted(hwreg_summary.items(), key=lambda x: -(x[1]['read'] + x[1]['write']))
    }
    info['dispatch_table'] = dispatch_table
    info['undocumented_opcodes'] = undocumented
    info['indexed_accesses'] = indexed_accesses[:50]

    results['mec_analysis'] = info
    results['pm4_dispatch'] = {
        'total_entries': len(dispatch_table),
        'known': sum(1 for o in dispatch_table if o in PM4_OPCODES),
        'undocumented': len(undocumented),
        'entries': dispatch_table,
        'undocumented_list': undocumented,
    }

    print(f"  Handler count: {len(handlers)}")
    print(f"  Dispatch table entries: {len(dispatch_table)}")
    print(f"  Undocumented PM4 opcodes: {len(undocumented)}")
    print(f"  Indexed HWREG accesses: {len(indexed_accesses)}")
    print(f"  HWREG unique registers: {len(hwreg_summary)}")

    save_results()
    return instrs, handlers, dispatch_table, hwreg_summary


# ======================================================================
# STEP 2: RLC Firmware Analysis
# ======================================================================
def analyze_rlc(fw_path):
    """RLC firmware analysis — power/clock control, init sequences."""
    print("\n" + "=" * 70)
    print("STEP 2: RLC FIRMWARE ANALYSIS")
    print("=" * 70, flush=True)

    data = open(fw_path, 'rb').read()
    size = len(data)
    sha256 = hashlib.sha256(data).hexdigest()

    # RLC has a different header structure (v2.3 typically)
    hdr_size = struct.unpack_from('<I', data, 0)[0]
    hdr_hdr_size = struct.unpack_from('<I', data, 4)[0]
    hdr_ver_major = struct.unpack_from('<H', data, 8)[0]
    hdr_ver_minor = struct.unpack_from('<H', data, 0xA)[0]

    print(f"  File: {fw_path} ({size} bytes)")
    print(f"  SHA256: {sha256[:16]}...")
    print(f"  Header version: {hdr_ver_major}.{hdr_ver_minor}")
    print(f"  Header size: 0x{hdr_hdr_size:04X}")

    # RLC has complex header with multiple code segments
    # Parse known offsets
    ucode_ver = struct.unpack_from('<I', data, 0x10)[0]
    print(f"  ucode_version: 0x{ucode_ver:08X}")

    # Find $PS1
    ps1_off = data.find(b'$PS1')
    print(f"  $PS1 offset: 0x{ps1_off:04X}" if ps1_off >= 0 else "  No $PS1 found")

    isa_off = ps1_off + 256 if ps1_off >= 0 else 0x200

    # For RLC, the header contains multiple segment offsets
    # Parse the extended header for segment info
    info = {
        'file': str(fw_path), 'size': size, 'sha256': sha256,
        'header_version': f'{hdr_ver_major}.{hdr_ver_minor}',
        'ucode_version': f'0x{ucode_ver:08X}',
        'ps1_offset': f'0x{ps1_off:04X}' if ps1_off >= 0 else None,
        'isa_offset': f'0x{isa_off:04X}',
    }

    # Decode instructions
    print("  Decoding RLC F32 ISA...", flush=True)
    instrs = decode_firmware(data, isa_off)

    # Find S_ENDPGM handlers
    handlers = []
    handler_start = 0
    for idx, ins in enumerate(instrs):
        if ins.word == 0x90000000:
            handler_size = idx - handler_start + 1
            non_nop = sum(1 for i in range(handler_start, idx + 1) if instrs[i].word != 0)
            handlers.append({
                'id': len(handlers),
                'start_offset': f'0x{instrs[handler_start].offset:06X}',
                'end_offset': f'0x{instrs[idx].offset:06X}',
                'size_words': handler_size,
                'non_nop_words': non_nop,
            })
            handler_start = idx + 1

    print(f"  Found {len(handlers)} handlers")

    # HWREG analysis
    hwreg_summary = defaultdict(lambda: {'read': 0, 'write': 0, 'other': 0})
    for ins in instrs:
        if ins.category == 'HWREG':
            reg_id = ins.word & 0xFFFF
            mode = (ins.word >> 16) & 0xFF
            if mode in [0x00, 0x01, 0x08]:
                hwreg_summary[reg_id]['read'] += 1
            elif mode in [0x40, 0x41, 0x80]:
                hwreg_summary[reg_id]['write'] += 1
            else:
                hwreg_summary[reg_id]['other'] += 1

    # Look for register initialization lists (sequences of S_MOV + S_STORE)
    # RLC typically has long init sequences that write MMIO registers
    init_sequences = []
    for idx in range(len(instrs) - 1):
        ins = instrs[idx]
        if ins.category == 'SCALAR_MOV':
            next_ins = instrs[idx + 1]
            if next_ins.category == 'SCALAR_STORE':
                imm = ins.word & 0xFFFF
                dst = next_ins.word & 0xFFFF
                init_sequences.append({
                    'offset': f'0x{ins.offset:06X}',
                    'value': f'0x{imm:04X}',
                    'dest': f'0x{dst:04X}',
                })

    # Look for power/clock gating patterns
    # These typically involve specific register addresses related to CGCG/MGCG
    power_patterns = []
    for idx, ins in enumerate(instrs):
        if ins.category == 'SCALAR_MOV':
            imm = ins.word & 0xFFFF
            # Known power gating registers
            if imm in [0x0DA3, 0x0DAC, 0x0DC5,  # GRBM_PWR_CNTL, GFX_CLKEN, PWR_CNTL2
                        0x1DD6,  # CP_CPC_MGCG_SYNC_CNTL
                        0x1E1F,  # CP_DEBUG
                        0x1E21]:  # CP_CPC_DEBUG
                power_patterns.append({
                    'offset': f'0x{ins.offset:06X}',
                    'reg': f'0x{imm:04X}',
                })

    # Category stats
    cat_counts = Counter(ins.category for ins in instrs)

    txt_lines = ["RLC FIRMWARE ANALYSIS", "=" * 70, ""]
    txt_lines.append(f"Total handlers: {len(handlers)}")
    txt_lines.append(f"HWREG unique registers: {len(hwreg_summary)}")
    txt_lines.append(f"Init sequences (MOV+STORE pairs): {len(init_sequences)}")
    txt_lines.append(f"Power/clock patterns: {len(power_patterns)}")
    txt_lines.append("")

    txt_lines.append("TOP 20 HANDLERS BY SIZE:")
    txt_lines.append("-" * 60)
    handlers_sorted = sorted(handlers, key=lambda h: -h['non_nop_words'])
    for h in handlers_sorted[:20]:
        txt_lines.append(
            f"  Handler {h['id']:3d}: {h['start_offset']} - {h['end_offset']}  "
            f"size={h['size_words']:5d}  non_nop={h['non_nop_words']:5d}"
        )

    txt_lines.append("")
    txt_lines.append("HWREG ACCESS SUMMARY:")
    txt_lines.append("-" * 60)
    for reg_id, s in sorted(hwreg_summary.items(), key=lambda x: -(x[1]['read'] + x[1]['write'])):
        name = CP_INTERNAL_REGS.get(reg_id, '')
        total = s['read'] + s['write'] + s['other']
        txt_lines.append(
            f"  REG 0x{reg_id:04X} {name:30s} R={s['read']:3d} W={s['write']:3d} O={s['other']:3d}")

    txt_lines.append("")
    txt_lines.append("POWER/CLOCK PATTERNS:")
    for pp in power_patterns[:30]:
        txt_lines.append(f"  {pp['offset']}: reg {pp['reg']}")

    txt_lines.append("")
    txt_lines.append("INSTRUCTION CATEGORIES:")
    for cat, cnt in cat_counts.most_common():
        txt_lines.append(f"  {cat}: {cnt}")

    save_txt('rlc_analysis', '\n'.join(txt_lines))

    info['handler_count'] = len(handlers)
    info['hwreg_unique'] = len(hwreg_summary)
    info['init_sequences'] = len(init_sequences)
    info['power_patterns'] = power_patterns[:30]
    info['hwreg_summary'] = {
        f'0x{reg_id:04X}': dict(s)
        for reg_id, s in sorted(hwreg_summary.items(), key=lambda x: -(x[1]['read'] + x[1]['write']))[:30]
    }
    info['instruction_categories'] = dict(cat_counts.most_common())
    info['handlers_by_size'] = handlers_sorted[:20]

    results['rlc_analysis'] = info
    print(f"  Handlers: {len(handlers)}")
    print(f"  HWREG unique: {len(hwreg_summary)}")
    print(f"  Init sequences: {len(init_sequences)}")
    print(f"  Power patterns: {len(power_patterns)}")

    save_results()
    return instrs, handlers


# ======================================================================
# STEP 3: PFP Firmware Analysis
# ======================================================================
def analyze_pfp_or_me(fw_path, name):
    """Analyze PFP or ME firmware."""
    print(f"\n{'=' * 70}")
    print(f"STEP 3: {name.upper()} FIRMWARE ANALYSIS")
    print("=" * 70, flush=True)

    data = open(fw_path, 'rb').read()
    size = len(data)
    sha256 = hashlib.sha256(data).hexdigest()

    ucode_ver = struct.unpack_from('<I', data, 0x10)[0]
    ucode_offset = struct.unpack_from('<I', data, 0x18)[0]

    ps1_off = data.find(b'$PS1')
    isa_off = ps1_off + 256 if ps1_off >= 0 else ucode_offset

    print(f"  File: {fw_path} ({size} bytes)")
    print(f"  ucode_version: 0x{ucode_ver:08X}")
    print(f"  ISA start: 0x{isa_off:04X}")

    info = {
        'file': str(fw_path), 'size': size, 'sha256': sha256,
        'ucode_version': f'0x{ucode_ver:08X}',
        'isa_offset': f'0x{isa_off:04X}',
    }

    instrs = decode_firmware(data, isa_off)

    # Count handlers
    handler_count = sum(1 for ins in instrs if ins.word == 0x90000000)

    # HWREG access
    hwreg_accesses = defaultdict(int)
    for ins in instrs:
        if ins.category == 'HWREG':
            reg_id = ins.word & 0xFFFF
            hwreg_accesses[reg_id] += 1

    # Look for PM4 parsing (PFP is the front-end parser)
    # PFP reads PM4 headers and either handles them or forwards to ME/MEC
    pm4_refs = []
    for idx in range(len(instrs) - 2):
        ins = instrs[idx]
        if ins.category == 'SCALAR_MOV':
            imm = ins.word & 0xFFFF
            if imm in PM4_OPCODES or (0x10 <= imm <= 0xFF):
                # Check if followed by compare/branch
                for j in range(1, 4):
                    if idx + j < len(instrs) and instrs[idx + j].category == 'BRANCH':
                        pm4_refs.append({
                            'offset': f'0x{ins.offset:06X}',
                            'value': f'0x{imm:02X}',
                            'name': PM4_OPCODES.get(imm, 'unknown'),
                            'branch': instrs[idx + j].operands,
                        })
                        break

    cat_counts = Counter(ins.category for ins in instrs)

    info['handler_count'] = handler_count
    info['hwreg_unique'] = len(hwreg_accesses)
    info['pm4_references'] = pm4_refs[:50]
    info['instruction_categories'] = dict(cat_counts.most_common())
    info['hwreg_accesses'] = {f'0x{k:04X}': v for k, v in
                               sorted(hwreg_accesses.items(), key=lambda x: -x[1])[:30]}

    results[f'{name}_analysis'] = info

    print(f"  Handlers: {handler_count}")
    print(f"  HWREG unique: {len(hwreg_accesses)}")
    print(f"  PM4 opcode references: {len(pm4_refs)}")

    save_results()
    return instrs


# ======================================================================
# STEP 4: Kernel Driver Cross-Reference
# ======================================================================
def analyze_kernel_source():
    """Cross-reference firmware findings with kernel driver source."""
    print("\n" + "=" * 70)
    print("STEP 4: KERNEL DRIVER CROSS-REFERENCE")
    print("=" * 70, flush=True)

    kernel_src = Path('/home/ikaros/src/linux/drivers/gpu/drm/amd')
    if not kernel_src.exists():
        print("  [SKIP] Kernel source not found")
        return

    findings = {}

    # 1. Extract ALL PM4 PACKET3 opcodes from soc15d.h
    soc15d = kernel_src / 'amdgpu' / 'soc15d.h'
    if soc15d.exists():
        print("  Parsing soc15d.h for PACKET3 opcodes...", flush=True)
        kernel_opcodes = {}
        with open(soc15d) as f:
            for line in f:
                line = line.strip()
                if '#define' in line and 'PACKET3_' in line and 'PACKET3_SET_' not in line.split()[1] + '_x':
                    parts = line.split()
                    if len(parts) >= 3:
                        name = parts[1]
                        val_str = parts[2]
                        if val_str.startswith('0x') or val_str.startswith('0X'):
                            try:
                                val = int(val_str, 16)
                                if 0 < val < 0x100:
                                    kernel_opcodes[val] = name.replace('PACKET3_', '')
                            except ValueError:
                                pass
        findings['kernel_pm4_opcodes'] = {f'0x{k:02X}': v for k, v in sorted(kernel_opcodes.items())}
        print(f"  Found {len(kernel_opcodes)} PACKET3 opcodes in kernel")

    # 2. Check gfx_v11_0.c for debug/test modes
    gfx11 = kernel_src / 'amdgpu' / 'gfx_v11_0.c'
    if gfx11.exists():
        print("  Scanning gfx_v11_0.c for debug/test patterns...", flush=True)
        debug_refs = []
        trap_refs = []
        perf_refs = []
        with open(gfx11) as f:
            for i, line in enumerate(f, 1):
                ll = line.lower()
                if 'debug' in ll and ('reg' in ll or 'cntl' in ll or 'enable' in ll):
                    debug_refs.append(f"L{i}: {line.strip()[:100]}")
                if 'trap' in ll and ('handler' in ll or 'tba' in ll or 'tma' in ll or 'enable' in ll):
                    trap_refs.append(f"L{i}: {line.strip()[:100]}")
                if 'perf' in ll and ('count' in ll or 'enable' in ll or 'ctrl' in ll):
                    perf_refs.append(f"L{i}: {line.strip()[:100]}")
        findings['debug_references'] = debug_refs[:30]
        findings['trap_references'] = trap_refs[:30]
        findings['perf_references'] = perf_refs[:30]
        print(f"  Debug refs: {len(debug_refs)}, Trap refs: {len(trap_refs)}, Perf refs: {len(perf_refs)}")

    # 3. Look for register definitions in gc_11_0_0_offset.h
    offset_h = kernel_src / 'include' / 'asic_reg' / 'gc' / 'gc_11_0_0_offset.h'
    if offset_h.exists():
        print("  Parsing gc_11_0_0_offset.h for register map...", flush=True)
        interesting_regs = {}
        with open(offset_h) as f:
            for line in f:
                if '#define reg' in line and ('DEBUG' in line.upper() or 'TRAP' in line.upper()
                                               or 'PERF' in line.upper() or 'SCRATCH' in line.upper()
                                               or 'MEC' in line.upper() or 'CP_CPC' in line.upper()):
                    parts = line.strip().split()
                    if len(parts) >= 3 and '_BASE_IDX' not in parts[1]:
                        try:
                            val = int(parts[2], 16)
                            interesting_regs[parts[1]] = f'0x{val:04X}'
                        except ValueError:
                            pass
        findings['interesting_registers'] = interesting_regs
        print(f"  Interesting registers: {len(interesting_regs)}")

    # 4. Check for hidden/internal PM4 opcodes in KFD
    kfd_dir = kernel_src / '..' / 'amdkfd'  # try relative
    if not kfd_dir.exists():
        kfd_dir = kernel_src.parent / 'amdkfd'
    # Also check pm4 headers
    pm4_headers = list(kernel_src.rglob('*pm4*.h')) + list(kernel_src.rglob('*packet*.h'))
    if pm4_headers:
        print(f"  Found {len(pm4_headers)} PM4/packet headers", flush=True)
        extra_opcodes = {}
        for ph in pm4_headers:
            try:
                with open(ph) as f:
                    for line in f:
                        if 'PACKET3' in line or 'PM4_' in line or 'MES_' in line:
                            parts = line.strip().split()
                            if len(parts) >= 3 and '#define' in line:
                                try:
                                    val = int(parts[2], 16)
                                    if 0 < val < 0x100:
                                        extra_opcodes[val] = parts[1]
                                except:
                                    pass
            except:
                pass
        findings['extra_opcodes_from_headers'] = {f'0x{k:02X}': v for k, v in sorted(extra_opcodes.items())}
        print(f"  Extra opcodes: {len(extra_opcodes)}")

    results['kernel_xref'] = findings
    save_results()
    return findings


# ======================================================================
# STEP 5: Identify Exploitable Patterns
# ======================================================================
def identify_exploitable(dispatch_table, hwreg_summary, kernel_xref):
    """Identify potential attack surfaces based on firmware analysis."""
    print("\n" + "=" * 70)
    print("STEP 5: EXPLOITABLE PATTERN IDENTIFICATION")
    print("=" * 70, flush=True)

    findings = []
    txt_lines = ["EXPLOITABLE PATTERNS", "=" * 70, ""]
    txt_lines.append("SAFETY: This is READ-ONLY analysis. No exploitation attempted.")
    txt_lines.append("")

    # A) PM4 opcodes that write to arbitrary MMIO addresses
    txt_lines.append("=== A) ARBITRARY MMIO WRITE OPCODES ===")
    write_opcodes = ['WRITE_DATA', 'COPY_DATA', 'DMA_DATA', 'REG_RMW']
    for opc, entry in dispatch_table.items():
        if entry['pm4_name'] in write_opcodes:
            f = {
                'type': 'ARBITRARY_MMIO_WRITE',
                'pm4_opcode': entry['pm4_opcode'],
                'pm4_name': entry['pm4_name'],
                'risk': 'HIGH — can write to any MMIO register via PM4 packet',
                'notes': 'WRITE_DATA/COPY_DATA can target arbitrary register addresses',
            }
            findings.append(f)
            txt_lines.append(f"  {entry['pm4_opcode']} {entry['pm4_name']}: CAN write arbitrary MMIO")

    # B) Trap handler base address modification
    txt_lines.append("")
    txt_lines.append("=== B) TRAP HANDLER BASE (TBA/TMA) ===")
    trap_regs = [r for r in hwreg_summary if r in [0x0053, 0x004F]]
    if trap_regs:
        for r in trap_regs:
            f = {
                'type': 'TRAP_HANDLER_CONTROL',
                'register': f'0x{r:04X}',
                'name': CP_INTERNAL_REGS.get(r, 'unknown'),
                'risk': 'MEDIUM — CP internal trap/debug register accessed by firmware',
            }
            findings.append(f)
            txt_lines.append(f"  REG 0x{r:04X} ({CP_INTERNAL_REGS.get(r, '?')}): accessed by firmware")
    else:
        txt_lines.append("  No direct trap handler register access found in firmware")

    # C) Debug registers that might be writable
    txt_lines.append("")
    txt_lines.append("=== C) DEBUG REGISTERS ===")
    debug_regs = {
        0x00D9: 'CP_DEBUG_REG',
        0x004F: 'CP_SCRATCH/DEBUG',
        0x0040: 'CP_PERF_CTRL',
    }
    for reg_id, name in debug_regs.items():
        if reg_id in hwreg_summary:
            s = hwreg_summary[reg_id]
            f = {
                'type': 'DEBUG_REGISTER',
                'register': f'0x{reg_id:04X}',
                'name': name,
                'reads': s['read'],
                'writes': s['write'],
                'risk': 'MEDIUM — debug/perf register accessed by firmware',
            }
            findings.append(f)
            txt_lines.append(f"  REG 0x{reg_id:04X} ({name}): R={s['read']} W={s['write']}")

    # D) Timing oracles
    txt_lines.append("")
    txt_lines.append("=== D) TIMING ORACLES ===")
    # WAIT_REG_MEM is the most obvious timing oracle
    if 0x3C in dispatch_table:
        f = {
            'type': 'TIMING_ORACLE',
            'pm4_opcode': '0x3C',
            'pm4_name': 'WAIT_REG_MEM',
            'risk': 'LOW-MEDIUM — polls register until condition met, variable latency',
            'notes': 'Can measure internal state transitions via timing',
        }
        findings.append(f)
        txt_lines.append("  WAIT_REG_MEM (0x3C): polls register — timing side channel")

    # MEM_SEMAPHORE for synchronization timing
    if 0x39 in dispatch_table:
        f = {
            'type': 'TIMING_ORACLE',
            'pm4_opcode': '0x39',
            'pm4_name': 'MEM_SEMAPHORE',
            'risk': 'LOW — memory semaphore timing reveals contention state',
        }
        findings.append(f)
        txt_lines.append("  MEM_SEMAPHORE (0x39): reveals contention")

    # E) CP internal state readback
    txt_lines.append("")
    txt_lines.append("=== E) CP INTERNAL STATE READBACK ===")
    # Register 0x13 (CP_DISPATCH_STATE) — the most-accessed
    if 0x0013 in hwreg_summary:
        s = hwreg_summary[0x0013]
        f = {
            'type': 'INTERNAL_STATE_ACCESS',
            'register': '0x0013',
            'name': 'CP_DISPATCH_STATE',
            'reads': s['read'],
            'writes': s['write'],
            'risk': 'HIGH — most-referenced CP internal register, controls dispatch flow',
            'notes': f'Accessed by {len(hwreg_summary[0x0013].get("handlers", set()))} handlers. '
                     f'COPY_DATA PM4 might read this via internal source select.',
        }
        findings.append(f)
        txt_lines.append(f"  REG 0x0013 (CP_DISPATCH_STATE): R={s['read']} W={s['write']} — KEY REGISTER")

    # COPY_DATA can read internal CP registers (src_sel = 0 for register, dst = memory)
    if 0x40 in dispatch_table:
        f = {
            'type': 'CP_STATE_READBACK',
            'pm4_opcode': '0x40',
            'pm4_name': 'COPY_DATA',
            'risk': 'HIGH — COPY_DATA with src_sel=internal can read CP state to memory',
            'notes': 'src_sel=0 (REG), dst_sel=5 (MEM_ASYNC) — can dump any CP register to GPU memory',
        }
        findings.append(f)
        txt_lines.append("  COPY_DATA (0x40): can read CP internal state → GPU memory")

    # F) Undocumented PM4 opcodes
    txt_lines.append("")
    txt_lines.append("=== F) UNDOCUMENTED PM4 OPCODES ===")
    undoc = results['pm4_dispatch'].get('undocumented_list', [])
    for u in undoc:
        f = {
            'type': 'UNDOCUMENTED_PM4',
            'pm4_opcode': u['pm4_opcode'],
            'risk': 'UNKNOWN — behavior undetermined',
        }
        findings.append(f)
        txt_lines.append(f"  {u['pm4_opcode']}: UNDOCUMENTED handler exists in firmware")

    # G) NOP handler analysis
    txt_lines.append("")
    txt_lines.append("=== G) NOP HANDLER (0x10) ===")
    if 0x10 in dispatch_table:
        txt_lines.append("  NOP handler exists — may accept sub-commands in data words")
        txt_lines.append("  Kernel driver uses NOP with embedded data for padding and fence signals")
        f = {
            'type': 'NOP_HANDLER',
            'pm4_opcode': '0x10',
            'risk': 'LOW-MEDIUM — NOP handler processes data words, may have hidden sub-commands',
        }
        findings.append(f)

    # H) Queue management opcodes (MAP_QUEUES, UNMAP_QUEUES, SET_RESOURCES)
    txt_lines.append("")
    txt_lines.append("=== H) PRIVILEGED QUEUE MANAGEMENT ===")
    for opc in [0xA0, 0xA2, 0xA3, 0xA4]:
        if opc in dispatch_table:
            entry = dispatch_table[opc]
            f = {
                'type': 'QUEUE_MANAGEMENT',
                'pm4_opcode': entry['pm4_opcode'],
                'pm4_name': entry['pm4_name'],
                'risk': 'HIGH — changes queue/pipe mapping, normally KMD-only',
            }
            findings.append(f)
            txt_lines.append(f"  {entry['pm4_opcode']} {entry['pm4_name']}: queue management — normally privileged")

    # I) Scratch RAM access (used for CP state persistence)
    txt_lines.append("")
    txt_lines.append("=== I) SCRATCH RAM ACCESS ===")
    for opc in [0x7D, 0x7E, 0x80, 0x81, 0x83]:
        if opc in dispatch_table:
            entry = dispatch_table[opc]
            txt_lines.append(f"  {entry['pm4_opcode']} {entry['pm4_name']}: scratch/const RAM access")
            f = {
                'type': 'SCRATCH_RAM',
                'pm4_opcode': entry['pm4_opcode'],
                'pm4_name': entry['pm4_name'],
                'risk': 'MEDIUM — access to CP scratch RAM reveals/modifies internal state',
            }
            findings.append(f)

    txt_lines.append("")
    txt_lines.append(f"Total findings: {len(findings)}")

    save_txt('exploitable', '\n'.join(txt_lines))

    results['exploitable'] = {
        'total_findings': len(findings),
        'findings': findings,
        'summary': {
            'arbitrary_mmio_write': sum(1 for f in findings if f['type'] == 'ARBITRARY_MMIO_WRITE'),
            'debug_registers': sum(1 for f in findings if f['type'] == 'DEBUG_REGISTER'),
            'timing_oracles': sum(1 for f in findings if f['type'] == 'TIMING_ORACLE'),
            'state_readback': sum(1 for f in findings if f['type'] in ['INTERNAL_STATE_ACCESS', 'CP_STATE_READBACK']),
            'undocumented': sum(1 for f in findings if f['type'] == 'UNDOCUMENTED_PM4'),
            'queue_management': sum(1 for f in findings if f['type'] == 'QUEUE_MANAGEMENT'),
            'scratch_ram': sum(1 for f in findings if f['type'] == 'SCRATCH_RAM'),
        },
    }

    save_results()

    print(f"  Total findings: {len(findings)}")
    for ftype, count in results['exploitable']['summary'].items():
        if count > 0:
            print(f"    {ftype}: {count}")

    return findings


# ======================================================================
# STEP 6: Extended dispatch table — scan ALL possible PM4 opcodes
# ======================================================================
def scan_extended_dispatch(fw_path):
    """Scan for any PM4-like dispatch pattern beyond the known table."""
    print("\n" + "=" * 70)
    print("STEP 6: EXTENDED PM4 OPCODE SCAN")
    print("=" * 70, flush=True)

    data = open(fw_path, 'rb').read()
    ps1_off = data.find(b'$PS1')
    isa_off = ps1_off + 256 if ps1_off >= 0 else 0x100

    words = []
    for off in range(isa_off, len(data) - 3, 4):
        w = struct.unpack_from('<I', data, off)[0]
        words.append(w)

    # Search for ALL C00E_xxxx patterns (dispatch table loads to s14)
    # AND any triplet pattern (load + alu + store to 0xBF)
    all_table_entries = {}
    for i in range(len(words) - 2):
        w = words[i]
        if (w >> 16) == 0xC00E:  # s_mov s14, imm
            imm = w & 0xFFFF
            # Check store to 0xBF within 3 instructions
            for j in range(1, 4):
                if i + j < len(words):
                    wj = words[i + j]
                    if wj & 0xFFFF0000 == 0xCCC00000 and wj & 0xFFFF == 0x00BF:
                        pm4_opc = imm // 0x10
                        if pm4_opc not in all_table_entries:
                            all_table_entries[pm4_opc] = {
                                'key': f'0x{imm:04X}',
                                'offset': f'0x{isa_off + i*4:06X}',
                                'known': pm4_opc in PM4_OPCODES,
                                'name': PM4_OPCODES.get(pm4_opc, 'UNKNOWN'),
                            }
                        break

    # Also scan for large immediate loads that could be opcode indices
    # Some opcodes might use different table key scaling
    alt_entries = {}
    for i in range(len(words) - 2):
        w = words[i]
        if (w >> 24) == 0xC0:
            reg = (w >> 16) & 0xFF
            imm = w & 0xFFFF
            # Look for compare-then-branch patterns
            for j in range(1, 5):
                if i + j < len(words):
                    nw = words[i + j]
                    top = (nw >> 24) & 0xFF
                    if top in [0x94, 0x95, 0x97, 0x9A, 0x9B]:
                        if 0x10 <= imm <= 0xFF:
                            if imm not in alt_entries:
                                alt_entries[imm] = {
                                    'value': f'0x{imm:02X}',
                                    'reg': f's{reg}',
                                    'offset': f'0x{isa_off + i*4:06X}',
                                    'branch_type': f'0x{top:02X}',
                                    'known': imm in PM4_OPCODES,
                                    'name': PM4_OPCODES.get(imm, 'UNKNOWN'),
                                }
                        break

    # Report ALL opcodes (0x00-0xFF) and whether they have dispatch entries
    full_opcode_map = {}
    for opc in range(0x100):
        entry = {
            'in_main_table': opc in all_table_entries,
            'in_branch_check': opc in alt_entries,
            'known_name': PM4_OPCODES.get(opc, None),
        }
        if entry['in_main_table'] or entry['in_branch_check'] or entry['known_name']:
            full_opcode_map[f'0x{opc:02X}'] = entry

    # Identify undocumented opcodes WITH firmware handlers
    undocumented_with_handlers = []
    for opc, entry in sorted(all_table_entries.items()):
        if not entry['known'] and opc > 0:
            undocumented_with_handlers.append({
                'opcode': f'0x{opc:02X}',
                'key': entry['key'],
                'offset': entry['offset'],
            })

    print(f"  Main dispatch table entries: {len(all_table_entries)}")
    print(f"  Branch-check references: {len(alt_entries)}")
    print(f"  Undocumented with handlers: {len(undocumented_with_handlers)}")

    if undocumented_with_handlers:
        print("\n  UNDOCUMENTED PM4 OPCODES WITH FIRMWARE HANDLERS:")
        for u in undocumented_with_handlers:
            print(f"    {u['opcode']} — key={u['key']} @{u['offset']}")

    results['pm4_dispatch']['extended_scan'] = {
        'main_table_entries': len(all_table_entries),
        'branch_check_refs': len(alt_entries),
        'undocumented_handlers': undocumented_with_handlers,
        'full_opcode_map': full_opcode_map,
        'all_table_entries': all_table_entries,
    }

    save_results()
    return all_table_entries, undocumented_with_handlers


# ======================================================================
# MAIN
# ======================================================================
print("=" * 70)
print("z2339: Deep MEC/RLC/PFP Firmware RE — gfx1151 (Radeon 8060S)")
print("=" * 70)
print(f"Temperature: {get_temp()}C")
print(flush=True)

# Decompress firmware files
FW_DIR = Path('/tmp/fw_re')
FW_DIR.mkdir(exist_ok=True)

FW_FILES = {
    'mec': FW_DIR / 'gc_11_5_1_mec.bin',
    'rlc': FW_DIR / 'gc_11_5_1_rlc.bin',
    'pfp': FW_DIR / 'gc_11_5_1_pfp.bin',
    'me':  FW_DIR / 'gc_11_5_1_me.bin',
}

for name, path in FW_FILES.items():
    if not path.exists():
        zst = f'/lib/firmware/amdgpu/gc_11_5_1_{name}.bin.zst'
        if os.path.exists(zst):
            subprocess.run(['zstd', '-d', zst, '-o', str(path), '-f'],
                           check=True, capture_output=True)
            print(f"  Decompressed {zst} -> {path}")

# Record firmware metadata
for name, path in FW_FILES.items():
    if path.exists():
        data = open(path, 'rb').read()
        results['firmware'][name] = {
            'file': str(path),
            'size': len(data),
            'sha256': hashlib.sha256(data).hexdigest(),
        }

save_results()

# ---- STEP 1: MEC Analysis ----
if check_abort():
    sys.exit(1)
mec_instrs, mec_handlers, dispatch_table, hwreg_summary = analyze_mec(FW_FILES['mec'])

# ---- STEP 2: RLC Analysis ----
if check_abort():
    sys.exit(1)
rlc_instrs, rlc_handlers = analyze_rlc(FW_FILES['rlc'])

# ---- STEP 3: PFP and ME Analysis ----
if check_abort():
    sys.exit(1)
pfp_instrs = analyze_pfp_or_me(FW_FILES['pfp'], 'pfp')
me_instrs = analyze_pfp_or_me(FW_FILES['me'], 'me')

# ---- STEP 4: Kernel Cross-Reference ----
if check_abort():
    sys.exit(1)
kernel_xref = analyze_kernel_source()

# ---- STEP 5: Exploitable Patterns ----
if check_abort():
    sys.exit(1)
exploitable = identify_exploitable(dispatch_table, hwreg_summary, kernel_xref or {})

# ---- STEP 6: Extended Dispatch Table Scan ----
if check_abort():
    sys.exit(1)
ext_table, undoc_handlers = scan_extended_dispatch(FW_FILES['mec'])

# ---- Final Summary ----
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)
print(f"  MEC: {len(mec_handlers)} handlers, {len(dispatch_table)} PM4 dispatch entries")
print(f"  RLC: {len(rlc_handlers)} handlers")
print(f"  PFP: {results['pfp_analysis'].get('handler_count', '?')} handlers")
print(f"  ME:  {results['me_analysis'].get('handler_count', '?')} handlers")
print(f"  Undocumented PM4 opcodes: {len(undoc_handlers)}")
print(f"  Exploitable patterns: {results['exploitable']['total_findings']}")
print(f"  Temperature: {get_temp()}C")
print()
print("Output files:")
print(f"  {RESULTS / 'z2339_deep_mec_re.json'}")
print(f"  {RESULTS / 'z2339_mec_handlers.txt'}")
print(f"  {RESULTS / 'z2339_pm4_dispatch.txt'}")
print(f"  {RESULTS / 'z2339_rlc_analysis.txt'}")
print(f"  {RESULTS / 'z2339_exploitable.txt'}")
print(flush=True)

save_results()
print("\nDone.", flush=True)
