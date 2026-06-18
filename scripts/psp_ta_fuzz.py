#!/usr/bin/env python3
"""PSP RAS TA Fuzzer — probe PSP via ta_invoke with crafted inputs.

Commands:
  0: ENABLE_FEATURES  - block_id + error_type
  1: DISABLE_FEATURES - block_id + error_type
  2: TRIGGER_ERROR    - block_id + error_type + sub_block + address + value
  3: QUERY_BLOCK_INFO - (minimal input)
  4: QUERY_SUB_BLOCK_INFO - (minimal input)
  5: QUERY_ADDRESS    - addr_type + mca_addr/phy_addr

The shared_memory struct:
  cmd_id(4) + resp_id(4) + ras_status(4) + if_version(4) + input(1024) + output(1024)
"""

import struct
import os
import sys

TA_TYPE_RAS = 2
TA_IF_PATH = "/sys/kernel/debug/dri/0000:c3:00.0/ta_if"

# First we need to get the TA session ID. The RAS TA should already be loaded.
# We can try invoking with session_id = 0 first.

def build_invoke_buf(ta_type, ta_id, cmd_id, shared_buf):
    """Build the ta_invoke write buffer."""
    buf = struct.pack('<III', ta_type, ta_id, cmd_id)
    buf += struct.pack('<I', len(shared_buf))
    buf += shared_buf
    return buf

def build_ras_shared_memory(cmd_id, input_data=b''):
    """Build ta_ras_shared_memory struct."""
    # cmd_id(4) + resp_id(4) + ras_status(4) + if_version(4) = 16 bytes header
    # Then input union (1024 bytes) + output union (1024 bytes)
    hdr = struct.pack('<IIII', cmd_id, 0, 0, 0)
    input_padded = input_data + b'\x00' * (1024 - len(input_data))
    output_padded = b'\x00' * 1024
    return hdr + input_padded + output_padded

def build_trigger_error(block_id, error_type, sub_block, address, value):
    """Build trigger_error_input struct."""
    # block_id(4) + error_type(4) + sub_block_index(4) + padding(4) + address(8) + value(8)
    return struct.pack('<IIIIqq', block_id, error_type, sub_block, 0, address, value)

def build_query_address(addr_type, err_addr, ch_inst, umc_inst, node_inst, socket_id, pa, bank, channel_idx):
    """Build query_address_input struct."""
    return struct.pack('<I', addr_type) + \
           struct.pack('<QIIII', err_addr, ch_inst, umc_inst, node_inst, socket_id) + \
           struct.pack('<QII', pa, bank, channel_idx)

def try_invoke(cmd_id, input_data=b'', ta_id=0, desc=""):
    """Try to invoke RAS TA with given command."""
    shared = build_ras_shared_memory(cmd_id, input_data)
    buf = build_invoke_buf(TA_TYPE_RAS, ta_id, cmd_id, shared)

    invoke_path = os.path.join(TA_IF_PATH, "ta_invoke")
    print(f"[{desc}] cmd={cmd_id} ta_id={ta_id} buf_len={len(buf)}...", end=" ")

    try:
        fd = os.open(invoke_path, os.O_WRONLY)
        try:
            n = os.write(fd, buf)
            print(f"wrote {n} bytes OK")
        except OSError as e:
            print(f"write error: {e}")
        finally:
            os.close(fd)
    except OSError as e:
        print(f"open error: {e}")

def main():
    print("=== PSP RAS TA Fuzzer ===")
    print(f"TA interface: {TA_IF_PATH}")

    # Test 1: QUERY_BLOCK_INFO with valid block IDs
    print("\n--- Test 1: QUERY_BLOCK_INFO ---")
    for block_id in range(17):  # 0-16 are valid blocks
        input_data = struct.pack('<I', block_id)
        try_invoke(3, input_data, desc=f"BLOCK_INFO block={block_id}")

    # Test 2: TRIGGER_ERROR with TMR address
    print("\n--- Test 2: TRIGGER_ERROR with interesting addresses ---")
    addresses = [
        (0x97E0000000, "TMR_BO_base"),
        (0x97FF7A3000, "TMR_ioremap"),
        (0x80007EE000, "FW_trampoline_BO"),
        (0x0000000000, "zero"),
        (0xFFFFFFFFFFFFFFFF, "max"),
    ]
    for addr, name in addresses:
        for block_id in [0, 2, 8]:  # UMC, GFX, DF
            input_data = build_trigger_error(block_id, 2, 0, addr, 0)
            try_invoke(2, input_data, desc=f"TRIGGER block={block_id} addr={name}")

    # Test 3: QUERY_ADDRESS — try to translate TMR physical address
    print("\n--- Test 3: QUERY_ADDRESS ---")
    tmr_phys = 0x2060000000  # TMR BO physical address
    input_data = build_query_address(1, 0, 0, 0, 0, 0, tmr_phys, 0, 0)  # PA_TO_MCA
    try_invoke(5, input_data, desc="PA_TO_MCA tmr_phys")

    input_data = build_query_address(0, tmr_phys, 0, 0, 0, 0, 0, 0, 0)  # MCA_TO_PA
    try_invoke(5, input_data, desc="MCA_TO_PA tmr_addr")

    # Test 4: Invalid command IDs
    print("\n--- Test 4: Invalid/boundary commands ---")
    for cmd_id in [6, 7, 0xFF, 0x100, 0xFFFFFFFF]:
        try_invoke(cmd_id, desc=f"invalid_cmd={cmd_id}")

    # Test 5: Oversized shared buffer
    print("\n--- Test 5: Oversized buffer ---")
    huge_buf = b'\x41' * 0x10000  # 64KB of 'A's
    try_invoke(3, huge_buf, desc="oversized_64K")

    # Test 6: TRIGGER_ERROR with DF block and crafted address
    print("\n--- Test 6: DF block error injection ---")
    # TA_RAS_BLOCK__DF = 8, might interact with Data Fabric memory protection
    for val in [0x1, 0xFF, 0xDEADBEEF]:
        input_data = build_trigger_error(8, 4, 0, 0x97E0000000, val)
        try_invoke(2, input_data, desc=f"DF_TRIGGER val={val:#x}")

    print("\n=== Fuzzing complete ===")
    print("Check dmesg for PSP responses and errors")

if __name__ == "__main__":
    main()
