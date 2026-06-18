#!/usr/bin/env python3
"""z2352g: Find actual MEC firmware in VRAM by scanning for f32 opcodes.

We know:
- MEC firmware is f32 format (not RS64, not encrypted)
- Known opcodes: 0xC424000B, 0x800003B0, 0xD800008B, 0xC0310800
- PSP loaded it into VRAM somewhere
- VRAM is writable through BAR0
- BAR0 bypasses TSME

Strategy: Scan VRAM in 1MB chunks looking for f32 opcode patterns.
Also check the firmware file header for the exact ucode body to match.
"""
import mmap, struct, os, json, time, sys, zstandard

BAR0_PHYS = 0x6800000000
MMIO_BASE = 0xB4400000
VRAM_SIZE = 256 * 1024 * 1024  # 256MB visible through BAR0
results = {'start_time': time.strftime('%Y-%m-%d %H:%M:%S')}

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def save(tag=""):
    p = '/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/z2352_find_fw_vram.json'
    with open(p, 'w') as f:
        json.dump(results, f, indent=2)
    if tag:
        log(f"  [saved: {tag}]")

# === STEP 0: Extract known firmware opcodes from the MEC firmware file ===
log("--- Step 0: Extract MEC firmware signature bytes ---")
fw_path = '/lib/firmware/amdgpu/gc_11_5_1_mec.bin.zst'
try:
    with open(fw_path, 'rb') as f:
        compressed = f.read()
    dctx = zstandard.ZstdDecompressor()
    fw_data = dctx.decompress(compressed)
    log(f"  MEC firmware: {len(fw_data)} bytes")

    # Parse header
    # struct common_firmware_header (offset 4 = ucode_size_bytes, offset 8 = ucode_array_offset_bytes)
    hdr_size = struct.unpack_from('<I', fw_data, 4)[0]   # header size
    ucode_size = struct.unpack_from('<I', fw_data, 8)[0]  # ucode_size_bytes
    ucode_offset = struct.unpack_from('<I', fw_data, 12)[0]  # ucode_array_offset_bytes
    log(f"  Header size: {hdr_size}, ucode_size: 0x{ucode_size:X}, ucode_offset: 0x{ucode_offset:X}")

    # Extract first 64 bytes of actual ucode body as signature
    ucode_body = fw_data[ucode_offset:ucode_offset+ucode_size]
    fw_sig_words = [struct.unpack_from('<I', ucode_body, i*4)[0] for i in range(min(16, len(ucode_body)//4))]
    log(f"  First 16 ucode DWORDs (search signature):")
    for i, w in enumerate(fw_sig_words):
        log(f"    [{i:3d}] 0x{w:08X}")

    # Also get a mid-section signature (offset 0x1000 into ucode body)
    mid_off = 0x1000
    if len(ucode_body) > mid_off + 64:
        mid_sig = [struct.unpack_from('<I', ucode_body, mid_off + i*4)[0] for i in range(16)]
        log(f"  Mid-section signature (offset 0x{mid_off:X}):")
        for i, w in enumerate(mid_sig):
            log(f"    [{i:3d}] 0x{w:08X}")
    else:
        mid_sig = []

    results['fw_info'] = {
        'size': len(fw_data),
        'ucode_size': ucode_size,
        'ucode_offset': ucode_offset,
        'first_16': [f"0x{w:08X}" for w in fw_sig_words],
    }
    save("fw_info")

except Exception as e:
    log(f"  Firmware parse error: {e}")
    fw_sig_words = [0xC424000B, 0x800003B0]  # fallback known opcodes
    ucode_body = None
    results['fw_info'] = {'error': str(e)}

# === STEP 1: Scan VRAM for firmware signature ===
log("\n--- Step 1: Scan VRAM for MEC firmware ---")
CHUNK = 4 * 1024 * 1024  # 4MB chunks
sig_word_0 = fw_sig_words[0] if fw_sig_words else 0xC424000B
sig_word_1 = fw_sig_words[1] if len(fw_sig_words) > 1 else 0

found_locations = []
vram_fd = os.open('/dev/mem', os.O_RDONLY | os.O_SYNC)

for chunk_start in range(0, min(VRAM_SIZE, 256*1024*1024), CHUNK):
    try:
        vram_mm = mmap.mmap(vram_fd, CHUNK, mmap.MAP_SHARED,
                           mmap.PROT_READ, offset=BAR0_PHYS + chunk_start)

        # Search for first signature word
        data = vram_mm.read()
        vram_mm.close()

        # Pack signature as bytes for fast search
        sig_bytes = struct.pack('<I', sig_word_0)
        pos = 0
        while True:
            idx = data.find(sig_bytes, pos)
            if idx == -1:
                break
            # Found first word — check second word
            if idx + 4 < len(data):
                second = struct.unpack_from('<I', data, idx + 4)[0]
                if second == sig_word_1:
                    abs_offset = chunk_start + idx
                    # Read context: 16 words around match
                    context_words = []
                    for j in range(min(16, (len(data) - idx) // 4)):
                        context_words.append(struct.unpack_from('<I', data, idx + j*4)[0])

                    log(f"  MATCH at VRAM offset 0x{abs_offset:08X}")
                    for k, w in enumerate(context_words[:8]):
                        log(f"    [{k}] 0x{w:08X}")
                    found_locations.append({
                        'offset': f"0x{abs_offset:08X}",
                        'context': [f"0x{w:08X}" for w in context_words[:16]],
                    })
            pos = idx + 4

    except Exception as e:
        log(f"  Chunk 0x{chunk_start:08X} error: {e}")
        break

    # Progress every 32MB
    if chunk_start > 0 and chunk_start % (32*1024*1024) == 0:
        log(f"  Scanned {chunk_start // (1024*1024)}MB...")

os.close(vram_fd)

log(f"\n  Total matches: {len(found_locations)}")
results['vram_fw_locations'] = found_locations
save("vram_scan")

# === STEP 2: If found, verify full match against firmware file ===
if found_locations and ucode_body is not None:
    log("\n--- Step 2: Verify full firmware match ---")
    for loc in found_locations[:3]:  # Check first 3 matches
        off = int(loc['offset'], 16)
        log(f"  Verifying at 0x{off:08X}...")
        try:
            vram_fd = os.open('/dev/mem', os.O_RDONLY | os.O_SYNC)
            # Read enough to cover the ucode body
            read_size = min(len(ucode_body), VRAM_SIZE - off)
            if read_size < 256:
                continue
            vram_mm = mmap.mmap(vram_fd, read_size, mmap.MAP_SHARED,
                               mmap.PROT_READ, offset=BAR0_PHYS + off)
            vram_data = vram_mm.read()
            vram_mm.close()
            os.close(vram_fd)

            # Compare
            match_bytes = 0
            total_check = min(len(ucode_body), len(vram_data))
            for i in range(total_check):
                if vram_data[i] == ucode_body[i]:
                    match_bytes += 1

            match_pct = match_bytes / total_check * 100
            log(f"  Match: {match_bytes}/{total_check} bytes ({match_pct:.1f}%)")

            # Find first mismatch
            first_mismatch = -1
            for i in range(total_check):
                if vram_data[i] != ucode_body[i]:
                    first_mismatch = i
                    break

            if first_mismatch >= 0:
                log(f"  First mismatch at offset {first_mismatch} (0x{first_mismatch:X})")
                fw_word = struct.unpack_from('<I', ucode_body, first_mismatch & ~3)[0]
                vram_word = struct.unpack_from('<I', vram_data, first_mismatch & ~3)[0]
                log(f"    FW: 0x{fw_word:08X}  VRAM: 0x{vram_word:08X}")

            loc['match_pct'] = match_pct
            loc['first_mismatch'] = first_mismatch
            loc['total_checked'] = total_check

        except Exception as e:
            log(f"  Verify error: {e}")
            loc['verify_error'] = str(e)

    save("verify")

# === STEP 3: Also scan for $PS1 PSP headers in VRAM ===
log("\n--- Step 3: Scan for $PS1 headers in VRAM ---")
ps1_sig = b'$PS1'
ps1_locations = []
vram_fd = os.open('/dev/mem', os.O_RDONLY | os.O_SYNC)

for chunk_start in range(0, min(VRAM_SIZE, 256*1024*1024), CHUNK):
    try:
        vram_mm = mmap.mmap(vram_fd, CHUNK, mmap.MAP_SHARED,
                           mmap.PROT_READ, offset=BAR0_PHYS + chunk_start)
        data = vram_mm.read()
        vram_mm.close()

        pos = 0
        while True:
            idx = data.find(ps1_sig, pos)
            if idx == -1:
                break
            abs_off = chunk_start + idx
            # Read header context
            hdr_words = [struct.unpack_from('<I', data, idx + i*4)[0]
                        for i in range(min(8, (len(data)-idx)//4))]
            ps1_locations.append({
                'offset': f"0x{abs_off:08X}",
                'header': [f"0x{w:08X}" for w in hdr_words],
            })
            log(f"  $PS1 at VRAM 0x{abs_off:08X}: {[f'0x{w:08X}' for w in hdr_words[:4]]}")
            pos = idx + 4
    except:
        break

os.close(vram_fd)
log(f"  Total $PS1 headers: {len(ps1_locations)}")
results['ps1_locations'] = ps1_locations
save("ps1_scan")

results['end_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
save("FINAL")
log("\nDone.")
