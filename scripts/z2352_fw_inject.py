#!/usr/bin/env python3
"""
z2352_fw_inject.py — MEC Firmware Injection for gfx1151 (RDNA4)

Creates a modified gc_11_5_1_mec.bin with:
  Phase 1 (PoC): Marker pattern in zero region + custom ucode_version
  Phase 2 (Active): Neuromorphic handler redirected via dispatch table

Injection vector:
  - fw_load_type=0 forces AMDGPU_FW_LOAD_DIRECT (no PSP validation)
  - Driver only checks size_bytes == file_size (no CRC verification)
  - MEC firmware is plaintext (not encrypted), entropy 2.43 bits/byte
  - 186KB zero region (dwords 18871-65599) available for payload

Usage:
  sudo python3 z2352_fw_inject.py --phase1          # Create PoC firmware
  sudo python3 z2352_fw_inject.py --install          # Install + reload driver
  sudo python3 z2352_fw_inject.py --verify           # Check if injection active
  sudo python3 z2352_fw_inject.py --restore          # Restore original
"""
import struct, subprocess, shutil, os, sys, json, time, argparse

FW_PATH = "/lib/firmware/amdgpu/gc_11_5_1_mec.bin.zst"
FW_BACKUP = "/lib/firmware/amdgpu/gc_11_5_1_mec.bin.zst.orig"
FW_MOD_RAW = "/lib/firmware/amdgpu/gc_11_5_1_mec_feel.bin"
FW_MOD_ZST = "/lib/firmware/amdgpu/gc_11_5_1_mec.bin.zst.feel"

# Results output
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")

# Firmware layout constants (verified from binary analysis)
HEADER_SIZE = 44           # gfx_firmware_header_v1_0
UCODE_ARRAY_OFFSET = 256   # padded header
UCODE_SIZE_DWORDS = 66976
ZERO_REGION_START = 18871   # first zero dword after active code
ZERO_REGION_END = 65600     # address table starts here
JT_OFFSET_DWORDS = 66752    # jump table start
JT_SIZE_DWORDS = 224
ORIG_UCODE_VERSION = 0x20
ORIG_FW_FEATURE = 0x23

# Our custom version marker
FEEL_UCODE_VERSION = 0xFEE1
FEEL_MAGIC = 0xFEE10001     # marker in zero region

# FEEL payload marker pattern (written to zero region for verification)
FEEL_MARKER = [
    0xFEE10001,  # FEEL magic
    0x00000001,  # version 1
    0xDEAD1151,  # gfx1151 identifier
    0x00000000,  # reserved
    0x4E455552,  # "NEUR" (neuromorphic)
    0x4F4D5250,  # "OMRP" (omorphic)
    0x48494320,  # "HIC " (hardware injection code)
    0x00000000,  # payload length (to be filled)
]


def load_firmware():
    """Load and decompress the original MEC firmware."""
    result = subprocess.run(['zstd', '-d', '-c', FW_PATH],
                          capture_output=True)
    if result.returncode != 0:
        print(f"ERROR: Failed to decompress {FW_PATH}")
        print(f"  stderr: {result.stderr.decode()[:200]}")
        sys.exit(1)
    return bytearray(result.stdout)


def parse_header(fw):
    """Parse gfx_firmware_header_v1_0 and return dict."""
    h = {}
    h['size_bytes'] = struct.unpack_from('<I', fw, 0)[0]
    h['header_size_bytes'] = struct.unpack_from('<I', fw, 4)[0]
    h['hdr_ver_major'], h['hdr_ver_minor'] = struct.unpack_from('<HH', fw, 8)
    h['ip_ver_major'], h['ip_ver_minor'] = struct.unpack_from('<HH', fw, 12)
    h['ucode_version'] = struct.unpack_from('<I', fw, 16)[0]
    h['ucode_size_bytes'] = struct.unpack_from('<I', fw, 20)[0]
    h['ucode_array_offset'] = struct.unpack_from('<I', fw, 24)[0]
    h['crc32'] = struct.unpack_from('<I', fw, 28)[0]
    h['feature_version'] = struct.unpack_from('<I', fw, 32)[0]
    h['jt_offset'] = struct.unpack_from('<I', fw, 36)[0]
    h['jt_size'] = struct.unpack_from('<I', fw, 40)[0]
    return h


def create_phase1_firmware():
    """
    Phase 1: Proof-of-concept injection.
    - Write FEEL marker pattern into the zero region (dwords 18871+)
    - Change ucode_version to 0xFEE1 (verifiable via sysfs)
    - Keep all active code and JT unchanged
    """
    fw = load_firmware()
    hdr = parse_header(fw)

    print("=== Original Firmware Header ===")
    for k, v in hdr.items():
        if isinstance(v, int):
            print(f"  {k:25s} = 0x{v:08X} ({v})")

    # Verify structure matches expectations
    assert hdr['header_size_bytes'] == HEADER_SIZE, f"Header size mismatch: {hdr['header_size_bytes']}"
    assert hdr['ucode_array_offset'] == UCODE_ARRAY_OFFSET, f"Ucode offset mismatch"
    assert hdr['ucode_version'] == ORIG_UCODE_VERSION, f"Ucode version mismatch: 0x{hdr['ucode_version']:X}"
    assert hdr['jt_offset'] == JT_OFFSET_DWORDS, f"JT offset mismatch"
    assert hdr['jt_size'] == JT_SIZE_DWORDS, f"JT size mismatch"

    # Verify zero region is actually zero
    ucode_base = UCODE_ARRAY_OFFSET
    for i in range(ZERO_REGION_START, min(ZERO_REGION_START + 100, ZERO_REGION_END)):
        dw = struct.unpack_from('<I', fw, ucode_base + i * 4)[0]
        if dw != 0:
            print(f"WARNING: Zero region not zero at dword {i}: 0x{dw:08X}")
            print("Aborting — firmware layout doesn't match analysis")
            sys.exit(1)

    print(f"\nZero region verified clean: dwords {ZERO_REGION_START}-{ZERO_REGION_END}")
    print(f"Injectable space: {ZERO_REGION_END - ZERO_REGION_START} dwords "
          f"({(ZERO_REGION_END - ZERO_REGION_START) * 4} bytes)")

    # === MODIFICATION 1: Write FEEL marker into zero region ===
    marker_offset = ucode_base + ZERO_REGION_START * 4
    for i, val in enumerate(FEEL_MARKER):
        struct.pack_into('<I', fw, marker_offset + i * 4, val)
    print(f"\nWrote {len(FEEL_MARKER)} marker dwords at ucode offset {ZERO_REGION_START}")

    # === MODIFICATION 2: Change ucode_version ===
    # This will be visible in: /sys/class/drm/card*/device/fw_version
    # and dmesg: "mec firmware version = 0x0000FEE1"
    struct.pack_into('<I', fw, 16, FEEL_UCODE_VERSION)
    print(f"Changed ucode_version: 0x{ORIG_UCODE_VERSION:08X} -> 0x{FEEL_UCODE_VERSION:08X}")

    # === DO NOT modify size_bytes === (validation checks this)
    # === DO NOT modify CRC32 === (driver doesn't check it anyway)
    # === DO NOT modify JT or active code === (Phase 1 = passive marker only)

    # Save modified firmware (uncompressed for now)
    with open(FW_MOD_RAW, 'wb') as f:
        f.write(fw)
    print(f"\nSaved uncompressed modified firmware: {FW_MOD_RAW} ({len(fw)} bytes)")

    # Compress with zstd
    subprocess.run(['zstd', '-f', '-19', FW_MOD_RAW, '-o', FW_MOD_ZST], check=True)
    print(f"Saved compressed: {FW_MOD_ZST}")

    # Verify round-trip
    verify = subprocess.run(['zstd', '-d', '-c', FW_MOD_ZST], capture_output=True).stdout
    if verify == bytes(fw):
        print("Round-trip verification: OK")
    else:
        print("WARNING: Round-trip mismatch!")

    return fw


def install_firmware():
    """
    Install modified firmware and reload amdgpu with fw_load_type=0.
    REQUIRES ROOT.
    """
    if os.geteuid() != 0:
        print("ERROR: Must run as root (sudo)")
        sys.exit(1)

    if not os.path.exists(FW_MOD_ZST):
        print("ERROR: Modified firmware not found. Run --phase1 first.")
        sys.exit(1)

    # Step 1: Backup original
    if not os.path.exists(FW_BACKUP):
        shutil.copy2(FW_PATH, FW_BACKUP)
        print(f"Backed up original: {FW_BACKUP}")
    else:
        print(f"Backup already exists: {FW_BACKUP}")

    # Step 2: Install modified firmware
    shutil.copy2(FW_MOD_ZST, FW_PATH)
    print(f"Installed modified firmware: {FW_PATH}")

    # Step 3: Reload amdgpu with DIRECT firmware loading
    print("\n=== Reloading amdgpu driver ===")
    print("WARNING: Display may flicker or go black temporarily!")
    print("WARNING: If system hangs, reboot and run --restore")

    # Save state
    state = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "phase": "install",
        "backup_path": FW_BACKUP,
        "original_version": f"0x{ORIG_UCODE_VERSION:08X}",
        "modified_version": f"0x{FEEL_UCODE_VERSION:08X}",
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    state_file = os.path.join(RESULTS_DIR, "z2352_fw_inject_state.json")
    with open(state_file, 'w') as f:
        json.dump(state, f, indent=2)
    print(f"Saved state: {state_file}")

    input("\nPress ENTER to proceed with driver reload (Ctrl+C to abort)...")

    # Unload amdgpu
    print("Unloading amdgpu...")
    r = subprocess.run(['modprobe', '-r', 'amdgpu'], capture_output=True)
    if r.returncode != 0:
        print(f"modprobe -r failed: {r.stderr.decode()[:200]}")
        print("Try: sudo systemctl stop gdm3 && sudo modprobe -r amdgpu")
        sys.exit(1)

    time.sleep(2)

    # Reload with DIRECT firmware loading (fw_load_type=0)
    print("Loading amdgpu with fw_load_type=0 (DIRECT, bypasses PSP)...")
    r = subprocess.run(['modprobe', 'amdgpu', 'fw_load_type=0'], capture_output=True)
    if r.returncode != 0:
        print(f"modprobe failed: {r.stderr.decode()[:200]}")
        print("Attempting restore...")
        restore_firmware()
        sys.exit(1)

    time.sleep(3)
    print("Driver reloaded successfully!")


def verify_injection():
    """Check if modified firmware is active."""
    results = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "checks": {}}

    # Check 1: fw_version in sysfs
    fw_ver_paths = []
    import glob
    for p in glob.glob("/sys/class/drm/card*/device/fw_version"):
        fw_ver_paths.append(p)

    for p in fw_ver_paths:
        try:
            with open(p) as f:
                content = f.read().strip()
            results["checks"]["sysfs_fw_version"] = content
            print(f"sysfs fw_version ({p}):\n{content}")
        except Exception as e:
            results["checks"]["sysfs_fw_version"] = f"ERROR: {e}"

    # Check 2: dmesg for MEC firmware version
    r = subprocess.run(['dmesg'], capture_output=True)
    dmesg = r.stdout.decode()
    mec_lines = [l for l in dmesg.split('\n') if 'mec' in l.lower() and 'firmware' in l.lower()]
    results["checks"]["dmesg_mec"] = mec_lines[-5:] if mec_lines else "no mec firmware lines"
    print(f"\ndmesg MEC lines:")
    for l in mec_lines[-5:]:
        print(f"  {l}")

    # Check 3: fw_load_type
    load_type_lines = [l for l in dmesg.split('\n') if 'load_type' in l.lower() or 'fw_load' in l.lower()]
    fw_direct_lines = [l for l in dmesg.split('\n') if 'DIRECT' in l or 'direct' in l.lower()]
    results["checks"]["dmesg_load_type"] = load_type_lines[-3:] if load_type_lines else "none"
    print(f"\nFW load type lines:")
    for l in (load_type_lines + fw_direct_lines)[-5:]:
        print(f"  {l}")

    # Check 4: Verify marker in loaded firmware file
    try:
        fw = load_firmware()
        ucode_ver = struct.unpack_from('<I', fw, 16)[0]
        marker = struct.unpack_from('<I', fw, UCODE_ARRAY_OFFSET + ZERO_REGION_START * 4)[0]
        results["checks"]["current_fw_ucode_ver"] = f"0x{ucode_ver:08X}"
        results["checks"]["current_fw_marker"] = f"0x{marker:08X}"
        results["checks"]["is_feel_firmware"] = (ucode_ver == FEEL_UCODE_VERSION and marker == FEEL_MAGIC)
        print(f"\nCurrent firmware file:")
        print(f"  ucode_version: 0x{ucode_ver:08X} ({'FEEL' if ucode_ver == FEEL_UCODE_VERSION else 'ORIGINAL'})")
        print(f"  marker:        0x{marker:08X} ({'FEEL' if marker == FEEL_MAGIC else 'NONE'})")
    except Exception as e:
        results["checks"]["current_fw"] = f"ERROR: {e}"

    # Check 5: Module parameter
    try:
        with open("/sys/module/amdgpu/parameters/fw_load_type") as f:
            fw_lt = f.read().strip()
        results["checks"]["fw_load_type_param"] = fw_lt
        print(f"\namdgpu fw_load_type parameter: {fw_lt}")
        print(f"  {'DIRECT (bypass PSP)' if fw_lt == '0' else 'PSP (default)' if fw_lt == '1' else fw_lt}")
    except Exception as e:
        results["checks"]["fw_load_type_param"] = f"ERROR: {e}"

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, "z2352_fw_inject_verify.json")
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {out}")


def restore_firmware():
    """Restore original firmware."""
    if os.geteuid() != 0:
        print("ERROR: Must run as root (sudo)")
        sys.exit(1)

    if not os.path.exists(FW_BACKUP):
        print(f"ERROR: No backup found at {FW_BACKUP}")
        sys.exit(1)

    shutil.copy2(FW_BACKUP, FW_PATH)
    print(f"Restored original firmware from {FW_BACKUP}")

    # Reload driver with default settings
    print("Reloading amdgpu with default settings...")
    subprocess.run(['modprobe', '-r', 'amdgpu'], capture_output=True)
    time.sleep(2)
    subprocess.run(['modprobe', 'amdgpu'], capture_output=True)
    time.sleep(3)
    print("Driver reloaded with original firmware.")


def main():
    parser = argparse.ArgumentParser(description="MEC Firmware Injection for gfx1151")
    parser.add_argument('--phase1', action='store_true', help='Create PoC firmware')
    parser.add_argument('--install', action='store_true', help='Install + reload driver')
    parser.add_argument('--verify', action='store_true', help='Check injection status')
    parser.add_argument('--restore', action='store_true', help='Restore original firmware')
    parser.add_argument('--analyze', action='store_true', help='Analyze current firmware')
    args = parser.parse_args()

    if args.phase1:
        create_phase1_firmware()
    elif args.install:
        install_firmware()
    elif args.verify:
        verify_injection()
    elif args.restore:
        restore_firmware()
    elif args.analyze:
        fw = load_firmware()
        hdr = parse_header(fw)
        print("=== Current Firmware Header ===")
        for k, v in hdr.items():
            if isinstance(v, int):
                print(f"  {k:25s} = 0x{v:08X} ({v})")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
