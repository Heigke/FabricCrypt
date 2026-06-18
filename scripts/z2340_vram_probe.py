#!/usr/bin/env python3
"""z2340: Deep VRAM access and debug interface probe for AMD Radeon 8060S (gfx1151).
READ-ONLY operations. TMZ disabled = VRAM readable from root.
"""
import os, sys, time, struct, json, subprocess, numpy as np
from pathlib import Path
from collections import OrderedDict

RESULTS_DIR = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results")
RESULTS_DIR.mkdir(exist_ok=True)

DEBUGFS = "/sys/kernel/debug/dri/0"
HWMON = "/sys/class/drm/card0/device/hwmon/hwmon7"
GPU_METRICS = "/sys/class/drm/card0/device/gpu_metrics"
THERMAL = "/sys/class/thermal/thermal_zone0/temp"
VRAM_FILE = f"{DEBUGFS}/amdgpu_vram"
MAX_TEMP = 85000  # 85C in millidegrees

def check_thermal():
    t = int(open(THERMAL).read().strip())
    if t > MAX_TEMP:
        print(f"ABORT: thermal {t/1000:.1f}C > 85C limit")
        sys.exit(1)
    return t / 1000

def save_result(name, content):
    path = RESULTS_DIR / name
    with open(path, 'w') as f:
        f.write(content)
    print(f"  Saved: {path}")

# ====================== STEP 1: VRAM ACCESS ======================
def step1_vram_access():
    print("\n" + "="*60)
    print("STEP 1: Direct VRAM Access")
    print("="*60)
    out = []

    # 1a) Read VRAM via debugfs
    vram_size = os.path.getsize(VRAM_FILE) if os.path.exists(VRAM_FILE) else 0
    out.append(f"VRAM file: {VRAM_FILE}")
    out.append(f"VRAM virtual size: {vram_size} bytes ({vram_size/(1024**3):.1f} GB)")

    try:
        with open(VRAM_FILE, 'rb') as f:
            # Read first 4KB
            data = f.read(4096)
            out.append(f"\nFirst 4KB readable: YES ({len(data)} bytes)")
            out.append(f"First 64 bytes hex: {data[:64].hex()}")
            out.append(f"All zeros: {all(b == 0 for b in data)}")

            # Check entropy (stale data vs fresh)
            byte_counts = np.zeros(256)
            for b in data:
                byte_counts[b] += 1
            entropy = -np.sum(byte_counts[byte_counts > 0] / len(data) * np.log2(byte_counts[byte_counts > 0] / len(data)))
            out.append(f"Entropy of first 4KB: {entropy:.3f} bits/byte (8.0 = random, 0 = constant)")

            # Sample at different offsets
            offsets_to_check = [0, 0x10000, 0x100000, 0x1000000, 0x10000000]
            out.append(f"\n--- VRAM Content at Various Offsets ---")
            for off in offsets_to_check:
                try:
                    f.seek(off)
                    chunk = f.read(64)
                    if chunk:
                        zero_pct = sum(1 for b in chunk if b == 0) / len(chunk) * 100
                        out.append(f"  Offset 0x{off:08X}: {chunk[:32].hex()} ... zeros={zero_pct:.0f}%")
                except Exception as e:
                    out.append(f"  Offset 0x{off:08X}: ERROR: {e}")

            # Read at 2GB boundary (above normal small VRAM)
            try:
                f.seek(0x80000000)  # 2GB
                chunk = f.read(64)
                out.append(f"  Offset 0x80000000 (2GB): {chunk[:32].hex() if chunk else 'empty'}")
            except Exception as e:
                out.append(f"  Offset 0x80000000 (2GB): ERROR: {e}")

    except PermissionError:
        out.append("VRAM read: PERMISSION DENIED (need root)")
    except Exception as e:
        out.append(f"VRAM read: ERROR: {e}")

    check_thermal()

    # 1b) PCI BAR info
    out.append(f"\n--- PCI BAR Resources ---")
    pci_path = "/sys/bus/pci/devices/0000:c3:00.0"
    try:
        with open(f"{pci_path}/resource") as f:
            bars = f.readlines()
        for i, line in enumerate(bars):
            parts = line.strip().split()
            if len(parts) >= 3:
                start, end, flags = int(parts[0], 16), int(parts[1], 16), int(parts[2], 16)
                if end > 0:
                    size = end - start + 1
                    out.append(f"  BAR{i}: 0x{start:012X}-0x{end:012X} ({size/(1024*1024):.1f} MB) flags=0x{flags:08X}")
    except Exception as e:
        out.append(f"  PCI resource read error: {e}")

    # 1c) Check iomem
    out.append(f"\n--- /proc/iomem GPU entries ---")
    try:
        result = subprocess.run(['sudo', 'grep', '-i', 'c3:00', '/proc/iomem'],
                              capture_output=True, text=True)
        out.append(result.stdout.strip())
    except:
        pass

    result = '\n'.join(out)
    save_result("z2340_vram_access.txt", result)
    print(result)
    return vram_size > 0

# ====================== STEP 2: DEBUG REGISTERS ======================
def step2_debug_interfaces():
    print("\n" + "="*60)
    print("STEP 2: Debug Register Probing")
    print("="*60)
    out = []

    # 2a) List all debugfs files with sizes
    out.append("--- All debugfs files ---")
    try:
        for entry in sorted(os.listdir(DEBUGFS)):
            path = f"{DEBUGFS}/{entry}"
            if os.path.isfile(path):
                try:
                    sz = os.path.getsize(path)
                    readable = os.access(path, os.R_OK)
                    out.append(f"  {entry}: {sz} bytes, readable={readable}")
                except:
                    out.append(f"  {entry}: (stat failed)")
        # Special interest files
        for name in ['ta_if']:
            dirpath = f"{DEBUGFS}/{name}"
            if os.path.isdir(dirpath):
                out.append(f"\n  Directory: {name}/")
                for sub in sorted(os.listdir(dirpath)):
                    out.append(f"    {sub}")
    except PermissionError:
        out.append("  PERMISSION DENIED (need root for debugfs listing)")

    check_thermal()

    # 2b) Read specific debug files
    readable_files = [
        'amdgpu_fence_info', 'amdgpu_firmware_info', 'amdgpu_pm_info',
        'amdgpu_gfxoff', 'amdgpu_gfxoff_count', 'amdgpu_gfxoff_status',
        'amdgpu_vm_info', 'amdgpu_sa_info', 'amdgpu_gem_info',
        'amdgpu_vram_mm', 'amdgpu_gtt_mm', 'amdgpu_gds_mm',
        'amdgpu_discovery', 'amdgpu_iomem',
    ]

    for fname in readable_files:
        fpath = f"{DEBUGFS}/{fname}"
        try:
            if fname == 'amdgpu_discovery':
                # Binary, read and summarize
                with open(fpath, 'rb') as f:
                    data = f.read(256)
                out.append(f"\n--- {fname} (first 256 bytes hex) ---")
                out.append(data.hex())
            elif fname == 'amdgpu_iomem':
                with open(fpath, 'rb') as f:
                    data = f.read(1024)
                out.append(f"\n--- {fname} ({len(data)} bytes) ---")
                out.append(data[:128].hex())
            else:
                with open(fpath, 'r') as f:
                    content = f.read(4096)
                if content.strip():
                    out.append(f"\n--- {fname} ---")
                    out.append(content.strip()[:2000])
        except PermissionError:
            out.append(f"\n--- {fname} --- PERMISSION DENIED")
        except Exception as e:
            out.append(f"\n--- {fname} --- ERROR: {e}")

    check_thermal()

    # 2c) gpu_metrics deep parse
    out.append(f"\n\n{'='*40}")
    out.append("GPU METRICS DEEP PARSE (v3.0)")
    out.append(f"{'='*40}")
    try:
        with open(GPU_METRICS, 'rb') as f:
            raw = f.read()
        out.append(f"Raw size: {len(raw)} bytes")
        out.append(f"Header hex: {raw[:16].hex()}")

        # Parse gpu_metrics_v3_0 header
        # struct_version (1B), content_revision (1B), format_revision (1B) ... actually:
        # uint16_t structure_size, uint8_t format_revision, uint8_t content_revision
        if len(raw) >= 4:
            # The header is: format_rev(1), content_rev(1), structure_size(2) in AMD's gpu_metrics
            # Actually: common_header: structure_size(2), format_revision(1), content_revision(1)
            fmt_rev = raw[2]
            cnt_rev = raw[3]
            struct_size = struct.unpack_from('<H', raw, 0)[0]
            out.append(f"Format revision: {fmt_rev}")
            out.append(f"Content revision: {cnt_rev}")
            out.append(f"Structure size from header: {struct_size}")

        # Parse known v3.0 fields based on amdgpu kernel headers
        # Offsets from gpu_metrics_v3_0:
        # +0x04: temperature_gfx (uint16)
        # +0x06: temperature_soc (uint16)
        # +0x08-0x1E: core temps array (uint16 x 16 max)
        # For v3.0 the layout varies. Let's dump all uint16 pairs to find populated fields
        out.append(f"\n--- All uint16 values (offset: value) ---")
        for i in range(0, min(len(raw), 264), 2):
            val = struct.unpack_from('<H', raw, i)[0]
            if val != 0 and val != 0xFFFF:
                out.append(f"  +0x{i:04X}: {val} (0x{val:04X})")

        # Also dump as uint32
        out.append(f"\n--- Non-zero uint32 values ---")
        for i in range(0, min(len(raw), 264), 4):
            val = struct.unpack_from('<I', raw, i)[0]
            if val != 0 and val != 0xFFFFFFFF:
                out.append(f"  +0x{i:04X}: {val} (0x{val:08X})")

        # Check for 64-bit timestamp field (typically at offset around 0x60-0x70)
        out.append(f"\n--- Potential 64-bit timestamps ---")
        for i in range(0, min(len(raw), 264), 8):
            val = struct.unpack_from('<Q', raw, i)[0]
            if val > 1000000 and val < 0x7FFFFFFFFFFFFFFF:
                out.append(f"  +0x{i:04X}: {val} (0x{val:016X})")

    except Exception as e:
        out.append(f"gpu_metrics error: {e}")

    check_thermal()

    # 2d) hwmon deep read
    out.append(f"\n\n{'='*40}")
    out.append("HWMON DEEP READ")
    out.append(f"{'='*40}")
    hwmon_path = Path(HWMON)
    for entry in sorted(hwmon_path.iterdir()):
        if entry.is_file() and not entry.name.startswith('.'):
            try:
                val = entry.read_text().strip()
                out.append(f"  {entry.name}: {val}")
            except:
                out.append(f"  {entry.name}: (unreadable)")

    result = '\n'.join(out)
    save_result("z2340_debug_interfaces.txt", result)
    print(result)

# ====================== STEP 3: RING BUFFERS ======================
def step3_ring_buffers():
    print("\n" + "="*60)
    print("STEP 3: Ring Buffer Snooping")
    print("="*60)
    out = []

    ring_files = [
        'amdgpu_ring_gfx_0.0.0',
        'amdgpu_ring_comp_1.0.0',
        'amdgpu_ring_comp_1.1.0',
        'amdgpu_ring_sdma0',
        'amdgpu_ring_mes_3.0.0',
        'amdgpu_ring_vcn_unified_0',
        'amdgpu_ring_vpe',
    ]

    for ring in ring_files:
        fpath = f"{DEBUGFS}/{ring}"
        try:
            with open(fpath, 'rb') as f:
                data = f.read()
            out.append(f"\n--- {ring} ({len(data)} bytes) ---")
            # First 12 bytes are header (rptr, wptr, count)
            if len(data) >= 12:
                header = data[:12]
                out.append(f"  Header (hex): {header.hex()}")
                # The ring dump format: first 4 bytes = wptr, next 4 = rptr, next 4 = ring count
                # Actually kernel prints text, let's check if text or binary
                try:
                    text = data.decode('utf-8', errors='replace')
                    lines = text.strip().split('\n')
                    out.append(f"  Lines: {len(lines)}")
                    # Show first and last few lines
                    for line in lines[:10]:
                        out.append(f"    {line.rstrip()}")
                    if len(lines) > 10:
                        out.append(f"    ... ({len(lines)-10} more lines)")
                        for line in lines[-3:]:
                            out.append(f"    {line.rstrip()}")
                except:
                    # Binary
                    non_zero = sum(1 for b in data if b != 0)
                    out.append(f"  Non-zero bytes: {non_zero}/{len(data)} ({non_zero/len(data)*100:.1f}%)")
                    out.append(f"  First 64 bytes: {data[:64].hex()}")

        except PermissionError:
            out.append(f"\n--- {ring} --- PERMISSION DENIED")
        except Exception as e:
            out.append(f"\n--- {ring} --- ERROR: {e}")

    check_thermal()

    # Read MQD files
    out.append(f"\n\n{'='*40}")
    out.append("MQD (Memory Queue Descriptors)")
    out.append(f"{'='*40}")

    mqd_files = [
        'amdgpu_mqd_gfx_0.0.0',
        'amdgpu_mqd_comp_1.0.0',
    ]
    for mqd in mqd_files:
        fpath = f"{DEBUGFS}/{mqd}"
        try:
            with open(fpath, 'rb') as f:
                data = f.read()
            out.append(f"\n--- {mqd} ({len(data)} bytes) ---")
            # MQD is a binary structure. Dump non-zero dwords
            non_zero_dwords = []
            for i in range(0, len(data), 4):
                val = struct.unpack_from('<I', data, i)[0]
                if val != 0:
                    non_zero_dwords.append((i, val))
            out.append(f"  Non-zero dwords: {len(non_zero_dwords)}/{len(data)//4}")
            for off, val in non_zero_dwords[:30]:
                out.append(f"    +0x{off:04X}: 0x{val:08X} ({val})")
            if len(non_zero_dwords) > 30:
                out.append(f"    ... ({len(non_zero_dwords)-30} more)")
        except PermissionError:
            out.append(f"\n--- {mqd} --- PERMISSION DENIED")
        except Exception as e:
            out.append(f"\n--- {mqd} --- ERROR: {e}")

    result = '\n'.join(out)
    save_result("z2340_ring_buffers.txt", result)
    print(result)

# ====================== STEP 4: WAVE STATE ======================
def step4_wave_state():
    print("\n" + "="*60)
    print("STEP 4: Wave State Deep Dump")
    print("="*60)
    out = []

    # 4a) Read wave state (idle GPU)
    try:
        with open(f"{DEBUGFS}/amdgpu_wave", 'r') as f:
            wave_data = f.read(16384)
        if wave_data.strip():
            out.append("--- Wave State (idle GPU) ---")
            out.append(wave_data[:4000])
        else:
            out.append("Wave state: EMPTY (no active waves - GPU idle)")
    except PermissionError:
        out.append("Wave state: PERMISSION DENIED")
    except Exception as e:
        out.append(f"Wave state: ERROR: {e}")

    check_thermal()

    # 4b) Read GPR state
    try:
        with open(f"{DEBUGFS}/amdgpu_gpr", 'r') as f:
            gpr_data = f.read(16384)
        if gpr_data.strip():
            out.append("\n--- GPR State ---")
            out.append(gpr_data[:4000])
        else:
            out.append("\nGPR state: EMPTY (no active waves)")
    except PermissionError:
        out.append("\nGPR state: PERMISSION DENIED")
    except Exception as e:
        out.append(f"\nGPR state: ERROR: {e}")

    # 4c) Read gprwave
    try:
        with open(f"{DEBUGFS}/amdgpu_gprwave", 'r') as f:
            gprwave_data = f.read(16384)
        if gprwave_data.strip():
            out.append("\n--- GPR Wave State ---")
            out.append(gprwave_data[:4000])
        else:
            out.append("\nGPR wave state: EMPTY (no active waves)")
    except PermissionError:
        out.append("\nGPR wave state: PERMISSION DENIED")
    except Exception as e:
        out.append(f"\nGPR wave state: ERROR: {e}")

    # 4d) Read sensors
    try:
        with open(f"{DEBUGFS}/amdgpu_sensors", 'r') as f:
            sensor_data = f.read(4096)
        if sensor_data.strip():
            out.append("\n--- Sensors ---")
            out.append(sensor_data[:2000])
        else:
            out.append("\nSensors: EMPTY")
    except PermissionError:
        out.append("\nSensors: PERMISSION DENIED")
    except Exception as e:
        out.append(f"\nSensors: ERROR: {e}")

    # Read GCA config
    try:
        with open(f"{DEBUGFS}/amdgpu_gca_config", 'r') as f:
            gca = f.read(4096)
        if gca.strip():
            out.append("\n--- GCA Config ---")
            out.append(gca[:2000])
    except:
        pass

    result = '\n'.join(out)
    save_result("z2340_wave_state.txt", result)
    print(result)

# ====================== STEP 5: FIRMWARE TIMESTAMP ======================
def step5_fw_timestamp():
    print("\n" + "="*60)
    print("STEP 5: Firmware Timestamp as Clock Source")
    print("="*60)
    out = []

    # Read gpu_metrics rapidly to extract firmware_timestamp
    timestamps = []
    wall_times = []
    N = 1000

    out.append(f"Reading gpu_metrics {N} times to extract firmware timestamp...")

    # First, identify which offset contains the timestamp
    # Read once and find 64-bit fields
    with open(GPU_METRICS, 'rb') as f:
        raw = f.read()

    # The firmware_timestamp in v3.0 is typically a 64-bit counter
    # Let's find it by looking for large monotonically increasing values
    candidate_offsets = []
    for i in range(0, min(len(raw), 264) - 7, 8):
        val = struct.unpack_from('<Q', raw, i)[0]
        if 1_000_000 < val < 0x7FFFFFFFFFFFFFFF:
            candidate_offsets.append((i, val))

    out.append(f"Candidate 64-bit timestamp offsets: {[(f'0x{o:X}', v) for o, v in candidate_offsets]}")

    if not candidate_offsets:
        # Try uint32
        for i in range(0, min(len(raw), 264) - 3, 4):
            val = struct.unpack_from('<I', raw, i)[0]
            if 100_000 < val < 0x7FFFFFFF:
                candidate_offsets.append((i, val))
        out.append(f"Candidate 32-bit timestamp offsets: {[(f'0x{o:X}', v) for o, v in candidate_offsets]}")

    check_thermal()

    # Now read rapidly
    if candidate_offsets:
        ts_offset = candidate_offsets[0][0]
        ts_is_64bit = True  # assume from first pass
        out.append(f"\nUsing timestamp at offset 0x{ts_offset:X}")

        for i in range(N):
            t0 = time.monotonic_ns()
            with open(GPU_METRICS, 'rb') as f:
                raw = f.read()
            t1 = time.monotonic_ns()

            if len(raw) > ts_offset + 7:
                ts = struct.unpack_from('<Q', raw, ts_offset)[0]
            elif len(raw) > ts_offset + 3:
                ts = struct.unpack_from('<I', raw, ts_offset)[0]
                ts_is_64bit = False
            else:
                break

            timestamps.append(ts)
            wall_times.append((t0 + t1) / 2)  # midpoint

        timestamps = np.array(timestamps, dtype=np.int64)
        wall_times = np.array(wall_times, dtype=np.float64)

        # Analyze
        ts_deltas = np.diff(timestamps)
        wall_deltas = np.diff(wall_times)  # nanoseconds

        # Filter out zero-delta readings (same firmware tick)
        nonzero = ts_deltas != 0

        out.append(f"\nTotal readings: {len(timestamps)}")
        out.append(f"Non-zero deltas: {nonzero.sum()}/{len(ts_deltas)}")
        out.append(f"Timestamp range: {timestamps[0]} to {timestamps[-1]}")
        out.append(f"Timestamp total delta: {timestamps[-1] - timestamps[0]}")

        if nonzero.sum() > 0:
            nz_ts = ts_deltas[nonzero]
            nz_wall = wall_deltas[nonzero] / 1e6  # to ms

            out.append(f"\nFirmware timestamp resolution:")
            out.append(f"  Min delta: {nz_ts.min()}")
            out.append(f"  Max delta: {nz_ts.max()}")
            out.append(f"  Mean delta: {nz_ts.mean():.1f}")
            out.append(f"  Std delta: {nz_ts.std():.1f}")

            out.append(f"\nWall clock between non-zero updates:")
            out.append(f"  Min: {nz_wall.min():.3f} ms")
            out.append(f"  Max: {nz_wall.max():.3f} ms")
            out.append(f"  Mean: {nz_wall.mean():.3f} ms")

            # Compute ratio: fw_ticks per wall_ns
            if timestamps[-1] != timestamps[0]:
                total_fw = float(timestamps[-1] - timestamps[0])
                total_wall = (wall_times[-1] - wall_times[0]) / 1e9  # seconds
                fw_freq = total_fw / total_wall
                out.append(f"\nFirmware clock frequency: {fw_freq:.1f} ticks/sec ({fw_freq/1e6:.3f} MHz)")

            # Check for jitter between fw clock and wall clock
            # Normalize both to [0,1] range and compute residuals
            if nonzero.sum() > 10:
                cum_ts = np.cumsum(ts_deltas[nonzero]).astype(float)
                cum_wall = np.cumsum(wall_deltas[nonzero])
                if cum_ts[-1] > 0 and cum_wall[-1] > 0:
                    norm_ts = cum_ts / cum_ts[-1]
                    norm_wall = cum_wall / cum_wall[-1]
                    residuals = norm_ts - norm_wall
                    out.append(f"\nClock jitter (fw vs wall):")
                    out.append(f"  Residual std: {residuals.std():.6f}")
                    out.append(f"  Residual max: {np.abs(residuals).max():.6f}")
                    out.append(f"  Residual mean: {residuals.mean():.6f}")
        else:
            out.append("All firmware timestamps identical (GPU in deep sleep?)")
    else:
        out.append("No candidate timestamp fields found in gpu_metrics!")

    result = '\n'.join(out)
    save_result("z2340_fw_timestamp.txt", result)
    print(result)

# ====================== STEP 6: VOLTAGE/POWER SIGNAL ======================
def step6_voltage_signal():
    print("\n" + "="*60)
    print("STEP 6: Voltage/Power as Direct Analog Signal")
    print("="*60)
    out = []

    # 6a) Rapid sampling of power/voltage from hwmon and gpu_metrics
    N = 5000  # 5000 samples

    hwmon_power = f"{HWMON}/power1_input"
    hwmon_temp = f"{HWMON}/temp1_input"
    hwmon_freq = f"{HWMON}/freq1_input"
    hwmon_v0 = f"{HWMON}/in0_input"
    hwmon_v1 = f"{HWMON}/in1_input"

    # First: hwmon-based sampling
    power_samples = []
    temp_samples = []
    freq_samples = []
    v0_samples = []
    v1_samples = []
    times = []

    out.append(f"Sampling hwmon sensors {N} times...")
    t_start = time.monotonic()

    for i in range(N):
        t = time.monotonic()
        try:
            with open(hwmon_power) as f: p = int(f.read().strip())
            with open(hwmon_temp) as f: temp = int(f.read().strip())
            with open(hwmon_freq) as f: freq = int(f.read().strip())
            with open(hwmon_v0) as f: v0 = int(f.read().strip())
            with open(hwmon_v1) as f: v1 = int(f.read().strip())
        except:
            continue
        power_samples.append(p)
        temp_samples.append(temp)
        freq_samples.append(freq)
        v0_samples.append(v0)
        v1_samples.append(v1)
        times.append(t)

        # Thermal safety check every 500 samples
        if i % 500 == 0:
            check_thermal()

    t_elapsed = time.monotonic() - t_start
    rate = len(power_samples) / t_elapsed if t_elapsed > 0 else 0

    out.append(f"Collected {len(power_samples)} samples in {t_elapsed:.2f}s ({rate:.0f} samples/sec)")

    power = np.array(power_samples, dtype=np.float64) / 1e6  # uW to W
    temp_arr = np.array(temp_samples, dtype=np.float64) / 1e3  # mC to C
    freq_arr = np.array(freq_samples, dtype=np.float64) / 1e6  # Hz to MHz
    v0_arr = np.array(v0_samples, dtype=np.float64)  # mV
    v1_arr = np.array(v1_samples, dtype=np.float64)  # mV
    t_arr = np.array(times) - times[0]

    for name, arr, unit in [
        ("Power", power, "W"),
        ("Temperature", temp_arr, "C"),
        ("Frequency", freq_arr, "MHz"),
        ("Voltage vddgfx", v0_arr, "mV"),
        ("Voltage vddnb", v1_arr, "mV"),
    ]:
        out.append(f"\n{name} ({unit}):")
        out.append(f"  Mean: {arr.mean():.4f}")
        out.append(f"  Std: {arr.std():.4f}")
        out.append(f"  Min: {arr.min():.4f}")
        out.append(f"  Max: {arr.max():.4f}")
        out.append(f"  Range: {arr.max() - arr.min():.4f}")
        unique = len(np.unique(arr))
        out.append(f"  Unique values: {unique}")
        if unique <= 20:
            vals, counts = np.unique(arr, return_counts=True)
            for v, c in zip(vals, counts):
                out.append(f"    {v:.4f}: {c} ({c/len(arr)*100:.1f}%)")

    # Autocorrelation of power signal
    if len(power) > 100:
        out.append(f"\nPower autocorrelation:")
        power_centered = power - power.mean()
        var = np.var(power_centered)
        if var > 0:
            for lag in [1, 2, 5, 10, 20, 50, 100]:
                if lag < len(power_centered):
                    acf = np.mean(power_centered[:-lag] * power_centered[lag:]) / var
                    out.append(f"  ACF(lag={lag}): {acf:.4f}")

    # PSD of power signal
    if len(power) > 256 and power.std() > 0:
        from numpy.fft import rfft, rfftfreq
        dt = np.mean(np.diff(t_arr)) if len(t_arr) > 1 else 1.0/rate
        freqs = rfftfreq(len(power), d=dt)
        psd = np.abs(rfft(power - power.mean()))**2 / len(power)

        out.append(f"\nPower PSD:")
        out.append(f"  Sampling dt: {dt*1000:.3f} ms")
        out.append(f"  Nyquist freq: {1/(2*dt):.1f} Hz")
        # Find dominant frequencies
        idx = np.argsort(psd[1:])[::-1][:10] + 1
        for i in idx:
            if freqs[i] > 0:
                out.append(f"  Peak at {freqs[i]:.2f} Hz: PSD={psd[i]:.2e}")

    check_thermal()

    # 6b) gpu_metrics based sampling for voltages that might be populated
    out.append(f"\n\n{'='*40}")
    out.append("GPU_METRICS Voltage Fields Sampling")
    out.append(f"{'='*40}")

    gm_samples = []
    N2 = 2000
    for i in range(N2):
        try:
            with open(GPU_METRICS, 'rb') as f:
                raw = f.read()
            gm_samples.append((time.monotonic(), raw))
        except:
            pass
        if i % 500 == 0:
            check_thermal()

    if gm_samples:
        out.append(f"Collected {len(gm_samples)} gpu_metrics samples")

        # Find fields that vary across samples
        sample_len = len(gm_samples[0][1])
        varying_offsets = []

        first = gm_samples[0][1]
        for off in range(0, sample_len - 1, 2):
            vals = set()
            for _, raw in gm_samples[:100]:  # Check first 100
                if off + 1 < len(raw):
                    vals.add(struct.unpack_from('<H', raw, off)[0])
            if len(vals) > 1:
                varying_offsets.append((off, vals))

        out.append(f"\nVarying uint16 fields across {min(100, len(gm_samples))} samples:")
        for off, vals in varying_offsets:
            sorted_vals = sorted(vals)
            out.append(f"  +0x{off:04X}: {len(vals)} unique, range [{sorted_vals[0]}, {sorted_vals[-1]}]")
            if len(vals) <= 10:
                out.append(f"    Values: {sorted_vals}")

        # Now extract all varying fields as time series
        if varying_offsets:
            out.append(f"\nTime series of varying fields ({len(gm_samples)} samples):")
            for off, _ in varying_offsets[:8]:  # Top 8 varying fields
                ts = []
                for _, raw in gm_samples:
                    if off + 1 < len(raw):
                        ts.append(struct.unpack_from('<H', raw, off)[0])
                ts = np.array(ts, dtype=np.float64)
                out.append(f"\n  +0x{off:04X}: mean={ts.mean():.2f} std={ts.std():.4f} unique={len(np.unique(ts))}")

    result = '\n'.join(out)
    save_result("z2340_voltage_signal.txt", result)
    print(result)

# ====================== MAIN ======================
def main():
    print("z2340: VRAM and Debug Interface Probe")
    print(f"Thermal: {check_thermal():.1f}C")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    summary = OrderedDict()
    summary["experiment"] = "z2340"
    summary["timestamp"] = time.strftime('%Y-%m-%dT%H:%M:%S')
    summary["gpu"] = "AMD Radeon 8060S (gfx1151)"

    # Step 1
    vram_ok = step1_vram_access()
    summary["step1_vram_access"] = {
        "vram_readable": vram_ok,
        "vram_file": VRAM_FILE,
        "vram_virtual_size_bytes": os.path.getsize(VRAM_FILE) if os.path.exists(VRAM_FILE) else 0,
        "pci_bar0": "0x6800000000 (256MB)",
        "pci_bar2": "0xb4000000 (2MB)",
    }
    print(f"\nThermal after step 1: {check_thermal():.1f}C")

    # Step 2
    step2_debug_interfaces()
    summary["step2_debug_interfaces"] = {
        "debugfs_available": True,
        "hwmon_path": HWMON,
        "gpu_metrics_size": os.path.getsize(GPU_METRICS) if os.path.exists(GPU_METRICS) else 0,
        "has_wave_scanner": os.path.exists(f"{DEBUGFS}/amdgpu_wave"),
        "has_gpr_read": os.path.exists(f"{DEBUGFS}/amdgpu_gpr"),
        "has_regs": os.path.exists(f"{DEBUGFS}/amdgpu_regs"),
        "has_regs2": os.path.exists(f"{DEBUGFS}/amdgpu_regs2"),
        "has_discovery": os.path.exists(f"{DEBUGFS}/amdgpu_discovery"),
        "has_vbios": os.path.exists(f"{DEBUGFS}/amdgpu_vbios"),
        "ring_buffers": ["gfx_0.0.0", "comp_1.0.0-1.3.1", "sdma0", "vcn_unified_0/1", "mes_3.0.0", "vpe"],
    }
    print(f"\nThermal after step 2: {check_thermal():.1f}C")

    # Step 3
    step3_ring_buffers()
    print(f"\nThermal after step 3: {check_thermal():.1f}C")

    # Step 4
    step4_wave_state()
    summary["step4_wave_state"] = {
        "note": "Wave state empty when GPU idle (GFXOFF). Need running kernel to observe.",
    }
    print(f"\nThermal after step 4: {check_thermal():.1f}C")

    # Step 5
    step5_fw_timestamp()
    print(f"\nThermal after step 5: {check_thermal():.1f}C")

    # Step 6
    step6_voltage_signal()
    print(f"\nThermal after step 6: {check_thermal():.1f}C")

    # Save summary
    summary["thermal_final"] = check_thermal()
    with open(RESULTS_DIR / "z2340_probe_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary: {RESULTS_DIR / 'z2340_probe_summary.json'}")

    print("\n" + "="*60)
    print("PROBE COMPLETE")
    print("="*60)

if __name__ == '__main__':
    main()
