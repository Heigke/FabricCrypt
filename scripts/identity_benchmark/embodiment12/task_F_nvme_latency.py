"""Task F: NVMe completion latency tail.

O_DIRECT pread 4KB at random offsets, 100k samples. Per-completion latency
measured via clock_gettime. Read from raw block device (read-only, safe).
"""
import os
import sys
import time
import ctypes
import ctypes.util
import numpy as np
import random

sys.path.insert(0, os.path.dirname(__file__))
from common import thermal_guard, save_json, hostname

CLOCK_MONOTONIC_RAW = 4
O_DIRECT = 0o40000
O_RDONLY = 0

class Timespec(ctypes.Structure):
    _fields_ = [('tv_sec', ctypes.c_long), ('tv_nsec', ctypes.c_long)]

libc = ctypes.CDLL(ctypes.util.find_library('c'), use_errno=True)
libc.clock_gettime.argtypes = [ctypes.c_int, ctypes.POINTER(Timespec)]
libc.pread.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_long]
libc.pread.restype = ctypes.c_long
libc.posix_memalign.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_size_t]
libc.posix_memalign.restype = ctypes.c_int

def now_ns():
    ts = Timespec()
    libc.clock_gettime(CLOCK_MONOTONIC_RAW, ctypes.byref(ts))
    return ts.tv_sec * 1_000_000_000 + ts.tv_nsec


def find_readable_device():
    # try the user's home partition (read-only access via O_RDONLY to a regular file works,
    # but we want raw block device for true NVMe behavior). Need root for /dev/nvme0n1.
    # Fallback: use a large file via O_DIRECT.
    candidates = ['/dev/nvme0n1', '/dev/nvme0n1p2', '/dev/nvme0n1p1']
    for c in candidates:
        if os.path.exists(c) and os.access(c, os.R_OK):
            return c, True
    # fallback: create a 200MB test file
    home = os.path.expanduser('~')
    fpath = os.path.join(home, '.cache', 'nvme_lat_test.bin')
    if not os.path.exists(fpath) or os.path.getsize(fpath) < 200 * 1024 * 1024:
        os.makedirs(os.path.dirname(fpath), exist_ok=True)
        with open(fpath, 'wb') as f:
            f.write(os.urandom(200 * 1024 * 1024))
        # drop caches if possible
        try:
            os.sync()
        except Exception:
            pass
    return fpath, False


def bench(device_path, n, block_size=4096):
    # open O_DIRECT for raw NVMe behavior
    try:
        fd = os.open(device_path, O_RDONLY | O_DIRECT)
    except OSError as e:
        print(f"O_DIRECT failed ({e}), falling back to plain O_RDONLY")
        fd = os.open(device_path, O_RDONLY)

    # determine size
    size = os.lseek(fd, 0, os.SEEK_END)
    os.lseek(fd, 0, os.SEEK_SET)
    print(f"[F] device={device_path} size={size/1e9:.2f}GB block={block_size}")

    # aligned buffer
    buf_p = ctypes.c_void_p()
    libc.posix_memalign(ctypes.byref(buf_p), 4096, block_size)

    max_offset = (size // block_size) - 1
    if max_offset <= 0:
        os.close(fd)
        raise RuntimeError("device too small")

    lat = np.empty(n, dtype=np.int64)
    last_guard = time.time()
    rng = random.Random(42)
    for i in range(n):
        off = rng.randint(0, max_offset) * block_size
        t0 = now_ns()
        r = libc.pread(fd, buf_p, block_size, off)
        t1 = now_ns()
        if r < 0:
            err = ctypes.get_errno()
            print(f"pread error at i={i}: errno={err}")
            lat = lat[:i]
            break
        lat[i] = t1 - t0
        if (i & 0xFFF) == 0 and time.time() - last_guard > 5:
            thermal_guard()
            last_guard = time.time()

    os.close(fd)
    return lat


def summarize(lat):
    return {
        'n': int(lat.size),
        'mean_ns': float(lat.mean()),
        'std_ns': float(lat.std()),
        'p50': float(np.percentile(lat, 50)),
        'p90': float(np.percentile(lat, 90)),
        'p99': float(np.percentile(lat, 99)),
        'p99_9': float(np.percentile(lat, 99.9)),
        'p99_99': float(np.percentile(lat, 99.99)),
        'min': int(lat.min()),
        'max': int(lat.max()),
    }


def main():
    N = 100_000
    host = hostname()
    print(f"[F] host={host} NVMe latency N={N}")
    thermal_guard()
    dev, is_raw = find_readable_device()
    print(f"[F] using {dev} raw={is_raw}")
    t0 = time.time()
    lat = bench(dev, N)
    print(f"[F] {lat.size} samples in {time.time()-t0:.1f}s")

    out = {
        'host': host,
        'device': dev,
        'is_raw_block': is_raw,
        'N': int(lat.size),
        'nvme_latency': summarize(lat),
        'raw_samples_ns': lat[::10].tolist(),
    }
    out_path = f"results/IDENTITY_BENCHMARK_2026-05-30/embodiment12/task_F_nvme_{host}.json"
    save_json(out_path, out)


if __name__ == '__main__':
    main()
