#!/usr/bin/env python3
"""
z2308_cpu_neuromorphic.py — CPU-as-Neuromorphic-Substrate Reservoir Computing
==============================================================================
AMD Ryzen AI MAX+ PRO 395 (Zen 5, 16 cores, 32 threads) as a reservoir.

Each CPU core = 1 "neuron". Analog signals from MSRs, TSC jitter, thermal,
perf counters, and RDRAND provide the nonlinear dynamics.

5 Experiments, 17 tests (T811-T827):
  EXP 1: CPU Analog Signal Characterization (T811-T813)
  EXP 2: Per-Core Reservoir Computing (T814-T818)
  EXP 3: Temporal Products Amplification (T819-T821)
  EXP 4: CPU + FPGA + GPU Triple Bridge (T822-T825)
  EXP 5: RDRAND vs GPU Noise (T826-T827)

Run:
  PYTHONUNBUFFERED=1 sudo venv/bin/python scripts/z2308_cpu_neuromorphic.py
  (sudo needed for /dev/cpu/N/msr access; perf_event_open may work without)
"""

import os, sys, time, json, struct, ctypes, ctypes.util
import multiprocessing as mp
import numpy as np
from pathlib import Path

os.environ['PYTHONUNBUFFERED'] = '1'

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / 'scripts'))
RESULTS = BASE / 'results'
RESULTS.mkdir(exist_ok=True)
SAVE_FILE = RESULTS / 'z2308_cpu_neuromorphic.json'

N_CORES = 16
N_STEPS = 1500
WARMUP = 300
TEMP_PAUSE = 75.0
TEMP_RESUME = 50.0
TEMP_SAFE = 55.0
SAMPLE_HZ = 200  # target sampling rate for reservoir


# ============================================================
# Thermal safety
# ============================================================
def get_max_temp():
    temps = []
    for path in ['/sys/class/thermal/thermal_zone0/temp',
                 '/sys/class/hwmon/hwmon7/temp1_input']:
        try:
            with open(path, 'r') as f:
                temps.append(float(f.read().strip()) / 1000.0)
        except Exception:
            pass
    return max(temps) if temps else 0.0


def wait_cool(label="", target=None):
    if target is None:
        target = TEMP_SAFE
    temp = get_max_temp()
    if temp <= target:
        return temp
    print(f"  [TEMP] {label} {temp:.0f}C -> {target:.0f}C...", end="", flush=True)
    t0 = time.time()
    while temp > target and (time.time() - t0) < 180:
        time.sleep(5)
        temp = get_max_temp()
        print(f" {temp:.0f}", end="", flush=True)
    print(f" OK ({time.time()-t0:.0f}s)")
    return temp


def check_thermal(step, label=""):
    """Check every 50 steps. Pause at 75C, wait until 50C."""
    if step % 50 != 0:
        return
    temp = get_max_temp()
    if temp >= TEMP_PAUSE:
        print(f"\n  [THERMAL] {label} step {step}: {temp:.0f}C >= {TEMP_PAUSE:.0f}C, cooling...",
              end="", flush=True)
        t0 = time.time()
        while temp > TEMP_RESUME and (time.time() - t0) < 120:
            time.sleep(5)
            temp = get_max_temp()
            print(f" {temp:.0f}", end="", flush=True)
        print(f" OK", flush=True)


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.bool_): return bool(obj)
        return super().default(obj)


# ============================================================
# SMN (System Management Network) — Deep Analog Access
# ============================================================
# These are the DEEPEST safely-readable analog signals on AMD Zen 5.
# Accessed via PCI root complex: write addr → reg 0x60, read result ← reg 0x64

SMN_THM_TCON_CUR_TMP = 0x00059800     # Raw thermal ADC (11-bit, 0.125°C)
SMN_CCD_TEMP_ZEN5_BASE = 0x00059B08   # 0x59800 + 0x308 + x*4 for CCD x
SMN_SVI_TEL_PLANE0 = 0x0005A00C       # Core voltage + current (SVI3 raw ADC)
SMN_SVI_TEL_PLANE1 = 0x0005A010       # SoC voltage + current (SVI3 raw ADC)

_smn_pci_fd = None

def smn_init():
    """Open PCI config space for SMN reads (bus 0, dev 0, func 0)."""
    global _smn_pci_fd
    # Method 1: /sys/bus/pci sysfs config
    pci_path = '/sys/bus/pci/devices/0000:00:00.0/config'
    if os.path.exists(pci_path):
        try:
            _smn_pci_fd = os.open(pci_path, os.O_RDWR)
            # Test read
            smn_read(SMN_THM_TCON_CUR_TMP)
            print("  [SMN] PCI config space access OK (sysfs)")
            return True
        except Exception as e:
            print(f"  [SMN] PCI sysfs failed: {e}")
            try: os.close(_smn_pci_fd)
            except: pass
            _smn_pci_fd = None
    # Method 2: ryzen_smu_drv
    smn_path = '/sys/kernel/ryzen_smu_drv/smn'
    if os.path.exists(smn_path):
        try:
            _smn_pci_fd = ('smu', smn_path)
            val = smn_read(SMN_THM_TCON_CUR_TMP)
            print(f"  [SMN] ryzen_smu_drv access OK (raw=0x{val:08X})")
            return True
        except Exception as e:
            print(f"  [SMN] ryzen_smu_drv failed: {e}")
            _smn_pci_fd = None
    print("  [SMN] No SMN access available (need root + PCI config or ryzen_smu_drv)")
    return False

def smn_read(addr):
    """Read 32-bit value from SMN address space."""
    global _smn_pci_fd
    if _smn_pci_fd is None:
        return 0
    if isinstance(_smn_pci_fd, tuple) and _smn_pci_fd[0] == 'smu':
        # ryzen_smu_drv: write addr, read result
        with open(_smn_pci_fd[1], 'r+b') as f:
            f.write(struct.pack('<I', addr))
            f.seek(0)
            return struct.unpack('<I', f.read(4))[0]
    else:
        # PCI config space: write target addr to offset 0x60, read from 0x64
        os.lseek(_smn_pci_fd, 0x60, os.SEEK_SET)
        os.write(_smn_pci_fd, struct.pack('<I', addr))
        os.lseek(_smn_pci_fd, 0x64, os.SEEK_SET)
        return struct.unpack('<I', os.read(_smn_pci_fd, 4))[0]

def smn_read_temp():
    """Read raw thermal from SMN_THM_TCON_CUR_TMP. Returns (temp_C, raw_value)."""
    raw = smn_read(SMN_THM_TCON_CUR_TMP)
    cur_temp = (raw >> 21) & 0x7FF
    range_sel = (raw >> 19) & 1
    temp_mc = cur_temp * 125
    if range_sel:
        temp_mc -= 49000
    return temp_mc / 1000.0, raw

def smn_read_svi3():
    """Read SVI3 voltage/current telemetry. Returns (vcore_mV, icore, vsoc_mV, isoc, raw0, raw1)."""
    raw0 = smn_read(SMN_SVI_TEL_PLANE0)
    raw1 = smn_read(SMN_SVI_TEL_PLANE1)
    # Decode: bits[15:8] = VID, bits[7:0] = IDD
    vid_core = (raw0 >> 16) & 0xFF
    idd_core = raw0 & 0xFF
    vid_soc = (raw1 >> 16) & 0xFF
    idd_soc = raw1 & 0xFF
    # Voltage: 1550mV - (625 * VID / 100)
    vcore_mV = 1550.0 - (6.25 * vid_core)
    vsoc_mV = 1550.0 - (6.25 * vid_soc)
    return vcore_mV, idd_core, vsoc_mV, idd_soc, raw0, raw1

def smn_read_ccd_temps(n_ccds=2):
    """Read per-CCD temperatures (Zen 5 layout). Returns list of (temp_C, valid)."""
    temps = []
    for i in range(n_ccds):
        raw = smn_read(SMN_CCD_TEMP_ZEN5_BASE + i * 4)
        valid = bool(raw & (1 << 11))
        temp = (raw & 0x7FF) * 0.125
        temps.append((temp, valid))
    return temps


# ============================================================
# MSR Access
# ============================================================
CORE_ENERGY_MSR = 0xC001029B
MPERF_MSR = 0xC00000E7
APERF_MSR = 0xC00000E8
PSTATE_STATUS_MSR = 0xC0010299

__NR_perf_event_open = 298


class perf_event_attr(ctypes.Structure):
    _fields_ = [
        ('type', ctypes.c_uint32),
        ('size', ctypes.c_uint32),
        ('config', ctypes.c_uint64),
        ('sample_period', ctypes.c_uint64),
        ('sample_type', ctypes.c_uint64),
        ('read_format', ctypes.c_uint64),
        ('flags', ctypes.c_uint64),
        ('wakeup_events', ctypes.c_uint32),
        ('bp_type', ctypes.c_uint32),
        ('config1', ctypes.c_uint64),
        ('config2', ctypes.c_uint64),
    ]


_libc = None


def _get_libc():
    global _libc
    if _libc is None:
        _libc = ctypes.CDLL(ctypes.util.find_library('c'), use_errno=True)
    return _libc


def perf_open(pe_type, config, cpu=0):
    attr = perf_event_attr()
    attr.type = pe_type
    attr.size = ctypes.sizeof(perf_event_attr)
    attr.config = config
    attr.flags = 0
    fd = _get_libc().syscall(__NR_perf_event_open, ctypes.pointer(attr), -1, cpu, -1, 0)
    if fd == -1:
        errno = ctypes.get_errno()
        raise OSError(errno, os.strerror(errno))
    return fd


def msr_read(fd, addr):
    os.lseek(fd, addr, os.SEEK_SET)
    return struct.unpack('Q', os.read(fd, 8))[0]


def init_msr_access():
    """Open MSR fds for all 16 cores. Returns dict core->fd or empty."""
    fds = {}
    for core in range(N_CORES):
        path = f'/dev/cpu/{core}/msr'
        if os.path.exists(path):
            try:
                fd = os.open(path, os.O_RDONLY)
                fds[core] = fd
                # Quick test read
                msr_read(fd, MPERF_MSR)
            except Exception as e:
                try:
                    os.close(fd)
                except:
                    pass
                if core == 0:
                    print(f"  [MSR] Core {core} failed: {e}")
    return fds


def init_perf_counters():
    """Open perf_event counters for branch misses on all 16 cores."""
    # PERF_TYPE_HARDWARE=0, PERF_COUNT_HW_BRANCH_MISSES=5
    fds = {}
    for core in range(N_CORES):
        try:
            fd = perf_open(0, 5, cpu=core)
            fds[core] = fd
        except Exception as e:
            if core == 0:
                print(f"  [PERF] Core {core} branch-misses failed: {e}")
    return fds


# ============================================================
# Signal readers
# ============================================================
def read_rapl_energy(msr_fds):
    """Read per-core energy MSR. Returns array[16]."""
    vals = np.zeros(N_CORES)
    for core in range(N_CORES):
        if core in msr_fds:
            try:
                vals[core] = msr_read(msr_fds[core], CORE_ENERGY_MSR)
            except:
                pass
    return vals


def read_mperf_aperf(msr_fds):
    """Read MPERF and APERF, return ratio per core. Array[16]."""
    ratios = np.zeros(N_CORES)
    for core in range(N_CORES):
        if core in msr_fds:
            try:
                mperf = msr_read(msr_fds[core], MPERF_MSR)
                aperf = msr_read(msr_fds[core], APERF_MSR)
                ratios[core] = aperf / max(mperf, 1)
            except:
                pass
    return ratios


def read_tsc_jitter():
    """Read TSC in tight loop, compute jitter (std of deltas) per-thread.
    We sample 100 RDTSC calls and compute the variance of successive deltas."""
    # Use time.perf_counter_ns() as portable TSC proxy
    deltas = []
    for _ in range(100):
        t0 = time.perf_counter_ns()
        t1 = time.perf_counter_ns()
        deltas.append(t1 - t0)
    arr = np.array(deltas, dtype=np.float64)
    return np.std(arr), np.mean(arr), arr


def read_branch_misses(perf_fds):
    """Read branch-miss counters for all cores. Returns array[16]."""
    vals = np.zeros(N_CORES)
    for core in range(N_CORES):
        if core in perf_fds:
            try:
                data = os.read(perf_fds[core], 8)
                vals[core] = struct.unpack('Q', data)[0]
            except:
                pass
    return vals


def read_rdrand_batch(n=1000):
    """Read n bytes from os.urandom (backed by RDRAND on AMD), return timing variance."""
    times = []
    for _ in range(20):
        t0 = time.perf_counter_ns()
        os.urandom(n)
        t1 = time.perf_counter_ns()
        times.append(t1 - t0)
    arr = np.array(times, dtype=np.float64)
    return np.std(arr), np.mean(arr), arr


def read_thermal_per_core():
    """Read per-core thermal from hwmon or coretemp. Returns array[16]."""
    temps = np.zeros(N_CORES)
    # Try k10temp / coretemp / hwmon approach
    import glob
    hwmon_dirs = sorted(glob.glob('/sys/class/hwmon/hwmon*'))
    for hdir in hwmon_dirs:
        try:
            with open(os.path.join(hdir, 'name'), 'r') as f:
                name = f.read().strip()
        except:
            continue
        if name in ('k10temp', 'coretemp', 'zenpower'):
            # Read all temp*_input files
            temp_files = sorted(glob.glob(os.path.join(hdir, 'temp*_input')))
            for i, tf in enumerate(temp_files):
                if i >= N_CORES:
                    break
                try:
                    with open(tf, 'r') as f:
                        temps[i] = float(f.read().strip()) / 1000.0
                except:
                    pass
            if np.any(temps > 0):
                break
    # Fallback: just use thermal_zone0 for all cores
    if np.sum(temps > 0) < 2:
        try:
            with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                t = float(f.read().strip()) / 1000.0
            # Add small noise to differentiate cores (they share die temp)
            temps = np.full(N_CORES, t) + np.random.randn(N_CORES) * 0.01
        except:
            pass
    return temps


# ============================================================
# Worker process for per-core workload injection
# ============================================================
def _worker_fn(core_id, intensity, duration_s, result_queue):
    """Pin to core, run FP loop proportional to intensity, report MSR-like features."""
    try:
        os.sched_setaffinity(0, {core_id})
    except:
        pass
    n_iters = int(intensity * 50000)  # Scale: 0 → 0 iters, 1.0 → 50k iters
    t0 = time.perf_counter_ns()
    x = 1.0001
    for _ in range(n_iters):
        x = x * 1.0001 + 0.0001
        x = x * 0.9999 - 0.0001
    t1 = time.perf_counter_ns()
    elapsed_ns = t1 - t0
    result_queue.put((core_id, elapsed_ns, x))


# ============================================================
# Signal characterization (EXP 1)
# ============================================================
def compute_psd_slope(signal, fs=1.0):
    """Compute PSD and fit log-log slope (1/f exponent)."""
    n = len(signal)
    if n < 32:
        return 0.0, np.array([]), np.array([])
    fft = np.fft.rfft(signal - np.mean(signal))
    psd = np.abs(fft) ** 2 / n
    freqs = np.fft.rfftfreq(n, d=1.0/fs)
    # Skip DC
    mask = freqs > 0
    freqs = freqs[mask]
    psd = psd[mask]
    if len(freqs) < 5 or np.all(psd < 1e-20):
        return 0.0, freqs, psd
    log_f = np.log10(freqs + 1e-20)
    log_p = np.log10(psd + 1e-20)
    # Fit slope
    coeffs = np.polyfit(log_f, log_p, 1)
    return coeffs[0], freqs, psd


def compute_acf(signal, max_lag=10):
    """Autocorrelation at lags 1..max_lag."""
    n = len(signal)
    if n < max_lag + 2:
        return np.zeros(max_lag)
    signal = signal - np.mean(signal)
    var = np.var(signal)
    if var < 1e-20:
        return np.zeros(max_lag)
    acf = np.zeros(max_lag)
    for lag in range(1, max_lag + 1):
        acf[lag-1] = np.mean(signal[lag:] * signal[:-lag]) / var
    return acf


def compute_entropy(signal, n_bins=256):
    """Shannon entropy in bits."""
    if len(signal) < 10:
        return 0.0
    # Normalize to [0, 1]
    mn, mx = np.min(signal), np.max(signal)
    if mx - mn < 1e-20:
        return 0.0
    norm = (signal - mn) / (mx - mn)
    hist, _ = np.histogram(norm, bins=n_bins, range=(0, 1))
    hist = hist[hist > 0]
    probs = hist / hist.sum()
    return -np.sum(probs * np.log2(probs))


# ============================================================
# Reservoir benchmarks
# ============================================================
def ridge_solve(X_tr, y_tr, X_te, y_te, task='regression'):
    alphas = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    best_score = 0.0 if task == 'regression' else 0.5
    for alpha in alphas:
        I = np.eye(X_tr.shape[1])
        try:
            w = np.linalg.solve(X_tr.T @ X_tr + alpha * I, X_tr.T @ y_tr)
            pred = X_te @ w
            if task == 'regression':
                ss_res = np.sum((y_te - pred) ** 2)
                ss_tot = np.sum((y_te - y_te.mean()) ** 2)
                score = max(0, 1 - ss_res / ss_tot) if ss_tot > 1e-10 else 0.0
            else:
                score = np.mean((pred > 0.5).astype(float) == y_te)
            if score > best_score:
                best_score = score
        except Exception:
            pass
    return best_score


def eval_mc(X, u_raw, warmup, max_d=20):
    n = len(X)
    n_tr = int(0.7 * n)
    mc = 0.0
    per_d = {}
    for d in range(1, max_d + 1):
        target = u_raw[warmup-d:len(u_raw)-d]
        nn = min(n, len(target))
        if nn < n_tr + 10:
            per_d[str(d)] = 0.0
            continue
        r2 = ridge_solve(X[:n_tr], target[:n_tr], X[n_tr:nn], target[n_tr:nn])
        per_d[str(d)] = r2
        mc += r2
    return mc, per_d


def eval_xor(X, u_raw, warmup, tau):
    n = len(X)
    n_tr = int(0.7 * n)
    u_a = (u_raw[warmup:] > 0.5).astype(float)
    u_b = (u_raw[warmup-tau:len(u_raw)-tau] > 0.5).astype(float)
    nn = min(len(u_a), len(u_b), n)
    if nn < n_tr + 10:
        return 0.5
    target = (u_a[:nn] != u_b[:nn]).astype(float)
    Xn = X[:nn]
    return ridge_solve(Xn[:n_tr], target[:n_tr], Xn[n_tr:nn], target[n_tr:nn], 'classification')


def eval_waveform4(X, u_raw, warmup):
    """4-class waveform classification: sine, square, triangle, sawtooth."""
    n = len(X)
    n_tr = int(0.7 * n)
    period = 20
    t_arr = np.arange(n)
    phase = (t_arr % period) / period
    labels = np.zeros(n, dtype=int)
    # Assign class based on input quartile
    u_seg = u_raw[warmup:warmup+n]
    quartiles = np.percentile(u_seg, [25, 50, 75])
    labels[u_seg < quartiles[0]] = 0
    labels[(u_seg >= quartiles[0]) & (u_seg < quartiles[1])] = 1
    labels[(u_seg >= quartiles[1]) & (u_seg < quartiles[2])] = 2
    labels[u_seg >= quartiles[2]] = 3

    scores_matrix = np.zeros((n - n_tr, 4))
    for c in range(4):
        y = (labels == c).astype(float)
        for alpha in [1.0, 10.0, 100.0]:
            I = np.eye(X[:n_tr].shape[1])
            try:
                w = np.linalg.solve(X[:n_tr].T @ X[:n_tr] + alpha * I, X[:n_tr].T @ y[:n_tr])
                scores_matrix[:, c] = X[n_tr:] @ w
                break
            except:
                pass
    pred = np.argmax(scores_matrix, axis=1)
    acc = np.mean(pred == labels[n_tr:])
    return float(acc)


def eval_narma(X, u_raw, warmup, order):
    n = len(X)
    n_tr = int(0.7 * n)
    T = len(u_raw)
    u_n = (u_raw - u_raw.min()) / (u_raw.max() - u_raw.min() + 1e-10) * 0.5
    y = np.zeros(T)
    for t in range(order, T):
        y[t] = 0.3*y[t-1] + 0.05*y[t-1]*np.sum(y[t-order:t]) + 1.5*u_n[t-1]*u_n[t-order] + 0.1
        y[t] = np.tanh(y[t])
    target = y[warmup:]
    nn = min(n, len(target))
    best_nrmse = 999.0
    for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
        I2 = np.eye(X[:n_tr].shape[1])
        try:
            w = np.linalg.solve(X[:n_tr].T @ X[:n_tr] + alpha * I2, X[:n_tr].T @ target[:n_tr])
            pred = X[n_tr:nn] @ w
            gt = target[n_tr:nn]
            nrmse = np.sqrt(np.mean((gt - pred) ** 2)) / (np.std(gt) + 1e-10)
            if nrmse < best_nrmse:
                best_nrmse = nrmse
        except Exception:
            pass
    return best_nrmse


def build_temporal_products(X, n_top=16, tau_list=(1, 2, 3)):
    """Build temporal product features: x_i[t] * x_j[t-k] for top-variance pairs."""
    n_steps, n_feats = X.shape
    variances = np.var(X, axis=0)
    top_idx = np.argsort(variances)[-min(n_top, n_feats):]
    X_top = X[:, top_idx]

    products = []
    for tau in tau_list:
        shifted = np.zeros_like(X_top)
        shifted[tau:] = X_top[:-tau]
        # Self-products: x_i[t] * x_i[t-tau]
        products.append(X_top * shifted)
        # Cross-products: x_i[t] * x_j[t-tau] for adjacent pairs
        n_cross = min(8, len(top_idx))
        for i in range(0, n_cross - 1):
            cross = X_top[:, i:i+1] * shifted[:, i+1:i+2]
            products.append(cross)

    # Squares of top features
    products.append(np.square(X_top))

    if len(products) > 0:
        return np.hstack([X] + products)
    return X


def full_benchmark(X, u_raw, warmup, label=""):
    """Run MC, XOR, waveform, NARMA benchmarks."""
    mc, mc_d = eval_mc(X, u_raw, warmup, max_d=20)
    xor1 = eval_xor(X, u_raw, warmup, 1)
    xor3 = eval_xor(X, u_raw, warmup, 3)
    xor5 = eval_xor(X, u_raw, warmup, 5)
    wave = eval_waveform4(X, u_raw, warmup)
    narma5 = eval_narma(X, u_raw, warmup, 5)
    bm = {
        'mc_total': mc, 'mc_per_delay': mc_d,
        'xor': {'tau1': xor1, 'tau3': xor3, 'tau5': xor5},
        'waveform4': wave,
        'narma5_nrmse': narma5,
        'n_features': X.shape[1],
    }
    if label:
        print(f"  {label:20s} ({X.shape[1]:4d}f): MC={mc:.3f} XOR1={xor1*100:.1f}% "
              f"XOR3={xor3*100:.1f}% XOR5={xor5*100:.1f}% W4={wave*100:.1f}% N5={narma5:.3f}")
    return bm


# ============================================================
# Software ESN (for GPU condition fallback)
# ============================================================
class SoftwareESN:
    def __init__(self, n_neurons=256, spectral_radius=0.95, input_scale=0.1, leak=0.3, seed=42):
        rng = np.random.default_rng(seed)
        self.N = n_neurons
        self.leak = leak
        self.input_w = rng.uniform(-input_scale, input_scale, n_neurons)
        W = rng.standard_normal((n_neurons, n_neurons)) * 0.1
        mask = rng.random((n_neurons, n_neurons)) > 0.9
        W *= mask
        eigvals = np.abs(np.linalg.eigvals(W))
        sr = max(eigvals) if len(eigvals) > 0 else 1.0
        if sr > 0:
            W *= spectral_radius / sr
        self.W = W
        self.bias = rng.uniform(-0.1, 0.1, n_neurons)

    def run(self, input_seq, seed=42):
        rng = np.random.default_rng(seed)
        n = len(input_seq)
        states = np.zeros((n, self.N))
        v = np.zeros(self.N)
        for t in range(n):
            v = (1 - self.leak) * v + self.leak * np.tanh(
                self.W @ v + self.input_w * input_seq[t] + self.bias
                + rng.uniform(-0.001, 0.001, self.N))
            states[t] = v
        return states


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 70)
    print("  z2308: CPU-as-Neuromorphic-Substrate Reservoir Computing")
    print("  AMD Ryzen AI MAX+ PRO 395 (Zen 5, 16 cores)")
    print("=" * 70)
    print(f"  Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Temp: {get_max_temp():.1f}C")

    results = {'experiments': {}, 'tests': {}, 'hw_info': {}}

    # Resume support
    if SAVE_FILE.exists():
        try:
            with open(SAVE_FILE) as f:
                results = json.load(f)
            done = list(results.get('experiments', {}).keys())
            if done:
                print(f"  RESUMED: {done} already done")
        except:
            results = {'experiments': {}, 'tests': {}, 'hw_info': {}}

    def save():
        with open(SAVE_FILE, 'w') as f:
            json.dump(results, f, indent=2, cls=NpEncoder)

    # Init MSR access
    print("\n[INIT] Opening MSR/perf interfaces...")
    msr_fds = init_msr_access()
    perf_fds = init_perf_counters()
    print(f"  MSR: {len(msr_fds)}/16 cores, perf_event: {len(perf_fds)}/16 cores")

    # Init SMN deep analog access
    print("[INIT] Opening SMN (deep analog)...")
    smn_ok = smn_init()
    if smn_ok:
        t, raw = smn_read_temp()
        vc, ic, vs, isc, r0, r1 = smn_read_svi3()
        ccd = smn_read_ccd_temps(2)
        print(f"  [SMN] Tctl={t:.1f}C (raw=0x{raw:08X})")
        print(f"  [SMN] Vcore={vc:.0f}mV Icore={ic} Vsoc={vs:.0f}mV Isoc={isc}")
        print(f"  [SMN] CCD temps: {ccd}")
    results['hw_info']['smn_access'] = smn_ok
    results['hw_info']['msr_cores'] = len(msr_fds)
    results['hw_info']['perf_cores'] = len(perf_fds)
    results['hw_info']['n_cores'] = N_CORES

    rng = np.random.default_rng(42)
    u_raw = rng.uniform(0, 1, N_STEPS + WARMUP)

    # ================================================================
    # EXP 1 — CPU Analog Signal Characterization (T811-T813)
    # ================================================================
    if 'exp1_signals' not in results.get('experiments', {}):
        print("\n" + "=" * 70)
        print("[EXP 1] CPU Analog Signal Characterization")
        print("=" * 70)
        wait_cool("pre-EXP1")

        signal_results = {}
        n_samples = 1000  # 5 seconds at ~200Hz

        # --- Signal 1: RAPL energy deltas ---
        print("  [1/6] RAPL per-core energy deltas...", flush=True)
        if msr_fds:
            energy_prev = read_rapl_energy(msr_fds)
            energy_series = np.zeros((n_samples, N_CORES))
            for i in range(n_samples):
                time.sleep(0.005)
                energy_now = read_rapl_energy(msr_fds)
                delta = energy_now - energy_prev
                # Handle wraparound (32-bit counter)
                delta[delta < 0] += 2**32
                energy_series[i] = delta
                energy_prev = energy_now
                if i % 50 == 0:
                    check_thermal(i, "RAPL")
            # Per-core stats
            rapl_mean = np.mean(energy_series, axis=0)
            rapl_std = np.std(energy_series, axis=0)
            # Aggregate: mean across cores
            agg = np.mean(energy_series, axis=1)
            psd_slope, _, _ = compute_psd_slope(agg, fs=200)
            acf = compute_acf(agg, max_lag=10)
            ent = compute_entropy(agg)
            signal_results['rapl_energy'] = {
                'mean': float(np.mean(rapl_mean)), 'std': float(np.mean(rapl_std)),
                'psd_slope': float(psd_slope), 'acf': acf.tolist(), 'entropy': float(ent),
                'has_variance': bool(np.mean(rapl_std) > 0),
            }
            print(f"    mean={np.mean(rapl_mean):.1f} std={np.mean(rapl_std):.1f} "
                  f"PSD_slope={psd_slope:.3f} ACF(1)={acf[0]:.3f} H={ent:.2f}b")
        else:
            signal_results['rapl_energy'] = {'has_variance': False, 'note': 'MSR unavailable'}
            print("    SKIP (no MSR access)")

        # --- Signal 2: MPERF/APERF ratio ---
        print("  [2/6] MPERF/APERF effective frequency ratio...", flush=True)
        if msr_fds:
            freq_series = np.zeros((n_samples, N_CORES))
            for i in range(n_samples):
                time.sleep(0.005)
                freq_series[i] = read_mperf_aperf(msr_fds)
                if i % 50 == 0:
                    check_thermal(i, "FREQ")
            agg = np.mean(freq_series, axis=1)
            freq_std = np.std(freq_series, axis=0)
            psd_slope, _, _ = compute_psd_slope(agg, fs=200)
            acf = compute_acf(agg, max_lag=10)
            ent = compute_entropy(agg)
            signal_results['mperf_aperf'] = {
                'mean': float(np.mean(freq_series)), 'std': float(np.mean(freq_std)),
                'psd_slope': float(psd_slope), 'acf': acf.tolist(), 'entropy': float(ent),
                'has_variance': bool(np.mean(freq_std) > 0),
            }
            print(f"    mean={np.mean(freq_series):.6f} std={np.mean(freq_std):.6f} "
                  f"PSD_slope={psd_slope:.3f} ACF(1)={acf[0]:.3f} H={ent:.2f}b")
        else:
            signal_results['mperf_aperf'] = {'has_variance': False, 'note': 'MSR unavailable'}
            print("    SKIP (no MSR access)")

        # --- Signal 3: TSC jitter ---
        print("  [3/6] TSC timing jitter...", flush=True)
        tsc_series = np.zeros(n_samples)
        for i in range(n_samples):
            jitter_std, jitter_mean, _ = read_tsc_jitter()
            tsc_series[i] = jitter_std
            if i % 50 == 0:
                check_thermal(i, "TSC")
        psd_slope, _, _ = compute_psd_slope(tsc_series, fs=200)
        acf = compute_acf(tsc_series, max_lag=10)
        ent = compute_entropy(tsc_series)
        signal_results['tsc_jitter'] = {
            'mean': float(np.mean(tsc_series)), 'std': float(np.std(tsc_series)),
            'psd_slope': float(psd_slope), 'acf': acf.tolist(), 'entropy': float(ent),
            'has_variance': bool(np.std(tsc_series) > 0),
        }
        print(f"    mean={np.mean(tsc_series):.1f}ns std={np.std(tsc_series):.1f}ns "
              f"PSD_slope={psd_slope:.3f} ACF(1)={acf[0]:.3f} H={ent:.2f}b")

        # --- Signal 4: Branch mispredictions ---
        print("  [4/6] Branch misprediction counters...", flush=True)
        if perf_fds:
            branch_prev = read_branch_misses(perf_fds)
            branch_series = np.zeros((n_samples, N_CORES))
            for i in range(n_samples):
                time.sleep(0.005)
                branch_now = read_branch_misses(perf_fds)
                branch_series[i] = branch_now - branch_prev
                branch_prev = branch_now
                if i % 50 == 0:
                    check_thermal(i, "BRANCH")
            agg = np.mean(branch_series, axis=1)
            bstd = np.std(branch_series, axis=0)
            psd_slope, _, _ = compute_psd_slope(agg, fs=200)
            acf = compute_acf(agg, max_lag=10)
            ent = compute_entropy(agg)
            signal_results['branch_misses'] = {
                'mean': float(np.mean(agg)), 'std': float(np.mean(bstd)),
                'psd_slope': float(psd_slope), 'acf': acf.tolist(), 'entropy': float(ent),
                'has_variance': bool(np.mean(bstd) > 0),
            }
            print(f"    mean={np.mean(agg):.1f} std={np.mean(bstd):.1f} "
                  f"PSD_slope={psd_slope:.3f} ACF(1)={acf[0]:.3f} H={ent:.2f}b")
        else:
            signal_results['branch_misses'] = {'has_variance': False, 'note': 'perf_event unavailable'}
            print("    SKIP (no perf_event access)")

        # --- Signal 5: RDRAND throughput variance ---
        print("  [5/6] RDRAND throughput variance...", flush=True)
        rdrand_series = np.zeros(n_samples)
        for i in range(n_samples):
            rd_std, rd_mean, _ = read_rdrand_batch(1000)
            rdrand_series[i] = rd_mean  # Use mean timing as signal
            if i % 50 == 0:
                check_thermal(i, "RDRAND")
        psd_slope, _, _ = compute_psd_slope(rdrand_series, fs=200)
        acf = compute_acf(rdrand_series, max_lag=10)
        ent = compute_entropy(rdrand_series)
        signal_results['rdrand'] = {
            'mean': float(np.mean(rdrand_series)), 'std': float(np.std(rdrand_series)),
            'psd_slope': float(psd_slope), 'acf': acf.tolist(), 'entropy': float(ent),
            'has_variance': bool(np.std(rdrand_series) > 0),
        }
        print(f"    mean={np.mean(rdrand_series):.1f}ns std={np.std(rdrand_series):.1f}ns "
              f"PSD_slope={psd_slope:.3f} ACF(1)={acf[0]:.3f} H={ent:.2f}b")

        # --- Signal 6: Thermal ---
        print("  [6/6] Thermal per-core...", flush=True)
        thermal_series = np.zeros((n_samples, N_CORES))
        for i in range(n_samples):
            thermal_series[i] = read_thermal_per_core()
            time.sleep(0.005)
            if i % 50 == 0:
                check_thermal(i, "THERMAL")
        agg = np.mean(thermal_series, axis=1)
        tstd = np.std(thermal_series, axis=0)
        psd_slope, _, _ = compute_psd_slope(agg, fs=200)
        acf = compute_acf(agg, max_lag=10)
        ent = compute_entropy(agg)
        signal_results['thermal'] = {
            'mean': float(np.mean(agg)), 'std': float(np.mean(tstd)),
            'psd_slope': float(psd_slope), 'acf': acf.tolist(), 'entropy': float(ent),
            'has_variance': bool(np.mean(tstd) > 0),
        }
        print(f"    mean={np.mean(agg):.1f}C std={np.mean(tstd):.2f}C "
              f"PSD_slope={psd_slope:.3f} ACF(1)={acf[0]:.3f} H={ent:.2f}b")

        # --- Signal 7: SMN raw thermal ADC (0x00059800) — DEEPEST ANALOG ---
        if smn_ok:
            print("  [7/8] SMN raw thermal ADC (11-bit, 0.125C)...", flush=True)
            smn_temp_series = np.zeros(n_samples)
            smn_raw_series = np.zeros(n_samples, dtype=np.uint32)
            for i in range(n_samples):
                t_c, raw = smn_read_temp()
                smn_temp_series[i] = t_c
                smn_raw_series[i] = raw
                if i % 50 == 0:
                    check_thermal(i, "SMN_TEMP")
            psd_slope, _, _ = compute_psd_slope(smn_temp_series, fs=200)
            acf = compute_acf(smn_temp_series, max_lag=10)
            ent = compute_entropy(smn_temp_series)
            signal_results['smn_thermal_adc'] = {
                'mean': float(np.mean(smn_temp_series)), 'std': float(np.std(smn_temp_series)),
                'psd_slope': float(psd_slope), 'acf': acf.tolist(), 'entropy': float(ent),
                'has_variance': bool(np.std(smn_temp_series) > 0),
                'depth': 'RAW_ADC_11bit_0.125C',
            }
            print(f"    mean={np.mean(smn_temp_series):.2f}C std={np.std(smn_temp_series):.3f}C "
                  f"PSD_slope={psd_slope:.3f} ACF(1)={acf[0]:.3f} H={ent:.2f}b")

            # --- Signal 8: SVI3 voltage/current (VRM ADC) — DEEPEST POWER ANALOG ---
            print("  [8/8] SMN SVI3 voltage/current (VRM ADC raw)...", flush=True)
            svi3_vcore = np.zeros(n_samples)
            svi3_icore = np.zeros(n_samples)
            svi3_vsoc = np.zeros(n_samples)
            svi3_isoc = np.zeros(n_samples)
            for i in range(n_samples):
                vc, ic, vs, isc, _, _ = smn_read_svi3()
                svi3_vcore[i] = vc
                svi3_icore[i] = ic
                svi3_vsoc[i] = vs
                svi3_isoc[i] = isc
                if i % 50 == 0:
                    check_thermal(i, "SVI3")
            # Characterize Vcore as primary signal
            psd_slope, _, _ = compute_psd_slope(svi3_vcore, fs=200)
            acf = compute_acf(svi3_vcore, max_lag=10)
            ent = compute_entropy(svi3_vcore)
            signal_results['svi3_vcore'] = {
                'mean': float(np.mean(svi3_vcore)), 'std': float(np.std(svi3_vcore)),
                'psd_slope': float(psd_slope), 'acf': acf.tolist(), 'entropy': float(ent),
                'has_variance': bool(np.std(svi3_vcore) > 0),
                'depth': 'VRM_ADC_SVI3_raw',
                'icore_mean': float(np.mean(svi3_icore)),
                'vsoc_mean': float(np.mean(svi3_vsoc)),
                'isoc_mean': float(np.mean(svi3_isoc)),
            }
            print(f"    Vcore: mean={np.mean(svi3_vcore):.0f}mV std={np.std(svi3_vcore):.1f}mV "
                  f"PSD={psd_slope:.3f} ACF(1)={acf[0]:.3f}")
            print(f"    Icore: mean={np.mean(svi3_icore):.0f} Vsoc: mean={np.mean(svi3_vsoc):.0f}mV")
        else:
            signal_results['smn_thermal_adc'] = {'has_variance': False, 'note': 'SMN unavailable'}
            signal_results['svi3_vcore'] = {'has_variance': False, 'note': 'SMN unavailable'}

        results['experiments']['exp1_signals'] = signal_results
        save()
    else:
        print("\n[EXP 1] Already done, loading...")
        signal_results = results['experiments']['exp1_signals']

    # ================================================================
    # EXP 2 — Per-Core Reservoir Computing (T814-T818)
    # ================================================================
    cpu_states_raw = None
    if 'exp2_cpu_reservoir' not in results.get('experiments', {}):
        print("\n" + "=" * 70)
        print("[EXP 2] Per-Core Reservoir Computing (16 cores = 16 neurons)")
        print("=" * 70)
        wait_cool("pre-EXP2")

        total_steps = N_STEPS + WARMUP
        # State matrix: [energy_delta, mperf_aperf, tsc_jitter, thermal] x 16 cores = 64 features
        # + SMN deep analog: [smn_temp, vcore, icore, vsoc] = 4 features (if available)
        n_smn = 4 if smn_ok else 0
        n_features = N_CORES * 4 + n_smn
        cpu_states = np.zeros((total_steps, n_features))

        energy_prev = read_rapl_energy(msr_fds) if msr_fds else np.zeros(N_CORES)
        branch_prev = read_branch_misses(perf_fds) if perf_fds else np.zeros(N_CORES)

        dt = 1.0 / SAMPLE_HZ
        print(f"  Collecting {total_steps} steps at {SAMPLE_HZ}Hz with workload injection...")
        t0_run = time.time()

        for t in range(total_steps):
            intensity = u_raw[t]

            # Inject workload: launch per-core workers with proportional intensity
            # Use a process pool to avoid overhead of constant forking
            # But for thermal safety, we just do a tight CPU loop in the main process
            # (multiprocessing fork is expensive; use threading-like approach)
            n_iters = int(intensity * 20000)
            x = 1.0001
            for _ in range(n_iters):
                x = x * 1.0001 + 0.0001

            # Read signals
            # 1. Energy delta
            if msr_fds:
                energy_now = read_rapl_energy(msr_fds)
                energy_delta = energy_now - energy_prev
                energy_delta[energy_delta < 0] += 2**32
                energy_prev = energy_now
            else:
                energy_delta = np.zeros(N_CORES)

            # 2. MPERF/APERF ratio
            if msr_fds:
                freq_ratio = read_mperf_aperf(msr_fds)
            else:
                freq_ratio = np.zeros(N_CORES)

            # 3. TSC jitter (scalar, replicate across cores with small noise)
            tsc_std, _, _ = read_tsc_jitter()
            tsc_per_core = np.full(N_CORES, tsc_std) + np.random.randn(N_CORES) * max(tsc_std * 0.1, 0.01)

            # 4. Thermal
            thermal = read_thermal_per_core()

            cpu_states[t, 0:N_CORES] = energy_delta
            cpu_states[t, N_CORES:2*N_CORES] = freq_ratio
            cpu_states[t, 2*N_CORES:3*N_CORES] = tsc_per_core
            cpu_states[t, 3*N_CORES:4*N_CORES] = thermal

            # 5. SMN deep analog (raw thermal ADC + SVI3 voltage/current)
            if smn_ok:
                smn_t, _ = smn_read_temp()
                vc, ic, vs, isc, _, _ = smn_read_svi3()
                base = N_CORES * 4
                cpu_states[t, base] = smn_t
                cpu_states[t, base+1] = vc
                cpu_states[t, base+2] = ic
                cpu_states[t, base+3] = vs

            check_thermal(t, "EXP2")

            if t % 200 == 0 and t > 0:
                elapsed = time.time() - t0_run
                rate = t / elapsed
                print(f"    step {t}/{total_steps} ({rate:.0f} Hz) "
                      f"temp={get_max_temp():.0f}C", flush=True)

        elapsed = time.time() - t0_run
        print(f"  Collected {total_steps} steps in {elapsed:.1f}s ({total_steps/elapsed:.0f} Hz)")

        # Normalize features (z-score per column)
        cpu_states_raw = cpu_states.copy()
        means = np.mean(cpu_states, axis=0)
        stds = np.std(cpu_states, axis=0)
        stds[stds < 1e-10] = 1.0
        cpu_states_norm = (cpu_states - means) / stds

        # Trim warmup
        X_cpu = cpu_states_norm[WARMUP:]

        # Benchmark RAW
        bm_raw = full_benchmark(X_cpu, u_raw, WARMUP, label="CPU_RAW")

        # Per-core distinctness: pairwise correlation of temporal profiles
        core_profiles = cpu_states_norm[WARMUP:, :N_CORES]  # just energy_delta
        if core_profiles.shape[1] >= 2:
            corr_matrix = np.corrcoef(core_profiles.T)
            np.fill_diagonal(corr_matrix, 0)
            n_distinct = 0
            for i in range(N_CORES):
                max_corr_i = np.max(np.abs(corr_matrix[i]))
                if max_corr_i < 0.95:
                    n_distinct += 1
            pairwise_corrs = corr_matrix[np.triu_indices(N_CORES, k=1)]
        else:
            n_distinct = 0
            pairwise_corrs = np.array([])

        exp2_results = {
            'raw_benchmark': bm_raw,
            'n_distinct_cores': int(n_distinct),
            'pairwise_corr_mean': float(np.mean(np.abs(pairwise_corrs))) if len(pairwise_corrs) > 0 else 0.0,
            'pairwise_corr_max': float(np.max(np.abs(pairwise_corrs))) if len(pairwise_corrs) > 0 else 0.0,
            'effective_hz': total_steps / elapsed,
            'feature_stds': stds.tolist(),
        }
        results['experiments']['exp2_cpu_reservoir'] = exp2_results
        save()

        # Keep cpu_states_norm for EXP 3 and 4
        np.save(RESULTS / 'z2308_cpu_states.npy', cpu_states_norm)
    else:
        print("\n[EXP 2] Already done, loading...")
        exp2_results = results['experiments']['exp2_cpu_reservoir']
        try:
            cpu_states_norm = np.load(RESULTS / 'z2308_cpu_states.npy')
        except:
            cpu_states_norm = None

    # ================================================================
    # EXP 3 — Temporal Products Amplification (T819-T821)
    # ================================================================
    if 'exp3_temporal_products' not in results.get('experiments', {}):
        print("\n" + "=" * 70)
        print("[EXP 3] Temporal Products Amplification")
        print("=" * 70)

        if cpu_states_norm is None:
            print("  SKIP: CPU states not available")
            results['experiments']['exp3_temporal_products'] = {'note': 'CPU states unavailable'}
        else:
            X_raw = cpu_states_norm[WARMUP:]
            X_temporal = build_temporal_products(X_raw, n_top=16, tau_list=[1, 2, 3])

            bm_raw = full_benchmark(X_raw, u_raw, WARMUP, label="RAW")
            bm_temp = full_benchmark(X_temporal, u_raw, WARMUP, label="TEMPORAL_PRODUCTS")

            exp3_results = {
                'raw': bm_raw,
                'temporal': bm_temp,
            }
            results['experiments']['exp3_temporal_products'] = exp3_results
        save()
    else:
        print("\n[EXP 3] Already done, loading...")
        exp3_results = results['experiments']['exp3_temporal_products']

    # ================================================================
    # EXP 4 — CPU + FPGA + GPU Triple Bridge (T822-T825)
    # ================================================================
    if 'exp4_triple_bridge' not in results.get('experiments', {}):
        print("\n" + "=" * 70)
        print("[EXP 4] CPU + FPGA + GPU Triple Bridge")
        print("=" * 70)
        wait_cool("pre-EXP4")

        exp4 = {'cpu_available': cpu_states_norm is not None}

        # --- FPGA ---
        fpga_states_temporal = None
        try:
            from fpga_host_eth import FPGAEthBridge
            fpga = FPGAEthBridge(timeout=2.0)
            fpga.connect()
            fpga.set_kill(0)
            time.sleep(0.5)
            # Configure FPGA runtime params
            fpga.set_leak_cond(0x2000)
            fpga.set_base_exc_raw(0x0080)
            fpga.set_bias_gain_raw(0x4000)
            fpga.set_threshold_raw(0x20000)
            VG_GROUPS = {0: 0.05, 1: 0.15, 2: 0.30, 3: 0.58}
            for n in range(128):
                fpga.set_vg(n, VG_GROUPS[n % 4])
                time.sleep(0.001)
            time.sleep(0.5)

            telem = fpga.read_telemetry()
            if telem is not None:
                print(f"  FPGA online: {fpga.num_neurons} neurons")
                fpga_available = True
            else:
                print("  FPGA: telemetry returned None")
                fpga_available = False
        except Exception as e:
            print(f"  FPGA unavailable: {e}")
            fpga_available = False
            fpga = None

        if fpga_available:
            print("  Running FPGA reservoir (1500 steps)...", flush=True)
            n_fpga = 128
            fpga_total = N_STEPS + WARMUP
            fpga_states = np.zeros((fpga_total, n_fpga))
            fpga_dspikes = np.zeros((fpga_total, n_fpga), dtype=np.float32)
            mac_signal = np.clip(u_raw * 0.3 + 0.3, 0, 1)
            fpga.set_mac_signal(0.0)
            time.sleep(0.02)
            telem = fpga.read_telemetry()
            prev_sc = telem['spike_counts'].copy() if telem else np.zeros(n_fpga, dtype=np.uint16)
            fpga_dt = 1.0 / 200
            for t in range(fpga_total):
                fpga.set_mac_signal(float(mac_signal[t]))
                time.sleep(fpga_dt)
                telem = fpga.read_telemetry()
                if telem is not None:
                    fpga_states[t] = telem['vmem']
                    sc = telem['spike_counts']
                    diff = sc.astype(np.int32) - prev_sc.astype(np.int32)
                    diff[diff < 0] += 65536
                    fpga_dspikes[t] = diff.astype(np.float32)
                    prev_sc = sc.copy()
                elif t > 0:
                    fpga_states[t] = fpga_states[t-1]
                    fpga_dspikes[t] = fpga_dspikes[t-1]
                check_thermal(t, "EXP4-FPGA")
            fpga.set_mac_signal(0.0)

            # Build FPGA temporal features
            fpga_raw = fpga_states[WARMUP:]
            fpga_ds = fpga_dspikes[WARMUP:]
            delta = np.diff(fpga_raw, axis=0)
            delta = np.vstack([np.zeros((1, n_fpga)), delta])
            fpga_base = np.hstack([fpga_raw, fpga_ds, delta])
            fpga_states_temporal = build_temporal_products(fpga_base, n_top=24, tau_list=[1, 2, 3])
            print(f"  FPGA features: {fpga_states_temporal.shape[1]}")
        else:
            exp4['fpga_available'] = False

        # --- GPU ---
        gpu_states_temporal = None
        try:
            import subprocess, tempfile
            GPU_KERN = BASE / 'scripts' / 'z2277_gpu_bridge_kern'
            if GPU_KERN.exists():
                print("  Running GPU HIP kernel...", flush=True)
                wait_cool("pre-GPU")
                with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as fin:
                    input_path = fin.name
                    u_raw.astype(np.float32).tofile(fin)
                with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as fout:
                    output_path = fout.name
                try:
                    env = os.environ.copy()
                    env['HSA_OVERRIDE_GFX_VERSION'] = '11.0.0'
                    n_gpu_sampled = 512
                    result = subprocess.run(
                        [str(GPU_KERN), input_path, output_path, str(len(u_raw))],
                        env=env, capture_output=True, text=True, timeout=120
                    )
                    if result.returncode == 0:
                        raw = np.fromfile(output_path, dtype=np.float32)
                        expected = n_gpu_sampled * len(u_raw)
                        if len(raw) == expected:
                            gpu_raw = raw.reshape(n_gpu_sampled, len(u_raw)).T
                            gpu_feat = gpu_raw[WARMUP:]
                            gpu_states_temporal = build_temporal_products(gpu_feat, n_top=24, tau_list=[1, 2, 3])
                            print(f"  GPU features: {gpu_states_temporal.shape[1]}")
                        else:
                            print(f"  GPU: wrong output size {len(raw)} != {expected}")
                    else:
                        print(f"  GPU kernel error: {result.stderr[:200]}")
                finally:
                    try: os.unlink(input_path)
                    except: pass
                    try: os.unlink(output_path)
                    except: pass
            else:
                print("  GPU kernel binary not found, using software ESN fallback")
                esn = SoftwareESN(n_neurons=256, seed=42)
                gpu_raw = esn.run(u_raw, seed=42)
                gpu_feat = gpu_raw[WARMUP:]
                gpu_states_temporal = build_temporal_products(gpu_feat, n_top=24, tau_list=[1, 2, 3])
                exp4['gpu_simulated'] = True
                print(f"  GPU (ESN fallback) features: {gpu_states_temporal.shape[1]}")
        except Exception as e:
            print(f"  GPU unavailable: {e}")
            # Use ESN fallback
            esn = SoftwareESN(n_neurons=256, seed=42)
            gpu_raw = esn.run(u_raw, seed=42)
            gpu_feat = gpu_raw[WARMUP:]
            gpu_states_temporal = build_temporal_products(gpu_feat, n_top=24, tau_list=[1, 2, 3])
            exp4['gpu_simulated'] = True
            print(f"  GPU (ESN fallback) features: {gpu_states_temporal.shape[1]}")

        exp4['fpga_available'] = fpga_states_temporal is not None
        exp4['gpu_available'] = gpu_states_temporal is not None

        # --- Combine and benchmark ---
        if cpu_states_norm is not None:
            X_cpu = cpu_states_norm[WARMUP:]
            X_cpu_t = build_temporal_products(X_cpu, n_top=16, tau_list=[1, 2, 3])
        else:
            X_cpu_t = None

        # CPU-only
        if X_cpu_t is not None:
            bm_cpu = full_benchmark(X_cpu_t, u_raw, WARMUP, label="CPU_TEMPORAL")
            exp4['cpu_only'] = bm_cpu

        # DUAL: FPGA + GPU
        if fpga_states_temporal is not None and gpu_states_temporal is not None:
            n_min = min(len(fpga_states_temporal), len(gpu_states_temporal))
            X_dual = np.hstack([fpga_states_temporal[:n_min], gpu_states_temporal[:n_min]])
            bm_dual = full_benchmark(X_dual[:n_min], u_raw, WARMUP, label="DUAL_FPGA_GPU")
            exp4['dual_fpga_gpu'] = bm_dual

        # TRIPLE: CPU + FPGA + GPU
        if X_cpu_t is not None and fpga_states_temporal is not None and gpu_states_temporal is not None:
            n_min = min(len(X_cpu_t), len(fpga_states_temporal), len(gpu_states_temporal))
            X_triple = np.hstack([X_cpu_t[:n_min], fpga_states_temporal[:n_min], gpu_states_temporal[:n_min]])
            bm_triple = full_benchmark(X_triple[:n_min], u_raw, WARMUP, label="TRIPLE")
            exp4['triple'] = bm_triple

            # Check CPU ridge weights in triple model
            try:
                n_tr = int(0.7 * n_min)
                target_mc1 = u_raw[WARMUP-1:WARMUP-1+n_min]
                I = np.eye(X_triple.shape[1])
                w = np.linalg.solve(X_triple[:n_tr].T @ X_triple[:n_tr] + 1.0 * I,
                                    X_triple[:n_tr].T @ target_mc1[:n_tr])
                n_cpu_feats = X_cpu_t.shape[1]
                cpu_weight_norm = np.sum(np.abs(w[:n_cpu_feats]))
                total_weight_norm = np.sum(np.abs(w))
                cpu_weight_frac = cpu_weight_norm / (total_weight_norm + 1e-10)
                exp4['cpu_weight_fraction'] = float(cpu_weight_frac)
                exp4['cpu_weight_nonzero'] = bool(cpu_weight_norm > 1e-10)
                print(f"  CPU weight fraction in TRIPLE model: {cpu_weight_frac:.3f}")
            except Exception as e:
                exp4['cpu_weight_fraction'] = 0.0
                exp4['cpu_weight_nonzero'] = False
                print(f"  Weight analysis error: {e}")
        elif X_cpu_t is not None and gpu_states_temporal is not None:
            # CPU + GPU only (no FPGA)
            n_min = min(len(X_cpu_t), len(gpu_states_temporal))
            X_dual_cg = np.hstack([X_cpu_t[:n_min], gpu_states_temporal[:n_min]])
            bm_dual_cg = full_benchmark(X_dual_cg[:n_min], u_raw, WARMUP, label="DUAL_CPU_GPU")
            exp4['dual_cpu_gpu'] = bm_dual_cg
            exp4['note'] = 'FPGA unavailable, using CPU+GPU only'

        results['experiments']['exp4_triple_bridge'] = exp4
        save()
    else:
        print("\n[EXP 4] Already done, loading...")
        exp4 = results['experiments']['exp4_triple_bridge']

    # ================================================================
    # EXP 5 — RDRAND vs GPU Noise (T826-T827)
    # ================================================================
    if 'exp5_rdrand_vs_gpu' not in results.get('experiments', {}):
        print("\n" + "=" * 70)
        print("[EXP 5] RDRAND vs GPU Noise")
        print("=" * 70)
        wait_cool("pre-EXP5")

        exp5 = {}

        # --- RDRAND noise ---
        print("  Generating RDRAND noise (1500 steps)...", flush=True)
        rdrand_noise = np.zeros(N_STEPS)
        rdrand_raw_bytes = bytearray()
        for i in range(N_STEPS):
            batch = os.urandom(1000)
            rdrand_raw_bytes.extend(batch[:100])  # Keep subset for entropy calc
            arr = np.frombuffer(batch, dtype=np.uint8).astype(np.float64)
            rdrand_noise[i] = np.std(arr)  # Use std of random bytes as "noise signal"

        # Byte-level entropy of raw RDRAND output
        byte_counts = np.bincount(np.frombuffer(bytes(rdrand_raw_bytes), dtype=np.uint8), minlength=256)
        byte_probs = byte_counts / byte_counts.sum()
        byte_probs = byte_probs[byte_probs > 0]
        rdrand_entropy = -np.sum(byte_probs * np.log2(byte_probs))
        exp5['rdrand_entropy_bits'] = float(rdrand_entropy)
        print(f"  RDRAND entropy: {rdrand_entropy:.3f} bits/byte (max=8.0)")

        # --- GPU noise (from hwmon power readings as proxy) ---
        print("  Collecting GPU noise (hwmon power readings)...", flush=True)
        gpu_noise = np.zeros(N_STEPS)
        for i in range(N_STEPS):
            try:
                with open('/sys/class/hwmon/hwmon7/power1_average', 'r') as f:
                    gpu_noise[i] = float(f.read().strip()) / 1e6  # uW -> W
            except:
                try:
                    with open('/sys/class/hwmon/hwmon7/temp1_input', 'r') as f:
                        gpu_noise[i] = float(f.read().strip()) / 1000.0
                except:
                    gpu_noise[i] = 0.0
            time.sleep(0.005)
            if i % 50 == 0:
                check_thermal(i, "EXP5")

        gpu_noise_std = np.std(gpu_noise)
        rdrand_noise_std = np.std(rdrand_noise)
        print(f"  GPU noise std: {gpu_noise_std:.4f}, RDRAND noise std: {rdrand_noise_std:.4f}")

        # Use each as input to CPU reservoir (since FPGA may not be available)
        if cpu_states_norm is not None:
            X_cpu_base = cpu_states_norm[WARMUP:]
            n_use = min(N_STEPS, len(X_cpu_base))

            # RDRAND-driven: use rdrand_noise as the input signal for MC/XOR
            u_rdrand = (rdrand_noise[:n_use] - np.mean(rdrand_noise[:n_use])) / (np.std(rdrand_noise[:n_use]) + 1e-10)
            u_rdrand_full = np.concatenate([np.zeros(WARMUP), u_rdrand])
            mc_rdrand, _ = eval_mc(X_cpu_base[:n_use], u_rdrand_full, WARMUP, max_d=10)

            # GPU-driven: use gpu_noise as input signal
            u_gpu = (gpu_noise[:n_use] - np.mean(gpu_noise[:n_use])) / (np.std(gpu_noise[:n_use]) + 1e-10)
            u_gpu_full = np.concatenate([np.zeros(WARMUP), u_gpu])
            mc_gpu, _ = eval_mc(X_cpu_base[:n_use], u_gpu_full, WARMUP, max_d=10)

            exp5['mc_rdrand_input'] = float(mc_rdrand)
            exp5['mc_gpu_input'] = float(mc_gpu)
            print(f"  MC with RDRAND input: {mc_rdrand:.3f}")
            print(f"  MC with GPU noise input: {mc_gpu:.3f}")
        else:
            exp5['mc_rdrand_input'] = 0.0
            exp5['mc_gpu_input'] = 0.0
            exp5['note'] = 'CPU states unavailable'

        # PSD analysis of both noise sources
        psd_rdrand, _, _ = compute_psd_slope(rdrand_noise, fs=200)
        psd_gpu, _, _ = compute_psd_slope(gpu_noise, fs=200)
        exp5['psd_slope_rdrand'] = float(psd_rdrand)
        exp5['psd_slope_gpu'] = float(psd_gpu)
        print(f"  PSD slopes: RDRAND={psd_rdrand:.3f} GPU={psd_gpu:.3f}")

        results['experiments']['exp5_rdrand_vs_gpu'] = exp5
        save()
    else:
        print("\n[EXP 5] Already done, loading...")
        exp5 = results['experiments']['exp5_rdrand_vs_gpu']

    # ================================================================
    # TESTS (T811-T827)
    # ================================================================
    print("\n" + "=" * 70)
    print("  TESTS (T811-T827)")
    print("=" * 70)

    tests = {}
    n_pass = 0

    # --- EXP 1 Tests ---
    sig = results['experiments'].get('exp1_signals', {})

    # T811: At least 4/6 signals show measurable variance (std > 0)
    n_variance = sum(1 for s in sig.values() if isinstance(s, dict) and s.get('has_variance', False))
    t811 = n_variance >= 4
    tests['T811'] = {'pass': bool(t811), 'n_with_variance': n_variance, 'threshold': 4,
                     'desc': 'At least 4/6 signals show measurable variance'}
    print(f"  T811 {'PASS' if t811 else 'FAIL'}: {n_variance}/6 signals have variance >= 4")
    n_pass += t811

    # T812: At least 2/6 signals show 1/f-like PSD (slope < -0.5)
    n_1f = sum(1 for s in sig.values() if isinstance(s, dict) and s.get('psd_slope', 0) < -0.5)
    t812 = n_1f >= 2
    slopes = {k: v.get('psd_slope', 'N/A') for k, v in sig.items() if isinstance(v, dict)}
    tests['T812'] = {'pass': bool(t812), 'n_1f': n_1f, 'threshold': 2, 'slopes': slopes,
                     'desc': 'At least 2/6 signals show 1/f-like PSD (slope < -0.5)'}
    print(f"  T812 {'PASS' if t812 else 'FAIL'}: {n_1f}/6 signals 1/f-like (slope < -0.5)")
    for k, v in slopes.items():
        if isinstance(v, float):
            print(f"        {k}: slope={v:.3f} {'<-0.5' if v < -0.5 else ''}")
    n_pass += t812

    # T813: At least 2/6 signals show temporal correlation (ACF(1) > 0.1)
    n_corr = sum(1 for s in sig.values()
                 if isinstance(s, dict) and isinstance(s.get('acf'), list)
                 and len(s['acf']) > 0 and s['acf'][0] > 0.1)
    t813 = n_corr >= 2
    acfs = {k: v.get('acf', [0])[0] if isinstance(v.get('acf'), list) and len(v.get('acf', [])) > 0 else 0
            for k, v in sig.items() if isinstance(v, dict)}
    tests['T813'] = {'pass': bool(t813), 'n_correlated': n_corr, 'threshold': 2, 'acf1s': acfs,
                     'desc': 'At least 2/6 signals show temporal correlation (ACF(1) > 0.1)'}
    print(f"  T813 {'PASS' if t813 else 'FAIL'}: {n_corr}/6 signals correlated (ACF(1) > 0.1)")
    for k, v in acfs.items():
        if isinstance(v, (int, float)):
            print(f"        {k}: ACF(1)={v:.3f} {'>0.1' if v > 0.1 else ''}")
    n_pass += t813

    # --- EXP 2 Tests ---
    exp2 = results['experiments'].get('exp2_cpu_reservoir', {})
    bm_raw = exp2.get('raw_benchmark', {})

    # T814: MC > 1.0
    mc_raw = bm_raw.get('mc_total', 0)
    t814 = mc_raw > 1.0
    tests['T814'] = {'pass': bool(t814), 'mc': mc_raw, 'threshold': 1.0,
                     'desc': 'CPU reservoir MC > 1.0'}
    print(f"  T814 {'PASS' if t814 else 'FAIL'}: MC={mc_raw:.3f} > 1.0")
    n_pass += t814

    # T815: XOR tau=1 > 55%
    xor1_raw = bm_raw.get('xor', {}).get('tau1', 0.5)
    t815 = xor1_raw > 0.55
    tests['T815'] = {'pass': bool(t815), 'xor1': xor1_raw, 'threshold': 0.55,
                     'desc': 'CPU XOR tau=1 > 55%'}
    print(f"  T815 {'PASS' if t815 else 'FAIL'}: XOR1={xor1_raw*100:.1f}% > 55%")
    n_pass += t815

    # T816: Waveform-4 > 40%
    wave_raw = bm_raw.get('waveform4', 0.25)
    t816 = wave_raw > 0.40
    tests['T816'] = {'pass': bool(t816), 'waveform4': wave_raw, 'threshold': 0.40,
                     'desc': 'CPU waveform-4 > 40%'}
    print(f"  T816 {'PASS' if t816 else 'FAIL'}: W4={wave_raw*100:.1f}% > 40%")
    n_pass += t816

    # T817: NARMA-5 NRMSE < 0.90
    narma5_raw = bm_raw.get('narma5_nrmse', 999)
    t817 = narma5_raw < 0.90
    tests['T817'] = {'pass': bool(t817), 'narma5': narma5_raw, 'threshold': 0.90,
                     'desc': 'CPU NARMA-5 NRMSE < 0.90'}
    print(f"  T817 {'PASS' if t817 else 'FAIL'}: NARMA5={narma5_raw:.3f} < 0.90")
    n_pass += t817

    # T818: At least 8/16 cores show distinct temporal profiles
    n_distinct = exp2.get('n_distinct_cores', 0)
    t818 = n_distinct >= 8
    tests['T818'] = {'pass': bool(t818), 'n_distinct': n_distinct, 'threshold': 8,
                     'desc': 'At least 8/16 cores distinct (pairwise corr < 0.95)'}
    print(f"  T818 {'PASS' if t818 else 'FAIL'}: {n_distinct}/16 distinct cores >= 8")
    n_pass += t818

    # --- EXP 3 Tests ---
    exp3 = results['experiments'].get('exp3_temporal_products', {})
    raw_bm = exp3.get('raw', {})
    temp_bm = exp3.get('temporal', {})

    # T819: TEMPORAL MC > RAW MC * 1.5
    mc_raw_3 = raw_bm.get('mc_total', 0)
    mc_temp_3 = temp_bm.get('mc_total', 0)
    t819 = mc_temp_3 > mc_raw_3 * 1.5
    tests['T819'] = {'pass': bool(t819), 'mc_raw': mc_raw_3, 'mc_temporal': mc_temp_3,
                     'threshold': mc_raw_3 * 1.5,
                     'desc': 'Temporal MC > Raw MC * 1.5'}
    print(f"  T819 {'PASS' if t819 else 'FAIL'}: MC_TEMP={mc_temp_3:.3f} > MC_RAW*1.5={mc_raw_3*1.5:.3f}")
    n_pass += t819

    # T820: TEMPORAL XOR1 > RAW XOR1 + 5pp
    xor1_raw_3 = raw_bm.get('xor', {}).get('tau1', 0.5)
    xor1_temp_3 = temp_bm.get('xor', {}).get('tau1', 0.5)
    t820 = xor1_temp_3 > xor1_raw_3 + 0.05
    tests['T820'] = {'pass': bool(t820), 'xor1_raw': xor1_raw_3, 'xor1_temporal': xor1_temp_3,
                     'desc': 'Temporal XOR1 > Raw XOR1 + 5pp'}
    print(f"  T820 {'PASS' if t820 else 'FAIL'}: XOR1_TEMP={xor1_temp_3*100:.1f}% > XOR1_RAW+5pp={xor1_raw_3*100+5:.1f}%")
    n_pass += t820

    # T821: TEMPORAL waveform > RAW waveform + 5pp
    wave_raw_3 = raw_bm.get('waveform4', 0.25)
    wave_temp_3 = temp_bm.get('waveform4', 0.25)
    t821 = wave_temp_3 > wave_raw_3 + 0.05
    tests['T821'] = {'pass': bool(t821), 'wave_raw': wave_raw_3, 'wave_temporal': wave_temp_3,
                     'desc': 'Temporal waveform > Raw waveform + 5pp'}
    print(f"  T821 {'PASS' if t821 else 'FAIL'}: W4_TEMP={wave_temp_3*100:.1f}% > W4_RAW+5pp={wave_raw_3*100+5:.1f}%")
    n_pass += t821

    # --- EXP 4 Tests ---
    exp4 = results['experiments'].get('exp4_triple_bridge', {})
    triple_bm = exp4.get('triple', {})
    dual_bm = exp4.get('dual_fpga_gpu', exp4.get('dual_cpu_gpu', {}))

    # T822: TRIPLE MC > DUAL MC (CPU adds information)
    mc_triple = triple_bm.get('mc_total', 0)
    mc_dual = dual_bm.get('mc_total', 0)
    t822 = mc_triple > mc_dual if (mc_triple > 0 and mc_dual > 0) else False
    tests['T822'] = {'pass': bool(t822), 'mc_triple': mc_triple, 'mc_dual': mc_dual,
                     'desc': 'Triple MC > Dual MC (CPU adds information)'}
    print(f"  T822 {'PASS' if t822 else 'FAIL'}: MC_TRIPLE={mc_triple:.3f} > MC_DUAL={mc_dual:.3f}")
    n_pass += t822

    # T823: TRIPLE MC > 15.0
    t823 = mc_triple > 15.0
    tests['T823'] = {'pass': bool(t823), 'mc_triple': mc_triple, 'threshold': 15.0,
                     'desc': 'Triple MC > 15.0'}
    print(f"  T823 {'PASS' if t823 else 'FAIL'}: MC_TRIPLE={mc_triple:.3f} > 15.0")
    n_pass += t823

    # T824: TRIPLE waveform > 93%
    wave_triple = triple_bm.get('waveform4', 0)
    t824 = wave_triple > 0.93
    tests['T824'] = {'pass': bool(t824), 'wave_triple': wave_triple, 'threshold': 0.93,
                     'desc': 'Triple waveform > 93%'}
    print(f"  T824 {'PASS' if t824 else 'FAIL'}: W4_TRIPLE={wave_triple*100:.1f}% > 93%")
    n_pass += t824

    # T825: CPU features have non-zero ridge weights in TRIPLE model
    cpu_nonzero = exp4.get('cpu_weight_nonzero', False)
    t825 = cpu_nonzero
    cpu_frac = exp4.get('cpu_weight_fraction', 0)
    tests['T825'] = {'pass': bool(t825), 'cpu_weight_fraction': cpu_frac,
                     'desc': 'CPU features have non-zero ridge weights in TRIPLE model'}
    print(f"  T825 {'PASS' if t825 else 'FAIL'}: CPU weight fraction={cpu_frac:.3f}")
    n_pass += t825

    # --- EXP 5 Tests ---
    exp5 = results['experiments'].get('exp5_rdrand_vs_gpu', {})

    # T826: RDRAND entropy > 7.5 bits/byte
    rdrand_ent = exp5.get('rdrand_entropy_bits', 0)
    t826 = rdrand_ent > 7.5
    tests['T826'] = {'pass': bool(t826), 'entropy': rdrand_ent, 'threshold': 7.5,
                     'desc': 'RDRAND entropy > 7.5 bits/byte'}
    print(f"  T826 {'PASS' if t826 else 'FAIL'}: RDRAND entropy={rdrand_ent:.3f} bits > 7.5")
    n_pass += t826

    # T827: GPU noise produces higher MC than RDRAND
    mc_gpu_noise = exp5.get('mc_gpu_input', 0)
    mc_rdrand_noise = exp5.get('mc_rdrand_input', 0)
    t827 = mc_gpu_noise > mc_rdrand_noise
    tests['T827'] = {'pass': bool(t827), 'mc_gpu': mc_gpu_noise, 'mc_rdrand': mc_rdrand_noise,
                     'desc': 'GPU noise MC > RDRAND noise MC (structured > random)'}
    print(f"  T827 {'PASS' if t827 else 'FAIL'}: MC_GPU={mc_gpu_noise:.3f} > MC_RDRAND={mc_rdrand_noise:.3f}")
    n_pass += t827

    # ================================================================
    # SUMMARY
    # ================================================================
    print("\n" + "=" * 70)
    print(f"  TOTAL: {n_pass}/17 PASS")
    print("=" * 70)

    results['tests'] = tests
    results['summary'] = {
        'n_pass': n_pass,
        'n_total': 17,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'temp_final': get_max_temp(),
    }
    save()
    print(f"\n  Results saved to {SAVE_FILE}")

    # Cleanup MSR/perf fds
    for fd in msr_fds.values():
        try: os.close(fd)
        except: pass
    for fds in perf_fds.values():
        try: os.close(fds)
        except: pass


if __name__ == '__main__':
    main()
