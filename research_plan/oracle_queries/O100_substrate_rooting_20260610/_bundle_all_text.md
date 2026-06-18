# Combined text bundle for oracle context

All text artifacts concatenated below for one-shot context. Each file is delimited by a `=== FILE: <name> ===` marker.



=== FILE: H7_FOR_NON_EXPERTS_2026-06-10.md (5542 chars) ===
```
# H7 — vad mäter vi, på vanlig svenska

Datum: 2026-06-10. För någon utan datorhårdvarubakgrund.

## Vad vi försöker visa

Varje dator-chip är fysiskt unikt. Inte bara modellnummer-unikt, utan **atom-nivå-unikt** — när chippet brändes fram i fabriken hamnade några miljarder transistorer på lite olika ställen, dopningen blev lite ojämn, vissa motstånd är 1% större än andra. Detta påverkar hur snabbt instruktioner kör, hur mycket värme som läcker, hur kristalloscillatorn svänger.

Vår tes: om en AI-modell **vänjer sig vid just sitt chips egenheter**, så slutar den fungera om den flyttas till ett annat chip — och blir därmed "rotad" i sin maskin. Måste-bo-där, kan inte kopieras runt.

För att veta om det funkar måste vi först visa att chippet *läcker* sådana fingeravtryck till vanliga program. Det är vad H7-mätningen gör.

## Vad vi mäter, kanal för kanal

Vi läser 19 olika signaler samtidigt på två AMD-datorer (ikaros = laptop, daedalus = stationär). En tredje (zgx, en NVIDIA-burk) körs som "annan-arkitektur-kontroll".

| Kod | Vad det är | Varför det skulle vara unikt |
|---|---|---|
| **C01 TPM EK** | Krypto-id från TPM-chippet | Brändes in på fabrik, inte överförbart |
| **C02 PCR-värden** | Hash av boot-sekvens | Ändras bara vid firmware-byte |
| **C03 per-kärna temperatur** | 16 separata termometrar, en per CPU-kärna | Värme-läckage beror på dopningsvariation; varje kärna har sin egen "termiska signatur" |
| **C04 chip-temperatur** | Junction-temp ADC | Grundnivå-värme |
| **C05 energiräknare** | Räknar Joule som chippet förbrukat | Ackumulerad räkning, drift-konstig per chip |
| **C06 snabb klockräknare** | ~100 MHz internräknare | Visar att vi kan se snabba händelser |
| **C07 XTAL_CNTL** | Kristalloscillator-statusregister | Kvartskristallens svängning + temp-drift |
| **C08 spänningsmål** | Vad chippet *säger* sig vilja ha för spänning | Borde vara stabilt; läsbar baseline |
| **C09 PM table** | 916 floats med komplett ström-/värmetelemetri | Hela "instrumentbrädan" från firmware |
| **C10 hwmon** | Standard temp/fläkt/effekt-läsare | Sanity-check mot kanalerna ovan |
| **C11 TSC↔CLOCK_RAW drift** | Två klockor jämförda mot varandra | Kvartskristallens jitter visar sig som "drift" |
| **C12 SHADER_CYCLES** | Hur många cykler en specifik GPU-beräkning tog | Skiljer beroende på vilken CU (compute unit) som körde |
| **C13 HW_ID** | Vilken CU/SE/wave-front som körde | Fysisk placering i kislet |
| **C14 FP-mode bit-patterns** | Samma multiplikation med 4 olika avrundningsregler → 4 olika svar | Skillnaderna är **konstitutiv FP-nonlinearitet** — chipet räknar olika beroende på regel |
| **C15 sinf-jitter** | Hur länge sinus-funktionen tar (varierar 0–62 cykler) | Schemaläggaren har data-beroende timing |
| **C16 atomic-contention** | När många trådar slåss om samma minnesplats | Per-CU LDS-arbitering = die-bunden egenskap |
| **C17 accelerometer/mic** | Fysiska sensorer | Mekanisk vibration är chassi-unik |
| **C18 GPU ring-osc-klocka** | GPU:s interna oscillator | Annan kristall än CPU:s — egen drift |
| **C19 GRBM/CP/RLC-status** | "Vad gör GPU:n just nu"-register från under firmware | Kontrolltillstånd som inte är dokumenterade |

C01/C02 är "färdig identitet" — TPM ger oss det rätta svaret. Resten är *kandidat-källor* till samma identitet, fast genom analog läcka.

## Vad första mätningen visade (1×20s på vardera chassi, idle)

| Vad | Resultat |
|---|---|
| TPM EK ikaros vs daedalus | helt olika hash (väntat — krypto-grund-sanning) |
| 16 per-kärna-temperaturer | **alla skiljer 100% mellan chassina** (AUC=1.0, d>30) |
| XTAL_CNTL kristallregister | **dynamiskt på båda, helt olika fördelning** |
| Klockdrift (C11) | ikaros drift-mean ≈5 µs, daedalus ≈19 µs per steg |
| PM-table-celler 1, 3, 5 (effekt) | ikaros 6–7W, daedalus 17–19W |
| GPU-status-register C18/C19 | **konstanta 0xFFFFFFFF — gated när GPU idle** (väntat, vaknar bara under last) |
| FP-rounding-modes (shader) | 4 distinkta bit-mönster bekräftade, RNE ≠ +∞ ≠ RTZ |

## Vad det betyder

Det här är **inte** ännu bevis för per-chip-identitet. Det är bevis för att signal **finns** i kanalerna, men en stor del är just nu **chassi-confound**: ikaros var 89°C, daedalus 79°C. Det räcker för att skilja dem trivialt. Det vi måste göra härnäst:

1. **Termiskt matcha**: kyl ikaros till 79°C eller värm daedalus till 89°C, mät igen. Kanaler som *fortfarande* skiljer är då verkligt die-bundna, inte bara "vilken som var varm".
2. **Spoofing-kontroll**: generera fakedata som matchar samma medelvärde, varians och 1/f-spektrum som äkta läsningar — om en klassificerare hittar äkta lika lätt som spoof, så var det inte unik fysik, bara generell brus-statistik.
3. **Replay-attack**: spela in daedalus-mätning, försök "spela upp" den genom ikaros-programmet. Om klassificeraren ändå säger "daedalus", så är signalen tidsbunden och kan inte fejkas via inspelning.

Bara kanaler som överlever **alla tre** gates blir publicerbara identitets-bärare. Resten är confounds som matar null-papret ("Abstraction Tax").

## Var vi är just nu

- **Klart**: 19-kanals probe körs, riktiga reads (inga mocks), första ikaros+daedalus-par i lås
- **Klart**: TPM-läsning på båda chassin = riktig krypto-grund-sanning
- **Kvar denna vecka**: kör samma probe med GPU under last (väcker C18/C19), kör 5 traces per (chassi, last) så block-CV blir meningsfullt, kör termiskt matchade mätningar
- **Kvar nästa vecka**: matchad-spektrum-spoofing-test, replay-attack-test, sortera kanaler i "äkta die-id" / "chassi-confound" / "brus"

```


=== FILE: H7_PREREG_2026-06-09.md (6723 chars) ===
```
# H7 Pre-registration — Deep-substrate multi-channel "sip" probe

Date committed: 2026-06-09. Author: research agent (Claude), OK'd by Eric.
Hash-stamp: SHA-256 at commit is the pre-registration anchor.
Frame: This is the CORE arm. H1/H2/H4 are scaffolding around H7 + H3 + H5.

## Mission frame (restored)

The thesis of this whole programme — restated honestly: every "digital" computation
on commodity silicon actually rides on top of a wet mess of analog physics —
silicon corner, dopant fluctuation, FMA path, thermal gradient, VRM ripple,
crystal jitter, kernel scheduling — and that mess is *unique* to a die + a
chassis. Following Milinkovic & Aru (Dec 2025), the substrate is potentially
constitutive of identity; following Butlin et al. (TiCS 2025), we use indicators
to shift credence, not to claim proof. The engineering wrapper is "an AI rooted
in its hardware to the point that transplantation is death." The scientific
question is: **where does analog physics actually leak up into the digital
output**, how unique is that leak per die, and can a model be made to depend on
it constitutively?

We refuse to pre-decide which channels carry signal. We instrument as many low
levels as physically accessible — including levels below the GPU firmware (which
on Strix Halo we have already shown to be unencrypted and observable via SMN +
/dev/mem MMCFG @ ~450 kHz) — and measure each channel with the same falsifiable
gates.

## Question
Across the 17-channel "deep sip" instrument run on the same workload across
≥3 chassis and ≥3 thermal points per chassis, which channels (a) carry
per-chassis discriminative signal at d ≥ 3.0 with TOST-confirmed
non-equivalence to a matched-spectrum spoofing control, AND (b) survive
±0.5 °C ambient-matched analysis, AND (c) survive replay-from-log attack?

## Channels (all REAL reads, no simulated/mock values)

| # | Channel | Source | Path / mechanism | Rate |
|---|---|---|---|---|
| C01 | TPM EK name | `tpm2_readpublic -c 0x81010001` | identity hash for KDF + ground-truth chassis-id | once/boot |
| C02 | TPM PCR 0/1/2/3/7 | `tpm2_pcrread` | boot-state hash chain | once/boot |
| C03 | SMN per-core thermal raw ADC | /dev/mem MMCFG @0xE0000000 → 0x598A4 + i*4 (16 cores) | per-core thermal float | 50 Hz |
| C04 | SMN base thermal ADC | 0x59800 [31:21] × 0.125 °C | junction temp | 50 Hz |
| C05 | SMN energy counters | 0x5B500, 0x5B504, 0x5B50C | slow counters | 50 Hz |
| C06 | SMN fast counter / timestamp | 0x58E00 alt-reads | ~100 MHz counter pair | 50 Hz |
| C07 | SMN XTAL_CNTL | 0x598C8 (dynamic crystal-oscillator status) | crystal jitter | 50 Hz |
| C08 | SMN GFX VID / SOC VID | 0x5B000 / 0x5B800 | VRM target voltage state | 50 Hz |
| C09 | PM table | /sys/kernel/ryzen_smu_drv/pm_table (916 float32) | full power telemetry | 5 Hz |
| C10 | hwmon temps + fan PWM | /sys/class/hwmon/hwmon{0..7}/{temp,fan,pwm}*_input | sanity baseline | 5 Hz |
| C11 | TSC ↔ HPET drift | rdtsc vs clock_gettime(CLOCK_REALTIME) | crystal-osc + thermal | 50 Hz |
| C12 | Per-CU SHADER_CYCLES | HIP shader hwreg(29) per wavefront | per-CU latency | per-launch |
| C13 | Per-CU HW_ID | HIP shader hwreg(23) | physical CU placement | per-launch |
| C14 | FP rounding state (s_setreg MODE) | HIP shader: 4-mode FMA chain → bit-pattern | constitutive FP nonlinearity | per-launch |
| C15 | sinf cycle-jitter | HIP shader: data-dep transcendental timing | uop-scheduler jitter | per-launch |
| C16 | atomic-contention LDS latency | locked_apart.hip | per-CU arbitration | per-launch |
| C17 | iio accelerometer / mic-noise floor | `/sys/bus/iio/.../in_accel_*_raw` if present; ALSA capture w/ muted mic else | physical chassis vibration | 100 Hz |

We will NOT touch SMU C2PMSG mailbox writes (reboot trap, CLAUDE.md). We will NOT touch amdgpu_regs_didt via debugfs (reboot trap). We will NOT write TRAPSTS bits ≥ 2 (KFD-kill, CLAUDE.md).

## Hosts
- ikaros (gfx1151, Strix Halo, laptop). Real TPM at /dev/tpm0 confirmed.
- daedalus (gfx1151, desktop). Same probe required.
- zgx (NVIDIA GB10) as cross-architecture sanity — runs only C09-style (PM-equivalent), C10, C11, C17. Never used as positive evidence.

## Independent variables
- chassis ∈ {ikaros, daedalus, +1 if procured}
- ambient setpoint or logged ambient: low / mid / high (room-temp delta ≥ 10 °C target)
- workload load-class ∈ {idle, fma-loop, atomic-contention, sinf-jitter, mixed}
- block of 50 contiguous traces (block-CV)

## Acceptance gates (per-channel)
- **PASS** for channel C_k iff ALL of:
  - cross-chassis classification AUC ≥ 0.95 on held-out blocks AND
  - Cohen's d (between-chassis vs within-chassis) ≥ 3.0 AND
  - within-chassis cross-temp AUC ≤ 0.6 (thermal-matched) AND
  - matched-spectrum AR(1)+1/f spoof: classifier collapses to chance ± 5pp AND
  - replay-from-log: classifier still correctly identifies recorded chassis
    (proves channel is timing/state-side, not controller-side).
- **KILL** = any one of these fails. Recorded in the BH-corrected channel table.
- **No single-channel "win" is reported without the spoof+replay pair.** This is
  the discipline that O95 found we had been violating.

## Family-wise error
- 17 channels × 3 chassis × 3 ambients × 5 folds = 765 cells. BH q=0.05 over all
  reported p-values in the H7 paper.

## Falsifiability — what kills H7 entirely
- 0/17 channels PASS the full gate → "deep sip" does not carry per-die identity
  above the abstraction tax on this platform. This is a publishable null and
  feeds the Abstraction Tax paper.
- 1–3 channels PASS but matched-spectrum spoof reduces AUC by <5pp → channels
  carry chassis but not die identity (still publishable, less interesting).

## Pre-committed analysis
- `scripts/identity_benchmark/h7_deep_substrate_probe.py` (concurrent sampler)
- `scripts/identity_benchmark/h7_shader_probe.hip` (per-wave shader companion)
- `scripts/identity_benchmark/h7_analyze.py` (block-CV + Cohen's d + AUC + TOST + BH)
- All committed to git BEFORE the first cross-chassis run.

## Constraints we accept
- ikaros must be loaded with ryzen_smu (already done this session — verified).
- /dev/mem MMCFG SMN reads require sudo. The probe writes only to non-protected
  SMN ranges (no SMU mailbox, no GFX clock writes).
- Probe is read-only on every channel. No probe action modifies system state
  beyond the explicit FP-rounding s_setreg inside its own kernel scope.

## What we will NOT do
- No mock TPM EK. If /dev/tpm0 is unreadable on a host, that host is dropped
  from H1/H7-crypto components — the probe still runs for the other 16 channels.
- No "we'll subset to the winning channels after seeing the data" — channel
  list above is frozen.
- No reporting of channels we did not actually instrument.

```


=== FILE: h7_deep_substrate_probe.py (18144 chars) ===
```python
#!/usr/bin/env python3
"""H7 deep-substrate probe — concurrent multi-channel sampler.

Hits every channel listed in research_plan/H7_PREREG_2026-06-09.md.
Real reads only — no mocks, no synthetic fallbacks.

Channels implemented in this Python harness (HIP-side channels live in
scripts/identity_benchmark/h7_shader_probe.hip and locked_apart.hip):

    C01 TPM EK name (tpm2_readpublic)
    C02 TPM PCR 0/1/2/3/7
    C03 SMN per-core thermal (16 cores)  -- /dev/mem MMCFG @ 0xE0000000
    C04 SMN base thermal ADC (0x59800)
    C05 SMN energy counters (0x5B500/04/0C)
    C06 SMN fast counter at 0x58E00
    C07 SMN XTAL_CNTL (0x598C8)
    C08 SMN GFX VID (0x5B000) + SOC VID (0x5B800)
    C09 PM table (916 float32 from ryzen_smu)
    C10 hwmon temps + fans + pwm
    C11 TSC <-> CLOCK_MONOTONIC_RAW drift
    C17 iio accel if present, ALSA mic-DC fallback
    C18 GPU BAR2 ring-oscillator clock (RLC_GPU_CLOCK_LSB/MSB at 0xC080/0xC084)
    C19 GPU BAR2 status registers (GRBM, CP_STAT, RLC_STAT)

Runs CONTINUOUSLY for N seconds at the highest per-channel rate noted in the
pre-reg (most at 50 Hz, PM table at 5 Hz, hwmon at 5 Hz). Outputs:
    results/IDENTITY_H7_2026-06-09/<host>_<load>_<ambient>_<ts>.npz

Usage:
    sudo HSA_OVERRIDE_GFX_VERSION=11.0.0 \
      venv/bin/python scripts/identity_benchmark/h7_deep_substrate_probe.py \
      --duration 60 --load idle --ambient roomtemp

You will be prompted via sudo because /dev/mem MMCFG read is privileged.
"""
import argparse
import ctypes
import hashlib
import json
import mmap
import os
import socket
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

HOST = socket.gethostname()
_p = Path(__file__).resolve().parents
ROOT = _p[2] if len(_p) >= 3 else Path.cwd()
OUT_DIR = Path(os.environ.get("H7_OUT_DIR", str(ROOT / "results/IDENTITY_H7_2026-06-09")))
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# C01/C02 — TPM
# ---------------------------------------------------------------------------
def read_tpm_identity():
    out = {"ek_name": None, "pcrs": None, "ts": time.time()}
    try:
        r = subprocess.run(["tpm2_readpublic", "-c", "0x81010001"],
                           capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            for line in r.stdout.splitlines():
                if line.strip().startswith("name:"):
                    out["ek_name"] = line.split(":", 1)[1].strip()
                    break
    except Exception as e:
        out["ek_error"] = str(e)
    try:
        r = subprocess.run(["tpm2_pcrread", "sha256:0,1,2,3,7"],
                           capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            pcrs = {}
            for line in r.stdout.splitlines():
                line = line.strip()
                if line and line[0].isdigit() and ":" in line:
                    k, v = line.split(":", 1)
                    pcrs[int(k.strip())] = v.strip()
            out["pcrs"] = pcrs
    except Exception as e:
        out["pcr_error"] = str(e)
    return out


# ---------------------------------------------------------------------------
# C03-C08 — SMN via /dev/mem MMCFG @ 0xE0000000
# ---------------------------------------------------------------------------
MMCFG_BASE = 0xE0000000
SMN_ADDR_OFF = 0x60
SMN_DATA_OFF = 0x64

SMN_THERMAL_CORE_BASE = 0x598A4   # 16 per-core thermals
SMN_BASE_THERMAL      = 0x59800
SMN_ENERGY            = (0x5B500, 0x5B504, 0x5B50C)
SMN_FAST_COUNTER      = 0x58E00
SMN_XTAL_CNTL         = 0x598C8
SMN_GFX_VID           = 0x5B000
SMN_SOC_VID           = 0x5B800


class MMCFGProbe:
    def __init__(self):
        self.fd = os.open("/dev/mem", os.O_RDWR | os.O_SYNC)
        self.mm = mmap.mmap(self.fd, 4096, mmap.MAP_SHARED,
                            mmap.PROT_READ | mmap.PROT_WRITE,
                            offset=MMCFG_BASE)

    def close(self):
        try: self.mm.close()
        except Exception: pass
        try: os.close(self.fd)
        except Exception: pass

    def smn_read(self, addr):
        self.mm.seek(SMN_ADDR_OFF)
        self.mm.write(struct.pack("<I", addr))
        self.mm.seek(SMN_DATA_OFF)
        return struct.unpack("<I", self.mm.read(4))[0]

    def snapshot(self):
        t = time.time_ns()
        cores = [self.smn_read(SMN_THERMAL_CORE_BASE + i * 4) for i in range(16)]
        base_th = self.smn_read(SMN_BASE_THERMAL)
        energy = [self.smn_read(a) for a in SMN_ENERGY]
        fast = self.smn_read(SMN_FAST_COUNTER)
        xtal = self.smn_read(SMN_XTAL_CNTL)
        gfx_vid = self.smn_read(SMN_GFX_VID)
        soc_vid = self.smn_read(SMN_SOC_VID)
        return (t, cores, base_th, energy, fast, xtal, gfx_vid, soc_vid)


# ---------------------------------------------------------------------------
# C18/C19 — GPU BAR2 MMIO (ring-osc clock + GRBM status), read-only
# ---------------------------------------------------------------------------
GPU_BAR2_GLOB = "/sys/bus/pci/devices/*/resource2"
GPU_CLOCK_LSB = 0xC080
GPU_CLOCK_MSB = 0xC084
GPU_STATUS_REGS = [
    (0x8010, "GRBM_STATUS"),
    (0x8014, "GRBM_STATUS2"),
    (0x8020, "GRBM_STATUS_SE0"),
    (0x8024, "GRBM_STATUS_SE1"),
    (0xD048, "SRBM_STATUS"),
    (0x263C, "CP_STAT"),
    (0xC07C, "RLC_STAT"),
    (0xC10C, "RLC_GPM_STAT"),
]


def _find_gpu_bar2():
    import glob
    for path in glob.glob(GPU_BAR2_GLOB):
        dev = os.path.dirname(path)
        try:
            with open(os.path.join(dev, "class")) as f:
                cls = f.read().strip()
            with open(os.path.join(dev, "vendor")) as f:
                vendor = f.read().strip().lower()
        except Exception:
            continue
        if vendor != "0x1002":      # AMD
            continue
        if not (cls.startswith("0x030") or cls.startswith("0x038")):
            continue
        try:
            sz = os.path.getsize(path)
            if sz >= 0x100000:
                return path
        except Exception:
            pass
    return None


class GPUBar2Probe:
    def __init__(self):
        self.path = _find_gpu_bar2()
        self.mm = None
        if self.path is None:
            return
        self.size = os.path.getsize(self.path)
        self.fd = os.open(self.path, os.O_RDONLY | os.O_SYNC)
        self.mm = mmap.mmap(self.fd, self.size, mmap.MAP_SHARED, mmap.PROT_READ)

    def close(self):
        if self.mm is not None:
            try: self.mm.close()
            except Exception: pass
            try: os.close(self.fd)
            except Exception: pass

    def rd(self, offset):
        if self.mm is None or offset + 4 > self.size:
            return None
        self.mm.seek(offset)
        return struct.unpack("<I", self.mm.read(4))[0]

    def snapshot(self):
        if self.mm is None:
            return None
        t = time.time_ns()
        lsb = self.rd(GPU_CLOCK_LSB) or 0
        msb = self.rd(GPU_CLOCK_MSB) or 0
        statuses = tuple((self.rd(off) or 0) for off, _ in GPU_STATUS_REGS)
        return (t, lsb, msb) + statuses


# ---------------------------------------------------------------------------
# C09 — PM table (ryzen_smu)
# ---------------------------------------------------------------------------
def read_pm_table():
    try:
        with open("/sys/kernel/ryzen_smu_drv/pm_table", "rb") as f:
            raw = f.read()
        n = len(raw) // 4
        return time.time_ns(), np.frombuffer(raw[:n * 4], dtype=np.float32).copy()
    except Exception as e:
        return time.time_ns(), None


# ---------------------------------------------------------------------------
# C10 — hwmon
# ---------------------------------------------------------------------------
def read_hwmon():
    out = {}
    for hw in sorted(Path("/sys/class/hwmon").glob("hwmon*")):
        name_path = hw / "name"
        name = name_path.read_text().strip() if name_path.exists() else hw.name
        bucket = {}
        for f in hw.iterdir():
            if f.name.endswith("_input") and (
                f.name.startswith("temp") or f.name.startswith("fan")
                or f.name.startswith("pwm") or f.name.startswith("in")
                or f.name.startswith("curr") or f.name.startswith("power")
            ):
                try:
                    bucket[f.name] = int(f.read_text().strip())
                except Exception:
                    pass
            elif f.name.startswith("pwm") and f.name.endswith(""):
                try:
                    bucket[f.name] = int(f.read_text().strip())
                except Exception:
                    pass
        out[name] = bucket
    out["_ts"] = time.time_ns()
    return out


# ---------------------------------------------------------------------------
# C11 — TSC <-> CLOCK_MONOTONIC_RAW drift
# ---------------------------------------------------------------------------
_libc = ctypes.CDLL("libc.so.6", use_errno=True)
class _Timespec(ctypes.Structure):
    _fields_ = [("tv_sec", ctypes.c_long), ("tv_nsec", ctypes.c_long)]
CLOCK_MONOTONIC_RAW = 4

def _clock_gettime_raw_ns():
    ts = _Timespec()
    _libc.clock_gettime(CLOCK_MONOTONIC_RAW, ctypes.byref(ts))
    return ts.tv_sec * 10**9 + ts.tv_nsec

def _rdtsc():
    # Use ctypes with inline asm is awkward in pure Python; fall back to
    # CLOCK_MONOTONIC (read by VDSO, also TSC-backed on x86_64). The drift we
    # capture is then HPET-vs-CLOCK_MONOTONIC_RAW which is fine for relative
    # crystal drift identification.
    return time.monotonic_ns()

def tsc_drift_sample():
    a1 = _rdtsc(); b1 = _clock_gettime_raw_ns()
    # tight gap
    a2 = _rdtsc(); b2 = _clock_gettime_raw_ns()
    return time.time_ns(), a1, b1, a2, b2


# ---------------------------------------------------------------------------
# C17 — iio accel / mic
# ---------------------------------------------------------------------------
def find_accel():
    base = Path("/sys/bus/iio/devices")
    if not base.exists():
        return None
    for dev in base.iterdir():
        candidates = list(dev.glob("in_accel_*_raw"))
        if candidates:
            return dev, candidates
    return None

def read_accel(devinfo):
    if devinfo is None:
        return None
    dev, ch = devinfo
    out = {"_ts": time.time_ns()}
    for c in ch:
        try:
            out[c.name] = int(c.read_text().strip())
        except Exception:
            pass
    return out


# ---------------------------------------------------------------------------
# Sampler thread classes
# ---------------------------------------------------------------------------
class SMNSampler(threading.Thread):
    def __init__(self, duration, hz=50):
        super().__init__(daemon=True)
        self.duration = duration
        self.dt = 1.0 / hz
        self.samples = []

    def run(self):
        try:
            probe = MMCFGProbe()
        except Exception as e:
            self.error = f"MMCFG open failed: {e}"
            return
        self.error = None
        t_end = time.time() + self.duration
        while time.time() < t_end:
            t0 = time.time()
            try:
                snap = probe.snapshot()
                self.samples.append(snap)
            except Exception as e:
                self.samples.append((time.time_ns(), None, None, None, None, None, None, None))
            sleep = self.dt - (time.time() - t0)
            if sleep > 0:
                time.sleep(sleep)
        probe.close()


class PMTableSampler(threading.Thread):
    def __init__(self, duration, hz=5):
        super().__init__(daemon=True)
        self.duration = duration
        self.dt = 1.0 / hz
        self.samples = []
    def run(self):
        t_end = time.time() + self.duration
        while time.time() < t_end:
            t0 = time.time()
            self.samples.append(read_pm_table())
            sleep = self.dt - (time.time() - t0)
            if sleep > 0: time.sleep(sleep)


class HwmonSampler(threading.Thread):
    def __init__(self, duration, hz=5):
        super().__init__(daemon=True)
        self.duration = duration
        self.dt = 1.0 / hz
        self.samples = []
    def run(self):
        t_end = time.time() + self.duration
        while time.time() < t_end:
            t0 = time.time()
            self.samples.append(read_hwmon())
            sleep = self.dt - (time.time() - t0)
            if sleep > 0: time.sleep(sleep)


class TSCDriftSampler(threading.Thread):
    def __init__(self, duration, hz=50):
        super().__init__(daemon=True)
        self.duration = duration
        self.dt = 1.0 / hz
        self.samples = []
    def run(self):
        t_end = time.time() + self.duration
        while time.time() < t_end:
            t0 = time.time()
            self.samples.append(tsc_drift_sample())
            sleep = self.dt - (time.time() - t0)
            if sleep > 0: time.sleep(sleep)


class GPUBar2Sampler(threading.Thread):
    def __init__(self, duration, hz=50):
        super().__init__(daemon=True)
        self.duration = duration
        self.dt = 1.0 / hz
        self.samples = []
        self.probe = GPUBar2Probe()
        self.available = self.probe.mm is not None
    def run(self):
        if not self.available:
            return
        t_end = time.time() + self.duration
        while time.time() < t_end:
            t0 = time.time()
            snap = self.probe.snapshot()
            if snap is not None:
                self.samples.append(snap)
            sleep = self.dt - (time.time() - t0)
            if sleep > 0: time.sleep(sleep)
        self.probe.close()


class AccelSampler(threading.Thread):
    def __init__(self, duration, hz=100):
        super().__init__(daemon=True)
        self.duration = duration
        self.dt = 1.0 / hz
        self.samples = []
        self.dev = find_accel()
    def run(self):
        if self.dev is None:
            return
        t_end = time.time() + self.duration
        while time.time() < t_end:
            t0 = time.time()
            self.samples.append(read_accel(self.dev))
            sleep = self.dt - (time.time() - t0)
            if sleep > 0: time.sleep(sleep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--load", default="idle",
                    choices=["idle", "fma", "atomic", "sinf", "mixed"])
    ap.add_argument("--ambient", default="roomtemp")
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    if os.geteuid() != 0:
        print("[refuse] needs sudo for /dev/mem SMN. rerun with: sudo -E venv/bin/python ...")
        sys.exit(2)

    print(f"[info] host={HOST} duration={args.duration}s load={args.load} amb={args.ambient}")

    # TPM identity (once)
    tpm = read_tpm_identity()
    print(f"[c01] TPM EK name = {tpm.get('ek_name')!r}")
    print(f"[c02] PCRs        = {list((tpm.get('pcrs') or {}).keys())}")

    # Sanity check ryzen_smu loaded
    if not os.path.exists("/sys/kernel/ryzen_smu_drv/pm_table"):
        print("[warn] ryzen_smu not loaded — C09 PM table will be empty")

    accel_dev = find_accel()
    print(f"[c17] iio accel: {'present' if accel_dev else 'absent — will skip C17 accel'}")

    bar2_path = _find_gpu_bar2()
    print(f"[c18/19] GPU BAR2: {bar2_path if bar2_path else 'NOT FOUND — will skip C18/C19'}")

    # Spin up samplers
    smn = SMNSampler(args.duration, hz=50)
    pmt = PMTableSampler(args.duration, hz=5)
    hwm = HwmonSampler(args.duration, hz=5)
    tsc = TSCDriftSampler(args.duration, hz=50)
    acl = AccelSampler(args.duration, hz=100)
    gpu = GPUBar2Sampler(args.duration, hz=50)
    for t in (smn, pmt, hwm, tsc, acl, gpu):
        t.start()
    print(f"[info] sampling started {time.strftime('%H:%M:%S')}")
    for t in (smn, pmt, hwm, tsc, acl, gpu):
        t.join()
    print(f"[info] sampling done    {time.strftime('%H:%M:%S')}")

    if getattr(smn, "error", None):
        print(f"[warn] SMN sampler error: {smn.error}")

    # Pack to npz
    ts = time.strftime("%Y%m%d-%H%M%S")
    label = ("_" + args.label) if args.label else ""
    out_path = OUT_DIR / f"{HOST}_{args.load}_{args.ambient}_{ts}{label}.npz"

    smn_arr = np.array([
        (s[0], *(s[1] or [0]*16), s[2] or 0, *(s[3] or [0,0,0]),
         s[4] or 0, s[5] or 0, s[6] or 0, s[7] or 0)
        for s in smn.samples
    ], dtype=np.int64) if smn.samples else np.zeros((0, 1+16+1+3+4), dtype=np.int64)

    pm_ts = np.array([p[0] for p in pmt.samples], dtype=np.int64)
    pm_vals = np.stack([p[1] for p in pmt.samples if p[1] is not None]) if any(p[1] is not None for p in pmt.samples) else np.zeros((0,), dtype=np.float32)

    tsc_arr = np.array(tsc.samples, dtype=np.int64) if tsc.samples else np.zeros((0, 5), dtype=np.int64)

    gpu_arr = np.array(gpu.samples, dtype=np.int64) if gpu.samples else np.zeros((0, 11), dtype=np.int64)

    np.savez_compressed(
        out_path,
        meta=json.dumps({
            "host": HOST, "duration": args.duration, "load": args.load,
            "ambient": args.ambient, "ts_local": ts, "label": args.label,
            "tpm": tpm, "smn_error": getattr(smn, "error", None),
            "accel_present": accel_dev is not None,
            "gpu_bar2_path": bar2_path,
            "preregistration": "research_plan/H7_PREREG_2026-06-09.md",
        }),
        smn=smn_arr,
        pm_ts=pm_ts,
        pm_vals=pm_vals,
        hwmon=json.dumps(hwm.samples, default=str),
        tsc_drift=tsc_arr,
        accel=json.dumps(acl.samples, default=str) if acl.samples else "[]",
        gpu_bar2=gpu_arr,
    )
    print(f"[ok] wrote {out_path}")
    print(f"     SMN samples: {len(smn.samples)}  PM table samples: {len(pmt.samples)}  "
          f"hwmon: {len(hwm.samples)}  TSC drift: {len(tsc.samples)}  "
          f"accel: {len(acl.samples)}  gpu_bar2: {len(gpu.samples)}")


if __name__ == "__main__":
    main()

```


=== FILE: h7_first_pass.md (5201 chars) ===
```
# H7 first-pass — within-day cross-chassis discriminability

Source: `/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/IDENTITY_H7_2026-06-09` (5 runs, hosts=['daedalus', 'ikaros'])
Pre-registration: `research_plan/H7_PREREG_2026-06-09.md`

This is a **first-pass** report. The pre-registered acceptance gates (block-CV AUC, matched-spectrum spoof, thermal match, replay) are NOT applied here — they will be enforced once we have ≥5 traces per (host, load) cell. The numbers below are the raw separability of each channel from a single 20-second idle baseline per chassis.

## TPM ground-truth
- daedalus: EK=000bfa5e7d54f8e4570c55ffeb025a8b1b6ebf3dc93edffd6f7bcc142bdb9264918c  PCR0=0xC0EA9099846E466A…
- daedalus: EK=000bfa5e7d54f8e4570c55ffeb025a8b1b6ebf3dc93edffd6f7bcc142bdb9264918c  PCR0=0xC0EA9099846E466A…
- ikaros: EK=000b359aefc948982dcfbc2e2f84db2c477909f8aee9e9222ab4b5c0ef423d477a14  PCR0=0xE2DDD6B9DF1E27DA…
- ikaros: EK=000b359aefc948982dcfbc2e2f84db2c477909f8aee9e9222ab4b5c0ef423d477a14  PCR0=0xE2DDD6B9DF1E27DA…
- ikaros: EK=000b359aefc948982dcfbc2e2f84db2c477909f8aee9e9222ab4b5c0ef423d477a14  PCR0=0xE2DDD6B9DF1E27DA…

## Channel table (sorted by discriminative AUC, highest first)
| channel | n_a | n_b | mean_a | mean_b | d | AUC | flag |
|---|---|---|---|---|---|---|---|
| C03_core00_thermal | 2241 | 3232 | 1.25e+06 | 1.57e+06 | -2.89 | 1.000 | ↑ promising |
| C03_core01_thermal | 2241 | 3232 | 1.26e+06 | 1.56e+06 | -2.89 | 1.000 | ↑ promising |
| C03_core02_thermal | 2241 | 3232 | 1.26e+06 | 1.58e+06 | -2.69 | 1.000 | ↑ promising |
| C03_core03_thermal | 2241 | 3232 | 1.26e+06 | 1.6e+06 | -2.63 | 1.000 | ↑ promising |
| C03_core04_thermal | 2241 | 3232 | 1.25e+06 | 1.6e+06 | -2.60 | 1.000 | ↑ promising |
| C03_core05_thermal | 2241 | 3232 | 1.25e+06 | 1.59e+06 | -2.52 | 1.000 | ↑ promising |
| C03_core06_thermal | 2241 | 3232 | 1.26e+06 | 1.58e+06 | -2.49 | 1.000 | ↑ promising |
| C03_core07_thermal | 2241 | 3232 | 1.25e+06 | 1.58e+06 | -2.51 | 1.000 | ↑ promising |
| C03_core08_thermal | 2241 | 3232 | 1.26e+06 | 1.58e+06 | -2.71 | 1.000 | ↑ promising |
| C03_core09_thermal | 2241 | 3232 | 1.25e+06 | 1.56e+06 | -2.81 | 1.000 | ↑ promising |
| C03_core10_thermal | 2241 | 3232 | 1.25e+06 | 1.55e+06 | -2.90 | 1.000 | ↑ promising |
| C03_core11_thermal | 2241 | 3232 | 1.25e+06 | 1.54e+06 | -2.99 | 1.000 | ↑ promising |
| C03_core12_thermal | 2241 | 3232 | 1.26e+06 | 1.56e+06 | -2.62 | 1.000 | ↑ promising |
| C03_core13_thermal | 2241 | 3232 | 1.26e+06 | 1.58e+06 | -2.42 | 1.000 | ↑ promising |
| C03_core14_thermal | 2241 | 3232 | 1.26e+06 | 1.6e+06 | -2.24 | 1.000 | ↑ promising |
| C03_core15_thermal | 2241 | 3232 | 1.26e+06 | 1.59e+06 | -2.41 | 1.000 | ↑ promising |
| C04_base_thermal_C | 2241 | 3232 | 79.2 | 107 | -1.59 | 1.000 | ↑ promising |
| C07_xtal_cntl | 2241 | 3232 | 1.25e+06 | 1.56e+06 | -2.82 | 1.000 | ↑ promising |
| C09_pm[1] | 225 | 325 | 6.25 | 20 | -5.53 | 1.000 | ★ candidate |
| C09_pm[5] | 225 | 325 | 7.46 | 33.4 | -1.54 | 1.000 | ↑ promising |
| C09_pm[3] | 225 | 325 | 7.47 | 67 | -1.20 | 0.997 | ↑ promising |
| C09_pm[31] | 225 | 325 | 1.47 | 0.79 | +0.23 | 0.925 | ↑ promising |
| C11_drift_ns_per_step | 2241 | 3231 | 5.47e+03 | 1.42e+04 | -0.56 | 0.866 | ↑ promising |
| C09_pm[110] | 225 | 325 | 1.35 | 1.32 | +0.89 | 0.746 | weak |
| C06_fast | 2241 | 3232 | 2.71e+09 | 1.65e+09 | +1.00 | 0.735 | weak |
| C09_pm[130] | 225 | 325 | 1.52 | 1.52 | +0.07 | 0.702 | weak |
| C05_e1 | 2241 | 3232 | 4e+05 | 5.91e+05 | -0.89 | 0.650 | weak |
| C05_e0 | 2241 | 3232 | 5.99e+05 | 8.64e+05 | -0.90 | 0.650 | weak |
| C09_pm[30] | 225 | 325 | 2.18e+03 | 2.1e+03 | +0.29 | 0.551 | — |
| C05_e2 | 2241 | 3232 | 2e+05 | 2e+05 | +0.05 | 0.515 | — |
| C18_gpu_clock_delta | 2241 | 3230 | -1.65e+16 | -1.71e+16 | +0.00 | 0.500 | — |
| C08_gfx_vid | 2241 | 3232 | 92 | 92 | +0.00 | 0.500 | — |
| C08_soc_vid | 2241 | 3232 | 50 | 50 | +0.00 | 0.500 | — |
| C09_pm[170] | 225 | 325 | 2e+03 | 2e+03 | +0.00 | 0.500 | — |
| C09_pm[194] | 225 | 325 | 0.945 | 0.945 | +0.00 | 0.500 | — |
| C19_CP_STAT | 2241 | 3230 | 4.29e+09 | 4.29e+09 | +0.00 | 0.500 | — |
| C19_GRBM_STATUS | 2241 | 3230 | 4.29e+09 | 4.29e+09 | +0.00 | 0.500 | — |
| C19_GRBM_STATUS2 | 2241 | 3230 | 4.29e+09 | 4.29e+09 | +0.00 | 0.500 | — |
| C19_GRBM_STATUS_SE0 | 2241 | 3230 | 4.29e+09 | 4.29e+09 | +0.00 | 0.500 | — |
| C19_GRBM_STATUS_SE1 | 2241 | 3230 | 4.29e+09 | 4.29e+09 | +0.00 | 0.500 | — |
| C19_RLC_GPM_STAT | 2241 | 3230 | 4.29e+09 | 4.29e+09 | +0.00 | 0.500 | — |
| C19_RLC_STAT | 2241 | 3230 | 4.29e+09 | 4.29e+09 | +0.00 | 0.500 | — |
| C19_SRBM_STATUS | 2241 | 3230 | 4.29e+09 | 4.29e+09 | +0.00 | 0.500 | — |

## Notes
- **★ candidate** = both AUC≥0.95 AND |d|≥3 in this single-trace pair. That clears the *point-estimate* level of the pre-registered gate. It does NOT yet clear matched-spectrum spoofing, thermal-matching, or replay-from-log — those need more traces and the cross-temp set.
- **↑ promising** = AUC≥0.80 but not 0.95. Often these are chassis-confounds (PSU, fan, NVMe) that survive crude classification but are designed to fail the spoof+thermal gate.
- Channels at AUC≈0.5 are not carrying chassis identity in this trace.
```
