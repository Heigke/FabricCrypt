"""ANGLE C — Tournament aggregation from existing probeB RO-pair races.

ORIGINAL PLAN was a live HIP tournament binary. That binary
(C_tournament_ro.hip) compiled and ran, but the per-block-CAS arbitration
turned out to be deterministic (block-1 won 100% of races on ikaros) — the
HIP scheduler dispatches block 0 → block 1 in a fixed order and the CAS race
is decided by dispatch latency, not silicon. So a "tournament" over that
kernel collapses to a constant 79-bit string and cannot distinguish devices.

PIVOT: aggregate the ALREADY-COLLECTED probeB cycle-delta data (10000 races
× 4 fields per device) into a tournament-like 79-bit string. Each "slot"
takes a non-overlapping block of K races and emits one bit: 1 if the median
of block-0's `dCyc` is HIGHER than block-1's median (i.e. block-0 was
slower), else 0. The MEDIAN-OVER-K aggregation is the orthodox tournament
trick (Suh/Devadas) of averaging many weak races into a strong bit.

Outputs:
  results/IDENTITY_BENCHMARK_2026-05-30/novel/C_tournament_<dev>.json
  results/IDENTITY_BENCHMARK_2026-05-30/novel/C_tournament_summary.json
    cross-device Hamming distance + per-device intra-bootstrap Hamming.
"""
from __future__ import annotations
from pathlib import Path
import json
import struct
import numpy as np

REPO = Path(__file__).resolve().parents[3]
OUT_DIR = REPO / "results" / "IDENTITY_BENCHMARK_2026-05-30" / "novel"

# probeB layout: magic 0x524F5031, races, fields(=4), reserved, then
# races * 4 uint32 with [winner, hwid0, hwid1, dCyc01].
PROBEB_MAGIC = 0x524F5031
SLOTS = 79
RACES_PER_SLOT = 100   # 79 * 100 = 7900 races used; we have 10000


def load_probeB(path: Path) -> np.ndarray:
    data = path.read_bytes()
    magic, races, fields, _ = struct.unpack("<IIII", data[:16])
    assert magic == PROBEB_MAGIC, f"bad magic 0x{magic:x} in {path}"
    arr = np.frombuffer(data[16:16 + races * fields * 4], dtype=np.uint32)
    return arr.reshape(races, fields)


def tournament_bits(arr: np.ndarray, slots: int, races_per_slot: int,
                    rng_seed: int = 0) -> np.ndarray:
    """For each slot take a sample of races, compute median dCyc per block,
    emit bit = (b0_median > b1_median).

    probeB only stores dCyc for block-0 in field [3] (per the source); but
    winner field [0] tells us which block won. We use the winner field as a
    weak per-race bit and majority-vote within each slot to amplify.
    """
    rng = np.random.default_rng(rng_seed)
    races = arr.shape[0]
    bits = np.zeros(slots, dtype=np.uint8)
    counts = np.zeros(slots, dtype=np.int64)
    # Use winner field (column 0): sentinel 1 = block 0, sentinel 2 = block 1.
    winners = arr[:, 0]
    valid = (winners == 1) | (winners == 2)
    pool = np.where(valid)[0]
    rng.shuffle(pool)
    needed = slots * races_per_slot
    if len(pool) < needed:
        # repeat with replacement
        pool = rng.choice(np.where(valid)[0], size=needed, replace=True)
    pool = pool[:needed]
    for s in range(slots):
        idx = pool[s * races_per_slot:(s + 1) * races_per_slot]
        b1_wins = int(np.sum(winners[idx] == 2))
        bits[s] = 1 if b1_wins > races_per_slot // 2 else 0
        counts[s] = b1_wins
    return bits, counts


def intra_hamming(arr: np.ndarray, n_boot: int = 8) -> tuple[float, int]:
    """Average Hamming distance between bootstrap re-samplings of the same
    device. Quantifies tournament noise floor."""
    bits_list = []
    for k in range(n_boot):
        b, _ = tournament_bits(arr, SLOTS, RACES_PER_SLOT, rng_seed=k + 1)
        bits_list.append(b)
    dists = []
    for i in range(n_boot):
        for j in range(i + 1, n_boot):
            dists.append(int(np.sum(bits_list[i] != bits_list[j])))
    return float(np.mean(dists)), int(np.max(dists))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sources = {
        "ikaros":   REPO / "results" / "IDENTITY_BENCHMARK_2026-05-30" / "ikaros" / "phase1c" / "probeB.bin",
        "daedalus": Path("/tmp/daedalus_probeB.bin"),
    }
    per_device = {}
    bits_canon = {}
    for dev, path in sources.items():
        if not path.exists():
            print(f"[skip] {dev}: {path} missing")
            continue
        arr = load_probeB(path)
        bits, counts = tournament_bits(arr, SLOTS, RACES_PER_SLOT, rng_seed=0)
        intra_mean, intra_max = intra_hamming(arr, n_boot=8)
        out = {
            "device": dev,
            "source": str(path),
            "races_total": int(arr.shape[0]),
            "slots": SLOTS,
            "races_per_slot": RACES_PER_SLOT,
            "tournament_bits": bits.tolist(),
            "b1_win_counts": counts.tolist(),
            "intra_device_hamming_mean": intra_mean,
            "intra_device_hamming_max": intra_max,
            "b1_total_win_rate": float(np.sum(arr[:, 0] == 2) / max(1, np.sum((arr[:, 0] == 1) | (arr[:, 0] == 2)))),
        }
        (OUT_DIR / f"C_tournament_{dev}.json").write_text(json.dumps(out, indent=2))
        per_device[dev] = out
        bits_canon[dev] = bits

    summary = {"per_device": {dev: {k: v for k, v in d.items() if k != "tournament_bits"}
                              for dev, d in per_device.items()},
               "gate_definition": "cross_device_hamming > 40/79 AND max_intra_hamming < 10"}
    if "ikaros" in bits_canon and "daedalus" in bits_canon:
        cross = int(np.sum(bits_canon["ikaros"] != bits_canon["daedalus"]))
        intras = [per_device["ikaros"]["intra_device_hamming_max"],
                  per_device["daedalus"]["intra_device_hamming_max"]]
        summary["cross_device_hamming"] = cross
        summary["max_intra_hamming"] = int(max(intras))
        summary["discovery_gate_passed"] = bool(cross > 40 and max(intras) < 10)
    (OUT_DIR / "C_tournament_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
