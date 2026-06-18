"""Replay last N minutes of telemetry — useful after a crash to see what
the system was doing right before the gap.

Usage:
    python scripts/replay_telemetry.py            # last 5 min
    python scripts/replay_telemetry.py --minutes 30
    python scripts/replay_telemetry.py --gaps     # show jumps > 60s in
                                                     timestamps (= reboot)
"""
from __future__ import annotations
import argparse, json
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TELEM = ROOT / "results/telemetry"


def parse_ts(s):
    return datetime.fromisoformat(s)


def find_jsonl_files():
    return sorted(TELEM.glob("run_*.jsonl"))


def find_gaps(records, threshold_s=60):
    """Return list of (gap_start, gap_end, duration_s) for time jumps."""
    gaps = []
    for i in range(1, len(records)):
        try:
            t0 = parse_ts(records[i-1]["ts"])
            t1 = parse_ts(records[i]["ts"])
            dt = (t1 - t0).total_seconds()
            if dt > threshold_s:
                gaps.append((records[i-1]["ts"], records[i]["ts"], dt))
        except Exception:
            pass
    return gaps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=int, default=5)
    ap.add_argument("--gaps", action="store_true",
                    help="show timestamp gaps > 60s (likely reboots)")
    ap.add_argument("--all", action="store_true", help="ignore --minutes")
    args = ap.parse_args()

    files = find_jsonl_files()
    if not files:
        print("[replay] no telemetry files found")
        return
    print(f"[replay] {len(files)} jsonl files in {TELEM}")
    records = []
    for f in files:
        try:
            for line in f.read_text().splitlines():
                if line.strip():
                    records.append(json.loads(line))
        except Exception as e:
            print(f"[replay] {f.name}: {e}")
    print(f"[replay] {len(records)} total samples")

    if args.gaps:
        gaps = find_gaps(records)
        if not gaps:
            print("[replay] no timestamp gaps > 60s — uninterrupted record")
        else:
            print(f"[replay] {len(gaps)} gaps > 60s found:")
            for s, e, d in gaps:
                print(f"  {s}  →  {e}   ({d/60:.1f} min)")
                # Find the temp/load right before the gap
                for r in records:
                    if r["ts"] == s:
                        T = r.get("max_temp_C")
                        L = r.get("load", [None])[0]
                        top = r.get("top_procs", ["?"])[0]
                        print(f"    last sample: T={T}°C load={L} top={top}")
                        break
        return

    cutoff = datetime.now() - timedelta(minutes=args.minutes)
    if args.all:
        recent = records
    else:
        recent = [r for r in records if parse_ts(r["ts"]) >= cutoff]
    print(f"[replay] showing last {len(recent)} samples")
    print(f"{'time':>20s}  {'T_max':>5s}  {'load':>5s}  {'mem_GB':>7s}  top1")
    print("-"*80)
    for r in recent[-200:]:        # cap rows
        T = r.get("max_temp_C")
        L = r.get("load", [0])[0]
        M = r.get("mem", {}).get("MemAvail_GB", 0)
        top = r.get("top_procs", ["?"])[0][:50]
        flag = ""
        if T and T >= 92: flag = "  ⚠CRIT"
        elif T and T >= 85: flag = "  ⚠WARN"
        print(f"{r['ts']:>20s}  {T:>5.1f}  {L:>5.2f}  {M:>7.2f}  {top}{flag}")
    Tmax = max((r.get("max_temp_C", 0) for r in recent if r.get("max_temp_C")),
                default=0)
    print(f"\n[replay] peak T over window: {Tmax:.1f}°C")


if __name__ == "__main__":
    main()
