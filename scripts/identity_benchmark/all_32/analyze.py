#!/usr/bin/env python3
"""Cross-host analyzer for ALL-32 identity campaign.

For each mechanism: extract a scalar feature per sample (typically cycle count),
compute Cohen's d between ikaros and daedalus, and within/between variance ratio.

Pre-registered discovery gate:
    d >= 3.0  AND  within_std / between_std <= 1/3   (i.e. between >= 3*within)
"""
import json, math
from pathlib import Path
import statistics

REPO = Path(__file__).resolve().parent.parent.parent.parent
LOCAL_DIR = REPO / "results" / "IDENTITY_BENCHMARK_2026-05-30" / "all_32"
DAEDALUS_DIR = REPO / "results" / "IDENTITY_BENCHMARK_2026-05-30" / "all_32" / "daedalus_pulled"

def cohen_d(a, b):
    if len(a) < 2 or len(b) < 2: return float("nan")
    ma, mb = statistics.mean(a), statistics.mean(b)
    sa, sb = statistics.pstdev(a), statistics.pstdev(b)
    pooled = math.sqrt((sa*sa + sb*sb) / 2)
    if pooled <= 0: return float("inf") if ma != mb else 0.0
    return (ma - mb) / pooled

def extract_cycles(rec_obj):
    """ISA probe -> list of cycle counts from samples[]. Drops counter-underflow
    artifacts: any value above 1e10 cycles is clearly bogus (a 1.4 GHz clock for
    1 second is only 1.4e9, our bursts are <1s)."""
    s = rec_obj.get("samples")
    if not isinstance(s, list): return []
    out = []
    for item in s:
        if isinstance(item, dict) and item.get("ok") and "cyc" in item:
            c = item["cyc"]
            if 0 < c < 1e10:
                out.append(c)
    return out

def extract_wall_us(rec_obj):
    s = rec_obj.get("samples")
    if not isinstance(s, list): return []
    out = []
    for item in s:
        if isinstance(item, dict) and item.get("ok") and "wall_us" in item:
            out.append(item["wall_us"])
    return out

def extract_payload_hashes(rec_obj):
    """For M18/M19 (residual leakage): payload hex distinct count."""
    s = rec_obj.get("samples")
    if not isinstance(s, list): return []
    out = []
    for item in s:
        if isinstance(item, dict) and item.get("ok") and "payload_hex" in item:
            out.append(item["payload_hex"])
    return out


def load_pair(mech: str, ikaros_dir: Path, daedalus_dir: Path):
    fi = ikaros_dir / f"{mech}_ikaros.json"
    fd = daedalus_dir / f"{mech}_daedalus.json"
    if not (fi.exists() and fd.exists()):
        return None, None
    return json.loads(fi.read_text()), json.loads(fd.read_text())


ISA = ["M2","M3","M4","M5","M6","M7","M9","M10","M11","M17",
       "M18","M19","M20","M22","M23","M24"]
NON_ISA = ["M15","M27","M28","M29","M31"]


def analyze(ikaros_dir: Path, daedalus_dir: Path) -> dict:
    table = []
    for m in ISA + NON_ISA:
        rec_i, rec_d = load_pair(m, ikaros_dir, daedalus_dir)
        if rec_i is None or rec_d is None:
            table.append({"mech": m, "skipped": "missing"})
            continue
        if "skip" in rec_i or "skip" in rec_d:
            table.append({"mech": m, "skipped": rec_i.get("skip") or rec_d.get("skip")})
            continue
        if m in ISA:
            xi = extract_cycles(rec_i); xd = extract_cycles(rec_d)
            if len(xi) < 3 or len(xd) < 3:
                # fall back to wall_us
                xi = extract_wall_us(rec_i); xd = extract_wall_us(rec_d)
            if len(xi) < 3 or len(xd) < 3:
                table.append({"mech": m, "skipped": "insufficient_ok"})
                continue
            d = cohen_d(xi, xd)
            wi = statistics.pstdev(xi); wd = statistics.pstdev(xd)
            within = max(wi, wd, 1e-9)
            between = abs(statistics.mean(xi) - statistics.mean(xd))
            ratio = within / between if between > 0 else float("inf")
            gate = (abs(d) >= 3.0) and (ratio <= 1/3)
            table.append({"mech": m, "n_i": len(xi), "n_d": len(xd),
                          "mean_i": statistics.mean(xi),
                          "mean_d": statistics.mean(xd),
                          "std_i": wi, "std_d": wd,
                          "cohen_d": d, "within_over_between": ratio,
                          "gate_pass": gate, "feature": "cycles"})
            # also payload-uniqueness (cross-device)
            phi = extract_payload_hashes(rec_i); phd = extract_payload_hashes(rec_d)
            if phi and phd:
                # is the set of unique payloads disjoint?
                disjoint = bool(set(phi).isdisjoint(set(phd)))
                table[-1]["payload_disjoint"] = disjoint
                table[-1]["payload_unique_i"] = len(set(phi))
                table[-1]["payload_unique_d"] = len(set(phd))
        else:
            # non-ISA — record summary only
            s_i = rec_i.get("samples", [])
            s_d = rec_d.get("samples", [])
            row = {"mech": m, "n_i_samples": len(s_i), "n_d_samples": len(s_d),
                   "skip_i": rec_i.get("skip"), "skip_d": rec_d.get("skip"),
                   "feature": "actuator-trace"}
            # if there are numeric traces (freq, temp), compute trace-mean delta
            def traces(rec, key):
                return [x[key] for x in rec.get("samples", []) if isinstance(x, dict) and key in x and x[key] is not None and x[key] >= 0]
            for key in ("gpu_freq_MHz", "power_W", "freq_MHz", "temp_C"):
                ti = traces(rec_i, key); td = traces(rec_d, key)
                if len(ti) > 3 and len(td) > 3:
                    row[f"{key}_d"] = cohen_d(ti, td)
                    row[f"{key}_mean_i"] = statistics.mean(ti)
                    row[f"{key}_mean_d"] = statistics.mean(td)
            table.append(row)
    return {"table": table}


if __name__ == "__main__":
    import argparse, sys
    ap = argparse.ArgumentParser()
    ap.add_argument("--ikaros-dir", default=str(LOCAL_DIR))
    ap.add_argument("--daedalus-dir", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    res = analyze(Path(args.ikaros_dir), Path(args.daedalus_dir))
    out_s = json.dumps(res, indent=2, default=str)
    if args.out:
        Path(args.out).write_text(out_s)
        print(f"wrote {args.out}")
    print(out_s)
