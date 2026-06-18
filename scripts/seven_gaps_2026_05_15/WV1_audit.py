"""Track 7 / WV1: Multi-cell measurement-data audit.

Scan data/sebas_*, docs/, nsram/ for IV-data files identifiable as multi-cell:
  - filename device tags
  - column headers containing device_id / cell_id / wafer_id
  - directory structures with multiple subdirectories per (VG1, VG2) bias

Output: results/WV1_audit/summary.json
"""
from __future__ import annotations
import json, os, re, csv
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "WV1_audit"
OUT.mkdir(parents=True, exist_ok=True)


DEVICE_RE = re.compile(r"StandardIV_HH_([A-Za-z0-9\-]+)_")
MULTI_KEYS = ("device_id", "cell_id", "wafer_id", "die_id", "site_id", "wafer", "die_")
# Note: bare "device" excluded — SPICE instance names (M1/M2) are not wafer-level identifiers.


def scan_files() -> dict:
    roots = [REPO / "data", REPO / "docs", REPO / "nsram"]
    csv_files: list[Path] = []
    for r in roots:
        if r.exists():
            for p in r.rglob("*.csv"):
                csv_files.append(p)
    devs_in_names: set[str] = set()
    multi_col_files: list[dict] = []
    for p in csv_files:
        m = DEVICE_RE.search(p.name)
        if m:
            devs_in_names.add(m.group(1))
        # check first row of header for multi-cell columns
        try:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                head = f.readline().strip().lower()
            if any(k in head for k in MULTI_KEYS):
                multi_col_files.append({"path": str(p), "header_preview": head[:200]})
        except Exception:
            pass
    return {
        "n_csv_total": len(csv_files),
        "devices_in_filenames": sorted(devs_in_names),
        "n_unique_devices": len(devs_in_names),
        "multi_cell_columns_in_csv": multi_col_files,
    }


def main() -> None:
    info = scan_files()
    info["verdict"] = "MULTI_DEVICE" if info["n_unique_devices"] > 1 or info["multi_cell_columns_in_csv"] else "SINGLE_DEVICE"
    info["path_for_WV"] = "WV2 (moment-match measured)" if info["verdict"] == "MULTI_DEVICE" else "WV3 (synthetic sigma sweep)"
    with open(OUT / "summary.json", "w") as f:
        json.dump(info, f, indent=2)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
