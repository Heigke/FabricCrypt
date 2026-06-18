"""z213_vth_isolate — isolate each Vth correction by zeroing params one at a time
in ngspice, then comparing to python port with same param zero'd.

Bias: VGS=1.2, VDS=2.5, VBS=0.  L=180n W=360n  T=27C.
"""
from __future__ import annotations
import re
import subprocess
import tempfile
import os
from pathlib import Path

import torch
from nsram.bsim4_port.model_card import BSIM4Model
from nsram.bsim4_port.geometry import Geometry
from nsram.bsim4_port.temp import compute_size_dep
from nsram.bsim4_port.dc import compute_dc

ROOT = Path(__file__).resolve().parents[1]
CARD = ROOT / "data" / "sebas_2026_04_22" / "PTM130bulkNSRAM.txt"
OUT_DIR = ROOT / "results" / "z213_vth_bisect"
OUT_DIR.mkdir(parents=True, exist_ok=True)

L = 180e-9
W = 360e-9
VGS, VDS, VBS = 1.2, 2.5, 0.0


# Read raw card body for inline embedding
def card_body() -> str:
    return CARD.read_text()


def ngspice_vth(extra_overrides: dict[str, float] | None = None) -> float:
    """Run ngspice and probe @m1[vth] with optional model overrides.
    Overrides are appended as additional + lines AFTER the .model NMOS block,
    relying on later definitions winning in ngspice param parsing -- but actually
    we instead REWRITE the card body, replacing param values inline.
    """
    body = card_body()
    if extra_overrides:
        for k, v in extra_overrides.items():
            # Replace `<key>=<oldval>` inside NMOS block (case-insensitive)
            # The card uses `+key= value`; substitute robustly
            # word-boundary, anywhere on line (not anchored)
            body = re.sub(rf"(?i)\b{re.escape(k)}\s*=\s*\S+",
                          f"{k}= {v}", body, count=1)

    deck = f"""* z213 isolate
.param Nparam = 1.58
.param Citparam = 0
.param Voffparam = -0.1368
.param K2Par = -0.070435
.param toxn = 4e-009
.param vsatn = 1.35e5
{body}
M1 d g s b NMOS L={L} W={W}
Vg g 0 {VGS}
Vd d 0 {VDS}
Vs s 0 0
Vb b 0 {VBS}
.option temp=27
.control
op
print @m1[vth]
.endc
.end
"""
    with tempfile.NamedTemporaryFile("w", suffix=".cir", delete=False) as f:
        f.write(deck)
        deck_path = f.name
    try:
        proc = subprocess.run(["ngspice", "-b", deck_path],
                              capture_output=True, text=True, timeout=30)
    finally:
        os.unlink(deck_path)
    out = proc.stdout
    m = re.search(r"@m1\[vth\]\s*=\s*([-\d.eE+]+)", out)
    if not m:
        print(out[-500:])
        return float("nan")
    return float(m.group(1))


def py_vth(extra_overrides: dict[str, float] | None = None) -> tuple[float, dict]:
    text = card_body()
    model = BSIM4Model.from_spice(text, model_type="nmos")
    if extra_overrides:
        for k, v in extra_overrides.items():
            model.set(k, v)
    geom = Geometry(L=L, W=W, NF=1)
    sd = compute_size_dep(model, geom, T_C=27.0)
    res = compute_dc(model, sd,
                     Vgs=torch.tensor(VGS, dtype=torch.float64),
                     Vds=torch.tensor(VDS, dtype=torch.float64),
                     Vbs=torch.tensor(VBS, dtype=torch.float64))
    return float(res.Vth.item()), {}


SCENARIOS = [
    ("baseline", {}),
    ("dvt0_zero", {"dvt0": 0.0}),                        # kills DVT short-L
    ("lpe0_zero", {"lpe0": 0.0}),                         # kills RSCE k1ox term
    ("kt1_zero",  {"kt1": 0.0, "kt1l": 0.0, "kt2": 0.0}), # kills temp term
    ("eta0_zero", {"eta0": 0.0}),                         # kills DIBL
    ("dvt0_lpe0_eta0", {"dvt0": 0.0, "lpe0": 0.0, "eta0": 0.0}),  # only body+vth0+temp
    ("all_off",  {"dvt0": 0.0, "lpe0": 0.0, "eta0": 0.0,
                  "kt1": 0.0, "kt1l": 0.0, "kt2": 0.0}),  # should ≈ vth0
]


def main():
    rows = []
    for name, ovr in SCENARIOS:
        ng = ngspice_vth(ovr)
        py, _ = py_vth(ovr)
        diff = (ng - py) * 1000.0
        rows.append((name, ng, py, diff, ovr))
        print(f"{name:25s}  ng={ng:.6f}  py={py:.6f}  diff={diff:+.2f} mV  ovr={ovr}")

    # Save
    lines = ["# Vth isolation table\n",
             "| Scenario | ngspice Vth | python Vth | gap (mV) | overrides |\n",
             "|---|---|---|---|---|\n"]
    for name, ng, py, diff, ovr in rows:
        lines.append(f"| {name} | {ng:.6f} | {py:.6f} | {diff:+.2f} | `{ovr}` |\n")
    (OUT_DIR / "isolation_table.md").write_text("".join(lines))


if __name__ == "__main__":
    main()


def sweep_test():
    """Sweep VGS, VDS, VBS to look for biases where Vth gap appears."""
    import itertools
    biases = list(itertools.product([0.0, 0.5, 1.0, 1.2, 1.5], [0.0, 1.0, 2.5], [0.0, -0.5, -1.0]))
    print(f"{'VGS':>6} {'VDS':>6} {'VBS':>6}  {'ng':>10}  {'py':>10}  {'gap_mV':>8}")
    global VGS, VDS, VBS
    for vg, vd, vb in biases:
        VGS, VDS, VBS = vg, vd, vb
        ng = ngspice_vth(None)
        py, _ = py_vth(None)
        gap = (ng - py) * 1000
        flag = " ***" if abs(gap) > 5 else ""
        print(f"{vg:6.2f} {vd:6.2f} {vb:6.2f}  {ng:10.6f}  {py:10.6f}  {gap:+8.2f}{flag}")

if __name__ == "__main__" and "SWEEP" in os.environ:
    sweep_test()
