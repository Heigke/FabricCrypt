"""M3c.1 gate test: η_lat=0 must reproduce F1.v2 exactly.

Runs forward_2t at one canonical bias with eta_lat=0 (default) and
eta_lat=1e-12 (effectively 0) and confirms Id matches to machine
precision. Then runs eta_lat=0.3 to confirm the path activates and
diverges (expected — full M3c.2 will route the lateral pair into Ic).

This is the regression gate from M3c plan checkpoint M3c.1.
"""
from __future__ import annotations
import importlib.util
from pathlib import Path
import torch
torch.set_default_dtype(torch.float64)

ROOT = Path(__file__).resolve().parent.parent
sp = importlib.util.spec_from_file_location("v1", ROOT / "scripts/z96_narma10_pilot.py")
v1 = importlib.util.module_from_spec(sp); sp.loader.exec_module(v1)

from nsram.bsim4_port.nsram_cell_2T import NSRAMCell2TConfig, forward_2t
from nsram.bsim4_port.bjt import GummelPoonNPN

M1, M2 = v1.build_calibrated_models()
bjt = GummelPoonNPN.from_sebas_card(); bjt.Bf = 100.0  # M3b honest

biases = [(0.6, 0.30), (0.4, 0.10), (0.2, 0.20)]

def run(eta_lat: float):
    cfg = NSRAMCell2TConfig(use_iii=True, use_gidl=True, use_bjt=True,
                             newton_max_iters=50)
    cfg.eta_lat = eta_lat
    rows = []
    Vd_seq = torch.tensor([1.5], dtype=torch.float64)
    for VG1, VG2 in biases:
        VG1_t = torch.tensor([VG1], dtype=torch.float64)
        VG2_t = torch.tensor([VG2], dtype=torch.float64)
        with torch.no_grad():
            out = forward_2t(
                cfg, model_M1=M1, model_M2=M2, bjt=bjt,
                VG1=VG1_t, VG2=VG2_t, Vd_seq=Vd_seq)
        rows.append({
            "VG1": VG1, "VG2": VG2,
            "Id": float(out["Id"][-1]),
            "Vb": float(out["Vb"][-1]),
            "Vsint": float(out["Vsint"][-1]),
            "Ib_lat_pair": float(out["Ib_lat_pair"][-1]) if "Ib_lat_pair" in out else float("nan"),
        })
    return rows

print(f"{'bias':>14s} {'Id_eta0':>14s} {'Id_eta_eps':>14s} {'reldiff':>10s} {'Id_eta0.3':>14s}")
print("-"*80)

r0 = run(0.0)
re = run(1e-12)
r3 = run(0.3)

max_reldiff = 0.0
for a, b, c in zip(r0, re, r3):
    bias = f"VG1={a['VG1']:.1f} VG2={a['VG2']:.2f}"
    rd = abs(a["Id"] - b["Id"]) / max(abs(a["Id"]), 1e-30)
    max_reldiff = max(max_reldiff, rd)
    print(f"{bias:>14s} {a['Id']:>14.6e} {b['Id']:>14.6e} {rd:>10.2e} {c['Id']:>14.6e}")

print()
if max_reldiff < 1e-12:
    print(f"PASS: η_lat=0 vs 1e-12 reldiff <{1e-12}: max={max_reldiff:.2e}")
else:
    print(f"FAIL: η_lat=0 vs 1e-12 reldiff exceeds 1e-12: max={max_reldiff:.2e}")

print()
print("Ib_lat_pair (should be ~0 at eta=0, nonzero at eta=0.3):")
for a, c in zip(r0, r3):
    print(f"  VG1={a['VG1']} VG2={a['VG2']}: eta=0 Ib_lat={a['Ib_lat_pair']:.3e}  eta=0.3 Ib_lat={c['Ib_lat_pair']:.3e}")
