"""z351 — pyport BSIM4 parser audit at flagship bias.

Loads M1 + M2 via nsram.bsim4_port.model_card.BSIM4Model.from_spice with the
cross-file .param scope. Prints the parsed values for the HSPICE-expression
parameters (rdsw, cgso, cgdo, cjs, cjsws, cjswgs, cgsl, cgdl) and compares
to the literal value those expressions evaluate to.
"""
from __future__ import annotations
import sys, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "nsram"))

from nsram.bsim4_port.model_card import BSIM4Model, parse_param_blocks

M1_PATH = ROOT / "data/sebas_2026_04_22/M1_130DNWFB.txt"
M2_PATH = ROOT / "data/sebas_2026_04_22/M2_130bulkNSRAM.txt"

m1_txt = M1_PATH.read_text()
m2_txt = M2_PATH.read_text()

# Match what production callers do: pass .param scope from M2 into M1
params = parse_param_blocks(m2_txt)
params = parse_param_blocks(m1_txt, params=params)
print("[scope] .param keys found:", sorted(params.keys()))
print()

m1 = BSIM4Model.from_spice(m1_txt, params=params, model_type="nmosdnwfb")
m2 = BSIM4Model.from_spice(m2_txt, params=params, model_type="nmos")

# Expected literal values (what they SHOULD be)
expected = {
    "rdsw":   100 - 140 * 1e6 * 1e-6 / int(1e-6 / 0.34e-6),  # = 30
    "cgso":   1.0 * 3.65e-10,
    "cgdo":   1.0 * 3.65e-10,
    "cgsl":   1.0 * 2.98e-11,
    "cgdl":   1.0 * 2.98e-11,
    "cjs":    1.0 * 0.0016995,
    "cjsws":  1.0 * 2.9299e-11,
    "cjswgs": 1.0 * 2.677e-10,
}

results = {"M1": {}, "M2": {}}
print(f"{'param':10s} {'expected':>14s} {'M1.value':>14s} {'M2.value':>14s} {'M1.given':>10s} {'M2.given':>10s}")
for k, v_exp in expected.items():
    v1 = m1.get(k); g1 = m1.is_given(k)
    v2 = m2.get(k); g2 = m2.is_given(k)
    print(f"{k:10s} {v_exp:>14.6g} {(v1 if v1 is not None else float('nan')):>14.6g} "
          f"{(v2 if v2 is not None else float('nan')):>14.6g} {str(g1):>10s} {str(g2):>10s}")
    results["M1"][k] = {"expected": v_exp, "parsed": v1, "given": g1,
                        "match": (v1 is not None and abs(v1 - v_exp)/(abs(v_exp)+1e-30) < 1e-6)}
    results["M2"][k] = {"expected": v_exp, "parsed": v2, "given": g2,
                        "match": (v2 is not None and abs(v2 - v_exp)/(abs(v_exp)+1e-30) < 1e-6)}

print()
print("Reference params from .param scope:")
for k in ("rcgon", "rcjn", "rcjswn", "rcjswgn", "toxn", "vth0n", "vsatn", "lintn", "wintn", "lpe0n", "k3n", "pvth0n"):
    print(f"  {k:10s} = {params.get(k)!r}")

# Top params that are *passed through identifier reference* without expression
results["scope"] = {k: params[k] for k in params}
out = ROOT / "results/z351_clean_card/pyport_audit.json"
out.write_text(json.dumps(results, indent=2, default=lambda x: None if x is None else x))
print(f"\n[ok] wrote {out}")
