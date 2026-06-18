"""z351 — Flagship-bias broken-vs-clean ngspice comparison.

For VG1=0.6, VG2=0.20, Vd=2.0, run ngspice on:
  - BROKEN: data/sebas_2026_04_22/M1_130DNWFB.txt + M2_130bulkNSRAM.txt
  - CLEAN:  results/z351_clean_card/M1_130DNWFB_CLEAN.txt + M2_..._CLEAN.txt

Read V(vsint), V(vb), Ids (m1[id]), Isub (m1[isub]), total drain current.

Writes flagship_compare.json.
"""
from __future__ import annotations
import json, re, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT  = ROOT / "results/z351_clean_card"
OUT.mkdir(parents=True, exist_ok=True)

BROKEN_M1 = ROOT / "data/sebas_2026_04_22/M1_130DNWFB.txt"
BROKEN_M2 = ROOT / "data/sebas_2026_04_22/M2_130bulkNSRAM.txt"
CLEAN_M1  = OUT / "M1_130DNWFB_CLEAN.txt"
CLEAN_M2  = OUT / "M2_130bulkNSRAM_CLEAN.txt"

VG1, VG2, VD = 0.6, 0.20, 2.0
# Forced-node bias matching z340's peak-Vb operating point at this gate set:
# (Vd=2.000, Vsint=0.4091, Vb=0.2646). This is the kink/snapback regime where
# Isub/IIMOD physically matters.
VSINT_FORCED = 0.4090864
VB_FORCED    = 0.2645637

# z338 best parasitic NPN (matches z341 deck)
Z338_BEST = dict(
    Bf=2605.288,
    Va=0.3567,
    Is=3.2906845928467974e-10,
)


def make_deck(m1_path: Path, m2_path: Path) -> str:
    return f""".title z351 flagship broken/clean compare
.include "{m1_path}"
.include "{m2_path}"
.model parasiticBJT NPN(is={Z338_BEST['Is']:.4e} va={Z338_BEST['Va']:.4e}
+ bf={Z338_BEST['Bf']:.4e} br=100 nc=2 ikr=100m rc=0.1
+ vje=0.7 re=0.1 cjc=1e-15 fc=0.5 cje=0.7e-15 ne=1.5 ise=0 tr=20e-12 tf=25e-12
+ itf=0.03 vtf=7 xtf=2)
Vdd       vd     0    DC {VD}
Vg1       vg1    0    DC {VG1}
Vg2       vg2    0    DC {VG2}
Vsint_src vsint  0    DC {VSINT_FORCED}
Vb_src    vb     0    DC {VB_FORCED}
M1 vd vg1 vsint vb NMOSdnwfb L=0.13u W=1u
M2 vsint vg2 0 0 NMOS L=0.234u W=1u
Q1 vsint vb 0 parasiticBJT area=1u
.options gmin=1e-15 abstol=1e-14 reltol=1e-4 itl1=500
.control
op
echo "FLAGSHIP_BIAS"
print v(vd) v(vsint) v(vb)
print @m1[id] @m1[ibd] @m1[ibs] @m1[isub] @m1[igidl] @m1[igisl] @m1[igb]
print @m2[id] @m2[ibd] @m2[ibs] @m2[isub] @m2[igidl] @m2[igisl] @m2[igb]
print @q1[ic] @q1[ib] @q1[ie]
print -i(Vdd)
quit
.endc
.end
"""


def run_ngspice(deck: str) -> str:
    p = subprocess.run(
        ["ngspice", "-b"], input=deck, capture_output=True, text=True, timeout=60
    )
    return p.stdout + "\n--STDERR--\n" + p.stderr


_NUM = r"[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?"


def extract(name: str, out: str) -> float | None:
    # ngspice prints like:  @m1[id]               = 1.234e-05
    m = re.search(rf"{re.escape(name)}\s*=\s*({_NUM})", out)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def extract_node(node: str, out: str) -> float | None:
    # 'v(vsint)' prints as: v(vsint)              = 1.234e-01
    m = re.search(rf"v\({node}\)\s*=\s*({_NUM})", out, re.IGNORECASE)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return None


def harvest(out: str) -> dict:
    fields = {
        "Vd":      extract_node("vd", out),
        "Vsint":   extract_node("vsint", out),
        "Vb":      extract_node("vb", out),
        "M1_Id":   extract("@m1[id]", out),
        "M1_Ibd":  extract("@m1[ibd]", out),
        "M1_Ibs":  extract("@m1[ibs]", out),
        "M1_Isub": extract("@m1[isub]", out),
        "M1_Igidl": extract("@m1[igidl]", out),
        "M1_Igisl": extract("@m1[igisl]", out),
        "M1_Igb":  extract("@m1[igb]", out),
        "M2_Id":   extract("@m2[id]", out),
        "M2_Isub": extract("@m2[isub]", out),
        "Q1_Ic":   extract("@q1[ic]", out),
        "Q1_Ib":   extract("@q1[ib]", out),
        "Q1_Ie":   extract("@q1[ie]", out),
    }
    # total drain current = -i(Vdd)
    m_tot = re.search(rf"\bv#branch\s*=\s*({_NUM})", out)
    if m_tot:
        fields["Itotal"] = float(m_tot.group(1))
    return fields


def main():
    print(f"[flagship] VG1={VG1} VG2={VG2} Vd={VD}")

    print("[broken] running ngspice on Sebas's raw cards...")
    broken_out = run_ngspice(make_deck(BROKEN_M1, BROKEN_M2))
    print("[clean]  running ngspice on CLEAN cards...")
    clean_out  = run_ngspice(make_deck(CLEAN_M1, CLEAN_M2))

    (OUT / "broken_ngspice.log").write_text(broken_out)
    (OUT / "clean_ngspice.log").write_text(clean_out)

    broken = harvest(broken_out)
    clean  = harvest(clean_out)

    # Ratio (cleaned / broken) with safe div
    def ratio(c, b):
        if c is None or b is None:
            return None
        if b == 0:
            return float("inf") if c != 0 else 1.0
        return c / b

    keys = sorted(set(broken) | set(clean))
    ratios = {k: ratio(clean.get(k), broken.get(k)) for k in keys}

    cmp = {
        "bias": {"VG1": VG1, "VG2": VG2, "Vd": VD},
        "broken": broken,
        "clean":  clean,
        "ratio_clean_over_broken": ratios,
    }
    (OUT / "flagship_compare.json").write_text(json.dumps(cmp, indent=2))

    print()
    print(f"{'field':12s}  {'broken':>15s}  {'clean':>15s}  {'ratio':>12s}")
    for k in keys:
        b = broken.get(k); c = clean.get(k); r = ratios[k]
        bs = f"{b:.4e}" if isinstance(b, float) else "NA"
        cs = f"{c:.4e}" if isinstance(c, float) else "NA"
        rs = f"{r:.4g}" if isinstance(r, float) else "NA"
        print(f"{k:12s}  {bs:>15s}  {cs:>15s}  {rs:>12s}")


if __name__ == "__main__":
    main()
