"""z342 — R-26: BSIM4 IIMOD card sweep.

Find which knob (alpha0, alpha1, beta0, lalpha0 cancellation) activates
@m1[isub] in ngspice at VG1=0.60, VG2=0.20, Vd=2.0 V (the regime where
silicon shows Id ~4e-5 and pyport+ngspice both stuck at ~1e-11).

Sweep alpha0 across 4 decades from default 7.84e-5: {7.84e-5, 7.84e-3,
7.84e-1, 7.84}. ALSO sweep extra conditions:
 - default                       (baseline, expects isub=0)
 - lalpha0=0  (kill cancellation, eff alpha0 stays at 7.84e-5 instead of 2.7e-6)
 - alpha1=1e-6 m (add length-independent term)
 - beta0_lo=5 (lower the exponential cut-off)

For each condition: write a patched M1 card, drive the same 2T topology
(Vsint and Vb FREE — let ngspice solve), record:
 Vsint, Vb, @m1[id], @m1[isub], @q1[ic], total Id at drain.

Compare to silicon at (VG1=0.60, VG2=0.20, Vd=2.0):
 Id_meas ≈ 4.048e-5 A  (z341 verdict table row 9)
"""
from __future__ import annotations
import os, sys, json, re, subprocess, shutil
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/z342_iimod_card_sweep"
OUT.mkdir(parents=True, exist_ok=True)

M1_CARD_SRC = ROOT / "data/sebas_2026_04_22/M1_130DNWFB.txt"
M2_CARD     = ROOT / "data/sebas_2026_04_22/M2_130bulkNSRAM.txt"

# Z338 best BJT params (same as z341)
Z338 = dict(
    alpha0 = 1.63357328192734e-05,  # pyport-fit override (unused for ngspice here)
    Bf     = 2605.2882016162002,
    Va     = 0.3567358318716285,
    Is     = 3.2906845928467974e-10,
    lat_BV = 4.018266147002578,
    body_pdiode_Rs = 5480383.486345367,
)

VG1, VG2 = 0.60, 0.20
VD = 2.0
ID_SILICON = 4.048e-5  # z341 row 9

# Conditions:
CONDS = [
    # baseline default card
    dict(name="default",            alpha0=7.83756e-5, lalpha0=-9.843026e-12, alpha1=0,    beta0=19,  lbeta0=-9.5e-7),
    # remove the cancellation in alpha0
    dict(name="lalpha0_zero",       alpha0=7.83756e-5, lalpha0=0.0,           alpha1=0,    beta0=19,  lbeta0=-9.5e-7),
    # alpha0 +1 decade
    dict(name="alpha0_x10",         alpha0=7.83756e-4, lalpha0=0.0,           alpha1=0,    beta0=19,  lbeta0=-9.5e-7),
    # alpha0 +2 decade
    dict(name="alpha0_x100",        alpha0=7.83756e-3, lalpha0=0.0,           alpha1=0,    beta0=19,  lbeta0=-9.5e-7),
    # alpha0 +4 decade
    dict(name="alpha0_x1e4",        alpha0=7.83756e-1, lalpha0=0.0,           alpha1=0,    beta0=19,  lbeta0=-9.5e-7),
    # alpha0 +5 decade
    dict(name="alpha0_x1e5",        alpha0=7.84e+0,    lalpha0=0.0,           alpha1=0,    beta0=19,  lbeta0=-9.5e-7),
    # alpha1 non-zero (length-independent IIMOD)
    dict(name="alpha1_1em6",        alpha0=7.83756e-5, lalpha0=-9.843026e-12, alpha1=1e-6, beta0=19,  lbeta0=-9.5e-7),
    dict(name="alpha1_1em3",        alpha0=7.83756e-5, lalpha0=-9.843026e-12, alpha1=1e-3, beta0=19,  lbeta0=-9.5e-7),
    # lower beta0 (exponential cut-off easier to satisfy)
    dict(name="beta0_lo5",          alpha0=7.83756e-5, lalpha0=-9.843026e-12, alpha1=0,    beta0=5,   lbeta0=0.0),
    # combined hard punch
    dict(name="combo_hard",         alpha0=7.83756e-1, lalpha0=0.0,           alpha1=1e-3, beta0=5,   lbeta0=0.0),
]


def patch_m1_card(card_text: str, c: dict) -> str:
    # Replace the three lines 57-59 in the source card; do it via regex on
    # whole tokens so we don't break formatting.
    txt = card_text
    # alpha0
    txt = re.sub(r"alpha0\s*=\s*[^ \n]+", f"alpha0 = {c['alpha0']:.6e}", txt, count=1)
    # lalpha0
    txt = re.sub(r"lalpha0\s*=\s*[^ \n]+", f"lalpha0 = {c['lalpha0']:.6e}", txt, count=1)
    # alpha1
    txt = re.sub(r"alpha1\s*=\s*[^ \n]+", f"alpha1 = {c['alpha1']:.6e}", txt, count=1)
    # beta0
    txt = re.sub(r"(?<!l)beta0\s*=\s*[^ \n]+", f"beta0 = {c['beta0']:.6e}", txt, count=1)
    # lbeta0
    txt = re.sub(r"lbeta0\s*=\s*[^ \n]+", f"lbeta0 = {c['lbeta0']:.6e}", txt, count=1)
    return txt


def make_deck(m1_card_path: Path) -> str:
    return f""".title z342 IIMOD sweep VG1={VG1} VG2={VG2} Vd={VD}

.include "{m1_card_path}"
.include "{M2_CARD}"

.model parasiticBJT NPN(is={Z338['Is']:.4e} va={Z338['Va']:.4e}
+ bf={Z338['Bf']:.4e} br=100 nc=2 ikr=100m rc=0.1
+ vje=0.7 re=0.1 cjc=1e-15 fc=0.5 cje=0.7e-15 ne=1.5 ise=0 tr=20e-12 tf=25e-12
+ itf=0.03 vtf=7 xtf=2)

Vdd       vd     0    DC {VD}
Vg1       vg1    0    DC {VG1}
Vg2       vg2    0    DC {VG2}

* Vsint and Vb are FREE nodes (no voltage sources — let ngspice solve)
* Add tiny body-to-ground resistor so Vb has a DC path
Rbpath    vb     0    1e9

M1  vd vg1 vsint vb NMOSdnwfb L=0.13u W=1u
M2  vsint vg2 0  0  NMOS       L=0.234u W=1u
Q1  vsint vb 0    parasiticBJT area=1u

.options gmin=1e-15 abstol=1e-14 reltol=1e-4 itl1=500

.control
op
print v(vd) v(vg1) v(vg2) v(vsint) v(vb)
print @m1[id] @m1[ibd] @m1[ibs] @m1[isub] @m1[igidl]
print @m2[id]
print @q1[ic] @q1[ib]
quit
.endc

.end
"""


def parse_op(text: str) -> dict:
    out = {}
    # match key = value or key: value or "v(vd) = 2.0"
    for m in re.finditer(r"(@?\w+(?:\[\w+\])?|v\(\w+\))\s*=\s*([-+]?\d+\.?\d*[eE]?[-+]?\d*)", text):
        out[m.group(1).lower()] = float(m.group(2))
    return out


def run_one(c: dict) -> dict:
    cdir = OUT / c["name"]
    cdir.mkdir(exist_ok=True)
    # patch M1 card
    src = M1_CARD_SRC.read_text()
    patched = patch_m1_card(src, c)
    m1p = cdir / "M1_patched.txt"
    m1p.write_text(patched)
    deck = make_deck(m1p)
    deckp = cdir / "deck.sp"
    deckp.write_text(deck)
    # run ngspice
    logp = cdir / "ngspice.log"
    try:
        r = subprocess.run(["ngspice", "-b", str(deckp)],
                           capture_output=True, text=True, timeout=60)
        logp.write_text(r.stdout + "\n----STDERR----\n" + r.stderr)
        parsed = parse_op(r.stdout)
    except Exception as e:
        logp.write_text(f"ERROR: {e}")
        parsed = {}
    # extract key signals (with safe fallback)
    def g(k): return parsed.get(k, float("nan"))
    Vsint = g("v(vsint)")
    Vb    = g("v(vb)")
    Id_m1 = g("@m1[id]")
    Isub  = g("@m1[isub]")
    Ic_q1 = g("@q1[ic]")
    Id_m2 = g("@m2[id]")
    # Total Id at drain ≈ -@m1[id] (BSIM4 sign convention: @m1[id] is from drain to source positive when current flows in)
    # match measured: |Id_drain| ≈ |Id_m1|
    Id_total = abs(Id_m1) if not np.isnan(Id_m1) else float("nan")
    return dict(name=c["name"], cond=c,
                Vsint=Vsint, Vb=Vb, Id_m1=Id_m1, Isub=Isub,
                Ic_q1=Ic_q1, Id_m2=Id_m2, Id_total=Id_total)


def main():
    print(f"# z342 IIMOD card sweep @ VG1={VG1} VG2={VG2} Vd={VD}")
    print(f"# silicon target Id = {ID_SILICON:.3e}")
    rows = []
    for c in CONDS:
        r = run_one(c)
        rows.append(r)
        print(f"{c['name']:18s}  Vb={r['Vb']:.3f}  Vsint={r['Vsint']:.3f}  "
              f"Id={r['Id_total']:.3e}  Isub={r['Isub']:.3e}  Ic_Q1={r['Ic_q1']:.3e}")
    # save
    (OUT / "sweep_results.json").write_text(json.dumps(rows, indent=2, default=str))
    # markdown audit
    md_lines = []
    md_lines.append("# z342 IIMOD card audit + sweep")
    md_lines.append("")
    md_lines.append(f"Bias: VG1={VG1} VG2={VG2} Vd={VD} V")
    md_lines.append(f"Silicon measured Id @ this bias: {ID_SILICON:.3e} A")
    md_lines.append("")
    md_lines.append("## Card defaults (from data/sebas_2026_04_22/M1_130DNWFB.txt:57-59)")
    md_lines.append("```")
    md_lines.append("alpha0  = 7.83756e-5   (BSIM4 IIMOD strength coefficient)")
    md_lines.append("lalpha0 = -9.843026e-12 (length-dep cancels alpha0 at L=0.13u → eff 2.7e-6)")
    md_lines.append("alpha1  = 0            (length-independent IIMOD term — DEAD)")
    md_lines.append("beta0   = 19           (IIMOD exponent — large = hard to activate)")
    md_lines.append("lbeta0  = -9.5e-7      (length-dep → eff beta0 ≈ 11.69 at L=0.13u)")
    md_lines.append("```")
    md_lines.append("")
    md_lines.append("## BSIM4 IIMOD physics (v4.5)")
    md_lines.append("Iii = (alpha0 + alpha1·Leff)/Leff · (Vds-Vdsat) · exp(-beta0/(Vds-Vdsat)) · Ids")
    md_lines.append("At L=0.13u, Vds=2, Vdsat~0.5 → (Vds-Vdsat)=1.5, beta0_eff/Vdiff≈7.8 → exp≈4e-4")
    md_lines.append("With alpha0_eff=2.7e-6 (cancellation!), Iii/Ids ≈ 0.012 → Iii ≈ 1.5e-13 (too small to lift Vb).")
    md_lines.append("")
    md_lines.append("BSIM4.5 has NO explicit IIMOD level flag (added in v4.6+); IIMOD is always on if alpha0>0.")
    md_lines.append("No 'bjtoff' parameter exists in BSIM4 (NMOS model only has device-level body diodes).")
    md_lines.append("")
    md_lines.append("## Sweep results")
    md_lines.append("| condition | Vsint | Vb | @m1[id] | @m1[isub] | @q1[ic] | |Id_total| |")
    md_lines.append("|---|---|---|---|---|---|---|")
    for r in rows:
        md_lines.append(f"| {r['name']} | {r['Vsint']:.4f} | {r['Vb']:.4f} | "
                        f"{r['Id_m1']:.3e} | {r['Isub']:.3e} | {r['Ic_q1']:.3e} | {r['Id_total']:.3e} |")
    md_lines.append("")
    # findings
    md_lines.append("## Vb activation (Vb>0.5 V triggers Q1 properly)")
    found = [r for r in rows if r["Vb"] > 0.5]
    if found:
        for r in found:
            md_lines.append(f"- **{r['name']}**: Vb={r['Vb']:.3f} Id={r['Id_total']:.3e}")
    else:
        md_lines.append("- **NONE** of the card-level conditions lifted Vb above 0.5 V.")
        md_lines.append("- Even with alpha0×1e5, alpha1=1e-3, beta0=5 → IIMOD still insufficient.")
    md_lines.append("")
    md_lines.append("## Recommendation (one line)")
    if found:
        best = max(found, key=lambda x: x["Id_total"])
        md_lines.append(f"Apply `{best['name']}` (best Vb activation, Id={best['Id_total']:.3e} vs silicon {ID_SILICON:.3e}).")
    else:
        md_lines.append("Card-level IIMOD knobs INSUFFICIENT. Must inject Iii externally (CCCS from Ids→Vb) — pyport.")
    (OUT / "card_audit.md").write_text("\n".join(md_lines))

    # plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        names = [r["name"] for r in rows]
        Vbs   = [r["Vb"] for r in rows]
        Isubs = [abs(r["Isub"]) if not np.isnan(r["Isub"]) else 1e-30 for r in rows]
        Ids   = [r["Id_total"] if not np.isnan(r["Id_total"]) else 1e-30 for r in rows]
        fig, axs = plt.subplots(3, 1, figsize=(11, 9))
        x = np.arange(len(names))
        axs[0].bar(x, Vbs); axs[0].axhline(0.6, ls="--", c="r", label="Q1 turn-on")
        axs[0].set_ylabel("Vb [V]"); axs[0].set_title(f"z342 IIMOD sweep @ VG1={VG1} VG2={VG2} Vd={VD}")
        axs[0].set_xticks(x); axs[0].set_xticklabels(names, rotation=30, ha="right"); axs[0].legend()
        axs[1].bar(x, Isubs); axs[1].set_yscale("log")
        axs[1].set_ylabel("|@m1[isub]| [A]"); axs[1].set_xticks(x)
        axs[1].set_xticklabels(names, rotation=30, ha="right")
        axs[2].bar(x, Ids); axs[2].axhline(ID_SILICON, ls="--", c="r", label=f"silicon Id={ID_SILICON:.2e}")
        axs[2].set_yscale("log"); axs[2].set_ylabel("|Id_total| [A]"); axs[2].set_xticks(x)
        axs[2].set_xticklabels(names, rotation=30, ha="right"); axs[2].legend()
        plt.tight_layout()
        plt.savefig(OUT / "comparison_table.png", dpi=120)
    except Exception as e:
        print(f"plot failed: {e}")
    print(f"\nwrote: {OUT}/sweep_results.json")
    print(f"wrote: {OUT}/card_audit.md")


if __name__ == "__main__":
    main()
