"""H7 SPATIAL transfer verdict — is the zone×sensor PDN coupling matrix DIE-SPECIFIC? (matched-temp).

Compares the spatial coupling matrix M[zone,channel] across runs, using ONLY steps inside a COMMON matched
temperature band (the control that the cross-die test lacked). Die-specific iff INTRA-die similarity
(ikaros_r1 vs ikaros_r2, same drive repeated) >> INTER-die similarity (ikaros vs daedalus).
Reports BOTH raw A_uv coupling and temperature-compensated C_uv (regularized to avoid /0 artifacts), as
cosine over the flattened matrix and as per-zone pattern correlation. Pre-registered die-specific:
mean intra cosine - mean inter cosine > 0.10 on the matched band.
"""
from __future__ import annotations
import json, socket
from pathlib import Path
import numpy as np

OUT = Path(__file__).resolve().parents[2] / "results/IDENTITY_H7_2026-06-09"
N_CH = 10
BAND_LO, BAND_HI = 49.0, 56.0   # common matched-temp band across machines


def coupling(fn):
    p = OUT/fn
    if not p.exists(): return None
    d = np.load(p); u = d["u"].astype(float); vz = d["vz"].astype(float)
    zone_of = d["zone_of"]; Tn = d["Tn"]; temps = d["temps"]; zones = list(d["zones"])
    L = len(u); flat = Tn.reshape(L, -1); taps = flat.shape[1]//N_CH
    perstep = flat.reshape(L, taps, N_CH).mean(1)
    band = (temps >= BAND_LO) & (temps <= BAND_HI)
    Araw = np.zeros((len(zones), N_CH)); Cnorm = np.zeros((len(zones), N_CH))
    kept = {}
    for zi, core in enumerate(zones):
        m = (zone_of == core) & band; kept[int(core)] = int(m.sum())
        if m.sum() < 40: continue
        uu = u[m]; vv = vz[m]; uc = uu-uu.mean(); vc = vv-vv.mean(); uvc = uc*vc
        A = np.stack([np.ones(m.sum()), uc, vc, uvc], 1)
        for c in range(N_CH):
            b, *_ = np.linalg.lstsq(A, perstep[m, c], rcond=None)
            Au, Av, Auv = b[1], b[2], b[3]
            Araw[zi, c] = Auv
            Cnorm[zi, c] = Auv/(np.sqrt(abs(Au*Av))+0.05)   # regularized
    return {"zones": zones, "Araw": Araw, "Cnorm": Cnorm, "kept": kept,
            "tmin": float(temps.min()), "tmax": float(temps.max())}


def cos(a, b):
    a = a.ravel(); b = b.ravel()
    return float(a@b/((np.linalg.norm(a)*np.linalg.norm(b))+1e-9))


def main():
    import itertools, glob
    # auto-discover all ikaros and daedalus spatial runs
    ik, da = {}, {}
    for p in sorted(glob.glob(str(OUT/"spatial_zones_raw_ikaros_*.npz"))):
        tag = Path(p).stem.split("ikaros_")[1]; c = coupling(Path(p).name)
        if c is not None: ik[tag] = c
    for p in sorted(glob.glob(str(OUT/"spatial_zones_raw_daedalus_*.npz"))):
        tag = Path(p).stem.split("daedalus_")[1]; c = coupling(Path(p).name)
        if c is not None: da[tag] = c
    print(f"ikaros runs: {list(ik)}   daedalus runs: {list(da)}", flush=True)
    if len(ik) < 2 or len(da) < 1:
        print("need >=2 ikaros + >=1 daedalus runs"); return
    for k, v in {**{f'ik_{t}': c for t,c in ik.items()}, **{f'da_{t}': c for t,c in da.items()}}.items():
        print(f"  {k}: kept/zone={v['kept']}", flush=True)

    def rowcos(A, B): return float(np.mean([cos(A[i], B[i]) for i in range(min(len(A), len(B)))]))
    out = {}
    for key in ["Araw", "Cnorm"]:
        intra = [cos(a[key], b[key]) for a, b in itertools.combinations(ik.values(), 2)]
        intra += [cos(a[key], b[key]) for a, b in itertools.combinations(da.values(), 2)]  # daedalus intra too
        inter = [cos(a[key], b[key]) for a in ik.values() for b in da.values()]
        intra_rz = [rowcos(a[key], b[key]) for a, b in itertools.combinations(ik.values(), 2)]
        inter_rz = [rowcos(a[key], b[key]) for a in ik.values() for b in da.values()]
        mi, ma = float(np.mean(intra)), float(np.mean(inter))
        gap = mi-ma
        # simple separation: does min intra exceed max inter? (clean PUF-style separation)
        clean_sep = (min(intra) > max(inter)) if intra and inter else False
        die_specific = gap > 0.10
        out[key] = {"intra_cosines": [round(x,3) for x in intra], "inter_cosines": [round(x,3) for x in inter],
                    "mean_intra": mi, "mean_inter": ma, "gap": gap,
                    "min_intra": float(min(intra)), "max_inter": float(max(inter)),
                    "clean_separation": bool(clean_sep),
                    "mean_intra_rowcos": float(np.mean(intra_rz)) if intra_rz else None,
                    "mean_inter_rowcos": float(np.mean(inter_rz)) if inter_rz else None,
                    "DIE_SPECIFIC_meangap": bool(die_specific)}
        print(f"\n[{key}] INTRA cosines={[round(x,2) for x in intra]} mean={mi:.3f}", flush=True)
        print(f"[{key}] INTER cosines={[round(x,2) for x in inter]} mean={ma:.3f}  gap={gap:+.3f}"
              f"  clean_sep(min_intra>max_inter)={clean_sep}"
              f"{'  <-- DIE-SPECIFIC' if die_specific else ''}", flush=True)
    verdict = out["Cnorm"]["DIE_SPECIFIC_meangap"] and out["Cnorm"]["clean_separation"]
    out["VERDICT"] = ("SPATIAL coupling die-specific (mean gap>0.10 AND clean min_intra>max_inter) — räkna-unikt SUPPORTED"
                      if verdict else
                      "weak/ambiguous — mean gap may exceed 0.10 but NOT cleanly separated, or raw disagrees; NOT established")
    (OUT/"spatial_transfer.json").write_text(json.dumps(out, indent=2))
    print(f"\n>>> {out['VERDICT']}", flush=True)


if __name__ == "__main__":
    main()
