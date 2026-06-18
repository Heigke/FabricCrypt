"""H7 RATIO-METRIC u·v coefficient — apply the O106 oracle consensus fix to data IN HAND (free, CPU-only).

All 4 oracles + web agent converged: our cross-die "not die-specific" was a THERMAL-CONFOUNDED (invalid) test
— the u-only control also failed to transfer = a PUF *reliability* failure, not a *uniqueness* test. The fix
(GPT-5 + Gemini + web): TEMPERATURE-COMPENSATE the u·v coefficient by normalizing it by the linear terms,
C_uv = A_uv / sqrt(|A_u·A_v|), which cancels first-order temperature/gain. Then ask: is the COMPENSATED
coefficient vector die-distinguishable (ikaros vs daedalus) even though the two sessions ran at different temps?

This is the cheap first look that decides whether the full matched-temperature multi-zone protocol is worth
the hot runs. n=1 session per die (weak — can't do stats), but if C_uv is wildly different per die after
compensation it's a positive signal; if near-identical, the die-specificity prior drops hard.
Per channel: regress per-step telemetry on centered [1, u1, v1, u1·v1] (u1=lag1 u, v1=lag1 v) -> A_u,A_v,A_uv.
Reports raw A_uv, compensated C_uv, cross-die cosine/L2, and which channels separate. Reads the two
cross_die_mixing_raw npz (+ mixing_strong_raw_ikaros if present for an intra-die stability check).
"""
from __future__ import annotations
import json, socket
from pathlib import Path
import numpy as np

OUT = Path(__file__).resolve().parents[2] / "results/IDENTITY_H7_2026-06-09"


def lag(x, k):
    y = np.zeros_like(x);
    if k > 0: y[k:] = x[:-k]
    return y if k > 0 else x.copy()


def coeffs(u, v, Tn):
    """per-channel A_u, A_v, A_uv from regression of per-step transient-mean on centered [1,u1,v1,u1v1]."""
    L = len(u); flat = Tn.reshape(L, -1)
    # collapse the NTAP taps to per-step per-channel mean (10 channels)
    nch = 10; taps = flat.shape[1] // nch
    perstep = flat.reshape(L, taps, nch).mean(1)   # (L,10)
    u1 = lag(u, 1).astype(float); v1 = lag(v, 1).astype(float)
    uc = u1-u1.mean(); vc = v1-v1.mean(); uvc = uc*vc
    A = np.stack([np.ones(L), uc, vc, uvc], 1)
    Au = np.zeros(nch); Av = np.zeros(nch); Auv = np.zeros(nch)
    for c in range(nch):
        b, *_ = np.linalg.lstsq(A[150:], perstep[150:, c], rcond=None)
        Au[c], Av[c], Auv[c] = b[1], b[2], b[3]
    Cuv = Auv / (np.sqrt(np.abs(Au*Av))+1e-9)   # temperature-compensated coefficient
    return Au, Av, Auv, Cuv


def load(name):
    p = OUT/f"{name}.npz"
    if not p.exists(): return None
    d = np.load(p); return d["u"].astype(int), d["v"].astype(int), d["Tn"]


def main():
    ds = {}
    for tag, fn in [("ikaros", "cross_die_mixing_raw_ikaros"),
                    ("daedalus", "cross_die_mixing_raw_daedalus"),
                    ("ikaros_strong", "mixing_strong_raw_ikaros")]:
        r = load(fn)
        if r is not None: ds[tag] = coeffs(*r)
    if "ikaros" not in ds or "daedalus" not in ds:
        print("need both cross_die raws"); return

    def show(tag):
        Au, Av, Auv, Cuv = ds[tag]
        print(f"  {tag:14s} A_uv=[{', '.join(f'{x:+.3f}' for x in Auv)}]")
        print(f"  {'':14s} C_uv=[{', '.join(f'{x:+.2f}' for x in Cuv)}]  (temp-compensated)")
    for t in ds: show(t)

    def cos(a, b): return float(a@b/((np.linalg.norm(a)*np.linalg.norm(b))+1e-9))
    ik_Cuv = ds["ikaros"][3]; da_Cuv = ds["daedalus"][3]
    ik_Auv = ds["ikaros"][2]; da_Auv = ds["daedalus"][2]
    out = {"raw_Auv_cos_ik_da": cos(ik_Auv, da_Auv),
           "comp_Cuv_cos_ik_da": cos(ik_Cuv, da_Cuv),
           "comp_Cuv_L2_ik_da": float(np.linalg.norm(ik_Cuv-da_Cuv)),
           "ikaros_Cuv": ik_Cuv.tolist(), "daedalus_Cuv": da_Cuv.tolist()}
    if "ikaros_strong" in ds:   # intra-die stability check (same die, different operating point)
        iks_Cuv = ds["ikaros_strong"][3]
        out["INTRA_ik_vs_ikstrong_Cuv_cos"] = cos(ik_Cuv, iks_Cuv)
        out["INTRA_ik_vs_ikstrong_Cuv_L2"] = float(np.linalg.norm(ik_Cuv-iks_Cuv))
    print(f"\n  raw A_uv cosine(ik,da)        = {out['raw_Auv_cos_ik_da']:+.3f}")
    print(f"  COMPENSATED C_uv cosine(ik,da) = {out['comp_Cuv_cos_ik_da']:+.3f}   L2={out['comp_Cuv_L2_ik_da']:.2f}")
    if "ikaros_strong" in ds:
        print(f"  INTRA-die C_uv cosine(ik,ik_strong) = {out['INTRA_ik_vs_ikstrong_Cuv_cos']:+.3f}  L2={out['INTRA_ik_vs_ikstrong_Cuv_L2']:.2f}")
        print(f"  >>> die-specific IF intra-die cosine >> inter-die cosine (compensated)")
    else:
        print("  (run heavy-v probe for intra-die stability check)")
    (OUT/"mixing_ratiometric.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
