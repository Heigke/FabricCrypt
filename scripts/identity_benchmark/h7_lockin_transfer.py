"""H7 LOCK-IN transfer verdict — is the frequency-domain intermod signature DIE-SPECIFIC? (K=2/die gate).

Loads lock-in feature tensors feat[zone,pair,ch,{f1+f2,f1-f2},{re,im}] from K runs per die. The per-run mad
baseline is unstable (|feat| varies orders of magnitude run-to-run), so cosine on raw feat is dominated by one
blown-up channel. Fix = WHITEN each of the 960 feature dimensions across the run population (subtract mean,
divide std) before cosine — neutralizes single-channel scale dominance, the pre-registered method.
Reports: raw cosine, whitened cosine, and a MATCHED-TEMP variant (only segments whose per-run Tmean falls in a
common band across all runs, using the saved T_z*_p* arrays). DIE-SPECIFIC requires intra >> inter:
clean separation (min_intra > max_inter) AND pre-reg margin min_intra - max_inter >= 0.03.
Honest gate: also compares intra reliability vs the spatial protocol's ~0.7 wall.
"""
from __future__ import annotations
import json, glob, itertools
from pathlib import Path
import numpy as np

OUT = Path(__file__).resolve().parents[2] / "results/IDENTITY_H7_2026-06-09"
N_CH = 10


def load_runs():
    runs = {}  # key -> dict(feat, segT)
    for p in sorted(glob.glob(str(OUT/"lockin_raw_*.npz"))):
        name = Path(p).stem.replace("lockin_raw_", "")  # e.g. ikaros_r1
        if "smoke" in name:
            continue
        d = np.load(p)
        feat = d["feat"].astype(np.float64)           # (Z,P,10,2,2)
        zones = list(d["zones"]); pairs = d["tonepairs"]
        # per-segment mean temp from saved T_z{core}_p{pj}
        Z, P = feat.shape[0], feat.shape[1]
        segT = np.full((Z, P), np.nan)
        for zi, core in enumerate(zones):
            for pj in range(P):
                k = f"T_z{int(core)}_p{pj}"
                if k in d.files:
                    segT[zi, pj] = float(np.mean(d[k]))
        runs[name] = {"feat": feat, "segT": segT, "zones": zones}
    return runs


def cos(a, b):
    a = a.ravel(); b = b.ravel()
    return float(a@b/((np.linalg.norm(a)*np.linalg.norm(b))+1e-12))


def whiten(mat):
    # mat: (n_runs, dim). standardize each dim across runs.
    mu = mat.mean(0, keepdims=True); sd = mat.std(0, keepdims=True)
    return (mat - mu)/(sd + 1e-9)


def split_die(keys):
    die = {}
    for k in keys:
        d = k.split("_")[0]
        die.setdefault(d, []).append(k)
    return die


def pair_stats(vecs, keys, label):
    die = split_die(keys)
    idx = {k: i for i, k in enumerate(keys)}
    intra, inter = [], []
    intra_lab, inter_lab = [], []
    for d, ks in die.items():
        for a, b in itertools.combinations(ks, 2):
            intra.append(cos(vecs[idx[a]], vecs[idx[b]])); intra_lab.append(f"{a}~{b}")
    dl = list(die)
    for i in range(len(dl)):
        for j in range(i+1, len(dl)):
            for a in die[dl[i]]:
                for b in die[dl[j]]:
                    inter.append(cos(vecs[idx[a]], vecs[idx[b]])); inter_lab.append(f"{a}~{b}")
    res = {"label": label,
           "intra": [round(x, 3) for x in intra], "intra_lab": intra_lab,
           "inter": [round(x, 3) for x in inter], "inter_lab": inter_lab,
           "mean_intra": float(np.mean(intra)) if intra else None,
           "mean_inter": float(np.mean(inter)) if inter else None,
           "min_intra": float(np.min(intra)) if intra else None,
           "max_inter": float(np.max(inter)) if inter else None}
    if intra and inter:
        res["gap"] = res["mean_intra"] - res["mean_inter"]
        res["margin_min_intra_minus_max_inter"] = res["min_intra"] - res["max_inter"]
        res["clean_separation"] = bool(res["min_intra"] > res["max_inter"])
        res["PASS_prereg_0.03"] = bool(res["margin_min_intra_minus_max_inter"] >= 0.03)
    return res


def main():
    runs = load_runs()
    keys = sorted(runs)
    print(f"runs: {keys}", flush=True)
    if len(keys) < 4:
        print("need >=4 runs (2/die)"); return
    Z, P = runs[keys[0]]["feat"].shape[:2]
    feats = {k: runs[k]["feat"].reshape(-1) for k in keys}
    dim = feats[keys[0]].size

    out = {"runs": keys, "n_dim": int(dim)}
    # report per-run scale + per-segment temp ranges (transparency)
    out["per_run"] = {k: {"feat_absmean": float(np.abs(runs[k]["feat"]).mean()),
                          "segT_min": float(np.nanmin(runs[k]["segT"])),
                          "segT_max": float(np.nanmax(runs[k]["segT"]))} for k in keys}

    # ---- (A) RAW cosine (scale-sensitive, for reference) ----
    raw_mat = np.stack([feats[k] for k in keys])
    out["A_raw"] = pair_stats(raw_mat, keys, "raw cosine (scale-sensitive)")

    # ---- (B) WHITENED cosine (pre-registered) ----
    wmat = whiten(raw_mat)
    out["B_whitened"] = pair_stats(wmat, keys, "whitened-per-dim cosine (PRE-REG)")

    # ---- (C) MATCHED-TEMP whitened: keep only segments in a common temp band across ALL runs ----
    segT = np.stack([runs[k]["segT"] for k in keys])  # (n,Z,P)
    band_lo, band_hi = 68.0, 73.0   # overlap of ikaros(~57-71) and daedalus(~72) operating temps
    common = np.all((segT >= band_lo) & (segT <= band_hi), axis=0)  # (Z,P) segments in-band for every run
    n_common = int(common.sum())
    out["C_matched_temp"] = {"band": [band_lo, band_hi], "n_common_segments": n_common,
                             "total_segments": int(Z*P)}
    if n_common >= 2:
        # mask: per segment keep all (ch,2,2)=40 dims
        mask = np.zeros((Z, P, N_CH, 2, 2), bool); mask[common] = True
        mask = mask.reshape(-1)
        mt_mat = whiten(raw_mat[:, mask])
        st = pair_stats(mt_mat, keys, f"matched-temp [{band_lo},{band_hi}]C whitened, {n_common} segs")
        out["C_matched_temp"].update(st)
    else:
        out["C_matched_temp"]["note"] = ("too few common-band segments at K=2 — ikaros and daedalus operate at "
                                         "different steady temps; matched-temp needs temp-controlled re-runs")

    # ---- verdict ----
    b = out["B_whitened"]
    spatial_intra = 0.7  # the wall the spatial protocol hit
    beats_spatial = (b["mean_intra"] is not None) and (b["mean_intra"] > spatial_intra)
    out["VERDICT"] = {
        "whitened_mean_intra": b["mean_intra"], "whitened_mean_inter": b["mean_inter"],
        "whitened_gap": b.get("gap"), "clean_separation": b.get("clean_separation"),
        "PASS_prereg_0.03": b.get("PASS_prereg_0.03"),
        "beats_spatial_0.7_reliability": bool(beats_spatial),
        "DECISION": None}
    if b.get("clean_separation") and b.get("PASS_prereg_0.03") and beats_spatial:
        out["VERDICT"]["DECISION"] = "GO — lock-in separates the 2 dies cleanly at K=2; scale to K=9 for a firmer demo"
    else:
        out["VERDICT"]["DECISION"] = ("NO-GO — lock-in did NOT cleanly separate / did not beat spatial reliability "
                                      "at K=2; räkna-unikt remains unestablished. Consolidate at 2.5/3.")

    (OUT/"lockin_transfer.json").write_text(json.dumps(out, indent=2))
    # print summary
    for key in ["A_raw", "B_whitened"]:
        r = out[key]
        print(f"\n[{r['label']}] intra={r['intra']} (mean {r['mean_intra']:.3f})", flush=True)
        print(f"  inter={r['inter']} (mean {r['mean_inter']:.3f})  gap={r['gap']:+.3f} "
              f"clean_sep={r['clean_separation']} margin={r['margin_min_intra_minus_max_inter']:+.3f} "
              f"prereg_pass={r['PASS_prereg_0.03']}", flush=True)
    c = out["C_matched_temp"]
    print(f"\n[matched-temp {c['band']}C] common segments={c['n_common_segments']}/{c['total_segments']}", flush=True)
    if "mean_intra" in c:
        print(f"  intra={c['intra']} (mean {c['mean_intra']:.3f}) inter={c['inter']} (mean {c['mean_inter']:.3f}) "
              f"clean_sep={c['clean_separation']}", flush=True)
    else:
        print(f"  {c.get('note','')}", flush=True)
    print(f"\n>>> DECISION: {out['VERDICT']['DECISION']}", flush=True)


if __name__ == "__main__":
    main()
