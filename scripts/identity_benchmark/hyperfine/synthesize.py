#!/usr/bin/env python3
"""Cross-host hyperfine synthesis.

Inputs:
  results/IDENTITY_BENCHMARK_2026-05-30/hyperfine/ikaros/*.json + *.npz
  results/IDENTITY_BENCHMARK_2026-05-30/hyperfine/daedalus/*.json + *.npz
  Optional: results/.../ikaros_drift/*.json (within-machine repeat for d_within calc)

Outputs:
  results/IDENTITY_BENCHMARK_2026-05-30/hyperfine/combined/SYNTHESIS.json
  results/IDENTITY_BENCHMARK_2026-05-30/hyperfine/combined/SYNTHESIS.md
"""
import os, sys, json, math, glob
import numpy as np

BASE = "results/IDENTITY_BENCHMARK_2026-05-30/hyperfine"
IK = os.path.join(BASE, "ikaros")
DA = os.path.join(BASE, "daedalus")
IK_DRIFT = os.path.join(BASE, "ikaros_drift")  # optional repeat
OUT = os.path.join(BASE, "combined")
os.makedirs(OUT, exist_ok=True)

def _load_json(p):
    try: return json.load(open(p))
    except Exception: return None

def _load_npz(p):
    try: return np.load(p)
    except Exception: return None

def extract_metrics(probe_dir):
    """Pull single-number summary metrics from each probe's json."""
    m = {}
    j = _load_json(os.path.join(probe_dir,"P1_lockin.json"))
    if j: m["P1.p_amp"] = j.get("p_amp"); m["P1.f_amp"]=j.get("f_amp"); m["P1.tg_amp"]=j.get("tg_amp")
    j = _load_json(os.path.join(probe_dir,"P2_fold.json"))
    if j: m["P2.fold_p_std"] = j.get("fold_p_std"); m["P2.z_p"]=j.get("z_p")
    j = _load_json(os.path.join(probe_dir,"P3_pump.json"))
    if j:
        m["P3.tau_p"] = j.get("tau_rise_p_s"); m["P3.peak_p"] = j.get("peak_p_w")
        m["P3.tau_tg"]= j.get("tau_rise_tg_s"); m["P3.peak_tg"]=j.get("peak_tg_c")
    j = _load_json(os.path.join(probe_dir,"P4_twotone.json"))
    if j and "snr" in j:
        for k,v in j["snr"].items(): m[f"P4.snr.{k}"] = v
    j = _load_json(os.path.join(probe_dir,"P5_step.json"))
    if j:
        m["P5.f_peak"] = j.get("f_peak_mhz"); m["P5.rise_s"]=j.get("rise_time_s"); m["P5.ringback_hz"]=j.get("ringback_hz")
    j = _load_json(os.path.join(probe_dir,"P6_allan.json"))
    if j and "allan_dev" in j:
        for tau, ad in zip(j["taus"], j["allan_dev"]):
            m[f"P6.allan_tau{tau}"] = ad
    j = _load_json(os.path.join(probe_dir,"P7_mi.json"))
    if j and "MI" in j:
        for pair, row in j["MI"].items():
            for li, lag in enumerate(j["lags_s"]):
                m[f"P7.MI.{pair}.lag{lag}"] = row[li]
    j = _load_json(os.path.join(probe_dir,"P8_bispec.json"))
    if j: m["P8.asym"] = j.get("bispec_asymmetry"); m["P8.magBmax"]=j.get("magB_max"); m["P8.magBmean"]=j.get("magB_mean")
    j = _load_json(os.path.join(probe_dir,"P9_sync.json"))
    if j:
        m["P9.p_mean"]=j.get("p_mean"); m["P9.p_std"]=j.get("p_std")
        m["P9.tg_mean"]=j.get("tg_mean"); m["P9.f_mean"]=j.get("f_mean")
    j = _load_json(os.path.join(probe_dir,"P10_count.json"))
    if j:
        m["P10.fano_p"]=j.get("fano_power"); m["P10.fano_f"]=j.get("fano_freq")
        m["P10.event_rate"]=j.get("p_event_rate_hz")
    return m

def compare(a, b):
    """Return relative effect-size: |a-b|/(0.5*(|a|+|b|)+1e-12)."""
    if a is None or b is None: return None
    try:
        a = float(a); b = float(b)
    except Exception:
        return None
    if math.isnan(a) or math.isnan(b): return None
    denom = 0.5*(abs(a)+abs(b)) + 1e-12
    return abs(a-b)/denom

def p9_differential():
    """P9 simultaneous-window common-mode rejection."""
    a = _load_npz(os.path.join(IK,"P9_sync.npz"))
    b = _load_npz(os.path.join(DA,"P9_sync.npz"))
    if a is None or b is None: return {"status":"missing"}
    # align by wall_start
    wa = float(a["wall_start"][()] if a["wall_start"].shape==() else a["wall_start"])
    wb = float(b["wall_start"][()] if b["wall_start"].shape==() else b["wall_start"])
    skew = wa - wb
    n = min(len(a["p"]), len(b["p"]))
    pa, pb = a["p"][:n], b["p"][:n]
    tga, tgb = a["tg"][:n], b["tg"][:n]
    # subtract per-host mean (common-mode), then look at residual scale
    pa0 = pa - pa.mean(); pb0 = pb - pb.mean()
    diff = pa0 - pb0
    common = 0.5*(pa0 + pb0)
    cm_rej_db = 20*math.log10((np.std(common)+1e-12)/(np.std(diff)+1e-12))
    return {
        "wall_skew_s": skew,
        "n_samples": int(n),
        "p_ikaros_mean": float(pa.mean()), "p_daedalus_mean": float(pb.mean()),
        "p_ikaros_std": float(pa.std()),   "p_daedalus_std": float(pb.std()),
        "diff_std": float(diff.std()),
        "common_std": float(common.std()),
        "common_to_diff_dB": cm_rej_db,
        "tg_ikaros_mean": float(tga.mean()), "tg_daedalus_mean": float(tgb.mean()),
        "tg_diff_mean": float((tga - tgb).mean()),
        "corr_pa_pb": float(np.corrcoef(pa, pb)[0,1]),
        "corr_tga_tgb": float(np.corrcoef(tga, tgb)[0,1]),
    }

def cross_host_summary():
    mi = extract_metrics(IK); md = extract_metrics(DA)
    keys = sorted(set(mi) | set(md))
    rows = []
    for k in keys:
        a, b = mi.get(k), md.get(k)
        rel = compare(a, b)
        rows.append({"metric": k, "ikaros": a, "daedalus": b, "rel_diff": rel})
    # if a drift-repeat exists, use it for within-host baseline
    drift = None
    if os.path.isdir(IK_DRIFT):
        m2 = extract_metrics(IK_DRIFT)
        drift = {}
        for k in keys:
            a, b = mi.get(k), m2.get(k)
            drift[k] = compare(a, b)
        for r in rows:
            d = drift.get(r["metric"])
            r["within_drift"] = d
            r["signal_to_drift"] = (r["rel_diff"]/d) if (d and d>1e-12 and r["rel_diff"] is not None) else None
    # rank
    rows.sort(key=lambda r: (r.get("signal_to_drift") or r.get("rel_diff") or 0), reverse=True)
    return rows

def p11_compare():
    a = _load_json(os.path.join(IK,"P11_smu_calib.json"))
    b = _load_json(os.path.join(DA,"P11_smu_calib.json"))
    if a is None or b is None: return {"status":"missing"}
    diffs = []; sames = 0; both_err = 0; differs = 0
    for k in set(a)|set(b):
        va, vb = a.get(k), b.get(k)
        s_va = str(va)[:60]; s_vb = str(vb)[:60]
        a_err = isinstance(va,str) and va.startswith("ERR")
        b_err = isinstance(vb,str) and vb.startswith("ERR")
        if a_err and b_err: both_err += 1; continue
        same = (s_va == s_vb)
        if same: sames += 1
        else:
            differs += 1
            diffs.append({"key": k, "ikaros": s_va, "daedalus": s_vb})
    return {"identical_keys": sames, "differing_keys": differs, "both_unavailable": both_err, "differences": diffs[:30]}

def main():
    out = {
        "cross_host_metrics": cross_host_summary(),
        "P9_differential": p9_differential(),
        "P11_calibration_compare": p11_compare(),
    }
    with open(os.path.join(OUT,"SYNTHESIS.json"),"w") as f:
        json.dump(out, f, indent=2, default=str)
    # markdown report
    lines = ["# Hyperfine Synthesis — ikaros vs daedalus","",
             "## Top metrics by signal-to-drift (or relative diff if no drift baseline)",""]
    lines.append("| metric | ikaros | daedalus | rel_diff | within_drift | sig/drift |")
    lines.append("|---|---|---|---|---|---|")
    for r in out["cross_host_metrics"][:40]:
        def f(x):
            if x is None: return "—"
            if isinstance(x,float): return f"{x:.4g}"
            return str(x)
        lines.append(f"| {r['metric']} | {f(r['ikaros'])} | {f(r['daedalus'])} | {f(r['rel_diff'])} | {f(r.get('within_drift'))} | {f(r.get('signal_to_drift'))} |")
    lines += ["","## P9 Differential (common-mode rejection)","",
              "```json", json.dumps(out["P9_differential"], indent=2, default=str), "```",
              "","## P11 Calibration constants",
              f"identical={out['P11_calibration_compare'].get('identical_keys')}, differing={out['P11_calibration_compare'].get('differing_keys')}, both_unavailable={out['P11_calibration_compare'].get('both_unavailable')}",""]
    if out["P11_calibration_compare"].get("differences"):
        lines.append("\nDifferences (truncated):\n")
        for d in out["P11_calibration_compare"]["differences"][:15]:
            lines.append(f"- `{d['key']}` :: ikaros={d['ikaros']!r} vs daedalus={d['daedalus']!r}")
    with open(os.path.join(OUT,"SYNTHESIS.md"),"w") as f:
        f.write("\n".join(lines))
    print(f"[OK] wrote {OUT}/SYNTHESIS.json and SYNTHESIS.md")

if __name__ == "__main__":
    main()
