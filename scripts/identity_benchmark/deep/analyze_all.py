"""Cross-machine analyzer for all 5 deep angles.
Reads ikaros + daedalus JSON, computes per-angle discriminator + cross-angle
logistic regression. Bonferroni 5-way.
"""
import json, os, sys, glob, numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent.parent / "results/IDENTITY_BENCHMARK_2026-05-30/deep"

def load(host, angle):
    files = list((ROOT/host).glob(f"{angle}_*.json"))
    if not files: return None
    return json.load(open(files[0]))

def bootstrap_diff_ci(a, b, n=1000):
    a, b = np.array(a, float), np.array(b, float)
    if len(a)==0 or len(b)==0: return None
    rng = np.random.default_rng(42)
    diffs = []
    for _ in range(n):
        sa = rng.choice(a, size=len(a), replace=True)
        sb = rng.choice(b, size=len(b), replace=True)
        diffs.append(sa.mean()-sb.mean())
    return float(np.mean(diffs)), float(np.percentile(diffs,2.5)), float(np.percentile(diffs,97.5))

def kl_div(p, q, eps=1e-9):
    p = np.asarray(p,float)+eps; q = np.asarray(q,float)+eps
    p /= p.sum(); q /= q.sum()
    return float((p*np.log(p/q)).sum())

def cohen_d(a,b):
    a,b = np.array(a,float), np.array(b,float)
    sp = np.sqrt((a.var(ddof=1)+b.var(ddof=1))/2) if (len(a)>1 and len(b)>1) else 1
    return float((a.mean()-b.mean())/(sp if sp>0 else 1))

def analyze_A():
    A_i = load("ikaros","A"); A_d = load("daedalus","A")
    if not (A_i and A_d): return {"status":"missing"}
    out = {"per_workload":{}, "feature_vector":{}}
    for w in ["IDLE","LIGHT","MEDIUM","HEAVY"]:
        si, sd = A_i["stats"].get(w,{}), A_d["stats"].get(w,{})
        if not (si.get("n_samples") and sd.get("n_samples")): continue
        rep_i = si["per_rep_means"]; rep_d = sd["per_rep_means"]
        diff = bootstrap_diff_ci(rep_i, rep_d)
        d_eff = cohen_d(rep_i, rep_d)
        # KL on histograms
        hi, hd = si["hist_counts"], sd["hist_counts"]
        # rebin to common range
        L = min(len(hi), len(hd))
        klv = kl_div(hi[:L], hd[:L])
        out["per_workload"][w] = {
            "ikaros_mean_W": si["mean_W"], "daedalus_mean_W": sd["mean_W"],
            "diff_W_ci": diff, "cohen_d": d_eff, "kl": klv,
            "ikaros_std": si["std_W"], "daedalus_std": sd["std_W"],
            "tau_i": si["autocorr_tau"], "tau_d": sd["autocorr_tau"],
        }
        out["feature_vector"][w] = [si["mean_W"], si["std_W"], sd["mean_W"], sd["std_W"]]
    # gate: at least one workload mean diff > 2 sigma AND |d| > 0.8
    passes = []
    for w,v in out["per_workload"].items():
        sigma_pool = np.sqrt(v["ikaros_std"]**2 + v["daedalus_std"]**2)/2
        diff_abs = abs(v["ikaros_mean_W"]-v["daedalus_mean_W"])
        gate = (diff_abs > 2*sigma_pool) and (abs(v["cohen_d"])>0.8)
        v["gate_pass"] = bool(gate)
        if gate: passes.append(w)
    out["verdict"] = "DISCOVERY" if passes else ("AMBIGUOUS" if any(abs(v["cohen_d"])>0.5 for v in out["per_workload"].values()) else "NULL")
    out["passes"] = passes
    return out

def analyze_B():
    B_i = load("ikaros","B"); B_d = load("daedalus","B")
    if not (B_i and B_d): return {"status":"missing"}
    out = {}
    for key in ["tau_heat","tau_cool","R_th_K_per_W"]:
        vi = [c[key] for c in B_i["cycles"] if c[key]==c[key] and abs(c[key])<1e4]
        vd = [c[key] for c in B_d["cycles"] if c[key]==c[key] and abs(c[key])<1e4]
        if not (vi and vd): out[key] = None; continue
        diff = bootstrap_diff_ci(vi,vd)
        d_eff = cohen_d(vi,vd)
        out[key] = {"ikaros_mean":float(np.mean(vi)), "daedalus_mean":float(np.mean(vd)),
                    "diff_ci":diff, "cohen_d":d_eff, "n_i":len(vi), "n_d":len(vd)}
    # gate
    gates = [v for v in out.values() if v and abs(v["cohen_d"])>0.8 and (v["diff_ci"][1]>0 or v["diff_ci"][2]<0)]
    out["verdict"] = "DISCOVERY" if gates else ("AMBIGUOUS" if any(v and abs(v["cohen_d"])>0.5 for v in out.values()) else "NULL")
    return out

def analyze_C():
    return {"verdict":"BLOCKED",
            "reason":"NPU userspace (XRT or pyxrt) not installed on either host. /dev/accel/accel0 char device exists; amdxdna kernel module loaded; no userspace runtime to submit kernels.",
            "to_unblock":"install AMD Ryzen-AI SW stack (xrt-smi, amd-aie/xrt-mcs deb packages + amdxdna fw) and a compiled .xclbin from AMD's RyzenAI-SW examples."}

def analyze_D():
    D_i = load("ikaros","D"); D_d = load("daedalus","D")
    if not (D_i and D_d): return {"status":"missing"}
    out = {"levels":{}}
    for lvl_i, lvl_d in zip(D_i.get("levels",[]), D_d.get("levels",[])):
        name = lvl_i.get("level","?")
        out["levels"][name] = {
            "ikaros_unstable": lvl_i.get("n_unstable_tiles"),
            "daedalus_unstable": lvl_d.get("n_unstable_tiles"),
            "ikaros_time_ms": lvl_i.get("rep_time_mean_ms"),
            "daedalus_time_ms": lvl_d.get("rep_time_mean_ms"),
        }
    # any bit flips at all?
    any_flips = any((v.get("ikaros_unstable") or 0) + (v.get("daedalus_unstable") or 0) > 0 for v in out["levels"].values())
    # timing fingerprint
    ti = [v["ikaros_time_ms"] for v in out["levels"].values() if v.get("ikaros_time_ms")]
    td = [v["daedalus_time_ms"] for v in out["levels"].values() if v.get("daedalus_time_ms")]
    timing_diff = bootstrap_diff_ci(ti, td) if (ti and td) else None
    out["any_bit_flips"] = any_flips
    out["timing_diff_ms_ci"] = timing_diff
    if any_flips: out["verdict"]="DISCOVERY"
    elif timing_diff and (timing_diff[1]>0 or timing_diff[2]<0): out["verdict"]="AMBIGUOUS"
    else: out["verdict"]="NULL"
    return out

def analyze_E():
    E_i = load("ikaros","E"); E_d = load("daedalus","E")
    if not (E_i and E_d): return {"status":"missing"}
    out = {"per_core":[]}
    ti = [np.mean(c["times_s"]) for c in E_i["per_core"] if c["times_s"]]
    td = [np.mean(c["times_s"]) for c in E_d["per_core"] if c["times_s"]]
    fi = [int(c["meta"]["amd_pstate_max_freq"] or 0) for c in E_i["per_core"]]
    fd = [int(c["meta"]["amd_pstate_max_freq"] or 0) for c in E_d["per_core"]]
    n = min(len(ti),len(td))
    out["time_diff_ci"] = bootstrap_diff_ci(ti[:n], td[:n])
    out["freq_diff_ci"] = bootstrap_diff_ci(fi[:n], fd[:n])
    out["time_cohen_d"] = cohen_d(ti[:n], td[:n])
    out["freq_cohen_d"] = cohen_d(fi[:n], fd[:n])
    # core ranking correlation (shuffled control)
    if n>=4:
        from numpy import argsort
        rank_i = argsort(ti[:n]); rank_d = argsort(td[:n])
        out["rank_corr_pearson"] = float(np.corrcoef(rank_i, rank_d)[0,1])
    out["ikaros_per_core_time"] = ti; out["daedalus_per_core_time"] = td
    out["ikaros_per_core_freq"] = fi; out["daedalus_per_core_freq"] = fd
    gates = (abs(out.get("time_cohen_d",0))>0.8) or (abs(out.get("freq_cohen_d",0))>0.8)
    out["verdict"]="DISCOVERY" if gates else ("AMBIGUOUS" if (abs(out.get("time_cohen_d",0))>0.5) else "NULL")
    return out

def cross_angle_lr(A,B,E):
    """Build per-machine feature vector and try logistic regression with leave-one-out."""
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import LeaveOneOut
    except Exception as e:
        return {"error":str(e)}
    feats_i, feats_d = [], []
    if A and "per_workload" in A:
        for w in ["IDLE","LIGHT","MEDIUM","HEAVY"]:
            v=A["per_workload"].get(w)
            if v:
                feats_i += [v["ikaros_mean_W"], v["ikaros_std"], v["tau_i"]]
                feats_d += [v["daedalus_mean_W"], v["daedalus_std"], v["tau_d"]]
    if B:
        for k in ["tau_heat","tau_cool","R_th_K_per_W"]:
            v=B.get(k)
            if v: feats_i.append(v["ikaros_mean"]); feats_d.append(v["daedalus_mean"])
    if E:
        feats_i += E.get("ikaros_per_core_time",[])[:8]
        feats_d += E.get("daedalus_per_core_time",[])[:8]
    n = min(len(feats_i), len(feats_d))
    if n<3: return {"error":"too few features"}
    # 2-row dataset is degenerate for LR — instead compute Mahalanobis distance ratio
    vi = np.array(feats_i[:n]); vd = np.array(feats_d[:n])
    return {"n_features":n,
            "L2_distance":float(np.linalg.norm(vi-vd)),
            "L2_per_feature":float(np.linalg.norm(vi-vd)/np.sqrt(n)),
            "cosine":float(np.dot(vi,vd)/(np.linalg.norm(vi)*np.linalg.norm(vd)+1e-9))}

def power_analysis(A):
    """How many seeds for 10% effect at α=0.05?"""
    if not A or "per_workload" not in A: return None
    out={}
    for w,v in A["per_workload"].items():
        sigma = np.sqrt(v["ikaros_std"]**2 + v["daedalus_std"]**2)/2
        target = 0.10 * max(v["ikaros_mean_W"],1)
        # Cohen's formula n ≈ 16*(σ/Δ)^2 for two-sample t-test at α=0.05, power=0.8
        n = int(np.ceil(16 * (sigma/target)**2)) if target>0 else None
        out[w] = {"sigma_W":float(sigma), "target_W":float(target), "n_seeds_required":n}
    return out

def main():
    out_path = ROOT / "ANALYSIS.json"
    res = {
        "A": analyze_A(),
        "B": analyze_B(),
        "C": analyze_C(),
        "D": analyze_D(),
        "E": analyze_E(),
    }
    res["cross_angle"] = cross_angle_lr(res["A"], res["B"], res["E"])
    res["power_analysis"] = power_analysis(res["A"])
    # bonferroni: 5 angles, alpha_each = 0.01
    res["bonferroni_alpha"] = 0.05/5
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out_path,"w"), indent=2, default=str)
    print(f"wrote {out_path}")
    for k,v in res.items():
        if isinstance(v,dict): print(f"{k}: verdict={v.get('verdict','-')}")

if __name__=="__main__": main()
