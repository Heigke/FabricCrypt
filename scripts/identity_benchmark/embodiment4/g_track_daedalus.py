"""G-track: replicate embodiment3 G1-G4 with daedalus as the host chassis.

Gates (mirror embodiment3, daedalus-centric):
  G1: daedalus self NRMSE ≤ 0.70
  G2: ikaros transplant NRMSE ≥ 3× G1
  G3: same-machine re-measured daedalus signature → NRMSE ≤ 1.5× G1
  G4: post-reboot daedalus signature → NRMSE ≤ 1.5× G1

Designed to be invoked in two stages:
  --stage pre   : G1+G2+G3 using daedalus_prereboot, daedalus_remeasure, ikaros sig
  --stage post  : G4 using daedalus_postreboot sig + cached weights+train_sig

Writes results JSON incrementally.
"""
from __future__ import annotations
import argparse, hashlib, json, sys, pickle
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "embodiment3"))
from v3_phase_c import (derive_structure, train_eval, transplant_eval, nrmse)
from robust_signature import load_signature, quantize_robust, quantized_to_bitstring

ROOT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
OUT_DIR = ROOT / "results/IDENTITY_BENCHMARK_2026-05-30/embodiment4"
SIGS = OUT_DIR / "signatures"
WEIGHTS_PKL = OUT_DIR / "g_track_weights.pkl"
RESULT = OUT_DIR / "g_track_result.json"


def load_bitstring(p: Path) -> str:
    return quantized_to_bitstring(quantize_robust(load_signature(str(p))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("pre", "post"), required=True)
    ap.add_argument("--seeds", type=int, default=10)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seeds = list(range(args.seeds))

    results = json.loads(RESULT.read_text()) if RESULT.exists() else {}

    if args.stage == "pre":
        bs_train = load_bitstring(SIGS / "daedalus_prereboot.json")
        struct_train = derive_structure(bs_train)
        print(f"[G] daedalus train mask density={struct_train[0].mean():.3f}", flush=True)

        g1, weights_list = [], []
        for s in seeds:
            nr, w = train_eval(struct_train, s)
            g1.append(nr); weights_list.append(w)
            print(f"[G][G1 daedalus] seed={s} NRMSE={nr:.4f}", flush=True)
        g1_med = float(np.median(g1))
        results["G1_daedalus"] = {"nrmse_per_seed": g1, "median": g1_med}
        print(f"[G][G1] daedalus median NRMSE={g1_med:.4f}", flush=True)

        # G2: ikaros transplant
        ikaros_sig = SIGS.parent.parent / "embodiment3/signatures/ikaros_v2a_t0.json"
        bs_ikaros = load_bitstring(ikaros_sig)
        struct_ik = derive_structure(bs_ikaros)
        g2 = [transplant_eval(w, struct_ik, s) for s, w in zip(seeds, weights_list)]
        g2_med = float(np.median(g2))
        results["G2_ikaros_transplant"] = {"nrmse_per_seed": g2, "median": g2_med,
                                            "factor": g2_med / max(1e-9, g1_med)}
        print(f"[G][G2] ikaros transplant NRMSE={g2_med:.4f} factor={g2_med/g1_med:.2f}x", flush=True)

        # G3 needs daedalus re-measure
        rm_path = SIGS / "daedalus_remeasure.json"
        if rm_path.exists():
            bs_rm = load_bitstring(rm_path)
            struct_rm = derive_structure(bs_rm)
            hd = sum(1 for i in range(min(len(bs_train), len(bs_rm)))
                     if bs_train[i] != bs_rm[i])
            n = min(len(bs_train), len(bs_rm))
            mo = float((struct_train[0] == struct_rm[0]).mean())
            g3 = [transplant_eval(w, struct_rm, s) for s, w in zip(seeds, weights_list)]
            g3_med = float(np.median(g3))
            results["G3_remeasure"] = {"hamming": hd, "n_bits": n, "pct": 100*hd/max(1,n),
                                        "mask_overlap": mo, "nrmse_per_seed": g3,
                                        "median": g3_med, "factor": g3_med / max(1e-9, g1_med)}
            print(f"[G][G3] remeasure drift={hd}/{n} ({100*hd/n:.2f}%) NRMSE={g3_med:.4f} factor={g3_med/g1_med:.2f}x", flush=True)

        # Stash weights+train bs for G4 stage
        with open(WEIGHTS_PKL, "wb") as f:
            pickle.dump({"weights_list": weights_list, "bs_train": bs_train,
                          "g1_med": g1_med, "g1": g1}, f)

    elif args.stage == "post":
        post_path = SIGS / "daedalus_postreboot.json"
        if not post_path.exists():
            raise SystemExit(f"missing post-reboot sig: {post_path}")
        bs_post = load_bitstring(post_path)
        with open(WEIGHTS_PKL, "rb") as f:
            cache = pickle.load(f)
        bs_train = cache["bs_train"]; weights_list = cache["weights_list"]; g1_med = cache["g1_med"]
        struct_train = derive_structure(bs_train)
        struct_post = derive_structure(bs_post)
        hd = sum(1 for i in range(min(len(bs_train), len(bs_post)))
                 if bs_train[i] != bs_post[i])
        n = min(len(bs_train), len(bs_post))
        mo = float((struct_train[0] == struct_post[0]).mean())
        g4 = [transplant_eval(w, struct_post, s) for s, w in zip(seeds, weights_list)]
        g4_med = float(np.median(g4))
        results["G4_post_reboot"] = {"hamming": hd, "n_bits": n, "pct": 100*hd/max(1,n),
                                      "mask_overlap": mo, "nrmse_per_seed": g4,
                                      "median": g4_med, "factor": g4_med / max(1e-9, g1_med)}
        print(f"[G][G4] post-reboot drift={hd}/{n} ({100*hd/n:.2f}%) NRMSE={g4_med:.4f} factor={g4_med/g1_med:.2f}x mask_overlap={mo:.3f}", flush=True)

    # Recompute verdict
    g1_med = results.get("G1_daedalus", {}).get("median")
    g2 = results.get("G2_ikaros_transplant", {}); g2_med = g2.get("median") if g2 else None
    g3 = results.get("G3_remeasure", {}); g3_med = g3.get("median") if g3 else None
    g4 = results.get("G4_post_reboot", {}); g4_med = g4.get("median") if g4 else None
    if g1_med is not None:
        thr = 0.70
        gates = {
            "G1_pass": g1_med <= thr, "G1_value": g1_med,
            "G2_pass": g2_med is not None and g2_med >= 3.0 * g1_med, "G2_value": g2_med,
            "G3_pass": g3_med is not None and g3_med <= 1.5 * g1_med, "G3_value": g3_med,
            "G4_pass": g4_med is not None and g4_med <= 1.5 * g1_med, "G4_value": g4_med,
        }
        if all(gates[k] for k in ("G1_pass", "G2_pass", "G3_pass", "G4_pass")):
            verdict = "GENUINE_CHASSI_EMBODIMENT_DAEDALUS"
        elif gates["G1_pass"] and gates["G2_pass"] and gates["G3_pass"] and g4_med is None:
            verdict = "PENDING_REBOOT"
        elif gates["G1_pass"] and gates["G2_pass"] and gates["G3_pass"] and not gates["G4_pass"]:
            verdict = "BOOT_STATE_BOUND"
        else:
            verdict = "FAILED"
        gates["VERDICT"] = verdict
        results["gates"] = gates
        print(f"[G] gates: {json.dumps(gates, indent=2, default=str)}", flush=True)
    RESULT.write_text(json.dumps(results, indent=2, default=str))
    print(f"[G] wrote {RESULT}", flush=True)


if __name__ == "__main__":
    main()
