"""G4 re-test using a HARDENED static signature.

Original robust_signature includes 'hwmon_enum' (a dict — serialization order
can shuffle across reboots if kernel probe order changes) and 'mem_total_kB'
(can vary by a few kB across reboots due to kernel allocations) in the
static hash. On daedalus the hwmon dict happened to re-order across the
reboot, flipping the static_hash even though every real chassi identifier
(DMI, CPU, microcode, GPU vid/did, PCI device list) was bit-identical.

This script:
  1. Re-derives static_hash using ONLY chassi-stable keys (no hwmon_enum,
     no mem_total_kB).
  2. Re-derives the reservoir structure from this hardened hash.
  3. Re-runs G3/G4 transplant for daedalus.
"""
from __future__ import annotations
import hashlib, json, sys, pickle
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "embodiment3"))
from v3_phase_c import (N, SPARSITY_BITS, ACT_CHOICES, transplant_eval, train_eval)
from robust_signature import load_signature

ROOT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
OUT_DIR = ROOT / "results/IDENTITY_BENCHMARK_2026-05-30/embodiment4"
SIGS = OUT_DIR / "signatures"
WEIGHTS_PKL = OUT_DIR / "g_track_weights.pkl"
RESULT = OUT_DIR / "g_track_robust_result.json"

HARDENED_STATIC_KEYS = (
    "dmi_board_name", "dmi_product_name", "dmi_bios_version", "dmi_sys_vendor",
    "cpu_model", "cpu_count", "cpu_microcode", "cpu_cache_size",
    "kernel_release", "arch", "hostname",
    "pci_device_ids", "gpu_vendor", "gpu_device", "gpu_revision",
)


def hardened_static_hash(sig_path: Path) -> str:
    sig = load_signature(str(sig_path))
    st = sig["static"]
    s = "|".join(f"{k}={st.get(k, '')}" for k in HARDENED_STATIC_KEYS)
    return hashlib.sha256(s.encode()).hexdigest()


def derive_struct_from_hash(static_hash_hex: str):
    """Reproduce v3_phase_c.derive_structure starting from a static-hash hex string."""
    static_bits = "".join(f"{b:08b}" for b in bytes.fromhex(static_hash_hex))[:256]
    h = hashlib.shake_256(static_bits.encode()).digest(2048 + 2048 + 128 + 128)
    MB = SPARSITY_BITS // 8
    flat = np.unpackbits(np.frombuffer(h[:MB], dtype=np.uint8))
    h2_bits = np.unpackbits(np.frombuffer(h[MB:2*MB], dtype=np.uint8))
    mask = (flat & h2_bits).reshape(N, N).astype(bool)
    np.fill_diagonal(mask, False)
    act_bytes = np.frombuffer(h[2*MB: 2*MB + N], dtype=np.uint8)
    acts = [ACT_CHOICES[b % len(ACT_CHOICES)] for b in act_bytes]
    perm_bytes = np.frombuffer(h[2*MB + N: 2*MB + 2*N], dtype=np.uint8)
    perm = np.argsort(perm_bytes).astype(np.int32)
    return mask, acts, perm


def main():
    pre = hardened_static_hash(SIGS / "daedalus_prereboot.json")
    rem = hardened_static_hash(SIGS / "daedalus_remeasure.json")
    post = hardened_static_hash(SIGS / "daedalus_postreboot.json")
    ikaros = hardened_static_hash(ROOT / "results/IDENTITY_BENCHMARK_2026-05-30/embodiment3/signatures/ikaros_v2a_t0.json")
    print(f"[GR] daedalus prereboot   static_hash={pre[:24]}", flush=True)
    print(f"[GR] daedalus remeasure   static_hash={rem[:24]}", flush=True)
    print(f"[GR] daedalus postreboot  static_hash={post[:24]}", flush=True)
    print(f"[GR] ikaros baseline      static_hash={ikaros[:24]}", flush=True)
    print(f"[GR] pre==remeasure: {pre == rem}", flush=True)
    print(f"[GR] pre==postreboot: {pre == post}", flush=True)
    print(f"[GR] pre==ikaros:     {pre == ikaros}", flush=True)

    # Build reservoir structures
    struct_train = derive_struct_from_hash(pre)
    struct_rm = derive_struct_from_hash(rem)
    struct_post = derive_struct_from_hash(post)
    struct_ik = derive_struct_from_hash(ikaros)

    # Train reservoir on daedalus hardened structure (10 seeds)
    seeds = list(range(10))
    g1, weights_list = [], []
    for s in seeds:
        nr, w = train_eval(struct_train, s)
        g1.append(nr); weights_list.append(w)
    g1_med = float(np.median(g1))
    print(f"[GR] G1 daedalus(hardened) NRMSE={g1_med:.4f}", flush=True)

    # G2: ikaros transplant
    g2 = [transplant_eval(w, struct_ik, s) for s, w in zip(seeds, weights_list)]
    g2_med = float(np.median(g2))
    print(f"[GR] G2 ikaros transplant NRMSE={g2_med:.4f} factor={g2_med/g1_med:.2f}x", flush=True)

    # G3: re-measure
    g3 = [transplant_eval(w, struct_rm, s) for s, w in zip(seeds, weights_list)]
    g3_med = float(np.median(g3))
    mo3 = float((struct_train[0] == struct_rm[0]).mean())
    print(f"[GR] G3 remeasure NRMSE={g3_med:.4f} factor={g3_med/g1_med:.2f}x mask_overlap={mo3:.3f}", flush=True)

    # G4: post-reboot
    g4 = [transplant_eval(w, struct_post, s) for s, w in zip(seeds, weights_list)]
    g4_med = float(np.median(g4))
    mo4 = float((struct_train[0] == struct_post[0]).mean())
    print(f"[GR] G4 post-reboot NRMSE={g4_med:.4f} factor={g4_med/g1_med:.2f}x mask_overlap={mo4:.3f}", flush=True)

    res = {
        "hardened_static_keys": list(HARDENED_STATIC_KEYS),
        "hashes": {"daedalus_pre": pre, "daedalus_remeasure": rem,
                    "daedalus_post": post, "ikaros": ikaros},
        "static_hash_matches": {"pre_vs_remeasure": pre == rem,
                                  "pre_vs_postreboot": pre == post,
                                  "pre_vs_ikaros": pre == ikaros},
        "G1_daedalus": {"nrmse_per_seed": g1, "median": g1_med},
        "G2_ikaros_transplant": {"nrmse_per_seed": g2, "median": g2_med, "factor": g2_med/g1_med},
        "G3_remeasure": {"nrmse_per_seed": g3, "median": g3_med, "factor": g3_med/g1_med, "mask_overlap": mo3},
        "G4_post_reboot": {"nrmse_per_seed": g4, "median": g4_med, "factor": g4_med/g1_med, "mask_overlap": mo4},
    }
    G1_THR = 0.70
    res["gates"] = {
        "G1_pass": g1_med <= G1_THR, "G1_value": g1_med,
        "G2_pass": g2_med >= 3.0 * g1_med, "G2_value": g2_med,
        "G3_pass": g3_med <= 1.5 * g1_med, "G3_value": g3_med,
        "G4_pass": g4_med <= 1.5 * g1_med, "G4_value": g4_med,
    }
    res["gates"]["VERDICT"] = "GENUINE_CHASSI_EMBODIMENT_DAEDALUS" if all(
        res["gates"][k] for k in ("G1_pass", "G2_pass", "G3_pass", "G4_pass")) else "FAILED"
    RESULT.write_text(json.dumps(res, indent=2, default=str))
    print(f"[GR] VERDICT: {res['gates']['VERDICT']}", flush=True)
    print(f"[GR] wrote {RESULT}", flush=True)


if __name__ == "__main__":
    main()
