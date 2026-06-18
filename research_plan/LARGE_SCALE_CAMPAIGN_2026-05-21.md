# Large-scale v5.5 NS-RAM campaign — 2026-05-21

Best calibrated model: **v5.5** (K1+ALPHA0+Tlpe1 fixes, DC=0.461 dec, fwd+bwd n=66).
Pushed to `Heigke/NSRAM @ backup-snapback-research-2026-05-15` (commit `0643b26d`).

## Compute available
- **ikaros** (local, AMD Radeon 8060S gfx1151 ROCm) — thermal-bounded, no FPGA loop
- **zgx** (192.168.0.41 naorw, NVIDIA GB10) — primary GPU host
- **daedalus** (192.168.0.37) — SSH refused, awaiting manual power on
- **minos** (192.168.0.38) — down, awaiting power on

## Jobs

| ID | Sim | Host | Est | Status |
|----|-----|------|-----|--------|
| J1 | Surrogate v5.5 regen (100K pts, Tlpe1 ON) | ikaros | 1-2h | dispatched |
| J2 | NARMA-10 scaling N∈{200,1k,4k,16k} × 30 seeds | zgx | 2-4h | dispatched |
| J3 | Topology zoo at N=10k (ER/MESH/SW/RING × 10 seeds × 3 ρ-norms) | zgx | 3-6h after J2 | queued |
| J4 | Cross-task batch (HDC, KWS, NARMA, LIF, STDP) at 5-10× current N | zgx | 4-8h after J3 | queued |
| J5 | Topology zoo replication | daedalus | 3-6h | blocked on machine |

## Acceptance gates (pre-registered)
- J1: surrogate vs forward_2t ≤ 0.39 dec across 32 reservoir biases (matches brief)
- J2: any N reaches NRMSE < 0.55 → DISCOVERY (beats current 0.612)
- J3: ER_SPARSE wins in ≥2/3 ρ-norm conventions → CONFIRMS brief
- J4: any use-case improves ≥5pp at 5-10× N → DISCOVERY

NO-CHEAT: fwd+bwd in every DC, seed ≥10 in any aggregate, results path quoted in claims.
