# Five verified HAL-bypass per-chip signals (replicated, 24 h apart, two physical machines)

Both machines: identical AMD Strix Halo SKU (gfx1151), identical microcode `0xb70001e`, identical kernel, identical git tree, identical binary. The five signals separate the dies; software ABI cannot.

| # | Probe                                         | inter-machine KS-D | intra-machine KS-D | ratio  |
|---|-----------------------------------------------|--------------------|--------------------|--------|
| 1 | `nanosleep(0)` latency distribution            | 0.7224             | 0.0152             | ~47×   |
| 2 | `sched_yield()` latency distribution           | 0.9931             | 0.0222             | ~45×   |
| 3 | inter-core cache-line ping-pong p50            | 0.9118             | small              | huge   |
| 4 | RDTSC offset between same-package cores        | 0.91               | small              | huge   |
| 5 | DRAM refresh-window timing pattern             | ~0.9               | small              | huge   |

## Properties

- **Replicated** across 24 h drift gap. Signature_v2 drift p95 = 0.19.
- **Constitutive A-vs-C swap gate PASS** — transplant the wrong-chip signature into a chip-conditioned model and NRMSE inflates by 57.9%.
- **No firmware modification, no PSP/SMU privilege, no microcode mod.** Pure userspace probes on stock Linux.
- **HAL-bypass:** all five probes run below or around the OS abstraction layer where the dies are supposed to be identical-by-design. The dies are not identical-by-design, they are identical-by-spec; physics says otherwise.
- **290-dim fused signature_v2** achieves 100% LOO classification accuracy between the two dies.

## Crypto-binding (Phase 14C)

The 290-dim signature is HMAC-mixed with a fresh 64-bit verifier nonce. Seven attack scenarios tested (honest_own, peer-impersonate, static-replay-no-nonce, static-replay-correct-nonce, dynamic-replay against 400-sig library, nonce-mismatch, honest-with-wrong-nonce). **All 7 gates PASS** at pre-registered thresholds. See `phase14c_spoof_v2.json`.

This is the same primitive that PCC / NVIDIA CC build their attestation stories on top of, but here it is achieved on commodity hardware with no vendor cooperation.
