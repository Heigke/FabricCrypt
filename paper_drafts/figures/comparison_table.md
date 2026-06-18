# TPM is Not Enough — Attestation Primitive Comparison

Comparison of FabricCrypt against existing hardware-attestation and device-identity primitives.

Legend: **Y** = yes, **N** = no, **P** = partial / conditional.

| Primitive | Per-die identity? | Vendor PKI required? | Special hardware? | Replay-resistant? | Static-binary independent? | Liveness? | Open-source verifiable? | AI-inference-bound? |
|---|---|---|---|---|---|---|---|---|
| TPM 2.0 [1]                                  | P (EK cert) | Y | Y (TPM chip / fTPM) | P (nonce in Quote, but binary trust) | N (PCR = static measurement) | N | P (spec open, keys vendor-signed) | N |
| Intel SGX [2]                                | P (provisioning key) | Y (IAS / DCAP) | Y (SGX-capable CPU) | P (REPORT nonce) | N (MRENCLAVE static) | N | N (proprietary uCode) | N |
| AMD SEV-SNP [3]                              | P (VCEK per chip)   | Y (KDS)         | Y (Zen3+ SEV-SNP)   | P (REPORT_DATA nonce) | N (launch measurement static) | N | P (spec open, keys vendor-signed) | N |
| Intel TDX [4]                                | P (per-platform TDX seam key) | Y | Y (Sapphire Rapids+) | P (REPORTDATA nonce)  | N (TDREPORT static)        | N | P | N |
| Apple Secure Enclave / PCC [5]               | P (UID per die) | Y (Apple) | Y (SEP)             | P                | N                          | N | N (closed) | N |
| NVIDIA Confidential Compute Manager [6]      | P (per-GPU ECID) | Y (NVIDIA NRAS) | Y (H100+)         | P                | N (driver/firmware measurement) | N | N (closed firmware) | N |
| DRAM PUF (Kim 2018) [7]                      | Y (DRAM cell variance) | N | N (commodity DRAM) | P (challenge-response) | Y | N (one-shot static fingerprint) | P (academic) | N |
| DRAWNAPART (NDSS'22) [8]                     | Y (GPU shader timing) | N | N (commodity GPU) | N (deterministic shader trace) | Y | N | Y (academic) | N |
| **FabricCrypt (ours)**                       | **Y** (analog fabric signature) | **N** | **N** (commodity APU/GPU) | **Y** (nonce-bound inference) | **Y** | **Y** (live inference required) | **Y** (open scripts + frozen seeds) | **Y** (signature is bound to model inference) |

## Source citations

- [1] Trusted Computing Group, *TPM 2.0 Library Specification* (Rev 1.59, 2019). Quote command: §23.16; EK certificate trust roots: §B.2.
- [2] Costan & Devadas, *Intel SGX Explained* (IACR 2016/086); Anati et al., *Innovative Technology for CPU Based Attestation and Sealing* (HASP 2013).
- [3] AMD, *SEV Secure Nested Paging Firmware ABI Specification*, Rev 1.55 (2023), §7 (ATTESTATION_REPORT), §8 (VCEK).
- [4] Intel, *TDX Module 1.5 ABI Spec* (2023), §TDG.MR.REPORT; *TDX DCAP Quote Generation* whitepaper (2022).
- [5] Apple, *Apple Platform Security Guide* (May 2024), "Secure Enclave" and "Private Cloud Compute" sections.
- [6] NVIDIA, *Confidential Computing on NVIDIA H100 GPUs* whitepaper (2023); NRAS attestation service docs.
- [7] Kim et al., *The DRAM Latency PUF: Quickly Evaluating Physical Unclonable Functions by Exploiting the Latency-Reliability Tradeoff in Modern Commodity DRAM Devices*, HPCA 2018.
- [8] Laor et al., *DRAWNAPART: A Device Identification Technique based on Remote GPU Fingerprinting*, NDSS 2022.
- [9] **FabricCrypt (this work)**: Phase 13–14C, `results/IDENTITY_BENCHMARK_2026-05-30/`. Per-die signature from analog GPU/APU telemetry under nonce-controlled inference workload.

## Headline differentiators

1. **No vendor PKI** — verification needs no trust root other than the published, frozen scripts.
2. **AI-inference-bound** — the signature only exists during live model inference, defeating static replay.
3. **Liveness** — verifier nonce gates the workload; pre-recorded telemetry is rejected (Phase 14C: static replay 0.6%, dynamic replay 1.2%).
4. **Open-source verifiable** — every script, seed, and signature is in the repo.
