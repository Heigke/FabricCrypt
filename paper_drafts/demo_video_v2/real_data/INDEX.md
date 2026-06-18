# demo_video_v2 — REAL data index

Every frame/numerical claim in the v2 demo video must be traceable to one of
the JSON / npz files listed below. **NO synthesis.** The v1 video used per-channel
PRNG-driven waveforms — that is explicitly forbidden here.

Render command:
```
venv/bin/python paper_drafts/demo_video_v2/real_data/signal_traces_real.py
```
Outputs PNGs to `paper_drafts/demo_video_v2/frames_real/` and writes
`manifest.json` here.

## Section → frame → source-of-truth

| Video section | Frame file (in `frames_real/`) | Source file | Real metric used |
|---|---|---|---|
| S1 — 5 signals: syscall p99.9 | `sig1_syscall_p999.png` | `results/IDENTITY_BENCHMARK_2026-05-30/embodiment12/task_D_syscall_{ikaros,daedalus}.json` | `raw_samples_ns_nanosleep0` (n=10000 each), aggregate `nanosleep0.p99_9` annotated |
| S1 — 5 signals: NVMe tail | `sig2_nvme_tail.png` | `embodiment12/task_F_nvme_{ikaros,daedalus}.json` | `raw_samples_ns` (n=10000 each), `nvme_latency.p99_9` annotated |
| S1 — 5 signals: RDRAND | `sig3_rdrand.png` | `embodiment12/task_E_rdrand_{ikaros,daedalus}.json` | `raw_samples_cyc`, `rdrand_cycles.p50` annotated. **Caveat**: governor-mediated; documented in Phase 12 analysis. |
| S1 — 5 signals: TSC inter-core | `sig4_tsc_intercore.png` | `embodiment12b/task_B_{ikaros,daedalus}.json` | per-pair `mean`/`p99` over CPU pairs (C0–C{1,2,4,7,8}) |
| S1 — 5 signals: cache-line ping-pong | `sig5_cacheline.png` | `embodiment12b/task_E_{ikaros,daedalus}.json` | per-pair `mean` over 7 CPU pairs |
| S1 — bonus: DRAM refresh | `sig6_dram_refresh.png` | `embodiment12b/task_G_{ikaros,daedalus}.json` | `walk_ns` + `spike_interval_samples` stat-dict bars; `spike_count` annotated |
| S1 — aggregated fingerprint | `sig_phase13_fingerprint.png` | `embodiment13/{ikaros,daedalus}_sig_v2.npz` | `vec` array (10 reps × 290 features) — the actual signature fed to the embodied head |
| S2 — identity claim numbers | `identity_pre_post_real.png` (left bars) | `paper_drafts/demo_video_v2/real_data/identity_samples.json` (copied from `/tmp/demo_samples.json`) | embodied PRE 40/40 correct, embodied POST 0/40 correct, vanilla PRE 15/40 correct |
| S3 — transplant moment | `identity_pre_post_real.png` (right trace) | same | confidence trace: μ_pre=0.961, μ_post=0.845 — model stays confident but flips host |
| S4 — spoof attack bars | `spoof_bars_real.png` | `embodiment14c/ikaros_spoof_v2.json` | `attacks.{honest_own,daedalus_peer,static_replay_no_nonce,permute_replay,gaussian_proxy,mean_proxy,flip_proxy}.accept_rate`, n_eval annotated |

## Identity samples (Section 2 + 3) — provenance

`identity_samples.json` is the **40 pre + 40 post** captured live during the v1
recording session via repeated `GET /api/identity` calls against
`scripts/demo_embodied_ai/demo_server.py`, with a `POST /api/transplant` in
between. Source path: `/tmp/demo_samples.json` (now copied here so the file is
checked next to the manifest).

Aggregate truth:
- pre  embodied correct: 40 / 40   (host name = ikaros, predicted = ikaros)
- post embodied correct:  0 / 40   (post-transplant, predicted = daedalus)
- pre  vanilla  correct: 15 / 40   (substrate-blind ~ chance)
- pre  embodied confidence mean: 0.961
- post embodied confidence mean: 0.845 (high but WRONG)

If those samples are ever lost, re-capture via:
```
venv/bin/python scripts/demo_embodied_ai/demo_server.py \
  --host-name ikaros --port 8770 \
  --own-sigs  results/IDENTITY_BENCHMARK_2026-05-30/embodiment14b/ikaros_sigs.npz \
  --peer-sigs results/IDENTITY_BENCHMARK_2026-05-30/embodiment14b/daedalus_sigs.npz &
# loop 20 GETs to /api/identity, save to pre[], then POST /api/transplant,
# 20 more GETs, save to post[], then POST /api/restore, kill server.
```

## Audit: synthesis still present in v2?

After this real_data pass, the only generated-not-captured values that should
remain in the v2 video pipeline are:
- text/captions/timing
- dashboard chrome (colors, fonts)
- TTS audio in `paper_drafts/demo_video_v2/audio/`

**No waveform, no histogram, no bar value, no confidence number above is from a
PRNG.** Every figure under `frames_real/` is regenerated deterministically from
JSON/npz on disk.

## Parallel agent hand-off

The v2 build pipeline (parallel agent at branch worktree `paper_drafts/demo_video_v2/`)
should pull frames by basename from `frames_real/`. The signal-trace PNGs replace
v1's `signal_chN_*.png` synthesized traces. Pre/post identity bars replace the
S2/S3 mock dashboard panels.
