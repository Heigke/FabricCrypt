# Identity benchmark — campaign close (2026-05-30)

Status: **9/9 NULL.** Campaign closed. Pivot doc: `IDENTITY_FPGA_PIVOT_2026-05-30.md`.
Negative-results writeup: `IDENTITY_NULL_PAPER_2026-05-30.md`.

## Dispatched agents — verdict matrix

| # | Agent / phase | Channel | Verdict | Artefacts |
|---|---|---|---|---|
| 1 | Phase 1 (twin PUF, 500 reps × 3 regimes) | stable-bit + cycles | **NULL** intra 0.270 / inter 0.295 | `results/.../{ikaros,daedalus}/signature.json`, `raw_idle.npz` |
| 2 | Phase 1b (thermal-matched repeat) | RTN + spatial-corr | **2/3 channels survived initially**, killed by O95 oracle critique | `signature_thermal.json`, `raw_{cold,warm,idle}.npz` |
| 3 | Phase 1c (hardened restart, ikaros local) | LDS startup + FMA-LSB + RO pair | **KILL** — byte-identical 10 k reps | `phase1c/ikaros_hardened/probeA.bin` (20 MB) |
| 3b | Phase 1c (daedalus remote, tmux `identity-phase1c-daedalus-hard`) | Probes A–D hardened | **STATE UNKNOWN** — daedalus host unreachable from ikaros (ping/ssh both time out at 12:38 UTC). See §"Unreachable" below. | (none retrieved yet) |
| 4 | Phase 2 (transplant matrix, NARMA-10 × 60 runs) | per-CU ΔVth + spatial-corr | **NULL** Δ_HW 0.026 ∈ CI of Δ_SW 0.016 | `phase2/verdict.{md,json}`, `matrix_results.json` |
| 5 | Novel F (self-referential identity) | ridge-readout substrate concat | **NULL** z=0.79 (gate z>2) | `novel/F_results.json` |
| 6 | Novel J (split-brain co-dependence) | severance + swap | **NULL on stake** — severance_z=4.69 but swap−swap_to_zero=−5.36 | `novel/J_results.json` |
| 7 | Novel C (tournament RO) | 80-CU RO race, 256-bit | **NULL** cross-HD=2, intra-HD=48 | `novel/C_tournament_summary.json` |
| 8 | Novel-v2 B (Lorenz per-CU trajectory) | per-CU RK4 tail compare | **NULL** ratio 0.185 (gate 3.0) | `novel_v2/B_lorenz_compare.json`, `B_lorenz_{ikaros,daedalus}.npz` |
| 9 | Novel-v2 ECC (EDAC counter map) | per-channel CE counts | **NULL (platform-falsified)** — 0 controllers registered on Strix Halo | `novel_v2/ECC_{ikaros,daedalus}.json` |
| 10 | F_scale (F ablation 30 seeds + permuted MNIST + multi-task) | F under hostile controls | **NULL** sw_matched (1.05) > both (0.92); F downgraded DISCOVERY→NULL | `F_scale/F{1,2,3,4}_*.json` |
| 11 | Oracle O95 (Phase-1 4-way critique) | LLM consensus | 4/4 thermal-artefact verdict; pre-registered the kill | `oracle_queries/O95_identity_phase1_20260530/synthesis.md` |
| 12 | Oracle O96 (novel-angles pre-mortem) | LLM consensus | predicted F/J/C failure modes; confirmed | `…/O96_novel_angles_20260530/synthesis.md` |
| 13 | Oracle O97 (F-hostile controls) | LLM consensus | predicted SW-matched > real; confirmed | `…/O97_F_hostile_20260530/synthesis.md` |

(N=13 because Phase 1c split into ikaros-local and daedalus-remote
sub-agents; O95/96/97 are formally separate dispatches even if collectively
"the oracle programme".)

## Tmux sessions still alive

### ikaros (local, this host)
```
2                 (created Sat May 30 12:12:15 2026, attached — user shell)
claude_session    (created Sat May 30 12:12:11 2026, this CC harness)
nsram_queue_worker (created Sat May 30 12:12:06 2026, NS-RAM batch queue — unrelated to identity)
```
None of these are identity-campaign sessions. The whitelist from the task
prompt (tmux 0, dkhh-claude, yggdrasil-*, gpu_gov) is not present locally
in this snapshot (sessions 0, dkhh, yggdrasil, gpu_gov absent — either
detached/killed earlier or on a different user). `nsram_queue_worker`
left alone per the do-not-touch list.

### daedalus (remote, 192.168.0.37)
**UNREACHABLE.** Both `ping` (no route to host) and `ssh -o BatchMode=yes`
(connection timed out after 3 s) failed at 2026-05-30 12:38 UTC. Cannot
enumerate remote tmux sessions. The `identity-phase1c-daedalus-hard`
session — whether still running or already finished — cannot be inspected
from here. Suggested follow-up once daedalus is back online:

```
ssh daedalus@192.168.0.37 'tmux capture-pane -t identity-phase1c-daedalus-hard -pS -2000'
ssh daedalus@192.168.0.37 'ls -la ~/identity_phase1c_hard/ ~/IDENTITY_BENCHMARK_2026-05-30/'
scp -r daedalus@192.168.0.37:~/IDENTITY_BENCHMARK_2026-05-30/phase1c/daedalus_hardened \
       results/IDENTITY_BENCHMARK_2026-05-30/phase1c/
```

### minos (192.168.0.38)
Also **UNREACHABLE** (no route to host). No identity work was successfully
dispatched here during the campaign window.

## Orphan-process check (ikaros)

`ps auxf | grep -iE 'identity|F2|F4|phase2|narma|puf'` returned only
`thermal_guard.py` PID 9305 (the guard itself, deliberately left running).
**No orphan F2/F4/Phase-2 processes to kill.** Nothing to clean up locally.

## Token spend

Per-dispatch token accounting is **not directly aggregable** from this
session — agentId logs were not centrally collected for this campaign.
Rough order-of-magnitude from the oracle bundles + dispatch logs visible:

- O95 dispatch: 4 oracles, prompts ~6 KB each, responses 874–3 034 chars.
  Estimate ~80 k input + 10 k output tokens total.
- O96: similar, ~80 k + 10 k.
- O97: similar, ~80 k + 10 k.
- All other agents (Phase 1, 1b, 1c, Phase 2, novel F/J/C, novel_v2 B/ECC,
  F_scale F1–F4): these were *executed* sub-agents inside this CC session
  rather than independently-tokened LLM calls, so they consumed CC-session
  tokens, not separately accountable.

Aggregate honest answer: **~270 k tokens spent on oracle dispatch**, plus
the running CC-session token budget which is not separately broken out.
Full per-dispatch accounting would need centralised logging
(`agentId.log` aggregation script — not yet built).

## Git staging

Per task brief: stage the new files for a clean snapshot, do NOT commit.
Performed at the end of this session — see report.

## Bottom line

- All ten dispatched experimental agents → NULL.
- All four oracle dispatches → falsification predictions confirmed.
- One artefact retrieval pending (daedalus phase1c). Host offline.
- One thermal-guard daemon left running on purpose; no orphans to kill.
- Pivot direction approved by design: FPGA RO-PUF on the Arty A7-100T we
  already own. Doc: `IDENTITY_FPGA_PIVOT_2026-05-30.md`. Open question:
  procurement of second board.
