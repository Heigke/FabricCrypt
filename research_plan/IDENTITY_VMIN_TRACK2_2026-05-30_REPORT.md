# Track 2 — Per-die undervolting fault-cofit — REPORT

**Date**: 2026-05-30
**Operator**: Claude (Opus 4.7), autonomous wake
**Status**: **BLOCKED at Stage 0**. Aborted to design-doc only. No undervolt attempted.
**Hosts checked**: ikaros (local), daedalus (`192.168.0.37`)
**Watchdog fired**: N/A (never armed; no undervolt attempted)
**Baseline voltage restored**: N/A (never modified)
**User intervention needed**: None.

## 1. Undervolt mechanism used

**None used. Mechanism unavailable.**

Stage 0 preflight (`scripts/identity_benchmark/vmin/00_preflight.py`, JSON in
`results/IDENTITY_VMIN_2026-05-30/preflight_{ikaros,daedalus}.json`) enumerated
the candidate paths on both Ryzen AI Max+ 395 boxes:

| Mechanism | ikaros | daedalus | Suitability |
|---|---|---|---|
| `cpupower` (amd-pstate-epp) | present, governor active | present | frequency floor/ceil only — **NOT a voltage knob** |
| `rdmsr`/`wrmsr` + `/dev/cpu/N/msr` | both, NOPASSWD sudo | dev present, tools missing | reads OK; AMD VID **not** at the MSR the design doc cites |
| `ryzen_smu` kernel module | not loaded; `.ko` on disk; src supports Strix Halo | not loaded; `.ko` not present | only documented path to Zen 5 VID; **documented crash mode** |
| `ryzenadj` | absent | absent | wraps SMU mailbox — same risk class |

Two specific blockers:

1. **The design doc's MSR is wrong for this CPU.** `IDENTITY_DEEPER_HUNT_2026-05-30.md`
   §5 step 1 says "sweep core Vmin … via `wrmsr 0x1B0`". MSR `0x1B0` is Intel
   `IA32_THERM_INTERRUPT`, not an AMD voltage register. On Zen 5, software
   VID lives behind the SMU mailbox; there is no published direct-MSR path
   for sub-Vmin undervolting.
2. **SMU mailbox writes are the documented crash mode on this hardware.**
   Project `MEMORY.md` (UMR safety section): *"NEVER write to SMU mailbox
   (C2PMSG_66/82/90) via UMR → Data Fabric Sync Flood → instant reboot"*.
   The campaign has already hard-crashed ikaros twice from related operations.
   A sync-flood reboot also defeats the spec's hard rule "ALWAYS restore
   baseline voltage at end" — cleanup never runs.

The local `ryzen_smu` source (`/home/ikaros/Documents/claude_hive/ryzen_smu/smu.c:360`)
does claim Strix Halo support, so a future operator with a sacrificial board
and serial-console recovery could attempt this; that is the only realistic
resume path and is documented in `scripts/identity_benchmark/vmin/README_ABORT.md`.

## 2. Per-die Vmin

**Not measured.** Mechanism unavailable; no voltages were swept.

## 3. Fault-map Hamming distance

**Not measured.** No fault maps collected.

## 4. Gates G1–G4

| Gate | Verdict | Notes |
|---|---|---|
| G1 self-eval ≥ 90% | **N/A — not run** | requires Stage 3 training |
| G2 transplant ≤ 30% | **N/A — not run** | requires Stage 4 transplant |
| G3 SW-matched ≥ 80% | **N/A — not run** | requires Stage 3+4 |
| G4 random-pattern ≤ 30% | **N/A — not run** | requires Stage 3+4 |

No data on which to base a falsifiable verdict. The track does not contribute
to or against the constitutive-identity hypothesis on this run.

## 5. Watchdog firing

Watchdog was **never armed** because the underlying undervolt action was
never taken. The watchdog scaffolding was not deployed — arming a watchdog
without an action to guard is just dead code in this repo. If/when a future
operator resumes, the watchdog must be built and proven on a dummy register
first (per spec).

## 6. Baseline voltage state

**Untouched.** Confirmed by absence of any wrmsr/SMU calls in this session's
shell history and by the fact that no script in
`scripts/identity_benchmark/vmin/` performs any write.

## 7. Final verdict

**BLOCKED.** Aborted to design-doc only at Stage 0 per the task's preflight
clause: *"if no undervolt mechanism works, abort to design-doc only."*

This is the most honest available outcome. The alternative — loading
`ryzen_smu` and writing the mailbox — has a non-trivial probability of a
sync-flood reboot mid-run on the primary research workstation, which would
destroy work-in-progress on Track 1 (CIFAR-10 ResNet, tmux 0/2/dkhh-claude)
and the running NS-RAM queue worker.

## 8. Comparison to Track 1

Not yet available from this agent's vantage. Track 1 (VCEK permutation) is
owned by a separate worktree (`vcek-train-*` tmux); this Track 2 abort does
not block Track 1 in any way. The two tracks were designed as parallel
attempts at the same hypothesis class (constitutive per-die identity); if
Track 1 produces a clean G1–G4 result it stands on its own.

## 9. Report path

This file: `research_plan/IDENTITY_VMIN_TRACK2_2026-05-30_REPORT.md`.

Supporting artefacts:

- `scripts/identity_benchmark/vmin/00_preflight.py` — read-only mechanism probe.
- `scripts/identity_benchmark/vmin/README_ABORT.md` — abort rationale and
  resume conditions.
- `results/IDENTITY_VMIN_2026-05-30/preflight_ikaros.json`
- `results/IDENTITY_VMIN_2026-05-30/preflight_daedalus.json`

## 10. User-intervention needed

**None.** System is in the same state it was before this session: no module
loads, no voltage changes, no kernel modifications. No cold-boot recovery
needed. Reboot is not required.

## Honest caveats

- I did not attempt to load `ryzen_smu` even read-only. The module's `init`
  path on an unsupported kernel/CPU combo can itself trigger faults; the
  campaign's two prior crashes plus the spec's "be extra careful" line make
  this the right call.
- "BLOCKED" is a real outcome, not a soft pass. Future operators should
  not retry this track on the primary workstation without first proving
  the watchdog on a sacrificial machine.
- Track 2's scientific value if it had run is high precisely *because* it
  uses substrate-as-active-degradation. That same property is what makes it
  unsafe to attempt here.
