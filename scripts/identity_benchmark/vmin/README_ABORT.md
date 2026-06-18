# Track 2 — Vmin fault-cofit — ABORTED at Stage 0

**Date**: 2026-05-30
**Status**: Aborted to design-doc only. **No undervolt was attempted.**
**Reason**: No safe user-space undervolt mechanism on Zen 5 Strix Halo.

## What is here

- `00_preflight.py` — read-only mechanism enumeration (run anywhere; never writes any HW knob).
- `README_ABORT.md` — this file.

## What is NOT here (and why)

- No Vmin discovery sweep. The design doc proposed `wrmsr 0x1B0` for AMD VID;
  MSR 0x1B0 is Intel `IA32_THERM_INTERRUPT`, not an AMD voltage register.
  Real AMD VID writes on Zen 5 require the SMU mailbox via `ryzen_smu`.
- No `ryzen_smu` load. The module supports Strix Halo (`smu.c:360`) but
  MEMORY.md (UMR safety section) records "NEVER write to SMU mailbox →
  Data Fabric Sync Flood → instant reboot" and the campaign has already
  hard-crashed ikaros twice from related operations. A crash here would
  also defeat the safety contract ("ALWAYS restore baseline voltage") because
  a sync-flood reboot doesn't run cleanup.
- No `ryzenadj` use. Same risk class (SMU mailbox under the hood).
- No fault-map collection, no MLP training, no transplant matrix.

## Resumption conditions

This track can be reactivated when one of:

1. A vendor-blessed user-space VID API for Strix Halo appears (none in
   2026-Q2; checked).
2. Operator accepts a hard reboot risk budget, has a serial console attached
   for post-crash recovery, and runs from a read-only test FS so a sync-flood
   reboot cannot corrupt data.
3. Run is moved to a sacrificial board (not the primary research workstation).

## Cross-track status

Track 1 (VCEK permutation) owns the primary discovery attempt and is not
blocked by this abort.
