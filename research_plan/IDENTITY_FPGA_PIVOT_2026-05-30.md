# Identity benchmark — FPGA pivot design doc

Date: 2026-05-30
Status: **DESIGN ONLY** — no build/bitstream work yet. Awaiting user approval.
Precursor: `IDENTITY_NULL_PAPER_2026-05-30.md` (9/9 NULL on gfx1151 user-space).

## 1. Hardware on hand

Per project memory (`fpga_nsram.md` + `hw_integration.md`):

- **Board**: Digilent **Arty A7-100T** (Xilinx Artix-7 `xc7a100tcsg324-1`).
- **Toolchain**: Vivado 2025.2 at `/opt/Xilinx/2025.2/Vivado/bin/vivado`,
  build via `scripts/build_vivado.tcl`, program via `openFPGALoader -b arty_a7_100t`
  or `fpga/scripts/program.tcl`.
- **Current bitstream**: `fpga/output/nsram_eth_top.bit` — 128-neuron NS-RAM
  reservoir bank with UDP Ethernet host interface (192.168.0.50:7700).
  Utilisation 7994 LUTs / 12 687 FFs / 9 BRAM / 10 DSP (12.6 % LUT, 4.2 % DSP).
  WNS +0.224 ns clean. Plenty of headroom for PUF instrumentation.
- **Host bridge**: `fpga_host_eth.FPGAEthBridge` (UDP, 0.82 ms RTT,
  1224 Hz max telemetry) and `fpga_host.FPGABridge` (UART fallback).
- **Twin question**: we own one Arty A7-100T. Whether a second is available
  (or borrowable) is the single biggest open question — see §6.

## 2. Why FPGA fixes the gfx1151 NULL

The nine attacks failed for one reason: the AMDGPU driver and ROCm runtime
homogenise the abstraction layer. Concretely —

| gfx1151 failure mode | FPGA equivalent |
|---|---|
| LDS zero-init on launch (ROCm 6.3+) kills SRAM-startup PUF | BRAM/LUT-RAM content under designer control; we can *choose* not to zero it |
| Per-CU clocks centrally governed → no RO-race fingerprint | Explicit ring-oscillator instantiation per LUT (Suh & Devadas 2007 reference circuit) with hard placement constraints in XDC |
| FMA pipeline deterministic → byte-identical across packages | Routing-delay variance is the *primary* PUF mechanism; deterministic from logic, stochastic from silicon |
| ECC counters not exposed for unified APU memory | Direct FPGA fabric — no ECC layer to hide behind |
| RTN/spatial-corr swamped by thermal | Single small die, controllable clock domains, can run under climate chamber |

The FPGA is precisely the substrate where the orthodox PUF literature
(Suh+Devadas 2007, Maiti+Schaumont 2010, Herder et al. 2014) was developed
and where it *works*. We are not gambling on a novel substrate — we are
returning to the textbook one.

## 3. Three phased experiments

All gates pre-registered identically to the gfx1151 protocol so cross-substrate
comparison is direct.

### Phase F1 — Stock RO-PUF
**Goal**: confirm a textbook PUF works on our specific Arty A7-100T board
before doing anything fancy.

- **Design**: 16–32 ring oscillators per "group" × 16 groups (256–512 ROs
  total). Each RO = 3 to 5 inverters in a feedback loop + 16-bit counter
  + arbiter. Hard-placement via XDC `LOC` constraints on adjacent CLBs.
  Reference circuit: Suh & Devadas 2007 Fig. 5.
- **Protocol**: 64 challenges → 64 pairwise RO frequency comparisons →
  64-bit response per device. Repeat N = 500 across three thermal regimes
  (room, fan off, climate-warmed) and three voltage regimes (nominal,
  Vccint −5 %, Vccint +5 % via Vivado-supported overdrive if board allows).
- **Metrics**: intra-Hamming (within board, across reps), inter-Hamming
  (across boards — needs second board, see §6), bit-reliability
  (% bits with intra-HD < 5 % flip).
- **Gate** (identical to gfx1151 stable-bit, item 1 of NULL paper):
  intra-HD ≤ 0.10 AND inter-HD ≥ 0.40 → DISCOVERY.
- **Estimated wall**: 1 day Vivado build, 2 hours data collection per board.

### Phase F2 — PUF response as reservoir substrate signature
**Goal**: feed the per-board PUF bits into the 128-neuron NS-RAM reservoir
as a constitutive substrate signal, then run the *same* NARMA-10 transplant
matrix we ran on gfx1151 (Phase 2 of the null paper).

- **Design**: extend `nsram_eth_top` so each of the 128 neurons reads a few
  bits of the PUF response as a per-neuron Vg offset or per-synapse weight
  bias. The PUF bits are baked into the running configuration *of that
  specific board* — transplanting a trained ridge readout from board A to
  board B means the substrate signal in B is the wrong signal.
- **Task**: NARMA-10, T_train = 2000, T_test = 500, 10 seeds, 4 conditions
  (HW_train_HW_eval, HW_train_SW_eval, SW_train_HW_eval, SW_iid control,
  SHUFFLE control). Mirror Phase 2 verdict table format exactly.
- **Gate** (identical to gfx1151 Phase 2): Δ-NRMSE(HW transplant) >
  Δ-NRMSE(SW-iid) + 2 σ AND > Δ-NRMSE(SHUFFLE) + 2 σ → DISCOVERY.
  This is the test that gfx1151 failed at 0.026 vs 0.016/0.014.
- **Estimated wall**: 2 days RTL extension + sim, 1 day FPGA timing closure,
  1 hour data collection per board.

### Phase F3 — Transplant matrix
**Goal**: confirm the PUF signature is identity-bearing by either (a) running
on two boards or (b) artificially "transplanting" via voltage/temperature
shifts on one board.

- **F3a (preferred — two boards)**: Run identical NARMA-10 readout-training
  on board A, evaluate on board A and board B. Repeat reversed. 2 × 2
  matrix. Expected: diagonal NRMSE < off-diagonal NRMSE if PUF is load-bearing.
- **F3b (single-board fallback)**: Use **temperature** (24 → 55 °C via
  heat-gun or climate box) and **voltage** (Vccint ± 5 %) as
  pseudo-twins. The PUF response *should* be bit-stable across these regimes
  (that is the point of a PUF — that is what the bit-reliability filter in
  F1 selects for). So in F3b we are testing intra-device stability under
  hostile environment, not inter-device transplant. Weaker but still
  publishable as a stability result.
- **Gate**: F3a — off-diagonal NRMSE > diagonal NRMSE by > 2 σ AND scales
  with Hamming distance between PUF signatures. F3b — bit-reliability ≥ 95 %
  across regime span; reservoir performance degradation ≤ 5 % NRMSE.
- **Estimated wall**: F3a 4 hours (mostly cable juggling). F3b 1 day
  (climate ramp protocol).

## 4. Pre-registered gates (summary table)

| Phase | Gate | Threshold | Falsifier |
|---|---|---|---|
| F1 | intra-HD, inter-HD on RO-PUF | intra ≤ 0.10, inter ≥ 0.40 | inter ≈ intra → RO routing is too uniform on Artix-7 |
| F2 | NARMA-10 transplant Δ vs SW-iid | Δ_HW > Δ_SW + 2 σ | Δ_HW ≤ Δ_SW → same brittle-ridge effect we saw on gfx1151 (would invalidate the entire approach, not just gfx1151) |
| F3a | 2-board off-diagonal NRMSE | off > diag + 2 σ, monotonic in HD | flat matrix → PUF bits not load-bearing in the reservoir |
| F3b | bit-reliability over T/V sweep | ≥ 95 % bits stable | < 95 % → not actually a PUF, just noise |

Direct comparison with NULL paper: the F2 gate is byte-for-byte the same
test that gfx1151 failed at Δ_HW = 0.026 vs Δ_SW = 0.016 (within CI).
If FPGA shows Δ_HW > 5 σ above Δ_SW, that is the publishable cross-substrate
contrast: *same architecture, same task, same gate; substrate matters.*

## 5. Risk and timeline

| Item | Risk | Mitigation |
|---|---|---|
| RTL build breaks 128-neuron timing closure when PUF added | Medium | Start with 32 ROs (not 256). Keep PUF clock domain separate from neuron clock. WNS already +0.224 ns gives ~2 ns margin. |
| OpenFPGALoader programming flakiness | Low | We already have working `program.tcl` |
| Single-board case has no inter-HD measurement | High (see §6) | F3b voltage/temperature fallback documented above |
| Thermal damage from climate-box ramp on Arty board (commercial-grade silicon, 0–85 °C) | Low–Medium | Cap at 70 °C, slow ramp, surface thermocouple |
| Phase F1 RO-PUF returns intra ≈ inter on Artix-7 (because the literature is older, did most of its work on Spartan-3 / Virtex-5) | Low | Modern literature (Maiti+Schaumont 2010 on Virtex-5; Maes et al. 2009 on Spartan-3) extends comfortably to 28 nm; Artix-7 is 28 nm. |

**Timeline** (with one board, working FEL toolchain, no surprises):

- Week 1: Phase F1 RTL + build + measurement.
- Week 2: Phase F2 RTL extension + transplant runs.
- Week 3: Phase F3a or F3b + writeup.

If RTL needs major rework or board #2 procurement slips: add 1–2 weeks.

## 6. Open question — second board

**This is the gating decision.** F3a (the strongest test, direct
inter-device transplant) requires two physically distinct Arty A7-100T
boards. Options:

1. **Buy a second Arty A7-100T**: ~$300 USD list. 2–4 week shipping.
   Cleanest path. Strongly recommended if budget allows.
2. **Borrow one**: From a partnering lab (KTH? Chalmers?). Unknown lead time.
3. **Use a different board as twin**: e.g. Arty S7-50, Basys 3, Nexys 4 DDR.
   Same Artix-7 family but different SKU → confound between "different die"
   and "different speed grade / package". Weaker but publishable with caveat.
4. **F3b only (single-board)**: Voltage/temperature stability. Strictly
   intra-device. Honest fallback, halves the strength of the result.

**Recommended path**: option 1 if budget approves; otherwise option 4 and
flag the limitation prominently. Do *not* mix Artix-7 SKUs (option 3) —
the confound is fatal.

## 7. What this doc does NOT cover

- We do not build the RTL here. That is the next dispatched task after
  user approval.
- We do not buy hardware. Procurement decision is the user's.
- We do not commit to a publication venue. Likely targets after F1/F2/F3
  complete: IEEE Trans. VLSI Systems (PUF angle), or a workshop paper at
  CHES / HOST / FCCM. The cross-substrate gfx1151-vs-FPGA contrast may
  itself be the most interesting framing for a Nature Comms-style venue.

## 8. Concrete next step (for user approval)

If approved, the immediate dispatched task is:
*"Set up Vivado project for Phase F1: 64 ring oscillators with XDC
placement constraints, FSM to enumerate 64 pairwise challenges, UART
readout of 64-bit response. Build and program. No PUF/reservoir coupling
yet — just confirm the PUF works on our specific board."*

Estimated wall: 1–2 days. Output: `fpga/output/ropuf_phase1.bit` and a
`results/IDENTITY_FPGA_2026-XX/F1_ropuf.json` with intra-HD across N reps.
