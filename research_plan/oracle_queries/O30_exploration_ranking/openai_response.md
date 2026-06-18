# openai response (gpt-5) — 62s

1) Rank (expected value per effort) and what to run first
- 1) I.1 Hetero-cell ratio: Biggest upside; heterogeneity reliably boosts reservoir richness and stability. 1 day.
- 2) I.5 Programmable VG2 schedules: Cheap way to add multiplicative gating/timescale switching; could unlock large gains. 0.5 day.
- 3) I.3 Lateral inhibition ring: Fast decorrelation/contrast; low code risk. 0.5 day.
- 4) I.2 Hierarchical depth: Known win in RCs; a bit more plumbing. 1 day.
- 5) I.6 Multi-frequency VG2 dither: Interesting but SNR/crosstalk uncertain. 0.5 day.
- 6) I.4 Spike vs analog readout: Primarily energy/latency play; accuracy may drop; do after a winner emerges.

Run first tonight: I.5 (fits a single wake-up, highest “option value” by exposing role-switching/gating). If time remains, add I.3 quick sweep.

2) Missing directions to add
- Topology:
  - Delay-augmented graphs (random edges with explicit delays).
  - Low-rank + sparse recurrent (W = UVᵀ + S) and orthogonal/near-orthogonal W for controlled spectral radius.
  - Fractal/H-tree modular hierarchy; ring-of-rings with sparse inter-ring skips.
  - E/I-balanced directed graphs with enforced sign structure.
- Plasticity:
  - Oja/BCM with homeostasis; intrinsic plasticity (gain/leak adaptation to target activity).
  - STDP/anti-STDP or short-term plasticity (STD/STF) emulation.
  - FORCE/RLS readout (on-line) and simple LMS; structural rewiring at low rate.
- Input encoding:
  - Multiplicative input on VG2/body (modulatory channel) + additive on VG1.
  - Tapped-delay input (multi-tau FIR fan-in) and random Fourier/wavelet features per subpopulation.
  - Delta/event encoding (only changes drive pulses).
- Readout:
  - Temporal multi-tap readout (windowed or exponential pools); gated/attention readout over subsets.
  - Nonlinear readout head (kernel ridge or shallow MLP) for ablations; early-exit thresholding for energy.

3) Phase IV chip components to prioritize
- First: IV.2 Per-cell programmable body-leak. Direct control of memory constant, stabilizes against latch-up; highest leverage across tasks.
- Second: IV.1 Active C-coupling. Adds short-term dynamics (STD/STF-like), synergizes with I.5/I.6.
- Third: IV.5 Quasi-2D body split. Gives asymmetry/bias control; modest area, high robustness to PVT.
- Defer: IV.4 Two-NPN (incremental gain), IV.3 Lateral-inhibition crossbar (area-heavy; emulate in topology first).

4) One killer benchmark
- Streaming keyword spotting (Speech Commands v2, 12-class) with always-on, <50 µJ/inference at <20 ms latency, robust to noise domain shift (on-line adaptation allowed at readout). Widely recognized, energy-latency where NS-RAM should excel.

5) Provocative ideas
- Frequency-division multiplexed virtualization: Run K virtual reservoirs on one fabric via orthogonal VG2 carriers; demod in readout. If clean separation holds, you get parallel tasks/time-scales on one die.
- Closed-loop edge-of-chaos bias servo: On-chip monitor drives VG2/leak to maintain target activity/spectral radius. If stable, it solves PVT drift and auto-tunes for tasks.

Top-3 to execute first
- I.1 Hetero-cell ratio
- I.5 Programmable VG2 schedules
- I.3 Lateral inhibition ring
Wildcard: FDM virtualization (above).
