# H7 — Substrate-Rooted AI Identity

Binding a language model to **one specific physical chip's live hardware signals**, so that its
text generation depends on that die's real-time substrate. Run on AMD Strix Halo (gfx1151).

**The goal:** an LM that writes coherent text when fed its *own* die's live hardware signal, but
breaks when fed a *different real chip's* signal, a statistics-matched spoof, or a temporally
shuffled version — and, ideally, whose output *style* is a graded function of the live signal.

This bundle is a self-contained snapshot of the code, results, and write-ups so others can build
on the findings. It is research-in-progress, reported **honestly including what failed**. It is a
**device-binding** demonstration, **not** a claim of machine consciousness.

## What's here

- `scripts/identity_benchmark/` — the pipeline: substrate sampler, FiLM+LoRA embodied model,
  trainers (v8→v12), the cross-die and graded probes, per-die stats generator, thermal watchdog.
  See `scripts/identity_benchmark/README.md` for the full map, metrics, and run instructions.
- `results/IDENTITY_H7_2026-06-09/` — per-run JSONs, plots, and the write-ups:
  - `V12_RESULTS_2026-06-11.md` — latest: graded coupling + both-ways cross-die 2×2
  - `V10_FALSIFICATION_SYNTHESIS_2026-06-10.md` — falsification battery + 4-oracle critique

## Headline findings

- **Anti-spoof / anti-shuffle dependence is robust** on both chips tested: the model's own real
  live dynamics → coherent text; wrong statistics (knockoff) or wrong temporal order (shuffle) →
  output breaks by **10³–10¹¹×** perplexity.
- **Graded coupling confirmed** (one chip): output entropy is a continuous function of the live
  channel dynamics, Pearson **r = +0.914**, collapsing to **−0.22** when the dynamics are shuffled.
- **Cross-die break is real but asymmetric** (3/4 of the 2×2): one chip's model breaks ~10× on the
  other chip's live signal; the reverse model stays coherent. Part of the break is a normalization /
  DC-operating-point effect. Making it fully symmetric and dynamics-based is the open problem (and
  the path toward a "cryptographic" device lock — see `V12_RESULTS_2026-06-11.md`).

## Not included

Model checkpoints (`*.pt`), raw substrate captures (`*.npz`), and logs are excluded (size). The
scripts regenerate them; see the reproduce section in `scripts/identity_benchmark/README.md`.

## Hardware / safety

Requires an AMD gfx1151 APU, `HSA_OVERRIDE_GFX_VERSION=11.0.0`, and root (substrate reads
`/dev/mem` / SMN). The APU has a 99 °C ACPI trip → always run the included `thermal_watchdog.sh`
alongside training. Substrate access is read-only.
