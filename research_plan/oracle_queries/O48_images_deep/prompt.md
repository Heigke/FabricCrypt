# O48 — DEEP visual extract from Sebas + Mario images

You are looking at NS-RAM device materials (Sebas Pazos & Mario Lanza, 130 nm
bulk + 180 nm triple-well CMOS). The 22 images attached are slides,
schematics, I-V plots, transient traces, and SPICE comparisons that we have
been receiving over the past weeks. We have already extracted numerics
(parameter values, axes) in a previous pass (O44/O47). **This pass is purely
about SCHEMATIC and TOPOLOGY understanding** — what is wired to what, what
ports/terminals exist, what test setup is implied.

For EACH image (01..22), answer these six points in a markdown subsection:

A. **CIRCUIT TOPOLOGY** — what is shown? (1T cell, 2T cell, full SNN,
   I-V test fixture, parameter-vs-bias plot, etc.) Draw the netlist
   mentally. Which terminal is V_G1, V_G2, V_d, V_b (bulk/well), source?
   If there is a body / bulk contact or floating node, call it out.

B. **TEST CONFIGURATION** — DC sweep? Transient pulse? SNN inference?
   What is the input stimulus, what is being measured?

C. **NUMBERED ELEMENTS not in standard schematics**. Sebas / Mario
   sometimes add things we may have missed: gate-source capacitor C_GS,
   well-substrate capacitor C_WS, body contact, dummy transistors,
   passive pull-downs, p-i-n diode for reset, etc. Identify every
   non-standard element.

D. **PLOT STRUCTURE** — axes, log vs linear, which curves correspond to
   which biases / branches, units, axis ranges.

E. **TEXT / ANNOTATIONS not to miss** — parameter names with units,
   device variant labels (M1, M2, thick-ox, 1T, 2T, p-diode), process
   node indicators (130 nm, 180 nm), dates, version markers, hand-written
   notes.

F. **CROSS-REFERENCE** — does this image agree or disagree with the other
   slides in the packet? E.g., does the 2T cell schematic (image 18) match
   the LTSpice netlist that one would expect from images 13, 14? Does the
   p-diode in image 21 match anything in image 10 or 17?

Special note for `22_sebas_2026_05_02_image-2.png` — this is the NEW image
from Sebas (2 May 2026), likely the source PNG that was hand-extracted into
`three_branch_params_extracted.json`. Describe its plot in detail:
- panels, axes, what's overlaid
- which "three branches" are which (we suspect a parameter (NFACTOR? ETAB?
  BETA0?) vs V_G2 with three curves for different V_G1 values, or vice versa)
- any annotations that pin down M1 vs M2, or branch identity

After the per-image sections, give a final **CROSS-IMAGE CONSISTENCY MAP**:
- which images describe the SAME device or same experiment
- which images contradict each other
- which topology features appear in ≥2 images (load-bearing) vs only 1
  (might be a one-off or might be the most recent revision)

Finally, list **≥3 SCHEMATIC / TOPOLOGY INSIGHTS** that a reader who only
read the BSIM4 + Gummel-Poon 3-branch I-V model would MISS. I.e., what is
physically present in the cell that a pure transport model does not
capture. (Examples of the *kind* of thing we mean — but find your own:
charge-storage capacitor on the floating bulk node; gate-overlap C_GD that
sets the spike rise time; well-to-substrate parasitic that couples
neighbours; reset diode for self-resetting LIF.)

Be specific. Reference image numbers. No hedging.
