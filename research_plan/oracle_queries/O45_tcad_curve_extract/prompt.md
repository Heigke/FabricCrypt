# O45 — TCAD curve numerical extraction

You are looking at slides from Sebastian Pazos (KAUST) and Mario Lanza
(NUS) describing the NS-RAM device. We need **APPROXIMATE numerical
values** read off the plotted curves so we can compare them against a
pyport replay of the published Sentaurus TCAD command files (Zenodo
bundle ships INPUTS only — no outputs — so the only ground truth is
what you can read off these published plots).

## Task

For EACH attached slide image that contains a measurable plot (I-V, I-V
family vs. Vg, transient V_b(t) or I(t), bulk-current curves, etc.):

1. Identify each curve in the plot (by line color / legend label /
   marker).
2. Sample **5–10 (x, y) points** along the curve, spaced reasonably
   (covering the visible range — include endpoints + interior).
3. Note the axis units exactly as printed (V, A, µA/µm, s, etc.) and
   whether each axis is log or linear.
4. Note the device context if visible (e.g., "VG1=0.6 V branch",
   "VG2 sweep", "L=130nm thick-ox", "measurement vs. simulation").

## Curves we especially need

- **Slide 06** — I-V family at 4 different VG1 values (meas vs. sim).
  Extract one curve per VG1 (4 curves total).
- **Slide 09** — 3-corner overlay meas vs. sim. Extract all 6 curves
  (3 corners × {meas, sim}).
- **Slide 14** — Bulk current meas vs. SPICE. Extract both curves.
- **Slide 15** — Transient V_d ramps (this is the one we most need,
  because z299 replays sweep ramps). Extract V(t) or I(t).
- **Slide 21** — pdiode dynamic response (May 1 update). Extract any
  V(t) or I(t) curve.
- **Slides 01–04, 07** — Parameter (NFACTOR, K1, ETAB, BETA0) vs. VG
  branches. Extract each branch as (VG, param) samples.
- **Slide 08** — Transient spike train + I-V noise band. Extract the
  underlying I-V envelope (top / bottom of the noise band).
- **Slide 13** — Semi-empirical bulk-current PWL. Extract the PWL.

## Output format — STRICT JSON ONLY

Wrap your entire reply in a single fenced code block ```json … ```.
No prose outside the JSON. Schema:

```json
{
  "extractions": [
    {
      "slide": 6,
      "curve_label": "VG1=0.6 V, measurement",
      "x_axis": "Vd [V]",
      "y_axis": "Id [A]",
      "x_scale": "linear",
      "y_scale": "log",
      "samples": [[0.0, 1e-12], [0.5, 3e-10], [1.0, 1e-8], [1.5, 5e-8], [2.0, 1.2e-7]],
      "notes": "approximate; small inset plot; estimate ±0.3 dec",
      "confidence": "low|medium|high"
    }
  ],
  "could_not_extract": [
    {"slide": 8, "reason": "plot too small to read individual points"}
  ]
}
```

If a slide has no extractable numerical plot, list it under
`could_not_extract`. If a value is genuinely unreadable, **say so** in
`could_not_extract` — do NOT fabricate numbers.

Estimate as best you can — log-scale plots are OK to read to ±0.3 dec
accuracy. Linear plots ±10 %.
