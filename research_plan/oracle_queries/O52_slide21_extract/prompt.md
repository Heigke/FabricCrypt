You are looking at slide 15 (transient_VD_ramps) and slide 21 (pdiode_dynamic_response).

These are Mario+Sebas's plots showing measured 2T NS-RAM cell response
under V_d ramps that go ABOVE 2V (where snapback / avalanche regime
lives). We need to extract digitizable (V_d, I_d) or (V_d, V_b) sample
points for use as model validation data.

For EACH visible curve in each slide:
1. Identify what's plotted (axes, units, log/lin, ramp rate if labeled)
2. Extract 8-15 sample points (V_d, value) at uniform V_d spacing
   covering 2.0V to highest visible V_d
3. Estimate uncertainty: ±X% on the y-value at each sample
4. Note any breakpoints visible (snapback fold, knee voltage, peak)

Output as JSON (and ONLY JSON, no surrounding prose, no markdown fences):
{
  "slide_15": [
    {"curve_label": "ramp 0.017 V/s", "y_axis": "Id (A)", "axis_scale": "log",
     "samples": [[v1, i1], [v2, i2], ...], "uncertainty_pct": 30,
     "snapback_peak_v": 2.6, "knee_v": 1.8}
  ],
  "slide_21": [...]
}
