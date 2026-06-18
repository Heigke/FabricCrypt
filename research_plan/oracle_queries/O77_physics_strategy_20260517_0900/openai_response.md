# openai response (gpt-5) — 247s

Q1 — ONE highest‑leverage experiment without A.12

Pick: Implement BSIM “rbodymod=1” distributed body resistance for M1 and M2, wire it into both DC and transient KCL, and calibrate a single global scale factor s_Rb against the forward↔backward asymmetry on the 33‑IV set. Then re‑run: (a) full cell‑wide DC fwd+bwd, and (b) a fast‑pulse smoke on the 4 standard biases, both with snapback_subcircuit disabled (SB_OFF). Script name: z461_rbodymod1_dc_tr.py.

Why this dominates:
- It directly targets the single biggest unimplemented physical mechanism we’ve flagged for weeks. All our best DC numbers collapse in the backward sweep because the body is effectively pinned and cannot relax from the high‑V_D warm state; a distributed Rb is exactly what BSIM4 introduced to avoid this pathology.
- It is orthogonal to all “parameter tweaking” (ALPHA0, PWL refits, etc.) which have repeatedly bounced off ~1–2 dec. It changes the topology of the body return path and the Newton basins, not just coefficients.
- It also underpins dynamics: a finite body network is the natural post‑snap discharge path necessary for self‑reset; without it, even a perfect snap trigger can’t recover.
- It uses assets we already have and trust (33 IV curves; M1/M2 cards; cap audit) and requires no A.12. The self‑heating knob can stay off; the body network alone should reduce the fwd↔bwd hysteresis that is killing our honest averages.

Exact experiment details:
- Network: BSIM4 rbodymod=1 star for each MOSFET body, i.e. three resistors from the body node to S, D, and bulk/well nodes: RB(SB), RB(DB), RB(PB) (naming per BSIM4 manual). Implement for M1 and M2, then tie M1 body to the modeled cell body node V_B via the same partitioning as the schematic (i.e., don’t create a phantom new body; use the cell’s V_B).
- Values: compute base resistances from device geometry and a sheet resistivity prior, then scale by a single s_Rb fitted to data.
  - Base per‑square prior: ρ_sheet,p‑well = 2–8 kΩ/□ for NA≈(3–10)×10^16 cm^-3; cite: Sze & Ng, Physics of Semiconductor Devices, 3rd ed., 2006 (ρ ≈ 1/(q μp NA), μp≈200 cm²/Vs) → ρ≈0.16–0.5 Ω·cm; for t≈1 μm effective thickness gives ρ_sheet≈1.6–5 kΩ/□. Deep‑nwell isolation increases path length (more squares), so end‑to‑end RB in the tens–hundreds kΩ is physically reasonable for our 0.36 μm × 0.18 μm device footprint plus routing.
  - Geometric squares: estimate 5–20 squares to S/D and 20–100 squares to well (layout‑dependent; we’ll parameterize). Start with RB(SB)=50 kΩ, RB(DB)=50 kΩ, RB(PB)=200 kΩ, then apply s_Rb.
  - Literature cross‑check for PNPN holding conditions: I_H≈10–100 μA, V_H≈0.8–1.2 V in bulk CMOS latch‑up (T. A. Lewis et al., IEEE TED 1991; classic latch‑up texts). This bounds how low RB can go without forcing permanent latch.
- Fit protocol: grid‑search s_Rb over logspace[0.25, 4]× (i.e., multiply all RB(*) by {0.25, 0.5, 1, 2, 4}). For each s_Rb:
  1) DC fwd and bwd over all 25 biases with SB_OFF, VBIC on (z443 baseline settings).
  2) Compute Δ_hyst = RMS(fwd−bwd) per curve and cell‑wide.
  3) Fast‑pulse smoke (the 4 standard biases) only to verify no catastrophic change in ns‑kick behavior (no pass/fail on transient amplitude here).
- PASS criteria (any pass locks the experiment in):
  - Primary DC target: reduce the backward sweep from 2.864 dec (z454 SB_OFF) to ≤1.9 dec while keeping forward ≤1.6 dec; i.e., cell‑wide avg ≤1.6 dec and Δ_hyst ≤0.30 dec. Report n=25/25 both directions.
  - Secondary transient sanity: the V_B(t) peak under the 10 ns pulse at VG1=0.6/VG2=0.0 changes by less than 20% vs SB_OFF baseline (we are not hunting transient amplitude yet; we’re preventing new pathologies).
- What we reuse:
  - 33 measured IVs (25 with parameter cards) as the objective.
  - M1/M2 cards, VBIC parasitic NPN (Bf=10^4, τ_F=25 ps).
  - cap_breakdown.json C_eff≈2.66 fF for the τ estimate sanity checks.
- What we assume from literature:
  - p‑well sheet resistance range and order‑of‑magnitude path lengths (Sze & Ng 2006; BSIM4 v4.8 User’s Manual, rbodymod=1 topology and recommended use).
  - Latch‑up holding current/voltage ballpark (10–100 μA, 0.8–1.2 V) to keep us out of “always‑on” PNPN.
- Why it beats the other candidates:
  - Re‑fitting PWL with finer V_DS doesn’t change the topology; we’ve already demonstrated parameter sweeps bounce at ~1–1.3 dec in one direction and blow up backward.
  - snap_Is grids fiddle a patch that we already know corrupts DC when enabled; the body network is a core device effect the patch assumes.
  - BSIM4 version nudge or parasitic L sweeps are second‑order vs the basin selection we’re failing today.
  - Self‑heating could help later for matching ~400 ns, but it won’t fix the DC backward catastrophe; rbodymod does.

Execution notes:
- This must be wired into both Newton DC and the PT integrator. The P4 “single‑R only in transient” was structurally a no‑op for DC; don’t repeat that. Replace the body stamp in the M1/M2 element builder so RB(*) participate in the DC conductance matrix.
- Enforce strict accounting: 25/25 biases in both directions; fail the run if any bias drops out.

VERDICT: Run rbodymod=1 distributed body network with a single global scale s_Rb, target DC avg ≤1.6 dec and Δ_hyst ≤0.30 dec, reusing the 33 IV set; this is the highest‑leverage physics add. Confidence 0.72.


Q2 — Innate LIF closure (z458 design review)

Resolution:
- Make it log‑spaced and slightly denser: 6×6 is the practical minimum; 4×4 is too coarse to catch the narrow transition between “race‑off” and “latch‑on.”
  - R_body ∈ [1e6, 3e6, 1e7, 3e7, 1e8, 3e8] Ω
  - snap_Is (body‑injection scale in the snapback block) ∈ [3e-8, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5] A
- Keep NX_1p8 (current‑gated NPN collector) ON; without it z456 showed permanent hold. Use the cap audit C_eff=2.66 fF.

Most likely failure mode:
- (a) Latch‑up — NPN never turns off, V_B pinned high.
  - Evidence: z456 showed no self‑reset even with R_body as low as 1 MΩ; the parasitic NPN’s hold current stays orders of magnitude above the R_body leak over the relevant V_B range. z457’s NX_1p8 helped DC but still left V_B(t) ≈ 0.635 at 10 ns with no decay. Until the collector clamp is tamed across the whole trajectory (not just at “mid‑DC”), the loop remains above I_H most of the time.
Runner‑up failures:
- (c) Race — at the smallest R_body, the body node drains before V_db crosses V_knee, preventing snap in marginal biases (you will see sub‑ns blips and no full spike).
- (b) Sub‑knee snap — less likely with NX_1p8, because z457 showed NPN gating still allows snap at V_knee=1.8 V; but it will appear in some low‑VG2 points in the grid corners with small snap_Is.

Predicted success rate on a 4×4 (if you insist on 4×4):
- Using R_body ∈ [1e6, 1e7, 1e8, 1e9] Ω and snap_Is ∈ [3e-7, 1e-6, 3e-6, 1e-5] A, with NX_1p8:
  - Clean single LIF spike (V_db crosses knee, NPN on, then off, V_B returns within 1 μs): 1–2 of 16 at best; most likely 1/16.
  - Expect 8–10/16 latch‑ups (a), 3–5/16 races (c), 1–2/16 sub‑knee (b). Multi‑fire/oscillation (d) will occur only if an implicit slow timescale exists (thermal or a very large effective R_body); with current code (no self‑heating) it’s rare and uncontrolled.

Recommendation for z458:
- Use 6×6 log grid above; add two observables per run: I_NPN(t) minimum after the spike, and V_B(t) at 1 μs. Tag “success” only if I_NPN,min < 2 μA and V_B(1 μs) < 50 mV in addition to the spike detection.
- Run at two biases: VG1=0.6/VG2=0.0 (easy corner) and VG1=0.4/VG2=0.2 (harder, closer to boundary).

VERDICT: 4×4 is too coarse; use a 6×6 log grid. Most likely failure is latch‑up; expect 1/16 successes on 4×4, 4–6/36 on 6×6 with NX_1p8. Confidence 0.66.


Q3 — No‑A.12 publishability

Strongest defensible claim with current assets + qualitative nod to Mario slide‑21:
- An open, reproducible 2T NS‑RAM compact modeling framework that achieves balanced forward/backward DC accuracy around 1.2–1.3 decades RMSE on the full 25‑bias subset of Sebas’s 33 IV curves using a pseudo‑transient (PT) body solver (z432/z446), together with a mechanism‑level audit that isolates what is missing for ns‑snap and oscillation (lack of a measured transient target; absence of a distributed body network; disabled self‑heating). We can additionally show a qualitative limit‑cycle in the hundreds‑of‑ns regime using literature‑plausible parameters, but we cannot claim a quantitative transient match.

Figures/metrics to include:
- Cell‑wide DC overlays: measured vs model for all 33 curves, grouped by VG1, with per‑branch log10‑RMSE. Show both directions and the balanced average for z446.PT_GP and z446.PT_VBIC. Target: 1.188–1.276 dec averages, n=25/25 each direction.
- Hysteresis map: heatmap of fwd minus bwd residuals vs (VG1, VG2); show that Newton‑only pipelines blow up backward to ~2.86 dec and why the PT integrator stabilizes the body.
- Ablation: VBIC vs GP, SB_OFF vs SB_ON (z454 table) demonstrating the DC trade‑off of snapback patches; include the cap audit (2.657 fF) that corrected the earlier 12.1 fF overestimate.
- NPN gating region map (z457): show how V_knee gating reduces mid‑DC collector current by ~2–4 orders (N_OFF vs NX_1p8), and the corresponding DC delta (−0.223 dec best case), even though it does not yet rescue transient self‑reset.
- Optional, labeled “illustrative/qualitative”: a transient panel where a phenomenological snapback block produces a nanosecond‑scale spike, with an added slow time constant (either large R_body or a single thermal RC) tuned to yield an oscillation period in the 200–800 ns band. This panel is explicitly not used for any quantitative claim.

Claims we must avoid:
- Any quantitative transient claim: “We reproduce I(t) or V_B(t) within X%/NRMSE over 0–1 μs,” “period = 400±Y ns,” “τ_relax extracted = …,” “self‑heating calibrated,” “holding current/voltage measured from silicon.”
- Any claim that mixes directions without equal denominators (no “avg ~1.19” unless both fwd and bwd have n=25/25).
- Any implication that SB_ON dynamics are “the physics” without the DC penalty presented side‑by‑side.

Language softening to use:
- “Qualitative match,” “consistent with an O(10^2–10^3 ns) limit cycle reported in prior NS‑RAM literature,” “order‑of‑magnitude,” “we deliberately disable X to preserve DC fidelity,” “we do not claim a calibrated transient; A.12 transient ground truth was unavailable.”
- For PT: “solver‑based stabilization of the body state,” not “physics‑accurate dynamics.”

Realistic venue:
- Without transient validation, this is a short, methods‑forward paper: TCAD Letters or IEEE EDL short communication if we keep it tight on mechanisms and honesty about limits; otherwise a solid workshop (e.g., SISPAD/DRC) or arXiv + code release. It does not merit a top‑tier full conference device‑physics slot until we either add rbodymod=1 and show balanced DC in Newton, or obtain a quantitative transient target.

VERDICT: Publish as an honest DC‑fidelity + mechanism audit with qualitative transient illustration; target TCAD‑Letter/EDL/workshop, not a full top‑tier conference, unless rbodymod=1 and balanced DC are demonstrated first. Confidence 0.74.
