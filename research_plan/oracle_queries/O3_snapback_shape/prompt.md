# Oracle Query O3 — Sharp Id snapback in DC sweep that BSIM4 port misses

**Audience:** senior device-physics reviewer (compact-modelling, BSIM4 internals, SOI/floating-body, Gummel-Poon BJT, DC continuation solvers).

**Ask:** identify the *physical* (or *numerical*) mechanism by which Sebas Pazos's LTSpice deck produces a near-vertical ~6-decade Id step in the DC Id–Vd characteristic of the 2T NS-RAM cell, while our PyTorch BSIM4 port produces a smooth monotonic ramp through the same Vd range with the same cards.

---

## 1. Device & cards

**Cell topology — 2T NS-RAM (Sebas's `2tnsram_simple.asc`, attached):**

- M1: 130 nm bulk NMOS, deep N-well, **floating P-body** (node `Vb`). Card `M1_130DNWFB.txt`.
- M2: 130 nm bulk NMOS, drain tied to M1 source. Card `M2_130bulkNSRAM.txt`.
- Parasitic vertical NPN (Gummel-Poon): emitter = source-of-M1 = GND, base = floating P-body Vb, collector = drain-of-M1 = Vd. Card `parasiticBJT.txt`.
- Deep N-well biased at `vnwell = +2 V` (well-to-body junction reverse-biased, but provides the GIDL/punch-through and well-body diode path).
- Sweep: VG1 ∈ {0.2, 0.4, 0.6} V, VG2 ∈ {0.0 … 1.0} V (33 curves total), Vd = 0 → 2 V DC sweep at each (VG1, VG2). VS = 0 V.

The measured I–V (`fit_vs_meas.png` — solid lines = measurement, dashed = our fit) shows, for several (VG1, VG2) corners:
- Subthreshold-like floor at Id ~ 1e-12 A through Vd ≈ 1.0 V,
- A near-vertical jump of ~5–6 decades around Vd ≈ 1.0–1.5 V,
- Saturation plateau at Id ~ 1e-6 to 1e-5 A.

LTSpice converges on this with no explicit avalanche element. **Sebas's 17 Apr note**: he dropped avalanche diodes for convergence reasons and relies on (i) BSIM4 §6.1 impact ionisation, (ii) "complementary bipolar current" through the parasitic NPN, (iii) body bias coupling, (iv) LDE.

## 2. What our port has

PyTorch differentiable port of BSIM4 v4.8.3, all in attached files:

- `compute_dc.py` (`dc.py` upstream) — DC core: VTH0 → Vgsteff → Vdsat → Idsat → Idsource via §5.4–5.6 (Idl, Idsat, Vdseff, output conductance with PVAG/PCLM/PSCBE — yes PSCBE1/PSCBE2 are wired in).
- `compute_iimpact_and_gidl.py` (`leak.py` upstream) — `compute_iimpact` (§6.1, IIMOD branches; α0/β0/α1, "new" vs "legacy" formulation) and `compute_gidl` / `compute_gisl` (§6.2, AGIDL/BGIDL/CGIDL, EGIDL).
- `compute_body_diodes.py` (`diode.py` upstream) — 4 body diodes: BS, BD, well-body (DNW→pwell), and S/D-to-body, with Isbs/Isbd/Nj, including breakdown via XJBV/BVJ if enabled.
- `bjt.py` — Gummel-Poon NPN (Is, Bf, Br, Vaf, Var, Ikf, Ne, Nc; emitter=GND, base=Vb, collector=Vd). No avalanche multiplication (BV/IKR not used).
- `nsram_cell_2T.py` — node assembly: residual KCL at Vd_intermediate (M1.S = M2.D) and at Vb (floating P-body — KCL between M1.iibody, M1.gidl/gisl, M2 body, NPN base current, body diodes; no other path to Vb).
- `arclength.py` — Keller's pseudo-arclength continuation: tol 1e-13, branch detection via tangent sign-flip, adaptive ds, fold detection by checking determinant sign of augmented Jacobian.

## 3. Status (z91g_two_model_validation)

- Median log-RMSE = **0.95 dec** (down from 4.23 yesterday after topology fix).
- 25/25 curves: full convergence end-to-end, no NaNs.
- **Continuation solver reports `n_folds = 0`** for every curve.
- LTSpice on the same cards produces the snapback fold cleanly.
- Visually our Id(Vd) is monotonic and smooth where the measurement is a step.

So either (a) physics needed for the snapback is missing from our port, or (b) physics is present but loop gain stays below 1 because of a missing coupling/sign, or (c) it's there and the arclength solver is stepping over the fold without registering it.

## 4. Specific questions (in priority order)

**(a) Physical mechanism without an explicit avalanche element.**
Without an `avalanche` device, what in M1's BSIM4 card or the topology produces the LTSpice DC snapback? Most likely candidates we're considering:
  - **PSCBE substrate-current second-order term** (§6.4 Vbseff feedback through PSCBE1/PSCBE2)?
  - **Junction breakdown** in the well-body or BS/BD diode (XJBV/BVJ/IJTHSREV) reverse-biased at Vd>BVJ, dumping current into Vb?
  - **NPN positive-feedback runaway**: Iii(M1) charges Vb → Vbe(NPN)↑ → Ic↑ → drops Vd_intermediate? But our NPN is wired emitter=GND so this path *should* dominate. Is the issue with our Gummel-Poon Ikf knee, or Vaf early-effect coupling?
  - **GIDL into floating body** — we have it but maybe the coefficients in our card don't excite it at Vd≈1V.
  - **Some interaction we missed** (e.g. DITS, RDSWMIN, RBODYMOD body resistance ladder)?

**(b) Loop-gain instrumentation.**
The snapback is a fold ⇒ at the critical Vd, dI/dV diverges ⇒ ∂F/∂Vb · ∂Vb/∂I · ∂I/∂Vd → 1 (or the eigenvalue of the small-signal feedback matrix passes 1). Given our floating-body KCL F(Vb, Vd)=0 with sources {Iii(Vds-Vdseff, Vbseff), Igidl, INPN(Vbe), Idiodes(Vb, Vnwell)}, what closed-form loop-gain expression should we instrument and log alongside the sweep to confirm we're below 1? Specifically: which partials dominate near the fold for a 130 nm DNW-FB device, and what magnitude should each have?

**(c) Solver-side false negative.**
Could arclength be silently stepping over the fold? Our `branch_detection` uses sign of det(augmented J) and tangent flip. With ds_max relatively large (we use ds_init=1e-3 in V on the param axis) is it plausible we jump from pre-fold to post-fold branch without registering n_folds? What test would falsify this — e.g. dropping ds_max to 1e-6 and looking for hysteresis on reverse sweep?

**(d) Single decisive experiment.**
What one experiment localises the missing piece between {missing physics, missing coupling, solver false-negative}? Specifically one we can run in <1 day with PyTorch + LTSpice oracle access. We can drive any LTSpice probe (V(Vb), I(NPN.B), I(M1.b), I(M1.dii)) and any internal in our PyTorch model (Iii, Igidl, INPN.Ic, Idiodes per junction).

## 5. What to return

1. Single best-guess **mechanism** with one-paragraph derivation.
2. Closed-form loop-gain expression to log.
3. Falsifiable single-experiment protocol (≤1 day, ≤2 setups).
4. If multiple mechanisms are plausible, rank and explain the tie-breaker.

Attached files: parameter cards (M1, M2, parasiticBJT, 2Tcell_BSIM_param_DC.csv), schematic (2tnsram_simple.asc), our PyTorch implementation (compute_dc.py, compute_iimpact_and_gidl.py, compute_body_diodes.py, bjt.py, nsram_cell_2T.py, arclength.py), validation script (z91g_two_model_validation.py), measured-vs-fit plot (fit_vs_meas.png), and run summary (summary.json).
