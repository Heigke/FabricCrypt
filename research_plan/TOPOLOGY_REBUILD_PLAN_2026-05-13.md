# TOPOLOGY REBUILD CAMPAIGN — 2026-05-13 starting 14:55

## Goal

Get NS-RAM pyport to reproduce silicon dynamics in the RIGHT REGIME for all four behaviors simultaneously:
1. **DC** — cell-wide median log-RMSE < 0.5 dec on Sebas 33 IV (no bimodal hiding)
2. **Snapback** — V_peak(V_G2) slope correct SIGN (matching -0.625) within ±20%
3. **Transient** — hysteresis 0.5-2× measured 2.6e-3 (not 10⁴× too small or too large)
4. **LIF behavior** — 2T setup produces correct firing dynamics: V_b accumulates, M2 fires, refractory pause

**Stop criterion**: not perfect numbers, but "all four in approximately right range, not missing any physical mechanism."

## What we currently miss (consolidated)

From SA3 (slides 1-21):
1. VNwell→VB parasitic diode with proper Cj + V-dep leakage
2. VB-VG2 designed coupling capacitor
3. VB-as-output node (M2 drain = Vspike)
4. NFACTOR(M2) coupled to BOTH V_G1 AND V_G2 via VB

From T4 oracle scan:
5. N1: oxide/interface traps multi-τ (µs→s)
6. N2: SRH gen-rec in body depletion (rate-dep trigger)
7. N3: BSIM rbodymod=1 / distributed Rbody

From P2 materials re-scan:
8. BSIM4 TAT block (pre-calibrated in cards, ignored)
9. Distributed Rbody 1MΩ-10GΩ (4 decades, in zenodo .asc)

From z313 bisection:
10. cfg.vnwell_Rs and cfg.use_lateral_collector are PARSED but NOT WIRED into _residuals (multi-day code work)

**Unaudited (DO FIRST)**:
11. `nsram/Zoom/2026-04-30 13.03.27 Zoom NSRAM/meeting_saved_closed_caption.txt` — Sebastian Zoom transcript
12. `nsram/Zoom/schematic&modelCards/`
13. `nsram/Zoom/2026-04-30 BSIMfitsBA/`
14. `nsram/Zoom/Slow I-Vs 2vHCa-2@VG2 VG1 vnwell=2 SRavg=0` — slow-IV sweeps (SRavg=0 = DC limit, this is REAL DATA we may have missed)

## Pre-registered gates (LOCKED before any compute)

### R-1 Zoom audit (no compute, 60 min)
- ≥3 NEW signals from Zoom transcript not in SA1-SA3+P2
- ≥1 actionable parameter value with citation
- Bonus: AMBITIOUS if quantitative law extractable (e.g., trap τ value)

### R-2 Materials inventory completeness (30 min, no compute)
- Compare nsram/Zoom/* file inventory to what we've actually parsed
- Map every file → (used / partial / unused) status

### R-3 Pyport infrastructure audit (subagent, 90 min, no compute)
- Walk src/nsram/bsim4_port/ and src/nsram_pyport*.py
- Identify which cfg flags are: WIRED to _residuals vs ORPHAN
- Produce table: param → consumed-by-function → effect on output

### R-4 Topology rebuild pyport_v5 (subagent, 6h, ikaros + daedalus)
- Implement properly into _residuals (not just cfg flags):
  - TAT current term using BSIM4 njts/vtss/xtss/jtss
  - Distributed Rbody (log-uniform with per-V_G1 prior)
  - VNwell→VB diode with V-dep Cj
  - VB-VG2 coupling cap (transient only)
  - Drain-end M(V_bc) avalanche
- Add per-flag unit tests: toggle param, assert I_d at one operating point CHANGES (z313 bisection found this is needed)
- Gate: cell-wide DC < 0.7 dec AND snapback slope correct sign AND TAT term computed non-zero

### R-5 LIF behavior test (zgx, 2h)
- Drive M1 with input current pulse, watch M2 spike output
- Stop criterion: clean 1V spike, refractory period 1-10 µs, threshold modulated by V_b state

### R-6 Brute-force PHYSICS sweep (all 3 GPUs, 4h)
- Sweep all 5 missing-physics flags ON/OFF + 4 magnitude params each
- 2^5 × 4^4 = 8192 jobs distributed via queue
- Output: heatmap (joint DC + snapback + transient + LIF) "in range"
- Gate: at least one cell satisfies all 4 stop criteria simultaneously

### R-7 Cell-wide refit with v5 (after R-4, ~2h)
- BBO with new wired flags
- Compare to z304 (0.99 dec) and z313 (2.91 dec)
- Gate: cell-wide < 0.5 dec, V_G1=0.2 < 1.5 dec (not 4.7 catastrophe)

### R-8 Re-run network sim on v5 surrogate (after R-7, ~2h)
- Re-do HDC at N=16384 with v5 surrogate
- z319 already showed network insensitive to DC bias → check that v5 still gives ≥83% UCI-HAR
- If accuracy drops > 2pp → v5 is hurting somehow → revert

### R-9 Oracle critique cycle on v5 (15 min)
- 3-way (gpt-5+gemini+grok) — does v5 close ALL 4 gaps?

### R-10 v4.5 brief (post-R-9, 1h)
- Update 4E brief if v5 closes gaps; otherwise honest "v5 closes X but not Y"

## Cron schedule additions

- `13 */1 * * *` — Topology rebuild progress check every hour
- Existing cron jobs continue

## Resource allocation

- **ikaros (gfx1151, governor active)**: R-4 (pyport_v5 build), R-7 (BBO)
- **daedalus (governor active)**: R-3 audit, R-6 brute-force CPU branch
- **zgx (NVIDIA, full throttle)**: R-5 LIF, R-6 GPU brute-force, R-8 network re-run
- **No machine**: R-1 Zoom transcript audit (subagent reading)

## NO-CHEAT (carried forward)

- Every gate pre-registered before compute
- Bootstrap CI for any new "median" claim
- All 5 oracle drift flags from O55/O56 carried forward in language
- HDC headline 83.86% stays locked + DEFENDED by z319
- Bayesian RNG stays locked
- v5 must IMPROVE on ALL 4 simultaneously to be called "the new model"

## Stop criteria

CAMPAIGN ENDS when EITHER:
- All 4 stop criteria met by single param vector → declare v4.5 model rebuild complete
- 24h elapsed AND v5 hasn't closed ≥2 of 4 → honest "topology rebuild incomplete, blocked on real silicon data from Sebas"
