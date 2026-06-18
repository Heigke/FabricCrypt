# openai response (gpt-5) — 100s

Here’s a pre-integration plan you can drop into the PyTorch port as a single edit and be ready to flip on as soon as the SPICE card arrives.

1) Most likely connection for the new p-diode (ranked)

- body ↔ deep-N-well (vnwell = +2 V). Confidence ~0.85.
  - It’s the physical P-well-to-DNW isolation junction under/around the floating P-body. With vnwell fixed at +2 V the junction is strongly reverse biased in normal operation (Vb ~ 0–0.7 V ⇒ Vpn = Vb − Vnw ≈ −1.3 to −2 V), so it behaves primarily as a voltage-dependent depletion capacitor with tiny DC leakage. That matches Sebas’ “capacitive response of the floating body” comment and the suggestion to approximate it as a 5–10 fF linear C.

- body ↔ substrate (GND). Confidence ~0.3.
  - Only plausible if the cell is not in a triple-well. But you already stated a deep-N-well at +2 V is present, so body-to-substrate is not a single junction; it’s body↔DNW↔substrate. A direct body↔GND diode would be forward-biased at Vb ≈ 0.49 V and would introduce a significant DC drain path. That’s inconsistent with the “capacitive response” emphasis unless he’s explicitly modeling a parasitic tie that was missing before.

- body ↔ Sint. Confidence ~0.15.
  - The P-body-to-N+ source/drain junctions already exist inside the BSIM device (Ibs/Ibd and their junction caps). Adding a discrete pdiode here would double-count unless there is a physical separate P–N junction to Sint (unusual for this cell).

2) Likely diode model parameters and reasonable defaults

Pick defaults that reproduce 5–10 fF around the operating reverse bias and keep DC leakage tiny:

- Area A: 22 µm² (5 µm × 4.4 µm)
- CJ0 per area (zero-bias): choose 0.45–0.65 fF/µm²
  - With Vj = 0.7 V and M = 0.5, at Vpn = Vb − Vnw ≈ −1.5 V the reverse-bias factor is (1 − Vpn/Vj)^(−M) = (1 + 2.1429)^(−0.5) ≈ 0.564.
  - To get 5–10 fF effective at −1.5 V, set CJ0_total = 5–10 fF / 0.564 = 8.9–17.7 fF, i.e., CJ0/area ≈ 0.405–0.804 fF/µm². A good mid pick is 0.56 fF/µm² (CJ0_total ≈ 12.3 fF ⇒ ≈ 7 fF at −1.5 V).

- Vj: 0.7 V (0.65–0.85 V typical)
- M: 0.5 (0.33–0.5 typical)
- IS density (A/µm²): very small; 1e−18 A/µm² is a safe default for a reverse-biased isolation junction (scales to ~22 aA total IS). If you prefer to mirror your earlier DNW↔body diode number, 3.4e−7 A/m² = 3.4e−19 A/µm² is even smaller.
- Emission coefficient n: 1.2 (1.05–1.3 typical)
- RS: 0–10 Ω is fine for reverse-bias dynamics; it’s irrelevant unless you forward bias.
- Optional TT (forward diffusion charge): 0 for now; can be enabled if you later see forward-bias body transients.

3) Quantitative effect if the diode were body↔substrate (GND) and forward biased at your OP

At Vb = 0.487 V, T = 300 K (Vt = 25.85 mV), n = 1.3:
- Exponent = Vpn/(n·Vt) = 0.487 / (1.3·0.02585) ≈ 14.5 ⇒ exp ≈ 2.0×10^6

Let IS_density be 1e−18, 1e−17, or 1e−16 A/µm². Total IS = A·IS_density = 22·IS_density.

Resulting forward current:
- IS_density = 1e−18 A/µm² ⇒ I ≈ 2.0e6 · 22e−18 ≈ 4.4e−11 A ≈ 44 pA
- IS_density = 1e−17 A/µm² ⇒ I ≈ 0.44 nA
- IS_density = 1e−16 A/µm² ⇒ I ≈ 4.4 nA

So a body↔GND diode would drain the body on the order of tens of pA to a few nA, depending on IS. By contrast, if the diode is body↔vnwell at +2 V, it is reverse biased by ~1.5 V and its additional DC current is ≈ −IS_total (aA–fA scale with the defaults), i.e., negligible in DC, acting almost purely as a depletion capacitor. This strongly reinforces “body↔vnwell” as the intended connection.

4) One-edit, env-var-controlled patch to nsram_cell_2T._residuals

Goal
- Add a single PN-junction element with Shockley conduction and reverse-bias depletion capacitance Cj(V) that can be connected between body and one of three nodes: vnwell, GND, or Sint.
- Defaults implement the likely intended case (body↔vnwell) with capacitance ≈ 7 fF at Vpn ≈ −1.5 V and negligible IS.
- Keep the existing I_well_body term OFF when the new diode is used to vnwell (avoid double-counting). Leave it as-is for ‘off’ or other topologies.

Minimal patch sketch

- Read env vars at import time or in __init__:
  - NSRAM_PDI_TO in {'off','vnwell','gnd','sint'}; default 'vnwell'
  - NSRAM_PDI_A_UM2 default '22.0'
  - NSRAM_PDI_CJ0_F_PER_UM2 default '5.6e-16'  (0.56 fF/µm²)
  - NSRAM_PDI_VJ default '0.7'
  - NSRAM_PDI_M default '0.5'
  - NSRAM_PDI_IS_A_PER_UM2 default '1e-18'
  - NSRAM_PDI_N default '1.2'
  - Optional: NSRAM_PDI_TT default '0' (forward diffusion charge), NSRAM_PDI_DISABLE_LEGACY_VNWELL default '1'

- Stamp into KCL in _residuals with symmetric contributions if the second node is Sint (unknown). For constant nodes (vnwell, gnd), only stamp the body node. Sign convention below assumes current from body (P) to the other node (N) is positive when Vpn = Vb − Vx > 0, so it leaves body; KCL into body gets a negative sign.

Code snippet (drop-in style; names may need adapting to your file)

Add near the top of the module:
- import os
- Read once:
  PDI = {
      'to': os.getenv('NSRAM_PDI_TO', 'vnwell'),
      'A_um2': float(os.getenv('NSRAM_PDI_A_UM2', '22.0')),
      'CJ0_F_per_um2': float(os.getenv('NSRAM_PDI_CJ0_F_PER_UM2', '5.6e-16')),
      'VJ': float(os.getenv('NSRAM_PDI_VJ', '0.7')),
      'M': float(os.getenv('NSRAM_PDI_M', '0.5')),
      'IS_A_per_um2': float(os.getenv('NSRAM_PDI_IS_A_PER_UM2', '1e-18')),
      'N': float(os.getenv('NSRAM_PDI_N', '1.2')),
      'TT': float(os.getenv('NSRAM_PDI_TT', '0')),  # optional forward diffusion time
      'kill_legacy_vnwell': int(os.getenv('NSRAM_PDI_DISABLE_LEGACY_VNWELL', '1')),
  }

Inside _residuals (after you compute Vb, Vsint, etc.):

- Determine the second node voltage and its time derivative:
  if PDI['to'] == 'off':
      pass  # no stamping
  else:
      Vb = V_body
      Vdot_b = dV_body_dt
      if PDI['to'] == 'vnwell':
          Vx = self.vnwell  # +2.0 V (tensor scalar)
          Vdot_x = 0.0*Vb
          x_is_unknown = False
      elif PDI['to'] == 'gnd':
          Vx = 0.0*Vb
          Vdot_x = 0.0*Vb
          x_is_unknown = False
      else:  # 'sint'
          Vx = V_sint
          Vdot_x = dV_sint_dt
          x_is_unknown = True

      # Shockley conduction
      Vt = self.kT_q  # 25.85 mV @ 300K, or compute from self.temp
      nVt = PDI['N'] * Vt
      Is_total = PDI['IS_A_per_um2'] * PDI['A_um2']
      # safe expm1
      arg = (Vb - Vx) / nVt
      arg = torch.clamp(arg, min=-40.0, max=40.0)
      I_cond = Is_total * torch.expm1(arg)   # >0 when forward
      # KCL at body: current leaving body is negative
      R_B = R_B - I_cond
      if x_is_unknown:
          R_Sint = R_Sint + I_cond

      # Depletion capacitance (reverse and small forward; clamp near forward conduction)
      Cj0_total = PDI['CJ0_F_per_um2'] * PDI['A_um2']
      x = 1.0 - (Vb - Vx)/PDI['VJ']
      x = torch.clamp(x, min=1e-3)  # avoid negative in deep forward
      C_dep = Cj0_total * torch.pow(x, -PDI['M'])
      I_cap = C_dep * (Vdot_b - Vdot_x)  # current from body to x
      R_B = R_B - I_cap
      if x_is_unknown:
          R_Sint = R_Sint + I_cap

      # Optional forward diffusion charge (disabled by default)
      if PDI['TT'] > 0.0:
          I_fwd = torch.clamp(I_cond, min=0.0)
          C_diff = (PDI['TT'] / nVt) * I_fwd
          I_dcap = C_diff * (Vdot_b - Vdot_x)
          R_B = R_B - I_dcap
          if x_is_unknown:
              R_Sint = R_Sint + I_dcap

- If PDI['to'] == 'vnwell' and PDI['kill_legacy_vnwell']:
    - Zero or remove the legacy I_well_body term to avoid double-counting. For example:
      if PDI['to'] == 'vnwell' and PDI['kill_legacy_vnwell']:
          I_well_body = 0.0

How to exercise the three candidate topologies without changing code

- Body↔vnwell (default; recommended)
  - NSRAM_PDI_TO=vnwell
  - NSRAM_PDI_CJ0_F_PER_UM2=5.6e-16
  - NSRAM_PDI_IS_A_PER_UM2=1e-18
  - NSRAM_PDI_DISABLE_LEGACY_VNWELL=1

- Body↔GND
  - NSRAM_PDI_TO=gnd
  - Keep same CJ0 and IS. Expect negligible DC change if you set IS tiny; if you want to estimate worst-case DC drain, sweep NSRAM_PDI_IS_A_PER_UM2 in [1e-18,1e-16].

- Body↔Sint
  - NSRAM_PDI_TO=sint
  - Same params. This one will add a B–Sint capacitive tie and conduction path; be mindful it can interact with M1’s internal Cbs/Cbd already present in BSIM.

Sanity checks you can run immediately

- With NSRAM_PDI_TO=vnwell and the defaults above, at Vb ~0.5 V you should see:
  - Additional DC body current ≈ −Is_total ≈ −(1e−18 · 22) A ≈ −22 aA (negligible)
  - Added capacitance ≈ 7 fF between body and vnwell at Vpn ≈ −1.5 V (scales with Vb)
- With NSRAM_PDI_TO=gnd and IS sweep, at Vb ≈ 0.487 V expect ≈ 44 pA (IS=1e−18), 0.44 nA (1e−17), 4.4 nA (1e−16) body-drain currents, as in section 3.

When the SPICE card arrives

- Map SPICE to the env vars:
  - IS → NSRAM_PDI_IS_A_PER_UM2 after dividing by the diode “area” scaling if the card uses area as a separate element (or directly set IS_A_total = IS*Area and set A_um2=1).
  - N → NSRAM_PDI_N
  - RS → ignore or add if you have a series-R wrapper; usually negligible here.
  - CJ, CJSW → if sidewall caps are provided, you can fold them into an effective CJ0 by adding C_area + C_perim once and dividing by 22 µm² to get an effective per-area CJ0, or extend the stamp to include a sidewall term if you know the perimeter.
  - VJ → NSRAM_PDI_VJ
  - M → NSRAM_PDI_M
  - FC/TT → you can leave FC implicit via the clamp; set TT via NSRAM_PDI_TT if the forward diffusion cap matters.

This keeps the change localized to a single block in _residuals and entirely controlled by env vars so you can A/B the three plausible topologies without touching the rest of the solver.
