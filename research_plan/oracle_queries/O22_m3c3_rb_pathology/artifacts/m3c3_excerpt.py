    # Parasitic NPN: collector=D, base=B, emitter=GND.
    # ──────────────────────────────────────────────────────────────────
    # IMPORTANT (2026-05-01, A.1.i finding): Sebastian's LTSpice schematic
    # `2tnsram_simple.asc` wires the parasitic NPN with **emitter to
    # ground**, not to Sint. This is the "complementary bipolar current"
    # he refers to in his Apr-17 email — its purpose is to provide a
    # body-charging path that fires when Vb climbs (Vbe = Vb − 0 = Vb,
    # not Vb − Vsint ≈ small). With emitter=Sint the BJT would never
    # turn on at low VG2 because Vb tracks Vsint. With emitter=GND, Vbe
    # tracks Vb directly and the NPN switches at Vb ~0.6 V.
    if cfg.use_bjt:
        Vbe = Vb                 # emitter = ground (legacy F1.v2 path)
        Vbc = Vb - Vd            # collector = drain
        bjt_out = compute_bjt(bjt, Vbe=Vbe, Vbc=Vbc, T_K=273.15 + cfg.T_C)
        Ic_Q1 = bjt_out["Ic"]    # collector current (drain → emitter = GND)
        Ib_Q1 = bjt_out["Ib"]    # base current (INTO base from external)
        Ie_Q1 = bjt_out["Ie"]    # emitter current at GND (= −(Ic+Ib))
    else:
        Ic_Q1 = torch.zeros_like(Vd)
        Ib_Q1 = torch.zeros_like(Vd)
        Ie_Q1 = torch.zeros_like(Vd)
    # M3c.3 local-base node is patched in below, after iii_gain/eta_lat are
    # defined (those need m1, m2). When use_local_base=True we do an inner
    # 1D solve for Vb_local and overwrite Ic_Q1, Ib_Q1, Ie_Q1.
    Vb_local = Vb  # default: local = global (F1.v2 reduction)

    # ---- Sint KCL: currents INTO Sint --------------------------------- #
    # M1 channel current Ids_M1 flows D→S — INTO Sint (M1 source). → +Ids_M1
    # M2 drain is Sint; M2 channel sinks current FROM drain → −Ids_M2
    # BJT emitter is now GND, NOT Sint — BJT no longer touches Sint node.
    # M1 junction: Ibs_M1 >0 ⇒ leaves body INTO source(=Sint). → +Ibs_M1
    # M2 junction: Ibd_M2 >0 ⇒ leaves body INTO drain(=Sint). → +Ibd_M2
    R_Sint = (
        m1["Ids"]
        - m2["Ids"]
        + m1["Ibs"]
        + m2["Ibd"]
    )

    # Deep-N-well to body diode (A.1.n: this is the missing body-charging path).
    # ──────────────────────────────────────────────────────────────────
    # When vnwell > Vb, the N-well/P-body junction forward-biases and pumps
    # current INTO the body. Modelled as a Shockley diode with series R:
    #
    #     I_ideal  = Js·A · (exp((vnwell − Vb)/(n·Vt)) − 1)
    #     I_Rs     = (vnwell − Vb) / Rs   (when forward biased)
    #     I_well_b = harmonic_mean(I_ideal, I_Rs)   smooth transition
    #
    # Reverse-bias contribution is tiny (Js·A ~1e-15 A) — included for
    # completeness so derivatives are continuous through Vb crossing vnwell.
    if cfg.use_well_diode:
        Vt = 0.02585 * (273.15 + cfg.T_C) / 300.0   # thermal voltage at T
        V_drive = cfg.vnwell - Vb
        # Clamp exponent to avoid overflow when V_drive >> Vt
        exp_arg = (V_drive / (cfg.vnwell_n * Vt)).clamp(max=40.0)
        I_ideal = cfg.vnwell_Js * cfg.vnwell_area * (torch.exp(exp_arg) - 1.0)
        # Series-R limited current (only forward; reverse bias = 0 here)
        I_Rs = torch.relu(V_drive) / cfg.vnwell_Rs
        # Smooth min via harmonic mean (differentiable, transitions at the
        # smaller of the two without a hard kink)
        eps = 1e-30
        I_well_body = (I_ideal * I_Rs) / (I_ideal.abs() + I_Rs + eps)
        # Scale by mbjt — the well-body junction belongs to the same
        # parasitic bipolar structure as Q1, so it follows the same
        # device-multiplier. Without this scaling, VG1=0.2 (where
        # mbjt=0.001 keeps the BJT off) would still see full well
        # coupling and the body would float high.
        I_well_body = I_well_body * cfg.vnwell_mbjt
    else:
        I_well_body = torch.zeros_like(Vd)

    # ---- Body KCL: currents INTO B ------------------------------------ #
    # Iii, Igidl, Igisl, Igb are already signed +INTO-body in the helpers.
    # Body junction diodes: Ibs and Ibd are POSITIVE-LEAVING-body, so we
    # subtract them.
    # BJT base current Ib (positive INTO base from external) — for the
    # floating body, the only external current into the base IS the body
    # node itself. Ib>0 ⇒ body sources current → leaves body. → −Ib_Q1
    # Well-body diode I_well_body is +INTO body (well pumps body up). → +I_well_body
    # A.10: extra parasitic pdiode at floating body (Sebas's 2026-05-02
    # email). Anode = body B, cathode = one of {vnwell, GND, Sint}. Default
    # OFF — turns on once we have his SPICE card. Sign convention: I_pdiode
    # = Js·area·(exp((Vb-Vc)/(n·Vt)) - 1), positive when forward-biased,
    # leaves the body → enters R_B with negative sign.
    if cfg.body_pdiode_to != "off":
        Vt_body = 0.02585 * (273.15 + cfg.T_C) / 300.0
        if cfg.body_pdiode_to == "vnwell":
            Vc_pdi = cfg.vnwell
        elif cfg.body_pdiode_to == "gnd":
            Vc_pdi = 0.0
        elif cfg.body_pdiode_to == "sint":
            Vc_pdi = Vsint
        else:
            Vc_pdi = 0.0
        Vab = Vb - Vc_pdi
        exp_arg = (Vab / (cfg.body_pdiode_n * Vt_body)).clamp(-40.0, 40.0)
        I_body_pdiode = (cfg.body_pdiode_Js * cfg.body_pdiode_area
                          * (torch.exp(exp_arg) - 1.0))
        # Phase-B (2026-05-03 10:34): sidewall (perimeter) parallel branch.
        # Off when perim_length=0 (default). Same Vab, separate ideality and
        # saturation per Sebas's 2026-05-02 card (ns=1.0851, isw=1.3664e-13 A/m).
        if cfg.body_pdiode_perim_length > 0.0:
            exp_arg_sw = (Vab / (cfg.body_pdiode_n_sw * Vt_body)).clamp(-40.0, 40.0)
            I_body_pdiode = I_body_pdiode + (
                cfg.body_pdiode_Js_sw * cfg.body_pdiode_perim_length
                * (torch.exp(exp_arg_sw) - 1.0))
    else:
        I_body_pdiode = torch.zeros_like(Vd)

    # A.3.d: scale M1 body diodes (was clamping Vb at ~0.5V at VG1=0.4 row,
    # preventing parasitic NPN from lighting; controlled via cfg.m1_diode_scale,
    # default 1.0). Set <1 to weaken the diode shunt and let Vb climb.
    m1_d = float(cfg.m1_diode_scale)
    # M3b F1.v2 (post-O19): Iii→Vb collection efficiency η ∈ [0, 1].
    # Per O19 openai critique, the previous unbounded `iii_gain` was
    # itself a non-physical fudge factor that re-introduced an unphysical
    # gain path while we tried to clamp Bf. The correct form is a
    # bounded collection efficiency:
    #   η_eff = sigmoid(slope · (Vds − Vds_th)) ∈ [0, 1]
    # which models the fraction of channel-impact-ion holes that reach
    # the parasitic-NPN base laterally vs. diffuse to the bulk.
    # cfg.eta_max ∈ [0, 1] is a hard ceiling. Defaults: η_max=1.0,
    # slope=10/V, Vds_th=1.0 V → snapback regime.
    # If `iii_body_gain` is set explicitly (legacy), it overrides η
    # but is flagged as non-physical in the run log.
    iii_gain_legacy = getattr(cfg, "iii_body_gain", None)
    if iii_gain_legacy is not None and float(iii_gain_legacy) > 1.0 + 1e-9:
        # Legacy non-physical multiplier path. Used pre-O19; kept for
        # back-compat reproducibility, NOT for new fits.
        iii_gain = float(iii_gain_legacy) * torch.ones_like(Vd)
    else:
        eta_max = float(getattr(cfg, "eta_max", 1.0))
        eta_slope = float(getattr(cfg, "eta_slope", 10.0))
        eta_vds_th = float(getattr(cfg, "eta_vds_th", 1.0))
        Vds_eff = (Vd - 0.0)  # M2 source = GND, so Vds_M_full ≈ Vd
        iii_gain = eta_max * torch.sigmoid(eta_slope * (Vds_eff - eta_vds_th))

    # M3c.1 (charge-conserving electron–hole pair accounting). The
    # impact-ionised holes split into two destinations:
    #   η_lat   → fraction reaches the lateral parasitic-NPN base
    #   1−η_lat → fraction diffuses to the bulk body (existing F1.v2 path)
    # At η_lat=0 the routing reproduces F1.v2 exactly (regression gate).
    # The full M(Vbc)·Ids_M1 lateral collector formulation is M3c.2.
    eta_lat = float(getattr(cfg, "eta_lat", 0.0))
    iii_total_for_routing = m1["Iii"] if cfg.m2_body_gnd else (m1["Iii"] + m2["Iii"])
    Ib_lat_pair = eta_lat * iii_gain * iii_total_for_routing
    iii_to_body_factor = (1.0 - eta_lat)
    # M3c.2 path B (AUGMENT, post-O20 unanimous): the lateral-pair Ib
    # drives β·Ib of additional collector current that exits via the
    # drain (parasitic-NPN collector = drain). β = bjt.Bf (clamped at
    # the BJT's Bf parameter, which is honest-physical 100 in M3b).
    # KCL at NPN emitter (= GND): Ie_lat = Ic_lat + Ib_lat_pair.
    # GND absorbs both; drain pin sees +Ic_lat extra inflow.
    Ic_lat = float(getattr(bjt, "Bf", 100.0)) * Ib_lat_pair

    # M3c.3 local-base node (post-O21 unanimous α verdict). When
    # cfg.use_local_base=True, decouple the BJT base from the global
    # body Vb via a spread resistor Rb. Iii + GIDL inject into Vb_local;
    # BJT sees Vbe_local = Vb_local; body diodes see Vb_global.
    # Inner 1D damped-Newton solve (~6-8 iters) at each outer Newton step.
    # Default cfg.use_local_base=False → Vb_local = Vb → reduces to F1.v2.
    if cfg.use_bjt and getattr(cfg, "use_local_base", False):
        Rb = float(getattr(cfg, "lat_Rb", 1e6))
        # Total inflow into Vb_local: (1-η_lat)·iii_gain·Iii + GIDL + Ib_lat_pair
        inflow_local = (
            iii_to_body_factor * iii_gain * iii_total_for_routing
            + m1["Igidl"] + m1["Igisl"]
            + Ib_lat_pair
        )
        Vb_local = Vb.clone().detach()  # warm start at the legacy answer
        for _it in range(10):
            bjt_l = compute_bjt(bjt, Vbe=Vb_local, Vbc=Vb_local - Vd,
                                 T_K=273.15 + cfg.T_C)
            Ib_at_local = bjt_l["Ib"]
            spread = (Vb_local - Vb) / Rb
            f = inflow_local - Ib_at_local - spread
            # Finite-difference Jacobian (Ib_at_local has steep exponential)
            eps = 1e-4
            bjt_p = compute_bjt(bjt, Vbe=Vb_local + eps, Vbc=Vb_local + eps - Vd,
                                 T_K=273.15 + cfg.T_C)
            dIb_dV = (bjt_p["Ib"] - Ib_at_local) / eps
            dfdV = -dIb_dV - 1.0 / Rb
            # Guard small derivatives + clamp step magnitude for stability
            step = -f / torch.where(dfdV.abs() > 1e-30, dfdV,
                                     torch.full_like(dfdV, -1.0 / Rb))
            step = torch.clamp(step, min=-0.1, max=0.1)
            Vb_local = Vb_local + step
            if float(step.abs().max()) < 1e-10:
                break
        # Recompute BJT outputs at the converged Vb_local
        bjt_out_local = compute_bjt(bjt, Vbe=Vb_local, Vbc=Vb_local - Vd,
                                     T_K=273.15 + cfg.T_C)
        Ic_Q1 = bjt_out_local["Ic"]
        Ib_Q1 = bjt_out_local["Ib"]
        Ie_Q1 = bjt_out_local["Ie"]

    # M3c.2 path C (TOGGLE, post-O20): avalanche multiplier on the
    # channel current. Activated only when cfg.use_lateral_collector is
    # True. Default False → identical to path B baseline.
    # Vbc = Vb − Vd. In snapback regime Vd > Vb → Vbc < 0 (reverse-bias
    # B-C). We multiply only by the reverse-biased magnitude:
    #   M(Vbc) = 1 + (max(−Vbc, 0) / BV)^N      smoothed via softplus
    #   M_safe = 1 + (M − 1) · sigmoid((BV_max − |Vbc|) / δ)
    # so M saturates as |Vbc| → BV_max. Ic_avalanche is the EXTRA
    # collector current beyond Ids_M1 — added to drain only, never to
    # Ids_M1 itself (avoids double-count per O19 openai critique).
    if getattr(cfg, "use_lateral_collector", False):
        BV = float(getattr(cfg, "lat_BV", 6.0))
        N_av = float(getattr(cfg, "lat_N", 4.0))
        BV_max = float(getattr(cfg, "lat_BV_max", BV * 1.1))
        delta = float(getattr(cfg, "lat_M_smooth_delta", 0.5))
        # Reverse-bias magnitude: positive only when Vd > Vb
        Vbc = Vb - Vd
        rev_mag = torch.clamp(-Vbc, min=0.0)
        M_raw = 1.0 + (rev_mag / BV) ** N_av
        # Smooth saturation as |Vbc| approaches BV_max
        sat = torch.sigmoid((BV_max - rev_mag) / delta)
        M_safe = 1.0 + (M_raw - 1.0) * sat
        Ic_avalanche = (M_safe - 1.0) * m1["Ids"]
    else:
        Ic_avalanche = torch.zeros_like(Vd)

    if getattr(cfg, "use_local_base", False) and cfg.use_bjt:
        # M3c.3: Iii + GIDL + Ib_lat_pair routed through Vb_local.
        # Global body Vb sees only the spread current arriving from
        # local (replacing the direct Iii/GIDL terms), plus body
        # diodes / well diodes / Igb (which clamp Vb_global at ~0.4 V
        # but no longer dominate Vb_local).
        Rb = float(getattr(cfg, "lat_Rb", 1e6))
        spread_in = (Vb_local - Vb) / Rb
        if cfg.m2_body_gnd:
            R_B = (
                spread_in                       # Iii/GIDL/Ib_lat now arrive here via Rb
                + m1["Igb"]
                - m1_d * m1["Ibs"] - m1_d * m1["Ibd"]
                + I_well_body
                - I_body_pdiode
            )
        else:
            R_B = (
                spread_in
                + m1["Igb"] + m2["Igb"]
                - m1["Ibs"] - m1["Ibd"]
                - m2["Ibs"] - m2["Ibd"]
                + I_well_body
                - I_body_pdiode
            )
    elif cfg.m2_body_gnd:
        # A.1.u: M2's body is GND, so its body-current contributions do
        # NOT enter the floating-body KCL — they flow between M2's nodes
        # and ground, not the floating Vb.
        R_B = (
            iii_to_body_factor * iii_gain * m1["Iii"]
            + m1["Igidl"] + m1["Igisl"]
            + m1["Igb"]
            - m1_d * m1["Ibs"] - m1_d * m1["Ibd"]
            - Ib_Q1
            - Ib_lat_pair
            + I_well_body
            - I_body_pdiode
        )
    else:
        R_B = (
            iii_to_body_factor * iii_gain * (m1["Iii"] + m2["Iii"])
            + m1["Igidl"] + m1["Igisl"] + m2["Igidl"] + m2["Igisl"]
            + m1["Igb"] + m2["Igb"]
            - m1["Ibs"] - m1["Ibd"]
            - m2["Ibs"] - m2["Ibd"]
            - Ib_Q1
            - Ib_lat_pair
            + I_well_body
            - I_body_pdiode
        )

    # Oracle-recommended gmin shunts — ngspice-style parallel conductance
    # in PARALLEL with each pn junction, NOT a single shunt to ground.
    # This is what gives the body a tendency to track (Vd+Vs)/2 in absence
    # of other forces, matching ngspice's behavior.
    #   I_gmin_bd = gmin * (Vd - Vb)   flows INTO body from drain
    #   I_gmin_bs = gmin * (Vs - Vb) = -gmin * Vb (since Vs=0)
    #                                   flows INTO body from source
