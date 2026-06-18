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
