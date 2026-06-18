    DjctSatT = Js0_d * js_temp_factor_d

    # ---- Pre-computed Vth/Xdep quantities (b4temp.c §1322-1520) -----------
    # phi = 2·Vtm0·log(NDEP/ni) + phin   (BSIM4 v4.8.3, A.5.c fix 2026-05-01)
    # Prior code had `Vtm0·log(...) + phin + 0.4` — missing factor of 2 +
    # spurious +0.4 fudge. Three independent agents (A9 subagent, GPT-5,
    # Gemini-2.5-pro O5 packet) converged on this; produced ~60 mV uniform
    # Vth shift in compute_dc vs ngspice on isolated M2.
    ndep = max(model["ndep"], 1e10)   # safety: never log(0)
    phi = 2.0 * ctx.Vtm0 * math.log(ndep / max(ctx.ni, 1e-30)) + model["phin"]
    if phi <= 0:
        phi = 0.4   # fallback; should warn if Sebas's ndep produces this
    sqrtPhi = math.sqrt(phi)
    phis3 = sqrtPhi * phi
    # Xdep0 = sqrt(2·epssub/(q·NDEP·1e6)) · sqrtPhi  (NDEP in cm⁻³ → m⁻³ via 1e6)
    Xdep0 = math.sqrt(2.0 * ctx.epssub / (Charge_q * ndep * 1.0e6)) * sqrtPhi
    sqrtXdep0 = math.sqrt(Xdep0)
    # vbi = Vtm0·log(NSD·NDEP/ni²)
    nsd = max(model["nsd"], 1e10)
    vbi = ctx.Vtm0 * math.log(nsd * ndep / max(ctx.ni * ctx.ni, 1e-60))
    # vbsc body-bias clamp: simplified — use vbm if given, else -3.0
    vbm = model["vbm"]
    vbsc = vbm if vbm < 0 else -3.0
    # k1ox, k2ox: oxide-thickness scaling (b4temp.c §1516, 1802)
    # Same ref-default ordering issue as toxp: re-resolve toxm against toxe.
    if model.is_given("toxm"):
        toxm = model["toxm"] if model["toxm"] > 0 else ctx.toxe
    else:
        toxm = ctx.toxe
    k1ox = scaled["k1"] * ctx.toxe / toxm
    k2ox = scaled["k2"] * ctx.toxe / toxm
    # litl screening length (b4temp.c §1347)
    epsrox = ctx.epsrox
    litl = math.sqrt(3.0 * 3.9 / epsrox * model["xj"] * ctx.toxe)
    # theta0vb0 — DIBL prefactor at zero Vbs (b4temp.c §1538-1560)
    # Theta0 = exp(-0.5·dvt0·Leff/litl) approx (full form b4ld.c handles)
    # We compute a simple approximation here; full form computed in dc.py.
    theta0vb0 = math.exp(-0.5 * scaled["dvt0"] * eff.leff / max(litl, 1e-12))

    # ---- Vgsteff bridge regularizers (b4temp.c §1373, 1425-1427) ----------
    # mstar = 0.5 + atan(minv)/pi
    minv_v = scaled.get("minv", model.get("minv", 0.0))
    mstar = 0.5 + math.atan(minv_v) / math.pi
    # voffcbn = voff + voffl/Leff
    voff_v = scaled.get("voff", model.get("voff", -0.08))
    voffl_v = model.get("voffl", 0.0)
    voffcbn = voff_v + voffl_v / max(eff.leff, 1e-12)
    # cdep0 = sqrt(q·epssub·NDEP·1e6 / 2 / phi)   (b4temp.c §1373-1375)
    ndep_scaled = max(scaled.get("ndep", ndep), 1e10)
    cdep0 = math.sqrt(Charge_q * ctx.epssub * ndep_scaled * 1.0e6 / 2.0 / max(phi, 1e-3))

    # ---- Tcen / Coxeff inputs (b4ld.c §1789-1805, b4temp.c §1786, §180) ----
    # coxp = epsrox·EPS0 / toxp; if toxp not given, fall back to toxe.
    # NOTE: model_card.py resolves "ref" defaults BEFORE user overrides, so
    # toxp ends up frozen to the default-toxe value (3e-9) even if the user
    # set toxe=4e-9. Re-resolve here using is_given().
