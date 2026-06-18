"""z388 — Test A: Vdsat / (Vds-Vdsat) / Iii diagnostic at VG1 ∈ {0.2, 0.4, 0.6}.

Hypothesis: at high VG1=0.6, the source-degeneration R_S (vnwell_Rs ~ 1e10 by
default but in PER_VG1 ~ 1e7 at VG1=0.6) drives Vsint so high that
Vds_M1 = Vd - Vsint shrinks to ≤ Vdsat. Then diffVds = (Vds - Vdsat) → 0 and
the BSIM4 impact-ion arm

        Iii = T2 · diff · exp(-beta0/diff) · Idsa·Vdseff

collapses (exp argument → -∞). VG1=0.2 / 0.4 don't suffer because R_S is
much larger (1889 / 1092 Ω in PER_VG1 mapping) ... wait, actually the
PER_VG1 R values are in the OPPOSITE direction (0.2 → 1889, 0.6 → 417),
which means at VG1=0.6 the well series-R is SMALLEST → least Vsint pumping.
Either way, we measure the actual Vdsat at the converged operating point.

Outputs:
  results/z388_vdsat/summary.json
  results/z388_vdsat/vdsat_diagnostic.png

Pass criterion (DISCOVERY): if Iii(VG1=0.6) < (1/100)·Iii(VG1=0.2) at Vd=1.5,
that confirms Vdsat-saturation is THE cause for VG1=0.6 fold being inverted.

Config: clamp-off + etab=20 (matches S3-B finding where fold appears at
VG1=0.2/0.4 but not at 0.6).
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _z384_shared import (ROOT, build_base, load_sebas_params, load_measured,
                          find_or_impute_row, make_overrides, patch_sd_scaled,
                          PER_VG1)

OUT = ROOT / "results/z388_vdsat"; OUT.mkdir(parents=True, exist_ok=True)
LOG = OUT / "run.log"

VG1_LIST = [0.2, 0.4, 0.6]
VG2_BY_VG1 = {0.2: 0.10, 0.4: 0.20, 0.6: 0.20}
ETAB = 20.0
VD_PROBE = 1.5  # voltage for the Iii ratio reporting


def _log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f: f.write(line + "\n")


def diagnose_vg1(cfg, M1, M2, bjt, rows, vg1, vg2, etab=ETAB):
    """Run forward_2t in clamp-off + etab=20 mode, then for each Vd compute
    Vdsat / Vdseff / Iii of M1 explicitly via compute_dc + compute_iimpact
    at the converged Vsint."""
    from nsram.bsim4_port.nsram_cell_2T import forward_2t
    from nsram.bsim4_port.dc import compute_dc
    from nsram.bsim4_port.leak import compute_iimpact

    # Apply PER_VG1 calibration (iii_body_gain, vnwell_Rs).
    _, iii, Rs = PER_VG1[vg1]
    cfg.iii_body_gain = iii
    cfg.vnwell_Rs = Rs

    sd_M1 = cfg.size_dep_M1(M1); sd_M2 = cfg.size_dep_M2(M2)

    Vd_m, Id_m, _fname = load_measured(vg1, vg2)
    row = find_or_impute_row(rows, vg1, vg2)
    P_M1, P_M2 = make_overrides(row, etab_override=etab)
    Vd_t = torch.tensor(Vd_m, dtype=torch.float64)

    with patch_sd_scaled(sd_M1, P_M1), patch_sd_scaled(sd_M2, P_M2):
        out = forward_2t(cfg, model_M1=M1, model_M2=M2, bjt=bjt, Vd_seq=Vd_t,
                         VG1=torch.tensor(vg1, dtype=torch.float64),
                         VG2=torch.tensor(vg2, dtype=torch.float64),
                         warm_start=True)
        Vsint = out["Vsint"].detach().to(torch.float64).cpu().numpy().reshape(-1)
        Vb    = out["Vb"].detach().to(torch.float64).cpu().numpy().reshape(-1)
        Ids_M1 = np.abs(out["Ids_M1"].detach().to(torch.float64).cpu().numpy().reshape(-1))
        Id_p   = np.abs(out["Id"].detach().to(torch.float64).cpu().numpy().reshape(-1))

        # Per-Vd: bias mapping for M1 is Vgs = VG1 - Vsint, Vds = Vd - Vsint, Vbs = Vb - Vsint.
        # Call compute_dc and compute_iimpact one Vd at a time (keeps M1 SD scaled).
        Vdsat_arr = np.zeros_like(Vd_m); Vdseff_arr = np.zeros_like(Vd_m)
        Iii_arr   = np.zeros_like(Vd_m); diff_arr  = np.zeros_like(Vd_m)
        Vds_arr   = np.zeros_like(Vd_m); Vgs_arr   = np.zeros_like(Vd_m)
        Vbs_arr   = np.zeros_like(Vd_m)
        for i, vd in enumerate(Vd_m):
            vgs = float(vg1)  - float(Vsint[i])
            vds = float(vd)   - float(Vsint[i])
            vbs = float(Vb[i]) - float(Vsint[i])
            try:
                dc = compute_dc(M1, sd_M1,
                                Vgs=torch.tensor(vgs, dtype=torch.float64),
                                Vds=torch.tensor(vds, dtype=torch.float64),
                                Vbs=torch.tensor(vbs, dtype=torch.float64))
                iii = compute_iimpact(M1, sd_M1, dc, Vds=torch.tensor(vds, dtype=torch.float64))
                Vdsat_arr[i]  = float(dc.Vdsat.item())
                Vdseff_arr[i] = float(dc.Vdseff.item())
                Iii_arr[i]    = float(iii.item())
                diff_arr[i]   = float(vds - dc.Vdseff.item())
                Vds_arr[i]    = vds
                Vgs_arr[i]    = vgs
                Vbs_arr[i]    = vbs
            except Exception as e:
                Vdsat_arr[i]  = float("nan")
                Vdseff_arr[i] = float("nan")
                Iii_arr[i]    = float("nan")
                _log(f"  VG1={vg1} Vd={vd:.3f}: EXC {e}")
    return dict(Vd=Vd_m, Id_m=Id_m, Id_p=Id_p, Ids_M1=Ids_M1,
                Vsint=Vsint, Vb=Vb, Vdsat=Vdsat_arr, Vdseff=Vdseff_arr,
                Vds=Vds_arr, Vgs=Vgs_arr, Vbs=Vbs_arr,
                diffVds=diff_arr, Iii=Iii_arr)


def main():
    if LOG.exists(): LOG.unlink()
    rows = load_sebas_params()
    t0 = time.time()

    # Build clamp-off cfg (matches S3-B): body_pdiode='off', use_well_diode=False.
    cfg, M1, M2, bjt = build_base()
    cfg.use_well_diode = False
    cfg.body_pdiode_to = "off"
    _log(f"clamp-off + etab={ETAB}  use_well_diode={cfg.use_well_diode}  "
         f"body_pdiode_to={cfg.body_pdiode_to}")

    results = {}
    for vg1 in VG1_LIST:
        vg2 = VG2_BY_VG1[vg1]
        _log(f"=== VG1={vg1} VG2={vg2} ===")
        r = diagnose_vg1(cfg, M1, M2, bjt, rows, vg1, vg2)
        results[vg1] = r
        # Headline numbers at Vd≈VD_PROBE.
        idx = int(np.argmin(np.abs(r["Vd"] - VD_PROBE)))
        _log(f"  Vd={r['Vd'][idx]:.3f}  Vsint={r['Vsint'][idx]:.4f}  "
             f"Vds_M1={r['Vds'][idx]:.4f}  Vdsat={r['Vdsat'][idx]:.4f}  "
             f"diffVds={r['diffVds'][idx]:.4e}  Iii={r['Iii'][idx]:.3e}  "
             f"Ids_M1={r['Ids_M1'][idx]:.3e}")
        # Also max(Iii) over the sweep
        ii_max = float(np.nanmax(r["Iii"]))
        _log(f"  max(Iii) over sweep = {ii_max:.3e}")

    # Headline ratio at VD_PROBE: Iii(VG1=0.6)/Iii(VG1=0.2)
    def at_probe(arr, vd):
        idx = int(np.argmin(np.abs(vd - VD_PROBE)))
        return float(arr[idx])
    Iii_02 = at_probe(results[0.2]["Iii"], results[0.2]["Vd"])
    Iii_06 = at_probe(results[0.6]["Iii"], results[0.6]["Vd"])
    ratio  = (Iii_06 / Iii_02) if Iii_02 > 0 else float("inf")
    _log(f"Iii(0.6)/Iii(0.2) @ Vd={VD_PROBE} = {ratio:.3e}")
    discovery = ratio < 1.0/100.0

    # --- Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    cmap = {0.2: "tab:blue", 0.4: "tab:orange", 0.6: "tab:red"}
    # (a) Vdsat & Vds(M1) vs Vd
    ax = axes[0]
    for vg1 in VG1_LIST:
        r = results[vg1]
        ax.plot(r["Vd"], r["Vdsat"], color=cmap[vg1], ls="-", label=f"Vdsat VG1={vg1}")
        ax.plot(r["Vd"], r["Vds"],   color=cmap[vg1], ls=":", alpha=0.6)
    ax.set_xlabel("Vd_cell [V]"); ax.set_ylabel("V")
    ax.set_title("Vdsat (solid) & Vds_M1 (dotted)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    # (b) Iii(Vd)
    ax = axes[1]
    for vg1 in VG1_LIST:
        r = results[vg1]
        ax.semilogy(r["Vd"], np.maximum(r["Iii"], 1e-30), color=cmap[vg1], label=f"VG1={vg1}")
    ax.set_xlabel("Vd_cell [V]"); ax.set_ylabel("Iii_M1 [A]")
    ax.set_title("Impact-ion current Iii_M1"); ax.legend(); ax.grid(alpha=0.3, which="both")
    # (c) (Vds - Vdsat) vs Vd
    ax = axes[2]
    for vg1 in VG1_LIST:
        r = results[vg1]
        ax.plot(r["Vd"], r["diffVds"], color=cmap[vg1], label=f"VG1={vg1}")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("Vd_cell [V]"); ax.set_ylabel("Vds_M1 − Vdsat [V]")
    ax.set_title("diffVds (IIMOD exp arg)"); ax.legend(); ax.grid(alpha=0.3)
    fig.suptitle(f"z388 Vdsat diagnostic  clamp-off + etab={ETAB}  "
                 f"Iii(0.6)/Iii(0.2)@Vd=1.5 = {ratio:.2e}")
    fig.tight_layout()
    fig.savefig(OUT / "vdsat_diagnostic.png", dpi=120)
    plt.close(fig)

    # --- Summary ---
    elapsed = time.time() - t0
    def _to_list(d):
        return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in d.items()}
    summary = {
        "config": {"clamp_off": True, "etab": ETAB, "vd_probe": VD_PROBE},
        "per_vg1": {str(vg1): _to_list(r) for vg1, r in results.items()},
        "headline": {
            "Iii_at_Vd1.5_VG1_0.2": Iii_02,
            "Iii_at_Vd1.5_VG1_0.6": Iii_06,
            "ratio_06_over_02":     ratio,
            "discovery_under_1e-2": discovery,
        },
        "elapsed_s": elapsed,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    _log(f"DONE in {elapsed:.1f}s  ratio={ratio:.3e}  DISCOVERY={discovery}")


if __name__ == "__main__":
    main()
