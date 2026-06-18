"""C1 'fair' transplant — strip absolute-scale advantage.

Use daedalus's trained weights but with ikaros's mean/std (which the
transplant model would learn given even 1 sample). Tests: does
daedalus model capture the DYNAMICS of substrate? Or only the absolute
levels of daedalus chassis?

If fair-transplant NRMSE >> self NRMSE: daedalus model captures
ikaros-specific dynamics poorly → embodiment win is real.

If fair-transplant NRMSE ~ self NRMSE: gap was just scale mismatch →
embodiment win is trivial.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import c1_self_prediction as c1

ROOT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
OUT = ROOT / "results/IDENTITY_BENCHMARK_2026-05-30/embodiment5"


def main():
    ikaros_data = np.load(OUT / "c1_ikaros" / "c1_ikaros_data.npy")
    daed_npz = np.load(OUT / "c1_daedalus_model.npz")
    iks_npz = np.load(OUT / "c1_ikaros" / "c1_ikaros_model.npz")
    rows_fair_daed = []; rows_self = []
    for i, seed in enumerate(range(5)):
        (Xtr, Ytr), (Xte, Yte) = c1.make_windows(ikaros_data,
                                                  c1.N_TRAIN, c1.N_TEST,
                                                  seed=seed)
        # IKAROS-fit scaler (the steel-man for transplant)
        mu_i, sd_i = c1.fit_scaler(Xtr)
        Xte_si = c1.apply_scaler(Xte, mu_i, sd_i)

        # Self prediction (ikaros weights, ikaros scaler)
        m_self = c1.TinyARPred(seed=seed); m_self.W = iks_npz["W"][i]
        pred_s = m_self.predict(Xte_si)
        pred = c1.inverse_scaler(pred_s, mu_i, sd_i)
        nr_s, _ = c1.nrmse(pred, Yte)
        rows_self.append({"seed": seed, "nrmse": nr_s})

        # Fair transplant (daedalus weights, ikaros scaler)
        m_d = c1.TinyARPred(seed=seed); m_d.W = daed_npz["W"][i]
        pred_d_s = m_d.predict(Xte_si)
        pred_d = c1.inverse_scaler(pred_d_s, mu_i, sd_i)
        nr_d, _ = c1.nrmse(pred_d, Yte)
        rows_fair_daed.append({"seed": seed, "nrmse": nr_d})
        print(f"  seed={seed} self={nr_s:.4f} fair-transplant={nr_d:.4f}")

    self_med = float(np.median([r["nrmse"] for r in rows_self]))
    fair_med = float(np.median([r["nrmse"] for r in rows_fair_daed]))
    gap_pct = (fair_med - self_med) / self_med * 100
    print(f"\n[C1 FAIR] self={self_med:.4f}  fair-transplant={fair_med:.4f}  "
          f"gap={gap_pct:+.1f}% (positive = self wins on dynamics, not just scale)")
    out = {"self_med": self_med, "fair_transplant_med": fair_med,
            "gap_pct": gap_pct,
            "self_rows": rows_self, "fair_transplant_rows": rows_fair_daed,
            "PASS_gap_ge_30pct": bool(gap_pct >= 30.0)}
    (OUT / "c1_fair_transplant.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
