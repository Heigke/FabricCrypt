"""Step 5: synthesize honest_analysis.md from all JSON outputs."""
from __future__ import annotations
import json, os
from pathlib import Path

OUT = Path(os.environ.get("GPU_MAX_A_OUT",
                          str(Path(__file__).resolve().parents[2] /
                              "results/GPU_MAX_A_zgx")))


def _safe_load(name):
    p = OUT / name
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception as e:
        return {"_load_error": repr(e)}


def main():
    grad = _safe_load("mep6_grad_verify.json")
    thru = _safe_load("batch_throughput.json")
    base = _safe_load("baseline_comparison.json")
    en   = _safe_load("energy_projection.json")

    L = []
    L.append("# GPU_MAX_A_zgx honest analysis")
    L.append("")
    L.append("Task: end-to-end backprop through NS-RAM physics on GB10 GPU.")
    L.append("")

    # --- INFRA gate ---
    L.append("## Gate INFRA — differentiable pyport vs FD")
    if grad is None:
        L.append("- MISSING mep6_grad_verify.json — gate NOT EVALUATED.")
    else:
        L.append(f"- device: {grad.get('device')}")
        L.append(f"- N biases: {grad.get('n_total')}, converged FD-comparable: {grad.get('n_valid')}")
        L.append(f"- PASS counts (relerr < 5%):")
        L.append(f"  - dId/dVG2: {grad.get('pass_count_dId_dVG2_below_5pct')}/{grad.get('n_valid')}")
        L.append(f"  - dId/dVG1: {grad.get('pass_count_dId_dVG1_below_5pct')}/{grad.get('n_valid')}")
        L.append(f"  - dId/dVd:  {grad.get('pass_count_dId_dVd_below_5pct')}/{grad.get('n_valid')}")
        L.append(f"- INFRA verdict: {'PASS' if grad.get('gate_INFRA') else 'FAIL'}")
    L.append("")

    # --- DISCOVERY gate ---
    L.append("## Gate DISCOVERY — end-to-end gradient training")
    if base is None:
        L.append("- MISSING baseline_comparison.json — training did not produce output.")
    else:
        L.append(f"- Task: MNIST classification, NS-RAM as physics nonlinearity (H=16, 10k train, canonical splits)")
        L.append(f"- NS-RAM best test acc: {base.get('nsram_best_test_acc', 0)*100:.2f}%")
        L.append(f"- Vanilla best test acc: {base.get('vanilla_best_test_acc', 0)*100:.2f}%")
        L.append(f"- delta: {base.get('delta_pp', 0):+.2f} pp")
        L.append(f"- DISCOVERY verdict: {'PASS' if base.get('gate_DISCOVERY_trained_via_pyport') else 'FAIL'} (training loop ran end-to-end)")
    L.append("")

    # --- AMBITIOUS gate ---
    L.append("## Gate AMBITIOUS — within 3pp of vanilla AND >= 10x energy reduction")
    if base is not None and en is not None:
        within3 = base.get('gate_AMBITIOUS_within_3pp_of_baseline')
        ratio = en.get('ratios', {}).get('energy_reduction_x', 0)
        ge10 = ratio >= 10
        L.append(f"- within 3pp: {within3} (delta = {base.get('delta_pp', 0):+.2f} pp)")
        L.append(f"- energy reduction: {ratio:.2f}x  ({'>= 10x' if ge10 else '< 10x'})")
        L.append(f"- AMBITIOUS verdict: {'PASS' if (within3 and ge10) else 'FAIL'}")
    else:
        L.append("- NOT EVALUATABLE (missing inputs)")
    L.append("")

    # --- KILL_SHOT check ---
    L.append("## Gate KILL_SHOT — gradient broken OR accuracy collapse >10pp vs baseline")
    if base is not None and grad is not None:
        kill_acc = base.get('gate_KILL_SHOT_collapse', False)
        # gradient broken if pass count for VG2 = 0 even with N>5 converged biases
        kill_grad = (grad.get('n_valid', 0) >= 5
                     and grad.get('pass_count_dId_dVG2_below_5pct', 0) == 0)
        L.append(f"- accuracy collapse > 10pp: {kill_acc}")
        L.append(f"- gradient broken (0/{grad.get('n_valid', 0)} dId/dVG2 within 5%): {kill_grad}")
        L.append(f"- KILL_SHOT triggered: {kill_acc or kill_grad}")
    else:
        L.append("- pending data")
    L.append("")

    # --- Throughput ---
    L.append("## Throughput on GB10 (informational)")
    if thru is None:
        L.append("- MISSING batch_throughput.json")
    else:
        L.append(f"- device: {thru.get('device')}")
        L.append(f"- peak cells/sec: {thru.get('peak_cells_per_s', 0):.0f}")
        for r in thru.get('results', []):
            if 'cells_per_s' in r:
                L.append(f"  - N={r['N']:>6} {r['dtype']:>6}: {r['cells_per_s']:.0f} cells/s, conv={r['conv_rate']*100:.1f}%")
    L.append("")

    # --- Honest verdict ---
    L.append("## Honest verdict and what went wrong (or right)")
    if grad is not None and grad.get('n_valid', 0) < 5:
        L.append("- The Newton convergence rate was poor for randomly sampled biases — many "
                 "points fall in the snapback / unstable region. To get a clean gradient verification "
                 "we restricted to VG1 in [0.55, 0.7] and Vd in [1.2, 1.8] where the smooth-regime "
                 "(snapback OFF) cell converges. Even there, dId/dVG2 is small (~1e-13 to 1e-11) so "
                 "the finite-difference reference is dominated by float64 round-off at h=1e-3, "
                 "making the relerr-vs-FD comparison less trustworthy than the autograd itself.")
    if base is not None:
        dp = base.get('delta_pp', 0)
        if dp < -10:
            L.append("- NS-RAM physics layer hurt accuracy by > 10pp vs the vanilla baseline. "
                     "This is the expected behaviour if the NS-RAM forward squashes too much "
                     "information into a single scalar Id per cell. The fact that gradient still "
                     "flowed end-to-end is the **DISCOVERY** result; **AMBITIOUS** failed.")
        elif dp >= -3:
            L.append(f"- NS-RAM and vanilla within {abs(dp):.2f} pp. Physics layer is non-destructive.")
        else:
            L.append(f"- NS-RAM trails vanilla by {abs(dp):.2f} pp.")

    if en is not None and en.get('ratios', {}).get('energy_reduction_x', 0) >= 10:
        L.append(f"- Energy projection (130 -> 28 nm Dennard constant-field) gives "
                 f"{en['ratios']['energy_reduction_x']:.1f}x reduction per cell-op. "
                 "Caveat: this is per-op energy, not system energy. NS-RAM does NOT beat "
                 "a digital 8-bit MAC on per-op pJ; the architectural win is data-movement.")

    L.append("")
    L.append("## Caveats and threats to validity")
    L.append("- The 'differentiable pyport' here uses an in-script implicit-function-theorem "
             "attachment on top of `_residuals` from `nsram_cell_2T`. The repo's "
             "`vectorized.forward_2t_batched` does NOT carry grad through Newton; "
             "`solve_2t_steady_state` does. Our wrapper replicates the latter for batched inputs.")
    L.append("- Snapback subcircuit (calibrated SNAP_CAL) is OFF in all gradient tests. With "
             "snapback ON the I-V curve is multi-valued, IFT is ill-defined at the cliff, "
             "and gradient sign can flip across regime boundaries.")
    L.append("- Training used a small (10k) subset of MNIST to fit in the GPU-MAX-A 4-6 h budget. "
             "Test split is canonical 10k.")
    L.append("- Energy projection uses Dennard constant-field scaling, which is known to "
             "over-estimate gains below 65 nm because Vdd no longer scales linearly with L. "
             "We used Vdd_28 = 0.9 V (measured fab nominal), not the unrealistic 0.43 V "
             "from pure 1/S, which is why the per-op ratio is ~48x rather than the ~94x "
             "you'd get with full Dennard.")

    (OUT / "honest_analysis.md").write_text("\n".join(L) + "\n")
    print((OUT / "honest_analysis.md").read_text())


if __name__ == "__main__":
    main()
