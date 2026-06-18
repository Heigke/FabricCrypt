"""Phase 21B — final report writer.

Loads:
  - results/train_log_vanilla_dae_200.json
  - results/train_log_chip_dae_200.json
  - results/gen_vanilla.jsonl
  - results/gen_chip.jsonl
  - results/stylometry_result.json

Emits PHASE21B_REPORT.md with:
  - PASS/FAIL gate
  - sample counts, thermal events
  - top discriminative tokens
  - mean-feature deltas
  - honest verdict
"""
from __future__ import annotations
import os, sys, json, argparse
import numpy as np


def load_json(p):
    if not os.path.exists(p):
        return None
    return json.load(open(p))


def count_jsonl(p):
    if not os.path.exists(p):
        return 0
    return sum(1 for _ in open(p))


def main(results_dir, out_md):
    vlog = load_json(os.path.join(results_dir, 'train_log_vanilla_dae_200.json'))
    clog = load_json(os.path.join(results_dir, 'train_log_chip_dae_200.json'))
    n_van = count_jsonl(os.path.join(results_dir, 'gen_vanilla.jsonl'))
    n_chip = count_jsonl(os.path.join(results_dir, 'gen_chip.jsonl'))
    sty = load_json(os.path.join(results_dir, 'stylometry_result.json'))

    L = []
    L.append("# Phase 21B Report — Personality Emergence Test (RETRY)\n")
    L.append("Date: 2026-06-01\n")
    L.append("Host: daedalus (gfx1151, distilgpt2 82M)\n")
    L.append("Thermal band: 68/62/50 (abort/pause/cool) — STRICT, no exceptions\n\n")

    L.append("## Training\n")
    for name, lg in [('vanilla', vlog), ('chip', clog)]:
        if lg is None:
            L.append(f"- **{name}**: NO LOG\n")
            continue
        L.append(f"- **{name}**: steps_done={lg.get('steps_done')}/200, "
                 f"thermal_aborted={lg.get('thermal_aborted')}, "
                 f"thermal_events={len(lg.get('thermal_events', []))}, "
                 f"mean_step_ms={lg.get('mean_step_ms', 0):.0f}, "
                 f"wall_s={lg.get('wall_s', 0):.0f}\n")
        if lg.get('losses'):
            losses = lg['losses']
            L.append(f"  - loss: first 5={[round(x,2) for x in losses[:5]]} "
                     f"last 5={[round(x,2) for x in losses[-5:]]}\n")
            L.append(f"  - max temp during train: {max(lg.get('temp_log',[0])):.1f}C\n")

    L.append("\n## Generation\n")
    L.append(f"- vanilla completions: {n_van}\n")
    L.append(f"- chip completions: {n_chip}\n")

    L.append("\n## Stylometric classifier\n")
    if sty is None:
        L.append("- NO STYLOMETRY RESULT\n")
    else:
        cls = sty.get('classifier', {})
        L.append(f"- n_records: {sty.get('n_records')}\n")
        L.append(f"- classes: {sty.get('classes')} counts={sty.get('n_per_class')}\n")
        L.append(f"- 5-fold mean acc: {cls.get('mean_acc'):.3f} "
                 f"CI=[{cls.get('ci_lo'):.3f}, {cls.get('ci_hi'):.3f}]\n")
        L.append(f"- fold accs: {[round(x,3) for x in cls.get('fold_accs', [])]}\n")
        L.append(f"- pre-reg threshold: {sty.get('pre_reg_threshold')}\n")
        L.append(f"- **pre-reg PASS: {sty.get('pre_reg_pass')}**\n")

        L.append("\n### Top discriminative tokens (log-ratio)\n")
        for lbl, toks in (sty.get('top_tokens_by_class') or {}).items():
            L.append(f"**{lbl}** favors:\n")
            for entry in toks[:10]:
                w, lr, ca, cb = entry[:4]
                L.append(f"  - `{w}` lr={lr:.2f} (n={ca} vs rivals={cb})\n")
            L.append("\n")

        L.append("\n### Per-class mean stylometric features\n")
        means = sty.get('per_class_means', {})
        if len(means) >= 2:
            classes = sorted(means.keys())
            keys = sorted(next(iter(means.values())).keys())
            L.append("| feature | " + " | ".join(classes) + " | delta |\n")
            L.append("|---|" + "|".join(["---"] * (len(classes) + 1)) + "|\n")
            for k in keys:
                vals = [means[c].get(k, 0.0) for c in classes]
                delta = vals[-1] - vals[0] if len(vals) >= 2 else 0
                L.append("| " + k + " | "
                         + " | ".join(f"{v:.4f}" for v in vals)
                         + f" | {delta:+.4f} |\n")

    L.append("\n## Chip-injection magnitude (sanity)\n")
    if clog and clog.get('sig_norm_log'):
        s = clog['sig_norm_log']
        L.append(f"- chip |sig| mean={np.mean(s):.2f}, "
                 f"std={np.std(s):.2f}, min={np.min(s):.2f}, max={np.max(s):.2f}\n")
        L.append(f"- alpha (perturbation magnitude): {clog.get('alpha_lora')}\n")

    L.append("\n## Honest verdict\n")
    if sty is None:
        L.append("- STYLOMETRY NOT RUN — cannot judge\n")
    else:
        passed = sty.get('pre_reg_pass')
        acc = sty.get('classifier', {}).get('mean_acc', 0)
        if passed:
            L.append(f"- **PASS**: classifier achieved {acc:.3f} >= 0.75. "
                     "Stylometric signal IS detectable between vanilla and "
                     "chip-injected distilgpt2 outputs.\n")
        else:
            L.append(f"- **FAIL**: classifier {acc:.3f} < 0.75. "
                     "No detectable personality emergence from chip injection "
                     "at this scale/duration.\n")
        L.append("\n### Caveats\n")
        L.append("- distilgpt2 is small (82M); 200 steps is short\n")
        L.append("- LoRA-style perturbation is rank-1 per attention layer per step\n")
        L.append("- alpha=1e-3 chosen for thermal safety; larger may amplify signal\n")
        L.append("- stylometric features are surface-level (POS-lite, token freq); "
                 "deeper signal may exist semantically\n")
        L.append("- baseline drift from initial weights is identical (same starting checkpoint), "
                 "so any difference = chip injection effect\n")

    os.makedirs(os.path.dirname(out_md) or '.', exist_ok=True)
    open(out_md, 'w').write(''.join(L))
    print(f"[report] wrote {out_md}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--results_dir', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    main(args.results_dir, args.out)
