"""Phase 21 — build personality_showcase.html visual proof.

Reads gen_*.jsonl + stylometry_*/stylometry_result.json, builds a side-by-side
HTML where 10 prompts are shown anonymously with two completions (A and B).
Reader guesses which is chip vs vanilla. Reveal at bottom + report classifier
accuracy + 3 most-distinctive stylometric features.
"""
from __future__ import annotations
import os, json, argparse, html, random


def html_escape(s):
    return html.escape(s).replace('\n', '<br/>')


def load_jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def build(class_a_jsonl, class_b_jsonl, label_a, label_b,
         stylometry_json, out_html, n_pairs=10, seed=7):
    A = load_jsonl(class_a_jsonl)
    B = load_jsonl(class_b_jsonl)
    sty = json.load(open(stylometry_json))
    cls = sty['classifier']
    pm = sty['per_class_means']
    keys = sty['feat_keys']
    # Pick top-3 features by absolute diff between class means
    pa = pm.get(label_a, {}); pb = pm.get(label_b, {})
    diffs = []
    for k in keys:
        if k in pa and k in pb:
            d = pa[k] - pb[k]
            diffs.append((k, d, pa[k], pb[k]))
    diffs.sort(key=lambda x: abs(x[1]), reverse=True)
    top_feats = diffs[:5]

    # Pair by prompt_idx
    A_by_p = {}; B_by_p = {}
    for r in A: A_by_p.setdefault(r['prompt_idx'], []).append(r)
    for r in B: B_by_p.setdefault(r['prompt_idx'], []).append(r)
    common = sorted(set(A_by_p.keys()) & set(B_by_p.keys()))
    rng = random.Random(seed)
    rng.shuffle(common)
    picked = common[:n_pairs]

    # Build rows. Randomize A/B side per row (so right column not always chip).
    rows = []
    reveal = []
    for pi in picked:
        ra = A_by_p[pi][0]
        rb = B_by_p[pi][0]
        side_flip = rng.random() < 0.5
        left, right = (ra, rb) if not side_flip else (rb, ra)
        left_lbl = label_a if not side_flip else label_b
        right_lbl = label_b if not side_flip else label_a
        rows.append((ra['prompt'], left['completion'], right['completion']))
        reveal.append({'prompt_idx': pi, 'left': left_lbl, 'right': right_lbl})

    # Style
    css = """
    body { font-family: Georgia, serif; max-width: 1100px; margin: 2em auto;
           padding: 1em; line-height: 1.5; color: #222; background: #fafaf7; }
    h1, h2 { font-family: Helvetica, sans-serif; }
    h1 { border-bottom: 3px solid #333; padding-bottom: 0.3em; }
    .pair { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5em;
            border: 1px solid #ccc; border-radius: 6px; padding: 1em;
            background: white; margin-bottom: 1.5em; }
    .pair .prompt { grid-column: 1 / span 2; font-weight: bold;
                    background: #ffeec8; padding: 0.5em; border-radius: 4px; }
    .compl { font-size: 0.92em; }
    .side-label { font-weight: bold; color: #888; }
    .stats { background: white; border: 1px solid #ccc; padding: 1em;
             border-radius: 6px; margin-bottom: 1.5em; }
    .stats td { padding: 0.2em 0.8em; }
    .reveal { background: #efefe7; border: 1px dashed #888; padding: 1em;
              border-radius: 6px; }
    code { background: #eee; padding: 0.1em 0.3em; border-radius: 3px; }
    .hidden { display: none; }
    button { font-size: 1em; padding: 0.5em 1em; cursor: pointer; }
    """

    n_a = sty['n_per_class'].get(label_a, 0)
    n_b = sty['n_per_class'].get(label_b, 0)
    acc = cls['mean_acc']
    ci_lo = cls['ci_lo']; ci_hi = cls['ci_hi']
    pass_75 = "PASS (>=0.75)" if acc >= 0.75 else "FAIL (<0.75)"

    html_parts = [f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Phase 21 Personality Showcase</title><style>{css}</style></head><body>
<h1>Phase 21 — Personality bound to the chip?</h1>
<p>Below are <b>{n_pairs} prompts</b>. Each pair shows two 200-token
completions: one from a model fine-tuned with the chip's live signature
injected ("<b>{label_a}</b>"), one from a control ("<b>{label_b}</b>").
Sides are randomized. <i>Can you tell which is which without scrolling?</i></p>

<div class="stats">
<h2>Classifier verdict</h2>
<table>
<tr><td>Samples</td><td>{n_a} {label_a} vs {n_b} {label_b}</td></tr>
<tr><td>5-fold accuracy</td>
    <td><b>{acc:.3f}</b> 95% CI [{ci_lo:.3f}, {ci_hi:.3f}]</td></tr>
<tr><td>Chance</td><td>0.500</td></tr>
<tr><td>Pre-reg target</td><td>0.750</td></tr>
<tr><td>Verdict</td><td><b>{pass_75}</b></td></tr>
</table>

<h2>Top 5 stylometric differences ({label_a} − {label_b})</h2>
<table>
<tr><th>feature</th><th>{label_a}</th><th>{label_b}</th><th>Δ</th></tr>
"""]
    for k, d, ca, cb in top_feats:
        html_parts.append(
            f"<tr><td><code>{k}</code></td>"
            f"<td>{ca:.4f}</td><td>{cb:.4f}</td>"
            f"<td><b>{d:+.4f}</b></td></tr>"
        )
    html_parts.append("</table></div>")

    html_parts.append("<h2>The completions</h2>")
    for i, (prompt, left, right) in enumerate(rows):
        html_parts.append(f"""
<div class="pair">
  <div class="prompt">PROMPT {i+1}: {html_escape(prompt)}</div>
  <div class="compl"><span class="side-label">A:</span> {html_escape(left)}</div>
  <div class="compl"><span class="side-label">B:</span> {html_escape(right)}</div>
</div>""")

    html_parts.append("""
<button onclick="document.getElementById('rev').classList.toggle('hidden')">
SHOW / HIDE REVEAL</button>
<div id="rev" class="reveal hidden">
<h2>Reveal</h2><ol>""")
    for r in reveal:
        html_parts.append(
            f"<li>prompt_idx={r['prompt_idx']}: A={r['left']}, B={r['right']}</li>"
        )
    html_parts.append("</ol></div></body></html>")

    with open(out_html, 'w') as f:
        f.write('\n'.join(html_parts))
    print(f"[showcase] wrote {out_html}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--a_jsonl', required=True)
    ap.add_argument('--b_jsonl', required=True)
    ap.add_argument('--label_a', required=True)
    ap.add_argument('--label_b', required=True)
    ap.add_argument('--stylometry_json', required=True)
    ap.add_argument('--out_html', required=True)
    ap.add_argument('--n_pairs', type=int, default=10)
    args = ap.parse_args()
    build(args.a_jsonl, args.b_jsonl, args.label_a, args.label_b,
         args.stylometry_json, args.out_html, n_pairs=args.n_pairs)
