"""course_build_html.py — build static HTML bundle for the NS-RAM course.

Generates docs/course/html/index.html + one .html per module with inline
figure paths (relative), styled like GitHub. Open index.html in any
browser — no server needed.
"""
from __future__ import annotations
from pathlib import Path
import markdown
import re

SRC = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/docs/course")
OUT = SRC / "html"
OUT.mkdir(exist_ok=True)

CSS = """
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;
max-width:900px;margin:2em auto;padding:0 1em;line-height:1.6;color:#24292e;background:#fff}
h1{border-bottom:1px solid #eaecef;padding-bottom:0.3em}
h2{border-bottom:1px solid #eaecef;padding-bottom:0.3em;margin-top:1.5em}
h3{margin-top:1.2em}
code{background:#f6f8fa;padding:0.2em 0.4em;border-radius:3px;font-size:85%}
pre{background:#f6f8fa;padding:16px;overflow:auto;border-radius:6px}
pre code{background:transparent;padding:0}
blockquote{color:#6a737d;border-left:4px solid #dfe2e5;padding:0 1em;margin:0}
img{max-width:100%;height:auto;display:block;margin:1em auto;border:1px solid #ddd;
background:#fff;padding:4px;border-radius:4px}
table{border-collapse:collapse;margin:1em 0}
th,td{border:1px solid #dfe2e5;padding:6px 13px}
th{background:#f6f8fa}
nav{background:#f6f8fa;padding:1em 1.5em;border-radius:6px;margin-bottom:2em}
nav ul{margin:0.3em 0;padding-left:1.5em}
a{color:#0366d6;text-decoration:none}
a:hover{text-decoration:underline}
.nav-bottom{margin-top:3em;padding-top:1em;border-top:1px solid #eaecef;
display:flex;justify-content:space-between}
.quiz{background:#fffbea;border-left:4px solid #f1c40f;padding:1em 1.5em;margin:1em 0;
border-radius:0 6px 6px 0}
"""

MODULES = [
    ("README.md", "index.html", "Index"),
    ("01_semi_basics.md", "01_semi_basics.html", "1 — Halvledargrund"),
    ("02_mosfet.md", "02_mosfet.html", "2 — MOSFET"),
    ("03_compact_models.md", "03_compact_models.html", "3 — Kompaktmodeller"),
    ("04_floating_body.md", "04_floating_body.html", "4 — Floating body"),
    ("05_parasitic_bjt.md", "05_parasitic_bjt.html", "5 — Parasitisk BJT"),
    ("06_impact_ionization.md", "06_impact_ionization.html", "6 — Impact ionization"),
    ("07_bsim_soi.md", "07_bsim_soi.html", "7 — BSIM-SOI"),
    ("08_1t_dram.md", "08_1t_dram.html", "8 — 1T-DRAM"),
    ("09_nsram.md", "09_nsram.html", "9 — NS-RAM"),
    ("10_method.md", "10_method.html", "10 — Metod"),
    ("answers.md", "answers.html", "Facit"),
]


def build_nav(current_idx):
    prev_txt = next_txt = ""
    if current_idx > 0:
        p_src, p_out, p_title = MODULES[current_idx - 1]
        prev_txt = f'<a href="{p_out}">← {p_title}</a>'
    if current_idx < len(MODULES) - 1:
        n_src, n_out, n_title = MODULES[current_idx + 1]
        next_txt = f'<a href="{n_out}">{n_title} →</a>'
    return f'<div class="nav-bottom"><div>{prev_txt}</div><div>{next_txt}</div></div>'


def build_top_nav():
    items = "\n".join(f'<li><a href="{out}">{title}</a></li>'
                       for _, out, title in MODULES[1:-1])
    return (f'<nav><strong>Kurs:</strong><ul>{items}</ul>'
             f'<a href="answers.html">Facit</a></nav>')


MATHJAX = r"""
<script>
MathJax = {
  tex: {
    inlineMath: [['$', '$'], ['\\(', '\\)']],
    displayMath: [['$$', '$$'], ['\\[', '\\]']],
    processEscapes: true
  },
  options: { skipHtmlTags: ['script','noscript','style','textarea','pre','code'] }
};
</script>
<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
"""


def wrap(title, body_html, current_idx):
    top_nav = build_top_nav() if current_idx != 0 else ""
    bottom = build_nav(current_idx)
    return f"""<!DOCTYPE html>
<html lang="sv"><head>
<meta charset="utf-8"><title>{title}</title>
<style>{CSS}</style>
{MATHJAX}
</head><body>
{top_nav}
{body_html}
{bottom}
</body></html>"""


def convert(src_name, out_name, current_idx):
    src = SRC / src_name
    md_text = src.read_text()
    # Fix image paths from "figures/foo.png" → "../figures/foo.png"
    md_text = re.sub(r'!\[([^\]]*)\]\(figures/([^)]+)\)',
                      r'![\1](../figures/\2)', md_text)
    # Fix internal .md links → .html
    md_text = re.sub(r'\]\((\d+_[a-z_]+)\.md\)', r'](\1.html)', md_text)
    md_text = re.sub(r'\]\(answers\.md\)', r'](answers.html)', md_text)
    md_text = re.sub(r'\]\(README\.md\)', r'](index.html)', md_text)
    # Wrap quiz sections in styled div
    html = markdown.markdown(md_text,
                               extensions=["tables", "fenced_code", "toc"])
    # Style quiz questions: find "## Quiz" and wrap following content
    html = re.sub(r'(<h2>Quiz</h2>)(.*?)(<hr\s*/?>|$)',
                   lambda m: m.group(1) + '<div class="quiz">' + m.group(2) + '</div>' + m.group(3),
                   html, flags=re.DOTALL)
    title = out_name.replace(".html", "").replace("_", " ")
    (OUT / out_name).write_text(wrap(title, html, current_idx))


for i, (src, out, _) in enumerate(MODULES):
    convert(src, out, i)

print(f"Built {len(MODULES)} HTML files in {OUT}")
print(f"Open: file://{OUT}/index.html")
