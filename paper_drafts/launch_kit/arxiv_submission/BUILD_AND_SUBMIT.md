# arXiv submission instructions — FabricCrypt v3.1

Paper source-of-truth: `paper_drafts/fabriccrypt_v3_1.md`
(10,309 words; O116 mandatory edits applied — see
`paper_drafts/fabriccrypt_v3_1_EDIT_LOG.md` for the audit trail).

The `fabriccrypt.tex` in this directory is a condensed arXiv version
of v3.1 that propagates all O116 honest claims (5 HAL-bypass + 3
$\mu$-arch + 7 board-level = 15 signals; empirical attack-cost only,
no formal bit-security; per-die *primitive* at $n=2$ chassi; \S5.0
protocol-evolution and \S5.4 wrapped-response classifier
clarifications; Phase 21b moved out of abstract).

## Files in this directory
- `fabriccrypt.tex` — main paper source
- `fabriccrypt.bib` — bibliography
- `abstract.txt` — 250-word abstract (paste into arXiv abstract field)
- `CITATION.cff` — citation metadata
- `figures/` — PDF figures embedded in the paper

## Local build (sanity check)
```bash
cd paper_drafts/launch_kit/arxiv_submission
pdflatex fabriccrypt.tex
bibtex fabriccrypt
pdflatex fabriccrypt.tex
pdflatex fabriccrypt.tex
```
Expected output: `fabriccrypt.pdf` (~10-12 pages).

## arXiv upload
1. Build the tarball:
   ```bash
   tar -czf fabriccrypt_arxiv.tar.gz fabriccrypt.tex fabriccrypt.bib figures/
   ```
2. Go to https://arxiv.org/submit
3. **Categories**:
   - Primary: `cs.CR` (Cryptography and Security)
   - Secondary: `cs.LG` (Machine Learning)
   - Optional tertiary: `cs.AR` (Hardware Architecture)
4. **License**: CC-BY 4.0 recommended (matches MIT repo license).
5. **Title**: FabricCrypt: Software-Discoverable, Vendor-Key-Free
   Per-Die Attestation *Primitive* for AI Inference on Commodity GPUs
   (at n=2 chassi)
6. **Comments field**: "11 pages, 2 figures, 2 tables. v3.1 (O116
   mandatory edits applied). Companion code:
   https://github.com/Heigke/FabricCrypt"
7. Upload tarball.
8. Preview the rendered PDF before clicking "Submit".

## Timing
Submit Friday/Saturday evening (US time) so the announcement lands
on Monday morning UTC. arXiv announces at 20:00 ET the prior day.
This puts maximum eyes on the preprint before the Tuesday 14:00 UTC
X-thread launch.

## After submission
- Note the arXiv ID (`arXiv:XXXX.XXXXX`).
- Update `CITATION.cff` `url:` field.
- Update GitHub README link to preprint.
- Paste arXiv ID into `x_thread.md` tweet 10 and `hn_post.md`.
