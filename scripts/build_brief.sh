#!/usr/bin/env bash
# Build the NS-RAM proposal brief and verify figures actually embedded.
#
# Codifies the lesson from 2026-05-06 00:30: pdflatex silently rendered
# placeholder boxes for missing figures because the .tex used relative
# `figures/...` paths and was being run from results/ where that
# resolved to results/figures/ (a different unrelated directory).
# The fix was \graphicspath{{../}{./}}; this script verifies the fix
# stays in place by failing if zero raster images are embedded.
#
# Usage: scripts/build_brief.sh

set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TEX="results/nsram_proposal_short_v4_2.tex"
PDF="results/nsram_proposal_short_v4_2.pdf"
MIN_IMAGES=5  # we have 5 figures + subfigures; pdfimages lists at least this many

cd "$ROOT/results"

echo "[build] cleaning aux files..."
rm -f nsram_proposal_short_v4_2.{aux,log,out,toc}

echo "[build] pdflatex pass 1..."
pdflatex -interaction=nonstopmode nsram_proposal_short_v4_2.tex > /tmp/build_brief_p1.log 2>&1 || true

echo "[build] pdflatex pass 2 (xrefs)..."
pdflatex -interaction=nonstopmode nsram_proposal_short_v4_2.tex > /tmp/build_brief_p2.log 2>&1 || true

if [[ ! -f nsram_proposal_short_v4_2.pdf ]]; then
    echo "[FAIL] no PDF produced. See /tmp/build_brief_p2.log"
    exit 1
fi

# Check 1: page count and file size
PAGES=$(pdfinfo nsram_proposal_short_v4_2.pdf | awk '/^Pages:/ {print $2}')
SIZE=$(stat -c %s nsram_proposal_short_v4_2.pdf)
echo "[verify] pages=$PAGES, size=${SIZE} bytes"

if [[ "$PAGES" -lt 5 ]] || [[ "$PAGES" -gt 12 ]]; then
    echo "[FAIL] page count $PAGES outside expected range [5, 12]"
    exit 1
fi

# Check 2: figures actually embedded (the silent-failure killer)
N_IMG=$(pdfimages -list nsram_proposal_short_v4_2.pdf | tail -n +3 | wc -l)
echo "[verify] embedded raster images = $N_IMG"

if [[ "$N_IMG" -lt "$MIN_IMAGES" ]]; then
    echo "[FAIL] only $N_IMG raster images embedded (expected >= $MIN_IMAGES)."
    echo "       This usually means pdflatex is not finding figures and"
    echo "       silently using draft-mode placeholder boxes."
    echo "       Check that \\graphicspath{{../}{./}} is in the preamble"
    echo "       and that you are running from results/ (or project root)."
    grep -i "file.*not found\|! Package pdftex.def Error" /tmp/build_brief_p2.log || true
    exit 1
fi

# Check 3: no undefined references
UNDEF=$(grep -c "Reference.*undefined" nsram_proposal_short_v4_2.log || true)
if [[ "$UNDEF" -gt 0 ]]; then
    echo "[WARN] $UNDEF undefined references after pass 2"
    grep "Reference.*undefined" nsram_proposal_short_v4_2.log
fi

echo "[OK] brief built: $PAGES pages, $SIZE bytes, $N_IMG raster images"
echo "[OK] $PDF"
