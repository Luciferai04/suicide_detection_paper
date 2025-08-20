#!/usr/bin/env bash
# Generate PDF from the latest manuscript draft using Pandoc if available.
set -euo pipefail

MANUSCRIPTS=(results/manuscript_draft_*.md)
LATEST_MD=""
if compgen -G "results/manuscript_draft_*.md" > /dev/null; then
  LATEST_MD=$(ls -t results/manuscript_draft_*.md | head -n1)
else
  echo "No manuscript draft found yet." >&2
  exit 0
fi

OUT="${LATEST_MD%.md}.pdf"

if command -v pandoc >/dev/null 2>&1; then
  echo "Generating PDF: $OUT"
pandoc "$LATEST_MD" -o "$OUT" --pdf-engine=xelatex \
    --from markdown \
    -V documentclass=IEEEtran \
    -V geometry:margin=1in \
--citeproc \
    --bibliography=references/references.bib \
    --csl=https://www.zotero.org/styles/ieee
  echo "PDF generated at $OUT"
else
  echo "Pandoc not found; skipping PDF generation." >&2
fi

