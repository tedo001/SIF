# SIH 2026 Idea Presentation — PS 26165

`SIH2026_WellDrop_SENTRA_PS26165.pptx` — built inside the official
`SIH2026 IDEA Presentation Format` template.

## Before you submit

1. **Fill the Team ID** on slide 1 (`Team ID – ______________`) from the SIH portal.
2. **Export to PDF** — the portal accepts PDF only. In PowerPoint:
   `File → Save As → PDF`. (LibreOffice is not usable in the build environment, so
   the PDF is not generated here.)
3. Check the slide count is **6** — the template's instruction slide has already
   been removed, as its own instructions require.

## Template rules this deck follows

| Rule | How it is met |
| --- | --- |
| Max 6 slides including the title | Exactly 6 |
| Avoid paragraphs; use points, diagrams, infographics | Point cards, two generated diagrams, three product screenshots |
| Use the provided template unchanged | Official template file; section titles, badges, footer and page numbers untouched |
| Delete the instructions slide | Removed |

## Slide size

| Property | Value |
| --- | --- |
| Canvas | 13.333 in x 7.5 in |
| Pixels at 96 dpi | 1280 x 720 |
| Aspect ratio | 16:9 widescreen |

This is the official template's own page setup and is left unchanged. Every
shape sits inside the canvas except the template's own title placeholders,
which the template itself bleeds slightly past the top edge.

## Rebuilding

```bash
python build_deck.py      # writes the .pptx from the template + assets/
python render.py SIH2026_WellDrop_SENTRA_PS26165.pptx qa   # approximate layout QA images
python fit_audit.py SIH2026_WellDrop_SENTRA_PS26165.pptx   # text-overflow check
```

`fit_audit.py` estimates the wrapped height of every text box and reports any
that would overflow its shape or run off the slide. It measures with DejaVu
Sans, which is wider than the template's Calibri, so its verdict is
deliberately pessimistic: a deck that passes here has headroom in PowerPoint.

`build_deck.py` expects the official template as `sih_format.pptx` in the same
directory. Diagrams in `assets/` are generated; the screenshots come from
`docs/` in the repository root.
