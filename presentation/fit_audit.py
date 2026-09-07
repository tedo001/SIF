"""Estimate whether every text box's content fits its shape.

Calibri is not installed here, so measurements use DejaVu Sans, which is ~8-10%
wider at the same point size. That makes every estimate pessimistic: anything
this reports as fitting will fit in PowerPoint with room to spare.
"""
import sys
from PIL import ImageFont
from pptx import Presentation
from pptx.util import Emu

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
CACHE = {}


def font(size_pt, bold):
    key = (round(size_pt, 1), bold)
    if key not in CACHE:
        CACHE[key] = ImageFont.truetype(BOLD if bold else FONT, max(int(size_pt * 4), 4))
    return CACHE[key]


def line_count(text, size_pt, bold, width_in):
    """Lines after wrapping at width_in inches (1 pt = 1/72 in)."""
    if not text.strip():
        return 1
    f = font(size_pt, bold)
    limit = width_in * 72 * 4          # same 4x scale as the font
    words, lines, current = text.split(), 1, ""
    for word in words:
        trial = (current + " " + word).strip()
        if f.getlength(trial) <= limit or not current:
            current = trial
        else:
            lines += 1
            current = word
    return lines


def audit(path, slack=0.0):
    prs = Presentation(path)
    slide_h = Emu(prs.slide_height).inches
    issues = []
    for index, slide in enumerate(prs.slides, 1):
        for sh in slide.shapes:
            if not sh.has_text_frame or sh.width is None:
                continue
            text = sh.text_frame.text
            if not text.strip():
                continue
            width = Emu(sh.width).inches - 0.14      # internal padding
            height = Emu(sh.height).inches
            top = Emu(sh.top).inches
            needed = 0.0
            for para in sh.text_frame.paragraphs:
                runs = para.runs
                size = next((r.font.size.pt for r in runs if r.font.size), 12)
                bold = bool(runs and runs[0].font.bold)
                content = "".join(r.text for r in runs) or para.text
                lines = line_count(content, size, bold, max(width, 0.4))
                spacing = para.space_after.pt if para.space_after is not None else 2
                needed += lines * size * 1.20 / 72 + spacing / 72
            if needed > height + slack:
                issues.append((index, sh.name, round(needed, 2), round(height, 2),
                               round(top + needed, 2), slide_h,
                               text[:58].replace("\n", " ")))
    return issues


if __name__ == "__main__":
    path = sys.argv[1]
    rows = audit(path)
    print(f"{len(rows)} text box(es) estimated to overflow (DejaVu metrics, pessimistic)\n")
    for slide, name, needed, height, bottom, slide_h, text in rows:
        flag = "  << RUNS OFF THE SLIDE" if bottom > slide_h else ""
        print(f"  slide {slide} | {name[:22]:24} needs {needed:5.2f}in in {height:5.2f}in"
              f" | ends at {bottom:5.2f} of {slide_h:.2f}{flag}")
        print(f"      {text!r}")
