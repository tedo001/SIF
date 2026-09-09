# Sample reports for testing

Test material for the SIF Insight Console. Nothing here is a real Oil India
record: the narratives are written to look like field reporting and to exercise
one path through the console each.

| File | Path it exercises | What to expect |
| --- | --- | --- |
| `near_miss_reports.csv` | **Import CSV of reports** | 18 reports; 5 flag SIF potential, and all 18 reach the review queue across three triggers |
| `shift_log.txt` | **Add documents** (text) | One night-shift log; splits into 6 blocks, 5 of them analysable |
| `permit_observation.pdf` | **Add documents** (PDF text layer) | Read without OCR by `pypdfium2`; analyses as SIF potential, risk 80.7, Work Authorisation |
| `scanned_uauc_report.png` | **Add documents** (PaddleOCR) | A page with no text layer, so the OCR path has to run |
| `multilingual_report.txt` | **Language / translation** | Five blocks in Hindi, Marathi, Tamil, Telugu and Kannada |

## Working through them

1. **Start with the CSV.** *Ingest and OCR → Import CSV of reports →
   `near_miss_reports.csv`*. Eighteen reports analyse in a few seconds.
2. **Look at the dashboard and the hotspots.** Duliajan OCS-4 and Rig-12 each
   appear more than once on purpose, so the hotspot detector has repeats to rank.
3. **Work the review queue.** Every trigger is represented:

   | Trigger | Reports | Why |
   | --- | --- | --- |
   | Critical risk | NM-2601, 2603, 2604, 2606, 2618 | Scored in the top band; verify before acting |
   | Thin evidence | NM-2607, 2609-2613, 2614, 2615 | Short or vague narratives - the engine says so rather than guessing |
   | Energy, no barrier | NM-2602, 2605, 2608, 2616, 2617 | High energy and a rule matched, but no failed barrier recognised |

   Decide each with `1`, `2` or `3`. The bench advances by itself, decisions are
   written to disk as you go, and the trail tab keeps every one of them.
4. **Then train.** With eight or more decided reports carrying both verdicts,
   *Train model* learns from **your decisions** rather than from the engine's own
   output - the status line says which of the two it used.

## What each sample is for

**`near_miss_reports.csv`** - the main one. Columns are `report_id`, `date`,
`site`, `activity`, `reported_by`, `description`; the importer reads
`description` as the narrative and `report_id` as the reference, and would
equally accept a contractor spreadsheet with different headers.

**`shift_log.txt`** - blocks separated by blank lines, which is how the console
splits a document into report-sized pieces. Use it to check that one file
becomes several analysed rows.

**`permit_observation.pdf`** - a permit close-out with a real text layer. It
proves the PDF path works without any OCR model present.

**`scanned_uauc_report.png`** - deliberately has **no** text layer: it is a
rendered page with rotation, speckle and soft focus, so the console must fall
back to PaddleOCR. Use it to confirm an OCR install is genuinely working. If
PaddleOCR has not been installed, or its models cannot be downloaded, this file
is what makes the console say so.

**`multilingual_report.txt`** - the same kinds of incident written in five Indian
languages. With Ollama running, build 2 renders each into English before
analysis; without it, the language handling degrades and says so.

## Honest notes

* The reports were written first and analysed afterwards, not tuned until the
  engine liked them. Several realistic narratives do **not** flag - the barrier
  vocabulary does not yet cover phrasing like "car-sealed open with no tag". That
  is the recall gap the review queue exists to surface, and working the queue is
  what fixes it.
* `scanned_uauc_report.png` has been verified to contain no text layer, but its
  OCR output has not been checked in this repository's build environment, which
  cannot reach the PaddleOCR model hosts.
* **The importer reads the narrative column only.** `site`, `date`, `activity`
  and `reported_by` are in the CSV because a real export has them, but they do
  not enter the system: the engine derives location and activity from the words
  of the report itself. So searching the console for "Duliajan" finds nothing,
  while searching for "permit" finds four reports. Carrying the reported site
  through as metadata would be a change to the ingestion contract, not a bug in
  these files.
