# Sample reports for testing

Test material for the SIF Insight Console. Nothing here is a real Oil India
record: the narratives are written to look like field reporting and to exercise
one path through the console each.

| File | Path it exercises | What to expect |
| --- | --- | --- |
| `near_miss_reports.csv` | **Import CSV of reports** | 18 reports; 13 flag SIF potential and 16 reach the review queue |
| `shift_log.txt` | **Add documents** (text) | One night-shift log; splits into 6 blocks, 5 of them analysable |
| `permit_observation.pdf` | **Add documents** (PDF text layer) | Read without OCR by `pypdfium2`; analyses as SIF potential, risk 80.7, Work Authorisation |
| `scanned_uauc_report.png` | **Add documents** (PaddleOCR) | A page with no text layer, so the OCR path has to run |
| `multilingual_report.txt` | **Language / translation** | Five blocks in Hindi, Marathi, Tamil, Telugu and Kannada |
| `languages/` | **One report per language** | Six full reports - Tamil, Hindi, Marathi, Telugu, Kannada, Urdu - each its own file, with its own README |

## Working through them

1. **Start with the CSV.** *Ingest and OCR → Import CSV of reports →
   `near_miss_reports.csv`*. Eighteen reports analyse in a few seconds.
2. **Look at the dashboard and the hotspots.** Duliajan OCS-4 and Rig-12 each
   appear more than once on purpose, so the hotspot detector has repeats to rank.
3. **Work the review queue.** Measured with the offline encoder:

   | Trigger | Count | Why |
   | --- | --- | --- |
   | Critical risk | 11 | Scored in the top band; verify before it drives an intervention |
   | Thin evidence | 5 | Short or vague narratives - the engine says so rather than guessing |

   The five low-consequence reports (NM-2609 to NM-2613: a loose plate, a canteen
   spill, back strain, and two deliberately terse ones) do not flag. That is the
   correct answer for them.

   These counts moved once already: before the barrier-vocabulary work of
   `evaluation/`, only 5 of the 18 flagged and five more reached the queue as
   "Energy, no barrier" - high energy, rule matched, no barrier recognised. They
   now carry a named barrier and arrive as findings instead. Re-measure with
   `python evaluation/evaluate.py` after any change to the knowledge base.

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

## The language samples

`languages/` holds one complete report per language rather than blocks in one
file, so each can be fed through the console on its own with the matching OCR
language selected. Measured behaviour, with the offline encoder and no Ollama:

| File | Rule matched without translation |
| --- | --- |
| `tamil_report.txt` | Energy Isolation (the words LOTO and 11 kV survive in Latin script) |
| `marathi_report.txt` | Confined Space (H2S survives) |
| `hindi_report.txt`, `telugu_report.txt`, `kannada_report.txt`, `urdu_report.txt` | Unclassified - thin evidence |

That is the point of them, not a defect: the rule engine reads English, so a
Devanagari narrative reaches it as thin evidence and **goes to a human** rather
than being silently cleared. Turn Ollama on, tick *Translate non-English reports*,
and the same files analyse on their merits.

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
