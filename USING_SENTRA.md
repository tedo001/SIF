# Using SENTRA — step by step

*Sense the Risk · Stop the Incident*

This is the click-by-click flow. For the rules the system must be operated under,
how to train on real labels, and what has to change before a plant deployment,
read [INSTRUCTION.md](INSTRUCTION.md) instead — this file is the daily walkthrough.

Every button name below is the exact text on screen.

---

## Part 0 — Start it once

```bash
cd E:\SIF                       # wherever the project lives
.venv\Scripts\python.exe app2.py
```

The window opens on the **Workflow map**. That page is the whole system in eight
cards, and every card has a live status line telling you what is ready **on this
machine right now**. Green means ready, amber means it needs something first, red
means a dependency is missing and that capability is off.

**Read the map before you do anything else.** It answers "why is this not
working?" before you have to ask.

### One-time setup, in the order that matters

| # | Do this | Where | You are done when |
| --- | --- | --- | --- |
| 1 | **Download / verify models** (workflow card 2) — the same control on the Engines page reads **Download / verify OCR models (once)** | Workflow card 2, or Engines page | Card 2 reads `PaddleOCR ready — models are on this machine … No download needed` |
| 2 | **Start Ollama and pull a model**, then press **Check Ollama** on card 3 | A terminal: `ollama pull llama3.2` | Card 3 reads `Ollama ready — llama3.2` |
| 3 | **Check the encoder** | Engines page | Header reads `encoder: transformer` (or `hashing`, which works offline) |
| 4 | **Put your name in** | Human review → **Reviewer** box | Your name appears on every decision and audit entry |

Steps 1 and 2 are once per machine, not once per session. If card 2 still says
"download once per machine" after step 1, run `python -m sif.ocr --list` — it
prints the cache directory and what is in it.

---

## Part 1 — Get reports in

Three ways in. Pick the one that matches what you have.

### A. A spreadsheet export (the usual case)

1. **Ingest and OCR** → **Import CSV export** (or `Ctrl+O`).
2. Choose your file. `samples/near_miss_reports.csv` is a worked example.
3. The importer reads the narrative column — `report`, `description`,
   `narrative`, `text`, `observation`, `details` or `incident` — and uses
   `report_id` / `id` / `ref` as the reference. Any other columns are ignored.
4. Rows analyse as they arrive. The status bar counts them off.

> **Note:** `site`, `date` and `activity` columns are *not* read. SENTRA derives
> location and activity from the words of the report itself. Searching for a site
> name will therefore find nothing — search for what the narrative says.

### B. Documents — PDFs, scans, photographs

1. **Ingest and OCR** → **Add documents (PDF, PNG, JPG, TIFF, TXT)** (or `Ctrl+D`).
2. **Before you do**, set **OCR LANGUAGE** if the paperwork is not in English.
3. Each file is read: a PDF with a text layer is read directly (fast, exact); a
   scan or photograph goes through PaddleOCR. The **Extracted documents** table
   shows which backend was used, the page count and the OCR confidence.
4. The text is split into report-sized blocks on blank lines.
5. Click **Analyse extracted blocks**. (**Clear extraction list** drops what is
   waiting without analysing it.)

**A low OCR confidence is a warning, not a detail.** If it reads below about
0.60, check the extracted text preview before trusting the analysis.

### C. Paste one report

1. Type or paste into the **Report text** box.
2. Blank lines separate reports — paste five at once if you have five.
3. Click **Analyse text**.

Or click **Load 5 seed incidents** to see the whole system work with no data of
your own.

**The same report analysed twice replaces itself.** Import a corpus, fix a
column, import it again — you get one copy, and the status bar says how many
repeats it replaced. To start genuinely empty: **File → Clear the corpus**
(your review decisions survive; they belong to the reports).

---

## Part 2 — Read what came back

### Step 1 — Dashboard

The five tiles across the top:

| Tile | What it means |
| --- | --- |
| TOTAL REPORTS | Everything analysed in this corpus |
| SIF-POTENTIAL | High energy **and** a failed barrier — the ones that could have killed someone |
| MEAN RISK SCORE | Ordinal 0–100, for ranking a queue. Not a probability |
| AWAITING REVIEW | Still owed to a person. Falls as you work the queue |
| ENGINE AGREEMENT | How often the trained model agrees with the pipeline |

Then the three exposure charts. **The failed-barrier chart is usually the most
actionable** — it names the control that keeps failing, and a control is
something you can fix.

### Step 2 — Reports and evidence

The full matrix, one row per report. Click any row and the right-hand panel shows
the whole case: verdict, risk, the five extracted fields, and **EVIDENCE AND
REASONING** — the exact cues that produced the verdict and the decision path.

If you cannot see why a report was flagged, it is in that panel. If the panel
does not convince you, that is what the review queue is for.

### Step 3 — Risk hotspots

Sites, activities, rule-at-a-location and repeat barrier failures that occur more
than once, **ranked by SIF-precursor density, not by volume**. A small busy site
cannot hide behind a large quiet one, and a 2-of-2 group cannot outrank a
20-of-30 one (a Wilson lower bound discounts the small sample).

A hotspot is a *system* problem. Route it to the asset owner, not to the person
who filed the report.

---

## Part 3 — Work the review queue

This is the part that makes the product improve. **Human review** in the sidebar,
or **Open review queue** on workflow card 7.

### The queue

Sorted by how much a human is needed, not by date:

| Trigger | What happened | What to do |
| --- | --- | --- |
| **Disagreement** | Rules and the semantic model reached opposite conclusions | Decide which is right — highest-value row in the system |
| **Model disagreement** | The trained model contradicts the pipeline | Same, and your call becomes a label |
| **LLM disagreement** | The local model contradicts the pipeline | Same |
| **Critical risk** | Scored in the top band | Verify before it drives an intervention |
| **Thin evidence** | Very little extractable text | Usually a reporting-quality problem — go back to the reporter |
| **Unclassified exposure** | High energy, no rule matched | Vocabulary the system has not seen |
| **Energy, no barrier** | Energy and a rule matched, no failed barrier found | Confirm the barrier held, or name the one that did not |

### Deciding one

1. Click a row. The whole case appears on the right.
2. **Check the language bar** above the narrative:
   - `ENGLISH — TRANSLATED FROM TAMIL FOR REVIEW` → you are reading the English;
     **Show the original** is beside it.
   - `ENGLISH AS WRITTEN` → nothing to translate.
   - `NOT TRANSLATED …` in red → **stop**. Start Ollama and re-analyse. A verdict
     on a narrative the engine could not read is a verdict about the language.
3. Read **WHAT EACH ENGINE SAID** — rules, semantic, model, LLM, side by side.
4. Decide with the keyboard:

   | Key | Decision | Means |
   | --- | --- | --- |
   | **1** | Confirm SIF potential | This could have killed someone |
   | **2** | Not SIF potential | It could not |
   | **3** | Unclear — need more information | You looked and cannot call it |

5. The bench **advances by itself**. Work the queue without touching the mouse.

**A note is optional but it is what makes the decision reusable** six months
later. One sentence: *why*.

### Rules of the bench

- **Unclear is not a label.** It records that an expert looked and could not
  call it — different from "not SIF" — and training never sees it.
- **Overturning the engine is normal and valuable.** The row says so, and those
  are the reports that teach the system most.
- **Decisions are written to disk as you make them.** An hour of queue work
  survives a crash.
- **Changing a decision supersedes it without erasing it.** The **Decision
  trail** tab holds every entry ever recorded, and exports to CSV.
- **Undo the last decision** is for the misclick, not for a change of mind.

---

## Part 4 — Let it learn

Once the queue has produced decisions: **Engines** → **Train XGBoost on the
analysed corpus**, or **Train model** on workflow card 8.

The status line tells you which labels it used, and the difference matters:

- **`on N reviewed decision(s)`** — it learned from your experts. This is the
  real thing. Needs 8+ decided reports carrying both verdicts.
- **`on N pipeline verdict(s)`** — not enough human labels yet, so it learned to
  reproduce the rules. Useful for speed; it adds no knowledge.

Metrics land on the **Analytics** page with the feature importances — in safety
language, not `f37`. The run is logged to MLflow, so you can compare this model
against the last one and answer "which model was live, and what had it seen?"

**Recall is the number to watch.** A missed precursor is an incident nobody
looked at; a false positive costs a reviewer two minutes.

---

## Part 5 — Get things out

| You want | Do this |
| --- | --- |
| The analysed corpus | **File → Export results CSV** (`Ctrl+S`) |
| The decision trail | Human review → **Decision trail** tab → **Export the trail as CSV** |
| The audit trail | **File → Export the audit trail**, or Settings → Audit trail |
| What the software did | Settings → **System logging** (rotates) |
| Who did what, and when | Settings → **Audit trail** (append-only, never rotates) |

The audit trail separates **SYSTEM** (what the software did by itself) from
**FUNCTIONALITY** (what you asked for and what came back). Filter with **Show**,
then export.

---

## The daily loop, in six lines

1. Import the shift's reports.
2. Glance at the dashboard — how many carry fatal potential?
3. Work the review queue until it is clear. Keys 1, 2, 3.
4. Act on hotspots: fix the control, not the incident.
5. Retrain when you have decisions worth learning from.
6. Export the trail if anyone asks who decided what.

---

## When something looks wrong

| Symptom | Cause | Fix |
| --- | --- | --- |
| Card 2 keeps saying models download | Cache empty, or `PADDLE_PDX_CACHE_HOME` moved | `python -m sif.ocr --list`, then `python -m sif.ocr en` |
| Card 3 red, `not pulled` | Ollama running but the model is missing | `ollama pull llama3.2`, or point Engines at the model you have |
| Report shows a red NOT TRANSLATED bar | Analysed before Ollama was up | Start Ollama, re-analyse the document |
| A report you expected to flag did not | The barrier phrasing may not be in the vocabulary | It still reaches the queue — confirm it there, and the label teaches the model |
| Corpus counts look doubled | Older builds appended repeats | Fixed; **File → Clear the corpus** and re-import to reset |
| `encoder: hashing` when you wanted the transformer | No network, or the model is not cached | Engines → Semantic encoder → *Transformer*, then re-analyse |

## Checking the engine yourself

```bash
python evaluation/evaluate.py            # score against 40 labelled reports
python evaluation/evaluate.py --errors   # every miss, with its text
python -m unittest discover -p "test_*.py"   # the full suite
```

Add your own reports to `evaluation/labelled_reports.csv` — label them from the
safety logic, keep the negative controls, and re-run. That is how you find out
whether SENTRA works on **your** corpus rather than on ours.
