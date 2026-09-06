# SIF Insight Console — Operating Instructions

Oil India Limited · Problem Statement 26165 · UA/UC and near-miss intelligence.

This document is the working manual: the rules the system must be used under, the
step-by-step process to run it, how to train the risk model, and what has to
change before it can be trusted on live safety data.

Read [Section 1](#1-ground-rules) before running anything. The rest is sequential —
Sections 2–4 get it running, Section 5 is the daily loop, Section 6 is training,
Section 7 is the path to production.

---

## 1. Ground rules

These are not style preferences. Each one exists because breaking it produces a
specific, predictable failure.

### 1.1 What the system is

A **decision-support prototype**. It reads UA/UC and near-miss narratives and
tells a safety professional where to look first. It is a triage tool, not an
investigator and not a system of record.

### 1.2 Rules of use

| # | Rule | Why |
| --- | --- | --- |
| R1 | **A human decides.** No output may close, downgrade or escalate a report on its own. | The model has no context beyond the words in the narrative. |
| R2 | **Nothing here replaces statutory reporting.** Regulatory notification follows the existing procedure, unchanged and on its own clock. | Legal duty sits with the operator, not the tool. |
| R3 | **Never treat a "not SIF-potential" as an all-clear.** The system reads only what was written; an unrecorded barrier failure cannot be detected. | Absence of evidence in a terse report is not evidence of control. |
| R4 | **Every flag must be explainable before it is acted on.** Open the report, read the decision path and cues in the evidence panel. | A flag nobody can audit is worse than no flag. |
| R5 | **Work the review queue.** Disagreements and critical-risk rows are the queue's whole purpose; an unworked queue means the model is never corrected. | Section 6.2 depends on this queue for labels. |
| R6 | **Risk scores rank; they do not predict.** A 90 outranks a 60. Neither is a probability of harm. | The score is ordinal by construction (Section 6.5). |
| R7 | **Say which encoder produced a result.** The status bar and every exported row record it. | Offline and transformer runs are not comparable. |
| R8 | **Metrics from a small corpus are indicative only.** The console warns when it had too few reports; carry that warning into any report you circulate. | Fewer than ~200 reports cannot support a defensible estimate. |

### 1.3 Data rules

| # | Rule |
| --- | --- |
| D1 | Use **de-identified narratives**. Strip worker names, contractor employee IDs and anything personal before import — the system stores whatever it is given, in `logs/`, `models/` and exports. |
| D2 | Keep the corpus **inside the asset's own environment**. The transformer and OCR models are the only things that ever come from outside, and only during setup. |
| D3 | **One incident per row or per blank-line-separated block.** Two incidents in one block are analysed as one and will be classified as one. |
| D4 | Preserve the **original narrative**. Never edit a report so it "classifies better" — fix the pattern or the label instead (Section 6.6). |
| D5 | Treat `models/`, `mlruns/`, `mlflow.db` and `logs/` as **operational data**, not source. They are git-ignored on purpose. |

### 1.4 Change rules

| # | Rule |
| --- | --- |
| C1 | `python -m unittest` must pass before any change is used on real data. |
| C2 | Every rule, energy or barrier pattern added to `sif/lexical.py` needs a test that fails without it. |
| C3 | Thresholds (`SIFClassifier.THRESHOLD`, `RiskScorer.BANDS`, `ReviewQueue.CONFIDENCE_FLOOR`) are tuned deliberately (Section 7.5), recorded, and never adjusted to make a single report come out differently. |
| C4 | A model is only promoted through the acceptance gate in Section 6.5. |

---

## 2. Prerequisites

| Requirement | Notes |
| --- | --- |
| Python 3.10+ | 3.11 is what the suite is verified against. |
| ~4 GB free disk | Mostly torch; the offline-only install is ~150 MB. |
| A desktop session | It is a Qt application. Headless servers need `QT_QPA_PLATFORM=offscreen` and can only run the CLI and tests. |
| Internet, **once** | For the sentence-transformer (~90 MB) and PaddleOCR models (~10 MB). Air-gapped sites: Section 7.2. |

---

## 3. Install — step by step

**Step 1 — get the code.**

```bash
git clone https://github.com/tedo001/SIF.git
cd SIF
git checkout tedo
```

**Step 2 — create an isolated environment.** Never install into the system Python.

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
```

**Step 3 — install.**

```bash
pip install -r requirements.txt
```

Slim install (no learned model, no OCR, no transformer — the deterministic
engine only, ~150 MB):

```bash
pip install PyQt6 numpy pypdfium2
```

**Step 4 — verify before first use.** This is Rule C1.

```bash
python -m unittest -q          # headless: QT_QPA_PLATFORM=offscreen python -m unittest -q
```

Expect `OK`. A failure here means the install is wrong; do not continue.

---

## 4. Run the application — step by step

**Step 1 — launch.**

```bash
python app.py
```

**Step 2 — confirm the stack that actually loaded.** Go to **Settings**. Three
lines tell you what you are running:

* *Model* — `no model trained yet`, or the metrics and date of the loaded one.
* *MLflow tracking* — the store, or that MLflow is missing.
* *Document ingestion* — press **Check OCR availability**; it loads the engine
  for real and reports the true outcome, including a failed model download.

**Step 3 — smoke-test with the seed data.** Dashboard → **Load 5 Seed
Incidents**. Expect 5 rows, 4 SIF-potential, 1 housekeeping observation cleared.
If you see 5 of 5 flagged, something is wrong — raise it rather than working
around it.

**Step 4 — read one report end to end.** Click row 1. The right panel must show
the rule, energy, failed barrier, activity, location, the decision path and the
cues. That is the Rule R4 check, and it is also how you learn to distrust the
tool at the right moments.

**Step 5 — choose the encoder deliberately** (ingestion panel):

| Choice | Use when |
| --- | --- |
| **Auto** | Normal operation. Transformer if it loads, offline rules otherwise. |
| **Transformer** | You require the semantic layer and want a hard failure if it is unavailable. |
| **Offline** | Air-gapped, or you need bit-identical, deterministic output. |

Equivalent environment variables, for scripted runs:

```bash
export SIF_ENCODER=auto|transformer|hashing
export SIF_ENCODER_MODEL=sentence-transformers/all-MiniLM-L6-v2   # or a local directory
```

**First transformer run downloads ~90 MB.** The status bar reports it; the window
stays responsive because loading happens on a worker thread.

---

## 5. Daily operating process

The loop, in order. Steps 1–3 are ingestion; 4–6 are where the value is.

**Step 1 — bring reports in.** One of:

* *Paste* — Dashboard ingestion box, blank line between incidents, **Analyse Report**.
* *CSV* — **Batch Import CSV**. Narrative column may be named `report`,
  `description`, `narrative`, `text`, `observation`, `details` or `incident`; a
  `report_id`/`id`/`ref` column becomes the row reference. Without a recognised
  column, the longest cell in each row is used.
* *Documents* — **Batch Upload** → **Add Documents** for PDFs, scans and
  photographs. Check the *Backend* column: `pdf-text` means the text layer was
  read exactly; `paddleocr` means it was recognised, and *OCR confidence* below
  ~0.85 deserves a look at the original. Then **Analyse Extracted Blocks**.

**Step 2 — check the KPI row.** Total, SIF-potential and %, mean risk, awaiting
review, model agreement.

**Step 3 — read the exposure charts.** Which Life-Saving Rule dominates, which
energies are live on the asset, which barrier keeps failing. The barrier chart is
usually the most actionable: it names the control to fix.

**Step 4 — work the review queue** (Rule R5), in the order it is sorted:

| Trigger | What it means | What to do |
| --- | --- | --- |
| **Disagreement** | Rules and semantic model reached opposite conclusions. | Decide which is right. This is the highest-value row in the system. |
| **Model disagreement** | The trained model contradicts the pipeline. | Same — and label it (Section 6.2). |
| **Critical risk** | Scored in the top band. | Verify before it drives an intervention. |
| **Thin evidence** | Very little extractable text. | Usually a reporting-quality problem: go back to the reporter. |
| **Unclassified exposure** | High energy, no rule matched. | Vocabulary the system has not seen — candidate for a new pattern (Rule C2). |

**Step 5 — act on hotspots.** A cluster of ≥2 reports at one location, one rule at
one location, or one barrier repeating is a *system* problem, not an incident.
Route it to the asset owner.

**Step 6 — export the record.** `File → Export Results CSV` (Ctrl+S). Every row
carries its verdict, risk, extracted fields, review trigger and explanation.

**Step 7 — check the log.** Settings → System Logging. Level filter, live view,
rotating file at `logs/sif_console.log`. Any `ERROR` line gets read before the
day's results are circulated.

---

## 6. Train the model — step by step

The learned layer (XGBoost) is a **third opinion**. It never overrides the rules;
where it disagrees, the report goes to review. Training is worth doing only in
the order below.

### 6.1 Phase 1 — baseline from the rules (day one)

With no labels, training distils the pipeline's own verdicts.

*In the app:* Settings → **Train XGBoost on analysed corpus** (needs ≥4 analysed
reports with both outcomes present).

*From the command line* — repeatable, and what you should use for anything real:

```bash
python train_model.py data/reports.csv --dry-run       # inspect the corpus first
python train_model.py data/reports.csv                 # then train
```

**Understand what this is.** The model learns to *reproduce the rules*. It adds
no knowledge. It is a baseline and a consistency check — nothing more. The run
records its label source as `weak (pipeline verdicts)` so nobody mistakes it
later.

### 6.2 Phase 2 — build a labelled corpus (weeks 1–8)

This is the step that makes the model worth having, and it is human work.

1. Run the daily loop (Section 5) so the queue fills.
2. For each queued report, a competent HSE reviewer records the **true** answer:
   was this a genuine SIF precursor?
3. Put that decision in your export as a column named `sif_label` (or `label`,
   `reviewed_sif`, `is_sif`): `1` = SIF-potential, `0` = not, **blank = not yet
   reviewed** (those rows are ignored by training, not guessed).
4. Prioritise labelling: every *Disagreement* and *Model disagreement* first —
   each one corrects either the rules or the model. Then a random sample of
   ordinary reports, so the corpus is not only hard cases.

**Target:** ≥200 labelled reports with ≥30 in the minority class before treating
any metric as meaningful. Below that the console will keep warning you, correctly.

### 6.3 Phase 3 — train on reviewed labels

```bash
python train_model.py data/reviewed_reports.csv \
    --label-column sif_label \
    --encoder transformer \
    --tracking-uri sqlite:///mlflow.db \
    --experiment oil-india-2026
```

What happens, in order: every narrative is analysed → labelled rows are kept →
features are built → an `XGBClassifier` is fitted with stratified cross-validation
→ params, metrics, feature importances and the model artifact are logged to
MLflow → the booster is saved to `models/sif_xgboost.json`.

Restart the app (or press Train once) to attach the new model; the **Model** column
and **Model agreement** tile populate from then on.

### 6.4 Read the output honestly

```
Trained: 240 samples (58 positive) | accuracy=0.883 f1=0.781 precision=0.812 recall=0.752 roc_auc=0.901
Top features: p_sif=0.31, barrier_criticality=0.14, energy::Electrical energy=0.09
```

* **Recall matters more than precision here.** A missed precursor costs more than
  an extra review. Optimise recall subject to a review load your team can carry.
* **Perfect scores mean a problem, not success.** On weak labels they are
  tautological; on real labels they mean leakage or a corpus too small to split.
* **Feature importance is a sanity check.** Barrier criticality and energy family
  near the top is expected. `n_chars` at the top means the model has learnt report
  length, not safety.
* **Warnings are part of the result.** "Small corpus … regularisation relaxed" and
  "metrics are in-sample" travel with the number (Rule R8).

### 6.5 Acceptance gate (Rule C4)

Promote a model only if **all** hold:

1. ≥200 labelled reports, ≥30 in the minority class.
2. Cross-validated **recall ≥ 0.80** on the SIF-potential class.
3. It does not contradict the rules on the seed incidents (run them; the four
   engineered precursors must stay flagged).
4. Feature importances are safety-plausible.
5. The MLflow run id is recorded in your change log.

Otherwise keep the previous model: `models/` holds the last promoted one, and
MLflow holds the history to compare against.

### 6.6 Retraining cadence

| Trigger | Action |
| --- | --- |
| +100 newly labelled reports | Retrain, compare to the previous run in MLflow. |
| Quarterly | Retrain even without new data; confirm metrics have not drifted. |
| New asset, plant or contractor | Retrain — vocabulary and barrier mix change. |
| Repeated model disagreements on one theme | Fix the *rules* first (Rule C2), then retrain. |

---

## 7. Making it real — prototype to production

Ordered by dependency. Do not skip ahead: P2 is worthless without P1, and P5 is
dangerous without P4.

### 7.1 P0 — connect the real data source

Today reports arrive by paste, CSV or document upload. In production they should
arrive on their own.

1. Get a scheduled export from the existing HSE reporting system (CSV or DB view)
   with narrative, report id, date, site and reporter role.
2. Land it in a watched folder or a table.
3. Wrap `SIFPipeline` in a scheduled job (`sif/pipeline.py` has no Qt dependency —
   this is why). `train_model.py` is the pattern to copy.
4. Write results back with the report id as the key, so the console and the source
   system agree on identity.

### 7.2 P1 — make the models available where the plant is

The two model downloads are the only external dependency, and most control-network
machines cannot reach them. Pre-fetch on a connected machine and copy:

```bash
# Sentence-transformer (on a connected machine)
python -c "from sentence_transformers import SentenceTransformer; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').save('minilm-local')"
# copy the minilm-local/ directory to the target machine, then:
export SIF_ENCODER_MODEL=/opt/sif/minilm-local
export SIF_ENCODER=transformer
```

For PaddleOCR, run one OCR job on the connected machine and copy the populated
`~/.paddlex` cache to the target machine's home directory. Verify with **Check OCR
availability** — it loads the engine for real, so a green result means the plant
machine genuinely can OCR.

*Verified caveat:* in the build environment used to develop this, the model hosts
were unreachable, so the transformer and OCR paths were exercised end-to-end
against locally-built models but never against the real published weights. **P1 is
therefore the first thing to prove on your hardware**, before P2.

### 7.3 P2 — earn the labelled corpus

Section 6.2, run for real, for two months, by named reviewers. There is no
shortcut, and no model is better than its labels. Budget it explicitly: roughly
2–4 minutes per report, ~10 hours of reviewer time for 200 reports.

### 7.4 P3 — validate the way it will be used

In-sample metrics flatter. Before trusting a model:

* **Split by time**, not at random: train on months 1–4, test on months 5–6. That
  is the real task — yesterday's reports predicting tomorrow's.
* **Split by site** for a second run: train on assets A/B, test on C. This is what
  tells you whether the vocabulary generalises.
* **Report both**, and the gap between them.

### 7.5 P4 — calibrate the thresholds to your appetite

The shipped values are reasoned defaults, not measurements:

| Setting | Where | Meaning |
| --- | --- | --- |
| `_calibrate` band (0.22–0.55) | `sif/heads.py` | Cosine similarity → confidence. Re-fit on real embeddings once P1 is done. |
| `SIFClassifier.THRESHOLD` (0.5) | `sif/heads.py` | Where the semantic path raises a flag. |
| `RiskScorer.BANDS` (70/50/30) | `sif/scoring.py` | Band cut-offs — set them so the Critical band matches what your team can actually verify each day. |
| `ReviewQueue.CONFIDENCE_FLOOR` (0.35) | `sif/review.py` | Thin-evidence trigger. |
| `PatternDetector.MIN_REPORTS` (2) | `sif/patterns.py` | When a repeat becomes a hotspot. |

Change one at a time, record the before/after on a fixed evaluation set, and keep
the record (Rule C3).

### 7.6 P5 — deploy properly

1. **Pin everything**: `pip freeze > requirements.lock.txt`, install from the lock
   file on every machine.
2. **One shared MLflow store** so runs from every analyst land in one history
   (`--tracking-uri` pointing at a shared SQLite file or a tracking server).
3. **Package** if analysts should not manage Python: PyInstaller onedir from
   `app.py`, with the model directory shipped alongside.
4. **Back up** `models/`, `mlflow.db` and the labelled corpus. Losing the corpus
   costs months (P2); losing the model costs an afternoon.
5. **Version the rules**: `sif/lexical.py` and `sif/prototypes.py` are the safety
   knowledge base. Changes go through review like code, because they are.

### 7.7 P6 — governance

* **Model card** per promoted model: corpus size and window, label provenance,
  metrics from P3, known failure modes, MLflow run id, approver.
* **Audit trail**: exports carry the explanation and cues; `logs/` carries the run
  history. Keep both for the retention period the HSE function requires.
* **Named owner** for the rules and for the model. Unowned models rot.
* **Sign-off** on any threshold change (Rule C3).

### 7.8 P7 — monitor after go-live

| Watch | Signal that something is wrong |
| --- | --- |
| Model agreement % (KPI tile) | A sustained fall means the corpus has drifted from the training set. |
| `Unclassified / General HSE` share | Rising = new vocabulary the rules do not cover. |
| Thin-evidence rate | Rising = reporting quality dropping at source; fix that, not the model. |
| Review queue age | Growing = R5 is not being honoured, and labels have stopped arriving. |
| Hotspot recurrence after an intervention | Unchanged = the intervention did not work. |

### 7.9 What must never be automated

Closing a report. Downgrading a severity. Standing down a control. Notifying a
regulator. Every one of these needs a competent person who has read the narrative
(Rules R1, R2).

---

## 8. Verification checklist

Run before first use, after any change, and after any environment move.

| # | Command / action | Expected |
| --- | --- | --- |
| 1 | `python -m unittest -q` | `OK` |
| 2 | `python app.py` | Window opens; Settings shows the encoder, MLflow and OCR states |
| 3 | Load 5 Seed Incidents | 5 rows, **4** SIF-potential |
| 4 | Click row 1 | Evidence panel shows rule, energy, barrier, decision path, cues |
| 5 | Settings → Check OCR availability | A definite answer either way (not "unknown") |
| 6 | `python train_model.py sample_reports.csv --dry-run` | Corpus summary, exit code 0 |
| 7 | Settings → Train XGBoost | Metrics appear; a run appears in *Recent training runs* |
| 8 | Batch Upload → add a PDF | Backend column reads `pdf-text` or `paddleocr` |
| 9 | `File → Export Results CSV` | File written with explanations |
| 10 | Settings → System Logging | Live lines; `logs/sif_console.log` exists |

---

## 9. Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| `PyQt6 is required` | Wrong interpreter, or venv not active | `source .venv/bin/activate`, reinstall |
| Status bar says `hashing: lexical-hash` when you chose Auto | Transformer could not load (usually no network) | Expected fallback. Section 7.2, or accept offline mode |
| Everything flagged SIF-potential | Almost always a threshold or corpus problem | Run the seed set; if it gives 4/5, your corpus really is that hazardous |
| Nothing flagged | Encoder offline **and** narratives paraphrase around the vocabulary | Add patterns (Rule C2); confirm the encoder in Settings |
| "Training refused: needs both classes" | Every analysed report has the same outcome | Analyse a wider sample; a model cannot learn one class |
| Perfect metrics | Weak labels, or too few reports | Expected in Phase 1 — see Section 6.4 |
| OCR unavailable | `paddlepaddle` missing, or models unreachable | `pip install paddleocr paddlepaddle`; Section 7.2 for offline |
| Image import fails, PDFs fine | Image needs OCR; PDF had a text layer | Same fix; the text layer never needed OCR |
| MLflow "file store maintenance mode" | A `file:` tracking URI on MLflow 3 | Use `sqlite:///mlflow.db` (the default) |
| Window cramped on a small screen | — | Pages scroll; use the scroll controls or maximise |

---

## 10. Reference

**Commands**

| Command | Purpose |
| --- | --- |
| `python app.py` | Run the console |
| `python -m unittest -q` | Full test suite |
| `python train_model.py <csv> [--label-column …]` | Train from a file |
| `python -m sif.pipeline` | Analyse the seed incidents on the command line |
| `mlflow ui --backend-store-uri sqlite:///mlflow.db` | Browse training history |

**Environment variables**

| Variable | Values | Effect |
| --- | --- | --- |
| `SIF_ENCODER` | `auto` \| `transformer` \| `hashing` | Encoder backend (an explicit choice in the app always wins) |
| `SIF_ENCODER_MODEL` | hub id or local directory | Which sentence-transformer to load |
| `QT_QPA_PLATFORM` | `offscreen` | Headless runs (tests, CI) |

**Files and directories**

| Path | Contents | In git? |
| --- | --- | --- |
| `app.py` | Launcher | yes |
| `main.py`, `ui/` | Controller and interface | yes |
| `sif/` | Pipeline, rules, OCR, MLOps, logging | yes |
| `train_model.py` | Command-line trainer | yes |
| `models/` | Trained booster + metadata | no — operational |
| `mlruns/`, `mlflow.db` | MLflow history | no — operational |
| `logs/sif_console.log` | Rotating log | no — operational |
| `reports/` | Generated PDF analysis report | yes |

**Exit codes** — `train_model.py` returns `0` on success, `1` on any refusal
(missing file, unusable label column, too few reports, single-class corpus), so it
can be scheduled and alarmed on.
