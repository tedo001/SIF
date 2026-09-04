# SIF Insight Console

Prototype for **Oil India Limited — Problem Statement 26165**: turning raw
Unsafe Act / Unsafe Condition (UA/UC) and near-miss reports into structured,
decision-grade **SIF (Serious Injury & Fatality) intelligence**.

![Console](docs/screenshot.png)

The dashboard has three panels — the incident matrix above, plus
[risk hotspots](docs/hotspots.png) and the
[human review queue](docs/review-queue.png).

## Run it

```bash
pip install -r requirements.txt
python app.py
```

Click **Load 5 Seed Incidents** for an instant demo, or **Batch Import CSV** and
pick `sample_reports.csv`. The first run downloads the sentence-transformer
(~90 MB); the status bar reports progress and the UI stays responsive.

To run with no model and no network, pick **Offline — lexical rules only** in the
encoder selector, or export `SIF_ENCODER=hashing`.

## Architecture

```
                         OIL REPORT
                             │
                    ┌────────▼────────┐
                    │ NLP PREPROCESSOR│  sif/preprocessing.py
                    └────────┬────────┘  clean · expand LOTO/PTW/GGS · segment
                             │
                  ┌──────────▼───────────┐
                  │ Semantic NLP Engine  │  sif/encoders.py
                  │ Transformer Encoder  │  all-MiniLM-L6-v2 · offline fallback
                  └──────────┬───────────┘
                             │
        ┌────────────────────┼─────────────────────┐
        ▼                    ▼                     ▼
  SIF Classifier      Rule Classifier        NER / Extraction     sif/heads.py
  P(SIF)=energy×      IOGP Life-Saving       ┌──────┼───────┐
  barrier             Rule                   ▼      ▼       ▼
        │                    │           Activity Location Barrier
        └────────────────────┼─────────────────────┘
                             ▼
                    ┌─────────────────┐
                    │ Evidence Engine │  sif/evidence.py
                    └────────┬────────┘  cues · neighbours · decision path
                             ▼
                     ┌───────────────┐
                     │ SIF Risk Score│  sif/scoring.py   0–100 + band
                     └───────┬───────┘
                  ┌──────────┴──────────┐
                  ▼                     ▼
          Pattern Detection        Human Review        sif/patterns.py
          (sif/patterns.py)        (sif/review.py)
                  │                     │
                  ▼                     │
           ┌───────────────┐            │
           │ Risk Hotspots │◄───────────┘
           └───────┬───────┘
                   ▼
            HSE Intelligence          sif/pipeline.py → Intelligence
                   ▼
               Dashboard              main.py (PyQt6)
```

### Stage by stage

| Stage | Module | What it does |
| --- | --- | --- |
| 1. Preprocessor | `sif/preprocessing.py` | Cleans the text, expands upstream shorthand (LOTO, PTW, GGS, H2S…) for the encoder while preserving the reporter's surface wording for the patterns, and segments sentences. |
| 2. Semantic encoder | `sif/encoders.py` | `all-MiniLM-L6-v2` sentence embeddings via `sentence-transformers`, behind a small interface. A deterministic hashing encoder is the offline fallback; resolution is eager, so a missing model degrades before the batch starts, not halfway through. |
| 3a. SIF classifier | `sif/heads.py` | `P(SIF) = energy_score × barrier_score` — the "high energy **AND** failed barrier" rule expressed continuously. Each factor is the max of the lexical determination and the calibrated similarity to the energy / barrier prototypes. |
| 3b. Rule classifier | `sif/heads.py` | Fuses per-rule lexical scores with cosine similarity to the IOGP rule prototypes (0.55/0.45 with a model, 0.8/0.2 without), and returns `Unclassified` below a floor rather than guessing. |
| 3c. NER / extraction | `sif/heads.py` | Activity, location and barrier: gazetteer and pattern hits first, semantic nearest-prototype only where the lexical layer fell back, and an explicit "not stated" below the similarity floor. |
| 4. Evidence engine | `sif/evidence.py` | Collects lexical cues, nearest semantic prototypes with scores, per-field provenance and the decision path, then writes the one-line explanation shown in the UI. |
| 5. Risk score | `sif/scoring.py` | `100 × P(SIF) × energy severity × barrier criticality × evidence factor`, banded Critical / High / Medium / Low. Ordinal, for ranking a queue — not an actuarial probability. |
| 6a. Pattern detection | `sif/patterns.py` | Location clusters, rule-at-location repeats and repeat barrier failures across the corpus (≥2 reports), ranked by SIF count then mean risk. |
| 6b. Human review | `sif/review.py` | Queues what a person must verify: model/rule **disagreement**, **critical risk**, **thin evidence**, or **high energy with no rule match**. |
| 7. Dashboard | `main.py` | KPI header, incident matrix, hotspot panel, review queue and an evidence pane for the selected report. |

### Why fusion, not replacement

The semantic layer **extends** the deterministic rules; it never overturns them.
A report the patterns flag stays flagged, and the model can add a flag the
patterns missed — so recall only grows. Two guards keep that honest:

* the offline fallback is explicitly non-semantic: it can rank and enrich, but it
  never raises a flag of its own, so offline mode is exactly as precise as the
  rules; and
* a **discrimination guard** — if an encoder scores every prototype alike (an
  untrained, mis-loaded or otherwise degenerate model), the ranking is treated as
  uninformative and the decision falls back to the lexical rules.

Every result records which path decided it (`evidence.decision_path`), and any
disagreement between the two goes to the human review queue.

## Module map

| File | Responsibility |
| --- | --- |
| `sif/pipeline.py` | `SIFPipeline` — orchestration, `PipelineResult`, corpus `Intelligence`. |
| `sif/lexical.py` | `LexicalEngine` — IOGP, energy, barrier, activity and location knowledge as patterns; the deterministic backbone. Holds the 5 seed narratives. |
| `sif/prototypes.py` | Natural-language label descriptions for zero-shot semantic classification. |
| `main.py` | PyQt6 layer: `MainWindow`, `AnalysisWorker` (`QThread`), KPI cards, three panels. |
| `app.py` | Launcher — dependency check, `QApplication` bootstrap, event loop. |
| `test_sif.py` | 42 unit tests across every stage, the fusion guards and the Qt worker. |
| `sample_reports.csv` | Six mock rows for the batch-import demo. |
| `reports/` | Generated analysis report (PDF). |

## Result fields

`SIFPipeline.analyze(text)` returns a `PipelineResult`. The five fields the
problem statement asks for keep their names: `sif_potential`, `iogp_rule`,
`activity`, `location`, `barrier_failure`. Alongside them:

`p_sif`, `risk_score`, `risk_band`, `energy_source`, `high_energy`,
`barrier_failed`, `lexical_flag`, `semantic_flag`, `semantic_active`,
`rule_confidence`, `confidence`, `severity_hint`, `needs_review`,
`review_trigger`, `review_reason`, `explanation`, `evidence`, `encoder`,
`elapsed_ms`.

Every extractor degrades to an explicit fallback (`Unclassified / General HSE`,
`Unspecified activity`, `Location not stated`, `No barrier failure identified`)
rather than raising, so a malformed row never breaks a batch.

## Concurrency

Model loading, CSV reading and every pipeline stage run inside
`AnalysisWorker(QThread)`. Results stream back one row at a time over
`pyqtSignal` — `row_ready(dict)`, `progress(int, int)`, `status(str)`,
`failed(str)`, `completed(int)` — so the event loop is never blocked, including
during the first-run model download. A running worker is interrupted cleanly on
window close.

## Tests

```bash
python -m unittest -v
```

On a headless machine prefix with `QT_QPA_PLATFORM=offscreen`. The suite pins the
offline encoder, so it needs no model download and is deterministic.
