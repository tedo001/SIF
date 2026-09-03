# SIF Insight Console

Prototype for **Oil India Limited — Problem Statement 26165**: turning raw
Unsafe Act / Unsafe Condition (UA/UC) and near-miss reports into structured,
decision-grade **SIF (Serious Injury & Fatality) intelligence**.

![Console](docs/screenshot.png)

## Run it

```bash
pip install -r requirements.txt
python app.py
```

Click **Load 5 Seed Incidents** for an instant demo, or **Batch Import CSV**
and pick `sample_reports.csv`.

## Module map

| File | Responsibility |
| --- | --- |
| `sif_engine.py` | `SIFEngine` — dependency-free heuristic parser (keyword maps, regex barrier/energy signatures) plus the 5 seed narratives. |
| `main.py` | PyQt6 layer: `MainWindow`, `AnalysisWorker` (`QThread`), `DashboardHeader`, `MetricCard`. |
| `app.py` | Launcher — dependency check, `QApplication` bootstrap, event loop. |
| `test_sif_engine.py` | 15 unit tests covering classification, fallbacks, CSV import and thread isolation. |
| `sample_reports.csv` | Six mock rows for the batch-import demo. |

## Analytical model

A report is flagged **SIF-potential** only when **both** conditions hold — the
model used by IOGP / EI serious-injury programmes:

1. a **high-energy source** is present (gravity, electrical, pressure,
   suspended load, fire, toxic atmosphere, vehicle motion, excavation, thermal), **and**
2. a **critical barrier** was absent, bypassed, defeated or ineffective.

High energy with intact barriers is controlled work. A barrier lapse with no
high energy is a housekeeping issue. The intersection is where fatalities come
from.

`SIFEngine.analyze(text)` returns:

| Key | Meaning |
| --- | --- |
| `sif_potential` | `bool` — high energy **AND** barrier failure |
| `iogp_rule` | IOGP Life-Saving Rule (Working at Height, Energy Isolation, Line of Fire, Confined Space, Safe Mechanical Lifting, Hot Work, Driving, Bypassing Safety Controls, Work Authorisation, + two upstream categories) |
| `activity` | Extracted core task |
| `location` | Extracted site section / zone |
| `barrier_failure` | Failed administrative or physical guard |
| `energy_source`, `high_energy`, `barrier_failed`, `confidence`, `severity_hint`, `evidence`, `raw_text` | Supporting evidence for audit and dashboard colour-coding |

Every extractor degrades to an explicit fallback (`Unclassified / General HSE`,
`Unspecified activity`, `Location not stated`, `No barrier failure identified`)
rather than raising, so a malformed row never breaks a batch run.

## Concurrency

All parsing **and** CSV file reading happen inside `AnalysisWorker(QThread)`.
Results are streamed row-by-row through `pyqtSignal` — `row_ready(dict)`,
`progress(int, int)`, `failed(str)`, `completed(int)` — so the Qt event loop is
never blocked and the matrix fills progressively. Controls are disabled for the
duration of a run and a running worker is interrupted cleanly on window close.

## Tests

```bash
python -m unittest -v
```

On a headless machine prefix with `QT_QPA_PLATFORM=offscreen`.
