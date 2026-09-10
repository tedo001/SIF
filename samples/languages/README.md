# Language samples for testing

One report per language, each in the script SENTRA's OCR build supports.
Use them to test the OCR language selector and, with Ollama running, the
translate-before-analysis step.

| File | Language | OCR code | Incident |
| --- | --- | --- | --- |
| `tamil_report.txt` | Tamil / தமிழ் | `ta` | Energy isolation - no LOTO on an 11 kV feeder |
| `hindi_report.txt` | Hindi / हिन्दी | `hi` | Work at height - lanyard clipped to a handrail |
| `marathi_report.txt` | Marathi / मराठी | `mr` | Confined space - no gas test, H2S found |
| `telugu_report.txt` | Telugu / తెలుగు | `te` | Lifting - crew under a suspended 4 t load |
| `kannada_report.txt` | Kannada / ಕನ್ನಡ | `ka` | Hot work - grinder beside filled LPG cylinders |
| `urdu_report.txt` | Urdu / اردو | `ur` | Driving - overtaking in fog, 11 hours at the wheel |

## How to use them

1. **Ingest and OCR** → pick the language in **OCR LANGUAGE**.
2. **Add documents** → choose the file. A `.txt` is read directly, so this
   tests the language path without needing the OCR models.
3. **Analyse extracted blocks**.
4. With Ollama running and *Translate non-English reports* ticked, the
   report is rendered into English first and the original is kept as the
   audit record. Without Ollama the text is analysed as it stands.

Each report describes an incident the rule engine already knows in English,
so you can compare: analyse the English equivalent from
`../near_miss_reports.csv` and the verdict should agree once translated.

Bengali, Gujarati, Punjabi, Malayalam, Odia and Assamese have no recogniser
in this PaddleOCR build, so no sample is provided - those reports have to be
typed in or translated at source.
