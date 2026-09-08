# CongressLegislationNLP

NLP pipeline to classify U.S. Congressional legislation (93rd–119th Congress) as
China-related or not.

## Pipeline steps

1. **Filter** — a recall-first keyword scan of bill titles, summaries, and CRS
   subjects, cutting the corpus to a manageable candidate set. 
   recover it. Optionally copies the survivors into a filtered output tree.
2. **Annotate** — sample candidate bills into per-annotator packets, then merge
   and adjudicate the returned labels.
3. **Features** — read each labeled bill's raw `data.json` and write one row per
   bill to `data/processed/features.csv`.
4. **Classify** — classification models 
5. **Evaluate** — score predictions at a chosen probability threshold.

## Setup

```bash
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e . 
```

## Usage

```bash
python scripts/01_scan.py --congress-range 101-118 --copy
python scripts/01_scan.py --congress 119
python scripts/diagnose_filter.py
python scripts/diagnose_keywords.py
python scripts/diagnose_keywords.py --candidate "keyword one" "keyword two"
python scripts/02_build_packets.py
python scripts/03_merge_annotations.py
python scripts/04_extract_features.py
python scripts/05_train_baseline.py --view title_subjects
```

Every script in `scripts/` is a thin entry point; the code lives in
`congress_nlp/`.

## Layout

- `congress_nlp/` — the package. Five step subpackages (`filtering`,
  `annotation`, `features`, `classifiers`, `evaluation`) plus four shared leaf
  modules (`paths`, `splits`, `ids`, `rawdata`) that no step owns.
- `scripts/` — numbered entry points carrying the run order, plus unnumbered
  diagnostics.
- `tests/` — pytest suite.

## Data

- `raw_data/` — raw legislation JSON (gitignored).
- `data/raw/twl_coded_legislation_101_to_118.csv` — previous TWL manual labels
- `data/raw/Summer2026InternsData/` — TWL 2026 summer interns annotation labels.
- `data/annotation/` — annotation sprint intermediates, regenerable from the
  intern sheets.
- `data/processed/manifests/china_filter_<label>.csv` — one filter manifest per
  scanned congress or range. Adding a congress means adding a file here.
- `data/processed/filter_coverage_analysis.csv` — filter results joined with the
  gold labels.
- `data/processed/features.csv` — one row per labeled bill. Stores base columns
  only; model input text is rebuilt at load time by
  `congress_nlp/features/views.py`.
