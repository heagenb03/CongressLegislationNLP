# CongressLegislationNLP

NLP pipeline to classify U.S. Congressional legislation (~270,000 bills, 93rd–119th Congress) as China-related or not.

## Pipeline

1. **Stage 1 — Keyword filter**: Scans bill titles, summaries, and subjects using 58 China-related keywords (~90% corpus reduction). Achieves 100% recall and 82.6% precision on the labeled set.
2. **Stage 2 — File copy**: Copies matched legislation into a filtered output directory.
3. **Stage 3 (planned)**: Fine-tuned transformer classifier on the filtered corpus.

## Setup

```bash
# Activate virtual environment
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Run Stage 1 + Stage 2
python main.py

# Analyze Stage 1 filter coverage against gold-standard labels
python scripts/analyze_filter_coverage.py

# Analyze per-keyword TP/FP/unique-TP stats
python scripts/analyze_keyword_effectiveness.py

# Test candidate keywords without modifying constants.py or re-running the pipeline
python scripts/analyze_keyword_effectiveness.py --candidate "keyword one" "keyword two"
```

## Data

- `raw_data/` — Raw legislation JSON files (gitignored).
- `data/raw/twl_coded_legislation_101_to_118.csv` — 1,216 manually labeled bills (101st–118th Congress): 1,017 positive, 199 negative, 0 unlabeled.
- `data/processed/china_filter_results.csv` — Stage 1 output manifest.
- `data/processed/filter_coverage_analysis.csv` — Per-bill filter coverage joined with gold-standard labels.
- `data/processed/features.csv` — Extracted ML features for each labeled bill.
