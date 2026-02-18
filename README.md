# CongressLegislationNLP

NLP pipeline to classify U.S. Congressional legislation (~270,000 bills, 93rd–119th Congress) as China-related or not.

## Pipeline

1. **Stage 1 — Keyword filter**: Scans bill titles, summaries, and subjects using 33 China-related keywords (~90% corpus reduction).
2. **Stage 2 — File copy**: Copies matched legislation into a filtered output directory.
3. **Stage 3 (in progress)**: Fine-tuned transformer classifier on the filtered corpus.

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
```

## Data

- `raw_data/` — Raw legislation JSON files (gitignored).
- `coded_data/twl_coded_legislation_101_to_118.csv` — ~1,366 manually labeled bills (101st–118th Congress).
- `coded_data/china_filter_results.csv` — Stage 1 output manifest.
