# Intern Annotation Sprint — Design

**Date:** 2026-07-21
**Status:** Approved design → ready for implementation plan
**Owner:** Heagen

## Problem

Eight high-school interns are available for one week (~3–4 hrs of hands-on work
total each) to hand-annotate Congressional legislation as China-related (Yes) or
not (No). The current labeled set has two weaknesses this sprint should fix:

1. **Broken test split.** The 118th-Congress test set is 96.9% positive (only 6
   negatives), so precision/F1 on it are meaningless.
2. **Too few negatives.** The training data is 84% positive (1,017 pos / 199 neg).
   The Stage-2 transformer's only job is to reject China-*mentioning* bills that
   are not China-*focused*, and it has just 199 examples of that hard class.

The interns must do **only one thing**: read a bill and pick Yes / No / Unsure.
No data entry, no formulas, no risk of Excel coding errors. Their labels must
then flow cleanly into the training/validation/test dataset.

## Goals

- Produce a **fresh, balanced Congress 119 test set** (replacing the broken 118 split).
- Produce a batch of **hard training negatives** from Congresses 101–116.
- Guarantee **label reliability** via double-annotation + adjudication + a hidden
  gold-standard trap set, and produce a per-intern accuracy number for the paper.
- Make the intern-facing task **error-proof**: dropdown-only, one bill at a time.
- Make **migration back into the dataset fully scripted** — no manual merging.

## Non-goals

- Labeling the full ~18k unlabeled Stage-1 survivors (out of scope for one week).
- Building a custom annotation web app (good future idea at thousands-of-bills
  scale; not worth the ROI for a ~450-bill pilot — a hardened Google Sheet wins).
- Modifying `data/raw/twl_coded_legislation_101_to_118.csv` (source of truth,
  never touched — intern labels go into new files).

## Locked decisions

| Decision | Choice | Reason |
|---|---|---|
| Annotation tool | Hardened Google Sheet | Zero build/distribute overhead; free multi-user, live progress, version history. Migration keys off `con_legis_num` (same row as label), so positional row-shift is not a failure mode. |
| Sheet build method | `gspread` automated script | Reproducible, scales; one-time service-account setup. |
| Label scheme | Yes / No / **Unsure** | Unsure is an escape hatch for ambiguous omnibus bills → routes to expert, never enters training raw. Avoids forced-guess noise on the hard cases. |
| Reliability | Double-label + adjudicate + ~10 gold traps/intern | Agreements auto-accept (~80%); only disagreements + Unsure reach Heagen (~20%). Gold traps yield per-intern accuracy. |
| Allocation | 119 test **first**, then negatives | The bounded high-leverage deliverable (clean eval) completes before time runs out. |
| 119 role | **New primary test set** | Freshest temporal holdout with real negatives. 118 retired to secondary/legacy; 117 stays validation. |

## Capacity model

- 8 interns × ~3 effective hrs (subtract ~45 min day-1 onboarding/calibration)
  × ~40 bills/hr ≈ **~900 label-actions**.
- Double-labeled → **~450 unique bills**, minus gold-trap overhead (~10/intern).

| Target | Unique bills (approx) | Source |
|---|---|---|
| 119 test set | ~300 | Random sample of Congress 119 Stage-1 survivors (~50 negatives expected). |
| Training negatives | ~140 | Congresses 101–116 survivors, biased toward likely-negatives. |

Numbers are estimates; the sprint stops at whatever is complete. The 119 test set
is filled first so it is guaranteed complete even if throughput is lower than modeled.

## Prerequisite: extend Stage 1 to Congress 119

`data/processed/china_filter_results.csv` currently stops at Congress 118.
Congress 119 raw data exists (`raw_data/raw_legislation/119/`) but has never been
filtered. Before sampling, extend `main.py`'s congress range to include 119 and
re-run Stage 1 to produce the 119 survivor pool.

## Component 1 — packet builder (`scripts/annotation/build_packets.py`)

Selects bills, assigns annotators, injects traps, and creates the Google Sheet.

**Sampling:**
- **119 test sample:** random sample (fixed seed) of ~300 from 119 survivors.
- **Negative-biased 101–116 sample:** ~140 survivors where
  `keyword_count == 1 AND has_strong_keyword == False` (a single weak keyword such
  as "tariff" or "human rights", no high-specificity keyword like `prc`/`pla`/`uyghur`).
  These are where incidental-mention negatives concentrate. Ground truth still comes
  from the intern; the bias only raises the negative yield.

**Assignment:**
- Each selected bill is assigned to **exactly 2 interns** (round-robin so load is even).
- **~10 gold-trap bills** (known labels drawn from the existing gold set — a mix of
  clear positives, clear negatives, and a few subtle negatives) are woven into every
  intern's tab, visually identical to real bills.
- Each row carries a stable `row_uid` and the true `con_legis_num` so downstream
  merging is by ID, never by position.

**Output:** one Google Sheet, one tab per intern, via `gspread`. Interns fill in
**nothing that enters the dataset except the label** — every identifying field is
pre-filled and locked. Human-readable columns (not the cryptic `con_legis_num`):

| Col | Field | Editable? | Notes |
|---|---|---|---|
| A | `congress` (e.g. 118) | locked | pre-filled |
| B | `chamber` (House / Senate) | locked | derived from type |
| C | `bill_number` (e.g. HR 1153, S 442) | locked | derived from type + number |
| D | `title` | locked | pre-filled |
| E | `link` (congress.gov) | locked | pre-filled |
| **F** | **`label`** | **dropdown** | data-validation = `Yes / No / Unsure`, reject-on-invalid |
| **G** | **`notes`** | **free text** | optional; advisory only, never parsed into the dataset |
| H | `con_legis_num` | locked/hidden | true ID; rides along so migration is keyed by ID, never position |

- Only **F (label)** and **G (notes)** are editable; A–E and H are protected read-only;
  sheet sort/filter disabled for editors.
- The `notes` field is safe by construction: it is captured for the adjudication
  queue but never parsed, so a typo there cannot corrupt any dataset value.
- A parallel local `data/annotation/assignments.csv` records the bill→intern→trap
  mapping (kept off the interns' sheet) for the merge step.

**Google auth:** a service-account JSON credential (one-time setup), path via env
var, never committed. Documented in the implementation plan.

## Component 2 — merge & adjudicate (`scripts/annotation/merge_annotations.py`)

1. Pull all tabs (via `gspread`) → long form `(con_legis_num, intern, label, notes)`.
2. Join to `assignments.csv`; verify each bill has its 2 expected labels; report
   missing/incomplete.
3. **Gold traps:** compare intern labels to known → per-intern accuracy + confusion
   summary → `data/annotation/reliability_report.csv`.
4. **Real bills:**
   - both `Yes` → `1`; both `No` → `0` (auto-accepted).
   - any disagreement, or any `Unsure` → **adjudication queue**
     (`data/annotation/adjudication_queue.csv`: id, title, link, the two labels,
     and both interns' `notes` to speed resolution).
5. Heagen resolves the queue (adds a `final_label` column); re-run ingests it.
6. Emit resolved labels to new raw files:
   - `data/raw/intern_coded_119.csv` → `split = "test"`
   - `data/raw/intern_coded_negatives_101_116.csv` → `split = "train"`
   - Columns: `con_legis_num`, `congress`, `title`, `manual_coding` (0/1), `split`,
     `source = "intern_2026"`.

## Component 3 — dataset integration (extend `scripts/modeling/extract_features.py`)

- Add Congress 119 raw-JSON extraction (same fields as existing bills:
  `official_title`, `short_title`, `summary_text`, `subjects`).
- Union the two intern-coded raw files with the existing gold set before feature
  building; dedupe by `con_legis_num` (gold set wins on any collision).
- **Split assignment:**
  - 101–116 → `train` (now includes new negatives)
  - 117 → `val` (unchanged)
  - 118 → `test_legacy` (kept, no longer primary)
  - 119 → `test` (new primary)
- Regenerate `data/processed/features.csv`. Flag `has_summary` on 119 rows so eval
  can be reported summary-present vs title-only.

## Data flow

```
extend Stage 1 → 119 survivors
        │
        ▼
build_packets.py ──► Google Sheet (8 tabs, dropdown-only)  ──► interns label
        │                                                            │
        └──► assignments.csv ◄───────── merge_annotations.py ◄───────┘
                                              │
              ┌───────────────┬──────────────┼────────────────┐
              ▼               ▼               ▼                ▼
      reliability_report  adjudication_   intern_coded_*.csv  (auto-accepted
       (gold traps)       queue (Heagen)   (resolved labels)   agreements)
                                              │
                                              ▼
                              extract_features.py (119 + unions + splits)
                                              ▼
                                    data/processed/features.csv
```

## Risks

- **119 summary coverage.** An in-progress Congress may have many bills without a
  CRS summary yet. Interns still label accurately (they can open the full bill),
  but the *model* sees title-only for those. Mitigation: `has_summary` flag on 119
  rows; report test metrics split by it.
- **Negative-sample bias.** The `keyword_count==1` heuristic enriches for negatives
  but is not purely negative — acceptable, since ground truth comes from the intern
  and imbalance is handled by class weighting, not by the sample ratio.
- **Intern calibration drift.** First session must include a short calibration on
  shared examples with the coding rule ("primarily *about* China policy, not merely
  mentioning China"). Gold traps monitor drift across the week.
- **Throughput below model.** If fewer bills get labeled, the 119 test set (filled
  first) still completes; the negatives batch simply ends up smaller.

## Deliverables / task list

1. Extend Stage 1 to Congress 119 (`main.py` range) and re-run.
2. `scripts/annotation/build_packets.py` — sampling, 2-annotator assignment, gold
   traps, `gspread` sheet creation, `assignments.csv`.
3. Google Sheet template wiring (protection + dropdown) inside the script.
4. `scripts/annotation/merge_annotations.py` — agreement, adjudication queue,
   gold-trap reliability, resolved raw CSVs.
5. Extend `scripts/modeling/extract_features.py` — 119 extraction, intern-file
   union, new split scheme, regenerate `features.csv`.
6. A one-page intern coding guide (the Yes/No/Unsure rule + 3–4 worked examples).

## Open items (resolve during planning)

- Exact gold-trap composition (which known bills, and how many clear vs subtle).
- Service-account credential setup steps (documented in the plan, run once by Heagen).
- Final target counts once the 119 survivor pool size is known post-Stage-1.
