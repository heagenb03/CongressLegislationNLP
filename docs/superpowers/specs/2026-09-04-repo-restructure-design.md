# Repo restructure: one package, ordered steps, declarative features

Date: 2026-09-04
Status: approved (Heagen), ready for implementation planning

## Problem

The code that produces `features.csv` is split across `stage1/` and three
subdirectories of `scripts/`, named under three different schemes: `stage1` is an
ordinal, `modeling` is a topic, `pipeline` is a category. Nothing in the tree says
what runs after what.

Four concrete symptoms:

1. **"Stage 2" has three definitions.** In code, `legislation_pipeline.stage2()`
   is the file copy. `stage2/CLAUDE.md` says stage2 is the transformer classifier.
   `README.md` says Stage 2 is the file copy and Stage 3 (planned) is the
   transformer. `stage2/` itself holds one empty `__init__.py`.
2. **Steps are misfiled.** `scripts/annotation/scan_congress_119.py` is a Stage 1
   keyword scan living under annotation. `scripts/pipeline/analyze_*.py` are
   diagnostics of Stage 1 and live nowhere near it.
3. **Ten `sys.path.insert` shims** (6 in scripts, 4 in tests) and hardcoded
   `data/processed/...` paths in 8 files. Every script must be run from the
   project root or it fails.
4. **Split boundaries live inside a step.** `TRAIN_MAX_CONGRESS`,
   `VAL_CONGRESS`, `TEST_LEGACY_CONGRESS`, `TEST_CONGRESS` are module constants in
   `scripts/modeling/extract_features.py`, so any evaluation code that needs to
   know what "val" means must import a feature-extraction script.

Two more findings drive the design:

- **Five separate implementations of bill-ID handling** exist, with genuinely
  different contracts (see `ids.py` below). Consolidating them wrongly changes
  which rows survive de-duplication in `features.csv`.
- **`features.csv` is 22.7 MB and tracked.** 19.8 MB of that is the same summary
  text stored three times: `summary_text` (6.38 MB), `text` (6.70 MB),
  `text_with_keywords` (6.76 MB).

## Locked decisions

| # | Decision | Consequence |
|---|----------|-------------|
| 1 | CLAUDE.md and `.wolf/` stay gitignored | Fix staleness in place; no git change. `README.md` is tracked and gets fixed too. |
| 2 | `features.csv` stores base columns only | Derived text columns are rebuilt at load time. ~22.7 MB to ~9 MB. |
| 3 | Per-congress manifests under `data/processed/manifests/`, found by glob | Adding the 120th Congress never rescans 101-118. |
| 4 | Comments: fix accuracy, cut volume, ban rerun-dependent numbers | Row counts and P/R/F1 live in `.wolf/STATUS.md` and script output only. |
| 5 | One installable package (`pip install -e .`) | All 10 `sys.path.insert` shims deleted. |
| 6 | `pyspark` dropped; `transformers` and `torch` added | Nothing imports pyspark. BART is the next step. |

## Target layout

```
congress_nlp/
  __init__.py
  paths.py          PROJECT_ROOT (from __file__) and every data path
  splits.py         congress -> split table, assign_split()
  ids.py            all bill-ID parsing, normalizing, and path derivation
  filtering/
    __init__.py
    keywords.py     CHINA_KEYWORDS, AMENDMENTS_START_CONGRESS   (was stage1/constants.py)
    pipeline.py     LegislationPipeline, ChinaLegislationPipeline (was stage1/legislation_pipeline.py)
    diagnostics.py  filter coverage + keyword effectiveness      (was scripts/pipeline/analyze_*.py)
  annotation/
    __init__.py
    utils.py        (was scripts/annotation/annotation_utils.py, minus ID functions)
    packets.py      (was build_packets.py)
    sheets.py       (was build_packets_sheet.py)
    merge.py        (was merge_annotations.py)
  features/
    __init__.py
    extract.py      raw JSON -> features.csv           (was extract_features.py)
    views.py        the text-view registry
    load.py         load_features(view=...)
  classifiers/
    __init__.py
    baseline.py     TF-IDF + LR                        (was train_baseline.py)
    zeroshot_bart.py  (new, next quest)
    finetune.py       (new, later)
  evaluation/
    __init__.py
    metrics.py      threshold sweep, per-subgroup report, both base rates

scripts/
  01_scan.py                  --congress 120  |  --congress-range 101-118
  02_build_packets.py
  03_merge_annotations.py
  04_extract_features.py
  05_train_baseline.py
  diagnose_filter.py          not numbered - diagnostics, run any time
  diagnose_keywords.py

pyproject.toml                package metadata + pytest pythonpath
tests/conftest.py             replaces 4 test-level sys.path shims
```

`stage1/` becomes `filtering/`. **`stage2/` is deleted.** The file copy is
`filtering.pipeline.copy_survivors()` — part of the filter step, not its own
stage — and the transformer goes in `classifiers/`. This removes the
three-meanings problem at its root.

`main.py` is deleted; `scripts/01_scan.py` replaces it and subsumes
`scan_congress_119.py`.

### Why numbers go on scripts, not packages

Order matters at the command line, where you experience it. It does not matter in
an import path, and a numbered package breaks the day a step is inserted between
two others.

### Why five packages and not one

The re-run loops nest, and the package boundaries mark where you decide what to
re-run:

| Changed | Run | Cost |
|---------|-----|------|
| New congress (120) | `01_scan.py --congress 120`, then `04_extract_features.py` | scans 120 only |
| New text view | `05_train_baseline.py --view <name>` | zero; views build at load |
| A keyword | `01_scan.py` full, `diagnose_*`, `04`, `05` | full ~270k rescan |
| Split boundary | edit `splits.py`, re-run `04` | ~1,743 rows |

## Module contracts

### `congress_nlp/paths.py`

Derives `PROJECT_ROOT` from `Path(__file__).resolve().parents[1]`, never from
`Path(".")`. This is what removes the run-from-project-root requirement.

Exports at minimum:

```
PROJECT_ROOT
RAW_LEGISLATION      = PROJECT_ROOT / "raw_data" / "raw_legislation"
FILTERED_OUTPUT      = PROJECT_ROOT / "raw_data" / "china_legislation"   # renamed, see below
DATA_RAW             = PROJECT_ROOT / "data" / "raw"
GOLD_LABELS          = DATA_RAW / "twl_coded_legislation_101_to_118.csv"
INTERN_DIR           = DATA_RAW / "Summer2026InternsData"
DATA_PROCESSED       = PROJECT_ROOT / "data" / "processed"
MANIFEST_DIR         = DATA_PROCESSED / "manifests"
COVERAGE_CSV         = DATA_PROCESSED / "filter_coverage_analysis.csv"
FEATURES_CSV         = DATA_PROCESSED / "features.csv"
ANNOTATION_DIR       = PROJECT_ROOT / "data" / "annotation"
MODELS_DIR           = PROJECT_ROOT / "models"
OUTPUTS_DIR          = PROJECT_ROOT / "outputs"
```

Plus `manifest_paths() -> list[Path]`, which globs `MANIFEST_DIR /
"china_filter_*.csv"` sorted by name. This replaces the hardcoded two-file list at
`extract_features.py:438-439`.

**One directory rename rides here.** `raw_data/china_legislation_101_118/` becomes
`raw_data/china_legislation/`. The congress range in the name is wrong the moment
Stage 1 runs on the 119th, and the directory is gitignored, so this is a local
`mv` with no history consequence. It is referenced only by `main.py:6`, which the
restructure deletes anyway.

### `congress_nlp/splits.py`

Owns the temporal split. No step owns it; every step imports it.

```
SPLITS: dict[int, str] = {
    **{c: "train" for c in range(93, 117)},   # 93-116
    117: "val",
    118: "test_legacy",
    119: "test",
}
```

`assign_split(congress: int) -> str` looks up the table and **raises
`UnknownCongressError`** on a miss. The message must name the congress and point
at this file, e.g.:

> `Congress 120 has no split assignment. Add it to SPLITS in
> congress_nlp/splits.py before extracting features. Do not let a new congress
> fall into an existing split by default.`

**This is a deliberate behavior change.** Today `assign_split` ends with
`return "test"  # 119+`, so scanning the 120th Congress and re-extracting would
silently add it to the test split. 119 stays `"test"` — that is where the intern
data lives — and 120 becomes an error until someone decides.

### `congress_nlp/ids.py`

Merges five implementations that have **different contracts**. The spec fixes each
contract explicitly, because getting this wrong changes which rows land in
`features.csv`.

| New function | Replaces | Contract |
|---|---|---|
| `normalize_id_key(con_legis_num) -> str` | `annotation_utils.normalize_id_key` | Collapses dot variants so `118_h.r.1153` and `118_hr.1153` yield one key. Never returns None. Used by de-duplication. |
| `to_canonical_id(con_legis_num) -> str \| None` | `analyze_filter_coverage.normalize_twl_id` | Converts a TWL `con_legis_num` to the pipeline's dotted form. **Returns None for amendments** (`s.amdt.*` / `h.amdt.*`) — they are not in the gold set. |
| `to_json_path(con_legis_num, raw_root) -> Path \| None` | `extract_features.con_legis_num_to_path` and `analyze_keyword_effectiveness._con_legis_num_to_path` | Derives the `data.json` path. **Returns None for amendments and for unparseable IDs.** |
| `make_legislation_id(congress, type, number) -> str` | `legislation_pipeline._make_legislation_id` | Builds an ID from directory metadata. Does **not** parse an existing ID and does **not** read the JSON `bill_id` field. |

Amendment behavior differs per function and must be preserved exactly as listed.

**Correctness risk to guard.** `extract.dedupe_records` keys on
`normalize_id_key`, and gold records are appended before intern records so gold
wins collisions. If the merged normalizer changes behavior, de-duplication
changes and the row count moves. See "Acceptance" below.

### `congress_nlp/features/views.py`

The registry that makes adding or removing a model input a single edit.

```
VIEWS: dict[str, Callable[[pd.DataFrame], pd.Series]] = {
    "title_subjects":   ...,   # official_title + ", ".join(subjects)   [primary]
    "title_summary":    ...,   # official_title + summary_text
    "keyword_prefixed": ...,   # "[FILTER: prc, pla] " + title_summary
}
DEFAULT_VIEW = "title_subjects"
```

Adding a view is one function plus one dict entry; removing it is deleting both.
No NamedTuple edit, no `features.csv` regeneration, no four-place change. Every
model script exposes `--view` with `choices=VIEWS.keys()`.

### `congress_nlp/features/load.py`

```
load_features(view: str = DEFAULT_VIEW, splits: Sequence[str] | None = None) -> pd.DataFrame
```

Reads `FEATURES_CSV`, applies the named view to produce a `text` column, and
optionally filters to the given splits. This is the single entry point for
`baseline.py`, `zeroshot_bart.py`, and `finetune.py`.

## `features.csv` schema change

**Stored (base columns):** `con_legis_num`, `congress`, `official_title`,
`short_title`, `summary_text`, `subjects`, `manual_coding`, `split`,
`has_summary`, `matched_keywords`, `keyword_count`, `has_strong_keyword`.

**Removed (rebuilt at load time):** `text`, `text_title_subjects`,
`text_with_keywords`.

All three are recoverable from the stored columns, so nothing is lost. Expected
size: ~22.7 MB to ~9 MB.

`STRONG_KEYWORDS` stays in `features/extract.py`. It is a curated subset chosen by
effectiveness analysis, not the full filter list, and must not be merged with
`filtering/keywords.py`.

### CLI change this forces

`train_baseline.py` today takes `--input-field {text_title_subjects,text}`,
reading a column that will no longer exist. The replacement is `--view`, with
choices from the registry. `keyword_prefixed` remains available as a view even
though `stage2/CLAUDE.md`'s advice to use `text_with_keywords` is superseded —
title+subjects is the primary input.

## Manifest migration

This is a data migration, not just a code change. Use `git mv` so history follows:

```
data/processed/china_filter_results.csv      -> data/processed/manifests/china_filter_101_118.csv
data/processed/china_filter_results_119.csv  -> data/processed/manifests/china_filter_119.csv
```

Readers that must be updated: `main.py:7` (`csv_path`),
`scripts/annotation/scan_congress_119.py:25`, `build_packets.py:285-286`,
`extract_features.py:438-439`.

`scripts/01_scan.py` writes `MANIFEST_DIR / f"china_filter_{range_label}.csv"`
where `range_label` is `119` for a single congress and `101_118` for a range.

## Documentation

### Files to fix (all currently stale)

| File | What is wrong |
|---|---|
| `README.md` (tracked) | Documents `python scripts/analyze_filter_coverage.py` — wrong path. Says 1,216 labeled bills. Its "Stage 1 / Stage 2 / Stage 3" vocabulary is replaced by the step names (filter, annotate, features, classify, evaluate), since `stage2/` no longer exists. |
| `CLAUDE.md` (root) | Cites `md_files/PLAN1.MD`; the file is at `docs/PLAN1.MD`. Carries live numbers. |
| `data/CLAUDE.md` | Says features.csv is 1,216 rows (it is 1,743). Gives split as test=118. Omits `text_title_subjects`, `data/annotation/`, and `Summer2026InternsData/`. |
| `scripts/modeling/CLAUDE.md` | Says "Evaluate on val (117), not test (118)" — root CLAUDE.md says report on the 119 test split. Names `TEST_CONGRESS = 118`. Quotes stale val P/R/F1. |
| `scripts/CLAUDE.md` | 5-step run order; omits `scan_congress_119.py`, `build_packets.py`, `merge_annotations.py`. |
| `scripts/pipeline/CLAUDE.md` | Paths change when scripts move. |
| `stage1/CLAUDE.md` | Directory ceases to exist. |
| `stage2/CLAUDE.md` | Directory ceases to exist. Also says to use `text_with_keywords`, cites test=118 and `md_files/PLAN1.MD`. |
| `models/CLAUDE.md`, `outputs/CLAUDE.md` | 0 bytes. **Delete them** — both directories already have `.gitkeep`. An empty instruction file is worse than none. |

CLAUDE.md files follow the new package tree: one per top-level package
(`congress_nlp/filtering/`, `annotation/`, `features/`, `classifiers/`,
`evaluation/`), plus root, `data/`, and `scripts/`.

### The no-live-numbers rule

No CLAUDE.md or README may contain a number that changes when a script is re-run:
row counts, split sizes, precision, recall, F1, ROC-AUC, keyword counts. Those
live in `.wolf/STATUS.md` and in each script's printed report. This rule is what
prevents the current root-vs-`scripts/modeling` contradiction from recurring, and
it goes in root `CLAUDE.md` as an explicit instruction.

Structural facts stay: column names, join keys, file responsibilities, gotchas
such as the `congress_session` URL bleed and the whole-word regex behavior.

### Comment volume

`features/extract.py` inherits ~120 lines of docstring for a 468-line file.
Docstrings that restate the signature get cut. Docstrings that record a decision
or a non-obvious constraint stay — for example, why `text_title_subjects` is
primary (CRS summary lag on a current Congress) and why the `[FILTER: ...]` prefix
sits at the front (512-token truncation).

## Testing

Existing tests move with their modules:

```
tests/annotation/test_annotation_utils.py       -> tests/test_ids.py + tests/annotation/test_utils.py
tests/annotation/test_build_packets.py          -> tests/annotation/test_packets.py
tests/annotation/test_merge_annotations.py      -> tests/annotation/test_merge.py
tests/modeling/test_extract_features_splits.py  -> tests/features/test_extract.py + tests/test_splits.py
```

`tests/conftest.py` replaces the 4 test-level `sys.path.insert` lines.

**Known test edit:** `tests/modeling/test_extract_features_splits.py:93` asserts
`INTERN_DIR == "data/raw/Summer2026InternsData"` as a string literal. That value
moves to `paths.INTERN_DIR` as a `Path`, so the assertion must change.

New tests required:

- `test_splits.py`: `assign_split` returns the right label for 101, 116, 117, 118,
  119, and **raises `UnknownCongressError` for 120**.
- `test_ids.py`: the amendment behavior of each `ids.py` function, per the
  contract table — `to_canonical_id` and `to_json_path` return None for
  amendments, `normalize_id_key` does not.
- `test_views.py`: every entry in `VIEWS` produces a non-empty string for a row
  with a title, and `load_features(view=...)` returns a `text` column.

## Acceptance

The restructure is behavior-preserving except for the two changes named above
(`assign_split` raising on 120, and the `features.csv` schema slimming).

**The acceptance check runs twice — after commit 2 and again after commit 3.**
Commit 2 moves files and should not touch behavior at all; commit 3 merges the
five ID implementations, which is where a regression would actually appear.
Regenerate `features.csv` at each point and confirm the row count and per-split
class balance match the current values:

```
total 1743
train        848  (558 pos)
val          222  (179 pos)
test_legacy  184  (178 pos)
test         489  (250 pos)
```

Any drift means the `ids.py` merge changed de-duplication. Investigate before
proceeding — do not accept a new number.

`05_train_baseline.py --view title_subjects` must reproduce the current
validation numbers to four decimals after the schema change, since the view
rebuilds the same string the stored column held.

## Commit sequence

Branch off `main` first. Per `~/.claude/rules/git-workflow.md`, no attribution
trailers on any commit.

1. **`feat: 2026 intern annotation sprint data and adjudication`** — the 8
   modified files and the 6 untracked ones, committed as-is before anything moves.
   Two classes of new file: `intern_coded_119.csv` and
   `intern_coded_negatives_101_116.csv` are real inputs that `features.csv`
   depends on; `data/annotation/{responses_raw,resolved_auto,adjudication_queue,reliability_report}.csv`
   are intermediates regenerable from the intern sheets.
2. **`refactor: move code into congress_nlp package`** — file moves and import
   rewrites only. No logic changes, so the diff reads as "moves only." Includes
   `pyproject.toml`, `tests/conftest.py`, and the `git mv` of the manifests.
3. **`refactor: extract paths, splits, ids, and the view registry`** — the four
   shared modules, the `assign_split` behavior change, the `features.csv` schema
   slimming, and the `--view` CLI. Requirements change (drop pyspark, add
   transformers and torch) rides here.
4. **`docs: rewrite CLAUDE.md tree and README for the new layout`** — last,
   because steps 2 and 3 change every path and count it would document.

Run `openwolf scan` after step 4 to regenerate `.wolf/anatomy.md`, which the
SessionStart hook already flags as stale and which steps 2-3 invalidate further.

## Out of scope

- No `congress` CLI wrapper. The run order is five commands and documented.
- No ruff/black/mypy configuration consolidation in `pyproject.toml` beyond the
  package metadata and pytest `pythonpath` the restructure needs.
- No `requirements.txt` changes beyond dropping `pyspark` and adding
  `transformers` and `torch`.
- No model work. Zero-shot BART and fine-tuning are the next quest, not this one.
- No change to `stage1` keyword content or filter behavior.
