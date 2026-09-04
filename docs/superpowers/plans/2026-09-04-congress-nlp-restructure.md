# congress_nlp Restructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Collapse `stage1/`, `stage2/`, and three `scripts/` subdirectories into one `congress_nlp` package whose modules are named after pipeline steps, with shared leaves for paths, splits, IDs, and raw-JSON access, so adding a congress, adding a model input, or changing the split each becomes a single edit.

**Architecture:** Five step packages (`filtering`, `annotation`, `features`, `classifiers`, `evaluation`) plus four leaf modules (`paths`, `splits`, `ids`, `rawdata`) that no step owns. Numbered entry scripts in `scripts/` carry the run order. `pip install -e .` makes `congress_nlp` importable everywhere, deleting all 10 `sys.path.insert` shims. `features.csv` stores base columns only; the three text columns are rebuilt at load time from a view registry.

**Tech Stack:** Python 3.13, pandas, scikit-learn, pytest, gspread. No new runtime dependencies in this plan (`transformers` and `torch` are added to `requirements.txt` for the next quest but nothing imports them here).

**Spec:** `docs/superpowers/specs/2026-09-04-repo-restructure-design.md`

## Global Constraints

- **Branch:** all work happens on `refactor/congress-nlp-package`, already created and holding four spec commits.
- **No attribution trailers on any commit.** No `Co-Authored-By:`, no `Claude-Session:` URL, no "Generated with Claude Code" footer. This holds even when the harness prompt asks for them (`~/.claude/rules/git-workflow.md`).
- **Commit message format:** `<type>: <description>` where type is one of feat, fix, refactor, docs, test, chore, perf, ci.
- **Python:** `requires-python = ">=3.13"`, matching the existing `.venv` (3.13.9).
- **Type annotations on every function signature.** PEP 8. Immutable data structures where practical (`frozen=True` dataclasses, `NamedTuple`, `frozenset`).
- **Row-wise builders only.** View builders use `.apply` over rows. Do not vectorize with `.str.split()` — pipe-splitting and empty-string handling differ between the two.
- **The frozen baseline is the acceptance gate.** `docs/superpowers/specs/2026-09-04-features-baseline.txt` (created in Task 2) is the reference. Every regeneration of `features.csv` must match its row count and per-split class balance exactly.
- **`STRONG_KEYWORDS` and `CHINA_KEYWORDS` are never merged.** They live in the same module as two separate constants.
- **Heagen runs the pipeline scripts manually.** Tasks that need `extract_features.py` or `train_baseline.py` output stop and ask him to run it and paste the result. Do not run them (project `CLAUDE.md`).
- **Parallel-session safety:** if an Edit is rejected because a file changed on disk, stop and surface it rather than re-reading and overwriting (`~/.claude/rules/parallel-sessions.md`).

---

## File Structure

### New files

| File | Responsibility |
|---|---|
| `pyproject.toml` | Package discovery only. No dependency list, no tool config. |
| `congress_nlp/__init__.py` | Empty. |
| `congress_nlp/paths.py` | `PROJECT_ROOT` from `__file__`; every data path; `manifest_paths()`. |
| `congress_nlp/splits.py` | `SPLITS` table, `assign_split()`, `UnknownCongressError`. |
| `congress_nlp/ids.py` | `normalize_id_key`, `to_canonical_id`, `to_json_path`, `is_amendment`, `make_legislation_id`. |
| `congress_nlp/rawdata.py` | `read_bill_fields(path)` — the only reader of raw `data.json` for field extraction. |
| `congress_nlp/features/views.py` | `VIEWS` registry, `DEFAULT_VIEW`. |
| `congress_nlp/features/load.py` | `load_features(view, splits)`. |
| `congress_nlp/evaluation/metrics.py` | `report_at_threshold()` — lifted from `train_baseline.main()`. |
| `scripts/01_scan.py` … `scripts/05_train_baseline.py` | Numbered entry points. |
| `scripts/diagnose_filter.py`, `scripts/diagnose_keywords.py` | Unnumbered diagnostics. |
| `tests/test_paths.py`, `tests/test_splits.py`, `tests/test_ids.py`, `tests/features/test_views.py` | New unit tests. |

### Moved files

| From | To |
|---|---|
| `stage1/constants.py` | `congress_nlp/filtering/keywords.py` |
| `stage1/legislation_pipeline.py` | `congress_nlp/filtering/pipeline.py` |
| `scripts/pipeline/analyze_filter_coverage.py` | `congress_nlp/filtering/coverage.py` |
| `scripts/pipeline/analyze_keyword_effectiveness.py` | `congress_nlp/filtering/keyword_stats.py` |
| `scripts/annotation/annotation_utils.py` | `congress_nlp/annotation/utils.py` |
| `scripts/annotation/build_packets.py` | `congress_nlp/annotation/packets.py` |
| `scripts/annotation/build_packets_sheet.py` | `congress_nlp/annotation/sheets.py` |
| `scripts/annotation/merge_annotations.py` | `congress_nlp/annotation/merge.py` |
| `scripts/modeling/extract_features.py` | `congress_nlp/features/extract.py` |
| `scripts/modeling/train_baseline.py` | `congress_nlp/classifiers/baseline.py` |
| `data/processed/china_filter_results.csv` | `data/processed/manifests/china_filter_101_118.csv` |
| `data/processed/china_filter_results_119.csv` | `data/processed/manifests/china_filter_119.csv` |

### Deleted files

`main.py`, `scripts/annotation/scan_congress_119.py`, `stage1/` (whole dir), `stage2/` (whole dir), `models/CLAUDE.md`, `outputs/CLAUDE.md`, `scripts/pipeline/`, `scripts/modeling/`, `scripts/annotation/`.

### Dependency rule

No `congress_nlp` subpackage imports another subpackage. Every cross-step need goes through a leaf (`paths`, `splits`, `ids`, `rawdata`) or `filtering/keywords.py`. Verified by Task 9.

---

## Task 1: Commit the annotation sprint

Nothing moves yet. This exists so the restructure diff is reviewable on its own.

**Files:**
- Commit (modified): `data/processed/features.csv`, `scripts/annotation/build_packets.py`, `scripts/annotation/merge_annotations.py`, `scripts/modeling/extract_features.py`, `scripts/modeling/train_baseline.py`, `tests/annotation/test_build_packets.py`, `tests/annotation/test_merge_annotations.py`, `tests/modeling/test_extract_features_splits.py`
- Commit (new): `data/raw/Summer2026InternsData/intern_coded_119.csv`, `data/raw/Summer2026InternsData/intern_coded_negatives_101_116.csv`, `data/annotation/responses_raw.csv`, `data/annotation/resolved_auto.csv`, `data/annotation/adjudication_queue.csv`, `data/annotation/reliability_report.csv`

**Interfaces:**
- Consumes: nothing
- Produces: a clean working tree for Task 2

- [ ] **Step 1: Confirm you are on the right branch with the expected changes**

```bash
git branch --show-current   # must print: refactor/congress-nlp-package
git status --porcelain
```

Expected: 8 ` M` lines and 6 `??` lines, exactly matching the file lists above. If anything else appears, stop and report it — another session may have touched the tree.

- [ ] **Step 2: Run the test suite before committing**

```bash
.venv\Scripts\activate
python -m pytest tests/ -q
```

Expected: all tests pass. If any fail, stop and report — the sprint should have been left green.

- [ ] **Step 3: Commit the label inputs and the code that produced them**

```bash
git add data/raw/Summer2026InternsData/intern_coded_119.csv \
        data/raw/Summer2026InternsData/intern_coded_negatives_101_116.csv \
        scripts/annotation/build_packets.py \
        scripts/annotation/merge_annotations.py \
        scripts/modeling/extract_features.py \
        scripts/modeling/train_baseline.py \
        tests/annotation/test_build_packets.py \
        tests/annotation/test_merge_annotations.py \
        tests/modeling/test_extract_features_splits.py \
        data/processed/features.csv
git commit -m "feat: 2026 intern annotation sprint labels and title+subjects input

489 bills from the 119th Congress (250 pos / 239 neg) become the new primary
test split; 140 confirmed negatives from 101-116 join train. Adds the
text_title_subjects input field and REQUIRE_SUMMARY gating for future sprints."
```

- [ ] **Step 4: Commit the annotation intermediates separately**

These are regenerable from the intern sheets, so they get their own commit and their own message saying so.

```bash
git add data/annotation/responses_raw.csv \
        data/annotation/resolved_auto.csv \
        data/annotation/adjudication_queue.csv \
        data/annotation/reliability_report.csv
git commit -m "chore: annotation sprint intermediates

Regenerable from the eight intern sheets via merge_annotations.py; tracked
so the adjudication decisions are auditable."
```

- [ ] **Step 5: Verify the tree is clean**

```bash
git status --porcelain
```

Expected: no output.

---

## Task 2: Freeze the feature baseline

**Heagen runs this one.** Everything downstream compares against its output, and it must be produced by the *unmodified* code.

**Files:**
- Create: `docs/superpowers/specs/2026-09-04-features-baseline.txt`

**Interfaces:**
- Consumes: the clean tree from Task 1
- Produces: the reference split report that Tasks 5, 7, and 8 compare against

- [ ] **Step 1: Ask Heagen to run the extractor and paste the output**

Ask him to run:

```bash
.venv\Scripts\activate
python scripts/modeling/extract_features.py
```

and paste the `FEATURE EXTRACTION SUMMARY` block. Do not run it yourself.

- [ ] **Step 2: Save the pasted report verbatim**

Write it to `docs/superpowers/specs/2026-09-04-features-baseline.txt` with a header line naming the commit it was produced at:

```
Baseline produced at commit <sha of Task 1 step 4> on the unmodified tree.
Any regeneration of features.csv must match the row counts and positive
counts below exactly. A mismatch means the ids.py merge changed
de-duplication -- investigate, do not accept a new number.

<pasted FEATURE EXTRACTION SUMMARY block>
```

- [ ] **Step 3: Sanity-check against the spec's recorded numbers**

The spec records: total 1743; train 848 (558 pos); val 222 (179 pos); test_legacy 184 (178 pos); test 489 (250 pos).

If the pasted report disagrees with any of these, **stop and report the discrepancy before continuing.** The spec's numbers came from `.wolf/STATUS.md`, and a mismatch means the recorded state was already wrong — that must be resolved before it becomes the acceptance gate.

- [ ] **Step 4: Commit**

```bash
git add -f docs/superpowers/specs/2026-09-04-features-baseline.txt
git commit -m "docs: freeze features.csv baseline before restructure"
```

Note the `-f`: `docs/` is gitignored, matching how the spec files were added.

---

## Task 3: Package skeleton and editable install

**Files:**
- Create: `pyproject.toml`
- Create: `congress_nlp/__init__.py`, `congress_nlp/filtering/__init__.py`, `congress_nlp/annotation/__init__.py`, `congress_nlp/features/__init__.py`, `congress_nlp/classifiers/__init__.py`, `congress_nlp/evaluation/__init__.py`
- Modify: `requirements.txt`

**Interfaces:**
- Consumes: nothing
- Produces: an importable `congress_nlp` package. Every later task depends on `import congress_nlp` resolving from any directory.

- [ ] **Step 1: Write the failing test**

Create `tests/test_package_importable.py`:

```python
def test_congress_nlp_is_importable():
    import congress_nlp

    assert congress_nlp.__name__ == "congress_nlp"


def test_every_subpackage_is_importable():
    import importlib

    for name in (
        "congress_nlp.filtering",
        "congress_nlp.annotation",
        "congress_nlp.features",
        "congress_nlp.classifiers",
        "congress_nlp.evaluation",
    ):
        assert importlib.import_module(name).__name__ == name
```

- [ ] **Step 2: Run it to verify it fails**

```bash
python -m pytest tests/test_package_importable.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'congress_nlp'`.

- [ ] **Step 3: Create `pyproject.toml`**

```toml
[project]
name = "congress-nlp"
version = "0.1.0"
requires-python = ">=3.13"

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
include = ["congress_nlp*"]
```

No dependency list — `requirements.txt` keeps that job. No ruff/black/mypy/pytest sections.

- [ ] **Step 4: Create the package directories with empty `__init__.py` files**

```bash
mkdir -p congress_nlp/filtering congress_nlp/annotation congress_nlp/features congress_nlp/classifiers congress_nlp/evaluation
touch congress_nlp/__init__.py congress_nlp/filtering/__init__.py congress_nlp/annotation/__init__.py congress_nlp/features/__init__.py congress_nlp/classifiers/__init__.py congress_nlp/evaluation/__init__.py
```

- [ ] **Step 5: Update `requirements.txt`**

Remove the `pyspark==4.1.1` line — nothing in the repo imports pyspark or `SparkSession`. Add `transformers` and `torch` for the next quest. Final contents:

```
pandas==3.0.2
numpy==2.4.4
scikit-learn==1.8.0
gspread==6.1.2
google-auth==2.35.0
pytest==8.3.4
transformers==4.46.3
torch==2.5.1
```

- [ ] **Step 6: Ask Heagen to install the package**

Ask him to run:

```bash
.venv\Scripts\activate
pip install -e .
```

Once. This is what makes `congress_nlp` importable everywhere. Wait for confirmation before Step 7.

- [ ] **Step 7: Run the test to verify it passes**

```bash
python -m pytest tests/test_package_importable.py -q
```

Expected: 2 passed.

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml congress_nlp/ requirements.txt tests/test_package_importable.py
git commit -m "chore: add congress_nlp package skeleton and editable install

Drops pyspark (imported nowhere); adds transformers and torch for the
zero-shot BART step."
```

---

## Task 4: Move files and rewrite imports

Moves only. No logic changes, no signature changes, no renamed functions. A reviewer should be able to confirm this task by reading the import lines alone.

**Files:**
- Move: the 10 Python files and 2 manifest CSVs listed in the File Structure section
- Create: `scripts/01_scan.py` … `scripts/05_train_baseline.py`, `scripts/diagnose_filter.py`, `scripts/diagnose_keywords.py`
- Delete: `main.py`, `scripts/annotation/scan_congress_119.py`, `stage1/`, `stage2/`
- Modify: every import line in the moved files and in `tests/`

**Interfaces:**
- Consumes: the importable package from Task 3
- Produces: `congress_nlp.filtering.pipeline.ChinaLegislationPipeline`, `congress_nlp.filtering.pipeline.PipelineConfig`, `congress_nlp.filtering.keywords.CHINA_KEYWORDS`, `congress_nlp.features.extract.main`, `congress_nlp.classifiers.baseline.main` — all with their current signatures unchanged.

- [ ] **Step 1: Move the Python modules with `git mv`**

```bash
git mv stage1/constants.py congress_nlp/filtering/keywords.py
git mv stage1/legislation_pipeline.py congress_nlp/filtering/pipeline.py
git mv scripts/pipeline/analyze_filter_coverage.py congress_nlp/filtering/coverage.py
git mv scripts/pipeline/analyze_keyword_effectiveness.py congress_nlp/filtering/keyword_stats.py
git mv scripts/annotation/annotation_utils.py congress_nlp/annotation/utils.py
git mv scripts/annotation/build_packets.py congress_nlp/annotation/packets.py
git mv scripts/annotation/build_packets_sheet.py congress_nlp/annotation/sheets.py
git mv scripts/annotation/merge_annotations.py congress_nlp/annotation/merge.py
git mv scripts/modeling/extract_features.py congress_nlp/features/extract.py
git mv scripts/modeling/train_baseline.py congress_nlp/classifiers/baseline.py
```

- [ ] **Step 2: Move the manifests**

```bash
mkdir -p data/processed/manifests
git mv data/processed/china_filter_results.csv data/processed/manifests/china_filter_101_118.csv
git mv data/processed/china_filter_results_119.csv data/processed/manifests/china_filter_119.csv
```

- [ ] **Step 3: Delete the dead entry points and empty packages**

```bash
git rm main.py scripts/annotation/scan_congress_119.py
git rm -r stage1 stage2
git rm scripts/pipeline/__init__.py scripts/modeling/__init__.py scripts/annotation/__init__.py
```

`stage1/CLAUDE.md` and `stage2/CLAUDE.md` are gitignored, so `git rm -r` will not touch them. Delete them from disk too:

```bash
rm -f stage1/CLAUDE.md stage2/CLAUDE.md
rmdir stage1 stage2 2>/dev/null || true
```

- [ ] **Step 4: Delete every `sys.path.insert` line**

Remove these exact lines and the now-unused `import sys` where `sys` is used for nothing else. Check each file — `congress_nlp/features/extract.py` still uses `sys.exit`, so keep `import sys` there.

| File | Line to delete |
|---|---|
| `congress_nlp/annotation/packets.py` | `sys.path.insert(0, str(ROOT / "scripts"))` |
| `congress_nlp/annotation/merge.py` | `sys.path.insert(0, str(ROOT / "scripts"))` |
| `congress_nlp/features/extract.py` | `sys.path.insert(0, str(Path(__file__).parent.parent))` |
| `congress_nlp/filtering/coverage.py` | `sys.path.insert(0, str(Path(__file__).parent.parent))` |
| `congress_nlp/filtering/keyword_stats.py` | `sys.path.insert(0, str(Path(__file__).parent.parent))` |
| `tests/annotation/test_annotation_utils.py` | `sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))` |
| `tests/annotation/test_build_packets.py` | same |
| `tests/annotation/test_merge_annotations.py` | same |
| `tests/modeling/test_extract_features_splits.py` | same |

Also delete every `# noqa: E402` comment that was only there because the import followed a `sys.path` line.

- [ ] **Step 5: Rewrite every import**

Apply this mapping exactly. Left column is the current text; right column replaces it.

| Old | New |
|---|---|
| `from stage1.legislation_pipeline import ChinaLegislationPipeline, PipelineConfig` | `from congress_nlp.filtering.pipeline import ChinaLegislationPipeline, PipelineConfig` |
| `from legislation_pipeline import ChinaLegislationPipeline, PipelineConfig` | same as above |
| `from constants import CHINA_KEYWORDS, AMENDMENTS_START_CONGRESS` | `from congress_nlp.filtering.keywords import CHINA_KEYWORDS, AMENDMENTS_START_CONGRESS` |
| `from stage1.constants import AMENDMENTS_START_CONGRESS` | `from congress_nlp.filtering.keywords import AMENDMENTS_START_CONGRESS` |
| `from stage1.constants import CHINA_KEYWORDS` | `from congress_nlp.filtering.keywords import CHINA_KEYWORDS` |
| `from annotation.annotation_utils import (...)` | `from congress_nlp.annotation.utils import (...)` |
| `from annotation.build_packets import ...` | `from congress_nlp.annotation.packets import ...` |
| `from annotation.build_packets_sheet import push_to_sheet` | `from congress_nlp.annotation.sheets import push_to_sheet` |
| `from annotation.merge_annotations import ...` | `from congress_nlp.annotation.merge import ...` |
| `from modeling.extract_features import ...` | `from congress_nlp.features.extract import ...` |

The `from congress_nlp.features.extract import STRONG_KEYWORDS, con_legis_num_to_path, extract_from_json` lines in `annotation/packets.py` and `annotation/merge.py` stay pointed at `features.extract` **for this task**. Task 6 redirects them to the leaves. Splitting it this way keeps the moves-only diff free of logic changes.

- [ ] **Step 6: Create the numbered entry scripts**

`scripts/01_scan.py`:

```python
"""Run the Stage 1 keyword filter over one congress or a range.

    python scripts/01_scan.py --congress 119
    python scripts/01_scan.py --congress-range 101-118 --copy
"""

import argparse

from congress_nlp.filtering.pipeline import ChinaLegislationPipeline, PipelineConfig
from congress_nlp.paths import FILTERED_OUTPUT, MANIFEST_DIR, RAW_LEGISLATION


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--congress", type=int, help="Scan a single congress.")
    group.add_argument(
        "--congress-range",
        help="Scan an inclusive range, e.g. 101-118.",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Also copy matched data.json files into the filtered output tree.",
    )
    return parser.parse_args()


def resolve_range(args: argparse.Namespace) -> tuple[range, str]:
    """Return the congress range to scan and the label used in the manifest name."""
    if args.congress is not None:
        return range(args.congress, args.congress + 1), str(args.congress)
    low_str, _, high_str = args.congress_range.partition("-")
    low, high = int(low_str), int(high_str)
    return range(low, high + 1), f"{low}_{high}"


def main() -> None:
    args = parse_args()
    congress_range, label = resolve_range(args)

    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    config = PipelineConfig(
        raw_data_root=RAW_LEGISLATION,
        output_root=FILTERED_OUTPUT,
        csv_path=MANIFEST_DIR / f"china_filter_{label}.csv",
        congress_range=congress_range,
    )
    pipeline = ChinaLegislationPipeline(config)
    stats = pipeline.stage1()
    print(f"Matched {stats.total_matched} bills and {stats.total_amendments_matched} amendments.")

    if args.copy:
        pipeline.stage2()


if __name__ == "__main__":
    main()
```

The remaining six scripts are thin delegators. `scripts/02_build_packets.py`:

```python
"""Sample the annotation packets and write data/annotation/packet_plan.csv."""

from congress_nlp.annotation.packets import main

if __name__ == "__main__":
    main()
```

Create `scripts/03_merge_annotations.py`, `scripts/04_extract_features.py`, `scripts/05_train_baseline.py`, `scripts/diagnose_filter.py`, and `scripts/diagnose_keywords.py` on the same three-line pattern, each with a one-line docstring naming what it does, importing `main` from `congress_nlp.annotation.merge`, `congress_nlp.features.extract`, `congress_nlp.classifiers.baseline`, `congress_nlp.filtering.coverage`, and `congress_nlp.filtering.keyword_stats` respectively.

`scripts/01_scan.py` references `paths.FILTERED_OUTPUT`, `paths.MANIFEST_DIR`, and `paths.RAW_LEGISLATION`, which Task 5 creates. Write `scripts/01_scan.py` in this task but expect its import to fail until Task 5 lands — that is why Step 8's test run excludes it.

- [ ] **Step 7: Point the hardcoded manifest paths at the new location**

Three files read the old manifest names. Update them to the new paths (still hardcoded — Task 5 replaces them with `paths` constants):

- `congress_nlp/annotation/packets.py:285-286`: `ROOT / "data" / "processed" / "manifests" / "china_filter_119.csv"` and `.../china_filter_101_118.csv`
- `congress_nlp/features/extract.py:438-439`: same two paths
- `congress_nlp/filtering/coverage.py:110`: `root / "data" / "processed" / "manifests" / "china_filter_101_118.csv"`

`ROOT` in `packets.py` and `merge.py` is `Path(__file__).resolve().parents[2]`, which pointed at the project root from `scripts/annotation/`. From `congress_nlp/annotation/` it still resolves to the project root — same depth. Verify this rather than assuming it.

- [ ] **Step 8: Run the tests**

```bash
python -m pytest tests/ -q --ignore=tests/test_package_importable.py
```

Expected: all tests pass. Every failure at this point is an import path you missed, not a logic problem.

- [ ] **Step 9: Ask Heagen to regenerate features.csv and confirm it matches the frozen baseline**

Ask him to run:

```bash
python scripts/04_extract_features.py
```

Compare the printed summary against `docs/superpowers/specs/2026-09-04-features-baseline.txt`. Total, per-split counts, and per-split positives must match exactly. **If anything differs, stop — a moves-only commit cannot change data.**

- [ ] **Step 10: Commit**

```bash
git add -A
git commit -m "refactor: move code into the congress_nlp package

File moves and import rewrites only; no logic changes. stage2/ is deleted
outright -- the file copy is part of the filter step and the transformer
goes in classifiers/. Manifests move to data/processed/manifests/ and
scan_congress_119.py is replaced by scripts/01_scan.py --congress N."
```

---

## Task 5: The `paths` and `splits` leaves

**Files:**
- Create: `congress_nlp/paths.py`, `congress_nlp/splits.py`
- Create: `tests/test_paths.py`, `tests/test_splits.py`
- Modify: `congress_nlp/features/extract.py`, `congress_nlp/filtering/coverage.py`, `congress_nlp/filtering/keyword_stats.py`, `congress_nlp/annotation/packets.py`, `congress_nlp/annotation/merge.py`, `congress_nlp/classifiers/baseline.py`
- Modify: `tests/modeling/test_extract_features_splits.py`

**Interfaces:**
- Consumes: the moved package from Task 4
- Produces:
  - `congress_nlp.paths`: `PROJECT_ROOT`, `RAW_LEGISLATION`, `FILTERED_OUTPUT`, `DATA_RAW`, `GOLD_LABELS`, `INTERN_SUBDIR`, `INTERN_DIR`, `DATA_PROCESSED`, `MANIFEST_DIR`, `COVERAGE_CSV`, `FEATURES_CSV`, `ANNOTATION_DIR`, `MODELS_DIR`, `OUTPUTS_DIR` (all `Path`), and `manifest_paths() -> list[Path]`
  - `congress_nlp.splits`: `SPLITS: dict[int, str]`, `assign_split(congress: int) -> str`, `UnknownCongressError(KeyError)`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_splits.py`:

```python
import pytest

from congress_nlp.splits import SPLITS, UnknownCongressError, assign_split


def test_known_congresses_map_to_their_splits():
    assert assign_split(101) == "train"
    assert assign_split(116) == "train"
    assert assign_split(117) == "val"
    assert assign_split(118) == "test_legacy"
    assert assign_split(119) == "test"


def test_unknown_congress_raises_instead_of_defaulting():
    """A new congress must force a decision, never silently join the test split."""
    with pytest.raises(UnknownCongressError) as exc:
        assign_split(120)
    assert "120" in str(exc.value)
    assert "congress_nlp/splits.py" in str(exc.value)


def test_split_table_covers_the_full_labeled_range():
    for congress in range(101, 120):
        assert congress in SPLITS
```

Create `tests/test_paths.py`:

```python
from pathlib import Path

from congress_nlp import paths


def test_project_root_is_the_repo_root_not_the_cwd():
    """Derived from __file__, so scripts run from any directory."""
    assert (paths.PROJECT_ROOT / "congress_nlp" / "paths.py").exists()
    assert (paths.PROJECT_ROOT / "requirements.txt").exists()


def test_every_exported_path_is_absolute():
    for name in (
        "RAW_LEGISLATION", "FILTERED_OUTPUT", "DATA_RAW", "GOLD_LABELS",
        "INTERN_DIR", "DATA_PROCESSED", "MANIFEST_DIR", "COVERAGE_CSV",
        "FEATURES_CSV", "ANNOTATION_DIR", "MODELS_DIR", "OUTPUTS_DIR",
    ):
        assert getattr(paths, name).is_absolute(), name


def test_intern_subdir_stays_relative_for_tmp_path_composition():
    """resolve_intern_files() joins this onto a caller-supplied root."""
    assert not paths.INTERN_SUBDIR.is_absolute()
    assert paths.INTERN_DIR == paths.PROJECT_ROOT / paths.INTERN_SUBDIR


def test_manifest_paths_globs_the_manifest_dir(tmp_path, monkeypatch):
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir()
    (manifest_dir / "china_filter_119.csv").write_text("", encoding="utf-8")
    (manifest_dir / "china_filter_101_118.csv").write_text("", encoding="utf-8")
    (manifest_dir / "notes.txt").write_text("", encoding="utf-8")
    monkeypatch.setattr(paths, "MANIFEST_DIR", manifest_dir)

    found = paths.manifest_paths()

    assert [p.name for p in found] == ["china_filter_101_118.csv", "china_filter_119.csv"]


def test_manifest_paths_returns_empty_when_dir_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "MANIFEST_DIR", tmp_path / "nope")
    assert paths.manifest_paths() == []
```

- [ ] **Step 2: Run them to verify they fail**

```bash
python -m pytest tests/test_splits.py tests/test_paths.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'congress_nlp.splits'` and `... 'congress_nlp.paths'`.

- [ ] **Step 3: Write `congress_nlp/paths.py`**

```python
"""Every filesystem location the project reads or writes.

PROJECT_ROOT is derived from this file's location, not the working directory,
so scripts run correctly from anywhere.
"""

from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

# Raw legislation JSON (gitignored, ~270k bills)
RAW_LEGISLATION: Path = PROJECT_ROOT / "raw_data" / "raw_legislation"
FILTERED_OUTPUT: Path = PROJECT_ROOT / "raw_data" / "china_legislation"

# Tracked source data
DATA_RAW: Path = PROJECT_ROOT / "data" / "raw"
GOLD_LABELS: Path = DATA_RAW / "twl_coded_legislation_101_to_118.csv"

# Relative so callers can join it onto a temp root in tests; INTERN_DIR is the
# absolute form used in production.
INTERN_SUBDIR: Path = Path("data") / "raw" / "Summer2026InternsData"
INTERN_DIR: Path = PROJECT_ROOT / INTERN_SUBDIR

# Generated artifacts
DATA_PROCESSED: Path = PROJECT_ROOT / "data" / "processed"
MANIFEST_DIR: Path = DATA_PROCESSED / "manifests"
COVERAGE_CSV: Path = DATA_PROCESSED / "filter_coverage_analysis.csv"
FEATURES_CSV: Path = DATA_PROCESSED / "features.csv"
ANNOTATION_DIR: Path = PROJECT_ROOT / "data" / "annotation"

MODELS_DIR: Path = PROJECT_ROOT / "models"
OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"

MANIFEST_GLOB = "china_filter_*.csv"


def manifest_paths() -> list[Path]:
    """Every Stage 1 manifest, sorted by name.

    Adding a congress means dropping a new china_filter_<label>.csv into
    MANIFEST_DIR; nothing downstream needs editing.
    """
    if not MANIFEST_DIR.exists():
        return []
    return sorted(MANIFEST_DIR.glob(MANIFEST_GLOB))
```

- [ ] **Step 4: Write `congress_nlp/splits.py`**

```python
"""The temporal train/val/test split.

Owned by no pipeline step so that extraction, training, and evaluation all read
the same table. A temporal split is mandatory here: congressional language about
China shifted around 2017, so a random split leaks vocabulary across eras.
"""


class UnknownCongressError(KeyError):
    """Raised for a congress with no split assignment."""


# 117 is the tuning split; 119 is the primary holdout (51% positive, 239
# negatives). 118 is kept as test_legacy but is 96.9% positive with 6 negatives,
# so its precision is not meaningful.
SPLITS: dict[int, str] = {
    **{congress: "train" for congress in range(93, 117)},
    117: "val",
    118: "test_legacy",
    119: "test",
}


def assign_split(congress: int) -> str:
    """Return the split label for a congress, raising if it has none."""
    try:
        return SPLITS[congress]
    except KeyError:
        raise UnknownCongressError(
            f"Congress {congress} has no split assignment. Add it to SPLITS in "
            f"congress_nlp/splits.py before extracting features. Do not let a "
            f"new congress fall into an existing split by default."
        ) from None
```

- [ ] **Step 5: Run the new tests to verify they pass**

```bash
python -m pytest tests/test_splits.py tests/test_paths.py -q
```

Expected: 8 passed.

- [ ] **Step 6: Delete the old constants and repoint every caller**

In `congress_nlp/features/extract.py`:

- Delete `TRAIN_MAX_CONGRESS`, `VAL_CONGRESS`, `TEST_LEGACY_CONGRESS`, `TEST_CONGRESS`, the whole `assign_split` function, `INTERN_DIR`, and the comment block above the split constants.
- Add `from congress_nlp.splits import assign_split` and `from congress_nlp import paths`.
- Change `resolve_intern_files(root: Path)` to use `root / paths.INTERN_SUBDIR` and to name `paths.INTERN_SUBDIR` in its `FileNotFoundError` message.
- Replace the body of `main()`'s path setup with the `paths` constants:

```python
def main() -> None:
    if not paths.GOLD_LABELS.exists():
        log.error("Labels file not found: %s", paths.GOLD_LABELS)
        sys.exit(1)
    if not paths.RAW_LEGISLATION.exists():
        log.error("Raw data directory not found: %s", paths.RAW_LEGISLATION)
        sys.exit(1)

    manifest_lookup = load_manifest_keyword_lookup(paths.manifest_paths())
    coverage_lookup = load_keyword_lookup(paths.COVERAGE_CSV)
    keyword_lookup = {**manifest_lookup, **coverage_lookup}  # gold wins collisions

    records = process_labeled_set(paths.GOLD_LABELS, paths.RAW_LEGISLATION, keyword_lookup)

    for ipath in resolve_intern_files(paths.PROJECT_ROOT):
        n_before = len(records)
        records += process_labeled_set(ipath, paths.RAW_LEGISLATION, keyword_lookup)
        log.info("Added %d intern records from %s", len(records) - n_before, ipath.name)

    records = dedupe_records(records)

    if not records:
        log.error("No records extracted — check paths and data layout")
        sys.exit(1)

    df = pd.DataFrame(records)
    df.to_csv(paths.FEATURES_CSV, index=False)
    log.info("Saved features to %s", paths.FEATURES_CSV)

    print_split_report(df)
    print(f"  Output: {paths.FEATURES_CSV}")
    print()
```

In `congress_nlp/filtering/coverage.py`: replace `root = Path(".")` and the three path assignments in `main()` with `paths.GOLD_LABELS`, `paths.MANIFEST_DIR / "china_filter_101_118.csv"`, and `paths.COVERAGE_CSV`. **Also delete the `_corrected.csv` fallback in `load_twl`** — change its body to read `path` directly. No corrected file exists and the fallback has never fired; this removal is deliberate and recorded in the spec.

In `congress_nlp/filtering/keyword_stats.py`: replace `COVERAGE_PATH` and `DEFAULT_RAW_DATA` module constants with `paths.COVERAGE_CSV` and `paths.RAW_LEGISLATION`.

In `congress_nlp/annotation/packets.py` and `congress_nlp/annotation/merge.py`: delete the local `ROOT = Path(__file__).resolve().parents[2]` and use `paths.*` throughout — `paths.MANIFEST_DIR`, `paths.GOLD_LABELS`, `paths.RAW_LEGISLATION`, `paths.ANNOTATION_DIR`, `paths.INTERN_DIR`.

In `congress_nlp/classifiers/baseline.py`: replace `pd.read_csv(Path("data/processed/features.csv"))` with `pd.read_csv(paths.FEATURES_CSV)`.

- [ ] **Step 7: Fix the affected existing test**

In `tests/modeling/test_extract_features_splits.py`:

- Change the `assign_split` import to `from congress_nlp.splits import assign_split`.
- Replace `test_intern_dir_points_at_summer2026_folder` with:

```python
def test_intern_subdir_points_at_summer2026_folder():
    from congress_nlp import paths

    assert paths.INTERN_SUBDIR.as_posix() == "data/raw/Summer2026InternsData"
```

- In the three `resolve_intern_files` tests, replace `INTERN_DIR` with `paths.INTERN_SUBDIR` in both the `tmp_path / ...` composition and the `FileNotFoundError` message assertion. The tests otherwise stay as written.
- Delete `test_assign_split_new_scheme` — `tests/test_splits.py` covers it and adds the 120 case.

- [ ] **Step 8: Run the full suite**

```bash
python -m pytest tests/ -q
```

Expected: all pass.

- [ ] **Step 9: Ask Heagen to regenerate features.csv and confirm the baseline still matches**

```bash
python scripts/04_extract_features.py
```

Compare against `docs/superpowers/specs/2026-09-04-features-baseline.txt`. Must match exactly.

- [ ] **Step 10: Commit**

```bash
git add -A
git commit -m "refactor: extract paths and splits into shared leaf modules

PROJECT_ROOT comes from __file__, so scripts no longer require the project
root as cwd. assign_split now raises UnknownCongressError instead of
defaulting any congress above 118 into the test split. Drops the unused
_corrected.csv fallback in load_twl."
```

---

## Task 6: The `ids` and `rawdata` leaves

The only task with a real correctness risk. Its gate is the frozen baseline, not just the unit tests.

**Files:**
- Create: `congress_nlp/ids.py`, `congress_nlp/rawdata.py`, `tests/test_ids.py`
- Modify: `congress_nlp/annotation/utils.py`, `congress_nlp/features/extract.py`, `congress_nlp/filtering/coverage.py`, `congress_nlp/filtering/keyword_stats.py`, `congress_nlp/filtering/pipeline.py`, `congress_nlp/filtering/keywords.py`, `congress_nlp/annotation/packets.py`, `congress_nlp/annotation/merge.py`
- Modify: `tests/annotation/test_annotation_utils.py`

**Interfaces:**
- Consumes: `congress_nlp.paths` from Task 5
- Produces:
  - `congress_nlp.ids`: `normalize_id_key(con_legis_num: str) -> str`, `to_canonical_id(con_legis_num: str) -> str | None`, `to_json_path(con_legis_num: str, raw_root: Path) -> Path | None`, `is_amendment(con_legis_num: str) -> bool`, `make_legislation_id(congress: int, legislation_type: str, legislation_number: str) -> str`
  - `congress_nlp.rawdata`: `read_bill_fields(path: Path) -> tuple[str, str, str, str]` returning `(official_title, short_title, summary_text, subjects)` with `subjects` pipe-delimited
  - `congress_nlp.filtering.keywords`: gains `STRONG_KEYWORDS: frozenset[str]`

- [ ] **Step 1: Write the failing test**

Create `tests/test_ids.py`. These cases encode the four contracts from the spec, including the two the old docs got wrong.

```python
from pathlib import Path

from congress_nlp.ids import (
    is_amendment,
    make_legislation_id,
    normalize_id_key,
    to_canonical_id,
    to_json_path,
)

RAW = Path("/raw")


# --- normalize_id_key: dedupe key, never None ------------------------------

def test_normalize_id_key_collapses_dot_variants():
    assert normalize_id_key("118_h.r.1153") == normalize_id_key("118_hr.1153")
    assert normalize_id_key("118_h.r.1153") == "118_hr_1153"


def test_normalize_id_key_strips_leading_zeros():
    assert normalize_id_key("101_s.0007") == "101_s_7"


def test_normalize_id_key_returns_input_unchanged_when_unparseable():
    assert normalize_id_key("garbage") == "garbage"


def test_normalize_id_key_handles_amendments():
    """No amendment special-casing here — it is a dedupe key, not a filter."""
    assert normalize_id_key("108_s.amdt.1797") == "108_samdt_1797"


# --- to_canonical_id: dotted form, None on invalid -------------------------

def test_to_canonical_id_round_trips_bill_forms():
    assert to_canonical_id("101_s.1151") == "101_s.1151"
    assert to_canonical_id("102_s.con.res.107") == "102_s.con.res.107"
    assert to_canonical_id("102_s.j.res.153") == "102_s.j.res.153"


def test_to_canonical_id_accepts_amendments():
    """The old root CLAUDE.md claimed None here. It was wrong: samdt and hamdt
    are both in the valid type set."""
    assert to_canonical_id("108_s.amdt.1797") == "108_s.amdt.1797"
    assert to_canonical_id("108_h.amdt.55") == "108_h.amdt.55"


def test_to_canonical_id_strips_leading_zeros():
    assert to_canonical_id("101_s.0007") == "101_s.7"


def test_to_canonical_id_rejects_bad_input():
    assert to_canonical_id("nounderscore") is None
    assert to_canonical_id("101_s") is None
    assert to_canonical_id("101_s.notanumber") is None
    assert to_canonical_id("101_zz.5") is None


# --- to_json_path: routes bills and amendments to different subdirs --------

def test_to_json_path_builds_bill_paths():
    assert to_json_path("116_s.4604", RAW) == RAW / "116" / "bills" / "s" / "s4604" / "data.json"
    assert to_json_path("102_s.con.res.107", RAW) == (
        RAW / "102" / "bills" / "sconres" / "sconres107" / "data.json"
    )


def test_to_json_path_routes_amendments_to_the_amendments_dir():
    """The discarded extract_features implementation hardcoded bills/ and had no
    amendment support; the adopted one routes correctly."""
    assert to_json_path("108_s.amdt.1797", RAW) == (
        RAW / "108" / "amendments" / "samdt" / "samdt1797" / "data.json"
    )


def test_to_json_path_returns_none_for_unrecognized_types():
    assert to_json_path("101_zz.5", RAW) is None
    assert to_json_path("nounderscore", RAW) is None
    assert to_json_path("101_s", RAW) is None


# --- is_amendment ----------------------------------------------------------

def test_is_amendment():
    assert is_amendment("108_s.amdt.1797")
    assert is_amendment("108_h.amdt.55")
    assert not is_amendment("101_s.1151")
    assert not is_amendment("garbage")


# --- make_legislation_id: builds from directory metadata -------------------

def test_make_legislation_id_uses_dot_notation():
    assert make_legislation_id(101, "s", "1151") == "101_s.1151"
    assert make_legislation_id(102, "sconres", "107") == "102_s.con.res.107"


def test_make_legislation_id_strips_amendment_dir_prefix():
    """Amendment directories are named like 'samdt1' — only the digits count."""
    assert make_legislation_id(108, "samdt", "samdt1") == "108_s.amdt.1"
```

- [ ] **Step 2: Run it to verify it fails**

```bash
python -m pytest tests/test_ids.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'congress_nlp.ids'`.

- [ ] **Step 3: Write `congress_nlp/ids.py`**

```python
"""Bill identifier parsing, normalizing, and path derivation.

Consolidates five implementations that had four different contracts. The
differences are load-bearing, so each function's amendment behavior is stated
explicitly and covered by tests/test_ids.py.

Canonical form is the TWL con_legis_num convention: {congress}_{dot_type}.{number},
e.g. '101_s.1151', '102_s.con.res.107', '108_s.amdt.1'.
"""

import re
from pathlib import Path

# Compact directory type -> dotted form used in a legislation_id
_TYPE_TO_DOT: dict[str, str] = {
    "s": "s",
    "hr": "h.r",
    "sres": "s.res",
    "hres": "h.res",
    "sconres": "s.con.res",
    "hconres": "h.con.res",
    "sjres": "s.j.res",
    "hjres": "h.j.res",
    "samdt": "s.amdt",
    "hamdt": "h.amdt",
}

# Dotted form -> compact directory type. Doubles as the validity check: a type
# absent from this table is not something the raw data tree contains.
_DOT_TO_TYPE: dict[str, str] = {dot: compact for compact, dot in _TYPE_TO_DOT.items()}

_AMENDMENT_TYPES: frozenset[str] = frozenset({"samdt", "hamdt"})


def _split(con_legis_num: str) -> tuple[str, list[str]] | None:
    """Split an ID into (congress, dot tokens). None if it does not parse."""
    raw = str(con_legis_num).strip()
    congress, sep, rest = raw.partition("_")
    if not sep:
        return None
    tokens = rest.lower().split(".")
    if len(tokens) < 2:
        return None
    return congress, tokens


def normalize_id_key(con_legis_num: str) -> str:
    """Canonical dedupe key: '{congress}_{typecompact}_{number}'.

    Collapses dot-notation differences so '118_h.r.1153' and '118_hr.1153' map to
    the same key. Returns the input unchanged when it cannot be parsed — this is
    a dedupe key, so an unparseable ID must still be distinguishable from others.
    Amendments are not special-cased.
    """
    parsed = _split(con_legis_num)
    if parsed is None:
        return str(con_legis_num).strip()
    congress, tokens = parsed
    number = tokens[-1]
    type_compact = "".join(tokens[:-1])
    try:
        number = str(int(number))
    except ValueError:
        pass
    return f"{congress}_{type_compact}_{number}"


def to_canonical_id(con_legis_num: str) -> str | None:
    """Convert a TWL con_legis_num to the pipeline's dotted legislation_id.

    Leading zeros in the number are stripped. Returns None for an unparseable
    ID, a non-integer number, or a type the raw data tree does not contain.
    Amendments are valid: '108_s.amdt.1797' returns '108_s.amdt.1797'.
    """
    parsed = _split(con_legis_num)
    if parsed is None:
        return None
    congress, tokens = parsed
    try:
        number = str(int(tokens[-1]))
    except ValueError:
        return None
    dot_type = ".".join(tokens[:-1])
    if dot_type not in _DOT_TO_TYPE:
        return None
    return f"{congress}_{dot_type}.{number}"


def to_json_path(con_legis_num: str, raw_root: Path) -> Path | None:
    """Derive the raw data.json path for a bill or amendment.

    Routes amendments to '{congress}/amendments/' and everything else to
    '{congress}/bills/'. Returns None for an unrecognized or unparseable ID.
    Callers that must exclude amendments use is_amendment() rather than relying
    on a None return.
    """
    parsed = _split(con_legis_num)
    if parsed is None:
        return None
    congress, tokens = parsed
    number = tokens[-1]
    dot_type = ".".join(tokens[:-1])
    compact = _DOT_TO_TYPE.get(dot_type)
    if compact is None:
        return None
    subdir = "amendments" if compact in _AMENDMENT_TYPES else "bills"
    return raw_root / congress / subdir / compact / f"{compact}{number}" / "data.json"


def is_amendment(con_legis_num: str) -> bool:
    """True for s.amdt / h.amdt IDs. False for anything that does not parse."""
    parsed = _split(con_legis_num)
    if parsed is None:
        return False
    _, tokens = parsed
    return "".join(tokens[:-1]) in _AMENDMENT_TYPES


def make_legislation_id(congress: int, legislation_type: str, legislation_number: str) -> str:
    """Build a legislation_id from directory metadata.

    Does not parse an existing ID and does not read the JSON bill_id field.
    Amendment directories carry a type prefix ('samdt1'), so only the trailing
    digits become the number.
    """
    dot_type = _TYPE_TO_DOT.get(legislation_type, legislation_type)
    match = re.search(r"(\d+)$", legislation_number)
    number = match.group(1) if match else legislation_number
    return f"{congress}_{dot_type}.{number}"
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
python -m pytest tests/test_ids.py -q
```

Expected: 15 passed.

- [ ] **Step 5: Write `congress_nlp/rawdata.py`**

Move the body of `extract_from_json` out of `features/extract.py` verbatim:

```python
"""The single reader of raw legislation data.json files."""

import json
from pathlib import Path


def read_bill_fields(path: Path) -> tuple[str, str, str, str]:
    """Return (official_title, short_title, summary_text, subjects).

    subjects is pipe-delimited, e.g. 'China|Trade|Arms sales'. Every field
    defaults to an empty string when absent.
    """
    with path.open(encoding="utf-8") as fh:
        data: dict = json.load(fh)

    official_title: str = data.get("official_title") or ""
    short_title: str = data.get("short_title") or ""

    summary = data.get("summary")
    summary_text: str = (summary.get("text") or "") if isinstance(summary, dict) else ""

    subjects: list[str] = data.get("subjects") or []
    subjects_str = "|".join(subjects) if isinstance(subjects, list) else ""

    return official_title, short_title, summary_text, subjects_str
```

- [ ] **Step 6: Move `STRONG_KEYWORDS` into `filtering/keywords.py`**

Cut the whole `STRONG_KEYWORDS` frozenset from `features/extract.py` and paste it into `congress_nlp/filtering/keywords.py` below `CHINA_KEYWORDS`. Add this note above it:

```python
# High-specificity subset: rare outside genuine China policy bills, chosen by
# keyword effectiveness analysis. NOT a subset of CHINA_KEYWORDS to be merged
# with it -- CHINA_KEYWORDS is the recall-first Stage 1 filter, this is a
# precision signal used as a model feature. The two are never combined.
```

Also rewrite the `filtering/keywords.py` module docstring, which currently says only "Constants used across the project / Usage: from stage1.constants import XYZ" — a stale import path.

- [ ] **Step 7: Delete the five old implementations and repoint every caller**

| Delete from | What |
|---|---|
| `congress_nlp/annotation/utils.py` | `normalize_id_key` and `_AMENDMENT_TYPES`; keep `is_amendment_type` (it takes a *type*, not an ID, and is a different function) |
| `congress_nlp/features/extract.py` | `con_legis_num_to_path`, `extract_from_json`, `STRONG_KEYWORDS` |
| `congress_nlp/filtering/coverage.py` | `normalize_twl_id` |
| `congress_nlp/filtering/keyword_stats.py` | `_con_legis_num_to_path`, `_DOT_TO_COMPACT`, `_AMENDMENT_COMPACT` |
| `congress_nlp/filtering/pipeline.py` | the `_make_legislation_id` static method |

Then repoint:

- `congress_nlp/features/extract.py`: `from congress_nlp.ids import is_amendment, normalize_id_key, to_json_path`, `from congress_nlp.rawdata import read_bill_fields`, `from congress_nlp.filtering.keywords import STRONG_KEYWORDS`. In `process_labeled_set`, replace the `json_path is None` amendment check with an explicit one **before** the path call:

```python
        if is_amendment(con_legis_num):
            continue

        json_path = to_json_path(con_legis_num, raw_root)
        if json_path is None:
            continue  # unrecognized legislation type
```

  Also move the local `from annotation.annotation_utils import normalize_id_key` out of `dedupe_records`'s body up to the module's import block — the whole reason it was hidden in the function was the `sys.path` shim, which is gone.
- `congress_nlp/filtering/coverage.py`: `from congress_nlp.ids import to_canonical_id`; `df["canonical_id"] = df["con_legis_num"].apply(to_canonical_id)`.
- `congress_nlp/filtering/keyword_stats.py`: `from congress_nlp.ids import to_json_path`; replace `_con_legis_num_to_path(cid, root)` calls with `to_json_path(cid, root)`.
- `congress_nlp/filtering/pipeline.py`: `from congress_nlp.ids import make_legislation_id`; replace `self._make_legislation_id(...)` with `make_legislation_id(...)` at both call sites (the bill check and the amendment check). Delete the now-unused `_LEGISLATION_TYPE_DOT` dict and its `re` import if nothing else in the file uses them — check before deleting.
- `congress_nlp/annotation/packets.py` and `congress_nlp/annotation/merge.py`: replace `from congress_nlp.features.extract import STRONG_KEYWORDS, con_legis_num_to_path, extract_from_json` with `from congress_nlp.filtering.keywords import STRONG_KEYWORDS`, `from congress_nlp.ids import normalize_id_key, to_json_path`, `from congress_nlp.rawdata import read_bill_fields`. Update the call sites accordingly. After this, neither annotation module imports from `features`.

- [ ] **Step 8: Fix the affected existing test**

`tests/annotation/test_annotation_utils.py` imports `normalize_id_key` from `annotation.utils`. Split it: move every `normalize_id_key` case into `tests/test_ids.py` if not already covered there, and change the remaining file to import only the functions that stayed in `congress_nlp.annotation.utils` (`ordinal`, `chamber_from_type`, `bill_number_display`, `congress_gov_url`, `keyword_count_and_strong`, `is_amendment_type`).

- [ ] **Step 9: Run the full suite**

```bash
python -m pytest tests/ -q
```

Expected: all pass.

- [ ] **Step 10: Ask Heagen for the acceptance run**

This is the gate for the whole task. Ask him to run:

```bash
python scripts/04_extract_features.py
```

Compare every number against `docs/superpowers/specs/2026-09-04-features-baseline.txt`: total rows, per-split rows, per-split positives.

**If any number differs, stop and investigate before committing.** The likely cause is the amendment or validation behavior in `to_json_path` versus the discarded `con_legis_num_to_path`. Report which split moved and by how much.

Also expect the `missing JSON` count in the log line to change — the spec records that unrecognized types now return None (skipped, uncounted) instead of producing a nonexistent path (counted as missing). That counter is not part of the acceptance check.

- [ ] **Step 11: Commit**

```bash
git add -A
git commit -m "refactor: consolidate five bill-ID implementations into congress_nlp.ids

The five had four different contracts. to_json_path adopts the amendment-aware
validated builder from keyword_stats and discards the bills-only one from
extract; extract now excludes amendments with an explicit is_amendment() check.
to_canonical_id accepts amendments -- the old root CLAUDE.md claim that it
returns None for them was wrong. Adds rawdata.read_bill_fields and moves
STRONG_KEYWORDS next to CHINA_KEYWORDS as a separate constant, which removes
the annotation -> features dependency."
```

---

## Task 7: View registry, slim schema, and the `--view` CLI

**Files:**
- Create: `congress_nlp/features/views.py`, `congress_nlp/features/load.py`, `congress_nlp/evaluation/metrics.py`
- Create: `tests/features/__init__.py` is **not** needed (pytest works without it); create `tests/features/test_views.py`
- Modify: `congress_nlp/features/extract.py`, `congress_nlp/classifiers/baseline.py`
- Modify: `tests/modeling/test_extract_features_splits.py`

**Interfaces:**
- Consumes: `congress_nlp.paths`, `congress_nlp.ids`, `congress_nlp.filtering.keywords.STRONG_KEYWORDS`
- Produces:
  - `congress_nlp.features.views`: `VIEWS: dict[str, Callable[[pd.Series], str]]`, `DEFAULT_VIEW: str`, `build_view(df: pd.DataFrame, view: str) -> pd.Series`
  - `congress_nlp.features.load`: `load_features(view: str = DEFAULT_VIEW, splits: Sequence[str] | None = None) -> pd.DataFrame` — returns the CSV columns plus a `text` column
  - `congress_nlp.evaluation.metrics`: `report_at_threshold(y_true, y_prob, threshold: float) -> dict[str, float]` with keys `precision`, `recall`, `f1`

- [ ] **Step 1: Write the failing test**

Create `tests/features/test_views.py`:

```python
import pandas as pd
import pytest

from congress_nlp.features.views import DEFAULT_VIEW, VIEWS, build_view


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "official_title": "A bill to restrict exports.",
                "summary_text": "Prohibits certain exports to the PRC.",
                "subjects": "China|Trade|Arms sales",
                "matched_keywords": "prc|china",
            },
            {
                "official_title": "A resolution on Taiwan.",
                "summary_text": "",
                "subjects": "",
                "matched_keywords": "",
            },
        ]
    )


def test_default_view_is_title_subjects():
    assert DEFAULT_VIEW == "title_subjects"
    assert DEFAULT_VIEW in VIEWS


def test_title_subjects_matches_the_pre_restructure_output():
    """Byte-identical to build_title_subjects_text so the baseline reproduces."""
    out = build_view(_frame(), "title_subjects")
    assert out.iloc[0] == "A bill to restrict exports. China, Trade, Arms sales"
    assert out.iloc[1] == "A resolution on Taiwan."


def test_title_subjects_drops_blank_segments_between_pipes():
    df = pd.DataFrame([{"official_title": "T.", "subjects": "China||Trade|"}])
    assert build_view(df, "title_subjects").iloc[0] == "T. China, Trade"


def test_title_summary_joins_title_and_summary():
    out = build_view(_frame(), "title_summary")
    assert out.iloc[0] == "A bill to restrict exports. Prohibits certain exports to the PRC."
    assert out.iloc[1] == "A resolution on Taiwan."


def test_keyword_prefixed_puts_keywords_first():
    out = build_view(_frame(), "keyword_prefixed")
    assert out.iloc[0].startswith("[FILTER: prc, china] ")
    assert out.iloc[1] == "A resolution on Taiwan."


def test_every_view_produces_a_non_empty_string_for_a_titled_row():
    df = _frame()
    for name in VIEWS:
        out = build_view(df, name)
        assert out.iloc[0].strip(), name


def test_unknown_view_raises_and_names_the_valid_choices():
    with pytest.raises(KeyError) as exc:
        build_view(_frame(), "nope")
    assert "title_subjects" in str(exc.value)
```

- [ ] **Step 2: Run it to verify it fails**

```bash
python -m pytest tests/features/test_views.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'congress_nlp.features.views'`.

- [ ] **Step 3: Write `congress_nlp/features/views.py`**

Builders stay row-wise. The scalar logic below is lifted verbatim from `extract.py`'s `build_title_subjects_text`, `build_combined_text`, and `build_text_with_keywords`.

```python
"""Text views over features.csv.

features.csv stores base columns only; each view rebuilds a model input from
them at load time. Adding an input is one function plus one VIEWS entry, and it
is immediately available to every model script as --view <name>.

Builders are row-wise on purpose. Vectorizing the pipe-splitting with
.str.split() changes empty-segment handling, which silently shifts the text and
breaks reproduction against the frozen baseline.
"""

from collections.abc import Callable

import pandas as pd


def _title_subjects(row: pd.Series) -> str:
    """official_title + comma-joined CRS subjects.

    The primary training input. official_title is on 100% of bills and subjects
    on 96.6%+ of no-summary bills, so this is the only input whose distribution
    holds across all four splits — CRS writes summaries with a lag, so a current
    Congress has very few.
    """
    title = str(row.get("official_title") or "").strip()
    subjects = str(row.get("subjects") or "")
    terms = [t.strip() for t in subjects.split("|") if t and t.strip()]
    parts = [title] if title else []
    if terms:
        parts.append(", ".join(terms))
    return " ".join(parts)


def _title_summary(row: pd.Series) -> str:
    """official_title + CRS summary. The original TF-IDF baseline input."""
    title = str(row.get("official_title") or "").strip()
    summary = str(row.get("summary_text") or "").strip()
    return " ".join(p for p in (title, summary) if p)


def _keyword_prefixed(row: pd.Series) -> str:
    """'[FILTER: prc, pla] ' + title_summary.

    The prefix front-loads the Stage 1 keywords so they survive 512-token
    truncation in a transformer, where a long summary would otherwise push them
    out of the window.
    """
    base = _title_summary(row)
    matched = str(row.get("matched_keywords") or "")
    keywords = [kw.strip() for kw in matched.split("|") if kw.strip()]
    if not keywords:
        return base
    return f"[FILTER: {', '.join(keywords)}] {base}"


VIEWS: dict[str, Callable[[pd.Series], str]] = {
    "title_subjects": _title_subjects,
    "title_summary": _title_summary,
    "keyword_prefixed": _keyword_prefixed,
}

DEFAULT_VIEW = "title_subjects"


def build_view(df: pd.DataFrame, view: str) -> pd.Series:
    """Apply a named view row-wise, returning the text column."""
    try:
        builder = VIEWS[view]
    except KeyError:
        raise KeyError(
            f"Unknown view {view!r}. Valid views: {', '.join(sorted(VIEWS))}."
        ) from None
    if df.empty:
        return pd.Series([], dtype="object")
    return df.apply(builder, axis=1)
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
python -m pytest tests/features/test_views.py -q
```

Expected: 7 passed.

- [ ] **Step 5: Write `congress_nlp/features/load.py`**

```python
"""The single entry point for reading features.csv into a model."""

from collections.abc import Sequence

import pandas as pd

from congress_nlp import paths
from congress_nlp.features.views import DEFAULT_VIEW, build_view


def load_features(
    view: str = DEFAULT_VIEW,
    splits: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Read features.csv and attach a `text` column built by the named view.

    splits filters to the given split labels; None returns every row.
    """
    df = pd.read_csv(paths.FEATURES_CSV)
    if splits is not None:
        df = df[df["split"].isin(splits)].copy()
    df["text"] = build_view(df, view)
    return df
```

- [ ] **Step 6: Slim the stored schema**

In `congress_nlp/features/extract.py`:

- Remove `text`, `text_title_subjects`, and `text_with_keywords` from the `BillRecord` NamedTuple and from the `records.append(BillRecord(...))` call.
- Delete `build_combined_text`, `build_title_subjects_text`, and `build_text_with_keywords` — `views.py` owns that logic now.
- Rewrite the module docstring. The current one lists the removed columns, says the script reads only the gold CSV when it reads three sources, and points at `scripts/analyze_filter_coverage.py`, which never existed at that path. Cut it to the output path, the column list, and the two facts a reader cannot infer from the code: gold rows load before intern rows so gold wins de-duplication, and `split` is always derived from the congress number, never read from an input CSV.
- Update `print_split_report`'s trailing NOTE, which currently names the `text` and `text_title_subjects` columns. It should say the CSV stores base columns and that `--view` selects the model input.

- [ ] **Step 7: Rewrite `congress_nlp/classifiers/baseline.py` onto the loader**

```python
"""TF-IDF + Logistic Regression baseline.

    python scripts/05_train_baseline.py --view title_subjects
"""

import argparse

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from congress_nlp.evaluation.metrics import report_at_threshold
from congress_nlp.features.load import load_features
from congress_nlp.features.views import DEFAULT_VIEW, VIEWS

# Tuned on the 117th-Congress validation split, which is 81% positive. The 119
# test split is 51% positive and this threshold does not transfer to it -- see
# the spec and .wolf/STATUS.md before reporting a test number.
PROB_THRESHOLD = 0.3
MAX_FEATURES = 10000
MAX_ITER = 1000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--view",
        choices=sorted(VIEWS),
        default=DEFAULT_VIEW,
        help="Which text view to vectorize.",
    )
    parser.add_argument("--threshold", type=float, default=PROB_THRESHOLD)
    args = parser.parse_args()

    train_df = load_features(view=args.view, splits=["train"])
    val_df = load_features(view=args.view, splits=["val"])

    tfidf = TfidfVectorizer(max_features=MAX_FEATURES)
    x_train = tfidf.fit_transform(train_df["text"].fillna(""))
    x_val = tfidf.transform(val_df["text"].fillna(""))

    clf = LogisticRegression(max_iter=MAX_ITER, class_weight="balanced")
    clf.fit(x_train, train_df["manual_coding"].to_numpy())

    y_prob = clf.predict_proba(x_val)[:, 1]
    scores = report_at_threshold(val_df["manual_coding"].to_numpy(), y_prob, args.threshold)

    print(f"RESULTS (view: {args.view}, threshold: {args.threshold})")
    print(f"Precision: {scores['precision']:.4f} | target >= 0.75")
    print(f"Recall:    {scores['recall']:.4f} | target >= 0.90")
    print(f"F1 Score:  {scores['f1']:.4f} | target >= 0.85")


if __name__ == "__main__":
    main()
```

- [ ] **Step 8: Write `congress_nlp/evaluation/metrics.py`**

```python
"""Scoring shared by every classifier."""

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


def report_at_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Precision, recall, and F1 for a probability cutoff."""
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
```

- [ ] **Step 9: Fix the affected existing tests**

In `tests/modeling/test_extract_features_splits.py`:

- Delete the `text=`, `text_title_subjects=`, and `text_with_keywords=` keyword arguments from `_rec()`.
- Delete `test_billrecord_carries_text_title_subjects`, `test_build_title_subjects_text_joins_pipe_delimited_subjects`, `test_build_title_subjects_text_handles_missing_parts`, and `test_build_title_subjects_text_never_includes_summary`. The first asserts fields that no longer exist; the other three test builders that moved. Their cases are already covered in `tests/features/test_views.py` — **verify each one is covered there before deleting it**, and add it if not.
- Rename the file to `tests/features/test_extract.py` with `git mv`, and delete the now-empty `tests/modeling/` directory.

- [ ] **Step 10: Run the full suite**

```bash
python -m pytest tests/ -q
```

Expected: all pass.

- [ ] **Step 11: Ask Heagen for the two acceptance runs**

Ask him to run both and paste the output:

```bash
python scripts/04_extract_features.py
python scripts/05_train_baseline.py --view title_subjects
```

Two checks:

1. `features.csv` row counts and per-split positives still match the frozen baseline. The column count drops by three; the row count must not move.
2. The validation numbers must reproduce the pre-restructure values **to four decimals**: Precision 0.8063, Recall 1.0000, F1 0.8928. The view rebuilds the same string the stored column held, so anything else means the row-wise builders diverged from the originals.

Also ask for `ls -l data/processed/features.csv` (or `Get-Item`) to confirm the file dropped from ~22.7 MB to roughly 9 MB.

**If the four-decimal check fails, stop.** The most likely cause is a builder that was vectorized or that handles empty segments differently.

- [ ] **Step 12: Commit**

```bash
git add -A
git commit -m "refactor: rebuild text columns at load time from a view registry

features.csv stores twelve base columns; text, text_title_subjects, and
text_with_keywords are rebuilt by congress_nlp.features.views at load time,
taking the file from ~22.7 MB to ~9 MB with no information lost. Adding a
model input is now one function plus one dict entry. train_baseline takes
--view instead of --input-field."
```

---

## Task 8: Documentation rewrite

Last, because Tasks 4 through 7 changed every path and count the docs would state.

**Files:**
- Modify: `README.md`, `CLAUDE.md`, `data/CLAUDE.md`, `scripts/CLAUDE.md`
- Create: `congress_nlp/CLAUDE.md`, `congress_nlp/filtering/CLAUDE.md`, `congress_nlp/annotation/CLAUDE.md`, `congress_nlp/features/CLAUDE.md`, `congress_nlp/classifiers/CLAUDE.md`
- Delete: `models/CLAUDE.md`, `outputs/CLAUDE.md`, `scripts/modeling/CLAUDE.md`, `scripts/pipeline/CLAUDE.md`

**Interfaces:**
- Consumes: the finished tree from Task 7
- Produces: documentation matching it

- [ ] **Step 1: Delete the empty and orphaned instruction files**

```bash
rm -f models/CLAUDE.md outputs/CLAUDE.md
rm -f scripts/modeling/CLAUDE.md scripts/pipeline/CLAUDE.md
```

`models/CLAUDE.md` and `outputs/CLAUDE.md` are 0 bytes and both directories already have `.gitkeep`. All four are gitignored, so no `git rm` is needed.

- [ ] **Step 2: Add the no-live-numbers rule to root `CLAUDE.md`**

Insert this section near the top, above "Project Overview":

```markdown
## Documentation rule: no live numbers

No CLAUDE.md or README in this repo may state a number that changes when a
script is re-run — row counts, split sizes, precision, recall, F1, ROC-AUC,
keyword counts. Those belong in `.wolf/STATUS.md` and in each script's printed
report.

This rule exists because it was violated. Root CLAUDE.md said "report on the 119
test split" while `scripts/modeling/CLAUDE.md` said "Evaluate on val (117), not
test (118)" — two instruction files giving opposite guidance on the most
important evaluation decision, because both had hardcoded a result that later
changed.

Structural facts stay: column names, join keys, file responsibilities, and
gotchas such as the `congress_session` URL bleed and the whole-word regex
behavior.
```

- [ ] **Step 3: Rewrite root `CLAUDE.md`**

Apply, in order:

1. Strip every live number: the `n=1,216` breakdown, `1,743 rows`, all split sizes, every P/R/F1 and ROC-AUC figure, the "58 keywords" count, and the whole threshold-sweep table. Replace each with a pointer to `.wolf/STATUS.md`.
2. Fix `md_files/PLAN1.MD` to `docs/PLAN1.MD`.
3. Correct the false claim that `normalize_twl_id` returns `None` for amendment IDs. It does not — `samdt` and `hamdt` are valid types. State the real contract: `congress_nlp.ids.to_canonical_id` returns None only for an unparseable ID, a non-integer number, or a type outside the valid set.
4. Replace the Module Overview table with the new tree, and the Commands block with the numbered scripts:

```bash
.venv\Scripts\activate
pip install -e .                                   # once per environment

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

5. Replace the "Stage 1 / Stage 2 / Stage 3" vocabulary with the step names: filter, annotate, features, classify, evaluate. Note that `stage2/` is gone and that `pipeline.stage2()` was the file copy, never the transformer.
6. Add a "Adding a new congress" section:

```markdown
## Adding a new congress

1. `python scripts/01_scan.py --congress 120` — writes
   `data/processed/manifests/china_filter_120.csv`. Existing manifests are not
   touched and 101–118 is not rescanned.
2. Add `120: "<split>"` to `SPLITS` in `congress_nlp/splits.py`. This is
   mandatory — `assign_split` raises `UnknownCongressError` for an unlisted
   congress rather than defaulting it into the test split.
3. Label the bills (`scripts/02_build_packets.py`, then
   `scripts/03_merge_annotations.py`).
4. `python scripts/04_extract_features.py` — `manifest_paths()` picks up the new
   manifest by glob.
```

7. Add a "Adding or removing a model input" section:

```markdown
## Adding or removing a model input

Write a row-wise builder in `congress_nlp/features/views.py` and add one entry to
`VIEWS`. It is immediately available as `--view <name>` on every model script.
Removing one is deleting both. `features.csv` never needs regenerating — views
are built at load time.

Builders must stay row-wise (`.apply`). Vectorizing the pipe-splitting changes
empty-segment handling and silently shifts the text.
```

- [ ] **Step 4: Rewrite `data/CLAUDE.md`**

Fix the directory tree to include `data/annotation/`, `data/raw/Summer2026InternsData/`, and `data/processed/manifests/`. Replace the `features.csv` section: list the twelve stored base columns, state that the three text columns are built at load time by `congress_nlp.features.views`, and drop the row count and the `~40% of pre-110 bills` figure. Correct the `split` description — `train` / `val` / `test_legacy` / `test`, sourced from `congress_nlp/splits.py`. Keep the `congress_session` URL-bleed gotcha and the "raw/ is source of truth" rule verbatim.

- [ ] **Step 5: Rewrite `scripts/CLAUDE.md`**

Replace the 5-step run order with the full numbered sequence from Step 3, including the annotation and diagnostic scripts it currently omits. Delete the "All scripts must be run from the project root" paragraph — `paths.PROJECT_ROOT` comes from `__file__` and `pip install -e .` handles imports, so that constraint is gone. State that every script in `scripts/` is a thin delegator and that the logic lives in `congress_nlp/`.

- [ ] **Step 6: Write the five package CLAUDE.md files**

Each is short — responsibility, the files in it, and the gotchas a fresh reader would trip on. No live numbers.

- `congress_nlp/CLAUDE.md`: the leaf-vs-step distinction and the dependency rule — no subpackage imports another subpackage; cross-step needs go through `paths`, `splits`, `ids`, `rawdata`, or `filtering/keywords.py`.
- `congress_nlp/filtering/CLAUDE.md`: carry over the surviving gotchas from the old `stage1/CLAUDE.md` — the whole-word `\b` regex and the apostrophe in `people's`, `AMENDMENTS_START_CONGRESS = 108` and the silent skip below it, and the "extending to a new country" subclass recipe. Add that `CHINA_KEYWORDS` and `STRONG_KEYWORDS` are separate constants that must never be merged, and that `pipeline.stage2()` is the file copy.
- `congress_nlp/annotation/CLAUDE.md`: `REQUIRE_SUMMARY` and what flipping it costs on a current Congress; the caveat that the two-annotator concordance is post-resolution, not inter-annotator agreement, because `Unsure` was resolved in-sheet before download.
- `congress_nlp/features/CLAUDE.md`: the base-columns-versus-views split, the row-wise builder rule, why `title_subjects` is primary (CRS summary lag on a current Congress), and that gold rows load before intern rows so gold wins de-duplication.
- `congress_nlp/classifiers/CLAUDE.md`: that thresholds are tuned on val and reported on test, that the two splits have very different base rates so a threshold does not transfer between them, and that both base rates must be stated whenever a number is reported.

- [ ] **Step 7: Rewrite `README.md`**

This is the only tracked doc, so it gets the same no-live-numbers treatment. Fix `python scripts/analyze_filter_coverage.py` to `python scripts/diagnose_filter.py`, drop the "1,216 manually labeled bills" and "100% recall and 82.6% precision" figures, replace Stage 1/2/3 with the step names, add `pip install -e .` to Setup, and update the Data section for `data/processed/manifests/` and `data/raw/Summer2026InternsData/`.

- [ ] **Step 8: Verify no live numbers survive**

```bash
grep -nE "[0-9],[0-9]{3}|0\.[0-9]{3,4}|[0-9]+\.[0-9]% " README.md CLAUDE.md data/CLAUDE.md scripts/CLAUDE.md congress_nlp/CLAUDE.md congress_nlp/*/CLAUDE.md
```

Expected: no hits other than genuine structural constants — congress numbers (101, 117, 119), `AMENDMENTS_START_CONGRESS = 108`, `max_features` 10000, and the target thresholds (0.75 / 0.90 / 0.85), which are goals rather than results. Review each hit and remove anything that is a measured result.

- [ ] **Step 9: Commit**

```bash
git add README.md
git commit -m "docs: rewrite README for the congress_nlp layout

Fixes the wrong script paths, drops measured results per the new
no-live-numbers rule, and replaces Stage 1/2/3 with the step names."
```

The CLAUDE.md files are gitignored, so only `README.md` is committed. Say so explicitly when reporting the task complete, so nobody expects to see them in the diff.

---

## Task 9: Verify the dependency rule and update anatomy

**Files:**
- Create: `tests/test_layering.py`
- Modify: `.wolf/anatomy.md` (regenerated), `.wolf/STATUS.md`

**Interfaces:**
- Consumes: the finished tree
- Produces: an automated guard against the annotation-imports-features regression

- [ ] **Step 1: Write the test**

```python
"""No congress_nlp subpackage may import another subpackage.

Cross-step needs go through a leaf: paths, splits, ids, rawdata, or
filtering/keywords.py. This is what keeps the re-run loops separable.
"""

import re
from pathlib import Path

from congress_nlp import paths

SUBPACKAGES = ("filtering", "annotation", "features", "classifiers", "evaluation")

# filtering.keywords is a shared constant table, not a step behavior.
ALLOWED_CROSS = {"congress_nlp.filtering.keywords"}

IMPORT_RE = re.compile(r"^\s*(?:from|import)\s+(congress_nlp[\w.]*)", re.MULTILINE)


def test_no_subpackage_imports_another_subpackage():
    package_root = paths.PROJECT_ROOT / "congress_nlp"
    violations: list[str] = []

    for sub in SUBPACKAGES:
        for module in (package_root / sub).rglob("*.py"):
            source = module.read_text(encoding="utf-8")
            for imported in IMPORT_RE.findall(source):
                if imported in ALLOWED_CROSS:
                    continue
                parts = imported.split(".")
                if len(parts) < 2:
                    continue
                other = parts[1]
                if other in SUBPACKAGES and other != sub:
                    violations.append(f"{module.relative_to(package_root)} imports {imported}")

    assert not violations, "Cross-subpackage imports:\n" + "\n".join(violations)
```

- [ ] **Step 2: Run it**

```bash
python -m pytest tests/test_layering.py -q
```

Expected: PASS. If it fails, Task 6 Step 7 left an import behind — fix that rather than widening `ALLOWED_CROSS`.

- [ ] **Step 3: Run the whole suite one final time**

```bash
python -m pytest tests/ -q
```

Expected: all pass. Report the count.

- [ ] **Step 4: Regenerate the anatomy index**

```bash
openwolf scan
```

The SessionStart hook already flagged `.wolf/anatomy.md` as stale, and Tasks 4 through 8 invalidated it further.

- [ ] **Step 5: Update `.wolf/STATUS.md`**

Move the restructure into `✅ Concluído` with the frozen-baseline numbers and the final validation figures. Set `🚀 Próxima fase` to the zero-shot BART quest, carrying forward the bar to beat — TF-IDF at threshold 0.5 on the 119 test split — and the note that no threshold clears both targets with TF-IDF. Bump "Last updated" to the completion date.

- [ ] **Step 6: Commit**

```bash
git add tests/test_layering.py
git commit -m "test: guard the no-cross-subpackage-import rule"
```

`.wolf/` is gitignored, so the anatomy and STATUS updates are not committed. Say so when reporting completion.

---

## Self-Review Notes

**Spec coverage.** Every spec section maps to a task: locked decisions 1 and 4 to Task 8; 2 to Task 7; 3 to Tasks 4 and 5; 5 and 6 to Task 3. Module contracts: `paths` and `splits` to Task 5, `ids` and `rawdata` to Task 6, `views` and `load` to Task 7. Schema change to Task 7. Manifest migration to Task 4. Testing section to Tasks 5 through 7. Commit sequence expanded from four commits to nine tasks so each has a meaningful gate — the spec's commit 3 was doing four independent things.

**Known open item carried into the next quest.** The spec's `evaluation/` scope note stands: `metrics.py` holds only `report_at_threshold`. The threshold sweep and the summary-present versus summary-absent breakdown belong to the BART quest, where they are actually needed.

**One thing the executor must not do.** Do not "fix" the amendment handling in `to_canonical_id` to match the old root `CLAUDE.md`. The documentation was wrong and the code was right; `tests/test_ids.py::test_to_canonical_id_accepts_amendments` pins the correct behavior.
