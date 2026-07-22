# Intern Annotation Sprint Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the scripts that let 8 high-school interns label bills Yes/No/Unsure in a locked Google Sheet, then merge those labels (with double-annotation + adjudication + gold-trap reliability) into a fresh Congress 119 test set and a batch of training negatives inside `features.csv`.

**Architecture:** Five new/modified Python modules under `scripts/annotation/` plus an extension to `scripts/modeling/extract_features.py`. Pure logic (sampling, ID/URL derivation, agreement resolution, gold-trap scoring) is unit-tested with pytest. Heavy I/O (Stage-1 scan of Congress 119, Google Sheet creation via `gspread`, full feature regeneration) is run manually by Heagen. Data flows: 119 scan → packet plan → Google Sheet → intern labels → merge → resolved raw CSVs → `extract_features.py` → `features.csv`.

**Tech Stack:** Python 3.12, pandas, `gspread` + `google-auth` (Google Sheets), pytest. Reuses the existing `ChinaLegislationPipeline` (Stage 1) and `extract_features.py` machinery.

## Global Constraints

- **Source of truth is read-only:** never modify `data/raw/twl_coded_legislation_101_to_118.csv`. Intern labels go into NEW files under `data/raw/`.
- **Do not run the heavy pipeline scripts** (`scan_congress_119.py`, the `gspread` sheet build in `build_packets.py`, the sheet pull in `merge_annotations.py`, or the full `extract_features.py`). These are marked **[MANUAL — Heagen runs]**; verification is Heagen confirming output. Agents DO run the cheap pytest unit tests.
- **All scripts run from the project root** and use `sys.path` insertion to import project modules (existing convention).
- **Deterministic sampling:** every random operation uses `random.Random(SEED)` with `SEED = 20260721`. No bare `random.*`, no `Math.random`-style nondeterminism — reproducibility is required for the paper.
- **Migration is keyed by `con_legis_num`, never by row position.** The label always travels on the same row as its ID.
- **Label vocabulary is exactly** `Yes` / `No` / `Unsure` (title-case). Internal numeric mapping: `Yes → 1`, `No → 0`, `Unsure → sentinel (never enters training raw)`.
- **Split scheme after this work:** `train` = Congresses 101–116, `val` = 117, `test_legacy` = 118, `test` = 119.
- New Python deps to add to `requirements.txt`: `gspread`, `google-auth`, `pytest`.

---

## File Structure

| Type | Path | Responsibility |
|---|---|---|
| Create | `scripts/annotation/__init__.py` | Package marker. |
| Create | `scripts/annotation/annotation_utils.py` | Pure helpers: ordinal, chamber, bill-number display, congress.gov URL, keyword-count/strong, id normalization key, amendment check. |
| Create | `scripts/annotation/scan_congress_119.py` | Run Stage 1 on Congress 119 only → `data/processed/china_filter_results_119.csv`. |
| Create | `scripts/annotation/build_packets.py` | Sample 119 test bills + 101–116 negatives, pick gold traps, assign 2 interns/bill, write `data/annotation/packet_plan.csv`; then build the Google Sheet from it. |
| Create | `scripts/annotation/merge_annotations.py` | Pull filled sheet, score gold traps, resolve agreements, emit adjudication queue, finalize resolved raw CSVs. |
| Modify | `scripts/modeling/extract_features.py` | New split scheme + union intern-coded files + Congress 119 extraction. |
| Create | `scripts/annotation/gold_trap_selection` (logic in build_packets) | Deterministic gold-trap picks from the gold set. |
| Create | `docs/annotation/intern_coding_guide.md` | One-page Yes/No/Unsure rule + worked examples. |
| Create | `tests/annotation/test_annotation_utils.py` | Unit tests for helpers. |
| Create | `tests/annotation/test_build_packets.py` | Unit tests for sampling/assignment/gold-trap logic. |
| Create | `tests/annotation/test_merge_annotations.py` | Unit tests for resolution/gold-trap scoring. |
| Create | `tests/modeling/test_extract_features_splits.py` | Unit tests for the new split + dedupe logic. |

---

### Task 1: Dependencies + pure helper module

**Files:**
- Modify: `requirements.txt`
- Create: `scripts/annotation/__init__.py`
- Create: `scripts/annotation/annotation_utils.py`
- Test: `tests/annotation/test_annotation_utils.py`

**Interfaces:**
- Produces:
  - `ordinal(n: int) -> str` — `101 → "101st"`, `112 → "112th"`, `119 → "119th"`
  - `chamber_from_type(legislation_type: str) -> str` — compact type (`"hr"`, `"s"`, `"sres"`, …) → `"House"` / `"Senate"`
  - `bill_number_display(legislation_type: str, number: str) -> str` — `("hr","1153") → "H.R. 1153"`
  - `congress_gov_url(congress: int, legislation_type: str, number: str) -> str`
  - `keyword_count_and_strong(matched_keywords: str, strong: frozenset[str]) -> tuple[int, bool]`
  - `normalize_id_key(con_legis_num: str) -> str` — `"118_h.r.1153"` and `"118_hr.1153"` both → `"118_hr_1153"`
  - `is_amendment_type(legislation_type: str) -> bool`

- [ ] **Step 1: Add dependencies**

Append to `requirements.txt`:

```
gspread==6.1.2
google-auth==2.35.0
pytest==8.3.4
```

- [ ] **Step 2: Create the package marker**

Create `scripts/annotation/__init__.py` (empty file):

```python
```

- [ ] **Step 3: Write the failing tests**

Create `tests/annotation/test_annotation_utils.py`:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from annotation.annotation_utils import (
    ordinal,
    chamber_from_type,
    bill_number_display,
    congress_gov_url,
    keyword_count_and_strong,
    normalize_id_key,
    is_amendment_type,
)

STRONG = frozenset({"prc", "pla", "uyghur"})


def test_ordinal_common():
    assert ordinal(101) == "101st"
    assert ordinal(102) == "102nd"
    assert ordinal(103) == "103rd"
    assert ordinal(119) == "119th"


def test_ordinal_teens_are_th():
    assert ordinal(111) == "111th"
    assert ordinal(112) == "112th"
    assert ordinal(113) == "113th"


def test_chamber_from_type():
    assert chamber_from_type("hr") == "House"
    assert chamber_from_type("hres") == "House"
    assert chamber_from_type("s") == "Senate"
    assert chamber_from_type("sconres") == "Senate"


def test_bill_number_display():
    assert bill_number_display("hr", "1153") == "H.R. 1153"
    assert bill_number_display("s", "442") == "S. 442"
    assert bill_number_display("sconres", "7") == "S.Con.Res. 7"


def test_congress_gov_url_bill():
    assert congress_gov_url(118, "hr", "1153") == (
        "https://www.congress.gov/bill/118th-congress/house-bill/1153"
    )
    assert congress_gov_url(119, "s", "442") == (
        "https://www.congress.gov/bill/119th-congress/senate-bill/442"
    )


def test_congress_gov_url_resolution():
    assert congress_gov_url(117, "hres", "9") == (
        "https://www.congress.gov/bill/117th-congress/house-resolution/9"
    )


def test_keyword_count_and_strong():
    assert keyword_count_and_strong("tariff", STRONG) == (1, False)
    assert keyword_count_and_strong("china|prc", STRONG) == (2, True)
    assert keyword_count_and_strong("", STRONG) == (0, False)


def test_normalize_id_key_dot_variants_match():
    assert normalize_id_key("118_h.r.1153") == normalize_id_key("118_hr.1153")
    assert normalize_id_key("118_h.r.1153") == "118_hr_1153"
    assert normalize_id_key("102_s.con.res.107") == "102_sconres_107"


def test_is_amendment_type():
    assert is_amendment_type("samdt") is True
    assert is_amendment_type("hamdt") is True
    assert is_amendment_type("hr") is False
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `python -m pytest tests/annotation/test_annotation_utils.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'annotation.annotation_utils'`

- [ ] **Step 5: Implement the helpers**

Create `scripts/annotation/annotation_utils.py`:

```python
"""Pure, side-effect-free helpers shared by the annotation scripts.

No file or network I/O here — everything is unit-tested.
"""
from __future__ import annotations

# Compact legislation type -> chamber
_CHAMBER = {"h": "House", "s": "Senate"}

# Compact type -> human display prefix
_BILL_DISPLAY = {
    "hr": "H.R.", "s": "S.",
    "hres": "H.Res.", "sres": "S.Res.",
    "hconres": "H.Con.Res.", "sconres": "S.Con.Res.",
    "hjres": "H.J.Res.", "sjres": "S.J.Res.",
}

# Compact type -> congress.gov URL path segment
_URL_TYPE = {
    "hr": "house-bill", "s": "senate-bill",
    "hres": "house-resolution", "sres": "senate-resolution",
    "hconres": "house-concurrent-resolution", "sconres": "senate-concurrent-resolution",
    "hjres": "house-joint-resolution", "sjres": "senate-joint-resolution",
}

_AMENDMENT_TYPES = frozenset({"samdt", "hamdt"})


def ordinal(n: int) -> str:
    """Return the English ordinal string for n (e.g. 101 -> '101st')."""
    if 11 <= (n % 100) <= 13:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def chamber_from_type(legislation_type: str) -> str:
    """Map a compact legislation type to 'House' or 'Senate'."""
    return _CHAMBER.get(legislation_type[:1].lower(), "Unknown")


def bill_number_display(legislation_type: str, number: str) -> str:
    """Human-readable bill label, e.g. ('hr','1153') -> 'H.R. 1153'."""
    prefix = _BILL_DISPLAY.get(legislation_type.lower(), legislation_type.upper())
    return f"{prefix} {number}"


def congress_gov_url(congress: int, legislation_type: str, number: str) -> str:
    """Construct the canonical congress.gov URL for a bill or resolution."""
    seg = _URL_TYPE.get(legislation_type.lower(), "bill")
    return f"https://www.congress.gov/bill/{ordinal(congress)}-congress/{seg}/{number}"


def keyword_count_and_strong(matched_keywords: str, strong: frozenset[str]) -> tuple[int, bool]:
    """Given a pipe-delimited keyword string, return (count, any_strong)."""
    if not matched_keywords or not matched_keywords.strip():
        return 0, False
    kws = [k.strip().lower() for k in matched_keywords.split("|") if k.strip()]
    return len(kws), any(k in strong for k in kws)


def normalize_id_key(con_legis_num: str) -> str:
    """Canonical dedupe key: '{congress}_{typecompact}_{number}'.

    Collapses dot-notation differences so '118_h.r.1153' and '118_hr.1153'
    map to the same key. Returns the raw string unchanged if it can't be parsed.
    """
    raw = str(con_legis_num).strip()
    parts = raw.split("_", 1)
    if len(parts) != 2:
        return raw
    congress, rest = parts
    tokens = rest.lower().split(".")
    if len(tokens) < 2:
        return raw
    number = tokens[-1]
    type_compact = "".join(tokens[:-1])
    try:
        number = str(int(number))
    except ValueError:
        pass
    return f"{congress}_{type_compact}_{number}"


def is_amendment_type(legislation_type: str) -> bool:
    """True for amendment types (samdt/hamdt), which interns never label."""
    return legislation_type.lower() in _AMENDMENT_TYPES
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest tests/annotation/test_annotation_utils.py -v`
Expected: PASS (all 9 tests)

- [ ] **Step 7: Commit**

```bash
git add requirements.txt scripts/annotation/__init__.py scripts/annotation/annotation_utils.py tests/annotation/test_annotation_utils.py
git commit -m "feat: annotation helper module + deps"
```

---

### Task 2: Congress 119 Stage-1 scan (119-only, non-destructive)

**Files:**
- Create: `scripts/annotation/scan_congress_119.py`

**Interfaces:**
- Consumes: `ChinaLegislationPipeline`, `PipelineConfig` from `stage1/legislation_pipeline.py`.
- Produces: `data/processed/china_filter_results_119.csv` (same columns as `china_filter_results.csv`: `legislation_id, congress, legislation_type, legislation_number, matched_keywords, category`).

**Why a separate script/CSV:** re-running `main.py` rescans all ~270k bills and overwrites the existing manifest. This scans only Congress 119 and writes a separate file, leaving `china_filter_results.csv` untouched.

- [ ] **Step 1: Write the script**

Create `scripts/annotation/scan_congress_119.py`:

```python
"""Run the Stage 1 China keyword filter on Congress 119 ONLY.

Writes a separate manifest (china_filter_results_119.csv) so the existing
china_filter_results.csv (101-118) is not touched and ~270k bills are not
re-scanned.

Run from the project root:
    python scripts/annotation/scan_congress_119.py
"""
import sys
from pathlib import Path

# Put stage1/ directly on the path so `legislation_pipeline` and its internal
# `from constants import ...` both resolve regardless of package layout.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "stage1"))

from legislation_pipeline import ChinaLegislationPipeline, PipelineConfig  # noqa: E402


def main() -> None:
    config = PipelineConfig(
        raw_data_root=ROOT / "raw_data" / "raw_legislation",
        output_root=ROOT / "raw_data" / "china_legislation_119",  # unused (no stage2 call)
        csv_path=ROOT / "data" / "processed" / "china_filter_results_119.csv",
        congress_range=range(119, 120),
    )
    pipeline = ChinaLegislationPipeline(config)
    stats = pipeline.stage1()  # writes china_filter_results_119.csv; no stage2 copy
    print(f"Congress 119 survivors written: {stats.total_matched} matched")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: [MANUAL — Heagen runs] Generate the 119 manifest**

Heagen runs:
```bash
python scripts/annotation/scan_congress_119.py
```
Verify: `data/processed/china_filter_results_119.csv` exists and has a plausible survivor count (Congress 118 had ~1,015). Report the actual count — it feeds the target-size decision in Task 3.

- [ ] **Step 3: Commit**

```bash
git add scripts/annotation/scan_congress_119.py
git commit -m "feat: Congress 119 Stage-1 scan script"
```

---

### Task 3: Packet planning (sampling + assignment + gold traps)

**Files:**
- Create: `scripts/annotation/build_packets.py` (planning half; the Google-Sheet half is Task 4)
- Test: `tests/annotation/test_build_packets.py`

**Interfaces:**
- Consumes: `annotation_utils` helpers; `STRONG_KEYWORDS` from `scripts/modeling/extract_features.py`.
- Produces (pure functions, all take an injected `rng: random.Random`):
  - `sample_119_test(df119: pd.DataFrame, n: int, rng) -> pd.DataFrame`
  - `sample_train_negatives(df_all: pd.DataFrame, gold_keys: set[str], n: int, rng) -> pd.DataFrame`
  - `select_gold_traps(gold_df: pd.DataFrame, n_pos: int, n_neg: int, rng) -> pd.DataFrame` (returns cols `con_legis_num, gold_label`)
  - `assign_interns(bill_ids: list[str], interns: list[str]) -> list[tuple[str, str, str]]` (bill_id, intern_a, intern_b — a≠b, load-balanced)
  - `PLAN_COLUMNS: list[str]` — the packet_plan.csv schema
- Produces file (Task 4 step, via `build_plan_rows`): `data/annotation/packet_plan.csv` with columns
  `intern, con_legis_num, congress, legislation_type, chamber, bill_number, title, link, target, is_gold_trap, gold_label, display_order`.

- [ ] **Step 1: Write the failing tests**

Create `tests/annotation/test_build_packets.py`:

```python
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from annotation.build_packets import (
    sample_119_test,
    sample_train_negatives,
    select_gold_traps,
    assign_interns,
)


def test_sample_119_excludes_amendments_and_is_deterministic():
    df = pd.DataFrame({
        "legislation_id": [f"119_hr.{i}" for i in range(10)] + ["119_samdt.1"],
        "congress": [119] * 11,
        "legislation_type": ["hr"] * 10 + ["samdt"],
        "legislation_number": [str(i) for i in range(10)] + ["1"],
        "matched_keywords": ["china"] * 11,
    })
    a = sample_119_test(df, 5, random.Random(1))
    b = sample_119_test(df, 5, random.Random(1))
    assert len(a) == 5
    assert list(a["legislation_id"]) == list(b["legislation_id"])  # deterministic
    assert "samdt" not in set(a["legislation_type"])               # no amendments


def test_sample_train_negatives_filters_weak_single_keyword_and_gold():
    df = pd.DataFrame({
        "legislation_id": ["105_hr.1", "105_hr.2", "105_hr.3", "105_hr.4"],
        "congress": [105, 105, 105, 105],
        "legislation_type": ["hr"] * 4,
        "legislation_number": ["1", "2", "3", "4"],
        # hr.1: single weak kw (keep). hr.2: strong kw (drop). hr.3: 2 kws (drop).
        # hr.4: single weak kw but in gold (drop).
        "matched_keywords": ["tariff", "prc", "china|tariff", "tariff"],
    })
    gold_keys = {"105_hr_4"}  # normalize_id_key form
    out = sample_train_negatives(df, gold_keys, 10, random.Random(1))
    assert list(out["legislation_id"]) == ["105_hr.1"]


def test_select_gold_traps_balanced_and_deterministic():
    gold = pd.DataFrame({
        "con_legis_num": [f"110_hr.{i}" for i in range(20)],
        "manual_coding": [1] * 10 + [0] * 10,
        "matched_keywords": ["prc"] * 10 + ["tariff"] * 10,
    })
    traps = select_gold_traps(gold, n_pos=3, n_neg=4, rng=random.Random(2))
    assert (traps["gold_label"] == 1).sum() == 3
    assert (traps["gold_label"] == 0).sum() == 4
    again = select_gold_traps(gold, n_pos=3, n_neg=4, rng=random.Random(2))
    assert list(traps["con_legis_num"]) == list(again["con_legis_num"])


def test_assign_interns_two_distinct_and_balanced():
    interns = [f"i{n}" for n in range(8)]
    bills = [f"b{n}" for n in range(80)]
    pairs = assign_interns(bills, interns)
    assert len(pairs) == 80
    for _bid, a, b in pairs:
        assert a != b
    # Each intern's total real-bill load within +/-1 of the mean (2*80/8 = 20)
    from collections import Counter
    load = Counter()
    for _bid, a, b in pairs:
        load[a] += 1
        load[b] += 1
    assert max(load.values()) - min(load.values()) <= 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/annotation/test_build_packets.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'annotation.build_packets'`

- [ ] **Step 3: Implement the planning functions**

Create `scripts/annotation/build_packets.py` (planning half — Google Sheet functions added in Task 4):

```python
"""Build intern annotation packets: sample bills, pick gold traps, assign
two annotators per bill, and write data/annotation/packet_plan.csv.

The Google Sheet is created from packet_plan.csv by push_to_sheet() (Task 4).

Run from the project root:
    python scripts/annotation/build_packets.py            # writes packet_plan.csv
    python scripts/annotation/build_packets.py --push     # also creates the Google Sheet
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from annotation.annotation_utils import (  # noqa: E402
    bill_number_display,
    chamber_from_type,
    congress_gov_url,
    is_amendment_type,
    keyword_count_and_strong,
    normalize_id_key,
)
from modeling.extract_features import (  # noqa: E402
    STRONG_KEYWORDS,
    con_legis_num_to_path,
    extract_from_json,
)

# --- Configuration (edit before the sprint) ---
SEED = 20260721
INTERNS: list[str] = [f"intern{n}" for n in range(1, 9)]  # replace with real names
N_TEST_119 = 300
N_TRAIN_NEG = 140
N_GOLD_POS = 4
N_GOLD_NEG = 6

PLAN_COLUMNS = [
    "intern", "con_legis_num", "congress", "legislation_type", "chamber",
    "bill_number", "title", "link", "target", "is_gold_trap", "gold_label",
    "display_order",
]


def sample_119_test(df119: pd.DataFrame, n: int, rng: random.Random) -> pd.DataFrame:
    """Random sample of n non-amendment Congress-119 survivors."""
    pool = df119[~df119["legislation_type"].map(is_amendment_type)].copy()
    idx = list(pool.index)
    rng.shuffle(idx)
    return pool.loc[idx[:n]].reset_index(drop=True)


def sample_train_negatives(
    df_all: pd.DataFrame, gold_keys: set[str], n: int, rng: random.Random
) -> pd.DataFrame:
    """Sample n likely-negative survivors from Congresses 101-116.

    Keep rows with exactly one weak keyword (keyword_count == 1 and no strong
    keyword) that are NOT already in the gold set. Non-amendments only.
    """
    df = df_all[(df_all["congress"] >= 101) & (df_all["congress"] <= 116)].copy()
    df = df[~df["legislation_type"].map(is_amendment_type)]

    def _keep(mk: str) -> bool:
        count, strong = keyword_count_and_strong(str(mk), STRONG_KEYWORDS)
        return count == 1 and not strong

    df = df[df["matched_keywords"].fillna("").map(_keep)]
    df = df[~df["legislation_id"].map(normalize_id_key).isin(gold_keys)]

    idx = list(df.index)
    rng.shuffle(idx)
    return df.loc[idx[:n]].reset_index(drop=True)


def select_gold_traps(
    gold_df: pd.DataFrame, n_pos: int, n_neg: int, rng: random.Random
) -> pd.DataFrame:
    """Pick clear positives and negatives from the gold set as hidden traps.

    Prefers negatives with a single weak keyword (the subtle cases). Returns
    columns con_legis_num, gold_label.
    """
    pos = gold_df[gold_df["manual_coding"] == 1]
    neg = gold_df[gold_df["manual_coding"] == 0]

    def _pick(frame: pd.DataFrame, k: int) -> list[str]:
        ids = list(frame["con_legis_num"].astype(str))
        rng.shuffle(ids)
        return ids[:k]

    rows = [(cid, 1) for cid in _pick(pos, n_pos)] + [(cid, 0) for cid in _pick(neg, n_neg)]
    return pd.DataFrame(rows, columns=["con_legis_num", "gold_label"])


def assign_interns(bill_ids: list[str], interns: list[str]) -> list[tuple[str, str, str]]:
    """Assign each bill to two distinct, load-balanced interns.

    Uses neighbor rotation: bill k -> interns[k % m] and interns[(k+1) % m].
    Guarantees a != b and an even load of 2*len(bills)/m per intern.
    """
    m = len(interns)
    if m < 2:
        raise ValueError("Need at least 2 interns for double annotation")
    pairs: list[tuple[str, str, str]] = []
    for k, bid in enumerate(bill_ids):
        a = interns[k % m]
        b = interns[(k + 1) % m]
        pairs.append((bid, a, b))
    return pairs


def _display_fields(row: pd.Series, raw_root: Path) -> tuple[str, str]:
    """Return (title, link) for a survivor row, loading title from raw JSON."""
    ltype = str(row["legislation_type"])
    number = str(row["legislation_number"])
    congress = int(row["congress"])
    # Reconstruct a con_legis_num the path helper understands.
    cid = f"{congress}_{ltype}.{number}"
    json_path = con_legis_num_to_path(cid, raw_root)
    title = ""
    if json_path is not None and json_path.exists():
        official_title, _short, _summary, _subjects = extract_from_json(json_path)
        title = official_title
    link = congress_gov_url(congress, ltype, number)
    return title, link


def build_plan_rows(
    test_df: pd.DataFrame,
    neg_df: pd.DataFrame,
    traps: pd.DataFrame,
    interns: list[str],
    raw_root: Path,
    rng: random.Random,
) -> pd.DataFrame:
    """Assemble the full packet_plan (one row per intern-bill pairing)."""
    records: list[dict] = []

    # Real bills: test + negatives, each assigned to two interns.
    real = []
    for _i, row in test_df.iterrows():
        real.append((row, "test_119"))
    for _i, row in neg_df.iterrows():
        real.append((row, "train_neg"))

    bill_ids = [f"{int(r['congress'])}_{r['legislation_type']}.{r['legislation_number']}"
                for r, _t in real]
    pairs = assign_interns(bill_ids, interns)

    for (row, target), (cid, a, b) in zip(real, pairs):
        title, link = _display_fields(row, raw_root)
        ltype = str(row["legislation_type"])
        number = str(row["legislation_number"])
        congress = int(row["congress"])
        for intern in (a, b):
            records.append({
                "intern": intern,
                "con_legis_num": cid,
                "congress": congress,
                "legislation_type": ltype,
                "chamber": chamber_from_type(ltype),
                "bill_number": bill_number_display(ltype, number),
                "title": title,
                "link": link,
                "target": target,
                "is_gold_trap": False,
                "gold_label": "",
            })

    # Gold traps: every intern sees every trap.
    for _i, trow in traps.iterrows():
        cid = str(trow["con_legis_num"])
        parts = cid.split("_", 1)
        congress = int(parts[0])
        # Derive type/number from the dot-notation id for display.
        toks = parts[1].split(".")
        number = toks[-1]
        ltype = "".join(toks[:-1])
        json_path = con_legis_num_to_path(cid, raw_root)
        title = ""
        if json_path is not None and json_path.exists():
            title = extract_from_json(json_path)[0]
        link = congress_gov_url(congress, ltype, number)
        for intern in interns:
            records.append({
                "intern": intern,
                "con_legis_num": cid,
                "congress": congress,
                "legislation_type": ltype,
                "chamber": chamber_from_type(ltype),
                "bill_number": bill_number_display(ltype, number),
                "title": title,
                "link": link,
                "target": "gold",
                "is_gold_trap": True,
                "gold_label": int(trow["gold_label"]),
            })

    plan = pd.DataFrame.from_records(records)

    # Per-intern display shuffle so traps aren't clustered; keyed by intern.
    plan["display_order"] = 0
    out_frames = []
    for intern, grp in plan.groupby("intern", sort=True):
        g = grp.sample(frac=1.0, random_state=rng.randint(0, 2**31 - 1)).reset_index(drop=True)
        g["display_order"] = range(len(g))
        out_frames.append(g)
    return pd.concat(out_frames, ignore_index=True)[PLAN_COLUMNS]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--push", action="store_true", help="also create the Google Sheet")
    args = parser.parse_args()

    rng = random.Random(SEED)
    raw_root = ROOT / "raw_data" / "raw_legislation"

    df119 = pd.read_csv(ROOT / "data" / "processed" / "china_filter_results_119.csv")
    df_all = pd.read_csv(ROOT / "data" / "processed" / "china_filter_results.csv")
    gold = pd.read_csv(ROOT / "data" / "raw" / "twl_coded_legislation_101_to_118.csv",
                       on_bad_lines="skip")
    gold = gold[gold["manual_coding"].isin([0, 1])].copy()
    gold["manual_coding"] = gold["manual_coding"].astype(int)
    gold_keys = set(gold["con_legis_num"].astype(str).map(normalize_id_key))

    test_df = sample_119_test(df119, N_TEST_119, rng)
    neg_df = sample_train_negatives(df_all, gold_keys, N_TRAIN_NEG, rng)
    traps = select_gold_traps(gold, N_GOLD_POS, N_GOLD_NEG, rng)

    plan = build_plan_rows(test_df, neg_df, traps, INTERNS, raw_root, rng)

    out_dir = ROOT / "data" / "annotation"
    out_dir.mkdir(parents=True, exist_ok=True)
    plan.to_csv(out_dir / "packet_plan.csv", index=False)
    print(f"Wrote {len(plan)} plan rows for {plan['intern'].nunique()} interns "
          f"({len(test_df)} test, {len(neg_df)} neg, {len(traps)} traps).")

    if args.push:
        from annotation.build_packets_sheet import push_to_sheet
        push_to_sheet(plan)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/annotation/test_build_packets.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: [MANUAL — Heagen runs] Generate the plan (after Task 2 gives 119 counts)**

Heagen adjusts `N_TEST_119` / `N_TRAIN_NEG` / `INTERNS` at the top of `build_packets.py` to real values, then runs:
```bash
python scripts/annotation/build_packets.py
```
Verify: `data/annotation/packet_plan.csv` exists; each real bill appears under exactly two distinct interns; each gold trap appears under all interns; titles/links look right on a spot-check.

- [ ] **Step 6: Commit**

```bash
git add scripts/annotation/build_packets.py tests/annotation/test_build_packets.py
git commit -m "feat: intern packet planning (sampling, traps, assignment)"
```

---

### Task 4: Google Sheet creation (locked, dropdown-only)

**Files:**
- Create: `scripts/annotation/build_packets_sheet.py`
- Create: `docs/annotation/google_sheet_setup.md`

**Interfaces:**
- Consumes: `packet_plan.csv` (via a DataFrame passed to `push_to_sheet`).
- Produces: a Google Sheet with one tab per intern; columns `Congress | Chamber | Bill # | Title | Link | Label | Notes`. Columns A–E + a hidden `con_legis_num` column are protected; `Label` is a Yes/No/Unsure dropdown; `Notes` is free text.

- [ ] **Step 1: Write the service-account setup doc**

Create `docs/annotation/google_sheet_setup.md`:

```markdown
# Google Sheet setup (one-time)

1. Go to https://console.cloud.google.com/ → create a project (or reuse one).
2. Enable **Google Sheets API** and **Google Drive API** for the project.
3. Create a **Service Account** → add a **JSON key** → download it.
4. Save the JSON somewhere private (NOT in the repo). Set an env var:
   - PowerShell: `$env:GOOGLE_APPLICATION_CREDENTIALS = "C:\path\to\key.json"`
5. Create an empty Google Sheet in your Drive. Share it (Editor) with the
   service-account email (found inside the JSON as `client_email`).
6. Copy the Sheet's ID from its URL and set:
   - PowerShell: `$env:ANNOTATION_SHEET_ID = "<sheet id>"`
7. Share the Sheet (Editor) with each intern's Google account.

The key file is never committed. `.gitignore` already excludes `*.json` under
credential paths — keep the key outside the repo to be safe.
```

- [ ] **Step 2: Implement the sheet writer**

Create `scripts/annotation/build_packets_sheet.py`:

```python
"""Create the locked, dropdown-only Google Sheet from packet_plan.csv.

Requires env vars GOOGLE_APPLICATION_CREDENTIALS (service-account JSON) and
ANNOTATION_SHEET_ID (target Sheet). See docs/annotation/google_sheet_setup.md.

This module performs network I/O and is run manually by Heagen.
"""
from __future__ import annotations

import os

import gspread
import pandas as pd
from google.oauth2.service_account import Credentials

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

# Visible header shown to interns. con_legis_num is written to a trailing
# hidden/locked column so migration stays keyed by ID.
HEADER = ["Congress", "Chamber", "Bill #", "Title", "Link", "Label", "Notes", "con_legis_num"]
LABEL_COL_INDEX = 6   # 1-based column F
LABEL_CHOICES = ["Yes", "No", "Unsure"]


def _client() -> gspread.Client:
    path = os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
    creds = Credentials.from_service_account_file(path, scopes=SCOPES)
    return gspread.authorize(creds)


def _tab_rows(plan: pd.DataFrame, intern: str) -> list[list]:
    grp = plan[plan["intern"] == intern].sort_values("display_order")
    rows = [HEADER]
    for _i, r in grp.iterrows():
        rows.append([
            r["congress"], r["chamber"], r["bill_number"], r["title"],
            r["link"], "", "", r["con_legis_num"],
        ])
    return rows


def push_to_sheet(plan: pd.DataFrame) -> None:
    client = _client()
    sh = client.open_by_key(os.environ["ANNOTATION_SHEET_ID"])

    for intern in sorted(plan["intern"].unique()):
        rows = _tab_rows(plan, intern)
        n = len(rows)
        try:
            ws = sh.worksheet(intern)
            sh.del_worksheet(ws)
        except gspread.WorksheetNotFound:
            pass
        ws = sh.add_worksheet(title=intern, rows=n + 5, cols=len(HEADER))
        ws.update("A1", rows, value_input_option="RAW")

        # Data-validation dropdown on the Label column (rows 2..n).
        ws.add_validation(
            f"F2:F{n}",
            gspread.worksheet.ValidationConditionType.one_of_list,
            LABEL_CHOICES,
            strict=True,
            showCustomUi=True,
        )
        # Hide the con_legis_num column (H).
        ws.hide_columns(7, 8)
        # Protect everything, then unprotect Label (F) and Notes (G).
        ws.add_protected_range("A1:H1", description="header")
        ws.add_protected_range(f"A2:E{n}", description="bill fields")
        ws.add_protected_range(f"H2:H{n}", description="id")
        # Freeze header row.
        ws.freeze(rows=1)

    print(f"Pushed {plan['intern'].nunique()} intern tabs to the Sheet.")
```

- [ ] **Step 3: [MANUAL — Heagen runs] Create the Sheet**

Heagen completes `docs/annotation/google_sheet_setup.md`, sets the two env vars, then runs:
```bash
python scripts/annotation/build_packets.py --push
```
Verify in the browser: each intern tab has the readable columns, the Label column is a Yes/No/Unsure dropdown that rejects free-typing, columns A–E and the hidden con_legis_num are protected, and Notes is editable. Fix protection/validation gaps before the interns start.

> **Note on `gspread` API:** protection and validation method names/signatures vary slightly across `gspread` versions. If `add_validation` / `add_protected_range` signatures differ in the installed version, adjust to the installed API (this is expected I/O glue, not a logic change). Confirm the pinned `gspread==6.1.2` API before running.

- [ ] **Step 4: Commit**

```bash
git add scripts/annotation/build_packets_sheet.py docs/annotation/google_sheet_setup.md
git commit -m "feat: locked Google Sheet builder for intern packets"
```

---

### Task 5: Merge, adjudicate, and finalize labels

**Files:**
- Create: `scripts/annotation/merge_annotations.py`
- Test: `tests/annotation/test_merge_annotations.py`

**Interfaces:**
- Consumes: `packet_plan.csv` (expected assignments + trap answers); the filled Google Sheet (pulled to a long-form responses DataFrame with columns `con_legis_num, intern, label, notes`).
- Produces (pure functions):
  - `LABEL_TO_INT: dict[str, int | None]` — `{"Yes":1,"No":0,"Unsure":None}`
  - `score_gold_traps(plan: pd.DataFrame, resp: pd.DataFrame) -> pd.DataFrame` (cols `intern, n_traps, n_correct, accuracy`)
  - `resolve_labels(plan: pd.DataFrame, resp: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]` → `(auto_df, adjudication_df)`
    - `auto_df` cols: `con_legis_num, congress, target, manual_coding`
    - `adjudication_df` cols: `con_legis_num, congress, target, title, link, label_a, label_b, notes_a, notes_b, final_label`
  - `finalize(auto_df, adjudicated_df, plan) -> dict[str, pd.DataFrame]` keyed by target → raw-file frame with cols `con_legis_num, congress, title, manual_coding, split, source`
- Produces files:
  - `data/annotation/reliability_report.csv`
  - `data/annotation/adjudication_queue.csv`
  - `data/annotation/resolved_auto.csv`
  - `data/raw/intern_coded_119.csv`, `data/raw/intern_coded_negatives_101_116.csv` (after adjudication)

- [ ] **Step 1: Write the failing tests**

Create `tests/annotation/test_merge_annotations.py`:

```python
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from annotation.merge_annotations import (
    score_gold_traps,
    resolve_labels,
    finalize,
)


def _plan(rows):
    cols = ["intern", "con_legis_num", "congress", "target", "title", "link",
            "is_gold_trap", "gold_label"]
    return pd.DataFrame(rows, columns=cols)


def test_score_gold_traps():
    plan = _plan([
        ["i1", "110_hr.1", 110, "gold", "t", "u", True, 1],
        ["i2", "110_hr.1", 110, "gold", "t", "u", True, 1],
        ["i1", "110_hr.2", 110, "gold", "t", "u", True, 0],
        ["i2", "110_hr.2", 110, "gold", "t", "u", True, 0],
    ])
    resp = pd.DataFrame([
        ["110_hr.1", "i1", "Yes", ""],   # correct (1)
        ["110_hr.1", "i2", "No", ""],    # wrong
        ["110_hr.2", "i1", "No", ""],    # correct (0)
        ["110_hr.2", "i2", "No", ""],    # correct (0)
    ], columns=["con_legis_num", "intern", "label", "notes"])
    rep = score_gold_traps(plan, resp).set_index("intern")
    assert rep.loc["i1", "n_correct"] == 2 and rep.loc["i1", "n_traps"] == 2
    assert rep.loc["i2", "n_correct"] == 1
    assert abs(rep.loc["i2", "accuracy"] - 0.5) < 1e-9


def test_resolve_labels_agree_disagree_unsure():
    plan = _plan([
        ["i1", "119_hr.1", 119, "test_119", "T1", "L1", False, ""],
        ["i2", "119_hr.1", 119, "test_119", "T1", "L1", False, ""],
        ["i1", "105_hr.9", 105, "train_neg", "T2", "L2", False, ""],
        ["i2", "105_hr.9", 105, "train_neg", "T2", "L2", False, ""],
        ["i1", "119_hr.7", 119, "test_119", "T3", "L3", False, ""],
        ["i2", "119_hr.7", 119, "test_119", "T3", "L3", False, ""],
    ])
    resp = pd.DataFrame([
        ["119_hr.1", "i1", "Yes", ""], ["119_hr.1", "i2", "Yes", ""],   # agree -> 1
        ["105_hr.9", "i1", "No", "x"], ["105_hr.9", "i2", "Yes", "y"],  # disagree
        ["119_hr.7", "i1", "Unsure", ""], ["119_hr.7", "i2", "No", ""], # unsure
    ], columns=["con_legis_num", "intern", "label", "notes"])

    auto, adj = resolve_labels(plan, resp)
    assert set(auto["con_legis_num"]) == {"119_hr.1"}
    assert int(auto.iloc[0]["manual_coding"]) == 1
    assert set(adj["con_legis_num"]) == {"105_hr.9", "119_hr.7"}
    # Adjudication rows carry both labels and notes for the disagreement.
    d = adj.set_index("con_legis_num").loc["105_hr.9"]
    assert {d["label_a"], d["label_b"]} == {"No", "Yes"}


def test_finalize_splits_by_target():
    plan = _plan([
        ["i1", "119_hr.1", 119, "test_119", "T1", "L1", False, ""],
        ["i1", "105_hr.9", 105, "train_neg", "T2", "L2", False, ""],
    ])
    auto = pd.DataFrame([
        ["119_hr.1", 119, "test_119", 1],
    ], columns=["con_legis_num", "congress", "target", "manual_coding"])
    adjudicated = pd.DataFrame([
        ["105_hr.9", 105, "train_neg", 0],
    ], columns=["con_legis_num", "congress", "target", "manual_coding"])

    out = finalize(auto, adjudicated, plan)
    assert set(out["test_119"]["split"]) == {"test"}
    assert set(out["train_neg"]["split"]) == {"train"}
    assert int(out["train_neg"].iloc[0]["manual_coding"]) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/annotation/test_merge_annotations.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'annotation.merge_annotations'`

- [ ] **Step 3: Implement merge/adjudicate/finalize**

Create `scripts/annotation/merge_annotations.py`:

```python
"""Merge intern labels into resolved training/test data.

Two phases:
  1. build : pull the Sheet, score gold traps, auto-accept agreements, and
             write an adjudication queue for disagreements + Unsure.
  2. finalize : after Heagen fills final_label in the adjudication queue,
                combine auto + adjudicated -> data/raw/intern_coded_*.csv.

Run from the project root:
    python scripts/annotation/merge_annotations.py build
    python scripts/annotation/merge_annotations.py finalize

The Sheet pull (pull_responses) is network I/O run manually by Heagen.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from annotation.annotation_utils import normalize_id_key  # noqa: E402
from modeling.extract_features import con_legis_num_to_path, extract_from_json  # noqa: E402

LABEL_TO_INT: dict[str, object] = {"Yes": 1, "No": 0, "Unsure": None}

ANN = ROOT / "data" / "annotation"
RAW = ROOT / "data" / "raw"

TARGET_SPLIT = {"test_119": "test", "train_neg": "train"}
TARGET_FILE = {
    "test_119": RAW / "intern_coded_119.csv",
    "train_neg": RAW / "intern_coded_negatives_101_116.csv",
}


def score_gold_traps(plan: pd.DataFrame, resp: pd.DataFrame) -> pd.DataFrame:
    """Per-intern accuracy on the hidden gold-trap bills."""
    traps = plan[plan["is_gold_trap"]][["intern", "con_legis_num", "gold_label"]].copy()
    traps["gold_label"] = traps["gold_label"].astype(int)
    merged = traps.merge(resp, on=["intern", "con_legis_num"], how="left")
    merged["pred"] = merged["label"].map(LABEL_TO_INT)
    merged["correct"] = merged["pred"] == merged["gold_label"]
    rows = []
    for intern, grp in merged.groupby("intern"):
        n = len(grp)
        c = int(grp["correct"].sum())
        rows.append({"intern": intern, "n_traps": n, "n_correct": c,
                     "accuracy": (c / n) if n else 0.0})
    return pd.DataFrame(rows)


def resolve_labels(plan: pd.DataFrame, resp: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split real (non-trap) bills into auto-accepted and adjudication-needed."""
    real = plan[~plan["is_gold_trap"]].drop_duplicates("con_legis_num").copy()
    resp_real = resp.merge(
        real[["con_legis_num"]], on="con_legis_num", how="inner"
    )

    auto_rows, adj_rows = [], []
    for cid, meta in real.set_index("con_legis_num").iterrows():
        r = resp_real[resp_real["con_legis_num"] == cid]
        labels = list(r["label"])
        notes = list(r["notes"].fillna("")) if "notes" in r else ["", ""]
        while len(labels) < 2:
            labels.append("")
            notes.append("")
        a, b = labels[0], labels[1]
        na, nb = notes[0], notes[1]
        ai, bi = LABEL_TO_INT.get(a), LABEL_TO_INT.get(b)

        if ai is not None and ai == bi:
            auto_rows.append({
                "con_legis_num": cid, "congress": int(meta["congress"]),
                "target": meta["target"], "manual_coding": int(ai),
            })
        else:
            adj_rows.append({
                "con_legis_num": cid, "congress": int(meta["congress"]),
                "target": meta["target"], "title": meta.get("title", ""),
                "link": meta.get("link", ""), "label_a": a, "label_b": b,
                "notes_a": na, "notes_b": nb, "final_label": "",
            })

    auto_df = pd.DataFrame(auto_rows, columns=[
        "con_legis_num", "congress", "target", "manual_coding"])
    adj_df = pd.DataFrame(adj_rows, columns=[
        "con_legis_num", "congress", "target", "title", "link",
        "label_a", "label_b", "notes_a", "notes_b", "final_label"])
    return auto_df, adj_df


def finalize(auto_df: pd.DataFrame, adjudicated_df: pd.DataFrame,
             plan: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Combine auto + adjudicated into per-target raw-file frames."""
    combined = pd.concat([
        auto_df[["con_legis_num", "congress", "target", "manual_coding"]],
        adjudicated_df[["con_legis_num", "congress", "target", "manual_coding"]],
    ], ignore_index=True)
    combined["manual_coding"] = combined["manual_coding"].astype(int)

    titles = plan.drop_duplicates("con_legis_num").set_index("con_legis_num")["title"]
    combined["title"] = combined["con_legis_num"].map(titles).fillna("")
    combined["split"] = combined["target"].map(TARGET_SPLIT)
    combined["source"] = "intern_2026"

    out: dict[str, pd.DataFrame] = {}
    for target, grp in combined.groupby("target"):
        out[target] = grp[[
            "con_legis_num", "congress", "title", "manual_coding", "split", "source"
        ]].reset_index(drop=True)
    return out


def pull_responses() -> pd.DataFrame:
    """[MANUAL I/O] Pull all intern tabs into long form (con_legis_num, intern, label, notes)."""
    import gspread
    from google.oauth2.service_account import Credentials

    scopes = ["https://www.googleapis.com/auth/spreadsheets",
              "https://www.googleapis.com/auth/drive"]
    creds = Credentials.from_service_account_file(
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"], scopes=scopes)
    sh = gspread.authorize(creds).open_by_key(os.environ["ANNOTATION_SHEET_ID"])

    frames = []
    for ws in sh.worksheets():
        records = ws.get_all_records()  # header row -> dicts
        if not records:
            continue
        df = pd.DataFrame(records)
        df["intern"] = ws.title
        df = df.rename(columns={"Label": "label", "Notes": "notes"})
        frames.append(df[["con_legis_num", "intern", "label", "notes"]])
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["build", "finalize"])
    args = parser.parse_args()

    plan = pd.read_csv(ANN / "packet_plan.csv")
    plan["is_gold_trap"] = plan["is_gold_trap"].astype(bool)

    if args.phase == "build":
        resp = pull_responses()
        resp.to_csv(ANN / "responses_raw.csv", index=False)  # audit trail

        score_gold_traps(plan, resp).to_csv(ANN / "reliability_report.csv", index=False)
        auto_df, adj_df = resolve_labels(plan, resp)
        auto_df.to_csv(ANN / "resolved_auto.csv", index=False)
        adj_df.to_csv(ANN / "adjudication_queue.csv", index=False)
        print(f"Auto-accepted {len(auto_df)}; {len(adj_df)} need adjudication. "
              f"Fill final_label in adjudication_queue.csv, then run finalize.")
        return

    # finalize
    auto_df = pd.read_csv(ANN / "resolved_auto.csv")
    adj = pd.read_csv(ANN / "adjudication_queue.csv")
    adj = adj[adj["final_label"].isin([0, 1, "0", "1"])].copy()
    adj["manual_coding"] = adj["final_label"].astype(int)

    out = finalize(auto_df, adj, plan)
    for target, frame in out.items():
        path = TARGET_FILE[target]
        frame.to_csv(path, index=False)
        print(f"Wrote {len(frame)} rows -> {path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/annotation/test_merge_annotations.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: [MANUAL — Heagen runs] After interns finish**

```bash
python scripts/annotation/merge_annotations.py build     # -> reliability_report, adjudication_queue, resolved_auto
# Heagen fills the `final_label` column (0/1) in data/annotation/adjudication_queue.csv
python scripts/annotation/merge_annotations.py finalize   # -> data/raw/intern_coded_*.csv
```
Verify: `reliability_report.csv` accuracies are acceptable; `adjudication_queue.csv` size ≈ 20% of bills; the two `intern_coded_*.csv` files have the expected row counts and only 0/1 in `manual_coding`.

- [ ] **Step 6: Commit**

```bash
git add scripts/annotation/merge_annotations.py tests/annotation/test_merge_annotations.py
git commit -m "feat: merge/adjudicate intern labels with gold-trap scoring"
```

---

### Task 6: Integrate intern labels + Congress 119 into features.csv

**Files:**
- Modify: `scripts/modeling/extract_features.py`
- Test: `tests/modeling/test_extract_features_splits.py`

**Interfaces:**
- Consumes: `data/raw/intern_coded_119.csv`, `data/raw/intern_coded_negatives_101_116.csv` (from Task 5); Congress 119 raw JSON under `raw_data/raw_legislation/119/`.
- Produces: updated `assign_split` semantics + `dedupe_records(records)`; regenerated `data/processed/features.csv` including 119 (`split="test"`), 118 (`split="test_legacy"`), and new train negatives.

- [ ] **Step 1: Write the failing tests**

Create `tests/modeling/test_extract_features_splits.py`:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from modeling.extract_features import assign_split, dedupe_records, BillRecord


def _rec(cid, congress, coding):
    return BillRecord(
        con_legis_num=cid, congress=congress, official_title="t", short_title="",
        summary_text="s", subjects="", manual_coding=coding, split=assign_split(congress),
        text="t s", text_with_keywords="t s", has_summary=True,
        matched_keywords="", keyword_count=0, has_strong_keyword=False,
    )


def test_assign_split_new_scheme():
    assert assign_split(105) == "train"
    assert assign_split(116) == "train"
    assert assign_split(117) == "val"
    assert assign_split(118) == "test_legacy"
    assert assign_split(119) == "test"


def test_dedupe_prefers_first_occurrence_across_dot_variants():
    gold = _rec("118_h.r.1153", 118, 1)     # gold first
    intern = _rec("118_hr.1153", 118, 0)    # same bill, different notation
    out = dedupe_records([gold, intern])
    assert len(out) == 1
    assert out[0].manual_coding == 1        # gold wins
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/modeling/test_extract_features_splits.py -v`
Expected: FAIL — `assign_split(118)` returns `"test"` (old scheme) and `dedupe_records` does not exist (`ImportError`).

- [ ] **Step 3: Update the split boundaries**

In `scripts/modeling/extract_features.py`, replace the split-boundary block (lines ~43–47) and `assign_split` (lines ~129–135):

```python
# --- Temporal split boundaries (must not change once set) ---
# Temporal split prevents data leakage from vocabulary drift across congressional eras.
TRAIN_MAX_CONGRESS = 116    # 101–116 → train
VAL_CONGRESS = 117          # 117 → val (primary eval for precision/F1)
TEST_LEGACY_CONGRESS = 118  # 118 → test_legacy (skewed 96.9% positive; secondary only)
TEST_CONGRESS = 119         # 119 → test (new primary holdout with real negatives)
```

```python
def assign_split(congress: int) -> str:
    """Map a congress number to the data split label."""
    if congress <= TRAIN_MAX_CONGRESS:
        return "train"
    if congress == VAL_CONGRESS:
        return "val"
    if congress == TEST_LEGACY_CONGRESS:
        return "test_legacy"
    return "test"  # 119+
```

- [ ] **Step 4: Add the dedupe helper**

Add to `scripts/modeling/extract_features.py` (after `process_labeled_set`):

```python
def dedupe_records(records: list[BillRecord]) -> list[BillRecord]:
    """Drop duplicate bills, keeping the FIRST occurrence.

    Gold-set records are appended before intern records by the caller, so gold
    labels win on any collision. Dedupe key normalizes dot-notation so
    '118_h.r.1153' and '118_hr.1153' are treated as the same bill.
    """
    from annotation.annotation_utils import normalize_id_key
    seen: set[str] = set()
    out: list[BillRecord] = []
    for r in records:
        key = normalize_id_key(r.con_legis_num)
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out
```

Add near the top with the other `sys.path` line so the import resolves:

```python
sys.path.insert(0, str(Path(__file__).parent.parent))  # already present: enables `annotation`, `modeling`, `stage1`
```

(The existing `sys.path.insert(0, str(Path(__file__).parent.parent))` already puts `scripts/` on the path, so `from annotation.annotation_utils import ...` works.)

- [ ] **Step 5: Wire intern files into `main()`**

In `scripts/modeling/extract_features.py`, replace the body of `main()` from `records = process_labeled_set(...)` through `df = pd.DataFrame(records)` with:

```python
    keyword_lookup = load_keyword_lookup(coverage_path)

    # Gold records FIRST so they win de-duplication against intern labels.
    records = process_labeled_set(twl_path, raw_root, keyword_lookup)

    intern_files = [
        root / "data" / "raw" / "intern_coded_119.csv",
        root / "data" / "raw" / "intern_coded_negatives_101_116.csv",
    ]
    for ipath in intern_files:
        if ipath.exists():
            n_before = len(records)
            records += process_labeled_set(ipath, raw_root, keyword_lookup)
            log.info("Added %d intern records from %s", len(records) - n_before, ipath.name)
        else:
            log.info("Intern file not present (skipping): %s", ipath.name)

    records = dedupe_records(records)

    if not records:
        log.error("No records extracted — check paths and data layout")
        sys.exit(1)

    df = pd.DataFrame(records)
```

> `process_labeled_set` already reads any CSV with `con_legis_num` + `manual_coding` columns and resolves raw JSON via `con_legis_num_to_path`, which handles Congress 119 (it parses the congress prefix generically). No change needed there.

- [ ] **Step 6: Add `test_legacy` to the split report**

In `print_split_report`, change the loop line:

```python
    for split_name in ("train", "val", "test_legacy", "test"):
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `python -m pytest tests/modeling/test_extract_features_splits.py -v`
Expected: PASS (2 tests)

- [ ] **Step 8: [MANUAL — Heagen runs] Regenerate features**

```bash
python scripts/modeling/extract_features.py
```
Verify the summary report shows four splits, `test` (119) now has real negatives, `train` gained negatives, and `test_legacy` (118) is unchanged in size.

- [ ] **Step 9: Commit**

```bash
git add scripts/modeling/extract_features.py tests/modeling/test_extract_features_splits.py
git commit -m "feat: union intern labels + 119 test set into features.csv"
```

---

### Task 7: Intern coding guide

**Files:**
- Create: `docs/annotation/intern_coding_guide.md`

- [ ] **Step 1: Write the guide**

Create `docs/annotation/intern_coding_guide.md`:

```markdown
# Bill Coding Guide (Yes / No / Unsure)

You will see one bill per row: its Congress, chamber, bill number, title, and a
link. Read it, then pick **one** value in the **Label** column. Add a short note
in **Notes** only if something is tricky.

## The question

**Is this bill PRIMARILY about China?**

- **Yes** — China (its government, military, economy, human rights, Taiwan/Hong
  Kong/Tibet/Xinjiang, trade with China, the CCP, etc.) is the MAIN subject.
- **No** — China is only mentioned in passing, or the bill is mainly about
  something/somewhere else and China is one small part (e.g. a giant defense bill
  with one China section, or a bill about another country that name-drops China).
- **Unsure** — you genuinely can't tell after reading. Don't guess. Pick Unsure
  and add a one-line note. An expert will review these.

## How to decide

1. Read the **title** first. Often enough.
2. If unclear, click the **Link** and skim the summary on congress.gov.
3. Ask: "If I had to file this bill in ONE folder, would it go in the China
   folder?" Yes → Yes. Some other folder → No.

## Worked examples

| Title | Label | Why |
|---|---|---|
| "To impose sanctions with respect to the People's Republic of China…" | **Yes** | China is the whole point. |
| "National Defense Authorization Act for FY2024" (mentions China in one section) | **No** | Mostly about the whole military; China is one slice. |
| "A bill to support Taiwan's participation in the World Health Organization" | **Yes** | Taiwan/China policy is the subject. |
| "Foreign Relations Authorization Act" (covers many countries incl. China) | **No** | Not primarily about China. |
| A short bill you can't parse even after reading the summary | **Unsure** | Flag it; don't guess. |

## Rules

- One label per row. Never leave a row blank once you've read it.
- Only the **Label** and **Notes** cells are editable — everything else is locked.
- Don't sort or rearrange rows.
- Work top to bottom; your progress saves automatically.
```

- [ ] **Step 2: Commit**

```bash
git add docs/annotation/intern_coding_guide.md
git commit -m "docs: intern coding guide"
```

---

## Self-Review

**1. Spec coverage:**
- Prerequisite (Stage 1 → 119) → Task 2. ✓
- Packet builder (sampling, 2-annotator assignment, gold traps, sheet) → Tasks 3–4. ✓
- Google Sheet template (protection + dropdown, readable columns + notes) → Task 4. ✓
- Merge/adjudicate + gold-trap reliability → Task 5. ✓
- extract_features extension (119 + intern union + new splits) → Task 6. ✓
- Intern coding guide → Task 7. ✓
- Negative-bias heuristic (`keyword_count==1 & not strong`) → Task 3 `sample_train_negatives`. ✓
- 119=test / 118=test_legacy / 117=val split → Task 6. ✓
- Notes field advisory-only, never parsed → carried in adjudication only. ✓
- Never modify gold CSV → intern labels go to new files (Tasks 5–6). ✓

**2. Placeholder scan:** No TBD/TODO. Sample sizes (`N_TEST_119`, `N_TRAIN_NEG`) are explicit config constants Heagen finalizes after the 119 count is known (Task 3 Step 5) — flagged, not a placeholder.

**3. Type consistency:** `normalize_id_key` used identically in Tasks 3/5/6. `LABEL_TO_INT` maps Yes/No/Unsure consistently. `packet_plan.csv` schema (`PLAN_COLUMNS`) matches what `merge_annotations.py` reads (`is_gold_trap`, `gold_label`, `target`, `title`, `link`). `target` values `test_119`/`train_neg` consistent across Tasks 3/5/6. `BillRecord` reused unchanged.

**Open items carried from the spec (resolve during execution):** exact gold-trap composition is deterministic in code (4 pos / 6 neg, configurable); credential setup is Task 4 Step 1; final target counts set at Task 3 Step 5.
