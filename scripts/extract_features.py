"""
Feature extraction for the transformer classifier.

Reads the gold-standard labeled set (coded_data/twl_coded_legislation_101_to_118.csv),
loads each bill's raw data.json, and extracts text fields used as model inputs.
Also joins matched_keywords from filter_coverage_analysis.csv (Stage 1 output).

Output: coded_data/features.csv

Columns in output:
    con_legis_num       - unique bill ID (e.g. "101_s.1151")
    congress            - congress number (int)
    official_title      - from data.json (always present)
    short_title         - from data.json (often empty)
    summary_text        - from data.json summary.text (missing for ~40% of older bills)
    subjects            - pipe-delimited CRS subject terms from data.json
    manual_coding       - gold label: 1 = China-related, 0 = not
    split               - "train" (101-116) / "val" (117) / "test" (118)
    text                - clean combined input: official_title + summary_text
    text_with_keywords  - transformer input with keyword prefix: "[FILTER: prc, pla] title..."
    has_summary         - True if summary_text is non-empty
    matched_keywords    - pipe-delimited keywords that triggered Stage 1 filter (empty if not in filter)
    keyword_count       - number of distinct keywords matched (0 if not in filter)
    has_strong_keyword  - True if any high-specificity China keyword was matched

Run from the project root:
    python scripts/extract_features.py
"""

import json
import logging
import sys
from pathlib import Path
from typing import NamedTuple

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# --- Temporal split boundaries (must not change once set) ---
# Temporal split prevents data leakage from vocabulary drift across congressional eras.
TRAIN_MAX_CONGRESS = 116   # 101–116 → train
VAL_CONGRESS = 117         # 117 → val (primary eval for precision/F1)
TEST_CONGRESS = 118        # 118 → test (final holdout)

# High-specificity keywords: rare outside genuine China policy bills.
# Derived from keyword effectiveness analysis — these produce few false positives.
# Contrast with broad terms like "china", "human rights", "tariff" which generate many FPs.
STRONG_KEYWORDS: frozenset[str] = frozenset({
    "prc",
    "people's republic of china",
    "pla",
    "people's liberation army",
    "pla navy",
    "pla air force",
    "rocket force",
    "chinese communist party",
    "ccp",
    "communist party of china",
    "xi jinping",
    "li keqiang",
    "li qiang",
    "hu jintao",
    "xinjiang",
    "uyghur",
    "uighur",
    "pboc",
    "people's bank of china",
    "mss",
    "ministry of state security",
})


class BillRecord(NamedTuple):
    """One row in the output features.csv."""
    con_legis_num: str
    congress: int
    official_title: str
    short_title: str
    summary_text: str
    subjects: str            # pipe-delimited, e.g. "China|Trade|Arms sales"
    manual_coding: int
    split: str               # "train", "val", or "test"
    text: str                # clean combined input for TF-IDF baseline
    text_with_keywords: str  # keyword-prefixed input for transformer
    has_summary: bool
    matched_keywords: str    # pipe-delimited Stage 1 keywords (empty if not in filter)
    keyword_count: int       # number of distinct keywords matched
    has_strong_keyword: bool # True if any high-specificity keyword matched


def con_legis_num_to_path(con_legis_num: str, raw_root: Path) -> Path | None:
    """
    Derive the raw data.json path from a con_legis_num string.

    Examples:
        "101_s.1151"        →  raw_root/101/bills/s/s1151/data.json
        "101_hr.2304"       →  raw_root/101/bills/hr/hr2304/data.json
        "102_s.con.res.107" →  raw_root/102/bills/sconres/sconres107/data.json
        "102_s.j.res.153"   →  raw_root/102/bills/sjres/sjres153/data.json
        "101_h.res.5"       →  raw_root/101/bills/hres/hres5/data.json

    Returns None if the ID cannot be parsed or is an amendment.
    """
    raw = str(con_legis_num).strip()

    parts = raw.split("_", 1)
    if len(parts) != 2:
        return None
    congress_str, rest = parts

    tokens = rest.lower().split(".")
    if len(tokens) < 2:
        return None

    number = tokens[-1]
    type_compact = "".join(tokens[:-1])

    if "amdt" in type_compact:
        return None

    dir_name = f"{type_compact}{number}"
    return raw_root / congress_str / "bills" / type_compact / dir_name / "data.json"


def assign_split(congress: int) -> str:
    """Map a congress number to the data split label."""
    if congress <= TRAIN_MAX_CONGRESS:
        return "train"
    if congress == VAL_CONGRESS:
        return "val"
    return "test"


def build_combined_text(official_title: str, summary_text: str) -> str:
    """
    Combine title and summary into a clean string for TF-IDF baselines.
    No markers — keeps the vocabulary pure for bag-of-words models.
    """
    parts = [p.strip() for p in (official_title, summary_text) if p and p.strip()]
    return " ".join(parts)


def build_text_with_keywords(official_title: str, summary_text: str, matched_keywords: str) -> str:
    """
    Build transformer input with a keyword prefix.

    The prefix "[FILTER: kw1, kw2]" gives the transformer compact signal about
    which China-related terms triggered the Stage 1 filter — useful because:
      - Long summaries get truncated at 512 tokens; keywords in the tail may be lost
      - Keyword specificity (prc vs. tariff) is a strong FP predictor
      - All training bills passed the filter, so the prefix is always non-empty

    Example output:
        "[FILTER: prc, pla, beijing] A bill to restrict exports to China..."
    """
    base = build_combined_text(official_title, summary_text)
    if not matched_keywords:
        return base
    # Use comma-separated keywords in the prefix for readability
    kw_display = ", ".join(kw.strip() for kw in matched_keywords.split("|") if kw.strip())
    return f"[FILTER: {kw_display}] {base}"


def compute_keyword_features(matched_keywords: str) -> tuple[int, bool]:
    """
    Derive keyword_count and has_strong_keyword from the pipe-delimited keyword string.

    Returns (keyword_count, has_strong_keyword).
    """
    if not matched_keywords or not matched_keywords.strip():
        return 0, False
    kws = [kw.strip().lower() for kw in matched_keywords.split("|") if kw.strip()]
    count = len(kws)
    strong = any(kw in STRONG_KEYWORDS for kw in kws)
    return count, strong


def extract_from_json(path: Path) -> tuple[str, str, str, str]:
    """
    Load a data.json and return (official_title, short_title, summary_text, subjects).
    All fields default to empty string if missing.
    """
    with path.open(encoding="utf-8") as f:
        data: dict = json.load(f)

    official_title: str = data.get("official_title") or ""
    short_title: str = data.get("short_title") or ""

    summary = data.get("summary")
    summary_text: str = (summary.get("text") or "") if isinstance(summary, dict) else ""

    subjects: list[str] = data.get("subjects") or []
    subjects_str = "|".join(subjects) if isinstance(subjects, list) else ""

    return official_title, short_title, summary_text, subjects_str


def load_keyword_lookup(coverage_path: Path) -> dict[str, str]:
    """
    Build a {con_legis_num: matched_keywords} lookup from filter_coverage_analysis.csv.

    Bills not in the filter will be absent from this dict — treat as empty string.
    This file is generated by scripts/analyze_filter_coverage.py and must exist before
    running this script.
    """
    if not coverage_path.exists():
        log.warning("filter_coverage_analysis.csv not found at %s — matched_keywords will be empty", coverage_path)
        return {}
    df = pd.read_csv(coverage_path, on_bad_lines="skip")
    lookup: dict[str, str] = {}
    for _, row in df.iterrows():
        key = str(row.get("con_legis_num", "")).strip()
        kws = str(row.get("matched_keywords", "")) if pd.notna(row.get("matched_keywords")) else ""
        if key:
            lookup[key] = kws
    log.info("Loaded keyword lookup: %d entries from %s", len(lookup), coverage_path)
    return lookup


def process_labeled_set(
    twl_path: Path,
    raw_root: Path,
    keyword_lookup: dict[str, str],
) -> list[BillRecord]:
    """
    Iterate over all labeled bills, extract text features from raw JSON,
    and join matched_keywords from the Stage 1 filter output.

    Bills whose raw JSON is missing are logged and skipped.
    """
    df = pd.read_csv(twl_path, on_bad_lines="skip")
    log.info("Loaded %d labeled rows from %s", len(df), twl_path)

    df = df[df["manual_coding"].notna()].copy()
    log.info("Rows with valid manual_coding label: %d", len(df))

    records: list[BillRecord] = []
    missing_json = 0
    missing_summary = 0

    for _, row in df.iterrows():
        con_legis_num = str(row["con_legis_num"]).strip()
        manual_coding = int(row["manual_coding"])

        congress_part = con_legis_num.split("_", 1)[0]
        try:
            congress = int(congress_part)
        except ValueError:
            log.warning("Cannot parse congress from: %s — skipping", con_legis_num)
            continue

        json_path = con_legis_num_to_path(con_legis_num, raw_root)

        if json_path is None:
            continue  # amendment — skip

        if not json_path.exists():
            log.debug("Missing JSON for %s at %s", con_legis_num, json_path)
            missing_json += 1
            continue

        official_title, short_title, summary_text, subjects_str = extract_from_json(json_path)

        has_summary = bool(summary_text.strip())
        if not has_summary:
            missing_summary += 1

        matched_keywords = keyword_lookup.get(con_legis_num, "")
        keyword_count, has_strong_keyword = compute_keyword_features(matched_keywords)

        records.append(BillRecord(
            con_legis_num=con_legis_num,
            congress=congress,
            official_title=official_title,
            short_title=short_title,
            summary_text=summary_text,
            subjects=subjects_str,
            manual_coding=manual_coding,
            split=assign_split(congress),
            text=build_combined_text(official_title, summary_text),
            text_with_keywords=build_text_with_keywords(official_title, summary_text, matched_keywords),
            has_summary=has_summary,
            matched_keywords=matched_keywords,
            keyword_count=keyword_count,
            has_strong_keyword=has_strong_keyword,
        ))

    log.info("Processed: %d records | missing JSON: %d | missing summary: %d",
             len(records), missing_json, missing_summary)
    return records


def print_split_report(df: pd.DataFrame) -> None:
    """Print class balance and keyword coverage per split — sanity check before training."""
    print()
    print("=" * 70)
    print("  FEATURE EXTRACTION SUMMARY")
    print("=" * 70)
    for split_name in ("train", "val", "test"):
        subset = df[df["split"] == split_name]
        n = len(subset)
        n_pos = (subset["manual_coding"] == 1).sum()
        n_neg = (subset["manual_coding"] == 0).sum()
        n_summary = subset["has_summary"].sum()
        n_strong = subset["has_strong_keyword"].sum()
        pct_pos = (n_pos / n * 100) if n else 0.0
        pct_summary = (n_summary / n * 100) if n else 0.0
        pct_strong = (n_strong / n * 100) if n else 0.0
        print(
            f"  {split_name:<6}: {n:4d} bills | "
            f"{n_pos} pos / {n_neg} neg ({pct_pos:.0f}% positive) | "
            f"summary: {pct_summary:.0f}% | strong kw: {pct_strong:.0f}%"
        )
    print("=" * 70)
    print()


def main() -> None:
    root = Path(".")
    twl_path = root / "coded_data" / "twl_coded_legislation_101_to_118.csv"
    coverage_path = root / "coded_data" / "filter_coverage_analysis.csv"
    raw_root = root / "raw_data" / "raw_legislation"
    out_path = root / "coded_data" / "features.csv"

    if not twl_path.exists():
        log.error("Labels file not found: %s", twl_path)
        sys.exit(1)
    if not raw_root.exists():
        log.error("Raw data directory not found: %s", raw_root)
        sys.exit(1)

    keyword_lookup = load_keyword_lookup(coverage_path)
    records = process_labeled_set(twl_path, raw_root, keyword_lookup)

    if not records:
        log.error("No records extracted — check paths and data layout")
        sys.exit(1)

    df = pd.DataFrame(records)
    df.to_csv(out_path, index=False)
    log.info("Saved features to %s", out_path)

    print_split_report(df)
    print(f"  Output: {out_path}")
    print()


if __name__ == "__main__":
    main()
