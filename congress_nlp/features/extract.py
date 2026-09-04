"""
Feature extraction for the transformer classifier.

Reads the gold-standard labeled set (data/raw/twl_coded_legislation_101_to_118.csv),
loads each bill's raw data.json, and extracts text fields used as model inputs.
Also joins matched_keywords from filter_coverage_analysis.csv (Stage 1 output).

Output: data/processed/features.csv

Columns in output:
    con_legis_num       - unique bill ID (e.g. "101_s.1151")
    congress            - congress number (int)
    official_title      - from data.json (always present)
    short_title         - from data.json (often empty)
    summary_text        - from data.json summary.text (missing for ~40% of older bills)
    subjects            - pipe-delimited CRS subject terms from data.json
    manual_coding       - gold label: 1 = China-related, 0 = not
    split               - "train" (101-116) / "val" (117) / "test_legacy" (118) / "test" (119+)
    text                - clean combined input: official_title + summary_text
    text_with_keywords  - transformer input with keyword prefix: "[FILTER: prc, pla] title..."
    has_summary         - True if summary_text is non-empty
    matched_keywords    - pipe-delimited keywords that triggered Stage 1 filter (empty if not in filter)
    keyword_count       - number of distinct keywords matched (0 if not in filter)
    has_strong_keyword  - True if any high-specificity China keyword was matched

Run from the project root:
    python scripts/modeling/extract_features.py
"""

import logging
import sys
from pathlib import Path
from typing import NamedTuple

import pandas as pd

from congress_nlp import paths
from congress_nlp.filtering.keywords import STRONG_KEYWORDS
from congress_nlp.ids import is_amendment, normalize_id_key, to_json_path
from congress_nlp.rawdata import read_bill_fields
from congress_nlp.splits import assign_split


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# --- Intern-labeled data (2026 sprint) ---
# The sprint CSVs live in a subfolder (paths.INTERN_SUBDIR), NOT directly
# under data/raw/.
INTERN_FILENAMES = ("intern_coded_119.csv", "intern_coded_negatives_101_116.csv")

class BillRecord(NamedTuple):
    """One row in the output features.csv."""
    con_legis_num: str
    congress: int
    official_title: str
    short_title: str
    summary_text: str
    subjects: str            # pipe-delimited, e.g. "China|Trade|Arms sales"
    manual_coding: int
    split: str               # "train", "val", "test_legacy", or "test"
    text: str                # title + summary — the original TF-IDF baseline input
    text_title_subjects: str # title + subjects — primary input; present on 100% of bills
    text_with_keywords: str  # keyword-prefixed input for transformer
    has_summary: bool
    matched_keywords: str    # pipe-delimited Stage 1 keywords (empty if not in filter)
    keyword_count: int       # number of distinct keywords matched
    has_strong_keyword: bool # True if any high-specificity keyword matched


def build_combined_text(official_title: str, summary_text: str) -> str:
    """
    Combine title and summary into a clean string for TF-IDF baselines.
    No markers — keeps the vocabulary pure for bag-of-words models.
    """
    parts = [p.strip() for p in (official_title, summary_text) if p and p.strip()]
    return " ".join(parts)


def build_title_subjects_text(official_title: str, subjects: str) -> str:
    """Combine title and CRS subject terms — the fields present on every bill.

    CRS summaries are written with a lag, so only 33% of the 119th Congress has
    one while train/val are at 100%. Title and subjects are available across all
    four splits, which makes this the only input configuration whose
    distribution does not shift between training and test.

    Subjects arrive pipe-delimited ("China|Trade|Arms sales"); pipes are noise
    for bag-of-words models, so they become comma-separated.
    """
    subject_terms = [t.strip() for t in subjects.split("|") if t and t.strip()]
    parts = [official_title.strip()] if official_title and official_title.strip() else []
    if subject_terms:
        parts.append(", ".join(subject_terms))
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


def load_manifest_keyword_lookup(manifest_paths: list[Path]) -> dict[str, str]:
    """Build {legislation_id: matched_keywords} from Stage-1 manifest CSVs.

    Covers survivors not in the gold coverage file (e.g. Congress 119 and
    sampled training negatives). Missing files are skipped. Keyed by the
    canonical dotted legislation_id, matching the con_legis_num used by
    intern-coded rows.
    """
    lookup: dict[str, str] = {}
    for path in manifest_paths:
        if not path.exists():
            log.info("Manifest not found (skipping): %s", path)
            continue
        df = pd.read_csv(path, on_bad_lines="skip")
        n_loaded = 0
        for _, row in df.iterrows():
            key = str(row.get("legislation_id", "")).strip()
            kws = str(row.get("matched_keywords", "")) if pd.notna(row.get("matched_keywords")) else ""
            if key:
                lookup[key] = kws
                n_loaded += 1
        log.info("Loaded manifest keyword lookup: %d entries from %s", n_loaded, path)
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

        if is_amendment(con_legis_num):
            continue

        json_path = to_json_path(con_legis_num, raw_root)
        if json_path is None:
            continue  # unrecognized legislation type

        if not json_path.exists():
            log.debug("Missing JSON for %s at %s", con_legis_num, json_path)
            missing_json += 1
            continue

        official_title, short_title, summary_text, subjects_str = read_bill_fields(json_path)

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
            text_title_subjects=build_title_subjects_text(official_title, subjects_str),
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


def dedupe_records(records: list[BillRecord]) -> list[BillRecord]:
    """Drop duplicate bills, keeping the FIRST occurrence.

    Gold-set records are appended before intern records by the caller, so gold
    labels win on any collision. Dedupe key normalizes dot-notation so
    '118_h.r.1153' and '118_hr.1153' are treated as the same bill.
    """
    seen: set[str] = set()
    out: list[BillRecord] = []
    for r in records:
        key = normalize_id_key(r.con_legis_num)
        if key in seen:
            log.debug("Dropped duplicate row for %s (normalized key %s already seen)", r.con_legis_num, key)
            continue
        seen.add(key)
        out.append(r)
    return out


def resolve_intern_files(root: Path) -> list[Path]:
    """Locate the intern label CSVs, raising if the whole set is missing.

    A single missing file is a legitimate partial run and only warns. All of
    them missing means the directory moved or the merge never ran — failing
    loudly beats writing a features.csv with no intern rows in it.
    """
    directory = root / paths.INTERN_SUBDIR
    found = [directory / name for name in INTERN_FILENAMES]
    present = [p for p in found if p.exists()]

    if not present:
        raise FileNotFoundError(
            f"No intern label files found in {paths.INTERN_SUBDIR}. Expected "
            f"{list(INTERN_FILENAMES)}. Run "
            f"'python scripts/03_merge_annotations.py finalize' first, "
            f"or correct INTERN_SUBDIR in congress_nlp/paths.py if the data moved."
        )

    for path in found:
        if path not in present:
            log.warning("Intern file not present (skipping): %s", path)

    return present


def print_split_report(df: pd.DataFrame) -> None:
    """Print class balance and keyword coverage per split — sanity check before training."""
    print()
    print("=" * 70)
    print("  FEATURE EXTRACTION SUMMARY")
    print("=" * 70)
    for split_name in ("train", "val", "test_legacy", "test"):
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
    print("-" * 70)
    print("  NOTE: 'summary' is CRS coverage. The `text` field (title+summary)")
    print("  shrinks where it is low; `text_title_subjects` does not. Prefer the")
    print("  latter for training - see CLAUDE.md.")
    print("=" * 70)
    print()


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

    # Gold records FIRST so they win de-duplication against intern labels.
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


if __name__ == "__main__":
    main()
