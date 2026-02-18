"""
Analyze how many gold-standard labeled bills (coded_data/twl_coded_legislation_101_to_118.csv)
were captured by the Stage 1 keyword pre-filter (coded_data/china_filter_results.csv).

Key metrics reported:
  - Overall filter coverage
  - Coverage by manual_coding group (positive / negative / unlabeled)
  - Missed positives (manual_coding=1, not in filter) — critical false negatives

Run from the project root:
    python scripts/analyze_filter_coverage.py
"""

import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def normalize_twl_id(con_legis_num: str) -> str | None:
    """
    Parse a TWL con_legis_num into the canonical bill_id format used by the filter.

    TWL uses dot-separated tokens where the last token is always the bill number
    and the preceding tokens (joined, dots removed) form the bill type:

        '101_s.1151'         -> 's1151-101'
        '102_s.con.res.107'  -> 'sconres107-102'
        '102_s.j.res.153'    -> 'sjres153-102'
        '102_s.res.107'      -> 'sres107-102'
        '102_h.con.res.30'   -> 'hconres30-102'
        '102_s.amdt.1797'    -> None  (amendments not in filter; 'samdt' is unknown type)

    Returns None if the input cannot be parsed or produces an unrecognized bill type.
    """
    # Valid bill types present in china_filter_results.csv
    VALID_TYPES = {"s", "hr", "sres", "hres", "sconres", "hconres", "sjres", "hjres"}

    raw = str(con_legis_num).strip()

    # Split congress from the rest on the first underscore
    parts = raw.split("_", 1)
    if len(parts) != 2:
        return None
    congress, rest = parts

    # Split all dot-separated tokens; last token is the bill number
    tokens = rest.lower().split(".")
    if len(tokens) < 2:
        return None

    bill_number_str = tokens[-1]
    bill_type = "".join(tokens[:-1])  # join type tokens without separator

    # Strip leading zeros by casting through int
    try:
        bill_number = str(int(bill_number_str))
    except ValueError:
        return None

    if bill_type not in VALID_TYPES:
        return None

    type_part = ".".join(tokens[:-1])
    return f"{congress}_{type_part}.{bill_number}"


def load_twl(path: Path) -> pd.DataFrame:
    corrected = path.parent / (path.stem + "_corrected.csv")
    load_path = corrected if corrected.exists() else path
    if load_path != path:
        log.info("Using corrected CSV: %s", load_path)
    df = pd.read_csv(load_path, on_bad_lines="skip")
    log.info("Loaded %d rows from %s", len(df), load_path)
    df["canonical_id"] = df["con_legis_num"].apply(normalize_twl_id)
    failed = df["canonical_id"].isna().sum()
    if failed:
        log.warning("%d rows could not be parsed and will be excluded from join", failed)
    return df


def load_filter(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    log.info("Loaded %d rows from %s", len(df), path)
    return df


def print_group_stats(label: str, group: pd.DataFrame) -> None:
    total = len(group)
    in_filt = group["in_filter"].sum()
    not_in = total - in_filt
    pct = (in_filt / total * 100) if total else 0.0
    print(f"  {label:20s}: {total:5d} total | {in_filt:5d} in filter ({pct:5.1f}%) | {not_in:5d} not in filter")


def main() -> None:
    root = Path(".")
    twl_path = root / "coded_data" / "twl_coded_legislation_101_to_118.csv"
    filter_path = root / "coded_data" / "china_filter_results.csv"
    out_path = root / "coded_data" / "filter_coverage_analysis.csv"

    for p in (twl_path, filter_path):
        if not p.exists():
            log.error("Required file not found: %s", p)
            sys.exit(1)

    twl = load_twl(twl_path)
    filt = load_filter(filter_path)

    # Build O(1) lookup set from filter legislation_ids
    filter_ids: set[str] = set(filt["legislation_id"].dropna().str.strip())

    # Flag each TWL bill
    twl["in_filter"] = twl["canonical_id"].apply(
        lambda x: x in filter_ids if pd.notna(x) else False
    )

    # Merge matched_keywords back for the output CSV
    merged = twl.merge(
        filt[["legislation_id", "matched_keywords"]],
        left_on="canonical_id",
        right_on="legislation_id",
        how="left",
    )

    # --- Summary report ---
    print()
    print("=" * 60)
    print("  KEYWORD FILTER COVERAGE ANALYSIS")
    print("=" * 60)
    print()

    parseable = twl[twl["canonical_id"].notna()]
    print_group_stats("All coded bills", parseable)
    print()

    positives = parseable[parseable["manual_coding"] == 1]
    negatives = parseable[parseable["manual_coding"] == 0]
    unlabeled = parseable[parseable["manual_coding"].isna()]

    print("  Breakdown by manual_coding:")
    print_group_stats("Positive (China=1)", positives)
    print_group_stats("Negative (China=0)", negatives)
    print_group_stats("Unlabeled (NaN)", unlabeled)
    print()

    # Recall on positives (primary metric)
    n_pos = len(positives)
    n_pos_captured = positives["in_filter"].sum()
    recall = (n_pos_captured / n_pos * 100) if n_pos else 0.0
    print(f"  Filter recall on positives : {recall:.1f}%  (target >= 90%)")

    # Precision on filter hits within labeled set
    labeled = parseable[parseable["manual_coding"].notna()]
    hits_in_labeled = labeled[labeled["in_filter"]]
    true_positives = hits_in_labeled[hits_in_labeled["manual_coding"] == 1]
    precision = (len(true_positives) / len(hits_in_labeled) * 100) if len(hits_in_labeled) else 0.0
    print(f"  Filter precision (labeled) : {precision:.1f}%  (target >= 75%)")
    print()

    # Missed positives — most critical errors
    missed = positives[~positives["in_filter"]][["con_legis_num", "canonical_id", "title"]]
    print(f"  Missed positives (FN): {len(missed)}")
    if len(missed):
        print()
        print("  con_legis_num         canonical_id          title")
        print("  " + "-" * 90)
        for _, row in missed.iterrows():
            title_trunc = str(row.get("title", ""))[:55]
            print(f"  {str(row['con_legis_num']):<22}{str(row['canonical_id']):<22}{title_trunc}")
    print()

    # Filtered negatives (FP) — informational only, transformer handles these
    fp = negatives[negatives["in_filter"]]
    print(f"  Filtered negatives (FP): {len(fp)}  (transformer will reject these)")
    print()

    # Save detailed results
    merged.to_csv(out_path, index=False)
    log.info("Detailed results saved to %s", out_path)
    print(f"  Output saved: {out_path}")
    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
