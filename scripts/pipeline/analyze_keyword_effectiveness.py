"""
Analyze the effectiveness of each keyword in CHINA_KEYWORDS against the gold-standard
labeled set (data/processed/filter_coverage_analysis.csv).

For each keyword reports:
  - TP hits: labeled positives (manual_coding=1) matched by this keyword
  - FP hits: labeled negatives (manual_coding=0) matched by this keyword
  - FP rate: FP hits / total hits
  - Unique TPs: positives where this is the ONLY keyword matched (sole contributor to recall)
  - Verdict: recommendation based on signal/noise ratio

--candidate mode (no pipeline re-run needed):
  Pass one or more candidate keywords to test before adding them to constants.py.
  Scans raw data.json files for all gold-standard bills and reports:
    - New recall: currently-missed positives (in_filter=False) that the keyword would catch
    - Redundant TP: already-caught positives that the keyword also matches
    - FP: labeled negatives matched by the keyword
  Example:
    python scripts/pipeline/analyze_keyword_effectiveness.py --candidate "non-proliferation" "NPT"

Requires filter_coverage_analysis.csv to exist (run scripts/pipeline/analyze_filter_coverage.py first).

Run from the project root:
    python scripts/pipeline/analyze_keyword_effectiveness.py
"""

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from stage1.constants import CHINA_KEYWORDS


COVERAGE_PATH = Path("data/processed/filter_coverage_analysis.csv")
DEFAULT_RAW_DATA = Path("raw_data/raw_legislation")

# Unique TP threshold below which a keyword is flagged as low-value
UNIQUE_TP_WARN = 3
# FP rate threshold above which a keyword is flagged as noisy
FP_RATE_WARN = 0.25

# Reverse of _LEGISLATION_TYPE_DOT: dot notation → compact type for path building
_DOT_TO_COMPACT: dict[str, str] = {
    "s": "s",
    "h.r": "hr",
    "s.res": "sres",
    "h.res": "hres",
    "s.con.res": "sconres",
    "h.con.res": "hconres",
    "s.j.res": "sjres",
    "h.j.res": "hjres",
    "s.amdt": "samdt",
    "h.amdt": "hamdt",
}
_AMENDMENT_COMPACT = {"samdt", "hamdt"}


def _con_legis_num_to_path(con_legis_num: str, raw_data_root: Path) -> Path | None:
    """Map a con_legis_num (e.g. '116_s.4604') to its raw data.json path."""
    raw = str(con_legis_num).strip()
    halves = raw.split("_", 1)
    if len(halves) != 2:
        return None
    congress, rest = halves
    tokens = rest.lower().split(".")
    if len(tokens) < 2:
        return None
    number = tokens[-1]
    dot_type = ".".join(tokens[:-1])
    compact = _DOT_TO_COMPACT.get(dot_type)
    if compact is None:
        return None
    subdir = "amendments" if compact in _AMENDMENT_COMPACT else "bills"
    return raw_data_root / congress / subdir / compact / f"{compact}{number}" / "data.json"


def _extract_text(bill: dict) -> str:
    """Mirror _extract_searchable_text from legislation_pipeline.py."""
    parts: list[str] = []
    for key in ("official_title", "short_title", "popular_title"):
        if value := bill.get(key):
            parts.append(value)
    for title_entry in bill.get("titles", []):
        if text := title_entry.get("title"):
            parts.append(text)
    summary = bill.get("summary")
    if isinstance(summary, dict):
        if text := summary.get("text"):
            parts.append(text)
    for subject in bill.get("subjects", []):
        if isinstance(subject, str):
            parts.append(subject)
        elif isinstance(subject, dict) and (name := subject.get("name")):
            parts.append(name)
    return " ".join(parts).lower()


def keyword_stats(df_pos: pd.DataFrame, df_neg: pd.DataFrame, keyword: str) -> dict:
    kw = keyword.lower()

    tp_hits = df_pos[df_pos["matched_keywords"].fillna("").str.contains(kw, case=False)]
    fp_hits = df_neg[df_neg["matched_keywords"].fillna("").str.contains(kw, case=False)]

    total_hits = len(tp_hits) + len(fp_hits)
    fp_rate = len(fp_hits) / total_hits if total_hits else 0.0

    def is_sole_match(kws_str: str) -> bool:
        parts = [k.strip().lower() for k in kws_str.split("|")]
        return all(kw in p for p in parts)

    unique_tp = tp_hits[tp_hits["matched_keywords"].fillna("").apply(is_sole_match)]

    return {
        "keyword": keyword,
        "tp_hits": len(tp_hits),
        "fp_hits": len(fp_hits),
        "total_hits": total_hits,
        "fp_rate": fp_rate,
        "unique_tps": len(unique_tp),
        "unique_tp_examples": unique_tp[["con_legis_num", "title"]].head(3).values.tolist(),
    }


def candidate_stats(
    df: pd.DataFrame, keyword: str, raw_data_root: Path
) -> dict:
    """
    Test a candidate keyword by scanning raw data.json files for all gold-standard bills.
    Distinguishes new recall (currently missed positives) from redundant TPs and FPs.
    """
    pattern = re.compile(r"\b" + re.escape(keyword.lower()) + r"\b")

    new_recall: list[tuple[str, str]] = []
    redundant_tp: list[tuple[str, str]] = []
    fp_hits: list[tuple[str, str]] = []
    skipped = 0

    for _, row in df.iterrows():
        coding = row.get("manual_coding")
        if pd.isna(coding):
            continue
        path = _con_legis_num_to_path(str(row["con_legis_num"]), raw_data_root)
        if path is None or not path.exists():
            skipped += 1
            continue
        try:
            with path.open(encoding="utf-8") as fh:
                bill = json.load(fh)
            text = _extract_text(bill)
        except Exception:
            skipped += 1
            continue

        if not pattern.search(text):
            continue

        entry = (str(row["con_legis_num"]), str(row.get("title", "")))
        if int(coding) == 1:
            already_caught = bool(str(row.get("in_filter", "False")).lower() == "true")
            if already_caught:
                redundant_tp.append(entry)
            else:
                new_recall.append(entry)
        else:
            fp_hits.append(entry)

    total = len(new_recall) + len(redundant_tp) + len(fp_hits)
    fp_rate = len(fp_hits) / total if total else 0.0

    return {
        "keyword": keyword,
        "new_recall": new_recall,
        "redundant_tp": redundant_tp,
        "fp_hits": fp_hits,
        "total_hits": total,
        "fp_rate": fp_rate,
        "skipped": skipped,
    }


def verdict(stats: dict) -> str:
    if stats["total_hits"] == 0:
        return "NO HITS — keyword may be new or too specific; re-run after pipeline update"
    if stats["unique_tps"] == 0 and stats["fp_hits"] > 0:
        return "REMOVE — zero unique recall contribution, only adds FPs"
    if stats["unique_tps"] < UNIQUE_TP_WARN and stats["fp_rate"] > FP_RATE_WARN:
        return "REVIEW — low unique TPs, high FP rate"
    if stats["fp_rate"] > FP_RATE_WARN:
        return "NOISY — consider tightening (high FP rate)"
    if stats["unique_tps"] == 0:
        return "REDUNDANT — no unique recall; safe to remove if needed"
    return "OK"


def _candidate_verdict(stats: dict) -> str:
    if stats["total_hits"] == 0:
        return "NO HITS in gold-standard set"
    if not stats["new_recall"] and stats["fp_hits"]:
        return "SKIP — no new recall, only FPs"
    if stats["fp_rate"] > FP_RATE_WARN and not stats["new_recall"]:
        return "SKIP — high FP rate, zero new recall"
    if stats["fp_rate"] > FP_RATE_WARN:
        return "NOISY — new recall gain but high FP rate"
    if not stats["new_recall"]:
        return "REDUNDANT — already covered by existing keywords"
    return "ADD"


def run_candidate_mode(candidates: list[str], raw_data_root: Path) -> None:
    if not COVERAGE_PATH.exists():
        print(f"ERROR: {COVERAGE_PATH} not found. Run scripts/analyze_filter_coverage.py first.")
        sys.exit(1)

    df = pd.read_csv(COVERAGE_PATH)
    total_pos = int((df["manual_coding"] == 1).sum())
    total_neg = int((df["manual_coding"] == 0).sum())
    missed_pos = int(((df["manual_coding"] == 1) & (df["in_filter"] != True)).sum())  # noqa: E712

    print()
    print("=" * 80)
    print("  CANDIDATE KEYWORD ANALYSIS")
    print(f"  Gold-standard positives: {total_pos} | negatives: {total_neg}")
    print(f"  Currently missed positives (in_filter=False): {missed_pos}")
    print(f"  Raw data root: {raw_data_root.resolve()}")
    print("=" * 80)
    print()

    for kw in candidates:
        print(f"  Scanning: '{kw}' ...")
        stats = candidate_stats(df, kw, raw_data_root)
        v = _candidate_verdict(stats)

        print(f"\n  [{kw}]  →  {v}")
        print(f"    New recall (missed positives now caught): {len(stats['new_recall'])}")
        print(f"    Redundant TP (already caught):            {len(stats['redundant_tp'])}")
        print(f"    FP (labeled negatives matched):           {len(stats['fp_hits'])}")
        fp_pct = f"{stats['fp_rate']*100:.0f}%" if stats["total_hits"] else "—"
        print(f"    FP rate:                                  {fp_pct}")
        if stats["skipped"]:
            print(f"    Skipped (file missing/unreadable):        {stats['skipped']}")

        if stats["new_recall"]:
            print(f"\n    New recall examples:")
            for leg_id, title in stats["new_recall"][:5]:
                print(f"      + {leg_id}: {title[:70]}")

        if stats["fp_hits"]:
            print(f"\n    FP examples:")
            for leg_id, title in stats["fp_hits"][:5]:
                print(f"      - {leg_id}: {title[:70]}")

        print()

    print("=" * 80)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze keyword effectiveness against the gold-standard labeled set."
    )
    parser.add_argument(
        "--candidate",
        nargs="+",
        metavar="KEYWORD",
        help="Test candidate keywords against raw data without modifying constants.py.",
    )
    parser.add_argument(
        "--raw-data",
        type=Path,
        default=DEFAULT_RAW_DATA,
        metavar="PATH",
        help=f"Root of raw legislation data (default: {DEFAULT_RAW_DATA}).",
    )
    args = parser.parse_args()

    if args.candidate:
        run_candidate_mode(args.candidate, args.raw_data)
        return

    if not COVERAGE_PATH.exists():
        print(f"ERROR: {COVERAGE_PATH} not found. Run scripts/analyze_filter_coverage.py first.")
        sys.exit(1)

    df = pd.read_csv(COVERAGE_PATH)

    df_pos = df[df["manual_coding"] == 1].copy()
    df_neg = df[df["manual_coding"] == 0].copy()

    print()
    print("=" * 80)
    print("  KEYWORD EFFECTIVENESS ANALYSIS")
    print(f"  Gold-standard positives: {len(df_pos)} | negatives: {len(df_neg)}")
    print("=" * 80)
    print()
    print(f"  {'Keyword':<35} {'TP':>4} {'FP':>4} {'FP%':>6} {'UniqTP':>7}  Verdict")
    print("  " + "-" * 78)

    rows = []
    for kw in CHINA_KEYWORDS:
        s = keyword_stats(df_pos, df_neg, kw)
        s["verdict"] = verdict(s)
        rows.append(s)

    for s in sorted(rows, key=lambda x: (-x["fp_rate"], -x["fp_hits"])):
        fp_pct = f"{s['fp_rate']*100:.0f}%" if s["total_hits"] else "—"
        print(f"  {s['keyword']:<35} {s['tp_hits']:>4} {s['fp_hits']:>4} {fp_pct:>6} {s['unique_tps']:>7}  {s['verdict']}")

    print()

    # Detail section: keywords with unique TPs
    print("  UNIQUE TP DETAIL (bills only catchable by this keyword)")
    print("  " + "-" * 78)
    for s in rows:
        if s["unique_tps"] > 0:
            print(f"\n  [{s['keyword']}]  ({s['unique_tps']} unique TPs, FP rate {s['fp_rate']*100:.0f}%)")
            for leg_id, title in s["unique_tp_examples"]:
                print(f"    - {leg_id}: {str(title)[:70]}")

    print()
    print("=" * 80)
    print()

    # Keywords with no hits — likely newly added, needs pipeline re-run
    no_hits = [s["keyword"] for s in rows if s["total_hits"] == 0]
    if no_hits:
        print(f"  Keywords with zero hits (new or not yet in filter results):")
        for kw in no_hits:
            print(f"    - {kw}")
        print("  Re-run main.py then analyze_filter_coverage.py to populate these.")
        print()


if __name__ == "__main__":
    main()
