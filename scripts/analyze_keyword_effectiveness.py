"""
Analyze the effectiveness of each keyword in CHINA_KEYWORDS against the gold-standard
labeled set (coded_data/filter_coverage_analysis.csv).

For each keyword reports:
  - TP hits: labeled positives (manual_coding=1) matched by this keyword
  - FP hits: labeled negatives (manual_coding=0) matched by this keyword
  - FP rate: FP hits / total hits
  - Unique TPs: positives where this is the ONLY keyword matched (sole contributor to recall)
  - Verdict: recommendation based on signal/noise ratio

Requires filter_coverage_analysis.csv to exist (run scripts/analyze_filter_coverage.py first).

Run from the project root:
    python scripts/analyze_keyword_effectiveness.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from constants import CHINA_KEYWORDS


COVERAGE_PATH = Path("coded_data/filter_coverage_analysis.csv")
# Unique TP threshold below which a keyword is flagged as low-value
UNIQUE_TP_WARN = 3
# FP rate threshold above which a keyword is flagged as noisy
FP_RATE_WARN = 0.25


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


def main() -> None:
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
