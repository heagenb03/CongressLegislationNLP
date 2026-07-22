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
