import json

import pandas as pd
import pytest

from congress_nlp import paths
from congress_nlp.annotation import draw
from congress_nlp.annotation.draw import draw_sample, labeling_sheet
from congress_nlp.ids import is_amendment, normalize_id_key


def _manifest() -> pd.DataFrame:
    rows = []
    for c in (101, 102, 103, 104, 118):
        for n in range(1, 101):
            rows.append((f"{c}_h.r.{n}", c, "hr", str(n)))
        for n in range(1, 6):
            rows.append((f"{c}_s.amdt.{n}", c, "samdt", str(n)))
    return pd.DataFrame(
        rows,
        columns=["legislation_id", "congress", "legislation_type", "legislation_number"],
    )


M = _manifest()


def _write_bill(raw_root, congress, ltype, num, summary="S"):
    d = raw_root / str(congress) / "bills" / ltype / f"{ltype}{num}"
    d.mkdir(parents=True, exist_ok=True)
    body = {"official_title": "T", "summary": {"text": summary}, "subjects": []}
    (d / "data.json").write_text(json.dumps(body), encoding="utf-8")


def test_draw_takes_per_round_times_rounds_from_each_congress_in_range():
    d = draw_sample(M, set(), per_round=3, n_rounds=2, lo=101, hi=102)
    assert d.groupby("congress").size().to_dict() == {101: 6, 102: 6}
    assert d.groupby(["congress", "round"]).size().eq(3).all()


def test_draw_excludes_amendments_and_out_of_range_congresses():
    d = draw_sample(M, set(), per_round=30, lo=101, hi=102)
    assert not d.con_legis_num.map(is_amendment).any()
    assert 118 not in set(d.congress)


def test_draw_is_deterministic_and_order_independent():
    a = draw_sample(M, set(), lo=101, hi=102)
    b = draw_sample(M.sample(frac=1, random_state=0), set(), lo=101, hi=102)
    pd.testing.assert_frame_equal(a, b)


def test_round_two_never_repeats_round_one():
    d = draw_sample(M, set(), lo=101, hi=102)
    assert d.con_legis_num.is_unique


def test_draw_includes_already_labeled_bills_and_marks_them():
    keys = {normalize_id_key(i) for i in M.legislation_id}
    d = draw_sample(M, keys, lo=101, hi=102)
    assert len(d) == 120 and d.already_labeled.all()


def _draw_with_labeled(tmp_path):
    d = draw_sample(M, set(), lo=101, hi=104)
    d["already_labeled"] = [i % 4 == 0 for i in range(len(d))]
    for r in d.itertuples():
        n = r.con_legis_num.split(".")[-1]
        _write_bill(tmp_path, r.congress, "hr", n)
    return d


def test_sheet_holds_only_unlabeled_draws_and_no_keywords_or_labels(tmp_path):
    d = _draw_with_labeled(tmp_path)
    s = labeling_sheet(d, M, tmp_path)
    assert set(s.con_legis_num) == set(d.loc[~d.already_labeled, "con_legis_num"])
    assert "matched_keywords" not in s.columns
    assert s.manual_coding.isna().all()
    assert s.congress_gov_url.str.startswith("https://www.congress.gov/").all()


def test_sheet_puts_round_one_first_and_shuffles_congresses(tmp_path):
    d = _draw_with_labeled(tmp_path)
    s = labeling_sheet(d, M, tmp_path)
    assert list(s["round"]) == sorted(s["round"])
    assert not s[s["round"] == 1].congress.is_monotonic_increasing


def test_sheet_keeps_unlabeled_row_with_blank_text_when_json_missing(tmp_path):
    d = draw_sample(M, set(), per_round=2, n_rounds=1, lo=101, hi=101)
    s = labeling_sheet(d, M, tmp_path)
    assert len(s) == 2 and (s.official_title == "").all()


def test_main_refuses_to_overwrite_an_existing_draw(tmp_path, monkeypatch):
    existing = tmp_path / "draw.csv"
    existing.write_text("sentinel")
    monkeypatch.setattr(paths, "RANDOM_DRAW_CSV", existing)
    monkeypatch.setattr(paths, "RANDOM_DRAW_LABELS", tmp_path / "labels.csv")
    with pytest.raises(FileExistsError):
        draw.main()
    assert existing.read_text() == "sentinel"
    assert not (tmp_path / "labels.csv").exists()
