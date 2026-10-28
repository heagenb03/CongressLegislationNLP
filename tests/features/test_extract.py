from pathlib import Path

import pandas as pd
import pytest

from congress_nlp import paths
from congress_nlp.splits import assign_split

from congress_nlp.features.extract import (
    attach_draw_rounds,
    dedupe_records,
    validate_draw_labels,
    load_manifest_keyword_lookup,
    BillRecord,
)


def _rec(cid, congress, coding):
    return BillRecord(
        con_legis_num=cid, congress=congress, official_title="t", short_title="",
        summary_text="s", subjects="", manual_coding=coding, split=assign_split(congress),
        has_summary=True,
        matched_keywords="", keyword_count=0, has_strong_keyword=False,
    )


def test_dedupe_prefers_first_occurrence_across_dot_variants():
    gold = _rec("118_h.r.1153", 118, 1)     # gold first
    intern = _rec("118_hr.1153", 118, 0)    # same bill, different notation
    out = dedupe_records([gold, intern])
    assert len(out) == 1
    assert out[0].manual_coding == 1        # gold wins


def test_load_manifest_keyword_lookup_reads_multiple_manifests(tmp_path):
    manifest = tmp_path / "china_filter_results_119.csv"
    manifest.write_text(
        "legislation_id,congress,legislation_type,legislation_number,matched_keywords,category\n"
        "119_h.r.21,119,hr,21,china|prc,bill\n"
        "119_h.r.22,119,hr,22,,bill\n",
        encoding="utf-8",
    )
    missing = tmp_path / "does_not_exist.csv"

    lookup = load_manifest_keyword_lookup([manifest, missing])

    assert lookup["119_h.r.21"] == "china|prc"
    assert lookup["119_h.r.22"] == ""
    assert "does_not_exist" not in str(lookup)
    assert len(lookup) == 2


# --- intern label file resolution -------------------------------------------

def test_resolve_intern_files_returns_existing(tmp_path):
    from congress_nlp.features.extract import resolve_intern_files, INTERN_FILENAMES
    d = tmp_path / paths.INTERN_SUBDIR
    d.mkdir(parents=True)
    for name in INTERN_FILENAMES:
        (d / name).write_text("con_legis_num,manual_coding\n119_hr.1,1\n", encoding="utf-8")

    found = resolve_intern_files(tmp_path)
    assert [p.name for p in found] == list(INTERN_FILENAMES)


def test_resolve_intern_files_allows_partial(tmp_path):
    """One file present is a legitimate partial run — warn, don't fail."""
    from congress_nlp.features.extract import resolve_intern_files, INTERN_FILENAMES
    d = tmp_path / paths.INTERN_SUBDIR
    d.mkdir(parents=True)
    (d / INTERN_FILENAMES[0]).write_text(
        "con_legis_num,manual_coding\n119_hr.1,1\n", encoding="utf-8")

    found = resolve_intern_files(tmp_path)
    assert [p.name for p in found] == [INTERN_FILENAMES[0]]


def test_resolve_intern_files_raises_when_none_found(tmp_path):
    """The whole intern set vanishing must NOT be swallowed by an exists() guard."""
    import pytest
    from congress_nlp.features.extract import resolve_intern_files
    with pytest.raises(FileNotFoundError) as exc:
        resolve_intern_files(tmp_path)
    assert str(paths.INTERN_SUBDIR) in str(exc.value)


def test_intern_subdir_points_at_summer2026_folder():
    from congress_nlp import paths

    assert paths.INTERN_SUBDIR.as_posix() == "data/raw/Summer2026InternsData"


def test_draw_labels_rank_after_existing_labels():
    gold = _rec("105_hr.9", 105, 1)
    drawn = _rec("105_h.r.9", 105, 0)
    assert dedupe_records([gold, drawn])[0].manual_coding == 1


def test_bad_draw_label_values_stop_extraction_and_are_named(tmp_path):
    p = tmp_path / "labels.csv"
    p.write_text("con_legis_num,manual_coding\n101_hr.1,1\n101_hr.2,Yes\n101_hr.3,2\n101_hr.4,\n101_hr.5, 0 \n")
    with pytest.raises(ValueError) as exc:
        validate_draw_labels(p)
    assert "101_hr.2" in str(exc.value) and "101_hr.3" in str(exc.value)
    assert "101_hr.4" not in str(exc.value) and "101_hr.5" not in str(exc.value)


def test_attach_marks_only_drawn_survivors():
    df = pd.DataFrame({"con_legis_num": ["101_h.r.1", "101_h.r.2", "101_h.r.3"]})
    draw = pd.DataFrame({"con_legis_num": ["101_hr.1", "101_hr.2"], "round": [1, 2]})
    out = attach_draw_rounds(df, draw, survivor_keys={"101_hr_1"})
    assert out.random_draw_round.tolist()[0] == 1
    assert out.random_draw_round.isna().tolist()[1:] == [True, True]
    assert str(out.random_draw_round.dtype) == "Int64"


def test_malformed_sheet_row_stops_extraction(tmp_path):
    p = tmp_path / "labels.csv"
    p.write_text(
        "con_legis_num,manual_coding,notes\n"
        "101_hr.1,1,ok\n"
        "101_hr.2,1,hard case, maybe\n"
        "101_hr.3,0,fine\n"
    )
    with pytest.raises(Exception) as exc:
        validate_draw_labels(p)
    assert "line 3" in str(exc.value) or "Expected 3 fields" in str(exc.value)


def test_sheet_row_count_must_match_unlabeled_draw_rows(tmp_path):
    draw = tmp_path / "draw.csv"
    draw.write_text(
        "con_legis_num,congress,round,already_labeled,seed\n"
        "101_hr.1,101,1,False,1\n101_hr.2,101,1,False,1\n101_hr.3,101,2,True,1\n"
    )
    sheet = tmp_path / "labels.csv"
    sheet.write_text("con_legis_num,manual_coding\n101_hr.1,1\n")
    with pytest.raises(ValueError, match="1 rows.*2"):
        validate_draw_labels(sheet, draw_path=draw)
    sheet.write_text("con_legis_num,manual_coding\n101_hr.1,1\n101_hr.2,\n")
    validate_draw_labels(sheet, draw_path=draw)  # matches: no error


def test_unlabeled_draw_by_congress_and_round():
    from congress_nlp.features.extract import unlabeled_draw_by_round

    draw = pd.DataFrame(
        {
            "con_legis_num": ["101_hr.1", "101_hr.2", "102_hr.1"],
            "congress": [101, 101, 102],
            "round": [1, 2, 2],
        }
    )
    df = pd.DataFrame({"con_legis_num": ["101_h.r.1"]})
    assert unlabeled_draw_by_round(df, draw) == {(101, 2): 1, (102, 2): 1}
