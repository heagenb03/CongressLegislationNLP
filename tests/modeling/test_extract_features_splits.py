import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from modeling.extract_features import (
    assign_split,
    dedupe_records,
    load_manifest_keyword_lookup,
    BillRecord,
)


def _rec(cid, congress, coding):
    return BillRecord(
        con_legis_num=cid, congress=congress, official_title="t", short_title="",
        summary_text="s", subjects="", manual_coding=coding, split=assign_split(congress),
        text="t s", text_title_subjects="t", text_with_keywords="t s", has_summary=True,
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
    from modeling.extract_features import resolve_intern_files, INTERN_DIR, INTERN_FILENAMES
    d = tmp_path / INTERN_DIR
    d.mkdir(parents=True)
    for name in INTERN_FILENAMES:
        (d / name).write_text("con_legis_num,manual_coding\n119_hr.1,1\n", encoding="utf-8")

    found = resolve_intern_files(tmp_path)
    assert [p.name for p in found] == list(INTERN_FILENAMES)


def test_resolve_intern_files_allows_partial(tmp_path):
    """One file present is a legitimate partial run — warn, don't fail."""
    from modeling.extract_features import resolve_intern_files, INTERN_DIR, INTERN_FILENAMES
    d = tmp_path / INTERN_DIR
    d.mkdir(parents=True)
    (d / INTERN_FILENAMES[0]).write_text(
        "con_legis_num,manual_coding\n119_hr.1,1\n", encoding="utf-8")

    found = resolve_intern_files(tmp_path)
    assert [p.name for p in found] == [INTERN_FILENAMES[0]]


def test_resolve_intern_files_raises_when_none_found(tmp_path):
    """The whole intern set vanishing must NOT be swallowed by an exists() guard."""
    import pytest
    from modeling.extract_features import resolve_intern_files, INTERN_DIR
    with pytest.raises(FileNotFoundError) as exc:
        resolve_intern_files(tmp_path)
    assert INTERN_DIR in str(exc.value)


def test_intern_dir_points_at_summer2026_folder():
    from modeling.extract_features import INTERN_DIR
    assert INTERN_DIR == "data/raw/Summer2026InternsData"


# --- title + subjects input field -------------------------------------------

def test_build_title_subjects_text_joins_pipe_delimited_subjects():
    from modeling.extract_features import build_title_subjects_text
    out = build_title_subjects_text(
        "A bill to restrict exports.", "China|Trade|Arms sales")
    assert out == "A bill to restrict exports. China, Trade, Arms sales"


def test_build_title_subjects_text_handles_missing_parts():
    from modeling.extract_features import build_title_subjects_text
    assert build_title_subjects_text("Just a title.", "") == "Just a title."
    assert build_title_subjects_text("", "China|Trade") == "China, Trade"
    assert build_title_subjects_text("", "") == ""
    # Blank segments between pipes must not leave stray commas.
    assert build_title_subjects_text("T.", "China||Trade|") == "T. China, Trade"


def test_build_title_subjects_text_never_includes_summary():
    """The whole point of this field is that it is available when summary isn't."""
    from modeling.extract_features import build_title_subjects_text
    out = build_title_subjects_text("Title.", "China")
    assert "summary" not in out.lower()


def test_billrecord_carries_text_title_subjects():
    from modeling.extract_features import BillRecord
    assert "text_title_subjects" in BillRecord._fields
    # The existing baseline field must survive unchanged.
    assert "text" in BillRecord._fields
