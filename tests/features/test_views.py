import pandas as pd
import pytest

from congress_nlp.features.views import DEFAULT_VIEW, VIEWS, build_view


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "official_title": "A bill to restrict exports.",
                "summary_text": "Prohibits certain exports to the PRC.",
                "subjects": "China|Trade|Arms sales",
                "matched_keywords": "prc|china",
            },
            {
                "official_title": "A resolution on Taiwan.",
                "summary_text": "",
                "subjects": "",
                "matched_keywords": "",
            },
        ]
    )


def test_default_view_is_title_subjects():
    assert DEFAULT_VIEW == "title_subjects"
    assert DEFAULT_VIEW in VIEWS


def test_title_subjects_matches_the_pre_restructure_output():
    """Byte-identical to build_title_subjects_text so the baseline reproduces."""
    out = build_view(_frame(), "title_subjects")
    assert out.iloc[0] == "A bill to restrict exports. China, Trade, Arms sales"
    assert out.iloc[1] == "A resolution on Taiwan."


def test_title_subjects_drops_blank_segments_between_pipes():
    df = pd.DataFrame([{"official_title": "T.", "subjects": "China||Trade|"}])
    assert build_view(df, "title_subjects").iloc[0] == "T. China, Trade"


def test_title_subjects_with_no_title_returns_subjects_only():
    """Carried over from test_build_title_subjects_text_handles_missing_parts."""
    df = pd.DataFrame([
        {"official_title": "", "subjects": "China|Trade"},
        {"official_title": "", "subjects": ""},
    ])
    out = build_view(df, "title_subjects")
    assert out.iloc[0] == "China, Trade"
    assert out.iloc[1] == ""


def test_views_treat_nan_as_empty_not_as_the_string_nan():
    """pd.read_csv turns a blank cell into NaN, and NaN is truthy.

    `str(row.get("subjects") or "")` therefore yields 'nan', which would append
    a literal ' nan' token to the model input. features.csv really does contain
    blank subjects and blank summary_text cells, so this is not hypothetical.
    """
    df = pd.DataFrame([{
        "official_title": "A bill.",
        "summary_text": float("nan"),
        "subjects": float("nan"),
        "matched_keywords": float("nan"),
    }])
    for name in VIEWS:
        assert build_view(df, name).iloc[0] == "A bill.", name


def test_title_subjects_never_includes_the_summary():
    """The whole point of this view is being available when the summary is not.

    If someone 'improves' _title_subjects by appending summary_text, the primary
    input silently regains a field that is missing on two thirds of the 119 test
    split. This assertion is what stops that.
    """
    out = build_view(_frame(), "title_subjects")
    assert "Prohibits certain exports" not in out.iloc[0]


def test_title_summary_joins_title_and_summary():
    out = build_view(_frame(), "title_summary")
    assert out.iloc[0] == "A bill to restrict exports. Prohibits certain exports to the PRC."
    assert out.iloc[1] == "A resolution on Taiwan."


def test_keyword_prefixed_puts_keywords_first():
    out = build_view(_frame(), "keyword_prefixed")
    assert out.iloc[0].startswith("[FILTER: prc, china] ")
    assert out.iloc[1] == "A resolution on Taiwan."


def test_every_view_produces_a_non_empty_string_for_a_titled_row():
    df = _frame()
    for name in VIEWS:
        out = build_view(df, name)
        assert out.iloc[0].strip(), name


def test_unknown_view_raises_and_names_the_valid_choices():
    with pytest.raises(KeyError) as exc:
        build_view(_frame(), "nope")
    assert "title_subjects" in str(exc.value)
