"""Text views over features.csv.

features.csv stores base columns only; each view rebuilds a model input from
them at load time. Adding an input is one function plus one VIEWS entry, and it
is immediately available to every model script as --view <name>.

Builders are row-wise on purpose. Vectorizing the pipe-splitting with
.str.split() changes empty-segment handling, which silently shifts the text and
breaks reproduction against the frozen baseline.
"""

from collections.abc import Callable

import pandas as pd


def _field(row: pd.Series, key: str) -> str:
    """Read one column as a string, mapping a missing value to ''.

    features.csv is read with pd.read_csv, which turns a blank cell into NaN.
    NaN is truthy, so the obvious `str(row.get(key) or "")` yields the string
    'nan' and appends a literal ' nan' token to the model input. The stored
    columns really do carry blank subjects and blank summary_text cells, so
    this guard is what keeps a rebuilt view byte-identical to the old one.
    """
    value = row.get(key)
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value)


def _title_subjects(row: pd.Series) -> str:
    """official_title + comma-joined CRS subjects.

    The primary training input. official_title is on 100% of bills and subjects
    on 96.6%+ of no-summary bills, so this is the only input whose distribution
    holds across all four splits — CRS writes summaries with a lag, so a current
    Congress has very few.
    """
    title = _field(row, "official_title").strip()
    subjects = _field(row, "subjects")
    terms = [t.strip() for t in subjects.split("|") if t and t.strip()]
    parts = [title] if title else []
    if terms:
        parts.append(", ".join(terms))
    return " ".join(parts)


def _title_summary(row: pd.Series) -> str:
    """official_title + CRS summary. The original TF-IDF baseline input."""
    title = _field(row, "official_title").strip()
    summary = _field(row, "summary_text").strip()
    return " ".join(p for p in (title, summary) if p)


def _keyword_prefixed(row: pd.Series) -> str:
    """'[FILTER: prc, pla] ' + title_summary.

    The prefix front-loads the Stage 1 keywords so they survive 512-token
    truncation in a transformer, where a long summary would otherwise push them
    out of the window.
    """
    base = _title_summary(row)
    matched = _field(row, "matched_keywords")
    keywords = [kw.strip() for kw in matched.split("|") if kw.strip()]
    if not keywords:
        return base
    return f"[FILTER: {', '.join(keywords)}] {base}"


VIEWS: dict[str, Callable[[pd.Series], str]] = {
    "title_subjects": _title_subjects,
    "title_summary": _title_summary,
    "keyword_prefixed": _keyword_prefixed,
}

DEFAULT_VIEW = "title_subjects"


def build_view(df: pd.DataFrame, view: str) -> pd.Series:
    """Apply a named view row-wise, returning the text column."""
    try:
        builder = VIEWS[view]
    except KeyError:
        raise KeyError(
            f"Unknown view {view!r}. Valid views: {', '.join(sorted(VIEWS))}."
        ) from None
    if df.empty:
        return pd.Series([], dtype="object")
    return df.apply(builder, axis=1)
