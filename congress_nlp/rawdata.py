"""The single reader of raw legislation data.json files."""

import json
from pathlib import Path


def read_bill_fields(path: Path) -> tuple[str, str, str, str]:
    """Return (official_title, short_title, summary_text, subjects).

    subjects is pipe-delimited, e.g. 'China|Trade|Arms sales'. Every field
    defaults to an empty string when absent.
    """
    with path.open(encoding="utf-8") as fh:
        data: dict = json.load(fh)

    official_title: str = data.get("official_title") or ""
    short_title: str = data.get("short_title") or ""

    summary = data.get("summary")
    summary_text: str = (summary.get("text") or "") if isinstance(summary, dict) else ""

    subjects: list[str] = data.get("subjects") or []
    subjects_str = "|".join(subjects) if isinstance(subjects, list) else ""

    return official_title, short_title, summary_text, subjects_str
