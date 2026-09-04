from pathlib import Path

from congress_nlp.ids import (
    is_amendment,
    make_legislation_id,
    normalize_id_key,
    to_canonical_id,
    to_json_path,
)

RAW = Path("/raw")


# --- normalize_id_key: dedupe key, never None ------------------------------

def test_normalize_id_key_collapses_dot_variants():
    assert normalize_id_key("118_h.r.1153") == normalize_id_key("118_hr.1153")
    assert normalize_id_key("118_h.r.1153") == "118_hr_1153"
    assert normalize_id_key("102_s.con.res.107") == "102_sconres_107"


def test_normalize_id_key_strips_leading_zeros():
    assert normalize_id_key("101_s.0007") == "101_s_7"


def test_normalize_id_key_returns_input_unchanged_when_unparseable():
    assert normalize_id_key("garbage") == "garbage"


def test_normalize_id_key_handles_amendments():
    """No amendment special-casing here — it is a dedupe key, not a filter."""
    assert normalize_id_key("108_s.amdt.1797") == "108_samdt_1797"


# --- to_canonical_id: dotted form, None on invalid -------------------------

def test_to_canonical_id_round_trips_bill_forms():
    assert to_canonical_id("101_s.1151") == "101_s.1151"
    assert to_canonical_id("102_s.con.res.107") == "102_s.con.res.107"
    assert to_canonical_id("102_s.j.res.153") == "102_s.j.res.153"


def test_to_canonical_id_accepts_amendments():
    """The old root CLAUDE.md claimed None here. It was wrong: samdt and hamdt
    are both in the valid type set."""
    assert to_canonical_id("108_s.amdt.1797") == "108_s.amdt.1797"
    assert to_canonical_id("108_h.amdt.55") == "108_h.amdt.55"


def test_to_canonical_id_strips_leading_zeros():
    assert to_canonical_id("101_s.0007") == "101_s.7"


def test_to_canonical_id_rejects_bad_input():
    assert to_canonical_id("nounderscore") is None
    assert to_canonical_id("101_s") is None
    assert to_canonical_id("101_s.notanumber") is None
    assert to_canonical_id("101_zz.5") is None


# --- to_json_path: routes bills and amendments to different subdirs --------

def test_to_json_path_builds_bill_paths():
    assert to_json_path("116_s.4604", RAW) == RAW / "116" / "bills" / "s" / "s4604" / "data.json"
    assert to_json_path("102_s.con.res.107", RAW) == (
        RAW / "102" / "bills" / "sconres" / "sconres107" / "data.json"
    )


def test_to_json_path_routes_amendments_to_the_amendments_dir():
    """The discarded extract_features implementation hardcoded bills/ and had no
    amendment support; the adopted one routes correctly."""
    assert to_json_path("108_s.amdt.1797", RAW) == (
        RAW / "108" / "amendments" / "samdt" / "samdt1797" / "data.json"
    )


def test_to_json_path_returns_none_for_unrecognized_types():
    assert to_json_path("101_zz.5", RAW) is None
    assert to_json_path("nounderscore", RAW) is None
    assert to_json_path("101_s", RAW) is None


# --- is_amendment ----------------------------------------------------------

def test_is_amendment():
    assert is_amendment("108_s.amdt.1797")
    assert is_amendment("108_h.amdt.55")
    assert not is_amendment("101_s.1151")
    assert not is_amendment("garbage")


# --- make_legislation_id: builds from directory metadata -------------------

def test_make_legislation_id_uses_dot_notation():
    assert make_legislation_id(101, "s", "1151") == "101_s.1151"
    assert make_legislation_id(102, "sconres", "107") == "102_s.con.res.107"


def test_make_legislation_id_strips_amendment_dir_prefix():
    """Amendment directories are named like 'samdt1' — only the digits count."""
    assert make_legislation_id(108, "samdt", "samdt1") == "108_s.amdt.1"
