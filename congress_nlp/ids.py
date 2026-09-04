"""Bill identifier parsing, normalizing, and path derivation.

Consolidates five implementations that had four different contracts. The
differences are load-bearing, so each function's amendment behavior is stated
explicitly and covered by tests/test_ids.py.

Canonical form is the TWL con_legis_num convention: {congress}_{dot_type}.{number},
e.g. '101_s.1151', '102_s.con.res.107', '108_s.amdt.1'.
"""

import re
from pathlib import Path

# Compact directory type -> dotted form used in a legislation_id
_TYPE_TO_DOT: dict[str, str] = {
    "s": "s",
    "hr": "h.r",
    "sres": "s.res",
    "hres": "h.res",
    "sconres": "s.con.res",
    "hconres": "h.con.res",
    "sjres": "s.j.res",
    "hjres": "h.j.res",
    "samdt": "s.amdt",
    "hamdt": "h.amdt",
}

# Dotted form -> compact directory type. Doubles as the validity check: a type
# absent from this table is not something the raw data tree contains.
_DOT_TO_TYPE: dict[str, str] = {dot: compact for compact, dot in _TYPE_TO_DOT.items()}

# Public: annotation.utils.is_amendment_type reads this too, so the set of
# amendment types is defined once on the leaf rather than in two places.
AMENDMENT_TYPES: frozenset[str] = frozenset({"samdt", "hamdt"})


def _split(con_legis_num: str) -> tuple[str, list[str]] | None:
    """Split an ID into (congress, dot tokens). None if it does not parse."""
    raw = str(con_legis_num).strip()
    congress, sep, rest = raw.partition("_")
    if not sep:
        return None
    tokens = rest.lower().split(".")
    if len(tokens) < 2:
        return None
    return congress, tokens


def normalize_id_key(con_legis_num: str) -> str:
    """Canonical dedupe key: '{congress}_{typecompact}_{number}'.

    Collapses dot-notation differences so '118_h.r.1153' and '118_hr.1153' map to
    the same key. Returns the input unchanged when it cannot be parsed — this is
    a dedupe key, so an unparseable ID must still be distinguishable from others.
    Amendments are not special-cased.
    """
    parsed = _split(con_legis_num)
    if parsed is None:
        return str(con_legis_num).strip()
    congress, tokens = parsed
    number = tokens[-1]
    type_compact = "".join(tokens[:-1])
    try:
        number = str(int(number))
    except ValueError:
        pass
    return f"{congress}_{type_compact}_{number}"


def to_canonical_id(con_legis_num: str) -> str | None:
    """Convert a TWL con_legis_num to the pipeline's dotted legislation_id.

    Leading zeros in the number are stripped. Returns None for an unparseable
    ID, a non-integer number, or a type the raw data tree does not contain.
    Amendments are valid: '108_s.amdt.1797' returns '108_s.amdt.1797'.
    """
    parsed = _split(con_legis_num)
    if parsed is None:
        return None
    congress, tokens = parsed
    try:
        number = str(int(tokens[-1]))
    except ValueError:
        return None
    dot_type = ".".join(tokens[:-1])
    if dot_type not in _DOT_TO_TYPE:
        return None
    return f"{congress}_{dot_type}.{number}"


def to_json_path(con_legis_num: str, raw_root: Path) -> Path | None:
    """Derive the raw data.json path for a bill or amendment.

    Routes amendments to '{congress}/amendments/' and everything else to
    '{congress}/bills/'. Returns None for an unrecognized or unparseable ID.
    Callers that must exclude amendments use is_amendment() rather than relying
    on a None return.
    """
    parsed = _split(con_legis_num)
    if parsed is None:
        return None
    congress, tokens = parsed
    number = tokens[-1]
    dot_type = ".".join(tokens[:-1])
    compact = _DOT_TO_TYPE.get(dot_type)
    if compact is None:
        return None
    subdir = "amendments" if compact in AMENDMENT_TYPES else "bills"
    return raw_root / congress / subdir / compact / f"{compact}{number}" / "data.json"


def is_amendment(con_legis_num: str) -> bool:
    """True for s.amdt / h.amdt IDs. False for anything that does not parse."""
    parsed = _split(con_legis_num)
    if parsed is None:
        return False
    _, tokens = parsed
    return "".join(tokens[:-1]) in AMENDMENT_TYPES


def make_legislation_id(congress: int, legislation_type: str, legislation_number: str) -> str:
    """Build a legislation_id from directory metadata.

    Does not parse an existing ID and does not read the JSON bill_id field.
    Amendment directories carry a type prefix ('samdt1'), so only the trailing
    digits become the number.
    """
    dot_type = _TYPE_TO_DOT.get(legislation_type, legislation_type)
    match = re.search(r"(\d+)$", legislation_number)
    number = match.group(1) if match else legislation_number
    return f"{congress}_{dot_type}.{number}"
