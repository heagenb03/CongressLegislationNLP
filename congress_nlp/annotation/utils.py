"""Pure, side-effect-free helpers shared by the annotation scripts.

No file or network I/O here — everything is unit-tested.
"""
from __future__ import annotations

from congress_nlp.ids import AMENDMENT_TYPES

# Compact legislation type -> chamber
_CHAMBER = {"h": "House", "s": "Senate"}

# Compact type -> human display prefix
_BILL_DISPLAY = {
    "hr": "H.R.", "s": "S.",
    "hres": "H.Res.", "sres": "S.Res.",
    "hconres": "H.Con.Res.", "sconres": "S.Con.Res.",
    "hjres": "H.J.Res.", "sjres": "S.J.Res.",
}

# Compact type -> congress.gov URL path segment
_URL_TYPE = {
    "hr": "house-bill", "s": "senate-bill",
    "hres": "house-resolution", "sres": "senate-resolution",
    "hconres": "house-concurrent-resolution", "sconres": "senate-concurrent-resolution",
    "hjres": "house-joint-resolution", "sjres": "senate-joint-resolution",
}

def ordinal(n: int) -> str:
    """Return the English ordinal string for n (e.g. 101 -> '101st')."""
    if 11 <= (n % 100) <= 13:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def chamber_from_type(legislation_type: str) -> str:
    """Map a compact legislation type to 'House' or 'Senate'."""
    return _CHAMBER.get(legislation_type[:1].lower(), "Unknown")


def bill_number_display(legislation_type: str, number: str) -> str:
    """Human-readable bill label, e.g. ('hr','1153') -> 'H.R. 1153'."""
    prefix = _BILL_DISPLAY.get(legislation_type.lower(), legislation_type.upper())
    return f"{prefix} {number}"


def congress_gov_url(congress: int, legislation_type: str, number: str) -> str:
    """Construct the canonical congress.gov URL for a bill or resolution."""
    seg = _URL_TYPE.get(legislation_type.lower(), "bill")
    return f"https://www.congress.gov/bill/{ordinal(congress)}-congress/{seg}/{number}"


def keyword_count_and_strong(matched_keywords: str, strong: frozenset[str]) -> tuple[int, bool]:
    """Given a pipe-delimited keyword string, return (count, any_strong)."""
    if not matched_keywords or not matched_keywords.strip():
        return 0, False
    kws = [k.strip().lower() for k in matched_keywords.split("|") if k.strip()]
    return len(kws), any(k in strong for k in kws)


def is_amendment_type(legislation_type: str) -> bool:
    """True for amendment types (samdt/hamdt), which interns never label."""
    return legislation_type.lower() in AMENDMENT_TYPES
