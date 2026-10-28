"""The temporal train/val/test split.

Owned by no pipeline step so that extraction, training, and evaluation all read
the same table. A temporal split is mandatory here: congressional language about
China shifted around 2017, so a random split leaks vocabulary across eras.
"""


class UnknownCongressError(KeyError):
    """Raised for a congress with no split assignment."""


PARKED = "parked"

# Scope is congresses 101-117. 118 and 119 are parked because CRS has not
# finished processing them (subject terms mature with a multi-year lag), so they
# are excluded from the default feature load. See the complete-congress-scope
# spec under docs/superpowers/specs/.
SPLITS: dict[int, str] = {
    **{congress: "train" for congress in range(93, 116)},
    116: "val",
    117: "test",
    118: PARKED,
    119: PARKED,
}


def assign_split(congress: int) -> str:
    """Return the split label for a congress, raising if it has none."""
    try:
        return SPLITS[congress]
    except KeyError:
        raise UnknownCongressError(
            f"Congress {congress} has no split assignment. Add it to SPLITS in "
            f"congress_nlp/splits.py before extracting features. Do not let a "
            f"new congress fall into an existing split by default."
        ) from None
