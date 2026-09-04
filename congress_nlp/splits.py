"""The temporal train/val/test split.

Owned by no pipeline step so that extraction, training, and evaluation all read
the same table. A temporal split is mandatory here: congressional language about
China shifted around 2017, so a random split leaks vocabulary across eras.
"""


class UnknownCongressError(KeyError):
    """Raised for a congress with no split assignment."""


# 117 is the tuning split; 119 is the primary holdout (51% positive, 239
# negatives). 118 is kept as test_legacy but is 96.9% positive with 6 negatives,
# so its precision is not meaningful.
SPLITS: dict[int, str] = {
    **{congress: "train" for congress in range(93, 117)},
    117: "val",
    118: "test_legacy",
    119: "test",
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
