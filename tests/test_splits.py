import pytest

from congress_nlp.splits import SPLITS, UnknownCongressError, assign_split


def test_known_congresses_map_to_their_splits():
    assert assign_split(101) == "train"
    assert assign_split(116) == "train"
    assert assign_split(117) == "val"
    assert assign_split(118) == "test_legacy"
    assert assign_split(119) == "test"


def test_unknown_congress_raises_instead_of_defaulting():
    """A new congress must force a decision, never silently join the test split."""
    with pytest.raises(UnknownCongressError) as exc:
        assign_split(120)
    assert "120" in str(exc.value)
    assert "congress_nlp/splits.py" in str(exc.value)


def test_split_table_covers_the_full_labeled_range():
    for congress in range(101, 120):
        assert congress in SPLITS
