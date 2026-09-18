import numpy as np
import pandas as pd
import pytest

from congress_nlp.evaluation.protocol import (
    Fold,
    final_holdout,
    forward_chain_folds,
    loco_folds,
    range_holdout,
)


def _frame() -> pd.DataFrame:
    """Three congresses of 30 rows, one tiny congress of 3, plus 119.

    The tiny congress stands in for the real 104th: too small to be a fold,
    but its rows must still appear in every other fold's training set.
    """
    rows = []
    for congress, n in ((101, 30), (102, 30), (103, 3), (118, 30), (119, 10)):
        for i in range(n):
            rows.append({"congress": congress, "manual_coding": i % 2})
    return pd.DataFrame(rows)


def test_loco_skips_a_congress_below_min_rows_but_keeps_its_training_rows():
    df = _frame()
    folds = loco_folds(df, lo=101, hi=118, min_rows=25)
    assert [f.label for f in folds] == ["101", "102", "118"]

    held_out_102 = next(f for f in folds if f.label == "102")
    train = df.iloc[held_out_102.train_idx]
    assert set(train["congress"]) == {101, 103, 118}
    assert (train["congress"] == 103).sum() == 3


def test_loco_test_fold_is_exactly_that_congress():
    df = _frame()
    fold = next(f for f in loco_folds(df) if f.label == "118")
    assert set(df.iloc[fold.test_idx]["congress"]) == {118}
    assert len(fold.test_idx) == 30


def test_loco_never_trains_or_tests_on_119():
    df = _frame()
    for fold in loco_folds(df):
        assert 119 not in set(df.iloc[fold.train_idx]["congress"])
        assert 119 not in set(df.iloc[fold.test_idx]["congress"])


def test_forward_chain_trains_on_everything_up_to_n():
    df = _frame()
    folds = forward_chain_folds(df, start=101, end=118, min_rows=25)
    assert [f.label for f in folds] == ["102", "118"]

    to_118 = next(f for f in folds if f.label == "118")
    assert set(df.iloc[to_118.train_idx]["congress"]) == {101, 102, 103}
    assert set(df.iloc[to_118.test_idx]["congress"]) == {118}


def test_forward_chain_skips_a_single_class_test_fold():
    df = _frame()
    df.loc[df["congress"] == 118, "manual_coding"] = 1
    assert [f.label for f in forward_chain_folds(df, start=101, end=118)] == ["102"]


def test_final_holdout_trains_on_101_to_118_and_tests_on_119():
    df = _frame()
    fold = final_holdout(df)
    assert fold.label == "119"
    assert set(df.iloc[fold.train_idx]["congress"]) == {101, 102, 103, 118}
    assert set(df.iloc[fold.test_idx]["congress"]) == {119}


def test_range_holdout_labels_itself_by_both_ranges():
    df = _frame()
    fold = range_holdout(df, train=(101, 102), test=(118, 119))
    assert fold.label == "101-102 -> 118-119"
    assert set(df.iloc[fold.train_idx]["congress"]) == {101, 102}
    assert set(df.iloc[fold.test_idx]["congress"]) == {118, 119}


def test_fold_unpacks_as_a_tuple():
    train_idx, test_idx, label = final_holdout(_frame())
    assert isinstance(train_idx, np.ndarray)
    assert label == "119"
