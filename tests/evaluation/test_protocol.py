import numpy as np
import pandas as pd
import pytest

from congress_nlp.evaluation.protocol import (
    Fold,
    final_holdout,
    forward_chain_folds,
    loco_folds,
    range_holdout,
    scored_mask,
)


@pytest.fixture
def df() -> pd.DataFrame:
    """Five scored congresses plus parked 118.

    103 has too few drawn rows to be a fold but must still train. The 118 rows
    stand in for parked rows reaching protocol code.
    """
    rows = []
    for congress, n, drawn in (
        (101, 40, 30),
        (102, 40, 30),
        (103, 40, 3),
        (116, 40, 30),
        (117, 40, 30),
        (118, 10, 10),
    ):
        for i in range(n):
            rows.append(
                {
                    "congress": congress,
                    "manual_coding": i % 2,
                    "random_draw_round": 1.0 if i < drawn else np.nan,
                }
            )
    return pd.DataFrame(rows)


def test_loco_test_rows_are_the_drawn_rows_of_that_congress(df):
    fold = next(f for f in loco_folds(df) if f.label == "116")
    test = df.iloc[fold.test_idx]
    assert set(test.congress) == {116} and len(test) == 30
    assert test.random_draw_round.notna().all()


def test_loco_trains_on_every_labeled_row_of_the_other_congresses(df):
    fold = next(f for f in loco_folds(df) if f.label == "102")
    train = df.iloc[fold.train_idx]
    assert set(train.congress) == {101, 103, 116}
    assert len(train) == 120  # drawn and undrawn rows alike


def test_loco_skips_a_congress_with_too_few_drawn_rows_but_trains_on_it(df):
    assert [f.label for f in loco_folds(df)] == ["101", "102", "116"]


def test_loco_never_touches_117_or_parked_congresses(df):
    for fold in loco_folds(df):
        seen = set(df.iloc[np.r_[fold.train_idx, fold.test_idx]].congress)
        assert seen.isdisjoint({117, 118, 119})


def test_loco_skips_a_congress_whose_draw_has_one_class(df):
    df.loc[(df.congress == 116) & df.random_draw_round.notna(), "manual_coding"] = 1
    assert "116" not in [f.label for f in loco_folds(df)]


def test_final_holdout_trains_on_101_to_116_and_scores_117_draw(df):
    fold = final_holdout(df)
    assert fold.label == "117"
    assert set(df.iloc[fold.train_idx].congress) == {101, 102, 103, 116}
    assert len(fold.test_idx) == 30


def test_forward_chain_last_fold_is_the_final_holdout(df):
    last = forward_chain_folds(df, start=101)[-1]
    assert last.label == "117"
    assert np.array_equal(last.test_idx, final_holdout(df).test_idx)


def test_range_holdout_scores_drawn_rows_only(df):
    fold = range_holdout(df, train=(101, 102), test=(116, 117))
    assert len(fold.test_idx) == 60


def test_no_drawn_rows_yields_no_folds(df):
    df["random_draw_round"] = np.nan
    assert loco_folds(df) == []


def test_missing_score_column_raises_with_a_fix(df):
    with pytest.raises(KeyError, match="04_extract_features"):
        loco_folds(df.drop(columns="random_draw_round"))


def test_fold_unpacks_as_a_tuple(df):
    train_idx, test_idx, label = final_holdout(df)
    assert isinstance(train_idx, np.ndarray)
    assert label == "117"


# --- rounds filter and forward-chain lower bound ----------------------------


@pytest.fixture
def two_round_df(df) -> pd.DataFrame:
    """Same as df, but each drawn block splits into round 1 (first 20) and 2."""
    out = df.copy()
    pos = out.groupby("congress").cumcount()
    drawn = out["random_draw_round"].notna()
    out.loc[drawn & (pos >= 20), "random_draw_round"] = 2.0
    return out


def test_scored_mask_rounds_none_means_every_drawn_round(two_round_df):
    assert scored_mask(two_round_df).sum() == scored_mask(two_round_df, rounds=None).sum()
    assert scored_mask(two_round_df).sum() == 30 * 4 + 3 + 10


def test_scored_mask_rounds_selects_only_those_rounds(two_round_df):
    mask = scored_mask(two_round_df, rounds={1})
    assert set(two_round_df.loc[mask, "random_draw_round"]) == {1.0}
    assert mask.sum() < scored_mask(two_round_df).sum()


def test_loco_with_rounds_scores_only_round_one_rows(two_round_df):
    folds = loco_folds(two_round_df, min_rows=5, rounds={1})
    assert folds
    for fold in folds:
        assert set(two_round_df.iloc[fold.test_idx].random_draw_round) == {1.0}
        assert len(fold.test_idx) == 20


def test_loco_rounds_default_is_unchanged(two_round_df):
    a = loco_folds(two_round_df)
    b = loco_folds(two_round_df, rounds=None)
    assert [f.label for f in a] == [f.label for f in b]
    assert all(np.array_equal(x.test_idx, y.test_idx) for x, y in zip(a, b))


def test_other_builders_accept_rounds(two_round_df):
    assert len(final_holdout(two_round_df, rounds={1}).test_idx) == 20
    assert len(range_holdout(two_round_df, (101, 102), (116, 117), rounds={2}).test_idx) == 20
    fc = forward_chain_folds(two_round_df, start=101, min_rows=5, rounds={1})
    assert fc and all(len(f.test_idx) == 20 for f in fc)


def test_forward_chain_last_fold_equals_final_holdout_with_pre_101_rows(df):
    extra = pd.DataFrame(
        {"congress": [100] * 40, "manual_coding": [i % 2 for i in range(40)],
         "random_draw_round": [1.0] * 40}
    )
    full = pd.concat([extra, df], ignore_index=True)
    last = forward_chain_folds(full, start=101)[-1]
    fh = final_holdout(full)
    assert np.array_equal(last.train_idx, fh.train_idx)
    assert np.array_equal(last.test_idx, fh.test_idx)
    assert 100 not in set(full.iloc[last.train_idx].congress)
