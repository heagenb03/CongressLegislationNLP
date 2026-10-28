"""Fold generation for the three evaluation protocols.

This module knows no model and computes no metric. It returns positional index
pairs, so every protocol is testable without fitting anything.

The protocols and what each one licenses are in evaluation/CLAUDE.md. Two rules
live here. Training indices are every labeled row of the training congresses;
test indices are only the rows with a random_draw_round, so every fold is scored
on the frozen random draw. And loco_folds stops at hi=116, so LOCO cannot touch
the 117 holdout by accident. forward_chain_folds is the exception -- its last
fold tests 117 and is identical to final_holdout, which is why only the TF-IDF
baseline runs that protocol.
"""

from typing import NamedTuple

import numpy as np
import pandas as pd

CONGRESS_COL = "congress"
LABEL_COL = "manual_coding"
SCORE_COL = "random_draw_round"

LOCO_LO = 101
LOCO_HI = 116
MIN_ROWS = 25
FINAL_TEST_CONGRESS = 117


class Fold(NamedTuple):
    """One train/test partition, as positional indices for df.iloc.

    label is a string rather than an int because range_holdout produces folds
    spanning several congresses ("101-108 -> 109-112").
    """

    train_idx: np.ndarray
    test_idx: np.ndarray
    label: str


def _where(mask: pd.Series) -> np.ndarray:
    return np.flatnonzero(mask.to_numpy())


def scored_mask(df: pd.DataFrame) -> pd.Series:
    """True for rows in the random draw (random_draw_round not null)."""
    if SCORE_COL not in df.columns:
        raise KeyError(
            f"{SCORE_COL!r} column missing from features; "
            "re-run scripts/04_extract_features.py"
        )
    return df[SCORE_COL].notna()


def _both_classes(df: pd.DataFrame, mask: pd.Series, label_col: str) -> bool:
    return df.loc[mask, label_col].nunique() > 1


def loco_folds(
    df: pd.DataFrame,
    lo: int = LOCO_LO,
    hi: int = LOCO_HI,
    min_rows: int = MIN_ROWS,
    label_col: str = LABEL_COL,
) -> list[Fold]:
    """Leave one congress out, over lo..hi inclusive.

    Test rows are the congress's drawn rows. A congress with fewer than
    min_rows drawn bills is not a fold, but its rows stay in every other fold's training set -- dropping them would change
    the training data rather than the evaluation.
    """
    scored = scored_mask(df)
    in_range = df[CONGRESS_COL].between(lo, hi)
    folds: list[Fold] = []
    for congress in sorted(df.loc[in_range, CONGRESS_COL].unique()):
        other = in_range & (df[CONGRESS_COL] != congress)
        held_out = in_range & (df[CONGRESS_COL] == congress) & scored
        if int(held_out.sum()) < min_rows:
            continue
        if not _both_classes(df, held_out, label_col):
            continue
        folds.append(Fold(_where(other), _where(held_out), str(congress)))
    return folds


def forward_chain_folds(
    df: pd.DataFrame,
    start: int = 107,
    end: int = LOCO_HI,
    min_rows: int = MIN_ROWS,
    label_col: str = LABEL_COL,
) -> list[Fold]:
    """Train on every congress <= N, test on N + 1, for N in start..end.

    Tests score drawn rows only. With end=116 the last fold tests 117 and is
    identical to final_holdout. That is deliberate: the frozen gate contains
    that row. Only the TF-IDF
    baseline runs this protocol -- see evaluation/CLAUDE.md.
    """
    scored = scored_mask(df)
    folds: list[Fold] = []
    for n in range(start, end + 1):
        test_mask = (df[CONGRESS_COL] == n + 1) & scored
        if int(test_mask.sum()) < min_rows:
            continue
        if not _both_classes(df, test_mask, label_col):
            continue
        folds.append(Fold(_where(df[CONGRESS_COL] <= n), _where(test_mask), str(n + 1)))
    return folds


def final_holdout(df: pd.DataFrame) -> Fold:
    """Train 101-116, score 117's drawn rows. Spend once; never tune on the result."""
    train = df[CONGRESS_COL].between(LOCO_LO, LOCO_HI)
    test = (df[CONGRESS_COL] == FINAL_TEST_CONGRESS) & scored_mask(df)
    return Fold(_where(train), _where(test), str(FINAL_TEST_CONGRESS))


def range_holdout(
    df: pd.DataFrame,
    train: tuple[int, int],
    test: tuple[int, int],
) -> Fold:
    """An arbitrary congress span against another.

    Used for the metadata-matched drift checks, where both sides must have
    mature CRS subject terms and summaries. Test rows are drawn rows only.
    """
    train_mask = df[CONGRESS_COL].between(*train)
    test_mask = df[CONGRESS_COL].between(*test) & scored_mask(df)
    label = f"{train[0]}-{train[1]} -> {test[0]}-{test[1]}"
    return Fold(_where(train_mask), _where(test_mask), label)
