"""The single entry point for reading features.csv into a model."""

from collections.abc import Sequence

import pandas as pd

from congress_nlp import paths
from congress_nlp.features.views import DEFAULT_VIEW, build_view
from congress_nlp.splits import PARKED


def load_features(
    view: str = DEFAULT_VIEW,
    splits: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Read features.csv and attach a `text` column built by the named view.

    splits filters to the given split labels; None returns every row except the
    parked ones, which load only when named explicitly.
    """
    df = pd.read_csv(paths.FEATURES_CSV)
    if splits is None:
        df = df[df["split"] != PARKED].copy()
    else:
        df = df[df["split"].isin(splits)].copy()
    df["text"] = build_view(df, view)
    return df
