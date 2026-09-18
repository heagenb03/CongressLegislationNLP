"""Run one scorer over one protocol; return a tidy frame, one row per fold.

Every model in the project goes through this function, which is what makes the
comparison table comparable. A model satisfies Scorer structurally and never
imports this module -- the dependency rule in congress_nlp/CLAUDE.md forbids
classifiers/ from importing evaluation behavior beyond metrics, so the wiring
lives in scripts/06-08.

torch and transformers are read through importlib.metadata, never imported, so
this module works in an environment where neither is installed.
"""

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from congress_nlp.evaluation.metrics import summarize
from congress_nlp.evaluation.protocol import Fold
from congress_nlp.features.views import subject_counts

TEXT_COL = "text"
LABEL_COL = "manual_coding"
ID_COL = "con_legis_num"
RICH_SUBJECTS = 5


class Scorer(Protocol):
    """What every model implements. close() is optional."""

    def fit(self, texts: Sequence[str], labels: np.ndarray) -> None: ...

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray: ...


@dataclass(frozen=True)
class RunResult:
    folds: pd.DataFrame
    predictions: pd.DataFrame


def package_version(name: str) -> str:
    """Installed version of a package, or 'not installed'.

    Recorded on every results row so a number in the poster can be traced back
    to the library versions that produced it.
    """
    try:
        return version(name)
    except PackageNotFoundError:
        return "not installed"


def _append_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = not path.exists()
    pd.DataFrame([row]).to_csv(path, mode="a", header=header, index=False)


def run_protocol(
    df: pd.DataFrame,
    folds: Iterable[Fold],
    make_scorer: Callable[[int], Scorer],
    *,
    model: str,
    view: str,
    seed: int,
    out_csv: Path | str | None = None,
) -> RunResult:
    """Fit a fresh scorer on each fold's training rows and score its test rows.

    make_scorer is a factory, not an instance: a scorer reused across folds
    carries the previous fold's fitted state into the next one.

    Rows are appended to out_csv as each fold finishes, so a run that dies at
    fold 12 of 17 keeps the first 11.
    """
    rows: list[dict] = []
    predictions: list[pd.DataFrame] = []
    out_path = Path(out_csv) if out_csv is not None else None

    for fold in folds:
        train = df.iloc[fold.train_idx]
        test = df.iloc[fold.test_idx]

        scorer = make_scorer(seed)
        scorer.fit(train[TEXT_COL].tolist(), train[LABEL_COL].to_numpy())
        y_prob = np.asarray(scorer.predict_proba(test[TEXT_COL].tolist()), dtype=float)
        close = getattr(scorer, "close", None)
        if close is not None:
            close()

        y_true = test[LABEL_COL].to_numpy()
        row = {
            "fold": fold.label,
            "n_train": int(len(train)),
            **summarize(y_true, y_prob),
            "model": model,
            "view": view,
            "seed": seed,
            "transformers": package_version("transformers"),
            "torch": package_version("torch"),
        }
        rows.append(row)
        if out_path is not None:
            _append_row(out_path, row)

        predictions.append(
            pd.DataFrame(
                {
                    "fold": fold.label,
                    ID_COL: test[ID_COL].to_numpy(),
                    "y_true": y_true,
                    "y_prob": y_prob,
                    "model": model,
                    "seed": seed,
                }
            )
        )

    return RunResult(
        folds=pd.DataFrame(rows),
        predictions=(
            pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
        ),
    )


def fold_summary(
    folds: pd.DataFrame,
    columns: Sequence[str] = ("roc_auc", "pr_auc", "lift"),
) -> pd.DataFrame:
    """Unweighted mean and sd(ddof=0) across folds.

    Unweighted because each congress is one observation of cross-congress
    transfer: a 29-row fold and a 222-row fold count equally. ddof=0 because a
    reader recomputing with ddof=1 gets a different number and will think
    something is broken -- say which one you used.
    """
    return pd.DataFrame(
        {
            "metric": list(columns),
            "mean": [float(folds[c].mean()) for c in columns],
            "sd": [float(folds[c].std(ddof=0)) for c in columns],
        }
    )


def pooled_summary(predictions: pd.DataFrame) -> dict[str, float]:
    """summarize() over every out-of-fold prediction concatenated.

    Reported next to the fold means, never instead of them: the two answer
    different questions, and the pooled figure is dominated by the big folds.
    """
    return summarize(
        predictions["y_true"].to_numpy(), predictions["y_prob"].to_numpy()
    )


def default_subgroups(test_df: pd.DataFrame) -> dict[str, np.ndarray]:
    """The summary-present/absent and rich/thin-subject masks, as booleans."""
    has_summary = test_df["has_summary"].astype(bool).to_numpy()
    rich = (subject_counts(test_df) >= RICH_SUBJECTS).to_numpy()
    return {
        "summary present": has_summary,
        "summary absent": ~has_summary,
        "subjects >=5": rich,
        "subjects <5": ~rich,
    }
