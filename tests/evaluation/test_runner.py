import numpy as np
import pandas as pd
import pytest

from congress_nlp.evaluation.protocol import Fold
from congress_nlp.evaluation.runner import (
    default_subgroups,
    fold_summary,
    package_version,
    pooled_summary,
    run_protocol,
)


class CountingScorer:
    """Scores 1.0 for a text containing 'yes', 0.0 otherwise. Counts its fits."""

    instances: list["CountingScorer"] = []

    def __init__(self, seed: int):
        self.seed = seed
        self.fitted = 0
        self.closed = False
        CountingScorer.instances.append(self)

    def fit(self, texts, labels):
        self.fitted += 1

    def predict_proba(self, texts):
        return np.array([0.9 if "yes" in t else 0.1 for t in texts])

    def close(self):
        self.closed = True


def _frame() -> pd.DataFrame:
    rows = []
    for congress in (101, 102):
        for i in range(4):
            positive = i % 2
            rows.append(
                {
                    "con_legis_num": f"{congress}_s.{i}",
                    "congress": congress,
                    "manual_coding": positive,
                    "text": "yes china" if positive else "no",
                    "has_summary": bool(i % 2),
                    "subjects": "China|Trade|Arms|Tibet|Taiwan" if i == 0 else "China",
                }
            )
    return pd.DataFrame(rows)


def _folds(df):
    a = np.flatnonzero((df["congress"] == 101).to_numpy())
    b = np.flatnonzero((df["congress"] == 102).to_numpy())
    return [Fold(a, b, "102"), Fold(b, a, "101")]


def test_run_protocol_returns_one_row_per_fold_with_base_rate_and_versions():
    CountingScorer.instances.clear()
    df = _frame()
    result = run_protocol(
        df, _folds(df), CountingScorer, model="dummy", view="title_subjects", seed=20260721
    )
    assert list(result.folds["fold"]) == ["102", "101"]
    assert list(result.folds["n"]) == [4, 4]
    assert list(result.folds["n_train"]) == [4, 4]
    assert result.folds["base_rate"].tolist() == [0.5, 0.5]
    assert set(result.folds["model"]) == {"dummy"}
    assert set(result.folds["seed"]) == {20260721}
    assert set(result.folds["view"]) == {"title_subjects"}
    assert "transformers" in result.folds.columns
    assert "torch" in result.folds.columns


def test_run_protocol_builds_a_fresh_scorer_per_fold_and_closes_it():
    CountingScorer.instances.clear()
    df = _frame()
    run_protocol(df, _folds(df), CountingScorer, model="dummy", view="v", seed=7)
    assert len(CountingScorer.instances) == 2
    assert [s.fitted for s in CountingScorer.instances] == [1, 1]
    assert all(s.closed for s in CountingScorer.instances)
    assert all(s.seed == 7 for s in CountingScorer.instances)


def test_predictions_carry_the_bill_id_and_probability():
    df = _frame()
    result = run_protocol(df, _folds(df), CountingScorer, model="dummy", view="v", seed=1)
    assert len(result.predictions) == 8
    assert set(result.predictions.columns) >= {
        "fold",
        "con_legis_num",
        "y_true",
        "y_prob",
        "model",
        "seed",
    }
    positive = result.predictions[result.predictions["y_true"] == 1]
    assert (positive["y_prob"] == 0.9).all()


def test_out_csv_is_appended_fold_by_fold(tmp_path):
    df = _frame()
    out = tmp_path / "run.csv"
    run_protocol(
        df, _folds(df), CountingScorer, model="dummy", view="v", seed=1, out_csv=out
    )
    written = pd.read_csv(out)
    assert len(written) == 2
    assert list(written["fold"].astype(str)) == ["102", "101"]


def test_fold_summary_is_unweighted_with_ddof_zero():
    folds = pd.DataFrame({"roc_auc": [0.8, 1.0], "pr_auc": [0.5, 0.7], "lift": [0.1, 0.3]})
    out = fold_summary(folds).set_index("metric")
    assert out.loc["roc_auc", "mean"] == pytest.approx(0.9)
    assert out.loc["roc_auc", "sd"] == pytest.approx(0.1)  # ddof=1 would be 0.1414


def test_pooled_summary_concatenates_predictions():
    preds = pd.DataFrame({"y_true": [1, 1, 0, 0], "y_prob": [0.9, 0.8, 0.2, 0.1]})
    out = pooled_summary(preds)
    assert out["n"] == 4
    assert out["roc_auc"] == pytest.approx(1.0)


def test_default_subgroups_splits_on_summary_and_subject_richness():
    df = _frame()
    groups = default_subgroups(df)
    assert set(groups) == {
        "summary present",
        "summary absent",
        "subjects >=5",
        "subjects <5",
    }
    assert groups["subjects >=5"].sum() == 2
    assert (groups["summary present"] | groups["summary absent"]).all()


def test_package_version_reports_missing_packages_instead_of_raising():
    assert package_version("no-such-package-anywhere") == "not installed"
    assert package_version("pandas") != "not installed"
