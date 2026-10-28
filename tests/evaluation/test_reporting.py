import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from congress_nlp.evaluation.reporting import (
    bootstrap_ci,
    congress_report,
    era_report,
    wilson_interval,
)


def _fold(label: str, n_pos: int, n_neg: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y = np.array([1] * n_pos + [0] * n_neg)
    p = np.clip(0.35 * y + rng.uniform(0, 0.65, len(y)), 0, 1)
    return pd.DataFrame({"fold": label, "y_true": y, "y_prob": p})


def _auc(df: pd.DataFrame) -> float:
    return roc_auc_score(df.y_true, df.y_prob)


def test_wilson_matches_a_known_value():
    lo, hi = wilson_interval(30, 60)
    assert lo == pytest.approx(0.3773, abs=1e-4)
    assert hi == pytest.approx(0.6227, abs=1e-4)


def test_bootstrap_ci_brackets_the_point_estimate_and_is_deterministic():
    d = _fold("101", 30, 30)
    y, p = d.y_true.to_numpy(), d.y_prob.to_numpy()
    a = bootstrap_ci(y, p, roc_auc_score)
    b = bootstrap_ci(y, p, roc_auc_score)
    assert a == b and a[0] <= roc_auc_score(y, p) <= a[1]


def test_bootstrap_ci_is_nan_when_one_class_dominates():
    y = np.array([1] * 59 + [0])
    p = np.linspace(0, 1, 60)
    assert all(np.isnan(bootstrap_ci(y, p, roc_auc_score)))


def test_congress_report_flags_a_fold_with_few_negatives():
    preds = pd.concat([_fold("101", 30, 30), _fold("102", 57, 3, seed=1)])
    r = congress_report(preds).set_index("congress")
    assert not r.loc[101, "flagged"] and r.loc[102, "flagged"]
    assert r.loc[101, "base_rate"] == 0.5


def test_recall_uses_the_fixed_threshold():
    preds = pd.DataFrame(
        {"fold": "101", "y_true": [1, 1, 1, 1], "y_prob": [0.4, 0.5, 0.6, 0.9]}
    )
    assert congress_report(preds).recall_at_0_5.iloc[0] == 0.75


def test_one_class_fold_gives_nan_metrics_and_is_flagged():
    preds = pd.DataFrame(
        {"fold": "101", "y_true": [1, 1, 1, 1], "y_prob": [0.4, 0.5, 0.6, 0.9]}
    )
    r = congress_report(preds).iloc[0]
    assert r.flagged and r.base_rate == 1.0
    assert all(np.isnan(r[c]) for c in ["roc_auc", "auc_lo", "lift", "lift_lo", "lift_hi"])


def test_no_positives_gives_nan_recall():
    preds = pd.DataFrame({"fold": "101", "y_true": [0, 0], "y_prob": [0.4, 0.9]})
    assert np.isnan(congress_report(preds).recall_at_0_5.iloc[0])


def test_era_report_pools_by_era_and_reports_unweighted_congress_mean():
    f = {c: _fold(str(c), 30, 30, seed=c) for c in (101, 102, 115, 116)}
    preds = pd.concat(f.values())
    e = era_report(preds, eras={"a": (101, 102), "b": (115, 116)}).set_index("era")
    assert e.loc["a", "n"] == 120
    assert e.loc["a", "mean_congress_auc"] == pytest.approx(
        (_auc(f[101]) + _auc(f[102])) / 2
    )
    assert e.loc["a", "auc_lo"] <= e.loc["a", "pooled_auc"] <= e.loc["a", "auc_hi"]


def test_era_report_one_class_era_is_nan():
    preds = pd.DataFrame({"fold": "101", "y_true": [1, 1], "y_prob": [0.2, 0.9]})
    e = era_report(preds, eras={"a": (101, 101)}).iloc[0]
    assert np.isnan(e.pooled_auc) and np.isnan(e.mean_congress_auc)


# --- incomplete-round warning -------------------------------------------------


def test_unlabeled_draw_counts_per_congress_and_rounds(tmp_path):
    from congress_nlp.evaluation.reporting import unlabeled_draw_counts

    draw = pd.DataFrame(
        {
            "con_legis_num": ["101_hr.1", "101_hr.2", "101_hr.3", "102_hr.1", "102_hr.2"],
            "congress": [101, 101, 101, 102, 102],
            "round": [1, 1, 2, 1, 2],
        }
    )
    # 101_hr.1 labeled (dotted variant), 102_hr.1 labeled; the rest are not.
    df = pd.DataFrame({"con_legis_num": ["101_h.r.1", "102_h.r.1", "105_h.r.9"]})
    assert unlabeled_draw_counts(df, draw, rounds=None, lo=101, hi=117) == {101: 2, 102: 1}
    assert unlabeled_draw_counts(df, draw, rounds={1}, lo=101, hi=117) == {101: 1}
    assert unlabeled_draw_counts(df, draw, rounds=None, lo=102, hi=117) == {102: 1}
