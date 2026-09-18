import numpy as np
import pandas as pd
import pytest

from congress_nlp.evaluation.metrics import (
    lift_at_recall,
    precision_at_recall,
    subgroup_report,
    summarize,
    sweep,
)


def _brute_force_precision_at_recall(y, p, target=0.90):
    """The reference implementation the frozen gate was produced by."""
    best = 0.0
    for thr in np.unique(p):
        pred = p >= thr
        tp = int((pred & (y == 1)).sum())
        fp = int((pred & (y == 0)).sum())
        fn = int((~pred & (y == 1)).sum())
        if tp + fn == 0 or tp + fp == 0:
            continue
        if tp / (tp + fn) >= target:
            best = max(best, tp / (tp + fp))
    return best


def test_precision_at_recall_on_a_hand_computed_case():
    # 4 positives, 4 negatives. At thr=0.5: TP=4, FP=1 -> R=1.00, P=0.80.
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    p = np.array([0.9, 0.8, 0.7, 0.5, 0.6, 0.4, 0.3, 0.2])
    assert precision_at_recall(y, p, target=0.90) == pytest.approx(0.8)


def test_precision_at_recall_matches_the_brute_force_reference():
    rng = np.random.default_rng(20260721)
    for _ in range(50):
        n = int(rng.integers(20, 200))
        y = rng.integers(0, 2, n)
        p = rng.random(n)
        assert precision_at_recall(y, p) == pytest.approx(
            _brute_force_precision_at_recall(y, p)
        )


def test_precision_at_recall_falls_to_the_threshold_that_catches_every_positive():
    """The one positive is ranked last, so only thr=0.1 reaches recall 0.99.

    At that threshold everything is predicted positive: P = 1/4. The assertion
    is 0.25, not 0.0 -- a target that nothing reaches is a different case, and
    it is covered by the no-positives test below.
    """
    y = np.array([1, 0, 0, 0])
    p = np.array([0.1, 0.9, 0.8, 0.7])
    assert precision_at_recall(y, p, target=0.99) == pytest.approx(0.25)


def test_precision_at_recall_is_zero_with_no_positives():
    y = np.zeros(5, dtype=int)
    p = np.linspace(0.1, 0.9, 5)
    assert precision_at_recall(y, p) == 0.0


def test_lift_subtracts_the_base_rate():
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    p = np.array([0.9, 0.8, 0.7, 0.5, 0.6, 0.4, 0.3, 0.2])
    assert lift_at_recall(y, p) == pytest.approx(0.8 - 0.5)


def test_sweep_thresholds_are_exact_hundredths():
    y = np.array([1, 0, 1, 0])
    p = np.array([0.31, 0.29, 0.71, 0.11])
    out = sweep(y, p, step=0.01)
    assert out["thr"].iloc[29] == 0.30
    assert 0.3 in set(out["thr"])
    assert 0.7 in set(out["thr"])
    assert len(out) == 99


def test_sweep_counts_predictions_at_or_above_the_threshold():
    y = np.array([1, 1, 0, 0])
    p = np.array([0.9, 0.6, 0.6, 0.1])
    row = sweep(y, p).set_index("thr").loc[0.6]
    assert row["pred_pos"] == 3
    assert row["precision"] == pytest.approx(2 / 3)
    assert row["recall"] == pytest.approx(1.0)


def test_summarize_always_carries_the_base_rate_next_to_precision():
    y = np.array([1, 1, 0, 0])
    p = np.array([0.9, 0.8, 0.2, 0.1])
    out = summarize(y, p)
    assert set(out) == {"n", "base_rate", "roc_auc", "pr_auc", "p_at_r90", "lift"}
    assert out["n"] == 4
    assert out["base_rate"] == pytest.approx(0.5)
    assert out["roc_auc"] == pytest.approx(1.0)
    assert out["lift"] == pytest.approx(out["p_at_r90"] - out["base_rate"])


def test_summarize_returns_nan_auc_for_a_single_class_split():
    y = np.ones(4, dtype=int)
    p = np.array([0.9, 0.8, 0.2, 0.1])
    out = summarize(y, p)
    assert np.isnan(out["roc_auc"])
    assert np.isnan(out["pr_auc"])


def test_subgroup_report_one_row_per_group():
    y = np.array([1, 1, 0, 0])
    p = np.array([0.9, 0.8, 0.2, 0.1])
    mask = np.array([True, True, False, False])
    out = subgroup_report(y, p, {"present": mask, "absent": ~mask})
    assert list(out["group"]) == ["present", "absent"]
    assert list(out["n"]) == [2, 2]
    assert "base_rate" in out.columns
