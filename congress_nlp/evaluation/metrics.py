"""Scoring shared by every classifier.

summarize() returns base_rate in the same dict as p_at_r90 by construction. A
precision figure at a fixed recall is governed by the split's base rate, so a
caller cannot print one of these numbers without having the other in hand.

precision_at_recall scans np.unique(y_prob) rather than using
sklearn.metrics.precision_recall_curve. The two agree in theory; the frozen
acceptance gate in docs/superpowers/specs/ was produced by this scan, and the
gate is checked to four decimals.
"""

from collections.abc import Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

DEFAULT_RECALL_TARGET = 0.90


def report_at_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Precision, recall, and F1 for a probability cutoff."""
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def precision_at_recall(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target: float = DEFAULT_RECALL_TARGET,
) -> float:
    """The best precision reachable while holding recall >= target.

    Returns 0.0 when no threshold reaches the target, or when the split has no
    positives at all.
    """
    y = np.asarray(y_true)
    p = np.asarray(y_prob, dtype=float)
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
    return float(best)


def lift_at_recall(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target: float = DEFAULT_RECALL_TARGET,
) -> float:
    """precision_at_recall minus the base rate.

    At recall 0.90 a random ranker already scores close to the base rate, so
    this is the part of the precision figure that the model earned. It is the
    only precision-like statistic comparable across splits.
    """
    y = np.asarray(y_true)
    return precision_at_recall(y, y_prob, target) - float(y.mean())


def sweep(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    step: float = 0.01,
) -> pd.DataFrame:
    """Precision, recall and F1 at every threshold from step to 1 - step.

    Thresholds are rounded to 10 decimals so that 0.3 is the float 0.3 and not
    0.30000000000000004 -- the comparison is `>=`, and the un-rounded value
    silently drops a prediction whose probability is exactly 0.3.
    """
    y = np.asarray(y_true)
    p = np.asarray(y_prob, dtype=float)
    rows = []
    for i in range(1, int(round(1 / step))):
        thr = round(i * step, 10)
        pred = p >= thr
        tp = int((pred & (y == 1)).sum())
        fp = int((pred & (y == 0)).sum())
        fn = int((~pred & (y == 1)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append(
            {
                "thr": thr,
                "pred_pos": int(pred.sum()),
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return pd.DataFrame(rows)


def summarize(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target: float = DEFAULT_RECALL_TARGET,
) -> dict[str, float]:
    """Every headline number for one split, base rate included.

    roc_auc and pr_auc are NaN for a single-class split, where they are
    undefined rather than zero.
    """
    y = np.asarray(y_true)
    p = np.asarray(y_prob, dtype=float)
    both_classes = len(np.unique(y)) > 1
    p_at_r90 = precision_at_recall(y, p, target)
    base_rate = float(y.mean())
    return {
        "n": int(len(y)),
        "base_rate": base_rate,
        "roc_auc": float(roc_auc_score(y, p)) if both_classes else float("nan"),
        "pr_auc": float(average_precision_score(y, p)) if both_classes else float("nan"),
        "p_at_r90": p_at_r90,
        "lift": p_at_r90 - base_rate,
    }


def subgroup_report(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    by: Mapping[str, np.ndarray],
    target: float = DEFAULT_RECALL_TARGET,
) -> pd.DataFrame:
    """summarize() per boolean mask, one row per group.

    Compare groups by lift, not by raw precision: on the final holdout the
    summary-present / summary-absent precision gap is a base-rate gap, and the
    two groups reverse order once the floor is subtracted.
    """
    y = np.asarray(y_true)
    p = np.asarray(y_prob, dtype=float)
    rows = []
    for name, mask in by.items():
        mask = np.asarray(mask, dtype=bool)
        rows.append({"group": name, **summarize(y[mask], p[mask], target)})
    return pd.DataFrame(rows)
