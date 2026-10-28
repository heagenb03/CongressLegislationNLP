"""Per-congress and per-era tables from one seed's per-bill predictions.

Every row carries its base rate. AUC and lift are NaN for a fold with a single
class, and a fold with fewer than MIN_CLASS of either class is `flagged`: its
AUC and lift are reported but are not headline numbers. Confidence intervals
are percentile bootstraps (N_BOOT resamples); eras resample within congress.
Base-rate intervals are Wilson.
"""

import math
from collections.abc import Callable, Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from congress_nlp.evaluation.metrics import lift_at_recall

N_BOOT = 2000
BOOT_SEED = 20261027
MIN_CLASS = 5
FIXED_THRESHOLD = 0.5
ERAS: dict[str, tuple[int, int]] = {"pre-2017": (101, 114), "post-2017": (115, 116)}

_NAN = float("nan")


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion k / n."""
    if n <= 0:
        return (_NAN, _NAN)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    *,
    strata: np.ndarray | None = None,
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
) -> tuple[float, float]:
    """Percentile 95% interval, resampling bills (within strata when given).

    A resample with fewer than MIN_CLASS bills of either class is skipped (this
    includes one-class resamples); (nan, nan) if more than half are. The brief's
    literal "one class" rule never fires on a 59/1 sample (only ~36% of its
    resamples lose the lone negative), so the MIN_CLASS floor is what makes a
    flagged fold's interval NaN rather than a misleadingly tight one.
    """
    y = np.asarray(y_true)
    p = np.asarray(y_prob, dtype=float)
    rng = np.random.default_rng(seed)
    if strata is None:
        groups = [np.arange(len(y))]
    else:
        s = np.asarray(strata)
        groups = [np.flatnonzero(s == v) for v in np.unique(s)]
    values: list[float] = []
    skipped = 0
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups])
        yb = y[idx]
        if min(yb.sum(), len(yb) - yb.sum()) < MIN_CLASS:
            skipped += 1
            continue
        values.append(float(metric(yb, p[idx])))
    if skipped > n_boot / 2 or not values:
        return (_NAN, _NAN)
    lo, hi = np.percentile(values, [2.5, 97.5])
    return (float(lo), float(hi))


def _with_congress(predictions: pd.DataFrame) -> pd.DataFrame:
    df = predictions[["fold", "y_true", "y_prob"]].copy()
    df["congress"] = df["fold"].astype(int)
    return df


def _has_both(y: np.ndarray) -> bool:
    return len(np.unique(y)) > 1


def congress_report(predictions: pd.DataFrame) -> pd.DataFrame:
    """One row per fold: base rate, recall at 0.5, AUC and lift with CIs."""
    df = _with_congress(predictions)
    rows = []
    for congress, g in df.groupby("congress", sort=True):
        y = g["y_true"].to_numpy().astype(int)
        p = g["y_prob"].to_numpy(dtype=float)
        n, n_pos = len(y), int(y.sum())
        n_neg = n - n_pos
        base_lo, base_hi = wilson_interval(n_pos, n)
        row = {
            "congress": int(congress),
            "n": n,
            "n_pos": n_pos,
            "n_neg": n_neg,
            "base_rate": n_pos / n,
            "base_lo": base_lo,
            "base_hi": base_hi,
            "recall_at_0_5": (
                float(((p >= FIXED_THRESHOLD) & (y == 1)).sum() / n_pos) if n_pos else _NAN
            ),
            "roc_auc": _NAN,
            "auc_lo": _NAN,
            "auc_hi": _NAN,
            "lift": _NAN,
            "lift_lo": _NAN,
            "lift_hi": _NAN,
            "flagged": min(n_pos, n_neg) < MIN_CLASS,
        }
        if _has_both(y):
            row["roc_auc"] = float(roc_auc_score(y, p))
            row["auc_lo"], row["auc_hi"] = bootstrap_ci(y, p, roc_auc_score)
            row["lift"] = float(lift_at_recall(y, p))
            row["lift_lo"], row["lift_hi"] = bootstrap_ci(y, p, lift_at_recall)
        rows.append(row)
    return pd.DataFrame(rows)


def era_report(
    predictions: pd.DataFrame,
    eras: Mapping[str, tuple[int, int]] = ERAS,
) -> pd.DataFrame:
    """One row per era: pooled AUC (CI stratified by congress) and the
    unweighted mean of per-congress AUCs over congresses with both classes."""
    df = _with_congress(predictions)
    rows = []
    for era, (lo, hi) in eras.items():
        g = df[(df["congress"] >= lo) & (df["congress"] <= hi)]
        row = {
            "era": era,
            "congresses": f"{lo}-{hi}",
            "n": len(g),
            "base_rate": float(g["y_true"].mean()) if len(g) else _NAN,
            "pooled_auc": _NAN,
            "auc_lo": _NAN,
            "auc_hi": _NAN,
            "mean_congress_auc": _NAN,
        }
        if len(g):
            y = g["y_true"].to_numpy().astype(int)
            p = g["y_prob"].to_numpy(dtype=float)
            if _has_both(y):
                row["pooled_auc"] = float(roc_auc_score(y, p))
                row["auc_lo"], row["auc_hi"] = bootstrap_ci(
                    y, p, roc_auc_score, strata=g["congress"].to_numpy()
                )
            aucs = [
                roc_auc_score(c["y_true"], c["y_prob"])
                for _, c in g.groupby("congress")
                if _has_both(c["y_true"].to_numpy())
            ]
            if aucs:
                row["mean_congress_auc"] = float(np.mean(aucs))
        rows.append(row)
    return pd.DataFrame(rows)
