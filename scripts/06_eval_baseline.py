"""TF-IDF + Logistic Regression through the evaluation protocols.

Every fold is scored on the frozen random draw only (random_draw_round).

The 2026-09-15 acceptance gate (docs/superpowers/specs/2026-09-15-evaluation-baseline.txt)
is retired by this protocol change: the scored rows are different, so its
four-decimal targets no longer apply. A new gate is frozen from this script's
first run on rounds 1 + 2.

    python scripts/06_eval_baseline.py --view title_subjects
    python scripts/06_eval_baseline.py --view title_subjects --final-holdout

Without --final-holdout the script runs LOCO and forward chaining through 116
and does NOT touch the 117 holdout. --final-holdout adds the last forward-chain
fold (test 117) and protocol 3; the 117 holdout is spent once.
"""

import argparse

import numpy as np

from congress_nlp import paths
from congress_nlp.classifiers.baseline import TfidfLogRegScorer
from congress_nlp.evaluation import protocol
from congress_nlp.evaluation.metrics import subgroup_report, summarize, sweep
from congress_nlp.evaluation.reporting import (
    congress_report,
    era_report,
    format_congress_report,
    format_era_report,
    print_incomplete_rounds,
)
from congress_nlp.evaluation.runner import (
    default_subgroups,
    fold_summary,
    pooled_summary,
    run_protocol,
)
from congress_nlp.features.load import load_features
from congress_nlp.features.views import DEFAULT_VIEW, VIEWS

MODEL = "tfidf_logreg"
SEED = 20260721
SWEEP_ROWS = (0.3, 0.4, 0.5, 0.6, 0.7)
DRIFT_CHECKS = (((101, 108), (109, 112)), ((101, 112), (113, 116)))


def _run(df, folds, view, out_csv=None):
    return run_protocol(
        df,
        folds,
        lambda seed: TfidfLogRegScorer(),
        model=MODEL,
        view=view,
        seed=SEED,
        out_csv=out_csv,
    )


def _protocol_1(df, view, out_csv, rounds) -> None:
    lo, hi = protocol.LOCO_LO, protocol.LOCO_HI
    print(f"PROTOCOL 1 - LOCO over congresses {lo}-{hi}, scored rows only, min_rows={protocol.MIN_ROWS}")
    print_incomplete_rounds(df, rounds, lo, hi)
    folds = protocol.loco_folds(df, rounds=rounds)
    kept = {f.label for f in folds}
    scored = df[protocol.scored_mask(df, rounds) & df["congress"].between(lo, hi)]
    counts = scored.groupby("congress").size().reindex(range(lo, hi + 1), fill_value=0)
    for congress, n in counts.items():
        if str(congress) not in kept:
            why = f"n < {protocol.MIN_ROWS}" if n < protocol.MIN_ROWS else "one class"
            print(f"{congress:>5} {n:>5}   SKIPPED ({why})")

    if not folds:
        print("  no scored folds — run 02b_draw_random_sample.py, 04_extract_features.py, and label the draw")
        return
    result = _run(df, folds, view, out_csv)
    print(f"{'cong':>5} {'n':>5} {'base':>7} {'roc_auc':>8} {'pr_auc':>7} {'P@R90':>7} {'lift':>7}")
    for _, r in result.folds.iterrows():
        print(
            f"{r['fold']:>5} {r['n']:>5} {r['base_rate']:>7.4f} {r['roc_auc']:>8.4f} "
            f"{r['pr_auc']:>7.4f} {r['p_at_r90']:>7.4f} {r['lift']:>7.4f}"
        )

    print(f"\n  folds={len(result.folds)}  (unweighted means, sd ddof=0)")
    for _, r in fold_summary(result.folds).iterrows():
        print(f"  {r['metric']:<8} mean={r['mean']:.4f}  sd(ddof=0)={r['sd']:.4f}")

    pooled = pooled_summary(result.predictions)
    print(
        f"  POOLED out-of-fold: n={pooled['n']} base={pooled['base_rate']:.4f} "
        f"roc_auc={pooled['roc_auc']:.4f} pr_auc={pooled['pr_auc']:.4f} "
        f"P@R90={pooled['p_at_r90']:.4f} lift={pooled['lift']:.4f}"
    )

    print("\n  per-congress report (rec@0.5 uses the fixed 0.5 threshold; FLAG = under 5 of a class):")
    print(format_congress_report(congress_report(result.predictions)))
    print("\n  per-era report:")
    print(format_era_report(era_report(result.predictions)))


def _protocol_2(df, view, final_holdout: bool, rounds) -> None:
    end = protocol.LOCO_HI if final_holdout else protocol.LOCO_HI - 1
    print(f"\nPROTOCOL 2 - forward chaining, train <=N, test N+1, last test {end + 1}")
    folds = protocol.forward_chain_folds(df, end=end, rounds=rounds)
    if not folds:
        print("  no scored folds — run 02b_draw_random_sample.py, 04_extract_features.py, and label the draw")
        return
    result = _run(df, folds, view)
    print(f"{'test':>5} {'n_tr':>5} {'n_te':>5} {'base':>7} {'roc_auc':>8} {'pr_auc':>7} {'lift':>7}")
    for _, r in result.folds.iterrows():
        print(
            f"{r['fold']:>5} {r['n_train']:>5} {r['n']:>5} {r['base_rate']:>7.4f} "
            f"{r['roc_auc']:>8.4f} {r['pr_auc']:>7.4f} {r['lift']:>7.4f}"
        )
    congresses = result.folds["fold"].astype(float).to_numpy()
    aucs = result.folds["roc_auc"].to_numpy()
    if len(aucs) >= 2:
        slope = np.polyfit(congresses, aucs, 1)[0]
        print(f"  ROC-AUC trend = {slope:+.5f} per congress over {len(aucs)} points")
    else:
        print("  ROC-AUC trend needs at least 2 folds; skipped")

    print("\n  metadata-matched drift checks (both sides mature):")
    for train_range, test_range in DRIFT_CHECKS:
        fold = protocol.range_holdout(df, train=train_range, test=test_range, rounds=rounds)
        if len(fold.test_idx) == 0 or df.iloc[fold.test_idx]["manual_coding"].nunique() < 2:
            print(f"    train {train_range[0]}-{train_range[1]} -> test {test_range[0]}-{test_range[1]}: no scorable rows; skipped")
            continue
        r = _run(df, [fold], view).folds.iloc[0]
        print(
            f"    train {train_range[0]}-{train_range[1]} -> test "
            f"{test_range[0]}-{test_range[1]}: n={r['n']} base={r['base_rate']:.4f} "
            f"roc_auc={r['roc_auc']:.4f}"
        )


def _protocol_3(df, view) -> None:  # final holdout always scores rounds 1 + 2
    lo, hi, final = protocol.LOCO_LO, protocol.LOCO_HI, protocol.FINAL_TEST_CONGRESS
    print(f"\nPROTOCOL 3 - FINAL HOLDOUT: train {lo}-{hi} -> test {final}  (spend once)")
    print_incomplete_rounds(df, None, final, final)
    fold = protocol.final_holdout(df)
    if len(fold.test_idx) == 0 or df.iloc[fold.test_idx]["manual_coding"].nunique() < 2:
        print("  no scored folds — run 02b_draw_random_sample.py, 04_extract_features.py, and label the draw")
        return
    result = _run(df, [fold], view)
    r = result.folds.iloc[0]
    print(f"  n={r['n']}  base_rate={r['base_rate']:.4f}")
    print(f"  roc_auc={r['roc_auc']:.4f}  pr_auc={r['pr_auc']:.4f}")
    print(f"  P@R>=0.90={r['p_at_r90']:.4f}  lift={r['lift']:.4f}")

    y_true = result.predictions["y_true"].to_numpy()
    y_prob = result.predictions["y_prob"].to_numpy()
    table = sweep(y_true, y_prob)
    print("   thr  pred_pos       P       R      F1")
    for thr in SWEEP_ROWS:
        row = table[table["thr"] == thr].iloc[0]
        print(
            f"   {thr:.1f}  {int(row['pred_pos']):8d}  {row['precision']:.4f}  "
            f"{row['recall']:.4f}  {row['f1']:.4f}"
        )

    print("\n  per-congress report:")
    print(format_congress_report(congress_report(result.predictions)))

    print(f"\n  subgroups on {final} (compare by lift, not by raw precision):")
    groups = subgroup_report(y_true, y_prob, default_subgroups(df.iloc[fold.test_idx]))
    for _, g in groups.iterrows():
        print(
            f"    {g['group']:<16} n={g['n']:>4} base={g['base_rate']:.4f} "
            f"roc_auc={g['roc_auc']:.4f} P@R90={g['p_at_r90']:.4f} lift={g['lift']:.4f}"
        )

    out = paths.OUTPUTS_DIR / f"eval_baseline_{view}_holdout_predictions.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    result.predictions.to_csv(out, index=False)
    print(f"\n  per-bill {final} probabilities written to {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view", choices=sorted(VIEWS), default=DEFAULT_VIEW)
    parser.add_argument(
        "--final-holdout",
        action="store_true",
        help="Also spend the final holdout: forward chain's last fold and protocol 3.",
    )
    parser.add_argument(
        "--rounds",
        choices=("1", "all"),
        default="all",
        help="Score round 1 only (progress runs) or every drawn round.",
    )
    args = parser.parse_args()
    if args.final_holdout and args.rounds != "all":
        parser.error(
            f"{protocol.FINAL_TEST_CONGRESS} is spent once, on rounds 1+2; "
            "--final-holdout needs --rounds all."
        )
    rounds = {1} if args.rounds == "1" else None

    df = load_features(view=args.view)
    folds_csv = paths.OUTPUTS_DIR / f"eval_baseline_{args.view}_folds.csv"
    folds_csv.unlink(missing_ok=True)

    print(f"view={args.view}  rows={len(df)}  model={MODEL}\n")
    _protocol_1(df, args.view, folds_csv, rounds)
    _protocol_2(df, args.view, args.final_holdout, rounds)
    if args.final_holdout:
        _protocol_3(df, args.view)
    else:
        print(
            f"\nThe {protocol.FINAL_TEST_CONGRESS} final holdout was not touched. "
            "To spend it (once): python scripts/06_eval_baseline.py --final-holdout"
        )
    print(f"\nfold rows written to {folds_csv}")


if __name__ == "__main__":
    main()
