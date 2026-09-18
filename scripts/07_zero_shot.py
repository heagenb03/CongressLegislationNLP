"""Zero-shot BART through the evaluation protocols.

    python scripts/07_zero_shot.py --protocol loco --wording both
    python scripts/07_zero_shot.py --protocol final_holdout --wording topical

The wording comparison is LOCO-only. Choosing a wording from a 119 result would
be tuning on the one unbiased precision measurement in the project.
"""

import argparse

from congress_nlp import paths
from congress_nlp.classifiers.zero_shot import MODEL_NAME, WORDINGS, ZeroShotScorer
from congress_nlp.evaluation import protocol
from congress_nlp.evaluation.runner import fold_summary, pooled_summary, run_protocol
from congress_nlp.features.load import load_features
from congress_nlp.features.views import DEFAULT_VIEW, VIEWS

SEED = 20260721


def _report(result, wording: str) -> None:
    print(f"\n=== zero-shot BART, wording={wording!r} ===")
    print(f"{'fold':>5} {'n':>5} {'base':>7} {'roc_auc':>8} {'pr_auc':>7} {'P@R90':>7} {'lift':>7}")
    for _, r in result.folds.iterrows():
        print(
            f"{r['fold']:>5} {r['n']:>5} {r['base_rate']:>7.4f} {r['roc_auc']:>8.4f} "
            f"{r['pr_auc']:>7.4f} {r['p_at_r90']:>7.4f} {r['lift']:>7.4f}"
        )
    if len(result.folds) > 1:
        print(f"  folds={len(result.folds)}  (unweighted means, sd ddof=0)")
        for _, r in fold_summary(result.folds).iterrows():
            print(f"  {r['metric']:<8} mean={r['mean']:.4f}  sd(ddof=0)={r['sd']:.4f}")
        pooled = pooled_summary(result.predictions)
        print(
            f"  POOLED out-of-fold: n={pooled['n']} base={pooled['base_rate']:.4f} "
            f"roc_auc={pooled['roc_auc']:.4f} pr_auc={pooled['pr_auc']:.4f} "
            f"P@R90={pooled['p_at_r90']:.4f} lift={pooled['lift']:.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view", choices=sorted(VIEWS), default=DEFAULT_VIEW)
    parser.add_argument("--protocol", choices=("loco", "final_holdout"), default="loco")
    parser.add_argument("--wording", choices=(*sorted(WORDINGS), "both"), default="both")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    if args.protocol == "final_holdout" and args.wording == "both":
        parser.error(
            "Pick one wording for the final holdout. Comparing wordings on 119 "
            "would be tuning on the split that is spent once."
        )

    df = load_features(view=args.view)
    folds = (
        protocol.loco_folds(df)
        if args.protocol == "loco"
        else [protocol.final_holdout(df)]
    )
    wordings = sorted(WORDINGS) if args.wording == "both" else [args.wording]

    for wording in wordings:
        out_csv = paths.OUTPUTS_DIR / f"zero_shot_{wording}_{args.protocol}_folds.csv"
        out_csv.unlink(missing_ok=True)
        result = run_protocol(
            df,
            folds,
            lambda seed: ZeroShotScorer(
                hypothesis=WORDINGS[wording], batch_size=args.batch_size
            ),
            model=f"{MODEL_NAME}:{wording}",
            view=args.view,
            seed=SEED,
            out_csv=out_csv,
        )
        _report(result, wording)
        preds = paths.OUTPUTS_DIR / f"zero_shot_{wording}_{args.protocol}_predictions.csv"
        result.predictions.to_csv(preds, index=False)
        print(f"  predictions -> {preds}")


if __name__ == "__main__":
    main()
