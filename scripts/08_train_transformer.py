"""Fine-tune an encoder and evaluate it through the shared protocols.

    python scripts/08_train_transformer.py --model roberta-base --protocol loco
    python scripts/08_train_transformer.py --model distilbert-base-uncased --protocol loco
    python scripts/08_train_transformer.py --model roberta-base --protocol final_holdout

Every run is 5 seeds. The reported number is mean +/- sd over seeds -- never a
single run, never the best seed. Forward chaining is deliberately not offered:
its last fold is the final holdout, and 119 is spent once.

Hyperparameters are fixed, not tuned. A LOCO fold has no validation set, so
there is nothing inside a fold to tune on; the flags exist so a run can be
reproduced, not so values can be searched.
"""

import argparse

import pandas as pd

from congress_nlp import paths
from congress_nlp.classifiers.transformer import SEEDS, TransformerClassifier
from congress_nlp.evaluation import protocol
from congress_nlp.evaluation.runner import fold_summary, pooled_summary, run_protocol
from congress_nlp.features.load import load_features
from congress_nlp.features.views import DEFAULT_VIEW, VIEWS

MODELS = ("roberta-base", "distilbert-base-uncased")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODELS[0], help=f"HF model id; tested: {MODELS}")
    parser.add_argument("--view", choices=sorted(VIEWS), default=DEFAULT_VIEW)
    parser.add_argument("--protocol", choices=("loco", "final_holdout"), default="loco")
    parser.add_argument("--seeds", type=int, default=len(SEEDS), help="How many of the fixed seeds to run.")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=1, help="Raise to 2 with --batch-size 8 if VRAM is short.")
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()
    if not 1 <= args.seeds <= len(SEEDS):
        parser.error(f"--seeds must be between 1 and {len(SEEDS)}; the seeds are fixed.")

    df = load_features(view=args.view)
    folds = (
        protocol.loco_folds(df)
        if args.protocol == "loco"
        else [protocol.final_holdout(df)]
    )
    tag = args.model.replace("/", "_")
    folds_csv = paths.OUTPUTS_DIR / f"transformer_{tag}_{args.protocol}_folds.csv"
    preds_csv = paths.OUTPUTS_DIR / f"transformer_{tag}_{args.protocol}_predictions.csv"
    folds_csv.unlink(missing_ok=True)

    print(
        f"model={args.model} view={args.view} protocol={args.protocol} "
        f"folds={len(folds)} seeds={args.seeds} batch={args.batch_size}x{args.grad_accum}"
    )

    all_folds, all_preds = [], []
    for seed in SEEDS[: args.seeds]:
        print(f"\n=== seed {seed} ===")
        result = run_protocol(
            df,
            folds,
            lambda s: TransformerClassifier(
                model_name=args.model,
                seed=s,
                epochs=args.epochs,
                lr=args.lr,
                batch_size=args.batch_size,
                grad_accum=args.grad_accum,
                max_length=args.max_length,
            ),
            model=args.model,
            view=args.view,
            seed=seed,
            out_csv=folds_csv,
        )
        all_folds.append(result.folds)
        all_preds.append(result.predictions)
        # Rewrite after every seed so a crash mid-grid keeps the finished seeds.
        pd.concat(all_preds, ignore_index=True).to_csv(preds_csv, index=False)
        for _, r in result.folds.iterrows():
            print(
                f"  fold {r['fold']:>5} n={r['n']:>4} base={r['base_rate']:.4f} "
                f"roc_auc={r['roc_auc']:.4f} pr_auc={r['pr_auc']:.4f} lift={r['lift']:.4f}"
            )
        if len(result.folds) > 1:
            for _, r in fold_summary(result.folds).iterrows():
                print(f"  seed {seed} {r['metric']:<8} mean={r['mean']:.4f} sd(ddof=0)={r['sd']:.4f}")

    folds_df = pd.concat(all_folds, ignore_index=True)
    preds_df = pd.concat(all_preds, ignore_index=True)
    preds_df.to_csv(preds_csv, index=False)

    print(f"\n=== across {args.seeds} seeds, {args.model} ===")
    per_seed = folds_df.groupby("seed")[["roc_auc", "pr_auc", "lift"]].mean()
    for metric in ("roc_auc", "pr_auc", "lift"):
        print(
            f"  {metric:<8} seed-mean={per_seed[metric].mean():.4f} "
            f"seed-sd(ddof=0)={per_seed[metric].std(ddof=0):.4f}"
        )
    print("  A gap smaller than 2 seed-sd between two encoders is NO DIFFERENCE.")

    for seed, group in preds_df.groupby("seed"):
        pooled = pooled_summary(group)
        print(
            f"  pooled seed {seed}: n={pooled['n']} base={pooled['base_rate']:.4f} "
            f"roc_auc={pooled['roc_auc']:.4f} P@R90={pooled['p_at_r90']:.4f} "
            f"lift={pooled['lift']:.4f}"
        )
    print(f"\nfolds -> {folds_csv}\npredictions -> {preds_csv}")


if __name__ == "__main__":
    main()
