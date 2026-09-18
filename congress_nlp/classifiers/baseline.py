"""TF-IDF + Logistic Regression baseline.

    python scripts/05_train_baseline.py --view title_subjects
"""

import argparse
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from congress_nlp.evaluation.metrics import report_at_threshold
from congress_nlp.features.load import load_features
from congress_nlp.features.views import DEFAULT_VIEW, VIEWS

# Tuned on the 117th-Congress validation split, which is 81% positive. The 119
# test split is 51% positive and this threshold does not transfer to it -- see
# the spec and .wolf/STATUS.md before reporting a test number.
PROB_THRESHOLD = 0.3
MAX_FEATURES = 10000
MAX_ITER = 1000


@dataclass
class TfidfLogRegScorer:
    """The frozen baseline as a Scorer: TF-IDF + balanced logistic regression.

    Hyperparameters default to the values the acceptance gate was produced
    with. fit() builds a new vectorizer every call, so the same instance can be
    refit on another fold without carrying the previous fold's vocabulary.
    """

    max_features: int = MAX_FEATURES
    max_iter: int = MAX_ITER
    vectorizer: TfidfVectorizer | None = field(default=None, init=False, repr=False)
    model: LogisticRegression | None = field(default=None, init=False, repr=False)

    def fit(self, texts: Sequence[str], labels: np.ndarray) -> None:
        self.vectorizer = TfidfVectorizer(max_features=self.max_features)
        matrix = self.vectorizer.fit_transform(list(texts))
        self.model = LogisticRegression(
            max_iter=self.max_iter, class_weight="balanced"
        )
        self.model.fit(matrix, np.asarray(labels))

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        if self.vectorizer is None or self.model is None:
            raise RuntimeError("fit() must be called before predict_proba().")
        return self.model.predict_proba(self.vectorizer.transform(list(texts)))[:, 1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--view",
        choices=sorted(VIEWS),
        default=DEFAULT_VIEW,
        help="Which text view to vectorize.",
    )
    parser.add_argument("--threshold", type=float, default=PROB_THRESHOLD)
    args = parser.parse_args()

    train_df = load_features(view=args.view, splits=["train"])
    val_df = load_features(view=args.view, splits=["val"])

    tfidf = TfidfVectorizer(max_features=MAX_FEATURES)
    x_train = tfidf.fit_transform(train_df["text"].fillna(""))
    x_val = tfidf.transform(val_df["text"].fillna(""))

    clf = LogisticRegression(max_iter=MAX_ITER, class_weight="balanced")
    clf.fit(x_train, train_df["manual_coding"].to_numpy())

    y_prob = clf.predict_proba(x_val)[:, 1]
    scores = report_at_threshold(val_df["manual_coding"].to_numpy(), y_prob, args.threshold)

    print(f"RESULTS (view: {args.view}, threshold: {args.threshold})")
    print(f"Precision: {scores['precision']:.4f} | target >= 0.75")
    print(f"Recall:    {scores['recall']:.4f} | target >= 0.90")
    print(f"F1 Score:  {scores['f1']:.4f} | target >= 0.85")


if __name__ == "__main__":
    main()
