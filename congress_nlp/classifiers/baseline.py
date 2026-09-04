import argparse

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score

from congress_nlp import paths


# Lower threshold to prioritize recall
PROB_THRESHOLD = 0.3
MAX_FEATURES = 10000
MAX_ITER = 1000

# title+subjects is present on 100% of bills; title+summary ("text") is not.
INPUT_FIELDS = ("text_title_subjects", "text")
DEFAULT_INPUT_FIELD = "text_title_subjects"

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-field", choices=INPUT_FIELDS,
                        default=DEFAULT_INPUT_FIELD,
                        help="Which text column to vectorize.")
    args = parser.parse_args()
    field = args.input_field

    features = pd.read_csv(paths.FEATURES_CSV)
    if field not in features.columns:
        raise KeyError(
            f"features.csv has no column {field!r}. Re-run "
            f"scripts/04_extract_features.py to regenerate it."
        )
    train_df = features[features["split"] == "train"]
    val_df = features[features["split"] == "val"]
    
    X_train = train_df[field].fillna("").to_numpy()
    y_train = train_df["manual_coding"].to_numpy()
    
    X_val = val_df[field].fillna("").to_numpy()
    y_val = val_df["manual_coding"].to_numpy()
    
    tfidf = TfidfVectorizer(max_features=MAX_FEATURES)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_val_tfidf = tfidf.transform(X_val)
    
    clf = LogisticRegression(max_iter=MAX_ITER, class_weight="balanced")
    clf.fit(X_train_tfidf, y_train)
    
    y_prob = clf.predict_proba(X_val_tfidf)[:, 1]
    y_pred = (y_prob >= PROB_THRESHOLD).astype(int)
    
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    
    print(f"RESULTS (input field: {field}):")
    print(f"Precision: {precision:.4f} | target >= 0.75")
    print(f"Recall: {recall:.4f} | target >= 0.90")
    print(f"F1 Score: {f1:.4f} | target >= 0.85")
    
    
if __name__ == "__main__":
    main()