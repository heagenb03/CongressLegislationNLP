import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score


# Lower threshold to prioritize recall
PROB_THRESHOLD = 0.3
MAX_FEATURES = 10000
MAX_ITER = 1000

def main() -> None:
    features = pd.read_csv(Path("data/processed/features.csv"))
    train_df = features[features["split"] == "train"]
    val_df = features[features["split"] == "val"]
    
    X_train = train_df["text"].to_numpy()
    y_train = train_df["manual_coding"].to_numpy()
    
    X_val = val_df["text"].to_numpy()
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
    
    print("RESULTS:")
    print(f"Precision: {precision:.4f} | target >= 0.75")
    print(f"Recall: {recall:.4f} | target >= 0.90")
    print(f"F1 Score: {f1:.4f} | target >= 0.85")
    
    
if __name__ == "__main__":
    main()