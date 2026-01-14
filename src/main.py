from pathlib import Path
import re
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


def extract_rating_num(rating_text: str):
    m = re.search(r"(\d)", str(rating_text))
    return float(m.group(1)) if m else np.nan


def main():
    data_path = Path(__file__).resolve().parents[1] / "data" / "Amazon_Reviews_clean.csv"
    df = pd.read_csv(data_path)

    TEXT_COL = "Review Text"
    RATING_COL = "Rating"

    df = df.dropna(subset=[TEXT_COL, RATING_COL]).copy()
    df["Rating_num"] = df[RATING_COL].astype(str).apply(extract_rating_num)
    df = df.dropna(subset=["Rating_num"]).copy()

    df["sentiment"] = df["Rating_num"].apply(lambda x: "positive" if x >= 4 else "negative")

    X = df[TEXT_COL].astype(str)
    y = df["sentiment"].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english")
    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec = tfidf.transform(X_test)

    model = LogisticRegression(max_iter=2000, class_weight="balanced")
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)

    print("\n=== Classification report ===\n")
    print(classification_report(y_test, y_pred))

    print("\n=== Confusion matrix ===\n")
    print(confusion_matrix(y_test, y_pred, labels=model.classes_))

    # Interpretation
    feature_names = np.array(tfidf.get_feature_names_out())
    coef = model.coef_[0]

    class_neg = model.classes_[0]
    class_pos = model.classes_[1]

    top_pos_idx = np.argsort(coef)[-15:]
    top_neg_idx = np.argsort(coef)[:15]

    print(f"\nTop words for '{class_pos}':\n", feature_names[top_pos_idx][::-1])
    print(f"\nTop words for '{class_neg}':\n", feature_names[top_neg_idx])


if __name__ == "__main__":
    main()
