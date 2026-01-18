from pathlib import Path
import re
import tempfile
import urllib.request

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# ----------------------------
# Config
# ----------------------------
st.set_page_config(page_title="Sentiment Analysis", layout="centered")

DATA_URL = "https://github.com/MouadShl/customer-sentiment-analysis-nlp/releases/download/v1.0/Amazon_Reviews_clean.csv"
DATA_FILENAME = "Amazon_Reviews_clean.csv"


# ----------------------------
# Helpers
# ----------------------------
def get_dataset_path() -> Path:
    """
    Returns a path to the dataset.
    - Local dev: if data/Amazon_Reviews_clean.csv exists, use it
    - Online: download to temp folder once and reuse
    """
    # Try local repo data folder first (if you have it locally)
    local_path = Path(__file__).resolve().parents[1] / "data" / DATA_FILENAME
    if local_path.exists():
        return local_path

    # Otherwise use temp folder (Streamlit Cloud / Codespaces friendly)
    dest = Path(tempfile.gettempdir()) / DATA_FILENAME
    if not dest.exists():
        st.info("Téléchargement du dataset pour la démo en ligne...")
        urllib.request.urlretrieve(DATA_URL, dest)
        st.success("Dataset téléchargé ✅")
    return dest


def extract_rating_num(rating_text: str):
    """
    Extracts a numeric rating (1-5) from strings like 'Rated 4 out of 5 stars'
    """
    m = re.search(r"(\d)", str(rating_text))
    return float(m.group(1)) if m else np.nan


def load_dataframe(path: Path) -> pd.DataFrame:
    """
    Robust CSV loader for messy files / encoding issues.
    """
    try:
        return pd.read_csv(path, engine="python", on_bad_lines="skip", encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, engine="python", on_bad_lines="skip", encoding="latin-1")


# ----------------------------
# Train pipeline (cached)
# ----------------------------
@st.cache_resource
def train_pipeline():
    data_path = get_dataset_path()
    df = load_dataframe(data_path)

    TEXT_COL = "Review Text"
    RATING_COL = "Rating"

    # Basic cleaning
    df = df.dropna(subset=[TEXT_COL, RATING_COL]).copy()

    # Create numeric rating and sentiment
    df["Rating_num"] = df[RATING_COL].astype(str).apply(extract_rating_num)
    df = df.dropna(subset=["Rating_num"]).copy()
    df["sentiment"] = df["Rating_num"].apply(lambda x: "positive" if x >= 4 else "negative")

    X = df[TEXT_COL].astype(str)
    y = df["sentiment"].astype(str)

    # Train/test split
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # TF-IDF + Logistic Regression
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english")
    X_train_vec = tfidf.fit_transform(X_train)

    model = LogisticRegression(max_iter=2000, class_weight="balanced")
    model.fit(X_train_vec, y_train)

    return tfidf, model


# ----------------------------
# UI
# ----------------------------
tfidf, model = train_pipeline()

st.title("Customer Review Sentiment (NLP)")
text = st.text_area("Paste a review text", height=160)

if st.button("Predict"):
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        vec = tfidf.transform([text])
        pred = model.predict(vec)[0]
        st.success(f"Prediction: {pred.upper()}")
