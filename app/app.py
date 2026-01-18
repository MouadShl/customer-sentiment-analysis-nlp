from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import re
from pathlib import Path
import urllib.request
import streamlit as st
import urllib.request
import tempfile
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import re

DATA_URL = "https://github.com/MouadShl/customer-sentiment-analysis-nlp/releases/download/v1.0/Amazon_Reviews_clean.csv"

def get_dataset_path() -> Path:
    # Use temp folder (works on Streamlit Cloud / Codespaces)
    dest = Path(tempfile.gettempdir()) / "Amazon_Reviews_clean.csv"

    if not dest.exists():
        st.info("Downloading dataset for the online demo...")
        urllib.request.urlretrieve(DATA_URL, dest)
        st.success("Dataset downloaded ✅")

    return dest


DATA_URL = "https://github.com/MouadShl/customer-sentiment-analysis-nlp/releases/download/v1.0/Amazon_Reviews_clean.csv"

def get_dataset_path() -> Path:
    data_dir = Path(__file__).resolve().parents[1] / "data"
    data_dir.mkdir(exist_ok=True)
    local_path = data_dir / "Amazon_Reviews_clean.csv"

    # Local dev: use existing file if present
    if local_path.exists():
        return local_path

    # Online (Streamlit Cloud): download it once
    st.info("Téléchargement du dataset pour la démo en ligne...")
    urllib.request.urlretrieve(DATA_URL, local_path)
    st.success("Dataset téléchargé ✅")
    return local_path


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Sentiment Analysis", layout="centered")

def extract_rating_num(rating_text: str):
    m = re.search(r"(\d)", str(rating_text))
    return float(m.group(1)) if m else np.nan

@st.cache_resource
def train_pipeline():
    data_path = get_dataset_path()

try:
    df = pd.read_csv(data_path, engine="python", on_bad_lines="skip", encoding="utf-8")
except UnicodeDecodeError:
    df = pd.read_csv(data_path, engine="python", on_bad_lines="skip", encoding="latin-1")

    TEXT_COL = "Review Text"
    RATING_COL = "Rating"

    df = df.dropna(subset=[TEXT_COL, RATING_COL]).copy()
    df["Rating_num"] = df[RATING_COL].astype(str).apply(extract_rating_num)
    df = df.dropna(subset=["Rating_num"]).copy()
    df["sentiment"] = df["Rating_num"].apply(lambda x: "positive" if x >= 4 else "negative")

    X = df[TEXT_COL].astype(str)
    y = df["sentiment"].astype(str)

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english")
    X_train_vec = tfidf.fit_transform(X_train)

    model = LogisticRegression(max_iter=2000, class_weight="balanced")
    model.fit(X_train_vec, y_train)

    return tfidf, model

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
