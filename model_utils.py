import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def clean_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_dataset(path):
    df = pd.read_excel(path)

    # Normalize column names
    df.columns = [c.lower().strip() for c in df.columns]

    # Case 1: dataset uses label / text
    if "label" in df.columns and "text" in df.columns:
        df = df.rename(columns={
            "label": "disease",
            "text": "symptoms"
        })

    # Case 2: dataset already uses disease / symptoms
    elif "disease" in df.columns and "symptoms" in df.columns:
        pass

    # Case 3: exactly 2 columns but unknown names
    elif len(df.columns) == 2:
        df.columns = ["disease", "symptoms"]

    else:
        raise ValueError(
            "Dataset must have two columns: (label, text) or (disease, symptoms)"
        )

    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    df["cleaned_symptoms"] = df["symptoms"].apply(clean_text)
    return df


def build_tfidf(corpus):
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=5000
    )
    matrix = vectorizer.fit_transform(corpus)
    return vectorizer, matrix


def predict_disease(user_input, df, tfidf, tfidf_matrix, top_k=3):
    cleaned = clean_text(user_input)
    vec = tfidf.transform([cleaned])

    similarity = cosine_similarity(vec, tfidf_matrix).flatten()
    df_temp = df.copy()
    df_temp["score"] = similarity

    # Sort by similarity
    df_temp = df_temp.sort_values("score", ascending=False)

    # 🔥 REMOVE DUPLICATE DISEASE NAMES
    df_unique = df_temp.drop_duplicates(subset="disease")

    # Take top K UNIQUE diseases
    top = df_unique.head(top_k)

    return top[["disease"]].rename(columns={"disease": "Disease"})
