import os
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ContentBased:
    def __init__(self, max_features=7000):
        self.max_features = max_features
        self.vectorizer = None
        self.tfidf_matrix = None
        self.anime_df = None

        self.id_to_idx = {}
        self.idx_to_id = {}

    def fit(self, df_anime):
        df = df_anime.copy()
        self.anime_df = df

        self.id_to_idx = {aid: i for i, aid in enumerate(df["id"])}
        self.idx_to_id = {i: aid for aid, i in self.id_to_idx.items()}

        combined = (
            df["title"].fillna("") + " " +
            df["genres"].fillna("") + " " +
            df.get("synopsis_clean", "").fillna("")
        )

        print("[CONTENT] Training TF-IDF...")
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=self.max_features
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(combined)
        print("[CONTENT] Done!")

    def similar_items(self, anime_id, top_k=10):
        if anime_id not in self.id_to_idx:
            return []

        idx = self.id_to_idx[anime_id]
        vec = self.tfidf_matrix[idx]

        scores = cosine_similarity(vec, self.tfidf_matrix)[0]
        ids = list(enumerate(scores))

        ids.sort(key=lambda x: x[1], reverse=True)

        results = []
        for i, s in ids:
            if i == idx:
                continue
            results.append((self.idx_to_id[i], float(s)))
            if len(results) >= top_k:
                break

        return results

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        joblib.dump({
            "vectorizer": self.vectorizer,
            "tfidf_matrix": self.tfidf_matrix,
            "anime_df": self.anime_df,
            "id_to_idx": self.id_to_idx,
            "idx_to_id": self.idx_to_id,
        }, os.path.join(path, "cbf_model.joblib"))
        print("[CONTENT] Saved model!")

    def load(self, path):
        obj = joblib.load(os.path.join(path, "cbf_model.joblib"))
        self.vectorizer = obj["vectorizer"]
        self.tfidf_matrix = obj["tfidf_matrix"]
        self.anime_df = obj["anime_df"]
        self.id_to_idx = obj["id_to_idx"]
        self.idx_to_id = obj["idx_to_id"]
        print("[CONTENT] Loaded model!")
