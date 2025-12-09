import pandas as pd
import numpy as np

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

import joblib, os


#  MODEL-BASED COLLABORATIVE FILTERING (SVD – MATRIX FACTORIZATION)
class CollaborativeFiltering:
    def __init__(self, n_factors=50):
        self.n_factors = n_factors

        self.svd = None
        self.user_factors = None
        self.item_factors = None

        self.user_id_to_idx = {}
        self.idx_to_user_id = {}

        self.item_id_to_idx = {}
        self.idx_to_item_id = {}

    def fit(self, df_ratings):
        df = df_ratings.copy()

        # rating explicitly 1–10
        df["score"] = df["rating"].astype(float)

        # map user_id ↔ index
        users = df["user_id"].unique()
        items = df["anime_id"].unique()

        self.user_id_to_idx = {uid: i for i, uid in enumerate(users)}
        self.idx_to_user_id = {i: uid for uid, i in self.user_id_to_idx.items()}

        self.item_id_to_idx = {aid: i for i, aid in enumerate(items)}
        self.idx_to_item_id = {i: aid for aid, i in self.item_id_to_idx.items()}

        n_users = len(users)
        n_items = len(items)

        rows = df["user_id"].map(self.user_id_to_idx).values
        cols = df["anime_id"].map(self.item_id_to_idx).values
        data = df["score"].astype(float).values

        # build sparse matrix
        matrix = csr_matrix((data, (rows, cols)), shape=(n_users, n_items))

        # train SVD (matrix factorization)
        self.svd = TruncatedSVD(n_components=self.n_factors, random_state=42)
        self.user_factors = self.svd.fit_transform(matrix)     # U × K
        self.item_factors = self.svd.components_.T             # I × K

    def predict(self, user_id, anime_id):
        if user_id not in self.user_id_to_idx or anime_id not in self.item_id_to_idx:
            return 0.0

        u = self.user_factors[self.user_id_to_idx[user_id]]
        i = self.item_factors[self.item_id_to_idx[anime_id]]
        return float(np.dot(u, i))

    def recommend(self, user_id, all_anime_ids, df_ratings, top_k=10):
        if user_id not in self.user_id_to_idx:
            return []

        user_rows = df_ratings[df_ratings["user_id"] == user_id]
        if user_rows.empty:
            print(f"[CF] User {user_id} chưa rating anime nào → return []")
            return []
        
        seen = set(df_ratings[df_ratings["user_id"] == user_id]["anime_id"].tolist())

        results = []
        for aid in all_anime_ids:
            if aid not in self.item_id_to_idx:
                continue
            if aid in seen:
                continue

            score = self.predict(user_id, aid)
            results.append((aid, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        joblib.dump({
            "n_factors": self.n_factors,
            "svd": self.svd,
            "user_factors": self.user_factors,
            "item_factors": self.item_factors,
            "user_id_to_idx": self.user_id_to_idx,
            "idx_to_user_id": self.idx_to_user_id,
            "item_id_to_idx": self.item_id_to_idx,
            "idx_to_item_id": self.idx_to_item_id,
        }, os.path.join(path, "cf_model.joblib"))

    def load(self, path):
        obj = joblib.load(os.path.join(path, "cf_model.joblib"))
        self.n_factors = obj["n_factors"]
        self.svd = obj["svd"]
        self.user_factors = obj["user_factors"]
        self.item_factors = obj["item_factors"]
        self.user_id_to_idx = obj["user_id_to_idx"]
        self.idx_to_user_id = obj["idx_to_user_id"]
        self.item_id_to_idx = obj["item_id_to_idx"]
        self.idx_to_item_id = obj["idx_to_item_id"]


#  MODEL-BASED CONTENT-BASED (TF-IDF)
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

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=self.max_features
        )

        self.tfidf_matrix = self.vectorizer.fit_transform(combined)

    def similar_items(self, anime_id, top_k=10):
        if anime_id not in self.id_to_idx:
            return []

        idx = self.id_to_idx[anime_id]
        vec = self.tfidf_matrix[idx]

        scores = cosine_similarity(vec, self.tfidf_matrix)[0]
        pairs = list(enumerate(scores))

        pairs.sort(key=lambda x: x[1], reverse=True)

        results = []
        for i, s in pairs:
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
        }, os.path.join(path, "content_model.joblib"))

    def load(self, path):
        obj = joblib.load(os.path.join(path, "content_model.joblib"))
        self.vectorizer = obj["vectorizer"]
        self.tfidf_matrix = obj["tfidf_matrix"]
        self.anime_df = obj["anime_df"]
        self.id_to_idx = obj["id_to_idx"]
        self.idx_to_id = obj["idx_to_id"]


# ============================================================
#  RECOMMENDER SYSTEM (CF + CONTENT)
# ============================================================
class RecommenderSystem:
    def __init__(self, anime_path="data/anime_clean.csv",
                 rating_path="data/ratings.csv",
                 model_dir="models"):
        self.anime_path = anime_path
        self.rating_path = rating_path
        self.model_dir = model_dir

        self.df_anime = None
        self.df_ratings = None

        self.cf = CollaborativeFiltering(n_factors=20)
        self.content = ContentBased(max_features=7000)

    def load_data(self):
        self.df_anime = pd.read_csv(self.anime_path)
        self.df_ratings = pd.read_csv(self.rating_path)

    def train(self):
        self.load_data()

        print("Training CF (rating 1–10)...")
        self.cf.fit(self.df_ratings)

        print("Training Content TF-IDF...")
        self.content.fit(self.df_anime)

    def save(self):
        self.cf.save(self.model_dir)
        self.content.save(self.model_dir)

    def load(self):
        self.cf.load(self.model_dir)
        self.content.load(self.model_dir)

    # ---------- CF Recommend ----------
    def recommend_cf(self, user_id, top_k=10):
        all_ids = self.df_anime["id"].tolist()
        pairs = self.cf.recommend(user_id, all_ids, self.df_ratings, top_k)
        return self._info(pairs)

    # ---------- Content Recommend ----------
    def recommend_content(self, anime_id, top_k=10):
        pairs = self.content.similar_items(anime_id, top_k)
        return self._info(pairs)

    # helper
    def _info(self, pairs):
        results = []
        for aid, score in pairs:
            row = self.df_anime[self.df_anime["id"] == aid]
            if row.empty:
                continue
            r = row.iloc[0]
            results.append({
                "id": int(r["id"]),
                "title": r["title"],
                "image": r.get("image", ""),
                "genres": r.get("genres", ""),
                "score": float(score)
            })
        return results

if __name__ == "__main__":
    rec = RecommenderSystem(
        anime_path="data/anime_clean.csv",
        rating_path="data/ratings.csv",
        model_dir="models"
    )

    print("🚀 Training models...")
    rec.train()
    rec.save()
    print("Training complete. Models saved in /models/")