import numpy as np
import joblib, os
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix

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

        self.global_mean = 0.0

    def fit(self, df_ratings):
        df = df_ratings.copy()

        self.global_mean = float(df["rating"].mean())
        print(f"[CF] Global Mean = {self.global_mean:.3f}")

        df["centered_rating"] = df["rating"].astype(float) - self.global_mean

        users = df["user_id"].unique()
        items = df["anime_id"].unique()

        self.user_id_to_idx = {uid: i for i, uid in enumerate(users)}
        self.idx_to_user_id = {i: uid for uid, i in self.user_id_to_idx.items()}

        self.item_id_to_idx = {aid: i for i, aid in enumerate(items)}
        self.idx_to_item_id = {i: aid for aid, i in self.item_id_to_idx.items()}

        rows = df["user_id"].map(self.user_id_to_idx).values
        cols = df["anime_id"].map(self.item_id_to_idx).values
        data = df["centered_rating"].astype(float).values

        matrix = csr_matrix((data, (rows, cols)),
                            shape=(len(users), len(items)))

        print("[CF] Training SVD...")
        self.svd = TruncatedSVD(n_components=self.n_factors, random_state=42)
        self.user_factors = self.svd.fit_transform(matrix)
        self.item_factors = self.svd.components_.T
        print("[CF] Done!")

    def predict(self, user_id, anime_id, clip=True):
        if user_id not in self.user_id_to_idx or anime_id not in self.item_id_to_idx:
            return np.nan

        u = self.user_factors[self.user_id_to_idx[user_id]]
        i = self.item_factors[self.item_id_to_idx[anime_id]]

        centered = float(np.dot(u, i))
        pred = centered + self.global_mean

        if clip:
            pred = float(np.clip(pred, 1, 10))

        return pred

    def recommend(self, user_id, all_anime_ids, df_ratings, top_k=10):
        if user_id not in self.user_id_to_idx:
            return []

        seen = set(df_ratings[df_ratings["user_id"] == user_id]["anime_id"].tolist())

        results = []
        for aid in all_anime_ids:
            if aid not in self.item_id_to_idx:
                continue

            if aid in seen:
                continue

            score = self.predict(user_id, aid)
            if not np.isnan(score):
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
            "global_mean": self.global_mean,
        }, os.path.join(path, "cf_model.joblib"))
        print("[CF] Saved model!")

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
        self.global_mean = obj["global_mean"]
        print("[CF] Loaded model!")
