import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class CollaborativeFiltering:
    def __init__(self, anime_path="data/anime_clean.csv", interactions_path="data/favorites.csv"):
        print("📦 Loading data...")

        self.df_anime = pd.read_csv(anime_path)
        self.interactions_path = interactions_path

        # load & build lần đầu
        self.reload()

    # ======================================================
    # Reload từ file favorites → build lại ma trận
    # ======================================================
    def reload(self):
        try:
            df = pd.read_csv(self.interactions_path)
        except:
            df = pd.DataFrame(columns=["user_id", "anime_id"])

        if df.empty:
            print("⚠ No interactions yet → CF will return empty results.")
            # delegate to set_interactions to keep logic consistent
            self.set_interactions(df)
            return

        # delegate to set_interactions which will add scores and build matrices
        self.set_interactions(df)

    def set_interactions(self, df):
        """
        Set interactions DataFrame directly (in-memory) and rebuild internal matrices.
        This is useful for evaluation where we want to use a train/test split
        without writing to disk.
        """
        if df is None or df.empty:
            self.df_inter = pd.DataFrame(columns=["user_id", "anime_id"])
            self.user_item = None
            self.sim_df = None
            return

        df = df.copy()
        # implicit feedback: mỗi tương tác = 1
        if "score" not in df.columns:
            df["score"] = 1

        self.df_inter = df
        self.build_user_item_matrix()
        self.compute_item_similarity()

    # ======================================================
    # 1. Build user-item matrix (0/1 implicit)
    # ======================================================
    def build_user_item_matrix(self):
        print("🧱 Building user-item matrix...")

        self.user_item = self.df_inter.pivot_table(
            index="user_id",
            columns="anime_id",
            values="score",
            fill_value=0
        )

    # ======================================================
    # 2. Item-item cosine similarity
    # ======================================================
    def compute_item_similarity(self):
        print("📐 Computing item similarity...")

        if self.user_item is None or self.user_item.empty:
            self.sim_df = None
            return

        item_vectors = self.user_item.T  # item × user
        sim_matrix = cosine_similarity(item_vectors)

        self.sim_df = pd.DataFrame(
            sim_matrix,
            index=item_vectors.index,
            columns=item_vectors.index
        )

    # 3. Recommend implicit CF (không có rating)
    def recommend(self, user_id, top_k=10, include_liked=False):
        if self.user_item is None or self.sim_df is None:
            return []

        if user_id not in self.user_item.index:
            return []

        user_vector = self.user_item.loc[user_id]
        liked = user_vector[user_vector > 0].index.tolist()

        if not liked:
            return []

        scores = pd.Series(0.0, index=self.sim_df.index)

        for aid in liked:
            if aid in self.sim_df.columns:
                scores += self.sim_df[aid]

        # Only drop liked items if include_liked is False
        if not include_liked:
            scores = scores.drop(labels=liked, errors="ignore")

        top_ids = scores.sort_values(ascending=False).head(top_k).index.tolist()

        results = []
        for aid in top_ids:
            row = self.df_anime[self.df_anime["id"] == aid]
            if not row.empty:
                r = row.iloc[0]
                results.append({
                    "id": int(r["id"]),
                    "title": r["title"],
                    "image": r["image"],
                    "genres": r["genres"],
                    "score": float(scores[aid])
                })

        return results
