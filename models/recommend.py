import pandas as pd
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.collaborative import CollaborativeFiltering
from models.content_based import ContentBased


class RecommenderSystem:
    def __init__(self,
                 anime_path="data/anime_clean.csv",
                 rating_path="data/ratings.csv",
                 model_dir="models"):
        self.anime_path = anime_path
        self.rating_path = rating_path
        self.model_dir = model_dir

        self.df_anime = None
        self.df_ratings = None

        # Keep same field names, but CF is now MF-SGD engine
        self.cf = CollaborativeFiltering(n_factors=64, lr=0.01, reg=0.02, epochs=10, seed=42)
        self.content = ContentBased(max_features=7000)

        # For real-time: detect new appended rows in ratings.csv
        self._ratings_mtime = None
        self._ratings_len = 0

    def load_data(self):
        self.df_anime = pd.read_csv(self.anime_path)
        self.df_ratings = pd.read_csv(self.rating_path)

        # initialize trackers
        try:
            self._ratings_mtime = os.path.getmtime(self.rating_path)
        except Exception:
            self._ratings_mtime = None
        self._ratings_len = len(self.df_ratings)

    def _refresh_ratings_and_update_cf(self):
        """
        Real-time update trigger:
        - ratings.csv is appended immediately after a user rates
        - next time app calls recommend_cf(), we detect new rows and partial_update() only for new rows.

        Assumption: ratings.csv is appended (not rewritten/reordered).
        """
        try:
            mtime = os.path.getmtime(self.rating_path)
        except Exception:
            return

        if self._ratings_mtime is None:
            self._ratings_mtime = mtime
            return

        # No change
        if mtime == self._ratings_mtime:
            return

        # Changed -> reload csv
        new_df = pd.read_csv(self.rating_path)
        new_len = len(new_df)

        # If file was rewritten or truncated, safest fallback: reload full into memory (no online replay)
        # (You can also force retrain here, but that's heavy; we keep app stable.)
        if new_len < self._ratings_len:
            self.df_ratings = new_df
            self._ratings_len = new_len
            self._ratings_mtime = mtime
            return

        # Take only appended rows
        if new_len > self._ratings_len:
            appended = new_df.iloc[self._ratings_len:new_len][["user_id", "anime_id", "rating"]]
            # Apply online updates
            for _, r in appended.iterrows():
                self.cf.partial_update(int(r["user_id"]), int(r["anime_id"]), float(r["rating"]))
            # Update in-memory df
            self.df_ratings = new_df
            self._ratings_len = new_len

        self._ratings_mtime = mtime

    def train(self):
        self.load_data()

        print("🚀 Training Collaborative Filtering (MF-SGD)...")
        self.cf.fit(self.df_ratings)

        print("🚀 Training Content-based TF-IDF...")
        self.content.fit(self.df_anime)

    def save(self):
        self.cf.save(self.model_dir)
        self.content.save(self.model_dir)

    def load(self):
        # load models first
        self.cf.load(self.model_dir)
        self.content.load(self.model_dir)

        # ensure data is loaded for recommenders
        self.load_data()

    # CF Recommend (now real-time capable)
    def recommend_cf(self, user_id, top_k=10):
        # Real-time: if new ratings appended -> update CF before recommending
        self._refresh_ratings_and_update_cf()

        all_ids = self.df_anime["id"].astype(int).tolist()
        pairs = self.cf.recommend(int(user_id), all_ids, self.df_ratings, top_k)
        return self._info(pairs)

    # Content Recommend
    def recommend_content(self, anime_id, top_k=10):
        pairs = self.content.similar_items(int(anime_id), top_k)
        return self._info(pairs)

    def _info(self, pairs):
        results = []
        for aid, score in pairs:
            row = self.df_anime[self.df_anime["id"] == int(aid)]
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
    rec = RecommenderSystem()
    rec.train()
    rec.save()
    print("✅ All models saved.")
