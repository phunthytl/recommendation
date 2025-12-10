import pandas as pd
import sys, os
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

        self.cf = CollaborativeFiltering(n_factors=20)
        self.content = ContentBased(max_features=7000)

    def load_data(self):
        self.df_anime = pd.read_csv(self.anime_path)
        self.df_ratings = pd.read_csv(self.rating_path)

    def train(self):
        self.load_data()

        print("🚀 Training Collaborative Filtering...")
        self.cf.fit(self.df_ratings)

        print("🚀 Training Content-based TF-IDF...")
        self.content.fit(self.df_anime)

    def save(self):
        self.cf.save(self.model_dir)
        self.content.save(self.model_dir)

    def load(self):
        self.cf.load(self.model_dir)
        self.content.load(self.model_dir)

    # CF Recommend
    def recommend_cf(self, user_id, top_k=10):
        all_ids = self.df_anime["id"].tolist()
        pairs = self.cf.recommend(user_id, all_ids, self.df_ratings, top_k)
        return self._info(pairs)

    # Content Recommend
    def recommend_content(self, anime_id, top_k=10):
        pairs = self.content.similar_items(anime_id, top_k)
        return self._info(pairs)

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
    rec = RecommenderSystem()
    rec.train()
    rec.save()
    print("✅ All models saved.")
