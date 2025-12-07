from models.content_based import ContentBasedRecommender

model = ContentBasedRecommender("data/anime_clean.csv")

results = model.recommend(anime_id=10, top_k=5)

for r in results:
    print(r["id"], r["title"], r["score"])
