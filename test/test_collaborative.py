from models.collaborative import CollaborativeFiltering

model = CollaborativeFiltering(
    anime_path="data/anime_clean.csv",
    interactions_path="data/interactions.csv"
)

# Fake interactions để test demo
demo_data = [
    (1, 10), (1, 22), (1, 33),
    (2, 10), (2, 44),
    (3, 10), (3, 22),
    (4, 22), (4, 33),
]

for u, a in demo_data:
    model.add_interaction(user_id=u, anime_id=a)

print(">>> Recommend for user 1:")
results = model.recommend(1, top_k=5)

for r in results:
    print(r["id"], r["title"], r["score"])
