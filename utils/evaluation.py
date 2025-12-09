import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# 1) RMSE & MAE for CF model-based (SVD)
# ============================================================
def evaluate_cf_rmse_mae(cf_model, df_ratings, test_ratio=0.2):
    df = df_ratings.copy()
    df = df.sample(frac=1, random_state=42)

    cut = int(len(df) * (1 - test_ratio))
    train_df = df.iloc[:cut]
    test_df  = df.iloc[cut:]

    # Auto adjust latent factors
    num_items = train_df["anime_id"].nunique()
    cf_model.n_factors = max(2, min(cf_model.n_factors, num_items))

    cf_model.fit(train_df)

    preds = []
    trues = []

    for _, row in test_df.iterrows():
        uid = row["user_id"]
        aid = row["anime_id"]

        true = float(row["rating"])
        pred = cf_model.predict(uid, aid)

        preds.append(pred)
        trues.append(true)

    mse  = mean_squared_error(trues, preds)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(trues, preds)

    return rmse, mae



# ============================================================
# 2) Precision@K & Recall@K for CF SVD
# ============================================================
def precision_recall_at_k(cf_model, df_ratings, k=10):
    df = df_ratings.copy()
    users = df["user_id"].unique()

    precisions = []
    recalls = []

    for user in users:
        user_rows = df[df["user_id"] == user]

        if len(user_rows) < 5:
            continue

        test = user_rows.sample(max(1, len(user_rows)//5), random_state=42)
        train = user_rows.drop(test.index)

        # Auto adjust latent factors theo số item trong train
        num_items = train["anime_id"].nunique()
        cf_model.n_factors = max(2, min(cf_model.n_factors, num_items))

        cf_model.fit(train)

        all_items = df["anime_id"].unique().tolist()
        recs = cf_model.recommend(user, all_items, train, top_k=k)
        rec_ids = {aid for aid, _ in recs}

        true_items = set(test["anime_id"])
        tp = len(rec_ids & true_items)

        precisions.append(tp / k)
        recalls.append(tp / len(true_items))

    if not precisions:
        return 0, 0

    return np.mean(precisions), np.mean(recalls)



# ============================================================
# 3) Coverage = số item có thể recommend
# ============================================================
def coverage(cf_model, total_items):
    return len(cf_model.item_id_to_idx) / total_items if total_items else 0



# ============================================================
# 4) Content-Based evaluation (genre match)
# ============================================================
def evaluate_content_based(cb_model, df_anime, k=10):
    precisions = []
    recalls = []

    sample = df_anime.sample(40, random_state=42)

    for _, row in sample.iterrows():
        aid = row["id"]
        genre = row["genres"]

        true_items = set(df_anime[df_anime["genres"] == genre]["id"])
        rec_pairs = cb_model.recommend_content(aid, top_k=k)
        rec_ids = set([p["id"] for p in rec_pairs])

        tp = len(true_items & rec_ids)

        precisions.append(tp / k)
        recalls.append(tp / max(len(true_items), 1))

    return np.mean(precisions), np.mean(recalls)
