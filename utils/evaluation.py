import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error


# ============================================================
# 1) RMSE & MAE for Collaborative Filtering (SVD)
# ============================================================
def evaluate_cf_rmse_mae(cf_model, df_ratings, test_ratio=0.2):
    """Evaluate CF model WITHOUT retraining - use pretrained model"""
    df = df_ratings.sample(frac=1, random_state=42).reset_index(drop=True)

    cut = int(len(df) * (1 - test_ratio))
    train_df = df.iloc[:cut]
    test_df = df.iloc[cut:]

    # ❌ DON'T RETRAIN - model is already pretrained
    # cf_model.fit(train_df)  # REMOVED!

    preds, trues = [], []

    for _, row in test_df.iterrows():
        uid = row["user_id"]
        aid = row["anime_id"]

        true = float(row["rating"])
        pred = cf_model.predict(uid, aid)

        if np.isnan(pred):
            continue

        preds.append(pred)
        trues.append(true)

    if len(preds) == 0:
        return 0, 0

    mse = mean_squared_error(trues, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(trues, preds)

    return rmse, mae


# ============================================================
# 2) Precision@K & Recall@K (per-user relevance)
# ============================================================
def precision_recall_at_k(cf_model, df_ratings, k=10, test_ratio=0.2, min_ratings=5):
    df = df_ratings.copy()
    users = df["user_id"].unique()

    precisions, recalls = [], []

    for user in users:
        rows = df[df["user_id"] == user]

        # user quá ít rating → bỏ qua
        if len(rows) < min_ratings:
            continue

        rows = rows.sample(frac=1)
        cut = int(len(rows) * (1 - test_ratio))

        train_u = rows.iloc[:cut]
        test_u = rows.iloc[cut:]

        if len(test_u) == 0:
            continue

        # train CF trên dataset FULL (khuyến nghị)
        # cf_model.fit(train_df)  # ❌ không được train lại ở đây

        all_items = df["anime_id"].unique().tolist()
        recs = cf_model.recommend(user, all_items, train_u, top_k=k)
        rec_ids = {a for a, _ in recs}

        true_items = set(test_u["anime_id"])

        tp = len(rec_ids & true_items)

        precisions.append(tp / k)
        recalls.append(tp / len(true_items))

    if not precisions:
        return 0.0, 0.0

    return float(np.mean(precisions)), float(np.mean(recalls))


# ============================================================
# 3) Coverage (item coverage trong CF)
# ============================================================
def coverage(cf_model, total_items):
    if total_items == 0:
        return 0
    return len(cf_model.item_id_to_idx) / total_items


# ============================================================
# 4) Content-based evaluation (TF-IDF similarity)
# ============================================================
def evaluate_content_based(rec_system, df_anime, k=10):
    """
    Evaluate content-based recommendations using anime similarity.
    Returns precision and recall metrics.
    """
    if rec_system.content.tfidf_matrix is None:
        return 0.0, 0.0

    precisions = []
    recalls = []

    # Sample some anime items to evaluate
    sample_size = min(50, len(df_anime))
    sampled_animes = df_anime.sample(n=sample_size, random_state=42)

    for _, anime in sampled_animes.iterrows():
        anime_id = anime["id"]

        try:
            recs = rec_system.recommend_content(anime_id, top_k=k)
            if not recs:
                continue

            rec_ids = {r["id"] for r in recs}

            # Simple heuristic: check genre overlap for "true" similar items
            anime_genres = str(anime.get("genres", "")).lower()

            true_similar = set()
            for _, other in df_anime.iterrows():
                if other["id"] == anime_id:
                    continue
                other_genres = str(other.get("genres", "")).lower()
                # Consider similar if any genre overlap
                if anime_genres and other_genres:
                    if any(g.strip() in other_genres for g in anime_genres.split(",")):
                        true_similar.add(other["id"])

            if not true_similar:
                continue

            tp = len(rec_ids & true_similar)
            precisions.append(tp / k if k > 0 else 0)
            recalls.append(tp / len(true_similar) if len(true_similar) > 0 else 0)

        except Exception as e:
            continue

    if not precisions:
        return 0.0, 0.0

    return float(np.mean(precisions)), float(np.mean(recalls))
