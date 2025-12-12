import numpy as np
import pandas as pd
import random
from sklearn.metrics import mean_squared_error, mean_absolute_error
from collections import defaultdict


# ============================================================
# 1) RMSE & MAE (pretrained model, no retrain)
# ============================================================
def evaluate_cf_rmse_mae(cf_model, df_ratings, test_ratio=0.2):
    df = df_ratings.sample(frac=1, random_state=42).reset_index(drop=True)

    cut = int(len(df) * (1 - test_ratio))
    test_df = df.iloc[cut:]

    preds, trues = [], []

    for _, row in test_df.iterrows():
        uid = int(row["user_id"])
        aid = int(row["anime_id"])
        true = float(row["rating"])

        pred = cf_model.predict(uid, aid)
        if np.isnan(pred):
            continue

        preds.append(float(pred))
        trues.append(true)

    if len(preds) == 0:
        return 0.0, 0.0

    mse = mean_squared_error(trues, preds)
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(trues, preds))
    return rmse, mae


# ============================================================
# 2) Precision@K & Recall@K (METHOD 2: negative sampling)
# ============================================================
def precision_recall_at_k(
    cf_model,
    df_ratings,
    k=10,
    test_ratio=0.2,
    rating_threshold=7,
    n_negatives=200,
    min_ratings=10,
    seed=42
):
    """
    Method 2:
    - For each user:
        - split user's ratings into train_u / test_u
        - positives = items in test_u with rating >= threshold
        - negatives = sample from global item set excluding items rated by this user (train_u + test_u)
        - rank candidates (positives + negatives) using cf_model.predict()
        - compute Precision@K, Recall@K
    """
    rng = random.Random(seed)
    df = df_ratings.copy()
    df["user_id"] = df["user_id"].astype(int)
    df["anime_id"] = df["anime_id"].astype(int)
    df["rating"] = df["rating"].astype(float)

    all_items = df["anime_id"].unique().tolist()
    all_items_set = set(all_items)

    precisions, recalls = [], []

    for user_id in df["user_id"].unique():
        rows = df[df["user_id"] == user_id]
        if len(rows) < min_ratings:
            continue

        rows = rows.sample(frac=1, random_state=seed).reset_index(drop=True)
        cut = int(len(rows) * (1 - test_ratio))
        train_u = rows.iloc[:cut]
        test_u = rows.iloc[cut:]

        positives = set(test_u.loc[test_u["rating"] >= rating_threshold, "anime_id"].tolist())
        if len(positives) == 0:
            continue

        rated_by_user = set(rows["anime_id"].tolist())  # train+test of this user
        negatives_pool = list(all_items_set - rated_by_user)
        if len(negatives_pool) == 0:
            continue

        sampled_negs = rng.sample(negatives_pool, min(n_negatives, len(negatives_pool)))
        candidates = set(sampled_negs) | positives

        # score candidates
        scored = []
        for aid in candidates:
            pred = cf_model.predict(int(user_id), int(aid))
            if not np.isnan(pred):
                scored.append((int(aid), float(pred)))

        if len(scored) < k:
            continue

        scored.sort(key=lambda x: x[1], reverse=True)
        topk = {aid for aid, _ in scored[:k]}

        hit = len(topk & positives)
        precisions.append(hit / k)
        recalls.append(hit / len(positives))

    if not precisions:
        return 0.0, 0.0

    return float(np.mean(precisions)), float(np.mean(recalls))


# ============================================================
# 3) Coverage
# ============================================================
def coverage(cf_model, total_items):
    if total_items == 0:
        return 0.0
    return len(getattr(cf_model, "item_id_to_idx", {})) / float(total_items)


# ============================================================
# 4) Content-based evaluation (unchanged)
# ============================================================
def evaluate_content_based(rec_system, df_anime, k=10):
    if rec_system.content.tfidf_matrix is None:
        return 0.0, 0.0

    precisions = []
    recalls = []

    sample_size = min(50, len(df_anime))
    sampled_animes = df_anime.sample(n=sample_size, random_state=42)

    for _, anime in sampled_animes.iterrows():
        anime_id = int(anime["id"])

        try:
            recs = rec_system.recommend_content(anime_id, top_k=k)
            if not recs:
                continue

            rec_ids = {int(r["id"]) for r in recs}

            anime_genres = str(anime.get("genres", "")).lower()
            true_similar = set()

            for _, other in df_anime.iterrows():
                oid = int(other["id"])
                if oid == anime_id:
                    continue
                other_genres = str(other.get("genres", "")).lower()
                if anime_genres and other_genres:
                    if any(g.strip() in other_genres for g in anime_genres.split(",")):
                        true_similar.add(oid)

            if not true_similar:
                continue

            tp = len(rec_ids & true_similar)
            precisions.append(tp / k if k > 0 else 0.0)
            recalls.append(tp / len(true_similar) if len(true_similar) > 0 else 0.0)

        except Exception:
            continue

    if not precisions:
        return 0.0, 0.0

    return float(np.mean(precisions)), float(np.mean(recalls))
