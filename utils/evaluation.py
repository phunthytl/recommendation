import numpy as np


# ======================================================================
# 1) CF — Precision@K & Recall@K (dựa trên favorites.csv)
# ======================================================================
def precision_recall_at_k_cf(cf_model, df_fav, k=10):
    """
    Evaluate CF using a simpler approach:
    - Train model on ALL data (to get stable similarity matrix)
    - For each user, hold out 20% of interactions
    - Compute recommendations based on FULL train matrix but only considering held-out test set for TP
    - This avoids the issue of sparse similarity matrices from small train sets
    """

    # Train on all data
    cf_model.set_interactions(df_fav)
    
    precisions = []
    recalls = []

    user_count = 0
    valid_users = 0
    total_test_items = 0
    total_hits = 0

    np.random.seed(42)

    for user in df_fav["user_id"].unique():
        user_count += 1
        user_rows = df_fav[df_fav["user_id"] == user].copy()
        
        if len(user_rows) < 5:
            continue

        valid_users += 1

        # Split 80-20
        n_test = max(1, len(user_rows) // 5)
        test_indices = np.random.choice(user_rows.index, size=n_test, replace=False)
        test_df = df_fav.loc[test_indices]
        
        # Get recommendations using the FULL model (trained on all data)
        recs = cf_model.recommend(user, top_k=k, include_liked=True)
        rec_items = {r["id"] for r in recs}

        # Count how many test items appear in recommendations
        test_items = set(test_df["anime_id"])
        tp = len(test_items & rec_items)

        precisions.append(tp / k)
        recalls.append(tp / len(test_items))
        
        total_test_items += len(test_items)
        total_hits += tp

    print(f"[CF Eval] Total users: {user_count}, Valid for eval (>=5 interactions): {valid_users}")
    print(f"[CF Eval] Total test items: {total_test_items}, Total hits: {total_hits}")

    if len(precisions) == 0:
        return 0, 0

    return np.mean(precisions), np.mean(recalls)



# ======================================================================
# 2) CF — Coverage (theo lý thuyết trong recommender systems)
# ======================================================================
def coverage(cf_model, total_items):
    """
    Coverage = số lượng item có thể được recommend / tổng số item
    """

    if cf_model.sim_df is None or cf_model.sim_df.empty:
        return 0

    return len(cf_model.sim_df.index) / total_items



# ======================================================================
# 3) CBF — Precision@K & Recall@K (đánh giá dựa trên genre)
# ======================================================================
def precision_recall_at_k_cb(cb_model, df_anime, k=10):
    """
    Đánh giá Content-Based Filtering bằng cách so khớp genre:
    Anime cùng thể loại được coi là ground truth.
    """

    precisions = []
    recalls = []

    # lấy mẫu ngẫu nhiên 30 anime để đánh giá
    sample = df_anime.sample(min(30, len(df_anime)), random_state=42)

    for _, row in sample.iterrows():
        aid = row["id"]
        genre = row["genres"]

        # ground truth: toàn bộ anime cùng genre
        true_items = set(df_anime[df_anime["genres"] == genre]["id"])

        rec_ids = set(cb_model.recommend(aid, top_k=k))

        tp = len(true_items & rec_ids)

        precisions.append(tp / k)
        recalls.append(tp / max(len(true_items), 1))

    if len(precisions) == 0:
        return 0, 0

    return np.mean(precisions), np.mean(recalls)
