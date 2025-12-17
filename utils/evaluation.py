import numpy as np
import pandas as pd
import random
from sklearn.metrics import mean_squared_error, mean_absolute_error
from collections import defaultdict


# Đánh giá RMSE và MAE cho mô hình CF đã huấn luyện sẵn

def evaluate_cf_rmse_mae(cf_model, df_ratings, test_ratio=0.2):
    # Trộn ngẫu nhiên dữ liệu rating để chia train/test
    df = df_ratings.sample(frac=1, random_state=42).reset_index(drop=True)

    # Xác định điểm cắt tập test theo test_ratio
    cut = int(len(df) * (1 - test_ratio))
    test_df = df.iloc[cut:]

    # Lưu trữ giá trị dự đoán và giá trị thực
    preds, trues = [], []

    # Duyệt từng rating trong tập test để dự đoán
    for _, row in test_df.iterrows():
        uid = int(row["user_id"])
        aid = int(row["anime_id"])
        true = float(row["rating"])

        # Dự đoán rating từ mô hình CF
        pred = cf_model.predict(uid, aid)
        if np.isnan(pred):
            continue

        preds.append(float(pred))
        trues.append(true)

    # Trường hợp không có dự đoán hợp lệ
    if len(preds) == 0:
        return 0.0, 0.0

    # Tính RMSE và MAE từ tập test
    mse = mean_squared_error(trues, preds)
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(trues, preds))
    return rmse, mae


# Tính Precision@K và Recall@K

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
    # Khởi tạo bộ sinh số ngẫu nhiên để đảm bảo tái lập kết quả
    rng = random.Random(seed)

    # Ép kiểu dữ liệu
    df = df_ratings.copy()
    df["user_id"] = df["user_id"].astype(int)
    df["anime_id"] = df["anime_id"].astype(int)
    df["rating"] = df["rating"].astype(float)

    # Lấy danh sách toàn bộ item
    all_items = df["anime_id"].unique().tolist()
    all_items_set = set(all_items)

    # Lưu trữ precision và recall của từng user
    precisions, recalls = [], []

    # Duyệt từng user trong tập dữ liệu
    for user_id in df["user_id"].unique():
        rows = df[df["user_id"] == user_id]

        # Bỏ qua user có quá ít rating
        if len(rows) < min_ratings:
            continue

        # Trộn và chia train/test cho từng user
        rows = rows.sample(frac=1, random_state=seed).reset_index(drop=True)
        cut = int(len(rows) * (1 - test_ratio))
        train_u = rows.iloc[:cut]
        test_u = rows.iloc[cut:]

        # Xác định các item tích cực trong tập test
        positives = set(test_u.loc[test_u["rating"] >= rating_threshold, "anime_id"].tolist())
        if len(positives) == 0:
            continue

        # Xác định các item user đã từng rating
        rated_by_user = set(rows["anime_id"].tolist())
        negatives_pool = list(all_items_set - rated_by_user)
        if len(negatives_pool) == 0:
            continue

        # Lấy mẫu negative items để đánh giá
        sampled_negs = rng.sample(negatives_pool, min(n_negatives, len(negatives_pool)))
        candidates = set(sampled_negs) | positives

        # Dự đoán điểm cho tập candidate
        scored = []
        for aid in candidates:
            pred = cf_model.predict(int(user_id), int(aid))
            if not np.isnan(pred):
                scored.append((int(aid), float(pred)))

        # Bỏ qua nếu không đủ item để lấy top-k
        if len(scored) < k:
            continue

        # Sắp xếp theo điểm dự đoán giảm dần
        scored.sort(key=lambda x: x[1], reverse=True)
        topk = {aid for aid, _ in scored[:k]}

        # Tính số hit trong top-k
        hit = len(topk & positives)
        precisions.append(hit / k)
        recalls.append(hit / len(positives))

    # Trường hợp không có user hợp lệ để đánh giá
    if not precisions:
        return 0.0, 0.0

    # Trả về precision và recall trung bình
    return float(np.mean(precisions)), float(np.mean(recalls))



# Tính coverage của mô hình CF dựa trên số item đã học

def coverage(cf_model, total_items):
    # Tránh chia cho 0 khi không có item
    if total_items == 0:
        return 0.0
    return len(getattr(cf_model, "item_id_to_idx", {})) / float(total_items)



# Đánh giá content-based filtering bằng Precision và Recall

def evaluate_content_based(rec_system, df_anime, k=10):
    # Bỏ qua nếu chưa xây dựng TF-IDF matrix
    if rec_system.content.tfidf_matrix is None:
        return 0.0, 0.0

    # Lưu trữ precision và recall cho từng anime
    precisions = []
    recalls = []

    # Lấy mẫu một tập anime để đánh giá
    sample_size = min(50, len(df_anime))
    sampled_animes = df_anime.sample(n=sample_size, random_state=42)

    # Duyệt từng anime trong tập mẫu
    for _, anime in sampled_animes.iterrows():
        anime_id = int(anime["id"])

        try:
            # Lấy danh sách anime gợi ý theo content-based
            recs = rec_system.recommend_content(anime_id, top_k=k)
            if not recs:
                continue

            rec_ids = {int(r["id"]) for r in recs}

            # Lấy genres của anime gốc để làm ground truth
            anime_genres = str(anime.get("genres", "")).lower()
            true_similar = set()

            # Xác định các anime có chung ít nhất một thể loại
            for _, other in df_anime.iterrows():
                oid = int(other["id"])
                if oid == anime_id:
                    continue
                other_genres = str(other.get("genres", "")).lower()
                if anime_genres and other_genres:
                    if any(g.strip() in other_genres for g in anime_genres.split(",")):
                        true_similar.add(oid)

            # Bỏ qua nếu không xác định được tập tương tự
            if not true_similar:
                continue

            # Tính precision và recall cho anime hiện tại
            tp = len(rec_ids & true_similar)
            precisions.append(tp / k if k > 0 else 0.0)
            recalls.append(tp / len(true_similar) if len(true_similar) > 0 else 0.0)

        # Bỏ qua anime gây lỗi trong quá trình gợi ý
        except Exception:
            continue

    # Trường hợp không có kết quả đánh giá hợp lệ
    if not precisions:
        return 0.0, 0.0

    # Trả về precision và recall trung bình
    return float(np.mean(precisions)), float(np.mean(recalls))
