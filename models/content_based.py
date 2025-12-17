import os
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Content-Based Filtering sử dụng TF-IDF và cosine similarity
class ContentBased:
    def __init__(self, max_features=7000):
        self.max_features = max_features # Số lượng feature tối đa cho TF-IDF
        self.vectorizer = None # Bộ vectorizer TF-IDF
        self.tfidf_matrix = None # Ma trận TF-IDF của toàn bộ anime
        self.anime_df = None # DataFrame lưu thông tin anime
        self.id_to_idx = {} # Mapping anime_id sang index
        self.idx_to_id = {} # Mapping index sang anime_id

    # Huấn luyện mô hình content-based từ dữ liệu anime
    def fit(self, df_anime):
        # Sao chép dữ liệu anime để tránh thay đổi dữ liệu gốc
        df = df_anime.copy()
        self.anime_df = df

        # Tạo mapping giữa anime_id và index trong ma trận TF-IDF
        self.id_to_idx = {aid: i for i, aid in enumerate(df["id"])}
        self.idx_to_id = {i: aid for aid, i in self.id_to_idx.items()}

        # Ghép title, genres và synopsis_clean thành một chuỗi văn bản
        combined = (
            df["title"].fillna("") + " " +
            df["genres"].fillna("") + " " +
            df.get("synopsis_clean", "").fillna("")
        )

        # Khởi tạo và huấn luyện TF-IDF vectorizer
        print("[CONTENT] Training TF-IDF...")
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=self.max_features
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(combined)
        print("[CONTENT] Done!")

    # Tìm các anime tương tự dựa trên cosine similarity
    def similar_items(self, anime_id, top_k=10):
        # Kiểm tra anime_id có tồn tại trong mô hình hay không
        if anime_id not in self.id_to_idx:
            return []

        # Lấy vector TF-IDF của anime truy vấn
        idx = self.id_to_idx[anime_id]
        vec = self.tfidf_matrix[idx]

        # Tính độ tương đồng cosine với toàn bộ anime
        scores = cosine_similarity(vec, self.tfidf_matrix)[0]
        ids = list(enumerate(scores))

        # Sắp xếp theo độ tương đồng giảm dần
        ids.sort(key=lambda x: x[1], reverse=True)

        # Lấy top_k anime tương tự, bỏ qua chính nó
        results = []
        for i, s in ids:
            if i == idx:
                continue
            results.append((self.idx_to_id[i], float(s)))
            if len(results) >= top_k:
                break

        return results

    # Lưu mô hình content-based ra file joblib
    def save(self, path):
        os.makedirs(path, exist_ok=True)
        joblib.dump({
            "vectorizer": self.vectorizer,
            "tfidf_matrix": self.tfidf_matrix,
            "anime_df": self.anime_df,
            "id_to_idx": self.id_to_idx,
            "idx_to_id": self.idx_to_id,
        }, os.path.join(path, "cbf_model.joblib"))
        print("[CONTENT] Saved model!")

    # Load mô hình content-based đã huấn luyện từ file
    def load(self, path):
        obj = joblib.load(os.path.join(path, "cbf_model.joblib"))
        self.vectorizer = obj["vectorizer"]
        self.tfidf_matrix = obj["tfidf_matrix"]
        self.anime_df = obj["anime_df"]
        self.id_to_idx = obj["id_to_idx"]
        self.idx_to_id = obj["idx_to_id"]
        print("[CONTENT] Loaded model!")
